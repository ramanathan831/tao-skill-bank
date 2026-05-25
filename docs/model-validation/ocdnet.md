Model: ocdnet

Supported actions tested:
- train: pass
- evaluate: pass
- inference: pass
- export: pass
- deploy/gen_trt_engine: pass
- deploy/inference on tao-deployed gen_trt_engine model: pass
- deploy/evaluate: pass
- prune: pass
- quantize: fail
- resume training through train.resume_training_checkpoint_path: pass
- retrain: unsupported as a standalone PyT action; removed from parent model metadata
- dataset convert: unsupported
- other: stale parent gen_trt_engine action confirmed unsupported by PyT CLI

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_train/train.tar.gz
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_train/train/img.tar.gz
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_val/test.tar.gz
- Notes: Extracted ICDAR-style folders under train/ and test/. Train split has 45 images and 45 gt files; validation/test split has 25 images and 25 gt files; calibration image archive has 25 images.
- Any dataset compatibility issues: none after using extracted folders/lists for dataset paths and a flat image folder for inference.

Training result:
- Training completed: yes
- Best checkpoint produced: yes
- Best checkpoint path: /workspace/results/train/model_best.pth
- Other checkpoints produced:
  - /workspace/results/train/model_epoch_000_step_00023.pth
  - /workspace/results/train/model_epoch_001_step_00046.pth

Checkpoint/action verification:
- Eval checkpoint used: /workspace/results/train/model_best.pth
- Inference checkpoint used: /workspace/results/train/model_best.pth
- Export checkpoint used: /workspace/results/train/model_best.pth
- Prune checkpoint used: /workspace/results/train/model_best.pth
- Quantize checkpoint/artifacts used:
  - /workspace/results/train/model_best.pth failed because it lacks pytorch-lightning_version.
  - /workspace/results/train/model_epoch_001_step_00046.pth reached the full-checkpoint load path but failed in the SDK with missing OCDnetModel dm/task constructor args.
  - /workspace/results/export/ocdnet.onnx reached ONNX calibration but failed because modelopt.onnx.quantization is unavailable in the default PyT image.
- Resume/retrain checkpoint used: /workspace/results/train/model_epoch_001_step_00046.pth
- Were checkpoint paths selected through the proper resolver: yes for validated actions after metadata fix; quantize now documents the full-checkpoint requirement instead of using model_best.pth blindly.
- Any incorrect latest-checkpoint behavior found: yes, prior metadata lacked parent checkpoint mappings and would allow fragile generic selection; fixed to explicit parent_model/resume_model mappings.

Issues found:
- Model skill issues:
  - Parent PyT metadata advertised unsupported retrain and gen_trt_engine actions.
  - Parent metadata lacked checkpoint and output handoff mappings for evaluate, export, inference, prune, quantize, and train resume.
  - Parent instructions mixed deploy TensorRT handoffs into the PyT skill.
- Config issues:
  - Deploy gen_trt_engine template lacked results_dir/gen_trt_engine.results_dir and failed immediately.
  - One-epoch smoke training with warmup_epoch=1 fails because num_epochs must not equal warmup_epoch.
- Dataset issues:
  - none.
- Checkpoint issues:
  - model_best.pth is valid for evaluate/inference/export/prune but is not a full Lightning checkpoint for quantize.
  - Quantize with a full epoch checkpoint still fails inside the current SDK/default image.
- Docker/local execution issues:
  - PyT quantize ONNX path fails because modelopt.onnx.quantization is missing from the default image.
  - Deploy telemetry reports a non-fatal decode warning after successful deploy commands.
- Fresh-install issues:
  - Default-image OCDNet quantize remains blocked even with correct artifacts.

Fixes made:
- Removed unsupported parent retrain and parent gen_trt_engine from OCDNet metadata/manifests.
- Added parent checkpoint and output mappings in references/skill_info.yaml.
- Clarified checkpoint handoff behavior in SKILL.md, including best-weight vs full Lightning checkpoint behavior.
- Added deploy gen_trt_engine results_dir defaults to the deploy template and deploy skill metadata.
- Updated deploy instructions to keep results_dir fields aligned and to avoid references to unsupported parent distillation/retrain actions.

Remaining issues:
- OCDNet quantize remains unresolved in the current default PyT image. The CLI action is supported, but both PyTorch-checkpoint and ONNX quantization paths fail in toolkit/default-image code after correct artifact selection.

Files changed:
- models/ocdnet/SKILL.md
- models/ocdnet/references/skill_info.yaml
- models/ocdnet/references/spec_template_deploy_gen_trt_engine.yaml
- models/ocdnet/deploy/SKILL.md
- models/ocdnet/deploy/skill_info.yaml
- models/ocdnet/schemas/manifest.json
- models/schemas.manifest.json
- docs/model-validation/ocdnet.md

Final status:
- Partially validated: all train/evaluate/inference/export/prune/resume and deploy actions pass; quantize is blocked by current toolkit/default-image defects.
