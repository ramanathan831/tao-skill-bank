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
- quantize: pass in the rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`
- resume training through train.resume_training_checkpoint_path: pass
- AutoML default train route: pass with Bayesian automl_max_recommendations=2
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
- Source-fixed rerun checkpoints:
  - /tmp/tao-source-fixed-rerun/current/ocdnet/results/train/model_best.pth
  - /tmp/tao-source-fixed-rerun/current/ocdnet/results/train/model_epoch_000_step_00006.pth

AutoML default training rerun:
- Default model training was rerun through the model skill's AutoML-enabled train route after confirming the previous direct validation had exercised normal training only.
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_train/train.tar.gz
- Eval source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_val/test.tar.gz
- Algorithm: bayesian
- Recommendations requested: 2
- Metric: loss_epoch, minimize; logs emitted train_loss_epoch and status.json recorded train_loss.
- Tuned parameters: train.optimizer.args.lr, train.optimizer.args.weight_decay
- Recommendation 0: job 9e0fdb3f-5a67-4c4b-ba5d-ded4b38e1f6d, lr 0.0011327421097186941, weight_decay 9.84430986920081e-05, loss_epoch 35.31, checkpoint /tmp/tao-automl-validation/ocdnet/results/9e0fdb3f-5a67-4c4b-ba5d-ded4b38e1f6d/results_dir/train/model_epoch_000_step_00006.pth, best-weight file /tmp/tao-automl-validation/ocdnet/results/9e0fdb3f-5a67-4c4b-ba5d-ded4b38e1f6d/results_dir/train/model_best.pth
- Recommendation 1: job a934d058-9c8c-46e8-9378-14380e77210f, lr 0.0005825796915910182, weight_decay 5.197158933969869e-05, loss_epoch 33.194, checkpoint /tmp/tao-automl-validation/ocdnet/results/a934d058-9c8c-46e8-9378-14380e77210f/results_dir/train/model_epoch_000_step_00006.pth, best-weight file /tmp/tao-automl-validation/ocdnet/results/a934d058-9c8c-46e8-9378-14380e77210f/results_dir/train/model_best.pth
- Best recommendation: rec 1, selected by the AutoML controller summary.
- Generated spec verification: both recommendations used SDK-extracted real S3 train/test archives, train batch size 8, train.num_epochs 1, train.lr_scheduler.args.warmup_epoch 0, and distinct Bayesian lr/weight_decay values within the requested ranges.

Checkpoint/action verification:
- Eval checkpoint used: /workspace/results/train/model_best.pth
- Inference checkpoint used: /workspace/results/train/model_best.pth
- Export checkpoint used: /workspace/results/train/model_best.pth
- Prune checkpoint used: /workspace/results/train/model_best.pth
- Quantize checkpoint/artifacts used:
  - /workspace/results/train/model_best.pth failed because it lacks pytorch-lightning_version.
  - /workspace/results/train/model_epoch_001_step_00046.pth reached the full-checkpoint load path but failed in the SDK with missing OCDnetModel dm/task constructor args.
  - /workspace/results/export/ocdnet.onnx reached ONNX calibration but failed because modelopt.onnx.quantization is unavailable in the default PyT image.
- Source-fixed quantize checkpoint used: /workspace/results/train/model_epoch_000_step_00006.pth, selected through `tao_sdk.checkpoints.get_checkpoint_path(..., epoch=0, step=6, allow_latest=False)`. The rerun intentionally did not use `model_best.pth`.
- Resume/retrain checkpoint used: /workspace/results/train/model_epoch_001_step_00046.pth
- Were checkpoint paths selected through the proper resolver: yes for validated actions after metadata fix; quantize now documents the full-checkpoint requirement instead of using model_best.pth blindly.
- Any incorrect latest-checkpoint behavior found: yes, prior metadata lacked parent checkpoint mappings and would allow fragile generic selection; fixed to explicit parent_model/resume_model mappings.

Issues found:
- Model skill issues:
  - Parent PyT metadata advertised unsupported retrain and gen_trt_engine actions.
  - Parent metadata lacked checkpoint and output handoff mappings for evaluate, export, inference, prune, quantize, and train resume.
  - Parent instructions mixed deploy TensorRT handoffs into the PyT skill.
  - AutoML metric guidance was missing for fresh one-epoch Bayesian smoke runs.
- Config issues:
  - Deploy gen_trt_engine template lacked results_dir/gen_trt_engine.results_dir and failed immediately.
  - One-epoch smoke training with warmup_epoch=1 fails because num_epochs must not equal warmup_epoch.
- Dataset issues:
  - none.
- Checkpoint issues:
  - model_best.pth is valid for evaluate/inference/export/prune but is not a full Lightning checkpoint for quantize.
  - Quantize with a full epoch checkpoint failed inside the original default SDK/image but passed in the rebuilt PyT image.
- Docker/local execution issues:
  - PyT quantize ONNX path fails because modelopt.onnx.quantization is missing from the default image.
  - Deploy telemetry reports a non-fatal decode warning after successful deploy commands.
- Fresh-install issues:
  - Default-image OCDNet quantize remains blocked even with correct artifacts.

Fixes made:
- Removed unsupported parent retrain and parent gen_trt_engine from OCDNet metadata/manifests.
- Added parent checkpoint and output mappings in references/skill_info.yaml.
- Clarified checkpoint handoff behavior in SKILL.md, including best-weight vs full Lightning checkpoint behavior.
- Added AutoML metric guidance to SKILL.md: use train_loss_epoch/train_loss with minimize direction and set warmup_epoch to 0 for one-epoch local smoke runs.
- Added deploy gen_trt_engine results_dir defaults to the deploy template and deploy skill metadata.
- Updated deploy instructions to keep results_dir fields aligned and to avoid references to unsupported parent distillation/retrain actions.
- Reran source-fixed quantize on `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525` with the full resolver-selected epoch checkpoint; the action completed successfully and wrote `/workspace/results/quantize/quantized_model_torchao.pth`.

Remaining issues:
- The original default PyT image still has the quantize blockers listed above; the rebuilt source-fixed PyT image passes the packaged PyTorch-checkpoint quantize path.
- ONNX quantize was not rerun because the packaged source-fixed action path validated here uses PyTorch checkpoint quantization with `torchao`.

Files changed:
- models/ocdnet/SKILL.md
- models/ocdnet/references/skill_info.yaml
- models/ocdnet/references/spec_template_deploy_gen_trt_engine.yaml
- models/ocdnet/deploy/SKILL.md
- models/ocdnet/deploy/skill_info.yaml
- models/ocdnet/schemas/manifest.json
- models/schemas.manifest.json
- docs/model-validation/ocdnet.md
- docs/model-validation/action-run-inventory.md

Final status:
- Fully validated for packaged model-skill actions after the source-fixed image rerun: train/evaluate/inference/export/prune/resume, AutoML default train, deploy actions, and quantize all pass. Quantize requires the rebuilt PyT image and a full epoch checkpoint, not `model_best.pth`.
