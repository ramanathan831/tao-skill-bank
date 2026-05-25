Model: grounding-dino

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- deploy inference on tao-deployed gen_trt_engine model: pass
- deploy eval on tao-deployed gen_trt_engine model: pass
- quantize: fail
- resume/retrain: pass through train.resume_training_checkpoint_path
- dataset convert: unsupported/not advertised by the Grounding-DINO model skill
- prune: unsupported/not advertised by the Grounding-DINO model skill
- parent gen_trt_engine: fail; not supported by the PyTorch CLI and removed from parent model metadata

Dataset used:
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_odvg/
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_odvg/
- Notes: Train split used ODVG annotations and label map. Validation, evaluation, calibration, and deploy evaluation used COCO annotations from the validation split.
- Any dataset compatibility issues: Quantize calibration must use COCO JSON, not ODVG JSONL. Using ODVG JSONL for calibration fails JSON parsing in the calibration dataloader.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; the one-epoch run produced an epoch/step checkpoint.
- Best checkpoint path: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Other checkpoints produced: /tmp/tao-model-validation/grounding-dino/results/resume/train/model_epoch_001_step_00092.pth; gdino_model_latest.pth symlinks were produced but not used for checkpoint-dependent actions.

Checkpoint/action verification:
- Eval checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Inference checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Export checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Resume/retrain checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Deploy checkpoint path: deploy used /tmp/tao-model-validation/grounding-dino/results/export/model.onnx exported from the exact checkpoint above.
- Quantize checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth, but the action failed in the container before quantized artifact creation.
- Were checkpoint paths selected through the proper resolver: yes in model metadata after the fix; direct local-docker validation used explicit exact paths to verify the resolver contract.
- Any incorrect latest-checkpoint behavior found: no runtime action selected gdino_model_latest.pth; the parent metadata was missing resolver mappings before the fix.

Issues found:
- Model skill issues:
  - Parent skill advertised gen_trt_engine even though the PyTorch grounding_dino CLI only supports evaluate, export, inference, quantize, train, and default_specs.
  - Inference metadata advertised dataset.infer_data_sources.classmap, but Grounding-DINO inference requires dataset.infer_data_sources.captions.
  - Export and quantize inputs were empty in skill_info.yaml, preventing reliable parent artifact handoff.
  - spec_params were empty, so parent checkpoint and ONNX output resolver mappings were not declared.
- Config issues:
  - Quantize calibration documentation pointed at ODVG JSONL; the quantize calibration dataloader expects COCO JSON.
  - Deploy specs must carry forward structural settings and export input resolution from the checkpoint/export spec.
- Dataset issues:
  - No dataset gap for train/eval/inference/export/deploy; the S3 ODVG train and COCO val subsets are compatible.
- Checkpoint issues:
  - No incorrect latest behavior found in validated actions.
  - Quantize with a PyTorch checkpoint is blocked by an SDK/container bug where cap_lists is passed as None during checkpoint load.
- Docker/local execution issues:
  - The default PyTorch image requires access to bert-base-uncased for tokenizer/model loading; providing a Hugging Face token avoids rate-limit failures.
  - modelopt.onnx is listed as an available backend, but the default PyTorch image lacks modelopt.onnx.quantization, so ONNX quantize fails after calibration setup.
- Fresh-install issues:
  - Fresh local-docker users need network or cached Hugging Face assets for the default bert-base-uncased text encoder.

Fixes made:
- Removed parent gen_trt_engine from Grounding-DINO model manifests; TensorRT actions now route through the deploy subskill.
- Added export, inference, and quantize inputs/outputs to skill_info.yaml.
- Added parent checkpoint and ONNX output resolver mappings to skill_info.yaml.
- Corrected inference instructions from classmap to captions.
- Documented COCO calibration requirements for quantize.
- Documented exact checkpoint selection and shape-carry-forward requirements.
- Documented the remaining quantize SDK/image blockers.

Remaining issues:
- Quantize remains unresolved in the default rc-226 PyTorch image:
  - PyTorch checkpoint quantize fails because the container script loads Grounding-DINO with cap_lists=None.
  - ONNX quantize reaches calibration when using COCO JSON, then fails because modelopt.onnx.quantization is unavailable in the image.

Files changed:
- models/grounding-dino/SKILL.md
- models/grounding-dino/references/skill_info.yaml
- models/grounding-dino/schemas/manifest.json
- models/schemas.manifest.json
- docs/model-validation/grounding-dino.md

Final status:
- Partially validated. Train, eval, inference, export, resume, deploy gen_trt_engine, deploy inference, and deploy evaluation pass end-to-end on local-docker with image=default. Quantize is blocked by SDK/container issues outside the model skill metadata.
