Model: mask-grounding-dino

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass (`gen_trt_engine`, TensorRT `evaluate`, TensorRT `inference`)
- prune: unsupported by this model skill
- quantize: pass
- retrain: pass via resume training
- dataset convert: unsupported by this model skill
- other: AutoML policy noted as enabled in metadata, but not executed because workflow skills were explicitly out of scope

Dataset used:
- Source: `s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_train/`
- Source: `s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_val/`
- Notes: Used real S3 Mask Grounding DINO segmentation data. Training used 49 ODVG training samples with an ODVG label map; validation/evaluation/inference used the COCO-style validation split with 48 images. Deploy evaluation expanded the validation split to 377 image-caption pairs.
- Any dataset compatibility issues: None. Validation category IDs were compatible with `eval_class_ids: [0, 1, 2]`, and OD inference captions were `person`, `bicycle`, and `car`.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-checkpoint artifact was emitted for this 1-epoch run
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/train/model_epoch_000_step_00049.pth`
  - `/workspace/run/results/train/train/mask_gdino_model_latest.pth -> model_epoch_000_step_00049.pth`
  - `/workspace/run/results/resume_train/train/model_epoch_001_step_00098.pth`
  - `/workspace/run/results/resume_train/train/mask_gdino_model_latest.pth -> model_epoch_001_step_00098.pth`
- Train KPIs: `val_loss=53.01388168334961`, `train_loss=65466.5390625`
- Resume KPIs: `val_loss=52.398773193359375`, `train_loss=22547.775390625`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00049.pth`
- Inference checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00049.pth`
- Export checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00049.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00049.pth`
- Quantize checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00049.pth`
- Deploy engine source: `/workspace/run/results/export/model.onnx`
- Deploy engine used: `/workspace/run/results/deploy_gen_trt_engine/mask-grounding-dino.engine`
- Were checkpoint paths selected through the proper resolver: yes. The model skill maps checkpoint fields through `parent_model` / `resume_model`; direct Docker validation pinned the exact resolved epoch-step checkpoint rather than a latest symlink.
- Any incorrect latest-checkpoint behavior found: No. Latest symlinks existed, but eval, inference, export, quantize, and resume used exact epoch-step checkpoint paths.

Issues found:
- Model skill issues:
  - Deploy carry-forward guidance did not explicitly list `num_select`, `max_text_len`, `num_region_queries`, and `has_mask`, all of which are part of the validated Mask Grounding DINO TensorRT shape contract for deploy evaluate/inference.
- Config issues:
  - None.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - Deploy status files end with `RUNNING` records that say the action finished successfully rather than a terminal `PASS` status.
  - Deploy commands emitted telemetry warnings after successful completion: `Telemetry data couldn't be sent` and `'str' object has no attribute 'decode'`.
  - Deploy evaluation/inference downloaded the BERT tokenizer anonymously because no HF token was passed into the container, intentionally avoiding secret exposure in generated artifacts.
- Fresh-install issues:
  - No model-specific pretrained checkpoint was required for this validation; the run trained from scratch with the default backbone/tokenizer downloads.

Fixes made:
- Updated the Mask Grounding DINO deploy skill guidance to carry transformer and mask shape fields forward from export into deploy evaluate/inference specs.
- Updated deploy metadata notes so generated user guidance includes `num_queries`, `num_select`, `max_text_len`, `num_region_queries`, and `has_mask`.
- Added a parent skill error-pattern note for deploy model shape mismatches.

Remaining issues:
- Deploy status files still do not record a terminal `PASS` status despite successful exit codes and success messages.
- Telemetry warnings remain after successful deploy actions.

Files changed:
- `models/mask-grounding-dino/SKILL.md`
- `models/mask-grounding-dino/deploy/SKILL.md`
- `models/mask-grounding-dino/deploy/skill_info.yaml`
- `validation-reports/mask-grounding-dino.md`

Final status:
- Fully validated
