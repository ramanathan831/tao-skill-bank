Model: mae

Supported actions tested:
- train: pass (`pretrain`, `finetune`, and finetune resume)
- eval: pass
- inference: pass
- export: pass
- deploy: pass (`gen_trt_engine`); deploy `evaluate` and deploy `inference` are unsupported by this deploy sub-skill
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- retrain: pass via resume training
- dataset convert: unsupported by this model skill
- other: AutoML policy noted as enabled in metadata, but not executed because workflow skills were explicitly out of scope

Dataset used:
- Source: `s3://nvcf-storage-handling/data/classification_train/`
- Source: `s3://nvcf-storage-handling/data/classification_val/`
- Source: `s3://nvcf-storage-handling/data/classification_test/`
- Notes: Used real S3 image-classification data. To honor the common `num_classes=6` setting, derived six-class train/val/test tarballs and extracted folders from the S3 20-class classification splits. Classes used: `sofa`, `tvmonitor`, `pottedplant`, `person`, `bird`, `motorbike`.
- Any dataset compatibility issues: Direct local Docker MAE training with the local `images_train_6class.tar.gz` path produced a zero-sample dataloader. Extracted `images_train`, `images_val`, and `images_test` folders worked for local Docker validation.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-checkpoint artifact was emitted for these 1-epoch/2-epoch runs
- Best checkpoint path: n/a
- Other checkpoints produced:
  - Pretrain: `/workspace/run/results/pretrain/train/model_epoch_000_step_00030.pth`
  - Pretrain latest symlink: `/workspace/run/results/pretrain/train/convnextv2_atto_latest.pth -> model_epoch_000_step_00030.pth`
  - Finetune: `/workspace/run/results/finetune/train/model_epoch_000_step_00030.pth`
  - Finetune latest symlink: `/workspace/run/results/finetune/train/convnextv2_atto_latest.pth -> model_epoch_000_step_00030.pth`
  - Resume: `/workspace/run/results/resume_train/train/model_epoch_001_step_00060.pth`
  - Resume latest symlink: `/workspace/run/results/resume_train/train/convnextv2_atto_latest.pth -> model_epoch_001_step_00060.pth`
- Pretrain KPI: `train_loss=8.211153030395508`
- Finetune KPIs: `train_loss=1.7963725328445435`, `val_loss=1.7946824789047242`, `ACC_all=0.2777777910232544`
- Resume KPIs: `train_loss=1.7471632957458496`, `val_loss=1.8451834678649903`, `ACC_all=0.2222222238779068`

Checkpoint/action verification:
- Finetune pretrained checkpoint used: `/workspace/run/results/pretrain/train/model_epoch_000_step_00030.pth`
- Eval checkpoint used: `/workspace/run/results/finetune/train/model_epoch_000_step_00030.pth`
- Inference checkpoint used: `/workspace/run/results/finetune/train/model_epoch_000_step_00030.pth`
- Export checkpoint used: `/workspace/run/results/finetune/train/model_epoch_000_step_00030.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/finetune/train/model_epoch_000_step_00030.pth`
- Deploy engine source: `/workspace/run/results/export/mae.onnx`
- Deploy engine used: `/workspace/run/results/deploy_gen/mae.engine`
- Were checkpoint paths selected through the proper resolver: yes. The model skill maps checkpoint fields through `parent_model` / `resume_model`; direct Docker validation pinned exact epoch-step artifacts rather than latest symlinks.
- Any incorrect latest-checkpoint behavior found: No. Latest symlinks existed, but dependent actions used exact epoch-step checkpoint paths.

Issues found:
- Model skill issues:
  - The skill documented tar archive inputs but did not distinguish SDK artifact upload behavior from direct local Docker behavior. Direct local MAE CLI runs need extracted image folders.
- Config issues:
  - Local Docker spec with `dataset.train_data_sources: /workspace/run/data/images_train_6class.tar.gz` failed with `ValueError: num_samples should be a positive integer value, but got num_samples=0`.
  - Local Docker spec with extracted folder paths passed.
- Dataset issues:
  - The available S3 classification data has 20 classes, so a six-class real-data subset was derived to match the requested `num_classes=6`.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - MAE finetune/evaluate/inference/export emits deprecation warnings advising future use of `classification_pyt` for downstream finetuning.
  - Deploy status file ends with a `RUNNING` record that says `Gen_trt_engine finished successfully` rather than a terminal `PASS` status.
  - Deploy command emitted telemetry warnings after successful completion: `Telemetry data couldn't be sent` and `'str' object has no attribute 'decode'`.
- Fresh-install issues:
  - Direct local Docker users need extracted MAE data folders; tar paths alone are not enough for the local dataloader.

Fixes made:
- Added MAE skill guidance explaining that SDK/app job inputs use `images_*.tar.gz`, while direct local Docker specs should point to extracted `images_train`, `images_val`, and `images_test` folders.

Remaining issues:
- Direct local Docker tar paths still produce a zero-sample dataloader; the validated local path is extracted folders.
- MAE finetune deprecation warnings remain.
- Deploy status files still do not record a terminal `PASS` status despite successful exit code and success message.
- Telemetry warnings remain after successful deploy actions.

Files changed:
- `models/mae/SKILL.md`
- `validation-reports/mae.md`

Final status:
- Fully validated
