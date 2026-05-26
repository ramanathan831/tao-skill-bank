Model: pose-classification

Supported actions tested:
- train: pass
- resume/retrain: pass via `pose_classification train` with `train.resume_training_checkpoint_path`
- eval: pass
- inference: pass
- export: pass
- default_specs: pass
- dataset convert: not run: preconverted dataset provided; converter requires raw DeepStream BodyPose JSON not present in the validation S3 data
- deploy: unsupported by this model skill
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- other: AutoML/HPO not run because this validation pass is restricted to model skill actions only

Dataset used:
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_pose_classification_train/nvidia/`
- Notes: Used `train_data.npy`, `train_label.pkl`, `val_data.npy`, `val_label.pkl`, `test_data.npy`, and `test_label.pkl`. Each split has 60 samples with shape `(60, 3, 300, 34, 1)` and labels in the range `0..5`.
- Any dataset compatibility issues: The S3 data is already converted to TAO-ready `.npy` / `.pkl` format, so it is not compatible with the raw BodyPose JSON `dataset_convert` input.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; Lightning produced epoch/step checkpoints
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/model_epoch_000_step_00003.pth`
  - `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`

Checkpoint/action verification:
- Resume checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00003.pth`
- Eval checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`
- Were checkpoint paths selected through the proper resolver: yes for the model skill wiring; the skill maps train resume to `resume_model` and evaluate/export/inference to `parent_model`. Direct local-docker validation used the exact produced checkpoint paths after artifact inspection, not a blind latest-file lookup.
- Any incorrect latest-checkpoint behavior found: none.

Issues found:
- Model skill issues:
  - The evaluate and inference templates contain both train/val dataset blocks and action-specific `test_dataset` blocks. A naive path replacement can populate the wrong fields and leave `evaluate.test_dataset.data_path` or `inference.test_dataset.data_path` blank.
- Config issues:
  - Inference requires an explicit writable `inference.output_file`; the template default is blank.
  - Dataset conversion is listed by the CLI but only applies to raw DeepStream BodyPose JSON, not the converted S3 validation files.
- Dataset issues:
  - No compatible raw JSON exists in the S3 validation folder for `dataset_convert`.
- Checkpoint issues:
  - None. Resume restored exactly from `model_epoch_000_step_00003.pth`, and downstream actions used `model_epoch_001_step_00006.pth`.
- Docker/local execution issues:
  - None blocking.
- Fresh-install issues:
  - Users/agents need to override action-specific dataset paths for evaluate/inference, not just the first dataset path fields in the YAML.

Fixes made:
- Added parent skill guidance for action-specific evaluate/inference dataset path overrides.

Remaining issues:
- AutoML/HPO was not executed in this model-only pass.
- `dataset_convert` remains unvalidated because compatible raw DeepStream BodyPose JSON is not present in `s3://nvcf-storage-handling/data/`.

Files changed:
- `models/pose-classification/SKILL.md`
- `validation-reports/pose-classification.md`

Final status:
- Fully validated for supported model actions available from the preconverted validation dataset; partially validated if counting AutoML/HPO or raw `dataset_convert`, which were outside this model-only/preconverted-data pass.
