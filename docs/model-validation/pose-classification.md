# Model: pose-classification

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the Pose Classification model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- train: pass, AutoML default route with two Bayesian recommendations
- eval/evaluate: pass after adding downstream `dataset.label_map`
- inference: pass after adding downstream `dataset.label_map`
- export: pass after adding downstream `dataset.label_map`
- resume training: pass through `pose_classification train -e` with `train.resume_training_checkpoint_path`
- dataset convert: blocked; the PyT CLI supports it and model-skill metadata/template were added, but no compatible raw DeepStream BodyPose JSON exists in the S3 source
- deploy: unsupported by the packaged Pose Classification model skill
- prune: unsupported by the packaged Pose Classification model skill
- quantize: unsupported by the packaged Pose Classification model skill
- retrain: unsupported as a standalone action; resume uses `train`
- other: no deploy sub-skill is packaged for this model

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_pose_classification_train/nvidia/`
- Notes: used `train_data.npy`, `train_label.pkl`, `val_data.npy`, `val_label.pkl`, `test_data.npy`, and `test_label.pkl` with `dataset.num_classes=6`, the six-class synthetic `dataset.label_map`, and `model.graph_layout=nvidia`.
- Any dataset compatibility issues: train/evaluate/inference/export/resume were compatible. `dataset_convert` needs raw DeepStream BodyPose JSON, but this S3 folder is already converted to NumPy/Pickle files.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/pose-classification/results/1f3f2c89-edda-4110-83ca-f8909f611e2e/results_dir/train/model_epoch_000_step_00003.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/pose-classification/results/e0e2c5cf-3525-43a7-ad34-f611248fdad8/results_dir/train/model_epoch_000_step_00003.pth`
  - resume produced `/tmp/tao-automl-validation/pose-classification/1e70f650-43d4-4a1f-b8f8-4a4336d251c6/results_dir/train/model_epoch_001_step_00006.pth`
- AutoML recommendations:
  - rec 0: job `e0e2c5cf-3525-43a7-ad34-f611248fdad8`, `model.dropout=0.4240251959568916`, `train.optim.lr=0.08886715384174658`, `val_loss=1.815`
  - rec 1: job `1f3f2c89-edda-4110-83ca-f8909f611e2e`, `model.dropout=0.4239359647327556`, `train.optim.lr=0.058533818887061044`, `val_loss=1.794`, selected as best

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 1 `model_epoch_000_step_00003.pth`; evaluation passed with total accuracy `16.6667`.
- Inference checkpoint used: best AutoML rec 1 `model_epoch_000_step_00003.pth`; inference wrote `/tmp/tao-automl-validation/pose-classification/manual_outputs/pose_inference.txt`.
- Export checkpoint used: best AutoML rec 1 `model_epoch_000_step_00003.pth`; export wrote `/tmp/tao-automl-validation/pose-classification/manual_outputs/pose_classification.onnx`.
- Resume/retrain checkpoint used: best AutoML rec 1 `model_epoch_000_step_00003.pth`; resume log restored that exact checkpoint and produced epoch 1 step 6.
- Were checkpoint paths selected through the proper resolver: yes, validation used the checkpoint resolver and selected `model_epoch_000_step_00003.pth` from the best AutoML child job instead of `pc_model_latest.pth`.
- Any incorrect latest-checkpoint behavior found: yes in metadata before fixes. The model metadata did not expose checkpoint inputs/mappings for train resume, evaluate, inference, or export, which would force fragile manual/latest checkpoint selection.

## Issues found

- Model skill issues:
  - `dataset_convert` existed in the real PyT CLI but was missing from model metadata and had no packaged spec template.
  - Train resume, evaluate, inference, and export metadata lacked checkpoint inputs and resolver mappings.
  - Export and inference need explicit output file spec fields, but those file paths must not be declared as pre-created file outputs for the current local runner.
- Config issues:
  - Evaluate, inference, and export templates omitted `dataset.label_map`; checkpoint load failed with `AttributeError: 'NoneType' object has no attribute 'keys'`.
- Dataset issues:
  - No compatible raw DeepStream BodyPose JSON was found for `dataset_convert`.
- Checkpoint issues:
  - The resolver correctly selected `model_epoch_000_step_00003.pth`; no `pc_model_latest.pth` fallback was used.
- Docker/local execution issues:
  - None after the template/metadata fixes.
- Fresh-install issues:
  - A fresh install would fail downstream checkpoint-consuming actions because of missing metadata and missing `dataset.label_map`.

## Fixes made

- Added `dataset_convert` metadata and `references/spec_template_dataset_convert.yaml`.
- Added train resume, evaluate, export, and inference checkpoint inputs plus `spec_params` resolver mappings in `models/pose-classification/references/skill_info.yaml`.
- Added `dataset.label_map` to evaluate, export, and inference templates.
- Kept `export.onnx_file` and `inference.output_file` as resolver/spec mappings without declaring them as file outputs, avoiding local runner path pre-creation.
- Updated parent skill instructions for supported actions, dataset conversion limitations, checkpoint handoff, and output file behavior.

## Remaining issues

- `dataset_convert` remains blocked until a compatible raw DeepStream BodyPose JSON dataset is available. The available S3 sample is already converted.

## Files changed

- `models/pose-classification/SKILL.md`
- `models/pose-classification/references/skill_info.yaml`
- `models/pose-classification/references/spec_template_dataset_convert.yaml`
- `models/pose-classification/references/spec_template_evaluate.yaml`
- `models/pose-classification/references/spec_template_export.yaml`
- `models/pose-classification/references/spec_template_inference.yaml`
- `docs/model-validation/pose-classification.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Partially validated: all packaged train/evaluate/inference/export/resume flows pass after model-skill metadata/template fixes. Dataset conversion is wired but blocked by missing compatible raw BodyPose JSON source data.
