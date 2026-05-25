# Model: optical-inspection

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the Optical Inspection model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- train: pass, AutoML default route with two Bayesian recommendations
- eval/evaluate: pass
- inference: pass
- export: pass
- deploy gen_trt_engine: pass after removing the pre-created engine file output from deploy metadata
- deploy evaluate on tao-deployed gen_trt_engine model: pass after fixing deploy evaluate dataset wiring and batch size
- deploy inference on tao-deployed gen_trt_engine model: pass
- prune: unsupported by the packaged Optical Inspection PyT CLI/model skill
- quantize: unsupported by the packaged Optical Inspection PyT CLI/model skill
- retrain/resume: pass
- dataset convert: blocked/not packaged as a model skill action; the PyT CLI has `dataset_convert`, but the available S3 data is already converted and does not include the required raw Factory PCB/golden CSV layout
- other: parent PyT `gen_trt_engine` is unsupported and was removed from parent model metadata; deploy sub-skill owns TensorRT actions

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_optical_inspection_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_optical_inspection_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_optical_inspection_test/`
- Notes: each split contains preconverted `images.tar.gz` and `dataset.csv`. The smoke run used `dataset.batch_size=4`, `dataset.workers=0`, `train.num_epochs=1`, and `train.validation_interval=1`.
- Any dataset compatibility issues: no issue for train/evaluate/inference/export/deploy. No compatible raw PCB/golden CSV dataset was found for `dataset_convert`.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/optical-inspection/results/41f2d2ac-d664-4331-bb86-a0fc4394f6fb/results_dir/train/model_epoch_000_step_00006.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/optical-inspection/results/2845721c-ca01-48f0-b7ec-6ffa71ba2402/results_dir/train/model_epoch_000_step_00006.pth`
  - resume/retrain produced `/tmp/tao-automl-validation/optical-inspection/50c2314f-3e9a-4785-b9d9-c43fc7320a80/results_dir/train/model_epoch_001_step_00012.pth`
- AutoML recommendations:
  - rec 0: job `41f2d2ac-d664-4331-bb86-a0fc4394f6fb`, `model.margin=1.962035482185719`, `train.optim.lr=0.00031440839995364135`, metric `96.0`, selected as best by tie-break
  - rec 1: job `2845721c-ca01-48f0-b7ec-6ffa71ba2402`, `model.margin=1.4936729161388924`, `train.optim.lr=0.00045518408939321383`, metric `96.0`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 checkpoint above; evaluation passed with total accuracy `96.0`.
- Inference checkpoint used: best AutoML rec 0 checkpoint above; inference passed on the real test split.
- Export checkpoint used: best AutoML rec 0 checkpoint above; export wrote `/tmp/tao-automl-validation/optical-inspection/manual_outputs/optical_inspection.onnx`.
- Resume/retrain checkpoint used: best AutoML rec 0 checkpoint above; resume train log set `train.resume_training_checkpoint_path` to the resolver-selected checkpoint and produced epoch 1 output.
- Deploy artifact handoff used: exported ONNX to deploy `gen_trt_engine`, generated `/tmp/tao-automl-validation/optical-inspection/manual_outputs/optical_inspection.engine`, then deploy evaluate/inference used that exact engine.
- Were checkpoint paths selected through the proper resolver: yes, validation used `tao_sdk.checkpoints.get_checkpoint_path` and selected `model_epoch_000_step_00006.pth` from the best AutoML child job.
- Any incorrect latest-checkpoint behavior found: no latest-checkpoint fallback was used after metadata fixes; downstream actions used the selected AutoML checkpoint.

## Issues found

- Model skill issues:
  - Parent metadata advertised PyT `gen_trt_engine`, but `optical_inspection --help` in the PyT image does not expose that subtask.
  - Parent metadata lacked checkpoint inputs and resolver mappings for evaluate, inference, export, and train resume.
  - Deploy metadata declared `gen_trt_engine.trt_engine` as a file input/output, which made the local runner pre-create the engine path as a directory.
  - Deploy evaluate metadata declared `dataset.test_dataset.*`, but the deploy evaluate implementation reads `dataset.infer_dataset.*`.
- Config issues:
  - Deploy evaluate defaulted to `evaluate.batch_size=-1`, which produced a TensorRT static-shape mismatch for the default static-batch ONNX export.
- Dataset issues:
  - No compatible raw Factory PCB/golden CSV source was available for dataset conversion.
- Checkpoint issues:
  - No model checkpoint naming issue after adding metadata and resolver mappings.
- Docker/local execution issues:
  - Deploy telemetry reports a non-fatal decode warning after successful deploy commands.
- Fresh-install issues:
  - A fresh install would fail parent checkpoint-consuming actions and deploy evaluate before these metadata/template fixes.

## Fixes made

- Added train resume, evaluate, export, and inference checkpoint inputs plus `spec_params` mappings in `models/optical-inspection/references/skill_info.yaml`.
- Removed unsupported parent PyT `gen_trt_engine` from the parent model metadata.
- Removed pre-created deploy engine file input/output metadata from `models/optical-inspection/deploy/skill_info.yaml`.
- Changed deploy evaluate inputs to `dataset.infer_dataset.csv_path` and `dataset.infer_dataset.images_dir`.
- Set deploy `evaluate.batch_size: 1` in `models/optical-inspection/references/spec_template_deploy_experiment.yaml`.
- Updated parent and deploy skill instructions for AutoML default training, checkpoint resolution, deploy-only TensorRT, deploy evaluate dataset wiring, and dataset-convert limitations.

## Remaining issues

- `dataset_convert` remains unvalidated because it is not packaged as a model skill action and the available S3 Optical Inspection data is already converted. Validating it requires a compatible raw Factory PCB dataset with golden CSV inputs.

## Files changed

- `models/optical-inspection/SKILL.md`
- `models/optical-inspection/references/skill_info.yaml`
- `models/optical-inspection/references/spec_template_deploy_experiment.yaml`
- `models/optical-inspection/deploy/SKILL.md`
- `models/optical-inspection/deploy/skill_info.yaml`
- `docs/model-validation/optical-inspection.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Partially validated: all packaged Optical Inspection parent and deploy skill actions pass after model-skill metadata/template fixes. Dataset conversion remains blocked by missing compatible raw source data and missing packaged model-skill action metadata.
