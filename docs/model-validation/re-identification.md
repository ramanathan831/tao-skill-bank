# Model: re-identification

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the Re-Identification model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- train: pass, AutoML default route with two Bayesian recommendations using `cmc_rank_1` maximize
- eval/evaluate: pass
- inference: pass
- export: pass
- resume training: pass through `re_identification train -e` with `train.resume_training_checkpoint_path`
- deploy: unsupported by the packaged Re-Identification model skill; no deploy sub-skill is packaged
- prune: unsupported by the packaged Re-Identification model skill
- quantize: unsupported by the packaged Re-Identification model skill
- retrain: unsupported as a standalone action; resume uses `train`
- dataset convert: unsupported by the packaged Re-Identification PyT CLI/model skill
- other: `default_specs` exists in the real CLI but is not a user action in the model skill

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_re_identification_train/`
- Notes: used `sample_train.tar.gz`, `sample_test.tar.gz`, and `sample_query.tar.gz`. The train archive contains 100 identities, so validation used `dataset.num_classes=100` instead of the common six-class override.
- Any dataset compatibility issues: no issue for train/evaluate/inference/export/resume. The model does not package a dataset conversion action.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/re-identification/results_cmc/65dd9046-3cf1-4604-8551-ccee50f6f75f/results_dir/train/model_epoch_000_step_00099.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/re-identification/results_cmc/d7438a4e-5b1a-477d-bd85-3eb57b75cc4f/results_dir/train/model_epoch_000_step_00099.pth`
  - resume produced `/tmp/tao-automl-validation/re-identification/896f771a-d611-4057-8e97-aacae4c2f542/results_dir/train/model_epoch_001_step_00198.pth`
- AutoML recommendations:
  - rec 0: job `65dd9046-3cf1-4604-8551-ccee50f6f75f`, `model.dropout_rate=0.08763728054156714`, `train.optim.base_lr=0.00044366938191368`, `cmc_rank_1=0.025751073`, selected as best
  - rec 1: job `d7438a4e-5b1a-477d-bd85-3eb57b75cc4f`, `model.dropout_rate=0.015402953228639516`, `train.optim.base_lr=0.00046691817070518467`, `cmc_rank_1=0.025751073`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 `model_epoch_000_step_00099.pth`; evaluate status logged loading that exact checkpoint and finished successfully.
- Inference checkpoint used: best AutoML rec 0 `model_epoch_000_step_00099.pth`; inference wrote `/tmp/tao-automl-validation/re-identification/manual_outputs/reid_inference.json`.
- Export checkpoint used: best AutoML rec 0 `model_epoch_000_step_00099.pth`; export wrote `/tmp/tao-automl-validation/re-identification/manual_outputs/re_identification.onnx`.
- Resume/retrain checkpoint used: best AutoML rec 0 `model_epoch_000_step_00099.pth`; resume log restored that exact checkpoint and produced epoch 1 step 198.
- Were checkpoint paths selected through the proper resolver: yes, validation used `tao_sdk.checkpoints.get_checkpoint_path` and selected `model_epoch_000_step_00099.pth` from the best AutoML child job.
- Any incorrect latest-checkpoint behavior found: yes in metadata before fixes. Re-ID writes `reid_model_latest.pth` as a latest symlink, but downstream actions now have checkpoint inputs and resolver mappings so callers do not need to guess or blindly use latest.

## Issues found

- Model skill issues:
  - Train resume, evaluate, inference, and export metadata lacked checkpoint inputs and resolver mappings.
  - Export, inference, and evaluate plot paths need explicit spec fields, but those file paths must not be declared as pre-created file outputs for the current local runner.
- Config issues:
  - The first AutoML validation attempt used a generic `val_loss` extractor. Re-ID train status files report `cmc_rank_1`, `cmc_rank_5`, `cmc_rank_10`, and `mAP`, not `val_loss`, so successful train jobs were marked AutoML metric failures until the metric was changed to `cmc_rank_1` maximize.
- Dataset issues:
  - None for the supported actions.
- Checkpoint issues:
  - Re-ID emits both `model_epoch_*.pth` and `reid_model_latest.pth`; latest must not be used unless explicitly requested.
- Docker/local execution issues:
  - None after metadata fixes. Checkpoint-loading actions used `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted local resume/eval/export/inference checkpoint loading.
- Fresh-install issues:
  - A fresh install would fail or miswire checkpoint-consuming actions because the model metadata did not expose checkpoint inputs/mappings.

## Fixes made

- Added train resume, evaluate, export, and inference checkpoint inputs plus `spec_params` resolver mappings in `models/re-identification/references/skill_info.yaml`.
- Kept `export.onnx_file`, `inference.output_file`, `evaluate.output_cmc_curve_plot`, and `evaluate.output_sampled_matches_plot` as resolver/spec mappings without declaring them as file outputs, avoiding local runner path pre-creation.
- Updated parent skill instructions for supported actions, AutoML metric direction, unsupported actions, checkpoint handoff, and output file behavior.
- Updated the per-network action inventory.

## Remaining issues

- None for packaged Re-Identification model-skill actions. Dataset conversion, deploy, prune, quantize, and standalone retrain are not advertised by the packaged Re-Identification model skill.

## Files changed

- `models/re-identification/SKILL.md`
- `models/re-identification/references/skill_info.yaml`
- `docs/model-validation/re-identification.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Fully validated: all packaged Re-Identification model-skill actions pass after model-skill metadata fixes. Unsupported dataset conversion, deploy, prune, quantize, and standalone retrain are documented as unsupported by the real packaged CLI/model skill.
