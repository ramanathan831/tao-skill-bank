# Model: ml-recog

Validated on 2026-05-25 with `platform=local-docker`, `image=default`
(`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch` and
`nvcr.io/nvstaging/tao/tao-toolkit-deploy:7.0.0-rc-171-multiarch`),
`num_gpus=1`, and direct model/deploy skill actions only. Workflow skills were
not run.

## Supported Actions Tested

- train: pass
- eval: pass after trusted-checkpoint load env
- inference: pass after trusted-checkpoint load env
- export: pass after trusted-checkpoint load env
- deploy: pass for TAO Deploy `gen_trt_engine`, TensorRT inference, and TensorRT evaluate
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain: pass through `train.resume_training_checkpoint_path` after trusted-checkpoint load env
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised
- other: parent PyT `gen_trt_engine` was advertised by metadata but rejected by the real PyT CLI; removed from the parent model action metadata and manifest because TAO Deploy owns it

## Dataset Used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_ml_recog_train/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/`
- Notes: used the real retail product recognition archives. Train used `known_classes/train.tar.gz` with known-class `reference.tar.gz` and `val.tar.gz`; checkpoint eval/inference used `unknown_classes/reference.tar.gz` and `unknown_classes/test.tar.gz`; deploy calibration/engine validation used `known_classes/test.tar.gz` as available calibration/input data.
- Any dataset compatibility issues: none blocking. The unknown-class eval/test split contains one class, so it is suitable for smoke validation of the action path but not broad retrieval-quality measurement. The common `num_classes=6` override is not a model field for ML-Recog.

## Training Result

- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact; the one-epoch run produced one concrete epoch/step checkpoint, which was selected for downstream validation
- Best checkpoint path: `/tmp/tao-model-validation/ml-recog/results/train/model_epoch_000_step_00044.pth`
- Other checkpoints produced: `/tmp/tao-model-validation/ml-recog/results/train/ml_model_latest.pth` symlink to the exact train checkpoint; resume validation produced `/tmp/tao-model-validation/ml-recog/results/retrain/model_epoch_001_step_00088.pth` and a `ml_model_latest.pth` symlink

## AutoML Default Training Rerun

- Default direct model training used AutoML after the default policy was corrected to `automl_policy=on`.
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_ml_recog_train/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/known_classes/`
- Algorithm: bayesian
- Recommendations requested: 2
- Metric: `val Precision at Rank 1`, maximize
- Tuned parameters: `train.optim.trunk.base_lr`, `train.optim.embedder.base_lr`
- Recommendation 0: job `2b01e3b1-341b-4713-9ae2-5c68d162259c`, metric `0.5676683235816979`, checkpoint `/tmp/tao-automl-validation/ml-recog/results/2b01e3b1-341b-4713-9ae2-5c68d162259c/results_dir/train/model_epoch_000_step_00088.pth`
- Recommendation 1: job `f110600d-318a-4ab6-aa6b-cf2182e8fbdf`, metric `0.571052860192041`, checkpoint `/tmp/tao-automl-validation/ml-recog/results/f110600d-318a-4ab6-aa6b-cf2182e8fbdf/results_dir/train/model_epoch_000_step_00088.pth`
- Best recommendation: rec 1, selected by the AutoML controller summary
- Generated spec verification: both recommendations used SDK-extracted real S3 train/reference/query archives, `dataset.num_instance=4`, `train.batch_size=4`, `train.val_batch_size=4`, and distinct Bayesian trunk/embedder base learning-rate values within the requested ranges.

## Checkpoint/Action Verification

- Eval checkpoint used: `/workspace/results/train/model_epoch_000_step_00044.pth`
- Inference checkpoint used: `/workspace/results/train/model_epoch_000_step_00044.pth`
- Export checkpoint used: `/workspace/results/train/model_epoch_000_step_00044.pth`
- Resume/retrain checkpoint used: `/workspace/results/train/model_epoch_000_step_00044.pth`
- Were checkpoint paths selected through the proper resolver: yes for the fixed parent-model metadata contract on evaluate/export/inference; direct local-docker validation supplied the exact epoch/step path because workflow/SDK resolver execution was out of scope. Resume used the explicit `train.resume_training_checkpoint_path` action input.
- Any incorrect latest-checkpoint behavior found: no executed action used `ml_model_latest.pth`. The metadata was fragile before the fix because evaluate/export/inference lacked checkpoint inputs and `parent_model` mappings.

## Issues Found

- Model skill issues:
  - Parent `skill_info.yaml` advertised `gen_trt_engine`, but the PyT `ml_recog` CLI supports only `train`, `evaluate`, `export`, `inference`, and `default_specs`.
  - Parent `skill_info.yaml` did not declare `evaluate.checkpoint`, `export.checkpoint`, or `inference.checkpoint` inputs, nor the matching `parent_model` resolver mappings.
  - The PyTorch 2.6 checkpoint-load pitfall was documented only for evaluate/inference, but export and resume hit the same trusted-checkpoint failure.
- Config issues:
  - Deploy inference template and metadata omitted `inference.input_path`. The deploy CLI failed with `Missing mandatory value: inference.input_path` when only `dataset.val_dataset.query` was configured.
- Dataset issues:
  - Unknown-class eval/test is a one-class split; fine for action validation, limited for metric interpretation.
- Checkpoint issues:
  - Evaluate, export, and resume failed without `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because the TAO Lightning checkpoint contains OmegaConf objects under PyTorch 2.6 defaults. Reruns passed with the env var for the trusted local checkpoint.
- Docker/local execution issues:
  - TAO Deploy returned process exit 0 even when the initial deploy inference run logged a configuration error, so logs/status files must be inspected, not just shell exit status.
- Fresh-install issues:
  - Fresh installs would expose stale parent `gen_trt_engine` and missing checkpoint handoff metadata before these fixes.

## Fixes Made

- Added checkpoint inputs and `parent_model` mappings for evaluate, export, and inference in `models/ml-recog/references/skill_info.yaml`.
- Added `export.onnx_file` as an output mapped by `create_onnx_file`.
- Added optional `train.resume_training_checkpoint_path` input metadata for resume validation.
- Removed parent PyT `gen_trt_engine` from model action metadata and from `schemas/manifest.json`.
- Added deploy `inference.input_path` to the deploy template and deploy skill metadata.
- Updated ML-Recog documentation for deploy inference input wiring and trusted-checkpoint env usage on evaluate, inference, export, and resume/retrain.
- No additional ML-Recog model skill code change was needed for the AutoML default rerun.

## Remaining Issues

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is still required for trusted ML-Recog checkpoint-consuming actions in this TAO/PyTorch image until upstream checkpoint loading is updated.
- SDK `parent_model` resolver execution was not invoked because workflow/SDK paths were out of scope; the metadata contract is now present and direct runs used the exact checkpoint path.

## Files Changed

- `models/ml-recog/SKILL.md`
- `models/ml-recog/deploy/SKILL.md`
- `models/ml-recog/deploy/skill_info.yaml`
- `models/ml-recog/references/skill_info.yaml`
- `models/ml-recog/references/spec_template_deploy_inference.yaml`
- `models/ml-recog/schemas/manifest.json`
- `models/schemas.manifest.json`
- `docs/model-validation/ml-recog.md`
- `docs/model-validation/action-run-inventory.md`

## Final Status

Fully validated for the ML-Recog model skill's supported parent, AutoML default
train, and deploy actions on local Docker after the metadata/template fixes.
