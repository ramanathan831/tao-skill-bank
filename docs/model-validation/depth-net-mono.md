# Model: depth-net-mono

Validated on 2026-05-25 with `platform=local-docker`, `image=default`.
The PyT image resolved to
`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch`.
The source-fixed quantize rerun used
`nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
The original validation pass used direct model training because AutoML routing
was explicitly out of scope. After the default AutoML request, train was rerun
through `AutoMLRunner` + `DockerSDK` with a two-trial Bayesian search.

## Supported actions tested

- train: pass after config fix; an initial relative config with non-null `dataset.min_depth` / `dataset.max_depth` failed before training.
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`.
- eval: pass.
- inference: pass.
- export: pass.
- deploy: pass for deploy `gen_trt_engine`, deploy `inference`, and deploy `evaluate`.
- prune: not supported by the model skill.
- quantize: pass in the rebuilt PyT image after the DepthNet quantize
  checkpoint-loading fix.
- retrain/resume: pass.
- dataset convert: not packaged as a model skill action. The PyT `depth_net` CLI exposes `convert`, but this skill has no convert schema/action wiring.
- parent gen_trt_engine: fail before fix; the PyT `depth_net` CLI rejects `gen_trt_engine`. TensorRT engine generation is supported through the deploy sub-skill and passed.

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_test/`
- Notes: the available depth dataset is stereo-style, with left image, right image, and depth columns. For mono validation, derived small mono smoke-test annotations by keeping the left image and depth columns for train/eval, and the left image column for inference. The AutoML rerun used two real train rows and two real validation rows.
- Any dataset compatibility issues: no native mono dataset was present in S3. The derived mono annotations were compatible with `RelativeMonoDataset` once the relative config used `dataset.dataset_name: MonoDataset`, `model.model_type: RelativeDepthAnything`, and null metric depth bounds.

## Training result

- Training completed: yes.
- AutoML completed: yes, 2/2 Bayesian recommendations succeeded.
- Best checkpoint produced: no explicit best-checkpoint artifact was produced by the one-epoch smoke runs.
- Best checkpoint path:
  `/tmp/tao-automl-validation/depth-net-mono/results/6d6ebc64-162f-4532-aecf-17d7c49478e3/results_dir/train/model_epoch_000_step_00002.pth`
- AutoML best result: rec 0, job
  `6d6ebc64-162f-4532-aecf-17d7c49478e3`, `val_loss=87.105`,
  `train.optim.lr=0.00010705877915001426`,
  `train.optim.lr_decay=0.17659394812061824`.
- Other AutoML result: rec 1, job
  `836dd391-299b-435e-b8d5-c62ecfd39cd7`, `val_loss=98.719`,
  checkpoint
  `/tmp/tao-automl-validation/depth-net-mono/results/836dd391-299b-435e-b8d5-c62ecfd39cd7/results_dir/train/model_epoch_000_step_00002.pth`.
- Other checkpoints produced by the direct action run:
  `/tmp/tao-model-validation/depth-net-mono/results/train_relative/train/model_epoch_000_step_00001.pth`,
  `dn_model_latest.pth -> model_epoch_000_step_00001.pth`, and
  `/tmp/tao-model-validation/depth-net-mono/results/resume/train/model_epoch_001_step_00002.pth`.
- Source-fixed rerun prerequisite checkpoint:
  `/tmp/tao-source-fixed-rerun/current/depth-net-mono/results/train/train/model_epoch_000_step_00002.pth`.

## Checkpoint/action verification

- Eval checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Inference checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Export checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Original quantize checkpoint used:
  `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Source-fixed quantize checkpoint used:
  `/workspace/results/train/train/model_epoch_000_step_00002.pth`, selected
  with `tao_sdk.checkpoints.get_checkpoint_path(..., epoch=0, step=2,
  allow_latest=False)`.
- Resume/retrain checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Deploy handoff used: exported ONNX `/tao-workspace/results/export/depth_net_mono.onnx`, then generated TensorRT engine `/tao-workspace/results/deploy_gen_trt_engine/depth_net_mono.engine`.
- AutoML best checkpoint used: the best trial checkpoint above, selected by `val_loss` from AutoML state and not by the latest symlink.
- Were checkpoint paths selected through the proper resolver: yes for the manual user workflow and the source-fixed quantize rerun; the validated paths used exact epoch/step checkpoint names rather than the latest symlink. The skill docs require SDK/model resolver selection for best, epoch, step, and explicit latest behavior.
- Any incorrect latest-checkpoint behavior found: no runtime action blindly selected `dn_model_latest.pth`, but the model skill metadata was missing export/quantize checkpoint inputs before the direct-action fix.

## Issues found

- Model skill issues:
  - `schemas/manifest.json` and global `models/schemas.manifest.json` advertised parent PyT `gen_trt_engine`, but `depth_net gen_trt_engine` is rejected by the real PyT CLI.
  - Parent spec templates inherited `StereoDataset` and `MetricDepthAnything` defaults, which are wrong for a fresh mono relative run.
  - `references/skill_info.yaml` did not declare `export.checkpoint`, `export.onnx_file`, or `quantize.model_path` artifact handoff.
  - Deploy `skill_info.yaml` treated generated `gen_trt_engine.trt_engine` as an input artifact.
- Config issues:
  - Relative mono training fails if `dataset.min_depth` / `dataset.max_depth` are non-null.
  - The mono parent templates used YAML anchors for `data_sources`; in the first AutoML rerun, overriding validation data also changed the train data source. The results were discarded and the fixed template was rerun.
  - Deploy template used `workspace_size: 1024`, but the current deploy image interprets the value as GiB.
- Dataset issues:
  - Only stereo depth data was available in S3, so mono annotations had to be derived from left-image plus depth columns.
- Checkpoint issues:
  - No named best checkpoint was emitted by the one-epoch smoke run; exact epoch/step checkpoint selection was required.
- Docker/local execution issues:
  - Parent quantize failed inside the original TAO PyT image with `AttributeError: 'MonoDepthNetPlModel' object has no attribute 'load_state_dict_from_checkpoint'`.
  - Deploy actions completed successfully but printed a non-fatal telemetry warning.
- Fresh-install issues:
  - Fresh users following packaged templates could start from stereo/metric defaults unless they supplied all mono overrides manually.

## Fixes made

- Removed parent PyT `gen_trt_engine` from the depth-net-mono schema manifests; deploy `gen_trt_engine` remains supported by the deploy sub-skill.
- Updated mono parent templates to default to `dataset.dataset_name: MonoDataset`, `RelativeMonoDataset` sources, and `model.model_type: RelativeDepthAnything`.
- Removed YAML aliases from the mono parent spec templates so train, val, test, and infer data sources are independent when action overrides are merged.
- Added export and quantize checkpoint/artifact inputs to `references/skill_info.yaml`.
- Removed generated deploy engine path from deploy inputs and documented it as an output.
- Updated deploy defaults to FP32 and `workspace_size: 4`.
- Documented the exact checkpoint pattern, resolver expectations, relative mono `min_depth` / `max_depth` behavior, quantize SDK failure, and deploy handoff rules.
- Reran train through AutoML with Bayesian search, `automl_max_recommendations=2`, metric `val_loss`, and explicit minimal search over `train.optim.lr` and `train.optim.lr_decay`.
- Reran `depth_net quantize` on the rebuilt PyT image after the source fix;
  it initialized the `torchao` backend and saved
  `/workspace/results/quantize/quantize/quantized_model_torchao.pth`.

## Remaining issues

- Dataset convert remains unvalidated as a model skill action because the skill does not package a convert schema/action, even though the raw PyT CLI exposes `convert`.

## Files changed

- `models/depth-net-mono/SKILL.md`
- `models/depth-net-mono/references/skill_info.yaml`
- `models/depth-net-mono/references/spec_template_train.yaml`
- `models/depth-net-mono/references/spec_template_evaluate.yaml`
- `models/depth-net-mono/references/spec_template_export.yaml`
- `models/depth-net-mono/references/spec_template_inference.yaml`
- `models/depth-net-mono/references/spec_template_quantize.yaml`
- `models/depth-net-mono/references/spec_template_gen_trt_engine.yaml`
- `models/depth-net-mono/references/spec_template_deploy.yaml`
- `models/depth-net-mono/deploy/SKILL.md`
- `models/depth-net-mono/deploy/skill_info.yaml`
- `models/depth-net-mono/schemas/manifest.json`
- `models/schemas.manifest.json`
- `docs/model-validation/depth-net-mono.md`

## Final status

- Fully validated for packaged model-skill actions: train, default AutoML train routing with two Bayesian recommendations, eval, inference, export, resume/retrain, deploy engine generation, deploy inference, deploy evaluate, and source-fixed quantize all passed. Dataset convert remains unadvertised by this model skill.
