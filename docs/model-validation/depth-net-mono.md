# Model: depth-net-mono

## Supported actions tested

- train: pass after config fix; an initial relative config with non-null `dataset.min_depth` / `dataset.max_depth` failed before training.
- eval: pass.
- inference: pass.
- export: pass.
- deploy: pass for deploy `gen_trt_engine`, deploy `inference`, and deploy `evaluate`.
- prune: not supported by the model skill.
- quantize: fail, blocked in TAO SDK code after correct checkpoint handoff.
- retrain/resume: pass.
- dataset convert: not packaged as a model skill action. The PyT `depth_net` CLI exposes `convert`, but this skill has no convert schema/action wiring.
- parent gen_trt_engine: fail before fix; the PyT `depth_net` CLI rejects `gen_trt_engine`. TensorRT engine generation is supported through the deploy sub-skill and passed.

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_test/`
- Notes: the available depth dataset is stereo-style, with left image, right image, and depth columns. For mono validation, derived small mono smoke-test annotations by keeping the left image and depth columns for train/eval, and the left image column for inference.
- Any dataset compatibility issues: no native mono dataset was present in S3. The derived mono annotations were compatible with `RelativeMonoDataset` once the relative config used `dataset.dataset_name: MonoDataset`, `model.model_type: RelativeDepthAnything`, and null metric depth bounds.

## Training result

- Training completed: yes.
- Best checkpoint produced: no explicit best-checkpoint artifact was produced by the one-epoch smoke run.
- Best checkpoint path: n/a.
- Other checkpoints produced:
  - `/tmp/tao-model-validation/depth-net-mono/results/train_relative/train/model_epoch_000_step_00001.pth`
  - `/tmp/tao-model-validation/depth-net-mono/results/train_relative/train/dn_model_latest.pth -> model_epoch_000_step_00001.pth`
  - Resume produced `/tmp/tao-model-validation/depth-net-mono/results/resume/train/model_epoch_001_step_00002.pth`.

## Checkpoint/action verification

- Eval checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Inference checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Export checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Quantize checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Resume/retrain checkpoint used: `/tao-workspace/results/train_relative/train/model_epoch_000_step_00001.pth`.
- Deploy handoff used: exported ONNX `/tao-workspace/results/export/depth_net_mono.onnx`, then generated TensorRT engine `/tao-workspace/results/deploy_gen_trt_engine/depth_net_mono.engine`.
- Were checkpoint paths selected through the proper resolver: yes for the manual user workflow; the validated paths used exact epoch/step checkpoint names rather than the latest symlink. The skill docs now require SDK/model resolver selection for best, epoch, step, and explicit latest behavior.
- Any incorrect latest-checkpoint behavior found: no runtime action blindly selected `dn_model_latest.pth`, but the model skill metadata was missing export/quantize checkpoint inputs before this fix.

## Issues found

- Model skill issues:
  - `schemas/manifest.json` and global `models/schemas.manifest.json` advertised parent PyT `gen_trt_engine`, but `depth_net gen_trt_engine` is rejected by the real PyT CLI.
  - Parent spec templates inherited `StereoDataset` and `MetricDepthAnything` defaults, which are wrong for a fresh mono relative run.
  - `references/skill_info.yaml` did not declare `export.checkpoint`, `export.onnx_file`, or `quantize.model_path` artifact handoff.
  - Deploy `skill_info.yaml` treated generated `gen_trt_engine.trt_engine` as an input artifact.
- Config issues:
  - Relative mono training fails if `dataset.min_depth` / `dataset.max_depth` are non-null.
  - Deploy template used `workspace_size: 1024`, but the current deploy image interprets the value as GiB.
- Dataset issues:
  - Only stereo depth data was available in S3, so mono annotations had to be derived from left-image plus depth columns.
- Checkpoint issues:
  - No named best checkpoint was emitted by the one-epoch smoke run; exact epoch/step checkpoint selection was required.
- Docker/local execution issues:
  - Parent quantize failed inside the TAO PyT image with `AttributeError: 'MonoDepthNetPlModel' object has no attribute 'load_state_dict_from_checkpoint'`.
  - Deploy actions completed successfully but printed a non-fatal telemetry warning.
- Fresh-install issues:
  - Fresh users following packaged templates could start from stereo/metric defaults unless they supplied all mono overrides manually.

## Fixes made

- Removed parent PyT `gen_trt_engine` from the depth-net-mono schema manifests; deploy `gen_trt_engine` remains supported by the deploy sub-skill.
- Updated mono parent templates to default to `dataset.dataset_name: MonoDataset`, `RelativeMonoDataset` sources, and `model.model_type: RelativeDepthAnything`.
- Added export and quantize checkpoint/artifact inputs to `references/skill_info.yaml`.
- Removed generated deploy engine path from deploy inputs and documented it as an output.
- Updated deploy defaults to FP32 and `workspace_size: 4`.
- Documented the exact checkpoint pattern, resolver expectations, relative mono `min_depth` / `max_depth` behavior, quantize SDK failure, and deploy handoff rules.

## Remaining issues

- `depth_net quantize` remains unresolved in the TAO PyT image; the skill passes the exact checkpoint correctly, but the SDK fails before quantization can run.
- Dataset convert remains unvalidated as a model skill action because the skill does not package a convert schema/action, even though the raw PyT CLI exposes `convert`.

## Files changed

- `models/depth-net-mono/SKILL.md`
- `models/depth-net-mono/references/skill_info.yaml`
- `models/depth-net-mono/references/spec_template_train.yaml`
- `models/depth-net-mono/references/spec_template_evaluate.yaml`
- `models/depth-net-mono/references/spec_template_export.yaml`
- `models/depth-net-mono/references/spec_template_inference.yaml`
- `models/depth-net-mono/references/spec_template_quantize.yaml`
- `models/depth-net-mono/references/spec_template_deploy.yaml`
- `models/depth-net-mono/deploy/SKILL.md`
- `models/depth-net-mono/deploy/skill_info.yaml`
- `models/depth-net-mono/schemas/manifest.json`
- `models/schemas.manifest.json`
- `docs/model-validation/depth-net-mono.md`

## Final status

- Partially validated: train, eval, inference, export, resume/retrain, deploy engine generation, deploy inference, and deploy evaluate passed. Quantize is blocked by an SDK-side failure after correct checkpoint handoff.
