# Model: depth-net-stereo

Validated on 2026-05-25 with `platform=local-docker`, `image=default`.
The PyT image resolved to
`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch`.
The original validation pass used direct model training because AutoML routing
was explicitly out of scope. After the default AutoML request, train was rerun
through `AutoMLRunner` + `DockerSDK` with a two-trial Bayesian search.

## Supported actions tested

- train: pass after geometry config fix; initial 112x112/64 and 112x112/112 smoke configs failed.
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`.
- eval: pass.
- inference: pass.
- export: pass.
- deploy: partial. Deploy `gen_trt_engine` pass and deploy `inference` pass; deploy `evaluate` failed after prediction generation in the deploy stereo evaluator.
- prune: not supported by the model skill.
- quantize: fail, blocked in TAO SDK code after correct checkpoint handoff.
- retrain/resume: pass.
- dataset convert: not packaged as a model skill action. The PyT `depth_net` CLI exposes `convert`, but this skill has no convert schema/action wiring.
- parent gen_trt_engine: fail before fix; the PyT `depth_net` CLI rejects `gen_trt_engine`. TensorRT engine generation is supported through the deploy sub-skill and passed.

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_test/`
- Notes: used real stereo rows with left image, right image, and PFM disparity. For smoke runtime, derived two-row stereo subsets for train/eval and two-column left/right subsets for inference. The AutoML rerun used two real train rows and two real validation rows.
- Any dataset compatibility issues: no issue with the data itself. The model requires shape-consistent FoundationStereo smoke settings; `[128, 128]` with `max_disparity: 128` passed.

## Training result

- Training completed: yes.
- AutoML completed: yes, 2/2 Bayesian recommendations succeeded.
- Best checkpoint produced: no explicit best-checkpoint artifact was produced by the one-epoch smoke runs.
- Best checkpoint path:
  `/tmp/tao-automl-validation/depth-net-stereo/results/8d8e12f6-3d32-47da-ba0c-1186ce2f67a1/results_dir/train/model_epoch_000_step_00002.pth`
- AutoML best result: rec 0, job
  `8d8e12f6-3d32-47da-ba0c-1186ce2f67a1`, `val_loss=8.334`,
  `train.optim.lr=6.051742212282045e-05`,
  `train.optim.lr_decay=0.07720342437763335`.
- Other AutoML result: rec 1, job
  `de8e0af8-158c-4437-b270-d39e6860d794`, `val_loss=9.666`,
  checkpoint
  `/tmp/tao-automl-validation/depth-net-stereo/results/de8e0af8-158c-4437-b270-d39e6860d794/results_dir/train/model_epoch_000_step_00002.pth`.
- Other checkpoints produced by the direct action run:
  `/tmp/tao-model-validation/depth-net-stereo/results/train_128/train/model_epoch_000_step_00002.pth`,
  `dn_model_latest.pth -> model_epoch_000_step_00002.pth`, and
  `/tmp/tao-model-validation/depth-net-stereo/results/resume/train/model_epoch_001_step_00004.pth`.
- Notes: failed train attempts also wrote partial step-0 checkpoints; downstream actions intentionally used only the successful `train_128` checkpoint.

## Checkpoint/action verification

- Eval checkpoint used: `/tao-workspace/results/train_128/train/model_epoch_000_step_00002.pth`.
- Inference checkpoint used: `/tao-workspace/results/train_128/train/model_epoch_000_step_00002.pth`.
- Export checkpoint used: `/tao-workspace/results/train_128/train/model_epoch_000_step_00002.pth`.
- Quantize checkpoint used: `/tao-workspace/results/train_128/train/model_epoch_000_step_00002.pth`.
- Resume/retrain checkpoint used: `/tao-workspace/results/train_128/train/model_epoch_000_step_00002.pth`.
- Deploy handoff used: exported ONNX `/tao-workspace/results/export/depth_net_stereo.onnx`, then generated TensorRT engine `/tao-workspace/results/deploy_gen_trt_engine/depth_net_stereo.engine`.
- AutoML best checkpoint used: the best trial checkpoint above, selected by `val_loss` from AutoML state and not by the latest symlink.
- Were checkpoint paths selected through the proper resolver: yes for the manual user workflow; the validated paths used exact epoch/step checkpoint names rather than the latest symlink. The skill docs require SDK/model resolver selection for best, epoch, step, and explicit latest behavior.
- Any incorrect latest-checkpoint behavior found: no runtime action blindly selected `dn_model_latest.pth`, but the model skill metadata was missing export/quantize checkpoint inputs before the direct-action fix.

## Issues found

- Model skill issues:
  - `schemas/manifest.json` and global `models/schemas.manifest.json` advertised parent PyT `gen_trt_engine`, but `depth_net gen_trt_engine` is rejected by the real PyT CLI.
  - Parent spec templates inherited `MetricDepthAnything`, `vitl`, `max_disparity: 416`, and 518x518 crop defaults that are not fresh-install safe for FoundationStereo.
  - `references/skill_info.yaml` did not declare `export.checkpoint`, `export.onnx_file`, or `quantize.model_path` artifact handoff.
  - Deploy `skill_info.yaml` treated generated `gen_trt_engine.trt_engine` as an input artifact.
- Config issues:
  - `max_disparity: 64` with a 112x112 crop failed in the training loss with a valid-mask/prediction shape mismatch.
  - 112x112 with `max_disparity: 112` failed in the FoundationStereo cost aggregation upsample path.
  - 128x128 with `max_disparity: 128` passed train/resume/export/deploy engine validation.
  - The stereo parent templates used YAML anchors for `data_sources`; this would make validation overrides overwrite train data sources in AutoML/direct merged specs.
  - Deploy template used `workspace_size: 1024`, but the current deploy image interprets the value as GiB.
- Dataset issues:
  - None for stereo; S3 had compatible left/right/disparity rows.
- Checkpoint issues:
  - No named best checkpoint was emitted by the one-epoch smoke run; exact epoch/step checkpoint selection was required.
- Docker/local execution issues:
  - Parent quantize failed inside the TAO PyT image with `AttributeError: 'StereoDepthNetPlModel' object has no attribute 'load_state_dict_from_checkpoint'`.
  - Deploy `evaluate` failed inside `stereo_evaluator.py` with `TypeError: only 0-dimensional arrays can be converted to Python scalars` after predictions were generated.
  - Deploy engine generation passed but took 331.857 seconds even at 128x128 FP32.
  - Deploy actions printed non-fatal telemetry warnings.
- Fresh-install issues:
  - Fresh users following packaged templates could start from mono/metric defaults unless they supplied all FoundationStereo overrides manually.

## Fixes made

- Removed parent PyT `gen_trt_engine` from the depth-net-stereo schema manifests; deploy `gen_trt_engine` remains supported by the deploy sub-skill.
- Updated stereo parent templates to default to `FoundationStereo`, `vits`, `max_disparity: 128`, and 128x128 fresh-install-safe crops.
- Set stereo train/eval/export/quantize template source defaults to `Middlebury` and inference template source defaults to `GenericDataset`.
- Removed YAML aliases from the stereo parent spec templates so train, val, test, and infer data sources are independent when action overrides are merged.
- Added export and quantize checkpoint/artifact inputs to `references/skill_info.yaml`.
- Removed generated deploy engine path from deploy inputs and documented it as an output.
- Updated deploy defaults to FP32 and `workspace_size: 4`.
- Documented the exact checkpoint pattern, resolver expectations, shape/max-disparity smoke constraints, quantize SDK failure, deploy evaluate failure, and deploy handoff rules.
- Reran train through AutoML with Bayesian search, `automl_max_recommendations=2`, metric `val_loss`, and explicit minimal search over `train.optim.lr` and `train.optim.lr_decay`.

## Remaining issues

- `depth_net quantize` remains unresolved in the TAO PyT image; the skill passes the exact checkpoint correctly, but the SDK fails before quantization can run.
- Deploy `depth_net evaluate` remains unresolved in the TAO Deploy image; predictions are generated, then the stereo evaluator fails while converting array metrics to scalars.
- Dataset convert remains unvalidated as a model skill action because the skill does not package a convert schema/action, even though the raw PyT CLI exposes `convert`.

## Files changed

- `models/depth-net-stereo/SKILL.md`
- `models/depth-net-stereo/references/skill_info.yaml`
- `models/depth-net-stereo/references/spec_template_train.yaml`
- `models/depth-net-stereo/references/spec_template_evaluate.yaml`
- `models/depth-net-stereo/references/spec_template_export.yaml`
- `models/depth-net-stereo/references/spec_template_inference.yaml`
- `models/depth-net-stereo/references/spec_template_quantize.yaml`
- `models/depth-net-stereo/references/spec_template_gen_trt_engine.yaml`
- `models/depth-net-stereo/references/spec_template_deploy.yaml`
- `models/depth-net-stereo/deploy/SKILL.md`
- `models/depth-net-stereo/deploy/skill_info.yaml`
- `models/depth-net-stereo/schemas/manifest.json`
- `models/schemas.manifest.json`
- `docs/model-validation/depth-net-stereo.md`

## Final status

- Partially validated: train, default AutoML train routing with two Bayesian recommendations, eval, inference, export, resume/retrain, deploy engine generation, and deploy inference passed. Quantize and deploy evaluate are blocked by SDK/deploy evaluator failures after correct artifact handoff.
