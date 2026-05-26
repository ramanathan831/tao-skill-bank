# Model: depth-net-stereo

## Supported actions tested
- train: pass
- resume / retrain from checkpoint: pass
- eval: pass
- inference: pass
- export: pass
- quantize: pass
- dataset convert: pass
- deploy gen_trt_engine: pass
- inference on tao-deployed gen_trt_engine model: pass
- eval on tao-deployed gen_trt_engine model: pass
- prune: unsupported by this model skill
- AutoML: not run; AutoML is enabled for the model, but workflow skills are out of scope for this validation pass

## Dataset used
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_depth_net_test/`
- Notes: Used the available Middlebury-style stereo depth data from the shared DepthNet S3 datasets. The downloaded annotations were rewritten to the local Docker mount path under `/workspace/run/data/<split>/annotations.txt`.
- Any dataset compatibility issues: The dataset is tiny and intended only as a fresh-install smoke dataset. Metrics are not representative. For model inference, a two-column `GenericDataset` annotation file was derived from the validation left/right image pairs.

## Training result
- Training completed: yes
- Best checkpoint produced: no dedicated best-checkpoint artifact was produced by this one-epoch smoke run
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/model_epoch_000_step_00004.pth`
  - `/workspace/run/results/train/dn_model_latest.pth -> model_epoch_000_step_00004.pth`
  - `/workspace/run/results/resume_train/model_epoch_001_step_00008.pth`
  - `/workspace/run/results/resume_train/dn_model_latest.pth -> model_epoch_001_step_00008.pth`
- Training metrics:
  - `val_loss: 20.877222061157227`
  - `train_loss: 33.69976043701172`
- Resume metrics:
  - `val_loss: 10.926003456115723`
  - `train_loss: 19.939163208007812`

## Checkpoint/action verification
- Eval checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00004.pth`
- Inference checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00004.pth`
- Export checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00004.pth`
- Quantize model path used: `/workspace/run/results/train/model_epoch_000_step_00004.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00004.pth`
- Deploy gen_trt_engine model used: `/workspace/run/results/export/depth_net_stereo.onnx`
- Deploy evaluate/inference engine used: `/workspace/run/results/deploy/depth_net_stereo.engine`
- Were checkpoint paths selected through the proper resolver: yes. The model skill's `parent_model` / `resume_model` action wiring maps these actions to the parent train result. The scratch specs were verified to reference the exact epoch checkpoint rather than `dn_model_latest.pth`.
- Any incorrect latest-checkpoint behavior found: no. The latest symlink existed, but dependent actions did not rely on it.

## Issues found
- Model skill issues:
  - The `depth_net convert` documentation did not include the mandatory `results_dir` field required by the fresh-install conversion spec.
  - The local Docker example did not set writable home/cache environment variables for host-UID execution, which can break PyTorch/matplotlib cache creation in fresh installs.
- Config issues:
  - The deploy template set `gen_trt_engine.verbose: true`, producing unnecessary TensorRT builder diagnostics by default. The runtime deploy spec used `verbose: false`.
  - Parent export still emitted a very large ONNX trace in `export.log` despite `export.verbose: false`; export passed and produced a valid ONNX.
- Dataset issues:
  - No model-specific pretrained Depth Anything v2 / EdgeNeXt checkpoint was present in the inspected S3 data. Training therefore ran from scratch for smoke validation.
- Checkpoint issues:
  - No fragile latest-checkpoint behavior found.
- Docker/local execution issues:
  - Running as the host UID requires writable `HOME`, `MPLCONFIGDIR`, `TORCHINDUCTOR_CACHE_DIR`, and `XDG_CACHE_HOME` paths under the mounted output directory.
- Fresh-install issues:
  - Deploy commands succeed with zero exit code and write expected artifacts, but several deploy `status.json` files finish with `status: RUNNING` after a success message. This appears to be deploy entrypoint/status behavior rather than model skill wiring.
  - Deploy telemetry reports `'str' object has no attribute 'decode'` after successful commands. This did not affect action success.

## Fixes made
- Added `results_dir` to the stereo dataset conversion template in `models/depth-net-stereo/SKILL.md`.
- Updated the local Docker execution instructions in `models/depth-net-stereo/SKILL.md` to create and pass writable home/cache paths when using `--user`.
- Changed `models/depth-net-stereo/references/spec_template_deploy.yaml` so `gen_trt_engine.verbose` defaults to `false`.

## Remaining issues
- Deploy `status.json` final status can remain `RUNNING` even when the command logs success and exits zero.
- Parent export log can still include a large ONNX graph trace despite `verbose: false`.
- No dedicated best-checkpoint artifact was produced during the one-epoch smoke run; downstream actions used the exact epoch checkpoint produced by train.

## Files changed
- `models/depth-net-stereo/SKILL.md`
- `models/depth-net-stereo/references/spec_template_deploy.yaml`
- `validation-reports/depth-net-stereo.md`

## Final status
- Fully validated
