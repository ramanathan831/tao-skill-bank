# Model: centerpose

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: fail for TensorRT evaluate/inference; engine generation passed
- prune: not supported by this model skill
- retrain: not supported by this model skill
- dataset convert: not supported by this model skill
- other: resume-from-checkpoint train variant: pass
- other: gen_trt_engine: pass
- other: inference on tao-deployed gen_trt_engine model: fail

## Dataset used

- Source: s3://nvcf-storage-handling/data/purpose_built_models_centerpose_train/train.tar.gz, s3://nvcf-storage-handling/data/purpose_built_models_centerpose_val/val.tar.gz, and s3://nvcf-storage-handling/data/purpose_built_models_centerpose_test/test.tar.gz
- Notes: Used the packaged CenterPose bike tarballs and extracted them before launch. Train, val, and test each contained one bike sequence folder with paired `.png` and `.json` files. The specs used category `bike`, `num_classes: 1`, `num_joints: 8`, and one-GPU local-docker execution.
- Any dataset compatibility issues: none. The dataset includes camera intrinsics in the JSON annotations; inference/deploy specs used intrinsics from the extracted sample data.

## Training result

- Training completed: yes
- Best checkpoint produced: yes, as the single concrete epoch checkpoint from this validation run
- Best checkpoint path: `/workspace/run/results/train/model_epoch_000_step_00008.pth`
- Other checkpoints produced:
  - `/workspace/run/results/train/centerpose_model_latest.pth` symlink to `model_epoch_000_step_00008.pth`
  - Resume validation produced `/workspace/run/results/resume_train/model_epoch_001_step_00016.pth`
  - Resume validation produced `/workspace/run/results/resume_train/centerpose_model_latest.pth` symlink to `model_epoch_001_step_00016.pth`

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00008.pth`
- Inference checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00008.pth`
- Export checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00008.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00008.pth`
- Deploy artifacts used: `/workspace/run/results/export/centerpose.onnx` for `gen_trt_engine`, then `/workspace/run/results/gen_trt_engine/centerpose.engine` for TensorRT evaluate/inference
- Were checkpoint paths selected through the proper resolver: yes for the direct local-docker model-skill path. The packaged skill documents using the concrete `model_epoch_...pth` checkpoint when not using the SDK resolver, and all checkpoint-dependent specs used that exact file.
- Any incorrect latest-checkpoint behavior found: no. `centerpose_model_latest.pth` was present but was not used for checkpoint-dependent actions.

## Issues found

- Model skill issues:
  - The deploy skill note was too narrow: it described the CenterPose TensorRT postprocessor failure as a 7.0 RC deploy-alias issue, but the requested `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525` image also builds the engine and then fails deploy evaluate/inference with the same `scores` scalar-conversion error.
- Config issues:
  - `automl_policy=on` conflicts with the explicit instruction not to run workflow skills for this validation pass. Because `centerpose` is AutoML-enabled and the AutoML path is an application/workflow skill, this report validates the direct model-skill action path and does not execute the AutoML wrapper.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - The host `nvidia-smi` smoke test prints valid GPU tables but exits nonzero due infoROM warnings. Docker GPU access still worked for all CenterPose PyTorch and deploy actions.
  - Deploy TensorRT evaluate and inference fail in `/usr/local/lib/python3.12/dist-packages/nvidia_tao_deploy/cv/centerpose/utils.py` at `item['score'] = float(dets['scores'][i][j])` because the score has a trailing singleton dimension.
- Fresh-install issues:
  - TAO Deploy `gen_trt_engine` with the validation-fixes image writes the engine and exits 0, but deploy evaluate/inference must still be run and checked separately.

## Fixes made

- Updated `models/centerpose/SKILL.md` and `models/centerpose/deploy/SKILL.md` to document that the validation-fixes deploy image can build the engine while TensorRT evaluate/inference still fail, and to require separate pass/fail handling for `gen_trt_engine` versus deploy evaluate/inference.

## Remaining issues

- TensorRT evaluate and TensorRT inference remain blocked by the CenterPose deploy runtime postprocessor in the requested validation-fixes deploy image.
- AutoML workflow execution was intentionally not tested because workflow skills are out of scope for this request.

## Files changed

- `models/centerpose/SKILL.md`
- `models/centerpose/deploy/SKILL.md`
- `validation-reports/centerpose.md`

## Final status

- Partially validated. Train, checkpoint evaluate, checkpoint inference, export, resume, and TensorRT engine generation pass. Deploy TensorRT evaluate/inference fail after loading the generated engine because of the deploy postprocessor scalar-conversion bug.
