# Model: action-recognition

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: not supported by this model skill
- prune: not supported by this model skill
- retrain: not supported by this model skill
- dataset convert: not supported by this model skill
- other: resume-from-checkpoint train variant: pass

## Dataset used

- Source: s3://nvcf-storage-handling/data/purpose_built_models_action_recognition_train/
- Notes: Used the packaged action-recognition sample archives: `train.tar.gz`, `test.tar.gz`, and `test/smile.tar.gz`. Downloaded to a local scratch workspace and extracted before launch because the model skill requires directories, not tarballs, for train/evaluate/inference.
- Any dataset compatibility issues: none. The extracted dataset matched the model skill layout: `train/<class>/<clip>/rgb/*.png`, `test/<class>/<clip>/rgb/*.png`, and a `smile` inference clip.

## Training result

- Training completed: yes
- Best checkpoint produced: yes, as the single concrete epoch checkpoint from this validation run
- Best checkpoint path: `/workspace/run/results/train/model_epoch_000_step_00005.pth`
- Other checkpoints produced:
  - `/workspace/run/results/train/ar_model_latest.pth` symlink to `model_epoch_000_step_00005.pth`
  - Resume validation produced `/workspace/run/results/resume_train/model_epoch_001_step_00010.pth`
  - Resume validation produced `/workspace/run/results/resume_train/ar_model_latest.pth` symlink to `model_epoch_001_step_00010.pth`

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00005.pth`
- Inference checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00005.pth`
- Export checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00005.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00005.pth`
- Were checkpoint paths selected through the proper resolver: yes for the direct local-docker model-skill path. The packaged skill explicitly documents selecting the concrete `model_epoch_...pth` file when not using the SDK resolver, and all downstream specs used that exact file.
- Any incorrect latest-checkpoint behavior found: no. `ar_model_latest.pth` was present but was not used for any checkpoint-dependent action.

## Issues found

- Model skill issues:
  - None found for the supported model-skill actions.
- Config issues:
  - `automl_policy=on` conflicts with the explicit instruction not to run workflow skills for this validation pass. Because `action-recognition` is AutoML-enabled and the AutoML path is an application/workflow skill, this report validates the direct model-skill action path and does not execute the AutoML wrapper.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - The host `nvidia-smi` smoke test prints valid GPU tables but exits nonzero due infoROM warnings. Docker GPU access still worked for all TAO container actions.
- Fresh-install issues:
  - The host initially lacked `aws`, `PyYAML`, `boto3`, and Docker's Python package. Installed generic local tooling only; no credentials were written into repo files or generated specs.

## Fixes made

- No model skill code or instruction fixes were required for `action-recognition`.

## Remaining issues

- AutoML workflow execution was intentionally not tested because workflow skills are out of scope for this request.

## Files changed

- `validation-reports/action-recognition.md`

## Final status

- Fully validated for all supported `action-recognition` model-skill actions on local-docker with `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
