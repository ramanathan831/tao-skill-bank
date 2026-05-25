# Model: bevfusion

## Supported actions tested

- train: fail
- eval: fail
- inference: pass
- export: not supported by this model skill
- deploy: not supported by this model skill
- prune: not supported by this model skill
- retrain: not supported by this model skill
- dataset convert: pass
- other: resume-from-checkpoint train variant: fail final status, checkpoint restore verified

## Dataset used

- Source: s3://nvcf-storage-handling/data/purpose_built_models_bevfusion_train/
- Notes: Used `ImageSets.tar.gz` and `training.tar.gz`, extracted into a KITTI-style root. The converter produced `kitti_person_infos_train.pkl`, `kitti_person_infos_val.pkl`, `kitti_person_infos_trainval.pkl`, and `training/velodyne_reduced` under the same root.
- Any dataset compatibility issues: none blocking. The packaged data contains KITTI camera, LiDAR, label, calibration, and split files. The converter reported ignored `Car` annotations because the model skill default class list is `["person"]`.

## Training result

- Training completed: no. The one-epoch train loop saved a checkpoint, then the validation/evaluation tail ended with TAO `Execution status: FAIL` after `Signal 11 (SIGSEGV)` in `cuMemRetainAllocationHandle`.
- Best checkpoint produced: no explicit best checkpoint was produced.
- Best checkpoint path: none
- Other checkpoints produced:
  - `/workspace/run/results/train/epoch_1.pth`
  - `/workspace/run/results/train/last_checkpoint` pointing at `epoch_1.pth`
  - Two-GPU retry produced `/workspace/run/results/train_2gpu/epoch_1.pth`
  - No-validation-interval retry produced `/workspace/run/results/train_no_val/epoch_1.pth`, but MMEngine still ran final validation and hit the same failure
  - Resume validation produced `/workspace/run/results/resume_train/epoch_2.pth`

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/run/results/train/epoch_1.pth`
- Inference checkpoint used: `/workspace/run/results/train/epoch_1.pth`
- Export checkpoint used: not applicable
- Resume/retrain checkpoint used: `/workspace/run/results/train/epoch_1.pth`
- Were checkpoint paths selected through the proper resolver: yes for the direct local-docker model-skill path. BEVFusion produced the model-specific `epoch_1.pth` pattern, and all checkpoint-dependent specs used that exact checkpoint rather than the `last_checkpoint` pointer.
- Any incorrect latest-checkpoint behavior found: no. `last_checkpoint` was present but was not used as an implicit best checkpoint.

## Issues found

- Model skill issues:
  - The skill did not warn that BEVFusion 5.5 can return Docker exit code 0 while TAO records `Execution status: FAIL` after a post-evaluation SIGSEGV. This could cause a fresh-install user or wrapper to mark train/evaluate successful incorrectly.
- Config issues:
  - The requested PyTorch validation image contains the `bevfusion` CLI entrypoint but lacks `mmdet3d`, failing before action specs parse. The model skill already documents the BEVFusion-specific `nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt` requirement, so the supported actions were run with that model-required image.
  - `automl_policy=on` conflicts with the explicit instruction not to run workflow skills for this validation pass. Because `bevfusion` is AutoML-enabled and the AutoML path is an application/workflow skill, this report validates the direct model-skill action path and does not execute the AutoML wrapper.
- Dataset issues:
  - None blocking.
- Checkpoint issues:
  - No fragile latest-checkpoint handoff was observed. The unresolved issue is that train/evaluate/resume reach the checkpoint or prediction stage and then fail after post-processing.
- Docker/local execution issues:
  - The host `nvidia-smi` smoke test prints valid GPU tables but exits nonzero due infoROM warnings. Docker GPU access still worked.
  - BEVFusion 5.5 train, evaluate, two-GPU train, train with validation interval pushed past the single epoch, and resume train all failed after prediction conversion with `Signal 11 (SIGSEGV)` in `cuMemRetainAllocationHandle`.
  - TAO telemetry returned HTTP 403 in the 5.5 container; this did not block dataset conversion or inference.
- Fresh-install issues:
  - The shared TAO PyTorch validation image is not sufficient for this model because `mmdet3d` is absent; the BEVFusion-specific 5.5 container must be available.

## Fixes made

- Updated `models/bevfusion/SKILL.md` to document the post-evaluation SIGSEGV failure mode, require checking TAO status/logs instead of Docker exit code alone, and preserve exact checkpoint selection when using a checkpoint produced before the failure for downstream diagnostics.

## Remaining issues

- Train, evaluate, and resume remain blocked by the BEVFusion 5.5 post-evaluation SIGSEGV on local-docker. The model skill can now surface the failure accurately, but the underlying crash is in the TAO 5.5 runtime path.
- AutoML workflow execution was intentionally not tested because workflow skills are out of scope for this request.

## Files changed

- `models/bevfusion/SKILL.md`
- `validation-reports/bevfusion.md`

## Final status

- Partially validated. `dataset_convert` and `inference` pass; checkpoint-dependent handoff uses the exact `epoch_1.pth`; train, evaluate, and resume are blocked by the BEVFusion 5.5 SIGSEGV despite producing intermediate artifacts.
