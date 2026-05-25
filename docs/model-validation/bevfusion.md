# Model: bevfusion

## Supported actions tested

- dataset convert: pass
- train: pass
- eval: pass
- inference: pass
- resume train: pass
- export: unsupported by this model skill
- deploy: unsupported by this model skill
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- retrain: unsupported as a separate action by this model skill
- other: not applicable

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_bevfusion_train/`
- Files: `ImageSets.tar.gz`, `training.tar.gz`, `testing.tar.gz`
- Local root: `/tmp/tao-model-validation/bevfusion/data`
- Notes: KITTI-style person dataset with `training/{calib,image_2,label_2,velodyne}` and `testing/{calib,image_2,velodyne}`. `bevfusion convert` produced `kitti_person_infos_train.pkl`, `kitti_person_infos_val.pkl`, `kitti_person_infos_trainval.pkl`, and `training/velodyne_reduced`.
- Any dataset compatibility issues: the dataset is single-class `person`; the requested common `num_classes=6` was not applied because BEVFusion class count must match annotations.

## Training result

- Training completed: yes
- Best checkpoint produced: no explicit best-checkpoint artifact was produced in this one-epoch run.
- Best checkpoint path: not applicable; downstream actions used the exact epoch checkpoint.
- Other checkpoints produced:
  - `/tmp/tao-model-validation/bevfusion/results/train/epoch_1.pth`
  - `/tmp/tao-model-validation/bevfusion/results/resume/epoch_2.pth`

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/results/train/epoch_1.pth`
- Inference checkpoint used: `/workspace/results/train/epoch_1.pth`
- Export checkpoint used: not applicable; export is not supported by this model skill.
- Resume/retrain checkpoint used: `/workspace/results/train/epoch_1.pth`
- Were checkpoint paths selected through the proper resolver: yes for direct local-docker validation; the exact `epoch_1.pth` checkpoint was selected from the BEVFusion train results and verified against `last_checkpoint`.
- Any incorrect latest-checkpoint behavior found: no runtime action blindly selected latest. The skill lacked sufficient direct local-docker checkpoint handoff guidance, so explicit `epoch_N.pth` guidance was added.

## Issues found

- Model skill issues:
  - `skill_info.yaml` pointed BEVFusion at the shared TAO PyTorch 7.0 RC image, which does not include `mmdet3d`.
  - The action command used `bevfusion dataset_convert`; the working BEVFusion 5.5 CLI subtask is `bevfusion convert`.
  - Packaged train/evaluate/inference templates used a newer config surface rejected by the BEVFusion 5.5 container.
- Config issues:
  - Templates used stale data prefixes `training/lidar_reduced` and `training/images/`; conversion produced `training/velodyne_reduced` and the dataset uses `training/image_2`.
  - `dataset_convert` with `results_dir` outside `root_dir` wrote info pickles but failed during reduced point cloud generation because the 5.5 converter expects the info files under `root_dir`.
  - Empty strings for `train.pretrained_checkpoint` and `train.resume_training_checkpoint_path` were interpreted as checkpoint paths; 5.5 expects YAML null for "no checkpoint".
  - Train/evaluate/inference templates needed non-running action stubs because the 5.5 runners materialize the full experiment config.
- Dataset issues:
  - No issue after using the extracted KITTI layout and converted info files.
- Checkpoint issues:
  - No unsupported checkpoint filename pattern was found. BEVFusion produced `epoch_1.pth` and `last_checkpoint` containing `/workspace/results/train/epoch_1.pth`.
- Docker/local execution issues:
  - The shared/default 7.0 RC container fails before BEVFusion actions parse due to missing `mmdet3d`.
  - The 5.5 container prints a benign telemetry 403 after each action and an Open3D DISPLAY warning during inference, but actions complete successfully.
- Fresh-install issues:
  - `tao_sdk` is not installed in this environment, so validation used the real model CLI through Docker.
  - AutoML was requested as `on`, but workflow skills were explicitly prohibited for this validation, so no AutoML workflow was run.

## Fixes made

- Set BEVFusion's model skill container image to `nvcr.io/nvidia/tao/tao-toolkit:5.5.0-pyt`.
- Changed the dataset conversion command to `bevfusion convert -e {config_path}`.
- Updated BEVFusion instructions for the 5.5 container, local-docker conversion behavior, data prefixes, config surface, null checkpoint values, and exact checkpoint handoff.
- Updated train/evaluate/inference templates and schemas to remove invalid 5.5 keys.
- Updated data prefix defaults to `training/velodyne_reduced` and `training/image_2`.
- Added required train/evaluate/inference stubs to checkpoint-dependent action templates and schemas.

## Remaining issues

- BEVFusion remains pinned to the 5.5 TAO container until a newer shared image includes BEVFusion's `mmdet3d` stack.
- No explicit best checkpoint is produced by the one-epoch validation run; downstream actions use the exact produced epoch checkpoint.
- Export/deploy/prune/quantize/retrain are not declared actions for this model skill.

## Files changed

- `models/bevfusion/SKILL.md`
- `models/bevfusion/references/skill_info.yaml`
- `models/bevfusion/references/spec_template_train.yaml`
- `models/bevfusion/references/spec_template_evaluate.yaml`
- `models/bevfusion/references/spec_template_inference.yaml`
- `models/bevfusion/schemas/train.schema.json`
- `models/bevfusion/schemas/evaluate.schema.json`
- `models/bevfusion/schemas/inference.schema.json`
- `docs/model-validation/bevfusion.md`

## Final status

Fully validated for all actions declared by the BEVFusion model skill on `local-docker` through the BEVFusion TAO container workflow.
