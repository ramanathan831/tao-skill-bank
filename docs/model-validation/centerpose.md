# Model: centerpose

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- gen_trt_engine: pass
- deploy evaluate: pass
- deploy inference: pass
- resume train: pass
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- retrain: unsupported as a separate action by this model skill
- dataset convert: unsupported by this model skill
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- other: not applicable

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_centerpose_train/train.tar.gz`, `s3://nvcf-storage-handling/data/purpose_built_models_centerpose_val/val.tar.gz`, `s3://nvcf-storage-handling/data/purpose_built_models_centerpose_test/test.tar.gz`
- Local root: `/tmp/tao-model-validation/centerpose/data/extracted`
- Notes: archives were extracted before local-docker execution. The data is a single-category `bike` CenterPose dataset with image/JSON pairs under `train/`, `val/`, and `test/`.
- Any dataset compatibility issues: the dataset has one class, so `dataset.num_classes=1` and `dataset.category=bike` were used instead of the common `num_classes=6`.

## Training result

- Training completed: yes
- Best checkpoint produced: no explicit best-checkpoint artifact was produced in this one-epoch validation run.
- Best checkpoint path: not applicable; downstream actions used the exact epoch/step checkpoint.
- Other checkpoints produced:
  - `/tmp/tao-model-validation/centerpose/results/train/model_epoch_000_step_00008.pth`
  - `/tmp/tao-model-validation/centerpose/results/train/centerpose_model_latest.pth` symlink to `model_epoch_000_step_00008.pth`
  - `/tmp/tao-model-validation/centerpose/results/resume/model_epoch_001_step_00016.pth`

## AutoML default-path rerun

- Training mode before rerun: normal/direct TAO model training. This happened because the earlier validation explicitly prohibited workflow/AutoML routing.
- Secrets source: `~/.tao/secrets.env`, sourced without printing values. `ACCESS_KEY`, `SECRET_KEY`, `NGC_KEY`, and `HF_TOKEN` were passed through the local run environment.
- AutoML route: `AutoMLRunner` with `skill_dir=/localhome/local-rarunachalam/tao-skills-external/models/centerpose`, `platform=local-docker`, default PyT image, `algorithm=bayesian`, `metric=val_3DIoU`, `direction=maximize`, and `automl_max_recommendations=2`.
- Search parameters: `train.optim.lr` and `train.optim.lr_decay`; the minimal validation ranges were `lr=0.00001..0.0001` and `lr_decay=0.05..0.2`.
- Result: pass. Two real Docker TAO child jobs completed and emitted `val_3DIoU` in `status.json`.
- Rec 0: job `1c78e7bc-d930-4929-89b4-eef252fff0d2`, `train.optim.lr=0.00004216849595128809`, `train.optim.lr_decay=0.12466171513728731`, `val_3DIoU=0.0`; selected as best by the runner.
- Rec 1: job `099b8af9-2031-4897-908a-bb8b98b3d3f9`, `train.optim.lr=0.00009266435625398663`, `train.optim.lr_decay=0.18414348945579645`, `val_3DIoU=0.0`.
- AutoML checkpoints produced:
  - `/tmp/tao-automl-validation/centerpose/results/automl_train/1c78e7bc-d930-4929-89b4-eef252fff0d2/results_dir/train/model_epoch_000_step_00004.pth`
  - `/tmp/tao-automl-validation/centerpose/results/automl_train/099b8af9-2031-4897-908a-bb8b98b3d3f9/results_dir/train/model_epoch_000_step_00004.pth`
- Generated specs contained dataset paths and hyperparameters only; no credentials were written to the generated specs.

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/results/train/model_epoch_000_step_00008.pth`
- Inference checkpoint used: `/workspace/results/train/model_epoch_000_step_00008.pth`
- Export checkpoint used: `/workspace/results/train/model_epoch_000_step_00008.pth`
- Resume/retrain checkpoint used: `/workspace/results/train/model_epoch_000_step_00008.pth`
- Deploy engine input used: `/workspace/results/export/centerpose.onnx`
- Deploy evaluate/inference engine used: `/workspace/results/gen_trt_engine_626/centerpose.engine`
- Were checkpoint paths selected through the proper resolver: yes for direct local-docker validation; the exact model-specific epoch/step checkpoint was selected instead of the latest symlink.
- Any incorrect latest-checkpoint behavior found: no runtime action blindly selected latest. The skill docs needed explicit CenterPose checkpoint handoff guidance.

## Issues found

- Model skill issues:
  - The prose examples pointed users at S3 tarballs, but local-docker CenterPose actions consume extracted folders.
  - Parent `gen_trt_engine` metadata did not identify the deploy container or ONNX/engine artifacts.
  - Deploy metadata used the generic 7.0 RC deploy alias, which built an engine but failed deploy evaluate/inference postprocessing.
- Config issues:
  - `gen_trt_engine.tensorrt.calibration.cal_image_dir` was a scalar in deploy templates/schema, but TAO Deploy expects a list.
  - Deploy `dataset.test_data` and `dataset.inference_data` metadata were typed as files, but the working CenterPose deploy actions consume folders.
- Dataset issues:
  - No issue after extracting the tarballs and using `category=bike`.
- Checkpoint issues:
  - No unsupported checkpoint pattern was found. CenterPose produced `model_epoch_000_step_00008.pth` plus a latest symlink.
- Docker/local execution issues:
  - `nvcr.io/nvstaging/tao/tao-toolkit-deploy:7.0.0-rc-171-multiarch` failed deploy evaluate/inference with `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
  - `nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy` passed engine generation, deploy evaluate, and deploy inference.
- Fresh-install issues:
  - none after the AutoML default-path rerun used the local checkout with the Docker/runtime dependencies installed.

## Fixes made

- Documented extracted-folder inputs for CenterPose local-docker runs.
- Added exact checkpoint handoff guidance for `model_epoch_*_step_*.pth` checkpoints and latest symlinks.
- Pinned CenterPose deploy metadata to `nvcr.io/nvidia/tao/tao-toolkit:6.26.3-deploy`.
- Added parent `gen_trt_engine` action-level deploy image and ONNX/engine inputs/outputs.
- Changed deploy `dataset.test_data` and `dataset.inference_data` metadata from file to folder.
- Changed deploy and parent gen_trt calibration image-dir defaults to lists.
- Reran the train path through AutoML with a two-recommendation Bayesian configuration.

## Remaining issues

- The 7.0 RC deploy alias remains incompatible with CenterPose deploy evaluate/inference for this exported ONNX; the model skill now routes CenterPose deploy actions to 6.26.3-deploy.
- No explicit best checkpoint was produced by the one-epoch validation run.
- Prune, quantize, retrain, and dataset convert are not declared CenterPose actions.

## Files changed

- `models/centerpose/SKILL.md`
- `models/centerpose/deploy/SKILL.md`
- `models/centerpose/deploy/skill_info.yaml`
- `models/centerpose/references/skill_info.yaml`
- `models/centerpose/references/spec_template_deploy_gen_trt_engine.yaml`
- `models/centerpose/references/spec_template_gen_trt_engine.yaml`
- `models/centerpose/schemas/gen_trt_engine.schema.json`
- `docs/model-validation/centerpose.md`

## Final status

Fully validated for all actions declared by the CenterPose model skill and CenterPose deploy model skill on `local-docker`, plus the AutoML default train route with two Bayesian recommendations.
