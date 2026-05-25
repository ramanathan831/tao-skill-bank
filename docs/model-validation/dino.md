# Model: dino

Validated on 2026-05-25 with `platform=local-docker`, `image=default`.
The PyT image resolved to
`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch`.
The original validation pass used direct model training because AutoML routing
was explicitly out of scope. After the default AutoML request, train was rerun
through `AutoMLRunner` + `DockerSDK` with a two-trial Bayesian search.

## Supported actions tested

- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval/evaluate: pass
- inference: pass
- export: pass
- quantize: pass
- distill: pass
- retrain/resume: pass
- deploy gen_trt_engine: pass through `models/dino/deploy`
- deploy inference on TensorRT engine: pass
- deploy evaluate on TensorRT engine: pass
- parent gen_trt_engine: fail; removed from parent model-skill action metadata because the PyT CLI does not support it
- dataset convert: fail; SDK/container schema bug before spec load, not advertised as a model-skill action
- prune: unsupported

## Dataset used

- Source: `s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_no_convert/`
- Source: `s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_no_convert/`
- Files: `images.tar.gz`, `annotations.json`, `label_map.txt`
- Notes: COCO-format object-detection subset with category ids 1-4. The direct validation used the packaged S3 subset. The AutoML rerun used a smaller derived subset from the same S3 data: 12 train images and 8 validation images.
- Dataset convert attempt: used `s3://nvcf-storage-handling/data/purpose_built_models_bevfusion_train/training.tar.gz` as a compatible KITTI-style source, but `dino convert` failed during SDK schema initialization before reading the spec.
- Any dataset compatibility issues: none for train/evaluate/inference/export/quantize/distill/deploy.

## Training result

- Training completed: yes
- AutoML completed: yes, 2/2 Bayesian recommendations succeeded
- Best checkpoint produced: no separate best-checkpoint artifact was produced in the one-epoch smoke runs
- Best checkpoint path:
  `/tmp/tao-automl-validation/dino/results/6dc856bc-4a80-480f-907d-2bfaa3d62270/results_dir/train/model_epoch_000_step_00006.pth`
- AutoML best result: rec 0, job
  `6dc856bc-4a80-480f-907d-2bfaa3d62270`, `mAP50=0.0`,
  `train.optim.lr=7.715930445345001e-05`,
  `train.optim.lr_backbone=2.504662332427296e-05`
- Other AutoML result: rec 1, job
  `e64fe02e-8ce9-4311-9994-6a0783153799`, `mAP50=0.0`,
  checkpoint
  `/tmp/tao-automl-validation/dino/results/e64fe02e-8ce9-4311-9994-6a0783153799/results_dir/train/model_epoch_000_step_00006.pth`
- Other checkpoints produced by the direct action run:
  `/tmp/tao-model-validation/dino/results/train/train/model_epoch_000_step_00025.pth`,
  `dino_model_latest.pth -> model_epoch_000_step_00025.pth`,
  `/tmp/tao-model-validation/dino/results/resume/train/model_epoch_001_step_00050.pth`,
  `/tmp/tao-model-validation/dino/results/teacher_train/train/model_epoch_000_step_00025.pth`, and
  `/tmp/tao-model-validation/dino/results/distill/model_epoch_000_step_00025.pth`

## Checkpoint/action verification

- Eval checkpoint used: `/tao-workspace/results/train/train/model_epoch_000_step_00025.pth`
- Inference checkpoint used: `/tao-workspace/results/train/train/model_epoch_000_step_00025.pth`
- Export checkpoint used: `/tao-workspace/results/train/train/model_epoch_000_step_00025.pth`
- Quantize model path used: `/tao-workspace/results/train/train/model_epoch_000_step_00025.pth`
- Resume/retrain checkpoint used: `/tao-workspace/results/train/train/model_epoch_000_step_00025.pth`
- Distill teacher checkpoint used: `/tao-workspace/results/teacher_train/train/model_epoch_000_step_00025.pth`
- Deploy engine used: `/tao-workspace/results/deploy_gen_trt_engine/dino.engine`
- AutoML best checkpoint used: the best trial checkpoint above, selected by AutoML state. Both trials tied at `mAP50=0.0`, so rec 0 remained best.
- Were checkpoint paths selected through the proper resolver: yes for the fixed model metadata (`parent_model` mappings now exist for evaluate/export/inference/quantize/distill); local validation selected exact epoch/step artifacts directly from SDK output folders.
- Any incorrect latest-checkpoint behavior found: the docs over-emphasized `dino_model_latest.pth`; updated guidance requires exact/best resolver selection and reserves latest only for explicit latest requests.

## Issues found

- Model skill issues:
  - Parent `dino gen_trt_engine` was listed in model manifests/metadata, but the PyT CLI rejects it as an invalid subtask.
  - Parent `skill_info.yaml` had empty `inputs` for distill, quantize, export, and inference.
  - Parent `skill_info.yaml` had no checkpoint-producing/consuming `spec_params` for evaluate/export/inference/quantize/distill.
- Config issues:
  - `spec_template_distill.yaml` omitted the required `distill` action block, causing `TypeError: 'NoneType' object is not subscriptable`.
  - Distill requires a FAN-family teacher; using the default ResNet teacher config fails with `Teacher arch resnet_50 not supported`.
  - Deploy templates defaulted to `fan_small`/`num_select: 100`, which does not match the parent DINO export default of `resnet_50`/`num_select: 300`.
- Dataset issues:
  - No issue for the COCO train/validation dataset.
  - `dino convert` is unusable in the tested container because SDK schema construction fails with `Incompatible value 'None' for field of type 'str'`.
- Checkpoint issues:
  - No action used the latest symlink blindly after validation.
  - No separate best checkpoint was produced by the one-epoch smoke run.
- Docker/local execution issues:
  - Deploy actions run in the TAO Deploy image, not the PyT image.
  - Deploy telemetry emitted a warning after successful commands; it did not affect action status.
- Fresh-install issues:
  - Fresh metadata would have exposed an invalid parent `gen_trt_engine` action and omitted required action inputs for several checkpoint-dependent actions.

## Fixes made

- Removed parent `gen_trt_engine` from DINO parent model manifests and `references/skill_info.yaml`.
- Added required action inputs for distill, quantize, export, and inference.
- Added parent-model `spec_params` for evaluate/export/inference/quantize/distill checkpoint handoff.
- Added the required `distill` block, FAN teacher defaults, and output bindings to `spec_template_distill.yaml`.
- Updated deploy templates to match parent export defaults for `model.backbone` and `model.num_select`.
- Updated DINO docs to require exact/best checkpoint resolver behavior and to route TensorRT actions through `models/dino/deploy`.
- Documented the SDK/container blocker for `dino convert`.
- Reran train through AutoML with Bayesian search, `automl_max_recommendations=2`, metric `mAP50`, and a custom DINO metric extractor as required by the model skill.

## Remaining issues

- `dino convert` remains blocked by the SDK/container dataclass schema bug and is not advertised as a supported model-skill action.
- Parent PyT `dino gen_trt_engine` remains unsupported by the container; TensorRT is validated through the deploy sub-skill.
- A longer run is needed to verify best-checkpoint selection when the SDK produces multiple scored checkpoints.
- The tiny one-epoch AutoML smoke run produced `mAP50=0.0` for both recommendations; this validates wiring, not model quality.

## Files changed

- `models/dino/SKILL.md`
- `models/dino/references/skill_info.yaml`
- `models/dino/references/spec_template_distill.yaml`
- `models/dino/references/spec_template_deploy_gen_trt_engine.yaml`
- `models/dino/references/spec_template_deploy_inference.yaml`
- `models/dino/references/spec_template_deploy_evaluate.yaml`
- `models/dino/schemas/manifest.json`
- `models/schemas.manifest.json`
- `docs/model-validation/dino.md`

## Final status

Partially validated. All advertised DINO parent model-skill actions now pass except unsupported actions that were removed or documented. Default AutoML train routing passed with two Bayesian recommendations. Deploy TensorRT generation, deploy inference, and deploy evaluation pass through the DINO deploy sub-skill. Dataset conversion is blocked by an SDK/container schema issue outside the model-skill layer.
