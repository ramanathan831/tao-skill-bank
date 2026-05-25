# Model: classification-pyt

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- gen_trt_engine: pass via TAO Deploy
- deploy evaluate: pass after template batch-size fix
- deploy inference: pass
- quantize: pass
- distill: pass after distill LR-policy fix
- resume train: pass
- prune: unsupported by this model skill
- retrain: unsupported as a separate action by this model skill; resume train was tested
- dataset convert: unsupported by this model skill
- other: not applicable

## Dataset used

- Source: `s3://nvcf-storage-handling/data/classification_train/`, `s3://nvcf-storage-handling/data/classification_val/`, `s3://nvcf-storage-handling/data/classification_test/`
- Local root: `/tmp/tao-model-validation/classification-pyt/data/extracted`
- Notes: `images_train.tar.gz`, `images_val.tar.gz`, and `images_test.tar.gz` were extracted before local-docker execution. The extracted folders contain 20 class subdirectories and the S3 `classes.txt` file was mounted as `dataset.classes_file`.
- Any dataset compatibility issues: the dataset has 20 classes, so `dataset.num_classes=20` was used instead of the common `num_classes=6`.

## Training result

- Training completed: yes
- Best checkpoint produced: no explicit best-checkpoint artifact was produced in this one-epoch validation run.
- Best checkpoint path: not applicable; downstream actions used the exact epoch checkpoint.
- Other checkpoints produced:
  - `/tmp/tao-model-validation/classification-pyt/results/train/model_epoch_000.pth`
  - `/tmp/tao-model-validation/classification-pyt/results/train/classifier_model_latest.pth` symlink to `model_epoch_000.pth`
  - `/tmp/tao-model-validation/classification-pyt/results/resume/model_epoch_001.pth`
  - `/tmp/tao-model-validation/classification-pyt/results/distill/model_epoch_000.pth`

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/results/train/model_epoch_000.pth`
- Inference checkpoint used: `/workspace/results/train/model_epoch_000.pth`
- Export checkpoint used: `/workspace/results/train/model_epoch_000.pth`
- Quantize checkpoint used: `/workspace/results/train/model_epoch_000.pth`
- Distill teacher checkpoint used: `/workspace/results/train/model_epoch_000.pth`
- Resume/retrain checkpoint used: `/workspace/results/train/model_epoch_000.pth`
- Deploy engine input used: `/workspace/results/export/classification-pyt.onnx`
- Deploy evaluate/inference engine used: `/workspace/results/gen_trt_engine/classification-pyt.engine`
- Were checkpoint paths selected through the proper resolver: yes for direct local-docker validation; the exact model-specific epoch checkpoint was selected instead of the latest symlink.
- Any incorrect latest-checkpoint behavior found: no runtime action blindly selected latest. The skill docs needed explicit Classification PyT checkpoint handoff guidance.

## Issues found

- Model skill issues:
  - The prose examples pointed local-docker users at S3 tarballs, but Classification PyT action metadata expects extracted image folders.
  - Parent `gen_trt_engine` metadata inherited the PyT container and had no ONNX/engine artifact inputs, while the command is available only in TAO Deploy.
  - The distill template inherited `train.optim.policy: linear`, which crashes the 7.0 distiller with `UnboundLocalError: interval`.
- Config issues:
  - Deploy evaluate defaulted to `evaluate.batch_size: 8`, but the exported ONNX is static batch 1 and the generated TensorRT engine rejects batch 8.
- Dataset issues:
  - No issue after extracting the tarballs and using 20 classes.
- Checkpoint issues:
  - No unsupported checkpoint pattern was found. Classification PyT produced `model_epoch_000.pth` plus a latest symlink.
- Docker/local execution issues:
  - TAO Deploy 7.0 RC actions passed after using batch 1 for evaluate. The deploy entrypoint logged a telemetry warning after successful commands.
- Fresh-install issues:
  - AutoML was requested as `on`, but workflow skills were explicitly prohibited for this validation, so no AutoML workflow was run.

## Fixes made

- Documented extracted-folder inputs for Classification PyT local-docker runs.
- Added exact checkpoint handoff guidance for `model_epoch_*.pth` checkpoints and latest symlinks.
- Added parent `gen_trt_engine` action-level deploy image and ONNX/engine inputs/outputs.
- Changed deploy evaluate default batch size from 8 to 1.
- Changed distill template/schema default `train.optim.policy` from `linear` to `step`.
- Updated deploy notes to call out batch-1 runtime requirements for both TensorRT inference and evaluation.

## Remaining issues

- No explicit best checkpoint was produced by the one-epoch validation run.
- TAO Deploy logs a non-fatal telemetry warning after successful deploy commands.
- AutoML routing was not exercised because workflow skills were disallowed by the validation request.
- Prune, dataset convert, and a separate retrain action are not declared Classification PyT actions.

## Files changed

- `models/classification-pyt/SKILL.md`
- `models/classification-pyt/deploy/SKILL.md`
- `models/classification-pyt/deploy/skill_info.yaml`
- `models/classification-pyt/references/skill_info.yaml`
- `models/classification-pyt/references/spec_template_deploy_evaluate.yaml`
- `models/classification-pyt/references/spec_template_distill.yaml`
- `models/classification-pyt/schemas/distill.schema.json`
- `docs/model-validation/classification-pyt.md`

## Final status

Fully validated for all actions declared by the Classification PyT model skill and Classification PyT deploy model skill on `local-docker`.
