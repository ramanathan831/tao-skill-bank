# Model: classification-pyt

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass after fixing deploy runtime batch-size guidance
- prune: not supported by this model skill
- retrain: not supported by this model skill
- dataset convert: not supported by this model skill
- other: resume-from-checkpoint train variant: pass
- other: quantize: pass
- other: distill: pass
- other: gen_trt_engine: pass
- other: inference on tao-deployed gen_trt_engine model: pass

## Dataset used

- Source: s3://nvcf-storage-handling/data/classification_train/images_train.tar.gz, s3://nvcf-storage-handling/data/classification_val/images_val.tar.gz, s3://nvcf-storage-handling/data/classification_test/images_test.tar.gz, and s3://nvcf-storage-handling/data/classification_train/classes.txt
- Notes: Used the real packaged classification dataset. The extracted data has 20 class directories, 398 training images, 60 validation images, and 100 test images. Specs used `num_classes: 20` because the dataset requires it.
- Any dataset compatibility issues: none. The class folders match `classes.txt`.

## Training result

- Training completed: yes
- Best checkpoint produced: yes, as the single concrete epoch checkpoint from this validation run
- Best checkpoint path: `/workspace/run/results/train/model_epoch_000.pth`
- Other checkpoints produced:
  - `/workspace/run/results/train/classifier_model_latest.pth` symlink to `model_epoch_000.pth`
  - Resume validation produced `/workspace/run/results/resume_train/model_epoch_001.pth`
  - Distill produced `/workspace/run/results/distill/model_epoch_000.pth`
  - Quantize produced `/workspace/run/results/quantize/quantized_model_torchao.pth`
  - Export produced `/workspace/run/results/export/classification_pyt.onnx`
  - Deploy produced `/workspace/run/results/gen_trt_engine/classification_pyt.engine`

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/run/results/train/model_epoch_000.pth`
- Inference checkpoint used: `/workspace/run/results/train/model_epoch_000.pth`
- Export checkpoint used: `/workspace/run/results/train/model_epoch_000.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000.pth`
- Quantize model used: `/workspace/run/results/train/model_epoch_000.pth`
- Distill teacher checkpoint used: `/workspace/run/results/train/model_epoch_000.pth`
- Deploy artifacts used: `/workspace/run/results/export/classification_pyt.onnx` for `gen_trt_engine`, then `/workspace/run/results/gen_trt_engine/classification_pyt.engine` for TensorRT evaluate/inference
- Were checkpoint paths selected through the proper resolver: yes for the direct local-docker model-skill path. The packaged skill documents using the concrete `model_epoch_...pth` checkpoint when not using the SDK resolver, and all checkpoint-dependent specs used that exact file.
- Any incorrect latest-checkpoint behavior found: no. `classifier_model_latest.pth` was present but was not used for checkpoint-dependent actions.

## Issues found

- Model skill issues:
  - The deploy skill said to set `evaluate.batch_size: 1` and `inference.batch_size: 1` for static batch-1 ONNX exports, but the validation-fixes deploy image still used `dataset.batch_size: 8` unless it was overridden too. The first deploy evaluate attempt failed with a TensorRT static dimension mismatch: set dimensions `[8,3,224,224]`, expected `[1,3,224,224]`.
- Config issues:
  - The common `num_classes=6` does not fit this dataset; the model requires `num_classes: 20` to match the class folders and `classes.txt`.
  - `automl_policy=on` conflicts with the explicit instruction not to run workflow skills for this validation pass. Because `classification-pyt` is AutoML-enabled and the AutoML path is an application/workflow skill, this report validates the direct model-skill action path and does not execute the AutoML wrapper.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - The host `nvidia-smi` smoke test prints valid GPU tables but exits nonzero due infoROM warnings. Docker GPU access still worked for all Classification PyT actions.
  - TAO Deploy telemetry returned a nonblocking warning after engine generation.
- Fresh-install issues:
  - Deploy evaluate/inference specs need `dataset.batch_size: 1` in addition to action-level batch size when using a static batch-1 ONNX/engine.

## Fixes made

- Updated `models/classification-pyt/deploy/SKILL.md` so the deploy override guidance and engine-profile pitfall require `dataset.batch_size: 1` alongside `evaluate.batch_size: 1` and `inference.batch_size: 1`.
- Reran deploy evaluate after applying the corrected scratch spec; it passed with top-1 accuracy `0.06`.

## Remaining issues

- AutoML workflow execution was intentionally not tested because workflow skills are out of scope for this request.

## Files changed

- `models/classification-pyt/deploy/SKILL.md`
- `validation-reports/classification-pyt.md`

## Final status

- Fully validated for all supported `classification-pyt` model-skill actions on local-docker with the validation-fixes PyTorch and Deploy images, after the deploy batch-size guidance fix.
