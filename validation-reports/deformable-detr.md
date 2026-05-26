# Model: deformable-detr

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- prune: not supported by this model skill
- retrain: not supported by this model skill
- dataset convert: not supported by this model skill
- other: resume-from-checkpoint train variant: pass
- other: quantize: pass
- other: gen_trt_engine: pass
- other: inference on tao-deployed gen_trt_engine model: pass

## Dataset used

- Source: s3://nvcf-storage-handling/data/object_detection_pyt_train/ and s3://nvcf-storage-handling/data/object_detection_pyt_val/
- Notes: Used the real packaged COCO-style object detection data. For runtime, staged a small real-data subset from those archives: 24 training images with 143 annotations and 12 validation/inference images with 104 annotations. Specs used `num_classes: 5` because the dataset has four foreground object categories plus background.
- Any dataset compatibility issues: the S3 train and val COCO category metadata use different ID/name ordering. Train categories are `(1, fire_extinguisher)`, `(2, cone)`, `(3, cart)`, `(4, forklift)`, while val categories are `(1, cone)`, `(2, cart)`, `(3, fire_extinguisher)`, `(4, forklift)`. All actions executed successfully with `eval_class_ids: [1, 2, 3, 4]`, but the mismatch can make class-name interpretation and metrics misleading.

## Training result

- Training completed: yes
- Best checkpoint produced: yes, as the single concrete epoch checkpoint from this validation run. No separate best-checkpoint alias was produced.
- Best checkpoint path: `/workspace/run/results/train/model_epoch_000_step_00012.pth`
- Other checkpoints produced:
  - `/workspace/run/results/train/dd_model_latest.pth` symlink to `model_epoch_000_step_00012.pth`
  - Resume validation produced `/workspace/run/results/resume_train/model_epoch_001_step_00024.pth`
  - Resume validation produced `/workspace/run/results/resume_train/dd_model_latest.pth` symlink to `model_epoch_001_step_00024.pth`
  - Quantize produced `/workspace/run/results/quantize/quantized_model_torchao.pth`
  - Export produced `/workspace/run/results/export/deformable_detr.onnx`
  - Deploy produced `/workspace/run/results/deploy_gen_trt_engine/deformable_detr.engine`

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00012.pth`
- Inference checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00012.pth`
- Export checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00012.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00012.pth`
- Quantize model used: `/workspace/run/results/train/model_epoch_000_step_00012.pth`
- Deploy artifacts used: `/workspace/run/results/export/deformable_detr.onnx` for `gen_trt_engine`, then `/workspace/run/results/deploy_gen_trt_engine/deformable_detr.engine` for TensorRT evaluate/inference
- Were checkpoint paths selected through the proper resolver: yes for the direct local-docker model-skill path. The packaged skill documents using the concrete `model_epoch_...pth` checkpoint when not using the SDK resolver, and all checkpoint-dependent specs used that exact file.
- Any incorrect latest-checkpoint behavior found: no. `dd_model_latest.pth` was present but was not used for checkpoint-dependent actions.

## Issues found

- Model skill issues:
  - None found.
- Config issues:
  - The common `num_classes=6` does not fit this dataset; Deformable DETR needed `num_classes: 5` for four foreground categories plus background.
  - `automl_policy=on` conflicts with the explicit instruction not to run workflow skills for this validation pass. Because `deformable-detr` is AutoML-enabled and the AutoML path is an application/workflow skill, this report validates the direct model-skill action path and does not execute the AutoML wrapper.
- Dataset issues:
  - The packaged train and val COCO annotation files have inconsistent category ID/name ordering, as noted above.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - The host `nvidia-smi` smoke test prints valid GPU tables but exits nonzero due infoROM warnings. Docker GPU access still worked for all Deformable DETR PyTorch and deploy actions.
  - TAO Deploy telemetry returned nonblocking warnings after TensorRT actions.
- Fresh-install issues:
  - None found when following the model skill guidance to carry dataset class metadata, export dimensions, and structural model fields into deploy specs.

## Fixes made

- No Deformable DETR model-skill code changes were needed.

## Remaining issues

- The S3 train/val category metadata mismatch remains a dataset quality issue.
- AutoML workflow execution was intentionally not tested because workflow skills are out of scope for this request.

## Files changed

- `validation-reports/deformable-detr.md`

## Final status

- Fully validated for all supported `deformable-detr` model-skill actions on local-docker with the validation-fixes PyTorch and Deploy images.
