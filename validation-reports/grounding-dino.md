Model: grounding-dino

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass (`gen_trt_engine`, TensorRT `evaluate`, TensorRT `inference`)
- prune: unsupported by this model skill
- quantize: pass
- retrain: pass via resume training
- dataset convert: unsupported by this model skill
- other: AutoML policy noted as enabled in metadata, but not executed because workflow skills were explicitly out of scope

Dataset used:
- Source: `s3://nvcf-storage-handling/data/object_detection_grounding_dino_train/`
- Source: `s3://nvcf-storage-handling/data/object_detection_grounding_dino_val/`
- Notes: Used real S3 Grounding DINO object detection data. Training used a 64-image ODVG subset with labels `head`, `helmet`, and `person`; validation/evaluation/inference used a 50-image COCO subset derived from the S3 validation split.
- Any dataset compatibility issues: None. Validation category IDs were already 0-based and compatible with `eval_class_ids: [0, 1, 2]`.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-checkpoint artifact was emitted for this 1-epoch run
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/model_epoch_000_step_00064.pth`
  - `/workspace/run/results/train/gdino_model_latest.pth -> model_epoch_000_step_00064.pth`
  - `/workspace/run/results/resume_train/model_epoch_001_step_00128.pth`
  - `/workspace/run/results/resume_train/gdino_model_latest.pth -> model_epoch_001_step_00128.pth`
- Train KPIs: `val_mAP=8.040580381672016e-05`, `val_mAP50=0.0004907120968314948`, `val_loss=310.7561340332031`, `train_loss=3603.2734375`
- Resume KPIs: `val_mAP=5.769239071109526e-05`, `val_mAP50=0.0003359909846030228`, `val_loss=1136.4202880859375`, `train_loss=319.0074157714844`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00064.pth`
- Inference checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00064.pth`
- Export checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00064.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00064.pth`
- Quantize checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00064.pth`
- Deploy engine source: `/workspace/run/results/export_default/grounding_dino.onnx`
- Deploy engine used: `/workspace/run/results/deploy_gen/grounding-dino.engine`
- Were checkpoint paths selected through the proper resolver: yes. The model skill maps checkpoint fields through `parent_model` / `resume_model`; direct Docker validation pinned the exact resolved epoch-step checkpoint rather than a latest symlink.
- Any incorrect latest-checkpoint behavior found: No. Latest symlinks existed, but eval, inference, export, quantize, and resume used exact epoch-step checkpoint paths.

Issues found:
- Model skill issues:
  - The skill documented the default Grounding DINO export size, but did not explicitly warn that very small smoke-test export dimensions can trigger a PyTorch ONNX shape-inference assertion in the contrastive text head.
  - Deploy carry-forward guidance mentioned query count, but not `num_select` and `max_text_len`, both of which are part of the validated shape contract.
- Config issues:
  - Export with a manually reduced `128x128`, `batch_size: 1` smoke config failed in `torch.onnx.export` with `minus_one_pos != -1 INTERNAL ASSERT FAILED` during ONNX shape inference.
  - Export with the packaged template defaults `960x544`, opset 17, and `batch_size: -1` passed.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - Deploy status files end with `RUNNING` records that say the action finished successfully rather than a terminal `PASS` status.
  - Deploy commands emitted telemetry warnings after successful completion: `Telemetry data couldn't be sent` and `'str' object has no attribute 'decode'`.
  - Deploy evaluation/inference downloaded the BERT tokenizer anonymously because no HF token was passed into the container, intentionally avoiding secret exposure in generated artifacts.
- Fresh-install issues:
  - No compatible Grounding DINO pretrained checkpoint was found in the S3 checkpoint folder, so the validation trained from scratch using the default backbone/tokenizer downloads.

Fixes made:
- Added Grounding DINO export guidance to keep smoke-test export specs at the template `960x544` resolution and documented the PyTorch ONNX assertion seen at `128x128`.
- Updated deploy guidance to carry `num_queries`, `num_select`, and `max_text_len` forward from export into TensorRT evaluation/inference specs.

Remaining issues:
- The tiny `128x128` export configuration remains invalid for Grounding DINO; the validated path is the packaged export default resolution.
- Deploy status files still do not record a terminal `PASS` status despite successful exit codes and success messages.
- Telemetry warnings remain after successful deploy actions.

Files changed:
- `models/grounding-dino/SKILL.md`
- `models/grounding-dino/deploy/SKILL.md`
- `models/grounding-dino/deploy/skill_info.yaml`
- `validation-reports/grounding-dino.md`

Final status:
- Fully validated
