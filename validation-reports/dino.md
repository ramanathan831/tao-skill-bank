# Model: dino

## Supported actions tested
- train: pass
- resume / retrain from checkpoint: pass
- eval: pass
- inference: pass
- export: pass
- quantize: pass
- distill: pass
- deploy gen_trt_engine: pass
- eval on tao-deployed gen_trt_engine model: pass
- inference on tao-deployed gen_trt_engine model: pass
- dataset convert: unsupported by this model skill; the skill documents the current DINO convert schema failure and does not advertise it as a supported action
- prune: unsupported by this model skill
- AutoML: not run; AutoML is enabled for DINO, but this validation is constrained to model skills only and does not run AutoML/workflow skills

## Dataset used
- Source: `s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_no_convert/`
- Source: `s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_no_convert/`
- Notes: Used the COCO-format DINO smoke dataset with `images.tar.gz`, `annotations.json`, and `label_map.txt`. Archives extract with a top-level `images/` folder, so direct Docker specs used `/workspace/run/data/<split>/images/images`.
- Any dataset compatibility issues: The train and val COCO category name ordering differs for some category ids, while `label_map.txt` is consistent. This did not block smoke validation, but metrics are not meaningful. The dataset max category id is 4; `dataset.num_classes` was set to 6 per the common validation config.

## Training result
- Training completed: yes
- Best checkpoint produced: no dedicated best-checkpoint artifact was produced by this one-epoch smoke run
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/model_epoch_000_step_00050.pth`
  - `/workspace/run/results/train/dino_model_latest.pth -> model_epoch_000_step_00050.pth`
  - `/workspace/run/results/resume_train/model_epoch_001_step_00100.pth`
  - `/workspace/run/results/teacher_train/model_epoch_000_step_00050.pth`
  - `/workspace/run/results/distill/model_epoch_000_step_00050.pth`
- Training metrics:
  - `val_loss: 30.006860733032227`
  - `val_mAP: 0.0`
  - `val_mAP50: 0.0`
  - `train_loss: 49.04970169067383`
- Resume metrics:
  - `val_loss: 31.601879119873047`
  - `train_loss: 43.7039794921875`
- Distill metrics:
  - `val_loss: 56.289268493652344`
  - `train_loss: 51.7139892578125`

## Checkpoint/action verification
- Eval checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00050.pth`
- Inference checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00050.pth`
- Export checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00050.pth`
- Quantize model path used: `/workspace/run/results/train/model_epoch_000_step_00050.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00050.pth`
- Distill teacher checkpoint used: `/workspace/run/results/teacher_train/model_epoch_000_step_00050.pth`
- Deploy gen_trt_engine model used: `/workspace/run/results/export/dino.onnx`
- Deploy evaluate/inference engine used: `/workspace/run/results/deploy/dino.engine`
- Were checkpoint paths selected through the proper resolver: yes. The model skill maps dependent actions through `parent_model`, and the scratch specs were verified to use exact epoch checkpoints rather than `dino_model_latest.pth`.
- Any incorrect latest-checkpoint behavior found: no. Latest symlinks existed for train, resume, teacher train, and distill, but dependent actions used concrete epoch checkpoint files.

## Issues found
- Model skill issues:
  - The deploy skill did not document that TensorRT evaluate expects at least 100 selected detections per image. A reduced smoke setting with `model.num_select: 50` produced predictions, then failed in COCO metric loading with `IndexError: index 50 is out of bounds`.
  - The first distill attempt with a ResNet teacher checkpoint failed because DINO distill only supports FAN teachers. This was expected from the skill text; no compatible FAN checkpoint was present in S3, so a one-epoch `fan_tiny` teacher was trained with the DINO model skill and distill then passed.
- Config issues:
  - Deploy evaluate needs `model.num_select >= 100` even when train/export use a smaller postprocess top-k, as long as the value is within `model.num_queries * dataset.num_classes`.
- Dataset issues:
  - Category id/name ordering differs between train and val annotations; smoke validation passed, but metrics should not be treated as quality signals.
- Checkpoint issues:
  - No fragile latest-checkpoint behavior found.
- Docker/local execution issues:
  - Host-UID Docker execution required writable `HOME`, `MPLCONFIGDIR`, `TORCHINDUCTOR_CACHE_DIR`, and `XDG_CACHE_HOME` under the mounted scratch directory.
- Fresh-install issues:
  - Deploy commands succeed with zero exit code and write expected artifacts, but deploy `status.json` files finish with `status: RUNNING` after success messages.
  - Deploy telemetry reports `'str' object has no attribute 'decode'` after successful commands. This did not affect action success.

## Fixes made
- Updated `models/dino/deploy/SKILL.md` to document the TensorRT evaluate `model.num_select >= 100` requirement and the `IndexError` failure mode.
- Updated `models/dino/SKILL.md` to point deploy TensorRT evaluation users to the deploy skill note.

## Remaining issues
- Deploy `status.json` final status can remain `RUNNING` even when the command logs success and exits zero.
- No dedicated best-checkpoint artifact was produced during the one-epoch smoke run.
- DINO AutoML was not run because this pass is restricted to model skills.

## Files changed
- `models/dino/SKILL.md`
- `models/dino/deploy/SKILL.md`
- `validation-reports/dino.md`

## Final status
- Fully validated
