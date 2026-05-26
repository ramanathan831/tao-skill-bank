Model: mal

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: unsupported by this model skill
- deploy: unsupported by this model skill
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- retrain: pass via resume training
- dataset convert: unsupported by this model skill
- other: AutoML policy noted as enabled in metadata, but not executed because workflow skills were explicitly out of scope

Dataset used:
- Source: `s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_train/`
- Source: `s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_val/`
- Notes: Used real S3 COCO-style segmentation data. Direct local Docker validation used extracted image folders from `images.tar.gz` so file names matched the COCO `annotations.json` entries.
- Any dataset compatibility issues: None. Train split had 49 images and 470 annotations; val split had 48 images and 382 annotations.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-checkpoint artifact was emitted for these 1-epoch/2-epoch runs
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/train/model_epoch_000_step_00242.pth`
  - `/workspace/run/results/train/train/mal_model_latest.pth -> model_epoch_000_step_00242.pth`
  - `/workspace/run/results/resume_train/train/model_epoch_001_step_00484.pth`
  - `/workspace/run/results/resume_train/train/mal_model_latest.pth -> model_epoch_001_step_00484.pth`
- Train KPIs: `mIoU=0.5574135780334473`, `mIoU_small=0.5398660898208618`, `mIoU_medium=0.5905464887619019`, `mIoU_large=0.5291963815689087`, `train_loss=0.10711286216974258`
- Resume KPIs: `mIoU=0.5559796690940857`, `mIoU_small=0.5476466417312622`, `mIoU_medium=0.5863739252090454`, `mIoU_large=0.5277162194252014`, `train_loss=0.8664335608482361`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00242.pth`
- Inference checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00242.pth`
- Export checkpoint used: n/a
- Resume/retrain checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00242.pth`
- Inference label dump path: `/workspace/run/results/inference/instances_val2017_mal.json`
- Were checkpoint paths selected through the proper resolver: yes. The model skill maps checkpoint fields through `parent_model` / `resume_model`; direct Docker validation pinned exact epoch-step artifacts rather than latest symlinks.
- Any incorrect latest-checkpoint behavior found: No. Latest symlinks existed, but eval, inference, and resume used exact epoch-step checkpoint paths.

Issues found:
- Model skill issues:
  - None.
- Config issues:
  - None.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - The container logged `Stage -1 not found. Freezing options: dict_keys([...])` warnings for the default `model.frozen_stages: [-1]`, but training/evaluation/inference completed successfully.
- Fresh-install issues:
  - None.

Fixes made:
- None.

Remaining issues:
- Non-fatal `Stage -1 not found` warnings remain with the default frozen-stage setting.

Files changed:
- `validation-reports/mal.md`

Final status:
- Fully validated
