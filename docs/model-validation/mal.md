Model: mal

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: unsupported/not advertised by the MAL model skill
- deploy: unsupported/not advertised by the MAL model skill
- prune: unsupported/not advertised by the MAL model skill
- quantize: unsupported/not advertised by the MAL model skill
- retrain/resume: pass through train.resume_training_checkpoint_path
- dataset convert: unsupported/not advertised by the MAL model skill
- other: no additional MAL CLI actions beyond train, evaluate, inference, and default_specs

Dataset used:
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/images.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/annotations.json
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/images.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/annotations.json
- Notes: The train split contains 100 COCO-style images and 768 annotations. The validation split contains 5 images and 51 annotations. Validation used the COCO image directories extracted under the local-docker data mount so the annotation file_name entries resolved correctly.
- Any dataset compatibility issues: s3://nvcf-storage-handling/data/mvtec_ad_mgcn/ was inspected but is a dataset.csv plus image archive layout and is not directly compatible with MAL's COCO-style annotation requirement without conversion.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; the one-epoch run produced an epoch/step checkpoint.
- Best checkpoint path: /tmp/tao-model-validation/mal/results/train/train/model_epoch_000_step_00766.pth
- Other checkpoints produced: /tmp/tao-model-validation/mal/results/resume/train/model_epoch_001_step_01532.pth; mal_model_latest.pth symlinks were produced by the runtime but were not used for checkpoint-dependent actions.

Checkpoint/action verification:
- Eval checkpoint used: /tmp/tao-model-validation/mal/results/train/train/model_epoch_000_step_00766.pth
- Inference checkpoint used: /tmp/tao-model-validation/mal/results/train/train/model_epoch_000_step_00766.pth
- Export checkpoint used: unsupported/not advertised
- Resume/retrain checkpoint used: /tmp/tao-model-validation/mal/results/train/train/model_epoch_000_step_00766.pth
- Deploy checkpoint path: unsupported/not advertised
- Were checkpoint paths selected through the proper resolver: yes in model metadata after the fix; direct local-docker validation used explicit exact paths to verify the resolver contract.
- Any incorrect latest-checkpoint behavior found: no action selected mal_model_latest.pth. The model metadata was missing resolver mappings before the fix, which made fresh-install parent handoff fragile.

Issues found:
- Model skill issues:
  - spec_params were empty in skill_info.yaml, so evaluate and inference did not declare parent checkpoint resolver mappings for fresh installs.
  - inference.label_dump_path was documented in SKILL.md but not declared as an output in skill_info.yaml.
- Config issues:
  - Direct local-docker specs must point dataset train_img_dir, val_img_dir, and inference.img_dir to the extracted image directories whose relative filenames match the COCO annotations.
- Dataset issues:
  - The available MVTec anomaly dataset is not directly usable for MAL because it does not provide COCO-style annotation JSON.
- Checkpoint issues:
  - No runtime latest-checkpoint misuse was found. The metadata gap could cause generated workflows to omit the model-specific resolver and require users to hand-pick checkpoint files.
- Docker/local execution issues:
  - The container emits duplicate TAO logging and timm deprecation warnings, but they do not affect execution.
- Fresh-install issues:
  - Fresh installs needed the model-level spec_params mappings so parent train results can resolve to the selected checkpoint and MAL inference can create its result JSON path.

Fixes made:
- Added evaluate.checkpoint and inference.checkpoint parent_model resolver mappings to models/mal/references/skill_info.yaml.
- Added inference.label_dump_path as an output and mapped it to create_inference_result_file_mal.
- Documented MAL's COCO-style dataset expectation and exact checkpoint selection guidance in models/mal/SKILL.md.

Remaining issues:
- None for the advertised MAL model actions after the metadata fixes.

Files changed:
- models/mal/SKILL.md
- models/mal/references/skill_info.yaml
- docs/model-validation/mal.md

Final status:
- Fully validated for supported parent actions on local-docker with image=default.
