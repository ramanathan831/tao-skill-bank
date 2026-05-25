Model: mask-grounding-dino

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass through the TAO Deploy skill
- deploy inference on tao-deployed gen_trt_engine model: pass
- deploy eval on tao-deployed gen_trt_engine model: pass
- prune: unsupported/not advertised by the Mask Grounding DINO model skill
- quantize: pass
- retrain/resume: pass through train.resume_training_checkpoint_path
- AutoML default train route: pass with Bayesian automl_max_recommendations=2
- dataset convert: unsupported/not advertised by the Mask Grounding DINO model skill
- other: parent PyTorch gen_trt_engine was advertised before this validation, but the parent CLI rejects it. TensorRT engine generation belongs to the deploy skill and the parent action was removed from model metadata.

Dataset used:
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_train/images.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_train/annotations_odvg.jsonl
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_train/annotations_odvg_labelmap.json
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_val/images.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_val/annotations.json
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_val/label_map.json
- Notes: The train split contains 49 images and 49 ODVG annotation lines. The validation split contains 48 images, 382 COCO annotations, and 80 categories. Validation, inference, deploy inference, and deploy evaluate used the extracted image directories under the local-docker data mount so annotation file_name entries resolved correctly.
- Any dataset compatibility issues: none for the selected S3 dataset.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; the one-epoch smoke run produced an exact epoch/step checkpoint.
- Best checkpoint path: /tmp/tao-model-validation/mask-grounding-dino/results/train/train/model_epoch_000_step_00049.pth
- Other checkpoints produced: /tmp/tao-model-validation/mask-grounding-dino/results/resume/train/model_epoch_001_step_00098.pth; mask_gdino_model_latest.pth symlinks were produced by the runtime but were not used for checkpoint-dependent actions.

AutoML default training rerun:
- Default direct model training used AutoML after the default policy was corrected to automl_policy=on.
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_train/
- Source: s3://nvcf-storage-handling/data/segmentation_mask_grounding_dino_val/
- Algorithm: bayesian
- Recommendations requested: 2
- Metric: val_loss, minimize
- Tuned parameters: train.optim.lr, train.optim.lr_backbone
- Recommendation 0: job dbfc3d41-3d3c-48b7-90fd-4212cd45e5a9, val_loss 45.55223083496094, checkpoint /tmp/tao-automl-validation/mask-grounding-dino/results/dbfc3d41-3d3c-48b7-90fd-4212cd45e5a9/results_dir/train/model_epoch_000_step_00049.pth
- Recommendation 1: job 39fd2b78-a8b9-4850-a41c-e0e6a3bee5c3, val_loss 59.45210647583008, checkpoint /tmp/tao-automl-validation/mask-grounding-dino/results/39fd2b78-a8b9-4850-a41c-e0e6a3bee5c3/results_dir/train/model_epoch_000_step_00049.pth
- Best recommendation: rec 0, selected by the AutoML controller summary
- Generated spec verification: both recommendations used exactly one train_data_sources entry with SDK-extracted images, annotations_odvg.jsonl, and annotations_odvg_labelmap.json; validation used SDK-extracted images plus annotations.json with data_type=OD. Each recommendation used dataset.batch_size=1, dataset.workers=0, and distinct Bayesian learning-rate/backbone-learning-rate values within the requested ranges.

Checkpoint/action verification:
- Eval checkpoint used: /tmp/tao-model-validation/mask-grounding-dino/results/train/train/model_epoch_000_step_00049.pth
- Inference checkpoint used: /tmp/tao-model-validation/mask-grounding-dino/results/train/train/model_epoch_000_step_00049.pth
- Export checkpoint used: /tmp/tao-model-validation/mask-grounding-dino/results/train/train/model_epoch_000_step_00049.pth
- Quantize checkpoint used: /tmp/tao-model-validation/mask-grounding-dino/results/train/train/model_epoch_000_step_00049.pth
- Resume/retrain checkpoint used: /tmp/tao-model-validation/mask-grounding-dino/results/train/train/model_epoch_000_step_00049.pth
- Deploy checkpoint/model path: deploy gen_trt_engine used /tmp/tao-model-validation/mask-grounding-dino/results/export/model.onnx, which was exported from the exact checkpoint above; deploy inference and deploy evaluate used /tmp/tao-model-validation/mask-grounding-dino/results/deploy/mask_grounding_dino.engine.
- Were checkpoint paths selected through the proper resolver: yes in model metadata after the fix; direct local-docker validation used explicit exact paths to verify the resolver contract.
- Any incorrect latest-checkpoint behavior found: no action selected mask_gdino_model_latest.pth. The model metadata was missing resolver mappings before the fix, which made fresh-install parent handoff fragile.

Issues found:
- Model skill issues:
  - skill_info.yaml advertised parent gen_trt_engine even though the parent PyTorch CLI supports only train, evaluate, inference, export, quantize, and default_specs.
  - spec_params were empty for evaluate, export, inference, and quantize, so fresh installs did not declare parent checkpoint resolver mappings.
  - export inputs/outputs and quantize inputs were incomplete, which made model-specific config generation and parent handoff fragile.
  - inference metadata and instructions expected classmap/json_file inputs even though inference requires image_dir plus text prompt captions.
- Config issues:
  - The packaged parent spec templates had a second blank train_data_sources entry, which caused the fresh AutoML train path to inherit an empty ODVG source unless callers replaced the full list.
  - The deploy inference and deploy evaluate templates used test_threshold, but TAO Deploy expects text_threshold for Mask Grounding DINO.
  - Deploy engine generation needed the 960x544 export shape carried into the deploy spec so the generated TensorRT profile matched the ONNX export.
- Dataset issues:
  - None for the selected Mask Grounding DINO ODVG/COCO S3 datasets.
- Checkpoint issues:
  - No runtime latest-checkpoint misuse was found. The missing resolver mappings could cause generated workflows to omit the model-specific parent checkpoint selection.
- Docker/local execution issues:
  - The deploy image attempted optional unauthenticated tokenizer metadata requests during deploy inference, but the action still completed using available tokenizer assets.
  - The containers emit telemetry and dependency warnings that do not affect execution.
  - Deploy evaluate is slow for this open-vocabulary model because the smoke run evaluates many image-caption pairs.
- Fresh-install issues:
  - Fresh installs needed parent resolver mappings, complete export/quantize metadata, correct inference data source metadata, and deploy templates that use text_threshold.

Fixes made:
- Removed invalid parent gen_trt_engine from the Mask Grounding DINO model manifests and skill_info.yaml while keeping TensorRT generation under the deploy workflow.
- Added evaluate.checkpoint, inference.checkpoint, export.checkpoint, export.onnx_file, and quantize.model_path resolver mappings to models/mask-grounding-dino/references/skill_info.yaml.
- Completed export and quantize input/output metadata in models/mask-grounding-dino/references/skill_info.yaml.
- Corrected inference metadata and SKILL.md examples to use text prompt captions instead of classmap/json_file inputs.
- Fixed the deploy inference and deploy evaluate templates to use text_threshold.
- Documented exact checkpoint selection guidance and the parent/deploy CLI boundary in models/mask-grounding-dino/SKILL.md.
- Removed the second blank train_data_sources entry from all Mask Grounding DINO parent spec templates so AutoML/direct training inherits only the configured ODVG source.

Remaining issues:
- None for the advertised parent model actions or the deploy TensorRT path after the fixes. Metrics from the one-epoch smoke checkpoint are expectedly near zero and are not an action failure.

Files changed:
- models/mask-grounding-dino/SKILL.md
- models/mask-grounding-dino/references/skill_info.yaml
- models/mask-grounding-dino/references/spec_template_deploy_evaluate.yaml
- models/mask-grounding-dino/references/spec_template_deploy_inference.yaml
- models/mask-grounding-dino/references/spec_template_train.yaml
- models/mask-grounding-dino/references/spec_template_evaluate.yaml
- models/mask-grounding-dino/references/spec_template_export.yaml
- models/mask-grounding-dino/references/spec_template_gen_trt_engine.yaml
- models/mask-grounding-dino/references/spec_template_inference.yaml
- models/mask-grounding-dino/references/spec_template_quantize.yaml
- models/mask-grounding-dino/schemas/manifest.json
- models/schemas.manifest.json
- docs/model-validation/mask-grounding-dino.md
- docs/model-validation/action-run-inventory.md

Final status:
- Fully validated for supported parent actions, AutoML default train, and TAO Deploy TensorRT actions on local-docker with image=default.
