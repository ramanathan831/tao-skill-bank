Model: mask2former

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass through the TAO Deploy skill
- deploy inference on tao-deployed gen_trt_engine model: pass
- deploy eval on tao-deployed gen_trt_engine model: pass
- prune: unsupported/not advertised by the Mask2Former model skill
- quantize: fail
- retrain/resume: pass through train.resume_training_checkpoint_path
- AutoML default train route: pass with Bayesian automl_max_recommendations=2
- dataset convert: unsupported/not advertised by the Mask2Former model skill
- other: parent PyTorch gen_trt_engine was advertised before this validation, but the parent CLI rejects it. TensorRT engine generation belongs to the deploy skill and the parent action was removed from model metadata.

Dataset used:
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/images.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/images_panoptic.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/annotations.json
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/annotations_panoptic.json
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/label_map_panoptic.json
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/images.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/images_panoptic.tar.gz
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/annotations.json
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/annotations_panoptic.json
- Notes: The train split contains 100 COCO images, 768 instance annotations, 100 panoptic annotations, and 133 panoptic categories. The validation split contains 5 images, 51 instance annotations, 5 panoptic annotations, and 133 categories. The panoptic raw category ids reach 200, so validation used `dataset.contiguous_id: false` and `model.sem_seg_head.num_classes: 201`.
- Any dataset compatibility issues: none for the selected S3 panoptic dataset.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; the one-epoch smoke run produced an exact epoch/step checkpoint.
- Best checkpoint path: /tmp/tao-model-validation/mask2former/results/train/model_epoch_000_step_00100.pth
- Other checkpoints produced: /tmp/tao-model-validation/mask2former/results/resume/model_epoch_001_step_00200.pth; mask2former_model_latest.pth symlinks were produced by the runtime but were not used for checkpoint-dependent actions.

AutoML default training rerun:
- Default direct model training used AutoML after the default policy was corrected to automl_policy=on.
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/
- Algorithm: bayesian
- Recommendations requested: 2
- Metric: mIoU, maximize
- Tuned parameters: train.optim.lr, train.optim.weight_decay
- Recommendation 0: job 1870ad51-dfeb-4ff6-9447-9b1a5848b04d, mIoU 0.005142086651176214, checkpoint /tmp/tao-automl-validation/mask2former/results/1870ad51-dfeb-4ff6-9447-9b1a5848b04d/results_dir/train/model_epoch_000_step_00100.pth
- Recommendation 1: job 92a0840d-2f2d-4430-b829-7d435213c709, mIoU 0.005142086651176214, checkpoint /tmp/tao-automl-validation/mask2former/results/92a0840d-2f2d-4430-b829-7d435213c709/results_dir/train/model_epoch_000_step_00100.pth
- Best recommendation: rec 0, selected by the AutoML controller summary
- Generated spec verification: both recommendations used the real S3 COCO panoptic inputs after SDK extraction, dataset train/val/test type=coco_panoptic, dataset.contiguous_id=false, model.sem_seg_head.num_classes=201, dataset.train.batch_size=1, and distinct Bayesian learning-rate/weight-decay values within the requested ranges.

Checkpoint/action verification:
- Eval checkpoint used: /tmp/tao-model-validation/mask2former/results/train/model_epoch_000_step_00100.pth
- Inference checkpoint used: /tmp/tao-model-validation/mask2former/results/train/model_epoch_000_step_00100.pth
- Export checkpoint used: /tmp/tao-model-validation/mask2former/results/train/model_epoch_000_step_00100.pth
- Quantize checkpoint/model used: checkpoint-based torchao quantize used /tmp/tao-model-validation/mask2former/results/train/model_epoch_000_step_00100.pth and failed in the runtime checkpoint loader; ONNX quantize used /tmp/tao-model-validation/mask2former/results/export/model.onnx from the exact checkpoint and failed because the default image lacks modelopt.onnx quantization support.
- Resume/retrain checkpoint used: /tmp/tao-model-validation/mask2former/results/train/model_epoch_000_step_00100.pth
- Deploy checkpoint/model path: deploy gen_trt_engine used /tmp/tao-model-validation/mask2former/results/export/model.onnx, which was exported from the exact checkpoint above; deploy inference and deploy evaluate used /tmp/tao-model-validation/mask2former/results/deploy/mask2former.engine.
- Were checkpoint paths selected through the proper resolver: yes in model metadata after the fix; direct local-docker validation used explicit exact paths to verify the resolver contract.
- Any incorrect latest-checkpoint behavior found: no action selected mask2former_model_latest.pth. The model metadata was missing resolver mappings before the fix, which made fresh-install parent handoff fragile.

Issues found:
- Model skill issues:
  - skill_info.yaml advertised parent gen_trt_engine even though the parent PyTorch CLI supports only train, evaluate, inference, export, quantize, and default_specs.
  - spec_params were empty for evaluate, export, inference, and quantize, so fresh installs did not declare parent checkpoint or ONNX resolver mappings.
  - export inputs/outputs and quantize.model_path were missing from skill_info.yaml.
  - deploy metadata and templates used ADE-style fields and an invalid top-level dataset.type instead of the COCO panoptic split-level fields accepted by the deploy schema.
- Config issues:
  - The S3 panoptic label map has 133 categories but raw category ids up to 200. Raw-id semantic validation therefore requires num_classes 201.
  - TAO Deploy evaluate supports semantic engines, so the deploy path was validated with model.mode semantic.
  - ONNX quantize calibration required a fixed dataset.test.target_size; otherwise calibration image tensors had inconsistent shapes.
- Dataset issues:
  - None for train/eval/inference/export/deploy. Quantize reached runtime/package issues after the dataset shape fix.
- Checkpoint issues:
  - No runtime latest-checkpoint misuse was found. The missing resolver mappings could cause generated workflows to omit exact parent checkpoint or export artifact selection.
- Docker/local execution issues:
  - Checkpoint-based quantize failed in the default PyTorch image because mask2former/scripts/quantize.py passes experiment_spec to Mask2formerPlModule.load_from_checkpoint instead of the required cfg argument.
  - ONNX quantize failed in the default PyTorch image because modelopt.onnx.quantization is not installed.
  - The containers emit telemetry and dependency warnings that do not affect the passing actions.
- Fresh-install issues:
  - Fresh installs needed parent resolver mappings, complete export/quantize metadata, and deploy templates that match the deploy schema.

Fixes made:
- Removed invalid parent gen_trt_engine from the Mask2Former model manifests and skill_info.yaml while keeping TensorRT generation under the deploy skill.
- Added evaluate.checkpoint, inference.checkpoint, export.checkpoint, export.onnx_file, and quantize.model_path resolver mappings to models/mask2former/references/skill_info.yaml.
- Completed export and quantize input/output metadata in models/mask2former/references/skill_info.yaml.
- Updated Mask2Former deploy metadata and templates to use COCO panoptic split-level fields and removed invalid top-level dataset.type.
- Documented exact checkpoint selection, raw-id num_classes behavior, semantic deploy evaluation, deploy schema requirements, and quantize runtime blockers.
- No additional Mask2Former model skill code change was needed for the AutoML default rerun.

Remaining issues:
- Quantize remains unresolved in the default image. The model skill can now pass the correct artifact path, but the runtime checkpoint loader and missing ONNX quantization package block both tested quantize paths.

Files changed:
- models/mask2former/SKILL.md
- models/mask2former/deploy/SKILL.md
- models/mask2former/deploy/skill_info.yaml
- models/mask2former/references/skill_info.yaml
- models/mask2former/references/spec_template_deploy_evaluate.yaml
- models/mask2former/references/spec_template_deploy_inference.yaml
- models/mask2former/schemas/manifest.json
- models/schemas.manifest.json
- docs/model-validation/mask2former.md
- docs/model-validation/action-run-inventory.md

Final status:
- Partially validated on local-docker with image=default. AutoML default train and all advertised parent/deploy actions passed except quantize, which is blocked by default-image runtime/package issues after model-skill metadata fixes.
