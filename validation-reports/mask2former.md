Model: mask2former

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- prune: unsupported
- quantize: pass
- retrain: pass
- dataset convert: unsupported
- deploy gen_trt_engine: pass
- deploy evaluate: pass
- deploy inference: pass
- AutoML/HPO train path: not run because this validation is constrained to model skills only and cannot invoke the separate AutoML skill/workflow path

Dataset used:
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_train/
- Source: s3://nvcf-storage-handling/data/segmentation_mask2former_val/
- Notes: Used COCO panoptic train and validation data with `images.tar.gz`, `images_panoptic.tar.gz`, `annotations.json`, `annotations_panoptic.json`, and panoptic label maps. Train split has 100 images and 100 panoptic masks; validation split has 5 images and 5 panoptic masks.
- Any dataset compatibility issues: The panoptic label map has 133 categories but raw category ids reach 200, so validation used `dataset.contiguous_id: false` and `model.sem_seg_head.num_classes: 201`.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact was produced in this one-epoch smoke run
- Best checkpoint path: n/a; selected exact epoch checkpoint `/workspace/run/results/train/train/model_epoch_000_step_00100.pth`
- Other checkpoints produced: `/workspace/run/results/train/train/model_epoch_000_step_00100.pth`; `/workspace/run/results/train/train/mask2former_model_latest.pth` symlinked to `model_epoch_000_step_00100.pth`; resume produced `/workspace/run/results/resume_train/train/model_epoch_001_step_00200.pth`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00100.pth`
- Inference checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00100.pth`
- Export checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00100.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00100.pth`
- Quantize checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00100.pth`
- Deploy engine source: `/workspace/run/results/export/model.onnx`
- Deploy evaluate/inference engine used: `/workspace/run/results/deploy_gen_trt_engine/mask2former.engine`
- Were checkpoint paths selected through the proper resolver: yes for the model-skill parent mapping; direct Docker validation used the exact resolved epoch artifact path because no SDK runner was invoked
- Any incorrect latest-checkpoint behavior found: no. The latest symlink was inspected but not used for checkpoint-dependent actions.

Issues found:
- Model skill issues:
  - The skill did not warn that carrying tiny smoke-test dimensions into export can fail before ONNX generation. A `128x128` export hit PyTorch ONNX `minus_one_pos != -1` shape inference, while the packaged `960x544` export passed.
- Config issues:
  - Export must use `model.mode: semantic` for deploy evaluator validation.
  - Raw COCO panoptic ids required `num_classes: 201` with `contiguous_id: false`.
- Dataset issues:
  - None after selecting the Mask2Former panoptic train/validation S3 datasets.
- Checkpoint issues:
  - No fragile latest-checkpoint behavior found.
- Docker/local execution issues:
  - TAO Deploy status files ended with `RUNNING` records containing success messages instead of an explicit final `PASS`.
  - TAO Deploy logged telemetry decode warnings after successful commands.
- Fresh-install issues:
  - The old quantize pitfall text was too broad for images that include the checkpoint-load fix; the default `torchao` checkpoint quantize path passed in the validation image.

Fixes made:
- Documented the known-good `960x544` export/deploy shape and warned against using tiny `128x128` export dimensions unless separately verified.
- Clarified that the checkpoint-based quantize failure applies to older images and that images with the quantize fix support the default `torchao` flow.
- Added deploy notes to carry the verified export input shape into TensorRT engine generation and runtime validation.

Remaining issues:
- Deploy status files still use final `RUNNING` records even when the command exits successfully.
- Deploy telemetry emits `'str' object has no attribute 'decode'` warnings after successful actions.
- AutoML/HPO behavior was not exercised because this validation run is limited to model skills.

Files changed:
- models/mask2former/SKILL.md
- models/mask2former/deploy/SKILL.md
- models/mask2former/deploy/skill_info.yaml
- validation-reports/mask2former.md

Final status:
- Fully validated for model-skill train, resume/retrain, evaluate, inference, export, quantize, and deploy gen_trt_engine/evaluate/inference actions. AutoML/HPO remains unvalidated under the model-skill-only constraint.
