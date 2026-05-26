Model: segformer

Supported actions tested:
- default_specs: pass
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy/gen_trt_engine: pass
- deploy/evaluate on tao-deployed gen_trt_engine model: pass
- deploy/inference on tao-deployed gen_trt_engine model: pass
- prune: unsupported
- quantize: pass
- retrain/resume: pass
- dataset convert: unsupported
- distill: unsupported
- AutoML/HPO: not run; model skill metadata is AutoML-enabled, but this validation was restricted to model skills only and did not run workflow or AutoML skills.

Dataset used:
- Source: s3://nvcf-storage-handling/data/segmentation_segformer_train/
- Source: s3://nvcf-storage-handling/data/segmentation_segformer_val/
- Notes: Used the real SegFormer UNet-style S3 dataset. Each root contains `images/train`, `images/val`, `images/test`, `masks/train`, and `masks/val` after extracting the packaged tarballs. Train has 20 images/masks, validation has 10 images/masks, and test has 30 images.
- Any dataset compatibility issues: The dataset is binary segmentation, so the runnable config used `dataset.segment.num_classes: 2` instead of the common `num_classes=6`. The sample images are grayscale PNGs; SegFormer accepted them through the normal `SFDataset` path.

Training result:
- Training completed: yes
- Best checkpoint produced: yes
- Best checkpoint path: /workspace/run/results/resume_train/train/model_epoch_001_step_00040.pth
- Other checkpoints produced: /workspace/run/results/train/train/model_epoch_000_step_00020.pth, plus `segformer_model_latest.pth` symlinks in train and resume result directories.
- Train metrics: val_loss=0.43508777022361755, val_acc=0.8013030886650085, val_miou=0.5147340893745422, train_loss=0.48704323172569275.
- Resume metrics: val_loss=0.38932085037231445, val_acc=0.8184402585029602, val_miou=0.5613797008991241, train_loss=0.3484508991241455.

Checkpoint/action verification:
- Eval checkpoint used: /workspace/run/results/resume_train/train/model_epoch_001_step_00040.pth
- Inference checkpoint used: /workspace/run/results/resume_train/train/model_epoch_001_step_00040.pth
- Export checkpoint used: /workspace/run/results/resume_train/train/model_epoch_001_step_00040.pth
- Resume/retrain checkpoint used: /workspace/run/results/train/train/model_epoch_000_step_00020.pth
- Quantize checkpoint used: /workspace/run/results/resume_train/train/model_epoch_001_step_00040.pth
- Deploy engine source used: /workspace/run/results/export/segformer.onnx
- Deploy evaluate/inference engine used: /workspace/run/results/deploy_gen_trt_engine/segformer.engine
- Were checkpoint paths selected through the proper resolver: yes for user-facing handoff semantics. The validation specs pinned the resolver-selected exact epoch checkpoints and did not use `segformer_model_latest.pth`.
- Any incorrect latest-checkpoint behavior found: no. Latest symlinks were produced but not used by downstream actions.

Issues found:
- Model skill issues:
  - The parent export template defaulted to `544x544` while SegFormer train/deploy defaults use `dataset.segment.img_size: 256`; the fresh-install export/deploy path needs these dimensions to stay aligned.
  - The SegFormer deploy metadata typed `dataset.segment.validation_split` and `dataset.segment.predict_split` as file inputs, but they are split-name strings such as `val` and `test`.
  - The SegFormer deploy shorthand mapped `batch_size` to `dataset.batch_size`; the real key is `dataset.segment.batch_size`.
- Config issues:
  - The common `num_classes=6` setting does not apply to this binary dataset; `dataset.segment.num_classes: 2` was required.
  - TensorBoard remained disabled per the SegFormer skill guidance.
- Dataset issues:
  - None blocking. The train and val S3 prefixes appear to contain the same tarball set, but both are compatible UNet-style roots.
- Checkpoint issues:
  - None found after pinning exact epoch checkpoints.
- Docker/local execution issues:
  - Lightning checkpoint loads used `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted local full-checkpoint restore/evaluate/export/inference/quantize paths.
  - TAO Deploy status files still end with `status: RUNNING` even when the final message says the deploy action finished successfully.
  - TAO Deploy logs include nonblocking telemetry warnings after successful completion.
- Fresh-install issues:
  - Export template shape and deploy metadata issues were fixed here.

Fixes made:
- Changed SegFormer parent export template input size from `544x544` to the validated `256x256` default.
- Added SegFormer skill guidance to keep export input size aligned with `dataset.segment.img_size`.
- Removed deploy split-name fields from file-input metadata.
- Fixed the SegFormer deploy `batch_size` shorthand to `dataset.segment.batch_size`.
- Added deploy guidance that validation and prediction splits are strings, not files.

Remaining issues:
- AutoML/HPO was not run because this pass was constrained to model skills and explicitly avoided workflow skills.
- TAO Deploy final `status.json` state remains `RUNNING` despite successful completion messages.
- TAO Deploy telemetry warning remains nonblocking.

Files changed:
- models/segformer/SKILL.md
- models/segformer/deploy/SKILL.md
- models/segformer/deploy/skill_info.yaml
- models/segformer/references/spec_template_export.yaml
- validation-reports/segformer.md

Final status:
- Fully validated for the supported SegFormer model-skill and SegFormer deploy-skill actions on local Docker with the requested validation images. AutoML/HPO was intentionally not run under the model-only constraint.
