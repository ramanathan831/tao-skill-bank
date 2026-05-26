Model: rtdetr

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
- distill: pass
- AutoML/HPO: not run; model skill metadata is AutoML-enabled, but this validation was restricted to model skills only and did not run workflow or AutoML skills.

Dataset used:
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_no_convert/
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_no_convert/
- Notes: COCO object-detection sample with 50 train images and 50 validation images. Images were extracted from the packaged `images.tar.gz` archives; extraction creates nested `images/images` directories.
- Any dataset compatibility issues: Category IDs are one-based, so the runnable RT-DETR config used `dataset.num_classes: 5` and `dataset.eval_class_ids: [1, 2, 3, 4]`. The train and validation JSON category name order differs, so metric values are only useful as smoke-test evidence.

Training result:
- Training completed: yes
- Best checkpoint produced: yes; the one-epoch smoke run produced a single epoch checkpoint.
- Best checkpoint path: /workspace/run/results/resume_train/model_epoch_001.pth
- Other checkpoints produced: /workspace/run/results/train/model_epoch_000.pth, /workspace/run/results/distill/model_epoch_000.pth, plus `rtdetr_model_latest.pth` symlinks in the train, resume, and distill result directories.
- Train metrics: val_mAP=3.574135870000104e-06, val_mAP50=2.198998356248729e-05, val_loss=2.6039955615997314, train_loss=30.632455825805664.
- Resume metrics: val_mAP=2.2312090363965974e-06, val_mAP50=2.2312090363965977e-05, val_loss=2.4878644943237305, train_loss=29.746984481811523.

Checkpoint/action verification:
- Eval checkpoint used: /workspace/run/results/resume_train/model_epoch_001.pth
- Inference checkpoint used: /workspace/run/results/resume_train/model_epoch_001.pth
- Export checkpoint used: /workspace/run/results/resume_train/model_epoch_001.pth
- Resume/retrain checkpoint used: /workspace/run/results/train/model_epoch_000.pth
- Quantize checkpoint used: /workspace/run/results/resume_train/model_epoch_001.pth
- Distill teacher checkpoint used: /workspace/run/results/resume_train/model_epoch_001.pth
- Deploy engine source used: /workspace/run/results/export/rtdetr.onnx
- Deploy evaluate/inference engine used: /workspace/run/results/deploy_gen_trt_engine/rtdetr.engine
- Were checkpoint paths selected through the proper resolver: yes for user-facing handoff semantics. The validation specs pinned the resolver-selected exact epoch checkpoints and did not use the latest symlink.
- Any incorrect latest-checkpoint behavior found: no. `rtdetr_model_latest.pth` was produced, but downstream actions used the intended exact checkpoint or export artifact.

Issues found:
- Model skill issues:
  - The parent export and deploy templates used `960x544`, while RT-DETR training and the documented default use `640x640`. Export at `960x544` failed during ONNX tracing with a tensor-size mismatch in `hybrid_encoder.py` positional embedding addition.
- Config issues:
  - The common `num_classes=6` setting does not match this S3 sample's one-based category IDs. RT-DETR required `dataset.num_classes: 5` for IDs 1-4.
  - Deploy COCO evaluation expects `maxDets=100`; the validation kept `model.num_select: 100` instead of lower smoke-test values.
- Dataset issues:
  - Train and validation JSON files contain the same class set but different category order. This does not block the action smoke tests but makes the tiny-run metrics non-representative.
- Checkpoint issues:
  - None found after pinning resolver-selected exact checkpoint paths.
- Docker/local execution issues:
  - The local Docker run emitted a nonblocking cache-directory warning until `/workspace/run/.cache/xdg/torch/kernels` existed.
  - TAO Deploy status files still end with `status: RUNNING` even when the final message says the deploy action finished successfully.
  - TAO Deploy logs include nonblocking telemetry warnings after successful completion.
- Fresh-install issues:
  - None beyond the export/deploy template shape mismatch fixed here.

Fixes made:
- Changed RT-DETR parent export template input size from `960x544` to `640x640`.
- Changed RT-DETR deploy evaluate and deploy inference template input sizes from `960x544` to `640x640`.
- Added an RT-DETR skill error-pattern note explaining the export shape mismatch and the validated `640x640` default.

Remaining issues:
- AutoML/HPO was not run because this pass was constrained to model skills and explicitly avoided workflow skills.
- TAO Deploy final `status.json` state remains `RUNNING` despite successful completion messages.
- The S3 sample dataset category ordering mismatch remains; it does not block smoke validation.

Files changed:
- models/rtdetr/SKILL.md
- models/rtdetr/references/spec_template_export.yaml
- models/rtdetr/references/spec_template_deploy_evaluate.yaml
- models/rtdetr/references/spec_template_deploy_inference.yaml
- validation-reports/rtdetr.md

Final status:
- Fully validated for the supported RT-DETR model-skill and RT-DETR deploy-skill actions on local Docker with the requested validation images. AutoML/HPO was intentionally not run under the model-only constraint.
