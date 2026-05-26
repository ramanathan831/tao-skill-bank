Model: ml-recog

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- prune: unsupported
- quantize: unsupported
- retrain: pass
- dataset convert: unsupported
- deploy gen_trt_engine: pass
- deploy evaluate: pass
- deploy inference: pass
- AutoML/HPO train path: not run because this validation is constrained to model skills only and cannot invoke the separate AutoML skill/workflow path

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ml_recog_train/metric_learning_recognition/retail-product-checkout-dataset_classification_demo/
- Notes: Used the real retail-product ML-Recog dataset. Known classes: train 5 classes/178 images, reference 5 classes/178 images, val 5 classes/381 images, test 5 classes/336 images. Unknown classes: reference 1 class/29 images, test 1 class/87 images.
- Any dataset compatibility issues: None. Data is ImageNet-folder style and matched the model-skill reference/query requirements.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact was produced in this one-epoch smoke run
- Best checkpoint path: n/a; selected exact epoch/step checkpoint `/workspace/run/results/train/train/model_epoch_000_step_00044.pth`
- Other checkpoints produced: `/workspace/run/results/train/train/model_epoch_000_step_00044.pth`; `/workspace/run/results/train/train/ml_model_latest.pth` symlinked to `model_epoch_000_step_00044.pth`; resume produced `/workspace/run/results/resume_train/train/model_epoch_001_step_00088.pth`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00044.pth`
- Inference checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00044.pth`
- Export checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00044.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/train/model_epoch_000_step_00044.pth`
- Deploy engine source: `/workspace/run/results/export/model.onnx`
- Deploy evaluate/inference engine used: `/workspace/run/results/deploy_gen_trt_engine/ml-recog.engine`
- Were checkpoint paths selected through the proper resolver: yes for the model-skill parent mapping; direct Docker validation used the exact resolved epoch/step artifact path because no SDK runner was invoked
- Any incorrect latest-checkpoint behavior found: no. The latest symlink was inspected but not used for checkpoint-dependent actions.

Issues found:
- Model skill issues:
  - Deploy evaluate/inference template batch sizes could silently drop a final partial batch. With `evaluate.batch_size: 8`, deploy evaluation processed 80 of 87 unknown query images. Rerunning with `batch_size: 1` processed all 87 images.
- Config issues:
  - Trusted ML-Recog checkpoint actions require `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` with the current PyTorch runtime; this was already documented in the skill and was used for resume, evaluate, inference, and export.
  - INT8 deploy engine generation required real calibration images and a writable calibration cache path.
- Dataset issues:
  - None.
- Checkpoint issues:
  - No fragile latest-checkpoint behavior found.
- Docker/local execution issues:
  - TAO Deploy status files ended with `RUNNING` records containing success messages instead of an explicit final `PASS`.
  - TAO Deploy logged telemetry decode warnings after successful commands.
- Fresh-install issues:
  - The deploy templates used unsafe default runtime batch sizes for small or non-divisible datasets.

Fixes made:
- Changed deploy evaluate and inference templates to default to `batch_size: 1`.
- Added deploy skill notes warning that larger runtime batch sizes can silently drop final partial batches.
- Added deploy metadata notes to keep validation at batch size 1 unless dataset divisibility is verified.

Remaining issues:
- Deploy status files still use final `RUNNING` records even when commands exit successfully.
- Deploy telemetry emits `'str' object has no attribute 'decode'` warnings after successful actions.
- AutoML/HPO behavior was not exercised because this validation run is limited to model skills.

Files changed:
- models/ml-recog/references/spec_template_deploy_evaluate.yaml
- models/ml-recog/references/spec_template_deploy_inference.yaml
- models/ml-recog/deploy/SKILL.md
- models/ml-recog/deploy/skill_info.yaml
- validation-reports/ml-recog.md

Final status:
- Fully validated for model-skill train, resume/retrain, evaluate, inference, export, and deploy gen_trt_engine/evaluate/inference actions. AutoML/HPO remains unvalidated under the model-skill-only constraint.
