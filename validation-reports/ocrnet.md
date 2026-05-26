Model: ocrnet

Supported actions tested:
- dataset convert: pass
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass (`gen_trt_engine`, deploy `evaluate`, deploy `inference`)
- prune: pass
- quantize: pass
- retrain: pass
- other: n/a
- AutoML/HPO train path: not run because this validation is constrained to model skills only and cannot invoke the separate AutoML skill/workflow path

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_train/train.tar.gz
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_train/train/gt_new.txt
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_val/test.tar.gz
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_val/test/gt_new.txt
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_val/character_list
- Notes: Used the real OCR text-recognition dataset. Train has 25 images but 20 labeled rows; validation has 25 images and 25 labeled rows; the character list has 36 entries.
- Any dataset compatibility issues: The validation `gt_new.txt` starts with a UTF-8 BOM on the first filename. The first conversion pass still exited PASS but created only 24 samples. A sanitized local copy of the same GT file was used for final validation conversion, evaluation, and deploy evaluation, producing all 25 samples.

Training result:
- Training completed: yes
- Best checkpoint produced: yes
- Best checkpoint path: `/workspace/run/results/resume_train/best_accuracy.pth`
- Other checkpoints produced: initial train produced `/workspace/run/results/train/model_epoch_000_step_00005.pth`; resume produced `/workspace/run/results/resume_train/model_epoch_001_step_00010.pth`; pruned retrain produced `/workspace/run/results/retrain/model_epoch_000_step_00005.pth`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/resume_train/best_accuracy.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train/best_accuracy.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/best_accuracy.pth`
- Prune checkpoint used: `/workspace/run/results/resume_train/best_accuracy.pth`
- Quantize checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00010.pth` with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
- Resume/retrain checkpoint used: resume used `/workspace/run/results/train/model_epoch_000_step_00005.pth`; model-skill retrain used `/workspace/run/results/prune/pruned_0.4.pth` via `model.pruned_graph_path`
- Deploy engine input used: `/workspace/run/results/export/ocrnet.onnx`
- Deploy evaluate/inference engine used: `/workspace/run/results/deploy_gen_trt_engine/ocrnet.engine`
- Were checkpoint paths selected through the proper resolver: yes for model-skill parent mapping rules; direct Docker validation used exact resolved artifacts from model outputs.
- Any incorrect latest-checkpoint behavior found: no. No action used a latest checkpoint alias.

Issues found:
- Model skill issues:
  - Deploy `gen_trt_engine.tensorrt.calibration.cal_image_dir` must be serialized as a YAML list. The first deploy engine run failed schema validation when the field was a scalar folder path.
  - The skill did not document that direct `dataset_convert` writes `data.mdb` and `lock.mdb` directly under `dataset_convert.results_dir`; downstream LMDB actions should use that folder.
  - The skill did not document the GT-file UTF-8 BOM failure mode.
- Config issues:
  - Quantize was run with the trusted local full checkpoint and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, as required by the model skill for PyTorch 2.6+ checkpoint loading.
  - Deploy engine generation used FP16, matching the deploy skill recommendation.
- Dataset issues:
  - Validation GT had a BOM on the first row, which caused raw dataset conversion to skip `word_1.png` until a sanitized local GT copy was used.
- Checkpoint issues:
  - None. Best-checkpoint actions used `best_accuracy.pth`; resume and quantize used exact full epoch checkpoints; retrain used the exact prune output.
- Docker/local execution issues:
  - Deploy actions logged telemetry warnings after successful completion.
  - Status files often ended with final `RUNNING` records containing success messages; command logs contained explicit success or PASS lines.
- Fresh-install issues:
  - None after documenting LMDB output layout, BOM handling, and deploy calibration-list shape.

Fixes made:
- Updated `models/ocrnet/SKILL.md` with direct dataset-convert LMDB output guidance and the GT BOM error pattern.
- Updated `models/ocrnet/deploy/SKILL.md` and `models/ocrnet/deploy/skill_info.yaml` to document that `gen_trt_engine.tensorrt.calibration.cal_image_dir` must be a YAML list.

Remaining issues:
- AutoML/HPO behavior was not exercised because this validation run is limited to model skills.
- Direct train from raw tarball was not separately used; the real raw dataset was converted through the model-supported `dataset_convert` action and all training, quantize, and retrain runs consumed the converted LMDB outputs.

Files changed:
- models/ocrnet/SKILL.md
- models/ocrnet/deploy/SKILL.md
- models/ocrnet/deploy/skill_info.yaml
- validation-reports/ocrnet.md

Final status:
- Fully validated for OCRNet model-skill dataset_convert, train, resume/retrain, evaluate, inference, export, prune, quantize, deploy engine generation, deploy evaluation, and deploy inference on local Docker with the validation images.
