Model: optical-inspection

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass (`gen_trt_engine`, TensorRT `evaluate`, TensorRT `inference`)
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- retrain: pass via resume training
- dataset convert: not run: preconverted dataset provided; converter requires raw Factory PCB layout not present in the validation S3 data
- default_specs: pass
- other: AutoML/HPO not run because this validation pass is restricted to model/deploy skill actions only

Dataset used:
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_optical_inspection_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_optical_inspection_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_optical_inspection_test/`
- Notes: Each split contains `images.tar.gz` plus `dataset.csv`; each CSV has 25 samples plus header. The tarballs extract an outer `images/` wrapper, so specs must point at the inner directory containing `golden/` and board folders, for example `/workspace/run/data/train/images/images`.
- Any dataset compatibility issues: The data is already in TAO-ready Optical Inspection format. It is not compatible with `dataset_convert` because the raw Factory PCB conversion inputs are not included.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; Lightning produced epoch/step checkpoints
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/model_epoch_000_step_00003.pth`
  - `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00006.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00003.pth`
- Deploy checkpoint/engine used:
  - `gen_trt_engine.onnx_file`: `/workspace/run/results/export/optical_inspection.onnx`
  - `evaluate.trt_engine`: `/workspace/run/results/deploy_gen_trt_engine/optical_inspection.engine`
  - `inference.trt_engine`: `/workspace/run/results/deploy_gen_trt_engine/optical_inspection.engine`
- Were checkpoint paths selected through the proper resolver: yes for the model skill wiring; `train.resume_training_checkpoint_path` maps to `resume_model`, and evaluate/export/inference map to `parent_model`. Direct local-docker validation used the exact produced checkpoint paths after artifact inspection, not a blind latest-file lookup.
- Any incorrect latest-checkpoint behavior found: none. A first manual guess at `model_epoch_000_step_00004.pth` was rejected before execution; the actual produced checkpoint was `model_epoch_000_step_00003.pth`, and resume/downstream specs were corrected to exact artifacts.

Issues found:
- Model skill issues:
  - The parent skill did not call out the extra extracted `images/` wrapper in the S3 validation tarballs, which can make a fresh local spec point at the wrong image root.
- Config issues:
  - Deploy evaluate must use `dataset.infer_dataset.*`; using `dataset.test_dataset.*` leaves the deploy container reading stale or missing paths.
  - Default ONNX export uses a static batch in this smoke configuration, so deploy batch/profile settings must stay aligned at batch size 1.
- Dataset issues:
  - Dataset conversion cannot be validated from the available S3 splits because they are already converted and lack raw Factory PCB conversion inputs.
- Checkpoint issues:
  - No incorrect latest checkpoint behavior found. The checkpoint names are step-based and must be resolved or listed rather than guessed.
- Docker/local execution issues:
  - None blocking. Telemetry emitted non-fatal warnings after successful deploy actions.
- Fresh-install issues:
  - Users need an explicit note to inspect the extracted image root before setting `dataset.*.images_dir`.

Fixes made:
- Added parent skill guidance for preconverted S3 tarballs that extract with an outer `images/` wrapper and require specs to target the inner image root.
- Added an error pattern for image-root mismatches when CSV-relative paths cannot be found.
- Added deploy metadata notes that TensorRT evaluate reads `dataset.infer_dataset.*` and that static-batch ONNX exports require aligned TensorRT batch/profile settings.

Remaining issues:
- AutoML/HPO was not executed in this model-only pass.
- `dataset_convert` remains unvalidated for this model because compatible raw conversion input data is not present in `s3://nvcf-storage-handling/data/`.

Files changed:
- `models/optical-inspection/SKILL.md`
- `models/optical-inspection/deploy/skill_info.yaml`
- `validation-reports/optical-inspection.md`

Final status:
- Fully validated for supported model and deploy actions available from the preconverted validation dataset; partially validated if counting AutoML/HPO or raw `dataset_convert`, which were outside this model-only/preconverted-data pass.
