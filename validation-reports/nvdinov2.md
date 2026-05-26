Model: nvdinov2

Supported actions tested:
- train: pass
- eval: unsupported; SSL evaluation is downstream-task specific and no `eval` action is advertised by the model skill
- inference: pass
- export: pass
- deploy: pass
- prune: unsupported
- quantize: unsupported
- retrain: pass
- dataset convert: unsupported
- deploy gen_trt_engine: pass
- deploy inference on gen_trt_engine model: unsupported; the deploy sub-skill exposes only `gen_trt_engine`, and PyT inference consumes `.pth`/`.tlt` checkpoints rather than TensorRT engines
- distill: unsupported; the skill explicitly documents that the current TAO PyT CLI exposes train, inference, export, and default_specs, not a standalone distill action
- AutoML/HPO train path: not run because this validation is constrained to model skills only and cannot invoke the separate AutoML skill/workflow path

Dataset used:
- Source: s3://nvcf-storage-handling/data/nvdinov2_train_cats_dogs/images_train.tar.gz
- Source: s3://nvcf-storage-handling/data/nvdinov2_test_cats_dogs/images_test.tar.gz
- Notes: Used the real NvDINOv2 cats/dogs SSL image dataset. Train split has 6,404 image files after ignoring two `_DS_Store` files, with 3,200 cat images and 3,204 dog images. Test split has 2,023 image files after ignoring two `_DS_Store` files, with 1,011 cat images and 1,012 dog images.
- Any dataset compatibility issues: None. The `_DS_Store` files were present in both tarballs but were ignored by the image loader; inference produced 2,023 result rows plus the CSV header.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact was produced in this one-epoch SSL smoke run
- Best checkpoint path: n/a; selected exact epoch/step student checkpoint `/workspace/run/results/resume_train/student_epoch_001_step_01600.pth` for inference/export
- Other checkpoints produced: initial train produced `/workspace/run/results/train/model_epoch_000_step_00800.pth`, `/workspace/run/results/train/student_epoch_000_step_00800.pth`, and `/workspace/run/results/train/teacher_epoch_000_step_00800.pth`; resume/retrain produced `/workspace/run/results/resume_train/model_epoch_001_step_01600.pth`, `/workspace/run/results/resume_train/student_epoch_001_step_01600.pth`, and `/workspace/run/results/resume_train/teacher_epoch_001_step_01600.pth`

Checkpoint/action verification:
- Eval checkpoint used: n/a; eval is unsupported for this model skill
- Inference checkpoint used: `/workspace/run/results/resume_train/student_epoch_001_step_01600.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/student_epoch_001_step_01600.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00800.pth`
- Deploy engine source: `/workspace/run/results/export/nvdinov2.onnx`
- Were checkpoint paths selected through the proper resolver: yes for the model-skill parent mapping; direct Docker validation used the exact resolved epoch/step artifacts because no SDK runner was invoked
- Any incorrect latest-checkpoint behavior found: no. No latest symlink was used; resume used the full `model_epoch_*` training checkpoint while inference/export used the selected `student_epoch_*` checkpoint.

Issues found:
- Model skill issues:
  - None.
- Config issues:
  - Export and deploy must carry the smoke-run structural settings forward (`vit_s`, `img_size: 224`, `use_custom_attention: false`, and `224x224` export dimensions). The existing skill/deploy notes already document this requirement.
  - The PyT export action emitted the full ONNX graph to the log even with `export.verbose: false`, making the generated export log very large.
- Dataset issues:
  - The train and test tarballs each contain two `_DS_Store` files. The loader ignored them and processed the expected image count.
- Checkpoint issues:
  - None. Full training checkpoints and student checkpoints were used for their correct action types.
- Docker/local execution issues:
  - TAO Deploy status files ended with `RUNNING` records containing success messages instead of an explicit final `PASS`.
  - TAO Deploy logged telemetry decode warnings after successful engine generation.
- Fresh-install issues:
  - None for direct model-skill local Docker execution with the validation images.

Fixes made:
- None. The model and deploy skill instructions already covered the checkpoint handoff and deploy shape requirements validated here.

Remaining issues:
- Deploy status files still use final `RUNNING` records even when the command exits successfully.
- Deploy telemetry emits `'str' object has no attribute 'decode'` warnings after successful actions.
- PyT export logs the full ONNX graph despite `export.verbose: false`.
- AutoML/HPO behavior was not exercised because this validation run is limited to model skills.

Files changed:
- validation-reports/nvdinov2.md

Final status:
- Fully validated for model-skill train, resume/retrain, inference, export, and deploy gen_trt_engine actions. Eval, prune, quantize, dataset convert, deploy inference, and standalone distill are unsupported by this model skill. AutoML/HPO remains unvalidated under the model-skill-only constraint.
