Model: nvpanoptix3d

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: unsupported; no deploy sub-skill or deploy action is advertised for this model
- prune: unsupported
- quantize: unsupported
- retrain: pass
- dataset convert: unsupported
- AutoML/HPO train path: not run because this validation is constrained to model skills only and cannot invoke the separate AutoML skill/workflow path

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_train/
- Source: s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_val/
- Source: s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_test/
- Notes: Used the real Front3D-style NvPanoptix3D dataset. Train has 20 JSON entries, validation has 5 JSON entries, and test has 10 JSON entries. Each split includes `meta/colormap.json`, `meta/frustum_mask.npz`, split JSON, and `data/images.tar.gz`.
- Any dataset compatibility issues: The train/evaluate loaders expect scene folders under `base_dir/data/<scene_id>/...`, so the S3 `data/images.tar.gz` archives were extracted into each split's `data/` directory. Inference scans only top-level `.jpg`/`.png` files, so the 10 real test RGB images were copied into a flat folder for inference.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact was produced in this one-epoch smoke run
- Best checkpoint path: n/a; selected exact retrained epoch/step checkpoint `/workspace/run/results/resume_train_3/model_epoch_001_step_00040.pth`
- Other checkpoints produced: initial train produced `/workspace/run/results/train/model_epoch_000_step_00020.pth`; first resume boundary check produced `/workspace/run/results/resume_train/model_epoch_000_step_00020.pth`; final resume/retrain produced `/workspace/run/results/resume_train_3/model_epoch_000_step_00020.pth` and `/workspace/run/results/resume_train_3/model_epoch_001_step_00040.pth`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/resume_train_3/model_epoch_001_step_00040.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train_3/model_epoch_001_step_00040.pth`
- Export checkpoint used: `/workspace/run/results/resume_train_3/model_epoch_001_step_00040.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00020.pth`
- Export artifact: `/workspace/run/results/export_final/nvpanoptix3d_2d.onnx`; export also produced 604 external-data sidecar files in the result directory, for 605 files total and about 4.4 GB
- Were checkpoint paths selected through the proper resolver: yes for the model-skill parent mapping; direct Docker validation used exact resolved epoch/step artifact paths because no SDK runner was invoked
- Any incorrect latest-checkpoint behavior found: no. No latest checkpoint alias was used for checkpoint-dependent actions.

Issues found:
- Model skill issues:
  - Resume from an end-of-epoch one-epoch smoke checkpoint with `train.num_epochs: 2` restored the checkpoint and exited successfully without advancing to a new checkpoint. Setting `train.num_epochs: 3` advanced to `model_epoch_001_step_00040.pth`.
- Config issues:
  - This model requires the packaged Python module entrypoint, `train.precision: fp32`, `dataset.contiguous_id: true`, `dataset.enable_3d: true`, and `model.sem_seg_head.num_classes: 13` for this dataset.
  - Export writes only the 2D ONNX model in this image, plus external-data sidecars. No 3D ONNX artifact was produced.
- Dataset issues:
  - Inference required a flat folder derived from the real test RGB images because the S3 archive extracts to scene subdirectories.
- Checkpoint issues:
  - End-of-epoch resume requires an epoch target high enough to force a new epoch if the validation goal is true retraining, not just checkpoint restoration.
- Docker/local execution issues:
  - Status files ended with final `RUNNING` records containing success messages; command logs contained explicit `Execution status: PASS`.
  - First train run spent extra time building sparse convolution benchmark cache.
- Fresh-install issues:
  - None after following the model skill's module-entrypoint, fp32, contiguous-id, and flat-inference-folder guidance.

Fixes made:
- Added a model-skill error-pattern note explaining the end-of-epoch resume behavior and requiring verification of a new exact epoch/step checkpoint before downstream handoff.

Remaining issues:
- Status files still use final `RUNNING` records even when commands exit successfully.
- Export writes a large external-data sidecar set next to the 2D ONNX; consumers must keep the ONNX and sidecars together.
- AutoML/HPO behavior was not exercised because this validation run is limited to model skills.

Files changed:
- models/nvpanoptix3d/SKILL.md
- validation-reports/nvpanoptix3d.md

Final status:
- Fully validated for model-skill train, resume/retrain, evaluate, inference, and export actions. Deploy, prune, quantize, and dataset convert are unsupported by this model skill. AutoML/HPO remains unvalidated under the model-skill-only constraint.
