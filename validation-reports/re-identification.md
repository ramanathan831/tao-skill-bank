Model: re-identification

Supported actions tested:
- train: pass
- resume/retrain: pass via `re_identification train` with `train.resume_training_checkpoint_path`
- eval: pass
- inference: pass
- export: pass
- default_specs: pass
- dataset convert: unsupported by this model skill
- deploy: unsupported by this model skill
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- other: AutoML/HPO not run because this validation pass is restricted to model skill actions only

Dataset used:
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_re_identification_train/`
- Notes: Used `sample_train.tar.gz`, `sample_test.tar.gz`, and `sample_query.tar.gz`. The archives extract to flat Market1501-style image folders with 1741 train images, 1854 gallery/test images, 466 query images, 100 identities, and 6 cameras.
- Any dataset compatibility issues: The common `num_classes=6` override is incompatible with this dataset; `dataset.num_classes` was set to the dataset-correct value of 100.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; the train jobs produced epoch checkpoints and a latest symlink
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/model_epoch_000_step_00099.pth`
  - `/workspace/run/results/train/reid_model_latest.pth` -> latest symlink, not used for downstream validation
  - `/workspace/run/results/resume_train/model_epoch_001_step_00198.pth`
  - `/workspace/run/results/resume_train/reid_model_latest.pth` -> latest symlink, not used for downstream validation

Checkpoint/action verification:
- Resume checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00099.pth`
- Eval checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00198.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00198.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00198.pth`
- Were checkpoint paths selected through the proper resolver: yes for the model skill wiring; the skill maps train resume to `resume_model` and evaluate/export/inference to `parent_model`. Direct local-docker validation used exact produced epoch checkpoints after artifact inspection, not `reid_model_latest.pth`.
- Any incorrect latest-checkpoint behavior found: none. Re-ID writes `reid_model_latest.pth`, but downstream actions were validated with explicit epoch checkpoints.

Issues found:
- Model skill issues:
  - The PyTorch 2.6 checkpoint-load guidance only named evaluate/inference, but resume and export also load the full trusted checkpoint and need the same `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` environment setting.
  - `default_specs` is a packaged CLI action, but invoking it with the usual `-e <spec.yaml>` path leaves `results_dir` unset. It must be invoked with a Hydra `results_dir=...` override.
- Config issues:
  - `dataset.num_classes` must match the 100 identities in the validation dataset.
  - Evaluate and inference require explicit query/gallery dataset paths and output file/plot paths.
- Dataset issues:
  - None blocking.
- Checkpoint issues:
  - None. Resume restored exactly from `model_epoch_000_step_00099.pth`; downstream actions used `model_epoch_001_step_00198.pth`.
- Docker/local execution issues:
  - None blocking.
- Fresh-install issues:
  - Trusted checkpoint consumers need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` under the current PyTorch behavior.

Fixes made:
- Updated `models/re-identification/SKILL.md` to include resume/export in the trusted checkpoint-load guidance.
- Added `default_specs` invocation guidance for the required Hydra `results_dir` override.

Remaining issues:
- AutoML/HPO was not executed in this model-only pass.
- Deploy, prune, quantize, dataset conversion, and standalone retrain are not supported by this model skill.

Files changed:
- `models/re-identification/SKILL.md`
- `validation-reports/re-identification.md`

Final status:
- Fully validated for supported model actions; partially validated if counting AutoML/HPO, which was outside this model-only pass.
