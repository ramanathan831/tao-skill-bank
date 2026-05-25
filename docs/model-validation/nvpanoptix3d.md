# Model: nvpanoptix3d

Validated on 2026-05-25 with `platform=local-docker`, `image=default`
(`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch`),
`num_gpus=1`, direct model skill actions, and the model skill's AutoML-enabled
default train route. Workflow skills were not run.

## Supported Actions Tested

- train: pass after using the toolkit-required `train.precision: fp32`
- eval: pass
- inference: pass after providing a flat folder of real RGB test images
- export: pass for the implemented 2D ONNX export
- deploy: unsupported/not advertised
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain: pass through `train.resume_training_checkpoint_path`
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised
- other: fp16 train/retrain was probed and rejected by the real CLI with `ValueError: Only fp32 precision is supported.` Export was probed with both 2D and 3D ONNX output fields; the current export entrypoint only produces `export.onnx_file_2d`.

## Dataset Used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_test/`
- Notes: used the real packaged Front3D-style splits. Train contains 20 JSON samples across 4 scenes and 140 extracted files; val contains 5 JSON samples across 2 scenes and 35 extracted files; test contains 10 JSON samples across 1 scene and 70 extracted files. Inference used a flat folder created from the 10 real test `rgb_*.png` files because the predict dataset scans only top-level `.jpg`/`.png` files.
- Any dataset compatibility issues: none blocking. The common `num_classes=6` override is not compatible with this model; `model.sem_seg_head.num_classes` remained 13 to match the packaged label map.

## Training Result

- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact; the one-epoch run produced one concrete epoch/step checkpoint, which was selected for downstream validation
- Best checkpoint path: `/tmp/tao-model-validation/nvpanoptix3d/results/train/model_epoch_000_step_00020.pth`
- Other checkpoints produced: `/tmp/tao-model-validation/nvpanoptix3d/results/train/nvpanoptix3d_model_latest.pth` symlink to the exact train checkpoint. Resume validation wrote `/tmp/tao-model-validation/nvpanoptix3d/results/retrain/model_epoch_000_step_00020.pth` and a `nvpanoptix3d_model_latest.pth` symlink.

## AutoML Default Training Rerun

- Default model training was rerun through the model skill's AutoML-enabled train route after confirming the previous direct validation had exercised normal training only.
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_train/`
- Eval source: `s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_val/`
- Test source: `s3://nvcf-storage-handling/data/purpose_built_models_nvpanoptix3d_test/`
- Algorithm: bayesian
- Recommendations requested: 2
- Metric: `train_loss`, minimize; TAO `status.json` recorded PRQ/RSQ/RRQ only, so the AutoML runner used the training progress log for `train_loss`.
- Tuned parameter: `train.optim.lr`
- Recommendation 0: job `098172d6-4a38-4ee0-b72e-c134d2a8b8ca`, `train.optim.lr=0.0002868659351565142`, `train_loss=214.992`, checkpoint `/tmp/tao-automl-validation/nvpanoptix3d/results/098172d6-4a38-4ee0-b72e-c134d2a8b8ca/results_dir/train/model_epoch_000_step_00020.pth`
- Recommendation 1: job `8a7c013d-0b24-4ed3-b16b-063dff90b6e4`, `train.optim.lr=0.0002700214525931551`, `train_loss=277.819`, checkpoint `/tmp/tao-automl-validation/nvpanoptix3d/results/8a7c013d-0b24-4ed3-b16b-063dff90b6e4/results_dir/train/model_epoch_000_step_00020.pth`
- Best recommendation: rec 0, selected by the AutoML controller summary.
- Generated spec verification: both recommendations used SDK-staged real S3 train/val/test roots, `dataset.enable_3d=true`, `dataset.contiguous_id=true`, `model.sem_seg_head.num_classes=13`, `train.precision=fp32`, `train.num_epochs=1`, and `train.optim.monitor_name=train_loss`.

## Checkpoint/Action Verification

- Eval checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`; status log confirmed `Loading checkpoint`.
- Inference checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`; status log confirmed `Loading checkpoint`, and the run predicted 10/10 images.
- Export checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`; output was `/workspace/results/export/nvpanoptix3d_2d.onnx` plus ONNX external-data weight files.
- Resume/retrain checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`; logs confirmed `Setting resume checkpoint` and `Restored all states`.
- Were checkpoint paths selected through the proper resolver: yes for the fixed parent-model metadata contract on evaluate/export/inference; direct local-docker validation supplied the exact epoch/step path because workflow/SDK resolver execution was out of scope. Resume used the explicit `train.resume_training_checkpoint_path` action input.
- Any incorrect latest-checkpoint behavior found: no executed downstream action used `nvpanoptix3d_model_latest.pth`. The metadata was fragile before the fix because evaluate/export/inference lacked checkpoint inputs and `parent_model` mappings.

## Issues Found

- Model skill issues:
  - Parent metadata did not declare `evaluate.checkpoint`, `export.checkpoint`, `inference.checkpoint`, staged train checkpoint inputs, or resolver mappings.
  - Documentation said fp16 was recommended, but the real train code only supports fp32.
  - Documentation said export produces separate 2D and 3D ONNX models, but the real export entrypoint only calls the 2D exporter.
- Config issues:
  - The first train and resume specs used fp16 based on the docs and failed. Reruns passed with `train.precision: fp32`.
  - The first inference run pointed at extracted scene directories and returned PASS with a zero-length predict dataloader. Rerun passed after using a flat folder of real RGB images.
- Dataset issues:
  - None blocking.
- Checkpoint issues:
  - Missing metadata meant checkpoint-consuming actions could not be safely chained through the skill resolver before the fix.
- Docker/local execution issues:
  - Large checkpoints are produced: train and resume artifacts are each about 5.3 GB, each AutoML recommendation folder was about 5.4 GB, and ONNX export with external data is about 4.4 GB.
- Fresh-install issues:
  - Fresh installs would expose missing checkpoint handoff metadata, fp16 guidance that fails at runtime, and inference image-dir guidance that can silently produce a zero-length PASS.

## Fixes Made

- Added optional train `checkpoint_2d`, `checkpoint_3d`, and `resume_training_checkpoint_path` inputs.
- Added evaluate/export/inference checkpoint inputs and `parent_model` mappings.
- Added `export.onnx_file_2d` output metadata with a `create_onnx_file_2d` mapping.
- Updated the model docs for fp32-only train/resume, flat RGB inference folders, 2D-only ONNX export, and exact checkpoint handoff behavior.
- No additional NvPanoptix3D model skill code change was needed for the AutoML default rerun.

## Remaining Issues

- `export.onnx_file_3d` remains present in the generated schema/template, but the current toolkit image does not produce a 3D ONNX artifact.
- SDK `parent_model` resolver execution was not invoked because workflow/SDK paths were out of scope; the metadata contract is now present and direct runs used exact checkpoint paths.

## Files Changed

- `models/nvpanoptix3d/SKILL.md`
- `models/nvpanoptix3d/references/skill_info.yaml`
- `docs/model-validation/nvpanoptix3d.md`
- `docs/model-validation/action-run-inventory.md`

## Final Status

Fully validated for the NvPanoptix3D model skill's advertised parent and AutoML
default train actions on local Docker after the metadata/docs fixes. The
current image supports 2D ONNX export only.
