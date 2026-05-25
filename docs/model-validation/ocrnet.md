# Model: ocrnet

Validated on 2026-05-25 with `platform=local-docker`, `image=default`
(`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch` for parent
model actions and `nvcr.io/nvstaging/tao/tao-toolkit-deploy:7.0.0-rc-171-multiarch`
for deploy), `num_gpus=1`, direct model/deploy skill actions, and the model
skill's AutoML-enabled default train route. Workflow skills were not run.

## Supported Actions Tested

- train: pass through the AutoML default train route with Bayesian `automl_max_recommendations=2`
- eval/evaluate: pass
- inference: pass
- export: pass after adding the missing `export.onnx_file` output handoff
- deploy/gen_trt_engine: pass after regenerating the deploy experiment template
- deploy/inference on tao-deployed gen_trt_engine model: pass
- deploy/evaluate: pass
- prune: pass after adding the missing `prune.pruned_file` output handoff
- quantize: pass in the rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`
- retrain: pass after routing the model-skill action through `ocrnet train -e` with `model.pruned_graph_path`
- dataset convert: pass
- other: parent PyT `gen_trt_engine` was probed and confirmed unsupported by the real PyT CLI; TensorRT is handled by the deploy sub-skill

## Dataset Used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_val/`
- Notes: train used `train.tar.gz`, `train/gt_new.txt`, and `character_list`; validation/test used `test.tar.gz`, `test/gt_new.txt`, and `character_list`. The train archive has 20 cropped word images; the validation archive has 25 images.
- Any dataset compatibility issues: one validation image logs as corrupted, but train/evaluate/inference/deploy actions still complete successfully. Accuracy is 0.0 on the tiny smoke dataset, which is expected for a one-epoch validation run.

## Training Result

- Training completed: yes, through AutoML default train route
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/ocrnet/results/497d5fea-1366-407a-9a18-46213a3293c9/results_dir/train/best_accuracy.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/ocrnet/results/497d5fea-1366-407a-9a18-46213a3293c9/results_dir/train/model_epoch_000_step_00003.pth`
  - retrain produced `/tmp/tao-automl-validation/ocrnet/94db88ac-1660-4f99-b22d-6402b11cf140/results_dir/train/best_accuracy.pth`
  - retrain produced `/tmp/tao-automl-validation/ocrnet/94db88ac-1660-4f99-b22d-6402b11cf140/results_dir/train/model_epoch_000_step_00003.pth`
- Source-fixed rerun checkpoints:
  - `/tmp/tao-source-fixed-rerun/current/ocrnet/results/train/best_accuracy.pth`
  - `/tmp/tao-source-fixed-rerun/current/ocrnet/results/train/model_epoch_000_step_00003.pth`

## AutoML Default Training Rerun

- Default model training was run through `AutoMLRunner` with `skill_dir=/localhome/local-rarunachalam/tao-skills-external/models/ocrnet`, proving direct model invocation routes through AutoML by default.
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_train/train.tar.gz`
- Eval source: `s3://nvcf-storage-handling/data/purpose_built_models_ocrnet_val/test.tar.gz`
- Algorithm: bayesian
- Recommendations requested: 2
- Metric: `val_acc_1`, maximize
- Tuned parameters: `train.optim.lr`, `dataset.augmentation.aug_prob`
- Recommendation 0: job `497d5fea-1366-407a-9a18-46213a3293c9`, `train.optim.lr=1.0886464000835228`, `dataset.augmentation.aug_prob=0.04002793769574087`, `val_acc_1=0.0`, selected as best by tie-break
- Recommendation 1: job `c3cb3c5e-febb-4964-8191-990c04cef199`, `train.optim.lr=0.8321097682828124`, `dataset.augmentation.aug_prob=0.05780296670689622`, `val_acc_1=0.0`
- Generated spec verification: both recommendations used the real S3 train/val archives, `train.num_epochs=1`, `dataset.batch_size=8`, `dataset.workers=0`, and distinct Bayesian recommendations.

## Checkpoint/Action Verification

- Eval checkpoint used: `best_accuracy.pth` from AutoML rec 0; log loaded the selected checkpoint and evaluated 25 samples.
- Inference checkpoint used: `best_accuracy.pth` from AutoML rec 0; log predicted all validation images.
- Export checkpoint used: `best_accuracy.pth` from AutoML rec 0; output ONNX `/tmp/tao-automl-validation/ocrnet/manual_outputs/ocrnet_export_retry.onnx`.
- Prune checkpoint used: `best_accuracy.pth` from AutoML rec 0; output PTH `/tmp/tao-automl-validation/ocrnet/manual_outputs/ocrnet_pruned_retry.pth`.
- Resume/retrain checkpoint used: prune output `/tmp/tao-automl-validation/ocrnet/manual_outputs/ocrnet_pruned_retry.pth`; rerun through `ocrnet train -e` completed and wrote fresh checkpoints.
- Deploy checkpoint/artifact handoff used: exported ONNX to deploy gen_trt_engine; generated engine `/tmp/tao-automl-validation/ocrnet/manual_outputs/ocrnet_deploy_schema_retry.engine`; deploy evaluate/inference used that exact engine.
- Quantize checkpoint/artifacts used:
  - `best_accuracy.pth` failed as a saved model object after trusted PyTorch load.
  - exact `model_epoch_000_step_00003.pth` failed in `OCRNetModel.load_from_checkpoint` with missing `dm`.
  - exported ONNX reached calibration and then failed because `modelopt.onnx.quantization` is missing in the default PyT image.
- Source-fixed quantize checkpoint used: `/workspace/results/train/model_epoch_000_step_00003.pth`, selected through `tao_sdk.checkpoints.get_checkpoint_path(..., epoch=0, step=3, allow_latest=False)`. The rerun intentionally did not use `best_accuracy.pth`.
- Were checkpoint paths selected through the proper resolver: yes. Direct validation used `tao_sdk.checkpoints.get_checkpoint_path`, which selected `best_accuracy.pth` for best-checkpoint actions and exact `model_epoch_000_step_00003.pth` for the epoch-specific checkpoint probe. The source-fixed quantize rerun also used the exact epoch/step checkpoint.
- Any incorrect latest-checkpoint behavior found: yes. Metadata previously lacked the checkpoint/output mappings needed to avoid fragile handoff; fixed for evaluate, inference, export, prune, quantize, train resume, and retrain.

## Issues Found

- Model skill issues:
  - Parent metadata advertised PyT `gen_trt_engine`, but the PyT CLI does not expose that subtask.
  - Parent metadata routed retrain to `ocrnet retrain`, but the real CLI exposes retraining through `ocrnet train -e` with `model.pruned_graph_path`.
  - Parent metadata lacked checkpoint inputs and resolver mappings for checkpoint-consuming actions.
  - Parent metadata lacked file outputs for `export.onnx_file` and `prune.pruned_file`.
  - Deploy metadata lacked calibration cache/image inputs and deploy evaluate ground-truth input.
- Config issues:
  - Deploy experiment template was stale and failed Hydra validation on `dataset.input_channel`.
- Dataset issues:
  - One validation image is logged as corrupted, but it is non-blocking for smoke validation.
- Checkpoint issues:
  - Quantize does not accept the OCRNet best-weight file; the exact Lightning epoch checkpoint failed in the original default PyT image but passed in the rebuilt PyT image.
- Docker/local execution issues:
  - PyT quantize ONNX path is blocked because the default image lacks `modelopt.onnx.quantization`.
  - Deploy telemetry reports a non-fatal decode warning after successful deploy commands.
- Fresh-install issues:
  - A fresh install would fail export, prune, retrain, and deploy gen_trt_engine before these metadata/template fixes.

## Fixes Made

- Added checkpoint inputs and `parent_model`/`resume_model` mappings in `models/ocrnet/references/skill_info.yaml`.
- Added `export.onnx_file` and `prune.pruned_file` output metadata.
- Routed model-skill `retrain` through `ocrnet train -e` with `model.pruned_graph_path`.
- Removed unsupported parent PyT `gen_trt_engine` from the parent model metadata and documented TensorRT as deploy-only.
- Regenerated the deploy experiment template from the deploy container default spec shape.
- Added deploy gen_trt_engine calibration cache/image metadata and deploy evaluate GT-file metadata.
- Updated OCRNet instructions for checkpoint selection, raw eval GT files, deploy TensorRT routing, and quantize caveats.
- Reran source-fixed quantize on `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525` with the resolver-selected full epoch checkpoint; the action completed successfully and wrote `/workspace/results/quantize/quantized_model_torchao.pth`.

## Remaining Issues

- The original default PyT image still has the quantize blockers listed above; the rebuilt source-fixed PyT image passes the packaged PyTorch-checkpoint quantize path.
- ONNX quantize was not rerun because the packaged source-fixed action path validated here uses PyTorch checkpoint quantization with `torchao`.

## Files Changed

- `models/ocrnet/SKILL.md`
- `models/ocrnet/references/skill_info.yaml`
- `models/ocrnet/references/spec_template_deploy_experiment.yaml`
- `models/ocrnet/deploy/skill_info.yaml`
- `docs/model-validation/ocrnet.md`
- `docs/model-validation/action-run-inventory.md`

## Final Status

Fully validated for packaged model-skill actions after the source-fixed image
rerun. All advertised parent model actions and deploy TensorRT actions pass;
quantize requires the rebuilt PyT image and a full epoch checkpoint, not
`best_accuracy.pth`.
