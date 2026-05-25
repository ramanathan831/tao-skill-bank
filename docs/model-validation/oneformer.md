# Model: oneformer

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the OneFormer model skill as `skill_dir`.

## Supported actions tested

- train: pass, AutoML default route with two Bayesian recommendations
- eval: pass
- inference: pass after fixing the model skill to declare `inference.images_dir`
- export: pass after fixing model metadata to avoid pre-creating `export.onnx_file`
- deploy gen_trt_engine: fail, deploy image builder bug on OneFormer two-input ONNX
- deploy evaluate: blocked, no TensorRT engine produced
- deploy inference: blocked, no TensorRT engine produced
- prune: unsupported by the packaged OneFormer PyT CLI
- quantize: pass after preserving the parent AutoML job path expected by checkpoint hparams
- retrain/resume: pass
- dataset convert: unsupported by the packaged OneFormer PyT CLI
- other: parent PyT `gen_trt_engine` is unsupported and was removed from model metadata; deploy sub-skill owns TensorRT actions

## Dataset used

- Source: `s3://nvcf-storage-handling/data/segmentation_oneformer_train/`
- Source: `s3://nvcf-storage-handling/data/segmentation_oneformer_val/`
- Notes: used real COCO-panoptic-style files: `images.tar.gz`, `images_panoptic.tar.gz`, `annotations.json`, and `label_map.json`.
- Any dataset compatibility issues: the dataset label map has 133 classes, so OneFormer used `model.sem_seg_head.num_classes=133` and `dataset.contiguous_id=True` instead of the common `num_classes=6`.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/oneformer/results/2c60b110-4617-4967-aaa3-b4fd34961b44/results_dir/train/model_epoch_000_step_00017.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/oneformer/results/f912e1e3-e575-404f-856c-2bf01143e176/results_dir/train/model_epoch_000_step_00017.pth`
  - resume/retrain produced `/tmp/tao-automl-validation/oneformer/38c295c2-5eb2-4b5e-ad5f-9829c8069957/results_dir/train/model_epoch_001_step_00034.pth`
- Best metric: rec 0 reported `mIoU=0.002` to AutoML; train status recorded `mIoU=0.0015951207606121898`
- Rec 1 metric: `mIoU=0.001`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 checkpoint above; eval passed with `test_mIoU=0.0014672691468149424`
- Inference checkpoint used: best AutoML rec 0 checkpoint above; inference rerun processed 15/15 real validation images and wrote 15 JPEG predictions
- Export checkpoint used: best AutoML rec 0 checkpoint above; export wrote `/tmp/tao-automl-validation/oneformer/manual_outputs/oneformer_export_640.onnx`
- Resume/retrain checkpoint used: best AutoML rec 0 checkpoint above; logs show `Setting resume checkpoint` and `Restored all states`
- Quantize checkpoint used: best AutoML rec 0 checkpoint above; quantize wrote `quantized_model_torchao.pth`
- Were checkpoint paths selected through the proper resolver: yes, validation used `tao_sdk.checkpoints.get_checkpoint_path`
- Any incorrect latest-checkpoint behavior found: no latest-checkpoint fallback was used; all downstream actions used the selected AutoML best checkpoint
- Additional checkpoint note: OneFormer Lightning checkpoints retain train-time absolute dataset paths in hparams, so downstream checkpoint-loading jobs must preserve the parent AutoML job path at `/results/<parent_job_id>` inside the action container.

## Issues found

- Model skill issues:
  - `inference` declared `dataset.test.images`, but the OneFormer predict dataloader reads `inference.images_dir`; the first inference run passed with no predictions.
  - Parent PyT `gen_trt_engine` was listed even though the OneFormer PyT CLI does not support it.
  - `export.onnx_file` as a declared file output caused the local runner to pre-create a directory at the ONNX path, and OneFormer export asserts the path must not exist.
  - Deploy `gen_trt_engine.trt_engine` as a declared file output had the same pre-create problem risk for engine output paths.
- Config issues:
  - Tiny 128x128 export shape can trigger a PyTorch ONNX shape-inference failure. The default 640x640 export shape passed.
- Dataset issues:
  - No blocking dataset issue; the selected S3 OneFormer dataset uses 133 classes.
- Checkpoint issues:
  - Quantize failed until the parent AutoML job path was preserved at the checkpoint's saved `/results/<job_id>` hparams location.
- Docker/local execution issues:
  - Deploy `gen_trt_engine` failed in `nvcr.io/nvstaging/tao/tao-toolkit-deploy:7.0.0-rc-171-multiarch` with `IndexError: Out of bounds`; the builder assumes every ONNX input is 4D and fails on OneFormer's 2D `task_tokens` input.
- Fresh-install issues:
  - OneFormer downstream checkpoint actions need `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for trusted TAO-produced checkpoints because the checkpoint includes OmegaConf objects.

## Fixes made

- Added checkpoint-dependent inputs and `spec_params` for train resume, eval, inference, export, and quantize in `models/oneformer/references/skill_info.yaml`.
- Added `inference.images_dir` as a declared inference input.
- Removed unsupported parent PyT `gen_trt_engine` action from OneFormer metadata.
- Removed `export.onnx_file` as a pre-created file output and documented the explicit ONNX file override requirement.
- Removed `gen_trt_engine.trt_engine` as a pre-created deploy file output/input and documented the explicit engine file override requirement.
- Added OneFormer skill guidance for AutoML checkpoint path preservation, inference image wiring, export path behavior, and deploy builder limitations.

## Remaining issues

- TAO Deploy `gen_trt_engine` remains unresolved in the current deploy image because the common deploy builder fails on OneFormer's two-input ONNX (`images`, `task_tokens`). TensorRT evaluate and inference remain blocked by that missing engine.

## Files changed

- `models/oneformer/SKILL.md`
- `models/oneformer/references/skill_info.yaml`
- `models/oneformer/deploy/SKILL.md`
- `models/oneformer/deploy/skill_info.yaml`
- `docs/model-validation/oneformer.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

- Partially validated: all parent OneFormer model actions tested and fixed where possible; deploy TensorRT actions are blocked by a deploy-image builder bug outside the model skill metadata.
