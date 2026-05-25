# Model: segformer

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the SegFormer model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- train: pass, AutoML default route with two Bayesian recommendations using `val_miou` maximize
- eval/evaluate: pass
- inference: pass
- export: pass
- deploy: pass through `models/segformer/deploy` for `gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`
- prune: unsupported by the packaged SegFormer model skill
- quantize: pass with the resolver-selected training checkpoint
- retrain/resume: pass through `segformer train -e` with `train.resume_training_checkpoint_path`
- dataset convert: unsupported by the packaged SegFormer model skill
- other: parent PyT `gen_trt_engine` is unsupported by the real PyT CLI; TensorRT is deploy-only

## Dataset used

- Source:
  - `s3://nvcf-storage-handling/data/segmentation_segformer_train/`
  - `s3://nvcf-storage-handling/data/segmentation_segformer_val/`
- Notes: used the packaged binary segmentation image/mask tarballs under `images/{train,val,test}.tar.gz` and `masks/{train,val}.tar.gz`. Validation used `dataset.segment.num_classes=2`, `dataset.segment.img_size=128`, `dataset.segment.batch_size=1`, and `dataset.segment.workers=1`.
- Any dataset compatibility issues: none for the packaged SegFormer actions. No dataset conversion action is packaged.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/segformer/919645ee-3170-40f8-9b6b-a7a1dc91e94a/results_dir/train/model_epoch_000_step_00020.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/segformer/a6b733dd-dd3a-4a36-b12d-83491bf1d35c/results_dir/train/model_epoch_000_step_00020.pth`
  - resume produced `/tmp/tao-automl-validation/segformer/0dc30e82-dfae-419b-a722-274c1138c64c/results_dir/train/model_epoch_001_step_00040.pth`
- AutoML recommendations:
  - rec 0: job `919645ee-3170-40f8-9b6b-a7a1dc91e94a`, `train.optim.lr=3.799977909725112e-05`, `train.optim.weight_decay=0.008812998704102622`, `val_miou=0.44207941740751266`, selected as best
  - rec 1: job `a6b733dd-dd3a-4a36-b12d-83491bf1d35c`, `train.optim.lr=6.887397910037069e-05`, `train.optim.weight_decay=0.005456882624518751`, `val_miou=0.4308849647641182`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 `model_epoch_000_step_00020.pth`; evaluate status logged loading that exact checkpoint and produced `test_miou=0.44207941740751266`.
- Inference checkpoint used: best AutoML rec 0 `model_epoch_000_step_00020.pth`; inference status logged loading that exact checkpoint.
- Export checkpoint used: best AutoML rec 0 `model_epoch_000_step_00020.pth`; export wrote `/tmp/tao-automl-validation/segformer/manual_outputs/segformer.onnx`.
- Quantize checkpoint used: best AutoML rec 0 `model_epoch_000_step_00020.pth`; quantize wrote `/tmp/tao-automl-validation/segformer/a064e287-9246-476b-b0f6-9e2eb6cb6a3e/results_dir/quantize/quantized_model_torchao.pth`.
- Resume/retrain checkpoint used: best AutoML rec 0 `model_epoch_000_step_00020.pth`; resume restored the selected checkpoint and produced `model_epoch_001_step_00040.pth`.
- Deploy artifacts used: export ONNX `/tmp/tao-automl-validation/segformer/manual_outputs/segformer.onnx`; deploy engine `/tmp/tao-automl-validation/segformer/manual_outputs/segformer.engine`.
- Were checkpoint paths selected through the proper resolver: yes, validation used `tao_sdk.checkpoints.get_checkpoint_path` and selected `model_epoch_000_step_00020.pth` for resume/evaluate/inference/export/quantize and for `epoch=0, step=20`.
- Any incorrect latest-checkpoint behavior found: SegFormer writes `segformer_model_latest.pth` as a latest symlink, but the resolver selected `model_epoch_000_step_00020.pth`; latest was not used unless explicitly requested.

## Issues found

- Model skill issues:
  - Parent metadata advertised stale parent `gen_trt_engine` even though the real PyT CLI rejects it.
  - Checkpoint-consuming parent actions needed explicit inputs and `spec_params` resolver mappings.
  - Parent train metadata did not expose resume/pretrained paths needed for reliable checkpoint handoff on fresh installs.
- Config issues:
  - Default AutoML metric for SegFormer needs `val_miou` with maximize direction; `val_loss` is not the correct default objective for model-skill train invocations.
  - Deploy templates used a backbone/input-shape/normalization combination that did not match the trained/exported validation run.
- Dataset issues:
  - None for supported actions. The selected S3 dataset is binary segmentation, so the common `num_classes=6` config had to be overridden to `2`.
- Checkpoint issues:
  - SegFormer emits both `model_epoch_*_step_*.pth` and `segformer_model_latest.pth`; downstream actions must resolve the intended epoch/step/best checkpoint instead of guessing latest.
- Docker/local execution issues:
  - Deploy metadata declared the engine path as an output file, causing the local runner to pre-create the target as a directory before TensorRT could write the engine.
- Fresh-install issues:
  - Fresh model-skill installs would discover the unsupported parent TensorRT action, risk brittle checkpoint handoff, and fail deploy engine generation before the metadata/template fixes.

## Fixes made

- Removed parent PyT `gen_trt_engine` from SegFormer metadata and schema manifests; TensorRT flow is now deploy-only.
- Removed the stale parent PyT `gen_trt_engine` schema/template artifacts so fresh installs do not discover the unsupported parent action by filename.
- Added checkpoint inputs and resolver mappings for train resume, evaluate, inference, export, and quantize in `models/segformer/references/skill_info.yaml`.
- Updated parent SegFormer instructions for supported actions, AutoML default routing, `val_miou` metric extraction, checkpoint handoff, and deploy-only TensorRT behavior.
- Removed deploy engine-file pre-creation from deploy action metadata while preserving `gen_trt_engine.trt_engine: create_engine_file`.
- Updated deploy templates to match the validated FAN-small SegFormer run and kept deploy validation at the exported `128x128` input size.
- Updated the per-network action inventory.
- Corrected the aggregate schema manifest so stale parent `gen_trt_engine` discovery is removed for both SegFormer and the previously validated RT-DETR entry.

## Remaining issues

- None for packaged SegFormer model-skill actions. Dataset conversion, pruning, and standalone retrain are not advertised as separate SegFormer actions; resume uses `train`.

## Files changed

- `models/segformer/SKILL.md`
- `models/segformer/deploy/SKILL.md`
- `models/segformer/deploy/skill_info.yaml`
- `models/segformer/references/skill_info.yaml`
- `models/segformer/references/spec_template_deploy_gen_trt_engine.yaml`
- `models/segformer/references/spec_template_deploy_evaluate.yaml`
- `models/segformer/references/spec_template_deploy_inference.yaml`
- `models/segformer/schemas/manifest.json`
- `models/segformer/schemas/gen_trt_engine.schema.json`
- `models/segformer/references/spec_template_gen_trt_engine.yaml`
- `models/schemas.manifest.json`
- `docs/model-validation/segformer.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Fully validated: all packaged SegFormer parent model-skill actions and deploy sub-skill actions pass after model-skill metadata/template fixes.
