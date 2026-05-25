# Model: rtdetr

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the RT-DETR model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- train: pass, AutoML default route with two Bayesian recommendations using `mAP50` maximize
- eval/evaluate: pass
- inference: pass
- export: pass
- deploy: pass through `models/rtdetr/deploy` for `gen_trt_engine`, TensorRT `evaluate`, and TensorRT `inference`
- prune: unsupported by the packaged RT-DETR model skill
- quantize: pass with the resolver-selected training checkpoint
- distill: pass after fixing the template to use the SDK-supported RT-DETR IOU binding
- retrain/resume: pass through `rtdetr train -e` with `train.resume_training_checkpoint_path`
- dataset convert: unsupported by the packaged RT-DETR model skill
- other: parent PyT `gen_trt_engine` is unsupported by the real PyT CLI; TensorRT is deploy-only

## Dataset used

- Source:
  - `s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_no_convert/`
  - `s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_no_convert/`
- Notes: used `images.tar.gz`, `annotations.json`, and `label_map.txt` from the object-detection synthetic subset. COCO category IDs are 1-4 (`fire_extinguisher`, `cone`, `cart`, `forklift`), so validation used `dataset.num_classes=5` and `dataset.eval_class_ids=[1, 2, 3, 4]`.
- Any dataset compatibility issues: none for the packaged RT-DETR actions. No dataset conversion action is packaged.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/rtdetr/results/4c11e0ba-b02f-4f7b-8a6a-387f7b7f0974/results_dir/train/model_epoch_000.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/rtdetr/results/4a722503-0fe9-4854-8df1-70dc5a0334c1/results_dir/train/model_epoch_000.pth`
  - distill produced `/tmp/tao-automl-validation/rtdetr/4746c326-785c-418d-ad01-2477876a81a3/results_dir/distill/model_epoch_000.pth`
  - resume produced `/tmp/tao-automl-validation/rtdetr/ec12e7fc-b9de-467c-82d2-f17e2c5ec6e7/results_dir/train/model_epoch_001.pth`
- AutoML recommendations:
  - rec 0: job `4c11e0ba-b02f-4f7b-8a6a-387f7b7f0974`, `train.optim.lr=0.00012009640240111689`, `train.optim.weight_decay=0.0000952450086052724`, `mAP50=4.35102103960396e-05`, selected as best
  - rec 1: job `4a722503-0fe9-4854-8df1-70dc5a0334c1`, `train.optim.lr=0.00015740156611367684`, `train.optim.weight_decay=0.0001740562189603568`, `mAP50=0.0`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 `model_epoch_000.pth`; evaluate status logged loading that exact checkpoint and produced `test_mAP50=4.35102103960396e-05`.
- Inference checkpoint used: best AutoML rec 0 `model_epoch_000.pth`; inference status logged loading that exact checkpoint.
- Export checkpoint used: best AutoML rec 0 `model_epoch_000.pth`; export wrote `/tmp/tao-automl-validation/rtdetr/manual_outputs/rtdetr.onnx`.
- Quantize checkpoint used: best AutoML rec 0 `model_epoch_000.pth`; quantize wrote `/tmp/tao-automl-validation/rtdetr/44c9ae75-6a70-41ac-9ed5-db028c8847ed/results_dir/quantize/quantized_model_torchao.pth`.
- Distill teacher checkpoint used: best AutoML rec 0 `model_epoch_000.pth`; distill finished successfully with the IOU `srcs` binding.
- Resume/retrain checkpoint used: best AutoML rec 0 `model_epoch_000.pth`; resume restored the selected checkpoint and produced `model_epoch_001.pth`.
- Deploy artifacts used: export ONNX `/tmp/tao-automl-validation/rtdetr/manual_outputs/rtdetr.onnx`; deploy engine `/tmp/tao-automl-validation/rtdetr/manual_outputs/rtdetr.engine`.
- Were checkpoint paths selected through the proper resolver: yes, validation used `tao_sdk.checkpoints.get_checkpoint_path` and selected `model_epoch_000.pth` for resume/evaluate/inference/export/quantize/distill and for `epoch=0`.
- Any incorrect latest-checkpoint behavior found: RT-DETR writes `rtdetr_model_latest.pth` as a latest symlink, but the resolver selected `model_epoch_000.pth`; latest was not used unless explicitly requested.

## Issues found

- Model skill issues:
  - Parent metadata advertised or retained parent `gen_trt_engine` even though the real PyT CLI rejects it.
  - Checkpoint-consuming parent actions needed explicit inputs and `spec_params` resolver mappings.
  - Distill template bindings were fragile: output names such as `pred_logits` / `pred_boxes` failed construction, and arbitrary decoder-head bindings captured `None`.
- Config issues:
  - Default AutoML metric for RT-DETR needs `mAP50` with maximize direction; `val_loss` is not the correct default objective for model-skill train invocations.
  - Deploy `gen_trt_engine` template included schema-invalid `input_channel`, `input_width`, and `input_height` under `gen_trt_engine`.
- Dataset issues:
  - None for supported actions. The selected S3 dataset requires `dataset.num_classes=5` because the category IDs are 1-4.
- Checkpoint issues:
  - RT-DETR emits both `model_epoch_*.pth` and `rtdetr_model_latest.pth`; downstream actions must resolve the intended epoch/best checkpoint instead of guessing latest.
- Docker/local execution issues:
  - Deploy metadata declared the engine path as an output file, causing the local runner to pre-create the target as a directory before TensorRT could write the engine.
- Fresh-install issues:
  - Fresh model-skill installs would misroute parent TensorRT generation, risk brittle checkpoint handoff, and fail deploy engine generation before the template fixes.

## Fixes made

- Removed parent PyT `gen_trt_engine` from RT-DETR metadata and schema manifests; TensorRT flow is now deploy-only.
- Removed the stale parent PyT `gen_trt_engine` schema/template artifacts so fresh installs do not discover the unsupported parent action by filename.
- Added checkpoint inputs and resolver mappings for train resume, evaluate, inference, export, quantize, and distill in `models/rtdetr/references/skill_info.yaml`.
- Updated parent RT-DETR instructions for supported actions, AutoML default routing, `mAP50` metric extraction, checkpoint handoff, and deploy-only TensorRT behavior.
- Updated distill template to use the RT-DETR distiller's supported IOU `srcs` binding.
- Removed unsupported deploy `gen_trt_engine.input_channel`, `input_width`, and `input_height` keys.
- Removed deploy engine-file pre-creation from deploy action metadata while preserving `gen_trt_engine.trt_engine: create_engine_file`.
- Updated deploy instructions to keep input size on deploy `evaluate` / `inference`, not deploy `gen_trt_engine`.
- Updated the per-network action inventory.

## Remaining issues

- None for packaged RT-DETR model-skill actions. Dataset conversion, pruning, and standalone retrain are not advertised as separate RT-DETR actions; resume uses `train`.

## Files changed

- `models/rtdetr/SKILL.md`
- `models/rtdetr/deploy/SKILL.md`
- `models/rtdetr/deploy/skill_info.yaml`
- `models/rtdetr/references/skill_info.yaml`
- `models/rtdetr/references/spec_template_distill.yaml`
- `models/rtdetr/references/spec_template_deploy_gen_trt_engine.yaml`
- `models/rtdetr/references/spec_template_deploy_evaluate.yaml`
- `models/rtdetr/references/spec_template_deploy_inference.yaml`
- `models/rtdetr/schemas/manifest.json`
- `models/rtdetr/schemas/gen_trt_engine.schema.json`
- `models/rtdetr/references/spec_template_gen_trt_engine.yaml`
- `models/schemas.manifest.json`
- `docs/model-validation/rtdetr.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Fully validated: all packaged RT-DETR parent model-skill actions and deploy sub-skill actions pass after model-skill metadata/template fixes.
