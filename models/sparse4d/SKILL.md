---
name: sparse4d
description: Sparse4D for multi-camera temporal 3D object detection and tracking. Uses sparse queries with deformable attention
  across camera views and time for end-to-end 3D perception. Includes instance bank for temporal tracking.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: '0.1'
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- temporal
- 3d
- detection
- tracking
---

# Sparse4D

Sparse4D for multi-camera temporal 3D object detection and tracking. Uses sparse queries with deformable attention across camera views and time for end-to-end 3D perception. Includes instance bank for temporal tracking.

Use a pretrained ResNet-101 backbone when one is available by setting
`train.pretrained_model_path`. For local smoke validation, Sparse4D training
can run with an empty `train.pretrained_model_path`, but production runs should
still use a compatible PTM.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, `export`, and deploy flows stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Training Requirements

- **Dataset type:** sparse4d
- **Formats:** ovpkl
- **Monitoring metric:** val_mAP
- Current TAO Sparse4D training emits this value in status/logs as
  `img_bbox_NuScenes/mAP` and `mAP`; AutoML metric
  extractors should treat those emitted keys as aliases for `val_mAP`.
  Multi-fidelity AutoML algorithms such as Hyperband, ASHA, and BOHB may
  promote a checkpoint to a resume job that completes without emitting a fresh
  `val_mAP` alias. In that case, compare AutoML's carried metric to the source
  rung job that emitted `img_bbox_NuScenes/mAP` or `mAP`, while still verifying
  that the promoted job resumed from the explicit epoch/step checkpoint,
  produced a real checkpoint, and is usable for evaluate/inference.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| dataset_convert | aicity.root | id |  | No |
| evaluate | dataset.data_root | eval_dataset | (from convert job, spec: aicity.split) | No |
| evaluate | model.head.instance_bank.anchor | train_datasets | /results/{dataset_convert_job_id}/anchor_init.npy | No |
| evaluate | dataset.train_dataset.ann_file | train_datasets | (from convert job, spec: aicity.split) | No |
| evaluate | dataset.val_dataset.ann_file | eval_dataset | (from convert job, spec: aicity.split) | No |
| evaluate | dataset.test_dataset.ann_file | inference_dataset | (from convert job, spec: aicity.split) | No |
| export | model.head.instance_bank.anchor | train_datasets | /results/{dataset_convert_job_id}/anchor_init.npy | No |
| inference | dataset.data_root | inference_dataset | (from convert job, spec: aicity.split) | No |
| inference | model.head.instance_bank.anchor | train_datasets | /results/{dataset_convert_job_id}/anchor_init.npy | No |
| inference | dataset.train_dataset.ann_file | train_datasets | (from convert job, spec: aicity.split) | No |
| inference | dataset.val_dataset.ann_file | eval_dataset | (from convert job, spec: aicity.split) | No |
| inference | dataset.test_dataset.ann_file | inference_dataset | (from convert job, spec: aicity.split) | No |
| quantize | dataset.data_root | train_datasets | (from convert job, spec: aicity.split) | No |
| quantize | model.head.instance_bank.anchor | train_datasets | /results/{dataset_convert_job_id}/anchor_init.npy | No |
| quantize | dataset.train_dataset.ann_file | train_datasets | (from convert job, spec: aicity.split) | No |
| quantize | dataset.val_dataset.ann_file | eval_dataset | (from convert job, spec: aicity.split) | No |
| quantize | dataset.test_dataset.ann_file | inference_dataset | (from convert job, spec: aicity.split) | No |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets |  | No |
| train | dataset.data_root | train_datasets | (from convert job, spec: aicity.split) | No |
| train | model.head.instance_bank.anchor | train_datasets | /results/{dataset_convert_job_id}/anchor_init.npy | No |
| train | dataset.train_dataset.ann_file | train_datasets | (from convert job, spec: aicity.split) | No |
| train | dataset.val_dataset.ann_file | eval_dataset | (from convert job, spec: aicity.split) | No |
| train | dataset.test_dataset.ann_file | inference_dataset | (from convert job, spec: aicity.split) | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
CONVERTED_SCENE = "<scene-from-converter>"  # e.g. "subsetscene+bev-sensor-random-0"
```

**train (mandatory data sources):**
```python
CONVERTED = "s3://bucket/results/<dataset_convert_job_id>"
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.sequences.split_num": 90,
    "dataset.train_dataset.sequences_split_num": 90,
    "dataset.data_root": f"{S3_TRAIN}/train",
    "model.head.instance_bank.anchor": f"{CONVERTED}/anchor_init.npy",
    "dataset.train_dataset.ann_file": f"{CONVERTED}/train/{CONVERTED_SCENE}_infos_train.pkl",
    "dataset.val_dataset.ann_file": f"{CONVERTED}/val/{CONVERTED_SCENE}_infos_val.pkl",
    "dataset.test_dataset.ann_file": f"{CONVERTED}/test/{CONVERTED_SCENE}_infos_test.pkl",
}
```

**evaluate (mandatory data sources):**
```python
CONVERTED = "s3://bucket/results/<dataset_convert_job_id>"
{
    "dataset.data_root": f"{S3_EVAL}/val",
    "model.head.instance_bank.anchor": f"{CONVERTED}/anchor_init.npy",
    "dataset.train_dataset.ann_file": f"{CONVERTED}/train/{CONVERTED_SCENE}_infos_train.pkl",
    "dataset.val_dataset.ann_file": f"{CONVERTED}/val/{CONVERTED_SCENE}_infos_val.pkl",
    "dataset.test_dataset.ann_file": f"{CONVERTED}/test/{CONVERTED_SCENE}_infos_test.pkl",
}
```

**export (mandatory data sources):**
```python
CONVERTED = "s3://bucket/results/<dataset_convert_job_id>"
{
    "model.head.instance_bank.anchor": f"{CONVERTED}/anchor_init.npy",
}
```

**inference (mandatory data sources):**
```python
CONVERTED = "s3://bucket/results/<dataset_convert_job_id>"
{
    "dataset.data_root": f"{S3_EVAL}/test",
    "model.head.instance_bank.anchor": f"{CONVERTED}/anchor_init.npy",
    "dataset.train_dataset.ann_file": f"{CONVERTED}/train/{CONVERTED_SCENE}_infos_train.pkl",
    "dataset.val_dataset.ann_file": f"{CONVERTED}/val/{CONVERTED_SCENE}_infos_val.pkl",
    "dataset.test_dataset.ann_file": f"{CONVERTED}/test/{CONVERTED_SCENE}_infos_test.pkl",
}
```

**quantize (mandatory data sources):**
```python
CONVERTED = "s3://bucket/results/<dataset_convert_job_id>"
{
    "dataset.data_root": f"{S3_TRAIN}/train",
    "model.head.instance_bank.anchor": f"{CONVERTED}/anchor_init.npy",
    "dataset.train_dataset.ann_file": f"{CONVERTED}/train/{CONVERTED_SCENE}_infos_train.pkl",
    "dataset.val_dataset.ann_file": f"{CONVERTED}/val/{CONVERTED_SCENE}_infos_val.pkl",
    "dataset.test_dataset.ann_file": f"{CONVERTED}/test/{CONVERTED_SCENE}_infos_test.pkl",
    "dataset.quant_calibration_dataset.images_dir": f"{S3_TRAIN}",
}
```

For local-docker runs, keep Sparse4D conversion rooted at `/data/aicity_root` and train/eval data roots at the converted split folder, for example `/data/aicity_root/train`. Mount the same host directory at `/data/aicity_root` for dataset_convert, train, evaluate, and inference; the converter extracts RGB frames there and writes those absolute frame paths into the pickle files. The annotations converter writes absolute RGB paths under the conversion root and relative depth paths under the split, so both mounts must stay stable across conversion and training. When `aicity.depth_format: h5`, normalize the converted pickle depth tuples before training if the H5 files live under `depth_maps/`:

```bash
models/sparse4d/scripts/normalize_depth_paths.py \
  --data-root /path/to/aicity_root/train \
  /path/to/results/<dataset_convert_job_id>/results_dir/train
```

Use the actual converted annotation filename emitted by Data Services. For the
packaged AICity smoke dataset the basename is
`subsetscene+bev-sensor-random-0`, so the train file is
`train/subsetscene+bev-sensor-random-0_infos_train.pkl`; do not strip the
BEV-sensor suffix back to `subsetscene_infos_train.pkl`.

For small local smoke runs with fewer camera streams than the production
default, keep `model.head.deformable_model.max_num_cams: 20` if the resulting
checkpoint will be exported. The current Sparse4D ONNX exporter constructs a
20-camera dummy input, so checkpoints trained with `max_num_cams` reduced to the
dataset camera count can load for evaluate/inference but fail export with a
deformable-attention reshape error. It is safe to keep `max_num_cams: 20` while
training/evaluating on fewer real cameras because the runtime projection matrix
controls the active camera count. Only reduce `num_cams` for smoke data when
needed; leave `max_num_cams` at 20 for export-compatible checkpoints.

For export-compatible smoke checkpoints, also keep the default anchor contract:
`model.head.instance_bank.num_anchor: 900`,
`model.head.instance_bank.num_temp_instances: 600`, and
`model.head.num_output: 900`. Sparse4D export currently creates cached feature
and anchor tensors sized for the default 600 temporal instances. If a tiny
dataset conversion produces fewer anchors, for example a 3-frame conversion that
emits a 72-row `anchor_init.npy`, evaluate/inference can still run with matching
reduced config values but export will fail during memory-bank update. Prefer
rerunning `dataset_convert` with enough real frames to initialize 900 anchors
instead of padding or inventing anchors.

When reusing a previous dataset conversion for AutoML or repeated training,
copy or mount the conversion output by the explicit `dataset_convert_job_id`,
not by the first `results_dir` found under a results root. Before launching
train/evaluate/inference, verify all required converted artifacts exist:

```bash
CONVERTED="/path/to/results/${dataset_convert_job_id}/results_dir"
test -f "${CONVERTED}/anchor_init.npy"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-0_infos_train.pkl"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-1_infos_train.pkl"
test -f "${CONVERTED}/train/subsetscene+bev-sensor-random-2_infos_train.pkl"
```

If any check fails, rerun `dataset_convert` with the `tao_toolkit.data_services`
image instead of launching train. A wrong conversion artifact often surfaces as
`FileNotFoundError: .../anchor_init.npy` during model construction.

The AICity converter may extract the full camera videos to RGB frames even when
`aicity.num_frames` is set to a small value for the converted pickle. Plan for
the raw-data mount to hold the extracted frames as well as the H5 depth maps.
## Eval Dataset

Optional. Val/test splits configured via dataset ann_file paths.

## Important Parameters

- **model.backbone**: Backbone. Default resnet_101.
- **model.neck.out_channels**: FPN output channels. Default 256. num_outs=4.
- **model.input_shape**: Input image shape [W, H]. Default [1408, 512].
- **model.head.num_output**: Number of detection output queries. Default 300.
- **model.head.num_decoder**: Number of decoder layers. Default 6.
- **model.head.temporal**: Enable temporal reasoning. Default True.
- **model.head.instance_bank.num_anchor**: Instance bank anchors. Default 900.
- **model.head.instance_bank.num_temp_instances**: Temporal instance count. Default 600.
- **model.depth_branch.loss_weight**: Depth supervision loss weight. Default 0.2.
- **dataset.batch_size**: Per-GPU batch size. Default 2.
- **dataset.num_frames**: Sequence length. Default 200.
- **dataset.classes**: Detection classes. Default [person, gr1_t2, agility_digit, nova_carter]. num_ids=70 for tracking.
- **train.optim.lr**: Learning rate. Default 5e-5. img_backbone lr_mult=0.2.
- **train.lr_scheduler**: Cosine scheduler with linear warmup (500 iters, ratio 0.333).
- **train.grad_clip.max_norm**: Gradient clipping. Default 25.
- **train.precision**: Options: bf16, fp16, fp32. Default bf16.
- **evaluate.metrics**: Eval metrics. Default ["detection"]. Optional tracking evaluation.
- **evaluate.tracking.enabled**: Enable tracking evaluation. tracking_threshold=0.2.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Multi-GPU strategy: `ddp_find_unused_parameters_true` (no fsdp support)
- `sync_batchnorm` is always enabled (True)
- Iterations per epoch computed as: `num_frames * num_bev_groups / (num_nodes * num_gpus * batch_size)`
- **Scaling:** When increasing GPUs, effective batch size grows and iterations-per-epoch shrinks proportionally

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 2 GPU(s), recommended 8 GPU(s). 40GB+ (A100 recommended) VRAM per GPU. Multi-camera temporal model is memory intensive. bf16 required for practical training. Multi-GPU strongly recommended. Instance bank requires substantial memory for temporal reasoning.

## Error Patterns

**dataset_convert required**: Must run dataset_convert first to produce annotation pickles and anchor_init.npy.

**dataset_convert container/command**: Sparse4D conversion is an AICity to
OVPKL annotations conversion. Launch `dataset_convert` with the action-level
`tao_toolkit.data_services` image and `annotations convert -e {config_path}`;
do not use the PyTorch `sparse4d` CLI for conversion. Train/evaluate/export/
inference still use the model-level PyTorch image.

**Stable raw-data path**: The AICity to OVPKL converter writes image paths into
the generated pickle files. Keep `aicity.root` at `/data/aicity_root` during
conversion, then point `dataset.data_root` at the split folder, for example
`/data/aicity_root/train` for training or `/data/aicity_root/val` for
evaluation. This preserves the converter's absolute RGB paths and relative
depth paths.

**H5 depth tuple mismatch**: If training fails with an H5 path error where the
trainer tries to open a camera directory such as
`/data/aicity_root/train/<scene>/Camera`, run
`models/sparse4d/scripts/normalize_depth_paths.py --data-root <host-aicity-root>/train <converted-ann-dir>`
after `dataset_convert` and before train/evaluate/inference. The helper rewrites
converted `depth_map_path` tuples to point at
`<scene>/depth_maps/<camera>.h5` with the H5 dataset key basename.

**Missing anchor file**: Set model.head.instance_bank.anchor to the anchor_init.npy path from dataset_convert results.

**Temporal OOM**: Reduce dataset.num_frames or dataset.batch_size if running out of memory during temporal training.

**Quantize image compatibility**: The model-skill wiring should pass
`quantize.model_path` through the parent-model resolver, and checkpoint handoff
should select the exact epoch/step checkpoint just like evaluate, inference,
export, and resume. TorchAO checkpoint quantization passes in the
`validation-fixes-20260525` PyT image and writes
`quantized_model_torchao.pth`. Older 7.0.0-rc PyT images may fail inside the
Sparse4D quantize entrypoint or lack ONNX quantization dependencies; do not
remove or skip the advertised `quantize` action if that occurs. Report the
container/image failure and keep the exact checkpoint path visible.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `sparse4d.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| dataset_convert | `results_dir` | `output_dir` | current job results directory |
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
