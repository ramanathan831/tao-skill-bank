# Depth Net Mono

Monocular depth estimation using Metric Depth Anything v2 or Relative Depth Anything architectures. Predicts per-pixel depth from single RGB images.

Uses pretrained Depth Anything v2 encoder. Set model.mono_backbone.pretrained_path.

## Eval Dataset

Optional. Val dataset configured via dataset.val_dataset.data_sources.

## Important Parameters

- **model.model_type**: Model architecture. Options: MetricDepthAnything, RelativeDepthAnything. Default MetricDepthAnything.
- **model.mono_backbone.encoder**: Backbone encoder. Options: vits, vitb, vitl, vitg. Default vitl.
- **model.mono_backbone.pretrained_path**: Path to pretrained Depth Anything v2 encoder weights.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW).
- **train.lr_scheduler**: LR scheduler. Options: MultiStepLR, StepLR, CustomMultiStepLRScheduler, LambdaLR, PolynomialLR, OneCycleLR, CosineAnnealingLR.
- **train.precision**: Training precision. Options: bf16, fp16, fp32.
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.
- **train.activation_checkpoint**: Enable activation checkpointing. Default False.
- **dataset.train_dataset.data_sources**: List of dataset source dicts. Each has data_file and dataset_name fields. Multiple sources per split.
- **dataset.augmentation.crop_size**: Training crop size. Default [518, 518].
- **dataset.max_depth**: Maximum depth range for metric depth estimation.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` forces precision to FP16

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Export / TRT Defaults

- TRT data types: FP32, FP16 only

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 24GB+ VRAM per GPU. ViT-Large encoder is memory intensive. Use bf16 precision and activation checkpointing for larger inputs.

## Error Patterns

**Depth range mismatch**: Ensure dataset.max_depth / min_depth match the actual depth range in your data.

**Missing pretrained weights**: DepthAnything v2 encoder requires pretrained_path to be set for fine-tuning.
