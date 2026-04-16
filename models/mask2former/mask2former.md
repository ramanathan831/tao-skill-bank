# Mask2Former

Mask2Former for universal image segmentation (panoptic, instance, and semantic). Transformer-based with masked attention for high-quality segmentation results.

Set model.backbone.pretrained_weights for Swin backbone weights.

## Eval Dataset

Optional. Val data sources are part of the dataset config alongside train.

## Important Parameters

- **model.sem_seg_head.num_classes**: Number of segmentation classes. Default 200. Must match your annotation categories.
- **model.backbone.swin.type**: Swin Transformer variant. Default tiny. Options include tiny, small, base, large.
- **model.mode**: Segmentation mode. Default panoptic. Options: panoptic, instance, semantic.
- **train.optim.lr**: Learning rate. Default 2e-4 (AdamW).
- **dataset.train.batch_size**: Per-GPU batch size. Default 1. Mask2Former is memory-intensive due to per-pixel predictions.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- Same DDP/FSDP behavior as DINO (activation checkpoint aware)
- FAN backbones auto-enable `sync_batchnorm`
- `fsdp` forces FP16

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Export / TRT Defaults

- TRT data types: FP32, FP16 only — **INT8 is NOT supported**

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Mask2Former is memory-heavy. batch_size=1 is the default for good reason. Multi-GPU recommended for reasonable training speed.

## Error Patterns

**CUDA out of memory**: batch_size is already 1 by default. Reduce image resolution in augmentation config or use a smaller Swin variant.

**Panoptic vs instance format mismatch**: Ensure you provide the correct annotation format matching model.mode setting.
