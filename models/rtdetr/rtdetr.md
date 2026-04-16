# RT-DETR

RT-DETR (Real-Time DEtection TRansformer) for 2D object detection. Designed for real-time inference with competitive accuracy. Supports distillation and quantization for deployment optimization.

Set model.pretrained_backbone_path for backbone weights or train.pretrained_model_path for full model.

## Eval Dataset

Optional. Provides validation mAP at each checkpoint if supplied.

## Important Parameters

- **dataset.num_classes**: Number of classes. Default 80 (MSCOCO 80-class). Must match your dataset annotations.
- **model.backbone**: Default resnet_50. Supported: ResNet variants, ConvNeXt, FAN, EfficientViT. RT-DETR is optimized for real-time with lighter backbones.
- **train.optim.lr**: Learning rate. Default 1e-4 (lower than DINO's 2e-4). lr_backbone defaults to 1e-5.
- **dataset.augmentation.train_spatial_size**: Training input size. Default [640, 640]. Smaller than DINO's multi-scale (up to 1333). Key to RT-DETR's speed.
- **model.num_feature_levels**: Default 3 (vs DINO's 4). return_interm_indices is [1,2,3].
- **train.enable_ema**: Exponential moving average. Default False. Enable for potentially smoother convergence.
- **dataset.remap_mscoco_category**: Default False. Set True only for original MSCOCO dataset with 91-to-80 category ID remapping.

## Multi-GPU / Multi-Node

**Launch method:** `torchrun` (LIGHTNING_EXCLUDED_NETWORK). The entrypoint runs `torchrun --nnodes=N --nproc-per-node=M train.py`, NOT plain `python`.

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs per node | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- `CUDA_VISIBLE_DEVICES` is explicitly set (unlike Lightning-managed models which use `TAO_VISIBLE_DEVICES`)
- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` supported, forces FP16

**Multi-node env vars** (set by orchestrator):

| Variable | Purpose |
|----------|---------|
| `WORLD_SIZE` | Number of nodes (triggers multinode mode) |
| `NODE_RANK` | This node's rank (0-indexed) |
| `MASTER_ADDR` | Rank-0 node IP |
| `MASTER_PORT` | Rank-0 port (default 29500) |
| `NUM_GPU_PER_NODE` | GPUs per node (default: all visible) |

**CRITICAL:** `NODE_RANK` is copied to `RANK` if `RANK` is unset. This is required for torchrun multinode.

## Export / TRT Defaults

- Export input: 640x640, opset 17
- TRT data types: FP32, FP16, INT8
- TRT workspace: 1024 MB
- TRT max_batch_size: 4

## Distillation

RT-DETR supports knowledge distillation with a teacher model. Requires `distill` action with teacher model path and distillation bindings configuration.

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. RT-DETR is more memory-efficient than DINO/GDINO due to smaller input size (640x640) and fewer feature levels. Trains well on single GPU for small-medium datasets.

## Error Patterns

**CUDA out of memory**: Reduce batch_size. RT-DETR at 640x640 is lighter than DINO at 1333px, but batch_size > 8 may still OOM on 16GB GPUs.

**num_classes mismatch**: RT-DETR defaults to 80 (not 91 like DINO). Ensure dataset.num_classes matches your annotation categories.

**return_interm_indices vs num_feature_levels**: Default is [1,2,3] with num_feature_levels=3. Must be consistent if changed.
