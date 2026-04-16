# Deformable DETR

Deformable DETR for 2D object detection. Uses deformable attention for efficient multi-scale feature processing. Lighter than DINO with competitive accuracy.

Uses pretrained backbone weights. Set model.pretrained_backbone_path for backbone-only loading.

## Eval Dataset

Optional. If provided, validation mAP is computed at each checkpoint interval.

## Important Parameters

- **dataset.num_classes**: Number of object classes. Default 91 (COCO). Must match annotations.
- **model.backbone**: Default resnet_50. Supported: resnet_50, gcvit_tiny, gcvit_small, gcvit_base, gcvit_large, gcvit_large_384 (more limited than DINO).
- **train.optim.lr**: Learning rate. Default 2e-4 (AdamW). lr_backbone is 2e-5.
- **train.optim.lr_steps**: MultiStep LR schedule. Default [40]. For short runs, set to match ~80% of total epochs.
- **model.num_queries**: Number of object queries. Default 300. Valid range 100-900.
- **model.dropout_ratio**: Dropout in transformer layers. Default 0.3 (higher than DINO's 0.0). Reduce for large datasets, increase for small datasets.
- **model.dim_feedforward**: FFN hidden dim. Default 1024 (vs DINO's 2048). Increasing improves capacity but costs memory.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Same DDP/FSDP behavior as DINO. Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars set by orchestrator.

## Export / TRT Defaults

- Export input: 640x640, opset 17
- TRT data types: FP32, FP16, INT8
- TRT workspace: 1024 MB
- TRT max_batch_size: 1

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. Slightly lighter than DINO due to smaller FFN. batch_size=4 fits on most 16GB+ GPUs.

## Error Patterns

**CUDA out of memory**: Reduce batch_size (4 -> 2 -> 1).

**num_select must be < num_queries * num_classes**: Same constraint as DINO.

**return_interm_indices length must match num_feature_levels**: Default [1,2,3,4] with num_feature_levels=4.

**Dataset size smaller than total batch size**: Reduce batch_size or num_gpus.
