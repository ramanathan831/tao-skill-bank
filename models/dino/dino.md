# DINO

DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with denoising training, multi-scale features, and optional distillation support.

Uses pretrained backbone weights (e.g. ResNet-50 ImageNet). Set model.pretrained_backbone_path for backbone-only or train.pretrained_model_path for full model.

## Eval Dataset

Optional. Eval dataset is optional. If provided, validation mAP is computed each epoch. Uses the same COCO format as training data.

## Important Parameters

- **dataset.num_classes**: Number of object classes. Default is 91 (COCO). Must match the category count in your annotations JSON.
- **model.backbone**: Backbone architecture. Default resnet_50. Supported: resnet_34, resnet_50, fan_small_12_p4_hybrid, fan_base_16_p4_hybrid, fan_large_16_p4_hybrid, gcvit_tiny, gcvit_small, gcvit_base, gcvit_large, nvdinov2_vit_large_legacy, swin_tiny_224_1k, swin_small_224_1k, swin_base_224_22k, swin_large_224_22k, efficientvit_l2_224, efficientvit_l2_384.
- **train.optim.lr**: Learning rate. Default 2e-4 (AdamW). lr_backbone defaults to 2e-5 (10x lower). Reduce both if training diverges.
- **train.num_epochs**: DINO typically needs 30-50+ epochs for good mAP on real datasets. The default of 10 is suitable for quick iteration.
- **train.optim.lr_steps**: MultiStep LR decay schedule. Default [11]. For longer training, set to e.g. [30, 40] for a 50-epoch run.
- **model.num_queries**: Number of object queries. Default 300. Increase for dense scenes with many objects per image. num_select must be < num_queries * num_classes.
- **dataset.batch_size**: Per-GPU batch size. Default 4. Reduce to 2 if OOM on 16GB GPUs. Total batch = batch_size * num_gpus.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers internally).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- When `num_gpus > 1` with activation checkpointing: uses `ddp` (find_unused_parameters=False)
- When `num_gpus > 1` without activation checkpointing: uses `ddp_find_unused_parameters_true`
- `fsdp` forces precision to FP16
- FAN backbones auto-enable `sync_batchnorm`

**Multi-node env vars** (set by orchestrator, not user):

| Variable | Purpose |
|----------|---------|
| `WORLD_SIZE` | Number of nodes (triggers multinode mode) |
| `NODE_RANK` | This node's rank (0-indexed) |
| `MASTER_ADDR` | Rank-0 node IP |
| `MASTER_PORT` | Rank-0 port (default 29500) |
| `NUM_GPU_PER_NODE` | GPUs per node (default: all visible) |

## Export / TRT Defaults

- Export input: 640x640, opset 17
- TRT data types: FP32, FP16, INT8
- TRT workspace: 1024 MB
- TRT max_batch_size: 1

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Transformer-based detection is memory-intensive. batch_size=4 fits on 24GB GPUs. For 16GB GPUs, reduce to batch_size=2. Multi-GPU with 4+ GPUs recommended for datasets > 10k images.

## Error Patterns

**CUDA out of memory**: Reduce dataset.batch_size (4 -> 2 -> 1). DINO uses multi-scale features that consume significant GPU memory, especially with high-resolution images (default max 1333px).

**num_select must be < num_queries * num_classes**: Ensure model.num_select (default 300) is less than num_queries * dataset.num_classes.

**Error merging spec.yaml with schema**: Hydra/OmegaConf validation error. num_epochs and num_gpus must be under 'train.*', not at spec root. Use the SDK spec_shorthand_keys mapping.

**Dataset size smaller than total batch size**: Total batch = batch_size * num_gpus. If val dataset has fewer samples, reduce dataset.batch_size or num_gpus. The agent should proactively check this.

**return_interm_indices length must match num_feature_levels**: Default is [1,2,3,4] with num_feature_levels=4. If changing one, update the other.
