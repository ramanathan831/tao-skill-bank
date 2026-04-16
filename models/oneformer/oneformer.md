# OneFormer

OneFormer for universal image segmentation. Unifies panoptic, instance, and semantic segmentation with a single architecture using task-conditioned queries.

Set train.pretrained_backbone and/or train.pretrained_model.

## Eval Dataset

Optional. Val data configured alongside train in the dataset config.

## Important Parameters

- **model.sem_seg_head.num_classes**: Number of segmentation classes. Default 133 (COCO panoptic).
- **model.backbone.name**: Default D2SwinTransformer (Swin-based). embed_dim=192, depths=[2,2,18,2] by default.
- **train.num_epochs**: Default 50 — significantly higher than most TAO models. OneFormer needs more epochs for convergence.
- **train.optim.lr**: Learning rate. Default 1e-5. Lower than Mask2Former's 2e-4.
- **model.task_toggling**: Enable/disable specific tasks: semantic_on, instance_on, panoptic_on.
- **export.task**: Export task mode. Options: semantic, instance, panoptic. Default semantic. Export input defaults to 640x640.
- **inference.mode**: Inference mode. Options: semantic, instance, panoptic. Default semantic. image_size defaults to [1024, 1024].
- **evaluate.iou_per_class**: Report per-class IoU in evaluation. Default True.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Uses explicit `DDPStrategy` with `find_unused_parameters=True`, `gradient_as_bucket_view=True`, `process_group_backend="nccl"`
- `sync_batchnorm` is always enabled
- No fsdp support — DDP only

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 2 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. OneFormer is memory-intensive like Mask2Former. batch_size=1 is the default. Multi-GPU needed for reasonable training speed, especially with 50 epochs.

## Error Patterns

**CUDA out of memory**: batch_size is already 1. Reduce image resolution or use a smaller Swin configuration.

**Slow training**: 50 default epochs with batch_size=1 is slow on single GPU. Use multi-GPU distributed training.
