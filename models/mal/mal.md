# MAL

MAL (Minimal Annotation Learning) for weakly-supervised segmentation. Produces segmentation masks from minimal annotations (e.g., point or box annotations). Uses ViT-MAE backbone.

Set train.pretrained_model_path for ViT-MAE pretrained weights.

## Eval Dataset

Optional. Val images and annotations configured alongside train paths.

## Important Parameters

- **model.arch**: ViT-MAE backbone variant. Default vit-mae-base/16. Options include vit-mae-large/16 and other ViT-MAE variants.
- **train.lr**: Learning rate. Default 1e-6 (very low — fine-tuning ViT).
- **model.crop_size**: Training crop size. Default 512.
- **train.warmup_epochs**: Warmup epochs before full learning rate.
- **model.load_mask**: Whether to load pre-computed masks.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Multi-GPU strategy: `ddp_find_unused_parameters_true`
- No fsdp support
- **LR auto-scaling:** `lr = lr * num_devices * batch_size` (learning rate is scaled automatically by device count and batch size)

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. ViT-MAE backbone at crop_size=512 needs 24GB+ GPU memory.

## Error Patterns

**CUDA out of memory**: Reduce model.crop_size (512 -> 384 -> 256) or use a smaller ViT-MAE variant (base vs large).
