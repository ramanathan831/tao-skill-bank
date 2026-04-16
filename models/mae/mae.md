# MAE

MAE (Masked Autoencoder) for self-supervised pretraining and fine-tuning. Masks random patches and reconstructs them to learn visual representations. Supports pretrain and finetune stages.

Set train.pretrained_model_path for pretrained MAE weights when fine-tuning.

## Eval Dataset

Optional. Pretraining does not need eval data. Fine-tuning optionally uses val set.

## Important Parameters

- **train.stage**: Training stage. Options: pretrain, finetune. Pretrain learns representations via masking. Finetune adds a classification head.
- **model.arch**: Architecture. Default convnextv2_base. Wide range of options including ConvNeXt, Hiera, ViT variants.
- **model.num_classes**: Number of classes for fine-tuning. Default 1000 (ImageNet). Only relevant in finetune stage.
- **model.mask_ratio**: Fraction of patches to mask during pretraining. Typically 0.75.
- **model.norm_pix_loss**: Whether to normalize pixel values in reconstruction loss.
- **train.optim.lr**: Learning rate. Default 2e-4.
- **dataset.augmentation**: Augmentation settings including mixup, cutmix for fine-tuning.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- `ddp` uses `find_unused_parameters=True`
- `fsdp` forces FP16
- Multi-GPU strongly recommended for pretraining (large batch sizes needed)

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 2 GPU(s), recommended 8 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. MAE pretraining benefits from large batch sizes across many GPUs. Fine-tuning is more modest in resource requirements.

## Error Patterns

**Stage mismatch**: Ensure train.stage matches your intent (pretrain vs finetune). Fine-tuning without a pretrained_model_path trains from scratch.

**num_classes mismatch (finetune only)**: Ensure model.num_classes matches your dataset class count when fine-tuning.
