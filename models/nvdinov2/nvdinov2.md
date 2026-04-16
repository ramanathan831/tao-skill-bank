# NVDINOv2

NVDINOv2 for self-supervised visual representation learning. Trains vision transformers via self-distillation (teacher-student) without labels. Produces general-purpose visual features.

Set train.pretrained_model_path for pretrained ViT weights.

## Eval Dataset

Optional. SSL training does not use labels. Evaluation is downstream task-specific.

## Important Parameters

- **model.backbone.teacher_type**: Teacher ViT variant. Default vit_l (ViT-Large).
- **model.backbone.student_type**: Student ViT variant. Default vit_l. Typically matches teacher.
- **model.backbone.img_size**: Input image size. Default 518. Higher resolution produces better features but costs more memory.
- **model.backbone.patch_size**: ViT patch size. Default 14.
- **dataset.batch_size**: Per-GPU batch size. Default 4. SSL training is memory-intensive due to dual (teacher+student) forward passes.
- **train.layerwise_decay**: Layer-wise learning rate decay. Important for ViT fine-tuning.
- **train.clip_grad_norm**: Gradient clipping. Important for stable SSL training.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Strategy: `auto` (Lightning picks best strategy automatically)
- `sync_batchnorm` is always enabled — critical for SSL training with teacher-student framework
- Multi-GPU strongly recommended (4-8 GPUs) for meaningful SSL training

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 4 GPU(s), recommended 8 GPU(s). 40GB+ (A100 recommended) VRAM per GPU. SSL with ViT-Large teacher+student is very memory-intensive. Requires A100 40GB+ GPUs. Multi-GPU strongly recommended.

## Error Patterns

**CUDA out of memory**: ViT-Large teacher+student with img_size=518 requires 40GB+ GPU memory. Reduce batch_size, img_size, or use smaller ViT variant.

**Slow convergence**: SSL needs many epochs. Default 10 is for quick testing; production runs typically use 100+ epochs.
