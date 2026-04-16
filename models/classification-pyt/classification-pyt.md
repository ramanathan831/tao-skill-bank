# Classification PyT

PyTorch image classification. Supports a wide range of backbones (FAN, EfficientNet, ResNet, etc.) with distillation and quantization for deployment.

Set model.backbone.pretrained_backbone_path for backbone weights or train.pretrained_model_path for full model.

## Eval Dataset

Optional. Validation images are provided as a separate tar alongside training images.

## Important Parameters

- **dataset.num_classes**: Number of classes. Default 20. Must match the number of subdirectories in your image tarballs.
- **model.backbone.type**: Default fan_small_12_p4_hybrid. Supported backbones and their head in_channels (from model_params_mapping.py): FAN: fan_tiny, fan_small_12_p4_hybrid, fan_base_16_p4_hybrid, fan_large_16_p4_hybrid. GCViT: gcvit_tiny through gcvit_large. FasterViT: fastervit_0 through fastervit_6. ViT/EVA/DINO: vit_large_patch14_dinov2, eva02_large_patch14, etc. SigLIP-CLIPA: ViT-H-14-SigLIP-CLIPA-224, etc. Some backbones require non-default input resolution (384, 512, 768).
- **dataset.classes_file**: Path to classes.txt listing class names.
- **train.optim.lr**: Learning rate. Default 6e-5.
- **dataset.img_size**: Input image size. Default 224.
- **dataset.batch_size**: Per-GPU batch size. Default 8.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Multi-GPU strategy: `ddp_find_unused_parameters_true`
- No fsdp support

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. Classification is generally lightweight. Most backbones at 224x224 fit well on 16GB GPUs with batch_size=8.

## Error Patterns

**CUDA out of memory**: Reduce batch_size or use a smaller backbone.

**num_classes mismatch**: Ensure dataset.num_classes matches the actual class directories in your image tarballs and classes.txt.

**Empty class directory**: Every class in classes.txt must have at least one image in the corresponding subdirectory.
