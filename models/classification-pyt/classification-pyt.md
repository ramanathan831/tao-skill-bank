# Classification PyT

PyTorch image classification. Supports a wide range of backbones (FAN, EfficientNet, ResNet, etc.) with distillation and quantization for deployment.

Set model.backbone.pretrained_backbone_path for backbone weights or train.pretrained_model_path for full model.

## Training Requirements

- **Dataset type:** image_classification
- **Formats:** classification_pyt
- **Monitoring metric:** val_acc_1

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| distill | dataset.train_dataset.images_dir | train_datasets | images_train.tar.gz | No |
| distill | dataset.classes_file | train_datasets | classes.txt | No |
| distill | dataset.val_dataset.images_dir | eval_dataset | images_val.tar.gz | No |
| evaluate | dataset.val_dataset.images_dir | eval_dataset | images_val.tar.gz | No |
| evaluate | dataset.classes_file | eval_dataset | classes.txt | No |
| evaluate | dataset.test_dataset.images_dir | inference_dataset | images_test.tar.gz | No |
| export | dataset.root_dir | train_datasets |  | No |
| inference | dataset.val_dataset.images_dir | eval_dataset | images_val.tar.gz | No |
| inference | dataset.classes_file | eval_dataset | classes.txt | No |
| inference | dataset.test_dataset.images_dir | inference_dataset | images_test.tar.gz | No |
| quantize | dataset.train_dataset.images_dir | train_datasets | images_train.tar.gz | No |
| quantize | dataset.classes_file | train_datasets | classes.txt | No |
| quantize | dataset.val_dataset.images_dir | eval_dataset | images_val.tar.gz | No |
| quantize | dataset.quant_calibration_dataset.images_dir | calibration_dataset | images_train.tar.gz | No |
| train | dataset.train_dataset.images_dir | train_datasets | images_train.tar.gz | No |
| train | dataset.classes_file | train_datasets | classes.txt | No |
| train | dataset.val_dataset.images_dir | eval_dataset | images_val.tar.gz | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "aws://bucket/data/train"
S3_EVAL = "aws://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 2,
    "train.validation_interval": 2,
    "train.checkpoint_interval": 2,
    "train.num_gpus": 1,
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
    "dataset.classes_file": f"{S3_TRAIN}/classes.txt",
    "dataset.val_dataset.images_dir": f"{S3_EVAL}/images_val.tar.gz",
}
```

**export (mandatory data sources):**
```python
{
    "export.input_height": 224,
    "export.input_width": 224,
    "dataset.root_dir": f"{S3_TRAIN}",
}
```

**gen_trt_engine:**
```python
{
    "gen_trt_engine.tensorrt.data_type": "fp16",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.batch_size": 1,
    "dataset.val_dataset.images_dir": f"{S3_EVAL}/images_val.tar.gz",
    "dataset.classes_file": f"{S3_EVAL}/classes.txt",
    "dataset.test_dataset.images_dir": f"{S3_EVAL}/images_test.tar.gz",
}
```

**distill (mandatory data sources):**
```python
{
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
    "dataset.classes_file": f"{S3_TRAIN}/classes.txt",
    "dataset.val_dataset.images_dir": f"{S3_EVAL}/images_val.tar.gz",
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.val_dataset.images_dir": f"{S3_EVAL}/images_val.tar.gz",
    "dataset.classes_file": f"{S3_EVAL}/classes.txt",
    "dataset.test_dataset.images_dir": f"{S3_EVAL}/images_test.tar.gz",
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
    "dataset.classes_file": f"{S3_TRAIN}/classes.txt",
    "dataset.val_dataset.images_dir": f"{S3_EVAL}/images_val.tar.gz",
    "dataset.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
}
```
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
