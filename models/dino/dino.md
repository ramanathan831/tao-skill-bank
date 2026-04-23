# DINO

DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with denoising training, multi-scale features, and optional distillation support.

Uses pretrained backbone weights (e.g. ResNet-50 ImageNet). Set `model.pretrained_backbone_path` for backbone-only or `train.pretrained_model_path` for full model.

## Eval Dataset

Eval dataset is optional. If provided, validation mAP is computed each epoch. Uses the same COCO format as training data.

## Dataset

COCO JSON format. train_data_sources and val_data_sources are lists supporting multiple data source entries. Each entry has image_dir (tar.gz of images) and json_file (COCO annotations JSON).

Supported formats: coco, coco_raw.

### Train Data Sources

- **image_dir**: `images.tar.gz`
- **json_file**: `annotations.json`

### Inference Data Sources

- **image_dir**: `images.tar.gz`
- **classmap**: `label_map.txt`

## Important Parameters

- **dataset.num_classes**: Number of object classes. Default is 91 (COCO). Must match the category count in your annotations JSON.
- **model.backbone**: Backbone architecture. Default resnet_50. Supported: resnet_34, resnet_50, fan_small_12_p4_hybrid, fan_base_16_p4_hybrid, fan_large_16_p4_hybrid, gcvit_tiny, gcvit_small, gcvit_base, gcvit_large, nvdinov2_vit_large_legacy, swin_tiny_224_1k, swin_small_224_1k, swin_base_224_22k, swin_large_224_22k, efficientvit_l2_224, efficientvit_l2_384.
- **train.optim.lr**: Learning rate. Default 2e-4 (AdamW). lr_backbone defaults to 2e-5 (10x lower). Reduce both if training diverges.
- **train.num_epochs**: DINO typically needs 30-50+ epochs for good mAP on real datasets. The default of 10 is suitable for quick iteration.
- **train.optim.lr_steps**: MultiStep LR decay schedule. Default [11]. For longer training, set to e.g. [30, 40] for a 50-epoch run.
- **model.num_queries**: Number of object queries. Default 300. Increase for dense scenes with many objects per image. num_select must be < num_queries * num_classes.
- **dataset.batch_size**: Per-GPU batch size. Default 4. Reduce to 2 if OOM on 16GB GPUs. Total batch = batch_size * num_gpus.

## Default Values

- **num_epochs**: `10`
- **batch_size**: `4`
- **learning_rate**: `2e-4`
- **lr_backbone**: `2e-5`
- **num_classes**: `91`
- **backbone**: `resnet_50`

## Export Defaults

- **input_width**: `640`
- **input_height**: `640`
- **opset_version**: `17`
- **trt_data_types**: `[FP32, FP16, INT8]`
- **trt_workspace_size_mb**: `1024`

## Hardware

- **Minimum**: 1 GPU
- **Recommended**: 4 GPUs
- **GPU Memory**: 24GB+ (A100 recommended)

Transformer-based detection is memory-intensive. batch_size=4 fits on 24GB GPUs. For 16GB GPUs, reduce to batch_size=2. Multi-GPU with 4+ GPUs recommended for datasets > 10k images.

## Error Patterns

**CUDA out of memory**: Reduce dataset.batch_size (4 -> 2 -> 1). DINO uses multi-scale features that consume significant GPU memory, especially with high-resolution images (default max 1333px).

**num_select must be < num_queries * num_classes**: Ensure model.num_select (default 300) is less than num_queries * dataset.num_classes.

**Error merging spec.yaml with schema**: Hydra/OmegaConf validation error. num_epochs and num_gpus must be under 'train.*', not at spec root. Use the SDK spec_shorthand_keys mapping.

**Dataset size smaller than total batch size**: Total batch = batch_size * num_gpus. If val dataset has fewer samples, reduce dataset.batch_size or num_gpus. The agent should proactively check this.

**return_interm_indices length must match num_feature_levels**: Default is [1,2,3,4] with num_feature_levels=4. If changing one, update the other.

## AutoML / HPO Notes

### defaults-train.json

DINO ships without `references/spec_template_train.yaml`. You must create `~/tao-skills-external/models/dino/defaults-train.json` from `tao-pytorch/nvidia_tao_pytorch/cv/dino/experiment_specs/train.yaml` (convert YAML to JSON, replace `"???"` placeholders with empty strings).

### Data Sources Workaround

DINO's `config.json` uses the `mapping` style for `data_sources`, which the runner's `_apply_data_sources` does NOT handle (it only handles `path` and `path_from_format`). You must set data paths explicitly via `spec_overrides`:

```python
S3_BASE = "s3://bucket/data/my_dataset"
spec_overrides={
    "dataset.train_data_sources": [
        {
            "image_dir": f"{S3_BASE}/images.tar.gz",
            "json_file": f"{S3_BASE}/annotations.json",
        }
    ],
    "dataset.val_data_sources": [
        {
            "image_dir": f"{S3_BASE}/images.tar.gz",
            "json_file": f"{S3_BASE}/annotations.json",
        }
    ],
    "dataset.num_classes": 91,
}
```

### num_classes Pitfall

DINO's default `num_classes=91` matches COCO. If your dataset has category IDs 1–N, you must set `num_classes` to at least `max(category_id) + 1`. Setting it too low causes `CUDA error: device-side assert triggered` (label index out of bounds). When in doubt, keep 91 — extra output neurons are harmless.

### val_data_sources Required

Even if you only have a train split, `val_data_sources` must be set to valid paths. DINO's dataloader unconditionally builds a val dataset. Reuse the train data for val if no separate eval split exists.

### Recommended AutoML Configuration

```python
automl_hyperparameters=[
    "train.optim.lr",
    "train.optim.weight_decay",
    "model.num_queries",
]
custom_param_ranges={
    "train.optim.lr": {"valid_min": 1e-5, "valid_max": 5e-4},
    "model.num_queries": {"valid_min": 100, "valid_max": 900},
}
```

The `kpi` metric (mAP) is emitted during validation — use `metric="kpi"` with `direction="maximize"`.

`train.optim.weight_decay` is not in the default DINO spec schema — the runner will accept it with a warning. It still works; the DINO training code picks it up from the config.
