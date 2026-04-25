# DINO

DINO (DETR with Improved DeNoising Anchor Boxes) for 2D object detection. Transformer-based detector with denoising training, multi-scale features, and optional distillation support.

Uses pretrained backbone weights (e.g. ResNet-50 ImageNet). Set `model.pretrained_backbone_path` for backbone-only or `train.pretrained_model_path` for full model.

## Training Requirements

The agent MUST read this section before generating any training or AutoML script for DINO.

- **Dataset type:** object_detection
- **Formats:** coco, coco_raw
- **Accepted dataset intents:** training, evaluation, testing, calibration
- **Monitoring metric:** val_mAP50

**Required datasets — MUST prompt the user for both:**

| Dataset | Required | Why |
|---|---|---|
| Train dataset URI | Yes | Training data (COCO format) |
| Validation dataset URI | **Yes — ALWAYS** | DINO unconditionally builds a val dataloader. Omitting `val_data_sources` causes `FileNotFoundError` at startup regardless of the metric or workflow. If the user has no separate eval split, reuse the train URI. |

**Required user prompts before generating any training spec:**

1. **Train dataset URI** — S3 path to COCO-format training data
2. **Validation dataset URI** — S3 path to COCO-format val data (can be same as train)
3. **`image_dir` format** — Is the image data a folder of individual files (`images/`) or a tar.gz archive (`images.tar.gz`)? Using the wrong format causes `FileNotFoundError` with misleading paths like `/mnt/lustre/.../images/001762.jpg`. Check the dataset layout if unsure.
4. **`num_classes`** — How many object classes? Default 91 (COCO). Must be >= `max(category_id) + 1`. Too low causes `CUDA error: device-side assert triggered`.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| distill | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| distill | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| evaluate | dataset.test_data_sources.image_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.test_data_sources.json_file | eval_dataset | annotations.json | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |
| inference | dataset.infer_data_sources.image_dir | inference_dataset | images.tar.gz | Yes |
| inference | dataset.infer_data_sources.classmap | inference_dataset | label_map.txt | No |
| quantize | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| quantize | dataset.quant_calibration_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | No |
| train | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |
| train | dataset.val_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations.json | Yes |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — DINO's `config.json` has empty `data_sources` because the runner cannot auto-resolve array-of-objects spec keys (see Internal Details). The agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "aws://bucket/data/train"
S3_VAL = "aws://bucket/data/val"    # can be same as S3_TRAIN
S3_EVAL = "aws://bucket/data/eval"  # for evaluate/inference
# CRITICAL: use "images" (folder) or "images.tar.gz" (archive) — ASK the user
IMG = "images"  # or "images.tar.gz"
```

**train (mandatory):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMG}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMG}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.num_classes": "<num_classes> + 1",
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.test_data_sources.image_dir": f"{S3_EVAL}/{IMG}",
    "dataset.test_data_sources.json_file": f"{S3_EVAL}/annotations.json",
    "dataset.num_classes": "<num_classes> + 1",
}
```

**export:**
```python
{
    "dataset.num_classes": "<num_classes> + 1",
}
```

**gen_trt_engine (mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/{IMG}"],
    "gen_trt_engine.tensorrt.data_type": "FP16",
    "dataset.num_classes": "<num_classes> + 1",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.infer_data_sources.image_dir": [f"{S3_EVAL}/{IMG}"],
    "dataset.infer_data_sources.classmap": f"{S3_EVAL}/label_map.txt",
    "dataset.num_classes": "<num_classes> + 1",
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMG}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMG}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.quant_calibration_data_sources": {
        "image_dir": f"{S3_TRAIN}/{IMG}", "json_file": f"{S3_TRAIN}/annotations.json"
    },
    "dataset.num_classes": "<num_classes> + 1",
}
```

**distill (mandatory data sources):**
```python
{
    "dataset.train_data_sources": [
        {"image_dir": f"{S3_TRAIN}/{IMG}", "json_file": f"{S3_TRAIN}/annotations.json"}
    ],
    "dataset.val_data_sources": [
        {"image_dir": f"{S3_VAL}/{IMG}", "json_file": f"{S3_VAL}/annotations.json"}
    ],
    "dataset.num_classes": "<num_classes> + 1",
}
```

## Dataset

COCO JSON format. train_data_sources and val_data_sources are lists supporting multiple data source entries. Each entry has image_dir and json_file (COCO annotations JSON).

**`image_dir` format**: `image_dir` can be either:
- A **folder** containing individual images (e.g. `aws://bucket/data/images`) — the SDK mounts/downloads the folder as-is.
- A **tar.gz archive** of images (e.g. `aws://bucket/data/images.tar.gz`) — the SDK extracts the archive into a folder.

Check your dataset layout before setting `image_dir`. If the S3 path contains individual `.jpg`/`.png` files, use the folder path. If it's a single `.tar.gz` file, use the tar.gz path. Using the wrong format causes `FileNotFoundError` at training time.

Supported formats: coco, coco_raw.

### Train Data Sources

- **image_dir**: `images/` (folder) or `images.tar.gz` (archive) — depends on dataset layout
- **json_file**: `annotations.json`

### Val Data Sources (ALWAYS required)

- **image_dir**: `images/` (folder) or `images.tar.gz` (archive)
- **json_file**: `annotations.json`

### Inference Data Sources

- **image_dir**: `images/` (folder) or `images.tar.gz` (archive)
- **classmap**: `label_map.txt`

## Important Parameters

- **dataset.num_classes**: Number of object classes. Default is 91 (COCO). Must be >= `max(category_id) + 1`. Too low causes `CUDA error: device-side assert triggered`.
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

**`FileNotFoundError` on images**: Wrong `image_dir` format. The dataset has a folder of individual images but the spec says `images.tar.gz`, or vice versa. Always confirm the format with the user (see Training Requirements).

**`FileNotFoundError` at startup (val)**: `val_data_sources` missing or pointing to non-existent data. DINO unconditionally builds a val dataloader — this is required even when only optimizing `train_loss`.

**`CUDA device-side assert`**: `num_classes` too low. Set `num_classes >= max(category_id) + 1`.

**`config.json` has empty `"inputs": {}`**: The SDK's script_runner won't download S3 data — the container sees raw `aws://...` URIs as filesystem paths. Verify `config.json` declares `inputs` with `[0]`-indexed spec keys (see Internal Details).

## AutoML / HPO Notes

AutoML runs training — all requirements from **Training Requirements** above apply. The agent must read that section first.

**Recommended metric:** `metric="kpi"` with `direction="maximize"` — mAP is emitted during validation.

**Recommended hyperparameters:**

```python
automl_hyperparameters=[
    "train.optim.lr",
    "train.optim.weight_decay",
    "model.backbone",
    "model.num_queries",
    "model.dropout_ratio",
]
custom_param_ranges={
    "train.optim.lr": {"valid_min": 1e-5, "valid_max": 5e-4},
    "model.num_queries": {"valid_min": 100, "valid_max": 900},
    "model.dropout_ratio": {"valid_min": 0.0, "valid_max": 0.3},
}
```

`train.optim.weight_decay` is not in the default DINO spec schema — the runner accepts it with a warning. It still works; the DINO training code picks it up from the config.

**Backbone constraint for AutoML:** The LLM brain may propose backbone names not in the supported list (see Important Parameters above), e.g. `fan_small`, `fan_tiny`, `efficientvit_b2`. These cause training failures. Use `custom_param_ranges` to constrain categorical params when possible.

### Internal Details

#### defaults-train.json

DINO ships without `references/spec_template_train.yaml`. You must create `~/tao-skills-external/models/dino/defaults-train.json` from `tao-pytorch/nvidia_tao_pytorch/cv/dino/experiment_specs/train.yaml` (convert YAML to JSON, replace `"???"` placeholders with empty strings).

#### Data Sources Gap

DINO's `config.json` has `"data_sources": {}` (empty). The runner's `_apply_data_sources()` only handles flat spec keys (like cosmos-rl's `custom.train_dataset.annotation_path`), but DINO's data sources are **arrays of objects** (`dataset.train_data_sources[{image_dir, json_file}]`). The tao-core microservices config (`tao-core/nvidia_tao_core/microservices/handlers/network_configs/dino.config.json`) has the full mapping using a `mapping` sub-structure, but the runner doesn't support that format.

**Consequence:** The runner cannot auto-resolve data URIs for DINO. Data paths MUST be set manually via `spec_overrides` (see Training Requirements above). The skill's `config.json` instead declares `inputs` in the train action with `[0]`-indexed spec keys so the SDK's script_runner downloads S3 data at runtime:

```json
"inputs": {
    "dataset.train_data_sources[0].image_dir": {"type": "folder"},
    "dataset.train_data_sources[0].json_file": {"type": "file"},
    "dataset.val_data_sources[0].image_dir": {"type": "folder"},
    "dataset.val_data_sources[0].json_file": {"type": "file"}
}
```

All model-specific metadata (dataset type, formats, metrics, required datasets) is documented in the **Training Requirements** section above.

**TODO:** Extend the runner's `_apply_data_sources()` to handle the `mapping` sub-structure from tao-core so DINO can use auto-resolved data sources like cosmos-rl does.
