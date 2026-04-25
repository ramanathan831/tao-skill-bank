# Optical Inspection

Optical inspection for defect detection using Siamese networks. Compares image pairs to detect manufacturing defects, anomalies, or quality issues.

Set train.pretrained_model_path for pretrained Siamese weights.

## Training Requirements

- **Dataset type:** optical_inspection
- **Formats:** default
- **Monitoring metric:** val_acc

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_dataset.images_dir | eval_dataset | images.tar.gz | No |
| evaluate | dataset.test_dataset.csv_path | eval_dataset | dataset.csv | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | images.tar.gz | Yes |
| inference | dataset.infer_dataset.images_dir | inference_dataset | images.tar.gz | No |
| inference | dataset.infer_dataset.csv_path | inference_dataset | dataset.csv | No |
| train | dataset.train_dataset.images_dir | train_datasets | images.tar.gz | No |
| train | dataset.train_dataset.csv_path | train_datasets | dataset.csv | No |
| train | dataset.validation_dataset.images_dir | eval_dataset | images.tar.gz | No |
| train | dataset.validation_dataset.csv_path | eval_dataset | dataset.csv | No |
| train | dataset.test_dataset.images_dir | eval_dataset | images.tar.gz | No |
| train | dataset.test_dataset.csv_path | eval_dataset | dataset.csv | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "aws://bucket/data/train"
S3_EVAL = "aws://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
    "dataset.train_dataset.csv_path": f"{S3_TRAIN}/dataset.csv",
    "dataset.validation_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.validation_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
    "dataset.test_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.test_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**gen_trt_engine (mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.data_type": "fp16",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/images.tar.gz"],
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.test_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.test_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.infer_dataset.images_dir": f"{S3_EVAL}/images.tar.gz",
    "dataset.infer_dataset.csv_path": f"{S3_EVAL}/dataset.csv",
}
```
## Eval Dataset

Optional. Eval dataset uses same format (images + CSV).

## Important Parameters

- **model.model_type**: Siamese variant. Options include Siamese, Siamese_3.
- **model.model_backbone**: Default custom.
- **model.embedding_vectors**: Number of embedding dimensions. Default 5.
- **train.optim.lr**: Learning rate. Default 5e-4.
- **dataset.num_input**: Number of input images per comparison.
- **dataset.input_map**: Mapping of input channels / image pairs.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node only
- Lightweight Siamese network, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. Siamese networks for inspection are lightweight. Single GPU sufficient.

## Error Patterns

**CSV format error**: Ensure dataset.csv has the correct column format for image pair paths and labels.
