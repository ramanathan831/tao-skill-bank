# OCDNet

OCDNet for scene text detection. Detects arbitrary-oriented text regions in natural images using a differentiable binarization approach.

Set train.pretrained_model_path for pretrained weights.

## Training Requirements

- **Dataset type:** ocdnet
- **Formats:** default
- **Monitoring metric:** hmean

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.validate_dataset.data_path | eval_dataset | test.tar.gz | Yes |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset | train/img.tar.gz | Yes |
| inference | inference.input_folder | eval_dataset | test/img.tar.gz | No |
| prune | dataset.validate_dataset.data_path | eval_dataset | test.tar.gz | Yes |
| quantize | dataset.train_dataset.data_path | train_datasets | train.tar.gz | Yes |
| quantize | dataset.validate_dataset.data_path | eval_dataset | test.tar.gz | Yes |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets | train/img.tar.gz | No |
| retrain | dataset.train_dataset.data_path | train_datasets | train.tar.gz | Yes |
| retrain | dataset.validate_dataset.data_path | eval_dataset | test.tar.gz | Yes |
| train | dataset.train_dataset.data_path | train_datasets | train.tar.gz | Yes |
| train | dataset.validate_dataset.data_path | eval_dataset | test.tar.gz | Yes |

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
    "dataset.train_dataset.loader.batch_size": 16,
    "dataset.train_dataset.data_path": [f"{S3_TRAIN}/train.tar.gz"],
    "dataset.validate_dataset.data_path": [f"{S3_EVAL}/test.tar.gz"],
}
```

**gen_trt_engine (mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.data_type": "INT8",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}/train/img.tar.gz"],
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.validate_dataset.data_path": [f"{S3_EVAL}/test.tar.gz"],
}
```

**inference (mandatory data sources):**
```python
{
    "inference.input_folder": f"{S3_EVAL}/test/img.tar.gz",
}
```

**prune (mandatory data sources):**
```python
{
    "dataset.validate_dataset.data_path": [f"{S3_EVAL}/test.tar.gz"],
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_dataset.data_path": [f"{S3_TRAIN}/train.tar.gz"],
    "dataset.validate_dataset.data_path": [f"{S3_EVAL}/test.tar.gz"],
    "dataset.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/train/img.tar.gz",
}
```

**retrain (mandatory data sources):**
```python
{
    "dataset.train_dataset.data_path": [f"{S3_TRAIN}/train.tar.gz"],
    "dataset.validate_dataset.data_path": [f"{S3_EVAL}/test.tar.gz"],
}
```
## Eval Dataset

Optional. Test dataset provided as separate tarball.

## Important Parameters

- **model.backbone**: Default deformable_resnet18. Deformable convolutions improve text region detection for irregular text.
- **train.optimizer.args.lr**: Learning rate. Default 0.001 (Adam).
- **postprocess.thresh**: Binarization threshold for text region extraction.
- **postprocess.box_thresh**: Box confidence threshold for filtering detections.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.distributed_strategy` | `ddp`, `fsdp`, or `deepspeed_stage_3_offload` | `ddp` |

- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` forces FP16
- **`deepspeed_stage_3_offload`** is uniquely supported for OCDNet (forces FP16)
- FAN backbones auto-enable `sync_batchnorm`

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. OCDNet is lightweight. Single GPU is sufficient for most datasets.

## Error Patterns

**Low detection rate**: Tune postprocess.thresh and box_thresh. Default thresholds may be too aggressive for some datasets.
