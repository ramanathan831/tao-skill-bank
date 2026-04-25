# OCRNet

OCRNet for scene text recognition. Recognizes text content from cropped text region images. Supports CTC and attention-based decoders.

Set train.pretrained_model_path for pretrained OCR weights.

## Training Requirements

- **Dataset type:** ocrnet
- **Formats:** default
- **Monitoring metric:** val_acc

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| dataset_convert | dataset_convert.input_img_dir | id |  | No |
| dataset_convert | dataset_convert.gt_file | id |  | No |
| evaluate | dataset.character_list_file | eval_dataset | character_list | No |
| evaluate | evaluate.test_dataset_dir | eval_dataset | results/{dataset_convert_job_id}/dataset_convert/lmdb | No |
| export | dataset.character_list_file | eval_dataset | character_list | No |
| gen_trt_engine | gen_trt_engine.tensorrt.calibration.cal_image_dir | calibration_dataset |  | Yes |
| inference | dataset.character_list_file | eval_dataset | character_list | No |
| inference | inference.inference_dataset_dir | eval_dataset | test.tar.gz | No |
| prune | dataset.character_list_file | eval_dataset | character_list | No |
| quantize | dataset.train_dataset_dir | train_datasets | results/{dataset_convert_job_id}/dataset_convert/lmdb | Yes |
| quantize | dataset.val_dataset_dir | eval_dataset | results/{dataset_convert_job_id}/dataset_convert/lmdb | No |
| quantize | dataset.character_list_file | eval_dataset | character_list | No |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets | train.tar.gz | No |
| retrain | dataset.train_dataset_dir | train_datasets | results/{dataset_convert_job_id}/dataset_convert/lmdb | Yes |
| retrain | dataset.val_dataset_dir | eval_dataset | results/{dataset_convert_job_id}/dataset_convert/lmdb | No |
| retrain | dataset.character_list_file | eval_dataset | character_list | No |
| train | dataset.train_dataset_dir | train_datasets | results/{dataset_convert_job_id}/dataset_convert/lmdb | Yes |
| train | dataset.val_dataset_dir | eval_dataset | results/{dataset_convert_job_id}/dataset_convert/lmdb | No |
| train | dataset.character_list_file | eval_dataset | character_list | No |

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
    "dataset.batch_size": 16,
    "dataset.train_dataset_dir": [f"{S3_TRAIN}/results/{dataset_convert_job_id}/dataset_convert/lmdb"],
    "dataset.val_dataset_dir": f"{S3_EVAL}/results/{dataset_convert_job_id}/dataset_convert/lmdb",
    "dataset.character_list_file": f"{S3_EVAL}/character_list",
}
```

**gen_trt_engine (mandatory data sources):**
```python
{
    "gen_trt_engine.tensorrt.data_type": "fp16",
    "gen_trt_engine.tensorrt.calibration.cal_image_dir": [f"{S3_TRAIN}"],
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.character_list_file": f"{S3_EVAL}/character_list",
    "evaluate.test_dataset_dir": f"{S3_EVAL}/results/{dataset_convert_job_id}/dataset_convert/lmdb",
}
```

**export (mandatory data sources):**
```python
{
    "dataset.character_list_file": f"{S3_EVAL}/character_list",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.character_list_file": f"{S3_EVAL}/character_list",
    "inference.inference_dataset_dir": f"{S3_EVAL}/test.tar.gz",
}
```

**prune (mandatory data sources):**
```python
{
    "dataset.character_list_file": f"{S3_EVAL}/character_list",
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_dataset_dir": [f"{S3_TRAIN}/results/{dataset_convert_job_id}/dataset_convert/lmdb"],
    "dataset.val_dataset_dir": f"{S3_EVAL}/results/{dataset_convert_job_id}/dataset_convert/lmdb",
    "dataset.character_list_file": f"{S3_EVAL}/character_list",
    "dataset.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/train.tar.gz",
}
```

**retrain (mandatory data sources):**
```python
{
    "dataset.train_dataset_dir": [f"{S3_TRAIN}/results/{dataset_convert_job_id}/dataset_convert/lmdb"],
    "dataset.val_dataset_dir": f"{S3_EVAL}/results/{dataset_convert_job_id}/dataset_convert/lmdb",
    "dataset.character_list_file": f"{S3_EVAL}/character_list",
}
```
## Eval Dataset

Optional. Test data provided as separate tarball.

## Important Parameters

- **dataset.character_list_file**: Path to character list defining the supported character set. This determines the output vocabulary size.
- **model.backbone**: Default ResNet.
- **model.prediction**: Decoder type. CTC or Attn (attention-based).
- **train.optim.lr**: Learning rate. Default 1.0 (Adadelta optimizer). High default is specific to Adadelta.
- **dataset.batch_size**: Per-GPU batch size. Default 16.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.distributed_strategy` | Strategy name | `auto` |

- Strategy: `auto` for single-GPU, reads `train.distributed_strategy` from config when multi-GPU
- No explicit `num_nodes` in train script — single-node oriented
- Lightweight model, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. OCR text recognition is lightweight. Single GPU is typically sufficient.

## Error Patterns

**dataset_convert required**: If using raw images + gt files, run dataset_convert first to produce LMDB format.

**Character list mismatch**: All characters in training data must be present in the character_list file.
