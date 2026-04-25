# Grounding DINO

Grounding DINO for open-set object detection. Combines DINO-style detection with BERT text encoder for language-guided detection. Detects objects described by text prompts without fixed class vocabulary.

Set train.pretrained_model_path for full Grounding DINO weights or model.pretrained_backbone_path for backbone-only.

## Training Requirements

- **Dataset type:** object_detection
- **Formats:** odvg, coco, raw
- **Monitoring metric:** val_mAP50

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_data_sources | eval_dataset | image_dir: images.tar.gz, json_file: annotations.json | No |
| inference | dataset.infer_data_sources | inference_dataset | image_dir: images.tar.gz, classmap: label_map.txt | No |
| quantize | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations_odvg.jsonl, label_map: annotations_odvg_labelmap.json | Yes |
| quantize | dataset.val_data_sources | eval_dataset | image_dir: images.tar.gz, json_file: annotations.json | No |
| quantize | dataset.quant_calibration_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations_odvg.jsonl, label_map: annotations_odvg_labelmap.json | No |
| train | dataset.train_data_sources | train_datasets | image_dir: images.tar.gz, json_file: annotations_odvg.jsonl, label_map: annotations_odvg_labelmap.json | Yes |
| train | dataset.val_data_sources | eval_dataset | image_dir: images.tar.gz, json_file: annotations.json | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "aws://bucket/data/train"
S3_EVAL = "aws://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 10,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations_odvg.jsonl", "label_map": f"{S3_TRAIN}/annotations_odvg_labelmap.json"}],
    "dataset.val_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
}
```

**gen_trt_engine:**
```python
{
    "gen_trt_engine.tensorrt.data_type": "FP16",
}
```

**inference (mandatory data sources):**
```python
{
    "dataset.infer_data_sources.captions": [
        "person"
    ],
    "dataset.infer_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "classmap": f"{S3_EVAL}/label_map.txt"},
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.test_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_data_sources": [{"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations_odvg.jsonl", "label_map": f"{S3_TRAIN}/annotations_odvg_labelmap.json"}],
    "dataset.val_data_sources": {"image_dir": f"{S3_EVAL}/images.tar.gz", "json_file": f"{S3_EVAL}/annotations.json"},
    "dataset.quant_calibration_data_sources": {"image_dir": f"{S3_TRAIN}/images.tar.gz", "json_file": f"{S3_TRAIN}/annotations_odvg.jsonl", "label_map": f"{S3_TRAIN}/annotations_odvg_labelmap.json"},
}
```
## Eval Dataset

Optional. Validation uses COCO-format annotations for mAP even though training can use ODVG format.

## Important Parameters

- **model.backbone**: Default swin_tiny_224_1k. Also supports resnet_50 and other Swin variants. Swin generally performs better for grounding tasks.
- **model.text_encoder_type**: BERT model for text encoding. Default bert-base-uncased. max_text_len defaults to 256.
- **train.optim.lr**: Learning rate. Default 2e-4. lr_backbone 2e-5. Supports bf16 precision in addition to fp16/fp32.
- **dataset.max_labels**: Maximum labels per image during training. Default 50. Increase for dense annotation datasets.
- **model.num_queries**: Object queries. Default 900 (higher than DINO's 300) due to open-vocabulary nature.
- **train.optim.lr_steps**: MultiStep LR schedule. Default [10].

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Same DDP/FSDP behavior as DINO. Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars set by orchestrator.

## Export / TRT Defaults

- Export input: 960x544 (larger than other OD models), opset 17
- TRT data types: FP32, FP16 only — **INT8 is NOT supported**
- TRT workspace: 8192 MB (8x larger than other OD models)
- TRT max_batch_size: 4

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Grounding DINO is heavier than standard DINO due to the text encoder (BERT). 24GB+ GPU memory recommended. Reduce batch_size for 16GB GPUs.

## Error Patterns

**CUDA out of memory**: Reduce batch_size (4 -> 2 -> 1). The BERT text encoder adds significant memory overhead on top of the vision backbone.

**Val annotation category IDs**: Validation annotations should have category IDs starting from 0 for correct loss computation. Use annotation format conversion if needed.

**Text encoder loading error**: Ensure the container has access to download bert-base-uncased weights or provide a local path.
