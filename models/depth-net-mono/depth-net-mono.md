# Depth Net Mono

Monocular depth estimation using Metric Depth Anything v2 or Relative Depth Anything architectures. Predicts per-pixel depth from single RGB images.

Uses pretrained Depth Anything v2 encoder. Set model.mono_backbone.pretrained_path.

## Training Requirements

- **Dataset type:** depth_net_mono
- **Formats:** ThreeDVLM, FSD, NvCLIP, IssacStereo, Crestereo, Middlebury, NYUDV2, NYUDV2Relative, BaseRelativeMonoDataset, BaseMetricMonoDataset
- **Monitoring metric:** val/loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | dataset.test_dataset.data_sources | eval_dataset | data_file: annotations.txt | Yes |
| inference | dataset.infer_dataset.data_sources | inference_dataset | data_file: annotations.txt | Yes |
| quantize | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt | Yes |
| quantize | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt | Yes |
| quantize | dataset.quant_calibration_dataset.images_dir | train_datasets | images.tar.gz | No |
| train | dataset.train_dataset.data_sources | train_datasets | data_file: annotations.txt | Yes |
| train | dataset.val_dataset.data_sources | eval_dataset | data_file: annotations.txt | Yes |

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
    "model.model_type": "RelativeDepthAnything",
    "dataset.train_dataset.data_sources": [{"data_file": f"{S3_TRAIN}/annotations.txt"}],
    "dataset.val_dataset.data_sources": [{"data_file": f"{S3_EVAL}/annotations.txt"}],
}
```

**evaluate (mandatory data sources):**
```python
{
    "model.model_type": "RelativeDepthAnything",
    "dataset.test_dataset.data_sources": [{"data_file": f"{S3_EVAL}/annotations.txt"}],
}
```

**export:**
```python
{
    "model.model_type": "RelativeDepthAnything",
    "export.input_height": 518,
    "export.input_width": 518,
}
```

**inference (mandatory data sources):**
```python
{
    "model.model_type": "RelativeDepthAnything",
    "dataset.infer_dataset.data_sources": [{"data_file": f"{S3_EVAL}/annotations.txt"}],
}
```

**quantize (mandatory data sources):**
```python
{
    "dataset.train_dataset.data_sources": [{"data_file": f"{S3_TRAIN}/annotations.txt"}],
    "dataset.val_dataset.data_sources": [{"data_file": f"{S3_EVAL}/annotations.txt"}],
    "dataset.quant_calibration_dataset.images_dir": f"{S3_TRAIN}/images.tar.gz",
}
```
## Eval Dataset

Optional. Val dataset configured via dataset.val_dataset.data_sources.

## Important Parameters

- **model.model_type**: Model architecture. Options: MetricDepthAnything, RelativeDepthAnything. Default MetricDepthAnything.
- **model.mono_backbone.encoder**: Backbone encoder. Options: vits, vitb, vitl, vitg. Default vitl.
- **model.mono_backbone.pretrained_path**: Path to pretrained Depth Anything v2 encoder weights.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW).
- **train.lr_scheduler**: LR scheduler. Options: MultiStepLR, StepLR, CustomMultiStepLRScheduler, LambdaLR, PolynomialLR, OneCycleLR, CosineAnnealingLR.
- **train.precision**: Training precision. Options: bf16, fp16, fp32.
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.
- **train.activation_checkpoint**: Enable activation checkpointing. Default False.
- **dataset.train_dataset.data_sources**: List of dataset source dicts. Each has data_file and dataset_name fields. Multiple sources per split.
- **dataset.augmentation.crop_size**: Training crop size. Default [518, 518].
- **dataset.max_depth**: Maximum depth range for metric depth estimation.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

- `ddp` with activation checkpointing: `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- `fsdp` forces precision to FP16

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Export / TRT Defaults

- TRT data types: FP32, FP16 only

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 24GB+ VRAM per GPU. ViT-Large encoder is memory intensive. Use bf16 precision and activation checkpointing for larger inputs.

## Error Patterns

**Depth range mismatch**: Ensure dataset.max_depth / min_depth match the actual depth range in your data.

**Missing pretrained weights**: DepthAnything v2 encoder requires pretrained_path to be set for fine-tuning.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `depth_net_mono.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| gen_trt_engine | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | model file inferred from the parent job results folder |
| gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TensorRT engine path |
| gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `dataset.dataset_name` | `MonoDataset` | MonoDataset |
| train | `model.mono_backbone.pretrained_path` | `{'link': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'destination_path': '/ptm/depth_net/mono_backbone/dinov2_vitl14_pretrain.pth'}` | {'link': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth', 'destination_path': '/ptm/depth_net/mono_backbone/dinov2_vitl14_pretrain.pth'} |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
