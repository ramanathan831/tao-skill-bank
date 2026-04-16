# OCDNet

OCDNet for scene text detection. Detects arbitrary-oriented text regions in natural images using a differentiable binarization approach.

Set train.pretrained_model_path for pretrained weights.

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
