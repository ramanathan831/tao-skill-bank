# Depth Net Stereo

Stereo depth estimation using FoundationStereo architecture. Predicts disparity maps from stereo image pairs for 3D reconstruction.

Uses pretrained Depth Anything v2 and EdgeNeXt encoders. Set model.stereo_backbone.depth_anything_v2_pretrained_path and model.stereo_backbone.edgenext_pretrained_path.

## Eval Dataset

Optional. Val dataset configured via dataset.val_dataset.data_sources.

## Important Parameters

- **model.model_type**: Architecture. Default FoundationStereo for stereo.
- **model.stereo_backbone.encoder**: Backbone encoder. Options: vits, vitb, vitl, vitg. Default vitl.
- **model.max_disparity**: Maximum disparity range. Default 416, range 1-416.
- **model.hidden_dims**: Hidden dimensions in GRU refinement. Default [128,128,128].
- **model.train_iters**: GRU refinement iterations during training. Default 22.
- **model.volume_dim**: Cost volume dimension. Default 32.
- **model.low_memory**: Memory optimization level. Range 0-4. Higher = less memory.
- **dataset.baseline**: Stereo camera baseline. Default 193.001/1e3 meters.
- **dataset.focal_x**: Camera focal length X. Default 1998.842.
- **train.optim.lr**: Learning rate. Default 1e-4 (AdamW).
- **train.precision**: Training precision. Options: bf16, fp16, fp32.
- **train.distributed_strategy**: Distribution strategy. Options: ddp, fsdp.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Same DDP/FSDP behavior as depth-net-mono. Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars.

## Export / TRT Defaults

- TRT data types: FP32, FP16 only

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Stereo matching is memory intensive due to cost volume. Use low_memory > 0 for constrained GPUs. bf16 recommended.

## Error Patterns

**Disparity overflow**: Reduce model.max_disparity if targets exceed range or OOM occurs.

**Missing pretrained paths**: Both depth_anything_v2_pretrained_path and edgenext_pretrained_path should be set for fine-tuning.
