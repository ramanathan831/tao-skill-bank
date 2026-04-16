# NVPanoptix3D

NVPanoptix3D for panoptic 3D scene reconstruction from posed RGB images. Produces 3D panoptic segmentation (semantic, instance, and panoptic masks) with occupancy completion. Built on VGGT backbone with Mask2Former-style head and 3D frustum reconstruction.

Uses 2D and 3D stage checkpoints. Set train.checkpoint_2d and train.checkpoint_3d for staged initialization.

## Eval Dataset

Optional. Val/test splits configured via dataset.val and dataset.test paths.

## Important Parameters

- **model.sem_seg_head.num_classes**: Number of semantic classes. Default 13.
- **model.mode**: Prediction mode. Options: panoptic, instance, semantic. Default panoptic.
- **model.backbone_type**: Backbone. Default vggt (only option in schema).
- **model.mask_former.num_object_queries**: Object queries. Default 100.
- **model.mask_former.dec_layers**: Decoder layers. Default 10.
- **model.frustum3d.truncation**: 3D frustum truncation. Default 3.
- **model.frustum3d.panoptic_weight**: Panoptic loss weight. Default 25.
- **model.frustum3d.completion_weights**: Completion loss weights. Default [50, 25, 10].
- **dataset.name**: Dataset name. Options: front3d, matterport, synthetic_hospital, synthetic_warehouse.
- **dataset.downsample_factor**: Image downsample factor. Default 1 (Front3D), 2 (Matterport).
- **dataset.target_size**: Target image size. Default [320, 240].
- **dataset.depth_min**: Min depth. Default 0.4 meters.
- **dataset.depth_max**: Max depth. Default 6.0 meters.
- **train.lr**: Learning rate. Default 2e-4. backbone_multiplier=0.1.
- **train.lr_scheduler**: Options: MultiStep, Warmuppoly. Milestones [88, 96].
- **train.precision**: Options: fp16, fp32. Default fp16.
- **train.distributed_strategy**: Options: ddp, fsdp. activation_checkpoint=True by default.
- **train.clip_grad_norm**: Gradient clipping norm. Default 0.1.
- **export.onnx_file_2d**: ONNX path for 2D model component.
- **export.onnx_file_3d**: ONNX path for 3D model component.
- **export.max_voxels**: Max voxels for engine input. Default 700000.
- **inference.mode**: Options: semantic, instance, panoptic.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` only | `ddp` |

- **`fsdp` is NOT supported** for NVPanoptix3D (code only handles `ddp`)
- `ddp` with activation checkpointing (enabled by default): `find_unused_parameters=False`
- `ddp` without: `find_unused_parameters=True`
- FAN backbones with 3D enabled auto-enable `sync_batchnorm`

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Export / TRT Defaults

- Exports separate 2D and 3D ONNX models (onnx_file_2d, onnx_file_3d)
- TRT data types: FP32, FP16 only
- max_voxels: 700000 (engine input tensor limit)

## Hardware

Minimum 2 GPU(s), recommended 4 GPU(s). 40GB+ (A100 recommended) VRAM per GPU. 3D reconstruction is very memory intensive. fp16 recommended. activation_checkpoint enabled by default. FSDP for multi-node. AutoML is disabled for this model.

## Error Patterns

**Missing frustum mask**: Ensure meta/frustum_mask.npz is present in the dataset directory.

**Downsample factor mismatch**: Use downsample_factor=2 for Matterport3D, 1 for Front3D / synthetic datasets.

**3D occupancy OOM**: Reduce frustum_dims or grid_dimensions if running out of GPU memory during 3D reconstruction.
