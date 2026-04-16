# BEVFusion

BEVFusion for multi-sensor 3D object detection. Fuses LiDAR point clouds and camera images in bird's-eye-view (BEV) space. Used in autonomous driving for robust 3D perception.

Set pretrained backbone paths for Swin image backbone.

## Eval Dataset

Optional. Val dataset split is configured via ann_file in dataset config.

## Important Parameters

- **dataset.classes**: List of detection classes. Default ["person"]. Must match the annotation categories.
- **dataset.type**: Dataset type. Options: KittiPersonDataset, TAO3DSyntheticDataset, TAO3DDataset.
- **dataset.root_dir**: Root directory of the KITTI-style dataset.
- **dataset.box_type_3d**: 3D box coordinate frame. Options: lidar, camera. Default lidar.
- **train.optimizer.lr**: Learning rate. Default 2e-4 (AdamW). Use AmpOptimWrapper for mixed precision via optimizer.wrapper_type.
- **input_modality**: Dict controlling sensor modalities. Keys: use_lidar (True), use_camera (True), use_radar (False), use_map (False).
- **model.img_backbone**: Image backbone. Default mmdet.SwinTransformer (Swin-Tiny). embed_dims=96, depths=[2,2,6,2].
- **model.view_transform.type**: View transform for BEV projection. Options: DepthLSSTransform, LSSTransform. Default DepthLSSTransform.
- **model.point_cloud_range**: Spatial extent of LiDAR. Default [0,-40,-3,70.4,40,1].
- **model.voxel_size**: Voxel dimensions. Default [0.05, 0.05, 0.1].
- **dataset.train_dataset.batch_size**: Per-GPU batch size. Default 4.

## Multi-GPU / Multi-Node

**Launch method:** `torchrun` (LIGHTNING_EXCLUDED_NETWORK). The entrypoint runs `torchrun --nnodes=N --nproc-per-node=M train.py`, NOT plain `python`.

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs per node | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- `CUDA_VISIBLE_DEVICES` is explicitly set from `TAO_VISIBLE_DEVICES`
- BEVFusion uses mmdet3d-based distributed training, not Lightning DDP
- `NODE_RANK` is copied to `RANK` if `RANK` is unset

**Multi-node env vars** (set by orchestrator):

| Variable | Purpose |
|----------|---------|
| `WORLD_SIZE` | Number of nodes |
| `NODE_RANK` | This node's rank |
| `MASTER_ADDR` | Rank-0 node IP |
| `MASTER_PORT` | Rank-0 port (default 29500) |
| `NUM_GPU_PER_NODE` | GPUs per node |

## Hardware

Minimum 2 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. BEVFusion is memory-intensive due to multi-sensor fusion. A100 GPUs strongly recommended. Multi-GPU training expected.

## Error Patterns

**dataset_convert required**: Run dataset_convert before training to produce info pickle files.

**Missing modality data**: Ensure both camera images and LiDAR point clouds are present if using multi-modal fusion.

**Epoch numbering**: BEVFusion checkpoint epoch numbers may not follow standard zero-padded format.
