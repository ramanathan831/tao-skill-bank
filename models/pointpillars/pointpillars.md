# PointPillars

PointPillars for 3D object detection from LiDAR point clouds. Encodes point clouds into a pseudo-image via pillar-based representation, then applies 2D detection. Used in autonomous driving / robotics.

Typically trained from scratch. Provide train.resume_training_checkpoint_path to resume.

## Eval Dataset

Optional. Validation data (val.tar.gz) is separate from training. Used for mAP evaluation.

## Important Parameters

- **train.num_epochs**: Default 80 (much higher than other TAO models). PointPillars needs more epochs for convergence on 3D detection.
- **train.lr**: Learning rate. Default 0.003 (adam_onecycle scheduler).
- **dataset.class_names**: List of 3D object classes. Default 7 classes (KITTI-style). Modify to match your dataset.
- **dataset.data_path**: Path to point cloud data directory.
- **dataset.data_info_path**: Path to data info files from dataset_convert step.
- **dataset.point_cloud_range**: Spatial extent of the point cloud to consider. Must match your sensor configuration.
- **model.dense_head.anchor_generator_config**: Anchor configurations per class. Must be tuned for your object sizes and the point cloud range.

## Multi-GPU / Multi-Node

**Launch method:** `torchrun` (LIGHTNING_EXCLUDED_NETWORK). Uses PyTorch native `DistributedDataParallel` (NOT Lightning Trainer).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs per node | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- `CUDA_VISIBLE_DEVICES` is explicitly set from `TAO_VISIBLE_DEVICES`
- Uses `nn.parallel.DistributedDataParallel` directly (not Lightning strategy)
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

Minimum 1 GPU(s), recommended 4 GPU(s). 16GB+ (V100 or A100) VRAM per GPU. PointPillars is relatively efficient for 3D detection. The main bottleneck is data I/O for large point cloud datasets.

## Error Patterns

**dataset_convert required**: Training will fail if data_info_path is not populated from a prior dataset_convert job. Always run convert first.

**Point cloud range mismatch**: If point_cloud_range does not match the actual sensor data extent, detections will be poor or empty.

**Epoch numbering**: PointPillars checkpoint epoch numbers may be offset by 1 from status.json reported epochs.
