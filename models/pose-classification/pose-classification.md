# Pose Classification

Pose classification using ST-GCN (Spatial Temporal Graph Convolutional Network). Classifies skeleton sequences into action categories from pose keypoint data.

Typically trained from scratch on skeleton data.

## Eval Dataset

Optional. Validation data is provided alongside training as val_data.npy / val_label.pkl.

## Important Parameters

- **dataset.num_classes**: Number of pose action classes. Default 6.
- **model.graph_layout**: Skeleton graph layout. Options: nvidia, openpose. Determines joint connectivity.
- **model.graph_strategy**: Graph partitioning strategy for GCN.
- **train.optim.lr**: Learning rate. Default 0.1 (SGD). Higher than vision models due to graph convolution properties.
- **model.dropout**: Dropout rate for regularization.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node only
- Lightweight model, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. Pose classification is very lightweight — skeleton data is small. Single GPU is sufficient.

## Error Patterns

**Graph layout mismatch**: Ensure model.graph_layout matches the skeleton format in your .npy data files.

**Label shape mismatch**: train_label.pkl class indices must be in range [0, num_classes).
