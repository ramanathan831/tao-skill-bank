# Pose Classification

Pose classification using ST-GCN (Spatial Temporal Graph Convolutional Network). Classifies skeleton sequences into action categories from pose keypoint data.

Typically trained from scratch on skeleton data.

## Training Requirements

- **Dataset type:** pose_classification
- **Formats:** default
- **Monitoring metric:** val_acc

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| evaluate | evaluate.test_dataset.data_path | train_datasets |  | No |
| evaluate | evaluate.test_dataset.label_path | train_datasets |  | No |
| inference | inference.test_dataset.data_path | train_datasets |  | No |
| train | dataset.train_dataset.data_path | train_datasets |  | No |
| train | dataset.train_dataset.label_path | train_datasets |  | No |
| train | dataset.val_dataset.data_path | train_datasets |  | No |
| train | dataset.val_dataset.label_path | train_datasets |  | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "aws://bucket/data/train"
```

**train (mandatory data sources):**
```python
{
    "train.num_epochs": 30,
    "train.checkpoint_interval": 10,
    "train.validation_interval": 10,
    "train.num_gpus": 1,
    "num_classes": 6,
    "graph_layout": "nvidia",
    "dataset.train_dataset.data_path": f"{S3_TRAIN}",
    "dataset.train_dataset.label_path": f"{S3_TRAIN}",
    "dataset.val_dataset.data_path": f"{S3_TRAIN}",
    "dataset.val_dataset.label_path": f"{S3_TRAIN}",
}
```

**evaluate (mandatory data sources):**
```python
{
    "evaluate.test_dataset.data_path": f"{S3_TRAIN}",
    "evaluate.test_dataset.label_path": f"{S3_TRAIN}",
}
```

**inference (mandatory data sources):**
```python
{
    "inference.test_dataset.data_path": f"{S3_TRAIN}",
}
```
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
