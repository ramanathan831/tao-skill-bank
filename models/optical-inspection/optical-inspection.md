# Optical Inspection

Optical inspection for defect detection using Siamese networks. Compares image pairs to detect manufacturing defects, anomalies, or quality issues.

Set train.pretrained_model_path for pretrained Siamese weights.

## Eval Dataset

Optional. Eval dataset uses same format (images + CSV).

## Important Parameters

- **model.model_type**: Siamese variant. Options include Siamese, Siamese_3.
- **model.model_backbone**: Default custom.
- **model.embedding_vectors**: Number of embedding dimensions. Default 5.
- **train.optim.lr**: Learning rate. Default 5e-4.
- **dataset.num_input**: Number of input images per comparison.
- **dataset.input_map**: Mapping of input channels / image pairs.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node only
- Lightweight Siamese network, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. Siamese networks for inspection are lightweight. Single GPU sufficient.

## Error Patterns

**CSV format error**: Ensure dataset.csv has the correct column format for image pair paths and labels.
