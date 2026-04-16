# ML Recog

Metric learning recognition for fine-grained visual recognition. Learns embeddings for retrieval-based matching (e.g., retail product recognition). Uses triplet/contrastive losses.

Set model.pretrained_model_path for pretrained backbone.

## Eval Dataset

Required. Evaluation requires reference and query datasets for retrieval metrics.

## Important Parameters

- **model.backbone**: Default resnet_50. Options: resnet_50, resnet_101, fan_small, fan_base, fan_large, fan_tiny, nvdinov2_vit_large_legacy.
- **model.feat_dim**: Embedding dimension. Default 256. Output feature vector size for similarity matching.
- **train.batch_size**: Per-GPU batch size. Default 4. val_batch_size also 4.
- **dataset.num_instance**: Instances per identity in a batch (P/K sampling). Default 4. Controls how many images of the same class appear together.
- **train.optim.trunk.base_lr**: Learning rate for the trunk (backbone). Default 3.5e-4 (Adam).
- **train.optim.embedder.base_lr**: Learning rate for the embedding head. Default 3.5e-4.
- **train.optim.triplet_loss_margin**: Margin for triplet loss. Default 0.3. smooth_loss=True by default.
- **train.optim.miner_function_margin**: Hard mining margin. Default 0.1. Controls pair mining difficulty.
- **train.optim.steps**: LR decay steps. Default [40, 70] with gamma=0.1.
- **dataset.train_dataset**: Path to training images organized in class folders.
- **dataset.val_dataset**: Dict with 'reference' and 'query' keys pointing to ImageNet-format directories for retrieval evaluation.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node oriented

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ VRAM per GPU. Metric learning benefits from larger batch sizes for better triplet sampling but is otherwise moderate on memory.

## Error Patterns

**Reference/query mismatch**: Ensure reference and query datasets share compatible class namespaces for evaluation.
