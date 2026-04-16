# Re-Identification

Person re-identification. Learns discriminative embeddings to match the same person across different camera views. Metric learning based.

Set model.pretrained_model_path for pretrained weights.

## Eval Dataset

Required. Evaluation requires test and query datasets for retrieval-based metrics (CMC, mAP).

## Important Parameters

- **dataset.num_classes**: Number of identities. Default 751. Must match the number of unique identities in training data.
- **model.backbone**: Default resnet_50.
- **optim.base_lr**: Base learning rate. Default 3.5e-4.
- **dataset.batch_size**: Per-GPU batch size. Default 64. Re-ID benefits from large batches for better triplet/contrastive sampling.
- **dataset.num_instances**: Number of instances per identity in a batch. Controls sampling strategy for metric learning.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Multi-GPU strategy: `ddp_find_unused_parameters_true`
- `sync_batchnorm` is always enabled
- Precision forced to FP16 (`16-mixed`)
- No explicit `num_nodes` config — single-node oriented

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ VRAM per GPU. Re-ID models are relatively lightweight but benefit from large batch sizes for metric learning.

## Error Patterns

**num_classes mismatch**: Ensure dataset.num_classes equals the number of unique identity folders in the training set.

**Query/gallery mismatch**: Query and test (gallery) datasets must share the same identity namespace.
