# Action Recognition

Action recognition from video sequences. Supports RGB, optical flow, and joint (multi-stream) input types for classifying temporal actions in video clips.

Set model.pretrained_model_path for pretrained backbone weights.

## Eval Dataset

Optional. Test dataset is provided as test.tar.gz separate from training.

## Important Parameters

- **model.model_type**: Input type: rgb, of (optical flow), or joint (multi-stream).
- **model.backbone**: Default resnet_18. Used as the spatial feature extractor.
- **dataset.label_map**: Dictionary mapping class names to indices.
- **model.rgb_seq_length**: Number of frames per clip for RGB input.
- **model.of_seq_length**: Number of frames for optical flow input.
- **train.optim.lr**: Learning rate. Default 5e-4.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node oriented

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ VRAM per GPU. Memory depends on sequence length and input resolution. batch_size=2 is conservative for video data.

## Error Patterns

**Sequence length mismatch**: Ensure video clips have enough frames for the configured rgb_seq_length or of_seq_length.
