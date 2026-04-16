# OCRNet

OCRNet for scene text recognition. Recognizes text content from cropped text region images. Supports CTC and attention-based decoders.

Set train.pretrained_model_path for pretrained OCR weights.

## Eval Dataset

Optional. Test data provided as separate tarball.

## Important Parameters

- **dataset.character_list_file**: Path to character list defining the supported character set. This determines the output vocabulary size.
- **model.backbone**: Default ResNet.
- **model.prediction**: Decoder type. CTC or Attn (attention-based).
- **train.optim.lr**: Learning rate. Default 1.0 (Adadelta optimizer). High default is specific to Adadelta.
- **dataset.batch_size**: Per-GPU batch size. Default 16.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.distributed_strategy` | Strategy name | `auto` |

- Strategy: `auto` for single-GPU, reads `train.distributed_strategy` from config when multi-GPU
- No explicit `num_nodes` in train script — single-node oriented
- Lightweight model, single GPU typically sufficient

## Hardware

Minimum 1 GPU(s), recommended 1 GPU(s). 8GB+ VRAM per GPU. OCR text recognition is lightweight. Single GPU is typically sufficient.

## Error Patterns

**dataset_convert required**: If using raw images + gt files, run dataset_convert first to produce LMDB format.

**Character list mismatch**: All characters in training data must be present in the character_list file.
