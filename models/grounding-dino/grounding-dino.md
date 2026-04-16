# Grounding DINO

Grounding DINO for open-set object detection. Combines DINO-style detection with BERT text encoder for language-guided detection. Detects objects described by text prompts without fixed class vocabulary.

Set train.pretrained_model_path for full Grounding DINO weights or model.pretrained_backbone_path for backbone-only.

## Eval Dataset

Optional. Validation uses COCO-format annotations for mAP even though training can use ODVG format.

## Important Parameters

- **model.backbone**: Default swin_tiny_224_1k. Also supports resnet_50 and other Swin variants. Swin generally performs better for grounding tasks.
- **model.text_encoder_type**: BERT model for text encoding. Default bert-base-uncased. max_text_len defaults to 256.
- **train.optim.lr**: Learning rate. Default 2e-4. lr_backbone 2e-5. Supports bf16 precision in addition to fp16/fp32.
- **dataset.max_labels**: Maximum labels per image during training. Default 50. Increase for dense annotation datasets.
- **model.num_queries**: Object queries. Default 900 (higher than DINO's 300) due to open-vocabulary nature.
- **train.optim.lr_steps**: MultiStep LR schedule. Default [10].

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

Same DDP/FSDP behavior as DINO. Multi-node requires `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` env vars set by orchestrator.

## Export / TRT Defaults

- Export input: 960x544 (larger than other OD models), opset 17
- TRT data types: FP32, FP16 only — **INT8 is NOT supported**
- TRT workspace: 8192 MB (8x larger than other OD models)
- TRT max_batch_size: 4

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Grounding DINO is heavier than standard DINO due to the text encoder (BERT). 24GB+ GPU memory recommended. Reduce batch_size for 16GB GPUs.

## Error Patterns

**CUDA out of memory**: Reduce batch_size (4 -> 2 -> 1). The BERT text encoder adds significant memory overhead on top of the vision backbone.

**Val annotation category IDs**: Validation annotations should have category IDs starting from 0 for correct loss computation. Use annotation format conversion if needed.

**Text encoder loading error**: Ensure the container has access to download bert-base-uncased weights or provide a local path.
