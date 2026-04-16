# Mask Grounding DINO

Mask Grounding DINO for grounded instance segmentation. Extends Grounding DINO with mask prediction head for open-set segmentation guided by text prompts.

Set train.pretrained_model_path for full model weights.

## Eval Dataset

Optional. Validation uses COCO-format annotations even when training uses ODVG.

## Important Parameters

- **model.backbone**: Default swin_tiny_224_1k. Same backbone options as Grounding DINO.
- **train.optim.lr**: Learning rate. Default 2e-4. lr_backbone 2e-5. Reuses GDINOTrainExpConfig — same training setup as Grounding DINO.
- **model.num_queries**: Object queries. Default 900.
- **model.has_mask**: Enables mask prediction head. Default True. Adds mask/dice/rela loss coefficients.
- **model.num_region_queries**: Number of region queries for mask prediction. Default 100.
- **model.loss_types**: Loss components. Default [labels, boxes, masks]. Includes mask_loss_coef, dice_loss_coef, rela_loss_coef.
- **evaluate.ioi_threshold**: IoI threshold for mask evaluation. Default 0.5.
- **evaluate.nms_threshold**: NMS threshold. Default 0.2.
- **evaluate.text_threshold**: Text matching threshold. Default 0.3.
- **dataset.has_mask**: Dataset includes mask annotations. Default True. val_data_sources default data_type is "VG".

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed. Same DDP/FSDP behavior as Grounding DINO.

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | `ddp` or `fsdp` | `ddp` |

## Hardware

Minimum 1 GPU(s), recommended 4 GPU(s). 24GB+ (A100 recommended) VRAM per GPU. Heavier than Grounding DINO due to mask prediction head. 24GB+ GPU memory recommended.

## Error Patterns

**CUDA out of memory**: Reduce batch_size. Mask prediction adds overhead on top of Grounding DINO.
