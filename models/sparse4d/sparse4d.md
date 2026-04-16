# Sparse4D

Sparse4D for multi-camera temporal 3D object detection and tracking. Uses sparse queries with deformable attention across camera views and time for end-to-end 3D perception. Includes instance bank for temporal tracking.

Requires pretrained ResNet-101 backbone. Set train.pretrained_model_path.

## Eval Dataset

Optional. Val/test splits configured via dataset ann_file paths.

## Important Parameters

- **model.backbone**: Backbone. Default resnet_101.
- **model.neck.out_channels**: FPN output channels. Default 256. num_outs=4.
- **model.input_shape**: Input image shape [W, H]. Default [1408, 512].
- **model.head.num_output**: Number of detection output queries. Default 300.
- **model.head.num_decoder**: Number of decoder layers. Default 6.
- **model.head.temporal**: Enable temporal reasoning. Default True.
- **model.head.instance_bank.num_anchor**: Instance bank anchors. Default 900.
- **model.head.instance_bank.num_temp_instances**: Temporal instance count. Default 600.
- **model.depth_branch.loss_weight**: Depth supervision loss weight. Default 0.2.
- **dataset.batch_size**: Per-GPU batch size. Default 2.
- **dataset.num_frames**: Sequence length. Default 200.
- **dataset.classes**: Detection classes. Default [person, gr1_t2, agility_digit, nova_carter]. num_ids=70 for tracking.
- **train.optim.lr**: Learning rate. Default 5e-5. img_backbone lr_mult=0.2.
- **train.lr_scheduler**: Cosine scheduler with linear warmup (500 iters, ratio 0.333).
- **train.grad_clip.max_norm**: Gradient clipping. Default 25.
- **train.precision**: Options: bf16, fp16, fp32. Default bf16.
- **evaluate.metrics**: Eval metrics. Default ["detection"]. Optional tracking evaluation.
- **evaluate.tracking.enabled**: Enable tracking evaluation. tracking_threshold=0.2.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |

- Multi-GPU strategy: `ddp_find_unused_parameters_true` (no fsdp support)
- `sync_batchnorm` is always enabled (True)
- Iterations per epoch computed as: `num_frames * num_bev_groups / (num_nodes * num_gpus * batch_size)`
- **Scaling:** When increasing GPUs, effective batch size grows and iterations-per-epoch shrinks proportionally

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

Minimum 2 GPU(s), recommended 8 GPU(s). 40GB+ (A100 recommended) VRAM per GPU. Multi-camera temporal model is memory intensive. bf16 required for practical training. Multi-GPU strongly recommended. Instance bank requires substantial memory for temporal reasoning.

## Error Patterns

**dataset_convert required**: Must run dataset_convert first to produce annotation pickles and anchor_init.npy.

**Missing anchor file**: Set model.head.instance_bank.anchor to the anchor_init.npy path from dataset_convert results.

**Temporal OOM**: Reduce dataset.num_frames or dataset.batch_size if running out of memory during temporal training.
