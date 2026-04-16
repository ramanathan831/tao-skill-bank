# CenterPose

CenterPose for keypoint / pose estimation. Detects object centers and regresses keypoint locations. Used for 6-DoF object pose estimation.

Set model.backbone.pretrained_backbone_path.

## Eval Dataset

Optional. Val and test datasets are provided as separate tarballs.

## Important Parameters

- **dataset.num_classes**: Number of object categories. Default 1.
- **dataset.num_joints**: Number of keypoints per object. Fixed at 8 (bbox keypoints). Valid range: exactly 8.
- **dataset.input_res**: Input resolution. Fixed at 512. Output resolution fixed at 128.
- **dataset.category**: Object category name. Default "cereal_box".
- **model.backbone.model_type**: Default fan_small. Backbone options limited in schema.
- **train.optim.lr**: Learning rate. Default 6e-5. MultiStep scheduler with lr_steps=[90, 120], lr_decay=0.1.
- **train.loss_config**: Rich loss config with toggles: mse_loss, obj_scale, obj_scale_uncertainty, hps_uncertainty, reg_bbox, hm_hp. Weights: wh_weight=0.1, off_weight=1, hp_weight=1.
- **inference.use_pnp**: Use PnP for 6-DoF pose. Default True. Requires camera intrinsics (focal_length_x/y, principle_point_x/y).
- **export.input_width**: Export input size. Fixed at 512x512. opset_version=16.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |

- Strategy: `auto` (Lightning picks the best strategy automatically)
- No explicit `num_nodes` or `distributed_strategy` config — single-node only
- No `sync_batchnorm`

## Export / TRT Defaults

- Export input: 512x512 (fixed), opset 16
- TRT data types: FP32, FP16, INT8
- TRT opt_batch_size: 4, max_batch_size: 8

## Hardware

Minimum 1 GPU(s), recommended 2 GPU(s). 16GB+ VRAM per GPU. CenterPose is moderately memory-intensive depending on input resolution and number of keypoints.

## Error Patterns

**num_joints mismatch**: Ensure dataset.num_joints matches the keypoint count in your annotations.
