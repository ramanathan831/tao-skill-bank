# SDG Pipeline YAML Schema Reference

Execution: **`omni_replicator.sh --exec "…/sdg_pipeline.py --config <this-file.yaml>"`** or via Docker.
Mode is **`pipeline_type`**: `good` | `defect` | `missing`.

## Pipeline types

| Template file | `pipeline_type` | Behavior |
|---------------|-----------------|----------|
| `good_image.yaml` | `good` | Single pass; no pose defects; class semantics on components |
| `defect_image.yaml` | `defect` | Single pass; shift/tombstone/sideflip defects |
| `missing_image.yaml` | `missing` | Two passes per trigger: `reference/` then `defective/` (hidden components) |

### Top-level

| Key | Type | Description |
|-----|------|-------------|
| `pipeline_type` | string | Required: `good`, `defect`, or `missing` |

## Common parameters (all pipeline types)

### Scene

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `scene` | string | USD scene path (local or Nucleus URL) | `/home/ubuntu/pcb-aoi/assets/temp_scene.usd` |
| `output` | string | Output root directory | `/home/user/.../sdg_output/good` |
| `num_triggers` | int | Number of randomization triggers to run | `1` |

### Seed & Output Control

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `seed` | int | Skip first `seed` scan-grid cells on trigger 0; first frame index = `seed` | `0` |
| `random_seed` | int | Base for lighting / camera / defect-prep / augmentation NumPy streams | `42` |
| `max_image_count` | int | Total writer output frames cap (-1 = no limit) | `-1` |

### Camera & Render

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `camera_path` | string | USD prim path of the camera | `/World/camera_light/Camera` |
| `camera_xform_path` | string | Camera parent Xform path | `/World/camera_light` |
| `ring_light_root` | string | Ring light root path | `/World/camera_light/aoi_ring_light` |
| `resolution` | [int, int] | Render resolution [width, height] | `[1920, 1080]` |
| `horizontal_aperture` | float | Orthographic aperture in mm | `200.0` |

### PathTracing

Templates only set `spp` and `total_spp`; add any optional key under `pathtracing:` only when you want to override Kit defaults (the pipeline calls `settings.set` only for keys present in YAML). See [RTX Interactive Path Tracing](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html).

| Key | Type | Description | Doc default |
|-----|------|-------------|-------------|
| `pathtracing.spp` | int | Samples per pixel per frame (1–32) | `1` |
| `pathtracing.total_spp` | int | Max accumulated SPP per pixel (`0` = unlimited) | templates: `32` or `64` |
| `pathtracing.adaptive_sampling_enabled` | bool | Non-uniform sampling by noise threshold | `false` |
| `pathtracing.adaptive_sampling_target_error` | float | Noise threshold when adaptive sampling is on | `0.001` |
| `pathtracing.max_bounces` | int | Max ray bounces (any type) | `4` |
| `pathtracing.max_specular_and_transmission_bounces` | int | Max specular/transmission bounces | `6` |
| `pathtracing.max_volume_bounces` | int | Max SSS volume scattering bounces | `64` |
| `pathtracing.ptfog_max_bounces` | int | Max bounces in fog volumes | `2` |
| `pathtracing.ptvol_max_bounces` | int | Max bounces for non-uniform volumes | `2` |
| `pathtracing.fractional_cutout_opacity` | bool | Stochastic cutout / translucency | `true` |
| `pathtracing.reset_pt_accum_on_anim_time_change` | bool | Restart PT accumulation when MDL anim time changes | `false` |
| `pathtracing.cached_enabled` | bool | Path tracing result caching | `true` |
| `pathtracing.lightcache_cached_enabled` | bool | Many-light sampling cache | `true` |
| `pathtracing.ris_mesh_lights` | bool | Sample emissive mesh geometry | `false` |
| `pathtracing.aa_op` | int | AA pattern: Box `0`, Triangle `1`, Gaussian `2`, Uniform `3` | `1` |
| `pathtracing.aa_filter_radius` | float | AA footprint radius (pixels) | `1.0` |
| `pathtracing.firefly_filter_enabled` | bool | Reduce bright firefly pixels | `true` |
| `pathtracing.optix_denoiser_enabled` | bool | OptiX denoiser on radiance | `true` |
| `pathtracing.optix_denoiser_temporal_enabled` | bool | Temporal denoising for sequences | `false` |
| `pathtracing.optix_denoiser_blend_factor` | float | Blend denoised vs raw (`1.0` = no denoise visible per doc) | `1.0` |
| `pathtracing.optix_denoiser_denoise_aovs` | bool | Denoise AOVs as well | `true` |

### Scan Grid

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `scan_grid.x_start` | float | Grid X start position (mm) | `21.6` |
| `scan_grid.x_end` | float | Grid X end position (mm) | `-106` |
| `scan_grid.y_start` | float | Grid Y start position (mm) | `23.2` |
| `scan_grid.y_end` | float | Grid Y end position (mm) | `-77` |
| `scan_grid.step` | float | Grid step size (mm) | `10` |
| `scan_grid.z` | float | Camera Z height (mm) | `0` |

### PCBA & Component Discovery

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `pcba_root` | string | USD prim path of the PCBA root | `/World/pcba_main_s_detail/PCBA/tn__60014242BASEA04_fM9E` |
| `component_types` | [string, ...] | Package type scope names | `[_0402_H060, _0603_H070, ...]` |

### Lighting Randomization

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `lighting.ring_light` | bool | `true` = per-layer RGB (soldering light), `false` = white light | `true` |
| `lighting.exposure_range` | [float, float] | Exposure range | `[0.01, 1.0]` |
| `lighting.cone_angle_range` | [float, float] | Cone angle range in degrees | `[90, 150]` |
| `lighting.cone_softness_range` | [float, float] | Softness range | `[0.5, 1.0]` |

#### Per-Layer Colors (`lighting.layers.<LayerName>`)

Used when `ring_light: true`. Layer names: Inner_Red, Middle_Green, Outer_Blue.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `intensity` | [float, float] | Light intensity range | `[5000, 8000]` |
| `color_r` | [float, float] | Red channel range (0-1) | `[0.95, 1.0]` |
| `color_g` | [float, float] | Green channel range (0-1) | `[0.0, 0.05]` |
| `color_b` | [float, float] | Blue channel range (0-1) | `[0.0, 0.05]` |

#### White Light (`lighting.white_light`)

Used when `ring_light: false`. Same schema as per-layer colors.

### Camera Randomization

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `camera_rotation.x_range` | [float, float] | Rotation range around X axis (degrees) | `[-5, 5]` |
| `camera_rotation.y_range` | [float, float] | Rotation range around Y axis (degrees) | `[-5, 5]` |
| `camera_rotation.z_fixed` | float | Fixed Z rotation (degrees) | `-90` |

### Augmentation

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `augmentation.motion_blur.probability` | float | Probability of MotionBlur (0-1) | `0.8` |
| `augmentation.motion_blur.alpha_range` | [float, float] | Blur strength range | `[0.01, 0.3]` |
| `augmentation.motion_blur.kernel_choices` | [int, ...] | Kernel sizes to choose from | `[5, 7]` |

### Component Writer

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `component_xform_depth` | int | Xform hierarchy depth for ComponentInstanceWriter to merge mesh-level IDs to component-level | `7` |

### Writer — Good & Defect Pipelines

Flat `writer:` — passed to BasicWriter. **`semantic_types`** is typically `[class]` (good) or `[class, defect]` (defect).

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `writer.rgb` | bool | Output RGB images | `true` |
| `writer.image_output_format` | string | Image format (png, jpg) | `png` |
| `writer.bounding_box_2d_tight` | bool | Tight 2D bounding boxes | `false` |
| `writer.bounding_box_2d_loose` | bool | Loose 2D bounding boxes | `false` |
| `writer.bounding_box_3d` | bool | 3D bounding boxes | `false` |
| `writer.semantic_segmentation` | bool | Semantic segmentation masks | `false` |
| `writer.colorize_semantic_segmentation` | bool | Colorized semantic segmentation PNG | `false` |
| `writer.instance_id_segmentation` | bool | Instance ID segmentation masks | `false` |
| `writer.colorize_instance_id_segmentation` | bool | Colorized instance segmentation PNG | `false` |
| `writer.component_level_segmentation` | bool | Merge instance segmentation to component level (requires `instance_id_segmentation: true`) | `false` |
| `writer.distance_to_camera` | bool | Depth from camera origin | `false` |
| `writer.distance_to_image_plane` | bool | Depth from image plane | `false` |
| `writer.colorize_depth` | bool | Colorized depth PNG visualization | `false` |
| `writer.camera_params` | bool | Output camera parameters | `true` |
| `writer.semantic_types` | [string, ...] | Semantic type keys | `[class]` or `[class, defect]` |
| `writer.frame_padding` | int | Zero-padding for frame numbers | `4` |

## Defect Pipeline Only

### Defects (`defects.<type>`)

Common fields for all defect types:

| Key | Type | Description |
|-----|------|-------------|
| `enabled` | bool | Enable this defect type |
| `ratio` | float | Fraction of components to apply this defect to |

#### Shift (`defects.shift`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `translate_range` | float | Max XY translation offset (mm, +/-) | `0.2` |
| `rotate_z_range` | float | Max Z-axis rotation (degrees, +/-) | `15` |

#### Tombstone (`defects.tombstone`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `angle_min` | float | Minimum tilt angle around Y axis (degrees) | `80` |
| `angle_max` | float | Maximum tilt angle around Y axis (degrees) | `90` |

#### Sideflip (`defects.sideflip`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `angle_min` | float | Minimum flip angle around X axis (degrees) | `80` |
| `angle_max` | float | Maximum flip angle around X axis (degrees) | `90` |

## Missing Pipeline Only

### Missing defect

| Key | Type | Description |
|-----|------|-------------|
| `missing.ratio` | float | Fraction of component pool to hide per trigger (0–1) |

### Writer (two-pass)

Nested **`writer.reference`** and **`writer.defective`** — each is a dict of writer kwargs (see annotator keys above). Outputs go under `trigger_XXXX/reference/` and `trigger_XXXX/defective/`. No top-level flat `writer` for `missing`.

#### Reference Pass (`writer.reference`)
All components visible; missing components have `defect=missing` semantic label.

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `writer.reference.rgb` | bool | Output RGB images | `false` |
| `writer.reference.image_output_format` | string | Image format | `png` |
| `writer.reference.semantic_segmentation` | bool | Semantic segmentation masks | `true` |
| `writer.reference.colorize_semantic_segmentation` | bool | Colorized semantic segmentation | `true` |
| `writer.reference.instance_id_segmentation` | bool | Instance ID segmentation | `false` |
| `writer.reference.bounding_box_2d_tight` | bool | Tight 2D bounding boxes | `true` |
| `writer.reference.bounding_box_2d_loose` | bool | Loose 2D bounding boxes | `false` |
| `writer.reference.bounding_box_3d` | bool | 3D bounding boxes | `false` |
| `writer.reference.distance_to_camera` | bool | Depth from camera origin | `false` |
| `writer.reference.distance_to_image_plane` | bool | Depth from image plane | `false` |
| `writer.reference.semantic_types` | [string, ...] | Semantic type keys | `[class, defect]` |
| `writer.reference.frame_padding` | int | Zero-padding for frame numbers | `4` |

#### Defective Pass (`writer.defective`)
Missing components are hidden (invisible).

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `writer.defective.rgb` | bool | Output RGB images | `true` |
| `writer.defective.image_output_format` | string | Image format | `png` |
| `writer.defective.semantic_segmentation` | bool | Semantic segmentation | `false` |
| `writer.defective.instance_id_segmentation` | bool | Instance ID segmentation | `false` |
| `writer.defective.bounding_box_2d_tight` | bool | Tight 2D bounding boxes | `false` |
| `writer.defective.bounding_box_2d_loose` | bool | Loose 2D bounding boxes | `false` |
| `writer.defective.bounding_box_3d` | bool | 3D bounding boxes | `false` |
| `writer.defective.distance_to_camera` | bool | Depth from camera origin | `false` |
| `writer.defective.distance_to_image_plane` | bool | Depth from image plane | `false` |
| `writer.defective.frame_padding` | int | Zero-padding for frame numbers | `4` |
