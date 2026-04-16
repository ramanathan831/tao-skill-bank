# Config — YAML Configuration Reference

YAML config files for the standalone SDG pipelines.

| File | Pipeline |
|------|----------|
| `good_image.yaml` | `sdg/standalone/sdg_good_image_pipeline.py` |
| `defect_image.yaml` | `sdg/standalone/sdg_defect_image_pipeline.py` |

## Common Parameters

These parameters are shared by both pipelines.

### Scene

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `scene` | string | USD scene path (local or Nucleus URL) | `omniverse://10.63.172.135/.../temp_scene.usd` |
| `output` | string | Output root directory | `/home/user/.../sdg_output/good` |
| `num_triggers` | int | Number of randomization triggers to run | `3` |

### Camera & Render

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `camera_path` | string | USD prim path of the camera | `/World/camera_light/Camera` |
| `camera_xform_path` | string | USD prim path of the camera's parent Xform (used for grid translation) | `/World/camera_light` |
| `ring_light_root` | string | USD prim path of the ring light root | `/World/camera_light/aoi_ring_light` |
| `resolution` | [int, int] | Render resolution [width, height] | `[1920, 1080]` |
| `horizontal_aperture` | float | Orthographic camera horizontal aperture (mm) | `200.0` |

### PathTracing

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `pathtracing.spp` | int | Samples per pixel per frame | `1` |
| `pathtracing.total_spp` | int | Total accumulated SPP (higher = less noise, slower) | `64` |

### Scan Grid

Controls the orthographic camera grid that scans the PCB. The camera translates through all X/Y positions at the given Z height.

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `scan_grid.x_start` | float | Grid X start position (mm) | `21.6` |
| `scan_grid.x_end` | float | Grid X end position (mm) | `-106` |
| `scan_grid.y_start` | float | Grid Y start position (mm) | `23.2` |
| `scan_grid.y_end` | float | Grid Y end position (mm) | `-77` |
| `scan_grid.step` | float | Grid step size (mm) | `10` |
| `scan_grid.z` | float | Camera Z height (mm) | `0` |

### Lighting Randomization

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `lighting.ring_light` | bool | `true` = per-layer RGB colors, `false` = all layers use white light | `true` |
| `lighting.exposure_range` | [float, float] | Uniform random range for light exposure | `[0.01, 1.0]` |
| `lighting.cone_angle_range` | [float, float] | Uniform random range for cone angle (degrees) | `[90, 150]` |
| `lighting.cone_softness_range` | [float, float] | Uniform random range for cone softness | `[0.5, 1.0]` |

#### Per-Layer Colors (`lighting.layers.<LayerName>`)

Used when `ring_light: true`. Each layer (Inner_Red, Middle_Green, Outer_Blue) has:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `intensity` | [float, float] | Uniform random range for light intensity | `[5000, 8000]` |
| `color_r` | [float, float] | Uniform random range for red channel (0–1) | `[0.95, 1.0]` |
| `color_g` | [float, float] | Uniform random range for green channel (0–1) | `[0.0, 0.05]` |
| `color_b` | [float, float] | Uniform random range for blue channel (0–1) | `[0.0, 0.05]` |

#### White Light (`lighting.white_light`)

Used when `ring_light: false`. All layers share the same randomization ranges (same schema as per-layer colors above).

### Camera Randomization

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `camera_rotation.x_range` | [float, float] | Random rotation range around X axis (degrees) | `[-5, 5]` |
| `camera_rotation.y_range` | [float, float] | Random rotation range around Y axis (degrees) | `[-5, 5]` |
| `camera_rotation.z_fixed` | float | Fixed Z rotation (degrees) | `-90` |

### Augmentation

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `augmentation.motion_blur.probability` | float | Probability of applying MotionBlur per trigger (0–1) | `0.8` |
| `augmentation.motion_blur.alpha_range` | [float, float] | Uniform random range for blur strength | `[0.01, 0.3]` |
| `augmentation.motion_blur.kernel_choices` | [int, ...] | Kernel sizes to randomly choose from | `[5, 7]` |

### Writer (BasicWriter Annotators)

Controls which annotators are enabled in the output.

| Key | Type | Description | Default |
|-----|------|-------------|---------|
| `writer.rgb` | bool | Output RGB images | `true` |
| `writer.image_output_format` | string | Image format (`png`, `jpg`, etc.) | `png` |
| `writer.bounding_box_2d_tight` | bool | Tight 2D bounding boxes | `false` |
| `writer.bounding_box_2d_loose` | bool | Loose 2D bounding boxes | `false` |
| `writer.bounding_box_3d` | bool | 3D bounding boxes | `false` |
| `writer.semantic_segmentation` | bool | Semantic segmentation masks | `false` |
| `writer.colorize_semantic_segmentation` | bool | Colorized semantic segmentation PNG | `false` |
| `writer.instance_id_segmentation` | bool | Instance ID segmentation masks | `true` |
| `writer.colorize_instance_id_segmentation` | bool | Colorized instance segmentation PNG | `false` |
| `writer.distance_to_camera` | bool | Depth from camera origin | `false` |
| `writer.distance_to_image_plane` | bool | Depth from image plane | `false` |
| `writer.colorize_depth` | bool | Colorized depth PNG visualization | `false` |
| `writer.semantic_types` | [string, ...] | Semantic type keys to include | `[class, defect]` |
| `writer.frame_padding` | int | Zero-padding for frame numbers in filenames | `4` |

## Defect Pipeline Only (`defect_image.yaml`)

These parameters are only used by `sdg_defect_image_pipeline.py`.

### PCBA & Component Discovery

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `pcba_root` | string | USD prim path of the PCBA root | `/World/pcba_main_s_detail/PCBA/tn__60014242BASEA04_fM9E` |
| `component_types` | [string, ...] | List of Scope names to search for components (e.g. package types) | `[_0402_H060, _0603_H070, ...]` |

### Defects (`defects.<type>`)

Each defect type has a common `enabled` and `ratio` field, plus type-specific parameters.

#### Common Fields

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `enabled` | bool | Enable this defect type | `true` |
| `ratio` | float | Fraction of component pool to apply this defect to | `0.01` |

#### Shift (`defects.shift`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `translate_range` | float | Max XY translation offset (mm, ±) | `0.2` |
| `rotate_z_range` | float | Max Z-axis rotation (degrees, ±) | `15` |

#### Tombstone (`defects.tombstone`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `angle_min` | float | Minimum tilt angle around Y axis (degrees) | `30` |
| `angle_max` | float | Maximum tilt angle around Y axis (degrees) | `90` |

#### Sideflip (`defects.sideflip`)

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `angle_min` | float | Minimum flip angle around X axis (degrees) | `0` |
| `angle_max` | float | Maximum flip angle around X axis (degrees) | `30` |
