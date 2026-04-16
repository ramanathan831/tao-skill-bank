"""
SDG Good Image Pipeline — multi-trigger scan with randomized lighting + augmentation.
Run inside Isaac Sim Script Editor.

Each trigger:
  1. Randomize aoi_ring_light 3-layer lighting parameters
  2. Register random augmentation (MotionBlur, etc.)
  3. Scan entire PCB with camera grid (~143 frames)
  4. Output to per-trigger folder + metadata.json
"""

import asyncio
import json
import os

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

# === Config ===
NUM_TRIGGERS = 3
OUTPUT_ROOT = "INPUT_YOUR_OUTPUT_PATH/good_image"

CAMERA_PATH = "/World/camera_light/Camera"
CAMERA_XFORM_PATH = "/World/camera_light"
RING_LIGHT_ROOT = "/World/camera_light/aoi_ring_light"

RESOLUTION = (1920, 1080)

# PathTracing settings
PT_SPP = 1
PT_TOTAL_SPP = 10

# Scan grid
X_START = 21.6
X_END = -106
Y_START = 23.2
Y_END = -77
STEP = 10
Z = 0
HORIZONTAL_APERTURE = 200.0

# --- Lighting randomization ranges ---
LIGHT_LAYERS = {
    "Inner_Red": {
        "intensity": (5000, 8000),
        "color_r": (0.95, 1.0),
        "color_g": (0.0, 0.05),
        "color_b": (0.0, 0.05),
    },
    "Middle_Green": {
        "intensity": (2000, 4000),
        "color_r": (0.0, 0.05),
        "color_g": (0.95, 1.0),
        "color_b": (0.0, 0.05),
    },
    "Outer_Blue": {
        "intensity": (3000, 6000),
        "color_r": (0.0, 0.05),
        "color_g": (0.0, 0.05),
        "color_b": (0.95, 1.0),
    },
}

# Global light params (applied to all lights)
# cone_angle: angle from center axis to edge (UsdLuxShapingAPI)
#         ╱│╲
#        ╱ │ ╲
#       ╱  │  ╲  ← coneAngle (center axis to edge)
#      ╱   │   ╲
#     ╱    │    ╲
#          │
#          ▼
#        surface
EXPOSURE_RANGE = (0.01, 1.0)
CONE_ANGLE_RANGE = (90, 150)
CONE_SOFTNESS_RANGE = (0.5, 1.0)

# --- Augmentation randomization ---
# Each trigger has a probability of applying augmentation with random params
AUG_MOTION_BLUR_PROB = 0.8
MOTION_BLUR_ALPHA_RANGE = (0.01, 0.3)
MOTION_BLUR_KERNEL_CHOICES = [5, 7]


# === Scan positions ===
def build_scan_positions():
    positions = []
    y = Y_START
    while y >= Y_END:
        x = X_START
        while x >= X_END:
            positions.append((x, y, Z))
            x -= STEP
        y -= STEP
    return positions


# === Lighting randomization ===
def randomize_lighting(stage):
    """Randomize 3-layer light parameters. Returns metadata dict."""
    metadata = {}

    # Global params
    exposure = float(np.random.uniform(*EXPOSURE_RANGE))
    cone_angle = float(np.random.uniform(*CONE_ANGLE_RANGE))
    cone_softness = float(np.random.uniform(*CONE_SOFTNESS_RANGE))
    metadata["global"] = {
        "exposure": exposure,
        "cone_angle": cone_angle,
        "cone_softness": cone_softness,
    }

    for layer_name, ranges in LIGHT_LAYERS.items():
        layer_path = f"{RING_LIGHT_ROOT}/{layer_name}"
        layer_prim = stage.GetPrimAtPath(layer_path)
        if not layer_prim.IsValid():
            print(f"[Warning] Layer not found: {layer_path}")
            continue

        # Random per-layer params
        intensity = float(np.random.uniform(*ranges["intensity"]))
        color_r = float(np.random.uniform(*ranges["color_r"]))
        color_g = float(np.random.uniform(*ranges["color_g"]))
        color_b = float(np.random.uniform(*ranges["color_b"]))
        color = Gf.Vec3f(color_r, color_g, color_b)

        metadata[layer_name] = {
            "intensity": intensity,
            "color": [color_r, color_g, color_b],
        }

        # Apply to all DiskLights under this layer
        _apply_to_all_lights(layer_prim, intensity, color, exposure, cone_angle, cone_softness)

    print(f"[Pipeline] Lighting randomized: exposure={exposure:.2f}, cone={cone_angle:.0f}°")
    for name in LIGHT_LAYERS:
        if name in metadata:
            m = metadata[name]
            print(f"  {name}: intensity={m['intensity']:.0f}, color=({m['color'][0]:.2f},{m['color'][1]:.2f},{m['color'][2]:.2f})")

    return metadata


# === Camera randomization ===
def randomize_camera():
    """Apply small X/Y rotation to orthographic camera via Replicator, simulating PCB tilt."""
    camera = rep.get.prim_at_path(CAMERA_PATH)
    with camera:
        rep.modify.pose(
            rotation=rep.distribution.uniform(
                (-5, -5, -90),
                (5, 5, -90),
            )
        )
    print(f"[Pipeline] Camera rotation randomized (±5° X/Y)")


# === Augmentation setup ===
def setup_augmentation(render_product):
    """Randomly decide whether to apply augmentation per trigger. Returns metadata dict."""
    aug_meta = {"applied": False}

    if np.random.random() < AUG_MOTION_BLUR_PROB:
        alpha = float(np.random.uniform(*MOTION_BLUR_ALPHA_RANGE))
        kernel_size = int(np.random.choice(MOTION_BLUR_KERNEL_CHOICES))

        ldr_color = rep.annotators.get("LdrColor", device="cuda")
        ldr_color = ldr_color.augment("MotionBlur", motionAngle=np.random.uniform(0, 360), strength=alpha, kernelSize=kernel_size)
        ldr_color.attach(render_product)

        aug_meta = {
            "applied": True,
            "type": "MotionBlur",
            "strength": alpha,
            "kernelSize": kernel_size,
        }
        print(f"[Pipeline] Augmentation: MotionBlur strength={alpha:.2f}, kernel={kernel_size}")
    else:
        print(f"[Pipeline] Augmentation: none")

    return aug_meta


def _apply_to_all_lights(layer_prim, intensity, color, exposure, cone_angle, cone_softness):
    """Recursively traverse all DiskLights under a layer and set their parameters."""
    for child in layer_prim.GetChildren():
        if child.IsA(UsdLux.DiskLight):
            light = UsdLux.DiskLight(child)
            light.GetIntensityAttr().Set(intensity)
            light.GetColorAttr().Set(color)
            light.GetExposureAttr().Set(exposure)
            shaping = UsdLux.ShapingAPI(child)
            shaping.GetShapingConeAngleAttr().Set(cone_angle)
            shaping.GetShapingConeSoftnessAttr().Set(cone_softness)
        elif child.GetChildren():
            _apply_to_all_lights(child, intensity, color, exposure, cone_angle, cone_softness)


# === Main pipeline ===
settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", "PathTracing")
settings.set("/rtx/pathtracing/spp", PT_SPP)
settings.set("/rtx/pathtracing/totalSpp", PT_TOTAL_SPP)

stage = omni.usd.get_context().get_stage()
if stage is None:
    print("[Error] No USD stage available.")
else:
    # Set camera aperture
    cam = UsdGeom.Camera.Get(stage, CAMERA_PATH)
    cam.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE)

    positions = build_scan_positions()
    print(f"[Pipeline] {NUM_TRIGGERS} triggers x {len(positions)} positions = {NUM_TRIGGERS * len(positions)} total frames")
    print(f"[Pipeline] Output: {OUTPUT_ROOT}")

    async def run_pipeline():
        for trigger_idx in range(NUM_TRIGGERS):
            trigger_dir = os.path.join(OUTPUT_ROOT, f"trigger_{trigger_idx:04d}")
            os.makedirs(trigger_dir, exist_ok=True)

            # Step 1: Randomize lighting + camera per trigger
            light_meta = randomize_lighting(stage)
            randomize_camera()

            # Step 2: Setup render product, augmentation, and writer for this trigger
            render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)

            # Apply per-trigger augmentation with random params
            aug_meta = setup_augmentation(render_product)

            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=trigger_dir,
                rgb=True,
                instance_id_segmentation=True,
                colorize_instance_id_segmentation=True,
                semantic_types=["class"],
            )
            writer.attach([render_product])

            # Step 3: Scan all grid positions by moving camera xform
            xform = UsdGeom.Xformable.Get(stage, CAMERA_XFORM_PATH)
            translate_op = None
            for op in xform.GetOrderedXformOps():
                if op.GetOpName() == "xformOp:translate":
                    translate_op = op
                    break

            if translate_op is None:
                print("[Error] camera_light missing xformOp:translate")
                return

            for i, (x, y, z) in enumerate(positions):
                translate_op.Set(Gf.Vec3d(x, y, z))
                await rep.orchestrator.step_async()
                if (i + 1) % 20 == 0:
                    print(f"  [{trigger_idx + 1}/{NUM_TRIGGERS}] scan {i + 1}/{len(positions)}")

            # Step 4: Detach writer to flush outputs
            writer.detach()

            # Step 5: Save per-trigger metadata (lighting, augmentation, scan config)
            metadata = {
                "trigger_idx": trigger_idx,
                "num_positions": len(positions),
                "scan_config": {
                    "x_start": X_START, "x_end": X_END,
                    "y_start": Y_START, "y_end": Y_END,
                    "step": STEP, "z": Z,
                },
                "camera": {
                    "horizontal_aperture": HORIZONTAL_APERTURE,
                    "resolution": list(RESOLUTION),
                },
                "lighting": light_meta,
                "augmentation": aug_meta,
                "pathtracing": {"spp": PT_SPP, "totalSpp": PT_TOTAL_SPP},
            }
            meta_path = os.path.join(trigger_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"[Pipeline] Trigger {trigger_idx + 1}/{NUM_TRIGGERS} done → {trigger_dir}")

        print(f"[Pipeline] All done! {NUM_TRIGGERS} triggers completed.")

    asyncio.ensure_future(run_pipeline())
