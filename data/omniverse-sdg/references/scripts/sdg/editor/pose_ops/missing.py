"""
Two-Pass single component test — verify missing component workflow.
Run in Isaac Sim Script Editor.

Target component: tn__0402_H040_1271_
Pass 1: all visible → RGB + BBox + Segmentation
Pass 2: hide target → RGB only
"""

import asyncio
import os

import carb
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom

settings = carb.settings.get_settings()
settings.set("/rtx/pathtracing/spp", 1)
settings.set("/rtx/pathtracing/totalSpp", 32)

# === Config ===
CAMERA_PATH = "/World/camera_light/Camera"
CAMERA_XFORM_PATH = "/World/camera_light"
RESOLUTION = (1280, 720)
HORIZONTAL_APERTURE = 200.0
OUTPUT_DIR = "INPUT_YOUR_OUTPUT_DIR_HERE"

# Component to hide
TARGET_PRIM = "INPUT_YOUR_TARGET_PRIM_PATH_HERE"

# === Setup ===
stage = omni.usd.get_context().get_stage()

# Verify prim exists
target = stage.GetPrimAtPath(TARGET_PRIM)
if not target.IsValid():
    print(f"[Error] Prim not found: {TARGET_PRIM}")
else:
    print(f"[Info] Target prim: {TARGET_PRIM}")
    print(f"[Info] Prim type: {target.GetTypeName()}")

# Setup camera
cam = UsdGeom.Camera.Get(stage, CAMERA_PATH)
cam.GetHorizontalApertureAttr().Set(HORIZONTAL_APERTURE)

# Read component bounding box center, move camera there
bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
bbox = bbox_cache.ComputeWorldBound(target)
bbox_range = bbox.ComputeAlignedRange()
center = (bbox_range.GetMin() + bbox_range.GetMax()) / 2
print(f"[Info] Component bbox center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
print(f"[Info] Component bbox min: {bbox_range.GetMin()}")
print(f"[Info] Component bbox max: {bbox_range.GetMax()}")

# Camera position = component center XY, keep Z at 0
CAM_POS = (center[0], center[1], 0)
print(f"[Info] Camera position: {CAM_POS}")

render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def test_two_pass():
    # Move camera above the component
    xform = UsdGeom.Xformable.Get(stage, CAMERA_XFORM_PATH)
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            op.Set(Gf.Vec3d(*CAM_POS))
            break

    # --- Pass 1: Reference (all visible) ---
    print("[Pass 1] Reference - all visible")
    writer_ref = rep.WriterRegistry.get("BasicWriter")
    writer_ref.initialize(
        output_dir=os.path.join(OUTPUT_DIR, "reference"),
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
    )
    writer_ref.attach([render_product])
    await rep.orchestrator.step_async()
    writer_ref.detach()
    print("[Pass 1] Done")

    # --- Hide component ---
    print(f"[Hide] {TARGET_PRIM}")
    UsdGeom.Imageable(target).MakeInvisible()

    # --- Pass 2: Defective (missing component) ---
    print("[Pass 2] Defective - component hidden")
    writer_def = rep.WriterRegistry.get("BasicWriter")
    writer_def.initialize(
        output_dir=os.path.join(OUTPUT_DIR, "defective"),
        rgb=True,
    )
    writer_def.attach([render_product])
    await rep.orchestrator.step_async()
    writer_def.detach()
    print("[Pass 2] Done")

    # --- Restore ---
    UsdGeom.Imageable(target).MakeVisible()
    print(f"[Restore] {TARGET_PRIM} visible again")

    print(f"\n[Done] Output saved to {OUTPUT_DIR}")
    print(f"  reference/  → RGB + bbox + segmentation (component visible)")
    print(f"  defective/  → RGB only (component hidden)")


asyncio.ensure_future(test_two_pass())
