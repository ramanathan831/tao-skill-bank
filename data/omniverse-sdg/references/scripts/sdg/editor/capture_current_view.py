"""
Capture Current View — capture all annotations at the current camera position.
Run in Isaac Sim Script Editor.
"""

import asyncio
import os

import carb
import omni.replicator.core as rep
import omni.usd
from pxr import Sdf

settings = carb.settings.get_settings()
settings.set("/rtx/pathtracing/spp", 1)
settings.set("/rtx/pathtracing/totalSpp", 32)

CAMERA_PATH = "/World/camera_light/Camera"
RESOLUTION = (1280, 720)
OUTPUT_DIR = "INPUT_YOUR_OUTPUT_DIR_HERE"

os.makedirs(OUTPUT_DIR, exist_ok=True)
render_product = rep.create.render_product(CAMERA_PATH, RESOLUTION)


async def capture():
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True,
        bounding_box_2d_tight=True,
        bounding_box_2d_loose=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=True,
        instance_id_segmentation=True,
        colorize_instance_id_segmentation=True,
        semantic_types=["class", "defect"],
    )
    writer.attach([render_product])
    await rep.orchestrator.step_async()
    writer.detach()
    print(f"[Done] Output: {OUTPUT_DIR}")


asyncio.ensure_future(capture())
