"""
Missing component operations: select, hide, restore visibility.
"""

import numpy as np
import omni.usd
from pxr import UsdGeom
from omni.replicator.core.functional import modify as rep_modify


def select_missing_components(component_pool, missing_cfg):
    """Randomly select components to hide. Returns list of prim paths."""
    ratio = missing_cfg["ratio"]
    n = max(1, int(len(component_pool) * ratio))
    selected = np.random.choice(component_pool, size=n, replace=False).tolist()
    print(f"[Pipeline] Missing: {n} components selected ({ratio*100:.1f}%)")
    for p in selected[:10]:
        print(f"  -> {p.split('/')[-1]}")
    if len(selected) > 10:
        print(f"  ... and {len(selected) - 10} more")
    return selected


def apply_missing_semantics(stage, prim_paths):
    """Apply semantic label defect=missing to selected components."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and not prim.HasAttribute("semantics:labels:defect"):
            rep_modify.semantics(prim, value={"defect": "missing"})
    print(f"[Pipeline] Applied defect=missing semantic to {len(prim_paths)} components")


def hide_components(stage, prim_paths):
    """Make components invisible."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def restore_components(stage, prim_paths):
    """Restore component visibility."""
    for path in prim_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeVisible()


def build_writer_kwargs(writer_cfg, output_dir):
    """Build BasicWriter.initialize() kwargs from config dict."""
    kwargs = {"output_dir": output_dir}
    for key in [
        "rgb", "image_output_format",
        "bounding_box_2d_tight", "bounding_box_2d_loose", "bounding_box_3d",
        "semantic_segmentation", "colorize_semantic_segmentation",
        "instance_id_segmentation", "colorize_instance_id_segmentation",
        "distance_to_camera", "distance_to_image_plane", "colorize_depth",
        "semantic_types", "frame_padding",
    ]:
        if key in writer_cfg:
            kwargs[key] = writer_cfg[key]
    return kwargs
