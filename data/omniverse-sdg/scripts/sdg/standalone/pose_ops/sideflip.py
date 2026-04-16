"""
pose_ops – Side-flip defect
Run directly in Isaac Sim Script Editor (requires omni.replicator.core 1.13.3)

Component flips sideways, rotating around the bottom edge Y-axis.
Operates on parent Xform (not instance proxy).
"""

import numpy as np
from omni.replicator.core.scripts.functional.modify import TransformToken, pose_ops

from pathlib import Path
import sys

try:
    _base_dir = Path(__file__).resolve().parent
except NameError:
    _base_dir = Path.cwd()
sys.path.insert(0, str(_base_dir))
from utils import uniform_sample_scalar, get_prim_and_original_transform, apply_semantic_label


def apply_sideflip(component_path, angle_min=0, angle_max=30):
    """Apply side-flip defect to a component.

    Args:
        component_path: USD prim path of the target component
        angle_min: minimum flip angle in degrees (default 0)
        angle_max: maximum flip angle in degrees (default 30)
    """
    prim, original_flat = get_prim_and_original_transform(component_path)
    if prim is None:
        return

    prims = [prim]
    count = len(prims)

    apply_semantic_label(prim, "sideflip")

    sign = np.random.choice([-1, 1])
    angle = uniform_sample_scalar(count, angle_min, angle_max) * sign

    transform_list = [
        (TransformToken.TRANSFORM, original_flat),
        (TransformToken.ROTATE_X, angle),
        (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
    ]

    pose_ops(prims, transform_list)
    print(f"[Done] Side-flip applied to '{component_path}' (angle: {float(angle[0]):.1f}°)")


if __name__ == "__main__":
    COMPONENT_PATH = "INPUT_YOUR_TARGET_PRIM_PATH_HERE"
    apply_sideflip(COMPONENT_PATH)
