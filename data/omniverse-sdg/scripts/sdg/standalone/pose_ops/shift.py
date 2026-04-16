"""
pose_ops – Shift defect
Run directly in Isaac Sim Script Editor (requires omni.replicator.core 1.13.3)

Small translation + Z-axis rotation to simulate shift defect.
Operates on parent Xform (not instance proxy).
"""

from omni.replicator.core.scripts.functional.modify import TransformToken, pose_ops

from pathlib import Path
import sys

try:
    _base_dir = Path(__file__).resolve().parent
except NameError:
    _base_dir = Path.cwd()
sys.path.insert(0, str(_base_dir))
from utils import uniform_sample_scalar, uniform_sample_vec3, get_prim_and_original_transform, apply_semantic_label


def apply_shift(component_path, translate_range=0.2, rotate_z_range=15):
    """Apply shift defect to a component.

    Args:
        component_path: USD prim path of the target component
        translate_range: max XY translation in mm (default 0.2)
        rotate_z_range: max Z rotation in degrees (default 15)
    """
    prim, original_flat = get_prim_and_original_transform(component_path)
    if prim is None:
        return

    prims = [prim]
    count = len(prims)

    apply_semantic_label(prim, "shift")

    transform_list = [
        (TransformToken.TRANSFORM, original_flat),
        (TransformToken.TRANSLATE, uniform_sample_vec3(count, [-translate_range, -translate_range, 0], [translate_range, translate_range, 0])),
        (TransformToken.ROTATE_Z, uniform_sample_scalar(count, -rotate_z_range, rotate_z_range)),
        (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
    ]

    pose_ops(prims, transform_list)
    print(f"[Done] Shift defect applied to '{component_path}'")


if __name__ == "__main__":
    COMPONENT_PATH = "INPUT_YOUR_TARGET_PRIM_PATH_HERE"
    apply_shift(COMPONENT_PATH)
