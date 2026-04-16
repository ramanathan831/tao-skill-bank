"""
pose_ops shared utility functions.
"""

import numpy as np
import omni.usd
from omni.replicator.core.scripts.functional.modify import TagCache
from omni.replicator.core.functional import modify as rep_modify


def uniform_sample_scalar(n, low, high):
    return np.random.uniform(low, high, size=n)


def uniform_sample_vec3(n, min_bound, max_bound):
    min_b, max_b = np.array(min_bound), np.array(max_bound)
    return min_b + (max_b - min_b) * np.random.rand(n, 3)


def get_prim_and_original_transform(component_path):
    """Get prim and read its original transform matrix.

    Returns:
        (prim, original_flat) or (None, None) if prim not found
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(component_path)

    if not prim.IsValid():
        print(f"[Error] Prim not found: {component_path}")
        return None, None

    xform_attr = prim.GetAttribute("xformOp:transform")
    original_matrix = xform_attr.Get()
    original_flat = []
    for row in original_matrix:
        for val in row:
            original_flat.append(float(val))
    print(f"[Info] Prim: {component_path}")
    print(f"[Info] Original translate: ({original_flat[12]:.3f}, {original_flat[13]:.3f}, {original_flat[14]:.3f})")

    return prim, original_flat


def apply_semantic_label(prim, defect_type):
    """Apply semantic label; skip if already present.

    Writing semantics triggers USD→Fabric sync, which clears TagCache tags
    on the Fabric layer. Clear TagCache after writing so pose_ops can rebuild tags.

    Args:
        prim: USD prim
        defect_type: defect label string (e.g. "shift", "tombstone", "sideflip")
    """
    if not prim.HasAttribute("semantics:labels:defect"):
        rep_modify.semantics(prim, value={"defect": defect_type})
        print(f"[Info] Applied semantic label: defect={defect_type}")
    else:
        print(f"[Info] Semantic label already exists, skipping")

    # Always clear TagCache to ensure pose_ops can recreate tags
    TagCache._cache.clear()
    TagCache._path_order_cache.clear()
