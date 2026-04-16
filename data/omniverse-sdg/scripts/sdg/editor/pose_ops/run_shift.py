"""
Shift defect — run directly in Isaac Sim Script Editor.
Small translation + Z-axis rotation.

Workaround for Fabric/USD sync conflict:
1. Write semantics before pose_ops (skip if already applied)
2. Always clear TagCache to force tag recreation
"""

import omni.usd
import numpy as np
from omni.replicator.core.bindings import _omni_replicator_core as _rep
from omni.replicator.core.scripts.functional.modify import TransformToken, pose_ops, TagCache
from omni.replicator.core.functional import modify as rep_modify

# --- Config ---
COMPONENT_PATH = "/World/pcba_main_s_detail/PCBA/tn__60014242BASEA04_fM9E/_0402_H060/tn__0402_H060_186_"

# --- Helper ---
def uniform_sample_scalar(n, low, high):
    return np.random.uniform(low, high, size=n)

def uniform_sample_vec3(n, min_bound, max_bound):
    min_b, max_b = np.array(min_bound), np.array(max_bound)
    return min_b + (max_b - min_b) * np.random.rand(n, 3)

# --- Get prim & original transform ---
stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath(COMPONENT_PATH)

if not prim.IsValid():
    print(f"[Error] Prim not found: {COMPONENT_PATH}")
else:
    prims = [prim]
    count = len(prims)
    print(f"[Info] Operating on {count} prim(s): {COMPONENT_PATH}")

    xform_attr = prim.GetAttribute("xformOp:transform")
    original_matrix = xform_attr.Get()
    original_flat = []
    for row in original_matrix:
        for val in row:
            original_flat.append(float(val))
    print(f"[Info] Original translate: ({original_flat[12]:.3f}, {original_flat[13]:.3f}, {original_flat[14]:.3f})")

    # Step 1: Write semantic label (skip if already applied)
    if not prim.HasAttribute("semantics:labels:defect"):
        rep_modify.semantics(prim, value={"defect": "shift"})
        print(f"[Info] Applied semantic label: defect=shift")
    else:
        print(f"[Info] Semantic label already exists, skipping")

    # Step 2: Always clear TagCache to ensure pose_ops can recreate tags
    TagCache._cache.clear()
    TagCache._path_order_cache.clear()

    # Step 3: Apply shift transform (Fabric layer)
    transform_list = [
        (TransformToken.TRANSFORM, original_flat),
        (TransformToken.TRANSLATE, uniform_sample_vec3(count, [-0.2, -0.2, 0], [0.2, 0.2, 0])),
        (TransformToken.ROTATE_Z, uniform_sample_scalar(count, -15, 15)),
        (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
    ]

    pose_ops(prims, transform_list)
    print(f"[Done] Shift defect applied to '{COMPONENT_PATH}'")
