"""
Defect operations: shift, tombstone, sideflip via pose_ops.

Tombstone defect uses manual pivot rotation around the component's bottom edge,
computing bounding box to find the long axis and rotating around the fixed end.
"""

import numpy as np
import omni.usd
from pxr import UsdGeom
from omni.replicator.core.scripts.functional.modify import TransformToken, pose_ops, TagCache
from omni.replicator.core.functional import modify as rep_modify


def _get_prim_and_transform(prim_path):
    """Get prim and its original transform as row-major 4x4 numpy array and flat list."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        print(f"[Error] Prim not found: {prim_path}")
        return None, None, None
    matrix = prim.GetAttribute("xformOp:transform").Get()
    M = np.array([[float(v) for v in row] for row in matrix])
    flat = M.flatten().tolist()
    return prim, flat, M


def _compute_tombstone_transform(prim, M_orig, angle_min, angle_max):
    """Compute tombstone pivot rotation matrix.

    Finds the long axis via bounding box, picks a random end to lift,
    and rotates around the opposite (fixed) end as pivot.

    Returns:
        final_flat: flat 16-element list of the final 4x4 transform
        params: dict with angle_deg, sign, long_axis for record storage
    """
    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
    local_bbox = bbox_cache.ComputeLocalBound(prim)
    bbox_range = local_bbox.GetRange()
    vmin = np.array(bbox_range.GetMin())
    vmax = np.array(bbox_range.GetMax())
    center = (vmin + vmax) / 2.0
    half = (vmax - vmin) / 2.0

    long_axis = int(np.argmax(half))
    axis_name = ["X", "Y", "Z"][long_axis]

    sign = int(np.random.choice([-1, 1]))
    angle_deg = float(np.random.uniform(angle_min, angle_max)) * sign
    angle_rad = np.radians(angle_deg)

    # Pivot point: the end that stays on the board (opposite to lifted end)
    pivot_vec = np.zeros(3)
    pivot_vec[long_axis] = -sign
    pivot_point = center + half * pivot_vec

    # Rotation axis: perpendicular to long axis and board normal (Z)
    # long_axis=X -> rotate around Y; long_axis=Y -> rotate around X; long_axis=Z -> rotate around X
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.eye(4)
    if long_axis == 0:  # X -> rotate around Y
        R[0, 0] = c;  R[0, 2] = -s
        R[2, 0] = s;  R[2, 2] = c
    elif long_axis == 1:  # Y -> rotate around X
        R[1, 1] = c;  R[1, 2] = s
        R[2, 1] = -s; R[2, 2] = c
    else:  # Z -> rotate around X
        R[1, 1] = c;  R[1, 2] = s
        R[2, 1] = -s; R[2, 2] = c

    # Pivot rotation: T(-P) @ R @ T(+P) @ M_orig
    T_neg = np.eye(4)
    T_neg[3, 0:3] = -pivot_point
    T_pos = np.eye(4)
    T_pos[3, 0:3] = pivot_point

    M_final = T_neg @ R @ T_pos @ M_orig
    final_flat = M_final.flatten().tolist()

    lifted_end = f"{axis_name}+" if sign > 0 else f"{axis_name}-"
    print(f"[Tombstone] axis={axis_name}, angle={angle_deg:.1f}°, lifted={lifted_end}")

    params = {"angle_deg": angle_deg, "sign": sign, "long_axis": long_axis}
    return final_flat, params


def _apply_semantic(prim, defect_type):
    """Apply semantic label + clear TagCache for Fabric/USD sync."""
    if not prim.HasAttribute("semantics:labels:defect"):
        rep_modify.semantics(prim, value={"defect": defect_type})
    TagCache._cache.clear()
    TagCache._path_order_cache.clear()


def prepare_defects(component_pool, defects_cfg):
    """Select components, generate random params, apply defects, and store params for re-apply.

    Returns:
        defect_records: list of dicts with path, defect type, and stored transform params
    """
    defect_records = []
    used_paths = set()

    for defect_type, cfg in defects_cfg.items():
        if not cfg.get("enabled", False):
            continue

        available = [p for p in component_pool if p not in used_paths]
        n = max(1, int(len(component_pool) * cfg["ratio"]))
        n = min(n, len(available))
        selected = np.random.choice(available, size=n, replace=False).tolist()
        used_paths.update(selected)

        for prim_path in selected:
            prim, original_flat, M_orig = _get_prim_and_transform(prim_path)
            if prim is None:
                continue

            _apply_semantic(prim, defect_type)

            record = {"path": prim_path, "defect": defect_type, "original_flat": original_flat}

            if defect_type == "shift":
                t = cfg["translate_range"]
                rz = cfg["rotate_z_range"]
                translate = np.random.uniform([-t, -t, 0], [t, t, 0], size=(1, 3))
                rotate_z = np.random.uniform(-rz, rz, size=1)
                record["translate"] = translate
                record["rotate_z"] = rotate_z
                transform_list = [
                    (TransformToken.TRANSFORM, original_flat),
                    (TransformToken.TRANSLATE, translate),
                    (TransformToken.ROTATE_Z, rotate_z),
                    (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
                ]
                pose_ops([prim], transform_list)

            elif defect_type == "tombstone":
                final_flat, ts_params = _compute_tombstone_transform(
                    prim, M_orig, cfg["angle_min"], cfg["angle_max"]
                )
                record["tombstone_final_flat"] = final_flat
                record["tombstone_params"] = ts_params
                transform_list = [
                    (TransformToken.TRANSFORM, final_flat),
                ]
                pose_ops([prim], transform_list)

            elif defect_type == "sideflip":
                sign = np.random.choice([-1, 1])
                angle = np.random.uniform(cfg["angle_min"], cfg["angle_max"], size=1) * sign
                record["rotate_x"] = angle
                transform_list = [
                    (TransformToken.TRANSFORM, original_flat),
                    (TransformToken.ROTATE_X, angle),
                    (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
                ]
                pose_ops([prim], transform_list)

            defect_records.append(record)

        print(f"[Pipeline] {defect_type}: {n} components ({cfg['ratio']*100:.1f}%)")

    print(f"[Pipeline] Total defects: {len(defect_records)} / {len(component_pool)} components")
    for d in defect_records[:10]:
        print(f"  {d['defect']:12s} -> {d['path'].split('/')[-1]}")
    if len(defect_records) > 10:
        print(f"  ... and {len(defect_records) - 10} more")

    return defect_records


def reapply_defects(defect_records):
    """Re-apply all defect transforms using stored params (called before each step_async)."""
    stage = omni.usd.get_context().get_stage()
    for record in defect_records:
        prim = stage.GetPrimAtPath(record["path"])
        if not prim.IsValid():
            continue

        original_flat = record["original_flat"]
        defect_type = record["defect"]

        if defect_type == "shift":
            transform_list = [
                (TransformToken.TRANSFORM, original_flat),
                (TransformToken.TRANSLATE, record["translate"]),
                (TransformToken.ROTATE_Z, record["rotate_z"]),
                (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
            ]
        elif defect_type == "tombstone":
            transform_list = [
                (TransformToken.TRANSFORM, record["tombstone_final_flat"]),
            ]
        elif defect_type == "sideflip":
            transform_list = [
                (TransformToken.TRANSFORM, original_flat),
                (TransformToken.ROTATE_X, record["rotate_x"]),
                (TransformToken.PIVOT_TIMES_MINUS_HALF_EXTENT, [0, 0, 1]),
            ]
        else:
            continue

        pose_ops([prim], transform_list)
