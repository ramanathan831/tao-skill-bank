"""
Shared utilities for the unified SDG pipeline.
"""

import json
import os

import numpy as np
import carb
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux
from omni.replicator.core.functional import modify as rep_modify


def build_scan_positions(grid):
    positions = []
    y = grid["y_start"]
    while y >= grid["y_end"]:
        x = grid["x_start"]
        while x >= grid["x_end"]:
            positions.append((x, y, grid["z"]))
            x -= grid["step"]
        y -= grid["step"]
    return positions


def _apply_to_all_lights(layer_prim, intensity, color, exposure, cone_angle, cone_softness):
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


def randomize_lighting(stage, cfg):
    light_cfg = cfg["lighting"]
    ring_light_root = cfg["ring_light_root"]
    use_ring_light = light_cfg.get("ring_light", True)
    metadata = {"ring_light": use_ring_light}

    exposure = float(np.random.uniform(*light_cfg["exposure_range"]))
    cone_angle = float(np.random.uniform(*light_cfg["cone_angle_range"]))
    cone_softness = float(np.random.uniform(*light_cfg["cone_softness_range"]))
    metadata["global"] = {
        "exposure": exposure,
        "cone_angle": cone_angle,
        "cone_softness": cone_softness,
    }

    if "layers" in light_cfg:
        layer_names = list(light_cfg["layers"].keys())
    else:
        root_prim = stage.GetPrimAtPath(ring_light_root)
        layer_names = [c.GetName() for c in root_prim.GetChildren()] if root_prim.IsValid() else []

    for layer_name in layer_names:
        layer_path = f"{ring_light_root}/{layer_name}"
        layer_prim = stage.GetPrimAtPath(layer_path)
        if not layer_prim.IsValid():
            print(f"[Warning] Layer not found: {layer_path}")
            continue

        if use_ring_light:
            ranges = light_cfg["layers"][layer_name]
        else:
            ranges = light_cfg["white_light"]

        intensity = float(np.random.uniform(*ranges["intensity"]))
        color_r = float(np.random.uniform(*ranges["color_r"]))
        color_g = float(np.random.uniform(*ranges["color_g"]))
        color_b = float(np.random.uniform(*ranges["color_b"]))
        color = Gf.Vec3f(color_r, color_g, color_b)

        metadata[layer_name] = {
            "intensity": intensity,
            "color": [color_r, color_g, color_b],
        }
        _apply_to_all_lights(layer_prim, intensity, color, exposure, cone_angle, cone_softness)

    mode = "ring_light" if use_ring_light else "white_light"
    print(f"[Pipeline] Lighting randomized ({mode}): exposure={exposure:.2f}, cone={cone_angle:.0f}")
    for name in layer_names:
        if name in metadata:
            m = metadata[name]
            print(f"  {name}: intensity={m['intensity']:.0f}, color=({m['color'][0]:.2f},{m['color'][1]:.2f},{m['color'][2]:.2f})")
    return metadata


def setup_augmentation(render_product, cfg):
    aug_cfg = cfg["augmentation"]["motion_blur"]
    aug_meta = {"applied": False}

    if np.random.random() < aug_cfg["probability"]:
        alpha = float(np.random.uniform(*aug_cfg["alpha_range"]))
        kernel_size = int(np.random.choice(aug_cfg["kernel_choices"]))
        ldr_color = rep.annotators.get("LdrColor", device="cuda")
        ldr_color = ldr_color.augment(
            "MotionBlur",
            motionAngle=np.random.uniform(0, 360),
            strength=alpha,
            kernelSize=kernel_size,
        )
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


async def open_usd_stage(app, usd_path):
    ctx = omni.usd.get_context()
    ctx.disable_save_to_recent_files()
    result, error = await ctx.open_stage_async(usd_path)
    ctx.enable_save_to_recent_files()
    if not result:
        raise RuntimeError(f"Cannot open USD file: {usd_path} ({error})")

    while not app.is_app_ready() or not ctx.get_stage():
        await app.next_update_async()

    while ctx.get_stage_loading_status()[2] > 0:
        await app.next_update_async()

    print(f"[Pipeline] Opened stage: {usd_path}")


def configure_pathtracing(cfg):
    pt = cfg["pathtracing"]
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "PathTracing")
    settings.set("/rtx/pathtracing/spp", pt["spp"])
    settings.set("/rtx/pathtracing/totalSpp", pt["total_spp"])
    print(f"[Pipeline] PathTracing: spp={pt['spp']}, totalSpp={pt['total_spp']}")


def _find_scope_instances(root_prim, scope_name, result):
    """Recursively find all Xform instances under a named Scope."""
    for child in root_prim.GetChildren():
        if child.GetName() == scope_name and child.GetTypeName() == "Scope":
            for instance in child.GetChildren():
                if instance.GetTypeName() == "Xform":
                    result.append(str(instance.GetPath()))
        elif child.GetChildren():
            _find_scope_instances(child, scope_name, result)


def build_component_pool(stage, pcba_root, component_types):
    """Build a list of all component prim paths from the PCBA hierarchy."""
    pcba_prim = stage.GetPrimAtPath(pcba_root)
    if not pcba_prim.IsValid():
        print(f"[Error] PCBA root not found: {pcba_root}")
        return []

    all_paths = []
    for scope_name in component_types:
        paths = []
        _find_scope_instances(pcba_prim, scope_name, paths)
        all_paths.extend(paths)

    print(f"[Pipeline] Component pool: {len(all_paths)} components from {len(component_types)} types")
    return all_paths


def apply_semantics(stage, pcba_root, component_types):
    """Apply semantic labels {class: <scope_name>} to all components."""
    pcba_prim = stage.GetPrimAtPath(pcba_root)
    if not pcba_prim.IsValid():
        print(f"[Error] PCBA root not found: {pcba_root}")
        return 0

    total = 0
    for scope_name in component_types:
        paths = []
        _find_scope_instances(pcba_prim, scope_name, paths)
        for prim_path in paths:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                rep_modify.semantics(prim, value={"class": "capacitor"})
                total += 1
        if paths:
            print(f"  [Semantics] {scope_name}: {len(paths)} prims labeled as capacitor")

    print(f"[Pipeline] Applied semantic labels to {total} components")
    return total


def find_translate_op(stage, camera_xform_path):
    """Find the translate xformOp on the camera xform."""
    xform = UsdGeom.Xformable.Get(stage, camera_xform_path)
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            return op
    raise RuntimeError(f"{camera_xform_path} missing xformOp:translate")


def save_metadata(trigger_dir, metadata):
    with open(os.path.join(trigger_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
