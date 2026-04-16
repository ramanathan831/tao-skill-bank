"""
Unified SDG Pipeline — supports good / defect / missing modes via config.
Run via omni.replicator's Kit App.

The pipeline type is determined by the `pipeline_type` field in the YAML config:
  - good:    single-pass scan, no defects
  - defect:  single-pass scan with pose_ops defects (shift/tombstone/sideflip)
  - missing: two-pass scan (reference + defective with hidden components)

Usage:
    ./omni_replicator.sh --no-window --exec \
        "path/sdg_pipeline.py --config path/config.yaml"
"""

import argparse
import asyncio
import os
import sys

import omni.kit.app
import omni.replicator.core as rep
import omni.usd
import yaml
from pxr import Gf, UsdGeom

# Ensure local imports work when executed via --exec
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (
    build_scan_positions,
    randomize_lighting,
    setup_augmentation,
    open_usd_stage,
    configure_pathtracing,
    build_component_pool,
    apply_semantics,
    find_translate_op,
    save_metadata,
)
from component_writer import register_component_writer

# === Parse: only --config ===
parser = argparse.ArgumentParser(description="Unified SDG Pipeline")
parser.add_argument("--config", type=str, required=True, help="YAML config file path")
_cli_args, _ = parser.parse_known_args()

with open(_cli_args.config) as f:
    CFG = yaml.safe_load(f)

PIPELINE_TYPE = CFG.get("pipeline_type", "good")
print(f"[Pipeline] Loaded config: {_cli_args.config} (type: {PIPELINE_TYPE})")

if PIPELINE_TYPE not in ("good", "defect", "missing"):
    raise ValueError(f"Unknown pipeline_type: {PIPELINE_TYPE}. Must be good, defect, or missing.")

# Conditional imports
if PIPELINE_TYPE == "defect":
    from defect_ops import prepare_defects, reapply_defects
elif PIPELINE_TYPE == "missing":
    from missing_ops import (
        select_missing_components,
        apply_missing_semantics,
        hide_components,
        restore_components,
    )

register_component_writer()
_app = omni.kit.app.get_app()


# === Scan functions (one per pipeline type) ===

def _get_writers(writer_cfg, output_dir):
    """Create writer(s) based on config.

    Always creates a BasicWriter for standard annotators.
    When ``component_level_segmentation`` is enabled, also creates a
    ComponentInstanceWriter that merges mesh-level IDs to component-level.
    BasicWriter's own instance_id_segmentation is disabled to avoid duplicate
    output.

    Returns a list of writers.
    """
    cfg = dict(writer_cfg)
    cfg.pop("semantic_filter_predicate", None)
    use_component_seg = cfg.pop("component_level_segmentation", False)

    writers = []

    # ComponentInstanceWriter for merged instance segmentation
    if use_component_seg and cfg.get("instance_id_segmentation", False):
        comp_writer = rep.WriterRegistry.get("ComponentInstanceWriter")
        comp_writer.initialize(
            output_dir=output_dir,
            colorize=cfg.get("colorize_instance_id_segmentation", True),
            frame_padding=cfg.get("frame_padding", 4),
        )
        writers.append(comp_writer)
        # Disable in BasicWriter to avoid duplicate work
        cfg["instance_id_segmentation"] = False
        cfg["colorize_instance_id_segmentation"] = False

    basic_writer = rep.WriterRegistry.get("BasicWriter")
    basic_writer.initialize(output_dir=output_dir, **cfg)
    writers.append(basic_writer)

    return writers


async def _scan_good(trigger_dir, render_product, translate_op, positions, pt_total_spp,
                     trigger_idx, num_triggers, light_meta, aug_meta):
    writers = _get_writers(CFG["writer"], trigger_dir)
    for w in writers:
        w.attach([render_product])

    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        await rep.orchestrator.step_async(rt_subframes=pt_total_spp, delta_time=0.0)
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] scan {i + 1}/{len(positions)}")

    for w in writers:
        w.detach()

    metadata = {
        "trigger_idx": trigger_idx,
        "num_positions": len(positions),
        "config": CFG,
        "lighting": light_meta,
        "augmentation": aug_meta,
    }
    save_metadata(trigger_dir, metadata)


async def _scan_defect(trigger_dir, render_product, translate_op, positions, pt_total_spp,
                       trigger_idx, num_triggers, light_meta, aug_meta, defect_records):
    writers = _get_writers(CFG["writer"], trigger_dir)
    for w in writers:
        w.attach([render_product])

    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        reapply_defects(defect_records)
        await rep.orchestrator.step_async(rt_subframes=pt_total_spp, delta_time=0.0)
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] scan {i + 1}/{len(positions)}")

    for w in writers:
        w.detach()

    defect_meta = [{"path": r["path"], "defect": r["defect"]} for r in defect_records]
    metadata = {
        "trigger_idx": trigger_idx,
        "num_positions": len(positions),
        "config": CFG,
        "lighting": light_meta,
        "augmentation": aug_meta,
        "defects": defect_meta,
    }
    save_metadata(trigger_dir, metadata)


async def _scan_missing(trigger_dir, render_product, translate_op, positions, pt_total_spp,
                        trigger_idx, num_triggers, light_meta, aug_meta,
                        stage, component_pool):
    ref_dir = os.path.join(trigger_dir, "reference")
    def_dir = os.path.join(trigger_dir, "defective")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(def_dir, exist_ok=True)

    missing_paths = select_missing_components(component_pool, CFG["missing"])
    apply_missing_semantics(stage, missing_paths)

    # --- Pass 1: Reference (all visible) ---
    print(f"[Pipeline] Pass 1: reference (all visible, {len(positions)} positions)")
    ref_writers = _get_writers(CFG["writer"]["reference"], ref_dir)
    for w in ref_writers:
        w.attach([render_product])

    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        await rep.orchestrator.step_async(rt_subframes=pt_total_spp, delta_time=0.0)
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] reference {i + 1}/{len(positions)}")

    for w in ref_writers:
        w.detach()

    # --- Hide missing components ---
    print(f"[Pipeline] Hiding {len(missing_paths)} components")
    hide_components(stage, missing_paths)

    # --- Pass 2: Defective (components hidden) ---
    print(f"[Pipeline] Pass 2: defective ({len(positions)} positions)")
    def_writers = _get_writers(CFG["writer"]["defective"], def_dir)
    for w in def_writers:
        w.attach([render_product])

    for i, (x, y, z) in enumerate(positions):
        translate_op.Set(Gf.Vec3d(x, y, z))
        await rep.orchestrator.step_async(rt_subframes=pt_total_spp, delta_time=0.0)
        if (i + 1) % 20 == 0:
            print(f"  [{trigger_idx + 1}/{num_triggers}] defective {i + 1}/{len(positions)}")

    for w in def_writers:
        w.detach()

    # --- Restore visibility ---
    restore_components(stage, missing_paths)
    print(f"[Pipeline] Components restored")

    metadata = {
        "trigger_idx": trigger_idx,
        "defect_type": "missing",
        "num_positions": len(positions),
        "config": CFG,
        "lighting": light_meta,
        "augmentation": aug_meta,
        "missing_components": missing_paths,
    }
    save_metadata(trigger_dir, metadata)


# === Main pipeline ===
async def run_pipeline():
    usd_path = CFG["scene"]
    if not usd_path.startswith("omniverse://"):
        usd_path = os.path.abspath(usd_path)
    output_root = CFG["output"]
    num_triggers = CFG["num_triggers"]
    camera_path = CFG["camera_path"]
    camera_xform_path = CFG["camera_xform_path"]
    resolution = tuple(CFG["resolution"])
    pt_total_spp = CFG["pathtracing"]["total_spp"]

    await open_usd_stage(_app, usd_path)
    configure_pathtracing(CFG)

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage available.")

    cam = UsdGeom.Camera.Get(stage, camera_path)
    cam.GetHorizontalApertureAttr().Set(float(CFG["horizontal_aperture"]))

    render_product = rep.create.render_product(camera_path, resolution)

    # Warmup (defect and missing need Fabric initialization)
    if PIPELINE_TYPE in ("defect", "missing"):
        print("[Pipeline] Warmup steps to initialize Fabric + annotators...")
        for _ in range(5):
            await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0)
        print("[Pipeline] Warmup done (5 steps)")

    # Component pool (defect and missing need it)
    component_pool = None
    if PIPELINE_TYPE in ("defect", "missing"):
        component_pool = build_component_pool(stage, CFG["pcba_root"], CFG["component_types"])
        if not component_pool:
            print("[Error] No components found. Check pcba_root and component_types in config.")
            _app.shutdown()
            return

    # Good pipeline: apply class semantics
    if PIPELINE_TYPE == "good" and "pcba_root" in CFG and "component_types" in CFG:
        apply_semantics(stage, CFG["pcba_root"], CFG["component_types"])

    # Defect pipeline: prepare defects once
    defect_records = None
    if PIPELINE_TYPE == "defect":
        defect_records = prepare_defects(component_pool, CFG["defects"])

    positions = build_scan_positions(CFG["scan_grid"])
    total_frames = num_triggers * len(positions)
    if PIPELINE_TYPE == "missing":
        total_frames *= 2
    print(f"[Pipeline] {num_triggers} triggers x {len(positions)} positions = {total_frames} total frames")
    print(f"[Pipeline] Output: {output_root}")

    translate_op = find_translate_op(stage, camera_xform_path)

    # Setup camera randomizer
    camera = rep.get.prim_at_path(camera_path)
    cam_rot = CFG["camera_rotation"]

    for trigger_idx in range(num_triggers):
        trigger_dir = os.path.join(output_root, f"trigger_{trigger_idx:04d}")
        os.makedirs(trigger_dir, exist_ok=True)

        # Shared: randomize lighting
        light_meta = randomize_lighting(stage, CFG)

        # Shared: randomize camera rotation
        with camera:
            rep.modify.pose(
                rotation=rep.distribution.uniform(
                    (cam_rot["x_range"][0], cam_rot["y_range"][0], cam_rot["z_fixed"]),
                    (cam_rot["x_range"][1], cam_rot["y_range"][1], cam_rot["z_fixed"]),
                )
            )
        print(f"[Pipeline] Camera rotation randomized")

        # Shared: augmentation
        aug_meta = setup_augmentation(render_product, CFG)

        # Dispatch scan based on pipeline type
        scan_args = dict(
            trigger_dir=trigger_dir,
            render_product=render_product,
            translate_op=translate_op,
            positions=positions,
            pt_total_spp=pt_total_spp,
            trigger_idx=trigger_idx,
            num_triggers=num_triggers,
            light_meta=light_meta,
            aug_meta=aug_meta,
        )

        if PIPELINE_TYPE == "good":
            await _scan_good(**scan_args)
        elif PIPELINE_TYPE == "defect":
            await _scan_defect(**scan_args, defect_records=defect_records)
        elif PIPELINE_TYPE == "missing":
            await _scan_missing(**scan_args, stage=stage, component_pool=component_pool)

        print(f"[Pipeline] Trigger {trigger_idx + 1}/{num_triggers} done -> {trigger_dir}")

    render_product.destroy()
    print(f"[Pipeline] All done! {num_triggers} triggers completed.")
    _app.shutdown()


# === Schedule ===
asyncio.ensure_future(run_pipeline())
