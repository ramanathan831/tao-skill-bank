"""
ComponentInstanceWriter — merges mesh-level instance_id_segmentation into
component-level (Xform) segmentation.

Only handles instance_id_segmentation.  All other annotators (rgb, semantic,
bbox, camera_params, depth …) are left to BasicWriter which runs alongside.

Usage:
    from component_writer import register_component_writer
    register_component_writer()

    # BasicWriter handles everything else
    basic = rep.WriterRegistry.get("BasicWriter")
    basic.initialize(output_dir=..., rgb=True, instance_id_segmentation=False, ...)
    basic.attach([render_product])

    # ComponentInstanceWriter handles only instance segmentation
    comp = rep.WriterRegistry.get("ComponentInstanceWriter")
    comp.initialize(output_dir=..., colorize=True)
    comp.attach([render_product])
"""

import json
import os

import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry


# /World/pcba_main_s_detail/PCBA/tn__60014242BASEA04_fM9E/_0402_H060/tn__0402_H060_339_/...
#   0      1                   2     3                        4            5            6
COMPONENT_XFORM_DEPTH = 7
PCBA_MARKER = "/PCBA/"


def _extract_component_key(prim_path):
    """Map a mesh prim path to its parent component Xform path."""
    if PCBA_MARKER not in prim_path:
        return "BOARD"
    parts = prim_path.split("/")
    if len(parts) >= COMPONENT_XFORM_DEPTH:
        return "/".join(parts[:COMPONENT_XFORM_DEPTH])
    return prim_path


def _parse_label_value(label_val):
    """Extract prim path string from idToLabels value."""
    if isinstance(label_val, str):
        return label_val
    if isinstance(label_val, dict):
        return label_val.get("prim_path", str(label_val))
    return str(label_val)


class ComponentInstanceWriter(Writer):
    """Replicator writer that only outputs component-level instance segmentation.

    Meshes under the same component Xform are merged into a single ID/color.
    Attach alongside BasicWriter (with instance_id_segmentation=False) so
    BasicWriter handles all other annotators unchanged.
    """

    def __init__(self, output_dir, colorize=True, frame_padding=4):
        self.version = "0.0.1"
        self._output_dir = output_dir
        self._frame_padding = frame_padding
        self._colorize = colorize
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = 0

        # Only need instance_id_segmentation (no semanticTypes filter
        # so we get full prim paths in idToLabels)
        self.annotators.append(
            AnnotatorRegistry.get_annotator("instance_id_segmentation"))

    def write(self, data: dict):
        fid = str(self._frame_id).zfill(self._frame_padding)

        if self._frame_id == 0:
            print(f"[ComponentInstanceWriter] data keys: {list(data.keys())}")

        for key in data:
            if key.lower().split("-")[0] == "instance_id_segmentation":
                self._write_component_instance(data[key], fid)

        self._frame_id += 1

    def _write_component_instance(self, seg_data, fid):
        pixel_data = seg_data["data"]
        id_to_labels = seg_data["info"]["idToLabels"]

        if self._frame_id == 0:
            print(f"[ComponentInstanceWriter] pixel shape={pixel_data.shape}, "
                  f"dtype={pixel_data.dtype}")
            sample = dict(list(id_to_labels.items())[:3])
            print(f"[ComponentInstanceWriter] idToLabels sample: {sample}")

        # Build old_id -> new_id remap
        comp_id_map = {}    # component_key -> new_id
        old_to_new = {}     # old_id -> new_id
        comp_labels = {}    # new_id -> component_key
        next_id = 1

        old_to_new[0] = 0
        comp_labels[0] = "BACKGROUND"

        for str_id, label_val in id_to_labels.items():
            old_id = int(str_id)
            if old_id == 0:
                continue
            prim_path = _parse_label_value(label_val)
            comp_key = _extract_component_key(prim_path)

            if comp_key not in comp_id_map:
                comp_id_map[comp_key] = next_id
                comp_labels[next_id] = comp_key
                next_id += 1
            old_to_new[old_id] = comp_id_map[comp_key]

        if self._frame_id == 0:
            print(f"[ComponentInstanceWriter] {len(id_to_labels)} mesh IDs "
                  f"-> {next_id - 1} component IDs")

        # Vectorized remap via LUT
        max_old_id = int(pixel_data.max()) if pixel_data.size > 0 else 0
        lut = np.zeros(max_old_id + 1, dtype=np.uint32)
        for old_id, new_id in old_to_new.items():
            if old_id <= max_old_id:
                lut[old_id] = new_id
        remapped = lut[pixel_data.astype(np.uint32)]

        # Colorized PNG + labels JSON
        h, w = remapped.shape[:2]
        rng = np.random.RandomState(42)
        colors = np.zeros((next_id, 4), dtype=np.uint8)
        colors[0] = [0, 0, 0, 0]
        for i in range(1, next_id):
            colors[i] = [rng.randint(30, 255),
                         rng.randint(30, 255),
                         rng.randint(30, 255), 255]
        color_img = colors[remapped.flatten().clip(0, next_id - 1)]
        self.backend.write_image(
            f"component_instance_{fid}.png", color_img.reshape(h, w, 4))

        # Labels JSON: only colors present in this frame
        present_ids = set(np.unique(remapped))
        labels = {}
        for cid in present_ids:
            c = colors[cid]
            rgba_key = f"({c[0]}, {c[1]}, {c[2]}, {c[3]})"
            labels[rgba_key] = comp_labels[cid]
        out_path = os.path.join(
            self._output_dir, f"component_instance_labels_{fid}.json")
        with open(out_path, "w") as f:
            json.dump(labels, f, indent=2)

    def on_final_frame(self):
        self._frame_id = 0


def register_component_writer():
    """Register ComponentInstanceWriter with the Replicator WriterRegistry."""
    WriterRegistry.register(ComponentInstanceWriter)
