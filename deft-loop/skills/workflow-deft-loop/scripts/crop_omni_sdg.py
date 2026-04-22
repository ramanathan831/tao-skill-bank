#!/usr/bin/env python3
"""
Crop per-component images from SDG frames (defect + matching good reference).

Each image is cropped using its OWN bbox so the component is centered.
Pairs are matched via IoU between defect and good bboxes.
Pairs with no good-frame match are discarded.

Usage:
    python crop_sdg_components.py \
        --defect-dir <sdg_output>/defect \
        --good-dir   <sdg_output>/good \
        --defect-out <sdg_crops>/defect \
        --good-out   <sdg_crops>/good \
        [--padding 30] [--min-iou 0.05] [--max-occlusion 0.5]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image

MIN_CROP_PX = 8


def find_best_match(defect_coords, good_bboxes, img_w, img_h, min_iou):
    """Find the best-matching good-frame bbox via IoU."""
    best_iou = 0.0
    best_coords = None
    for gb in good_bboxes:
        gc = (
            max(0, int(gb["x_min"])),
            max(0, int(gb["y_min"])),
            min(img_w, int(gb["x_max"])),
            min(img_h, int(gb["y_max"])),
        )
        if gc[2] <= gc[0] or gc[3] <= gc[1]:
            continue
        x1 = max(defect_coords[0], gc[0])
        y1 = max(defect_coords[1], gc[1])
        x2 = min(defect_coords[2], gc[2])
        y2 = min(defect_coords[3], gc[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a_area = (defect_coords[2] - defect_coords[0]) * (defect_coords[3] - defect_coords[1])
        b_area = (gc[2] - gc[0]) * (gc[3] - gc[1])
        union = a_area + b_area - inter
        score = inter / union if union > 0 else 0.0
        if score > best_iou:
            best_iou = score
            best_coords = gc
    if best_iou < min_iou:
        return None
    return best_coords


def pad_box(coords, img_w, img_h, padding):
    """Add padding to a bbox, clamped to image bounds."""
    return (
        max(0, coords[0] - padding),
        max(0, coords[1] - padding),
        min(img_w, coords[2] + padding),
        min(img_h, coords[3] + padding),
    )


def process_trigger(trigger_name, defect_root, good_root, defect_out, good_out,
                    padding, min_iou, max_occlusion, max_frames=None):
    defect_dir = defect_root / trigger_name
    good_dir   = good_root / trigger_name

    if not defect_dir.exists():
        print(f"  SKIP: defect dir missing for {trigger_name}")
        return 0
    if not good_dir.exists():
        print(f"  SKIP: good dir missing for {trigger_name}")
        return 0

    rgb_files = sorted(defect_dir.glob("rgb_*.png"))
    if max_frames is not None:
        rgb_files = rgb_files[:max_frames]

    count = 0
    skipped_no_match = 0
    skipped_edge = 0

    for rgb_path in rgb_files:
        frame_num = rgb_path.stem[4:]

        bbox_npy_path    = defect_dir / f"bounding_box_2d_tight_{frame_num}.npy"
        bbox_labels_path = defect_dir / f"bounding_box_2d_tight_labels_{frame_num}.json"
        good_rgb_path    = good_dir   / f"rgb_{frame_num}.png"
        good_bbox_path   = good_dir   / f"bounding_box_2d_tight_{frame_num}.npy"

        if not bbox_npy_path.exists() or not bbox_labels_path.exists():
            continue
        if not good_rgb_path.exists():
            continue
        if not good_bbox_path.exists():
            continue

        defect_img = Image.open(rgb_path).convert("RGB")
        good_img   = Image.open(good_rgb_path).convert("RGB")
        img_w, img_h = defect_img.size

        bboxes = np.load(bbox_npy_path)
        with open(bbox_labels_path) as f:
            labels = json.load(f)

        good_bboxes = np.load(good_bbox_path)

        for comp_idx, bbox in enumerate(bboxes):
            semantic_id = str(int(bbox["semanticId"]))
            occ_ratio   = float(bbox["occlusionRatio"])

            if semantic_id not in labels or "defect" not in labels[semantic_id]:
                continue
            if occ_ratio != -1.0 and occ_ratio > max_occlusion:
                continue

            defect_type = labels[semantic_id]["defect"]

            d_coords = (
                max(0, int(bbox["x_min"])),
                max(0, int(bbox["y_min"])),
                min(img_w, int(bbox["x_max"])),
                min(img_h, int(bbox["y_max"])),
            )

            if (d_coords[2] - d_coords[0]) < MIN_CROP_PX or (d_coords[3] - d_coords[1]) < MIN_CROP_PX:
                skipped_edge += 1
                continue

            g_coords = find_best_match(d_coords, good_bboxes, img_w, img_h, min_iou)
            if g_coords is None:
                skipped_no_match += 1
                continue

            fname = f"{defect_type}_{trigger_name}_frame{frame_num}_comp{comp_idx}.png"

            defect_img.crop(pad_box(d_coords, img_w, img_h, padding)).save(defect_out / fname)
            good_img  .crop(pad_box(g_coords, img_w, img_h, padding)).save(good_out   / fname)
            count += 1

    if skipped_edge:
        print(f"    skipped {skipped_edge} edge-clipped components")
    if skipped_no_match:
        print(f"    skipped {skipped_no_match} with no good-frame match")
    return count


def main():
    parser = argparse.ArgumentParser(description="Crop SDG component pairs")
    parser.add_argument("--defect-dir", required=True, help="SDG defect output dir")
    parser.add_argument("--good-dir", required=True, help="SDG good output dir")
    parser.add_argument("--defect-out", required=True, help="Output dir for defect crops")
    parser.add_argument("--good-out", required=True, help="Output dir for good crops")
    parser.add_argument("--padding", type=int, default=30, help="Padding around bbox (default: 30)")
    parser.add_argument("--min-iou", type=float, default=0.05, help="Min IoU for matching (default: 0.05)")
    parser.add_argument("--max-occlusion", type=float, default=0.5, help="Max occlusion ratio (default: 0.5)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per trigger")
    args = parser.parse_args()

    defect_root = Path(args.defect_dir)
    good_root   = Path(args.good_dir)
    defect_out  = Path(args.defect_out)
    good_out    = Path(args.good_out)

    defect_out.mkdir(parents=True, exist_ok=True)
    good_out.mkdir(parents=True, exist_ok=True)

    triggers = sorted(d.name for d in defect_root.iterdir() if d.is_dir())
    total = 0

    for trigger_name in triggers:
        n = process_trigger(trigger_name, defect_root, good_root, defect_out, good_out,
                            args.padding, args.min_iou, args.max_occlusion, args.max_frames)
        print(f"  {trigger_name}: {n} crops")
        total += n

    print(f"\nTotal SDG crops produced: {total}")


if __name__ == "__main__":
    main()
