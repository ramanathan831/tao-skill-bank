"""Generate TAO VisualChangeNet / SiameseOI CSV.

What TAO looks for with the current spec:

    {images_dir}/{input_path}/{object_name}_SolderLight.jpg
    {images_dir}/{golden_path}/{object_name}_SolderLight.jpg

Concrete example:

If the real files are:

    /data/datasets/mining0402/defect/bridge_PCB+solder_00000_SolderLight.jpg
    /data/datasets/mining0402/golden/bridge_PCB+solder_00000_SolderLight.jpg

then the correct CSV row is:

    data_mining_0402/defect,data_mining_0402/golden,bridge,bridge_PCB+solder_00000

The old mistake was assuming `object_name` itself should contain the full
filename or that TAO would infer the light suffix. It does not. TAO only does
string concatenation.

This script therefore normalizes filenames in-place before writing the CSV:
if a file does not already end with a known AOI light suffix, it is renamed to
add `_SolderLight`.
example usage:
cd DATASET
python generate_csv.py --defect-dir defect --golden-dir golden --output ./dataset.csv
"""

import argparse
import os
from PIL import Image

DEFAULT_ROOT_FOLDER = "data_mining_0402"

KNOWN_LIGHT_SUFFIXES = (
    "LowAngleLight",
    "SolderLight",
    "UniformLight",
    "WhiteLight",
)

KNOWN_LABEL_PREFIXES = (
    "bottle_broken_large",
    "bottle_broken_small",
    "bottle_contamination",
    "hazelnut_crack",
    "hazelnut_cut",
    "hazelnut_hole",
    "hazelnut_print",
    "leather_color",
    "excess_solder",
    "missing_solder",
    "bridge",
    "missing",
    "shift",
)


def _parse_label(filename: str) -> str:
    """Extract class label from '{label}_{original}' filename.

    bottle_broken_large_bottle+broken_large_00000.png -> bottle_broken_large
    bridge_PCB+solder_00000.png                         -> bridge
    excess_solder_PCB+solder_00000.png                  -> excess_solder
    missing_frame0003_tn__0201.png                      -> missing
    """
    stem = _strip_light_suffix(filename.rsplit(".", 1)[0])

    for prefix in sorted(KNOWN_LABEL_PREFIXES, key=len, reverse=True):
        if stem == prefix or stem.startswith(prefix + "_"):
            return prefix

    english_prefix = []
    for token in stem.split("_"):
        if token.isascii() and token.isalpha():
            english_prefix.append(token)
            continue
        break

    if english_prefix:
        return "_".join(english_prefix)

    return stem.split("_", 1)[0]


def _stem(filename: str) -> str:
    return filename.rsplit(".", 1)[0]


def _normalize_stem(stem: str) -> str:
    """Collapse duplicate underscores created by broken filename assembly."""
    return "_".join(part for part in stem.split("_") if part)


def _has_light_suffix(stem: str) -> bool:
    """Return True when the filename stem already ends with a known light."""
    return any(stem.endswith(f"_{suffix}") for suffix in KNOWN_LIGHT_SUFFIXES)


def _strip_light_suffix(stem: str) -> str:
    """Strip the trailing AOI light suffix from a filename stem."""
    for suffix in KNOWN_LIGHT_SUFFIXES:
        marker = f"_{suffix}"
        if stem.endswith(marker):
            return stem[: -len(marker)]
    return stem


def _ensure_jpg_files(directory: str) -> None:
    """Create `.jpg` copies for files that are not already `.jpg`.

    The training spec hardcodes `image_ext: .jpg`, so TAO will only look for
    `.jpg` files even when the source images were `.png` or `.jpeg`.
    """
    converted = 0
    for fname in sorted(os.listdir(directory)):
        src = os.path.join(directory, fname)
        if not os.path.isfile(src):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() == ".jpg":
            continue

        dst = os.path.join(directory, f"{stem}.jpg")
        if os.path.exists(dst):
            continue

        with Image.open(src) as img:
            rgb = img.convert("RGB")
            rgb.save(dst, format="JPEG")
        converted += 1
        print(f"CONVERTED: {fname} -> {stem}.jpg")

    if converted:
        print(f"Created {converted} jpg files in {directory}")


def _normalize_light_suffix(directory: str, default_light: str = "SolderLight") -> None:
    """Rename files in-place to match TAO's light-suffixed filename convention."""
    renamed = 0
    for fname in sorted(os.listdir(directory)):
        src = os.path.join(directory, fname)
        if not os.path.isfile(src):
            continue

        stem, ext = os.path.splitext(fname)
        normalized_stem = _normalize_stem(stem)
        target_stem = (
            normalized_stem
            if _has_light_suffix(normalized_stem)
            else f"{normalized_stem}_{default_light}"
        )
        dst_name = f"{target_stem}{ext}"
        dst = os.path.join(directory, dst_name)
        if dst_name == fname:
            continue
        if os.path.exists(dst):
            raise FileExistsError(
                f"Cannot rename {src} -> {dst}: target already exists"
            )
        os.rename(src, dst)
        renamed += 1
        print(f"RENAMED: {fname} -> {dst_name}")

    if renamed:
        print(f"Normalized {renamed} files in {directory}")


def _collect_pairs(input_dir, golden_dir, label_fn, root_folder: str):
    """Yield (input_dir_name, golden_dir_name, label, object_name) tuples."""
    input_dir_name = os.path.join(
        root_folder, os.path.basename(os.path.normpath(input_dir))
    )
    golden_dir_name = os.path.join(
        root_folder, os.path.basename(os.path.normpath(golden_dir))
    )
    golden_files = {fname for fname in os.listdir(golden_dir) if fname.lower().endswith(".jpg")}
    skipped = 0
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        if fname not in golden_files:
            skipped += 1
            continue
        yield (
            input_dir_name,
            golden_dir_name,
            label_fn(fname),
            _normalize_stem(_strip_light_suffix(_stem(fname))),
        )
    if skipped:
        print(f"WARN: {skipped} files in {input_dir} had no golden match")


def generate_csv(
    defect_dir: str,
    golden_dir: str,
    output_csv: str,
    label_override: str | None = None,
    root_folder: str = DEFAULT_ROOT_FOLDER,
) -> None:
    _normalize_light_suffix(defect_dir)
    _normalize_light_suffix(golden_dir)
    _ensure_jpg_files(defect_dir)
    _ensure_jpg_files(golden_dir)

    label_fn = (lambda _: label_override) if label_override else _parse_label
    rows = list(_collect_pairs(defect_dir, golden_dir, label_fn, root_folder))

    with open(output_csv, "w") as f:
        f.write("input_path,golden_path,label,object_name\n")
        for inp, gld, lbl, obj in rows:
            f.write(f"{inp},{gld},{lbl},{obj}\n")

    print(f"Written {len(rows)} rows to {output_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--defect-dir", required=True)
    p.add_argument("--golden-dir", required=True)
    p.add_argument("--output", "-o", default="dataset.csv")
    p.add_argument("--root-folder", default=DEFAULT_ROOT_FOLDER)
    p.add_argument("--label", "-l", default=None,
                   help="Force single label for all rows")
    args = p.parse_args()
    generate_csv(args.defect_dir, args.golden_dir, args.output,
                 label_override=args.label, root_folder=args.root_folder)


if __name__ == "__main__":
    main()
