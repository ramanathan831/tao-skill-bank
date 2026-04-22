---
name: tao-changenet-data-prepare
description: >-
  Prepare dataset CSV for NVIDIA TAO Visual ChangeNet training/validation from
  paired image directories (e.g., SDG output or mined results). Generates
  minimal CSV with input_path, golden_path, label columns. Use when the user
  asks to create a ChangeNet dataset, prepare TAO VCN data, pair OK/NG images,
  convert SDG output, or process mined_similar_files.csv for ChangeNet.
---

# TAO Visual ChangeNet — Data Preparation

Prepare a minimal `input_path,golden_path,label` CSV for the
[NVIDIA TAO Visual ChangeNet](https://docs.nvidia.com/tao/tao-toolkit/text/visual_changenet/)
pipeline from two paired image directories. Typical use case: pairing
SDG-generated NG images with their OK (golden) counterparts for ChangeNet
training.

## Script

The utility script is bundled at
[scripts/generate_csv.py](scripts/generate_csv.py).

Copy it to the target directory before use:

```bash
SKILL_DIR="<workspace>/.cursor/skills/tao-changenet-data-prepare"
cp "${SKILL_DIR}/scripts/generate_csv.py" <target_dir>/generate_csv.py
```

---

## Quick Start

```bash
python generate_csv.py \
  --input-dir <path/to/ng_images> \
  --golden-dir <path/to/ok_images> \
  -o dataset.csv
```

### From SDG Output

SDG output directories contain `original_image/` (source clean images used as
input) and `reconstructed_image/` (synthetic defect images). For VCN pairing,
the mapping depends on the comparison task:

| VCN Task | `--input-dir` | `--golden-dir` |
|----------|---------------|----------------|
| Detect synthetic defects (NG vs OK) | `reconstructed_image/` | `original_image/` |
| Any custom OK/NG pair | your NG dir | your OK dir |

Example — pair SDG reconstructed (NG) against original (OK):

```bash
cd SDG_Result/Demo_PCB
python generate_csv.py \
  --input-dir reconstructed_image \
  --golden-dir original_image \
  --label bridge \
  -o dataset.csv
```

---

## CLI Reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | — | Directory containing input images (typically NG) |
| `--golden-dir` | Yes | — | Directory containing golden/reference images (typically OK) |
| `--output`, `-o` | No | `dataset.csv` | Output CSV path |
| `--label`, `-l` | No | auto-detect | Force label for all rows. If omitted, parses from filename pattern `<prefix>+<label>_<id>.png` (fallback: `NG`) |

### Label Resolution Order

1. `--label` flag (if provided, all rows use this)
2. Parse from filename: `PCB+bridge_00000.png` → `bridge`
3. Fallback: `NG`

---

## Output Format

Minimal CSV compatible with NV_PCB_Siamese and VCN pipelines:

```csv
input_path,golden_path,label
/path/to/reconstructed_image/PCB+bridge_00000.png,/path/to/original_image/PCB+bridge_00000.png,bridge
```

The script pairs files by **exact filename match** between the two directories.
Files without a match are skipped with a warning.

---

## Workflow: Creating VCN Dataset from SDG Output

1. **Run SDG pipeline** (see `sdg-inference` skill) — produces
   `<output_dir>/original_image/` and `<output_dir>/reconstructed_image/`

2. **Generate CSV:**

```bash
python generate_csv.py \
  --input-dir <output_dir>/reconstructed_image \
  --golden-dir <output_dir>/original_image \
  -o <output_dir>/dataset.csv
```

3. **Verify row count:**

```bash
wc -l <output_dir>/dataset.csv
head -3 <output_dir>/dataset.csv
```

4. **Use in VCN training** — pass the CSV path to the VCN dataloader.

---

## Multi-Defect Datasets

For datasets with multiple defect types, run once per defect and concatenate:

```bash
# Generate per-defect CSVs
for defect in bridge excess_solder missing; do
    python generate_csv.py \
      --input-dir SDG_Result/${defect}/reconstructed_image \
      --golden-dir SDG_Result/${defect}/original_image \
      --label ${defect} \
      -o /tmp/${defect}.csv
done

# Merge (keep header from first file only)
head -1 /tmp/bridge.csv > combined_dataset.csv
for f in /tmp/bridge.csv /tmp/excess_solder.csv /tmp/missing.csv; do
    tail -n +2 "$f" >> combined_dataset.csv
done
```

---

## Data Source: Mined CSV

Mining pipelines produce a `mined_similar_files.csv` with a single column:

```csv
filepath
/path/to/AG_OV_aggregate/bridge_defect_PCB+solder_00002.png
/path/to/AG_OV_aggregate/bridge_good_frame0005_tn__0402_H040_612_.png
/path/to/AG_OV_aggregate/shift_defect_PCB+solder_00045.png
```

Filename convention: `<defect_type>_<role>_<rest>.png` where role is `defect`
(NG) or `good` (OK).

**Current status:** The mined CSV is a flat file list. To build a ChangeNet
dataset from it, the images need to be split into per-defect `normal/` and
`abnormal/` directories so the script can pair them via `dirs` mode. The
future directory structure will provide this separation — when it does, use:

```bash
python generate_csv.py \
  --input-dir <abnormal_dir>/<defect_type> \
  --golden-dir <normal_dir>/<defect_type> \
  --label <defect_type> \
  -o dataset.csv
```

---

## CRITICAL: ChangeNet File Naming Convention (`_SolderLight.jpg`)

The TAO ChangeNet dataloader does **NOT** use filenames directly from the CSV.
It constructs the actual file path by combining the `object_name` column with
the `input_map` key and `image_ext` from the spec YAML:

```
actual_path = {images_dir}/{input_path}/{object_name}_{input_map_key}.{image_ext}
```

For the standard PCB AOI spec with `input_map: { SolderLight: 0 }` and
`image_ext: .jpg`, a CSV row with `object_name=PCB+bridge_00000` causes the
dataloader to load:

```
{images_dir}/{input_path}/PCB+bridge_00000_SolderLight.jpg
```

**This means all image files must be named `{stem}_SolderLight.jpg`**, not just
`{stem}.png` or `{stem}.jpg`. If your source images are PNG without the suffix,
you must convert them before training:

```python
from PIL import Image
from pathlib import Path

for png in Path(ng_dir).glob("*.png"):
    stem = png.stem
    jpg_path = png.parent / f"{stem}_SolderLight.jpg"
    Image.open(png).convert("RGB").save(jpg_path, "JPEG", quality=95)
```

And the CSV `object_name` column must contain **only the stem** (e.g.,
`PCB+bridge_00000`), not the full filename with `_SolderLight.jpg`.

**Failure mode:** If files don't follow this convention, training will crash at
the first batch with `FileNotFoundError: Image file wasn't found at ...` but
the error only shows the expected path — not what exists. This is the #1 cause
of silent training failures when integrating synthetic data.

