# ChangeNet Data Prepare

Prepare CSV annotations for Visual ChangeNet / SiameseOI training from paired defect/golden image directories.

## What It Does

1. **Light-suffix normalization** -- renames files in-place to add `_SolderLight` if no known AOI light suffix is present
2. **JPG conversion** -- creates `.jpg` copies for non-`.jpg` files (TAO spec hardcodes `image_ext: .jpg`)
3. **Pairing** -- matches files between defect/ and golden/ directories by exact `.jpg` filename
4. **CSV output** -- writes 4-column CSV: `input_path,golden_path,label,object_name`

## Output Format

```csv
input_path,golden_path,label,object_name
data_mining_0402/defect,data_mining_0402/golden,bridge,bridge_PCB+solder_00000
```

TAO builds file paths by concatenation: `{images_dir}/{input_path}/{object_name}_SolderLight.jpg`. The `object_name` must NOT include the light suffix or file extension.

## CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--defect-dir` | Yes | -- | Directory containing defect/input images (NG) |
| `--golden-dir` | Yes | -- | Directory containing golden/reference images (OK) |
| `--output`, `-o` | No | `dataset.csv` | Output CSV path |
| `--root-folder` | No | `data_mining_0402` | Root folder prefix for path columns |
| `--label`, `-l` | No | auto-detect | Force single label for all rows |

## Label Auto-Detection

When `--label` is not provided, labels are auto-detected from filenames using longest-prefix matching against known defect labels (e.g., `bridge`, `excess_solder`, `missing_solder`, `bottle_broken_large`).

## Dependencies

Requires **Pillow** for PNG-to-JPG conversion.

## Known Light Suffixes

Recognized suffixes (no rename needed): `LowAngleLight`, `SolderLight`, `UniformLight`, `WhiteLight`.

## Caveats

- **In-place renaming**: Files are renamed on disk to add `_SolderLight`. Not reversible without backup.
- **Defect rows only**: Generates defect rows. If your pipeline requires balanced PASS/defect data, add PASS rows separately.
- **Multi-defect datasets**: Run once per defect type with `--label`, then concatenate CSVs (keep header from first file only).
