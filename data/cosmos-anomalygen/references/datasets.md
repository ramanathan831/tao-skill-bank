# Datasets Reference — UC1 / UC2 / UC3

How to obtain a ready-to-use `dataset_dir` for each supported use case.
The table below shows what is pre-packaged on NGC and what must be prepared locally.

| UC | Subject | Anomaly types | NGC provides | Local prep required |
|---|---|---|---|---|
| UC1 | PCB | bridge, missing, excess_solder | Full dataset | None |
| UC2 | Metal surface (Magnetic Tile) | MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Uneven | Nothing | Run `prepare_dataset_uc2.py` |
| UC3 | Mobile phone screen | oil, scratch, stain | masks + `defect_spec.jsonl` | Run `prepare_dataset_uc3.py` for images |

---

## UC1 — PCB

The complete UC1 dataset (anomaly images, masks, clean images, `defect_spec.jsonl`) is
shipped as NGC resources and requires no local preparation.

**NGC resources:**

- `nv-metropolis-dev/metropolis-sdg/cosmos-anomalygen-pcb-dataset:1.0`
- `nv-metropolis-dev/metropolis-sdg/cosmos-anomalygen-pcb-assets:1.0`

---

## UC2 — Metal Surface (Magnetic Tile)

The UC2 dataset is downloaded automatically from the public GitHub repository
[abin24/Magnetic-tile-defect-datasets](https://github.com/abin24/Magnetic-tile-defect-datasets).

**Run the preparation script:**

```bash
python3 -m scripts.utilities.prepare_dataset_uc2 <output_dir>
```

Optional — keep the raw zip for debugging:

```bash
python3 -m scripts.utilities.prepare_dataset_uc2 <output_dir> \
    --keep-download /tmp/magnetic_tile_raw
```

**Output layout:**

```
<output_dir>/
  metal_surface/
    anomaly_image/
      MT_Blowhole/   5 images
      MT_Break/      5 images
      MT_Crack/      5 images
      MT_Fray/       5 images
      MT_Uneven/     5 images
    mask/
      MT_Blowhole/   5 masks
      MT_Break/      5 masks
      MT_Crack/      5 masks
      MT_Fray/       5 masks
      MT_Uneven/     5 masks
    clean_image/     20 images
  defect_spec.jsonl
```

The script selects a curated subset (5 anomaly images + masks per type, 20 clean images)
matching the reference UC2 dataset. Pass `<output_dir>` as `dataset_dir`.

---

## UC3 — Mobile Phone Screen

The UC3 anomaly images come from a Roboflow dataset. Masks and `defect_spec.jsonl`
are shipped as NGC assets and require no local preparation.

### Step 1 — Download the Roboflow zip (manual, browser required)

Roboflow does not support unauthenticated programmatic download.

1. Go to `https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen`
2. Click **Export Dataset** → select **COCO** format → download the zip.

### Step 2 — Run the preparation script

```bash
python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> \
    --zip <path/to/downloaded.zip>
```

Preview without writing files:

```bash
python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> \
    --zip <path/to/downloaded.zip> --dry-run
```

**Output layout (images only — script does not touch masks or defect_spec):**

```
<output_dir>/
  Phone/
    anomaly_image/
      oil/       5 images
      scratch/   5 images
      stain/     5 images
    clean_image/ (images from Roboflow)
```

### Step 3 — Obtain NGC assets

Masks and `defect_spec.jsonl` are available on NGC as
`nv-metropolis-dev/metropolis-sdg/cosmos-anomalygen-glass-dataset:1.0`.
Copy them into the directory produced by Step 2 so the final layout is:

```
<output_dir>/
  Phone/
    anomaly_image/
      oil/       5 images  (from Roboflow)
      scratch/   5 images  (from Roboflow)
      stain/     5 images  (from Roboflow)
    clean_image/            (from Roboflow)
    mask/
      oil/                  (from NGC)
      scratch/              (from NGC)
      stain/                (from NGC)
  defect_spec.jsonl         (from NGC)
```

Pass `<output_dir>` as `dataset_dir` once both steps are complete.
