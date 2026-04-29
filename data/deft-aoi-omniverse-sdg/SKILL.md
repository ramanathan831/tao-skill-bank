---
name: deft-aoi-omniverse-sdg
description: Generate ChangeNet-style paired SDG data (golden + defect + mask) for PCB AOI training. Use this skill whenever
  the user mentions paired images, ChangeNet data, golden/defect pairs, SDG pair generation, PCB defect data, pose defects
  (shift/tombstone/sideflip), missing components, or runs /ov-generate-sdg. Also trigger when user asks about generating synthetic
  defect data, building pair datasets, or creating training data for anomaly detection on PCBs.
license: Apache-2.0
compatibility: Standalone — no external runtime requirements.
metadata:
  author: Sherry Jiang
  version: '0.1'
allowed-tools: Read Bash
tags:
- data
- omniverse
- sdg
dependencies:
- bash
- docker
---

# SDG Pair Data Generator

Generate paired golden/defect images with pixel-aligned masks for ChangeNet training. The pipeline renders synthetic PCB scenes inside an Omniverse-based Docker container, then post-processes the outputs into a `Pair-dataset/{golden, defect, mask}` structure.

Two generation modes exist because the underlying rendering approach differs:

- **Mode A — Pose defects** (shift / tombstone / sideflip): Runs two separate pipelines (good + defect) with the same `random_seed`, so the only visual difference is the component pose. Requires two Docker runs in parallel.
- **Mode B — Missing components**: Runs a single pipeline that outputs both a reference (golden) and defective image per trigger. Only one Docker run needed.

## Resources

| Resource | Path | When to read |
|----------|------|--------------|
| Good pipeline template | `resources/config/good_image.yaml` | Mode A — read as base, copy and customize for the good pipeline |
| Defect pipeline template | `resources/config/defect_image.yaml` | Mode A — read as base, copy and customize for the defect pipeline |
| Missing pipeline template | `resources/config/missing_image.yaml` | Mode B — read as base, copy and customize |
| YAML schema reference | `resources/schema.md` | Read when you need to understand any config field, its type, default, or which pipeline types it applies to |
| Post-process script | `scripts/postprocess/build_pair_dataset.py` | Bundled inside Docker image; no need to read — just call it during post-process step |
| Docker image | `nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature` | All modes |
| Generated config dir | `config/pair_runs/` | Created at runtime for each run |

## Step 0: Parse Intent and Gather Requirements

Before touching any config, extract these from the user's request:

1. **Defect types** — which defects? (shift, tombstone, sideflip, missing, or "all")
   - Pose defects (shift/tombstone/sideflip) → **Mode A**
   - Missing → **Mode B**
   - Both mentioned (e.g., "shift + missing") → split into separate runs, one per mode
2. **Image count** — how many paired images to generate
3. **Output path** — where to write the results. **Ask the user** if not specified.
4. **Random seed** — for reproducibility. If not specified, generate one automatically in Step 3.

Confirm the summary with the user before proceeding:
> "I'll generate **N** paired images using **Mode A (shift, tombstone)** with output at **/path/to/output**. Sound good?"

## Step 1: Verify Environment

Check that the host has the required GPU setup and Docker access:

```bash
# 1. Check GPU availability — need at least 1 GPU with OptiX support
nvidia-smi --list-gpus

# 2. Check OptiX ray-tracing binary exists (required by Omniverse renderer)
ls /usr/share/nvidia/nvoptix.bin

# 3. Check for existing SDG containers that might conflict — do this BEFORE launching anything new
docker ps -a | grep pcb-aoi-ov-sdg

# 4. Check Docker GPU access using the SDG image itself (also confirms image is pulled)
docker run --rm --gpus all --entrypoint nvidia-smi \
  nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature
```

**If `nvoptix.bin` is missing**: The host GPU driver doesn't include OptiX support. The user needs a driver version ≥ 525 with OptiX installed. This is a hard requirement — rendering will fail without it.

**If an existing container is running**: Ask the user whether to stop it or use a different output path to avoid conflicts.

## Step 2: Prepare and Validate Output Directories

### Why this matters

The container runs as user `ubuntu` (uid 1000). Every directory and file it creates will be owned by uid 1000. If the host user is a different uid, they won't be able to `chmod` or delete those files later without `sudo`. The solution: pre-create **all** directories on the host (owned by host user) and set them world-writable **before** any Docker run. The container can then write into them without creating new directories itself.

### Directory map — who creates what

**Mode A (pose defects):**

| Path | Created by | When | Contents |
|------|-----------|------|----------|
| `$OUTPUT_PATH/` | host (Step 2) | pre-flight | top-level output root |
| `$OUTPUT_PATH/good/` | host (Step 2) | pre-flight | mount target for good pipeline |
| `$OUTPUT_PATH/good/trigger_NNNN/` | container (render) | Step 5 | rendered frames per trigger |
| `$OUTPUT_PATH/good/trigger_NNNN/rgb/` | container (render) | Step 5 | good RGB images |
| `$OUTPUT_PATH/defect/` | host (Step 2) | pre-flight | mount target for defect pipeline |
| `$OUTPUT_PATH/defect/trigger_NNNN/` | container (render) | Step 5 | rendered frames per trigger |
| `$OUTPUT_PATH/defect/trigger_NNNN/rgb/` | container (render) | Step 5 | defect RGB images |
| `$OUTPUT_PATH/defect/trigger_NNNN/semantic_segmentation/` | container (render) | Step 5 | segmentation masks |
| `$OUTPUT_PATH/Pair-dataset/` | host (Step 2) | pre-flight | post-process output root |
| `$OUTPUT_PATH/Pair-dataset/golden/` | container (post-process) | Step 6 | paired golden PNGs |
| `$OUTPUT_PATH/Pair-dataset/defect/` | container (post-process) | Step 6 | paired defect PNGs |
| `$OUTPUT_PATH/Pair-dataset/mask/` | container (post-process) | Step 6 | mask PNGs + JSONs |

**Mode B (missing):**

| Path | Created by | When | Contents |
|------|-----------|------|----------|
| `$OUTPUT_PATH/` | host (Step 2) | pre-flight | top-level output root |
| `$OUTPUT_PATH/missing/` | host (Step 2) | pre-flight | mount target for missing pipeline |
| `$OUTPUT_PATH/missing/trigger_NNNN/reference/` | container (render) | Step 5 | reference pass (all components visible) |
| `$OUTPUT_PATH/missing/trigger_NNNN/reference/semantic_segmentation/` | container (render) | Step 5 | segmentation with `defect=missing` labels |
| `$OUTPUT_PATH/missing/trigger_NNNN/defective/` | container (render) | Step 5 | defective pass (missing components hidden) |
| `$OUTPUT_PATH/missing/trigger_NNNN/defective/rgb/` | container (render) | Step 5 | defective RGB images |
| `$OUTPUT_PATH/Pair-dataset/` | host (Step 2) | pre-flight | post-process output root |
| `$OUTPUT_PATH/Pair-dataset/golden/` | container (post-process) | Step 6 | paired golden PNGs |
| `$OUTPUT_PATH/Pair-dataset/defect/` | container (post-process) | Step 6 | paired defect PNGs |
| `$OUTPUT_PATH/Pair-dataset/mask/` | container (post-process) | Step 6 | mask PNGs + JSONs |

### The permission problem

`Pair-dataset/golden/`, `Pair-dataset/defect/`, and `Pair-dataset/mask/` are created by `build_pair_dataset.py` inside the container via `os.makedirs(..., exist_ok=True)`. These subdirectories end up owned by uid 1000 (the container user). After that, a non-root host user **cannot** `chmod -R 777` them because they don't own those dirs.

**Fix: pre-create the Pair-dataset subdirectories on the host too**, so they're already owned by the host user. The container's `os.makedirs(exist_ok=True)` will see they exist and skip creation.

### Create all directories

```bash
OUTPUT_PATH=<output_path>

# Mode A
mkdir -p $OUTPUT_PATH/good \
         $OUTPUT_PATH/defect \
         $OUTPUT_PATH/Pair-dataset/golden \
         $OUTPUT_PATH/Pair-dataset/defect \
         $OUTPUT_PATH/Pair-dataset/mask

# Mode B
mkdir -p $OUTPUT_PATH/missing \
         $OUTPUT_PATH/Pair-dataset/golden \
         $OUTPUT_PATH/Pair-dataset/defect \
         $OUTPUT_PATH/Pair-dataset/mask

# Make everything world-writable so container (uid 1000) can write files into these dirs
chmod -R 777 $OUTPUT_PATH
```

### Validate

```bash
for dir in $OUTPUT_PATH/good $OUTPUT_PATH/defect $OUTPUT_PATH/missing \
           $OUTPUT_PATH/Pair-dataset/golden $OUTPUT_PATH/Pair-dataset/defect $OUTPUT_PATH/Pair-dataset/mask; do
  if [ -d "$dir" ]; then
    touch "$dir/.write_test" && rm "$dir/.write_test" \
      && echo "OK: $dir" \
      || echo "FAIL: $dir — not writable"
  fi
done
```

If any directory fails, fix permissions before proceeding. Do NOT skip this step — a missing or unwritable directory will cause the container to fail mid-render with no clear error message, or create directories owned by uid 1000 that the host user cannot manage later.

## Step 3: Generate Configs

```bash
mkdir -p config/pair_runs/
```

### Generate a random seed (if user didn't specify one)

The seed ensures good and defect renders produce identical scenes except for the defect. This is critical for Mode A — without a matching seed, the golden/defect pairs won't align.

```bash
docker run --rm --entrypoint python3 \
  nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature \
  -c "import random; print(random.randint(0, 99999))"
```

### Calculate `num_triggers`

Each trigger renders multiple viewpoints based on the scan grid:

```
num_triggers = ceil(desired_count / grid_positions)
actual_count = num_triggers × grid_positions
```

Default grid in templates (`step: 10`): **13 × 11 = 143** positions per trigger.

Example: user wants **500** paired frames → `ceil(500 / 143) = 4` triggers → **572** frames actual.

The actual count will always be ≥ the requested count. Tell the user the real number in the Step 4 summary so there are no surprises.

For **missing mode**, if the user only cares about RGB images from the defective pass: the count is still `num_triggers × grid_positions`, but the reference pass may produce fewer images depending on `writer.reference` settings.

### Mode A: Create two configs

Read `resources/config/good_image.yaml` and `resources/config/defect_image.yaml` as templates. For any field you're unsure about, consult `resources/schema.md`. Generate two config files with a timestamp suffix:

- `config/pair_runs/pair_good_<timestamp>.yaml` — set `pipeline_type: good`, output → `<output_path>/good`
- `config/pair_runs/pair_defect_<timestamp>.yaml` — set `pipeline_type: defect`, output → `<output_path>/defect`

**The following fields MUST be identical in both configs** — any mismatch will produce misaligned pairs:

`random_seed`, `seed`, `max_image_count`, `scene`, `num_triggers`, `camera_path`, `camera_xform_path`, `ring_light_root`, `resolution`, `horizontal_aperture`, `pathtracing`, `scan_grid`, `pcba_root`, `component_types`, `lighting`, `camera_rotation`, `augmentation`

### Mode B: Create one config

Read `resources/config/missing_image.yaml` as template. Generate:

- `config/pair_runs/pair_missing_<timestamp>.yaml` — set `pipeline_type: missing`

Enable these writer flags (the post-process script needs both the reference RGB and its segmentation mask):
- `writer.reference.rgb: true`
- `writer.reference.semantic_segmentation: true`
- `writer.defective.rgb: true`

### Defect selection (Mode A only)

The config template has three defect sections (shift, tombstone, sideflip), each with an `enabled` flag. Set them based on user intent:

| User says | shift | tombstone | sideflip |
|-----------|-------|-----------|----------|
| specific defects (e.g., "shift and tombstone") | match request | match request | match request |
| "all defects" | `true` | `true` | `true` |
| doesn't specify | keep template defaults | keep template defaults | keep template defaults |

### Writer config reference

| Mode | Pipeline | `semantic_types` | `rgb` | `semantic_segmentation` | `colorize_semantic_segmentation` |
|------|----------|-------------------|-------|--------------------------|----------------------------------|
| A | Good | `[class]` | `true` | — | — |
| A | Defect | `[class, defect]` | `true` | `true` | `true` |
| B | Reference | `[class, defect]` | `true` | `true` | — |
| B | Defective | — | `true` | — | — |

## Step 4: Show Summary and Confirm

Before executing, present the full run parameters to the user:

```
Mode:          A (pose defects)
Defect types:  shift, tombstone
Random seed:   42857
Num triggers:  4  (→ 572 images with 13×11=143 grid)
Output path:   /data/sdg-run-001
Good config:   config/pair_runs/pair_good_20250413.yaml
Defect config: config/pair_runs/pair_defect_20250413.yaml
```

Wait for user confirmation before proceeding.

## Step 5: Execute Rendering

### Mode A: Run good + defect pipelines in parallel

```bash
# Good pipeline
docker run --gpus all --rm --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v <output_path>/good:<output_path>/good \
  -v <config_dir>:/config:ro \
  nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature \
  "/home/ubuntu/pcb-aoi/scripts/sdg/standalone/sdg_pipeline.py --config /config/<good_config>.yaml" &

# Defect pipeline (in parallel)
docker run --gpus all --rm --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v <output_path>/defect:<output_path>/defect \
  -v <config_dir>:/config:ro \
  nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature \
  "/home/ubuntu/pcb-aoi/scripts/sdg/standalone/sdg_pipeline.py --config /config/<defect_config>.yaml" &

# Wait for both
wait
```

### Mode B: Run single missing pipeline

```bash
docker run --gpus all --rm --network host \
  -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v <output_path>/missing:<output_path>/missing \
  -v <config_dir>:/config:ro \
  nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature \
  "/home/ubuntu/pcb-aoi/scripts/sdg/standalone/sdg_pipeline.py --config /config/<missing_config>.yaml"
```

### Expected Timeline

Rendering time depends on `num_triggers`, resolution, and GPU:

| Triggers | Resolution | Approx. time (1× L40s) |
|----------|-----------|-------------------------|
| 1 | 1024×1024 | ~10 min |
| 5 | 1024×1024 | ~60-90 min |
| 20 | 1024×1024 | ~300-360 min |

Mode A runs two containers in parallel, so total wall time ≈ single pipeline time (not 2×).

### Monitoring

```bash
# Check if containers are still running
docker ps | grep pcb-aoi-ov-sdg

# Check output progress — count rendered triggers
ls <output_path>/good/ | grep trigger | wc -l
ls <output_path>/defect/ | grep trigger | wc -l
```

If a container exits early, check logs:
```bash
docker logs <container_id>
```

## Step 6: Post-Process

Build the paired dataset by matching golden/defect frames for each trigger. The `build_pair_dataset.py` script aligns frames by index, generates binary masks from semantic segmentation, and skips frames where no defect is visible.

### Mode A: Process each trigger

```bash
for trigger_dir in <output_path>/good/trigger_*; do
  trigger_name=$(basename $trigger_dir)
  docker run --rm \
    -v <output_path>:<output_path> \
    --entrypoint python3 \
    nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature \
    /home/ubuntu/pcb-aoi/scripts/postprocess/build_pair_dataset.py \
    --good <output_path>/good/$trigger_name \
    --defect <output_path>/defect/$trigger_name \
    --output <output_path>/Pair-dataset
done
```

### Mode B: Process each trigger

```bash
for trigger_dir in <output_path>/missing/trigger_*; do
  trigger_name=$(basename $trigger_dir)
  docker run --rm \
    -v <output_path>:<output_path> \
    --entrypoint python3 \
    nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:pair-feature \
    /home/ubuntu/pcb-aoi/scripts/postprocess/build_pair_dataset.py \
    --good <output_path>/missing/$trigger_name/reference \
    --defect <output_path>/missing/$trigger_name/defective \
    --output <output_path>/Pair-dataset
done
```

Note: Missing mode uses `semantic_types: [class, defect]` where `defect=missing` marks hidden components in the segmentation output.

After post-processing, fix file permissions — the directories are already host-owned (from Step 2), but the **files** inside are owned by container uid 1000:
```bash
# chmod the files only; dirs are already host-owned
find <output_path>/Pair-dataset -type f -exec chmod 666 {} +
```

## Step 7: Validate and Report Results

```bash
# Count generated pairs
GOLDEN_COUNT=$(ls <output_path>/Pair-dataset/golden/*.png 2>/dev/null | wc -l)
DEFECT_COUNT=$(ls <output_path>/Pair-dataset/defect/*.png 2>/dev/null | wc -l)
MASK_COUNT=$(ls <output_path>/Pair-dataset/mask/*.png 2>/dev/null | wc -l)

echo "Golden: $GOLDEN_COUNT | Defect: $DEFECT_COUNT | Mask: $MASK_COUNT"
```

All three counts should match. If they don't, something went wrong in post-processing.

Report to the user:
- Total paired images generated
- How many frames were skipped (no visible defect in mask)
- Output path: `<output_path>/Pair-dataset/`

### Output Structure

```
Pair-dataset/
  golden/   0000_SolderLight.png, 0001_SolderLight.png, ...    (good RGB)
  defect/   0000_SolderLight.png, 0001_SolderLight.png, ...    (defect RGB)
  mask/     0000_SolderLight.png, 0001_SolderLight.png, ...    (semantic segmentation — defect regions colored)
            0000_SolderLight.json, 0001_SolderLight.json, ...  (color → defect type mapping)
```

Light mode suffix (`SolderLight` or `WhiteLight`) is auto-detected from `metadata.json` (`lighting.ring_light: true` → SolderLight, `false` → WhiteLight). Can be overridden with `--light-mode`.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `nvoptix.bin: No such file` | GPU driver missing OptiX support | Upgrade to driver ≥ 525 with OptiX |
| Container exits immediately | GPU memory insufficient or OptiX init failure | Check `docker logs`; ensure no other GPU-heavy process is running |
| Pair count = 0 after post-process | No defects visible in rendered frames | Increase `num_triggers` or check defect `enabled` flags in config |
| Golden/defect frame count mismatch | Configs not identical (Mode A) | Verify `random_seed` and all shared fields match exactly |
| `Permission denied` writing output | Container user can't write to host dir | Run `chmod -R 777 <output_path>` before execution |
| `chmod: cannot access '<path>': No such file or directory` | Trying to chmod a directory that was never created (e.g., `mkdir` failed silently or was skipped) | Always run `mkdir -p` first, then `chmod`, then validate with the write test in Step 2. Never assume a directory exists just because it's in the config |
| Container can't create `trigger_0000` subdirectory | Output directory doesn't exist on the host — Docker volume mount created it as root-owned, or `mkdir -p` was never run | Re-run Step 2 completely: `mkdir -p` all output dirs, then `chmod -R 777`. Verify with the write test before retrying |
| Mask images are all black | `semantic_segmentation` not enabled in writer | Mode A defect config: set `semantic_segmentation: true` and `colorize_semantic_segmentation: true` |
| `docker: Error response from daemon: could not select device driver` | NVIDIA Container Toolkit not installed | Install `nvidia-container-toolkit` package |
