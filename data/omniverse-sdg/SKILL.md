---
name: omniverse-sdg
description: End-to-end SDG pipeline execution from natural language. Parse user intent → generate YAML config → run via Docker container. Trigger when user wants to generate synthetic PCB AOI images, mentions SDG/synthetic data, specifies defect types (shift/tombstone/sideflip/missing), image counts, annotations (bbox/segmentation), or lighting modes. Also trigger on /sdg-run.
---

# SDG Run

Parse natural language → generate YAML config → confirm → execute pipeline via Docker.

## Paths

| Resource | Path |
|----------|------|
| Docker image | `nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:computex_ver1` |
| Pipeline script (container) | `/home/ubuntu/pcb-aoi/scripts/sdg/standalone/sdg_pipeline.py` |
| Good template | `config/good_image.yaml` |
| Defect template | `config/defect_image.yaml` |
| Missing template | `config/missing_image.yaml` |
| Default config dir | `~/.sdg_config/` |
| Schema reference | [schema.md](schema.md) |

## Workflow

1. **Parse intent** from `$ARGUMENTS` — extract pipeline type (`good` / `defect` / `missing`), image count, defect types, annotations, lighting, resolution. If pipeline type is ambiguous, ask.

   **Cross-pipeline detection:** If the user mentions defect types that span multiple pipelines (e.g. "shift" = `defect` pipeline + "missing" = `missing` pipeline), split into **multiple runs**. Tell the user: "This requires 2 separate runs: one `defect` pipeline (shift only) and one `missing` pipeline. They will run in parallel." Then repeat steps 2–8 for each pipeline, generating a separate config for each. Launch the docker containers in parallel (multiple Bash tool calls in the same message).

2. **Read template** — `config/<type>_image.yaml` and [schema.md](schema.md). (For multi-run: read each template.)
3. **Calculate num_triggers** if user specified image count (see formula below).
4. **Ask user for output path** — if user didn't specify output directory, ask: "Where should the output images be saved?" Use the answer as the host output path and set it in the config's `output` field. The host output path will be mounted into the container at the same path. (For multi-run: use separate subdirectories, e.g. `<output_path>/defect_image/` and `<output_path>/missing_image/`.)
5. **Confirm config location** — default path is `~/.sdg_config/<type>_image_<YYYYMMDD_HHMMSS>.yaml`. Tell the user: "The generated config will be saved to `~/.sdg_config/<filename>.yaml` by default. Press OK to continue, or specify a different path." If user confirms (OK / no objection), use the default. The directory containing the config will be mounted to `/config` in the container.
6. **Generate config** — run `mkdir -p <config_dir>` first. Preserve template structure/comments. Keep defaults for unspecified fields. The config's `output` field must use the host output path (since it's mounted at the same path inside the container).
7. **Show summary** — display config diff and actual image count. (For multi-run: show summary for each config.)
8. **Pre-flight checks & Execute** (no need to ask for confirmation — the Bash tool sandbox will prompt the user):

   **Step 8a — Verify OptiX binary exists:**
   ```bash
   ls /usr/share/nvidia/nvoptix.bin
   ```
   If not found, warn the user and stop.

   **Step 8b — Verify Omniverse credentials are set:**
   ```bash
   echo "OMNI_USER=${OMNI_USER:-(not set)}" && echo "OMNI_PASS=${OMNI_PASS:-(not set)}"
   ```
   If either is empty or `(not set)`, warn the user: "OMNI_USER and OMNI_PASS must be set for Nucleus authentication. Please run `export OMNI_USER=<user>` and `export OMNI_PASS=<pass>` first." Then stop.

   **Step 8c — Verify output directory (must be writable by container):**
   ```bash
   test -d <host_output_path> || mkdir -p <host_output_path>
   chmod 777 <host_output_path>
   ```
   - Directory doesn't exist → `mkdir -p` first.
   - Always `chmod 777` — the container process user (root/ubuntu) differs from the host user, so the directory must be world-writable.
   - If chmod fails, warn user and stop.

   **Step 8d — Verify config directory (must be readable by container):**
   ```bash
   test -d <host_config_dir> || mkdir -p <host_config_dir>
   chmod 755 <host_config_dir>
   ```
   - Directory doesn't exist → `mkdir -p` first.
   - `chmod 755` — container only needs to read the config file.
   - If chmod fails, warn user and stop.

   **Step 8e — Run container:**
   ```bash
   docker run --gpus all --rm --network host \
     -e OMNI_USER -e OMNI_PASS \
     -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
     -v <host_output_path>:<host_output_path> \
     -v <host_config_dir>:/config \
     nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:computex_ver1 \
     "/home/ubuntu/pcb-aoi/scripts/sdg/standalone/sdg_pipeline.py --config /config/<config_filename>.yaml"
   ```

   **Volume mounts (3 required):**
   | # | Host path | Container path | Purpose |
   |---|-----------|----------------|---------|
   | 1 | `/usr/share/nvidia/nvoptix.bin` | `/usr/share/nvidia/nvoptix.bin` (ro) | OptiX ray tracing binary — verify exists before run |
   | 2 | `<host_output_path>` | `<host_output_path>` (same path) | SDG output images & annotations |
   | 3 | `<host_config_dir>` | `/config` | Generated YAML config file |

   **Environment variables:** `OMNI_USER` and `OMNI_PASS` must be set in the host shell for Omniverse Nucleus authentication.

## Pipeline Types

| Type | `pipeline_type` | Description |
|------|-----------------|-------------|
| Good | `good` | Normal PCB images without defects |
| Defect | `defect` | Images with injected defects (shift/tombstone/sideflip) |
| Missing | `missing` | Two-pass scan: reference + defective with hidden components |

## Image Count Calculation

```
grid_positions = floor(abs(x_start - x_end) / step + 1) × floor(abs(y_start - y_end) / step + 1)
num_triggers   = ceil(desired_count / grid_positions)
actual_count   = num_triggers × grid_positions
```

Default grid: `x_start=21.6, x_end=-106, y_start=23.2, y_end=-77, step=10` → 13 × 11 = **143 images/trigger**.

Example: 500 images → `ceil(500/143) = 4` triggers → 572 actual images.

## Keyword → Config Mapping

| Keyword | Config field | Value |
|---|---|---|
| `good` / `normal` | `pipeline_type` | `good` |
| `defect` | `pipeline_type` | `defect` |
| `missing` | `pipeline_type` | `missing` |
| `N images` | `num_triggers` | `ceil(N / grid_positions)` |
| `shift` | `defects.shift.enabled` | `true` |
| `tombstone` | `defects.tombstone.enabled` | `true` |
| `sideflip` | `defects.sideflip.enabled` | `true` |
| `2d bbox` | `writer.bounding_box_2d_tight` | `true` |
| `3d bbox` | `writer.bounding_box_3d` | `true` |
| `segmentation` | `writer.semantic_segmentation` | `true` |
| `instance segmentation` | `writer.instance_id_segmentation` | `true` |
| `depth` | `writer.distance_to_camera` | `true` |
| `white light` | `lighting.ring_light` | `false` |
| `RGB` / `ring light` | `lighting.ring_light` | `true` |
| `4K` | `resolution` | `[3840, 2160]` |
| `1080p` / `FHD` | `resolution` | `[1920, 1080]` |

## Config Rules

- **Good pipeline**: includes `pcba_root`, `component_types`; no `defects` section; `semantic_types: [class]`
- **Defect pipeline**: includes `pcba_root`, `component_types`, `defects`; `semantic_types: [class, defect]`
- **Missing pipeline**: includes `pcba_root`, `component_types`, `missing`; two-pass writer (`writer.reference` + `writer.defective`); `semantic_types: [class, defect]` in reference pass

### Defect Selection (IMPORTANT)

The generated config must always include ALL three defect sections (shift, tombstone, sideflip). Control which defects are active via `enabled`:

- User mentions specific types (e.g. "shift defect") → set those to `enabled: true`, set ALL others to `enabled: false`
- User says "all defects" → all `enabled: true`
- User doesn't mention defect types → keep template defaults

Example: user says "shift defect" →
```yaml
defects:
  shift:
    enabled: true        # user requested
    ratio: 0.3
    translate_range: 0.2
    rotate_z_range: 15
  tombstone:
    enabled: false       # NOT requested → disable
    ratio: 0.2
    angle_min: 15
    angle_max: 45
  sideflip:
    enabled: false       # NOT requested → disable
    ratio: 0.2
    angle_min: 15
    angle_max: 45
```

### Missing Selection

- `missing.ratio`: fraction of components to hide per trigger (default 0.5)
- User can specify ratio, e.g. "hide 30% components" → `missing.ratio: 0.3`

### Annotation Selection

- User mentions specific annotations → enable only those, set the rest to `false`
- User doesn't mention annotations → keep template defaults
- `rgb: true` always stays enabled
