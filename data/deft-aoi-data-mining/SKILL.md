---
name: deft-aoi-data-mining
description: Mine similar images from source datasets to augment training data using k-NN embedding search, consuming gap analysis results from upstream RCA skill
dependencies:
  - bash
  - docker
  - python3
---

# Data Mining for Training Pipeline

Mine visually similar images from a pre-generated AnomalyGen source pool to augment the next model training iteration. The source pool contains synthetic defect/golden image pairs from prior AnomalyGen runs — NOT real factory data. This skill is the downstream step after the `rca-changenet` skill, which provides the gap analysis (target images representing model weaknesses).

## Resources

```
data-mining/
└── SKILL.md
```

## How It Works

This skill is part of the DEFT Loop: **Evaluate → RCCA → SDG → Retrain → Deploy**.

The upstream RCA skill identifies problematic samples (FP, FN, failure modes, under-represented defect types). This skill takes those real failure images as "targets" and mines pre-generated AnomalyGen synthetic datasets for visually similar samples to strengthen the next training round. Note that the mined output is synthetic data — it contributes to the synthetic data ratio alongside SDG and AnomalyGen arms.

**Runtime environment (local Docker):**
- Embedding generation: `nvcr.io/nvidian/iva/embed:latest` — runs `image_embeddings.py` with `accelerate`
- Nearest neighbor mining: `nvcr.io/nvidian/iva/mining:latest` — RAPIDS cuML/cuDF pre-installed
- SigLIP model: downloaded inside the embed container on first run (cached to a mounted volume for reuse)

> **IMPORTANT:** All data and model paths are local directories mounted into containers via `-v`.

**Pipeline overview:**
```
RCA gap analysis
        │
        ▼
Target Images (directory of problematic samples)
        │
Source Dataset ──→ Parquet Index ──→ Embeddings ──┐
Target Images  ──→ Parquet Index ──→ Embeddings ──┤
                                                   ▼
                                             k-NN Mining
                                                   │
                                                   ▼
                                         mined_similar_files.csv
```

## Prerequisites

### Directory Layout

Ask the user to confirm these local paths before proceeding:

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `<workspace>` | Working directory for intermediate files and model cache | `/home/user/mining_workspace` |
| `<source_dir>` | Source image directory to mine from | `/home/user/datasets/source_images` |
| `<target_dir>` | Target image directory (from RCA) | `/home/user/datasets/target_images` |

The SigLIP model will be downloaded inside the container and cached to `<workspace>/models/` for reuse across runs.

If the user does not provide explicit paths, search the filesystem first:
```bash
find / -type d -name "<dataset_name>" 2>/dev/null
```

## Upstream: RCA Gap Analysis

This skill consumes the output of the `rca-changenet` skill. The RCA report identifies:

- **Failure Mode Clustering** — Every misclassified sample with defect type and failure mode
- **Training Data Gaps** — Unseen defect subtypes that need more data
- **Data Distribution Issues** — Under-represented defect types
- **Fix Priority** — Ranked list of what data to augment

The user will typically provide a **directory of target images** already prepared from RCA findings. The source and target directories should be ready to use as-is — no additional preparation needed.

---

## Step 0: Validate Inputs and Docker Images

Before starting, confirm directories and ensure required Docker images are available locally — pull them if not.

```bash
# Verify directories
ls <source_dir>/ | head -10 && echo "Source total: $(ls <source_dir> | wc -l)"
ls <target_dir>/ | head -10 && echo "Target total: $(ls <target_dir> | wc -l)"
```

> **IMPORTANT:** `--desired-unique-count` is a total count across all targets, not per-target. If you want N images per target, use `N × number_of_target_images`. This value is also capped by source dataset size — do not set it higher than the number of source images.

**Check and pull Docker images before any container step:**

```bash
# Check and pull embed image
if ! docker image inspect nvcr.io/nvidian/iva/embed:latest > /dev/null 2>&1; then
    echo "Pulling embed image..."
    docker pull nvcr.io/nvidian/iva/embed:latest
else
    echo "embed image already present"
fi

# Check and pull mining image
if ! docker image inspect nvcr.io/nvidian/iva/mining:latest > /dev/null 2>&1; then
    echo "Pulling mining image..."
    docker pull nvcr.io/nvidian/iva/mining:latest
else
    echo "mining image already present"
fi
```

---

## Step 0.5: Check Source Parquet Schema

If using a pre-computed source embeddings parquet, **always check its column
names before proceeding**. The embedding column may be `image_embed`, `embedding`,
or something else. The filepath column is usually `filepath` but verify:

```python
import pandas as pd
df = pd.read_parquet("<source_parquet>")
print("Columns:", list(df.columns))
print("Shape:", df.shape)
print(df.head(3))
```

Use the actual column names when passing `--source-embed-column-name` and
`--source-filepath-column-name` to the mining step. **Do not assume defaults.**

---

## Step 1: Create Parquet Indexes

Both source and target datasets need a parquet index (a file listing image paths) before embedding generation.

```python
import os
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

def directory_to_parquet(dir_path: str, output_path: str, path_prefix: str = None) -> str:
    """Index images in dir_path into a parquet file.

    Args:
        dir_path: Local directory containing images.
        output_path: Where to write the parquet index.
        path_prefix: If set, store paths as <path_prefix>/<filename> instead of
                     absolute local paths. Useful when the parquet will be read
                     inside a container with a different mount point.
    """
    filepaths = []
    for f in sorted(os.listdir(dir_path)):
        path = os.path.join(dir_path, f)
        try:
            with Image.open(path) as img:
                if img.mode != "RGB":
                    img.convert("RGB").save(path)
        except Exception:
            continue
        stored_path = f"{path_prefix}/{f}" if path_prefix else path
        filepaths.append(stored_path)

    table = pa.table({"filepath": filepaths})
    pq.write_table(table, output_path)
    print(f"Wrote {len(filepaths)} entries to {output_path}")
    return output_path

# Use container paths so the parquet is readable inside Docker
directory_to_parquet("<source_dir>", "<workspace>/source_index.parquet", path_prefix="/data/source")
directory_to_parquet("<target_dir>", "<workspace>/target_index.parquet", path_prefix="/data/target")
```

> **CRITICAL:** The `path_prefix` must match the container mount point used in subsequent Docker commands. If source is mounted at `/data/source`, use `path_prefix="/data/source"`.

---

## Step 2: Generate Embeddings

Compute embeddings for both source and target datasets.

**Container:** `nvcr.io/nvidian/iva/embed:latest`

> **IMPORTANT — Container startup:** Some hosts cannot execute `/usr/bin/bash` or `/usr/bin/sleep` inside this image (instruction set mismatch). Always start the container with `--entrypoint sh` and `tail -f /dev/null` to keep it alive, then exec with `sh -c`. Never use `bash -c` or `sleep infinity` in this image.

> **IMPORTANT — `image_embeddings.py` bug:** The script has a known bug where `image_embeds` (a 2D numpy array) is passed directly into `pd.DataFrame`, causing `ValueError: Per-column arrays must each be 1-dimensional`. Patch the script after starting the container (see below).

```bash
# Start container using sh entrypoint + tail to keep alive
docker run -d --name embed-worker \
    --gpus all \
    --ipc=host \
    -e HF_TOKEN=${HF_TOKEN:-} \
    --entrypoint sh \
    -v <workspace>:/data/workspace \
    -v <source_dir>:/data/source \
    -v <target_dir>:/data/target \
    nvcr.io/nvidian/iva/embed:latest \
    -c "tail -f /dev/null"

# Patch image_embeddings.py to fix the 2D array bug
docker exec embed-worker sh -c "
    sed -i \"s/'image_embed': image_embeds/'image_embed': list(image_embeds)/\" /embed/image_embeddings.py
    echo 'Patch applied'
"

# Download SigLIP model (cached to <workspace>/models/ — only needed once)
# Uses Python snapshot_download — more reliable than huggingface-cli
docker exec embed-worker sh -c "
    if [ ! -d /data/workspace/models/siglip-base-patch16-224 ]; then
        python3 -m pip install huggingface_hub -q &&
        python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download('google/siglip-base-patch16-224',
                  local_dir='/data/workspace/models/siglip-base-patch16-224')
print('Model downloaded')
\"
    else
        echo 'Model already cached, skipping download'
    fi
"

# Generate source embeddings
docker exec embed-worker sh -c "
    mkdir -p /data/workspace/source_embeddings &&
    cd /embed &&
    python3 -m accelerate.commands.launch --num_processes <num_gpus> image_embeddings.py \
        --input-parquet /data/workspace/source_index.parquet \
        --output-parquet /data/workspace/source_embeddings/embeddings.parquet \
        --model SigLIP \
        --model_path /data/workspace/models/siglip-base-patch16-224
"

# Generate target embeddings
docker exec embed-worker sh -c "
    mkdir -p /data/workspace/target_embeddings &&
    cd /embed &&
    python3 -m accelerate.commands.launch --num_processes <num_gpus> image_embeddings.py \
        --input-parquet /data/workspace/target_index.parquet \
        --output-parquet /data/workspace/target_embeddings/embeddings.parquet \
        --model SigLIP \
        --model_path /data/workspace/models/siglip-base-patch16-224
"

docker stop embed-worker && docker rm embed-worker
```

> **NOTE:** The model is downloaded from https://huggingface.co/google/siglip-base-patch16-224 on first run and cached to `<workspace>/models/siglip-base-patch16-224/`. Subsequent runs skip the download.

**Output parquet schema:**
- `filepath` — Image file path (container path)
- Embedding vector column — name varies by container version (see below)

> **IMPORTANT — Column name varies by container version:** The embed container may write
> the embedding under the column name `embedding` or `image_embed` depending on version
> and whether the 2D-array patch was applied. **Always inspect the actual output before
> passing to mining:**
> ```python
> import pandas as pd
> df = pd.read_parquet("<workspace>/target_embeddings/embeddings.parquet")
> print("Target columns:", list(df.columns))   # check actual embedding column name
> ```
> Similarly check the source parquet schema (Step 0.5). Source and target may use
> **different** column names — pass each explicitly:
> ```
> --source-embed-column-name <source_col>   # e.g. image_embed
> --target-embed-column-name <target_col>   # e.g. embedding
> ```

### Timeline

| Dataset size | GPUs | Time |
|-------------|------|------|
| ~1,000 images | 1 | ~2-5 min |
| ~10,000 images | 8 | ~5-10 min |
| ~100,000 images | 8 | ~30-60 min |

---

## Step 3: Run Nearest Neighbor Mining

Find visually similar images from the source dataset based on target samples.

**Container:** `nvcr.io/nvidian/iva/mining:latest`

> **IMPORTANT — Container startup:** Same as embed — use `--entrypoint sh` and `tail -f /dev/null`. The mining script is at `/mining/nearest_neighbors.py`.

> **IMPORTANT — Output directory permissions:** The mining container runs as `nobody`. Output files written inside the container will be owned by `nobody` and may not be writable from the host. Write the CSV from the host side (Step 4), not from inside the container.

```bash
docker run -d --name mining-worker \
    --gpus all \
    --ipc=host \
    --entrypoint sh \
    -v <workspace>:/data/workspace \
    nvcr.io/nvidian/iva/mining:latest \
    -c "tail -f /dev/null"

docker exec mining-worker sh -c "
    mkdir -p /data/workspace/mining_output &&
    cd /mining &&
    python nearest_neighbors.py \
        --source-parquet /data/workspace/source_embeddings/embeddings.parquet \
        --target-parquet /data/workspace/target_embeddings/embeddings.parquet \
        --output-parquet /data/workspace/mining_output/final_unique_files.parquet \
        --desired-unique-count <count> \
        --knn-metric cosine \
        --source-embed-column-name <source_col> \
        --target-embed-column-name <target_col>
"

docker stop mining-worker && docker rm mining-worker
```

### CLI Reference

**Required parameters:**

| Parameter | Description |
|-----------|-------------|
| `--source-parquet` | Path to source embeddings parquet (or directory of parquets) |
| `--target-parquet` | Path to target embeddings parquet (or directory of parquets) |
| `--output-parquet` | Output parquet file path for results |
| `--desired-unique-count` | Total number of unique source files to retrieve |

**Core options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--knn-metric` | `euclidean` | Distance metric: `euclidean`, `cosine`, `manhattan` |
| `--source-embed-column-name` | `embedding` | Embedding column name in source parquet |
| `--target-embed-column-name` | `embedding` | Embedding column name in target parquet |
| `--source-filepath-column-name` | `filepath` | Filepath column name in source parquet |
| `--target-filepath-column-name` | `filepath` | Filepath column name in target parquet |

**Advanced options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--exclude-parquet` | None | Parquet with `filepath` column of images to exclude from source |
| `--save-embeddings` | False | Include embeddings in output (increases file size) |
| `--mode` | `simple` | Retrieval mode (see below) |
| `--source-detection-file` | None | COCO JSON for source (required for `simple_rare_inclusive`, `class_balanced`) |
| `--target-detection-file` | None | COCO JSON for target (required for `class_balanced`) |
| `--rare-class-list` | None | Comma-separated rare class names (e.g., `"bridge,shift"`) |

**Retrieval modes:**

| Mode | Description | When to use |
|------|-------------|-------------|
| `simple` | Basic k-NN similarity retrieval | Default — good for general augmentation |
| `simple_rare_inclusive` | Similarity + include ALL rare class images | Ensure all rare defect types are represented |
| `class_balanced` | Proportional allocation by class | Target has class imbalance, want balanced augmentation |

> **CRITICAL:** When using `cosine` metric (recommended for SigLIP), the script L2-normalizes embeddings internally. Do not pre-normalize.

> **IMPORTANT:** For `class_balanced` mode, you need COCO-format detection JSON files for both source and target.

### Algorithm: Iterative TMM (Target-Matched Mining)

1. **k-NN search** (GPU via cuML) — Find nearest neighbors per target image
2. **Row-by-row dedup** — No source image assigned to multiple targets
3. **Iterate** — Increase search range until `desired_unique_count` reached (max 8 iterations)

**Output:**
```
<workspace>/mining_output/
├── final_unique_files.parquet    # Main result: unique source filepaths (owned by nobody)
└── *_iteration_*_topn_*.parquet  # Per-iteration detailed results
```

---

## Step 4: Export Results as CSV

> **NOTE:** `final_unique_files.parquet` is owned by `nobody` (written inside the container). Read it fine from the host, but write the CSV to a location the current user owns — use `/tmp` as intermediate if needed, then copy.

```python
import os
import pandas as pd

df = pd.read_parquet("<workspace>/mining_output/final_unique_files.parquet")
print(f"Mined {len(df)} unique similar images")
print(df.head())

# Step 1: translate container paths back to host paths
df["filepath"] = df["filepath"].str.replace("/data/source", "<source_dir>")

# Step 2: verify translated paths exist — parquet may encode special chars differently
# than the actual filesystem (e.g., '+' stored as '_' in some parquet writers).
# Check a sample before writing the CSV:
missing = [p for p in df["filepath"] if not os.path.exists(p)]
if missing:
    print(f"WARNING: {len(missing)}/{len(df)} translated paths not found on disk")
    print("Sample missing:", missing[:3])
    # Common fix: parquet encodes '+' as '_' in filenames
    # Add additional replacements as needed, e.g.:
    # df["filepath"] = df["filepath"].str.replace("_PCB_solder_", "_PCB+solder_")
    # Re-check after each replacement until all paths resolve.

# Write to /tmp first if workspace output dir is owned by nobody
df.to_csv("/tmp/mined_similar_files.csv", index=False)
```

Then copy to final destination:
```bash
cp /tmp/mined_similar_files.csv <workspace>/mining_output/mined_similar_files.csv
```

**Output CSV schema:** single column `filepath` with paths to mined source images.

> **Path encoding pitfall:** Parquet filepath strings may not match the literal filesystem
> paths if the parquet was written with URL-encoding or a different character substitution
> (e.g., `+` in a directory name stored as `_`). Always verify that translated paths
> exist on disk before using them. If a large fraction are missing, inspect a few raw
> values from the parquet and compare against `ls` output to find the encoding pattern.

> **TIP:** Review mined images before adding to training. Use t-SNE visualization or manual spot-checking to verify match quality.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `unrecognized arguments: --output-dir` | Mining CLI uses `--output-parquet <file>`, not `--output-dir <dir>` | Replace `--output-dir /path/to/dir` with `--output-parquet /path/to/dir/results.parquet` |
| `KeyError: 'image_embed'` or `KeyError: 'embedding'` | Embedding column name mismatch between source parquet and target parquet | Inspect both parquets (`print(df.columns)`) and pass the actual column names via `--source-embed-column-name` and `--target-embed-column-name` — they may differ from each other |
| `OOM during k-NN` | Source too large for GPU memory | Reduce `--desired-unique-count` or use larger VRAM GPU |
| `0 entries in parquet` | Empty dir or all images corrupted | Verify: `ls <dir> \| wc -l` |
| `ModuleNotFoundError: cuml` | Not inside mining container | Use `nvcr.io/nvidian/iva/mining:latest` |
| Few results despite large source | `desired-unique-count` exceeds source size | Cap count at source dataset size; algorithm exhausts source after ~6 iterations |
| `ValueError: Per-column arrays must each be 1-dimensional` | `image_embeddings.py` bug — 2D numpy array passed to DataFrame | Patch: `sed -i "s/'image_embed': image_embeds/'image_embed': list(image_embeds)/" /embed/image_embeddings.py` |
| `cannot execute binary file` for bash/sleep | Host CPU missing instructions used in container binaries | Use `--entrypoint sh` and `tail -f /dev/null`; exec with `sh -c` not `bash -c` |
| `PermissionError` writing CSV | Output dir owned by `nobody` (container user) | Write CSV to `/tmp` first, then `cp` to destination |
| `FileNotFoundError` on model path | Model download failed or path wrong | Verify `ls <workspace>/models/siglip-base-patch16-224/` on host |
| Container paths in output CSV | Filepaths reflect container mount points | Translate paths back to host paths (see Step 4) |

---

## Example: Full Pipeline Run

```bash
# === Host paths (adjust to your setup) ===
WORKSPACE=$HOME/mining_workspace
SOURCE_DIR=$HOME/datasets/source_images
TARGET_DIR=$HOME/datasets/target_images     # From RCA gap analysis
NUM_GPUS=1
# Set desired count = N per target × number of targets, capped at source size
SOURCE_COUNT=$(ls $SOURCE_DIR | wc -l)
TARGET_COUNT=$(ls $TARGET_DIR | wc -l)
N_PER_TARGET=5
DESIRED_COUNT=$(( N_PER_TARGET * TARGET_COUNT ))
# Cap at source size
[ $DESIRED_COUNT -gt $SOURCE_COUNT ] && DESIRED_COUNT=$SOURCE_COUNT

mkdir -p $WORKSPACE

# Step 0: Check and pull Docker images
for IMAGE in nvcr.io/nvidian/iva/embed:latest nvcr.io/nvidian/iva/mining:latest; do
    if ! docker image inspect $IMAGE > /dev/null 2>&1; then
        echo "Pulling $IMAGE..."
        docker pull $IMAGE
    else
        echo "$IMAGE already present"
    fi
done

# Step 1: Create parquet indexes (on host, needs pyarrow + Pillow)
python3 -c "
import os
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

def idx(d, o, prefix):
    fps = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        try:
            with Image.open(p) as img:
                if img.mode != 'RGB': img.convert('RGB').save(p)
        except: continue
        fps.append(prefix + '/' + f)
    pq.write_table(pa.table({'filepath': fps}), o)
    print(f'{len(fps)} -> {o}')

idx('$SOURCE_DIR', '$WORKSPACE/source_index.parquet', '/data/source')
idx('$TARGET_DIR', '$WORKSPACE/target_index.parquet', '/data/target')
"

# Step 2: Generate embeddings
docker run -d --name embed-worker \
    --gpus all --ipc=host \
    --entrypoint sh \
    -e HF_TOKEN=${HF_TOKEN:-} \
    -v $WORKSPACE:/data/workspace \
    -v $SOURCE_DIR:/data/source \
    -v $TARGET_DIR:/data/target \
    nvcr.io/nvidian/iva/embed:latest \
    -c "tail -f /dev/null"

# Patch image_embeddings.py (fix 2D array bug)
docker exec embed-worker sh -c "
    sed -i \"s/'image_embed': image_embeds/'image_embed': list(image_embeds)/\" /embed/image_embeddings.py
"

# After embed runs, check BOTH parquets for actual column names before mining:
# SOURCE_EMBED_COL=$(python3 -c "import pandas as pd; df=pd.read_parquet('$WORKSPACE/source_embeddings/embeddings.parquet'); print([c for c in df.columns if c != 'filepath'][0])")
# TARGET_EMBED_COL=$(python3 -c "import pandas as pd; df=pd.read_parquet('$WORKSPACE/target_embeddings/embeddings.parquet'); print([c for c in df.columns if c != 'filepath'][0])")
# echo "Source col: $SOURCE_EMBED_COL  Target col: $TARGET_EMBED_COL"
SOURCE_EMBED_COL=image_embed   # replace with actual value from inspection above
TARGET_EMBED_COL=embedding     # replace with actual value from inspection above

# Download model (only on first run — cached to $WORKSPACE/models/)
docker exec embed-worker sh -c "
    if [ ! -d /data/workspace/models/siglip-base-patch16-224 ]; then
        python3 -m pip install huggingface_hub -q &&
        python3 -c \"
from huggingface_hub import snapshot_download
snapshot_download('google/siglip-base-patch16-224',
                  local_dir='/data/workspace/models/siglip-base-patch16-224')
\"
    fi
"

docker exec embed-worker sh -c "
    mkdir -p /data/workspace/source_embeddings /data/workspace/target_embeddings &&
    cd /embed &&
    python3 -m accelerate.commands.launch --num_processes $NUM_GPUS image_embeddings.py \
        --input-parquet /data/workspace/source_index.parquet \
        --output-parquet /data/workspace/source_embeddings/embeddings.parquet \
        --model SigLIP --model_path /data/workspace/models/siglip-base-patch16-224 &&
    python3 -m accelerate.commands.launch --num_processes $NUM_GPUS image_embeddings.py \
        --input-parquet /data/workspace/target_index.parquet \
        --output-parquet /data/workspace/target_embeddings/embeddings.parquet \
        --model SigLIP --model_path /data/workspace/models/siglip-base-patch16-224
"

docker stop embed-worker && docker rm embed-worker

# Step 3: Mine nearest neighbors
docker run -d --name mining-worker \
    --gpus all --ipc=host \
    --entrypoint sh \
    -v $WORKSPACE:/data/workspace \
    nvcr.io/nvidian/iva/mining:latest \
    -c "tail -f /dev/null"

docker exec mining-worker sh -c "
    mkdir -p /data/workspace/mining_output &&
    cd /mining &&
    python nearest_neighbors.py \
        --source-parquet /data/workspace/source_embeddings/embeddings.parquet \
        --target-parquet /data/workspace/target_embeddings/embeddings.parquet \
        --output-parquet /data/workspace/mining_output/final_unique_files.parquet \
        --desired-unique-count $DESIRED_COUNT \
        --knn-metric cosine \
        --source-embed-column-name $SOURCE_EMBED_COL \
        --target-embed-column-name $TARGET_EMBED_COL
"

docker stop mining-worker && docker rm mining-worker

# Step 4: Export to CSV (write to /tmp first, then copy)
# Always verify translated paths exist on disk before writing CSV
python3 -c "
import os, pandas as pd
df = pd.read_parquet('$WORKSPACE/mining_output/final_unique_files.parquet')
df['filepath'] = df['filepath'].str.replace('/data/source', '$SOURCE_DIR')
missing = [p for p in df['filepath'] if not os.path.exists(p)]
if missing:
    print(f'WARNING: {len(missing)} paths not found — check for filename encoding mismatches')
    print('Sample:', missing[:2])
else:
    print(f'All {len(df)} paths verified on disk')
df.to_csv('/tmp/mined_similar_files.csv', index=False)
print(f'Exported {len(df)} files')
"
cp /tmp/mined_similar_files.csv $WORKSPACE/mined_similar_files.csv
```

---

## Cleanup 

```bash
docker stop embed-worker mining-worker 2>/dev/null
docker rm embed-worker mining-worker 2>/dev/null

# Optional: remove intermediate indexes (keep embeddings for future iterations)
rm -f <workspace>/source_index.parquet <workspace>/target_index.parquet
```
