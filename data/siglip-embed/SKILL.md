---
name: siglip-embed
description: "SigLIP image embedding model for computing visual similarity in mining pipelines. Use when computing image embeddings for similarity search, k-NN mining, or data retrieval."
---

# SigLIP Embed

SigLIP image embedding model for computing visual similarity.

## Overview

Used in SDA mining pipelines to embed gap images and source pool images for k-NN search. Computes dense vector embeddings from images using the SigLIP vision encoder.

## Input / Output

- **Input:** Parquet file with a `filepath` column pointing to images.
- **Output:** Parquet file with `filepath` and `embedding` columns.

## Configuration

- **GPU:** 1 (configurable via `num_gpus`). Uses `accelerate` for distributed inference across multiple GPUs.
- **Model:** Supports `SigLIP`, `CLIP`, and `RADIO` models via the `--model` flag. Defaults to `SigLIP`.

## Container

`nvcr.io/nvidian/iva/embed:latest`

## Usage

The embed action takes an input parquet, computes embeddings for each image, and writes the results to an output parquet. The output parquet can then be fed into `knn-mining` for nearest neighbor search.

## CRITICAL: Patch Before First Use

The container's `image_embeddings.py` has a bug that crashes on first use. **Apply this patch immediately after container start, before running any embeddings:**

```bash
sed -i "s/'image_embed': image_embeds/'image_embed': list(image_embeds)/" /embed/image_embeddings.py
```

Without this patch, you get `ValueError: Per-column arrays must each be 1-dimensional` because `image_embeds` is a 2D numpy array passed directly into `pd.DataFrame`.

## CRITICAL: Output Column Name

The output column is `image_embed`, **NOT** `embedding`. When passing to knn-mining, you MUST specify:
```
--source-embed-column-name image_embed --target-embed-column-name image_embed
```

If you forget these flags, knn-mining raises `KeyError: 'embedding'` because it defaults to looking for an `embedding` column.

## Input Parquet Preparation

Before embedding, you need a parquet with a `filepath` column. To create one from a directory of images:

```python
import pandas as pd
from pathlib import Path

image_dir = Path("/data/images")
rows = [{"filepath": str(p)} for p in image_dir.rglob("*.jpg")]
df = pd.DataFrame(rows)
df.to_parquet("/data/input.parquet", index=False)
```

## Known Issues

**Container entrypoint**: Some hosts cannot execute `/usr/bin/bash` or `/usr/bin/sleep` inside this image (instruction set mismatch). Use `--entrypoint sh` and `tail -f /dev/null` to keep alive, then exec with `sh -c` (not `bash -c`).

**SigLIP model download**: The model is downloaded from HuggingFace on first run. Cache it to a mounted volume (`/data/workspace/models/siglip-base-patch16-224/`) for reuse across runs.
