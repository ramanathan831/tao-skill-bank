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

## Known Issues

**image_embeddings.py 2D array bug**: The container's `image_embeddings.py` has a known bug where `image_embeds` (a 2D numpy array) is passed directly into `pd.DataFrame`, causing `ValueError: Per-column arrays must each be 1-dimensional`. Patch before use:
```bash
sed -i "s/'image_embed': image_embeds/'image_embed': list(image_embeds)/" /embed/image_embeddings.py
```

**Embedding column name**: The output column name is `image_embed`, NOT `embedding`. When passing to knn-mining, use `--source-embed-column-name image_embed --target-embed-column-name image_embed`.

**Container entrypoint**: Some hosts cannot execute `/usr/bin/bash` or `/usr/bin/sleep` inside this image (instruction set mismatch). Use `--entrypoint sh` and `tail -f /dev/null` to keep alive, then exec with `sh -c` (not `bash -c`).

**SigLIP model download**: The model is downloaded from HuggingFace on first run. Cache it to a mounted volume (`/data/workspace/models/siglip-base-patch16-224/`) for reuse across runs.
