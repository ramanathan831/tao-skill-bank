---
name: knn-mining
description: "GPU-accelerated k-NN nearest neighbor mining using cuML for similarity-based data retrieval. Use when mining similar images from a source pool or performing nearest-neighbor search on embeddings."
---

# k-NN Mining

GPU-accelerated k-NN nearest neighbor search using cuML and RAPIDS.

## Overview

Used in SDA mining pipelines to find similar source images for each gap/failure case. Given pre-computed embeddings from `siglip-embed`, performs fast GPU-accelerated nearest neighbor lookup to identify the most relevant source samples.

## Input / Output

- **Input:** Source embeddings parquet + target embeddings parquet (both with `embedding` column).
- **Output:** Mined pairs parquet with a `filepath` column containing the matched source file paths.

## Configuration

- **Distance Metrics:** Supports `cosine` (default) and `euclidean` distance metrics via `--knn-metric`.
- **Top-N:** Number of nearest neighbors to return per target sample. Defaults to 5.
- **Label Filtering:** Optional `--filter-by-label` flag to restrict mining to same-label pairs only.
- **GPU:** 1 (uses cuML for GPU-accelerated search).

## Container

`nvcr.io/nvidian/iva/mining:latest`

## Retrieval Modes

| Mode | Description | When to use |
|---|---|---|
| `simple` | Basic k-NN similarity retrieval | Default -- good for general augmentation |
| `simple_rare_inclusive` | Similarity + include ALL rare class images | Ensure all rare defect types are represented |
| `class_balanced` | Proportional allocation by class | Target has class imbalance, want balanced augmentation |

For `simple_rare_inclusive`, pass `--rare-class-list "bridge,shift"` and `--source-detection-file`. For `class_balanced`, also pass `--target-detection-file` (COCO format).

## Usage

The mine action takes source and target embedding parquets, performs k-NN search, and outputs a parquet of mined pairs. The source parquet typically contains the full data pool embeddings, while the target parquet contains gap/failure case embeddings identified during evaluation.

## Known Issues

**Embedding column name mismatch**: siglip-embed outputs `image_embed` column, but knn-mining defaults to `embedding`. Always pass `--source-embed-column-name image_embed --target-embed-column-name image_embed` when using siglip-embed output.

**Container entrypoint**: Same as siglip-embed -- use `--entrypoint sh` and `tail -f /dev/null`. The mining script is at `/mining/nearest_neighbors.py`.

**Output file permissions**: The mining container runs as `nobody`. Output files will be owned by `nobody` and may not be writable from the host. Write CSVs from the host side, not from inside the container.

**desired-unique-count**: This is a **total** count across all targets, not per-target. If you want N images per target, use `N * number_of_target_images`. Capped by source dataset size.

**OOM during k-NN**: Large source datasets may exceed GPU memory. Reduce `--desired-unique-count` or use a larger VRAM GPU.

**Cosine metric normalization**: When using `cosine` metric (recommended for SigLIP), the script L2-normalizes embeddings internally. Do not pre-normalize.
