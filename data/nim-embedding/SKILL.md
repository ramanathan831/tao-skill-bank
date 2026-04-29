---
name: nim-embedding
description: Video embeddings via NVIDIA NIM endpoint for visualization and analysis. Use when computing video embeddings
  for t-SNE visualization or distribution analysis.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: Arif Ahmed
  version: '0.1'
allowed-tools: Read Bash
tags:
- embedding
- video
- nim
---

# NIM Embedding

Computes video embeddings using the NVIDIA NIM embedding service. This is an optional component in the DEFT pipeline, used for t-SNE visualization of data distributions.

## Purpose

After synthetic data is generated and filtered, it can be useful to visualize how the generated data distribution compares to the original training data and the weak/gap data. NIM Embedding computes vector embeddings for each video, which can then be projected into 2D space via t-SNE for visual comparison.

This stage is optional and does not affect the training pipeline. It provides diagnostic insight into whether the synthetic data fills the intended distribution gaps.

## Inputs

- **input_parquet**: Parquet file with a `filepath` column containing paths to video files.
- **nim_endpoint**: URL of the NIM embedding service (user-provided).

## Outputs

- **output_parquet**: Parquet file with `filepath` and `video_embed` columns. The `video_embed` column contains the embedding vector for each video.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| nim_endpoint | (required, user-provided) | URL of the NIM embedding service |

## Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| Connection refused / timeout | NIM endpoint is down or unreachable | Verify nim_endpoint URL and service availability |
| Missing filepath column | Input parquet schema mismatch | Ensure input parquet has a filepath column |
| Embedding dimension mismatch | NIM model version change | Verify the NIM endpoint is running the expected model version |
