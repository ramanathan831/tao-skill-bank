---
name: vcn-source-pool
description: "Prepare gap queries and source pool parquets for SigLIP embedding and k-NN mining. Use when setting up inputs for the mining pipeline in a DEFT iteration."
---

# VCN Source Pool

Prepares the inputs for the embedding + mining pipeline. Takes gap cases from gap analysis and the source pool CSV, and outputs two parquets ready for SigLIP embedding.

## Inputs

- **gaps-parquet**: Gap cases from vcn-gap-analysis
- **source-csv**: Large source pool CSV (unlabeled/weakly-labeled dataset)
- **source-images-dir**: S3 URI of source pool images

## Outputs

- **output-target-parquet**: Parquet with gap image paths (for target embedding)
- **output-source-parquet**: Parquet with source pool image paths (for source embedding)
