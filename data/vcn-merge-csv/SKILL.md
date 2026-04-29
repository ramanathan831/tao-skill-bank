---
name: vcn-merge-csv
description: Merge mined samples with previous training CSV for Visual ChangeNet retraining. Use when combining augmented
  data with existing training data in a DEFT iteration.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: Arif Ahmed
  version: '0.1'
allowed-tools: Read Bash
tags:
- merge
- csv
- vcn
- aoi
- training-data
---

# VCN Merge CSV

Takes mined pairs from k-NN mining, resolves their file paths from the source pool, and merges them with the previous iteration's training CSV. Handles continual_dataset accumulation.

## Inputs

- **mined-parquet**: Mined pairs from knn-mining
- **source-pool-parquet**: Source pool metadata from vcn-source-pool
- **prev-train-csv**: Previous iteration's training CSV (or initial training data)

## Outputs

- **output-csv**: Merged training CSV ready for Visual ChangeNet training
