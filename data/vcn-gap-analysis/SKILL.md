---
name: vcn-gap-analysis
description: "Analyze VCN inference results — identify FP/FN gaps and compute evaluation metrics. Use when evaluating VCN performance or identifying failure cases for data augmentation."
---

# VCN Gap Analysis

Compares Visual ChangeNet inference output against ground truth to identify failure cases (false positives and false negatives) and compute evaluation metrics.

Two modes:
- **eval-only**: Compute metrics only (precision, recall, F1, FAR)
- **gap-analysis**: Compute metrics AND output a gaps parquet with failure cases for mining

## Inputs

- **inference-results**: Folder with VCN inference CSV
- **threshold-json**: Optimal threshold from vcn-threshold-optimize
- **kpi-csv**: Ground truth KPI dataset CSV

## Outputs

- **output-metrics**: JSON with evaluation metrics
- **output-gaps**: Parquet with failure cases (gap-analysis mode only)
