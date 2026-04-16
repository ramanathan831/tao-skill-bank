---
name: vcn-threshold-optimize
description: "Sweep siamese score thresholds to find optimal FAR/recall operating point for Visual ChangeNet. Use when optimizing classification thresholds after VCN inference."
---

# VCN Threshold Optimize

Sweeps candidate thresholds on Visual ChangeNet inference scores to find the optimal operating point that satisfies a minimum recall constraint while minimizing False Alarm Rate (FAR).

## Inputs

- **inference-results**: Folder containing VCN inference output (CSV with `siamese_score` column)
- **min-recall**: Minimum NO_PASS recall constraint (default: 1.0 = 100%)

## Outputs

- **output-json**: JSON file with optimal threshold, FAR, recall, F1, and per-threshold metrics
