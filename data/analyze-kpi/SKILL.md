---
name: analyze-kpi
description: "KPI threshold optimization — sweep thresholds on inference scores to find optimal FAR/recall/F1 operating point. Use when evaluating model performance, computing optimal thresholds, or deciding whether to continue a DEFT loop."
---

# Analyze KPI

Threshold analysis for binary classification models (Visual ChangeNet, SiameseOI). Given inference output with siamese scores, sweeps 200+ candidate thresholds and reports optimal operating points.

## What It Does

1. Load inference CSV with `siamese_score` column
2. Evaluate each candidate threshold: compute TP, FP, TN, FN
3. Find two key thresholds:
   - **100% Recall threshold**: lowest FAR while maintaining 100% NO_PASS recall
   - **Best F1 threshold**: maximizes F1 score
4. Generate confusion matrices and score distribution plots
5. Output detailed metrics CSV and missed-sample analysis

## Inputs

- **inference-csv**: CSV with at least `siamese_score` and `label` columns (from Visual ChangeNet inference output)

## Outputs

- `threshold_metrics.csv` — all thresholds with precision/recall/F1/FAR
- `best_f1_missed_no_pass_samples.csv` — false negatives at best-F1 threshold (for gap analysis)
- `summary.txt` — human-readable summary
- `*.png` — confusion matrix and score distribution plots

## Usage (standalone)

```bash
python scripts/analyze_kpi.py \
  --inference-csv /path/to/inference.csv \
  --output-dir /path/to/output/
```

## Key Metrics

- **FAR (False Alarm Rate)**: PASS images misclassified as defective. Lower is better.
- **Recall**: Defective images correctly identified. Higher is better (target: 100%).
- **F1**: Harmonic mean of precision and recall.
- **Threshold**: Siamese distance below which an image pair is classified as non-defective.

## DEFT Integration

Used after each DEFT iteration to evaluate whether the KPI target is met. The `best_f1_missed_no_pass_samples.csv` feeds into gap analysis for the next iteration.
