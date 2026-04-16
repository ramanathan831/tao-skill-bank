#!/usr/bin/env python3
"""Classify VCN inference results and identify FP/FN gaps.

Reads a VCN inference CSV (searched recursively under results_dir),
applies a threshold to classify samples as PASS / NO_PASS, and
compares against ground-truth labels in the KPI CSV. Mismatches
(false positives / false negatives) are emitted as gap records.

Two modes are supported:
  - ``eval-only``: compute and output metrics only (accuracy,
    precision, recall, F1, confusion matrix).
  - ``gap-analysis``: metrics plus a parquet of per-image gap
    filepaths (expanded per lighting condition from the train
    config YAML).

Supports both local paths and S3 URIs (s3://) via fsspec.
"""
import argparse
import json
import os

import fsspec
import pandas as pd


def _is_remote(path):
    return "://" in path


def _find_csv(results_dir):
    """Find *.csv recursively under results_dir (local or S3)."""
    if _is_remote(results_dir):
        fs, _ = fsspec.core.url_to_fs(results_dir)
        root = results_dir.split("://", 1)[1]
        matches = fs.glob(f"{root}/**/*.csv")
        if not matches:
            raise FileNotFoundError(
                f"No CSV found under {results_dir}"
            )
        proto = results_dir.split("://")[0]
        return f"{proto}://{matches[0]}"
    else:
        import glob as _glob
        pattern = os.path.join(results_dir, "**", "*.csv")
        matches = _glob.glob(pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(
                f"No CSV found under {results_dir}"
            )
        return matches[0]


def _read_csv(path):
    """Read a CSV from a local or remote path."""
    with fsspec.open(path, "r") as f:
        return pd.read_csv(f)


def _read_yaml(path):
    """Read a YAML file from a local or remote path."""
    import yaml
    with fsspec.open(path, "r") as f:
        return yaml.safe_load(f)


def _read_json(path):
    """Read a JSON file from a local or remote path."""
    with fsspec.open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    """Write a JSON file to a local or remote path."""
    if not _is_remote(path):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with fsspec.open(path, "w") as f:
        json.dump(data, f, indent=2)


def _join(base, name):
    """Join path components — works for both local and S3."""
    return f"{base.rstrip('/')}/{name}"


def analyze_vcn_inference_gaps(
    inference_results: str,
    threshold_json: str,
    kpi_csv: str,
    output_metrics: str,
    output_gaps: str,
    mode: str = "gap-analysis",
    train_config: str = "",
) -> str:
    """Classify VCN inference results and identify FP/FN gaps.

    Args:
        inference_results: Directory tree containing VCN inference
                           CSV (searched recursively for ``*.csv``).
        threshold_json:    JSON file with ``optimal_threshold`` key.
        kpi_csv:           Ground-truth KPI CSV with ``label`` column.
        output_metrics:    Path for the output metrics JSON.
        output_gaps:       Path for the output gaps parquet.
        mode:              ``"eval-only"`` (metrics only) or
                           ``"gap-analysis"`` (metrics + gap parquet).
        train_config:      VCN train YAML — required for gap-analysis
                           mode to expand filepaths per lighting
                           condition.

    Returns:
        Path to the output metrics JSON.
    """
    # Load the optimal threshold.
    threshold_data = _read_json(threshold_json)
    threshold = float(threshold_data["optimal_threshold"])
    print(f"Using threshold: {threshold}")

    # Find and read the inference CSV.
    csv_path = _find_csv(inference_results)
    print(f"Reading inference CSV: {csv_path}")
    df = _read_csv(csv_path)

    # Read KPI ground truth — merge labels if the inference CSV
    # doesn't already carry them.
    kpi_df = _read_csv(kpi_csv)

    # If inference CSV lacks a label column, merge from KPI CSV.
    if "label" not in df.columns:
        key_cols = [c for c in ("input_path", "object_name") if c in df.columns and c in kpi_df.columns]
        if key_cols:
            df = df.merge(kpi_df[key_cols + ["label"]], on=key_cols, how="left")
        else:
            raise ValueError(
                "Inference CSV has no 'label' column and no common "
                "key columns with KPI CSV for merging."
            )

    # Classify: predicted NO_PASS when siamese_score > threshold.
    predicted_no_pass = df["siamese_score"] > threshold
    actual_no_pass = (
        df["label"].astype(str).str.strip().str.upper() != "PASS"
    )

    # Build confusion matrix.
    tp = int((predicted_no_pass & actual_no_pass).sum())
    fp = int((predicted_no_pass & ~actual_no_pass).sum())
    tn = int((~predicted_no_pass & ~actual_no_pass).sum())
    fn = int((~predicted_no_pass & actual_no_pass).sum())

    total = len(df)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        "threshold": threshold,
        "total_samples": total,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }

    _write_json(output_metrics, metrics)
    print(f"\n=== Metrics (threshold={threshold}) ===")
    print(f"Total samples:  {total}")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1:             {f1:.4f}")
    print(f"Confusion: TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Metrics saved to {output_metrics}")

    if mode == "eval-only":
        return output_metrics

    # --- gap-analysis mode ---
    if not train_config:
        raise ValueError(
            "--train-config is required in gap-analysis mode "
            "(needed for lighting conditions and image extension)."
        )

    # Identify FP/FN samples.
    weak_mask = predicted_no_pass != actual_no_pass
    weak_df = df.loc[weak_mask]

    # Read lighting conditions and image extension from train config.
    config_data = _read_yaml(train_config)
    lightings = config_data["dataset"]["classify"]["input_map"]
    ext = config_data["dataset"]["classify"]["image_ext"]

    # Expand each weak sample into one filepath per lighting,
    # carrying the ground-truth label for downstream mining.
    gap_records = []
    for _, row in weak_df.iterrows():
        input_path = str(row["input_path"])
        obj = str(row["object_name"])
        label = str(row["label"]).strip()
        for lighting in lightings:
            gap_records.append({
                "filepath": os.path.join(input_path, f"{obj}_{lighting}{ext}"),
                "label": label,
            })

    gaps_df = pd.DataFrame(gap_records)

    if not _is_remote(output_gaps):
        gaps_dir = os.path.dirname(output_gaps)
        if gaps_dir:
            os.makedirs(gaps_dir, exist_ok=True)

    gaps_df.to_parquet(output_gaps, index=False)

    # Print breakdown by label.
    label_counts = (
        weak_df["label"]
        .astype(str).str.strip().str.upper()
        .value_counts()
    )
    print(f"\n=== Gap Analysis ===")
    print(f"Total weak (FP/FN) samples: {len(weak_df)}")
    print(f"Gap filepaths emitted:      {len(gaps_df)}")
    for label, count in label_counts.items():
        pct = 100.0 * count / len(weak_df) if len(weak_df) else 0
        print(f"  {label}: {count} ({pct:.1f}%)")
    print(f"Gaps parquet saved to {output_gaps}")

    return output_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Classify VCN inference results and identify FP/FN gaps"
    )
    parser.add_argument(
        "--inference-results", required=True,
        help="Directory containing VCN inference CSV (searched recursively)",
    )
    parser.add_argument(
        "--threshold-json", required=True,
        help="JSON file with optimal_threshold key",
    )
    parser.add_argument(
        "--kpi-csv", required=True,
        help="Ground-truth KPI CSV with label column",
    )
    parser.add_argument(
        "--output-metrics", required=True,
        help="Output path for metrics JSON",
    )
    parser.add_argument(
        "--output-gaps", required=True,
        help="Output path for gaps parquet (gap-analysis mode)",
    )
    parser.add_argument(
        "--mode", default="gap-analysis",
        choices=["eval-only", "gap-analysis"],
        help="eval-only (metrics only) or gap-analysis (metrics + gaps parquet)",
    )
    parser.add_argument(
        "--train-config", default="",
        help="VCN train config YAML (required for gap-analysis mode)",
    )
    args = parser.parse_args()

    analyze_vcn_inference_gaps(
        inference_results=args.inference_results,
        threshold_json=args.threshold_json,
        kpi_csv=args.kpi_csv,
        output_metrics=args.output_metrics,
        output_gaps=args.output_gaps,
        mode=args.mode,
        train_config=args.train_config,
    )


if __name__ == "__main__":
    main()
