#!/usr/bin/env python3
"""Find the optimal classification threshold from VCN inference results.

Sweeps all unique ``siamese_score`` values in the inference CSV to
find the threshold that maximises NO_PASS-class F1 while maintaining
at least ``min_recall`` recall. Ties are broken by precision, then
threshold value.

The result is persisted as a JSON file so downstream scripts
(gap analysis, evaluation) can consume it.

Supports both local paths and S3 URIs (s3://) via fsspec.
"""
import argparse
import json
import math
import os
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _Row:
    is_pass: bool
    score: float


@dataclass(frozen=True)
class _Metrics:
    threshold: float
    precision: float
    recall: float
    f1: float


def compute_vcn_optimal_threshold(
    inference_results: str,
    output_json: str,
    min_recall: float = 1.0,
) -> dict:
    """Sweep thresholds and find the best one for NO_PASS classification.

    A sample is predicted NO_PASS when ``siamese_score > threshold``.

    Args:
        inference_results: Directory tree containing VCN inference
                           CSV (searched recursively for ``*.csv``).
        output_json:       Output path for the threshold JSON.
        min_recall:        Minimum NO_PASS recall a candidate threshold
                           must achieve (0.0 -- 1.0, default 1.0).

    Returns:
        Dict with optimal_threshold, f1, precision, recall.

    Raises:
        FileNotFoundError: If no CSV is found under inference_results.
        ValueError: If no threshold achieves the required recall.
    """
    csv_path = _find_csv(inference_results)
    print(f"Reading inference CSV: {csv_path}")
    df = _read_csv(csv_path)

    rows = [
        _Row(
            is_pass=str(r["label"]).strip().upper() == "PASS",
            score=float(r["siamese_score"]),
        )
        for _, r in df.iterrows()
    ]
    print(f"Loaded {len(rows)} samples")

    # Sweep every unique score as a candidate threshold, plus one
    # value below the minimum so we also test "predict all NO_PASS".
    unique_scores = sorted({r.score for r in rows})
    first = math.nextafter(unique_scores[0], float("-inf"))
    thresholds = [first, *unique_scores]
    candidates = []

    for thr in thresholds:
        tp = fp = tn = fn = 0
        for r in rows:
            no_pass_actual = not r.is_pass
            no_pass_pred = r.score > thr
            if no_pass_actual and no_pass_pred:
                tp += 1
            elif not no_pass_actual and no_pass_pred:
                fp += 1
            elif not no_pass_actual and not no_pass_pred:
                tn += 1
            else:
                fn += 1

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        denom = prec + rec
        f1 = (2.0 * prec * rec) / denom if denom else 0.0

        # Only keep thresholds meeting the minimum recall constraint.
        if (tp + fn) > 0 and rec >= min_recall - 1e-12:
            candidates.append(_Metrics(
                threshold=thr,
                precision=prec,
                recall=rec,
                f1=f1,
            ))

    if not candidates:
        raise ValueError(
            f"No threshold achieves {min_recall:.0%} recall "
            "on the NO_PASS class."
        )

    # Pick the threshold with the best F1, breaking ties by
    # precision then threshold value.
    best = max(
        candidates,
        key=lambda m: (m.f1, m.precision, m.threshold),
    )

    result = {
        "optimal_threshold": best.threshold,
        "f1": round(best.f1, 6),
        "precision": round(best.precision, 6),
        "recall": round(best.recall, 6),
    }

    if not _is_remote(output_json):
        parent = os.path.dirname(output_json)
        if parent:
            os.makedirs(parent, exist_ok=True)

    with fsspec.open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Optimal Threshold ===")
    print(f"Threshold:  {best.threshold}")
    print(f"F1:         {best.f1:.4f}")
    print(f"Precision:  {best.precision:.4f}")
    print(f"Recall:     {best.recall:.4f}")
    print(f"Candidates evaluated: {len(candidates)} / {len(thresholds)}")
    print(f"Saved to {output_json}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal VCN classification threshold from inference results"
    )
    parser.add_argument(
        "--inference-results", required=True,
        help="Directory containing VCN inference CSV (searched recursively)",
    )
    parser.add_argument(
        "--output-json", required=True,
        help="Output path for the threshold JSON",
    )
    parser.add_argument(
        "--min-recall", type=float, default=1.0,
        help="Minimum NO_PASS recall required (default: 1.0)",
    )
    args = parser.parse_args()

    compute_vcn_optimal_threshold(
        inference_results=args.inference_results,
        output_json=args.output_json,
        min_recall=args.min_recall,
    )


if __name__ == "__main__":
    main()
