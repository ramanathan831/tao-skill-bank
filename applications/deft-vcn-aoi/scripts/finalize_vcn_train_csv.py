#!/usr/bin/env python3
"""Produce the merged training CSV for a VCN SDA iteration.

Maps mined filepaths back to source pool training rows, merges
them onto the base (previous) training set, and deduplicates by
training identity ``(input_path, object_name, label)``.

Base training resolution:
  - If ``--prev-train-csv`` is provided and exists, load it as
    the base training set (supports continual learning where each
    iteration builds on the previous merged CSV).
  - Otherwise, start from an empty frame so the first SDA
    iteration trains on mined rows only.

Supports both local paths and S3 URIs (s3://) via fsspec.
"""
import argparse
import os

import fsspec
import pandas as pd


def _is_remote(path):
    return "://" in path


def _path_exists(path):
    """Check if a path exists (local or remote)."""
    if _is_remote(path):
        fs, _ = fsspec.core.url_to_fs(path)
        remote_path = path.split("://", 1)[1]
        return fs.exists(remote_path)
    else:
        return os.path.exists(path)


def _read_csv(path):
    """Read a CSV from a local or remote path."""
    with fsspec.open(path, "r") as f:
        return pd.read_csv(f)


def finalize_vcn_train_csv(
    mined_parquet: str,
    source_pool_parquet: str,
    output_csv: str,
    prev_train_csv: str = "",
) -> str:
    """Merge mined rows into the training set.

    Args:
        mined_parquet:       Mining output parquet (filepath column).
        source_pool_parquet: Source pool parquet from
                             ``build_vcn_source_pool`` (filepath plus
                             all original CSV columns).
        output_csv:          Where to write the merged training CSV.
        prev_train_csv:      Previous iteration's training CSV (or
                             empty for first iteration).

    Returns:
        Path to the output CSV.
    """
    summary_lines = []

    # --- Resolve base training set ---
    if prev_train_csv and _path_exists(prev_train_csv):
        base_df = _read_csv(prev_train_csv)
        summary_lines.append(
            f"Base training CSV: {prev_train_csv} ({len(base_df)} rows)"
        )
    else:
        base_df = pd.DataFrame()
        if prev_train_csv:
            summary_lines.append(
                f"Previous CSV not found ({prev_train_csv}); "
                "starting from empty base."
            )
        else:
            summary_lines.append(
                "No previous training CSV provided; merged CSV will "
                "contain mined rows only."
            )

    # --- Map mined filepaths back to source pool training rows ---
    idx = pd.read_parquet(source_pool_parquet)
    mined = pd.read_parquet(mined_parquet)

    print(f"Mined parquet:  {len(mined)} filepaths")
    print(f"Source pool:    {len(idx)} rows")

    idx_deduped = idx.drop_duplicates(subset=["filepath"], keep="first")
    mined_rows = (
        mined[["filepath"]]
        .drop_duplicates()
        .merge(idx_deduped, on="filepath", how="inner")
        .drop(columns=["filepath"])
    )

    unmatched = len(mined["filepath"].drop_duplicates()) - len(mined_rows)
    if unmatched:
        warn = f"Warning: {unmatched} mined filepaths not found in source index"
        summary_lines.append(warn)
        print(warn)

    # Dedup mined rows by training identity.
    key_cols = ["input_path", "object_name", "label"]
    mined_rows = mined_rows.drop_duplicates(subset=key_cols)

    # Align columns to match base CSV schema.
    if len(mined_rows) > 0 and len(base_df.columns) > 0:
        mined_rows = mined_rows.reindex(columns=base_df.columns)

    # --- Merge and deduplicate ---
    merged = pd.concat([base_df, mined_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_cols, keep="first")

    # --- Write output ---
    if not _is_remote(output_csv):
        parent = os.path.dirname(output_csv)
        if parent:
            os.makedirs(parent, exist_ok=True)

    with fsspec.open(output_csv, "w") as f:
        merged.to_csv(f, index=False)

    # --- Report statistics ---
    new_unique_count = len(merged) - len(base_df)
    summary_lines.append("")
    summary_lines.append(
        f"Merged training CSV: base={len(base_df)} + "
        f"new_unique={new_unique_count} -> {len(merged)} rows"
    )
    summary_lines.append(f"Written to: {output_csv}")

    if "label" in merged.columns:
        normalize = lambda s: s.astype(str).str.strip().str.upper()
        merged_labels = normalize(merged["label"]).value_counts()
        base_labels = (
            normalize(base_df["label"]).value_counts()
            if len(base_df) and "label" in base_df.columns
            else pd.Series(dtype=int)
        )
        new_labels = merged_labels.subtract(base_labels, fill_value=0).astype(int)

        summary_lines.append("")
        summary_lines.append("New unique samples by label:")
        for label, count in sorted(new_labels.items()):
            if count > 0:
                pct = 100.0 * count / new_unique_count if new_unique_count else 0
                summary_lines.append(f"  {label}: {count} ({pct:.1f}%)")

        summary_lines.append("")
        summary_lines.append(
            f"Overall merged pool by label ({len(merged)} total):"
        )
        for label, count in sorted(merged_labels.items()):
            pct = 100.0 * count / len(merged)
            summary_lines.append(f"  {label}: {count} ({pct:.1f}%)")

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    # Write summary alongside the output CSV.
    if not _is_remote(output_csv):
        summary_path = os.path.join(
            os.path.dirname(output_csv) or ".",
            "finalize_train_csv_summary.txt",
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
        print(f"\nSummary written to: {summary_path}")

    return output_csv


def main():
    parser = argparse.ArgumentParser(
        description="Merge mined samples into VCN training CSV"
    )
    parser.add_argument(
        "--mined-parquet", required=True,
        help="Mining output parquet (filepath column)",
    )
    parser.add_argument(
        "--source-pool-parquet", required=True,
        help="Source pool parquet from build_vcn_source_pool",
    )
    parser.add_argument(
        "--prev-train-csv", default="",
        help="Previous iteration training CSV (optional, empty for first iteration)",
    )
    parser.add_argument(
        "--output-csv", required=True,
        help="Output path for the merged training CSV",
    )
    args = parser.parse_args()

    finalize_vcn_train_csv(
        mined_parquet=args.mined_parquet,
        source_pool_parquet=args.source_pool_parquet,
        output_csv=args.output_csv,
        prev_train_csv=args.prev_train_csv,
    )


if __name__ == "__main__":
    main()
