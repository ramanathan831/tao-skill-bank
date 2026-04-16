#!/usr/bin/env python3
"""Build target queries and source pool parquets for NIM embedding mining.

Reads the gap parquet (from gap analysis) and the source CSV (VCN
training format with input_path, object_name, label columns). Expands
each row into one filepath per lighting condition from the train config
YAML, producing two parquets:

  - **target_queries**: filepaths of gap images (already expanded
    per lighting in the gaps parquet).
  - **source_pool**: filepaths of all source images with their
    original CSV columns, so mined filepaths can be mapped back
    to training rows in ``finalize_vcn_train_csv.py``.

Supports both local paths and S3 URIs (s3://) via fsspec.
"""
import argparse
import os

import fsspec
import pandas as pd


def _is_remote(path):
    return "://" in path


def _read_csv(path):
    """Read a CSV from a local or remote path."""
    with fsspec.open(path, "r") as f:
        return pd.read_csv(f)


def _read_yaml(path):
    """Read a YAML file from a local or remote path."""
    import yaml
    with fsspec.open(path, "r") as f:
        return yaml.safe_load(f)


def build_vcn_source_pool(
    gaps_csv: str,
    source_csv: str,
    source_images_dir: str,
    train_config: str,
    output_target_parquet: str,
    output_source_parquet: str,
) -> tuple:
    """Build target queries and source pool parquets.

    Args:
        gaps_csv:               Path to gap parquet/CSV from gap
                                analysis (filepath, label columns).
        source_csv:             VCN training CSV (input_path,
                                object_name, label columns).
        source_images_dir:      Root directory for source image paths
                                (same as train_media_path).
        train_config:           VCN train YAML for input_map and
                                image_ext.
        output_target_parquet:  Output path for target queries parquet.
        output_source_parquet:  Output path for source pool parquet.

    Returns:
        Tuple of (target_parquet_path, source_parquet_path).
    """
    # --- Read gap data (target queries) ---
    # The gaps file may be parquet or CSV depending on upstream.
    if gaps_csv.endswith(".parquet"):
        gaps_df = pd.read_parquet(gaps_csv)
    else:
        gaps_df = _read_csv(gaps_csv)

    if "filepath" not in gaps_df.columns:
        raise ValueError(
            f"Gaps file missing 'filepath' column; "
            f"found {list(gaps_df.columns)}"
        )

    # Target queries: just the filepath column from gap analysis
    # (already expanded per lighting condition).
    target_df = gaps_df[["filepath"]].drop_duplicates()

    # --- Read source CSV and expand per lighting ---
    src_df = _read_csv(source_csv)
    required = ("input_path", "object_name", "label")
    missing = [c for c in required if c not in src_df.columns]
    if missing:
        raise ValueError(
            f"Source CSV missing columns {missing}; "
            f"found {list(src_df.columns)}"
        )

    # Read lighting conditions and image extension from train config.
    config_data = _read_yaml(train_config)
    lightings = config_data["dataset"]["classify"]["input_map"]
    ext = config_data["dataset"]["classify"]["image_ext"]

    # Expand each source CSV row into one filepath per lighting,
    # keeping all original columns so mined filepaths can be mapped
    # back to training rows in finalize_vcn_train_csv.
    records = []
    for _, row in src_df.iterrows():
        base = os.path.join(source_images_dir, str(row["input_path"]))
        obj = str(row["object_name"])
        row_dict = row.to_dict()
        for lighting in lightings:
            rec = {"filepath": os.path.join(base, f"{obj}_{lighting}{ext}")}
            rec.update(row_dict)
            records.append(rec)

    source_pool_df = pd.DataFrame(records)

    # --- Write outputs ---
    for path in (output_target_parquet, output_source_parquet):
        if not _is_remote(path):
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    target_df.to_parquet(output_target_parquet, index=False)
    source_pool_df.to_parquet(output_source_parquet, index=False)

    print(f"\n=== Target Queries ===")
    print(f"Gap samples:    {len(gaps_df)}")
    print(f"Unique targets: {len(target_df)}")
    print(f"Saved to {output_target_parquet}")

    print(f"\n=== Source Pool ===")
    print(f"Source CSV rows:     {len(src_df)}")
    print(f"Expanded filepaths:  {len(source_pool_df)}")
    print(f"Lighting conditions: {len(lightings)} ({', '.join(lightings)})")
    print(f"Saved to {output_source_parquet}")

    return output_target_parquet, output_source_parquet


def main():
    parser = argparse.ArgumentParser(
        description="Build target queries and source pool parquets for mining"
    )
    parser.add_argument(
        "--gaps-csv", required=True,
        help="Gap parquet/CSV from gap analysis (filepath, label columns)",
    )
    parser.add_argument(
        "--source-csv", required=True,
        help="VCN training CSV (input_path, object_name, label columns)",
    )
    parser.add_argument(
        "--source-images-dir", required=True,
        help="Root directory for source image paths",
    )
    parser.add_argument(
        "--train-config", required=True,
        help="VCN train config YAML (for input_map and image_ext)",
    )
    parser.add_argument(
        "--output-target-parquet", required=True,
        help="Output path for target queries parquet",
    )
    parser.add_argument(
        "--output-source-parquet", required=True,
        help="Output path for source pool parquet",
    )
    args = parser.parse_args()

    build_vcn_source_pool(
        gaps_csv=args.gaps_csv,
        source_csv=args.source_csv,
        source_images_dir=args.source_images_dir,
        train_config=args.train_config,
        output_target_parquet=args.output_target_parquet,
        output_source_parquet=args.output_source_parquet,
    )


if __name__ == "__main__":
    main()
