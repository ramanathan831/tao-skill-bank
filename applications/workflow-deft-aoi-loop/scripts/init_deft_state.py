# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initialize ${RESULTS_DIR}/deft_state.json with a guaranteed-unique key set.

Why this exists: earlier inline-dict writes drifted from the canonical schema
in `references/deft_state.json` and produced duplicate top-level keys (`kpi_target`, `results_dir`, `max_iterations`, `current_iteration`) — Python
3.12+ now emits a `SyntaxWarning` for these and the loop's resume logic reads
whichever copy parsing keeps, which is not stable across edits.

This script builds the dict with literal-once keys and writes the JSON. Atomic
write (tmp + os.replace). Refuses to overwrite an existing file unless `--force`
is passed — the resume path is supposed to read disk, not regenerate.

CLI:

    python scripts/init_deft_state.py \
        --results-dir /home/shadeform/workspace/results/run_20260514_143000 \
        --workspace /home/shadeform/workspace \
        --kpi-target "FAR < 10% at recall=100%" \
        --max-iterations 2 \
        --num-gpus 4 \
        --num-epochs 20 \
        --num-sdg 20 \
        --project nvpcb \
        --step 14000

The output schema mirrors `references/deft_state.json` exactly.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import tempfile


_COMPLETED_STEP_VALUES = [
    "evaluate",
    "rca",
    "anomalygen_finetune",
    "anomalygen",
    "routing",
    "data_mining",
    "train",
    "loop_stop",
]
_STATUS_VALUES = ["pending", "in_progress", "complete", "failed"]
_DEFAULT_TRAIN_CONTAINER = "nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt"
_DEFAULT_AG_CONTAINER = (
    "nvcr.io/nv-metropolis-dev/metropolis-sdg/cosmos-anomalygen:1.0.3-36cdfca9.main"
)


def build_state(args: argparse.Namespace) -> dict:
    ws = args.workspace.resolve()
    rd = args.results_dir.resolve()

    state = {
        "version": 2,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "kpi_target": args.kpi_target,
        "results_dir": str(rd),
        "max_iterations": args.max_iterations,
        "current_iteration": 0,
        "config": {
            "specs_file": str(ws / "specs" / "baseline_spec.yaml"),
            "training_csv": str(ws / "train" / "base" / "training_set.csv"),
            "validation_csv": str(ws / "train" / "base" / "validation_set.csv"),
            "kpi_test_csv": str(ws / "kpi" / "testing_set.csv"),
            "images_dir": str(ws / "kpi" / "images"),
            "backbone_weight_dir": str(ws / "augmentation" / "backbone"),
            "train_container": args.train_container,
            "num_gpus": args.num_gpus,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "anomalygen": {
                "sub_skill": "cosmos-anomalygen",
                "mode": "inference_only",
                "project": args.project,
                # defect_spec lives under `datasets/<project>/` (sibling of
                # `checkpoints/<project>/`), per references/cosmos-anomalygen.md.
                "defect_spec": str(
                    ws
                    / "augmentation"
                    / "anomalygen"
                    / "datasets"
                    / args.project
                    / "defect_spec.jsonl"
                ),
                # ag_checkpoint_dir: the directory holding ag_config.yaml +
                # checkpoints/{latest_checkpoint.txt, model/iter_<step>.pt, ...}.
                # The underlying skill takes this as `ag_checkpoint_dir`.
                "checkpoint_dir": str(
                    ws
                    / "augmentation"
                    / "anomalygen"
                    / "checkpoints"
                    / args.project
                ),
                # dataset_dir: parent-staged pool root; not the raw datasets/.
                # Resolved per-iteration to `${RESULTS_DIR}/iter${N}/pool_anomalygen/inputs/`.
                "dataset_dir_source": str(
                    ws / "augmentation" / "anomalygen" / "datasets" / args.project
                ),
                "step": args.step,
                "num_SDG": args.num_sdg,
                "container": args.ag_container,
            },
            "mining_filter": {
                "sub_skill": "deft-aoi-mining",
                "top_k_per_target": args.top_k_per_target,
                "metric": args.knn_metric,
                "min_similarity": args.min_similarity,
            },
        },
        "iterations": {},
        "_completed_step_values": list(_COMPLETED_STEP_VALUES),
        "_status_values": list(_STATUS_VALUES),
    }
    return state


def write_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize deft_state.json with a guaranteed-unique key set. "
            "Refuses to overwrite an existing file unless --force."
        ),
    )
    parser.add_argument("--results-dir", required=True, type=pathlib.Path)
    parser.add_argument("--workspace", required=True, type=pathlib.Path)
    parser.add_argument(
        "--kpi-target",
        required=True,
        help='e.g. "FAR < 10% at recall=100%%"',
    )
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--num-gpus", required=True, type=int)
    parser.add_argument("--num-epochs", required=True, type=int)
    parser.add_argument("--num-sdg", required=True, type=int)
    parser.add_argument("--project", required=True, help="AnomalyGen project name (e.g. nvpcb)")
    parser.add_argument("--step", required=True, type=int, help="AnomalyGen checkpoint step")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--top-k-per-target", default=5, type=int)
    parser.add_argument(
        "--knn-metric",
        default="cosine",
        choices=("cosine", "euclidean", "manhattan"),
    )
    parser.add_argument(
        "--min-similarity",
        default=None,
        type=float,
        help="Cosine similarity threshold for mining (e.g. 0.9). Omit for none.",
    )
    parser.add_argument("--train-container", default=_DEFAULT_TRAIN_CONTAINER)
    parser.add_argument("--ag-container", default=_DEFAULT_AG_CONTAINER)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing deft_state.json. Off by default to protect resume state.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = args.results_dir / "deft_state.json"
    if out.exists() and not args.force:
        print(
            f"init_deft_state: refusing to overwrite {out} (use --force).",
            file=sys.stderr,
        )
        return 2
    state = build_state(args)
    write_atomic(out, state)
    print(f"init_deft_state: wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
