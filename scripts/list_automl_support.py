#!/usr/bin/env python3
"""Print packaged TAO AutoML model support without scanning model folders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=Path(os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")),
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def load_support(skill_bank: Path) -> dict[str, Any]:
    """Load the packaged AutoML support summary."""
    support_path = skill_bank.expanduser() / "models" / "automl_support.json"
    with support_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_text(support: dict[str, Any]) -> str:
    """Format support data for quick human-readable answers."""
    supported = [item["model"] for item in support.get("supported", [])]
    unsupported = support.get("unsupported", [])

    lines = ["Supported AutoML models:"]
    lines.extend(f"- {model}" for model in supported)
    lines.append("")
    lines.append("Not supported:")
    if unsupported:
        lines.extend(f"- {item['model']}: {item['reason']}" for item in unsupported)
    else:
        lines.append("- None")
    return "\n".join(lines)


def main() -> int:
    """Run the support summary helper."""
    args = parse_args()
    support = load_support(args.skill_bank)
    if args.format == "json":
        print(json.dumps(support, indent=2, sort_keys=True))
    else:
        print(format_text(support))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
