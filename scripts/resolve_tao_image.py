#!/usr/bin/env python3
"""Resolve the default TAO container image for a model action.

The helper reads packaged model metadata instead of relying on hand-written
prompts. Launch workflows should show this image to the user and accept an
explicit override before generating runner artifacts or submitting jobs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument(
        "--model",
        "--network",
        dest="model",
        required=True,
        help="Packaged model/network name, for example cosmos-rl.",
    )
    parser.add_argument(
        "--action",
        default="train",
        help="Model action to resolve, for example train, evaluate, inference, or export.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def resolve_image(skill_bank: Path, model: str, action: str) -> dict[str, Any]:
    """Resolve action-level image first, then model-level image."""
    config_path = skill_bank.expanduser() / "models" / model / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    config = load_json(config_path)
    actions = config.get("actions", {})
    if not isinstance(actions, dict):
        actions = {}

    action_config = actions.get(action)
    if action_config is None:
        available = ", ".join(sorted(actions)) if actions else "none"
        raise ValueError(
            f"Action '{action}' is not packaged for model '{model}'. "
            f"Available actions: {available}"
        )
    if not isinstance(action_config, dict):
        raise ValueError(f"models/{model}/config.json actions.{action} must be an object")

    candidates = [
        ("action.container_image", action_config.get("container_image")),
        ("action.image", action_config.get("image")),
        ("model.container_image", config.get("container_image")),
        ("model.image", config.get("image")),
    ]
    for source, image in candidates:
        if isinstance(image, str) and image.strip():
            return {
                "schema_version": 1,
                "model": model,
                "network_arch": config.get("network_arch", model),
                "action": action,
                "image": image.strip(),
                "source": source,
                "config_path": str(config_path),
                "confirmation_required": True,
                "override_key": "image",
            }

    raise ValueError(
        f"No container image found for model '{model}' action '{action}' in {config_path}"
    )


def format_text(data: dict[str, Any]) -> str:
    """Format resolved image metadata for launch prompts."""
    return "\n".join(
        [
            "TAO container image resolution:",
            f"- model: {data['model']} ({data['network_arch']})",
            f"- action: {data['action']}",
            f"- default image: {data['image']}",
            f"- source: {data['source']} in {data['config_path']}",
            "- confirmation: ask the user to use this image or provide image=<override> before launch",
        ]
    )


def main() -> int:
    """Run the image resolver."""
    args = parse_args()
    data = resolve_image(args.skill_bank, args.model, args.action)
    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(format_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
