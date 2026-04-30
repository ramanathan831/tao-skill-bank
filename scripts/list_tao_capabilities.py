#!/usr/bin/env python3
"""Summarize TAO Skill Bank capabilities for plugin capability answers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from list_tao_models import build_all_models, build_automl_support
from list_tao_platforms import prompt_defaults, supported_platforms


DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")
)
FLOW_ACTIONS = ("train", "evaluate", "inference", "export", "gen_trt_engine")


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
        "--format",
        choices=("json", "text"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def parse_frontmatter(path: Path) -> dict[str, str]:
    """Parse the simple YAML frontmatter shape used by bundled skills."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    frontmatter: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        frontmatter.append(line)

    data: dict[str, str] = {}
    index = 0
    while index < len(frontmatter):
        line = frontmatter[index]
        if not line.strip() or line.startswith(" "):
            index += 1
            continue
        if ":" not in line:
            index += 1
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in {">", ">-", "|", "|-"}:
            block: list[str] = []
            index += 1
            while index < len(frontmatter):
                next_line = frontmatter[index]
                if next_line and not next_line.startswith(" "):
                    break
                if next_line.strip():
                    block.append(next_line.strip())
                index += 1
            data[key] = " ".join(block)
            continue

        data[key] = value.strip("\"'")
        index += 1

    return data


def application_capabilities(skill_bank: Path) -> list[dict[str, Any]]:
    """Read top-level application skills and turn them into capability records."""
    application_root = skill_bank.expanduser() / "applications"
    records: list[dict[str, Any]] = []

    for skill_md in sorted(application_root.glob("*/SKILL.md")):
        metadata = parse_frontmatter(skill_md)
        app_dir = skill_md.parent
        files = sorted(item.name for item in app_dir.iterdir() if item.is_file())
        records.append(
            {
                "name": metadata.get("name", app_dir.name),
                "path": str(skill_md.relative_to(skill_bank.expanduser())),
                "files": files,
                "capability": metadata.get(
                    "description",
                    f"Use the {app_dir.name} application workflow.",
                ),
            }
        )
    return records


def build_capabilities(skill_bank: Path) -> dict[str, Any]:
    """Build a capability summary from application skills and model manifests."""
    all_models = build_all_models(skill_bank)
    automl = build_automl_support(skill_bank)
    model_records = all_models["models"]

    training_models = [
        item["model"] for item in model_records if "train" in item.get("actions", [])
    ]
    full_finetune_models = [
        item["model"]
        for item in model_records
        if all(action in item.get("actions", []) for action in FLOW_ACTIONS)
    ]

    return {
        "schema_version": 1,
        "applications": application_capabilities(skill_bank),
        "platforms": {
            "source": "platform/platforms.manifest.json",
            "prompt_defaults": prompt_defaults(skill_bank),
            "supported": supported_platforms(skill_bank),
        },
        "model_workflows": {
            "source": "models/schemas.manifest.json",
            "actions": list(FLOW_ACTIONS),
            "training_capable_models": training_models,
            "full_train_eval_infer_export_trt_models": full_finetune_models,
            "models": model_records,
        },
        "automl": automl,
    }


def csv(items: list[str]) -> str:
    """Format compact comma-separated text."""
    return ", ".join(items) if items else "none"


def format_capabilities_text(data: dict[str, Any]) -> str:
    """Format the capability summary for a plugin capability answer."""
    lines = ["TAO Skill Bank capabilities", "", "Application workflows:"]
    for app in data["applications"]:
        lines.append(f"- {app['name']}: {app['capability']}")

    platforms = data["platforms"]
    defaults = platforms["prompt_defaults"]
    lines.extend(
        [
            "",
            "Workflow launch support:",
            "- Supported platforms are generated from platform/platforms.manifest.json through scripts/list_tao_platforms.py.",
            "- Platforms: "
            + csv([item["name"] for item in platforms["supported"]]),
            "- Long-running monitoring prompt default: "
            + f"{str(defaults['long_running_enabled']).lower()}, "
            + f"status every {defaults['status_interval_minutes']} minutes.",
        ]
    )

    model_workflows = data["model_workflows"]
    lines.extend(
        [
            "",
            "Model fine-tuning and deployment workflows:",
            "- Packaged model schemas declare train, evaluate, inference, export, and TensorRT engine generation actions per model.",
            "- Full train/evaluate/inference/export/TensorRT flow: "
            + csv(model_workflows["full_train_eval_infer_export_trt_models"]),
            "- Training-capable models: "
            + csv(model_workflows["training_capable_models"]),
            "",
            "Per-model action support:",
        ]
    )
    for model in model_workflows["models"]:
        actions = csv(model["actions"])
        lines.append(f"- {model['model']}: {actions}")

    automl = data["automl"]
    lines.extend(
        [
            "",
            "AutoML support:",
            f"- {automl['support_rule']}",
            "- Supported AutoML models: "
            + csv([item["model"] for item in automl["supported"]]),
        ]
    )
    if automl["unsupported"]:
        lines.append(
            "- Not supported for AutoML: "
            + "; ".join(
                f"{item['model']} ({item['reason']})" for item in automl["unsupported"]
            )
        )
    else:
        lines.append("- Not supported for AutoML: none")

    return "\n".join(lines)


def main() -> int:
    """Run the TAO capability summary helper."""
    args = parse_args()
    data = build_capabilities(args.skill_bank.expanduser())

    if args.format == "json":
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(format_capabilities_text(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
