---
name: tao-automl
description: >-
  Run hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner. Handles algorithm
  selection (bayesian, hyperband, asha, bohb, llm, hybrid, autoresearch), WandB experiment tracking,
  job execution on any TAO SDK platform, result interpretation, and per-rec custom evaluation hooks.
  Use when the user mentions TAO AutoML, hyperparameter optimization, HPO, automl, automl_settings,
  AutoMLRunner, tao_automl, bayesian search, hyperband, ASHA, LLM-guided search, autoresearch, or
  wants to tune training hyperparameters for any TAO network. Platform-agnostic — runs on any SDK
  (Lepton, Brev, SLURM, Kubernetes, Docker).
license: Apache-2.0
metadata:
  author: "NVIDIA Corporation"
  version: "0.1"
  compatibility: >-
    Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
  tags:
  - "automl"
  - "hpo"
  - "workflow"
  - "training"
  - "optimization"
  - "llm"
allowed-tools: Read Bash Write
---
# TAO AutoML Skill

Run automated hyperparameter optimization (HPO) for TAO training through
`AutoMLRunner`. Keep this file as the launch router and load references only
for the branch the user actually needs.

## Authority Order

1. User request and explicit run overrides.
2. Shared launch intake in `skills/tao-workflow-launch/SKILL.md`.
3. Selected model skill at `models/<network>/SKILL.md`.
4. Generated model schema at `models/<network>/schemas/train.schema.json`.
5. AutoML references in this directory.

Do not encode model-specific hyperparameter names, ranges, metrics, dataset
layouts, archive names, class counts, spec keys, container images, checkpoint
quirks, or metric regexes in this workflow skill. Read them from the selected
model skill and generated schema.

## Reference Map

| Need | Read |
|---|---|
| Parse launch intent, schema gate, required fields | `references/intent-and-schema.md` |
| Choose bayesian / ASHA / hyperband / LLM / autoresearch | `references/algorithms.md` |
| Build `AutoMLRunner` calls and settings | `references/runner-api.md` |
| Add WandB tracking | `references/wandb.md` |
| Configure LLM, hybrid, or autoresearch modes | `references/llm-agentic.md` |
| Add custom metrics, monitor, resume, and report results | `references/hooks-monitoring-results.md` |
| Diagnose pitfalls, query status, or use conversation examples | `references/status-pitfalls-examples.md` |

## Preflight

This skill needs `nvidia-tao-automl`, which pulls `nvidia-tao-sdk` as a
transitive dependency. Neither package is published to public PyPI yet; install
from the NVIDIA GitLab repo with the platform extra the user chose.

```bash
python -c "import tao_automl" 2>/dev/null || {
  REPO='git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-automl.git'
  echo "MISSING: nvidia-tao-automl not installed. Pick the platform extra you need:"
  echo "  pip install \"nvidia-tao-automl[lepton] @ $REPO\"      # DGX Cloud / Lepton"
  echo "  pip install \"nvidia-tao-automl[slurm] @ $REPO\"       # on-prem SLURM cluster"
  echo "  pip install \"nvidia-tao-automl[kubernetes] @ $REPO\"  # K8s"
  echo "  pip install \"nvidia-tao-automl[docker] @ $REPO\"      # local Docker daemon"
  echo "  pip install \"nvidia-tao-automl[brev] @ $REPO\"        # Brev GPU instances"
  echo "  pip install \"nvidia-tao-automl[all] @ $REPO\"         # all platforms"
  exit 1
}
```

If missing, ask before installing, install only after approval, then rerun the
preflight. Do not create runner files, workspaces, logs, compatibility shims, or
state files before the shared launch preflight passes.

## Quick Support Queries

When the user asks which models support AutoML, run the packaged helper instead
of answering from memory:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_models.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --scope automl --format text
```

The compatibility wrapper is also valid:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_automl_support.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

Return both runnable AutoML models and models blocked on schema packaging.
Runnable AutoML requires model-level `automl_enabled: true` plus a valid packaged
`models/<network>/schemas/train.schema.json`.

## Launch Checklist

Before generating or launching anything:

1. Run the shared launch intake from `tao-workflow-launch`.
2. Ask the user to choose a supported platform from `scripts/list_tao_platforms.py`.
3. Filter credentials with `scripts/list_tao_platforms.py --platform <platform>`.
4. Resolve the train image with `scripts/resolve_tao_image.py --model <network> --action train` and require confirmation or `image=<override>`.
5. Read `models/<network>/SKILL.md` for Training Requirements, Per-Action Dataset Requirements, Typical Spec Overrides, AutoML / HPO Notes, and Error Patterns.
6. Read `models/<network>/schemas/train.schema.json` and `schemas/manifest.json`.
7. Verify platform access and dataset visibility from the selected platform's point of view.
8. Verify model-specific required annotation fields documented by the model skill.
9. Decide whether this is quick-start AutoML or a customized search.

If any required launch field is missing, ask for that field and stop before
creating artifacts.

## Quick-Start Defaults

Use these defaults for a plain "run AutoML" request after all required launch
inputs are known:

| Field | Default |
|---|---|
| `algorithm` | `bayesian`, unless the model/workflow profile declares another algorithm |
| `automl_max_recommendations` | model/workflow default if declared, otherwise `10` |
| `automl_hyperparameters` | `None`, so AutoML uses schema params marked `automl_enabled=true` |
| `custom_param_ranges` | `None`, so ranges/options/defaults come from the generated schema |
| `long_running_enabled` | `true` |
| `status_interval_minutes` | `5` |

Offer customization only after required quick-start fields are resolved. If the
user customizes algorithms, budgets, hyperparameters, ranges, WandB, hooks, or
LLM-powered modes, load the matching reference from the Reference Map.

## Runner Rules

- Create a fresh timestamped runner for every new AutoML request after preflight passes.
- Resume an existing runner only when the user explicitly asks to resume, continue, recover, or inspect an existing experiment.
- Use a timestamped `workspace_path`; never use a flat reusable workspace.
- Pass the selected model directory as `AutoMLRunner(skill_dir=<bank-root>/models/<network>, action="train")`.
- Pass the confirmed container image into `runner.run(..., image=chosen_image, ...)` or the SDK adapter.
- Build `spec_overrides` from the selected model skill; do not invent spec keys in this workflow.
- Keep chat-side monitoring attached while `long_running_enabled=true` until a terminal state or explicit detach request.

Minimal runner shape lives in `references/runner-api.md`; use it only after the
launch checklist has passed.

## LLM Algorithm Gate

For `llm`, `hybrid`, or `autoresearch`, resolve all three settings before script
generation:

1. `llm_endpoint`: user input -> `AUTOML_LLM_ENDPOINT` -> `https://inference-api.nvidia.com`
2. `llm_model`: user input -> `AUTOML_LLM_MODEL` -> model default from the user/model guidance
3. `llm_api_key`: `AUTOML_LLM_API_KEY` -> `NVIDIA_API_KEY` -> allowed local secret source -> prompt

Do not ask for LLM settings on the default bayesian quick-start path. If the
runner lacks valid LLM settings, the LLM brain can fall back to random sampling
and waste GPU budget.

## Result Reporting

Report:

- selected model/action/platform and job ids;
- confirmed image;
- workspace path;
- best recommendation and metric;
- failed recommendations and failure-analysis summary;
- whether the metric came from logs, a custom `metric_extractor`, or `eval_fn`;
- exact resume/status command or watcher path for long-running jobs.

If all recommendations fail, summarize common failure modes across jobs, include
one or two representative log excerpts, and do not claim AutoML found a best
configuration.
