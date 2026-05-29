---
name: tao-run-automl
description: Run AutoML / hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner. Handles algorithm
  selection (bayesian, hyperband, asha, bohb, llm, hybrid, autoresearch), WandB experiment tracking, job execution on any TAO SDK
  platform, result interpretation, and per-rec custom evaluation hooks. Use when the user mentions TAO AutoML, hyperparameter
  optimization, HPO, automl, automl_settings, AutoMLRunner, tao_automl, bayesian search, hyperband, ASHA, LLM-guided search,
  autoresearch, or wants to tune training hyperparameters for any TAO network. Platform-agnostic — runs on any SDK (Brev,
  SLURM, Kubernetes, Docker).
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: '0.2'
allowed-tools: Read Bash Write
tags:
- automl
- hpo
- workflow
- training
- optimization
- llm
---

# TAO AutoML

Run automated hyperparameter optimization for a TAO model by combining:

1. The selected model skill under `models/<network>/`.
2. The selected platform skill under `platform/<platform>/`.
3. `AutoMLRunner`, which generates recommendations, launches train jobs,
   extracts metrics, and feeds results back to the optimizer.

Do not launch until model metadata, platform preflight, data visibility,
credentials, image choice, and compute shape are all proven.

## Reference Map

- `references/skill_info.yaml`: this workflow's structured metadata.
- Split detailed references: `automl-preflight-concepts.md` for prerequisites
  and support checks; `automl-intent-algorithms.md` for search policy;
  `automl-runner-configuration.md` for runner/API/WandB details;
  `automl-advanced-monitoring.md` for hooks, resume, and pitfalls; and
  `automl-examples.md` for conversation examples. `detailed-guide.md` is only
  the map.
- `models/<network>/SKILL.md`: model-specific dataset requirements, metrics,
  HPO notes, checkpoint handoff, and known failures.
- `models/<network>/references/skill_info.yaml`: train action contract,
  container image, inputs, outputs, upload exclusions, and `mode`.
- `platform/<platform>/SKILL.md`: selected platform preflight, credentials,
  resource shape, monitoring, and cancellation.
- `skills/tao-launch-workflow/SKILL.md`: shared intake pattern for platform,
  credentials, dataset visibility, image confirmation, and user confirmation.

## Preflight

1. Run the shared launch intake. If the user has not chosen a platform, ask;
   Brev, SLURM, Kubernetes, and Docker are equal peers.
2. Run the selected platform skill's preflight before generating runner files.
3. Verify `nvidia-tao-automl` imports with the selected platform extra:

```bash
python -c "import tao_automl; from tao_automl.runner import AutoMLRunner; print('OK')"
```

If missing, show the exact install command and ask before installing:

```bash
REPO='git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-run-automl.git'
pip install "nvidia-tao-automl[<platform>] @ $REPO"
```

Use `[all]` only for development machines that need every backend. Add `,llm`
only when the user requests LLM-guided algorithms.

## Model Support Gate

Before every run:

1. Read the model `SKILL.md` and `references/skill_info.yaml`.
2. Confirm `automl_enabled: true` for the model or that the model skill
   explicitly routes train-stage requests to AutoML.
3. Confirm `<skill_dir>/schemas/train.schema.json` exists and parses. This is
   the AutoML search-space gate.
4. For non-TAO-Core models such as Cosmos-RL and CLIP, also require
   `references/spec_template_train.yaml`; otherwise the runner has no complete
   train defaults.
5. If any gate fails, do not improvise a search space. Report the missing
   package artifact.

## Inputs

Collect these before runner construction:

| Input | Requirement |
|---|---|
| `network_arch` | Model directory name under `models/`. |
| `platform` | One of the supported TAO platform skills. |
| `train_dataset` / `eval_dataset` | Use model-specific spec keys and dataset layout. |
| `results_root` | Local, Lustre, or S3 path appropriate for the platform. |
| `gpu_count`, `num_nodes` | Respect model and platform limits. |
| `container_image` | Resolve through model metadata and `versions.yaml`; show it to the user. |
| `automl_algorithm` | Default `bayesian` unless user asks for another algorithm or the model skill recommends one. |
| `metric`, `direction` | Prefer the model skill's validation/task metric. |
| `automl_budget` | Recommendation count, max epochs/rungs, concurrency, or population size as required by the algorithm. |

Never ask for secret values. Verify required env vars with
`[ -n "$VAR_NAME" ] && echo SET || echo UNSET`.

## Algorithm Policy

| Algorithm | Good fit | Required knobs |
|---|---|---|
| `bayesian` | Default for small/medium budgets and few parameters. | `num_recommendations`, metric, direction |
| `hyperband`, `asha` | Many configs with cheap early rungs; ASHA supports parallelism. | `max_epochs`, `reduction_factor`, optional `max_concurrent` |
| `bohb`, `dehb` | Mixed Bayesian/evolutionary search with multi-fidelity budgets. | same rung budget fields as Hyperband |
| `pbt` | Long training where schedules should mutate during training. | population and generation budget |
| `llm`, `hybrid`, `autoresearch` | User explicitly wants LLM-guided search and has an endpoint configured. | LLM endpoint config plus budget |

Prefer the model skill's recommendation over generic defaults. Avoid ASHA or
Hyperband when the model skill says startup, validation, or checkpoint cost
dominates short trials.

## Spec And Search Space

Build specs as nested dictionaries. If a model skill lists paths in dotted
notation for readability, walk the path and assign the nested leaf; do not store
flat dotted strings as spec keys.

Use the packaged train schema for:

- `automl_default_parameters`
- `automl_disabled_parameters`
- valid min/max ranges
- enums, option weights, conditions, dependencies, and popular parameters

User-provided search spaces must stay inside schema constraints. For integer
knobs with discrete choices, include the schema's required integer option shape
instead of a loose list if the model skill calls that out.

Data source overrides are mandatory unless the model skill says the launcher can
derive them. Preserve exact user-provided spec keys when the dataset uses direct
annotation/media paths.

## Metric Policy

Training loss is cheap but can be misleading. Prefer the model skill's task
metric. Use one of these:

- Log metric: `metric=<name>`, `direction=maximize|minimize`.
- `metric_extractor(logs, metric_name)`: parse the model's logs when the
  default resolver is ambiguous.
- `eval_fn(rec, train_job_id)`: run the model's evaluate action after each
  recommendation when the user wants a downstream task metric.

Do not map `kpi` to a metric unless the model skill explicitly defines that
mapping.

## Runner Construction

Use the selected platform SDK only after its preflight passes. Construct SDKs
without embedding credentials in code.

```python
from pathlib import Path
from tao_automl.runner import AutoMLRunner

skill_bank = Path("<absolute-tao-skill-bank>")
skill_dir = skill_bank / "models" / network_arch

runner = AutoMLRunner(
    skill_dir=str(skill_dir),
    platform_sdk=sdk,
    workspace_dir="<automl_workspace>",
)

result = runner.run(
    automl_algorithm=algorithm,
    automl_settings=automl_settings,
    spec_overrides=spec_overrides,
    automl_hyperparameters=automl_hyperparameters,
    custom_param_ranges=custom_param_ranges,
    metric_extractor=metric_extractor,  # optional
    eval_fn=eval_fn,                    # optional
)
```

Only resume an existing workspace when the user explicitly asks to resume,
continue, recover, or inspect an existing experiment. Treat a plain "run
AutoML" request as a fresh run.

## Monitoring

Use `runner` status output and the platform SDK's `get_job_status`,
`get_job_logs`, and `get_failure_analysis`. For active jobs, report:

- recommendation id / trial id
- platform job id
- status
- current metric
- best metric so far
- selected hyperparameters for the current/best recommendation

On failure, classify whether it is infrastructure, data visibility, image,
credential, spec/schema, or model-code failure. Fix only the minimal cause and
do not silently spend additional budget on repeated invalid recommendations.

## Result Handoff

At completion:

1. Identify the best recommendation by the selected metric and direction.
2. Return the best train child job id and its result path.
3. Resolve the model checkpoint using the model skill's checkpoint metadata and
   SDK helpers; do not guess filenames such as `latest`.
4. Report the exact search space, algorithm, budget, metric, and platform.
5. If this feeds a workflow such as AutoML + DEFT, pass the winning spec
   overrides and checkpoint through the workflow's declared handoff fields.

## Common Pitfalls

- Do not expect `~/tao-core` at runtime. Schemas and templates must be packaged
  inside the model skill.
- Do not infer dataset URIs from previous runs.
- Do not precompute SDK-managed output paths; non-URI output values are routed
  by the SDK.
- For SLURM, stage large datasets on Lustre rather than burning GPU allocation
  time on large S3 downloads.
- For gated HuggingFace models, verify `HF_TOKEN` is set without reading it.
- If all recommendations fail, stop and summarize the shared root cause instead
  of launching more trials.
