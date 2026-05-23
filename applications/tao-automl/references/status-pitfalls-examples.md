# Status Pitfalls Examples

Model-specific note policy, common pitfalls, status queries, algorithm decision tree, and example conversations.

## Contents

- Source sections copied from the prior `applications/tao-automl/SKILL.md`.
- Read only when the compact AutoML skill points to this detail.

## Model-Specific Notes

Model-specific notes do not belong in this AutoML skill. For every requested `network_arch`, read `<bank-root>/models/<network>/SKILL.md` and use its **Training Requirements**, **Per-Action Dataset Requirements**, **Typical Spec Overrides**, **AutoML / HPO Notes**, and **Error Patterns** sections as the source of truth.

---

## Common Pitfalls

1. **`skill_dir` not passed (or wrong path).** `AutoMLRunner(skill_dir=...)` requires an absolute path to a model directory inside the skill bank. The runner raises `FileNotFoundError: skill_info.yaml not found at <skill_dir>/references/skill_info.yaml` if the path is wrong. Use the same bank root the agent loaded this SKILL.md from; combine with `models/<network>/`.
2. **Wrong LLM endpoint (404).** The code hardcodes `https://integrate.api.nvidia.com/v1` as the default, which returns 404. The correct endpoint is `https://inference-api.nvidia.com`. ALWAYS pass `llm_endpoint` explicitly in `automl_settings`. The LLM brain silently falls back to random sampling on 404, so you won't see a crash — just useless random configs.
3. **Model-specific training failures (data format, missing datasets, invalid params).** Each network has unique training requirements. ALWAYS read `<bank-root>/models/<network>/SKILL.md` — the "Training Requirements" and "Error Patterns" sections document model-specific failure modes that apply to AutoML recs too.
4. **Workspace path collisions.** Running the same script twice overwrites the previous experiment. Always include a timestamp: `workspace_path=f"./automl_workspace/{TIMESTAMP}"` where `TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")`.
5. **Using a weak proxy metric.** The brain can optimize a metric that does not reflect real task quality. Use the metric recommended by the model skill or provide `eval_fn`.
6. **Implicit direction trap.** If the metric name does not imply the desired direction, set `direction` explicitly.
7. **Spec-override typos.** `save_freq_in_epochs` (plural) used to silently do nothing; now raises `ValueError` with suggestion. If you see that error, it's the fix working.
8. **Orchestrator dies mid-sweep.** Relaunch with the same `workspace_path` and `resume=True`. In-flight jobs are recovered from `active_jobs.json`.
9. **Rec never reports a metric.** Check the model skill's metric-emission requirements and custom extractor guidance.
10. **Parallel Bayesian arms.** Bayesian is inherently sequential. If you want parallelism, use `asha`. If you use multiple `AutoMLRunner` instances, give each its own `<SDK>(state_file=...)` (e.g., `LeptonSDK(state_file=...)`, `KubernetesSDK(state_file=...)`) to avoid SQLite write races on the SDK's job store.
11. **LLM brain returning random configs.** If every LLM recommendation looks random, the LLM endpoint is probably failing silently. Check the logs for "LLM call failed" warnings. Verify your API key and endpoint are correct. Common cause: using the wrong endpoint URL (see pitfall #2).
12. **`openai` package not installed.** The `llm`, `hybrid`, and `autoresearch` algorithms require the `openai` Python package. Install with `pip install openai` or reinstall tao-automl with the `[llm]` extra (see Preflight for the `git+https://...` direct-URL form).
13. **WandB not logging.** Ensure `wandb_config={"enabled": True}` is passed and either `api_key` is in the config or `WANDB_API_KEY` is set in the environment. Check logs for "WandB initialized" confirmation.
14. **`No default train specs found` for a network.** The skill bank model directory is missing `references/spec_template_train.yaml`, or the packaged AutoML support check is missing `schemas/train.schema.json`. Generate both during skill-bank maintenance and ship them with the plugin; do not expect `~/tao-core` to exist on the runtime machine.
15. **`conda run` buffers output.** When running AutoML via `conda run -n tao_sdk python script.py`, all output is buffered until completion. Use `PYTHONUNBUFFERED=1 ~/miniconda3/envs/tao_sdk/bin/python script.py` for real-time output.

---

## Querying Experiment Status

Use `query_status()` to check experiment progress from a separate process — no need to read JSON files or parse logs.

```python
from tao_automl import query_status

status = query_status("./my_experiment")

# Progress summary
p = status["progress"]
print(f"{p['completed']}/{p['total']} recs done, "
      f"{p['succeeded']} succeeded, {p['failed']} failed")

# Best config
if status["best"]:
    print(f"Best: rec {status['best']['rec_id']}, "
          f"metric={status['best']['metric_value']}, "
          f"specs={status['best']['specs']}")

# Per-rec details
for rec in status["recommendations"]:
    print(f"  Rec {rec['rec_id']}: {rec['status']} "
          f"metric={rec['metric_value']} specs={rec['specs']}")

# In-flight jobs
for job in status["active_jobs"]:
    print(f"  Active: rec {job['rec_id']} job {job['job_id']}")
```

The function reads from the persisted state store (`<workspace>/.automl/`) and `active_jobs.json`. It is safe to call while the runner is active — no locking conflicts.

The `AutoML` class also exposes `get_status()` for in-process queries:

```python
automl = AutoML(workspace=..., ...)
status = automl.get_status()
```

---

## Quick Reference: Algorithm Decision Tree

```
Is your budget tiny (≤10 recs)?
  YES → bayesian
  NO  ↓

Do you have an LLM API key and want AI-guided search?
  YES → Do you want full autonomy? → autoresearch
        Just LLM proposals?        → llm
        LLM orchestrating phases?  → hybrid
  NO  ↓

Do you need parallel execution?
  YES → asha (or bohb for smarter sampling)
  NO  ↓

Is your search space large (10+ parameters)?
  YES → hyperband or dehb
  NO  ↓

Do hyperparameters need to change during training (schedules)?
  YES → pbt
  NO  → bayesian (safe default)
```

---

## Example Conversations

### User: "Optimize `<network_arch>` training on my dataset"

```
Agent: I'll read the model skill for `<network_arch>` first, then generate a fresh timestamped runner using its required datasets, spec overrides, metric, and recommended search space. I need:
1. The dataset URI or URIs required by that model skill
2. How many trials? (default: 10)

User: s3://bucket/data/train, 10 recs

Agent: Running a fresh job with a new runner, log, state file, and workspace. The model-specific metric and overrides come from `<network>.md`.
[executes runner.run(...)]
```

### User: "I want the real task metric, not the default proxy"

```
Agent: For a real task metric, I'll use the eval_fn hook described by the model skill. This adds per-rec cost, so I’ll adjust the budget if needed.
[executes runner.run(metric=task_metric, direction=direction, eval_fn=model_specific_eval, ...)]
```

### User: "Use the LLM to figure out the best hyperparameters"

```
Agent: I'll use the LLM algorithm — it reasons about your network architecture and learns from each experiment.
I need three things for the LLM brain:
1. LLM endpoint URL (default: https://inference-api.nvidia.com)
2. LLM model name (default: meta/llama-3.1-70b-instruct, or e.g. gcp/google/gemini-3.1-pro-preview)
3. API key for the endpoint (or set NVIDIA_API_KEY env var)

User: endpoint "https://inference-api.nvidia.com", model "gcp/google/gemini-3.1-pro-preview", key "sk-abc123"

Agent: Running LLM-guided search with 10 recs. The LLM will explain its reasoning for each config choice in the logs.
[executes runner.run(automl_settings={
    "algorithm": "llm",
    "llm_endpoint": "https://inference-api.nvidia.com",
    "llm_model": "gcp/google/gemini-3.1-pro-preview",
    "llm_api_key": "sk-abc123",
    ...
})]
```

### User: "Run a fully autonomous search — let the AI figure everything out"

```
Agent: I'll use the autoresearch algorithm. It will:
1. Retrieve tuning knowledge for your network
2. Propose modifications, pre-screen them, verify before launch
3. Make keep/discard decisions with reasoning after each experiment
4. Automatically stop when it detects diminishing returns

I'll cap it at 30 experiments. I need:
- Dataset URI
- LLM endpoint (default: https://inference-api.nvidia.com)
- LLM model (default: meta/llama-3.1-70b-instruct)
- LLM API key (or set NVIDIA_API_KEY env var)

User: dataset s3://bucket/data, endpoint https://inference-api.nvidia.com, model gcp/google/gemini-3.1-pro-preview, key sk-abc123

[executes runner.run(automl_settings={
    "algorithm": "autoresearch", "automl_max_experiments": 30,
    "llm_endpoint": "https://inference-api.nvidia.com",
    "llm_model": "gcp/google/gemini-3.1-pro-preview",
    "llm_api_key": "sk-abc123",
    ...
})]
```

### User: "Resume the experiment from yesterday"

```
Agent: Resuming. The runner will poll any jobs that were in flight when we stopped, report their results, then continue.
[executes runner.run(..., workspace_path="./my_experiment", resume=True)]
```

### User: "Switch to ASHA with 30 trials and track in WandB"

```
Agent: Switching to ASHA — parallel + early-stopping. I’ll first check the model skill for any warning that short trials are dominated by startup, validation, or checkpoint cost, then generate a fresh runner with WandB tracking.
[executes runner.run(
    automl_settings={"algorithm": "asha", "automl_max_epochs": 9, ...},
    wandb_config={"enabled": True, "project": "tao-hpo"},
)]
```

### User: "I don't know what parameters to tune for `<network_arch>`"

```
Agent: I can generate a configuration from your goal description. What are you trying to achieve?

User: Maximize my task metric on a small custom dataset with 1 GPU.

Agent: Based on your constraints, I recommend:
- Algorithm: bayesian (sample-efficient for small budgets)
- Parameters: from the model skill's AutoML / HPO Notes and the generated schema
- Budget: 12 recs
- Ranges: from the model skill and user constraints
[uses NLConfigGenerator, then executes runner.run with the generated config]
```
