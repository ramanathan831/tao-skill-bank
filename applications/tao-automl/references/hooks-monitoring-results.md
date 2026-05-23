# Hooks Monitoring Results

Optional metric/eval hooks, monitoring, resume, result interpretation, and user reporting guidance.

## Contents

- Source sections copied from the prior `applications/tao-automl/SKILL.md`.
- Read only when the compact AutoML skill points to this detail.

## Advanced hooks (opt-in)

Both hooks are optional. If neither is provided, the runner uses its built-in log regex extractor.

### `metric_extractor(logs: str, metric_name: str) → float | None`

Called on every poll of the training container's logs. Return the most recent/final metric value seen, or `None` if the metric isn't yet present.

Use it when:
- Your container emits the metric in a non-standard log format the built-in regex misses.
- You want to parse values from log lines instead of using the generic patterns.
- Your metric needs derivation from multiple log fields.

```python
import re

def extract_custom_metric(logs: str, metric_name: str):
    m = re.search(rf"{re.escape(metric_name)}:\s*([0-9.]+)", logs)
    return float(m.group(1)) if m else None

runner.run(..., metric_extractor=extract_custom_metric)
```

Exceptions raised inside the extractor are caught and logged; the runner continues polling.

### `eval_fn(rec, train_job_id: str) → float | None`

Called once after a rec's training job reaches a terminal state, before the result is reported to the brain. Whatever it returns **overrides** any value captured by `metric_extractor` and becomes what the brain optimizes on.

Use it when:
- The real task metric lives outside the training logs.
- You want a true-test-metric sweep without building surrounding plumbing yourself.
- Per-rec cost is acceptable relative to `metric_extractor`.

```python
def eval_on_held_out(rec, train_job_id):
    # Implement the model-specific evaluation flow documented in the model skill.
    metric_value = run_model_specific_eval(rec, train_job_id)
    return metric_value

runner.run(
    ...,
    automl_settings={"metric": task_metric, "direction": direction, ...},
    eval_fn=eval_on_held_out,
)
```

Exceptions from `eval_fn` are caught and logged — the runner falls back to the log-extracted metric for that rec.

---

## Step 4: Monitor Progress

`runner.run()` blocks until all recommendations complete. Use callbacks to report progress to the user:

```python
def on_rec(rec):
    print(f"Rec {rec.id}: trying {rec.specs}")

def on_result(rec, metric, status):
    print(f"Rec {rec.id}: {status}, metric={metric}")

result = runner.run(..., on_recommendation=on_rec, on_result=on_result)
```

Each rec takes 10–90 minutes depending on model size, dataset, epochs, and checkpoint save cost. Don't assume failure during long uploads.

### Resume after interruption

If the orchestrator dies mid-run (network timeout, machine sleep, Ctrl-C), re-run with `resume=True` and the **full suffixed path** (including the `run_<timestamp>` directory):

```python
result = runner.run(
    ...,
    workspace_path="./my_experiment/run_20260423_183015",   # full suffixed path
    resume=True,
)
```

When `resume=True`, the runner does NOT append a new timestamp suffix — it reuses the path as-is.

Behaviour on resume:
1. **Brain state** is reloaded from `<workspace>/.automl/*` — all completed rec results are already registered.
2. **Any in-flight jobs** recorded in `<workspace>/active_jobs.json` (persisted after each submission) are polled to terminal, their metrics extracted, and reported to the brain — *before* the main propose-new-rec loop starts. No duplicate submissions; no leaked GPU work from the previous orchestrator.
3. After recovery, the loop continues normally until `automl.is_complete()`.

---

## Step 5: Interpret Results

The result is a plain dict:

```python
{
    "best": {
        "rec_id": 4,
        "specs": {"<param_name>": "<value>", "...": "..."},
        "metric_value": 0.7077,
    },
    "progress": {
        "completed": 8, "total": 8,
        "best_metric": 0.7077, "best_rec_id": 4,
        "algorithm": "bayesian",
    },
    "history": [
        {"rec_id": 0, "metric": 0.6308, "status": "success"},
        {"rec_id": 1, "metric": 0.7077, "status": "success"},
        ...
    ],
}
```

Metric values in `best` and `history` are always in the original scale the user provided — direction inversion (if any) is undone before the dict is returned.

### How to report to the user

1. **Best config** — show the winning hyperparameters and metric value.
2. **Comparison table** — rank all recs by metric, highlight the best.
3. **Insights** — call out what the optimizer learned from the requested parameters and metric.
4. **WandB link** — if tracking was enabled, provide the dashboard URL.
5. **Next steps** — suggest:
   - More recs (re-run with `resume=True` + higher `automl_max_recommendations`).
   - Train longer with the best config using `sdk.create_job(specs=result["best"]["specs"])`.
   - Run a downstream evaluation on the best checkpoint.
   - Run the model skill's recommended export/deploy workflow for the best model.

### If all recs failed

Check common issues:
- **Dataset path wrong** — verify the URI points to the layout required by the model skill.
- **Metric never appears** — verify the model skill's required metric-related overrides and custom extractor are present.
- **Checkpoint or eval artifact missing** — verify the model skill's checkpoint/export/eval requirements.
- **Model or data download timeout** — inspect backend logs and model-skill error patterns.
- **OOM** — reduce the model-specific batch, resolution, sequence length, or memory-heavy knobs recommended by the model skill.
- **Cached data corruption** — inspect the model skill's dataset/cache error patterns and clear only the affected cache path if documented.
- **LLM endpoint unreachable** (llm/hybrid/autoresearch only) — the brain falls back to random sampling. Check `AUTOML_LLM_ENDPOINT` and `AUTOML_LLM_API_KEY`. Verify with: `curl -s $AUTOML_LLM_ENDPOINT/models -H "Authorization: Bearer $AUTOML_LLM_API_KEY"`.

---
