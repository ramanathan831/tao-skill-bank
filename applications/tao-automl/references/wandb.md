# Wandb

WandB setup and reporting guidance for AutoML sweeps.

## Contents

- Source sections copied from the prior `applications/tao-automl/SKILL.md`.
- Read only when the compact AutoML skill points to this detail.

## WandB Experiment Tracking

AutoML optionally integrates with [Weights & Biases](https://wandb.ai) to track all experiments in a single dashboard.

### Setup

```bash
pip install wandb
# or (when reinstalling tao-automl with the wandb extra):
#   pip install "nvidia-tao-automl[wandb] @ git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-automl.git"
```

### How it works

When `wandb_config={"enabled": True}` is passed:

1. The controller creates a WandB **run** named `automl_brain` in the specified project.
2. All recommendations are grouped under a WandB **group** (e.g. `automl_abc123`) so parent + child training runs appear together in the dashboard.
3. After every result, a **WandB table** (`automl_experiments`) is logged containing:
   - `experiment_id`, `job_id`, `status`, metric value, `best_epoch_number`
   - All varying hyperparameter values
4. Call `automl.finish()` (or let `runner.run()` complete) to finalize the WandB run.

### Minimal WandB setup

```python
# Option 1: via config dict
result = runner.run(
    ...,
    wandb_config={
        "enabled": True,
        "project": "tao-hpo",
        "api_key": "your-key",  # or set WANDB_API_KEY env var
    },
)

# Option 2: environment variable (simpler)
# export WANDB_API_KEY=your-key
result = runner.run(
    ...,
    wandb_config={"enabled": True, "project": "tao-hpo"},
)
```

### Dashboard features

Once tracking is active, you can:
- **Compare all trials** side-by-side in the WandB table view
- **Sort by metric** to find the best config instantly
- **Group by hyperparameter** to see which values correlate with good results
- **Link to child training runs** if the compute backend also logs to WandB (group name is available via `automl.wandb_group`)

---
