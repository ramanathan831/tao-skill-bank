---
name: tao-finetune-nv-tesseract-ad-diffusion
description: >-
  NV-Tesseract AD Diffusion — diffusion-based anomaly detection and fine-tuning
  for multivariate time series. Use when the user asks to "fine-tune NV-Tesseract",
  "run AD diffusion inference", "detect anomalies with diffusion", "time series
  anomaly detection", "finetune ad-diffusion", "use perform_anomaly_analysis_with_diffusion",
  "inference_ad_tesseract2", or mentions "curriculum_medium.yaml", "final_model.pth",
  "nv-tesseract-ad-diffusion", "ad_diffusion", or "TSDiffuser_Generic".
license: Apache-2.0
compatibility: Requires Python 3.12+ and uv. CUDA GPU recommended; CPU-only supported.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
  - anomaly-detection
  - time-series
  - diffusion
  - finetune
  - inference
  - nv-tesseract
---

# NV-Tesseract AD Diffusion

Diffusion-based anomaly detection and fine-tuning for multivariate time series. The model
reconstructs randomly masked segments and scores each timestep by MAE between reconstruction
and original signal; adaptive thresholding (SCS or MACS) converts scores to binary labels.

**Source code:** https://github.com/NVIDIA/NV-Tesseract  
**Pretrained weights:** https://huggingface.co/nvidia/nv-tesseract-ad-diffusion

## External dependencies

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.12+ | Runtime | https://www.python.org/downloads/ |
| uv | Package + environment manager | `pip install uv` |
| CUDA toolkit (optional) | GPU acceleration | https://developer.nvidia.com/cuda-downloads |
| huggingface_hub | Weight download from HF | Bundled via `uv sync` |

## Credentials

| Variable | Required | Description |
|---|---|---|
| `HUGGINGFACE_HUB_TOKEN` | Only if repo is gated | Token from https://huggingface.co/settings/tokens |

```bash
export HUGGINGFACE_HUB_TOKEN="hf_..."
# or authenticate once via CLI:
uv run huggingface-cli login
```

## Quick start

```bash
cd /path/to/NV-Tesseract/ad_diffusion
uv sync                              # install dependencies (one-time)

# Inference — synthetic data, auto-downloads weights from HF on first run
uv run python examples/quick_example.py

# Inference — your own CSV
uv run python examples/quick_example.py \
  --model-path final_model.pth \
  --config-path curriculum_medium.yaml \
  --dataset-path /path/to/data.csv

# Fine-tune on your own normal-behavior data
uv run python examples/finetune_example.py \
  --csv /path/to/normal_training_data.csv \
  --timestamp-col timestamp \
  --label-col is_anomaly \
  --epochs 20 \
  --output-dir artifacts/finetune_my_data
```

## Inference

### Choosing between the two inference APIs

| | `perform_anomaly_analysis_with_diffusion` | `inference_ad_tesseract2` |
|---|---|---|
| **Location** | `sdk/anomaly_analysis.py` | `sdk/inference_ad.py` |
| **Calls** | `inference_ad_tesseract2_mp` internally | single-GPU diffusion directly |
| **Multi-GPU** | automatic — uses all visible GPUs | single-GPU only |
| **Input validation** | rejects non-numeric columns; checks rows ≥ `target_dim` | none — caller's responsibility |
| **Thresholding** | applies SCS or MACS → binary `Anomaly` labels | none — raw MAE scores |
| **Output** | original DataFrame + `Anomaly` (0/1) and `MAE` columns | `dict`: `residual`, `residual_l2`, `target`, `recon`, `target_dim` |
| **Length alignment** | truncates model output to match input DataFrame | not handled |
| **Use when** | want labels attached to DataFrame, multi-GPU automatic | need raw scores, custom threshold, or `recon`/`target` access |

### High-level helper (recommended)

```python
import sys, pandas as pd
sys.path.append("/path/to/NV-Tesseract/ad_diffusion")
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion

df = pd.read_csv("your_data.csv")

results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",   # "scs" (fast) or "macs" (adaptive)
    model_path=None,             # None → auto-download final_model.pth from HF
    config_path=None,            # None → auto-download curriculum_medium.yaml from HF
    nsample=15,                  # diffusion samples per window; ↑ accuracy, ↑ latency
)
# results columns: Anomaly (0/1), MAE (float), plus all original columns
print(results[["Anomaly", "MAE"]].describe())
```

### Low-level inference function

Single-GPU, no thresholding, no input validation. Use when you need raw scores,
a custom threshold, or direct access to `recon`/`target` arrays.

```python
from sdk.inference_ad import inference_ad_tesseract2, set_seed

set_seed(42)  # optional — ensures reproducible diffusion samples
results = inference_ad_tesseract2(
    data=df,
    model_path="final_model.pth",
    config_path="curriculum_medium.yaml",
    nsample=30,
    use_dpm_solver=True,   # 50-100× speedup over standard 500-step diffusion
    dpm_steps=20,          # 10–50; fewer = faster
)
# dict keys: "residual" (MAE), "residual_l2", "target", "recon", "target_dim"
```

### Multi-GPU inference

```python
from sdk.inference_ad import inference_ad_tesseract2_mp

results = inference_ad_tesseract2_mp(
    data=df,
    model_path="final_model.pth",
    nsample=15,
    num_processes=None,   # auto-detect GPU count
    use_dpm_solver=True,
    dpm_steps=20,
)
# Falls back to single-GPU when only 1 GPU is available.
# 4 GPUs × 50× DPM-Solver ≈ 200× vs single-GPU standard diffusion.
```

### Inference CLI reference

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | synthetic | CSV with numeric feature columns |
| `--model-path` | auto-download | Path to `.pth` checkpoint |
| `--config-path` | auto-download | Path to `curriculum_medium.yaml` |
| `--download-weights` | — | Fetch weights from HF and exit |
| `--skip-download` | false | Require local weights; skip HF fetch |

## Fine-tuning

Fine-tune on **normal-behavior windows** from your domain. The model learns masked
imputation on your data, shifting the reconstruction baseline to match your signal
patterns. Anomalies in the training set degrade the baseline — filter them out first.

```bash
uv run python examples/finetune_example.py \
  --csv /path/to/normal_data.csv \
  --val-csv /path/to/val_data.csv \   # optional; otherwise --val-ratio splits --csv
  --pretrained-model final_model.pth \
  --epochs 20 --batch-size 16 --lr 1e-5 \
  --output-dir artifacts/finetune_my_data
```

### Fine-tuning arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | **required** | Training CSV (normal behavior only) |
| `--val-csv` | — | Separate validation CSV |
| `--val-ratio` | `0.3` | Validation fraction when `--val-csv` not used (temporal split) |
| `--timestamp-col` | `timestamp` | Column to drop from features |
| `--label-col` | — | Label column to drop |
| `--drop-cols` | — | Comma-separated extra columns to drop |
| `--pretrained-model` | `final_model.pth` | Warm-start checkpoint (auto-downloaded if missing) |
| `--config` | `curriculum_medium.yaml` | Model config YAML |
| `--epochs` | `10` | Training epochs |
| `--batch-size` | `16` | Per-GPU batch size |
| `--lr` | `1e-5` | AdamW learning rate |
| `--weight-decay` | `1e-6` | AdamW weight decay |
| `--grad-clip` | `1.0` | Gradient norm clip |
| `--window-length` | config (100) | Sliding window length in timesteps |
| `--window-stride` | `1` | Step between consecutive windows |
| `--split` | config (10) | Alternating mask segments per window |
| `--mask-ratio` | `0.7` | Fraction of each window masked during training |
| `--seed` | `42` | Random seed |
| `--output-dir` | `artifacts/finetune` | Output directory |
| `--no-download` | false | Fail if pretrained weights are not local |

### Running inference with a fine-tuned checkpoint

```python
results = perform_anomaly_analysis_with_diffusion(
    df=df,
    threshold_strategy="scs",
    model_path="artifacts/finetune_my_data/best_finetuned_model.pth",
    config_path="artifacts/finetune_my_data/finetune_config.yaml",
    nsample=15,
)
```

## Data requirements

| Property | Requirement |
|---|---|
| Rows | ≥ `window_length` (default **100**); ≥ `target_dim` (default **18**) for PCA |
| Columns | Any count of numeric columns; non-numeric dropped automatically |
| Values | No NaN / ±Inf — fill before passing to the API |
| Feature count > `target_dim` | PCA reduction to `target_dim`; needs ≥ `target_dim` rows |
| Feature count < `target_dim` | Zero-padded to `target_dim` |

```
timestamp,sensor_1,sensor_2,sensor_3
2024-01-01 00:00:00,0.42,1.10,-0.33
...
```

Pass only numeric feature columns; drop `timestamp`, IDs, and label columns before
calling the API. The high-level helper does this automatically.

## Output structure

**Inference** (`examples/quick_example.py`):

```
examples/datasets/
└── anomaly_results.csv      # original columns + Anomaly (0/1) + MAE
```

**Fine-tuning** (`--output-dir artifacts/finetune_my_data`):

```
artifacts/finetune_my_data/
├── best_finetuned_model.pth     # checkpoint with lowest validation loss
├── final_finetuned_model.pth    # checkpoint from last epoch
├── metrics.json                 # per-epoch train_loss and val_loss
└── finetune_config.yaml         # config used during training (for reproducibility)
```

## Model configuration (`curriculum_medium.yaml`)

| Field | Default | Description |
|---|---|---|
| `model.target_dim` | `18` | Internal feature dim; data is PCA'd/padded to this |
| `dataset.window_length` | `100` | Sliding window size in timesteps |
| `dataset.split` | `10` | Alternating mask segments per window |
| `dataset.scale_factor` | `1` | Scale multiplier after min-max normalization |
| `diffusion.num_steps` | `500` | Full diffusion steps (overridden by DPM-Solver) |
| `diffusion.channels` | `128` | Model hidden dimension |
| `diffusion.layers` | `6` | Transformer encoder layers |

## Hardware

| Tier | Setup | Notes |
|---|---|---|
| Minimum | 1× CPU | Functional; DPM-Solver reduces steps 500 → 20 |
| Recommended | 1× NVIDIA GPU (≥8 GB VRAM) | Strongly recommended for fine-tuning |
| Multi-GPU inference | 2–8× NVIDIA GPUs | `inference_ad_tesseract2_mp`; scales linearly |

## Known pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| `HfHubHTTPError: 401` | Repo gated or token missing | `export HUGGINGFACE_HUB_TOKEN="hf_..."` or `huggingface-cli login` |
| `ValueError: No numeric columns` | All columns are strings/dates | Drop non-numeric columns before calling API |
| `ValueError: PCA needs at least target_dim rows` | Fewer rows than `target_dim` (18) | Provide a longer time series |
| `ValueError: Need at least N rows` (finetune) | Split shorter than `window_length` | Ensure each train/val split has ≥ 100 rows |
| `RuntimeError: CUDA out of memory` | Batch too large | Reduce `--batch-size` or `nsample` |
| All MAE scores identical | Constant-value columns | Drop zero-variance columns before calling API |
| `ModuleNotFoundError: sdk` | Wrong working directory | `cd ad_diffusion/` before `uv run`, or add it to `sys.path` |
| Slow inference on CPU | Many diffusion windows | Reduce `nsample` to 5–10 for smoke tests |
