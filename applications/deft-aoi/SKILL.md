---
name: deft-aoi
description: "Unified DEFT pipeline for AOI defect detection — iterative gap analysis, multi-arm augmentation, and retraining with KPI-driven loop control. Use when running iterative model improvement for PCB defect detection or visual inspection."
---

# DEFT AOI Pipeline

Data-Efficient Fine-Tuning for Automated Optical Inspection. Iteratively improves Visual ChangeNet defect detection by identifying failure cases, augmenting training data, and retraining until KPI targets are met.

## Prerequisites

| Name | Required | Description |
|------|:--------:|-------------|
| `kpi_dataset_uri` | Yes | S3 URI of KPI/validation dataset (CSV + images/) |
| `platform` | Yes | Compute backend: lepton, brev |
| `train_dataset_uri` | No | Initial training data (triggers SFT before loop) |
| `mining_source_csv` | No | Source pool CSV for k-NN mining arm |
| `mining_source_images` | No | Source pool images directory |
| `kpi_target` | No | Target KPI string, e.g. "FAR < 0.5 at 100% recall" |
| `max_iterations` | No | Max DEFT iterations (default: 3) |

## Storage Layout

All intermediate artifacts are organized under a `storage_root` per iteration. Use the job's `results_dir` for SDK-executed steps, or a fixed S3 path for the iteration.

```
s3://{bucket}/deft-aoi/{run_id}/
├── init/
│   ├── inference/                    ← zero-shot VCN inference output
│   │   └── inference.csv
│   ├── threshold.json                ← optimal threshold + metrics
│   ├── metrics.json                  ← eval metrics (FAR, recall, F1)
│   ├── sft/                          ← optional SFT results
│   │   ├── train/                    ← training checkpoints
│   │   └── inference/                ← SFT inference output
│   ├── sft_threshold.json
│   └── sft_metrics.json
│
├── iter_1/
│   ├── gaps/
│   │   └── gaps.parquet              ← FP/FN failure cases
│   ├── embeddings/
│   │   ├── target_queries.parquet    ← gap images for embedding
│   │   ├── source_pool.parquet       ← source pool metadata
│   │   ├── target_embeddings.parquet ← SigLIP embeddings of gaps
│   │   └── source_embeddings.parquet ← SigLIP embeddings of source
│   ├── mining/
│   │   └── mined_pairs.parquet       ← k-NN mined similar images
│   ├── sdg_output/
│   │   ├── anomalygen/               ← AnomalyGen synthetic images
│   │   └── omniverse/                ← Omniverse SDG images
│   ├── training_data/
│   │   └── annotations.csv           ← merged training CSV
│   ├── train/                        ← training checkpoints
│   ├── inference/                    ← post-training inference
│   │   └── inference.csv
│   ├── threshold.json
│   ├── eval_metrics.json
│   └── eval_gaps.parquet
│
├── iter_2/
│   └── ...
│
└── deft_state.json                   ← crash recovery state
```

## Pipeline Steps — Init Phase

### Step 0: Zero-shot Inference
- **Skill:** `visual-changenet` action `inference`
- **Input:** `kpi_dataset_uri` (eval dataset)
- **Output:** `{init}/inference/` → produces `inference.csv`
- **Note:** Uses pretrained checkpoint (no fine-tuning yet)

### Step 1: Threshold Optimization
- **Skill:** `vcn-threshold-optimize` action `compute`
- **Input:** `{init}/inference/` (from step 0)
- **Input:** `min_recall` (default: 1.0)
- **Output:** `{init}/threshold.json`

### Step 2: Baseline Evaluation
- **Skill:** `vcn-gap-analysis` action `analyze` mode `eval-only`
- **Input:** `{init}/inference/` (from step 0)
- **Input:** `{init}/threshold.json` (from step 1)
- **Input:** `kpi_dataset_uri` (ground truth CSV)
- **Output:** `{init}/metrics.json`

### Step 3: SFT Training (conditional — only if `train_dataset_uri` provided)
- **Skill:** `visual-changenet` action `train`
- **Input:** `train_dataset_uri`
- **Output:** `{init}/sft/train/` → checkpoint

### Step 4-6: SFT Inference + Threshold + Eval (conditional — after step 3)
Same as steps 0-2 but using the SFT checkpoint.

## Pipeline Steps — Iteration Loop

Each iteration uses `{iter_N}` as storage root. `{prev}` refers to the previous iteration's storage root (or `{init}` for iteration 1).

### Step 1: RCA (agent-native)
- **Skill:** `rca-changenet`
- **Input:** Agent reads inference results + images from `{prev}/inference/`
- **Output:** RCA report (markdown) — agent uses findings to decide which SDG arms to activate
- **No SDK job** — agent runs this itself using 6 parallel subagents

### Step 2: Gap Analysis
- **Skill:** `vcn-gap-analysis` action `analyze` mode `gap-analysis`
- **Inputs:**
  - `inference-results` ← `{prev}/inference/`
  - `threshold-json` ← `{prev}/threshold.json`
  - `kpi-csv` ← `kpi_dataset_uri`
- **Outputs:**
  - `output-metrics` → `{iter_N}/metrics.json`
  - `output-gaps` → `{iter_N}/gaps/gaps.parquet`

### Step 3: Source Pool Expansion
- **Skill:** `vcn-source-pool` action `expand`
- **Inputs:**
  - `gaps-parquet` ← `{iter_N}/gaps/gaps.parquet` (from step 2)
  - `source-csv` ← `mining_source_csv` (user-provided)
  - `source-images-dir` ← `mining_source_images` (user-provided)
- **Outputs:**
  - `output-target-parquet` → `{iter_N}/embeddings/target_queries.parquet`
  - `output-source-parquet` → `{iter_N}/embeddings/source_pool.parquet`

### Step 4a: Embed Target Images
- **Skill:** `siglip-embed` action `embed`
- **Inputs:**
  - `input_parquet` ← `{iter_N}/embeddings/target_queries.parquet` (from step 3)
- **Outputs:**
  - `output_parquet` → `{iter_N}/embeddings/target_embeddings.parquet`

### Step 4b: Embed Source Pool
- **Skill:** `siglip-embed` action `embed`
- **Inputs:**
  - `input_parquet` ← `{iter_N}/embeddings/source_pool.parquet` (from step 3)
- **Outputs:**
  - `output_parquet` → `{iter_N}/embeddings/source_embeddings.parquet`
- **Note:** Steps 4a and 4b can run in parallel

### Step 5: k-NN Mining
- **Skill:** `knn-mining` action `mine`
- **Inputs:**
  - `source_parquet` ← `{iter_N}/embeddings/source_embeddings.parquet` (from step 4b)
  - `target_parquet` ← `{iter_N}/embeddings/target_embeddings.parquet` (from step 4a)
- **Outputs:**
  - `output_parquet` → `{iter_N}/mining/mined_pairs.parquet`

### Step 6: Augmentation Arms (parallel, conditional)

**Arm A — AnomalyGen** (if RCA recommends):
- **Skill:** `anomalygen` action `inference`
- **Input:** Clean images + ROI masks + defect descriptions from RCA
- **Output:** `{iter_N}/sdg_output/anomalygen/` → paired defect/golden images

**Arm B — Omniverse SDG** (if RCA recommends):
- **Skill:** `omniverse-sdg` action `generate`
- **Input:** USD scene file + target defect types from RCA
- **Output:** `{iter_N}/sdg_output/omniverse/` → paired defect/golden images

**Arm C — Data Mining** (always, if source pool available):
- Steps 3-5 above produce mined pairs

Arms A and B run in parallel with each other (and with the mining pipeline).

### Step 7: Data Prep
- **Skill:** `changenet-data-prepare` action `prepare`
- **Input:** SDG output directories (anomalygen + omniverse + mined images)
- **Output:** CSV annotations for the augmented images

### Step 8: Merge Training CSV
- **Skill:** `vcn-merge-csv` action `merge`
- **Inputs:**
  - `mined-parquet` ← `{iter_N}/mining/mined_pairs.parquet` (from step 5)
  - `source-pool-parquet` ← `{iter_N}/embeddings/source_pool.parquet` (from step 3)
  - `prev-train-csv` ← `{prev}/training_data/annotations.csv` (or `train_dataset_uri` for iter 1)
- **Outputs:**
  - `output-csv` → `{iter_N}/training_data/annotations.csv`

### Step 9: Training
- **Skill:** `visual-changenet` action `train`
- **Input:** `{iter_N}/training_data/annotations.csv` (from step 8)
- **Input:** Previous checkpoint (from `{prev}/train/` or init SFT)
- **Output:** `{iter_N}/train/` → new checkpoint

### Step 10: Inference
- **Skill:** `visual-changenet` action `inference`
- **Input:** `kpi_dataset_uri` + checkpoint from step 9
- **Output:** `{iter_N}/inference/` → `inference.csv`

### Step 11: Threshold Optimization
- **Skill:** `vcn-threshold-optimize` action `compute`
- **Input:** `{iter_N}/inference/` (from step 10)
- **Output:** `{iter_N}/threshold.json`

### Step 12: Evaluation
- **Skill:** `vcn-gap-analysis` action `analyze` mode `eval-only`
- **Inputs:**
  - `inference-results` ← `{iter_N}/inference/` (from step 10)
  - `threshold-json` ← `{iter_N}/threshold.json` (from step 11)
  - `kpi-csv` ← `kpi_dataset_uri`
- **Outputs:**
  - `output-metrics` → `{iter_N}/eval_metrics.json`

### Step 13: KPI Check (agent-native)
- **Skill:** `analyze-kpi`
- Agent downloads `{iter_N}/inference/inference.csv` from S3
- Runs threshold analysis locally
- **Decision:**
  - KPI met → STOP, report final metrics
  - KPI not met, iterations remaining → CONTINUE to iter N+1
  - KPI not met, max iterations reached → STOP, select best iteration

## Checkpoint Chaining

| Iteration | Checkpoint used |
|-----------|----------------|
| Iter 1 (with SFT) | `{init}/sft/train/changenet_model_classify_latest.pth` |
| Iter 1 (no SFT) | Pretrained (bundled in container) |
| Iter N > 1 | `{iter_{N-1}}/train/changenet_model_classify_latest.pth` |

Pass checkpoint via spec override: `train.pretrained_model_path` (weight-loading, resets epoch counter).
Do NOT use `train.resume_training_checkpoint_path` (that resumes epoch counter).

## Dataset Accumulation

| Iteration | Training data |
|-----------|---------------|
| Iter 1 (with initial data) | `train_dataset_uri` merged with iter_1 augmented data |
| Iter 1 (no initial data) | iter_1 augmented data only |
| Iter N > 1 | `{iter_{N-1}}/training_data/annotations.csv` merged with iter_N augmented data |

## State Management

Write `deft_state.json` after every step:

```json
{
  "version": 1,
  "kpi_target": "FAR < 0.5 at 100% recall",
  "results_dir": "s3://bucket/deft-aoi/run-abc/",
  "max_iterations": 3,
  "current_iteration": 1,
  "current_step": "training",
  "iterations": {
    "baseline": {
      "status": "complete",
      "far": 0.85,
      "recall": 100,
      "threshold": 0.31,
      "checkpoint": "s3://bucket/.../init/sft/train/model_latest.pth"
    },
    "iter1": {
      "status": "in_progress",
      "completed_step": "gap_analysis",
      "sdg_images_added": 150,
      "mined_images_added": 200
    }
  }
}
```

## Crash Recovery

Six hooks in `hooks/`:

- **session-start-resume** — on new session, reads `deft_state.json`, injects state + iteration summaries
- **pre-bash-guard** — blocks `rm -rf` on results dir, `docker prune`, raw `>` into state file
- **post-bash-validate** — after Docker commands, catches OOM/CUDA/NCCL errors, records to state
- **pre-compact-checkpoint** — before context compression, reminds agent to save state
- **post-compact-restore** — after compression, re-injects state + summaries
- **stop-continue** — blocks session stop if loop is incomplete

## Skills Reference

| Skill | Type | Layer | Purpose |
|-------|------|-------|---------|
| `visual-changenet` | SDK | models | Training, inference |
| `rca-changenet` | Agent-native | applications | Root cause analysis (6 parallel subagents) |
| `vcn-gap-analysis` | SDK | data | Identify FP/FN + compute metrics |
| `vcn-threshold-optimize` | SDK | data | Sweep thresholds for optimal FAR/recall |
| `vcn-source-pool` | SDK | data | Prepare mining inputs |
| `siglip-embed` | SDK | data | Image embeddings |
| `knn-mining` | SDK | data | k-NN nearest neighbor retrieval |
| `anomalygen` | SDK | models | Diffusion-based SDG |
| `omniverse-sdg` | SDK | data | Physics-based SDG |
| `changenet-data-prepare` | SDK | data | CSV + image prep |
| `vcn-merge-csv` | SDK | data | Merge mined/augmented data |
| `analyze-kpi` | Agent-native | data | KPI threshold analysis |

