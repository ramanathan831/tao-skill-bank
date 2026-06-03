---
name: tao-finetune-huggingface-model
description: >
  Fine-tune any HuggingFace CV / VLM / LLM model on local NVIDIA GPUs inside an
  NGC PyTorch container. Use when the user wants to fine-tune a HuggingFace
  model (full or LoRA), train a vision / VLM / LLM model end-to-end, generate a
  reproducible HF training pipeline, smoke-test a HuggingFace model locally
  before scale-up, push a fine-tuned model to the HF Hub with a model card, or
  emit a self-contained rerun skill for an existing HuggingFace finetune.
  Supports image classification, object detection, semantic / instance /
  panoptic segmentation, depth estimation, image-text-to-text VLM (SFT / LoRA),
  and LLM SFT / DPO / GRPO. Six-step workflow: inspect and qualify, hardware
  and NGC image, research, generate and smoke, train + eval + infer, push and
  emit rerun skill.
license: Apache-2.0
tags:
  - finetuning
  - huggingface
  - nvidia-tao
  - computer-vision
  - training
compatibility: Requires docker + nvidia-container-toolkit, NVIDIA GPU (driver ≥ 545, ≥ 24 GB VRAM for ≤3B models), ~40 GB free disk. Optional credentials (loaded from `~/.config/tao/.env` by the SessionStart hook) — HF_TOKEN is read only when the model/dataset is gated or `push_to_hub` is on; WANDB_API_KEY and WANDB_PROJECT only when WandB logging is enabled.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash Write
---
<!--
Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# tao-finetune-huggingface-model

Local NVIDIA GPU fine-tuning for HuggingFace models, grounded in live-fetched
documentation, with curated references as a fallback safety net. One NGC
container, a small set of focused scripts, one push to HF Hub. Behavior is
governed by the rules in this file — follow them, do not improvise around them.

**Order of authority (highest first):**

1. **User input** — explicit `model_id`, `dataset_id`, `training_method`, `config.yaml` overrides.
2. **Live research** — model card, HF repo example, author finetune script, HF task docs, paper. Always fetched. See Step 3 + `references/research-priorities.md`.
3. **Curated references** (`references/*.md`) — fallback when live research is silent or ambiguous.
4. **Your training-data memory** — last resort. Treat as suspect; cross-check against (2) or (3).

If (2) and (3) conflict on an API call, (2) wins (newer). If they conflict on a
method detail (collator, LoRA targets, augmentation), (2) wins for the *specific*
model; (3) for the generic shape. Note the discrepancy in a comment at the source
line.

---

## Inputs

**Required:**
- `model_id` — HuggingFace model ID, e.g. `google/vit-base-patch16-224`

**Conditional credentials (loaded by the SessionStart hook from `~/.config/tao/.env` when present):**
- `HF_TOKEN` — required only when the model or dataset is **gated** (read access) or `push_to_hub` is on (write access). Public model + public dataset + `push_to_hub: false` runs do not need it. The agent never reads the value — only checks presence with `[ -n "$HF_TOKEN" ]`.
- `WANDB_API_KEY`, `WANDB_PROJECT` — required only when WandB monitoring is enabled. Set `WANDB_MODE=disabled` to opt out.

**Dataset — exactly one:**
- `dataset_id` — HuggingFace dataset ID *(source: `hf`)*
- `local_dataset_path` — local folder or file *(source: `local`)*. Optionally
  `local_dataset_format` ∈ {auto, imagefolder, coco, voc, jsonl, arrow,
  parquet, csv}. Default: auto-detect.
- *(omit)* — agent recommends popular datasets *(source: `recommend`)*

**Optional (have defaults):**
- `task_type` — auto-detected from config + model card
- `n_train=10000`, `n_eval=1000`, `n_epochs=3`, `lora_r=16`
- `output_dir=./output/<model_short_name>`
- `hf_model_repo` — push target. If unset and HF_TOKEN has write access,
  auto-derived as `<whoami>/<model_short_name>-finetuned`.
- `push_to_hub=True` — set explicitly to `False` to skip
- `skip_baseline=False` — skip zero-shot baseline eval

**Optional deliverables (off by default):**
```yaml
emit_progress_log: false   # output_dir/PROGRESS.md (per-step ✅/⚠️/❌ journal)
emit_report:       false   # reports/report.{pdf,html} with curves & samples
emit_unit_tests:   false   # tests/ with fake-data heterogeneous-batch tests
```

All values live in `output_dir/config.yaml`. Never hardcode in Python.

---

## Execution platform

This skill orchestrates *what* to run; the platform skills own *how* to run it
on a GPU host. Read those skills first and do not redraft their conventions
here.

| Concern | Authoritative skill |
|---|---|
| GPU host runtime — NVIDIA driver 580, CUDA Toolkit 13.0, NVIDIA Container Toolkit 1.19.0 | [`tao-skill-bank:tao-setup-nvidia-gpu-host`](../../platform/tao-setup-nvidia-gpu-host/SKILL.md) |
| `docker run` flags, NGC auth, `--gpus`, mounts, env passthrough, `--ipc=host`/`--shm-size`, common error modes | [`tao-skill-bank:tao-run-on-docker`](../../platform/tao-run-on-docker/SKILL.md) |
| Local Docker job preflight (daemon reachable, GPU smoke) | [`tao-skill-bank:tao-run-on-local-docker`](../../platform/tao-run-on-local-docker/SKILL.md) |

**Default platform:** `local-docker`. This workflow builds a one-off image
(`run-<short>:latest`) and runs it on the local Docker daemon — the same
pattern documented in `skills/platform/tao-run-on-local-docker/SKILL.md`. Ask the user only when
they explicitly need a different backend (Brev for a remote GPU instance,
SLURM/Kubernetes for managed scheduling); in that case run the chosen
platform's Preflight section first, generate the choices via
`${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py
--format text`, then route the `docker run` commands in Steps 4–5 through that
platform's execution pattern.

**GPU runtime preflight:** Step 2a runs the `tao-setup-nvidia-gpu-host` skill's
`--check-only` mode. Do not duplicate the NCT / driver / `--gpus all` smoke
logic here — if it needs to change, change it in `tao-setup-nvidia-gpu-host`.

**Credentials preflight:** the SessionStart hook
(`hooks/session_start.sh`) loads `~/.config/tao/.env` into the session
env and lists the variable names (never values) in the session banner.
Step 2a only confirms presence of credentials that the current run
*actually* needs — `HF_TOKEN` for gated downloads or `push_to_hub`,
`WANDB_API_KEY`/`WANDB_PROJECT` if WandB is enabled — instead of hard-
requiring them up front.

**Docker run conventions:** every `docker run` invocation in
`references/docker-runs.md` follows the canonical flag set from
`skills/platform/tao-run-on-docker/SKILL.md` (`--gpus all`, `--ipc=host` or `--shm-size=…`,
`-e VAR` passthrough, bind mounts, `--rm` for one-shots). Treat that skill as
the spec; this one only adds workflow-specific flags
(`--entrypoint /bin/bash -lc`, `PYTORCH_CUDA_ALLOC_CONF`, `--name hft_train`).

---

## References — fallback safety net

Consulted **only** when live research is silent, ambiguous, or unavailable. Live
docs always win for the specific model and current API.

Always-on references: `core-rules.md`, `error-playbook.md`,
`compat-workarounds.md`, `model-discovery.md`, `dataset-recommendations.md`,
`dataset-sources.md`, `dataset-patterns.md`, `hardware-container.md`,
`research-priorities.md`, `cv-scripts.md`, `vlm-scripts.md`,
`docker-runs.md`, `hub-push.md`, `pipeline-skill-template.md`, and
`deliverables.md`.

Opt-in references: `progress-tracking.md` for `emit_progress_log: true`, `testing.md` for `emit_unit_tests: true`, `reporting.md` for `emit_report: true`,
`workflow-intake-preflight.md` for probes/preflight, `workflow-generate-train.md` for generation/training, and `workflow-push-rerun.md` for Hub/rerun details. `detailed-workflow.md` is only the map.

**Rule:** before falling back to a reference, log the live source you tried and
why it was insufficient (in `config.yaml` `notes:`, and PROGRESS.md if enabled).

**`[FETCH LIVE]` markers in `cv-scripts.md` / `vlm-scripts.md`** are a
research checklist, not code to inline. If a `[FETCH LIVE]` block has no Step 3
finding, refetch the listed URL.

---

## Core rules

The non-negotiable behaviors the agent must follow across the workflow. Full
text in `references/core-rules.md`. **Short version:**

- **Your HF-library knowledge is outdated.** Fetch live docs (model card, HF
  repo example, task doc) before writing any ML code. Don't generate trainer
  args / collator / transforms from memory — see Step 3.
- **Smoke-test on real data with `--max_steps 1`** before any full run. No
  batch launches without a verified smoke.
- **Never silently substitute** model_id, dataset_id, or training_method. If
  what the user asked for doesn't load, stop and ask.
- **Error recovery is minimal-change.** OOM → halve batch, double grad_accum,
  enable gradient checkpointing — don't switch to LoRA without approval. NaN
  → reduce LR 10×. Flat loss → inspect collator. Same error 3× → stop and
  ask. Don't loop.
- **Dataset columns verified BEFORE writing the collator.** Mismatch +
  rename → fix in `prepare_data.py`; restructuring needed → stop and ask.
- **Hardware-sizing rule of thumb (bf16):** ≤3B → 24 GB, 7–13B → 80 GB, 30B+ →
  multi-GPU or LoRA on 1× 80 GB, 70B+ → 8× 80 GB or LoRA. If a full finetune
  won't fit and the user didn't ask for LoRA, ask before switching.

Consult `references/core-rules.md` for the full enumeration (hallucinated
imports list, never-without-approval list, full error-recovery table, full
hardware sizing table) before training-time decisions.

---

## Workflow — 6 steps

Single pass, sequential. Each step has a clear gate before the next begins.

### Step 1 — Inspect & qualify

**Goal:** decide whether to proceed at all. Probe model, probe dataset, apply
accept/reject, register applicable compat fixes, write the initial `config.yaml`.

Prerequisites: `MODEL_ID`, optional `DATASET_ID` / `local_dataset_path`,
optional `HF_TOKEN`, and `OUTPUT_DIR` (default `./output/<model_short_name>`).
Step 1 probes run in a small CPU-only Docker container so the host does not need
a Python virtualenv. Docker itself must exist before probing:

```bash
TAO_SKILL_BANK_ROOT="${TAO_SKILL_BANK_PATH:-${TAO_SKILL_BANK_ROOT:-$PWD}}"
SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

if ! command -v docker >/dev/null 2>&1; then
  echo "MISSING: docker is required for Step 1's containerized probe."
  echo "After user approval, run the platform installer (same one Step 2a uses):"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install --yes"
  echo "Then re-source your shell or 'newgrp docker' so the new group membership applies."
  exit 1
fi
```

Run the model probe and dataset probe with the containerized pattern in
`references/model-discovery.md` and `references/dataset-sources.md`: normalize
`OUTPUT_DIR` to an absolute path, write probe scripts under
`$OUTPUT_DIR/.probe/`, run `python:3.12-slim` with `HOME=/probe`,
`PIP_USER=1`, `HF_HOME=/probe/.cache`, and a bind-mounted scratch directory.

Probe requirements:

- Model: load `AutoConfig`, read model-card tags, detect task from
  `architectures` + tags + card examples, and log any fallback to
  `references/model-discovery.md`.
- Dataset: for recommended datasets, first present 3-5 choices from
  `references/dataset-recommendations.md`; for local data, bind-mount the path
  read-only and use `references/dataset-sources.md` format detection.
- Reject early if the model config fails, the task is out of scope, no recipe
  source exists, or the dataset cannot load / match the task schema.
- Evaluate `references/compat-workarounds.md` against the detected model/task;
  defer hardware-dependent rules until Step 2.

Write the initial `config.yaml`:

```yaml
model_id: <…>
task: <…>
dataset_id: <…>             # or local_dataset_path
research_sources: []         # filled in Step 3
applicable_workarounds: [<…>]
notes: []                    # log any reference fallback
push_to_hub: true            # default
```

Optionally clean up the probe scratch dir once the gate is met:

```bash
rm -rf "$OUTPUT_DIR/.probe"
```

**Gate:** `config.yaml` exists with model, dataset, task, applicable_workarounds.
Do not proceed if any field is missing.

---

### Step 2 — Hardware audit & NGC image

**Goal:** verify Docker + GPU + disk, pick the NGC PyTorch image live, finalize
hardware-dependent compat rules.

**2a. Audit (hard gate):** the GPU host runtime check is owned by the
`tao-setup-nvidia-gpu-host` skill (driver branch 580, CUDA Toolkit 13.0, NVIDIA
Container Toolkit 1.19.0). Invoke it in `--check-only` mode; on failure, ask
the user to authorize the install, then re-run. Credentials come from the
SessionStart hook (`~/.config/tao/.env`) — only check the ones the current
run actually needs.

```bash
# 1) GPU host runtime — delegated to tao-setup-nvidia-gpu-host
TAO_SKILL_BANK_ROOT="${TAO_SKILL_BANK_PATH:-${TAO_SKILL_BANK_ROOT:-$PWD}}"
SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

bash "$SETUP_SCRIPT" --backend docker --check-only || {
  echo "MISSING: TAO GPU host runtime not ready."
  echo "After user approval, run: bash \"$SETUP_SCRIPT\" --backend docker --install --yes"
  exit 1
}

# 2) Free-disk soft-warn (override via MIN_DISK_GB; default 100 GB)
min_disk_gb="${MIN_DISK_GB:-100}"
disk_free_gb=$(df -BG / | awk 'NR==2 {print $4}' | tr -d G)
if [ "${disk_free_gb:-0}" -lt "$min_disk_gb" ]; then
  echo "WARN: only ${disk_free_gb}G free on /; recommend ≥ ${min_disk_gb}G for NGC base (~20G) + HF cache + checkpoints + dataset." >&2
fi

# 3) Conditional credential presence checks (no values are read)
#    HF_TOKEN: only when the model/dataset is gated, or push_to_hub is on.
#    WANDB_*:  only when WandB logging is enabled in config.yaml.
```

**Do not proceed to Step 4 on a hard-fail** — Step 4's `docker build` pulls a
20+ GB NGC base image, and a missing `nvidia-container-toolkit` only surfaces
at `prepare_data.py` time as the cryptic `could not select device driver ""
with capabilities: [[gpu]]`.

Record `gpu_count`, `gpu_name`, `driver_major`, `vram_gb_per_gpu` in
`config.yaml`.

**2b. Pick NGC image (live):**

Open the NVIDIA Deep Learning Frameworks support matrix and inspect the PyTorch
NGC container section: <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>.

Find the **PyTorch NGC container** section. Pick the highest-versioned image
where:
- `Min driver ≤ detected driver_major`
- Container CUDA is `≤` host CUDA Toolkit version (drivers are forward-
  compatible, but match closely so cuDNN / TensorRT versions line up with
  the host toolchain).

Do **not** reject an image because the PyTorch version carries `aN`, `bN`, or
`rcN`; NGC validates the full image. Pick the newest CUDA-aligned image and let
`compat-workarounds.md` handle real per-version issues.

If the support matrix cannot be reached, use the fallback rules in `references/hardware-container.md`. Default
fallback: `nvcr.io/nvidia/pytorch:24.09-py3` (driver ≥ 545; SDPA+GQA bug — if
the model has `num_key_value_heads < num_attention_heads`, set
`attn_implementation: "eager"` in config).

Record `ngc_image` in `config.yaml`.

**2c. Re-evaluate hardware-dependent compat rules:**

Re-run the `compat-workarounds.md` walk for entries whose `detect` expression
needs `hw`. Update `applicable_workarounds:` in place.

**2d. Model-fit check:** estimate `param_bytes ≈ 2×param_count` (bf16). If
> 60% of `vram_gb_per_gpu × 1e9`, recommend LoRA in the user-facing summary.

**Gate:** `config.yaml` has `ngc_image`, `gpu_count`, `gpu_name`, `driver_major`,
`vram_gb_per_gpu`. Hardware-dependent compat fixes are recorded.

---

### Step 3 — Research the recipe

**Goal:** fetch the live recipe. The agent's training-data knowledge of
`transformers`/`trl`/`peft` is treated as suspect — Step 3 is non-negotiable.

Walk `references/research-priorities.md` in priority order (Priority 1 → 6).
Stop once you have, for the detected task:

- `AutoModel` / processor class
- Train + eval transforms
- Collator
- `compute_metrics`
- Hyperparameter hints (LR, batch size, epochs, scheduler)

Record findings in `meta/recipe.md` and append source URLs to
`config.yaml: research_sources:`. If a slot has no live finding, fall back to
the matching scaffold reference (`cv-scripts.md` / `vlm-scripts.md`) and log
"fallback to scaffold — no live source for <slot>" under `notes:`.

**Conflict resolution rules** are in `references/research-priorities.md`.

**Gate:** every required slot above is filled, with a source URL or an explicit
scaffold-fallback note.

---

### Step 4 — Generate project & smoke-test

**Goal:** write all scripts, build the image, prepare data, run a 1-step smoke
on real data. One `docker build`, two `docker run`s.

**4a. Generate project files** in `output_dir/`:

Generate `config.yaml`, `Dockerfile`, `requirements.txt`, `prepare_data.py`,
`train.py`, `run_eval.py`, `infer.py`, optional `merge_lora.py`, optional
`tests/`, and `.gitignore`. Use live Step 3 research as authority; use
`references/cv-scripts.md` or `references/vlm-scripts.md` only for scaffold
shape. Apply every `applicable_workarounds` entry as a Dockerfile block,
requirement pin, config override, or runtime env var.

`run_eval.py` must keep that exact filename to avoid colliding with the HF
`evaluate` package. Every generated `.py` file must start with the NVIDIA
Apache-2.0 copyright header; if you emit files from a script, make the emitter
fail when the header is missing. If `emit_unit_tests: true`, generate and run
tests using `references/testing.md`.

Dockerfile shape: `ARG NGC_IMAGE`, `FROM ${NGC_IMAGE}`, `ENTRYPOINT
["/bin/bash", "-c"]`, `WORKDIR /workspace`, install `requirements.txt`, inject
compat blocks/env vars, then copy generated Python files and `config.yaml`.

**4b. Build, prepare, smoke** (commands: `references/docker-runs.md` §1-3):

```bash
docker build -t run-<short>:latest .
# → docker-runs.md §2: prepare_data
# → docker-runs.md §3: smoke (--smoke --max_steps 1)
```

Smoke pass criteria (in `logs/smoke.log`):
- No exception
- Loss is finite (not `0.0`, not `NaN`)
- `grad_norm > 0` at step 1

If `emit_unit_tests: true`, also run `pytest tests/` inside the container.
Failure → STOP. Do not proceed.

**4c. Preflight summary** — print and verify reference URL, dataset columns,
Hub target, monitoring target, NGC image, hardware, and smoke loss/grad norm
before full training.

**Gate:** project files written, image built, smoke PASSED, preflight has no
blank fields.

---

### Step 5 — Train, evaluate, infer

**Goal:** baseline eval, full training, post-train eval, optional LoRA merge,
5 inference samples. All commands: `references/docker-runs.md` §4-8.

| Sub-step | docker-runs.md | Skip if |
|---|---|---|
| 5a. Baseline eval (zero-shot) | §4 | `skip_baseline: true` |
| 5b. Full training (detached) | §5 | — |
| 5c. LoRA merge | §6 | not VLM-with-LoRA |
| 5d. Post-train eval | §7 | — |
| 5e. Inference (5 samples) | §8 | — |

Multi-GPU: prepend `torchrun --nproc_per_node=$gpu_count` to `python train.py`.

While training streams, watch `docker logs -f hft_train` for:
- Loss drops within 10-20 steps → working
- Flat loss → collator / label-masking bug; stop
- NaN loss → LR too high; stop, reduce LR, retry
- OOM → halve batch, double grad_accum, enable gradient checkpointing

If `emit_report: true`, run `report.py` after Step 5e per `references/reporting.md`.

**Gate:** all of:
- `checkpoints/final/` (or `checkpoints/merged/` for LoRA) exists
- `reports/eval_results.json` has a numeric primary metric
- `reports/baseline_results.json` exists (unless skipped)
- `reports/inference_samples/` has 5 samples
- wandb URL shows descending loss

---

### Step 6 — Push & emit rerun skill

**Goal:** publish the run and ensure it can be reproduced without re-research.

Use `references/hub-push.md` unless `push_to_hub: false` is explicit. Push
weights, model card, eval/baseline JSONs, `config.yaml`, `Dockerfile`,
`requirements.txt`, inference samples, and reports when emitted.

Emit `<output_dir>/skills/run-<short>/SKILL.md` from
`references/pipeline-skill-template.md`. Substitute every placeholder, include
full YAML metadata and the NVIDIA copyright HTML comment, and make any emitter
script fail if those requirements are missing.

**Gate (Done criteria):** all of:
- Step 5 gate met
- HF Hub repo exists at the resolved URL with weights + card + `results/`
  (unless `push_to_hub: false`)
- `<output_dir>/skills/run-<short>/SKILL.md` exists, no `<placeholder>` left,
  with metadata + copyright HTML comment per `pipeline-skill-template.md`

Final message: wandb URL, HF Hub URL, baseline -> fine-tuned primary metric,
`reports/inference_samples/`, and the emitted rerun skill path.

---

## Error playbook

When you hit a known runtime error, consult `references/error-playbook.md`
before redesigning anything. It carries the symptom -> minimal-fix table for
NGC entrypoint issues, PyTorch/Transformers regressions, numpy ABI breaks,
Albumentations bbox errors, PEFT/checkpointing, LoRA target breadth, CV
augmentation gaps, and OOM at step 0.

When a row in that table fires twice across runs, lift it into
`compat-workarounds.md` with a `detect` rule — that registry is the
durable form, auto-applied in Step 1d before the error has a chance to fire.

---

## Communication style

- Terse. No filler, no restating the request. One-word answers when appropriate.
- Always include direct Hub and wandb URLs when referencing artifacts.
- On error: state what went wrong, why, what you changed. No menus.
- Never present "Option A/B/C" for a request that has a clear answer. Act.
