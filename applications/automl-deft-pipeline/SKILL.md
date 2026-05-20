---
name: automl-deft-pipeline
description: >
  Run the canonical NVIDIA AOI three-phase training pipeline — Phase 1 AutoML baseline (HPO),
  Phase 2 DEFT loop (RCA → SDG → mining → plain-train retrain), Phase 3 AutoML refinement on
  the DEFT-augmented dataset. This is the default entry point for any "run the AOI workflow",
  "fine-tune my PCB AOI model end-to-end", "improve my AOI ChangeNet model", or "AOI workflow
  with AutoML" request — route here instead of workflow-deft-aoi-loop directly unless the user
  explicitly asks for the DEFT loop ONLY (e.g. "run JUST the DEFT loop", "skip AutoML, only
  DEFT"). Also handles the same three-phase pattern for non-AOI DEFT applications — AutoML
  baseline then DEFT loop warm-started from AutoML's winning HPs then post-DEFT AutoML
  refinement on the iteration-augmented dataset. Trigger phrases include "run the AOI
  workflow", "AOI end-to-end", "AutoML + DEFT", "AutoML then DEFT", "tune hyperparameters then
  DEFT", "DEFT with AutoML at both ends", "warm-start DEFT", "improve my AOI model".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit. Sub-skills (tao-automl, workflow-deft-aoi-loop) declare additional requirements.
metadata:
  author: NVIDIA Corporation
  version: "0.2"
allowed-tools: Read Bash Write Skill
---

# AutoML + DEFT Pipeline

A workflow-bridge skill that runs **three phases** in sequence by delegating to two existing skills — `tao-automl` for HPO and a DEFT application skill (default `workflow-deft-aoi-loop` for AOI; other `applications/deft-*` skills for non-AOI cases) for the iterative data-improvement loop.

This skill **does not** re-implement AutoML or DEFT. It owns only the connective tissue: HPO spec inputs, the spec-handoff between AutoML and DEFT, and the post-DEFT AutoML re-run on the augmented dataset.

## When this skill applies

- User asks to "run the AOI workflow" or "improve my AOI ChangeNet model" — **default to this skill**, not `workflow-deft-aoi-loop` directly. The bare DEFT loop is the inner stage of this pipeline.
- User wants AutoML and DEFT chained on the same model/dataset
- User says "AutoML at both ends", "tune HPs then DEFT", "warm-start DEFT", "AutoML before and after DEFT"
- User has an AutoML-tuned spec and asks how to feed it into DEFT

## When this skill does NOT apply

- User explicitly asks for the DEFT loop only ("run JUST the DEFT loop", "skip AutoML") → use `workflow-deft-aoi-loop` directly
- User wants only AutoML with no follow-on DEFT → use `tao-automl` directly
- User is doing zero-shot eval, RAG, or non-training workflows

---

## The mental model

```
Phase 1 (AutoML baseline)        Phase 2 (DEFT loop, plain train)        Phase 3 (AutoML refinement)
─────────────────────────        ────────────────────────────────        ───────────────────────────
specs/baseline_spec.yaml         specs/baseline_spec.yaml                ${RESULTS_DIR}/iter${N}/dataset/
train/base/training_set.csv         (patched with Phase 1's winning HPs) train_combined_iter${N}.csv
        │                                       │                                       │
        ▼                                       ▼                                       ▼
[ AutoML HPO sweep ]               [ DEFT: baseline → iter 1..N ]          [ AutoML HPO sweep ]
   N recommendations                  RCA / route / SDG / mining            re-tunes HPs against the
   pick best by val_loss / FAR        plain-train retrain each iter         DEFT-augmented dataset
        │                                       │                                       │
        ▼                                       ▼                                       ▼
best HPs spec ──────────►        DEFT-augmented CSV   ──────────►        final best checkpoint
                                 + last-iter checkpoint                  (the deliverable)
```

The handoffs are:

- **Phase 1 → Phase 2**: a *spec file* (the AutoML-winning hyperparameters) → DEFT's `specs/baseline_spec.yaml` is replaced/patched with this. DEFT itself stays plain-train (`automl_policy: off` inside the DEFT loop is preserved).
- **Phase 2 → Phase 3**: a *training CSV* (`train_combined_iter${N_final}.csv`) → fed to AutoML for the refinement sweep. Phase 3's winning checkpoint is the pipeline's deliverable.

## Why three phases instead of two

- **Phase 1 alone** finds good HPs on the *original* training distribution, but the model still has the distributional gaps DEFT is designed to fill.
- **Phase 2 alone** (just DEFT) fills the gaps but uses whatever HPs `specs/baseline_spec.yaml` was hand-authored with — usually not optimal.
- **Phase 3 alone** would run AutoML against the augmented dataset, but without a tuned baseline the DEFT loop's iteration cost is higher (slower convergence, more iterations to hit the KPI).

Running all three: AutoML cheap-tunes once on the original data, DEFT does the heavy data work with reasonable HPs, then AutoML tunes again on the now-richer dataset. Phase 3 is the most important of the three for the final deployed FAR/recall.

## Cost up-front

The pipeline is sequential. Total wall-clock ≈ Phase 1 (N_automl × per-rec train) + Phase 2 (1 baseline train + M iterations × per-iter cost) + Phase 3 (N_automl × per-rec train). Surface this to the user before kickoff — typically Phase 2 dominates because it includes SDG + retrain per iteration, but Phase 1 and Phase 3 each add several hours on a single-GPU box. Use the per-job estimate from the user's setup (if they have one) rather than guessing minutes.

---

## Phase 1 — AutoML baseline

Invoke `tao-skill-bank:tao-automl` with:

| Input | AOI default | Notes |
|---|---|---|
| `network_arch` | `visual-changenet` | Same model the DEFT loop expects |
| `train_dataset_uri` | `<workspace>/train/base/training_set.csv` | Same training set DEFT will start from |
| `eval_dataset_uri` | `<workspace>/train/base/validation_set.csv` | Held-out — must NOT be the KPI test set (`<workspace>/kpi/testing_set.csv`), since that set is reserved for DEFT's final reporting |
| `metric` | FAR @ 100% recall (preferred) or `val_loss` | See **Metric pitfalls** below — ChangeNet AOI is class-imbalanced, val_loss alone can mode-collapse |
| `algorithm` | `bayesian` | LLM-brain or `autoresearch` if compute is tight |
| `automl_max_recommendations` | 5–10 for AOI | More recs = better HPs but linear in compute |
| `spec_overrides` | Pin epochs / batch_size; sweep optimizer-related HPs only | Otherwise AutoML wanders into long-train regimes that blow Phase 2's budget |

After the sweep finishes, AutoML's `result["best"]["specs"]` is the winning hyperparameter dict.

### Handoff to Phase 2

Write the winning spec to `<workspace>/specs/baseline_spec_automl.yaml` by deep-merging `result["best"]["specs"]` onto `<workspace>/specs/baseline_spec.yaml` (i.e. preserve dataset paths, model architecture, lighting layout; overwrite only the HPs AutoML tuned). Then tell the DEFT loop to use that file:

```bash
# When invoking workflow-deft-aoi-loop, pass:
#   specs/baseline_spec.yaml = <workspace>/specs/baseline_spec_automl.yaml
# OR: copy the merged spec onto the path the DEFT loop reads by default
cp <workspace>/specs/baseline_spec_automl.yaml <workspace>/specs/baseline_spec.yaml.deft_input
```

The DEFT loop itself stays unmodified — `automl_policy: off` inside the loop is preserved. Phase 1's contribution is one file: the spec.

### Quality check before handing off

Run a quick eval of the winning checkpoint against the held-out set:

- Per-class prediction counts — if it collapsed to one class, the winning HPs are useless for Phase 2. Evaluate the 2nd or 3rd best instead.
- Compare to a zero-shot ChangeNet baseline. If AutoML did not improve over zero-shot, surface that to the user and pause before continuing.

---

## Phase 2 — DEFT loop (plain training)

Invoke `tao-skill-bank:workflow-deft-aoi-loop` (read its `SKILL.md` for the full interface). For non-AOI applications, invoke the matching DEFT skill; the handoff shape is the same.

The DEFT loop runs exactly as documented in its own SKILL.md — Pre-Flight → Baseline → iterations → loop-end report. **Do not modify its `automl_policy: off` invariant.** The only difference vs. running DEFT standalone is that `specs/baseline_spec.yaml` now carries Phase 1's tuned HPs.

The DEFT loop owns:

- The user gate (Pre-Flight Summary + approval)
- The full RCA → routing → SDG → mining → assemble → train cycle
- KPI gating and stop conditions
- `${RESULTS_DIR}/` layout, `deft_state.json`, `loop_log.jsonl`, `DEFT_Loop_Report.html`

After the loop exits (KPI met or `max_iterations` reached), capture two values from `deft_state.json`:

- `iterations.<best>.best_ckpt_path` — the loop's best plain-train checkpoint
- The final iteration label `N_final` — used to locate the augmented training CSV

If the DEFT loop hard-stops on an unrecoverable gate, **skip Phase 3**. There is no validated augmented CSV to feed AutoML.

---

## Phase 3 — AutoML refinement on the DEFT-augmented dataset

Re-invoke `tao-skill-bank:tao-automl` with the augmented training CSV as the train dataset and the same held-out validation CSV as before:

| Input | AOI value |
|---|---|
| `network_arch` | `visual-changenet` |
| `train_dataset_uri` | `${RESULTS_DIR}/iter${N_final}/dataset/train_combined_iter${N_final}.csv` |
| `eval_dataset_uri` | Same as Phase 1 (`<workspace>/train/base/validation_set.csv`) — keep the comparison apples-to-apples |
| `metric` | Same metric as Phase 1 |
| `algorithm` | Same as Phase 1 |
| `automl_max_recommendations` | 5–10 |
| Initial spec | Start from `<workspace>/specs/baseline_spec_automl.yaml` (Phase 1's winner) — gives the sweep a strong centroid to refine around |

Output goes to `${RESULTS_DIR}/final_automl/`. The winning checkpoint of this sweep is the pipeline's deliverable.

### Wiring Phase 3's output back into the DEFT report

`workflow-deft-aoi-loop`'s `scripts/prepare_inference_spec.py` selects the lowest-`far_pct` entry from `deft_state.json["iterations"]`. To make Phase 3's checkpoint visible to the handoff:

1. Append an entry to `${RESULTS_DIR}/deft_state.json` under `iterations.final_automl` with the same shape as iteration entries (`best_ckpt_path`, `threshold`, `far_pct`) — populate from Phase 3's eval output.
2. Re-run `python ${TAO_SKILL_BANK_PATH}/applications/workflow-deft-aoi-loop/scripts/prepare_inference_spec.py --results-dir ${RESULTS_DIR}`. The script's `_pick_best` will now see the Phase 3 entry and select it on `far_pct` (or fall back to the loop's best if Phase 3 regressed — see safety note below).

**Safety note.** Phase 3 is not guaranteed to beat the loop's best iteration — AutoML can over-fit a small augmented dataset. The `_pick_best` lowest-`far_pct` tie-break protects against this: if Phase 3's checkpoint is worse, the iteration winner is still selected. Surface both numbers to the user in the final summary so the regression is visible.

---

## Pitfalls and quality checks

These apply to both AutoML phases. Bake them into agent behavior — don't just paste once.

### Metric pitfalls — AOI is class-imbalanced

ChangeNet AOI datasets are typically PASS-dominant (90%+ PASS rate). `val_loss` (cross-entropy) on imbalanced data has a well-known failure mode: the model can minimize CE by confidently predicting PASS for everything, achieving very low val_loss while having zero recall on defects. The val_loss winner of an AutoML sweep can be a mode-collapsed model.

For AOI, prefer:

- **FAR @ 100%-recall** as the AutoML metric directly (matches the deployment KPI; never collapses)
- Or run val_loss with a **`pred_counts` sanity check**: discard any rec whose predictions collapse to one class
- Or eval all top-K configs by FAR @ 100%-recall on the held-out set before picking — val_loss is the sort key, FAR @ 100%-recall is the decision rule

For balanced datasets and regression tasks (non-AOI DEFT applications), val_loss is fine.

### Run-to-run noise

AutoML can show 2–3× variance in metric for the same HP config across runs (seeds, dataloader shuffles). If the AutoML winner is suspiciously better than the runner-up, re-run with a fresh seed and confirm the metric holds before committing the spec to Phase 2.

### Cleanliness (data leakage)

Both AutoML phases must use a validation set distinct from the KPI test set (`<workspace>/kpi/testing_set.csv`). The KPI test set is reserved for DEFT's final reporting — touching it during AutoML biases the final number upward. The standard split: `train/base/training_set.csv` for AutoML training, `train/base/validation_set.csv` for AutoML val, `kpi/testing_set.csv` left alone until DEFT's evaluate stage.

Phase 3's train_dataset is the DEFT-augmented CSV, which contains synthetic + mined real samples beyond the base training set. The validation set stays the same — that keeps Phase 1 and Phase 3 metric numbers comparable.

### Compute budget

Total cost is roughly:
- Phase 1: `N_automl × per-rec train`
- Phase 2: `1 × baseline train + M_iter × (RCA + SDG + mining + retrain)` — usually the largest term because SDG generates synthetic images
- Phase 3: `N_automl × per-rec train` on the (larger) augmented dataset, so per-rec time is somewhat higher than Phase 1

Surface the structure to the user up front. Ask them for their per-job time and give a wall-clock range only after that — don't make up minute numbers.

---

## Quick Start (AOI worked example)

This is what the agent says to the user when starting fresh from "run the AOI workflow":

> I'll run the canonical AOI training pipeline in three phases:
>
> **Phase 1 — AutoML baseline.** I'll sweep `<N>` configs over `<HP list>` against `<workspace>/train/base/validation_set.csv` using `bayesian` with FAR @ 100%-recall as the metric (AOI is class-imbalanced, val_loss alone risks mode collapse). After it finishes I'll spot-check per-class prediction counts before declaring a winner. The winning spec is saved to `specs/baseline_spec_automl.yaml` and feeds Phase 2.
>
> **Phase 2 — DEFT loop.** Plain training inside the loop (`automl_policy: off` preserved), but starting from Phase 1's tuned HPs. The loop runs Pre-Flight → Baseline → iterations until the KPI target is met or `max_iterations` is reached. I'll keep its built-in user gate at Pre-Flight Summary intact.
>
> **Phase 3 — AutoML refinement.** Final AutoML sweep on the DEFT-augmented CSV (`train_combined_iter${N_final}.csv`). The winning checkpoint of this sweep is the deliverable. I'll register it under `state.iterations.final_automl` and re-run `prepare_inference_spec.py` so `best_model.json` and `best_model_inference_spec.yaml` point to it — unless Phase 3 regresses, in which case the loop's best iteration wins on the same metric.
>
> Total cost is `<N_automl>` AutoML training jobs × 2 sweeps + `<M_iter>` DEFT iterations (each with SDG + retrain). If you can tell me roughly how long one ChangeNet training run takes on your hardware I can give you a wall-clock estimate. OK to proceed?

After confirmation, invoke `tao-skill-bank:tao-automl`, write the merged spec, invoke `tao-skill-bank:workflow-deft-aoi-loop` (which has its own Pre-Flight gate — both gates are user-facing), then `tao-skill-bank:tao-automl` again. Summarize the trajectory: baseline AutoML best → DEFT iter 1 → ... → DEFT iter N_final → Phase 3 best, so the user sees where the gains came from.

## Non-AOI DEFT applications

Same three-phase pattern applies to other DEFT skills. Swap:

- `network_arch` to the relevant model
- The DEFT skill invoked in Phase 2
- The "best HP spec file" path convention to whatever the target DEFT skill expects
- The augmented-CSV path in Phase 3 to whatever the target DEFT skill produces

The handoff shape — Phase 1 emits a spec, Phase 2 consumes it and emits an augmented dataset, Phase 3 emits the final checkpoint — is identical.

---

## See also

- `tao-skill-bank:tao-automl` — AutoML interface, algorithms, HP ranges
- `tao-skill-bank:workflow-deft-aoi-loop` — full DEFT AOI loop (Phase 2 default)
- `tao-skill-bank:visual-changenet` — underlying ChangeNet train/eval/infer skill (used by both AutoML and DEFT)
- Other `applications/deft-*` skills — non-AOI Phase 2 targets
