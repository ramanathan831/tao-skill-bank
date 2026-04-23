---
name: workflow-deft-aoi-loop
description: >-
  Top-level DEFT Loop orchestrator — runs the full Evaluate → RCCA → SDG →
  Retrain → Deploy loop end-to-end for NVIDIA PCB AOI ChangeNet models.
  Use when the user wants to iteratively improve a model to meet a KPI target.
---

# Skill: workflow-deft-aoi-loop

**Trigger:** Invoke this skill when the user wants to iteratively improve a model to meet a KPI target, run the DEFT loop, or says something like:
- "I want my model to be performant at KPI: FAR < 0.1% at recall=100%"
- "Run the DEFT loop on my model"
- "Fine-tune until KPI is met"

---

## Sub-Skills Required

This orchestrator calls the following sub-skills. All must be installed before running:

- `deft-aoi-deft-aoi-rca-changenet` — root cause analysis on inference results
- `deft-aoi-deft-aoi-anomalygen-inference` — diffusion-based defect inpainting (Arm A)
- `deft-aoi-deft-aoi-omniverse-sdg` — physics-based ray-traced SDG (Arm B)
- `deft-aoi-deft-aoi-data-mining` — k-NN retrieval from pre-generated AnomalyGen pool (Arm C)
- `deft-aoi-brev` — workspace and dataset staging on Brev (optional)

If any sub-skill is missing, stop and ask the user to install the `workflow-deft-aoi-loop` plugin (which bundles all of the above).

---

## Agent Behavior

> **This is a fully autonomous skill.** After collecting inputs, run the entire loop
> without asking for confirmation. Do not pause between steps. Do not ask "want me to
> continue?" — just continue. Only stop if a step fails with an unrecoverable error.
> Print status updates at each step milestone so the user can follow progress.

---

## State Management (crash recovery + context persistence)

The DEFT loop runs for hours. Sessions can crash, API connections can drop, and
context compaction can discard earlier iteration details. Two mechanisms handle this:

### 1. State File: `${RESULTS_DIR}/deft_state.json`

**Read on startup. Write after every step completes.** This is the crash recovery
checkpoint — if the session dies and restarts, the agent reads this file and
resumes from the last completed step.

**Schema:**
```json
{
  "version": 1,
  "started_at": "<ISO timestamp>",
  "kpi_target": "<user's KPI string>",
  "results_dir": "<path>",
  "max_iterations": 3,
  "current_iteration": 1,
  "current_step": "augmentation",
  "config": {
    "training_csv": "<path>",
    "validation_csv": "<path>",
    "images_dir": "<path>",
    "specs_file": "<path>",
    "backbone_weight_dir": "<path>",
    "train_container": "<image>",
    "num_gpus": 1,
    "batch_size": 16,
    "num_epochs": 200,
    "data_prep_script": "<path>",
    "analyze_script": "<path>",
    "arm_a_config": { "enabled": true, "...": "..." },
    "arm_b_config": { "enabled": true, "...": "..." },
    "arm_c_config": { "enabled": true, "...": "..." }
  },
  "iterations": {
    "baseline": {
      "status": "complete",
      "far": 0.52,
      "recall": 100,
      "threshold": 0.31,
      "checkpoint": "<path>",
      "training_rows": 1234,
      "rca_report": "<path to RCA_Report.md>"
    },
    "iter1": {
      "status": "in_progress",
      "completed_step": "rca",
      "far": null,
      "recall": null,
      "threshold": null,
      "checkpoint": null,
      "sdg_images_added": null,
      "mined_images_added": null,
      "training_rows": null,
      "rca_report": "<path>",
      "rca_target_defects": ["shift", "tombstone"],
      "arms_enabled": {"2a": true, "2b": true, "2c": false}
    }
  }
}
```

**Checkpoint rules:**
- Write the state file with `jq` or Python (not manual echo) to guarantee valid JSON
- Update `current_iteration`, `current_step`, and the relevant `iterations.<iter>`
  entry after each step completes
- Checkpoints happen at these points:
  - After Phase 1 (inputs collected) → `current_step: "validated"`
  - After Step 0 baseline train → `iterations.baseline.status: "trained"`
  - After Step 0 baseline eval → `iterations.baseline.status: "evaluated"` + metrics
  - After Step 0 baseline RCA → `iterations.baseline.status: "complete"` + rca_report path
  - After Step 1 RCA → `iterations.iterN.completed_step: "rca"`
  - After Step 2 augmentation → `iterations.iterN.completed_step: "augmentation"` + image counts
  - After Step 3 data prep → `iterations.iterN.completed_step: "data_prep"` + training_rows
  - After Step 4 training → `iterations.iterN.completed_step: "training"` + checkpoint path
  - After Step 5 evaluation → `iterations.iterN.status: "complete"` + metrics

**Write example (use after each step):**
```bash
python3 -c "
import json, sys
with open('${RESULTS_DIR}/deft_state.json', 'r') as f:
    state = json.load(f)
state['current_iteration'] = ${ITER}
state['current_step'] = 'training'
state['iterations']['iter${ITER}']['completed_step'] = 'data_prep'
state['iterations']['iter${ITER}']['training_rows'] = ${ROW_COUNT}
with open('${RESULTS_DIR}/deft_state.json', 'w') as f:
    json.dump(state, f, indent=2)
"
```

### 2. Iteration Summaries: `${RESULTS_DIR}/iterN_summary.md`

**Write after each iteration completes (Step 5).** These are compact digests
(~300 words max) that capture the essential findings from that iteration.
After context compaction, the agent **must re-read these files** instead of
relying on memory of earlier iterations.

**Template:**
```markdown
# Iteration N Summary

## Metrics
- FAR @ 100% Recall: X% (delta: +/-Y% from previous)
- Threshold: T
- Checkpoint: <path>

## RCA Key Findings
- Top failure modes: <ranked list with counts>
- Root causes: <1-2 sentence each>
- Actionable vs non-actionable: <which gaps SDG/mining can address>

## Augmentation Applied
- Arm 2A (AnomalyGen): <N images, defect types: [...]> / SKIPPED
- Arm 2B (Omniverse SDG): <N images, defect types: [...]> / SKIPPED
- Arm 2C (Data Mining): <N images mined> / SKIPPED
- Total training rows: N (base: X + augmented: Y)

## What Changed
- <1-2 sentences: what improved and what didn't compared to previous>

## Next Iteration Guidance
- <what the RCA suggests targeting next, if loop continues>
```

Save to: `${RESULTS_DIR}/iter${ITER}_summary.md` (or `baseline_summary.md`)

---

## Phase 0 — Resume Detection (run before anything else)

**Before collecting inputs, check for an existing state file:**

```bash
test -f "${RESULTS_DIR}/deft_state.json" && echo "STATE_FILE_FOUND" || echo "FRESH_START"
```

If the user provides a results directory and it contains `deft_state.json`:

1. Read the state file:
   ```bash
   cat ${RESULTS_DIR}/deft_state.json
   ```

2. Read all existing iteration summaries:
   ```bash
   cat ${RESULTS_DIR}/*_summary.md 2>/dev/null
   ```

3. Report what was found:
   ```
   [Resume] Found existing DEFT loop state:
     - Started: <timestamp>
     - KPI target: <target>
     - Current iteration: N, step: <step>
     - Completed iterations: baseline (FAR=X%), iter1 (FAR=Y%), ...
     - Resuming from: iteration N, step <step>
   ```

4. **Skip to the correct point in Phase 3.** Do NOT re-collect inputs (use
   `config` from state file). Do NOT re-validate (already validated). Do NOT
   re-run completed steps. Jump directly to the incomplete step.

5. **Re-read the most recent RCA report** (path from state file) before
   making any strategy decisions. This restores the context that may have
   been lost to compaction.

**If no state file exists:** proceed to Phase 1 (fresh start).

**After compaction mid-run:** A PostCompact hook automatically re-injects the
state file and all iteration summaries into your context. You do NOT need to
manually re-read them — they will appear as `additionalContext` after any
auto-compaction. However, if the restored context references an RCA report
path, **do re-read that full report** before making strategy decisions, as
the hook only includes the path (not the full report content).

---

## Phase 1 — Collect All Inputs (ask once, then go)

Ask the user for all required information **in a single message**. Do not proceed until
all required fields are answered. Accept optional fields with defaults if not provided.

### Questions to ask:

```
To run the DEFT loop, I need the following information:

1. **KPI target** — what metric are you trying to hit?
   (e.g., "FAR < 0.1% at recall=100%")

2. **Dataset paths:**
   - Training CSV path
   - Validation CSV path (for KPI measurement)
   - Images root directory
   - Training spec YAML path
   - Backbone weights directory

3. **Baseline checkpoint** — do you have a trained baseline checkpoint?
   If yes, provide the path. If no, I'll train one from scratch.

4. **Results directory** — where should I write all outputs?

5. **Augmentation arms** — any combination of the three can run in parallel each iteration:

   **Arm A: AnomalyGen** (diffusion inpainting on real PCB photos)
   - Enable? (default: yes if prerequisites available)
   - Clean image directory
   - ROI directory
   - Submask directory
   - Defect description JSONL
   - AnomalyGen checkpoint path and step
   - Pretrained checkpoints directory (Cosmos, DINOv2, C-RADIO)
   - Number of seeds per image (default: 5)

   **Arm B: Omniverse SDG** (physics-based ray tracing via `deft-aoi-omniverse-sdg`)
   - Enable? (default: yes if prerequisites available)
   - Output directory for raw SDG frames
   - Defect types to generate: shift / tombstone / sideflip (any combination; agent selects based on RCA)
   - **Desired crop count per defect type** (default: 200 crops — NOT frames, see calculation below)
   - USD scene path (auto-discover: search `augmentation/omniverse/scene/` for `*.usd` files; no Nucleus credentials needed)
   - Crop extraction script path (default: `scripts/crop_omni_sdg.py`, relative to workspace)

   **Arm C: Data Mining** (k-NN retrieval from pre-generated AnomalyGen source pool)
   - Enable? (default: yes if source parquet available)
   - Source embeddings parquet path (pre-computed SigLIP embeddings of the pre-generated AnomalyGen image pool)
   - Source images host directory (contains pre-generated AnomalyGen `defect/` and `golden/` pairs)
   - Source images container path prefix (for path translation from parquet to host)
   - Desired unique count (default: 50)
   - k-NN metric (default: cosine)
   
   > **NOTE:** The source pool is NOT real factory data. It contains pre-generated
   > AnomalyGen synthetic images (defect + golden pairs). Mining finds synthetic images
   > that are visually similar to the real failure cases identified by RCA. This means
   > ALL augmented data (Arms A, B, and C) is synthetic — only the base training CSV
   > contains real production images.

7. **Training configuration:**
   - Docker image (default: nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt)
   - Number of GPUs (default: 1)
   - Batch size (default: 16)
   - Number of epochs (default: 200)

8. **Max iterations** (default: 3)

9. **Data prep script path** (default: `scripts/changenet_data_pair_prepare.py`, bundled in this skill)

10. **Analysis script path** (default: `scripts/analyze_kpi.py` bundled with this plugin)
```

If the user provides some values inline with their initial prompt, don't re-ask those.
Fill in defaults for anything marked optional that the user doesn't provide.

---

## Phase 2 — Validate

### 2A. Verify and create workspace structure

The workflow can run standalone (without `deft-aoi-brev`). Ensure the workspace
skeleton exists before validating inputs. Create any missing directories:

```bash
WORKSPACE="${WORKSPACE:-$(pwd)}"

mkdir -p \
  "${WORKSPACE}/kpi/images" \
  "${WORKSPACE}/train/base" \
  "${WORKSPACE}/results" \
  "${WORKSPACE}/augmentation/anomalygen/checkpoint" \
  "${WORKSPACE}/augmentation/anomalygen/clean_images" \
  "${WORKSPACE}/augmentation/anomalygen/roi" \
  "${WORKSPACE}/augmentation/anomalygen/submasks" \
  "${WORKSPACE}/augmentation/omniverse/scene" \
  "${WORKSPACE}/augmentation/mining/source_images" \
  "${WORKSPACE}/augmentation/backbone" \
  "${WORKSPACE}/specs"

echo "Workspace root: ${WORKSPACE}"
```

If the workspace is empty (fresh start without the brev setup skill), tell the user
which data archives need to be extracted where. Refer to the **Workspace Layout**
section at the end of this skill for the full directory map and archive mapping.

### 2B. Validate inputs

Before starting any work, validate **all** inputs in parallel:

```bash
# Verify all paths exist
test -f "$base_training_csv"    && echo "OK: training CSV" || echo "MISSING: $base_training_csv"
test -f "$validation_csv"       && echo "OK: validation CSV" || echo "MISSING: $validation_csv"
test -d "$images_dir"           && echo "OK: images dir" || echo "MISSING: $images_dir"
test -f "$specs_file"           && echo "OK: spec YAML" || echo "MISSING: $specs_file"
test -d "$backbone_weight_dir"  && echo "OK: backbone weights" || echo "MISSING: $backbone_weight_dir"
test -f "$baseline_checkpoint"  && echo "OK: baseline checkpoint" || echo "SKIP: will train baseline"
# SDG prerequisites (check based on selected sub-skill)
if [ "$sdg_subskill" = "deft-aoi-anomalygen-inference" ]; then
  test -d "$sdg_clean_image_dir"  && echo "OK: SDG clean images" || echo "MISSING: $sdg_clean_image_dir"
  test -d "$sdg_roi_dir"          && echo "OK: SDG ROI" || echo "MISSING: $sdg_roi_dir"
  test -d "$sdg_submask_dir"      && echo "OK: SDG submask" || echo "MISSING: $sdg_submask_dir"
  test -f "$sdg_defect_desc"      && echo "OK: defect description" || echo "MISSING: $sdg_defect_desc"
  test -d "$sdg_checkpoint_path"  && echo "OK: SDG checkpoint" || echo "MISSING: $sdg_checkpoint_path"
  test -d "$sdg_pretrained_dir"   && echo "OK: pretrained models" || echo "MISSING: $sdg_pretrained_dir"
else  # deft-aoi-omniverse-sdg
  # OptiX is bundled inside the pcb-aoi-ov-sdg container — no host check needed
  echo "OK: OptiX (bundled in container)"
  test -f "$crop_script_path"     && echo "OK: crop_from_sdg.py" || echo "MISSING: $crop_script_path"
  docker image inspect nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:computex_ver1 > /dev/null 2>&1 && echo "OK: deft-aoi-omniverse-sdg image" || echo "MISSING: pcb-aoi-ov-sdg image"
fi
test -f "$data_prep_script"     && echo "OK: data prep script" || echo "MISSING: $data_prep_script"
test -f "$analyze_script"       && echo "OK: analysis script" || echo "MISSING: $analyze_script"

# Data Mining paths (only if enabled)
test -f "$source_embeddings_parquet" && echo "OK: source embeddings parquet" || echo "MISSING: $source_embeddings_parquet"
test -d "$source_images_host_dir"    && echo "OK: source images dir" || echo "MISSING: $source_images_host_dir"

# Verify Docker images
docker image inspect "$train_container" > /dev/null 2>&1 && echo "OK: TAO image" || echo "MISSING: $train_container"
# AnomalyGen image only needed if deft-aoi-anomalygen-inference sub-skill selected
[ "$sdg_subskill" = "deft-aoi-anomalygen-inference" ] && \
  { docker image inspect "$anomalygen_container" > /dev/null 2>&1 && echo "OK: AnomalyGen image" || echo "MISSING: $anomalygen_container"; }

# Verify GPU access
nvidia-smi > /dev/null 2>&1 && echo "OK: GPU" || echo "MISSING: GPU access"
```

If anything is missing, report **all** missing items at once and ask the user to fix them.
Do not proceed until validation passes.

### Pre-Flight Summary (print before starting)

Once validation passes, print a clear summary of the workflow configuration for the user to confirm at a glance. This is the last thing the user sees before the loop runs autonomously.

```
╔══════════════════════════════════════════════════════════╗
║                DEFT Loop — Pre-Flight Summary            ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  KPI Target:        FAR < 0.1% at Recall=100%            ║
║  Max Iterations:    3                                    ║
║  Training Epochs:   200 per iteration                    ║
║  Batch Size:        16                                   ║
║  GPUs:              1                                    ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  KPI Dataset                                             ║
║    Validation CSV:  <path>                               ║
║    Validation rows: <N> (X defect types)                 ║
║    Images dir:      <path>                               ║
║                                                          ║
║  Training Data                                           ║
║    Training CSV:    <path>                               ║
║    Training rows:   <N>                                  ║
║    Baseline:        <checkpoint path> / training fresh    ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  Augmentation Arms Discovered                            ║
║                                                          ║
║  Arm A — AnomalyGen:                                     ║
║    Checkpoint:      <path> (FOUND / MISSING)             ║
║    Clean images:    <N> images across <M> defect types   ║
║    ROI:             <N> types with ROI masks             ║
║    Submasks:        <N> types with submasks              ║
║    → AVAILABLE for: <defect types with all 3>            ║
║                                                          ║
║  Arm B — Omniverse SDG:                                  ║
║    USD scene:       <auto-discovered from augmentation/omniverse/scene/*.usd> ║
║    OptiX binary:    bundled in container (no host check)  ║
║    Docker image:    FOUND / MISSING                      ║
║    → AVAILABLE for: shift, tombstone, sideflip, missing  ║
║                                                          ║
║  Arm C — Data Mining (from pre-generated AnomalyGen pool):║
║    Source parquet:   <path> (FOUND / MISSING)            ║
║    Pool size:        <N> embeddings                      ║
║    Pool defect types: <list with counts>                 ║
║    → AVAILABLE for: <types present in pool>              ║
║                                                          ║
║  Backbone weights:  <path> (FOUND / MISSING)             ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  Docker Images                                           ║
║    TAO toolkit:     ✅ / ❌                               ║
║    AnomalyGen:      ✅ / ❌ / N/A                         ║
║    Embed:           ✅ / ❌                               ║
║    Mining:          ✅ / ❌                               ║
║    PCB-AOI-SDG:     ✅ / ❌ / N/A                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

Starting DEFT loop now. Will report progress at each step.
```

To populate the summary, run these discovery commands:
```bash
# KPI dataset stats
wc -l "$validation_csv" | awk '{print $1 - 1, "rows"}'
head -1 "$validation_csv"  # show columns

# Training data stats
wc -l "$base_training_csv" | awk '{print $1 - 1, "rows"}'

# AnomalyGen artifact discovery
ls "$sdg_clean_image_dir"/ 2>/dev/null | wc -l   # defect type count
ls "$sdg_roi_dir"/ 2>/dev/null | wc -l
ls "$sdg_submask_dir"/ 2>/dev/null | wc -l

# Mining pool stats
python3 -c "import pandas as pd; df=pd.read_parquet('$source_embeddings_parquet'); print(f'{len(df)} embeddings'); print(df['filepath'].str.extract(r'([^/]+)_\d+\.\w+$')[0].value_counts().to_string())" 2>/dev/null
```

**Ask the user to confirm before proceeding.** Wait for explicit approval (e.g., "looks good", "go", "yes"). Do not start the loop until the user confirms.

### Initialize State File

After validation passes (and only for a fresh start — skip if resuming):

```bash
mkdir -p ${RESULTS_DIR}
python3 -c "
import json
from datetime import datetime, timezone
state = {
    'version': 1,
    'started_at': datetime.now(timezone.utc).isoformat(),
    'kpi_target': '${KPI_TARGET}',
    'results_dir': '${RESULTS_DIR}',
    'max_iterations': ${MAX_ITER},
    'current_iteration': 0,
    'current_step': 'validated',
    'config': {
        'training_csv': '${TRAINING_CSV}',
        'validation_csv': '${VALIDATION_CSV}',
        'images_dir': '${IMAGES_DIR}',
        'specs_file': '${SPECS_FILE}',
        'backbone_weight_dir': '${BACKBONE_WEIGHT_DIR}',
        'train_container': '${TRAIN_CONTAINER}',
        'num_gpus': ${NUM_GPUS},
        'batch_size': ${BATCH_SIZE},
        'num_epochs': ${NUM_EPOCHS},
        'data_prep_script': '${DATA_PREP_SCRIPT}',
        'analyze_script': '${ANALYZE_SCRIPT}'
    },
    'iterations': {}
}
with open('${RESULTS_DIR}/deft_state.json', 'w') as f:
    json.dump(state, f, indent=2)
print('State file initialized: ${RESULTS_DIR}/deft_state.json')
"
```

---

## Phase 3 — Execute (fully autonomous, sequential)

Run the entire loop without human intervention. Print a one-line status at each milestone.

> **IMPORTANT: Execution order.** Steps must run sequentially — each step depends
> on the previous step's output. In particular, SDG (Step 2) must run **after**
> evaluation and gap analysis (Step 0 + Step 1), because gap analysis identifies
> which defect types to target. Do NOT run SDG in parallel with baseline training.

### Long-Running Job Monitoring

For any step that takes more than 5 minutes (training, SDG inference), set up
periodic status reporting using `/loop`:

```
/loop 10m <status check command>
```

This ensures the user sees progress updates every 10 minutes. Include:
- Current epoch or image count
- Elapsed time
- Whether the process is still running

### Step 0 — Baseline

If baseline checkpoint exists, skip to evaluation. Otherwise train from scratch.

1. Train the model on the base training CSV
2. **Monitor training with `/loop 10m`** to report epoch progress
3. Run inference on the full validation set
4. Analyze results — record FAR, Recall, threshold
5. If KPI already met, print final report and stop
6. **Run full RCA on baseline results** (call `deft-aoi-rca-changenet` skill)

```
[Step 0] Training baseline... (200 epochs)
[Step 0] Baseline training complete. Running inference...
[Step 0] Baseline: FAR=X% at Recall=100%. KPI not met.
[Step 0] Running full RCA on baseline...
[Step 0] RCA complete. Top root causes: ...
[Step 0] Entering DEFT loop.
```

#### Step 0 RCA (MANDATORY)

After baseline evaluation, run the **`deft-aoi-rca-changenet`** skill on the baseline
inference results. This is critical because the RCA:

- Identifies which defect types are failing and why (data gap, golden quality, etc.)
- Produces the target failure images needed for data mining in iteration 1
- Quantifies the impact of each root cause via counterfactual simulations
- Determines whether the KPI gap is addressable by SDG/mining or requires
  data fixes (mislabels, golden image corrections, etc.)

**Without baseline RCA, the first iteration's augmentation is blind** — the agent
cannot make an informed decision about which SDG defect types to target or which
failure images to mine for.

**Inputs to `deft-aoi-rca-changenet`:**
- Experiment result directory: `${RESULTS_DIR}/baseline/`
- Dataset directory: the images root + CSV paths
- Target KPI: the user's KPI target

**IMPORTANT:** Before launching RCA, read the `deft-aoi-rca-changenet` skill's SKILL.md
and follow its instructions. Pass the relevant skill sections directly into
each subagent's prompt — do not summarize.

**Output location:** Save the RCA report into the baseline results directory:
```
${RESULTS_DIR}/baseline/rca_results/<timestamp>/RCA_Report.md
```

**Output used by iteration 1:**
- List of target defect types for SDG
- Target failure images for data mining (lowest-scoring defect samples)
- Prioritized fix recommendations informing augmentation strategy

---

### Loop: Repeat for each iteration N (starting at N=1)

```bash
ITER=N
mkdir -p ${RESULTS_DIR}/iter${ITER}/{sdg_output,mining_output,dataset,train,inference}
```

---

### Step 1 — Root Cause Analysis & Gap Analysis (call `deft-aoi-rca-changenet` skill)

Run the **`deft-aoi-rca-changenet`** skill on the previous iteration's inference results to
perform a deep, image-evidence-driven investigation of model failures.

**IMPORTANT:** Before launching RCA, read the `deft-aoi-rca-changenet` skill's SKILL.md
and follow its instructions. Pass the relevant skill sections directly into
each subagent's prompt — do not summarize.

**Inputs to `deft-aoi-rca-changenet`:**
- Experiment result directory: `${RESULTS_DIR}/baseline/` (or `iter${N-1}/`)
- Dataset directory: the images root + CSV paths
- Target KPI: the user's KPI target (e.g., FAR < 0.1% at recall=100%)

The RCA skill will:
1. **Phase 1 (Score Analysis):** Compute score statistics, tier, threshold sweep,
   per-defect-type breakdown, drop-N analysis
2. **Phase 2-4 (Parallel Investigation):** Launch 6 parallel subagents for image
   evidence (golden audit, failure clustering, FP analysis), data analysis,
   config audit, and exploratory investigation
3. **Phase 5 (Counterfactual):** What-if simulations to quantify each root cause's
   impact on KPI

From the RCA output, extract:
- List of target defect types for the SDG step
- Which defect types SDG can cover vs cannot cover
- Prioritized fix recommendations

**Output location:** Save the RCA report into the previous iteration's results
directory (the iteration being analyzed):
```
${RESULTS_DIR}/baseline/rca_results/<timestamp>/RCA_Report.md    # if analyzing baseline
${RESULTS_DIR}/iter${N-1}/rca_results/<timestamp>/RCA_Report.md  # if analyzing iter N-1
```

Get the timestamp with `date +%Y-%m-%d_%H%M%S`. The RCA report must be a
self-contained document that can be reviewed independently.

```
[Iter N, Step 1] RCA complete. Top root causes: ...
[Iter N, Step 1] SDG can cover: ... | Cannot cover: ...
[Iter N, Step 1] RCA report saved to: ${RESULTS_DIR}/<prev_iter>/rca_results/<timestamp>/RCA_Report.md
```

**Output:** RCA report + list of target defect types for SDG + target failure
images for data mining (lowest-scoring defect samples from inference).

---

### Step 2 — Augment (three parallel arms)

> **IMPORTANT:** This step must run **after** Step 1 (gap analysis) completes,
> because gap analysis determines which defect types to target.

---

#### Step 2-Pre: Capability Discovery (run before deciding anything)

Before selecting which arms to enable, the agent must **discover what each arm
can actually generate** given the current state of artifacts and data pools.
This prevents augmenting for defect types that an arm cannot cover, and avoids
wasting training slots on defect types not in the RCA gap list.

Run all three discovery checks in parallel:

**Arm 2A — AnomalyGen: inspect artifact directories**

```bash
# For each defect type, check if ALL THREE required subdirectories exist
# IMPORTANT: Exclude macOS ._ metadata files — they show up as entries but aren't real images
for defect_type in $(ls <clean_image_dir>/ | grep -v '^\.'); do
  clean_count=$(find <clean_image_dir>/$defect_type -type f ! -name '._*' ! -name '.DS_Store' 2>/dev/null | wc -l)
  has_roi=$(test -d <roi_dir>/$defect_type && echo yes || echo NO)
  has_sub=$(test -d <submask_dir>/$defect_type && echo yes || echo NO)
  echo "AnomalyGen | $defect_type | clean=$clean_count | roi=$has_roi | submask=$has_sub"
done
```

> **macOS artifact warning:** Archives created on macOS contain `._*` metadata files
> and `.DS_Store` entries. Always filter these when counting images or listing directories:
> `find <dir> -type f ! -name '._*' ! -name '.DS_Store'`

A defect type is **runnable** by AnomalyGen only if it has entries in all three
directories: `clean_image_dir/`, `roi_dir/`, `submask_dir/`. Missing any one → skip
that type. Do not attempt to run AnomalyGen for types with incomplete artifacts.

**Arm 2B — Omniverse SDG: fixed schema lookup**

Omniverse deft-aoi-omniverse-sdg supports a fixed set of defect types defined by the pipeline schema.
No filesystem check needed — the supported types are known statically:

| Defect type | Omniverse support |
|-------------|------------------|
| `shift` | ✅ `defect` pipeline, `defects.shift` |
| `tombstone` | ✅ `defect` pipeline, `defects.tombstone` |
| `sideflip` | ✅ `defect` pipeline, `defects.sideflip` |
| `missing` | ✅ `missing` pipeline |
| `lifted_lead` | ❌ Not in schema (requires deformable solder physics) |
| `excess_solder` | ❌ Not in schema (requires solder flow physics) |
| `upside_down` | ❌ Not in schema |
| `polarity` | ❌ Not in schema |
| anything else | ❌ Not in schema |

**Arm 2C — Data Mining: scan parquet pool for present defect types**

```python
import pandas as pd, re

df = pd.read_parquet("<source_parquet>")
# Extract defect type prefix from filename (everything before last _NNNNN.ext)
df['defect_type'] = df['filepath'].apply(
    lambda p: re.sub(r'_\d+\.\w+$', '', p.split('/')[-1])
)
print(df['defect_type'].value_counts().to_string())
```

A defect type is **mineable** only if it has entries in the parquet. Print the full
count per type so the agent can see pool composition at a glance.

**After discovery, produce a capability table:**

```
[Iter N, Step 2-Pre] Capability discovery results:

  RCA gap defect types (from Step 1): <list, ranked by impact>

  Arm 2A (AnomalyGen):
    runnable types:  <list — has clean + ROI + submask>
    missing artifacts: <list — has clean but no ROI/submask>
    no artifacts:    <list — nothing at all>

  Arm 2B (Omniverse SDG):
    runnable types:  shift, tombstone, sideflip, missing  [fixed]
    not supported:   lifted_lead, excess_solder, upside_down, polarity, ...

  Arm 2C (Data Mining):
    pool contents:   <defect_type: count, ...>
    mineable types:  <types present in pool AND in RCA gap list>
    pool gaps:       <RCA gap types with 0 entries in pool>

  Coverage vs RCA gaps:
    <defect_type>: covered by [2A, 2B, 2C] / [2B only] / [NONE — gap!]
    ...
```

For any RCA gap type covered by **NONE** of the arms, report it explicitly:
```
  ⚠️  UNCOVERED GAP: <defect_type> — no arm can generate training data.
      Suggested fix: <build AnomalyGen artifacts / extend Omniverse schema / add to mining pool>
```

---

#### Strategy Selection

After capability discovery, select arms based on the intersection of RCA gaps and
arm coverage. Enable an arm only if it can cover **at least one** RCA gap defect type.

| Arm | Enable if |
|-----|-----------|
| **2A: AnomalyGen** | At least one RCA gap type has complete artifacts (clean + ROI + submask) |
| **2B: Omniverse SDG** | At least one RCA gap type is in the Omniverse schema (shift/tombstone/sideflip/missing) |
| **2C: Data Mining** | At least one RCA gap type has entries in the parquet pool |

The agent must print its final decision:

```
[Iter N, Step 2] Augmentation strategy decision:
  - RCA top failure modes: <list>
  - Arm 2A (AnomalyGen): ENABLED for [<types>] / SKIPPED — <reason>
  - Arm 2B (Omniverse SDG): ENABLED for [<types>] / SKIPPED — <reason>
  - Arm 2C (Mining): ENABLED for [<types>] / SKIPPED — <reason>
  - Uncovered gaps (no arm can help): <list or "none">
  - Launching enabled arms in parallel now.
```

Then launch all enabled arms **in a single message** using parallel tool calls:

```
Step 1: RCA → gap defect types ranked by impact
        │
        └── Step 2-Pre: Capability discovery (AnomalyGen artifacts + Omniverse schema + Mining pool scan)
                │
                ├──── Step 2A: AnomalyGen          ~30 min  [if runnable types ∩ RCA gaps ≠ ∅]
                │         └── anomalygen_output/reconstructed_image/ + original_image/
                │
                ├──── Step 2B: Omniverse SDG        ~20–40 min  [if schema types ∩ RCA gaps ≠ ∅]
                │         └── sdg_crops/defect/ + sdg_crops/good/
                │
                └──── Step 2C: Data Mining          ~5 min  [if pool types ∩ RCA gaps ≠ ∅]
                          └── mining_output/mined_similar_files.csv
                │
                ▼
        Step 3: Data Prep (merge covered sources into training CSV)
```

---

#### Step 2A — AnomalyGen (call `deft-aoi-anomalygen-inference` skill)

> **Skip if prerequisites unavailable (see decision criteria above).**

Diffusion-based inpainting — generates defect images by inpainting onto real PCB photos.

```
[Iter N, Step 2A] Running AnomalyGen for defect types: ...
[Iter N, Step 2A] AMP complete: X masks generated
[Iter N, Step 2A] SDG inference running (~30 min)...
[Iter N, Step 2A] AnomalyGen complete: X synthetic image pairs
```

**Monitor with `/loop 10m`** (~30 min).

**Output:**
- `anomalygen_output/reconstructed_image/` — synthetic defect images (NG)
- `anomalygen_output/original_image/` — paired clean images (OK)

---

#### Step 2B — Omniverse SDG (call `deft-aoi-omniverse-sdg` skill)

> **Skip if prerequisites unavailable (see decision criteria above).**

Physics-based ray tracing — generates full-board AOI scans then extracts per-component crops.

**Step 2B-i — Run deft-aoi-omniverse-sdg pipelines (defect + good in parallel):**

Use the `deft-aoi-omniverse-sdg` skill to generate configs for both pipelines, then launch them together:

```bash
# Ensure host output dirs are writable by the container user before launch
mkdir -p <output_dir>/defect <output_dir>/good
chmod -R a+rwx <output_dir>/defect <output_dir>/good

# Defect pipeline (NG frames) — background
# IMPORTANT: Pass the full script invocation as a single payload string
docker run --gpus all --rm --network host \
  -e OMNI_USER -e OMNI_PASS \
  # OptiX is bundled in the container; host mount only needed if using a custom build
  # -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v <output_dir>/defect:<output_dir>/defect \
  -v ~/.sdg_config:/config \
  nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:computex_ver1 \
  "/home/ubuntu/pcb-aoi/scripts/sdg/standalone/sdg_pipeline.py --config /config/<defect_config>.yaml" &

# Good pipeline (OK reference frames)
docker run --gpus all --rm --network host \
  -e OMNI_USER -e OMNI_PASS \
  # OptiX is bundled in the container; host mount only needed if using a custom build
  # -v /usr/share/nvidia/nvoptix.bin:/usr/share/nvidia/nvoptix.bin:ro \
  -v <output_dir>/good:<output_dir>/good \
  -v ~/.sdg_config:/config \
  nvcr.io/nvidian/iva/pcb-aoi-ov-sdg:computex_ver1 \
  "/home/ubuntu/pcb-aoi/scripts/sdg/standalone/sdg_pipeline.py --config /config/<good_config>.yaml"
```

> **Local USD scene:** Auto-discover the scene file by searching `augmentation/omniverse/scene/`
> for `*.usd` files (e.g., `find ~/workspace/augmentation/omniverse/scene -name "*.usd"`).
> Set this path in both defect and good configs. No Nucleus credentials needed for local scenes.

> **Config requirements:** defect config needs `bounding_box_2d_tight: true`,
> `semantic_types: [class, defect]`; good config needs `semantic_types: [class]`.

> **Container permissions:** The SDG container runs as `ubuntu` (uid=1000), NOT root.
> Output directories must be writable by uid=1000. Either create them with `chmod 777`
> before launching, or use `mkdir -p` from the host user. If a previous failed run
> created root-owned directories, remove them with a container:
> `docker run --rm -v <parent>:/mnt <image> bash -c "rm -rf /mnt/<dir> && mkdir -m 777 /mnt/<dir>"`

> **Config from scratch:** Never write SDG configs from memory. Always copy an existing
> working config and change only what's needed (e.g., `num_triggers`). The full config
> has ~100 lines covering camera, lighting, scan grid, component types, defect params,
> augmentation, and writer settings. A minimal config will fail with KeyError on the
> first missing field.

**Monitor with `/loop 10m`** (~20–40 min per pipeline).

**Step 2B-ii — Set num_triggers:**

> **ALWAYS use num_triggers = 1.** A single trigger renders 143 frames across all
> grid positions. With ~20 crops/frame, that produces ~2,800 crops per trigger —
> far more than needed. Multiple triggers multiply this exponentially and create
> massive datasets that slow down training with no benefit.
>
> Use the mandatory subsampling step (2B-iv) to cap at the desired crop count.

Set `num_triggers: 1` in both defect and good configs. Do not increase this value.

> **Crop budget: at most 50 per defect type.** After crop extraction, subsample to
> a maximum of 50 crops per defect type (shift, tombstone, sideflip, missing).
> More than 50 per type does not improve KPI and inflates training time.

**Step 2B-iii — Extract per-component crops:**

Frame-index pairing: frame N in the defect run = same camera position as frame N in the good run.

> **CRITICAL: Crop each image using its OWN bbox, matched via IoU.**
> For each defective component, find its bbox in the good frame via IoU-based
> spatial matching (threshold ≥ 0.05). Crop the defect image with the defect
> bbox + 30px padding, crop the good image with the good bbox + 30px padding.
> This keeps each component centered in its crop. **Discard pairs where no
> good-frame match is found** — do not fall back to defect coords.
> Also discard zero-area crops (edge-clipped components).
>
> Script: `scripts/crop_omni_sdg.py`

```bash
python3 scripts/crop_omni_sdg.py \
  --defect-dir <output_dir>/defect \
  --good-dir   <output_dir>/good \
  --defect-out <results_dir>/iter${ITER}/sdg_crops/defect \
  --good-out   <results_dir>/iter${ITER}/sdg_crops/good \
  --padding 30
```

**Step 2B-iv — Mandatory subsampling to crop budget:**

Raw crop output will almost always exceed the desired crop count. **Always
subsample** — do not pass raw output directly to data prep.

```bash
# Count raw crops per defect type
RAW_COUNT=$(ls <results_dir>/iter${ITER}/sdg_crops/defect/*_ng.jpg | wc -l)

if [ "$RAW_COUNT" -gt "$DESIRED_CROPS" ]; then
  echo "[Iter N, Step 2B] Subsampling: ${RAW_COUNT} raw crops → ${DESIRED_CROPS} (budget)"
  # Random subsample: move excess to a holdout dir (don't delete — may be useful later)
  mkdir -p <results_dir>/iter${ITER}/sdg_crops/defect_holdout
  python3 -c "
import os, random, shutil, glob

crops = sorted(glob.glob('<results_dir>/iter${ITER}/sdg_crops/defect/*_ng.jpg'))
random.seed(42)
random.shuffle(crops)
keep = set(crops[:${DESIRED_CROPS}])
for f in crops:
    if f not in keep:
        # Move both _ng and _ok of the pair
        for suffix in ['_ng.jpg', '_ok.jpg']:
            base = f.replace('_ng.jpg', suffix)
            dst = base.replace('/defect/', '/defect_holdout/')
            if os.path.exists(base):
                shutil.move(base, dst)
print(f'Kept ${DESIRED_CROPS}, moved {len(crops) - ${DESIRED_CROPS}} to holdout')
"
else
  echo "[Iter N, Step 2B] Raw crops (${RAW_COUNT}) within budget (${DESIRED_CROPS}) — no subsampling needed"
fi
```

**Output:**
- `sdg_crops/defect/` — subsampled paired `*_ng.jpg` / `*_ok.jpg` crop pairs (within budget)
- `sdg_crops/defect_holdout/` — excess crops (preserved, not deleted)
- `sdg_crops/good/` — PASS crop pairs (use when OOD component coverage is needed)

```
[Iter N, Step 2B] Omniverse SDG complete: ${DESIRED_CROPS} defect pairs (from ${RAW_COUNT} raw), Y PASS pairs
```

---

#### Step 2C — Data Mining (call `deft-aoi-data-mining` skill)

> **Skip if source parquet unavailable (see decision criteria above).**

Mine visually similar images from a pre-generated AnomalyGen source pool using k-NN search.
The source pool contains synthetic defect/golden pairs from prior AnomalyGen runs — NOT
real factory images.

**Inputs:**
- **Target images:** lowest-scoring NG samples from RCA (real failure cases from inference)
- **Source embeddings parquet:** pre-computed `filepath` + `image_embed` (SigLIP 768-dim) of the AnomalyGen source pool
- **Desired unique count:** how many similar synthetic images to retrieve

**Procedure:**
1. Extract target images from inference CSV (worst-scoring defect samples)
2. Copy to `${RESULTS_DIR}/iter${ITER}/mining_output/targets/`
3. Generate target embeddings via `nvcr.io/nvidian/iva/embed:latest`
4. Run k-NN via `nvcr.io/nvidian/iva/mining:latest` against source parquet
5. Export `mined_similar_files.csv` with translated host paths

```
[Iter N, Step 2C] Mining: X target failure images from RCA
[Iter N, Step 2C] Target embeddings generated (768-dim SigLIP)
[Iter N, Step 2C] k-NN mining against source pool (Y images)...
[Iter N, Step 2C] Mined Z similar images → mining_output/mined_similar_files.csv
```

**Path translation:** Source parquet uses container paths — translate back to host
paths using the user-provided source images dir and container path prefix.

**Output:**
- `mining_output/mined_similar_files.csv` — host paths to mined source images
- `mining_output/targets/` — target images used for mining

> **NOTE:** Mined images are individual files (not pre-paired). Data prep must pair
> them with corresponding golden images from the source pool's `golden/` directory.

---

### Step 3 — Data Prep (call data-prep sub-skill)

Process output from all enabled augmentation arms, then merge into the combined training CSV.
Run 3A, 3B, 3C in parallel since they are independent; then merge (3D) after all complete.

**Step 3A — Process AnomalyGen output (if arm 2A was enabled):**

Use `changenet_data_pair_prepare.py` to pair `reconstructed_image/` (NG) with `original_image/` (OK)
and copy into the dataset tree.

```
[Iter N, Step 3A] Processing AnomalyGen output → anomalygen_pairs.csv
```

**Step 3B — Process Omniverse SDG output (if arm 2B was enabled):**

deft-aoi-omniverse-sdg crops are already paired (`*_ng.jpg` / `*_ok.jpg`) from the crop extraction step.
Use `changenet_data_pair_prepare.py` pointing at `sdg_crops/defect/` as the NG directory and
`sdg_crops/defect/` (OK counterparts) or `sdg_crops/good/` for PASS pairs.

```
[Iter N, Step 3B] Processing Omniverse SDG crops → sdg_pairs.csv
```

**Step 3C — Process mined data (if arm 2C was enabled):**

Do not assume the mined source pool already exposes paired directories.
`mined_similar_files.csv` is a flat list of selected files. Stage filtered
mined pairs into temporary `defect/` and `golden/` directories first, then
run `changenet_data_pair_prepare.py` on those staged dirs. If the source pool
does have paired `defect/` and `golden/` directories, filter to only files in
`mined_similar_files.csv` before running.

```bash
python changenet_data_pair_prepare.py \
  --input-dir <source_images_host_dir>/defect \
  --golden-dir <source_images_host_dir>/golden \
  --images-dir <images_dir> \
  --subdir iter${ITER}_mined \
  -o ${RESULTS_DIR}/iter${ITER}/dataset/mined_pairs.csv
```

```
[Iter N, Step 3C] Processing mined data → mined_pairs.csv
```

**Step 3D — Merge all sources:**

Concatenate in order: base CSV + anomalygen CSV (if 2A) + sdg CSV (if 2B) + mined CSV (if 2C).

```
[Iter N, Step 3D] Merging:
  Base:       X rows
  AnomalyGen: Y rows  (arm 2A)
  Omniverse:  Z rows  (arm 2B)
  Mined:      W rows  (arm 2C)
  Total:      T rows → train_combined_iter${ITER}.csv
```

> **CRITICAL: Trailing newline.** Always check before appending:
> `[ -n "$(tail -c 1 file)" ] && echo "" >> file`

> **CRITICAL: Synthetic data ratio.** All augmented data (SDG, AnomalyGen, AND mined)
> is synthetic — the mining source pool contains pre-generated AnomalyGen images, not
> real factory data. Only the base training CSV has real images. After merging, compute
> the ratio of synthetic (all augmented arms) to real (base CSV only). If synthetic
> exceeds 80% of total, **subsample synthetic data** to cap at ~500 per defect type.
> Excessive synthetic data causes domain gap — the model learns synthetic appearance
> and regresses on real hard cases. This was observed empirically: at 96% synthetic
> (iter3), FAR regressed from 35.56% to 46.69% despite more data. Print the ratio:
> ```
> [Iter N, Step 3D] Synthetic ratio: X% (Y augmented / Z total) — only base CSV is real
> [Iter N, Step 3D] ⚠️ Ratio exceeds 80% — subsampling SDG to 500/type
> ```

---

### Step 4 — Train (TAO ChangeNet)

Run TAO ChangeNet training directly via the TAO container. Pass key arguments:

```
train_csv=<merged training CSV from Step 3D>
val_csv=<validation CSV>
images_dir=<images root>
results_dir=<results dir>
backbone_weight=<backbone weights path>
pretrained_model_path=<previous iteration's checkpoint>
mode=toolkit
```

**Monitor training with `/loop 10m`** to report epoch progress and latest checkpoint.

```
[Iter N, Step 4] Training with augmented data... (X epochs, batch_size=Y)
[Iter N, Step 4] Training complete. Checkpoint: ...
```

> **CRITICAL: Load weights only, reset epoch counter.** When fine-tuning from a
> previous checkpoint, use the mechanism that loads weights but resets training
> state (epochs, optimizer). Using a "resume" mechanism restores the epoch counter,
> causing training to terminate immediately if `max_epochs` hasn't changed.
>
> In TAO ChangeNet: use `train.pretrained_model_path`, NOT `train.resume_training_checkpoint_path`.

---

### Step 5 — Evaluate and Decide

Run inference and analyze metrics. Decide whether to loop or stop.

```
[Iter N, Step 5] Running inference on validation set...
[Iter N, Step 5] Results: FAR=X% at Recall=100%
[Iter N, Step 5] KPI not met. Continuing to iteration N+1...
```

| Condition | Action |
|---|---|
| KPI met | **STOP** — print final report |
| KPI not met AND `N < max_iterations` | **LOOP** — go back to Step 1 |
| KPI not met AND `N == max_iterations` | **STOP** — print final report with best iteration |

#### Step 5B — Write Iteration Summary and Checkpoint

**Always run this after Step 5 evaluation, before deciding to loop or stop.**

1. **Write the iteration summary** to `${RESULTS_DIR}/iter${ITER}_summary.md`
   (or `baseline_summary.md` for Step 0). Follow the template from the
   State Management section. Keep it under 300 words — this is what you'll
   re-read after compaction, so it must be dense and actionable.

2. **Update the state file** with final metrics:
   ```bash
   python3 -c "
   import json
   with open('${RESULTS_DIR}/deft_state.json', 'r') as f:
       state = json.load(f)
   state['iterations']['iter${ITER}']['status'] = 'complete'
   state['iterations']['iter${ITER}']['far'] = ${FAR}
   state['iterations']['iter${ITER}']['recall'] = ${RECALL}
   state['iterations']['iter${ITER}']['threshold'] = ${THRESHOLD}
   state['iterations']['iter${ITER}']['checkpoint'] = '${CHECKPOINT_PATH}'
   state['current_step'] = 'evaluated'
   with open('${RESULTS_DIR}/deft_state.json', 'w') as f:
       json.dump(state, f, indent=2)
   "
   ```

3. **If looping:** before starting Step 1 of the next iteration, re-read
   all `*_summary.md` files to refresh context on the full loop history.
   This is critical after compaction — do not rely on memory alone.

---

## Phase 4 — Final Report

Always print this at the end, whether KPI was met or not.

### 4A. Summary Table

Print the summary to the user AND save it as a markdown file at
`${RESULTS_DIR}/DEFT_Loop_Report.md`. The report must be a self-contained
document that captures the full loop history.

```markdown
# DEFT Loop Report

**Date:** <YYYY-MM-DD HH:MM:SS>
**KPI target:** <kpi>
**KPI met:** YES / NO
**Iterations run:** <N>
**Data augmentation:** SDG (anomalygen) + Data Mining (if enabled)

## Results per Iteration

| Iteration | FAR @ 100% Recall | Threshold | SDG Images | Mined Images | Training Rows |
|-----------|-------------------|-----------|------------|--------------|---------------|
| baseline  | X%                | T         | —          | —            | N             |
| iter1     | X%                | T         | Y          | Z            | N             |
| iter2     | X%                | T         | Y          | Z            | N             |

## Best Result

- **Best iteration:** <N>
- **Best FAR @ 100% recall:** X%
- **Best checkpoint:** <path>

## Configuration

- Training CSV: <path>
- Validation CSV: <path>
- Epochs per iteration: <N>
- Batch size: <N>
- Source embeddings parquet: <path> (if data mining enabled)
- SDG N (seeds per image): <N>
- Mining desired unique count: <N> (if data mining enabled)

## Per-Iteration Details

### Baseline
- Training rows: <N>
- Checkpoint: <path>
- FAR @ 100% recall: X%

### Iteration 1
- SDG images added: <N>
- Mined images added: <N>
- Total training rows: <N>
- Checkpoint: <path>
- FAR @ 100% recall: X%
- Delta from previous: <+/- X%>

(repeat for each iteration)
```

Save to: `${RESULTS_DIR}/DEFT_Loop_Report.md`

### 4B. Final RCA (call `deft-aoi-rca-changenet` skill)

Run the **`deft-aoi-rca-changenet`** skill on the **best iteration's** inference results
to produce a comprehensive final analysis. This gives the user actionable
next steps regardless of whether the KPI was met.

**If KPI was met:** The RCA identifies remaining weaknesses and risks —
what could regress, which component types are borderline, data quality
issues to fix for robustness.

**If KPI was not met:** The RCA provides the prioritized fix path —
what-if simulations showing exactly which fixes would close the gap,
and whether the KPI is achievable with data/config changes alone.

```
[Final] Running RCA on best iteration results...
[Final] RCA report saved to: <results_dir>/<best_iter>/rca_results/<timestamp>/RCA_Report.md
```

The RCA report is the primary deliverable of the DEFT loop — it tells
the user not just *what* the numbers are, but *why* and *what to do next*.

---

## Sub-Skill Interfaces

Each sub-skill has a defined **input/output contract**. The specific implementation is swappable — what matters is that it honors the interface.

### Augmentation Arms

Three independent arms, all delivering paired NG/OK crops to the data-prep step:

| | Arm A: `deft-aoi-anomalygen-inference` | Arm B: `deft-aoi-omniverse-sdg` (Omniverse) | Arm C: `deft-aoi-data-mining` |
|---|---|---|---|
| **Approach** | Diffusion inpainting on real PCB photos | Physics-based ray tracing from USD scene | k-NN retrieval from pre-generated AnomalyGen output pool (NOT real factory images) |
| **Input** | Clean images, ROI, submasks, descriptions, checkpoint | USD scene, defect config, output dir | Source parquet, target failure images |
| **Output** | `anomalygen_output/reconstructed_image/` + `original_image/` | `sdg_crops/defect/` + `sdg_crops/good/` | `mining_output/mined_similar_files.csv` |
| **Defect types** | Any (text-described) | shift, tombstone, sideflip | Whatever exists in pre-generated AnomalyGen source pool |
| **Runtime** | ~30 min | ~20–40 min | ~5 min |
| **Prerequisites** | Real board photos + ROI/masks + checkpoint | pcb-aoi-ov-sdg image (OptiX bundled inside) | Source embeddings parquet |
| **Best for** | Realistic texture variation on real boards | Zero-shot geometric defects (tombstone, sideflip) | Closing domain gap with real data |

### Data Prep

| | Description |
|---|---|
| **Input** | Paired image directories (NG + OK), target dataset layout info (images_dir, column format, naming convention) |
| **Output** | Training CSV compatible with the dataloader, images placed in the expected directory structure |
| **Default script** | `scripts/changenet_data_pair_prepare.py` (bundled in this skill) |

### Train

| | Description |
|---|---|
| **Input** | Training CSV, spec YAML, pretrained checkpoint to fine-tune from |
| **Output** | New model checkpoint |
| **Default sub-skill** | `tao-changenet` (classification mode) |

### Evaluate

| | Description |
|---|---|
| **Input** | Model checkpoint, validation CSV, spec YAML |
| **Output** | `inference.csv` with scores, metrics summary (FAR, Recall, F1, threshold) |
| **Default sub-skill** | TAO inference + analysis script |

### Data Mining (Arm C)

| | Description |
|---|---|
| **Input** | Pre-computed source embeddings parquet (SigLIP 768-dim), target failure images from RCA, desired unique count |
| **Output** | `mined_similar_files.csv` — filepaths to similar synthetic images from the pre-generated AnomalyGen source pool |
| **Containers** | `nvcr.io/nvidian/iva/embed:latest` (target embeddings), `nvcr.io/nvidian/iva/mining:latest` (k-NN search) |
| **Sub-skill** | `deft-aoi-data-mining` |
| **Runtime** | ~5 min (source embeddings pre-computed; only target embedding + k-NN needed) |

---

## Workspace Layout

The DEFT workspace separates concerns into 5 top-level directories. The setup
skill (`deft-aoi-brev`) creates this structure automatically from the workspace
archive. All paths in this skill are relative to the workspace root (`~/workspace/`).

```
workspace/
├── kpi/                                    # Validation set — FIXED, read-only during loop
│   ├── val.csv                             # KPI validation CSV
│   └── images/                             # Shared image pool (referenced by all CSVs)
│
├── train/                                  # All training data
│   ├── base/                               # Original base training set
│   │   └── train.csv
│   ├── iter1/                              # Augmented data from iteration 1
│   │   ├── anomalygen/                     # Arm A: AnomalyGen diffusion output
│   │   │   ├── reconstructed_image/        #   synthetic defect images (NG)
│   │   │   └── original_image/             #   paired clean images (OK)
│   │   ├── omniverse/                      # Arm B: Omniverse SDG output
│   │   │   ├── frames/                     #   raw full-board renders
│   │   │   │   ├── defect/trigger_NNNN/    #     143 frames per trigger
│   │   │   │   └── good/trigger_NNNN/
│   │   │   └── crops/                      #   extracted per-component crops
│   │   │       ├── defect/                 #     component_type_trigger_frame_comp.png
│   │   │       └── good/
│   │   ├── mining/                         # Arm C: Data mining output
│   │   │   ├── targets/                    #   RCA failure images used as query
│   │   │   ├── embeddings/                 #   SigLIP embeddings of targets
│   │   │   └── mined_files.csv             #   retrieved similar images
│   │   ├── anomalygen_pairs.csv            # Processed pair CSVs per arm
│   │   ├── sdg_pairs.csv
│   │   ├── mined_pairs.csv
│   │   └── train_combined.csv              # Final merged: base + all arms
│   └── iter2/                              # (same structure)
│
├── results/                                # Evaluation + loop state
│   ├── deft_state.json                     # Loop checkpoint (crash recovery)
│   ├── deft_report.md                      # Final summary report
│   ├── baseline/
│   │   ├── checkpoint/                     # Trained model weights
│   │   ├── inference/                      # inference.csv + analysis
│   │   ├── rca/                            # RCA_Report.md + evidence images
│   │   └── summary.md                      # Compact iteration digest
│   └── iter1/
│       ├── checkpoint/
│       ├── inference/
│       ├── rca/
│       └── summary.md
│
├── augmentation/                           # Static prerequisites (don't change per iter)
│   ├── anomalygen/                         # Arm A prerequisites
│   │   ├── checkpoint/                     #   trained AnomalyGen model (project-specific)
│   │   ├── pretrained/                     #   Cosmos, T5, DINOv2, C-RADIO (auto-downloaded)
│   │   ├── clean_images/                   #   per-defect-type clean PCB images
│   │   ├── roi/                            #   per-defect-type ROI masks
│   │   ├── submasks/                       #   per-defect-type submask templates
│   │   └── defect_descriptions.jsonl
│   ├── omniverse/                          # Arm B prerequisites
│   │   └── scene/                          #   USD scene + component assets
│   ├── mining/                             # Arm C prerequisites
│   │   ├── source_embeddings.parquet       #   pre-computed SigLIP embeddings
│   │   └── source_images/                  #   source image pool (defect + golden)
│   └── backbone/                           # Shared backbone weights (C-RADIOv2)
│
└── specs/                                  # Training config YAMLs
    ├── baseline_spec.yaml
    └── inference_spec.yaml
```

### Key paths for this skill

| Variable | Path | Description |
|----------|------|-------------|
| `images_dir` | `kpi/images/` | Shared image pool (train + val CSVs both reference this) |
| `training_csv` | `train/base/train.csv` | Base training CSV |
| `validation_csv` | `kpi/val.csv` | KPI validation CSV |
| `results_dir` | `results/` | Loop state, checkpoints, inference, RCA |
| `specs_file` | `specs/baseline_spec.yaml` | Training spec YAML |
| `backbone_weight_dir` | `augmentation/backbone/` | C-RADIOv2 weights |
| `anomalygen_checkpoint` | `augmentation/anomalygen/checkpoint/` | AnomalyGen model |
| `anomalygen_pretrained` | `augmentation/anomalygen/pretrained/` | Auto-downloaded at first run |
| `clean_image_dir` | `augmentation/anomalygen/clean_images/` | Per-defect-type clean images |
| `roi_dir` | `augmentation/anomalygen/roi/` | Per-defect-type ROI masks |
| `submask_dir` | `augmentation/anomalygen/submasks/` | Per-defect-type submask templates |
| `sdg_scene` | `augmentation/omniverse/scene/` | USD scene for Omniverse SDG |
| `source_parquet` | `augmentation/mining/source_embeddings.parquet` | Mining source embeddings |
| `source_images_dir` | `augmentation/mining/source_images/` | Mining source image pool |

### Bundled scripts (in plugin)

| Script | Usage |
|--------|-------|
| `scripts/analyze_kpi.py` | `python3 scripts/analyze_kpi.py --input inference.csv` |
| `scripts/crop_omni_sdg.py` | `python3 scripts/crop_omni_sdg.py --defect-dir ... --good-dir ...` |
| `scripts/changenet_data_pair_prepare.py` | `python3 scripts/changenet_data_pair_prepare.py --input-dir ... --golden-dir ...` |

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `ParserError: Expected N fields, saw M` | Missing trailing newline when concatenating CSVs | `[ -n "$(tail -c 1 file)" ] && echo "" >> file` before appending |
| Training completes instantly at epoch 0 | Checkpoint resume restored epoch counter | Use weight-loading (not resume) to reset epoch counter |
| Augmented images not found by dataloader | Dataloader expects specific path construction from CSV columns | Data-prep sub-skill must restructure images to match dataloader convention |
| KPI never improves across iterations | Augmented data not targeting actual failure modes | Check gap analysis — ensure SDG covers the actual gap defect types |
| Scores all cluster around 0.5 | Training failed silently or architecture mismatch | Check training logs for errors; use same container/model for train+inference |
| FAR differs between eval sets | Filtered vs full validation set | Always evaluate on the same full validation set |
| Docker output files owned by root | Container runs as root | `sudo chown -R $(whoami)` on results dir after docker run |
| deft-aoi-omniverse-sdg: `OMNI_USER`/`OMNI_PASS` not set | Container tries Nucleus authentication | Set env vars, OR use a local scene from `augmentation/omniverse/scene/` (`find ~/workspace/augmentation/omniverse/scene -name "*.usd"`) — no credentials needed |
| deft-aoi-omniverse-sdg: `PermissionError` creating `trigger_0000` | Host-mounted output dirs not writable by container user | `mkdir -p <out>` and `chmod -R a+rwx <out>` before `docker run` |
| deft-aoi-omniverse-sdg: wrapper starts but script exits with missing `--config` | Script and flags were split incorrectly across argv | Pass the full script invocation as one payload string to the image wrapper |
| deft-aoi-omniverse-sdg: no crops extracted after pipeline | Bbox npy files missing or wrong semantic_types | Ensure `bounding_box_2d_tight: true` and `semantic_types: [class, defect]` in defect config |
| deft-aoi-omniverse-sdg: crop count much lower than expected | High `occlusionRatio` filtering | Lower occlusion threshold in `crop_from_sdg.py` (default 0.5) or check scene framing |
| `crop_from_sdg.py` fails on host but works in container | Host Python missing deps or helper path is host-specific | Run crop extraction inside TAO container: `docker cp` script in, then `docker exec python /tmp/crop_omni_sdg.py ...` |
| Multi-GPU slower than single GPU | DDP sync overhead on small datasets | Use 1 GPU with larger batch size for datasets <1000 rows |
---

## Notes for Implementers

- **Gap analysis is critical.** If SDG can only generate defect types that don't appear in
  validation, augmentation will have no effect on KPI. The agent should warn but still continue.
- **Dataloader compatibility.** The data-prep sub-skill must understand how the training
  dataloader constructs image paths from CSV columns. This varies by framework.
- **Fine-tuning vs resuming.** Ensure the training state (epoch counter, optimizer) is reset
  when fine-tuning. A "resume" mechanism may cause training to terminate immediately.
- **GPU scaling.** Multi-GPU DDP adds sync overhead. For small datasets, prefer 1 GPU with
  larger batch size.
