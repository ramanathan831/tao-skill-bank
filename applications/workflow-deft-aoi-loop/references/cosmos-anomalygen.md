# Cosmos AnomalyGen — DEFT Loop Reference

Read this when the parent runs the `anomalygen` stage. The underlying skill
`tao-skill-bank:cosmos-anomalygen` (`data/cosmos-anomalygen/SKILL.md`) owns
the standalone 8-phase pipeline and parameter reference. This file is the
DEFT-loop overlay: what to pass, how mounts resolve, the few invariants
that gate the run, and the failure mode the loop has actually hit.

The DEFT loop only needs Phases 2 (`prep_testcase.sh`) and 3 (`run_sdg.sh`).
Phases 4–7 (eval / search / filter+regen) are SDG-quality optimization and
do not contribute to the loop's training pairs. Skip them by setting
`num_search_run=0` and `nn_threshold=0`, or invoke the two wrappers
directly (see *Direct invocation* below).

## Workspace Inputs

Three independent inputs under `<workspace>/augmentation/anomalygen/` plus
the Cosmos base-checkpoints root.

| Input | Path (this workspace; `<project>=UC1`) | Holds |
|---|---|---|
| `checkpoint_dir` | `augmentation/anomalygen/checkpoints/<project>/` | `ag_config.yaml` + `checkpoints/{latest_checkpoint.txt, model/iter_<step>.pt, …}` |
| `dataset_dir` | `augmentation/anomalygen/datasets/<project>/` | Per-defect reference data + `semantic_segmentation_labels.json`. Sibling to `checkpoints/`. |
| `defect_spec` | `augmentation/anomalygen/datasets/<project>/defect_spec.jsonl` | One entry per defect_type (`<T>+<A>`); `spatial_dependency ∈ {free, text, cad}` |
| `cosmos_models_dir` | `${COSMOS_MODELS_DIR}` (resolved by Pre-Flight) | Cosmos base checkpoints — `nvidia/Cosmos-Predict2-2B-Text2Image/`, `google-t5/`, `NVDINOV2/`, … |

`dataset_dir` and `clean_dir` resolve to the same path on this workspace —
clean images live under `<dataset_dir>/<T>/clean_image/` which is the
container's first probe hit. The container handles both flat and
split-by-texture layouts transparently via `validate_amp_inputs.py`; the
loop passes the workspace dir verbatim, no pre-staging.

## Invariants

Verify these before invoking; the rest is up to the container.

1. **`cad_mask` preserves per-class RGB.** `cad2roi` looks up each pixel's
   RGB tuple in `semantic_segmentation_labels.json`. A flattened binary
   `(0,0,0)`/`(255,255,255)` cad_mask yields zero ROIs everywhere (see
   *AMP no-ROI failure mode* below). Verify with
   `Image.open(cad_mask).convert('RGB').getcolors(maxcolors=64)` —
   unique tuples must overlap the labels file.
2. **`defect_spec.jsonl` `text` entries have non-empty
   `roi_prompt_defect_location`.** `cad` and `free` entries don't need it.
3. **`<T>/cad_mask/` and `<T>/clean_image/` are non-empty and paired by
   stem.** Missing pair → record dropped silently.
4. **`semantic_segmentation_labels.json` exists at `datasets/<project>/`.**

Mask file format, image-size agreement, and channel mode do **not** gate
`mode=inference_only` — AMP processes each record at its native size. See
the underlying skill's `references/inference.md` if you need the full
list for `mode=full` / `mode=finetune_only`.

## AMP "no ROI candidates" failure mode

`run_auto_roi_amp.py` silently skips a sample when the cad_mask doesn't
have enough free area for the requested anomaly mask shape. The wrapper
does **not** propagate this — `num_SDG=N` quietly degrades to whatever
AMP could allocate, and the loop only notices via a smaller
`SDG_result.csv`.

Symptoms:

```
WARNING ... <stem>/<T>+<A>: no ROI candidates, skipping
INFO ... <T>+<A>: 0/N with ROI, 0/0 seeds OK
wrote 4 entries to testcase.jsonl       # <-- expected 20
```

Diagnose in this order:

1. **cad_mask class mapping** — invariant #1 above. Most common cause.
2. **Anomaly mask shape vs cad free area** — if the anomaly mask's
   bounding box exceeds every connected component in the cad_mask, AMP
   can't place it. Provide smaller anomaly masks or switch
   `spatial_dependency: free` to skip ROI placement entirely.
3. **Isolate the failing defect** — filter `defect_spec.jsonl` to just
   `<T>+<A>` and re-run `prep_testcase.sh --num-sdg 1`.

After Phase 2, parse `<output_dir>/allocation.json` to confirm per-defect
counts before launching Phase 3 — GPU + model load cost is fixed, so a
4-of-20 yield is worth aborting on.

## DEFT-Loop Parameters

The parent invokes `tao-skill-bank:cosmos-anomalygen` (or the wrappers
directly) with:

| Param | Value | Notes |
|---|---|---|
| `mode` | `inference_only` (or omit when calling wrappers directly) | DEFT loop never runs Phase 1 |
| `checkpoint_dir` | `<workspace>/augmentation/anomalygen/checkpoints/<project>` | |
| `step` | int parsed from `checkpoint_dir/checkpoints/latest_checkpoint.txt` | strip `iter_` prefix and `.pt` suffix |
| `dataset_dir` | `<workspace>/augmentation/anomalygen/datasets/<project>/` | passed verbatim |
| `clean_dir` | same as `dataset_dir` | |
| `defect_spec` | `${dataset_dir}/defect_spec.jsonl` | |
| `num_SDG` | per-iter budget from `deft_state.json` | proportionally allocated across defect types by mask count |
| `num_gpus` | `1` | |
| `model_size` | from `ag_config.yaml` (`2b` or `14b`) | |
| `output_dir` | `${RESULTS_DIR}/iter${N}/anomalygen/sdg/` | receives `reconstructed_image/`, `original_image/`, `SDG_result.csv` |
| `cosmos_models_dir` | `${COSMOS_MODELS_DIR}` | resolved in Pre-Flight |
| `num_search_run` | `0` | skip Phase 5 search rounds |
| `nn_threshold` | `0` | skip Phase 7 filter+regen |

## Direct invocation (the actual two-step path)

The whole loop-relevant pipeline is two `docker run` calls. Use this
form when invoking the skill via the orchestrator is overkill.

```bash
set -a; source <workspace>/.env; set +a
WS=<workspace>
DS=$WS/augmentation/anomalygen/datasets/<project>
CKPT=$WS/augmentation/anomalygen/checkpoints/<project>
COSMOS=$WS/augmentation/anomalygen/base_checkpoints
RUN_DIR=$WS/results/run_<TS>/iter${N}/anomalygen
STEP=$(sed 's/^iter_0*\([0-9]*\)\.pt$/\1/' $CKPT/checkpoints/latest_checkpoint.txt)
: "${AG_IMAGE:=$(${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/resolve_versions_key.py images.metropolis_sdg.cosmos_anomalygen)}"  # reuses Pre-Flight export if set; resolves on demand otherwise

mkdir -p $RUN_DIR/amp $RUN_DIR/sdg

# Phase 2: AMP routing → testcase.jsonl  (~10s, no GPU)
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/workspace/cosmos-anomalygen \
  -v $WS:$WS -v $COSMOS:/workspace/cosmos-anomalygen/checkpoints:ro \
  -w /workspace/cosmos-anomalygen $AG_IMAGE \
  bash -lc "\${ANOMALYGEN_SCRIPTS}/prep_testcase.sh \
    --name iter${N} --num-sdg $NUM_SDG \
    --dataset-dir $DS --clean-dir $DS --defect-spec $DS/defect_spec.jsonl \
    --amp-output-dir $RUN_DIR/amp --output-jsonl $RUN_DIR/testcase.jsonl"

# Phase 3: SDG diffusion → reconstructed_image/ + original_image/  (1-3 min on Blackwell)
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/workspace/cosmos-anomalygen \
  -v $WS:$WS -v $COSMOS:/workspace/cosmos-anomalygen/checkpoints:ro \
  -w /workspace/cosmos-anomalygen $AG_IMAGE \
  bash -lc "\${ANOMALYGEN_SCRIPTS}/run_sdg.sh \
    --checkpoint_dir $CKPT --step $STEP \
    --input_jsonl $RUN_DIR/testcase.jsonl --output_dir $RUN_DIR/sdg \
    --model_size 2b --num_gpus 1"
```

Required mounts: `<workspace>:<workspace>` (same path) +
`<cosmos_models_dir>:/workspace/cosmos-anomalygen/checkpoints:ro`.
Required env: `HF_TOKEN`, `HF_HUB_DISABLE_XET=1`,
`PYTHONPATH=/workspace/cosmos-anomalygen`. Required workdir:
`/workspace/cosmos-anomalygen` (the `-m scripts.…` invocation needs CWD).

## Output layout

```
<output_dir>/
├── SDG_result.csv                          # one row per generated sample (image, mask, params, PSNR)
├── reconstructed_image/<T>+<A>_<idx>.png   # synthetic NG — ChangeNet input_path
├── original_image/<T>+<A>_<idx>.png        # paired OK — ChangeNet golden_path
├── original_mask/, cropped_image/, cropped_mask/, annotated_image/   # intermediates
└── timing_summary.json
```

## Log Stage

```bash
python3 <skill_root>/scripts/log_stage.py \
    --log-path results/loop_log.jsonl \
    --iter-label iter${N} \
    --stage anomalygen --status ok \
    --summary "SDG: requested=N, AMP-allocated=M, generated=K by type"
```

When `M < N` (AMP yield gap), include both requested and allocated counts
— that's the signal a reviewer needs to spot allocation-vs-generation
bottlenecks.
