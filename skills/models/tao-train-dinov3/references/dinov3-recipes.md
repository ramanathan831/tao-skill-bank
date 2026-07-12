# DINOv3 continual-SSL recipes, evaluation protocol, and rules of thumb

General guidance for continual self-supervised adaptation of DINOv3 backbones, informed by
internal industrial experiments across multiple backbone sizes (ViT-B/L/H+) and resolutions
(224-768px).

> **Caveat:** these are empirical observations from a limited set of domains and schedules,
> not laws. Any given rule may or may not apply to your data, backbone size, or compute
> budget — treat them as priors worth keeping in mind, and let your own milestone evaluations
> (see the protocol below) have the final word.

## The rules

1. **Never ship the last checkpoint.** Save step-unit milestone checkpoints and evaluate
   several. Downstream quality typically peaks mid-schedule; later checkpoints can regress
   held-out domains even while train loss keeps improving.
2. **Loss curves do not validate SSL.** A slowly decreasing train loss (typical range ~18 -> ~17
   for continual PT) only proves stability. Every quality decision must come from downstream
   probes on *frozen* features — a cheap proxy for whatever the target task is (e.g. kNN for
   classification/retrieval, a small linear head or lightweight decoder for dense tasks such
   as segmentation, detection, or depth).
3. **Score with a strict per-probe envelope, not a mean.** For each probe, record its metric
   direction and compute a directional delta vs the baseline: `metric - baseline` for
   higher-is-better (top-1, mAP, mIoU), `baseline - metric` for lower-is-better (RMSE,
   abs-rel). Give each probe its own pass threshold (typically `delta >= 0`, or a small
   agreed tolerance for probes you are willing to trade). A checkpoint passes only if
   **every** probe passes its own threshold. Never average across probes (a mean lets one
   probe's gain hide another's regression) and never take a raw minimum across probes with
   different scales or directions — compare deltas only within a single probe.
4. **More domain SSL is not better.** Held-out domains drift first while in-domain metrics keep
   climbing. Budget steps modestly (order 10k-20k optimizer steps at global batch ~1024 is a
   sensible starting envelope for a corpus of a few hundred thousand images) and rely on
   milestone selection.
5. **Separate SSL training data from evaluation data.** Training on unlabeled target-domain
   images and evaluating on a labeled split of that same domain is the normal adaptation
   setup — but the eval split must be disjoint from the SSL corpus at the sample level and,
   where leakage is possible, by entity/time/site (same part, board, patient, or capture
   session must not appear on both sides). Additionally, a *fully held-out* domain is a
   valuable optional forgetting/generalization probe: it is where over-adaptation shows up
   first (rule 4).
6. **Do not reuse a small-model recipe on a bigger backbone unchanged.** Recipes that adapt one
   size cleanly can severely regress a larger one. Re-validate per size with a short run
   before committing compute.
7. **Prefer `precision: 16-mixed` over bf16 for large backbones.** bf16 can measurably degrade
   dense (patch-level) feature quality at high resolution — hurting any dense task — while
   global-task metrics barely notice.
8. **Evaluate and convert the teacher (EMA), never the student.** Shipping candidates are
   `teacher_epoch_*_step_*.pth`; `convert.source` defaults to `teacher` for this reason.
9. **Never resume mid-epoch.** The SSL dataloader is not step-resumable; a mid-epoch resume is
   not a trusted reproduction of the original trajectory. Resume only from checkpoint
   boundaries via `train.resume_training_checkpoint_path`, and prefer seeding a *fresh* run
   from a stripped teacher checkpoint (`train.pretrained_model_path`) when changing recipe.
10. **Checkpoint interpolation cannot rescue a Pareto tradeoff.** Linear weight interpolation
    between a balanced early checkpoint and a drifted later one rarely contains a useful
    point; do not spend eval budget on it.
11. **High-res continuation is high-risk.** A high-resolution phase (with or without Gram) can
    trade held-out quality for in-domain gains. Run Phase 1 only with a dense-task reason,
    keep it short, and always A/B the result against the Phase-0 incumbent under the envelope.
12. **Backbone sizing:** continual SSL tends to improve global-task transfer at every size,
    with the largest gains where the official weights start weakest. vit_b is the efficient
    default; vit_l is the safest all-rounder when dense-task transfer (segmentation,
    detection, depth) matters; vit_h_plus demands fp16-mixed and per-size retuning.

## Phase structure

**Phase 0 — domain adaptation at base resolution (the workhorse).**
`spec_template_train.yaml` as-is: img 256, crops 256/112, Gram OFF, sinkhorn, EMA 0.9999,
LR `5e-5 * sqrt(global_batch/1024)`, `checkpoint_interval_unit: step` with an interval that
yields >= 4-8 milestones across the schedule. Rescale all warmups to the actual total step
count (~5-10% for LR warmup; keep the teacher-temperature warmup proportionally longer, and
`freeze_steps` ~1-2% of total). Worked example at global batch 1024 and ~20k total steps:
LR warmup ~1250, teacher-temp warmup ~4700, freeze_steps ~150, milestones every ~2500-5000.

**Phase 1 — high-resolution adaptation (EXPERIMENTAL).**
`spec_template_train_highres.yaml`: seed `pretrained_model_path` with the *selected Phase-0
stripped teacher checkpoint*, img/global 768 (or 512), locals 336 (or 224), batch 4/GPU,
Gram ON (`w_gram 2.0`, `teacher_source: ema`, `refresh_interval 10000`, `teacher_scale 1.0`),
short (~2 epochs), warmup 2000. Gate adoption on the envelope beating the Phase-0 incumbent
(rule 11).

## Evaluation protocol (between/after phases)

1. At each milestone, `dinov3 convert` the teacher to timm safetensors (`validate: true`).
2. Probe frozen features on >= 2 downstream probes whose eval splits are sample/entity/
   time-site disjoint from the SSL corpus (rule 5) — ideally including one fully held-out
   domain as a forgetting probe — using a cheap proxy for each task you actually care about: kNN (e.g. k=5) on summary/CLS
   features for global tasks (classification, retrieval, anomaly detection); a small linear
   head or lightweight decoder on patch features for dense tasks (segmentation, detection,
   depth). Probe at deployment resolution (both 224 and 512 if unsure).
3. Score every checkpoint with the per-probe envelope (rule 3) — directional delta and pass
   threshold per probe — against the official same-size DINOv3 baseline evaluated under the
   *identical* protocol.
4. Select the milestone, record baseline + scores next to the run, and only then decide
   whether a Phase 1 is worth attempting.

## Checkpoint file taxonomy (results_dir/train)

| File | What it is | Use for |
|---|---|---|
| `model_epoch_XXX_step_YYYYY.pth` | full Lightning checkpoint | `train.resume_training_checkpoint_path` only |
| `dinov3_model_latest.pth` | link/copy of latest full checkpoint | resume convenience; never evaluate/ship |
| `teacher_epoch_XXX_step_YYYYY.pth` | stripped EMA-teacher backbone | evaluate, `convert`, `export`, Phase-1 seed |
| `student_epoch_XXX_step_YYYYY.pth` | stripped student backbone | diagnostics only |

## Hardware sizing

- Train ViT-B @256, batch 16/GPU: fits A100-40GB with xformers (`use_custom_attention: true`).
- Train @768: attention is ~9x the 256 cost — batch 4/GPU, consider
  `train.distributed_strategy: fsdp`; keep `gram.teacher_scale: 1.0`.
- If xformers custom attention fails on your stack, `use_custom_attention: false` falls back to
  native attention — slower and *recipe-relevant*; re-validate with a short run after
  switching (rule 6).
- Downstream high-res feature extraction over large probe sets is host-RAM hungry (feature
  caching can require hundreds of GB); size eval jobs accordingly.
