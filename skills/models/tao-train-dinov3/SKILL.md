---
name: tao-train-dinov3
description: DINOv3 continual self-supervised pre-training. Domain-adapts public DINOv3 ViT backbones
  on unlabeled images via teacher-student self-distillation (DINO + iBOT + KoLeo, optional Gram anchoring)
  and converts the EMA teacher into a timm-format backbone for downstream tasks. Trigger phrases include
  "train DINOv3", "DINOv3 SSL", "domain-adapt a foundation backbone", "continual pretraining", "self-supervised
  finetune DINOv3".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- self
- supervised
- learning
- dinov3
---

# DINOv3

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Continual self-supervised pre-training of DINOv3 vision transformers on unlabeled domain images. Starts from public (timm-format) DINOv3 weights, keeps training with DINO + iBOT + KoLeo (+ optional Gram anchoring), and hands the adapted EMA-teacher backbone to downstream TAO tasks via `dinov3 convert`.

Before choosing spec values, read `references/dinov3-method.md` (method-to-config mapping). Before planning a run or picking a checkpoint, read `references/dinov3-recipes.md` (recipes, evaluation protocol, rules of thumb).

> **TODO(release):** the `tao_toolkit.pyt` image will be updated to a tag containing the `dinov3` entrypoint (family merged to tao-pytorch main 2026-06/07; ViT-S/S+ on 2026-07-08 — 7.0.1-pyt predates it); this skill is not registered in `.claude-plugin/marketplace.json` until then. Release gate: once `versions.yaml` bumps the image (`scripts/stamp_versions.py` re-stamps the pin in `references/skill_info.yaml`), preflight `docker run --rm <image> dinov3 --help`, then register the skill in the marketplace.

## Dataclass Schemas

Generated schemas are packaged in `schemas/<action>.schema.json` with `schemas/manifest.json` listing actions (train, inference, export, convert). They are generated from the dinov3 dataclasses in tao-pytorch (`nvidia_tao_pytorch/config/dinov3/default_config.py`). Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, bounds, enums, and popular parameters. The `references/spec_template_*.yaml` files mirror the shipped experiment specs.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

**Caveat:** the only in-loop AutoML metric is `train_loss` (minimize), and for SSL it is a weak proxy — checkpoints with lower loss have measurably worse downstream transfer (recipes doc, rules 1-2). Prefer milestone-checkpoint selection with downstream probes over loss-driven HPO; use AutoML mainly for stability-range sweeps.

Non-train actions (`inference`, `export`, `convert`) stay in this model skill.

## Training Requirements

- **Dataset type:** image_classification
- **Formats:** ssl — a folder of unlabeled images (recursive; jpg/jpeg/png/ppm/bmp/pgm/tif/tiff/webp). No annotations.
- **Monitoring metric:** train_loss (stability only — see caveat above)
- **Pretrained weights (required):** timm-format DINOv3, gated on Hugging Face — user must accept the `facebook/dinov3-*` license, then `hf download timm/vit_base_patch16_dinov3.lvd1689m --local-dir <weights_dir>`.
- **Data hygiene:** training on unlabeled target-domain images is the normal setup, but eval splits must be disjoint from the SSL corpus by sample and, where leakage is possible, by entity/time/site. A fully held-out domain is a useful optional forgetting probe.

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | dataset.train_dataset.images_dir | train_datasets | images_train.tar.gz | No |
| inference | dataset.test_dataset.images_dir | inference_dataset | images_test.tar.gz | No |

### Typical Spec Overrides

Data source overrides are **mandatory for train and inference**.

**train (Phase 0 — the default recipe):** derive the schedule from the corpus before launching; do not copy fixed step values between runs of different scale.

```python
# Schedule derivation. The dataloader drops the incomplete batch each epoch, so
# steps are counted per-epoch, not from the raw image total:
num_images      = <count of images in the SSL corpus>
global_batch    = 16 * num_gpus * num_nodes            # template batch_size = 16
steps_per_epoch = num_images // global_batch           # drop-last
target_epochs   = max(1, round(15_000 / steps_per_epoch))  # aim mid 10k-20k envelope (recipes doc, rule 4)
total_steps     = target_epochs * steps_per_epoch

{
    "train.num_gpus": num_gpus,
    "train.num_nodes": num_nodes,
    "train.num_epochs": target_epochs,
    "train.checkpoint_interval": total_steps // 6,          # >= 4-8 milestones
    "train.checkpoint_interval_unit": "step",
    "train.pretrained_model_path": "<downloaded timm DINOv3 weights dir>",
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
    # Warmups scale with the schedule (see dinov3-method.md):
    "train.schedulers.learning_rate.warm_up_steps": int(0.07 * total_steps),        # ~5-10%
    "train.schedulers.last_layer_learning_rate.warm_up_steps": int(0.07 * total_steps),
    "train.schedulers.last_layer_learning_rate.freeze_steps": int(0.01 * total_steps),
    "train.schedulers.teacher_temperature.warm_up_steps": int(0.25 * total_steps),
}
```

**local smoke run / AutoML validation:**
```python
{
    "wandb.enable": False,
    "model.backbone.teacher_type": "vit_s",
    "model.backbone.student_type": "vit_s",
    # Required: match the arch. Omitting this trains from random init and does NOT
    # validate the continual-pretraining path (weight remap + teacher mirroring).
    "train.pretrained_model_path": "<downloaded timm vit_small_patch16_dinov3.lvd1689m weights dir>",
    "dataset.batch_size": 8,
    "dataset.workers": 2,
    "train.num_epochs": 1,
    "train.checkpoint_interval": 1,
    "train.num_prototypes": 1024,
    "train.num_gpus": 1,
    "dataset.train_dataset.images_dir": f"{S3_TRAIN}/images_train.tar.gz",
}
```

**convert (the deliverable — run on the selected milestone):**
```python
{
    "convert.checkpoint": "<selected teacher_epoch_*_step_*.pth>",
    "convert.source": "teacher",
    "convert.validate": True,
}
```

**export / inference:** pass the selected stripped `teacher_epoch_*_step_*.pth` as `export.checkpoint` / `inference.checkpoint`, and mirror the train values of `model.backbone.*` and `train.use_custom_attention`.

For high-resolution Phase 1 (experimental), use `references/spec_template_train_highres.yaml` and the gating rules in the recipes doc.

## Eval Dataset

Optional in-loop. SSL quality is judged offline: convert milestone teachers, probe frozen features on disjoint eval splits with a cheap proxy for each target task (kNN on CLS features for global tasks; a small linear head or lightweight decoder on patch features for dense tasks like segmentation, detection, or depth), and require every probe to pass its own directional threshold vs baseline. Protocol: `references/dinov3-recipes.md`.

## Important Parameters

- **model.backbone.teacher_type / student_type**: vit_s | vit_s_plus | vit_b (default) | vit_l | vit_h_plus | vit_7b. Start with vit_b; vit_l when dense-task (segmentation/detection/depth) transfer matters.
- **model.centering_method**: sinkhorn (default). Fall back to softmax on instability/collapse.
- **model.gram.***: Gram anchoring — OFF for Phase 0, ON for high-res Phase 1.
- **model.backbone.rope_theta**: 100.0, parity-critical with timm weights. Never tune.
- **train.schedulers.momentum.val_base**: 0.9999 — the continual-PT anchor. Lower = faster adaptation + faster forgetting.
- **train.precision**: 16-mixed. Do not use bf16 for large backbones (dense-feature regression).
- **dataset.transform.***: crop sizes must be multiples of patch 16.

## Multi-GPU / Multi-Node

**Launch method:** Lightning-managed (single `python` process, Lightning spawns workers).

| Spec Key | Description | Default |
|----------|-------------|---------|
| `train.num_gpus` | Number of GPUs | 1 |
| `train.gpu_ids` | GPU device indices | [0] |
| `train.num_nodes` | Number of nodes | 1 |
| `train.distributed_strategy` | auto / ddp / fsdp | auto |

- LR auto-scales with global batch via the template's `${eval:}` expression; rescale warmup steps when node count changes.
- `sync_batchnorm` is always enabled. Use `fsdp` for vit_h_plus or 768-res runs that OOM under DDP.
- Multi-GPU strongly recommended; the reference industrial recipe used 64 GPUs (global batch 1024).

**Multi-node env vars** (set by orchestrator): `WORLD_SIZE`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `NUM_GPU_PER_NODE`.

## Hardware

vit_b @256, batch 16/GPU fits A100-40GB. 768-res or vit_h_plus: batch 4/GPU, 80GB and/or FSDP recommended. High-res downstream feature extraction over large probe sets can need >200GB host RAM.

## Error Patterns

**Training fails with a determinism/backward error**: xformers memory_efficient_attention has no deterministic backward. `train.cudnn.deterministic` must stay `false` (template default).

**Crop or export size rejected / grid mismatch**: all sizes must be multiples of patch 16 (112, 224, 256, 336, 512, 768).

**Pretrained weights fail to load**: `train.pretrained_model_path` accepts timm-format DINOv3 (dir or file) or TAO DINOv3 checkpoints only — not DINOv2/NVDINOv2 weights. Gated HF download requires accepting the facebook/dinov3-* license.

**Inference/export checkpoint has unexpected Lightning keys**: pass the stripped `teacher_epoch_*_step_*.pth`, not `model_epoch_*.pth` or `dinov3_model_latest.pth`. Full checkpoints are for `train.resume_training_checkpoint_path` only.

**Loss collapses to a constant / features degenerate**: suspect centering first — set `model.centering_method=softmax` and restart from the last good checkpoint.

**CUDA OOM at high resolution**: reduce `dataset.batch_size` (4 at 768), keep `model.gram.teacher_scale: 1.0`, switch `train.distributed_strategy: fsdp`.

**Custom attention fails on this stack**: `train.use_custom_attention: false` falls back to native attention — slower and recipe-relevant; re-validate with a short run (recipes doc, rule 6).

**Resumed run does not reproduce**: the SSL dataloader is not step-resumable; never resume mid-epoch. Resume from checkpoint boundaries or seed a fresh run from a stripped teacher checkpoint.

**AutoML metric not found**: use `train_loss` with minimize direction; `train_loss_epoch` is a fallback alias only.

**Downstream transfer got worse despite lower loss**: expected past the adaptation sweet spot — select an earlier milestone (recipes doc, rules 1-4).

## Spec Param Inference

Model-specific handoff mappings:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| train | `encryption_key` | `key` | encryption key |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | full `model_epoch_*.pth` from the current job |
| export | `encryption_key` | `key` | encryption key |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `results_dir` | `output_dir` | current job results directory |
| convert | `encryption_key` | `key` | encryption key |
| convert | `results_dir` | `output_dir` | current job results directory |

There is no `parent_job_id` or automatic parent-checkpoint resolution mechanism (see `tao-run-platform` orchestration patterns) — and no automatic rule could pick the right file anyway, because the shipped artifact is a *milestone-selected* EMA teacher, not the latest checkpoint. After the train job completes, list `<train results_dir>/train`, evaluate the milestone `teacher_epoch_*_step_*.pth` files per the recipes doc, and set `export.checkpoint` / `inference.checkpoint` / `convert.checkpoint` explicitly to the selected file. Do not patch generated runners to guess checkpoint paths.

## Downstream Handoff

The product of this skill is the converted timm-format teacher backbone (`dinov3 convert`, `validate: true`), consumed by `cv/backbone_v2` registry entries (e.g. `dinov3_vitb16`) via `pretrained_backbone_path` in downstream classification/detection/segmentation tasks. ONNX `export` additionally serves TensorRT consumers.
