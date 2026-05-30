---
name: tao-finetune-cosmos-reason
description: Cosmos-Reason2-8B video QA supervised fine-tuning with FSDP parallelism. Use when training or evaluating video
  question-answering models, fine-tuning Cosmos-Reason2 with SFT, or working with Cosmos-RL. Trigger phrases include
  "fine-tune Cosmos-Reason", "Cosmos-RL SFT", "video QA fine-tune", "Cosmos-Reason2-8B training".
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  author: NVIDIA Corporation
  version: '0.1'
allowed-tools: Read Bash
tags:
- video
- qa
- cosmos
- sft
- reasoning
- vlm
---

# Cosmos-RL

Supervised fine-tuning (SFT) of **nvidia/Cosmos-Reason2-8B** on video reasoning tasks. Pretrained weights are sourced from HuggingFace, not NGC. This is a **gated model** — requires `HF_TOKEN`.

Uses FSDP-based parallelism with `dp_shard_size` for GPU count and `dp_replicate_size` for node count (not the standard `num_gpus`/`num_nodes`).

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, and `quantize` stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Credentials

- **HF_TOKEN** (required): HuggingFace access token. The user must accept the model agreement at <https://huggingface.co/nvidia/Cosmos-Reason2-8B> and provide a token with read access. Passed to the container as a `docker_env_var`.

## Training Requirements

- **Dataset type:** vlm
- **Formats:** llava
- **Accepted dataset intents:** training, evaluation, testing
- **Monitoring metric:** val/avg_loss, val/reward_avg, val/loss
- **Dataset URI examples:** `s3://bucket/cosmos/train`, `s3://bucket/cosmos/eval`, `/lustre/fsw/tao_datasets/cosmos_rl/train`, `/lustre/fsw/tao_datasets/cosmos_rl/eval`
- **Input modes:** accept either dataset roots or direct spec-key paths. Root mode maps `<root>/annotations.json` plus `<root>` as the media path. Direct spec mode is valid when annotations and media live in different locations, for example `custom.train_dataset.annotation_path=/lustre/.../train.json` and `custom.train_dataset.media_path=/lustre/.../videos.tar.gz`.
- **Media handling:** do not ask the user to choose `videos.tar.gz` vs `images.tar.gz` unless they are using direct spec mode or the model/action requires a single media archive. In root mode, pass the dataset root as the media path.
- **Annotation validation:** before launching train/AutoML/evaluate, sample the
  annotation JSON from the selected platform and require `video_fps` in each
  sampled record. Missing `video_fps` causes the Cosmos-RL SFT loader to fail
  with `Error processing sample: 'video_fps'` after the SLURM job starts.

### Launch Intake Reminder

When prompting for Cosmos-RL train or AutoML data, list the actual spec keys as
an option. Users may provide roots, or they may directly provide:

- `custom.train_dataset.annotation_path`
- `custom.train_dataset.media_path`
- `custom.val_dataset.annotation_path`
- `custom.val_dataset.media_path`

For root mode, explain the automatic mapping: `train_root` maps to
`custom.train_dataset.annotation_path=train_root/annotations.json` and
`custom.train_dataset.media_path=train_root`; `eval_root` maps the same way for
`custom.val_dataset`.

Before train or AutoML runner generation, resolve the action=train container
image from `references/skill_info.yaml` and `versions.yaml` (or the packaged
`scripts/resolve_tao_image.py` helper), show the exact image to the user, and
ask whether to use it or override with `image=<override>`. Do not silently
launch on the default image. This skill does not package a
`models/tao-finetune-cosmos-reason/config.json` file.

For launch preflight, pass the concrete annotation paths to the shared helper
and require `video_fps`:

```bash
scripts/check_tao_launch_preflight.py --platform slurm \
  --path train_annotation=/lustre/.../train/annotations.json \
  --path train_media=/lustre/.../train \
  --path val_annotation=/lustre/.../eval/annotations.json \
  --path val_media=/lustre/.../eval \
  --json-required-field train_annotation=video_fps \
  --json-required-field val_annotation=video_fps
```

### Per-Action Dataset Requirements

The packaged Cosmos-RL model action metadata declares **train**, **evaluate**,
**inference**, and **quantize** (`references/skill_info.yaml` and
`schemas/manifest.json`). Do not advertise export, prune, deploy, or dataset
convert for Cosmos-RL unless those actions are added to the model metadata.

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | custom.train_dataset.annotation_path | train_datasets | annotations.json | No |
| train | custom.train_dataset.media_path | train_datasets | dataset root containing media payload | No |
| train | custom.val_dataset.annotation_path | eval_dataset | annotations.json | No |
| train | custom.val_dataset.media_path | eval_dataset | dataset root containing media payload | No |
| evaluate | dataset.annotation_path | eval_dataset | annotations.json | No |
| evaluate | dataset.media_dir | eval_dataset | dataset root containing media payload | No |
| inference | media | inference_dataset | one image/video or a media folder/archive | No |
| quantize | dataset.annotation_path | calibration_dataset | annotations.json | No |
| quantize | dataset.media_dir | calibration_dataset | dataset root containing media payload | No |

## Spec construction

cosmos-rl is `mode: config`. **Always start from the packaged
`references/spec_template_<action>.yaml` for the requested action** — load it
as your base spec via `yaml.safe_load(...)` and apply user overrides on top.
Don't rebuild from scratch. See `platform/tao-run-platform/SKILL.md`'s "Constructing the
spec / args" section for the load-template-then-override pattern.

```python
import yaml
from pathlib import Path

skill = Path.home() / "tao-sdk/tao-skills-external/models/tao-finetune-cosmos-reason"
action = "train"  # train, evaluate, inference, or quantize
specs = yaml.safe_load((skill / f"references/spec_template_{action}.yaml").read_text())
# Now apply your overrides on top of `specs` (next section).
```

The reference TOML (and the spec the model actually consumes) is **nested dicts**, not flat dotted keys. The dotted notation in the override examples below denotes *paths into the nested spec* — the agent must walk the path and assign at the leaf, not store the dotted string as a literal key. See `platform/tao-run-platform/SKILL.md`'s "spec is nested dicts" callout.

### Typical Spec Overrides

These are the typical override **paths** to apply on top of the template (not the full spec). The agent reads each `key.subkey.leaf` as a dotted path and assigns the value at that nested location in the template-loaded `specs` dict.

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above.

For direct local Docker runs, mount user data somewhere other than
`/workspace` (for example `/tao-workspace`). The Cosmos-RL image keeps its
Python package under `/workspace/cosmos_rl`; bind-mounting over `/workspace`
hides the package and makes `cosmos-rl` fail with
`ModuleNotFoundError: No module named 'cosmos_rl'`.

```python
TRAIN_DATASET_URI = "s3://bucket/data/train"
EVAL_DATASET_URI = "s3://bucket/data/eval"
# Slurm/internal example:
# TRAIN_DATASET_URI = "/lustre/fsw/tao_datasets/cosmos_rl/train"
# EVAL_DATASET_URI = "/lustre/fsw/tao_datasets/cosmos_rl/eval"
# Direct spec-path example:
# TRAIN_ANNOTATION_PATH = "/lustre/fsw/.../annotations_train.json"
# TRAIN_MEDIA_PATH = "/lustre/fsw/.../videos_train.tar.gz"
# EVAL_ANNOTATION_PATH = "/lustre/fsw/.../annotations_eval.json"
# EVAL_MEDIA_PATH = "/lustre/fsw/.../eval_videos"
```

**train (mandatory data sources):**
```python
{
    "custom.train_dataset": {
        "annotation_path": f"{TRAIN_DATASET_URI}/annotations.json",
        "media_path": TRAIN_DATASET_URI,
    },
    "custom.val_dataset": {
        "annotation_path": f"{EVAL_DATASET_URI}/annotations.json",
        "media_path": EVAL_DATASET_URI,
    },
    "policy.model_name_or_path": "hf_model://nvidia/Cosmos-Reason2-8B",
    "policy.model_max_length": 81920,
    "policy.parallelism.dp_shard_size": 4,
    "policy.parallelism.dp_replicate_size": 1,
    "policy.lora.lora_alpha": 256,
    "policy.lora.r": 16,
    "policy.lora.lora_dropout": 0.05,
    "train.epoch": 2,
    "train.train_batch_per_replica": 32,
    "train.optm_lr": 2e-5,
    "train.optm_impl": "fused",
    "train.deterministic": True,
    "train.ckpt.save_freq_in_epoch": 1,
    "train.ckpt.max_keep": 2,
    "train.train_policy.mini_batch": 1,
    "train.train_policy.dataset.test_size": 0,
    "train.train_policy.dataloader_num_workers": 4,
    "train.train_policy.dataloader_prefetch_factor": 4,
    "validation.freq_in_epoch": 1,
    "validation.batch_size": 1,
    "validation.enable_dataset_cache": False,
    # custom.vision.nframes defaults to 8 from the spec template for bounded
    # 1-GPU memory use. Switch to fps only when the GPU/context budget supports it.
    "custom.system_prompt": "You are a helpful assistant.",
    "logging.logger": ["console", "tao"],
}
```

`custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` are
valid train schema fields and are seeded in the packaged train template. Strict
validators must preserve those keys so AutoML can optimize against an explicit
validation set instead of silently falling back to training-only data.

**evaluate (mandatory data sources):**
```python
{
    "dataset.annotation_path": f"{EVAL_DATASET_URI}/annotations.json",
    "dataset.media_dir": EVAL_DATASET_URI,
    # vision.nframes defaults to 8 — see Vision Encoders for fps vs nframes.
    "model.enable_lora": True,
    "model.base_model_path": "hf_model://nvidia/Cosmos-Reason2-8B",
}
```

**inference (mandatory media):**
```python
{
    "model_path": "nvidia/Cosmos-Reason2-8B",
    "media": "/tao-workspace/media/videos/example.mp4",
    "type": "video",
    "prompt": "Briefly describe this video.",
    "fps": 1,
    "total_pixels": 200704,
    "max_new_tokens": 32,
    "results_dir": "/results/inference",
    "enable_lora": False,
}
```

**quantize (mandatory calibration data):**
```python
{
    "model.model_path": "nvidia/Cosmos-Reason2-8B",
    "dataset.annotation_path": "/tao-workspace/calibration/annotations.json",
    "dataset.media_dir": "/tao-workspace/calibration/videos",
    "quantize.num_calibration_samples": 1,
    "quantize.max_sequence_length": 4096,
    "quantize.quantization_scheme": "W4A16",
    "quantize.skip_test_generation": True,
    "results_dir": "/results/quantize",
}
```

The quantize wrapper includes a compatibility shim for the current image's
`compressed_tensors`/`llmcompressor` import mismatch. Keep that shim in the
model-skill action metadata until the container packages matching versions.

## Critical Overrides (Train)

These are the keys whose template defaults are wrong or where omission flips the run into a different mode:

| Parameter | Template Default | Required Value | Why |
|---|---|---|---|
| `policy.model_name_or_path` | `nvidia/Cosmos-Reason2-8B` | Direct Docker: `nvidia/Cosmos-Reason2-8B` or a local HF snapshot path. SDK/managed platform predownload: `hf_model://nvidia/Cosmos-Reason2-8B`. | The direct `cosmos-rl --config ...` CLI loads from HuggingFace at runtime and does not need the SDK URI wrapper. The `hf_model://` URI form is for SDK/platform launchers that pre-download model inputs before the container command starts. |
| `policy.model_max_length` | 40960 | Keep at 40960 or higher | Smaller than ~40k causes `vision_embeds` shape mismatch on video inputs |
| `train.train_batch_per_replica` | 32 | Any multiple of `train.train_policy.mini_batch` | Mismatch raises an immediate AssertionError |
| `train.train_policy.type` | `"sft"` | Keep as `"sft"` for SFT workflows | If dropped during agent regeneration, cosmos-rl flips to RL mode → rollout replica allocated → multi-node attempted → hostname errors when `num_nodes=1` |

## Evaluate

The `actions.evaluate` block in `references/skill_info.yaml` declares the action's inputs (annotation file + media folder + model) and outputs (results directory). For SDK invocation see `platform/tao-run-platform/SKILL.md`.

### Config format

The evaluator reads a **flat TOML** config with top-level keys: `dataset`,
`model`, `task`, `evaluation`, `vision`, `generation`, `metrics`, `results`,
`num_gpus`, and `results_dir`. The defaults template
(`references/spec_template_evaluate.yaml`) matches this flat structure. Use
dotted overrides such as `dataset.annotation_path`, `model.model_name`, and
`evaluation.batch_size`.

### Task type

- Empty string (`""`) — General Evaluator. Auto-detects binary classification (yes/no) from ground truth and computes TP/FP/TN/FN/accuracy/precision/recall/F1.
- `"its_directionality"` — ITS-specific evaluator for left/right/straight classification. Do NOT use for collision detection.

### LoRA Evaluation

To evaluate a fine-tuned LoRA model, pass the checkpoint path via spec_overrides:

```python
spec_overrides={
    'model.model_name': 's3://bucket/results/{train_job_id}/safetensors/epoch_2',
    'model.enable_lora': True,
    'model.base_model_path': 'nvidia/Cosmos-Reason2-8B',
    'evaluation.batch_size': 10,
}
```

The LoRA adapter is downloaded from S3/Lustre before the evaluator runs; the evaluator merges it with the base model and runs inference on the merged weights.

### Selective download

When the input declaration carries a `selective` block (`{annotation, format, keys}`), only the files referenced in `dataset.annotation_path` (under the `video` key) are pulled — not the full media folder. For a 112-sample collision dataset, this downloads ~500MB instead of the full 4.8GB folder.

### Results

- `results.json` — per-sample predictions with `video_id`, `response`, `question`, `gt`
- Binary metrics: accuracy, balanced accuracy, precision, recall, F1
- Text metrics: BLEU, ROUGE, BERTScore
- When Lustre is available, results write to Lustre for cross-job persistence (e.g., gap analysis reads directly), then upload to S3.

## Datasets

The `data_sources` config in config.json maps dataset URIs to spec paths. It
appends `annotations.json` to the dataset directory URI by convention. If your
annotations and media do not share a root, or if the annotation file has a
different name, use direct spec overrides instead of forcing a root:

```python
spec_overrides={
    'custom.train_dataset': {
        'annotation_path': 's3://bucket/train/my_annotations.json',
        'media_path': 's3://bucket/media/videos_train.tar.gz',
    },
    'custom.val_dataset': {
        'annotation_path': 's3://bucket/eval/my_annotations.json',
        'media_path': 's3://bucket/eval/videos/',
    },
}
```

**Eval dataset** is optional for plain training only when `train.train_policy.dataset.test_size` is used to auto-split training data. For AutoML or any workflow optimizing a validation metric such as `val/avg_loss`, require either an explicit `custom.val_dataset` or a deliberate auto-split setting before launch preflight passes. If a validation dataset is provided, validation metrics are computed at the frequency set by `validation.freq_in_epoch`.

Every sampled annotation record must include `video_fps`. If this field is
absent, stop before runner generation and ask the user to add it to the train
and validation annotation files or provide corrected direct spec paths. Do not
start AutoML to discover this inside torchrun.

## AutoML / HPO Notes

When the user asks for "Cosmos Reason 3", "Cosmos3 Nano Reasoner", or
`nvidia/Cosmos3-Nano-Reasoner`, route the request to this `cosmos-rl` skill and
override the base model to `nvidia/Cosmos3-Nano-Reasoner` unless the user
provides a different HuggingFace model id, `hf_model://...` URI, or
cluster-local snapshot. The packaged template default is still
`nvidia/Cosmos-Reason2-8B`; do not silently launch a Reason 3 request on
Reason2 weights. Apply the same base model override consistently to train
(`policy.model_name_or_path`) and post-training evaluation
(`model.base_model_path`).

Do not hardcode dataset paths in this reusable model skill. Dataset locations
must come from the user's current request, a selected dataset profile, or direct
spec overrides for that run. For a user-provided Cosmos-RL train/eval root, map
the run inputs to concrete spec keys:

```text
custom.train_dataset.annotation_path=<train_root>/annotations.json
custom.train_dataset.media_path=<train_root>/videos
custom.val_dataset.annotation_path=<eval_root>/annotations.json
custom.val_dataset.media_path=<eval_root>/videos
```

When annotation `video` values are relative to a `videos/` subdirectory, use
direct spec mode for `media_path` rather than plain dataset-root mode. If media
is packaged as `videos.tar.gz`, use the extracted `videos/` directory when
present, or the archive only if the selected runtime extracts it before dataset
lookup. If the original annotation files do not contain `video_fps`, create
patched annotation copies under the run workspace and point
`custom.*.annotation_path` at those copies; do not edit the user's source dataset
in place.

If the user's objective names `accuracy` or an accuracy target such as
`>=90%`, optimize an evaluation metric, not `val/avg_loss`. Use AutoMLRunner's
`eval_fn` to run the model skill's `evaluate` action on the validation dataset
after each recommendation, with `task=""`, `model.enable_lora=true`, and
`model.base_model_path` set to the same base model used for training. Return
the evaluator's `accuracy` value and set `direction="maximize"`. Use
`val/avg_loss` only when the user accepts a proxy metric or no task metric is
available.

For the evaluator prompt "search over learning rate, batch size, number of
epochs, weight decay, warmup ratio", map the requested knobs to:

```text
learning rate     -> train.optm_lr
batch size        -> train.train_batch_per_replica
number of epochs  -> train.epoch
weight decay      -> train.optm_weight_decay
warmup ratio      -> train.optm_warmup_epochs, computed as round(train.epoch * ratio)
```

The schema exposes `train.optm_warmup_epochs`, not a native warmup-ratio field.
If the evaluator requires a ratio to be preserved exactly, stop and report that
the current Cosmos-RL schema needs a first-class warmup-ratio parameter.

Example custom ranges for the Cosmos Reason 3 AutoML evaluation prompt:

```python
automl_hyperparameters=[
    "train.optm_lr",
    "train.train_batch_per_replica",
    "train.epoch",
    "train.optm_weight_decay",
    "train.optm_warmup_epochs",
]
custom_param_ranges={
    "train.optm_lr": {"valid_min": 1e-5, "valid_max": 1e-3},
    "train.train_batch_per_replica": {
        "value_type": "ordered_int",
        "valid_options": [8, 16, 32],
    },
    "train.epoch": {
        "value_type": "ordered_int",
        "valid_options": [3, 5, 10],
    },
    "train.optm_weight_decay": {"valid_min": 0.0, "valid_max": 0.1},
    "train.optm_warmup_epochs": {
        "value_type": "ordered_int",
        "valid_options": [0, 1, 2, 3, 4, 5],
    },
}
```

Keep `train.train_policy.mini_batch=1` unless the user explicitly changes it,
so all listed batch sizes remain divisible by the micro-batch size. For small
datasets, also cap `train.train_batch_per_replica` so it does not exceed
`num_train_samples / policy.parallelism.dp_shard_size`.
For integer knobs with discrete choices, include `value_type: "ordered_int"`
with `valid_options`; integer `valid_options` alone are ignored by the current
Bayesian sampler.

## Important Parameters

### Training Loop
- **train.epoch**: Number of training epochs. Default 10. Use at least 2 for
  local smoke or AutoML runs that need a host-visible best checkpoint for
  evaluate/inference; one-epoch runs can leave only a broken `best` symlink
  after checkpoint cleanup.
- **train.train_batch_per_replica**: Global batch size per training step. Ideally >= 32 for stability. CRITICAL: must be divisible by `train.train_policy.mini_batch` (default 1 in the packaged smoke-safe template). Recommended production value: 32.
- **train.compile**: Set to true for potential speedup on newer GPUs (H100), else false.
- **train.output_dir**: Output directory for checkpoints and logs.

### Model & Policy
- **policy.model_name_or_path**: HuggingFace model path. The packaged default is `nvidia/Cosmos-Reason2-8B`. For a Cosmos Reason 3 evaluation, override this with `nvidia/Cosmos3-Nano-Reasoner` or the exact user-provided Reason 3 `hf_model://...` URI or cluster-local snapshot path; do not rely on the Reason2 default.
- **policy.model_max_length**: Context window size. Must be 40960 for video SFT. Affected by FPS, resolution, and prompt length.
- **policy.model_gradient_checkpointing**: Save VRAM by recomputing activations. Keep true for large models.

### Parallelism (Multi-GPU / Multi-Node)
- **policy.parallelism.dp_shard_size**: Data-parallel shard size. CRITICAL: should equal **GPUs per node** (the Cosmos-RL equivalent of `num_gpus`).
- **policy.parallelism.dp_replicate_size**: Data-parallel replication = **node count** (equivalent of `num_nodes`). For single-node training set to 1.
- **policy.parallelism.tp_size**: Tensor parallelism. Default 1.
- **policy.parallelism.cp_size**: Context parallelism. Default 1.
- **policy.parallelism.pp_size**: Pipeline parallelism. Default 1.

For multi-node, set `dp_replicate_size = num_nodes` and `dp_shard_size = gpus_per_node`. Cosmos-RL handles the distributed init internally via FSDP — it does **not** rely on the platform-level `MASTER_ADDR` / `WORLD_SIZE` env vars the way `torchrun`-launched jobs do. Just submit with `gpu_count=<gpus_per_node>` and `num_nodes=<N>` on the SDK; the Cosmos-RL spec keys drive the actual sharding.

For platform-side multi-node setup (sbatch flags on SLURM, Indexed Job + Service on Kubernetes), see the platform skill's "Multi-node training" section: `platform/tao-run-on-slurm`, `platform/tao-run-on-kubernetes`. Brev and local Docker are single-host only.

### Optimization & Data Loading
- **train.optm_lr**: Learning rate. Default 1e-6.
- **train.train_policy.type**: Training policy. Default `sft`.
- **train.train_policy.mini_batch**: Micro-batch size per GPU. If OOM, reduce this. Constraint: `train_batch_per_replica % mini_batch == 0`.
- **train.train_policy.dataset.name**: Unique ID for dataset cache. IMPORTANT: change this if you modify `fps` or `total_pixels` to force cache regeneration.
- **train.train_policy.dataset.test_size**: Validation split. Float (0.0–1.0) = ratio; Int = absolute number.

### Vision Encoders
- **custom.vision.fps** *or* **custom.vision.nframes** — **mutually exclusive**, set exactly one.
  - `nframes` (default in template): extract this many frames evenly across the clip. Use at least 2; qwen-vl-utils rejects `nframes=1`. This is the safest default for 1-GPU AutoML smoke runs.
  - `fps`: extract frames at this rate. High motion: 3. Low motion/static: 1–2. Use when the selected videos, `policy.model_max_length`, and GPU memory can absorb the expanded token count.
  - Setting both makes qwen-vl-utils' decord backend error out (`Only accept either fps or nframes`) and silently fall back to torchvision, which deadlocks under multi-worker dataloading (`BlockingIOError [Errno 11]` swscaler errors). If you switch from `fps` to `nframes`, also delete `fps` from your spec.
- **custom.vision.total_pixels**: Resolution constraint. Increase if the object of focus is small relative to the frame. Default 3136000.
- **custom.system_prompt**: Instructions prepended to every prompt.

### Checkpointing
- **train.ckpt.save_freq_in_epoch**: Save every N epochs. Default 1.
- **train.ckpt.max_keep**: Keep N most recent checkpoints. Default 2 for
  AutoML/minimal runs so the best LoRA adapter remains available even when the
  container records the best validation step before later epoch cleanup.
- **train.ckpt.export_safetensors**: Export in safetensors format. Default true.

When verifying downstream handoff, prefer `train_output_dir/best/safetensors`
only if it resolves inside the results mount. In the current local Docker image,
epoch-based saving writes concrete artifacts under
`train_output_dir/<timestamp>/safetensors/epoch_N` and
`train_output_dir/<timestamp>/checkpoints/epoch_N/policy`, but
`best/best_score.json` and the `best/{safetensors,checkpoints}` symlinks can
record `step_N` targets that do not exist. If a `best` symlink points at a
missing `step_*` directory, resolve the best validation step back to the
corresponding retained `epoch_*` directory and use that exact folder. Do not
fall back to "latest" silently.

For evaluate, pass the resolved LoRA folder directly:
`model.model_name=<train_output_dir>/<timestamp>/safetensors/epoch_N`,
`model.enable_lora=true`, and
`model.base_model_path=nvidia/Cosmos-Reason2-8B` (or the local base-model
snapshot path). For resume/retrain, pass the exact Cosmos checkpoint policy
folder as a string:
`train.resume=<train_output_dir>/<timestamp>/checkpoints/epoch_N/policy`.
Avoid `train.resume=true` for local Docker epoch-based checkpoints because the
current resolver scans `step_*` checkpoint directories and can miss the
`epoch_*` folders. Do not count downloaded base-model shards under `ptm/`,
launcher staging files under `inputs/`, or the broken `best` symlink itself as
fine-tuned checkpoints for handoff.

### Validation
- **validation.freq_in_epoch**: Run validation every N epochs. Too frequent slows training.

### Logging
- **logging.logger**: Options: `console`, `wandb`.
- **logging.project_name** / **logging.experiment_name**: W&B experiment tracking.

## Hardware

Cosmos-RL models are 8B parameters and benefit from multi-GPU training with FSDP sharding. `dp_shard_size` should equal total GPU count. Recommended: 8x A100 or H100 (80GB each).

## Error Patterns

**CUDA out of memory (train)**: Reduce `train.train_policy.mini_batch` or increase `dp_shard_size`. Enable `fsdp_offload` if GPU memory is limited. Also check `custom.vision.total_pixels` — high resolution increases memory significantly.

**OOM during evaluation with LoRA**: Loading the base model + LoRA adapter uses more memory than zero-shot eval. If zero-shot eval passes but post-training eval OOMs, reduce `evaluation.batch_size` (e.g., from 10 to 1) or lower `vision.total_pixels`. The OOM typically manifests as the node killing the process mid-run (no Python traceback — just `ERR_PROGRAM` with a node-level OOM event). This is especially likely in DEFT workflows where the same eval spec is used for both zero-shot and post-training evaluation.

**NaN loss**: Learning rate may be too high. Reduce `optm_lr` and increase `optm_warmup_epochs`.

**vision_embeds.shape[0] must be equal to n_tokens**: `model_max_length` is too small for the video input at the current FPS and resolution. Increase `policy.model_max_length` to 40960.

**Quantize image/video token mismatch**: `Mismatch in image token count between
text and input_ids` during calibration means `quantize.max_sequence_length` is
too small for the sampled media tokens. The packaged smoke template uses 4096;
do not lower it to tiny values such as 128 for video calibration.

**train_batch_per_replica not divisible by mini_batch**: The default `train_batch_per_replica=1` from the TAO Core schema is invalid because `mini_batch` defaults to 4. Immediate AssertionError on all ranks. Fix: set `train_batch_per_replica` to a multiple of `mini_batch` (recommended: 32 for large datasets, 4 for small datasets).

**train_batch_per_replica larger than samples per rank**: With FSDP, each rank sees `total_samples / dp_shard_size` samples. If `train_batch_per_replica` exceeds this, the trainer completes 0 training steps and attempts to save a checkpoint before the optimizer/scheduler is initialized, crashing with `'NoneType' object has no attribute 'state_dict'`. Fix: ensure `train_batch_per_replica <= total_samples / dp_shard_size`. For small datasets (e.g., 31 DEFT-generated samples on 8 GPUs = ~4 per rank), set `train_batch_per_replica` to 4.

**Stale dataset cache after changing fps/total_pixels**: Change `train.train_policy.dataset.name` to a new unique identifier to force cache regeneration.

**Checkpoint save failure (scheduler is None)**: The cosmos-rl trainer crashes with `'NoneType' object has no attribute 'state_dict'` when saving a checkpoint before any training step has executed. This happens when the dataset is too small for the batch size (0 steps per epoch). See the batch size error above.

**You are trying to access a gated repo**: The HuggingFace model `nvidia/Cosmos-Reason2-8B` requires authentication. All ranks will retry in a loop until they time out. Fix: ensure `HF_TOKEN` is set in your environment (e.g., in `~/.config/tao/.env`) and passed into the container with `-e HF_TOKEN`. The user must also accept the model agreement at <https://huggingface.co/nvidia/Cosmos-Reason2-8B>.

**TAO_API_JOB_ID status logging warnings in direct Docker**: `cosmos-rl-evaluate`, `cosmos-rl-inference`, and `cosmos-rl-quantize` may log a traceback from `tao_status_logger.py` when `TAO_API_JOB_ID` is unset. For direct local-Docker model-skill validation this is nonfatal if the process exits 0 and the action writes its expected result files. Do not hide a real action failure behind this warning, but do not mark an otherwise successful local run failed only because status-file logging was unavailable.

## DEFT Support

Cosmos-RL implements the DEFT workflow contract for video QA tasks. Use the
packaged model metadata and `workflow/deft/deft.md` for the pipeline overview;
this skill does not package a `config.json`.

### Gap Analysis (`scripts/analyze_gaps.py`)

Model-specific script that identifies failure cases from cosmos-rl evaluation output.

- **Eval output format:** `results.json` with fields: `video_id`, `response`, `question`, `gt`
- **Comparison:** exact string match after `.lower().strip()` — requires eval prompts that force short constrained answers (e.g., yes/no)
- **Output:** parquet with `video_id` (full path), `question`, `ground_truth`

**Limitation:** Brittle exact match. If the model responds with full sentences instead of constrained answers, mismatches will be over-reported. The eval prompt design must account for this.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

- **Checkpoint metadata:** format: safetensors, folder: true

Inference mappings from TAO Core `cosmos-rl.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `model.model_name` | `parent_model_folder` | model folder inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| inference | `model_path` | `parent_model_folder` | model folder inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `model.model_path` | `parent_model_folder` | model folder inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.output_dir` | `output_dir` | current job results directory |
| train | `train.resume` | `resume_model` | exact checkpoint policy folder inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
