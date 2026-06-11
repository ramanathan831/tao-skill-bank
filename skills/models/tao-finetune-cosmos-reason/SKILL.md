---
name: tao-finetune-cosmos-reason
description: Cosmos3-Nano video QA supervised fine-tuning with FSDP parallelism. Use when training or evaluating video
  question-answering models, fine-tuning Cosmos3-Nano or compatible Cosmos Reason models with SFT/LoRA, or working with
  Cosmos-RL. Trigger phrases include "fine-tune Cosmos", "Cosmos3 Nano Reasoner", "Cosmos-RL SFT",
  "video QA fine-tune", "Cosmos3-Nano training".
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

Supervised fine-tuning (SFT) of Cosmos Reason video QA models. The packaged
default base model is **hf_model://nvidia/Cosmos3-Nano**. Pretrained weights
are sourced from HuggingFace, not NGC. Gated HuggingFace models require
`HF_TOKEN`. Some Cosmos-RL images cannot load the native Cosmos3 Omni checkpoint
format directly; for those images, convert Cosmos3-Nano to a Qwen3-VL HF
safetensors directory before train/evaluate and use that converted directory as
the PTM path.

Uses FSDP-based parallelism with `dp_shard_size` for GPU count and `dp_replicate_size` for node count (not the standard `num_gpus`/`num_nodes`). Extra references: `cosmos-data-specs.md` for datasets/specs, `cosmos-actions-parameters.md` for eval/parameters/errors, and `cosmos-automl-deft.md` for AutoML/DEFT notes; `detailed-guide.md` is only the map.

Requests for "Cosmos Reason 3", "Cosmos3 Nano Reasoner", or
`nvidia/Cosmos3-Nano` are handled by this skill. There is no separate Cosmos3
model directory in the skill bank; route those requests here. Override the base
HuggingFace model only when the user explicitly asks for a different model.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-run-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate`, `inference`, and `quantize` stay in this model skill. The per-run `automl_policy` override does not change model metadata.

## Credentials

- **HF_TOKEN** (required for gated models): HuggingFace access token. For the
  packaged default, the user must accept the model agreement at
  <https://huggingface.co/nvidia/Cosmos3-Nano> and provide a token with read
  access. If the user explicitly overrides the base model, they must accept
  that target model's agreement too. Passed to the container as a
  `docker_env_var`.

## Cosmos3 Checkpoint Conversion

When a selected image cannot load the native Cosmos3 checkpoint format
(`model_type="cosmos3_omni"` or `Cosmos3ForConditionalGeneration`), do not patch
QwenVL, Transformers, or vLLM first. Use the upstream Cosmos Framework VLM
conversion path to produce a Qwen3-VL HF safetensors directory, then point
Cosmos-RL specs at that converted directory.

The model skill packages a helper:

```bash
python skills/models/tao-finetune-cosmos-reason/scripts/prepare_cosmos3_vlm_checkpoint.py \
  --checkpoint-path /abs/path/Cosmos3-Nano \
  --output-path /abs/path/Cosmos3-Nano-VLM \
  --secrets-env ~/.tao/secrets.env \
  --validate-with-image <cosmos-rl-image>
```

After conversion, use the converted directory consistently as the PTM:

```text
train:    policy.model_name_or_path=/abs/path/Cosmos3-Nano-VLM
evaluate: model.model_name=/abs/path/Cosmos3-Nano-VLM
evaluate: model.base_model_path=/abs/path/Cosmos3-Nano-VLM
```

For local Docker, mount the converted directory read-only into the Cosmos-RL
container and set the spec to the container path. If a converted copy already
exists and validates, reuse it for PTM baseline evaluation, AutoML
recommendations, and final best-checkpoint evaluation rather than converting
again.

## Training Requirements

- **Dataset type:** vlm
- **Formats:** llava, daft
- **Accepted dataset intents:** training, evaluation, testing
- **Monitoring metric:** val/avg_loss, val/reward_avg, val/loss
- **Dataset URI examples:** `s3://bucket/cosmos/train`, `s3://bucket/cosmos/eval`, `/lustre/fsw/tao_datasets/cosmos_rl/train`, `/lustre/fsw/tao_datasets/cosmos_rl/eval`
- **Input modes:** accept either dataset roots or direct spec-key paths. Root mode maps `<root>/annotations.json` plus `<root>` as the media path. Direct spec mode is valid when annotations and media live in different locations, for example `custom.train_dataset.annotation_path=/lustre/.../train.json` and `custom.train_dataset.media_path=/lustre/.../videos.tar.gz`.
- **Media handling:** do not ask the user to choose `videos.tar.gz` vs `images.tar.gz` unless they are using direct spec mode or the model/action requires a single media archive. In root mode, pass the dataset root as the media path.
- **Annotation validation:** before launching train/AutoML/evaluate, verify the
  annotation JSON is readable and the referenced media path or archive is
  visible from the selected platform. Do not block, patch, or mutate
  annotations solely because optional fields are absent.
- **Per-record video FPS:** the packaged train template uses
  `custom.vision.nframes`, so per-record `video_fps` is not required by
  default. If the user switches to `custom.vision.fps`, selects a dataset
  profile that requires per-record timing, or uses an image/version that
  requires `video_fps`, make it a preflight requirement with
  `--json-required-field train_annotation=video_fps` and
  `--json-required-field val_annotation=video_fps` before any download or
  job launch.

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
`skills/models/tao-finetune-cosmos-reason/config.json` file.

For launch preflight, pass the concrete annotation and media paths to the
shared helper:

```bash
scripts/check_tao_launch_preflight.py --platform slurm \
  --path train_annotation=/lustre/.../train/annotations.json \
  --path train_media=/lustre/.../train \
  --path val_annotation=/lustre/.../eval/annotations.json \
  --path val_media=/lustre/.../eval \
  --gpu-min-count 4 \
  --gpu-min-memory-gb 80 \
  --gpu-arch-allowlist cosmos_rl=sm_80,sm_90,sm_100,sm_120
```

For local Docker, pass the resolved Cosmos-RL image so preflight can enforce
NVIDIA runtime, host GPU memory, helper-container, and known image architecture
checks before any model/data download:

```bash
scripts/check_tao_launch_preflight.py --platform local-docker \
  --container-image <resolved-cosmos-rl-image> \
  --path train_annotation=/abs/path/train/annotations.json \
  --path train_media=/abs/path/train \
  --path val_annotation=/abs/path/eval/annotations.json \
  --path val_media=/abs/path/eval
```

For `s3://` paths, if this helper reports that `aws` is missing, ask for
approval and rerun the same command with `--install-missing-tools` so the helper
installs `awscli` and immediately verifies the dataset paths.

Cosmos-RL video datasets can include large `videos.tar.gz` archives. Before
AutoML, stage S3-backed media once to a platform-local/shared path and point
every recommendation at the staged directory or archive; do not let each trial
download the same large S3 object through the container. Prefer an extracted
directory when annotations reference individual files. Keep a
`<workspace>/evaluations/data_staging.json` record with the original S3 URI, the
staged path, and the command/log used to verify the copy.

For Cosmos-RL, count and memory are necessary but not sufficient. Treat the run
as launchable only when the target has at least 4 GPUs with 80GB-class memory or
higher, the GPU architecture is in the image-supported allowlist above, and the
normal Docker/platform, S3, and credential preflight checks pass. A remote image
manifest that advertises `linux/arm64` only proves CPU architecture support; it
does not prove CUDA SM support. Spark/GB10 `sm_121` must be blocked for this
image unless direct image introspection confirms `sm_121` support or the user
chooses a newer compatible image.

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

For DAFT-style annotation files, use direct spec mode when the annotation file
name is not `annotations.json` or when media is not colocated with the
annotation file. Preserve the user's source files. Do not create compatibility
patches for optional annotation fields unless the user explicitly asks for that
dataset mutation.

## Spec construction

cosmos-rl is `mode: config`. **Always start from the packaged
`references/spec_template_<action>.yaml` for the requested action** — load it
as your base spec via `yaml.safe_load(...)` and apply user overrides on top.
Don't rebuild from scratch. See `skills/platform/tao-run-platform/SKILL.md`'s "Constructing the
spec / args" section for the load-template-then-override pattern.

```python
import yaml
from pathlib import Path

skill = Path.home() / "tao-sdk/tao-skills-external/skills/models/tao-finetune-cosmos-reason"
action = "train"  # train, evaluate, inference, or quantize
specs = yaml.safe_load((skill / f"references/spec_template_{action}.yaml").read_text())
# Now apply your overrides on top of `specs` (next section).
```

The reference TOML (and the spec the model actually consumes) is **nested dicts**, not flat dotted keys. The dotted notation in the override examples below denotes *paths into the nested spec* — the agent must walk the path and assign at the leaf, not store the dotted string as a literal key. See `skills/platform/tao-run-platform/SKILL.md`'s "spec is nested dicts" callout.

### Typical Spec Overrides

These are the typical override **paths** to apply on top of the template. Treat
dotted notation as a path into the nested `specs` dict, not as a literal flat
key.

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above.

For direct local Docker runs, mount user data somewhere other than
`/workspace` (for example `/tao-workspace`). The Cosmos-RL image keeps its
Python package under `/workspace/cosmos_rl`; bind-mounting over `/workspace`
hides the package and makes `cosmos-rl` fail with
`ModuleNotFoundError: No module named 'cosmos_rl'`.

For root-style runs, map `TRAIN_DATASET_URI` and `EVAL_DATASET_URI` to:

- train: `custom.train_dataset.annotation_path=<train>/annotations.json`,
  `custom.train_dataset.media_path=<train>`,
  `custom.val_dataset.annotation_path=<eval>/annotations.json`,
  `custom.val_dataset.media_path=<eval>`.
- evaluate: `dataset.annotation_path=<eval>/annotations.json`,
  `dataset.media_dir=<eval>`.
- inference: set `media`, `type`, `prompt`, `model_path`, and `results_dir`.
- quantize: set `dataset.annotation_path`, `dataset.media_dir`,
  `model.model_path`, `quantize.num_calibration_samples`,
  `quantize.max_sequence_length`, and `quantize.quantization_scheme`.

For direct spec-path mode, set the annotation and media fields explicitly rather
than deriving them from a root.

Common train overrides: `policy.model_name_or_path`, `policy.model_max_length`,
`policy.parallelism.dp_shard_size`, `policy.parallelism.dp_replicate_size`,
LoRA settings, `train.epoch`, `train.train_batch_per_replica`,
`train.optm_lr`, `train.train_policy.mini_batch`, checkpoint retention,
validation frequency, and logging. The packaged template keeps
`custom.vision.nframes=8` for bounded 1-GPU memory; switch to `fps` only after
checking token budget and GPU memory.

Do not require per-record `video_fps` for the packaged `nframes` template. If a
run switches to `custom.vision.fps` or a selected dataset/image profile
requires per-record timing, validate the annotation files before launching:

```bash
scripts/check_tao_launch_preflight.py --platform <platform> \
  --path train_annotation=/path/to/train.json \
  --path val_annotation=/path/to/val.json \
  --json-required-field train_annotation=video_fps \
  --json-required-field val_annotation=video_fps
```

The packaged train/evaluate/inference/quantize templates default to
`hf_model://nvidia/Cosmos3-Nano` for base-model fields. Override that only when
the user provides a different HuggingFace model id, `hf_model://...` URI, or
cluster-local snapshot path.

`custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` are
valid train schema fields and are seeded in the packaged train template. Strict
validators must preserve those keys so AutoML can optimize against an explicit
validation set instead of silently falling back to training-only data.

The quantize wrapper includes a compatibility shim for the current image's
`compressed_tensors`/`llmcompressor` import mismatch. Keep that shim in the
model-skill action metadata until the container packages matching versions.

## Critical Overrides (Train)

These are the keys whose template defaults are wrong or where omission flips the run into a different mode:

| Parameter | Template Default | Required Value | Why |
|---|---|---|---|
| `policy.model_name_or_path` | `hf_model://nvidia/Cosmos3-Nano` | Direct Docker: `nvidia/Cosmos3-Nano`, `hf_model://nvidia/Cosmos3-Nano`, or a local HF snapshot path. SDK/managed platform predownload: `hf_model://nvidia/Cosmos3-Nano`. | Keep the train and evaluate base model aligned. |
| `policy.model_max_length` | 40960 | Keep at 40960 or higher | Smaller than ~40k causes `vision_embeds` shape mismatch on video inputs |
| `train.train_batch_per_replica` | 32 | Any multiple of `train.train_policy.mini_batch` | Mismatch raises an immediate AssertionError |
| `train.train_policy.type` | `"sft"` | Keep as `"sft"` for SFT workflows | If dropped during agent regeneration, cosmos-rl flips to RL mode → rollout replica allocated → multi-node attempted → hostname errors when `num_nodes=1` |

## Evaluate

The `actions.evaluate` block in `references/skill_info.yaml` declares the action's inputs (annotation file + media folder + model) and outputs (results directory). For SDK invocation see `skills/platform/tao-run-platform/SKILL.md`.

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
    'model.base_model_path': 'hf_model://nvidia/Cosmos3-Nano',
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

Do not infer dataset paths from prior validation runs. Ask the user for the
train and validation roots or direct spec paths unless a selected workflow
profile explicitly supplies them. Missing optional annotation fields are not a
launch blocker for current Cosmos-RL SFT training.

## AutoML / HPO Notes

The packaged default base model is `hf_model://nvidia/Cosmos3-Nano`. Apply this
base model consistently to train (`policy.model_name_or_path`) and
post-training evaluation (`model.base_model_path`) unless the user explicitly
provides a different HuggingFace model id, `hf_model://...` URI, or
cluster-local snapshot, or converted `Cosmos3-Nano-VLM` directory. If the
conversion helper was required for the selected image, treat the converted
directory as the PTM for the whole run.

Do not hardcode dataset paths in this reusable model skill. Dataset locations
must come from the user's current request, a selected dataset profile, or direct
spec overrides for that run. For a user-provided Cosmos-RL train/eval root, map
the run inputs to concrete spec keys:

```text
custom.train_dataset.annotation_path=<train_root>/annotations.json
custom.train_dataset.media_path=<train_root>
custom.val_dataset.annotation_path=<eval_root>/annotations.json
custom.val_dataset.media_path=<eval_root>
```

When annotation `video` values are relative to a `videos/` subdirectory, use
direct spec mode for `media_path` rather than plain dataset-root mode. If media
is packaged as `videos.tar.gz`, use the extracted `videos/` directory when
present, or the archive only if the selected runtime extracts it before dataset
lookup. Do not edit or patch the user's source annotation files unless the user
explicitly asks for a dataset repair.

If the user's objective names `accuracy` or an accuracy target such as
`>=90%`, optimize an evaluation metric, not `val/avg_loss`. Use AutoMLRunner's
`eval_fn` to run the model skill's `evaluate` action on the validation dataset
after each recommendation, with `task=""`, `model.enable_lora=true`, and
`model.base_model_path` set to the same base model used for training. Return
the evaluator's task metric and set `direction="maximize"`. Use `accuracy` for
constrained classification prompts and BERTScore F1 for free-form
summarization/answering prompts when the user asks for semantic text quality.
Use `val/avg_loss` only when the user accepts a proxy metric or no task metric
is available.

Before launching AutoML for an accuracy objective, run the model's evaluate
action once after preflight and before recommendation jobs on the same
validation subset. Use the selected base model or starting checkpoint,
`task=""`, and the same prompt/metric setup planned for per-recommendation
evaluation. Report that eval job id, result path, and accuracy in the launch
review before asking for confirmation to start recommendations. The final
AutoML summary must compare this baseline accuracy, every recommendation's
accuracy, and the selected best recommendation.

For the evaluator prompt "search over learning rate, batch size, number of
epochs, weight decay, warmup ratio", map the requested knobs to:

```text
learning rate     -> train.optm_lr
batch size        -> train.train_batch_per_replica
number of epochs  -> fixed train.epoch=2 by default; do not include in search unless explicitly requested
weight decay      -> train.optm_weight_decay
warmup ratio      -> fixed train.optm_warmup_epochs=0 by default; do not include in search unless explicitly requested
```

The schema exposes `train.optm_warmup_epochs`, not a native warmup-ratio field.
If the evaluator requires a ratio to be preserved exactly, stop and report that
the current Cosmos-RL schema needs a first-class warmup-ratio parameter.

Example custom ranges for the Cosmos Reason 3 AutoML evaluation prompt:

```python
automl_hyperparameters=[
    "train.optm_lr",
    "train.train_batch_per_replica",
    "train.optm_weight_decay",
]
custom_param_ranges={
    "train.optm_lr": {"valid_min": 1e-5, "valid_max": 1e-3},
    "train.train_batch_per_replica": {
        "value_type": "ordered_int",
        "valid_options": [8, 16, 32],
    },
    "train.optm_weight_decay": {"valid_min": 0.0, "valid_max": 0.1},
}
```

Keep `train.train_policy.mini_batch=1` unless the user explicitly changes it,
so all listed batch sizes remain divisible by the micro-batch size. For small
datasets, cap `train.train_batch_per_replica` so it does not exceed
`floor(num_train_samples / policy.parallelism.dp_shard_size)`. When the
annotation count is known, pass it as `automl_settings["train_sample_count"]`;
current `AutoMLRunner` versions use that to cap invalid batch-size
recommendations before launch and record the adjustment in AutoML history.
For integer knobs with discrete choices, include `value_type: "ordered_int"`
with `valid_options`; integer `valid_options` alone are ignored by the current
Bayesian sampler.

Before launching recommendation jobs, show the user the exact number of
recommendations, search parameters, ranges/defaults, planned dataset subset
size, expected runtime per recommendation, and total expected runtime. If the
first sampled recommendation is available before launch, include its concrete
config. If the estimate exceeds the user's time limit, reduce budget or search
space only after user confirmation.

## Important Parameters

### Training Loop
- **train.epoch**: Number of training epochs. Default 2. Keep it fixed for
  default Cosmos-RL AutoML searches unless the user explicitly asks to tune it.
  Use at least 2 for local AutoML or validation runs that need a host-visible best checkpoint for
  evaluate/inference; one-epoch runs can leave only a broken `best` symlink
  after checkpoint cleanup.
- **train.train_batch_per_replica**: Global batch size per training step. Ideally >= 32 for stability. CRITICAL: must be divisible by `train.train_policy.mini_batch` (default 1 in the packaged smoke-safe template). Recommended production value: 32.
- **train.compile**: Set to true for potential speedup on newer GPUs (H100), else false.
- **train.output_dir**: Output directory for checkpoints and logs.

### Model & Policy
- **policy.model_name_or_path**: HuggingFace model path. The packaged default is `hf_model://nvidia/Cosmos3-Nano`. Override this only when the user provides a different HuggingFace model id, `hf_model://...` URI, or cluster-local snapshot path.
- **policy.model_max_length**: Context window size. Must be 40960 for video SFT. Affected by FPS, resolution, and prompt length.
- **policy.model_gradient_checkpointing**: Save VRAM by recomputing activations. Keep true for large models.

### Parallelism (Multi-GPU / Multi-Node)
- **policy.parallelism.dp_shard_size**: Data-parallel shard size. CRITICAL: should equal **GPUs per node** (the Cosmos-RL equivalent of `num_gpus`).
- **policy.parallelism.dp_replicate_size**: Data-parallel replication = **node count** (equivalent of `num_nodes`). For single-node training set to 1.
- **policy.parallelism.tp_size**: Tensor parallelism. Default 1.
- **policy.parallelism.cp_size**: Context parallelism. Default 1.
- **policy.parallelism.pp_size**: Pipeline parallelism. Default 1.

For multi-node, set `dp_replicate_size = num_nodes` and `dp_shard_size = gpus_per_node`. Cosmos-RL handles the distributed init internally via FSDP — it does **not** rely on the platform-level `MASTER_ADDR` / `WORLD_SIZE` env vars the way `torchrun`-launched jobs do. Just submit with `gpu_count=<gpus_per_node>` and `num_nodes=<N>` on the SDK; the Cosmos-RL spec keys drive the actual sharding.

Training and evaluation can use different Slurm shapes. If the user requests
multi-node training and single-node evaluation, preserve that distinction:
submit train/AutoML recommendations with the requested multi-node shape and run
Cosmos-RL evaluation on the smaller eval shape, usually 1 node with the
requested GPUs per node.

For platform-side multi-node setup (sbatch flags on SLURM, Indexed Job + Service on Kubernetes), see the platform skill's "Multi-node training" section: `skills/platform/tao-run-on-slurm`, `skills/platform/tao-run-on-kubernetes`. Brev and local Docker are single-host only.

### Optimization & Data Loading
- **train.optm_lr**: Learning rate. Default 1e-6.
- **train.train_policy.type**: Training policy. Default `sft`.
- **train.train_policy.mini_batch**: Micro-batch size per GPU. If OOM, reduce this. Constraint: `train_batch_per_replica % mini_batch == 0`.
- **train.train_policy.dataset.name**: Unique ID for dataset cache. IMPORTANT: change this if you modify `fps` or `total_pixels` to force cache regeneration.
- **train.train_policy.dataset.test_size**: Validation split. Float (0.0–1.0) = ratio; Int = absolute number.

### Vision Encoders
- **custom.vision.fps** *or* **custom.vision.nframes** — **mutually exclusive**, set exactly one.
  - `nframes` (default in template): extract this many frames evenly across the clip. This is the safest default for 1-GPU AutoML validation runs.
  - `fps`: extract frames at this rate. High motion: 3. Low motion/static: 1–2. Select this only when the selected videos, `policy.model_max_length`, and GPU memory can absorb the expanded token count.
  - Setting both makes qwen-vl-utils' decord backend error out (`Only accept either fps or nframes`) and silently fall back to torchvision, which deadlocks under multi-worker dataloading (`BlockingIOError [Errno 11]` swscaler errors). If you switch from `fps` to `nframes`, also delete `fps` from your spec.
  - Default Cosmos AutoML search must not tune `custom.vision.fps` while the packaged template uses `custom.vision.nframes`. Only include `custom.vision.fps` when the user explicitly requests FPS tuning and the assembled spec deletes `custom.vision.nframes`.
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
`model.base_model_path=<same base model used for training>` (default
`hf_model://nvidia/Cosmos3-Nano`, or the local base-model snapshot path). For
resume/retrain, pass the exact Cosmos checkpoint policy folder as a string:
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

**You are trying to access a gated repo**: The HuggingFace model `nvidia/Cosmos3-Nano` requires authentication. All ranks will retry in a loop until they time out. Fix: ensure `HF_TOKEN` is set in the process environment or a user-approved secret env file such as `~/.tao/secrets.env` or `~/.config/tao/.env`, verify only presence, and pass it into the container with `-e HF_TOKEN`. The user must also accept the model agreement at <https://huggingface.co/nvidia/Cosmos3-Nano>.

**Cosmos-RL GPU resource and architecture gate**: The actionable launch gate is
at least 4 GPUs with 80GB-class memory or higher, plus a GPU architecture
supported by the selected Cosmos-RL image, plus normal platform, container, S3,
and credential preflight. Run
`scripts/check_tao_launch_preflight.py --gpu-min-count 4 --gpu-min-memory-gb 80 --gpu-arch-allowlist cosmos_rl=sm_80,sm_90,sm_100,sm_120`
before launching. If the target architecture is known but cannot be detected
from the launch host, pass `--gpu-arch sm_XX` explicitly. Spark/GB10 `sm_121`
is not launchable with this image unless image introspection confirms `sm_121`
support or a newer compatible image is selected. If a resource-qualified
platform still fails with a kernel JIT error such as
`nvrtc: invalid --gpu-architecture`, classify it as an image/toolchain defect to
fix with a compatible image, not as a platform resource incompatibility.

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
