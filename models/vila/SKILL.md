---
name: vila
description: VILA vision-language model for multimodal understanding tasks. Supports video and image-based question answering,
  captioning, and reasoning. Fine-tunable with LoRA or full fine-tuning on custom VLM datasets.
license: Apache-2.0
compatibility: Requires docker + nvidia-container-toolkit.
metadata:
  version: '0.1'
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- vision
- language
- model
---

# VILA

VILA vision-language model for multimodal understanding tasks. Supports video and image-based question answering, captioning, and reasoning. Fine-tunable with LoRA or full fine-tuning on custom VLM datasets.

Set `model_path` to a base VILA/NVILA model checkpoint. The checkpoint must be
a VILA wrapper model whose `config.json` has `model_type: llava_llama`, such as
`Efficient-Large-Model/NVILA-Lite-2B` or a local folder with the same structure.
Raw Qwen2.5-VL checkpoints with `model_type: qwen2_5_vl` are not compatible with
the current VILA fine-tuning entrypoints.

## Dataclass Schemas

Generated TAO Core schemas are packaged in `schemas/<action>.schema.json`, with `schemas/manifest.json` listing available actions. Each generated schema also emits `references/spec_template_<action>.yaml` from the schema top-level `default` field. AutoML enablement is declared at the model layer in `references/skill_info.yaml` via `automl_enabled`. Runnable AutoML still requires `schemas/train.schema.json` and `references/spec_template_train.yaml` to exist and parse. Use the packaged train schema for `automl_default_parameters`, `automl_disabled_parameters`, defaults, min/max bounds, enums, option weights, math conditions, dependencies, and popular parameters. Do not expect `~/tao-core` at runtime; maintainers regenerate schemas/templates before packaging the skill bank.

## Train Action Policy

This model is AutoML-enabled at the model layer. Before handling any train-stage request, read `references/skill_info.yaml` and resolve the run override from either an explicit `automl_policy` value or the user's workflow request. Use `automl_policy: on` by default and only expose `on` / `off` in new launch prompts. Treat phrases like "turn off AutoML", "disable AutoML", "no HPO", or "plain training" as `automl_policy: off` for this run only. When `automl_policy: on`, `automl_enabled: true`, and both `schemas/train.schema.json` and `references/spec_template_train.yaml` are packaged, route the train action through `tao-skill-bank:tao-automl` by default with this model's `skill_dir`. Preserve workflow/application overrides for datasets, specs, output directories, GPU/platform settings, parent checkpoints, and `automl_policy`. Use direct model training only when `automl_policy: off` or the packaged train schema/template is missing; in the missing-schema case, report that AutoML is enabled but not runnable for this model until schemas are generated.

Non-train actions such as `evaluate` and `inference` stay in this model skill.
The per-run `automl_policy` override does not change model metadata.

The default VILA container exposes action entrypoints as `vila-train`, `vila-evaluate`, and `vila-inference`; it does not expose a `vila <action> -e <spec>` dispatcher. The model skill adapts the packaged YAML specs to those flat CLI entrypoints in `references/skill_info.yaml`.

## Training Requirements

- **Dataset type:** vlm
- **Formats:** default, raw
- **Accepted dataset intents:** training, evaluation, testing
- **Monitoring metric:** loss
- **Default AutoML parameter:** train.learning_rate

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | train.dataset.dataset_yaml_path | train_datasets | dataset.yaml | No |
| train | train.dataset.data_path | train_datasets | annotations JSON | No |
| train | train.dataset.media_dir | train_datasets | media directory or tar/tar.gz | No |
| evaluate | evaluate.dataset_yaml_path | eval_dataset | dataset.yaml | No |
| evaluate | evaluate.data_path | eval_dataset | annotations JSON | No |
| evaluate | evaluate.video_dir | eval_dataset | video directory or tar/tar.gz | No |
| inference | inference.media | inference_dataset | media file or dataset root | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "s3://bucket/data/train"
S3_EVAL = "s3://bucket/data/eval"
```

**train (mandatory data sources):**
```python
{
    "model_path": "/path/or/s3/to/base_vila_model",
    "train.dataset.dataset_yaml_path": f"{S3_TRAIN}/dataset.yaml",
    "train.dataset.data_path": f"{S3_TRAIN}/annotations.json",
    "train.dataset.media_dir": f"{S3_TRAIN}/dataset.tar.gz",
    "train.num_epochs": 1,
    "train.batch_size": 8,
    "train.learning_rate": 1e-4,
    "train.vision_learning_rate": 1e-5,
    "train.lora_r": 16,
    "train.llm_mode": "lora",
}
```

**evaluate (mandatory data sources):**
```python
{
    "model_path": "/path/or/s3/to/trained_or_base_vila_model",
    "evaluate.task": "youcook2_val_rtl",
    "evaluate.dataset_yaml_path": f"{S3_EVAL}/dataset.yaml",
    "evaluate.data_path": f"{S3_EVAL}/youcookii_val_rtl.json",
    "evaluate.video_dir": f"{S3_EVAL}/videos.tar.gz",
    "evaluate.model_base": "/path/or/s3/to/base_vila_model_if_model_path_is_peft",
    "evaluate.num_time_tokens": 0,
}
```

**inference (mandatory data sources):**
```python
{
    "model_path": "/path/or/s3/to/trained_or_base_vila_model",
    "inference.model_base": "/path/or/s3/to/base_vila_model_if_model_path_is_peft",
    "inference.media": f"{S3_EVAL}/sample.mp4",
}
```

## Eval Dataset

Optional. Evaluation dataset configured via evaluate.dataset_yaml_path.

## Important Parameters

- **model_path**: Path to base VILA model. Required.
- **train.num_epochs**: Training epochs. Default 1 (VLM fine-tuning is typically short).
- **train.batch_size**: Per-GPU batch size. Default 8.
- **train.learning_rate**: LLM learning rate. Default 1e-4.
- **train.vision_learning_rate**: Vision encoder learning rate. Default 1e-5 (lower for stability).
- **train.lora_r**: LoRA rank. Default 16.
- **train.llm_mode**: LLM training mode. Default "lora". Options: lora, ft (full).
- **train.vision_mode**: Vision encoder mode. Default "ft" (full fine-tune).
- **train.model_max_length**: Maximum sequence length. Default 32768.
- **train.max_tiles**: Maximum image tiles. Default 12.
- **train.video_max_tiles**: Maximum video tiles. Default 6.
- **train.num_video_frames**: Number of video frames. Default 8.
- **train.gradient_accumulation_steps**: Gradient accumulation. Default 2.
- **train.warmup_ratio**: LR warmup ratio. Default 0.03.
- **train.dataset.dataset_yaml_path**: Path to dataset YAML configuration.
- **evaluate.task**: Evaluation benchmark. Default "youcook2_val".
- **evaluate.num_time_tokens**: Number of RTL time tokens to decode when evaluating `llava.eval.rtl` datasets. Default 0.
- **evaluate.time_token_format**: Format string for RTL time tokens. Default "<t{t}>".
- **inference.text**: Inference prompt. Default "What is this video about?"
- **inference.conv_mode**: Conversation mode. Default "auto".

## Hardware

Minimum 1 GPU(s), recommended 8 GPU(s). 40GB+ (A100 80GB recommended) VRAM per GPU. VLM fine-tuning is very memory intensive. LoRA mode (llm_mode=lora) significantly reduces memory. Use multi-GPU with DDP for reasonable training times. Full fine-tune (llm_mode=ft) requires more GPUs.

## Error Patterns

**CUDA out of memory**: Switch to llm_mode=lora, reduce batch_size, or reduce model_max_length / max_tiles.

**Incompatible base checkpoint**: If training or inference fails with
`Transformers does not recognize this architecture` for `model_type:
qwen2_5_vl`, the provided `model_path` is a raw Qwen2.5-VL checkpoint rather
than a VILA/NVILA wrapper checkpoint. Use a `llava_llama` VILA checkpoint or a
local folder downloaded from one.

**Missing dataset YAML**: Ensure train.dataset.dataset_yaml_path points to a valid YAML file.

**Remote paths inside dataset YAML**: The VILA container expects `data_path` and
`media_dir` to be local paths. When a dataset YAML contains `aws://` or `s3://`
members, pass `train.dataset.data_path` / `train.dataset.media_dir` or
`evaluate.data_path` / `evaluate.video_dir` as model skill inputs so the runner
downloads them before launching the VILA entrypoint.

**Container entrypoint mismatch**: If `vila: command not found` or `vila train`
fails, verify `references/skill_info.yaml` still wraps the flat
`vila-train` / `vila-evaluate` / `vila-inference` commands.

**RTL evaluation time tokens**: `llava.eval.rtl` expects `model.config.num_time_tokens`
to be an integer. PEFT/base loading can drop the trained adapter's value, so the
model skill restores `evaluate.num_time_tokens` and `evaluate.time_token_format`
before invoking the RTL module.

**Video frame loading**: Ensure num_video_frames and video_max_tiles are compatible with available GPU memory.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong in this MD file, not in `config.json`. Generated runners should read this section and apply the mappings with SDK helpers before `create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `vila.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| evaluate | `evaluate.model_base` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| evaluate | `model_path` | `parent_model_folder` | LoRA output folder inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| inference | `inference.model_base` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| inference | `model_path` | `parent_model_folder` | LoRA output folder inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| train | `model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |

For `parent_model` or `parent_model_folder`, pass the upstream train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent result folder, filters checkpoint artifacts, and returns the selected model file or folder. Do not add these mappings back to `config.json` and do not patch generated runner scripts to guess checkpoint paths.
