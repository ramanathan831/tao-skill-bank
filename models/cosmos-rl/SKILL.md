---
name: cosmos-rl
description: "Cosmos-Reason2-8B video QA supervised fine-tuning with FSDP parallelism. Use when training or evaluating video question-answering models or working with Cosmos-RL."
---

# Cosmos-RL

Supervised fine-tuning (SFT) of **nvidia/Cosmos-Reason2-8B** on video reasoning tasks. Pretrained weights are sourced from HuggingFace, not NGC. This is a **gated model** — requires `HF_TOKEN`.

Uses FSDP-based parallelism with `dp_shard_size` for GPU count and `dp_replicate_size` for node count (not the standard `num_gpus`/`num_nodes`).

## Credentials

- **HF_TOKEN** (required): HuggingFace access token. The user must accept the model agreement at <https://huggingface.co/nvidia/Cosmos-Reason2-8B> and provide a token with read access. Passed to the container as a `docker_env_var`.

## Training Requirements

- **Dataset type:** vlm
- **Formats:** llava
- **Accepted dataset intents:** training, evaluation, testing
- **Monitoring metric:** val/avg_loss, val/reward_avg, val/loss

### Per-Action Dataset Requirements

| Action | Spec Key | Source | Files | List? |
|---|---|---|---|---|
| train | custom.train_dataset.annotation_path | train_datasets | annotations.json | No |
| train | custom.train_dataset.media_path | train_datasets | videos.tar.gz (or images.tar.gz) | No |
| train | custom.val_dataset.annotation_path | eval_dataset | annotations.json | No |
| train | custom.val_dataset.media_path | eval_dataset | videos.tar.gz (or images.tar.gz) | No |
| evaluate | dataset.annotation_path | eval_dataset | annotations.json | No |
| evaluate | dataset.media_dir | eval_dataset | videos.tar.gz (or images.tar.gz) | No |
| quantize | calibration_dataset.annotation_path | calibration_dataset | annotations.json | No |
| quantize | calibration_dataset.media_dir | calibration_dataset | videos.tar.gz (or images.tar.gz) | No |

### Typical Spec Overrides

Data source overrides are **mandatory for every action** — the agent MUST construct data source paths from the Per-Action Dataset Requirements table above and include them in `spec_overrides`.

```python
S3_TRAIN = "aws://bucket/data/train"
S3_EVAL = "aws://bucket/data/eval"
# Media is typically videos.tar.gz for video QA tasks, images.tar.gz for image QA
MEDIA = "videos.tar.gz"  # or "images.tar.gz" — ASK the user
```

**train (mandatory data sources):**
```python
{
    "custom.train_dataset.annotation_path": f"{S3_TRAIN}/annotations.json",
    "custom.train_dataset.media_path": f"{S3_TRAIN}/{MEDIA}",
    "custom.val_dataset.annotation_path": f"{S3_EVAL}/annotations.json",
    "custom.val_dataset.media_path": f"{S3_EVAL}/{MEDIA}",
    "policy.model_name_or_path": "hf_model://nvidia/Cosmos-Reason2-8B",
    "policy.model_max_length": 81920,
    "policy.parallelism.dp_shard_size": 4,
    "policy.parallelism.dp_replicate_size": 1,
    "policy.lora.lora_alpha": 256,
    "policy.lora.r": 16,
    "policy.lora.lora_dropout": 0.05,
    "train.epoch": 1,
    "train.train_batch_per_replica": 32,
    "train.optm_lr": 2e-5,
    "train.optm_impl": "fused",
    "train.deterministic": True,
    "train.ckpt.save_freq_in_epoch": 1,
    "train.ckpt.max_keep": 1,
    "train.train_policy.mini_batch": 1,
    "train.train_policy.dataset.test_size": 0,
    "train.train_policy.dataloader_num_workers": 4,
    "train.train_policy.dataloader_prefetch_factor": 4,
    "validation.freq_in_epoch": 1,
    "validation.batch_size": 1,
    "validation.enable_dataset_cache": False,
    "custom.vision.nframes": 8,
    "custom.system_prompt": "You are a helpful assistant.",
    "logging.logger": ["console", "tao"],
}
```

**evaluate (mandatory data sources):**
```python
{
    "dataset.annotation_path": f"{S3_EVAL}/annotations.json",
    "dataset.media_dir": f"{S3_EVAL}/{MEDIA}",
    "vision.nframes": 8,
    "model.enable_lora": True,
    "model.base_model_path": "hf_model://nvidia/Cosmos-Reason2-8B",
}
```

**quantize (mandatory data sources):**
```python
{
    "calibration_dataset.annotation_path": f"{S3_TRAIN}/annotations.json",
    "calibration_dataset.media_dir": f"{S3_TRAIN}/{MEDIA}",
    "model.enable_lora": True,
    "model.base_model_path": "hf_model://nvidia/Cosmos-Reason2-8B",
}
```

**inference (mandatory data sources):**
```python
{
    "media": "aws://bucket/data/videos/test_video.mp4",
    "prompt": "When does something happen in the video?",
    "enable_lora": True,
    "base_model_path": "hf_model://nvidia/Cosmos-Reason2-8B",
}
```

## Critical Overrides (Train)

The TAO Core schema has broken defaults for cosmos-rl training. These are applied automatically via `key_defaults` in config.json:

| Parameter | Schema Default | Required Value | Why |
|---|---|---|---|
| `policy.model_name_or_path` | `nvidia/Cosmos-Reason1-7B` | `nvidia/Cosmos-Reason2-8B` | CR1 is outdated |
| `policy.model_max_length` | 4096 | 40960 | Too small for video — causes `vision_embeds` shape mismatch |
| `train.train_batch_per_replica` | 1 | 32 (or any multiple of mini_batch) | Default not divisible by mini_batch=4 — immediate AssertionError |

## Evaluate

Evaluate uses the **script runner** (not container_handler). The `actions.evaluate` config in config.json declares inputs/outputs, and the script runner handles S3 I/O, selective download, and Lustre caching.

### Config format

The evaluator reads a **flat TOML** config with top-level keys: `dataset`, `model`, `task`, `evaluation`, `vision`, `generation`, `metrics`, `results`, `num_gpus`, `results_dir`. The defaults file (`defaults-evaluate.json`) matches this flat structure.

### Task type

- Empty string (`""`) — General Evaluator. Auto-detects binary classification (yes/no) from ground truth and computes TP/FP/TN/FN/accuracy/precision/recall/F1.
- `"its_directionality"` — ITS-specific evaluator for left/right/straight classification. Do NOT use for collision detection.

### LoRA Evaluation

To evaluate a fine-tuned LoRA model, pass the checkpoint path via spec_overrides:

```python
spec_overrides={
    'model.model_name': 's3://bucket/results/{train_job_id}/safetensors/epoch_1',
    'model.enable_lora': True,
    'model.base_model_path': 'nvidia/Cosmos-Reason2-8B',
    'evaluation.batch_size': 10,
}
```

The script runner downloads the LoRA adapter from S3/Lustre and the evaluator merges it with the base model before running inference.

### Selective download

The script runner reads `dataset.annotation_path`, extracts referenced video paths via the `video` key, and downloads only those files. For a 112-sample collision dataset, this downloads ~500MB instead of the full 4.8GB folder.

### Results

- `results.json` — per-sample predictions with `video_id`, `response`, `question`, `gt`
- Binary metrics: accuracy, balanced accuracy, precision, recall, F1
- Text metrics: BLEU, ROUGE, BERTScore
- When Lustre is available, results write to Lustre for cross-job persistence (e.g., gap analysis reads directly), then upload to S3.

## Datasets

The `data_sources` config in config.json maps dataset URIs to spec paths. It appends `annotations.json` to the dataset directory URI by convention. If your dataset uses a different annotation filename, override the annotation path via spec_overrides:

```python
spec_overrides={
    'custom.val_dataset.annotation_path': 's3://bucket/eval/my_annotations.json',
    'custom.val_dataset.media_path': 's3://bucket/eval/',
}
```

**Eval dataset** is optional for training. If provided, validation metrics are computed at the frequency set by `validation.freq_in_epoch`. If not provided, use `dataset.test_size` to auto-split training data.

## Important Parameters

### Training Loop
- **train.epoch**: Number of training epochs. Default 10.
- **train.train_batch_per_replica**: Global batch size per training step. Ideally >= 32 for stability. CRITICAL: must be divisible by `train.train_policy.mini_batch` (default 4). Recommended: 32.
- **train.compile**: Set to true for potential speedup on newer GPUs (H100), else false.
- **train.output_dir**: Output directory for checkpoints and logs.

### Model & Policy
- **policy.model_name_or_path**: HuggingFace model path. Must be `nvidia/Cosmos-Reason2-8B`.
- **policy.model_max_length**: Context window size. Must be 40960 for video SFT. Affected by FPS, resolution, and prompt length.
- **policy.model_gradient_checkpointing**: Save VRAM by recomputing activations. Keep true for large models.

### Parallelism (Multi-GPU)
- **policy.parallelism.dp_shard_size**: Data-parallel shard size. CRITICAL: should equal total GPU count. This is the Cosmos-RL equivalent of `num_gpus`.
- **policy.parallelism.dp_replicate_size**: Data-parallel replication (node count). Equivalent of `num_nodes`.
- **policy.parallelism.tp_size**: Tensor parallelism. Default 1.
- **policy.parallelism.cp_size**: Context parallelism. Default 1.
- **policy.parallelism.pp_size**: Pipeline parallelism. Default 1.

### Optimization & Data Loading
- **train.optm_lr**: Learning rate. Default 1e-6.
- **train.train_policy.type**: Training policy. Default `sft`.
- **train.train_policy.mini_batch**: Micro-batch size per GPU. If OOM, reduce this. Constraint: `train_batch_per_replica % mini_batch == 0`.
- **train.train_policy.dataset.name**: Unique ID for dataset cache. IMPORTANT: change this if you modify `fps` or `total_pixels` to force cache regeneration.
- **train.train_policy.dataset.test_size**: Validation split. Float (0.0–1.0) = ratio; Int = absolute number.

### Vision Encoders
- **custom.vision.fps**: Frames per second extracted from video. High motion: 3. Low motion/static: 1–2.
- **custom.vision.total_pixels**: Resolution constraint. Increase if the object of focus is small relative to the frame. Default 3136000.
- **custom.system_prompt**: Instructions prepended to every prompt.

### Checkpointing
- **train.ckpt.save_freq_in_epoch**: Save every N epochs. Default 10.
- **train.ckpt.max_keep**: Keep N most recent checkpoints. Default 8 (use 1 to save storage).
- **train.ckpt.export_safetensors**: Export in safetensors format. Default true.

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

**train_batch_per_replica not divisible by mini_batch**: The default `train_batch_per_replica=1` from the TAO Core schema is invalid because `mini_batch` defaults to 4. Immediate AssertionError on all ranks. Fix: set `train_batch_per_replica` to a multiple of `mini_batch` (recommended: 32 for large datasets, 4 for small datasets).

**train_batch_per_replica larger than samples per rank**: With FSDP, each rank sees `total_samples / dp_shard_size` samples. If `train_batch_per_replica` exceeds this, the trainer completes 0 training steps and attempts to save a checkpoint before the optimizer/scheduler is initialized, crashing with `'NoneType' object has no attribute 'state_dict'`. Fix: ensure `train_batch_per_replica <= total_samples / dp_shard_size`. For small datasets (e.g., 31 DEFT-generated samples on 8 GPUs = ~4 per rank), set `train_batch_per_replica` to 4.

**Stale dataset cache after changing fps/total_pixels**: Change `train.train_policy.dataset.name` to a new unique identifier to force cache regeneration.

**Checkpoint save failure (scheduler is None)**: The cosmos-rl trainer crashes with `'NoneType' object has no attribute 'state_dict'` when saving a checkpoint before any training step has executed. This happens when the dataset is too small for the batch size (0 steps per epoch). See the batch size error above.

**You are trying to access a gated repo**: The HuggingFace model `nvidia/Cosmos-Reason2-8B` requires authentication. All ranks will retry in a loop until they time out. Fix: ensure `HF_TOKEN` is set in `secrets.json` and passed as a `docker_env_var`. The user must also accept the model agreement at <https://huggingface.co/nvidia/Cosmos-Reason2-8B>.

## DEFT Support

Cosmos-RL implements the DEFT workflow contract for video QA tasks. See `config.json` for the full DEFT section and `workflow/deft/deft.md` for the pipeline overview.

### Gap Analysis (`scripts/analyze_gaps.py`)

Model-specific script that identifies failure cases from cosmos-rl evaluation output.

- **Eval output format:** `results.json` with fields: `video_id`, `response`, `question`, `gt`
- **Comparison:** exact string match after `.lower().strip()` — requires eval prompts that force short constrained answers (e.g., yes/no)
- **Output:** parquet with `video_id` (full path), `question`, `ground_truth`

**Limitation:** Brittle exact match. If the model responds with full sentences instead of constrained answers, mismatches will be over-reported. The eval prompt design must account for this.
