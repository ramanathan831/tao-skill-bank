---
name: cosmos-predict-2-5
description: "Cosmos Predict 2.5 text-to-video generation for synthetic data augmentation. Use when generating synthetic training videos from text captions or augmenting video datasets."
---

# Cosmos Predict 2.5

NVIDIA's text-to-video generation model. Cosmos Predict 2.5 generates synthetic training videos from text captions, serving as a core component of the DEFT (Data-Efficient Fine-Tuning) pipeline.

## Purpose

Given text captions describing desired video content, Cosmos Predict 2.5 generates realistic synthetic videos. These generated videos augment real training data to improve downstream model performance, particularly for tasks where collecting real-world video data is expensive or dangerous.

## Inputs

- **input_prompt_json**: JSONL file with `video_id` and `caption` fields. Each line describes one video to generate.
- **weak_data_list**: Parquet file with `video_id`, `question`, and `ground_truth` fields. Identifies the gap data that needs synthetic augmentation.

## Outputs

- **output_dir**: Directory containing generated video files and an output parquet mapping video IDs to generated file paths.

## Parallel Execution

Supports splitting the input JSONL across multiple GPU groups for parallel generation. Each split runs independently, and results are merged after all splits complete.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_gpus | 8 | Number of GPUs for generation |
| num_output_frames | 500 | Frames per generated video |
| guidance | 7 | Guidance scale — higher values produce videos more faithful to the caption but potentially less diverse |

## Guardrails

Cosmos Predict 2.5 includes safety guardrails that block generation of certain content. For DEFT collision detection pipelines, guardrails **must be disabled** via the `disable-guardrails` flag (configured in `config.json`).

**Why:** Prompts describing vehicle collisions trigger the guardrail, which blocks the prompt on rank 0 and skips the diffusion step. However, other GPU ranks have already entered the NCCL collective for the diffusion denoising. Rank 0 moves to the next sample while ranks 1-N are stuck waiting, causing an NCCL collective timeout and SIGABRT on all ranks. This crashes the entire job even though non-collision prompts would succeed.

## Error Patterns

| Error | Cause | Fix |
|-------|-------|-----|
| NCCL timeout / SIGABRT | Guardrail blocks a prompt mid-batch, desynchronizing GPU ranks | Disable guardrails (`--disable-guardrails` flag) |
| CUDA OOM | Video resolution or batch too large for GPU memory | Reduce batch size or num_output_frames |
| HF auth failure | Missing or invalid HF_TOKEN | Ensure HF_TOKEN is set in secrets.json with a valid HuggingFace token that has access to the model |
| Low quality / artifacts | Guidance scale too low or too high | Adjust guidance parameter (typical range: 5-10) |
| Slow generation | Insufficient GPU count | Increase num_gpus or use faster GPU hardware (H100 preferred) |
