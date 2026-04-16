---
name: clip
description: "CLIP vision-language model for image classification, zero-shot recognition, and embeddings. Use when training or fine-tuning CLIP, running zero-shot classification, or computing image embeddings."
---

# CLIP

Contrastive Language-Image Pre-training model for zero-shot and fine-tuned image classification and retrieval. Fine-tuning adapts CLIP's vision-language alignment to domain-specific datasets.

No default NGC pretrained checkpoint — uses HuggingFace CLIP weights built into the container.

## Eval Dataset

Optional. CLIP training does not require a separate eval dataset. If provided, validation metrics are computed at each checkpoint interval.

## Important Parameters

- **num_epochs**: CLIP fine-tuning typically converges quickly. 10–20 epochs is usually sufficient for domain adaptation. Increase if validation loss is still decreasing at the end of training.
- **train.optim.lr**: Learning rate for fine-tuning. CLIP is sensitive to high learning rates — use 1e-6 to 1e-5. Higher values risk catastrophic forgetting of the pretrained representations.
- **model.freeze_text_encoder**: Whether to freeze the text encoder during training. Set to true (default) for most fine-tuning tasks. Only unfreeze if you have a large dataset and want to adapt both modalities.

## Hardware

CLIP is relatively lightweight compared to detection models. Single GPU training works for small datasets. Use 4+ GPUs for datasets with >100k images. 16GB+ VRAM per GPU (V100 or A100).

## Error Patterns

**CUDA out of memory**: Reduce batch_size (32 → 16 → 8). CLIP's memory footprint is dominated by the vision encoder resolution — if OOM persists, check `model.image_size` in the spec.

**NaN loss**: Learning rate is too high for fine-tuning. Reduce to 1e-7 and increase warmup steps. Also verify that input images are normalized correctly.

**Zero accuracy after training**: Check that the dataset class names match the text prompts used during training. CLIP matches images to text descriptions, so class label format matters.

**Dataset size smaller than total batch size**: The total batch size is `batch_size × num_gpus`. For example, batch_size=16 with num_gpus=8 gives a total batch size of 128. If the dataset (especially val) has fewer samples than this, training fails with ValueError. Fix: reduce `dataset.val.batch_size` or `dataset.train.batch_size` so that `batch_size × num_gpus <= dataset_size`. The agent should proactively check this when num_gpus > 1 and the dataset is known to be small.

**Error merging spec.yaml with schema**: A Hydra/OmegaConf config validation error. Usually caused by spec keys placed at the wrong nesting level. Common cause: `num_epochs` and `num_gpus` must be under `train.*`, not at the spec root. Use the SDK's spec_shorthand_keys mapping to ensure correct placement.
