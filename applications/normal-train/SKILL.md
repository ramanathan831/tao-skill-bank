---
name: normal-train
description: "Standard single-step train/eval/export workflow for any TAO model. Use when training a model on a dataset without iterative augmentation."
---

# Normal Train

Standard supervised fine-tuning: train a model on a labeled dataset, optionally evaluate, then optionally export. The most common TAO workflow for adapting a pretrained model to a new dataset.

## Steps

1. **train** — always executed
2. **eval** — executed if `eval_dataset_uri` is resolved
3. **export** — optional, on user request after training

## Prerequisites

### Required
- **model**: A compatible TAO model (e.g., clip, nvdinov2, grounding_dino)
- **train_dataset_uri**: URI of the training dataset (e.g., `s3://bucket/train/`)
- **platform**: Compute backend: lepton, slurm, kubernetes, or docker

### Optional
- **eval_dataset_uri**: Some model skills mark this as required — check the resolved model skill before treating it as optional.
- **base_checkpoint**: If not provided, defaults to the NGC pretrained checkpoint listed in the model skill, or trains from scratch if no NGC checkpoint exists.
- **image**: If not provided, resolved automatically from the model's network config. Use to pin a specific TAO toolkit version.
