# Cosmos AnomalyGen

Synthetic defect image generation for AOI (Automated Optical Inspection) using the Cosmos AnomalyGen diffusion model. Generates realistic anomaly images by compositing defect masks onto clean PCB images.

This is an **image-based** SDG pipeline (unlike Cosmos Predict 2.5 which generates video). It's used in the AOI DEFT loop to augment training data for Visual ChangeNet.

## Pipeline Steps

1. **Automatic Mask Placement (AMP)** -- place defect submask templates onto clean images using per-image ROI masks. Generates N placements per image via seed variation.
2. **Prepare sdg_inference.jsonl** -- pair clean images with auto-placed masks and generation parameters (guidance, shift, rotation, morph). This step is scripted inline, not a container action.
3. **SDG Inference** -- generate synthetic anomaly images using the pretrained AnomalyGen model via torchrun.

## Credentials

- **HF_TOKEN** (required): HuggingFace access token. The Cosmos Predict2 2B model is gated -- requires model agreement acceptance at https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image.

## Pretrained Models

The container requires three pretrained models mounted at `checkpoints/`:

| Model | Path | Source |
|---|---|---|
| Cosmos Predict2 2B | `nvidia/Cosmos-Predict2-2B-Text2Image/` | HuggingFace (gated) |
| NV-DINOv2 | `NVDINOV2/nv_dinov2_classification_model.ckpt` | NGC |
| C-RADIOv3-B | `nvidia/C-RADIO-V3/model.safetensors` | HuggingFace |

Download all with the built-in script: `python -m scripts.download_checkpoints --model_types text2image --model_sizes 2B`. The script is idempotent and handles transitive dependencies (T5-11B text encoder, tokenizer).

## Data Format

### Input Structure

```
clean_image_dir/
  bridge/              # Per-defect subdirectory
    image_A.jpg
    image_B.jpg

roi_dir/
  bridge/              # ROI file stems must match clean image stems
    image_A.png
    image_B.png

submask_dir/
  bridge/              # One or more submask template files
    template_mask.jpg
```

### Defect Description (JSONL)

```jsonl
{"defect_type": "PCB+bridge", "spatial_dependency": "position_dependent"}
{"defect_type": "PCB+excess_solder", "spatial_dependency": "free"}
```

Spatial dependency controls mask placement:
- **position_dependent**: no shift, no rotation, no morphing (defect must appear at exact location)
- **free**: random shift [-100,100], random rotation [0,180], random morphing

### SDG Inference JSONL Entry

```json
{
  "image_filename": "/data/clean_image_dir/bridge/image_A.jpg",
  "mask_filename": "/data/amp_output/bridge/image_A/seed_0001/auto_placed_mask_with_1_rois_seed_1.png",
  "anomaly_type": "PCB+bridge",
  "guidance": 7.0,
  "num_steps": 35,
  "crop_and_paste": true,
  "crop_ratio": 2.0,
  "num_generated_images": 1,
  "shift_values": "0,0",
  "rotation_angle": 0,
  "morph_operation": "none"
}
```

### Output Structure

```
output_dir/
  sdg_inference.jsonl        # Input JSONL
  reconstructed_image/       # Final synthetic anomaly images
  annotated_image/           # Annotated overlays
  original_image/            # Source clean images (copied)
  original_mask/             # Source masks (copied)
  SDG_result.csv             # Metadata CSV for all generated images
```

## Important Parameters

- **step**: Checkpoint step number to load from the fine-tuned AnomalyGen model. Must match an available checkpoint in the checkpoint directory.
- **N** (seed count): Number of mask placements per clean image. More seeds = more diversity but longer AMP time. Typical: 5.
- **guidance**: Classifier-free guidance scale. Options: 1.5 (subtle), 5.0 (moderate), 7.0 (strong). Randomized per entry in the JSONL.
- **num_steps**: Diffusion steps. Default 35. More steps = higher quality but slower.
- **nproc_per_node**: Number of GPUs for torchrun. Default 1. Multi-GPU support available.

## Hardware

- **Minimum**: 1 GPU with 24GB+ VRAM (L40 or A100)
- AMP step is CPU-bound (~3-5 min for 5 images x 5 seeds)
- SDG inference is GPU-bound (~30 min for 25 entries on L40)

## Container Notes

- Use `--entrypoint sh` and `tail -f /dev/null` to keep alive, then `docker exec` for each step
- Container shell is POSIX `sh`, not bash -- avoid bash-isms (`[[ ]]`, `<<<`, `${var,,}`)
- The `--` before `experiment=predict2_anomaly_gen_fsdp_2b` is required to separate argparse from config overrides

## Error Patterns

**FileNotFoundError on pretrained models**: The pipeline loads models at hardcoded relative paths under `checkpoints/`. Errors appear one at a time. Fix: mount the pretrained checkpoints directory and run the download script.

**401 Unauthorized on HuggingFace download**: `nvidia/Cosmos-Predict2-2B-Text2Image` is gated. Fix: ensure HF_TOKEN is set and the user has accepted the model agreement.

**Non-binary mask ValueError**: The SDG pipeline requires mask pixel values to be strictly 0 or 255. Fix: threshold the mask before use.

**Unsupported anomaly type**: The checkpoint's `ag_config.yaml` lists supported types. If a defect type is not in the list, the pipeline will fail. Fix: check `ag_config.yaml` before running and skip unsupported types.

**Container binary execution errors**: Some hosts can't execute `/usr/bin/bash` or `/usr/bin/sleep` inside this image. Fix: use `--entrypoint sh` and `tail -f /dev/null`; exec with `sh -c` not `bash -c`.

**Long-running SDG inference killed**: Agent runner timeout may kill the process. Fix: set timeout >= 45 min. Partial output is still usable -- check `reconstructed_image/` for completed images.

**CRITICAL: Never use the mask/ directory**: Only use masks generated by the AMP step. Pre-existing `mask/` directories in the dataset are from a different pipeline and will produce incorrect results.
