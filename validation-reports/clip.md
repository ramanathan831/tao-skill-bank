# Model: clip

## Supported actions tested

- train: blocked
- eval: blocked
- inference: fail for PyTorch `clip inference` with `inference.checkpoint: null`
- export: pass
- deploy: partially pass
- prune: not supported by this model skill
- retrain: blocked because training did not produce a checkpoint
- dataset convert: not supported by this model skill
- other: gen_trt_engine: pass for the separate vision ONNX path
- other: inference on tao-deployed gen_trt_engine model: pass for image-only TensorRT embeddings
- other: TensorRT evaluate/full retrieval deployment: blocked

## Dataset used

- Source: s3://nvcf-storage-handling/data/classification_test/images_test.tar.gz for image-only inference/deploy validation
- Notes: Extracted 100 real images from the S3 classification test archive and used them only as unlabeled image inputs for CLIP embedding extraction. This was not used as CLIP train/eval data.
- Any dataset compatibility issues: No native CLIP custom image-caption dataset or WebDataset shard set was found in the S3 top-level inventory. The available classification archive has class folders and `classes.txt`, but no paired captions, image list, or WDS shards. Per the CLIP skill, image-classification data must not be silently converted into captions unless the user explicitly authorizes that plumbing-only fallback.

## Training result

- Training completed: no
- Best checkpoint produced: no
- Best checkpoint path: none
- Other checkpoints produced:
  - none
  - Export produced `/workspace/run/results/export/clip_model_vision.onnx`, `/workspace/run/results/export/clip_model_text.onnx`, `/workspace/run/results/export/clip_model_config.yaml`, and `/workspace/run/results/export/clip_model_tokenizer/`
  - Deploy produced `/workspace/run/results/gen_trt_engine/clip_vision.engine`
  - Deploy inference produced `/workspace/run/results/deploy_inference/trt_inference/image_embeddings.h5`

## Checkpoint/action verification

- Eval checkpoint used: none; blocked because no compatible CLIP validation captions were available
- Inference checkpoint used: none; PyTorch `clip inference` with `inference.checkpoint: null` failed before embedding extraction with `TypeError: expected str, bytes or os.PathLike object, not NoneType`
- Export checkpoint used: none by design; `export.checkpoint: null` passed and exported separate encoders from the selected `ViT-L-14-SigLIP-CLIPA-224` architecture
- Resume/retrain checkpoint used: none; blocked because training did not run
- Deploy artifacts used: `/workspace/run/results/export/clip_model_vision.onnx` for `gen_trt_engine`, then `/workspace/run/results/gen_trt_engine/clip_vision.engine` for TensorRT inference
- Were checkpoint paths selected through the proper resolver: no checkpoint-dependent path was available to resolve. The model skill documents using the exact resolved CLIP checkpoint such as `model_epoch_000_step_00020.pth` rather than `clip_latest.pth` for checkpoint-backed actions.
- Any incorrect latest-checkpoint behavior found: no latest-checkpoint use occurred. The observed bug was the opposite failure mode: PyTorch inference did not resolve a pretrained or trained checkpoint when `inference.checkpoint` was null.

## Issues found

- Model skill issues:
  - `models/clip/SKILL.md` said unset `inference.checkpoint` loads pretrained weights. In the validation-fixes PyTorch image, `clip inference` passes `None` into `load_model_from_checkpoint` and fails before extracting embeddings.
  - The deploy path remains image-only. Separate vision ONNX builds and runs, but text/full retrieval TensorRT deployment is still documented as blocked until text input profiles are supported.
- Config issues:
  - `automl_policy=on` cannot be exercised here without a CLIP-compatible training dataset, and workflow skills are out of scope for this validation pass.
  - Export with `export.checkpoint: null` passed, but the log included `No pretrained weights loaded... Model initialized randomly`; this validates the export/deploy plumbing, not pretrained model quality.
- Dataset issues:
  - No compatible CLIP training/evaluation dataset was found under `s3://nvcf-storage-handling/data/`. Candidate VLM/video prefixes such as `vlm_inference/`, `lita_subset/`, and `cosmos-embed/` are not CLIP image-caption custom data in the format required by this skill.
- Checkpoint issues:
  - No training checkpoint was produced because train was blocked by dataset availability.
  - PyTorch inference needs a concrete checkpoint path in this image; it does not fall back to a pretrained model when `inference.checkpoint` is null.
- Docker/local execution issues:
  - None for export, deploy engine build, or deploy inference with the requested validation-fixes images.
- Fresh-install issues:
  - A fresh install with only the provided S3 bucket cannot fully validate CLIP train/evaluate/retrain unless a real CLIP image-caption or WDS dataset is added.

## Fixes made

- Updated `models/clip/SKILL.md` to make unset checkpoint behavior action-specific, document the PyTorch inference null-checkpoint failure, and remove the stale hardcoded deploy image version wording from the validated image-only TensorRT path.

## Remaining issues

- CLIP train, PyTorch evaluate, checkpoint-backed PyTorch inference, and retrain remain blocked until a compatible image-caption or WDS dataset is available.
- TensorRT retrieval evaluation/full text deployment remains blocked by the model skill's documented text-input TensorRT limitation.

## Files changed

- `models/clip/SKILL.md`
- `validation-reports/clip.md`

## Final status

- Partially validated. Export, vision TensorRT engine generation, and image-only TensorRT inference pass on local-docker with the requested validation-fixes PyTorch and Deploy images. Dataset-backed CLIP actions remain blocked by the absence of a compatible CLIP dataset in the provided S3 source, and PyTorch image-only inference with a null checkpoint fails as documented.
