Model: clip

Supported actions tested:
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass after trusted-checkpoint PyTorch load override
- inference: pass after trusted-checkpoint PyTorch load override
- export: pass after trusted-checkpoint PyTorch load override
- deploy / gen_trt_engine: partial pass; image-only `_vision.onnx` -> `_vision.engine` passed, combined and text ONNX failed
- deploy / inference on tao-deployed gen_trt_engine model: partial pass; image-only TensorRT inference passed
- deploy / evaluate on tao-deployed gen_trt_engine model: fail; full retrieval evaluation requires a text engine
- prune: unsupported
- retrain/resume: pass
- quantize: unsupported
- dataset convert: unsupported
- other: separate encoder export pass

Dataset used:
- Source: `s3://nvcf-storage-handling/data/classification_train/images_train.tar.gz`, `s3://nvcf-storage-handling/data/classification_val/images_val.tar.gz`, and `s3://nvcf-storage-handling/data/classification_test/images_test.tar.gz`
- Notes: No native CLIP image-caption dataset was found under the inspected S3 data prefix. For CLIP plumbing validation only, I derived a small custom CLIP dataset from the real S3 classification images by writing one caption file per image from the class label.
- Any dataset compatibility issues: The derived dataset is suitable for exercising train/eval/inference/export wiring, but it is not a real image-caption quality dataset.

Training result:
- Training completed: yes
- Best checkpoint produced: no dedicated best-named artifact; CLIP produced an epoch/step checkpoint and `clip_latest.pth` symlink, with validation metrics logged at the only epoch
- Best checkpoint path: not emitted as a separate artifact; the resolved checkpoint used for downstream actions was `/tmp/tao-model-validation/clip/results/train/model_epoch_000_step_00020.pth`
- Other checkpoints produced: `/tmp/tao-model-validation/clip/results/resume/model_epoch_001_step_00040.pth`

AutoML default train route:
- Default policy verified: fixed packaged CLIP train schema so default AutoML search parameters are `train.optim.text_lr` and `train.optim.vision_lr`; direct model training remains reserved for explicit `automl_policy: off`.
- Rerun completed: yes, through `AutoMLRunner` + `DockerSDK` on local Docker.
- Algorithm/config: Bayesian, `automl_max_recommendations=2`, metric `val/t2i_mAP`, direction maximize, 1 epoch, batch size 2, one GPU.
- Search parameters/ranges: `train.optim.vision_lr` and `train.optim.text_lr`, each constrained to `1e-6..1e-5`.
- Recommendation 0: job `eea8da96-a5a2-4b5c-9a33-f72f690fa3f5`, `vision_lr=9.600163651041579e-06`, `text_lr=3.852141246008874e-06`, `val/t2i_mAP=0.16424861387322684`, checkpoint `/tmp/tao-automl-validation/clip/results/eea8da96-a5a2-4b5c-9a33-f72f690fa3f5/results_dir/train/model_epoch_000_step_00020.pth`.
- Recommendation 1: job `415dcab2-9645-479f-915a-d2e74a9e0743`, `vision_lr=3.06551996648308e-06`, `text_lr=1.1e-06`, `val/t2i_mAP=0.17988698285718407`, checkpoint `/tmp/tao-automl-validation/clip/results/415dcab2-9645-479f-915a-d2e74a9e0743/results_dir/train/model_epoch_000_step_00020.pth`.
- Best recommendation: rec 1 / job `415dcab2-9645-479f-915a-d2e74a9e0743`.

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`
- Inference checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`
- Export checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`
- Resume/retrain checkpoint used: `/workspace/results/train/model_epoch_000_step_00020.pth`
- Were checkpoint paths selected through the proper resolver: yes; actions used the explicit resolved `model_epoch_000_step_00020.pth` artifact rather than the latest symlink
- Any incorrect latest-checkpoint behavior found: no model-skill latest fallback was required or used

Issues found:
- Model skill issues:
  - Parent `gen_trt_engine` metadata inherited the PyTorch image instead of the deploy image and did not expose the engine path as an output.
  - CLIP instructions did not describe the PyTorch 2.6 trusted-checkpoint load override needed by evaluate, inference, export, and resume for TAO CLIP Lightning checkpoints.
  - CLIP instructions overstated full TensorRT deploy support for combined/text ONNX in the current deploy image.
- Config issues:
  - Deploy template defaulted to batch 16 and text inference, which does not match the validated static batch-1 image-only TensorRT path.
  - Deploy metadata typed `dataset.val.datasets[0].caption_dir` as a file instead of a folder and did not expose inference image/text inputs.
  - CLIP train schema advertised model-level AutoML support but had no default searchable parameters, so a default AutoML launch would not tune CLIP unless the user named parameters explicitly.
- Dataset issues:
  - No native CLIP image-caption dataset was found in the inspected S3 prefix.
- Checkpoint issues:
  - No incorrect checkpoint selection found. Trusted checkpoint loads require `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` because of PyTorch 2.6 default checkpoint safety behavior.
- Docker/local execution issues:
  - `clip gen_trt_engine` failed on the combined ONNX and text ONNX with `IndexError: Out of bounds` while parsing text inputs shaped like `(1, 77)`.
  - TensorRT retrieval evaluation failed with the vision-only engine because no text engine could be built.
- Fresh-install issues:
  - Fresh local Docker users need sidecar files (`clip_model_config.yaml` and `clip_model_tokenizer/`) kept with the engine. The deploy command copied them correctly for the validated vision-engine path.

Fixes made:
- Added deploy-image and engine-artifact metadata for the parent CLIP `gen_trt_engine` action.
- Corrected CLIP deploy metadata for caption folders and deploy inference dataset/text inputs.
- Made the deploy template conservative for batch-1 image-only inference, matching the validated static TensorRT engine path.
- Updated CLIP instructions for extracted custom datasets, generated-caption fallback documentation, exact checkpoint handoff, trusted-checkpoint PyTorch load override, and current TensorRT text/full retrieval deployment limits.
- Enabled `train.optim.text_lr` and `train.optim.vision_lr` as packaged CLIP default AutoML parameters in `schemas/train.schema.json` and `schemas/manifest.json`.

Remaining issues:
- Full CLIP TensorRT retrieval deployment remains blocked by the deploy container builder, which assumes all ONNX inputs have image-like height/width dimensions.
- Text-only TensorRT inference is blocked for the same reason because `clip_model_text.onnx` cannot be converted by `clip gen_trt_engine`.

Files changed:
- `models/clip/SKILL.md`
- `models/clip/references/skill_info.yaml`
- `models/clip/references/spec_template_deploy.yaml`
- `models/clip/deploy/skill_info.yaml`
- `models/clip/schemas/train.schema.json`
- `models/clip/schemas/manifest.json`
- `docs/model-validation/clip.md`

Final status:
- Partially validated. PyTorch train/eval/inference/export/resume, default AutoML train routing with two Bayesian recommendations, and image-only deploy gen_trt_engine/inference are validated. Full TensorRT text/retrieval deployment is blocked by the current deploy image.
