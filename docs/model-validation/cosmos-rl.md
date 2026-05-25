Model: cosmos-rl

Validated on 2026-05-25 with `platform=local-docker`, `image=default`
resolved to `nvcr.io/nvstaging/tao/cosmos_rl:7.0.0-rc-176-multiarch`.

Supported actions tested:
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass with exact trained LoRA folder
- inference: pass after adding the packaged model-skill action metadata
- export: unsupported
- deploy: unsupported
- prune: unsupported
- retrain/resume: pass with exact Cosmos checkpoint policy folder
- dataset convert: unsupported
- quantize: pass after adding the packaged model-skill action metadata
- other: HF gated model access check: pass; local-docker image import/mount check: pass after using a non-`/workspace` mount

Dataset used:
- Train source: `s3://nvcf-storage-handling/data/cosmos_rl_wts_train_subset/`
- Eval source: `s3://nvcf-storage-handling/data/cosmos_rl_wts_val_subset/`
- Inference/quantize source: `s3://nvcf-storage-handling/data/cosmos_rl_wts_train_subset/`
- Files: `annotations_video_fps_30.json` and `videos.tar.gz`
- Notes: the archive was extracted locally. The annotations store video paths
  relative to the archive's `videos/` directory, so local Docker specs used the
  extracted media root ending in `media/videos`. Train/AutoML used bounded real
  WTS subsets. Inference and quantize used one real annotation/video pair from
  the same S3 dataset.
- Any dataset compatibility issues: `cosmos_rl_its_subset` was inspected but
  its annotations do not include `video_fps`, which the SFT loader requires.
  The WTS validation subset exists in S3, but the bounded local run reused WTS
  train media for several smoke actions to keep runtime and disk use practical.

Training result:
- Training completed: yes
- Best checkpoint produced: yes, but the container's `best` symlinks were broken
- Best checkpoint path: resolved to
  `/tmp/tao-model-validation/cosmos-rl/results/train/20260525024058/safetensors/epoch_1`
  for evaluate and
  `/tmp/tao-model-validation/cosmos-rl/results/train/20260525024058/checkpoints/epoch_1/policy`
  for resume
- Other checkpoints produced:
  `/tmp/tao-model-validation/cosmos-rl/results/resume_exact/20260525024808/safetensors/epoch_2`
  and
  `/tmp/tao-model-validation/cosmos-rl/results/resume_exact/20260525024808/checkpoints/epoch_2/policy`

AutoML default train route:
- Rerun completed: yes, through `AutoMLRunner` + `DockerSDK` on local Docker.
- Algorithm/config: Bayesian, `automl_max_recommendations=2`, metric `val/avg_loss`, direction minimize, 1 epoch, 8 train records, 4 validation records, one GPU.
- Search parameters/ranges: `train.optm_lr` constrained to `5e-7..2e-6` and `policy.lora.lora_dropout` constrained to `0.0..0.02`.
- Recommendation 0: job `f69d57c9-ac3e-416c-a3f2-a45ca37008fb`, `train.optm_lr=1.1157898708340944e-06`, `policy.lora.lora_dropout=0.01559456423653458`, `val/avg_loss=12.780737400054932`, checkpoints under `/tmp/tao-automl-validation/cosmos-rl/results/f69d57c9-ac3e-416c-a3f2-a45ca37008fb/train_output_dir/20260525085453/{safetensors,checkpoints}/epoch_1`.
- Recommendation 1: job `1ea439a1-99eb-4a29-a06f-c666fe6c99b2`, `train.optm_lr=1.9857283768336603e-06`, `policy.lora.lora_dropout=1e-07`, `val/avg_loss=12.763819217681885`, checkpoints under `/tmp/tao-automl-validation/cosmos-rl/results/1ea439a1-99eb-4a29-a06f-c666fe6c99b2/train_output_dir/20260525085854/{safetensors,checkpoints}/epoch_1`.
- Best recommendation: rec 1 / job `1ea439a1-99eb-4a29-a06f-c666fe6c99b2`.

Inference result:
- Job: `8d57b6c2-dec8-4ac7-9a32-feedaa49ba7c`
- Model path: `nvidia/Cosmos-Reason2-8B`
- Media: `/tao-workspace/media/videos/20230929_28_SY3_T1/overhead_view/20230929_28_SY3_T1_192.168.0.14_1.mp4`
- Result: pass, one video processed, zero failures, result files written under `/tmp/tao-automl-validation/cosmos-rl-actions/results/inference/inference_results/`

Quantize result:
- Job: `eb8daca7-b6b6-43d5-8921-509ca20ba3fe`
- Model path: `nvidia/Cosmos-Reason2-8B`
- Calibration data: one real WTS annotation/video pair
- Config: `num_calibration_samples=1`, `max_sequence_length=4096`, `quantization_scheme=W4A16`, `skip_test_generation=true`
- Result: pass, quantized model saved under `/tmp/tao-automl-validation/cosmos-rl-actions/results/quantize`
- Output artifacts include `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`, tokenizer/config files, and `recipe.yaml`.

Checkpoint/action verification:
- Eval checkpoint used: `/tao-workspace/results/train/20260525024058/safetensors/epoch_1`
- Inference checkpoint used: none for base-model smoke inference; the action metadata maps `model_path` through `parent_model_folder` when a parent job is provided
- Export checkpoint used: not applicable
- Quantize checkpoint used: none for base-model quantize; the action metadata maps `model.model_path` through `parent_model_folder` when a parent job is provided
- Resume/retrain checkpoint used: `/tao-workspace/results/train/20260525024058/checkpoints/epoch_1/policy`
- Were checkpoint paths selected through the proper resolver: yes for direct
  model-skill validation after resolving the model-specific epoch artifact.
  The added action metadata now exposes `parent_model_folder` mappings for
  inference and quantize instead of relying on stale markdown-only mappings.
- Any incorrect latest-checkpoint behavior found: yes. `best/best_score.json`
  recorded `checkpoints/step_8`, and `best/{checkpoints,safetensors}` pointed
  at non-existent `step_8` targets, while the actual artifacts were `epoch_1`.
  The resume helper also scans `step_*` folders when `train.resume=true`, so
  exact string resume paths are required for current epoch-named local Docker
  checkpoints.

Issues found:
- Model skill issues:
  - Cosmos-RL inference and quantize were supported by the container but absent from `references/skill_info.yaml` and `schemas/manifest.json`; prior validation trusted the packaged metadata and missed them.
  - The old instructions said inference/quantize were not packaged even though the container exposes `cosmos-rl-inference` and `cosmos-rl-quantize`.
  - The image-resolution instructions referenced `models/cosmos-rl/config.json`, which is not packaged.
  - Direct local Docker examples did not warn that mounting over `/workspace` hides the package inside the Cosmos-RL image and causes `ModuleNotFoundError: No module named 'cosmos_rl'`.
  - The packaged train schema modeled `train.resume` as bool-only even though the runtime supports an exact checkpoint string.
- Config issues:
  - Direct container training should use the bare HuggingFace id or a local HF snapshot path. `hf_model://...` is only appropriate for SDK/platform predownload paths.
  - For the WTS archive, the usable media root is the extracted `videos/` directory, not the parent extraction directory.
  - The packaged train schema listed `custom.val_dataset` as an empty collection, so AutoML runner key validation rejected `custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` even though the model skill requires them for validation-backed AutoML.
  - The first quantize wrapper version used literal `{}` defaults inside a command that is later processed by `.format(config_path=...)`, causing `IndexError: Replacement index 0 out of range`.
  - The first quantize smoke override used `max_sequence_length=128`, which was too small for the real video calibration sample and caused an image/video token mismatch.
- Dataset issues:
  - `cosmos_rl_its_subset` is incompatible until `video_fps` is added to every annotation record.
- Checkpoint issues:
  - Best symlinks and `best_score.json` can reference missing `step_*` folders while actual checkpoint folders are `epoch_*`.
  - `train.resume=true` is fragile with epoch-named checkpoint folders; use the exact `.../checkpoints/epoch_N/policy` folder.
- Docker/local execution issues:
  - Do not bind-mount the run directory at `/workspace`; use a neutral path such as `/tao-workspace`.
  - The image emits non-fatal TAO status warnings when `TAO_API_JOB_ID` is unset during direct local execution.
  - The image emits non-fatal torchao/Triton compatibility warnings during load/evaluate/quantize.
- Fresh-install issues:
  - The user must have accepted the gated HuggingFace model agreement and provide `HF_TOKEN`.
  - The base model cache is large; this run used a large HF cache and quantize produced multi-GB safetensors.

Fixes made:
- Added Cosmos-RL `inference` action metadata, schema, and spec template.
- Added Cosmos-RL `quantize` action metadata, schema, and spec template.
- Added `parent_model_folder` mappings for inference and quantize in `spec_params`.
- Added a quantize wrapper compatibility shim for the current image's `compressed_tensors`/`llmcompressor` import mismatch.
- Set the quantize smoke default to `max_sequence_length=4096` so real video calibration samples do not get truncated to an invalid token sequence.
- Documented `references/skill_info.yaml`/`versions.yaml` image resolution instead of a missing `config.json`.
- Updated Cosmos-RL instructions so train/evaluate/inference/quantize are the declared model-skill actions.
- Added direct local Docker mount guidance to avoid hiding `/workspace/cosmos_rl`.
- Clarified direct Docker model path handling versus SDK `hf_model://` predownload handling.
- Added model-specific checkpoint resolver guidance for broken `best/step_*` symlinks, exact `safetensors/epoch_N` evaluation, and exact `checkpoints/epoch_N/policy` resume.
- Updated the train schema so `train.resume` may be a bool or an exact checkpoint string.
- Added `custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` to the packaged train schema/template so AutoML train can accept explicit validation data.

Remaining issues:
- The current container still creates broken `best` symlinks for epoch-based saves; the model skill now documents the required resolver behavior.
- Direct local eval/inference/quantize log TAO status warnings when `TAO_API_JOB_ID` is absent, but the actions completed successfully.
- Export, prune, deploy, and dataset convert are not packaged actions for Cosmos-RL.

Files changed:
- `models/cosmos-rl/SKILL.md`
- `models/cosmos-rl/references/skill_info.yaml`
- `models/cosmos-rl/references/spec_template_inference.yaml`
- `models/cosmos-rl/references/spec_template_quantize.yaml`
- `models/cosmos-rl/schemas/inference.schema.json`
- `models/cosmos-rl/schemas/manifest.json`
- `models/cosmos-rl/schemas/quantize.schema.json`
- `models/cosmos-rl/schemas/train.schema.json`
- `models/cosmos-rl/references/spec_template_train.yaml`
- `docs/model-validation/cosmos-rl.md`
- `docs/model-validation/action-run-inventory.md`

Final status:
- Fully validated for all actions declared by the Cosmos-RL model skill on `local-docker`: train, default AutoML train routing with two Bayesian recommendations, evaluate, inference, quantize, and checkpoint-dependent resume.
