Model: cosmos-rl

Validated on 2026-05-25 with `platform=local-docker`, `image=default`
resolved to `nvcr.io/nvstaging/tao/cosmos_rl:7.0.0-rc-176-multiarch`.

Supported actions tested:
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass with exact trained LoRA folder
- inference: unsupported; not declared in `references/skill_info.yaml` or `schemas/manifest.json`
- export: unsupported
- deploy: unsupported
- prune: unsupported
- retrain/resume: pass with exact Cosmos checkpoint policy folder
- dataset convert: unsupported
- quantize: unsupported
- other: HF gated model access check: pass; local-docker image import/mount check: pass after using a non-`/workspace` mount

Dataset used:
- Source: `s3://nvcf-storage-handling/data/cosmos_rl_wts_train_subset/`
- Eval source: `s3://nvcf-storage-handling/data/cosmos_rl_wts_val_subset/`
- Files: `annotations_video_fps_30.json` and `videos.tar.gz`
- Notes: the archive was extracted locally. The annotations store video paths
  relative to the archive's `videos/` directory, so local Docker specs used the
  extracted media root ending in `media/videos`. An 8-record smoke annotation
  file was derived from the S3 annotations for the run; all records retained
  real WTS media paths and `video_fps`.
- Any dataset compatibility issues: `cosmos_rl_its_subset` was inspected but
  its annotations do not include `video_fps`, which the SFT loader requires.
  The WTS validation subset exists in S3, but this validation used the WTS train
  subset for train, validation, and evaluate to keep the single-model local run
  bounded while still using compatible real S3 data.

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

Checkpoint/action verification:
- Eval checkpoint used: `/tao-workspace/results/train/20260525024058/safetensors/epoch_1`
- Inference checkpoint used: not applicable; inference is not a packaged Cosmos-RL model-skill action
- Export checkpoint used: not applicable
- Resume/retrain checkpoint used: `/tao-workspace/results/train/20260525024058/checkpoints/epoch_1/policy`
- Were checkpoint paths selected through the proper resolver: yes for direct
  model-skill validation after resolving the model-specific epoch artifact. SDK
  `parent_model_folder` resolution was not invoked because workflow/SDK paths
  were explicitly out of scope.
- Any incorrect latest-checkpoint behavior found: yes. `best/best_score.json`
  recorded `checkpoints/step_8`, and `best/{checkpoints,safetensors}` pointed
  at non-existent `step_8` targets, while the actual artifacts were
  `epoch_1`. The resume helper also scans `step_*` folders when
  `train.resume=true`, so exact string resume paths are required for current
  epoch-named local Docker checkpoints.

Issues found:
- Model skill issues:
  - The image-resolution instructions referenced `models/cosmos-rl/config.json`, which is not packaged.
  - The instructions included quantize/inference examples even though the packaged action metadata only supports train and evaluate.
  - Direct local Docker examples did not warn that mounting over `/workspace` hides the package inside the Cosmos-RL image and causes `ModuleNotFoundError: No module named 'cosmos_rl'`.
  - The packaged train schema modeled `train.resume` as bool-only even though the runtime supports an exact checkpoint string.
- Config issues:
  - Direct container training should use the bare HuggingFace id or a local HF snapshot path. `hf_model://...` is only appropriate for SDK/platform predownload paths.
  - For the WTS archive, the usable media root is the extracted `videos/` directory, not the parent extraction directory.
  - The packaged train schema listed `custom.val_dataset` as an empty collection, so AutoML runner key validation rejected `custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` even though the model skill requires them for validation-backed AutoML.
- Dataset issues:
  - `cosmos_rl_its_subset` is incompatible until `video_fps` is added to every annotation record.
- Checkpoint issues:
  - Best symlinks and `best_score.json` can reference missing `step_*` folders while actual checkpoint folders are `epoch_*`.
  - `train.resume=true` is fragile with epoch-named checkpoint folders; use the exact `.../checkpoints/epoch_N/policy` folder.
- Docker/local execution issues:
  - Do not bind-mount the run directory at `/workspace`; use a neutral path such as `/tao-workspace`.
  - The image emits non-fatal TAO status warnings when `TAO_API_JOB_ID` is unset during direct local execution.
  - The image emits non-fatal torchao/Triton compatibility warnings during load/evaluate.
- Fresh-install issues:
  - The user must have accepted the gated HuggingFace model agreement and provide `HF_TOKEN`.
  - The base model cache is large; this run used about 33 GB for HF cache and produced large full-policy checkpoints.

Fixes made:
- Documented `references/skill_info.yaml`/`versions.yaml` image resolution instead of a missing `config.json`.
- Documented that packaged Cosmos-RL model-skill actions are train and evaluate only.
- Added direct local Docker mount guidance to avoid hiding `/workspace/cosmos_rl`.
- Clarified direct Docker model path handling versus SDK `hf_model://` predownload handling.
- Added model-specific checkpoint resolver guidance for broken `best/step_*` symlinks, exact `safetensors/epoch_N` evaluation, and exact `checkpoints/epoch_N/policy` resume.
- Updated the train schema so `train.resume` may be a bool or an exact checkpoint string.
- Added `custom.val_dataset.annotation_path` and `custom.val_dataset.media_path` to the packaged train schema/template so AutoML train can accept explicit validation data.

Remaining issues:
- The current container still creates broken `best` symlinks for epoch-based saves; the model skill now documents the required resolver behavior.
- Direct local eval logs TAO status warnings when `TAO_API_JOB_ID` is absent, but the action completed successfully.
- Quantize, export, prune, deploy, dataset convert, and standalone model-skill inference are not packaged actions for Cosmos-RL.

Files changed:
- `models/cosmos-rl/SKILL.md`
- `models/cosmos-rl/schemas/train.schema.json`
- `models/cosmos-rl/references/spec_template_train.yaml`
- `docs/model-validation/cosmos-rl.md`

Final status:
- Fully validated for all actions declared by the Cosmos-RL model skill on `local-docker`: train, default AutoML train routing with two Bayesian recommendations, evaluate, and checkpoint-dependent resume.
