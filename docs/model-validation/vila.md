# Model: vila

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The successful run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the VILA model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- train: pass, AutoML default route with two Bayesian recommendations over `train.learning_rate` using `loss` minimize
- eval/evaluate: pass after fixing RTL time-token handling in the VILA model skill wrapper
- inference: pass
- export: unsupported by the packaged VILA model skill
- deploy: unsupported by the packaged VILA model skill
- prune: unsupported by the packaged VILA model skill
- quantize: unsupported by the packaged VILA model skill
- retrain/resume: unsupported as a separate VILA model skill action
- dataset convert: unsupported by the packaged VILA model skill
- other: no deploy sub-skill, export, prune, quantize, or standalone retrain action is packaged for VILA

## Dataset used

- Source:
  - Train: `s3://nvcf-storage-handling/data/vila_ft_youcook2_subsampled_yaml/dataset.yaml`
  - Train annotations/media: `s3://nvcf-storage-handling/data/vila_ft_youcook2_subsampled/annotations.json`, `s3://nvcf-storage-handling/data/vila_ft_youcook2_subsampled/dataset.tar.gz`
  - Evaluate: `s3://nvcf-storage-handling/data/lita_subset_val_yaml/dataset.yaml`
  - Evaluate annotations/media: `s3://nvcf-storage-handling/data/lita_subset/youcookii_val_rtl.json`, `s3://nvcf-storage-handling/data/lita_subset/videos.tar.gz`
  - Inference media: `s3://nvcf-storage-handling/data/vlm_inference/images/000777.png`
- Notes: the VILA dataset YAML files contain `aws://` member paths. The model skill now declares the underlying annotation/media fields as inputs so the runner downloads them and patches a local dataset YAML before invoking the VILA flat entrypoints.
- Any dataset compatibility issues: `s3://nvcf-storage-handling/data/checkpoints/wts_81k_sft/step_44/` was not compatible with the VILA image because its config uses `model_type: qwen2_5_vl`. `Efficient-Large-Model/VILA1.5-3b` loaded but lacks a tokenizer chat template in this image. `Efficient-Large-Model/Llama-3-VILA1.5-8B` was compatible and was used as the base model.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations over `train.learning_rate`
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/vila/docker_results_localdata_fastckpt/fc943d36-bc6a-4f2e-8096-f0a2d13a262e/results_dir/lora`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/vila/docker_results_localdata_fastckpt/fc943d36-bc6a-4f2e-8096-f0a2d13a262e/results_dir/lora/checkpoint-50`
  - `/tmp/tao-automl-validation/vila/docker_results_localdata_fastckpt/32a1cc05-91e9-4f08-8a3b-216027a8bb62/results_dir/lora`
  - `/tmp/tao-automl-validation/vila/docker_results_localdata_fastckpt/32a1cc05-91e9-4f08-8a3b-216027a8bb62/results_dir/lora/checkpoint-50`
- AutoML recommendations:
  - rec 0: job `fc943d36-bc6a-4f2e-8096-f0a2d13a262e`, `train.learning_rate=0.00010791230925396626`, final `loss=2.5671`, selected as best
  - rec 1: job `32a1cc05-91e9-4f08-8a3b-216027a8bb62`, final `loss=3.4954`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 LoRA folder `/tmp/tao-automl-validation/vila/docker_results_localdata_fastckpt/fc943d36-bc6a-4f2e-8096-f0a2d13a262e/results_dir/lora` with base `Efficient-Large-Model/Llama-3-VILA1.5-8B`; evaluate job `6f7f709c-79d4-4fed-b416-0291db6212c9` produced `iou=0.4307709283334269` and `precision@0.5=0.3275862068965517`.
- Inference checkpoint used: best AutoML rec 0 LoRA folder with the same base model; inference job `41952eea-07b0-4a3b-bce5-363d721bcd48` returned "The image shows a warehouse with boxes and pallets on the floor."
- Export checkpoint used: unsupported by the packaged VILA model skill.
- Resume/retrain checkpoint used: unsupported by the packaged VILA model skill.
- Were checkpoint paths selected through the proper resolver: partially. The validation used the AutoML best-rec result and passed the resolved LoRA folder explicitly to avoid workflow skills. The model skill now declares `parent_model_folder` for evaluate/inference and `ptm_if_no_resume_model` for the base model so fresh installs have model-specific resolver metadata instead of guessing a latest `.pth`.
- Any incorrect latest-checkpoint behavior found: no. VILA produces LoRA folders and `checkpoint-50` directories, not a generic `model.pth`; downstream actions used the selected best AutoML LoRA folder and did not guess by newest file.

## Issues found

- Model skill issues:
  - `versions.yaml` mapped `tao_toolkit.vila` to `nvcr.io/nvidia/tao/tao-toolkit:6.26.3-vila`, which has no Docker manifest.
  - `skill_info.yaml` used non-existent `vila train/evaluate/inference -e` dispatch commands, but the image exposes `vila-train`, `vila-evaluate`, and `vila-inference`.
  - Train metadata did not expose `train.dataset.data_path` or `train.dataset.media_dir`, so remote dataset YAML member paths were not localized before VILA training.
  - Evaluate metadata used the stale `eval.dataset_yaml_path` key and did not expose `evaluate.data_path` / `evaluate.video_dir`.
  - Evaluate/inference did not declare PEFT base-model or parent LoRA-folder resolver mappings.
  - RTL evaluation crashed when PEFT/base loading left `model.config.num_time_tokens` as `None`.
- Config issues:
  - The packaged training schema did not declare any AutoML default parameter even though VILA is AutoML-enabled.
  - The local `tao_automl.schema.generate_schema('vila', 'train')` path still does not infer the packaged default AutoML parameter, so validation supplied `automl_hyperparameters=['train.learning_rate']` while also fixing the packaged model schema.
- Dataset issues:
  - The S3 VILA dataset YAMLs require localizing member annotation/media paths before invoking VILA.
  - The only S3 checkpoint family under `data/checkpoints/wts_81k_sft/step_44/` is not compatible with this VILA image.
- Checkpoint issues:
  - No fragile latest `.pth` behavior was found. The important handoff is folder-based LoRA plus explicit base model.
- Docker/local execution issues:
  - The valid VILA image is staged at `nvcr.io/nvstaging/tao/vila-finetuning-sop:20250722`; the old default image tag cannot run.
- Fresh-install issues:
  - Without the metadata fixes, a fresh install would fail before or during train/evaluate because the image entrypoints, remote dataset member paths, AutoML parameter metadata, and PEFT eval/inference handoff were not wired correctly.

## Fixes made

- Updated `tao_toolkit.vila` in `versions.yaml` to `nvcr.io/nvstaging/tao/vila-finetuning-sop:20250722`.
- Wrapped the VILA flat entrypoints in `models/vila/references/skill_info.yaml`.
- Added train inputs for dataset YAML, annotation file, and media directory/archive.
- Added evaluate inputs for dataset YAML, annotation file, video directory/archive, PEFT base model, and RTL time-token fields.
- Added inference input for PEFT base model and media file.
- Added VILA `spec_params` mappings for train output/PTM fallback and evaluate/inference parent LoRA folder plus base model handoff.
- Added `train.learning_rate` as the packaged AutoML default parameter and marked it AutoML-enabled in the train schema.
- Added VILA evaluate schema/template fields for `evaluate.data_path`, `evaluate.video_dir`, `evaluate.num_time_tokens`, and `evaluate.time_token_format`, and allowed `youcook2_val_rtl`.
- Updated VILA instructions for AutoML default routing, valid entrypoints, dataset member localization, PEFT handoff, and RTL evaluation.
- Updated the per-network action inventory.

## Remaining issues

- VILA export, deploy, prune, quantize, dataset convert, and standalone retrain are not packaged model skill actions.
- The `tao_automl.schema.generate_schema` helper still does not infer VILA's packaged AutoML default parameter from the skill bank; the model skill now carries the correct schema metadata, but the helper path remains outside this model-skill-only fix scope.

## Files changed

- `versions.yaml`
- `models/vila/SKILL.md`
- `models/vila/references/skill_info.yaml`
- `models/vila/references/spec_template_evaluate.yaml`
- `models/vila/schemas/evaluate.schema.json`
- `models/vila/schemas/manifest.json`
- `models/vila/schemas/train.schema.json`
- `docs/model-validation/vila.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Fully validated for packaged VILA model skill actions: AutoML train, evaluate, and inference pass through the real model skill workflow. Non-packaged actions are documented as unsupported.
