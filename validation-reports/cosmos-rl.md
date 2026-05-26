# Model: cosmos-rl

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: not supported by this model skill
- deploy: not supported by this model skill
- prune: not supported by this model skill
- quantize: pass
- retrain: pass for resume-from-checkpoint train variant
- dataset convert: not supported by this model skill
- other: checkpoint resolver handoff to LoRA safetensors: pass
- other: checkpoint resolver handoff to Cosmos policy folder for resume: pass

## Dataset used

- Source: s3://nvcf-storage-handling/data/cosmos_rl_wts_train_subset/ and s3://nvcf-storage-handling/data/cosmos_rl_wts_val_subset/
- Notes: Used real WTS Cosmos-RL data. Training used `annotations_video_fps_30.json` plus `videos.tar.gz` from the train subset; evaluation/inference used `annotations_video_fps_30.json` plus `videos.tar.gz` from the val subset. For runtime cost, smoke specs used 4 train rows and 2 validation rows copied from the real annotations, with matching extracted videos.
- Any dataset compatibility issues: `cosmos_rl_its_subset/` was smaller but blocked by preflight because its annotation records lack required `video_fps`. The WTS train/val subset has `video_fps` on all sampled records and matching media paths.

## Training result

- Training completed: yes
- Best checkpoint produced: yes, but the recorded `best` symlinks point to missing `step_8` paths
- Best checkpoint path: resolved to retained exact LoRA folder `/results/train/20260525235306/safetensors/epoch_2`
- Other checkpoints produced:
  - Cosmos policy checkpoint: `/results/train/20260525235306/checkpoints/epoch_2/policy`
  - Merged model folder used by evaluation: `/results/train/20260525235306/merged/epoch_2`
  - `best/best_score.json` recorded `best_ckpt_abs_dir: /results/train/20260525235306/checkpoints/step_8`
  - Resume loaded `/results/train/20260525235306/checkpoints/epoch_2/policy` and completed successfully
  - Quantize produced `/results/quantize/` with two sharded safetensors files and tokenizer/config assets

## Checkpoint/action verification

- Eval checkpoint used: `/results/train/20260525235306/safetensors/epoch_2` with `model.enable_lora=true` and `model.base_model_path=nvidia/Cosmos-Reason2-8B`
- Inference checkpoint used: `/results/train/20260525235306/safetensors/epoch_2` with `--enable_lora true` and `--base_model_path nvidia/Cosmos-Reason2-8B`
- Export checkpoint used: not applicable; export is not supported
- Resume/retrain checkpoint used: `/results/train/20260525235306/checkpoints/epoch_2/policy`
- Quantize model used: `/results/train/20260525235306/safetensors/epoch_2` with the model-skill quantize compatibility shim
- Were checkpoint paths selected through the proper resolver: yes for the direct local-docker model-skill path. The broken `best` step target was resolved back to the retained `epoch_2` safetensors and policy folders, matching the model skill guidance.
- Any incorrect latest-checkpoint behavior found: no latest-checkpoint behavior. The issue is the known broken `best/step_*` bookkeeping, and the validation avoided it.

## Issues found

- Model skill issues:
  - Direct local-Docker evaluate, inference, and quantize runs log `TAO_API_JOB_ID` status-file tracebacks when no TAO job id is present, even when the action exits 0 and writes outputs. Added guidance that this warning is nonfatal only when the action result is otherwise successful.
- Config issues:
  - Cosmos-RL uses the dedicated `nvcr.io/nvstaging/tao/cosmos_rl:7.0.0-rc-176-multiarch` image and TOML configs, not the generic PyTorch validation image.
  - `automl_policy=on` was not executed because AutoML is a workflow path and workflow skills are out of scope for this validation request.
- Dataset issues:
  - `cosmos_rl_its_subset/` lacks `video_fps`; WTS subsets were used instead.
- Checkpoint issues:
  - `best/checkpoints` and `best/safetensors` point to missing `step_8` folders. The retained exact artifacts are `epoch_2` folders.
  - The resume smoke run loaded the exact `epoch_2/policy` folder successfully. Its own `max_keep=1` cleanup removed the newly produced `epoch_3` artifacts because its `best` entry again referred to a missing `step_9` target.
- Docker/local execution issues:
  - Nonfatal warnings observed: missing `pyarmor_runtime_001219` for telemetry, torchao C++ extension version warning, vLLM Triton kernel import warnings, tokenizer regex warning, and vLLM engine shutdown after successful evaluation.
- Fresh-install issues:
  - A valid HuggingFace token with access to `nvidia/Cosmos-Reason2-8B` is required. The token was passed only via environment variables and cached under the scratch HuggingFace cache.

## Fixes made

- Updated `models/cosmos-rl/SKILL.md` to document nonfatal `TAO_API_JOB_ID` status logging tracebacks for direct local-Docker runs.

## Remaining issues

- No unresolved supported-action failures.
- The broken `best/step_*` symlink behavior remains a container/runtime issue, but the model skill already documents the resolver workaround and this validation used it successfully.

## Files changed

- `models/cosmos-rl/SKILL.md`
- `validation-reports/cosmos-rl.md`

## Final status

- Fully validated for all supported `cosmos-rl` model-skill actions on local-docker with the model skill's default Cosmos-RL container. Train, resume, evaluate, inference, and quantize all passed using exact checkpoint folders rather than `best` or latest shortcuts.
