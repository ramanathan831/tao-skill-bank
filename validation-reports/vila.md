Model: vila

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: unsupported, not present in the VILA schema manifest or skill_info actions
- deploy: unsupported, not present in the VILA schema manifest or skill_info actions
- prune: unsupported, not present in the VILA schema manifest or skill_info actions
- retrain: unsupported, no retrain/resume action or train resume field is exposed
- dataset convert: unsupported, no dataset_convert action is exposed
- quantize: unsupported, not present in the VILA schema manifest or skill_info actions
- AutoML/HPO: not run; VILA declares AutoML metadata, but this validation was constrained to model skills only
- checkpoint compatibility check: fail for the only S3 checkpoint under `checkpoints/`, then pass with a compatible VILA/NVILA base model

Dataset used:
- Source: `s3://nvcf-storage-handling/data/vila_ft_youcook2_subsampled/`
- Source: `s3://nvcf-storage-handling/data/vila_ft_youcook2_subsampled_yaml/dataset.yaml`
- Source: `s3://nvcf-storage-handling/data/vila_lita_ft_youcook2/`
- Source: `s3://nvcf-storage-handling/data/vila_lita_ft_youcook2_val_yaml/dataset.yaml`
- Source: `s3://nvcf-storage-handling/data/vlm_inference/videos/test_video.mp4`
- Notes: Train used one real YouCook2 video annotation from the S3 subsampled set (`GLd3aX16zBg_0.mkv`). Evaluate used one real RTL validation item (`validation/104/Nbh64ntT3EM_3`). Inference used the provided S3 video sample.
- Any dataset compatibility issues: The S3 dataset YAMLs contain `aws://` members, so the local-docker run needed local staged annotations/media and container-visible YAML paths. The VILA model-skill wrapper correctly patched eval YAML paths into `results/evaluate/dataset.local.yaml`.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint; VILA LoRA train produced the final PEFT output folder and one step checkpoint
- Best checkpoint path: not applicable for this one-step LoRA run
- Trained model folder: `/workspace/run/results/train/lora`
- Other checkpoints produced: `/workspace/run/results/train/lora/checkpoint-1`
- Base model used: `Efficient-Large-Model/NVILA-Lite-2B`
- Container image: `nvcr.io/nvstaging/tao/vila-finetuning-sop:20250722`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/train/lora` with `evaluate.model_base=Efficient-Large-Model/NVILA-Lite-2B`
- Inference checkpoint used: `/workspace/run/results/train/lora` with `inference.model_base=Efficient-Large-Model/NVILA-Lite-2B`
- Export checkpoint used: unsupported
- Resume/retrain checkpoint used: unsupported
- Were checkpoint paths selected through the proper resolver: yes for the model-skill wiring; `skill_info.yaml` maps evaluate/inference `model_path` through `parent_model_folder` and `model_base` through `ptm_if_no_resume_model`. Direct local-docker validation used those exact resolved values and did not do a latest-file lookup.
- Any incorrect latest-checkpoint behavior found: no. VILA downstream actions consume the trained LoRA folder, not a guessed `.pth` or latest checkpoint file.

Issues found:
- Model skill issues:
  - The skill instructions implied export/deploy could be model-skill actions, but VILA only exposes train, evaluate, and inference.
  - The skill did not state that `model_path` must be a VILA/NVILA wrapper checkpoint with `model_type: llava_llama`.
- Config issues:
  - The only checkpoint present under `s3://nvcf-storage-handling/data/checkpoints/` is `wts_81k_sft/step_44`, whose `config.json` has `model_type: qwen2_5_vl`. The VILA container's Transformers 4.46.0 build does not recognize that raw architecture through the VILA train path.
- Dataset issues:
  - Remote `aws://` paths in the S3 YAML must be replaced or patched with local container paths for local-docker runs.
- Checkpoint issues:
  - Raw Qwen2.5-VL checkpoints fail before training with `Transformers does not recognize this architecture`.
- Docker/local execution issues:
  - No blocking local-docker issues after using the compatible NVILA base and container-visible data paths.
- Fresh-install issues:
  - A fresh install using only the provided S3 checkpoint prefix is blocked for VILA because no compatible `llava_llama` base checkpoint is present there.

Fixes made:
- Updated `models/vila/SKILL.md` to document the compatible base checkpoint requirement and the raw Qwen2.5-VL failure mode.
- Updated `models/vila/SKILL.md` so it only describes evaluate/inference as non-train model-skill actions.
- Added this per-model validation report.

Remaining issues:
- The S3 checkpoint prefix should include a compatible VILA/NVILA base checkpoint folder, or the model skill inputs should clearly direct users to a supported VILA/NVILA checkpoint source.
- AutoML/HPO remains unvalidated for this pass because running it would leave the model-skill-only validation path.

Files changed:
- `models/vila/SKILL.md`
- `validation-reports/vila.md`

Final status:
- Partially validated: train, evaluate, and inference passed end-to-end through the VILA model skill wrappers; unsupported actions are not exposed; S3-provided base checkpoint coverage remains blocked by checkpoint incompatibility.
