Model: cosmos-embed

Supported actions tested:
- train: pass with container protobuf pin
- eval: pass with exact checkpoint and container protobuf pin
- inference: pass with exact checkpoint and container protobuf pin
- export: pass for ONNX/external-data export and HuggingFace-format export
- deploy: unsupported
- prune: unsupported
- retrain/resume: pass with exact checkpoint and `model.fsdp_shard_size=1`
- quantize: unsupported
- dataset convert: unsupported
- other: AutoML workflow unsupported; no model schemas are packaged

Dataset used:
- Source: `s3://nvcf-storage-handling/data/cosmos-embed/msrvtt-subset-8/`
- Notes: Used the S3-provided MSR-VTT subset with `msrvtt_test_1k.json`, eight MP4 videos, and smoke-sized model specs.
- Any dataset compatibility issues: none for the direct model CLI; paths must be staged locally because the CLI consumes local globs, not raw S3 URIs.

Training result:
- Training completed: yes
- Best checkpoint produced: no dedicated best-named artifact
- Best checkpoint path: not emitted separately; the validated downstream checkpoint was `/tmp/tao-model-validation/cosmos-embed/results/train/checkpoints/iter_000000001.pt`
- Other checkpoints produced: `/tmp/tao-model-validation/cosmos-embed/results/resume_exact/train/checkpoints/iter_000000002.pt`; `cosmos_embed1_model_latest.pth` symlinks were created in train and resume result folders

Checkpoint/action verification:
- Eval checkpoint used: `/results/train/checkpoints/iter_000000001.pt`
- Inference checkpoint used: `/results/train/checkpoints/iter_000000001.pt`
- Export checkpoint used: `/results/train/checkpoints/iter_000000001.pt`
- Resume/retrain checkpoint used: `/results/train/checkpoints/iter_000000001.pt`
- Were checkpoint paths selected through the proper resolver: yes; downstream actions used the exact iteration checkpoint from `checkpoints/latest_checkpoint.txt`/train output rather than the latest symlink
- Any incorrect latest-checkpoint behavior found: the S3/default specs point to `cosmos_embed1_model_latest.pth`; validation overrode that to the exact iteration checkpoint and the skill now documents exact checkpoint resolution

Issues found:
- Model skill issues:
  - `automl_enabled` was true, but no Cosmos-Embed schemas are packaged; this could route users toward a workflow skill that cannot run for this model.
  - Quick-start commands did not include the container protobuf pin required by the current image.
  - Checkpoint guidance used the latest symlink instead of the exact `iter_#########.pt` artifact.
- Config issues:
  - Single-GPU resume from a consolidated checkpoint fails unless `model.fsdp_shard_size=1`.
  - Setting `qformer_pretrain_ckpt: null` does not stop the current container from downloading `google-bert/bert-base-uncased`.
- Dataset issues:
  - None after staging the S3 subset locally.
- Checkpoint issues:
  - Default specs use `cosmos_embed1_model_latest.pth`; exact iteration checkpoint use is safer for eval, inference, export, and resume.
- Docker/local execution issues:
  - The current image has `wandb==0.21.0` with `protobuf==7.34.1`, causing `cannot import name 'Imports' from 'wandb.proto.wandb_telemetry_pb2'`. Pinning `protobuf<7` inside the container fixes the import.
  - Fresh ephemeral containers need `HF_TOKEN` or a persistent HF cache for the BERT/Q-Former download path.
- Fresh-install issues:
  - The Cosmos-Embed image is not usable as-is for these actions until protobuf is pinned below 7.

Fixes made:
- Marked Cosmos-Embed as not AutoML-enabled until model schemas exist.
- Added the protobuf pin to Cosmos-Embed action command metadata so model-skill launches use the working container preamble.
- Added `model.fsdp_shard_size: 1` to Cosmos-Embed spec templates for local single-GPU runs.
- Updated instructions with the protobuf pin, W&B-disabled environment, HF token/cache guidance, exact checkpoint resolver behavior, and the resume FSDP workaround.

Remaining issues:
- The published container should be rebuilt with compatible `wandb`/`protobuf` versions so users do not need an in-container pip pin.
- The container still downloads the BERT/Q-Former component even when smoke specs set `qformer_pretrain_ckpt: null`.

Files changed:
- `models/cosmos-embed/SKILL.md`
- `models/cosmos-embed/references/skill_info.yaml`
- `models/cosmos-embed/references/spec_template_train.yaml`
- `models/cosmos-embed/references/spec_template_evaluate.yaml`
- `models/cosmos-embed/references/spec_template_inference.yaml`
- `models/cosmos-embed/references/spec_template_export_onnx.yaml`
- `models/cosmos-embed/references/spec_template_export_hf.yaml`
- `docs/model-validation/cosmos-embed.md`

Final status:
- Fully validated with documented container startup workaround.
