# Model: cosmos-embed

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: not supported by this model skill
- prune: not supported by this model skill
- quantize: not supported by this model skill
- retrain: pass for resume-from-checkpoint train variant
- dataset convert: not supported by this model skill
- other: export ONNX: pass
- other: export HuggingFace format: pass

## Dataset used

- Source: s3://nvcf-storage-handling/data/cosmos-embed/msrvtt-subset-8/
- Notes: Used the real S3 MSR-VTT-style smoke dataset for Cosmos-Embed: 8 MP4 files under `video/`, `msrvtt_test_1k.json`, and packaged smoke specs. The metadata rows include `video`, `video_id`, and `caption`.
- Any dataset compatibility issues: none. The video filenames matched the metadata and all actions accepted the local `/data/video/*.mp4` glob.

## Training result

- Training completed: yes
- Best checkpoint produced: yes, as the single concrete checkpoint from this one-iteration smoke run
- Best checkpoint path: `/results/train/checkpoints/iter_000000001.pt`
- Other checkpoints produced:
  - `/results/train/cosmos_embed1_model_latest.pth` symlink to `checkpoints/iter_000000001.pt`
  - `/results/train/checkpoints/latest_checkpoint.txt` containing `iter_000000001.pt`
  - Resume validation produced `/results/resume_train/train/checkpoints/iter_000000001.pt` and `/results/resume_train/train/checkpoints/iter_000000002.pt`
  - ONNX export produced `/results/export/cosmos_embed1_combined.onnx` and `/results/export/cosmos_embed1_combined.onnx_data`
  - HuggingFace export produced `/results/export_hf/cosmos_embed1_hf/` with sharded safetensors and model/tokenizer files

## Checkpoint/action verification

- Eval checkpoint used: `/results/train/checkpoints/iter_000000001.pt`
- Inference checkpoint used: `/results/train/checkpoints/iter_000000001.pt`
- Export checkpoint used: `/results/train/checkpoints/iter_000000001.pt` for both ONNX and HuggingFace export modes
- Resume/retrain checkpoint used: `/results/train/checkpoints/iter_000000001.pt`
- Were checkpoint paths selected through the proper resolver: yes for the direct local-docker model-skill path. The checkpoint directory and `latest_checkpoint.txt` were inspected, then every checkpoint-dependent scratch spec was rewritten to the concrete `iter_000000001.pt` file rather than the latest symlink.
- Any incorrect latest-checkpoint behavior found: yes. The packaged downstream spec templates defaulted to `cosmos_embed1_model_latest.pth`, and the skill quick-start export examples used that symlink too. Those templates/examples were fixed to require a resolver-selected exact checkpoint.

## Issues found

- Model skill issues:
  - `spec_template_evaluate.yaml`, `spec_template_inference.yaml`, `spec_template_export_onnx.yaml`, and `spec_template_export_hf.yaml` defaulted to the latest symlink, conflicting with the skill's own checkpoint guidance.
  - `models/cosmos-embed/SKILL.md` export examples also used the latest symlink instead of an exact `iter_#########.pt` checkpoint.
- Config issues:
  - This model does not use the generic TAO PyTorch validation image; the model skill resolves `image=default` to the dedicated `nvcr.io/nvstaging/tao/cosmos-embed:7.0.0-rc-44` container.
  - AutoML is not packaged for this model because there are no schemas, so `automl_policy=on` does not route through AutoML.
- Dataset issues:
  - None.
- Checkpoint issues:
  - The actual action runs used exact checkpoints and passed. The template defaults were the fragile part and were fixed.
- Docker/local execution issues:
  - The container requires the documented startup preamble `python -m pip install "protobuf<7"`.
  - Even with `model.pretrained_model_path: null`, the container downloaded `google-bert/bert-base-uncased`; `HF_TOKEN` was passed only via environment and was not written into specs or reports.
- Fresh-install issues:
  - Fresh runs need network or a persistent HuggingFace cache for the Q-Former dependency.

## Fixes made

- Set checkpoint defaults to `null` in the Cosmos-Embed evaluate, inference, ONNX export, and HuggingFace export spec templates so callers must supply the resolver-selected exact checkpoint.
- Updated the Cosmos-Embed quick-start export examples and checkpoint section to use an exact `checkpoints/iter_#########.pt` checkpoint rather than `cosmos_embed1_model_latest.pth`.

## Remaining issues

- No unresolved model-skill action failures.
- Q-Former/HuggingFace dependency downloads remain a fresh-install requirement, but this is already documented in the model skill.

## Files changed

- `models/cosmos-embed/SKILL.md`
- `models/cosmos-embed/references/spec_template_evaluate.yaml`
- `models/cosmos-embed/references/spec_template_inference.yaml`
- `models/cosmos-embed/references/spec_template_export_onnx.yaml`
- `models/cosmos-embed/references/spec_template_export_hf.yaml`
- `validation-reports/cosmos-embed.md`

## Final status

- Fully validated for all supported `cosmos-embed` model-skill actions on local-docker with the model skill's default Cosmos-Embed container. Train, resume, evaluate, inference, ONNX export, and HuggingFace export all passed after fixing the checkpoint-template guidance.
