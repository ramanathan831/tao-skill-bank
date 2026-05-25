# Model: sparse4d

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the Sparse4D model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- dataset convert: pass through the Data Services `annotations convert -e` action
- train: pass, AutoML default route with two Bayesian recommendations using `val_mAP` maximize
- eval/evaluate: pass
- inference: pass
- export: pass
- deploy: unsupported by the packaged Sparse4D model skill
- prune: unsupported by the packaged Sparse4D model skill
- quantize: fail in the current PyT image after correct checkpoint handoff; ONNX quantize variant also fails because the image lacks `modelopt.onnx.quantization`
- retrain/resume: pass through `sparse4d train -e` with `train.resume_training_checkpoint_path`
- other: no standalone retrain or deploy sub-skill is packaged

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_sparse4d_train/`
- Notes: used the AICity subset with calibration, ground truth, map, three MP4 camera streams, and H5 depth maps under `train/subsetscene`. The `purpose_built_models_sparse4d_train_wo_depthmaps/` variant was not used because the default Sparse4D spec enables H5 depth supervision.
- Any dataset compatibility issues: Data Services conversion produced train split OVPKL files only, so smoke validation reused `train/subsetscene+bev-sensor-random-0_infos_train.pkl` for train, val, and test annotations. The converter extracted the full videos to RGB frames even with `aicity.num_frames=30`, so the local scratch grew to about 30 GB. No compatible Sparse4D PTM was present under the inspected S3 data/checkpoint paths; smoke training ran from an empty `train.pretrained_model_path`.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations over `train.optim.lr`
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/sparse4d/7b61ed30-4948-4b7e-a1de-882e6b741488/results_dir/train/model_epoch_000_step_00004.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/sparse4d/cd9eb142-c575-4c5e-84a1-dfcbc09edeed/results_dir/train/model_epoch_000_step_00004.pth`
  - resume produced `/tmp/tao-automl-validation/sparse4d/manual_outputs/resume_train/results_dir/train/model_epoch_000_step_00004.pth` with a fresh mtime after restoring the selected checkpoint
- AutoML recommendations:
  - rec 0: job `7b61ed30-4948-4b7e-a1de-882e6b741488`, `train.optim.lr=4.679761717966119e-05`, `val_mAP=0.0`, selected as best
  - rec 1: job `cd9eb142-c575-4c5e-84a1-dfcbc09edeed`, `train.optim.lr=2.699814617932766e-05`, `val_mAP=0.0`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 `model_epoch_000_step_00004.pth`; evaluate logs loaded that exact checkpoint and produced NuScenes mAP metrics.
- Inference checkpoint used: best AutoML rec 0 `model_epoch_000_step_00004.pth`; inference logs loaded that exact checkpoint and wrote NuScenes/NVSchema predictions.
- Export checkpoint used: best AutoML rec 0 `model_epoch_000_step_00004.pth`; export wrote `/tmp/tao-automl-validation/sparse4d/manual_outputs/export/sparse4d.onnx`.
- Quantize checkpoint used: best AutoML rec 0 `model_epoch_000_step_00004.pth`; the action reached the Sparse4D quantize script and failed inside toolkit model loading.
- Resume/retrain checkpoint used: best AutoML rec 0 `model_epoch_000_step_00004.pth`; resume logs restored all states from that exact checkpoint and completed training.
- Were checkpoint paths selected through the proper resolver: yes, validation used `tao_sdk.checkpoints.get_checkpoint_path` with `epoch=0`, `step=4`, and `allow_latest=False` for resume/evaluate/inference/export/quantize.
- Any incorrect latest-checkpoint behavior found: Sparse4D emits `sparse4d_model_latest.pth` as a latest symlink, but the resolver selected the concrete `model_epoch_000_step_00004.pth`; latest was not used.

## Issues found

- Model skill issues:
  - `skill_info.yaml` did not expose optional train pretrained/resume inputs.
  - Checkpoint-consuming actions did not expose checkpoint/model inputs.
  - `spec_params` was empty, so fresh installs had no model-specific resolver wiring for outputs, checkpoints, resume, or export ONNX creation.
  - Sparse4D instructions implied a PTM was mandatory even though no compatible PTM was available in the validation dataset location and smoke training can run from scratch.
- Config issues:
  - Local smoke runs require small temporal settings such as `dataset.num_frames=4`, `dataset.sequences.split_num=4`, and `dataset.batch_size=1`.
  - Export needs an explicit ONNX path and was validated with the trained model input shape, `1408x512`.
- Dataset issues:
  - The selected S3 dataset has only a train split conversion output; no converted val/test split exists in S3.
  - Conversion expanded the MP4 streams to full RGB frame folders, which is expensive on a fresh local install.
  - The no-depth-map dataset variant is incompatible with the default H5 depth-supervised spec.
- Checkpoint issues:
  - Sparse4D writes both concrete epoch/step checkpoints and a latest symlink; downstream actions must use the resolver-selected concrete path.
- Docker/local execution issues:
  - Checkpoint-backed TorchAO quantize fails in `nvidia_tao_pytorch/cv/sparse4d/scripts/quantize.py` with `Sparse4DPlModel.__init__() missing 1 required positional argument: 'experiment_spec'`.
  - ONNX quantize reaches the `modelopt.onnx` backend but fails because `modelopt.onnx.quantization` is not installed in the PyT image.
- Fresh-install issues:
  - Without the metadata fixes, fresh model-skill installs would not reliably pass parent train checkpoints to evaluate, inference, export, quantize, or resume.

## Fixes made

- Added optional train pretrained/resume checkpoint inputs to `models/sparse4d/references/skill_info.yaml`.
- Added checkpoint/model inputs for evaluate, inference, export, and quantize.
- Added Sparse4D `spec_params` mappings for output directories, checkpoint handoff, resume checkpoint resolution, PTM fallback, and export ONNX creation.
- Updated Sparse4D instructions for PTM handling, local-docker conversion behavior, and current quantize blockers.
- Updated the per-network action inventory.

## Remaining issues

- Quantize remains unresolved in the current container image. The model-skill now passes the correct checkpoint/ONNX paths, but the PyT image needs a Sparse4D quantize entrypoint fix for checkpoint-backed PyTorch quantize and a `modelopt.onnx.quantization` dependency for ONNX quantize.
- No deploy, prune, or standalone retrain action is packaged for Sparse4D.
- The available S3 dataset supports smoke validation but does not provide separate converted val/test splits.

## Files changed

- `models/sparse4d/SKILL.md`
- `models/sparse4d/references/skill_info.yaml`
- `docs/model-validation/sparse4d.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Partially validated: dataset conversion, AutoML default training, evaluate, inference, export, and resume pass through the real Sparse4D model skill. Quantize is blocked by current PyT image defects after correct model-skill checkpoint handoff.
