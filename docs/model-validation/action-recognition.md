# Model: action-recognition

Validated on 2026-05-25 with `platform=local-docker`, `image=default`
(`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch`),
`num_gpus=1`. The original action validation used direct model-skill actions
only. AutoML default train routing was rerun afterward with Bayesian search and
two recommendations.

## Supported actions tested

- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: unsupported by this model skill
- prune: unsupported
- quantize: unsupported
- retrain: unsupported as a separate action; resume training was tested through `train.resume_training_checkpoint_path`
- dataset convert: unsupported
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- other: resume train checkpoint-dependent path: pass

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_action_recognition_train/`
- Notes: used `train.tar.gz`, `test.tar.gz`, and `test/smile.tar.gz`; archives were downloaded to scratch storage and extracted before launching TAO because the action-recognition entrypoints require directories.
- Any dataset compatibility issues: the original model skill examples incorrectly pointed spec fields at `.tar.gz` files. The TAO train entrypoint failed with `NotADirectoryError` until the archives were extracted and the spec paths pointed at `train/`, `test/`, and `smile_infer/smile/`.

## Training result

- Training completed: yes
- Best checkpoint produced: yes; the one-epoch validation run produced one concrete checkpoint, so it is the selected best checkpoint for downstream validation.
- Best checkpoint path: `/tmp/tao-model-validation/action-recognition/results/train/model_epoch_000_step_00005.pth`
- Other checkpoints produced: `/tmp/tao-model-validation/action-recognition/results/train/ar_model_latest.pth` symlink to `model_epoch_000_step_00005.pth`; resume validation produced `/tmp/tao-model-validation/action-recognition/results/resume/model_epoch_001_step_00010.pth`.

## AutoML default-path rerun

- Training mode before rerun: normal/direct TAO model training. This happened because the earlier validation explicitly prohibited workflow/AutoML routing.
- Secrets source: `~/.tao/secrets.env`, sourced without printing values. `ACCESS_KEY`, `SECRET_KEY`, `NGC_KEY`, and `HF_TOKEN` were passed through the local run environment.
- AutoML route: `AutoMLRunner` with `skill_dir=/localhome/local-rarunachalam/tao-skills-external/models/action-recognition`, `platform=local-docker`, default PyT image, `algorithm=bayesian`, `metric=val_loss`, `direction=minimize`, and `automl_max_recommendations=2`.
- Search parameters: `model.dropout_ratio` and `train.optim.lr`; the minimal validation ranges were `dropout_ratio=0.25..0.6` and `lr=0.0001..0.001`.
- Result: pass. Two real Docker TAO child jobs completed and emitted `val_loss`.
- Rec 0: job `bc268250-497a-4335-abb2-a5c60eee7e11`, `model.dropout_ratio=0.47866654219899346`, `train.optim.lr=0.0007576244504365072`, `val_loss=0.713`.
- Rec 1: job `7b229726-65fb-43c1-ab35-902354b948e6`, `model.dropout_ratio=0.5157714571599434`, `train.optim.lr=0.00033437700723338773`, `val_loss=0.700`; selected as best.
- AutoML checkpoints produced:
  - `/tmp/tao-automl-validation/action-recognition/results/bc268250-497a-4335-abb2-a5c60eee7e11/results_dir/train/model_epoch_000_step_00005.pth`
  - `/tmp/tao-automl-validation/action-recognition/results/7b229726-65fb-43c1-ab35-902354b948e6/results_dir/train/model_epoch_000_step_00005.pth`
- Generated specs contained dataset paths and hyperparameters only; no credentials were written to the generated specs.

## Checkpoint/action verification

- Eval checkpoint used: `/workspace/results/train/model_epoch_000_step_00005.pth`
- Inference checkpoint used: `/workspace/results/train/model_epoch_000_step_00005.pth`
- Export checkpoint used: `/workspace/results/train/model_epoch_000_step_00005.pth`
- Resume/retrain checkpoint used: `/workspace/results/train/model_epoch_000_step_00005.pth`
- Were checkpoint paths selected through the proper resolver: yes for direct local-docker validation; selected the concrete epoch/step checkpoint emitted by the model action. The SDK `parent_model` resolver was not invoked because workflow skills were explicitly out of scope.
- Any incorrect latest-checkpoint behavior found: yes in the instructions, not in the executed actions. The skill did not previously warn direct local-docker users away from the `ar_model_latest.pth` symlink; the reportable downstream actions were rerun with the concrete checkpoint path.

## Issues found

- Model skill issues:
  - Dataset examples pointed `dataset.train_dataset_dir`, `dataset.val_dataset_dir`, `evaluate.test_dataset_dir`, and `inference.inference_dataset_dir` at archives, but TAO expects extracted directories.
  - Direct local-docker checkpoint handoff lacked guidance for selecting concrete epoch/step checkpoints instead of the latest symlink.
- Config issues:
  - `spec_template_export.yaml` omitted `export.onnx_file` even though the schema and SKILL mapping define it. Without the explicit field, export wrote the ONNX next to the train checkpoint instead of under the export results directory.
- Dataset issues:
  - No incompatible dataset issue after extraction. The sample dataset has two classes (`catch`, `smile`), so the common `num_classes=6` setting was not applicable.
- Checkpoint issues:
  - No runtime checkpoint failure after using the concrete `model_epoch_000_step_00005.pth` file.
- Docker/local execution issues:
  - The platform preflight's older `--runtime=nvidia` check failed on this host, but `docker run --gpus all ... nvidia-smi` and the default TAO image worked with CUDA forward compatibility.
- Fresh-install issues:
  - `tao_sdk` is not installed. This did not block direct model actions, but it means SDK `parent_model` resolver behavior could not be exercised without invoking SDK/workflow paths.
  - AutoML local-Docker execution required installing the Python runtime dependencies missing from the local checkout environment: `docker`, `fsspec`, `s3fs`, `toml`, `pandas`, `pyyaml`, and `huggingface_hub`.

## Fixes made

- Updated `models/action-recognition/SKILL.md` to require extracted dataset directories for train/evaluate/inference.
- Added direct local-docker checkpoint handoff guidance for exact epoch/step checkpoint selection and resume training.
- Added an export override example with `export.onnx_file`.
- Added `export.onnx_file` to `models/action-recognition/references/spec_template_export.yaml`.
- Reran export with `export.onnx_file: /workspace/results/export/action_recognition.onnx`; ONNX output was produced under the export results directory.
- Updated the model-skill train policy wording to make `automl_policy: on` the default and reserve direct model training for explicit `automl_policy: off` or missing packaged AutoML schemas/templates.
- Reran the train path through AutoML with a two-recommendation Bayesian configuration.

## Remaining issues

- SDK `parent_model` checkpoint resolution was not exercised because the SDK is not installed and workflow/SDK paths were out of scope.

## Files changed

- `models/action-recognition/SKILL.md`
- `models/action-recognition/references/spec_template_export.yaml`
- `docs/model-validation/action-recognition.md`

## Final status

Fully validated for the action-recognition model skill's direct supported actions on local Docker, plus the AutoML default train route with two Bayesian recommendations.
