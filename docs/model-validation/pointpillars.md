# Model: pointpillars

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. The run used `AutoMLRunner` with `DockerSDK` on local Docker, `image=default`, `algorithm=bayesian`, `automl_max_recommendations=2`, `num_gpus=1`, and the PointPillars model skill as `skill_dir`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- dataset convert: pass
- train: pass, AutoML default route with two Bayesian recommendations
- eval/evaluate: pass
- inference: pass
- export: pass
- deploy gen_trt_engine: pass after removing pre-created engine output metadata
- deploy evaluate on tao-deployed gen_trt_engine model: pass
- deploy inference on tao-deployed gen_trt_engine model: pass
- prune: pass after adding a non-empty prune key and verifying nonzero `pruned_0.1.tlt`
- retrain: pass through `pointpillars train -e` with `train.pruned_model_path`
- resume training: pass from the selected epoch-1 checkpoint and produced `checkpoint_epoch_2.pth`
- quantize: unsupported by the packaged PointPillars PyT CLI/model skill
- other: parent PyT `gen_trt_engine` and standalone `retrain` are unsupported by the real PyT CLI; deploy sub-skill owns TensorRT actions

## Dataset used

- Source: `s3://nvcf-storage-handling/data/purpose_built_models_pointpillars_train.tar.gz`
- Notes: archive contains `train/label`, `train/lidar`, `val/label`, and `val/lidar`. The dataset uses the packaged seven-class PointPillars default (`Car`, `Truck`, `Van`, `Tram`, `Pedestrian`, `Cyclist`, `Misc`), so this model intentionally did not use the common `num_classes=6` override.
- Any dataset compatibility issues: no issue. `dataset_convert` produced `dbinfos_train.pkl`, `infos_train.pkl`, `infos_val.pkl`, `infos_train_val.pkl`, and `gt_database/`.

## Training result

- Training completed: yes, through AutoML default route
- AutoML search: Bayesian, two recommendations
- Best checkpoint produced: yes
- Best checkpoint path: `/tmp/tao-automl-validation/pointpillars/results/92695587-673a-4f30-8481-79a1cc34ec83/results_dir/checkpoint_epoch_1.pth`
- Other checkpoints produced:
  - `/tmp/tao-automl-validation/pointpillars/results/e6ddc14d-24a0-4f0c-8be7-9e52fff5284a/results_dir/checkpoint_epoch_1.pth`
  - resume produced `/tmp/tao-automl-validation/pointpillars/1578c4df-6375-4e2d-b237-3ec185b4a45a/results_dir/checkpoint_epoch_2.pth`
  - retrain produced `/tmp/tao-automl-validation/pointpillars/096bed3e-8ea9-4736-8c94-4f707c0b9182/results_dir/checkpoint_epoch_1.tlt`
- AutoML recommendations:
  - rec 0: job `92695587-673a-4f30-8481-79a1cc34ec83`, `train.lr=0.0019874878957165617`, `train.weight_decay=0.011398569137582483`, loss `1.5250883102416992`, selected as best
  - rec 1: job `e6ddc14d-24a0-4f0c-8be7-9e52fff5284a`, `train.lr=0.0036`, `train.weight_decay=0.018881271836457512`, loss `1.7649275064468384`

## Checkpoint/action verification

- Eval checkpoint used: best AutoML rec 0 `checkpoint_epoch_1.pth`; evaluation passed with `bev mAP: 0.0000` and `3d mAP: 0.0000` on the two-sample validation split.
- Inference checkpoint used: best AutoML rec 0 `checkpoint_epoch_1.pth`; inference wrote predictions under the job `results_dir/infer`.
- Export checkpoint used: best AutoML rec 0 `checkpoint_epoch_1.pth`; export wrote `/tmp/tao-automl-validation/pointpillars/manual_outputs/pointpillars.onnx`.
- Prune checkpoint used: best AutoML rec 0 `checkpoint_epoch_1.pth`; prune wrote nonzero `/tmp/tao-automl-validation/pointpillars/72646583-404b-415b-9f59-401bcbd0a81d/results_dir/pruned_0.1.tlt`.
- Resume/retrain checkpoint used: resume used best AutoML rec 0 `checkpoint_epoch_1.pth` and produced epoch 2; retrain used the exact pruned model above and logged `Pruned model loaded from`.
- Deploy artifact handoff used: exported ONNX to deploy `gen_trt_engine`, generated `/tmp/tao-automl-validation/pointpillars/manual_outputs/pointpillars.engine`, then deploy evaluate/inference used that exact engine.
- Were checkpoint paths selected through the proper resolver: yes, validation used the checkpoint resolver and selected `checkpoint_epoch_1.pth` from the best AutoML child job.
- Any incorrect latest-checkpoint behavior found: yes in metadata before fixes. Parent checkpoint-consuming actions had no inputs or mappings, which would leave callers to fragile latest-file guessing. The metadata now declares action-specific checkpoint inputs and resolver mappings.

## Issues found

- Model skill issues:
  - Parent metadata advertised PyT `gen_trt_engine`, but `pointpillars --help` in the PyT image does not expose that subtask.
  - Parent metadata declared `pointpillars retrain -e`, but the PyT CLI does not expose a standalone `retrain` subtask.
  - Parent metadata lacked checkpoint inputs and resolver mappings for train resume, evaluate, inference, export, prune, and retrain.
- Config issues:
  - Prune and retrain templates omitted `key`. PointPillars prune encrypts `.tlt` output and needs a non-empty key.
  - Local Docker direct jobs need the converted `data_info` path as mounted in the current job. A path valid in the AutoML runner mount can be invalid in a direct job mount.
- Dataset issues:
  - None after running `dataset_convert` on the S3 archive.
- Checkpoint issues:
  - PointPillars emits `checkpoint_epoch_*.pth`; selecting `model.pth` or a blind latest fallback is incorrect.
  - A prune run without `key` exited with Docker success but produced a zero-byte `pruned_0.1.tlt`; artifact validation is required.
- Docker/local execution issues:
  - Some toolkit failures can still end with an entrypoint `Execution status: PASS` and Docker exit code 0. The status file and expected artifacts must be checked.
- Fresh-install issues:
  - A fresh install would fail or miswire parent checkpoint-consuming actions, deploy engine generation, prune, and retrain before these model-skill metadata/template fixes.

## Fixes made

- Added train resume, prune, evaluate, export, inference, and retrain inputs plus `spec_params` resolver mappings in `models/pointpillars/references/skill_info.yaml`.
- Removed unsupported parent PyT `gen_trt_engine` from the parent model metadata.
- Routed retrain through `pointpillars train -e` with `train.pruned_model_path`.
- Added `key: tlt_encode` to the prune and retrain spec templates.
- Removed pre-created deploy engine file input/output metadata from `models/pointpillars/deploy/skill_info.yaml`.
- Changed deploy `dataset.data_info_path` inputs from file to folder.
- Updated parent and deploy skill instructions for AutoML default training, supported actions, checkpoint resolution, data-info mounts, deploy-only TensorRT, and prune/retrain artifact validation.

## Remaining issues

- None for packaged PointPillars parent and deploy model-skill actions. Quantize is not advertised by the packaged model skill.

## Files changed

- `models/pointpillars/SKILL.md`
- `models/pointpillars/references/skill_info.yaml`
- `models/pointpillars/references/spec_template_prune.yaml`
- `models/pointpillars/references/spec_template_retrain.yaml`
- `models/pointpillars/deploy/SKILL.md`
- `models/pointpillars/deploy/skill_info.yaml`
- `docs/model-validation/pointpillars.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Fully validated: all packaged PointPillars parent and deploy skill actions pass after model-skill metadata/template fixes. Unsupported parent PyT `gen_trt_engine`, standalone `retrain`, and quantize are documented as unsupported by the real packaged CLI/model skill.
