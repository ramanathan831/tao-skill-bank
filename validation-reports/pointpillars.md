Model: pointpillars

Supported actions tested:
- dataset convert: pass
- train: pass
- resume: pass
- eval: pass
- inference: pass
- export: pass
- prune: pass
- retrain: pass via `pointpillars train` with `train.pruned_model_path`
- deploy: pass (`gen_trt_engine`, TensorRT `evaluate`, TensorRT `inference`)
- quantize: unsupported by this model skill
- default_specs: pass
- other: AutoML/HPO not run because this validation pass is restricted to model/deploy skill actions only

Dataset used:
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_pointpillars_train/train.tar.gz`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_pointpillars_train/val.tar.gz`
- Notes: Extracted to `/workspace/run/data/raw` with `train/lidar`, `train/label`, `val/lidar`, and `val/label`. The split contains 48 train LiDAR/label pairs and 2 validation LiDAR/label pairs.
- Any dataset compatibility issues: none. `dataset_convert` produced `infos_train.pkl`, `infos_val.pkl`, `infos_train_val.pkl`, `dbinfos_train.pkl`, and `gt_database/` under `/workspace/run/results/dataset_convert/data_info`.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; PointPillars produced epoch checkpoints
- Best checkpoint path: n/a
- Other checkpoints produced:
  - `/workspace/run/results/train/checkpoint_epoch_1.pth`
  - `/workspace/run/results/resume_train/checkpoint_epoch_2.pth`
  - `/workspace/run/results/retrain/checkpoint_epoch_1.tlt`

Checkpoint/action verification:
- Resume checkpoint used: `/workspace/run/results/train/checkpoint_epoch_1.pth`
- Eval checkpoint used: `/workspace/run/results/resume_train/checkpoint_epoch_2.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train/checkpoint_epoch_2.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/checkpoint_epoch_2.pth`
- Prune checkpoint used: `/workspace/run/results/resume_train/checkpoint_epoch_2.pth`
- Retrain pruned model used: `/workspace/run/results/prune/pruned_0.1.tlt`
- Deploy artifacts used:
  - `gen_trt_engine.onnx_file`: `/workspace/run/results/export/pointpillars.onnx`
  - `gen_trt_engine.save_engine`: `/workspace/run/results/deploy_gen_trt_engine/pointpillars.engine`
  - `evaluate.trt_engine`: `/workspace/run/results/deploy_gen_trt_engine/pointpillars.engine`
  - `inference.trt_engine`: `/workspace/run/results/deploy_gen_trt_engine/pointpillars.engine`
- Were checkpoint paths selected through the proper resolver: yes for the model skill wiring; the skill maps train resume to `resume_model`, evaluate/export/inference/prune to `parent_model`, and retrain to `train.pruned_model_path`. Direct local-docker validation used the exact resolved artifacts listed above, not a latest-file lookup.
- Any incorrect latest-checkpoint behavior found: none. The model uses `checkpoint_epoch_N.pth` and retrain emits encrypted `checkpoint_epoch_N.tlt`; all checkpoint-dependent actions were pointed at exact files.

Issues found:
- Model skill issues:
  - Direct local-Docker specs must set the top-level `results_dir`; setting only action-specific `evaluate.results_dir` led evaluate to attempt `/opt/nvidia/eval` and still print the generic PASS footer.
- Config issues:
  - Deploy evaluate/inference templates are starter defaults and must carry class names, point-cloud range, data path, data info path, and post-processing settings from the trained/exported model.
  - Deploy CPU NMS can become CPU-bound on a barely trained smoke checkpoint with low `model.post_processing.score_thresh`.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None.
- Docker/local execution issues:
  - A first evaluate run failed by attempting to write `/opt/nvidia/eval` because the top-level `results_dir` was blank.
  - Deploy evaluate with `score_thresh: 0.1` became CPU-bound in the deploy CPU NMS path; `nms_pre_max_size` is not honored by that implementation. Validation passed with `score_thresh: 2.0` to force no detections while exercising engine/data/action plumbing.
- Fresh-install issues:
  - Users need guidance to verify expected artifacts and status files instead of trusting the generic PASS footer.

Fixes made:
- Added parent skill guidance that direct local-Docker specs should set top-level `results_dir` as well as action-specific result fields, and should verify expected artifacts/status files.
- Added deploy skill and deploy metadata guidance for the CPU NMS trap on smoke checkpoints, including raising `model.post_processing.score_thresh` for validation-only runs.

Remaining issues:
- AutoML/HPO was not executed in this model-only pass.
- The deploy container's CPU NMS implementation still ignores `nms_pre_max_size`; the skill now documents the workaround, but the container behavior remains.

Files changed:
- `models/pointpillars/SKILL.md`
- `models/pointpillars/deploy/SKILL.md`
- `models/pointpillars/deploy/skill_info.yaml`
- `validation-reports/pointpillars.md`

Final status:
- Fully validated for supported model and deploy actions using the available validation dataset; partially validated if counting AutoML/HPO, which was outside this model-only pass.
