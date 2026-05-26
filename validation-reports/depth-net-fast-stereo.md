Model: depth-net-fast-stereo

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- deploy evaluate on tao-deployed gen_trt_engine model: pass
- deploy inference on tao-deployed gen_trt_engine model: pass
- resume: pass
- prune: not supported by this model skill
- quantize: not supported by this model skill
- retrain: not a separate action; resume was tested with the train action
- dataset convert: not supported by this model skill
- other: AutoML not run because validating it would require the AutoML/workflow skill path, which is excluded by the validation constraints

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_depth_net_train/
- Source: s3://nvcf-storage-handling/data/purpose_built_models_depth_net_val/
- Source: s3://nvcf-storage-handling/data/purpose_built_models_depth_net_test/
- Notes: Used the real Middlebury-style stereo annotations and images from S3. The smoke run staged 4 train annotation rows and 3 val annotation rows, and derived a 2-column inference annotation from the val rows.
- Any dataset compatibility issues: The dataset is compatible with `FastFoundationStereo` using `dataset_name: Middlebury` for train/evaluate and `dataset_name: GenericDataset` for 2-column inference. No bp2 commercial checkpoint was available in S3 `data/checkpoints/`, the repo, or local searchable paths, so training was from scratch and metrics are not representative of the intended bp2 model.

Training result:
- Training completed: yes
- Best checkpoint produced: yes; this one-epoch smoke run produced one monitor checkpoint and a `dn_model_latest.pth` symlink
- Best checkpoint path: /workspace/run/results/train/train/model_epoch_000_step_00004.pth
- Other checkpoints produced: /workspace/run/results/resume_train/train/model_epoch_001_step_00008.pth from resume

Checkpoint/action verification:
- Eval checkpoint used: /workspace/run/results/train/train/model_epoch_000_step_00004.pth
- Inference checkpoint used: /workspace/run/results/train/train/model_epoch_000_step_00004.pth
- Export checkpoint used: /workspace/run/results/train/train/model_epoch_000_step_00004.pth
- Resume/retrain checkpoint used: /workspace/run/results/train/train/model_epoch_000_step_00004.pth
- Deploy checkpoint used: none directly; deploy used exported ONNX /workspace/run/results/export/depth_net_fast_stereo.onnx produced from the exact train checkpoint, then engine /workspace/run/results/deploy/depth_net_fast_stereo.engine
- Were checkpoint paths selected through the proper resolver: yes for the direct local-Docker path; the model skill's exact checkpoint pattern `model_epoch_<epoch>_step_<step>.pth` was used, and logs verified the exact checkpoint was loaded
- Any incorrect latest-checkpoint behavior found: no. `dn_model_latest.pth` existed but was not used for eval, inference, export, or resume.

Issues found:
- Model skill issues:
  - The documented direct Docker command used `--user $(id -u):$(id -g)` without setting `USER`, `LOGNAME`, `HOME`, or writable cache paths. On a fresh host UID that is not present inside the container, the first train attempt failed before action execution with `KeyError: 'getpwuid(): uid not found: 2583'` from PyTorch inductor cache setup.
  - The deploy template defaulted `gen_trt_engine.verbose: true`, which produced a 29 MB TensorRT trace for one 160x320 smoke engine and makes normal validation logs unnecessarily noisy.
- Config issues:
  - The common `num_classes=6` setting is not applicable to stereo depth estimation.
  - The model metadata is AutoML-enabled, but the requested validation rules prohibit workflow/AutoML skill execution. Direct train was used for model-skill validation.
- Dataset issues:
  - No compatible bp2 checkpoint artifact (`model_best_bp2_serialize.pth`) was found in the available S3 data/checkpoint paths or local repo paths. Scratch training validates action wiring only.
  - PyTorch inference with a 2-column annotation has no GT, so metric lines are `nan`; prediction artifacts were produced successfully.
- Checkpoint issues:
  - None found in the exercised direct-Docker chain.
- Docker/local execution issues:
  - Direct `--user` execution requires writable home/cache env vars as described above.
  - Deploy actions exited 0 and wrote output artifacts, but deploy status JSON often leaves the final success message with status `RUNNING`; `trt_evaluate` also records a `SUCCESS` metric row.
- Fresh-install issues:
  - The UID/cache environment issue is a fresh-install blocker for users following the original direct Docker command.

Fixes made:
- Updated `models/depth-net-fast-stereo/SKILL.md` to create writable home/cache directories and set `USER`, `LOGNAME`, `HOME`, `MPLCONFIGDIR`, `TORCHINDUCTOR_CACHE_DIR`, and `XDG_CACHE_HOME` when using `--user`.
- Updated `models/depth-net-fast-stereo/references/spec_template_deploy.yaml` to default `gen_trt_engine.verbose` to `false`, with a note to enable it only for detailed TensorRT diagnostics.

Remaining issues:
- No bp2 commercial checkpoint was available, so quality metrics are scratch-training smoke metrics only.
- AutoML was not tested because that path routes through workflow/AutoML skills, which were explicitly out of scope.
- Deploy status JSON final rows should ideally mark completed actions as success, but the model skill can only document this unless deploy entrypoint code is changed.

Files changed:
- models/depth-net-fast-stereo/SKILL.md
- models/depth-net-fast-stereo/references/spec_template_deploy.yaml
- validation-reports/depth-net-fast-stereo.md

Final status:
- Fully validated for supported direct model-skill and deploy model-skill actions on local-docker with the validation images; metrics are not representative because no bp2 checkpoint was available.
