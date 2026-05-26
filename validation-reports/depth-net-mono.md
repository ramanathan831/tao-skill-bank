Model: depth-net-mono

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- deploy evaluate on tao-deployed gen_trt_engine model: pass
- deploy inference on tao-deployed gen_trt_engine model: pass
- quantize: pass
- dataset convert: pass after model skill template fix
- resume: pass
- prune: not supported by this model skill
- retrain: not a separate action; resume was tested with the train action
- other: AutoML not run because validating it would require the AutoML/workflow skill path, which is excluded by the validation constraints

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_depth_net_train/
- Source: s3://nvcf-storage-handling/data/purpose_built_models_depth_net_val/
- Source: s3://nvcf-storage-handling/data/purpose_built_models_depth_net_test/
- Notes: No mono-specific dataset was present in S3. Used the real S3 depth dataset by deriving mono annotations from the existing stereo rows: left RGB image plus GT depth. The smoke run used 4 train rows and 3 val rows; convert was validated on the staged real train/val/test images.
- Any dataset compatibility issues: The derived annotations are compatible with `RelativeDepthAnything` and `RelativeMonoDataset`. No TAO mono pretrained checkpoint was found in S3 `data/checkpoints/`, the repo, or local searchable paths, so the run used scratch training with `encoder: vits` and 224x224 crops for functional validation only.

Training result:
- Training completed: yes
- Best checkpoint produced: yes; this one-epoch smoke run produced one monitor checkpoint and a `dn_model_latest.pth` symlink
- Best checkpoint path: /workspace/run/results/train/model_epoch_000_step_00004.pth
- Other checkpoints produced: /workspace/run/results/resume_train/model_epoch_001_step_00008.pth from resume

Checkpoint/action verification:
- Eval checkpoint used: /workspace/run/results/train/model_epoch_000_step_00004.pth
- Inference checkpoint used: /workspace/run/results/train/model_epoch_000_step_00004.pth
- Export checkpoint used: /workspace/run/results/train/model_epoch_000_step_00004.pth
- Quantize checkpoint used: /workspace/run/results/train/model_epoch_000_step_00004.pth
- Resume/retrain checkpoint used: /workspace/run/results/train/model_epoch_000_step_00004.pth
- Deploy checkpoint used: none directly; deploy used exported ONNX /workspace/run/results/export/depth_net_mono.onnx produced from the exact train checkpoint, then engine /workspace/run/results/deploy/depth_net_mono.engine
- Were checkpoint paths selected through the proper resolver: yes for the direct local-Docker path; the model skill's exact checkpoint pattern `model_epoch_<epoch>_step_<step>.pth` was used, and logs verified the exact checkpoint was loaded
- Any incorrect latest-checkpoint behavior found: no. `dn_model_latest.pth` existed but was not used for eval, inference, export, quantize, or resume.

Issues found:
- Model skill issues:
  - The documented direct Docker command used `--user $(id -u):$(id -g)` without setting `USER`, `LOGNAME`, `HOME`, or writable cache paths. This is the same fresh-install blocker found in the sibling depth model skills.
  - The documented `convert_spec.yaml` omitted mandatory `results_dir`. The first real `depth_net convert` run failed with `Missing mandatory value: results_dir`; adding `results_dir` made convert pass and write `train.txt` / `val.txt`.
  - The deploy template defaulted `gen_trt_engine.verbose: true`, which is unnecessarily noisy for normal validation.
- Config issues:
  - The common `num_classes=6` setting is not applicable to mono depth estimation.
  - The model metadata is AutoML-enabled, but the requested validation rules prohibit workflow/AutoML skill execution. Direct train was used for model-skill validation.
  - Export was requested with opset 17, but the current exporter retained opset 18 because ONNX version conversion for `Resize` was unavailable. The exported ONNX was verified and deploy accepted it.
- Dataset issues:
  - No dedicated mono dataset or TAO mono pretrained checkpoint was available. Metrics are scratch-training smoke metrics only.
  - PyTorch relative-depth evaluate reported `val/abs_rel: NaN` on this scratch model/data pairing while still completing; train loss was finite and deploy evaluate produced finite `abs_rel` and `d1`.
- Checkpoint issues:
  - None found in the exercised direct-Docker chain.
- Docker/local execution issues:
  - Direct `--user` execution requires writable home/cache env vars as described above.
  - PyT and deploy status JSON frequently leave successful final rows as `RUNNING`; deploy evaluate also records a `SUCCESS` metric row.
- Fresh-install issues:
  - The UID/cache environment issue and missing `results_dir` in the convert template are fresh-install blockers for users following the original docs.

Fixes made:
- Updated `models/depth-net-mono/SKILL.md` to include mandatory `results_dir` in the dataset convert spec template.
- Updated `models/depth-net-mono/SKILL.md` to create writable home/cache directories and set `USER`, `LOGNAME`, `HOME`, `MPLCONFIGDIR`, `TORCHINDUCTOR_CACHE_DIR`, and `XDG_CACHE_HOME` when using `--user`.
- Updated `models/depth-net-mono/references/spec_template_deploy.yaml` to default `gen_trt_engine.verbose` to `false`, with a note to enable it only for detailed TensorRT diagnostics.

Remaining issues:
- No TAO mono pretrained checkpoint was available, so quality metrics are not representative.
- AutoML was not tested because that path routes through workflow/AutoML skills, which were explicitly out of scope.
- Status JSON final rows should ideally mark completed actions as success, but fixing that would require entrypoint/runtime code rather than model skill changes.

Files changed:
- models/depth-net-mono/SKILL.md
- models/depth-net-mono/references/spec_template_deploy.yaml
- validation-reports/depth-net-mono.md

Final status:
- Fully validated for supported direct model-skill and deploy model-skill actions on local-docker with the validation images; metrics are not representative because no mono pretrained checkpoint was available.
