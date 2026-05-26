Model: visual-changenet

Supported actions tested:
- train: pass for classify and segment
- eval: pass for classify and segment
- inference: pass for classify and segment
- export: pass for classify and segment
- deploy: pass for classify and segment TensorRT `gen_trt_engine`, TensorRT evaluate, and TensorRT inference
- prune: unsupported, no parent or deploy action is declared
- quantize: pass for classify and segment
- retrain/resume: resume pass for classify and segment via `train.resume_training_checkpoint_path`; retrain as a distinct action is unsupported
- dataset convert: unsupported, no dataset_convert action is exposed
- AutoML/HPO: not run; the model is AutoML-enabled, but this validation was constrained to model-skill actions only

Dataset used:
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_classify_train/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_classify_val/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_classify_test/`
- Source: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_segment_train/`
- Notes: Classify used the S3 `dataset.csv` plus `images.tar.gz` splits. Segment used the S3 `A.tar.gz`, `B.tar.gz`, `list.tar.gz`, and `label.tar.gz` structure. The C-RADIO classify backbone was staged from `nvidia/C-RADIOv2-B/model.safetensors` to a local file and mounted into the container.
- Any dataset compatibility issues: The classify tarballs extract under an `images/` subdirectory, so `images_dir` must point to the extracted `.../images/images` folder. The classify CSV has four columns: `input_path,golden_path,label,object_name`.

Training result:
- Training completed: yes for classify and segment
- Best checkpoint produced: no separate best checkpoint file; TAO emitted epoch/step checkpoints
- Best checkpoint path: not applicable
- Classify checkpoint from initial train: `/workspace/run/results/classify/train/model_epoch_000_step_00012.pth`
- Classify checkpoint from resume: `/workspace/run/results/classify_resume/train/model_epoch_001_step_00024.pth`
- Segment checkpoint from initial train: `/workspace/run/results/segment/train/model_epoch_000_step_00016.pth`
- Segment checkpoint from resume: `/workspace/run/results/segment_resume/train/model_epoch_001_step_00032.pth`
- Other artifacts: classify ONNX `/workspace/run/results/classify/export/changenet-classify.onnx`, classify engine `/workspace/run/results/deploy_classify/gen_trt_engine/changenet-classify.engine`, segment ONNX `/workspace/run/results/segment/export/changenet-segment.onnx`, segment engine `/workspace/run/results/deploy_segment/gen_trt_engine/changenet-segment.engine`

Checkpoint/action verification:
- Eval checkpoint used: classify `/workspace/run/results/classify_resume/train/model_epoch_001_step_00024.pth`; segment `/workspace/run/results/segment_resume/train/model_epoch_001_step_00032.pth`
- Inference checkpoint used: classify `/workspace/run/results/classify_resume/train/model_epoch_001_step_00024.pth`; segment `/workspace/run/results/segment_resume/train/model_epoch_001_step_00032.pth`
- Export checkpoint used: classify `/workspace/run/results/classify_resume/train/model_epoch_001_step_00024.pth`; segment `/workspace/run/results/segment_resume/train/model_epoch_001_step_00032.pth`
- Resume checkpoint used: classify `/workspace/run/results/classify/train/model_epoch_000_step_00012.pth`; segment `/workspace/run/results/segment/train/model_epoch_000_step_00016.pth`
- Quantize checkpoint used: classify `/workspace/run/results/classify_resume/train/model_epoch_001_step_00024.pth`; segment `/workspace/run/results/segment_resume/train/model_epoch_001_step_00032.pth`
- Deploy artifacts used: classify ONNX for `gen_trt_engine`, classify engine for TensorRT evaluate/inference; segment ONNX for `gen_trt_engine`, segment engine for TensorRT evaluate/inference
- Were checkpoint paths selected through the proper resolver: yes in the model metadata after fixes; `evaluate`, `inference`, `export`, and `quantize` map to `parent_model`, and resume maps to `resume_model`. Direct local-docker validation used the exact resolved epoch/step files.
- Any incorrect latest-checkpoint behavior found: yes. The templates/reference text implied `changenet_model_*_latest.pth`, but the validation image produced only `model_epoch_*_step_*.pth`. Downstream specs were corrected to use exact checkpoint files, and the model skill docs now direct users to the resolver.

Issues found:
- Model skill issues:
  - PyTorch `export` and `quantize` were supported by the container and templates but missing from parent `skill_info.yaml` action wiring.
  - The model docs said classify CSVs had three columns; the actual skill/data format requires four columns including `golden_path`.
- Config issues:
  - TAO experiment specs reject workflow-level keys such as `platform`, `image`, `automl_policy`, `monitor`, and `status_interval_minutes`; those must remain launch metadata, not spec YAML fields.
- Dataset issues:
  - Classify images require the extracted `images/images` root.
- Checkpoint issues:
  - No `changenet_model_classify_latest.pth` or `changenet_model_segment_latest.pth` was produced; exact epoch/step checkpoints must be resolved.
- Docker/local execution issues:
  - Deploy classify templates carried stale `model.classify.diff_module`, lacked `task: classify`, and had mismatched dataset/action batch sizes.
  - Deploy segment templates used `img_size: 256` and a broader profile than the 224x224 export used in this smoke run.
- Fresh-install issues:
  - The C-RADIO backbone must be staged as a local file; the container does not dereference HF URLs or repo IDs for `pretrained_backbone_path`.

Fixes made:
- Added parent model-skill wiring for classify `export` and `quantize`.
- Added parent model-skill wiring for `segment_export` and `segment_quantize`.
- Updated checkpoint and CSV guidance in `models/visual-changenet/SKILL.md`.
- Fixed deploy classify templates to use `task: classify`, C-RADIO defaults, valid `difference_module`, single-light defaults, and consistent batch sizes.
- Fixed deploy segment templates to use `task: segment`, 224 image size, and batch/profile values that match static ONNX exports.
- Added this per-model validation report.

Remaining issues:
- `schemas/manifest.json` still lists the generated schema actions only; export/quantize wiring is now in `skill_info.yaml`, but generated action schemas should be regenerated in a future packaging pass.
- AutoML/HPO remains unvalidated because the requested pass excluded non-model-skill execution paths.

Files changed:
- `models/visual-changenet/SKILL.md`
- `models/visual-changenet/references/skill_info.yaml`
- `models/visual-changenet/references/spec_template_deploy_classify_gen_trt_engine.yaml`
- `models/visual-changenet/references/spec_template_deploy_classify_evaluate.yaml`
- `models/visual-changenet/references/spec_template_deploy_classify_inference.yaml`
- `models/visual-changenet/references/spec_template_deploy_segment_gen_trt_engine.yaml`
- `models/visual-changenet/references/spec_template_deploy_segment_evaluate.yaml`
- `models/visual-changenet/references/spec_template_deploy_segment_inference.yaml`
- `validation-reports/visual-changenet.md`

Final status:
- Fully runtime validated across classify and segment PyTorch actions and deploy actions; packaging has a remaining generated-schema manifest follow-up for the newly wired export/quantize actions.
