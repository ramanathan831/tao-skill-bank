Model: grounding-dino

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- deploy inference on tao-deployed gen_trt_engine model: pass
- deploy eval on tao-deployed gen_trt_engine model: pass
- quantize: fail
- resume/retrain: pass through train.resume_training_checkpoint_path
- AutoML default train route: pass with Bayesian automl_max_recommendations=2
- dataset convert: unsupported/not advertised by the Grounding-DINO model skill
- prune: unsupported/not advertised by the Grounding-DINO model skill
- parent gen_trt_engine: fail; not supported by the PyTorch CLI and removed from parent model metadata

Dataset used:
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_odvg/
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_odvg/
- Notes: Train split used ODVG annotations and label map. Validation, evaluation, calibration, and deploy evaluation used COCO annotations from the validation split.
- Any dataset compatibility issues: Quantize calibration must use COCO JSON, not ODVG JSONL. Using ODVG JSONL for calibration fails JSON parsing in the calibration dataloader.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; the one-epoch run produced an epoch/step checkpoint.
- Best checkpoint path: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Other checkpoints produced: /tmp/tao-model-validation/grounding-dino/results/resume/train/model_epoch_001_step_00092.pth; gdino_model_latest.pth symlinks were produced but not used for checkpoint-dependent actions.

AutoML default training rerun:
- Default direct model training used AutoML after the default policy was corrected to automl_policy=on.
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_train_odvg/
- Source: s3://nvcf-storage-handling/data/tao_od_synthetic_subset_val_odvg/
- Algorithm: bayesian
- Recommendations requested: 2
- Metric: mAP50, maximize
- Tuned parameters: train.optim.lr, train.optim.lr_backbone
- Recommendation 0: job a1b093ff-13b0-4efa-9ad8-33b3ab222afc, mAP50 0.0, checkpoint /tmp/tao-automl-validation/grounding-dino/results/a1b093ff-13b0-4efa-9ad8-33b3ab222afc/results_dir/train/model_epoch_000_step_00006.pth
- Recommendation 1: job 8f5bd99a-1d45-4ae4-a34d-fcb4941cd2c6, mAP50 0.0, checkpoint /tmp/tao-automl-validation/grounding-dino/results/8f5bd99a-1d45-4ae4-a34d-fcb4941cd2c6/results_dir/train/model_epoch_000_step_00006.pth
- Best recommendation: rec 0, selected by the AutoML controller summary
- Generated spec verification: both recommendations used exactly one train_data_sources entry with /data/train/images, /data/train/annotations_odvg.jsonl, and /data/train/annotations_odvg_labelmap.json; validation used /data/val/images and /data/val/annotations.json.

Checkpoint/action verification:
- Eval checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Inference checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Export checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Resume/retrain checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth
- Deploy checkpoint path: deploy used /tmp/tao-model-validation/grounding-dino/results/export/model.onnx exported from the exact checkpoint above.
- Quantize checkpoint used: /tmp/tao-model-validation/grounding-dino/results/train/train/model_epoch_000_step_00046.pth, but the action failed in the container before quantized artifact creation.
- Were checkpoint paths selected through the proper resolver: yes in model metadata after the fix; direct local-docker validation used explicit exact paths to verify the resolver contract.
- Any incorrect latest-checkpoint behavior found: no runtime action selected gdino_model_latest.pth; the parent metadata was missing resolver mappings before the fix.

Issues found:
- Model skill issues:
  - Parent skill advertised gen_trt_engine even though the PyTorch grounding_dino CLI only supports evaluate, export, inference, quantize, train, and default_specs.
  - Inference metadata advertised dataset.infer_data_sources.classmap, but Grounding-DINO inference requires dataset.infer_data_sources.captions.
  - Export and quantize inputs were empty in skill_info.yaml, preventing reliable parent artifact handoff.
  - spec_params were empty, so parent checkpoint and ONNX output resolver mappings were not declared.
- Config issues:
  - The packaged parent spec templates had a second blank train_data_sources entry, which caused fresh AutoML train jobs to fail with FileNotFoundError for an empty json_file path.
  - Quantize calibration documentation pointed at ODVG JSONL; the quantize calibration dataloader expects COCO JSON.
  - Deploy specs must carry forward structural settings and export input resolution from the checkpoint/export spec.
- Dataset issues:
  - No dataset gap for train/eval/inference/export/deploy; the S3 ODVG train and COCO val subsets are compatible.
- Checkpoint issues:
  - No incorrect latest behavior found in validated actions.
  - Quantize with a PyTorch checkpoint is blocked by an SDK/container bug where cap_lists is passed as None during checkpoint load.
- Docker/local execution issues:
  - The default PyTorch image requires access to bert-base-uncased for tokenizer/model loading; providing a Hugging Face token avoids rate-limit failures.
  - modelopt.onnx is listed as an available backend, but the default PyTorch image lacks modelopt.onnx.quantization, so ONNX quantize fails after calibration setup.
- Fresh-install issues:
  - Fresh local-docker users need network or cached Hugging Face assets for the default bert-base-uncased text encoder.

Fixes made:
- Removed parent gen_trt_engine from Grounding-DINO model manifests; TensorRT actions now route through the deploy subskill.
- Added export, inference, and quantize inputs/outputs to skill_info.yaml.
- Added parent checkpoint and ONNX output resolver mappings to skill_info.yaml.
- Corrected inference instructions from classmap to captions.
- Documented COCO calibration requirements for quantize.
- Documented exact checkpoint selection and shape-carry-forward requirements.
- Documented the remaining quantize SDK/image blockers.
- Removed the second blank train_data_sources entry from all Grounding-DINO parent spec templates so direct model AutoML training inherits only the user-configured ODVG source.

Remaining issues:
- Quantize remains unresolved in the default rc-226 PyTorch image:
  - PyTorch checkpoint quantize fails because the container script loads Grounding-DINO with cap_lists=None.
  - ONNX quantize reaches calibration when using COCO JSON, then fails because modelopt.onnx.quantization is unavailable in the image.

Files changed:
- models/grounding-dino/SKILL.md
- models/grounding-dino/references/skill_info.yaml
- models/grounding-dino/schemas/manifest.json
- models/grounding-dino/references/spec_template_train.yaml
- models/grounding-dino/references/spec_template_evaluate.yaml
- models/grounding-dino/references/spec_template_export.yaml
- models/grounding-dino/references/spec_template_gen_trt_engine.yaml
- models/grounding-dino/references/spec_template_inference.yaml
- models/grounding-dino/references/spec_template_quantize.yaml
- models/schemas.manifest.json
- docs/model-validation/grounding-dino.md

Final status:
- Partially validated. Train, AutoML default train, eval, inference, export, resume, deploy gen_trt_engine, deploy inference, and deploy evaluation pass end-to-end on local-docker with image=default. Quantize is blocked by SDK/container issues outside the model skill metadata.
