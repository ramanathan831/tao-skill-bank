Model: ocdnet

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass (`gen_trt_engine`, deploy `evaluate`, deploy `inference`)
- prune: pass
- quantize: pass
- retrain: pass (`train` resume from full checkpoint, and `train` from pruned graph)
- dataset convert: unsupported
- default_specs: pass
- AutoML/HPO train path: not run because this validation is constrained to model skills only and cannot invoke the separate AutoML skill/workflow path

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_train/train.tar.gz
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_train/train/img.tar.gz
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_val/test.tar.gz
- Source: s3://nvcf-storage-handling/data/purpose_built_models_ocdnet_val/test/img.tar.gz
- Notes: Used the real ICDAR-style OCDNet dataset from S3. Extracted train has 45 image/label pairs plus 25 image-only calibration files; extracted validation has 25 image/label pairs plus 25 image-only inference files. The action specs used the extracted `img/` and `gt/` folder layout expected by the OCDNet loaders.
- Any dataset compatibility issues: none. Train/evaluate/prune/quantize used the extracted dataset folders, and PyT/deploy inference used the image-only validation folder.

Training result:
- Training completed: yes
- Best checkpoint produced: yes
- Best checkpoint path: `/workspace/run/results/resume_train/model_best.pth`
- Other checkpoints produced: initial train produced `/workspace/run/results/train/model_epoch_000_step_00023.pth`; resume/retrain produced `/workspace/run/results/resume_train/model_epoch_001_step_00046.pth`; pruned retrain produced `/workspace/run/results/pruned_retrain/model_epoch_000_step_00023.pth`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/resume_train/model_best.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train/model_best.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/model_best.pth`
- Prune checkpoint used: `/workspace/run/results/resume_train/model_best.pth`
- Quantize checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00046.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00023.pth`
- Pruned retrain artifact used: `/workspace/run/results/prune/pruned_0.1.pth`
- Deploy engine input used: `/workspace/run/results/export/ocdnet.onnx`
- Deploy evaluate/inference engine used: `/workspace/run/results/deploy_gen_trt_engine/ocdnet.engine`
- Were checkpoint paths selected through the proper resolver: yes for the model-skill parent mapping rules; direct Docker validation used exact resolved artifacts from the model outputs.
- Any incorrect latest-checkpoint behavior found: no. No action used a latest checkpoint alias.

Issues found:
- Model skill issues:
  - The model instructions did not explicitly describe pruned-graph retrain through `ocdnet train` with `model.load_pruned_graph: true` and `model.pruned_graph_path`.
  - The quantize note was stale for the validation PyT image: quantize now succeeds with the full epoch checkpoint.
  - The `default_specs` utility needs an explicit writable `results_dir`.
- Config issues:
  - One-epoch train/retrain validation requires `train.lr_scheduler.args.warmup_epoch: 0`; the default warmup equals the one-epoch smoke budget.
  - Deploy `gen_trt_engine` was validated with `tensorrt.data_type: FP32` from the exported ONNX. INT8 calibration parameter variants were not swept.
- Dataset issues:
  - None.
- Checkpoint issues:
  - None. Best-checkpoint actions used `model_best.pth`; resume and quantize used exact full epoch checkpoints; pruned retrain used the exact prune output.
- Docker/local execution issues:
  - Deploy actions logged telemetry warnings after successful completion.
  - Status files often ended with final `RUNNING` records containing success messages; command logs contained explicit success or PASS lines.
  - The top-level `default_specs --help` path imports CUDA modules; run it with the GPU Docker pattern on a fresh install.
- Fresh-install issues:
  - None after documenting the pruned-retrain, quantize, and default_specs handoffs.

Fixes made:
- Updated `models/ocdnet/SKILL.md` to document pruned-graph retrain through `train`, the required `default_specs` `results_dir`, and the current full-checkpoint quantize behavior for the validation PyT image.

Remaining issues:
- AutoML/HPO behavior was not exercised because this validation run is limited to model skills.
- INT8 deploy engine generation was not separately validated; the deploy action itself passed with a FP32 engine from the exported ONNX.

Files changed:
- models/ocdnet/SKILL.md
- validation-reports/ocdnet.md

Final status:
- Fully validated for OCDNet model-skill train, resume/retrain, pruned retrain, evaluate, inference, export, prune, quantize, default_specs, deploy engine generation, deploy evaluation, and deploy inference on local Docker with the validation images. Dataset convert is unsupported.
