Model: oneformer

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass (`gen_trt_engine`, deploy `evaluate`, deploy `inference`)
- prune: unsupported by this model skill
- quantize: pass
- retrain: pass via resume training
- dataset convert: unsupported by this model skill
- other: pass (`default_specs` with `results_dir=...` override)
- AutoML/HPO train path: not run because this validation is constrained to model skills only and cannot invoke the separate AutoML skill/workflow path

Dataset used:
- Source: s3://nvcf-storage-handling/data/segmentation_oneformer_train/
- Source: s3://nvcf-storage-handling/data/segmentation_oneformer_val/
- Notes: Used the real COCO panoptic OneFormer train/validation data. Train split has 17 annotated images plus one extra image/mask in the tarballs; validation has 15 annotated images and masks. Label maps contain 133 categories, and validation used `dataset.contiguous_id: true` with `model.sem_seg_head.num_classes: 133`.
- Any dataset compatibility issues: The S3 tarballs extract nested wrapper folders (`images/images` and `panoptic/images_panoptic`). The first train attempt pointed at the wrapper directory and failed on a missing annotation image; all final actions used the actual folders containing image and panoptic files.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact was produced in this smoke run
- Best checkpoint path: n/a; selected exact resumed epoch checkpoint `/workspace/run/results/resume_train/model_epoch_001_step_00034.pth`
- Other checkpoints produced: initial train produced `/workspace/run/results/train/model_epoch_000_step_00017.pth` and `oneformer_model_latest.pth`; resume produced `/workspace/run/results/resume_train/model_epoch_001_step_00034.pth` and `oneformer_model_latest.pth`
- Training metrics: initial train `train_loss_epoch=147.110`, `mIoU=0.000`, `all_acc=0.000`; resume train `train_loss_epoch=116.143`, `mIoU=0.000`, `all_acc=0.000`

Checkpoint/action verification:
- Eval checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00034.pth`
- Inference checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00034.pth`
- Export checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00034.pth`
- Quantize checkpoint used: `/workspace/run/results/resume_train/model_epoch_001_step_00034.pth`
- Resume/retrain checkpoint used: `/workspace/run/results/train/model_epoch_000_step_00017.pth`
- Deploy engine input used: `/workspace/run/results/export/oneformer.onnx`
- Deploy evaluate/inference engine used: `/workspace/run/results/deploy_gen_trt_engine/oneformer.engine`
- Were checkpoint paths selected through the proper resolver: yes for model-skill parent mapping rules; direct Docker validation used exact resolved epoch artifacts from model outputs.
- Any incorrect latest-checkpoint behavior found: no. `oneformer_model_latest.pth` symlinks were produced but not used for checkpoint-dependent actions.

Issues found:
- Model skill issues:
  - The skill did not document that the provided S3 tarballs extract wrapper directories; local specs must point to the inner folders containing files.
  - The skill did not document that `default_specs` needs `results_dir=...` as a CLI override; using `-e` or no override leaves Hydra `results_dir` missing.
  - Deploy guidance still described the OneFormer `task_tokens` TensorRT failure as current, but the requested `validation-fixes-20260525` deploy image successfully built and ran the engine.
- Config issues:
  - `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` was required for trusted downstream checkpoint load actions.
  - Export used a concrete non-existing ONNX file path under the results tree.
- Dataset issues:
  - Train tarballs include one extra image/mask not referenced by annotations; this did not affect training.
- Checkpoint issues:
  - No best checkpoint was emitted; downstream actions used the exact resumed epoch checkpoint.
- Docker/local execution issues:
  - Deploy actions logged telemetry warnings after successful completion.
  - Deploy evaluation/inference downloaded tokenizer files anonymously because no HF token was passed into the container.
- Fresh-install issues:
  - Bare `oneformer default_specs` fails until `results_dir=...` is supplied.

Fixes made:
- Updated `models/oneformer/SKILL.md` with extracted-tarball inner-folder guidance and the `default_specs results_dir=...` requirement.
- Updated `models/oneformer/deploy/SKILL.md` and `models/oneformer/deploy/skill_info.yaml` to clarify that older 7.0.0 RC deploy images can fail on `task_tokens`, while the requested validation-fixes deploy image builds and runs OneFormer TensorRT.

Remaining issues:
- AutoML/HPO behavior was not exercised because this validation run is limited to model skills.
- Deploy telemetry still emits warnings after successful actions.

Files changed:
- models/oneformer/SKILL.md
- models/oneformer/deploy/SKILL.md
- models/oneformer/deploy/skill_info.yaml
- validation-reports/oneformer.md

Final status:
- Fully validated for OneFormer model-skill train, resume/retrain, evaluate, inference, export, quantize, default_specs, deploy engine generation, deploy evaluation, and deploy inference on local Docker with the validation images.
