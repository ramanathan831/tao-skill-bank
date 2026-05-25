Model: mae

Supported actions tested:
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- quantize: unsupported/not advertised by the MAE model skill
- prune: unsupported/not advertised by the MAE model skill
- retrain/resume: pass through train.resume_training_checkpoint_path
- dataset convert: unsupported/not advertised by the MAE model skill
- parent gen_trt_engine: fail; not supported by the PyTorch CLI and removed from parent model metadata

Dataset used:
- Source: s3://nvcf-storage-handling/data/classification_train/images_train.tar.gz
- Source: s3://nvcf-storage-handling/data/classification_val/images_val.tar.gz
- Source: s3://nvcf-storage-handling/data/classification_test/images_test.tar.gz
- Notes: The archives extract to ImageFolder-style class directories with 20 classes, so this model required model.num_classes=20 instead of the common num_classes=6.
- Any dataset compatibility issues: None for finetune, evaluate, inference, export, or deploy. MAE inference is not supported for train.stage=pretrain, so validation used train.stage=finetune.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best checkpoint artifact; the one-epoch run produced an epoch/step checkpoint.
- Best checkpoint path: /tmp/tao-model-validation/mae/results/train/train/model_epoch_000_step_00099.pth
- Other checkpoints produced: /tmp/tao-model-validation/mae/results/resume/train/model_epoch_001_step_00198.pth; convnextv2_atto_latest.pth symlinks were produced but not used for checkpoint-dependent actions.

Checkpoint/action verification:
- Eval checkpoint used: /tmp/tao-model-validation/mae/results/train/train/model_epoch_000_step_00099.pth
- Inference checkpoint used: /tmp/tao-model-validation/mae/results/train/train/model_epoch_000_step_00099.pth
- Export checkpoint used: /tmp/tao-model-validation/mae/results/train/train/model_epoch_000_step_00099.pth
- Resume/retrain checkpoint used: /tmp/tao-model-validation/mae/results/train/train/model_epoch_000_step_00099.pth
- Deploy checkpoint path: deploy used /tmp/tao-model-validation/mae/results/export/model.onnx exported from the exact checkpoint above.
- Were checkpoint paths selected through the proper resolver: yes in model metadata after the fix; direct local-docker validation used explicit exact paths to verify the resolver contract.
- Any incorrect latest-checkpoint behavior found: no runtime action selected convnextv2_atto_latest.pth; the parent metadata was missing resolver mappings before the fix.

Issues found:
- Model skill issues:
  - Parent skill advertised gen_trt_engine even though the PyTorch mae CLI only supports evaluate, export, inference, train, and default_specs.
  - Export inputs/outputs were empty in skill_info.yaml, preventing reliable checkpoint and ONNX handoff.
  - spec_params were empty, so parent checkpoint and ONNX output resolver mappings were not declared.
- Config issues:
  - Deploy template included dataset.segment, which is not valid for the MAE deploy schema.
  - Export and deploy specs must use the MAE image size from training/export; the generic 960x544 image-size default is not valid for this finetune run.
- Dataset issues:
  - The selected classification dataset has 20 classes, requiring a model-specific num_classes override.
- Checkpoint issues:
  - No incorrect latest behavior found.
- Docker/local execution issues:
  - Finetune stage emits deprecation warnings but still runs successfully.
- Fresh-install issues:
  - None beyond normal Docker/GPU requirements.

Fixes made:
- Removed parent gen_trt_engine from MAE model manifests; TensorRT engine generation now routes through the deploy subskill.
- Added export inputs/outputs to skill_info.yaml.
- Added evaluate, export, and inference checkpoint resolver mappings to skill_info.yaml.
- Removed invalid dataset.segment from the MAE deploy gen_trt_engine template.
- Documented that inference requires a finetune checkpoint and that exact epoch/step checkpoints should be selected instead of latest symlinks.

Remaining issues:
- None for the supported parent actions after the metadata fixes. The finetune path is deprecated by the container but still functional.

Files changed:
- models/mae/SKILL.md
- models/mae/references/skill_info.yaml
- models/mae/references/spec_template_deploy_gen_trt_engine.yaml
- models/mae/schemas/manifest.json
- models/schemas.manifest.json
- docs/model-validation/mae.md

Final status:
- Fully validated for supported parent actions and deploy gen_trt_engine on local-docker with image=default.
