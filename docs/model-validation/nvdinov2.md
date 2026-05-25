# Model: nvdinov2

Validated on 2026-05-25 with `platform=local-docker`, `image=default`
(`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch` and
`nvcr.io/nvstaging/tao/tao-toolkit-deploy:7.0.0-rc-171-multiarch`),
`num_gpus=1`, and direct model/deploy skill actions only. Workflow skills were
not run.

## Supported Actions Tested

- train: pass
- eval: unsupported/not advertised
- inference: pass with exact `student_epoch_*` checkpoint
- export: pass with exact `student_epoch_*` checkpoint
- deploy: pass for TAO Deploy `gen_trt_engine`
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain: pass through `train.resume_training_checkpoint_path`
- dataset convert: unsupported/not advertised
- other: stale parent `distill` action was advertised by metadata but rejected by the real PyT CLI; removed from the parent model metadata and manifests. PyT inference on the generated TensorRT engine was probed and failed with the toolkit's `.pth`/`.tlt`-only loader, so it is documented as unsupported.

## Dataset Used

- Source: `s3://nvcf-storage-handling/data/nvdinov2_train_cats_dogs/images_train.tar.gz`
- Source: `s3://nvcf-storage-handling/data/nvdinov2_test_cats_dogs/images_test.tar.gz`
- Source: `s3://nvcf-storage-handling/data/nvdinov2_val_cats_dogs/images_val.tar.gz`
- Notes: used real cats/dogs SSL image folders from S3. The smoke run used a real subset: 64 cats and 64 dogs for train, 16 cats and 16 dogs for test, and 16 cats and 16 dogs for val. The validation config used ViT-S, `img_size=224`, `batch_size=8`, `num_epochs=1`, `num_prototypes=1024`, and `use_custom_attention=false` to keep the local Docker pass tractable.
- Any dataset compatibility issues: none. NvDINOv2 is self-supervised, so the class folder names are only image organization for this validation; the common `num_classes=6` override is not a model field.

## Training Result

- Training completed: yes
- Best checkpoint produced: no separate best-named checkpoint artifact; the one-epoch run produced one concrete epoch/step checkpoint set, which was selected for downstream validation
- Best checkpoint path: `/tmp/tao-model-validation/nvdinov2/results/train/student_epoch_000_step_00015.pth` for inference/export handoff
- Other checkpoints produced: `/tmp/tao-model-validation/nvdinov2/results/train/model_epoch_000_step_00015.pth`, `/tmp/tao-model-validation/nvdinov2/results/train/teacher_epoch_000_step_00015.pth`, and `nvdinov2_model_latest.pth` symlink to the full training checkpoint. Resume validation produced `model_epoch_001_step_00030.pth`, `student_epoch_001_step_00030.pth`, and `teacher_epoch_001_step_00030.pth`.

## Checkpoint/Action Verification

- Eval checkpoint used: not applicable; eval is not exposed by the NvDINOv2 PyT CLI.
- Inference checkpoint used: `/workspace/results/train/student_epoch_000_step_00015.pth`; log confirmed `loading model from /workspace/results/train/student_epoch_000_step_00015.pth`.
- Export checkpoint used: `/workspace/results/train/student_epoch_000_step_00015.pth`; output ONNX was `/workspace/results/export/nvdinov2.onnx`.
- Resume/retrain checkpoint used: `/workspace/results/train/model_epoch_000_step_00015.pth`; log confirmed `Setting resume checkpoint to /workspace/results/train/model_epoch_000_step_00015.pth` and `Restored all states from the checkpoint`.
- Deploy artifact used: `/workspace/results/export/nvdinov2.onnx`; TensorRT engine written to `/workspace/results/deploy_gen_trt_engine/nvdinov2.engine`.
- Were checkpoint paths selected through the proper resolver: yes for the fixed metadata contract on export/inference; direct local-docker validation supplied exact epoch/step paths because workflow/SDK resolver execution was out of scope. Resume used the explicit `train.resume_training_checkpoint_path` action input.
- Any incorrect latest-checkpoint behavior found: yes in the stale documentation/metadata assumptions, not in the corrected runs. The executed inference/export actions used `student_epoch_000_step_00015.pth`; resume used `model_epoch_000_step_00015.pth`; no downstream action used `nvdinov2_model_latest.pth`.

## Issues Found

- Model skill issues:
  - Parent metadata advertised `nvdinov2 distill`, but the real PyT CLI supports only `export`, `inference`, `train`, and `default_specs`.
  - Parent metadata did not declare `export.checkpoint`, `export.onnx_file`, or `inference.checkpoint`, nor the resolver mappings needed for parent train-output handoff.
  - Documentation implied `inference.trt_engine` could be resolved from a parent job, but the packaged PyT inference implementation rejects TensorRT engine paths.
- Config issues:
  - Deploy template used stale `model.backbone.type`; the Deploy schema requires `teacher_type` and `student_type`.
  - Deploy template omitted `results_dir`, causing TAO Deploy to fail before engine build.
  - Deploy template used dynamic opt/max batch sizes above 1; the exported NvDINOv2 ONNX graph failed TensorRT reshape constraints until the profile was fixed at batch 1.
- Dataset issues:
  - None blocking.
- Checkpoint issues:
  - `nvdinov2_model_latest.pth` points to a full training checkpoint and is not suitable for inference/export. The skill now documents `student_epoch_*.pth` for inference/export and `model_epoch_*.pth` for resume/retrain.
- Docker/local execution issues:
  - TAO Deploy returned process exit 0 even for schema/build failures, so logs/status files must be inspected, not just shell exit status.
- Fresh-install issues:
  - Fresh installs would expose unsupported `distill`, omit checkpoint handoff metadata, and ship a deploy template that failed schema validation before these fixes.

## Fixes Made

- Removed unsupported parent `distill` action from `models/nvdinov2/references/skill_info.yaml`, `models/nvdinov2/schemas/manifest.json`, and `models/schemas.manifest.json`.
- Added optional train pretrained/resume checkpoint inputs to the parent skill metadata.
- Added export and inference checkpoint inputs plus `parent_model`/`create_onnx_file` mappings for exact train-output handoff.
- Corrected the deploy template backbone keys, added `results_dir`, and changed the TensorRT profile defaults to fixed batch 1.
- Updated parent and deploy skill docs with exact checkpoint guidance, unsupported TensorRT-engine inference behavior, and the current CLI-supported action set.

## Remaining Issues

- PyT `nvdinov2 inference` does not run on the generated TensorRT engine; it raises `NotImplementedError: Model path format is only supported for .tlt or .pth`. The generated engine is valid for downstream TensorRT consumers, but there is no model-skill inference action for that engine in this toolkit image.
- `automl_policy=on` cannot be honored without routing train through the `tao-automl` workflow skill, which was explicitly prohibited for this validation. Direct model-skill train was validated instead.
- SDK `parent_model` resolver execution was not invoked because workflow/SDK paths were out of scope; the metadata contract is now present and direct runs used exact checkpoint paths.

## Files Changed

- `models/nvdinov2/SKILL.md`
- `models/nvdinov2/deploy/SKILL.md`
- `models/nvdinov2/deploy/skill_info.yaml`
- `models/nvdinov2/references/skill_info.yaml`
- `models/nvdinov2/references/spec_template_deploy_gen_trt_engine.yaml`
- `models/nvdinov2/schemas/manifest.json`
- `models/schemas.manifest.json`
- `docs/model-validation/nvdinov2.md`

## Final Status

Fully validated for the NvDINOv2 model skill's advertised parent and deploy
actions on local Docker after the metadata/template fixes. TensorRT-engine
inference is documented as unsupported by the current PyT inference command.
