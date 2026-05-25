Model: deformable-detr

Validated on 2026-05-25 with `platform=local-docker`, `image=default`.
The parent PyT image resolved to
`nvcr.io/nvstaging/tao/tao-toolkit-pyt:7.0.0-rc-226-multiarch`; the deploy
image resolved to
`nvcr.io/nvstaging/tao/tao-toolkit-deploy:7.0.0-rc-171-multiarch`. The user
requested model-skill validation only, so AutoML/workflow skills were not run
even though the common config requested AutoML.

Supported actions tested:
- train: pass
- eval: pass with exact trained checkpoint
- inference: pass with exact trained checkpoint
- export: pass with exact trained checkpoint
- deploy: pass for deploy `gen_trt_engine`, deploy `evaluate`, and deploy `inference`
- prune: unsupported; not declared in the Deformable-DETR model skill metadata
- quantize: pass with exact trained checkpoint
- retrain/resume: pass with exact trained checkpoint
- dataset convert: unsupported by the packaged model skill metadata
- other: native PyT `gen_trt_engine` metadata check: fail before fix because the PyT CLI does not support that subtask; fixed by keeping TensorRT engine generation in the deploy sub-skill

Dataset used:
- Source: `s3://nvcf-storage-handling/data/object_detection_pyt_train/` and
  `s3://nvcf-storage-handling/data/object_detection_pyt_val/`
- Files: `annotations.json`, `images.tar.gz`, and `label_map.txt`
- Notes: used real S3 COCO data. The local run used a 24-image train subset
  with 143 annotations and a 12-image validation subset with 104 annotations.
  The validation category ids were remapped to the train label map before use.
  Inference used four real validation images; deploy calibration used four real
  validation images.
- Any dataset compatibility issues: train and validation label-map ordering
  differs in S3, so the validation subset must be remapped before mixed
  train/eval use. `dataset.num_classes` must be object classes plus background
  and `dataset.eval_class_ids` must include all foreground category ids.

Training result:
- Training completed: yes
- Best checkpoint produced: no separate best-checkpoint artifact was produced
- Best checkpoint path: not applicable
- Other checkpoints produced:
  `/tmp/tao-model-validation/deformable-detr/results/train/model_epoch_000_step_00024.pth`
  and
  `/tmp/tao-model-validation/deformable-detr/results/resume/model_epoch_001_step_00048.pth`

Checkpoint/action verification:
- Eval checkpoint used:
  `/tao-workspace/results/train/model_epoch_000_step_00024.pth`
- Inference checkpoint used:
  `/tao-workspace/results/train/model_epoch_000_step_00024.pth`
- Export checkpoint used:
  `/tao-workspace/results/train/model_epoch_000_step_00024.pth`
- Quantize checkpoint used:
  `/tao-workspace/results/train/model_epoch_000_step_00024.pth`
- Deploy checkpoint used: deploy consumed the exported ONNX
  `/tao-workspace/results/export/deformable_detr.onnx`, then the generated
  TensorRT engine
  `/tao-workspace/results/deploy_gen_trt_engine/deformable_detr_deploy.engine`
- Resume/retrain checkpoint used:
  `/tao-workspace/results/train/model_epoch_000_step_00024.pth`
- Were checkpoint paths selected through the proper resolver: yes for this
  direct model-skill validation. The exact epoch/step checkpoint was selected
  instead of the `dd_model_latest.pth` symlink for checkpoint-dependent actions.
- Any incorrect latest-checkpoint behavior found: no runtime failure, but the
  skill guidance was fragile because it did not document the emitted checkpoint
  pattern or warn against treating `dd_model_latest.pth` as the default
  dependent-action input.

Issues found:
- Model skill issues:
  - Parent Deformable-DETR metadata advertised `gen_trt_engine`, but the PyT
    container accepts only `convert`, `evaluate`, `export`, `inference`,
    `quantize`, `train`, and `default_specs`.
  - Parent instructions mixed deploy `gen_trt_engine` requirements into the
    PyT parent model action table.
  - Deploy metadata listed `gen_trt_engine.trt_engine` as both an input and an
    output, which made the generated engine path look like a required existing
    file.
- Config issues:
  - Custom datasets need `dataset.num_classes` set to foreground classes plus
    background.
  - The default `dataset.eval_class_ids: [1]` evaluates only class id 1; custom
    multi-class COCO datasets need every foreground id.
  - Architecture-affecting overrides used for a short validation run
    (`model.num_queries`, `model.num_select`, `model.enc_layers`,
    `model.dec_layers`, `model.dim_feedforward`, image dimensions) must be
    carried into evaluate, inference, export, quantize, and deploy.
- Dataset issues:
  - The train and validation S3 label maps use different category ordering.
- Checkpoint issues:
  - The skill did not document the exact emitted checkpoint pattern
    `model_epoch_<epoch>_step_<step>.pth`.
- Docker/local execution issues:
  - Direct local Docker should mount a neutral workspace path such as
    `/tao-workspace` so specs and generated artifacts are easy to hand off
    between PyT and deploy containers.
- Fresh-install issues:
  - None after using the default resolved images and real S3 object-detection
    data.

Fixes made:
- Removed stale parent `gen_trt_engine` action metadata from
  `references/skill_info.yaml`, `schemas/manifest.json`, and the global schema
  manifest.
- Removed `gen_trt_engine.trt_engine` from deploy action inputs and left it as a
  generated output mapped by `create_engine_file`.
- Updated parent instructions to state the actual PyT-supported action set and
  direct TensorRT engine generation through `deploy/SKILL.md`.
- Added Deformable-DETR checkpoint guidance for exact epoch/step checkpoint
  selection and resume handoff.
- Added dataset class-id guidance and architecture-carryover guidance for
  checkpoint-dependent and deploy actions.

Remaining issues:
- Native PyT `gen_trt_engine` remains unsupported by the container and is now no
  longer advertised as a parent model action. TensorRT engine generation is
  validated through the deploy sub-skill.
- Dataset convert is available in the raw CLI but is not packaged as a
  Deformable-DETR model-skill action.
- The validation subsets are intentionally small smoke subsets, so AP metrics
  were 0.0 and are not accuracy claims.

Files changed:
- `models/deformable-detr/SKILL.md`
- `models/deformable-detr/deploy/SKILL.md`
- `models/deformable-detr/deploy/skill_info.yaml`
- `models/deformable-detr/references/skill_info.yaml`
- `models/deformable-detr/schemas/manifest.json`
- `models/schemas.manifest.json`
- `docs/model-validation/deformable-detr.md`

Final status:
- Fully validated for all actions declared by the Deformable-DETR parent model
  skill on `local-docker`, plus the supported Deformable-DETR deploy sub-skill
  actions.
