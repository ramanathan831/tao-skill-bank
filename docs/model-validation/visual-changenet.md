# Model: visual-changenet

Validation date: 2026-05-25

Default model invocation was routed through AutoML, not normal direct training. Both Visual ChangeNet task variants were run through `AutoMLRunner` with `DockerSDK` on local Docker, the model skill as `skill_dir`, Bayesian search, `automl_max_recommendations=2`, and `num_gpus=1`. Secrets were loaded from `~/.tao/secrets.env`; no credentials were written into repo files or reports.

## Supported actions tested

- train: pass for classify, AutoML default route with two Bayesian recommendations over `train.optim.lr`
- segment_train: pass, AutoML default route with two Bayesian recommendations over `train.optim.lr`
- eval/evaluate: pass for classify and segment
- inference: pass for classify and segment
- export: unsupported by the packaged parent model skill
- deploy: blocked; a deploy sub-skill is packaged, but it requires an exported ONNX parent and no compatible ONNX artifact was present in S3 or produced by the parent skill
- prune: unsupported by the packaged model skill
- quantize: unsupported by the packaged parent model skill
- retrain/resume: no standalone retrain action is packaged; resolver verification for resume selected the exact epoch/step checkpoints
- dataset convert: unsupported by the packaged model skill
- other: staged the required C-RADIO backbone from Hugging Face for classify actions

## Dataset used

- Source:
  - Classify train: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_classify_train/`
  - Classify val: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_classify_val/`
  - Classify test/inference: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_classify_test/`
  - Segment train/eval/inference: `s3://nvcf-storage-handling/data/purpose_built_models_visual_changenet_segment_train/`
- Notes: classify folders provide `dataset.csv` and `images.tar.gz`; the run used `dataset.classify.num_input=1` with `SolderLight`. Segment uses `A.tar.gz`, `B.tar.gz`, `label.tar.gz`, and `list.tar.gz`, which the model skill workflow downloaded and extracted into `A/`, `B/`, `label/`, and `list/`.
- Any dataset compatibility issues: no compatible ONNX/export artifact was found in the inspected S3 dataset folders, so deploy could not be exercised.

## Training result

- Training completed: yes, through AutoML default route for both classify and segment
- Best checkpoint produced: yes
- Best checkpoint path:
  - Classify: `/tmp/tao-automl-validation/visual-changenet/classify_results_ranged/cab0569f-fa21-4310-9bde-90eab8481a2e/results_dir/train/model_epoch_000_step_00012.pth`
  - Segment: `/tmp/tao-automl-validation/visual-changenet/segment_results_final/b341668e-a1cb-4753-b4f9-7278881800db/results_dir/train/model_epoch_000_step_00032.pth`
- Other checkpoints produced:
  - Classify rec 0: `/tmp/tao-automl-validation/visual-changenet/classify_results_ranged/99e1bb26-0a11-4dc5-9755-0f8713fe1a49/results_dir/train/model_epoch_000_step_00012.pth`
  - Segment rec 0: `/tmp/tao-automl-validation/visual-changenet/segment_results_final/a8a15403-d6c0-496a-8856-14d809f1e0e5/results_dir/train/model_epoch_000_step_00032.pth`
- AutoML recommendations:
  - Classify rec 0: `train.optim.lr=1.3605733999490419e-05`, `val_loss=0.638`
  - Classify rec 1: `train.optim.lr=5.500000000000001e-06`, `val_loss=0.626`, selected as best
  - Segment rec 0: `train.optim.lr=0.00015139546386747015`, `val_loss=0.193`
  - Segment rec 1: `train.optim.lr=0.00010173822115500998`, `val_loss=0.163`, selected as best

## Checkpoint/action verification

- Eval checkpoint used:
  - Classify: exact best AutoML rec checkpoint `model_epoch_000_step_00012.pth`; evaluate job `7db152f7-6800-49b4-878f-fd4cdafc119a` passed with total accuracy `4.0`, false negative `0.0`, false alarm `96.0`, defect captured `100.0`.
  - Segment: exact best AutoML rec checkpoint `model_epoch_000_step_00032.pth`; evaluate job `46f4fd36-2d8b-49fd-94ca-ee6fb6481402` passed with `acc=0.9697259664535522` and `miou=0.4848629832267761`.
- Inference checkpoint used:
  - Classify: exact best AutoML rec checkpoint `model_epoch_000_step_00012.pth`; inference job `0f66ba54-0771-4618-af6f-849e3faadefc` wrote `inference.csv` with `siamese_score`.
  - Segment: exact best AutoML rec checkpoint `model_epoch_000_step_00032.pth`; inference job `096ef550-cdae-4446-a9c5-4dd17102ee5b` wrote prediction and combined-visualization images.
- Export checkpoint used: unsupported by the packaged parent model skill.
- Resume/retrain checkpoint used: no standalone retrain action is packaged; `tao_sdk.checkpoints.get_checkpoint_path(..., action="resume", allow_latest=False)` selected the same concrete classify and segment epoch/step files.
- Were checkpoint paths selected through the proper resolver: yes for verification. Evaluate and inference jobs were launched with the resolver-selected concrete checkpoint paths; no action used the `changenet_model_*_latest.pth` symlinks.
- Any incorrect latest-checkpoint behavior found: the training jobs produce latest symlinks, but downstream validation used concrete `model_epoch_000_step_*` files. No incorrect latest-checkpoint behavior remained after adding `spec_params` mappings for parent-model resolution.

## Issues found

- Model skill issues:
  - Classify `evaluate` and `inference` templates and schemas defaulted `task` to `segment`, so a fresh user running those actions without an explicit task override would take the wrong task path.
  - Classify `evaluate` and `inference` did not declare `model.backbone.pretrained_backbone_path` as an input even though the runtime loads the staged C-RADIO backbone.
  - `skill_info.yaml` had no `spec_params`, so fresh installs had no model-specific output, parent checkpoint, or resume checkpoint resolver wiring.
  - Documentation implied export/quantize were runnable parent-skill actions even though they are not declared in `references/skill_info.yaml` or the schema manifest.
- Config issues:
  - Minimal AutoML validation used explicit `automl_hyperparameters=['train.optim.lr']` with `valid_min`/`valid_max` ranges to keep the Bayesian search to two practical experiments.
  - Passing the literal string `default` as a Docker image attempts to pull `default:latest`; the final validation used the skill's resolved default image from `container_image: tao_toolkit.pyt`.
- Dataset issues:
  - No compatible ONNX artifact was available under the inspected S3 dataset folders for deploy validation.
- Checkpoint issues:
  - No downstream action used latest symlinks, but resolver mappings were missing before the fix.
- Docker/local execution issues:
  - The classify C-RADIO backbone had to be staged from Hugging Face before local Docker execution because the TAO container does not dereference HF URLs in `model.backbone.pretrained_backbone_path`.
- Fresh-install issues:
  - Without the metadata and template fixes, classify eval/inference could default to the wrong task and parent checkpoint handoff would require manual path guessing.

## Fixes made

- Changed classify `spec_template_evaluate.yaml`, `spec_template_inference.yaml`, and matching schemas to default `task: classify`.
- Added classify eval/infer C-RADIO backbone input declarations.
- Added train and segment-train resume checkpoint inputs.
- Added `spec_params` mappings for output directory, parent checkpoint, and resume checkpoint resolution across classify and segment actions.
- Updated Visual ChangeNet instructions to distinguish declared parent actions from deploy-only or non-packaged actions and to document deploy's ONNX prerequisite.
- Updated the per-network action inventory.

## Remaining issues

- Deploy remains blocked until an exported ONNX model is available. The deploy sub-skill itself declares `gen_trt_engine`, TensorRT evaluate, and TensorRT inference, but the parent skill does not expose export and no compatible ONNX artifact was found in S3.
- Export, quantize, prune, dataset convert, and standalone retrain are not packaged parent model-skill actions.
- The packaged AutoML default parameter lists still contain both classify and segment parameters; validation used an explicit minimal Bayesian search space to avoid cross-task tuning during smoke runs.

## Files changed

- `models/visual-changenet/SKILL.md`
- `models/visual-changenet/references/skill_info.yaml`
- `models/visual-changenet/references/spec_template_evaluate.yaml`
- `models/visual-changenet/references/spec_template_inference.yaml`
- `models/visual-changenet/schemas/evaluate.schema.json`
- `models/visual-changenet/schemas/inference.schema.json`
- `docs/model-validation/visual-changenet.md`
- `docs/model-validation/action-run-inventory.md`

## Final status

Partially validated: all packaged parent Visual ChangeNet actions pass through the real model skill workflow for classify and segment with AutoML default training. Deploy is blocked by the missing ONNX prerequisite.
