# Model Action Run Inventory

Generated from the per-model validation reports on 2026-05-25. This lists
actions already run in this validation branch; models marked "not run yet" have
not yet reached a per-model validation report.

## action-recognition
- train: pass
- eval: pass
- inference: pass
- export: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- deploy: unsupported by this model skill
- prune: unsupported
- quantize: unsupported
- retrain: unsupported as a separate action; resume training pass
- dataset convert: unsupported

## bevfusion
- dataset convert: pass
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass
- inference: pass
- resume train: pass
- export: unsupported by this model skill
- deploy: unsupported by this model skill
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- retrain: unsupported as a separate action by this model skill

## centerpose
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass
- inference: pass
- export: pass
- gen_trt_engine: pass
- deploy evaluate: pass
- deploy inference: pass
- resume train: pass
- prune: unsupported by this model skill
- quantize: unsupported by this model skill
- retrain: unsupported as a separate action by this model skill
- dataset convert: unsupported by this model skill

## classification-pyt
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass
- inference: pass
- export: pass
- gen_trt_engine: pass via TAO Deploy
- deploy evaluate: pass after template batch-size fix
- deploy inference: pass
- quantize: pass
- distill: pass after distill LR-policy fix
- resume train: pass
- prune: unsupported by this model skill
- retrain: unsupported as a separate action by this model skill; resume train pass
- dataset convert: unsupported by this model skill

## clip
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass after trusted-checkpoint PyTorch load override
- inference: pass after trusted-checkpoint PyTorch load override
- export: pass after trusted-checkpoint PyTorch load override
- deploy/gen_trt_engine: partial pass; image-only engine pass, combined/text ONNX fail
- deploy inference on TensorRT engine: partial pass; image-only inference pass
- deploy evaluate on TensorRT engine: fail; full retrieval evaluation requires text engine
- prune: unsupported
- retrain/resume: pass
- quantize: unsupported
- dataset convert: unsupported

## cosmos-embed
- train: pass with container protobuf pin
- eval: pass with exact checkpoint and container protobuf pin
- inference: pass with exact checkpoint and container protobuf pin
- export: pass for ONNX/external-data export and HuggingFace-format export
- deploy: unsupported
- prune: unsupported
- retrain/resume: pass with exact checkpoint and `model.fsdp_shard_size=1`
- quantize: unsupported
- dataset convert: unsupported
- AutoML: unsupported; no model schemas are packaged

## cosmos-rl
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass with exact trained LoRA folder
- inference: unsupported; not declared in metadata
- export: unsupported
- deploy: unsupported
- prune: unsupported
- retrain/resume: pass with exact Cosmos checkpoint policy folder
- dataset convert: unsupported
- quantize: unsupported
- gated model access and local-docker image/mount checks: pass

## deformable-detr
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass with exact trained checkpoint
- inference: pass with exact trained checkpoint
- export: pass with exact trained checkpoint
- deploy gen_trt_engine/evaluate/inference: pass
- prune: unsupported
- quantize: pass with exact trained checkpoint
- retrain/resume: pass with exact trained checkpoint
- dataset convert: unsupported by packaged metadata

## depth-net-fast-stereo
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass with exact trained checkpoint
- inference: pass with exact trained checkpoint
- export: pass with exact trained checkpoint
- deploy: partial; gen_trt_engine and inference pass, evaluate fail after predictions
- prune: unsupported
- quantize: unsupported by packaged metadata
- retrain/resume: pass with exact trained checkpoint
- dataset convert: unsupported by packaged metadata

## depth-net-mono
- train: pass after config fix
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass
- inference: pass
- export: pass
- deploy gen_trt_engine/inference/evaluate: pass
- prune: unsupported
- quantize: fail in TAO SDK code after correct checkpoint handoff
- retrain/resume: pass
- dataset convert: not packaged as a model skill action

## depth-net-stereo
- train: pass after geometry config fix
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass
- inference: pass
- export: pass
- deploy: partial; gen_trt_engine and inference pass, evaluate fail after predictions
- prune: unsupported
- quantize: fail in TAO SDK code after correct checkpoint handoff
- retrain/resume: pass
- dataset convert: not packaged as a model skill action

## dino
- train: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval/evaluate: pass
- inference: pass
- export: pass
- quantize: pass
- distill: pass
- retrain/resume: pass
- deploy gen_trt_engine: pass
- deploy inference on TensorRT engine: pass
- deploy evaluate on TensorRT engine: pass
- dataset convert: fail; SDK/container schema bug before spec load
- prune: unsupported

## grounding-dino
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- deploy inference on TensorRT engine: pass
- deploy eval on TensorRT engine: pass
- quantize: fail
- resume/retrain: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised
- prune: unsupported/not advertised

## mae
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- quantize: unsupported/not advertised
- prune: unsupported/not advertised
- retrain/resume: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised

## mal
- train: pass
- eval: pass
- inference: pass
- export: unsupported/not advertised
- deploy: unsupported/not advertised
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain/resume: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised

## mask-grounding-dino
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- deploy inference on TensorRT engine: pass
- deploy eval on TensorRT engine: pass
- prune: unsupported/not advertised
- quantize: pass
- retrain/resume: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised

## mask2former
- train: pass
- eval: pass
- inference: pass
- export: pass
- deploy: pass
- deploy gen_trt_engine: pass
- deploy inference on TensorRT engine: pass
- deploy eval on TensorRT engine: pass
- prune: unsupported/not advertised
- quantize: fail
- retrain/resume: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised

## ml-recog
- train: pass
- eval: pass after trusted-checkpoint load env
- inference: pass after trusted-checkpoint load env
- export: pass after trusted-checkpoint load env
- deploy gen_trt_engine, TensorRT inference, and TensorRT evaluate: pass
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain: pass through `train.resume_training_checkpoint_path`
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised

## nvdinov2
- train: pass
- eval: unsupported/not advertised
- inference: pass with exact `student_epoch_*` checkpoint
- export: pass with exact `student_epoch_*` checkpoint
- deploy gen_trt_engine: pass
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain: pass through `train.resume_training_checkpoint_path`
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised

## nvpanoptix3d
- train: pass after using toolkit-required `train.precision: fp32`
- eval: pass
- inference: pass after providing a flat folder of real RGB test images
- export: pass for implemented 2D ONNX export
- deploy: unsupported/not advertised
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain: pass through `train.resume_training_checkpoint_path`
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- dataset convert: unsupported/not advertised

## ocdnet
- train: pass
- evaluate: pass
- inference: pass
- export: pass
- deploy/gen_trt_engine: pass
- deploy/inference on TensorRT engine: pass
- deploy/evaluate: pass
- prune: pass
- quantize: fail
- resume training through `train.resume_training_checkpoint_path`: pass
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- retrain: unsupported as a standalone PyT action
- dataset convert: unsupported

## ocrnet
- train: pass through AutoML default route with Bayesian `automl_max_recommendations=2`
- dataset convert: pass
- evaluate: pass
- inference: pass
- export: pass after adding `export.onnx_file` output metadata
- deploy/gen_trt_engine: pass after deploy template refresh
- deploy/inference on TensorRT engine: pass
- deploy/evaluate: pass
- prune: pass after adding `prune.pruned_file` output metadata
- quantize: fail
- retrain: pass after routing the action through `ocrnet train -e` with `model.pruned_graph_path`
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- parent PyT gen_trt_engine: unsupported by the real PyT CLI; TensorRT is deploy-only

## oneformer
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations
- eval: pass
- inference: pass after declaring and using `inference.images_dir`
- export: pass with explicit non-existing `export.onnx_file` at default 640x640 shape
- quantize: pass with best AutoML checkpoint after preserving parent AutoML job path
- retrain/resume: pass from the selected best AutoML checkpoint
- deploy gen_trt_engine: fail, current deploy image builder fails on OneFormer `task_tokens` ONNX input
- deploy evaluate: blocked, no TensorRT engine produced
- deploy inference: blocked, no TensorRT engine produced
- prune: unsupported by the packaged OneFormer PyT CLI
- dataset convert: unsupported by the packaged OneFormer PyT CLI

## optical-inspection
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations
- eval/evaluate: pass
- inference: pass
- export: pass
- deploy gen_trt_engine: pass after removing pre-created engine output metadata
- deploy evaluate on TensorRT engine: pass after fixing deploy evaluate `dataset.infer_dataset.*` wiring and `evaluate.batch_size`
- deploy inference on TensorRT engine: pass
- retrain/resume: pass from the selected best AutoML checkpoint
- prune: unsupported by the packaged Optical Inspection PyT CLI/model skill
- quantize: unsupported by the packaged Optical Inspection PyT CLI/model skill
- dataset convert: blocked/not packaged as a model skill action; no compatible raw Factory PCB/golden CSV dataset in S3
- parent PyT gen_trt_engine: unsupported by the real PyT CLI; TensorRT is deploy-only

## pointpillars
- dataset convert: pass
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations
- evaluate: pass with resolver-selected `checkpoint_epoch_1.pth`
- inference: pass with resolver-selected `checkpoint_epoch_1.pth`
- export: pass with resolver-selected `checkpoint_epoch_1.pth`
- prune: pass after adding a non-empty prune key and verifying nonzero `pruned_0.1.tlt`
- retrain: pass through `pointpillars train -e` with `train.pruned_model_path`
- resume training: pass from the resolver-selected epoch-1 checkpoint and produced `checkpoint_epoch_2.pth`
- deploy gen_trt_engine: pass after removing pre-created engine output metadata
- deploy evaluate on TensorRT engine: pass
- deploy inference on TensorRT engine: pass
- quantize: unsupported/not advertised
- parent PyT gen_trt_engine: unsupported by the real PyT CLI; TensorRT is deploy-only

## pose-classification
- not run yet in this validation branch

## re-identification
- not run yet in this validation branch

## rtdetr
- not run yet in this validation branch

## segformer
- not run yet in this validation branch

## sparse4d
- not run yet in this validation branch

## vila
- not run yet in this validation branch

## visual-changenet
- not run yet in this validation branch
