# Model Action Run Inventory

Generated from the per-model validation reports on 2026-05-25. This lists
actions already run in this validation branch. Actions marked as fixed in a
rebuilt image failed in the original default validation image and have source
fixes plus rebuilt runtime images, but still need a full end-to-end rerun with
those rebuilt tags before being counted as pass.

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
- deploy/gen_trt_engine: partial pass in the original default image; image-only engine pass, combined/text ONNX fail due common deploy builder 2D-input bug; fixed in rebuilt deploy image `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`, full rerun pending
- deploy inference on TensorRT engine: partial pass in the original default image; image-only inference pass, full rerun pending after text-engine fix
- deploy evaluate on TensorRT engine: fail in the original default image because full retrieval evaluation requires the text engine; source fixed in rebuilt deploy image, full rerun pending
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
- inference: pass after adding packaged model-skill metadata and wrapper
- export: unsupported
- deploy: unsupported
- prune: unsupported
- retrain/resume: pass with exact Cosmos checkpoint policy folder
- dataset convert: unsupported
- quantize: pass after adding packaged model-skill metadata and wrapper
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
- deploy: pass in rebuilt deploy image `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`; source-fixed rerun generated `ffs_fast_stereo_224.engine`, then deploy inference and deploy evaluate both finished successfully
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
- quantize: pass in rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`; rerun used resolver-selected `model_epoch_000_step_00002.pth` and wrote `quantized_model_torchao.pth`
- retrain/resume: pass
- dataset convert: not packaged as a model skill action

## depth-net-stereo
- train: pass after geometry config fix
- AutoML default train route: pass with Bayesian `automl_max_recommendations=2`
- eval: pass
- inference: pass
- export: pass
- deploy: pass in rebuilt deploy image `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`; source-fixed rerun generated `depth_net_stereo.engine`, then deploy inference and deploy evaluate both finished successfully
- prune: unsupported
- quantize: pass in rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`; rerun used resolver-selected `model_epoch_000_step_00002.pth` and wrote `quantized_model_torchao.pth`
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
- dataset convert: unsupported/not advertised by the DINO model skill; a manual `dino convert` CLI probe failed in SDK schema initialization, but it should not be counted as a supported model-skill action
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
- quantize: fail in the original default image because quantize did not derive required caption/category lists for `GDINOPlModel`; fixed in rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`, full rerun pending
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
- quantize: fail in the original default image due incorrect Lightning constructor argument and missing ModelOpt ONNX extras; fixed in rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`, full rerun pending
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
- quantize: fail in the original default image due missing `dm`/`task` constructor context and missing ModelOpt ONNX extras; fixed in rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`, full rerun pending
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
- quantize: fail in the original default image due missing data-module constructor context and missing ModelOpt ONNX extras; fixed in rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`, full rerun pending
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
- deploy gen_trt_engine: fail in the original default image because the common deploy builder treated 2D `task_tokens` as BCHW; fixed in rebuilt deploy image `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`, full rerun pending
- deploy evaluate: blocked in the original default image because no TensorRT engine was produced; full rerun pending after deploy builder fix
- deploy inference: blocked in the original default image because no TensorRT engine was produced; full rerun pending after deploy builder fix
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
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations
- evaluate: pass after adding downstream `dataset.label_map`
- inference: pass after adding downstream `dataset.label_map`
- export: pass after adding downstream `dataset.label_map`
- resume training: pass through `train.resume_training_checkpoint_path`
- dataset convert: blocked; PyT CLI supports it and model metadata/template were added, but no compatible raw DeepStream BodyPose JSON exists in S3
- deploy: unsupported/not advertised
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- standalone retrain: unsupported by the real PyT CLI; resume uses train

## re-identification
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations using `cmc_rank_1` maximize
- evaluate: pass with resolver-selected `model_epoch_000_step_00099.pth`
- inference: pass with resolver-selected `model_epoch_000_step_00099.pth`
- export: pass with resolver-selected `model_epoch_000_step_00099.pth`
- resume training: pass through `train.resume_training_checkpoint_path`, restored the selected epoch checkpoint and produced `model_epoch_001_step_00198.pth`
- dataset convert: unsupported/not advertised
- deploy: unsupported/not advertised; no deploy sub-skill is packaged
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- standalone retrain: unsupported by the real PyT CLI; resume uses train

## rtdetr
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations using `mAP50` maximize
- evaluate: pass with resolver-selected `model_epoch_000.pth`
- inference: pass with resolver-selected `model_epoch_000.pth`
- export: pass with resolver-selected `model_epoch_000.pth`
- quantize: pass with resolver-selected `model_epoch_000.pth`
- distill: pass after switching the template to the RT-DETR IOU `srcs` binding
- resume training: pass through `train.resume_training_checkpoint_path`, restored the selected epoch checkpoint and produced `model_epoch_001.pth`
- deploy gen_trt_engine: pass after removing invalid shape keys and engine-file pre-creation from deploy metadata/templates
- deploy evaluate on TensorRT engine: pass
- deploy inference on TensorRT engine: pass
- prune: unsupported/not advertised
- dataset convert: unsupported/not advertised
- standalone retrain: unsupported by the real PyT CLI; resume uses train
- parent PyT gen_trt_engine: unsupported by the real PyT CLI; TensorRT is deploy-only

## segformer
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations using `val_miou` maximize
- evaluate: pass with resolver-selected `model_epoch_000_step_00020.pth`
- inference: pass with resolver-selected `model_epoch_000_step_00020.pth`
- export: pass with resolver-selected `model_epoch_000_step_00020.pth`
- quantize: pass with resolver-selected `model_epoch_000_step_00020.pth`
- resume training: pass through `train.resume_training_checkpoint_path`, restored the selected epoch/step checkpoint and produced `model_epoch_001_step_00040.pth`
- deploy gen_trt_engine: pass after removing parent PyT action discovery and deploy engine-file pre-creation from metadata
- deploy evaluate on TensorRT engine: pass
- deploy inference on TensorRT engine: pass
- prune: unsupported/not advertised
- dataset convert: unsupported/not advertised
- standalone retrain: unsupported by the real PyT CLI; resume uses train
- parent PyT gen_trt_engine: unsupported by the real PyT CLI; TensorRT is deploy-only

## sparse4d
- dataset convert: pass through the Data Services `annotations convert` action
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations using `val_mAP` maximize
- evaluate: pass with resolver-selected `model_epoch_000_step_00004.pth`
- inference: pass with resolver-selected `model_epoch_000_step_00004.pth`
- export: pass with resolver-selected `model_epoch_000_step_00004.pth`
- quantize: fail in the original default image after correct checkpoint handoff due incorrect `Sparse4DPlModel` checkpoint loading and missing ModelOpt ONNX extras; fixed in rebuilt PyT image `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`, full rerun pending
- retrain/resume: pass through `train.resume_training_checkpoint_path`, restored the selected epoch/step checkpoint and completed training
- deploy: unsupported/not advertised
- prune: unsupported/not advertised
- standalone retrain: unsupported by the real PyT CLI; resume uses train

## vila
- train: pass, default model invocation routed through AutoML with two Bayesian recommendations using `loss` minimize
- evaluate: pass on `youcook2_val_rtl` after fixing RTL time-token handoff for PEFT/base model loading
- inference: pass with the selected best AutoML LoRA folder and base model
- export: unsupported/not advertised
- deploy: unsupported/not advertised
- prune: unsupported/not advertised
- quantize: unsupported/not advertised
- retrain/resume: unsupported as a separate VILA model skill action
- dataset convert: unsupported/not advertised

## visual-changenet
- train: pass for classify, default model invocation routed through AutoML with two Bayesian recommendations using `val_loss` minimize
- segment_train: pass, default model invocation routed through AutoML with two Bayesian recommendations using `val_loss` minimize
- evaluate: pass for classify with resolver-selected `model_epoch_000_step_00012.pth`
- inference: pass for classify with resolver-selected `model_epoch_000_step_00012.pth`
- segment_evaluate: pass with resolver-selected `model_epoch_000_step_00032.pth`
- segment_inference: pass with resolver-selected `model_epoch_000_step_00032.pth`
- export: unsupported/not advertised by the packaged parent model skill
- deploy gen_trt_engine/evaluate/inference: blocked; deploy sub-skill requires an ONNX parent and no ONNX artifact was available
- prune: unsupported/not advertised
- quantize: unsupported/not advertised by the packaged parent model skill
- retrain/resume: no standalone retrain action; resume resolver selected exact epoch/step checkpoints
- dataset convert: unsupported/not advertised
