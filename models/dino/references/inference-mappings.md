# DINO Inference Mappings

Model-specific inference mappings for TAO SDK orchestration. Read this when
generating SDK-backed DINO jobs that infer checkpoint, output, or encryption
fields from parent jobs.

## Contents

- Spec parameter and parent-model inference table copied from
  `models/dino/SKILL.md`.
- Parent job handling rule for generated runners.

## Spec Param / Parent Model Inference

Model-specific inference mappings belong here, not in `config.json`. Generated
runners should read this section and apply the mappings with SDK helpers before
`create_job()`. This mirrors the old microservices `infer_params.py` flow.

Inference mappings from TAO Core `dino.config.json`:

| Action | Spec Field | Inference Function | Meaning |
|---|---|---|---|
| distill | `distill.pretrained_teacher_model_path` | `parent_model` | model file inferred from the parent job results folder |
| distill | `encryption_key` | `key` | encryption key |
| distill | `results_dir` | `output_dir` | current job results directory |
| evaluate | `encryption_key` | `key` | encryption key |
| evaluate | `evaluate.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `evaluate.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| evaluate | `results_dir` | `output_dir` | current job results directory |
| export | `encryption_key` | `key` | encryption key |
| export | `export.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| export | `export.onnx_file` | `create_onnx_file` | output ONNX path |
| export | `results_dir` | `output_dir` | current job results directory |
| gen_trt_engine | `encryption_key` | `key` | encryption key |
| gen_trt_engine | `gen_trt_engine.onnx_file` | `parent_model` | model file inferred from the parent job results folder |
| gen_trt_engine | `gen_trt_engine.tensorrt.calibration.cal_cache_file` | `create_cal_cache` | calibration cache path |
| gen_trt_engine | `gen_trt_engine.trt_engine` | `create_engine_file` | output TensorRT engine path |
| gen_trt_engine | `results_dir` | `output_dir` | current job results directory |
| inference | `encryption_key` | `key` | encryption key |
| inference | `inference.checkpoint` | `parent_model` | model file inferred from the parent job results folder |
| inference | `inference.trt_engine` | `parent_model` | model file inferred from the parent job results folder |
| inference | `results_dir` | `output_dir` | current job results directory |
| quantize | `encryption_key` | `key` | encryption key |
| quantize | `quantize.model_path` | `parent_model` | model file inferred from the parent job results folder |
| quantize | `results_dir` | `output_dir` | current job results directory |
| train | `encryption_key` | `key` | encryption key |
| train | `model.pretrained_backbone_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `results_dir` | `output_dir` | current job results directory |
| train | `train.pretrained_model_path` | `ptm_if_no_resume_model` | PTM when no resume checkpoint exists |
| train | `train.resume_training_checkpoint_path` | `resume_model` | model file inferred from the current job results folder |

For `parent_model` or `parent_model_folder`, pass the upstream
train/export/AutoML child job id as `parent_job_id`. The SDK lists the parent
result folder, filters checkpoint artifacts, and returns the selected model file
or folder. Do not add these mappings back to `config.json` and do not patch
generated runner scripts to guess checkpoint paths.
