# Failing Action Root Cause Follow-Up

Updated on 2026-05-25 after the model-by-model validation pass.

## DINO Dataset Convert

DINO does not advertise a dataset convert action in the packaged model skill
metadata. The failed `dino convert` entry came from a manual CLI capability
probe, not from a supported DINO model-skill action. The probe failed before
spec load because the container's DINO convert schema has string fields with
`None` defaults, but that should not be counted as a DINO supported-action
failure.

## Source Fixes And Rebuilds

- CLIP deploy TensorRT text/combined engines: root cause is the common TAO
  Deploy engine builder assuming every ONNX input is 4D BCHW. Text inputs such
  as `(1, 77)` hit an out-of-bounds shape index. Fixed in `tao-deploy` by
  profiling ONNX inputs by rank and preserving static 2D text inputs. Rebuilt
  deploy image: `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`.
- OneFormer deploy TensorRT engine: same common builder bug. The 2D
  `task_tokens` ONNX input was handled as BCHW. Fixed by the same
  rank-aware profile generation in `tao-deploy`. Rebuilt deploy image:
  `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`.
- DepthNet Fast Stereo and Stereo deploy evaluate: predictions were generated,
  then the deploy stereo evaluator returned one-element NumPy arrays through
  `float(...)`, which fails on non-0D arrays. Fixed in `tao-deploy` by reducing
  accumulated metric arrays to Python scalars and handling empty totals.
  Rebuilt deploy image:
  `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`.
- TAO Deploy common builder import: `builder.py` imported `EngineCalibrator`,
  and `calibrator.py` imported `pycuda.autoinit` at module import time.
  `pycuda.autoinit` segfaulted in both the previous deploy image and the
  rebuilt image on this host. Fixed in `tao-deploy` by deferring CUDA context
  creation until calibration allocates device memory. The rebuilt image now
  imports `EngineBuilder` and `EngineCalibrator` cleanly.
- DepthNet Mono/Stereo quantize: quantize called a non-existent
  `load_state_dict_from_checkpoint` method on the Lightning module. Fixed in
  `tao-pytorch` by loading through the model-specific Lightning checkpoint
  class used by evaluate/export. Rebuilt PyT image:
  `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
- Grounding-DINO quantize: the quantize script passed `cap_lists=None` into
  `GDINOPlModel`, while the model requires dataset captions/category names.
  Fixed in `tao-pytorch` by deriving `cap_lists` from calibration/test data
  and by matching evaluate's public-checkpoint parsing behavior. Rebuilt PyT
  image: `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
- Mask2Former quantize: `Mask2formerPlModule.load_from_checkpoint` was called
  with `experiment_spec=cfg`, but the module constructor expects `cfg`. Fixed
  in `tao-pytorch`. Rebuilt PyT image:
  `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
- OCDNet quantize: quantize used Lightning `load_from_checkpoint` without the
  required `dm` and `task` constructor arguments. Fixed in `tao-pytorch` by
  constructing `OCDDataModule`/`OCDnetModel` and loading the checkpoint through
  the model's existing utility loader. Rebuilt PyT image:
  `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
- OCRNet quantize: quantize used Lightning `load_from_checkpoint` without the
  required data module. Fixed in `tao-pytorch` by constructing
  `OCRDataModule`/`OCRNetModel` and loading checkpoints through the OCRNet
  utility loader. Rebuilt PyT image:
  `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
- Sparse4D quantize: quantize called `Sparse4DPlModel.load_from_checkpoint`
  with `config=cfg`, but the constructor expects `experiment_spec`. Fixed in
  `tao-pytorch` by using the same `load_pretrained_weights` path used by
  evaluate/inference/export. Rebuilt PyT image:
  `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
- ONNX quantize variants for Mask2Former/OCDNet/OCRNet/Sparse4D: the PyT
  Dockerfile installs `nvidia-modelopt[all]` with `--no-deps`, so several
  ModelOpt ONNX extras were absent from the default image. Fixed in
  `tao-pytorch` by adding the missing ONNX extra packages to the quantization
  requirements file and installing them in the release image. The rebuilt PyT
  image imports `modelopt.onnx.quantization` and `onnxruntime` successfully.

## Rebuild Smoke Status

- PyT wheel build: pass.
- PyT image build: pass,
  `nvcr.io/nvstaging/tao/tao-toolkit-pyt:validation-fixes-20260525`.
- PyT smoke tests: `modelopt.onnx.quantization` import pass; changed quantize
  module imports pass for DepthNet, Grounding-DINO, Mask2Former, OCDNet,
  OCRNet, and Sparse4D.
- Deploy wheel build: pass.
- Deploy image build: pass,
  `nvcr.io/nvstaging/tao/tao-toolkit-deploy:validation-fixes-20260525`.
- Deploy smoke tests: `EngineBuilder`/`EngineCalibrator` import pass; DepthNet
  stereo evaluator scalar-metric smoke pass.
- Full deploy builder instantiation remains blocked on this host by TensorRT
  CUDA initialization error 35, which points to the local driver/runtime
  environment rather than the Python source fixes.
- `tao-dataservices` had no failing-action root cause requiring a source
  change in this pass. Sparse4D dataset conversion passed through Data
  Services; the remaining dataset-convert blockers are data/action
  availability issues rather than Data Services code failures, so no Data
  Services image was rebuilt.

## Still Data Or Artifact Blocked

- Optical Inspection dataset convert: no compatible raw Factory PCB/golden CSV
  dataset was found in `s3://nvcf-storage-handling/data/`, and conversion is
  not packaged as a model-skill action.
- Pose Classification dataset convert: the PyT CLI supports conversion and the
  skill metadata/template were added, but no compatible raw DeepStream BodyPose
  JSON dataset was found in S3.
- Visual ChangeNet deploy: the deploy sub-skill requires an ONNX parent
  artifact, but the parent skill does not expose export and no compatible ONNX
  artifact was available from S3 or prior action output.
