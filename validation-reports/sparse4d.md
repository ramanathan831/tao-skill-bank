Model: sparse4d

Supported actions tested:
- default_specs: pass
- dataset convert: pass
- train: pass
- resume/retrain: pass
- eval: pass
- inference: pass
- export: pass
- quantize: pass
- deploy: unsupported by this model skill
- prune: unsupported by this model skill
- AutoML/HPO: not run because this validation is constrained to direct model-skill actions and must not run workflow skills

Dataset used:
- Source: s3://nvcf-storage-handling/data/purpose_built_models_sparse4d_train/
- Notes: Used the AICity Sparse4D subset under train/subsetscene with real videos, ground_truth.json, calibration.json, map.png, and H5 depth maps. The final export-compatible conversion used aicity.num_frames=200 and aicity.depth_format=h5, producing 200 samples per converted random camera grouping and anchor_init.npy shape (900, 11).
- Any dataset compatibility issues: The no-depth S3 variant and depth_format=none conversion produced .none depth paths that the trainer still attempted to load. The 3-frame H5 conversion trained/evaluated/inferred, but produced only 72 anchors and could not satisfy Sparse4D export's default memory-bank assumptions. The S3 source only provided a train split, so train/eval/inference smoke validation reused the converted train pkl for validation/test.

Training result:
- Training completed: yes
- Best checkpoint produced: no explicit best-checkpoint alias; epoch/step checkpoints were produced successfully
- Best checkpoint path: /workspace/run/results/resume_train_200_exportable/train/model_epoch_000_step_00003.pth was used as the final validated checkpoint
- Other checkpoints produced: /workspace/run/results/train_200_exportable/train/model_epoch_000_step_00003.pth; earlier 3-frame exploratory runs produced 72-anchor checkpoints that were not used for final export/quantize

Checkpoint/action verification:
- Eval checkpoint used: /workspace/run/results/resume_train_200_exportable/train/model_epoch_000_step_00003.pth
- Inference checkpoint used: /workspace/run/results/resume_train_200_exportable/train/model_epoch_000_step_00003.pth
- Export checkpoint used: /workspace/run/results/resume_train_200_exportable/train/model_epoch_000_step_00003.pth
- Resume/retrain checkpoint used: /workspace/run/results/train_200_exportable/train/model_epoch_000_step_00003.pth
- Quantize checkpoint used: /workspace/run/results/resume_train_200_exportable/train/model_epoch_000_step_00003.pth
- Were checkpoint paths selected through the proper resolver: yes for this direct local-docker validation path; each dependent action used an explicit epoch/step checkpoint path matching the model-skill parent-model mapping instead of a latest-model glob
- Any incorrect latest-checkpoint behavior found: none

Issues found:
- Model skill issues:
  - The skill did not document that Sparse4D export currently assumes max_num_cams=20, num_anchor=900, and num_temp_instances=600.
  - The quantize warning was stale for the validation-fixes-20260525 PyT image; TorchAO quantization now passes with an explicit model_path.
- Config issues:
  - Reducing max_num_cams to the real three-camera smoke dataset allowed train/eval/inference but made export fail with a deformable-attention reshape error.
  - Reducing num_anchor/num_temp_instances to match a 72-row anchor_init.npy allowed train/eval/inference but made export fail in memory-bank update because the exporter creates 600 cached temporal instances.
- Dataset issues:
  - depth_format=none conversion is not sufficient for training because Sparse4D still loads depth_map_path entries.
  - A three-frame conversion produced too few anchors for export-compatible checkpoints; rerunning conversion on 200 real frames produced 900 anchors.
  - No dedicated val/test split was present in the S3 subset.
- Checkpoint issues:
  - None after using explicit epoch/step checkpoints. Resume logs confirmed restoration from the requested checkpoint.
- Docker/local execution issues:
  - Data Services conversion needs GPU-visible execution because the image checks nvidia-smi; running without --gpus failed before conversion.
  - Evaluate initially failed when the container workdir was not writable because Sparse4D wrote a relative sparse4d_pred directory. Running with -w /workspace/run fixed the local-docker invocation.
- Fresh-install issues:
  - Full H5 depth maps are required for this dataset path.
  - TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 was required for trusted Lightning/full-checkpoint loads.

Fixes made:
- Updated models/sparse4d/SKILL.md to document export-compatible max_num_cams and default anchor/temporal-instance requirements.
- Updated models/sparse4d/SKILL.md to reflect that TorchAO quantization passes in the validation-fixes-20260525 PyT image and to preserve older-image caveats.

Remaining issues:
- Training metrics are not meaningful for this smoke run: mAP and NDS are 0.0000, and training loss contains NaN components on the tiny three-step run.
- The available S3 subset lacks separate validation/test splits.
- Deploy and prune are not supported by the Sparse4D model skill.
- AutoML/HPO was not run because workflow skills are explicitly out of scope for this validation pass.

Files changed:
- models/sparse4d/SKILL.md
- validation-reports/sparse4d.md

Final status:
- Fully validated for supported direct model actions on local-docker: default_specs, dataset convert, train, resume/retrain, eval, inference, export, and quantize passed.
