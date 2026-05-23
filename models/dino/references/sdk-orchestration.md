# DINO SDK Orchestration Internals

Read this when running DINO through TAO SDK orchestration, S3 I/O wrapping, or
AutoML.

## Contents

- Spec template source notes.
- Data source auto-resolution limitation.
- DINO checkpoint inference source of truth.

## Spec Templates

DINO ships without `references/spec_template_train.yaml` or
`references/spec_template_evaluate.yaml`. To use SDK orchestration, generate
them from upstream:

- `spec_template_train.yaml` from `tao-pytorch/nvidia_tao_pytorch/cv/dino/experiment_specs/train.yaml`; replace `"???"` placeholders with empty strings.
- `spec_template_evaluate.yaml` from `tao-pytorch/nvidia_tao_pytorch/cv/dino/experiment_specs/evaluate.yaml` plus the shared `evaluate.checkpoint` field expected by `initialize_evaluation_experiment()`.

## Data Sources Gap

DINO's `config.json` has `"data_sources": {}`. The runner's
`_apply_data_sources()` handles flat spec keys, but DINO data sources are arrays
of objects such as `dataset.train_data_sources[{image_dir, json_file}]`. The
tao-core microservices config has the full mapping using a `mapping`
sub-structure, but the runner does not support that format.

Consequence: data paths must be set manually via `spec_overrides` from the
Training Requirements table in `models/dino/SKILL.md`. The skill's `config.json`
declares `[0]`-indexed train inputs so the SDK script runner downloads S3 data
at runtime:

```json
"inputs": {
    "dataset.train_data_sources[0].image_dir": {"type": "file"},
    "dataset.train_data_sources[0].json_file": {"type": "file"},
    "dataset.val_data_sources[0].image_dir": {"type": "file"},
    "dataset.val_data_sources[0].json_file": {"type": "file"}
}
```

Evaluate inputs are also declared so generated eval runners do not need to patch
`script_runner` by hand:

```json
"inputs": {
    "evaluate.checkpoint": {"type": "file"},
    "dataset.test_data_sources.image_dir": {"type": "file"},
    "dataset.test_data_sources.json_file": {"type": "file"}
}
```

## Checkpoint Inference

```text
checkpoint format: pth
evaluate.checkpoint: parent_model
```

All model-specific metadata, including dataset type, formats, metrics, and
required datasets, remains in the Training Requirements section of
`models/dino/SKILL.md`.
