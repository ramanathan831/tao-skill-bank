---
name: tao-sdk
description: TAO Execution SDK for submitting and monitoring GPU training jobs on supported platforms (Lepton, Brev, SLURM, local Docker). Use when the user wants to train, evaluate, or run inference on a model via GPU compute.
---

# TAO Execution SDK

The SDK provides job submission and monitoring for GPU training on supported
platforms. The agent does all the thinking (reading skills, resolving specs,
mapping data). The SDK handles infrastructure (entrypoint building, mount
detection, job submission).

## Setup

```python
from tao_sdk import TaoExecutionSDK
sdk = TaoExecutionSDK(creds_file='secrets.json')
```

Credentials in `secrets.json`:
```json
{
  "LEPTON_WORKSPACE_ID": "...",
  "LEPTON_AUTH_TOKEN": "...",
  "BREV_API_TOKEN": "...",
  "SLURM_USER": "...",
  "SLURM_HOSTNAME": "login-1,login-2",
  "SLURM_BASE_RESULTS_DIR": "/lustre/fsw/portfolios/edgeai/users/<user>",
  "DOCKER_HOST": "unix:///var/run/docker.sock",
  "DOCKER_NETWORK": "tao_default",
  "NGC_KEY": "...",
  "ACCESS_KEY": "...",
  "SECRET_KEY": "...",
  "S3_ENDPOINT_URL": "...",
  "S3_BUCKET_NAME": "...",
  "HF_TOKEN": "..."
}
```

## Skill Discovery

```python
catalog = sdk.list_skills()
# Returns: [{name, layer, description, actions, tags, path, has_skill_md, agent_native}, ...]
```

Each skill is at the returned `path`. Read:
- `<model>.md` or `SKILL.md` — full instructions, error patterns, data format docs
- `config.json` — container image, commands, inputs/outputs, data_sources
- `defaults-<action>.json` — default spec for the requested action

## Reading a Model Skill

Prefer the `SkillBank` API when generating action runners. It is the same
loader used by the plugin and avoids file searching:

```python
from tao_sdk.planner import SkillBank

bank = SkillBank()
model_info = bank.get_model_config(network_arch)
specs = bank.get_default_specs(network_arch, action=action)
```

If you already have a catalog entry from `sdk.list_skills()`, direct reads are
also valid:

```python
import json

skill_path = catalog[0]["path"]  # from sdk.list_skills()
action = "evaluate"

with open(f"{skill_path}/config.json", encoding="utf-8") as f:
    model_info = json.load(f)

with open(f"{skill_path}/defaults-{action}.json", encoding="utf-8") as f:
    specs = json.load(f)
```

`model_info["actions"][action]` is the source of truth for the command,
config format, script-runner inputs, outputs, and upload excludes. Do not
debug generated runner scripts to rediscover these fields; fix the skill bank
metadata instead.

## Generated Action Runner Contract

Generated runners for normal actions (`train`, `evaluate`, `export`,
`inference`, etc.) should be thin wrappers over the skill bank and SDK:

1. Load `model_info = SkillBank().get_model_config(network_arch)`.
2. Load `specs = SkillBank().get_default_specs(network_arch, action)`.
3. Apply the model MD's "Spec Param / Parent Model Inference" guidance through SDK inference helpers.
4. Apply user overrides and model-skill documented path conventions.
5. Build `script_runner` directly from `model_info["actions"][action]`.
6. Submit with `TaoExecutionSDK.create_job(...)`.
7. Write and sync an `ActionWorkflow` folder.

Do not inspect or patch the generated runner script to fix missing inputs,
checkpoint paths, config format, commands, or upload excludes. Those are skill
bank metadata bugs. Update the model skill's `config.json`,
`defaults-<action>.json`, or instructions, then regenerate and rerun.

## Spec Param Inference

Model-specific inference mappings belong in the model Markdown file, not
`config.json`. The MD may document microservices-style `spec_params`, for
example:

```json
{
  "spec_params": {
    "evaluate": {
      "evaluate.checkpoint": "parent_model"
    }
  }
}
```

Generated runners should read that model MD section and apply the documented
mappings before submission. For `parent_model`, pass the upstream train job id
as `parent_job_id`; the SDK mirrors the old microservices path by listing the
parent job's results folder, filtering model checkpoint files, and returning
the selected `.pth`/`.tlt`/`.hdf5` path.

```python
import uuid

def set_nested(specs, dotted_key, value):
    target = specs
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value

job_id = str(uuid.uuid4())
parent_job_id = train_job_id  # train job, AutoML best child job, export job, etc.

# Derived from the current model MD, not from config.json.
spec_param_map = {
    "evaluate.checkpoint": "parent_model",
}

for field_name, inference_fn in spec_param_map.items():
    value = sdk.resolve_spec_param(
        job_id,
        inference_fn,
        network_arch=network_arch,
        parent_job_id=parent_job_id,
    )
    if value:
        set_nested(specs, field_name, value)
```

When submitting the action, preserve the relationship:

```python
job = sdk.create_job(..., job_id=job_id, parent_job_id=parent_job_id)
```

Do not hardcode a checkpoint path when `parent_model` is documented. If the SDK
cannot list the parent result folder, fix the result-folder metadata or the
model MD checkpoint guidance instead of patching the runner script.

## Applying Overrides

The `spec_shorthand_keys` in model_info/config maps flat keys to nested spec paths:

```yaml
# model_info/config
spec_shorthand_keys:
  num_epochs: train.num_epochs
  num_gpus: train.num_gpus
  batch_size: train.batch_size
```

To apply `num_epochs=10`:
```python
# Read the shorthand mapping
path = model_info['spec_shorthand_keys']['num_epochs']  # "train.num_epochs"
parts = path.split('.')
target = specs
for part in parts[:-1]:
    target = target.setdefault(part, {})
target[parts[-1]] = 10
```

## Mapping Dataset Files to Spec Fields

Each model SKILL.md documents which spec fields need dataset paths. The general pattern:

1. **Read the model SKILL.md** — find the spec field mapping table and any exact artifact-name conventions.
2. **Set spec fields** from those conventions when the skill documents exact filenames.
3. **List the dataset** with `sdk.list_path(dataset_uri)` only when the model skill does not provide exact filenames or the user says their layout is nonstandard.

```python
base_uri = dataset_uri.rstrip('/')  # IMPORTANT: strip trailing slash to avoid double slashes

# Agent reads model SKILL.md, finds which spec fields need data.
# If the skill gives exact filenames, construct them directly:
specs[...] = f"{base_uri}/path/to/actual/file"
```

Do not invent filenames. Use the model skill's declared filenames when present.
If the skill only describes a format or the user provides a custom layout, list
the remote path and match actual files to the documented spec fields.

## How inputs and specs work together

The `inputs` dict in `script_runner` tells the container WHICH spec keys have remote data to download. The actual URIs are in the specs dict.

```python
# 1. Agent sets URIs in specs (from model SKILL.md conventions, or dataset listing when needed)
specs['some.input.path'] = "s3://bucket/data/actual_file.csv"

# 2. inputs declares which keys to download — from model_info/config, NOT URIs
inputs = model_info['actions'][action]['inputs']
# e.g. {"some.input.path": {"type": "file"}, "some.folder.path": {"type": "folder"}}
```

The script_runner:
1. Reads the spec value at the declared key → gets the S3 URI
2. Checks the input type (file or folder)
3. Downloads from S3
4. Rewrites the spec value to the local download path

## Submitting a Job

```python
action = 'train'
action_config = model_info['actions'][action]

job = sdk.create_job(
    network_arch=model_info.get('network_arch', '<model>'),
    action=action,
    image=model_info['container_image'],
    specs=specs,
    train_dataset_uri=dataset_uri,
    workspace_id=sdk._workspace_id,
    script_runner={
        'command': action_config['command'],
        'config_format': action_config.get('config_format', 'yaml'),
        'inputs': action_config.get('inputs', {}),
        'outputs': action_config.get('outputs', {}),
        'upload_excludes': action_config.get('upload_excludes', []),
    },
    backend_details={
        'backend_type': 'lepton',
        'num_gpus': 8,
    },
)

print(f"Job submitted: {job.id}")
print(f"Results will be at: {job.results_dir}")
```

## Non-AutoML Action Workflow Folders

For direct plugin-launched actions (`train`, `evaluate`, `export`,
`inference`, etc.), create a timestamped workflow folder and sync status
through the SDK. Do **not** start a long-lived per-job local process just to
monitor status. AutoML needs a long-lived brain process because it proposes
and launches multiple child train jobs; ordinary actions are single backend
jobs and should refresh status on demand.

```python
from datetime import datetime
from tao_sdk.action_workflow import ActionWorkflow
from tao_sdk.sdk import TaoExecutionSDK

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
workflow = ActionWorkflow(
    root_dir="./eval_runs",
    run_name=f"{network_arch}_{action}",
    timestamp=timestamp,
)

sdk = TaoExecutionSDK(
    creds_file="~/tao-sdk/secrets.json",
    state_file=str(workflow.workspace / "tao_session_state.json"),
)

workflow.write_metadata(
    network_arch=network_arch,
    action=action,
    state_file=str(workflow.workspace / "tao_session_state.json"),
)

job = sdk.create_job(...)
workflow.write_submission(job=job, specs=specs, script_runner=script_runner)
workflow.sync_from_sdk(sdk, job.id)  # one immediate refresh, then exit
```

When the user asks for status later, re-open the same workflow folder and do a
single refresh:

```python
workflow = ActionWorkflow.from_workspace(workspace_path)
sdk = TaoExecutionSDK(
    creds_file="~/tao-sdk/secrets.json",
    state_file=f"{workspace_path}/tao_session_state.db",
)
status = workflow.sync_from_sdk(sdk, job_id)
```

This keeps `metadata.json`, `status.json`, `status_events.jsonl`,
`active_jobs.json`, `latest_logs.txt`, and `failure_analysis.json` aligned with
Lepton status without requiring a separate PID for every non-AutoML job.

## Monitoring

```python
status = sdk.get_job_status(job.id)
print(status.status)  # Pending, Running, Complete, Error, Canceled

logs = sdk.get_job_logs(job.id, tail=50)
print(logs)

# On failure:
analysis = sdk.get_failure_analysis(job.id)
if analysis:
    print(analysis['err_class'])
    print(analysis['suggestion'])
```

## Polling Pattern

Only use a local polling loop when the user explicitly asks to watch a job in
the foreground. For normal plugin operation, prefer the workflow-folder
`sync_from_sdk()` refresh shown above.

```python
import time
while True:
    status = sdk.get_job_status(job.id)
    if status.status in ('Complete', 'Error', 'Canceled'):
        break
    print(f"Status: {status.status}")
    time.sleep(30)

if status.status == 'Error':
    logs = sdk.get_job_logs(job.id)
    analysis = sdk.get_failure_analysis(job.id)
    # Diagnose and fix
```

## Multi-Step Workflows

Chain jobs using the upstream job id and skill `spec_params`:

```python
# Step 1: Train
train_job = sdk.create_job(network_arch='<model>', action='train', ...)
# Wait for completion...

# Step 2: Evaluate (uses training results)
checkpoint = sdk.resolve_spec_param(
    eval_job_id,
    "parent_model",
    network_arch="<model>",
    parent_job_id=train_job.id,
)
set_nested(specs, "evaluate.checkpoint", checkpoint)
eval_job = sdk.create_job(
    network_arch="<model>",
    action="evaluate",
    job_id=eval_job_id,
    parent_job_id=train_job.id,
    specs=specs,
    ...
)
```

## Parallel Execution

```python
jobs = []
for split_id in range(8):
    j = sdk.create_job(network_arch='cosmos-predict-2-5', action='generate', ...)
    jobs.append(j)

# Wait for all
for j in jobs:
    while sdk.get_job_status(j.id).status not in ('Complete', 'Error'):
        time.sleep(30)
```

## Dataset Validation

Validate datasets before calling `create_job()` when the model skill does not
declare deterministic filenames, when the user provided a custom layout, or
when a previous run failed due to missing data. Cross-reference actual contents
against the model's documented data mapping. Mismatches cause failures inside
the container.

```python
# 1. Check dataset exists
assert sdk.check_path(dataset_uri), f"Dataset not found: {dataset_uri}"

# 2. List actual contents only if needed
files = sdk.list_path(dataset_uri)
print(files)

# 3. Read the model SKILL.md to see what spec fields need data
# The SKILL.md has a table mapping spec fields to expected data types

# 4. Match actual files to spec fields when no exact convention exists
# Set each spec field to the actual S3 path where the data lives
```

For standard layouts documented by a model skill, do not list just to rediscover
known filenames. For example, DINO standard datasets use `images.tar.gz` and
`annotations.json`; construct those URIs from the dataset base URI.

## Platform-Specific Notes

### Lepton
- Jobs run as containers on DGX Cloud
- NFS/Lustre mounts auto-detected
- `backend_details.num_gpus` maps to resource_shape

### Brev
- Jobs run on GPU instances via `brev exec`
- No shared storage — S3 only
- Pass `instance_id` in backend_details to reuse an instance
- `backend_details.gpu_type` for instance type selection

### SLURM
- Jobs submit over SSH to a login node with `sbatch` and run containers through
  Pyxis/Enroot `srun --container-image`
- Use `backend_details.backend_type = "slurm"` and pass `partition` when the
  user requests a specific queue
- Dataset paths must be cluster-visible, usually `lustre:///absolute/path`; do
  not pass local or `file://` paths to SLURM jobs
- Results default to
  `/lustre/fsw/portfolios/edgeai/users/<slurm_user>/results/<job_id>`
- Status uses both scheduler state (`squeue`/`sacct`) and `status.json` from the
  shared results directory

### Local Docker
- Jobs run as named Docker containers on the configured `DOCKER_HOST`
- Requires Docker daemon access and NVIDIA Container Toolkit on GPU hosts
- Follows the Lepton/Brev SDK principle: run the baked SDK entrypoint directly
  and keep platform metadata in SDK state/Docker labels, not in TAO Core
  control-plane env vars
- Do not inject `BACKEND`, `HOST_PLATFORM`, `MONGOSECRET`, `DOCKER_HOST`, or
  `DOCKER_NETWORK` into the training container
- Use explicit `num_gpu` values for shared machines; `-1` requests all visible
  GPUs
- Local and `file://` paths are valid only when reachable inside the container
- Logs are read with the Docker client from the job container

## Error Patterns

**No image provided**: `create_job()` requires `image`. Read it from `model_info['container_image']`.

**Double slash in S3 path**: Strip trailing slashes from URIs before concatenating: `base_uri.rstrip('/')`.

**Credential missing**: The SDK validates credentials lazily when the handler is first used. Check `secrets.json` has the required keys for your platform.

**Job stuck in Pending**: Check `sdk.get_job_replicas(job_id)` for `readiness_issue` — usually image pull or resource scheduling.

**SLURM local path rejected**: Remote backends reject local dataset paths. Copy
the data to the cluster filesystem and use `lustre:///...`.

**Local Docker client unavailable**: Set `DOCKER_HOST` and verify the process can
access the Docker socket.
