---
name: tao-sdk
description: TAO Execution SDK for submitting and monitoring GPU training jobs on cloud platforms (Lepton, Brev). Use when the user wants to train, evaluate, or run inference on a model via cloud GPU.
---

# TAO Execution SDK

The SDK provides job submission and monitoring for GPU training on cloud platforms. The agent does all the thinking (reading skills, resolving specs, mapping data). The SDK handles infrastructure (entrypoint building, mount detection, job submission).

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
- `SKILL.md` — full instructions, error patterns, data format docs
- `references/model_info.yaml` — container image, commands, inputs/outputs, data_sources
- `references/spec_template.yaml` — default training spec

## Reading a Model Skill

```python
import yaml
skill_path = catalog[0]['path']  # from sdk.list_skills()

# Execution metadata
with open(f'{skill_path}/references/model_info.yaml') as f:
    model_info = yaml.safe_load(f)
# model_info has: container_image, actions, spec_shorthand_keys, tags, pretrained_models

# Default spec
with open(f'{skill_path}/references/spec_template_train.yaml') as f:
    specs = yaml.safe_load(f)
```

## Applying Overrides

The `spec_shorthand_keys` in model_info.yaml maps flat keys to nested spec paths:

```yaml
# model_info.yaml
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

1. **List the dataset** — `sdk.list_path(dataset_uri)` to see actual contents
2. **Read the model SKILL.md** — find the spec field mapping table (which fields expect which data)
3. **Set spec fields** to the actual S3 paths found in the dataset

```python
base_uri = dataset_uri.rstrip('/')  # IMPORTANT: strip trailing slash to avoid double slashes

# Agent reads model SKILL.md, finds which spec fields need data,
# lists the dataset, matches files to fields, sets them in specs:
specs[...] = f"{base_uri}/path/to/actual/file"
```

**Do NOT assume file names or directory structure.** Always list the dataset and match against what the model SKILL.md says it expects.

## How inputs and specs work together

The `inputs` dict in `script_runner` tells the container WHICH spec keys have remote data to download. The actual URIs are in the specs dict.

```python
# 1. Agent sets URIs in specs (from dataset listing + model SKILL.md)
specs['some.input.path'] = "s3://bucket/data/actual_file.csv"

# 2. inputs declares which keys to download — from model_info.yaml, NOT URIs
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

Chain jobs using `job.results_dir`:

```python
# Step 1: Train
train_job = sdk.create_job(network_arch='<model>', action='train', ...)
# Wait for completion...

# Step 2: Evaluate (uses training results)
# Read the model SKILL.md to find the checkpoint path convention
eval_job = sdk.create_job(network_arch='<model>', action='evaluate',
    specs={..., 'checkpoint': f"{train_job.results_dir}/train/model_latest.pth"}, ...)
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

## Dataset Validation (MUST DO BEFORE SUBMITTING)

**Always validate datasets BEFORE calling create_job().** Cross-reference the actual dataset contents against the model's `data_sources` mapping. Mismatches cause silent failures inside the container.

```python
# 1. Check dataset exists
assert sdk.check_path(dataset_uri), f"Dataset not found: {dataset_uri}"

# 2. List actual contents
files = sdk.list_path(dataset_uri)
print(files)  # See what's actually in the dataset

# 3. Read the model SKILL.md to see what spec fields need data
# The SKILL.md has a table mapping spec fields to expected data types

# 4. Match actual files to spec fields
# Set each spec field to the actual S3 path where the data lives
# Do NOT assume default filenames — always check what exists
```

**Always list before setting.** Datasets have different structures — files may be in subdirectories, have different names, or use different formats than the spec template defaults. The model SKILL.md describes what format the model expects; the agent finds matching files in the actual dataset.

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

## Error Patterns

**No image provided**: `create_job()` requires `image`. Read it from `model_info.yaml['container_image']`.

**Double slash in S3 path**: Strip trailing slashes from URIs before concatenating: `base_uri.rstrip('/')`.

**Credential missing**: The SDK validates credentials lazily when the handler is first used. Check `secrets.json` has the required keys for your platform.

**Job stuck in Pending**: Check `sdk.get_job_replicas(job_id)` for `readiness_issue` — usually image pull or resource scheduling.
