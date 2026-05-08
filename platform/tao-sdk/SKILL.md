---
name: tao-sdk
description: TAO Execution SDK for submitting and monitoring GPU training jobs on supported platforms (Lepton, Brev, SLURM, local Docker, Kubernetes). Use when the user wants job tracking, S3 I/O wrapping, multi-node distributed training, or platform-specific features that docker-run can't provide.
license: Apache-2.0
compatibility: Requires Python 3.10+ and the nvidia-tao-sdk package (pip install nvidia-tao-sdk).
metadata:
  author: Ramanathan Arunachalam, Arif Ahmed
  version: '0.2'
allowed-tools: Read Bash
tags:
- platform
- tao
- sdk
---

# TAO Execution SDK

The SDK is the **optional** Python layer for users who need job handles, S3 I/O wrapping, or platform-specific features (Lepton multi-node, SLURM/Lustre queues, Kubernetes Jobs, local Docker debugging, Brev instance reuse). Most TAO skills run with just `docker run` and don't need it. Reach for the SDK when:

- You want a `Job` handle to poll status and stream logs over time.
- The platform is API-only (Lepton has no docker-run equivalent).
- You need S3-aware input download / output upload baked into the entrypoint.
- You're chaining multiple jobs and want persisted state.

## Preflight

```bash
python -c "import tao_sdk" 2>/dev/null || {
  echo "MISSING: nvidia-tao-sdk not installed. Run:"
  echo "  pip install nvidia-tao-sdk[lepton]   # or [brev], [all]"
  exit 1
}
```

If missing, the agent prompts the user to authorize the install via Bash, then re-runs the preflight. Never auto-install silently.

## Setup

Credentials come from **environment variables** — sourced from `~/.config/tao/.env` (auto-loaded by the skill bank's SessionStart hook).

```python
from tao_sdk.platforms.lepton import LeptonSDK   # DGX Cloud
from tao_sdk.platforms.brev   import BrevSDK     # Brev GPU instances

sdk = LeptonSDK()    # reads LEPTON_WORKSPACE_ID, LEPTON_AUTH_TOKEN
# or
sdk = BrevSDK()      # reads BREV_API_TOKEN (optional — falls back to brev login)
```

Both SDKs validate credentials lazily on first use and raise `CredentialError` with a clear message if a required env var is missing. Required env vars:

| Platform | Required | Optional |
|---|---|---|
| Lepton | `LEPTON_WORKSPACE_ID`, `LEPTON_AUTH_TOKEN` | — |
| Brev | — (manual `brev login` works) | `BREV_API_TOKEN` |
| S3 I/O (any platform) | `S3_BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY` | `S3_ENDPOINT_URL`, `CLOUD_REGION` |
| Container env | `NGC_KEY` | `HF_TOKEN` |

The agent never reads credential values — it only checks presence with `[ -n "$VAR_NAME" ]`.

## Workflow Launch Intake

For any TAO workflow or action launch, first confirm the user goal. Then ask
for platform and monitoring preferences before credentials or launch details.
Generate the supported platform choices from the packaged helper, not by
scanning platform docs or folders:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

Ask:

1. Which supported platform should run this workflow?
2. Should long-running monitoring stay enabled? Default: enabled. This means
   the agent remains attached and posts status until terminal state, including
   long `PENDING` queue waits.
3. How many minutes between status updates? Default: 5 minutes.

After the model/action are known, resolve the default container image from the
packaged metadata and ask the user to confirm it or provide `image=<override>`
before creating runner files:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/resolve_tao_image.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --model <network_arch> --action <action> --format text
```

After the platform is selected, get the credential filter:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> --format text
```

Ask only for credentials returned for the selected platform. For example, SLURM
needs `SLURM_USER` and `SLURM_HOSTNAME`; it does not need Lepton credentials.
Kubernetes and local Docker do not need Lepton or SLURM credentials. Ask storage
credentials such as S3 keys only when the selected platform and the data/result
URIs require them.

## Core API

All platform SDKs implement the same core shape:

```python
sdk.create_job(image, command, gpu_count=1, env_vars=None, inputs=None, outputs=None, **kwargs) -> Job
sdk.get_job_status(job_id) -> JobStatus
sdk.get_job_logs(job_id, tail=None) -> str
sdk.cancel_job(job_id) -> bool
sdk.get_failure_analysis(job_id) -> dict | None
sdk.get_job_results_dir(job_id) -> str
sdk.check_path(remote_path) -> bool
sdk.list_path(remote_path) -> list[str]
```

Lepton-only:
- `sdk.get_job_replicas(job_id)` — replica-level diagnostics for stuck-pending jobs.

Brev-only:
- `sdk.delete_instance(instance_id)` — clean up an ephemeral instance.
- `sdk.list_instances()` — list active instances.

## Submitting a Job

The agent reads the skill's `references/skill_info.yaml` to get `container_image` and `actions.<action>.command`, builds the spec, then constructs the full shell command and calls `create_job()`:

```python
import yaml
from tao_sdk.platforms.lepton import LeptonSDK
from tao_sdk.versions import resolve_container_image

# 1. Read the skill's metadata
skill_info = yaml.safe_load(open(f"{skill_path}/references/skill_info.yaml"))
image = resolve_container_image(skill_info["container_image"])  # accepts key or absolute URI
command_template = skill_info["actions"]["train"]["command"]
# e.g. "dino train -e {config_path}"

# 2. Build the spec (agent reads SKILL.md to know what fields to set)
specs = {
    "dataset": {
        "train_data_sources": [{
            "image_dir": "/data/train/images",
            "json_file": "/data/train/annotations.json",
        }],
        "val_data_sources": [{
            "image_dir": "/data/val/images",
            "json_file": "/data/val/annotations.json",
        }],
        "num_classes": 80,
    },
    "train": {"num_epochs": 10, "num_gpus": 8},
    "results_dir": "/results",
}

# 3. Construct the full shell command (spec heredoc + the action command)
spec_heredoc = f"""cat > /tmp/spec.yaml <<'EOF'
{yaml.safe_dump(specs)}
EOF
"""
command = spec_heredoc + command_template.replace("{config_path}", "/tmp/spec.yaml")

# 4. Submit
sdk = LeptonSDK()
job = sdk.create_job(
    image=image,
    command=command,
    gpu_count=8,
    inputs={
        "/data/train/images":     "s3://my-bucket/coco/train/images",
        "/data/train/annotations.json": "s3://my-bucket/coco/train/annotations.json",
        "/data/val/images":       "s3://my-bucket/coco/val/images",
        "/data/val/annotations.json":   "s3://my-bucket/coco/val/annotations.json",
    },
    outputs=["/results/"],
    env_vars={"NGC_KEY": os.environ["NGC_KEY"]},
    # Lepton-specific:
    dedicated_node_group="my-h100-pool",   # optional
    num_nodes=1,
)

print(f"Job submitted: {job.id}")
print(f"Results: {job.results_dir}")
```

`inputs` and `outputs` are S3-aware: the SDK injects a script_runner entrypoint that downloads inputs to the listed container paths before the command runs, then uploads the listed output paths to S3 (under `s3://$S3_BUCKET_NAME/results/<job_id>/`) after.

## Resolving container images

Skills declare images either by key (`tao_toolkit.pyt`) or as an absolute URI (`nvcr.io/...`). Use `resolve_container_image()` to handle both:

```python
from tao_sdk.versions import resolve_container_image
image = resolve_container_image(skill_info["container_image"])
```

Behind the scenes it walks `versions.yaml` for keys; absolute URIs are returned as-is.

## Monitoring

```python
status = sdk.get_job_status(job.id)
print(status.status)   # Pending, Running, Complete, Error, Canceled
print(status.message)  # platform-specific detail

logs = sdk.get_job_logs(job.id, tail=200)
print(logs)
```

For stuck-Pending Lepton jobs, replica diagnostics reveal the cause (image pull, scheduling, mount errors):

```python
for r in sdk.get_job_replicas(job.id):
    issue = r["status"].get("readiness_issue")
    if issue:
        print(issue["reason"], issue["message"])
        # e.g. "InProgress" / "Pulling image"  (normal for big images)
        #      "Failed"     / "ImagePullBackOff" (NGC_KEY problem)
        #      "ConfigError" / "Mount point not found" (bad node)
```

On failure, `get_failure_analysis()` classifies the root cause:

```python
analysis = sdk.get_failure_analysis(job.id)
if analysis:
    print(analysis["err_class"])   # ERR_PROGRAM, ERR_INFRA, etc.
    print(analysis["suggestion"])  # human-readable fix
    for event in analysis.get("job_failure_by_node_event", []):
        print(event["node_event_name"], event["message"])  # OOM, GPU error, etc.
```

## Polling pattern

For interactive runs where the user wants to watch:

```python
import time
status_interval_minutes = status_interval_minutes or 5
while True:
    status = sdk.get_job_status(job.id)
    if status.status in ("Complete", "Error", "Canceled"):
        break
    print(f"  {status.status}")
    time.sleep(status_interval_minutes * 60)

if status.status == "Error":
    print(sdk.get_job_logs(job.id, tail=100))
    print(sdk.get_failure_analysis(job.id))
```

With long-running monitoring enabled, do not stop after 30 minutes or after a
few unchanged polls. Keep emitting updates every `status_interval_minutes`
until the job finishes, fails, is canceled, or the user asks to detach/stop.
If the chat/runtime cannot remain open that long, say so explicitly and provide
the durable workflow/log path for manual status refresh.

Do not use a final response for non-terminal monitored jobs. Finalizing the
turn detaches the chat watcher. Keep non-terminal status messages in progress
updates and continue polling; only finalize at terminal state, explicit user
detach/stop, or a real runtime limit that prevents further polling.

For background runs, persist `job.id` and the `state_file` path, then re-attach later by constructing the same SDK and calling `get_job_status(job_id)` — job state is read from the on-disk store.

## Multi-step workflows

The agent chains jobs by waiting for a parent to complete, then constructing the next job's command using the parent's results directory:

```python
# Step 1: train
train = sdk.create_job(image=img, command=train_cmd, gpu_count=8, ...)
while sdk.get_job_status(train.id).status not in ("Complete", "Error"):
    time.sleep(30)
assert sdk.get_job_status(train.id).status == "Complete"

# Step 2: evaluate (uses the train results dir)
ckpt = f"{train.results_dir}/best.pth"
eval_cmd = make_eval_command(checkpoint=ckpt, ...)
eval_job = sdk.create_job(image=img, command=eval_cmd, gpu_count=1, ...)
```

There is no `SkillBank`, `Planner`, or `parent_job_id` mechanism — workflow orchestration is the agent's job, not the SDK's. (The SDK does ship an `ActionWorkflow` helper for run-folder durability — see below.)

## Run-folder durability with `ActionWorkflow`

Optional state-persistence helper for skills that want a durable run folder
across context breaks. Decoupled from any specific platform.

```python
from datetime import datetime
from tao_sdk.action_workflow import ActionWorkflow
from tao_sdk.platforms.lepton import LeptonSDK

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
workflow = ActionWorkflow(root_dir="./runs", run_name="dino-train", timestamp=ts)
sdk = LeptonSDK(state_file=str(workflow.workspace / "tao_session_state.json"))

workflow.write_metadata(network="dino", action="train", dataset_uri="s3://bucket/coco/")
job = sdk.create_job(image=..., command=..., gpu_count=8, ...)
workflow.write_submission(job=job, specs=specs, script_runner={})
workflow.sync_from_sdk(sdk, job.id)  # writes status.json + latest_logs.txt + failure_analysis.json
```

The folder layout (`./runs/dino-train/<timestamp>/`):
- `metadata.json` — what the user asked for
- `status.json` — current job status snapshot
- `status_events.jsonl` — append-only event log
- `active_jobs.json` — in-flight job IDs (drained on terminal)
- `latest_logs.txt` — last polled log tail
- `failure_analysis.json` — populated on failure

Re-attach later with `ActionWorkflow.from_workspace(path)`. Works with any
SDK that has `get_job_status` / `get_job_logs` / `get_failure_analysis` —
Lepton, Brev, Docker, SLURM, Kubernetes.

## Parallel execution

```python
jobs = [sdk.create_job(image=img, command=make_cmd(i), gpu_count=1, ...) for i in range(8)]
# Poll all
while not all(sdk.get_job_status(j.id).status in ("Complete", "Error") for j in jobs):
    time.sleep(30)
```

## Dataset utilities

When the skill's documented filenames don't match the user's layout, list the dataset to confirm:

```python
assert sdk.check_path("s3://my-bucket/coco/")
files = sdk.list_path("s3://my-bucket/coco/train/")
# Use the actual paths to set spec fields.
```

For S3 paths, strip trailing slashes when concatenating to avoid `//`:

```python
base = dataset_uri.rstrip("/")
spec["dataset.train_csv"] = f"{base}/train.csv"
```

## Platform-specific notes

### Lepton (`from tao_sdk.platforms.lepton import LeptonSDK`)
- Jobs run as containers on DGX Cloud.
- NFS/Lustre mounts auto-detected from the node group; the SDK builds the appropriate `Mount` objects.
- `gpu_count` resolves to a Lepton resource shape; or pass `dedicated_node_group="<name>"` for guaranteed allocation.
- `num_nodes=N` (N>1) enables distributed training.

### Brev (`from tao_sdk.platforms.brev import BrevSDK`)
- Jobs run on GPU instances via `brev exec`.
- No shared storage — S3 only.
- Pass `instance_id="<id>"` in kwargs to reuse an existing instance (skip 2–5 min boot).
- Pass `gpu_type="L40S"` to control instance class for ephemeral instances.
- Use `sdk.delete_instance(instance_id)` when done with an ephemeral one.

### SLURM
- Jobs submit over SSH to a login node with `sbatch` and run containers through
  Pyxis/Enroot `srun --container-image`.
- Use the platform helper output to ask only for SLURM credentials and storage
  settings. Do not ask for Lepton, Brev, or Kubernetes credentials.
- Dataset paths must be visible from the cluster job, usually absolute Lustre or
  shared filesystem paths; do not pass agent-host local paths to SLURM jobs.
- Use the packaged SLURM runtime defaults unless the user gives a validated
  override. For the common `polar,polar3,polar4,grizzly` queues, prefer the
  four-hour default rather than generating 12-hour wrappers.

### Kubernetes
- Jobs run as Kubernetes Jobs on a configured GPU cluster.
- Auth uses kubeconfig (`KUBECONFIG` or `~/.kube/config`) or an in-cluster
  service account.
- Requires NVIDIA GPU Operator or equivalent `nvidia.com/gpu` device plugin.
- Do not ask for Lepton, Brev, or SLURM credentials for Kubernetes runs.
- A local path on the agent host is not proof that the path is mounted inside
  the job pod.

### Local Docker
- Jobs run on the local Docker daemon host.
- Multi-node is not supported; multi-GPU on the local host is supported.
- Verify local dataset paths, Docker daemon access, and NVIDIA runtime before
  generating or launching runner artifacts.

## Error patterns

**`CredentialError: Missing LEPTON_WORKSPACE_ID`**: env var not loaded. Run `source ~/.config/tao/.env` or check the SessionStart hook fired.

**`CredentialError: S3_BUCKET_NAME env var required`**: any `inputs` or `outputs` argument needs S3 credentials. Set `S3_BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY` (and `S3_ENDPOINT_URL` for non-AWS).

**Job stuck in `Pending` (Lepton)**: call `get_job_replicas(job_id)` and inspect `readiness_issue`. Most common: image pull (waited too long) or `ConfigError` on a bad node — cancel and resubmit.

**`Image pull failed`**: `NGC_KEY` is invalid or expired. The SDK auto-creates a Lepton image-pull-secret from `$NGC_KEY`; refresh the key and resubmit.

**Double slash in S3 URI**: `dataset_uri.rstrip("/")` before concatenating, or use `os.path.join` (note: not `posixpath.join` — that doesn't strip).

**Brev instance won't start**: GPU type unavailable in the user's region. Try a different `gpu_type` or wait.

## What the SDK does NOT do

- It does **not** read or interpret skills. The agent reads `SKILL.md` and `references/skill_info.yaml`; the SDK just submits whatever command the agent constructs.
- It does **not** do hyperparameter optimization. For HPO, use `applications/tao-automl` (which uses this SDK as a building block).
- It does **not** generate config files. The agent writes the spec heredoc into the `command` argument.
- It does **not** select platforms automatically. Pick the requested platform SDK explicitly.
- It does **not** orchestrate multi-step workflows. The agent chains jobs by polling and constructing the next command.
