---
name: lepton
description: "DGX Cloud Lepton managed GPU compute platform with run/status/cancel interface. Use when submitting jobs to DGX Cloud or managing Lepton GPU resources."
---

# Lepton

Managed GPU compute platform on DGX Cloud. Jobs are submitted as container workloads that run on dedicated or shared GPU node groups. Lepton handles scheduling, image pulling, log collection, and job lifecycle.

Use Lepton when you need cloud-based GPU compute without managing Kubernetes or SLURM infrastructure directly.

## Credentials

- **LEPTON_WORKSPACE_ID** (required): Determines which cluster and billing account the job runs under.
- **LEPTON_AUTH_TOKEN** (required): API token for authenticating with the Lepton control plane.
- **NGC_KEY** (optional): Used to create image pull secrets for pulling TAO container images from nvcr.io.
- **ACCESS_KEY** / **SECRET_KEY** (optional): S3-compatible storage keys for dataset and checkpoint URIs.
- **S3_ENDPOINT_URL** (optional): Custom S3 endpoint (e.g., for MinIO or non-AWS S3).
- **S3_BUCKET_NAME** (optional): Bucket for job output artifacts.
- **CLOUD_REGION** (optional): Storage region (e.g., us-east-1).

## Launch Preflight

Before generating scripts or submitting jobs:

1. Verify `LEPTON_WORKSPACE_ID` and `LEPTON_AUTH_TOKEN` are set.
2. Verify the workspace API is reachable with the packaged helper:
   `scripts/check_tao_launch_preflight.py --platform lepton ...`.
3. For `s3://` datasets/results, verify `ACCESS_KEY` and `SECRET_KEY` are set
   and the exact paths are readable with `aws s3 ls`.
4. For NFS/Lustre mounted paths, require proof from Lepton volume/storage
   permissions that the path will be mounted into the job. Do not treat a local
   filesystem `test -e` on the agent host as proof for Lepton jobs.
5. Verify model-specific credentials such as `HF_TOKEN` before launch.

## Backend Details

- **resource_shape**: GPU resource shape ID (e.g., `gpu.a100-80gb.1`, `gpu.h100.8`). If omitted, Lepton assigns from the default pool.
- **dedicated_node_group**: Node group ID for guaranteed GPU allocation (no preemption). Omit for shared resources.
- **num_nodes**: Number of nodes for distributed training. Default 1. When > 1, enables intra-job communication and PyTorch distributed initialization.

## Cloud Storage

Even though the platform is Lepton, the storage layer is S3-compatible. Always use `aws` as the `cloud_metadata` key and `s3://` as the URI protocol for both datasets and `results_dir`.

- Correct: `s3://bucket-name/path`
- Incorrect: `lepton://bucket-name/path`

The container's `get_cloud_storage_class_object()` parses the URI protocol to look up credentials in `CLOUD_METADATA[protocol][bucket]`.

## Shared Storage (NFS/Lustre)

Node groups can have NFS or Lustre volumes attached. The SDK auto-detects these and mounts them into containers for persistent cross-job data sharing.

### SDK Functions

- `sdk.get_volumes(node_group_id=None)` — returns available volumes (name, from_path, type) from node group spec
- `sdk.get_storage_permissions(volume_name, node_group_id)` — returns allowed path prefixes for a volume

The agent_runner calls these automatically to detect mounts and create the appropriate `Mount` objects for job specs.

### How the script runner uses mounts

When a Lustre mount is available:
- **Inputs**: S3 paths are mapped to Lustre (`s3://bucket/path` → `/mnt/lustre/bucket/path`). If the file exists on Lustre, it's used directly (zero download). If missing, it's downloaded from S3 to Lustre and persists for future jobs.
- **Outputs**: Results write to Lustre first (fast, persistent), then upload to S3 (durable). Downstream jobs (e.g., gap analysis) can read results directly from Lustre without an S3 round-trip.

### Volume preference order

lustre > filestore > first available

### Lustre Cache Invalidation

Lustre caches files persistently across jobs. There is no built-in invalidation. If upstream data changes but the S3 path stays the same, Lustre serves the stale cached version. To force a cache miss:

- **Rename the file** on S3 (e.g., `prompt_v2.txt` instead of overwriting `prompt.txt`)
- **Use a new storage_root** between iterations to avoid cross-iteration staleness
- **Use a new path** for any regenerated artifacts

## Monitoring

### Job Status
Use `sdk.get_job_status(job_id)` for high-level status (Pending, Running, Complete, Error).

### Replica Status
Use `sdk.get_job_replicas(job_id)` during startup for detailed replica-level info. Each replica is a dict:

```python
replicas = sdk.get_job_replicas(job_id)
for r in replicas:
    node = r["status"]["node"]["name"]           # e.g., "node-ip-10-50-111-24"
    node_group = r["status"]["node"]["node_group_id"]
    cpu = r["status"]["cpu"]                      # e.g., 2
    memory_mb = r["status"]["memory_in_mb"]       # e.g., 8192
    readiness = r["status"].get("readiness_issue")
    if readiness:
        reason = readiness["reason"]   # "InProgress", "Failed", "ConfigError"
        message = readiness["message"] # "Pulling image", "Mount point not found", etc.
```

Key readiness_issue patterns:
- `reason="InProgress"`, `message="Pulling image"` — image pull in progress (normal for large images)
- `reason="Failed"` — image pull failed (check NGC_KEY)
- `reason="ConfigError"` — node issue (mount failure, GPU error)
- No `readiness_issue` — replica is running

Replica status is especially useful when a job is stuck in Pending — it reveals whether the issue is image pulling, resource scheduling, or node health.

### Job Logs
Use `sdk.get_job_logs(job_id, tail=N)` for the most recent N log lines. Logs are fetched from Lepton's log collection service.

### Parallel Jobs
For workflow stages that run in parallel (e.g., video generation x8):

1. **Launch:** Call `execute_step(plan, step_id, extra_args={"split_id": i})` for each split. Each call returns immediately with a job_id.
2. **Monitor:** Poll all jobs: `sdk.get_job_status(job_id)` for each. Use `get_job_replicas(job_id)` for startup diagnostics.
3. **Completion:** All jobs done when every status is `Complete` or `Error`.
4. **Partial failure:** Retry only failed splits — successful splits don't need re-running. Pass the same `split_id` to `execute_step`.

## Failure Analysis

When a job fails, use `sdk.get_failure_analysis(job_id)` for automatic root cause detection:

```python
analysis = sdk.get_failure_analysis(job_id)
if analysis:
    print(analysis["err_class"])    # e.g., "ERR_PROGRAM"
    print(analysis["suggestion"])   # Human-readable fix
    for event in analysis.get("job_failure_by_node_event", []):
        print(event["node_event_name"], event["message"])
        # e.g., "OOM", "OOM encountered, victim process: cosmos-rl-evalu, pid: 3368483"
```

Returns:
- `err_class`: Error classification (`ERR_PROGRAM`, `ERR_INFRA`, etc.)
- `suggestion`: What likely went wrong and how to fix it
- `job_failure_by_node_event`: Node-level events (OOM kills, GPU errors, mount failures)
- `log_streams`: Relevant log snippets with error context

Always call this on failed jobs before retrying — it distinguishes user errors (bad config, OOM) from infrastructure issues (node failure, eviction).

## Failure Modes

**OOM killed**: Container exceeded GPU or system memory. Detection: `get_failure_analysis()` returns `node_event_name: "OOM"`. Common causes: `evaluation.batch_size` too high, `max_length` too large for available KV cache. Recovery: reduce batch_size, add GPUs with tensor parallelism, or reduce max_length.

**Image pull failure**: The TAO container image cannot be pulled from nvcr.io. Usually caused by a missing or expired image pull secret. The SDK auto-provisions the secret from NGC_KEY, but if NGC_KEY is invalid, the job will fail. Detection: check `get_job_replicas()` — `readiness_issue.reason` will show `InProgress` with `message = "Pulling image"` for extended periods, or `Failed` if the pull fails. Recovery: verify NGC_KEY is valid.

**Resource unavailable**: The requested GPU shape is not available. Job enters Queueing state indefinitely. Detection: Pending > 15 minutes, replicas show no node assignment. Recovery: try a different resource_shape or dedicated_node_group, or wait for resources.

**Auth failure**: Invalid or expired LEPTON_AUTH_TOKEN. All API calls fail with 401/403. Detection: job creation raises an exception immediately. Recovery: refresh the token and reinitialize the SDK.

**Unhealthy node**: The assigned node has infrastructure issues (mount failures, GPU errors, network problems). Detection: check `get_job_replicas()` — `readiness_issue.reason = "ConfigError"` with messages like `"Mount point not found"`. The job stays Pending indefinitely on the bad node. Recovery: cancel the job and resubmit — Lepton will schedule on a different node. If the issue recurs, try a different `dedicated_node_group` or `resource_shape`.

**Job eviction**: On shared node groups, Lepton may evict jobs under resource pressure. Detection: job unexpectedly transitions from Running to Error. Recovery: retry, or use a dedicated_node_group.
