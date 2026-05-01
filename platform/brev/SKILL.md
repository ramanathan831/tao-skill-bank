---
name: brev
description: "Brev managed GPU instances with Docker support. Use when running training or inference on Brev GPU instances or managing Brev deployments."
---

# Brev

NVIDIA Brev provides on-demand GPU instances across multiple cloud providers. Instances come pre-loaded with NVIDIA drivers, CUDA, Docker, and NVIDIA Container Toolkit.

Brev is instance-based (not job-based like Lepton). You create an instance, run commands on it via `brev exec`, and delete it when done. The TAO SDK's BrevHandler wraps this into the standard job interface.

## Authentication

Two options:

1. **Automated (recommended)**: Get an API token from the Brev console settings page. Add `BREV_API_TOKEN` to `secrets.json`. The handler auto-authenticates via `brev login --token` on first use — same UX as Lepton.

2. **Manual**: Run `brev login` (opens browser). Tokens expire hourly — the handler refreshes automatically.

S3 credentials (ACCESS_KEY, SECRET_KEY) are needed separately for data transfer.

## Launch Preflight

Before generating scripts or submitting jobs:

1. Verify `BREV_API_TOKEN` is set.
2. Verify the `brev` CLI is installed and can list instances, for example
   `brev ls --json`. If needed, authenticate with `brev login --token`.
3. For `s3://` datasets/results, verify `ACCESS_KEY` and `SECRET_KEY` are set
   and the exact paths are readable with `aws s3 ls`.
4. Do not accept local `/path` inputs for Brev unless the user has proven those
   paths exist on the target Brev instance or are mounted into it.
5. Verify model-specific credentials such as `HF_TOKEN` before launch.

## Instance Lifecycle

The agent controls instance lifecycle:

- **Reuse**: Pass `instance_id` in `backend_details` to run multiple jobs on the same instance. Efficient for multi-step workflows.
- **Ephemeral**: Omit `instance_id` — the handler creates a new instance per job. Clean but slower (instance boot ~2-5 min).

## GPU Types

Available via `brev search`:
- L40S, A100 80GB, H100 (availability varies by provider)
- Use `--gpu-name` to filter, `--min-vram` for memory requirements

## Storage

No shared NFS/Lustre. All data flows through S3 via the script_runner's fsspec integration. Instance-local disk at `/home/ubuntu/` persists across stop/start but not across delete/create.

## Docker on Brev

VM Mode instances have Docker pre-installed. For TAO container images:

```bash
# NGC auth (one-time per instance)
brev exec <instance> -- docker login nvcr.io -u '$oauthtoken' -p <NGC_KEY>

# Run a TAO training job
brev exec <instance> -- docker run --gpus all --rm \
  -v /home/ubuntu/data:/data \
  nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt \
  visual_changenet train -e /data/spec.yaml
```

## Mixed-Platform Workflows

Brev can be mixed with Lepton in the same workflow. Per-stage platform assignment:

```json
{"skill": "vcn-gap-analysis", "action": "analyze", "platform": "brev"},
{"skill": "visual-changenet", "action": "train", "platform": "lepton"}
```

CPU stages (gap analysis, data merge) run cheaply on Brev. GPU stages (training) run on Lepton H100s.

## Error Patterns

**brev CLI not found**: Install from https://docs.nvidia.com/brev/.

**Token expired**: Handler auto-refreshes via `brev ls`. If persistent, run `brev login` manually.

**Instance stuck in provisioning**: Some GPU types have limited availability. Try a different `--gpu-name` or provider.

**Docker pull fails on nvcr.io**: NGC_KEY not set or expired. Run `docker login nvcr.io` on the instance.
