---
name: local-docker
description: Local Docker execution for TAO SDK job containers using the host Docker daemon and NVIDIA GPU runtime. Use when
  running jobs on the current machine or a directly attached Docker host.
license: Apache-2.0
compatibility: Standalone — no external runtime requirements.
metadata:
  author: Ramanathan Arunachalam
  version: '0.1'
allowed-tools: Read Bash
tags:
- platform
- local
- docker
---

# Local Docker

Single-node execution platform that runs TAO jobs as named Docker containers on
the local Docker daemon. It is useful for development, debugging, small runs,
and machines where the agent host already has the required GPUs, NVIDIA driver,
Docker, and NVIDIA Container Toolkit.

Use local Docker when the data is local to the Docker host or accessible through
mounted volumes/cloud credentials. Do not use it for remote cluster scheduling,
multi-node training, or jobs that need SLURM queueing.

## Credentials

There are no platform credentials required beyond access to the Docker daemon.

Optional environment:

- **DOCKER_HOST**: Optional Docker daemon URL. If unset, the SDK uses the
  Docker Python client's normal environment/default socket resolution.
- **DOCKER_NETWORK**: Docker network for job containers. Default is
  `tao_default`.
- **DOCKER_USERNAME**: Registry username. Default is `$oauthtoken` for NGC.
- **NGC_KEY**: Used when pulling private images from `nvcr.io`.
- **HOST_SSH_PATH**: Mounted into AutoML brain containers when they need SSH keys
  to monitor remote SLURM child jobs.
- **ACCESS_KEY**, **SECRET_KEY**, **S3_ENDPOINT_URL**, **S3_BUCKET_NAME**:
  Optional S3-compatible storage settings for jobs that still read/write cloud
  storage from a local container.

## Backend Details

Use the SDK backend value `local-docker`. The local backend schema has no extra
backend details, so most routing is controlled by environment and job
parameters:

```json
{
  "backend_type": "local-docker",
  "num_gpu": 1
}
```

Following the Lepton/Brev SDK design, platform/control-plane values stay in SDK
state and Docker labels. The SDK does not inject `BACKEND`, `HOST_PLATFORM`,
`MONGOSECRET`, `DOCKER_HOST`, or `DOCKER_NETWORK` into the training container.

## Container Execution

The TAO SDK local Docker handler starts containers through the Docker Python
client:

- Backend job name uses the `tao-job-<job_id>` form used by SDK handlers.
- Command is usually `["/bin/bash", "-c", "<job command>"]`.
- Containers run detached. The SDK keeps containers by default so status and
  logs remain inspectable, unless `DOCKER_AUTO_REMOVE=true`.
- `/dev/shm` is mounted as tmpfs.
- The configured Docker network is applied by the Docker daemon for the job
  container; it is not passed through as a process environment variable.
- Existing containers with the same job id are stopped and removed before a
  replacement starts.

For GPU access, the handler auto-detects the host type:

- Tegra or Jetson hosts use `runtime="nvidia"` plus
  `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES=all`.
- Standard x86 hosts use Docker `device_requests` with GPU capabilities.

If `num_gpus` is `0`, no GPUs are assigned. If `num_gpus` is `-1`, all visible
GPUs are requested. Prefer explicit GPU counts for shared development machines.

## Storage

Local Docker accepts local and `file://` paths because the container runs on the
same Docker host. Make sure every path in the spec is either:

- mounted into the container by the handler or surrounding service,
- reachable from inside the container already, or
- a cloud URI with matching credentials.

For remote/shared filesystems, prefer the platform that owns that filesystem.
For example, use SLURM plus `lustre:///...` for Lustre paths on a cluster.

## Monitoring

- The SDK handler maps Docker container state directly: created -> Pending,
  running/restarting -> Running, paused -> Paused, exit code 0 -> Complete,
  nonzero exit -> Error.
- Logs come directly from the named container through the Docker Python client
  (`docker logs tao-job-<job_id>`).

If the container has exited, died, is being removed, or cannot be found, status
reconciliation treats the backend process as terminated.

## Cancellation

Cancellation stops the named container. GPU ownership is managed by Docker /
the NVIDIA runtime, not by TAO Core's local GPU manager.

## Failure Modes

**Docker client not initialized**: Verify the Docker Python package is installed,
set `DOCKER_HOST` if you are not using the default local socket, and confirm the
process can talk to the daemon.

**GPU assignment failed**: Requested GPUs are unavailable, the NVIDIA Container
Toolkit is not configured, or the Docker daemon cannot create GPU device
requests. Use fewer GPUs, wait for another job to finish, or verify
`docker run --gpus ...` works on the host.

**Image pull auth failed**: Set a valid `NGC_KEY` for private `nvcr.io` images
or run `docker login nvcr.io -u '$oauthtoken'` on the Docker host.

**Container exited unexpectedly**: Check `docker logs tao-job-<job_id>`, the
configured `DOCKER_NETWORK`, and the command produced by the SDK action runner.

**Path missing inside container**: A local path on the host is not necessarily
mounted into the job container. Use a path convention supported by the action
runner or configure an explicit volume through the surrounding service.
