---
name: tao-run-on-docker
description: The Docker execution platform for TAO jobs — a local daemon or a remote GPU box via DOCKER_HOST=ssh://user@host. Implements the four-verb consumer contract (submit/status/logs/cancel) over the docker CLI, wired to the job-record, tao-data-io staging, and the redact lint, on top of the underlying docker conventions (--gpus, mounts, NGC auth, inspection, data-root relocation, error modes). Use to run any single-node TAO container action on Docker without the SDK. Trigger keywords — docker, docker run, run on docker, DOCKER_HOST, remote docker, nvcr.io, --gpus, single-node GPU job.
license: Apache-2.0
compatibility: Requires NVIDIA driver branch 580, CUDA Toolkit 13.0, Docker, and NVIDIA Container Toolkit 1.19.0.
metadata:
  version: "0.1.0"
  author: NVIDIA Corporation
allowed-tools: Read Bash
tags:
- platform
- docker
---

# Docker for NVIDIA GPU Workloads

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

The Docker execution platform: a **consumer** that runs a model/data skill's
spec-bundle by implementing four verbs (`submit`/`status`/`logs`/`cancel`) over
the docker CLI, on a **local daemon or a remote GPU box via
`DOCKER_HOST=ssh://`**. The verbs (§ Execution) sit on top of the docker
conventions in the rest of this file — GPU flags, mounts, NGC auth, inspection,
error modes — which are the *how* the model/data skill defers to. Single-node
only; for multi-node use SLURM or Kubernetes.

Sources: official Docker CLI reference (<https://docs.docker.com/reference/cli/docker/>) and NVIDIA Container Toolkit docs.

## Prerequisites

1. **Host GPU runtime** — NVIDIA driver branch 580, CUDA Toolkit 13.0, and NVIDIA Container Toolkit 1.19.0. Check with the `tao-setup-nvidia-gpu-host` skill before any GPU workflow starts.
2. **Docker** — `docker --version` must return ≥ 20.10. Install: <https://docs.docker.com/engine/install/>.
3. **NGC API key** for `nvcr.io/*` pulls. Get from <https://ngc.nvidia.com/>.

```bash
TAO_SKILL_BANK_ROOT="${TAO_SKILL_BANK_ROOT:-$PWD}"
SETUP_SCRIPT="${TAO_SKILL_BANK_ROOT}/platform/tao-setup-nvidia-gpu-host/scripts/setup-nvidia-gpu-host.sh"

bash "$SETUP_SCRIPT" --backend docker --check-only || {
  echo "MISSING: TAO GPU host runtime is not ready."
  echo "After user approval, run (append --yes for non-interactive agent runs):"
  echo "  bash \"$SETUP_SCRIPT\" --backend docker --install"
  exit 1
}

docker --version
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
[ -n "$NGC_KEY" ] || echo "NGC_KEY unset — cannot pull nvcr.io images"
```

## NGC authentication

```bash
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Persists in `~/.docker/config.json` across reboots. Re-run on `unauthorized` errors.

## Execution — the four verbs

Run a spec-bundle by implementing exactly these four verbs, mutating only the
job-record. Status values are the fixed vocabulary from `tao-artifacts`
(`PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`); native docker states map
below, with the raw state carried in the transition `message`. `$BANK` =
`${TAO_SKILL_BANK_PATH}`.

### submit

1. **Stage** inputs via `tao-data-io`: it picks the storage tier and returns the
   mount args + compute-frame paths. Docker uses **tier A** (bind-mount a host
   dir, `-v /host/data:/data`) as the norm, or **tier C** (pass S3 creds, the
   container fetches). Author the spec file at `<stage>/spec.yaml` with those
   compute-frame paths.
2. **Lint** the assembled command — `redact_secrets.py lint` must pass (no inline
   secrets; pass creds as `-e VAR` with no value).
3. **Open the record — this mints the id and binds `results_dir` BEFORE launch:**
   ```bash
   JOB_ID=$("$BANK/scripts/tao_job_record.py" open \
     --platform docker --image "$IMAGE" \
     --network-arch "$ARCH" --action "$ACTION" \
     --storage-tier "$TIER" --results-root "$RESULTS_ROOT")
   ```
4. **Launch detached**, naming the container after the id so the other verbs find
   it (keep `--rm` OFF so an exited container stays inspectable):
   ```bash
   CID=$(docker run -d --name "$JOB_ID" --label "tao-job=$JOB_ID" \
     --gpus "$GPUS" --ipc=host \
     -v "$STAGE:/workspace" \
     -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e HF_TOKEN -e NGC_KEY \
     "$IMAGE" <bundle command, reading /workspace/spec.yaml>)
   ```
5. **Record RUNNING:**
   `"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING --backend-ref "$CID"`.

A submit that skipped step 3 has no id, so it cannot launch — that is the
record-then-launch invariant.

### status

```bash
read -r st code < <(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$JOB_ID" 2>/dev/null) || st=missing
```

| docker state | vocab |
|---|---|
| `created` / `restarting` | `PENDING` |
| `running` / `paused` | `RUNNING` |
| `exited`, code 0 | `COMPLETE` |
| `exited`, code ≠ 0 | `ERROR` |
| `dead` / missing | `UNKNOWN` (confirm via `docker ps -a`) |

On a terminal state, `mark` it — and for **tier C**, `tao-data-io` uploads
results **before** you `docker rm` (the container is the only copy).

### logs

```bash
docker logs --tail "${N:-200}" "$JOB_ID"    # add -f to follow in-turn
```

### cancel

```bash
docker rm -f "$JOB_ID"
"$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state CANCELED --source agent
```

## Local vs remote (DOCKER_HOST)

There is no separate "remote docker" — point the daemon at an SSH-reachable box:
`export DOCKER_HOST=ssh://user@gpu-host`. Every verb above is **byte-identical**;
the docker CLI marshals the request over SSH (reuses your key, avoids
nested-quoting the command). One consequence: **`-v` bind-mount sources then refer
to the *remote* host's filesystem, not the launcher** — stage data there (tier A)
or fetch in-container (tier C). Fallback for `sudo docker`-only hosts:
`ssh host 'sudo docker …'` (same skill, different prefix).

## `docker run` — canonical flags

```bash
docker run \
  --gpus all \                        # all GPUs (requires nvidia-container-toolkit)
  --rm \                              # delete container after exit (image is preserved)
  --ipc=host \                        # shared mem for torchrun / DataLoader
  -v /host/data:/data \               # bind-mount input
  -v /host/results:/results \         # bind-mount output
  -e HF_TOKEN -e NGC_KEY \            # env-var passthrough (values from parent shell)
  <image> \
  <command>
```

Notes:

- `--gpus '"device=0,1"'` — specific GPUs (double-quote-escaped). Without nvidia-container-toolkit: `could not select device driver "" with capabilities: [[gpu]]`.
- `--rm` — clean up the container at exit; omit when you want `docker logs` after exit.
- `--ipc=host` — torchrun + PyTorch DataLoaders hit shared-memory limits otherwise. Required for multi-GPU training. Alternative: `--shm-size=8g`.
- `-v host:container` — bind mount; the command references container paths only.
- `-e VAR` — passthrough from parent shell (no value needed if already set). Use this form for secrets.

## Container name collision

`docker run --name X` fails if a container named `X` already exists. Defensive pattern before reusing a name:

```bash
docker stop my-worker 2>/dev/null; docker rm my-worker 2>/dev/null
docker run --name my-worker ...
```

## Detached + exec pattern

For multi-step workflows on the same container (download → run → post-process), avoid restart cost:

```bash
docker run -d --name <worker> \
  --gpus all --ipc=host \
  -v <mounts...> -e <envs...> \
  --entrypoint sh \
  <image> -c "tail -f /dev/null"

docker exec <worker> <step_1>
docker exec <worker> <step_2>

docker stop <worker> && docker rm <worker>
```

## Pull-if-missing idiom

```bash
docker image inspect <image> >/dev/null 2>&1 || docker pull <image>
```

## Labels for discovery

Tag containers for filtered listing later:

```bash
docker run --label tao-toolkit ...
docker ps --filter 'label=tao-toolkit'
```

## Mount patterns

The container expects its data at conventional paths defined by the image (often `/data`, `/results`, `/workspace/checkpoints`). The host side is arbitrary. The command inside docker run references container paths only.

## Env-var conventions

Common passthrough vars for TAO-style workloads (the calling skill declares which it needs):

- `NGC_KEY` — `nvcr.io` pulls; some runtimes also read at runtime
- `HF_TOKEN` — gated HuggingFace model downloads
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL` — S3 I/O inside the container
- `WANDB_API_KEY` — optional W&B logging

Use `-e VAR` (no `=value`) when the var is in the parent shell. Avoid placing secrets on the command line.

Alternative GPU selection: `-e NVIDIA_VISIBLE_DEVICES=0,1` (or `all`) and `-e NVIDIA_DRIVER_CAPABILITIES=all` instead of `--gpus`. The `--gpus` flag is preferred on standard x86 hosts; the env-var form is older and is what `runtime=nvidia` (Tegra/Jetson) requires.

## Container inspection

```bash
docker ps                                # running containers only
docker ps -a                             # all containers, including exited
docker ps --filter status=running --format '{{.Names}} {{.Image}}'
docker logs <name_or_id>                 # stdout/stderr
docker logs -f <name_or_id>              # follow (tail -f equivalent)
docker logs --tail 100 <name_or_id>      # last N lines
docker inspect <name_or_id>              # full config, mounts, env, network, state (JSON)
docker inspect --format '{{.State.Status}}' <name_or_id>
docker stats                             # live CPU/mem/network/block I/O
docker stats --no-stream                 # one snapshot, non-interactive
```

`docker inspect` is the canonical source of truth for a container's mounts, env, cmd, network, and exit code. Use it to debug why a container isn't behaving as expected.

## Image management

```bash
docker pull <image>
docker image ls
docker system df                # disk usage
docker system prune -a --volumes # reclaim space — destructive, removes unused images + volumes
```

Pull once per host; `docker run` reuses cached image. NVIDIA images are typically 5-40GB.

## Split-disk data-root relocation

Some cloud GPU providers ship with a small root volume + larger ephemeral. Docker writes to `/var/lib/docker` on root by default — large images fill it. Check:

```bash
df -h /         # root volume size/free
lsblk           # all block devices and mount points
```

If `/` is smaller than your total image footprint and there's a larger disk mounted elsewhere, relocate **before pulling images**:

```bash
sudo systemctl stop docker
sudo mkdir -p <large_volume_path>/docker
sudo rsync -aP /var/lib/docker/ <large_volume_path>/docker/
sudo mv /var/lib/docker /var/lib/docker.old

sudo tee /etc/docker/daemon.json <<'EOF'
{ "data-root": "<large_volume_path>/docker" }
EOF

sudo systemctl start docker
docker info | grep 'Docker Root Dir'
sudo rm -rf /var/lib/docker.old
```

## Networks (multi-container patterns)

For microservice containers that talk to each other by name, create a docker network and attach containers:

```bash
docker network create tao-net
docker run --network tao-net --name api ...
docker run --network tao-net --name worker ...   # can resolve `api` by name
```

Most TAO training workloads don't need this — single container per job.

## Common error modes

**`could not select device driver "" with capabilities: [[gpu]]`** — NVIDIA Container Toolkit missing or Docker is not configured for the NVIDIA runtime. Run `tao-setup-nvidia-gpu-host` with `--backend docker --install` after user approval (append `--yes` for a non-interactive agent run), then restart Docker.

**`unauthorized: authentication required`** on `docker pull` — NGC key invalid/missing. Re-run `docker login nvcr.io`.

**`no space left on device`** — root volume full. `docker system df` to inspect; relocate `data-root` (above) or `docker system prune -a --volumes`.

**`Bus error` / `DataLoader worker exited unexpectedly`** — `/dev/shm` too small. Add `--ipc=host` or `--shm-size=8g`.

**`permission denied` on bind-mounted paths** — container UID ≠ host UID. Either `-u $(id -u):$(id -g)`, or pre-create host files owned by the host user, or `chmod 777` (dev only).

**`Error: No such container: <name>` after `docker run -d`** — container crashed on startup. `docker ps -a` shows exited; `docker logs <name>` for cause. Drop `--rm` while debugging.

## Scope boundary

This skill both **runs** TAO jobs on Docker (§ Execution) and documents the docker
*how* that other skills defer to. Related:

- `tao-skill-bank:tao-run-on-brev` — provisions a Brev instance, then defers the
  container-how to these same docker verbs.
- `tao-skill-bank:tao-launch-workflow` — the intake/routing front door and the
  platform-agnostic four-verb contract this skill implements.
- `tao-skill-bank:tao-data-io` — the storage-tier decision + staging + the
  compute-frame verify gate the `submit` verb calls.

Model and data skills produce the spec-bundle (**what**); this skill runs it (**how**).
