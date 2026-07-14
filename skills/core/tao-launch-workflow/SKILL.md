---
name: tao-launch-workflow
description: >-
  Shared launch intake for any TAO workflow or action. Use when the user wants
  to run TAO AutoML, train, evaluate, infer, export, generate TensorRT engines,
  or launch DEFT/workflow jobs on an execution platform.
license: Apache-2.0
compatibility: Requires the packaged TAO skill bank helper scripts.
metadata:
  author: NVIDIA Corporation
  version: "0.1.0"
allowed-tools: Read Bash
tags:
- tao
- workflow
- launch
---

# TAO Workflow Launch Intake

> **Standalone install?** If this session was not initialized by the TAO skill bank plugin, run the `tao-setup` skill first (host preflight, credentials, cross-skill discovery).

Use this skill before launching any TAO workflow or model action.

## Quick Start

Run the platform helper, ask for platform and monitoring preferences, then run
the selected platform detail helper before asking for credentials.

## Non-Negotiable Launch Gate

This gate is model-agnostic. Apply it to every TAO model, data action, and
application workflow before launching side-effecting work.

Do **not** create runner scripts, launch scripts, compatibility shims,
workspace folders, state files, logs, or dependency-install side effects until
the launch preflight passes.

Preflight passes only after all of these are true:

1. The execution platform is selected from the packaged platform helper.
2. Platform credentials and required credential groups are satisfied.
3. Model-specific credentials are satisfied.
4. The default container image is resolved from packaged model/action metadata,
   shown to the user, and either confirmed or replaced by an explicit
   `image=<override>`.
5. The platform access check succeeds from the launch host.
6. Dataset inputs are mapped to concrete spec keys and verified from the
   selected platform's point of view.
7. Required compute shape fields from the model/workflow skill are known.
8. Required local tools for the selected data/platform path are present, or the
   user approved installing the smallest missing dependency and preflight was
   rerun.
9. A launch review with image, platform, datasets, compute shape, expected
   runtime, and any generated/default configuration changes has been shown and
   confirmed by the user. For AutoML, the launch review must explicitly state
   recommendation count/budget, max concurrency, algorithm, metric, direction,
   and searched parameters/ranges even when defaults are used.

If any item is missing, ask for the missing input and stop before generating
artifacts. This applies to AutoML, normal train/eval/infer/export/TRT, and
DEFT/application workflows.

When preflight work clears a blocker, keep track of the original user request.
After the fix, rerun the relevant preflight and continue toward that request;
do not stop at "blocker fixed" unless the user explicitly asked only for the
repair.

## The Four-Verb Execution Contract

Once the launch gate passes and the producing model/data skill has authored the
spec-bundle (schema: `tao-artifacts`), execution is exactly four verbs. Every
platform skill (`tao-run-on-docker`, `-slurm`, `-kubernetes`, `-brev`)
implements them over its native CLI; nothing else is platform-specific.
`$BANK` = `${TAO_SKILL_BANK_PATH}`.

- **submit(spec-bundle)** — stage inputs via `tao-data-io` (it picks the storage
  tier and returns compute-frame paths), lint the assembled command with
  `redact_secrets.py lint`, then **open the record and launch, in that order**:
  ```bash
  JOB_ID=$("$BANK/scripts/tao_job_record.py" open --platform <p> --image <img> \
    --network-arch <arch> --action <action> --storage-tier <A|B|C> --results-root <root>)
  # <native launch, naming the backend object after $JOB_ID>
  "$BANK/scripts/tao_job_record.py" mark "$JOB_ID" --state RUNNING --backend-ref <ref>
  ```
- **status(id)** — poll the native backend, map to the fixed vocabulary
  `PENDING RUNNING COMPLETE ERROR CANCELED UNKNOWN`; the native sub-state
  (`ImagePullBackOff`, `PENDING`-resources, slurm `COMPLETING`) rides in the
  transition `message`. Never read "what's running" from records — poll the backend.
- **logs(id, tail)** — native log fetch.
- **cancel(id)** — native cancel + orphan teardown, then `mark <id> --state CANCELED`.

**Record-then-launch is the ordering invariant.** `open` mints the id and binds
`results_dir` *before* any launch, and the id it returns is the only handle the
launch can use — a submit that skipped the gate or the open has no id, so it
cannot launch. This is what keeps a run recoverable across a context break:
`results_dir` is recorded before the backend object (which K8s TTL or docker
`--rm` may later delete) ever exists.

### Failure analysis & retry

When `status` reaches `ERROR`, **read the tail of the logs and classify the
failure before deciding to retry** — never blindly resubmit (a bad spec would
burn the whole retry budget on GPU-hours). This is a judgment you make from the
log; the signals below are guidance, not an exhaustive table.

- **Infrastructure → retriable** (a node / hardware / transport fault a resubmit
  may escape): SLURM `NODE_FAIL` / `BOOT_FAIL` / preemption; GPU `Xid` / `ECC` /
  "fell off the bus"; driver-too-old; genuine `NCCL` / InfiniBand / RDMA transport
  failures; `ImagePullBackOff`; pod `Evicted`. Resubmit under a NEW job-record
  with `--retry-of <id>` (mark the old one `ERROR --err-class ERR_INFRA`), up to
  **10**. SLURM's `#SBATCH --requeue` handles node-fault/preemption at the
  scheduler level for free.
- **Program → never retry** (a code / data / config fault a resubmit will just
  repeat): `OOMKilled` / CUDA out-of-memory (reduce batch); CUDA **device-side
  assert** / illegal-memory-access / CUBLAS/CUDNN status (a code or label-range
  bug); missing files; Python tracebacks; timeout / deadline (raise the limit).
  Surface the cause to the user.

Two context calls to apply with judgment: a **device-side assert** is a program
bug *even when it cascades into a CUDA/NCCL error* — the assert is the root cause;
and a bare traceback that is *downstream* of a real `Xid` / node fault is a
symptom — retry the node fault. (These are exactly what a regex table gets wrong,
which is why classification lives here as agent judgment, not in a script.)

**In-turn** monitoring is the agent polling `status` / `logs` at the interval.
**Post-turn** (the turn may end before a long job finishes): background a poller
with the harness (`run_in_background` wait-loop, `CronCreate`, or `/loop`). The
poller auto-resubmits only the **unambiguous state-based** infra cases
(`NODE_FAIL` / `BOOT_FAIL` / `PREEMPTED`) and, for auto-cleanup backends (K8s
`ttlSecondsAfterFinished`, docker `--rm`), writes the **daemon-independent
terminal record** (`tao_job_record mark --state <terminal> --source poller`)
*before* the backend object is deleted; anything nuanced re-wakes the agent to
classify. Pollers are idempotent — on re-attach, just re-establish one.

## Initial Questions

After the user confirms what they want to do, ask which **execution platform**
should run it. Discover the choices from the **platform skills installed in this
session** — you already see them by name and description (`tao-run-on-docker`,
`-slurm`, `-kubernetes`, `-brev`, plus any externally installed one such as the
official `brev-cli` skill). There is no central platform registry to read. If
your runtime surfaces only the core router skills (e.g. Codex), list the bank's
platform skills by reading `skills/platform/tao-run-on-*/SKILL.md` frontmatter
(name + one-line description) under `${TAO_SKILL_BANK_PATH}`.

Then ask:

- Which supported platform should run this workflow?
- Should I monitor the run in this chat? Monitoring means I keep polling the
  backend/job logs after launch and report progress until the job finishes,
  fails, or you ask me to stop, even if the job stays queued for hours or days.
  If disabled, I launch the job, give you the job id/log path, and stop
  polling. Default: monitor in chat.
- How often should I post status? Default: every 5 minutes. Use 1-2 minutes for
  smoke tests, 5 minutes for normal training, or 10-15 minutes for long runs.

Use `long_running_enabled=true` and `status_interval_minutes=5` when the user
accepts the defaults.

When monitoring is enabled, do not send a final summary just because several
polls have elapsed or the job is still `PENDING`. Keep the turn attached and
emit status every `status_interval_minutes` until a terminal state or explicit
user stop/detach request. If the runtime environment cannot keep the chat turn
open, say that clearly and leave a durable watcher/log path; do not imply that
chat updates will continue after the turn ends.

Final-answer rule: a `final` response ends chat-side monitoring. While
`long_running_enabled=true` and any launched job is non-terminal, status
messages must be sent as in-progress updates and the agent must continue
polling. Only send a final response when the workflow reaches terminal state,
the user explicitly asks to detach/stop monitoring, or the runtime genuinely
cannot keep the turn open; in that last case, say it is a runtime limitation
and provide the exact durable status command/log path.

## Missing-Input Prompt Shape

When asking for launch inputs, include concrete examples and both dataset input
modes. Do not ask only for "dataset root".

Use this structure and adapt spec keys to the selected model/action:

```text
I need these launch inputs before I can create specs or runner files:

1. Execution platform: brev, slurm, local-docker, or kubernetes.

2. Dataset inputs. You can provide either mode:
   A) Root mode: give train/eval roots and I map required files automatically.
      Example Cosmos-RL:
      train_root=/lustre/fsw/.../cosmos/train
      -> custom.train_dataset.annotation_path=train_root/annotations.json
      -> custom.train_dataset.media_path=train_root
   B) Direct spec mode: give the exact config/spec parameters yourself.
      Example:
      custom.train_dataset.annotation_path=/lustre/fsw/.../train_annotations.json
      custom.train_dataset.media_path=/lustre/fsw/.../videos_train.tar.gz
      custom.val_dataset.annotation_path=/lustre/fsw/.../eval_annotations.json
      custom.val_dataset.media_path=/lustre/fsw/.../eval_videos/

   Platform examples:
   - SLURM/Lustre: /lustre/fsw/.../data/train or lustre:///lustre/fsw/.../data/train
   - Brev/Kubernetes: s3://bucket/path/train and s3://bucket/path/eval
   - local-docker: /data/tao/<model>/train or file:///data/tao/<model>/eval

3. Container image. I will resolve the default from packaged model metadata and
   show it before launch, for example:
   default image for <model>/<action>: <resolved container image>
   Use this image, or provide image=<override> to pin a different TAO build.

4. Compute shape required by the model, for example GPUs/nodes.

5. Required credentials from platform/model docs, for example HF_TOKEN for
   gated Hugging Face models.

6. Monitoring preference. By default I monitor in this chat and post progress
   every 5 minutes; choose 1-2 minutes for smoke tests or 10-15 minutes for
   long training.
```

## Container Image Confirmation

Before creating specs, runner scripts, workspaces, logs, state files, or
submitting a job, resolve the image for the selected model/action:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/resolve_tao_image.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --model <network> --action <action> --format text
```

If the helper is unavailable, read `skills/models/<network>/config.json`
directly. Resolve image fields in this order:

1. `actions.<action>.container_image`
2. `actions.<action>.image`
3. top-level `container_image`
4. top-level `image`

Show the exact image and ask:

```text
Container image for <network>/<action>:
default=<resolved image>

Use this image, or provide image=<override>?
```

If the user accepts, pass the resolved image as the job `image`. If the user
overrides, require a non-empty image reference and pass that value instead.
Do not silently launch on the default image. This confirmation applies to
training, AutoML recommendations, evaluation, inference, export, TensorRT
engine generation, and application workflows that submit TAO containers.

## Credential Filtering

After the user chooses a platform, get the credential list for **only that
platform** from the chosen skill itself — its `## Credentials` section and, if
present, `references/skill_info.yaml` (`required_credentials`, `credential_groups`,
`optional_credentials`). The launch preflight (`check_tao_launch_preflight.py`)
reads that same per-skill `skill_info.yaml` to enforce the credential gate; a
credential-free platform (e.g. Docker) may ship only prose, in which case rely on
its Preflight section.

Ask only for credentials that platform actually needs, plus model-specific
credentials from the selected model skill. Do not ask for Brev credentials on
SLURM, Kubernetes, or Docker. Do not ask for SLURM credentials on Brev,
Kubernetes, or Docker. Ask S3 credentials only when the selected
platform and the dataset/result URIs require `s3://` access.
Credentials may already be present in the process environment or in a
user-approved secret env file such as `~/.tao/secrets.env` or
`~/.config/tao/.env`; source such files only when needed and never print,
grep, cat, paste, or log their contents. Verify only variable presence.

For initial launch intake, ask for required credentials and required credential
groups only. Treat the helper's optional credentials/settings section as
reference material; do not request those values unless their `only_when`
condition applies, the selected workflow cannot proceed without them, or the
user asks to customize that setting.

When the helper output includes a "Required credential groups" section, satisfy
one credential from each group before proceeding. Explain each requested value
using the helper's description and "How to get it" text.

For SLURM, user-facing prompts should ask for `SSH_KEY_PATH` first. Mention
`SSH_AUTH_SOCK` only if the user says they already use an SSH agent.

## Dependency Remediation

If a required CLI/library is missing, say exactly what is missing and why it is
needed, then ask before installing. Examples:

- S3 dataset or results path -> require an S3-capable client such as `aws`.
- Local Docker path -> require the Docker CLI and the configured Docker
  network.

After user approval and installation, rerun the same preflight. Do not create
runner files or launch jobs between the failed check and the rerun.

## Dataset Intake

Accept dataset inputs in either mode:

- **Dataset root mode:** the user gives train/eval/calibration roots, and the
  model skill maps required files by convention. Example for Cosmos-RL train:
  `custom.train_dataset.annotation_path=<root>/annotations.json` and
  `custom.train_dataset.media_path=<root>`.
- **Direct spec mode:** the user gives exact spec-key paths when annotations,
  media archives, videos, or image folders live in different places. Preserve
  those keys directly, for example
  `custom.train_dataset.annotation_path=/lustre/.../train_annotations.json`
  and `custom.train_dataset.media_path=/lustre/.../videos.tar.gz`.

Ask for dataset examples that match the selected platform:

- SLURM: shared cluster paths such as
  `/lustre/fsw/portfolios/<team>/<your-dir>/data/<model>/train` (where
  `<your-dir>` is your per-user directory on the cluster), or direct
  spec paths under `/lustre/...`.
- Brev, Kubernetes: usually `s3://bucket/path/train` and
  `s3://bucket/path/eval` unless the platform profile mounts shared storage.
- Local Docker: local paths visible to the Docker host, such as
  `/data/tao/<model>/train`, or direct spec paths visible inside the planned
  container mount.
- Remote Docker: absolute paths visible on the remote Docker host named by
  `DOCKER_HOST`, not paths on the local agent machine.

Do not assume "dataset root" is the only acceptable input. When direct spec
paths are supplied, validate the exact spec paths rather than appending default
filenames.

## Platform Preflight

Run the selected platform's preflight checks before any launch artifact is
created — prefer the packaged helper `scripts/check_tao_launch_preflight.py`
(`--platform <p> --container-image <img> --path <label>=<path> ...`). It verifies
credentials, client tools, platform/cluster/object-store access, dataset paths
from the compute frame, GPU/runtime health, and image-architecture fit; treat any
failure as blocking. Never use `--skip-platform-access` for a real launch.

See `references/platform-preflight.md` for the full per-platform detail (SLURM
SSH/key setup + resource defaults, docker/remote-docker GPU + bind-mount checks,
Brev/Kubernetes API + object-store checks, annotation content-field checks, and
data staging).

## Runtime And Configuration Review

Before any side-effecting launch, show a concise review:

- selected platform and exact container image
- GPU ids/count and nodes, including any GPUs avoided because they are already
  occupied
- dataset roots or direct spec paths, with sample counts when available
- important model/workflow overrides that differ from template defaults
- estimated runtime and the assumptions behind it
- monitoring interval and whether chat-side monitoring will stay attached

For AutoML, also show the algorithm, metric/direction, recommendation budget,
search parameters, ranges, and generated/default recommendation details as
described in `skills/applications/tao-run-automl/SKILL.md`. Ask for confirmation after
this review. If the user supplied a time limit, flag any plan that exceeds it
and offer concrete reductions before launch.
