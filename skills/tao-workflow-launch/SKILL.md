---
name: tao-workflow-launch
description: >-
  Shared launch intake for any TAO workflow or action. Use when the user wants
  to run TAO AutoML, train, evaluate, infer, export, generate TensorRT engines,
  or launch DEFT/workflow jobs on an execution platform.
---

# TAO Workflow Launch Intake

Use this skill before launching any TAO workflow or model action.

## Non-Negotiable Launch Gate

Do **not** create runner scripts, launch scripts, compatibility shims,
workspace folders, state files, logs, or dependency-install side effects until
the launch preflight passes.

Preflight passes only after all of these are true:

1. The execution platform is selected from the packaged platform helper.
2. Platform credentials and required credential groups are satisfied.
3. Model-specific credentials are satisfied.
4. The platform access check succeeds from the launch host.
5. Dataset inputs are mapped to concrete spec keys and verified from the
   selected platform's point of view.
6. Required compute shape fields from the model/workflow skill are known.

If any item is missing, ask for the missing input and stop before generating
artifacts. This applies to AutoML, normal train/eval/infer/export/TRT, and
DEFT/application workflows.

## Initial Questions

After the user confirms what they want to do, ask for the execution platform
using the packaged helper. Do not scan platform docs, skill folders, or config
folders to build the choices.

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} --format text
```

Then ask:

- Which supported platform should run this workflow?
- Should I monitor the run in this chat? Monitoring means I keep polling the
  backend/job logs after launch and report progress until the job finishes,
  fails, or you ask me to stop. If disabled, I launch the job, give you the job
  id/log path, and stop polling. Default: monitor in chat.
- How often should I post status? Default: every 5 minutes. Use 1-2 minutes for
  smoke tests, 5 minutes for normal training, or 10-15 minutes for long runs.

Use `long_running_enabled=true` and `status_interval_minutes=5` when the user
accepts the defaults.

## Missing-Input Prompt Shape

When asking for launch inputs, include concrete examples and both dataset input
modes. Do not ask only for "dataset root".

Use this structure and adapt spec keys to the selected model/action:

```text
I need these launch inputs before I can create specs or runner files:

1. Execution platform: lepton, brev, slurm, local-docker, or kubernetes.

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
   - Lepton/Brev/Kubernetes: s3://bucket/path/train and s3://bucket/path/eval
   - local-docker: /data/tao/<model>/train or file:///data/tao/<model>/eval

3. Compute shape required by the model, for example GPUs/nodes.

4. Required credentials from platform/model docs, for example HF_TOKEN for
   gated Hugging Face models.

5. Monitoring preference. By default I monitor in this chat and post progress
   every 5 minutes; choose 1-2 minutes for smoke tests or 10-15 minutes for
   long training.
```

## Credential Filtering

After the user chooses a platform, get the credential list for only that
platform:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/list_tao_platforms.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> --format text
```

Ask only for credentials returned by that command, plus model-specific
credentials from the selected model skill. Do not ask for Lepton credentials on
SLURM, Kubernetes, or local Docker. Do not ask for SLURM credentials on Lepton,
Brev, Kubernetes, or local Docker. Ask S3 credentials only when the selected
platform and the dataset/result URIs require `s3://` access.

When the helper output includes a "Required credential groups" section, satisfy
one credential from each group before proceeding. Explain each requested value
using the helper's description and "How to get it" text.

For SLURM, user-facing prompts should ask for `SSH_KEY_PATH` first. Mention
`SSH_AUTH_SOCK` only if the user says they already use an SSH agent.

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
  `/lustre/fsw/portfolios/<team>/users/<user>/data/<model>/train`, or direct
  spec paths under `/lustre/...`.
- Lepton, Brev, Kubernetes: usually `s3://bucket/path/train` and
  `s3://bucket/path/eval` unless the platform profile mounts shared storage.
- Local Docker: local paths visible to the Docker host, such as
  `/data/tao/<model>/train`, or direct spec paths visible inside the planned
  container mount.

Do not assume "dataset root" is the only acceptable input. When direct spec
paths are supplied, validate the exact spec paths rather than appending default
filenames.

## Platform Preflight

Run the selected platform's preflight checks before any launch artifact is
created.

Prefer the packaged preflight helper when the needed inputs are available:

```bash
${TAO_SKILL_BANK_PATH:-~/tao-skills-external}/scripts/check_tao_launch_preflight.py \
  --skill-bank ${TAO_SKILL_BANK_PATH:-~/tao-skills-external} \
  --platform <platform> \
  --path train_annotation=<path> \
  --path train_media=<path>
```

Pass exact direct spec paths when the user supplied them. For root-mode inputs,
expand model-required files first, then pass those concrete annotation/media
paths to the helper.

Do not use `--skip-platform-access` for a real launch. That flag is only for
dry environment checks or for cases where the user has already provided explicit
manual proof of platform and storage access. If the helper cannot verify remote
API, CLI, cluster, or object-store access, treat preflight as failed and do not
generate launch artifacts.

For SLURM:

1. Require `SLURM_USER`, `SLURM_HOSTNAME`, and one of `SSH_KEY_PATH` or
   `SSH_AUTH_SOCK`.
2. Split comma-separated `SLURM_HOSTNAME`, resolve hosts where possible, and
   require passwordless `ssh -o BatchMode=yes` to at least one host.
3. If SSH fails, do not offer several equivalent choices. Ask for
   `SSH_KEY_PATH=/path/to/private_key` and show the passwordless setup steps:
   create a key if needed with
   `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519`; install it with
   `ssh-copy-id -i ~/.ssh/id_ed25519.pub <SLURM_USER>@<login-host>`; trust the
   host with `ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts`; set
   `chmod 600 ~/.ssh/id_ed25519`; verify with
   `ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 <SLURM_USER>@<login-host> 'hostname'`;
   then rerun with `SSH_KEY_PATH=~/.ssh/id_ed25519`.
4. After SSH passes, validate dataset annotation/media paths on the remote login
   host with `test -e` or an equivalent read-only command.
5. Only then create runner scripts, specs, workspaces, or submit jobs.

For local Docker, validate Docker/GPU access and local dataset paths before
writing launch artifacts. For Lepton, Brev, and Kubernetes, validate API or
cluster access plus object-storage credentials and `aws s3 ls` readability for
`s3://` inputs before writing launch artifacts. For mounted shared-storage or
PVC paths on those remote platforms, require manual proof that the path is
mounted into the job environment; the helper fails closed rather than accepting
unverified remote mount paths.
