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
- Should long-running monitoring stay enabled? If enabled, keep the agent
  attached and emit status until terminal completion. Default: enabled.
- How many minutes between status updates? Default: 5 minutes.

Use `long_running_enabled=true` and `status_interval_minutes=5` when the user
accepts the defaults.

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

For SLURM:

1. Require `SLURM_USER`, `SLURM_HOSTNAME`, and one of `SSH_KEY_PATH` or
   `SSH_AUTH_SOCK`.
2. Split comma-separated `SLURM_HOSTNAME`, resolve hosts where possible, and
   require passwordless `ssh -o BatchMode=yes` to at least one host.
3. If SSH fails, stop and tell the user to install the public key with
   `ssh-copy-id`, fix key permissions with `chmod 600`, trust the host key with
   an interactive login or `ssh-keyscan`, or start `ssh-agent` and expose
   `SSH_AUTH_SOCK`.
4. After SSH passes, validate dataset annotation/media paths on the remote login
   host with `test -e` or an equivalent read-only command.
5. Only then create runner scripts, specs, workspaces, or submit jobs.

For local Docker, validate Docker/GPU access and local dataset paths before
writing launch artifacts. For Lepton, Brev, and Kubernetes, validate API or
cluster access plus object-storage credentials for `s3://` inputs before writing
launch artifacts.
