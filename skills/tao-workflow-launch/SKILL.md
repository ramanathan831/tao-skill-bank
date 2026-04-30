---
name: tao-workflow-launch
description: >-
  Shared launch intake for any TAO workflow or action. Use when the user wants
  to run TAO AutoML, train, evaluate, infer, export, generate TensorRT engines,
  or launch DEFT/workflow jobs on an execution platform.
---

# TAO Workflow Launch Intake

Use this skill before launching any TAO workflow or model action.

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
