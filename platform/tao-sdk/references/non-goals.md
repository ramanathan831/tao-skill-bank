# TAO SDK Non-Goals

Read this when deciding whether an agent or the SDK owns a piece of behavior.

## Contents

- Skill interpretation ownership.
- AutoML ownership.
- Spec construction ownership.
- Platform selection ownership.
- Multi-step workflow ownership.

## What The SDK Does Not Do

- It does not read or interpret skills. The agent reads `SKILL.md` and `references/skill_info.yaml`; the SDK submits the command the agent constructs.
- It does not do hyperparameter optimization by itself. The agent owns the model-level AutoML policy: when model metadata has `automl_enabled: true`, use `applications/tao-automl` unless the workflow passes `automl_policy: off` or the user explicitly asks for a plain single training run.
- It does not decide what goes in the spec. The agent constructs the spec dict by loading templates and applying overrides, then passes it to `build_entrypoint`.
- It does not select platforms automatically. Pick the SDK matching the target backend explicitly: `LeptonSDK`, `BrevSDK`, `DockerSDK`, `SlurmSDK`, or `KubernetesSDK`.
- It does not orchestrate multi-step workflows. The agent chains jobs by polling and constructing the next command.
