# TAO-Next Skill Bank

Consolidated skill bank for the TAO-Next agent-driven ML training platform. Skills are structured, discoverable units with explicit scope, inputs, actions, and execution references. The agent uses these skills to move from high-level intent to concrete runnable sandbox code without hard-coding per-workflow logic.

Based on the TAO-Next Skills Taxonomy proposal.

## Taxonomy Overview

The taxonomy is organized around the layers of the TAO-Next stack. Each layer has its own purpose while depending on the layers beneath it.

| Layer | Primary Purpose | Examples | Depends On |
|---|---|---|---|
| **Applications** | Publish a complete workflow or product/use-case capability | DEFT (SDA), HALP, AutoML, AOI | Models, Data, Platform |
| **Models** | Publish flattened network-centric skills and action knowledge | Cosmos-RL, Visual ChangeNet, CLIP, AnomalyGen | Platform |
| **Data** | Publish data acquisition, preparation, analysis, and enhancement capabilities | SigLIP Embed, k-NN Mining, Qwen Caption, Pseudolabelling | Platform |
| **Optimization** | Publish model-agnostic optimization skills (latency vs accuracy) | Quantization, Pruning, Distillation | Models, Platform |
| **Deployment** | Publish inference serving runtime skills | GroundingDINO serving, VLM serving | Platform, Models |
| **Platform** | Publish where and how GPU jobs run | Available platforms, runner interface (run/status/cancel) | Independent base layer |

> **Note:** Optimization and Deployment layers are defined in the taxonomy but not yet populated in this repo. They will be added as skills are developed.

## Directory Structure

```
tao-skill-external/
  applications/              # Workflow and use-case skills
    deft-cosmos-rl/          #   DEFT iterative data improvement for video QA
    sda-vcn/                 #   Mining-based SDA for visual change detection
    normal-train/            #   Standard single-step train/eval/export
  models/                    # Network-centric skills
    cosmos-rl/               #   Cosmos-Reason2-8B video QA SFT
    visual-changenet/        #   Binary classification + segmentation for AOI
    clip/                    #   CLIP vision-language model
    cosmos-predict-2-5/      #   Text-to-video generation
    anomalygen/              #   Cosmos AnomalyGen synthetic defect generation
  data/                      # Data processing skills
    siglip-embed/            #   SigLIP image/video embeddings
    knn-mining/              #   k-NN similarity mining (cuML)
    qwen-caption/            #   VLM captioning via Qwen endpoint
    nim-embedding/           #   Video embeddings via NIM endpoint
    changenet-data-prepare/  #   CSV generation for VCN training
  platform/                  # Compute backends
    lepton/                  #   DGX Cloud Lepton managed GPU compute
```

## Skill Layers

### Applications

Application skills represent the highest layer of abstraction. They package a full product or workflow view while explicitly depending on referenced model, data, and platform skills rather than duplicating their execution logic.

Application skills define `init` and `iteration` stages in their `config.json`, with stage dependencies and conditional execution. Each stage references a model or data skill by name, or runs an inline script.

Examples:
- **deft-cosmos-rl**: DEFT pipeline for video QA — 10-stage iterative loop (gap analysis, captioning, video generation, data merge, training, evaluation)
- **sda-vcn**: Mining-based SDA for AOI — embedding, k-NN search, merge, retrain loop

The DEFT AOI use case shows how an application skill decomposes into domain-specific stages: data mining for retrieval, anomaly generation using AnomalyGen/Cosmos Predict 2.5, gap analysis, data enhancement, fine-tuning, and loop-back evaluation.

### Models

Model skills are flattened into network-centric skills. The agent discovers a model as a named network and then learns the operational details needed to run actions on that network.

Each model skill publishes:
- Container URI or runnable image reference
- Checkpoint references
- Specification templates or schemas (via `defaults-{action}.json`)
- Action-specific configuration (train, evaluate, inference, export)
- Supported data formats
- Platform requirements

The model layer is not only executable — it is also **recommendatory**. It helps the agent answer: which network is best suited for a given data type and purpose.

Examples:
- **cosmos-rl**: Cosmos-Reason2-8B video QA SFT with FSDP parallelism
- **visual-changenet**: Siamese classification + pixel-level segmentation for AOI
- **anomalygen**: Cosmos AnomalyGen synthetic defect image generation

### Data

Data skills capture capabilities applied to datasets rather than networks — preparation, analysis, enhancement, and transformation operations.

Application skills depend heavily on data skills. For example, the AOI flow uses data mining (k-NN retrieval), anomaly generation, embedding, and CSV preparation to improve coverage before fine-tuning begins.

Examples:
- **siglip-embed**: SigLIP image embeddings for similarity search
- **knn-mining**: GPU-accelerated k-NN nearest neighbor mining
- **changenet-data-prepare**: CSV generation with filename normalization for VCN

### Platform

Platform skills form the lowest layer and serve as the execution foundation. This layer has two responsibilities:

1. **Platform discovery** — publish available compute platforms so the agent knows where jobs can run
2. **Runner interface** — publish a common interface for long-running GPU jobs: run, status, cancel

For normal model training, the agent uses the selected model skill to assemble the network-specific execution contract and uses the platform skill to launch the job through the common runner.

## Skill Publication Contract

Each skill must provide enough metadata to function as a discoverable, composable contract.

| Contract Field | Purpose |
|---|---|
| Name and hierarchy path | Identity, discoverability, and placement in the taxonomy |
| Purpose and scope | What the skill is for and what it should not be used for |
| Inputs and outputs | Accepted inputs, produced outputs, and data formats |
| Actions | Operations the agent can request from the skill |
| Execution references | Container URIs, checkpoints, schemas, configs |
| Platform requirements | Runner constraints, GPU expectations, environment needs |
| Troubleshooting guide | Expert knowledge, error patterns, and behavior notes |

In this repo, the contract is split across two files:

- **`config.json`** — structured metadata: actions, inputs/outputs, execution references, credentials, data sources
- **`{skill-name}.md`** — agent-readable documentation: purpose/scope, data formats, parameters, troubleshooting

## Skill Package Format

```
{skill-name}/
  config.json              # Structured config (contract fields above)
  {skill-name}.md          # Agent documentation (plain markdown, no frontmatter)
  defaults-{action}.json   # Per-action default spec values (JSON only)
  scripts/                 # Inline scripts for workflow stages (optional)
```

### defaults-{action}.json

Complete default spec for each action. The planner loads these and applies user overrides on top. All defaults are JSON format regardless of what the container expects — the script runner converts to the container's native format (YAML/TOML/JSON) at runtime.

## Execution Patterns

The taxonomy supports distinct execution patterns depending on whether the user request maps to a known network action or a higher-level application workflow.

| Scenario | Skills Composed by Agent | Result |
|---|---|---|
| Normal training or inference on a known network | Model skill + Platform skill | Agent generates sandbox code using the network's container, schemas, configs, command line, env vars, and platform requirements |
| High-level application workflow (e.g., AutoML) | Application skill + referenced skills + Platform skill | Agent generates sandbox code from high-level workflow knowledge while grounding execution in the platform layer |
| Application use case (e.g., DEFT AOI) | Application skill + referenced Model/Data skills + Platform skill | Agent decomposes the use case, selects required sub-skills, and orchestrates the closed-loop workflow |

## Skill Discovery

The TAO-Next planner discovers skills via the `SkillBank` class:

1. Model skills are searched in both `models/` and `data/` directories
2. Workflow skills are loaded from `applications/`
3. Platform configs are loaded from `platform/`
4. Skill names use hyphen-case (`cosmos-rl`, not `cosmos_rl`)

The `TAO_SKILL_BANK_PATH` environment variable overrides the default discovery path.

## Design Rules

- Keep application skills workflow-centric — avoid embedding low-level runtime duplication when a referenced model or data skill already exists.
- Keep model skills flattened at the network level so the agent can reason cleanly about network choice and action contracts.
- Treat the platform layer as the base execution contract for all GPU jobs.
- Publish enough execution metadata for code synthesis, not just narrative descriptions.
- Use explicit references between skills so composition remains deterministic and explainable.

## Adding a New Skill

1. Choose the correct layer: `models/` for trainable networks, `data/` for data processing, `applications/` for workflows, `platform/` for compute backends
2. Create a directory: `{layer}/{skill-name}/`
3. Add `config.json` with actions, inputs, outputs, credentials, execution references
4. Add `{skill-name}.md` with agent documentation (purpose, data formats, parameters, error patterns)
5. Add `defaults-{action}.json` for each action with default spec values
6. If the skill has inline scripts, add them under `scripts/`
7. Test: verify `SkillBank().get_model_config('{skill-name}')` returns your config
