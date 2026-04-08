# TAO-Next Skill Bank

Consolidated skill bank for the TAO-Next agent-driven ML training platform. Skills are organized into four layers following the TAO-Next Skills Taxonomy.

## Directory Structure

```
tao-skill-external/
  applications/          # High-level workflows (multi-step pipelines)
    deft-cosmos-rl/      #   DEFT iterative data improvement for video QA
    sda-vcn/             #   Mining-based SDA for visual change detection
    normal-train/        #   Standard single-step train/eval/export
  models/                # Trainable networks
    cosmos-rl/           #   Cosmos-Reason2-8B video QA SFT
    visual-changenet/    #   Binary image classification + segmentation
    clip/                #   CLIP vision-language model
    cosmos-predict-2-5/  #   Text-to-video generation
  data/                  # Data processing skills
    siglip-embed/        #   SigLIP image/video embeddings
    knn-mining/          #   k-NN similarity mining (cuML)
    qwen-caption/        #   VLM captioning via Qwen endpoint
    nim-embedding/       #   Video embeddings via NIM endpoint
    changenet-data-prepare/  # CSV generation for VCN training
  platform/              # Compute backends
    lepton/              #   DGX Cloud Lepton managed GPU compute
```

## Skill Layers

### Applications

High-level multi-step workflows that compose model and data skills into pipelines. Each application defines `init` and `iteration` stages in its `config.json`, with stage dependencies and conditional execution.

Examples:
- **deft-cosmos-rl**: 10-stage iterative loop (gap analysis, captioning, video generation, data merge, training, evaluation)
- **sda-vcn**: Mining-based SDA with embedding, k-NN search, and retraining

### Models

Trainable networks with TAO container-based execution. Each model skill defines actions (`train`, `evaluate`, `inference`, `export`), data source mappings, and spec parameter injection rules.

Examples:
- **cosmos-rl**: HuggingFace-based SFT with FSDP parallelism
- **visual-changenet**: Siamese classification + pixel-level segmentation

### Data

Data processing skills for embeddings, mining, captioning, and dataset preparation. These are non-trainable skills that transform or enrich data as part of larger workflows.

Examples:
- **siglip-embed**: Generate SigLIP embeddings for images
- **knn-mining**: GPU-accelerated k-NN similarity search
- **changenet-data-prepare**: CSV generation with filename normalization

### Platform

Compute backend configurations defining credential requirements, resource shapes, and failure modes.

## Skill Package Format

Each skill is a directory containing:

```
{skill-name}/
  config.json              # Structured config: actions, inputs, outputs, credentials
  {skill-name}.md          # Agent documentation (plain markdown, no frontmatter)
  defaults-{action}.json   # Per-action default spec values (JSON only)
  scripts/                 # Inline scripts for workflow stages (optional)
```

### config.json

The structured config defines everything the planner and execution engine need:

- **actions**: Commands, config format, inputs/outputs, upload excludes
- **data_sources**: How dataset URIs map to spec paths
- **spec_params**: Runtime-injected values (results_dir, checkpoints)
- **required_credentials**: Platform credentials needed
- **key_defaults**: Override broken upstream defaults

### {skill-name}.md

Agent-readable documentation covering:
- Model/skill overview and use cases
- Data format requirements
- Important parameters and their effects
- Hardware recommendations
- Error patterns and fixes

### defaults-{action}.json

Complete default spec for each action. The planner loads these and applies user overrides on top. All defaults are JSON format regardless of what the container expects (the script runner converts to YAML/TOML/JSON at runtime).

## Execution Patterns

| Pattern | Description | Example |
|---|---|---|
| **Single-step** | One model action (train/eval) | `generate_plan(network_arch="cosmos-rl", action="train")` |
| **Multi-step workflow** | Orchestrated pipeline with init + iteration stages | `generate_plan(workflow="deft-cosmos-rl")` |
| **Config mode** | Script runner writes spec file, runs command | Visual ChangeNet (YAML spec) |
| **Args mode** | Script runner builds CLI args from config | Cosmos Predict 2.5, SigLIP embed |

## Skill Discovery

The TAO-Next planner discovers skills via the `SkillBank` class:

1. Model skills are searched in both `models/` and `data/` directories
2. Workflow skills are loaded from `applications/`
3. Platform configs are loaded from `platform/`
4. Skill names use hyphen-case (`cosmos-rl`, not `cosmos_rl`)

The `TAO_SKILL_BANK_PATH` environment variable overrides the default discovery path.

## Adding a New Skill

1. Create a directory under the appropriate layer (`models/`, `data/`, `applications/`, `platform/`)
2. Add `config.json` with actions, inputs, outputs
3. Add `{name}.md` with agent documentation
4. Add `defaults-{action}.json` for each action with default spec values
5. If the skill has inline scripts, add them under `scripts/`
6. Test: verify `SkillBank().get_model_config('{name}')` returns your config
