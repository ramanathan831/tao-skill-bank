# NVIDIA [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skills-bank)

Portable agent skills for training, evaluating, and running inference on NVIDIA TAO models. Works with Claude Code, Codex, Gemini CLI, or any coding agent that speaks the [Agent Skills open standard](https://agentskills.io). **Zero Python required** for local docker workflows — install the plugin, install docker + nvidia-container-toolkit, and an agent can run every skill by constructing `docker run` commands directly. For advanced features (job tracking, multi-node, S3 I/O wrapping), an optional Python layer — the [TAO Execution SDK](#optional-python-layer) — sits on top.

## Install

The skill bank works with both Claude Code and Codex. Pick the runtime you use.

### Claude Code

In a Claude Code session, add the marketplace and install the plugin:

```
/plugin marketplace add ssh://git@gitlab-master.nvidia.com:12051/nvidia-tao-toolkit/tao-skills-external.git
/plugin install tao-skills@tao-skill-bank
```

That's it — no `git clone`, no `pip install`. The `tao-skills` plugin bundles all 56 skills (every model, data, platform, and application). The plugin's [`SessionStart`](hooks/session_start.sh) hook loads the [`AGENTS.md`](AGENTS.md) identity at the start of every session.

### Codex

Codex setup has **two independent pieces** — the plugin (which surfaces the skills to Codex) and `AGENTS.md` (which loads the agent identity). You need both for parity with Claude Code.

#### One command (recommended)

```bash
curl -fsSL https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-skills-external/-/raw/main/scripts/install-codex-agents.sh | bash
```

…or, if you've already cloned or extracted the repo from a zip, run
`scripts/install-codex-agents.sh` from that directory. The script registers the
marketplace, installs `tao-skill-bank`, and copies `AGENTS.md` to
`~/.codex/AGENTS.md` so the TAO identity loads in every Codex session. It's
idempotent and backs up any existing `~/.codex/AGENTS.md` before overwriting.
Override the source with `TAO_SKILL_BANK_MARKETPLACE=…` and
`TAO_SKILL_BANK_REF=…` to use a fork, pinned ref, or local absolute path:

```bash
cd /absolute/path/to/tao-skills-external
TAO_SKILL_BANK_MARKETPLACE=/absolute/path/to/tao-skills-external \
  scripts/install-codex-agents.sh
```

#### Manual steps

If you'd rather drive each step yourself:

**1. Install the plugin.** Either use the VS Code Codex extension's plugin UI, or from the CLI:

```bash
codex plugin marketplace add ssh://git@gitlab-master.nvidia.com:12051/nvidia-tao-toolkit/tao-skills-external.git
codex plugin add tao-skill-bank@tao-local-plugins
```

This installs the bundle to `~/.codex/plugins/cache/tao-local-plugins/tao-skill-bank/<version>/` (the `tao-local-plugins` segment comes from the `name` field in `.agents/plugins/marketplace.json`).

For a local zip or clone, use the absolute path instead of the Git URL:

```bash
codex plugin marketplace add /absolute/path/to/tao-skills-external
codex plugin add tao-skill-bank@tao-local-plugins
```

**2. Load the agent identity (`AGENTS.md`).** The plugin install does **not** auto-load [`AGENTS.md`](AGENTS.md) — Codex's `AGENTS.md` discovery walks down from the project root, not into the plugin cache (see [openai/codex#16430](https://github.com/openai/codex/issues/16430) for why plugin-bundled `SessionStart` hooks don't fix this yet). Pick one:

- **Per-project**: `git clone` this repo and launch `codex` from inside the clone. Codex auto-loads `AGENTS.md` from the project root per the [agents.md](https://agents.md/) cross-runtime spec.
- **Globally** (one-time copy): `cp ~/.codex/plugins/cache/tao-local-plugins/tao-skill-bank/<version>/AGENTS.md ~/.codex/AGENTS.md`. The identity then loads in every Codex session, anywhere.

Once Codex starts honoring plugin-bundled hooks, the identity will install automatically alongside the plugin — until then, this manual step is needed.

### Credentials

Credentials can be exported in the current shell or stored in an env file that
the agent is allowed to source, such as `~/.config/tao/.env` or
`~/.tao/secrets.env`. On first session start, the plugin looks for
`~/.config/tao/.env` and auto-loads it. To set up:

```bash
mkdir -p ~/.config/tao
cp "${CLAUDE_PLUGIN_ROOT}/.env.example" ~/.config/tao/.env  # template ships in the plugin
# Edit ~/.config/tao/.env and fill in NGC_KEY, S3 keys, etc.
```

The `.env.example` is also at the [repo root](.env.example) for direct
reference. The agent never reads credential values — it only checks presence.
When a workflow needs Hugging Face access, get a token from
[Hugging Face settings](https://huggingface.co/settings/tokens) and accept the
model or dataset license before launch.

If a readiness check reports a missing CLI, container image, backbone, or
credential, the TAO skills can often install or stage the missing piece after
you approve the action. Ask the agent to continue the original workflow after a
blocker is resolved; it should rerun preflight and proceed from the same task.

### When does the SDK get installed?

The TAO SDK is **opt-in** and installed lazily. Most skills (any model or data skill) run with just `docker run` and need no Python. Only `platform/tao-run-platform` (`tao-run-platform`), the managed-platform skills (slurm/kubernetes/docker), and `applications/tao-run-automl` (`tao-run-automl`) require the SDK; their Preflight blocks tell the agent to run a `pip install "nvidia-tao-sdk[<platform>] @ git+https://..."` direct-URL the first time the skill is invoked. (The SDK isn't on public PyPI yet — see each skill's Preflight for the exact command.)

### Updating

**Claude Code:**

```
/plugin marketplace update tao-skill-bank
/reload-plugins
```

If skills look stale (cached contents):

```bash
rm -rf ~/.claude/plugins/cache/tao-skill-bank
```

then re-run `/plugin install`.

**Codex:**

```bash
codex plugin marketplace upgrade tao-skill-bank
```

If you copied `AGENTS.md` to `~/.codex/AGENTS.md`, re-copy from the upgraded plugin cache to pick up identity changes.

## Getting started (5 minutes)

The quickest way to verify your setup: run a Visual ChangeNet inference on a sample image.

### Prerequisites

```shell
docker --version
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
echo "$NGC_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

If any check fails, see `platform/tao-run-on-docker/SKILL.md` for install/troubleshooting.

### Smoke test

In a Claude Code session with the plugin installed, ask:

> *"Run Visual ChangeNet inference on this sample image: /tmp/sample.png. Write results to /tmp/vcn-out/."*

The agent will read `models/tao-train-visual-changenet/SKILL.md` (skill name `tao-train-visual-changenet`, plus its `references/skill_info.yaml` if present), construct a `docker run --gpus all ...` invocation, and execute via Bash. **No Python needed.** No SDK install. Just docker + the plugin. For classify mode, expect per-image PASS/NO_PASS-style predictions and result files under `/tmp/vcn-out/`. For segment mode, expect binary change-mask outputs under the requested results directory.

For more complex workflows, see `applications/tao-run-deft-aoi/SKILL.md`
(`tao-run-deft-aoi`) for iterative fine-tuning with synthetic data augmentation
and `applications/tao-run-automl/SKILL.md` (`tao-run-automl`) for
hyperparameter optimization. AutoML launch reviews should show the number of
recommendations, metric, search space, expected runtime, and resolved train
image before long-running jobs start.

## What's in the bank

| Layer | Purpose | Examples |
|---|---|---|
| `models/` | Network-centric skills: containers, commands, data formats, checkpoints | `tao-finetune-cosmos-reason`, `tao-train-visual-changenet`, `tao-finetune-clip`, `tao-train-dino`, `tao-train-segformer`, … |
| `data/` | Data preparation, analysis, and enhancement | `tao-mine-aoi-images`, `tao-analyze-gaps-visual-changenet`, `tao-route-visual-changenet-samples`, `tao-analyze-gaps-vlm-bcq`, `tao-convert-dataset-format`, `tao-validate-dataset-format`, `tao-generate-image-grounding`, `tao-generate-referring-expressions`, `tao-generate-video-reasoning-annotations` |
| `platform/` | Where and how jobs run | `tao-run-on-docker` (conventions), `tao-run-on-brev` (instance-based GPU), `tao-run-on-slurm` (remote SLURM cluster), `tao-run-on-kubernetes` (k8s), `tao-run-on-local-docker` (local Docker daemon), `tao-run-platform` (optional Python SDK) |
| `applications/` | End-to-end workflows composing the layers above | `tao-run-deft-aoi`, `tao-run-automl-deft-pipeline`, `tao-analyze-changenet-rca`, `tao-train-single-step`, `tao-run-automl`, `tao-finetune-huggingface-model`, `tao-port-huggingface-model`, `tao-run-inference-service` |

Each skill is a directory with `SKILL.md` (agent-readable instructions). Optional `references/skill_info.yaml` provides structured metadata for SDK-orchestrated execution; optional `scripts/` bundles supporting code.

The top-level `skills/` directory is not a second copy of the skill bank. It is the Codex plugin surface for small helper/router skills, such as capability discovery and launch intake. Canonical model, data, platform, and application skills live once in the layer directories above; do not add symlinks or copies under `skills/`.

## Optional Python layer

For users who want job handles, S3 I/O wrapping via `script_runner`, state persistence, multi-node distributed training, or failure analysis, the [TAO Execution SDK](https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-sdk) provides a single wheel with optional extras:

```shell
# The SDK is not on public PyPI yet — install via pip direct-URL:
REPO='git+https://gitlab-master.nvidia.com/nvidia-tao-toolkit/tao-sdk.git'
pip install "nvidia-tao-sdk @ $REPO"                  # core
pip install "nvidia-tao-sdk[brev] @ $REPO"            # + Brev (wraps brev CLI with Job handles)
pip install "nvidia-tao-sdk[slurm] @ $REPO"           # + SLURM
pip install "nvidia-tao-sdk[kubernetes] @ $REPO"      # + Kubernetes
pip install "nvidia-tao-sdk[docker] @ $REPO"          # + local Docker
pip install "nvidia-tao-sdk[all] @ $REPO"             # all platforms
```

You don't have to pre-install — the relevant skills (`tao-run-platform`, `tao-run-automl`) run a Preflight that prompts the agent to install the right extra on first use. If you're running locally on your own GPU or on Brev via `brev exec`, you don't need the SDK at all.

## Contributing a new skill

See [docs/authoring.md](docs/authoring.md) for the full guide. The minimum viable skill is just `SKILL.md` — `references/skill_info.yaml` and friends are optional and only added when they earn their keep.

In brief:

1. Pick the layer (`models/`, `data/`, `platform/`, `applications/`).
2. Copy a template from [`templates/skill-skeleton/`](templates/skill-skeleton/) — `minimal/` for the bare path, `model/`, `data/`, `platform/`, or `workflow/` for richer scaffolding.
3. Fill in frontmatter and SKILL.md body. Body must contain a `## Quick Start` section, a `docker run` block, an SDK call, or a link to `references/skill_info.yaml`.
4. Add the skill path to [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json) under the relevant plugin(s).
5. Do not add a mirror entry under top-level `skills/`; Codex helper skills route to the canonical layer directories.
6. Validate with `scripts/validate-skills.sh` before submitting a PR.

## Repository structure

```
tao-skills-external/
├── .claude-plugin/
│   ├── marketplace.json              # marketplace catalog (plugin definitions)
│   └── plugin.json                   # plugin manifest (fallback when loaded directly)
├── hooks/
│   ├── hooks.json                    # SessionStart hook registration
│   └── session_start.sh              # emits agent guidance; sources ~/.config/tao/.env
├── .codex-plugin/
│   └── plugin.json                   # Codex plugin manifest
├── .agents/
│   └── plugins/marketplace.json      # Codex marketplace entry
├── .env.example                      # credential template (copy to ~/.config/tao/.env)
├── versions.yaml                     # single source of truth: container images + SDK wheel versions
├── README.md
├── docs/
│   ├── authoring.md                  # guide for adding new skills
│   └── maintenance.md                # RC bump procedure for versions.yaml
├── templates/skill-skeleton/         # copy-paste starting points (minimal + per-layer)
├── scripts/
│   ├── validate-skills.sh            # CI validator
│   ├── verify-standalone.sh          # end-to-end smoke (docker-only path)
│   ├── install-codex-agents.sh       # one-shot Codex install: marketplace + plugin + AGENTS.md
│   └── migrate-to-version-keys.py    # one-shot: literal nvcr.io paths → versions.yaml keys
├── applications/
├── data/
├── models/
├── platform/
└── skills/                           # Codex helper/router skills only; no mirrored skill symlinks
```

## CI

The repo runs three CI suites in parallel:

- **NV-ACES skill evaluation** (`.skill-eval.yml`) — Tier 1/2 quality scoring, security scan.
- **Skill execution eval** (`.gitlab-ci.yml`) — runs each skill's `eval.config` on a real GPU runner.
- **`validate-skills`** (`scripts/validate-skills.sh`) — marketplace path resolution, no `skills/` mirrors, frontmatter, body has runnable info, no SDK leaks, hook references resolve.

PRs must pass all three before merge.

## Design rules

- **Docker-native first.** Every model/data skill should be runnable with just `docker run` + the contents of `SKILL.md`. SDK invocation is an optional enhancement, documented in `platform/tao-run-platform`.
- **Generic docker conventions live once** in `platform/tao-run-on-docker`. Other skills defer to it for `--gpus`, NGC auth, mount patterns, data-root relocation, etc.
- **No SDK leaks in model/data/application skills.** `tao_sdk`-specific imports, `sdk.create_job` calls, and credential-file references belong only in `platform/tao-run-platform`.
- **Minimum-viable skill is `SKILL.md` only.** Add `references/skill_info.yaml` only when SDK orchestration or multi-action structured metadata earn their keep.
- **One canonical location per skill.** Model, data, platform, and application skills live only in their layer directories; `skills/` is for Codex helper/router skills, not mirrored copies.
- **Prefer portability over cleverness.** A skill that works across three coding agents is more valuable than a skill that works perfectly in one.
