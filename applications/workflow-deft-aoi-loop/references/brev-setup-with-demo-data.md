---
name: brev-setup-with-demo-data
description: Reference procedure for deploying the full DEFT Loop (Evaluate-RCA-SDG-Retrain) on a Brev GPU instance. Bundled with the `workflow-deft-aoi-loop` skill — used when setting up the DEFT pipeline on a cloud GPU with the demo dataset.
---

# DEFT Loop on Brev

Deploy the complete DEFT loop (Data-Efficient Fine-Tuning) on a Brev GPU instance — OpenClaw agent + Docker containers + datasets + skills, ready to run autonomous training iterations.

The DEFT loop is an agentic pipeline: Claude Opus 4.6 (via OpenClaw) autonomously runs Root Cause Analysis, Synthetic Data Generation, Data Mining, and ChangeNet Retraining in a loop until a target KPI is met.

## Approach

**Collect everything → Validate fast → Deploy unattended.** The goal is minimal human interaction: gather all inputs in a single pass, run quick sanity checks so the user knows immediately if something is wrong, then kick off the entire deployment hands-free. The user should be able to go grab a coffee and come back to a live DEFT agent.

**Hardware requirements:**
- GPU: L40S or better (48GB+ VRAM for training + SDG inference)
- Disk: 500GB+ **on the root volume** (Docker images ~139GB, datasets ~6GB, training checkpoints). Some providers split storage into a small root (~97GB) plus a separate ephemeral volume — Docker writes to root by default and will run out of space. If the instance has a split disk layout, Docker's data-root and containerd storage must be relocated to the large volume before pulling images.
- CPU instances will NOT work — DEFT requires GPU for training and inference

---

## Phase 1: Collect All Inputs (single message, single pass)

Ask the user for ALL of the following in **one message**. Do not proceed piecemeal — collect everything upfront so you never have to interrupt the deployment later.

```
To deploy the DEFT loop on Brev, I need the following:

**Required:**
1. NGC API key — from https://ngc.nvidia.com/ (for Docker image pulls from nvcr.io)
2. NVIDIA Inference Hub key — from https://inference.nvidia.com/ → Key Management → Personal key
   (sk-* or nvapi-* format — powers the Claude Opus 4.6 LLM behind OpenClaw)
3. Path to the `deft-packaged/` folder on your local machine, containing:
   - `deft-kpi-dataset.tar.gz` (~3.5GB) — validation images + CSVs
   - `deft-train-csv.tar.gz` (~4KB) — training CSVs
   - `deft-backbone.tar.gz` (~222MB) — C-RADIOv2 backbone weights
   - `deft-mining-pool.tar.gz` (~79MB) — source embeddings + images for data mining

**Optional (have sensible defaults):**
4. Brev instance — existing GPU instance name, or I'll create a new one (default: A100 80GB)
5. Augmentation strategy:
   - Data Mining only (minimal — 2 Docker images, fastest setup)
   - Data Mining + AnomalyGen (also needs from `deft-packaged/`):
     - `deft-anomalygen-prereqs.tar.gz` (~2.7MB) — clean images, ROI, submasks
     - `anomaly_gen_checkpoint.tar.gz` (~794MB) — pretrained model
   - All three arms (also needs `PEGATRON_AOI.zip` ~619MB + Omniverse credentials)
6. HuggingFace token — for gated model downloads
7. OMNI_USER / OMNI_PASS — only if using Omniverse SDG arm
```

If the user provides some values inline with their initial prompt, don't re-ask those. Fill defaults for anything optional that isn't provided.

---

## Phase 2: Validate (fast — fail before spending money or time)

Run ALL checks before creating any instance or transferring any data. Every critical check must pass before proceeding.

### 2A. Brev CLI + login

```bash
which brev
brev ls 2>&1
```

If `brev` is not found, install it:
```bash
# macOS
brew install brevdev/homebrew-brev/brev
# Linux / WSL
sudo bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"
```

If `brev ls` fails with an auth error → **stop and tell the user to run `brev login`** in their terminal. This opens a browser and cannot be automated. **Do not proceed until login succeeds** — the Brev session can expire at any time, and hitting this mid-deploy wastes all progress.

### 2B. NGC API key test

```bash
echo "<NGC_API_KEY>" | docker login nvcr.io -u '$oauthtoken' --password-stdin 2>&1
docker logout nvcr.io 2>/dev/null
```

If login fails → **stop**. Tell the user their NGC key is invalid or expired. Link: https://ngc.nvidia.com/

### 2C. NVIDIA Inference Hub key test

```bash
# For sk-* keys
curl -sf -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer <NVIDIA_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"aws/anthropic/bedrock-claude-opus-4-6","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  "https://inference-api.nvidia.com/chat/completions"

# For nvapi-* keys
curl -sf -o /dev/null -w "%{http_code}" \
  -H "x-api-key: <NVIDIA_KEY>" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"azure/anthropic/claude-opus-4-6","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  "https://inference-api.nvidia.com/v1/messages"
```

Accept any 2xx. If 401/403 → **stop**. Link: https://inference.nvidia.com/ → Key Management.

### 2D. Local files sanity check

```bash
ls -lh <path>/deft-kpi-dataset.tar.gz
ls -lh <path>/deft-train-csv.tar.gz
ls -lh <path>/deft-backbone.tar.gz
ls -lh <path>/deft-mining-pool.tar.gz
# Optional based on augmentation strategy:
ls -lh <path>/deft-anomalygen-prereqs.tar.gz     # if AnomalyGen arm
ls -lh <path>/anomaly_gen_checkpoint.tar.gz       # if AnomalyGen arm
ls -lh <path>/PEGATRON_AOI.zip                    # if Omniverse arm
```

If any required archive is missing → **stop**.

### 2E. Print validation summary

```
Pre-flight checks:
  [PASS] Brev CLI installed and authenticated
  [PASS] NGC API key valid
  [PASS] NVIDIA Inference Hub key valid (sk-* → openai-completions mode)
  [PASS] deft-loop-workspace-essential.tar.gz found (4.1GB)
  [SKIP] anomaly_gen_checkpoint.tar.gz — not needed for data-mining-only
  [SKIP] pegatron_aoi.zip — not needed for data-mining-only

All checks passed. Ready to deploy.
```

If ANY critical check fails, **stop and help the user fix it** before proceeding.

---

## Phase 3: Deploy (unattended — user can walk away)

Once all checks pass, tell the user:

> **All pre-flight checks passed. Starting deployment now.**
> **This takes ~45 minutes (mostly Docker image pulls). You can go grab a coffee — I'll have a status summary ready when you're back.**

Then execute Steps 3A through 3F without pausing for confirmation.

### 3A. Create or verify GPU instance

If using an existing instance:
```bash
brev ls | grep <instance_name>
```

If creating a new one:
```bash
brev create <instance_name> --type massedcompute_A100_sxm4_80G
```

Default GPU selection priority (sorted by price):
1. `massedcompute_A100_sxm4_80G` ($1.49/hr, 625GB disk)
2. `hyperstack_A100_80G` ($1.62/hr, 850GB disk)
3. Fall back to: `brev create <name>` with auto-select (defaults: ≥48GB VRAM, ≥500GB disk)

Wait for READY status before proceeding.

### 3B. Create remote directories BEFORE copying

**IMPORTANT:** `brev cp` fails silently if the destination directory doesn't exist on the remote. Always create directories first:

```bash
brev exec <instance> "mkdir -p ~/workspace/.claude/skills ~/workspace/kpi/images ~/workspace/train/base ~/workspace/results ~/workspace/augmentation/{anomalygen/{checkpoint,submasks,roi,clean_images},omniverse/scene,mining/{source_images,model_cache},backbone} ~/workspace/specs"
```

### 3C. Transfer setup script + skills first (small, fast)

The 5 DEFT skills live in the `tao-skills-external` repo:
- `applications/workflow-deft-aoi-loop/` (orchestrator — also holds this reference doc and the shared scripts)
- `data/deft-aoi-rca-changenet/`
- `data/deft-aoi-anomalygen-inference/`
- `data/deft-aoi-omniverse-sdg/`
- `data/deft-aoi-data-mining/`

Transfer them into `~/workspace/.claude/skills/` on the instance:

```bash
# Setup script (bundled with the orchestrator)
brev cp <REPO>/applications/workflow-deft-aoi-loop/scripts/setup-deft-loop.sh <instance>:/tmp/

# Orchestrator
scp -r <REPO>/applications/workflow-deft-aoi-loop <instance>:~/workspace/.claude/skills/workflow-deft-aoi-loop

# 4 sub-skills
for s in deft-aoi-rca-changenet deft-aoi-anomalygen-inference deft-aoi-omniverse-sdg deft-aoi-data-mining; do
  scp -r <REPO>/data/$s <instance>:~/workspace/.claude/skills/$s
done
```

Where `<REPO>` is the local path to your `tao-skills-external` clone.

Note: Use `scp` instead of `brev cp` for directories to avoid escaping issues.

### 3D. Start archive transfers in background

Transfer all archives in parallel while the setup script runs Phases 1-3. The setup script gracefully skips missing archives.

```bash
# Required (run in parallel)
scp <path>/deft-kpi-dataset.tar.gz <instance>:/tmp/ &
scp <path>/deft-train-csv.tar.gz <instance>:/tmp/ &
scp <path>/deft-backbone.tar.gz <instance>:/tmp/ &
scp <path>/deft-mining-pool.tar.gz <instance>:/tmp/ &

# Optional (based on augmentation strategy)
scp <path>/deft-anomalygen-prereqs.tar.gz <instance>:/tmp/ &
scp <path>/anomaly_gen_checkpoint.tar.gz <instance>:/tmp/ &
scp <path>/PEGATRON_AOI.zip <instance>:/tmp/ &
wait
```

Note: Use `scp` via SSH hostname (not `brev cp`) to avoid escaping issues.

### 3E. Run setup script (parallel with archive transfer)

**IMPORTANT — NVM in non-interactive shells:** `brev exec` runs a non-interactive shell where NVM isn't loaded. On re-runs, always source NVM first:

```bash
brev exec <instance> "source ~/.nvm/nvm.sh 2>/dev/null; bash /tmp/setup-deft-loop.sh <NGC_API_KEY> --nvidia-key <NVIDIA_KEY>"
```

The script handles:
- **Phase 1:** OpenClaw installation + config (200K context, shell+web tools)
- **Phase 2:** System checks (GPU, Docker, disk space)
- **Phase 3:** NGC Docker login + image pulls (~139GB, 1-2 hours — the longest step)
- **Phase 4:** Workspace extraction (from /tmp/ archives — skips if not yet transferred)
- **Phase 5:** DEFT skills registration + verification

**If the script fails with an OpenClaw config validation error:** The setup script writes the current config schema (with `plugins.entries` for perplexity). If you hit config errors on older/newer OpenClaw versions, run `openclaw doctor --fix` and then re-run with `--skip-openclaw`:

```bash
brev exec <instance> "source ~/.nvm/nvm.sh && openclaw doctor --fix"
brev exec <instance> "source ~/.nvm/nvm.sh 2>/dev/null; bash /tmp/setup-deft-loop.sh <NGC_API_KEY> --nvidia-key <NVIDIA_KEY> --skip-openclaw"
```

Optional flags:
```bash
--skip-openclaw          # Skip Phase 1 (already installed)
--skip-docker            # Skip Phase 3 (images already pulled)
--skip-workspace         # Skip Phase 4 (workspace already set up)
--with-systemd           # Use systemd instead of nohup for gateway
--hf-token <TOKEN>       # HuggingFace token
--omni-user <USER>       # Omniverse credentials
--omni-pass <PASS>
```

### 3F. Extract workspace data (after archive transfers complete)

If transfers were still running when the setup script hit Phase 4, re-run extraction only:

```bash
brev exec <instance> "bash /tmp/setup-deft-loop.sh <NGC_API_KEY> --nvidia-key <NVIDIA_KEY> --skip-openclaw --skip-docker"
```

Or extract manually:
```bash
ssh <instance> "tar -xzf /tmp/deft-kpi-dataset.tar.gz -C ~/workspace/kpi/"
ssh <instance> "tar -xzf /tmp/deft-train-csv.tar.gz -C ~/workspace/train/base/"
ssh <instance> "tar -xzf /tmp/deft-backbone.tar.gz -C ~/workspace/augmentation/backbone/"
ssh <instance> "tar -xzf /tmp/deft-mining-pool.tar.gz -C ~/workspace/augmentation/mining/"
# Optional:
ssh <instance> "tar -xzf /tmp/deft-anomalygen-prereqs.tar.gz -C ~/workspace/augmentation/anomalygen/"
ssh <instance> "tar -xzf /tmp/anomaly_gen_checkpoint.tar.gz -C ~/workspace/augmentation/anomalygen/checkpoint/"
ssh <instance> "unzip -o /tmp/PEGATRON_AOI.zip -d ~/workspace/augmentation/omniverse/scene/"
```

Fix permissions:
```bash
ssh <instance> "sudo chown -R \$(whoami) ~/workspace/"
```

### 3G. Verify skills are discoverable

Skills are at `~/workspace/.claude/skills/deft-loop/`. Claude Code auto-discovers skills under `.claude/` in the working directory. Verify the agent will run from `~/workspace/`:

```bash
ssh <instance> "ls ~/workspace/.claude/skills/deft-loop/SKILL.md && echo 'Skills OK'"
```

### 3H. Verify deployment

```bash
brev exec <instance> "source ~/.nvm/nvm.sh && openclaw health"
brev exec <instance> "docker images | grep -E 'tao-toolkit|anomalygen|embed|mining|pcb-aoi'"
brev exec <instance> "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
brev exec <instance> "ls ~/workspace/kpi/ ~/workspace/specs/ ~/workspace/.claude/skills/deft-loop/"
```

### 3I. Print final status

```
============================================================
  DEFT Loop Deployment Complete!
============================================================

  Instance:    <instance_name> (A100 80GB, $1.49/hr)
  Gateway:     http://localhost:18789/#token=<TOKEN>

  NEXT STEPS:

  1. In a SEPARATE terminal (must stay open), run:
     brev port-forward <instance_name> -p 18789:18789

  2. Open the gateway URL above in your browser

  3. Paste this prompt to start the DEFT loop:
     "Run the DEFT loop on the NV_PCB_Siamese dataset.
      Target: FAR < 0.1% at recall=100%.
      Max iterations: 3."

  Token: brev exec <instance> "cat ~/.openclaw/.gateway-token"
============================================================
```

### Prompt for Telegram setup

After deployment is confirmed working, **always ask**:

> "DEFT loop is deployed! Would you like to set up Telegram so you can monitor it from your phone? Just need a bot token from @BotFather — takes 2 minutes."

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **OpenClaw config validation error** | Config schema may change between OpenClaw versions (e.g., perplexity config location). Run: `brev exec <instance> "source ~/.nvm/nvm.sh && openclaw doctor --fix"` then re-run setup with `--skip-openclaw`. |
| **`brev cp` fails for directories** | Remote directory must exist first. Run `brev exec <instance> "mkdir -p <path>"` before copying. |
| **Brev session expired mid-deploy** | Run `brev login` locally (opens browser). Then resume — instance is still running. Use `--skip-openclaw --skip-docker` flags to skip completed phases. |
| **`brev exec` retries and re-runs script** | SSH drops cause `brev exec` to reconnect and re-execute. On re-runs, use `--skip-openclaw` to avoid corrupting existing config. |
| **`openclaw` not found via `brev exec`** | NVM isn't loaded in non-interactive shells. Always prefix with `source ~/.nvm/nvm.sh 2>/dev/null;` or run: `brev exec <instance> "sudo ln -sf \$(which openclaw) /usr/local/bin/openclaw"` |
| **"Please accept license on the browser"** | Some NGC images (e.g., `tao-toolkit`) require EULA acceptance. Visit the image page on https://ngc.nvidia.com/, sign in, and click "Accept" before pulling. |
| **Docker pull fails for nvcr.io/nvidian/ images** | Internal NVIDIA images. NGC key needs org-level access to `nvcr.io/nvidian/iva/`. Contact your NGC org admin. |
| **"No space left on device" during Docker pull** | Use an instance with 500GB+ root volume (e.g., `massedcompute_L40`, `massedcompute_A100_sxm4_80G`). If stuck on a split-disk instance, check `df -h` — if there's a large `/ephemeral` or `/mnt` volume, relocate Docker and containerd: `sudo systemctl stop docker containerd && sudo mkdir -p /ephemeral/docker /ephemeral/containerd && sudo rsync -aP /var/lib/docker/ /ephemeral/docker/ && sudo rsync -aP /var/lib/containerd/ /ephemeral/containerd/ && sudo rm -rf /var/lib/docker /var/lib/containerd && sudo ln -sf /ephemeral/docker /var/lib/docker && sudo ln -sf /ephemeral/containerd /var/lib/containerd && sudo bash -c 'echo "{\"data-root\":\"/ephemeral/docker\"}" > /etc/docker/daemon.json' && sudo systemctl start containerd docker`. |
| **nvidia-container-toolkit not working** | Run: `sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker` |
| **Training fails with OOM** | Reduce batch size in TAO spec. L40S (48GB) handles batch_size=8. For smaller GPUs, try 4 or 2. |
| **AnomalyGen SDG takes too long** | Normal — ~30 min per iteration on L40S. Cosmos model is large. |
| **Archives not found in /tmp/** | Transfer may still be running. Check with: `brev exec <instance> "ls -lh /tmp/*.tar.gz /tmp/*.zip"` |
| **Hallucinations / bad output** | Verify context window: `brev exec <instance> "cat ~/.openclaw/openclaw.json \| grep contextWindow"` — should be 200000. |
| **Agent can't run commands** | `tools.allow` missing `"group:shell"`. Check config and restart gateway. |
| **401 "expected sk-" with nvapi key** | Wrong API mode. Re-run setup with correct key type — script auto-detects `sk-*` vs `nvapi-*`. |
| **Port forward drops immediately** | `brev port-forward` must run in its own terminal — it's a foreground process. Cannot be backgrounded. |
| **Lost gateway token** | `brev exec <instance> "cat ~/.openclaw/.gateway-token"` |
| **Gateway "port in use"** | `brev exec <instance> "fuser -k 18789/tcp"` then restart gateway. |
| **deft_skills not registered** | Skills should be at `~/workspace/.claude/skills/deft-loop/`. Claude Code auto-discovers skills under `.claude/skills/` in the workspace directory. |

---

## Quick Reference

```bash
# Local machine
brev ls                                                # List instances
brev shell <instance>                                  # SSH in
brev port-forward <instance> -p 18789:18789            # UI access (own terminal!)
brev exec <instance> "command"                         # Remote command
brev cp <local_file> <instance>:<remote_path>          # Upload file

# On the instance (always source nvm first in brev exec)
source ~/.nvm/nvm.sh && openclaw health                # Health check
openclaw tui                                           # Terminal chat
openclaw models status                                 # Model config
nvidia-smi                                             # GPU status
docker images | grep -E 'tao|anomaly|embed|mining'     # Check images
```

## File Locations (v2 layout)

| Path | Contents |
|------|----------|
| `~/.openclaw/openclaw.json` | OpenClaw main config |
| `~/.openclaw/.gateway-token` | Gateway auth token |
| `~/workspace/` | DEFT workspace root |
| `~/workspace/kpi/` | Validation CSVs + shared images (read-only) |
| `~/workspace/train/` | Base + per-iteration augmented training data |
| `~/workspace/train/base/` | Original training CSV + images |
| `~/workspace/train/iter{N}/` | Iteration N augmentation (3 arms separated) |
| `~/workspace/results/` | Checkpoints, inference, RCA reports, loop state |
| `~/workspace/results/deft_state.json` | Loop state file (crash recovery) |
| `~/workspace/augmentation/` | Static prerequisites for 3 arms + backbone |
| `~/workspace/augmentation/anomalygen/` | AnomalyGen checkpoint, clean images, ROI, submasks |
| `~/workspace/augmentation/omniverse/` | USD scene for Omniverse SDG |
| `~/workspace/augmentation/mining/` | Source embeddings + source images |
| `~/workspace/augmentation/backbone/` | Backbone weights (C-RADIOv2) |
| `~/workspace/specs/` | Training YAML specs |
| `~/workspace/.claude/skills/deft-loop/` | DEFT loop plugin (auto-discovered when cwd is ~/workspace/) |
| `~/workspace/.claude/skills/deft-loop/skills/` | 7 bundled sub-skills |
| `~/.deft_env` | Environment variables (NGC key, etc.) |
| `/tmp/openclaw-gateway.log` | Gateway log (nohup mode) |
