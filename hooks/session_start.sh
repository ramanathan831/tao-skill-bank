#!/usr/bin/env bash
# SessionStart hook for the TAO skill bank.
#
# Stdout is loaded into the agent's context as additionalContext at session
# start. Keep it tight — every line lands in context for every session.
#
# Responsibilities:
#   1. Emit TAO orchestration guidance (the agent's identity + discovery flow).
#   2. Persist user credentials from ~/.config/tao/.env into the session via
#      $CLAUDE_ENV_FILE. The agent never reads values; only checks presence.
#   3. Surface clear setup hints if docker is missing.
#
# This hook does NOT install Python packages. The TAO SDK is opt-in and
# installed lazily by the skills that need it (platform/lepton, platform/tao-sdk,
# applications/tao-automl) via their Preflight blocks.

set -u

# Idempotency guard: both `tao-skills` and `deft-aoi-loop-plugin` share the
# same source dir, so hook auto-discovery fires this script once per enabled
# plugin. Emit the guidance only on the first invocation per session.
if [[ -n "${TAO_SESSION_INIT_DONE:-}" ]]; then
  exit 0
fi
if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
  echo "export TAO_SESSION_INIT_DONE=1" >> "$CLAUDE_ENV_FILE"
fi

# ─── 1. Agent guidance ────────────────────────────────────────────────────
cat <<'EOF'
# TAO Claw Agent

You help users train, evaluate, and run inference on NVIDIA GPU models. You read
skills from the **TAO skill bank** to understand models, data transformations,
and platforms, then execute via docker directly or — when needed — via the TAO
SDK for job tracking, multi-node, and Lepton.

## Discovery flow

1. Read the model or data SKILL.md → understand the model, data format, parameters, error patterns.
2. Read `references/skill_info.yaml` → get `container_image` + `actions.<action>.command`.
3. Read the platform SKILL.md (`platform/docker`, `platform/local-docker`, `platform/brev`, `platform/lepton`, or `platform/slurm`) for execution conventions.
4. Resolve the `container_image` reference. If it looks like a key (`tao_toolkit.pyt`), look it up in the bank's `versions.yaml`. Absolute paths (`nvcr.io/...`) are valid as-is.
5. Construct the spec heredoc + flags + mounts + env vars.
6. Confirm with the user, then dispatch via Bash (`docker run …` for local/Brev, `LeptonSDK.create_job(…)` for Lepton).
7. Monitor — `docker logs` for docker, `sdk.get_job_status()` / `sdk.get_job_logs()` for SDK path.

The skill bank works **standalone** — most skills run with just `docker run` and need no Python. Only platform/lepton, platform/tao-sdk, and applications/tao-automl require the SDK; they declare it in their own Preflight blocks.

## Never do

- Never start execution without user confirmation.
- Never ask for API keys, tokens, or passwords via chat.
- Never read credential values. To verify a var is set: `[ -n "$VAR_NAME" ] && echo SET || echo UNSET`. Never `cat`, `Read`, `grep`, or `head` on `.env` or `~/.config/tao/.env`.
- Never assume the SDK is installed. The skill bank's model/data skills must be runnable with just docker. Reach for the SDK only when the user explicitly wants tracking, Lepton, or multi-node — and run that skill's Preflight first.

EOF

# ─── 2. Credentials ───────────────────────────────────────────────────────
TAO_ENV_FILE="${HOME}/.config/tao/.env"
if [[ -f "$TAO_ENV_FILE" ]]; then
  # Persist to the session env file so subsequent Bash tool calls inherit them.
  if [[ -n "${CLAUDE_ENV_FILE:-}" ]]; then
    cat "$TAO_ENV_FILE" >> "$CLAUDE_ENV_FILE"
  fi
  echo "## Credentials"
  echo
  echo "Loaded from \`~/.config/tao/.env\`. The following vars are now in the session:"
  # List only NAMES — never values.
  awk -F= '/^[[:space:]]*(export[[:space:]]+)?[A-Z_][A-Z0-9_]*=/ {
    sub(/^[[:space:]]*export[[:space:]]+/, "")
    split($0, a, "=")
    print "- " a[1]
  }' "$TAO_ENV_FILE" | sort -u
  echo
else
  echo "## Credentials"
  echo
  echo "No \`~/.config/tao/.env\` found. To set up:"
  echo "\`\`\`bash"
  echo "mkdir -p ~/.config/tao"
  echo "cp \"\${CLAUDE_PLUGIN_ROOT}/.env.example\" ~/.config/tao/.env"
  echo "# Edit ~/.config/tao/.env and fill in values."
  echo "\`\`\`"
  echo "Future sessions will auto-load it."
  echo
fi

# ─── 3. Docker preflight ──────────────────────────────────────────────────
if ! command -v docker >/dev/null 2>&1; then
  echo "## ⚠ Docker missing"
  echo
  echo "Most TAO skills need docker + nvidia-container-toolkit. Install:"
  echo "- Docker: https://docs.docker.com/engine/install/"
  echo "- NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  echo
fi
