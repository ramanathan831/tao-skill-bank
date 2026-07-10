# TAO on NemoClaw (host MCP server)

Give a [NemoClaw](https://github.com/NVIDIA/NemoClaw) sandbox agent the ability
to run NVIDIA TAO training/inference on the host GPU — **without** granting the
sandbox Docker, GPU, or NGC credentials.

A small MCP server runs on the host and exposes typed tools over the OpenShell
host bridge. The agent calls tools (list/read/write the workspace, run/monitor/
stop TAO containers); the host executes the containers. Everything the sandbox
cannot safely do — Docker, the GPU, holding secrets — stays host-side.

```
agent (sandbox)  --HTTP over OpenShell bridge-->  tao-mcp server (host)  -->  docker run TAO container (host GPU)
```

## Why a host MCP server

The OpenShell sandbox is deny-by-default on egress and has no Docker. Direct
approaches fail: raw SSH/TCP can't leave the sandbox, and the managed
`nemoclaw mcp add` path needs an attested public-DNS endpoint. A host MCP
server registered directly in `openclaw.json` (the pattern the
[VSS blueprint](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization)
uses) reaches the local bridge cleanly and moves execution to where it works.

## Quick start

Prerequisites: a NemoClaw sandbox already onboarded with the **OpenClaw** agent
(tested with **Claude Opus 4.8** as the agent brain), and the host logged into
NGC (`docker login nvcr.io`). All agent-side setup is written into the
sandbox's `openclaw.json`, so an OpenClaw-based NemoClaw sandbox is required.

Then, on the host:

```bash
./setup-tao-nemoclaw.sh <sandbox-name>          # workspace defaults to ~/tao-workspace
```

The script starts the server (bound to the docker-bridge IP, off the LAN),
installs the TAO skills into the sandbox, registers the MCP server, opens the
bridge policy, and verifies reachability. Then in the agent
(`nemoclaw <sandbox> connect` → `openclaw tui`): *"What MCP tools do you have?"*

Put datasets under `<workspace>/<name>/`; the agent discovers them with `tao_ls`.

## Tools

| Tool | Purpose |
|------|---------|
| `tao_ls` / `tao_read` / `tao_write` | Inspect and author files in the host workspace |
| `tao_run` | Launch a container on the host GPU (constrained: `nvcr.io/*` NGC images only, workspace-confined mounts, `shm_size` for DataLoaders) |
| `tao_status` / `tao_logs` | Monitor a job |
| `tao_stop` / `tao_rm` | Stop/remove a job's container (TAO containers only) |

## Security

**This server is the security boundary — keep it.** `tao_run` refuses any image
outside `nvcr.io/*` (the NVIDIA NGC registry) and confines all mounts to a fixed
`--workspace-root` the agent cannot escape. This admits TAO images plus
QA/staging and data-generation images from other NGC orgs, but still refuses
arbitrary registries (Docker Hub, private, etc.). The agent gets NGC-image
execution, not arbitrary host control.

> **Do not** substitute a generic public Docker MCP server (e.g.
> `ckreiling/mcp-server-docker`). Those expose unconstrained `run_container`
> with arbitrary images and host mounts — equivalent to giving the sandboxed
> agent root on the host (it can read the host filesystem and the OpenShell
> credential store). Use the constrained server here unless you fully accept
> that exposure.

Two properties that keep it safe: the server binds the **docker-bridge IP**
(reachable only by sandbox containers, not the LAN), and the sandbox reaches it
only through the **bridge egress policy** the setup applies.

## Scope

Runs TAO workflows on the host GPU. The agent reads the skill, authors the
spec, stages models (containers inherit the host network, so HuggingFace / NGC
/ S3 pulls work in-container), launches `tao_run`, and monitors — orchestrating
multi-step workflows itself over the tools. Verified on hardware (DINO, Visual
ChangeNet, DEFT AOI).

AutoML's managed search loop may still need the Claude Code / Codex plugin
runtime; other TAO workflows run through this surface.

## Files

| File | What |
|------|------|
| `server.py` | The MCP server (stdlib + `mcp` + `uvicorn`) |
| `setup-tao-nemoclaw.sh` | One-command setup for a sandbox |
| `VERIFIED-RUNBOOK.md` | The manual step-by-step the script automates |

## Notes / gotchas (baked into the script)

- Server binds the sandbox's docker-bridge gateway IP; discover per host.
- `policy-add` needs `--yes` non-interactively (else it hangs silently).
- Editing `openclaw.json` must `chmod 660` after (else `GATEWAY_UNSAFE_CONFIG_PATH`).
- The MCP SDK's DNS-rebinding guard requires `host.openshell.internal` in the
  server's `allowed_hosts` (already set in `server.py`).
- Adding/changing tools requires `nemoclaw <sandbox> gateway restart` for the
  agent to re-fetch the tool list.
- Tested against OpenShell 0.0.72. **Experimental** — NemoClaw is alpha.
