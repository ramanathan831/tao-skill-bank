#!/usr/bin/env python3
"""Minimal TAO MCP server — runs TAO training containers on the host GPU.

The NemoClaw sandbox agent calls this over MCP (Streamable HTTP); the host runs
the container. The agent never touches Docker, the GPU, or NGC credentials.

Security boundary (this server IS the boundary):
  - Only images under nvcr.io/ (the NVIDIA NGC registry) may run. This admits
    TAO images plus QA/staging and data-generation images from other NGC orgs;
    it still refuses arbitrary registries (Docker Hub, private, etc.).
  - Data and results are confined to subpaths of a fixed --workspace-root that
    the agent cannot change or escape (no absolute paths, no `..`).
  - Reachability is gated by the OpenShell egress policy: the sandbox reaches
    this server only through the host bridge on the one allowed port. A bearer
    token is therefore optional — set TAO_MCP_TOKEN to require one, or leave it
    unset for the host-local bridge topology (the VSS pattern).

Run (plain HTTP; the sandbox reaches it via host.openshell.internal):
  uv run --with mcp --with uvicorn python server.py \
    --workspace-root /home/tao-dev/tao-workspace --host 0.0.0.0 --port 9901
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

NGC_IMAGE_PREFIX = "nvcr.io/"

# The sandbox reaches this server as host.openshell.internal (the OpenShell
# bridge). The MCP SDK's DNS-rebinding guard validates the Host header, so that
# name must be allowlisted or every request fails with "Invalid Host header".
mcp = FastMCP(
    "tao",
    transport_security=TransportSecuritySettings(
        allowed_hosts=["host.openshell.internal", "host.openshell.internal:9901"],
        allowed_origins=["http://host.openshell.internal:9901"],
    ),
)
WORKSPACE_ROOT: Path  # set from the required --workspace-root flag at startup


def _resolve_under_root(subdir: str) -> str:
    """Resolve a caller-supplied subpath under the workspace root, or raise."""
    if subdir in ("", "."):
        target = WORKSPACE_ROOT
    else:
        if Path(subdir).is_absolute():
            raise ValueError(f"path must be relative to the workspace root: {subdir}")
        target = (WORKSPACE_ROOT / subdir).resolve()
    root = WORKSPACE_ROOT.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path escapes the workspace root: {subdir}")
    return str(target)


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=120
    )


@mcp.tool()
def tao_ls(subdir: str = "") -> dict:
    """List the host workspace so the agent can find datasets and results.

    subdir: path relative to the workspace root (the container's /data). Returns
    entries with name, type (file/dir), and size in bytes. Read-only; confined
    to the workspace root.
    """
    target = Path(_resolve_under_root(subdir))
    if not target.exists():
        raise ValueError(f"path does not exist under workspace: {subdir}")
    if target.is_file():
        return {"entries": [{"name": target.name, "type": "file", "size": target.stat().st_size}]}
    entries = []
    for p in sorted(target.iterdir()):
        entries.append({
            "name": p.name,
            "type": "dir" if p.is_dir() else "file",
            "size": p.stat().st_size if p.is_file() else None,
        })
    return {"path": subdir or ".", "entries": entries}


@mcp.tool()
def tao_read(subpath: str, max_bytes: int = 65536) -> dict:
    """Read a small text file from the host workspace (e.g. an annotations or
    spec file) so the agent can inspect data and fill in a training spec.

    subpath: path relative to the workspace root. Returns up to max_bytes of
    UTF-8 text (capped at 1 MiB). Read-only; confined to the workspace root.
    """
    max_bytes = max(1, min(max_bytes, 1_048_576))
    target = Path(_resolve_under_root(subpath))
    if not target.is_file():
        raise ValueError(f"not a file under workspace: {subpath}")
    data = target.read_bytes()[:max_bytes]
    return {
        "subpath": subpath,
        "size": target.stat().st_size,
        "truncated": target.stat().st_size > max_bytes,
        "text": data.decode("utf-8", errors="replace"),
    }


@mcp.tool()
def tao_write(subpath: str, content: str) -> dict:
    """Write a text file into the host workspace (e.g. an experiment spec the
    container will read at /data/... or a state file). Creates parent dirs.
    Confined to the workspace root; refuses to overwrite a directory.

    subpath: path relative to the workspace root.
    content: UTF-8 text to write (capped at 4 MiB).
    """
    if len(content.encode("utf-8")) > 4_194_304:
        raise ValueError("content exceeds 4 MiB")
    target = Path(_resolve_under_root(subpath))
    if target.is_dir():
        raise ValueError(f"path is a directory: {subpath}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"subpath": subpath, "bytes_written": len(content.encode("utf-8"))}


@mcp.tool()
def tao_run(
    image: str,
    command: list[str],
    data_subdir: str = "",
    results_subdir: str = "results",
    gpus: int = 1,
    shm_size: str = "8g",
) -> dict:
    """Launch a TAO training container on the host GPU. Returns a job_id.

    image: full nvcr.io/... image reference (any NVIDIA NGC image — TAO,
      QA/staging, or data-generation).
    command: container command as an argument list (no shell).
    data_subdir/results_subdir: paths relative to the host workspace root,
      mounted at /data and /results in the container. /data is read-write so
      the container can stage pretrained backbones and write state/checkpoints
      alongside the dataset (TAO training expects this).
    gpus: number of GPUs to attach.
    shm_size: /dev/shm size (e.g. "8g", "16g"). TAO PyTorch DataLoaders need a
      large shared-memory segment; the docker default (64m) causes "Bus error"
      / "DataLoader worker exited". Raise for many workers or multi-GPU.
    """
    if not image.startswith(NGC_IMAGE_PREFIX):
        raise ValueError(f"image must start with {NGC_IMAGE_PREFIX}")
    if not isinstance(command, list) or not all(isinstance(a, str) for a in command):
        raise ValueError("command must be a list of strings")
    if not (1 <= gpus <= 8):
        raise ValueError("gpus must be between 1 and 8")
    if not re.fullmatch(r"\d+[bkmg]?", shm_size, re.IGNORECASE):
        raise ValueError("shm_size must look like '8g', '16g', '512m'")

    data = _resolve_under_root(data_subdir)
    results = _resolve_under_root(results_subdir)
    Path(results).mkdir(parents=True, exist_ok=True)

    proc = _docker(
        "run", "-d", "--gpus", str(gpus),
        "--shm-size", shm_size,
        "-v", f"{data}:/data",
        "-v", f"{results}:/results",
        image, *command,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker run failed: {proc.stderr.strip()}")
    return {"job_id": proc.stdout.strip()}


@mcp.tool()
def tao_status(job_id: str) -> dict:
    """Return the state and exit code of a job started by tao_run."""
    proc = _docker(
        "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", job_id
    )
    if proc.returncode != 0:
        raise RuntimeError(f"unknown job_id: {job_id}")
    state, _, exit_code = proc.stdout.strip().partition(" ")
    return {"state": state, "exit_code": int(exit_code)}


@mcp.tool()
def tao_logs(job_id: str, tail: int = 100) -> dict:
    """Return the last `tail` lines of a job's logs."""
    proc = _docker("logs", "--tail", str(tail), job_id)
    if proc.returncode != 0:
        raise RuntimeError(f"unknown job_id: {job_id}")
    return {"text": proc.stdout + proc.stderr}


def _assert_ngc_container(job_id: str) -> None:
    """Confirm job_id is an NGC (nvcr.io) container this server would have
    launched, so the agent cannot stop/remove arbitrary host containers."""
    proc = _docker("inspect", "-f", "{{.Config.Image}}", job_id)
    if proc.returncode != 0:
        raise RuntimeError(f"unknown job_id: {job_id}")
    if not proc.stdout.strip().startswith(NGC_IMAGE_PREFIX):
        raise ValueError(f"refusing: {job_id} is not an nvcr.io container")


@mcp.tool()
def tao_stop(job_id: str) -> dict:
    """Stop a running TAO job. Only nvcr.io containers may be stopped."""
    _assert_ngc_container(job_id)
    proc = _docker("stop", job_id)
    if proc.returncode != 0:
        raise RuntimeError(f"docker stop failed: {proc.stderr.strip()}")
    return {"stopped": proc.stdout.strip()}


@mcp.tool()
def tao_rm(job_id: str, force: bool = False) -> dict:
    """Remove a TAO job's container (frees disk). Only nvcr.io containers may be
    removed. Set force=True to remove one that is still running."""
    _assert_ngc_container(job_id)
    args = ["rm", job_id] if not force else ["rm", "-f", job_id]
    proc = _docker(*args)
    if proc.returncode != 0:
        raise RuntimeError(f"docker rm failed: {proc.stderr.strip()}")
    return {"removed": proc.stdout.strip()}


class BearerAuth(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        self.expected = f"Bearer {token}"

    async def dispatch(self, request, call_next):
        if request.headers.get("authorization") != self.expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


def main() -> None:
    global WORKSPACE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace-root", required=True,
                    help="host directory that confines all data/results mounts")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--ssl-keyfile")
    ap.add_argument("--ssl-certfile")
    args = ap.parse_args()

    WORKSPACE_ROOT = Path(args.workspace_root).resolve()
    if not WORKSPACE_ROOT.is_dir():
        raise SystemExit(f"workspace root does not exist: {WORKSPACE_ROOT}")

    app = mcp.streamable_http_app()
    token = os.environ.get("TAO_MCP_TOKEN")
    if token:  # optional — enforced only when a token is provided
        app.add_middleware(BearerAuth, token=token)
    uvicorn.run(
        app, host=args.host, port=args.port,
        ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile,
    )


if __name__ == "__main__":
    main()
