#!/usr/bin/env python3
"""Stdlib-only Docker launch helpers for the NemoClaw TAO bridge.

Keeping command construction here lets ownership and path-confinement behavior
be tested without importing the MCP or ASGI dependencies used by ``server.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import pwd
from typing import Sequence


CONTAINER_HOME = "/results/.tao-runtime/home"


@dataclass(frozen=True)
class HostIdentity:
    """Numeric host identity used for files written through bind mounts."""

    uid: int
    gid: int
    username: str
    supplementary_gids: tuple[int, ...] = ()


def current_host_identity() -> HostIdentity:
    """Return the identity of the host process running the MCP server."""
    uid = os.getuid()
    gid = os.getgid()
    try:
        username = pwd.getpwuid(uid).pw_name
    except KeyError:
        # USER/LOGNAME still need a stable, non-empty value in containers whose
        # passwd database does not contain the numeric host uid.
        username = f"tao-{uid}"
    supplementary_gids = tuple(
        group_id
        for group_id in sorted({int(group) for group in os.getgroups()})
        if group_id != gid
    )
    return HostIdentity(
        uid=uid,
        gid=gid,
        username=username,
        supplementary_gids=supplementary_gids,
    )


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def prepare_runtime_home(
    results_dir: str | Path,
    workspace_root: str | Path,
    identity: HostIdentity,
) -> Path:
    """Create a host-owned writable HOME/cache tree below ``results_dir``.

    Both paths are resolved again here so a pre-existing ``.tao-runtime``
    symlink cannot redirect the cache mount outside the configured workspace or
    results tree. Existing root-owned results fail early with a useful error
    instead of letting training fail later or create more inaccessible files.
    """
    workspace = Path(workspace_root).resolve()
    results = Path(results_dir).resolve()
    if not _is_within(results, workspace):
        raise ValueError(f"results directory escapes workspace root: {results}")
    if not os.access(results, os.W_OK | os.X_OK):
        raise PermissionError(
            f"results directory is not writable by host uid:gid "
            f"{identity.uid}:{identity.gid}: {results}. Normalize its ownership "
            "before launching another TAO container."
        )

    runtime_root = (results / ".tao-runtime").resolve()
    home = (runtime_root / "home").resolve()
    if not _is_within(runtime_root, results) or not _is_within(home, results):
        raise ValueError(
            f"runtime HOME must remain under the results directory: {home}"
        )

    try:
        runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        home.mkdir(mode=0o700, exist_ok=True)
        for relative in (
            ".cache/huggingface",
            ".cache/torch",
            ".cache/triton",
            ".cache/torchinductor",
            ".cache/matplotlib",
            ".config",
            ".local",
        ):
            cache_dir = (home / relative).resolve()
            if not _is_within(cache_dir, home):
                raise ValueError(
                    f"runtime cache path must remain under runtime HOME: {cache_dir}"
                )
            cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"results directory is not writable by host uid:gid "
            f"{identity.uid}:{identity.gid}: {results}. Normalize its ownership "
            "before launching another TAO container."
        ) from exc

    if not os.access(home, os.W_OK | os.X_OK):
        raise PermissionError(
            f"runtime HOME is not writable by host uid:gid "
            f"{identity.uid}:{identity.gid}: {home}. Normalize the results "
            "directory ownership before launching another TAO container."
        )
    return home


def build_docker_run_args(
    *,
    image: str,
    command: Sequence[str],
    data_dir: str,
    results_dir: str,
    gpus: int,
    shm_size: str,
    identity: HostIdentity,
) -> list[str]:
    """Build arguments passed after the ``docker`` executable.

    The numeric user mapping is the ownership invariant: anything written to a
    writable bind mount is owned by the same host user that runs this bridge.
    Explicit HOME/cache redirects avoid writes to image-owned locations such as
    ``/root`` when the image's baked-in user is overridden.
    """
    environment = {
        "HOME": CONTAINER_HOME,
        "USER": identity.username,
        "LOGNAME": identity.username,
        "XDG_CACHE_HOME": f"{CONTAINER_HOME}/.cache",
        "HF_HOME": f"{CONTAINER_HOME}/.cache/huggingface",
        "TORCH_HOME": f"{CONTAINER_HOME}/.cache/torch",
        "TRITON_CACHE_DIR": f"{CONTAINER_HOME}/.cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{CONTAINER_HOME}/.cache/torchinductor",
        "MPLCONFIGDIR": f"{CONTAINER_HOME}/.cache/matplotlib",
    }

    args = [
        "run",
        "-d",
        "--gpus",
        str(gpus),
        "--user",
        f"{identity.uid}:{identity.gid}",
    ]
    for group_id in identity.supplementary_gids:
        args.extend(("--group-add", str(group_id)))
    args.extend(("--shm-size", shm_size))
    for name, value in environment.items():
        args.extend(("-e", f"{name}={value}"))
    args.extend(
        (
            "-v",
            f"{data_dir}:/data",
            "-v",
            f"{results_dir}:/results",
            image,
            *command,
        )
    )
    return args
