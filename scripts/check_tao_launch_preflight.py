#!/usr/bin/env python3
"""Validate TAO launch prerequisites before generating workflow artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SKILL_BANK = Path(
    os.environ.get("TAO_SKILL_BANK_PATH", Path.home() / "tao-skills-external")
)
MANIFEST_REL = Path("platform") / "platforms.manifest.json"
REMOTE_SCHEMES = ("s3://", "azure://", "gs://", "http://", "https://")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-bank",
        type=Path,
        default=DEFAULT_SKILL_BANK,
        help="Path to the packaged TAO skill bank.",
    )
    parser.add_argument("--platform", required=True, help="TAO execution platform.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Dataset/spec path to verify. May be repeated.",
    )
    parser.add_argument(
        "--skip-platform-access",
        action="store_true",
        help="Only validate environment variables and paths.",
    )
    return parser.parse_args()


def load_manifest(skill_bank: Path) -> dict[str, Any]:
    with (skill_bank.expanduser() / MANIFEST_REL).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_platform(skill_bank: Path, requested: str) -> dict[str, Any]:
    normalized = requested.strip().lower()
    for platform in load_manifest(skill_bank).get("platforms", []):
        names = [platform.get("name", "")]
        names.extend(platform.get("aliases", []))
        if normalized in {str(name).lower() for name in names}:
            return platform
    raise SystemExit(f"Unknown platform: {requested}")


def parse_paths(values: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            label, path = value, value
        parsed.append((label.strip() or path.strip(), path.strip()))
    return parsed


def env_missing(platform: dict[str, Any]) -> list[str]:
    missing = []
    for item in platform.get("required_credentials", []):
        name = item.get("name")
        if item.get("source") == "env_var" and name and not os.environ.get(name):
            missing.append(name)
    for group in platform.get("credential_groups", []):
        choices = [name for name in group.get("require_one_of", []) if name]
        if choices and not any(os.environ.get(name) for name in choices):
            missing.append("one of " + ", ".join(choices))
    return missing


def normalize_local_path(path: str) -> str | None:
    if path.startswith(REMOTE_SCHEMES):
        return None
    if path.startswith("file://"):
        return path[len("file://") :]
    if path.startswith("lustre:///"):
        return "/" + path[len("lustre:///") :].lstrip("/")
    return path


def run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "command timed out",
        )


def ssh_command(host: str, remote_command: str) -> list[str]:
    user = os.environ["SLURM_USER"]
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
    ]
    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path:
        command.extend(
            ["-i", str(Path(key_path).expanduser()), "-o", "IdentitiesOnly=yes"]
        )
    command.extend([f"{user}@{host}", remote_command])
    return command


def check_slurm(
    platform: dict[str, Any],
    paths: list[tuple[str, str]],
    skip_access: bool,
) -> bool:
    ok = True
    missing = env_missing(platform)
    if missing:
        print("Missing SLURM requirement(s): " + ", ".join(missing))
        return False

    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path and not Path(key_path).expanduser().exists():
        print(f"SSH_KEY_PATH does not exist: {key_path}")
        return False

    hosts = [host.strip() for host in os.environ["SLURM_HOSTNAME"].split(",") if host.strip()]
    if not hosts:
        print("SLURM_HOSTNAME did not contain any hosts")
        return False

    working_host = ""
    if not skip_access:
        for host in hosts:
            try:
                socket.getaddrinfo(host, 22)
            except socket.gaierror as exc:
                print(f"Host did not resolve: {host} ({exc})")
                continue
            result = run(ssh_command(host, "echo TAO_SSH_OK"), timeout=25)
            if result.returncode == 0 and "TAO_SSH_OK" in result.stdout:
                working_host = host
                print(f"Passwordless SSH OK: {host}")
                break
            reason = (result.stderr or result.stdout).strip().splitlines()
            detail = reason[-1] if reason else "exit " + str(result.returncode)
            print(f"Passwordless SSH failed: {host}: {detail}")

        if not working_host:
            print(
                "SLURM preflight failed before artifact generation. Install the public key with "
                "ssh-copy-id, fix key permissions with chmod 600, trust the host key, or start "
                "ssh-agent and provide SSH_AUTH_SOCK."
            )
            return False
    else:
        working_host = hosts[0]

    for label, raw_path in paths:
        path = normalize_local_path(raw_path)
        if path is None:
            print(f"Skipped remote object-store path check for {label}: {raw_path}")
            continue
        if not path.startswith("/"):
            print(f"SLURM dataset path is not absolute for {label}: {raw_path}")
            ok = False
            continue
        if skip_access:
            continue
        result = run(ssh_command(working_host, f"test -e {shlex.quote(path)}"), timeout=25)
        if result.returncode == 0:
            print(f"Remote path OK: {label}={path}")
        else:
            print(f"Remote path missing or inaccessible: {label}={path}")
            ok = False

    return ok


def check_local_docker(paths: list[tuple[str, str]], skip_access: bool) -> bool:
    ok = True
    if not skip_access:
        if not shutil.which("docker"):
            print("docker executable not found")
            ok = False
        else:
            result = run(["docker", "info"], timeout=20)
            if result.returncode == 0:
                print("Docker daemon OK")
            else:
                print("Docker daemon check failed")
                ok = False

    for label, raw_path in paths:
        path = normalize_local_path(raw_path)
        if path is None:
            print(f"Skipped remote object-store path check for {label}: {raw_path}")
            continue
        if Path(path).exists():
            print(f"Local path OK: {label}={path}")
        else:
            print(f"Local path missing: {label}={path}")
            ok = False
    return ok


def check_env_only(platform: dict[str, Any], paths: list[tuple[str, str]]) -> bool:
    missing = env_missing(platform)
    if missing:
        print("Missing requirement(s): " + ", ".join(missing))
        return False
    for label, raw_path in paths:
        if raw_path.startswith(REMOTE_SCHEMES):
            print(f"Path accepted for remote platform: {label}={raw_path}")
        else:
            print(
                "Path provided for remote platform, verify it is mounted in the "
                f"job: {label}={raw_path}"
            )
    return True


def main() -> int:
    args = parse_args()
    platform = resolve_platform(args.skill_bank, args.platform)
    paths = parse_paths(args.path)
    name = platform["name"]

    if name == "slurm":
        ok = check_slurm(platform, paths, args.skip_platform_access)
    elif name == "local-docker":
        ok = check_local_docker(paths, args.skip_platform_access)
    else:
        ok = check_env_only(platform, paths)

    if ok:
        print("TAO launch preflight passed")
        return 0
    print("TAO launch preflight failed")
    return 2


if __name__ == "__main__":
    sys.exit(main())
