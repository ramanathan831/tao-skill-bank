#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate TAO launch prerequisites before generating workflow artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
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
MANIFEST_REL = Path("skills") / "platform" / "platforms.manifest.json"
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
        "--json-required-field",
        action="append",
        default=[],
        metavar="LABEL=FIELD[,FIELD...]",
        help=(
            "Require one or more top-level fields in sample records from a JSON "
            "annotation file identified by LABEL. May be repeated."
        ),
    )
    parser.add_argument(
        "--json-sample-limit",
        type=int,
        default=20,
        help="Number of JSON annotation records to sample for required fields.",
    )
    parser.add_argument(
        "--gpu-arch-allowlist",
        action="append",
        default=[],
        metavar="LABEL=SM[,SM...]",
        help=(
            "Require target GPU architectures to be supported by a model/image, "
            "for example cosmos_rl=sm_80,sm_90,sm_100,sm_120."
        ),
    )
    parser.add_argument(
        "--gpu-arch",
        action="append",
        default=[],
        metavar="SM",
        help=(
            "Known target GPU architecture such as sm_90 or 12.0. May be repeated. "
            "If omitted, local nvidia-smi is used when an allowlist is provided."
        ),
    )
    parser.add_argument(
        "--gpu-min-count",
        type=int,
        default=None,
        help="Require at least this many target GPUs before launch.",
    )
    parser.add_argument(
        "--gpu-min-memory-gb",
        type=float,
        default=None,
        help="Require each counted target GPU to have at least this much memory in GiB.",
    )
    parser.add_argument(
        "--target-gpu-count",
        type=int,
        default=None,
        help="Known target GPU count when nvidia-smi is not available on the launch host.",
    )
    parser.add_argument(
        "--target-gpu-memory-gb",
        action="append",
        type=float,
        default=[],
        help=(
            "Known target GPU memory in GiB. May be repeated once per GPU, or "
            "provided once with --target-gpu-count to apply to all target GPUs."
        ),
    )
    parser.add_argument(
        "--effective-batch-limit",
        action="append",
        default=[],
        metavar="LABEL=BATCH_SIZE,SHARD_COUNT",
        help=(
            "Require BATCH_SIZE <= JSON record_count / SHARD_COUNT for a "
            "previously supplied annotation path label. May be repeated."
        ),
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


def parse_required_fields(values: list[str]) -> dict[str, list[str]]:
    fields_by_label: dict[str, list[str]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                "--json-required-field must use LABEL=FIELD[,FIELD...] syntax"
            )
        label, fields = value.split("=", 1)
        label = label.strip()
        parsed_fields = [field.strip() for field in fields.split(",") if field.strip()]
        if not label or not parsed_fields:
            raise SystemExit(
                "--json-required-field must include a label and at least one field"
            )
        fields_by_label.setdefault(label, []).extend(parsed_fields)
    return fields_by_label


def normalize_gpu_arch(value: str) -> str:
    normalized = value.strip().lower().replace("compute_", "sm_")
    normalized = normalized.replace("-", "_")
    if not normalized:
        raise SystemExit("GPU architecture value must not be empty")
    if re.fullmatch(r"sm_?\d{2,3}", normalized):
        digits = normalized.split("_", 1)[-1] if "_" in normalized else normalized[2:]
        return "sm_" + digits
    if re.fullmatch(r"\d{2,3}", normalized):
        return "sm_" + normalized
    match = re.fullmatch(r"(\d+)\.(\d+)", normalized)
    if match:
        major, minor = match.groups()
        return f"sm_{major}{minor}"
    raise SystemExit(
        f"Unsupported GPU architecture format: {value}. Use sm_90, 90, or 9.0."
    )


def parse_gpu_arch_allowlists(values: list[str]) -> dict[str, set[str]]:
    allowlists: dict[str, set[str]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--gpu-arch-allowlist must use LABEL=SM[,SM...] syntax")
        label, raw_arches = value.split("=", 1)
        label = label.strip()
        arches = {
            normalize_gpu_arch(arch)
            for arch in raw_arches.split(",")
            if arch.strip()
        }
        if not label or not arches:
            raise SystemExit(
                "--gpu-arch-allowlist must include a label and at least one SM value"
            )
        allowlists[label] = arches
    return allowlists


def parse_effective_batch_limits(values: list[str]) -> dict[str, list[tuple[int, int]]]:
    limits: dict[str, list[tuple[int, int]]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                "--effective-batch-limit must use LABEL=BATCH_SIZE,SHARD_COUNT syntax"
            )
        label, raw_values = value.split("=", 1)
        parts = [part.strip() for part in raw_values.split(",") if part.strip()]
        if len(parts) != 2:
            raise SystemExit(
                "--effective-batch-limit must include exactly BATCH_SIZE,SHARD_COUNT"
            )
        try:
            batch_size = int(parts[0])
            shard_count = int(parts[1])
        except ValueError as exc:
            raise SystemExit(
                "--effective-batch-limit values must be integers"
            ) from exc
        if batch_size <= 0 or shard_count <= 0:
            raise SystemExit(
                "--effective-batch-limit values must be positive integers"
            )
        limits.setdefault(label.strip(), []).append((batch_size, shard_count))
    return limits


def env_missing(platform: dict[str, Any]) -> list[str]:
    missing = []
    for item in platform.get("required_credentials", []):
        name = item.get("name")
        if item.get("source") == "env_var" and name and not os.environ.get(name):
            missing.append(name)
    for group in platform.get("credential_groups", []):
        choices = [name for name in group.get("require_one_of", []) if name]
        if choices and not any(os.environ.get(name) for name in choices):
            preferred = group.get("preferred")
            if preferred:
                missing.append(f"{preferred} ({group.get('description', 'required')})")
            else:
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


def run(
    command: list[str],
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "command timed out",
        )


def detect_local_gpu_arches() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    result = run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        timeout=20,
    )
    if result.returncode != 0:
        return []
    arches = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        arches.append(normalize_gpu_arch(value))
    return arches


def detect_local_gpu_memory_gb() -> list[float]:
    if not shutil.which("nvidia-smi"):
        return []
    result = run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        timeout=20,
    )
    if result.returncode != 0:
        return []
    memory_gb = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            memory_gb.append(float(value) / 1024.0)
        except ValueError:
            continue
    return memory_gb


def check_gpu_arch_allowlists(
    allowlists: dict[str, set[str]],
    provided_arches: list[str],
    skip_access: bool,
) -> bool:
    if not allowlists:
        return True

    if provided_arches:
        target_arches = [normalize_gpu_arch(arch) for arch in provided_arches]
    elif skip_access:
        labels = ", ".join(sorted(allowlists))
        print(
            "GPU architecture allowlist present but target GPU detection was skipped: "
            f"{labels}. Provide --gpu-arch sm_XX when the target architecture is known."
        )
        return True
    else:
        target_arches = detect_local_gpu_arches()

    if not target_arches:
        labels = ", ".join(sorted(allowlists))
        print(
            "GPU architecture check failed: could not detect target GPU architecture "
            f"for {labels}. Run on the target GPU host or provide --gpu-arch sm_XX."
        )
        return False

    ok = True
    for label, allowed in allowlists.items():
        label_ok = True
        for arch in target_arches:
            if arch not in allowed:
                print(
                    f"GPU architecture unsupported for {label}: target={arch}, "
                    f"allowed={','.join(sorted(allowed))}"
                )
                label_ok = False
                ok = False
        if label_ok:
            print(
                f"GPU architecture OK for {label}: "
                f"target={','.join(target_arches)}, allowed={','.join(sorted(allowed))}"
            )
    return ok


def check_gpu_resources(
    min_count: int | None,
    min_memory_gb: float | None,
    target_count: int | None,
    target_memory_gb: list[float],
    skip_access: bool,
) -> bool:
    if min_count is None and min_memory_gb is None:
        return True
    if min_count is not None and min_count <= 0:
        raise SystemExit("--gpu-min-count must be a positive integer")
    if min_memory_gb is not None and min_memory_gb <= 0:
        raise SystemExit("--gpu-min-memory-gb must be positive")

    if target_memory_gb:
        memory_gb = list(target_memory_gb)
        if target_count and len(memory_gb) == 1:
            memory_gb = memory_gb * target_count
    elif target_count:
        memory_gb = [0.0] * target_count
    elif skip_access:
        print(
            "GPU resource requirement present but target GPU detection was skipped. "
            "Provide --target-gpu-count and --target-gpu-memory-gb when the target "
            "hardware is known."
        )
        return True
    else:
        memory_gb = detect_local_gpu_memory_gb()

    if not memory_gb:
        print(
            "GPU resource check failed: could not detect target GPU memory/count. "
            "Run on the target GPU host or provide --target-gpu-count and "
            "--target-gpu-memory-gb."
        )
        return False

    if min_memory_gb is None:
        qualifying = len(memory_gb)
    else:
        qualifying = sum(1 for value in memory_gb if value >= min_memory_gb)

    required_count = min_count or 1
    if qualifying < required_count:
        detected = ",".join(f"{value:.1f}GiB" for value in memory_gb)
        print(
            "GPU resource check failed: "
            f"qualifying_gpus={qualifying} < required={required_count}; "
            f"min_memory_gb={min_memory_gb if min_memory_gb is not None else 'any'}; "
            f"detected={detected}"
        )
        return False

    detected = ",".join(f"{value:.1f}GiB" for value in memory_gb)
    print(
        "GPU resources OK: "
        f"qualifying_gpus={qualifying}, required={required_count}, "
        f"min_memory_gb={min_memory_gb if min_memory_gb is not None else 'any'}, "
        f"detected={detected}"
    )
    return True


JSON_FIELD_CHECK_SCRIPT = r"""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
fields = [field for field in sys.argv[2].split(",") if field]
limit = int(sys.argv[3])

with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)

if isinstance(payload, list):
    records = payload
elif isinstance(payload, dict):
    for key in ("annotations", "data", "samples", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            records = value
            break
    else:
        records = [payload]
else:
    raise SystemExit(f"unsupported JSON top-level type: {type(payload).__name__}")

if not records:
    raise SystemExit("annotation JSON has no records")

checked = records[: max(1, min(limit, len(records)))]
missing = []
for index, record in enumerate(checked):
    if not isinstance(record, dict):
        missing.append(f"record {index}: not an object")
        continue
    absent = [field for field in fields if field not in record]
    if absent:
        missing.append(f"record {index}: missing {','.join(absent)}")

if missing:
    raise SystemExit("; ".join(missing[:5]))

print(f"checked={len(checked)} fields={','.join(fields)}")
"""


def check_json_required_fields_local(
    label: str,
    path: str,
    fields: list[str],
    sample_limit: int,
) -> bool:
    command = [
        sys.executable,
        "-c",
        JSON_FIELD_CHECK_SCRIPT,
        path,
        ",".join(fields),
        str(sample_limit),
    ]
    result = run(command, timeout=30)
    if result.returncode == 0:
        print(f"JSON fields OK: {label}={path}: {result.stdout.strip()}")
        return True
    reason = (result.stderr or result.stdout).strip().splitlines()
    detail = reason[-1] if reason else "exit " + str(result.returncode)
    print(f"JSON fields missing or invalid: {label}={path}: {detail}")
    return False


def check_json_required_fields_remote(
    host: str,
    label: str,
    path: str,
    fields: list[str],
    sample_limit: int,
) -> bool:
    remote_command = " ".join(
        [
            "python3",
            "-c",
            shlex.quote(JSON_FIELD_CHECK_SCRIPT),
            shlex.quote(path),
            shlex.quote(",".join(fields)),
            str(sample_limit),
        ]
    )
    result = run(ssh_command(host, remote_command), timeout=45)
    if result.returncode == 0:
        print(f"Remote JSON fields OK: {label}={path}: {result.stdout.strip()}")
        return True
    reason = (result.stderr or result.stdout).strip().splitlines()
    detail = reason[-1] if reason else "exit " + str(result.returncode)
    print(f"Remote JSON fields missing or invalid: {label}={path}: {detail}")
    return False


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


def s3_paths(paths: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(label, path) for label, path in paths if path.startswith("s3://")]


def has_unverified_remote_mounts(
    platform_name: str,
    paths: list[tuple[str, str]],
    skip_access: bool,
) -> bool:
    if skip_access or platform_name in {"slurm", "local-docker"}:
        return False

    failed = False
    for label, raw_path in paths:
        if raw_path.startswith(REMOTE_SCHEMES):
            continue
        print(
            f"{platform_name} path requires mounted-volume proof before launch: "
            f"{label}={raw_path}. Use s3://, or verify the mount/PVC/volume from "
            "the platform and rerun only after that manual proof exists."
        )
        failed = True
    return failed


def check_s3_storage(paths: list[tuple[str, str]], skip_access: bool) -> bool:
    targets = s3_paths(paths)
    if not targets:
        return True

    missing = [key for key in ("ACCESS_KEY", "SECRET_KEY") if not os.environ.get(key)]
    if missing:
        print("Missing S3 requirement(s): " + ", ".join(missing))
        return False

    if skip_access:
        print("S3 credentials are present; skipped object-store access checks.")
        return True

    aws = shutil.which("aws")
    if not aws:
        print(
            "aws CLI not found, so s3:// dataset paths cannot be verified. "
            "After user approval, install it with: python -m pip install awscli. "
            "Alternatively, manually prove the paths are readable before launch."
        )
        return False

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY"]
    env["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_KEY"]
    env.setdefault("AWS_DEFAULT_REGION", os.environ.get("CLOUD_REGION", "us-east-1"))

    ok = True
    for label, uri in targets:
        command = [aws]
        if os.environ.get("S3_ENDPOINT_URL"):
            command.extend(["--endpoint-url", os.environ["S3_ENDPOINT_URL"]])
        command.extend(["s3", "ls", uri])
        result = run(command, timeout=45, env=env)
        if result.returncode == 0:
            print(f"S3 path OK: {label}={uri}")
        else:
            detail = (result.stderr or result.stdout).strip().splitlines()
            reason = detail[-1] if detail else "exit " + str(result.returncode)
            print(f"S3 path missing or inaccessible: {label}={uri}: {reason}")
            ok = False
    return ok


def load_json_payload(path: str) -> Any:
    if path.startswith("s3://"):
        aws = shutil.which("aws")
        if not aws:
            raise RuntimeError(
                "aws CLI not found. After user approval, install it with: "
                "python -m pip install awscli"
            )
        missing = [key for key in ("ACCESS_KEY", "SECRET_KEY") if not os.environ.get(key)]
        if missing:
            raise RuntimeError("Missing S3 requirement(s): " + ", ".join(missing))
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = os.environ["ACCESS_KEY"]
        env["AWS_SECRET_ACCESS_KEY"] = os.environ["SECRET_KEY"]
        env.setdefault("AWS_DEFAULT_REGION", os.environ.get("CLOUD_REGION", "us-east-1"))
        command = [aws]
        if os.environ.get("S3_ENDPOINT_URL"):
            command.extend(["--endpoint-url", os.environ["S3_ENDPOINT_URL"]])
        command.extend(["s3", "cp", path, "-"])
        result = run(command, timeout=60, env=env)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            reason = detail[-1] if detail else "exit " + str(result.returncode)
            raise RuntimeError(f"S3 annotation download failed: {reason}")
        return json.loads(result.stdout)

    local_path = normalize_local_path(path)
    if local_path is None:
        raise RuntimeError(f"Cannot count records for remote path: {path}")
    with Path(local_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_record_count(path: str) -> int:
    payload = load_json_payload(path)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("annotations", "data", "samples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            records = [payload]
    else:
        raise RuntimeError(f"unsupported JSON top-level type: {type(payload).__name__}")
    if not records:
        raise RuntimeError("annotation JSON has no records")
    return len(records)


def check_effective_batch_limits(
    paths: list[tuple[str, str]],
    limits: dict[str, list[tuple[int, int]]],
    skip_access: bool,
) -> bool:
    if not limits:
        return True
    if skip_access:
        print(
            "Effective batch limits present; skipped annotation record-count checks."
        )
        return True

    path_by_label = dict(paths)
    ok = True
    count_cache: dict[str, int] = {}
    for label, label_limits in limits.items():
        path = path_by_label.get(label)
        if not path:
            print(f"Effective batch check failed: no --path found for label {label}")
            ok = False
            continue
        try:
            count = count_cache.setdefault(path, json_record_count(path))
        except Exception as exc:
            print(f"Effective batch check failed: {label}={path}: {exc}")
            ok = False
            continue
        for batch_size, shard_count in label_limits:
            max_batch = count / shard_count
            if batch_size > max_batch:
                print(
                    "Effective batch check failed: "
                    f"{label} records={count}, batch_size={batch_size}, "
                    f"shard_count={shard_count}, max_batch_per_replica={max_batch:g}"
                )
                ok = False
            else:
                print(
                    "Effective batch OK: "
                    f"{label} records={count}, batch_size={batch_size}, "
                    f"shard_count={shard_count}, max_batch_per_replica={max_batch:g}"
                )
    return ok


def check_brev(platform: dict[str, Any], skip_access: bool) -> bool:
    missing = env_missing(platform)
    if missing:
        print("Missing Brev requirement(s): " + ", ".join(missing))
        return False
    if skip_access:
        print("Brev credentials are present; skipped CLI/API access check.")
        return True

    brev = shutil.which("brev")
    if not brev:
        print("brev CLI not found. Install from https://docs.nvidia.com/brev/.")
        return False

    token = os.environ.get("BREV_API_TOKEN")
    if token:
        login = run([brev, "login", "--token", token], timeout=45)
        if login.returncode != 0:
            detail = (login.stderr or login.stdout).strip().splitlines()
            reason = detail[-1] if detail else "exit " + str(login.returncode)
            print(f"Brev token login failed: {reason}")
            return False

    result = run([brev, "ls", "--json"], timeout=60)
    if result.returncode != 0 and token:
        # Headless `brev ls` occasionally hits an auth-EOF even after a
        # successful token login — the cached session desyncs. Force one
        # refresh and retry before declaring failure.
        run([brev, "login", "--token", token], timeout=45)
        result = run([brev, "ls", "--json"], timeout=60)
    if result.returncode == 0:
        print("Brev CLI/API OK")
        return True
    detail = (result.stderr or result.stdout).strip().splitlines()
    reason = detail[-1] if detail else "exit " + str(result.returncode)
    print(f"Brev CLI/API check failed: {reason}")
    return False


def kubectl_base() -> list[str]:
    command = ["kubectl"]
    if os.environ.get("KUBECONFIG"):
        command.extend(["--kubeconfig", os.environ["KUBECONFIG"]])
    if os.environ.get("TAO_K8S_CONTEXT"):
        command.extend(["--context", os.environ["TAO_K8S_CONTEXT"]])
    return command


def check_kubernetes(platform: dict[str, Any], skip_access: bool) -> bool:
    missing = env_missing(platform)
    if missing:
        print("Missing Kubernetes requirement(s): " + ", ".join(missing))
        return False
    if skip_access:
        print("Kubernetes environment is present; skipped cluster access check.")
        return True

    if not shutil.which("kubectl"):
        print("kubectl not found. Install kubectl or run from inside the cluster.")
        return False

    namespace = os.environ.get("TAO_K8S_NAMESPACE", "default")
    base = kubectl_base()
    auth = run(base + ["auth", "can-i", "create", "jobs", "-n", namespace], timeout=30)
    if auth.returncode != 0 or auth.stdout.strip().lower() != "yes":
        detail = (auth.stderr or auth.stdout).strip().splitlines()
        reason = detail[-1] if detail else "not allowed"
        print(f"Kubernetes job-create permission check failed: {reason}")
        return False

    nodes = run(base + ["get", "nodes", "-o", "json"], timeout=45)
    if nodes.returncode != 0:
        detail = (nodes.stderr or nodes.stdout).strip().splitlines()
        reason = detail[-1] if detail else "exit " + str(nodes.returncode)
        print(f"Kubernetes node/GPU check failed: {reason}")
        return False
    try:
        payload = json.loads(nodes.stdout)
        gpu_total = 0
        for node in payload.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            value = allocatable.get("nvidia.com/gpu", "0")
            gpu_total += int(str(value))
    except Exception as exc:
        print(f"Kubernetes node/GPU check failed: could not parse nodes JSON: {exc}")
        return False
    if gpu_total <= 0:
        print("Kubernetes node/GPU check failed: no allocatable nvidia.com/gpu found")
        return False
    print(f"Kubernetes API OK: namespace={namespace}, allocatable_gpus={gpu_total}")
    return True


def check_slurm(
    platform: dict[str, Any],
    paths: list[tuple[str, str]],
    required_json_fields: dict[str, list[str]],
    json_sample_limit: int,
    skip_access: bool,
) -> bool:
    ok = True
    missing = env_missing(platform)
    if missing:
        print("Missing SLURM requirement(s): " + ", ".join(missing))
        if any("SSH_KEY_PATH" in item for item in missing):
            print(
                "Provide SSH_KEY_PATH=/path/to/private_key. To set up passwordless "
                "access: 1) ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519; "
                "2) ssh-copy-id -i ~/.ssh/id_ed25519.pub $SLURM_USER@<login-host>; "
                "3) ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts; "
                "4) chmod 600 ~/.ssh/id_ed25519; 5) verify with "
                "ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 "
                "$SLURM_USER@<login-host> 'hostname'."
            )
        return False

    key_path = os.environ.get("SSH_KEY_PATH")
    if key_path and not Path(key_path).expanduser().exists():
        print(f"SSH_KEY_PATH does not exist: {key_path}")
        return False

    hosts = [host.strip() for host in os.environ["SLURM_HOSTNAME"].split(",") if host.strip()]
    if not hosts:
        print("SLURM_HOSTNAME did not contain any hosts")
        return False

    if not check_slurm_runtime(platform):
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
                "SLURM preflight failed before artifact generation. Provide "
                "SSH_KEY_PATH=/path/to/private_key for a key accepted by at least "
                "one SLURM login host. To set up passwordless access: "
                "1) create a key if needed: ssh-keygen -t ed25519 -N '' -f "
                "~/.ssh/id_ed25519; 2) install it once: ssh-copy-id -i "
                "~/.ssh/id_ed25519.pub $SLURM_USER@<login-host>; 3) trust the "
                "host key: ssh-keyscan -H <login-host> >> ~/.ssh/known_hosts; "
                "4) lock permissions: chmod 600 ~/.ssh/id_ed25519; 5) verify: "
                "ssh -o BatchMode=yes -i ~/.ssh/id_ed25519 "
                "$SLURM_USER@<login-host> 'hostname'. Then rerun with "
                "SSH_KEY_PATH set to the private key path."
            )
            return False
    else:
        working_host = hosts[0]

    for label, raw_path in paths:
        path = normalize_local_path(raw_path)
        if path is None:
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
            continue
        fields = required_json_fields.get(label)
        if fields and not check_json_required_fields_remote(
            working_host,
            label,
            path,
            fields,
            json_sample_limit,
        ):
            ok = False

    return ok


def parse_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise SystemExit(f"{name} must be a number of hours, got: {value}")


def check_slurm_runtime(platform: dict[str, Any]) -> bool:
    defaults = platform.get("resource_defaults", {})
    default_partition = defaults.get("partition")
    default_time = float(defaults.get("time_hours", 4))
    default_timeout = float(defaults.get("timeout_hours", max(default_time - 0.2, 0.1)))
    max_time = float(defaults.get("max_time_hours", default_time))

    requested_partition = os.environ.get("SLURM_PARTITION", default_partition)
    requested_time = parse_float_env("SLURM_TIME_HOURS")
    requested_timeout = parse_float_env("SLURM_TIMEOUT_HOURS")

    if requested_time is None:
        print(
            "SLURM runtime default: "
            f"partition={requested_partition}, SLURM_TIME_HOURS={default_time:g}, "
            f"SLURM_TIMEOUT_HOURS={default_timeout:g}"
        )
        requested_time = default_time
    else:
        print(
            "SLURM runtime requested: "
            f"partition={requested_partition}, SLURM_TIME_HOURS={requested_time:g}"
        )

    if requested_timeout is None:
        requested_timeout = min(default_timeout, max(requested_time - 0.1, 0.1))
    print(f"SLURM internal timeout: SLURM_TIMEOUT_HOURS={requested_timeout:g}")

    if requested_time > max_time:
        print(
            "SLURM runtime exceeds packaged partition limit: "
            f"requested {requested_time:g}h > max {max_time:g}h. "
            "Use SLURM_TIME_HOURS=4 or provide a different partition with a "
            "known-good wall-time limit before launch."
        )
        return False
    if requested_timeout >= requested_time:
        print(
            "SLURM_TIMEOUT_HOURS must be smaller than SLURM_TIME_HOURS "
            f"({requested_timeout:g} >= {requested_time:g})."
        )
        return False
    return True


def check_local_docker(
    paths: list[tuple[str, str]],
    required_json_fields: dict[str, list[str]],
    json_sample_limit: int,
    skip_access: bool,
) -> bool:
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
            continue
        if Path(path).exists():
            print(f"Local path OK: {label}={path}")
        else:
            print(f"Local path missing: {label}={path}")
            ok = False
            continue
        fields = required_json_fields.get(label)
        if fields and not check_json_required_fields_local(
            label,
            path,
            fields,
            json_sample_limit,
        ):
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
    required_json_fields = parse_required_fields(args.json_required_field)
    gpu_arch_allowlists = parse_gpu_arch_allowlists(args.gpu_arch_allowlist)
    effective_batch_limits = parse_effective_batch_limits(args.effective_batch_limit)
    name = platform["name"]

    if name == "slurm":
        platform_ok = check_slurm(
            platform,
            paths,
            required_json_fields,
            args.json_sample_limit,
            args.skip_platform_access,
        )
    elif name == "local-docker":
        platform_ok = check_local_docker(
            paths,
            required_json_fields,
            args.json_sample_limit,
            args.skip_platform_access,
        )
    elif name == "brev":
        platform_ok = check_brev(platform, args.skip_platform_access)
    elif name == "kubernetes":
        platform_ok = check_kubernetes(platform, args.skip_platform_access)
    else:
        platform_ok = check_env_only(platform, paths)

    storage_ok = check_s3_storage(paths, args.skip_platform_access)
    mounts_ok = not has_unverified_remote_mounts(
        name,
        paths,
        args.skip_platform_access,
    )
    gpu_arch_ok = check_gpu_arch_allowlists(
        gpu_arch_allowlists,
        args.gpu_arch,
        args.skip_platform_access,
    )
    gpu_resources_ok = check_gpu_resources(
        args.gpu_min_count,
        args.gpu_min_memory_gb,
        args.target_gpu_count,
        args.target_gpu_memory_gb,
        args.skip_platform_access,
    )
    effective_batch_ok = check_effective_batch_limits(
        paths,
        effective_batch_limits,
        args.skip_platform_access,
    )
    ok = (
        platform_ok
        and storage_ok
        and mounts_ok
        and gpu_arch_ok
        and gpu_resources_ok
        and effective_batch_ok
    )

    if ok:
        print("TAO launch preflight passed")
        return 0
    print("TAO launch preflight failed")
    return 2


if __name__ == "__main__":
    sys.exit(main())
