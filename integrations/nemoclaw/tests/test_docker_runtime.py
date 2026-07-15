#!/usr/bin/env python3
"""Unit tests for the stdlib-only NemoClaw Docker launch helpers."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docker_runtime import (  # noqa: E402
    CONTAINER_HOME,
    HostIdentity,
    build_docker_run_args,
    current_host_identity,
    prepare_runtime_home,
)


class RuntimeHomeTests(unittest.TestCase):
    def test_prepares_writable_home_and_caches_below_results(self):
        identity = HostIdentity(os.getuid(), os.getgid(), "host-user")
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            results = workspace / "experiment" / "results"
            results.mkdir(parents=True)

            home = prepare_runtime_home(results, workspace, identity)

            self.assertEqual(home, results / ".tao-runtime" / "home")
            self.assertTrue(os.access(home, os.W_OK | os.X_OK))
            self.assertEqual(home.stat().st_uid, os.getuid())
            for relative in (
                ".cache/huggingface",
                ".cache/torch",
                ".cache/triton",
                ".cache/torchinductor",
                ".cache/matplotlib",
            ):
                self.assertTrue((home / relative).is_dir())

    def test_rejects_runtime_symlink_that_escapes_results(self):
        identity = HostIdentity(os.getuid(), os.getgid(), "host-user")
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as outside,
        ):
            workspace = Path(tmp)
            results = workspace / "results"
            results.mkdir()
            (results / ".tao-runtime").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must remain under"):
                prepare_runtime_home(results, workspace, identity)


class DockerRunArgsTests(unittest.TestCase):
    def test_maps_host_identity_and_redirects_home_and_caches(self):
        identity = HostIdentity(
            uid=1234,
            gid=5678,
            username="data-scientist",
            supplementary_gids=(99, 100),
        )

        args = build_docker_run_args(
            image="nvcr.io/nvidia/tao/example:1",
            command=["train", "--spec", "/data/spec.yaml"],
            data_dir="/host/workspace/data",
            results_dir="/host/workspace/results",
            gpus=2,
            shm_size="16g",
            identity=identity,
        )

        self.assertEqual(args[:2], ["run", "-d"])
        user_index = args.index("--user")
        self.assertEqual(args[user_index + 1], "1234:5678")
        group_indexes = [i for i, value in enumerate(args) if value == "--group-add"]
        self.assertEqual([args[i + 1] for i in group_indexes], ["99", "100"])
        self.assertIn(f"HOME={CONTAINER_HOME}", args)
        self.assertIn("USER=data-scientist", args)
        self.assertIn("LOGNAME=data-scientist", args)
        self.assertIn(f"XDG_CACHE_HOME={CONTAINER_HOME}/.cache", args)
        self.assertIn("/host/workspace/data:/data", args)
        self.assertIn("/host/workspace/results:/results", args)
        self.assertEqual(
            args[-4:],
            [
                "nvcr.io/nvidia/tao/example:1",
                "train",
                "--spec",
                "/data/spec.yaml",
            ],
        )

    def test_identity_fallback_is_nonempty_when_uid_has_no_passwd_entry(self):
        with patch("docker_runtime.os.getuid", return_value=4242), patch(
            "docker_runtime.os.getgid", return_value=4343
        ), patch(
            "docker_runtime.os.getgroups", return_value=[4343, 99, 100, 99]
        ), patch("docker_runtime.pwd.getpwuid", side_effect=KeyError):
            identity = current_host_identity()

        self.assertEqual(
            identity,
            HostIdentity(4242, 4343, "tao-4242", supplementary_gids=(99, 100)),
        )


if __name__ == "__main__":
    unittest.main()
