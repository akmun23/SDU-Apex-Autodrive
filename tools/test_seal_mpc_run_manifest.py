import subprocess
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.seal_mpc_run_manifest import (
    assert_containers_stopped,
    docker_environment,
    image_provenance,
    map_record,
    parse_pose,
    workspace_manifest,
)


class RunManifestTest(unittest.TestCase):
    def test_parses_finite_numeric_initial_pose(self):
        self.assertEqual(parse_pose('{"x": 1, "y": -2.5, "yaw": 0.3}'),
                         {"x": 1.0, "y": -2.5, "yaw": 0.3})

    def test_rejects_incomplete_non_numeric_or_non_finite_pose(self):
        for raw in ('{}', '{"x": 0, "y": 0}',
                    '{"x": true, "y": 0, "yaw": 0}',
                    '{"x": NaN, "y": 0, "yaw": 0}', 'not-json'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_pose(raw)

    def test_workspace_manifest_hashes_source_but_excludes_generated_builds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name",
                            "Manifest Test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email",
                            "manifest@example.invalid"], check=True)
            (root / "tracked.c").write_text("baseline\n")
            subprocess.run(["git", "-C", str(root), "add", "tracked.c"],
                           check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm",
                            "baseline"], check=True)
            (root / "tracked.c").write_text("changed\n")
            (root / "new_tool.py").write_text("source\n")
            (root / "build_task").mkdir()
            (root / "build_task" / "large-object.o").write_bytes(b"artifact")

            manifest = workspace_manifest(root)

        hashes = manifest["changed_or_untracked_file_sha256"]
        self.assertIn("tracked.c", hashes)
        self.assertIn("new_tool.py", hashes)
        self.assertNotIn("build_task/large-object.o", hashes)
        self.assertEqual(manifest["excluded_generated_roots"], ["build_task"])
        self.assertNotIn("build_task", manifest["status_porcelain"])
        self.assertTrue(manifest["dirty"])

    def test_generated_build_directory_does_not_mark_source_dirty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name",
                            "Manifest Test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email",
                            "manifest@example.invalid"], check=True)
            (root / "tracked.c").write_text("baseline\n")
            subprocess.run(["git", "-C", str(root), "add", "tracked.c"],
                           check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm",
                            "baseline"], check=True)
            (root / "build_task").mkdir()
            (root / "build_task" / "artifact.o").write_bytes(b"ignored")
            manifest = workspace_manifest(root)

        self.assertFalse(manifest["dirty"])
        self.assertEqual(manifest["status_porcelain"], "")

    def test_map_manifest_hashes_yaml_and_referenced_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "maps").mkdir()
            (root / "maps" / "track.pgm").write_bytes(b"P5\n")
            (root / "maps" / "track.yaml").write_text(
                "image: track.pgm\nresolution: 0.05\n")
            record = map_record(root, "maps/track.yaml")

        self.assertEqual(Path(record["yaml"]["path"]).name, "track.yaml")
        self.assertEqual(Path(record["image"]["path"]).name, "track.pgm")
        self.assertEqual(record["image"]["sha256"],
                         "84b4ed3715c617fa7cf5faa45afac3b9c162a907d55a63e31d38c45f6f5260c8")

    @mock.patch("tools.seal_mpc_run_manifest.subprocess.run")
    def test_prelaunch_seal_rejects_running_named_container(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["docker"], returncode=0,
            stdout="sim-container\nother\n", stderr="")
        with self.assertRaisesRegex(ValueError, "sim-container"):
            assert_containers_stopped(("sim-container", "controller"))

    @mock.patch("tools.seal_mpc_run_manifest.subprocess.run")
    def test_prelaunch_seal_requires_docker_state_check(self, run):
        run.side_effect = FileNotFoundError("docker unavailable")
        with self.assertRaises(FileNotFoundError):
            assert_containers_stopped(("sim-container", "controller"))

    def test_docker_environment_preserves_explicit_host(self):
        with mock.patch.dict("os.environ", {"DOCKER_HOST": "unix:///explicit.sock"},
                             clear=True):
            self.assertEqual(docker_environment()["DOCKER_HOST"],
                             "unix:///explicit.sock")

    def test_docker_environment_selects_repo_rootless_socket(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime_dir = Path(temporary)
            docker_socket = runtime_dir / "docker.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(docker_socket))
                with mock.patch.dict("os.environ", {
                        "XDG_RUNTIME_DIR": temporary}, clear=True):
                    self.assertEqual(docker_environment()["DOCKER_HOST"],
                                     f"unix://{docker_socket}")
            finally:
                server.close()

    @mock.patch("tools.seal_mpc_run_manifest.subprocess.run")
    def test_docker_queries_receive_selected_daemon(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["docker"], returncode=0, stdout="", stderr="")
        with mock.patch.dict("os.environ", {
                "DOCKER_HOST": "unix:///repo-rootless.sock"}, clear=True):
            assert_containers_stopped(("sim-container",))
            image_provenance("controller:dev")
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"]["DOCKER_HOST"],
                             "unix:///repo-rootless.sock")


if __name__ == "__main__":
    unittest.main()
