#!/usr/bin/env python3
"""Seal MPC experiment provenance before simulator/controller launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import time

import yaml

GENERATED_UNTRACKED_ROOTS = frozenset({"build_task", "install_task", "log_task"})


def docker_environment() -> dict[str, str]:
    """Select the same rootless daemon used by the repository launch scripts."""
    environment = os.environ.copy()
    if environment.get("DOCKER_HOST"):
        return environment
    runtime_dir = Path(environment.get("XDG_RUNTIME_DIR",
                                      f"/run/user/{os.getuid()}"))
    for socket_path in (runtime_dir / "docker.sock",
                        Path("/tmp/apex-rootless/docker.sock")):
        try:
            if stat.S_ISSOCK(socket_path.stat().st_mode):
                environment["DOCKER_HOST"] = f"unix://{socket_path}"
                break
        except OSError:
            continue
    return environment


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def workspace_manifest(root: Path) -> dict:
    # Keep build products out of the source manifest. Their build settings are
    # recorded from CMakeCache and the running image is pinned separately; a
    # recursive hash of object files and logs can otherwise make sealing slow
    # and produce a huge, non-source manifest.
    status_lines = git(root, "status", "--porcelain",
                       "--untracked-files=normal").splitlines()
    source_status_lines = []
    for line in status_lines:
        path_text = line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        if Path(path_text).parts and \
                Path(path_text).parts[0] in GENERATED_UNTRACKED_ROOTS:
            continue
        source_status_lines.append(line)
    status = "\n".join(source_status_lines)
    tracked_patch = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"], check=True,
        capture_output=True).stdout
    paths = set(git(root, "diff", "--name-only", "HEAD").splitlines())
    paths.update(git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    file_hashes = {}
    for relative in sorted(paths):
        if Path(relative).parts and Path(relative).parts[0] in GENERATED_UNTRACKED_ROOTS:
            continue
        path = root / relative
        if path.is_file():
            file_hashes[relative] = sha256_file(path)
    excluded_generated = [name for name in sorted(GENERATED_UNTRACKED_ROOTS)
                          if (root / name).exists()]
    return {
        "commit_sha": git(root, "rev-parse", "HEAD"),
        "branch": git(root, "branch", "--show-current"),
        "dirty": bool(status),
        "status_porcelain": status,
        "tracked_diff_sha256": hashlib.sha256(tracked_patch).hexdigest(),
        "changed_or_untracked_file_sha256": file_hashes,
        "excluded_generated_roots": excluded_generated,
    }


def assert_containers_stopped(container_names: tuple[str, ...]) -> None:
    result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                            check=True, capture_output=True, text=True,
                            timeout=5.0, env=docker_environment())
    running = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    active = sorted(set(container_names) & running)
    if active:
        raise ValueError("refusing to seal a pre-launch manifest while these "
                         "requested containers are running: " + ", ".join(active))


def image_provenance(reference: str) -> dict:
    result = subprocess.run(
        ["docker", "image", "inspect", "--format",
         "{{.Id}}|{{join .RepoDigests \",\"}}", reference],
        check=False, capture_output=True, text=True, timeout=5.0,
        env=docker_environment())
    if result.returncode != 0:
        return {"reference": reference, "local_image_id": None,
                "repo_digests": [], "inspect_error": result.stderr.strip()}
    image_id, _, digests = result.stdout.strip().partition("|")
    return {"reference": reference, "local_image_id": image_id or None,
            "repo_digests": [value for value in digests.split(",") if value]}


def cpu_topology() -> dict:
    result = subprocess.run(["lscpu", "--json"], check=False,
                            capture_output=True, text=True, timeout=3.0)
    parsed = {}
    if result.returncode == 0:
        try:
            parsed = {entry["field"].rstrip(":"): entry["data"]
                      for entry in json.loads(result.stdout).get("lscpu", [])}
        except (json.JSONDecodeError, KeyError, TypeError):
            parsed = {}
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass
    return {"logical_cpu_count": os.cpu_count(), "affinity": affinity,
            "lscpu": parsed,
            "cpu_max": _read_first_line(Path("/sys/fs/cgroup/cpu.max")),
            "memory_max": _read_first_line(Path("/sys/fs/cgroup/memory.max"))}


def _read_first_line(path: Path) -> str | None:
    try:
        return path.read_text().splitlines()[0]
    except (OSError, IndexError):
        return None


def file_record(root: Path, raw_path: str) -> dict:
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"required run input does not exist: {path}")
    return {"path": str(path), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size}


def validate_mpc_overlay(path: Path) -> None:
    """Reject ROS selectors that would silently leave the MPC unchanged.

    The maintained launch names its unnamespaced component
    /mpc_controller_node. A typo in a YAML selector is otherwise accepted by
    ROS while applying none of the intended experiment parameters.
    """
    validate_node_overlay(path, "mpc_controller_node")


def validate_node_overlay(path: Path, node_name: str) -> None:
    """Fail before launching if ROS would silently ignore an overlay."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    selectors = {"/**", node_name, f"/{node_name}"}
    if not isinstance(document, dict) or not document:
        raise ValueError(f"{node_name} overlay must contain ROS parameters: {path}")
    for selector, entry in document.items():
        if selector not in selectors:
            raise ValueError(f"overlay selector {selector!r} does not target "
                             f"/{node_name}: {path}")
        parameters = entry.get("ros__parameters") if isinstance(entry, dict) else None
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError(f"overlay has no parameters for {selector}: {path}")


def map_record(root: Path, raw_path: str) -> dict:
    yaml_record = file_record(root, raw_path)
    yaml_path = Path(yaml_record["path"])
    image_value = None
    for line in yaml_path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() == "image":
            image_value = value.strip().strip("'\"")
            break
    if not image_value:
        raise ValueError(f"map YAML has no image entry: {yaml_path}")
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image_record = file_record(root, str(image_path))
    return {"yaml": yaml_record, "image": image_record}


def parse_pose(raw: str) -> dict:
    try:
        pose = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("initial pose must be JSON with x, y, yaw") from error
    if (not isinstance(pose, dict) or
            any(key not in pose for key in ("x", "y", "yaw")) or
            any(isinstance(pose[key], bool) or
                not isinstance(pose[key], (int, float)) or
                not math.isfinite(pose[key]) for key in ("x", "y", "yaw"))):
        raise ValueError("initial pose must contain numeric x, y, and yaw")
    return {key: float(pose[key]) for key in ("x", "y", "yaw")}


def build_manifest(args: argparse.Namespace, root: Path) -> dict:
    if not args.acknowledge_not_started:
        raise ValueError("refusing to seal: pass --acknowledge-not-started before launch")
    assert_containers_stopped((args.simulator_container,
                               args.controller_container))
    pose = parse_pose(args.initial_pose)
    config_records = []
    for item in args.config:
        name, separator, value = item.partition("=")
        if not separator or not name or not value:
            raise ValueError("--config values must be NAME=PATH")
        record = {"name": name, **file_record(root, value)}
        if name == "overlay":
            validate_mpc_overlay(Path(record["path"]))
        elif name == "amcl_overlay":
            validate_node_overlay(Path(record["path"]), "gpu_amcl_cpp")
        config_records.append(record)
    if not config_records:
        raise ValueError("provide at least one loaded YAML/config with --config")
    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = root / build_dir
    cache_path = build_dir / "CMakeCache.txt"
    if cache_path.is_file():
        build_record = {"cache": file_record(root, str(cache_path)),
                        "cache_values": _cmake_values(cache_path)}
    else:
        build_record = {"cache": None, "cache_values": {},
                        "note": "CMakeCache.txt not present at seal time"}
    source = workspace_manifest(root)
    machine = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_topology_and_limits": cpu_topology(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "selected_environment": {
            key: os.environ.get(key) for key in (
                "ROS_DOMAIN_ID", "SDU_APEX_MPC_NATIVE_TUNING",
                "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
            if os.environ.get(key) is not None
        },
        "build": build_record,
    }
    runtime = {
        "track": args.track,
        "unity_mode": args.unity_mode,
        "simulator_image": image_provenance(args.simulator_image),
        "controller_image": image_provenance(args.controller_image),
        "simulator_container": args.simulator_container,
        "controller_container": args.controller_container,
        "map": map_record(root, args.map),
        "trajectory": file_record(root, args.trajectory),
        "loaded_configs": config_records,
        "initial_pose": pose,
        "supervisor_version_or_command": args.supervisor,
        "diagnostics": args.diagnostics,
        "ros_executor_threads": args.ros_executor_threads,
        "openmp_threads": args.openmp_threads,
        "jacobian_workers": args.jacobian_workers,
        "launch_command": args.launch_command,
    }
    return {
        "schema": "sdu-apex-mpc-run-manifest/v1",
        "run_id": args.run_id,
        "sealed_before_launch": True,
        "sealed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "runtime": runtime,
        # Flat compatibility keys are consumed by export_mpc_run_report.py.
        "commit_sha": source["commit_sha"],
        "compiler": {key: build_record["cache_values"].get(key)
                     for key in ("CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER")},
        "build_flags": build_record["cache_values"],
        "cpu_model": machine["cpu_topology_and_limits"]["lscpu"].get(
            "Model name"),
        "cpu_topology": machine["cpu_topology_and_limits"],
        "ros_executor_threads": args.ros_executor_threads,
        "openmp_settings": {
            "threads": args.openmp_threads,
            "jacobian_workers": args.jacobian_workers,
            "environment": machine["selected_environment"],
        },
        "container_limits": {
            "cpu_max": machine["cpu_topology_and_limits"]["cpu_max"],
            "memory_max": machine["cpu_topology_and_limits"]["memory_max"],
        },
        "unity_mode": args.unity_mode,
        "trajectory_sha256": runtime["trajectory"]["sha256"],
        "loaded_yaml_hashes": {
            item["name"]: item["sha256"] for item in config_records
        },
        "initial_pose": pose,
        "map_hash": runtime["map"]["yaml"]["sha256"],
        "map_image_hash": runtime["map"]["image"]["sha256"],
        "supervisor_version": args.supervisor,
    }


def _cmake_values(path: Path) -> dict:
    selected = ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER",
                "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "SDU_APEX_MPC_NATIVE_TUNING")
    values = {}
    for line in path.read_text(errors="replace").splitlines():
        if ":" not in line or "=" not in line or line.startswith("//"):
            continue
        key, value = line.split("=", 1)
        name = key.split(":", 1)[0]
        if name in selected:
            values[name] = value
    return values


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acknowledge-not-started", action="store_true",
                        help="confirm neither simulator nor controller has launched")
    parser.add_argument("--track", required=True)
    parser.add_argument("--unity-mode", choices=("batchmode", "gui"), required=True)
    parser.add_argument("--simulator-image", required=True)
    parser.add_argument("--controller-image", required=True)
    parser.add_argument("--simulator-container", required=True)
    parser.add_argument("--controller-container", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--config", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--initial-pose", required=True, metavar='JSON')
    parser.add_argument("--supervisor", required=True)
    parser.add_argument("--diagnostics", choices=("on", "off"), required=True)
    parser.add_argument("--ros-executor-threads", type=int, required=True)
    parser.add_argument("--openmp-threads", type=int, required=True)
    parser.add_argument("--jacobian-workers", type=int, required=True)
    parser.add_argument("--build-dir", default="build/f1tenth_mpc")
    parser.add_argument("--launch-command", action="append", default=[])
    args = parser.parse_args()
    if min(args.ros_executor_threads, args.openmp_threads,
           args.jacobian_workers) < 1:
        parser.error("thread and worker counts must be positive")
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        parser.error(f"refusing to overwrite existing run manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = build_manifest(args, root)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"sealed pre-launch run manifest: {output}")
    print(f"source={manifest['source']['commit_sha']} dirty={manifest['source']['dirty']}")
    print(f"trajectory_sha256={manifest['runtime']['trajectory']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
