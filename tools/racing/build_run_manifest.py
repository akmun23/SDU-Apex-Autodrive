#!/usr/bin/env python3
"""Fingerprint the exact runtime/configuration inputs before a simulator run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PRACTICE_SIM = (
    "autodriveecosystem/autodrive_roboracer_sim@sha256:"
    "b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76"
)

CONFIG_FILES = {
    "mpc_config_sha256": "f1tenth_mpc/config/mpc_competition.yaml",
    "odom_config_sha256": "f1tenth_localization/config/sensor_odometry.yaml",
    "ekf_config_sha256": "f1tenth_localization/config/ekf.yaml",
    "amcl_config_sha256": "f1tenth_localization/config/gpu_amcl_cpp_params.yaml",
}
RUNTIME_SOURCE_DIRS = (
    "sdu_apex_autodrive/launch",
    "sdu_apex_autodrive/sdu_apex_autodrive",
    "f1tenth_mpc/src",
    "f1tenth_mpc/include",
    "f1tenth_localization/src",
    "f1tenth_localization/gpu_amcl_cpp/src",
    "f1tenth_localization/gpu_amcl_cpp/include",
)
RUNTIME_SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_hashes() -> dict[str, str]:
    files = {
        path
        for relative in RUNTIME_SOURCE_DIRS
        for path in (ROOT / relative).rglob("*")
        if path.is_file() and path.suffix in RUNTIME_SOURCE_SUFFIXES
    }
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(files)
    }


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def docker_environment() -> dict[str, str]:
    docker_env = os.environ.copy()
    if not docker_env.get("DOCKER_HOST"):
        runtime_dir = Path(
            docker_env.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        )
        rootless_socket = runtime_dir / "docker.sock"
        fallback_socket = Path("/tmp/apex-rootless/docker.sock")
        if rootless_socket.is_socket():
            docker_env["DOCKER_HOST"] = f"unix://{rootless_socket}"
        elif fallback_socket.is_socket():
            docker_env["DOCKER_HOST"] = f"unix://{fallback_socket}"
    return docker_env


def image_id(image: str) -> str:
    return subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True,
        env=docker_environment(),
    ).stdout.strip()


def image_fingerprint(image: str) -> tuple[str, str]:
    result = subprocess.run(
        [
            "docker", "image", "inspect", image,
            "--format", "{{.Id}} {{range .RepoDigests}}{{.}} {{end}}",
        ],
        check=True, capture_output=True, text=True, env=docker_environment(),
    ).stdout.split()
    image_id_value = result[0]
    digest = next((value for value in result[1:] if "@sha256:" in value), "")
    if not digest:
        raise RuntimeError(f"image has no immutable repository digest: {image}")
    return image_id_value, digest


def within_repo(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    map_yaml = within_repo(args.map_yaml)
    trajectory = within_repo(args.trajectory)
    map_image = map_yaml.with_suffix(".pgm")
    if not map_image.is_file():
        raise FileNotFoundError(map_image)
    run_dir = args.run_dir.resolve()
    run_dir.relative_to(ROOT / "live_runs")
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    status = git_value("status", "--porcelain")
    config_hashes = {
        key: sha256(ROOT / relative)
        for key, relative in CONFIG_FILES.items()
    }
    source_hashes = runtime_source_hashes()
    overlay_hash = None
    if args.mpc_overlay:
        overlay_hash = sha256(within_repo(args.mpc_overlay))

    simulator_image_id, simulator_digest = image_fingerprint(args.simulator_image)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "track": args.track,
        "git_sha": git_value("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "worktree_status": status.splitlines(),
        "map_yaml": map_yaml.relative_to(ROOT).as_posix(),
        "map_sha256": sha256(map_yaml),
        "map_image": map_image.relative_to(ROOT).as_posix(),
        "map_image_sha256": sha256(map_image),
        "trajectory": trajectory.relative_to(ROOT).as_posix(),
        "trajectory_sha256": sha256(trajectory),
        **config_hashes,
        "path_tracking_config_sha256": sha256(
            ROOT / "f1tenth_control/config/path_tracking_autodrive.yaml"
        ),
        "vehicle_profile_sha256": sha256(
            ROOT / "f1tenth_planning/config/autodrive_sim_vehicle.yaml"
        ),
        "mpc_overlay_sha256": overlay_hash,
        "runtime_source_sha256": source_hashes,
        "simulator_image_ref": args.simulator_image,
        "simulator_image_digest": simulator_digest,
        "simulator_image_id": simulator_image_id,
        "controller_image": args.controller_image,
        "controller_image_id": image_id(args.controller_image),
        "provenance_tool_sha256": sha256(Path(__file__).resolve()),
        "launch_environment": {
            "controller": os.environ.get("SDU_APEX_CONTROLLER", "mpc"),
            "simulator_mode": os.environ.get("SDU_APEX_SIM_MODE", "batchmode"),
            "build_mpc": os.environ.get("SDU_APEX_BUILD_MPC", "1"),
            "build_localization": os.environ.get("SDU_APEX_BUILD_LOCALIZATION", "0"),
            "controller_max_speed_mps": os.environ.get(
                "SDU_APEX_CONTROLLER_MAX_SPEED", "16.0"
            ),
            "mpc_publish_diagnostics": os.environ.get(
                "SDU_APEX_MPC_PUBLISH_DIAGNOSTICS", "true"
            ),
            "with_rviz": os.environ.get("SDU_APEX_WITH_RVIZ", "false"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=("practice", "final"))
    parser.add_argument("--map-yaml", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument(
        "--controller-image",
        default=os.environ.get(
            "SDU_APEX_IMAGE", "sdu-apex-autodrive:dev-practice-odom-aggressive-20261005"
        ),
    )
    parser.add_argument("--simulator-image", default=PRACTICE_SIM)
    parser.add_argument("--mpc-overlay", type=Path)
    args = parser.parse_args()
    manifest = create_manifest(args)
    output = args.run_dir.resolve() / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote run provenance: {output}")
    print(f"Source revision: {manifest['git_sha']}; dirty worktree: {manifest['worktree_dirty']}")
    print(f"Trajectory SHA-256: {manifest['trajectory_sha256']}")
    print(f"Controller image ID: {manifest['controller_image_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
