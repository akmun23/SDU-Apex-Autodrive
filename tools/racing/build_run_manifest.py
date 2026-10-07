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

import yaml

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
    "actuator_config_sha256": "sdu_apex_autodrive/config/actuator_interface.yaml",
}
VEHICLE_PROFILE = "f1tenth_planning/config/autodrive_sim_vehicle.yaml"
VEHICLE_MODEL_SPEC = "config/racing/racing_vehicle_model.json"
SIMULATOR_EXECUTABLE_IN_IMAGE = "./AutoDRIVE Simulator.x86_64"
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


def image_file_sha256(image: str, image_path: str) -> str:
    """Hash a deployment file without starting the simulator executable."""
    output = subprocess.run(
        [
            "docker", "run", "--rm", "--network=none", "--ulimit", "core=0",
            "--entrypoint", "/bin/bash", image, "-lc",
            'sha256sum -- "$1"', "manifest-hash", image_path,
        ],
        check=True, capture_output=True, text=True, env=docker_environment(),
    ).stdout.split()
    if not output or len(output[0]) != 64:
        raise RuntimeError(f"could not hash simulator executable in {image}: {image_path}")
    return output[0]


def generate_vehicle_contract(
    *,
    run_dir: Path,
    optimizer_config: Path,
    resolved_optimizer_config: Path,
    centerline: Path | None,
    config_hashes: dict[str, str],
    overlay_hash: str | None,
) -> tuple[Path, str]:
    """Save the resolved candidate vehicle model and the source hashes it depends on."""
    optimizer_cfg = yaml.safe_load(optimizer_config.read_text(encoding="utf-8")) or {}
    resolved_cfg = yaml.safe_load(
        resolved_optimizer_config.read_text(encoding="utf-8")
    ) or {}
    profile_path = ROOT / VEHICLE_PROFILE
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    contract_path = run_dir / "vehicle_contract.json"
    if contract_path.exists():
        raise FileExistsError(contract_path)

    contract = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "optimizer_config": {
            "path": optimizer_config.relative_to(ROOT).as_posix(),
            "sha256": sha256(optimizer_config),
        },
        "resolved_optimizer_config": {
            "path": resolved_optimizer_config.relative_to(ROOT).as_posix(),
            "sha256": sha256(resolved_optimizer_config),
        },
        "centerline": ({
            "path": centerline.relative_to(ROOT).as_posix(),
            "sha256": sha256(centerline),
        } if centerline else None),
        "vehicle_profile": {
            "path": VEHICLE_PROFILE,
            "sha256": sha256(profile_path),
            "vehicle": profile.get("vehicle", {}),
            "geometry": profile.get("geometry", {}),
        },
        "optimizer_model": {
            "vehicle_model": resolved_cfg.get("vehicle_model", {}),
            "lateral_envelope": resolved_cfg.get("lateral_envelope", {}),
            "constraints": (resolved_cfg.get("config", {}) or {}).get(
                "limits", {}
            ),
            "track_constraints": (resolved_cfg.get("config", {}) or {}).get(
                "track", {}
            ),
            "combined_acceleration": (resolved_cfg.get("config", {}) or {}).get(
                "combined_acceleration", {}
            ),
        },
        "runtime_config_sha256": config_hashes,
        "mpc_overlay_sha256": overlay_hash,
        "optimizer_settings": optimizer_cfg,
    }
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract_path, sha256(contract_path)


def within_repo(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def create_manifest(args: argparse.Namespace) -> dict[str, Any]:
    map_yaml = within_repo(args.map_yaml)
    trajectory = within_repo(args.trajectory)
    model_spec = ROOT / VEHICLE_MODEL_SPEC
    candidate_manifest_path = trajectory.parent / "candidate_manifest.json"
    candidate_manifest = None
    candidate_manifest_sha256 = None
    declared_vehicle_model_sha256 = None
    if candidate_manifest_path.is_file():
        candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        candidate_manifest_sha256 = sha256(candidate_manifest_path)
        declared_vehicle_model_sha256 = candidate_manifest.get("vehicle_model_sha256")
        if (declared_vehicle_model_sha256 is not None
                and declared_vehicle_model_sha256 != sha256(model_spec)):
            raise ValueError(
                "candidate vehicle_model_sha256 does not match the current shared controller model"
            )
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

    optimizer_config = within_repo(args.optimizer_config) if args.optimizer_config else None
    resolved_optimizer_config = (
        within_repo(args.resolved_optimizer_config)
        if args.resolved_optimizer_config else None
    )
    centerline = within_repo(args.centerline) if args.centerline else None
    if (optimizer_config is None) != (resolved_optimizer_config is None):
        raise ValueError(
            "--optimizer-config and --resolved-optimizer-config must be provided together"
        )
    vehicle_contract = None
    if optimizer_config and resolved_optimizer_config:
        vehicle_contract = generate_vehicle_contract(
            run_dir=run_dir,
            optimizer_config=optimizer_config,
            resolved_optimizer_config=resolved_optimizer_config,
            centerline=centerline,
            config_hashes=config_hashes,
            overlay_hash=overlay_hash,
        )

    simulator_image_id, simulator_digest = image_fingerprint(args.simulator_image)
    simulator_executable_sha256 = image_file_sha256(
        args.simulator_image, SIMULATOR_EXECUTABLE_IN_IMAGE
    )
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
        "vehicle_model_sha256": sha256(model_spec),
        "candidate_manifest": (
            candidate_manifest_path.relative_to(ROOT).as_posix()
            if candidate_manifest is not None else None
        ),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "candidate_declared_vehicle_model_sha256": declared_vehicle_model_sha256,
        **config_hashes,
        "path_tracking_config_sha256": sha256(
            ROOT / "f1tenth_control/config/path_tracking_autodrive.yaml"
        ),
        "vehicle_profile_sha256": sha256(ROOT / VEHICLE_PROFILE),
        "mpc_overlay_sha256": overlay_hash,
        "optimizer_config_sha256": sha256(optimizer_config) if optimizer_config else None,
        "resolved_optimizer_config_sha256": (
            sha256(resolved_optimizer_config) if resolved_optimizer_config else None
        ),
        "centerline_sha256": sha256(centerline) if centerline else None,
        "vehicle_contract": (
            vehicle_contract[0].relative_to(ROOT).as_posix()
            if vehicle_contract else None
        ),
        "vehicle_contract_sha256": vehicle_contract[1] if vehicle_contract else None,
        "runtime_source_sha256": source_hashes,
        "simulator_image_ref": args.simulator_image,
        "simulator_image_digest": simulator_digest,
        "simulator_image_id": simulator_image_id,
        "simulator_executable_sha256": simulator_executable_sha256,
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
            "mpc_start_delay_sec": os.environ.get(
                "SDU_APEX_MPC_START_DELAY_SEC", "2.0"
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
    parser.add_argument("--optimizer-config", type=Path)
    parser.add_argument("--resolved-optimizer-config", type=Path)
    parser.add_argument("--centerline", type=Path)
    args = parser.parse_args()
    manifest = create_manifest(args)
    output = args.run_dir.resolve() / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote run provenance: {output}")
    print(f"Source revision: {manifest['git_sha']}; dirty worktree: {manifest['worktree_dirty']}")
    print(f"Trajectory SHA-256: {manifest['trajectory_sha256']}")
    print(f"Controller image ID: {manifest['controller_image_id']}")
    print(f"Simulator executable SHA-256: {manifest['simulator_executable_sha256']}")
    if manifest["vehicle_contract"]:
        print(f"Vehicle contract: {manifest['vehicle_contract']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
