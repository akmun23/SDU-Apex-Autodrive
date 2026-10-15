#!/usr/bin/env python3
"""Compare a Unity raceline geometry candidate under identical source physics.

Runs the preserved feedback baseline, a geometry-only ablation, and the full
candidate through tools/start_simulator.sh in practice batch mode. Every run
uses the same source-built player, warm-up, and track. Unity aborts on its first
track collision; map clearance and completed-lap status are checked offline.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from optimize_unity_spatial_raceline import (
    PRACTICE_MAP,
    PRACTICE_MAP_RESOLUTION_M,
    REPO,
    _read_pgm8,
    _track_clearance_summary,
)


DEFAULT_BASELINE = REPO / (
    "live_runs/unity_source_raceline_feedback_mintime_20261014_r115/"
    "raceline_feedback_profile.json")
DEFAULT_GEOMETRY = REPO / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02/"
    "raceline_feedback_profile.json")
DEFAULT_PLAYER = REPO / (
    "live_runs/unity_source_slip_guard_player_20261014_r05/player_overlay")
START_SIM = REPO / "tools/start_simulator.sh"
MIN_CLEARANCE_M = 0.20


def _read_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("raceline_feedback_control") is not True:
        raise ValueError(f"not a Unity feedback raceline profile: {path}")
    return value


def _geometry_only_profile(baseline: dict[str, Any],
                           candidate: dict[str, Any]) -> dict[str, Any]:
    """Use candidate route points while preserving baseline control/speed."""
    profile = copy.deepcopy(baseline)
    warmup = int(baseline["warmup_command_count"])
    if warmup != int(candidate["warmup_command_count"]):
        raise ValueError("baseline and candidate warm-up boundaries differ")
    base_commands = profile["commands"]
    candidate_commands = candidate["commands"]
    if len(base_commands) - warmup != len(candidate_commands) - warmup:
        raise ValueError("baseline and candidate route point counts differ")
    route_xy = []
    for index in range(warmup, len(base_commands)):
        source = candidate_commands[index]
        target = base_commands[index]
        if not source.get("has_reference_position") or not target.get("has_raceline_target"):
            raise ValueError(f"invalid route geometry at command {index}")
        target["reference_x_m"] = float(source["reference_x_m"])
        target["reference_z_m"] = float(source["reference_z_m"])
        route_xy.append((target["reference_x_m"], target["reference_z_m"]))
    points = np.asarray(route_xy, dtype=np.float64)
    if points.shape[0] < 100 or not np.isfinite(points).all():
        raise ValueError("candidate route is incomplete or non-finite")
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(lengths <= 1.0e-6):
        raise ValueError("candidate route contains duplicate adjacent points")
    arc = np.r_[0.0, np.cumsum(lengths[:-1])]
    for route_index, index in enumerate(range(warmup, len(base_commands))):
        base_commands[index]["route_arc_m"] = float(arc[route_index])
    profile["raceline_path_length_m"] = float(np.sum(lengths))
    profile["geometry_ablation_source"] = "candidate route positions only"
    profile["geometry_ablation_baseline"] = "baseline speed and control fields retained"
    return profile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_path(dynamics: Path) -> Path | None:
    captures = sorted(dynamics.glob("wheel_dynamics_*.csv"))
    return captures[0] if len(captures) == 1 else None


def _evaluate(name: str, profile: dict[str, Any], profile_source: str,
              player: Path, output: Path, map_clearance: np.ndarray,
              timeout_s: float) -> dict[str, Any]:
    run_dir = output / name
    run_dir.mkdir()
    profile_path = run_dir / "command_profile.json"
    profile_path.write_text(json.dumps(profile, separators=(",", ":")) + "\n",
                            encoding="utf-8")
    dynamics = run_dir / "dynamics"
    env = os.environ.copy()
    env.update({
        "SDU_APEX_SIM_TRACK": "practice",
        "SDU_APEX_SIM_MODE": "batchmode",
        "SDU_APEX_SIM_CONTAINER": f"sdu_apex_geom_{name}",
        "SDU_APEX_SIM_PLAYER_DIR": str(player),
        "SDU_APEX_SIM_DYNAMICS_DIR": str(dynamics),
        "SDU_APEX_SIM_DYNAMICS_RUN_ID": name,
        "SDU_APEX_SIM_OFFLINE_COMMAND_PROFILE": str(profile_path),
        "SDU_APEX_SIM_OFFLINE_PRACTICE_TRACK": "1",
        "SDU_APEX_SIM_LOG_FILE": "-",
    })
    started = time.monotonic()
    timed_out = False
    try:
        with (run_dir / "simulator.log").open("wb") as log:
            completed = subprocess.run(
                [str(START_SIM)], cwd=REPO, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout_s, check=False)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        exit_code = None
    result_path = dynamics / "offline_result.json"
    result = (json.loads(result_path.read_text(encoding="utf-8"))
              if result_path.is_file() else {})
    capture = _capture_path(dynamics)
    row: dict[str, Any] = {
        "candidate": name,
        "profile_source": profile_source,
        "profile_sha256": _sha256(profile_path),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_time_s": time.monotonic() - started,
        "unity_status": result.get("status", "missing_result"),
        "unity_reason": result.get("reason", ""),
        "fixed_steps_applied": result.get("fixed_steps_applied"),
        "collisions_observed": result.get("collisions_observed"),
        "lap_completed": result.get("raceline_lap_completed"),
        "lap_time_s": result.get("scored_duration_s"),
        "path_length_m": result.get("raceline_path_length_m"),
        "tracking_rms_m": result.get("path_tracking_error_rms_m"),
        "tracking_max_m": result.get("path_tracking_error_max_m"),
        "target_speed_rmse_mps": result.get("target_speed_rmse_mps"),
        "peak_speed_mps": result.get("peak_body_speed_mps"),
        "capture": str(capture) if capture else None,
    }
    if capture and row["lap_time_s"] is not None:
        gate_time = int(profile["warmup_command_count"]) * 0.025
        clearance = _track_clearance_summary(
            capture, gate_time, gate_time + float(row["lap_time_s"]),
            map_clearance)
        row.update(clearance)
        row["valid_lap"] = bool(
            row["unity_status"] == "complete" and row["lap_completed"] is True
            and row["collisions_observed"] == 0
            and row["track_clearance_min_m"] >= MIN_CLEARANCE_M)
    else:
        row["valid_lap"] = False
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--geometry-candidate", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--player", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    baseline_path = args.baseline.resolve()
    candidate_path = args.geometry_candidate.resolve()
    player = args.player.resolve()
    output = args.output.resolve()
    for path in (baseline_path, candidate_path, PRACTICE_MAP):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (player / "AutoDRIVE Simulator.x86_64").is_file():
        raise FileNotFoundError(f"source-built Unity player is missing: {player}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation run: {output}")
    if args.timeout_s <= 0.0:
        raise ValueError("timeout must be positive")
    baseline = _read_profile(baseline_path)
    candidate = _read_profile(candidate_path)
    geometry_only = _geometry_only_profile(baseline, candidate)
    output.mkdir(parents=True)
    map_pixels = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(map_pixels == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M
    manifest: dict[str, Any] = {
        "status": "running",
        "track": "practice",
        "simulator_image": "official practice image through tools/start_simulator.sh",
        "mode": "batchmode",
        "player": str(player),
        "player_gameassembly_sha256": _sha256(player / "GameAssembly.so"),
        "baseline_profile": str(baseline_path),
        "baseline_profile_sha256": _sha256(baseline_path),
        "geometry_candidate_profile": str(candidate_path),
        "geometry_candidate_profile_sha256": _sha256(candidate_path),
        "map_clearance_minimum_m": MIN_CLEARANCE_M,
        "runs": [],
    }
    (output / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")
    runs = (
        ("baseline", baseline, str(baseline_path)),
        ("geometry_only", geometry_only, str(candidate_path)),
        ("combined_candidate", candidate, str(candidate_path)),
    )
    for name, profile, source in runs:
        row = _evaluate(name, profile, source, player, output,
                        map_clearance, args.timeout_s)
        manifest["runs"].append(row)
        manifest["status"] = "running"
        (output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, separators=(",", ":")), flush=True)
        if row["unity_status"] == "collision":
            # Unity itself stops on its first extracted-track collision.
            continue
    valid = [row for row in manifest["runs"] if row["valid_lap"]]
    manifest["best_valid_lap_s"] = min(
        (float(row["lap_time_s"]) for row in valid), default=None)
    manifest["status"] = "complete" if len(manifest["runs"]) == 3 else "incomplete"
    (output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if len(manifest["runs"]) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
