#!/usr/bin/env python3
"""Test evidence-based local speed increases on the slow practice corners.

Starts from one completed source-Unity feedback raceline. Raised target speed
is smoothly tapered across adjacent route sectors; the unchanged Unity player
and its 1 kHz plant determine whether the extra corner speed is feasible.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from evaluate_unity_geometry_ablation import (
    DEFAULT_PLAYER,
    MIN_CLEARANCE_M,
    _evaluate,
    _capture_path,
    _read_profile,
    _sha256,
)
from optimize_unity_spatial_raceline import (
    PRACTICE_MAP,
    PRACTICE_MAP_RESOLUTION_M,
    REPO,
    _read_pgm8,
)


DEFAULT_SEED = REPO / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02/"
    "raceline_feedback_profile.json")
DEFAULT_BASELINE_RUN = REPO / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02")
DEFAULT_GAIN = 0.10


def _speed_bump(profile: dict[str, Any], sectors: tuple[int, ...],
                increase: float, course_velocity_frame: bool
                ) -> dict[str, Any]:
    result = copy.deepcopy(profile)
    warmup = int(result["warmup_command_count"])
    commands = result["commands"]
    count = len(commands) - warmup
    if count < 100 or not 0.0 < increase <= 0.25:
        raise ValueError("invalid raceline size or speed increase")
    indices = np.arange(count, dtype=np.float64)
    position = indices * 12.0 / count
    bump = np.zeros(count, dtype=np.float64)
    for sector in sectors:
        center = float(sector) - 0.5
        distance = np.abs(position - center)
        distance = np.minimum(distance, 12.0 - distance)
        weight = np.where(distance <= 1.0,
                          0.5 * (1.0 + np.cos(np.pi * distance)), 0.0)
        bump = np.maximum(bump, weight)
    for route_index, command_index in enumerate(range(warmup, len(commands))):
        commands[command_index]["target_speed_mps"] = float(
            commands[command_index]["target_speed_mps"] *
            (1.0 + increase * bump[route_index]))
    result["speed_push_sectors"] = list(sectors)
    result["speed_push_peak_fraction"] = increase
    result["speed_push_shape"] = "cosine taper, one sector either side"
    result["raceline_course_velocity_frame"] = course_velocity_frame
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-profile", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--player", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--increase", type=float, default=DEFAULT_GAIN)
    parser.add_argument("--corners", choices=("s6", "s11", "both"),
                        nargs="+", default=("s6", "s11", "both"))
    parser.add_argument("--course-velocity-frame", action="store_true",
                        help="use Unity ground-course direction for path steering")
    parser.add_argument("--steering-gain", type=float,
                        help="override the Unity development follower steering gain")
    parser.add_argument("--lookahead-speed-s", type=float,
                        help="override speed-dependent Unity follower lookahead")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    seed_path = args.seed_profile.resolve()
    baseline_dir = args.baseline_run.resolve()
    player = args.player.resolve()
    output = args.output.resolve()
    for path in (seed_path, PRACTICE_MAP):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (baseline_dir / "dynamics/offline_result.json").is_file():
        raise FileNotFoundError("baseline run result is missing")
    if _capture_path(baseline_dir / "dynamics") is None:
        raise FileNotFoundError("baseline Unity capture is missing")
    if not (player / "GameAssembly.so").is_file():
        raise FileNotFoundError(f"Unity player is missing: {player}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment: {output}")
    if not 0.0 < args.increase <= 0.25:
        raise ValueError("speed increase must be in (0, 0.25]")

    profile = _read_profile(seed_path)
    previous = json.loads((baseline_dir / "dynamics/offline_result.json")
                          .read_text(encoding="utf-8"))
    expected_player_hash = _sha256(player / "GameAssembly.so")
    output.mkdir(parents=True)
    map_pixels = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(map_pixels == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M
    manifest: dict[str, Any] = {
        "status": "running",
        "track": "practice",
        "simulator": "tools/start_simulator.sh, official image, batchmode",
        "player": str(player),
        "player_gameassembly_sha256": expected_player_hash,
        "seed_profile": str(seed_path),
        "seed_profile_sha256": _sha256(seed_path),
        "same_player_baseline": str(baseline_dir),
        "baseline_lap_time_s": previous["scored_duration_s"],
        "baseline_path_clearance_min_m": previous.get("track_clearance_min_m"),
        "peak_target_speed_increase_fraction": args.increase,
        "course_velocity_frame": args.course_velocity_frame,
        "steering_gain_override": args.steering_gain,
        "lookahead_speed_s_override": args.lookahead_speed_s,
        "map_clearance_minimum_m": MIN_CLEARANCE_M,
        "runs": [],
    }
    (output / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")
    sector_by_corner = {"s6": (6,), "s11": (11,), "both": (6, 11)}
    percent = int(round(100.0 * args.increase))
    suffix = f"plus{percent:02d}"
    if args.course_velocity_frame:
        suffix += "_course"
    for corner in args.corners:
        sectors = sector_by_corner[corner]
        name = f"{corner}_{suffix}"
        candidate = _speed_bump(profile, sectors, args.increase,
                                args.course_velocity_frame)
        if args.steering_gain is not None:
            if not 0.0 < args.steering_gain <= 1.0:
                raise ValueError("steering gain must be in (0, 1]")
            candidate["raceline_steering_gain"] = args.steering_gain
        if args.lookahead_speed_s is not None:
            if args.lookahead_speed_s < 0.0:
                raise ValueError("lookahead speed coefficient must be nonnegative")
            candidate["raceline_lookahead_speed_s"] = args.lookahead_speed_s
        row = _evaluate(name, candidate, str(seed_path), player, output,
                        map_clearance, args.timeout_s)
        baseline_time = float(previous["scored_duration_s"])
        row["lap_time_change_vs_same_player_baseline_s"] = (
            None if row["lap_time_s"] is None else
            float(row["lap_time_s"]) - baseline_time)
        row["speed_push_sectors"] = list(sectors)
        row["speed_push_peak_fraction"] = args.increase
        manifest["runs"].append(row)
        (output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, separators=(",", ":")), flush=True)
    valid = [row for row in manifest["runs"] if row["valid_lap"]]
    manifest["best_valid_lap_s"] = min(
        [baseline_time] + [float(row["lap_time_s"]) for row in valid])
    manifest["status"] = ("complete" if len(manifest["runs"]) == len(args.corners)
                          else "incomplete")
    (output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if len(manifest["runs"]) == len(args.corners) else 1


if __name__ == "__main__":
    raise SystemExit(main())
