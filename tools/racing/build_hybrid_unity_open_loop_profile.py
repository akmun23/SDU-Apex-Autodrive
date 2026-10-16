#!/usr/bin/env python3
"""Splice a supported spatial segment from one Unity profile into a valid lap.

The base profile supplies the warm-up, spatial route, and controls outside the
selected interval. Candidate controls are interpolated by candidate route arc
fraction—not command index or elapsed time—then blended smoothly at the
interval edges. This is for empirical Unity replay; its model-predicted time is
metadata only and is never reported as a measured lap time.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _smoothstep(value: np.ndarray) -> np.ndarray:
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def merge_profiles(base: dict[str, Any], candidate: dict[str, Any],
                   progress_start: float, progress_end: float,
                   fade_fraction: float,
                   steering_mix: float = 1.0,
                   throttle_mix: float = 1.0) -> dict[str, Any]:
    """Return a base-route profile with a smooth candidate-control segment."""
    if not 0.0 < progress_start < progress_end < 1.0:
        raise ValueError("patch progress must satisfy 0 < start < end < 1")
    if (not np.isfinite(fade_fraction) or fade_fraction <= 0.0
            or 2.0 * fade_fraction >= progress_end - progress_start
            or progress_start - fade_fraction < 0.0
            or progress_end + fade_fraction > 1.0):
        raise ValueError("fade must fit inside the lap and patch interval")
    if (not np.isfinite(steering_mix) or not 0.0 <= steering_mix <= 1.0
            or not np.isfinite(throttle_mix)
            or not 0.0 <= throttle_mix <= 1.0):
        raise ValueError("control component mixes must be in [0, 1]")
    if (base.get("fixed_delta_time_s") != 0.001
            or candidate.get("fixed_delta_time_s") != 0.001
            or base.get("physics_ticks_per_command") != 25
            or candidate.get("physics_ticks_per_command") != 25):
        raise ValueError("both profiles must use the established 1 kHz / 40 Hz grid")

    warmup = int(base.get("warmup_command_count", -1))
    if warmup < 1 or warmup != int(candidate.get("warmup_command_count", -2)):
        raise ValueError("base and candidate warm-up boundaries must match")
    base_commands = base.get("commands")
    candidate_commands = candidate.get("commands")
    if (not isinstance(base_commands, list) or not isinstance(candidate_commands, list)
            or len(base_commands) <= warmup or len(candidate_commands) <= warmup):
        raise ValueError("both profiles must contain scored commands")

    base_route = base_commands[warmup:]
    candidate_route = candidate_commands[warmup:]
    route_xy = np.asarray([
        [float(row["reference_x_m"]), float(row["reference_z_m"])]
        for row in base_route
    ], dtype=np.float64)
    if (route_xy.ndim != 2 or route_xy.shape[1] != 2
            or not np.isfinite(route_xy).all()
            or not all(row.get("has_reference_position") is True for row in base_route)):
        raise ValueError("base profile must provide finite spatial route points")
    segment_lengths = np.linalg.norm(np.roll(route_xy, -1, axis=0) - route_xy, axis=1)
    if np.any(segment_lengths <= 1.0e-6):
        raise ValueError("base route contains duplicate adjacent points")
    base_length = float(np.sum(segment_lengths))
    base_progress = np.r_[0.0, np.cumsum(segment_lengths[:-1])] / base_length

    candidate_length = float(candidate.get("raceline_path_length_m", 0.0))
    if not np.isfinite(candidate_length) or candidate_length <= 0.0:
        raise ValueError("candidate needs a positive raceline_path_length_m")
    candidate_arc = np.asarray([
        float(row["route_arc_m"]) for row in candidate_route
    ], dtype=np.float64)
    candidate_controls = np.asarray([
        [float(row["steering_norm"]), float(row["throttle_norm"])]
        for row in candidate_route
    ], dtype=np.float64)
    if (not np.isfinite(candidate_arc).all()
            or not np.isfinite(candidate_controls).all()
            or np.any(np.diff(candidate_arc) <= 0.0)
            or candidate_arc[0] < -1.0e-8
            or candidate_arc[-1] >= candidate_length):
        raise ValueError("candidate controls need increasing route_arc_m samples")
    candidate_progress = candidate_arc / candidate_length
    if candidate_progress[0] > progress_start - fade_fraction:
        raise ValueError("candidate does not cover the requested patch and fade-in")
    if candidate_progress[-1] < progress_end + fade_fraction:
        raise ValueError("candidate does not cover the requested patch and fade-out")

    weight = np.zeros_like(base_progress)
    rise = (base_progress >= progress_start - fade_fraction) & (
        base_progress < progress_start)
    hold = ((base_progress >= progress_start)
            & (base_progress <= progress_end))
    fall = (base_progress > progress_end) & (
        base_progress <= progress_end + fade_fraction)
    weight[rise] = _smoothstep(
        (base_progress[rise] - (progress_start - fade_fraction)) / fade_fraction)
    weight[hold] = 1.0
    weight[fall] = 1.0 - _smoothstep(
        (base_progress[fall] - progress_end) / fade_fraction)

    candidate_steering = np.interp(
        base_progress, candidate_progress, candidate_controls[:, 0])
    candidate_throttle = np.interp(
        base_progress, candidate_progress, candidate_controls[:, 1])
    merged = copy.deepcopy(base)
    merged_commands = merged["commands"]
    for index, fraction in enumerate(base_progress):
        if weight[index] <= 0.0:
            continue
        row = merged_commands[warmup + index]
        steering_weight = weight[index] * steering_mix
        throttle_weight = weight[index] * throttle_mix
        row["steering_norm"] = float(np.clip(
            (1.0 - steering_weight) * float(row["steering_norm"])
            + steering_weight * candidate_steering[index], -1.0, 1.0))
        row["throttle_norm"] = float(np.clip(
            (1.0 - throttle_weight) * float(row["throttle_norm"])
            + throttle_weight * candidate_throttle[index], 0.0, 1.0))

    merged["hybrid_control_splice"] = {
        "base_route_and_controls_outside_patch": True,
        "candidate_controls_only_inside_patch": True,
        "progress_start": float(progress_start),
        "progress_end": float(progress_end),
        "fade_fraction_each_side": float(fade_fraction),
        "candidate_steering_mix": float(steering_mix),
        "candidate_throttle_mix": float(throttle_mix),
        "interpolation_coordinate": "candidate route arc / candidate path length",
        "candidate_predicted_lap_time_s_is_not_measured": candidate.get(
            "bottom_up_optimizer_predicted_lap_time_s"),
    }
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-start", type=float, required=True)
    parser.add_argument("--progress-end", type=float, required=True)
    parser.add_argument("--fade-fraction", type=float, required=True)
    parser.add_argument("--steering-mix", type=float, default=1.0)
    parser.add_argument("--throttle-mix", type=float, default=1.0)
    args = parser.parse_args()

    base_path, candidate_path, output = (
        args.base_profile.resolve(), args.candidate_profile.resolve(),
        args.output_dir.resolve())
    if not base_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError(base_path if not base_path.is_file() else candidate_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {output}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    profile = merge_profiles(base, candidate, args.progress_start,
                             args.progress_end, args.fade_fraction,
                             args.steering_mix, args.throttle_mix)
    output.mkdir(parents=True)
    profile_path = output / "unity_hybrid_open_loop_profile.json"
    profile_path.write_text(json.dumps(profile, separators=(",", ":")) + "\n",
                            encoding="utf-8")
    report = {
        "status": "candidate_profile_built_not_sim_validated",
        "base_profile": str(base_path),
        "base_profile_sha256": _sha256(base_path),
        "candidate_profile": str(candidate_path),
        "candidate_profile_sha256": _sha256(candidate_path),
        "hybrid_profile": str(profile_path),
        "hybrid_profile_sha256": _sha256(profile_path),
        "splice": profile["hybrid_control_splice"],
        "measured_lap_time_s": None,
    }
    (output / "profile_build.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
