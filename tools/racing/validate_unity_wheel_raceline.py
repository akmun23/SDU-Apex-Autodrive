#!/usr/bin/env python3
"""Replay one bottom-up wheel-optimizer profile in the practice Unity player."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from tools.racing.optimize_unity_centerline_control_bayesian import (
    DEFAULT_REFERENCE_CAPTURE,
    KNOTS,
    PARAMETER_NAMES,
    _finish_gate,
    _read_json,
    _score_trial,
)
from tools.racing.optimize_unity_sector_raceline import DEFAULT_OPTIMIZER_PLAYER
from tools.racing.optimize_unity_spatial_raceline import (
    PRACTICE_MAP,
    PRACTICE_MAP_RESOLUTION_M,
    _read_pgm8,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    ackermann_center_from_wheel_angles,
)

CONTROL_DT_S = 0.025


def _validate_time_parameterized_profile(profile: dict) -> tuple[int, int]:
    warmup = int(profile.get("warmup_command_count", -1))
    commands = profile.get("commands")
    predicted = profile.get("bottom_up_optimizer_predicted_lap_time_s")
    if (not isinstance(commands, list) or warmup < 0 or warmup >= len(commands)
            or predicted is None or not np.isfinite(float(predicted))
            or float(predicted) <= 0.0):
        raise ValueError("candidate profile has invalid lap/time metadata")
    if (profile.get("bottom_up_optimizer_time_parameterized") is not True
            or not np.isclose(
                float(profile.get("bottom_up_optimizer_time_sample_interval_s", 0.0)),
                CONTROL_DT_S, rtol=0.0, atol=1.0e-12)):
        raise ValueError("candidate profile is not sampled on the 40 Hz time grid")
    scored_count = len(commands) - warmup
    expected_count = math.ceil(float(predicted) / CONTROL_DT_S - 1.0e-12)
    if (scored_count != expected_count
            or int(profile.get("scored_horizon_commands", -1)) != scored_count):
        raise ValueError(
            "candidate horizon length disagrees with its predicted lap time: "
            f"{scored_count} commands versus {expected_count} at 40 Hz")
    return warmup, scored_count


def _prediction_tracking_metrics(capture_path: Path, states_path: Path,
                                 gate_start_s: float, gate_stop_s: float
                                 ) -> dict:
    """Compare Unity states with the nearest point on the OCP path.

    This is a spatial path-following check, not an equal-time free rollout:
    the measured 40 Hz samples are matched monotonically to the nearest OCP
    node. It separates route/path mismatch from speed, lateral velocity,
    yaw-rate, and road-wheel steering mismatch without claiming those errors
    are independent.
    """
    if gate_stop_s <= gate_start_s:
        raise ValueError("tracking interval must be increasing")

    with states_path.open("r", encoding="utf-8", newline="") as stream:
        states = list(csv.DictReader(stream))
    required_states = (
        "x_m", "y_m", "speed_mps", "lateral_speed_mps",
        "yaw_rate_radps", "steering_rad",
    )
    if len(states) < 3 or any(name not in states[0] for name in required_states):
        raise ValueError("OCP state file lacks fields needed for tracking metrics")
    predicted = {
        name: np.asarray([float(row[name]) for row in states], dtype=np.float64)
        for name in required_states
    }
    if not all(np.isfinite(values).all() for values in predicted.values()):
        raise ValueError("OCP state file contains non-finite tracking values")
    route_xy = np.column_stack((predicted["x_m"], predicted["y_m"]))
    route_count = len(route_xy)
    route_tree = cKDTree(route_xy)

    capture_fields = (
        "sim_time_s", "pos_world_x_m", "pos_world_z_m",
        "vel_body_x_mps", "vel_body_z_mps", "omega_body_y_rps",
        "fl_steer_deg", "fr_steer_deg",
    )
    with capture_path.open("r", encoding="utf-8", newline="") as stream:
        header_line = next((line for line in stream if not line.startswith("#")), None)
    if header_line is None:
        raise ValueError("Unity capture has no CSV header")
    columns = {name: index for index, name in enumerate(next(csv.reader([header_line])))}
    missing = [name for name in capture_fields if name not in columns]
    if missing:
        raise ValueError(f"Unity capture lacks tracking fields: {missing}")
    indices = [columns[name] for name in capture_fields]
    with capture_path.open("r", encoding="utf-8", newline="") as stream:
        data_lines = (line for line in stream if not line.startswith("#"))
        next(data_lines)  # header
        capture = np.loadtxt(
            data_lines, delimiter=",", usecols=indices,
            dtype=np.float64, ndmin=2)
    if len(capture) < 2 or not np.isfinite(capture).all():
        raise ValueError("Unity capture is empty or contains non-finite values")

    sample_times = np.arange(gate_start_s, gate_stop_s, CONTROL_DT_S)
    time_index = np.searchsorted(capture[:, 0], sample_times).clip(
        0, len(capture) - 1)
    previous_index = np.maximum(time_index - 1, 0)
    use_previous = (
        np.abs(capture[previous_index, 0] - sample_times)
        < np.abs(capture[time_index, 0] - sample_times)
    )
    time_index = np.where(use_previous, previous_index, time_index)
    sample_times = capture[time_index, 0]
    valid = (sample_times >= gate_start_s - 1.0e-6) & (
        sample_times < gate_stop_s + 1.0e-6)
    time_index = time_index[valid]
    sample_times = sample_times[valid]
    if len(time_index) < 3:
        raise ValueError("Unity capture has too few samples in the scored lap")
    # Remove duplicate capture rows if the requested interval is shorter than
    # the sampling period or a boundary rounds to the same physics tick.
    keep = np.r_[True, np.diff(time_index) > 0]
    time_index = time_index[keep]
    sample_times = sample_times[keep]

    rows = capture[time_index]
    actual_xy = np.column_stack((rows[:, 2] + 0.24, -rows[:, 1] + 0.05))
    _, nearby_nodes = route_tree.query(
        actual_xy, k=min(4, route_count))
    if nearby_nodes.ndim == 1:
        nearby_nodes = nearby_nodes[:, None]
    distance = np.empty(len(actual_xy), dtype=np.float64)
    route_index = np.empty(len(actual_xy), dtype=np.float64)
    signed_path_error = np.empty(len(actual_xy), dtype=np.float64)
    for sample, point in enumerate(actual_xy):
        candidate_segments = {
            int(node) % route_count
            for nearby in nearby_nodes[sample]
            for node in (nearby - 1, nearby)
        }
        best_distance_squared = float("inf")
        best_fraction = 0.0
        best_segment = 0
        for segment in candidate_segments:
            following = (segment + 1) % route_count
            start = route_xy[segment]
            vector = route_xy[following] - start
            fraction = float(np.clip(
                np.dot(point - start, vector)
                / max(float(np.dot(vector, vector)), 1.0e-15),
                0.0, 1.0))
            residual = point - (start + fraction * vector)
            distance_squared = float(np.dot(residual, residual))
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_segment = segment
                best_fraction = fraction
        following = (best_segment + 1) % route_count
        vector = route_xy[following] - route_xy[best_segment]
        projected = route_xy[best_segment] + best_fraction * vector
        normal = np.asarray((-vector[1], vector[0])) / max(
            float(np.linalg.norm(vector)), 1.0e-15)
        signed_path_error[sample] = float(np.dot(point - projected, normal))
        route_index[sample] = best_segment + best_fraction
        distance[sample] = np.sqrt(best_distance_squared)
    for index in range(1, len(route_index)):
        while route_index[index] < route_index[index - 1] - route_count / 2:
            route_index[index] += route_count
        while route_index[index] > route_index[index - 1] + route_count / 2:
            route_index[index] -= route_count
    route_index = np.maximum.accumulate(route_index)
    route_segment = np.floor(route_index).astype(np.int64)
    segment_fraction = route_index - route_segment
    state_index = route_segment % route_count
    state_next = (state_index + 1) % route_count
    fraction = segment_fraction

    def predicted_at(name: str) -> np.ndarray:
        return (predicted[name][state_index] * (1.0 - fraction)
                + predicted[name][state_next] * fraction)

    actual_u = rows[:, 4]
    actual_v = -rows[:, 3]
    actual_yaw_rate = -rows[:, 5]
    actual_steering = np.asarray([
        ackermann_center_from_wheel_angles(
            -np.deg2rad(fl), -np.deg2rad(fr))
        for fl, fr in rows[:, 6:8]
    ])
    errors = {
        "path_distance_m": distance,
        "forward_speed_mps": actual_u - predicted_at("speed_mps"),
        "lateral_speed_mps": actual_v - predicted_at("lateral_speed_mps"),
        "yaw_rate_radps": actual_yaw_rate - predicted_at("yaw_rate_radps"),
        "road_wheel_steering_rad": actual_steering - predicted_at("steering_rad"),
    }

    def summarize(values: np.ndarray) -> dict:
        absolute = np.abs(values)
        return {
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "p95_abs": float(np.percentile(absolute, 95)),
            "max_abs": float(np.max(absolute)),
        }

    first_divergence = {}
    thresholds = {
        "path_distance_m": 0.10,
        "forward_speed_mps": 0.50,
        "lateral_speed_mps": 0.20,
        "yaw_rate_radps": 0.20,
        "road_wheel_steering_rad": 0.05,
    }
    for name, threshold in thresholds.items():
        exceed = np.abs(errors[name]) > threshold
        first_divergence[name] = (
            float(sample_times[np.flatnonzero(exceed)[0]] - gate_start_s)
            if np.any(exceed) else None
        )
    segment_lengths = np.linalg.norm(
        np.roll(route_xy, -1, axis=0) - route_xy, axis=1)
    route_arc = np.r_[0.0, np.cumsum(segment_lengths)]
    total_route_length = float(route_arc[-1])
    if total_route_length <= 0.0:
        raise ValueError("OCP path has zero geometric length")
    unwrapped_arc = (
        route_arc[state_index]
        + segment_fraction * segment_lengths[state_index]
        + np.floor_divide(route_segment, route_count) * total_route_length
    )
    route_fraction = np.mod(unwrapped_arc / total_route_length, 1.0)
    sector_index = np.minimum(11, np.floor(route_fraction * 12).astype(int))
    path_bias_by_sector = []
    for sector in range(12):
        selected = signed_path_error[sector_index == sector]
        path_bias_by_sector.append({
            "route_fraction": (sector + 0.5) / 12.0,
            "sample_count": int(len(selected)),
            "median_signed_path_error_m": (
                float(np.median(selected)) if len(selected) else None),
        })
    return {
        "method": "40Hz samples projected monotonically onto OCP path segments; state values interpolated by path progress, not equal-time free rollout",
        "sample_count": int(len(time_index)),
        "route_length_m": total_route_length,
        "first_threshold_crossing_s": first_divergence,
        "signed_path_error_m": {
            "median": float(np.median(signed_path_error)),
            "p10": float(np.percentile(signed_path_error, 10)),
            "p90": float(np.percentile(signed_path_error, 90)),
        },
        "path_bias_by_route_fraction_12": path_bias_by_sector,
        "errors": {name: summarize(values) for name, values in errors.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--reference-capture", type=Path,
                        default=DEFAULT_REFERENCE_CAPTURE)
    parser.add_argument("--player-dir", type=Path,
                        default=DEFAULT_OPTIMIZER_PLAYER)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()

    profile_path = args.profile.resolve()
    profile = _read_json(profile_path)
    warmup, _ = _validate_time_parameterized_profile(profile)
    commands = profile["commands"]
    scored = commands[warmup:]
    if (len(scored) < 100 or
            not all(row.get("has_reference_position") is True for row in scored)):
        raise ValueError("candidate profile must include a complete spatial lap")
    route = np.asarray([[float(row["reference_x_m"]),
                         float(row["reference_z_m"])] for row in scored],
                       dtype=np.float64)
    if not np.isfinite(route).all():
        raise ValueError("candidate route contains non-finite coordinates")

    reference_capture = args.reference_capture.resolve()
    gate_origin, gate_direction, _ = _finish_gate(reference_capture, warmup)
    map_pixels = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(map_pixels == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M

    run_dir = profile_path.parent
    command_profile = run_dir / "command_profile.json"
    report_path = run_dir / "unity_validation.json"
    if command_profile.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite prior Unity validation in {run_dir}")
    command_profile.write_bytes(profile_path.read_bytes())
    result = _score_trial(
        run_dir, np.zeros(2 * KNOTS), gate_origin, gate_direction, warmup,
        route, args.player_dir.resolve(), args.timeout_s, map_clearance)
    capture_path = Path(result["capture_path"]) if result.get("capture_path") else None
    states_path = run_dir / "unity_wheel_hybrid_states.csv"
    gate_crossings = result.get("forward_gate_crossings_s", [])
    if capture_path is not None and len(gate_crossings) >= 2 and states_path.exists():
        result["prediction_tracking"] = _prediction_tracking_metrics(
            capture_path, states_path,
            float(gate_crossings[0]), float(gate_crossings[1]))
    result.update({
        "candidate_profile": str(profile_path),
        "candidate_predicted_lap_time_s": profile.get(
            "bottom_up_optimizer_predicted_lap_time_s"),
        "validation_method": "established practice batch-mode Unity runner",
        "reference_capture": str(reference_capture),
        "practice_map": str(PRACTICE_MAP),
    })
    report_path.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "unity_status": result.get("unity_status"),
        "lap_valid": result.get("lap_valid"),
        "lap_time_s": result.get("unity_lap_time_s"),
        "predicted_lap_time_s": result.get(
            "candidate_predicted_lap_time_s"),
        "collisions": result.get("collisions_observed"),
        "clearance_min_m": result.get("track_clearance_min_m"),
        "report": str(report_path),
    }, separators=(",", ":")), flush=True)
    return 0 if result.get("lap_valid") else 3


if __name__ == "__main__":
    raise SystemExit(main())
