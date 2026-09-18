#!/usr/bin/env python3
"""Offline ablation: compare left, right, and paired rear encoders.

Only the encoder angles are used to form candidate speed measurements. The
simulator pose is used strictly as an offline scoring target and for labelling
straight/turn and powered/braking regimes. Nothing produced here is consumed
by a runtime node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any

import yaml


_CANDIDATES = (
    "paired_mean", "left_only", "right_only", "yaw_direction_selected")


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _yaw(row: dict[str, str]) -> float | None:
    x = _finite(row.get("simulator_orientation_quaternion_x"))
    y = _finite(row.get("simulator_orientation_quaternion_y"))
    z = _finite(row.get("simulator_orientation_quaternion_z"))
    w = _finite(row.get("simulator_orientation_quaternion_w"))
    if None in (x, y, z, w):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])


def _error_summary(errors: list[float]) -> dict[str, float | int | None]:
    absolute = [abs(value) for value in errors]
    return {
        "samples": len(errors),
        "signed_bias_mps": sum(errors) / len(errors) if errors else None,
        "mae_mps": sum(absolute) / len(absolute) if absolute else None,
        "abs_p50_mps": _percentile(absolute, 0.50),
        "abs_p95_mps": _percentile(absolute, 0.95),
        "abs_max_mps": max(absolute) if absolute else None,
    }


def _scale_speed(raw_speed: float, speeds: list[float], scales: list[float]) -> float:
    if len(speeds) < 2 or len(speeds) != len(scales):
        raise ValueError("observer speed-scale table must contain matching points")
    if raw_speed <= speeds[0]:
        scale = scales[0]
    elif raw_speed >= speeds[-1]:
        scale = scales[-1]
    else:
        upper = bisect_right(speeds, raw_speed)
        lower = upper - 1
        alpha = (raw_speed - speeds[lower]) / (speeds[upper] - speeds[lower])
        scale = scales[lower] + alpha * (scales[upper] - scales[lower])
    return raw_speed * scale


def score_packets(
    rows: list[dict[str, str]],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Score rolling left/right/mean encoder speed from source-timed packets."""
    radius = float(parameters["wheel_radius_m"])
    window = float(parameters["wheel_speed_window_s"])
    speeds = [float(value) for value in parameters["wheel_speed_scale_speeds_mps"]]
    scales = [float(value) for value in parameters["wheel_speed_scale_values"]]
    if radius <= 0.0 or window <= 0.0:
        raise ValueError("wheel radius and encoder window must be positive")

    samples: list[dict[str, float]] = []
    for row in rows:
        values = {
            "time": _finite(row.get("simulation_time_s")),
            "x": _finite(row.get("simulator_position_x")),
            "y": _finite(row.get("simulator_position_y")),
            "left": _finite(row.get("simulator_encoder_angles_left")),
            "right": _finite(row.get("simulator_encoder_angles_right")),
            "yaw": _yaw(row),
            "yaw_rate": _finite(row.get("simulator_angular_velocity_z")),
            "throttle": _finite(row.get("simulator_feedback_throttle_norm")),
        }
        if any(value is None for value in values.values()):
            continue
        samples.append({key: float(value) for key, value in values.items()})
    samples.sort(key=lambda sample: sample["time"])
    if len(samples) < 2:
        raise ValueError("need at least two complete source-timed packets")

    errors: dict[str, dict[str, list[float]]] = {
        regime: {candidate: [] for candidate in _CANDIDATES}
        for regime in ("all_moving", "powered_straight", "powered_turn",
                       "full_brake_straight", "full_brake_turn",
                       "powered_turn_positive_yaw", "powered_turn_negative_yaw",
                       "full_brake_turn_positive_yaw",
                       "full_brake_turn_negative_yaw")
    }
    moving_count = 0
    repeated = {"left": 0, "right": 0, "both": 0, "left_only": 0, "right_only": 0}
    window_samples = 0
    times = [sample["time"] for sample in samples]

    for index in range(1, len(samples)):
        current = samples[index]
        previous = samples[index - 1]
        dt = current["time"] - previous["time"]
        if dt <= 0.0:
            continue
        yaw_mid = previous["yaw"] + 0.5 * _wrap(current["yaw"] - previous["yaw"])
        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]
        truth_u_instant = (dx * math.cos(yaw_mid) + dy * math.sin(yaw_mid)) / dt
        truth_v_instant = (-dx * math.sin(yaw_mid) + dy * math.cos(yaw_mid)) / dt
        moving = math.hypot(truth_u_instant, truth_v_instant) > 0.5
        left_delta = current["left"] - previous["left"]
        right_delta = current["right"] - previous["right"]
        left_fresh = abs(left_delta) > 1.0e-10
        right_fresh = abs(right_delta) > 1.0e-10
        if moving:
            moving_count += 1
            repeated["left"] += int(not left_fresh)
            repeated["right"] += int(not right_fresh)
            repeated["both"] += int(not left_fresh and not right_fresh)
            repeated["left_only"] += int(not left_fresh and right_fresh)
            repeated["right_only"] += int(left_fresh and not right_fresh)

        start = bisect_right(times, current["time"] - window) - 1
        if start < 0 or start >= index:
            continue
        interval = current["time"] - samples[start]["time"]
        if interval <= 0.0:
            continue
        left_rate = radius * (current["left"] - samples[start]["left"]) / interval
        right_rate = radius * (current["right"] - samples[start]["right"]) / interval
        if not moving:
            continue
        # Compare each rolling encoder estimate to displacement over the same
        # source-time window; adjacent-packet differencing would unfairly
        # compare a 100 ms wheel average to a noisy 25 ms pose derivative.
        window_yaw = samples[start]["yaw"] + 0.5 * _wrap(
            current["yaw"] - samples[start]["yaw"])
        window_dx = current["x"] - samples[start]["x"]
        window_dy = current["y"] - samples[start]["y"]
        truth_u_window = (
            window_dx * math.cos(window_yaw) + window_dy * math.sin(window_yaw)
        ) / interval
        window_samples += 1
        candidate_speeds = {
            "paired_mean": _scale_speed(abs(0.5 * (left_rate + right_rate)), speeds, scales),
            "left_only": _scale_speed(abs(left_rate), speeds, scales),
            "right_only": _scale_speed(abs(right_rate), speeds, scales),
        }
        power_mode = (
            "full_brake" if abs(current["throttle"]) <= 1.0e-4 else "powered"
            if current["throttle"] > 1.0e-4 else None
        )
        if power_mode is None:
            continue
        steering_mode = "straight" if abs(current["yaw_rate"]) < 0.1 else "turn"
        regime = f"{power_mode}_{steering_mode}"
        if power_mode == "powered" and steering_mode == "turn":
            selected = "left_only" if current["yaw_rate"] > 0.0 else "right_only"
            candidate_speeds["yaw_direction_selected"] = candidate_speeds[selected]
        else:
            # Keep the established paired estimate on straights and while
            # braking: the dedicated full-brake traces show both encoders
            # freezing together, so a side selector cannot restore motion.
            candidate_speeds["yaw_direction_selected"] = candidate_speeds[
                "paired_mean"]
        regimes = [regime]
        if steering_mode == "turn":
            direction = "positive_yaw" if current["yaw_rate"] > 0.0 else "negative_yaw"
            regimes.append(f"{power_mode}_turn_{direction}")
        for candidate, estimate in candidate_speeds.items():
            error = estimate - truth_u_window
            errors["all_moving"][candidate].append(error)
            for selected_regime in regimes:
                errors[selected_regime][candidate].append(error)

    return {
        "schema_version": 1,
        "offline_only": True,
        "ground_truth_use": "source-time scoring and offline regime labels only",
        "sensor_inputs": ["left_encoder_angle", "right_encoder_angle"],
        "observer_calibration": {
            "wheel_radius_m": radius,
            "wheel_speed_window_s": window,
            "speed_scale_speeds_mps": speeds,
            "speed_scale_values": scales,
        },
        "freshness": {
            "moving_adjacent_intervals": moving_count,
            "moving_intervals_with_windowed_speed": window_samples,
            "left_repeated_count": repeated["left"],
            "right_repeated_count": repeated["right"],
            "both_repeated_count": repeated["both"],
            "left_only_stale_count": repeated["left_only"],
            "right_only_stale_count": repeated["right_only"],
            "left_only_stale_fraction": (
                repeated["left_only"] / moving_count if moving_count else None),
            "right_only_stale_fraction": (
                repeated["right_only"] / moving_count if moving_count else None),
        },
        "error_by_regime": {
            regime: {
                candidate: _error_summary(candidate_errors)
                for candidate, candidate_errors in candidate_map.items()
            }
            for regime, candidate_map in errors.items()
        },
        "promotion_note": (
            "This is a speed-sensor ablation, not an odometry promotion. "
            "yaw_direction_selected uses left-only for positive yaw and "
            "right-only for negative yaw in powered turns, and paired mean "
            "elsewhere. Promote only after a held-out full observer replay "
            "shows an aggregate pose/speed benefit without a material "
            "regression in either turn direction."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator_packets_csv", type=Path)
    parser.add_argument("--observer-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    with args.simulator_packets_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    config = yaml.safe_load(args.observer_config.read_text(encoding="utf-8"))
    parameters = config["sensor_odometry"]["ros__parameters"]
    result = score_packets(rows, parameters)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
