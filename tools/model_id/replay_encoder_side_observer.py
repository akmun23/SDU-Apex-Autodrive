#!/usr/bin/env python3
"""Offline paired-encoder observer replay on a source-stamped simulator trace.

The observer receives only encoder, IMU acceleration, IMU yaw-rate, and IMU
orientation fields. Packet pose/velocity and actuator feedback are kept out of
the replay and used only for offline scoring/regime labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_ROOT = _REPO_ROOT / "sdu_apex_autodrive"
sys.path.insert(0, str(_PACKAGE_ROOT))

from sdu_apex_autodrive.odometry_analysis.packet_reconstruction import (  # noqa: E402
    _normalize_imu_yaw,
)
from sdu_apex_autodrive.odometry_analysis.reference_observer import (  # noqa: E402
    ReferenceObserver,
)


MIN_DT_S = 0.015
MAX_DT_S = 0.035
DT_QUANTIZATION_TOLERANCE_S = 1.0e-9
POSE_SPEED_WINDOW_S = 0.100


def _finite(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _yaw(row: dict[str, str]) -> float:
    x = _finite(row, "simulator_orientation_quaternion_x")
    y = _finite(row, "simulator_orientation_quaternion_y")
    z = _finite(row, "simulator_orientation_quaternion_z")
    w = _finite(row, "simulator_orientation_quaternion_w")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "signed_mean": None, "mae": None,
                "abs_p50": None, "abs_p95": None, "abs_max": None}
    absolute = sorted(abs(value) for value in values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(absolute) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return absolute[lower]
        alpha = position - lower
        return absolute[lower] * (1.0 - alpha) + absolute[upper] * alpha

    return {
        "samples": len(values),
        "signed_mean": sum(values) / len(values),
        "mae": sum(abs(value) for value in values) / len(values),
        "abs_p50": percentile(0.50),
        "abs_p95": percentile(0.95),
        "abs_max": absolute[-1],
    }


def _read_trace(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError(f"{path} needs at least two source packets")
    required = {
        "simulation_time_s", "simulation_physics_step", "packet_sequence",
        "simulator_position_x", "simulator_position_y",
        "simulator_orientation_quaternion_x",
        "simulator_orientation_quaternion_y",
        "simulator_orientation_quaternion_z",
        "simulator_orientation_quaternion_w",
        "simulator_linear_velocity_x", "simulator_linear_velocity_y",
        "simulator_angular_velocity_z", "simulator_linear_acceleration_x",
        "simulator_linear_acceleration_y", "simulator_encoder_angles_left",
        "simulator_encoder_angles_right", "simulator_feedback_throttle_norm",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")

    times = [_finite(row, "simulation_time_s") for row in rows]
    steps = [int(_finite(row, "simulation_physics_step")) for row in rows]
    sequences = [int(_finite(row, "packet_sequence")) for row in rows]
    if any(second <= first for first, second in zip(times, times[1:])):
        raise ValueError("source simulation times are not strictly increasing")
    if any(second <= first for first, second in zip(steps, steps[1:])):
        raise ValueError("source physics steps are not strictly increasing")
    if any(second != first + 1 for first, second in zip(sequences, sequences[1:])):
        raise ValueError("packet sequence gap; refusing observer replay")
    dts = [second - first for first, second in zip(times, times[1:])]
    outside = [dt for dt in dts if not (
        MIN_DT_S - DT_QUANTIZATION_TOLERANCE_S <= dt <=
        MAX_DT_S + DT_QUANTIZATION_TOLERANCE_S)]
    if outside:
        raise ValueError(
            f"{len(outside)} source intervals fall outside {MIN_DT_S:.3f}--"
            f"{MAX_DT_S:.3f} s; refusing to retime or fill the trace")
    return rows, {
        "packet_count": len(rows),
        "transition_count": len(dts),
        "source_dt_s_median": float(np.median(dts)),
        "source_dt_s_p95": float(np.percentile(dts, 95)),
        "source_dt_s_max": max(dts),
        "boundary_tolerance_s": DT_QUANTIZATION_TOLERANCE_S,
        "sequence_gaps": 0,
        "source_samples_are_not_upsampled": True,
    }


def _make_observations(rows: list[dict[str, str]]) -> pd.DataFrame:
    """Whitelist sensor fields; no truth or actuation feedback enters replay."""
    observations = pd.DataFrame([
        {
            "stamp_s": _finite(row, "simulation_time_s"),
            "left_angle_rad": _finite(row, "simulator_encoder_angles_left"),
            "right_angle_rad": _finite(row, "simulator_encoder_angles_right"),
            "ax_mps2": _finite(row, "simulator_linear_acceleration_x"),
            "ay_mps2": _finite(row, "simulator_linear_acceleration_y"),
            "yaw_rate_radps": _finite(row, "simulator_angular_velocity_z"),
            "yaw_rad": _yaw(row),
        }
        for row in rows
    ])
    _normalize_imu_yaw(observations)
    return observations


def _truth_pose_speed(rows: list[dict[str, str]], times: list[float]) -> list[float | None]:
    """Estimate GPS-pose point speed over the same 100 ms window as the wheel signal."""
    x = [_finite(row, "simulator_position_x") for row in rows]
    y = [_finite(row, "simulator_position_y") for row in rows]
    yaw = [_yaw(row) for row in rows]
    result: list[float | None] = [None] * len(rows)
    for index, time_s in enumerate(times):
        start = bisect_right(times, time_s - POSE_SPEED_WINDOW_S) - 1
        if start < 0 or start >= index:
            continue
        dt = time_s - times[start]
        yaw_mid = yaw[start] + 0.5 * _wrap(yaw[index] - yaw[start])
        dx = x[index] - x[start]
        dy = y[index] - y[start]
        u = (dx * math.cos(yaw_mid) + dy * math.sin(yaw_mid)) / dt
        v = (-dx * math.sin(yaw_mid) + dy * math.cos(yaw_mid)) / dt
        result[index] = math.hypot(u, v)
    return result


def _score_variant(
    estimates: list[Any], rows: list[dict[str, str]],
    truth_pose_speeds: list[float | None],
) -> dict[str, Any]:
    times = [_finite(row, "simulation_time_s") for row in rows]
    truth_yaw = [_yaw(row) for row in rows]
    initial_yaw = truth_yaw[0]
    c0, s0 = math.cos(initial_yaw), math.sin(initial_yaw)
    initial_x = _finite(rows[0], "simulator_position_x")
    initial_y = _finite(rows[0], "simulator_position_y")

    groups: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in (
            "longitudinal_m", "lateral_m", "position_euclidean_m",
            "speed_pose_point_mps", "speed_com_mps")}
        for name in ("moving", "powered_turn", "powered_turn_positive_yaw",
                     "powered_turn_negative_yaw", "full_brake_turn",
                     "full_brake_straight")
    }
    for index, estimate in enumerate(estimates):
        if index == 0:
            continue
        row = rows[index]
        truth_x_world = _finite(row, "simulator_position_x") - initial_x
        truth_y_world = _finite(row, "simulator_position_y") - initial_y
        truth_x = c0 * truth_x_world + s0 * truth_y_world
        truth_y = -s0 * truth_x_world + c0 * truth_y_world
        yaw_local = _wrap(truth_yaw[index] - initial_yaw)
        dx = estimate.x_m - truth_x
        dy = estimate.y_m - truth_y
        along = math.cos(yaw_local) * dx + math.sin(yaw_local) * dy
        lateral = -math.sin(yaw_local) * dx + math.cos(yaw_local) * dy
        speed_pose = truth_pose_speeds[index]
        speed_com = math.hypot(
            _finite(row, "simulator_linear_velocity_x"),
            _finite(row, "simulator_linear_velocity_y"))
        if speed_pose is None or speed_pose <= 0.5:
            continue
        measured = estimate.speed_mps - speed_pose
        groups["moving"]["longitudinal_m"].append(along)
        groups["moving"]["lateral_m"].append(lateral)
        groups["moving"]["position_euclidean_m"].append(math.hypot(dx, dy))
        groups["moving"]["speed_pose_point_mps"].append(measured)
        groups["moving"]["speed_com_mps"].append(estimate.speed_mps - speed_com)

        throttle = _finite(row, "simulator_feedback_throttle_norm")
        yaw_rate = _finite(row, "simulator_angular_velocity_z")
        is_turn = abs(yaw_rate) >= 0.1
        if throttle > 1.0e-4 and is_turn:
            groups["powered_turn"]["longitudinal_m"].append(along)
            groups["powered_turn"]["lateral_m"].append(lateral)
            groups["powered_turn"]["position_euclidean_m"].append(math.hypot(dx, dy))
            groups["powered_turn"]["speed_pose_point_mps"].append(measured)
            groups["powered_turn"]["speed_com_mps"].append(estimate.speed_mps - speed_com)
            direction = ("powered_turn_positive_yaw" if yaw_rate > 0.0 else
                         "powered_turn_negative_yaw")
            groups[direction]["longitudinal_m"].append(along)
            groups[direction]["lateral_m"].append(lateral)
            groups[direction]["position_euclidean_m"].append(math.hypot(dx, dy))
            groups[direction]["speed_pose_point_mps"].append(measured)
            groups[direction]["speed_com_mps"].append(estimate.speed_mps - speed_com)
        elif throttle <= 1.0e-4:
            brake_group = "full_brake_turn" if is_turn else "full_brake_straight"
            groups[brake_group]["longitudinal_m"].append(along)
            groups[brake_group]["lateral_m"].append(lateral)
            groups[brake_group]["position_euclidean_m"].append(math.hypot(dx, dy))
            groups[brake_group]["speed_pose_point_mps"].append(measured)
            groups[brake_group]["speed_com_mps"].append(estimate.speed_mps - speed_com)

    report: dict[str, Any] = {}
    for name, metrics in groups.items():
        report[name] = {
            metric: _stats(values) for metric, values in metrics.items()
        }
    report["score_contract"] = {
        "position_axes": "truth-heading longitudinal and lateral, separately",
        "euclidean_position": "secondary only",
        "simulator_truth": "offline scoring and powered/braking regime labels only",
        "position_reference": "source GPS pose point, translation-aligned at first packet",
        "speed_primary_reference": "100 ms source-time GPS pose-point displacement",
        "speed_secondary_reference": "simulator COM body-speed magnitude",
    }
    return report


def replay(trace: Path, observer_config: Path) -> dict[str, Any]:
    rows, source_quality = _read_trace(trace)
    observations = _make_observations(rows)
    times = [_finite(row, "simulation_time_s") for row in rows]
    pose_speeds = _truth_pose_speed(rows, times)
    variants: dict[str, Any] = {}
    for name, select_side in (("paired_mean", False),
                              ("yaw_direction_selected", True)):
        observer = ReferenceObserver.from_yaml(observer_config)
        observer.use_yaw_direction_encoder_side = select_side
        estimates = [observer.update(row) for _, row in observations.iterrows()]
        variants[name] = _score_variant(estimates, rows, pose_speeds)
    return {
        "schema_version": 1,
        "status": "offline_observer_replay_not_runtime_promotion",
        "trace": str(trace),
        "observer_config": str(observer_config),
        "runtime_input_fields": [
            "left_encoder_angle", "right_encoder_angle", "imu_acceleration_x",
            "imu_acceleration_y", "imu_yaw_rate", "imu_orientation",
        ],
        "source_quality": source_quality,
        "variants": variants,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("simulator_packets_csv", type=Path)
    parser.add_argument("--observer-config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    report = replay(args.simulator_packets_csv, args.observer_config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n",
                                encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_json),
        "status": report["status"],
        "source_quality": report["source_quality"],
        "moving_paired": report["variants"]["paired_mean"]["moving"],
        "moving_direction_selected": report["variants"][
            "yaw_direction_selected"]["moving"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
