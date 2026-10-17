#!/usr/bin/env python3
"""Score a recursive PhysX rollout against capture labels kept out of input."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning import evaluate_physx_no_drive_plant_one_step as replay


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_mechanics_dynamic_validation_20261013_r12g/"
    "unity_source_capture/wheel_dynamics_unity_mechanics_dynamic_validation_r12g.csv")
DEFAULT_HORIZONS_MS = (25, 100, 250, 500, 750, 1000)
DEFAULT_HORIZONS_S = (2, 5, 10, 20, 30, 40)


def _yaw(q: np.ndarray) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * y + x * z),
                      1.0 - 2.0 * (y * y + z * z))


def _rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(a, b)))))


def _world_to_body(rotation_xyzw: np.ndarray,
                   vector_world: np.ndarray) -> np.ndarray:
    rotation_xyzw = rotation_xyzw / np.linalg.norm(rotation_xyzw)
    inverse = np.asarray((-rotation_xyzw[0], -rotation_xyzw[1],
                          -rotation_xyzw[2], rotation_xyzw[3]), dtype=np.float32)
    return replay._rotate_xyzw(
        inverse, np.asarray(vector_world, dtype=np.float32)).astype(np.float64)


def _rmse(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _read_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def evaluate(capture: Path, rollout: Path) -> dict:
    predicted = _read_json(rollout)
    if predicted.get("mode") != "recursive_command_only":
        raise ValueError("native output is not a recursive command-only rollout")
    results = predicted["results"]
    if not results:
        raise ValueError("native rollout contains no results")
    start_row = int(results[0]["capture_row"])
    end_row = int(results[-1]["capture_row"]) + 1
    metadata = replay._metadata(capture)
    frame = pd.read_csv(capture, comment="#", usecols=replay._columns(),
                        skiprows=0)
    if start_row < 0 or end_row >= len(frame):
        raise ValueError("rollout labels exceed capture bounds")

    position_3d: list[float] = []
    position_planar: list[float] = []
    orientation: list[float] = []
    yaw_error: list[float] = []
    speed_error: list[float] = []
    forward_speed_error: list[float] = []
    lateral_speed_error: list[float] = []
    yaw_rate_error: list[float] = []
    wheel_error: list[float] = []
    wheel_error_by_name = {wheel: [] for wheel in replay.WHEELS}
    motion_error_rows: list[dict[str, float | int]] = []
    for item in results:
        capture_row = int(item["capture_row"])
        target = frame.iloc[capture_row + 1]
        gt_position = np.asarray(
            [target[f"pos_world_{axis}_m"] for axis in "xyz"], dtype=float)
        pred_position = np.asarray(item["scene_solver_position_world_m"],
                                   dtype=float)
        delta = pred_position - gt_position
        position_3d.append(float(np.linalg.norm(delta)))
        position_planar.append(float(np.linalg.norm(delta[[0, 2]])))

        gt_rotation = np.asarray(
            [target[f"rot_world_{axis}"] for axis in "xyzw"], dtype=float)
        pred_rotation = np.asarray(item["scene_solver_rotation_world_xyzw"],
                                   dtype=float)
        orientation.append(_rotation_error(pred_rotation, gt_rotation))
        yaw_error.append(math.atan2(
            math.sin(_yaw(pred_rotation) - _yaw(gt_rotation)),
            math.cos(_yaw(pred_rotation) - _yaw(gt_rotation))))

        # Speed magnitude is frame-invariant, avoiding assumptions about the
        # legacy capture's missing explicit Rigidbody-vs-Transform frame tag.
        gt_velocity_body = np.asarray(
            [target[f"vel_body_{axis}_mps"] for axis in "xyz"], dtype=float)
        pred_velocity_world = np.asarray(
            item["scene_solver_linear_velocity_world_mps"], dtype=float)
        pred_velocity_body = _world_to_body(pred_rotation, pred_velocity_world)
        forward_delta = float(pred_velocity_body[2] - gt_velocity_body[2])
        lateral_delta = float(pred_velocity_body[0] - gt_velocity_body[0])
        pred_angular_body = _world_to_body(
            pred_rotation,
            item["scene_solver_angular_velocity_world_radps"])
        yaw_rate_delta = float(pred_angular_body[1] - target.omega_body_y_rps)
        forward_speed_error.append(forward_delta)
        lateral_speed_error.append(lateral_delta)
        yaw_rate_error.append(yaw_rate_delta)
        motion_error_rows.append({
            "capture_row": capture_row,
            "time_s": float(item["time_s"]),
            "forward_speed_error_mps": forward_delta,
            "lateral_speed_error_mps": lateral_delta,
            "yaw_rate_error_radps": yaw_rate_delta,
        })
        speed_error.append(float(np.linalg.norm(pred_velocity_world)
                                 - np.linalg.norm(gt_velocity_body)))
        for index, wheel in enumerate(replay.WHEELS):
            gt_omega = float(target[f"{wheel}_rpm"]) * 2.0 * math.pi / 60.0
            error = float(item["wheels"][index]["omega_radps"] - gt_omega)
            wheel_error.append(error)
            wheel_error_by_name[wheel].append(error)

    actual_start = frame.iloc[start_row]
    actual_end = frame.iloc[end_row]
    commands = {
        "mean_speed_mps": float(np.mean(np.linalg.norm(
            frame.loc[start_row:end_row - 1,
                      [f"vel_body_{axis}_mps" for axis in "xyz"]]
            .to_numpy(dtype=float), axis=1))),
        "max_abs_steer_deg": float(np.max(np.abs(
            frame.loc[start_row:end_row - 1,
                      [f"{wheel}_steer_deg" for wheel in replay.WHEELS]]
            .to_numpy(dtype=float)))),
        "mean_abs_steer_deg": float(np.mean(np.max(np.abs(
            frame.loc[start_row:end_row - 1,
                      [f"{wheel}_steer_deg" for wheel in replay.WHEELS]]
            .to_numpy(dtype=float)), axis=1))),
        "throttle_min": float(frame.loc[start_row:end_row - 1,
                                         "throttle_command"].min()),
        "throttle_max": float(frame.loc[start_row:end_row - 1,
                                         "throttle_command"].max()),
        "distance_m": float(np.linalg.norm(
            np.asarray([actual_end[f"pos_world_{axis}_m"] for axis in "xyz"])
            - np.asarray([actual_start[f"pos_world_{axis}_m"] for axis in "xyz"]))),
        "yaw_change_rad": math.atan2(math.sin(
            _yaw(np.asarray([actual_end[f"rot_world_{axis}"]
                             for axis in "xyzw"]))
            - _yaw(np.asarray([actual_start[f"rot_world_{axis}"]
                               for axis in "xyzw"]))), math.cos(
            _yaw(np.asarray([actual_end[f"rot_world_{axis}"]
                             for axis in "xyzw"]))
            - _yaw(np.asarray([actual_start[f"rot_world_{axis}"]
                               for axis in "xyzw"])))),
    }
    horizons = []
    requested_horizons = [
        (horizon_ms / 1000.0, {"horizon_ms": horizon_ms})
        for horizon_ms in DEFAULT_HORIZONS_MS]
    requested_horizons.extend(
        (float(horizon_s), {"horizon_s": horizon_s})
        for horizon_s in DEFAULT_HORIZONS_S)
    for horizon_s, label in requested_horizons:
        n = int(round(horizon_s / predicted["simulation_step_s"]))
        if n > len(results):
            continue
        horizons.append({
            **label,
            "position_3d_at_horizon_m": position_3d[n - 1],
            "position_planar_at_horizon_m": position_planar[n - 1],
            "position_planar_rmse_through_horizon_m": _rmse(
                position_planar[:n]),
            "orientation_at_horizon_rad": orientation[n - 1],
            "yaw_at_horizon_rad": yaw_error[n - 1],
            "yaw_rmse_through_horizon_rad": _rmse(yaw_error[:n]),
            "speed_error_at_horizon_mps": speed_error[n - 1],
            "speed_rmse_through_horizon_mps": _rmse(speed_error[:n]),
            "wheel_omega_rmse_through_horizon_radps": _rmse(
                wheel_error[:4 * n]),
        })
    return {
        "capture": str(capture.resolve().relative_to(ROOT)),
        "rollout": str(rollout.resolve().relative_to(ROOT)),
        "mode": predicted["mode"],
        "initial_capture_row": start_row,
        "duration_s": len(results) * predicted["simulation_step_s"],
        "physics_integration_step_s": predicted.get(
            "physics_step_s", predicted["simulation_step_s"]),
        "scoring_output_stride_steps": predicted.get(
            "output_stride_steps", 1),
        "scoring_output_interval_s": predicted["simulation_step_s"],
        "commands": commands,
        "errors_over_entire_recursive_rollout": {
            "position_planar_rmse_m": _rmse(position_planar),
            "position_planar_p95_m": float(np.quantile(position_planar, 0.95)),
            "position_planar_max_m": float(max(position_planar)),
            "position_3d_rmse_m": _rmse(position_3d),
            "orientation_rmse_rad": _rmse(orientation),
            "orientation_max_rad": float(max(orientation)),
            "yaw_rmse_rad": _rmse(yaw_error),
            "yaw_abs_p95_rad": float(np.quantile(np.abs(yaw_error), 0.95)),
            "yaw_abs_max_rad": float(max(abs(value) for value in yaw_error)),
            "speed_magnitude_rmse_mps": _rmse(speed_error),
            "speed_magnitude_abs_p95_mps": float(np.quantile(
                np.abs(speed_error), 0.95)),
            "speed_magnitude_abs_max_mps": float(max(
                abs(value) for value in speed_error)),
            "body_forward_speed_mps": replay._statistics(forward_speed_error),
            "body_lateral_speed_mps": replay._statistics(lateral_speed_error),
            "body_yaw_rate_radps": replay._statistics(yaw_rate_error),
            "wheel_omega_rmse_radps": _rmse(wheel_error),
            "wheel_omega_abs_p95_radps": float(np.quantile(
                np.abs(wheel_error), 0.95)),
            "wheel_omega_by_wheel_radps": {
                wheel: replay._statistics(values)
                for wheel, values in wheel_error_by_name.items()},
        },
        "largest_forward_speed_error_samples": sorted(
            motion_error_rows,
            key=lambda row: abs(float(row["forward_speed_error_mps"])),
            reverse=True)[:10],
        "largest_lateral_speed_error_samples": sorted(
            motion_error_rows,
            key=lambda row: abs(float(row["lateral_speed_error_mps"])),
            reverse=True)[:10],
        "largest_yaw_rate_error_samples": sorted(
            motion_error_rows,
            key=lambda row: abs(float(row["yaw_rate_error_radps"])),
            reverse=True)[:10],
        "horizons": horizons,
        "limits": [
            "Recorded per-wheel steering/torque are plant inputs, not controller command mapping.",
            "Capture labels are compared offline and are not in the native rollout input.",
            "Legacy twist fields have no explicit frame metadata; only speed magnitude is scored.",
            "This is Unity-source-parity evidence for this captured regime, not Unity runtime parity or production odometry validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.capture, args.rollout)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
