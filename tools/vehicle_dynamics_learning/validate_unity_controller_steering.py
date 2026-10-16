#!/usr/bin/env python3
"""No-fit source replay of Unity steering against existing wheel captures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning.unity_vehicle_controller_source import (
    STEERING_LIMIT_DEG,
    STEERING_RATE_DEG_S,
    TRACK_WIDTH_MM,
    WHEELBASE_MM,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURES = (
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv",
    ROOT / "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv",
    ROOT / "live_runs/unity_mechanics_dynamic_validation_20261013_r12g/"
    "unity_source_capture/"
    "wheel_dynamics_unity_mechanics_dynamic_validation_r12g.csv",
)


def _metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, separator, value = line[2:].strip().partition("=")
            if separator:
                metadata[key] = value
    return metadata


def validate_capture(path: Path) -> dict[str, float | int | str]:
    """Replay steering from each previous captured state and current command.

    Only five numeric columns are read; the 1 kHz captures are several hundred
    megabytes each. All source actuator arithmetic is explicitly float32.
    """
    metadata = _metadata(path)
    if "fixed_delta_time_s" not in metadata:
        raise ValueError(f"capture lacks fixed_delta_time_s: {path}")
    nominal_dt = float(metadata["fixed_delta_time_s"])
    columns = ("sim_time_s", "steering_command", "fl_steer_deg",
               "fr_steer_deg")
    data = pd.read_csv(path, comment="#", usecols=columns)
    raw_count = len(data)
    data = data.dropna(subset=columns).reset_index(drop=True)
    time = data["sim_time_s"].to_numpy(dtype=np.float64)
    command = np.clip(data["steering_command"].to_numpy(dtype=np.float32),
                      np.float32(-1.0), np.float32(1.0))
    observed_left = data["fl_steer_deg"].to_numpy(dtype=np.float64)
    observed_right = data["fr_steer_deg"].to_numpy(dtype=np.float64)
    left_tangent = np.tan(np.deg2rad(observed_left))
    right_tangent = np.tan(np.deg2rad(observed_right))
    twice_wheelbase = 2.0 * float(WHEELBASE_MM)
    center_left = (twice_wheelbase * left_tangent /
                   (twice_wheelbase - float(TRACK_WIDTH_MM) * left_tangent))
    center_right = (twice_wheelbase * right_tangent /
                    (twice_wheelbase + float(TRACK_WIDTH_MM) * right_tangent))
    center_observed = np.rad2deg(np.arctan(0.5 * (center_left + center_right)))
    # Ackermann inversion has a few-microdegree float error at the stops. The
    # C# actuator cannot exceed the serialized +/-30 degree range.
    center_state = np.clip(center_observed, -30.0, 30.0).astype(np.float32)
    dt = np.diff(time)
    valid_dt = np.isclose(dt, nominal_dt, rtol=0.0, atol=2e-7)
    previous = center_state[:-1]
    current_command = command[1:]
    increment = (np.float32(STEERING_RATE_DEG_S)
                 * dt.astype(np.float32))
    threshold = previous / np.float32(STEERING_LIMIT_DEG)
    decreasing = current_command > threshold
    stepped = np.where(decreasing, previous - increment,
                       previous + increment).astype(np.float32)
    target = (-current_command * np.float32(STEERING_LIMIT_DEG)).astype(
        np.float32)
    predicted_center = np.where(
        decreasing, np.where(stepped <= target, target, stepped),
        np.where(stepped >= target, target, stepped)).astype(np.float32)
    predicted_center = np.clip(predicted_center, -30.0, 30.0)
    radians_per_degree = np.float32(np.pi / 180.0)
    degree_per_radian = np.float32(180.0 / np.pi)
    tangent = np.tan(predicted_center * radians_per_degree).astype(np.float32)
    numerator = (np.float32(2.0) * np.float32(WHEELBASE_MM) * tangent)
    left_pred = (np.arctan(numerator /
                           (np.float32(2.0) * np.float32(WHEELBASE_MM)
                            + np.float32(TRACK_WIDTH_MM) * tangent))
                 * degree_per_radian).astype(np.float32)
    right_pred = (np.arctan(numerator /
                            (np.float32(2.0) * np.float32(WHEELBASE_MM)
                             - np.float32(TRACK_WIDTH_MM) * tangent))
                  * degree_per_radian).astype(np.float32)
    left_error = left_pred.astype(np.float64) - observed_left[1:]
    right_error = right_pred.astype(np.float64) - observed_right[1:]
    valid_left = left_error[valid_dt]
    valid_right = right_error[valid_dt]
    center_error = predicted_center.astype(np.float64)[valid_dt] - center_observed[1:][valid_dt]
    if not len(center_error):
        raise ValueError(f"capture has no valid consecutive steps: {path}")
    errors = np.concatenate((valid_left, valid_right))
    clamp_count = int(np.count_nonzero(
        (np.abs(predicted_center - target) < np.float32(2e-6)) & valid_dt))
    return {
        "capture": str(path.resolve().relative_to(ROOT)),
        "rows": raw_count,
        "missing_rows": raw_count - len(data),
        "valid_steps": int(np.count_nonzero(valid_dt)),
        "invalid_dt_steps": int(len(dt) - np.count_nonzero(valid_dt)),
        "target_clamp_steps": clamp_count,
        "wheel_angle_rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "wheel_angle_p99_abs_deg": float(np.quantile(np.abs(errors), 0.99)),
        "wheel_angle_max_abs_deg": float(np.max(np.abs(errors))),
        "center_angle_rmse_deg": float(np.sqrt(np.mean(center_error**2))),
        "center_angle_max_abs_deg": float(np.max(np.abs(center_error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", type=Path,
                        default=DEFAULT_CAPTURES)
    args = parser.parse_args()
    for capture in args.captures:
        print(validate_capture(capture))


if __name__ == "__main__":
    main()
