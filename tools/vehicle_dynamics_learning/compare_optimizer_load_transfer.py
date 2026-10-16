#!/usr/bin/env python3
"""Compare the raceline optimizer's quasi-static wheel loads to Unity labels.

The comparison uses simulator truth only for offline diagnosis: body velocity
and attitude produce a filtered body acceleration, which is passed through the
optimizer's existing load-transfer equation and compared with WheelHit.force.
No parameters are fitted and no runtime model is changed by this script.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
    GRAVITY_MPS2,
    RIGIDBODY_MASS_KG,
    TRACK_M,
    WHEELBASE_ACKERMANN_M,
)


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
VALIDATION = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_exact_replay_20261012_r11/"
    "wheel_dynamics_unity_vehicle_model_parity_source_exact_replay_20261012_r11.csv")
OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "optimizer_load_transfer_comparison_20261015_v2.json")
WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_METADATA = {
    "fl": "front_left_sprung_mass_kg",
    "fr": "front_right_sprung_mass_kg",
    "rl": "rear_left_sprung_mass_kg",
    "rr": "rear_right_sprung_mass_kg",
}
FRONT_SIGN = np.asarray((-1.0, -1.0, 1.0, 1.0))
SIDE_SIGN = np.asarray((-1.0, 1.0, -1.0, 1.0))
WINDOWS = (11, 21, 41)


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, separator, value = line[2:].strip().partition("=")
            if separator:
                result[key] = value
    return result


def _rotate_rows(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    xyz = quaternion[:, :3]
    twice_cross = 2.0 * np.cross(xyz, vectors)
    return (vectors + quaternion[:, 3, None] * twice_cross
            + np.cross(xyz, twice_cross))


def _score(error: np.ndarray) -> dict[str, float | int]:
    error = error[np.isfinite(error)]
    return {
        "samples": int(error.size),
        "bias_n": float(np.mean(error)),
        "rmse_n": float(np.sqrt(np.mean(error**2))),
        "mae_n": float(np.mean(np.abs(error))),
        "p95_abs_n": float(np.quantile(np.abs(error), 0.95)),
    }


def _capture(path: Path) -> dict[str, Any]:
    metadata = _metadata(path)
    dt = float(metadata["fixed_delta_time_s"])
    columns = [
        "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
        "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
    ]
    for wheel in WHEELS:
        columns += [f"{wheel}_grounded", f"{wheel}_contact_force_n"]
    frame = pd.read_csv(path, comment="#", usecols=columns)
    quaternion = frame[["rot_world_x", "rot_world_y", "rot_world_z",
                        "rot_world_w"]].to_numpy(dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=1)[:, None]
    velocity_body = frame[["vel_body_x_mps", "vel_body_y_mps",
                           "vel_body_z_mps"]].to_numpy(dtype=np.float64)
    velocity_world = _rotate_rows(quaternion, velocity_body)
    acceleration_world: dict[int, np.ndarray] = {}
    acceleration_body: dict[int, np.ndarray] = {}
    for window in WINDOWS:
        accel_world = np.column_stack([
            savgol_filter(velocity_world[:, axis], window, 3, deriv=1,
                          delta=dt, mode="interp")
            for axis in range(3)])
        # Inverse quaternion rotation converts the world acceleration into
        # Unity body axes. Unity +z is vehicle-forward; ROS-left is -Unity-x.
        inverse = quaternion.copy()
        inverse[:, :3] *= -1.0
        accel_body = _rotate_rows(inverse, accel_world)
        acceleration_world[window] = accel_world
        acceleration_body[window] = accel_body

    result: dict[str, Any] = {
        "capture": str(path.relative_to(ROOT)),
        "run_id": metadata.get("run_id", path.stem),
        "rows": int(len(frame)),
        "dt_s": dt,
        "windows": {},
    }
    grounded = np.column_stack([
        frame[f"{wheel}_grounded"].to_numpy(dtype=np.float64) > 0.5
        for wheel in WHEELS])
    actual_loads = np.column_stack([
        frame[f"{wheel}_contact_force_n"].to_numpy(dtype=np.float64)
        for wheel in WHEELS])
    speed = np.linalg.norm(velocity_body[:, (0, 2)], axis=1)
    static_loads = np.asarray([
        float(metadata[WHEEL_METADATA[wheel]]) * GRAVITY_MPS2
        for wheel in WHEELS])
    mass = float(metadata["mass_kg"])
    height = CAPTURED_COM_HEIGHT_ABOVE_ROAD_M

    for window in WINDOWS:
        body_accel = acceleration_body[window]
        ax = body_accel[:, 2]
        ay = -body_accel[:, 0]
        longitudinal = mass * height * ax / (2.0 * WHEELBASE_ACKERMANN_M)
        lateral = mass * height * ay / (4.0 * TRACK_M)
        predicted = (static_loads[None, :]
                     + longitudinal[:, None] * FRONT_SIGN[None, :]
                     + lateral[:, None] * SIDE_SIGN[None, :])
        actual_smoothed = np.column_stack([
            savgol_filter(actual_loads[:, wheel], window, 3, mode="interp")
            for wheel in range(4)])
        static_prediction = np.broadcast_to(static_loads, actual_smoothed.shape)
        valid = grounded & np.isfinite(actual_smoothed) \
            & np.isfinite(predicted)
        valid[:window] = False
        valid[-window:] = False
        per_wheel = {}
        errors = predicted - actual_smoothed
        static_errors = static_prediction - actual_smoothed
        for wheel_index, wheel in enumerate(WHEELS):
            mask = valid[:, wheel_index]
            per_wheel[wheel] = {
                "quasi_static": _score(errors[mask, wheel_index]),
                "static_only": _score(static_errors[mask, wheel_index]),
            }
        result["windows"][str(window)] = {
            "smoothing_window_ms": round(window * dt * 1000.0, 3),
            "pooled_all_wheels": _score(errors[valid]),
            "static_only_pooled_all_wheels": _score(static_errors[valid]),
            "per_wheel": per_wheel,
            "speed_bins_mps": {},
            "lateral_acceleration_bins_mps2": {},
            "longitudinal_acceleration_bins_mps2": {},
            "model_diagnostics": {
                "negative_unclamped_load_fraction": float(
                    np.mean(predicted[valid] < 0.0)),
                "predicted_load_min_n": float(np.min(predicted[valid])),
                "predicted_load_max_n": float(np.max(predicted[valid])),
                "actual_load_min_n": float(np.min(actual_smoothed[valid])),
                "actual_load_max_n": float(np.max(actual_smoothed[valid])),
                "rms_load_transfer_from_static_n": float(np.sqrt(np.mean(
                    (predicted[valid] - np.broadcast_to(
                        static_loads, predicted.shape)[valid])**2))),
                "rms_unity_load_change_from_static_n": float(np.sqrt(np.mean(
                    (actual_smoothed[valid] - np.broadcast_to(
                        static_loads, actual_smoothed.shape)[valid])**2))),
            },
        }
        window_result = result["windows"][str(window)]
        for axis_name, values, edges in (
                ("speed_bins_mps", speed, (0, 2, 4, 6, 8, 10, 12, np.inf)),
                ("lateral_acceleration_bins_mps2", np.abs(ay),
                 (0, 0.5, 1, 2, 3, 5, 8, np.inf)),
                ("longitudinal_acceleration_bins_mps2", np.abs(ax),
                 (0, 0.5, 1, 2, 3, 5, 8, np.inf))):
            bins = {}
            for lower, upper in zip(edges[:-1], edges[1:]):
                row_mask = (values >= lower) & (values < upper)
                wheel_mask = valid & row_mask[:, None]
                if np.count_nonzero(wheel_mask) < 20:
                    continue
                label = (f"{lower:g}_plus" if np.isinf(upper)
                         else f"{lower:g}_{upper:g}")
                bins[label] = _score(errors[wheel_mask])
            window_result[axis_name] = bins
    del frame
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=TRAIN)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path in (args.training, args.validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    report = {
        "schema_version": 1,
        "status": "offline diagnostic only; no fit and no model promoted",
        "question": "How well does the optimizer's quasi-static four-wheel load-transfer equation reproduce Unity WheelHit.force when given simulator-truth body acceleration?",
        "equation": {
            "static_load_n": "per-wheel sprung_mass_kg * 9.80665",
            "longitudinal_each_wheel_n": "mass * h * ax / (2 * wheelbase)",
            "lateral_each_wheel_n": "mass * h * ay / (4 * track_width); 50/50 front/rear split",
            "com_height_above_road_m": CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
            "wheelbase_m": WHEELBASE_ACKERMANN_M,
            "track_width_m": TRACK_M,
            "mass_kg": RIGIDBODY_MASS_KG,
            "axes": "ax=Unity body +z; ay=-Unity body +x (vehicle-left)",
        },
        "acceleration_method": "differentiate quaternion-rotated Unity Rigidbody body velocity in world coordinates, then rotate acceleration back into body axes; symmetric Savitzky-Golay windows 11/21/41 samples at the recorded 1 kHz physics interval",
        "score_method": "compare predicted normal loads against same-sample WheelHit.force, with both smoothed by the same window; only grounded wheel samples; no time-shift tuning",
        "captures": {
            "training": _capture(args.training),
            "independent_validation": _capture(args.validation),
        },
        "limitations": [
            "The body-acceleration input is simulator truth and is used only to isolate load-transfer error; this is not an online runtime estimator.",
            "WheelHit.force is the native solver output, which is not itself an independently measured physical tire load.",
            "This analysis does not validate recursive vehicle motion, tire combined-slip forces, or lap-time predictions.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    for name, capture in report["captures"].items():
        print(name, capture["run_id"], f"rows={capture['rows']}")
        for window, scores in capture["windows"].items():
            pooled = scores["pooled_all_wheels"]
            static = scores["static_only_pooled_all_wheels"]
            print(f"  {window} samples: quasi-static RMSE="
                  f"{pooled['rmse_n']:.3f} N, static-only="
                  f"{static['rmse_n']:.3f} N, p95="
                  f"{pooled['p95_abs_n']:.3f} N, "
                  f"bias={pooled['bias_n']:.3f} N, n={pooled['samples']}")
            if window == "21":
                for wheel, wheel_scores in scores["per_wheel"].items():
                    quasi = wheel_scores["quasi_static"]
                    print(f"    {wheel}: RMSE={quasi['rmse_n']:.3f} N, "
                          f"bias={quasi['bias_n']:.3f} N")
                for band in ("5_8", "8_plus"):
                    metric = scores["lateral_acceleration_bins_mps2"].get(band)
                    if metric:
                        print(f"    |ay| {band} m/s^2: RMSE="
                              f"{metric['rmse_n']:.3f} N, "
                              f"n={metric['samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
