#!/usr/bin/env python3
"""Cross-schedule screen for explicit Unity axle-force mechanisms.

This consumes the compact row diagnostics written by
``identify_unity_axle_force_response.py``.  One complete experiment is used
for fitting and the other complete experiment is used for validation in both
directions.  The purpose is to distinguish a repeatable Unity state response
from a schedule-specific gain or an overfit to one excitation.

The candidates are deliberately limited to quantities exposed by the Unity
trace: recorded contact-load/curve proxy, speed, measured slips, serialized
curve demand, wheel rotational state, actual steering angle, and wheel pose.
No tire coefficient, friction coefficient, or hidden residual is created. The
result is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(values.size),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values * values))),
        "p95_abs": float(np.percentile(np.abs(values), 95.0)),
        "mean": float(np.mean(values)),
    }


def _design(frame: pd.DataFrame, axle: str, kind: str) -> np.ndarray:
    proxy = frame[f"{axle}_proxy_n"].to_numpy(float)
    speed = (frame.speed_mps.to_numpy(float) - 11.0) / 5.0
    if kind == "proxy":
        return proxy[:, None]
    if kind == "proxy_speed":
        return np.column_stack((proxy, proxy * speed))
    if kind == "proxy_slip":
        return np.column_stack((
            proxy,
            proxy * frame.max_abs_forward_slip.to_numpy(float),
            proxy * frame.max_abs_sideways_slip.to_numpy(float),
            proxy * speed,
        ))
    if kind == "proxy_curve_demand":
        return np.column_stack((
            proxy,
            proxy * frame[f"{axle}_forward_curve_demand"].to_numpy(float),
            proxy * frame[f"{axle}_sideways_curve_demand"].to_numpy(float),
            proxy * frame[f"{axle}_combined_curve_demand"].to_numpy(float),
        ))
    if kind == "proxy_runtime_state":
        wheel_indices = (0, 1) if axle == "front" else (2, 3)
        omega = np.mean(np.column_stack([
            frame[f"wheel{wheel}_omega_radps"].to_numpy(float)
            for wheel in wheel_indices]), axis=1)
        domega = np.mean(np.column_stack([
            frame[f"wheel{wheel}_domega_radps2"].to_numpy(float)
            for wheel in wheel_indices]), axis=1)
        steer = np.mean(np.column_stack([
            np.abs(frame[f"wheel{wheel}_steer_angle_deg"].to_numpy(float))
            for wheel in wheel_indices]), axis=1)
        pose_y = np.mean(np.column_stack([
            frame[f"wheel{wheel}_pose_body_from_com_y_m"].to_numpy(float)
            for wheel in wheel_indices]), axis=1)
        pose_y_rate = np.mean(np.column_stack([
            frame[f"wheel{wheel}_pose_body_y_rate_mps"].to_numpy(float)
            for wheel in wheel_indices]), axis=1)
        # Fixed engineering scales are units conversions for conditioning;
        # they are not fitted physical parameters.
        return np.column_stack((
            proxy,
            proxy * speed,
            proxy * omega / 200.0,
            proxy * domega / 1000.0,
            proxy * steer / 5.0,
            proxy * pose_y / 0.05,
            proxy * pose_y_rate,
        ))
    raise ValueError(f"unknown design: {kind}")


def _fit_design(train: pd.DataFrame, test: pd.DataFrame, axle: str,
                kind: str) -> dict[str, Any]:
    x_train = _design(train, axle, kind)
    x_test = _design(test, axle, kind)
    y_train = train[f"recovered_{axle}_force_n"].to_numpy(float)
    y_test = test[f"recovered_{axle}_force_n"].to_numpy(float)
    coefficients, _, rank, singular = np.linalg.lstsq(
        x_train, y_train, rcond=None)
    train_error = y_train - x_train @ coefficients
    test_error = y_test - x_test @ coefficients
    return {
        "design": kind,
        "rank": int(rank),
        "columns": int(x_train.shape[1]),
        "coefficients": [float(value) for value in coefficients],
        "singular_values": [float(value) for value in singular],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_error": _stats(train_error),
        "test_error": _stats(test_error),
    }


def _load_power(frame: pd.DataFrame, axle: str, exponent: float) -> np.ndarray:
    load = frame[f"{axle}_load_n"].to_numpy(float)
    demand = frame[f"{axle}_sideways_curve_demand"].to_numpy(float)
    # The fitted coefficient is applied to an explicit per-axle Unity curve
    # load basis.  The exponent is only a mechanism screen, not a promoted
    # tire/load law.
    return 10.0 * demand * np.power(np.maximum(load, 1.0e-6) / 10.0,
                                    exponent)


def _fit_power(train: pd.DataFrame, test: pd.DataFrame, axle: str) -> dict[str, Any]:
    y_train = train[f"recovered_{axle}_force_n"].to_numpy(float)
    y_test = test[f"recovered_{axle}_force_n"].to_numpy(float)

    def fit_gain(exponent: float, frame: pd.DataFrame,
                 target: np.ndarray) -> float:
        basis = _load_power(frame, axle, exponent)
        return float(np.linalg.lstsq(basis[:, None], target, rcond=None)[0][0])

    def objective(exponent: float) -> float:
        basis = _load_power(train, axle, exponent)
        gain = fit_gain(exponent, train, y_train)
        error = y_train - gain * basis
        return float(np.mean(error * error))

    result = minimize_scalar(objective, bounds=(0.10, 2.00), method="bounded")
    exponent = float(result.x)
    gain = fit_gain(exponent, train, y_train)
    train_error = y_train - gain * _load_power(train, axle, exponent)
    test_error = y_test - gain * _load_power(test, axle, exponent)
    return {
        "design": "load_power",
        "load_exponent_p": exponent,
        "gain": gain,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_error": _stats(train_error),
        "test_error": _stats(test_error),
        "promotion": False,
    }


def analyze(old_rows: Path, raceline_rows: Path, output: Path) -> dict[str, Any]:
    old = pd.read_csv(old_rows)
    raceline = pd.read_csv(raceline_rows)
    required = {
        "speed_mps", "max_abs_forward_slip", "max_abs_sideways_slip",
    }
    for axle in ("front", "rear"):
        required |= {
            f"{axle}_proxy_n", f"recovered_{axle}_force_n",
            f"{axle}_load_n", f"{axle}_forward_curve_demand",
            f"{axle}_sideways_curve_demand",
            f"{axle}_combined_curve_demand",
        }
    runtime_state_columns = {
        f"wheel{wheel}_{field}"
        for wheel in range(4)
        for field in ("omega_radps", "domega_radps2", "steer_angle_deg",
                      "pose_body_from_com_y_m", "pose_body_y_rate_mps")
    }
    for name, frame in (("old", old), ("raceline", raceline)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} rows missing columns: {sorted(missing)}")
        missing_runtime = runtime_state_columns - set(frame.columns)
        if missing_runtime:
            raise ValueError(
                f"{name} rows missing runtime-state columns: "
                f"{sorted(missing_runtime)}")

    candidates = (
        "proxy", "proxy_speed", "proxy_slip", "proxy_curve_demand",
        "proxy_runtime_state",
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "offline_unity_axle_force_cross_schedule_screen",
        "source": {"old_rows": str(old_rows), "raceline_rows": str(raceline_rows)},
        "schedule_rows": {"old": int(len(old)), "raceline": int(len(raceline))},
        "directions": {},
        "promotion": False,
        "runtime_use": False,
        "runtime_state_basis": {
            "design": "proxy_runtime_state",
            "terms": [
                "proxy",
                "proxy * speed_normalized",
                "proxy * mean_axle_wheel_omega / 200 rad/s",
                "proxy * mean_axle_wheel_domega / 1000 rad/s^2",
                "proxy * mean_abs_axle_steer / 5 deg",
                "proxy * mean_axle_wheel_pose_body_y / 0.05 m",
                "proxy * mean_axle_wheel_pose_body_y_rate / 1 m/s",
            ],
            "promotion": False,
        },
    }
    for train_name, train, test_name, test in (
            ("old", old, "raceline", raceline),
            ("raceline", raceline, "old", old)):
        direction: dict[str, Any] = {"train": train_name, "test": test_name}
        for axle in ("front", "rear"):
            direction[axle] = {
                "linear_candidates": [
                    _fit_design(train, test, axle, candidate)
                    for candidate in candidates
                ],
                "load_power": _fit_power(train, test, axle),
            }
        result["directions"][f"{train_name}_to_{test_name}"] = direction
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-rows", type=Path, required=True)
    parser.add_argument("--raceline-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.old_rows, args.raceline_rows, args.output)
    compact = {}
    for direction, values in result["directions"].items():
        compact[direction] = {
            axle: {
                "linear": [
                    (item["design"], item["test_error"]["rmse"])
                    for item in values[axle]["linear_candidates"]
                ],
                "load_power": values[axle]["load_power"]["test_error"]["rmse"],
            }
            for axle in ("front", "rear")
        }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
