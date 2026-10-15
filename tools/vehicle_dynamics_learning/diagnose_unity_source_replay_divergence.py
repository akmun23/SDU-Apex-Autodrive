#!/usr/bin/env python3
"""Locate the first state divergence in the source-mechanics command replay.

The diagnostic uses the existing training-only force-balance fit and identical
1 kHz command-only windows on r07 and the independent r11 capture. Future truth
is read only at scoring checkpoints. It does not fit on or feed future truth to
the replay.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DEFAULT_FORCE_FIT,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    DT_S,
    HORIZONS_STEPS,
    ROOT,
    WHEELS,
    _advance,
    _initial_acceleration,
    _load_capture,
    _pose_truth,
    _truth_state,
    _wrap,
)


DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_first_divergence_20261013.json")
MAX_HORIZON_STEPS = 1000
WINDOW_STRIDE_STEPS = 5000
CHECKPOINTS = tuple(range(25, MAX_HORIZON_STEPS + 1, 25))
STATE_THRESHOLDS = {
    "u": 0.1,
    "v": 0.1,
    "yaw_rate": 0.1,
    "wheel_omega_rms": 10.0,
    "position": 0.1,
    "heading": 0.1,
}


def _force_parameters() -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray],
                                 np.ndarray]:
    report = json.loads(DEFAULT_FORCE_FIT.read_text(encoding="utf-8"))
    fit = report["curve_model"]["training_fit"]
    params = {
        "longitudinal_force_scale": float(fit["longitudinal_scale"]),
        "longitudinal_zero_slip_tangent": float(
            fit["longitudinal_zero_slip_tangent"]),
        "lateral_force_scale": float(fit["lateral_scale"]),
        "lateral_zero_slip_tangent": float(fit["lateral_zero_slip_tangent"]),
        "wheel_inertia_kgm2": float(report["wheel_spin_balance"][
            "training_capture_fit"]["fitted_shared_wheel_inertia_kgm2"]),
    }
    curves = (
        np.asarray((0.15, 0.9, 0.25, 0.58, 0.8), dtype=np.float64),
        np.asarray((0.01, 1.0, 0.10, 0.50, 1.0), dtype=np.float64),
    )
    static_loads = np.asarray((0.817168832, 0.8160193, 0.9189912,
                               0.9178206), dtype=np.float64) * 9.80665
    return params, curves, static_loads


def _summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "rmse": float(np.sqrt(np.mean(values * values))),
        "abs_p50": float(np.quantile(absolute, 0.5)),
        "abs_p95": float(np.quantile(absolute, 0.95)),
        "abs_max": float(np.max(absolute)),
    }


def _evaluate_capture(path: Path, sample_stride_s: float,
                      brake_impulse_mode: bool) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    if not math.isclose(float(metadata["fixed_delta_time_s"]), DT_S,
                        rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"expected {DT_S}s fixed step in {path}")
    acceleration_x, acceleration_y = _initial_acceleration(data, DT_S)
    params, curves, static_loads = _force_parameters()
    window_steps = MAX_HORIZON_STEPS
    stride_steps = round(sample_stride_s / DT_S)
    starts = list(range(100, len(data["speed_mps"]) - window_steps - 1,
                        stride_steps))
    starts = [start for start in starts if data["speed_mps"][start] >= 2.0]
    if not starts:
        raise ValueError(f"no qualifying rollout windows in {path}")

    state_errors: dict[int, dict[str, list[float]]] = {
        step: {name: [] for name in (
            "u", "v", "yaw_rate", "wheel_omega_rms", "wheel_omega_each",
            "steering", "position", "heading")}
        for step in CHECKPOINTS}
    first_crossing: dict[str, list[float]] = {
        name: [] for name in STATE_THRESHOLDS}
    window_rows: list[dict[str, Any]] = []
    brake_mode = ("unilateral_impulse_lock" if brake_impulse_mode
                  else "continuous_coulomb")

    for start in starts:
        state = _truth_state(data, start, acceleration_x, acceleration_y)
        pose = _pose_truth(data, start)
        initial_speed = float(data["speed_mps"][start])
        initial_steering = abs(float(state[7]))
        crossing = {name: None for name in STATE_THRESHOLDS}
        for offset in range(window_steps):
            index = start + offset
            state_next = _advance(
                state, data["steering_command"][index],
                data["throttle_command"][index], 0.5,
                params["longitudinal_force_scale"],
                params["longitudinal_zero_slip_tangent"],
                params["lateral_force_scale"],
                params["lateral_zero_slip_tangent"],
                params["wheel_inertia_kgm2"], curves[0], curves[1],
                static_loads, brake_impulse_mode, 0)
            u_mid = 0.5 * (state[0] + state_next[0])
            v_mid = 0.5 * (state[1] + state_next[1])
            r_mid = 0.5 * (state[2] + state_next[2])
            heading_mid = pose[2] + 0.5 * r_mid * DT_S
            pose_next = pose.copy()
            pose_next[0] += (u_mid * math.sin(heading_mid)
                             + v_mid * math.cos(heading_mid)) * DT_S
            pose_next[1] += (u_mid * math.cos(heading_mid)
                             - v_mid * math.sin(heading_mid)) * DT_S
            pose_next[2] += r_mid * DT_S
            if not (np.isfinite(state_next).all()
                    and np.isfinite(pose_next).all()):
                raise FloatingPointError(
                    f"non-finite replay in {path} at sample {index}")
            state, pose = state_next, pose_next
            elapsed_steps = offset + 1

            truth = _truth_state(data, index + 1,
                                 acceleration_x, acceleration_y)
            state_error = state[:7] - truth[:7]
            pose_error = pose - _pose_truth(data, index + 1)
            pose_error[2] = _wrap(float(pose_error[2]))
            row = {
                "u": float(state_error[0]),
                "v": float(state_error[1]),
                "yaw_rate": float(state_error[2]),
                "wheel_omega_rms": float(np.sqrt(np.mean(state_error[3:7]**2))),
                "wheel_omega_each": state_error[3:7].tolist(),
                "steering": float(state[7] - truth[7]),
                "position": float(np.linalg.norm(pose_error[:2])),
                "heading": float(pose_error[2]),
            }
            for name, threshold in STATE_THRESHOLDS.items():
                if crossing[name] is None and abs(row[name]) >= threshold:
                    crossing[name] = elapsed_steps * DT_S
            if elapsed_steps in state_errors:
                for name, value in row.items():
                    if name == "wheel_omega_each":
                        state_errors[elapsed_steps][name].extend(value)
                    else:
                        state_errors[elapsed_steps][name].append(value)
        for name, value in crossing.items():
            if value is not None:
                first_crossing[name].append(value)
        window_rows.append({
            "start_sample": start,
            "start_speed_mps": initial_speed,
            "start_abs_steering_rad": initial_steering,
            "first_threshold_crossing_s": crossing,
        })

    checkpoints: dict[str, Any] = {}
    for step, values in state_errors.items():
        checkpoints[f"{step * DT_S:.3f}s"] = {
            name: _summarize(np.asarray(errors))
            for name, errors in values.items() if errors
        }

    crossing_summary = {}
    for name, values in first_crossing.items():
        crossing_summary[name] = {
            "threshold": STATE_THRESHOLDS[name],
            "windows_crossing": len(values),
            "fraction_crossing_within_1s": len(values) / len(starts),
            "median_first_crossing_s": (
                float(np.median(values)) if values else None),
            "p95_first_crossing_s": (
                float(np.quantile(values, 0.95)) if values else None),
        }
    return {
        "capture": str(path.relative_to(ROOT)),
        "brake_mode": brake_mode,
        "rows": len(data["speed_mps"]),
        "window_count": len(starts),
        "window_stride_s": stride_steps * DT_S,
        "start_speed_p05_p50_p95_mps": np.quantile(
            [row["start_speed_mps"] for row in window_rows],
            (0.05, 0.5, 0.95)).tolist(),
        "start_abs_steering_p05_p50_p95_rad": np.quantile(
            [row["start_abs_steering_rad"] for row in window_rows],
            (0.05, 0.5, 0.95)).tolist(),
        "checkpoint_errors": checkpoints,
        "first_threshold_crossing": crossing_summary,
        "window_diagnostics": window_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--window-stride-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.window_stride_seconds <= 0.0:
        parser.error("window stride must be positive")
    report = {
        "schema_version": 1,
        "purpose": "state-first divergence diagnosis of the existing command-only source-mechanics replay",
        "model": "r07-only force-balance fit; 1 ms midpoint integration; legacy slip law; continuous Coulomb brake branch",
        "future_capture_truth_used_as_input": False,
        "training_capture_by_brake_mode": {
            mode: _evaluate_capture(
                args.train, args.window_stride_seconds,
                brake_impulse_mode=(mode == "unilateral_impulse_lock"))
            for mode in ("continuous_coulomb", "unilateral_impulse_lock")
        },
        "independent_heldout_capture_by_brake_mode": {
            mode: _evaluate_capture(
                args.heldout, args.window_stride_seconds,
                brake_impulse_mode=(mode == "unilateral_impulse_lock"))
            for mode in ("continuous_coulomb", "unilateral_impulse_lock")
        },
        "interpretation_note": (
            "Crossing times are descriptive per-window thresholds, not a causal proof. "
            "The first variable to cross is the next component to inspect."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    for split in ("training_capture_by_brake_mode",
                  "independent_heldout_capture_by_brake_mode"):
        for mode, item in report[split].items():
            print(split, mode, "windows", item["window_count"])
            for key in ("0.025s", "0.100s", "0.250s", "0.500s", "1.000s"):
                row = item["checkpoint_errors"][key]
                print(key, "u", round(row["u"]["rmse"], 4),
                      "v", round(row["v"]["rmse"], 4),
                      "r", round(row["yaw_rate"]["rmse"], 4),
                      "wheel", round(row["wheel_omega_rms"]["rmse"], 3),
                      "pos", round(row["position"]["rmse"], 4))
            print("thresholds", item["first_threshold_crossing"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
