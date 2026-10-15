#!/usr/bin/env python3
"""Evaluate the frozen yaw-acceleration residual in recursive 1 kHz rollouts.

Each window starts from one simulator-truth state. Thereafter the model sees
only its own predicted state and the recorded commands, with the residual
refreshed at the 25 ms controller cadence. Ground truth is used only for the
initial state and scoring. This is a research evaluation, not runtime odometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numba import njit

from tools.vehicle_dynamics_learning.fit_unity_planar_yaw_residual import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    OUTPUT as DEFAULT_FIT_REPORT,
    SAMPLE_STRIDE,
    _feature_vector,
    _parameters,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DT_S,
    HORIZONS_STEPS,
    WHEELS,
    WHEEL_SPRUNG_MASSES_KG,
    _apply_brake_impulse,
    _derivative,
    _initial_acceleration,
    _load_capture,
    _pose_truth,
    _truth_state,
    _wrap,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = DEFAULT_FIT_REPORT.with_suffix(".joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "planar_yaw_residual_recursive_20261013.json")
UPDATE_STEPS = SAMPLE_STRIDE
WINDOW_STRIDE_STEPS = 5000


@njit(cache=False)
def _advance_with_yaw_residual(state, steering_command, throttle,
                               yaw_accel_residual, slip_floor,
                               longitudinal_scale, longitudinal_tangent,
                               lateral_scale, lateral_tangent, wheel_inertia,
                               longitudinal_curve, lateral_curve,
                               static_loads, algebraic_acceleration_state):
    next_steering = state[7]
    target = max(-STEERING_LIMIT_RAD,
                 min(STEERING_LIMIT_RAD,
                     steering_command * STEERING_LIMIT_RAD))
    step = STEERING_RATE_RADPS * DT_S
    next_steering += max(-step, min(step, target - next_steering))
    d0 = _derivative(state, state[7], throttle, slip_floor,
                     longitudinal_scale, longitudinal_tangent,
                     lateral_scale, lateral_tangent, wheel_inertia,
                     longitudinal_curve, lateral_curve, static_loads, True)
    midpoint = state + 0.5 * DT_S * d0
    if algebraic_acceleration_state:
        midpoint[8] = d0[8]
        midpoint[9] = d0[9]
    midpoint[7] = 0.5 * (state[7] + next_steering)
    _apply_brake_impulse(midpoint, throttle, 0.5 * DT_S, wheel_inertia)
    dmid = _derivative(midpoint, midpoint[7], throttle, slip_floor,
                       longitudinal_scale, longitudinal_tangent,
                       lateral_scale, lateral_tangent, wheel_inertia,
                       longitudinal_curve, lateral_curve, static_loads, True)
    dmid[2] += yaw_accel_residual
    result = state + DT_S * dmid
    if algebraic_acceleration_state:
        # These slots represent acceleration used for the next load estimate,
        # not acceleration derivatives. Carry the newly computed values rather
        # than integrating m/s^2 as if it were jerk.
        result[8] = dmid[8]
        result[9] = dmid[9]
    _apply_brake_impulse(result, throttle, DT_S, wheel_inertia)
    result[7] = next_steering
    return result


@njit(cache=False)
def _advance_segment(state, pose, steering_commands, throttle_commands,
                     yaw_accel_residual, slip_floor, longitudinal_scale,
                     longitudinal_tangent, lateral_scale, lateral_tangent,
                     wheel_inertia, longitudinal_curve, lateral_curve,
                     static_loads, algebraic_acceleration_state):
    states = np.empty((len(steering_commands), 10), dtype=np.float64)
    poses = np.empty((len(steering_commands), 3), dtype=np.float64)
    current_state = state.copy()
    current_pose = pose.copy()
    for i in range(len(steering_commands)):
        following = _advance_with_yaw_residual(
            current_state, steering_commands[i], throttle_commands[i],
            yaw_accel_residual, slip_floor, longitudinal_scale,
            longitudinal_tangent, lateral_scale, lateral_tangent,
            wheel_inertia, longitudinal_curve, lateral_curve, static_loads,
            algebraic_acceleration_state)
        u = 0.5 * (current_state[0] + following[0])
        v = 0.5 * (current_state[1] + following[1])
        yaw_rate = 0.5 * (current_state[2] + following[2])
        heading_mid = current_pose[2] + 0.5 * yaw_rate * DT_S
        current_pose[0] += (u * math.sin(heading_mid)
                            + v * math.cos(heading_mid)) * DT_S
        current_pose[1] += (u * math.cos(heading_mid)
                            - v * math.sin(heading_mid)) * DT_S
        current_pose[2] += yaw_rate * DT_S
        current_state = following
        states[i] = current_state
        poses[i] = current_pose
    return current_state, current_pose, states, poses


def _summary(errors: list[dict[str, Any]]) -> dict[str, Any]:
    if not errors:
        return {"windows": 0}
    result: dict[str, Any] = {"windows": len(errors)}
    for key in ("u_mps", "v_mps", "yaw_rate_radps", "position_m",
                "heading_rad", "steering_rad"):
        values = np.asarray([row[key] for row in errors], dtype=np.float64)
        result[key] = {
            "rmse": float(np.sqrt(np.mean(values**2))),
            "median_abs": float(np.median(np.abs(values))),
            "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        }
    wheel = np.asarray([row["wheel_omega_radps"] for row in errors],
                       dtype=np.float64)
    result["wheel_omega_radps"] = {
        "per_wheel_rmse": np.sqrt(np.mean(wheel**2, axis=0)).tolist(),
        "pooled_rmse": float(np.sqrt(np.mean(wheel**2))),
        "p95_abs": float(np.quantile(np.abs(wheel), 0.95)),
    }
    return result


def _paired_rmse_change(baseline: list[dict[str, Any]],
                        candidate: list[dict[str, Any]],
                        field: str) -> dict[str, Any]:
    if len(baseline) != len(candidate) or not baseline:
        return {"windows": 0}
    base = np.asarray([row[field] for row in baseline], dtype=np.float64)
    changed = np.asarray([row[field] for row in candidate], dtype=np.float64)
    if base.ndim > 1:
        base = np.mean(base**2, axis=tuple(range(1, base.ndim)))
        changed = np.mean(changed**2, axis=tuple(range(1, changed.ndim)))
    else:
        base = base**2
        changed = changed**2
    delta = np.sqrt(changed) - np.sqrt(base)
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(delta), size=(5000, len(delta)))
    sampled_delta = np.sqrt(changed[draws].mean(axis=1)) \
        - np.sqrt(base[draws].mean(axis=1))
    return {
        "windows": int(len(delta)),
        "candidate_better_windows": int(np.count_nonzero(delta < 0.0)),
        "mean_paired_window_rmse_delta": float(np.mean(delta)),
        "bootstrap_95pct_paired_rmse_delta": [
            float(x) for x in np.quantile(sampled_delta, (0.025, 0.975))],
    }


def _run_capture(path: Path, model: Any, params: dict[str, float],
                 curves: tuple[np.ndarray, np.ndarray], window_seconds: float,
                 sample_limit: int) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    dt = float(metadata["fixed_delta_time_s"])
    if not math.isclose(dt, DT_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"expected 1 kHz source capture, got dt={dt}")
    ax, ay = _initial_acceleration(data, dt)
    steps = round(window_seconds / dt)
    static_loads = np.asarray(WHEEL_SPRUNG_MASSES_KG) * 9.80665
    starts = list(range(100, len(data["speed_mps"]) - steps - 1,
                        WINDOW_STRIDE_STEPS))
    starts = [start for start in starts if data["speed_mps"][start] >= 2.0]
    starts = [start for start in starts if all(
        data[f"{wheel}_grounded"][start:start + steps].min() > 0.5
        for wheel in WHEELS)]
    if sample_limit > 0:
        starts = starts[:sample_limit]

    modes = tuple(
        f"{accel_mode}_{yaw_mode}"
        for accel_mode in ("integrated_accel_state", "algebraic_accel_state")
        for yaw_mode in ("baseline", "yaw_residual"))
    horizon_errors: dict[str, dict[str, list[dict[str, Any]]]] = {
        f"{step * dt:g}s": {mode: [] for mode in modes}
        for step in HORIZONS_STEPS if step <= steps
    }
    completed_windows = {mode: 0 for mode in modes}
    nonfinite_windows = {mode: 0 for mode in modes}
    long_curve, lateral_curve = curves

    for start in starts:
        initial = _truth_state(data, start, ax, ay)
        initial_pose = _pose_truth(data, start)
        for mode in modes:
            algebraic_acceleration_state = mode.startswith("algebraic_")
            use_residual = mode.endswith("yaw_residual")
            state = initial.copy()
            pose = initial_pose.copy()
            completed = 0
            while completed < steps:
                i = start + completed
                segment_length = min(UPDATE_STEPS, steps - completed)
                current_steer_command = float(data["steering_command"][i])
                current_throttle = float(data["throttle_command"][i])
                d = _derivative(
                    state, state[7], current_throttle, 0.5,
                    params["longitudinal_force_scale"],
                    params["longitudinal_zero_slip_tangent"],
                    params["lateral_force_scale"],
                    params["lateral_zero_slip_tangent"],
                    params["wheel_inertia_kgm2"], long_curve, lateral_curve,
                    static_loads, True)
                previous = max(0, i - UPDATE_STEPS)
                feature = _feature_vector(
                    state, current_throttle, current_steer_command,
                    (current_throttle - data["throttle_command"][previous])
                    / (UPDATE_STEPS * dt),
                    (current_steer_command - data["steering_command"][previous])
                    / (UPDATE_STEPS * dt), d)
                correction = (float(model.predict(feature.reshape(1, -1))[0])
                              if use_residual else 0.0)
                end = i + segment_length
                state, pose, states, poses = _advance_segment(
                    state, pose, data["steering_command"][i:end],
                    data["throttle_command"][i:end], correction, 0.5,
                    params["longitudinal_force_scale"],
                    params["longitudinal_zero_slip_tangent"],
                    params["lateral_force_scale"],
                    params["lateral_zero_slip_tangent"],
                    params["wheel_inertia_kgm2"], long_curve, lateral_curve,
                    static_loads, algebraic_acceleration_state)
                for local in range(segment_length):
                    elapsed = completed + local + 1
                    if elapsed not in HORIZONS_STEPS or elapsed > steps:
                        continue
                    truth_index = start + elapsed
                    truth_pose = _pose_truth(data, truth_index)
                    pose_error = poses[local] - truth_pose
                    pose_error[2] = _wrap(pose_error[2])
                    truth_state = np.asarray((
                        data["vel_body_z_mps"][truth_index],
                        -data["vel_body_x_mps"][truth_index],
                        -data["omega_body_y_rps"][truth_index],
                        *[data[f"{wheel}_rpm"][truth_index] * 2.0 * math.pi / 60.0
                          for wheel in WHEELS]), dtype=np.float64)
                    state_error = states[local][:7] - truth_state
                    truth_steer = -math.radians(0.5 * (
                        data["fl_steer_deg"][truth_index]
                        + data["fr_steer_deg"][truth_index]))
                    bucket = horizon_errors[f"{elapsed * dt:g}s"][mode]
                    bucket.append({
                        "start_index": start,
                        "start_speed_mps": float(data["speed_mps"][start]),
                        "max_steering_rad": float(np.nanmax(np.abs(
                            -np.deg2rad(0.5 * (
                                data["fl_steer_deg"][start:start + steps]
                                + data["fr_steer_deg"][start:start + steps]))))),
                        "u_mps": float(state_error[0]),
                        "v_mps": float(state_error[1]),
                        "yaw_rate_radps": float(state_error[2]),
                        "wheel_omega_radps": state_error[3:7].tolist(),
                        "position_m": float(np.linalg.norm(pose_error[:2])),
                        "heading_rad": float(pose_error[2]),
                        "steering_rad": float(states[local][7] - truth_steer),
                    })
                    if not (np.isfinite(states[local]).all()
                            and np.isfinite(poses[local]).all()):
                        nonfinite_windows[mode] += 1
                        completed = steps
                        break
                completed += segment_length
                if not np.isfinite(state).all() or not np.isfinite(pose).all():
                    nonfinite_windows[mode] += 1
                    break
            if completed >= steps:
                completed_windows[mode] += 1

    return {
        "capture": str(path.relative_to(ROOT)),
        "window_seconds": window_seconds,
        "window_stride_seconds": WINDOW_STRIDE_STEPS * dt,
        "windows_started": len(starts),
        "windows_with_scores": completed_windows,
        "nonfinite_windows": nonfinite_windows,
        "horizons": {
            horizon: {
                **{mode: _summary(scores[mode]) for mode in modes},
                "algebraic_residual_vs_baseline": {
                    field: _paired_rmse_change(
                        scores["algebraic_accel_state_baseline"],
                        scores["algebraic_accel_state_yaw_residual"], field)
                    for field in ("u_mps", "v_mps", "yaw_rate_radps",
                                  "wheel_omega_radps", "position_m",
                                  "heading_rad")
                },
                "largest_candidate_errors": {
                    field: [
                        {key: row[key] for key in (
                            "start_index", "start_speed_mps",
                            "max_steering_rad", field)}
                        for row in sorted(
                            scores["algebraic_accel_state_yaw_residual"],
                            key=lambda item: abs(item[field]), reverse=True)[:5]
                    ]
                    for field in ("u_mps", "v_mps", "yaw_rate_radps",
                                  "position_m", "heading_rad")
                },
            }
            for horizon, scores in horizon_errors.items()
        },
        "initialization": "one simulator-truth state including causal acceleration history at each window start",
        "post_initialization_inputs": "recorded commands only; candidate residual refreshed every 25 ms from predicted state; no future simulator truth",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--fit-report", type=Path, default=DEFAULT_FIT_REPORT)
    parser.add_argument("--capture", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    bundle = joblib.load(args.bundle)
    params, curves = _parameters()
    report = {
        "schema_version": 1,
        "status": "recursive research evaluation; not production odometry",
        "fit_report": str(args.fit_report.resolve().relative_to(ROOT)),
        "candidate_bundle": str(args.bundle.resolve().relative_to(ROOT)),
        "candidate_bundle_sha256": hashlib.sha256(
            args.bundle.read_bytes()).hexdigest(),
        "training_capture": str(DEFAULT_TRAIN.relative_to(ROOT)),
        "heldout_capture": str(args.capture.resolve().relative_to(ROOT)),
        "fitting_split": "r07 training only; complete r11 capture held out",
        "simulation_dt_s": DT_S,
        "controller_update_period_s": UPDATE_STEPS * DT_S,
        "physics": "source-based planar 4-wheel model; 1 ms midpoint integration; unilateral zero-speed brake constraint; r07 force parameters frozen",
        "future_ground_truth_used": False,
        "heldout": _run_capture(args.capture.resolve(), bundle["model"],
                                params, curves, args.window_seconds,
                                args.sample_limit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    key_horizons = ("0.025s", "0.1s", "0.25s", "0.5s", "1s", "2s", "5s")
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "windows": report["heldout"]["windows_with_scores"],
        "nonfinite": report["heldout"]["nonfinite_windows"],
        "horizons": {
            h: {
                mode: {field: modes[mode][field]["rmse"]
                       if field != "wheel_omega_radps"
                       else modes[mode][field]["pooled_rmse"]
                       for field in ("u_mps", "v_mps", "yaw_rate_radps",
                                     "position_m", "heading_rad",
                                     "wheel_omega_radps")}
                for mode in ("algebraic_accel_state_baseline",
                             "algebraic_accel_state_yaw_residual")
            } | {"paired": modes["algebraic_residual_vs_baseline"]}
            for h, modes in report["heldout"]["horizons"].items()
            if h in key_horizons
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
