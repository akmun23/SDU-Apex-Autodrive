#!/usr/bin/env python3
"""Fit command/state-conditioned 25 ms motion residuals and score 5 s rollouts.

The fitted outputs are body-lateral acceleration, body-forward acceleration,
and yaw angular acceleration residuals. They are applied at each 1 ms plant
step for one 25 ms interval. Fit labels use r07 truth; r11 is held out and is
used only for scoring. No future truth or future sensor is a model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_unity_3d_yaw_residual import (
    CORRECTION_PERIOD,
    FEATURE_NAMES,
    _evaluation_starts,
    _features,
    _path_label,
    _reset_transition_mask,
)
from tools.vehicle_dynamics_learning.replay_unity_source_3d_plant import (
    DEFAULT_SLIP_FIT,
    DEFAULT_SUSPENSION_FIT,
    DEFAULT_TIRE_FIT,
    DEFAULT_WHEEL_FIT,
    HORIZON_MILLISECONDS,
    ROOT,
    _advance,
    _initial_state,
    _metadata,
    _prepare_constants,
    _state_error,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    DT_S,
    WHEELS,
    _load_capture,
)


DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_3d_coupled_state_residual_20261013.json")
TARGETS = ("body_lateral_velocity", "body_forward_velocity", "yaw_rate")
STATE_SLOTS = (7, 9, 11)
TARGET_UNITS = ("m/s", "m/s", "rad/s")


def _training_rows(path: Path, constants: Any
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data, _ = _load_capture(path)
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    excluded_airborne = 0
    excluded_command_change = 0
    for start in range(100, len(data["speed_mps"]) - CORRECTION_PERIOD - 1,
                       CORRECTION_PERIOD):
        stop = start + CORRECTION_PERIOD
        if data["speed_mps"][start] < 1.0:
            continue
        if any(np.min(data[f"{wheel}_grounded"][start:stop + 1]) < 0.5
               for wheel in WHEELS):
            excluded_airborne += 1
            continue
        steering = data["steering_command"][start:stop]
        throttle = data["throttle_command"][start:stop]
        if not (np.all(steering == steering[0])
                and np.all(throttle == throttle[0])):
            excluded_command_change += 1
            continue
        state = _initial_state(data, start)
        feature = _features(
            state, float(steering[0]), float(throttle[0]),
            float(data["steering_command"][start - 1]),
            float(data["throttle_command"][start - 1]), constants)
        for index in range(start, stop):
            state = _advance(state, data["steering_command"][index],
                             data["throttle_command"][index], constants, DT_S)
        truth = _initial_state(data, stop)
        inputs.append(feature)
        targets.append(np.asarray([
            truth[slot] - state[slot] for slot in STATE_SLOTS],
            dtype=np.float64))
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if len(y) < 500:
        raise ValueError(f"too few supported training transitions: {len(y)}")
    diagnostics = {
        "rows": len(y),
        "excluded_airborne_windows": excluded_airborne,
        "excluded_nonconstant_command_windows": excluded_command_change,
        "target_names": list(TARGETS),
        "target_units": list(TARGET_UNITS),
        "target_rmse": np.sqrt(np.mean(y * y, axis=0)).tolist(),
        "target_p95_abs": np.quantile(np.abs(y), 0.95, axis=0).tolist(),
    }
    return x, y, diagnostics


def _fit_models(x: np.ndarray, y: np.ndarray
                ) -> tuple[dict[str, HistGradientBoostingRegressor], dict[str, Any]]:
    models = {}
    diagnostics = {}
    for output, (name, unit) in enumerate(zip(TARGETS, TARGET_UNITS)):
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=10.0,
            random_state=31 + output,
        )
        model.fit(x, y[:, output])
        fit = model.predict(x)
        models[name] = model
        diagnostics[name] = {
            "unit": unit,
            "in_sample_rmse": float(np.sqrt(np.mean(
                (fit - y[:, output]) ** 2))),
            "target_rmse": float(np.sqrt(np.mean(y[:, output] ** 2))),
            "parameters": model.get_params(),
        }
    return models, diagnostics


def _score(path: Path, constants: Any,
           models: dict[str, HistGradientBoostingRegressor],
           horizon_s: float, stride_s: float) -> dict[str, Any]:
    data, _ = _load_capture(path)
    horizon_steps = round(horizon_s / DT_S)
    stride_steps = round(stride_s / DT_S)
    starts = _evaluation_starts(data, horizon_steps, stride_steps)
    if not starts:
        raise ValueError(f"no fully grounded windows in {path}")
    checkpoints = {round(ms / 1000.0 / DT_S): f"{ms / 1000.0:.3f}s"
                   for ms in HORIZON_MILLISECONDS
                   if ms / 1000.0 <= horizon_s}
    components = {
        "baseline": (),
        "yaw_only": ("yaw_rate",),
        "translation_only": ("body_lateral_velocity",
                             "body_forward_velocity"),
        "coupled": TARGETS,
    }
    errors = {mode: {h: [] for h in checkpoints.values()}
              for mode in components}
    corrections = {mode: {name: [] for name in names}
                   for mode, names in components.items()}
    window_rows = []
    for start in starts:
        states = {mode: _initial_state(data, start) for mode in components}
        initial_abs_steering = abs(float(states["baseline"][17]))
        window_error = {mode: {} for mode in components}
        correction_blocks = {mode: {name: [] for name in names}
                             for mode, names in components.items()}
        residual_acceleration = {
            mode: {"body_lateral_velocity": 0.0,
                   "body_forward_velocity": 0.0,
                   "yaw_rate": 0.0}
            for mode in components}
        for offset in range(horizon_steps):
            index = start + offset
            steering = float(data["steering_command"][index])
            throttle = float(data["throttle_command"][index])
            if offset % CORRECTION_PERIOD == 0:
                previous_steering = float(
                    data["steering_command"][max(0, index - 1)])
                previous_throttle = float(
                    data["throttle_command"][max(0, index - 1)])
                for mode, names in components.items():
                    if not names:
                        continue
                    feature = _features(
                        states[mode], steering, throttle,
                        previous_steering, previous_throttle, constants)
                    for name in names:
                        residual = float(models[name].predict(
                            feature.reshape(1, -1))[0])
                        residual_acceleration[mode][name] = residual / (
                            CORRECTION_PERIOD * DT_S)
                        correction_blocks[mode][name].append(residual)
                        corrections[mode][name].append(residual)
            for mode, state in states.items():
                state[:] = _advance(
                    state, steering, throttle, constants, DT_S,
                    residual_acceleration[mode]["yaw_rate"],
                    residual_acceleration[mode]["body_lateral_velocity"],
                    residual_acceleration[mode]["body_forward_velocity"])
            elapsed = offset + 1
            if elapsed in checkpoints:
                truth = _initial_state(data, index + 1)
                for mode, state in states.items():
                    error = _state_error(state, truth)
                    errors[mode][checkpoints[elapsed]].append(error)
                    window_error[mode][checkpoints[elapsed]] = error
        window_rows.append({
            "start_sample": start,
            "initial_speed_mps": float(data["speed_mps"][start]),
            "initial_abs_steering_rad": initial_abs_steering,
            "errors": window_error,
        })

    summaries = {}
    for mode, horizons in errors.items():
        summaries[mode] = {}
        for horizon, rows in horizons.items():
            position = np.asarray([row["position_horizontal_m"] for row in rows])
            heading = np.asarray([row["heading_rad"] for row in rows])
            velocity = np.asarray([row["body_velocity_xyz_mps"] for row in rows])
            omega = np.asarray([row["body_omega_xyz_radps"] for row in rows])
            wheel = np.asarray([row["wheel_omega_radps"] for row in rows])
            summaries[mode][horizon] = {
                "samples": len(rows),
                "position_horizontal_rmse_m": float(np.sqrt(np.mean(position**2))),
                "position_horizontal_p95_m": float(np.quantile(position, 0.95)),
                "heading_rmse_rad": float(np.sqrt(np.mean(heading**2))),
                "heading_bias_rad": float(np.mean(heading)),
                "body_velocity_xyz_rmse_mps": np.sqrt(
                    np.mean(velocity**2, axis=0)).tolist(),
                "body_omega_xyz_rmse_radps": np.sqrt(
                    np.mean(omega**2, axis=0)).tolist(),
                "wheel_omega_each_rmse_radps": np.sqrt(
                    np.mean(wheel**2, axis=0)).tolist(),
            }
    return {
        "capture": _path_label(path),
        "window_count": len(starts),
        "reset_discontinuity_count": int(np.count_nonzero(
            _reset_transition_mask(data))),
        "reset_window_filter": "reject any rollout window crossing an adjacent 1 ms world-position jump > 0.05 m",
        "horizon_s": horizon_steps * DT_S,
        "window_stride_s": stride_steps * DT_S,
        "errors": summaries,
        "residual_increment_abs_p50_p95_max": {
            mode: {name: [float(x) for x in np.quantile(
                np.abs(values), (0.50, 0.95, 1.0))]
                for name, values in names.items() if values}
            for mode, names in corrections.items()},
        "windows": window_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--slip-fit", type=Path, default=DEFAULT_SLIP_FIT)
    parser.add_argument("--suspension-fit", type=Path,
                        default=DEFAULT_SUSPENSION_FIT)
    parser.add_argument("--wheel-fit", type=Path, default=DEFAULT_WHEEL_FIT)
    parser.add_argument("--horizon-seconds", type=float, default=5.0)
    parser.add_argument("--window-stride-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 0.025 <= args.horizon_seconds <= 5.0:
        parser.error("horizon must be between 0.025 and 5 seconds")
    _thread_limiter = threadpool_limits(limits=1)
    constants, base_description = _prepare_constants(
        _metadata(args.train), args.tire_fit, args.slip_fit,
        args.suspension_fit, args.wheel_fit)
    x_train, y_train, training = _training_rows(args.train, constants)
    models, fit_diagnostics = _fit_models(x_train, y_train)
    report = {
        "schema_version": 1,
        "status": "research-only coupled continuous dynamics residual; not runtime odometry",
        "training_capture": _path_label(args.train),
        "heldout_capture": _path_label(args.heldout),
        "baseline_model": base_description,
        "state_residuals": {
            "targets": list(TARGETS),
            "units": list(TARGET_UNITS),
            "application": "each predicted 25 ms state-increment residual is converted to constant acceleration and applied at each 1 ms source-plant step",
            "models": fit_diagnostics,
        },
        "training_diagnostics": training,
        "feature_names": list(FEATURE_NAMES),
        "reference_yaw_only_result": str((ROOT / (
            "live_runs/unity_exact_binary_audit_20261012/"
            "source_mechanics_3d_yaw_residual_continuous_dagger_20261013.json"
        )).relative_to(ROOT)),
        "recursive_heldout": _score(
            args.heldout, constants, models, args.horizon_seconds,
            args.window_stride_seconds),
        "limitations": [
            "one independently captured trajectory is held out; this does not estimate between-run uncertainty",
            "learned force/torque residuals approximate native WheelCollider behavior and are not a Unity solver reconstruction",
            "wheel angular speeds remain propagated by the explicit source torque balance, not directly corrected",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    print("training rows", training["rows"], "targets", training["target_rmse"])
    for horizon in ("0.025s", "0.100s", "0.500s", "1.000s", "5.000s"):
        scores = report["recursive_heldout"]["errors"]
        if horizon not in scores["baseline"]:
            continue
        print(horizon)
        for mode in ("baseline", "yaw_only", "translation_only", "coupled"):
            row = scores[mode][horizon]
            print(" ", mode,
                  "pos", round(row["position_horizontal_rmse_m"], 4),
                  "heading", round(row["heading_rmse_rad"], 4),
                  "v", [round(v, 4) for v in row[
                      "body_velocity_xyz_rmse_mps"]],
                  "omega", [round(v, 4) for v in row[
                      "body_omega_xyz_rmse_radps"]],
                  "wheel", [round(v, 4) for v in row[
                      "wheel_omega_each_rmse_radps"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
