#!/usr/bin/env python3
"""Fit a causal yaw-rate correction at Unity's 1 ms physics step.

The source-mechanics plant and all other state equations remain unchanged.
This learns only the one-step yaw-rate increment residual from r07 and tests
recursive command-only motion on the held-out r11 capture. Ground truth is
used for training labels and post-rollout scores, never as a rollout input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_unity_3d_yaw_residual import (
    DEFAULT_SLIP_FIT,
    DEFAULT_SUSPENSION_FIT,
    DEFAULT_TIRE_FIT,
    DEFAULT_WHEEL_FIT,
    FEATURE_NAMES,
    ROOT,
    _evaluation_starts,
    _features,
)
from tools.vehicle_dynamics_learning.replay_unity_source_3d_plant import (
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
    _load_capture,
)


HORIZON_MS = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000)
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_3d_yaw_residual_1ms_20261013.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_rows(path: Path, constants: Any
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data, _ = _load_capture(path)
    count = len(data["speed_mps"])
    grounded = np.ones(count, dtype=bool)
    for wheel in ("fl", "fr", "rl", "rr"):
        grounded &= data[f"{wheel}_grounded"] > 0.5
    eligible = np.flatnonzero(
        (np.arange(count) >= 100)
        & (np.arange(count) < count - 1)
        & (data["speed_mps"] >= 1.0)
        & grounded
    )
    x = np.empty((len(eligible), len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(len(eligible), dtype=np.float32)
    for row, index_raw in enumerate(eligible):
        index = int(index_raw)
        state = _initial_state(data, index)
        steering = float(data["steering_command"][index])
        throttle = float(data["throttle_command"][index])
        previous_steering = float(data["steering_command"][index - 1])
        previous_throttle = float(data["throttle_command"][index - 1])
        x[row] = _features(
            state, steering, throttle, previous_steering,
            previous_throttle, constants)
        predicted = _advance(state, steering, throttle, constants, DT_S)
        following = _initial_state(data, index + 1)
        y[row] = following[11] - predicted[11]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("1 ms training rows contain non-finite values")
    return x, y, {
        "rows": int(len(y)),
        "excluded_low_speed_or_airborne_rows": int(count - 101 - len(y)),
        "target_mean_radps_per_step": float(np.mean(y)),
        "target_rmse_radps_per_step": float(np.sqrt(np.mean(y * y))),
        "target_abs_p95_radps_per_step": float(np.quantile(np.abs(y), 0.95)),
    }


def _metrics(errors: list[dict[str, Any]]) -> dict[str, Any]:
    position = np.asarray([row["position_horizontal_m"] for row in errors])
    heading = np.asarray([row["heading_rad"] for row in errors])
    velocity = np.asarray([row["body_velocity_xyz_mps"] for row in errors])
    omega = np.asarray([row["body_omega_xyz_radps"] for row in errors])
    wheels = np.asarray([row["wheel_omega_radps"] for row in errors])
    return {
        "windows": int(len(errors)),
        "position_horizontal_rmse_m": float(np.sqrt(np.mean(position**2))),
        "position_horizontal_p95_m": float(np.quantile(position, 0.95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading**2))),
        "heading_bias_rad": float(np.mean(heading)),
        "body_velocity_xyz_rmse_mps": np.sqrt(
            np.mean(velocity**2, axis=0)).tolist(),
        "body_omega_xyz_rmse_radps": np.sqrt(
            np.mean(omega**2, axis=0)).tolist(),
        "wheel_omega_each_rmse_radps": np.sqrt(
            np.mean(wheels**2, axis=0)).tolist(),
    }


def _evaluate(path: Path, constants: Any,
              model: HistGradientBoostingRegressor,
              horizon_s: float = 5.0,
              stride_s: float = 5.0,
              minimum_model_speed_mps: float = 0.0) -> dict[str, Any]:
    data, _ = _load_capture(path)
    horizon_steps = round(horizon_s / DT_S)
    stride_steps = round(stride_s / DT_S)
    starts = _evaluation_starts(data, horizon_steps, stride_steps)
    if not starts:
        raise ValueError(f"no fully grounded rollout windows in {path}")
    checkpoint_steps = {
        int(round(milliseconds / 1000.0 / DT_S)): f"{milliseconds / 1000.0:.3f}s"
        for milliseconds in HORIZON_MS
        if milliseconds / 1000.0 <= horizon_s
    }
    errors = {"source_mechanics": {label: [] for label in checkpoint_steps.values()},
              "source_mechanics_plus_1ms_yaw_residual": {
                  label: [] for label in checkpoint_steps.values()}}
    per_window = []
    for start in starts:
        base = _initial_state(data, start)
        corrected = base.copy()
        window_errors: dict[str, dict[str, Any]] = {
            "source_mechanics": {},
            "source_mechanics_plus_1ms_yaw_residual": {},
        }
        for offset in range(horizon_steps):
            index = start + offset
            steering = float(data["steering_command"][index])
            throttle = float(data["throttle_command"][index])
            next_base = _advance(base, steering, throttle, constants, DT_S)
            current_feature = _features(
                corrected, steering, throttle,
                float(data["steering_command"][max(0, index - 1)]),
                float(data["throttle_command"][max(0, index - 1)]),
                constants,
            )
            residual_delta = 0.0
            if np.linalg.norm(corrected[7:10]) >= minimum_model_speed_mps:
                residual_delta = float(model.predict(
                    current_feature.reshape(1, -1))[0])
            corrected = _advance(
                corrected, steering, throttle, constants, DT_S,
                residual_delta / DT_S)
            base = next_base
            elapsed = offset + 1
            if elapsed not in checkpoint_steps:
                continue
            truth = _initial_state(data, index + 1)
            label = checkpoint_steps[elapsed]
            base_error = _state_error(base, truth)
            corrected_error = _state_error(corrected, truth)
            errors["source_mechanics"][label].append(base_error)
            errors["source_mechanics_plus_1ms_yaw_residual"][label].append(
                corrected_error)
            window_errors["source_mechanics"][label] = base_error
            window_errors["source_mechanics_plus_1ms_yaw_residual"][label] = (
                corrected_error)
        per_window.append({
            "start_sample": int(start),
            "initial_speed_mps": float(data["speed_mps"][start]),
            "errors": window_errors,
        })
    return {
        "capture": str(path.relative_to(ROOT)),
        "window_count": int(len(starts)),
        "window_stride_s": stride_steps * DT_S,
        "horizon_s": horizon_steps * DT_S,
        "minimum_model_speed_mps": minimum_model_speed_mps,
        "errors": {
            model_name: {label: _metrics(rows)
                         for label, rows in horizons.items()}
            for model_name, horizons in errors.items()
        },
        "windows": per_window,
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model_path = args.output.with_suffix(".joblib")
    if args.output.exists() or model_path.exists():
        raise FileExistsError(f"refusing to overwrite {args.output} or {model_path}")

    started = time.time()
    constants, plant_description = _prepare_constants(
        _metadata(args.train), args.tire_fit, args.slip_fit,
        args.suspension_fit, args.wheel_fit)
    x_train, y_train, training_diagnostics = _training_rows(
        args.train, constants)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=160,
        max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=10.0, early_stopping=False,
        random_state=20261013,
    )
    with threadpool_limits(limits=4):
        model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    joblib.dump({
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "target": "per-1ms-step yaw-rate increment residual, rad/s",
        "sample_period_s": DT_S,
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
    }, model_path, compress=3)
    heldout = _evaluate(args.heldout, constants, model)
    report = {
        "schema_version": 1,
        "status": "research-only; 1 ms yaw increment residual; not runtime odometry",
        "training_capture": str(args.train.relative_to(ROOT)),
        "training_capture_sha256": _sha256(args.train),
        "heldout_capture": str(args.heldout.relative_to(ROOT)),
        "heldout_capture_sha256": _sha256(args.heldout),
        "model_artifact": str(model_path.relative_to(ROOT)),
        "model_artifact_sha256": _sha256(model_path),
        "sample_period_s": DT_S,
        "plant_description": plant_description,
        "feature_names": list(FEATURE_NAMES),
        "training_diagnostics": training_diagnostics,
        "in_sample_fit_rmse_radps_per_step": float(np.sqrt(np.mean(
            (train_prediction - y_train) ** 2))),
        "heldout_recursive_command_only": heldout,
        "truth_or_future_sensors_used_during_rollout": False,
        "test_or_final_test_archives_used": False,
        "limitations": [
            "single r07 training capture and one held-out r11 capture",
            "the learned residual changes yaw acceleration only; all other mechanics remain source-based",
            "the held-out captures are source-player runs, not exact pinned-player parity proof",
        ],
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "model": str(model_path),
        "training_rows": training_diagnostics["rows"],
        "training_target_rmse_radps_per_step": training_diagnostics[
            "target_rmse_radps_per_step"],
        "heldout": heldout["errors"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
