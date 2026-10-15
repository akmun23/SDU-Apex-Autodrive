#!/usr/bin/env python3
"""Fit a 25 ms yaw-transition correction to the source-derived plant.

Unlike an instantaneous-acceleration fit, each training label is the constant
yaw-acceleration correction which closes the baseline model's yaw-rate error
over exactly one controller interval. Training uses r07 only; all of r11 is
held out. The frozen artifact is evaluated recursively by
``evaluate_unity_planar_yaw_residual_rollout.py``.
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
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.vehicle_dynamics_learning.evaluate_unity_planar_yaw_residual_rollout import (
    _advance_segment,
)
from tools.vehicle_dynamics_learning.fit_unity_planar_yaw_residual import (
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    FEATURE_NAMES,
    _feature_vector,
    _parameters,
    _steering_trace,
)
from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    DT_S,
    WHEELS,
    WHEEL_SPRUNG_MASSES_KG,
    _derivative,
    _initial_acceleration,
    _load_capture,
)


ROOT = Path(__file__).resolve().parents[2]
INTERVAL_STEPS = 25
OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "planar_yaw_transition_residual_20261013.json")
CURVES = (
    np.asarray((0.15, 0.9, 0.25, 0.58, 0.8), dtype=np.float64),
    np.asarray((0.01, 1.0, 0.10, 0.50, 1.0), dtype=np.float64),
)


def _sample_capture(path: Path, params: dict[str, float]
                    ) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    dt = float(metadata["fixed_delta_time_s"])
    if not math.isclose(dt, DT_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"expected a 1 kHz Unity source capture, got {dt}")
    ax, ay = _initial_acceleration(data, dt)
    state = np.zeros((len(data["speed_mps"]), 10), dtype=np.float64)
    state[:, 0] = data["vel_body_z_mps"]
    state[:, 1] = -data["vel_body_x_mps"]
    state[:, 2] = -data["omega_body_y_rps"]
    for index, wheel in enumerate(WHEELS):
        state[:, 3 + index] = data[f"{wheel}_rpm"] * (2.0 * math.pi / 60.0)
    state[:, 7] = _steering_trace(data)
    state[:, 8] = ax
    state[:, 9] = ay
    static_loads = np.asarray(WHEEL_SPRUNG_MASSES_KG) * 9.80665
    interval_s = INTERVAL_STEPS * dt
    rows: list[np.ndarray] = []
    labels: list[float] = []
    base_errors: list[float] = []
    speed: list[float] = []
    steer: list[float] = []
    source_index: list[int] = []

    # Align labels and inference to a 25 ms controller interval. Leave an
    # initial command-history interval before the first sample.
    first = 2 * INTERVAL_STEPS
    for i in range(first, len(state) - INTERVAL_STEPS, INTERVAL_STEPS):
        if not np.isfinite(state[i]).all():
            continue
        throttle = float(data["throttle_command"][i])
        steering_command = float(data["steering_command"][i])
        derivative = _derivative(
            state[i], state[i, 7], throttle, 0.5,
            params["longitudinal_force_scale"],
            params["longitudinal_zero_slip_tangent"],
            params["lateral_force_scale"],
            params["lateral_zero_slip_tangent"],
            params["wheel_inertia_kgm2"], CURVES[0], CURVES[1],
            static_loads, True)
        previous = i - INTERVAL_STEPS
        feature = _feature_vector(
            state[i], throttle, steering_command,
            (throttle - data["throttle_command"][previous]) / interval_s,
            (steering_command - data["steering_command"][previous])
            / interval_s, derivative)
        _, _, baseline_states, _ = _advance_segment(
            state[i], np.zeros(3, dtype=np.float64),
            data["steering_command"][i:i + INTERVAL_STEPS],
            data["throttle_command"][i:i + INTERVAL_STEPS],
            0.0, 0.5, params["longitudinal_force_scale"],
            params["longitudinal_zero_slip_tangent"],
            params["lateral_force_scale"],
            params["lateral_zero_slip_tangent"],
            params["wheel_inertia_kgm2"], CURVES[0], CURVES[1],
            static_loads, True)
        baseline_next_yaw = float(baseline_states[-1, 2])
        truth_next_yaw = float(state[i + INTERVAL_STEPS, 2])
        rows.append(feature)
        labels.append((truth_next_yaw - baseline_next_yaw) / interval_s)
        base_errors.append(baseline_next_yaw - truth_next_yaw)
        speed.append(float(data["speed_mps"][i]))
        steer.append(abs(float(state[i, 7])))
        source_index.append(i)

    features = np.asarray(rows, dtype=np.float32)
    target = np.asarray(labels, dtype=np.float32)
    finite = np.isfinite(features).all(axis=1) & np.isfinite(target)
    return {
        "capture": str(path.relative_to(ROOT)),
        "features": features[finite],
        "target": target[finite],
        "base_error": np.asarray(base_errors, dtype=np.float64)[finite],
        "speed": np.asarray(speed, dtype=np.float64)[finite],
        "steering": np.asarray(steer, dtype=np.float64)[finite],
        "indices": np.asarray(source_index, dtype=np.int64)[finite],
    }


def _metrics(error: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(error.size),
        "rmse_radps": float(np.sqrt(np.mean(error**2))),
        "bias_radps": float(np.mean(error)),
        "abs_p95_radps": float(np.quantile(np.abs(error), 0.95)),
    }


def _block_summary(base_error: np.ndarray, corrected_error: np.ndarray,
                   indices: np.ndarray) -> dict[str, Any]:
    block_ids = indices // 5000
    pairs = []
    for key in np.unique(block_ids):
        mask = block_ids == key
        if mask.sum() < 20:
            continue
        pairs.append((np.sqrt(np.mean(base_error[mask]**2)),
                      np.sqrt(np.mean(corrected_error[mask]**2))))
    values = np.asarray(pairs)
    delta = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(delta), size=(5000, len(delta)))
    ci = np.quantile(delta[draws].mean(axis=1), (0.025, 0.975))
    return {
        "five_second_blocks": int(len(values)),
        "blocks_improved": int(np.count_nonzero(delta < 0)),
        "mean_paired_rmse_delta_radps": float(np.mean(delta)),
        "bootstrap_95pct_delta": ci.tolist(),
    }


def run(train_path: Path, heldout_path: Path, output_path: Path
        ) -> dict[str, Any]:
    bundle_path = output_path.with_suffix(".joblib")
    if output_path.exists() or bundle_path.exists():
        raise FileExistsError("refusing to overwrite an existing transition fit")
    params, _ = _parameters()
    train = _sample_capture(train_path, params)
    heldout = _sample_capture(heldout_path, params)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=140,
        max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=5.0,
        early_stopping=False, random_state=20261013)
    model.fit(train["features"], train["target"])
    correction = model.predict(heldout["features"])
    corrected_error = heldout["base_error"] + correction * (
        INTERVAL_STEPS * DT_S)
    report = {
        "schema_version": 1,
        "status": "one-interval yaw transition candidate; recursive held-out evaluation required",
        "training_capture": train["capture"],
        "heldout_capture": heldout["capture"],
        "training_rows": int(len(train["target"])),
        "heldout_rows": int(len(heldout["target"])),
        "training_only": True,
        "physics_integration": "source-derived planar model at 1 ms; acceleration-valued load state updated algebraically; unilateral brake stop constraint",
        "update_interval_s": INTERVAL_STEPS * DT_S,
        "label": "constant yaw-acceleration correction needed to close baseline r error at the next 25 ms boundary",
        "heldout_one_interval_yaw_rate_error": {
            "baseline": _metrics(heldout["base_error"]),
            "corrected": _metrics(corrected_error),
            "paired_five_second_blocks": _block_summary(
                heldout["base_error"], corrected_error, heldout["indices"]),
        },
        "features": list(FEATURE_NAMES),
        "feature_policy": "current state, current/past command and explicit source-planar mechanics; no future truth input",
        "bundle": str(bundle_path.relative_to(ROOT)),
    }
    bundle = {
        "research_only": True,
        "model": model,
        "feature_names": list(FEATURE_NAMES),
        "parameters": params,
        "curve_landmarks": [CURVES[0], CURVES[1]],
        "training_capture": train["capture"],
        "heldout_capture": heldout["capture"],
        "update_interval_steps": INTERVAL_STEPS,
        "load_state_mode": "algebraic_acceleration",
        "current_state_only": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path, compress=3)
    report["bundle_sha256"] = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.train.resolve(), args.heldout.resolve(),
                 args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "rows": {"train": report["training_rows"],
                 "heldout": report["heldout_rows"]},
        "one_interval": report["heldout_one_interval_yaw_rate_error"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
