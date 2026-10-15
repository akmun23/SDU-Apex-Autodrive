#!/usr/bin/env python3
"""Fit a train-only yaw-acceleration residual on the explicit source plant.

The source tire, wheel, steering, and planar Newton-Euler equations remain the
baseline. A HistGradientBoostingRegressor learns only the residual between its
current-state yaw acceleration and a causal 31 ms yaw-rate derivative. Inputs
are current predicted state and current/past commands, so the artifact can be
tested recursively without future simulator state.
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
from scipy.signal import savgol_coeffs
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.vehicle_dynamics_learning.replay_unity_source_mechanics import (
    ACKERMANN_WHEELBASE_M,
    AXLE_SPAN_M,
    CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
    DEFAULT_FORCE_FIT,
    DEFAULT_TRAIN,
    DEFAULT_VALIDATION,
    DT_S,
    GRAVITY_MPS2,
    MOTOR_TORQUE_NM,
    RIGIDBODY_MASS_KG,
    RIGIDBODY_YAW_INERTIA_KGM2,
    TRACK_M,
    WHEELS,
    WHEEL_DAMPING_RATE_NMS,
    WHEEL_RADIUS_M,
    WHEEL_SPRUNG_MASSES_KG,
    WHEEL_X_M,
    WHEEL_Y_M,
    _derivative,
    _initial_acceleration,
    _integrate_pose,
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
YAW_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "wheel_spin_state_inputs_v10_20261013.json")
OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "planar_yaw_residual_20261013.json")
WINDOW = 31
SAMPLE_STRIDE = 25
FEATURE_NAMES = (
    "u_mps", "v_mps", "yaw_rate_rps", "steering_feedback_rad",
    "throttle_command", "steering_command", "throttle_slew_per_s",
    "steering_slew_per_s", "ax_body_mps2", "ay_body_mps2",
    "baseline_yaw_acceleration_radps2",
    *(name for wheel in WHEELS for name in (
        f"{wheel}_wheel_omega_radps", f"{wheel}_surface_minus_contact_mps",
        f"{wheel}_longitudinal_slip", f"{wheel}_lateral_slip",
        f"{wheel}_normal_load_n", f"{wheel}_lateral_contact_speed_mps")),
)


def _parameters() -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:
    force = json.loads(DEFAULT_FORCE_FIT.read_text(encoding="utf-8"))
    source = force["curve_model"]["training_fit"]
    wheel_fit = force["wheel_spin_balance"]["training_capture_fit"]
    yaw = json.loads(YAW_FIT.read_text(encoding="utf-8"))[
        "source_force_application_point_lateral_fit"]["fit"]["parameters"]
    params = {
        "longitudinal_force_scale": source["longitudinal_scale"],
        "longitudinal_zero_slip_tangent": source[
            "longitudinal_zero_slip_tangent"],
        "lateral_force_scale": yaw[0],
        "lateral_zero_slip_tangent": yaw[1],
        "wheel_inertia_kgm2": wheel_fit[
            "fitted_shared_wheel_inertia_kgm2"],
        "wheel_damping_nms": WHEEL_DAMPING_RATE_NMS,
    }
    curves = (
        np.asarray((0.15, 0.9, 0.25, 0.58, 0.8), dtype=np.float64),
        np.asarray((0.01, 1.0, 0.10, 0.50, 1.0), dtype=np.float64),
    )
    return params, curves


def _steering_trace(data: dict[str, np.ndarray], initial: int = 0
                    ) -> np.ndarray:
    command = data["steering_command"]
    result = np.empty_like(command)
    actual = -math.radians(0.5 * (
        data["fl_steer_deg"][initial] + data["fr_steer_deg"][initial]))
    result[:initial + 1] = actual
    maximum_step = STEERING_RATE_RADPS * DT_S
    for index in range(initial + 1, len(result)):
        target = float(np.clip(command[index], -1.0, 1.0)) \
            * STEERING_LIMIT_RAD
        actual += np.clip(target - actual, -maximum_step, maximum_step)
        result[index] = actual
    return result


def _slip_load_features(state: np.ndarray,
                        steering: float) -> tuple[np.ndarray, np.ndarray]:
    u, v, yaw_rate = state[:3]
    wheel_omega = state[3:7]
    ax_load, ay_load = state[8:10]
    tangent = math.tan(steering)
    num = 2.0 * ACKERMANN_WHEELBASE_M * tangent
    left = math.atan2(
        num, 2.0 * ACKERMANN_WHEELBASE_M - TRACK_M * tangent)
    right = math.atan2(
        num, 2.0 * ACKERMANN_WHEELBASE_M + TRACK_M * tangent)
    angles = (left, right, 0.0, 0.0)
    wheel_features = np.empty((4, 6), dtype=np.float64)
    static_loads = np.asarray(WHEEL_SPRUNG_MASSES_KG) * GRAVITY_MPS2
    for index in range(4):
        x, y = WHEEL_X_M[index], WHEEL_Y_M[index]
        wheel_u = u - yaw_rate * y
        wheel_v = v + yaw_rate * x
        delta = angles[index]
        c, s = math.cos(delta), math.sin(delta)
        contact_u = c * wheel_u + s * wheel_v
        contact_v = -s * wheel_u + c * wheel_v
        denom = max(abs(contact_u), 0.5)
        slip_x = (WHEEL_RADIUS_M * wheel_omega[index] - contact_u) / denom
        slip_y = contact_v / denom
        load_long = (RIGIDBODY_MASS_KG * ax_load
                     * CAPTURED_COM_HEIGHT_ABOVE_ROAD_M
                     / (2.0 * AXLE_SPAN_M))
        load_lat = (RIGIDBODY_MASS_KG * ay_load
                    * CAPTURED_COM_HEIGHT_ABOVE_ROAD_M
                    / (2.0 * TRACK_M))
        side_sign = -1.0 if index % 2 == 0 else 1.0
        axle_sign = -1.0 if index < 2 else 1.0
        load = max(0.0, static_loads[index] + axle_sign * load_long
                   + side_sign * load_lat)
        wheel_features[index] = (
            wheel_omega[index], WHEEL_RADIUS_M * wheel_omega[index] - contact_u,
            slip_x, slip_y, load, contact_v)
    return wheel_features, np.asarray(angles)


def _feature_vector(state: np.ndarray, throttle: float,
                    steering_command: float, throttle_slew: float,
                    steering_slew: float, derivative: np.ndarray
                    ) -> np.ndarray:
    wheel_features, _ = _slip_load_features(state, float(state[7]))
    base = np.asarray((
        *state[:3], state[7], throttle, steering_command,
        throttle_slew, steering_slew, state[8], state[9], derivative[2],
    ), dtype=np.float32)
    return np.concatenate((base, wheel_features.astype(np.float32).ravel()))


def _sample_data(path: Path, params: dict[str, float],
                 curves: tuple[np.ndarray, np.ndarray]
                 ) -> dict[str, Any]:
    data, metadata = _load_capture(path)
    dt = float(metadata["fixed_delta_time_s"])
    if not math.isclose(dt, DT_S, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"source capture must be sampled at 1 kHz, got {dt}")
    ax, ay = _initial_acceleration(data, dt)
    state_matrix = np.zeros((len(data["speed_mps"]), 10), dtype=np.float64)
    state_matrix[:, 0] = data["vel_body_z_mps"]
    state_matrix[:, 1] = -data["vel_body_x_mps"]
    state_matrix[:, 2] = -data["omega_body_y_rps"]
    for idx, wheel in enumerate(WHEELS):
        state_matrix[:, 3 + idx] = (
            data[f"{wheel}_rpm"] * (2.0 * math.pi / 60.0))
    steering = _steering_trace(data)
    state_matrix[:, 7] = steering
    state_matrix[:, 8] = ax
    state_matrix[:, 9] = ay
    yaw_rate = state_matrix[:, 2]
    coeff = savgol_coeffs(WINDOW, 3, deriv=1, delta=dt,
                          pos=WINDOW - 1, use="dot")
    target = np.full(len(yaw_rate), np.nan, dtype=np.float64)
    for i in range(WINDOW - 1, len(yaw_rate)):
        target[i] = np.dot(yaw_rate[i - WINDOW + 1:i + 1], coeff)

    feature_rows = []
    target_rows = []
    speed_rows = []
    steer_rows = []
    source_indices = []
    indices = np.arange(WINDOW - 1, len(yaw_rate) - 1, SAMPLE_STRIDE)
    for i in indices:
        d = _derivative(
            state_matrix[i], steering[i], data["throttle_command"][i], 0.5,
            params["longitudinal_force_scale"],
            params["longitudinal_zero_slip_tangent"],
            params["lateral_force_scale"],
            params["lateral_zero_slip_tangent"],
            params["wheel_inertia_kgm2"], curves[0], curves[1],
            np.asarray(WHEEL_SPRUNG_MASSES_KG) * GRAVITY_MPS2, True)
        predicted_alpha = float(d[2])
        previous = max(0, i - 25)
        throttle_slew = ((data["throttle_command"][i]
                          - data["throttle_command"][previous]) / 0.025)
        steering_slew = ((data["steering_command"][i]
                          - data["steering_command"][previous]) / 0.025)
        feature_rows.append(_feature_vector(
            state_matrix[i], data["throttle_command"][i],
            data["steering_command"][i], throttle_slew, steering_slew, d))
        target_rows.append(target[i] - predicted_alpha)
        speed_rows.append(data["speed_mps"][i])
        steer_rows.append(abs(steering[i]))
        source_indices.append(i)
    features = np.asarray(feature_rows, dtype=np.float32)
    target_residual = np.asarray(target_rows, dtype=np.float32)
    finite = np.isfinite(features).all(axis=1) & np.isfinite(target_residual)
    return {
        "capture": str(path.relative_to(ROOT)),
        "features": features[finite],
        "target_residual": target_residual[finite],
        "baseline_alpha": np.asarray([
            feature_rows[k][10] for k in range(len(feature_rows))],
            dtype=np.float64)[finite],
        "target_alpha": np.asarray(target_rows, dtype=np.float64)[finite]
            + np.asarray([
                feature_rows[k][10] for k in range(len(feature_rows))],
                dtype=np.float64)[finite],
        "speed": np.asarray(speed_rows, dtype=np.float64)[finite],
        "steering": np.asarray(steer_rows, dtype=np.float64)[finite],
        "source_indices": np.asarray(source_indices, dtype=np.int64)[finite],
        "sample_count": int(np.count_nonzero(finite)),
    }


def _metrics(y_true: np.ndarray, prediction: np.ndarray,
             mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)
    error = prediction[mask] - y_true[mask]
    if error.size == 0:
        return {"samples": 0, "rmse_radps2": None, "bias_radps2": None}
    return {"samples": int(error.size),
            "rmse_radps2": float(np.sqrt(np.mean(error**2))),
            "bias_radps2": float(np.mean(error)),
            "abs_p95_radps2": float(np.quantile(np.abs(error), 0.95))}


def _block_bootstrap(y: np.ndarray, base: np.ndarray, corrected: np.ndarray,
                     source_indices: np.ndarray) -> dict[str, Any]:
    block = source_indices // 5000
    pairs = []
    for key in np.unique(block):
        mask = block == key
        if mask.sum() < 20:
            continue
        pairs.append((np.sqrt(np.mean((base[mask] - y[mask])**2)),
                      np.sqrt(np.mean((corrected[mask] - y[mask])**2))))
    values = np.asarray(pairs)
    delta = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(delta), size=(5000, len(delta)))
    ci = np.quantile(delta[draws].mean(axis=1), (0.025, 0.975))
    return {"blocks_5s": int(len(delta)),
            "blocks_improved": int(np.sum(delta < 0.0)),
            "mean_paired_block_rmse_delta_radps2": float(np.mean(delta)),
            "bootstrap_95pct_delta": ci.tolist()}


def run(train_path: Path, heldout_path: Path, output_path: Path
        ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    bundle_path = output_path.with_suffix(".joblib")
    if bundle_path.exists():
        raise FileExistsError(f"refusing to overwrite {bundle_path}")
    params, curves = _parameters()
    train = _sample_data(train_path, params, curves)
    heldout = _sample_data(heldout_path, params, curves)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=140,
        max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=5.0,
        early_stopping=False, random_state=20261013)
    model.fit(train["features"], train["target_residual"])
    residual = model.predict(heldout["features"])
    baseline = heldout["baseline_alpha"]
    corrected = baseline + residual
    y = heldout["target_alpha"]
    bins = {}
    for speed_lo, speed_hi in ((0, 3), (3, 6), (6, 9), (9, 12.1)):
        for steer_lo, steer_hi in ((0, 0.15), (0.15, 0.35), (0.35, 0.6)):
            mask = ((heldout["speed"] >= speed_lo)
                    & (heldout["speed"] < speed_hi)
                    & (heldout["steering"] >= steer_lo)
                    & (heldout["steering"] < steer_hi))
            if mask.sum() == 0:
                continue
            key = (f"speed_{speed_lo:g}_{speed_hi:g}_steer_"
                   f"{steer_lo:g}_{steer_hi:g}")
            bins[key] = {"baseline": _metrics(y, baseline, mask),
                         "corrected": _metrics(y, corrected, mask)}
    report = {
        "schema_version": 1,
        "status": "one-step planar yaw residual candidate; recursive evaluation required before promotion",
        "training_capture": train["capture"],
        "heldout_capture": heldout["capture"],
        "fixed_delta_time_s": DT_S,
        "label": "causal 31 ms trailing derivative of simulator yaw rate",
        "fit_data": {"training_rows": train["sample_count"],
                     "heldout_rows": heldout["sample_count"],
                     "sample_period_ms": SAMPLE_STRIDE * DT_S * 1000,
                     "training_only": True},
        "features": list(FEATURE_NAMES),
        "feature_policy": "current body/wheel/steering state and current/past command; no future truth; steering state initialized once and advanced from source rate-limited actuator code",
        "heldout_one_step": {
            "baseline": _metrics(y, baseline),
            "corrected": _metrics(y, corrected),
            "paired_five_second_blocks": _block_bootstrap(
                y, baseline, corrected, heldout["source_indices"]),
            "by_speed_and_steering": bins,
        },
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "next_check": "load the frozen model in the 1 kHz command-only source-plant replay, compute the residual from predicted state at each 25 ms boundary, and score the same full r11 windows without future-state feedback",
    }
    bundle = {"research_only": True, "model": model,
              "feature_names": list(FEATURE_NAMES), "parameters": params,
              "curve_landmarks": [curves[0], curves[1]],
              "training_capture": train["capture"],
              "heldout_capture": heldout["capture"],
              "current_state_only": True}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, bundle_path, compress=3)
    report["candidate_bundle_sha256"] = hashlib.sha256(
        bundle_path.read_bytes()).hexdigest()
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
    print(json.dumps({"output": args.output.resolve().relative_to(ROOT).as_posix(),
                      "baseline": report["heldout_one_step"]["baseline"],
                      "corrected": report["heldout_one_step"]["corrected"],
                      "blocks": report["heldout_one_step"][
                          "paired_five_second_blocks"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
