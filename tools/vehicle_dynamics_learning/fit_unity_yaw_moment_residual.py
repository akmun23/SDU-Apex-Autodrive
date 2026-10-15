#!/usr/bin/env python3
"""Fit a causal residual to the Unity-derived yaw-moment balance.

This is a diagnostic model, not production odometry. The rigid-body and
WheelCollider curve prediction remain explicit; a small boosted regressor
models only the unexplained body-yaw moment. Training uses r07 only and the
whole r11 capture is held out. Features are current body/actuator/wheel state,
mechanically reconstructed slip/load, and optional current attitude. Future
WheelHit values, future IMU values, and future body state are never features.
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
from scipy.spatial.transform import Rotation
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.vehicle_dynamics_learning.analyze_unity_3d_moment_balance import (
    DEFAULT_CAPTURES,
    _balance_data,
)
from tools.vehicle_dynamics_learning.evaluate_unity_wheel_spin_state_inputs import (
    DEFAULT_SLIP_FIT,
    DEFAULT_SUSPENSION_FIT,
    _state_reconstructed_data,
    _total_contact_balance,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    RIGIDBODY_INERTIA_DIAGONAL_KGM2,
    RIGIDBODY_INERTIA_ROTATION_XYZW,
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
    TRACK_M,
    WHEELBASE_ACKERMANN_M,
    inertia_tensor_body,
)


ROOT = Path(__file__).resolve().parents[2]
TIRE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_20261012.json")
OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "yaw_moment_state_residual_20261013_v3.json")
WHEEL_ORDER = ("fl", "fr", "rl", "rr")
RIGIDBODY_INERTIA_KGM2 = np.asarray(inertia_tensor_body(
    RIGIDBODY_INERTIA_DIAGONAL_KGM2,
    RIGIDBODY_INERTIA_ROTATION_XYZW), dtype=np.float64)


def _features(data: dict[str, Any], reconstructed: dict[str, Any],
              include_attitude: bool) -> tuple[np.ndarray, list[str]]:
    count = len(data["speed"])
    velocity = data["velocity_body"]
    omega = data["omega_body"]
    command = data["steering_command"]
    throttle = data["throttle_command"]
    steering_center = reconstructed["predicted_steering_center_rad"]
    dt = float(data["metadata"]["fixed_delta_time_s"])
    parts = [velocity, omega, command[:, None], throttle[:, None],
             steering_center[:, None],
             np.hypot(velocity[:, 0], velocity[:, 2])[:, None]]
    names = ["velocity_body_x", "velocity_body_y", "velocity_body_z",
             "omega_body_x", "omega_body_y", "omega_body_z",
             "steering_command", "throttle_command", "steering_feedback_center",
             "body_speed"]
    for label, values in (("steering_command", command),
                          ("throttle_command", throttle),
                          ("steering_feedback", steering_center)):
        rate = np.zeros(count, dtype=np.float64)
        rate[1:] = np.diff(values) / dt
        parts.append(rate[:, None])
        names.append(f"{label}_rate")
    for wheel_name, wheel in zip(WHEEL_ORDER, reconstructed["wheels"]):
        wheel_features = np.column_stack((
            wheel["slip_x"], wheel["slip_y"], wheel["load"],
            wheel["omega"], wheel["reconstructed_contact_speed_mps"],
            wheel["reconstructed_lateral_contact_speed_mps"]))
        parts.append(wheel_features)
        names.extend((f"{wheel_name}_slip_x", f"{wheel_name}_slip_y",
                      f"{wheel_name}_normal_load_n",
                      f"{wheel_name}_wheel_omega_radps",
                      f"{wheel_name}_contact_u_mps",
                      f"{wheel_name}_contact_v_mps"))
    attitude_names: list[str] = []
    if include_attitude:
        quaternion = data["rotation_world_xyzw"]
        body_up = Rotation.from_quat(quaternion).apply(
            np.broadcast_to((0.0, 1.0, 0.0), (count, 3)))
        body_forward = Rotation.from_quat(quaternion).apply(
            np.broadcast_to((0.0, 0.0, 1.0), (count, 3)))
        roll = np.arctan2(body_up[:, 0], body_up[:, 1])
        pitch = np.arctan2(-body_forward[:, 1],
                           np.hypot(body_forward[:, 0], body_forward[:, 2]))
        attitude = np.column_stack((roll, pitch))
        parts.append(attitude)
        attitude_names.extend(("body_roll_from_road_rad",
                               "body_pitch_from_road_rad"))
        names.extend(attitude_names)
    force, torque = _total_contact_balance(
        reconstructed, reconstructed["fit_parameters"],
        use_force_application_point=True)
    parts.extend((force, torque))
    names.extend(("predicted_force_x_n", "predicted_force_y_n",
                  "predicted_force_z_n", "predicted_torque_x_nm",
                  "predicted_torque_y_nm", "predicted_torque_z_nm"))
    features = np.column_stack(parts).astype(np.float32)
    return features, names


def _predict_source_steering(data: dict[str, Any]) -> tuple[np.ndarray,
                                                             np.ndarray]:
    """Replay source steering rate/geometry from one initial feedback value."""
    dt = float(data["metadata"]["fixed_delta_time_s"])
    command = data["steering_command"]
    measured_left = data["wheels"][0]["steering_rad"]
    measured_right = data["wheels"][1]["steering_rad"]
    actual_center = 0.5 * (measured_left + measured_right)
    center = np.empty_like(actual_center)
    initial_valid = np.flatnonzero(
        np.isfinite(actual_center) & np.isfinite(command))
    if initial_valid.size == 0:
        raise ValueError("capture has no finite initial steering feedback")
    initial = int(initial_valid[0])
    center[:initial + 1] = actual_center[initial]
    maximum_step = STEERING_RATE_RADPS * dt
    for index in range(initial + 1, len(center)):
        center[index] = center[index - 1]
        if not np.isfinite(command[index]):
            continue
        target = -float(np.clip(command[index], -1.0, 1.0)) \
            * STEERING_LIMIT_RAD
        center[index] = center[index - 1] + np.clip(
            target - center[index - 1], -maximum_step, maximum_step)
    tangent = np.tan(center)
    numerator = 2.0 * WHEELBASE_ACKERMANN_M * tangent
    left = np.arctan2(
        numerator, 2.0 * WHEELBASE_ACKERMANN_M + TRACK_M * tangent)
    right = np.arctan2(
        numerator, 2.0 * WHEELBASE_ACKERMANN_M - TRACK_M * tangent)
    for wheel, angles in zip(data["wheels"][:2], (left, right)):
        wheel["steering_rad"] = angles
    data["wheels"][2]["steering_rad"] = np.zeros_like(center)
    data["wheels"][3]["steering_rad"] = np.zeros_like(center)
    return center, actual_center


def _paired_block_summary(data: dict[str, Any], rows: np.ndarray,
                          correction: np.ndarray,
                          block_seconds: float = 5.0,
                          bootstrap_replicates: int = 5000
                          ) -> dict[str, Any]:
    dt = float(data["data"]["metadata"]["fixed_delta_time_s"])
    block_steps = max(1, int(round(block_seconds / dt)))
    truth = data["data"]["torque_target_total"][rows, 1]
    baseline = data["predicted_torque"][rows, 1] - truth
    corrected = baseline + correction
    block_id = rows // block_steps
    per_block = []
    for block in np.unique(block_id):
        mask = block_id == block
        if np.count_nonzero(mask) < 20:
            continue
        per_block.append((
            float(np.sqrt(np.mean(baseline[mask] ** 2))),
            float(np.sqrt(np.mean(corrected[mask] ** 2))),
        ))
    values = np.asarray(per_block, dtype=np.float64)
    if len(values) < 2:
        return {"block_count": int(len(values)), "block_seconds": block_seconds}
    deltas = values[:, 1] - values[:, 0]
    rng = np.random.default_rng(20261013)
    draws = rng.integers(0, len(values),
                         size=(bootstrap_replicates, len(values)))
    bootstrap_delta = deltas[draws].mean(axis=1)
    return {
        "block_count": int(len(values)),
        "block_seconds": block_seconds,
        "baseline_block_rmse_median_nm": float(np.median(values[:, 0])),
        "corrected_block_rmse_median_nm": float(np.median(values[:, 1])),
        "mean_paired_block_delta_nm": float(np.mean(deltas)),
        "paired_block_bootstrap_95pct_delta_nm": [
            float(x) for x in np.quantile(bootstrap_delta, (0.025, 0.975))],
        "blocks_improved": int(np.count_nonzero(deltas < 0.0)),
        "per_block_baseline_corrected_rmse_nm": values.tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare(path: Path, derivative_window: int, sample_stride: int,
             tire_parameters: np.ndarray,
             slip_fit: dict[str, Any],
             suspension_fit: dict[str, Any]
             ) -> dict[str, Any]:
    data = _balance_data(path, derivative_window, sample_stride)
    predicted_steering, actual_steering = _predict_source_steering(data)
    reconstructed, diagnostics = _state_reconstructed_data(
        data, slip_fit, suspension_fit)
    reconstructed["predicted_steering_center_rad"] = predicted_steering
    reconstructed["fit_parameters"] = tire_parameters
    features_base, names_base = _features(data, reconstructed, False)
    features_attitude, names_attitude = _features(data, reconstructed, True)
    predicted_force, predicted_torque = _total_contact_balance(
        reconstructed, tire_parameters, use_force_application_point=True)
    rows = data["rows"]
    # Drop invalid and non-contact rows from fitting and scoring.
    valid = np.isfinite(features_attitude).all(axis=1)
    valid &= np.isfinite(data["torque_target_total"]).all(axis=1)
    for wheel in reconstructed["wheels"]:
        valid &= np.isfinite(wheel["load"])
        valid &= np.isfinite(wheel["slip_x"]) & np.isfinite(wheel["slip_y"])
    rows = rows[valid[rows]]
    return {
        "path": path,
        "data": data,
        "reconstructed": reconstructed,
        "features_base": features_base,
        "features_attitude": features_attitude,
        "feature_names_base": names_base,
        "feature_names_attitude": names_attitude,
        "predicted_force": predicted_force,
        "predicted_torque": predicted_torque,
        "rows": rows,
        "diagnostics": diagnostics,
        "steering_feedback_error_rad": predicted_steering - actual_steering,
    }


def _score(data: dict[str, Any], rows: np.ndarray,
           correction_nm: np.ndarray | None = None
           ) -> dict[str, Any]:
    force_error = (data["predicted_force"][rows]
                   - data["data"]["force_target_total"][rows])
    torque_prediction = data["predicted_torque"][rows].copy()
    if correction_nm is not None:
        torque_prediction[:, 1] += correction_nm
    torque_error = (torque_prediction
                    - data["data"]["torque_target_total"][rows])
    alpha_error = torque_error @ np.linalg.inv(RIGIDBODY_INERTIA_KGM2).T
    return {
        "samples": int(rows.size),
        "force_vector_rmse_n": float(np.sqrt(np.mean(
            np.sum(force_error**2, axis=1)))),
        "torque_rmse_by_body_axis_nm": np.sqrt(
            np.mean(torque_error**2, axis=0)).tolist(),
        "angular_acceleration_rmse_by_body_axis_radps2": np.sqrt(
            np.mean(alpha_error**2, axis=0)).tolist(),
        "angular_acceleration_abs_p95_by_body_axis_radps2": np.quantile(
            np.abs(alpha_error), 0.95, axis=0).tolist(),
    }


def _fit(train: dict[str, Any], feature_key: str,
         sample_stride: int) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    rows = train["rows"]
    x = train[feature_key][rows]
    y = (train["data"]["torque_target_total"][rows, 1]
         - train["predicted_torque"][rows, 1])
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[finite], y[finite]
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06,
        max_iter=140, max_leaf_nodes=15, min_samples_leaf=60,
        l2_regularization=5.0, early_stopping=False,
        random_state=20261013)
    model.fit(x, y)
    return model, {"fit_rows": int(len(y)),
                   "feature_names": train[
                       "feature_names_attitude" if feature_key
                       == "features_attitude" else "feature_names_base"],
                   "sample_stride_1khz_rows": sample_stride}


def _conditional_scores(data: dict[str, Any], correction: np.ndarray,
                        rows: np.ndarray) -> dict[str, Any]:
    speed = data["data"]["speed"][rows]
    steering = np.abs(data["data"]["steering_command"][rows])
    target = data["data"]["torque_target_total"][rows, 1]
    baseline = data["predicted_torque"][rows, 1]
    bins = {}
    for lo, hi in ((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.1)):
        for steer_lo, steer_hi in ((0.0, 0.15), (0.15, 0.35), (0.35, 1.01)):
            mask = ((speed >= lo) & (speed < hi)
                    & (steering >= steer_lo) & (steering < steer_hi))
            if not np.any(mask):
                continue
            before = baseline[mask] - target[mask]
            after = baseline[mask] + correction[mask] - target[mask]
            key = (f"speed_{lo:g}_{hi:g}_steer_"
                   f"{steer_lo:g}_{steer_hi:g}")
            bins[key] = {
                "samples": int(np.count_nonzero(mask)),
                "baseline_yaw_moment_rmse_nm": float(
                    np.sqrt(np.mean(before**2))),
                "corrected_yaw_moment_rmse_nm": float(
                    np.sqrt(np.mean(after**2))),
                "baseline_bias_nm": float(np.mean(before)),
                "corrected_bias_nm": float(np.mean(after)),
            }
    return bins


def run(train_path: Path, heldout_path: Path, output_path: Path,
        derivative_window: int, sample_stride: int) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    bundle_path = output_path.with_suffix(".joblib")
    if bundle_path.exists():
        raise FileExistsError(f"refusing to overwrite {bundle_path}")
    tire_fit = json.loads(TIRE_FIT.read_text(encoding="utf-8"))[
        "curve_model"]["training_fit"]
    tire_parameters = np.asarray((
        tire_fit["longitudinal_scale"],
        tire_fit["longitudinal_zero_slip_tangent"],
        1.1319111146687113, 76.02943431880527), dtype=np.float64)
    slip_fit = json.loads(DEFAULT_SLIP_FIT.read_text(encoding="utf-8"))
    suspension_fit = json.loads(
        DEFAULT_SUSPENSION_FIT.read_text(encoding="utf-8"))
    train = _prepare(train_path, derivative_window, sample_stride,
                     tire_parameters, slip_fit, suspension_fit)
    heldout = _prepare(heldout_path, derivative_window, sample_stride,
                       tire_parameters, slip_fit, suspension_fit)
    base_model, base_meta = _fit(train, "features_base", sample_stride)
    attitude_model, attitude_meta = _fit(
        train, "features_attitude", sample_stride)
    rows = heldout["rows"]
    base_prediction = base_model.predict(heldout["features_base"][rows])
    attitude_prediction = attitude_model.predict(
        heldout["features_attitude"][rows])
    base_score = _score(heldout, rows)
    base_corrected_score = _score(heldout, rows, base_prediction)
    attitude_corrected_score = _score(heldout, rows, attitude_prediction)
    report = {
        "schema_version": 1,
        "status": "one-step causal yaw-moment residual diagnostic; no production or recursive plant integration",
        "training_capture": str(train_path.relative_to(ROOT)),
        "heldout_capture": str(heldout_path.relative_to(ROOT)),
        "split_policy": "all fits trained on r07 only; complete r11 capture held out",
        "fixed_delta_time_s": float(train["data"]["metadata"][
            "fixed_delta_time_s"]),
        "derivative_window_samples": derivative_window,
        "sampling_stride_samples": sample_stride,
        "tire_parameters_training_only": tire_parameters.tolist(),
        "model": {
            "algorithm": "HistGradientBoostingRegressor(max_iter=140,max_leaf_nodes=15,min_samples_leaf=60,l2_regularization=5)",
            "target": "required body yaw moment minus source-derived tire/body yaw moment (N m)",
            "baseline_inputs": "current rigid-body state, current command/actuator feedback, state-reconstructed slip/load, source WheelCollider curve and force application point",
            "attitude_ablation": "same causal features, plus current body quaternion converted to Euler angles; no future attitude",
            "wheelhit_future_labels_used_as_features": False,
            "future_body_truth_used_as_features": False,
            "future_imu_used_as_features": False,
            "recursive_rollout_evaluated": False,
        },
        "training": {
            "rows": int(train["rows"].size),
            "base_model": base_meta,
            "attitude_model": attitude_meta,
        },
        "heldout_one_step": {
            "baseline": base_score,
            "base_state_residual": base_corrected_score,
            "state_plus_attitude_residual": attitude_corrected_score,
            "base_state_yaw_moment_by_speed_and_steering":
                _conditional_scores(heldout, base_prediction, rows),
            "state_plus_attitude_yaw_moment_by_speed_and_steering":
                _conditional_scores(heldout, attitude_prediction, rows),
            "base_state_paired_five_second_blocks": _paired_block_summary(
                heldout, rows, base_prediction),
            "state_plus_attitude_paired_five_second_blocks":
                _paired_block_summary(heldout, rows, attitude_prediction),
        },
        "source_steering_replay": {
            "heldout_feedback_error_rmse_rad": float(np.sqrt(np.mean(
                heldout["steering_feedback_error_rad"][np.isfinite(
                    heldout["steering_feedback_error_rad"])]**2))),
            "heldout_feedback_error_abs_p95_rad": float(np.quantile(np.abs(
                heldout["steering_feedback_error_rad"][np.isfinite(
                    heldout["steering_feedback_error_rad"])]), 0.95)),
            "initialization": "one measured average front steering value at capture start; subsequent center steering from source rate limit and command; Ackermann wheel angles from VehicleController.cs",
        },
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "decision": "only consider recursive integration if r11 yaw-moment and angular-acceleration gains are broad across speed/steering bins; force, lateral acceleration and recursive motion must then be revalidated separately",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "research_only": True,
        "training_capture": str(train_path.relative_to(ROOT)),
        "heldout_capture": str(heldout_path.relative_to(ROOT)),
        "tire_parameters": tire_parameters,
        "base_state_model": base_model,
        "attitude_model": attitude_model,
        "base_feature_names": train["feature_names_base"],
        "attitude_feature_names": train["feature_names_attitude"],
        "state_input_policy": "current state and current command only; no future simulator truth; future attitude is excluded from base_state_model",
    }
    joblib.dump(bundle, bundle_path, compress=3)
    report["candidate_bundle_sha256"] = _sha256(bundle_path)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_CAPTURES[0])
    parser.add_argument("--heldout", type=Path, default=DEFAULT_CAPTURES[1])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--derivative-window", type=int, default=31)
    parser.add_argument("--sample-stride", type=int, default=25)
    args = parser.parse_args()
    report = run(args.train.resolve(), args.heldout.resolve(), args.output.resolve(),
                 args.derivative_window, args.sample_stride)
    print(json.dumps({
        "output": args.output.resolve().relative_to(ROOT).as_posix(),
        "baseline": report["heldout_one_step"]["baseline"],
        "base_state_residual": report["heldout_one_step"][
            "base_state_residual"],
        "state_plus_attitude_residual": report["heldout_one_step"][
            "state_plus_attitude_residual"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
