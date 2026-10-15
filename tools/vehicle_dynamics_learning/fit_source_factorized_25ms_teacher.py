#!/usr/bin/env python3
"""Fit and evaluate a Unity-source-informed, factorized 25 ms plant teacher.

Unity's deterministic actuator/geometry equations are calculated directly.
Separate data-driven heads represent the native WheelCollider response that
is not available in public Unity source: longitudinal/lateral body accelera-
tion, yaw acceleration, and the two observed rear-wheel surface-speed changes.
Training uses only whole runs marked ``train``. Validation and the separate
practice captures are scored as recursive rollouts; test/final-test rows are
never used for fitting, checkpoint selection, or scoring.

This is an offline teacher experiment, not a production odometry/MPC change.
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

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    MOTOR_TORQUE_NM,
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
    TRACK_M,
    WHEELBASE_ACKERMANN_M,
    WHEEL_X_M,
    WHEEL_Y_M,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928"
DEFAULT_DATASET = DATA_ROOT / (
    "plant_teacher_race_domain_v1/cooldown_1s/openplane_dynamics.npz")
DEFAULT_PRACTICE = DATA_ROOT / (
    "command_only_750ms_rollout_20261012/practice_two_run_dataset/"
    "openplane_dynamics.npz")
DEFAULT_OUTPUT = DATA_ROOT / "source_factorized_teacher_20261013"
DT_S = 0.025
COM_OFFSET_BODY_X_M = -0.00468
COM_OFFSET_BODY_Y_M = 0.00008
OUTPUT_NAMES = (
    "body_ax_mps2", "body_ay_mps2", "yaw_accel_rps2",
    "rear_left_surface_accel_mps2", "rear_right_surface_accel_mps2",
)
STATE_NAMES = (
    "pose_x_m", "pose_y_m", "yaw_rad", "u_com_mps", "v_com_mps",
    "yaw_rate_rps", "rear_left_surface_mps", "rear_right_surface_mps",
    "steering_feedback_rad", "throttle_feedback_norm",
)
HORIZONS_S = (0.025, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
FEATURE_VARIANTS = ("state_only", "source_derived")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _read(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "frames", "simulator_rigid_state", "simulator_pose_xyyaw",
            "sequence_bounds", "sequence_run_index", "run_ids",
            "run_splits",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"{path} missing required arrays: {missing}")
        data = {name: archive[name] for name in archive.files}
    dt = np.asarray(data["dt_s"], dtype=np.float64).reshape(-1)
    if len(dt) and not np.allclose(dt, DT_S, atol=1.0e-7, rtol=0.0):
        raise ValueError(f"expected fixed 25 ms grid; found dt={np.unique(dt)[:8]}")
    if data["frames"].shape[1] != 9:
        raise ValueError("unexpected fixed-40-Hz frame schema")
    if data["simulator_rigid_state"].shape[1] != 13:
        raise ValueError("unexpected rigid-state schema")
    return data


def _sequences(data: dict[str, np.ndarray], allowed: set[str]
               ) -> list[tuple[str, str, int, int]]:
    result = []
    ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["run_splits"]).astype(str)
    seq_runs = np.asarray(data["sequence_run_index"], dtype=np.int64)
    for sequence, (lo, hi) in enumerate(data["sequence_bounds"]):
        run_index = int(seq_runs[sequence])
        split = splits[run_index]
        if split not in allowed:
            continue
        lo, hi = int(lo), int(hi)
        if hi - lo >= 3:
            result.append((ids[run_index], split, lo, hi))
    return result


def _state_arrays(data: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    pose = np.asarray(data["simulator_pose_xyyaw"], dtype=np.float64)
    # The bag convention is already body-forward/body-left. The independent
    # frame audit found rigid[:,7] == u and rigid[:,8] == v_COM exactly; do not
    # rotate these velocity channels by the world-pose quaternion.
    state = np.column_stack((
        pose[:, 0], pose[:, 1], pose[:, 2],
        rigid[:, 7], rigid[:, 8], rigid[:, 12],
        frames[:, 5], frames[:, 6], frames[:, 3], frames[:, 4],
    ))
    return state, frames


def _features(data: dict[str, np.ndarray], indices: np.ndarray,
              current_state: np.ndarray, previous_state: np.ndarray,
              variant: str) -> tuple[np.ndarray, tuple[str, ...]]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    command = frames[indices, 7:9]
    prior_command = frames[indices - 1, 7:9]
    u, v, yaw_rate = (current_state[:, index] for index in (3, 4, 5))
    steer, throttle = current_state[:, 8], current_state[:, 9]
    wheel_l, wheel_r = current_state[:, 6], current_state[:, 7]
    du = (u - previous_state[:, 3]) / DT_S
    dv = (v - previous_state[:, 4]) / DT_S
    dr = (yaw_rate - previous_state[:, 5]) / DT_S
    dw_l = (wheel_l - previous_state[:, 6]) / DT_S
    dw_r = (wheel_r - previous_state[:, 7]) / DT_S
    dsteer = (steer - previous_state[:, 8]) / DT_S
    dthrottle = (throttle - previous_state[:, 9]) / DT_S
    common = [
        u, v, yaw_rate, steer, throttle, wheel_l, wheel_r,
        command[:, 0], command[:, 1], prior_command[:, 0], prior_command[:, 1],
        command[:, 0] - steer, command[:, 1] - throttle,
        command[:, 0] - prior_command[:, 0],
        command[:, 1] - prior_command[:, 1],
        du, dv, dr, dw_l, dw_r, dsteer, dthrottle,
        u * yaw_rate, v * yaw_rate, np.hypot(u, v),
        np.abs(steer), steer * u, steer * np.abs(u),
    ]
    names = [
        "u_com", "v_com", "yaw_rate", "steering_feedback", "throttle_feedback",
        "rear_left_surface_speed", "rear_right_surface_speed",
        "steering_command", "throttle_command", "previous_steering_command",
        "previous_throttle_command", "steering_tracking_error",
        "throttle_tracking_error", "steering_command_slew",
        "throttle_command_slew", "previous_u_dot", "previous_v_dot",
        "previous_yaw_accel", "previous_rear_left_surface_accel",
        "previous_rear_right_surface_accel", "steering_feedback_rate",
        "throttle_feedback_rate", "u_yaw_rate", "v_yaw_rate", "body_speed",
        "abs_steering", "steering_times_u", "steering_times_abs_u",
    ]
    if variant == "source_derived":
        delta = np.clip(steer, -STEERING_LIMIT_RAD, STEERING_LIMIT_RAD)
        tangent = np.tan(delta)
        numerator = 2.0 * WHEELBASE_ACKERMANN_M * tangent
        front_left = np.arctan2(
            numerator, 2.0 * WHEELBASE_ACKERMANN_M - TRACK_M * tangent)
        front_right = np.arctan2(
            numerator, 2.0 * WHEELBASE_ACKERMANN_M + TRACK_M * tangent)
        wheel_angles = (front_left, front_right,
                        np.zeros_like(front_left), np.zeros_like(front_left))
        contact_long = []
        contact_lat = []
        for wheel in range(4):
            vx = u - yaw_rate * WHEEL_Y_M[wheel]
            vy = v + yaw_rate * WHEEL_X_M[wheel]
            c = np.cos(wheel_angles[wheel])
            s = np.sin(wheel_angles[wheel])
            contact_long.append(c * vx + s * vy)
            contact_lat.append(-s * vx + c * vy)
        contact_long_a = np.column_stack(contact_long)
        contact_lat_a = np.column_stack(contact_lat)
        slip_angle = np.arctan2(contact_lat_a,
                                np.maximum(np.abs(contact_long_a), 1.0e-6))
        # Only rear wheel RPM-derived surface speeds are measured in the
        # permitted telemetry. Front longitudinal wheel slip is deliberately
        # not fabricated from a no-slip assumption.
        rear_mismatch_l = wheel_l - contact_long_a[:, 2]
        rear_mismatch_r = wheel_r - contact_long_a[:, 3]
        # VehicleController's exact CAWD branch: any nonzero drive applies
        # 107*throttle Nm/wheel; exact zero applies 428 Nm brake/wheel.
        zero_throttle = (throttle == 0.0).astype(np.float64)
        drive_torque = (MOTOR_TORQUE_NM / 4.0) * throttle * (1.0 - zero_throttle)
        brake_torque = MOTOR_TORQUE_NM * zero_throttle
        extra = [
            contact_long_a[:, 0], contact_lat_a[:, 0],
            contact_long_a[:, 1], contact_lat_a[:, 1],
            contact_long_a[:, 2], contact_lat_a[:, 2],
            contact_long_a[:, 3], contact_lat_a[:, 3],
            slip_angle[:, 0], slip_angle[:, 1],
            slip_angle[:, 2], slip_angle[:, 3],
            rear_mismatch_l, rear_mismatch_r,
            np.abs(rear_mismatch_l), np.abs(rear_mismatch_r),
            rear_mismatch_l - rear_mismatch_r,
            (wheel_l + wheel_r) * 0.5,
            wheel_l - wheel_r,
            drive_torque, brake_torque,
            drive_torque * (wheel_l + wheel_r) * 0.5,
            np.sign(throttle), zero_throttle,
        ]
        names.extend((
            "front_left_contact_longitudinal_speed", "front_left_contact_lateral_speed",
            "front_right_contact_longitudinal_speed", "front_right_contact_lateral_speed",
            "rear_left_contact_longitudinal_speed", "rear_left_contact_lateral_speed",
            "rear_right_contact_longitudinal_speed", "rear_right_contact_lateral_speed",
            "front_left_slip_angle", "front_right_slip_angle",
            "rear_left_slip_angle", "rear_right_slip_angle",
            "rear_left_wheel_body_speed_mismatch", "rear_right_wheel_body_speed_mismatch",
            "abs_rear_left_mismatch", "abs_rear_right_mismatch",
            "left_right_mismatch_difference", "rear_mean_surface_speed",
            "rear_surface_speed_difference", "source_drive_torque_per_wheel",
            "source_brake_torque_per_wheel", "drive_torque_times_rear_speed",
            "throttle_sign", "exact_zero_throttle_brake_branch",
        ))
    matrix = np.column_stack(common + (extra if variant == "source_derived" else []))
    return matrix.astype(np.float32), tuple(names)


def _targets(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    state, _ = _state_arrays(data)
    current, following = state[indices], state[indices + 1]
    u_mid = 0.5 * (current[:, 3] + following[:, 3])
    v_mid = 0.5 * (current[:, 4] + following[:, 4])
    r_mid = 0.5 * (current[:, 5] + following[:, 5])
    ax = (following[:, 3] - current[:, 3]) / DT_S - r_mid * v_mid
    ay = (following[:, 4] - current[:, 4]) / DT_S + r_mid * u_mid
    alpha = (following[:, 5] - current[:, 5]) / DT_S
    wheel_accel = (following[:, 6:8] - current[:, 6:8]) / DT_S
    return np.column_stack((ax, ay, alpha, wheel_accel)).astype(np.float32)


def _training_indices(data: dict[str, np.ndarray]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = _sequences(data, {"train"})
    rows, run_names = [], []
    for run, _, lo, hi in sequences:
        if hi - lo < 3:
            continue
        rows.extend(range(lo + 1, hi - 1))
        run_names.extend([run] * (hi - lo - 2))
    indices = np.asarray(rows, dtype=np.int64)
    runs = np.asarray(run_names)
    if not len(indices):
        raise ValueError("no train transitions")
    # Equal total weight per run prevents long records from becoming the
    # implicit objective. All eligible fixed-grid train transitions are used.
    _, inverse, counts = np.unique(runs, return_inverse=True,
                                   return_counts=True)
    weights = (1.0 / counts[inverse]).astype(np.float32)
    weights *= len(weights) / weights.sum()
    return indices, runs, weights


def _fit_variant(data: dict[str, np.ndarray], variant: str,
                 indices: np.ndarray, runs: np.ndarray,
                 weights: np.ndarray, max_iter: int,
                 threads: int) -> dict[str, Any]:
    state, _ = _state_arrays(data)
    x, feature_names = _features(
        data, indices, state[indices], state[indices - 1], variant)
    y = _targets(data, indices)
    models = []
    with threadpool_limits(limits=threads):
        for output in range(y.shape[1]):
            model = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.05,
                max_iter=max_iter, max_leaf_nodes=31,
                min_samples_leaf=120, l2_regularization=8.0,
                early_stopping=False, random_state=20261013 + output)
            model.fit(x, y[:, output], sample_weight=weights)
            models.append(model)
    return {
        "variant": variant,
        "feature_names": feature_names,
        "output_names": OUTPUT_NAMES,
        "models": models,
        "training_rows": int(len(indices)),
        "training_run_count": int(len(np.unique(runs))),
    }


def _state_from_indices(data: dict[str, np.ndarray], indices: np.ndarray
                        ) -> np.ndarray:
    state, _ = _state_arrays(data)
    return state[indices].copy()


def _advance(states: np.ndarray, previous_states: np.ndarray,
             outputs: np.ndarray, command_indices: np.ndarray,
             data: dict[str, np.ndarray]) -> np.ndarray:
    frames = np.asarray(data["frames"], dtype=np.float64)
    result = states.copy()
    u, v, yaw_rate = states[:, 3], states[:, 4], states[:, 5]
    ax, ay, alpha = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    du0 = ax + yaw_rate * v
    dv0 = ay - yaw_rate * u
    u_mid = u + 0.5 * DT_S * du0
    v_mid = v + 0.5 * DT_S * dv0
    r_mid = yaw_rate + 0.5 * DT_S * alpha
    result[:, 3] = u + DT_S * (ax + r_mid * v_mid)
    result[:, 4] = v + DT_S * (ay - r_mid * u_mid)
    result[:, 5] = yaw_rate + DT_S * alpha
    result[:, 6:8] = states[:, 6:8] + DT_S * outputs[:, 3:5]
    # Unity Rigidbody.position is its transform origin, while Rigidbody.velocity
    # is COM velocity. Use the serialized prefab COM offset for the exact planar
    # origin/COM kinematic conversion.
    origin_u = u_mid + r_mid * COM_OFFSET_BODY_Y_M
    origin_v = v_mid - r_mid * COM_OFFSET_BODY_X_M
    yaw_mid = states[:, 2] + 0.5 * DT_S * r_mid
    result[:, 0] = states[:, 0] + DT_S * (
        np.cos(yaw_mid) * origin_u - np.sin(yaw_mid) * origin_v)
    result[:, 1] = states[:, 1] + DT_S * (
        np.sin(yaw_mid) * origin_u + np.cos(yaw_mid) * origin_v)
    result[:, 2] = _wrap_angle(states[:, 2] + DT_S * r_mid)

    # The fixed-grid held-out records show the actuator feedback at k+1 tracks
    # the command packet observed at k-1 best. This is one sample of bridge/
    # controller transport delay on top of the source rate limiter.
    command = frames[command_indices - 1, 7]
    target_steering = np.clip(command, -STEERING_LIMIT_RAD,
                              STEERING_LIMIT_RAD)
    steering_change = np.clip(
        target_steering - states[:, 8],
        -STEERING_RATE_RADPS * DT_S,
        STEERING_RATE_RADPS * DT_S)
    result[:, 8] = states[:, 8] + steering_change
    # Use the same measured one-sample command transport delay as steering.
    # This never reads the next recorded actuator feedback during rollout.
    result[:, 9] = frames[command_indices - 1, 8]
    return result


def _metrics(error: np.ndarray) -> dict[str, float]:
    return {
        "position_rmse_m": float(np.sqrt(np.mean(
            np.sum(error[:, 0:2] ** 2, axis=1)))),
        "yaw_rmse_rad": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "u_rmse_mps": float(np.sqrt(np.mean(error[:, 3] ** 2))),
        "v_rmse_mps": float(np.sqrt(np.mean(error[:, 4] ** 2))),
        "yaw_rate_rmse_rps": float(np.sqrt(np.mean(error[:, 5] ** 2))),
        "rear_left_surface_rmse_mps": float(np.sqrt(np.mean(error[:, 6] ** 2))),
        "rear_right_surface_rmse_mps": float(np.sqrt(np.mean(error[:, 7] ** 2))),
        "position_p95_m": float(np.quantile(
            np.linalg.norm(error[:, 0:2], axis=1), 0.95)),
        "yaw_abs_p95_rad": float(np.quantile(np.abs(error[:, 2]), 0.95)),
    }


def _evaluate_split(data: dict[str, np.ndarray], bundle: dict[str, Any],
                    split_names: set[str], label: str,
                    maximum_windows_per_sequence: int
                    ) -> dict[str, Any]:
    sequences = _sequences(data, split_names)
    truth_state, frames = _state_arrays(data)
    horizons_steps = {int(round(value / DT_S)): value for value in HORIZONS_S}
    starts_all: list[int] = []
    horizons_all: list[int] = []
    run_ids_all: list[str] = []
    for run_id, _, lo, hi in sequences:
        for horizon_steps in horizons_steps:
            if hi - lo < horizon_steps + 2:
                continue
            available = hi - lo - horizon_steps - 1
            count = min(maximum_windows_per_sequence,
                        max(1, int(math.ceil(available / 20))))
            starts = np.unique(np.linspace(
                lo + 1, hi - horizon_steps - 1, count, dtype=np.int64))
            starts_all.extend(starts.tolist())
            horizons_all.extend([horizon_steps] * len(starts))
            run_ids_all.extend([run_id] * len(starts))
    if not starts_all:
        return {
            "label": label,
            "split_allowlist": sorted(split_names),
            "test_and_final_test_used": False,
            "runs_scored": [],
            "macro_run_metrics": {},
            "per_run": {},
        }

    starts = np.asarray(starts_all, dtype=np.int64)
    horizon_steps = np.asarray(horizons_all, dtype=np.int64)
    run_ids = np.asarray(run_ids_all)
    current = truth_state[starts].copy()
    previous = truth_state[starts - 1].copy()
    maximum_steps = int(horizon_steps.max())
    variant = bundle["variant"]
    errors_by_run_horizon: dict[tuple[str, str], list[np.ndarray]] = {}
    with threadpool_limits(limits=4):
        for step in range(maximum_steps):
            active = horizon_steps > step
            if not np.any(active):
                break
            ix = starts[active] + step
            active_state = current[active]
            active_previous = previous[active]
            x, _ = _features(data, ix, active_state, active_previous, variant)
            outputs = np.column_stack([
                model.predict(x) for model in bundle["models"]]).astype(float)
            next_state = _advance(active_state, active_previous, outputs,
                                  ix, data)
            elapsed = step + 1
            scored = horizon_steps[active] == elapsed
            if np.any(scored):
                truth = truth_state[ix[scored] + 1]
                error = next_state[scored] - truth
                error[:, 2] = _wrap_angle(error[:, 2])
                for run_id in np.unique(run_ids[active][scored]):
                    mask = (run_ids[active][scored] == run_id)
                    horizon = f"{elapsed * DT_S:.3f}s"
                    errors_by_run_horizon.setdefault(
                        (str(run_id), horizon), []).append(error[mask])
            previous[active], current[active] = active_state, next_state

    per_run: dict[str, dict[str, Any]] = {}
    for (run_id, horizon), rows in errors_by_run_horizon.items():
        error = np.concatenate(rows, axis=0)
        per_run.setdefault(run_id, {})[horizon] = _metrics(error)
        per_run[run_id][horizon]["windows"] = int(len(error))
    macro: dict[str, dict[str, float]] = {}
    for horizon in sorted({h for rows in per_run.values() for h in rows},
                          key=lambda value: float(value.removesuffix("s"))):
        metrics = (
            "position_rmse_m", "yaw_rmse_rad", "u_rmse_mps",
            "v_rmse_mps", "yaw_rate_rmse_rps",
            "rear_left_surface_rmse_mps", "rear_right_surface_rmse_mps",
            "position_p95_m", "yaw_abs_p95_rad",
        )
        macro[horizon] = {
            key: float(np.mean([row[horizon][key]
                                for row in per_run.values()
                                if horizon in row]))
            for key in metrics
        }
    return {
        "label": label,
        "split_allowlist": sorted(split_names),
        "test_and_final_test_used": False,
        "runs_scored": sorted(per_run),
        "macro_run_metrics": macro,
        "per_run": per_run,
        "windows_by_horizon": {
            horizon: int(sum(per_run[run_id][horizon]["windows"]
                             for run_id in per_run if horizon in per_run[run_id]))
            for horizon in macro
        },
    }


def _evaluate_baseline(data: dict[str, np.ndarray], split_names: set[str],
                       maximum_windows_per_sequence: int
                       ) -> dict[str, Any]:
    # Baseline holds body accelerations and wheel surface speeds at zero while
    # using the same source actuator laws and exact pose integration.
    empty = {
        "variant": "state_only",
        "models": [None] * len(OUTPUT_NAMES),
    }
    class Zero:
        def predict(self, x: np.ndarray) -> np.ndarray:
            return np.zeros(len(x), dtype=np.float64)
    empty["models"] = [Zero() for _ in OUTPUT_NAMES]
    return _evaluate_split(data, empty, split_names,
                           "zero_dynamic_acceleration", maximum_windows_per_sequence)


def fit(dataset_path: Path, practice_path: Path, output_dir: Path,
        max_iter: int, threads: int,
        max_windows_per_sequence: int) -> dict[str, Any]:
    started = time.time()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    data = _read(dataset_path)
    practice = _read(practice_path)
    train_indices, train_runs, sample_weights = _training_indices(data)
    train_y = _targets(data, train_indices)
    train_labels = {
        name: {
            "rmse": float(np.sqrt(np.mean(train_y[:, index] ** 2))),
            "p95_abs": float(np.quantile(np.abs(train_y[:, index]), 0.95)),
        }
        for index, name in enumerate(OUTPUT_NAMES)
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "source-informed factorized offline plant teacher",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "practice_dataset": str(practice_path),
        "practice_dataset_sha256": _sha256(practice_path),
        "sample_period_s": DT_S,
        "state_names": STATE_NAMES,
        "target_names": OUTPUT_NAMES,
        "training_rows": int(len(train_indices)),
        "training_runs": sorted(set(train_runs.tolist())),
        "training_target_scales": train_labels,
        "direct_source_terms": [
            "fixed 25 ms timebase",
            "front Ackermann angles and per-wheel contact velocities",
            "active CAWD motor/brake branch including exact-zero throttle",
            "steering saturation and 183.346 deg/s slew",
            "body-frame rigid kinematics and prefab COM offset for pose integration",
        ],
        "learned_unavailable_native_terms": [
            "tire force/slip/load response",
            "yaw moment response",
            "rear WheelCollider spin response",
        ],
        "limitations": [
            "public Unity source does not expose the native WheelCollider solver",
            "only rear wheel surface speed is present in permitted telemetry; front wheel spin is not inferred",
            "offline candidate only; not integrated into production odometry or MPC",
        ],
        "validation_or_practice_used_for_fitting": False,
        "test_and_final_test_used": False,
        "variants": {},
    }

    for variant in FEATURE_VARIANTS:
        bundle = _fit_variant(data, variant, train_indices, train_runs,
                              sample_weights, max_iter, threads)
        model_path = output_dir / f"{variant}.joblib"
        joblib.dump(bundle, model_path, compress=3)
        variant_result = {
            "model_path": str(model_path),
            "model_sha256": _sha256(model_path),
            "feature_names": bundle["feature_names"],
            "training_rows": bundle["training_rows"],
            "training_run_count": bundle["training_run_count"],
            "scores": {
                "openplane_validation": _evaluate_split(
                    data, bundle, {"validation"}, "openplane_validation",
                    max_windows_per_sequence),
                "unseen_practice": _evaluate_split(
                    practice, bundle, {"validation"}, "unseen_practice",
                    max_windows_per_sequence),
            },
        }
        result["variants"][variant] = variant_result
        del bundle

    result["zero_dynamic_baseline"] = {
        "openplane_validation": _evaluate_baseline(
            data, {"validation"}, max_windows_per_sequence),
        "unseen_practice": _evaluate_baseline(
            practice, {"validation"}, max_windows_per_sequence),
    }
    result["elapsed_seconds"] = time.time() - started
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n",
                           encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-windows-per-sequence", type=int, default=12)
    args = parser.parse_args()
    report = fit(args.dataset, args.practice, args.output, args.max_iter,
                 args.threads, args.max_windows_per_sequence)
    print(json.dumps({
        "output": str(args.output / "evaluation.json"),
        "elapsed_seconds": report["elapsed_seconds"],
        "training_rows": report["training_rows"],
        "training_runs": len(report["training_runs"]),
        "variants": {
            name: {
                split: values["macro_run_metrics"]
                for split, values in row["scores"].items()
            }
            for name, row in report["variants"].items()
        },
        "zero_dynamic_baseline": {
            split: values["macro_run_metrics"]
            for split, values in report["zero_dynamic_baseline"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
