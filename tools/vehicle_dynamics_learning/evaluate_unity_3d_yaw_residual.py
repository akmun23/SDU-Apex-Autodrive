#!/usr/bin/env python3
"""Fit a causal 25 ms yaw-rate residual on r07 and recursively score on r11.

The source-mechanics plant remains the baseline. This experiment learns only
the error in its 25 ms body-yaw-rate update from the current predicted rigid
body/wheel state and present actuator commands. Simulator truth is used to
form training labels and evaluation metrics, never as a rollout input.
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
from threadpoolctl import threadpool_limits

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


CORRECTION_PERIOD = 25
DEFAULT_OUTPUT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "source_mechanics_3d_yaw_residual_dagger_20261013.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


BASE_FEATURE_NAMES = (
    "body_vx", "body_vy", "body_vz",
    "body_omega_x", "body_omega_y", "body_omega_z",
    "wheel_omega_fl", "wheel_omega_fr", "wheel_omega_rl", "wheel_omega_rr",
    "center_steering", "steering_command_rad", "throttle_command",
    "previous_steering_command_rad", "previous_throttle_command",
    "body_up_x", "body_up_z",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    f"{wheel}_{name}"
    for wheel in WHEELS
    for name in ("contact_forward_speed", "contact_lateral_speed",
                 "longitudinal_slip", "lateral_slip", "normal_load_n"))
COMMAND_PROFILE_SUMMARY_NAMES = tuple(
    f"{command}_{stat}"
    for command in ("steering_rad", "throttle_norm")
    for stat in ("first", "last", "mean", "minimum", "maximum",
                 "total_variation", "max_step", "std"))
COMMAND_PROFILE_SAMPLE_NAMES = tuple(
    f"{command}_sample_{sample:02d}"
    for command in ("steering_rad", "throttle_norm")
    for sample in range(CORRECTION_PERIOD))
COMMAND_PROFILE_FEATURE_NAMES = COMMAND_PROFILE_SUMMARY_NAMES
FULL_COMMAND_PROFILE_FEATURE_NAMES = (
    COMMAND_PROFILE_SUMMARY_NAMES + COMMAND_PROFILE_SAMPLE_NAMES)


def _command_profile(data: dict[str, np.ndarray], start: int,
                     stop: int, full_resolution: bool = False
                     ) -> np.ndarray:
    """Summarize the known 1 kHz commands over one 25 ms plant interval."""
    if stop <= start:
        raise ValueError("command profile must contain at least one sample")
    summaries = []
    samples = []
    for name, scale in (("steering_command", math.radians(30.0)),
                        ("throttle_command", 1.0)):
        values = np.asarray(data[name][start:stop], dtype=np.float64) * scale
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("command profile contains no finite samples")
        if full_resolution and len(values) != CORRECTION_PERIOD:
            raise ValueError("full command profile requires exactly 25 samples")
        changes = np.abs(np.diff(values))
        summaries.extend((
            float(values[0]), float(values[-1]), float(np.mean(values)),
            float(np.min(values)), float(np.max(values)),
            float(np.sum(changes)),
            float(np.max(changes)) if len(changes) else 0.0,
            float(np.std(values)),
        ))
        if full_resolution:
            samples.extend(values.tolist())
    return np.asarray(summaries + samples, dtype=np.float64)


def _features(state: np.ndarray, steering_command: float,
              throttle_command: float, previous_steering: float,
              previous_throttle: float, constants: Any,
              command_profile: np.ndarray | None = None,
              full_command_profile: bool = False) -> np.ndarray:
    q = state[3:7]
    # Express world-up in the body frame. Using q instead of q^-1 would rotate
    # the roll/pitch feature with global heading and leak the world yaw angle.
    qv = -q[:3]
    twice_cross = 2.0 * np.cross(qv, np.asarray((0.0, 1.0, 0.0)))
    body_up = (np.asarray((0.0, 1.0, 0.0)) + q[3] * twice_cross
               + np.cross(qv, twice_cross))
    (mounts, down_local, wheel_q, _, springs, dampers, sprung,
     target_travel, _, _, _, _, _, com, _, slip_epsilon,
     radius_scale) = constants

    def rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        qv_local = quaternion[:3]
        cross = 2.0 * np.cross(qv_local, vector)
        return vector + quaternion[3] * cross + np.cross(qv_local, cross)

    q_inv = q.copy()
    q_inv[:3] *= -1.0
    ground_normal_body = rotate(q_inv, np.asarray((0.0, 1.0, 0.0)))
    tangent = math.tan(float(state[17]))
    numerator = 2.0 * 0.324 * tangent
    left = math.atan2(numerator, 2.0 * 0.324 - 0.236 * tangent)
    right = math.atan2(numerator, 2.0 * 0.324 + 0.236 * tangent)
    physical_steers = (-left, -right, 0.0, 0.0)
    wheel_features = []
    for index in range(4):
        center_world = state[:3] + rotate(q, mounts[index])
        down_world = rotate(q, down_local[index])
        denominator = float(down_world[1])
        if abs(denominator) < 1.0e-6:
            raise ValueError("wheel suspension ray is parallel to the ground")
        travel = (0.059 - float(center_world[1])) / denominator
        center_lever = mounts[index] + travel * down_local[index] - com
        velocity_center = state[7:10] + np.cross(state[10:13], center_lever)
        travel_rate = -float(np.dot(ground_normal_body, velocity_center)) / float(
            np.dot(ground_normal_body, down_local[index]))
        load = max(0.0, sprung[index] * 9.80665
                   + springs[index] * (target_travel[index] - travel)
                   - dampers[index] * travel_rate)

        steer = physical_steers[index]
        local_forward = np.asarray((math.sin(steer), 0.0, math.cos(steer)))
        wheel_forward = rotate(wheel_q[index], local_forward)
        wheel_forward -= ground_normal_body * float(np.dot(
            wheel_forward, ground_normal_body))
        wheel_forward /= max(float(np.linalg.norm(wheel_forward)), 1.0e-12)
        wheel_side = np.cross(ground_normal_body, wheel_forward)
        wheel_side /= max(float(np.linalg.norm(wheel_side)), 1.0e-12)
        contact = (mounts[index] + travel * down_local[index]
                   - 0.059 * ground_normal_body)
        contact_velocity = (state[7:10]
                            + np.cross(state[10:13], contact - com))
        contact_forward_speed = float(np.dot(contact_velocity, wheel_forward))
        contact_lateral_speed = float(np.dot(contact_velocity, wheel_side))
        wheel_surface_speed = state[13 + index] * 0.059 * radius_scale
        longitudinal_slip = (
            (wheel_surface_speed - contact_forward_speed)
            / (max(abs(wheel_surface_speed), abs(contact_forward_speed))
               + slip_epsilon))
        lateral_slip = math.atan2(contact_lateral_speed,
                                   abs(contact_forward_speed) + 1.0)
        wheel_features.extend((contact_forward_speed, contact_lateral_speed,
                               longitudinal_slip, lateral_slip, load))

    result = np.asarray((
        *state[7:13], *state[13:17], state[17],
        steering_command * math.radians(30.0), throttle_command,
        previous_steering * math.radians(30.0), previous_throttle,
        body_up[0], body_up[2],
        *wheel_features,
    ), dtype=np.float64)
    if command_profile is not None:
        profile_names = (FULL_COMMAND_PROFILE_FEATURE_NAMES
                         if full_command_profile
                         else COMMAND_PROFILE_FEATURE_NAMES)
        if command_profile.shape != (len(profile_names),):
            raise ValueError("unexpected command-profile feature shape")
        result = np.concatenate((result, command_profile))
    return result


def _training_rows(path: Path, constants: Any,
                   include_command_profile: bool = False,
                   full_command_profile: bool = False
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    data, _ = _load_capture(path)
    features: list[np.ndarray] = []
    targets: list[float] = []
    skipped_nonconstant_commands = 0
    included_nonconstant_commands = 0
    skipped_airborne = 0
    for start in range(100, len(data["speed_mps"]) - CORRECTION_PERIOD - 1,
                       CORRECTION_PERIOD):
        stop = start + CORRECTION_PERIOD
        if data["speed_mps"][start] < 1.0:
            continue
        if any(np.min(data[f"{wheel}_grounded"][start:stop + 1]) < 0.5
               for wheel in WHEELS):
            skipped_airborne += 1
            continue
        steering = data["steering_command"][start:stop]
        throttle = data["throttle_command"][start:stop]
        constant_command = (np.all(steering == steering[0])
                            and np.all(throttle == throttle[0]))
        if not constant_command and not include_command_profile:
            skipped_nonconstant_commands += 1
            continue
        if not constant_command:
            included_nonconstant_commands += 1

        state = _initial_state(data, start)
        previous_steering = float(data["steering_command"][start - 1])
        previous_throttle = float(data["throttle_command"][start - 1])
        profile = (_command_profile(data, start, stop, full_command_profile)
                   if include_command_profile else None)
        features.append(_features(
            state, float(steering[0]), float(throttle[0]),
            previous_steering, previous_throttle, constants, profile,
            full_command_profile))
        for offset in range(CORRECTION_PERIOD):
            index = start + offset
            state = _advance(state, data["steering_command"][index],
                             data["throttle_command"][index], constants, DT_S)
        targets.append(float(data["omega_body_y_rps"][stop] - state[11]))

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if len(y) < 500:
        raise ValueError(f"too few valid 25 ms training transitions: {len(y)}")
    return x, y, {
        "rows": len(y),
        "excluded_nonconstant_command_windows": skipped_nonconstant_commands,
        "included_nonconstant_command_windows": included_nonconstant_commands,
        "uses_1khz_command_profile": include_command_profile,
        "excluded_airborne_windows": skipped_airborne,
        "target_mean_radps": float(np.mean(y)),
        "target_rmse_radps": float(np.sqrt(np.mean(y * y))),
        "target_p95_abs_radps": float(np.quantile(np.abs(y), 0.95)),
    }


def _dagger_rows(path: Path, constants: Any,
                 rollout_model: HistGradientBoostingRegressor,
                 horizon_s: float, stride_s: float,
                 include_command_profile: bool = False,
                 full_command_profile: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect residual labels on predicted states visited by the current model."""
    data, _ = _load_capture(path)
    steps = round(horizon_s / DT_S)
    stride = round(stride_s / DT_S)
    starts = _evaluation_starts(data, steps, stride)
    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    skipped_nonconstant_commands = 0
    for start in starts:
        state = _initial_state(data, start)
        for block_start in range(start, start + steps, CORRECTION_PERIOD):
            steering = float(data["steering_command"][block_start])
            throttle = float(data["throttle_command"][block_start])
            previous_steering = float(
                data["steering_command"][max(0, block_start - 1)])
            previous_throttle = float(
                data["throttle_command"][max(0, block_start - 1)])
            endpoint = min(block_start + CORRECTION_PERIOD,
                           start + steps)
            profile = (_command_profile(
                data, block_start, endpoint, full_command_profile)
                       if include_command_profile else None)
            feature = _features(state, steering, throttle,
                                previous_steering, previous_throttle, constants,
                                profile, full_command_profile)
            baseline_predicted = state.copy()
            corrected_predicted = state.copy()
            residual_delta = float(rollout_model.predict(
                feature.reshape(1, -1))[0])
            residual_acceleration = residual_delta / (
                CORRECTION_PERIOD * DT_S)
            constant_command = (
                np.all(data["steering_command"][block_start:endpoint]
                       == steering)
                and np.all(data["throttle_command"][block_start:endpoint]
                           == throttle))
            for index in range(block_start, endpoint):
                command_steering = data["steering_command"][index]
                command_throttle = data["throttle_command"][index]
                baseline_predicted = _advance(
                    baseline_predicted, command_steering,
                    command_throttle, constants, DT_S)
                corrected_predicted = _advance(
                    corrected_predicted, command_steering, command_throttle,
                    constants, DT_S, residual_acceleration)
            target = float(data["omega_body_y_rps"][endpoint]
                           - baseline_predicted[11])
            if constant_command or include_command_profile:
                feature_rows.append(feature)
                target_rows.append(target)
            else:
                skipped_nonconstant_commands += 1
            # Continue with the correction applied as a yaw moment residual
            # during every 1 ms plant step, not as an endpoint state teleport.
            state = corrected_predicted
    x = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(target_rows, dtype=np.float64)
    return x, y, {
        "windows": len(starts),
        "rows": len(y),
        "excluded_nonconstant_command_windows": skipped_nonconstant_commands,
        "uses_1khz_command_profile": include_command_profile,
        "target_rmse_radps": float(np.sqrt(np.mean(y * y))),
        "target_p95_abs_radps": float(np.quantile(np.abs(y), 0.95)),
        "state_source": "current-model corrected recursive training-capture rollout; no state refresh within each window",
    }


def _reset_transition_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    positions = np.column_stack((
        data["pos_world_x_m"], data["pos_world_y_m"],
        data["pos_world_z_m"]))
    # Source captures can contain explicit reset-manager teleports between
    # experiment phases. A 1 ms physical step at the vehicle's tested speeds
    # is at most centimeters; exclude a complete rollout window if any
    # adjacent recorded poses jump by more than 5 cm.
    return np.linalg.norm(np.diff(positions, axis=0), axis=1) > 0.05


def _evaluation_starts(data: dict[str, np.ndarray], horizon_steps: int,
                       stride_steps: int) -> list[int]:
    reset_transition = _reset_transition_mask(data)
    reset_prefix = np.concatenate(([0], np.cumsum(reset_transition, dtype=np.int64)))
    starts = range(100, len(data["speed_mps"]) - horizon_steps - 1,
                   stride_steps)
    return [start for start in starts
            if data["speed_mps"][start] >= 2.0
            and reset_prefix[start + horizon_steps] - reset_prefix[start] == 0
            and all(np.min(data[f"{wheel}_grounded"][
                start:start + horizon_steps + 1]) > 0.5 for wheel in WHEELS)]


def _score(path: Path, constants: Any,
           models: dict[str, HistGradientBoostingRegressor],
           horizon_s: float, stride_s: float,
           profiled_models: set[str] | None = None,
           full_profile_models: set[str] | None = None) -> dict[str, Any]:
    data, _ = _load_capture(path)
    steps = round(horizon_s / DT_S)
    stride = round(stride_s / DT_S)
    starts = _evaluation_starts(data, steps, stride)
    if not starts:
        raise ValueError(f"no fully grounded evaluation windows in {path}")
    checkpoints = {round(ms / 1000.0 / DT_S): f"{ms / 1000.0:.3f}s"
                   for ms in HORIZON_MILLISECONDS
                   if ms / 1000.0 <= horizon_s}
    modes = ("baseline", *models.keys())
    errors: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {label: [] for label in checkpoints.values()} for mode in modes}
    correction_abs: dict[str, list[float]] = {mode: [] for mode in models}
    profiled_models = profiled_models or set()
    full_profile_models = full_profile_models or set()
    window_rows = []
    for start in starts:
        states = {mode: _initial_state(data, start) for mode in modes}
        initial_abs_steering = abs(float(states["baseline"][17]))
        window_corrections = {mode: [] for mode in models}
        per_window_errors = {mode: {} for mode in states}
        residual_acceleration = {mode: 0.0 for mode in modes}
        for offset in range(steps):
            index = start + offset
            steering = float(data["steering_command"][index])
            throttle = float(data["throttle_command"][index])
            if offset % CORRECTION_PERIOD == 0:
                previous_steering = float(
                    data["steering_command"][max(0, index - 1)])
                previous_throttle = float(
                    data["throttle_command"][max(0, index - 1)])
                for mode, model in models.items():
                    profile = (_command_profile(
                        data, index, index + CORRECTION_PERIOD,
                        mode in full_profile_models)
                        if mode in profiled_models else None)
                    feature = _features(
                        states[mode], steering, throttle,
                        previous_steering, previous_throttle, constants,
                        profile, mode in full_profile_models)
                    correction_delta = float(model.predict(
                        feature.reshape(1, -1))[0])
                    residual_acceleration[mode] = correction_delta / (
                        CORRECTION_PERIOD * DT_S)
                    correction_abs[mode].append(abs(correction_delta))
                    window_corrections[mode].append(correction_delta)
            for mode, state in states.items():
                state[:] = _advance(
                    state, steering, throttle, constants, DT_S,
                    residual_acceleration[mode])
            elapsed = offset + 1
            if elapsed in checkpoints:
                truth = _initial_state(data, index + 1)
                for mode, state in states.items():
                    error = _state_error(state, truth)
                    errors[mode][checkpoints[elapsed]].append(error)
                    per_window_errors[mode][checkpoints[elapsed]] = error
        window_rows.append({
            "start_sample": start,
            "initial_speed_mps": float(data["speed_mps"][start]),
            "initial_abs_steering_rad": initial_abs_steering,
            "residual_corrections_radps_per_25ms": {
                mode: {
                    "mean_abs": float(np.mean(np.abs(values))),
                    "net": float(np.sum(values)),
                }
                for mode, values in window_corrections.items()
            },
            "errors": per_window_errors,
        })

    summaries: dict[str, Any] = {}
    for mode, horizon_errors in errors.items():
        summaries[mode] = {}
        for label, rows in horizon_errors.items():
            position = np.asarray([row["position_horizontal_m"] for row in rows])
            heading = np.asarray([row["heading_rad"] for row in rows])
            velocity = np.asarray([row["body_velocity_xyz_mps"] for row in rows])
            omega = np.asarray([row["body_omega_xyz_radps"] for row in rows])
            wheel = np.asarray([row["wheel_omega_radps"] for row in rows])
            summaries[mode][label] = {
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
        "horizon_s": steps * DT_S,
        "window_stride_s": stride * DT_S,
        "errors": summaries,
        "residual_correction_abs_p50_p95_max_radps_per_25ms": {
            mode: [float(value) for value in np.quantile(
                values, (0.50, 0.95, 1.0))]
            for mode, values in correction_abs.items()
        },
        "windows": window_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--heldout", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--additional-heldout", type=Path, action="append",
                        default=[], help="additional untouched capture to score")
    parser.add_argument("--tire-fit", type=Path, default=DEFAULT_TIRE_FIT)
    parser.add_argument("--slip-fit", type=Path, default=DEFAULT_SLIP_FIT)
    parser.add_argument("--suspension-fit", type=Path,
                        default=DEFAULT_SUSPENSION_FIT)
    parser.add_argument("--wheel-fit", type=Path, default=DEFAULT_WHEEL_FIT)
    parser.add_argument("--horizon-seconds", type=float, default=5.0)
    parser.add_argument("--window-stride-seconds", type=float, default=5.0)
    parser.add_argument(
        "--command-profile-candidate", action="store_true",
        help="fit a paired candidate using 25 ms command summary features, "
             "including intervals with within-step input changes")
    parser.add_argument(
        "--full-command-profile-candidate", action="store_true",
        help="use all 25 one-millisecond command samples per 25 ms interval")
    parser.add_argument("--dagger-iterations", type=int, default=1,
                        help="number of sequential rollout-state aggregation rounds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-output", type=Path,
        help="where to save a reusable offline plant/residual bundle; defaults to the report path with .joblib")
    args = parser.parse_args()
    if args.command_profile_candidate and args.full_command_profile_candidate:
        parser.error("choose summary or full command-profile mode, not both")
    if args.dagger_iterations < 1:
        parser.error("--dagger-iterations must be at least 1")
    fit_command_profile = (args.command_profile_candidate
                           or args.full_command_profile_candidate)
    full_command_profile = args.full_command_profile_candidate
    profile_feature_names = (
        FULL_COMMAND_PROFILE_FEATURE_NAMES if full_command_profile
        else COMMAND_PROFILE_FEATURE_NAMES)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    model_output = (args.model_output if args.model_output is not None
                    else args.output.with_suffix(".joblib"))
    if model_output.exists():
        raise FileExistsError(f"refusing to overwrite {model_output}")
    if not 0.025 <= args.horizon_seconds <= 5.0:
        parser.error("horizon must be between 0.025 and 5 seconds")
    # Prediction is sequential during DAgger collection; avoid repeatedly
    # spawning a large OpenMP team for one-row tree predictions.
    _thread_limiter = threadpool_limits(limits=1)

    constants, base_description = _prepare_constants(
        _metadata(args.train), args.tire_fit, args.slip_fit,
        args.suspension_fit, args.wheel_fit)
    x_train, y_train, train_diag = _training_rows(args.train, constants)
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=10.0,
        random_state=17,
    )
    model.fit(x_train, y_train)
    training_fit = model.predict(x_train)
    first_pass_diag = {
        "training_rows": len(y_train),
        "in_sample_residual_rmse_radps": float(np.sqrt(np.mean(
            (training_fit - y_train) ** 2))),
        "training_target_rmse_radps": float(np.sqrt(np.mean(y_train ** 2))),
        "feature_names": list(FEATURE_NAMES),
        "algorithm": "regularized HistGradientBoostingRegressor; r07 only",
        "parameters": model.get_params(),
    }
    x_dagger, y_dagger, dagger_diag = _dagger_rows(
        args.train, constants, model, args.horizon_seconds,
        args.window_stride_seconds)
    dagger_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=10.0,
        random_state=19,
    )
    dagger_model.fit(np.concatenate((x_train, x_dagger)),
                     np.concatenate((y_train, y_dagger)))
    combined_fit = dagger_model.predict(np.concatenate((x_train, x_dagger)))
    combined_target = np.concatenate((y_train, y_dagger))
    dagger_model_diag = {
        "initial_teacher_forced_training": first_pass_diag,
        "dagger_rollout_data": dagger_diag,
        "combined_fit_rmse_radps": float(np.sqrt(np.mean(
            (combined_fit - combined_target) ** 2))),
        "algorithm": "regularized HistGradientBoostingRegressor on r07 teacher-forced plus first-pass DAgger states",
        "parameters": dagger_model.get_params(),
    }
    models_to_score = {
        "first_pass_continuous": model,
        "dagger_continuous": dagger_model,
    }
    profiled_models: set[str] = set()
    full_profile_models: set[str] = set()
    profile_diagnostics = None
    if fit_command_profile:
        profile_x, profile_y, profile_train_diag = _training_rows(
            args.train, constants, include_command_profile=True,
            full_command_profile=full_command_profile)
        profile_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=10.0,
            random_state=23,
        )
        profile_model.fit(profile_x, profile_y)
        profile_fit = profile_model.predict(profile_x)
        profile_first_pass_diag = {
            "training_rows": len(profile_y),
            "in_sample_residual_rmse_radps": float(np.sqrt(np.mean(
                (profile_fit - profile_y) ** 2))),
            "training_target_rmse_radps": float(np.sqrt(np.mean(
                profile_y ** 2))),
            "feature_names": [*FEATURE_NAMES, *profile_feature_names],
            "uses_within_interval_commands": True,
            "command_profile_resolution": (
                "full_1ms_samples" if full_command_profile else "summary"),
            "algorithm": "regularized HistGradientBoostingRegressor; r07 only",
            "parameters": profile_model.get_params(),
        }
        profile_dagger_rounds = []
        profile_dagger_data_x: list[np.ndarray] = []
        profile_dagger_data_y: list[np.ndarray] = []
        profile_dagger_model = profile_model
        for dagger_round in range(1, args.dagger_iterations + 1):
            round_x, round_y, round_diag = _dagger_rows(
                args.train, constants, profile_dagger_model,
                args.horizon_seconds, args.window_stride_seconds,
                include_command_profile=True,
                full_command_profile=full_command_profile)
            profile_dagger_data_x.append(round_x)
            profile_dagger_data_y.append(round_y)
            aggregated_x = np.concatenate((profile_x,
                                           *profile_dagger_data_x))
            aggregated_y = np.concatenate((profile_y,
                                           *profile_dagger_data_y))
            next_model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=100,
                l2_regularization=10.0,
                random_state=28 + dagger_round,
            )
            next_model.fit(aggregated_x, aggregated_y)
            fit_prediction = next_model.predict(aggregated_x)
            model_name = ("profile_dagger_continuous" if dagger_round == 1
                          else f"profile_dagger_round_{dagger_round}_continuous")
            models_to_score[model_name] = next_model
            profiled_models.add(model_name)
            if full_command_profile:
                full_profile_models.add(model_name)
            profile_dagger_rounds.append({
                "round": dagger_round,
                "model_name": model_name,
                "rollout_data": round_diag,
                "aggregated_training_rows": int(len(aggregated_y)),
                "aggregated_fit_rmse_radps": float(np.sqrt(np.mean(
                    (fit_prediction - aggregated_y) ** 2))),
                "random_state": 28 + dagger_round,
            })
            profile_dagger_model = next_model
        profile_diagnostics = {
            "teacher_forced_training": {
                **profile_first_pass_diag,
                "coverage": profile_train_diag,
            },
            "dagger_iterations": profile_dagger_rounds,
            "feature_names": [*FEATURE_NAMES, *profile_feature_names],
            "command_profile_resolution": (
                "full_1ms_samples" if full_command_profile else "summary"),
        }
        models_to_score.update({
            "profile_first_pass_continuous": profile_model,
        })
        profiled_models.add("profile_first_pass_continuous")
        if full_command_profile:
            full_profile_models.add("profile_first_pass_continuous")
    report = {
        "schema_version": 1,
        "status": "research-only causal yaw-acceleration residual; not runtime odometry",
        "training_capture": _path_label(args.train),
        "heldout_capture": _path_label(args.heldout),
        "baseline_model": base_description,
        "target": "25 ms body omega-y endpoint residual versus command-only mechanics; DAgger labels begin from predicted r07 states",
        "feature_timing": (
            "predicted state at correction boundary plus current/previous "
            "actuator command; no future sensor or truth features"),
        "correction_timing": "convert predicted 25 ms yaw-rate increment residual to constant yaw angular acceleration and apply it at each 1 ms plant step until the next boundary",
        "training_diagnostics": train_diag,
        "first_pass_reference": str((ROOT / (
            "live_runs/unity_exact_binary_audit_20261012/"
            "source_mechanics_3d_yaw_residual_windows_20261013.json"
        ))),
        "residual_model": dagger_model_diag,
        "command_profile_candidate": profile_diagnostics,
        "independent_heldout_recursive": _score(
            args.heldout, constants, models_to_score, args.horizon_seconds,
            args.window_stride_seconds, profiled_models,
            full_profile_models),
        "additional_independent_heldouts": {
            _path_label(path): _score(
                path, constants, models_to_score, args.horizon_seconds,
                args.window_stride_seconds, profiled_models,
                full_profile_models)
            for path in args.additional_heldout
        },
        "limitations": [
            "r11 is held out from fitting; this is one independent capture, not broad generalization proof",
            "the fitted correction affects yaw acceleration only and does not directly correct tire forces or other state equations",
            "truth is used only for r07 targets and post-rollout scoring",
        ],
    }
    feature_names_by_model = {
        "first_pass_continuous": list(FEATURE_NAMES),
        "dagger_continuous": list(FEATURE_NAMES),
    }
    if fit_command_profile:
        profiled_names = [*FEATURE_NAMES, *profile_feature_names]
        for name in models_to_score:
            if name.startswith("profile_"):
                feature_names_by_model[name] = profiled_names
    bundle = {
        "schema_version": 1,
        "status": "offline command-only 3-D source-mechanics plant with yaw-acceleration residual",
        "preferred_model": (next(
            name for name in reversed(list(models_to_score))
            if name.startswith("profile_dagger_")
        ) if full_command_profile else
                            "dagger_continuous"),
        "models": models_to_score,
        "constants": constants,
        "feature_names_by_model": feature_names_by_model,
        "profiled_models": sorted(profiled_models),
        "full_profile_models": sorted(full_profile_models),
        "sample_period_s": DT_S,
        "correction_period_steps": CORRECTION_PERIOD,
        "training_capture": _path_label(args.train),
        "training_capture_sha256": _sha256(args.train),
        "selection_capture": _path_label(args.heldout),
        "selection_capture_sha256": _sha256(args.heldout),
        "command_profile_resolution": (
            "25 x 1 ms commands" if full_command_profile else
            "summary" if fit_command_profile else "none"),
        "future_simulator_truth_used_during_rollout": False,
        "state_equations": (
            "source-mechanics rigid-body, wheel-spin, and suspension plant; "
            "learned correction is yaw angular acceleration, refreshed every 25 ms"),
    }
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output, compress=3)
    report["model_artifact"] = _path_label(model_output)
    report["model_artifact_sha256"] = _sha256(model_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    print("training target RMSE", train_diag["target_rmse_radps"])
    print("DAgger target RMSE", dagger_diag["target_rmse_radps"])
    scored_reports = {"primary": report["independent_heldout_recursive"],
                      **report["additional_independent_heldouts"]}
    for capture_name, scored in scored_reports.items():
        scores = scored["errors"]
        print(capture_name, "windows", scored["window_count"])
        for horizon in ("0.025s", "0.100s", "0.500s", "1.000s", "5.000s"):
            if horizon not in scores["baseline"]:
                continue
            b = scores["baseline"][horizon]
            print(horizon, "baseline",
                  "yaw", round(b["body_omega_xyz_rmse_radps"][1], 4),
                  "heading", round(b["heading_rmse_rad"], 4),
                  "position", round(b["position_horizontal_rmse_m"], 4))
            for mode in models_to_score:
                c = scores[mode][horizon]
                print(" ", mode,
                      "yaw", round(c["body_omega_xyz_rmse_radps"][1], 4),
                      "heading", round(c["heading_rmse_rad"], 4),
                      "position", round(c["position_horizontal_rmse_m"], 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
