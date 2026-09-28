#!/usr/bin/env python3
"""Whole-run nonlinear state/history sufficiency diagnostic for Explore bags.

This is a model-identification diagnostic, not a production plant or MPC
model. It compares causal feature ladders with a normalized k-nearest-neighbor
transition predictor and reports joint-support coverage. All splits are whole
bags; training/test samples are never randomly interleaved.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from tools import evaluate_open_plane_body_dynamics as body


FEATURE_LEVELS = {
    "M0_u_steering": ("u_mps", "steering_rad"),
    "M1_body_state": ("u_mps", "steering_rad", "v_rear_mps", "yaw_rate_rps"),
    "M2_throttle": ("u_mps", "steering_rad", "v_rear_mps", "yaw_rate_rps",
                    "throttle_feedback"),
    "M3_rear_wheel_slip": (
        "u_mps", "steering_rad", "v_rear_mps", "yaw_rate_rps",
        "throttle_feedback", "rear_left_slip_velocity_mps",
        "rear_right_slip_velocity_mps"),
    "M4_actuator_state": (
        "u_mps", "steering_rad", "v_rear_mps", "yaw_rate_rps",
        "throttle_feedback", "rear_left_slip_velocity_mps",
        "rear_right_slip_velocity_mps", "steering_rate_radps",
        "throttle_rate_per_s", "steering_command_rad", "throttle_command"),
    "M5_causal_history": (
        "u_mps", "steering_rad", "v_rear_mps", "yaw_rate_rps",
        "throttle_feedback", "rear_left_slip_velocity_mps",
        "rear_right_slip_velocity_mps", "steering_rate_radps",
        "throttle_rate_per_s", "steering_command_rad", "throttle_command",
        "q_mean_100ms", "q_mean_250ms", "q_mean_500ms",
        "throttle_mean_100ms", "throttle_mean_250ms",
        "throttle_mean_500ms", "q_abs_peak_500ms"),
    "M6_traction_history": (
        "u_mps", "steering_rad", "v_rear_mps", "yaw_rate_rps",
        "throttle_feedback", "rear_left_slip_velocity_mps",
        "rear_right_slip_velocity_mps", "steering_rate_radps",
        "throttle_rate_per_s", "steering_command_rad", "throttle_command",
        "q_mean_100ms", "q_mean_250ms", "q_mean_500ms",
        "throttle_mean_100ms", "throttle_mean_250ms",
        "throttle_mean_500ms", "q_abs_peak_500ms",
        "imu_ax_mps2", "imu_ay_mps2", "imu_ax_mean_100ms",
        "imu_ax_mean_250ms", "imu_ay_mean_100ms",
        "rear_left_slip_mean_250ms", "rear_right_slip_mean_250ms",
        "rear_left_slip_rate_mps2", "rear_right_slip_rate_mps2"),
}
FEATURE_LEVELS["M7_spin_accel_residual"] = (
    FEATURE_LEVELS["M6_traction_history"]
    + ("rear_spin_accel_residual_mean_mps2",
       "rear_spin_accel_residual_magnitude_mps2",
       "rear_spin_accel_residual_asymmetry_mps2"))

BASE_REQUIRED = FEATURE_LEVELS["M2_throttle"]
FULL_REQUIRED = FEATURE_LEVELS["M5_causal_history"]
TARGET_NAMES = ("u_dot_mps2", "v_dot_mps2", "yaw_accel_rps2")
STEERING_LIMIT_RAD = body.STEERING_LIMIT_RAD
MIN_DT_S = body.MIN_DT_S
MAX_DT_S = body.MAX_DT_S
NEIGHBOR_COUNT = 32


@dataclass(frozen=True)
class TransitionSet:
    values: dict[str, np.ndarray]
    target: np.ndarray
    truth_state: np.ndarray
    next_dt_s: np.ndarray
    phase: np.ndarray

    def select(self, names: tuple[str, ...] | list[str]) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.column_stack([self.values[name] for name in names])
        valid = np.isfinite(matrix).all(axis=1) & np.isfinite(self.target).all(axis=1)
        return matrix[valid], valid


def _trailing_mean(times: np.ndarray, values: np.ndarray, index: int,
                   window_s: float) -> float:
    """Sample mean over [t-window,t], requiring the full causal window."""
    end = float(times[index])
    if index == 0 or end - float(times[0]) < window_s - 0.012:
        return math.nan
    first = int(np.searchsorted(times[:index + 1], end - window_s, side="left"))
    selected = values[first:index + 1]
    return (float(np.mean(selected))
            if len(selected) and np.isfinite(selected).all() else math.nan)


def _causal_rate(times: np.ndarray, values: np.ndarray, index: int,
                 window_s: float) -> float:
    """Causal finite-difference rate over a nearby, fully elapsed window."""
    target = float(times[index]) - window_s
    prior = int(np.searchsorted(times[:index], target, side="right") - 1)
    if prior < 0:
        return math.nan
    elapsed = float(times[index] - times[prior])
    if not 0.6 * window_s <= elapsed <= 1.4 * window_s:
        return math.nan
    if not np.isfinite(values[[prior, index]]).all():
        return math.nan
    return float((values[index] - values[prior]) / elapsed)


def _state_for_sample(sample: body.MotionSample,
                      estimated_states: dict[int, np.ndarray] | None) -> np.ndarray:
    if estimated_states is None:
        return sample.state
    state = estimated_states.get(sample.source_stamp_ns)
    if state is not None:
        return state
    return np.full(3, math.nan, dtype=float)


def _transition_set(capture: body.Capture,
                    estimated_states: dict[int, np.ndarray] | None = None
                    ) -> TransitionSet:
    names = tuple(dict.fromkeys(name for level in FEATURE_LEVELS.values()
                                for name in level))
    columns: dict[str, list[float]] = {name: [] for name in names}
    targets: list[np.ndarray] = []
    truth_states: list[np.ndarray] = []
    next_steps: list[float] = []
    phases: list[str] = []

    for sequence_index, sequence in enumerate(capture.sequences):
        if len(sequence) < 3:
            continue
        times = np.asarray([sample.time_s for sample in sequence], dtype=float)
        states = np.asarray([_state_for_sample(sample, estimated_states)
                             for sample in sequence], dtype=float)
        actuators = np.asarray([sample.actuators for sample in sequence], dtype=float)
        imu_acceleration = np.asarray([
            (sample.imu_acceleration_mps2
             if sample.imu_acceleration_mps2 is not None else (math.nan, math.nan))
            for sample in sequence], dtype=float)
        q_history = states[:, 0] * np.tan(actuators[:, 0])
        rear_slip_history = np.full((len(sequence), 2), math.nan, dtype=float)
        for history_index, sample in enumerate(sequence):
            if (sample.rear_wheel_surface_mps is None
                    or not np.isfinite(states[history_index]).all()):
                continue
            raw = body._raw_features(
                states[history_index], sample.actuators,
                sample.rear_wheel_surface_mps)
            rear_slip_history[history_index] = raw[7:9]
        label = (capture.sequence_labels[sequence_index]
                 if sequence_index < len(capture.sequence_labels) else "")

        # Stride by two (nominally 20 Hz) to reduce adjacent-sample dependence.
        for index in range(1, len(sequence) - 1, 2):
            previous, current, following = sequence[index - 1:index + 2]
            if current.time_s < 0.0:
                continue
            span = following.time_s - previous.time_s
            forward_dt = following.time_s - current.time_s
            if (not 2.0 * MIN_DT_S <= span <= 2.0 * MAX_DT_S
                    or not MIN_DT_S <= forward_dt <= MAX_DT_S):
                continue

            feature_state = states[index]
            u, v, yaw_rate = feature_state
            steer, throttle = current.actuators[:2]
            rear_slips = rear_slip_history[index]
            slip_rates = np.asarray((
                _causal_rate(times, rear_slip_history[:, 0], index, 0.100),
                _causal_rate(times, rear_slip_history[:, 1], index, 0.100),
            ), dtype=float)
            spin_accel_residual = slip_rates - imu_acceleration[index, 0]

            prior = sequence[index - 1]
            rate_dt = current.time_s - prior.time_s
            steering_rate = ((steer - prior.actuators[0]) / rate_dt
                             if rate_dt > 0.0 else math.nan)
            throttle_rate = ((current.actuators[1] - prior.actuators[1]) / rate_dt
                             if rate_dt > 0.0 else math.nan)
            steering_command = float(np.float32(
                current.actuators[3] * STEERING_LIMIT_RAD))
            throttle_command = float(current.actuators[2])

            feature_row = {
                "u_mps": float(u),
                "steering_rad": float(steer),
                "v_rear_mps": float(v),
                "yaw_rate_rps": float(yaw_rate),
                "throttle_feedback": float(throttle),
                "rear_left_slip_velocity_mps": rear_slips[0],
                "rear_right_slip_velocity_mps": rear_slips[1],
                "steering_rate_radps": float(steering_rate),
                "throttle_rate_per_s": float(throttle_rate),
                "steering_command_rad": steering_command,
                "throttle_command": throttle_command,
                "q_mean_100ms": _trailing_mean(times, q_history, index, 0.100),
                "q_mean_250ms": _trailing_mean(times, q_history, index, 0.250),
                "q_mean_500ms": _trailing_mean(times, q_history, index, 0.500),
                "throttle_mean_100ms": _trailing_mean(
                    times, actuators[:, 1], index, 0.100),
                "throttle_mean_250ms": _trailing_mean(
                    times, actuators[:, 1], index, 0.250),
                "throttle_mean_500ms": _trailing_mean(
                    times, actuators[:, 1], index, 0.500),
                "q_abs_peak_500ms": (
                    float(np.max(np.abs(q_history[
                        max(0, int(np.searchsorted(
                            times, current.time_s - 0.500, side="left"))):index + 1])))
                    if current.time_s - times[0] >= 0.488 else math.nan),
                "imu_ax_mps2": float(imu_acceleration[index, 0]),
                "imu_ay_mps2": float(imu_acceleration[index, 1]),
                "imu_ax_mean_100ms": _trailing_mean(
                    times, imu_acceleration[:, 0], index, 0.100),
                "imu_ax_mean_250ms": _trailing_mean(
                    times, imu_acceleration[:, 0], index, 0.250),
                "imu_ay_mean_100ms": _trailing_mean(
                    times, imu_acceleration[:, 1], index, 0.100),
                "rear_left_slip_mean_250ms": _trailing_mean(
                    times, rear_slip_history[:, 0], index, 0.250),
                "rear_right_slip_mean_250ms": _trailing_mean(
                    times, rear_slip_history[:, 1], index, 0.250),
                "rear_left_slip_rate_mps2": float(slip_rates[0]),
                "rear_right_slip_rate_mps2": float(slip_rates[1]),
                "rear_spin_accel_residual_mean_mps2": float(
                    np.mean(spin_accel_residual)),
                "rear_spin_accel_residual_magnitude_mps2": float(
                    np.mean(np.abs(spin_accel_residual))),
                "rear_spin_accel_residual_asymmetry_mps2": float(
                    spin_accel_residual[0] - spin_accel_residual[1]),
            }
            for name in names:
                columns[name].append(feature_row[name])
            targets.append((following.state - previous.state) / span)
            truth_states.append(current.state)
            next_steps.append(forward_dt)
            phases.append(label)

    return TransitionSet(
        values={name: np.asarray(values, dtype=float)
                for name, values in columns.items()},
        target=np.asarray(targets, dtype=float).reshape((-1, 3)),
        truth_state=np.asarray(truth_states, dtype=float).reshape((-1, 3)),
        next_dt_s=np.asarray(next_steps, dtype=float),
        phase=np.asarray(phases, dtype=str),
    )


def _observer_error_summary(capture: body.Capture,
                            estimated_states: dict[int, np.ndarray]
                            ) -> dict[str, Any]:
    rows = [sample for sequence in capture.sequences for sample in sequence
            if sample.time_s >= 0.0
            and sample.source_stamp_ns in estimated_states]
    if not rows:
        return {"all": {"count": 0}}
    truth = np.asarray([sample.state for sample in rows], dtype=float)
    estimate = np.asarray([estimated_states[sample.source_stamp_ns]
                           for sample in rows], dtype=float)
    steering = np.asarray([sample.actuators[0] for sample in rows], dtype=float)
    cohorts = {
        "all": np.ones(len(rows), dtype=bool),
        "abs_steering_lt_0p15": np.abs(steering) < 0.15,
        "abs_steering_0p15_to_0p30": (
            (np.abs(steering) >= 0.15) & (np.abs(steering) < 0.30)),
        "abs_steering_gte_0p30": np.abs(steering) >= 0.30,
        "truth_u_negative": truth[:, 0] < 0.0,
        "abs_truth_yaw_rate_gte_2": np.abs(truth[:, 2]) >= 2.0,
    }
    summaries: dict[str, Any] = {}
    for name, mask in cohorts.items():
        if not np.any(mask):
            continue
        error = estimate[mask] - truth[mask]
        summaries[name] = {
            "count": int(np.count_nonzero(mask)),
            "u_bias_mps": float(np.mean(error[:, 0])),
            "u_rmse_mps": float(np.sqrt(np.mean(error[:, 0] ** 2))),
            "u_abs_p95_mps": float(np.quantile(np.abs(error[:, 0]), 0.95)),
            "v_rmse_mps": float(np.sqrt(np.mean(error[:, 1] ** 2))),
            "yaw_rate_rmse_rps": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        }
    return summaries


def _robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(matrix, axis=0)
    scale = np.quantile(matrix, 0.75, axis=0) - np.quantile(matrix, 0.25, axis=0)
    fallback = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return center, scale


def _calibration_distances(training: list[tuple[str, np.ndarray]],
                           center: np.ndarray, scale: np.ndarray,
                           max_queries: int = 12000) -> np.ndarray:
    """Run-held-out nearest distances within the training corpus."""
    distances: list[np.ndarray] = []
    for run_index, (_, own) in enumerate(training):
        other_runs = [points for index, (_, points) in enumerate(training)
                      if index != run_index and len(points)]
        if not len(own) or not other_runs:
            continue
        other = np.concatenate(other_runs, axis=0)
        if len(own) > max_queries:
            selected = np.linspace(0, len(own) - 1, max_queries, dtype=int)
            own = own[selected]
        tree = cKDTree((other - center) / scale)
        query_dist, _ = tree.query((own - center) / scale, k=1, workers=-1)
        distances.append(np.asarray(query_dist, dtype=float))
    if not distances:
        raise ValueError("support calibration requires at least two training runs")
    return np.concatenate(distances)


def _score(train_sets: list[tuple[str, TransitionSet]],
           test_name: str, test_set: TransitionSet,
           feature_names: tuple[str, ...], neighbors: int,
           common_rows: bool) -> dict[str, Any]:
    train_matrices: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    train_truth_states: list[np.ndarray] = []
    per_run: list[tuple[str, np.ndarray]] = []
    for run_name, dataset in train_sets:
        matrix, mask = dataset.select(feature_names)
        target = dataset.target[mask]
        truth_state = dataset.truth_state[mask]
        if common_rows:
            common_matrix, common_mask = dataset.select(FULL_REQUIRED)
            common_names = feature_names
            lookup = {name: i for i, name in enumerate(FULL_REQUIRED)}
            matrix = common_matrix[:, [lookup[name] for name in common_names]]
            target = dataset.target[common_mask]
            truth_state = dataset.truth_state[common_mask]
        if len(matrix) == 0:
            raise ValueError(f"no finite training rows for {run_name} / {feature_names}")
        train_matrices.append(matrix)
        train_targets.append(target)
        train_truth_states.append(truth_state)
        per_run.append((run_name, matrix))

    test_matrix, test_mask = test_set.select(feature_names)
    if common_rows:
        full_matrix, full_mask = test_set.select(FULL_REQUIRED)
        lookup = {name: i for i, name in enumerate(FULL_REQUIRED)}
        test_matrix = full_matrix[:, [lookup[name] for name in feature_names]]
        test_mask = full_mask
        test_target = test_set.target[test_mask]
        test_truth_state = test_set.truth_state[test_mask]
        test_observer_state = np.column_stack((
            test_set.values["u_mps"][test_mask],
            test_set.values["v_rear_mps"][test_mask],
            test_set.values["yaw_rate_rps"][test_mask]))
    test_target = test_set.target[test_mask]
    test_truth_state = test_set.truth_state[test_mask]
    test_observer_state = np.column_stack((
        test_set.values["u_mps"][test_mask],
        test_set.values["v_rear_mps"][test_mask],
        test_set.values["yaw_rate_rps"][test_mask]))
    test_dt = test_set.next_dt_s[test_mask]
    test_steer = test_set.values["steering_rad"][test_mask]
    test_speed = np.hypot(test_set.values["u_mps"][test_mask],
                          test_set.values["v_rear_mps"][test_mask])
    if not len(test_matrix):
        raise ValueError(f"no finite test rows for {test_name} / {feature_names}")

    x_train = np.concatenate(train_matrices, axis=0)
    y_train = np.concatenate(train_targets, axis=0)
    y_train_state = np.concatenate(train_truth_states, axis=0)
    center, scale = _robust_scale(x_train)
    x_train_scaled = (x_train - center) / scale
    x_test_scaled = (test_matrix - center) / scale
    tree = cKDTree(x_train_scaled)
    k = min(neighbors, len(x_train_scaled))
    distances, indices = tree.query(x_test_scaled, k=k, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-6) ** 2
    weights /= np.sum(weights, axis=1, keepdims=True)
    neighbor_targets = y_train[indices]
    neighbor_state_targets = y_train_state[indices]
    predicted = np.sum(neighbor_targets * weights[:, :, None], axis=1)
    predicted_state = np.sum(
        neighbor_state_targets * weights[:, :, None], axis=1)
    target_spread = np.sqrt(np.sum(
        weights[:, :, None] * (neighbor_targets - predicted[:, None, :]) ** 2,
        axis=1))

    calibration = _calibration_distances(per_run, center, scale)
    support_radius = float(np.quantile(calibration, 0.95))
    nearest = distances[:, 0]
    within_radius = distances <= support_radius
    prediction_error = predicted - test_target
    next_state_error = prediction_error * test_dt[:, None]
    state_correction_error = predicted_state - test_truth_state
    observer_state_error = test_observer_state - test_truth_state

    def metrics(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))
        if not count:
            return {"count": 0}
        return {
            "count": count,
            "derivative_rmse": {
                name: float(np.sqrt(np.mean(prediction_error[mask, axis] ** 2)))
                for axis, name in enumerate(TARGET_NAMES)
            },
            "next_interval_state_increment_rmse": {
                name: float(np.sqrt(np.mean(next_state_error[mask, axis] ** 2)))
                for axis, name in enumerate(("u_mps", "v_rear_mps", "yaw_rate_rps"))
            },
            "yaw_acceleration_neighbor_spread_median_rps2": float(
                np.median(target_spread[mask, 2])),
            "yaw_acceleration_neighbor_spread_p95_rps2": float(
                np.quantile(target_spread[mask, 2], 0.95)),
        }

    def state_metrics(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))
        if not count:
            return {"count": 0}
        axes = ("u_mps", "v_rear_mps", "yaw_rate_rps")
        return {
            "count": count,
            "observer_rmse": {
                name: float(np.sqrt(np.mean(observer_state_error[mask, axis] ** 2)))
                for axis, name in enumerate(axes)
            },
            "knn_corrected_rmse": {
                name: float(np.sqrt(np.mean(state_correction_error[mask, axis] ** 2)))
                for axis, name in enumerate(axes)
            },
            "knn_corrected_bias": {
                name: float(np.mean(state_correction_error[mask, axis]))
                for axis, name in enumerate(axes)
            },
            "knn_corrected_abs_error_p95": {
                name: float(np.quantile(
                    np.abs(state_correction_error[mask, axis]), 0.95))
                for axis, name in enumerate(axes)
            },
        }

    supported = nearest <= support_radius
    support_neighbor_count = np.sum(within_radius, axis=1)
    return {
        "test_run": test_name,
        "feature_names": list(feature_names),
        "cohort": "common_M5_complete_rows" if common_rows else "native_finite_rows",
        "training_rows": int(len(x_train)),
        "test_rows": int(len(test_matrix)),
        "support": {
            "calibration": "p95 cross-run nearest-neighbor distance within training runs",
            "radius_robust_scale_units": support_radius,
            "fraction_supported": float(np.mean(supported)),
            "nearest_distance_p50_p95": [
                float(np.quantile(nearest, 0.50)), float(np.quantile(nearest, 0.95))],
            "neighbors_within_calibrated_radius_p50": float(
                np.median(support_neighbor_count)),
            "fraction_with_at_least_8_local_neighbors": float(
                np.mean(support_neighbor_count >= 8)),
        },
        "all_rows": metrics(np.ones(len(test_matrix), dtype=bool)),
        "supported_rows": metrics(supported),
        "state_estimation_all_rows": state_metrics(
            np.ones(len(test_matrix), dtype=bool)),
        "state_estimation_supported_rows": state_metrics(supported),
        "high_steering_supported_rows": metrics(supported & (np.abs(test_steer) >= 0.30)),
        "speed_steering_slices": _slice_metrics(
            prediction_error, next_state_error, target_spread, supported,
            test_speed, test_steer),
    }


def _slice_metrics(error: np.ndarray, interval_error: np.ndarray,
                   spread: np.ndarray, supported: np.ndarray,
                   speed: np.ndarray, steering: np.ndarray) -> list[dict[str, Any]]:
    slices = []
    speed_bands = ((0.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0),
                   (6.0, 7.0), (7.0, math.inf))
    steer_bands = ((0.0, 0.15), (0.15, 0.30), (0.30, math.inf))
    for speed_low, speed_high in speed_bands:
        for steer_low, steer_high in steer_bands:
            mask = (supported & (speed >= speed_low) & (speed < speed_high)
                    & (np.abs(steering) >= steer_low)
                    & (np.abs(steering) < steer_high))
            count = int(np.count_nonzero(mask))
            if count < 20:
                continue
            slices.append({
                "speed_mps": [speed_low, speed_high if math.isfinite(speed_high) else None],
                "abs_steering_rad": [steer_low, steer_high
                                     if math.isfinite(steer_high) else None],
                "count": count,
                "u_v_r_next_interval_rmse": [
                    float(np.sqrt(np.mean(interval_error[mask, axis] ** 2)))
                    for axis in range(3)],
                "yaw_acceleration_neighbor_spread_median_rps2": float(
                    np.median(spread[mask, 2])),
            })
    return slices


def _capture(path: Path, role: str,
             estimated_state_path: Path | None = None
             ) -> tuple[str, TransitionSet, body.Capture, dict[str, Any] | None]:
    capture = body.load_capture(path)
    body._validate_capture(capture, role)
    estimated_states = (body._read_replayed_states(estimated_state_path)
                        if estimated_state_path is not None else None)
    dataset = _transition_set(capture, estimated_states)
    if estimated_states is not None:
        coverage = float(np.mean(dataset.select(FULL_REQUIRED)[1]))
        if coverage < 0.90:
            raise ValueError(
                f"{role} observer replay state coverage is only {coverage:.1%}; "
                "source bag may not match")
    state_error = (_observer_error_summary(capture, estimated_states)
                   if estimated_states is not None else None)
    return path.resolve().as_posix(), dataset, capture, state_error


def analyze(training_paths: list[Path], test_paths: list[Path],
            neighbors: int, train_state_paths: list[Path],
            test_state_paths: list[Path], oracle_state: bool) -> dict[str, Any]:
    if len(training_paths) < 2:
        raise ValueError("at least two independent --train-bag runs are required")
    if not test_paths:
        raise ValueError("at least one --test-bag whole-run holdout is required")
    resolved_train = {path.resolve() for path in training_paths}
    resolved_test = [path.resolve() for path in test_paths]
    if len(resolved_train) != len(training_paths):
        raise ValueError("training bags must be distinct whole runs")
    if any(path in resolved_train for path in resolved_test):
        raise ValueError("a whole-run test bag cannot also be a training bag")
    if neighbors < 2:
        raise ValueError("neighbors must be at least two")
    if oracle_state:
        if train_state_paths or test_state_paths:
            raise ValueError("--oracle-state cannot be combined with observer state bags")
    elif (len(train_state_paths) != len(training_paths)
          or len(test_state_paths) != len(test_paths)):
        raise ValueError(
            "provide one --train-state-bag and one --test-state-bag per source/test bag; "
            "use --oracle-state only for an explicit truth-state diagnostic")
    state_paths = [*train_state_paths, *test_state_paths]
    if len({path.resolve() for path in state_paths}) != len(state_paths):
        raise ValueError("observer state bags must be distinct whole-run replays")

    train_records = [
        _capture(path, f"training run {index + 1}",
                 None if oracle_state else train_state_paths[index])
        for index, path in enumerate(training_paths)]
    test_records = [
        _capture(path, f"test run {index + 1}",
                 None if oracle_state else test_state_paths[index])
        for index, path in enumerate(test_paths)]
    train_sets = [(name, data) for name, data, _, _ in train_records]

    results = []
    full_lookup = {name: i for i, name in enumerate(FULL_REQUIRED)}
    for test_name, test_data, test_capture, _ in test_records:
        ladder = []
        for level_name, feature_names in FEATURE_LEVELS.items():
            common_possible = all(name in full_lookup for name in feature_names)
            cohorts = {}
            cohorts["native"] = _score(
                train_sets, test_name, test_data, feature_names,
                neighbors, common_rows=False)
            if common_possible:
                cohorts["common_M5_complete_rows"] = _score(
                    train_sets, test_name, test_data, feature_names,
                    neighbors, common_rows=True)
            ladder.append({"level": level_name, "cohorts": cohorts})
        results.append({
            "test_bag": test_name,
            "has_phase_markers": test_capture.has_phase_markers,
            "final_lap_count": test_capture.final_lap_count,
            "valid_phases": test_capture.valid_phase_count,
            "unscored_phases": test_capture.unscored_phase_count,
            "ladder": ladder,
        })

    coverage = []
    for role, records in (("training", train_records), ("test", test_records)):
        for name, dataset, capture, state_error in records:
            total = len(dataset.target)
            complete = dataset.select(FULL_REQUIRED)[1]
            domain = capture.domain_samples
            coverage.append({
                "role": role,
                "bag": name,
                "has_phase_markers": capture.has_phase_markers,
                "final_lap_count": capture.final_lap_count,
                "transition_rows": total,
                "M5_complete_rows": int(np.count_nonzero(complete)),
                "M5_complete_fraction": float(np.mean(complete)) if total else 0.0,
                "speed_mps_p01_p50_p99": [
                    float(x) for x in np.percentile(domain[:, 0], (1, 50, 99))],
                "steering_rad_p01_p50_p99": [
                    float(x) for x in np.percentile(domain[:, 1], (1, 50, 99))],
                "max_abs_steering_rad": float(np.max(np.abs(domain[:, 1]))),
                "collision_count_start_end": [
                    capture.collision_count_start, capture.collision_count_end],
                "timing_faults": capture.timing_faults,
                "observer_vs_truth_state_error": state_error,
                "stream_stats_hz_p95gap_maxgap_ms": {
                    topic: [float(x) for x in values]
                    for topic, values in capture.stream_stats.items()
                },
            })

    return {
        "schema_version": 1,
        "method": "whole-run conditional one-step transition KNN; diagnostic only",
        "state_feature_source": (
            "simulator truth /autodrive/roboracer_1/odom (oracle diagnostic)"
            if oracle_state else
            "exact production sensor_odometry_node replay from encoders and IMU"),
        "target_source": (
            "simulator truth /autodrive/roboracer_1/odom; training labels only"),
        "uses_ground_truth_state_features": oracle_state,
        "reads_ips_topic": False,
        "reads_transform_topics": False,
        "M3_wheel_features": (
            "100 ms causal rear encoder surface-speed minus body-kinematic velocity; "
            "a slip-velocity proxy, not measured tire slip or force"),
        "M4_M5_causality": (
            "actuator feedback, current commands, past rates, and trailing histories "
            "at or before the prediction sample only"),
        "M6_traction_history": (
            "causal body-frame IMU acceleration, 100/250 ms acceleration means, "
            "250 ms rear slip-proxy means, and 100 ms slip-proxy rates"),
        "M7_spin_accel_residual": (
            "rear slip-proxy rates minus current longitudinal IMU acceleration; "
            "a kinematic residual, not a wheel-force measurement"),
        "target": "central approximately 50 ms body-twist derivative from recorded odometry",
        "state_correction_target": (
            "current simulator-truth body state; diagnostic only, not a production estimate"),
        "history_windows_ms": [100, 250, 500],
        "neighbors": neighbors,
        "training_bags": [name for name, _, _, _ in train_records],
        "observer_state_bags": ([str(path.resolve()) for path in state_paths]
                                 if not oracle_state else []),
        "coverage": coverage,
        "whole_run_results": results,
        "limitations": [
            "Neighbor predictions are not differentiable MPC dynamics and are not recursively validated.",
            "The support radius is calibrated from cross-run nearest distances with the supplied training runs.",
            "M0-M5 comparisons are observational and do not isolate physical tire forces.",
            "Per-wheel normal loads, contact forces, and suspension pose are unavailable in these bags.",
            "The truth-state mode is diagnostic only; use observer state bags for legal-input ablations.",
            "Simulator truth is used only for transition targets in observer-state mode.",
            "Increment RMSE excludes current observer-versus-truth state error; see observer_vs_truth_state_error.",
            "KNN state-correction scores are supervised by truth and are not a deployable estimator; they test causal feature observability only.",
            "M6/M7 condition on measured causal IMU and encoder histories; these are not recursively simulated future inputs.",
        ],
    }


def _print_summary(report: dict[str, Any]) -> None:
    mode = ("oracle truth-state" if report["uses_ground_truth_state_features"]
            else "production sensor-odometry state")
    print(f"State-sufficiency diagnostic ({mode}; conditional one-step KNN, not MPC)")
    for item in report["coverage"]:
        print(f"  {item['role']}: {Path(item['bag']).parent.parent.name}: "
              f"{item['transition_rows']} transitions, "
              f"M5-complete={item['M5_complete_rows']} "
              f"({item['M5_complete_fraction']:.1%})")
        state_error = item.get("observer_vs_truth_state_error")
        if state_error and state_error.get("all", {}).get("count", 0):
            all_state = state_error["all"]
            print(f"    observer/truth u RMSE={all_state['u_rmse_mps']:.3f} m/s "
                  f"(bias {all_state['u_bias_mps']:+.3f}), "
                  f"v RMSE={all_state['v_rmse_mps']:.3f} m/s; "
                  "yaw-rate RMSE="
                  f"{all_state['yaw_rate_rmse_rps']:.3f} rad/s")
    for test in report["whole_run_results"]:
        print(f"\nWhole-run holdout: {Path(test['test_bag']).parent.parent.name}")
        print(" level                    n     support    u/v/r increment RMSE (m/s, m/s, rad/s)")
        for row in test["ladder"]:
            scores = row["cohorts"].get("common_M5_complete_rows")
            if scores is None:
                scores = row["cohorts"]["native"]
            metric = scores["supported_rows"]
            rmse = metric.get("next_interval_state_increment_rmse", {})
            values = (rmse.get("u_mps", math.nan), rmse.get("v_rear_mps", math.nan),
                      rmse.get("yaw_rate_rps", math.nan))
            print(f" {row['level']:<24} {scores['test_rows']:>6} "
                  f"{scores['support']['fraction_supported']:>8.1%}   "
                  f"{values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f}")
        print(" state correction (supported rows): level / observer u-v-r RMSE -> "
              "KNN-corrected u-v-r RMSE")
        for row in test["ladder"]:
            scores = row["cohorts"].get("common_M5_complete_rows")
            if scores is None:
                scores = row["cohorts"]["native"]
            state = scores["state_estimation_supported_rows"]
            observer = state["observer_rmse"]
            corrected = state["knn_corrected_rmse"]
            before = tuple(observer[name] for name in
                           ("u_mps", "v_rear_mps", "yaw_rate_rps"))
            after = tuple(corrected[name] for name in
                          ("u_mps", "v_rear_mps", "yaw_rate_rps"))
            print(f"  {row['level']:<24} "
                  f"{before[0]:.3f}/{before[1]:.3f}/{before[2]:.3f} -> "
                  f"{after[0]:.3f}/{after[1]:.3f}/{after[2]:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bag", action="append", type=Path, default=[],
                        help="clean independent training bag; repeat at least twice")
    parser.add_argument("--test-bag", action="append", type=Path, default=[],
                        help="clean whole-run holdout; may be repeated")
    parser.add_argument("--train-state-bag", action="append", type=Path, default=[],
                        help="matching closed sensor-odometry replay for each train bag")
    parser.add_argument("--test-state-bag", action="append", type=Path, default=[],
                        help="matching closed sensor-odometry replay for each test bag")
    parser.add_argument("--oracle-state", action="store_true",
                        help="explicitly use simulator-truth state as the feature ablation")
    parser.add_argument("--neighbors", type=int, default=NEIGHBOR_COUNT,
                        help="nearest training transitions per conditional prediction")
    parser.add_argument("--output-json", type=Path,
                        help="optional report path; no dataset files are written")
    args = parser.parse_args()
    try:
        report = analyze(args.train_bag, args.test_bag, args.neighbors,
                         args.train_state_bag, args.test_state_bag,
                         args.oracle_state)
    except (ValueError, sqlite3.Error, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"state-sufficiency analysis failed: {exc}\n")
    _print_summary(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
