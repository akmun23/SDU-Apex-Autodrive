#!/usr/bin/env python3
"""Recursively test a train-only lateral-velocity increment model in MPC."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.racing.offline.controller.compare_mpc_odom_initialization import (
    MpcStageResult,
    _workspace_path,
)
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.racing.offline.controller.score_production_mpc_model import (
    DT_S,
    HORIZONS_S,
    MpcModelControl,
    MpcModelState,
    _project_run,
    _read_commands,
    _recorded_command_arrays,
)
from tools.vehicle_dynamics_learning.build_practice_transfer_benchmark import (
    _run_rows,
)
from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import COM_X_M
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "odom_wheel_signal_correction_20261011/"
    "mpc_prediction_analysis_20261012")
BENCHMARK = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "practice_transfer_benchmark_v1.json")
BASELINE_STEPWISE = ANALYSIS_DIR / "paired_odom_initialization_stepwise_report.json"
MODEL_LIBRARY = ANALYSIS_DIR / "libproduction_mpc.so"
CONFIG = ROOT / "f1tenth_mpc/config/mpc_competition.yaml"
TRAJECTORY = ROOT / (
    "f1tenth_planning/trajectories/autodrive_practice_20260924_b/"
    "autodrive_practice_20260924_b_mintime_raceline_speed_headroom_codex.csv")
BAGS = {
    "practice_model_validation_r02": ROOT /
    "live_runs/practice_model_validation_r02/run/run_0.db3",
    "practice_model_validation_r03": ROOT /
    "live_runs/practice_model_validation_r03/run/run_0.db3",
}
OUTPUT = ANALYSIS_DIR / "paired_lateral_vdot_divergence_context.json"
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
CANDIDATE_MODEL = "stable_ridge_alpha_100"
SUPPORTED_CANDIDATE_MODEL = "stable_ridge_alpha_100_supported"
KNN_SUPPORTED_CANDIDATE_MODEL = "stable_ridge_alpha_100_knn_supported"
STATE_NAMES = (
    "u_com_mps", "v_com_mps", "v_rear_mps", "yaw_rate_rps",
    "e_y_m", "e_psi_rad", "actual_steering_rad",
    "target_speed_mps", "steering_command_rad",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _vdot_features(state: MpcModelState) -> np.ndarray:
    u, v, r, delta = (float(state.u), float(state.v), float(state.r),
                      float(state.actual_steering_angle))
    return np.asarray((
        u, v, r, delta, u * r, u * delta,
        u * delta * abs(delta), v * abs(r), r * abs(r),
    ), dtype=np.float64)


def _predict_vdot(state: MpcModelState, model: dict[str, Any]) -> float:
    features = _vdot_features(state)
    normalized = ((features - np.asarray(model["feature_mean"], dtype=np.float64))
                  / np.asarray(model["feature_scale"], dtype=np.float64))
    value = float(np.dot(np.asarray(model["coef"], dtype=np.float64), normalized)
                  + float(model["intercept"]))
    if not math.isfinite(value):
        raise ValueError("lateral-velocity model produced a non-finite derivative")
    return value


def _smoothstep01(value: float) -> float:
    clipped = min(1.0, max(0.0, value))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _support_weight(state: MpcModelState,
                    model: dict[str, Any]) -> float:
    speed_min, speed_max = model["support_u_bounds_mps"]
    steer_min, steer_max = model["support_steering_bounds_rad"]
    speed = float(state.u)
    steering = float(state.actual_steering_angle)
    speed_margin = float(model["support_speed_margin_mps"])
    steer_margin = float(model["support_steering_margin_rad"])
    speed_weight = (
        _smoothstep01((speed - (speed_min - speed_margin)) / speed_margin)
        * _smoothstep01(((speed_max + speed_margin) - speed) / speed_margin))
    steering_weight = (
        _smoothstep01((steering - (steer_min - steer_margin)) / steer_margin)
        * _smoothstep01(((steer_max + steer_margin) - steering) / steer_margin))
    return speed_weight * steering_weight


def _knn_support_weight(state: MpcModelState,
                        model: dict[str, Any], support: cKDTree,
                        full_threshold: float,
                        zero_threshold: float) -> float:
    features = _vdot_features(state)
    normalized = ((features - np.asarray(model["feature_mean"], dtype=np.float64))
                  / np.asarray(model["feature_scale"], dtype=np.float64))
    distance = float(support.query(normalized, k=1)[0])
    if distance <= full_threshold:
        return 1.0
    if distance >= zero_threshold:
        return 0.0
    return 1.0 - _smoothstep01(
        (distance - full_threshold) / (zero_threshold - full_threshold))


def _candidate_geometry(state: MpcModelState, following: MpcModelState,
                        dt_s: float, curvature: float) -> tuple[float, float, float]:
    u0 = float(state.u)
    u1 = float(following.u)
    u_mid = 0.5 * (u0 + u1)
    r0, r1 = float(state.r), float(following.r)
    r_mid = 0.5 * (r0 + r1)
    v0, v1 = float(state.v), float(following.v)

    denominator0 = 1.0 - curvature * float(state.e_y)
    if abs(denominator0) < 0.05:
        raise ValueError("candidate reached invalid initial path projection")
    s_dot0 = (u0 * math.cos(float(state.e_psi))
              - v0 * math.sin(float(state.e_psi))) / denominator0
    e_y_dot0 = (u0 * math.sin(float(state.e_psi))
                + v0 * math.cos(float(state.e_psi)))
    e_psi_dot0 = r0 - curvature * s_dot0
    e_y_mid = float(state.e_y) + 0.5 * dt_s * e_y_dot0
    e_psi_mid = float(state.e_psi) + 0.5 * dt_s * e_psi_dot0

    denominator_mid = 1.0 - curvature * e_y_mid
    if abs(denominator_mid) < 0.05:
        raise ValueError("candidate reached invalid midpoint path projection")
    v_mid = 0.5 * (v0 + v1)
    s_dot_mid = (u_mid * math.cos(e_psi_mid)
                 - v_mid * math.sin(e_psi_mid)) / denominator_mid
    e_y_dot_mid = (u_mid * math.sin(e_psi_mid)
                   + v_mid * math.cos(e_psi_mid))
    e_psi_dot_mid = r_mid - curvature * s_dot_mid
    return (float(state.e_y) + dt_s * e_y_dot_mid,
            _wrap(float(state.e_psi) + dt_s * e_psi_dot_mid),
            dt_s * s_dot_mid)


def _rollout(library: ctypes.CDLL, mode: str, initial: MpcModelState,
             progress: float, local_index: int, steering: np.ndarray,
             target_speed: np.ndarray, trajectory: np.ndarray,
             lap_length: float, truth: np.ndarray, model: dict[str, Any],
             max_steps: int | None = None,
             debug_trace: list[dict[str, Any]] | None = None,
             knn_support: tuple[cKDTree, float, float] | None = None
             ) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    state = initial
    predictions: list[tuple[float, ...]] = []
    if max_steps is None:
        max_steps = round(max(HORIZONS_S) / DT_S)
    for step in range(max_steps):
        command_index = local_index + step
        control = MpcModelControl(
            float((steering[command_index + 1] - steering[command_index]) / DT_S),
            float((target_speed[command_index + 1]
                   - target_speed[command_index]) / DT_S))
        curvature = float(np.interp(
            progress % lap_length, trajectory[:, 0], trajectory[:, 4]))
        stage = library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(control),
            ctypes.c_float(DT_S), ctypes.c_float(curvature))
        if not stage.valid or not math.isfinite(stage.delta_s_m):
            return np.asarray(predictions, dtype=np.float64), {
                "failed_step": step + 1,
                "reason": "production_motion_model_invalid",
                "state_v_rear_mps": float(state.v),
                "state_u_mps": float(state.u),
                "state_yaw_rate_rps": float(state.r),
                "state_e_y_m": float(state.e_y),
                "state_e_psi_rad": float(state.e_psi),
                "state_actual_steering_rad": float(
                    state.actual_steering_angle),
                "path_curvature_radpm": curvature,
                "progress_m": float(progress),
                "branch_flags": int(stage.branch_flags),
            }
        following = stage.next
        if mode in ("direct_vdot_candidate", "support_gated_vdot_candidate",
                    "knn_supported_vdot_candidate"):
            try:
                if mode == "support_gated_vdot_candidate":
                    support_weight = _support_weight(state, model)
                elif mode == "knn_supported_vdot_candidate":
                    if knn_support is None:
                        raise ValueError("kNN support gate was not initialized")
                    support_tree, full_threshold, zero_threshold = knn_support
                    support_weight = _knn_support_weight(
                        state, model, support_tree, full_threshold,
                        zero_threshold)
                else:
                    support_weight = 1.0
                following.v = (float(state.v) + DT_S * support_weight
                               * _predict_vdot(state, model))
                following.e_y, following.e_psi, stage.delta_s_m = _candidate_geometry(
                    state, following, DT_S, curvature)
            except ValueError:
                return np.asarray(predictions, dtype=np.float64), {
                    "failed_step": step + 1,
                    "reason": "candidate_path_geometry_invalid",
                    "state_v_rear_mps": float(state.v),
                    "next_v_rear_mps": float(following.v),
                    "state_u_mps": float(state.u),
                    "state_yaw_rate_rps": float(state.r),
                    "state_e_y_m": float(state.e_y),
                    "state_e_psi_rad": float(state.e_psi),
                    "state_actual_steering_rad": float(
                        state.actual_steering_angle),
                    "path_curvature_radpm": curvature,
                    "progress_m": float(progress),
                }
        state = following
        progress += float(stage.delta_s_m)
        actual = (
            float(state.u), float(state.v + COM_X_M * state.r),
            float(state.v), float(state.r), float(state.e_y),
            float(state.e_psi), float(state.actual_steering_angle),
            float(state.target_speed), float(state.steering_command))
        predicted = np.asarray(actual)
        expected = truth[command_index + 1]
        error = predicted - expected
        error[5] = _wrap(float(error[5]))
        if debug_trace is not None:
            debug_trace.append({
                "step": step + 1,
                "elapsed_s": float((step + 1) * DT_S),
                "u_pred_mps": float(predicted[0]),
                "u_truth_mps": float(expected[0]),
                "u_error_mps": float(error[0]),
                "v_rear_pred_mps": float(predicted[2]),
                "v_rear_truth_mps": float(expected[2]),
                "v_rear_error_mps": float(error[2]),
                "r_pred_radps": float(predicted[3]),
                "r_truth_radps": float(expected[3]),
                "r_error_radps": float(error[3]),
                "e_psi_pred_rad": float(predicted[5]),
                "e_psi_truth_rad": float(expected[5]),
                "e_psi_error_rad": float(error[5]),
                "e_y_pred_m": float(predicted[4]),
                "e_y_truth_m": float(expected[4]),
                "e_y_error_m": float(error[4]),
                "steering_pred_rad": float(predicted[6]),
                "steering_truth_rad": float(expected[6]),
                "progress_m": float(progress),
                "target_speed_pred_mps": float(predicted[7]),
                "target_speed_truth_mps": float(expected[7]),
            })
        predictions.append(tuple(map(float, error)))
    return np.asarray(predictions, dtype=np.float64), None


def _training_matrix(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(dataset_path, allow_pickle=False) as archive:
        frames = archive["frames"].astype(np.float64)
        rigid = archive["simulator_rigid_state"].astype(np.float64)
        bounds = archive["sequence_bounds"].astype(np.int64)
        sequence_run = archive["sequence_run_index"].astype(np.int32)
        run_ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
        dt_values = archive["dt_s"].astype(np.float64)
        lap_count = archive["lap_count"].astype(np.int32)
    matches = np.flatnonzero(
        run_ids == "practice_filter_none_12lap_20260925")
    if len(matches) != 1 or splits[int(matches[0])] != "train":
        raise ValueError("practice teacher run is not a training capture")
    run_index = int(matches[0])
    v_rear = rigid[:, 8] - COM_X_M * rigid[:, 12]
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    lap_parts: list[np.ndarray] = []
    for (start_raw, end_raw), run_raw in zip(bounds, sequence_run):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if run != run_index or end - start < 2:
            continue
        current = np.arange(start, end - 1, dtype=np.int64)
        following = current + 1
        keep = ((lap_count[current] >= 1) & (lap_count[current] <= 8)
                & np.isclose(dt_values[following], DT_S,
                             rtol=0.0, atol=1e-7))
        current, following = current[keep], following[keep]
        if not len(current):
            continue
        u, v, r = rigid[current, 7], v_rear[current], rigid[current, 12]
        delta = frames[current, 3]
        feature_parts.append(np.column_stack((
            u, v, r, delta, u * r, u * delta,
            u * delta * np.abs(delta), v * np.abs(r), r * np.abs(r),
        )))
        target_parts.append((v_rear[following] - v_rear[current]) / DT_S)
        lap_parts.append(lap_count[current])
    if not feature_parts:
        raise ValueError("practice training capture has no eligible transitions")
    return (np.concatenate(feature_parts), np.concatenate(target_parts),
            np.concatenate(lap_parts))


def _fit_model_sweep(dataset_path: Path
                     ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    features, target, laps = _training_matrix(dataset_path)
    cross_validation: dict[str, Any] = {}
    for alpha in RIDGE_ALPHAS:
        fold_rmse = []
        stable_fold_rmse = []
        for heldout_lap in range(1, 9):
            train_mask = laps != heldout_lap
            valid_mask = laps == heldout_lap
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            estimator.fit(features[train_mask], target[train_mask])
            fold_rmse.append(_rmse(
                estimator.predict(features[valid_mask]) - target[valid_mask]))
            stable = _fit_stable_regression(
                features[train_mask], target[train_mask], alpha)
            stable_prediction = _predict_standardized(
                features[valid_mask], stable)
            stable_fold_rmse.append(_rmse(
                stable_prediction - target[valid_mask]))
        cross_validation[str(alpha)] = {
            "macro_lap_vdot_rmse_mps2": float(np.mean(fold_rmse)),
            "per_lap_vdot_rmse_mps2": fold_rmse,
            "stable_feedback_macro_lap_vdot_rmse_mps2": float(
                np.mean(stable_fold_rmse)),
            "stable_feedback_per_lap_vdot_rmse_mps2": stable_fold_rmse,
        }
    models: dict[str, dict[str, Any]] = {}
    for alpha in RIDGE_ALPHAS:
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        estimator.fit(features, target)
        scaler, ridge = estimator[0], estimator[1]
        beta_v = float(ridge.coef_[1] / scaler.scale_[1])
        models[f"ridge_alpha_{alpha:g}"] = {
            "feature_mean": scaler.mean_.tolist(),
            "feature_scale": scaler.scale_.tolist(),
            "coef": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
            "raw_vdot_coefficient_on_v_rear_per_s": beta_v,
            "discrete_v_feedback_pole_at_40hz": 1.0 + DT_S * beta_v,
        }
        models[f"stable_ridge_alpha_{alpha:g}"] = _fit_stable_regression(
            features, target, alpha)
    support_bounds = {
        "support_u_bounds_mps": [
            float(np.min(features[:, 0])), float(np.max(features[:, 0]))],
        "support_steering_bounds_rad": [
            float(np.min(features[:, 3])), float(np.max(features[:, 3]))],
        "support_speed_margin_mps": 0.5,
        "support_steering_margin_rad": 0.05,
    }
    for model in models.values():
        model.update(support_bounds)
    best_alpha = min(
        RIDGE_ALPHAS,
        key=lambda alpha: cross_validation[str(alpha)][
            "macro_lap_vdot_rmse_mps2"])
    metadata = {
        "training_run": "practice_filter_none_12lap_20260925",
        "training_laps": "1-8 only",
        "training_transition_count": int(len(target)),
        "features": ["u", "v_rear", "yaw_rate", "actual_steering",
                     "u*yaw_rate", "u*steering",
                     "u*steering*abs(steering)",
                     "v*abs(yaw_rate)", "yaw_rate*abs(yaw_rate)"],
        "held_out_lap_cross_validation": cross_validation,
        "minimum_one_step_cv_alpha": best_alpha,
        "stable_feedback_minimum_one_step_cv_alpha": min(
            RIDGE_ALPHAS,
            key=lambda alpha: cross_validation[str(alpha)][
                "stable_feedback_macro_lap_vdot_rmse_mps2"]),
        "stability_constraint": (
            "the fitted continuous-time coefficient on v_rear is constrained "
            "to [-40, 0] per second, keeping its isolated 40 Hz discrete pole "
            "within [0,1]"),
        "models": models,
    }
    return models, metadata


def _build_knn_support(dataset_path: Path,
                       model: dict[str, Any]
                       ) -> tuple[tuple[cKDTree, float, float], dict[str, Any]]:
    features, _, laps = _training_matrix(dataset_path)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    normalized = (features - mean) / scale
    heldout_distances = []
    for lap in np.unique(laps):
        valid = laps == lap
        reference = ~valid
        tree = cKDTree(normalized[reference])
        heldout_distances.append(tree.query(normalized[valid], k=1)[0])
    distances = np.concatenate(heldout_distances)
    full_threshold, zero_threshold = np.quantile(distances, (0.95, 0.99))
    return ((cKDTree(normalized), float(full_threshold), float(zero_threshold)),
            {
                "distance_definition": "nearest other-lap training transition in standardized 9-feature state",
                "leave_one_lap_out_distance_quantiles": {
                    str(q): float(np.quantile(distances, q))
                    for q in (0.5, 0.9, 0.95, 0.99, 1.0)},
                "full_weight_distance": float(full_threshold),
                "zero_weight_distance": float(zero_threshold),
                "held_out_laps": sorted(int(lap) for lap in np.unique(laps)),
                "training_rows": int(len(features)),
            })


def _predict_standardized(features: np.ndarray,
                          model: dict[str, Any]) -> np.ndarray:
    normalized = ((features - np.asarray(model["feature_mean"], dtype=np.float64))
                  / np.asarray(model["feature_scale"], dtype=np.float64))
    return (normalized @ np.asarray(model["coef"], dtype=np.float64)
            + float(model["intercept"]))


def _fit_stable_regression(features: np.ndarray, target: np.ndarray,
                           alpha: float) -> dict[str, Any]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(len(features)), normalized))
    if alpha > 0.0:
        penalty = np.zeros((features.shape[1], design.shape[1]))
        penalty[:, 1:] = np.eye(features.shape[1]) * math.sqrt(alpha)
        design = np.vstack((design, penalty))
        target_augmented = np.concatenate((target, np.zeros(features.shape[1])))
    else:
        target_augmented = target
    lower = np.full(design.shape[1], -np.inf)
    upper = np.full(design.shape[1], np.inf)
    lower[2] = -40.0 * scale[1]
    upper[2] = 0.0
    fitted = lsq_linear(
        design, target_augmented, bounds=(lower, upper), max_iter=500,
        lsmr_tol="auto")
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"stable regression failed: {fitted.message}")
    coef = fitted.x[1:]
    beta_v = float(coef[1] / scale[1])
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": float(fitted.x[0]),
        "raw_vdot_coefficient_on_v_rear_per_s": beta_v,
        "discrete_v_feedback_pole_at_40hz": 1.0 + DT_S * beta_v,
        "stability_constrained": True,
        "ridge_alpha": float(alpha),
    }


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _teacher_forced_step_report(
        library: ctypes.CDLL, indices: np.ndarray,
        projection: dict[str, np.ndarray],
        controls: tuple[np.ndarray, np.ndarray],
        rigid: np.ndarray, frames: np.ndarray, truth: np.ndarray,
        trajectory: np.ndarray, lap_length: float) -> dict[str, Any]:
    steering, target_speed = controls
    errors: list[np.ndarray] = []
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    failures = 0
    for index in range(2, len(indices) - 1):
        dataset_index = int(indices[index])
        state = MpcModelState(
            float(projection["e_y"][index]),
            float(projection["e_psi"][index]),
            float(rigid[dataset_index, 7]),
            float(rigid[dataset_index, 8]
                  - COM_X_M * rigid[dataset_index, 12]),
            float(rigid[dataset_index, 12]),
            float(target_speed[index]), float(steering[index]),
            float(steering[index - 1]), float(steering[index - 2]),
            float(frames[dataset_index, 3]))
        command = MpcModelControl(
            float((steering[index + 1] - steering[index]) / DT_S),
            float((target_speed[index + 1] - target_speed[index]) / DT_S))
        progress = float(projection["progress"][index])
        curvature = float(np.interp(
            progress % lap_length, trajectory[:, 0], trajectory[:, 4]))
        stage = library.mpc_vehicle_model_step(
            ctypes.byref(state), ctypes.byref(command),
            ctypes.c_float(DT_S), ctypes.c_float(curvature))
        if not stage.valid:
            failures += 1
            continue
        following = stage.next
        predicted = np.asarray((
            float(following.u),
            float(following.v + COM_X_M * following.r),
            float(following.v), float(following.r),
            float(following.e_y), float(following.e_psi),
            float(following.actual_steering_angle),
            float(following.target_speed),
            float(following.steering_command)), dtype=np.float64)
        error = predicted - truth[index + 1]
        error[5] = _wrap(float(error[5]))
        errors.append(error)
        speed = math.hypot(float(rigid[dataset_index, 7]),
                           float(rigid[dataset_index, 8]))
        abs_steering = abs(float(frames[dataset_index, 3]))
        speed_bin = ("speed_0_3" if speed < 3.0 else
                     "speed_3_5" if speed < 5.0 else
                     "speed_5_7" if speed < 7.0 else "speed_7_plus")
        steering_bin = ("steer_0_0p1" if abs_steering < 0.1 else
                        "steer_0p1_0p2" if abs_steering < 0.2 else
                        "steer_0p2_0p3" if abs_steering < 0.3 else
                        "steer_0p3_plus")
        groups[speed_bin].append(error)
        groups[steering_bin].append(error)

    matrix = np.asarray(errors, dtype=np.float64)
    if not len(matrix):
        raise ValueError("teacher-forced production step report is empty")
    per_state = {
        name: {
            "rmse": _rmse(matrix[:, channel]),
            "bias": float(np.mean(matrix[:, channel])),
            "p95_abs": float(np.quantile(np.abs(matrix[:, channel]), 0.95)),
            "max_abs": float(np.max(np.abs(matrix[:, channel]))),
        }
        for channel, name in enumerate(STATE_NAMES)}
    thresholds = (
        ("u_com_mps", 0, 0.1), ("v_com_mps", 1, 0.1),
        ("v_rear_mps", 2, 0.1), ("yaw_rate_rps", 3, 0.1),
        ("e_y_m", 4, 0.1), ("e_psi_rad", 5, 0.1),
        ("actual_steering_rad", 6, 0.05))
    first_crossings = {}
    for name, channel, threshold in thresholds:
        crossing = np.flatnonzero(np.abs(matrix[:, channel]) >= threshold)
        first_crossings[name] = (None if not len(crossing) else {
            "transition_index_from_run_start": int(crossing[0] + 2),
            "elapsed_s_from_run_start": float((crossing[0] + 3) * DT_S),
        })
    grouped = {}
    for name, values in groups.items():
        group_matrix = np.asarray(values, dtype=np.float64)
        grouped[name] = {
            "samples": int(len(group_matrix)),
            **{state_name: _rmse(group_matrix[:, channel])
               for channel, state_name in enumerate(STATE_NAMES)},
        }
    return {
        "teacher_forced_from_simulator_truth_each_step": True,
        "valid_transitions": int(len(matrix)),
        "invalid_transitions": failures,
        "per_state": per_state,
        "first_abs_error_threshold_crossing": first_crossings,
        "speed_and_steering_groups": grouped,
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_STEPWISE.read_text(encoding="utf-8"))
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or _sha256(Path(baseline["model_source"]))
            != baseline["model_source_sha256"]
            or _sha256(CONFIG) != baseline["config_sha256"]
            or _sha256(MODEL_LIBRARY) != baseline["model_library_sha256"]
            or _sha256(TRAJECTORY) != baseline["trajectory_sha256"]
            or _sha256(_workspace_path(benchmark["dataset"]))
            != benchmark["dataset_sha256"]
            or _sha256(_workspace_path(benchmark["dataset"]))
            != benchmark["dataset_sha256"]):
        raise ValueError("frozen benchmark or production model provenance changed")
    if set(BAGS) != set(benchmark["validation_reports"]):
        raise ValueError("candidate requires exactly the frozen r02/r03 bags")
    for run_id, bag in BAGS.items():
        expected = benchmark["validation_reports"][run_id]["bag_sha256"]
        if _sha256(bag) != expected:
            raise ValueError(f"held-out practice bag hash mismatch: {run_id}")

    dataset_path = _workspace_path(benchmark["dataset"])
    model_sweep, model_fit_metadata = _fit_model_sweep(dataset_path)
    knn_support, knn_support_metadata = _build_knn_support(
        dataset_path, model_sweep[CANDIDATE_MODEL])
    model_fit_metadata["knn_support_gate"] = knn_support_metadata
    data = _load_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        data["sample_time_ns"] = archive["sample_time_ns"].astype(
            np.int64, copy=False)
    window_data = {**data, "sequence_bounds": data["bounds"],
                   "sequence_run_index": data["seq_run"]}
    run_index_by_name = {str(name): index
                         for index, name in enumerate(data["run_ids"])}
    dataset_index_by_report = {
        report_id: run_index_by_name[
            f"practice_unseen_model_validation_20261001_"
            f"{report_id.rsplit('_', 1)[-1]}"]
        for report_id in benchmark["validation_reports"]}
    data_runs = {
        name: _run_rows(window_data, run_index)
        for name, run_index in dataset_index_by_report.items()}
    command_streams = {name: _read_commands(path)
                       for name, path in BAGS.items()}
    with np.load(dataset_path, allow_pickle=False) as archive:
        rigid = archive["simulator_rigid_state"].astype(np.float64)
        frames = archive["frames"].astype(np.float64)
        pose = archive["simulator_pose_xyyaw"].astype(np.float64)
        sample_stamps = archive["sample_time_ns"].astype(np.int64)
        lap_count = archive["lap_count"].astype(np.int32)
    trajectory = np.loadtxt(TRAJECTORY, delimiter=",", comments="#",
                            dtype=np.float64)

    library = ctypes.CDLL(str(MODEL_LIBRARY.resolve()))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(MpcModelState), ctypes.POINTER(MpcModelControl),
        ctypes.c_float, ctypes.c_float]
    library.mpc_vehicle_model_step.restype = MpcStageResult
    report_modes = (
        "production_hold_v", CANDIDATE_MODEL, SUPPORTED_CANDIDATE_MODEL,
        KNN_SUPPORTED_CANDIDATE_MODEL)
    report_by_run: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        mode: defaultdict(lambda: defaultdict(list))
        for mode in report_modes}
    report_by_group: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        mode: defaultdict(lambda: defaultdict(list))
        for mode in report_by_run}
    failures = {name: [] for name in report_by_run}
    window_count = 0

    with ProductionMpc(MODEL_LIBRARY, CONFIG, TRAJECTORY) as controller:
        lap_length = controller.lap_length_m
        projections: dict[str, dict[str, np.ndarray]] = {}
        controls_by_run: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        truth_by_run: dict[str, np.ndarray] = {}
        for report_id, dataset_index in dataset_index_by_report.items():
            indices = data_runs[report_id]
            projection = _project_run(
                controller, trajectory, lap_length, pose[indices])
            projections[report_id] = dict(zip(
                ("progress", "e_y", "e_psi", "curvature", "distance"),
                projection))
            controls_by_run[report_id] = _recorded_command_arrays(
                data, dataset_index, command_streams[report_id], indices)
            truth_by_run[report_id] = np.column_stack((
                rigid[indices, 7], rigid[indices, 8],
                rigid[indices, 8] - COM_X_M * rigid[indices, 12],
                rigid[indices, 12], projections[report_id]["e_y"],
                projections[report_id]["e_psi"], frames[indices, 3],
                controls_by_run[report_id][1], controls_by_run[report_id][0],
            ))

        for window in benchmark["windows"]:
            report_id = next(
                key for key in benchmark["validation_reports"]
                if key.endswith(window["run_id"].rsplit("_", 1)[-1]))
            indices = data_runs[report_id]
            local_index = int(np.searchsorted(
                indices, int(window["global_start_index"])))
            if (local_index >= len(indices)
                    or indices[local_index] != int(window["global_start_index"])
                    or local_index < 2
                    or local_index + round(max(HORIZONS_S) / DT_S) >= len(indices)):
                raise ValueError("frozen practice window is not continuously aligned")
            projection = projections[report_id]
            steering, target_speed = controls_by_run[report_id]
            values = truth_by_run[report_id]
            initial = MpcModelState(
                float(projection["e_y"][local_index]),
                float(projection["e_psi"][local_index]),
                float(rigid[indices[local_index], 7]),
                float(rigid[indices[local_index], 8]
                      - COM_X_M * rigid[indices[local_index], 12]),
                float(rigid[indices[local_index], 12]),
                float(target_speed[local_index]),
                float(steering[local_index]),
                float(steering[local_index - 1]),
                float(steering[local_index - 2]),
                float(frames[indices[local_index], 3]),
            )
            dataset_run = str(window["run_id"])
            start_steering = float(frames[indices[local_index], 3])
            start_speed = math.hypot(
                float(rigid[indices[local_index], 7]),
                float(rigid[indices[local_index], 8]))
            turn_group = ("negative_steering" if start_steering < -0.02 else
                          "positive_steering" if start_steering > 0.02 else
                          "near_zero_steering")
            speed_group = ("speed_0_3" if start_speed < 3.0 else
                           "speed_3_5" if start_speed < 5.0 else
                           "speed_5_7" if start_speed < 7.0 else
                           "speed_7_plus")
            window_groups = (turn_group, speed_group)
            for mode in report_by_run:
                candidate_model = (None if mode == "production_hold_v"
                                   else model_sweep[CANDIDATE_MODEL])
                rollout_mode = (
                    "production_hold_v" if candidate_model is None else
                    "support_gated_vdot_candidate"
                    if mode == SUPPORTED_CANDIDATE_MODEL else
                    "knn_supported_vdot_candidate"
                    if mode == KNN_SUPPORTED_CANDIDATE_MODEL else
                    "direct_vdot_candidate")
                prediction_error, failure = _rollout(
                    library, rollout_mode, initial,
                    float(projection["progress"][local_index]),
                    local_index, steering, target_speed, trajectory,
                    lap_length, values, candidate_model or {},
                    knn_support=knn_support)
                if prediction_error is None or failure is not None:
                    failures[mode].append({
                        "run_id": dataset_run,
                        "packet_sequence": int(window["packet_sequence"]),
                        **(failure or {}),
                    })
                if prediction_error is None or failure is not None:
                    continue
                for horizon in HORIZONS_S:
                    steps = round(horizon / DT_S)
                    horizon_error = prediction_error[:steps]
                    report_by_run[mode][dataset_run][f"{horizon:g}s"].append(
                        horizon_error)
                    for group in window_groups:
                        report_by_group[mode][group][f"{horizon:g}s"].append(
                            horizon_error)
            window_count += 1

    scores: dict[str, Any] = {}
    for mode, runs in report_by_run.items():
        scores[mode] = {"by_horizon": {}, "per_run": {}}
        for horizon in HORIZONS_S:
            key = f"{horizon:g}s"
            per_run: dict[str, dict[str, float]] = {}
            for run_id, horizons in runs.items():
                matrices = horizons.get(key, [])
                if not matrices:
                    continue
                per_run[run_id] = {
                    name: float(np.mean([_rmse(matrix[:, channel])
                                         for matrix in matrices]))
                    for channel, name in enumerate(STATE_NAMES)}
            scores[mode]["per_run"][key] = per_run
            scores[mode]["by_horizon"][key] = {
                name: float(np.mean([values[name] for values in per_run.values()]))
                for name in STATE_NAMES}
        scores[mode]["per_group"] = {}
        for group, horizons in report_by_group[mode].items():
            scores[mode]["per_group"][group] = {}
            for horizon in HORIZONS_S:
                key = f"{horizon:g}s"
                matrices = horizons.get(key, [])
                if matrices:
                    scores[mode]["per_group"][group][key] = {
                        "window_count": len(matrices),
                        **{
                            name: float(np.mean([
                                _rmse(matrix[:, channel])
                                for matrix in matrices]))
                            for channel, name in enumerate(STATE_NAMES)
                        },
                    }

    teacher_forced_reports = {}
    for report_id, dataset_index in dataset_index_by_report.items():
        indices = data_runs[report_id]
        teacher_forced_reports[report_id] = _teacher_forced_step_report(
            library, indices, projections[report_id],
            controls_by_run[report_id], rigid, frames,
            truth_by_run[report_id], trajectory, lap_length)

    full_run_reports: dict[str, Any] = {}
    for report_id in dataset_index_by_report:
        indices = data_runs[report_id]
        projection = projections[report_id]
        steering, target_speed = controls_by_run[report_id]
        truth = truth_by_run[report_id]
        local_index = 2
        if local_index + 2 >= len(indices):
            raise ValueError(f"validation run is too short for full replay: {report_id}")
        initial = MpcModelState(
            float(projection["e_y"][local_index]),
            float(projection["e_psi"][local_index]),
            float(rigid[indices[local_index], 7]),
            float(rigid[indices[local_index], 8]
                  - COM_X_M * rigid[indices[local_index], 12]),
            float(rigid[indices[local_index], 12]),
            float(target_speed[local_index]),
            float(steering[local_index]),
            float(steering[local_index - 1]),
            float(steering[local_index - 2]),
            float(frames[indices[local_index], 3]),
        )
        run_modes: dict[str, Any] = {}
        steps = len(indices) - local_index - 1
        full_run_models = {
            "production_hold_v": None,
            SUPPORTED_CANDIDATE_MODEL: model_sweep[CANDIDATE_MODEL],
            KNN_SUPPORTED_CANDIDATE_MODEL: model_sweep[CANDIDATE_MODEL],
            **{
                f"stable_ridge_alpha_{alpha:g}": model_sweep[
                    f"stable_ridge_alpha_{alpha:g}"]
                for alpha in RIDGE_ALPHAS
            },
        }
        for mode, candidate_model in full_run_models.items():
            rollout_mode = (
                "production_hold_v" if candidate_model is None else
                "support_gated_vdot_candidate"
                if mode == SUPPORTED_CANDIDATE_MODEL else
                "knn_supported_vdot_candidate"
                if mode == KNN_SUPPORTED_CANDIDATE_MODEL else
                "direct_vdot_candidate")
            trace = []
            errors, failure = _rollout(
                library, rollout_mode, initial,
                float(projection["progress"][local_index]), local_index,
                steering, target_speed, trajectory, lap_length, truth,
                candidate_model or {}, max_steps=steps,
                debug_trace=trace if mode in (
                    "production_hold_v", CANDIDATE_MODEL,
                    SUPPORTED_CANDIDATE_MODEL,
                    KNN_SUPPORTED_CANDIDATE_MODEL) else None,
                knn_support=knn_support)
            if errors is None or not len(errors):
                run_modes[mode] = {"failure": failure,
                                   "predicted_samples_before_failure": 0}
                continue
            per_state = {
                name: {
                    "rmse": _rmse(errors[:, channel]),
                    "bias": float(np.mean(errors[:, channel])),
                    "p95_abs": float(np.quantile(
                        np.abs(errors[:, channel]), 0.95)),
                    "max_abs": float(np.max(np.abs(errors[:, channel]))),
                }
                for channel, name in enumerate(STATE_NAMES)}
            aligned_laps = lap_count[indices[
                local_index + 1:local_index + 1 + len(errors)]]
            per_lap: dict[str, Any] = {}
            for lap in sorted(np.unique(aligned_laps).tolist()):
                mask = aligned_laps == lap
                if np.count_nonzero(mask) < 20:
                    continue
                per_lap[str(int(lap))] = {
                    "samples": int(np.count_nonzero(mask)),
                    "state_rmse": {
                        name: _rmse(errors[mask, channel])
                        for channel, name in enumerate(STATE_NAMES)},
                }
            crossing = {}
            crossing_channels = (
                ("u_com_abs_error_0p1_mps", 0, 0.1),
                ("rear_v_abs_error_0p1_mps", 2, 0.1),
                ("yaw_rate_abs_error_0p1_radps", 3, 0.1),
                ("e_y_abs_error_0p1_m", 4, 0.1),
                ("e_psi_abs_error_0p1_rad", 5, 0.1))
            for name, channel, threshold in crossing_channels + (
                    ("v_com_abs_error_0p1_mps", 1, 0.1),
                    ("actual_steering_abs_error_0p05_rad", 6, 0.05)):
                first = np.flatnonzero(np.abs(errors[:, channel]) >= threshold)
                crossing[name] = (None if not len(first) else {
                    "elapsed_s": float((int(first[0]) + 1) * DT_S),
                    "lap_count": int(aligned_laps[int(first[0])]),
                })
            trace_context = {}
            if trace:
                for name, channel, threshold in crossing_channels:
                    first = np.flatnonzero(np.abs(errors[:, channel]) >= threshold)
                    if len(first):
                        center = int(first[0])
                        trace_context[name] = trace[
                            max(0, center - 5):min(len(trace), center + 6)]
            run_modes[mode] = {
                "predicted_until_s": float(len(errors) * DT_S),
                "predicted_samples_before_failure": int(len(errors)),
                "failure": failure,
                "per_state": per_state,
                "per_lap": per_lap,
                "first_error_threshold_crossing": crossing,
                "first_error_context": trace_context,
            }
        full_run_reports[report_id] = run_modes

    result = {
        "schema_version": 1,
        "analysis": "recursive held-out practice replay of a practice-train-only direct v increment model versus production hold-v",
        "offline_only": True,
        "simulator_was_run": False,
        "runtime_model_changed": False,
        "truth_used_as_input_after_initialization": False,
        "test_or_final_test_used": False,
        "initialization": "truth u, rear-axle v, yaw-rate; identical recorded map pose, commands, actuator state and path curvature",
        "windows": window_count,
        "failed_rollouts": failures,
        "model_fit": {
            **model_fit_metadata,
            "evaluated_candidate_model": CANDIDATE_MODEL,
            "validation_runs": [
                "practice_unseen_model_validation_20261001_r02",
                "practice_unseen_model_validation_20261001_r03"],
            "equation": "v_rear_next = v_rear + dt * ridge([u,v_rear,r,delta,u*r,u*delta,u*delta*abs(delta),v*abs(r),r*abs(r)])",
        },
        "scores": scores,
        "teacher_forced_one_step_production_model": teacher_forced_reports,
        "full_recorded_command_run_replay": {
            "ground_truth_resets_during_run": 0,
            "initialization_sample_index": 2,
            "runs": full_run_reports,
        },
        "provenance": {
            "benchmark_sha256": _sha256(BENCHMARK),
            "dataset_sha256": _sha256(dataset_path),
            "model_source_sha256": _sha256(Path(baseline["model_source"])),
            "production_library_sha256": _sha256(MODEL_LIBRARY),
            "config_sha256": _sha256(CONFIG),
            "trajectory_sha256": _sha256(TRAJECTORY),
            "validation_bag_sha256": {
                name: _sha256(path) for name, path in BAGS.items()},
        },
        "limitations": [
            "only two independent validation captures are available; macro confidence is therefore limited",
            "this is recorded-command model replay, not solver-in-loop or live simulator evidence",
            "the direct residual is a candidate plant change, not yet integrated into MPC C or its analytic Jacobian",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "windows": result["windows"],
        "failed_rollouts": result["failed_rollouts"],
        "scores": {
            mode: {key: value for key, value in details["by_horizon"].items()}
            for mode, details in result["scores"].items()},
    }, indent=2, sort_keys=True))
