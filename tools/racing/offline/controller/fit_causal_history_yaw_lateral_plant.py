#!/usr/bin/env python3
"""Fit a causal yaw/lateral teacher using approved throttle and race data."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.racing.offline.controller import evaluate_mpc_practice_yaw_residual as yaw
from tools.racing.offline.controller.production_mpc import ProductionMpc
from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import COM_X_M
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[4]
PAIRED_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "swerve_throttle_slew_20261006/paired_teacher_dataset_with_practice_transfer/"
    "openplane_dynamics.npz")
OUTPUT = yaw.ANALYSIS_DIR / "causal_history_yaw_lateral_plant_dagger_weight_sweep.json"
BASELINE_YAW_REPORT = yaw.ANALYSIS_DIR / "practice_low_mid_steer_yaw_residual.json"
HISTORY = 4
DT = 0.025
VALIDATION_IDS = {
    "practice_unseen_model_validation_20261001_r02",
    "practice_unseen_model_validation_20261001_r03",
}
STATE_NAMES = (
    "u_mps", "v_rear_mps", "yaw_rate_rps", "steering_feedback_rad",
    "throttle_feedback_norm", "rear_left_wheel_mps",
    "rear_right_wheel_mps", "steering_command_rad",
    "throttle_command_norm", "imu_ax_mps2", "imu_ay_mps2")
OUTPUT_INDICES = (0, 1, 2, 3, 4, 5, 6, 9, 10)
TARGET_NAMES = (
    "du_mps", "dv_rear_mps", "dr_radps", "dsteering_rad",
    "dthrottle_norm", "dleft_wheel_mps", "dright_wheel_mps",
    "dimu_ax_mps2", "dimu_ay_mps2")
MODEL_KWARGS = {
    "loss": "squared_error", "learning_rate": 0.05,
    "max_iter": 120, "max_leaf_nodes": 15,
    "min_samples_leaf": 120, "l2_regularization": 1.0,
    "early_stopping": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_arrays(archive: Any) -> np.ndarray:
    rigid = archive["simulator_rigid_state"].astype(np.float64)
    sensors = archive["sensor_frames"].astype(np.float64)
    state = np.column_stack((
        rigid[:, 7], rigid[:, 8] - COM_X_M * rigid[:, 12], rigid[:, 12],
        sensors[:, 0], sensors[:, 1], sensors[:, 2], sensors[:, 3],
        sensors[:, 7], sensors[:, 8], sensors[:, 4], sensors[:, 5]))
    return state


def _feature_names() -> tuple[list[str], list[int]]:
    names = list(STATE_NAMES)
    current_width = len(names) + 2
    names.extend(("wheel_body_mismatch_mps", "wheel_asymmetry_mps"))
    lag_channels = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    for lag in (1, 2, 4):
        for channel in lag_channels:
            names.append(f"d{STATE_NAMES[channel]}_over_{lag}_frames")
    names.extend(("wheel_body_mismatch_rate_1frame_mpsps",
                  "wheel_body_mismatch_rate_100ms_mpsps",
                  "wheel_asymmetry_rate_1frame_mpsps",
                  "wheel_asymmetry_rate_100ms_mpsps"))
    names.extend(("u_times_steering", "steering_rate_times_speed",
                  "throttle_feedback_slew_times_steering",
                  "wheel_mismatch_times_steering"))
    return names, list(range(current_width))


FEATURE_NAMES, CURRENT_FEATURE_INDICES = _feature_names()


def _features(history: np.ndarray) -> np.ndarray:
    if history.shape != (HISTORY + 1, len(STATE_NAMES)):
        raise ValueError(f"expected {HISTORY + 1} causal state rows")
    now = history[-1]
    prior1, prior2, prior4 = history[-2], history[-3], history[0]
    mismatch = 0.5 * (now[5] + now[6]) - now[0]
    asymmetry = now[5] - now[6]
    mismatch1 = 0.5 * (prior1[5] + prior1[6]) - prior1[0]
    mismatch4 = 0.5 * (prior4[5] + prior4[6]) - prior4[0]
    asymmetry1 = prior1[5] - prior1[6]
    asymmetry4 = prior4[5] - prior4[6]
    steer_rate = (now[3] - prior1[3]) / DT
    throttle_rate = (now[4] - prior4[4]) / (HISTORY * DT)
    values = [*now, mismatch, asymmetry]
    for prior, interval in ((prior1, DT), (prior2, 2 * DT),
                            (prior4, HISTORY * DT)):
        values.extend(((now - prior) / interval).tolist())
    values.extend(((mismatch - mismatch1) / DT,
                   (mismatch - mismatch4) / (HISTORY * DT),
                   (asymmetry - asymmetry1) / DT,
                   (asymmetry - asymmetry4) / (HISTORY * DT),
                   now[0] * now[3], now[0] * steer_rate,
                   throttle_rate * now[3], mismatch * now[3]))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("causal model feature vector is invalid")
    return result


def _collect_rows(dataset_path: Path, split_policy: str,
                  run_ids_allowed: set[str] | None = None
                  ) -> dict[str, Any]:
    # NPZ members are compressed as whole arrays. Filtering run IDs after
    # _load_dataset() would still decompress sensor and truth arrays for every
    # run in the archive, including sealed test/final-test splits. Inspect only
    # split metadata first and refuse mixed sealed archives before touching any
    # numeric member; use the manifest-gated per-run corpus loader instead.
    with np.load(dataset_path, allow_pickle=False) as metadata:
        if not {"run_ids", "run_splits"}.issubset(metadata.files):
            raise ValueError(f"{dataset_path}: missing run split metadata")
        metadata_run_ids = metadata["run_ids"].astype(str)
        metadata_run_splits = metadata["run_splits"].astype(str)
        if (metadata_run_ids.ndim != 1
                or metadata_run_splits.shape != metadata_run_ids.shape):
            raise ValueError(f"{dataset_path}: malformed run split metadata")
        sealed = metadata_run_splits == "test"
        sealed |= metadata_run_splits == "final_test"
        sealed_run_ids = metadata_run_ids[sealed].tolist()
    if sealed_run_ids:
        raise ValueError(
            f"refusing to decompress {dataset_path}: archive mixes requested "
            f"data with sealed test/final_test runs {sealed_run_ids}; use a "
            "clean per-run training/validation archive")
    data = _load_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as archive:
        state = _state_arrays(archive)
        rigid = archive["simulator_rigid_state"].astype(np.float64)
        bounds = archive["sequence_bounds"].astype(np.int64)
        sequence_runs = archive["sequence_run_index"].astype(np.int32)
        run_ids = archive["run_ids"].astype(str)
        run_splits = archive["run_splits"].astype(str)
        sensors_valid = archive["sensor_valid"].astype(bool)
        dt = archive["dt_s"].astype(np.float64)
        packet = archive["packet_sequence"].astype(np.int64)
        pose = archive["simulator_pose_xyyaw"].astype(np.float64)
        frame_run_index = archive["frame_run_index"].astype(np.int32)
        sequence_run_index = archive["sequence_run_index"].astype(np.int32)
    (x_parts, y_parts, next_state_parts, run_parts, speed_parts,
     steer_parts, throttle_slew_parts) = ([], [], [], [], [], [], [])
    permitted = set(run_ids_allowed or ())
    for sequence_id, (start_raw, end_raw) in enumerate(bounds):
        run_index = int(sequence_runs[sequence_id])
        run_id = str(run_ids[run_index])
        split = str(run_splits[run_index])
        if split_policy == "train":
            if split != "train":
                continue
        elif run_id not in permitted:
            continue
        start, end = int(start_raw), int(end_raw)
        if end - start <= HISTORY + 1:
            continue
        if (not np.allclose(dt[start:end], DT, rtol=0.0, atol=1e-7)
                or np.any(np.diff(packet[start:end]) != 1)):
            raise ValueError(f"non-contiguous fixed-grid sequence: {run_id}")
        feature_rows, target_rows, next_state_rows = [], [], []
        row_speeds, row_steers, row_throttle_slews = [], [], []
        for index in range(start + HISTORY, end - 1):
            if not sensors_valid[index] or not np.isfinite(state[index:index + 2]).all():
                continue
            history = state[index - HISTORY:index + 1]
            feature_rows.append(_features(history))
            target_rows.append(state[index + 1, OUTPUT_INDICES]
                               - state[index, OUTPUT_INDICES])
            next_state_rows.append(state[index + 1, OUTPUT_INDICES])
            row_speeds.append(float(state[index, 0]))
            row_steers.append(float(state[index, 3]))
            row_throttle_slews.append(float(
                (state[index, 4] - state[index - HISTORY, 4])
                / (HISTORY * DT)))
        if not feature_rows:
            continue
        x_parts.append(np.asarray(feature_rows, dtype=np.float64))
        y_parts.append(np.asarray(target_rows, dtype=np.float64))
        next_state_parts.append(np.asarray(next_state_rows, dtype=np.float64))
        run_parts.extend([run_id] * len(feature_rows))
        speed_parts.extend(row_speeds)
        steer_parts.extend(row_steers)
        throttle_slew_parts.extend(row_throttle_slews)

    if not x_parts:
        raise ValueError(f"no samples for split policy {split_policy}")
    return {
        "x": np.concatenate(x_parts), "y": np.concatenate(y_parts),
        "next_y": np.concatenate(next_state_parts),
        "run_id": np.asarray(run_parts, dtype=str),
        "speed": np.asarray(speed_parts), "steering": np.asarray(steer_parts),
        "throttle_slew": np.asarray(throttle_slew_parts),
        "state": state, "rigid": rigid, "pose": pose,
        "run_ids": run_ids, "run_splits": run_splits,
        "frame_run_index": frame_run_index,
        "sequence_bounds": bounds,
        "sequence_run_index": sequence_run_index,
        "sensor_valid": sensors_valid,
        "dataset_sha256": _sha256(dataset_path),
    }


def _combine_training(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [str(run_id) for data in datasets for run_id in np.unique(data["run_id"])]
    if len(ids) != len(set(ids)):
        duplicates = sorted(run for run, count in Counter(ids).items() if count > 1)
        raise ValueError(f"training runs overlap across source datasets: {duplicates}")
    x = np.concatenate([data["x"] for data in datasets])
    y = np.concatenate([data["y"] for data in datasets])
    next_y = np.concatenate([data["next_y"] for data in datasets])
    run_id = np.concatenate([data["run_id"] for data in datasets])
    counts = Counter(run_id.tolist())
    weights = np.asarray([1.0 / counts[name] for name in run_id], dtype=np.float64)
    weights *= len(counts) / weights.sum()
    return {"x": x, "y": y, "next_y": next_y,
            "run_id": run_id, "weight": weights,
            "runs": sorted(counts), "samples_per_run": dict(counts)}


def _fit_heads(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
               feature_indices: list[int]) -> list[HistGradientBoostingRegressor]:
    heads = []
    selected_x = x[:, feature_indices]
    for target in range(y.shape[1]):
        model = HistGradientBoostingRegressor(**MODEL_KWARGS)
        model.fit(selected_x, y[:, target], sample_weight=weights)
        heads.append(model)
    return heads


def _predict(heads: list[HistGradientBoostingRegressor], x: np.ndarray,
             feature_indices: list[int]) -> np.ndarray:
    selected_x = x[:, feature_indices]
    return np.column_stack([model.predict(selected_x) for model in heads])


def _metric(error: np.ndarray) -> dict[str, float]:
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias": float(np.mean(error)),
            "p95_abs": float(np.quantile(np.abs(error), 0.95)),
            "max_abs": float(np.max(np.abs(error)))}


def _one_step_report(data: dict[str, Any], prediction: np.ndarray
                     ) -> dict[str, Any]:
    error = prediction - data["y"]
    channel = {name: _metric(error[:, index])
               for index, name in enumerate(TARGET_NAMES)}
    per_run = {}
    for run in np.unique(data["run_id"]):
        mask = data["run_id"] == run
        per_run[str(run)] = {
            name: _metric(error[mask, index])
            for index, name in enumerate(TARGET_NAMES)}
    bins = {}
    for field, values, edges in (
            ("speed_mps", data["speed"], (0, 3, 5, 7, 9, 12, 30)),
            ("abs_steering_rad", np.abs(data["steering"]),
             (0, 0.08, 0.14, 0.20, 0.30, 0.45, 1.0)),
            ("throttle_slew_normps", data["throttle_slew"],
             (-100, -0.15, -0.03, 0.03, 0.15, 100))):
        grouped = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (values >= lo) & (values < hi)
            if int(mask.sum()) < 30:
                continue
            grouped[f"{lo:g}_{hi:g}"] = {
                "samples": int(mask.sum()),
                "independent_runs": int(len(np.unique(data["run_id"][mask]))),
                "targets": {name: _metric(error[mask, index])
                            for index, name in enumerate(TARGET_NAMES)},
            }
        bins[field] = grouped
    return {"samples": int(len(error)), "per_target": channel,
            "per_run": per_run, "regime_bins": bins}


def _recursive_run(heads: list[HistGradientBoostingRegressor],
                   data: dict[str, Any], run_id: str,
                   feature_indices: list[int],
                   train_support: dict[str, tuple[float, float]],
                   prediction_mode: str = "delta") -> dict[str, Any]:
    run_matches = np.flatnonzero(data["run_ids"] == run_id)
    if len(run_matches) != 1:
        raise ValueError(f"validation run is not unique: {run_id}")
    run_index = int(run_matches[0])
    indices = np.flatnonzero(data["frame_run_index"] == run_index)
    if len(indices) < 20 or not np.all(np.diff(indices) == 1):
        raise ValueError(f"held-out practice capture is not one continuous run: {run_id}")
    state_truth = data["state"][indices]
    pose_truth = data["pose"][indices]
    if not np.isfinite(state_truth).all() or not np.isfinite(pose_truth).all():
        raise ValueError(f"held-out run has unavailable truth: {run_id}")

    history = [row.copy() for row in state_truth[:HISTORY + 1]]
    predicted = [state_truth[HISTORY].copy()]
    x, y, heading = map(float, pose_truth[HISTORY])
    predicted_pose = [np.asarray((x, y, heading), dtype=np.float64)]
    start = HISTORY
    failure = None
    support_exit = {}
    for local in range(start, len(indices) - 1):
        feature = _features(np.asarray(history[-(HISTORY + 1):]))
        delta = _predict(heads, feature[None, :], feature_indices)[0]
        next_row = history[-1].copy()
        if prediction_mode == "delta":
            next_row[list(OUTPUT_INDICES)] += delta
        elif prediction_mode == "absolute_next_state":
            next_row[list(OUTPUT_INDICES)] = delta
        else:
            raise ValueError(f"unknown prediction mode: {prediction_mode}")
        next_global = int(indices[local + 1])
        # Commands are exogenous inputs in this replay. No future feedback or
        # truth is copied into the candidate state.
        next_row[7] = data["state"][next_global, 7]
        next_row[8] = data["state"][next_global, 8]
        if not np.isfinite(next_row).all():
            failure = {"elapsed_s": (local - start + 1) * DT,
                       "reason": "non_finite_recursive_state"}
            break
        elapsed = (local - start + 1) * DT
        for channel, name in enumerate(STATE_NAMES):
            low, high = train_support[name]
            if name not in support_exit and not low <= next_row[channel] <= high:
                support_exit[name] = {
                    "elapsed_s": elapsed,
                    "predicted": float(next_row[channel]),
                    "training_q001_q999": [low, high],
                }
        current_row = history[-1]
        u_mid = 0.5 * (current_row[0] + next_row[0])
        v_mid = 0.5 * (current_row[1] + next_row[1])
        r_mid = 0.5 * (current_row[2] + next_row[2])
        yaw_mid = heading + 0.5 * DT * r_mid
        x += DT * (u_mid * math.cos(yaw_mid) - v_mid * math.sin(yaw_mid))
        y += DT * (u_mid * math.sin(yaw_mid) + v_mid * math.cos(yaw_mid))
        heading = math.atan2(math.sin(heading + DT * r_mid),
                             math.cos(heading + DT * r_mid))
        history.append(next_row)
        predicted.append(next_row.copy())
        predicted_pose.append(np.asarray((x, y, heading)))

    predicted = np.asarray(predicted)
    predicted_pose = np.asarray(predicted_pose)
    truth = state_truth[start:start + len(predicted)]
    pose = pose_truth[start:start + len(predicted)]
    state_error = predicted - truth
    pose_error = predicted_pose - pose
    pose_error[:, 2] = np.arctan2(np.sin(pose_error[:, 2]),
                                 np.cos(pose_error[:, 2]))
    crossing_limits = {
        "u_mps": (0, 0.20), "v_rear_mps": (1, 0.10),
        "yaw_rate_rps": (2, 0.10), "steering_feedback_rad": (3, 0.02),
        "throttle_feedback_norm": (4, 0.05),
        "rear_left_wheel_mps": (5, 0.25),
        "rear_right_wheel_mps": (6, 0.25),
    }
    first_crossing = {}
    for name, (channel, limit) in crossing_limits.items():
        hits = np.flatnonzero(np.abs(state_error[:, channel]) > limit)
        first_crossing[name] = (
            {"elapsed_s": float(hits[0] * DT), "abs_error": float(
                abs(state_error[hits[0], channel])), "threshold": limit}
            if len(hits) else None)
    for name, values, limit in (
            ("position_xy_m", np.linalg.norm(pose_error[:, :2], axis=1), 0.10),
            ("heading_rad", np.abs(pose_error[:, 2]), 0.05)):
        hits = np.flatnonzero(values > limit)
        first_crossing[name] = (
            {"elapsed_s": float(hits[0] * DT), "abs_error": float(values[hits[0]]),
             "threshold": limit} if len(hits) else None)
    horizon_scores = {}
    for horizon_s in (0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0):
        last = min(int(round(horizon_s / DT)), len(state_error) - 1)
        if last < 1:
            continue
        horizon_scores[f"{horizon_s:g}s"] = {
            "endpoint_abs_error": {
                name: float(abs(state_error[last, index]))
                for index, name in enumerate(STATE_NAMES)},
            "cumulative_rmse": {
                name: float(np.sqrt(np.mean(state_error[:last + 1, index] ** 2)))
                for index, name in enumerate(STATE_NAMES)},
            "position_endpoint_error_m": float(np.linalg.norm(pose_error[last, :2])),
            "heading_endpoint_error_rad": float(abs(pose_error[last, 2])),
        }
    return {
        "prediction_mode": prediction_mode,
        "predicted_until_s": float(len(predicted) * DT), "failure": failure,
        "first_error_threshold_crossing": first_crossing,
        "first_training_support_exit": support_exit,
        "horizon_scores": horizon_scores,
        "state_rmse": {name: float(np.sqrt(np.mean(state_error[:, index] ** 2)))
                       for index, name in enumerate(STATE_NAMES)},
        "state_p95_abs": {name: float(np.quantile(
            np.abs(state_error[:, index]), 0.95))
            for index, name in enumerate(STATE_NAMES)},
        "position_xy_rmse_m": float(np.sqrt(np.mean(
            np.sum(pose_error[:, :2] ** 2, axis=1)))),
        "position_xy_p95_m": float(np.quantile(
            np.linalg.norm(pose_error[:, :2], axis=1), 0.95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
        "heading_p95_abs_rad": float(np.quantile(
            np.abs(pose_error[:, 2]), 0.95)),
    }


def _sample_weights_for_current_data(data: dict[str, Any]) -> np.ndarray:
    counts = Counter(data["run_id"].tolist())
    weights = np.asarray([1.0 / counts[run] for run in data["run_id"]],
                         dtype=np.float64)
    weights *= len(counts) / weights.sum()
    return weights


def _collect_dagger_rows(
        datasets: list[dict[str, Any]],
        heads: list[HistGradientBoostingRegressor]
        ) -> dict[str, Any]:
    """Collect predicted-state training examples; truth is label-only."""
    samples_by_run: dict[str, list[tuple[dict[str, Any], int, int]]] = defaultdict(list)
    for data in datasets:
        for sequence_id, (start_raw, end_raw) in enumerate(data["sequence_bounds"]):
            run_index = int(data["sequence_run_index"][sequence_id])
            run_id = str(data["run_ids"][run_index])
            if str(data["run_splits"][run_index]) != "train":
                continue
            start, end = int(start_raw), int(end_raw)
            first = start + HISTORY
            last = end - 1 - 40
            if last <= first:
                continue
            candidate_starts = np.linspace(
                first, last, num=min(8, last - first + 1), dtype=np.int64)
            samples_by_run[run_id].extend(
                (data, start, int(index)) for index in np.unique(candidate_starts))

    features, targets, run_labels = [], [], []
    episode_reports = {}
    output_channels = np.asarray(OUTPUT_INDICES, dtype=np.int32)
    for run_id, episodes in sorted(samples_by_run.items()):
        if len(episodes) > 8:
            select = np.linspace(0, len(episodes) - 1, 8, dtype=np.int64)
            episodes = [episodes[int(i)] for i in np.unique(select)]
        episode_steps = []
        for data, sequence_start, local_start in episodes:
            global_start = local_start
            history = [row.copy() for row in data["state"][
                global_start - HISTORY:global_start + 1]]
            count = 0
            for local in range(global_start, min(global_start + 40,
                                                  len(data["state"]) - 1)):
                feature = _features(np.asarray(history[-(HISTORY + 1):]))
                predicted_delta = _predict(
                    heads, feature[None, :],
                    list(range(len(FEATURE_NAMES))))[0]
                truth_next = data["state"][local + 1]
                features.append(feature)
                targets.append(truth_next[output_channels]
                               - history[-1][output_channels])
                run_labels.append(run_id)
                next_row = history[-1].copy()
                next_row[output_channels] += predicted_delta
                next_row[7:9] = truth_next[7:9]  # known exogenous commands
                if not np.isfinite(next_row).all():
                    break
                if (next_row[0] < 0.0 or next_row[0] > 16.0
                        or abs(next_row[1]) > 1.0
                        or abs(next_row[2]) > 5.0
                        or abs(next_row[3]) > 0.5236
                        or not -0.05 <= next_row[4] <= 1.05
                        or abs(next_row[5]) > 20.0
                        or abs(next_row[6]) > 20.0
                        or abs(next_row[9]) > 30.0
                        or abs(next_row[10]) > 30.0):
                    break
                history.append(next_row)
                count += 1
            episode_steps.append(count)
        episode_reports[run_id] = {
            "episodes": len(episodes),
            "mean_rollout_steps": float(np.mean(episode_steps))
            if episode_steps else 0.0,
            "dagger_samples": int(sum(episode_steps)),
        }
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    run_id = np.asarray(run_labels, dtype=str)
    if not len(y):
        raise RuntimeError("no DAgger samples were collected")
    return {"x": x, "y": y, "run_id": run_id,
            "episode_reports": episode_reports}


def _balanced_mixed_weights(base_run_ids: np.ndarray,
                            dagger_run_ids: np.ndarray,
                            dagger_fraction: float) -> np.ndarray:
    base_counts = Counter(base_run_ids.tolist())
    dagger_counts = Counter(dagger_run_ids.tolist())
    if not 0.0 < dagger_fraction < 1.0:
        raise ValueError("DAgger fraction must be between zero and one")
    base_runs, dagger_runs = sorted(base_counts), sorted(dagger_counts)
    base_w = np.asarray([
        ((1.0 - dagger_fraction) / len(base_runs) / base_counts[run])
        for run in base_run_ids], dtype=np.float64)
    dagger_w = np.asarray([
        (dagger_fraction / len(dagger_runs) / dagger_counts[run])
        for run in dagger_run_ids], dtype=np.float64)
    return np.concatenate((base_w, dagger_w))


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    if benchmark.get("frozen") is not True:
        raise ValueError("the practice benchmark must remain frozen")
    base_dataset_path = yaw._workspace_path(benchmark["dataset"])
    if _sha256(base_dataset_path) != benchmark["dataset_sha256"]:
        raise ValueError("the frozen practice dataset hash changed")
    base_train = _collect_rows(base_dataset_path, "train")
    base_validation = _collect_rows(
        base_dataset_path, "validation", VALIDATION_IDS)
    paired_train = _collect_rows(PAIRED_DATASET, "train")
    if set(np.unique(base_validation["run_id"])) != VALIDATION_IDS:
        raise ValueError("validation must contain exactly frozen r02/r03")
    training = _combine_training((base_train, paired_train))
    if set(training["runs"]) & VALIDATION_IDS:
        raise ValueError("held-out practice run leaked into the training split")

    current_features = CURRENT_FEATURE_INDICES
    history_features = list(range(len(FEATURE_NAMES)))
    weights = training["weight"]
    current_heads = _fit_heads(training["x"], training["y"], weights,
                               current_features)
    history_heads = _fit_heads(training["x"], training["y"], weights,
                               history_features)
    current_pred = _predict(current_heads, base_validation["x"],
                            current_features)
    history_pred = _predict(history_heads, base_validation["x"],
                            history_features)
    dagger = _collect_dagger_rows(
        (base_train, paired_train), history_heads)
    mixed_x = np.concatenate((training["x"], dagger["x"]))
    mixed_y = np.concatenate((training["y"], dagger["y"]))
    dagger_fractions = (0.05, 0.15)
    dagger_models = {}
    for fraction in dagger_fractions:
        mixed_weights = _balanced_mixed_weights(
            training["run_id"], dagger["run_id"], fraction)
        heads = _fit_heads(mixed_x, mixed_y, mixed_weights,
                           history_features)
        dagger_models[f"{fraction:.0%}"] = {
            "heads": heads,
            "one_step": _predict(heads, base_validation["x"], history_features),
        }
    train_support = {
        name: tuple(float(v) for v in np.quantile(
            training["x"][:, index], (0.001, 0.999)))
        for index, name in enumerate(STATE_NAMES)
    }
    recursive = {}
    for run_id in sorted(VALIDATION_IDS):
        recursive[run_id] = {
            "current_state_only": _recursive_run(
                current_heads, base_validation, run_id, current_features,
                train_support),
            "causal_history": _recursive_run(
                history_heads, base_validation, run_id, history_features,
                train_support),
        }
        for fraction, model in dagger_models.items():
            recursive[run_id][f"causal_history_dagger_{fraction}"] = _recursive_run(
                model["heads"], base_validation, run_id, history_features,
                train_support)

    baseline_report = None
    if BASELINE_YAW_REPORT.exists():
        try:
            prior = json.loads(BASELINE_YAW_REPORT.read_text(encoding="utf-8"))
            baseline_report = {
                run_id: value.get("one_step_baseline")
                for run_id, value in prior.get(
                    "unseen_practice_validation", {}).items()}
        except (OSError, json.JSONDecodeError):
            baseline_report = None
    result = {
        "study": "causal history-conditioned yaw/lateral-response teacher",
        "offline_only": True,
        "simulator_was_run": False,
        "runtime_model_changed": False,
        "raceline_changed": False,
        "test_or_final_test_samples_used_for_fit_or_selection": False,
        "training": {
            "run_ids": training["runs"],
            "run_count": len(training["runs"]),
            "samples": int(len(training["y"])),
            "samples_per_run": training["samples_per_run"],
            "dagger_samples": int(len(dagger["y"])),
            "dagger_episode_rollouts": dagger["episode_reports"],
            "data_sources": {
                "race_domain_training_dataset_sha256": base_train[
                    "dataset_sha256"],
                "paired_throttle_slew_training_dataset": str(PAIRED_DATASET),
                "paired_throttle_slew_dataset_sha256": paired_train[
                    "dataset_sha256"],
                "paired_split_used": "train only",
            },
            "weighting": "equal total sample weight per independent run",
            "model": MODEL_KWARGS,
            "features": FEATURE_NAMES,
            "targets": TARGET_NAMES,
            "causal_window_s": HISTORY * DT,
        },
        "validation": {
            "run_ids": sorted(VALIDATION_IDS),
            "current_state_only_one_step": _one_step_report(
                base_validation, current_pred),
            "causal_history_one_step": _one_step_report(
                base_validation, history_pred),
            "causal_history_dagger_one_step": {
                fraction: _one_step_report(base_validation, model["one_step"])
                for fraction, model in dagger_models.items()},
            "recursive_whole_run": recursive,
            "production_mpc_one_step_yaw_reference": baseline_report,
            "scoring_notes": [
                "Inputs use current/past body state, actuator feedback, IMU, encoder-derived wheel speeds, and current commands only.",
                "Ground-truth body state is used as an offline teacher input for one-step scoring, never as a recursive input after initialization.",
                "During recursive rollout, the model predicts body, actuator, wheel, and IMU states; recorded commands are the exogenous replay input.",
                "Only the frozen unseen practice r02/r03 runs are scored; test/final-test splits are not used.",
            ],
        },
        "provenance": {
            "benchmark_sha256": _sha256(yaw.BENCHMARK),
            "model_library_sha256": _sha256(yaw.MODEL_LIBRARY),
            "config_sha256": _sha256(yaw.CONFIG),
            "trajectory_sha256": _sha256(yaw.TRAJECTORY),
            "validation_bag_sha256": {
                name: _sha256(path) for name, path in yaw.base.BAGS.items()},
        },
        "limitations": [
            "The learned teacher is an offline diagnostic and is not wired into odometry or MPC.",
            "The two whole-run practice captures provide limited independent-run uncertainty.",
            "The recorded throttle command is an exogenous replay input; it is not an MPC-legal new topic subscription.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "training_samples": report["training"]["samples"],
        "training_runs": report["training"]["run_count"],
        "current_only_one_step": report["validation"][
            "current_state_only_one_step"]["per_target"],
        "causal_history_one_step": report["validation"][
            "causal_history_one_step"]["per_target"],
        "causal_history_dagger_one_step": {
            fraction: values["per_target"] for fraction, values in report[
                "validation"]["causal_history_dagger_one_step"].items()},
        "recursive_whole_run": report["validation"]["recursive_whole_run"],
    }, indent=2, sort_keys=True))
