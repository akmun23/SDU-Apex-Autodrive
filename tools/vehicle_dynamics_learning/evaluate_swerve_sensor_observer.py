#!/usr/bin/env python3
"""Fit and whole-run score a causal sensor-only velocity observer candidate.

The candidate learns rear-axle longitudinal/lateral velocity from legal sensor
and actuator channels plus past samples. Offline odometry is used only as a
label. Runs, not packets, define the train/validation split. This is an
observer diagnostic, not a recursive vehicle plant or a runtime integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.vehicle_dynamics_learning.structured_body_models import (
    REAR_AXLE_TO_COM_X_M,
)


SENSOR_LAGS = (0, 1, 2, 4, 8)
SPEED_EDGES_MPS = (0.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.01)
STEERING_EDGES_RAD = (0.0, 0.10, 0.20, 0.30, 0.40, 0.5241)
MODEL_PARAMETERS = {
    "max_iter": 240,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 40,
    "l2_regularization": 2.0,
    "early_stopping": False,
    "random_state": 20261006,
}


def _lagged_sensor_matrix(
    sensors: np.ndarray,
    valid: np.ndarray,
    bounds: np.ndarray,
    sequence_run_index: np.ndarray,
    target_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack only present/past sensor samples; never cross sequence/reset edges."""
    feature_blocks: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    run_indices: list[np.ndarray] = []
    row_sequence_indices: list[np.ndarray] = []
    first_lag = max(SENSOR_LAGS)
    for sequence_id, (start_raw, end_raw) in enumerate(bounds):
        start, end = int(start_raw), int(end_raw)
        indices = np.arange(start + first_lag, end, dtype=np.int64)
        if not len(indices):
            continue
        row_valid = np.ones(len(indices), dtype=bool)
        for lag in SENSOR_LAGS:
            row_valid &= valid[indices - lag]
        if target_valid is not None:
            row_valid &= target_valid[indices]
        indices = indices[row_valid]
        if not len(indices):
            continue
        feature_blocks.append(np.concatenate(
            [sensors[indices - lag] for lag in SENSOR_LAGS], axis=1))
        frame_indices.append(indices)
        run_indices.append(np.full(
            len(indices), sequence_run_index[sequence_id], dtype=np.int32))
        row_sequence_indices.append(np.full(
            len(indices), sequence_id, dtype=np.int32))
    if not feature_blocks:
        return (np.empty((0, sensors.shape[1] * len(SENSOR_LAGS)), dtype=np.float32),
                np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32))
    indices = np.concatenate(frame_indices)
    return (np.concatenate(feature_blocks).astype(np.float32, copy=False),
            indices,
            np.concatenate(run_indices),
            np.concatenate(row_sequence_indices))


def _load_frozen_checkpoint(path: Path, sensor_names: list[str]) -> dict[str, Any]:
    """Load a diagnostic checkpoint only when its causal feature contract matches."""
    checkpoint = joblib.load(path)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a model metadata mapping")
    if checkpoint.get("sensor_feature_names") != sensor_names:
        raise ValueError("checkpoint sensor feature names do not match the dataset")
    if tuple(checkpoint.get("sensor_lags", ())) != SENSOR_LAGS:
        raise ValueError("checkpoint causal sensor lags do not match this evaluator")
    models = checkpoint.get("models_u_v")
    if not isinstance(models, (list, tuple)) or len(models) != 2 or any(
            not callable(getattr(model, "predict", None)) for model in models):
        raise ValueError("checkpoint must contain exactly two fitted u/v predictors")
    return checkpoint


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def _equal_run_sample_weights(training_run_ids: np.ndarray) -> np.ndarray:
    """Give every training capture equal total weight, independent of length."""
    _, inverse, counts = np.unique(
        np.asarray(training_run_ids).astype(str), return_inverse=True,
        return_counts=True)
    if not len(counts) or np.any(counts == 0):
        raise ValueError("equal-run weighting requires at least one training row")
    return (len(inverse) / len(counts) / counts[inverse]).astype(np.float64)


def _simulator_rear_axle_targets(rigid_state: np.ndarray) -> np.ndarray:
    """Convert bridge COM body velocity to [u_rear, v_rear, yaw-rate]."""
    rigid_state = np.asarray(rigid_state, dtype=np.float32)
    if rigid_state.ndim != 2 or rigid_state.shape[1] != 13:
        raise ValueError("simulator rigid state must have 13 fields per frame")
    yaw_rate = rigid_state[:, 12]
    return np.column_stack((
        rigid_state[:, 7],
        rigid_state[:, 8] - yaw_rate * REAR_AXLE_TO_COM_X_M,
        yaw_rate,
    )).astype(np.float32)


def _run_metrics(truth: np.ndarray, prediction: np.ndarray,
                 run_indices: np.ndarray, run_ids: np.ndarray
                 ) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for run_index in sorted(set(run_indices.tolist())):
        mask = run_indices == run_index
        error = prediction[mask] - truth[mask]
        results[str(run_ids[run_index])] = {
            "samples": int(mask.sum()),
            "u_rmse_mps": _rmse(truth[mask, 0], prediction[mask, 0]),
            "v_rmse_mps": _rmse(truth[mask, 1], prediction[mask, 1]),
            "u_mae_mps": float(np.mean(np.abs(error[:, 0]))),
            "v_mae_mps": float(np.mean(np.abs(error[:, 1]))),
            "u_bias_mps": float(np.mean(error[:, 0])),
            "v_bias_mps": float(np.mean(error[:, 1])),
        }
    return results


def _run_bootstrap_ci(run_differences: list[float], seed: int = 20261006,
                      draws: int = 10000) -> list[float] | None:
    if not run_differences:
        return None
    values = np.asarray(run_differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(sample, [0.025, 0.975])]


def _regime_metrics(truth: np.ndarray, candidate: np.ndarray,
                    baseline: np.ndarray, speed: np.ndarray,
                    steering: np.ndarray, run_indices: np.ndarray,
                    speed_edges: tuple[float, ...] = SPEED_EDGES_MPS,
                    steering_edges: tuple[float, ...] = STEERING_EDGES_RAD,
                    run_ids: np.ndarray | None = None,
                    ) -> list[dict[str, Any]]:
    speed_bin = np.searchsorted(speed_edges, np.abs(speed), side="right") - 1
    steering_bin = np.searchsorted(steering_edges, np.abs(steering), side="right") - 1
    rows = []
    for si in range(len(speed_edges) - 1):
        for di in range(len(steering_edges) - 1):
            mask = (speed_bin == si) & (steering_bin == di)
            if not np.any(mask):
                continue
            per_run = {}
            for run_index in sorted(set(run_indices[mask].tolist())):
                run_mask = mask & (run_indices == run_index)
                run_name = (str(run_ids[run_index]) if run_ids is not None
                            else str(run_index))
                per_run[run_name] = {
                    "samples": int(run_mask.sum()),
                    "u_candidate_rmse_mps": _rmse(
                        truth[run_mask, 0], candidate[run_mask, 0]),
                    "u_wheel_mean_baseline_rmse_mps": _rmse(
                        truth[run_mask, 0], baseline[run_mask, 0]),
                    "v_candidate_rmse_mps": _rmse(
                        truth[run_mask, 1], candidate[run_mask, 1]),
                    "v_zero_baseline_rmse_mps": _rmse(
                        truth[run_mask, 1], baseline[run_mask, 1]),
                    "u_candidate_bias_mps": float(np.mean(
                        candidate[run_mask, 0] - truth[run_mask, 0])),
                    "v_candidate_bias_mps": float(np.mean(
                        candidate[run_mask, 1] - truth[run_mask, 1])),
                }
            rows.append({
                "speed_bin_mps": [speed_edges[si], speed_edges[si + 1]],
                "abs_steering_bin_rad": [steering_edges[di], steering_edges[di + 1]],
                "samples": int(mask.sum()),
                "independent_runs": int(len(set(run_indices[mask].tolist()))),
                "u_candidate_rmse_mps": _rmse(truth[mask, 0], candidate[mask, 0]),
                "u_wheel_mean_baseline_rmse_mps": _rmse(truth[mask, 0], baseline[mask, 0]),
                "v_candidate_rmse_mps": _rmse(truth[mask, 1], candidate[mask, 1]),
                "v_zero_baseline_rmse_mps": _rmse(truth[mask, 1], baseline[mask, 1]),
                "per_run": per_run,
            })
    return rows


def _integrated_lap_metrics(
    frame_indices: np.ndarray,
    run_indices: np.ndarray,
    lap_counts: np.ndarray,
    predicted_uv: np.ndarray,
    simulator_pose: np.ndarray,
    imu_yaw_rate: np.ndarray,
    dt_s: np.ndarray,
    run_ids: np.ndarray,
    rear_axle_to_com_x_m: float = REAR_AXLE_TO_COM_X_M,
) -> dict[str, Any]:
    """Score relative pose drift within recorded lap-counter segments.

    Each segment starts at its simulator pose solely for offline scoring. This
    isolates per-lap integration error from accumulated error across laps.
    """
    rows: list[dict[str, Any]] = []
    for run_index in sorted(set(run_indices.tolist())):
        run_mask = run_indices == run_index
        run_frames = frame_indices[run_mask]
        run_laps = lap_counts[run_mask]
        run_velocity = predicted_uv[run_mask]
        for lap in sorted(set(run_laps.tolist())):
            if lap < 0:
                continue
            local = np.flatnonzero(run_laps == lap)
            indices = run_frames[local]
            if len(indices) < 2:
                continue
            if np.any(np.diff(indices) != 1):
                continue
            pose = simulator_pose[indices].astype(np.float64)
            yaw_rate = imu_yaw_rate[indices].astype(np.float64)
            steps = dt_s[indices].astype(np.float64)
            xy = np.zeros((len(indices), 2), dtype=np.float64)
            yaw = float(pose[0, 2])
            for point in range(1, len(indices)):
                dt = float(steps[point])
                yaw_next = yaw + 0.5 * float(
                    yaw_rate[point - 1] + yaw_rate[point]) * dt
                yaw_mid = 0.5 * (yaw + yaw_next)
                u, v_rear = run_velocity[local[point - 1]]
                v = v_rear + 0.5 * float(
                    yaw_rate[point - 1] + yaw_rate[point]) * rear_axle_to_com_x_m
                xy[point] = xy[point - 1] + dt * np.asarray((
                    np.cos(yaw_mid) * u - np.sin(yaw_mid) * v,
                    np.sin(yaw_mid) * u + np.cos(yaw_mid) * v,
                ))
                yaw = yaw_next
            truth_xy = pose[:, :2] - pose[0, :2]
            error = xy - truth_xy
            run_name = str(run_ids[run_index])
            rows.append({
                "run_id": run_name,
                "lap_count": int(lap),
                "samples": int(len(indices)),
                "duration_s": float(np.sum(steps[1:])),
                "candidate_path_rmse_m": float(np.sqrt(np.mean(
                    np.sum(error * error, axis=1)))),
                "candidate_endpoint_error_m": float(np.linalg.norm(error[-1])),
                "relative_displacement_m": float(np.linalg.norm(
                    pose[-1, :2] - pose[0, :2])),
            })
    return {
        "integration": "per lap-counter segment; each segment is independently anchored at its simulator pose for offline scoring",
        "segments": rows,
    }


def _integrated_sequence_metrics(
    frame_indices: np.ndarray,
    sequence_indices: np.ndarray,
    run_indices: np.ndarray,
    sequence_bounds: np.ndarray,
    predicted_uv: np.ndarray,
    baseline_uv: np.ndarray,
    simulator_pose: np.ndarray,
    imu_yaw_rate: np.ndarray,
    dt_s: np.ndarray,
    rear_axle_to_com_x_m: float = REAR_AXLE_TO_COM_X_M,
) -> dict[str, Any]:
    """Integrate sensor-only u/v from a truth-aligned initial pose per sequence."""
    by_run: dict[int, list[dict[str, float]]] = {}
    sequence_rows: list[dict[str, float | int]] = []
    skipped_noncontiguous_sequences = 0
    for sequence_id in sorted(set(sequence_indices.tolist())):
        mask = sequence_indices == sequence_id
        indices = frame_indices[mask]
        if len(indices) < 2:
            continue
        # Contiguous packet samples are mandatory. Sequence ids are emitted
        # only inside the bounds, and this assertion guards accidental joins.
        start, end = map(int, sequence_bounds[sequence_id])
        if np.any(indices < start) or np.any(indices >= end) or np.any(np.diff(indices) != 1):
            skipped_noncontiguous_sequences += 1
            continue
        true_pose = simulator_pose[indices].astype(np.float64)
        if not np.isfinite(true_pose).all():
            raise ValueError("validation simulator pose labels are incomplete")
        yaw_rate = imu_yaw_rate[indices].astype(np.float64)
        step = dt_s[indices].astype(np.float64)
        truth_xy = true_pose[:, :2] - true_pose[0, :2]
        output: dict[str, np.ndarray] = {}
        for name, velocity in (("candidate", predicted_uv[mask]),
                               ("wheel_mean_zero_v", baseline_uv[mask])):
            xy = np.zeros((len(indices), 2), dtype=np.float64)
            yaw = float(true_pose[0, 2])
            for point in range(1, len(indices)):
                dt = float(step[point])
                yaw_next = yaw + 0.5 * float(yaw_rate[point - 1] + yaw_rate[point]) * dt
                yaw_mid = 0.5 * (yaw + yaw_next)
                u, v_rear = velocity[point - 1]
                v = v_rear + 0.5 * float(
                    yaw_rate[point - 1] + yaw_rate[point]) * rear_axle_to_com_x_m
                xy[point] = xy[point - 1] + dt * np.asarray((
                    np.cos(yaw_mid) * u - np.sin(yaw_mid) * v,
                    np.sin(yaw_mid) * u + np.cos(yaw_mid) * v,
                ))
                yaw = yaw_next
            error = xy - truth_xy
            output[name] = np.sqrt(np.sum(error * error, axis=1))
        result = {
            "candidate_path_rmse_m": float(np.sqrt(np.mean(output["candidate"] ** 2))),
            "baseline_path_rmse_m": float(np.sqrt(np.mean(output["wheel_mean_zero_v"] ** 2))),
            "candidate_endpoint_error_m": float(output["candidate"][-1]),
            "baseline_endpoint_error_m": float(output["wheel_mean_zero_v"][-1]),
            "samples": int(len(indices)),
        }
        run_index = int(run_indices[mask][0])
        by_run.setdefault(run_index, []).append(result)
        sequence_rows.append({"sequence_index": int(sequence_id), **result})
    per_run = {}
    for run_index, sequence_results in by_run.items():
        per_run[str(run_index)] = {
            "sequences": len(sequence_results),
            **{name: float(np.mean([row[name] for row in sequence_results]))
               for name in ("candidate_path_rmse_m", "baseline_path_rmse_m",
                            "candidate_endpoint_error_m", "baseline_endpoint_error_m")},
        }
    candidate_diff = [row["baseline_path_rmse_m"] - row["candidate_path_rmse_m"]
                      for row in per_run.values()]
    endpoint_diff = [row["baseline_endpoint_error_m"] - row["candidate_endpoint_error_m"]
                     for row in per_run.values()]
    return {
        "integration": "25 ms body-frame u/v integrated with measured IMU yaw rate; each sequence starts at its offline simulator pose",
        "per_run_index": per_run,
        "mean_run_path_rmse_improvement_m": float(np.mean(candidate_diff)) if candidate_diff else None,
        "run_cluster_bootstrap_path_improvement_95pct_ci_m": _run_bootstrap_ci(candidate_diff, 20261008),
        "mean_run_endpoint_error_improvement_m": float(np.mean(endpoint_diff)) if endpoint_diff else None,
        "run_cluster_bootstrap_endpoint_improvement_95pct_ci_m": _run_bootstrap_ci(endpoint_diff, 20261009),
        "sequence_count": len(sequence_rows),
        "skipped_noncontiguous_sequences": skipped_noncontiguous_sequences,
        "sequences": sequence_rows,
    }


def evaluate(dataset_path: Path, output_dir: Path,
             additional_training_datasets: tuple[Path, ...] = (),
             training_weighting: str = "sample",
             frozen_checkpoint: Path | None = None) -> dict[str, Any]:
    if training_weighting not in ("sample", "equal-run"):
        raise ValueError(f"unsupported training weighting: {training_weighting}")
    if frozen_checkpoint is not None and additional_training_datasets:
        raise ValueError("frozen-checkpoint evaluation cannot include training datasets")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        required = {
            "schema_version", "frames", "sensor_frames", "sensor_valid",
            "sequence_bounds", "sequence_run_index", "run_ids", "run_splits",
            "sensor_feature_names", "simulator_pose_xyyaw", "lap_count", "dt_s",
            "simulator_rigid_state",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"dataset missing arrays: {sorted(missing)}")
        schema_version = int(archive["schema_version"][0])
        if schema_version != 7:
            raise ValueError(f"expected schema-7 dataset, got {schema_version}")
        frames = archive["frames"].astype(np.float32, copy=False)
        sensors = archive["sensor_frames"].astype(np.float32, copy=False)
        sensor_valid = archive["sensor_valid"].astype(bool, copy=False)
        bounds = archive["sequence_bounds"].astype(np.int64, copy=False)
        sequence_run_index = archive["sequence_run_index"].astype(np.int32, copy=False)
        run_ids = archive["run_ids"].astype(str)
        run_splits = archive["run_splits"].astype(str)
        sensor_names = archive["sensor_feature_names"].astype(str).tolist()
        simulator_pose = archive["simulator_pose_xyyaw"].astype(np.float32, copy=False)
        lap_count = archive["lap_count"].astype(np.int32, copy=False)
        simulator_rigid_state = archive["simulator_rigid_state"].astype(
            np.float32, copy=False)
        dt_s = archive["dt_s"].astype(np.float32, copy=False)

    if frames.ndim != 2 or frames.shape[1] < 3:
        raise ValueError("frames must contain supervised u, v and yaw-rate labels")
    if sensors.ndim != 2 or sensors.shape[1] != len(sensor_names):
        raise ValueError("sensor feature names and matrix disagree")
    if len(frames) != len(sensors) or len(sensor_valid) != len(frames):
        raise ValueError("frame, sensor and validity arrays are misaligned")
    if simulator_rigid_state.shape != (len(frames), 13):
        raise ValueError("simulator rigid state must have 13 fields per frame")
    if not np.isfinite(sensors[sensor_valid]).all():
        raise ValueError("dataset contains non-finite valid sensor data")
    if len(sequence_run_index) != len(bounds):
        raise ValueError("sequence/run metadata lengths disagree")

    truth_valid = np.isfinite(simulator_rigid_state[:, [7, 8, 12]]).all(axis=1)
    target_all = _simulator_rear_axle_targets(simulator_rigid_state)
    x, frame_index, row_run_index, row_sequence_index = _lagged_sensor_matrix(
        sensors, sensor_valid, bounds, sequence_run_index, truth_valid)
    if not len(frame_index):
        raise ValueError("no sensor-valid rows have the required causal history")
    row_split = run_splits[row_run_index]
    train = (row_split == "train") & (np.abs(target_all[frame_index, 0]) <= 12.0)
    validation = row_split == "validation"
    if not np.any(validation):
        raise ValueError("dataset must contain at least one whole-run validation split")
    candidate = np.empty((len(frame_index), 2), dtype=np.float64)
    train_sample_count = 0
    training_samples_per_run: dict[str, int] = {}
    checkpoint_provenance: dict[str, Any] = {}
    if frozen_checkpoint is not None:
        checkpoint = _load_frozen_checkpoint(frozen_checkpoint, sensor_names)
        candidate[validation] = np.column_stack([
            model.predict(x[validation]) for model in checkpoint["models_u_v"]
        ])
        train_run_ids = sorted(map(str, checkpoint.get("training_run_ids", [])))
        source_report = frozen_checkpoint.with_name("report.json")
        if source_report.is_file():
            checkpoint_provenance = json.loads(
                source_report.read_text(encoding="utf-8"))
        training_sources = checkpoint_provenance.get("training_sources", [{
            "dataset": checkpoint.get("dataset", "unknown"),
            "run_ids": train_run_ids,
            "samples": None,
        }])
        training_samples_per_run = checkpoint_provenance.get(
            "training_samples_per_run", {})
        train_sample_count = int(checkpoint_provenance.get(
            "samples", {}).get("all_training_sources", 0))
        training_weighting = checkpoint_provenance.get(
            "training_weighting", "checkpoint")
    else:
        if not np.any(train):
            raise ValueError("model fitting requires at least one whole-run training split")
        train_x_parts = [x[train]]
        train_y_parts = [target_all[frame_index[train], :2].astype(np.float64)]
        train_group_parts = [run_ids[row_run_index[train]]]
        train_run_id_set = set(run_ids[row_run_index[train]].tolist())
        training_sources = [{
            "dataset": str(dataset_path.resolve()),
            "run_ids": sorted(train_run_id_set),
            "samples": int(train.sum()),
        }]
        for extra_path in additional_training_datasets:
            with np.load(extra_path, allow_pickle=False) as archive:
                extra_required = {
                    "schema_version", "frames", "sensor_frames", "sensor_valid",
                    "sequence_bounds", "sequence_run_index", "run_ids", "run_splits",
                    "sensor_feature_names", "simulator_rigid_state",
                }
                missing_extra = extra_required - set(archive.files)
                if missing_extra:
                    raise ValueError(
                        f"additional training dataset {extra_path} lacks arrays: "
                        f"{sorted(missing_extra)}")
                extra_schema = int(archive["schema_version"][0])
                if extra_schema not in (7, 8, 9):
                    raise ValueError(
                        f"unsupported additional training schema {extra_schema}: {extra_path}")
                extra_sensors = archive["sensor_frames"].astype(np.float32, copy=False)
                extra_valid = archive["sensor_valid"].astype(bool, copy=False)
                extra_bounds = archive["sequence_bounds"].astype(np.int64, copy=False)
                extra_sequence_run = archive["sequence_run_index"].astype(np.int32, copy=False)
                extra_run_ids = archive["run_ids"].astype(str)
                extra_splits = archive["run_splits"].astype(str)
                extra_sensor_names = archive["sensor_feature_names"].astype(str).tolist()
                extra_rigid = archive["simulator_rigid_state"].astype(
                    np.float32, copy=False)
            if extra_sensor_names != sensor_names:
                raise ValueError(
                    f"additional dataset has incompatible sensor features: {extra_path}")
            extra_targets_all = _simulator_rear_axle_targets(extra_rigid)
            extra_target_valid = np.isfinite(extra_targets_all[:, :3]).all(axis=1)
            extra_x, extra_frame_index, extra_row_run, _ = _lagged_sensor_matrix(
                extra_sensors, extra_valid, extra_bounds, extra_sequence_run,
                extra_target_valid)
            extra_train = (
                (extra_splits[extra_row_run] == "train")
                & (np.abs(extra_targets_all[extra_frame_index, 0]) <= 12.0))
            extra_train_ids = set(extra_run_ids[extra_row_run[extra_train]].tolist())
            overlap_train = sorted(train_run_id_set & extra_train_ids)
            if overlap_train:
                raise ValueError(
                    "duplicate whole-run training IDs across datasets: "
                    + ", ".join(overlap_train))
            extra_rows = int(extra_train.sum())
            if extra_rows:
                train_x_parts.append(extra_x[extra_train])
                train_y_parts.append(
                    extra_targets_all[extra_frame_index[extra_train], :2].astype(np.float64))
                train_group_parts.append(extra_run_ids[extra_row_run[extra_train]])
                train_run_id_set.update(extra_train_ids)
            training_sources.append({
                "dataset": str(extra_path.resolve()),
                "run_ids": sorted(extra_train_ids),
                "samples": extra_rows,
                "nontraining_runs_excluded_without_scoring": int(
                    len(set(extra_run_ids.tolist()) - extra_train_ids)),
            })
        train_run_ids = sorted(train_run_id_set)
        train_x = np.concatenate(train_x_parts, axis=0)
        train_y = np.concatenate(train_y_parts, axis=0)
        train_group_ids = np.concatenate(train_group_parts).astype(str)
        sample_weight = (_equal_run_sample_weights(train_group_ids)
                         if training_weighting == "equal-run" else None)
        models = []
        for axis in range(2):
            model = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
            model.fit(train_x, train_y[:, axis], sample_weight=sample_weight)
            candidate[validation, axis] = model.predict(x[validation])
            models.append(model)
        train_sample_count = int(len(train_y))
        training_samples_per_run = {
            run_id: int(np.sum(train_group_ids == run_id))
            for run_id in sorted(set(train_group_ids.tolist()))
        }

    validation_run_ids = sorted(set(run_ids[row_run_index[validation]].tolist()))
    overlap = sorted(set(train_run_ids) & set(validation_run_ids))
    if overlap:
        raise ValueError(f"train/validation run leakage: {overlap}")

    target = target_all[frame_index, :2].astype(np.float64)

    # Production-adjacent sensor baselines: mean rear-wheel surface speed for
    # forward velocity, zero lateral velocity. No truth inputs are involved.
    sensor_indices = {name: sensor_names.index(name) for name in sensor_names}
    left = sensors[frame_index, sensor_indices["rear_left_surface_mps"]]
    right = sensors[frame_index, sensor_indices["rear_right_surface_mps"]]
    baseline_all = np.column_stack((0.5 * (left + right), np.zeros(len(left))))

    y_val = target[validation]
    pred_val = candidate[validation]
    baseline_val = baseline_all[validation]
    val_run_indices = row_run_index[validation]
    val_sequence_indices = row_sequence_index[validation]
    per_run_candidate = _run_metrics(y_val, pred_val, val_run_indices, run_ids)
    per_run_baseline = _run_metrics(y_val, baseline_val, val_run_indices, run_ids)
    common_run_ids = sorted(set(per_run_candidate) & set(per_run_baseline))
    improvement_u = [
        per_run_baseline[run]["u_rmse_mps"] - per_run_candidate[run]["u_rmse_mps"]
        for run in common_run_ids]
    improvement_v = [
        per_run_baseline[run]["v_rmse_mps"] - per_run_candidate[run]["v_rmse_mps"]
        for run in common_run_ids]
    trajectory_metrics = _integrated_sequence_metrics(
        frame_index[validation], val_sequence_indices, val_run_indices,
        bounds, pred_val, baseline_val, simulator_pose,
        sensors[:, sensor_indices["imu_yaw_rate_rps"]],
        dt_s)
    trajectory_metrics["per_run_index"] = {
        str(run_ids[int(index)]): value
        for index, value in trajectory_metrics["per_run_index"].items()
    }

    checkpoint_path = (frozen_checkpoint.resolve() if frozen_checkpoint is not None
                       else output_dir / "swerve_sensor_observer.joblib")
    report = {
        "schema_version": 1,
        "model": "HistGradientBoostingRegressor with 0,25,50,100,200ms causal sensor lags",
        "purpose": "sensor-only rear-axle u/v observer diagnostic; not an offline plant simulator",
        "input_policy": "current and past encoder, IMU, actuator-feedback and command channels only; no pose/state labels or future samples",
        "target": ["u_rear_mps", "v_rear_mps"],
        "supervision_labels": "offline simulator rigid-body state only; COM body velocity converted to rear-axle v with the repository COM offset",
        "baseline": "mean rear-wheel surface speed for u; zero for v",
        "dataset": str(dataset_path.resolve()),
        "training_runs": train_run_ids,
        "checkpoint_mode": ("frozen_evaluation_only" if frozen_checkpoint is not None
                            else "fit_and_evaluate"),
        "training_weighting": training_weighting,
        "training_samples_per_run": training_samples_per_run,
        "training_sources": training_sources,
        "training_speed_scope_mps": [0.0, 12.0],
        "validation_runs": validation_run_ids,
        "samples": {
            "primary_dataset_train": int(train.sum()),
            "all_training_sources": train_sample_count,
            "validation": int(validation.sum()),
        },
        "sensor_valid_fraction": float(sensor_valid[frame_index].mean()),
        "model_parameters": MODEL_PARAMETERS,
        "whole_run_validation": {
            "candidate": {
                "u_rmse_mps": _rmse(y_val[:, 0], pred_val[:, 0]),
                "v_rmse_mps": _rmse(y_val[:, 1], pred_val[:, 1]),
                "u_mae_mps": float(np.mean(np.abs(y_val[:, 0] - pred_val[:, 0]))),
                "v_mae_mps": float(np.mean(np.abs(y_val[:, 1] - pred_val[:, 1]))),
                "per_run": per_run_candidate,
            },
            "baseline": {
                "u_rmse_mps": _rmse(y_val[:, 0], baseline_val[:, 0]),
                "v_rmse_mps": _rmse(y_val[:, 1], baseline_val[:, 1]),
                "u_mae_mps": float(np.mean(np.abs(y_val[:, 0] - baseline_val[:, 0]))),
                "v_mae_mps": float(np.mean(np.abs(y_val[:, 1] - baseline_val[:, 1]))),
                "per_run": per_run_baseline,
            },
            "baseline_minus_candidate_rmse_improvement_mps": {
                "u_mean_across_runs": float(np.mean(improvement_u)),
                "u_run_cluster_bootstrap_95pct_ci": _run_bootstrap_ci(improvement_u, 20261006),
                "v_mean_across_runs": float(np.mean(improvement_v)),
                "v_run_cluster_bootstrap_95pct_ci": _run_bootstrap_ci(improvement_v, 20261007),
            },
            "regime_metrics": _regime_metrics(
                y_val, pred_val, baseline_val,
                target_all[frame_index[validation], 0],
                sensors[frame_index[validation], sensor_indices["steering_feedback_rad"]],
                val_run_indices, run_ids=run_ids),
            "integrated_lap_segments": _integrated_lap_metrics(
                frame_index[validation], val_run_indices,
                lap_count[frame_index[validation]], pred_val,
                simulator_pose, sensors[:, sensor_indices["imu_yaw_rate_rps"]],
                dt_s, run_ids),
            "integrated_swerve_sequences": trajectory_metrics,
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "validation_use": (
            "The saved checkpoint was evaluated without refitting on the whole-run captures listed above."
            if frozen_checkpoint is not None else
            "Whole-run held-out evaluation includes the validation captures listed above."),
        "interpretation_limit": (
            "This is a causal sensor-only u/v observer diagnostic, not a recursive plant simulator. "
            "Validation is diagnostic and is not an untouched final-test result."),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if frozen_checkpoint is None:
        joblib.dump({
            "models_u_v": models,
            "sensor_feature_names": sensor_names,
            "sensor_lags": SENSOR_LAGS,
            "parameters": MODEL_PARAMETERS,
            "training_run_ids": train_run_ids,
            "dataset": str(dataset_path.resolve()),
            "runtime_status": "diagnostic_only_not_integrated",
        }, checkpoint_path)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-training-dataset", type=Path, action="append",
                        default=[], help="extra dataset; only its whole-run train split is fit")
    parser.add_argument("--training-weighting", choices=("sample", "equal-run"),
                        default="sample",
                        help="sample frequency or equal total weight for each training run")
    parser.add_argument("--frozen-checkpoint", type=Path,
                        help="evaluate this saved diagnostic model without fitting or changing it")
    args = parser.parse_args()
    report = evaluate(args.dataset, args.output_dir,
                      tuple(args.additional_training_dataset),
                      args.training_weighting, args.frozen_checkpoint)
    summary = report["whole_run_validation"]
    print(json.dumps({
        "validation_runs": report["validation_runs"],
        "candidate_u_v_rmse_mps": [summary["candidate"]["u_rmse_mps"],
                                    summary["candidate"]["v_rmse_mps"]],
        "baseline_u_v_rmse_mps": [summary["baseline"]["u_rmse_mps"],
                                   summary["baseline"]["v_rmse_mps"]],
        "report": str(args.output_dir / "report.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
