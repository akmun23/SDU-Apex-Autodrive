#!/usr/bin/env python3
"""Fit local one-step yaw experts using competition-observable inputs only.

Simulator rigid-body yaw rate is the supervised target and evaluation truth.
It is never an input or regime selector. Runtime-style expert selection uses
causal rear-wheel surface speed, measured steering feedback, and current/past
command/feedback event history. The bundle is research-only; it is not wired
into odometry or MPC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import fit_fullband_yaw_regime_atlas as admission
    import train_yaw_multihorizon_teacher as teacher
    import yaw_source_packet_alignment as packet_alignment
except ModuleNotFoundError:
    from tools.racing.specialists import fit_fullband_yaw_regime_atlas as admission
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher
    from tools.racing.specialists import yaw_source_packet_alignment as packet_alignment


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "sensor_only_yaw_regime_atlas_command_age_v4")
DT_S = 0.025
DEFAULT_HISTORY_LAGS = (0, 1, 2, 4)  # Current, 25, 50, and 100 ms.
HISTORY_LAGS = DEFAULT_HISTORY_LAGS
SPEED_BIN_MPS = 0.5
STEERING_BIN_RAD = 0.025
SPEED_CENTERS = np.arange(0.25, 12.0, SPEED_BIN_MPS)
STEERING_CENTERS = np.arange(-0.525, 0.525 + 1.0e-9, STEERING_BIN_RAD)
MIN_EXPERT_SAMPLES = 80
MIN_EXPERT_RUNS = 2
MIN_SAMPLES_PER_RUN = 8
EVENTS = ("hold", "turn_in", "unwind", "reversal")
EVENT_DEFINITIONS = ("feedback_delta", "command_intent")
OBSERVATION_NAMES = (
    "steering_feedback_rad", "throttle_feedback_norm",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps",
    "steering_command_rad", "throttle_command_norm",
    "imu_roll_rad", "imu_roll_rate_rps",
)
DERIVED_NAMES = (
    "rear_wheel_mean_mps", "rear_wheel_split_mps",
    "rear_wheel_mean_rate_mps2", "steering_feedback_rate_radps",
    "throttle_feedback_rate_per_s", "steering_command_gap_rad",
    "throttle_command_gap_norm", "imu_yaw_rate_change_radps",
    "steering_command_age_s", "steering_feedback_age_s",
    "throttle_command_age_s", "throttle_feedback_age_s",
    "steering_command_feedback_age_gap_s",
    "throttle_command_feedback_age_gap_s",
)
AGE_WINDOW_STEPS = 16  # 400 ms at the fixed 25-ms packet grid.
AGE_CHANGE_THRESHOLDS = (0.005, 0.005, 0.005, 0.005)
ESTIMATOR = {
    "n_estimators": 48,
    "max_depth": 10,
    "min_samples_leaf": 6,
    "max_features": 0.8,
    "random_state": 20261008,
}


def _read_sensor_run(series: admission.RunSeries,
                     history_lags: tuple[int, ...] = HISTORY_LAGS,
                     exact_packet_imu: bool = False) -> dict[str, Any]:
    """Read sensor/attitude channels for a run already admitted by the split gate."""
    path = ROOT / series.source
    with np.load(path, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str).tolist()
        run_splits = archive["run_splits"].astype(str).tolist()
        if series.run_id not in run_ids:
            raise ValueError(f"run provenance mismatch: {series.run_id}")
        run_index = run_ids.index(series.run_id)
        if run_splits[run_index] != series.split:
            raise ValueError(f"run split changed after admission: {series.run_id}")
        required = (
            "sensor_frames", "sensor_valid", "imu_attitude_frames",
            "imu_attitude_valid", "simulator_rigid_state",
            "sequence_bounds", "sequence_run_index", "dt_s",
            "sample_time_ns",
        )
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise ValueError(f"{series.run_id}: missing arrays {missing}")
        sensors = np.asarray(archive["sensor_frames"], dtype=np.float32)
        sensor_valid = np.asarray(archive["sensor_valid"], dtype=bool)
        attitude = np.asarray(archive["imu_attitude_frames"], dtype=np.float32)
        attitude_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
        rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float32)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        sequence_run_index = np.asarray(
            archive["sequence_run_index"], dtype=np.int64)
        sample_time_ns = np.asarray(archive["sample_time_ns"], dtype=np.int64)
        dt = np.asarray(archive["dt_s"], dtype=np.float32)
        if (sensors.ndim != 2 or sensors.shape[1] < 9
                or attitude.ndim != 2 or attitude.shape[1] < 3
                or rigid.shape != (len(sensors), 13)
                or any(len(values) != len(sensors) for values in
                       (sensor_valid, attitude_valid, dt))
                or bounds.ndim != 2 or bounds.shape[1] != 2
                or len(sequence_run_index) != len(bounds)):
            raise ValueError(f"{series.run_id}: invalid sensor archive schema")
        if not np.allclose(dt, DT_S, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"{series.run_id}: archive is not on the 25-ms grid")

        source_alignment = None
        if exact_packet_imu:
            manifest_path = path.with_name("manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_rows = [row for row in manifest.get("runs", [])
                        if row.get("run_id") == series.run_id
                        and row.get("effective_split") == series.split]
            if len(run_rows) != 1:
                raise ValueError(f"{series.run_id}: raw bag provenance is ambiguous")
            bag_path = ROOT / run_rows[0]["bag"]
            exact_by_receipt, raw_packet_count = (
                packet_alignment.exact_imu_by_odom_receipt(bag_path))
            exact_count = 0
            changed_yaw = 0
            changed_yaw_over_0p1 = 0
            missing_count = 0
            for sequence_id in np.flatnonzero(sequence_run_index == run_index):
                begin, end = map(int, bounds[int(sequence_id)])
                for frame_index in range(begin, end):
                    values = exact_by_receipt.get(int(sample_time_ns[frame_index]))
                    if values is None:
                        missing_count += 1
                        sensor_valid[frame_index] = False
                        attitude_valid[frame_index] = False
                        continue
                    exact_count += 1
                    yaw_delta = abs(float(sensors[frame_index, 6]) - values[2])
                    changed_yaw += int(yaw_delta > 1.0e-6)
                    changed_yaw_over_0p1 += int(yaw_delta > 0.1)
                    sensors[frame_index, 4:7] = values[:3]
                    attitude[frame_index, :] = values[3:]
            if not exact_count:
                raise ValueError(f"{series.run_id}: no exact source-stamped IMU joins")
            source_alignment = {
                "method": "exact_imu_header_stamp_equals_odometry_header_stamp",
                "archive_samples_for_run": int(exact_count + missing_count),
                "raw_odometry_packets": int(raw_packet_count),
                "exact_packet_imu_matches": int(exact_count),
                "missing_exact_packet_imu": int(missing_count),
                "archived_yaw_values_changed": int(changed_yaw),
                "archived_yaw_values_changed_by_more_than_0p1_radps": int(
                    changed_yaw_over_0p1),
            }

        rows = []
        for sequence_id in np.flatnonzero(sequence_run_index == run_index):
            begin, end = map(int, bounds[int(sequence_id)])
            if end - begin <= max(history_lags) + 1:
                continue
            rows.append((sensors[begin:end], sensor_valid[begin:end],
                         attitude[begin:end], attitude_valid[begin:end],
                         rigid[begin:end]))
    if not rows:
        raise ValueError(f"{series.run_id}: no usable sensor sequences")
    return {"run_id": series.run_id, "split": series.split,
            "source": series.source, "sequences": rows,
            "source_alignment": source_alignment}


def _event(sensors: np.ndarray, index: int,
           definition: str = "feedback_delta") -> str:
    """Classify a steering phase using current and past command/feedback."""
    delta = float(sensors[index, 0])
    previous_delta = float(sensors[index - 1, 0])
    command = float(sensors[index, 7])
    previous_command = float(sensors[index - 1, 7])
    if definition == "command_intent":
        command_two_steps_back = float(sensors[index - 2, 7])
        delta_two_steps_back = float(sensors[index - 2, 0])
        command_crossed_sign = (
            abs(command) >= 0.008 and abs(command_two_steps_back) >= 0.008
            and command * command_two_steps_back < 0.0)
        command_opposes_feedback = (
            abs(command) >= 0.02 and abs(delta) >= 0.025
            and command * delta < 0.0)
        if (command_crossed_sign or command_opposes_feedback
                or delta * previous_delta < 0.0):
            return "reversal"
        if abs(command) <= abs(delta) - 0.02:
            return "unwind"
        if abs(command) >= abs(delta) + 0.02 and command * delta >= 0.0:
            return "turn_in"
        if abs(delta) > abs(delta_two_steps_back) + 0.003:
            return "turn_in"
        if abs(delta) < abs(delta_two_steps_back) - 0.003:
            return "unwind"
        return "hold"
    if definition != "feedback_delta":
        raise ValueError(f"unknown event definition: {definition}")
    command_reversed = (
        abs(command) >= 0.025 and abs(previous_command) >= 0.025
        and command * previous_command < 0.0)
    command_opposes_feedback = (
        abs(command) >= 0.025 and abs(delta) >= 0.025
        and command * delta < 0.0)
    feedback_crossed_zero = delta * previous_delta < 0.0
    if command_reversed or command_opposes_feedback or feedback_crossed_zero:
        return "reversal"
    current_magnitude = abs(delta)
    previous_magnitude = abs(previous_delta)
    if current_magnitude > previous_magnitude + 0.001:
        return "turn_in"
    if current_magnitude < previous_magnitude - 0.001:
        return "unwind"
    return "hold"


def _age_since_change(values: np.ndarray, threshold: float) -> np.ndarray:
    """Causal time since a signal last moved beyond its change threshold."""
    ages = np.empty(len(values), dtype=np.float32)
    last_change = 0
    ages[0] = 0.0
    for index in range(1, len(values)):
        if abs(float(values[index]) - float(values[index - 1])) > threshold:
            last_change = index
        ages[index] = min(index - last_change, AGE_WINDOW_STEPS) * DT_S
    return ages


def _rows(run: dict[str, Any], event_definition: str,
          history_lags: tuple[int, ...] = HISTORY_LAGS) -> dict[str, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[float] = []
    wheel_speed: list[float] = []
    steering: list[float] = []
    gt_speed: list[float] = []
    gt_yaw_current: list[float] = []
    current_imu_yaw: list[float] = []
    events: list[str] = []
    run_ids: list[str] = []
    for sensors, sensor_valid, attitude, attitude_valid, rigid in run["sequences"]:
        if (not np.isfinite(sensors).all() or not np.isfinite(attitude).all()
                or not np.isfinite(rigid).all()):
            continue
        signal_ages = (
            _age_since_change(sensors[:, 7], AGE_CHANGE_THRESHOLDS[0]),
            _age_since_change(sensors[:, 0], AGE_CHANGE_THRESHOLDS[1]),
            _age_since_change(sensors[:, 8], AGE_CHANGE_THRESHOLDS[2]),
            _age_since_change(sensors[:, 1], AGE_CHANGE_THRESHOLDS[3]),
        )
        for k in range(max(history_lags), len(sensors) - 1):
            if (not all(sensor_valid[k - lag] for lag in history_lags)
                    or not all(attitude_valid[k - lag] for lag in history_lags)):
                continue
            history = np.concatenate([
                np.concatenate((sensors[k - lag, :9],
                                attitude[k - lag, (0, 2)]))
                for lag in history_lags
            ]).astype(np.float32, copy=False)
            left = float(sensors[k, 2])
            right = float(sensors[k, 3])
            wheel_mean = 0.5 * (left + right)
            previous_mean = 0.5 * float(sensors[k - 1, 2] + sensors[k - 1, 3])
            derived = np.asarray((
                wheel_mean,
                left - right,
                (wheel_mean - previous_mean) / DT_S,
                (float(sensors[k, 0]) - float(sensors[k - 1, 0])) / DT_S,
                (float(sensors[k, 1]) - float(sensors[k - 1, 1])) / DT_S,
                float(sensors[k, 7] - sensors[k, 0]),
                float(sensors[k, 8] - sensors[k, 1]),
                float(sensors[k, 6] - sensors[k - 1, 6]),
                float(signal_ages[0][k]),
                float(signal_ages[1][k]),
                float(signal_ages[2][k]),
                float(signal_ages[3][k]),
                float(signal_ages[1][k] - signal_ages[0][k]),
                float(signal_ages[3][k] - signal_ages[2][k]),
            ), dtype=np.float32)
            speed_cell = int(wheel_mean // SPEED_BIN_MPS)
            steering_value = float(sensors[k, 0])
            if (speed_cell < 0 or speed_cell >= len(SPEED_CENTERS)
                    or not math.isfinite(steering_value)
                    or steering_value < STEERING_CENTERS[0] - 0.0125
                    or steering_value > STEERING_CENTERS[-1] + 0.0125):
                continue
            steering_cell = int(np.clip(round(
                (steering_value - STEERING_CENTERS[0]) / STEERING_BIN_RAD),
                0, len(STEERING_CENTERS) - 1))
            gt_yaw_next = float(rigid[k + 1, 12])
            imu_yaw_now = float(sensors[k, 6])
            actual_speed = float(np.hypot(rigid[k, 7], rigid[k, 8]))
            current_gt_yaw = float(rigid[k, 12])
            features.append(np.concatenate((history, derived)))
            targets.append(gt_yaw_next - imu_yaw_now)
            wheel_speed.append(wheel_mean)
            steering.append(steering_value)
            gt_speed.append(actual_speed)
            gt_yaw_current.append(current_gt_yaw)
            current_imu_yaw.append(imu_yaw_now)
            events.append(_event(sensors, k, event_definition))
            run_ids.append(run["run_id"])
    if not features:
        raise ValueError(f"{run['run_id']}: no valid one-step examples")
    return {
        "x": np.asarray(features, dtype=np.float32),
        "residual": np.asarray(targets, dtype=np.float32),
        "wheel_speed": np.asarray(wheel_speed, dtype=np.float32),
        "steering": np.asarray(steering, dtype=np.float32),
        "gt_speed": np.asarray(gt_speed, dtype=np.float32),
        "gt_yaw_current": np.asarray(gt_yaw_current, dtype=np.float32),
        "imu_yaw": np.asarray(current_imu_yaw, dtype=np.float32),
        "event": np.asarray(events, dtype="U16"),
        "run_id": np.asarray(run_ids, dtype="U128"),
    }


def _metrics(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return {"samples": 0}
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "p99_abs_radps": float(np.quantile(absolute, 0.99)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
        "fraction_abs_error_below_0p05": float(np.mean(absolute < 0.05)),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _balanced_weights(run_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(run_ids.astype(str), return_counts=True)
    count_by_id = dict(zip(ids.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_id[str(run_id)] for run_id in run_ids])
    return weights * (len(ids) / weights.sum())


def _fit(x: np.ndarray, y: np.ndarray, run_ids: np.ndarray,
         n_jobs: int) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(**ESTIMATOR, n_jobs=n_jobs)
    model.fit(x, y, sample_weight=_balanced_weights(run_ids))
    return model


def _speed_cell(wheel_speed: np.ndarray) -> np.ndarray:
    return np.floor(wheel_speed / SPEED_BIN_MPS).astype(np.int16)


def _steer_cell(steering: np.ndarray) -> np.ndarray:
    return np.clip(np.rint((steering - STEERING_CENTERS[0]) /
                           STEERING_BIN_RAD), 0,
                   len(STEERING_CENTERS) - 1).astype(np.int16)


def _macro_rmse_deltas(a: np.ndarray, b: np.ndarray, run_ids: np.ndarray
                       ) -> dict[str, Any]:
    delta_by_run = []
    for run_id in sorted(set(run_ids.astype(str))):
        mask = run_ids.astype(str) == run_id
        if np.any(mask):
            delta_by_run.append(float(np.sqrt(np.mean(a[mask] ** 2))
                                    - np.sqrt(np.mean(b[mask] ** 2))))
    if not delta_by_run:
        return {"independent_runs": 0}
    values = np.asarray(delta_by_run, dtype=np.float64)
    rng = np.random.default_rng(20261008)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "independent_runs": int(len(values)),
        "mean_run_rmse_delta_local_minus_global_radps": float(values.mean()),
        "median_run_rmse_delta_local_minus_global_radps": float(np.median(values)),
        "runs_local_better": int(np.count_nonzero(values < 0.0)),
        "runs_local_worse": int(np.count_nonzero(values > 0.0)),
        "bootstrap_95pct_ci": [float(np.quantile(draws, 0.025)),
                               float(np.quantile(draws, 0.975))],
    }


def _alignment_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [run["source_alignment"] for run in runs
            if run.get("source_alignment") is not None]
    if not rows:
        return {"mode": "receipt_causal"}
    keys = ("archive_samples_for_run", "raw_odometry_packets",
            "exact_packet_imu_matches", "missing_exact_packet_imu",
            "archived_yaw_values_changed",
            "archived_yaw_values_changed_by_more_than_0p1_radps")
    return {
        "mode": "exact_packet_source_stamp",
        "runs": len(rows),
        **{key: int(sum(row[key] for row in rows)) for key in keys},
        "per_run": {run["run_id"]: run["source_alignment"]
                    for run in runs if run.get("source_alignment") is not None},
    }


def fit(output: Path, n_jobs: int,
        event_definition: str = "feedback_delta",
        history_lags: tuple[int, ...] = DEFAULT_HISTORY_LAGS,
        exact_packet_imu: bool = False) -> dict[str, Any]:
    if event_definition not in EVENT_DEFINITIONS:
        raise ValueError(f"unknown event definition: {event_definition}")
    if (not history_lags or history_lags[0] != 0
            or any(lag < 0 for lag in history_lags)
            or tuple(sorted(set(history_lags))) != history_lags):
        raise ValueError("history lags must be unique, increasing, nonnegative, and start at 0")
    output = output.resolve()
    # This loader adds clean mixed train/validation archives only after checking
    # every run's split; it rejects any archive that mentions test/final-test.
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    runs = [_read_sensor_run(series, history_lags, exact_packet_imu)
            for series in admitted]
    train_runs = [run for run in runs if run["split"] == "train"]
    validation_runs = [run for run in runs if run["split"] == "validation"]
    train = [_rows(run, event_definition, history_lags) for run in train_runs]
    validation = [_rows(run, event_definition, history_lags) for run in validation_runs]
    feature_names = tuple(
        f"{name}_lag{lag * 25}ms"
        for lag in history_lags for name in OBSERVATION_NAMES
    ) + DERIVED_NAMES

    def combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {key: np.concatenate([part[key] for part in parts], axis=0)
                for key in parts[0]}

    tr = combine(train)
    va = combine(validation)
    train_speed_cell = _speed_cell(tr["wheel_speed"])
    train_steer_cell = _steer_cell(tr["steering"])
    valid_speed_cell = _speed_cell(va["wheel_speed"])
    valid_steer_cell = _steer_cell(va["steering"])
    training_models: dict[tuple[int, int, str], ExtraTreesRegressor] = {}
    support_rows: dict[tuple[int, int, str], dict[str, Any]] = {}

    global_event_models: dict[str, ExtraTreesRegressor] = {}
    for event in EVENTS:
        mask = tr["event"] == event
        run_count = len(set(tr["run_id"][mask].astype(str)))
        if np.count_nonzero(mask) >= MIN_EXPERT_SAMPLES and run_count >= MIN_EXPERT_RUNS:
            global_event_models[event] = _fit(
                tr["x"][mask], tr["residual"][mask], tr["run_id"][mask], n_jobs)

    train_indices: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for row_index, (speed_cell, steer_cell, event) in enumerate(zip(
            train_speed_cell, train_steer_cell, tr["event"])):
        train_indices[(int(speed_cell), int(steer_cell), str(event))].append(
            row_index)
    # Partition rows once; repeated full-dataset masks make a local atlas
    # needlessly quadratic in the number of occupied cells.
    for expert_index, key in enumerate(sorted(train_indices), start=1):
        s, d, event = key
        indices = np.asarray(train_indices[key], dtype=np.int64)
        ids, counts = np.unique(tr["run_id"][indices].astype(str), return_counts=True)
        eligible_ids = ids[counts >= MIN_SAMPLES_PER_RUN]
        fit_indices = indices[np.isin(
            tr["run_id"][indices].astype(str), eligible_ids)]
        n = int(len(fit_indices))
        if n < MIN_EXPERT_SAMPLES or len(eligible_ids) < MIN_EXPERT_RUNS:
            continue
        key_model = _fit(tr["x"][fit_indices], tr["residual"][fit_indices],
                         tr["run_id"][fit_indices], n_jobs=1)
        training_models[key] = key_model
        support_rows[key] = {
            "wheel_speed_center_mps": float(SPEED_CENTERS[s]),
            "steering_center_rad": float(STEERING_CENTERS[d]),
            "event": event,
            "training_samples": n,
            "training_runs": len(eligible_ids),
        }
        if expert_index % 100 == 0:
            print(f"fitted {expert_index}/{len(train_indices)} occupied local regimes",
                  flush=True)

    global_prediction = np.full(len(va["x"]), np.nan, dtype=np.float32)
    for event, model in global_event_models.items():
        mask = va["event"] == event
        if np.any(mask):
            global_prediction[mask] = model.predict(va["x"][mask]).astype(np.float32)
    local_prediction = np.full(len(va["x"]), np.nan, dtype=np.float32)
    local_support = np.zeros(len(va["x"]), dtype=bool)
    validation_indices: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for row_index, (speed_cell, steer_cell, event) in enumerate(zip(
            valid_speed_cell, valid_steer_cell, va["event"])):
        validation_indices[(int(speed_cell), int(steer_cell), str(event))].append(
            row_index)
    for key, model in training_models.items():
        indices = np.asarray(validation_indices.get(key, ()), dtype=np.int64)
        if len(indices) == 0:
            continue
        local_prediction[indices] = model.predict(va["x"][indices]).astype(np.float32)
        local_support[indices] = True

    baseline_prediction = np.zeros(len(va["x"]), dtype=np.float32)
    baseline_prediction[:] = 0.0  # Zero residual means persist current IMU yaw.
    global_error = global_prediction - va["residual"]
    local_error = local_prediction - va["residual"]
    persistence_error = baseline_prediction - va["residual"]
    paired = local_support & np.isfinite(global_prediction)
    actual_speed_cell = np.clip(np.floor(va["gt_speed"] / SPEED_BIN_MPS),
                                0, len(SPEED_CENTERS) - 1).astype(np.int16)
    actual_steer_cell = _steer_cell(va["steering"])

    summaries: dict[str, Any] = {}
    for name, pred, mask in (
            ("imu_persistence", np.zeros_like(va["residual"]), np.ones(len(local_support), dtype=bool)),
            ("global_event_expert", global_prediction, np.isfinite(global_prediction)),
            ("local_speed_steering_event_expert", local_prediction, local_support)):
        error = pred[mask] - va["residual"][mask]
        summaries[name] = {"coverage_fraction": float(np.mean(mask)),
                           **_metrics(error)}
    summaries["same_local_support"] = {
        "local": _metrics(local_error[paired]),
        "global_event": _metrics(global_error[paired]),
        "imu_persistence": _metrics(persistence_error[paired]),
        "run_cluster_comparison": _macro_rmse_deltas(
            local_error[paired], global_error[paired], va["run_id"][paired]),
    }

    per_event: dict[str, Any] = {}
    for event in EVENTS:
        mask = va["event"] == event
        local_mask = mask & local_support
        global_mask = mask & np.isfinite(global_prediction)
        per_event[event] = {
            "validation_rows": int(np.count_nonzero(mask)),
            "local_coverage_fraction": float(np.mean(local_support[mask])) if np.any(mask) else 0.0,
            "local": _metrics(local_error[local_mask]),
            "global_event": _metrics(global_error[global_mask]),
            "imu_persistence": _metrics(persistence_error[mask]),
            "same_local_support_local": _metrics(local_error[local_mask]),
            "same_local_support_global": _metrics(global_error[local_mask]),
        }

    per_gt_cell: list[dict[str, Any]] = []
    for s in range(len(SPEED_CENTERS)):
        for d in range(len(STEERING_CENTERS)):
            mask = (actual_speed_cell == s) & (actual_steer_cell == d)
            if np.count_nonzero(mask) < 20:
                continue
            local_mask = mask & local_support
            global_mask = mask & np.isfinite(global_prediction)
            per_gt_cell.append({
                "gt_speed_center_mps": float(SPEED_CENTERS[s]),
                "physical_steering_center_rad": float(STEERING_CENTERS[d]),
                "validation_samples": int(np.count_nonzero(mask)),
                "local_coverage_fraction": float(np.mean(local_support[mask])),
                "local": _metrics(local_error[local_mask]),
                "global_event": _metrics(global_error[global_mask]),
                "imu_persistence": _metrics(persistence_error[mask]),
            })
    gt_cells_with_data = len(per_gt_cell)
    gt_cells_local_support = sum(row["local"]["samples"] > 0 for row in per_gt_cell)
    failing_cells = [row for row in per_gt_cell
                     if row["local"]["samples"] >= 20
                     and (row["local"].get("p95_abs_radps", 0.0) > 0.1
                          or row["local"]["fraction_abs_error_below_0p1"] < 0.95)]

    worst = []
    indices = np.flatnonzero(local_support)
    order = indices[np.argsort(np.abs(local_error[indices]))[::-1][:100]]
    for i in order:
        s, d = int(valid_speed_cell[i]), int(valid_steer_cell[i])
        worst.append({
            "run_id": str(va["run_id"][i]),
            "gt_speed_mps": float(va["gt_speed"][i]),
            "wheel_speed_proxy_mps": float(va["wheel_speed"][i]),
            "steering_feedback_rad": float(va["steering"][i]),
            "event": str(va["event"][i]),
            "expert_wheel_speed_center_mps": float(SPEED_CENTERS[s]),
            "expert_steering_center_rad": float(STEERING_CENTERS[d]),
            "imu_yaw_rate_radps": float(va["imu_yaw"][i]),
            "gt_next_yaw_rate_radps": float(va["residual"][i] + va["imu_yaw"][i]),
            "predicted_next_yaw_rate_radps": float(local_prediction[i] + va["imu_yaw"][i]),
            "absolute_error_radps": float(abs(local_error[i])),
        })

    report = {
        "title": ("Competition-observable local one-step yaw response experts"
                  + (" with exact-packet IMU alignment" if exact_packet_imu else "")),
        "target": "simulator-truth yaw_rate[k+1] minus current IMU yaw_rate[k]",
        "sample_period_s": DT_S,
        "horizon_ms": 25,
        "input_contract": {
            "features": list(feature_names),
            "sensor_history_lags_ms": [lag * 25 for lag in history_lags],
            "actuator_change_age": {
                "window_ms": AGE_WINDOW_STEPS * 25,
                "change_thresholds": {
                    "steering_command_rad": AGE_CHANGE_THRESHOLDS[0],
                    "steering_feedback_rad": AGE_CHANGE_THRESHOLDS[1],
                    "throttle_command_norm": AGE_CHANGE_THRESHOLDS[2],
                    "throttle_feedback_norm": AGE_CHANGE_THRESHOLDS[3],
                },
                "timebase": "fixed 25-ms packet sequence; receipt timestamps are not used",
            },
            "imu_alignment": (
                "IMU acceleration, yaw, roll/pitch and body gyro are joined to the exact odometry packet source stamp"
                if exact_packet_imu else
                "latest IMU received no later than the odometry callback receipt time"),
            "future_truth_or_sensor_inputs": False,
            "simulator_truth_used_only_as_target_and_error": True,
            "expert_selection": "current measured rear-wheel mean speed, steering feedback, and causal steering event",
            "event_definition": event_definition,
            "event_rules": {
                "feedback_delta": {
                    "reversal": "command sign changed, command opposes measured steering, or feedback crossed zero",
                    "turn_in": "measured steering magnitude increased over one 25-ms sample",
                    "unwind": "measured steering magnitude decreased over one 25-ms sample",
                    "hold": "otherwise",
                },
                "command_intent": {
                    "reversal": "command crossed sign over 50 ms, sufficiently opposes measured steering, or feedback crossed zero",
                    "turn_in": "command target exceeds feedback magnitude by 0.02 rad, otherwise feedback magnitude rose over 50 ms",
                    "unwind": "command target is at least 0.02 rad closer to center than feedback, otherwise feedback magnitude fell over 50 ms",
                    "hold": "otherwise",
                },
            },
            "limitations": [
                "Wheel speed is an observable speed proxy and may be biased by wheelspin; GT speed is used only for stratified scoring.",
                "The empirical support is not a full Cartesian speed/steering grid and does not imply physically feasible combinations.",
                "The model is research-only and has not been integrated or tested in MPC/odometry.",
            ],
        },
        "training": {
            "runs": len(train_runs), "examples": int(len(tr["x"])),
            "expert_count": len(training_models),
            "expert_min_samples": MIN_EXPERT_SAMPLES,
            "expert_min_independent_runs": MIN_EXPERT_RUNS,
            "per_expert_minimum_samples_per_run": MIN_SAMPLES_PER_RUN,
            "events": {event: int(np.count_nonzero(tr["event"] == event))
                       for event in EVENTS},
            "examples_by_truth_speed_0p5_mps": np.bincount(
                np.clip(np.floor(np.minimum(tr["gt_speed"], 11.999) /
                                 SPEED_BIN_MPS).astype(int), 0, 23),
                minlength=24).tolist(),
            "imu_packet_alignment": _alignment_summary(train_runs),
        },
        "validation": {
            "runs": len(validation_runs), "examples": int(len(va["x"])),
            "actual_speed_max_mps": float(np.max(va["gt_speed"])),
            "abs_steering_max_rad": float(np.max(np.abs(va["steering"]))),
            "metrics": summaries,
            "by_event": per_event,
            "truth_speed_steering_cells_with_at_least_20_rows": gt_cells_with_data,
            "truth_cells_with_local_expert_support": int(gt_cells_local_support),
            "truth_cells_failing_p95_or_95pct_within_0p1": len(failing_cells),
            "worst_supported_truth_cells": sorted(
                failing_cells,
                key=lambda row: (row["local"].get("p95_abs_radps", 0.0),
                                 row["validation_samples"]),
                reverse=True)[:100],
            "worst_100_local_expert_errors": worst,
            "imu_packet_alignment": _alignment_summary(validation_runs),
        },
        "model": {
            "family": "ExtraTreesRegressor; one pooled event comparator plus local wheel-speed × signed-steering × event experts",
            "parameters": {**ESTIMATOR, "n_jobs": n_jobs},
            "run_balanced_training_weights": True,
            "train_validation_run_split": True,
            "test_final_test_arrays_read": False,
            "source_audit": source_audit,
            "event_definition": event_definition,
            "imu_alignment_mode": ("exact_packet_source_stamp"
                                   if exact_packet_imu else "receipt_causal"),
            "local_expert_support": list(support_rows.values()),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "sensor_only_yaw_regime_atlas_report.json"
    bundle_path = output / "sensor_only_yaw_regime_atlas.joblib"
    bundle = {
        "contract": report["input_contract"],
        "estimator": report["model"]["parameters"],
        "global_event_models": global_event_models,
        "local_expert_models": training_models,
        "local_expert_support": support_rows,
        "speed_centers_mps": SPEED_CENTERS,
        "steering_centers_rad": STEERING_CENTERS,
    }
    joblib.dump(bundle, bundle_path, compress=3)
    report["artifacts"] = {
        "report": str(report_path.relative_to(ROOT)),
        "model_bundle": str(bundle_path.relative_to(ROOT)),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "report": str(report_path.relative_to(ROOT)),
        "model_bundle": str(bundle_path.relative_to(ROOT)),
        "training_runs": len(train_runs), "validation_runs": len(validation_runs),
        "training_examples": int(len(tr["x"])),
        "validation_examples": int(len(va["x"])),
        "local_experts": len(training_models),
        "local_coverage": summaries["local_speed_steering_event_expert"]["coverage_fraction"],
        "local_rmse": summaries["local_speed_steering_event_expert"]["rmse_radps"],
        "local_max": summaries["local_speed_steering_event_expert"]["max_abs_radps"],
    }, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int,
                        default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--event-definition", choices=EVENT_DEFINITIONS,
                        default="command_intent")
    parser.add_argument("--history-lags", default="0,1,2,4",
                        help="causal sensor-history lags in 25-ms samples; e.g. 0,1,2,4,8,12,20")
    parser.add_argument("--exact-packet-imu", action="store_true",
                        help="join IMU/attitude by exact source stamp to the current odometry packet")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    try:
        history_lags = tuple(int(value.strip())
                             for value in args.history_lags.split(","))
    except ValueError:
        parser.error("--history-lags must be comma-separated integers")
    fit(args.output, args.jobs, args.event_definition, history_lags,
        args.exact_packet_imu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
