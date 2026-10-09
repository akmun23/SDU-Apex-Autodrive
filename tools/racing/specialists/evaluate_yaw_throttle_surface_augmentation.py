#!/usr/bin/env python3
"""Test whether the existing 1,500-phase throttle surface improves yaw.

Only whole throttle sequences whose topic-command to actuator-feedback delay
is exactly two simulator packets are admitted. Run r04 is training; the
independent continuation capture r05 remains held out. The evaluated domain is
GT body speed 0--12 m/s; higher-speed rows are not scored or fitted.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.export_throttle_surface_sequences import (
    PACKET_DEBUG_NAMES,
)
from tools.racing.specialists import audit_yaw_full_domain_exact_two as audit
from tools.racing.specialists import evaluate_yaw_sensor_speed_latent as corpus
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas


ROOT = Path(__file__).resolve().parents[3]
SOURCE = (ROOT / "live_runs/derived_dynamics_learning_20260928/"
          "throttle_surface_40hz_dataset_20260930")
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_throttle_surface_exact_two_augmentation")
TRAIN_RUN = "openplane_throttle_5pct_5deg_20260930_r04"
VALIDATION_RUN = "openplane_throttle_5pct_5deg_20260930_r05_resume"
MAX_GT_SPEED_MPS = 12.0
HISTORY_LAGS = (0, 1, 2, 4)


def _packet_delay(source: dict[str, np.ndarray], sequence: int,
                  phase: dict[str, Any], bounds: np.ndarray
                  ) -> tuple[int | None, dict[str, int | None]]:
    begin, end = map(int, bounds[sequence])
    rel = source["time_from_stimulus_s"][begin:end, 0]
    after = np.flatnonzero(rel >= 0.0)
    if not len(after):
        return None, {"command_packet": None, "feedback_packet": None,
                      "contiguous": False}
    offset = int(after[0])
    start = begin + offset
    baseline, target = map(float, phase["condition_key"][1:3])
    increasing = target > baseline
    command = source["command_topics"][start:end, 1]
    feedback = source["actuator_feedback"][start:end, 1]
    command_crossing = (command >= target - 0.005 if increasing
                        else command <= target + 0.005)
    feedback_crossing = (feedback >= baseline + 0.005 if increasing
                         else feedback <= baseline - 0.005)
    command_indices = np.flatnonzero(command_crossing)
    feedback_indices = np.flatnonzero(feedback_crossing)
    if not len(command_indices) or not len(feedback_indices):
        return None, {"command_packet": None, "feedback_packet": None,
                      "contiguous": False}
    command_index = start + int(command_indices[0])
    feedback_index = start + int(feedback_indices[0])
    packet_ids = source["packet_sequence"][:, 0]
    delay = int(packet_ids[feedback_index] - packet_ids[command_index])
    packet_span = packet_ids[command_index:feedback_index + 1]
    contiguous = (len(packet_span) == 3
                  and np.all(np.diff(packet_span) == 1))
    return delay, {
        "command_packet": int(packet_ids[command_index]),
        "feedback_packet": int(packet_ids[feedback_index]),
        "contiguous": bool(contiguous),
    }


def _load_throttle_runs(
        history_lags: tuple[int, ...] = HISTORY_LAGS
        ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with np.load(SOURCE / "throttle_surface_sequences.npz",
                 allow_pickle=False) as archive:
        source = {name: np.asarray(archive[name]) for name in archive.files}
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    bounds = np.asarray(source["sequence_bounds"], dtype=np.int64)
    debug = source["bridge_debug_telemetry"]
    debug_columns = {name: i for i, name in enumerate(PACKET_DEBUG_NAMES)}
    run_sequences: dict[str, list[tuple[np.ndarray, ...]]] = {
        TRAIN_RUN: [], VALIDATION_RUN: []}
    run_counts: Counter[str] = Counter()
    delay_counts: Counter[int | str] = Counter()
    admitted_speed_rows = {TRAIN_RUN: 0, VALIDATION_RUN: 0}

    for sequence, phase in enumerate(manifest["sequences"]):
        run_id = str(phase["run_id"])
        delay, packet_detail = _packet_delay(source, sequence, phase, bounds)
        delay_label = ("2_noncontiguous" if delay == 2
                       and not packet_detail["contiguous"] else
                       delay if delay is not None else "missing")
        delay_counts[delay_label] += 1
        if (run_id not in run_sequences or delay != 2
                or not packet_detail["contiguous"]):
            continue
        begin, end = map(int, bounds[sequence])
        # Use the causal 100-ms encoder-position speed already present in this
        # 40-Hz dataset. The first four rows after reset have no such estimate;
        # trim only that invalid prefix, never bridge a packet gap.
        sensors = np.column_stack((
            source["actuator_feedback"][begin:end, 0:2],
            source["encoder_surface_mps_100ms"][begin:end, 0:2],
            source["imu_acceleration_mps2"][begin:end, 0:2],
            source["imu_angular_velocity_rps"][begin:end, 2],
            source["command_topics"][begin:end, 0:2],
        )).astype(np.float32)
        attitude = np.column_stack((
            source["imu_roll_pitch_rad"][begin:end, 0:2],
            source["imu_angular_velocity_rps"][begin:end, 0:2],
        )).astype(np.float32)
        rigid = np.zeros((end - begin, 13), dtype=np.float32)
        rigid[:, 7:10] = debug[begin:end, [
            debug_columns["simulator_velocity_x_mps"],
            debug_columns["simulator_velocity_y_mps"],
            debug_columns["simulator_velocity_z_mps"],
        ]]
        rigid[:, 10:13] = debug[begin:end, [
            debug_columns["simulator_angular_x_rps"],
            debug_columns["simulator_angular_y_rps"],
            debug_columns["simulator_angular_z_rps"],
        ]]
        finite = (np.isfinite(sensors).all(axis=1)
                  & np.isfinite(attitude).all(axis=1)
                  & np.isfinite(rigid).all(axis=1))
        valid_indices = np.flatnonzero(finite)
        if not len(valid_indices):
            run_counts[f"{run_id}:no_finite_sensor_rows"] += 1
            continue
        first = int(valid_indices[0])
        last = int(valid_indices[-1]) + 1
        if not finite[first:last].all():
            run_counts[f"{run_id}:internal_sensor_gap"] += 1
            continue
        sensors, attitude, rigid = sensors[first:last], attitude[first:last], rigid[first:last]
        if len(sensors) <= max(history_lags) + 1:
            continue
        sensor_valid = np.ones(len(sensors), dtype=bool)
        attitude_valid = np.ones(len(sensors), dtype=bool)
        run_sequences[run_id].append((
            sensors, sensor_valid, attitude, attitude_valid, rigid))
        run_counts[f"{run_id}:exact_two_sequences"] += 1
        speed = np.hypot(rigid[:, 7], rigid[:, 8])
        admitted_speed_rows[run_id] += int(np.count_nonzero(speed <= MAX_GT_SPEED_MPS))

    runs = {}
    for run_id, sequences in run_sequences.items():
        if not sequences:
            raise ValueError(f"no exact-two throttle sequences for {run_id}")
        rows = _rows_full_domain(run_id, sequences, history_lags)
        gt_domain = rows["gt_speed"] <= MAX_GT_SPEED_MPS
        runs[run_id] = {
            key: value[gt_domain] if isinstance(value, np.ndarray)
            and value.shape[:1] == gt_domain.shape else value
            for key, value in rows.items()
        }

    report = {
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "source_sequences": int(len(manifest["sequences"])),
        "command_topic_to_ros_feedback_packet_delay_counts": {
            str(key): int(value) for key, value in sorted(
                delay_counts.items(), key=lambda item: str(item[0]))},
        "exact_two_sequences_by_run": {
            run_id: int(run_counts[f"{run_id}:exact_two_sequences"])
            for run_id in run_sequences},
        "input_samples_in_gt_0_12mps_by_run": admitted_speed_rows,
        "other_sequence_rejections": {
            key: int(value) for key, value in run_counts.items()
            if not key.endswith(":exact_two_sequences")},
        "split": {TRAIN_RUN: "training", VALIDATION_RUN: "held-out validation"},
    }
    return runs, report


def _rows_full_domain(run_id: str, sequences: list[tuple[np.ndarray, ...]],
                      history_lags: tuple[int, ...] = HISTORY_LAGS
                      ) -> dict[str, np.ndarray]:
    """Construct one-step rows without filtering by wheel-speed bins.

    Wheel speed may exceed GT body speed during spin; that mismatch is part of
    the target physics and must not remove otherwise in-domain examples.
    """
    features, targets, sequence_ids = [], [], []
    wheel_speed, steering, gt_speed = [], [], []
    gt_yaw_current, imu_yaw, events, run_ids = [], [], [], []
    for sequence_id, (sensors, sensor_valid, attitude, attitude_valid,
                      rigid) in enumerate(sequences):
        if (not np.isfinite(sensors).all() or not np.isfinite(attitude).all()
                or not np.isfinite(rigid).all()):
            continue
        signal_ages = (
            atlas._age_since_change(sensors[:, 7], atlas.AGE_CHANGE_THRESHOLDS[0]),
            atlas._age_since_change(sensors[:, 0], atlas.AGE_CHANGE_THRESHOLDS[1]),
            atlas._age_since_change(sensors[:, 8], atlas.AGE_CHANGE_THRESHOLDS[2]),
            atlas._age_since_change(sensors[:, 1], atlas.AGE_CHANGE_THRESHOLDS[3]),
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
            left, right = map(float, sensors[k, 2:4])
            wheel_mean = 0.5 * (left + right)
            previous_mean = 0.5 * float(sensors[k - 1, 2] + sensors[k - 1, 3])
            derived = np.asarray((
                wheel_mean,
                left - right,
                (wheel_mean - previous_mean) / atlas.DT_S,
                (float(sensors[k, 0]) - float(sensors[k - 1, 0])) / atlas.DT_S,
                (float(sensors[k, 1]) - float(sensors[k - 1, 1])) / atlas.DT_S,
                float(sensors[k, 7] - sensors[k, 0]),
                float(sensors[k, 8] - sensors[k, 1]),
                float(sensors[k, 6] - sensors[k - 1, 6]),
                float(signal_ages[0][k]), float(signal_ages[1][k]),
                float(signal_ages[2][k]), float(signal_ages[3][k]),
                float(signal_ages[1][k] - signal_ages[0][k]),
                float(signal_ages[3][k] - signal_ages[2][k]),
            ), dtype=np.float32)
            imu_now = float(sensors[k, 6])
            features.append(np.concatenate((history, derived)))
            targets.append(float(rigid[k + 1, 12]) - imu_now)
            sequence_ids.append(sequence_id)
            wheel_speed.append(wheel_mean)
            steering.append(float(sensors[k, 0]))
            gt_speed.append(float(np.hypot(rigid[k, 7], rigid[k, 8])))
            gt_yaw_current.append(float(rigid[k, 12]))
            imu_yaw.append(imu_now)
            events.append(atlas._event(sensors, k, "command_intent"))
            run_ids.append(run_id)
    if not features:
        raise ValueError(f"{run_id}: no valid one-step examples")
    return {
        "x": np.asarray(features, dtype=np.float32),
        "residual": np.asarray(targets, dtype=np.float32),
        "sequence_id": np.asarray(sequence_ids, dtype=np.int32),
        "wheel_speed": np.asarray(wheel_speed, dtype=np.float32),
        "steering": np.asarray(steering, dtype=np.float32),
        "gt_speed": np.asarray(gt_speed, dtype=np.float32),
        "gt_yaw_current": np.asarray(gt_yaw_current, dtype=np.float32),
        "imu_yaw": np.asarray(imu_yaw, dtype=np.float32),
        "event": np.asarray(events, dtype="U16"),
        "run_id": np.asarray(run_ids, dtype="U128"),
    }


def _fit(x: np.ndarray, y: np.ndarray, run_ids: np.ndarray):
    return audit.ExtraTreesRegressor(**audit.ESTIMATOR).fit(
        x, y, sample_weight=audit._run_balanced_weights(run_ids))


def _metric(error: np.ndarray) -> dict[str, Any]:
    return corpus._metric(error)


def _run_macro(error: np.ndarray, run_ids: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(run_ids).astype(str)
    metrics = {
        run: float(np.sqrt(np.mean(error[ids == run] ** 2)))
        for run in np.unique(ids)
    }
    values = np.asarray(list(metrics.values()), dtype=np.float64)
    rng = np.random.default_rng(20261009)
    draws = rng.integers(0, len(values), size=(10_000, len(values)))
    return {
        "runs": len(metrics),
        "run_macro_rmse": float(np.mean(values)),
        "bootstrap_95pct_ci": [float(np.quantile(
            np.sqrt(np.mean(values[draws] ** 2, axis=1)), q))
            for q in (0.025, 0.975)],
        "per_run_rmse": metrics,
    }


def _paired_run_delta(baseline: np.ndarray, candidate: np.ndarray,
                      run_ids: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(run_ids).astype(str)
    values = np.asarray([
        (np.sqrt(np.mean(baseline[ids == run] ** 2)),
         np.sqrt(np.mean(candidate[ids == run] ** 2)))
        for run in np.unique(ids)
    ], dtype=np.float64)
    rng = np.random.default_rng(20261009)
    draws = rng.integers(0, len(values), size=(10_000, len(values)))
    deltas = (np.sqrt(np.mean(values[draws, 1] ** 2, axis=1))
              - np.sqrt(np.mean(values[draws, 0] ** 2, axis=1)))
    return {
        "independent_runs": int(len(values)),
        "mean_per_run_rmse_delta_candidate_minus_baseline": float(
            np.mean(values[:, 1] - values[:, 0])),
        "bootstrap_95pct_delta_ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def run() -> dict[str, Any]:
    throttle, throttle_audit = _load_throttle_runs()
    existing_train, existing_valid, corpus_audit = corpus._load_parts(
        corpus.DEFAULT_SOURCE)
    history_width = len(HISTORY_LAGS) * len(atlas.OBSERVATION_NAMES)
    feature_width = history_width + len(atlas.DERIVED_NAMES)
    x_train = existing_train["x"].astype(np.float32)
    y_train = existing_train["residual"].astype(np.float32)
    train_ids = existing_train["run_id"].astype(str)
    throttle_train = throttle[TRAIN_RUN]
    throttle_valid = throttle[VALIDATION_RUN]
    x_train_throttle = throttle_train["x"].astype(np.float32)
    y_train_throttle = throttle_train["residual"].astype(np.float32)
    train_ids_throttle = throttle_train["run_id"].astype(str)

    x_valid = existing_valid["x"].astype(np.float32)
    y_valid = existing_valid["residual"].astype(np.float32)
    valid_ids = existing_valid["run_id"].astype(str)
    valid_events = existing_valid["event"].astype(str)
    throttle_x_valid = throttle_valid["x"].astype(np.float32)
    throttle_y_valid = throttle_valid["residual"].astype(np.float32)
    throttle_valid_ids = throttle_valid["run_id"].astype(str)
    throttle_valid_events = throttle_valid["event"].astype(str)

    if any(matrix.shape[1] != feature_width for matrix in (
            x_train, x_valid, x_train_throttle, throttle_x_valid)):
        raise ValueError("yaw feature schemas differ across source datasets")

    baseline_prediction = np.full(len(y_valid), np.nan, dtype=np.float64)
    augmented_prediction = np.full(len(y_valid), np.nan, dtype=np.float64)
    throttle_base_prediction = np.full(len(throttle_y_valid), np.nan,
                                       dtype=np.float64)
    throttle_augmented_prediction = np.full(len(throttle_y_valid), np.nan,
                                            dtype=np.float64)
    event_metrics = {}
    for event in sorted(set(existing_train["event"].astype(str))):
        source_mask = existing_train["event"].astype(str) == event
        ext_mask = throttle_train["event"].astype(str) == event
        val_mask = valid_events == event
        ext_val_mask = throttle_valid_events == event
        ids_for_event = train_ids[source_mask]
        if (source_mask.sum() < audit.MIN_EVENT_ROWS
                or len(np.unique(ids_for_event)) < audit.MIN_EVENT_RUNS):
            continue
        base_model = _fit(x_train[source_mask], y_train[source_mask],
                          ids_for_event)
        if ext_mask.any():
            augmented_x = np.concatenate((x_train[source_mask],
                                          x_train_throttle[ext_mask]))
            augmented_y = np.concatenate((y_train[source_mask],
                                          y_train_throttle[ext_mask]))
            augmented_ids = np.concatenate((ids_for_event,
                                            train_ids_throttle[ext_mask]))
        else:
            augmented_x, augmented_y, augmented_ids = (
                x_train[source_mask], y_train[source_mask], ids_for_event)
        augmented_model = _fit(augmented_x, augmented_y, augmented_ids)
        if val_mask.any():
            baseline_prediction[val_mask] = base_model.predict(x_valid[val_mask])
            augmented_prediction[val_mask] = augmented_model.predict(
                x_valid[val_mask])
        if ext_val_mask.any():
            throttle_base_prediction[ext_val_mask] = base_model.predict(
                throttle_x_valid[ext_val_mask])
            throttle_augmented_prediction[ext_val_mask] = augmented_model.predict(
                throttle_x_valid[ext_val_mask])
        event_metrics[event] = {
            "existing_validation_rows": int(val_mask.sum()),
            "throttle_holdout_rows": int(ext_val_mask.sum()),
            "existing_validation_baseline": _metric(
                base_model.predict(x_valid[val_mask]) - y_valid[val_mask])
                if val_mask.any() else {"samples": 0},
            "existing_validation_augmented": _metric(
                augmented_model.predict(x_valid[val_mask]) - y_valid[val_mask])
                if val_mask.any() else {"samples": 0},
            "throttle_holdout_baseline": _metric(
                base_model.predict(throttle_x_valid[ext_val_mask])
                - throttle_y_valid[ext_val_mask])
                if ext_val_mask.any() else {"samples": 0},
            "throttle_holdout_augmented": _metric(
                augmented_model.predict(throttle_x_valid[ext_val_mask])
                - throttle_y_valid[ext_val_mask])
                if ext_val_mask.any() else {"samples": 0},
        }
        print(f"fitted event={event} with throttle_train={int(ext_mask.sum())}",
              flush=True)

    base_error = baseline_prediction - y_valid
    augmented_error = augmented_prediction - y_valid
    base_mask = np.isfinite(base_error) & np.isfinite(augmented_error)
    throttle_base_error = throttle_base_prediction - throttle_y_valid
    throttle_aug_error = throttle_augmented_prediction - throttle_y_valid
    throttle_mask = (np.isfinite(throttle_base_error)
                     & np.isfinite(throttle_aug_error))

    def _regions(rows: dict[str, np.ndarray], base_error: np.ndarray,
                 candidate_error: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        result = {}
        speed = rows["gt_speed"]
        steering = np.abs(rows["steering"])
        wheel_mismatch = np.abs(rows["gt_speed"] - rows["wheel_speed"])
        partitions = (
            ("speed", speed, ((0, 1), (1, 2), (2, 4), (4, 6), (6, 8),
                               (8, 10), (10, 12.001))),
            ("abs_steering", steering, ((0, .1), (.1, .2), (.2, .3),
                                         (.3, .4), (.4, .525))),
            ("abs_gt_minus_wheel", wheel_mismatch,
             ((0, 1), (1, 3), (3, 6), (6, np.inf))),
        )
        for name, values, ranges in partitions:
            for low, high in ranges:
                selected = mask & (values >= low) & (values < high)
                if selected.any():
                    result[f"{name}:{low:g}-{high:g}"] = {
                        "baseline": _metric(base_error[selected]),
                        "throttle_surface_augmented": _metric(
                            candidate_error[selected]),
                    }
        return result

    result = {
        "title": "Exact-two throttle-surface augmentation for one-step yaw",
        "feature_contract": {
            "history_lags_ms": [lag * 25 for lag in HISTORY_LAGS],
            "input_features": [
                *(f"{name}_lag{lag * 25}ms"
                  for lag in HISTORY_LAGS for name in atlas.OBSERVATION_NAMES),
                *atlas.DERIVED_NAMES,
            ],
            "future_truth_used_as_input": False,
            "source_feature_matrix": "the frozen atlas x matrix: requested four sensor-history lags plus the same derived features for every corpus",
            "target": "next simulator-GT yaw rate minus current IMU yaw rate",
            "GT_speed_training_and_validation_domain_mps": [0.0, 12.0],
        },
        "existing_corpus": {
            "train_rows": int(len(y_train)),
            "train_runs": int(len(np.unique(train_ids))),
            "validation_rows": int(len(y_valid)),
            "validation_runs": int(len(np.unique(valid_ids))),
        },
        "throttle_surface_audit": throttle_audit,
        "validation": {
            "baseline_global_event_model": _metric(base_error[base_mask]),
            "with_exact_two_throttle_surface": _metric(
                augmented_error[base_mask]),
            "baseline_run_macro": _run_macro(base_error[base_mask],
                                               valid_ids[base_mask]),
            "augmented_run_macro": _run_macro(augmented_error[base_mask],
                                                valid_ids[base_mask]),
            "paired_run_rmse_delta": _paired_run_delta(
                base_error[base_mask], augmented_error[base_mask],
                valid_ids[base_mask]),
            "by_region": _regions(existing_valid, base_error,
                                  augmented_error, base_mask),
        },
        "throttle_surface_heldout": {
            "baseline": _metric(throttle_base_error[throttle_mask]),
            "with_exact_two_throttle_surface": _metric(
                throttle_aug_error[throttle_mask]),
            "by_region": _regions(throttle_valid, throttle_base_error,
                                  throttle_aug_error, throttle_mask),
        },
        "events_fitted": event_metrics,
        "source_corpus_audit": corpus_audit,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True,
                   default=corpus._json_value) + "\n", encoding="utf-8")
    np.savez_compressed(
        OUTPUT / "validation_predictions.npz",
        existing_validation_run_id=valid_ids,
        existing_validation_speed_mps=existing_valid["gt_speed"],
        existing_validation_steering_rad=existing_valid["steering"],
        existing_validation_target=y_valid,
        existing_validation_baseline=baseline_prediction,
        existing_validation_augmented=augmented_prediction,
        throttle_validation_run_id=throttle_valid_ids,
        throttle_validation_speed_mps=throttle_valid["gt_speed"],
        throttle_validation_steering_rad=throttle_valid["steering"],
        throttle_validation_target=throttle_y_valid,
        throttle_validation_baseline=throttle_base_prediction,
        throttle_validation_augmented=throttle_augmented_prediction,
    )
    print(f"wrote {OUTPUT / 'report.json'}", flush=True)
    return result


if __name__ == "__main__":
    run()
