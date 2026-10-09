#!/usr/bin/env python3
"""Score causal yaw candidates on a fresh whole-run holdout.

Only the baseline and mirror-augmented ExtraTrees models selected by the
existing training-capture comparison are refit here. No future sensor or
simulator truth is a feature; response class is an offline evaluation label.
This is a research diagnostic and changes neither odometry nor MPC.
"""

from __future__ import annotations

import bisect
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sklearn.ensemble import ExtraTreesRegressor

try:
    import evaluate_yaw_packet_response_conditioned_models as response
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as response
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/targeted_group_candidate_comparison_r09.json")
VALIDATION_RUN_ID = "openplane_yaw_error_packet_phase_validation_r09_20261008"
# Restrict every candidate comparison to the user's valid two-packet response
# assumption. Other packet counts are excluded, not modeled as separate regimes.
GROUPS = ((2, "reversal"), (2, "unwind"))
STEERING_LIMIT_RAD = 0.5236
PACKET_JOIN_TOLERANCE_NS = 5_000_000


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _weights(phase_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(phase_ids, return_inverse=True,
                                    return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * (len(weights) / weights.sum())


def _model() -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=240, max_depth=14, min_samples_leaf=4,
        max_features=0.9, random_state=20261008, n_jobs=4)


def _load_bridge_packet_timing(
        series: Any) -> tuple[list[dict[str, Any]], int]:
    """Read request association for offline stratification, never as input."""
    bag = response.alignment_audit._bag_for_run(series)
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = response.dynamics._topic_map(connection)
        topic_id, message_type = topics[
            "/autodrive/roboracer_1/bridge_packet_timing"]
        message_class = get_message(message_type)
        rows = []
        invalid_records = 0
        for _receipt, payload in connection.execute(
                "SELECT timestamp,data FROM messages "
                "WHERE topic_id=? ORDER BY timestamp,id", (topic_id,)):
            record = json.loads(deserialize_message(
                bytes(payload), message_class).data)
            stamp = record.get("bridge_receive_ros_stamp_ns")
            sent_norm = record.get("sent_steering_norm")
            if stamp is None or sent_norm is None:
                invalid_records += 1
                continue
            rows.append({
                "stamp_ns": int(stamp),
                "sent_steering_rad": float(sent_norm) * STEERING_LIMIT_RAD,
                "request_command_age_ms": record.get(
                    "steering_command_update_age_ms"),
            })
        return sorted(rows, key=lambda row: row["stamp_ns"]), invalid_records
    finally:
        connection.close()


def _request_alignment_diagnostic(
        rows: dict[str, Any], indices: np.ndarray,
        predictions: dict[str, np.ndarray],
        packet_timing: list[dict[str, Any]],
        invalid_debug_records: int) -> dict[str, Any]:
    """Stratify yaw error by command-at-receipt vs command-sent mismatch."""
    packet_stamps = [row["stamp_ns"] for row in packet_timing]
    topic_command = rows["x"][indices, 7].astype(np.float64)
    sent_command = np.full(len(indices), np.nan, dtype=np.float64)
    sent_age = np.full(len(indices), np.nan, dtype=np.float64)
    join_error_ns = np.full(len(indices), np.iinfo(np.int64).max,
                            dtype=np.int64)
    for local, index in enumerate(indices):
        stamp = int(rows["sample_time_ns"][index])
        insertion = bisect.bisect_left(packet_stamps, stamp)
        candidates = [candidate for candidate in (insertion - 1, insertion)
                     if 0 <= candidate < len(packet_timing)]
        if not candidates:
            continue
        match = min(candidates, key=lambda candidate: abs(
            packet_stamps[candidate] - stamp))
        delta = abs(packet_stamps[match] - stamp)
        if delta > PACKET_JOIN_TOLERANCE_NS:
            continue
        record = packet_timing[match]
        sent_command[local] = record["sent_steering_rad"]
        age = record["request_command_age_ms"]
        if age is not None:
            sent_age[local] = float(age)
        join_error_ns[local] = delta

    valid = np.isfinite(sent_command)
    if not np.any(valid):
        return {"candidate_rows": int(len(indices)), "matched_rows": 0}
    gap = np.abs(topic_command - sent_command)
    bins = (
        ("within_0p05_rad", valid & (gap <= 0.05)),
        ("0p05_to_0p20_rad", valid & (gap > 0.05) & (gap <= 0.20)),
        ("over_0p20_rad", valid & (gap > 0.20)),
    )
    by_gap: dict[str, Any] = {}
    for label, mask in bins:
        if not np.any(mask):
            by_gap[label] = {"samples": 0}
            continue
        by_gap[label] = {
            "samples": int(np.count_nonzero(mask)),
            "median_abs_prediction_error_radps_by_model": {
                name: float(np.median(np.abs(
                    prediction[mask] - rows["residual"][indices[mask]])))
                for name, prediction in predictions.items()
            },
            "over_0p1_by_model": {
                name: int(np.count_nonzero(np.abs(
                    prediction[mask] - rows["residual"][indices[mask]]) > 0.1))
                for name, prediction in predictions.items()
            },
        }
    selected_age = sent_age[valid & np.isfinite(sent_age)]
    return {
        "debug_topic_used_as_model_input": False,
        "debug_topic_used_only_for_error_stratification": True,
        "measurement": "latest steering-command topic value at packet stamp minus steering sent with that bridge request",
        "join_tolerance_ms": PACKET_JOIN_TOLERANCE_NS / 1.0e6,
        "candidate_rows": int(len(indices)),
        "matched_rows": int(np.count_nonzero(valid)),
        "unmatched_rows": int(len(indices) - np.count_nonzero(valid)),
        "invalid_debug_packet_records_excluded": int(invalid_debug_records),
        "packet_stamp_join_error_p95_ms": float(np.quantile(
            join_error_ns[valid] / 1.0e6, 0.95)),
        "request_command_update_age_median_ms": (
            float(np.median(selected_age)) if len(selected_age) else None),
        "by_command_gap": by_gap,
    }


def _mirror_features(features: np.ndarray, history_lags: tuple[int, ...]
                     ) -> np.ndarray:
    """Reflect a left-turn sample into its right-turn equivalent."""
    mirrored = np.asarray(features).copy()
    width = len(atlas.OBSERVATION_NAMES)
    signed_indices = (0, 5, 6, 7, 9, 10)
    for block in range(len(history_lags)):
        base = block * width
        mirrored[:, [base + index for index in signed_indices]] *= -1.0
        left = features[:, base + 2].copy()
        mirrored[:, base + 2] = features[:, base + 3]
        mirrored[:, base + 3] = left
    derived = len(history_lags) * width
    for offset in (1, 3, 5, 7):
        mirrored[:, derived + offset] *= -1.0
    # Remaining tail columns are the event ages/packet timing and are invariant.
    return mirrored


def _phase_rows(rows: dict[str, Any], phases: dict[str, dict[str, Any]],
                step: int, event: str, *, require_row_event: bool
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices, phase_ids = [], []
    phase_names = []
    for label, phase in phases.items():
        if phase["true_steps"] != step or phase["event"] != event:
            continue
        selected = np.asarray(phase["indices"], dtype=np.int64)
        if require_row_event:
            selected = selected[rows["event"][selected].astype(str) == event]
        indices.extend(selected.tolist())
        phase_ids.extend([label] * len(selected))
        phase_names.extend([label] * len(selected))
    return (np.asarray(indices, dtype=np.int64),
            np.asarray(phase_ids, dtype="U256"),
            np.asarray(phase_names, dtype="U256"))


def _fit_predict(train_rows: dict[str, Any], train_idx: np.ndarray,
                 train_phase_ids: np.ndarray, test_rows: dict[str, Any],
                 test_idx: np.ndarray, feature_key: str,
                 mirror: bool = False) -> np.ndarray:
    model = _model()
    x_train = train_rows[feature_key][train_idx]
    y_train = train_rows["residual"][train_idx]
    phase_ids = train_phase_ids
    if mirror:
        lags = (response.EXPANDED_HISTORY_LAGS
                if feature_key == "x_timed" else atlas.HISTORY_LAGS)
        x_train = np.concatenate((x_train, _mirror_features(x_train, lags)))
        y_train = np.concatenate((y_train, -y_train))
        phase_ids = np.concatenate((phase_ids, phase_ids))
    model.fit(x_train, y_train, sample_weight=_weights(phase_ids))
    return model.predict(test_rows[feature_key][test_idx]).astype(np.float32)


def _fit_predict_gt_delta(
        train_rows: dict[str, Any], train_idx: np.ndarray,
        train_phase_ids: np.ndarray, test_rows: dict[str, Any],
        test_idx: np.ndarray) -> np.ndarray:
    """Predict GT yaw change, then add current measured IMU yaw rate."""
    model = _model()
    x_train = train_rows["x_timed"][train_idx]
    y_train = train_rows["delta_gt"][train_idx]
    x_train = np.concatenate((x_train, _mirror_features(
        x_train, response.EXPANDED_HISTORY_LAGS)))
    y_train = np.concatenate((y_train, -y_train))
    phase_ids = np.concatenate((train_phase_ids, train_phase_ids))
    model.fit(x_train, y_train, sample_weight=_weights(phase_ids))
    predicted_next = (test_rows["imu_yaw"][test_idx]
                      + model.predict(test_rows["x_timed"][test_idx]))
    truth_next = (test_rows["residual"][test_idx]
                  + test_rows["imu_yaw"][test_idx])
    return (predicted_next - truth_next).astype(np.float32)


def _collect_group(run_rows: dict[str, Any], run_phases: dict[str, Any],
                   run_ids: list[str], step: int, event: str,
                   require_row_event: bool) -> tuple[dict[str, Any], np.ndarray]:
    parts = []
    for run_id in run_ids:
        rows, phases = run_rows[run_id], run_phases[run_id]
        indices, phase_ids, _ = _phase_rows(
            rows, phases, step, event, require_row_event=require_row_event)
        if not len(indices):
            continue
        parts.append({
            "run_id": run_id,
            "indices": indices,
            "x_timed": rows["x_timed"][indices],
            "x_short_timed": rows["x_short_timed"][indices],
            "x": rows["x"][indices],
            "residual": rows["residual"][indices],
            "delta_gt": (rows["residual"][indices]
                         + rows["imu_yaw"][indices]
                         - rows["gt_yaw_current"][indices]),
            "phase_ids": np.asarray([f"{run_id}:{p}" for p in phase_ids]),
        })
    if not parts:
        raise ValueError(f"no data for {event}/{step}-packet")
    combined = {key: np.concatenate([part[key] for part in parts], axis=0)
                for key in ("x_timed", "x_short_timed", "x", "residual",
                            "delta_gt", "phase_ids")}
    return combined, np.concatenate([
        np.full(len(part["residual"]), part["run_id"], dtype="U128")
        for part in parts])


def _training_condition_support(
        run_rows: dict[str, Any], run_phases: dict[str, Any],
        run_ids: list[str], step: int, event: str) -> dict[str, Any]:
    """Count training support only for the requested valid packet class."""
    support: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        rows = run_rows[run_id]
        for phase in run_phases[run_id].values():
            if phase["true_steps"] != step or phase["event"] != event:
                continue
            indices = np.asarray(phase["indices"], dtype=np.int64)
            indices = indices[rows["event"][indices].astype(str) == event]
            if not len(indices):
                continue
            key = json.dumps(phase["condition"], sort_keys=True)
            entry = support.setdefault(key, {"phases": 0, "rows": 0, "runs": []})
            entry["phases"] += 1
            entry["rows"] += int(len(indices))
            entry["runs"].append(run_id)
    return {
        key: {**value, "runs": sorted(set(value["runs"]))}
        for key, value in sorted(support.items())
    }


def _training_only_loco_model_selection(
        run_rows: dict[str, Any], run_phases: dict[str, Any],
        train_ids: list[str], step: int, event: str) -> dict[str, Any]:
    """Choose symmetry augmentation using only exact-class training folds."""
    per_model: dict[str, list[dict[str, Any]]] = {
        "extra_trees_long": [], "extra_trees_long_mirror": []}
    for held_out in train_ids:
        fit_ids = [run_id for run_id in train_ids if run_id != held_out]
        fit, _ = _collect_group(run_rows, run_phases, fit_ids, step, event,
                                require_row_event=True)
        val_rows = run_rows[held_out]
        val_indices, _, _ = _phase_rows(
            val_rows, run_phases[held_out], step, event,
            require_row_event=True)
        if not len(val_indices):
            continue
        for name, mirror in (("extra_trees_long", False),
                             ("extra_trees_long_mirror", True)):
            prediction = _fit_predict(
                fit, np.arange(len(fit["residual"])), fit["phase_ids"],
                val_rows, val_indices, "x_timed", mirror=mirror)
            per_model[name].append({
                "held_out_capture": held_out,
                "fit_captures": fit_ids,
                **_metric(prediction - val_rows["residual"][val_indices]),
            })
    macro_rmse = {
        name: float(np.mean([row["rmse_radps"] for row in rows]))
        for name, rows in per_model.items() if rows
    }
    paired = {row["held_out_capture"]: row["rmse_radps"]
              for row in per_model["extra_trees_long"]}
    mirrored = {row["held_out_capture"]: row["rmse_radps"]
                for row in per_model["extra_trees_long_mirror"]}
    shared = sorted(set(paired) & set(mirrored))
    mirror_delta = [mirrored[run_id] - paired[run_id] for run_id in shared]
    selected = min(macro_rmse, key=macro_rmse.get) if macro_rmse else None
    return {
        "packet_policy": "exactly two packets only; every other response count is excluded",
        "selection_split": "leave-one-training-capture-out; fresh r09 holdout was not used",
        "per_model_by_held_out_capture": per_model,
        "run_macro_rmse_radps": macro_rmse,
        "mirror_minus_plain_rmse_by_capture_radps": dict(zip(shared, mirror_delta)),
        "captures_mirror_better": int(sum(delta < 0 for delta in mirror_delta)),
        "captures_plain_better": int(sum(delta > 0 for delta in mirror_delta)),
        "selected_by_training_only_macro_rmse": selected,
    }


def _run(validation_run_id: str = VALIDATION_RUN_ID,
         exclude_train_run_ids: tuple[str, ...] = (),
         output_path: Path | None = None) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audits = response._collect_exact_two_packet(admitted)
    run_rows, run_phases = collected["rows"], collected["phases"]
    train_ids = sorted(run_id for run_id in response.TIMING_SUFFIX
                       if "_train_" in run_id)
    unknown_exclusions = set(exclude_train_run_ids) - set(train_ids)
    if unknown_exclusions:
        raise ValueError(
            f"cannot exclude unregistered training runs: {sorted(unknown_exclusions)}")
    train_ids = [run_id for run_id in train_ids
                 if run_id not in set(exclude_train_run_ids)]
    if validation_run_id not in response.TIMING_SUFFIX:
        raise ValueError(f"unregistered validation run: {validation_run_id}")
    validation_id = validation_run_id
    if validation_id in train_ids:
        raise ValueError(
            f"validation run is also in training; exclude it explicitly: {validation_id}")
    if len(train_ids) < 2:
        raise ValueError("at least two independent training captures are required")
    validation_key = f"heldout_{response.TIMING_SUFFIX[validation_id]}"
    validation_series = next(
        series for series in admitted if series.run_id == validation_id)
    packet_timing, invalid_debug_records = _load_bridge_packet_timing(
        validation_series)

    models = (("extra_trees_long", False),
              ("extra_trees_long_mirror", True))
    reports: dict[str, Any] = {}
    scored_rows: list[dict[str, Any]] = []
    for step, event in GROUPS:
        group_key = f"{event}/{step}_packet"
        group_report: dict[str, Any] = {
            "training_group_is_phase_event_packet_class": group_key,
            "row_event_restricted": True,
            "training_capture_count": len(train_ids),
            "selection_reference": "exact-two-packet leave-one-training-capture-out recomputed in this report",
            "training_capture_cross_validation_recomputed": True,
            "training_only_model_selection": _training_only_loco_model_selection(
                run_rows, run_phases, train_ids, step, event),
            validation_key: {},
        }
        fit, _ = _collect_group(run_rows, run_phases, train_ids, step, event,
                                require_row_event=True)
        group_report["training_condition_support"] = (
            _training_condition_support(
                run_rows, run_phases, train_ids, step, event))
        val_rows = run_rows[validation_id]
        val_idx, val_phase_labels, _ = _phase_rows(
            val_rows, run_phases[validation_id], step, event,
            require_row_event=True)
        group_report[validation_key]["samples"] = int(len(val_idx))
        group_report[validation_key]["gyro_persistence"] = _metric(
            -val_rows["residual"][val_idx])
        if not len(val_idx):
            group_report[validation_key]["status"] = (
                "no held-out phases realized this event/packet-response class")
            reports[group_key] = group_report
            continue
        heldout_predictions = {}
        for name, mirror in models:
            prediction = _fit_predict(
                fit, np.arange(len(fit["residual"])), fit["phase_ids"],
                val_rows, val_idx, "x_timed", mirror=mirror)
            heldout_predictions[name] = prediction
            group_report[validation_key][name] = _metric(
                prediction - val_rows["residual"][val_idx])
        age_ms = np.asarray([
            (int(val_rows["sample_time_ns"][index])
             - int(run_phases[validation_id][str(label)]["command_receipt_ns"]))
            / 1.0e6
            for index, label in zip(val_idx, val_phase_labels)],
            dtype=np.float64)
        temporal_bins = (
            ("before_command", -np.inf, 0.0),
            ("0_to_25ms", 0.0, 25.0),
            ("25_to_50ms", 25.0, 50.0),
            ("50_to_150ms", 50.0, 150.0),
            ("150ms_plus", 150.0, np.inf),
        )
        group_report[validation_key]["by_transition_age"] = {
            name: {
                model_name: _metric(
                    prediction[(age_ms >= lower) & (age_ms < upper)]
                    - val_rows["residual"][val_idx][
                        (age_ms >= lower) & (age_ms < upper)])
                for model_name, prediction in heldout_predictions.items()
            }
            for name, lower, upper in temporal_bins
            if np.any((age_ms >= lower) & (age_ms < upper))
        }
        delta_error = _fit_predict_gt_delta(
            fit, np.arange(len(fit["delta_gt"])), fit["phase_ids"],
            val_rows, val_idx)
        group_report[validation_key][
            "extra_trees_long_mirror_gt_delta_target"] = _metric(delta_error)
        current_imu_minus_gt = (
            val_rows["imu_yaw"][val_idx]
            - val_rows["gt_yaw_current"][val_idx])
        group_report[validation_key]["yaw_target_decomposition"] = {
            "current_imu_minus_current_gt_radps": _metric(current_imu_minus_gt),
            "current_to_next_gt_yaw_change_radps": _metric(
                val_rows["residual"][val_idx]
                + val_rows["imu_yaw"][val_idx]
                - val_rows["gt_yaw_current"][val_idx]),
            "note": "GT current yaw is used only to form offline labels and decompose the target; it is never a predictor input.",
        }
        group_report[validation_key]["packet_command_alignment_diagnostic"] = (
            _request_alignment_diagnostic(
                val_rows, val_idx, heldout_predictions, packet_timing,
                invalid_debug_records))

        phase_by_label = run_phases[validation_id]
        condition_rows: dict[str, list[int]] = {}
        for local, label in enumerate(val_phase_labels):
            key = json.dumps(phase_by_label[str(label)]["condition"],
                             sort_keys=True)
            condition_rows.setdefault(key, []).append(local)
        group_report[validation_key]["by_exact_probe_condition"] = {}
        for condition, local_rows in condition_rows.items():
            local = np.asarray(local_rows, dtype=np.int64)
            group_report[validation_key]["by_exact_probe_condition"][condition] = {
                name: _metric(heldout_predictions[name][local]
                              - val_rows["residual"][val_idx[local]])
                for name in ("extra_trees_long", "extra_trees_long_mirror")
            }

        mirror_error = (heldout_predictions["extra_trees_long_mirror"]
                        - val_rows["residual"][val_idx])
        worst_local = np.argsort(np.abs(mirror_error))[::-1][:20]
        worst_rows = []
        for local in worst_local:
            index = int(val_idx[local])
            phase_label = str(val_phase_labels[local])
            phase = phase_by_label[phase_label]
            x = val_rows["x"][index]
            worst_rows.append({
                "phase": phase_label,
                "condition": phase["condition"],
                "sample_time_ns": int(val_rows["sample_time_ns"][index]),
                "age_from_command_receipt_ms": float(
                    (val_rows["sample_time_ns"][index]
                     - phase["command_receipt_ns"]) / 1.0e6),
                "speed_rear_wheel_mean_mps": float(val_rows["wheel_speed"][index]),
                "steering_feedback_rad": float(x[0]),
                "steering_command_rad": float(x[7]),
                "command_feedback_gap_rad": float(x[7] - x[0]),
                "current_imu_yaw_radps": float(val_rows["imu_yaw"][index]),
                "actual_next_steering_feedback_rad_diagnostic_only": float(
                    val_rows["next_steering"][index]),
                "truth_next_yaw_radps": float(
                    val_rows["residual"][index] + val_rows["imu_yaw"][index]),
                "predicted_next_yaw_radps": float(
                    heldout_predictions["extra_trees_long_mirror"][local]
                    + val_rows["imu_yaw"][index]),
                "signed_error_radps": float(mirror_error[local]),
            })
        group_report[validation_key]["worst_20_mirrored_candidate_rows"] = worst_rows
        for local, index in enumerate(val_idx):
            phase_label = str(val_phase_labels[local])
            phase = phase_by_label[phase_label]
            x = val_rows["x"][index]
            scored_rows.append({
                "run_id": validation_id,
                "event": event,
                "response_packet_count": 2,
                "phase": phase_label,
                "speed_mps_rear_wheel_mean": float(val_rows["wheel_speed"][index]),
                "requested_speed_mps": float(phase["condition"]["speed_mps"]),
                "requested_abs_steering_rad": float(
                    phase["condition"]["steering_abs_rad"]),
                "turn_sign": int(phase["condition"]["turn_sign"]),
                "transition_mode": str(phase["condition"]["transition_mode"]),
                "transition_duration_s": float(phase["condition"]["duration_s"]),
                "scheduled_delay_s": float(phase["condition"]["scheduled_delay_s"]),
                "age_from_command_receipt_ms": float(age_ms[local]),
                "steering_feedback_rad": float(x[0]),
                "steering_command_rad": float(x[7]),
                "current_imu_yaw_rate_radps": float(val_rows["imu_yaw"][index]),
                "truth_next_yaw_rate_radps": float(
                    val_rows["residual"][index] + val_rows["imu_yaw"][index]),
                "gyro_persistence_next_yaw_rate_radps": float(val_rows["imu_yaw"][index]),
                "extra_trees_long_next_yaw_rate_radps": float(
                    heldout_predictions["extra_trees_long"][local]
                    + val_rows["imu_yaw"][index]),
                "extra_trees_long_mirror_next_yaw_rate_radps": float(
                    heldout_predictions["extra_trees_long_mirror"][local]
                    + val_rows["imu_yaw"][index]),
                "extra_trees_long_mirror_signed_error_radps": float(
                    mirror_error[local]),
            })
        reports[group_key] = group_report

    result = {
        "title": "Targeted causal-model comparison for the two largest yaw-error groups",
        "status": "offline comparison only; no runtime model or estimator changed",
        "target": "next 25-ms simulator-truth yaw-rate minus current exact-packet IMU yaw rate (rad/s)",
        "packet_response_policy": "Only exactly two-packet response phases are eligible; all other packet counts, including three, are excluded.",
        "inputs": "current and past permitted sensors/commands only; future steering and simulator truth are labels/diagnostics, not features",
        "bridge_packet_timing_policy": "Restricted bridge debug data is used only to stratify errors after prediction; it is never a model feature or production subscription.",
        "training_runs": train_ids,
        "excluded_training_runs": sorted(exclude_train_run_ids),
        "whole_run_holdout": validation_id,
        "validation_previously_used_during_model_development": False,
        "candidate_selection_reused_from_training_only_comparison": False,
        "candidate_selection_policy": "mirror augmentation is selected only from exact-two-packet leave-one-training-capture-out scores",
        "fresh_validation_used_for_candidate_selection": False,
        "response_packet_class_is_oracle_training_and_evaluation_label": True,
        "models": [name for name, _ in models],
        "phase_audits": phase_audits,
        "source_audit": source_audit,
        "groups": reports,
    }
    if output_path is None:
        selected_output = OUTPUT.with_name(
            f"targeted_group_candidate_comparison_{response.TIMING_SUFFIX[validation_id]}.json")
    else:
        selected_output = (output_path if output_path.is_absolute()
                           else ROOT / output_path)
    selected_output.parent.mkdir(parents=True, exist_ok=True)
    selected_output.write_text(json.dumps(
        result, indent=2, sort_keys=True,
        default=lambda value: value.item() if isinstance(value, np.generic)
        else str(value)) + "\n",
                      encoding="utf-8")
    csv_output = selected_output.with_name(
        f"{selected_output.stem}_exact2_rows.csv")
    if scored_rows:
        with csv_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scored_rows[0]))
            writer.writeheader()
            writer.writerows(scored_rows)
    print(json.dumps({
        "output": str(selected_output.relative_to(ROOT)),
        "exact_two_packet_rows_csv": str(csv_output.relative_to(ROOT)),
        "groups": {
            key: {
                validation_key: value[validation_key],
            } for key, value in reports.items()},
    }, indent=2))
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run-id", default=VALIDATION_RUN_ID)
    parser.add_argument("--exclude-train-run-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _run(args.validation_run_id, tuple(args.exclude_train_run_id), args.output)
