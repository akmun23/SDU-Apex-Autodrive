#!/usr/bin/env python3
"""Localize held-out yaw errors and support gaps in the two worst response groups.

The response-packet label is used only to form offline training/evaluation
subsets. The yaw model itself sees current/past legal sensor and command
features. r03 remains whole-run held out; no test/final-test data are opened.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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
          "yaw_large_error_audit_v1/worst_group_condition_support_r06.json")
# Only exactly-two-packet response phases are valid for this yaw study.
GROUPS = ((2, "reversal"), (2, "unwind"))
TRAIN_IDS = tuple(sorted(run_id for run_id in response.TIMING_SUFFIX
                         if "_train_" in run_id))
VALIDATION_ID = next(run_id for run_id in response.TIMING_SUFFIX
                     if "_validation_" in run_id)
MODEL = {
    "n_estimators": 240,
    "max_depth": 14,
    "min_samples_leaf": 4,
    "max_features": 0.9,
    "random_state": 20261008,
    "n_jobs": 4,
}


def _phase_rows(rows: dict[str, Any], phases: dict[str, Any],
                step: int, event: str) -> list[dict[str, Any]]:
    output = []
    for label, phase in phases.items():
        if phase["true_steps"] != step or phase["event"] != event:
            continue
        indices = np.asarray(phase["indices"], dtype=np.int64)
        indices = indices[rows["event"][indices].astype(str) == event]
        indices = indices[rows["next_steering_valid"][indices]]
        if len(indices):
            output.append({"label": label, "phase": phase,
                           "indices": indices})
    return output


def _phase_weighted_fit(x: np.ndarray, y: np.ndarray,
                        phase_ids: np.ndarray) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor
    ids, inverse, counts = np.unique(phase_ids, return_inverse=True,
                                    return_counts=True)
    weights = 1.0 / counts[inverse]
    weights *= len(weights) / weights.sum()
    return ExtraTreesRegressor(**MODEL).fit(x, y, sample_weight=weights)


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _condition_key(condition: dict[str, Any]) -> str:
    keys = ("speed_mps", "steering_abs_rad", "turn_sign",
            "transition_mode", "duration_s")
    return json.dumps({key: condition[key] for key in keys}, sort_keys=True)


def _bridge_request_age(series: Any, rows: dict[str, Any]
                        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a forbidden bridge-debug covariate for diagnosis only."""
    bag = response.alignment_audit._bag_for_run(series)
    source_by_receipt = response.packet_alignment.odom_source_stamp_by_receipt(bag)
    source_stamps = np.asarray([
        source_by_receipt.get(int(receipt), -1)
        for receipt in rows["sample_time_ns"]], dtype=np.int64)
    connection = response.sqlite3.connect(
        bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = response.dynamics._topic_map(connection)
        topic_id, topic_type = topics[response.dynamics.PACKET_TIMING]
        message_type = response.get_message(topic_type)
        age_by_source: dict[int, float] = {}
        packet_count = 0
        for (_receipt, payload) in connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
                (topic_id,)):
            message = response.deserialize_message(bytes(payload), message_type)
            debug = json.loads(message.data)
            stamp = debug.get("bridge_receive_ros_stamp_ns")
            age = debug.get("steering_command_update_age_ms")
            if stamp is None:
                continue
            packet_count += 1
            age_by_source[int(stamp)] = (
                float(age) if age is not None and np.isfinite(float(age))
                else float("nan"))
    finally:
        connection.close()
    aligned = np.asarray([
        age_by_source.get(int(stamp), float("nan"))
        for stamp in source_stamps], dtype=np.float32)
    audit = {
        "bridge_packet_rows": packet_count,
        "yaw_rows": int(len(source_stamps)),
        "exact_source_stamp_rows_with_age": int(np.count_nonzero(np.isfinite(aligned))),
        "rows_without_exact_bridge_age": int(np.count_nonzero(~np.isfinite(aligned))),
        "source_stamp_alignment": "exact bridge_receive_ros_stamp_ns == odometry source/header stamp",
        "use": "offline causal-factor diagnostic only; prohibited input for runtime model",
    }
    return aligned, audit


def _error_bands(rows: dict[str, Any], indices: np.ndarray,
                 errors: np.ndarray) -> dict[str, Any]:
    derived_start = len(atlas.HISTORY_LAGS) * len(atlas.OBSERVATION_NAMES)
    channels = {
        "wheel_speed_mps": rows["wheel_speed"][indices],
        "abs_measured_steering_rad": np.abs(rows["steering"][indices]),
        "steering_command_gap_rad": rows["x"][indices, derived_start + 5],
        "throttle_feedback_norm": rows["x"][indices, 1],
        "throttle_command_gap_norm": rows["x"][indices, derived_start + 6],
        "rear_wheel_split_mps": rows["x"][indices, derived_start + 1],
        "abs_current_imu_yaw_rate_radps": np.abs(rows["imu_yaw"][indices]),
        "actual_gt_speed_diagnostic_mps": rows["gt_speed"][indices],
    }
    edges = {
        "wheel_speed_mps": np.asarray([0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 12.0]),
        "abs_measured_steering_rad": np.asarray([0, .1, .2, .3, .35, .42, .5, .55]),
        "steering_command_gap_rad": np.asarray([-1, -.4, -.2, -.1, 0, .1, .2, .4, 1]),
        "throttle_feedback_norm": np.asarray([0, .05, .10, .12, .14, .16, .18, .25, 1]),
        "throttle_command_gap_norm": np.asarray([-1, -.3, -.1, -.05, 0, .05, .1, .3, 1]),
        "rear_wheel_split_mps": np.asarray([-3, -1, -.5, -.2, 0, .2, .5, 1, 3]),
        "abs_current_imu_yaw_rate_radps": np.asarray([0, .25, .5, .75, 1, 1.5, 2, 5]),
        "actual_gt_speed_diagnostic_mps": np.asarray([0, 2.5, 3, 3.5, 4, 4.5, 5, 12]),
    }
    result: dict[str, Any] = {}
    for name, values in channels.items():
        bins = np.digitize(values, edges[name][1:-1], right=False)
        output = []
        for bin_index in sorted(set(bins.tolist())):
            mask = bins == bin_index
            e = errors[mask]
            output.append({
                "lower_edge": float(edges[name][bin_index]),
                "upper_edge": float(edges[name][bin_index + 1]),
                **_metric(e),
            })
        result[name] = output
    return result


def run(output: Path = OUTPUT) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audit = response._collect_exact_two_packet(admitted)
    rows_by_run, phases_by_run = collected["rows"], collected["phases"]
    series_by_id = {series.run_id: series for series in admitted}
    bridge_age_by_run: dict[str, np.ndarray] = {}
    bridge_age_audit: dict[str, Any] = {}
    for run_id in (*TRAIN_IDS, VALIDATION_ID):
        age, audit = _bridge_request_age(series_by_id[run_id], rows_by_run[run_id])
        bridge_age_by_run[run_id] = age
        bridge_age_audit[run_id] = audit
    report: dict[str, Any] = {
        "title": "Condition support and error structure in worst yaw groups",
        "status": "offline research only; no runtime model changed",
        "model_features": "legal current/past sensor/command history plus current receipt/source timing",
        "packet_response_label_usage": "offline subgroup labels only; never an estimator input",
        "held_out_whole_run": VALIDATION_ID,
        "test_and_final_test_opened": False,
        "phase_audit": phase_audit,
        "groups": {},
        "bridge_debug_age_audit": bridge_age_audit,
        "source_audit": source_audit,
    }
    for step, event in GROUPS:
        x_parts, y_parts, phase_ids = [], [], []
        timed_x_parts, timed_y_parts, timed_phase_ids = [], [], []
        same_support_x_parts, same_support_y_parts = [], []
        same_support_phase_ids = []
        train_condition_counts: Counter[str] = Counter()
        train_phase_counts = Counter()
        for run_id in TRAIN_IDS:
            rows, phases = rows_by_run[run_id], phases_by_run[run_id]
            for phase_row in _phase_rows(rows, phases, step, event):
                condition_key = _condition_key(phase_row["phase"]["condition"])
                train_condition_counts[condition_key] += 1
                label = f"{run_id}:{phase_row['label']}"
                indices = phase_row["indices"]
                x_parts.append(rows["x_short_timed"][indices])
                y_parts.append(rows["residual"][indices])
                phase_ids.extend([label] * len(indices))
                timed_valid = np.isfinite(bridge_age_by_run[run_id][indices])
                timed_indices = indices[timed_valid]
                if len(timed_indices):
                    same_support_x_parts.append(
                        rows["x_short_timed"][timed_indices])
                    same_support_y_parts.append(rows["residual"][timed_indices])
                    same_support_phase_ids.extend([label] * len(timed_indices))
                    timed_x_parts.append(np.column_stack((
                        rows["x_short_timed"][timed_indices],
                        bridge_age_by_run[run_id][timed_indices])))
                    timed_y_parts.append(rows["residual"][timed_indices])
                    timed_phase_ids.extend([label] * len(timed_indices))
                train_phase_counts[run_id] += 1
        if not x_parts:
            raise ValueError(f"no training data for {event}/{step}-packet")
        train_x = np.concatenate(x_parts)
        train_y = np.concatenate(y_parts)
        model = _phase_weighted_fit(
            train_x, train_y, np.asarray(phase_ids, dtype="U256"))

        valid_rows = rows_by_run[VALIDATION_ID]
        valid_phases = phases_by_run[VALIDATION_ID]
        valid_phase_rows = _phase_rows(valid_rows, valid_phases, step, event)
        valid_indices = np.concatenate([row["indices"] for row in valid_phase_rows])
        prediction = model.predict(valid_rows["x_short_timed"][valid_indices])
        error = prediction - valid_rows["residual"][valid_indices]
        valid_debug_age = np.isfinite(bridge_age_by_run[VALIDATION_ID][valid_indices])
        diagnostic_indices = valid_indices[valid_debug_age]
        debug_train_x = np.concatenate(timed_x_parts)
        debug_train_y = np.concatenate(timed_y_parts)
        debug_train_phase = np.asarray(timed_phase_ids, dtype="U256")
        debug_model = _phase_weighted_fit(
            debug_train_x, debug_train_y, debug_train_phase)
        same_support_baseline = _phase_weighted_fit(
            np.concatenate(same_support_x_parts),
            np.concatenate(same_support_y_parts),
            np.asarray(same_support_phase_ids, dtype="U256"))
        debug_prediction = debug_model.predict(np.column_stack((
            valid_rows["x_short_timed"][diagnostic_indices],
            bridge_age_by_run[VALIDATION_ID][diagnostic_indices])))
        diagnostic_baseline_error = same_support_baseline.predict(
            valid_rows["x_short_timed"][diagnostic_indices]) - valid_rows["residual"][diagnostic_indices]
        diagnostic_debug_error = debug_prediction - valid_rows["residual"][diagnostic_indices]
        by_condition: dict[str, dict[str, Any]] = {}
        worst_phases = []
        cursor = 0
        for phase_row in valid_phase_rows:
            n = len(phase_row["indices"])
            phase_error = error[cursor:cursor + n]
            cursor += n
            condition = phase_row["phase"]["condition"]
            key = _condition_key(condition)
            row = by_condition.setdefault(key, {
                "condition": condition,
                "train_matching_packet_phase_count": int(train_condition_counts[key]),
                "validation_phases": 0,
                "samples": 0,
                "samples_over_0p1": 0,
                "sum_squared_error": 0.0,
                "max_abs_radps": 0.0,
            })
            row["validation_phases"] += 1
            row["samples"] += n
            row["samples_over_0p1"] += int(np.count_nonzero(np.abs(phase_error) > 0.1))
            row["sum_squared_error"] += float(np.sum(phase_error * phase_error))
            row["max_abs_radps"] = max(row["max_abs_radps"],
                                         float(np.max(np.abs(phase_error))))
            worst_phases.append({
                "condition": condition,
                "train_matching_packet_phase_count": int(train_condition_counts[key]),
                "rows": n,
                "rmse_radps": float(np.sqrt(np.mean(phase_error * phase_error))),
                "max_abs_radps": float(np.max(np.abs(phase_error))),
                "samples_over_0p1": int(np.count_nonzero(np.abs(phase_error) > 0.1)),
            })
        for row in by_condition.values():
            row["rmse_radps"] = float(np.sqrt(
                row.pop("sum_squared_error") / row["samples"]))
        report["groups"][f"{event}/{step}_packet"] = {
            "training_rows": int(len(train_y)),
            "training_phases_by_capture": dict(train_phase_counts),
            "validation_phases": int(len(valid_phase_rows)),
            "validation_rows": int(len(valid_indices)),
            "validation_yaw_error": _metric(error),
            "offline_bridge_request_age_ablation_same_support": {
                "training_rows_with_debug_age": int(len(debug_train_y)),
                "validation_rows_with_debug_age": int(len(diagnostic_indices)),
                "validation_debug_age_ms_quantiles": np.quantile(
                    bridge_age_by_run[VALIDATION_ID][diagnostic_indices],
                    [0, .1, .5, .9, 1]).tolist(),
                "permitted_features_only_baseline": _metric(diagnostic_baseline_error),
                "plus_forbidden_bridge_debug_request_age": _metric(diagnostic_debug_error),
                "debug_field": "steering_command_update_age_ms at bridge request boundary",
                "interpretation_limit": "diagnostic only; cannot be used by the competition odometry/controller",
            },
            "heldout_error_by_condition": sorted(
                by_condition.values(),
                key=lambda row: row["rmse_radps"], reverse=True),
            "worst_heldout_phases": sorted(
                worst_phases, key=lambda row: row["rmse_radps"], reverse=True)[:20],
            "heldout_error_by_observed_regime": _error_bands(
                valid_rows, valid_indices, error),
            "training_packet_phase_condition_count": len(train_condition_counts),
            "training_matching_condition_replicates": {
                key: int(value) for key, value in train_condition_counts.items()
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "groups": {key: {
            "training_rows": value["training_rows"],
            "training_phases_by_capture": value["training_phases_by_capture"],
            "validation_yaw_error": value["validation_yaw_error"],
            "lowest_matching_condition_repetitions": min(
                value["training_matching_condition_replicates"].values()),
            "worst_condition": value["heldout_error_by_condition"][0],
        } for key, value in report["groups"].items()},
    }, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run()
