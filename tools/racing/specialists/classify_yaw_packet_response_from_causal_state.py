#!/usr/bin/env python3
"""Historical multi-count selector; do not train under current policy.

The old two-versus-three-step classifier is invalid for current yaw work
because every non-two-packet sample is excluded. Feature extraction utilities
remain importable; current analyses use the exact-two-packet phase labels.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools import analyze_open_plane_dynamics as dynamics

try:
    import analyze_yaw_transition_timing_residuals as timing_audit
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
    import yaw_source_packet_alignment as packet_alignment
except ModuleNotFoundError:
    from tools.racing.specialists import analyze_yaw_transition_timing_residuals as timing_audit
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher
    from tools.racing.specialists import yaw_source_packet_alignment as packet_alignment


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/causal_packet_response_classifier_r04.json")
COMMAND_TOPIC = "/autodrive/roboracer_1/steering_command"
STEERING_LIMIT_RAD = 0.5236
RUNS = {
    "openplane_yaw_error_highsteer_reversal_train_r01_20261008": "r01",
    "openplane_yaw_error_highsteer_reversal_train_r02_20261008": "r02",
    "openplane_yaw_error_highsteer_reversal_validation_r03_20261008": "r03",
    "openplane_yaw_error_highsteer_reversal_train_r04_20261008": "r04",
}
LAGS = (0, 1, 2, 4)
SENSOR_NAMES = (
    "steering_feedback_rad", "throttle_feedback_norm",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps",
    "steering_command_rad", "throttle_command_norm",
    "imu_roll_rad", "imu_roll_rate_rps",
)
EVENT_FEATURE_NAMES = (
    "new_steering_command_rad", "steering_command_change_rad",
    "command_minus_latest_feedback_rad", "command_to_packet_age_ms",
    "command_to_packet_age_fraction_of_25ms",
)
PHASE_RE = re.compile(
    r"^probe_yawerr_(?:highsteer_reversal(?:_repeat)?_|packet_phase_r[0-9]+_)"
    r"(onset|unwind|reversal)_"
    r"v([-+0-9.]+)_a([-+0-9.]+)_turn([+-][0-9]+)_"
    r"delay([-+0-9.]+)_(step|ramp)([-+0-9.]+)s(?:_rep[0-9]+)?$")


def _bag_for(series: Any) -> Path:
    source = ROOT / series.source
    manifest = json.loads(source.with_name("manifest.json").read_text(
        encoding="utf-8"))
    rows = [row for row in manifest.get("runs", [])
            if row.get("run_id") == series.run_id
            and row.get("effective_split") == series.split]
    if len(rows) != 1:
        raise ValueError(f"{series.run_id}: raw bag provenance is ambiguous")
    path = ROOT / rows[0]["bag"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _phase_features(series: Any, suffix: str) -> list[dict[str, Any]]:
    bag = _bag_for(series)
    exact_imu, _ = packet_alignment.exact_imu_by_odom_receipt(bag)
    source_path = ROOT / series.source
    with np.load(source_path, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str).tolist()
        splits = archive["run_splits"].astype(str).tolist()
        run_index = run_ids.index(series.run_id)
        if splits[run_index] != series.split:
            raise ValueError(f"{series.run_id}: archive split changed")
        times = np.asarray(archive["sample_time_ns"], dtype=np.int64)
        sensors = np.asarray(archive["sensor_frames"], dtype=np.float32).copy()
        attitude = np.asarray(archive["imu_attitude_frames"], dtype=np.float32).copy()
        sensor_valid = np.asarray(archive["sensor_valid"], dtype=bool)
        attitude_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
        packet_sequence = np.asarray(archive["packet_sequence"], dtype=np.int64)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        sequence_run = np.asarray(archive["sequence_run_index"], dtype=np.int64)

    exact_match_by_time = exact_imu
    for index, receipt_ns in enumerate(times):
        values = exact_match_by_time.get(int(receipt_ns))
        if values is not None:
            sensors[index, 4:7] = values[:3]
            attitude[index, :] = values[3:]

    sequence_start = np.full(len(times), -1, dtype=np.int64)
    for sequence_id in np.flatnonzero(sequence_run == run_index):
        begin, end = map(int, bounds[int(sequence_id)])
        sequence_start[begin:end] = begin
    source_to_index: dict[int, int] = {}

    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        odom_topic = topics[dynamics.ODOM]
        odom_type = get_message(odom_topic[1])
        source_by_receipt = {}
        for receipt_ns, payload in connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
                (odom_topic[0],)):
            message = deserialize_message(bytes(payload), odom_type)
            source_by_receipt[int(receipt_ns)] = dynamics._stamp_ns(
                message.header.stamp)
        for index, receipt_ns in enumerate(times):
            source_ns = source_by_receipt.get(int(receipt_ns))
            if source_ns is not None:
                source_to_index[source_ns] = index

        command_topic = topics[COMMAND_TOPIC]
        command_type = get_message(command_topic[1])
        commands = []
        for receipt_ns, payload in connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
                (command_topic[0],)):
            message = deserialize_message(bytes(payload), command_type)
            value = float(message.data) * STEERING_LIMIT_RAD
            if np.isfinite(value):
                commands.append((int(receipt_ns), value))
        phases, _end = timing_audit._probe_phases(
            bag, expected_probe_count=216 if suffix == "r03" else None)
    finally:
        connection.close()

    timing_path = timing_audit.TIMING_DIR / f"steering_timing_{suffix}.json"
    timing_report = json.loads(timing_path.read_text(encoding="utf-8"))
    onset_by_phase = {
        row["phase"]: int(row["command_to_feedback_onset_steps"])
        for row in timing_report["packet_grid_measurements"]
    }
    source_stamps = np.asarray(sorted(source_to_index), dtype=np.int64)
    command_stamps = np.asarray([row[0] for row in commands], dtype=np.int64)
    command_values = np.asarray([row[1] for row in commands], dtype=np.float64)
    output: list[dict[str, Any]] = []

    for phase in phases:
        match = PHASE_RE.fullmatch(phase.label)
        if match is None:
            continue
        event, speed, angle, turn, delay, mode, duration = match.groups()
        steps = onset_by_phase.get(phase.label)
        if steps != 2:
            continue
        requested_ns = phase.start_ns + int(float(delay) * 1e9)
        before = ((command_stamps >= requested_ns - 150_000_000)
                  & (command_stamps < requested_ns - 25_000_000))
        if np.count_nonzero(before) < 2:
            continue
        command_before = float(np.median(command_values[before]))
        changed = np.flatnonzero(
            (command_stamps >= requested_ns - 10_000_000)
            & (np.abs(command_values - command_before) >= 0.01))
        if not len(changed):
            continue
        command_index = int(changed[0])
        command_ns = int(command_stamps[command_index])
        odom_index = int(np.searchsorted(source_stamps, command_ns, side="right") - 1)
        if odom_index < 0:
            continue
        source_ns = int(source_stamps[odom_index])
        frame_index = source_to_index[source_ns]
        sequence_begin = int(sequence_start[frame_index])
        if sequence_begin < 0:
            continue
        if any(frame_index - lag < sequence_begin for lag in LAGS):
            continue
        indices = [frame_index - lag for lag in LAGS]
        if not all(sensor_valid[index] and attitude_valid[index]
                   for index in indices):
            continue

        features: list[float] = []
        for index in indices:
            features.extend(float(value) for value in sensors[index, :9])
            features.extend((float(attitude[index, 0]),
                             float(attitude[index, 2])))
        previous_command = float(command_values[command_index - 1])
        command_value = float(command_values[command_index])
        command_age_ms = (command_ns - source_ns) / 1e6
        features.extend((
            command_value, command_value - previous_command,
            command_value - float(sensors[frame_index, 0]),
            command_age_ms, command_age_ms / 25.0,
        ))
        output.append({
            "run_id": series.run_id,
            "split": series.split,
            "phase_label": phase.label,
            "event": event,
            "response_steps": steps,
            "command_receipt_ns": command_ns,
            "source_stamp_at_command_ns": source_ns,
            "changed_command_rad": command_value,
            "features": features,
            "condition": {
                "speed_mps": float(speed), "steering_abs_rad": float(angle),
                "turn_sign": int(turn), "duration_s": float(duration),
                "transition_mode": mode,
                "scheduled_delay_s": float(delay),
            },
            "command_to_packet_age_ms": float(command_age_ms),
        })
    return output


def _feature_names() -> list[str]:
    return [f"{name}_lag{lag * 25}ms"
            for lag in LAGS for name in SENSOR_NAMES] + list(EVENT_FEATURE_NAMES)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int8)
    predicted = np.asarray(prediction, dtype=np.int8)
    matrix = confusion_matrix(labels, predicted, labels=[2, 3])
    recalls = []
    by_class = {}
    for row, label in enumerate((2, 3)):
        n = int(np.count_nonzero(labels == label))
        recall = float(matrix[row, row] / n) if n else None
        if recall is not None:
            recalls.append(recall)
        precision_denom = int(np.count_nonzero(predicted == label))
        precision = (float(matrix[row, row] / precision_denom)
                     if precision_denom else 0.0)
        by_class[str(label)] = {"support": n, "recall": recall,
                                "precision": precision}
    return {
        "samples": int(len(labels)),
        "class_counts": {str(label): int(np.count_nonzero(labels == label))
                         for label in (2, 3)},
        "accuracy": float(np.mean(labels == predicted)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "confusion_matrix_rows_true_2_3_columns_pred_2_3": matrix.tolist(),
        "by_class": by_class,
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    all_rows = []
    for run_id, suffix in RUNS.items():
        matches = [row for row in admitted if row.run_id == run_id]
        if len(matches) != 1:
            raise ValueError(f"expected one admitted run: {run_id}")
        rows = _phase_features(matches[0], suffix)
        all_rows.extend(rows)
        print(f"extracted {len(rows)} labeled 2/3-packet phases from {run_id}",
              flush=True)

    train = [row for row in all_rows if row["split"] == "train"]
    validation = [row for row in all_rows if row["split"] == "validation"]
    x_train = np.asarray([row["features"] for row in train], dtype=np.float64)
    y_train = np.asarray([row["response_steps"] for row in train], dtype=np.int8)
    x_val = np.asarray([row["features"] for row in validation], dtype=np.float64)
    y_val = np.asarray([row["response_steps"] for row in validation], dtype=np.int8)
    if not len(train) or not len(validation) or x_train.shape[1] != x_val.shape[1]:
        raise ValueError("phase feature extraction produced incomplete train/validation")

    majority = np.full(len(y_val), 2, dtype=np.int8)
    train_age = np.asarray([row["command_to_packet_age_ms"] for row in train])
    val_age = np.asarray([row["command_to_packet_age_ms"] for row in validation])
    threshold = float(0.5 * (np.median(train_age[y_train == 2])
                            + np.median(train_age[y_train == 3])))
    age_prediction = np.where(val_age >= threshold, 3, 2).astype(np.int8)

    models = {
        "balanced_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(
                class_weight="balanced", C=0.1, max_iter=3000,
                random_state=20261008)),
        "balanced_extra_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            max_features=0.8, class_weight="balanced",
            random_state=20261008, n_jobs=4),
        "balanced_random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            max_features=0.8, class_weight="balanced_subsample",
            random_state=20261008, n_jobs=4),
    }
    results: dict[str, Any] = {
        "majority_two_packet": _metrics(y_val, majority),
        "command_packet_age_threshold": {
            "threshold_ms_fit_on_training_runs": threshold,
            **_metrics(y_val, age_prediction),
        },
    }
    fitted_models = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted_models[name] = model
        results[name] = _metrics(y_val, model.predict(x_val))

    context_start = len(LAGS) * len(SENSOR_NAMES)
    feature_ablation_columns = {
        "sensor_history_without_command_event_context": np.arange(context_start),
        "command_target_change_and_gap_only": np.arange(context_start,
                                                        context_start + 3),
        "command_to_packet_age_only": np.arange(context_start + 3,
                                                  context_start + 5),
        "full_command_event_context": np.arange(context_start, x_train.shape[1]),
    }
    ablations = {}
    for name, columns in feature_ablation_columns.items():
        model = ExtraTreesClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            max_features=0.8, class_weight="balanced",
            random_state=20261008, n_jobs=4)
        model.fit(x_train[:, columns], y_train)
        ablations[name] = _metrics(y_val, model.predict(x_val[:, columns]))

    selected_model = fitted_models["balanced_extra_trees"]
    feature_names = _feature_names()
    if len(feature_names) != x_train.shape[1]:
        raise ValueError("causal response classifier feature-name width mismatch")
    group_columns: dict[str, list[int]] = {}
    for channel_index, channel in enumerate(SENSOR_NAMES):
        if channel.startswith("steering_feedback"):
            group = "measured_steering_history"
        elif channel.startswith("steering_command"):
            group = "past_steering_command_history"
        elif channel.startswith("throttle"):
            group = "throttle_history"
        elif channel.startswith("rear_"):
            group = "rear_encoder_speed_history"
        elif channel.startswith("imu_a"):
            group = "imu_acceleration_history"
        elif channel.startswith("imu_yaw"):
            group = "imu_yaw_history"
        else:
            group = "imu_attitude_history"
        group_columns.setdefault(group, []).extend(
            lag_index * len(SENSOR_NAMES) + channel_index
            for lag_index in range(len(LAGS)))
    group_columns["command_event_timing_and_target"] = list(
        range(len(LAGS) * len(SENSOR_NAMES), x_train.shape[1]))
    rng = np.random.default_rng(20261008)
    permutation_importance = {}
    for group, columns in group_columns.items():
        scores = []
        for _ in range(40):
            order = rng.permutation(len(x_val))
            perturbed = x_val.copy()
            perturbed[:, columns] = x_val[order][:, columns]
            scores.append(balanced_accuracy_score(
                y_val, selected_model.predict(perturbed)))
        values = np.asarray(scores, dtype=np.float64)
        permutation_importance[group] = {
            "balanced_accuracy_after_group_permutation_mean": float(values.mean()),
            "balanced_accuracy_after_group_permutation_p05_p95": [
                float(np.quantile(values, 0.05)),
                float(np.quantile(values, 0.95))],
            "drop_from_unpermuted_balanced_accuracy": float(
                results["balanced_extra_trees"]["balanced_accuracy"]
                - values.mean()),
        }

    event_results = {}
    for event in sorted({row["event"] for row in validation}):
        mask = np.asarray([row["event"] == event for row in validation])
        event_results[event] = {}
        event_results[event]["majority_two_packet"] = _metrics(
            y_val[mask], majority[mask])
        event_results[event]["command_packet_age_threshold"] = _metrics(
            y_val[mask], age_prediction[mask])
        for name, model in models.items():
            event_results[event][name] = _metrics(
                y_val[mask], model.predict(x_val[mask]))

    result = {
        "title": "Can current legal sensor/command history identify steering response packet phase?",
        "label_source": "offline bridge packet timing: command-to-measured-steering onset steps",
        "predictor_inputs": (
            "current/past permitted sensor and actuator values at the latest packet preceding the changed steering command, exact-source IMU alignment, new command/delta, and command-to-source-packet age"),
        "simulator_truth_used_as_predictor": False,
        "bridge_timing_used_as_predictor": False,
        "split": {
            "training_runs": sorted({row["run_id"] for row in train}),
            "validation_run": sorted({row["run_id"] for row in validation}),
            "train_phase_count": len(train),
            "validation_phase_count": len(validation),
            "validation_runs_are_whole_run_held_out": True,
            "test_and_final_test_opened": False,
        },
        "validation_metrics": results,
        "feature_ablation_metrics": ablations,
        "input_feature_names": feature_names,
        "validation_group_permutation_importance": permutation_importance,
        "validation_by_event": event_results,
        "training_class_counts": {
            str(label): int(np.count_nonzero(y_train == label))
            for label in (2, 3)},
        "validation_class_counts": {
            str(label): int(np.count_nonzero(y_val == label))
            for label in (2, 3)},
        "training_validation_source_audit": source_audit,
        "interpretation": [
            "A phase selector is not usable merely because overall accuracy is high under class imbalance; balanced accuracy and 3-step recall are decisive.",
            "Bridge timing is diagnostic label only; it is not permitted as a runtime predictor.",
            "A failed selector does not prove the yaw responses are unmodelable; it means this class label is not recoverable from these causal inputs at command time.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = output.with_suffix(".joblib")
    joblib.dump({
        "model": selected_model,
        "feature_names": feature_names,
        "classes": [2, 3],
        "contract": result["predictor_inputs"],
        "training_run_ids": result["split"]["training_runs"],
        "validation_run_ids": result["split"]["validation_run"],
        "validation_metrics": results["balanced_extra_trees"],
        "research_only": True,
    }, model_path, compress=3)
    result["classifier_artifact"] = str(model_path.relative_to(ROOT))
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)),
                      "classifier_artifact": str(model_path.relative_to(ROOT)),
                      "validation_metrics": results,
                      "feature_ablation_metrics": ablations,
                      "validation_group_permutation_importance": permutation_importance,
                      "validation_by_event": event_results}, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated multi-count selector: non-two-packet data are invalid "
        "for current yaw analysis.")
