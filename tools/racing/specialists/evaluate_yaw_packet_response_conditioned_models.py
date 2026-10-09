#!/usr/bin/env python3
"""Disabled historical evaluator; active yaw analysis admits exactly two packets.

Old multi-count results are invalid and superseded. Shared bag/feature helpers
remain imported by focused evaluators, but no fitting, routing, or scoring path
in this module may admit a response count other than exactly two.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tools import analyze_open_plane_dynamics as dynamics

try:
    import analyze_yaw_transition_timing_residuals as timing
    import audit_yaw_exact_packet_sensor_alignment as alignment_audit
    import audit_sensor_yaw_large_errors as audit
    import classify_yaw_packet_response_from_causal_state as selector
    import fit_sensor_only_yaw_regime_atlas as atlas
    import score_yaw_atlas_transition_events as phase_tools
    import train_yaw_multihorizon_teacher as teacher
    import yaw_source_packet_alignment as packet_alignment
except ModuleNotFoundError:
    from tools.racing.specialists import analyze_yaw_transition_timing_residuals as timing
    from tools.racing.specialists import audit_yaw_exact_packet_sensor_alignment as alignment_audit
    from tools.racing.specialists import audit_sensor_yaw_large_errors as audit
    from tools.racing.specialists import classify_yaw_packet_response_from_causal_state as selector
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import score_yaw_atlas_transition_events as phase_tools
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher
    from tools.racing.specialists import yaw_source_packet_alignment as packet_alignment


ROOT = Path(__file__).resolve().parents[3]
TIMING_SUFFIX = {
    "openplane_yaw_error_highsteer_reversal_train_r01_20261008": "r01",
    "openplane_yaw_error_highsteer_reversal_train_r02_20261008": "r02",
    "openplane_yaw_error_highsteer_reversal_validation_r03_20261008": "r03",
    "openplane_yaw_error_highsteer_reversal_train_r04_20261008": "r04",
    "openplane_yaw_error_highsteer_reversal_train_r06_20261008": "r06",
    "openplane_yaw_error_packet_phase_validation_r07_20261008": "r07",
    "openplane_yaw_error_packet_phase_validation_r09_20261008": "r09",
    "openplane_yaw_error_highsteer_reversal_residual_train_r10_20261008": "r10",
}
CLASSIFIER_PATH = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_large_error_audit_v1/causal_packet_response_classifier.joblib")
EXACT_ATLAS_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                   "sensor_only_yaw_regime_atlas_exact_packet_v1")
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/response_conditioned_yaw_models_r06_cause_and_persistence.json")
WINDOW_BEFORE_NS = 100_000_000
WINDOW_AFTER_NS = 500_000_000
EXPANDED_HISTORY_LAGS = (0, 1, 2, 4, 8, 12, 20)
STEERING_FEEDBACK_TOPIC = dynamics.STEERING
STEERING_COMMAND_TOPIC = "/autodrive/roboracer_1/steering_command"
STEERING_LIMIT_RAD = 0.5236
TIMING_FEATURE_NAMES = (
    "latest_steering_command_receipt_minus_odom_source_ms",
    "latest_steering_command_age_at_odom_receipt_ms",
    "latest_steering_feedback_receipt_minus_odom_source_ms",
    "latest_steering_feedback_age_at_odom_receipt_ms",
)


def _run_rows(series: Any,
              history_lags: tuple[int, ...] = EXPANDED_HISTORY_LAGS
              ) -> dict[str, Any]:
    run = atlas._read_sensor_run(
        series, history_lags=history_lags, exact_packet_imu=True)
    source = ROOT / series.source
    with np.load(source, allow_pickle=False) as archive:
        run_index = archive["run_ids"].astype(str).tolist().index(series.run_id)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        sequence_runs = np.asarray(archive["sequence_run_index"], dtype=np.int64)
        all_times = np.asarray(archive["sample_time_ns"], dtype=np.int64)
    sequence_times = []
    for sequence_id in np.flatnonzero(sequence_runs == run_index):
        begin, end = map(int, bounds[int(sequence_id)])
        if end - begin > max(history_lags) + 1:
            sequence_times.append(all_times[begin:end])
    if len(sequence_times) != len(run["sequences"]):
        raise ValueError(f"{series.run_id}: sequence timestamp provenance mismatch")
    run = {**run, "sequence_times": sequence_times}
    rows = atlas._rows(run, "command_intent", history_lags)
    times = alignment_audit._row_sample_times(run, history_lags)
    if len(times) != len(rows["residual"]):
        raise ValueError(f"{series.run_id}: timestamp/row alignment mismatch")
    source_by_receipt = packet_alignment.odom_source_stamp_by_receipt(
        alignment_audit._bag_for_run(series))
    source_stamps = np.asarray([
        source_by_receipt.get(int(receipt), -1) for receipt in times],
        dtype=np.int64)
    if np.any(source_stamps <= 0):
        raise ValueError(f"{series.run_id}: missing source stamps for yaw rows")
    bag = alignment_audit._bag_for_run(series)
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        command_topic = topics[STEERING_COMMAND_TOPIC]
        command_type = get_message(command_topic[1])
        command_samples = [
            (int(receipt), float(message.data) * STEERING_LIMIT_RAD)
            for receipt, payload in connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
                (command_topic[0],))
            for message in (deserialize_message(bytes(payload), command_type),)
            if np.isfinite(float(message.data))
        ]
        feedback = [
            (int(receipt), float(message.data) * STEERING_LIMIT_RAD)
            for receipt, message in dynamics._messages(
                connection, topics, STEERING_FEEDBACK_TOPIC)
            if np.isfinite(float(message.data))
        ]
    finally:
        connection.close()
    if not command_samples or not feedback:
        raise ValueError(f"{series.run_id}: missing steering command/feedback")
    command_times = np.asarray([row[0] for row in command_samples], dtype=np.int64)
    feedback_times = np.asarray([row[0] for row in feedback], dtype=np.int64)
    command_index = np.searchsorted(command_times, times, side="right") - 1
    feedback_index = np.searchsorted(feedback_times, times, side="right") - 1
    if np.any(command_index < 0) or np.any(feedback_index < 0):
        raise ValueError(f"{series.run_id}: yaw rows precede actuator samples")
    command_receipt = command_times[command_index]
    feedback_receipt = feedback_times[feedback_index]
    precise_timing = np.column_stack((
        (command_receipt - source_stamps) / 1.0e6,
        (times - command_receipt) / 1.0e6,
        (feedback_receipt - source_stamps) / 1.0e6,
        (times - feedback_receipt) / 1.0e6,
    )).astype(np.float32)
    base_history_width = len(atlas.HISTORY_LAGS) * len(atlas.OBSERVATION_NAMES)
    derived_width = len(atlas.DERIVED_NAMES)
    x_atlas = np.column_stack((rows["x"][:, :base_history_width],
                               rows["x"][:, -derived_width:]))
    next_steering = []
    next_steering_valid = []
    gt_body_velocity = []
    for sensors, sensor_valid, attitude, attitude_valid, rigid in run["sequences"]:
        if (not np.isfinite(sensors).all() or not np.isfinite(attitude).all()
                or not np.isfinite(rigid).all()):
            continue
        for k in range(max(history_lags), len(sensors) - 1):
            if (not all(sensor_valid[k - lag] for lag in history_lags)
                    or not all(attitude_valid[k - lag] for lag in history_lags)):
                continue
            wheel_mean = 0.5 * float(sensors[k, 2] + sensors[k, 3])
            speed_cell = int(wheel_mean // atlas.SPEED_BIN_MPS)
            steering = float(sensors[k, 0])
            if (speed_cell < 0 or speed_cell >= len(atlas.SPEED_CENTERS)
                    or steering < atlas.STEERING_CENTERS[0] - 0.0125
                    or steering > atlas.STEERING_CENTERS[-1] + 0.0125):
                continue
            next_steering.append(float(sensors[k + 1, 0]))
            next_steering_valid.append(bool(sensor_valid[k + 1]))
            gt_body_velocity.append((float(rigid[k, 7]), float(rigid[k, 8])))
    next_steering = np.asarray(next_steering, dtype=np.float32)
    next_steering_valid = np.asarray(next_steering_valid, dtype=bool)
    gt_body_velocity = np.asarray(gt_body_velocity, dtype=np.float32)
    if len(next_steering) != len(rows["residual"]):
        raise ValueError(f"{series.run_id}: next-steering labels do not align")
    if gt_body_velocity.shape != (len(rows["residual"]), 2):
        raise ValueError(f"{series.run_id}: GT body velocity rows do not align")
    # Keep the frozen 100-ms atlas contract separate from the candidate. The
    # candidate must consume every requested 0–500-ms lag; previously this
    # field accidentally reused x_atlas and silently discarded lags 200–500ms.
    return {**rows, "x_extended": rows["x"], "x": x_atlas,
            "x_short_timed": np.column_stack((x_atlas, precise_timing)),
            "x_timed": np.column_stack((rows["x"], precise_timing)),
            "precise_timing": precise_timing, "sample_time_ns": times,
            "next_steering": next_steering,
            "next_steering_valid": next_steering_valid,
            "gt_body_velocity": gt_body_velocity}


def _phase_metadata(series: Any, run_rows: dict[str, Any], suffix: str
                    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    bag = alignment_audit._bag_for_run(series)
    phases, _ = timing._probe_phases(
        bag, expected_probe_count=216 if suffix == "r03" else None)
    timing_path = timing.TIMING_DIR / f"steering_timing_{suffix}.json"
    timing_report = json.loads(timing_path.read_text(encoding="utf-8"))
    step_by_phase = {
        row["phase"]: int(row["command_to_feedback_onset_steps"])
        for row in timing_report["packet_grid_measurements"]
    }
    response_count_histogram: dict[str, int] = defaultdict(int)
    for steps in step_by_phase.values():
        response_count_histogram[str(steps)] += 1
    observed = selector._phase_features(series, suffix)
    observed_by_phase = {row["phase_label"]: row for row in observed}
    if len(observed_by_phase) != len(observed):
        raise ValueError(f"{series.run_id}: duplicate classifier phase rows")
    phase_index, phase_labels = phase_tools._phase_lookup(
        run_rows["sample_time_ns"], phases)
    label_to_phase = {label: phase for label, phase in zip(phase_labels, phases)}
    rows_by_phase: dict[str, list[int]] = defaultdict(list)
    for index, phase_id in enumerate(phase_index):
        if phase_id >= 0:
            rows_by_phase[phase_labels[int(phase_id)]].append(index)

    metadata: dict[str, dict[str, Any]] = {}
    for label, phase in label_to_phase.items():
        steps = step_by_phase.get(label)
        observed_row = observed_by_phase.get(label)
        if steps != 2 or observed_row is None:
            continue
        command_ns = int(observed_row["command_receipt_ns"])
        # Probe data are scored only around the actual changed command, not
        # the quiet lead-in or later unrelated phase transitions.
        indices = [index for index in rows_by_phase.get(label, ())
                   if command_ns - WINDOW_BEFORE_NS
                   <= int(run_rows["sample_time_ns"][index])
                   <= command_ns + WINDOW_AFTER_NS]
        if not indices:
            continue
        metadata[label] = {
            "event": str(observed_row["event"]),
            "true_steps": int(steps),
            "command_receipt_ns": command_ns,
            "indices": indices,
            "condition": observed_row["condition"],
            "phase_valid": bool(phase.valid),
        }
    invalid_response_counts = {
        count: total for count, total in sorted(response_count_histogram.items())
        if int(count) != 2
    }
    return metadata, {
        "candidate_phases": len(observed),
        "scored_phases": len(metadata),
        "response_phase_count_histogram": dict(sorted(
            response_count_histogram.items(), key=lambda item: int(item[0]))),
        "invalid_response_phases_excluded": int(sum(
            invalid_response_counts.values())),
        "invalid_response_counts_excluded": invalid_response_counts,
    }


def _collect(admitted: list[Any]
             ) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [series for series in admitted
                if series.run_id in TIMING_SUFFIX]
    if {series.run_id for series in selected} != set(TIMING_SUFFIX):
        missing = sorted(set(TIMING_SUFFIX) - {series.run_id for series in selected})
        raise ValueError(f"required whole-run probe captures not admitted: {missing}")
    rows_by_run: dict[str, Any] = {}
    phases_by_run: dict[str, Any] = {}
    phase_audits = {}
    for series in selected:
        rows = _run_rows(series)
        phases, phase_audit = _phase_metadata(
            series, rows, TIMING_SUFFIX[series.run_id])
        rows_by_run[series.run_id] = rows
        phases_by_run[series.run_id] = phases
        phase_audits[series.run_id] = {
            **phase_audit,
            "packet_response_steps_included": [2],
            "one_step_rows": int(len(rows["residual"])),
            "window_rows": int(sum(len(row["indices"])
                                    for row in phases.values())),
        }
        print(f"loaded {series.run_id}: {len(rows['residual'])} rows, "
              f"{len(phases)} scored response phases", flush=True)
    return {"rows": rows_by_run, "phases": phases_by_run}, phase_audits


def _collect_exact_two_packet(admitted: list[Any]
                              ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect only exactly-two-packet phases; all other counts are excluded."""
    return _collect(admitted)


def _phase_sample_indices(phases: dict[str, dict[str, Any]],
                          step: int | None = None) -> np.ndarray:
    return np.asarray([index for phase in phases.values()
                       if step is None or phase["true_steps"] == step
                       for index in phase["indices"]], dtype=np.int64)


def _fit_branch(rows: dict[str, Any], indices: np.ndarray,
                feature_key: str = "x_timed") -> ExtraTreesRegressor:
    if len(indices) < 100:
        raise ValueError(f"insufficient response-branch training rows: {len(indices)}")
    return ExtraTreesRegressor(
        n_estimators=240, max_depth=14, min_samples_leaf=4,
        max_features=0.9, random_state=20261008, n_jobs=4,
    ).fit(rows[feature_key][indices], rows["residual"][indices])


def _route_by_response_and_event(
        rows: dict[str, Any], phases: dict[str, dict[str, Any]],
        models: dict[tuple[int, str], ExtraTreesRegressor],
        fallback_models: dict[int, ExtraTreesRegressor],
        step_key: str, feature_key: str = "x_timed") -> np.ndarray:
    """Route each transition row by response class and causal row event."""
    prediction = np.full(len(rows["residual"]), np.nan, dtype=np.float32)
    row_events = rows["event"].astype(str)
    for phase in phases.values():
        step = (int(phase["true_steps"]) if step_key == "true_steps"
                else phase["predicted_steps"])
        if step != 2:
            continue
        indices = np.asarray(phase["indices"], dtype=np.int64)
        for event in np.unique(row_events[indices]):
            event_indices = indices[row_events[indices] == event]
            model = models.get((int(step), str(event)), fallback_models[int(step)])
            prediction[event_indices] = model.predict(
                rows[feature_key][event_indices]).astype(np.float32)
    return prediction


def _metric(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return {"samples": 0}
    absolute = np.abs(errors)
    return {
        "samples": int(len(errors)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
        "fraction_within_0p1": float(np.mean(absolute <= 0.1)),
    }


def _diagnose_group_residuals(
        rows: dict[str, Any], phases: dict[str, dict[str, Any]],
        prediction: np.ndarray) -> dict[str, Any]:
    """Stratify the frozen-selector residual by transition timing/outcomes.

    Measured next steering is deliberately used only as an outcome stratum
    after inference. It is not part of the predictor or selector inputs.
    """
    observation_index = {
        name: atlas.OBSERVATION_NAMES.index(name)
        for name in ("steering_feedback_rad", "steering_command_rad")
    }
    group_specs = (
        ("reversal/2_packet", "reversal", 2),
        ("unwind/2_packet", "unwind", 2),
    )
    time_bins = (
        ("before_command", -np.inf, 0.0),
        ("0_25ms", 0.0, 25.0),
        ("25_50ms", 25.0, 50.0),
        ("50_100ms", 50.0, 100.0),
        ("100_250ms", 100.0, 250.0),
        ("250_500ms", 250.0, 500.000001),
    )
    result: dict[str, Any] = {
        "packet_class_is_evaluation_label_only": True,
        "measured_next_steering_is_outcome_stratification_only": True,
        "future_measurements_used_as_predictor_inputs": False,
        "groups": {},
    }
    for label, event, step in group_specs:
        samples: list[dict[str, Any]] = []
        for phase_label, phase in phases.items():
            if phase["event"] != event or phase["true_steps"] != step:
                continue
            for index in phase["indices"]:
                feedback = float(rows["x"][index,
                    observation_index["steering_feedback_rad"]])
                command = float(rows["x"][index,
                    observation_index["steering_command_rad"]])
                next_valid = bool(rows["next_steering_valid"][index])
                next_feedback = (float(rows["next_steering"][index])
                                 if next_valid else None)
                samples.append({
                    "index": int(index),
                    "error": float(prediction[index]
                                    - rows["residual"][index]),
                    "age_ms": float((rows["sample_time_ns"][index]
                                     - phase["command_receipt_ns"]) / 1.0e6),
                    "next_feedback_delta_abs_rad": (
                        abs(next_feedback - feedback)
                        if next_feedback is not None else None),
                    "command_feedback_gap_abs_rad": abs(command - feedback),
                    "condition": phase["condition"],
                    "phase": phase_label,
                })

        def score(selected: list[dict[str, Any]]) -> dict[str, Any]:
            return _metric(np.asarray([item["error"] for item in selected],
                                      dtype=np.float64))

        by_time = {}
        for name, lower, upper in time_bins:
            selected = [item for item in samples
                        if lower <= item["age_ms"] < upper]
            if selected:
                by_time[name] = score(selected)

        next_feedback_bins = (
            ("no_change_le_0p005_rad", lambda value: value <= 0.005),
            ("small_change_0p005_to_0p05_rad",
             lambda value: 0.005 < value <= 0.05),
            ("change_gt_0p05_rad", lambda value: value > 0.05),
        )
        by_next_feedback = {}
        for name, predicate in next_feedback_bins:
            selected = [item for item in samples
                        if item["next_feedback_delta_abs_rad"] is not None
                        and predicate(item["next_feedback_delta_abs_rad"])]
            if selected:
                by_next_feedback[name] = score(selected)

        gap_bins = (
            ("gap_le_0p05_rad", lambda value: value <= 0.05),
            ("gap_0p05_to_0p25_rad",
             lambda value: 0.05 < value <= 0.25),
            ("gap_gt_0p25_rad", lambda value: value > 0.25),
        )
        by_gap = {}
        for name, predicate in gap_bins:
            selected = [item for item in samples
                        if predicate(item["command_feedback_gap_abs_rad"])]
            if selected:
                by_gap[name] = score(selected)

        condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in samples:
            condition_rows[json.dumps(item["condition"], sort_keys=True)].append(item)
        by_condition = {
            key: score(condition_samples)
            for key, condition_samples in sorted(condition_rows.items())
        }
        result["groups"][label] = {
            "phase_count": len({item["phase"] for item in samples}),
            "all": score(samples),
            "by_time_from_command_receipt": by_time,
            "by_measured_next_feedback_change_diagnostic_only": by_next_feedback,
            "by_current_command_feedback_gap": by_gap,
            "by_probe_condition": by_condition,
        }
    return result


def _score_rows(rows: dict[str, Any], indices: np.ndarray,
                prediction: np.ndarray) -> dict[str, Any]:
    error = prediction[indices] - rows["residual"][indices]
    return _metric(error)


def _bootstrap_phase_rmse(rows: dict[str, Any],
                          phases: dict[str, dict[str, Any]],
                          prediction: np.ndarray, step: int,
                          draws: int = 5000) -> list[float] | None:
    phase_errors = []
    for row in phases.values():
        if row["true_steps"] != step:
            continue
        idx = np.asarray(row["indices"], dtype=np.int64)
        errors = prediction[idx] - rows["residual"][idx]
        if len(errors):
            phase_errors.append(float(np.sqrt(np.mean(errors * errors))))
    if len(phase_errors) < 2:
        return None
    rng = np.random.default_rng(20261008 + step)
    values = np.asarray(phase_errors, dtype=np.float64)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975))]


def run(output: Path = OUTPUT) -> dict[str, Any]:
    raise RuntimeError(
        "historical multi-count evaluator is disabled; use the exact-two-packet "
        "targeted evaluator")

    selector_bundle = joblib.load(CLASSIFIER_PATH)
    classifier = selector_bundle["model"]
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audits = _collect(admitted, classifier)
    run_rows = collected["rows"]
    run_phases = collected["phases"]
    train_ids = sorted(run_id for run_id in TIMING_SUFFIX
                       if "_train_" in run_id)
    validation_id = next(run_id for run_id in TIMING_SUFFIX
                         if "_validation_" in run_id)

    train_rows = {
        key: np.concatenate([run_rows[run_id][key] for run_id in train_ids], axis=0)
        for key in run_rows[train_ids[0]]
        if key != "sample_time_ns"
    }
    train_phases: dict[str, dict[str, Any]] = {}
    cursor = 0
    for run_id in train_ids:
        for label, phase in run_phases[run_id].items():
            train_phases[f"{run_id}:{label}"] = {
                **phase,
                "indices": [cursor + index for index in phase["indices"]],
            }
        cursor += len(run_rows[run_id]["residual"])

    validation_rows = run_rows[validation_id]
    validation_phases = run_phases[validation_id]
    branch_models = {}
    short_branch_models = {}
    branch_train_indices = {}
    for step in (2,):
        indices = _phase_sample_indices(train_phases, step)
        branch_train_indices[step] = indices
        branch_models[step] = _fit_branch(train_rows, indices)
        short_branch_models[step] = _fit_branch(
            train_rows, indices, "x_short_timed")

    event_models: dict[tuple[int, str], ExtraTreesRegressor] = {}
    short_event_models: dict[tuple[int, str], ExtraTreesRegressor] = {}
    event_train_indices: dict[tuple[int, str], np.ndarray] = {}
    event_support: dict[str, Any] = {}
    for step in (2,):
        for event in sorted(set(train_rows["event"].astype(str))):
            phase_names = [name for name, phase in train_phases.items()
                           if phase["true_steps"] == step]
            indices = np.asarray([
                index for name in phase_names
                for index in train_phases[name]["indices"]
                if str(train_rows["event"][index]) == event
            ], dtype=np.int64)
            phase_support = sum(
                any(str(train_rows["event"][index]) == event
                    for index in train_phases[name]["indices"])
                for name in phase_names)
            fit_eligible = len(indices) >= 100 and phase_support >= 5
            event_support[f"{step}/{event}"] = {
                "rows": int(len(indices)),
                "training_phases": int(phase_support),
                "specialist_fitted": bool(fit_eligible),
                "fallback": f"{step}-packet pooled transition specialist",
            }
            if fit_eligible:
                event_models[(step, event)] = _fit_branch(train_rows, indices)
                short_event_models[(step, event)] = _fit_branch(
                    train_rows, indices, "x_short_timed")
                event_train_indices[(step, event)] = indices

    exact_report = json.loads((EXACT_ATLAS_DIR /
        "sensor_only_yaw_regime_atlas_report.json").read_text(encoding="utf-8"))
    exact_bundle = joblib.load(EXACT_ATLAS_DIR / "sensor_only_yaw_regime_atlas.joblib")
    _local, _global, baseline_prediction = audit._predict(
        validation_rows, exact_bundle,
        len(exact_report["input_contract"]["features"]))
    branch_predictions = {
        step: branch_models[step].predict(
            validation_rows["x_timed"]).astype(np.float32)
        for step in (2,)
    }
    true_routed = baseline_prediction.copy()
    predicted_routed = baseline_prediction.copy()
    confident_routed = baseline_prediction.copy()
    route_mask = np.zeros(len(validation_rows["residual"]), dtype=bool)
    selector_correct = []
    phase_rows: list[dict[str, Any]] = []
    for label, phase in validation_phases.items():
        true_step = int(phase["true_steps"])
        predicted_step = phase["predicted_steps"]
        indices = np.asarray(phase["indices"], dtype=np.int64)
        true_routed[indices] = branch_predictions[true_step][indices]
        route_mask[indices] = True
        if predicted_step == 2:
            predicted_routed[indices] = branch_predictions[int(predicted_step)][indices]
            selector_correct.append(int(predicted_step == true_step))
            if float(phase["selector_confidence"] or 0.0) >= 0.8:
                confident_routed[indices] = branch_predictions[int(predicted_step)][indices]
        phase_rows.append({
            "phase": label,
            "event": phase["event"],
            "true_response_steps": true_step,
            "predicted_response_steps": predicted_step,
            "selector_confidence": phase["selector_confidence"],
            "condition": phase["condition"],
            "rows": len(indices),
        })

    oracle_event_routed = _route_by_response_and_event(
        validation_rows, validation_phases, event_models, branch_models,
        "true_steps")
    classifier_event_routed = _route_by_response_and_event(
        validation_rows, validation_phases, event_models, branch_models,
        "predicted_steps")
    short_oracle_event_routed = _route_by_response_and_event(
        validation_rows, validation_phases, short_event_models,
        short_branch_models, "true_steps", "x_short_timed")
    short_classifier_event_routed = _route_by_response_and_event(
        validation_rows, validation_phases, short_event_models,
        short_branch_models, "predicted_steps", "x_short_timed")
    event_route_mask = np.isfinite(oracle_event_routed)
    classifier_event_route_mask = np.isfinite(classifier_event_routed)
    oracle_event_routed_filled = np.where(
        event_route_mask, oracle_event_routed, baseline_prediction)
    classifier_event_routed_filled = np.where(
        classifier_event_route_mask, classifier_event_routed,
        baseline_prediction)
    short_oracle_event_routed_filled = np.where(
        np.isfinite(short_oracle_event_routed), short_oracle_event_routed,
        baseline_prediction)
    short_classifier_event_routed_filled = np.where(
        np.isfinite(short_classifier_event_routed), short_classifier_event_routed,
        baseline_prediction)

    results: dict[str, Any] = {}
    for label, indices in (
            ("all_scored_transition_rows", np.flatnonzero(route_mask)),
            ("all_valid_validation_rows", np.arange(len(route_mask)))):
        results[label] = {
            "exact_packet_fullband_atlas": _score_rows(
                validation_rows, indices, baseline_prediction),
            "oracle_exact_two_packet_routing": _score_rows(
                validation_rows, indices, true_routed),
            "causal_classifier_routing": _score_rows(
                validation_rows, indices, predicted_routed),
            "oracle_response_and_current_event_routing": _score_rows(
                validation_rows, indices, oracle_event_routed_filled),
            "causal_response_classifier_and_current_event_routing": _score_rows(
                validation_rows, indices, classifier_event_routed_filled),
            "causal_classifier_routing_confidence_ge_0p8_else_atlas": _score_rows(
                validation_rows, indices, confident_routed),
            "matched_history_comparison": {
                "100ms_causal_event_and_response_model": _score_rows(
                    validation_rows, indices, short_classifier_event_routed_filled),
                "500ms_causal_event_and_response_model": _score_rows(
                    validation_rows, indices, classifier_event_routed_filled),
                "100ms_oracle_event_and_response_model": _score_rows(
                    validation_rows, indices, short_oracle_event_routed_filled),
                "500ms_oracle_event_and_response_model": _score_rows(
                    validation_rows, indices, oracle_event_routed_filled),
            },
        }

    by_group: dict[str, Any] = {}
    for step in (2,):
        group_phases = {name: row for name, row in validation_phases.items()
                        if row["true_steps"] == step}
        indices = _phase_sample_indices(group_phases)
        by_group[str(step)] = {
            "phase_count": len(group_phases),
            "rows": len(indices),
            "selector_correct_phases": sum(
                row["predicted_steps"] == step for row in group_phases.values()),
            "selector_phase_recall": float(np.mean([
                row["predicted_steps"] == step for row in group_phases.values()])),
            "phase_macro_rmse_bootstrap_95pct_ci_radps": {
                "oracle": _bootstrap_phase_rmse(
                    validation_rows, group_phases, true_routed, step),
                "classifier": _bootstrap_phase_rmse(
                    validation_rows, group_phases, predicted_routed, step),
            },
            "exact_packet_fullband_atlas": _score_rows(
                validation_rows, indices, baseline_prediction),
            "oracle_branch": _score_rows(
                validation_rows, indices, true_routed),
            "classifier_routed": _score_rows(
                validation_rows, indices, predicted_routed),
            "oracle_event_branch": _score_rows(
                validation_rows, indices, oracle_event_routed_filled),
            "classifier_event_branch": _score_rows(
                validation_rows, indices, classifier_event_routed_filled),
            "matched_history_comparison": {
                "100ms_causal_event_and_response_model": _score_rows(
                    validation_rows, indices, short_classifier_event_routed_filled),
                "500ms_causal_event_and_response_model": _score_rows(
                    validation_rows, indices, classifier_event_routed_filled),
                "100ms_oracle_event_and_response_model": _score_rows(
                    validation_rows, indices, short_oracle_event_routed_filled),
                "500ms_oracle_event_and_response_model": _score_rows(
                    validation_rows, indices, oracle_event_routed_filled),
            },
        }

    by_event: dict[str, Any] = {}
    for event in ("onset", "unwind", "reversal"):
        group_phases = {name: row for name, row in validation_phases.items()
                        if row["event"] == event}
        indices = _phase_sample_indices(group_phases)
        if len(indices):
            by_event[event] = {
                "phase_count": len(group_phases),
                "rows": len(indices),
                "exact_packet_fullband_atlas": _score_rows(
                    validation_rows, indices, baseline_prediction),
                "oracle_branch": _score_rows(
                    validation_rows, indices, true_routed),
                "classifier_routed": _score_rows(
                    validation_rows, indices, predicted_routed),
                "oracle_event_branch": _score_rows(
                    validation_rows, indices, oracle_event_routed_filled),
                "classifier_event_branch": _score_rows(
                    validation_rows, indices, classifier_event_routed_filled),
            }

    by_event_and_response: dict[str, Any] = {}
    for event in ("onset", "unwind", "reversal"):
        for step in (2,):
            group_phases = {name: row for name, row in validation_phases.items()
                            if row["event"] == event
                            and row["true_steps"] == step}
            indices = _phase_sample_indices(group_phases)
            by_event_and_response[f"{event}/{step}_packet"] = {
                "phase_count": len(group_phases),
                "rows": len(indices),
                "exact_packet_fullband_atlas": _score_rows(
                    validation_rows, indices, baseline_prediction),
                "oracle_step_and_event_branch": _score_rows(
                    validation_rows, indices, oracle_event_routed_filled),
                "classifier_step_and_event_branch": _score_rows(
                    validation_rows, indices, classifier_event_routed_filled),
                "matched_history_comparison": {
                    "100ms_causal_event_and_response_model": _score_rows(
                        validation_rows, indices, short_classifier_event_routed_filled),
                    "500ms_causal_event_and_response_model": _score_rows(
                        validation_rows, indices, classifier_event_routed_filled),
                    "100ms_oracle_event_and_response_model": _score_rows(
                        validation_rows, indices, short_oracle_event_routed_filled),
                    "500ms_oracle_event_and_response_model": _score_rows(
                        validation_rows, indices, oracle_event_routed_filled),
                },
            }

    residual_diagnosis = _diagnose_group_residuals(
        validation_rows, validation_phases, classifier_event_routed)

    # Test a causal, deliberately simple hypothesis suggested by the held-out
    # strata: when the command is far ahead of measured steering, the yaw rate
    # may persist for one tick instead of following a pooled transition mean.
    # Thresholds are fixed here (not selected against the validation labels).
    persistence_rules = {
        ("reversal", 2): 0.25,
        ("unwind", 2): 0.05,
    }
    persistence_prediction = classifier_event_routed_filled.copy()
    persistence_rows: dict[str, list[int]] = defaultdict(list)
    feedback_index = atlas.OBSERVATION_NAMES.index("steering_feedback_rad")
    command_index = atlas.OBSERVATION_NAMES.index("steering_command_rad")
    for phase in validation_phases.values():
        predicted_step = phase["predicted_steps"]
        key = (str(phase["event"]), predicted_step)
        threshold = persistence_rules.get(key)
        if threshold is None:
            continue
        indices = np.asarray(phase["indices"], dtype=np.int64)
        gap = np.abs(validation_rows["x"][indices, command_index]
                     - validation_rows["x"][indices, feedback_index])
        selected = indices[gap > threshold]
        persistence_prediction[selected] = 0.0
        persistence_rows[f"{key[0]}/{key[1]}_packet"].extend(
            selected.tolist())

    persistence_metrics = {
        "hypothesis": (
            "At large current steering command/feedback mismatch during the "
            "identified transient branch, predict one-tick yaw persistence."),
        "rules": [
            {"event": event, "predicted_packet_class": step,
             "absolute_command_feedback_gap_threshold_rad": threshold,
             "residual_prediction_when_selected": 0.0}
            for (event, step), threshold in persistence_rules.items()],
        "validation_label_or_future_measurement_used_for_routing": False,
        "selected_rows": {
            group: int(len(indices))
            for group, indices in persistence_rows.items()},
        "all_scored_transition_rows": _score_rows(
            validation_rows, np.flatnonzero(route_mask), persistence_prediction),
        "all_valid_validation_rows": _score_rows(
            validation_rows, np.arange(len(route_mask)), persistence_prediction),
        "by_event_and_response_group": {},
    }
    for event, step in (("reversal", 2), ("unwind", 2)):
        phase_subset = {
            name: phase for name, phase in validation_phases.items()
            if phase["event"] == event and phase["true_steps"] == step}
        indices = _phase_sample_indices(phase_subset)
        persistence_metrics["by_event_and_response_group"][
            f"{event}/{step}_packet"] = {
                "baseline_causal_model": _score_rows(
                    validation_rows, indices, classifier_event_routed_filled),
                "persistence_counterfactual": _score_rows(
                    validation_rows, indices, persistence_prediction),
            }

    worst_transition_rows = []
    transition_indices = np.flatnonzero(route_mask)
    routed_errors = (classifier_event_routed[transition_indices]
                     - validation_rows["residual"][transition_indices])
    worst_order = np.argsort(np.abs(routed_errors))[::-1]
    for local_index in worst_order[:50]:
        row_index = int(transition_indices[local_index])
        phase = next((phase for phase in validation_phases.values()
                      if row_index in phase["indices"]), None)
        x = validation_rows["x"][row_index]
        current_imu_yaw = float(validation_rows["imu_yaw"][row_index])
        prediction = float(classifier_event_routed[row_index])
        target_residual = float(validation_rows["residual"][row_index])
        step = (int(phase["predicted_steps"])
                if phase and phase["predicted_steps"] == 2
                else int(phase["true_steps"]) if phase else 2)
        event = str(validation_rows["event"][row_index])
        neighbor_indices = event_train_indices.get(
            (step, event), branch_train_indices[step])
        neighbor_x = train_rows["x_timed"][neighbor_indices].astype(np.float64)
        neighbor_y = train_rows["residual"][neighbor_indices].astype(np.float64)
        train_scale = np.maximum(np.std(neighbor_x, axis=0), 1.0e-4)
        normalized_delta = (neighbor_x - validation_rows["x_timed"][row_index]
                            .astype(np.float64)) / train_scale
        distances = np.sqrt(np.mean(normalized_delta * normalized_delta, axis=1))
        nearest = np.argsort(distances)[:min(10, len(distances))]
        nearest_targets = neighbor_y[nearest]
        worst_transition_rows.append({
            "sample_time_ns": int(validation_rows["sample_time_ns"][row_index]),
            "probe_event": phase["event"] if phase else "unknown",
            "row_event": str(validation_rows["event"][row_index]),
            "true_response_steps": int(phase["true_steps"]) if phase else None,
            "predicted_response_steps": phase["predicted_steps"] if phase else None,
            "selector_confidence": phase["selector_confidence"] if phase else None,
            "condition": phase["condition"] if phase else None,
            "rear_wheel_mean_mps": float(validation_rows["wheel_speed"][row_index]),
            "steering_feedback_rad": float(validation_rows["steering"][row_index]),
            "steering_command_rad": float(x[atlas.OBSERVATION_NAMES.index(
                "steering_command_rad")]),
            "steering_command_feedback_gap_rad": float(x[
                atlas.OBSERVATION_NAMES.index("steering_command_rad")]
                - x[atlas.OBSERVATION_NAMES.index("steering_feedback_rad")]),
            "current_exact_imu_yaw_radps": current_imu_yaw,
            "command_minus_source_ms": float(
                validation_rows["precise_timing"][row_index, 0]),
            "command_age_at_packet_receipt_ms": float(
                validation_rows["precise_timing"][row_index, 1]),
            "feedback_minus_source_ms": float(
                validation_rows["precise_timing"][row_index, 2]),
            "feedback_age_at_packet_receipt_ms": float(
                validation_rows["precise_timing"][row_index, 3]),
            "next_gt_yaw_radps": current_imu_yaw + target_residual,
            "predicted_next_yaw_radps": current_imu_yaw + prediction,
            "signed_error_radps": prediction - target_residual,
            "nearest_training_state_diagnostic": {
                "training_rows_in_selected_branch": int(len(neighbor_indices)),
                "nearest_10_normalized_feature_rms_distance": float(
                    np.mean(distances[nearest])),
                "nearest_10_training_target_residual_p10_p50_p90_radps": [
                    float(value) for value in np.quantile(
                        nearest_targets, [0.1, 0.5, 0.9])],
                "nearest_10_targets_within_0p1_of_validation_target": int(
                    np.count_nonzero(np.abs(nearest_targets - target_residual)
                                     <= 0.1)),
                "nearest_10_target_std_radps": float(np.std(nearest_targets)),
            },
        })

    result = {
        "title": "Do response-class-conditioned yaw regressors reduce held-out large yaw errors?",
        "status": "research evaluation only; no runtime model or odometry change",
        "target": "next simulator-truth yaw rate minus current exact-source-packet IMU yaw rate (rad/s)",
        "inputs": "causal sensor/actuator atlas features plus precise command/feedback receipt age relative to the odometry source packet; GT is target only; bridge packet class is training/evaluation label only; validation routing uses causal phase-start classifier and current causal event",
        "data": {
            "training_runs": train_ids,
            "held_out_validation_run": validation_id,
            "whole_run_holdout": True,
            "test_and_final_test_opened": False,
            "exact_source_packet_imu": True,
            "sensor_history_lags_ms": [
                int(lag * atlas.DT_S * 1000) for lag in EXPANDED_HISTORY_LAGS],
            "matched_history_comparison": {
                "short_model_history_lags_ms": [0, 25, 50, 100],
                "long_model_history_lags_ms": [
                    int(lag * atlas.DT_S * 1000)
                    for lag in EXPANDED_HISTORY_LAGS],
                "short_model_input_features": int(
                    train_rows["x_short_timed"].shape[1]),
                "long_model_input_features": int(train_rows["x_timed"].shape[1]),
                "same_training_and_validation_rows": True,
                "long_history_features_actually_used": True,
                "previous_v1_long_history_claim_valid": False,
                "previous_v1_issue": (
                    "The earlier candidate matrix accidentally truncated to the "
                    "100-ms atlas history while still filtering rows using the "
                    "500-ms history gate. Its long-history comparison is withdrawn."),
            },
            "additional_causal_timing_feature_names": list(TIMING_FEATURE_NAMES),
            "transition_window_ms": [-100, 500],
            "phase_audits": phase_audits,
            "training_branch_rows": {
                str(step): int(len(indices))
                for step, indices in branch_train_indices.items()},
            "selector_phase_accuracy": (float(np.mean(selector_correct))
                                        if selector_correct else None),
        },
        "validation": results,
        "by_true_packet_response_group": by_group,
        "by_transition_event": by_event,
        "by_event_and_response_group": by_event_and_response,
        "heldout_residual_cause_strata": residual_diagnosis,
        "pending_command_persistence_counterfactual": persistence_metrics,
        "worst_50_classifier_routed_transition_errors": worst_transition_rows,
        "event_specialist_training_support": event_support,
        "response_event_route_coverage": {
            "oracle": int(np.count_nonzero(event_route_mask)),
            "classifier": int(np.count_nonzero(classifier_event_route_mask)),
        },
        "phase_details": phase_rows,
        "source_audit": source_audit,
        "model": {
            "class": "ExtraTreesRegressor",
            "parameters": {"n_estimators": 240, "max_depth": 14,
                            "min_samples_leaf": 4, "max_features": 0.9},
            "research_only": True,
            "feature_revision": "Corrected experiment: candidate uses sensor history at 0/25/50/100/200/300/500 ms plus derived current-state features and causal command/feedback timing; matched 0/25/50/100-ms candidate is fit on exactly the same training rows. The frozen atlas baseline retains its original feature contract. No debug bridge timing is a regressor input.",
        },
        "interpretation": [
            "Oracle routing tests whether separating measured 2/3-packet response classes can help at all.",
            "Classifier routing tests the same experts without using bridge timing at prediction time.",
            "The result is not promotable unless classifier routing improves held-out errors without hiding large residuals; this experiment covers only the controlled high-steer transition probes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "output": str(output.relative_to(ROOT)),
        "validation": results,
        "by_true_packet_response_group": by_group,
        "by_transition_event": by_event,
        "by_event_and_response_group": by_event_and_response,
        "training_branch_rows": result["data"]["training_branch_rows"],
        "selector_phase_accuracy": result["data"]["selector_phase_accuracy"],
    }, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated multi-count evaluator: non-two-packet data are invalid; "
        "use the exact-two-packet targeted evaluator instead.")
