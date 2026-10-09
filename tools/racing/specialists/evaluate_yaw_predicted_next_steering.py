#!/usr/bin/env python3
"""Test whether predicting the next legal steering-feedback sample helps yaw.

The only yaw-model input addition is a steering-feedback prediction made from
current/past legal sensors, commands, and timestamp features. Actual next
steering is scored as an offline forbidden-input upper bound, never fed to the
causal candidate. Training steering predictions are leave-one-capture-out to
avoid target leakage. A separate whole-run holdout is scored; no runtime
component is modified.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np
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
          "yaw_large_error_audit_v1/predicted_next_steering_yaw_audit_r09.json")
VALIDATION_RUN_ID = "openplane_yaw_error_packet_phase_validation_r09_20261008"
GROUPS = ((2, "reversal"), (2, "unwind"))
MODEL = {
    "n_estimators": 240,
    "max_depth": 14,
    "min_samples_leaf": 4,
    "max_features": 0.9,
    "random_state": 20261008,
    "n_jobs": 4,
}


def _history_block_count(features: np.ndarray, signed_tail: bool = False
                         ) -> int:
    """Infer history depth from the fixed derived/timing feature suffix."""
    base = features[:, :-1] if signed_tail else features
    width = len(atlas.OBSERVATION_NAMES)
    history_width = base.shape[1] - len(atlas.DERIVED_NAMES) - 4
    if history_width <= 0 or history_width % width:
        raise ValueError(f"unexpected yaw feature width: {features.shape[1]}")
    return history_width // width


def _mirror_features(features: np.ndarray, signed_tail: bool = False
                     ) -> np.ndarray:
    """Reflect left/right steering, lateral acceleration, yaw and roll."""
    features = np.asarray(features)
    base = features[:, :-1] if signed_tail else features
    mirrored = base.copy()
    observation_width = len(atlas.OBSERVATION_NAMES)
    for block in range(_history_block_count(features, signed_tail)):
        start = block * observation_width
        mirrored[:, [start + index for index in (0, 5, 6, 7, 9, 10)]] *= -1.0
        left = base[:, start + 2].copy()
        mirrored[:, start + 2] = base[:, start + 3]
        mirrored[:, start + 3] = left
    derived = _history_block_count(features, signed_tail) * observation_width
    for offset in (1, 3, 5, 7):
        mirrored[:, derived + offset] *= -1.0
    if not signed_tail:
        return mirrored
    return np.column_stack((mirrored, -features[:, -1]))


def _next_steering_rows(series: Any) -> np.ndarray:
    run = atlas._read_sensor_run(
        series, history_lags=response.EXPANDED_HISTORY_LAGS,
        exact_packet_imu=True)
    values: list[float] = []
    for sensors, sensor_valid, attitude, attitude_valid, rigid in run["sequences"]:
        if (not np.isfinite(sensors).all() or not np.isfinite(attitude).all()
                or not np.isfinite(rigid).all()):
            continue
        for k in range(max(response.EXPANDED_HISTORY_LAGS), len(sensors) - 1):
            if (not all(sensor_valid[k - lag]
                        and attitude_valid[k - lag]
                        for lag in response.EXPANDED_HISTORY_LAGS)):
                continue
            wheel_mean = 0.5 * float(sensors[k, 2] + sensors[k, 3])
            speed_cell = int(wheel_mean // atlas.SPEED_BIN_MPS)
            steering = float(sensors[k, 0])
            if (speed_cell < 0 or speed_cell >= len(atlas.SPEED_CENTERS)
                    or not np.isfinite(steering)
                    or steering < atlas.STEERING_CENTERS[0] - 0.0125
                    or steering > atlas.STEERING_CENTERS[-1] + 0.0125):
                continue
            # Future steering feedback is the supervised actuator target only.
            values.append(float(sensors[k + 1, 0]))
    return np.asarray(values, dtype=np.float32)


def _group_data(rows: dict[str, Any], phases: dict[str, dict[str, Any]],
                future_steering: np.ndarray, step: int, event: str,
                history_lags: tuple[int, ...] = atlas.HISTORY_LAGS
                ) -> dict[str, np.ndarray]:
    features, targets, future, phase_ids = [], [], [], []
    phase_labels, age_ms, condition_labels = [], [], []
    for phase_index, (phase_label, phase) in enumerate(phases.items()):
        if phase["true_steps"] != step or phase["event"] != event:
            continue
        indices = np.asarray(phase["indices"], dtype=np.int64)
        indices = indices[rows["event"][indices].astype(str) == event]
        if not len(indices):
            continue
        feature_key = ("x_short_timed" if history_lags == atlas.HISTORY_LAGS
                       else "x_timed")
        features.append(rows[feature_key][indices])
        targets.append(rows["residual"][indices])
        future.append(future_steering[indices])
        phase_ids.extend([phase_index] * len(indices))
        phase_labels.extend([phase_label] * len(indices))
        age_ms.extend([
            (int(rows["sample_time_ns"][index])
             - int(phase["command_receipt_ns"])) / 1.0e6
            for index in indices])
        condition_labels.extend([
            json.dumps(phase["condition"], sort_keys=True)] * len(indices))
    if not features:
        raise ValueError(f"no rows for {event}/{step}-packet group")
    x = np.concatenate(features)
    return {
        "x": x,
        "y": np.concatenate(targets),
        "future_steering": np.concatenate(future),
        "actuator_rules": _actuator_rule_predictions(x),
        "phase_id": np.asarray(phase_ids, dtype=np.int64),
        "phase_label": np.asarray(phase_labels, dtype="U256"),
        "age_ms": np.asarray(age_ms, dtype=np.float64),
        "condition": np.asarray(condition_labels, dtype="U256"),
    }


def _actuator_rule_predictions(x: np.ndarray) -> dict[str, np.ndarray]:
    """Causal discrete-delay actuator hypotheses on the fixed 25-ms grid."""
    width = len(atlas.OBSERVATION_NAMES)
    feedback = x[:, 0]
    command_k = x[:, 7]
    command_k1 = x[:, width + 7]
    command_k2 = x[:, 2 * width + 7]
    limit = 3.2 * 0.025
    limited_k1 = feedback + np.clip(command_k1 - feedback, -limit, limit)
    decrease = np.abs(command_k1) < np.abs(feedback)
    pending_reversal = ((command_k1 * command_k2 < 0.0)
                        & (feedback * command_k2 > 0.0))
    pending_target = np.where(pending_reversal, command_k2, command_k1)
    pending_limited = feedback + np.clip(
        pending_target - feedback, -limit, limit)
    return {
        "hold_current_feedback": feedback.astype(np.float32),
        "direct_command_k": command_k.astype(np.float32),
        "direct_command_k_minus_1": command_k1.astype(np.float32),
        "direct_command_k_minus_2": command_k2.astype(np.float32),
        "rate_limit_3p2_k_minus_1": limited_k1.astype(np.float32),
        "hybrid_release_or_limit_k_minus_1": np.where(
            decrease, command_k1, limited_k1).astype(np.float32),
        "hybrid_one_extra_queue_on_pending_reversal": np.where(
            np.abs(pending_target) < np.abs(feedback), pending_target,
            pending_limited).astype(np.float32),
    }


def _phase_weights(phase_ids: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(phase_ids, return_inverse=True,
                                    return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights * len(weights) / weights.sum()


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


def _steering_metric(error: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_rad": float(np.sqrt(np.mean(error * error))),
        "p95_abs_error_rad": float(np.quantile(absolute, 0.95)),
        "max_abs_error_rad": float(np.max(absolute)),
        "samples_over_0p05_rad": int(np.count_nonzero(absolute > 0.05)),
    }


def _training_only_loco(
        train_by_run: dict[str, dict[str, np.ndarray]],
        event: str) -> dict[str, Any]:
    """Choose a causal steering rule and yaw-feature variant on whole runs."""
    rule_names = list(next(iter(train_by_run.values()))["actuator_rules"])
    rule_folds: dict[str, Any] = {}
    rule_metrics: dict[str, list[dict[str, Any]]] = {
        name: [] for name in rule_names}
    for run_id, data in train_by_run.items():
        rule_folds[run_id] = {}
        for name in rule_names:
            score = _steering_metric(
                data["actuator_rules"][name] - data["future_steering"])
            rule_folds[run_id][name] = score
            rule_metrics[name].append(score)
    rule_summary = {
        name: {
            "run_macro_rmse_rad": float(np.mean(
                [score["rmse_rad"] for score in scores])),
            "total_samples_over_0p05_rad": int(sum(
                score["samples_over_0p05_rad"] for score in scores)),
            "worst_capture_max_abs_error_rad": float(max(
                score["max_abs_error_rad"] for score in scores)),
        }
        for name, scores in rule_metrics.items()
    }
    selected_rule = min(rule_summary, key=lambda name: (
        rule_summary[name]["total_samples_over_0p05_rad"],
        rule_summary[name]["run_macro_rmse_rad"],
        rule_summary[name]["worst_capture_max_abs_error_rad"],
    ))

    candidate_names = ("baseline_plain", "baseline_mirrored",
                       "plus_predicted_next_steering_mirrored",
                       "plus_predicted_steering_change_mirrored")
    yaw_folds: dict[str, Any] = {}
    yaw_scores: dict[str, list[dict[str, Any]]] = {
        name: [] for name in candidate_names}
    for held_out, validation in train_by_run.items():
        fit_ids = [run_id for run_id in train_by_run if run_id != held_out]
        fit_x = np.concatenate([train_by_run[run_id]["x"]
                                for run_id in fit_ids])
        fit_y = np.concatenate([train_by_run[run_id]["y"]
                                for run_id in fit_ids])
        fit_rule = np.concatenate([
            train_by_run[run_id]["actuator_rules"][selected_rule]
            for run_id in fit_ids])
        fit_phase_ids = np.concatenate([
            np.asarray([f"{run_id}:{phase}" for phase in
                        train_by_run[run_id]["phase_id"]], dtype="U160")
            for run_id in fit_ids])
        validation_rule = validation["actuator_rules"][selected_rule]
        prediction = {
            "baseline_plain": _fit(
                fit_x, fit_y, fit_phase_ids).predict(validation["x"]),
            "baseline_mirrored": _fit(
                fit_x, fit_y, fit_phase_ids, mirror=True).predict(
                    validation["x"]),
        }
        absolute_model = _fit(
            np.column_stack((fit_x, fit_rule)), fit_y, fit_phase_ids,
            mirror=True, signed_tail=True)
        prediction["plus_predicted_next_steering_mirrored"] = (
            absolute_model.predict(np.column_stack((
                validation["x"], validation_rule))))
        change_model = _fit(
            np.column_stack((fit_x, fit_rule - fit_x[:, 0])), fit_y,
            fit_phase_ids, mirror=True, signed_tail=True)
        prediction["plus_predicted_steering_change_mirrored"] = (
            change_model.predict(np.column_stack((
                validation["x"], validation_rule - validation["x"][:, 0]))))
        yaw_folds[held_out] = {}
        for name, values in prediction.items():
            score = _metric(values - validation["y"])
            yaw_folds[held_out][name] = score
            yaw_scores[name].append(score)
    yaw_summary = {
        name: {
            "run_macro_rmse_radps": float(np.mean(
                [score["rmse_radps"] for score in scores])),
            "total_samples_over_0p1": int(sum(
                score["samples_over_0p1"] for score in scores)),
            "worst_capture_max_abs_radps": float(max(
                score["max_abs_radps"] for score in scores)),
        }
        for name, scores in yaw_scores.items()
    }
    selected_yaw = min(yaw_summary, key=lambda name: (
        yaw_summary[name]["total_samples_over_0p1"],
        yaw_summary[name]["run_macro_rmse_radps"],
        yaw_summary[name]["worst_capture_max_abs_radps"],
    ))
    return {
        "event": event,
        "selected_rule_by_training_only_loco": selected_rule,
        "steering_rule_selection": {
            "rule_summary": rule_summary,
            "heldout_capture_metrics": rule_folds,
        },
        "selected_yaw_candidate_by_training_only_loco": selected_yaw,
        "yaw_selection": {
            "candidate_summary": yaw_summary,
            "heldout_capture_metrics": yaw_folds,
        },
    }


def _fit(x: np.ndarray, y: np.ndarray, phase_ids: np.ndarray,
         mirror: bool = False, signed_tail: bool = False
         ) -> ExtraTreesRegressor:
    if mirror:
        x = np.concatenate((x, _mirror_features(x, signed_tail)), axis=0)
        y = np.concatenate((y, -y), axis=0)
        phase_ids = np.concatenate((phase_ids, phase_ids), axis=0)
    return ExtraTreesRegressor(**MODEL).fit(
        x, y, sample_weight=_phase_weights(phase_ids))


def run(output: Path = OUTPUT,
        validation_run_id: str = VALIDATION_RUN_ID,
        history_lags: tuple[int, ...] = atlas.HISTORY_LAGS
        ) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audits = response._collect_exact_two_packet(admitted)
    rows_by_run = collected["rows"]
    phases_by_run = collected["phases"]
    future_by_run: dict[str, np.ndarray] = {}
    for series in admitted:
        if series.run_id not in response.TIMING_SUFFIX:
            continue
        future = _next_steering_rows(series)
        if len(future) != len(rows_by_run[series.run_id]["residual"]):
            raise ValueError(f"{series.run_id}: future-steering rows do not align")
        future_by_run[series.run_id] = future

    train_ids = sorted(run_id for run_id in response.TIMING_SUFFIX
                       if "_train_" in run_id)
    validation_id = validation_run_id
    if validation_id not in response.TIMING_SUFFIX:
        raise ValueError(f"unregistered validation run: {validation_id}")
    reports: dict[str, Any] = {}
    for step, event in GROUPS:
        name = f"{event}/{step}_packet"
        train_by_run = {
            run_id: _group_data(rows_by_run[run_id], phases_by_run[run_id],
                                future_by_run[run_id], step, event,
                                history_lags)
            for run_id in train_ids
        }
        loco = _training_only_loco(train_by_run, event)
        validation = _group_data(
            rows_by_run[validation_id], phases_by_run[validation_id],
            future_by_run[validation_id], step, event, history_lags)
        train_x = np.concatenate([train_by_run[rid]["x"]
                                  for rid in train_ids])
        train_y = np.concatenate([train_by_run[rid]["y"]
                                  for rid in train_ids])
        train_future = np.concatenate([train_by_run[rid]["future_steering"]
                                       for rid in train_ids])
        train_phase_ids = np.concatenate([
            np.asarray([f"{rid}:{phase}" for phase in
                        train_by_run[rid]["phase_id"]], dtype="U160")
            for rid in train_ids])

        # OOF actuator estimates for yaw-model training: each capture's
        # next-steering predictions come from the other complete capture.
        oof_parts = []
        oof_mirrored_parts = []
        for held_out in train_ids:
            fit_ids = [rid for rid in train_ids if rid != held_out]
            fit_x = np.concatenate([train_by_run[rid]["x"] for rid in fit_ids])
            fit_y = np.concatenate([train_by_run[rid]["future_steering"]
                                    for rid in fit_ids])
            fit_phase_ids = np.concatenate([
                np.asarray([f"{rid}:{phase}" for phase in
                            train_by_run[rid]["phase_id"]], dtype="U160")
                for rid in fit_ids])
            actuator = _fit(fit_x, fit_y, fit_phase_ids)
            actuator_mirrored = _fit(
                fit_x, fit_y, fit_phase_ids, mirror=True)
            held_rows = train_by_run[held_out]
            oof_parts.append(actuator.predict(held_rows["x"]).astype(np.float32))
            oof_mirrored_parts.append(
                actuator_mirrored.predict(held_rows["x"]).astype(np.float32))
        train_oof_steering = np.concatenate(oof_parts)
        train_oof_mirrored_steering = np.concatenate(oof_mirrored_parts)

        # Final causal actuator predictor uses both train captures.
        actuator_model = _fit(train_x, train_future, train_phase_ids)
        mirrored_actuator_model = _fit(
            train_x, train_future, train_phase_ids, mirror=True)
        predicted_validation_steering = actuator_model.predict(
            validation["x"]).astype(np.float32)
        mirrored_predicted_validation_steering = (
            mirrored_actuator_model.predict(validation["x"])
            .astype(np.float32))
        steering_error = (predicted_validation_steering
                          - validation["future_steering"])
        mirrored_steering_error = (mirrored_predicted_validation_steering
                                   - validation["future_steering"])

        baseline = _fit(train_x, train_y, train_phase_ids)
        mirrored_baseline = _fit(train_x, train_y, train_phase_ids, mirror=True)
        yaw_with_predicted_actuator = _fit(
            np.column_stack((train_x, train_oof_steering)), train_y,
            train_phase_ids)
        yaw_with_mirrored_actuator = _fit(
            np.column_stack((train_x, train_oof_mirrored_steering)), train_y,
            train_phase_ids, mirror=True, signed_tail=True)
        yaw_with_oracle_actuator = _fit(
            np.column_stack((train_x, train_future)), train_y,
            train_phase_ids)
        yaw_with_oracle_actuator_mirrored = _fit(
            np.column_stack((train_x, train_future)), train_y,
            train_phase_ids, mirror=True, signed_tail=True)
        simple_yaw_models = {}
        simple_yaw_predictions = {}
        delta_yaw_prediction = None
        for rule_name in train_by_run[train_ids[0]]["actuator_rules"]:
            train_rule = np.concatenate([
                train_by_run[run_id]["actuator_rules"][rule_name]
                for run_id in train_ids])
            model = _fit(np.column_stack((train_x, train_rule)), train_y,
                         train_phase_ids, mirror=True, signed_tail=True)
            validation_rule = validation["actuator_rules"][rule_name]
            simple_yaw_predictions[rule_name] = model.predict(
                np.column_stack((validation["x"], validation_rule)))
            if rule_name == "hybrid_release_or_limit_k_minus_1":
                train_delta = (train_rule - train_x[:, 0]).astype(np.float32)
                delta_model = _fit(
                    np.column_stack((train_x, train_delta)), train_y,
                    train_phase_ids, mirror=True, signed_tail=True)
                validation_delta = (
                    validation_rule - validation["x"][:, 0]).astype(np.float32)
                delta_yaw_prediction = delta_model.predict(np.column_stack((
                    validation["x"], validation_delta)))
        baseline_prediction = baseline.predict(validation["x"])
        mirrored_baseline_prediction = mirrored_baseline.predict(
            validation["x"])
        predicted_actuator_prediction = yaw_with_predicted_actuator.predict(
            np.column_stack((validation["x"], predicted_validation_steering)))
        mirrored_actuator_prediction = yaw_with_mirrored_actuator.predict(
            np.column_stack((validation["x"],
                             mirrored_predicted_validation_steering)))
        oracle_prediction = yaw_with_oracle_actuator.predict(
            np.column_stack((validation["x"], validation["future_steering"])))
        mirrored_oracle_prediction = yaw_with_oracle_actuator_mirrored.predict(
            np.column_stack((validation["x"], validation["future_steering"])))
        result = {
            "training_rows": int(len(train_y)),
            "training_phases": int(len(np.unique(train_phase_ids))),
            "validation_rows": int(len(validation["y"])),
            "validation_phase_count": int(sum(
                phase["true_steps"] == step and phase["event"] == event
                for phase in phases_by_run[validation_id].values())),
            "next_steering_feedback_prediction": {
                "rmse_rad": float(np.sqrt(np.mean(steering_error ** 2))),
                "p95_abs_error_rad": float(np.quantile(np.abs(steering_error), 0.95)),
                "max_abs_error_rad": float(np.max(np.abs(steering_error))),
                "fraction_within_0p05_rad": float(np.mean(
                    np.abs(steering_error) <= 0.05)),
            },
            "next_steering_feedback_prediction_with_symmetry": {
                "rmse_rad": float(np.sqrt(np.mean(mirrored_steering_error ** 2))),
                "p95_abs_error_rad": float(np.quantile(
                    np.abs(mirrored_steering_error), 0.95)),
                "max_abs_error_rad": float(np.max(np.abs(mirrored_steering_error))),
                "fraction_within_0p05_rad": float(np.mean(
                    np.abs(mirrored_steering_error) <= 0.05)),
            },
            "yaw_residual_prediction": {
                "legal_current_history_only": _metric(
                    baseline_prediction - validation["y"]),
                "legal_current_history_with_symmetry": _metric(
                    mirrored_baseline_prediction - validation["y"]),
                "plus_causal_predicted_next_steering": _metric(
                    predicted_actuator_prediction - validation["y"]),
                "plus_causal_predicted_next_steering_with_symmetry": _metric(
                    mirrored_actuator_prediction - validation["y"]),
                "plus_forbidden_actual_next_steering_upper_bound": _metric(
                    oracle_prediction - validation["y"]),
                "plus_forbidden_actual_next_steering_upper_bound_with_symmetry": _metric(
                    mirrored_oracle_prediction - validation["y"]),
                **{
                    f"plus_deterministic_{name}_with_symmetry": _metric(
                        prediction - validation["y"])
                    for name, prediction in simple_yaw_predictions.items()
                },
                "plus_deterministic_hybrid_release_or_limit_k_minus_1_delta_with_symmetry": _metric(
                    delta_yaw_prediction - validation["y"]),
            },
            "deterministic_next_steering_rule_prediction": {
                name: _steering_metric(prediction - validation["future_steering"])
                for name, prediction in validation["actuator_rules"].items()
            },
            "actual_next_steering_is_not_a_candidate_input": True,
            "phase_selection_is_exact_packet_and_event_metadata_only": True,
            "training_only_loco_selection": loco,
        }
        selected_rule = loco["selected_rule_by_training_only_loco"]
        selected_candidate = loco["selected_yaw_candidate_by_training_only_loco"]
        if selected_candidate == "plus_predicted_next_steering_mirrored":
            selected_prediction = simple_yaw_predictions[selected_rule]
        elif selected_candidate == "plus_predicted_steering_change_mirrored":
            selected_prediction = delta_yaw_prediction
        elif selected_candidate == "baseline_mirrored":
            selected_prediction = mirrored_baseline_prediction
        else:
            selected_prediction = baseline_prediction
        residual_error = selected_prediction - validation["y"]
        temporal_bins = (("before_0ms", -np.inf, 0.0),
                         ("0_to_25ms", 0.0, 25.0),
                         ("25_to_50ms", 25.0, 50.0),
                         ("50_to_150ms", 50.0, 150.0),
                         ("150ms_plus", 150.0, np.inf))
        result["training_only_selected_yaw_candidate"] = {
            "name": selected_candidate,
            "actuator_rule": selected_rule,
            "metrics": _metric(residual_error),
            "by_transition_age": {
                label: _metric(residual_error[
                    (validation["age_ms"] >= lower)
                    & (validation["age_ms"] < upper)])
                for label, lower, upper in temporal_bins
                if np.any((validation["age_ms"] >= lower)
                          & (validation["age_ms"] < upper))
            },
            "rows_over_0p1_radps": [],
        }
        width = len(atlas.OBSERVATION_NAMES)
        derived_start = _history_block_count(validation["x"]) * width
        for local in np.flatnonzero(np.abs(residual_error) > 0.1):
            x = validation["x"][local]
            condition = json.loads(validation["condition"][local])
            result["training_only_selected_yaw_candidate"][
                "rows_over_0p1_radps"].append({
                    "phase": str(validation["phase_label"][local]),
                    "condition": condition,
                    "age_from_command_receipt_ms": float(
                        validation["age_ms"][local]),
                    "current_speed_wheel_mean_mps": float(x[
                        derived_start]),
                    "steering_feedback_rad": float(x[0]),
                    "steering_command_k_rad": float(x[7]),
                    "steering_command_k_minus_1_rad": float(x[width + 7]),
                    "current_imu_yaw_rate_radps": float(x[6]),
                    "actual_next_steering_feedback_rad_diagnostic_only": float(
                        validation["future_steering"][local]),
                    "truth_next_yaw_rate_radps": float(
                        validation["y"][local] + x[6]),
                    "predicted_next_yaw_rate_radps": float(
                        selected_prediction[local] + x[6]),
                    "signed_error_radps": float(residual_error[local]),
                })
        reports[name] = result

    result = {
        "title": "Does a causal steering-actuator predictor improve one-step yaw?",
        "status": "offline research only; no production component changed",
        "prediction_target": "next simulator-truth yaw-rate residual at fixed 25 ms",
        "data": {
            "train_runs": train_ids,
            "whole_run_validation": validation_id,
            "validation_previously_used_during_model_development": (
                validation_id != "openplane_yaw_error_packet_phase_validation_r09_20261008"),
            "test_and_final_test_opened": False,
            "history_lags_ms": [int(lag * 25) for lag in history_lags],
            "history_lags_steps": list(history_lags),
            "history_depth_ms": int(max(history_lags) * 25),
            "legal_features_include": "current/past permitted sensors, commands, and command/feedback receipt-to-source timing",
            "future_steering_training_label": "next recorded steering feedback; held out of causal yaw input except where predicted by a separate cross-fitted model",
            "yaw_training_uses_leave_one_capture_out_steering_predictions": True,
        },
        "phase_audits": phase_audits,
        "source_audit": source_audit,
        "groups": reports,
        "interpretation": "A held-out gain from the predicted actuator feature would justify continuing toward a two-stage sensor-only observer; only the forbidden actual-next-steering comparison is an upper bound, not a usable model.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output),
                      "groups": reports}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-run-id", default=VALIDATION_RUN_ID)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--history", choices=("short", "extended"),
                        default="short")
    args = parser.parse_args()
    history_lags = (atlas.HISTORY_LAGS if args.history == "short"
                    else response.EXPANDED_HISTORY_LAGS)
    run(args.output, args.validation_run_id, history_lags)
