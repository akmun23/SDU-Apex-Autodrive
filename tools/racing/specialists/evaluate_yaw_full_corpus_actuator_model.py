#!/usr/bin/env python3
"""Test whether all admitted training captures explain yaw outliers via steering.

Fits a causal one-step steering-feedback predictor from legal sensor/command
history across the complete train split. Whole captures are excluded when
creating actuator predictions for the yaw-model training rows, and the
high-steering r03 capture remains held out. This is offline research only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import evaluate_yaw_packet_response_conditioned_models as response
    import evaluate_yaw_predicted_next_steering as targeted
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as response
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as targeted
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/full_corpus_actuator_yaw_audit.json")
MODEL = {
    "learning_rate": 0.08,
    "max_iter": 160,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 2.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 12,
    "random_state": 20261008,
}
# Non-two-packet phases are invalid, never a model target or validation class.
GROUPS = ((2, "reversal"), (2, "unwind"))


def _actuator_rows(series: Any) -> dict[str, np.ndarray]:
    """Build legal/timed features and aligned, observed next-feedback labels."""
    rows = response._run_rows(series, history_lags=atlas.HISTORY_LAGS)
    return {"x": rows["x_short_timed"], "event": rows["event"],
            "next_steering": rows["next_steering"],
            "next_valid": rows["next_steering_valid"]}


def _corpus_weights(run_ids: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Equalize capture and command-intent event contributions."""
    run_ids = np.asarray(run_ids, dtype=str)
    events = np.asarray(events, dtype=str)
    keys = np.char.add(np.char.add(run_ids, "::"), events)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    per_group = 1.0 / counts[inverse]
    run_count = len(np.unique(run_ids))
    weights = per_group * run_count / np.sum(per_group)
    return weights.astype(np.float64)


def _fit_actuator(parts: list[dict[str, np.ndarray]], exclude_run: str | None = None
                  ) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    selected = [part for part in parts if part["run_id"] != exclude_run]
    x = np.concatenate([part["x"][part["next_valid"]] for part in selected])
    y = np.concatenate([
        part["next_steering"][part["next_valid"]] for part in selected])
    run_ids = np.concatenate([
        np.full(np.count_nonzero(part["next_valid"]), part["run_id"], dtype="U128")
        for part in selected])
    events = np.concatenate([
        part["event"][part["next_valid"]] for part in selected])
    model = HistGradientBoostingRegressor(**MODEL)
    model.fit(x, y, sample_weight=_corpus_weights(run_ids, events))
    return model, {"runs": len(selected), "rows": int(len(y)),
                   "rows_by_event": {
                       event: int(np.count_nonzero(events == event))
                       for event in atlas.EVENTS}}


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
        "over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _phase_ids_for_rows(phases: dict[str, dict[str, Any]], event: str,
                        step: int, row_count: int) -> np.ndarray:
    ids = np.full(row_count, -1, dtype=np.int32)
    phase_id = 0
    for phase in phases.values():
        if phase["true_steps"] != step or phase["event"] != event:
            continue
        indices = np.asarray(phase["indices"], dtype=np.int64)
        ids[indices] = phase_id
        phase_id += 1
    return ids


def _yaw_group(rows_by_run: dict[str, Any], phases_by_run: dict[str, Any],
               future_by_run: dict[str, np.ndarray], run_id: str,
               step: int, event: str) -> dict[str, np.ndarray]:
    rows = rows_by_run[run_id]
    features, targets, future, phase_ids, indices_out = [], [], [], [], []
    phase_id = 0
    for phase in phases_by_run[run_id].values():
        if phase["true_steps"] != step or phase["event"] != event:
            continue
        indices = np.asarray(phase["indices"], dtype=np.int64)
        indices = indices[rows["event"][indices].astype(str) == event]
        indices = indices[np.isfinite(future_by_run[run_id][indices])]
        if len(indices):
            features.append(rows["x_short_timed"][indices])
            targets.append(rows["residual"][indices])
            future.append(future_by_run[run_id][indices])
            phase_ids.extend([phase_id] * len(indices))
            indices_out.extend(indices.tolist())
        phase_id += 1
    if not features:
        raise ValueError(f"no valid-next-steering rows for {event}/{step}-packet")
    return {
        "x": np.concatenate(features),
        "y": np.concatenate(targets),
        "future_steering": np.concatenate(future),
        "phase_id": np.asarray(phase_ids, dtype=np.int64),
        "indices": np.asarray(indices_out, dtype=np.int64),
    }


def run(output: Path = OUTPUT) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    train_series = [series for series in admitted if series.split == "train"]
    train_ids = {series.run_id for series in train_series}
    special_train_ids = sorted(
        run_id for run_id in response.TIMING_SUFFIX
        if "_train_" in run_id)
    validation_id = next(run_id for run_id in response.TIMING_SUFFIX
                         if "_validation_" in run_id)
    required = set(special_train_ids) | {validation_id}
    if not required.issubset({series.run_id for series in admitted}):
        raise ValueError("required high-steering captures are not admitted")

    # Existing high-steering probe rows carry exact command/feedback timing;
    # their causal sensor features match the common 100-ms atlas contract.
    selected, phase_audits = response._collect_exact_two_packet(admitted)
    rows_by_run = selected["rows"]
    phases_by_run = selected["phases"]
    future_by_run: dict[str, np.ndarray] = {}
    for run_id in sorted(required):
        rows = rows_by_run[run_id]
        values = rows["next_steering"]
        if len(values) != len(rows["residual"]):
            raise ValueError(f"{run_id}: probe future-steering rows do not align")
        future_by_run[run_id] = np.where(
            rows["next_steering_valid"], values, np.nan)

    print(f"loading {len(train_series)} complete training captures", flush=True)
    corpus_parts = []
    for index, series in enumerate(train_series, start=1):
        part = _actuator_rows(series)
        part["run_id"] = series.run_id
        corpus_parts.append(part)
        print(f"  {index:02d}/{len(train_series)} {series.run_id}: "
              f"{len(part['x'])} rows, "
              f"{np.count_nonzero(part['next_valid'])} valid next labels",
              flush=True)

    # Capture-held-out actuator estimates prevent the yaw regressor from
    # learning from the next-steering labels of the rows it will train on.
    oof_steering: dict[str, np.ndarray] = {}
    actuator_fit_audits = {}
    for run_id in special_train_ids:
        model, fit_audit = _fit_actuator(corpus_parts, exclude_run=run_id)
        probe_x = rows_by_run[run_id]["x_short_timed"]
        oof_steering[run_id] = model.predict(probe_x).astype(np.float32)
        actuator_fit_audits[f"exclude_{run_id}"] = fit_audit
        print(f"OOF steering predictions for {run_id}: "
              f"{len(probe_x)} rows", flush=True)

    final_actuator, final_fit_audit = _fit_actuator(corpus_parts)
    validation_steering = final_actuator.predict(
        rows_by_run[validation_id]["x_short_timed"]).astype(np.float32)
    train_group_predictions = oof_steering
    reports: dict[str, Any] = {}
    for step, event in GROUPS:
        train_groups = [
            _yaw_group(rows_by_run, phases_by_run, future_by_run,
                       run_id, step, event)
            for run_id in special_train_ids]
        validation = _yaw_group(rows_by_run, phases_by_run, future_by_run,
                                validation_id, step, event)
        train_x = np.concatenate([group["x"] for group in train_groups])
        train_y = np.concatenate([group["y"] for group in train_groups])
        train_future = np.concatenate([
            group["future_steering"] for group in train_groups])
        train_predicted = np.concatenate([
            train_group_predictions[run_id][group["indices"]]
            for run_id, group in zip(special_train_ids, train_groups)])
        train_phase = np.concatenate([
            np.asarray([f"{run_id}:{phase}" for phase in group["phase_id"]],
                      dtype="U160")
            for run_id, group in zip(special_train_ids, train_groups)])
        phase_weights = targeted._phase_weights(train_phase)

        yaw_baseline = targeted._fit(train_x, train_y, train_phase)
        yaw_predicted_actuator = targeted._fit(
            np.column_stack((train_x, train_predicted)), train_y, train_phase)
        yaw_oracle_actuator = targeted._fit(
            np.column_stack((train_x, train_future)), train_y, train_phase)
        validation_group_indices = _group_indices(
            rows_by_run[validation_id], phases_by_run[validation_id],
            step, event)
        validation_group_indices = validation_group_indices[
            np.isfinite(future_by_run[validation_id][validation_group_indices])]
        predicted_group_steering = validation_steering[validation_group_indices]
        actual_group_steering = future_by_run[validation_id][validation_group_indices]
        baseline_prediction = yaw_baseline.predict(validation["x"])
        predicted_prediction = yaw_predicted_actuator.predict(
            np.column_stack((validation["x"], predicted_group_steering)))
        oracle_prediction = yaw_oracle_actuator.predict(
            np.column_stack((validation["x"], actual_group_steering)))
        steering_error = predicted_group_steering - actual_group_steering
        reports[f"{event}/{step}_packet"] = {
            "train_rows": int(len(train_y)),
            "validation_rows": int(len(validation["y"])),
            "validation_phases": int(len(np.unique(validation["phase_id"]))),
            "next_steering_feedback": {
                "full_corpus_predictor": _metric(steering_error),
                "current_feedback_persistence": _metric(
                    rows_by_run[validation_id]["x"][validation_group_indices, 0]
                    - actual_group_steering),
                "next_command_upper_baseline": _metric(
                    rows_by_run[validation_id]["x"][validation_group_indices, 7]
                    - actual_group_steering),
            },
            "yaw_residual": {
                "legal_current_history": _metric(
                    baseline_prediction - validation["y"]),
                "plus_full_corpus_causal_next_steering": _metric(
                    predicted_prediction - validation["y"]),
                "plus_actual_next_steering_forbidden_upper_bound": _metric(
                    oracle_prediction - validation["y"]),
            },
            "phase_balanced_yaw_fit": True,
        }

    report = {
        "title": "Full-training-corpus causal actuator model for worst yaw groups",
        "status": "offline research only; no production component changed",
        "target": "next steering-feedback sample at the fixed 25-ms grid",
        "features": "100-ms legal sensor/command/IMU history, derived rates/ages, and current packet receipt/source timing; no GT features",
        "training": {
            "run_count": len(train_series),
            "run_ids": sorted(train_ids),
            "rows": int(sum(len(part["x"]) for part in corpus_parts)),
            "valid_next_feedback_labels": int(sum(
                np.count_nonzero(part["next_valid"]) for part in corpus_parts)),
            "model": MODEL,
            "weighting": "equal total mass per capture/event; events are command_intent hold/turn_in/unwind/reversal",
            "crossfit_models": actuator_fit_audits,
            "final_fit": final_fit_audit,
        },
        "validation": {
            "whole_capture": validation_id,
            "probe_phase_audit": phase_audits.get(validation_id),
            "test_and_final_test_opened": False,
            "groups": reports,
        },
        "source_audit": source_audit,
        "limitations": [
            "The target is one-step recorded steering feedback, not physical actuator state between 25-ms samples.",
            "Yaw candidate is evaluated on the held-out high-steering capture and is not integrated into Odom or MPC.",
            "The actual next-steering feature is shown only as a forbidden-input upper bound.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 default=atlas._json_value) + "\n",
                      encoding="utf-8")
    print(json.dumps({"output": str(output.relative_to(ROOT)),
                      "groups": reports}, indent=2), flush=True)
    return report


def _group_indices(rows: dict[str, Any], phases: dict[str, dict[str, Any]],
                   step: int, event: str) -> np.ndarray:
    indices = []
    for phase in phases.values():
        if phase["true_steps"] != step or phase["event"] != event:
            continue
        phase_rows = np.asarray(phase["indices"], dtype=np.int64)
        phase_rows = phase_rows[rows["event"][phase_rows].astype(str) == event]
        indices.extend(phase_rows.tolist())
    return np.asarray(indices, dtype=np.int64)


if __name__ == "__main__":
    run()
