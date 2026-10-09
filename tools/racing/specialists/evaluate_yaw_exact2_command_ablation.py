#!/usr/bin/env python3
"""Run-capture evaluation of yaw feature contracts on exact-two-packet rows.

Compares the current long sensor/command history against shorter history and
ablations of newest/all command magnitudes. Selection uses training-capture
leave-one-out folds only. All non-two-packet response phases are excluded.
This is offline research; it does not change a runtime estimator.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    import evaluate_yaw_packet_response_conditioned_models as response
    import evaluate_yaw_targeted_group_candidates as targeted
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as response
    from tools.racing.specialists import evaluate_yaw_targeted_group_candidates as targeted
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "yaw_large_error_audit_v1/exact2_command_feature_ablation.json")
CSV_OUTPUT = OUTPUT.with_name("exact2_command_feature_ablation_rows.csv")
HOLDOUTS = (
    "openplane_yaw_error_highsteer_reversal_validation_r03_20261008",
    "openplane_yaw_error_packet_phase_validation_r09_20261008",
)
EVENTS = ("reversal", "unwind")
VIEWS = ("full_long", "short_100ms", "drop_current_command_values",
         "drop_all_command_values")
MLP_CANDIDATES = {
    "mlp_full_long_mirrored": ("full_long", True),
    "mlp_short_100ms_mirrored": ("short_100ms", True),
}


def _metrics(error: np.ndarray) -> dict[str, Any]:
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


def _feature_indices(view: str) -> np.ndarray | None:
    if view == "full_long":
        return None
    width = len(atlas.OBSERVATION_NAMES)
    lags = len(response.EXPANDED_HISTORY_LAGS)
    derived_start = width * lags
    if view == "short_100ms":
        return None
    if view == "drop_current_command_values":
        remove = {7, 8, derived_start + 5, derived_start + 6}
    elif view == "drop_all_command_values":
        remove = {block * width + offset for block in range(lags)
                  for offset in (7, 8)}
        remove.update((derived_start + 5, derived_start + 6))
    else:
        raise ValueError(f"unknown feature view: {view}")
    return np.asarray([index for index in range(
        width * lags + len(atlas.DERIVED_NAMES) + len(
            response.TIMING_FEATURE_NAMES)) if index not in remove],
        dtype=np.int64)


def _matrix(data: dict[str, np.ndarray], view: str,
            mirrored: bool = False) -> np.ndarray:
    if view == "short_100ms":
        key = "x_short_timed"
        lags = atlas.HISTORY_LAGS
    else:
        key = "x_timed"
        lags = response.EXPANDED_HISTORY_LAGS
    x = data[key]
    if mirrored:
        x = targeted._mirror_features(x, lags)
    indices = _feature_indices(view)
    return x if indices is None else x[:, indices]


def _fit_predict(fit: dict[str, np.ndarray], test: dict[str, np.ndarray],
                 view: str, mirror: bool,
                 family: str = "extra_trees") -> np.ndarray:
    x = _matrix(fit, view)
    y = fit["y"]
    phase_ids = fit["phase_ids"]
    if mirror:
        x = np.concatenate((x, _matrix(fit, view, mirrored=True)))
        y = np.concatenate((y, -y))
        phase_ids = np.concatenate((phase_ids, phase_ids))
    weights = targeted._weights(phase_ids)
    test_x = _matrix(test, view)
    if family == "extra_trees":
        model = targeted._model()
        model.fit(x, y, sample_weight=weights)
        return model.predict(test_x).astype(np.float32)
    if family == "mlp":
        scaler = StandardScaler().fit(x, sample_weight=weights)
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
            alpha=0.01, batch_size=64, learning_rate_init=0.001,
            max_iter=1000, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=40, tol=1.0e-5,
            random_state=20261008,
        ).fit(scaler.transform(x), y, sample_weight=weights)
        return model.predict(scaler.transform(test_x)).astype(np.float32)
    raise ValueError(f"unsupported model family: {family}")


def _event_data(rows: dict[str, Any], phases: dict[str, Any], run_id: str,
                event: str) -> dict[str, np.ndarray]:
    indices, phase_labels, _ = targeted._phase_rows(
        rows, phases, 2, event, require_row_event=True)
    return {
        "x_timed": rows["x_timed"][indices],
        "x_short_timed": rows["x_short_timed"][indices],
        "y": rows["residual"][indices],
        "indices": indices,
        "age_ms": np.asarray([
            (int(rows["sample_time_ns"][index])
             - int(phases[str(label)]["command_receipt_ns"])) / 1.0e6
            for index, label in zip(indices, phase_labels)], dtype=np.float64),
        "phase_labels": phase_labels,
        "phase_ids": np.asarray(
            [f"{run_id}:{label}" for label in phase_labels]),
    }


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in ("x_timed", "x_short_timed", "y", "age_ms",
                        "phase_labels", "phase_ids")}


def _choose_by_loco(train: dict[str, dict[str, np.ndarray]],
                    event: str) -> dict[str, Any]:
    candidates = {
        f"{view}__{'mirrored' if mirror else 'plain'}":
            (view, mirror, "extra_trees")
        for view in VIEWS for mirror in (False, True)
    }
    candidates.update({name: (view, mirror, "mlp")
                       for name, (view, mirror) in MLP_CANDIDATES.items()})
    fold_rows: dict[str, dict[str, Any]] = {}
    for heldout, validation in train.items():
        fit = _combine([data for run_id, data in train.items()
                        if run_id != heldout])
        predictions = {}
        for name, (view, mirror, family) in candidates.items():
            prediction = _fit_predict(fit, validation, view, mirror, family)
            predictions[name] = _metrics(prediction - validation["y"])
        fold_rows[heldout] = predictions
    summary = {}
    for name in candidates:
        rows = [fold[name] for fold in fold_rows.values()]
        summary[name] = {
            "run_macro_rmse_radps": float(np.mean(
                [row["rmse_radps"] for row in rows])),
            "total_samples_over_0p1": int(sum(
                row["samples_over_0p1"] for row in rows)),
            "worst_capture_max_abs_radps": float(max(
                row["max_abs_radps"] for row in rows)),
        }
    selected = min(summary, key=lambda name: (
        summary[name]["total_samples_over_0p1"],
        summary[name]["run_macro_rmse_radps"],
        summary[name]["worst_capture_max_abs_radps"],
    ))
    baseline = summary["full_long__mirrored"]
    best = summary[selected]
    return {
        "event": event,
        "selection_rule": "training-only leave-one-capture-out: minimize count above 0.1 rad/s, then run-macro RMSE, then maximum",
        "selected_candidate": selected,
        "selected_candidate_improves_training_cv_miss_count": (
            best["total_samples_over_0p1"]
            < baseline["total_samples_over_0p1"]),
        "full_long_mirrored_baseline": baseline,
        "summary": summary,
        "folds": fold_rows,
    }


def _holdout_report(train: dict[str, dict[str, np.ndarray]],
                    validation: dict[str, np.ndarray], run_id: str,
                    phases: dict[str, Any], rows: dict[str, Any],
                    event: str, selected: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fit = _combine(list(train.values()))
    if selected in MLP_CANDIDATES:
        view, mirror = MLP_CANDIDATES[selected]
        family = "mlp"
    else:
        view, mirror_text = selected.rsplit("__", 1)
        mirror = mirror_text == "mirrored"
        family = "extra_trees"
    models = {
        "full_long__mirrored": _fit_predict(
            fit, validation, "full_long", True),
        selected: _fit_predict(fit, validation, view, mirror, family),
    }
    report: dict[str, Any] = {"samples": int(len(validation["y"])),
                              "models": {}, "by_transition_age": {}}
    for name, prediction in models.items():
        error = prediction - validation["y"]
        report["models"][name] = _metrics(error)
    for label, low, high in (("0_to_25ms", 0.0, 25.0),
                             ("25_to_50ms", 25.0, 50.0),
                             ("50_to_150ms", 50.0, 150.0),
                             ("150ms_plus", 150.0, float("inf"))):
        mask = (validation["age_ms"] >= low) & (validation["age_ms"] < high)
        if np.any(mask):
            report["by_transition_age"][label] = {
                name: _metrics((prediction - validation["y"])[mask])
                for name, prediction in models.items()}
    selected_prediction = models[selected]
    baseline_prediction = models["full_long__mirrored"]
    output_rows = []
    for local, source_index in enumerate(validation["indices"]):
        phase_label = str(validation["phase_labels"][local])
        phase = phases[phase_label]
        x = rows["x"][source_index]
        output_rows.append({
            "run_id": run_id,
            "event": event,
            "response_packet_count": 2,
            "phase": phase_label,
            "age_from_command_receipt_ms": float(validation["age_ms"][local]),
            "requested_speed_mps": float(phase["condition"]["speed_mps"]),
            "requested_abs_steering_rad": float(
                phase["condition"]["steering_abs_rad"]),
            "turn_sign": int(phase["condition"]["turn_sign"]),
            "transition_mode": str(phase["condition"]["transition_mode"]),
            "truth_residual_radps": float(validation["y"][local]),
            "baseline_prediction_radps": float(baseline_prediction[local]),
            "selected_prediction_radps": float(selected_prediction[local]),
            "baseline_abs_error_radps": float(abs(
                baseline_prediction[local] - validation["y"][local])),
            "selected_abs_error_radps": float(abs(
                selected_prediction[local] - validation["y"][local])),
            "steering_feedback_rad": float(x[0]),
            "steering_command_rad": float(x[7]),
            "current_imu_yaw_rate_radps": float(x[6]),
        })
    return report, output_rows


def run() -> dict[str, Any]:
    admitted, _source_audit = teacher._discover_series_with_safe_mixed_archives()
    collected, phase_audits = response._collect_exact_two_packet(admitted)
    rows_by_run = collected["rows"]
    phases_by_run = collected["phases"]
    train_ids = sorted(run_id for run_id in response.TIMING_SUFFIX
                       if "_train_" in run_id and run_id in rows_by_run)
    train_by_event = {
        event: {run_id: _event_data(
            rows_by_run[run_id], phases_by_run[run_id], run_id, event)
            for run_id in train_ids}
        for event in EVENTS}
    selections = {event: _choose_by_loco(train_by_event[event], event)
                  for event in EVENTS}
    reports: dict[str, Any] = {}
    row_output: list[dict[str, Any]] = []
    for holdout in HOLDOUTS:
        if holdout not in rows_by_run:
            raise ValueError(f"missing registered holdout: {holdout}")
        reports[holdout] = {}
        for event in EVENTS:
            validation = _event_data(
                rows_by_run[holdout], phases_by_run[holdout], holdout, event)
            report, output_rows = _holdout_report(
                train_by_event[event], validation, holdout,
                phases_by_run[holdout], rows_by_run[holdout], event,
                selections[event]["selected_candidate"])
            reports[holdout][event] = report
            row_output.extend(output_rows)
    result = {
        "title": "Exact-two-packet yaw model command-feature ablation",
        "status": "offline analysis only; no estimator integrated",
        "target": "next 25-ms simulator-truth yaw-rate residual relative to current exact-packet IMU yaw rate (rad/s)",
        "packet_policy": "fit and score only phases with exactly two response packets; all other counts excluded",
        "input_contract": "current/past permitted sensors, commands, and receipt/source timing only; no future sensors, simulator truth, or restricted debug topic as model inputs",
        "training_runs": train_ids,
        "selected_by_training_only_leave_one_capture_out": selections,
        "whole_run_holdouts": reports,
        "phase_audits": phase_audits,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        fields = list(row_output[0]) if row_output else []
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row_output)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "report": str(OUTPUT.relative_to(ROOT)),
        "rows": str(CSV_OUTPUT.relative_to(ROOT)),
        "selection": {event: {
            "selected": value["selected_candidate"],
            "baseline": value["full_long_mirrored_baseline"],
            "selected_summary": value["summary"][value["selected_candidate"]],
        } for event, value in result["selected_by_training_only_leave_one_capture_out"].items()},
        "holdouts": result["whole_run_holdouts"],
    }, indent=2))
