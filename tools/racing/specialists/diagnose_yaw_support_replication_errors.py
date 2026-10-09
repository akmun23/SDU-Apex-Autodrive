#!/usr/bin/env python3
"""Export row-level causes for errors on the sealed support-replication run.

This deliberately fits only the already-evaluated sensor-only candidates and
never changes a runtime component. Simulator truth is used only as the
one-step target and for diagnostic grouping.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

try:
    import evaluate_yaw_full_spectrum_exact_two as evaluator
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/"
    "yaw_support_replication_error_diagnosis")


def _metrics(error: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(error, dtype=np.float64))
    return {
        "samples": int(len(absolute)),
        "rmse_radps": float(np.sqrt(np.mean(absolute ** 2))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _grouped_errors(error: np.ndarray, speed: np.ndarray,
                    steering: np.ndarray, event: np.ndarray
                    ) -> dict[str, Any]:
    speed_bin = np.floor(np.asarray(speed) / 0.5).astype(int)
    steer_bin = np.rint(np.asarray(steering) / 0.05).astype(int)
    keys = np.asarray([
        f"v{v * 0.5:.2f}_steer{d * 0.05:+.2f}_{e}"
        for v, d, e in zip(speed_bin, steer_bin, event.astype(str))])
    output: dict[str, Any] = {}
    for key in np.unique(keys):
        values = np.asarray(error)[keys == key]
        output[key] = _metrics(values)
    return output


def _predict_cell_event_specialists(
        train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
        train_x: np.ndarray, valid_x: np.ndarray,
        event_fallback: np.ndarray
        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit local experts by measured speed/steering cell and causal event.

    Unsupported cell-event pairs route to the already-fit causal-event
    specialist. The selector uses only wheel speed, measured steering, and
    the current-sensor event class.
    """
    train_cells = evaluator._cell_ids(train)
    valid_cells = evaluator._cell_ids(validation)
    train_events = train["causal_event"].astype(str)
    valid_events = validation["causal_event"].astype(str)
    train_phases = train["phase_id"].astype(str)
    train_runs = train["run_id"].astype(str)
    prediction = event_fallback.copy()
    support: dict[str, Any] = {}
    local_pairs = local_rows = fallback_rows = 0
    group_keys = np.asarray([
        f"{int(cell)}|{event}"
        for cell, event in zip(train_cells, train_events)])
    valid_group_keys = np.asarray([
        f"{int(cell)}|{event}"
        for cell, event in zip(valid_cells, valid_events)])
    for key in np.unique(valid_group_keys):
        train_idx = np.flatnonzero(group_keys == key)
        valid_idx = np.flatnonzero(valid_group_keys == key)
        phases = int(len(np.unique(train_phases[train_idx])))
        runs = int(len(np.unique(train_runs[train_idx])))
        supported = (len(train_idx) >= evaluator.MIN_LOCAL_ROWS
                     and phases >= evaluator.MIN_LOCAL_PHASES
                     and runs >= evaluator.MIN_LOCAL_INDEPENDENT_RUNS)
        support[key] = {
            "rows": int(len(train_idx)),
            "transition_phases": phases,
            "independent_training_runs": runs,
            "supported": bool(supported),
        }
        if supported:
            model = evaluator._fit(
                train_x[train_idx], train["y"][train_idx],
                train_phases[train_idx])
            prediction[valid_idx] = model.predict(valid_x[valid_idx])
            local_pairs += 1
            local_rows += len(valid_idx)
        else:
            fallback_rows += len(valid_idx)
    return prediction, {
        "local_cell_event_pairs_used": int(local_pairs),
        "local_rows_used": int(local_rows),
        "causal_event_fallback_rows": int(fallback_rows),
        "minimum_rows": evaluator.MIN_LOCAL_ROWS,
        "minimum_transition_phases": evaluator.MIN_LOCAL_PHASES,
        "minimum_independent_training_runs": (
            evaluator.MIN_LOCAL_INDEPENDENT_RUNS),
        "pair_support": support,
    }


def _cell_event_keys(data: dict[str, np.ndarray]) -> np.ndarray:
    cells = evaluator._cell_ids(data)
    events = data["causal_event"].astype(str)
    return np.asarray([f"{int(cell)}|{event}"
                       for cell, event in zip(cells, events)])


def _training_only_oof_gate(
        train_runs: dict[str, dict[str, np.ndarray]],
        throttle_train: dict[str, np.ndarray], fold_count: int
        ) -> dict[str, Any]:
    """Select local-vs-pooled event experts using run-grouped OOF data only.

    Every source run is predicted by models that did not see any rows from
    that run. Throttle training data remain in each fold because they are a
    separate training-only source. The selector is deliberately conservative:
    it needs repeated held-out-run wins in both RMSE and the requested 0.1
    rad/s tail metric before routing a cell/event to its local expert.
    """
    run_ids = sorted(train_runs)
    folds = min(int(fold_count), len(run_ids))
    if folds < 3:
        raise ValueError("run-grouped OOF gate requires at least three folds")
    row_groups = np.concatenate([
        np.full(len(train_runs[run_id]["y"]), run_id, dtype="U128")
        for run_id in run_ids])
    dummy_x = np.zeros((len(row_groups), 1), dtype=np.float32)
    splitter = GroupKFold(n_splits=folds)
    fold_runs: list[list[str]] = []
    ordered_run_ids = np.concatenate([
        np.full(len(train_runs[run_id]["y"]), run_id, dtype="U128")
        for run_id in run_ids])
    for _, heldout_indices in splitter.split(dummy_x, groups=row_groups):
        fold_runs.append(sorted(set(ordered_run_ids[heldout_indices].tolist())))

    # Store raw per-run residuals so selection is based on independent-run
    # summaries, not on pooled rows that would overweight long phases.
    residuals: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    fold_reports: list[dict[str, Any]] = []
    for fold_index, heldout_ids in enumerate(fold_runs):
        heldout_set = set(heldout_ids)
        fold_train = evaluator._combine({
            **{run_id: data for run_id, data in train_runs.items()
               if run_id not in heldout_set},
            evaluator.THROTTLE_TRAIN_ID: throttle_train,
        })
        fold_x = evaluator._model_features(fold_train)
        global_model = evaluator._fit(
            fold_x, fold_train["y"], fold_train["phase_id"])
        event_models: dict[str, Any] = {}
        fold_events = fold_train["causal_event"].astype(str)
        for event in evaluator.EVENTS:
            indices = np.flatnonzero(fold_events == event)
            phases = len(np.unique(fold_train["phase_id"][indices]))
            if len(indices) >= 200 and phases >= 8:
                event_models[event] = evaluator._fit(
                    fold_x[indices], fold_train["y"][indices],
                    fold_train["phase_id"][indices])

        fold_local_pairs = 0
        heldout_fold = evaluator._combine({
            run_id: train_runs[run_id] for run_id in heldout_ids})
        heldout_x = evaluator._model_features(heldout_fold)
        pooled = global_model.predict(heldout_x)
        heldout_events = heldout_fold["causal_event"].astype(str)
        for event, model in event_models.items():
            event_indices = np.flatnonzero(heldout_events == event)
            if len(event_indices):
                pooled[event_indices] = model.predict(heldout_x[event_indices])

        train_keys = _cell_event_keys(fold_train)
        valid_keys = _cell_event_keys(heldout_fold)
        local = pooled.copy()
        for key in np.unique(valid_keys):
            train_indices = np.flatnonzero(train_keys == key)
            valid_indices = np.flatnonzero(valid_keys == key)
            if not len(train_indices):
                continue
            phases = len(np.unique(fold_train["phase_id"][train_indices]))
            runs = len(np.unique(fold_train["run_id"][train_indices]))
            supported = (
                len(train_indices) >= evaluator.MIN_LOCAL_ROWS
                and phases >= evaluator.MIN_LOCAL_PHASES
                and runs >= evaluator.MIN_LOCAL_INDEPENDENT_RUNS)
            if supported:
                expert = evaluator._fit(
                    fold_x[train_indices], fold_train["y"][train_indices],
                    fold_train["phase_id"][train_indices])
                local[valid_indices] = expert.predict(heldout_x[valid_indices])
                fold_local_pairs += 1

        heldout_run_ids = heldout_fold["run_id"].astype(str)
        for run_id in heldout_ids:
            run_mask = heldout_run_ids == run_id
            for key in np.unique(valid_keys[run_mask]):
                indices = np.flatnonzero(run_mask & (valid_keys == key))
                entry = residuals.setdefault(key, {}).setdefault(
                    run_id, {"pooled": [], "local": []})
                entry["pooled"].append(
                    pooled[indices] - heldout_fold["y"][indices])
                entry["local"].append(
                    local[indices] - heldout_fold["y"][indices])
        fold_reports.append({
            "fold": int(fold_index),
            "heldout_runs": heldout_ids,
            "training_runs": sorted(heldout_set.symmetric_difference(
                set(run_ids))),
            "supported_local_experts_used": int(fold_local_pairs),
        })

    selector: dict[str, Any] = {}
    selected_keys: list[str] = []
    for key, run_records in sorted(residuals.items()):
        per_run: dict[str, Any] = {}
        rmse_wins = tail_wins = 0
        pooled_tail = local_tail = 0
        for run_id, models in sorted(run_records.items()):
            pooled_error = np.concatenate(models["pooled"])
            local_error = np.concatenate(models["local"])
            pooled_abs, local_abs = np.abs(pooled_error), np.abs(local_error)
            pooled_rmse = float(np.sqrt(np.mean(pooled_error ** 2)))
            local_rmse = float(np.sqrt(np.mean(local_error ** 2)))
            pooled_over = int(np.count_nonzero(pooled_abs > 0.1))
            local_over = int(np.count_nonzero(local_abs > 0.1))
            rmse_wins += int(local_rmse < pooled_rmse)
            tail_wins += int(local_over < pooled_over)
            pooled_tail += pooled_over
            local_tail += local_over
            per_run[run_id] = {
                "samples": int(len(pooled_error)),
                "pooled_rmse_radps": pooled_rmse,
                "local_rmse_radps": local_rmse,
                "pooled_over_0p1": pooled_over,
                "local_over_0p1": local_over,
                "pooled_p95_abs_radps": float(np.quantile(pooled_abs, 0.95)),
                "local_p95_abs_radps": float(np.quantile(local_abs, 0.95)),
            }
        run_count = len(per_run)
        minimum_wins = int(np.ceil(2.0 * run_count / 3.0))
        use_local = (
            run_count >= 3
            and sum(value["samples"] for value in per_run.values()) >= 40
            and rmse_wins >= minimum_wins
            and tail_wins >= minimum_wins
            and local_tail < pooled_tail)
        if use_local:
            selected_keys.append(key)
        selector[key] = {
            "use_local": bool(use_local),
            "independent_oof_runs": run_count,
            "run_level_rmse_wins": rmse_wins,
            "run_level_tail_wins": tail_wins,
            "pooled_over_0p1": pooled_tail,
            "local_over_0p1": local_tail,
            "per_run": per_run,
        }
    return {
        "fold_count": folds,
        "folds": fold_reports,
        "minimum_independent_oof_runs": 3,
        "selection_rule": (
            "local selected only with >=3 held-out source runs, >=40 rows, "
            "at least two-thirds run-level RMSE and >0.1-rad/s-tail wins, "
            "and fewer aggregate >0.1-rad/s errors"),
        "selected_cell_event_keys": selected_keys,
        "selected_cell_event_count": len(selected_keys),
        "cell_event_evidence": selector,
    }


def _write_failures(path: Path, data: dict[str, np.ndarray],
                    prediction: np.ndarray, model_name: str,
                    history_lags: tuple[int, ...]) -> int:
    error = np.asarray(prediction) - data["y"]
    selected = np.flatnonzero(np.abs(error) > 0.1)
    names = evaluator.atlas.OBSERVATION_NAMES
    width = len(history_lags) * len(names)
    derived = {name: width + i for i, name in
               enumerate(evaluator.atlas.DERIVED_NAMES)}
    fields = (
        "row_index", "model", "run_id", "condition_id", "phase_event",
        "sensor_event", "event_age_ms", "gt_speed_mps", "wheel_speed_mps",
        "wheel_body_mismatch_mps", "steering_rad", "rear_wheel_split_mps",
        "steering_feedback_rate_radps", "steering_command_gap_rad",
        "throttle_feedback_rate_per_s", "throttle_command_gap_norm",
        "imu_yaw_rate_rps", "imu_roll_rad", "imu_roll_rate_rps",
        "target_residual_radps", "prediction_residual_radps",
        "signed_error_radps", "absolute_error_radps")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in selected:
            x = data["x"][index]
            writer.writerow({
                "row_index": int(index), "model": model_name,
                "run_id": str(data["run_id"][index]),
                "condition_id": str(data["condition_id"][index]),
                "phase_event": str(data["phase_event"][index]),
                "sensor_event": str(data["causal_event"][index]),
                "event_age_ms": float(data["event_age_ms"][index]),
                "gt_speed_mps": float(data["gt_speed"][index]),
                "wheel_speed_mps": float(data["wheel_speed"][index]),
                "wheel_body_mismatch_mps": float(
                    data["wheel_speed"][index] - data["gt_speed"][index]),
                "steering_rad": float(data["steering"][index]),
                "rear_wheel_split_mps": float(x[derived["rear_wheel_split_mps"]]),
                "steering_feedback_rate_radps": float(
                    x[derived["steering_feedback_rate_radps"]]),
                "steering_command_gap_rad": float(
                    x[derived["steering_command_gap_rad"]]),
                "throttle_feedback_rate_per_s": float(
                    x[derived["throttle_feedback_rate_per_s"]]),
                "throttle_command_gap_norm": float(
                    x[derived["throttle_command_gap_norm"]]),
                "imu_yaw_rate_rps": float(x[6]),
                "imu_roll_rad": float(x[9]),
                "imu_roll_rate_rps": float(x[10]),
                "target_residual_radps": float(data["y"][index]),
                "prediction_residual_radps": float(prediction[index]),
                "signed_error_radps": float(error[index]),
                "absolute_error_radps": float(abs(error[index])),
            })
    return int(len(selected))


def run(output: Path, history_lags: tuple[int, ...],
        oof_folds: int = 0) -> dict[str, Any]:
    admitted, source_audit = evaluator.teacher._discover_series_with_safe_mixed_archives()
    by_id = {series.run_id: series for series in admitted}
    required_train_ids = set(evaluator.prior.TRAIN_IDS) | {
        evaluator.FULL_TRAIN_ID, evaluator.ENVELOPE_TRAIN_ID,
        evaluator.SPARSE_SUPPORT_TRAIN_ID,
        evaluator.SPARSE_CRAWL_TRAIN_ID,
        evaluator.SUPPORT_REPLICATION_TRAIN_ID}
    optional_train_ids = {evaluator.FULL_TRAIN_REPLICATION_ID}
    required_validation_ids = (
        evaluator.FULL_VALIDATION_ID,
        evaluator.SUPPORT_REPLICATION_VALIDATION_ID,
    )
    optional_validation_ids = {evaluator.FULL_VALIDATION_REPLICATION_ID}
    train_ids = required_train_ids | (optional_train_ids & set(by_id))
    validation_ids = (*required_validation_ids,
                      *(optional_validation_ids & set(by_id)))
    if required_train_ids - set(by_id) or set(required_validation_ids) - set(by_id):
        raise ValueError("required sealed train/validation sources are missing")
    if any(by_id[key].split != "train" for key in train_ids):
        raise ValueError("a candidate training run is not assigned to train")
    if any(by_id[key].split != "validation" for key in validation_ids):
        raise ValueError("a requested capture is not held out")

    train_runs: dict[str, dict[str, np.ndarray]] = {}
    for run_id in sorted(train_ids):
        data, _ = evaluator._collect_run(by_id[run_id], history_lags)
        data.pop("gt_body_state")
        train_runs[run_id] = data
    validations: dict[str, dict[str, np.ndarray]] = {}
    validation_audits = {}
    for validation_id in validation_ids:
        data, audit = evaluator._collect_run(by_id[validation_id], history_lags)
        data.pop("gt_body_state")
        validations[validation_id] = data
        validation_audits[validation_id] = audit
    throttle_runs, _ = evaluator._collect_throttle_runs(history_lags)
    throttle_train = throttle_runs[evaluator.THROTTLE_TRAIN_ID]
    train = evaluator._combine({**train_runs,
                                evaluator.THROTTLE_TRAIN_ID: throttle_train})
    train_x = evaluator._model_features(train)
    oof_gate = (_training_only_oof_gate(train_runs, throttle_train, oof_folds)
                if oof_folds else None)
    global_model = evaluator._fit(train_x, train["y"], train["phase_id"])
    event_support: dict[str, Any] = {}
    event_models = {}
    for event in evaluator.EVENTS:
        train_idx = np.flatnonzero(train["causal_event"].astype(str) == event)
        phases = int(len(np.unique(train["phase_id"][train_idx])))
        supported = len(train_idx) >= 200 and phases >= 8
        event_support[event] = {
            "rows": int(len(train_idx)), "transition_phases": phases,
            "supported": bool(supported)}
        if supported:
            event_models[event] = evaluator._fit(
                train_x[train_idx], train["y"][train_idx],
                train["phase_id"][train_idx])

    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "heldout_runs": list(validation_ids),
        "training_runs": sorted(train_ids),
        "exact_two_only": True,
        "runtime_or_physics_changed": False,
        "future_truth_used_as_input": False,
        "history_lags_packets": list(history_lags),
        "training_row_count": int(len(train["y"])),
        "validation_row_counts": {
            key: int(len(data["y"])) for key, data in validations.items()},
        "validation_capture_audits": validation_audits,
        "source_discovery_audit": source_audit,
        "event_training_support": event_support,
        "training_only_oof_gate": oof_gate,
        "per_validation_run": {},
    }
    for validation_id, validation in validations.items():
        valid_x = evaluator._model_features(validation)
        global_prediction = global_model.predict(valid_x)
        event_prediction = global_prediction.copy()
        for event, expert in event_models.items():
            valid_idx = np.flatnonzero(
                validation["causal_event"].astype(str) == event)
            if len(valid_idx):
                event_prediction[valid_idx] = expert.predict(valid_x[valid_idx])
        local_prediction, local_support = evaluator._predict_local(
            train, validation, train_x, valid_x, global_prediction)
        hybrid_prediction, hybrid_support = evaluator._local_event_hybrid(
            local_prediction, event_prediction, validation, local_support,
            minimum_runs=evaluator.MIN_LOCAL_INDEPENDENT_RUNS)
        cell_event_prediction, cell_event_support = (
            _predict_cell_event_specialists(
                train, validation, train_x, valid_x, event_prediction))
        predictions = {
            "all_data_global": global_prediction,
            "causal_event_specialists": event_prediction,
            "measured_speed_steering_local": local_prediction,
            "local_event_hybrid_min2_runs": hybrid_prediction,
            "cell_event_specialists": cell_event_prediction,
        }
        if oof_gate is not None:
            gated_prediction = event_prediction.copy()
            selected_keys = set(oof_gate["selected_cell_event_keys"])
            valid_keys = _cell_event_keys(validation)
            for key in selected_keys:
                mask = valid_keys == key
                gated_prediction[mask] = cell_event_prediction[mask]
            predictions["training_oof_selected_cell_event"] = gated_prediction
        run_report: dict[str, Any] = {
            "local_cell_support": local_support,
            "hybrid_selector_support": hybrid_support,
            "cell_event_selector_support": cell_event_support,
            "models": {},
        }
        for name, prediction in predictions.items():
            error = prediction - validation["y"]
            failure_count = _write_failures(
                output / f"{validation_id}_{name}_errors_over_0p1.csv",
                validation, prediction, name, history_lags)
            run_report["models"][name] = {
                "metrics": _metrics(error),
                "failure_rows_exported": failure_count,
                "error_groups_by_speed_steering_event": _grouped_errors(
                    error, validation["gt_speed"], validation["steering"],
                    validation["phase_event"]),
            }
        report["per_validation_run"][validation_id] = run_report
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True,
                   default=evaluator.atlas._json_value) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history-lags", default=",".join(
        map(str, evaluator.atlas.HISTORY_LAGS)))
    parser.add_argument(
        "--oof-folds", type=int, default=0,
        help=("optional run-grouped training-only CV folds used to select "
              "cell/event local experts; costs additional fitting time"))
    args = parser.parse_args()
    history_lags = tuple(int(value) for value in args.history_lags.split(","))
    report = run(args.output.resolve(), history_lags, args.oof_folds)
    print(json.dumps({
        "report": str((args.output.resolve() / "report.json").relative_to(ROOT)),
        "per_validation_run": {
            run: {name: value["metrics"]
                  for name, value in details["models"].items()}
            for run, details in report["per_validation_run"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
