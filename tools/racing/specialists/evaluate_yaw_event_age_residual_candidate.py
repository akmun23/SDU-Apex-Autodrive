#!/usr/bin/env python3
"""Test whether causal steering-event age explains yaw residuals.

The base teacher is the existing sensor-only causal event model. A second
regressor is trained only on run-grouped out-of-fold residuals and receives
event age plus measured speed/steering context. The scheduled phase/event is
used only to select the already-audited response window and its event-age
target; it is not an input feature. Final scoring is whole-run held out.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

try:
    import evaluate_yaw_full_spectrum_exact_two as evaluator
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/"
    "yaw_event_age_residual_candidate_r03")
VALIDATION_IDS = (
    evaluator.FULL_VALIDATION_ID,
    evaluator.SUPPORT_REPLICATION_VALIDATION_ID,
)
TRAIN_IDS = set(evaluator.prior.TRAIN_IDS) | {
    evaluator.FULL_TRAIN_ID,
    evaluator.ENVELOPE_TRAIN_ID,
    evaluator.SPARSE_SUPPORT_TRAIN_ID,
    evaluator.SPARSE_CRAWL_TRAIN_ID,
    evaluator.SUPPORT_REPLICATION_TRAIN_ID,
    evaluator.FULL_TRAIN_REPLICATION_ID,
}


def _metrics(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _event_age_features(data: dict[str, np.ndarray]) -> np.ndarray:
    age_s = np.asarray(data["event_age_ms"], dtype=np.float32) / 1000.0
    speed = np.asarray(data["wheel_speed"], dtype=np.float32)
    steering = np.asarray(data["steering"], dtype=np.float32)
    return np.column_stack((
        evaluator._model_features(data),
        age_s,
        speed,
        steering,
        age_s * speed,
        age_s * steering,
        age_s * np.abs(steering),
    )).astype(np.float32)


def _load_runs(history_lags: tuple[int, ...]):
    admitted, source_audit = (
        evaluator.teacher._discover_series_with_safe_mixed_archives())
    by_id = {series.run_id: series for series in admitted}
    required = TRAIN_IDS | set(VALIDATION_IDS)
    missing = required - set(by_id)
    if missing:
        raise ValueError(f"required runs missing from safe archives: {sorted(missing)}")
    if any(by_id[run_id].split != "train" for run_id in TRAIN_IDS):
        raise ValueError("event-age training source has a non-training split")
    if any(by_id[run_id].split != "validation" for run_id in VALIDATION_IDS):
        raise ValueError("sealed validation source split changed")

    train_runs: dict[str, dict[str, np.ndarray]] = {}
    validation_runs: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, Any] = {}
    for run_id in sorted(required):
        data, audit = evaluator._collect_run(by_id[run_id], history_lags)
        data.pop("gt_body_state")
        audits[run_id] = audit
        (train_runs if run_id in TRAIN_IDS else validation_runs)[run_id] = data
    throttle_runs, throttle_audit = evaluator._collect_throttle_runs(history_lags)
    throttle_train = throttle_runs[evaluator.THROTTLE_TRAIN_ID]
    audits[evaluator.THROTTLE_TRAIN_ID] = {
        "split": "train",
        "source": "existing exact-two throttle training archive",
        "samples": int(len(throttle_train["y"])),
    }
    return train_runs, validation_runs, throttle_train, audits, {
        "sources": source_audit,
        "throttle": throttle_audit,
    }


def _write_large_errors(path: Path, data: dict[str, np.ndarray],
                        baseline: np.ndarray, corrected: np.ndarray) -> int:
    error = corrected - data["y"]
    indices = np.flatnonzero(np.abs(error) > 0.1)
    fields = (
        "run_id", "condition_id", "phase_event", "causal_event",
        "event_age_ms", "wheel_speed_mps", "steering_rad",
        "target_residual_radps", "baseline_prediction_radps",
        "corrected_prediction_radps", "signed_error_radps",
        "absolute_error_radps",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in indices:
            writer.writerow({
                "run_id": str(data["run_id"][index]),
                "condition_id": str(data["condition_id"][index]),
                "phase_event": str(data["phase_event"][index]),
                "causal_event": str(data["causal_event"][index]),
                "event_age_ms": float(data["event_age_ms"][index]),
                "wheel_speed_mps": float(data["wheel_speed"][index]),
                "steering_rad": float(data["steering"][index]),
                "target_residual_radps": float(data["y"][index]),
                "baseline_prediction_radps": float(baseline[index]),
                "corrected_prediction_radps": float(corrected[index]),
                "signed_error_radps": float(error[index]),
                "absolute_error_radps": float(abs(error[index])),
            })
    return int(len(indices))


def run(output: Path, fold_count: int,
        history_lags: tuple[int, ...]) -> dict[str, Any]:
    train_runs, validation_runs, throttle_train, audits, source_audit = (
        _load_runs(history_lags))
    run_ids = sorted(train_runs)
    folds = min(fold_count, len(run_ids))
    if folds < 3:
        raise ValueError("run-grouped residual fitting requires >=3 runs")

    oof_rows: dict[str, list[np.ndarray]] = {key: [] for key in (
        "x", "age_x", "residual", "phase_id", "run_id")}
    splitter = GroupKFold(n_splits=folds)
    row_groups = np.concatenate([
        np.full(len(train_runs[run_id]["y"]), run_id, dtype="U128")
        for run_id in run_ids])
    dummy = np.zeros((len(row_groups), 1), dtype=np.float32)
    ordered_ids = np.concatenate([
        np.full(len(train_runs[run_id]["y"]), run_id, dtype="U128")
        for run_id in run_ids])
    fold_reports = []

    for fold_index, (_, held_indices) in enumerate(
            splitter.split(dummy, groups=row_groups)):
        held_ids = sorted(set(ordered_ids[held_indices].tolist()))
        held_set = set(held_ids)
        fold_yaw_train = evaluator._combine({
            run_id: data for run_id, data in train_runs.items()
            if run_id not in held_set})
        fold_train = evaluator._combine({
            **{run_id: data for run_id, data in train_runs.items()
               if run_id not in held_set},
            evaluator.THROTTLE_TRAIN_ID: throttle_train,
        })
        base = evaluator._fit(
            evaluator._model_features(fold_train), fold_train["y"],
            fold_train["phase_id"])
        held = evaluator._combine({
            run_id: train_runs[run_id] for run_id in held_ids})
        base_x = evaluator._model_features(held)
        prediction = base.predict(base_x)
        oof_rows["x"].append(base_x)
        oof_rows["age_x"].append(_event_age_features(held))
        oof_rows["residual"].append(held["y"] - prediction)
        oof_rows["phase_id"].append(held["phase_id"])
        oof_rows["run_id"].append(held["run_id"])
        fold_reports.append({
            "fold": int(fold_index),
            "training_runs": sorted(set(run_ids) - held_set),
            "heldout_runs": held_ids,
            "training_rows_including_throttle": int(len(fold_train["y"])),
            "oof_yaw_rows": int(len(fold_yaw_train["y"])),
        })

    residual_x = np.concatenate(oof_rows["age_x"])
    residual_y = np.concatenate(oof_rows["residual"])
    residual_phases = np.concatenate(oof_rows["phase_id"])
    residual_model = ExtraTreesRegressor(**evaluator.MODEL)
    residual_model.fit(
        residual_x, residual_y,
        sample_weight=evaluator.causal._phase_weights(residual_phases))

    full_train = evaluator._combine({
        **train_runs,
        evaluator.THROTTLE_TRAIN_ID: throttle_train,
    })
    base_model = evaluator._fit(
        evaluator._model_features(full_train), full_train["y"],
        full_train["phase_id"])

    output_runs: dict[str, Any] = {}
    output.mkdir(parents=True, exist_ok=True)
    for run_id, data in validation_runs.items():
        base_prediction = base_model.predict(evaluator._model_features(data))
        correction = residual_model.predict(_event_age_features(data))
        corrected = base_prediction + correction
        baseline_error = base_prediction - data["y"]
        corrected_error = corrected - data["y"]
        per_event = {}
        for event in evaluator.EVENTS:
            mask = data["phase_event"].astype(str) == event
            if np.any(mask):
                per_event[event] = {
                    "baseline": _metrics(baseline_error[mask]),
                    "event_age_residual": _metrics(corrected_error[mask]),
                }
        error_file = output / f"{run_id}_event_age_errors_over_0p1.csv"
        large_error_count = _write_large_errors(
            error_file, data, base_prediction, corrected)
        output_runs[run_id] = {
            "baseline": _metrics(baseline_error),
            "event_age_residual": _metrics(corrected_error),
            "by_transition_type": per_event,
            "corrected_errors_over_0p1_csv": str(error_file.relative_to(ROOT)),
            "corrected_errors_over_0p1_rows": large_error_count,
            "exact_two_audit": audits[run_id]["exact_two_audit"],
        }

    report = {
        "candidate": "run-grouped OOF residual correction with command-event age",
        "runtime_or_physics_changed": False,
        "validation_arrays_used_for_fit_or_selection": False,
        "phase_labels_are_model_inputs": False,
        "age_source": "measured steering command_start_ns relative to packet time",
        "event_age_features": [
            "age_seconds", "measured_rear_wheel_speed", "measured_steering",
            "age_times_speed", "age_times_signed_steering",
            "age_times_absolute_steering",
        ],
        "training_run_ids": run_ids,
        "validation_run_ids": list(validation_runs),
        "training_only_grouped_oof_folds": fold_reports,
        "oof_residual_rows": int(len(residual_y)),
        "source_discovery": source_audit,
        "capture_audits": audits,
        "per_validation_run": output_runs,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({"report": str(output / "report.json"),
                      "per_validation_run": output_runs}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oof-folds", type=int, default=3)
    parser.add_argument(
        "--history-lags", default=",".join(map(
            str, evaluator.atlas.HISTORY_LAGS)))
    args = parser.parse_args()
    history_lags = tuple(int(value) for value in args.history_lags.split(","))
    run(args.output.resolve(), args.oof_folds, history_lags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
