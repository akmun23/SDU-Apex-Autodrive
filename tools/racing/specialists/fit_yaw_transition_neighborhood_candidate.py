#!/usr/bin/env python3
"""Test local-neighborhood yaw experts for unwind/reversal support gaps.

This is a GT-supervised, sensor-only offline candidate. It fits a local model
only where the frozen atlas falls back globally, pooling adjacent measured
wheel-speed and steering-feedback cells. Whole-run validation is untouched by
fitting; test/final-test data are rejected by the shared split gate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import audit_sensor_yaw_large_errors as audit
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import audit_sensor_yaw_large_errors as audit
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_command_intent_v2"
ERROR_AUDIT = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1"
OUTPUT = ERROR_AUDIT / "unwind_reversal_neighborhood_v1"
TARGET_EVENTS = {"unwind", "reversal"}
SPEED_RADIUS_CELLS = 1
STEERING_RADIUS_CELLS = 2
MIN_SAMPLES = 80
MIN_RUNS = 2
MIN_PER_RUN = 8


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
        "fraction_within_0p1": float(np.mean(absolute <= 0.1)),
    }


def _macro_delta(candidate: np.ndarray, baseline: np.ndarray,
                 run_ids: np.ndarray) -> dict[str, Any]:
    values = []
    for run_id in sorted(set(run_ids.astype(str))):
        mask = run_ids.astype(str) == run_id
        values.append(float(np.sqrt(np.mean(candidate[mask] ** 2))
                            - np.sqrt(np.mean(baseline[mask] ** 2))))
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(20261008)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "runs": int(len(values)),
        "mean_candidate_minus_baseline_run_rmse_radps": float(values.mean()),
        "median_candidate_minus_baseline_run_rmse_radps": float(np.median(values)),
        "runs_candidate_better": int(np.count_nonzero(values < 0.0)),
        "runs_candidate_worse": int(np.count_nonzero(values > 0.0)),
        "paired_run_bootstrap_95pct_ci_radps": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise RuntimeError("empty run set")
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]}


def run() -> dict[str, Any]:
    report = json.loads((BASE / "sensor_only_yaw_regime_atlas_report.json")
                        .read_text(encoding="utf-8"))
    bundle = joblib.load(BASE / "sensor_only_yaw_regime_atlas.joblib")
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    train_parts, val_parts = [], []
    for series in admitted:
        if series.split not in ("train", "validation"):
            raise ValueError(f"unexpected split admitted: {series.run_id} {series.split}")
        run_rows = atlas._rows(atlas._read_sensor_run(series), "command_intent")
        (train_parts if series.split == "train" else val_parts).append(run_rows)
    tr, va = _combine(train_parts), _combine(val_parts)
    expected_features = len(report["input_contract"]["features"])
    x_train = tr["x"][:, :expected_features]
    x_val = va["x"][:, :expected_features]
    baseline_residual = audit._predict(va, bundle, expected_features)[2]
    baseline_error = baseline_residual - va["residual"]
    if not np.isfinite(baseline_error).all():
        raise RuntimeError("frozen atlas left a validation row unscored")

    train_speed = atlas._speed_cell(tr["wheel_speed"])
    train_steer = atlas._steer_cell(tr["steering"])
    val_speed = atlas._speed_cell(va["wheel_speed"])
    val_steer = atlas._steer_cell(va["steering"])
    train_groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    val_groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, (event, speed, steering) in enumerate(zip(
            tr["event"].astype(str), train_speed, train_steer)):
        train_groups[(event, int(speed), int(steering))].append(index)
    for index, (event, speed, steering) in enumerate(zip(
            va["event"].astype(str), val_speed, val_steer)):
        val_groups[(event, int(speed), int(steering))].append(index)

    # Only evaluate meaningful cells already identified in the held-out audit:
    # at least 20 validation rows or one >0.1-rad/s miss.
    risk_report = json.loads((ERROR_AUDIT / "sensor_yaw_large_error_audit.json")
                             .read_text(encoding="utf-8"))
    evaluated_keys = set()
    for row in risk_report["by_observable_speed_steering_event_cell"]:
        if row["event"] not in TARGET_EVENTS or row["global_fallback_rows"] == 0:
            continue
        speed = int(atlas._speed_cell(np.asarray(
            [row["wheel_speed_cell_center_mps"]], dtype=np.float32))[0])
        steering = int(atlas._steer_cell(np.asarray(
            [row["steering_cell_center_rad"]], dtype=np.float32))[0])
        key = (row["event"], speed, steering)
        if key in val_groups and key not in bundle["local_expert_models"]:
            evaluated_keys.add(key)

    candidate_residual = baseline_residual.copy()
    candidate_models = {}
    support_rows = {}
    evaluated_fallback = 0
    candidate_covered = 0
    for key in sorted(evaluated_keys):
        event, center_speed, center_steer = key
        selected_train = []
        for speed_delta in range(-SPEED_RADIUS_CELLS, SPEED_RADIUS_CELLS + 1):
            for steer_delta in range(-STEERING_RADIUS_CELLS,
                                     STEERING_RADIUS_CELLS + 1):
                neighbor_key = (event, center_speed + speed_delta,
                                center_steer + steer_delta)
                selected_train.extend(train_groups.get(neighbor_key, ()))
        if not selected_train:
            continue
        selected_train = np.asarray(selected_train, dtype=np.int64)
        run_ids, counts = np.unique(tr["run_id"][selected_train].astype(str),
                                    return_counts=True)
        eligible_runs = run_ids[counts >= MIN_PER_RUN]
        if len(eligible_runs) < MIN_RUNS:
            continue
        keep = np.isin(tr["run_id"][selected_train].astype(str), eligible_runs)
        fit_indices = selected_train[keep]
        if len(fit_indices) < MIN_SAMPLES:
            continue

        ids, id_counts = np.unique(tr["run_id"][fit_indices].astype(str),
                                   return_counts=True)
        count_by_id = dict(zip(ids.tolist(), id_counts.tolist()))
        weights = np.asarray([1.0 / count_by_id[str(run_id)]
                              for run_id in tr["run_id"][fit_indices]])
        weights *= len(ids) / weights.sum()
        model = ExtraTreesRegressor(**atlas.ESTIMATOR, n_jobs=1)
        model.fit(x_train[fit_indices], tr["residual"][fit_indices],
                  sample_weight=weights)
        val_indices = np.asarray(val_groups[key], dtype=np.int64)
        candidate_residual[val_indices] = model.predict(x_val[val_indices])
        candidate_models[key] = model
        candidate_covered += len(val_indices)
        evaluated_fallback += len(val_indices)
        support_rows[f"{event}:{center_speed}:{center_steer}"] = {
            "training_samples": int(len(fit_indices)),
            "training_runs": int(len(ids)),
            "heldout_rows": int(len(val_indices)),
            "neighbor_speed_radius_mps": SPEED_RADIUS_CELLS * atlas.SPEED_BIN_MPS,
            "neighbor_steering_radius_rad": STEERING_RADIUS_CELLS * atlas.STEERING_BIN_RAD,
        }

    candidate_error = candidate_residual - va["residual"]
    if not np.isfinite(candidate_error).all():
        raise RuntimeError("candidate produced non-finite validation error")
    event_metrics = {}
    for event in ("unwind", "reversal"):
        mask = va["event"] == event
        event_metrics[event] = {
            "baseline": _metric(baseline_error[mask]),
            "candidate": _metric(candidate_error[mask]),
            "run_macro": _macro_delta(candidate_error[mask], baseline_error[mask],
                                       va["run_id"][mask]),
        }
    changed = candidate_residual != baseline_residual
    whole_run_macro = _macro_delta(candidate_error, baseline_error,
                                   va["run_id"])
    artifact_eligible = (
        whole_run_macro["paired_run_bootstrap_95pct_ci_radps"][1] < 0.0
        and all(event_metrics[event]["candidate"]["rmse_radps"]
                < event_metrics[event]["baseline"]["rmse_radps"]
                for event in TARGET_EVENTS))
    result = {
        "title": "Adjacent-cell local expert candidate for unwind/reversal fallback regions",
        "supervision": "simulator GT next yaw rate; production odometry is never used as label",
        "selector": "causal measured rear-wheel speed, measured steering, and command-intent event",
        "fit_split": "62 whole training runs only",
        "validation_split": "36 whole validation runs only; test/final-test sealed",
        "training_rows": int(len(tr["residual"])),
        "validation_rows": int(len(va["residual"])),
        "neighborhood": {
            "speed_radius_cells": SPEED_RADIUS_CELLS,
            "speed_radius_mps": SPEED_RADIUS_CELLS * atlas.SPEED_BIN_MPS,
            "steering_radius_cells": STEERING_RADIUS_CELLS,
            "steering_radius_rad": STEERING_RADIUS_CELLS * atlas.STEERING_BIN_RAD,
            "minimum_samples_after_per_run_filter": MIN_SAMPLES,
            "minimum_runs": MIN_RUNS,
            "minimum_samples_per_run": MIN_PER_RUN,
        },
        "candidate_models_fitted": len(candidate_models),
        "candidate_fallback_rows_covered": int(candidate_covered),
        "candidate_changed_validation_rows": int(np.count_nonzero(changed)),
        "whole_validation_baseline": _metric(baseline_error),
        "whole_validation_candidate": _metric(candidate_error),
        "whole_run_macro": whole_run_macro,
        "by_target_event": event_metrics,
        "support_by_cell": support_rows,
        "candidate_model_artifact_saved": bool(artifact_eligible),
        "source_audit": source_audit,
        "conclusion": "research-only; requires whole-run transfer and runtime review before any integration",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True,
                   default=audit._json_default) + "\n", encoding="utf-8")
    if candidate_models and artifact_eligible:
        joblib.dump({
            "contract": report["input_contract"],
            "target": "GT_yaw_rate[k+1] - IMU_yaw_rate[k]",
            "selector": result["selector"],
            "models": candidate_models,
            "support": support_rows,
        }, OUTPUT / "candidate.joblib", compress=3)
    print("models/covered rows", len(candidate_models), candidate_covered)
    print("whole baseline", result["whole_validation_baseline"])
    print("whole candidate", result["whole_validation_candidate"])
    print("whole run macro", result["whole_run_macro"])
    print("event results", json.dumps(event_metrics, sort_keys=True))
    print("wrote", OUTPUT)
    return result


if __name__ == "__main__":
    run()
