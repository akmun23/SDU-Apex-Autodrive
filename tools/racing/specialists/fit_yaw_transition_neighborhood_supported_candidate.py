#!/usr/bin/env python3
"""Test neighbor-pooled experts on supported unwind/reversal atlas cells.

Unlike the fallback-gap candidate, this experiment replaces local experts in
all validation-observed, atlas-supported cells for the two target events.
The cell set is selected from sensor support only, not from validation error.
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
GAP_CANDIDATE = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v1"
OUTPUT = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported"
EVENTS = {"unwind", "reversal"}
SPEED_RADIUS = 1
STEERING_RADIUS = 2
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
        "runs_candidate_better": int(np.count_nonzero(values < 0.0)),
        "runs_candidate_worse": int(np.count_nonzero(values > 0.0)),
        "paired_run_bootstrap_95pct_ci_radps": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise RuntimeError("empty split")
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]}


def run() -> dict[str, Any]:
    report = json.loads((BASE / "sensor_only_yaw_regime_atlas_report.json")
                        .read_text(encoding="utf-8"))
    atlas_bundle = joblib.load(BASE / "sensor_only_yaw_regime_atlas.joblib")
    gap_bundle = joblib.load(GAP_CANDIDATE / "candidate.joblib")
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    train_parts, val_parts = [], []
    for series in admitted:
        if series.split not in ("train", "validation"):
            raise ValueError(f"unexpected split: {series.run_id} {series.split}")
        rows = atlas._rows(atlas._read_sensor_run(series), "command_intent")
        (train_parts if series.split == "train" else val_parts).append(rows)
    tr, va = _combine(train_parts), _combine(val_parts)
    width = len(report["input_contract"]["features"])
    x_train, x_val = tr["x"][:, :width], va["x"][:, :width]
    base_pred = audit._predict(va, atlas_bundle, width)[2]
    base_error = base_pred - va["residual"]

    train_speed = atlas._speed_cell(tr["wheel_speed"])
    train_steer = atlas._steer_cell(tr["steering"])
    val_speed = atlas._speed_cell(va["wheel_speed"])
    val_steer = atlas._steer_cell(va["steering"])
    train_groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    val_groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for i, key in enumerate(zip(tr["event"].astype(str), train_speed, train_steer)):
        train_groups[(key[0], int(key[1]), int(key[2]))].append(i)
    for i, key in enumerate(zip(va["event"].astype(str), val_speed, val_steer)):
        val_groups[(key[0], int(key[1]), int(key[2]))].append(i)

    # Keep the previously tested fallback gap-fill models exactly as they were.
    gap_pred = base_pred.copy()
    candidate_models = dict(gap_bundle["models"])
    candidate_support = dict(gap_bundle["support"])
    gap_keys = set(candidate_models)
    for key in gap_keys:
        val_idx = np.asarray(val_groups.get(key, ()), dtype=np.int64)
        if len(val_idx):
            gap_pred[val_idx] = candidate_models[key].predict(x_val[val_idx])
    gap_error = gap_pred - va["residual"]
    candidate_pred = gap_pred.copy()
    supported_keys = {
        key for key in val_groups
        if key[0] in EVENTS
        and (key[1], key[2], key[0]) in atlas_bundle["local_expert_models"]
    }
    fit_keys = sorted(supported_keys - gap_keys)
    fit_count, covered_rows = 0, 0
    for key in fit_keys:
        event, center_speed, center_steer = key
        indices = []
        for ds in range(-SPEED_RADIUS, SPEED_RADIUS + 1):
            for dd in range(-STEERING_RADIUS, STEERING_RADIUS + 1):
                indices.extend(train_groups.get((event, center_speed + ds,
                                                 center_steer + dd), ()))
        if not indices:
            continue
        indices = np.asarray(indices, dtype=np.int64)
        ids, counts = np.unique(tr["run_id"][indices].astype(str),
                                return_counts=True)
        eligible = ids[counts >= MIN_PER_RUN]
        keep = np.isin(tr["run_id"][indices].astype(str), eligible)
        fit_idx = indices[keep]
        if len(eligible) < MIN_RUNS or len(fit_idx) < MIN_SAMPLES:
            continue
        ids, id_counts = np.unique(tr["run_id"][fit_idx].astype(str),
                                   return_counts=True)
        count_by_id = dict(zip(ids.tolist(), id_counts.tolist()))
        weights = np.asarray([1.0 / count_by_id[str(run_id)]
                              for run_id in tr["run_id"][fit_idx]])
        weights *= len(ids) / weights.sum()
        model = ExtraTreesRegressor(**atlas.ESTIMATOR, n_jobs=1)
        model.fit(x_train[fit_idx], tr["residual"][fit_idx],
                  sample_weight=weights)
        val_idx = np.asarray(val_groups[key], dtype=np.int64)
        candidate_pred[val_idx] = model.predict(x_val[val_idx])
        candidate_models[key] = model
        candidate_support[f"{event}:{center_speed}:{center_steer}"] = {
            "training_samples": int(len(fit_idx)),
            "training_runs": int(len(ids)),
            "heldout_rows": int(len(val_idx)),
            "replaces_supported_exact_expert": True,
        }
        covered_rows += len(val_idx)
        fit_count += 1

    candidate_error = candidate_pred - va["residual"]
    by_event = {}
    for event in sorted(EVENTS):
        mask = va["event"] == event
        by_event[event] = {
            "baseline": _metric(base_error[mask]),
            "fallback_gap_fill_candidate": _metric(gap_error[mask]),
            "supported_neighborhood_candidate": _metric(candidate_error[mask]),
            "run_macro_vs_gap_fill": _macro_delta(
                candidate_error[mask], gap_error[mask], va["run_id"][mask]),
        }

    whole_macro = _macro_delta(candidate_error, gap_error, va["run_id"])
    artifact_eligible = (
        whole_macro["paired_run_bootstrap_95pct_ci_radps"][1] < 0.0
        and all(by_event[event]["supported_neighborhood_candidate"]["rmse_radps"]
                <= by_event[event]["fallback_gap_fill_candidate"]["rmse_radps"]
                for event in EVENTS))
    result = {
        "title": "Adjacent-cell neighborhood experts on supported unwind/reversal cells",
        "supervision": "simulator-GT next yaw rate; no production odometry labels or GT selector",
        "fit_split": "62 whole training runs only",
        "validation_split": "36 whole validation runs only; test/final-test sealed",
        "neighborhood": {"speed_radius_mps": SPEED_RADIUS * atlas.SPEED_BIN_MPS,
                         "steering_radius_rad": STEERING_RADIUS * atlas.STEERING_BIN_RAD},
        "supported_cells_available": len(supported_keys),
        "supported_cells_refit": fit_count,
        "supported_rows_refit": int(covered_rows),
        "whole_validation_baseline": _metric(base_error),
        "whole_validation_gap_fill_candidate": _metric(gap_error),
        "whole_validation_candidate": _metric(candidate_error),
        "whole_run_macro_vs_gap_fill_candidate": whole_macro,
        "by_event": by_event,
        "candidate_model_artifact_saved": bool(artifact_eligible),
        "source_audit": source_audit,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True,
                   default=audit._json_default) + "\n", encoding="utf-8")
    if artifact_eligible:
        joblib.dump({"models": candidate_models, "support": candidate_support,
                     "contract": report["input_contract"],
                     "target": "GT_yaw_rate[k+1] - IMU_yaw_rate[k]"},
                    OUTPUT / "candidate.joblib", compress=3)
    print("supported cells fit", fit_count, "heldout rows", covered_rows)
    print("baseline", result["whole_validation_baseline"])
    print("candidate", result["whole_validation_candidate"])
    print("macro", whole_macro)
    print("by event", json.dumps(by_event, sort_keys=True))
    print("artifact eligible", artifact_eligible)
    return result


if __name__ == "__main__":
    run()
