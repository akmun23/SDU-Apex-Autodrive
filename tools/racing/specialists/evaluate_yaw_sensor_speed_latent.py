#!/usr/bin/env python3
"""Test whether a sensor-only body-speed estimate improves yaw prediction.

The speed teacher is trained against simulator GT but consumes only causal
sensor/actuator features. Training predictions are grouped-run out-of-fold to
avoid feeding in-sample GT-speed estimates to the yaw model. Test and
final-test partitions are not admitted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from tools.racing.specialists import audit_yaw_full_domain_exact_two as audit
from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_gapfill_exact_two_short_history_r03")
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_sensor_speed_latent_r03")
N_SPLITS = 4


def _load_parts(source_dir: Path
                ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    admitted, corpus_audit = teacher._discover_series_with_safe_mixed_archives()
    by_id = {series.run_id: series for series in admitted}
    cache = json.loads((source_dir / "phase_audit.json").read_text(
        encoding="utf-8"))
    if not set(cache.get("admitted_run_ids", [])) <= set(by_id):
        raise ValueError("phase audit contains runs outside the admitted corpus")
    parts: dict[str, list[dict[str, np.ndarray]]] = {
        "train": [], "validation": []}
    for run_id, phase_audit in sorted(cache["runs"].items()):
        series = by_id.get(run_id)
        if series is None or series.split not in parts:
            continue
        rows = audit._collect_rows(series, phase_audit["measurements"])
        parts[series.split].append(rows)
    combined = {}
    for split, rows_by_run in parts.items():
        if not rows_by_run:
            raise ValueError(f"no exact-two {split} rows")
        keys = [key for key, value in rows_by_run[0].items()
                if isinstance(value, np.ndarray)]
        combined[split] = {
            key: np.concatenate([part[key] for part in rows_by_run])
            for key in keys
        }
    train_ids = set(combined["train"]["run_id"].astype(str))
    validation_ids = set(combined["validation"]["run_id"].astype(str))
    if train_ids & validation_ids:
        raise ValueError("train/validation run leakage")
    return combined["train"], combined["validation"], corpus_audit


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    if not len(error):
        return {"samples": 0}
    return {
        "samples": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
        "over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _fit(x: np.ndarray, y: np.ndarray, run_ids: np.ndarray
         ) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(**audit.ESTIMATOR).fit(
        x, y, sample_weight=audit._run_balanced_weights(run_ids))


def _score_regions(rows: dict[str, np.ndarray], error: np.ndarray,
                   selected: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    event = rows["response_event"].astype(str)
    for name in sorted(set(event[selected])):
        mask = selected & (event == name)
        result[f"response_event:{name}"] = _metric(error[mask])
    mismatch = np.abs(rows["gt_speed"] - rows["wheel_speed"])
    for low, high in ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, np.inf)):
        mask = selected & (mismatch >= low) & (mismatch < high)
        if np.any(mask):
            result[f"abs_gt_minus_wheel_speed:{low:g}+mps"] = {
                "mismatch_rows": int(mask.sum()),
                **_metric(error[mask]),
            }
    return result


def _paired_run_metrics(run_ids: np.ndarray, baseline_error: np.ndarray,
                        candidate_error: np.ndarray, selected: np.ndarray
                        ) -> dict[str, Any]:
    ids = np.asarray(run_ids).astype(str)
    run_names = np.unique(ids[selected])
    per_run = {
        run: (
            float(np.sqrt(np.mean(baseline_error[selected & (ids == run)] ** 2))),
            float(np.sqrt(np.mean(candidate_error[selected & (ids == run)] ** 2))),
        )
        for run in run_names
    }
    paired = np.asarray(list(per_run.values()), dtype=np.float64)
    delta = paired[:, 1] - paired[:, 0]
    rng = np.random.default_rng(20261009)
    draws = rng.integers(0, len(paired), size=(10_000, len(paired)))
    bootstrap_delta = np.sqrt(np.mean(paired[draws, 1] ** 2, axis=1)) - np.sqrt(
        np.mean(paired[draws, 0] ** 2, axis=1))
    return {
        "independent_runs": int(len(run_names)),
        "baseline_run_macro_rmse": float(np.mean(paired[:, 0])),
        "speed_latent_run_macro_rmse": float(np.mean(paired[:, 1])),
        "mean_per_run_rmse_delta_candidate_minus_baseline": float(np.mean(delta)),
        "bootstrap_95pct_delta_ci": [
            float(np.quantile(bootstrap_delta, 0.025)),
            float(np.quantile(bootstrap_delta, 0.975)),
        ],
    }


def run(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    train, validation, corpus_audit = _load_parts(source_dir)
    x_train = train["x_timed"].astype(np.float32)
    x_valid = validation["x_timed"].astype(np.float32)
    y_speed = train["gt_speed"].astype(np.float32)
    y_yaw_train = train["residual"].astype(np.float32)
    y_yaw_valid = validation["residual"].astype(np.float32)
    groups = train["run_id"].astype(str)
    unique_groups = np.unique(groups)
    if len(unique_groups) < N_SPLITS:
        raise ValueError("not enough independent training runs for grouped OOF")

    splitter = GroupKFold(n_splits=N_SPLITS)
    oof_speed = np.full(len(y_speed), np.nan, dtype=np.float32)
    fold_models = []
    for fold, (fit_idx, held_idx) in enumerate(
            splitter.split(x_train, y_speed, groups), start=1):
        model = _fit(x_train[fit_idx], y_speed[fit_idx], groups[fit_idx])
        oof_speed[held_idx] = model.predict(x_train[held_idx]).astype(np.float32)
        fold_models.append(model)
        print(f"speed OOF fold {fold}/{N_SPLITS}: "
              f"heldout_runs={len(np.unique(groups[held_idx]))}", flush=True)
    if not np.isfinite(oof_speed).all():
        raise RuntimeError("grouped speed OOF did not score every training row")
    # Average run-held-out speed teachers on validation to match the OOF
    # training-prediction protocol and avoid a full-fit/in-sample gap.
    valid_speed = np.mean(
        [model.predict(x_valid) for model in fold_models], axis=0
    ).astype(np.float32)

    speed_error = valid_speed - validation["gt_speed"]
    speed_metrics = {
        "all_validation": _metric(speed_error),
        "new_gapfill_run": _metric(speed_error[
            validation["run_id"].astype(str)
            == "openplane_yaw_exact_two_domain_gapfill_validation_20261008_r03"]),
    }
    mismatch = np.abs(validation["gt_speed"] - validation["wheel_speed"])
    for label, mask in (
        ("mismatch_ge_1mps", mismatch >= 1.0),
        ("mismatch_ge_3mps", mismatch >= 3.0),
        ("mismatch_ge_6mps", mismatch >= 6.0),
    ):
        speed_metrics[label] = _metric(speed_error[mask])

    x_train_aug = np.column_stack((x_train, oof_speed))
    x_valid_aug = np.column_stack((x_valid, valid_speed))
    train_event = train["event"].astype(str)
    valid_event = validation["event"].astype(str)
    base_prediction = np.full(len(y_yaw_valid), np.nan, dtype=np.float64)
    latent_prediction = np.full(len(y_yaw_valid), np.nan, dtype=np.float64)
    event_scores: dict[str, Any] = {}
    for name in sorted(set(train_event)):
        tr = train_event == name
        va = valid_event == name
        if not np.any(va):
            continue
        if tr.sum() < audit.MIN_EVENT_ROWS or len(np.unique(groups[tr])) < audit.MIN_EVENT_RUNS:
            continue
        baseline = _fit(x_train[tr], y_yaw_train[tr], groups[tr])
        augmented = _fit(x_train_aug[tr], y_yaw_train[tr], groups[tr])
        base_prediction[va] = baseline.predict(x_valid[va])
        latent_prediction[va] = augmented.predict(x_valid_aug[va])
        event_scores[name] = {
            "rows": int(va.sum()),
            "baseline": _metric(base_prediction[va] - y_yaw_valid[va]),
            "speed_latent": _metric(latent_prediction[va] - y_yaw_valid[va]),
        }
        print(f"yaw event {name}: baseline={event_scores[name]['baseline']} "
              f"speed_latent={event_scores[name]['speed_latent']}", flush=True)

    base_error = base_prediction - y_yaw_valid
    latent_error = latent_prediction - y_yaw_valid
    scored = np.isfinite(base_error) & np.isfinite(latent_error)
    new_run = (validation["run_id"].astype(str)
               == "openplane_yaw_exact_two_domain_gapfill_validation_20261008_r03")
    result = {
        "title": "Causal sensor-only body-speed latent for exact-two yaw model",
        "input_contract": (
            "causal encoder, IMU, actuator feedback/command histories only; "
            "GT speed is a training label/scoring target, never a feature"),
        "target_policy": "next-step GT yaw-rate residual in rad/s",
        "history_lags_ms": list(audit.HISTORY_LAGS),
        "grouped_speed_oof_folds": N_SPLITS,
        "training_rows": int(len(train["residual"])),
        "validation_rows": int(len(validation["residual"])),
        "training_runs": int(len(unique_groups)),
        "validation_runs": int(len(np.unique(validation["run_id"]))),
        "speed_teacher": speed_metrics,
        "yaw_validation": {
            "baseline_event_model": _metric(base_error[scored]),
            "sensor_speed_latent": _metric(latent_error[scored]),
            "paired_run_level": _paired_run_metrics(
                validation["run_id"], base_error, latent_error, scored),
            "new_gapfill_run_baseline": _metric(base_error[scored & new_run]),
            "new_gapfill_run_speed_latent": _metric(latent_error[scored & new_run]),
            "baseline_by_region": _score_regions(validation, base_error, scored),
            "speed_latent_by_region": _score_regions(validation, latent_error, scored),
        },
        "by_causal_event": event_scores,
        "source_corpus_audit": corpus_audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_value) + "\n",
        encoding="utf-8")
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        run_id=validation["run_id"],
        sample_time_ns=validation["sample_time_ns"],
        gt_speed_mps=validation["gt_speed"],
        wheel_speed_mps=validation["wheel_speed"],
        predicted_speed_mps=valid_speed,
        gt_yaw_residual_radps=y_yaw_valid,
        baseline_yaw_residual_radps=base_prediction,
        speed_latent_yaw_residual_radps=latent_prediction,
        response_event=validation["response_event"],
        causal_event=valid_event,
    )
    print("wrote", output_dir / "report.json", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.source_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
