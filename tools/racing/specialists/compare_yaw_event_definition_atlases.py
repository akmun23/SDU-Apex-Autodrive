#!/usr/bin/env python3
"""Paired whole-run comparison of the frozen feedback-delta and command-intent atlases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import audit_sensor_yaw_large_errors as audit
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import audit_sensor_yaw_large_errors as audit
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT / "live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_v1"
V2 = ROOT / "live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_command_intent_v2"
OUTPUT = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/event_definition_comparison.json"


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


def _macro_deltas(error_a: np.ndarray, error_b: np.ndarray,
                  run_ids: np.ndarray) -> dict[str, Any]:
    deltas = []
    for run_id in sorted(set(run_ids.astype(str))):
        mask = run_ids.astype(str) == run_id
        deltas.append(float(
            np.sqrt(np.mean(error_a[mask] ** 2))
            - np.sqrt(np.mean(error_b[mask] ** 2))))
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(20261008)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return {
        "runs": len(values),
        "mean_run_rmse_delta_feedback_delta_minus_command_intent_radps": float(values.mean()),
        "median_run_rmse_delta_radps": float(np.median(values)),
        "runs_feedback_delta_better": int(np.count_nonzero(values < 0.0)),
        "runs_feedback_delta_worse": int(np.count_nonzero(values > 0.0)),
        "paired_run_bootstrap_95pct_ci_radps": [
            float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def _score(rows: dict[str, np.ndarray], bundle: dict[str, Any],
           report_path: Path) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    residual_prediction = audit._predict(
        rows, bundle, len(report["input_contract"]["features"]))[2]
    error = residual_prediction - rows["residual"]
    if not np.isfinite(error).all():
        raise RuntimeError("one event atlas left an unscored validation row")
    return residual_prediction, error


def run() -> dict[str, Any]:
    v1_report = json.loads((V1 / "sensor_only_yaw_regime_atlas_report.json")
                           .read_text(encoding="utf-8"))
    v2_report = json.loads((V2 / "sensor_only_yaw_regime_atlas_report.json")
                           .read_text(encoding="utf-8"))
    v1_bundle = joblib.load(V1 / "sensor_only_yaw_regime_atlas.joblib")
    v2_bundle = joblib.load(V2 / "sensor_only_yaw_regime_atlas.joblib")
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    validation = [series for series in admitted if series.split == "validation"]
    rows_v1, rows_v2 = [], []
    for series in validation:
        sensor_run = atlas._read_sensor_run(series)
        rows_v1.append(atlas._rows(sensor_run, "feedback_delta"))
        rows_v2.append(atlas._rows(sensor_run, "command_intent"))
    joined: list[dict[str, np.ndarray]] = []
    for parts in (rows_v1, rows_v2):
        joined.append({key: np.concatenate([part[key] for part in parts], axis=0)
                       for key in parts[0]})
    a, b = joined
    if (len(a["residual"]) != len(b["residual"])
            or not np.array_equal(a["run_id"], b["run_id"])
            or not np.allclose(a["residual"], b["residual"], rtol=0, atol=0)):
        raise RuntimeError("event variants do not align on identical GT targets")

    pred_v1, error_v1 = _score(a, v1_bundle,
                               V1 / "sensor_only_yaw_regime_atlas_report.json")
    pred_v2, error_v2 = _score(b, v2_bundle,
                               V2 / "sensor_only_yaw_regime_atlas_report.json")
    events = ("hold", "turn_in", "unwind", "reversal")
    by_event = {}
    for event in events:
        mask_v1 = a["event"] == event
        mask_v2 = b["event"] == event
        by_event[event] = {
            "feedback_delta_atlas_on_its_event_rows": _metric(error_v1[mask_v1]),
            "command_intent_atlas_on_its_event_rows": _metric(error_v2[mask_v2]),
            "feedback_delta_run_macro": _macro_deltas(
                error_v1[mask_v1], error_v2[mask_v1], a["run_id"][mask_v1]),
            "command_intent_run_macro": _macro_deltas(
                error_v1[mask_v2], error_v2[mask_v2], b["run_id"][mask_v2]),
        }
    confusion = {left: dict(Counter(b["event"][a["event"] == left].tolist()))
                 for left in events}
    result = {
        "title": "Paired whole-run event-definition atlas comparison",
        "supervision": "next-tick simulator-GT yaw rate; production odometry is never a label",
        "runtime_input_policy": "sensor/command history only; no GT or bridge timing selector",
        "validation_runs": sorted(set(a["run_id"].astype(str))),
        "validation_rows": int(len(error_v1)),
        "feedback_delta_atlas": _metric(error_v1),
        "command_intent_atlas": _metric(error_v2),
        "paired_run_macro": _macro_deltas(error_v1, error_v2, a["run_id"]),
        "event_label_cross_tab_feedback_rows_to_command_columns": confusion,
        "by_event_definition": by_event,
        "source_audit": source_audit,
        "caveats": [
            "The two event definitions partition transitions differently; event-conditioned metrics are reported on each definition's own rows.",
            "Run-level paired comparison is on the same validation runs and same GT targets.",
            "This is a one-step yaw-rate comparison, not recursive Odom or MPC validation.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 default=audit._json_default) + "\n",
                      encoding="utf-8")
    print("feedback_delta", result["feedback_delta_atlas"])
    print("command_intent", result["command_intent_atlas"])
    print("paired run macro", result["paired_run_macro"])
    print("wrote", OUTPUT)
    return result


if __name__ == "__main__":
    run()
