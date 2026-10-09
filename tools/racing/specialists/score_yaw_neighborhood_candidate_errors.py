#!/usr/bin/env python3
"""Export all held-out >0.1-rad/s residuals from the frozen neighborhood atlas."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
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
BASE = ROOT / "live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_command_intent_v2"
CANDIDATE = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported"
OUTPUT = CANDIDATE / "candidate_remaining_errors_over_0p1.csv"
REPORT_OUTPUT = CANDIDATE / "remaining_error_summary.json"
THRESHOLD = 0.1


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > THRESHOLD)),
        "fraction_within_0p1": float(np.mean(absolute <= THRESHOLD)),
    }


def run() -> dict[str, Any]:
    model_report = json.loads((BASE / "sensor_only_yaw_regime_atlas_report.json")
                              .read_text(encoding="utf-8"))
    frozen_atlas = joblib.load(BASE / "sensor_only_yaw_regime_atlas.joblib")
    candidate = joblib.load(CANDIDATE / "candidate.joblib")
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    validation = [row for row in admitted if row.split == "validation"]
    parts = [atlas._rows(atlas._read_sensor_run(series), "command_intent")
             for series in validation]
    rows = {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]}
    width = len(model_report["input_contract"]["features"])
    baseline_delta = audit._predict(rows, frozen_atlas, width)[2]
    prediction_delta = baseline_delta.copy()
    speed = atlas._speed_cell(rows["wheel_speed"])
    steering = atlas._steer_cell(rows["steering"])
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for i, (event, speed_cell, steer_cell) in enumerate(zip(
            rows["event"].astype(str), speed, steering)):
        groups[(event, int(speed_cell), int(steer_cell))].append(i)
    for key, model in candidate["models"].items():
        indices = np.asarray(groups.get(key, ()), dtype=np.int64)
        if len(indices):
            prediction_delta[indices] = model.predict(
                rows["x"][indices, :width]).astype(np.float32)
    error = prediction_delta - rows["residual"]
    if not np.isfinite(error).all():
        raise RuntimeError("candidate left non-finite validation errors")
    absolute = np.abs(error)
    indices = np.flatnonzero(absolute > THRESHOLD)
    indices = indices[np.argsort(absolute[indices])[::-1]]
    support = candidate.get("support", {})
    columns = (
        "validation_row_index", "run_id", "event",
        "selected_model_source", "neighbor_train_samples", "neighbor_train_runs",
        "gt_speed_for_diagnosis_only_mps", "rear_wheel_mean_mps",
        "wheel_minus_gt_speed_mps_for_diagnosis_only",
        "steering_feedback_rad", "steering_command_feedback_gap_rad",
        "throttle_feedback_norm", "throttle_command_norm",
        "throttle_command_feedback_gap_norm", "imu_yaw_rate_radps",
        "current_gt_yaw_rate_for_diagnosis_only",
        "current_imu_gt_yaw_gap_radps_for_diagnosis_only",
        "next_gt_yaw_rate_radps", "predicted_next_yaw_rate_radps",
        "signed_error_radps",
    )
    by_event: dict[str, dict[str, Any]] = {}
    for event in sorted(set(rows["event"].astype(str))):
        mask = rows["event"].astype(str) == event
        by_event[event] = _metric(error[mask])
    source_counts = Counter()
    cell_counts: Counter[tuple[str, int, int, str]] = Counter()
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in indices:
            event = str(rows["event"][index])
            speed_cell, steer_cell = int(speed[index]), int(steering[index])
            key = (event, speed_cell, steer_cell)
            if key in candidate["models"]:
                source = "neighbor_expert"
                support_row = support.get(f"{event}:{speed_cell}:{steer_cell}", {})
            elif (speed_cell, steer_cell, event) in frozen_atlas["local_expert_models"]:
                source = "frozen_exact_expert"
                support_row = frozen_atlas["local_expert_support"].get(
                    (speed_cell, steer_cell, event), {})
            else:
                source = "global_event_fallback"
                support_row = {}
            source_counts[source] += 1
            cell_counts[(event, speed_cell, steer_cell, source)] += 1
            x = rows["x"][index]
            writer.writerow({
                "validation_row_index": int(index),
                "run_id": str(rows["run_id"][index]),
                "event": event,
                "selected_model_source": source,
                "neighbor_train_samples": int(support_row.get("training_samples", 0)),
                "neighbor_train_runs": int(support_row.get("training_runs", 0)),
                "gt_speed_for_diagnosis_only_mps": float(rows["gt_speed"][index]),
                "rear_wheel_mean_mps": float(rows["wheel_speed"][index]),
                "wheel_minus_gt_speed_mps_for_diagnosis_only": float(
                    rows["wheel_speed"][index] - rows["gt_speed"][index]),
                "steering_feedback_rad": float(rows["steering"][index]),
                "steering_command_feedback_gap_rad": float(x[7] - x[0]),
                "throttle_feedback_norm": float(x[1]),
                "throttle_command_norm": float(x[8]),
                "throttle_command_feedback_gap_norm": float(x[8] - x[1]),
                "imu_yaw_rate_radps": float(rows["imu_yaw"][index]),
                "current_gt_yaw_rate_for_diagnosis_only": float(
                    rows["gt_yaw_current"][index]),
                "current_imu_gt_yaw_gap_radps_for_diagnosis_only": float(abs(
                    rows["imu_yaw"][index] - rows["gt_yaw_current"][index])),
                "next_gt_yaw_rate_radps": float(
                    rows["imu_yaw"][index] + rows["residual"][index]),
                "predicted_next_yaw_rate_radps": float(
                    rows["imu_yaw"][index] + prediction_delta[index]),
                "signed_error_radps": float(error[index]),
            })

    result = {
        "title": "All remaining >0.1 rad/s errors after neighborhood candidate",
        "supervision": "simulator-GT next yaw rate; diagnostic truth fields are not model inputs",
        "validation_rows": int(len(error)),
        "validation_runs": sorted(set(rows["run_id"].astype(str))),
        "candidate_metrics": _metric(error),
        "remaining_large_error_rows": int(len(indices)),
        "large_errors_by_event": {
            event: metric["samples_over_0p1"] for event, metric in by_event.items()},
        "large_errors_by_model_source": dict(source_counts),
        "by_event": by_event,
        "top_100_cells": [
            {"event": key[0], "wheel_speed_cell": key[1],
             "steering_cell": key[2], "model_source": key[3], "outliers": n}
            for key, n in cell_counts.most_common(100)],
        "outliers_csv": str(OUTPUT),
        "source_audit": source_audit,
    }
    REPORT_OUTPUT.write_text(json.dumps(
        result, indent=2, sort_keys=True, default=audit._json_default) + "\n",
        encoding="utf-8")
    print("candidate", result["candidate_metrics"])
    print("outliers by event", result["large_errors_by_event"])
    print("outliers by model", result["large_errors_by_model_source"])
    print("wrote", OUTPUT, REPORT_OUTPUT)
    return result


if __name__ == "__main__":
    run()
