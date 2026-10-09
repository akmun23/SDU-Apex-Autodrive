#!/usr/bin/env python3
"""Stratify saved sensor-slip observer predictions by held-out speed/steering."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import analyze_yaw_slip_regime_factor as slip_diagnostic
    import evaluate_yaw_full_spectrum_exact_two as evaluator
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import analyze_yaw_slip_regime_factor as slip_diagnostic
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/"
    "sensor_slip_observer_r02/heldout_rows.csv")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/"
    "sensor_slip_observer_r02/slip_error_by_speed_steering.json")
SPEED_EDGES = np.asarray((0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0))
STEER_EDGES = np.asarray((0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.501))
SLIP_NAMES = slip_diagnostic.SLIP_NAMES


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    if not len(error):
        return {"samples": 0, "rmse": None, "mae": None, "p95_abs": None}
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(absolute)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
    }


def _load_prediction_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"prediction file has no rows: {path}")
    return rows


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    prediction_rows = _load_prediction_csv(input_path)
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    by_id = {series.run_id: series for series in admitted}
    run_id = evaluator.FULL_VALIDATION_ID
    if run_id not in by_id or by_id[run_id].split != "validation":
        raise ValueError("closed r02 is not admitted as validation data")
    data, audit = evaluator._collect_run(by_id[run_id], atlas.HISTORY_LAGS)
    if len(prediction_rows) != len(data["y"]):
        raise ValueError("saved predictions do not align with exact-two r02 rows")
    csv_condition = np.asarray([row["condition_id"] for row in prediction_rows])
    if not np.array_equal(csv_condition, data["condition_id"].astype(str)):
        raise ValueError("saved prediction row order/provenance differs from r02")

    truth_slip, slip_audit = slip_diagnostic._slip_features(data)
    predicted_slip = np.column_stack([
        np.asarray([float(row[f"estimated_{name}"]) for row in prediction_rows])
        for name in SLIP_NAMES]).astype(np.float32)
    truth_values = truth_slip[:, :len(SLIP_NAMES)]
    valid = truth_slip[:, len(SLIP_NAMES):] > 0.5
    slip_error = predicted_slip - truth_values
    baseline_error = np.asarray([
        float(row["baseline_yaw_error_radps"]) for row in prediction_rows])
    observer_error = np.asarray([
        float(row["estimated_slip_yaw_error_radps"]) for row in prediction_rows])
    estimated_front_slip = np.max(np.abs(predicted_slip[:, 2:4]), axis=1)
    gate = estimated_front_slip >= 0.01
    gated_yaw_error = np.where(gate, observer_error, baseline_error)
    speed = np.asarray(data["gt_speed"], dtype=np.float64)
    steer = np.abs(np.asarray(data["steering"], dtype=np.float64))

    def describe(mask: np.ndarray) -> dict[str, Any]:
        front_valid = np.all(valid[:, 2:4], axis=1)
        rear_valid = np.all(valid[:, :2], axis=1)
        front_err = slip_error[:, 2:4][mask & front_valid].reshape(-1)
        rear_err = slip_error[:, :2][mask & rear_valid].reshape(-1)
        return {
            "rows": int(np.count_nonzero(mask)),
            "front_lateral_slip_error": _metric(front_err),
            "rear_longitudinal_slip_error": _metric(rear_err),
            "yaw_baseline": _metric(baseline_error[mask]),
            "yaw_sensor_slip": _metric(observer_error[mask]),
            "yaw_front_slip_gated": _metric(gated_yaw_error[mask]),
            "front_slip_bands": {
                "below_peak": int(np.count_nonzero(
                    mask & front_valid
                    & (np.max(np.abs(truth_values[:, 2:4]), axis=1) < 0.01))),
                "peak_to_asymptote": int(np.count_nonzero(
                    mask & front_valid
                    & (np.max(np.abs(truth_values[:, 2:4]), axis=1) >= 0.01)
                    & (np.max(np.abs(truth_values[:, 2:4]), axis=1) < 0.10))),
                "at_or_beyond_asymptote": int(np.count_nonzero(
                    mask & front_valid
                    & (np.max(np.abs(truth_values[:, 2:4]), axis=1) >= 0.10))),
            },
            "rear_longitudinal_at_or_beyond_asymptote": int(np.count_nonzero(
                mask & rear_valid
                & (np.mean(np.abs(truth_values[:, :2]), axis=1) >= 0.25))),
        }

    by_speed: dict[str, Any] = {}
    for low, high in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        mask = (speed >= low) & (speed < high)
        by_speed[f"{low:g}-{high:g}mps"] = describe(mask)
    by_abs_steer: dict[str, Any] = {}
    for low, high in zip(STEER_EDGES[:-1], STEER_EDGES[1:]):
        mask = (steer >= low) & (steer < high)
        by_abs_steer[f"{low:g}-{high:g}rad"] = describe(mask)
    by_speed_steer: dict[str, Any] = {}
    for speed_low, speed_high in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:]):
        for steer_low, steer_high in zip(STEER_EDGES[:-1], STEER_EDGES[1:]):
            mask = ((speed >= speed_low) & (speed < speed_high)
                    & (steer >= steer_low) & (steer < steer_high))
            if np.count_nonzero(mask) < 80:
                continue
            label = (f"{speed_low:g}-{speed_high:g}mps__"
                     f"{steer_low:g}-{steer_high:g}rad")
            by_speed_steer[label] = describe(mask)

    report = {
        "run_id": run_id,
        "input_csv": str(input_path.relative_to(ROOT)),
        "timebase_ms": 25,
        "admission": "closed r02 exact-two contiguous steering windows only",
        "speed_bin_edges_mps": SPEED_EDGES.tolist(),
        "absolute_steering_bin_edges_rad": STEER_EDGES.tolist(),
        "front_slip_gate": "max(abs(predicted front Sy)) >= 0.01",
        "row_count": int(len(data["y"])),
        "run_audit": audit,
        "slip_reconstruction_audit": slip_audit,
        "by_speed": by_speed,
        "by_absolute_steering": by_abs_steer,
        "speed_steering_cells_n_ge_80": by_speed_steer,
        "source_admission_audit": source_audit,
        "notes": [
            "Slip values come from the documented per-wheel kinematic equations and are proxies, not direct force labels.",
            "GT is used to calculate held-out slip truth and regime bins only; observer predictions use saved causal sensor-only outputs.",
            "The active midpoint r04 bag is not read by this report.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        report, indent=2, sort_keys=True,
        default=slip_diagnostic._json_default) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.input.resolve(), args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "rows": report["row_count"],
        "by_speed": {
            key: value["front_lateral_slip_error"]
            for key, value in report["by_speed"].items()},
        "by_abs_steer": {
            key: value["front_lateral_slip_error"]
            for key, value in report["by_absolute_steering"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
