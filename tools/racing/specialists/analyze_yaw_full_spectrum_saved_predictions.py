#!/usr/bin/env python3
"""Summarize a completed full-spectrum validation prediction archive."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "p99_abs_radps": float(np.quantile(absolute, 0.99)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
        "fraction_within_0p1": float(np.mean(absolute <= 0.1)),
    }


def _group_metrics(error: np.ndarray, labels: np.ndarray
                   ) -> dict[str, Any]:
    labels = np.asarray(labels).astype(str)
    result = {}
    for label in sorted(set(labels)):
        result[label] = _metric(error[labels == label])
    return result


def _numeric_strata(rows: list[dict[str, str]], field: str,
                    edges: tuple[float, ...], labels: tuple[str, ...]
                    ) -> np.ndarray:
    values = np.asarray([abs(float(row[field])) for row in rows],
                        dtype=np.float64)
    return np.asarray(labels, dtype=str)[np.digitize(values, edges)]


def _outlier_summary(path: Path, expected_over_0p1: int) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    failing = [row for row in rows
               if float(row["absolute_error_radps"]) > 0.1]
    if len(failing) != expected_over_0p1:
        raise ValueError(
            "the saved top-outlier CSV does not contain every >0.1 sample: "
            f"found {len(failing)}, expected {expected_over_0p1}")
    error = np.asarray([
        float(row["predicted_yaw_residual_radps"])
        - float(row["target_yaw_residual_radps"])
        for row in failing], dtype=np.float64)
    strata: dict[str, np.ndarray] = {
        "phase_event": np.asarray([row["phase_event"] for row in failing]),
        "causal_event": np.asarray([row["causal_event"] for row in failing]),
        "gt_speed_mps": _numeric_strata(
            failing, "gt_speed_mps", (1, 2, 4, 6, 8, 10),
            ("0-1", "1-2", "2-4", "4-6", "6-8", "8-10", "10-12")),
        "abs_steering_rad": _numeric_strata(
            failing, "steering_rad", (.1, .2, .3, .4),
            ("0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5")),
        "event_age_ms": _numeric_strata(
            failing, "event_age_ms", (0, 100, 200, 300, 400),
            ("<0", "0-100", "100-200", "200-300", "300-400", "400-500")),
        "abs_steering_rate_radps": _numeric_strata(
            failing, "steering_rate_radps", (.5, 1, 2, 5),
            ("0-.5", ".5-1", "1-2", "2-5", "5+")),
        "abs_wheel_body_mismatch_mps": _numeric_strata(
            failing, "wheel_body_mismatch_mps", (.1, .3, .6, 1),
            ("0-.1", ".1-.3", ".3-.6", ".6-1", "1+")),
        "abs_imu_roll_rad": _numeric_strata(
            failing, "imu_roll_rad", (.01, .03, .05, .07),
            ("0-.01", ".01-.03", ".03-.05", ".05-.07", ".07+")),
        "abs_imu_roll_rate_rps": _numeric_strata(
            failing, "imu_roll_rate_rps", (.02, .05, .1, .2),
            ("0-.02", ".02-.05", ".05-.1", ".1-.2", ".2+")),
    }
    groups = {name: _group_metrics(error, labels)
              for name, labels in strata.items()}
    ordered = sorted(failing,
                     key=lambda row: float(row["absolute_error_radps"]),
                     reverse=True)
    top = [{
        key: row[key] for key in (
            "condition_id", "phase_event", "causal_event", "gt_speed_mps",
            "steering_rad", "wheel_body_mismatch_mps",
            "steering_rate_radps", "throttle_feedback_rate_per_s",
            "imu_roll_rad", "imu_roll_rate_rps", "event_age_ms",
            "target_yaw_residual_radps", "predicted_yaw_residual_radps",
            "absolute_error_radps")
    } for row in ordered[:30]]
    return {
        "failed_rows_in_outlier_table": int(len(failing)),
        "strata_of_all_samples_over_0p1": groups,
        "largest_30_errors": top,
    }


def analyze(predictions_path: Path, outliers_path: Path,
            output_path: Path) -> dict[str, Any]:
    with np.load(predictions_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    target = data["target"].astype(np.float64)
    speed = data["gt_speed_mps"].astype(np.float64)
    steering = data["steering_rad"].astype(np.float64)
    cell_speed = np.floor(speed / 0.5).astype(np.int64)
    cell_steering = np.clip(np.rint(steering / 0.05), -10, 10).astype(np.int64)
    model_keys = (
        "historical_global", "full_spectrum_yaw_only_global",
        "historical_plus_throttle_global", "all_data_global",
        "causal_event_specialists", "prediction",
    )
    models = {}
    for key in model_keys:
        error = data[key].astype(np.float64) - target
        by_cell = []
        for speed_bin in range(24):
            for steering_bin in range(-10, 11):
                selected = ((cell_speed == speed_bin)
                            & (cell_steering == steering_bin))
                if not selected.any():
                    continue
                by_cell.append({
                    "gt_speed_bin_mps": [speed_bin * 0.5,
                                         (speed_bin + 1) * 0.5],
                    "signed_steering_bin_rad": steering_bin * 0.05,
                    "metrics": _metric(error[selected]),
                })
        supported = [row for row in by_cell
                     if row["metrics"]["samples"] >= 20]
        models[key] = {
            "aggregate": _metric(error),
            "measured_state_cells": int(len(by_cell)),
            "cells_with_at_least_20_samples": int(len(supported)),
            "supported_cells_p95_over_0p1": int(sum(
                row["metrics"]["p95_abs_radps"] > 0.1
                for row in supported)),
            "supported_cells_max_over_0p1": int(sum(
                row["metrics"]["max_abs_radps"] > 0.1
                for row in supported)),
            "by_gt_speed_and_signed_steering": by_cell,
            "by_phase_event": _group_metrics(error, data["phase_event"]),
            "by_causal_event": _group_metrics(error, data["causal_event"]),
        }

    local_failure_count = models["prediction"]["aggregate"]["samples_over_0p1"]
    outlier_summary = _outlier_summary(outliers_path, local_failure_count)
    report = {
        "title": "Saved-prediction analysis: full-spectrum one-step yaw validation",
        "validation_run_id": str(data["run_id"].item()),
        "target": "next-step simulator-GT yaw rate minus current IMU yaw rate (rad/s)",
        "features_use_future_truth": False,
        "sample_count": int(len(target)),
        "state_cell_resolution": {
            "speed_mps": 0.5,
            "signed_steering_rad": 0.05,
            "speed_selector_for_report": "measured GT body speed; offline only",
        },
        "models": models,
        "local_model_errors_over_0p1_analysis": outlier_summary,
        "source_predictions": str(predictions_path),
        "source_outliers": str(outliers_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("outliers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.predictions, args.outliers, args.output)
    print(json.dumps({
        "output": str(args.output),
        "sample_count": result["sample_count"],
        "models": {name: value["aggregate"]
                   for name, value in result["models"].items()},
        "local_model_failed_samples": result[
            "local_model_errors_over_0p1_analysis"][
                "failed_rows_in_outlier_table"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
