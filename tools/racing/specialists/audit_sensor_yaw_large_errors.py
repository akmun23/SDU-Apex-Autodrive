#!/usr/bin/env python3
"""Audit every >0.1 rad/s held-out error from the frozen sensor-only yaw atlas.

Only whole-run validation captures are scored. Sensor streams are model inputs;
simulator truth is used for targets and post-fit error stratification only.
The tool writes a compact report plus one row per large error to a CSV in
live_runs; it does not fit a model or open a simulator.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
               "sensor_only_yaw_regime_atlas_command_intent_v2/"
               "sensor_only_yaw_regime_atlas_report.json")
BUNDLE_PATH = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
               "sensor_only_yaw_regime_atlas_command_intent_v2/"
               "sensor_only_yaw_regime_atlas.joblib")
OUTPUT_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
              "yaw_large_error_audit_v1")
ERROR_THRESHOLD_RADPS = 0.1


def _json_default(value: Any) -> Any:
    """Convert NumPy scalar metadata while rejecting unexpected objects."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _metric(errors: np.ndarray) -> dict[str, Any]:
    errors = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(errors)
    if not len(errors):
        return {"samples": 0, "samples_over_0p1": 0}
    return {
        "samples": int(len(errors)),
        "samples_over_0p1": int(np.count_nonzero(absolute > ERROR_THRESHOLD_RADPS)),
        "rmse_radps": float(np.sqrt(np.mean(errors * errors))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute <= ERROR_THRESHOLD_RADPS)),
    }


def _band(values: np.ndarray, edges: tuple[float, ...], suffix: str
          ) -> np.ndarray:
    labels = []
    for value in values:
        index = int(np.clip(np.searchsorted(edges, value, side="right") - 1,
                            0, len(edges) - 2))
        labels.append(f"{edges[index]:g}_to_{edges[index + 1]:g}{suffix}")
    return np.asarray(labels)


def _group_metrics(labels: np.ndarray, errors: np.ndarray,
                   covered: np.ndarray | None = None
                   ) -> dict[str, Any]:
    result = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if covered is not None:
            mask &= covered
        result[str(label)] = _metric(errors[mask])
    return result


def _predict(rows: dict[str, np.ndarray], bundle: dict[str, Any],
             expected_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_full = rows["x"]
    # v2 predates command-age features. Its input is current/past sensors plus
    # the first eight derived channels; do not feed later age fields to it.
    x = x_full[:, :expected_features]
    if x.shape[1] != expected_features:
        raise ValueError(f"feature width mismatch: {x.shape[1]} != {expected_features}")

    speed_cell = atlas._speed_cell(rows["wheel_speed"])
    steering_cell = atlas._steer_cell(rows["steering"])
    local = np.full(len(x), np.nan, dtype=np.float32)
    global_prediction = np.full(len(x), np.nan, dtype=np.float32)
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, (event, speed, steering) in enumerate(zip(
            rows["event"].astype(str), speed_cell, steering_cell)):
        groups[(event, int(speed), int(steering))].append(index)
    for (event, speed, steering), indices in groups.items():
        index_array = np.asarray(indices, dtype=np.int64)
        global_model = bundle["global_event_models"].get(event)
        if global_model is not None:
            global_prediction[index_array] = global_model.predict(
                x[index_array]).astype(np.float32)
        local_model = bundle["local_expert_models"].get(
            (speed, steering, event))
        if local_model is not None:
            local[index_array] = local_model.predict(x[index_array]).astype(np.float32)
    selected = np.where(np.isfinite(local), local, global_prediction)
    return local, global_prediction, selected


def _serialize_large_errors(rows: dict[str, np.ndarray],
                            errors: np.ndarray, prediction: np.ndarray,
                            local_support: np.ndarray,
                            bundle: dict[str, Any],
                            path: Path) -> list[dict[str, Any]]:
    absolute = np.abs(errors)
    indices = np.flatnonzero(absolute > ERROR_THRESHOLD_RADPS)
    order = indices[np.argsort(absolute[indices])[::-1]]
    columns = (
        "run_id", "event", "gt_speed_for_diagnosis_only_mps",
        "steering_feedback_rad",
        "rear_wheel_mean_mps", "wheel_minus_gt_speed_mps_for_diagnosis_only",
        "throttle_feedback_norm", "throttle_command_norm",
        "steering_command_feedback_gap_rad", "throttle_command_feedback_gap_norm",
        "current_imu_yaw_rate_radps", "current_gt_yaw_rate_for_diagnosis_only",
        "current_imu_gt_yaw_gap_radps_for_diagnosis_only",
        "selected_model_source", "selected_local_expert_train_samples",
        "selected_local_expert_train_runs",
        "next_gt_yaw_rate_radps", "prediction_radps", "signed_error_radps",
    )
    records = []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in order:
            x = rows["x"][index]
            speed_cell = int(atlas._speed_cell(
                rows["wheel_speed"][index:index + 1])[0])
            steering_cell = int(atlas._steer_cell(
                rows["steering"][index:index + 1])[0])
            expert_key = (speed_cell, steering_cell,
                          str(rows["event"][index]))
            support = bundle["local_expert_support"].get(expert_key, {})
            record = {
                "run_id": str(rows["run_id"][index]),
                "event": str(rows["event"][index]),
                "gt_speed_for_diagnosis_only_mps": float(
                    rows["gt_speed"][index]),
                "steering_feedback_rad": float(rows["steering"][index]),
                "rear_wheel_mean_mps": float(rows["wheel_speed"][index]),
                "wheel_minus_gt_speed_mps_for_diagnosis_only": float(
                    rows["wheel_speed"][index] - rows["gt_speed"][index]),
                "throttle_feedback_norm": float(x[1]),
                "throttle_command_norm": float(x[8]),
                "steering_command_feedback_gap_rad": float(x[7] - x[0]),
                "throttle_command_feedback_gap_norm": float(x[8] - x[1]),
                "current_imu_yaw_rate_radps": float(rows["imu_yaw"][index]),
                "current_gt_yaw_rate_for_diagnosis_only": float(
                    rows["gt_yaw_current"][index]),
                "current_imu_gt_yaw_gap_radps_for_diagnosis_only": float(abs(
                    rows["imu_yaw"][index] - rows["gt_yaw_current"][index])),
                "selected_model_source": (
                    "local_expert" if local_support[index]
                    else "global_event_fallback"),
                "selected_local_expert_train_samples": int(
                    support.get("training_samples", 0)),
                "selected_local_expert_train_runs": int(
                    support.get("training_runs", 0)),
                "next_gt_yaw_rate_radps": float(
                    rows["imu_yaw"][index] + rows["residual"][index]),
                "prediction_radps": float(rows["imu_yaw"][index] + prediction[index]),
                "signed_error_radps": float(errors[index]),
            }
            writer.writerow(record)
            if len(records) < 100:
                records.append(record)
    return records


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    bundle = joblib.load(BUNDLE_PATH)
    contract = report["input_contract"]
    history_lags = tuple(int(value // 25)
                         for value in contract["sensor_history_lags_ms"])
    event_definition = str(contract["event_definition"])
    expected_features = len(contract["features"])

    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    validation_series = [row for row in admitted if row.split == "validation"]
    rows_by_run = []
    for series in validation_series:
        sensor_run = atlas._read_sensor_run(series, history_lags)
        rows_by_run.append(atlas._rows(sensor_run, event_definition, history_lags))
    if not rows_by_run:
        raise RuntimeError("no whole-run validation data admitted")
    rows = {key: np.concatenate([part[key] for part in rows_by_run], axis=0)
            for key in rows_by_run[0]}
    local, global_prediction, selected = _predict(rows, bundle, expected_features)
    local_support = np.isfinite(local)
    target_residual = rows["residual"]
    local_error = local - target_residual
    global_error = global_prediction - target_residual
    selected_error = selected - target_residual
    selected_valid = np.isfinite(selected_error)
    if not np.all(selected_valid):
        raise RuntimeError("the atlas has no local or global expert for some rows")

    x = rows["x"]
    wheel_minus_gt = rows["wheel_speed"] - rows["gt_speed"]
    imu_gt_gap = np.abs(rows["imu_yaw"] - rows["gt_yaw_current"])
    steering_abs = np.abs(rows["steering"])
    steering_gap = x[:, 7] - x[:, 0]
    throttle_gap = x[:, 8] - x[:, 1]
    group_labels = {
        "event": rows["event"].astype(str),
        "gt_speed_band_diagnostic_only": _band(
            rows["gt_speed"], (0, 1, 2, 3, 4, 6, 8, 10, 12), "mps"),
        "absolute_steering_band": _band(
            steering_abs, (0, 0.1, 0.2, 0.35, 0.525), "rad"),
        "wheel_minus_gt_speed_diagnostic_only": _band(
            wheel_minus_gt, (-20, -1, -0.5, 0.5, 1, 20), "mps"),
        "current_imu_vs_gt_phase_diagnostic_only": _band(
            imu_gt_gap, (0, 0.05, 0.1, 10), "radps"),
        "absolute_steering_command_feedback_gap": _band(
            np.abs(steering_gap), (0, 0.02, 0.05, 0.1, 0.2, 1), "rad"),
        "absolute_throttle_command_feedback_gap": _band(
            np.abs(throttle_gap), (0, 0.02, 0.05, 0.1, 0.2, 1), "norm"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_path = output_dir / "selected_atlas_errors_over_0p1.csv"
    outlier_mask = np.abs(selected_error) > ERROR_THRESHOLD_RADPS
    top100 = _serialize_large_errors(rows, selected_error, selected,
                                     local_support, bundle, outlier_path)
    large_errors_by_source = {
        "local_expert": int(np.count_nonzero(outlier_mask & local_support)),
        "global_event_fallback": int(np.count_nonzero(
            outlier_mask & ~local_support)),
    }
    large_errors_by_event = {}
    for event in sorted(set(rows["event"].astype(str))):
        mask = outlier_mask & (rows["event"].astype(str) == event)
        large_errors_by_event[event] = {
            "samples_over_0p1": int(np.count_nonzero(mask)),
            "local_expert": int(np.count_nonzero(mask & local_support)),
            "global_event_fallback": int(np.count_nonzero(mask & ~local_support)),
        }

    by_run = {}
    for run_id in sorted(set(rows["run_id"].astype(str))):
        mask = rows["run_id"].astype(str) == run_id
        by_run[run_id] = _metric(selected_error[mask])
    by_event = {}
    for event in sorted(set(rows["event"].astype(str))):
        mask = rows["event"].astype(str) == event
        by_event[event] = {
            "selected_local_or_global_fallback": _metric(selected_error[mask]),
            "local_support_fraction": float(np.mean(local_support[mask])),
            "local_expert_on_support": _metric(local_error[mask & local_support]),
        }
    speed_cells = atlas._speed_cell(rows["wheel_speed"])
    steering_cells = atlas._steer_cell(rows["steering"])
    observable_cells = []
    for event, speed_cell, steering_cell in sorted(set(zip(
            rows["event"].astype(str).tolist(), speed_cells.tolist(),
            steering_cells.tolist()))):
        mask = ((rows["event"].astype(str) == event)
                & (speed_cells == speed_cell)
                & (steering_cells == steering_cell))
        cell_errors = selected_error[mask]
        n_large = int(np.count_nonzero(
            np.abs(cell_errors) > ERROR_THRESHOLD_RADPS))
        if np.count_nonzero(mask) < 20 and n_large == 0:
            continue
        support = bundle["local_expert_support"].get(
            (int(speed_cell), int(steering_cell), event), {})
        observable_cells.append({
            "event": event,
            "wheel_speed_cell_center_mps": float(
                atlas.SPEED_CENTERS[int(speed_cell)]),
            "steering_cell_center_rad": float(
                atlas.STEERING_CENTERS[int(steering_cell)]),
            "validation_rows": int(np.count_nonzero(mask)),
            "local_expert_rows": int(np.count_nonzero(mask & local_support)),
            "global_fallback_rows": int(np.count_nonzero(mask & ~local_support)),
            "errors_over_0p1": n_large,
            "training_samples_for_exact_expert": int(
                support.get("training_samples", 0)),
            "training_runs_for_exact_expert": int(
                support.get("training_runs", 0)),
            "selected_model_metrics": _metric(cell_errors),
        })
    groups = {
        name: _group_metrics(labels, selected_error)
        for name, labels in group_labels.items()
    }
    report_out = {
        "title": "Complete >0.1 rad/s validation error audit for sensor-only yaw atlas",
        "threshold_radps": ERROR_THRESHOLD_RADPS,
        "input_policy": "causal sensor features only; no simulator truth as feature or selector",
        "truth_policy": "simulator truth is the next-yaw target and diagnostic stratifier only",
        "selection_policy": "local speed x signed-steering x event model; global event fallback",
        "validation_runs": sorted(set(rows["run_id"].astype(str))),
        "validation_samples": int(len(selected_error)),
        "local_expert_coverage_fraction": float(np.mean(local_support)),
        "all_selected_model": _metric(selected_error),
        "local_expert_only": _metric(local_error[local_support]),
        "global_event_fallback_all_rows": _metric(global_error),
        "by_run": by_run,
        "by_event": by_event,
        "by_observable_speed_steering_event_cell": observable_cells,
        "by_measured_and_diagnostic_regime": groups,
        "large_error_rows": int(np.count_nonzero(
            np.abs(selected_error) > ERROR_THRESHOLD_RADPS)),
        "large_errors_by_model_source": large_errors_by_source,
        "large_errors_by_event_and_model_source": large_errors_by_event,
        "large_errors_csv": str(outlier_path),
        "largest_100_rows": top100,
        "source_audit": source_audit,
        "notes": [
            "Rows are not statistically independent: 40-Hz samples within one reset/event share history.",
            "GT speed, current GT yaw, and wheel-minus-GT speed appear only in post-fit diagnostic groups.",
            "CSV contains every selected-model sample whose absolute error exceeds 0.1 rad/s.",
            "This is a held-out one-step audit, not MPC rollout or full-lap accuracy.",
        ],
    }
    report_path = output_dir / "sensor_yaw_large_error_audit.json"
    report_path.write_text(json.dumps(report_out, indent=2, sort_keys=True,
                                      default=_json_default) + "\n",
                           encoding="utf-8")
    print(f"validation: {len(rows['residual'])} transitions / "
          f"{len(report_out['validation_runs'])} whole runs")
    print(f"local coverage: {np.mean(local_support):.3%}")
    print(f"selected >0.1 errors: {report_out['large_error_rows']}")
    print(f"selected RMSE={_metric(selected_error)['rmse_radps']:.5f}, "
          f"p95={_metric(selected_error)['p95_abs_radps']:.5f}, "
          f"max={_metric(selected_error)['max_abs_radps']:.5f} rad/s")
    print(f"wrote {report_path} and {outlier_path}")
    return report_out


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
