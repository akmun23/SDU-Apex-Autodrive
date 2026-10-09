#!/usr/bin/env python3
"""Evaluate existing high-speed yaw captures as a regime-specific expert.

This is an offline experiment only. It uses exact-two response windows,
sensor/actuator history as inputs, and simulator yaw rate only as target. The
high-speed specialist is scored on its independent scheduled capture and on
separate full-spectrum validation captures; no runtime model is changed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import evaluate_yaw_full_spectrum_exact_two as evaluator
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/"
    "continued_error_reduction_20261009/highspeed_regional_surface_transfer")
HISTORY_LAGS = (0, 1, 2, 4)
REGIONAL_MODEL = {
    "max_iter": 240,
    "learning_rate": 0.045,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 45,
    "l2_regularization": 1.0,
    "random_state": 20261009,
}
RUNS = {
    "grid_train_r01": (
        "openplane_yaw_full_spectrum_grid_train_20261009_r01", "train",
        "live_runs/derived_yaw_full_spectrum_grid_dataset_20261009/"
        "openplane_dynamics.npz"),
    "grid_train_r03": (
        "openplane_yaw_full_spectrum_grid_train_20261009_r03", "train",
        "live_runs/derived_yaw_full_spectrum_grid_train_20261009_r03/"
        "openplane_dynamics.npz"),
    "grid_validation_r02": (
        "openplane_yaw_full_spectrum_grid_validation_20261009_r02",
        "validation",
        "live_runs/derived_yaw_full_spectrum_grid_dataset_20261009/"
        "openplane_dynamics.npz"),
    "midpoint_validation_r04": (
        "openplane_yaw_full_spectrum_midpoint_validation_20261009_r04",
        "validation",
        "live_runs/racing_model_diagnostics_20261009/yaw_midpoint_r04_dataset/"
        "openplane_dynamics.npz"),
    "highspeed_train_r01": (
        "openplane_yaw_error_highspeed_steering_train_r01_20261007", "train",
        "live_runs/racing_model_diagnostics_20261007/"
        "yaw_error_highspeed_steering_train_r01_dataset/openplane_dynamics.npz"),
    "highspeed_train_r02": (
        "openplane_yaw_error_highspeed_steering_train_r02_20261007", "train",
        "live_runs/racing_model_diagnostics_20261007/"
        "yaw_error_highspeed_steering_train_r02_dataset/openplane_dynamics.npz"),
    "highspeed_validation_r03": (
        "openplane_yaw_error_highspeed_steering_validation_r03_20261007",
        "validation",
        "live_runs/racing_model_diagnostics_20261007/"
        "yaw_error_highspeed_steering_validation_r03_dataset/"
        "openplane_dynamics.npz"),
}


def _validate_provenance(run_id: str, split: str, source: str) -> None:
    manifest_path = ROOT / source
    manifest_path = manifest_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matches = [row for row in manifest["runs"]
               if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"{run_id}: expected exactly one manifest row")
    row = matches[0]
    if (row.get("effective_split") != split
            or row.get("aborted") is not False
            or row.get("reason") != "schedule complete"
            or not row.get("clean_stream_and_collision_gate")
            or row.get("quality_failures")
            or row.get("whole_bag_quality_failures")
            or int(row.get("timing_faults", -1)) != 0
            or any(int(value) != 0 for value in row.get("collisions", []))):
        raise ValueError(f"{run_id}: provenance/quality gate failed")


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "p95_abs_radps": float(np.quantile(absolute, 0.95)) if len(error) else None,
        "max_abs_radps": float(np.max(absolute)) if len(error) else None,
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
    }


def _score(model: Any, rows: dict[str, np.ndarray],
           mask: np.ndarray | None = None) -> tuple[dict[str, Any], np.ndarray]:
    if mask is None:
        mask = np.ones(len(rows["y"]), dtype=bool)
    selected = {key: value[mask] for key, value in rows.items()
                if isinstance(value, np.ndarray)}
    prediction = model.predict(evaluator._model_features(selected))
    return _metric(prediction - selected["y"]), prediction


def _gates(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    speed = rows["wheel_speed"]
    steering = np.abs(rows["steering"])
    unwind = rows["phase_event"].astype(str) == "unwind"
    return {
        "8_12mps_all_events_abs_steering_le_0p25": (
            (speed >= 8.0) & (speed < 12.0) & (steering < 0.25)),
        "8_12mps_unwind_abs_steering_lt_0p10": (
            unwind & (speed >= 8.0) & (speed < 12.0) & (steering < 0.10)),
        "8_12mps_unwind_abs_steering_0p10_0p25": (
            unwind & (speed >= 8.0) & (speed < 12.0)
            & (steering >= 0.10) & (steering < 0.25)),
    }


def _regional_gate(rows: dict[str, np.ndarray]) -> np.ndarray:
    return ((rows["wheel_speed"] >= 8.0)
            & (rows["wheel_speed"] < 12.0)
            & (np.abs(rows["steering"]) < 0.25))


def _fit_regional_surface(rows: dict[str, np.ndarray]) -> Any:
    """Fit a regime-local boosted tree model with left/right symmetry."""
    x = evaluator._model_features(rows)
    sensor = x[:, :-len(evaluator.EVENTS)]
    event_code = x[:, -len(evaluator.EVENTS):]
    mirror_input = np.column_stack((
        sensor, np.zeros((len(sensor), 4), dtype=np.float32)))
    mirrored = evaluator.causal._mirror_features(mirror_input)[:, :-4]
    mirrored = np.column_stack((mirrored, event_code))
    model = HistGradientBoostingRegressor(**REGIONAL_MODEL)
    model.fit(np.concatenate((x, mirrored)),
              np.concatenate((rows["y"], -rows["y"])))
    return model


def _lateral_accel_residual(history_features: np.ndarray) -> np.ndarray:
    """Return ay - r*u_wheel at each 25-ms input-history lag."""
    observation_width = len(evaluator.atlas.OBSERVATION_NAMES)
    history_width = len(HISTORY_LAGS) * observation_width
    if history_features.shape[1] < history_width:
        raise ValueError("yaw feature matrix is shorter than sensor history")
    residuals = []
    for lag_index in range(len(HISTORY_LAGS)):
        start = lag_index * observation_width
        left = history_features[:, start + 2]
        right = history_features[:, start + 3]
        lateral_accel = history_features[:, start + 5]
        yaw_rate = history_features[:, start + 6]
        residuals.append(lateral_accel - yaw_rate * 0.5 * (left + right))
    return np.column_stack(residuals).astype(np.float32)


class _LateralAccelFeatureModel:
    """Estimator wrapper preserving the base evaluator's predict contract."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def predict(self, history_features: np.ndarray) -> np.ndarray:
        return self.estimator.predict(np.column_stack((
            history_features, _lateral_accel_residual(history_features))))


def _fit_regional_surface_with_lateral_accel(
        rows: dict[str, np.ndarray]) -> _LateralAccelFeatureModel:
    x = evaluator._model_features(rows)
    sensor = x[:, :-len(evaluator.EVENTS)]
    event_code = x[:, -len(evaluator.EVENTS):]
    q = _lateral_accel_residual(x)
    mirror_input = np.column_stack((
        sensor, np.zeros((len(sensor), 4), dtype=np.float32)))
    mirrored_sensor = evaluator.causal._mirror_features(mirror_input)[:, :-4]
    mirrored_x = np.column_stack((
        mirrored_sensor, -q, event_code))
    model = HistGradientBoostingRegressor(**REGIONAL_MODEL)
    model.fit(np.concatenate((np.column_stack((x, q)), mirrored_x)),
              np.concatenate((rows["y"], -rows["y"])))
    return _LateralAccelFeatureModel(model)


def run(output: Path = OUTPUT,
        history_lags: tuple[int, ...] = HISTORY_LAGS) -> dict[str, Any]:
    data: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, Any] = {}
    for key, (run_id, split, source) in RUNS.items():
        _validate_provenance(run_id, split, source)
        series = SimpleNamespace(run_id=run_id, split=split, source=source)
        rows, audit = evaluator._collect_run(series, history_lags)
        data[key] = rows
        audits[key] = {
            "run_id": run_id,
            "split": split,
            "source": source,
            "exact_two_audit": audit["exact_two_audit"],
            "rows": int(len(rows["y"])),
        }

    broad_train = evaluator._combine({
        "grid_train_r01": data["grid_train_r01"],
        "grid_train_r03": data["grid_train_r03"],
    })
    highspeed_train = evaluator._combine({
        "highspeed_train_r01": data["highspeed_train_r01"],
        "highspeed_train_r02": data["highspeed_train_r02"],
    })
    pooled_train = evaluator._combine({
        "grid_train_r01": data["grid_train_r01"],
        "grid_train_r03": data["grid_train_r03"],
        "highspeed_train_r01": data["highspeed_train_r01"],
        "highspeed_train_r02": data["highspeed_train_r02"],
    })
    broad = evaluator._fit(
        evaluator._model_features(broad_train), broad_train["y"],
        broad_train["phase_id"])
    specialist = evaluator._fit(
        evaluator._model_features(highspeed_train), highspeed_train["y"],
        highspeed_train["phase_id"])
    pooled = evaluator._fit(
        evaluator._model_features(pooled_train), pooled_train["y"],
        pooled_train["phase_id"])

    broad_folds: dict[str, Any] = {}
    lateral_accel_folds: dict[str, Any] = {}
    for fit_key, held_key in (("grid_train_r01", "grid_train_r03"),
                              ("grid_train_r03", "grid_train_r01")):
        fit_rows = data[fit_key]
        held_rows = data[held_key]
        fit_mask = _regional_gate(fit_rows)
        held_mask = _regional_gate(held_rows)
        local = {key: value[fit_mask] for key, value in fit_rows.items()
                 if isinstance(value, np.ndarray)}
        regional_model = _fit_regional_surface(local)
        broad_model = evaluator._fit(
            evaluator._model_features(fit_rows), fit_rows["y"],
            fit_rows["phase_id"])
        regional_metric, _ = _score(regional_model, held_rows, held_mask)
        broad_metric, _ = _score(broad_model, held_rows, held_mask)
        lateral_accel_model = _fit_regional_surface_with_lateral_accel(local)
        lateral_accel_metric, _ = _score(
            lateral_accel_model, held_rows, held_mask)
        lateral_accel_folds[f"{fit_key}_to_{held_key}"] = {
            "broad_data_regional_surface": regional_metric,
            "explicit_ay_minus_r_u_history": lateral_accel_metric,
        }
        broad_folds[f"{fit_key}_to_{held_key}"] = {
            "broad_full_run_model": broad_metric,
            "regional_surface": regional_metric,
        }
    regional_selected = all(
        fold["regional_surface"]["rmse_radps"]
        < fold["broad_full_run_model"]["rmse_radps"]
        and fold["regional_surface"]["samples_over_0p1"]
        <= fold["broad_full_run_model"]["samples_over_0p1"]
        for fold in broad_folds.values())

    broad_region_train_mask = _regional_gate(broad_train)
    broad_region_train = {
        key: value[broad_region_train_mask]
        for key, value in broad_train.items() if isinstance(value, np.ndarray)}
    regional_surface = _fit_regional_surface(broad_region_train)
    lateral_accel_surface = _fit_regional_surface_with_lateral_accel(
        broad_region_train)
    models = {
        "broad_full_spectrum_model": broad,
        "highspeed_schedule_specialist": specialist,
        "pooled_full_spectrum_plus_highspeed": pooled,
        "broad_data_regional_surface": regional_surface,
        "broad_data_regional_surface_plus_ay_minus_r_u_history": (
            lateral_accel_surface),
    }

    scores: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []
    validation_keys = ("highspeed_validation_r03", "grid_validation_r02",
                       "midpoint_validation_r04")
    for run_key in validation_keys:
        rows = data[run_key]
        scores[run_key] = {"run_id": RUNS[run_key][0], "regions": {}}
        for gate_name, mask in _gates(rows).items():
            scores[run_key]["regions"][gate_name] = {}
            source_indices = np.flatnonzero(mask)
            for model_name, model in models.items():
                metric, prediction = _score(model, rows, mask)
                scores[run_key]["regions"][gate_name][model_name] = metric
                for local_index in np.flatnonzero(
                        np.abs(prediction - rows["y"][mask]) > 0.1):
                    source_index = int(source_indices[local_index])
                    error_rows.append({
                        "run_id": RUNS[run_key][0],
                        "model": model_name,
                        "gate": gate_name,
                        "phase_event": str(rows["phase_event"][source_index]),
                        "causal_event": str(rows["causal_event"][source_index]),
                        "event_age_ms": float(rows["event_age_ms"][source_index]),
                        "wheel_speed_mps": float(rows["wheel_speed"][source_index]),
                        "steering_rad": float(rows["steering"][source_index]),
                        "target_yaw_residual_radps": float(rows["y"][source_index]),
                        "predicted_yaw_residual_radps": float(prediction[local_index]),
                        "absolute_error_radps": float(abs(
                            prediction[local_index] - rows["y"][source_index])),
                    })

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "offline candidate evaluation; no runtime integration",
        "packet_admission": "exactly two contiguous response packets only",
        "truth_usage": "simulator yaw rate is target and score only",
        "input_history_lags_ms": [lag * 25 for lag in history_lags],
        "training_runs": {
            "broad_full_spectrum": [RUNS[key][0] for key in
                                     ("grid_train_r01", "grid_train_r03")],
            "highspeed_specialist": [RUNS[key][0] for key in
                                     ("highspeed_train_r01", "highspeed_train_r02")],
            "pooled_full_spectrum_plus_highspeed": [RUNS[key][0] for key in
                ("grid_train_r01", "grid_train_r03", "highspeed_train_r01",
                 "highspeed_train_r02")],
        },
        "regional_surface_selection": {
            "gate": "measured rear-wheel mean speed 8–12 m/s and absolute measured steering below 0.25 rad",
            "selection_rule": "both leave-one-full-spectrum-run-out folds must lower RMSE and not increase errors over 0.1 rad/s",
            "cross_run_folds": broad_folds,
            "passed_both_folds": bool(regional_selected),
            "hyperparameters": REGIONAL_MODEL,
            "training_rows": int(np.count_nonzero(broad_region_train_mask)),
        },
        "explicit_lateral_acceleration_residual_selection": {
            "feature": "IMU ay - IMU yaw rate * mean rear-wheel surface speed at each 0/25/50/100-ms history lag",
            "cross_run_folds": lateral_accel_folds,
            "passed_both_folds": all(
                fold["explicit_ay_minus_r_u_history"]["rmse_radps"]
                < fold["broad_data_regional_surface"]["rmse_radps"]
                and fold["explicit_ay_minus_r_u_history"]["samples_over_0p1"]
                <= fold["broad_data_regional_surface"]["samples_over_0p1"]
                for fold in lateral_accel_folds.values()),
            "training_rows": int(np.count_nonzero(broad_region_train_mask)),
        },
        "audits": audits,
        "validation_scores": scores,
        "limitations": [
            "The high-speed specialist is trained on repeated 10.5/11.1 m/s steering schedules; transfer beyond their measured envelope is not established.",
            "The regional surface is selected only if both whole-run full-spectrum folds pass; its operating gate is based on measured wheel speed and steering.",
            "The single largest residual in a gate can remain large despite strong RMSE and p95 gains.",
            "Validation captures are independent runs but were previously inspected during model diagnosis.",
            "This research artifact does not change odometry, MPC, or simulator behavior.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ("run_id", "model", "gate", "phase_event", "causal_event", "event_age_ms",
              "wheel_speed_mps", "steering_rad", "target_yaw_residual_radps",
              "predicted_yaw_residual_radps", "absolute_error_radps")
    with (output / "highspeed_specialist_errors_over_0p1.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(error_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--history-500ms", action="store_true",
        help="use causal sensor history at 0/25/50/100/200/300/500 ms")
    args = parser.parse_args()
    history_lags = (0, 1, 2, 4, 8, 12, 20) if args.history_500ms else HISTORY_LAGS
    report = run(args.output, history_lags)
    for run_key, result in report["validation_scores"].items():
        print(run_key)
        for gate, pair in result["regions"].items():
            print(f"  {gate}")
            for name, metric in pair.items():
                if isinstance(metric, dict) and "samples" in metric:
                    print(f"    {name}: {metric}")
                else:
                    for model_name, model_metric in metric.items():
                        print(f"    {model_name}: {model_metric}")
    print(f"report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
