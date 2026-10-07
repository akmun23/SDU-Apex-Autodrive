#!/usr/bin/env python3
"""Test whether measured pre-action vehicle state explains throttle-slew effects.

Pairs are grouped by complete simulator captures. Training captures select the
kernel regularization; held-out captures are used only for final evaluation.
Only state available at the throttle stimulus is used as an input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.build_empirical_throttle_response_model import (
    FEATURE_FIELDS,
    METRICS,
    REPO_ROOT,
    _cell_key,
    _read_split_rows,
)


CONDITION_FEATURES = (
    "speed_target_mps", "throttle_delta_norm",
    "throttle_rise_rate_norm_per_sec", "abs_steering_command_rad",
    "turn_sign",
)
MOTION_STATE_FEATURES = (
    "speed_error_mps", "turn_steering_feedback_rad", "turn_v_mps",
    "turn_yaw_rate_rps", "throttle_feedback_norm",
    "wheel_residual_abs_mps", "wheel_residual_common_mps",
    "wheel_residual_asymmetry_turn_mps", "abs_lateral_accel_mps2",
)
ROLL_STATE_FEATURES = (
    "abs_roll_rad", "turn_roll_rad", "turn_roll_rate_rps",
    "abs_roll_rate_rps",
)
FEATURE_SETS = {
    "condition_only": CONDITION_FEATURES,
    "condition_plus_motion_state": CONDITION_FEATURES + MOTION_STATE_FEATURES,
    "condition_plus_motion_and_roll": (
        CONDITION_FEATURES + MOTION_STATE_FEATURES + ROLL_STATE_FEATURES),
}
LENGTH_SCALES = (0.5, 1.0, 2.0, 4.0)
RIDGE_ALPHAS = (1.0e-3, 1.0e-2, 1.0e-1, 1.0)


def _phase_inputs(phase: dict[str, Any], target_speed: float,
                  turn_sign: float) -> dict[str, float]:
    state = phase["stimulus_state"]
    pre = phase["pre_window"]["median"]
    speed = float(state["speed_mps"])
    return {
        "speed_error_mps": speed - target_speed,
        "turn_steering_feedback_rad": (
            turn_sign * float(state["steering_feedback_rad"])),
        "turn_v_mps": turn_sign * float(state["vy_mps"]),
        "turn_yaw_rate_rps": turn_sign * float(state["yaw_rate_rps"]),
        "throttle_feedback_norm": float(state["throttle_feedback_norm"]),
        "wheel_residual_abs_mps": float(
            pre["mean_abs_rear_wheel_residual_mps"]),
        "wheel_residual_common_mps": float(
            pre["common_rear_wheel_residual_mps"]),
        "wheel_residual_asymmetry_turn_mps": turn_sign * float(
            pre["rear_wheel_residual_asymmetry_mps"]),
        "abs_lateral_accel_mps2": float(
            pre["abs_rigid_body_lateral_acceleration_mps2"]),
        "abs_roll_rad": float(pre["abs_imu_roll_rad"]),
        "turn_roll_rad": turn_sign * float(pre["imu_roll_rad"]),
        "turn_roll_rate_rps": turn_sign * float(pre["imu_roll_rate_rps"]),
        "abs_roll_rate_rps": float(pre["abs_imu_roll_rate_rps"]),
    }


def _load_pairs(paths: list[Path], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_reports = []
    for path in paths:
        selected, source = _read_split_rows([path], split)
        payload = json.loads(path.read_text(encoding="utf-8"))
        phases = {}
        for phase in payload.get("phase_results", []):
            phases[(str(phase.get("run_id")),
                    str(phase.get("condition_pair_id")),
                    str(phase.get("profile")))] = phase
        missing_states = []
        for pair in selected:
            run_id = str(pair["run_id"])
            pair_id = str(pair["condition_pair_id"])
            ramp = phases.get((run_id, pair_id, "ramp"))
            step = phases.get((run_id, pair_id, "step"))
            if ramp is None or step is None:
                missing_states.append(pair_id)
                continue
            turn_sign = 1.0 if pair["turn_direction"] == "left" else -1.0
            ramp_inputs = _phase_inputs(
                ramp, float(pair["speed_target_mps"]), turn_sign)
            step_inputs = _phase_inputs(
                step, float(pair["speed_target_mps"]), turn_sign)
            condition = {
                "speed_target_mps": float(pair["speed_target_mps"]),
                "throttle_delta_norm": float(pair["throttle_delta_norm"]),
                "throttle_rise_rate_norm_per_sec": float(
                    pair["throttle_rise_rate_norm_per_sec"]),
                "abs_steering_command_rad": float(
                    pair["abs_steering_command_rad"]),
                "turn_sign": turn_sign,
            }
            model_row = {
                "run_id": run_id,
                "split": split,
                "cell_key": _cell_key(pair),
                "features": {
                    name: float(condition[name] if name in condition else
                                0.5 * (ramp_inputs[name] + step_inputs[name]))
                    for name in CONDITION_FEATURES + MOTION_STATE_FEATURES
                    + ROLL_STATE_FEATURES
                },
                "targets": {
                    name: float(pair[field])
                    for name, field in METRICS.items()
                },
            }
            rows.append(model_row)
        if missing_states:
            raise ValueError(f"paired rows lack ramp/step stimulus state in {path}: "
                             f"{missing_states[:5]}")
        source_reports.extend(source)
    if not rows:
        raise ValueError(f"no eligible {split} rows")
    return rows, source_reports


def _design_matrix(rows: list[dict[str, Any]], feature_names: tuple[str, ...]
                   ) -> np.ndarray:
    return np.asarray([[row["features"][name] for name in feature_names]
                       for row in rows], dtype=np.float64)


def _standardize(train_x: np.ndarray, test_x: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1.0e-9] = 1.0
    return (train_x - mean) / scale, (test_x - mean) / scale


def _rbf_kernel(left: np.ndarray, right: np.ndarray,
                length_scale: float) -> np.ndarray:
    squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * squared / (length_scale * length_scale))


def _kernel_fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                        query_x: np.ndarray, length_scale: float,
                        alpha: float) -> np.ndarray:
    x_train, x_query = _standardize(train_x, query_x)
    y_mean = float(train_y.mean())
    kernel = _rbf_kernel(x_train, x_train, length_scale)
    weights = np.linalg.solve(
        kernel + alpha * np.eye(len(kernel)), train_y - y_mean)
    return y_mean + _rbf_kernel(x_query, x_train, length_scale) @ weights


def _ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                       query_x: np.ndarray, alpha: float) -> np.ndarray:
    x_train, x_query = _standardize(train_x, query_x)
    x_mean = x_train.mean(axis=0)
    y_mean = float(train_y.mean())
    centered_x = x_train - x_mean
    centered_y = train_y - y_mean
    coefficients = np.linalg.solve(
        centered_x.T @ centered_x + alpha * np.eye(centered_x.shape[1]),
        centered_x.T @ centered_y)
    return y_mean + (x_query - x_mean) @ coefficients


def _rmse(errors: list[float]) -> float | None:
    return math.sqrt(float(np.mean(np.square(errors)))) if errors else None


def _fit_group_means(rows: list[dict[str, Any]], group_key) -> dict[Any, dict[str, float]]:
    values: dict[Any, dict[str, list[float]]] = {}
    for row in rows:
        key = group_key(row)
        metric_values = values.setdefault(key, {})
        for name, value in row["targets"].items():
            metric_values.setdefault(name, []).append(value)
    return {key: {name: float(np.mean(samples))
                  for name, samples in metrics.items()}
            for key, metrics in values.items()}


def _baseline_predictions(train_rows: list[dict[str, Any]],
                          test_rows: list[dict[str, Any]],
                          kind: str) -> dict[str, np.ndarray]:
    groups = _fit_group_means(
        train_rows,
        (lambda row: row["cell_key"] if kind == "exact"
         else row["cell_key"][:3]))
    predictions: dict[str, list[float]] = {
        name: [] for name in METRICS
    }
    all_values = {name: float(np.mean([row["targets"][name]
                                       for row in train_rows]))
                  for name in METRICS}
    for row in test_rows:
        key = row["cell_key"] if kind == "exact" else row["cell_key"][:3]
        group = groups.get(key, all_values)
        for name in METRICS:
            predictions[name].append(group.get(name, all_values[name]))
    return {name: np.asarray(values) for name, values in predictions.items()}


def _group_run_rmse(rows: list[dict[str, Any]], predictions: dict[str, np.ndarray]
                    ) -> dict[str, dict[str, Any]]:
    by_run: dict[str, dict[str, list[float]]] = {}
    for index, row in enumerate(rows):
        metrics = by_run.setdefault(row["run_id"], {})
        for name in METRICS:
            metrics.setdefault(name, []).append(
                float(predictions[name][index] - row["targets"][name]))
    result = {}
    for run_id, metrics in by_run.items():
        result[run_id] = {name: _rmse(errors) for name, errors in metrics.items()}
    return result


def _macro_summary(per_run: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
    output = {}
    for metric in METRICS:
        values = [run[metric] for run in per_run.values()
                  if run[metric] is not None]
        output[metric] = float(np.mean(values)) if values else None
    return output


def _select_kernel_hyperparameters(train_rows: list[dict[str, Any]],
                                   metric: str,
                                   feature_names: tuple[str, ...]
                                   ) -> dict[str, Any]:
    run_ids = sorted({row["run_id"] for row in train_rows})
    scores = []
    x_all = _design_matrix(train_rows, feature_names)
    y_all = np.asarray([row["targets"][metric] for row in train_rows])
    row_runs = np.asarray([row["run_id"] for row in train_rows])
    for length_scale in LENGTH_SCALES:
        for alpha in RIDGE_ALPHAS:
            fold_scores = []
            for held_out in run_ids:
                fit = row_runs != held_out
                test = ~fit
                prediction = _kernel_fit_predict(
                    x_all[fit], y_all[fit], x_all[test], length_scale, alpha)
                fold_scores.append(_rmse(
                    list(prediction - y_all[test])) or 0.0)
            scores.append({
                "length_scale": length_scale,
                "alpha": alpha,
                "macro_run_rmse": float(np.mean(fold_scores)),
                "fold_rmse": fold_scores,
            })
    if len(run_ids) < 2:
        raise ValueError("need at least two training captures for grouped tuning")
    selected = min(scores, key=lambda item: item["macro_run_rmse"])
    return {"selected": selected, "grid": scores}


def evaluate(training_paths: list[Path], validation_paths: list[Path]
             ) -> dict[str, Any]:
    training_rows, training_sources = _load_pairs(training_paths, "train")
    validation_rows, validation_sources = _load_pairs(validation_paths, "validation")
    train_runs = {row["run_id"] for row in training_rows}
    validation_runs = {row["run_id"] for row in validation_rows}
    if train_runs & validation_runs:
        raise ValueError(f"capture leakage: {sorted(train_runs & validation_runs)}")
    if len(train_runs) < 2 or len(validation_runs) < 2:
        raise ValueError("require >=2 independent training and validation captures")

    selected_models: dict[str, Any] = {}
    validation_results: dict[str, Any] = {}
    train_matrix_by_set = {
        label: _design_matrix(training_rows, features)
        for label, features in FEATURE_SETS.items()
    }
    validation_matrix_by_set = {
        label: _design_matrix(validation_rows, features)
        for label, features in FEATURE_SETS.items()
    }
    for feature_set, features in FEATURE_SETS.items():
        metrics_hyperparameters = {}
        predictions: dict[str, np.ndarray] = {}
        for metric in METRICS:
            tuning = _select_kernel_hyperparameters(
                training_rows, metric, features)
            selected = tuning["selected"]
            train_y = np.asarray([row["targets"][metric]
                                  for row in training_rows])
            valid_y = np.asarray([row["targets"][metric]
                                  for row in validation_rows])
            predictions[metric] = _kernel_fit_predict(
                train_matrix_by_set[feature_set], train_y,
                validation_matrix_by_set[feature_set],
                selected["length_scale"], selected["alpha"])
            metrics_hyperparameters[metric] = tuning
        per_run = _group_run_rmse(validation_rows, predictions)
        validation_results[feature_set] = {
            "feature_names": list(features),
            "per_run_rmse": per_run,
            "macro_run_rmse": _macro_summary(per_run),
        }
        selected_models[feature_set] = metrics_hyperparameters

    ridge_results = {}
    for feature_set, features in FEATURE_SETS.items():
        matrix_train = train_matrix_by_set[feature_set]
        matrix_valid = validation_matrix_by_set[feature_set]
        predictions = {}
        alpha_by_metric = {}
        for metric in METRICS:
            y_train = np.asarray([row["targets"][metric]
                                  for row in training_rows])
            y_valid = np.asarray([row["targets"][metric]
                                  for row in validation_rows])
            row_runs = np.asarray([row["run_id"] for row in training_rows])
            alpha_scores = []
            for alpha in RIDGE_ALPHAS:
                fold_errors = []
                for held_out in sorted(train_runs):
                    fit = row_runs != held_out
                    test = ~fit
                    fold_prediction = _ridge_fit_predict(
                        matrix_train[fit], y_train[fit], matrix_train[test], alpha)
                    fold_errors.append(_rmse(list(
                        fold_prediction - y_train[test])) or 0.0)
                alpha_scores.append({"alpha": alpha,
                                     "macro_run_rmse": float(np.mean(fold_errors))})
            chosen_alpha = min(alpha_scores,
                               key=lambda result: result["macro_run_rmse"])["alpha"]
            alpha_by_metric[metric] = {"selected": chosen_alpha,
                                       "grid": alpha_scores}
            predictions[metric] = _ridge_fit_predict(
                matrix_train, y_train, matrix_valid, chosen_alpha)
        per_run = _group_run_rmse(validation_rows, predictions)
        ridge_results[feature_set] = {
            "feature_names": list(features),
            "per_run_rmse": per_run,
            "macro_run_rmse": _macro_summary(per_run),
            "hyperparameters": alpha_by_metric,
        }

    baseline_results = {}
    for baseline in ("coarse", "exact"):
        predictions = _baseline_predictions(training_rows, validation_rows,
                                            baseline)
        per_run = _group_run_rmse(validation_rows, predictions)
        baseline_results[baseline] = {
            "per_run_rmse": per_run,
            "macro_run_rmse": _macro_summary(per_run),
        }

    return {
        "schema_version": 1,
        "model_question": (
            "Do measured initial motion and roll states explain throttle-ramp "
            "response variation beyond the commanded speed/steering cell?"),
        "input_policy": (
            "Features use only measured state at the throttle stimulus and the "
            "pre-stimulus window. No post-stimulus sensor values are inputs."),
        "target_interpretation": (
            "step-minus-ramp paired change over 0.10-1.00 s; rear-wheel mismatch "
            "is a proxy, not direct tire slip or force."),
        "training_sources": training_sources,
        "validation_sources": validation_sources,
        "training_pair_count": len(training_rows),
        "validation_pair_count": len(validation_rows),
        "training_capture_count": len(train_runs),
        "validation_capture_count": len(validation_runs),
        "baselines": baseline_results,
        "kernel_ridge_state_models": validation_results,
        "ridge_state_models": ridge_results,
        "kernel_training_only_hyperparameters": selected_models,
        "selection_policy": (
            "Kernel scales and ridge alpha are selected only by leave-one-"
            "training-capture-out scoring; validation captures are not used "
            "for fitting or hyperparameter selection."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-analysis", type=Path, action="append",
                        required=True)
    parser.add_argument("--validation-analysis", type=Path, action="append",
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    training = [path if path.is_absolute() else REPO_ROOT / path
                for path in args.training_analysis]
    validation = [path if path.is_absolute() else REPO_ROOT / path
                  for path in args.validation_analysis]
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    report = evaluate(training, validation)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "training_pairs": report["training_pair_count"],
        "validation_pairs": report["validation_pair_count"],
        "coarse_baseline": report["baselines"]["coarse"]["macro_run_rmse"],
        "exact_lookup": report["baselines"]["exact"]["macro_run_rmse"],
        "kernel_models": {key: value["macro_run_rmse"]
                          for key, value in report[
                              "kernel_ridge_state_models"].items()},
        "output": str(output),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
