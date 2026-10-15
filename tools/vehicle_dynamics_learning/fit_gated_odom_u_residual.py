#!/usr/bin/env python3
"""Fit a run-gated Odom-u residual expert from train-only replay rows.

The gate is an observed encoder/odom speed disagreement, not a speed cap and
not an encoder innovation rejection. Outside the identified disagreement
regime, the candidate applies exactly zero correction. This tool never opens
bags or validation/test partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import train_odom_u_residual as trainer


THRESHOLDS_MPS = (3.0, 5.0)
ALPHA = 0.1
PROFILE_COUNT = 7


def _gate(features: np.ndarray, threshold_mps: float) -> np.ndarray:
    odom_index = trainer.FEATURE_NAMES.index("odom_u_mps")
    wheel_index = trainer.FEATURE_NAMES.index("encoder_mean_100ms_mps")
    return np.abs(features[:, wheel_index] - features[:, odom_index]) >= threshold_mps


def _run_metrics(residual: np.ndarray, prediction: np.ndarray,
                 groups: np.ndarray, active: np.ndarray
                 ) -> dict[str, Any]:
    baseline: dict[str, float] = {}
    corrected: dict[str, float] = {}
    active_baseline: dict[str, float] = {}
    active_corrected: dict[str, float] = {}
    for run in np.unique(groups):
        mask = groups == run
        baseline[str(run)] = float(np.sqrt(np.mean(residual[mask] ** 2)))
        corrected[str(run)] = float(np.sqrt(np.mean(
            (residual[mask] - prediction[mask]) ** 2)))
        selected = mask & active
        if np.any(selected):
            active_baseline[str(run)] = float(np.sqrt(np.mean(
                residual[selected] ** 2)))
            active_corrected[str(run)] = float(np.sqrt(np.mean(
                (residual[selected] - prediction[selected]) ** 2)))
    base_values = np.asarray(list(baseline.values()))
    corrected_values = np.asarray(list(corrected.values()))
    active_base_values = np.asarray(list(active_baseline.values()))
    active_corrected_values = np.asarray(list(active_corrected.values()))
    return {
        "run_macro_baseline_rmse_mps": float(np.mean(base_values)),
        "run_macro_gated_rmse_mps": float(np.mean(corrected_values)),
        "run_median_baseline_rmse_mps": float(np.median(base_values)),
        "run_median_gated_rmse_mps": float(np.median(corrected_values)),
        "runs_improved": int(sum(corrected[key] < value
                                 for key, value in baseline.items())),
        "active_sample_count": int(np.sum(active)),
        "active_run_count": len(active_baseline),
        "active_run_macro_baseline_rmse_mps": (
            float(np.mean(active_base_values)) if len(active_base_values) else None),
        "active_run_macro_gated_rmse_mps": (
            float(np.mean(active_corrected_values))
            if len(active_corrected_values) else None),
        "per_run_baseline_rmse_mps": baseline,
        "per_run_gated_rmse_mps": corrected,
        "per_run_active_sample_rmse_mps": active_corrected,
    }


def _fit_coefficients(features: np.ndarray, target: np.ndarray,
                      groups: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    weights = trainer._balanced_weights(groups)
    mean, scale = trainer._normalizer(features, weights)
    target_mean = float(np.average(target, weights=weights))
    normalized = (features - mean) / scale
    gram = normalized.T @ (weights[:, None] * normalized)
    rhs = normalized.T @ (weights * (target - target_mean))
    coefficient = np.linalg.solve(
        gram + ALPHA * np.eye(gram.shape[0]), rhs)
    return mean, scale, coefficient, target_mean


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit(rows_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")
    with np.load(rows_path, allow_pickle=False) as archive:
        features = archive["base_features"][:, :PROFILE_COUNT]
        residual = archive["base_residual_u"]
        groups = archive["base_run_id"]
    if len(features) != len(residual) or len(groups) != len(residual):
        raise ValueError("training row arrays have inconsistent lengths")
    if len(np.unique(groups)) < 10:
        raise ValueError("gated fitting requires at least ten unique training runs")

    output_dir.mkdir(parents=True, exist_ok=True)
    folds = trainer._group_folds(groups, n_splits=5)
    candidate_results: dict[str, Any] = {}
    for threshold in THRESHOLDS_MPS:
        active = _gate(features, threshold)
        fold_prediction = np.zeros(len(residual), dtype=np.float64)
        fold_support: list[dict[str, int]] = []
        for train_index, test_index in folds:
            train_active = _gate(features[train_index], threshold)
            selected_train = train_index[train_active]
            test_active = active[test_index]
            support_runs = (len(np.unique(groups[selected_train]))
                            if len(selected_train) else 0)
            fold_support.append({
                "training_active_samples": int(len(selected_train)),
                "training_active_runs": int(support_runs),
                "test_active_samples": int(np.sum(test_active)),
            })
            if support_runs < 1:
                continue
            prediction, _, _ = trainer._ridge_predict(
                features[selected_train], residual[selected_train],
                trainer._balanced_weights(groups[selected_train]),
                features[test_index[test_active]], ALPHA)
            fold_prediction[test_index[test_active]] = prediction
        metrics = _run_metrics(residual, fold_prediction, groups, active)
        candidate_results[f"mismatch_ge_{threshold:g}_mps"] = {
            "threshold_mps": threshold,
            "profile": "current_odom_and_permitted_sensors",
            "alpha": ALPHA,
            "grouped_cv": metrics,
            "fold_support": fold_support,
        }

        active_runs = np.unique(groups[active])
        if len(active_runs) < 2 or int(np.sum(active)) < 100:
            candidate_results[f"mismatch_ge_{threshold:g}_mps"][
                "candidate_model"] = None
            continue
        mean, scale, coefficient, intercept = _fit_coefficients(
            features[active], residual[active], groups[active])
        model = {
            "schema_version": 1,
            "target": "truth_u_minus_current_source_odom_u",
            "feature_names": list(trainer.FEATURE_NAMES[:PROFILE_COUNT]),
            "base_feature_names": list(trainer.FEATURE_NAMES[:PROFILE_COUNT]),
            "history_steps": 1,
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "coefficient": coefficient.tolist(),
            "intercept": intercept,
            "selected_alpha": ALPHA,
            "input_profile": "current_odom_sensor_inputs",
            "activation_gate": {
                "kind": "absolute_encoder_odom_forward_speed_mismatch_mps",
                "threshold_mps": threshold,
                "meaning": "apply expert only when abs(encoder_mean_100ms_mps - odom_u_mps) >= threshold",
            },
            "runtime_input_policy": (
                "current Odom u/v, existing 100 ms encoder mean, and IMU ax/ay/yaw; "
                "no command, simulator truth, transform, or future sensor input"),
            "training_provenance": {
                "rows_file": str(rows_path.resolve()),
                "rows_sha256": _sha256(rows_path),
                "run_count": int(len(np.unique(groups[active]))),
                "active_sample_count": int(np.sum(active)),
                "total_training_run_count": int(len(np.unique(groups))),
                "alpha": ALPHA,
                "selection": "fixed alpha from prior train-only run-disjoint fit; gate threshold is reported, not tuned on validation",
            },
        }
        model_path = output_dir / f"odom_u_residual_mismatch_ge_{threshold:g}mps.json"
        model_path.write_text(json.dumps(
            model, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        candidate_results[f"mismatch_ge_{threshold:g}_mps"][
            "candidate_model"] = str(model_path.resolve())

    report = {
        "training_rows": str(rows_path.resolve()),
        "training_rows_sha256": _sha256(rows_path),
        "training_runs": int(len(np.unique(groups))),
        "training_samples": int(len(residual)),
        "target": "one-step truth-u residual on current source Odom",
        "split_policy": "five-fold run-disjoint CV; train-only rows; no validation/test/final-test rows",
        "candidate_results": candidate_results,
    }
    report_path = output_dir / "gated_residual_cv.json"
    report_path.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = fit(args.training_rows.resolve(), args.output_dir.resolve())
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"gated Odom-u fit failed: {exc}\n")
    print(json.dumps({
        "runs": report["training_runs"],
        "samples": report["training_samples"],
        "cv": {key: value["grouped_cv"]["run_macro_gated_rmse_mps"]
               for key, value in report["candidate_results"].items()},
        "output": str(args.output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
