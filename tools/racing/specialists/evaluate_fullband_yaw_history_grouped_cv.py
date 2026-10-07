#!/usr/bin/env python3
"""Compare causal yaw-history features using grouped CV on training runs only."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

try:
    from fit_fullband_yaw_extratrees import _feature_schema, _run_balanced_weights
    import fit_fullband_yaw_regime_atlas as atlas
except ModuleNotFoundError:
    from tools.racing.specialists.fit_fullband_yaw_extratrees import (
        _feature_schema,
        _run_balanced_weights,
    )
    from tools.racing.specialists import fit_fullband_yaw_regime_atlas as atlas


ROOT = Path(__file__).resolve().parents[3]
TREE_CONFIGS = {
    "extra_trees_v1_depth5_leaf4": (5, 4),
    "extra_trees_history_v2_depth5_leaf4": (5, 4),
}


def _fit_tree(x: np.ndarray, y: np.ndarray, run_ids: np.ndarray,
              max_depth: int,
              feature_scales: np.ndarray
              ) -> tuple[np.ndarray, ExtraTreesRegressor]:
    mean = np.mean(x, axis=0)
    z = (x - mean) / feature_scales
    model = ExtraTreesRegressor(
        n_estimators=120,
        max_depth=max_depth,
        min_samples_leaf=4,
        max_features=0.8,
        random_state=20261007,
        n_jobs=1,
    )
    model.fit(z, y, sample_weight=_run_balanced_weights(run_ids))
    return mean, model


def _fit_tree_job(job):
    key, x, y, run_ids, max_depth, feature_scales = job
    mean, model = _fit_tree(x, y, run_ids, max_depth, feature_scales)
    return key, mean, model


def _metrics(errors: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(errors, dtype=np.float64)
    if not len(values):
        return {"samples": 0}
    absolute = np.abs(values)
    return {
        "samples": int(len(values)),
        "rmse_radps": float(np.sqrt(np.mean(values * values))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
    }


def _bootstrap_run_delta(deltas: np.ndarray, seed: int) -> dict[str, Any]:
    if not len(deltas):
        return {"independent_runs": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(deltas, size=(20000, len(deltas)), replace=True)
    means = np.mean(draws, axis=1)
    return {
        "independent_runs": int(len(deltas)),
        "mean_run_rmse_difference_radps": float(np.mean(deltas)),
        "median_run_rmse_difference_radps": float(np.median(deltas)),
        "wins": int(np.count_nonzero(deltas < 0.0)),
        "losses": int(np.count_nonzero(deltas > 0.0)),
        "bootstrap_95pct_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
    }


def evaluate(output_path: Path, workers: int = 12, folds: int = 4,
             extension: str = "lagged_history") -> dict[str, Any]:
    if workers < 1 or folds < 2:
        raise ValueError("workers must be positive and folds must be at least 2")
    if extension not in ("lagged_history", "imu_roll"):
        raise ValueError("extension must be lagged_history or imu_roll")
    candidate_label = ("extra_trees_history_v2" if extension == "lagged_history"
                       else "extra_trees_imu_roll_v2")
    include_history = extension == "lagged_history"
    include_roll = extension == "imu_roll"
    run_series, source_audit = atlas._discover_run_series()
    train_series = [series for series in run_series if series.split == "train"]
    if len(train_series) < folds:
        raise RuntimeError("not enough independent training runs for grouped CV")
    x_parts = []
    y_parts = []
    cell_parts = []
    phase_parts = []
    run_parts = []
    for series in train_series:
        common = atlas._make_rows(
            series, 0.0025, True, True, False, False,
            require_imu_roll=include_roll)
        extended = atlas._make_rows(
            series, 0.0025, True, True, False, include_history,
            include_imu_roll=include_roll, require_imu_roll=include_roll)
        if (len(common[1]) != len(extended[1])
                or not np.array_equal(common[2], extended[2])
                or not np.array_equal(common[3], extended[3])):
            raise RuntimeError(f"extended rows do not align for {series.run_id}")
        np.testing.assert_array_equal(common[4], extended[4])
        np.testing.assert_array_equal(common[5], extended[5])
        np.testing.assert_array_equal(common[6], extended[6])
        np.testing.assert_allclose(common[0], extended[0][:, :14],
                                   rtol=0.0, atol=0.0)
        x_parts.append(extended[0])
        y_parts.append(extended[1])
        cell_parts.append(extended[2])
        phase_parts.append(extended[3])
        run_parts.append(extended[4])
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    cells = np.concatenate(cell_parts)
    phases = np.concatenate(phase_parts)
    run_ids = np.concatenate(run_parts)
    groups = np.unique(run_ids)
    v1_names, v1_scales = _feature_schema(False)
    added_names = (atlas.LAGGED_HISTORY_FEATURE_NAMES if include_history
                   else atlas.IMU_ROLL_FEATURE_NAMES)
    added_scales = (atlas.LAGGED_HISTORY_FEATURE_SCALES if include_history
                    else atlas.IMU_ROLL_FEATURE_SCALES)
    v2_names = v1_names + added_names
    v2_scales = np.concatenate((v1_scales, added_scales))
    if len(v1_names) != 14 or len(v2_names) != x.shape[1]:
        raise RuntimeError("unexpected v1/v2 feature schema")

    errors_by_run: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    errors_by_speed: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    errors_by_regime: dict[tuple[int, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    errors_by_regime_run: dict[
        tuple[int, int, int], dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_errors: dict[str, list[float]] = defaultdict(list)
    supported_keys = set()
    splitter = GroupKFold(n_splits=folds)
    splits = list(splitter.split(np.zeros((len(y), 1)), y, groups=run_ids))
    for fold_index, (train_idx, hold_idx) in enumerate(splits):
        train_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        hold_groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for idx in train_idx:
            cell = cells[idx]
            train_groups[(int(cell[0]), int(cell[1]), int(phases[idx]))].append(int(idx))
        for idx in hold_idx:
            cell = cells[idx]
            hold_groups[(int(cell[0]), int(cell[1]), int(phases[idx]))].append(int(idx))

        linear_models = {}
        tree_jobs = {"v1": [], "v2": []}
        for key, raw_indices in train_groups.items():
            indices = np.asarray(raw_indices, dtype=np.int64)
            counts = Counter(run_ids[indices].tolist())
            eligible_runs = {
                run_id for run_id, count in counts.items()
                if count >= atlas.MIN_SAMPLES_PER_RUN
            }
            keep = np.asarray([run_id in eligible_runs
                               for run_id in run_ids[indices]], dtype=bool)
            indices = indices[keep]
            if (len(indices) < atlas.MIN_CELL_SAMPLES
                    or len(eligible_runs) < atlas.MIN_CELL_RUNS):
                continue
            model = atlas._fit_model(
                x[indices, :14], y[indices], run_ids[indices],
                feature_scales=v1_scales)
            if model is None:
                continue
            linear_models[key] = model
            tree_jobs["v1"].append((key, x[indices, :14], y[indices],
                                    run_ids[indices], TREE_CONFIGS[
                                        "extra_trees_v1_depth5_leaf4"][0],
                                    v1_scales))
            tree_jobs["v2"].append((key, x[indices], y[indices],
                                    run_ids[indices], TREE_CONFIGS[
                                        "extra_trees_history_v2_depth5_leaf4"][0],
                                    v2_scales))

        fitted = {}
        for family in ("v1", "v2"):
            fitted[family] = {}
            for key, mean, estimator in Parallel(
                    n_jobs=workers,
                    prefer="threads",
                    return_as="generator_unordered",
                    batch_size=1,
                    pre_dispatch=2 * workers,
            )(delayed(_fit_tree_job)(job) for job in tree_jobs[family]):
                fitted[family][key] = (mean, estimator)

        for key, raw_indices in hold_groups.items():
            if key not in linear_models:
                continue
            indices = np.asarray(raw_indices, dtype=np.int64)
            features = x[indices]
            truth = features[:, 0] + y[indices]
            linear = linear_models[key]
            z_linear = (features[:, :14] - linear["mean"]) / v1_scales
            pred_linear = features[:, 0] + linear["coefficients"][0] + (
                z_linear @ linear["coefficients"][1:])
            preds = {"huber_ridge_v1": pred_linear}
            for family, label in (("v1", "extra_trees_v1"),
                                  ("v2", candidate_label)):
                mean, estimator = fitted[family][key]
                scales = v1_scales if family == "v1" else v2_scales
                z = (features[:, :14] if family == "v1" else features) - mean
                z = z / scales
                preds[label] = features[:, 0] + estimator.predict(z)
            supported_keys.add(key)
            for label, prediction in preds.items():
                error = prediction - truth
                all_errors[label].extend(error.tolist())
                errors_by_speed[key[0]][label].extend(error.tolist())
                errors_by_regime[key][label].extend(error.tolist())
                for run_id in np.unique(run_ids[indices]):
                    local = run_ids[indices] == run_id
                    errors_by_run[str(run_id)][label].extend(error[local].tolist())
                    errors_by_regime_run[key][str(run_id)][label].extend(
                        error[local].tolist())
        print(f"completed grouped fold {fold_index + 1}/{folds}; "
              f"supported cells so far={len(supported_keys)}", flush=True)

    per_run = {}
    for run_id, methods in sorted(errors_by_run.items()):
        per_run[run_id] = {label: _metrics(values)
                           for label, values in sorted(methods.items())}
    deltas_v1_vs_linear = []
    deltas_candidate_vs_v1 = []
    for metrics in per_run.values():
        if "huber_ridge_v1" in metrics and "extra_trees_v1" in metrics:
            deltas_v1_vs_linear.append(
                metrics["extra_trees_v1"]["rmse_radps"]
                - metrics["huber_ridge_v1"]["rmse_radps"])
        if "extra_trees_v1" in metrics and candidate_label in metrics:
            deltas_candidate_vs_v1.append(
                metrics[candidate_label]["rmse_radps"]
                - metrics["extra_trees_v1"]["rmse_radps"])
    deltas_v1 = np.asarray(deltas_v1_vs_linear, dtype=np.float64)
    deltas_candidate = np.asarray(deltas_candidate_vs_v1, dtype=np.float64)
    per_regime = {}
    for key, methods in sorted(errors_by_regime.items()):
        local_runs = errors_by_regime_run[key]
        paired = []
        for run_methods in local_runs.values():
            if "extra_trees_v1" in run_methods and candidate_label in run_methods:
                baseline_rmse = _metrics(run_methods["extra_trees_v1"])[
                    "rmse_radps"]
                candidate_rmse = _metrics(run_methods[candidate_label])[
                    "rmse_radps"]
                paired.append(candidate_rmse - baseline_rmse)
        per_regime[f"speed_bin_{key[0]}_steering_bin_{key[1]}_phase_{key[2]}"] = {
            "speed_center_mps": float(atlas.SPEED_CENTERS[key[0]]),
            "steering_center_rad": float(atlas.STEERING_CENTERS[key[1]]),
            "phase": {"-1": "unwind", "0": "steady_or_low_rate",
                      "1": "turn_in"}[str(key[2])],
            "out_of_fold_metrics": {
                label: _metrics(values) for label, values in sorted(methods.items())
            },
            "paired_run_count": len(paired),
            "candidate_minus_v1_mean_run_rmse_radps": (
                float(np.mean(paired)) if paired else None),
            "candidate_wins": int(np.count_nonzero(np.asarray(paired) < 0.0)),
            "candidate_losses": int(np.count_nonzero(np.asarray(paired) > 0.0)),
        }
    report = {
        "title": f"Training-run grouped CV: {extension} yaw features",
        "split_policy": "4-fold GroupKFold by train run ID; validation/test/final-test values not used",
        "data_audit": {
            "training_runs": len(train_series),
            "training_rows": int(len(y)),
            "validation_runs_read_but_not_used": sum(
                series.split == "validation" for series in run_series),
            "test_or_final_test_arrays_read": False,
        },
        "features": {
            "v1": list(v1_names),
            "v2": list(v2_names),
            "extension": extension,
            "added_features": list(added_names),
            "all_features_at_or_before_k": True,
            "current_attitude_sensor_required": include_roll,
        },
        "tree_config": {
            "n_estimators": 120,
            "max_depth": 5,
            "min_samples_leaf": 4,
            "max_features": 0.8,
            "sample_weight": "equal total weight per contributing training run",
        },
        "supported_cell_phase_keys_any_fold": len(supported_keys),
        "out_of_fold_metrics": {
            label: _metrics(values) for label, values in sorted(all_errors.items())
        },
        "run_macro_uncertainty": {
            "extra_trees_v1_minus_huber": _bootstrap_run_delta(
                deltas_v1, 202610071),
            f"{candidate_label}_minus_extra_trees_v1": _bootstrap_run_delta(
                deltas_candidate, 202610072),
        },
        "per_run": per_run,
        "per_supported_speed_steering_phase_cell": per_regime,
        "per_speed_band": {
            f"{speed * 0.5:.1f}-{(speed + 1) * 0.5:.1f}": {
                label: _metrics(values) for label, values in sorted(methods.items())
            }
            for speed, methods in sorted(errors_by_speed.items())
        },
        "interpretation_limits": [
            "Out-of-fold predictions are from training runs only and are grouped by whole capture.",
            "A cell is scored only in folds where its training support passes the existing 40-sample/2-run gates.",
            "This is one-step GT-labelled prediction, not recursive plant validation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"report: {output_path}")
    print(json.dumps(report["out_of_fold_metrics"], indent=2))
    print(json.dumps(report["run_macro_uncertainty"], indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "live_runs/racing_model_diagnostics_20261007/"
                        "fullband_yaw_regime_atlas_v11_lowangle_unwind/"
                        "yaw_history_grouped_cv.json")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--extension", choices=("lagged_history", "imu_roll"),
                        default="lagged_history")
    args = parser.parse_args()
    evaluate(args.output, args.workers, args.folds, args.extension)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
