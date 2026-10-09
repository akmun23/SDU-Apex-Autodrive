#!/usr/bin/env python3
"""Evaluate left/right reflection augmentation for exact-two yaw prediction.

All feature reflections are physical sign changes or rear-wheel swaps. Models
are trained only on admitted training runs; untouched validation runs provide
the paired whole-run comparison. Ground truth is a label only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.specialists import audit_yaw_full_domain_exact_two as audit
from tools.racing.specialists import evaluate_yaw_sensor_speed_latent as data
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = data.DEFAULT_SOURCE
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_reflection_symmetry_r03")


def _mirror_features(x: np.ndarray) -> np.ndarray:
    """Reflect sensor histories across the vehicle longitudinal axis."""
    mirrored = np.asarray(x).copy()
    width = len(atlas.OBSERVATION_NAMES)
    base_signs = {
        "steering_feedback_rad", "imu_ay_mps2", "imu_yaw_rate_rps",
        "steering_command_rad", "imu_roll_rad", "imu_roll_rate_rps",
    }
    for lag_index in range(len(audit.HISTORY_LAGS)):
        start = lag_index * width
        for name in base_signs:
            index = start + atlas.OBSERVATION_NAMES.index(name)
            mirrored[:, index] *= -1.0
        left = start + atlas.OBSERVATION_NAMES.index("rear_left_surface_mps")
        right = start + atlas.OBSERVATION_NAMES.index("rear_right_surface_mps")
        mirrored[:, [left, right]] = mirrored[:, [right, left]]

    derived_start = len(audit.HISTORY_LAGS) * width
    derived_signs = {
        "rear_wheel_split_mps", "steering_feedback_rate_radps",
        "steering_command_gap_rad", "imu_yaw_rate_change_radps",
    }
    for name in derived_signs:
        index = derived_start + atlas.DERIVED_NAMES.index(name)
        mirrored[:, index] *= -1.0
    return mirrored


def _fit(x: np.ndarray, y: np.ndarray, run_ids: np.ndarray):
    return audit.ExtraTreesRegressor(**audit.ESTIMATOR).fit(
        x, y, sample_weight=audit._run_balanced_weights(run_ids))


def _metric(error: np.ndarray) -> dict[str, Any]:
    return data._metric(error)


def _paired_runs(run_ids: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                 selected: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(run_ids).astype(str)
    names = np.unique(ids[selected])
    values = np.asarray([
        (np.sqrt(np.mean(base[selected & (ids == name)] ** 2)),
         np.sqrt(np.mean(candidate[selected & (ids == name)] ** 2)))
        for name in names
    ], dtype=np.float64)
    rng = np.random.default_rng(20261009)
    picks = rng.integers(0, len(values), size=(10_000, len(values)))
    deltas = (np.sqrt(np.mean(values[picks, 1] ** 2, axis=1))
              - np.sqrt(np.mean(values[picks, 0] ** 2, axis=1)))
    return {
        "independent_runs": int(len(names)),
        "baseline_run_macro_rmse": float(np.mean(values[:, 0])),
        "reflection_run_macro_rmse": float(np.mean(values[:, 1])),
        "mean_per_run_rmse_delta_candidate_minus_baseline": float(
            np.mean(values[:, 1] - values[:, 0])),
        "bootstrap_95pct_delta_ci": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def run(source_dir: Path = DEFAULT_SOURCE,
        output_dir: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    train, valid, corpus_audit = data._load_parts(source_dir)
    x_train = train["x_timed"].astype(np.float32)
    x_valid = valid["x_timed"].astype(np.float32)
    y_train = train["residual"].astype(np.float32)
    y_valid = valid["residual"].astype(np.float32)
    train_ids = train["run_id"].astype(str)
    valid_ids = valid["run_id"].astype(str)
    train_event = train["event"].astype(str)
    valid_event = valid["event"].astype(str)
    baseline = np.full(len(y_valid), np.nan, dtype=np.float64)
    reflected = np.full(len(y_valid), np.nan, dtype=np.float64)
    event_metrics: dict[str, Any] = {}

    for event in sorted(set(train_event)):
        tr = train_event == event
        va = valid_event == event
        independent_train_runs = len(np.unique(train_ids[tr]))
        if (tr.sum() < audit.MIN_EVENT_ROWS
                or independent_train_runs < audit.MIN_EVENT_RUNS
                or not np.any(va)):
            continue
        base_model = _fit(x_train[tr], y_train[tr], train_ids[tr])
        x_train_mirror = _mirror_features(x_train[tr])
        mirrored_ids = np.concatenate((train_ids[tr], train_ids[tr]))
        mirrored_x = np.concatenate((x_train[tr], x_train_mirror))
        mirrored_y = np.concatenate((y_train[tr], -y_train[tr]))
        mirror_model = _fit(mirrored_x, mirrored_y, mirrored_ids)
        base_prediction = base_model.predict(x_valid[va])
        mirror_prediction = 0.5 * (
            mirror_model.predict(x_valid[va])
            - mirror_model.predict(_mirror_features(x_valid[va])))
        baseline[va] = base_prediction
        reflected[va] = mirror_prediction
        event_metrics[event] = {
            "validation_rows": int(va.sum()),
            "baseline": _metric(base_prediction - y_valid[va]),
            "reflection_augmented_and_antisymmetrized": _metric(
                mirror_prediction - y_valid[va]),
        }
        print(f"{event}: {event_metrics[event]}", flush=True)

    base_error = baseline - y_valid
    reflected_error = reflected - y_valid
    selected = np.isfinite(base_error) & np.isfinite(reflected_error)
    steering = valid["steering"].astype(np.float64)
    speed = valid["gt_speed"].astype(np.float64)
    mismatch = np.abs(valid["gt_speed"] - valid["wheel_speed"])
    regions: dict[str, Any] = {}
    for low, high in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
                      (4.0, 6.0), (6.0, 9.0), (9.0, 12.1)):
        mask = selected & (speed >= low) & (speed < high)
        if mask.any():
            regions[f"speed_{low:g}_{high:g}"] = {
                "baseline": _metric(base_error[mask]),
                "reflection": _metric(reflected_error[mask]),
            }
    for low, high in ((0.0, 0.10), (0.10, 0.20), (0.20, 0.30),
                      (0.30, 0.40), (0.40, 0.51)):
        mask = selected & (np.abs(steering) >= low) & (np.abs(steering) < high)
        if mask.any():
            regions[f"abs_steering_{low:.2f}_{high:.2f}"] = {
                "baseline": _metric(base_error[mask]),
                "reflection": _metric(reflected_error[mask]),
            }
    for low, high in ((0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, np.inf)):
        mask = selected & (mismatch >= low) & (mismatch < high)
        if mask.any():
            regions[f"abs_speed_mismatch_{low:g}+mps"] = {
                "baseline": _metric(base_error[mask]),
                "reflection": _metric(reflected_error[mask]),
            }

    result = {
        "title": "Reflection symmetry candidate for exact-two yaw prediction",
        "feature_transform": {
            "sign_flip": sorted({
                "steering_feedback_rad", "imu_ay_mps2", "imu_yaw_rate_rps",
                "steering_command_rad", "imu_roll_rad", "imu_roll_rate_rps",
                "rear_wheel_split_mps", "steering_feedback_rate_radps",
                "steering_command_gap_rad", "imu_yaw_rate_change_radps",
            }),
            "swap": ["rear_left_surface_mps", "rear_right_surface_mps"],
            "unchanged": "longitudinal/throttle/timing features",
            "target": "yaw-rate residual changes sign under reflection",
        },
        "input_contract": "causal sensor/actuator history only; GT is label/scoring only",
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_valid)),
        "training_runs": int(len(np.unique(train_ids))),
        "validation_runs": int(len(np.unique(valid_ids))),
        "baseline": _metric(base_error[selected]),
        "reflection_augmented_and_antisymmetrized": _metric(
            reflected_error[selected]),
        "paired_run_level": _paired_runs(
            valid_ids, base_error, reflected_error, selected),
        "new_gapfill_run_baseline": _metric(base_error[
            selected & (valid_ids == "openplane_yaw_exact_two_domain_gapfill_validation_20261008_r03")]),
        "new_gapfill_run_reflection": _metric(reflected_error[
            selected & (valid_ids == "openplane_yaw_exact_two_domain_gapfill_validation_20261008_r03")]),
        "by_causal_event": event_metrics,
        "by_speed_steering_mismatch_region": regions,
        "source_corpus_audit": corpus_audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True,
                   default=data._json_value) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "validation_predictions.npz",
        run_id=valid["run_id"], sample_time_ns=valid["sample_time_ns"],
        gt_speed_mps=valid["gt_speed"], steering_rad=steering,
        wheel_speed_mps=valid["wheel_speed"], yaw_target_residual=y_valid,
        baseline_residual=baseline, reflection_residual=reflected,
        response_event=valid["response_event"], causal_event=valid_event,
    )
    print(f"wrote {output_dir / 'report.json'}", flush=True)
    return result


if __name__ == "__main__":
    run()
