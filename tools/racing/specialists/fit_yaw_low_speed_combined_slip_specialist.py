#!/usr/bin/env python3
"""Fit a research-only yaw specialist for low-speed, high-steer slip.

The domain is selected from competition-visible odometry, encoders, steering,
throttle and IMU only. Simulator truth is the next-yaw training target and
offline score, never an input or selector. Whole runs retain their train or
validation split; test/final-test data are not admitted.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
                  "yaw_low_speed_combined_slip_v1")
V2_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "sensor_only_yaw_regime_atlas_command_intent_v2")
V4_DIR = (ROOT / "live_runs/racing_model_diagnostics_20261008/"
          "sensor_only_yaw_regime_atlas_command_age_v4")
LAGS = (0, 1, 2, 4)
EVENTS = ("hold", "turn_in", "unwind", "reversal")
SELECTOR = {
    "rear_wheel_mean_speed_mps": [0.0, 1.5],
    "absolute_steering_feedback_rad_min": 0.45,
    "throttle_feedback_norm_max": 0.02,
    "throttle_command_norm": [0.07, 0.12],
    "absolute_current_imu_yaw_rate_radps": [1.2, 2.2],
}
ESTIMATOR = {
    "n_estimators": 80,
    "max_depth": 9,
    "min_samples_leaf": 6,
    "max_features": 0.8,
    "random_state": 20261008,
}


def _features(seq: np.ndarray, attitude: np.ndarray,
              ages: tuple[np.ndarray, ...], k: int) -> tuple[
                  np.ndarray, np.ndarray, str, int, int,
                  np.ndarray, np.ndarray, float, float]:
    history = np.concatenate([
        np.concatenate((seq[k - lag, :9], attitude[k - lag, (0, 2)]))
        for lag in LAGS
    ]).astype(np.float32, copy=False)
    wheel_mean = 0.5 * float(seq[k, 2] + seq[k, 3])
    previous_wheel_mean = 0.5 * float(seq[k - 1, 2] + seq[k - 1, 3])
    base_derived = np.asarray((
        wheel_mean,
        float(seq[k, 2] - seq[k, 3]),
        (wheel_mean - previous_wheel_mean) / 0.025,
        (float(seq[k, 0]) - float(seq[k - 1, 0])) / 0.025,
        (float(seq[k, 1]) - float(seq[k - 1, 1])) / 0.025,
        float(seq[k, 7] - seq[k, 0]),
        float(seq[k, 8] - seq[k, 1]),
        float(seq[k, 6] - seq[k - 1, 6]),
    ), dtype=np.float32)
    age_derived = np.asarray((
        ages[0][k], ages[1][k], ages[2][k], ages[3][k],
        ages[1][k] - ages[0][k], ages[3][k] - ages[2][k],
    ), dtype=np.float32)
    x2 = np.concatenate((history, base_derived))
    x4 = np.concatenate((x2, age_derived))
    event = atlas._event(seq, k, "command_intent")
    event_one_hot = np.asarray([event == name for name in EVENTS],
                               dtype=np.float32)
    # Simulator-truth motion arrays are deliberately absent from predictors
    # and selectors. A second feature set tests whether longitudinal IMU
    # acceleration can correct the known 100-ms rolling-wheel-speed lag.
    candidate_x = np.concatenate((x4, event_one_hot))
    ax_window = seq[k - 4:k + 1, 4]
    mean_ax_100ms = float(np.mean(0.5 * (ax_window[:-1] + ax_window[1:])))
    # For a 100-ms interval-average speed under roughly constant acceleration,
    # v_end ~= v_average + a*T/2. This is a sensor-only candidate feature,
    # not an assumed vehicle law; held-out runs decide whether it helps.
    accel_corrected_wheel_speed = wheel_mean + 0.05 * mean_ax_100ms
    accel_x = np.concatenate((candidate_x, np.asarray((
        mean_ax_100ms, accel_corrected_wheel_speed), dtype=np.float32)))
    wheel_cell = int(wheel_mean // atlas.SPEED_BIN_MPS)
    steering_cell = int(atlas._steer_cell(
        np.asarray([seq[k, 0]], dtype=np.float32))[0])
    return (x2, x4, event, wheel_cell, steering_cell, candidate_x,
            accel_x, mean_ax_100ms, accel_corrected_wheel_speed)


def _collect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    by_source: dict[str, list[Any]] = defaultdict(list)
    for series in admitted:
        if series.split not in ("train", "validation"):
            raise ValueError(f"unexpected split admitted: {series.run_id} {series.split}")
        by_source[str(series.source)].append(series)

    rows: list[dict[str, Any]] = []
    for source, series_rows in by_source.items():
        with np.load(ROOT / source, allow_pickle=False) as archive:
            run_ids = archive["run_ids"].astype(str).tolist()
            sensors = np.asarray(archive["sensor_frames"], dtype=np.float32)
            sensor_valid = np.asarray(archive["sensor_valid"], dtype=bool)
            attitude = np.asarray(archive["imu_attitude_frames"], dtype=np.float32)
            attitude_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
            rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float32)
            bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
            sequence_run = np.asarray(archive["sequence_run_index"], dtype=np.int64)

            for series in series_rows:
                if series.run_id not in run_ids:
                    raise ValueError(f"{series.run_id}: missing from archive {source}")
                run_index = run_ids.index(series.run_id)
                for sequence_id in np.flatnonzero(sequence_run == run_index):
                    begin, end = map(int, bounds[int(sequence_id)])
                    seq = sensors[begin:end]
                    att = attitude[begin:end]
                    truth = rigid[begin:end]
                    sv = sensor_valid[begin:end]
                    av = attitude_valid[begin:end]
                    if (len(seq) <= max(LAGS) + 1
                            or not np.isfinite(seq).all()
                            or not np.isfinite(att).all()
                            or not np.isfinite(truth).all()):
                        continue
                    ages = (
                        atlas._age_since_change(seq[:, 7], 0.005),
                        atlas._age_since_change(seq[:, 0], 0.005),
                        atlas._age_since_change(seq[:, 8], 0.005),
                        atlas._age_since_change(seq[:, 1], 0.005),
                    )
                    for k in range(max(LAGS), len(seq) - 1):
                        if not all(sv[k - lag] and av[k - lag] for lag in LAGS):
                            continue
                        wheel = 0.5 * float(seq[k, 2] + seq[k, 3])
                        steering = float(seq[k, 0])
                        throttle_feedback = float(seq[k, 1])
                        throttle_command = float(seq[k, 8])
                        yaw = float(seq[k, 6])
                        if not (
                            SELECTOR["rear_wheel_mean_speed_mps"][0]
                            <= wheel
                            <= SELECTOR["rear_wheel_mean_speed_mps"][1]
                            and abs(steering)
                            >= SELECTOR["absolute_steering_feedback_rad_min"]
                            and throttle_feedback
                            <= SELECTOR["throttle_feedback_norm_max"]
                            and SELECTOR["throttle_command_norm"][0]
                            <= throttle_command
                            <= SELECTOR["throttle_command_norm"][1]
                            and SELECTOR["absolute_current_imu_yaw_rate_radps"][0]
                            <= abs(yaw)
                            <= SELECTOR["absolute_current_imu_yaw_rate_radps"][1]
                        ):
                            continue

                        (x2, x4, event, wheel_cell, steering_cell,
                         candidate_x, accel_x, mean_ax_100ms,
                         accel_corrected_wheel_speed) = _features(
                             seq, att, ages, k)
                        rows.append({
                            "run_id": series.run_id,
                            "split": series.split,
                            "x2": x2,
                            "x4": x4,
                            "x": candidate_x,
                            "x_accel": accel_x,
                            "wheel_cell": wheel_cell,
                            "steering_cell": steering_cell,
                            "event": event,
                            "current_yaw": yaw,
                            "next_gt_yaw": float(truth[k + 1, 12]),
                            "gt_speed_for_audit": float(np.hypot(
                                truth[k, 7], truth[k, 8])),
                            "wheel_speed": wheel,
                            "mean_ax_100ms": mean_ax_100ms,
                            "accel_corrected_wheel_speed": accel_corrected_wheel_speed,
                            "wheel_minus_gt_speed_for_audit": (
                                wheel - float(np.hypot(truth[k, 7], truth[k, 8]))),
                            "steering": steering,
                            "throttle_feedback": throttle_feedback,
                            "throttle_command": throttle_command,
                        })
    return rows, source_audit


def _predict_atlas(rows: list[dict[str, Any]], bundle: dict[str, Any],
                   feature_key: str) -> np.ndarray:
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    groups: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["wheel_cell"], row["steering_cell"], row["event"])].append(index)
    for (wheel_cell, steering_cell, event), indices in groups.items():
        model = bundle["local_expert_models"].get(
            (wheel_cell, steering_cell, event),
            bundle["global_event_models"].get(event))
        if model is None:
            continue
        x = np.stack([rows[index][feature_key] for index in indices])
        predictions[indices] = np.asarray(
            [rows[index]["current_yaw"] for index in indices]
        ) + model.predict(x)
    return predictions


def _balanced_weights(run_ids: np.ndarray) -> np.ndarray:
    ids, counts = np.unique(run_ids.astype(str), return_counts=True)
    count_by_id = dict(zip(ids.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_id[str(run_id)] for run_id in run_ids])
    return weights * (len(ids) / weights.sum())


def _metrics(error: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(error, dtype=np.float64))
    return {
        "samples": int(len(absolute)),
        "rmse_radps": float(np.sqrt(np.mean(absolute ** 2))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_below_0p1": float(np.mean(absolute < 0.1)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _run_metrics(rows: list[dict[str, Any]], errors: np.ndarray
                 ) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors):
        grouped[row["run_id"]].append(float(error))
    return {run_id: _metrics(np.asarray(values))
            for run_id, values in sorted(grouped.items())}


def _subgroup_metrics(rows: list[dict[str, Any]],
                      errors_by_method: dict[str, np.ndarray]
                      ) -> dict[str, dict[str, Any]]:
    """Report error by causal sensor-side regime, never GT-derived bins."""
    labels = {
        "event": np.asarray([row["event"] for row in rows]),
        "steering_sign": np.asarray([
            "left" if row["steering"] > 0.0 else "right" for row in rows]),
        "steering_magnitude": np.asarray([
            "0.45_to_0.475" if abs(row["steering"]) < 0.475
            else "0.475_to_0.525" for row in rows]),
        "rear_wheel_speed": np.asarray([
            "0_to_0.5" if row["wheel_speed"] < 0.5 else
            "0.5_to_1.0" if row["wheel_speed"] < 1.0 else "1.0_to_1.5"
            for row in rows]),
        "diagnostic_acceleration_correction": np.asarray([
            "below_minus0p1" if row["accel_corrected_wheel_speed"]
            - row["wheel_speed"] < -0.1 else
            "minus0p1_to_plus0p1" if abs(
                row["accel_corrected_wheel_speed"] - row["wheel_speed"])
            <= 0.1 else "above_plus0p1" for row in rows]),
        "throttle_command": np.asarray([
            "0.07_to_0.09" if row["throttle_command"] < 0.09
            else "0.09_to_0.12" for row in rows]),
        # These two axes use simulator truth only for post-fit diagnosis.
        # They are deliberately absent from the feature vector and selector.
        "diagnostic_gt_speed": np.asarray([
            f"{int(row['gt_speed_for_audit'])}_to_"
            f"{int(row['gt_speed_for_audit']) + 1}mps"
            for row in rows]),
        "diagnostic_wheel_minus_gt_speed": np.asarray([
            "below_minus1" if row["wheel_minus_gt_speed_for_audit"] < -1.0
            else "minus1_to_minus0p5" if row["wheel_minus_gt_speed_for_audit"] < -0.5
            else "minus0p5_to_plus0p5" if row["wheel_minus_gt_speed_for_audit"] < 0.5
            else "plus0p5_to_plus1" if row["wheel_minus_gt_speed_for_audit"] < 1.0
            else "above_plus1" for row in rows]),
    }
    result: dict[str, dict[str, Any]] = {}
    for axis, values in labels.items():
        result[axis] = {}
        for label in sorted(set(values.tolist())):
            mask = values == label
            result[axis][label] = {
                method: _metrics(error[mask])
                for method, error in errors_by_method.items()
            }
    return result


def _worst_examples(rows: list[dict[str, Any]], prediction: np.ndarray,
                    error: np.ndarray, count: int = 20
                    ) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(error))[-count:][::-1]
    return [{
        "run_id": rows[i]["run_id"],
        "rear_wheel_mean_mps": rows[i]["wheel_speed"],
        "steering_feedback_rad": rows[i]["steering"],
        "throttle_feedback_norm": rows[i]["throttle_feedback"],
        "throttle_command_norm": rows[i]["throttle_command"],
        "current_imu_yaw_rate_radps": rows[i]["current_yaw"],
        "next_gt_yaw_rate_radps": rows[i]["next_gt_yaw"],
        "gt_speed_mps_for_audit_only": rows[i]["gt_speed_for_audit"],
        "wheel_minus_gt_speed_mps_for_audit_only": (
            rows[i]["wheel_minus_gt_speed_for_audit"]),
        "mean_imu_ax_last_100ms_mps2": rows[i]["mean_ax_100ms"],
        "accel_corrected_wheel_speed_mps": (
            rows[i]["accel_corrected_wheel_speed"]),
        "prediction_radps": float(prediction[i]),
        "absolute_error_radps": float(abs(error[i])),
    } for i in order]


def fit(output: Path) -> dict[str, Any]:
    output = output.resolve()
    rows, source_audit = _collect()
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    if (len(train) < 100 or len(validation) < 30
            or len({row["run_id"] for row in train}) < 3
            or len({row["run_id"] for row in validation}) < 2):
        raise ValueError("combined-slip regime lacks independent train/validation support")

    x_train = np.stack([row["x"] for row in train])
    x_accel_train = np.stack([row["x_accel"] for row in train])
    target_train = np.asarray([
        row["next_gt_yaw"] - row["current_yaw"] for row in train
    ], dtype=np.float32)
    run_ids = np.asarray([row["run_id"] for row in train], dtype="U128")
    estimator = ExtraTreesRegressor(**ESTIMATOR, n_jobs=-1)
    estimator.fit(x_train, target_train, sample_weight=_balanced_weights(run_ids))
    accel_estimator = ExtraTreesRegressor(**ESTIMATOR, n_jobs=-1)
    accel_estimator.fit(
        x_accel_train, target_train, sample_weight=_balanced_weights(run_ids))

    v2 = joblib.load(V2_DIR / "sensor_only_yaw_regime_atlas.joblib")
    v4 = joblib.load(V4_DIR / "sensor_only_yaw_regime_atlas.joblib")
    val_current_yaw = np.asarray([row["current_yaw"] for row in validation])
    val_truth = np.asarray([row["next_gt_yaw"] for row in validation])
    val_candidate = val_current_yaw + estimator.predict(
        np.stack([row["x"] for row in validation]))
    val_accel_candidate = val_current_yaw + accel_estimator.predict(
        np.stack([row["x_accel"] for row in validation]))
    val_v2 = _predict_atlas(validation, v2, "x2")
    val_v4 = _predict_atlas(validation, v4, "x4")

    scored: dict[str, np.ndarray] = {
        "sensor_only_atlas_v2": val_v2 - val_truth,
        "sensor_only_command_age_atlas_v4": val_v4 - val_truth,
        "sensor_only_combined_slip_specialist": val_candidate - val_truth,
        "sensor_only_accel_corrected_wheel_proxy": (
            val_accel_candidate - val_truth),
    }
    candidate_errors = {
        name: scored[name]
        for name in ("sensor_only_atlas_v2",
                     "sensor_only_combined_slip_specialist",
                     "sensor_only_accel_corrected_wheel_proxy")
    }
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "yaw_low_speed_combined_slip_specialist.joblib"
    joblib.dump({
        "research_only": True,
        "selector": SELECTOR,
        "input_features": [
            *v4["contract"]["features"],
            *[f"event_{name}" for name in EVENTS],
        ],
        "estimator": ESTIMATOR,
        "model": estimator,
        "accel_corrected_model": accel_estimator,
        "accel_corrected_input_features": [
            *v4["contract"]["features"],
            *[f"event_{name}" for name in EVENTS],
            "mean_imu_ax_last_100ms_mps2",
            "wheel_mean_plus_0p05m_times_mean_ax_mps",
        ],
        "training_runs": sorted({row["run_id"] for row in train}),
        "split_policy": "whole-run train/validation only; test/final-test sealed",
        "ground_truth_policy": "next yaw target and offline score only",
    }, model_path)

    validation_runs = sorted({row["run_id"] for row in validation})
    report = {
        "title": "Low-speed, high-steer combined-slip one-step yaw specialist",
        "research_only": True,
        "production_integrated": False,
        "sample_period_s": 0.025,
        "selector": SELECTOR,
        "input_features": [
            *v4["contract"]["features"],
            *[f"event_{name}" for name in EVENTS],
        ],
        "accel_corrected_input_features": [
            *v4["contract"]["features"],
            *[f"event_{name}" for name in EVENTS],
            "mean_imu_ax_last_100ms_mps2",
            "wheel_mean_plus_0p05m_times_mean_ax_mps",
        ],
        "estimator": ESTIMATOR,
        "training": {
            "samples": len(train),
            "runs": sorted({row["run_id"] for row in train}),
            "score": _metrics(target_train),
            "run_balanced_weights": True,
        },
        "validation": {
            "samples": len(validation),
            "runs": validation_runs,
            "metrics": {
                name: _metrics(error) for name, error in scored.items()
            },
            "per_run": {
                name: _run_metrics(validation, error)
                for name, error in scored.items()
            },
            "sensor_only_subgroups": _subgroup_metrics(
                validation, candidate_errors),
            "worst_20_candidate_errors": _worst_examples(
                validation, val_candidate,
                scored["sensor_only_combined_slip_specialist"]),
            "worst_20_accel_corrected_errors": _worst_examples(
                validation, val_accel_candidate,
                scored["sensor_only_accel_corrected_wheel_proxy"]),
        },
        "source_audit": source_audit,
        "artifact": str(model_path),
        "limitations": [
            "Only the explicitly selected low-speed combined-slip domain is scored.",
            "Validation contains three independent runs; treat run-level spread as material.",
            "This one-step result is not a recursive rollout or full-lap validation.",
            "No runtime odometry/MPC integration is authorized by this fit alone.",
        ],
    }
    report_path = output / "yaw_low_speed_combined_slip_report.json"
    report_path.write_text(json.dumps(
        report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8")
    print(f"training: {len(train)} rows / {len(report['training']['runs'])} whole runs")
    print(f"validation: {len(validation)} rows / {len(validation_runs)} whole runs")
    for name, error in scored.items():
        metrics = _metrics(error)
        print(f"{name}: RMSE={metrics['rmse_radps']:.4f}, "
              f"p95={metrics['p95_abs_radps']:.4f}, "
              f"max={metrics['max_abs_radps']:.4f}, "
              f"<0.1={metrics['fraction_abs_error_below_0p1']:.3f}")
    print(f"wrote {model_path} and {report_path}")
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fit(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
