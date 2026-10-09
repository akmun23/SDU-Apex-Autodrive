#!/usr/bin/env python3
"""Test whether reconstructed tire-slip proxies explain held-out yaw errors.

All simulator-truth-derived slip features in this script are diagnostic
oracles, not runtime inputs. The comparison asks whether those physical
quantities contain useful information that a future sensor-only observer
could attempt to estimate. Only complete exact-two-packet steering windows
from training captures and one independent validation capture are admitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import evaluate_yaw_full_spectrum_exact_two as evaluator
    import evaluate_yaw_predicted_next_steering as causal
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as causal
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/yaw_slip_oracle_r02")
TRAIN_IDS = (evaluator.FULL_TRAIN_ID,
             evaluator.FULL_TRAIN_REPLICATION_ID)
VALIDATION_ID = evaluator.FULL_VALIDATION_ID
WHEELBASE_M = 0.324
TRACK_WIDTH_M = 0.236
WHEEL_RADIUS_M = 0.059
COM_X_M = 0.15532
HALF_TRACK_M = TRACK_WIDTH_M / 2.0
MIN_LONGITUDINAL_SPEED_MPS = 0.5
SENSOR_WIDTH = len(atlas.OBSERVATION_NAMES)
SLIP_NAMES = (
    "rear_left_longitudinal_slip", "rear_right_longitudinal_slip",
    "front_left_lateral_slip", "front_right_lateral_slip",
    "rear_left_lateral_slip", "rear_right_lateral_slip",
)


def _ackermann(steering: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wheel angles for positive-left steering and left/right wheel labels."""
    tangent = np.tan(np.asarray(steering, dtype=np.float64))
    numerator = 2.0 * WHEELBASE_M * tangent
    # The left wheel is inside on a positive (left) turn.
    left = np.arctan2(numerator, 2.0 * WHEELBASE_M - TRACK_WIDTH_M * tangent)
    right = np.arctan2(numerator, 2.0 * WHEELBASE_M + TRACK_WIDTH_M * tangent)
    return left, right


def _slip_features(data: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Reconstruct per-wheel kinematic slip from GT body motion and sensors."""
    state = np.asarray(data["gt_body_state"], dtype=np.float64)
    u, v = state[:, 0], state[:, 1]
    yaw_rate = np.asarray(data["gt_yaw_rate"], dtype=np.float64)
    steering = np.asarray(data["steering"], dtype=np.float64)
    # `data["x"]` starts with lag-0 sensor channels, then older 25-ms-grid
    # history. Rear encoder fields are surface speed, not raw tick rate.
    sensor = np.asarray(data["x"], dtype=np.float64)[:, :SENSOR_WIDTH]
    rear_surface_speed = sensor[:, 2:4]
    front_left_angle, front_right_angle = _ackermann(steering)

    # Wheel coordinates relative to CG; simulator body velocity labels are at
    # the CG. Rigid-body point velocity is v_i = v_CG + omega x r_i.
    wheel_xy = (
        (-COM_X_M, +HALF_TRACK_M),
        (-COM_X_M, -HALF_TRACK_M),
        (WHEELBASE_M - COM_X_M, +HALF_TRACK_M),
        (WHEELBASE_M - COM_X_M, -HALF_TRACK_M),
    )
    wheel_angles = (np.zeros_like(steering), np.zeros_like(steering),
                    front_left_angle, front_right_angle)
    longitudinal = []
    lateral = []
    for (x_m, y_m), wheel_angle in zip(wheel_xy, wheel_angles):
        vx_body = u - yaw_rate * y_m
        vy_body = v + yaw_rate * x_m
        cosine = np.cos(wheel_angle)
        sine = np.sin(wheel_angle)
        vx_wheel = cosine * vx_body + sine * vy_body
        vy_wheel = -sine * vx_body + cosine * vy_body
        longitudinal.append(vx_wheel)
        lateral.append(vy_wheel)
    vx = np.column_stack(longitudinal)
    vy = np.column_stack(lateral)

    # Forward-motion rows only. This avoids the singularity at standstill and
    # matches the guide's Sx denominator for the tested forward regimes.
    valid_vx = vx > MIN_LONGITUDINAL_SPEED_MPS
    sy = np.divide(vy, np.abs(vx), out=np.zeros_like(vy), where=valid_vx)
    sx = np.divide(rear_surface_speed - vx[:, :2], vx[:, :2],
                   out=np.zeros_like(rear_surface_speed), where=valid_vx[:, :2])
    # Feature contract: rear Sx, then front Sy, then rear Sy.
    lateral_order = (2, 3, 0, 1)
    raw = np.column_stack((sx, sy[:, lateral_order]))
    valid = np.column_stack((valid_vx[:, :2], valid_vx[:, lateral_order]))
    features = np.column_stack((raw, valid.astype(np.float64))).astype(np.float32)

    # Landmarks in the published piecewise tire curves. These bins mean
    # "below peak / peak-to-asymptote / beyond asymptote"; they are not direct
    # force measurements because no wheel-force or normal-load labels are in
    # these bags.
    rear_sx_abs_mean = np.mean(np.abs(sx), axis=1)
    front_sy_abs_max = np.max(np.abs(sy[:, 2:4]), axis=1)
    rear_sy_abs_max = np.max(np.abs(sy[:, :2]), axis=1)
    rear_long_valid = np.all(valid_vx[:, :2], axis=1)
    front_lat_valid = np.all(valid_vx[:, 2:4], axis=1)
    rear_lat_valid = np.all(valid_vx[:, :2], axis=1)
    summary = {
        "validity": {
            "rear_longitudinal_fraction": float(np.mean(rear_long_valid)),
            "front_lateral_fraction": float(np.mean(front_lat_valid)),
            "rear_lateral_fraction": float(np.mean(rear_lat_valid)),
        },
        "landmark_occupancy": {
            "rear_longitudinal_abs_slip": _landmark_counts(
                rear_sx_abs_mean, rear_long_valid, 0.15, 0.25),
            "front_lateral_abs_slip_max": _landmark_counts(
                front_sy_abs_max, front_lat_valid, 0.01, 0.10),
            "rear_lateral_abs_slip_max": _landmark_counts(
                rear_sy_abs_max, rear_lat_valid, 0.01, 0.10),
        },
    }
    return features, summary


def _landmark_counts(values: np.ndarray, valid: np.ndarray,
                     peak: float, asymptote: float) -> dict[str, int]:
    selected = np.asarray(values)[np.asarray(valid, dtype=bool)]
    return {
        "below_peak": int(np.count_nonzero(selected < peak)),
        "peak_to_asymptote": int(np.count_nonzero(
            (selected >= peak) & (selected < asymptote))),
        "at_or_beyond_asymptote": int(np.count_nonzero(selected >= asymptote)),
        "valid_rows": int(len(selected)),
    }


def _combine(runs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = tuple(next(iter(runs.values())).keys())
    return {key: np.concatenate([runs[run_id][key] for run_id in sorted(runs)])
            for key in keys}


def _fit_with_oracle_features(data: dict[str, np.ndarray],
                              extra: np.ndarray,
                              mirrored_extra: np.ndarray
                              ) -> ExtraTreesRegressor:
    base = evaluator._model_features(data)
    sensor = base[:, :-len(evaluator.EVENTS)]
    event = base[:, -len(evaluator.EVENTS):]
    fit_x = np.column_stack((base, extra))

    # Reflection swaps left/right wheels; lateral slip changes sign. Keep the
    # same augmentation and phase-balanced weights as the sensor-only model.
    mirror_input = np.column_stack((
        sensor, np.zeros((len(sensor), 4), dtype=np.float32)))
    mirror_sensor = causal._mirror_features(mirror_input)[:, :-4]
    mirrored_x = np.column_stack((mirror_sensor, event, mirrored_extra))

    phase_ids = data["phase_id"].astype(str)
    x = np.concatenate((fit_x, mirrored_x), axis=0)
    y = np.concatenate((data["y"], -data["y"]), axis=0)
    weights = causal._phase_weights(np.concatenate((phase_ids, phase_ids)))
    return ExtraTreesRegressor(**evaluator.MODEL).fit(
        x, y, sample_weight=weights)


def _mirror_slip_features(slip: np.ndarray) -> np.ndarray:
    mirrored = slip.copy()
    # Raw values: rear-left/right Sx, front-left/right Sy, rear-left/right Sy;
    # then the matching six validity indicators.
    mirrored[:, 0], mirrored[:, 1] = slip[:, 1], slip[:, 0]
    for left, right in ((2, 3), (4, 5)):
        mirrored[:, left] = -slip[:, right]
        mirrored[:, right] = -slip[:, left]
    for left, right in ((6, 7), (8, 9), (10, 11)):
        mirrored[:, left], mirrored[:, right] = slip[:, right], slip[:, left]
    return mirrored


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "mae_radps": float(np.mean(absolute)) if len(error) else None,
        "p95_abs_radps": float(np.quantile(absolute, 0.95)) if len(error) else None,
        "max_abs_radps": float(np.max(absolute)) if len(error) else None,
        "samples_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
        "fraction_within_0p1_radps": float(np.mean(absolute <= 0.1)) if len(error) else None,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _group_metrics(data: dict[str, np.ndarray], baseline_error: np.ndarray,
                   state_error: np.ndarray, slip_error: np.ndarray,
                   front_sy_max: np.ndarray,
                   rear_sx_mean: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, mask in (
        ("front_lateral_below_peak", front_sy_max < 0.01),
        ("front_lateral_peak_to_asymptote",
         (front_sy_max >= 0.01) & (front_sy_max < 0.10)),
        ("front_lateral_at_or_beyond_asymptote", front_sy_max >= 0.10),
        ("rear_longitudinal_below_peak", rear_sx_mean < 0.15),
        ("rear_longitudinal_peak_to_asymptote",
         (rear_sx_mean >= 0.15) & (rear_sx_mean < 0.25)),
        ("rear_longitudinal_at_or_beyond_asymptote", rear_sx_mean >= 0.25),
    ):
        result[name] = {
            "baseline": _metric(baseline_error[mask]),
            "GT_motion_state_oracle": _metric(state_error[mask]),
            "oracle_slip": _metric(slip_error[mask]),
        }
    by_event: dict[str, Any] = {}
    for event in evaluator.EVENTS:
        mask = data["phase_event"].astype(str) == event
        by_event[event] = {
            "baseline": _metric(baseline_error[mask]),
            "GT_motion_state_oracle": _metric(state_error[mask]),
            "oracle_slip": _metric(slip_error[mask]),
        }
    result["by_event"] = by_event
    return result


def _condition_bootstrap(data: dict[str, np.ndarray],
                         baseline_error: np.ndarray,
                         slip_error: np.ndarray) -> dict[str, Any]:
    """Paired condition-cluster bootstrap, not a run-level confidence interval."""
    ids = data["condition_id"].astype(str)
    unique = np.unique(ids)
    if len(unique) < 20:
        return {"conditions": int(len(unique)), "status": "insufficient groups"}
    base_mse = np.asarray([
        np.mean(baseline_error[ids == value] ** 2) for value in unique])
    slip_mse = np.asarray([
        np.mean(slip_error[ids == value] ** 2) for value in unique])
    rng = np.random.default_rng(20261009)
    draws = rng.integers(0, len(unique), size=(2000, len(unique)))
    base_rmse = np.sqrt(np.mean(base_mse[draws], axis=1))
    slip_rmse = np.sqrt(np.mean(slip_mse[draws], axis=1))
    delta = slip_rmse - base_rmse
    return {
        "conditions": int(len(unique)),
        "estimand": "equal-weight condition macro-RMSE delta, oracle minus baseline",
        "delta_median_radps": float(np.median(delta)),
        "delta_95pct_interval_radps": [float(np.quantile(delta, 0.025)),
                                       float(np.quantile(delta, 0.975))],
        "interpretation": "within-one-run condition bootstrap; not run-level uncertainty",
    }


def run(output: Path) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    by_id = {series.run_id: series for series in admitted}
    required = set(TRAIN_IDS) | {VALIDATION_ID}
    missing = required - set(by_id)
    if missing:
        raise ValueError(f"required sources absent from safe archives: {sorted(missing)}")
    if any(by_id[run_id].split != "train" for run_id in TRAIN_IDS):
        raise ValueError("slip fit attempted to use a non-training run")
    if by_id[VALIDATION_ID].split != "validation":
        raise ValueError("held-out r02 is not marked validation")

    train_runs: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, Any] = {}
    slip_summaries: dict[str, Any] = {}
    slip_by_run: dict[str, np.ndarray] = {}
    validation: dict[str, np.ndarray] | None = None
    validation_slip: np.ndarray | None = None
    for run_id in (*TRAIN_IDS, VALIDATION_ID):
        data, audit = evaluator._collect_run(by_id[run_id], atlas.HISTORY_LAGS)
        slip, slip_summary = _slip_features(data)
        audits[run_id] = audit
        slip_summaries[run_id] = slip_summary
        if run_id == VALIDATION_ID:
            validation, validation_slip = data, slip
        else:
            train_runs[run_id] = data
            slip_by_run[run_id] = slip
    if validation is None or validation_slip is None:
        raise AssertionError("held-out run was not collected")
    train = _combine(train_runs)
    train_slip = np.concatenate([slip_by_run[run_id] for run_id in sorted(slip_by_run)])

    baseline = evaluator._fit(
        evaluator._model_features(train), train["y"],
        train["phase_id"].astype(str))
    train_state = np.column_stack((
        train["gt_body_state"][:, :2], train["gt_yaw_rate"])).astype(np.float32)
    validation_state = np.column_stack((
        validation["gt_body_state"][:, :2],
        validation["gt_yaw_rate"])).astype(np.float32)
    mirrored_train_state = train_state.copy()
    mirrored_train_state[:, 1:] *= -1.0
    mirrored_validation_state = validation_state.copy()
    mirrored_validation_state[:, 1:] *= -1.0
    state_model = _fit_with_oracle_features(
        train, train_state, mirrored_train_state)
    candidate = _fit_with_oracle_features(
        train, np.column_stack((train_state, train_slip)),
        np.column_stack((mirrored_train_state,
                         _mirror_slip_features(train_slip))))
    baseline_prediction = baseline.predict(evaluator._model_features(validation))
    state_prediction = state_model.predict(np.column_stack((
        evaluator._model_features(validation), validation_state)))
    oracle_prediction = candidate.predict(np.column_stack((
        evaluator._model_features(validation), validation_state,
        validation_slip)))
    baseline_error = baseline_prediction - validation["y"]
    state_error = state_prediction - validation["y"]
    oracle_error = oracle_prediction - validation["y"]

    front_sy_max = np.max(np.abs(validation_slip[:, 2:4]), axis=1)
    rear_sx_mean = np.mean(np.abs(validation_slip[:, :2]), axis=1)
    valid_front = np.all(validation_slip[:, 8:10] > 0.5, axis=1)
    valid_rear_long = np.all(validation_slip[:, 6:8] > 0.5, axis=1)
    # Keep undefined near-standstill rows out of slip-regime comparisons.
    front_sy_max[~valid_front] = np.nan
    rear_sx_mean[~valid_rear_long] = np.nan
    regime = _group_metrics(validation, baseline_error, state_error, oracle_error,
                            front_sy_max, rear_sx_mean)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "objective": "test whether GT-reconstructed per-wheel slip explains held-out one-step yaw residuals",
        "status": "diagnostic only; no production integration",
        "timebase_ms": 25,
        "admission": "exactly two contiguous command-to-feedback packets; 1/3 packet rows rejected",
        "train_run_ids": list(TRAIN_IDS),
        "heldout_run_id": VALIDATION_ID,
        "train_rows": int(len(train["y"])),
        "heldout_rows": int(len(validation["y"])),
        "slip_feature_order": list(SLIP_NAMES) + [f"valid_{name}" for name in SLIP_NAMES],
        "slip_formulas": {
            "longitudinal": "Sx=(rear_encoder_surface_speed - GT wheel-local Vx)/GT wheel-local Vx",
            "lateral": "Sy=GT wheel-local Vy/abs(GT wheel-local Vx)",
            "wheel_velocity": "rigid-body CG velocity plus yaw-rate cross wheel-position, rotated into Ackermann wheel frame",
            "validity_gate": f"forward wheel-local Vx > {MIN_LONGITUDINAL_SPEED_MPS} m/s",
        },
        "official_curve_landmarks": {
            "longitudinal_peak_slip": 0.15,
            "longitudinal_asymptote_slip": 0.25,
            "lateral_peak_slip": 0.01,
            "lateral_asymptote_slip": 0.10,
            "interpretation": "slip-coordinate landmarks only; bag lacks per-wheel force and normal-load labels",
        },
        "sensor_only_baseline": _metric(baseline_error),
        "GT_motion_state_oracle": _metric(state_error),
        "GT_motion_state_plus_slip_oracle": _metric(oracle_error),
        "oracle_minus_baseline_rmse_radps": float(
            np.sqrt(np.mean(oracle_error ** 2))
            - np.sqrt(np.mean(baseline_error ** 2))),
        "slip_increment_beyond_GT_state_rmse_radps": float(
            np.sqrt(np.mean(oracle_error ** 2))
            - np.sqrt(np.mean(state_error ** 2))),
        "condition_cluster_bootstrap": _condition_bootstrap(
            validation, baseline_error, oracle_error),
        "slip_increment_condition_cluster_bootstrap": _condition_bootstrap(
            validation, state_error, oracle_error),
        "by_slip_regime": regime,
        "slip_landmark_occupancy_by_run": slip_summaries,
        "run_audits": audits,
        "source_admission_audit": source_audit,
        "caveats": [
            "GT-derived slip is an oracle feature and cannot be consumed directly by competition odometry/MPC.",
            "Front longitudinal slip is unavailable because front wheel angular speeds are not in the bags.",
            "Reconstructed Sy and rear Sx are kinematic proxies; tire force, suspension load transfer, and internal spline output are not measured.",
            "Validation is one independently randomized full-grid run; the condition bootstrap does not replace independent-run uncertainty.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default)
        + "\n", encoding="utf-8")

    row_path = output / "heldout_rows.csv"
    import csv
    fields = (
        "condition_id", "event", "gt_speed_mps", "steering_rad",
        *SLIP_NAMES, "baseline_error_radps", "GT_state_error_radps",
        "oracle_state_plus_slip_error_radps",
    )
    with row_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(len(validation["y"])):
            slips = validation_slip[index, :6]
            writer.writerow((
                str(validation["condition_id"][index]),
                str(validation["phase_event"][index]),
                float(validation["gt_speed"][index]),
                float(validation["steering"][index]),
                *(float(value) for value in slips),
                float(baseline_error[index]), float(state_error[index]),
                float(oracle_error[index]),
            ))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps({
        "output": str(args.output.resolve()),
        "baseline": report["sensor_only_baseline"],
        "GT_motion_state_oracle": report["GT_motion_state_oracle"],
        "GT_motion_state_plus_slip_oracle": report["GT_motion_state_plus_slip_oracle"],
        "rmse_delta_radps": report["oracle_minus_baseline_rmse_radps"],
        "bootstrap": report["condition_cluster_bootstrap"],
    }, indent=2))


if __name__ == "__main__":
    main()
