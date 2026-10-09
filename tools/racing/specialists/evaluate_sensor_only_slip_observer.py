#!/usr/bin/env python3
"""Evaluate a causal sensor-only wheel-slip observer and yaw-feature transfer.

Simulator truth labels kinematic slip during offline training/scoring only.
The observer receives current/past sensors and actuator signals, never GT or
future measurements. Yaw training uses run-grouped out-of-fold slip estimates
to avoid training on in-sample observer predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import analyze_yaw_slip_regime_factor as slip_diagnostic
    import evaluate_yaw_full_spectrum_exact_two as evaluator
    import evaluate_yaw_predicted_next_steering as causal
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import analyze_yaw_slip_regime_factor as slip_diagnostic
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as causal
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / (
    "live_runs/racing_model_diagnostics_20261009/sensor_slip_observer_r02")
TRAIN_IDS = slip_diagnostic.TRAIN_IDS
VALIDATION_ID = slip_diagnostic.VALIDATION_ID
SLIP_WIDTH = len(slip_diagnostic.SLIP_NAMES)
FRONT_LATERAL_PEAK_SLIP = 0.01
SLIP_MODEL = {
    "n_estimators": 72,
    "max_depth": 12,
    "min_samples_leaf": 8,
    "max_features": 0.8,
    "n_jobs": 4,
    "random_state": 20261009,
}


def _fit_slip_observer(data: dict[str, np.ndarray],
                       target: np.ndarray) -> ExtraTreesRegressor:
    base = evaluator._model_features(data)
    sensor = base[:, :-len(evaluator.EVENTS)]
    event = base[:, -len(evaluator.EVENTS):]
    mirror_input = np.column_stack((
        sensor, np.zeros((len(sensor), 4), dtype=np.float32)))
    mirrored_sensor = causal._mirror_features(mirror_input)[:, :-4]
    mirrored_target = slip_diagnostic._mirror_slip_features(target)
    x = np.concatenate((base, np.column_stack((mirrored_sensor, event))), axis=0)
    y = np.concatenate((target, mirrored_target), axis=0)
    phase_ids = data["phase_id"].astype(str)
    weights = causal._phase_weights(np.concatenate((phase_ids, phase_ids)))
    return ExtraTreesRegressor(**SLIP_MODEL).fit(
        x, y, sample_weight=weights)


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
        "mae": float(np.mean(absolute)) if len(error) else None,
        "p95_absolute": float(np.quantile(absolute, 0.95)) if len(error) else None,
        "bias": float(np.mean(error)) if len(error) else None,
    }


def _slip_scores(truth: np.ndarray, prediction: np.ndarray
                 ) -> dict[str, Any]:
    values = truth[:, :SLIP_WIDTH]
    valid = truth[:, SLIP_WIDTH:] > 0.5
    result: dict[str, Any] = {}
    for index, name in enumerate(slip_diagnostic.SLIP_NAMES):
        mask = valid[:, index]
        error = prediction[mask, index] - values[mask, index]
        result[name] = _metric(error)

    front_valid = np.all(valid[:, 2:4], axis=1)
    front_peak_band = (np.max(np.abs(values[:, 2:4]), axis=1) >= 0.01) & (
        np.max(np.abs(values[:, 2:4]), axis=1) < 0.10)
    front_saturated = np.max(np.abs(values[:, 2:4]), axis=1) >= 0.10
    rear_valid = np.all(valid[:, :2], axis=1)
    rear_saturated = np.mean(np.abs(values[:, :2]), axis=1) >= 0.25

    def pair_error(mask: np.ndarray, indices: tuple[int, int]) -> np.ndarray:
        selected = np.flatnonzero(mask)
        if not len(selected):
            return np.empty(0)
        return (prediction[np.ix_(selected, indices)]
                - values[np.ix_(selected, indices)]).reshape(-1)

    result["front_lateral_by_curve_regime"] = {
        "peak_to_asymptote": _metric(pair_error(
            front_valid & front_peak_band, (2, 3))),
        "at_or_beyond_asymptote": _metric(pair_error(
            front_valid & front_saturated, (2, 3))),
        "below_peak": _metric(pair_error(
            front_valid & ~(front_peak_band | front_saturated), (2, 3))),
    }
    result["rear_longitudinal_by_curve_regime"] = {
        "at_or_beyond_asymptote": _metric(pair_error(
            rear_valid & rear_saturated, (0, 1))),
        "below_asymptote": _metric(pair_error(
            rear_valid & ~rear_saturated, (0, 1))),
    }
    result["validity_prediction_accuracy"] = float(np.mean(
        (prediction[:, SLIP_WIDTH:] >= 0.5) == valid))
    return result


def _yaw_metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    if not len(error):
        return {
            "samples": 0,
            "rmse_radps": None,
            "mae_radps": None,
            "p95_abs_radps": None,
            "max_abs_radps": None,
            "samples_over_0p1_radps": 0,
            "fraction_within_0p1_radps": None,
        }
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
        "fraction_within_0p1_radps": float(np.mean(absolute <= 0.1)),
    }


def _yaw_by_slip_regime(data: dict[str, np.ndarray], truth_slip: np.ndarray,
                        baseline_error: np.ndarray,
                        observer_error: np.ndarray) -> dict[str, Any]:
    values = truth_slip[:, :SLIP_WIDTH]
    valid = truth_slip[:, SLIP_WIDTH:] > 0.5
    front = np.max(np.abs(values[:, 2:4]), axis=1)
    rear = np.mean(np.abs(values[:, :2]), axis=1)
    masks = {
        "front_lateral_below_peak": np.all(valid[:, 2:4], axis=1) & (front < 0.01),
        "front_lateral_peak_to_asymptote": (
            np.all(valid[:, 2:4], axis=1) & (front >= 0.01) & (front < 0.10)),
        "front_lateral_at_or_beyond_asymptote": (
            np.all(valid[:, 2:4], axis=1) & (front >= 0.10)),
        "rear_longitudinal_below_peak": np.all(valid[:, :2], axis=1) & (rear < 0.15),
        "rear_longitudinal_peak_to_asymptote": (
            np.all(valid[:, :2], axis=1) & (rear >= 0.15) & (rear < 0.25)),
        "rear_longitudinal_at_or_beyond_asymptote": (
            np.all(valid[:, :2], axis=1) & (rear >= 0.25)),
    }
    result = {}
    for name, mask in masks.items():
        result[name] = {
            "samples": int(np.count_nonzero(mask)),
            "baseline": _yaw_metric(baseline_error[mask]) if np.any(mask) else None,
            "sensor_slip_observer": _yaw_metric(observer_error[mask]) if np.any(mask) else None,
        }
    for event in evaluator.EVENTS:
        event_mask = data["phase_event"].astype(str) == event
        result[f"event_{event}"] = {
            "samples": int(np.count_nonzero(event_mask)),
            "baseline": _yaw_metric(baseline_error[event_mask]),
            "sensor_slip_observer": _yaw_metric(observer_error[event_mask]),
        }
    return result


def _condition_cluster_delta(data: dict[str, np.ndarray],
                             reference_error: np.ndarray,
                             candidate_error: np.ndarray) -> dict[str, Any]:
    condition_ids = data["condition_id"].astype(str)
    unique = np.unique(condition_ids)
    reference_mse = np.asarray([
        np.mean(reference_error[condition_ids == value] ** 2)
        for value in unique])
    candidate_mse = np.asarray([
        np.mean(candidate_error[condition_ids == value] ** 2)
        for value in unique])
    rng = np.random.default_rng(20261009)
    draws = rng.integers(0, len(unique), size=(2000, len(unique)))
    delta = (np.sqrt(np.mean(candidate_mse[draws], axis=1))
             - np.sqrt(np.mean(reference_mse[draws], axis=1)))
    return {
        "conditions": int(len(unique)),
        "estimand": "equal-weight condition macro-RMSE delta, candidate minus reference",
        "delta_median_radps": float(np.median(delta)),
        "delta_95pct_interval_radps": [float(np.quantile(delta, 0.025)),
                                       float(np.quantile(delta, 0.975))],
        "interpretation": "within-one-run condition bootstrap; not run-level uncertainty",
    }


def _front_lateral_gate(data: dict[str, np.ndarray], truth_slip: np.ndarray,
                        predicted_slip: np.ndarray,
                        baseline_error: np.ndarray,
                        observer_error: np.ndarray) -> dict[str, Any]:
    predicted_front = np.max(np.abs(predicted_slip[:, 2:4]), axis=1)
    gate = predicted_front >= FRONT_LATERAL_PEAK_SLIP
    gated_error = np.where(gate, observer_error, baseline_error)
    truth_values = truth_slip[:, :SLIP_WIDTH]
    truth_valid = truth_slip[:, SLIP_WIDTH:] > 0.5
    true_front = np.max(np.abs(truth_values[:, 2:4]), axis=1)
    valid_front = np.all(truth_valid[:, 2:4], axis=1)
    true_high = valid_front & (true_front >= FRONT_LATERAL_PEAK_SLIP)
    true_low = valid_front & ~true_high
    selected = np.flatnonzero(valid_front)
    true_high_selected = true_high[selected]
    gate_selected = gate[selected]
    true_positives = int(np.count_nonzero(gate_selected & true_high_selected))
    gate_count = int(np.count_nonzero(gate_selected))
    high_count = int(np.count_nonzero(true_high_selected))

    result: dict[str, Any] = {
        "rule": "use sensor-slip yaw candidate iff max(abs(predicted front-left/front-right Sy)) >= 0.01",
        "threshold_source": "published lateral tire-curve extremum; fixed, not fitted on this run",
        "selected_rows": int(np.count_nonzero(gate)),
        "selected_fraction": float(np.mean(gate)),
        "truth_high_regime_precision": true_positives / gate_count if gate_count else None,
        "truth_high_regime_recall": true_positives / high_count if high_count else None,
        "sensor_only_baseline": _yaw_metric(baseline_error),
        "ungated_sensor_slip_candidate": _yaw_metric(observer_error),
        "front_slip_gated_candidate": _yaw_metric(gated_error),
        "gated_minus_baseline_condition_bootstrap": _condition_cluster_delta(
            data, baseline_error, gated_error),
        "gated_minus_ungated_condition_bootstrap": _condition_cluster_delta(
            data, observer_error, gated_error),
        "true_front_slip_strata": {},
    }
    strata = {
        "below_peak": true_low & (true_front < FRONT_LATERAL_PEAK_SLIP),
        "peak_to_asymptote": (valid_front
            & (true_front >= FRONT_LATERAL_PEAK_SLIP) & (true_front < 0.10)),
        "at_or_beyond_asymptote": valid_front & (true_front >= 0.10),
    }
    for name, mask in strata.items():
        result["true_front_slip_strata"][name] = {
            "samples": int(np.count_nonzero(mask)),
            "baseline": _yaw_metric(baseline_error[mask]) if np.any(mask) else None,
            "ungated": _yaw_metric(observer_error[mask]) if np.any(mask) else None,
            "gated": _yaw_metric(gated_error[mask]) if np.any(mask) else None,
        }
    for event in evaluator.EVENTS:
        mask = data["phase_event"].astype(str) == event
        result.setdefault("by_event", {})[event] = {
            "baseline": _yaw_metric(baseline_error[mask]),
            "gated": _yaw_metric(gated_error[mask]),
        }
    return result


def run(output: Path, validation_id: str = VALIDATION_ID) -> dict[str, Any]:
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    by_id = {series.run_id: series for series in admitted}
    required = set(TRAIN_IDS) | {validation_id}
    if required - set(by_id):
        raise ValueError(f"missing safe run sources: {sorted(required - set(by_id))}")
    if any(by_id[run_id].split != "train" for run_id in TRAIN_IDS):
        raise ValueError("slip observer training source is not training split")
    if by_id[validation_id].split != "validation":
        raise ValueError(f"{validation_id} is not admitted as held-out validation")

    train_runs: dict[str, dict[str, np.ndarray]] = {}
    slip_by_run: dict[str, np.ndarray] = {}
    audits: dict[str, Any] = {}
    validation: dict[str, np.ndarray] | None = None
    validation_slip: np.ndarray | None = None
    for run_id in (*TRAIN_IDS, validation_id):
        data, audit = evaluator._collect_run(by_id[run_id], atlas.HISTORY_LAGS)
        truth_slip, _ = slip_diagnostic._slip_features(data)
        audits[run_id] = audit
        if run_id == validation_id:
            validation, validation_slip = data, truth_slip
        else:
            train_runs[run_id] = data
            slip_by_run[run_id] = truth_slip
    assert validation is not None and validation_slip is not None
    train = slip_diagnostic._combine(train_runs)
    train_slip = np.concatenate(
        [slip_by_run[run_id] for run_id in sorted(slip_by_run)])

    # Leave-one-whole-run-out predictions build the yaw model's training
    # features. No slip labels from a row are used to predict that row.
    oof_by_run: dict[str, np.ndarray] = {}
    for heldout_id in sorted(train_runs):
        fit_ids = [run_id for run_id in sorted(train_runs)
                   if run_id != heldout_id]
        fit_data = slip_diagnostic._combine({
            run_id: train_runs[run_id] for run_id in fit_ids})
        fit_slip = np.concatenate([slip_by_run[run_id] for run_id in fit_ids])
        observer = _fit_slip_observer(fit_data, fit_slip)
        oof_by_run[heldout_id] = observer.predict(
            evaluator._model_features(train_runs[heldout_id])).astype(np.float32)
    train_oof_slip = np.concatenate(
        [oof_by_run[run_id] for run_id in sorted(oof_by_run)])

    final_observer = _fit_slip_observer(train, train_slip)
    validation_pred_slip = final_observer.predict(
        evaluator._model_features(validation)).astype(np.float32)
    train_slip_score = _slip_scores(train_slip, train_oof_slip)
    validation_slip_score = _slip_scores(validation_slip, validation_pred_slip)

    baseline = evaluator._fit(
        evaluator._model_features(train), train["y"],
        train["phase_id"].astype(str))
    sensor_slip_yaw = slip_diagnostic._fit_with_oracle_features(
        train, train_oof_slip,
        slip_diagnostic._mirror_slip_features(train_oof_slip))
    baseline_error = baseline.predict(evaluator._model_features(validation)) - validation["y"]
    observer_error = sensor_slip_yaw.predict(np.column_stack((
        evaluator._model_features(validation), validation_pred_slip))) - validation["y"]
    gated_yaw = _front_lateral_gate(
        validation, validation_slip, validation_pred_slip,
        baseline_error, observer_error)

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "objective": "test whether sensor-only estimates of per-wheel slip improve one-step yaw prediction",
        "status": "research evaluation only; no runtime integration",
        "timebase_ms": 25,
        "train_run_ids": list(TRAIN_IDS),
        "heldout_run_id": validation_id,
        "train_rows": int(len(train["y"])),
        "heldout_rows": int(len(validation["y"])),
        "slip_observer_model": SLIP_MODEL,
        "slip_observer_inputs": "current/past encoder, IMU, actuator feedback and commands plus sensor-derived event; no GT/future inputs",
        "yaw_training_slip_features": "leave-one-whole-training-run-out slip predictions",
        "slip_estimation_train_oof": train_slip_score,
        "slip_estimation_heldout": validation_slip_score,
        "yaw_sensor_only_baseline": _yaw_metric(baseline_error),
        "yaw_with_sensor_estimated_slip": _yaw_metric(observer_error),
        "yaw_rmse_delta_radps": float(
            np.sqrt(np.mean(observer_error ** 2))
            - np.sqrt(np.mean(baseline_error ** 2))),
        "yaw_by_true_slip_regime": _yaw_by_slip_regime(
            validation, validation_slip, baseline_error, observer_error),
        "front_lateral_slip_gate": gated_yaw,
        "run_audits": audits,
        "source_admission_audit": source_audit,
        "caveats": [
            "Two whole-run training captures provide only a two-fold OOF observer check; unseen midpoint capture r04 is still the stronger transfer test after it closes.",
            "Slip labels remain kinematic proxies derived from simulator truth; per-wheel force, load, front RPM, and internal tire spline output are unavailable.",
            "This is not an odometry implementation and is not integrated into production.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True,
                        default=slip_diagnostic._json_default) + "\n",
        encoding="utf-8")

    fields = ["condition_id", "event", "gt_speed_mps", "steering_rad"]
    fields.extend(f"true_{name}" for name in slip_diagnostic.SLIP_NAMES)
    fields.extend(f"estimated_{name}" for name in slip_diagnostic.SLIP_NAMES)
    fields.extend(("baseline_yaw_error_radps", "estimated_slip_yaw_error_radps",
                   "front_slip_gated_yaw_error_radps"))
    with (output / "heldout_rows.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for index in range(len(validation["y"])):
            writer.writerow((
                str(validation["condition_id"][index]),
                str(validation["phase_event"][index]),
                float(validation["gt_speed"][index]),
                float(validation["steering"][index]),
                *(float(value) for value in validation_slip[index, :SLIP_WIDTH]),
                *(float(value) for value in validation_pred_slip[index, :SLIP_WIDTH]),
                float(baseline_error[index]), float(observer_error[index]),
                float(baseline_error[index] if
                      np.max(np.abs(validation_pred_slip[index, 2:4]))
                      < FRONT_LATERAL_PEAK_SLIP else observer_error[index]),
            ))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-run", default=VALIDATION_ID,
                        help="closed run with validation split; never use an active bag")
    args = parser.parse_args()
    report = run(args.output.resolve(), args.validation_run)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "slip_estimation_heldout": report["slip_estimation_heldout"],
        "yaw_sensor_only_baseline": report["yaw_sensor_only_baseline"],
        "yaw_with_sensor_estimated_slip": report["yaw_with_sensor_estimated_slip"],
        "yaw_rmse_delta_radps": report["yaw_rmse_delta_radps"],
        "front_lateral_slip_gate": report["front_lateral_slip_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
