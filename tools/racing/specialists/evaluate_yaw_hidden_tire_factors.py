#!/usr/bin/env python3
"""Screen physical tire-factor hypotheses against cached exact-two yaw rows.

Uses closed r01/r03 training extracts for leave-one-whole-run-out scoring and
an explicitly selected closed validation run for transfer scoring. GT-derived
slip/body state are diagnostic oracle inputs only.
No simulator, production odometry, MPC, or physics is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import analyze_yaw_slip_regime_factor as slip
    import evaluate_yaw_full_spectrum_exact_two as evaluator
    import evaluate_yaw_predicted_next_steering as causal
except ModuleNotFoundError:
    from tools.racing.specialists import analyze_yaw_slip_regime_factor as slip
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as causal


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "live_runs/racing_model_diagnostics_20261009/" \
    "front_slip_observer_candidate_r02/exact_two_row_cache"
DEFAULT_OUTPUT = ROOT / "live_runs/racing_model_diagnostics_20261009/" \
    "hidden_tire_factor_screen_r02/report.json"
RUN_FILES = {
    "r01": "openplane_yaw_full_spectrum_grid_train_20261009_r01.npz",
    "r03": "openplane_yaw_full_spectrum_grid_train_20261009_r03.npz",
    "r02": "openplane_yaw_full_spectrum_grid_validation_20261009_r02.npz",
}
R04_CACHE = ROOT / "live_runs/racing_model_diagnostics_20261009/" \
    "sensor_slip_relaxation_r04/exact_two_row_cache/" \
    "openplane_yaw_full_spectrum_midpoint_validation_20261009_r04.npz"
MODEL = dict(evaluator.MODEL)
DT_S = 0.025
LAGS_STEPS = (1, 2, 4, 8)


def _load(run: str) -> dict[str, np.ndarray]:
    if run == "r04":
        path = R04_CACHE
        expected = evaluator.FULL_VALIDATION_REPLICATION_ID
    else:
        path = CACHE / RUN_FILES[run]
        expected = {
            "r01": evaluator.FULL_TRAIN_ID,
            "r03": evaluator.FULL_TRAIN_REPLICATION_ID,
            "r02": evaluator.FULL_VALIDATION_ID,
        }[run]
    if not path.is_file():
        raise FileNotFoundError(f"required closed-run cache absent: {path}")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    actual = str(data.pop("cache_run_id").item())
    if actual != expected:
        raise ValueError(f"cache provenance mismatch: expected {expected}, got {actual}")
    return data


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = tuple(parts[0])
    if any(tuple(part) != keys for part in parts[1:]):
        raise ValueError("cache schemas differ")
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in keys}


def _mirror_sensor(sensor: np.ndarray) -> np.ndarray:
    # Add the four neutral timing columns expected by the shared mirror helper.
    padded = np.column_stack((sensor, np.zeros((len(sensor), 4), np.float32)))
    return causal._mirror_features(padded)[:, :-4]


def _state(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack((data["gt_body_state"][:, :2], data["gt_yaw_rate"]))


def _mirror_state(state: np.ndarray) -> np.ndarray:
    result = state.copy()
    result[:, 1:] *= -1.0
    return result


def _full_features(data: dict[str, np.ndarray]) -> np.ndarray:
    return evaluator._model_features(data)


def _oracle_base(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return sensor/event features and current GT state+slip oracle features."""
    truth = data["truth_slip"]
    extra = np.column_stack((_state(data), truth)).astype(np.float32)
    mirrored = np.column_stack((
        _mirror_state(_state(data)), slip._mirror_slip_features(truth)
    )).astype(np.float32)
    return extra, mirrored


def _previous_slip(data: dict[str, np.ndarray], steps: int
                   ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use only prior rows from the same exact-two response window.

    The capture is mapped to the stipulated 25-ms grid. A lag is admitted only
    when its actual event-age separation is within 12 ms of the requested
    grid lag, so a dropped archive row is not silently treated as one step.
    """
    n = len(data["y"])
    truth = data["truth_slip"].astype(np.float32)
    prior = np.zeros_like(truth)
    valid = np.zeros(n, dtype=bool)
    phase = data["phase_id"].astype(str)
    age = data["event_age_ms"].astype(np.float64)
    expected_ms = steps * DT_S * 1000.0
    deviations: list[float] = []
    for phase_id in np.unique(phase):
        indices = np.flatnonzero(phase == phase_id)
        indices = indices[np.argsort(age[indices], kind="stable")]
        for position in range(steps, len(indices)):
            current = int(indices[position])
            previous = int(indices[position - steps])
            separation = age[current] - age[previous]
            deviation = separation - expected_ms
            deviations.append(float(deviation))
            if abs(deviation) > 12.0:
                continue
            prior[current] = truth[previous]
            valid[current] = True
    extra = np.column_stack((prior[:, :6], prior[:, 6:], valid)).astype(np.float32)
    # Invalid lags are zeros with a separate flag; keep every candidate on the
    # same row set and allow the estimator to learn the validity boundary.
    return extra, valid, {
        "requested_lag_ms": expected_ms,
        "rows_with_valid_prior": int(np.count_nonzero(valid)),
        "fraction_with_valid_prior": float(np.mean(valid)),
        "observed_separation_error_ms_p95": (
            float(np.quantile(np.abs(deviations), 0.95)) if deviations else None),
        "rows_rejected_for_grid_gap": int(len(deviations) - np.count_nonzero(valid)),
    }


def _mirror_lag(extra: np.ndarray) -> np.ndarray:
    # Layout: six prior raw slips, six prior validity flags, one row-valid flag.
    raw = extra[:, :6]
    raw_full = np.column_stack((raw, np.zeros((len(raw), 6), np.float32)))
    mirrored_raw = slip._mirror_slip_features(raw_full)[:, :6]
    flags = extra[:, 6:12]
    flags_full = np.column_stack((np.zeros((len(flags), 6), np.float32), flags))
    mirrored_flags = slip._mirror_slip_features(flags_full)[:, 6:12]
    return np.column_stack((mirrored_raw, mirrored_flags, extra[:, 12]))


def _combined_interactions(data: dict[str, np.ndarray]) -> np.ndarray:
    truth = data["truth_slip"]
    sx = truth[:, :2]
    sy = truth[:, 4:6]
    norm_l = np.hypot(sx[:, 0], sy[:, 0])
    norm_r = np.hypot(sx[:, 1], sy[:, 1])
    return np.column_stack((
        sx[:, 0] * sy[:, 0], sx[:, 1] * sy[:, 1],
        np.abs(sx[:, 0]) * np.abs(sy[:, 0]),
        np.abs(sx[:, 1]) * np.abs(sy[:, 1]),
        norm_l, norm_r, norm_r - norm_l,
        np.max(np.abs(truth[:, 2:4]), axis=1)
        * np.mean(np.abs(sx), axis=1),
    )).astype(np.float32)


def _mirror_combined(data: dict[str, np.ndarray]) -> np.ndarray:
    reflected = dict(data)
    reflected["truth_slip"] = slip._mirror_slip_features(data["truth_slip"])
    return _combined_interactions(reflected)


def _load_transfer_interactions(data: dict[str, np.ndarray]) -> np.ndarray:
    x = data["x"]
    truth = data["truth_slip"]
    # Current sensor block is first; derived features follow four 11-channel
    # history blocks. These are observable roll/IMU/actuator terms interacting
    # with GT slip magnitude, not measurements of tire normal load.
    roll, ay, roll_rate = x[:, 9], x[:, 5], x[:, 10]
    steer_rate = x[:, 4 * 11 + 3]
    front_sy = np.max(np.abs(truth[:, 2:4]), axis=1)
    rear_sy = np.mean(np.abs(truth[:, 4:6]), axis=1)
    rear_sx = np.mean(np.abs(truth[:, :2]), axis=1)
    return np.column_stack((
        roll * front_sy,
        roll_rate * front_sy,
        ay * front_sy,
        steer_rate * front_sy,
        roll * rear_sy,
        roll_rate * rear_sy,
        np.abs(ay) * rear_sx,
        x[:, 4 * 11 + 4] * rear_sx,
    )).astype(np.float32)


def _mirror_load_transfer(data: dict[str, np.ndarray]) -> np.ndarray:
    reflected = dict(data)
    x = data["x"].copy()
    x[:, [0, 5, 6, 7, 9, 10]] *= -1.0
    x[:, 2], x[:, 3] = data["x"][:, 3], data["x"][:, 2]
    derived = 4 * 11
    x[:, [derived + 1, derived + 3, derived + 5, derived + 7]] *= -1.0
    reflected["x"] = x
    reflected["truth_slip"] = slip._mirror_slip_features(data["truth_slip"])
    return _load_transfer_interactions(reflected)


def _fit_predict(train: dict[str, np.ndarray], test: dict[str, np.ndarray],
                 extra_train: np.ndarray, extra_test: np.ndarray,
                 mirror_train: np.ndarray, mirror_test: np.ndarray,
                 *, include_oracle: bool) -> np.ndarray:
    base_train, base_test = _full_features(train), _full_features(test)
    sensor_train = base_train[:, :-len(evaluator.EVENTS)]
    sensor_test = base_test[:, :-len(evaluator.EVENTS)]
    event_train = base_train[:, -len(evaluator.EVENTS):]
    event_test = base_test[:, -len(evaluator.EVENTS):]
    mirrored_sensor_train = _mirror_sensor(sensor_train)
    mirrored_sensor_test = _mirror_sensor(sensor_test)
    if include_oracle:
        x_train = np.column_stack((base_train, extra_train))
        x_test = np.column_stack((base_test, extra_test))
        mx_train = np.column_stack((mirrored_sensor_train, event_train,
                                    mirror_train))
        _ = np.column_stack((mirrored_sensor_test, event_test, mirror_test))
    else:
        x_train, x_test = base_train, base_test
        mx_train = np.column_stack((mirrored_sensor_train, event_train))
    model = ExtraTreesRegressor(**MODEL)
    phase = train["phase_id"].astype(str)
    fit_x = np.concatenate((x_train, mx_train), axis=0)
    fit_y = np.concatenate((train["y"], -train["y"]))
    fit_phase = np.concatenate((phase, phase))
    model.fit(fit_x, fit_y, sample_weight=causal._phase_weights(fit_phase))
    return model.predict(x_test).astype(np.float32)


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
    }


def _candidate_specs(data: dict[str, np.ndarray]
                     ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    base, mirror_base = _oracle_base(data)
    specs = {"gt_state_plus_current_slip": (base, mirror_base)}

    combined = _combined_interactions(data)
    mirror_combined = _mirror_combined(data)
    specs["plus_rear_combined_slip_interactions"] = (
        np.column_stack((base, combined)),
        np.column_stack((mirror_base, mirror_combined)),
    )

    load = _load_transfer_interactions(data)
    mirror_load = _mirror_load_transfer(data)
    specs["plus_roll_ay_slip_interactions"] = (
        np.column_stack((base, load)),
        np.column_stack((mirror_base, mirror_load)),
    )

    for steps in LAGS_STEPS:
        lag, _valid, _ = _previous_slip(data, steps)
        mirror_lag = _mirror_lag(lag)
        # Add current-minus-past slip rate as a compact tire-relaxation proxy.
        raw_lag = lag[:, :6]
        current = data["truth_slip"][:, :6]
        dt = steps * DT_S
        rate = (current - raw_lag) / dt
        rate[lag[:, 12] < 0.5] = 0.0
        extra = np.column_stack((base, lag, rate)).astype(np.float32)
        mirror_current = slip._mirror_slip_features(data["truth_slip"])
        mirror_rate = (mirror_current[:, :6] - _mirror_lag(lag)[:, :6]) / dt
        mirror_rate[lag[:, 12] < 0.5] = 0.0
        mirror_extra = np.column_stack((mirror_base, mirror_lag, mirror_rate))
        specs[f"plus_slip_relaxation_lag_{steps*25}ms"] = (
            extra, mirror_extra.astype(np.float32))
    return specs


def _run(output: Path, *, focused: bool = False,
         validation_run: str = "r02",
         transfer_only: bool = False) -> dict[str, Any]:
    if validation_run not in ("r02", "r04"):
        raise ValueError(f"unsupported validation run: {validation_run}")
    runs = {name: _load(name) for name in ("r01", "r03", validation_run)}
    train_ids = ("r01", "r03")
    validation_id = validation_run
    folds: dict[str, Any] = {}
    oof: dict[str, dict[str, list[np.ndarray]]] = {}

    # Precompute deterministic factors separately for each run. Every held-out
    # prediction is fit on the other complete capture only.
    specs_by_run = {name: _candidate_specs(data) for name, data in runs.items()}
    candidate_names = tuple(specs_by_run["r01"])
    if any(tuple(specs_by_run[name]) != candidate_names
           for name in runs):
        raise ValueError("candidate feature schemas differ between runs")
    if focused:
        candidate_names = (
            "gt_state_plus_current_slip",
            "plus_slip_relaxation_lag_25ms",
            "plus_slip_relaxation_lag_100ms",
        )
    for candidate in candidate_names:
        oof[candidate] = {run: [] for run in train_ids}
    oof["sensor_only"] = {run: [] for run in train_ids}

    for held_id in (() if transfer_only else train_ids):
        fit_id = next(run for run in train_ids if run != held_id)
        fit_data, held_data = runs[fit_id], runs[held_id]
        train_specs = specs_by_run[fit_id]
        held_specs = specs_by_run[held_id]
        baseline = _fit_predict(
            fit_data, held_data, np.empty((len(fit_data["y"]), 0), np.float32),
            np.empty((len(held_data["y"]), 0), np.float32),
            np.empty((len(fit_data["y"]), 0), np.float32),
            np.empty((len(held_data["y"]), 0), np.float32),
            include_oracle=False)
        base_error = baseline - held_data["y"]
        oof["sensor_only"][held_id].append(base_error)
        folds[held_id] = {"sensor_only": _metric(base_error)}
        for candidate in candidate_names:
            train_extra, train_mirror = train_specs[candidate]
            test_extra, test_mirror = held_specs[candidate]
            prediction = _fit_predict(
                fit_data, held_data, train_extra, test_extra,
                train_mirror, test_mirror, include_oracle=True)
            error = prediction - held_data["y"]
            oof[candidate][held_id].append(error)
            folds[held_id][candidate] = _metric(error)

    train = _combine([runs[name] for name in train_ids])
    val = runs[validation_id]
    train_specs = _candidate_specs(train)
    val_specs = _candidate_specs(val)
    baseline = _fit_predict(
        train, val, np.empty((len(train["y"]), 0), np.float32),
        np.empty((len(val["y"]), 0), np.float32),
        np.empty((len(train["y"]), 0), np.float32),
        np.empty((len(val["y"]), 0), np.float32), include_oracle=False)
    development_errors: dict[str, np.ndarray] = {
        "sensor_only": baseline - val["y"]}
    for candidate in candidate_names:
        train_extra, train_mirror = train_specs[candidate]
        val_extra, val_mirror = val_specs[candidate]
        prediction = _fit_predict(
            train, val, train_extra, val_extra, train_mirror, val_mirror,
            include_oracle=True)
        development_errors[candidate] = prediction - val["y"]
    development = {name: _metric(error)
                   for name, error in development_errors.items()}

    event_metrics: dict[str, Any] = {}
    for event in sorted(np.unique(val["phase_event"].astype(str))):
        mask = val["phase_event"].astype(str) == event
        event_metrics[event] = {
            name: _metric(error[mask])
            for name, error in development_errors.items()
        }

    speed_edges = tuple(float(value) for value in range(13)) + (12.01,)
    steer_edges = tuple(value / 100.0 for value in range(0, 51, 5)) + (0.501,)
    regime_metrics: dict[str, Any] = {}
    speed = val["gt_speed"].astype(np.float64)
    abs_steering = np.abs(val["steering"].astype(np.float64))
    events = val["phase_event"].astype(str)
    for low_speed, high_speed in zip(speed_edges[:-1], speed_edges[1:]):
        for low_steer, high_steer in zip(steer_edges[:-1], steer_edges[1:]):
            cell = ((speed >= low_speed) & (speed < high_speed)
                    & (abs_steering >= low_steer)
                    & (abs_steering < high_steer))
            for event in sorted(np.unique(events)):
                mask = cell & (events == event)
                if not np.any(mask):
                    continue
                key = (f"speed_{low_speed:g}_{high_speed:g}__"
                       f"abs_steer_{low_steer:g}_{high_steer:g}__{event}")
                regime_metrics[key] = {
                    "speed_mps": [low_speed, high_speed],
                    "abs_steering_rad": [low_steer, high_steer],
                    "event": event,
                    "models": {
                        name: _metric(error[mask])
                        for name, error in development_errors.items()
                    },
                }

    oof_summary = {}
    for candidate, per_run in oof.items():
        if transfer_only:
            break
        run_metrics = {run: _metric(np.concatenate(errors))
                       for run, errors in per_run.items()}
        sensor_rmse = {
            run: _metric(oof["sensor_only"][run][0])["rmse_radps"]
            for run in train_ids
        }
        deltas = {}
        for run in train_ids:
            candidate_metric = run_metrics[run]
            reference = _metric(oof["gt_state_plus_current_slip"][run][0])
            deltas[run] = {
                "rmse_delta_vs_instantaneous_slip_radps": (
                    candidate_metric["rmse_radps"] - reference["rmse_radps"]),
                "samples_over_0p1_delta_vs_instantaneous_slip": (
                    candidate_metric["samples_over_0p1_radps"]
                    - reference["samples_over_0p1_radps"]),
            }
        oof_summary[candidate] = {
            "heldout_run_metrics": run_metrics,
            "delta_vs_instantaneous_slip_oracle": deltas,
            "both_run_rmse_improved_vs_instantaneous_slip": all(
                deltas[run]["rmse_delta_vs_instantaneous_slip_radps"] < 0
                for run in train_ids),
            "both_run_rmse_improved_vs_sensor_only": all(
                run_metrics[run]["rmse_radps"] < sensor_rmse[run]
                for run in train_ids),
        }

    lag_quality = {}
    for run in runs:
        lag_quality[run] = {
            f"{steps*25}ms": _previous_slip(runs[run], steps)[2]
            for steps in LAGS_STEPS
        }
    report = {
        "objective": "test whether combined rear-tire slip, slip relaxation, or slip-modulated roll/IMU loading adds held-out yaw information beyond instantaneous GT body state and slip",
        "status": "offline oracle-factor screen; no runtime integration",
        "data": {
            "train_runs": [evaluator.FULL_TRAIN_ID,
                           evaluator.FULL_TRAIN_REPLICATION_ID],
            "whole_run_oof_runs": list(train_ids),
            "validation_run": {
                "r02": evaluator.FULL_VALIDATION_ID,
                "r04": evaluator.FULL_VALIDATION_REPLICATION_ID,
            }[validation_id],
            "validation_run_key": validation_id,
            "r04_bag_read": validation_id == "r04",
            "rows": {name: int(len(data["y"])) for name, data in runs.items()},
            "timebase_ms": 25,
        },
        "method": {
            "model": MODEL,
            "mirror_augmentation": "left/right reflection with physically signed state and slip transformations",
            "selection": f"no tuning on {validation_id}; "
                         + ("score one frozen fit on selected run only"
                            if transfer_only else
                            "compare both whole-run OOF folds and score selected run as transfer evidence"),
            "target": "one-step simulator GT yaw-rate residual relative to current IMU yaw rate",
        },
        "whole_run_oof_by_fold": folds,
        "whole_run_oof_summary": oof_summary,
        f"{validation_id}_metrics": development,
        f"{validation_id}_by_event": event_metrics,
        f"{validation_id}_speed_x_abs_steering_x_event_atlas": regime_metrics,
        "slip_lag_alignment_quality": lag_quality,
        "factor_definitions": {
            "rear_combined_slip": "per-rear-wheel Sx*Sy, |Sx|*|Sy|, sqrt(Sx^2+Sy^2), left-right difference, and front-lateral/rear-longitudinal coupling",
            "roll_ay_slip_interactions": "measured roll, roll-rate, lateral acceleration, and steering rate multiplied by GT front/rear slip magnitudes; proxies for load-transfer modulation, not wheel normal-force measurements",
            "slip_relaxation": "causal 25/50/100/200-ms lagged GT per-wheel slip and finite-difference slip rate, strictly within the same response phase",
        },
        "limitations": [
            "GT slip and GT body state are oracle inputs and cannot be deployed directly.",
            f"Only two independent training captures exist for whole-run OOF; {validation_id} transfer evidence is not a blind external replication if already reviewed.",
            "No per-wheel normal loads, tire forces, front wheel speeds, suspension deflections, or internal tire-curve states are present in these captures.",
            "A predictive gain identifies information associated with the factor, not proof of causation or a calibrated physical force law.",
            "The prior 24.8% figure is fewer threshold-exceeding samples on r02, not a fraction of total error variance explained.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--focused", action="store_true",
        help="only score instantaneous-slip, 25-ms relaxation, and 100-ms relaxation models")
    parser.add_argument("--validation-run", choices=("r02", "r04"), default="r02")
    parser.add_argument(
        "--transfer-only", action="store_true",
        help="skip repeated run-held-out fits; fit on r01+r03 and score selected run")
    args = parser.parse_args()
    report = _run(args.output.resolve(), focused=args.focused,
                  validation_run=args.validation_run,
                  transfer_only=args.transfer_only)
    print(json.dumps({
        "report": str(args.output.resolve()),
        "validation_run": args.validation_run,
        "whole_run_oof_summary": report["whole_run_oof_summary"],
        "validation_metrics": report[f"{args.validation_run}_metrics"],
        "validation_by_event": report[f"{args.validation_run}_by_event"],
        "slip_lag_alignment_quality": report["slip_lag_alignment_quality"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
