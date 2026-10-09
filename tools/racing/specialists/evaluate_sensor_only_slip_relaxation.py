#!/usr/bin/env python3
"""Test causal slip estimates and diagnose remaining held-out yaw residuals.

Slip observers are fitted on r01/r03 whole-run folds and produce out-of-fold
features for yaw training. The frozen observer trained on both runs is scored
on a separate validation capture (r02 or midpoint r04). Simulator truth is
used only for fitting labels, offline diagnosis, and scoring.
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
    import fit_sensor_only_yaw_regime_atlas as atlas
    import analyze_yaw_slip_regime_factor as slip
    import compare_front_slip_observer as front
    import evaluate_yaw_full_spectrum_exact_two as evaluator
    import evaluate_yaw_predicted_next_steering as causal
    import train_yaw_multihorizon_teacher as teacher
except ModuleNotFoundError:
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import analyze_yaw_slip_regime_factor as slip
    from tools.racing.specialists import compare_front_slip_observer as front
    from tools.racing.specialists import evaluate_yaw_full_spectrum_exact_two as evaluator
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as causal
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "live_runs/racing_model_diagnostics_20261009/" \
    "front_slip_observer_candidate_r02/exact_two_row_cache"
DEFAULT_OUTPUT = ROOT / "live_runs/racing_model_diagnostics_20261009/" \
    "sensor_slip_relaxation_r02/report.json"
FILES = {
    "r01": "openplane_yaw_full_spectrum_grid_train_20261009_r01.npz",
    "r03": "openplane_yaw_full_spectrum_grid_train_20261009_r03.npz",
    "r02": "openplane_yaw_full_spectrum_grid_validation_20261009_r02.npz",
}
RUN_ID = {
    "r01": evaluator.FULL_TRAIN_ID,
    "r03": evaluator.FULL_TRAIN_REPLICATION_ID,
    "r02": evaluator.FULL_VALIDATION_ID,
}
DT_S = 0.025
FRONT_PEAK_GATE = 0.01


def _load(name: str) -> dict[str, np.ndarray]:
    if name in FILES:
        path = CACHE / FILES[name]
        expected = RUN_ID[name]
    else:
        raise ValueError(f"unsupported validation cache key: {name}")
    return _load_cache(path, expected)


def _load_cache(path: Path, expected_run_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    actual = str(data.pop("cache_run_id").item())
    if actual != expected_run_id:
        raise ValueError(f"cache provenance mismatch: expected {expected_run_id}, got {actual}")
    return data


def _load_midpoint_r04(cache_dir: Path) -> dict[str, np.ndarray]:
    run_id = evaluator.FULL_VALIDATION_REPLICATION_ID
    cache_path = cache_dir / f"{run_id}.npz"
    if cache_path.is_file():
        return _load_cache(cache_path, run_id)
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    series = [item for item in admitted if item.run_id == run_id]
    if len(series) != 1 or series[0].split != "validation":
        raise ValueError(
            f"r04 must be uniquely admitted as validation; audit={source_audit}")
    data, audit = evaluator._collect_run(series[0], atlas.HISTORY_LAGS)
    target, _summary = slip._slip_features(data)
    if audit["run_id"] != run_id or audit["quality"]["aborted"]:
        raise ValueError(f"r04 exact-two collection failed quality gate: {audit}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, cache_run_id=np.asarray(run_id),
                        **data, truth_slip=target)
    return data | {"truth_slip": target}


def _odometry_input_view(data: dict[str, np.ndarray]
                         ) -> dict[str, np.ndarray]:
    """Mask features to channels present in the current odometry packet.

    The C++ odometry path currently receives rear encoder angles and IMU ax,
    ay, yaw rate/orientation. Keep only rear wheel speeds and IMU ax/ay/yaw
    rate from each history block, plus derivations it can form from those
    signals. In particular, do not leak steering-based event labels.
    """
    result = dict(data)
    source = np.asarray(data["x"], dtype=np.float32)
    masked = np.zeros_like(source)
    channel_width = len(atlas.OBSERVATION_NAMES)
    history_width = len(atlas.HISTORY_LAGS) * channel_width
    for block in range(len(atlas.HISTORY_LAGS)):
        start = block * channel_width
        for channel in (2, 3, 4, 5, 6):
            masked[:, start + channel] = source[:, start + channel]
    derived_start = history_width
    # Mean/split/rate of the rear encoders and IMU yaw-rate change.
    for derived in (0, 1, 2, 7):
        index = derived_start + derived
        if index < source.shape[1]:
            masked[:, index] = source[:, index]
    result["x"] = masked
    result["causal_event"] = np.full(len(source), "hold", dtype="U16")
    return result


def _combine(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]}


def _predict_front(data: dict[str, np.ndarray],
                   target: np.ndarray) -> Any:
    return front._fit_front(data, target)


def _previous_prediction(data: dict[str, np.ndarray], current: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    previous = np.zeros_like(current, dtype=np.float32)
    valid = np.zeros(len(current), dtype=bool)
    phase = data["phase_id"].astype(str)
    age = data["event_age_ms"].astype(np.float64)
    deviations = []
    for phase_id in np.unique(phase):
        indices = np.flatnonzero(phase == phase_id)
        indices = indices[np.argsort(age[indices], kind="stable")]
        for position in range(1, len(indices)):
            row, prior = int(indices[position]), int(indices[position - 1])
            delta_ms = float(age[row] - age[prior])
            deviations.append(abs(delta_ms - 25.0))
            if abs(delta_ms - 25.0) > 12.0:
                continue
            previous[row] = current[prior]
            valid[row] = True
    return previous, valid, {
        "requested_lag_ms": 25.0,
        "valid_rows": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "p95_abs_grid_error_ms": (float(np.quantile(deviations, 0.95))
                                   if deviations else None),
    }


def _mirror_sensor(sensor: np.ndarray) -> np.ndarray:
    padded = np.column_stack((sensor, np.zeros((len(sensor), 4), np.float32)))
    return causal._mirror_features(padded)[:, :-4]


def _mirror_front(values: np.ndarray) -> np.ndarray:
    return np.column_stack((-values[:, 1], -values[:, 0])).astype(np.float32)


def _front_features(current: np.ndarray, previous: np.ndarray,
                    valid: np.ndarray, *, lagged: bool) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    if not lagged:
        return current
    delta = current - previous
    return np.column_stack((current, previous, delta,
                            valid.astype(np.float32))).astype(np.float32)


def _mirror_extra(extra: np.ndarray, *, lagged: bool) -> np.ndarray:
    if not lagged:
        return _mirror_front(extra)
    current = _mirror_front(extra[:, :2])
    previous = _mirror_front(extra[:, 2:4])
    delta = _mirror_front(extra[:, 4:6])
    return np.column_stack((current, previous, delta, extra[:, 6]))


def _fit_yaw(train: dict[str, np.ndarray], test: dict[str, np.ndarray],
             train_extra: np.ndarray | None,
             test_extra: np.ndarray | None, *, lagged: bool
             ) -> np.ndarray:
    train_base = evaluator._model_features(train)
    test_base = evaluator._model_features(test)
    train_sensor = train_base[:, :-len(evaluator.EVENTS)]
    test_sensor = test_base[:, :-len(evaluator.EVENTS)]
    train_event = train_base[:, -len(evaluator.EVENTS):]
    mirrored_sensor = _mirror_sensor(train_sensor)
    if train_extra is None:
        x = train_base
        mirrored = np.column_stack((mirrored_sensor, train_event))
        test_x = test_base
    else:
        x = np.column_stack((train_base, train_extra))
        mirrored = np.column_stack((
            mirrored_sensor, train_event,
            _mirror_extra(train_extra, lagged=lagged)))
        test_x = np.column_stack((test_base, test_extra))
    model = ExtraTreesRegressor(**evaluator.MODEL)
    phase = train["phase_id"].astype(str)
    fit_phase = np.concatenate((phase, phase))
    model.fit(np.concatenate((x, mirrored)),
              np.concatenate((train["y"], -train["y"])),
              sample_weight=causal._phase_weights(fit_phase))
    return model.predict(test_x).astype(np.float32)


def _age_residual_features(data: dict[str, np.ndarray],
                           estimated_slip_history: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Causal yaw/slip features plus elapsed time since observed command edge.

    ``event_age_ms`` is aligned to the measured steering-command transition
    in the bag and is a diagnostic deployment proxy; the schedule label and
    ``phase_event`` are not included in these features.
    """
    base = evaluator._model_features(data)
    sensor = base[:, :-len(evaluator.EVENTS)]
    event = base[:, -len(evaluator.EVENTS):]
    age = np.asarray(data["event_age_ms"], dtype=np.float32) / 1000.0
    speed = np.asarray(data["wheel_speed"], dtype=np.float32)
    steering = np.asarray(data["steering"], dtype=np.float32)
    context = np.column_stack((age, speed, steering, age * speed,
                               age * steering, age * np.abs(steering)))
    x = np.column_stack((base, estimated_slip_history, context)).astype(np.float32)

    mirrored_sensor = _mirror_sensor(sensor)
    mirrored_context = np.column_stack((age, speed, -steering, age * speed,
                                        -age * steering,
                                        age * np.abs(steering)))
    mirrored = np.column_stack((
        mirrored_sensor, event,
        _mirror_extra(estimated_slip_history, lagged=True),
        mirrored_context)).astype(np.float32)
    return x, mirrored


def _fit_age_residual_correction(
        runs: dict[str, dict[str, np.ndarray]],
        estimated_history: dict[str, np.ndarray]
        ) -> tuple[ExtraTreesRegressor, dict[str, np.ndarray],
                   dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Fit an additive correction from whole-run OOF yaw residuals only."""
    oof_residual: dict[str, np.ndarray] = {}
    fold_reports = []
    for held, fit in (("r01", "r03"), ("r03", "r01")):
        oof_base = _fit_yaw(
            runs[fit], runs[held], estimated_history[fit],
            estimated_history[held], lagged=True)
        oof_residual[held] = runs[held]["y"] - oof_base
        fold_reports.append({
            "heldout_run": RUN_ID[held],
            "base_model_training_run": RUN_ID[fit],
            "rows": int(len(oof_base)),
            "base_oof_rmse_radps": float(np.sqrt(np.mean(
                (oof_base - runs[held]["y"]) ** 2))),
        })

    train_x_parts = []
    mirror_x_parts = []
    feature_by_run: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    residual_parts = []
    phase_parts = []
    for run_id in ("r01", "r03"):
        x, mirrored = _age_residual_features(
            runs[run_id], estimated_history[run_id])
        feature_by_run[run_id] = (x, mirrored)
        train_x_parts.append(x)
        mirror_x_parts.append(mirrored)
        residual_parts.append(oof_residual[run_id])
        phase_parts.append(runs[run_id]["phase_id"].astype(str))

    x = np.concatenate(train_x_parts)
    mirrored = np.concatenate(mirror_x_parts)
    residual = np.concatenate(residual_parts)
    phase = np.concatenate(phase_parts)
    model = ExtraTreesRegressor(**evaluator.MODEL)
    model.fit(np.concatenate((x, mirrored)),
              np.concatenate((residual, -residual)),
              sample_weight=causal._phase_weights(
                  np.concatenate((phase, phase))))
    return model, oof_residual, feature_by_run, {
        "folds": fold_reports,
        "oof_residual_rows": int(len(residual)),
        "features": [
            "same current/past sensor and event-class features as base yaw model",
            "OOF estimated front-slip current and previous 25-ms state",
            "measured command-edge age, wheel speed, steering, and age interactions",
        ],
        "schedule_event_label_used_as_input": False,
        "runtime_command_edge_detector_already_present_in_odometry": False,
    }


def _regional_residual_gates(data: dict[str, np.ndarray]
                             ) -> dict[str, np.ndarray]:
    """Causal response-state gates for residual-specialist evaluation."""
    speed = np.asarray(data["wheel_speed"], dtype=np.float64)
    steer = np.abs(np.asarray(data["steering"], dtype=np.float64))
    age_ms = np.asarray(data["event_age_ms"], dtype=np.float64)
    event = data["causal_event"].astype(str)
    # Derived feature 12 is steering-feedback age minus steering-command age.
    # A positive two-packet gap means the command changed at least 50 ms more
    # recently than feedback, a distinct actuator-response state observed in
    # the held-out residual audit. Keep this as a gated expert hypothesis; its
    # admission still depends on both whole-run CV folds.
    steering_feedback_lag = np.asarray(data["x"][:, 56], dtype=np.float64)
    return {
        "steering_feedback_lag": steering_feedback_lag >= 0.05,
        "crawl_high_steer": (speed >= 0.5) & (speed < 2.5) & (steer >= 0.25),
        "top_speed_near_center": (speed >= 10.0) & (steer < 0.10),
        "medium_speed_unwind_window": (
            (speed >= 4.0) & (speed < 8.0) & (steer < 0.15)
            & (event == "unwind") & (age_ms >= 150.0) & (age_ms < 300.0)),
        "fast_unwind_release": (
            (speed >= 8.0) & (speed < 10.0) & (steer < 0.15)
            & (event == "unwind") & (age_ms >= 175.0) & (age_ms < 275.0)),
        "mid_low_unwind_steering_edge": (
            (speed >= 3.0) & (speed < 4.0) & (steer >= 0.10) & (steer < 0.20)
            & (event == "unwind") & (age_ms >= 0.0) & (age_ms < 300.0)),
    }


def _fit_residual_model(
        train_features: list[tuple[np.ndarray, np.ndarray]],
        train_residuals: list[np.ndarray],
        train_phase_ids: list[np.ndarray],
        train_masks: list[np.ndarray] | None = None
        ) -> ExtraTreesRegressor:
    x_parts, mirror_parts, residual_parts, phase_parts = [], [], [], []
    for index, ((x, mirrored), residual, phase) in enumerate(zip(
            train_features, train_residuals, train_phase_ids)):
        mask = (np.ones(len(residual), dtype=bool) if train_masks is None
                else train_masks[index])
        if not np.any(mask):
            continue
        x_parts.append(x[mask])
        mirror_parts.append(mirrored[mask])
        residual_parts.append(residual[mask])
        phase_parts.append(phase[mask])
    if not residual_parts:
        raise ValueError("residual expert has no supported training rows")
    residual = np.concatenate(residual_parts)
    phase = np.concatenate(phase_parts).astype(str)
    model = ExtraTreesRegressor(**evaluator.MODEL)
    model.fit(np.concatenate((*x_parts, *mirror_parts)),
              np.concatenate((residual, -residual)),
              sample_weight=causal._phase_weights(
                  np.concatenate((phase, phase))))
    return model


def _evaluate_regional_residual_specialists(
        runs: dict[str, dict[str, np.ndarray]],
        feature_by_run: dict[str, tuple[np.ndarray, np.ndarray]],
        oof_residual: dict[str, np.ndarray],
        validation: dict[str, np.ndarray],
        validation_features: tuple[np.ndarray, np.ndarray],
        validation_global_correction: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Cross-run test local residual experts and score a held-out capture."""
    train_ids = ("r01", "r03")
    train_gates = {run_id: _regional_residual_gates(runs[run_id])
                   for run_id in train_ids}
    validation_gates = _regional_residual_gates(validation)
    validation_x = validation_features[0]
    all_region_prediction = validation_global_correction.copy()
    selected_region_prediction = validation_global_correction.copy()
    report: dict[str, Any] = {}

    for region in next(iter(train_gates.values())):
        cv_global_errors: list[np.ndarray] = []
        cv_local_errors: list[np.ndarray] = []
        fold_rows = []
        supported = True
        for held, fit in (("r01", "r03"), ("r03", "r01")):
            fit_mask = train_gates[fit][region]
            held_mask = train_gates[held][region]
            fit_phases = runs[fit]["phase_id"].astype(str)[fit_mask]
            held_count = int(np.count_nonzero(held_mask))
            support_ok = (int(np.count_nonzero(fit_mask)) >= 100
                          and len(np.unique(fit_phases)) >= 20
                          and held_count >= 50)
            fold_rows.append({
                "training_run": RUN_ID[fit],
                "heldout_run": RUN_ID[held],
                "training_rows_in_region": int(np.count_nonzero(fit_mask)),
                "training_phases_in_region": int(len(np.unique(fit_phases))),
                "heldout_rows_in_region": held_count,
                "support_gate_passed": bool(support_ok),
            })
            if not support_ok:
                supported = False
                continue
            fit_feature = feature_by_run[fit]
            global_model = _fit_residual_model(
                [fit_feature], [oof_residual[fit]],
                [runs[fit]["phase_id"].astype(str)])
            local_model = _fit_residual_model(
                [fit_feature], [oof_residual[fit]],
                [runs[fit]["phase_id"].astype(str)], [fit_mask])
            held_x = feature_by_run[held][0]
            global_error = (global_model.predict(held_x[held_mask])
                            - oof_residual[held][held_mask])
            local_error = (local_model.predict(held_x[held_mask])
                           - oof_residual[held][held_mask])
            cv_global_errors.append(global_error)
            cv_local_errors.append(local_error)

        cv_global = (_metric(np.concatenate(cv_global_errors))
                     if supported and cv_global_errors else {"samples": 0})
        cv_local = (_metric(np.concatenate(cv_local_errors))
                    if supported and cv_local_errors else {"samples": 0})
        selected = bool(
            supported and cv_local.get("samples", 0) > 0
            and cv_local.get("rmse_radps", float("inf"))
                < cv_global.get("rmse_radps", float("inf"))
            and cv_local.get("samples_over_0p1_radps", float("inf"))
                <= cv_global.get("samples_over_0p1_radps", float("inf")))

        final_masks = [train_gates[run_id][region] for run_id in train_ids]
        if not supported:
            report[region] = {
                "cross_run_cv_global_correction": cv_global,
                "cross_run_cv_local_specialist": cv_local,
                "cross_run_support": fold_rows,
                "selected_for_validation_combination": False,
                "validation_rows_in_region": int(np.count_nonzero(
                    validation_gates[region])),
                "support_failure": "both train/held-out whole-run folds need adequate region rows and phase support",
            }
            continue
        final_model = _fit_residual_model(
            [feature_by_run[run_id] for run_id in train_ids],
            [oof_residual[run_id] for run_id in train_ids],
            [runs[run_id]["phase_id"].astype(str) for run_id in train_ids],
            final_masks)
        val_mask = validation_gates[region]
        local_prediction = final_model.predict(validation_x[val_mask])
        all_region_prediction[val_mask] = local_prediction
        if selected:
            selected_region_prediction[val_mask] = local_prediction
        report[region] = {
            "gate": {
                "signals": {
                    "steering_feedback_lag": "steering feedback has not changed for at least two 25-ms packets longer than steering command feedback age (feedback-age minus command-age >= 50 ms)",
                    "crawl_high_steer": "0.5 <= rear-wheel mean speed < 2.5 m/s and |steering feedback| >= 0.25 rad",
                    "top_speed_near_center": "rear-wheel mean speed >= 10 m/s and |steering feedback| < 0.10 rad",
                    "medium_speed_unwind_window": "4 <= rear-wheel mean speed < 8 m/s, |steering feedback| < 0.15 rad, causal event is unwind, command-edge age 150-300 ms",
                    "fast_unwind_release": "8 <= rear-wheel mean speed < 10 m/s, |steering feedback| < 0.15 rad, causal event is unwind, command-edge age 175-275 ms",
                    "mid_low_unwind_steering_edge": "3 <= rear-wheel mean speed < 4 m/s, 0.10 <= |steering feedback| < 0.20 rad, causal event is unwind, command-edge age 0-300 ms",
                }[region],
                "phase_event_schedule_annotation_used": False,
            },
            "cross_run_cv_global_correction": cv_global,
            "cross_run_cv_local_specialist": cv_local,
            "cross_run_support": fold_rows,
            "selected_for_validation_combination": selected,
            "validation_rows_in_region": int(np.count_nonzero(val_mask)),
            "validation_global_correction_error": None,
            "validation_local_specialist_error": None,
        }

    return all_region_prediction, selected_region_prediction, report


def _direct_yaw_features(data: dict[str, np.ndarray]
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Causal sensor/history plus observed steering-transition age features."""
    base = evaluator._model_features(data)
    event_width = len(evaluator.EVENTS)
    sensor, event = base[:, :-event_width], base[:, -event_width:]
    speed = np.asarray(data["wheel_speed"], dtype=np.float32)
    steering = np.asarray(data["steering"], dtype=np.float32)
    age = np.asarray(data["event_age_ms"], dtype=np.float32) / 1000.0
    context = np.column_stack((
        age, speed, steering, age * speed, age * steering,
        age * np.abs(steering))).astype(np.float32)
    mirrored_context = np.column_stack((
        age, speed, -steering, age * speed, -age * steering,
        age * np.abs(steering))).astype(np.float32)

    # Preserve the event one-hot tail while mirroring the physical sensor
    # history. This is the same reflection convention as the base yaw model.
    padded = np.column_stack((
        sensor, np.zeros((len(sensor), event_width), dtype=np.float32)))
    mirrored_sensor = causal._mirror_features(padded)[:, :-event_width]
    return (
        np.column_stack((sensor, event, context)).astype(np.float32),
        np.column_stack((mirrored_sensor, event, mirrored_context)).astype(
            np.float32),
    )


def _fit_direct_yaw_specialist(
        data: dict[str, np.ndarray],
        features: tuple[np.ndarray, np.ndarray],
        mask: np.ndarray | None = None) -> ExtraTreesRegressor:
    """Fit a one-step yaw expert, optionally on one sensor-gated regime."""
    x, mirrored = features
    rows = (np.ones(len(data["y"]), dtype=bool) if mask is None else mask)
    if not np.any(rows):
        raise ValueError("direct yaw specialist has no training rows")
    phase = data["phase_id"].astype(str)[rows]
    model = ExtraTreesRegressor(**evaluator.MODEL)
    model.fit(
        np.concatenate((x[rows], mirrored[rows])),
        np.concatenate((data["y"][rows], -data["y"][rows])),
        sample_weight=causal._phase_weights(np.concatenate((phase, phase))),
    )
    return model


def _evaluate_direct_yaw_specialists(
        runs: dict[str, dict[str, np.ndarray]],
        validation: dict[str, np.ndarray],
        reference_prediction: np.ndarray
        ) -> tuple[np.ndarray, dict[str, Any]]:
    """Cross-run test local direct-yaw experts against global direct-yaw fits."""
    train_ids = ("r01", "r03")
    train_gates = {run_id: _regional_residual_gates(runs[run_id])
                   for run_id in train_ids}
    validation_gates = _regional_residual_gates(validation)
    features = {run_id: _direct_yaw_features(runs[run_id])
                for run_id in train_ids}
    validation_features = _direct_yaw_features(validation)
    prediction = reference_prediction.copy()
    report: dict[str, Any] = {}

    for region in next(iter(train_gates.values())):
        fold_reports = []
        global_errors: list[np.ndarray] = []
        local_errors: list[np.ndarray] = []
        supported = True
        for held, fit in (("r01", "r03"), ("r03", "r01")):
            fit_mask, held_mask = train_gates[fit][region], train_gates[held][region]
            fit_phases = runs[fit]["phase_id"].astype(str)[fit_mask]
            enough = (int(np.count_nonzero(fit_mask)) >= 100
                      and len(np.unique(fit_phases)) >= 20
                      and int(np.count_nonzero(held_mask)) >= 50)
            fold = {
                "training_run": RUN_ID[fit],
                "heldout_run": RUN_ID[held],
                "training_rows_in_region": int(np.count_nonzero(fit_mask)),
                "training_phases_in_region": int(len(np.unique(fit_phases))),
                "heldout_rows_in_region": int(np.count_nonzero(held_mask)),
                "support_gate_passed": bool(enough),
            }
            fold_reports.append(fold)
            if not enough:
                supported = False
                continue
            global_model = _fit_direct_yaw_specialist(
                runs[fit], features[fit])
            local_model = _fit_direct_yaw_specialist(
                runs[fit], features[fit], fit_mask)
            held_x = features[held][0][held_mask]
            truth = runs[held]["y"][held_mask]
            global_error = global_model.predict(held_x) - truth
            local_error = local_model.predict(held_x) - truth
            global_errors.append(global_error)
            local_errors.append(local_error)
            fold["global_direct_yaw"] = _metric(global_error)
            fold["local_direct_yaw"] = _metric(local_error)

        global_cv = (_metric(np.concatenate(global_errors))
                     if supported and global_errors else {"samples": 0})
        local_cv = (_metric(np.concatenate(local_errors))
                    if supported and local_errors else {"samples": 0})
        # Require both held-out captures to improve RMSE and not add threshold
        # violations; aggregate-only gains are insufficient for a regime gate.
        fold_wins = supported and all(
            "global_direct_yaw" in fold
            and fold["local_direct_yaw"]["rmse_radps"]
                < fold["global_direct_yaw"]["rmse_radps"]
            and fold["local_direct_yaw"]["samples_over_0p1_radps"]
                <= fold["global_direct_yaw"]["samples_over_0p1_radps"]
            for fold in fold_reports)
        selected = bool(
            fold_wins and local_cv.get("rmse_radps", float("inf"))
                < global_cv.get("rmse_radps", float("inf")))
        val_mask = validation_gates[region]
        local_val_error: dict[str, Any] = {"samples": 0}
        reference_val_error: dict[str, Any] = {"samples": 0}
        if selected and np.any(val_mask):
            final_model = _fit_direct_yaw_specialist(
                {key: np.concatenate([runs["r01"][key], runs["r03"][key]])
                 for key in runs["r01"] if isinstance(runs["r01"][key], np.ndarray)},
                (np.concatenate((features["r01"][0], features["r03"][0])),
                 np.concatenate((features["r01"][1], features["r03"][1]))),
                np.concatenate((train_gates["r01"][region],
                                train_gates["r03"][region])))
            local_prediction = final_model.predict(
                validation_features[0][val_mask]).astype(np.float32)
            prediction[val_mask] = local_prediction
            local_val_error = _metric(local_prediction - validation["y"][val_mask])
            reference_val_error = _metric(
                reference_prediction[val_mask] - validation["y"][val_mask])
        report[region] = {
            "selection_rule": (
                "selected only if both leave-one-run-out folds lower RMSE and do not increase >0.1-rad/s errors"),
            "cross_run_support": fold_reports,
            "cross_run_global_direct_yaw": global_cv,
            "cross_run_local_direct_yaw": local_cv,
            "selected_by_cross_run_validation": bool(selected),
            "validation_rows_in_region": int(np.count_nonzero(val_mask)),
            "validation_reference_model_error": reference_val_error,
            "validation_local_direct_yaw_error": local_val_error,
        }
    return prediction, report


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    if not len(error):
        return {"samples": 0, "samples_over_0p1_radps": 0}
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
    }


def _residual_diagnostics(
        data: dict[str, np.ndarray], errors: dict[str, np.ndarray],
        estimated_slip: np.ndarray, estimated_previous: np.ndarray,
        estimated_previous_valid: np.ndarray,
        true_previous: np.ndarray, true_previous_valid: np.ndarray,
        output_dir: Path, validation_run: str,
        candidate_key: str = "estimated_front_slip_plus_prior_25ms"
        ) -> dict[str, Any]:
    """Write every >0.1-rad/s candidate residual with its causal context.

    Ground-truth speed, body state, and tire-slip proxies are diagnosis-only;
    they are explicitly named as such and are never used to form predictions.
    The CSV is intentionally limited to large-error rows, keeping the artifact
    compact while retaining enough context to inspect individual failures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(data["x"], dtype=np.float64)
    slip = np.asarray(data["truth_slip"], dtype=np.float64)
    candidate_error = np.asarray(errors[candidate_key], dtype=np.float64)
    baseline_error = np.asarray(errors["sensor_only"], dtype=np.float64)
    absolute = np.abs(candidate_error)
    large = absolute > 0.1
    top = absolute >= np.quantile(absolute, 0.99)

    # Physical/sensor quantities are summarized separately from the model
    # inputs. The latter include current and historical observations; this
    # table focuses on interpretable lag-0 measurements and residual context.
    quantities: dict[str, np.ndarray] = {
        "gt_speed_mps_diagnostic_only": np.asarray(data["gt_speed"], dtype=np.float64),
        "rear_wheel_mean_minus_gt_speed_mps_diagnostic_only": (
            np.asarray(data["wheel_speed"], dtype=np.float64)
            - np.asarray(data["gt_speed"], dtype=np.float64)),
        "gt_body_u_mps_diagnostic_only": np.asarray(data["gt_body_state"][:, 0], dtype=np.float64),
        "gt_body_v_mps_diagnostic_only": np.asarray(data["gt_body_state"][:, 1], dtype=np.float64),
        "gt_sideslip_rad_diagnostic_only": np.asarray(data["gt_body_state"][:, 2], dtype=np.float64),
        "steering_feedback_rad": x[:, 0],
        "throttle_feedback_norm": x[:, 1],
        "rear_left_surface_mps": x[:, 2],
        "rear_right_surface_mps": x[:, 3],
        "imu_ax_mps2": x[:, 4],
        "imu_ay_mps2": x[:, 5],
        "imu_yaw_rate_rps": x[:, 6],
        "steering_command_rad": x[:, 7],
        "throttle_command_norm": x[:, 8],
        "imu_roll_rad": x[:, 9],
        "imu_roll_rate_rps": x[:, 10],
        "rear_wheel_split_mps": x[:, 45],
        "rear_wheel_mean_rate_mps2": x[:, 46],
        "steering_feedback_rate_radps": x[:, 47],
        "throttle_feedback_rate_per_s": x[:, 48],
        "steering_command_gap_rad": x[:, 49],
        "throttle_command_gap_norm": x[:, 50],
        "imu_yaw_rate_change_radps": x[:, 51],
        "gt_front_left_sy_diagnostic_only": slip[:, 2],
        "gt_front_right_sy_diagnostic_only": slip[:, 3],
        "front_sy_abs_max_diagnostic_only": np.max(np.abs(slip[:, 2:4]), axis=1),
        "gt_rear_left_sx_diagnostic_only": slip[:, 0],
        "gt_rear_right_sx_diagnostic_only": slip[:, 1],
        "rear_sy_abs_max_diagnostic_only": np.max(np.abs(slip[:, 4:6]), axis=1),
        "estimated_front_left_sy": estimated_slip[:, 0],
        "estimated_front_right_sy": estimated_slip[:, 1],
        "estimated_front_sy_abs_max": np.max(np.abs(estimated_slip), axis=1),
        "estimated_front_sy_change_abs_max": np.max(
            np.abs(estimated_slip - estimated_previous), axis=1),
        "estimated_previous_slip_valid": estimated_previous_valid.astype(np.float64),
        "gt_front_sy_previous_abs_max_diagnostic_only": np.max(
            np.abs(true_previous), axis=1),
        "gt_front_sy_change_abs_max_diagnostic_only": np.where(
            true_previous_valid,
            np.max(np.abs(slip[:, 2:4] - true_previous), axis=1), np.nan),
        "gt_previous_slip_valid_diagnostic_only": true_previous_valid.astype(np.float64),
        "baseline_signed_error_radps": baseline_error,
        "candidate_signed_error_radps": candidate_error,
        "candidate_abs_error_radps": absolute,
        "baseline_to_candidate_abs_error_change_radps": (
            np.abs(baseline_error) - absolute),
        "gt_next_yaw_rate_rps_diagnostic_only": (
            np.asarray(data["gt_yaw_rate"], dtype=np.float64)
            + np.asarray(data["y"], dtype=np.float64)),
    }

    def distribution(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
        selected = np.asarray(values, dtype=np.float64)[mask]
        selected = selected[np.isfinite(selected)]
        if not len(selected):
            return {"samples": 0}
        return {
            "samples": int(len(selected)),
            "mean": float(np.mean(selected)),
            "median": float(np.median(selected)),
            "p90": float(np.quantile(selected, 0.90)),
            "p95": float(np.quantile(selected, 0.95)),
            "min": float(np.min(selected)),
            "max": float(np.max(selected)),
        }

    cohort_masks = {
        "all": np.ones(len(absolute), dtype=bool),
        "candidate_abs_error_gt_0p1": large,
        "candidate_abs_error_le_0p1": ~large,
        "candidate_worst_1_percent": top,
    }
    cohort_summary = {
        cohort: {name: distribution(values, mask)
                 for name, values in quantities.items()}
        for cohort, mask in cohort_masks.items()
    }

    by_event: dict[str, Any] = {}
    for event in sorted(np.unique(data["phase_event"].astype(str))):
        event_mask = data["phase_event"].astype(str) == event
        by_event[event] = {
            "all": _metric(candidate_error[event_mask]),
            "large_error_rows": int(np.count_nonzero(event_mask & large)),
            "large_error_fraction": float(np.mean(large[event_mask])),
            "worst_1_percent": _metric(candidate_error[event_mask & top]),
        }

    age_edges = np.asarray((-1000, -500, -250, -200, -150, -100, -75,
                            -50, -25, 0, 25, 50, 75, 100, 125, 150, 175,
                            200, 225, 250, 275, 300, 350, 400, 500, 750,
                            1000, np.inf), dtype=np.float64)
    event_age = np.asarray(data["event_age_ms"], dtype=np.float64)
    age_bin = np.searchsorted(age_edges, event_age, side="right") - 1
    by_event_age: dict[str, Any] = {}
    event_labels = data["phase_event"].astype(str)
    for event in sorted(np.unique(event_labels)):
        rows_by_age: dict[str, Any] = {}
        for bin_index in range(len(age_edges) - 1):
            mask = (event_labels == event) & (age_bin == bin_index)
            if not np.any(mask):
                continue
            lo, hi = age_edges[bin_index:bin_index + 2]
            label = f"{lo:g}_to_{hi:g}_ms"
            rows_by_age[label] = {
                "candidate": _metric(candidate_error[mask]),
                "sensor_only": _metric(baseline_error[mask]),
                "candidate_large_error_fraction": float(np.mean(large[mask])),
            }
        by_event_age[event] = rows_by_age

    # Find sustained residual bursts, not merely isolated extreme samples.
    # A 35-ms gap allows small timestamp jitter around the nominal 25-ms grid.
    clusters: list[dict[str, Any]] = []
    phase_ids = data["phase_id"].astype(str)
    for phase_id in np.unique(phase_ids):
        phase_rows = np.flatnonzero(phase_ids == phase_id)
        phase_rows = phase_rows[np.argsort(event_age[phase_rows], kind="stable")]
        run: list[int] = []
        for row in phase_rows:
            if large[row] and (not run or event_age[row] - event_age[run[-1]] <= 35.0):
                run.append(int(row))
            else:
                if len(run) >= 2:
                    values = np.abs(candidate_error[run])
                    clusters.append({
                        "phase_id": phase_id,
                        "phase_event": str(event_labels[run[0]]),
                        "start_age_ms": float(event_age[run[0]]),
                        "end_age_ms": float(event_age[run[-1]]),
                        "samples": len(run),
                        "peak_abs_error_radps": float(np.max(values)),
                        "mean_abs_error_radps": float(np.mean(values)),
                    })
                run = [int(row)] if large[row] else []
        if len(run) >= 2:
            values = np.abs(candidate_error[run])
            clusters.append({
                "phase_id": phase_id,
                "phase_event": str(event_labels[run[0]]),
                "start_age_ms": float(event_age[run[0]]),
                "end_age_ms": float(event_age[run[-1]]),
                "samples": len(run),
                "peak_abs_error_radps": float(np.max(values)),
                "mean_abs_error_radps": float(np.mean(values)),
            })
    cluster_sizes: dict[str, int] = {}
    for cluster in clusters:
        size = str(cluster["samples"])
        cluster_sizes[size] = cluster_sizes.get(size, 0) + 1
    clusters.sort(key=lambda row: (row["peak_abs_error_radps"], row["samples"]),
                  reverse=True)

    # Persist all threshold failures, sorted worst first. This gives a direct
    # audit trail instead of just aggregate counts and lets the next fit target
    # repeatable residual structure without reopening/re-aligning the bag.
    if candidate_key == "estimated_front_slip_plus_prior_25ms":
        candidate_slug = "slip_candidate"
    elif candidate_key == "selected_residual_specialists_plus_direct_yaw_regimes":
        candidate_slug = "slip_selected_residual_plus_direct_yaw_regimes"
    elif "selected_regional_specialists" in candidate_key:
        candidate_slug = "slip_selected_regional_specialists"
    else:
        candidate_slug = "slip_event_age_residual_candidate"
    csv_path = output_dir / (
        f"{validation_run}_{candidate_slug}_errors_over_0p1.csv")
    columns = (
        "row", "run_id", "phase_id", "condition_id", "phase_event",
        "causal_event", "event_age_ms", "gt_speed_mps_diagnostic_only",
        "steering_feedback_rad", "gt_front_left_sy_diagnostic_only",
        "gt_front_right_sy_diagnostic_only", "gt_rear_left_sx_diagnostic_only",
        "gt_rear_right_sx_diagnostic_only", "gt_rear_sy_abs_max_diagnostic_only",
        "estimated_front_left_sy", "estimated_front_right_sy",
        "estimated_front_sy_change_abs_max", "estimated_previous_slip_valid",
        "rear_wheel_mean_minus_gt_speed_mps_diagnostic_only",
        "steering_feedback_rate_radps", "throttle_feedback_rate_per_s",
        "steering_command_gap_rad", "throttle_command_gap_norm",
        "imu_ay_mps2", "imu_yaw_rate_rps", "imu_roll_rad",
        "imu_roll_rate_rps", "baseline_signed_error_radps",
        "candidate_signed_error_radps", "candidate_abs_error_radps",
        "baseline_to_candidate_abs_error_change_radps",
        "gt_next_yaw_rate_rps_diagnostic_only",
    )
    order = np.flatnonzero(large)
    order = order[np.argsort(absolute[order])[::-1]]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in order:
            record: dict[str, Any] = {
                "row": int(row),
                "run_id": str(data["run_id"][row]),
                "phase_id": str(data["phase_id"][row]),
                "condition_id": str(data["condition_id"][row]),
                "phase_event": str(data["phase_event"][row]),
                "causal_event": str(data["causal_event"][row]),
                "event_age_ms": float(data["event_age_ms"][row]),
            }
            record.update({name: float(values[row]) for name, values in quantities.items()
                           if name in columns})
            writer.writerow(record)

    return {
        "candidate": candidate_key,
        "threshold_radps": 0.1,
        "large_error_rows": int(np.count_nonzero(large)),
        "large_error_fraction": float(np.mean(large)),
        "csv": str(csv_path),
        "cohort_feature_distributions": cohort_summary,
        "by_phase_event": by_event,
        "by_phase_event_and_event_age": by_event_age,
        "consecutive_large_error_clusters": {
            "continuity_gap_limit_ms": 35.0,
            "clusters_by_sample_count": cluster_sizes,
            "multi_sample_cluster_count": len(clusters),
            "largest_20": clusters[:20],
        },
        "interpretation_warning": (
            "GT state and tire-slip quantities are post-fit diagnosis only; "
            "association does not establish causation or runtime availability."),
    }


def _slip_error(data: dict[str, np.ndarray], prediction: np.ndarray
                ) -> dict[str, Any]:
    truth = data["truth_slip"][:, 2:4]
    valid = np.all(data["truth_slip"][:, 8:10] > 0.5, axis=1)
    error = (prediction - truth)[valid]
    return {
        "rows": int(len(error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs": float(np.quantile(np.abs(error), 0.95)),
    }


def _run(output: Path, validation_run: str = "r02",
         odometry_inputs_only: bool = False) -> dict[str, Any]:
    run_names = ("r01", "r03", validation_run)
    canonical_r04_cache = (ROOT / "live_runs/racing_model_diagnostics_20261009/"
                           "sensor_slip_relaxation_r04/exact_two_row_cache")
    cache_dir = (canonical_r04_cache if validation_run == "r04"
                 and canonical_r04_cache.is_dir()
                 else output.parent / "exact_two_row_cache")
    runs = {name: _load(name) for name in ("r01", "r03")}
    if validation_run == "r02":
        runs["r02"] = _load("r02")
    elif validation_run == "r04":
        runs["r04"] = _load_midpoint_r04(cache_dir)
    else:
        raise ValueError(f"unsupported validation run: {validation_run}")
    if odometry_inputs_only:
        runs = {name: _odometry_input_view(data)
                for name, data in runs.items()}
    if tuple(runs) != run_names:
        raise AssertionError(f"unexpected run order: {tuple(runs)}")
    predicted: dict[str, np.ndarray] = {}
    observer_oof: dict[str, Any] = {}
    for held, fit in (("r01", "r03"), ("r03", "r01")):
        model = _predict_front(runs[fit], runs[fit]["truth_slip"])
        prediction = model.predict(
            evaluator._model_features(runs[held])).astype(np.float32)
        predicted[held] = prediction
        observer_oof[held] = _slip_error(runs[held], prediction)

    train = _combine([runs["r01"], runs["r03"]])
    train_prediction = np.concatenate((predicted["r01"], predicted["r03"]))
    validation = runs[validation_run]
    final_observer = _predict_front(train, train["truth_slip"])
    validation_prediction = final_observer.predict(
        evaluator._model_features(validation)).astype(np.float32)
    validation_slip_metrics = _slip_error(validation, validation_prediction)

    train_previous, train_previous_valid, train_lag_audit = _previous_prediction(
        train, train_prediction)
    val_previous, val_previous_valid, val_lag_audit = _previous_prediction(
        validation, validation_prediction)
    true_train_previous, true_train_valid, _ = _previous_prediction(
        train, train["truth_slip"][:, 2:4])
    true_val_previous, true_val_valid, _ = _previous_prediction(
        validation, validation["truth_slip"][:, 2:4])

    estimated_current_train = _front_features(
        train_prediction, train_previous, train_previous_valid, lagged=False)
    estimated_current_val = _front_features(
        validation_prediction, val_previous, val_previous_valid, lagged=False)
    estimated_lag_train = _front_features(
        train_prediction, train_previous, train_previous_valid, lagged=True)
    estimated_lag_val = _front_features(
        validation_prediction, val_previous, val_previous_valid, lagged=True)

    estimated_history_by_run: dict[str, np.ndarray] = {}
    for run_id in ("r01", "r03"):
        run_previous, run_previous_valid, _ = _previous_prediction(
            runs[run_id], predicted[run_id])
        estimated_history_by_run[run_id] = _front_features(
            predicted[run_id], run_previous, run_previous_valid, lagged=True)
    age_residual_model, yaw_oof_residuals, residual_feature_by_run, age_residual_audit = (
        _fit_age_residual_correction(runs, estimated_history_by_run))

    truth_current_train = _front_features(
        train["truth_slip"][:, 2:4], true_train_previous,
        true_train_valid, lagged=False)
    truth_current_val = _front_features(
        validation["truth_slip"][:, 2:4], true_val_previous,
        true_val_valid, lagged=False)
    truth_lag_train = _front_features(
        train["truth_slip"][:, 2:4], true_train_previous,
        true_train_valid, lagged=True)
    truth_lag_val = _front_features(
        validation["truth_slip"][:, 2:4], true_val_previous,
        true_val_valid, lagged=True)

    baseline_prediction = _fit_yaw(train, validation, None, None, lagged=False)
    estimated_current_prediction = _fit_yaw(
        train, validation, estimated_current_train, estimated_current_val,
        lagged=False)
    estimated_lag_prediction = _fit_yaw(
        train, validation, estimated_lag_train, estimated_lag_val,
        lagged=True)
    oracle_current_prediction = _fit_yaw(
        train, validation, truth_current_train, truth_current_val,
        lagged=False)
    oracle_lag_prediction = _fit_yaw(
        train, validation, truth_lag_train, truth_lag_val, lagged=True)
    age_residual_x, age_residual_mirror_x = _age_residual_features(
        validation, estimated_lag_val)
    age_residual_correction = age_residual_model.predict(
        age_residual_x).astype(np.float32)
    regional_correction, selected_regional_correction, regional_audit = (
        _evaluate_regional_residual_specialists(
            runs, residual_feature_by_run, yaw_oof_residuals, validation,
            (age_residual_x, age_residual_mirror_x), age_residual_correction))
    age_residual_prediction = estimated_lag_prediction + age_residual_correction
    regional_all_prediction = estimated_lag_prediction + regional_correction
    regional_selected_prediction = (
        estimated_lag_prediction + selected_regional_correction)
    direct_regional_prediction, direct_regional_audit = (
        _evaluate_direct_yaw_specialists(
            runs, validation, regional_selected_prediction))

    baseline_error = baseline_prediction - validation["y"]
    errors = {
        "sensor_only": baseline_error,
        "estimated_front_slip_current": estimated_current_prediction - validation["y"],
        "estimated_front_slip_plus_prior_25ms": estimated_lag_prediction - validation["y"],
        "GT_front_slip_current_oracle": oracle_current_prediction - validation["y"],
        "GT_front_slip_plus_prior_25ms_oracle": oracle_lag_prediction - validation["y"],
        "estimated_front_slip_plus_prior_25ms_plus_event_age_residual": (
            age_residual_prediction - validation["y"]),
        "estimated_front_slip_plus_prior_25ms_plus_event_age_regional_specialists": (
            regional_all_prediction - validation["y"]),
        "estimated_front_slip_plus_prior_25ms_plus_event_selected_regional_specialists": (
            regional_selected_prediction - validation["y"]),
        "selected_residual_specialists_plus_direct_yaw_regimes": (
            direct_regional_prediction - validation["y"]),
    }
    for region, mask in _regional_residual_gates(validation).items():
        if region in regional_audit:
            regional_audit[region]["validation_global_age_model_error"] = _metric(
                errors["estimated_front_slip_plus_prior_25ms_plus_event_age_residual"][mask])
            regional_audit[region]["validation_all_local_experts_error"] = _metric(
                errors["estimated_front_slip_plus_prior_25ms_plus_event_age_regional_specialists"][mask])
            regional_audit[region]["validation_cv_selected_experts_error"] = _metric(
                errors["estimated_front_slip_plus_prior_25ms_plus_event_selected_regional_specialists"][mask])
    events = validation["phase_event"].astype(str)
    gate = np.max(np.abs(validation_prediction), axis=1) >= FRONT_PEAK_GATE
    gate_errors = {}
    for name in ("estimated_front_slip_current",
                 "estimated_front_slip_plus_prior_25ms"):
        selected = errors[name].copy()
        selected[~gate] = baseline_error[~gate]
        gate_errors[name + "_fixed_peak_gated"] = selected

    routed_error = None
    if not odometry_inputs_only:
        # This regime rule was fixed from r02 before r04 was scored.
        routed_error = baseline_error.copy()
        routed_error[events == "turn_in"] = errors[
            "estimated_front_slip_plus_prior_25ms"][events == "turn_in"]
        routed_error[events == "unwind"] = errors[
            "estimated_front_slip_current"][events == "unwind"]
        routed_error[events == "reversal"] = gate_errors[
            "estimated_front_slip_plus_prior_25ms_fixed_peak_gated"][events == "reversal"]
        errors["event_routed_sensor_candidate"] = routed_error

        # Causal sensor-derived phase class; unlike phase_event, this is part
        # of the feature contract and can be computed online from actuator
        # feedback/command history.
        causal_events = validation["causal_event"].astype(str)
        causal_routed_error = baseline_error.copy()
        causal_routed_error[causal_events == "turn_in"] = errors[
            "estimated_front_slip_plus_prior_25ms"][causal_events == "turn_in"]
        causal_routed_error[causal_events == "unwind"] = errors[
            "estimated_front_slip_current"][causal_events == "unwind"]
        causal_routed_error[causal_events == "reversal"] = gate_errors[
            "estimated_front_slip_plus_prior_25ms_fixed_peak_gated"][
                causal_events == "reversal"]
        errors["causal_event_routed_sensor_candidate"] = causal_routed_error

    by_event: dict[str, Any] = {}
    for event in sorted(np.unique(events)):
        mask = events == event
        by_event[event] = {name: _metric(error[mask])
                           for name, error in {**errors, **gate_errors}.items()}

    speed_edges = tuple(float(value) for value in range(13)) + (12.01,)
    steer_edges = tuple(value / 100.0 for value in range(0, 51, 5)) + (0.501,)
    regime_table: dict[str, Any] = {}
    speed = validation["gt_speed"].astype(np.float64)
    steering = np.abs(validation["steering"].astype(np.float64))
    for low_speed, high_speed in zip(speed_edges[:-1], speed_edges[1:]):
        for low_steer, high_steer in zip(steer_edges[:-1], steer_edges[1:]):
            base_mask = ((speed >= low_speed) & (speed < high_speed)
                         & (steering >= low_steer) & (steering < high_steer))
            for event in sorted(np.unique(events)):
                mask = base_mask & (events == event)
                if not np.any(mask):
                    continue
                key = (f"speed_{low_speed:g}_{high_speed:g}__"
                       f"abs_steer_{low_steer:g}_{high_steer:g}__{event}")
                regime_table[key] = {
                    "samples": int(mask.sum()),
                    "models": {name: _metric(error[mask])
                               for name, error in {**errors, **gate_errors}.items()},
                }

    residual_diagnostics = _residual_diagnostics(
        validation, errors, validation_prediction, val_previous,
        val_previous_valid, true_val_previous, true_val_valid,
        output.parent / "residual_diagnostics", validation_run)
    age_residual_diagnostics = _residual_diagnostics(
        validation, errors, validation_prediction, val_previous,
        val_previous_valid, true_val_previous, true_val_valid,
        output.parent / "residual_diagnostics", validation_run,
        candidate_key=(
            "estimated_front_slip_plus_prior_25ms_plus_event_age_residual"))
    selected_regional_diagnostics = _residual_diagnostics(
        validation, errors, validation_prediction, val_previous,
        val_previous_valid, true_val_previous, true_val_valid,
        output.parent / "residual_diagnostics", validation_run,
        candidate_key=(
            "estimated_front_slip_plus_prior_25ms_plus_event_selected_regional_specialists"))
    direct_regional_diagnostics = _residual_diagnostics(
        validation, errors, validation_prediction, val_previous,
        val_previous_valid, true_val_previous, true_val_valid,
        output.parent / "residual_diagnostics", validation_run,
        candidate_key="selected_residual_specialists_plus_direct_yaw_regimes")

    report = {
        "objective": "replace the GT front-slip-history oracle with a causal sensor-only front-slip observer and test its one-step yaw value",
        "status": "offline development candidate; not production-integrated",
        "input_profile": (
            "current C++ odometry inputs only: rear encoder speeds and IMU ax/ay/yaw rate/history; no steering, throttle, roll, or phase-event inputs"
            if odometry_inputs_only else
            "all available causal sensor, actuator feedback/command, and event features"),
        "training_runs": [RUN_ID["r01"], RUN_ID["r03"]],
        "whole_run_oof_front_slip_observer": observer_oof,
        "whole_run_oof_event_age_residual_correction": {
            **age_residual_audit,
            "fold_residual_rmse_by_run": {
                RUN_ID[run_id]: _metric(residual)
                for run_id, residual in yaw_oof_residuals.items()},
        },
        "cross_run_tested_regional_residual_specialists": regional_audit,
        "cross_run_tested_direct_yaw_regime_specialists": direct_regional_audit,
        "validation_run": RUN_ID.get(
            validation_run, evaluator.FULL_VALIDATION_REPLICATION_ID),
        f"{validation_run}_front_slip_estimator": validation_slip_metrics,
        f"{validation_run}_yaw_metrics": {name: _metric(error)
                                          for name, error in {**errors, **gate_errors}.items()},
        f"{validation_run}_large_residual_diagnostics": residual_diagnostics,
        f"{validation_run}_event_age_residual_large_error_diagnostics": (
            age_residual_diagnostics),
        f"{validation_run}_selected_regional_large_error_diagnostics": (
            selected_regional_diagnostics),
        f"{validation_run}_direct_regional_large_error_diagnostics": (
            direct_regional_diagnostics),
        f"{validation_run}_yaw_by_event": by_event,
        f"{validation_run}_yaw_by_speed_abs_steering_event": regime_table,
        "event_routed_candidate_rule": {
            "availability": (
                "offline upper bound only; phase routing uses experiment annotations, not an online sensor-derived classifier"),
            "routing_signal": "phase_event schedule annotation",
            "turn_in": "estimated current + prior 25-ms front slip, ungated",
            "unwind": "estimated current front slip, ungated",
            "reversal": "estimated current + prior 25-ms front slip, gated at max(|Sy|)>=0.01; otherwise baseline",
            "hold_or_unknown": "sensor-only baseline",
            "selection_frozen_from": "r02 event decomposition before r04 transfer score",
            "routed_score": (_metric(routed_error)
                             if routed_error is not None else None),
            "not_evaluated_reason": (
                "phase/event labels are unavailable to the current odometry input path"
                if odometry_inputs_only else None),
        },
        "causal_event_routed_candidate_rule": {
            "availability": (
                "not evaluated because current odometry inputs omit actuator channels"
                if odometry_inputs_only else
                "sensor/actuator-history-derived event class; deployable only where those inputs are permitted and source-time aligned"),
            "routing_signal": "causal_event feature derived from current/past command and feedback",
            "turn_in": "estimated current + prior 25-ms front slip",
            "unwind": "estimated current front slip",
            "reversal": "estimated current + prior 25-ms front slip, peak-gated at |Sy|>=0.01",
            "hold_or_unknown": "sensor-only baseline",
            "routed_score": (_metric(errors["causal_event_routed_sensor_candidate"])
                             if "causal_event_routed_sensor_candidate" in errors
                             else None),
        },
        "lag_alignment": {"train": train_lag_audit,
                           validation_run: val_lag_audit},
        "gate": {
            "definition": "max absolute sensor-estimated front Sy >= 0.01",
            "threshold_source": "documented front-lateral tire-curve extremum, fixed before this screen",
            f"selected_fraction_{validation_run}": float(gate.mean()),
        },
        "method": {
            "slip_observer_input": "current/past allowed sensor, actuator feedback, command history and causal event only",
            "training_feature_predictions": "whole-run OOF: observer fit on r03 predicts r01 and vice versa",
            "GT_usage": "labels and evaluation only; GT front-slip oracle rows are separate diagnostic controls",
            "lag": "previous exact-two row in same transition phase, fixed 25-ms grid and <=12-ms timing deviation",
            "yaw_model": evaluator.MODEL,
            "feature_mask": ("current C++ odometry input subset"
                             if odometry_inputs_only else "full causal feature contract"),
        },
        "limitations": [
            "The selected validation capture is held out from fitting but may have informed prior hypothesis selection; treat its transfer result as independent-run evidence, not a final untouched benchmark.",
            "Two training captures are insufficient for reliable per-regime run-level uncertainty.",
            "A low one-step yaw residual does not by itself prove accurate recursive odometry, full-lap drift, or MPC prediction.",
            "The exact-two capture contains onset/unwind/reversal windows, not steady hold coverage.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validation-run", choices=("r02", "r04"), default="r02")
    parser.add_argument(
        "--odometry-inputs-only", action="store_true",
        help="mask model inputs to signals currently passed into C++ odometry")
    args = parser.parse_args()
    output = args.output or (
        DEFAULT_OUTPUT if args.validation_run == "r02" else
        ROOT / "live_runs/racing_model_diagnostics_20261009/"
        "sensor_slip_relaxation_r04/report.json")
    report = _run(output.resolve(), validation_run=args.validation_run,
                  odometry_inputs_only=args.odometry_inputs_only)
    print(json.dumps({
        "report": str(output.resolve()),
        "whole_run_oof_front_slip_observer": report[
            "whole_run_oof_front_slip_observer"],
        "validation_run": report["validation_run"],
        "front_slip_estimator": report[f"{args.validation_run}_front_slip_estimator"],
        "yaw_metrics": report[f"{args.validation_run}_yaw_metrics"],
        "yaw_by_event": report[f"{args.validation_run}_yaw_by_event"],
        "direct_yaw_regime_specialists": report[
            "cross_run_tested_direct_yaw_regime_specialists"],
        "event_routed_candidate_rule": report["event_routed_candidate_rule"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
