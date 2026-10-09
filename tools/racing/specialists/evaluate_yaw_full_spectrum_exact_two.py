#!/usr/bin/env python3
"""Train and evaluate causal yaw experts on exact-two full-spectrum captures.

Only reset-isolated steering response windows whose command-to-feedback delay
is exactly two contiguous packets are admitted. Simulator truth is used only
for the next-step yaw-rate target and offline stratification. This tool does
not modify runtime odometry or MPC.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

try:
    import audit_yaw_full_domain_exact_two as exact_two
    import evaluate_yaw_exact2_speed_steering_local_models as prior
    import evaluate_yaw_packet_response_conditioned_models as response
    import evaluate_yaw_throttle_surface_augmentation as throttle_source
    import evaluate_yaw_predicted_next_steering as causal
    import train_yaw_multihorizon_teacher as teacher
    import fit_sensor_only_yaw_regime_atlas as atlas
except ModuleNotFoundError:
    from tools.racing.specialists import audit_yaw_full_domain_exact_two as exact_two
    from tools.racing.specialists import evaluate_yaw_exact2_speed_steering_local_models as prior
    from tools.racing.specialists import evaluate_yaw_packet_response_conditioned_models as response
    from tools.racing.specialists import evaluate_yaw_throttle_surface_augmentation as throttle_source
    from tools.racing.specialists import evaluate_yaw_predicted_next_steering as causal
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "live_runs/racing_model_diagnostics_20261009/yaw_full_spectrum_exact_two"
FULL_TRAIN_ID = "openplane_yaw_full_spectrum_grid_train_20261009_r01"
FULL_VALIDATION_ID = "openplane_yaw_full_spectrum_grid_validation_20261009_r02"
FULL_TRAIN_REPLICATION_ID = "openplane_yaw_full_spectrum_grid_train_20261009_r03"
FULL_VALIDATION_REPLICATION_ID = (
    "openplane_yaw_full_spectrum_midpoint_validation_20261009_r04")
ENVELOPE_TRAIN_ID = "openplane_yaw_high_speed_envelope_gapfill_train_20261009_r02"
ENVELOPE_VALIDATION_ID = "openplane_yaw_high_speed_envelope_gapfill_validation_20261009_r03"
SPARSE_SUPPORT_TRAIN_ID = "openplane_yaw_sparse_cell_support_train_20261009_r02"
SPARSE_SUPPORT_VALIDATION_ID = "openplane_yaw_sparse_cell_support_validation_20261009_r03"
SPARSE_CRAWL_TRAIN_ID = "openplane_yaw_sparse_crawl_cell_support_train_20261009_r01"
SPARSE_CRAWL_VALIDATION_ID = "openplane_yaw_sparse_crawl_cell_support_validation_20261009_r02"
SUPPORT_REPLICATION_TRAIN_ID = "openplane_yaw_exact_two_support_replication_train_20261009_r01"
SUPPORT_REPLICATION_VALIDATION_ID = "openplane_yaw_exact_two_support_replication_validation_20261009_r02"
THROTTLE_TRAIN_ID = throttle_source.TRAIN_RUN
THROTTLE_VALIDATION_ID = throttle_source.VALIDATION_RUN
EVENTS = ("hold", "turn_in", "unwind", "reversal")
SPEED_BIN_MPS = 0.5
STEERING_BIN_RAD = 0.05
MIN_LOCAL_ROWS = 80
MIN_LOCAL_PHASES = 4
MIN_LOCAL_INDEPENDENT_RUNS = 2
MODEL = {
    **causal.MODEL,
    "n_estimators": 96,
    "max_depth": 10,
    "min_samples_leaf": 5,
}


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    if not len(error):
        return {"samples": 0}
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error * error))),
        "mae_radps": float(np.mean(absolute)),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "p99_abs_radps": float(np.quantile(absolute, 0.99)),
        "max_abs_radps": float(np.max(absolute)),
        "samples_over_0p1": int(np.count_nonzero(absolute > 0.1)),
        "fraction_within_0p1": float(np.mean(absolute <= 0.1)),
    }


def _nonresponse_quality_exceptions(run_id: str, phases: list[Any],
                                    experiment_end: dict[str, Any]
                                    ) -> list[str]:
    """Recognize only the known crawl approach sample-count artifact.

    Crawl targets are reached before 30 samples accrue in approach phases.
    They contain no steering-response labels; every probe and other gate
    must still pass before those rows can be used.
    """
    failures = list(experiment_end.get("quality_failures", []))
    if not failures:
        return []
    if run_id not in {SPARSE_SUPPORT_TRAIN_ID,
                      SPARSE_SUPPORT_VALIDATION_ID}:
        return []
    prefix = "approach_yawsupport_v0.25_"
    approaches = [phase for phase in phases
                  if phase.label.startswith(prefix)]
    expected = sorted(f"{phase.label}:samples<30" for phase in approaches)
    invalid_labels = {phase.label for phase in phases
                      if phase.valid is not True}
    expected_labels = {phase.label for phase in approaches}
    probes_valid = all(
        phase.valid is True for phase in phases
        if phase.label.startswith("probe_yawsupport_"))
    if (expected and sorted(failures) == expected
            and invalid_labels == expected_labels and probes_valid):
        return expected
    return []


def _collect_run(series: Any, history_lags: tuple[int, ...]
                 ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # Use the same selected causal sensor history for yaw and throttle sources.
    # Timing metadata is deliberately not an input: the throttle archive has
    # no equivalent receipt/source-stamp fields.
    rows = response._run_rows(series, history_lags=history_lags)
    bag = exact_two.alignment._bag_for_run(series)
    phases, experiment_end = exact_two._read_phases(bag)
    accepted_nonresponse_failures = _nonresponse_quality_exceptions(
        series.run_id, phases, experiment_end)
    if series.run_id in {FULL_TRAIN_ID, FULL_VALIDATION_ID,
                         ENVELOPE_TRAIN_ID, ENVELOPE_VALIDATION_ID,
                         SPARSE_SUPPORT_TRAIN_ID,
                         SPARSE_SUPPORT_VALIDATION_ID,
                         SPARSE_CRAWL_TRAIN_ID,
                         SPARSE_CRAWL_VALIDATION_ID,
                         SUPPORT_REPLICATION_TRAIN_ID,
                         SUPPORT_REPLICATION_VALIDATION_ID,
                         FULL_TRAIN_REPLICATION_ID,
                         FULL_VALIDATION_REPLICATION_ID}:
        if (experiment_end.get("aborted") is not False
                or experiment_end.get("reason") != "schedule complete"
                or (experiment_end.get("quality_failures", [])
                    and not accepted_nonresponse_failures)):
            raise ValueError(
                f"{series.run_id}: full-spectrum capture did not pass its "
                "complete, non-aborted schedule quality gate")
    measured = exact_two._measure_run(series, phases, experiment_end)
    phase_by_label = {phase.label: phase for phase in phases}
    times = rows["sample_time_ns"]
    records: dict[str, list[Any]] = defaultdict(list)
    transition_counts: Counter[str] = Counter()

    for item in measured["measurements"]:
        stimulus = item["stimulus"]
        if stimulus.get("channel") != "steering":
            continue
        count = item.get("response_packet_count")
        contiguous = item.get("packet_sequence_contiguous") is True
        if count != 2 or not contiguous:
            transition_counts[str(count if count is not None else
                                  item.get("classification", "unknown"))] += 1
            continue
        start_ns = item.get("command_start_ns")
        phase = phase_by_label.get(str(item["phase_label"]))
        if not isinstance(start_ns, int) or phase is None or phase.valid is not True:
            transition_counts["invalid_exact_two_metadata"] += 1
            continue
        selected = np.flatnonzero(
            (times >= max(int(phase.start_ns), start_ns - response.WINDOW_BEFORE_NS))
            & (times <= min(int(phase.end_ns), start_ns + response.WINDOW_AFTER_NS))
        )
        if not len(selected):
            transition_counts["empty_exact_two_window"] += 1
            continue

        stimulus_event = str(stimulus.get("event", "unknown"))
        target_event = "turn_in" if stimulus_event == "onset" else stimulus_event
        if target_event not in EVENTS:
            transition_counts[f"unsupported_event_{target_event}"] += 1
            continue
        phase_id = (f"{series.run_id}:{item['phase_label']}::"
                    f"transition{stimulus.get('transition_index', 0)}")
        condition_id = f"{series.run_id}:{item['phase_label']}"
        records["x"].append(rows["x_extended"][selected])
        records["y"].append(rows["residual"][selected])
        records["wheel_speed"].append(rows["wheel_speed"][selected])
        records["steering"].append(rows["steering"][selected])
        records["gt_speed"].append(rows["gt_speed"][selected])
        records["gt_yaw_rate"].append(rows["gt_yaw_current"][selected])
        body_velocity = rows["gt_body_velocity"][selected]
        records["gt_body_state"].append(np.column_stack((
            body_velocity,
            np.arctan2(body_velocity[:, 1], body_velocity[:, 0]),
        )).astype(np.float32))
        records["causal_event"].append(rows["event"][selected].astype(str))
        records["phase_event"].append(np.full(len(selected), target_event))
        records["phase_id"].append(np.full(len(selected), phase_id))
        records["condition_id"].append(np.full(len(selected), condition_id))
        records["run_id"].append(np.full(len(selected), series.run_id))
        records["event_age_ms"].append(
            (times[selected] - int(start_ns)) / 1.0e6)
        transition_counts["exact_two_rows_admitted"] += len(selected)
        transition_counts[f"exact_two_{target_event}_events"] += 1

    if not records["x"]:
        raise ValueError(f"{series.run_id}: no exact-two steering rows admitted")
    result = {key: np.concatenate(values) for key, values in records.items()}
    if not np.isfinite(result["x"]).all() or not np.isfinite(result["y"]).all():
        raise ValueError(f"{series.run_id}: non-finite causal features or labels")
    return result, {
        "run_id": series.run_id,
        "split": series.split,
        "bag": str(bag.relative_to(ROOT)),
        "experiment_end": experiment_end,
        "phase_count": int(len(phases)),
        "response_measurements": int(len(measured["measurements"])),
        "exact_two_audit": dict(transition_counts),
        "quality": {
            "aborted": bool(experiment_end.get("aborted", True)),
            "reason": experiment_end.get("reason"),
            "quality_failures": experiment_end.get("quality_failures", []),
            "accepted_nonresponse_phase_quality_failures": (
                accepted_nonresponse_failures),
        },
    }


def _collect_throttle_runs(
        history_lags: tuple[int, ...]
        ) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    source_runs, audit = throttle_source._load_throttle_runs(history_lags)
    converted: dict[str, dict[str, np.ndarray]] = {}
    for run_id, data in source_runs.items():
        sequence = data["sequence_id"].astype(np.int64)
        phase_ids = np.asarray(
            [f"{run_id}:sequence{value}" for value in sequence], dtype="U160")
        events = data["event"].astype(str)
        converted[run_id] = {
            "x": data["x"].astype(np.float32),
            "y": data["residual"].astype(np.float32),
            "wheel_speed": data["wheel_speed"].astype(np.float32),
            "steering": data["steering"].astype(np.float32),
            "gt_speed": data["gt_speed"].astype(np.float32),
            "causal_event": events,
            "phase_event": events,
            "phase_id": phase_ids,
            "condition_id": phase_ids.copy(),
            "run_id": np.full(len(events), run_id, dtype="U128"),
            # Throttle sequences are not steering-event windows. Keep this
            # sentinel out of model inputs; it only labels diagnostic exports.
            "event_age_ms": np.full(len(events), -1.0, dtype=np.float32),
        }
        if converted[run_id]["x"].shape[1] != (
                len(history_lags) * len(atlas.OBSERVATION_NAMES)
                + len(atlas.DERIVED_NAMES)):
            raise ValueError(f"{run_id}: throttle feature schema mismatch")
    return converted, audit


def _combine(runs: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not runs:
        raise ValueError("no runs to combine")
    return {key: np.concatenate([run[key] for run in runs.values()])
            for key in runs[next(iter(runs))]}


def _model_features(data: dict[str, np.ndarray]) -> np.ndarray:
    event = data["causal_event"].astype(str)
    event_code = np.asarray([EVENTS.index(value) if value in EVENTS else 0
                             for value in event], dtype=np.int64)
    event_one_hot = np.eye(len(EVENTS), dtype=np.float32)[event_code]
    return np.column_stack((data["x"], event_one_hot)).astype(np.float32)


def _fit(x: np.ndarray, y: np.ndarray,
         phase_ids: np.ndarray) -> ExtraTreesRegressor:
    sensor_features = x[:, :-len(EVENTS)]
    # The shared archive contract has no four receipt-timing features. Add
    # neutral placeholders only while applying the existing reflection map,
    # then remove them so they are never visible to a fitted model.
    mirror_input = np.column_stack((
        sensor_features, np.zeros((len(sensor_features), 4), dtype=np.float32)))
    mirrored = causal._mirror_features(mirror_input)[:, :-4]
    mirrored = np.column_stack((mirrored, x[:, -len(EVENTS):]))
    fit_x = np.concatenate((x, mirrored), axis=0)
    fit_y = np.concatenate((y, -y), axis=0)
    fit_phase = np.concatenate((phase_ids, phase_ids), axis=0)
    return ExtraTreesRegressor(**MODEL).fit(
        fit_x, fit_y, sample_weight=causal._phase_weights(fit_phase))


def _fit_gt_state_upper_bound(x: np.ndarray, y: np.ndarray,
                              phase_ids: np.ndarray,
                              gt_body_state: np.ndarray
                              ) -> ExtraTreesRegressor:
    """Fit a diagnostic-only model with simulator-truth body u/v and beta."""
    sensor_features = x[:, :-len(EVENTS)]
    event_code = x[:, -len(EVENTS):]
    mirror_input = np.column_stack((
        sensor_features,
        np.zeros((len(sensor_features), 4), dtype=np.float32)))
    mirrored_sensor = causal._mirror_features(mirror_input)[:, :-4]
    mirrored_state = gt_body_state.copy()
    mirrored_state[:, 1:] *= -1.0
    fit_x = np.concatenate((
        np.column_stack((x, gt_body_state)),
        np.column_stack((mirrored_sensor, mirrored_state, event_code)),
    ), axis=0)
    fit_y = np.concatenate((y, -y), axis=0)
    fit_phase = np.concatenate((phase_ids, phase_ids), axis=0)
    return ExtraTreesRegressor(**MODEL).fit(
        fit_x, fit_y, sample_weight=causal._phase_weights(fit_phase))


def _cell_ids(data: dict[str, np.ndarray]) -> np.ndarray:
    speed = np.floor(data["wheel_speed"] / SPEED_BIN_MPS).astype(np.int64)
    steering = np.clip(np.rint(data["steering"] / STEERING_BIN_RAD), -10, 10)
    return speed * 21 + steering.astype(np.int64) + 10


def _predict_local(train: dict[str, np.ndarray], validation: dict[str, np.ndarray],
                   x_train: np.ndarray, x_validation: np.ndarray,
                   global_prediction: np.ndarray
                   ) -> tuple[np.ndarray, dict[str, Any]]:
    train_cells = _cell_ids(train)
    validation_cells = _cell_ids(validation)
    train_phase = train["phase_id"].astype(str)
    train_runs = train["run_id"].astype(str)
    prediction = global_prediction.copy()
    fallback_rows = 0
    support: dict[str, Any] = {}
    for cell in np.unique(validation_cells):
        train_idx = np.flatnonzero(train_cells == cell)
        valid_idx = np.flatnonzero(validation_cells == cell)
        phases = int(len(np.unique(train_phase[train_idx])))
        runs = int(len(np.unique(train_runs[train_idx])))
        supported = (len(train_idx) >= MIN_LOCAL_ROWS
                     and phases >= MIN_LOCAL_PHASES)
        support[str(int(cell))] = {
            "rows": int(len(train_idx)),
            "transition_phases": phases,
            "independent_training_runs": runs,
            "supported": bool(supported),
        }
        if not supported:
            fallback_rows += len(valid_idx)
            continue
        model = _fit(x_train[train_idx], train["y"][train_idx],
                     train_phase[train_idx])
        prediction[valid_idx] = model.predict(x_validation[valid_idx])
    return prediction, {
        "supported_local_cells": int(sum(row["supported"]
                                          for row in support.values())),
        "fallback_rows_to_global_model": int(fallback_rows),
        "cell_support": support,
    }


def _local_event_hybrid(local_prediction: np.ndarray,
                        event_prediction: np.ndarray,
                        validation: dict[str, np.ndarray],
                        local_support: dict[str, Any],
                        minimum_runs: int
                        ) -> tuple[np.ndarray, dict[str, int]]:
    """Require independent-capture support for local experts.

    Cells without enough independent training captures use the causal-event
    model rather than the pooled global fallback. The selector uses only the
    current wheel-speed estimate and measured steering feedback.
    """
    cells = _cell_ids(validation)
    output = np.empty_like(local_prediction)
    local_cells = local_rows = fallback_cells = fallback_rows = 0
    for cell in np.unique(cells):
        mask = cells == cell
        support = local_support["cell_support"].get(str(int(cell)), {})
        use_local = (support.get("supported") is True
                     and int(support.get("independent_training_runs", 0))
                     >= minimum_runs)
        if use_local:
            output[mask] = local_prediction[mask]
            local_cells += 1
            local_rows += int(np.count_nonzero(mask))
        else:
            output[mask] = event_prediction[mask]
            fallback_cells += 1
            fallback_rows += int(np.count_nonzero(mask))
    return output, {
        "minimum_independent_training_runs": int(minimum_runs),
        "local_expert_cells_used": local_cells,
        "local_expert_rows_used": local_rows,
        "causal_event_fallback_cells": fallback_cells,
        "causal_event_fallback_rows": fallback_rows,
    }


def _metric_groups(error: np.ndarray, *groups: np.ndarray
                   ) -> dict[str, Any]:
    output: dict[str, Any] = {}
    keys = np.asarray(["|".join(str(value) for value in values)
                       for values in zip(*groups)], dtype=str)
    unique, inverse = np.unique(keys, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique))
    offsets = np.concatenate(([0], np.cumsum(counts)))
    for index, key in enumerate(unique):
        output[str(key)] = _metric(error[order[offsets[index]:offsets[index + 1]]])
    return output


def _derived(data: dict[str, np.ndarray], name: str,
             history_lags: tuple[int, ...]) -> np.ndarray:
    offset = (len(history_lags)
              * len(atlas.OBSERVATION_NAMES))
    return data["x"][:, offset + atlas.DERIVED_NAMES.index(name)]


def _quartiles(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(values, (0.25, 0.5, 0.75)))
    return np.asarray([f"q{value}" for value in np.digitize(values, edges)],
                      dtype=str)


def _condition_bootstrap(error: np.ndarray, conditions: np.ndarray,
                         seed: int = 20261009, draws: int = 2000
                         ) -> dict[str, Any]:
    unique, inverse = np.unique(conditions.astype(str), return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique))
    squared_error = np.bincount(
        inverse, weights=np.square(error), minlength=len(unique))
    per_condition = np.sqrt(squared_error / np.maximum(counts, 1))
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(per_condition), size=(draws, len(per_condition)))
    bootstrap = per_condition[choices].mean(axis=1)
    return {
        "clusters": "independent speed/steering/sign conditions within each held-out run",
        "condition_count": int(len(unique)),
        "condition_macro_rmse_radps": float(np.mean(per_condition)),
        "condition_bootstrap_95pct_ci_radps": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _write_outliers(path: Path, data: dict[str, np.ndarray],
                    prediction: np.ndarray,
                    history_lags: tuple[int, ...]) -> None:
    error = prediction - data["y"]
    order = np.argsort(np.abs(error))[::-1][:1000]
    names = list(atlas.OBSERVATION_NAMES)
    features = data["x"]
    history_width = len(history_lags) * len(names)
    derived_start = history_width
    derived_index = {name: derived_start + index
                     for index, name in enumerate(atlas.DERIVED_NAMES)}
    fields = ("row_index", "run_id", "condition_id", "phase_event", "causal_event",
              "gt_speed_mps", "wheel_mean_mps", "wheel_body_mismatch_mps",
              "steering_rad", "wheel_split_mps", "steering_rate_radps",
              "throttle_feedback_rate_per_s", "steering_command_gap_rad",
              "throttle_command_gap_norm", "imu_roll_rad", "imu_roll_rate_rps",
              "event_age_ms", "target_yaw_residual_radps",
              "predicted_yaw_residual_radps", "absolute_error_radps")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in order:
            row = {
                "row_index": int(index),
                "run_id": str(data["run_id"][index]),
                "condition_id": str(data["condition_id"][index]),
                "phase_event": str(data["phase_event"][index]),
                "causal_event": str(data["causal_event"][index]),
                "gt_speed_mps": float(data["gt_speed"][index]),
                "wheel_mean_mps": float(data["wheel_speed"][index]),
                "wheel_body_mismatch_mps": float(
                    data["wheel_speed"][index] - data["gt_speed"][index]),
                "steering_rad": float(data["steering"][index]),
                "wheel_split_mps": float(features[index, derived_index[
                    "rear_wheel_split_mps"]]),
                "steering_rate_radps": float(features[index, derived_index[
                    "steering_feedback_rate_radps"]]),
                "throttle_feedback_rate_per_s": float(features[index, derived_index[
                    "throttle_feedback_rate_per_s"]]),
                "steering_command_gap_rad": float(features[index, derived_index[
                    "steering_command_gap_rad"]]),
                "throttle_command_gap_norm": float(features[index, derived_index[
                    "throttle_command_gap_norm"]]),
                "imu_roll_rad": float(features[index, 9]),
                "imu_roll_rate_rps": float(features[index, 10]),
                "event_age_ms": float(data["event_age_ms"][index]),
                "target_yaw_residual_radps": float(data["y"][index]),
                "predicted_yaw_residual_radps": float(prediction[index]),
                "absolute_error_radps": float(abs(error[index])),
            }
            writer.writerow(row)


def evaluate(output: Path,
             history_lags: tuple[int, ...] = atlas.HISTORY_LAGS
             ) -> dict[str, Any]:
    if (not history_lags or history_lags[0] != 0
            or any(lag < 0 for lag in history_lags)
            or tuple(sorted(set(history_lags))) != history_lags):
        raise ValueError(
            "history lags must be unique, increasing, nonnegative, and start at zero")
    admitted, source_audit = teacher._discover_series_with_safe_mixed_archives()
    by_id = {series.run_id: series for series in admitted}
    train_ids = set(prior.TRAIN_IDS) | {
        FULL_TRAIN_ID, ENVELOPE_TRAIN_ID, SPARSE_SUPPORT_TRAIN_ID,
        SPARSE_CRAWL_TRAIN_ID, SUPPORT_REPLICATION_TRAIN_ID,
        FULL_TRAIN_REPLICATION_ID}
    validation_ids = set(prior.VALIDATION_IDS) | {
        FULL_VALIDATION_ID, ENVELOPE_VALIDATION_ID,
        SPARSE_SUPPORT_VALIDATION_ID, SPARSE_CRAWL_VALIDATION_ID,
        SUPPORT_REPLICATION_VALIDATION_ID,
        FULL_VALIDATION_REPLICATION_ID}
    missing = (train_ids | validation_ids) - set(by_id)
    if missing:
        raise ValueError(f"required clean train/validation runs missing: {sorted(missing)}")
    if any(by_id[run_id].split != "train" for run_id in train_ids):
        raise ValueError("a training run is not assigned to the train split")
    if any(by_id[run_id].split != "validation" for run_id in validation_ids):
        raise ValueError("a held-out run is not assigned to validation")

    yaw_train_runs: dict[str, dict[str, np.ndarray]] = {}
    validation_runs: dict[str, dict[str, np.ndarray]] = {}
    gt_body_state_by_run: dict[str, np.ndarray] = {}
    run_audits = {}
    for run_id in sorted(train_ids | validation_ids):
        data, audit = _collect_run(by_id[run_id], history_lags)
        run_audits[run_id] = audit
        gt_body_state_by_run[run_id] = data.pop("gt_body_state")
        (yaw_train_runs if run_id in train_ids else validation_runs)[run_id] = data

    throttle_runs, throttle_audit = _collect_throttle_runs(history_lags)
    throttle_train = throttle_runs[THROTTLE_TRAIN_ID]
    throttle_validation = throttle_runs[THROTTLE_VALIDATION_ID]
    run_audits[THROTTLE_TRAIN_ID] = {
        "source": "existing 1,508-phase exact-two throttle surface",
        "split": "train",
        "samples": int(len(throttle_train["y"])),
    }
    run_audits[THROTTLE_VALIDATION_ID] = {
        "source": "existing 1,508-phase exact-two throttle surface continuation",
        "split": "held-out validation",
        "samples": int(len(throttle_validation["y"])),
    }

    yaw_train = _combine(yaw_train_runs)
    all_train_runs = {**yaw_train_runs, THROTTLE_TRAIN_ID: throttle_train}
    train = _combine(all_train_runs)
    all_validation_runs = {**validation_runs,
                           THROTTLE_VALIDATION_ID: throttle_validation}
    validation_runs = all_validation_runs
    validation = _combine(validation_runs)
    yaw_train_x = _model_features(yaw_train)
    train_x = _model_features(train)
    validation_x = _model_features(validation)
    historical_ids = set(prior.TRAIN_IDS)
    historical = _combine({run_id: data for run_id, data in yaw_train_runs.items()
                           if run_id in historical_ids})
    historical_x = _model_features(historical)
    throttle_augmented = _combine({
        **{run_id: data for run_id, data in yaw_train_runs.items()
           if run_id in historical_ids},
        THROTTLE_TRAIN_ID: throttle_train,
    })
    throttle_augmented_x = _model_features(throttle_augmented)

    historical_model = _fit(historical_x, historical["y"], historical["phase_id"])
    yaw_full_model = _fit(yaw_train_x, yaw_train["y"], yaw_train["phase_id"])
    yaw_train_gt_state = np.concatenate([
        gt_body_state_by_run[run_id] for run_id in yaw_train_runs], axis=0)
    gt_state_upper_bound_model = _fit_gt_state_upper_bound(
        yaw_train_x, yaw_train["y"], yaw_train["phase_id"],
        yaw_train_gt_state)
    throttle_augmented_model = _fit(
        throttle_augmented_x, throttle_augmented["y"],
        throttle_augmented["phase_id"])
    full_model = _fit(train_x, train["y"], train["phase_id"])
    historical_prediction = historical_model.predict(validation_x)
    yaw_full_prediction = yaw_full_model.predict(validation_x)
    throttle_augmented_prediction = throttle_augmented_model.predict(validation_x)
    global_prediction = full_model.predict(validation_x)

    event_prediction = global_prediction.copy()
    event_support = {}
    for event in EVENTS:
        train_idx = np.flatnonzero(train["causal_event"].astype(str) == event)
        val_idx = np.flatnonzero(
            validation["causal_event"].astype(str) == event)
        phase_count = int(len(np.unique(train["phase_id"][train_idx])))
        supported = len(train_idx) >= 200 and phase_count >= 8
        event_support[event] = {
            "rows": int(len(train_idx)), "transition_phases": phase_count,
            "supported": bool(supported),
        }
        if supported and len(val_idx):
            model = _fit(train_x[train_idx], train["y"][train_idx],
                         train["phase_id"][train_idx])
            event_prediction[val_idx] = model.predict(validation_x[val_idx])

    local_prediction, local_support = _predict_local(
        train, validation, train_x, validation_x, global_prediction)
    hybrid_min_one_run, hybrid_min_one_support = _local_event_hybrid(
        local_prediction, event_prediction, validation, local_support,
        minimum_runs=1)
    hybrid_min_two_runs, hybrid_min_two_support = _local_event_hybrid(
        local_prediction, event_prediction, validation, local_support,
        minimum_runs=MIN_LOCAL_INDEPENDENT_RUNS)
    predictions_by_model: dict[str, dict[str, np.ndarray]] = {
        "historical_global": {}, "full_spectrum_yaw_only_global": {},
        "historical_plus_throttle_global": {}, "all_data_global": {},
        "causal_event_specialists": {}, "measured_speed_steering_local": {},
        "local_event_hybrid_min1_run": {},
        "local_event_hybrid_min2_runs": {},
    }
    cursor = 0
    for run_id, data in validation_runs.items():
        length = len(data["y"])
        indices = np.arange(cursor, cursor + length)
        predictions_by_model["historical_global"][run_id] = historical_prediction[indices]
        predictions_by_model["full_spectrum_yaw_only_global"][run_id] = yaw_full_prediction[indices]
        predictions_by_model["historical_plus_throttle_global"][run_id] = throttle_augmented_prediction[indices]
        predictions_by_model["all_data_global"][run_id] = global_prediction[indices]
        predictions_by_model["causal_event_specialists"][run_id] = event_prediction[indices]
        predictions_by_model["measured_speed_steering_local"][run_id] = local_prediction[indices]
        predictions_by_model["local_event_hybrid_min1_run"][run_id] = hybrid_min_one_run[indices]
        predictions_by_model["local_event_hybrid_min2_runs"][run_id] = hybrid_min_two_runs[indices]
        cursor += length

    gt_state_information_diagnostic = {}
    for run_id, data in validation_runs.items():
        if run_id == THROTTLE_VALIDATION_ID:
            continue
        run_x = _model_features(data)
        gt_state = gt_body_state_by_run[run_id]
        sensor_prediction = yaw_full_model.predict(run_x)
        gt_augmented_prediction = gt_state_upper_bound_model.predict(
            np.column_stack((run_x, gt_state)))
        gt_state_information_diagnostic[run_id] = {
            "sensor_only_yaw_model": _metric(sensor_prediction - data["y"]),
            "plus_forbidden_gt_body_u_v_sideslip_upper_bound": _metric(
                gt_augmented_prediction - data["y"]),
            "gt_features_are_diagnostic_only": True,
        }

    metrics: dict[str, Any] = {}
    for model_name, per_run in predictions_by_model.items():
        run_reports = {}
        all_errors = []
        all_conditions = []
        for run_id, prediction in per_run.items():
            data = validation_runs[run_id]
            error = prediction - data["y"]
            all_errors.append(error)
            all_conditions.append(data["condition_id"])
            run_reports[run_id] = {
                "metrics": _metric(error),
                "by_phase_event": _metric_groups(
                    error, data["phase_event"].astype(str)),
                "by_causal_sensor_event": _metric_groups(
                    error, data["causal_event"].astype(str)),
                "by_measured_gt_speed_bin_and_signed_steering_bin": _metric_groups(
                    error,
                    np.floor(data["gt_speed"] / SPEED_BIN_MPS).astype(int),
                    np.rint(data["steering"] / STEERING_BIN_RAD).astype(int)),
                "by_wheel_body_mismatch_quartile": _metric_groups(
                    error, _quartiles(data["wheel_speed"] - data["gt_speed"])),
                "by_steering_command_feedback_gap_quartile": _metric_groups(
                    error, _quartiles(_derived(
                        data, "steering_command_gap_rad", history_lags))),
                "by_throttle_command_feedback_gap_quartile": _metric_groups(
                    error, _quartiles(_derived(
                        data, "throttle_command_gap_norm", history_lags))),
                "by_throttle_feedback_slew_quartile": _metric_groups(
                    error, _quartiles(_derived(
                        data, "throttle_feedback_rate_per_s", history_lags))),
                "by_rear_wheel_split_quartile": _metric_groups(
                    error, _quartiles(_derived(
                        data, "rear_wheel_split_mps", history_lags))),
                "by_abs_imu_roll_quartile": _metric_groups(
                    error, _quartiles(np.abs(data["x"][:, 9]))),
                "condition_bootstrap": _condition_bootstrap(
                    error, data["condition_id"]),
            }
        merged_error = np.concatenate(all_errors)
        merged_condition = np.concatenate(all_conditions)
        metrics[model_name] = {
            "aggregate_validation_metrics": _metric(merged_error),
            "independent_validation_runs": len(run_reports),
            "run_macro_rmse_radps": float(np.mean([
                row["metrics"]["rmse_radps"] for row in run_reports.values()])),
            "per_run": run_reports,
        }

    # The full-grid run is the primary validation for cell-wise performance.
    full_validation = validation_runs[FULL_VALIDATION_ID]
    full_error = (predictions_by_model["measured_speed_steering_local"][
        FULL_VALIDATION_ID] - full_validation["y"])
    best_prediction = predictions_by_model["measured_speed_steering_local"][
        FULL_VALIDATION_ID]
    output.mkdir(parents=True, exist_ok=True)
    _write_outliers(output / "heldout_exact_two_yaw_outliers.csv",
                    full_validation, best_prediction, history_lags)
    np.savez_compressed(
        output / "heldout_predictions.npz",
        run_id=FULL_VALIDATION_ID,
        target=full_validation["y"],
        prediction=best_prediction,
        historical_global=predictions_by_model["historical_global"][FULL_VALIDATION_ID],
        full_spectrum_yaw_only_global=predictions_by_model[
            "full_spectrum_yaw_only_global"][FULL_VALIDATION_ID],
        historical_plus_throttle_global=predictions_by_model[
            "historical_plus_throttle_global"][FULL_VALIDATION_ID],
        all_data_global=predictions_by_model["all_data_global"][FULL_VALIDATION_ID],
        causal_event_specialists=predictions_by_model["causal_event_specialists"][FULL_VALIDATION_ID],
        local_event_hybrid_min1_run=predictions_by_model[
            "local_event_hybrid_min1_run"][FULL_VALIDATION_ID],
        local_event_hybrid_min2_runs=predictions_by_model[
            "local_event_hybrid_min2_runs"][FULL_VALIDATION_ID],
        phase_event=full_validation["phase_event"],
        causal_event=full_validation["causal_event"],
        gt_speed_mps=full_validation["gt_speed"],
        wheel_speed_mps=full_validation["wheel_speed"],
        steering_rad=full_validation["steering"],
        condition_id=full_validation["condition_id"],
    )
    envelope_validation = validation_runs[ENVELOPE_VALIDATION_ID]
    envelope_prediction = predictions_by_model[
        "measured_speed_steering_local"][ENVELOPE_VALIDATION_ID]
    _write_outliers(
        output / "heldout_high_speed_envelope_yaw_outliers.csv",
        envelope_validation, envelope_prediction, history_lags)
    np.savez_compressed(
        output / "heldout_high_speed_envelope_predictions.npz",
        run_id=ENVELOPE_VALIDATION_ID,
        target=envelope_validation["y"],
        prediction=envelope_prediction,
        gt_speed_mps=envelope_validation["gt_speed"],
        wheel_speed_mps=envelope_validation["wheel_speed"],
        steering_rad=envelope_validation["steering"],
        phase_event=envelope_validation["phase_event"],
        causal_event=envelope_validation["causal_event"],
        condition_id=envelope_validation["condition_id"],
    )
    sparse_validation = validation_runs[SPARSE_SUPPORT_VALIDATION_ID]
    sparse_prediction = predictions_by_model[
        "measured_speed_steering_local"][SPARSE_SUPPORT_VALIDATION_ID]
    _write_outliers(
        output / "heldout_sparse_cell_support_yaw_outliers.csv",
        sparse_validation, sparse_prediction, history_lags)
    np.savez_compressed(
        output / "heldout_sparse_cell_support_predictions.npz",
        run_id=SPARSE_SUPPORT_VALIDATION_ID,
        target=sparse_validation["y"],
        prediction=sparse_prediction,
        gt_speed_mps=sparse_validation["gt_speed"],
        wheel_speed_mps=sparse_validation["wheel_speed"],
        steering_rad=sparse_validation["steering"],
        phase_event=sparse_validation["phase_event"],
        causal_event=sparse_validation["causal_event"],
        event_age_ms=sparse_validation["event_age_ms"],
        condition_id=sparse_validation["condition_id"],
    )
    sparse_crawl_validation = validation_runs[SPARSE_CRAWL_VALIDATION_ID]
    sparse_crawl_prediction = predictions_by_model[
        "measured_speed_steering_local"][SPARSE_CRAWL_VALIDATION_ID]
    _write_outliers(
        output / "heldout_sparse_crawl_support_yaw_outliers.csv",
        sparse_crawl_validation, sparse_crawl_prediction, history_lags)
    np.savez_compressed(
        output / "heldout_sparse_crawl_support_predictions.npz",
        run_id=SPARSE_CRAWL_VALIDATION_ID,
        target=sparse_crawl_validation["y"],
        prediction=sparse_crawl_prediction,
        gt_speed_mps=sparse_crawl_validation["gt_speed"],
        wheel_speed_mps=sparse_crawl_validation["wheel_speed"],
        steering_rad=sparse_crawl_validation["steering"],
        phase_event=sparse_crawl_validation["phase_event"],
        causal_event=sparse_crawl_validation["causal_event"],
        event_age_ms=sparse_crawl_validation["event_age_ms"],
        condition_id=sparse_crawl_validation["condition_id"],
    )
    report = {
        "title": "Exact-two full-spectrum one-step yaw model evaluation",
        "target": "GT yaw rate at k+1 minus current exact-packet IMU yaw rate (rad/s)",
        "status": "offline research; no production odometry or MPC change",
        "packet_admission": "exactly two contiguous command-to-feedback packets only; all other delays excluded",
        "future_truth_used_as_input": False,
        "truth_usage": ["next-step yaw-rate target", "offline GT speed stratification", "post-fit residual diagnosis"],
        "test_or_final_test_arrays_opened": False,
        "training_runs": sorted(all_train_runs),
        "validation_runs": sorted(validation_runs),
        "throttle_surface_audit": throttle_audit,
        "full_spectrum_training_run": FULL_TRAIN_ID,
        "full_spectrum_heldout_run": FULL_VALIDATION_ID,
        "high_speed_envelope_training_run": ENVELOPE_TRAIN_ID,
        "high_speed_envelope_heldout_run": ENVELOPE_VALIDATION_ID,
        "sparse_cell_support_training_run": SPARSE_SUPPORT_TRAIN_ID,
        "sparse_cell_support_heldout_run": SPARSE_SUPPORT_VALIDATION_ID,
        "sparse_crawl_support_training_run": SPARSE_CRAWL_TRAIN_ID,
        "sparse_crawl_support_heldout_run": SPARSE_CRAWL_VALIDATION_ID,
        "exact_two_support_replication_training_run": SUPPORT_REPLICATION_TRAIN_ID,
        "exact_two_support_replication_heldout_run": SUPPORT_REPLICATION_VALIDATION_ID,
        "full_spectrum_replication_training_run": FULL_TRAIN_REPLICATION_ID,
        "full_spectrum_replication_heldout_run": FULL_VALIDATION_REPLICATION_ID,
        "throttle_training_run": THROTTLE_TRAIN_ID,
        "throttle_heldout_run": THROTTLE_VALIDATION_ID,
        "feature_contract": {
            "sensor_history_lags_ms": [lag * 25 for lag in history_lags],
            "feature_count_before_causal_event_code": int(train["x"].shape[1]),
            "actuator_receipt_timing_features_used": False,
            "reason_timing_features_omitted": "the existing throttle archive has no equivalent source-stamp timing fields",
            "future_truth_used_as_input": False,
        },
        "forbidden_gt_state_information_diagnostic": {
            "purpose": "test whether unobserved lateral/longitudinal body velocity state explains remaining causal yaw prediction errors",
            "features": ["simulator_truth_body_u_mps", "simulator_truth_body_v_mps", "simulator_truth_sideslip_rad"],
            "never_used_as_runtime_input_or_selector": True,
            "per_heldout_run": gt_state_information_diagnostic,
        },
        "model_parameters": MODEL,
        "cell_selector": {
            "speed_proxy": "mean rear-wheel surface speed from current sensor packet",
            "steering": "measured steering feedback",
            "speed_bin_mps": SPEED_BIN_MPS,
            "signed_steering_bin_rad": STEERING_BIN_RAD,
            "minimum_local_rows": MIN_LOCAL_ROWS,
            "minimum_local_transition_phases": MIN_LOCAL_PHASES,
            "fallback": "global sensor-only model for unsupported cells",
        },
        "local_event_hybrid_candidates": {
            "min1_run": hybrid_min_one_support,
            "min2_runs": hybrid_min_two_support,
            "fallback": "causal-event specialist when a local cell is unsupported or lacks the required number of independent training captures",
            "minimum_local_independent_training_runs": MIN_LOCAL_INDEPENDENT_RUNS,
        },
        "source_admission_audit": source_audit,
        "run_audits": run_audits,
        "training_support": {
            "rows": int(len(train["y"])),
            "exact_two_transition_phases": int(len(np.unique(train["phase_id"]))),
            "rows_by_source_run": {
                run_id: int(len(data["y"]))
                for run_id, data in all_train_runs.items()},
            "causal_event_rows": dict(Counter(train["causal_event"].astype(str))),
            "local_expert_support_by_event": event_support,
        },
        "local_validation_support": local_support,
        "models": metrics,
        "heldout_full_grid_artifacts": {
            "predictions": str((output / "heldout_predictions.npz").relative_to(ROOT)),
            "worst_1000_rows": str((output / "heldout_exact_two_yaw_outliers.csv").relative_to(ROOT)),
        },
        "heldout_high_speed_envelope_artifacts": {
            "predictions": str((output / "heldout_high_speed_envelope_predictions.npz").relative_to(ROOT)),
            "worst_1000_rows": str((output / "heldout_high_speed_envelope_yaw_outliers.csv").relative_to(ROOT)),
        },
        "heldout_sparse_cell_support_artifacts": {
            "predictions": str((output / "heldout_sparse_cell_support_predictions.npz").relative_to(ROOT)),
            "worst_1000_rows": str((output / "heldout_sparse_cell_support_yaw_outliers.csv").relative_to(ROOT)),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True,
                   default=atlas._json_value) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--history-lags", default=",".join(map(str, atlas.HISTORY_LAGS)),
        help="causal history packet offsets, e.g. 0,1,2,4,8,12,20")
    args = parser.parse_args()
    history_lags = tuple(int(value) for value in args.history_lags.split(","))
    report = evaluate(args.output.resolve(), history_lags)
    print(json.dumps({
        "report": str((args.output.resolve() / "report.json").relative_to(ROOT)),
        "training_runs": len(report["training_runs"]),
        "validation_runs": len(report["validation_runs"]),
        "training_rows": report["training_support"]["rows"],
        "models": {
            name: value["aggregate_validation_metrics"]
            for name, value in report["models"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
