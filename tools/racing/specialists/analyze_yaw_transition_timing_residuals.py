#!/usr/bin/env python3
"""Historical packet-count comparison; do not execute under current policy.

Earlier two-versus-three-packet analyses are superseded because every
non-two-packet sample is now invalid. Shared phase/time helpers remain
importable; current diagnostics must use exact-two-packet phases only.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from tools import analyze_open_plane_dynamics as dynamics

try:
    import audit_sensor_yaw_large_errors as audit
    import fit_sensor_only_yaw_regime_atlas as atlas
    import train_yaw_multihorizon_teacher as teacher
    from score_yaw_atlas_transition_events import (
        _phase_lookup, _probe_phases)
except ModuleNotFoundError:
    from tools.racing.specialists import audit_sensor_yaw_large_errors as audit
    from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as atlas
    from tools.racing.specialists import train_yaw_multihorizon_teacher as teacher
    from tools.racing.specialists.score_yaw_atlas_transition_events import (
        _phase_lookup, _probe_phases)


ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "openplane_yaw_error_highsteer_reversal_validation_r03_20261008"
BASE = ROOT / "live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_command_intent_v2"
CANDIDATE = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported"
TIMING = ROOT / "live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_v2/steering_timing_r03.json"
TIMING_DIR = ROOT / "live_runs/racing_model_diagnostics_20261008/sensor_only_yaw_regime_atlas_v2"
OUTPUT = ROOT / "live_runs/racing_model_diagnostics_20261008/yaw_large_error_audit_v1/unwind_reversal_neighborhood_v2_supported/timing_residual_join.json"
TIMING_RUNS = {
    "openplane_yaw_error_highsteer_reversal_train_r01_20261008": "r01",
    "openplane_yaw_error_highsteer_reversal_train_r02_20261008": "r02",
    RUN_ID: "r03",
}
PRACTICE_RUN_ID = "practice_yaw_lowfade060_closedloop_r01_20261006"
STEERING_COMMAND_TOPIC = "/autodrive/roboracer_1/steering_command"
BRIDGE_TIMING_TOPIC = "/autodrive/roboracer_1/bridge_packet_timing"
MPC_DIAGNOSTICS_TOPIC = "/mpc/diagnostics"
STEERING_LIMIT_RAD = 0.5236
ERROR_LIMIT = 0.1
WINDOW_BEFORE_S = 0.100
WINDOW_AFTER_S = 0.500
PHASE_RE = re.compile(
    r"^probe_yawerr_(?:highsteer_reversal_|packet_phase_r[0-9]+_)"
    r"(onset|unwind|reversal)_"
    r"v([-+0-9.]+)_a([-+0-9.]+)_turn([+-][0-9]+)_"
    r"delay([-+0-9.]+)_(step|ramp)([-+0-9.]+)s(?:_rep[0-9]+)?$")


def _sample_indices(series: Any) -> np.ndarray:
    """Reproduce the atlas row gates and retain each row's source frame."""
    with np.load(ROOT / series.source, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str).tolist()
        splits = archive["run_splits"].astype(str).tolist()
        if series.run_id not in run_ids:
            raise ValueError(f"run provenance mismatch: {series.run_id}")
        run_index = run_ids.index(series.run_id)
        if splits[run_index] != series.split:
            raise ValueError(f"run split mismatch: {series.run_id}")
        sensors = np.asarray(archive["sensor_frames"], dtype=np.float32)
        sensor_valid = np.asarray(archive["sensor_valid"], dtype=bool)
        attitude = np.asarray(archive["imu_attitude_frames"], dtype=np.float32)
        attitude_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
        rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float32)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        sequence_runs = np.asarray(archive["sequence_run_index"], dtype=np.int64)
        sample_times = np.asarray(archive["sample_time_ns"], dtype=np.int64)
        indices: list[int] = []
        lags = atlas.HISTORY_LAGS
        for sequence_id in np.flatnonzero(sequence_runs == run_index):
            begin, end = map(int, bounds[int(sequence_id)])
            if end - begin <= max(lags) + 1:
                continue
            seq = sensors[begin:end]
            seq_attitude = attitude[begin:end]
            seq_rigid = rigid[begin:end]
            if (not np.isfinite(seq).all()
                    or not np.isfinite(seq_attitude).all()
                    or not np.isfinite(seq_rigid).all()):
                continue
            for k in range(max(lags), end - begin - 1):
                if (not all(sensor_valid[begin + k - lag] for lag in lags)
                        or not all(attitude_valid[begin + k - lag]
                                   for lag in lags)):
                    continue
                wheel = 0.5 * float(seq[k, 2] + seq[k, 3])
                steering = float(seq[k, 0])
                speed_cell = int(wheel // atlas.SPEED_BIN_MPS)
                if (speed_cell < 0 or speed_cell >= len(atlas.SPEED_CENTERS)
                        or not np.isfinite(steering)
                        or steering < atlas.STEERING_CENTERS[0] - 0.0125
                        or steering > atlas.STEERING_CENTERS[-1] + 0.0125):
                    continue
                indices.append(begin + k)
    return np.asarray(indices, dtype=np.int64), sample_times


def _phase_spec(label: str) -> dict[str, Any] | None:
    match = PHASE_RE.fullmatch(label)
    if match is None:
        return None
    event, speed, steering, turn, delay, mode, duration = match.groups()
    return {
        "probe_event": event,
        "requested_speed_mps": float(speed),
        "requested_abs_steering_rad": float(steering),
        "turn_sign": int(turn),
        "scheduled_delay_s": float(delay),
        "transition_mode": mode,
        "transition_duration_s": float(duration),
    }


def _predict_candidate(rows: dict[str, np.ndarray], base_bundle: dict[str, Any],
                      candidate_bundle: dict[str, Any], width: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    baseline = audit._predict(rows, base_bundle, width)[2]
    prediction = baseline.copy()
    speed = atlas._speed_cell(rows["wheel_speed"])
    steering = atlas._steer_cell(rows["steering"])
    groups: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for i, (event, speed_cell, steer_cell) in enumerate(zip(
            rows["event"].astype(str), speed, steering)):
        groups[(event, int(speed_cell), int(steer_cell))].append(i)
    for key, model in candidate_bundle["models"].items():
        indices = np.asarray(groups.get(key, ()), dtype=np.int64)
        if len(indices):
            prediction[indices] = model.predict(
                rows["x"][indices, :width]).astype(np.float32)
    return baseline, prediction


def _matched_phase_comparison(
        summaries: dict[tuple[str, int], list[dict[str, Any]]]
        ) -> dict[str, Any]:
    """Compare 2- vs 3-packet phases sharing the same commanded condition."""
    result: dict[str, Any] = {}
    for event in ("unwind", "reversal"):
        conditions: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
        for step_count in (2, 3):
            for row in summaries.get((event, step_count), ()):
                key = (
                    row["requested_speed_mps"],
                    row["requested_abs_steering_rad"],
                    row["turn_sign"], row["transition_mode"],
                    row["transition_duration_s"],
                )
                conditions[key][step_count] = row
        pairs = []
        for key, modes in sorted(conditions.items()):
            if 2 not in modes or 3 not in modes:
                continue
            two, three = modes[2], modes[3]
            pairs.append({
                "matched_condition": {
                    "requested_speed_mps": key[0],
                    "requested_abs_steering_rad": key[1],
                    "turn_sign": key[2], "transition_mode": key[3],
                    "transition_duration_s": key[4],
                },
                "two_packet_candidate_rmse_radps": two["candidate"]["rmse_radps"],
                "three_packet_candidate_rmse_radps": three["candidate"]["rmse_radps"],
                "three_minus_two_rmse_radps": (
                    three["candidate"]["rmse_radps"]
                    - two["candidate"]["rmse_radps"]),
                "two_packet_rows_over_0p1": two["candidate_errors_over_0p1"],
                "three_packet_rows_over_0p1": three["candidate_errors_over_0p1"],
                "two_packet_max_abs_radps": two["max_abs_candidate_error_radps"],
                "three_packet_max_abs_radps": three["max_abs_candidate_error_radps"],
            })
        deltas = np.asarray([row["three_minus_two_rmse_radps"]
                             for row in pairs], dtype=np.float64)
        if len(deltas):
            rng = np.random.default_rng(20261008)
            draws = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
            interval = [float(np.quantile(draws, 0.025)),
                        float(np.quantile(draws, 0.975))]
            mean_delta = float(np.mean(deltas))
            median_delta = float(np.median(deltas))
            worse_count = int(np.count_nonzero(deltas > 0.0))
        else:
            interval, mean_delta, median_delta, worse_count = None, None, None, 0
        result[event] = {
            "matched_condition_pairs": len(pairs),
            "mean_three_minus_two_packet_phase_rmse_radps": mean_delta,
            "median_three_minus_two_packet_phase_rmse_radps": median_delta,
            "pairs_with_larger_error_in_three_packet_case": worse_count,
            "paired_condition_bootstrap_95pct_ci": interval,
            "pairs": pairs,
        }
    return result


def _timestamp_phase_rows(run_id: str, suffix: str) -> list[dict[str, Any]]:
    """Build a legal-input timestamp proxy plus a debug-only timing label."""
    bag = ROOT / "live_runs" / run_id / "run/run_0.db3"
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        commands = [
            (stamp, float(message.data) * STEERING_LIMIT_RAD)
            for stamp, message in dynamics._messages(
                connection, topics, STEERING_COMMAND_TOPIC)
            if np.isfinite(float(message.data))
        ]
        odom_stamps = []
        for _receipt_stamp, message in dynamics._messages(
                connection, topics, dynamics.ODOM):
            stamp = message.header.stamp
            odom_stamps.append(int(stamp.sec) * 1_000_000_000
                               + int(stamp.nanosec))
    finally:
        connection.close()
    odom_stamps.sort()

    timing_path = TIMING_DIR / f"steering_timing_{suffix}.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    onset = {row["phase"]: int(row["command_to_feedback_onset_steps"])
             for row in timing["packet_grid_measurements"]}
    phases, _end = _probe_phases(bag)
    rows: list[dict[str, Any]] = []
    for phase in phases:
        spec = _phase_spec(phase.label)
        if spec is None:
            continue
        response_steps = onset.get(phase.label)
        if response_steps not in (2, 3):
            continue
        scheduled_ns = phase.start_ns + int(spec["scheduled_delay_s"] * 1e9)
        prior_commands = [value for stamp, value in commands
                          if scheduled_ns - 150_000_000 <= stamp
                          < scheduled_ns - 25_000_000]
        if len(prior_commands) < 2:
            continue
        command_before = float(np.median(prior_commands))
        changed = [(stamp, value) for stamp, value in commands
                   if stamp >= scheduled_ns - 10_000_000
                   and abs(value - command_before) >= 0.01]
        if not changed:
            continue
        command_stamp_ns, _command_value = changed[0]
        prior_odom = [stamp for stamp in odom_stamps
                      if stamp <= command_stamp_ns]
        if not prior_odom:
            continue
        previous_receive_ns = prior_odom[-1]
        phase_age_ms = (command_stamp_ns - previous_receive_ns) / 1e6
        rows.append({
            "run_id": run_id,
            "split": "validation" if run_id == RUN_ID else "train",
            "event": spec["probe_event"],
            "response_steps_debug_label": response_steps,
            "command_update_age_since_latest_permitted_odom_ms": phase_age_ms,
        })
    return rows


def _timestamp_proxy_analysis() -> dict[str, Any]:
    rows = [row for run_id, suffix in TIMING_RUNS.items()
            for row in _timestamp_phase_rows(run_id, suffix)]
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    training_age = {
        step: np.asarray([
            row["command_update_age_since_latest_permitted_odom_ms"]
            for row in training if row["response_steps_debug_label"] == step],
            dtype=np.float64)
        for step in (2, 3)
    }
    if any(len(values) == 0 for values in training_age.values()):
        raise ValueError("training captures do not contain both 2- and 3-packet modes")
    threshold = float(0.5 * (np.median(training_age[2])
                             + np.median(training_age[3])))

    def score(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
        labels = np.asarray([row["response_steps_debug_label"]
                             for row in split_rows], dtype=np.int8)
        proxy = np.asarray([
            row["command_update_age_since_latest_permitted_odom_ms"]
            for row in split_rows], dtype=np.float64)
        predicted = np.where(proxy >= threshold, 3, 2)
        always_two = np.full(len(labels), 2, dtype=np.int8)
        recalls = {
            str(step): float(np.mean(predicted[labels == step] == step))
            for step in (2, 3) if np.any(labels == step)
        }
        baseline_recalls = {
            str(step): float(np.mean(always_two[labels == step] == step))
            for step in (2, 3) if np.any(labels == step)
        }
        return {
            "samples": int(len(labels)),
            "class_counts": {str(step): int(np.count_nonzero(labels == step))
                             for step in (2, 3)},
            "timestamp_threshold_ms_fitted_on_train_only": threshold,
            "timestamp_proxy_accuracy": float(np.mean(predicted == labels)),
            "timestamp_proxy_balanced_accuracy": float(
                np.mean(list(recalls.values()))),
            "timestamp_proxy_recall_by_packet_class": recalls,
            "always_two_packet_accuracy": float(np.mean(always_two == labels)),
            "always_two_packet_balanced_accuracy": float(
                np.mean(list(baseline_recalls.values()))),
            "always_two_packet_recall_by_packet_class": baseline_recalls,
        }

    group_summary = {}
    for split, split_rows in (("train", training), ("validation", validation)):
        group_summary[split] = {}
        for step in (2, 3):
            selected = [row for row in split_rows
                        if row["response_steps_debug_label"] == step]
            legal_age = np.asarray([
                row["command_update_age_since_latest_permitted_odom_ms"]
                for row in selected], dtype=np.float64)
            group_summary[split][str(step)] = {
                "n": int(len(selected)),
                "legal_command_to_latest_packet_age_ms_p10_p50_p90": (
                    [float(x) for x in np.quantile(legal_age, [0.1, 0.5, 0.9])]
                    if len(legal_age) else []),
            }
    return {
        "title": "Can permitted command/packet timestamps identify 2/3-packet response?",
        "legal_predictor": "bag receipt timestamp of changed steering command minus latest permitted odometry source timestamp",
        "label": "measured command-to-feedback packet-step count, used only as held-out outcome",
        "training_runs": sorted({row["run_id"] for row in training}),
        "validation_run": RUN_ID,
        "group_distributions": group_summary,
        "train_threshold_transfer": {
            "train": score(training),
            "heldout_validation": score(validation),
        },
        "interpretation": (
            "The permitted command-to-odom timestamp proxy does not reliably "
            "identify the response phase on the unseen run. The MPC's own "
            "command publish timestamp is more precise than bag receipt time, "
            "but remains upstream of the actuator and bridge callbacks."),
    }


def _quantiles(values: list[float]) -> list[float]:
    if not values:
        return []
    return [float(value) for value in np.quantile(
        np.asarray(values, dtype=np.float64), [0.05, 0.25, 0.50, 0.75, 0.95])]


def _practice_mpc_timing_audit(run_id: str) -> dict[str, Any]:
    """Check if the real MPC's own event clock exposes bridge request phase.

    Bridge timing is used only as an offline diagnostic label. The command
    events are MPC-published command-history timestamps; the compared bridge
    age is measured when its actuator-command callback last updated. The
    distinction is intentional: these are different points in the pipeline.
    """
    bag = ROOT / "live_runs" / run_id / "run/run_0.db3"
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = dynamics._topic_map(connection)
        diagnostics = [
            (stamp, json.loads(message.data))
            for stamp, message in dynamics._messages(
                connection, topics, MPC_DIAGNOSTICS_TOPIC)
        ]
        bridge = [
            (stamp, json.loads(message.data))
            for stamp, message in dynamics._messages(
                connection, topics, BRIDGE_TIMING_TOPIC)
        ]
    finally:
        connection.close()

    accepted = [row for row in diagnostics
                if str(row[1].get("status", "")).startswith("accepted")]
    source_dt_ms = [1000.0 * float(row[1]["source_dt_s"])
                    for row in accepted if row[1].get("source_dt_s") is not None]
    control_source_ms = [
        (int(row[1]["control_ros_stamp_ns"])
         - int(row[1]["source_stamp_ns"])) / 1e6
        for row in accepted
        if row[1].get("control_ros_stamp_ns") and row[1].get("source_stamp_ns")
    ]
    event_age_ms = []
    event_ros_stamps: set[int] = set()
    for _receipt, item in diagnostics:
        control_ros_ns = int(item.get("control_ros_stamp_ns", 0))
        if control_ros_ns <= 0:
            continue
        for event_stamp_ns in (item.get("control_time_prediction") or {}).get(
                "command_event_stamps_ns", []):
            event_stamp_ns = int(event_stamp_ns)
            if event_stamp_ns <= control_ros_ns:
                event_age_ms.append((control_ros_ns - event_stamp_ns) / 1e6)
                event_ros_stamps.add(event_stamp_ns)

    request_intervals_ms = []
    request_times = [int(item.get("request_monotonic_ns") or 0)
                     for _receipt, item in bridge]
    request_times = [stamp for stamp in request_times if stamp > 0]
    if len(request_times) > 1:
        request_intervals_ms = [
            (right - left) / 1e6
            for left, right in zip(request_times, request_times[1:])
            if right > left
        ]
    bridge_ages_ms = [
        float(item["steering_command_update_age_ms"])
        for _receipt, item in bridge
        if item.get("steering_command_update_age_ms") is not None
    ]

    events = np.asarray(sorted(event_ros_stamps), dtype=np.int64)
    age_comparisons: list[tuple[float, float]] = []
    if len(events):
        for _receipt, item in bridge:
            request_ns = int(item.get("request_monotonic_ns") or 0)
            arrival_ns = int(item.get("bridge_arrival_monotonic_ns") or 0)
            receive_ros_ns = int(item.get("bridge_receive_ros_stamp_ns") or 0)
            observed_age = item.get("steering_command_update_age_ms")
            if (request_ns <= 0 or arrival_ns <= 0 or receive_ros_ns <= 0
                    or observed_age is None):
                continue
            request_ros_ns = receive_ros_ns - (arrival_ns - request_ns)
            event_index = int(np.searchsorted(
                events, request_ros_ns, side="right") - 1)
            if event_index < 0:
                continue
            event_age = (request_ros_ns - int(events[event_index])) / 1e6
            age_comparisons.append((event_age, float(observed_age)))
    if age_comparisons:
        event_age_at_request = np.asarray(
            [pair[0] for pair in age_comparisons], dtype=np.float64)
        observed_bridge_age = np.asarray(
            [pair[1] for pair in age_comparisons], dtype=np.float64)
        difference = event_age_at_request - observed_bridge_age
        correlation = (float(np.corrcoef(event_age_at_request,
                                         observed_bridge_age)[0, 1])
                       if np.std(event_age_at_request) > 0.0
                       and np.std(observed_bridge_age) > 0.0 else None)
        age_comparison = {
            "pairs": int(len(difference)),
            "mpc_event_age_at_bridge_request_ms_p05_p25_p50_p75_p95":
                _quantiles(event_age_at_request.tolist()),
            "bridge_callback_update_age_ms_p05_p25_p50_p75_p95":
                _quantiles(observed_bridge_age.tolist()),
            "difference_mpc_event_minus_bridge_callback_ms_p05_p25_p50_p75_p95":
                _quantiles(difference.tolist()),
            "difference_mae_ms": float(np.mean(np.abs(difference))),
            "difference_rmse_ms": float(np.sqrt(np.mean(difference ** 2))),
            "correlation": correlation,
        }
    else:
        age_comparison = {"pairs": 0}

    return {
        "run_id": run_id,
        "offline_diagnostic_only": True,
        "diagnostic_rows": len(diagnostics),
        "accepted_mpc_rows": len(accepted),
        "bridge_timing_rows": len(bridge),
        "accepted_control_source_interval_ms_p05_p25_p50_p75_p95":
            _quantiles(source_dt_ms),
        "accepted_control_minus_source_stamp_ms_p05_p25_p50_p75_p95":
            _quantiles(control_source_ms),
        "latest_command_history_event_age_at_control_ms_p05_p25_p50_p75_p95":
            _quantiles(event_age_ms),
        "bridge_request_interval_ms_p05_p25_p50_p75_p95":
            _quantiles(request_intervals_ms),
        "bridge_command_callback_update_age_ms_p05_p25_p50_p75_p95":
            _quantiles(bridge_ages_ms),
        "controller_event_to_bridge_callback_age_comparison": age_comparison,
        "applied_command_sequence_nonnull_rows": sum(
            item.get("applied_command_sequence") is not None
            for _receipt, item in bridge),
        "interpretation": (
            "The MPC's regular control/source timing does not encode the exact "
            "bridge request phase. Its command-history event stamp is upstream "
            "of the actuator's Float32 publication and the bridge callback; "
            "the practice capture does not provide a stable per-command "
            "mapping to bridge callback age or an applied-command ID."),
    }


def analyze() -> dict[str, Any]:
    report = json.loads((BASE / "sensor_only_yaw_regime_atlas_report.json")
                        .read_text(encoding="utf-8"))
    width = len(report["input_contract"]["features"])
    base_bundle = joblib.load(BASE / "sensor_only_yaw_regime_atlas.joblib")
    candidate_bundle = joblib.load(CANDIDATE / "candidate.joblib")
    admitted, _source_audit = teacher._discover_series_with_safe_mixed_archives()
    matches = [row for row in admitted
               if row.run_id == RUN_ID and row.split == "validation"]
    if len(matches) != 1:
        raise ValueError(f"expected one admitted held-out run; got {len(matches)}")
    series = matches[0]
    rows = atlas._rows(atlas._read_sensor_run(series), "command_intent")
    frame_indices, sample_times = _sample_indices(series)
    if len(frame_indices) != len(rows["residual"]):
        raise ValueError("source-frame reconstruction does not align with atlas rows")
    row_times = sample_times[frame_indices]
    baseline, candidate = _predict_candidate(
        rows, base_bundle, candidate_bundle, width)
    baseline_error = baseline - rows["residual"]
    candidate_error = candidate - rows["residual"]

    bag = ROOT / "live_runs" / RUN_ID / "run/run_0.db3"
    phases, _end = _probe_phases(bag)
    phases = sorted(phases, key=lambda phase: phase.start_ns)
    timing = json.loads(TIMING.read_text(encoding="utf-8"))
    packet_steps = {
        item["phase"]: int(item["command_to_feedback_onset_steps"])
        for item in timing["packet_grid_measurements"]
    }
    phase_index, labels = _phase_lookup(row_times, phases)
    next_index, _ = _phase_lookup(row_times + 25_000_000, phases)
    phase_index[next_index != phase_index] = -1

    all_probe_indices: list[int] = []
    for i, idx in enumerate(phase_index):
        if idx < 0:
            continue
        label = labels[int(idx)]
        spec = _phase_spec(label)
        if spec is None:
            continue
        all_probe_indices.append(i)
    indices = np.asarray(all_probe_indices, dtype=np.int64)
    phase_summaries: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    all_samples_by_step: dict[tuple[str, int], list[int]] = defaultdict(list)
    outliers_by_step: dict[tuple[str, int], list[int]] = defaultdict(list)
    feature_names = (report["input_contract"]["features"]
                     + list(atlas.DERIVED_NAMES[8:]))
    feature_index = {name: index for index, name in enumerate(feature_names)}
    for phase_index_value, phase in enumerate(phases):
        spec = _phase_spec(phase.label)
        if spec is None:
            continue
        mask = phase_index == phase_index_value
        phase_sample_indices = np.flatnonzero(mask)
        if not len(phase_sample_indices):
            continue
        step_count = packet_steps.get(phase.label)
        stimulus_ns = phase.start_ns + int(spec["scheduled_delay_s"] * 1e9)
        t = row_times[phase_sample_indices]
        window = ((t >= stimulus_ns - int(WINDOW_BEFORE_S * 1e9))
                  & (t <= stimulus_ns + int(WINDOW_AFTER_S * 1e9)))
        selected = phase_sample_indices[window]
        if step_count is None:
            raise ValueError(f"missing packet-grid onset for {phase.label}")
        key = (str(spec["probe_event"]), step_count)
        all_samples_by_step[key].extend(selected.tolist())
        outliers_by_step[key].extend(
            selected[np.abs(candidate_error[selected]) > ERROR_LIMIT].tolist())
        phase_summaries[key].append({
            **spec,
            "phase": phase.label,
            "command_to_feedback_onset_steps": step_count,
            "window_rows": int(len(selected)),
            "baseline": audit._metric(baseline_error[selected]),
            "candidate": audit._metric(candidate_error[selected]),
            "max_abs_candidate_error_radps": (
                float(np.max(np.abs(candidate_error[selected])))
                if len(selected) else None),
            "candidate_errors_over_0p1": int(np.count_nonzero(
                np.abs(candidate_error[selected]) > ERROR_LIMIT)),
        })

    by_event_and_packet: dict[str, Any] = {}
    for (event, step_count), selected_rows in sorted(all_samples_by_step.items()):
        selected = np.asarray(selected_rows, dtype=np.int64)
        outlier_rows = np.asarray(outliers_by_step[(event, step_count)],
                                  dtype=np.int64)
        summary_key = f"{event}/{step_count}_packet"
        by_event_and_packet[summary_key] = {
            "probe_phases": phase_summaries[(event, step_count)],
            "phase_count": len(phase_summaries[(event, step_count)]),
            "transition_window_rows": int(len(selected)),
            "baseline_window_error": audit._metric(baseline_error[selected]),
            "candidate_window_error": audit._metric(candidate_error[selected]),
            "candidate_large_error_rows": int(len(outlier_rows)),
            "candidate_large_error_fraction": float(
                len(outlier_rows) / len(selected)) if len(selected) else None,
            "diagnostic_inputs_at_large_error": {},
        }
        if len(outlier_rows):
            x = rows["x"][outlier_rows]
            by_event_and_packet[summary_key]["diagnostic_inputs_at_large_error"] = {
                "median_abs_command_feedback_gap_rad": float(np.median(np.abs(
                    x[:, feature_index["steering_command_gap_rad"]]))),
                "fraction_abs_command_feedback_gap_over_0p05_rad": float(np.mean(
                    np.abs(x[:, feature_index["steering_command_gap_rad"]]) > 0.05)),
                "median_command_age_ms": float(1000.0 * np.median(
                    x[:, feature_index["steering_command_age_s"]])),
                "median_feedback_age_ms": float(1000.0 * np.median(
                    x[:, feature_index["steering_feedback_age_s"]])),
                "median_abs_current_imu_gt_yaw_gap_radps_diagnostic_only": float(
                    np.median(np.abs(rows["imu_yaw"][outlier_rows]
                                     - rows["gt_yaw_current"][outlier_rows]))),
            }

    largest: list[dict[str, Any]] = []
    for i in np.flatnonzero(np.abs(candidate_error[indices]) > ERROR_LIMIT):
        row_index = int(indices[i])
        idx = int(phase_index[row_index])
        phase = phases[idx]
        spec = _phase_spec(phase.label)
        x = rows["x"][row_index]
        largest.append({
            "phase": phase.label,
            "probe_event": spec["probe_event"] if spec else None,
            "onset_steps": packet_steps.get(phase.label),
            "sample_time_ns": int(row_times[row_index]),
            "candidate_error_radps": float(candidate_error[row_index]),
            "baseline_error_radps": float(baseline_error[row_index]),
            "rear_wheel_mean_mps": float(rows["wheel_speed"][row_index]),
            "ground_truth_speed_mps_diagnostic_only": float(
                rows["gt_speed"][row_index]),
            "steering_feedback_rad": float(rows["steering"][row_index]),
            "steering_command_rad": float(x[feature_index[
                "steering_command_rad_lag0ms"]]),
            "steering_command_age_ms": float(1000.0 * x[feature_index[
                "steering_command_age_s"]]),
            "steering_feedback_age_ms": float(1000.0 * x[feature_index[
                "steering_feedback_age_s"]]),
            "command_feedback_gap_rad": float(x[feature_index[
                "steering_command_gap_rad"]]),
            "current_imu_yaw_rate_radps": float(rows["imu_yaw"][row_index]),
            "next_gt_yaw_rate_radps": float(
                rows["imu_yaw"][row_index] + rows["residual"][row_index]),
            "predicted_next_yaw_rate_radps": float(
                rows["imu_yaw"][row_index] + candidate[row_index]),
        })
    largest.sort(key=lambda row: abs(row["candidate_error_radps"]), reverse=True)

    return {
        "title": "Held-out yaw residuals joined to measured command-feedback packet phase",
        "run_id": RUN_ID,
        "ground_truth_is_one_step_target": True,
        "bridge_timing_is_offline_diagnostic_only": True,
        "runtime_inputs_are_unchanged": True,
        "transition_window_s_relative_to_scheduled_command": [
            -WINDOW_BEFORE_S, WINDOW_AFTER_S],
        "source_row_alignment": {
            "rows": int(len(rows["residual"])),
            "reconstructed_frame_indices": int(len(frame_indices)),
            "probe_phases": len(phases),
            "rows_inside_probe_phases": int(len(indices)),
            "excluded_test_and_final_test_arrays_read": False,
        },
        "all_run_metrics": {
            "baseline": audit._metric(baseline_error),
            "candidate": audit._metric(candidate_error),
        },
        "by_probe_event_and_packet_onset": by_event_and_packet,
        "matched_two_vs_three_packet_conditions": _matched_phase_comparison(
            phase_summaries),
        "permitted_timestamp_proxy_analysis": _timestamp_proxy_analysis(),
        "actual_mpc_practice_timing_check": _practice_mpc_timing_audit(
            PRACTICE_RUN_ID),
        "largest_candidate_errors_in_probe_phases": largest[:100],
        "interpretation_limits": [
            "Packet-onset step is measured once per controlled phase and is not an individual-row applied-command ID.",
            "Transition-window summaries are from one held-out capture; packet-step association is diagnostic, not causal proof by itself.",
            "Simulator GT is used only as target and error diagnosis; it is never a predictor input.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    print(json.dumps({
        "all_run_metrics": result["all_run_metrics"],
        "by_probe_event_and_packet_onset": {
            key: {k: value for k, value in row.items()
                  if k not in ("probe_phases", "diagnostic_inputs_at_large_error")}
            for key, row in result["by_probe_event_and_packet_onset"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated packet-count comparison: only exact-two-packet data are "
        "valid for current yaw analysis.")
