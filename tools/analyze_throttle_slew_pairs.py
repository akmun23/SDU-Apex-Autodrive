#!/usr/bin/env python3
"""Analyze randomized, matched ramp-vs-step throttle probes from Explore bags.

The wheel-speed residual is an encoder/kinematic proxy, not a tire-force or
direct tire-slip measurement. Run captures, not 40 Hz samples or repeated
within-run pairs, are the uncertainty clusters.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as common
from tools import evaluate_open_plane_body_dynamics as body


RATE_HZ = 40.0
PERIOD_S = 1.0 / RATE_HZ
ENCODER_SCALE = 0.9665627113525416
PRE_WINDOW_S = (-0.20, -0.05)
POST_WINDOWS_S = ((0.10, 0.35), (0.35, 0.65), (0.65, 1.00))
PAIR_GATES = {
    "speed_mps": 0.15,
    "steering_feedback_rad": 0.02,
    "vy_mps": 0.10,
    "yaw_rate_rps": 0.15,
    "throttle_feedback_norm": 0.03,
}
BOOTSTRAP_SEED = 20260928
REPO_ROOT = Path(__file__).resolve().parents[1]


def _fixed_encoder_surface_by_source_stamp(
        bag: Path) -> tuple[dict[int, int], dict[int, np.ndarray], dict[str, float]]:
    """Join encoder angles by packet identity and differentiate at fixed 40 Hz."""
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        required = (common.PACKET_TIMING, common.LEFT_ENCODER,
                    common.RIGHT_ENCODER)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError(f"bag lacks packet-aligned encoder inputs: {missing}")
        packet_by_stamp: dict[int, int] = {}
        for _, message in common._messages(
                connection, topics, common.PACKET_TIMING):
            try:
                row = json.loads(message.data)
                packet_by_stamp[int(row["bridge_receive_ros_stamp_ns"])] = int(
                    row["packet_sequence"])
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
        angles: list[dict[int, float]] = []
        match_fractions: dict[str, float] = {}
        for topic, side in ((common.LEFT_ENCODER, "left"),
                            (common.RIGHT_ENCODER, "right")):
            rows = list(common._messages(connection, topics, topic))
            by_packet: dict[int, float] = {}
            matched = 0
            for _, message in rows:
                if not message.position:
                    continue
                stamp = common._stamp_ns(message.header.stamp)
                sequence = packet_by_stamp.get(stamp)
                if sequence is None:
                    continue
                by_packet[sequence] = float(message.position[0])
                matched += 1
            match_fractions[side] = matched / max(1, len(rows))
            angles.append(by_packet)
        wheel_speed_by_stamp: dict[int, np.ndarray] = {}
        for stamp, sequence in packet_by_stamp.items():
            if sequence < 4:
                continue
            if any(any((sequence - step) not in side_angles
                       for step in range(5)) for side_angles in angles):
                continue
            wheel_speed_by_stamp[stamp] = np.asarray([
                common.WHEEL_RADIUS_M * (side_angles[sequence]
                                         - side_angles[sequence - 4])
                / (4 * PERIOD_S)
                for side_angles in angles], dtype=np.float64)
        return packet_by_stamp, wheel_speed_by_stamp, match_fractions
    finally:
        connection.close()


def _phase_events(path: Path) -> tuple[dict[int, dict[str, Any]],
                                        dict[int, dict[str, Any]],
                                        dict[int, dict[str, Any]],
                                        dict[str, Any]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        if common.PHASE not in topics:
            raise ValueError(f"bag has no {common.PHASE} topic: {path}")
        starts: dict[int, dict[str, Any]] = {}
        ends: dict[int, dict[str, Any]] = {}
        stimuli: dict[int, dict[str, Any]] = {}
        experiment_end: dict[str, Any] = {}
        for receipt_ns, message in common._messages(connection, topics, common.PHASE):
            try:
                event = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            kind = event.get("event")
            if kind == "phase_start":
                starts[int(event["phase_index"])] = event | {"receipt_ns": int(receipt_ns)}
            elif kind == "phase_end":
                ends[int(event["phase_index"])] = event | {"receipt_ns": int(receipt_ns)}
            elif kind == "throttle_slew_stimulus":
                stimuli[int(event["phase_index"])] = event | {
                    "receipt_ns": int(receipt_ns)}
            elif kind == "experiment_end":
                experiment_end = event
        return starts, ends, stimuli, experiment_end
    finally:
        connection.close()


def _target_throttle(start: dict[str, Any], phase_elapsed_s: float) -> float:
    initial = float(start["throttle_start_norm"])
    final = float(start["throttle_end_norm"])
    delay = float(start["throttle_stimulus_delay_s"])
    elapsed = phase_elapsed_s - delay
    if elapsed < 0.0:
        return initial
    if start["throttle_profile"] == "step":
        return initial if elapsed < PERIOD_S else final
    ramp_s = float(start["throttle_ramp_duration_s"])
    return initial + min(1.0, max(0.0, elapsed / ramp_s)) * (final - initial)


def _state_at_stimulus(event: dict[str, Any]) -> dict[str, float]:
    return {
        "speed_mps": float(event["speed_mps"]),
        "steering_feedback_rad": float(event["steering_feedback_rad"]),
        "vy_mps": float(event["vy_mps"]),
        "yaw_rate_rps": float(event["yaw_rate_rps"]),
        "throttle_feedback_norm": float(event["throttle_feedback_norm"]),
    }


def _sample_values(sample: body.MotionSample) -> dict[str, float] | None:
    if sample.rear_wheel_surface_mps is None:
        return None
    u, _v, yaw = map(float, sample.state)
    half_track = common.TRACK_WIDTH_M * 0.5
    left = ENCODER_SCALE * float(sample.rear_wheel_surface_mps[0]) - (
        u - yaw * half_track)
    right = ENCODER_SCALE * float(sample.rear_wheel_surface_mps[1]) - (
        u + yaw * half_track)
    result = {
        "mean_abs_rear_wheel_residual_mps": 0.5 * (abs(left) + abs(right)),
        "common_rear_wheel_residual_mps": 0.5 * (left + right),
        "rear_wheel_residual_asymmetry_mps": 0.5 * (right - left),
        "u_mps": u,
        "v_mps": float(sample.state[1]),
        "yaw_rate_rps": yaw,
        "throttle_feedback_norm": float(sample.actuators[1]),
        "throttle_command_norm": float(sample.actuators[2]),
        "steering_feedback_rad": float(sample.actuators[0]),
    }
    if sample.imu_acceleration_mps2 is not None:
        result["imu_lateral_acceleration_mps2"] = float(
            sample.imu_acceleration_mps2[1])
    return result if all(math.isfinite(value) for value in result.values()) else None


def _median_window(samples: list[tuple[float, dict[str, float]]],
                   bounds: tuple[float, float]) -> tuple[dict[str, float], int]:
    selected = [row for time_s, row in samples if bounds[0] <= time_s < bounds[1]]
    keys = sorted(set.intersection(*(set(row) for row in selected))) if selected else []
    return ({key: float(statistics.median(row[key] for row in selected))
             for key in keys}, len(selected))


def _phase_result(start: dict[str, Any], end: dict[str, Any],
                  stimulus: dict[str, Any],
                  sequences: dict[str, list[body.MotionSample]],
                  packet_by_source_stamp: dict[int, int]) -> dict[str, Any]:
    label = str(start["label"])
    profile_epoch = float(start.get("phase_elapsed_s", 0.0))
    phase_start_ns = int(start["receipt_ns"])
    stimulus_ns = int(stimulus["receipt_ns"])
    phase_end_ns = int(end["receipt_ns"])
    packet_samples = []
    for sample in sequences.get(label, []):
        packet_sequence = packet_by_source_stamp.get(sample.source_stamp_ns)
        if (packet_sequence is None
                or sample.receipt_ns < phase_start_ns
                or sample.receipt_ns > phase_end_ns):
            continue
        packet_samples.append((packet_sequence, sample))
    packet_samples.sort(key=lambda row: row[0])
    deduplicated: dict[int, body.MotionSample] = {}
    for packet_sequence, sample in packet_samples:
        deduplicated.setdefault(packet_sequence, sample)
    packet_samples = sorted(deduplicated.items())
    phase_start_candidates = [
        i for i, (_sequence, sample) in enumerate(packet_samples)
        if sample.receipt_ns >= phase_start_ns]
    stimulus_candidates = [
        i for i, (_sequence, sample) in enumerate(packet_samples)
        if sample.receipt_ns >= stimulus_ns]
    if not phase_start_candidates or not stimulus_candidates:
        raise ValueError(f"{label}: no packet sample brackets phase events")
    phase_start_index = phase_start_candidates[0]
    stimulus_index = stimulus_candidates[0]
    phase_samples = packet_samples[phase_start_index:]
    phase_start_sequence = phase_samples[0][0]
    stimulus_sequence = packet_samples[stimulus_index][0]
    packet_gaps = [right[0] - left[0]
                   for left, right in zip(phase_samples, phase_samples[1:])]
    fixed_packet_timebase_pass = all(step == 1 for step in packet_gaps)
    relative_rows = []
    for sequence, sample in packet_samples:
        if sequence < phase_start_sequence:
            continue
        values = _sample_values(sample)
        if values is not None:
            relative_rows.append(((sequence - stimulus_sequence) * PERIOD_S,
                                  values))

    # Simulator samples advance by packet ordinal at 25 ms. Receipt offsets
    # only select the bracketing packet and are retained as a transport check.
    command_errors: list[float] = []
    feedback_errors: list[float] = []
    command_values: list[float] = []
    sample_elapsed: list[float] = []
    threshold_s: float | None = None
    initial = float(start["throttle_start_norm"])
    final = float(start["throttle_end_norm"])
    midpoint = initial + 0.5 * (final - initial)
    for sequence, sample in phase_samples:
        sample_elapsed.append(
            profile_epoch + (sequence - phase_start_sequence) * PERIOD_S)
        command_values.append(float(sample.actuators[2]))
    offsets = np.arange(-0.060, 0.0601, PERIOD_S / 10.0)
    if command_values:
        profile_scores = [
            np.mean(np.square(np.asarray(command_values) - np.asarray([
                _target_throttle(start, value + offset)
                for value in sample_elapsed])))
            for offset in offsets]
        clock_offset_s = float(offsets[int(np.argmin(profile_scores))])
    else:
        clock_offset_s = 0.0
    for (sequence, sample), elapsed_base in zip(phase_samples, sample_elapsed):
        elapsed = elapsed_base + clock_offset_s
        desired = _target_throttle(start, elapsed)
        command_errors.append(float(sample.actuators[2]) - desired)
        feedback_errors.append(float(sample.actuators[1]) - desired)
        delta = final - initial
        reached = (sample.actuators[1] >= midpoint if delta > 0.0
                   else sample.actuators[1] <= midpoint)
        if (threshold_s is None and reached
                and sequence >= stimulus_sequence):
            threshold_s = (sequence - stimulus_sequence) * PERIOD_S
    command_mismatch_count = sum(abs(error) > 0.005 for error in command_errors)
    command_max_abs_error = (max(map(abs, command_errors))
                             if command_errors else None)
    command_rmse_norm = (float(np.sqrt(np.mean(np.square(command_errors))))
                         if command_errors else None)
    step_edge_rmse = (
        math.sqrt(2.0) * abs(final - initial)
        / math.sqrt(len(command_errors)) + 0.002
        if command_errors else None)
    pre, pre_count = _median_window(relative_rows, PRE_WINDOW_S)
    post_windows = {}
    for low, high in POST_WINDOWS_S:
        key = f"{low:.2f}_{high:.2f}s"
        post, count = _median_window(relative_rows, (low, high))
        changes = {name: post[name] - pre[name]
                   for name in pre.keys() & post.keys()}
        post_windows[key] = {"median": post, "change_from_pre": changes,
                             "sample_count": count}
    start_state = _state_at_stimulus(stimulus)
    return {
        "phase_index": int(start["phase_index"]),
        "label": label,
        "condition_pair_id": str(start["condition_pair_id"]),
        "profile": str(start["throttle_profile"]),
        "speed_target_mps": float(start["target_speed_mps"]),
        "steering_command_rad": float(start["steering_command_rad"]),
        "throttle_start_norm": initial,
        "throttle_end_norm": final,
        "phase_valid": end.get("valid"),
        "phase_quality_failures": end.get("quality_failures", []),
        "stimulus_state": start_state,
        "pre_window": {"median": pre, "sample_count": pre_count},
        "post_windows": post_windows,
        "command_profile_rmse_norm": (
            command_rmse_norm),
        "command_profile_clock_offset_s": clock_offset_s,
        "command_profile_edge_mismatch_samples": command_mismatch_count,
        "command_profile_max_abs_error_norm": command_max_abs_error,
        "feedback_profile_rmse_norm": (
            float(np.sqrt(np.mean(np.square(feedback_errors))))
            if feedback_errors else None),
        "feedback_final_error_norm": (
            float(np.median(feedback_errors[-6:])) if feedback_errors else None),
        "feedback_half_response_s_from_stimulus": threshold_s,
        "stimulus_to_sample_receipt_delay_ms": (
            packet_samples[stimulus_index][1].receipt_ns - stimulus_ns) / 1e6,
        "fixed_packet_timebase_pass": fixed_packet_timebase_pass,
        "packet_gap_count": sum(step != 1 for step in packet_gaps),
        "analyzed_samples": len(relative_rows),
        "command_profile_pass": (
            bool(command_errors)
            and command_rmse_norm is not None
            # A continuous ramp is tight; asynchronous command and sensor
            # streams can straddle a step edge by one sample on each side.
            and command_rmse_norm <= (
                0.005 if start["throttle_profile"] == "ramp"
                else float(step_edge_rmse))
            and command_max_abs_error is not None
            and command_max_abs_error <= abs(final - initial) + 0.003
            and abs(clock_offset_s) <= 0.060
            and fixed_packet_timebase_pass),
        "feedback_profile_pass": (
            bool(feedback_errors)
            and abs(float(np.median(feedback_errors[-6:]))) <= 0.03
            and threshold_s is not None
            and threshold_s <= float(start["throttle_ramp_duration_s"]) + 0.20),
    }


def _pair_match(ramp: dict[str, Any], step: dict[str, Any]) -> tuple[bool, dict[str, float], list[str]]:
    a, b = ramp["stimulus_state"], step["stimulus_state"]
    deltas = {name: abs(float(a[name]) - float(b[name]))
              for name in PAIR_GATES}
    failures = [name for name, delta in deltas.items()
                if delta > PAIR_GATES[name]]
    for profile in (ramp, step):
        if profile["phase_valid"] is not True:
            failures.append(f"{profile['profile']}_phase_quality")
        if not profile["command_profile_pass"]:
            failures.append(f"{profile['profile']}_command_profile")
        if not profile["feedback_profile_pass"]:
            failures.append(f"{profile['profile']}_feedback_profile")
        if not profile["fixed_packet_timebase_pass"]:
            failures.append(f"{profile['profile']}_packet_timebase_gap")
        if abs(profile["stimulus_state"]["steering_feedback_rad"]
               - profile["steering_command_rad"]) > PAIR_GATES["steering_feedback_rad"]:
            failures.append(f"{profile['profile']}_steering_not_settled")
        if abs(profile["stimulus_state"]["throttle_feedback_norm"]
               - profile["throttle_start_norm"]) > PAIR_GATES["throttle_feedback_norm"]:
            failures.append(f"{profile['profile']}_throttle_not_at_baseline")
    return not failures, deltas, failures


def _bootstrap_run_effects(rows: list[dict[str, Any]], metric: str,
                           bootstrap_count: int,
                           rng: np.random.Generator) -> dict[str, Any]:
    by_run: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and math.isfinite(float(value)):
            by_run[str(row["run_id"])].append(float(value))
    run_values = {run: float(np.mean(values)) for run, values in by_run.items()}
    values = np.asarray(list(run_values.values()), dtype=float)
    if not len(values):
        return {"runs": 0, "mean_step_minus_ramp_change": None,
                "run_bootstrap_95pct_ci": None}
    draws = rng.choice(values, size=(bootstrap_count, len(values)), replace=True).mean(axis=1)
    return {
        "runs": len(values),
        "mean_step_minus_ramp_change": float(values.mean()),
        "run_bootstrap_95pct_ci": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "run_effects": run_values,
    }


def analyze(bag_paths: list[Path], output: Path,
            bootstrap_count: int = 5000) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    run_reports: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for bag in bag_paths:
        capture = body.load_capture(bag)
        starts, ends, stimuli, experiment_end = _phase_events(bag)
        packet_by_source_stamp, wheel_speed_by_source_stamp, encoder_matches = (
            _fixed_encoder_surface_by_source_stamp(bag))
        run_id = bag.parents[1].name
        sequences: dict[str, list[body.MotionSample]] = defaultdict(list)
        for label, sequence in zip(capture.sequence_labels, capture.sequences):
            sequences[label].extend(
                replace(
                    sample,
                    rear_wheel_surface_mps=wheel_speed_by_source_stamp.get(
                        sample.source_stamp_ns))
                for sample in sequence)
        phases_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for index, start in starts.items():
            if start.get("profile") != "throttle_slew_pair":
                continue
            if index not in ends or index not in stimuli:
                continue
            if start.get("throttle_profile") not in ("ramp", "step"):
                continue
            if start.get("condition_pair_id") is None:
                continue
            result = _phase_result(start, ends[index], stimuli[index], sequences,
                                   packet_by_source_stamp)
            result["run_id"] = run_id
            phase_rows.append(result)
            phases_by_pair[result["condition_pair_id"]][result["profile"]] = result

        valid_pairs = 0
        for pair_id, pair in phases_by_pair.items():
            if set(pair) != {"ramp", "step"}:
                pair_rows.append({"run_id": run_id, "condition_pair_id": pair_id,
                                  "valid": False, "failure": "missing ramp or step"})
                continue
            ramp, step = pair["ramp"], pair["step"]
            matched, differences, failures = _pair_match(ramp, step)
            if matched:
                valid_pairs += 1
            pair_result: dict[str, Any] = {
                "run_id": run_id,
                "condition_pair_id": pair_id,
                "speed_target_mps": ramp["speed_target_mps"],
                "abs_steering_command_rad": abs(ramp["steering_command_rad"]),
                "turn_direction": ("left" if ramp["steering_command_rad"] > 0 else "right"),
                "throttle_delta_direction": ("up" if ramp["throttle_end_norm"] > ramp["throttle_start_norm"] else "down"),
                "valid": matched,
                "start_state_abs_differences": differences,
                "match_failures": failures,
            }
            for low, high in POST_WINDOWS_S:
                window = f"{low:.2f}_{high:.2f}s"
                ramp_change = ramp["post_windows"][window]["change_from_pre"]
                step_change = step["post_windows"][window]["change_from_pre"]
                for metric in ("mean_abs_rear_wheel_residual_mps",
                               "common_rear_wheel_residual_mps",
                               "rear_wheel_residual_asymmetry_mps",
                               "imu_lateral_acceleration_mps2", "u_mps", "v_mps",
                               "yaw_rate_rps"):
                    if metric in ramp_change and metric in step_change:
                        pair_result[f"step_minus_ramp_{metric}_change_{window}"] = (
                            step_change[metric] - ramp_change[metric])
            pair_rows.append(pair_result)

        relevant = [
            (topic, stats) for topic, stats in
            (capture.phase_stream_stats or capture.stream_stats).items()
        ]
        streams_40hz = all(rate >= 38.0 and p95 <= 35.0
                           for _topic, (rate, p95, _maximum) in relevant)
        expected_phase_count = 48
        run_reports.append({
            "run_id": run_id,
            "bag": str(bag.resolve().relative_to(REPO_ROOT)),
            "simulator_timebase": (
                "consecutive packet sequence numbers are 0.025 s apart; receipt "
                "timestamps only select the packet bracketing an event"),
            "encoder_source_stamp_match_fraction": encoder_matches,
            "aborted": capture.aborted,
            "experiment_reason": capture.reason,
            "experiment_end_event": experiment_end,
            "collision_count_start_end": [capture.collision_count_start,
                                          capture.collision_count_end],
            "bridge_timing_faults": capture.timing_faults,
            "phase_count": capture.phase_count,
            "valid_phase_count": capture.valid_phase_count,
            "invalid_phase_count": capture.invalid_phase_count,
            "throttle_slew_probe_phases": len(phases_by_pair) * 2,
            "matched_complete_pairs": valid_pairs,
            "expected_pairs": 24,
            "streams_meet_40hz_receive_gate": streams_40hz,
            "stream_stats_hz_p95gap_ms_maxgap_ms": {
                topic: {"rate_hz": rate, "p95_gap_ms": p95,
                        "max_gap_ms": maximum}
                for topic, (rate, p95, maximum) in relevant
            },
            "capture_quality_pass": (
                not capture.aborted
                and capture.collision_count_start == 0
                and capture.collision_count_end == 0
                and capture.timing_faults == 0
                and streams_40hz
                and valid_pairs == expected_phase_count // 2
            ),
        })

    primary_metric = "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.35_0.65s"
    overall = _bootstrap_run_effects(
        [row for row in pair_rows if row.get("valid")], primary_metric,
        bootstrap_count, rng)
    by_speed = {}
    for speed in (4.5, 6.5):
        by_speed[str(speed)] = _bootstrap_run_effects(
            [row for row in pair_rows if row.get("valid")
             and math.isclose(float(row["speed_target_mps"]), speed)],
            primary_metric, bootstrap_count, rng)
    response_metrics = (
        "mean_abs_rear_wheel_residual_mps",
        "common_rear_wheel_residual_mps",
        "rear_wheel_residual_asymmetry_mps",
        "imu_lateral_acceleration_mps2",
        "u_mps", "v_mps", "yaw_rate_rps",
    )
    response_windows = tuple(f"{low:.2f}_{high:.2f}s"
                             for low, high in POST_WINDOWS_S)
    secondary_outcomes = {}
    for window in response_windows:
        for metric in response_metrics:
            outcome_key = f"step_minus_ramp_{metric}_change_{window}"
            selected_all = [row for row in pair_rows if row.get("valid")]
            outcome = {
                "all_speeds_run_cluster_bootstrap": _bootstrap_run_effects(
                    selected_all, outcome_key, bootstrap_count, rng),
                "by_speed_run_cluster_bootstrap": {
                    str(speed): _bootstrap_run_effects(
                        [row for row in selected_all
                         if math.isclose(float(row["speed_target_mps"]), speed)],
                        outcome_key, bootstrap_count, rng)
                    for speed in (4.5, 6.5)
                },
            }
            secondary_outcomes[outcome_key] = outcome
    by_condition = {}
    by_condition_auxiliary = {}
    for speed in (4.5, 6.5):
        for steering in (0.30, 0.42):
            for turn_direction in ("left", "right"):
                for throttle_direction in ("up", "down"):
                    selected = [
                        row for row in pair_rows
                        if row.get("valid")
                        and math.isclose(float(row["speed_target_mps"]), speed)
                        and math.isclose(float(row["abs_steering_command_rad"]), steering)
                        and row["turn_direction"] == turn_direction
                        and row["throttle_delta_direction"] == throttle_direction
                    ]
                    key = (f"speed_{speed:.1f}__steer_{steering:.2f}__"
                           f"turn_{turn_direction}__throttle_{throttle_direction}")
                    by_condition[key] = _bootstrap_run_effects(
                        selected, primary_metric, bootstrap_count, rng)
                    by_condition_auxiliary[key] = {
                        metric: _bootstrap_run_effects(
                            selected, metric, bootstrap_count, rng)
                        for metric in (
                            "step_minus_ramp_common_rear_wheel_residual_mps_change_0.35_0.65s",
                            "step_minus_ramp_imu_lateral_acceleration_mps2_change_0.35_0.65s",
                        )
                    }
    run_ids = sorted({str(row["run_id"]) for row in run_reports})
    result = {
        "schema_version": 1,
        "analysis": "randomized matched gradual-ramp versus rapid-step throttle change",
        "proxy_warning": (
            "Rear encoder speed minus yaw/track kinematic speed, with a frozen scale from "
            "four straight calibration runs; this is not direct tire slip or force."),
        "runs": run_reports,
        "run_ids": run_ids,
        "primary_outcome": {
            "metric": primary_metric,
            "definition": "step-minus-ramp difference in each probe's post-minus-pre median; positive means larger encoder/kinematic residual after the step",
            "all_speeds_run_cluster_bootstrap": overall,
            "by_speed_run_cluster_bootstrap": by_speed,
            "by_steering_turn_and_throttle_direction_run_cluster_bootstrap": by_condition,
            "by_condition_common_residual_and_imu_run_cluster_bootstrap": by_condition_auxiliary,
            "secondary_outcomes_run_cluster_bootstrap": secondary_outcomes,
        },
        "phase_results": phase_rows,
        "paired_results": pair_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True,
                        help="rosbag SQLite database; repeat once per independent run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-count", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_count < 100:
        parser.error("bootstrap-count must be at least 100")
    try:
        analyze(args.bag, args.output, args.bootstrap_count)
    except (OSError, ValueError, sqlite3.Error, KeyError, RuntimeError) as exc:
        parser.exit(2, f"analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
