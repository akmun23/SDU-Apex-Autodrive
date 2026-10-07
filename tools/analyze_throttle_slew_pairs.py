#!/usr/bin/env python3
"""Analyze randomized, matched ramp-vs-step throttle probes from Explore bags.

The wheel-speed residual is an encoder/kinematic proxy, not a tire-force or
direct tire-slip measurement. Run captures, not 40 Hz samples or repeated
within-run pairs, are the uncertainty clusters.
"""

from __future__ import annotations

import argparse
import csv
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
STEERING_COMMAND_SCALE_RAD = 0.5236
PRE_WINDOW_S = (-0.20, -0.05)
POST_WINDOWS_S = ((0.10, 0.35), (0.35, 0.65), (0.65, 1.00))
INTEGRATED_RESPONSE_WINDOW_S = (0.10, 1.00)
RESPONSE_METRICS = (
    "mean_abs_rear_wheel_residual_mps",
    "mean_abs_rear_wheel_longitudinal_slip_ratio_proxy",
    "common_rear_wheel_residual_mps",
    "rear_wheel_residual_asymmetry_mps",
    "imu_lateral_acceleration_mps2",
    "abs_imu_lateral_acceleration_mps2",
    "steering_feedback_signed_lateral_acceleration_mps2",
    "steering_command_signed_lateral_acceleration_mps2",
    "rigid_body_longitudinal_acceleration_mps2",
    "rigid_body_lateral_acceleration_mps2",
    "abs_rigid_body_lateral_acceleration_mps2",
    "rigid_body_yaw_acceleration_rps2",
    "steering_feedback_signed_rigid_body_lateral_acceleration_mps2",
    "steering_command_signed_rigid_body_lateral_acceleration_mps2",
    "imu_roll_rad", "abs_imu_roll_rad", "imu_pitch_rad",
    "imu_roll_rate_rps", "abs_imu_roll_rate_rps", "imu_pitch_rate_rps",
    "u_mps", "v_mps", "yaw_rate_rps",
)
PAIR_GATES = {
    "speed_mps": 0.15,
    "steering_feedback_rad": 0.02,
    "vy_mps": 0.10,
    "yaw_rate_rps": 0.15,
    "throttle_feedback_norm": 0.03,
}
PRE_WINDOW_REAR_WHEEL_RESIDUAL_GATE_MPS = 0.20
BOOTSTRAP_SEED = 20260928
REPO_ROOT = Path(__file__).resolve().parents[1]
SWERVE_PROFILES = {
    "race_domain_swerve_throttle_slew_train": "train",
    "race_domain_swerve_throttle_slew_validation": "validation",
    "race_domain_swerve_throttle_slew_frontier_validation": "validation",
    "race_domain_swerve_throttle_slew_11mps_replication": "validation",
    "race_domain_swerve_throttle_slew_moderate_validation": "validation",
    "race_domain_swerve_throttle_slew_lowsteer_validation": "validation",
    "race_domain_swerve_throttle_rate_sweep_validation": "validation",
    "race_domain_swerve_throttle_rate_sweep_highsteer_validation": "validation",
    "race_domain_swerve_throttle_rate_factorial_validation": "validation",
    "race_domain_swerve_throttle_rate_factorial_4p5_validation": "validation",
    "race_domain_swerve_throttle_rate_factorial_6p5_validation": "validation",
    "race_domain_swerve_throttle_rate_factorial_7p5_validation": "validation",
    "race_domain_swerve_throttle_slew_up_frontier_validation": "validation",
    "race_domain_swerve_throttle_slew_up_frontier_train": "train",
    "race_domain_swerve_throttle_rate_race_domain_train": "train",
    "race_domain_swerve_throttle_rate_race_domain_validation": "validation",
}
FIXED_STEERING_PROFILE = "throttle_slew_pair"
FIXED_STEERING_EXPECTED_PAIRS = 24
SWERVE_EXPECTED_PAIRS = 32
SWERVE_EXPECTED_PAIRS_BY_PROFILE = {
    "race_domain_swerve_throttle_slew_train": SWERVE_EXPECTED_PAIRS,
    "race_domain_swerve_throttle_slew_validation": SWERVE_EXPECTED_PAIRS,
    "race_domain_swerve_throttle_slew_frontier_validation": 18,
    "race_domain_swerve_throttle_slew_11mps_replication": 6,
    "race_domain_swerve_throttle_slew_moderate_validation": 24,
    "race_domain_swerve_throttle_slew_lowsteer_validation": 24,
    "race_domain_swerve_throttle_rate_sweep_validation": 12,
    "race_domain_swerve_throttle_rate_sweep_highsteer_validation": 4,
    "race_domain_swerve_throttle_rate_factorial_validation": 24,
    "race_domain_swerve_throttle_rate_factorial_4p5_validation": 24,
    "race_domain_swerve_throttle_rate_factorial_6p5_validation": 24,
    "race_domain_swerve_throttle_rate_factorial_7p5_validation": 24,
    "race_domain_swerve_throttle_slew_up_frontier_validation": 16,
    "race_domain_swerve_throttle_slew_up_frontier_train": 16,
    "race_domain_swerve_throttle_rate_race_domain_train": 24,
    "race_domain_swerve_throttle_rate_race_domain_validation": 24,
}
SWERVE_EXPECTED_RESETS_BY_PROFILE = {
    **SWERVE_EXPECTED_PAIRS_BY_PROFILE,
    # Low-steer probes are individually reset-isolated so that a swerve's
    # residual lateral velocity/yaw cannot contaminate the paired treatment.
    "race_domain_swerve_throttle_slew_lowsteer_validation": 48,
    "race_domain_swerve_throttle_rate_sweep_validation": 24,
    "race_domain_swerve_throttle_rate_sweep_highsteer_validation": 8,
    "race_domain_swerve_throttle_rate_factorial_validation": 48,
    "race_domain_swerve_throttle_rate_factorial_4p5_validation": 48,
    "race_domain_swerve_throttle_rate_factorial_6p5_validation": 48,
    "race_domain_swerve_throttle_rate_factorial_7p5_validation": 48,
    "race_domain_swerve_throttle_slew_up_frontier_validation": 32,
    "race_domain_swerve_throttle_slew_up_frontier_train": 32,
    "race_domain_swerve_throttle_rate_race_domain_train": 48,
    "race_domain_swerve_throttle_rate_race_domain_validation": 48,
}
MAX_RESET_POSITION_ERROR_M = 0.25


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
                                        dict[str, Any],
                                        list[dict[str, Any]]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        if common.PHASE not in topics:
            raise ValueError(f"bag has no {common.PHASE} topic: {path}")
        starts: dict[int, dict[str, Any]] = {}
        ends: dict[int, dict[str, Any]] = {}
        stimuli: dict[int, dict[str, Any]] = {}
        experiment_end: dict[str, Any] = {}
        reset_recoveries: list[dict[str, Any]] = []
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
            elif kind == "sim_reset_recovered":
                reset_recoveries.append(event)
        return starts, ends, stimuli, experiment_end, reset_recoveries
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
        "steering_command_rad": float(event.get(
            "steering_command_rad", event.get("steering_feedback_rad", 0.0))),
        "vy_mps": float(event["vy_mps"]),
        "yaw_rate_rps": float(event["yaw_rate_rps"]),
        "throttle_feedback_norm": float(event["throttle_feedback_norm"]),
    }


def _average_rear_contact_speeds_by_sequence(
        samples: list[tuple[int, body.MotionSample]]) -> dict[int, tuple[float, float]]:
    """Average rear contact-point speed over the encoder's four-packet window."""
    states = {sequence: tuple(map(float, sample.state))
              for sequence, sample in samples}
    half_track = common.TRACK_WIDTH_M * 0.5
    result: dict[int, tuple[float, float]] = {}
    for sequence in states:
        window = [states.get(sequence - offset)
                  for offset in range(4, -1, -1)]
        if any(state is None for state in window):
            continue
        wheel_contact_speeds = [
            (u - yaw * half_track, u + yaw * half_track)
            for u, _v, yaw in window
        ]
        result[sequence] = tuple(
            sum(weight * row[side] for weight, row in zip(
                (0.5, 1.0, 1.0, 1.0, 0.5), wheel_contact_speeds)) / 4.0
            for side in range(2)
        )
    return result


def _sample_values(
        sample: body.MotionSample,
        average_contact_speeds_mps: tuple[float, float] | None
        ) -> dict[str, float] | None:
    if (sample.rear_wheel_surface_mps is None
            or average_contact_speeds_mps is None):
        return None
    u, _v, yaw = map(float, sample.state)
    left = (ENCODER_SCALE * float(sample.rear_wheel_surface_mps[0])
            - average_contact_speeds_mps[0])
    right = (ENCODER_SCALE * float(sample.rear_wheel_surface_mps[1])
             - average_contact_speeds_mps[1])
    left_ratio = left / max(abs(average_contact_speeds_mps[0]), 0.5)
    right_ratio = right / max(abs(average_contact_speeds_mps[1]), 0.5)
    result = {
        "mean_abs_rear_wheel_residual_mps": 0.5 * (abs(left) + abs(right)),
        "mean_abs_rear_wheel_longitudinal_slip_ratio_proxy": (
            0.5 * (abs(left_ratio) + abs(right_ratio))),
        "common_rear_wheel_residual_mps": 0.5 * (left + right),
        "rear_wheel_residual_asymmetry_mps": 0.5 * (right - left),
        "u_mps": u,
        "v_mps": float(sample.state[1]),
        "yaw_rate_rps": yaw,
        "throttle_feedback_norm": float(sample.actuators[1]),
        "throttle_command_norm": float(sample.actuators[2]),
        "steering_feedback_rad": float(sample.actuators[0]),
        "steering_command_rad": (
            STEERING_COMMAND_SCALE_RAD * float(sample.actuators[3])),
    }
    if sample.imu_roll_pitch_rad is not None:
        roll, pitch = map(float, sample.imu_roll_pitch_rad)
        result.update({
            "imu_roll_rad": roll,
            "abs_imu_roll_rad": abs(roll),
            "imu_pitch_rad": pitch,
        })
    if sample.imu_roll_pitch_rate_rps is not None:
        roll_rate, pitch_rate = map(float, sample.imu_roll_pitch_rate_rps)
        result.update({
            "imu_roll_rate_rps": roll_rate,
            "abs_imu_roll_rate_rps": abs(roll_rate),
            "imu_pitch_rate_rps": pitch_rate,
        })
    if sample.imu_acceleration_mps2 is not None:
        lateral_acceleration = float(sample.imu_acceleration_mps2[1])
        result["imu_lateral_acceleration_mps2"] = lateral_acceleration
        result["abs_imu_lateral_acceleration_mps2"] = abs(
            lateral_acceleration)
        feedback_steering = result["steering_feedback_rad"]
        commanded_steering = result["steering_command_rad"]
        feedback_sign = (1.0 if feedback_steering > 0.005 else
                         -1.0 if feedback_steering < -0.005 else 0.0)
        command_sign = (1.0 if commanded_steering > 0.005 else
                        -1.0 if commanded_steering < -0.005 else 0.0)
        result["steering_feedback_signed_lateral_acceleration_mps2"] = (
            feedback_sign * lateral_acceleration)
        result["steering_command_signed_lateral_acceleration_mps2"] = (
            command_sign * lateral_acceleration)
    return result if all(math.isfinite(value) for value in result.values()) else None


def _rigid_body_acceleration_by_sequence(
        phase_samples: list[tuple[int, body.MotionSample]]) -> dict[int, dict[str, float]]:
    """Estimate body-frame acceleration from consecutive 25 ms rigid states."""
    acceleration_by_sequence: dict[int, dict[str, float]] = {}
    dt_s = 2.0 * PERIOD_S
    for index in range(1, len(phase_samples) - 1):
        previous_sequence, previous = phase_samples[index - 1]
        sequence, current = phase_samples[index]
        next_sequence, following = phase_samples[index + 1]
        if (sequence - previous_sequence != 1
                or next_sequence - sequence != 1):
            continue
        u_previous, v_rear_previous, yaw_previous = map(float, previous.state)
        u_current, v_rear_current, yaw_current = map(float, current.state)
        u_following, v_rear_following, yaw_following = map(
            float, following.state)
        v_com_previous = v_rear_previous + common.COM_X_M * yaw_previous
        v_com_current = v_rear_current + common.COM_X_M * yaw_current
        v_com_following = v_rear_following + common.COM_X_M * yaw_following
        u_dot = (u_following - u_previous) / dt_s
        v_com_dot = (v_com_following - v_com_previous) / dt_s
        yaw_dot = (yaw_following - yaw_previous) / dt_s
        acceleration_by_sequence[sequence] = {
            "rigid_body_longitudinal_acceleration_mps2": (
                u_dot - yaw_current * v_com_current),
            "rigid_body_lateral_acceleration_mps2": (
                v_com_dot + yaw_current * u_current),
            "abs_rigid_body_lateral_acceleration_mps2": abs(
                v_com_dot + yaw_current * u_current),
            "rigid_body_yaw_acceleration_rps2": yaw_dot,
        }
    return acceleration_by_sequence


def _median_window(samples: list[tuple[float, dict[str, float]]],
                   bounds: tuple[float, float]) -> tuple[dict[str, float], int]:
    selected = [row for time_s, row in samples if bounds[0] <= time_s < bounds[1]]
    keys = sorted(set.intersection(*(set(row) for row in selected))) if selected else []
    return ({key: float(statistics.median(row[key] for row in selected))
             for key in keys}, len(selected))


def _mean_window(samples: list[tuple[float, dict[str, float]]],
                 bounds: tuple[float, float]) -> tuple[dict[str, float], int]:
    selected = [row for time_s, row in samples if bounds[0] <= time_s < bounds[1]]
    keys = sorted(set.intersection(*(set(row) for row in selected))) if selected else []
    return ({key: float(np.mean([row[key] for row in selected]))
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
        if packet_sequence is None or sample.receipt_ns > phase_end_ns:
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
    phase_context_samples = packet_samples[max(0, phase_start_index - 4):]
    phase_start_sequence = phase_samples[0][0]
    stimulus_sequence = packet_samples[stimulus_index][0]
    packet_gaps = [right[0] - left[0]
                   for left, right in zip(phase_samples, phase_samples[1:])]
    fixed_packet_timebase_pass = all(step == 1 for step in packet_gaps)
    relative_rows = []
    rigid_acceleration = _rigid_body_acceleration_by_sequence(phase_samples)
    average_contact_speeds = _average_rear_contact_speeds_by_sequence(
        phase_context_samples)
    for sequence, sample in packet_samples:
        if sequence < phase_start_sequence:
            continue
        values = _sample_values(
            sample, average_contact_speeds.get(sequence))
        if values is not None:
            values.update(rigid_acceleration.get(sequence, {}))
            feedback_sign = (1.0 if values["steering_feedback_rad"] > 0.005
                             else -1.0 if values["steering_feedback_rad"] < -0.005
                             else 0.0)
            command_sign = (1.0 if values["steering_command_rad"] > 0.005
                            else -1.0 if values["steering_command_rad"] < -0.005
                            else 0.0)
            if "rigid_body_lateral_acceleration_mps2" in values:
                lateral_acceleration = values[
                    "rigid_body_lateral_acceleration_mps2"]
                values["steering_feedback_signed_rigid_body_lateral_acceleration_mps2"] = (
                    feedback_sign * lateral_acceleration)
                values["steering_command_signed_rigid_body_lateral_acceleration_mps2"] = (
                    command_sign * lateral_acceleration)
            relative_rows.append(((sequence - stimulus_sequence) * PERIOD_S,
                                  values))

    # Simulator samples advance by packet ordinal at 25 ms. Receipt offsets
    # only select the bracketing packet and are retained as a transport check.
    command_errors: list[float] = []
    feedback_errors: list[float] = []
    command_values: list[float] = []
    phase_duration_s = float(start.get(
        "phase_duration_s",
        1.85 if start.get("profile") in SWERVE_PROFILES else 1.80))
    profile_rows = []
    threshold_s: float | None = None
    initial = float(start["throttle_start_norm"])
    final = float(start["throttle_end_norm"])
    midpoint = initial + 0.5 * (final - initial)
    for sequence, sample in phase_samples:
        elapsed = profile_epoch + (sequence - phase_start_sequence) * PERIOD_S
        if elapsed < phase_duration_s - PERIOD_S:
            profile_rows.append((sequence, sample, elapsed))
    # The last packet can be received after the phase-end event because the
    # reset/next phase command is published in the same controller tick. Keep
    # that packet in the bag, but do not score it against the just-finished
    # profile's command. Use the exact phase duration to make this deterministic.
    command_values = [float(sample.actuators[2])
                      for _sequence, sample, _elapsed in profile_rows]
    sample_elapsed = [elapsed for _sequence, _sample, elapsed in profile_rows]
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
    for (sequence, sample, elapsed_base) in profile_rows:
        elapsed = elapsed_base + clock_offset_s
        desired = _target_throttle(start, elapsed)
        command_errors.append(float(sample.actuators[2]) - desired)
        feedback_errors.append(float(sample.actuators[1]) - desired)
    for sequence, sample in phase_samples:
        elapsed = (profile_epoch
                   + (sequence - phase_start_sequence) * PERIOD_S
                   + clock_offset_s)
        delta = final - initial
        reached = (sample.actuators[1] >= midpoint if delta > 0.0
                   else sample.actuators[1] <= midpoint)
        if (threshold_s is None and reached
                and sequence >= stimulus_sequence):
            threshold_s = (sequence - stimulus_sequence) * PERIOD_S
    for sequence, sample in phase_samples:
        elapsed = (profile_epoch
                   + (sequence - phase_start_sequence) * PERIOD_S
                   + clock_offset_s)
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
    integrated_mean, integrated_count = _mean_window(
        relative_rows, INTEGRATED_RESPONSE_WINDOW_S)
    integrated_changes = {
        name: integrated_mean[name] - pre[name]
        for name in pre.keys() & integrated_mean.keys()
    }
    start_state = _state_at_stimulus(stimulus)
    return {
        "phase_index": int(start["phase_index"]),
        "label": label,
        "condition_pair_id": str(start["condition_pair_id"]),
        "profile": str(start["throttle_profile"]),
        "capture_profile": str(start.get("profile", "unknown")),
        "speed_target_mps": float(start["target_speed_mps"]),
        "steering_command_rad": float(start["steering_command_rad"]),
        "steering_profile": start.get("steering_profile"),
        "steering_amplitude_rad": start.get("steering_amplitude_rad"),
        "steering_waypoints": start.get("steering_waypoints"),
        "throttle_start_norm": initial,
        "throttle_end_norm": final,
        "throttle_ramp_duration_s": float(
            start.get("throttle_ramp_duration_s", 0.0)),
        "phase_valid": end.get("valid"),
        "phase_quality_failures": end.get("quality_failures", []),
        "speed_governor_ticks": int(end.get("speed_governor_ticks", 0)),
        "measured_speed_max_mps": end.get("measured_speed_max_mps"),
        "stimulus_state": start_state,
        "pre_window": {"median": pre, "sample_count": pre_count},
        "post_windows": post_windows,
        "integrated_response_0.10_1.00s": {
            "mean": integrated_mean,
            "change_from_pre": integrated_changes,
            "sample_count": integrated_count,
        },
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


def _pair_match(
        ramp: dict[str, Any], step: dict[str, Any]
        ) -> tuple[bool, dict[str, float], dict[str, float], list[str]]:
    a, b = ramp["stimulus_state"], step["stimulus_state"]
    deltas = {name: abs(float(a[name]) - float(b[name]))
              for name in PAIR_GATES}
    failures = [name for name, delta in deltas.items()
                if delta > PAIR_GATES[name]]
    ramp_pre = ramp["pre_window"]["median"]
    step_pre = step["pre_window"]["median"]
    pre_deltas: dict[str, float] = {}
    for wheel, sign in (("left", -1.0), ("right", 1.0)):
        ramp_residual = (
            float(ramp_pre["common_rear_wheel_residual_mps"])
            + sign * float(ramp_pre["rear_wheel_residual_asymmetry_mps"]))
        step_residual = (
            float(step_pre["common_rear_wheel_residual_mps"])
            + sign * float(step_pre["rear_wheel_residual_asymmetry_mps"]))
        key = f"{wheel}_rear_wheel_residual_mps"
        pre_deltas[key] = abs(ramp_residual - step_residual)
        if pre_deltas[key] > PRE_WINDOW_REAR_WHEEL_RESIDUAL_GATE_MPS:
            failures.append(f"pre_{key}_mismatch")
    for metric, label in (("imu_roll_rad", "imu_roll_abs_difference_rad"),
                          ("imu_roll_rate_rps", "imu_roll_rate_abs_difference_rps")):
        if metric in ramp_pre and metric in step_pre:
            pre_deltas[label] = abs(float(ramp_pre[metric])
                                     - float(step_pre[metric]))
    if ramp["steering_profile"] != step["steering_profile"]:
        failures.append("steering_profile_mismatch")
    if ramp["steering_waypoints"] != step["steering_waypoints"]:
        failures.append("steering_waypoint_mismatch")
    waypoints = ramp["steering_waypoints"] or ()
    steering_rate_max = max((
        abs(float(right[1]) - float(left[1]))
        / (float(right[0]) - float(left[0]))
        for left, right in zip(waypoints, waypoints[1:])
        if float(right[0]) > float(left[0])), default=0.0)
    # Steering and phase events are sampled on separate 40 Hz streams. Across
    # the packet bracketing each phase event, the two recorded command traces
    # can therefore differ by up to two packet intervals even when the same
    # waypoint profile was commanded. Bound the resulting angle difference
    # by that quantization plus 5 mrad; larger differences mean the paired
    # steering inputs were not alike.
    steering_command_tolerance = max(
        0.005, steering_rate_max * 2.0 * PERIOD_S + 0.005)
    for low, high in POST_WINDOWS_S:
        window = f"{low:.2f}_{high:.2f}s"
        ramp_command = ramp["post_windows"][window]["median"].get(
            "steering_command_rad")
        step_command = step["post_windows"][window]["median"].get(
            "steering_command_rad")
        if (ramp_command is None or step_command is None
                or abs(ramp_command - step_command)
                > steering_command_tolerance):
            failures.append(f"steering_command_waveform_mismatch_{window}")
    for profile in (ramp, step):
        if profile["phase_valid"] is not True:
            failures.append(f"{profile['profile']}_phase_quality")
        if not profile["command_profile_pass"]:
            failures.append(f"{profile['profile']}_command_profile")
        if not profile["feedback_profile_pass"]:
            failures.append(f"{profile['profile']}_feedback_profile")
        if not profile["fixed_packet_timebase_pass"]:
            failures.append(f"{profile['profile']}_packet_timebase_gap")
        if (profile["steering_profile"] is None
                and abs(profile["stimulus_state"]["steering_feedback_rad"]
                        - profile["steering_command_rad"])
                > PAIR_GATES["steering_feedback_rad"]):
            failures.append(f"{profile['profile']}_steering_not_settled")
        if abs(profile["stimulus_state"]["throttle_feedback_norm"]
               - profile["throttle_start_norm"]) > PAIR_GATES["throttle_feedback_norm"]:
            failures.append(f"{profile['profile']}_throttle_not_at_baseline")
    return not failures, deltas, pre_deltas, failures


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


def _response_surface_rows(
        pair_rows: list[dict[str, Any]],
        phase_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Reduce valid matched pairs to regime rows with run-level uncertainty."""
    key_fields = (
        "capture_profile", "speed_target_mps", "throttle_delta_direction",
        "throttle_delta_norm", "throttle_rise_rate_norm_per_sec",
        "abs_steering_command_rad", "turn_direction",
    )
    metrics = {
        "wheel_residual_step_minus_ramp_mps": (
            "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.10_1.00s"),
        "slip_ratio_proxy_step_minus_ramp": (
            "step_minus_ramp_mean_abs_rear_wheel_longitudinal_slip_ratio_proxy_change_0.10_1.00s"),
        "body_longitudinal_acceleration_step_minus_ramp_mps2": (
            "step_minus_ramp_rigid_body_longitudinal_acceleration_mps2_change_0.10_1.00s"),
    }
    phase_metrics = {
        "ramp_wheel_residual_change_mps": (
            "ramp", "mean_abs_rear_wheel_residual_mps"),
        "step_wheel_residual_change_mps": (
            "step", "mean_abs_rear_wheel_residual_mps"),
        "ramp_slip_ratio_proxy_change": (
            "ramp", "mean_abs_rear_wheel_longitudinal_slip_ratio_proxy"),
        "step_slip_ratio_proxy_change": (
            "step", "mean_abs_rear_wheel_longitudinal_slip_ratio_proxy"),
        "ramp_body_longitudinal_acceleration_change_mps2": (
            "ramp", "rigid_body_longitudinal_acceleration_mps2"),
        "step_body_longitudinal_acceleration_change_mps2": (
            "step", "rigid_body_longitudinal_acceleration_mps2"),
        "ramp_abs_roll_change_rad": ("ramp", "abs_imu_roll_rad"),
        "step_abs_roll_change_rad": ("step", "abs_imu_roll_rad"),
        "ramp_abs_roll_rate_change_rps": (
            "ramp", "abs_imu_roll_rate_rps"),
        "step_abs_roll_rate_change_rps": (
            "step", "abs_imu_roll_rate_rps"),
    }
    phases = {
        (str(phase["run_id"]), str(phase["condition_pair_id"]),
         str(phase["profile"])): phase
        for phase in (phase_rows or [])
    }
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pair_rows:
        if pair.get("valid") is True:
            groups[tuple(pair.get(field) for field in key_fields)].append(pair)

    rows: list[dict[str, Any]] = []
    for key, pairs in sorted(groups.items(), key=lambda item: tuple(
            "" if value is None else value for value in item[0])):
        row = dict(zip(key_fields, key))
        run_ids = sorted({str(pair["run_id"]) for pair in pairs})
        row["valid_pair_count"] = len(pairs)
        row["run_count"] = len(run_ids)
        row["run_ids"] = ";".join(run_ids)
        for label, metric in metrics.items():
            run_effects = []
            for run_id in run_ids:
                values = [float(pair[metric]) for pair in pairs
                          if pair["run_id"] == run_id
                          and pair.get(metric) is not None]
                if values:
                    run_effects.append(float(np.mean(values)))
            row[f"{label}_mean"] = (
                float(np.mean(run_effects)) if run_effects else None)
            row[f"{label}_run_sd"] = (
                float(statistics.stdev(run_effects))
                if len(run_effects) > 1 else None)
            row[f"{label}_run_min"] = (
                min(run_effects) if run_effects else None)
            row[f"{label}_run_max"] = (
                max(run_effects) if run_effects else None)
        for label, (profile, metric) in phase_metrics.items():
            run_effects = []
            for run_id in run_ids:
                values = []
                for pair in pairs:
                    if str(pair["run_id"]) != run_id:
                        continue
                    phase = phases.get((run_id, str(pair["condition_pair_id"]),
                                        profile))
                    if phase is None:
                        continue
                    change = phase["integrated_response_0.10_1.00s"][
                        "change_from_pre"].get(metric)
                    if change is not None:
                        values.append(float(change))
                if values:
                    run_effects.append(float(np.mean(values)))
            row[f"{label}_mean"] = (
                float(np.mean(run_effects)) if run_effects else None)
            row[f"{label}_run_sd"] = (
                float(statistics.stdev(run_effects))
                if len(run_effects) > 1 else None)
            row[f"{label}_run_min"] = (
                min(run_effects) if run_effects else None)
            row[f"{label}_run_max"] = (
                max(run_effects) if run_effects else None)
        rows.append(row)
    return rows


def _write_response_surface_csv(pair_rows: list[dict[str, Any]],
                                phase_rows: list[dict[str, Any]],
                                output: Path) -> None:
    rows = _response_surface_rows(pair_rows, phase_rows)
    if not rows:
        raise ValueError("no valid matched pairs available for response surface")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(bag_paths: list[Path], output: Path,
            bootstrap_count: int = 5000,
            surface_csv: Path | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    run_reports: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for bag in bag_paths:
        capture = body.load_capture(bag)
        starts, ends, stimuli, experiment_end, reset_recoveries = _phase_events(bag)
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
        capture_profile = next((start.get("profile") for start in starts.values()
                                if start.get("profile") in SWERVE_PROFILES
                                or start.get("profile") == FIXED_STEERING_PROFILE),
                               None)
        if capture_profile is None:
            raise ValueError(f"{bag}: no supported throttle-slew capture profile")
        split = SWERVE_PROFILES.get(capture_profile, "legacy_fixed_steering")
        phases_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for index, start in starts.items():
            if start.get("profile") != capture_profile:
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
            matched, differences, pre_differences, failures = _pair_match(ramp, step)
            if matched:
                valid_pairs += 1
            pair_result: dict[str, Any] = {
                "run_id": run_id,
                "condition_pair_id": pair_id,
                "speed_target_mps": ramp["speed_target_mps"],
                "abs_steering_command_rad": float(
                    ramp["steering_amplitude_rad"]
                    if ramp["steering_amplitude_rad"] is not None
                    else abs(ramp["steering_command_rad"])),
                "turn_direction": (
                    "left" if ramp["stimulus_state"]["steering_command_rad"] > 0
                    else "right"),
                "throttle_delta_direction": ("up" if ramp["throttle_end_norm"] > ramp["throttle_start_norm"] else "down"),
                "throttle_ramp_duration_s": ramp["throttle_ramp_duration_s"],
                "throttle_delta_norm": abs(
                    ramp["throttle_end_norm"] - ramp["throttle_start_norm"]),
                "throttle_rise_rate_norm_per_sec": (
                    (ramp["throttle_end_norm"] - ramp["throttle_start_norm"])
                    / ramp["throttle_ramp_duration_s"]
                    if ramp["throttle_ramp_duration_s"] > 0.0 else None),
                "capture_profile": capture_profile,
                "split": split,
                "valid": matched,
                "start_state_abs_differences": differences,
                "pre_window_rear_wheel_residual_abs_differences_mps": (
                    pre_differences),
                "match_failures": failures,
                "ramp_speed_governor_ticks": ramp["speed_governor_ticks"],
                "step_speed_governor_ticks": step["speed_governor_ticks"],
                "ramp_measured_speed_max_mps": ramp["measured_speed_max_mps"],
                "step_measured_speed_max_mps": step["measured_speed_max_mps"],
            }
            for low, high in POST_WINDOWS_S:
                window = f"{low:.2f}_{high:.2f}s"
                ramp_change = ramp["post_windows"][window]["change_from_pre"]
                step_change = step["post_windows"][window]["change_from_pre"]
                for metric in ("mean_abs_rear_wheel_residual_mps",
                               "mean_abs_rear_wheel_longitudinal_slip_ratio_proxy",
                               "common_rear_wheel_residual_mps",
                               "rear_wheel_residual_asymmetry_mps",
                               "imu_lateral_acceleration_mps2", "u_mps", "v_mps",
                               "yaw_rate_rps",
                               "abs_imu_lateral_acceleration_mps2",
                               "steering_feedback_signed_lateral_acceleration_mps2",
                               "steering_command_signed_lateral_acceleration_mps2",
                               "rigid_body_longitudinal_acceleration_mps2",
                               "rigid_body_lateral_acceleration_mps2",
                               "abs_rigid_body_lateral_acceleration_mps2",
                               "rigid_body_yaw_acceleration_rps2",
                               "steering_feedback_signed_rigid_body_lateral_acceleration_mps2",
                               "steering_command_signed_rigid_body_lateral_acceleration_mps2",
                               "imu_roll_rad", "abs_imu_roll_rad",
                               "imu_roll_rate_rps", "abs_imu_roll_rate_rps"):
                    if metric in ramp_change and metric in step_change:
                        pair_result[f"step_minus_ramp_{metric}_change_{window}"] = (
                            step_change[metric] - ramp_change[metric])
            integrated_key = "integrated_response_0.10_1.00s"
            ramp_change = ramp[integrated_key]["change_from_pre"]
            step_change = step[integrated_key]["change_from_pre"]
            for metric in RESPONSE_METRICS:
                if metric in ramp_change and metric in step_change:
                    pair_result[
                        f"step_minus_ramp_{metric}_change_0.10_1.00s"] = (
                            step_change[metric] - ramp_change[metric])
            pair_rows.append(pair_result)

        relevant = [
            (topic, stats) for topic, stats in
            (capture.phase_stream_stats or capture.stream_stats).items()
        ]
        streams_40hz = all(rate >= 38.0 and p95 <= 35.0
                           for _topic, (rate, p95, _maximum) in relevant)
        expected_pairs = (SWERVE_EXPECTED_PAIRS_BY_PROFILE[capture_profile]
                          if capture_profile in SWERVE_EXPECTED_PAIRS_BY_PROFILE
                          else FIXED_STEERING_EXPECTED_PAIRS)
        expected_reset_recoveries = (
            SWERVE_EXPECTED_RESETS_BY_PROFILE.get(capture_profile, 0))
        reset_position_errors = [float(event["position_error_m"])
                                 for event in reset_recoveries
                                 if event.get("position_error_m") is not None]
        reset_recovery_pass = (
            len(reset_recoveries) == expected_reset_recoveries
            and len(reset_position_errors) == expected_reset_recoveries
            and all(error <= MAX_RESET_POSITION_ERROR_M
                    for error in reset_position_errors)
        )
        invalid_pairs = [
            {"condition_pair_id": row.get("condition_pair_id"),
             "failures": row.get("match_failures", [row.get("failure")]),
             "ramp_speed_governor_ticks": row.get(
                 "ramp_speed_governor_ticks"),
             "step_speed_governor_ticks": row.get(
                 "step_speed_governor_ticks"),
             "ramp_measured_speed_max_mps": row.get(
                 "ramp_measured_speed_max_mps"),
             "step_measured_speed_max_mps": row.get(
                 "step_measured_speed_max_mps")}
            for row in pair_rows
            if row.get("run_id") == run_id and not row.get("valid")
        ]
        quality_failures = []
        if capture.aborted:
            quality_failures.append("experiment_aborted")
        if capture.collision_count_start != 0 or capture.collision_count_end != 0:
            quality_failures.append("collision_count_nonzero")
        if capture.timing_faults != 0:
            quality_failures.append("bridge_timing_faults_nonzero")
        if not streams_40hz:
            quality_failures.append("required_streams_below_40hz_gate")
        if len(phases_by_pair) != expected_pairs:
            quality_failures.append(
                f"observed_pairs_{len(phases_by_pair)}_of_{expected_pairs}")
        if valid_pairs != expected_pairs:
            quality_failures.append(
                f"valid_pairs_{valid_pairs}_of_{expected_pairs}")
        if not reset_recovery_pass:
            quality_failures.append("reset_recovery_gate_failed")
        run_reports.append({
            "run_id": run_id,
            "capture_profile": capture_profile,
            "split": split,
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
            "expected_pairs": expected_pairs,
            "invalid_pairs": invalid_pairs,
            "reset_recoveries": len(reset_recoveries),
            "expected_reset_recoveries": expected_reset_recoveries,
            "reset_recovery_pass": reset_recovery_pass,
            "max_reset_position_error_m": (
                max(reset_position_errors) if reset_position_errors else None),
            "streams_meet_40hz_receive_gate": streams_40hz,
            "capture_quality_failures": quality_failures,
            "stream_stats_hz_p95gap_ms_maxgap_ms": {
                topic: {"rate_hz": rate, "p95_gap_ms": p95,
                        "max_gap_ms": maximum}
                for topic, (rate, p95, maximum) in relevant
            },
            "capture_quality_pass": not quality_failures,
        })

    primary_metric = (
        "step_minus_ramp_mean_abs_rear_wheel_residual_mps_change_0.10_1.00s")
    by_capture_profile = {}
    for capture_profile in sorted({
            str(report["capture_profile"]) for report in run_reports}):
        selected = [row for row in pair_rows if row.get("valid")
                    and row["capture_profile"] == capture_profile]
        selected_runs = [report for report in run_reports
                         if report["capture_profile"] == capture_profile]
        by_capture_profile[capture_profile] = {
            "all_conditions_run_cluster_bootstrap": _bootstrap_run_effects(
                selected, primary_metric, bootstrap_count, rng),
            "by_split_run_cluster_bootstrap": {
                split: _bootstrap_run_effects(
                    [row for row in selected if row["split"] == split],
                    primary_metric, bootstrap_count, rng)
                for split in sorted({str(report["split"])
                                     for report in selected_runs})
            },
            "by_speed_run_cluster_bootstrap": {
                str(speed): _bootstrap_run_effects(
                    [row for row in selected
                     if math.isclose(float(row["speed_target_mps"]), speed)],
                    primary_metric, bootstrap_count, rng)
                for speed in sorted({float(row["speed_target_mps"])
                                     for row in selected})
            },
        }
    # Preserve the original report field for the original fixed-steering
    # experiment only. Swerve and fixed-steering effects are never pooled.
    legacy_rows = [row for row in pair_rows if row.get("valid")
                   and row["capture_profile"] == FIXED_STEERING_PROFILE]
    overall = _bootstrap_run_effects(legacy_rows, primary_metric,
                                     bootstrap_count, rng)
    by_speed = {
        str(speed): _bootstrap_run_effects(
            [row for row in legacy_rows
             if math.isclose(float(row["speed_target_mps"]), speed)],
            primary_metric, bootstrap_count, rng)
        for speed in sorted({float(row["speed_target_mps"])
                             for row in legacy_rows})
    }
    response_windows = tuple(
        f"{low:.2f}_{high:.2f}s" for low, high in POST_WINDOWS_S) + (
            "0.10_1.00s",)
    secondary_outcomes = {}
    for window in response_windows:
        for metric in RESPONSE_METRICS:
            outcome_key = f"step_minus_ramp_{metric}_change_{window}"
            outcome = {
                "by_capture_profile": {
                    capture_profile: {
                        "all_speeds_run_cluster_bootstrap": _bootstrap_run_effects(
                            [row for row in pair_rows if row.get("valid")
                             and row["capture_profile"] == capture_profile],
                            outcome_key, bootstrap_count, rng),
                        "by_speed_run_cluster_bootstrap": {
                            str(speed): _bootstrap_run_effects(
                                [row for row in pair_rows if row.get("valid")
                                 and row["capture_profile"] == capture_profile
                                 and math.isclose(
                                     float(row["speed_target_mps"]), speed)],
                                outcome_key, bootstrap_count, rng)
                            for speed in sorted({
                                float(row["speed_target_mps"])
                                for row in pair_rows if row.get("valid")
                                and row["capture_profile"] == capture_profile})
                        },
                    }
                    for capture_profile in sorted({
                        str(report["capture_profile"])
                        for report in run_reports})
                },
            }
            secondary_outcomes[outcome_key] = outcome
    rate_sweep_profiles = {
        "race_domain_swerve_throttle_rate_sweep_validation",
        "race_domain_swerve_throttle_rate_sweep_highsteer_validation",
        "race_domain_swerve_throttle_rate_factorial_validation",
        "race_domain_swerve_throttle_rate_factorial_4p5_validation",
        "race_domain_swerve_throttle_rate_factorial_6p5_validation",
        "race_domain_swerve_throttle_rate_factorial_7p5_validation",
    }
    rate_sweep_rows = [row for row in pair_rows if row.get("valid")
                       and row["capture_profile"] in rate_sweep_profiles]
    rate_sweep_metrics = (
        primary_metric,
        "step_minus_ramp_mean_abs_rear_wheel_longitudinal_slip_ratio_proxy_change_0.10_1.00s",
        "step_minus_ramp_rigid_body_longitudinal_acceleration_mps2_change_0.10_1.00s",
        "step_minus_ramp_abs_imu_roll_rad_change_0.10_1.00s",
        "step_minus_ramp_abs_imu_roll_rate_rps_change_0.10_1.00s",
    )
    rate_sweep_tradeoff: dict[str, Any] = {}
    for speed in sorted({float(row["speed_target_mps"])
                         for row in rate_sweep_rows}):
        speed_rows = [row for row in rate_sweep_rows
                      if math.isclose(float(row["speed_target_mps"]), speed)]
        delta_results = {}
        for delta in sorted({float(row["throttle_delta_norm"])
                             for row in speed_rows}):
            delta_rows = [row for row in speed_rows
                          if math.isclose(
                              float(row["throttle_delta_norm"]), delta)]
            rate_results = {}
            for rate in sorted({
                    float(row["throttle_rise_rate_norm_per_sec"])
                    for row in delta_rows}):
                rate_rows = [row for row in delta_rows
                             if math.isclose(
                                 float(row["throttle_rise_rate_norm_per_sec"]),
                                 rate, rel_tol=0.0, abs_tol=1e-6)]
                profile_results = {}
                for capture_profile in sorted({
                        str(row["capture_profile"]) for row in rate_rows}):
                    profile_rows = [row for row in rate_rows
                                    if row["capture_profile"] == capture_profile]
                    steering_turn_results = {}
                    for steering in sorted({
                            float(row["abs_steering_command_rad"])
                            for row in profile_rows}):
                        for turn in ("left", "right"):
                            selected = [row for row in profile_rows
                                        if math.isclose(
                                            float(row["abs_steering_command_rad"]),
                                            steering)
                                        and row["turn_direction"] == turn]
                            steering_turn_results[f"{steering:.2f}_{turn}"] = {
                                metric: _bootstrap_run_effects(
                                    selected, metric, bootstrap_count, rng)
                                for metric in rate_sweep_metrics
                            }
                    profile_results[capture_profile] = {
                        "valid_pair_count": len(profile_rows),
                        "outcomes": {
                            metric: _bootstrap_run_effects(
                                profile_rows, metric, bootstrap_count, rng)
                            for metric in rate_sweep_metrics
                        },
                        "outcomes_by_steering_and_turn": steering_turn_results,
                    }
                rate_results[f"{rate:.3f}"] = {
                    "rise_rate_norm_per_sec": rate,
                    "ramp_duration_s": delta / rate,
                    "by_capture_profile": profile_results,
                }
            delta_results[f"{delta:.3f}"] = {
                "throttle_delta_norm": delta,
                "by_rise_rate_norm_per_sec": rate_results,
            }
        rate_sweep_tradeoff[f"{speed:.1f}"] = {
            "speed_target_mps": speed,
            "by_throttle_delta_norm": delta_results,
        }
    by_condition = {}
    by_condition_auxiliary = {}
    for capture_profile in sorted({
            str(report["capture_profile"]) for report in run_reports}):
        profile_rows = [row for row in pair_rows if row.get("valid")
                        and row["capture_profile"] == capture_profile]
        for speed in sorted({float(row["speed_target_mps"])
                             for row in profile_rows}):
            steering_levels = sorted({
                float(row["abs_steering_command_rad"])
                for row in profile_rows
                if math.isclose(float(row["speed_target_mps"]), speed)})
            for steering in steering_levels:
                for turn_direction in ("left", "right"):
                    for throttle_direction in ("up", "down"):
                        selected = [
                            row for row in profile_rows
                            if math.isclose(float(row["speed_target_mps"]), speed)
                            and math.isclose(
                                float(row["abs_steering_command_rad"]), steering)
                            and row["turn_direction"] == turn_direction
                            and row["throttle_delta_direction"] == throttle_direction
                        ]
                        key = (f"{capture_profile}__speed_{speed:.1f}__"
                               f"steer_{steering:.2f}__turn_{turn_direction}__"
                               f"throttle_{throttle_direction}")
                        by_condition[key] = _bootstrap_run_effects(
                            selected, primary_metric, bootstrap_count, rng)
                        by_condition_auxiliary[key] = {
                            metric: _bootstrap_run_effects(
                                selected, metric, bootstrap_count, rng)
                            for metric in (
                                "step_minus_ramp_common_rear_wheel_residual_mps_change_0.35_0.65s",
                                "step_minus_ramp_imu_lateral_acceleration_mps2_change_0.35_0.65s",
                                "step_minus_ramp_rigid_body_lateral_acceleration_mps2_change_0.35_0.65s",
                                "step_minus_ramp_abs_rigid_body_lateral_acceleration_mps2_change_0.35_0.65s",
                            )
                        }
    run_ids = sorted({str(row["run_id"]) for row in run_reports})
    result = {
        "schema_version": 1,
        "analysis": "randomized matched gradual-ramp versus rapid-step throttle change",
        "rigid_body_acceleration_method": (
            "25 ms packet-ordinal central difference over 50 ms; convert the "
            "stored rear-axle lateral velocity to COM velocity using the "
            "repository COM offset, then apply ax=du/dt-r*v and ay=dv/dt+r*u"),
        "proxy_warning": (
            "Rear encoder wheel-surface speed minus yaw/track kinematic contact speed, "
            "with a frozen scale from four straight calibration runs and both speeds "
            "averaged over the same 100 ms encoder window; this is a longitudinal "
            "slip proxy, not tire-force or contact-patch truth."),
        "wheel_speed_time_alignment": (
            "Encoder angular speed uses angle difference across four consecutive "
            "25 ms intervals. Its body-contact-speed reference is trapezoidally "
            "averaged over those same four intervals, rather than sampled at the "
            "interval endpoint."),
        "pair_match_gates": {
            "stimulus_state_absolute_differences": PAIR_GATES,
            "pre_window_per_rear_wheel_residual_difference_mps": (
                PRE_WINDOW_REAR_WHEEL_RESIDUAL_GATE_MPS),
            "pre_window_definition_s": PRE_WINDOW_S,
        },
        "runs": run_reports,
        "run_ids": run_ids,
        "primary_outcome": {
            "metric": primary_metric,
            "definition": (
                "step-minus-ramp difference in the mean absolute rear "
                "wheel/body-speed residual over 0.10-1.00 s, each relative "
                "to its own pre-stimulus median; positive means the rapid "
                "step produced a larger transient residual (the ramp reduced "
                "the measured longitudinal wheel-slip proxy). Windowed medians "
                "remain secondary diagnostics."),
            "all_speeds_run_cluster_bootstrap": overall,
            "by_speed_run_cluster_bootstrap": by_speed,
            "by_capture_profile_run_cluster_bootstrap": by_capture_profile,
            "rate_sweep_tradeoff_run_cluster_bootstrap": {
                "profiles": sorted(rate_sweep_profiles),
                "by_speed_mps": rate_sweep_tradeoff,
                "note": (
                    "Speed, throttle increment, actual rise rate, and capture "
                    "profile are kept separate. Each result uses only its own "
                    "reset-matched step pairs; no cross-speed or cross-profile "
                    "pooling is performed."),
            },
            "cross_speed_pooled_effect_computed": False,
            "cross_profile_pooled_effect_computed": False,
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
    if surface_csv is not None:
        _write_response_surface_csv(pair_rows, phase_rows, surface_csv)
        print(f"wrote {surface_csv}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True,
                        help="rosbag SQLite database; repeat once per independent run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--surface-csv", type=Path,
                        help="optional compact response surface with run-level summaries")
    parser.add_argument("--bootstrap-count", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap_count < 100:
        parser.error("bootstrap-count must be at least 100")
    try:
        analyze(args.bag, args.output, args.bootstrap_count, args.surface_csv)
    except (OSError, ValueError, sqlite3.Error, KeyError, RuntimeError) as exc:
        parser.exit(2, f"analysis failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
