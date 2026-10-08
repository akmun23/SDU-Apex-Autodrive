#!/usr/bin/env python3
"""Audit measured support in the reset-isolated yaw-error probe captures.

This is a data-coverage audit, not a yaw-model score. It reads only valid
phase sequences from a closed Explore rosbag and reports the measured state
on the target-steering plateau, so requested commands are never mistaken for
achieved speed or steering.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_open_plane_body_dynamics import (
    Capture, MotionSample, STEERING_LIMIT_RAD, load_capture)


STEERING_EVENT_LABEL = re.compile(
    r"^probe_yawerr_(crawl(?:_fine)?|highsteer_reversal|subcrawl)_"
    r"(onset|unwind|reversal)_v([0-9.]+)_a([0-9.]+)_"
    r"turn([+-]1)_delay([0-9.]+)_(step|ramp)([0-9.]+)s$"
)
LOWWHEEL_LABEL = re.compile(
    r"^probe_yawerr_lowwheel_v([0-9.]+)_a([0-9.]+)_"
    r"turn([+-]1)_up([0-9.]+)_(step|ramp)$"
)
MIDWHEEL_LABEL = re.compile(
    r"^probe_yawerr_midwheel_v([0-9.]+)_a([0-9.]+)_"
    r"turn([+-]1)_(up|down)_d([0-9.]+)_(step|ramp)$"
)
PROFILE_PREFIXES = (
    "probe_yawerr_crawl_", "probe_yawerr_highsteer_reversal_",
    "probe_yawerr_subcrawl_", "probe_yawerr_lowwheel_",
    "probe_yawerr_midwheel_")
SETTLE_PREFIXES = (
    "settle_yawerr_crawl_", "settle_yawerr_highsteer_reversal_",
    "settle_yawerr_subcrawl_")
EXPECTED_PROBES_BY_FAMILY = {
    "crawl": 96,
    "crawl_fine": 384,
    "highsteer_reversal": 216,
    "subcrawl": 144,
    "lowwheel": 144,
    "midwheel": 112,
}
STEERING_TOLERANCE_RAD = 0.05


def parse_label(label: str) -> dict[str, Any] | None:
    match = STEERING_EVENT_LABEL.fullmatch(label)
    if match is not None:
        family, event, speed, angle, sign, delay, mode, duration = match.groups()
        transition_delay = float(delay)
        transition_duration = float(duration)
        throttle_delta = None
    else:
        match = LOWWHEEL_LABEL.fullmatch(label)
        if match is not None:
            speed, angle, sign, delta, mode = match.groups()
            family, event = "lowwheel", "throttle_up"
            transition_delay = 0.80
            transition_duration = 0.025 if mode == "step" else 0.30
            throttle_delta = float(delta)
        else:
            match = MIDWHEEL_LABEL.fullmatch(label)
            if match is None:
                return None
            speed, angle, sign, direction, delta, mode = match.groups()
            family, event = "midwheel", f"throttle_{direction}"
            transition_delay = 0.80
            transition_duration = 0.025 if mode == "step" else 0.30
            throttle_delta = float(delta) * (1.0 if direction == "up" else -1.0)
    turn_sign = 1 if sign == "+1" else -1
    requested_angle = float(angle)
    target_steering = (
        turn_sign * requested_angle if event in (
            "onset", "throttle_up", "throttle_down") else
        0.0 if event == "unwind" else
        -turn_sign * requested_angle)
    return {
        "profile_family": family,
        "event": event,
        "requested_speed_mps": float(speed),
        "requested_abs_steering_rad": requested_angle,
        "turn_sign": turn_sign,
        "target_steering_rad": target_steering,
        "delay_s": transition_delay,
        "transition_mode": mode,
        "transition_duration_s": transition_duration,
        "throttle_delta_norm": throttle_delta,
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.quantile(array, (0.10, 0.50, 0.90))
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _sequence_metrics(samples: tuple[MotionSample, ...], spec: dict[str, Any]
                      ) -> dict[str, Any]:
    transition_end = spec["delay_s"] + spec["transition_duration_s"]
    plateau_start = transition_end + 0.15
    plateau_end = spec["delay_s"] + 1.10
    plateau: list[MotionSample] = []
    speeds: list[float] = []
    odometry_speeds: list[float] = []
    steering: list[float] = []
    steering_commands: list[float] = []
    throttle_feedback: list[float] = []
    throttle_commands: list[float] = []
    throttle_command_feedback_gap: list[float] = []
    odom_truth_speed_gap: list[float] = []
    wheel_body_mismatch: list[float] = []
    yaw_rate: list[float] = []
    pre_stimulus_speeds: list[float] = []
    pre_stimulus_speed_error: list[float] = []
    pre_stimulus_steering: list[float] = []
    pre_stimulus_steering_error: list[float] = []
    pre_stimulus_throttle_feedback: list[float] = []
    pre_stimulus_throttle_commands: list[float] = []
    pre_stimulus_wheel_body_mismatch: list[float] = []
    pre_start = max(0.0, spec["delay_s"] - 0.30)
    pre_end = max(pre_start, spec["delay_s"] - 0.05)
    pre_steering_target = (
        0.0 if spec["event"] == "onset" else
        spec["turn_sign"] * spec["requested_abs_steering_rad"])
    for sample in samples:
        rigid = sample.simulator_rigid_state
        if rigid is None or len(rigid) != 13 or not np.isfinite(rigid).all():
            continue
        truth_speed = math.hypot(float(rigid[7]), float(rigid[8]))
        wheels = sample.rear_wheel_surface_mps
        if pre_start <= sample.time_s <= pre_end:
            pre_stimulus_speeds.append(truth_speed)
            pre_stimulus_speed_error.append(
                abs(truth_speed - spec["requested_speed_mps"]))
            pre_stimulus_steering.append(float(sample.actuators[0]))
            pre_stimulus_steering_error.append(abs(
                float(sample.actuators[0]) - pre_steering_target))
            pre_stimulus_throttle_feedback.append(
                float(sample.actuators[1]))
            pre_stimulus_throttle_commands.append(
                float(sample.actuators[2]))
            if wheels is not None and len(wheels) == 2 and np.isfinite(wheels).all():
                pre_stimulus_wheel_body_mismatch.append(abs(
                    float(np.mean(np.abs(wheels))) - truth_speed))
        if sample.time_s >= plateau_start and sample.time_s <= plateau_end:
            plateau.append(sample)
            odom_speed = math.hypot(float(sample.state[0]),
                                    float(sample.state[1]))
            speeds.append(truth_speed)
            odometry_speeds.append(odom_speed)
            odom_truth_speed_gap.append(abs(odom_speed - truth_speed))
            steering.append(float(sample.actuators[0]))
            throttle_feedback.append(float(sample.actuators[1]))
            throttle_commands.append(float(sample.actuators[2]))
            throttle_command_feedback_gap.append(abs(
                float(sample.actuators[2]) - float(sample.actuators[1])))
            steering_commands.append(
                float(sample.actuators[3]) * STEERING_LIMIT_RAD)
            yaw_rate.append(float(rigid[12]))
            if wheels is not None and len(wheels) == 2 and np.isfinite(wheels).all():
                mean_wheel_speed = float(np.mean(np.abs(wheels)))
                body_speed = math.hypot(float(rigid[7]), float(rigid[8]))
                wheel_body_mismatch.append(abs(mean_wheel_speed - body_speed))

    target = float(spec["target_steering_rad"])
    target_fraction = (
        float(np.mean(np.abs(np.asarray(steering) - target)
                      <= STEERING_TOLERANCE_RAD)) if steering else None)
    speed_fraction = (
        float(np.mean(np.abs(np.asarray(speeds) - spec["requested_speed_mps"])
                      <= 0.15)) if speeds else None)
    command_feedback_gap = (
        np.abs(np.asarray(steering_commands) - np.asarray(steering))
        if steering else np.asarray([], dtype=np.float64))
    expected_throttle_delta = spec["throttle_delta_norm"]
    pre_command_median = (
        float(np.median(pre_stimulus_throttle_commands))
        if pre_stimulus_throttle_commands else None)
    pre_feedback_median = (
        float(np.median(pre_stimulus_throttle_feedback))
        if pre_stimulus_throttle_feedback else None)
    plateau_command_median = (
        float(np.median(throttle_commands)) if throttle_commands else None)
    plateau_feedback_median = (
        float(np.median(throttle_feedback)) if throttle_feedback else None)
    command_delta = (
        plateau_command_median - pre_command_median
        if plateau_command_median is not None and pre_command_median is not None
        else None)
    feedback_delta = (
        plateau_feedback_median - pre_feedback_median
        if plateau_feedback_median is not None and pre_feedback_median is not None
        else None)
    command_residual = (
        command_delta - expected_throttle_delta
        if command_delta is not None and expected_throttle_delta is not None
        else None)
    feedback_residual = (
        feedback_delta - expected_throttle_delta
        if feedback_delta is not None and expected_throttle_delta is not None
        else None)
    feedback_sign_matches = (
        bool(feedback_delta * expected_throttle_delta > 0.0)
        if feedback_delta is not None and expected_throttle_delta is not None
        else None)
    feedback_magnitude_fraction = (
        abs(feedback_delta / expected_throttle_delta)
        if feedback_delta is not None and expected_throttle_delta not in (None, 0.0)
        else None)
    return {
        **spec,
        "samples_in_target_plateau": len(plateau),
        "target_plateau_s": [plateau_start, plateau_end],
        "measured_speed_mps": _quantiles(speeds),
        "odometry_speed_mps": _quantiles(odometry_speeds),
        "odom_truth_speed_abs_gap_mps": _quantiles(odom_truth_speed_gap),
        "requested_speed_within_0p15_fraction": speed_fraction,
        "pre_stimulus_window_s": [pre_start, pre_end],
        "pre_stimulus_truth_speed_mps": _quantiles(pre_stimulus_speeds),
        "pre_stimulus_abs_speed_error_mps": _quantiles(
            pre_stimulus_speed_error),
        "pre_stimulus_steering_feedback_rad": _quantiles(
            pre_stimulus_steering),
        "pre_stimulus_abs_steering_error_rad": _quantiles(
            pre_stimulus_steering_error),
        "pre_stimulus_throttle_feedback_norm": _quantiles(
            pre_stimulus_throttle_feedback),
        "pre_stimulus_throttle_command_norm": _quantiles(
            pre_stimulus_throttle_commands),
        "pre_stimulus_wheel_body_speed_abs_mismatch_mps": _quantiles(
            pre_stimulus_wheel_body_mismatch),
        "measured_steering_feedback_rad": _quantiles(steering),
        "throttle_feedback_norm": _quantiles(throttle_feedback),
        "throttle_command_norm": _quantiles(throttle_commands),
        "throttle_command_feedback_abs_gap": _quantiles(
            throttle_command_feedback_gap),
        "throttle_change_tracking": {
            "expected_delta_norm": expected_throttle_delta,
            "pre_stimulus_command_median": pre_command_median,
            "plateau_command_median": plateau_command_median,
            "command_delta_norm": command_delta,
            "command_delta_residual_norm": command_residual,
            "pre_stimulus_feedback_median": pre_feedback_median,
            "plateau_feedback_median": plateau_feedback_median,
            "feedback_delta_norm": feedback_delta,
            "feedback_delta_residual_norm": feedback_residual,
            "feedback_delta_sign_matches": feedback_sign_matches,
            "feedback_delta_magnitude_fraction": feedback_magnitude_fraction,
        },
        "target_steering_within_0p05_fraction": target_fraction,
        "steering_command_feedback_abs_gap_rad": _quantiles(
            command_feedback_gap.tolist()),
        "rear_wheel_body_speed_abs_mismatch_mps": _quantiles(wheel_body_mismatch),
        "simulator_truth_yaw_rate_rps": _quantiles(yaw_rate),
        "valid_sequence_samples": len(samples),
    }


def _settle_metrics(label: str, samples: tuple[MotionSample, ...]
                    ) -> dict[str, Any] | None:
    if not label.startswith(SETTLE_PREFIXES):
        return None
    probe_label = "probe_yawerr_" + label[len("settle_yawerr_"):]
    spec = parse_label(probe_label)
    if spec is None:
        return None
    target = (0.0 if spec["event"] == "onset" else
              spec["turn_sign"] * spec["requested_abs_steering_rad"])
    speeds: list[float] = []
    steering: list[float] = []
    wheel_body_mismatch: list[float] = []
    low_speed_high_steer_samples = 0
    for sample in samples:
        if not 0.0 <= sample.time_s <= 0.75:
            continue
        rigid = sample.simulator_rigid_state
        if rigid is None or len(rigid) != 13 or not np.isfinite(rigid).all():
            continue
        speed = math.hypot(float(rigid[7]), float(rigid[8]))
        steer = float(sample.actuators[0])
        speeds.append(speed)
        steering.append(steer)
        if speed < 0.5 and abs(steer) >= 0.2:
            low_speed_high_steer_samples += 1
        wheels = sample.rear_wheel_surface_mps
        if wheels is not None and len(wheels) == 2 and np.isfinite(wheels).all():
            wheel_body_mismatch.append(
                abs(float(np.mean(np.abs(wheels))) - speed))
    return {
        **spec,
        "phase_label": label,
        "unscored_settle_phase": True,
        "samples_in_settle_window": len(speeds),
        "settle_window_s": [0.0, 0.75],
        "measured_speed_mps": _quantiles(speeds),
        "measured_steering_feedback_rad": _quantiles(steering),
        "start_steering_within_0p05_fraction": (
            float(np.mean(np.abs(np.asarray(steering) - target)
                          <= STEERING_TOLERANCE_RAD)) if steering else None),
        "rear_wheel_body_speed_abs_mismatch_mps": _quantiles(
            wheel_body_mismatch),
        "low_speed_high_steer_samples": low_speed_high_steer_samples,
        "valid_sequence_samples": len(samples),
    }


def audit(capture: Capture) -> dict[str, Any]:
    by_label: dict[str, list[MotionSample]] = defaultdict(list)
    settle_by_label: dict[str, list[MotionSample]] = defaultdict(list)
    segment_counts: dict[str, int] = defaultdict(int)
    for label, sequence in zip(capture.sequence_labels, capture.sequences):
        if label.startswith(PROFILE_PREFIXES):
            by_label[label].extend(sequence)
            segment_counts[label] += 1
        elif label.startswith(SETTLE_PREFIXES):
            settle_by_label[label].extend(sequence)

    probes = []
    unparsed = []
    for label in sorted(by_label):
        spec = parse_label(label)
        if spec is None:
            unparsed.append(label)
            continue
        result = _sequence_metrics(tuple(by_label[label]), spec)
        result["phase_label"] = label
        result["continuity_segments"] = segment_counts[label]
        probes.append(result)

    settle_phases = []
    unparsed_settle_labels = []
    for label in sorted(settle_by_label):
        result = _settle_metrics(label, tuple(settle_by_label[label]))
        if result is None:
            unparsed_settle_labels.append(label)
        else:
            settle_phases.append(result)

    families = {row["profile_family"] for row in probes}
    expected_probe_count = sum(
        EXPECTED_PROBES_BY_FAMILY[family] for family in families)
    plateau_samples = [row["samples_in_target_plateau"] for row in probes]
    attained = [row["target_steering_within_0p05_fraction"] for row in probes
                if row["target_steering_within_0p05_fraction"] is not None]
    speed_groups: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in probes:
        value = row["measured_speed_mps"]["p50"]
        if value is not None:
            speed_groups[(row["requested_speed_mps"],
                          row["requested_abs_steering_rad"])].append(value)

    return {
        "bag": str(capture.path),
        "phase_gate": {
            "phase_markers_present": capture.has_phase_markers,
            "completed_phases": capture.phase_count,
            "valid_phases": capture.valid_phase_count,
            "invalid_phases": capture.invalid_phase_count,
            "unscored_phases": capture.unscored_phase_count,
            "collision_count_start_end": [capture.collision_count_start,
                                           capture.collision_count_end],
            "bridge_timing_faults": capture.timing_faults,
            "runner_aborted": capture.aborted,
            "runner_reason": capture.reason,
        },
        "probe_count": len(probes),
        "expected_probe_count": expected_probe_count,
        "missing_probe_count_from_expected": max(
            0, expected_probe_count - len(probes)),
        "unparsed_probe_labels": unparsed,
        "probe_segments_split_by_packet_gaps": sum(
            max(0, count - 1) for count in segment_counts.values()),
        "unscored_settle_phase_count": len(settle_phases),
        "unparsed_settle_labels": unparsed_settle_labels,
        "settle_low_speed_high_steer_sample_count": sum(
            row["low_speed_high_steer_samples"] for row in settle_phases),
        "settle_low_speed_high_steer_phase_count": sum(
            row["low_speed_high_steer_samples"] > 0 for row in settle_phases),
        "settle_phases": settle_phases,
        "target_steering_within_0p05_fraction_summary": _quantiles(attained),
        "target_plateau_sample_count_summary": _quantiles(
            [float(value) for value in plateau_samples]),
        "measured_speed_by_requested_speed_and_abs_steering": [
            {
                "requested_speed_mps": key[0],
                "requested_abs_steering_rad": key[1],
                "probe_count": len(values),
                "measured_speed_mps_p50_across_probe_medians": float(
                    np.median(values)),
                "measured_speed_mps_min_max_across_probe_medians": [
                    float(min(values)), float(max(values))],
            }
            for key, values in sorted(speed_groups.items())
        ],
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="closed rosbag2 sqlite database")
    parser.add_argument("--output", type=Path, required=True,
                        help="write the audit JSON to this path")
    args = parser.parse_args()
    result = audit(load_capture(args.bag, include_nonvalid_phases=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "probe_count", "expected_probe_count", "missing_probe_count_from_expected",
        "probe_segments_split_by_packet_gaps", "target_steering_within_0p05_fraction_summary",
        "target_plateau_sample_count_summary", "unscored_settle_phase_count",
        "settle_low_speed_high_steer_sample_count",
        "settle_low_speed_high_steer_phase_count", "phase_gate")}, indent=2))
    print(f"wrote {args.output}")
    safe = (result["phase_gate"]["collision_count_start_end"][0]
            == result["phase_gate"]["collision_count_start_end"][1]
            and result["phase_gate"]["bridge_timing_faults"] == 0
            and not result["phase_gate"]["runner_aborted"]
            and result["phase_gate"]["invalid_phases"] == 0)
    return 0 if (result["probe_count"] == result["expected_probe_count"]
                 and not result["unparsed_probe_labels"]
                 and safe) else 2


if __name__ == "__main__":
    raise SystemExit(main())
