#!/usr/bin/env python3
"""Audit achieved vehicle state in the reset-isolated 1--5% throttle crawl test.

Throttle labels are treatments, not measured speed targets. The report uses
packet-aligned simulator truth only for offline diagnostics and keeps the
actual actuator feedback, encoder/body-speed mismatch, IMU, and odometry
separate. Run only after rosbag2 has closed the capture.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.analyze_open_plane_dynamics import PHASE
from tools.evaluate_open_plane_body_dynamics import (
    Capture, MotionSample, STEERING_LIMIT_RAD, load_capture)


LABEL = re.compile(
    r"^probe_yawerr_crawl_throttle_t(?P<throttle>\d{2})_"
    r"a(?P<steering>[+-]\d+\.\d{3})_r(?P<repeat>\d{2})$")
THROTTLE_LEVELS_PERCENT = (1, 2, 3, 4, 5)
STEERING_LEVELS_RAD = (0.0, -0.20, 0.20, -0.35, 0.35, -0.50, 0.50)
REPEATS = (1, 2)
HOLD_SECONDS = 8.0
FINAL_WINDOW_START_S = 6.0
SPEED_BANDS_MPS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5),
                   (1.5, 2.0), (2.0, 3.0), (3.0, 4.0),
                   (4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 12.0))
WHEELBASE_M = 0.324


def _quantiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"p10": None, "p50": None, "p90": None}
    p10, p50, p90 = np.quantile(array, (0.10, 0.50, 0.90))
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90)}


def _probe_label(label: str) -> dict[str, float | int] | None:
    match = LABEL.fullmatch(label)
    if match is None:
        return None
    return {
        "throttle_command_percent": int(match.group("throttle")),
        "steering_target_rad": float(match.group("steering")),
        "repeat": int(match.group("repeat")),
    }


def _probe_metrics(samples: list[MotionSample], spec: dict[str, float | int]
                   ) -> dict[str, Any]:
    samples = sorted(samples, key=lambda sample: sample.time_s)
    truth_speed: list[float] = []
    odom_speed: list[float] = []
    steering_feedback: list[float] = []
    steering_command_rad: list[float] = []
    throttle_feedback: list[float] = []
    throttle_command: list[float] = []
    yaw_truth: list[float] = []
    yaw_imu: list[float] = []
    roll: list[float] = []
    roll_rate: list[float] = []
    wheel_body_mismatch: list[float] = []
    times: list[float] = []

    for sample in samples:
        rigid = sample.simulator_rigid_state
        if rigid is None or len(rigid) != 13 or not np.isfinite(rigid).all():
            continue
        speed = math.hypot(float(rigid[7]), float(rigid[8]))
        times.append(float(sample.time_s))
        truth_speed.append(speed)
        odom_speed.append(math.hypot(float(sample.state[0]),
                                     float(sample.state[1])))
        steering_feedback.append(float(sample.actuators[0]))
        throttle_feedback.append(float(sample.actuators[1]))
        throttle_command.append(float(sample.actuators[2]))
        steering_command_rad.append(
            float(sample.actuators[3]) * STEERING_LIMIT_RAD)
        yaw_truth.append(float(rigid[12]))
        yaw_imu.append(
            float(sample.imu_yaw_rate_rps)
            if sample.imu_yaw_rate_rps is not None else math.nan)
        roll.append(
            float(sample.imu_roll_pitch_rad[0])
            if sample.imu_roll_pitch_rad is not None else math.nan)
        roll_rate.append(
            float(sample.imu_roll_pitch_rate_rps[0])
            if sample.imu_roll_pitch_rate_rps is not None else math.nan)
        wheels = sample.rear_wheel_surface_mps
        if wheels is not None and len(wheels) == 2 and np.isfinite(wheels).all():
            wheel_body_mismatch.append(
                abs(float(np.mean(np.abs(wheels))) - speed))
        else:
            wheel_body_mismatch.append(math.nan)

    final_indices = [index for index, time_s in enumerate(times)
                     if FINAL_WINDOW_START_S <= time_s <= HOLD_SECONDS]
    final_speed = [truth_speed[index] for index in final_indices]
    final_times = [times[index] for index in final_indices]
    final_speed_slope = None
    if len(final_speed) >= 10 and np.ptp(final_times) > 0.25:
        final_speed_slope = float(np.polyfit(final_times, final_speed, 1)[0])

    target_steer = float(spec["steering_target_rad"])
    target_throttle = float(spec["throttle_command_percent"]) / 100.0
    final_commands = [throttle_command[index] for index in final_indices]
    final_feedback = [throttle_feedback[index] for index in final_indices]
    final_steering = [steering_feedback[index] for index in final_indices]
    final_steering_commands = [steering_command_rad[index]
                               for index in final_indices]
    return {
        **spec,
        "sample_count": len(truth_speed),
        "duration_s": (max(times) - min(times)) if times else 0.0,
        "speed_truth_mps_all": _quantiles(truth_speed),
        "speed_truth_mps_final_2s": _quantiles(final_speed),
        "speed_truth_final_2s_slope_mps2": final_speed_slope,
        "speed_odom_mps_final_2s": _quantiles(
            [odom_speed[index] for index in final_indices]),
        "odom_truth_speed_abs_error_mps_final_2s": _quantiles(
            [abs(odom_speed[index] - truth_speed[index])
             for index in final_indices]),
        "throttle_command_norm_final_2s": _quantiles(final_commands),
        "throttle_feedback_norm_final_2s": _quantiles(final_feedback),
        "throttle_feedback_abs_error_norm_final_2s": _quantiles(
            [abs(value - target_throttle) for value in final_feedback]),
        "steering_feedback_rad_final_2s": _quantiles(final_steering),
        "steering_feedback_abs_error_rad_final_2s": _quantiles(
            [abs(value - target_steer) for value in final_steering]),
        "steering_command_feedback_abs_gap_rad_final_2s": _quantiles(
            [abs(command - feedback)
             for command, feedback in zip(final_steering_commands,
                                          final_steering)]),
        "wheel_body_speed_abs_mismatch_mps_final_2s": _quantiles(
            [wheel_body_mismatch[index] for index in final_indices]),
        "yaw_rate_truth_rps_final_2s": _quantiles(
            [yaw_truth[index] for index in final_indices]),
        "yaw_rate_imu_rps_final_2s": _quantiles(
            [yaw_imu[index] for index in final_indices]),
        "imu_roll_rad_final_2s": _quantiles(
            [roll[index] for index in final_indices]),
        "imu_roll_rate_rps_final_2s": _quantiles(
            [roll_rate[index] for index in final_indices]),
        "final_2s_speed_band_fraction": {
            f"{low:g}..{high:g}mps": (
                float(np.mean((np.asarray(final_speed) >= low)
                              & (np.asarray(final_speed) < high)))
                if final_speed else 0.0)
            for low, high in SPEED_BANDS_MPS
        },
    }


def _error_summary(errors: list[float] | np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(errors, dtype=np.float64))
    absolute = absolute[np.isfinite(absolute)]
    if not len(absolute):
        return {"samples": 0}
    return {
        "samples": int(len(absolute)),
        "abs_error_p50_radps": float(np.quantile(absolute, 0.50)),
        "abs_error_p95_radps": float(np.quantile(absolute, 0.95)),
        "abs_error_max_radps": float(np.max(absolute)),
        "count_abs_error_over_0p1_radps": int(np.count_nonzero(absolute > 0.1)),
    }


def _steady_yaw_bicycle_fit(probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure a low-speed steady yaw relation using runtime-like inputs.

    Odom body speed and measured steering feedback form the predictor. The
    simulator-truth yaw rate is only the offline target. Leave-out results are
    within-capture interpolation checks, not independent-run validation.
    """
    rows = [row for row in probes
            if row.get("speed_odom_mps_final_2s", {}).get("p50") is not None
            and row.get("steering_feedback_rad_final_2s", {}).get("p50") is not None
            and row.get("yaw_rate_truth_rps_final_2s", {}).get("p50") is not None]
    if not rows:
        return {"status": "no scored probe conditions"}
    speed = np.asarray([
        row["speed_odom_mps_final_2s"]["p50"] for row in rows], dtype=float)
    steering = np.asarray([
        row["steering_feedback_rad_final_2s"]["p50"] for row in rows], dtype=float)
    truth = np.asarray([
        row["yaw_rate_truth_rps_final_2s"]["p50"] for row in rows], dtype=float)
    bicycle = speed * np.tan(steering) / WHEELBASE_M
    denominator = float(np.dot(bicycle, bicycle))
    if denominator <= 1e-12:
        return {"status": "no nonzero steering excitation"}
    gain = float(np.dot(bicycle, truth) / denominator)
    paired_errors: list[float] = []
    speed_level_errors: list[float] = []
    pair_keys = {(int(row["throttle_command_percent"]),
                  float(row["steering_target_rad"])) for row in rows}
    throttle_levels = {int(row["throttle_command_percent"]) for row in rows}
    for key in pair_keys:
        train = np.asarray([
            (int(row["throttle_command_percent"]),
             float(row["steering_target_rad"])) != key for row in rows])
        test = ~train
        local_denom = float(np.dot(bicycle[train], bicycle[train]))
        if np.any(test) and local_denom > 1e-12:
            local_gain = float(np.dot(bicycle[train], truth[train]) / local_denom)
            paired_errors.extend((truth[test] - local_gain * bicycle[test]).tolist())
    for level in throttle_levels:
        train = np.asarray([
            int(row["throttle_command_percent"]) != level for row in rows])
        test = ~train
        local_denom = float(np.dot(bicycle[train], bicycle[train]))
        if np.any(test) and local_denom > 1e-12:
            local_gain = float(np.dot(bicycle[train], truth[train]) / local_denom)
            speed_level_errors.extend(
                (truth[test] - local_gain * bicycle[test]).tolist())
    return {
        "status": "exploratory steady-state fit",
        "predictor": "odom body speed × tan(measured steering feedback) / wheelbase",
        "target": "simulator-truth yaw rate; offline label only",
        "wheelbase_m": WHEELBASE_M,
        "conditions": len(rows),
        "throttle_steering_condition_groups": len(pair_keys),
        "fitted_yaw_authority_gain": gain,
        "unscaled_bicycle_in_sample": _error_summary(truth - bicycle),
        "gain_scaled_bicycle_in_sample": _error_summary(truth - gain * bicycle),
        "leave_both_repeats_of_condition_out": _error_summary(paired_errors),
        "leave_throttle_level_out_unseen_speed": _error_summary(speed_level_errors),
        "caveat": (
            "These repeated conditions are from one capture. Leave-out scores "
            "test in-capture interpolation, not independent-run or transient "
            "observer accuracy; do not integrate from this result alone."),
    }


def _reset_recovery_count(path: Path) -> int:
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String

    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT id FROM topics WHERE name=?", (PHASE,)).fetchone()
        if row is None:
            return 0
        return sum(
            json.loads(deserialize_message(bytes(payload), String).data).get(
                "event") == "sim_reset_recovered"
            for (payload,) in connection.execute(
                "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
                (int(row[0]),)))
    finally:
        connection.close()


def audit(capture: Capture) -> dict[str, Any]:
    by_label: dict[str, list[MotionSample]] = defaultdict(list)
    for label, sequence in zip(capture.sequence_labels, capture.sequences):
        if label.startswith("probe_yawerr_crawl_throttle_"):
            by_label[label].extend(sequence)

    probes = []
    unparsed_labels = []
    for label in sorted(by_label):
        spec = _probe_label(label)
        if spec is None:
            unparsed_labels.append(label)
            continue
        probes.append(_probe_metrics(by_label[label], spec) | {
            "phase_label": label})

    final_by_throttle: dict[int, list[float]] = defaultdict(list)
    final_by_band: dict[str, list[float]] = defaultdict(list)
    for probe in probes:
        throttle = int(probe["throttle_command_percent"])
        speed = probe["speed_truth_mps_final_2s"]["p50"]
        if speed is not None:
            final_by_throttle[throttle].append(float(speed))
        for band, fraction in probe["final_2s_speed_band_fraction"].items():
            if fraction > 0.0:
                final_by_band[band].append(float(fraction))

    missing_expected = []
    observed = {(int(row["throttle_command_percent"]),
                 float(row["steering_target_rad"]), int(row["repeat"]))
                for row in probes}
    for throttle in THROTTLE_LEVELS_PERCENT:
        for steering in STEERING_LEVELS_RAD:
            for repeat in REPEATS:
                key = (throttle, steering, repeat)
                if key not in observed:
                    missing_expected.append({
                        "throttle_command_percent": throttle,
                        "steering_target_rad": steering,
                        "repeat": repeat,
                    })

    return {
        "bag": str(capture.path),
        "units": {
            "yaw_rate": "rad/s",
            "steering": "rad",
            "throttle": "normalized 0..1",
            "speed": "m/s",
        },
        "run_quality": {
            "runner_aborted": capture.aborted,
            "runner_reason": capture.reason,
            "phase_count": capture.phase_count,
            "valid_phases": capture.valid_phase_count,
            "invalid_phases": capture.invalid_phase_count,
            "unscored_phases": capture.unscored_phase_count,
            "collision_count_start_end": [capture.collision_count_start,
                                           capture.collision_count_end],
            "bridge_timing_faults": capture.timing_faults,
            "packet_sequence_match_fraction": (
                capture.packet_sequence_matched_samples /
                capture.packet_sequence_total_samples
                if capture.packet_sequence_total_samples else None),
            "bagwide_stream_rate_hz_p50_max_gap_ms": capture.stream_stats,
            "within_phase_stream_rate_hz_p50_max_gap_ms": (
                capture.phase_stream_stats),
            "reset_recoveries": _reset_recovery_count(capture.path),
            "expected_reset_recoveries": 70,
        },
        "schedule": {
            "expected_probe_conditions": 70,
            "observed_probe_conditions": len(probes),
            "missing_probe_conditions": missing_expected,
            "unparsed_probe_labels": unparsed_labels,
        },
        "probe_condition_results": probes,
        "steady_state_yaw_bicycle_fit": _steady_yaw_bicycle_fit(probes),
        "final_2s_speed_medians_by_throttle_command_percent": {
            str(level): _quantiles(values)
            for level, values in sorted(final_by_throttle.items())
        },
        "speed_bands_observed_in_final_2s_by_condition": {
            band: len(values) for band, values in sorted(final_by_band.items())
        },
        "interpretation": (
            "Throttle commands are randomized treatments. Use measured truth "
            "speed and actuator feedback to classify achieved regimes; do not "
            "treat a command percentage as a speed target. Truth is diagnostic "
            "label data only, not a model input."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="closed rosbag2 SQLite database")
    parser.add_argument("--output", type=Path, required=True,
                        help="write full audit JSON")
    args = parser.parse_args()
    result = audit(load_capture(args.bag, include_nonvalid_phases=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    quality = result["run_quality"]
    schedule = result["schedule"]
    summary = {
        "runner_aborted": quality["runner_aborted"],
        "collisions": quality["collision_count_start_end"],
        "bridge_timing_faults": quality["bridge_timing_faults"],
        "phase_count": quality["phase_count"],
        "expected_phase_count": 210,
        "valid_phases": quality["valid_phases"],
        "reset_recoveries": quality["reset_recoveries"],
        "expected_reset_recoveries": quality["expected_reset_recoveries"],
        "probe_conditions": schedule["observed_probe_conditions"],
        "expected_probe_conditions": schedule["expected_probe_conditions"],
        "missing_probe_conditions": len(schedule["missing_probe_conditions"]),
        "final_2s_speed_medians_by_throttle_command_percent": result[
            "final_2s_speed_medians_by_throttle_command_percent"],
        "speed_bands_observed_final_2s": result[
            "speed_bands_observed_in_final_2s_by_condition"],
        "steady_state_yaw_bicycle_fit": result[
            "steady_state_yaw_bicycle_fit"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    safe = (
        not quality["runner_aborted"]
        and quality["collision_count_start_end"][0]
        == quality["collision_count_start_end"][1]
        and quality["bridge_timing_faults"] == 0
        and not quality["invalid_phases"]
        and quality["phase_count"] == 210
        and quality["valid_phases"] == 70
        and quality["reset_recoveries"] == 70
        and schedule["observed_probe_conditions"] == 70
        and not schedule["missing_probe_conditions"]
        and not schedule["unparsed_probe_labels"]
    )
    return 0 if safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
