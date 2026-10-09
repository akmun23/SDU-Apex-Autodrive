#!/usr/bin/env python3
"""Measure command-to-feedback onset from raw receipt timestamps in a probe bag."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_open_plane_body_dynamics import load_capture
from tools.racing.specialists.score_yaw_atlas_transition_events import (
    _probe_phases,
)
from tools.racing.specialists.audit_yaw_error_probe_capture import parse_label


STEERING_LIMIT_RAD = 0.5236
COMMAND_DEPARTURE_RAD = 0.01
FEEDBACK_DEPARTURE_RAD = 0.01


def _baseline(times_ns: np.ndarray, values: np.ndarray,
              begin_ns: int, end_ns: int) -> float | None:
    selected = values[(times_ns >= begin_ns) & (times_ns < end_ns)]
    if len(selected) < 2:
        return None
    return float(np.median(selected))


def _quantiles(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return {"n": 0}
    p10, p50, p90 = np.quantile(data, (0.1, 0.5, 0.9))
    return {
        "n": int(len(data)),
        "p10_ms": float(p10),
        "median_ms": float(p50),
        "p90_ms": float(p90),
        "min_ms": float(np.min(data)),
        "max_ms": float(np.max(data)),
        "count_below_50ms": int(np.count_nonzero(data < 50.0)),
        "count_50_to_75ms": int(np.count_nonzero(
            (data >= 50.0) & (data < 75.0))),
        "count_at_least_75ms": int(np.count_nonzero(data >= 75.0)),
    }


def analyze(bag: Path, expected_probe_count: int | None = 216
            ) -> dict[str, Any]:
    capture = load_capture(bag)
    phases, experiment_end = _probe_phases(
        bag, expected_probe_count=expected_probe_count)
    command_times, command_norm = capture.actuator_streams["steering_command"]
    feedback_times, feedback_rad = capture.actuator_streams["steering_feedback"]
    command_times = np.asarray(command_times, dtype=np.int64)
    command_rad = np.asarray(command_norm, dtype=np.float64) * STEERING_LIMIT_RAD
    feedback_times = np.asarray(feedback_times, dtype=np.int64)
    feedback_rad = np.asarray(feedback_rad, dtype=np.float64)

    measurements: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    packet_grid_measurements: list[dict[str, Any]] = []
    packet_grid_grouped: dict[str, list[int]] = defaultdict(list)
    for phase in phases:
        spec = parse_label(phase.label)
        if spec is None:
            continue
        scheduled_ns = phase.start_ns + int(spec["delay_s"] * 1e9)
        command_before = _baseline(
            command_times, command_rad, scheduled_ns - 150_000_000,
            scheduled_ns - 25_000_000)
        feedback_before = _baseline(
            feedback_times, feedback_rad, scheduled_ns - 250_000_000,
            scheduled_ns - 50_000_000)
        if command_before is None or feedback_before is None:
            continue

        target = float(spec["target_steering_rad"])
        direction = float(np.sign(target - command_before))
        if direction == 0.0:
            continue
        changed = np.flatnonzero(
            (command_times >= scheduled_ns - 10_000_000)
            & (np.abs(command_rad - command_before) >= COMMAND_DEPARTURE_RAD))
        if not len(changed):
            continue
        command_start_ns = int(command_times[changed[0]])
        moved = np.flatnonzero(
            (feedback_times >= command_start_ns)
            & ((feedback_rad - feedback_before) * direction
               >= FEEDBACK_DEPARTURE_RAD))
        if not len(moved):
            continue
        feedback_start_ns = int(feedback_times[moved[0]])
        latency_ms = (feedback_start_ns - command_start_ns) / 1e6
        key = f"{spec['event']}/{spec['transition_mode']}"
        grouped[key].append(float(latency_ms))
        measurements.append({
            "phase": phase.label,
            "event": spec["event"],
            "speed_mps": spec["requested_speed_mps"],
            "absolute_steering_rad": spec["requested_abs_steering_rad"],
            "turn_sign": spec["turn_sign"],
            "scheduled_delay_s": spec["delay_s"],
            "transition_mode": spec["transition_mode"],
            "transition_duration_s": spec["transition_duration_s"],
            "command_start_offset_from_schedule_ms":
                (command_start_ns - scheduled_ns) / 1e6,
            "command_to_feedback_onset_ms": float(latency_ms),
            "pre_command_rad": command_before,
            "pre_feedback_rad": feedback_before,
            "requested_target_rad": target,
        })

        # Recompute onset in simulator packet indices. Receipt-time jitter is
        # retained above as a transport diagnostic, but physical intervals
        # use the established 25-ms packet grid.
        sample_by_packet = {}
        for sequence in capture.sequences:
            for sample in sequence:
                if (phase.start_ns - 300_000_000 <= sample.receipt_ns
                        < phase.end_ns and sample.packet_sequence >= 0):
                    sample_by_packet[sample.packet_sequence] = sample
        packet_samples = sorted(sample_by_packet.values(),
                                key=lambda sample: sample.packet_sequence)
        prior_command = [sample for sample in packet_samples
                         if scheduled_ns - 150_000_000 <= sample.receipt_ns
                         < scheduled_ns - 25_000_000]
        prior_feedback = [sample for sample in packet_samples
                          if scheduled_ns - 250_000_000 <= sample.receipt_ns
                          < scheduled_ns - 50_000_000]
        if len(prior_command) >= 2 and len(prior_feedback) >= 2:
            grid_command_before = float(np.median([
                sample.actuators[3] * STEERING_LIMIT_RAD
                for sample in prior_command]))
            grid_feedback_before = float(np.median([
                sample.actuators[0] for sample in prior_feedback]))
            grid_command_start = next((sample for sample in packet_samples
                if sample.receipt_ns >= scheduled_ns - 10_000_000
                and abs(sample.actuators[3] * STEERING_LIMIT_RAD
                        - grid_command_before) >= COMMAND_DEPARTURE_RAD), None)
            if grid_command_start is not None:
                grid_feedback_start = next((sample for sample in packet_samples
                    if sample.packet_sequence >= grid_command_start.packet_sequence
                    and direction * (sample.actuators[0]
                                     - grid_feedback_before)
                    >= FEEDBACK_DEPARTURE_RAD), None)
                if grid_feedback_start is not None:
                    onset_steps = (grid_feedback_start.packet_sequence
                                   - grid_command_start.packet_sequence)
                    event_samples = [sample for sample in packet_samples
                        if grid_command_start.packet_sequence
                        <= sample.packet_sequence
                        <= grid_feedback_start.packet_sequence]
                    gap_free = all(
                        right.packet_sequence == left.packet_sequence + 1
                        for left, right in zip(event_samples, event_samples[1:]))
                    packet_grid_grouped[key].append(int(onset_steps))
                    packet_grid_measurements.append({
                        "phase": phase.label,
                        "event": spec["event"],
                        "speed_mps": spec["requested_speed_mps"],
                        "absolute_steering_rad": spec["requested_abs_steering_rad"],
                        "turn_sign": spec["turn_sign"],
                        "transition_mode": spec["transition_mode"],
                        "transition_duration_s": spec["transition_duration_s"],
                        "command_start_packet_sequence":
                            int(grid_command_start.packet_sequence),
                        "feedback_onset_packet_sequence":
                            int(grid_feedback_start.packet_sequence),
                        "command_to_feedback_onset_steps": int(onset_steps),
                        "nominal_delay_ms": float(onset_steps * 25.0),
                        "packet_sequence_gap_free": bool(gap_free),
                    })

    summary = {key: _quantiles(values)
               for key, values in sorted(grouped.items())}
    packet_grid_summary = {
        key: {
            **_quantiles([25.0 * value for value in values]),
            "step_counts": {
                str(step): int(values.count(step))
                for step in sorted(set(values))
            },
        }
        for key, values in sorted(packet_grid_grouped.items())
    }
    return {
        "title": "Raw steering command-to-feedback onset timing",
        "bag": str(bag),
        "measurement": {
            "timestamps": "raw command/feedback values, receipt times, and bridge packet sequence",
            "command_onset_threshold_rad": COMMAND_DEPARTURE_RAD,
            "feedback_onset_threshold_rad": FEEDBACK_DEPARTURE_RAD,
            "feedback_onset": "first feedback sample at least 0.01 rad toward the requested target after command onset",
            "pretransition_command_window_ms": [-150, -25],
            "pretransition_feedback_window_ms": [-250, -50],
            "receipt_time_ms_is_transport_diagnostic_not_simulation_dt": True,
            "physical_interval_assumption": "one simulator packet = 25 ms",
            "packet_grid_onset": "first fixed-grid command/feedback sample crossing the same 0.01-rad thresholds, differenced by simulator packet sequence",
        },
        "phase_count": len(measurements),
        "experiment_end": experiment_end,
        "summary_by_event_and_profile": summary,
        "packet_grid_summary_by_event_and_profile": packet_grid_summary,
        "measurements": measurements,
        "packet_grid_measurements": packet_grid_measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-probe-count", type=int, default=216,
                        help="expected valid probe count; 0 accepts any positive count")
    args = parser.parse_args()
    expected = args.expected_probe_count or None
    if expected is not None and expected < 1:
        parser.error("--expected-probe-count must be positive or 0")
    result = analyze(args.bag, expected_probe_count=expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(args.output)
    print(json.dumps(result["summary_by_event_and_profile"], sort_keys=True))
    print(json.dumps(result["packet_grid_summary_by_event_and_profile"],
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
