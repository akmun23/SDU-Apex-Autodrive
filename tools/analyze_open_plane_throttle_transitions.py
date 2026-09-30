#!/usr/bin/env python3
"""Audit a closed reset-isolated throttle-transition rosbag."""

from __future__ import annotations

import argparse
from array import array
import bisect
from collections import defaultdict
import json
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Int32, String

from tools import analyze_open_plane_dynamics as rosbag_tools


ODOM = rosbag_tools.ODOM
STEERING = rosbag_tools.STEERING
THROTTLE = rosbag_tools.THROTTLE_FEEDBACK
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
THROTTLE_COMMAND = rosbag_tools.THROTTLE_COMMAND
COLLISIONS = rosbag_tools.COLLISIONS
TIMING_FAULT = rosbag_tools.TIMING_FAULT
PHASE = rosbag_tools.PHASE
ACTIVE_STREAMS = (
    ODOM, STEERING, THROTTLE, rosbag_tools.LEFT_ENCODER,
    rosbag_tools.RIGHT_ENCODER, rosbag_tools.IMU,
    rosbag_tools.PACKET_TIMING, STEERING_COMMAND, THROTTLE_COMMAND,
)
THROTTLE_TRACKING_TOLERANCE = 0.005
COMMAND_TRACKING_TOLERANCE = 0.002


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def _median_window(times: array, values: array, lower: int,
                   upper: int) -> float | None:
    first = bisect.bisect_left(times, lower)
    last = bisect.bisect_left(times, upper)
    if first >= last:
        return None
    return float(statistics.median(values[first:last]))


def _decode_events(connection: sqlite3.Connection,
                   topics: dict[str, tuple[int, str]]) -> tuple[
                       dict[int, tuple[int, dict[str, Any]]],
                       dict[int, tuple[int, dict[str, Any]]],
                       dict[int, tuple[int, dict[str, Any]]],
                       list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    starts: dict[int, tuple[int, dict[str, Any]]] = {}
    ends: dict[int, tuple[int, dict[str, Any]]] = {}
    stimuli: dict[int, tuple[int, dict[str, Any]]] = {}
    resets: list[tuple[int, dict[str, Any]]] = []
    experiment_end: dict[str, Any] = {}
    for receipt_ns, message in rosbag_tools._messages(connection, topics, PHASE):
        try:
            event = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            continue
        kind = event.get("event")
        if kind == "phase_start":
            starts[int(event["phase_index"])] = (receipt_ns, event)
        elif kind == "phase_end":
            ends[int(event["phase_index"])] = (receipt_ns, event)
        elif kind == "throttle_slew_stimulus":
            stimuli[int(event["phase_index"])] = (receipt_ns, event)
        elif kind in ("sim_reset_start", "sim_reset_recovered"):
            resets.append((receipt_ns, event))
        elif kind == "experiment_end":
            experiment_end = event
    return starts, ends, stimuli, resets, experiment_end


def _load_scalar(connection: sqlite3.Connection,
                 topic_id: int, message_type: type) -> tuple[array, array]:
    times = array("q")
    values = array("d")
    for receipt_ns, payload in connection.execute(
            "SELECT timestamp, data FROM messages "
            "WHERE topic_id=? ORDER BY timestamp, id", (topic_id,)):
        message = rosbag_tools.deserialize_message(bytes(payload), message_type)
        value = float(message.data)
        if math.isfinite(value):
            times.append(int(receipt_ns))
            values.append(value)
    return times, values


def _load_odometry(connection: sqlite3.Connection,
                   topic_id: int) -> tuple[array, array]:
    times = array("q")
    speeds = array("d")
    for receipt_ns, payload in connection.execute(
            "SELECT timestamp, data FROM messages "
            "WHERE topic_id=? ORDER BY timestamp, id", (topic_id,)):
        message = rosbag_tools.deserialize_message(bytes(payload), Odometry)
        velocity = message.twist.twist.linear
        speed = math.hypot(float(velocity.x), float(velocity.y))
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        if all(math.isfinite(value) for value in (speed, x, y)):
            times.append(int(receipt_ns))
            speeds.append(speed)
    return times, speeds


def _active_stream_stats(times_by_topic: dict[str, array],
                         intervals: list[tuple[int, int]]) -> dict[str, dict[str, Any]]:
    result = {}
    for topic in ACTIVE_STREAMS:
        receipts = times_by_topic[topic]
        gaps: list[float] = []
        duration_s = 0.0
        for start_ns, end_ns in intervals:
            first = bisect.bisect_left(receipts, start_ns)
            last = bisect.bisect_right(receipts, end_ns)
            if last - first < 2:
                continue
            selected = receipts[first:last]
            local_gaps = [(b - a) / 1e6 for a, b in zip(selected, selected[1:])
                          if b > a]
            gaps.extend(local_gaps)
            duration_s += sum(local_gaps) / 1000.0
        rate_hz = len(gaps) / duration_s if duration_s > 0.0 else None
        p95_ms = _percentile(gaps, 0.95)
        max_ms = max(gaps) if gaps else None
        result[topic] = {
            "rate_hz": rate_hz,
            "gap_p95_ms": p95_ms,
            "gap_max_ms": max_ms,
            "intervals": len(gaps),
            "passes_40hz_gate": bool(
                rate_hz is not None and rate_hz >= 38.0
                and p95_ms is not None and p95_ms <= 35.0
                and max_ms is not None and max_ms <= 60.0),
        }
    return result


def analyze(bag: Path, output: Path) -> dict[str, Any]:
    if not bag.is_file():
        raise ValueError(f"bag does not exist: {bag}")
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = rosbag_tools._topic_map(connection)
        required = (*ACTIVE_STREAMS, COLLISIONS, TIMING_FAULT, PHASE)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError("missing bag topics: " + ", ".join(missing))
        starts, ends, stimuli, reset_events, experiment_end = _decode_events(
            connection, topics)
        if experiment_end.get("profile") != "throttle_transition_surface":
            raise ValueError("bag is not a throttle_transition_surface capture")

        throttle_times, throttle_values = _load_scalar(
            connection, topics[THROTTLE][0], Float32)
        throttle_command_times, throttle_command_values = _load_scalar(
            connection, topics[THROTTLE_COMMAND][0], Float32)
        steering_times, steering_values = _load_scalar(
            connection, topics[STEERING][0], Float32)
        steering_command_times, steering_command_values = _load_scalar(
            connection, topics[STEERING_COMMAND][0], Float32)
        _, collision_values = _load_scalar(
            connection, topics[COLLISIONS][0], Int32)
        _, timing_values = _load_scalar(
            connection, topics[TIMING_FAULT][0], Bool)
        odom_times, odom_speeds = _load_odometry(
            connection, topics[ODOM][0])

        times_by_topic: dict[str, array] = {}
        for topic in ACTIVE_STREAMS:
            times_by_topic[topic] = array(
                "q", (int(row[0]) for row in connection.execute(
                    "SELECT timestamp FROM messages WHERE topic_id=? "
                    "ORDER BY timestamp, id", (topics[topic][0],))))
    finally:
        connection.close()

    intervals = [(starts[index][0], ends[index][0])
                 for index in sorted(set(starts) & set(ends))]
    stream_stats = _active_stream_stats(times_by_topic, intervals)
    phases = []
    seen_pairs: set[tuple[int, int, int, int]] = set()
    duplicate_pairs = []
    initial_positions = []
    response_times = []
    speed_changes = []
    feedback_start_errors = []
    feedback_end_errors = []
    max_speeds = []
    max_displacements = []
    for index, (start_ns, start) in sorted(starts.items()):
        steering = float(start.get("steering_command_rad", math.nan))
        throttle_start = float(start.get("throttle_start_norm", math.nan))
        throttle_end = float(start.get("throttle_end_norm", math.nan))
        replicate = int(start.get("replicate_index", 1))
        key = (round(steering * 10_000), round(throttle_start * 100),
               round(throttle_end * 100), replicate)
        if key in seen_pairs:
            duplicate_pairs.append(key)
        seen_pairs.add(key)
        end_entry = ends.get(index)
        if end_entry is None:
            continue
        end_ns, end = end_entry
        initial = start.get("initial_state", {})
        initial_positions.append((float(initial.get("x_m", math.nan)),
                                  float(initial.get("y_m", math.nan))))
        stimulus = stimuli.get(index)
        stimulus_ns = stimulus[0] if stimulus is not None else None

        def median_window(times: array, values: array,
                          center_ns: int | None) -> float | None:
            if center_ns is None:
                return None
            return _median_window(times, values, center_ns - 500_000_000,
                                  center_ns)

        start_feedback = median_window(throttle_times, throttle_values, stimulus_ns)
        end_feedback = median_window(throttle_times, throttle_values, end_ns)
        start_command = median_window(throttle_command_times,
                                      throttle_command_values, stimulus_ns)
        end_command = median_window(throttle_command_times,
                                    throttle_command_values, end_ns)
        start_speed = median_window(odom_times, odom_speeds, stimulus_ns)
        end_speed = median_window(odom_times, odom_speeds, end_ns)
        steer_feedback = median_window(steering_times, steering_values, end_ns)
        steer_command = median_window(steering_command_times,
                                      steering_command_values, end_ns)
        start_error = (abs(start_feedback - throttle_start)
                       if start_feedback is not None else None)
        end_error = (abs(end_feedback - throttle_end)
                     if end_feedback is not None else None)
        start_command_error = (abs(start_command - throttle_start)
                               if start_command is not None else None)
        end_command_error = (abs(end_command - throttle_end)
                             if end_command is not None else None)
        command_tracking_ok = (
            start_command_error is not None and end_command_error is not None
            and start_command_error <= COMMAND_TRACKING_TOLERANCE
            and end_command_error <= COMMAND_TRACKING_TOLERANCE)
        if start_error is not None:
            feedback_start_errors.append(start_error)
        if end_error is not None:
            feedback_end_errors.append(end_error)
        if start_speed is not None and end_speed is not None:
            speed_changes.append(end_speed - start_speed)
        if end.get("response_observation_s") is not None:
            response_times.append(float(end["response_observation_s"]))
        if end.get("measured_speed_max_mps") is not None:
            max_speeds.append(float(end["measured_speed_max_mps"]))
        if end.get("maximum_displacement_from_spawn_m") is not None:
            max_displacements.append(float(end["maximum_displacement_from_spawn_m"]))

        row = {
            "phase_index": index,
            "replicate_index": replicate,
            "label": start.get("label"),
            "steering_command_rad": steering,
            "throttle_start_norm": throttle_start,
            "throttle_end_norm": throttle_end,
            "throttle_delta_command_norm": throttle_end - throttle_start,
            "start_feedback_median_norm": start_feedback,
            "end_feedback_median_norm": end_feedback,
            "start_feedback_abs_error_norm": start_error,
            "end_feedback_abs_error_norm": end_error,
            "start_command_median_norm": start_command,
            "end_command_median_norm": end_command,
            "start_command_abs_error_norm": start_command_error,
            "end_command_abs_error_norm": end_command_error,
            "speed_at_throttle_step_mps": start_speed,
            "speed_at_response_end_mps": end_speed,
            "response_speed_change_mps": (
                end_speed - start_speed
                if start_speed is not None and end_speed is not None else None),
            "baseline_dwell_s": end.get("baseline_dwell_s"),
            "steering_feedback_end_median": steer_feedback,
            "steering_command_end_median": steer_command,
            "steering_feedback_abs_error_rad": end.get(
                "steering_feedback_abs_error_rad"),
            "steering_feedback_tracking_ok": end.get(
                "steering_feedback_tracking_ok") is True,
            "feedback_response_t50_s": None,
            "initial_state": initial,
            "baseline_state": end.get("baseline_state"),
            "end_state": end.get("end_state"),
            "response_observation_s": end.get("response_observation_s"),
            "max_speed_mps": end.get("measured_speed_max_mps"),
            "max_displacement_from_spawn_m": end.get(
                "maximum_displacement_from_spawn_m"),
            "odom_rate_hz": end.get("odom_rate_hz_during_pair"),
            "odom_gap_p95_ms": end.get("odom_gap_p95_ms_during_pair"),
            "odom_gap_max_ms": end.get("odom_gap_max_ms_during_pair"),
            "condition_complete": end.get("valid") is True,
            "throttle_feedback_tracking_ok": bool(
                start_error is not None and end_error is not None
                and start_error <= THROTTLE_TRACKING_TOLERANCE
                and end_error <= THROTTLE_TRACKING_TOLERANCE),
            "command_tracking_ok": command_tracking_ok,
            "quality_failures": end.get("quality_failures", []),
        }
        row["usable_for_response_fit"] = bool(
            row["condition_complete"]
            and row["command_tracking_ok"]
            and row["throttle_feedback_tracking_ok"]
            and row["steering_feedback_tracking_ok"]
            and end.get("odom_stream_ok") is True)
        if stimulus_ns is not None and start_feedback is not None \
                and end_feedback is not None:
            delta = end_feedback - start_feedback
            threshold = start_feedback + 0.5 * delta
            first = bisect.bisect_left(throttle_times, stimulus_ns)
            last = bisect.bisect_right(throttle_times, end_ns)
            for receipt_ns, value in zip(throttle_times[first:last],
                                         throttle_values[first:last]):
                crossed = value >= threshold if delta > 0.0 else value <= threshold
                if delta != 0.0 and crossed:
                    row["feedback_response_t50_s"] = (
                        receipt_ns - stimulus_ns) / 1e9
                    break
        phases.append(row)

    steering_angles = [float(value) for value in
                       experiment_end.get("steering_angles_rad", [])]
    design = experiment_end.get("design", {})
    scheduled = design.get("scheduled_conditions") if design else None
    if scheduled is not None:
        if not isinstance(scheduled, list):
            raise ValueError("scheduled_conditions must be a list")
        expected = {
            (int(item["steering_key_1e4_rad"]),
             int(item["throttle_start_percent"]),
             int(item["throttle_end_percent"]),
             int(item["replicate_index"]))
            for item in scheduled
        }
        if len(expected) != len(scheduled):
            raise ValueError("scheduled_conditions contains duplicate keys")
    elif design:
        target_step = int(design["target_step_percent"])
        baseline_step = int(design["baseline_step_percent"])
        deltas = [int(value) for value in design["step_deltas_percent"]]
        repeat_count = int(design["repeat_count"])
        transitions = {
            (0, end) for end in range(target_step, 101, target_step)
        }
        for start in range(baseline_step, 100, baseline_step):
            transitions.update(
                (start, start + delta) for delta in deltas
                if start + delta <= 100)
            transitions.add((start, 100))
    else:
        # Backward-compatible audit for the original exhaustive 1% sweep.
        repeat_count = 1
        transitions = {
            (start, end) for start in range(100) for end in range(start + 1, 101)
        }
    expected = {
        (round(angle * 10_000), start, end, replicate)
        for angle in steering_angles
        for start, end in transitions
        for replicate in range(1, repeat_count + 1)
    }
    missing_pairs = expected - seen_pairs
    unexpected_pairs = seen_pairs - expected
    replicate_groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in phases:
        group_key = (
            round(float(row["steering_command_rad"]) * 10_000),
            round(float(row["throttle_start_norm"]) * 100),
            round(float(row["throttle_end_norm"]) * 100),
        )
        replicate_groups[group_key].append(row)

    replicate_consistency = []
    for (steering_millirad, start_percent, end_percent), rows in sorted(
            replicate_groups.items()):
        measures: dict[str, dict[str, float | None]] = {}
        for field in ("feedback_response_t50_s", "response_speed_change_mps",
                      "response_observation_s", "max_speed_mps"):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            measures[field] = {
                "mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        replicate_consistency.append({
            "steering_command_rad": steering_millirad / 10_000.0,
            "throttle_start_percent": start_percent,
            "throttle_end_percent": end_percent,
            "replicates_observed": len(rows),
            "metrics": measures,
        })
    recovered = [event for _, event in reset_events
                 if event.get("event") == "sim_reset_recovered"]
    reset_starts = [event for _, event in reset_events
                    if event.get("event") == "sim_reset_start"]
    reset_errors = [float(event["position_error_m"]) for event in recovered
                    if event.get("position_error_m") is not None]
    spawn_spread = None
    spawn_start = next((event.get("position_xy") for _, event in reset_events
                        if event.get("event") == "sim_reset_recovered"
                        and event.get("position_xy") is not None), None)
    if spawn_start is None:
        spawn_start = next((event.get("position_xy") for _, event in reset_events
                            if event.get("event") == "sim_reset_start"
                            and event.get("position_xy") is not None), None)
    if initial_positions and all(math.isfinite(x) and math.isfinite(y)
                                 for x, y in initial_positions) and spawn_start:
        ref_x, ref_y = map(float, spawn_start)
        spawn_spread = max(math.hypot(x - ref_x, y - ref_y)
                           for x, y in initial_positions)
    collision_count = max((int(value) for value in collision_values), default=None)
    timing_fault_count = sum(bool(value) for value in timing_values)
    all_streams_ok = all(item["passes_40hz_gate"]
                         for item in stream_stats.values())
    completed = len(phases) == len(expected) and not missing_pairs and not unexpected_pairs
    no_collision = collision_count == 0
    reset_complete = (
        experiment_end.get("reset_recovered") is True
        and len(recovered) == int(experiment_end.get("resets_requested", -1))
        and (not completed or len(recovered) == len(phases) + 1))
    passed = (
        experiment_end.get("aborted") is False
        and completed and not duplicate_pairs and no_collision
        and timing_fault_count == 0 and reset_complete and all_streams_ok
        and spawn_spread is not None and spawn_spread <= 0.25
    )
    summary = {
        "profile": "throttle_transition_surface",
        "bag": str(bag.resolve()),
        "passed_integrity_gates": passed,
        "experiment_end": experiment_end,
        "design": design or {"name": "legacy_exhaustive_1pct"},
        "condition_counts": {
            "expected": len(expected), "phase_starts": len(starts),
            "phase_ends": len(ends), "analyzed": len(phases),
            "complete": sum(row["condition_complete"] for row in phases),
            "incomplete": sum(not row["condition_complete"] for row in phases),
            "duplicate": len(duplicate_pairs), "missing": len(missing_pairs),
            "unexpected": len(unexpected_pairs),
        },
        "reset_audit": {
            "reset_requests": len(reset_starts),
            "reset_recoveries": len(recovered),
            "recoveries_expected_for_complete_sweep": len(phases) + 1,
            "position_error_max_m": max(reset_errors) if reset_errors else None,
            "max_condition_start_error_from_initial_spawn_m": spawn_spread,
            "no_distance_stop_condition": True,
        },
        "safety": {
            "max_collision_count": collision_count,
            "timing_fault_true_messages": timing_fault_count,
            "experiment_aborted": experiment_end.get("aborted"),
        },
        "active_streams": stream_stats,
        "response_summary": {
            "response_observation_time_median_s": (
                statistics.median(response_times) if response_times else None),
            "response_observation_time_p95_s": _percentile(response_times, 0.95),
            "response_speed_change_median_mps": (
                statistics.median(speed_changes) if speed_changes else None),
            "response_speed_change_p95_mps": _percentile(speed_changes, 0.95),
            "start_throttle_feedback_abs_error_p95": _percentile(
                feedback_start_errors, 0.95),
            "end_throttle_feedback_abs_error_p95": _percentile(
                feedback_end_errors, 0.95),
            "conditions_with_throttle_feedback_tracking": sum(
                row["throttle_feedback_tracking_ok"] for row in phases),
            "conditions_with_steering_feedback_tracking": sum(
                row["steering_feedback_tracking_ok"] for row in phases),
            "conditions_with_command_tracking": sum(
                row["command_tracking_ok"] for row in phases),
            "conditions_usable_for_response_fit": sum(
                row["usable_for_response_fit"] for row in phases),
            "throttle_feedback_tracking_tolerance_norm":
                THROTTLE_TRACKING_TOLERANCE,
            "command_tracking_tolerance_norm": COMMAND_TRACKING_TOLERANCE,
            "max_measured_speed_mps": max(max_speeds) if max_speeds else None,
            "max_measured_displacement_m": (
                max(max_displacements) if max_displacements else None),
        },
        "replicate_consistency": {
            "unique_conditions_observed": len(replicate_consistency),
            "conditions_with_all_replicates": sum(
                len(rows) == repeat_count for rows in replicate_groups.values()),
            "expected_replicates_per_condition": repeat_count,
            "conditions": replicate_consistency,
        },
        "conditions": phases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="closed rosbag2 run_0.db3")
    parser.add_argument("--output", type=Path,
                        help="summary path (default: beside the bag)")
    args = parser.parse_args()
    output = args.output or args.bag.parent.parent / "throttle_transition_analysis.json"
    try:
        summary = analyze(args.bag, output)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "passed_integrity_gates": summary["passed_integrity_gates"],
        "condition_counts": summary["condition_counts"],
        "reset_audit": summary["reset_audit"],
        "safety": summary["safety"],
        "active_streams": summary["active_streams"],
        "response_summary": summary["response_summary"],
        "output": str(output),
    }, indent=2, sort_keys=True))
    return 0 if summary["passed_integrity_gates"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
