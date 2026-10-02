#!/usr/bin/env python3
"""Strictly admit one fresh, validation-only six-lap practice capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning import (
    evaluate_sensor_observer_practice as practice,
    prepare_dataset,
)


EXPECTED_LAPS = 6
MAX_RACE_SPEED_MPS = 12.0
PACKET_MATCH_MIN = 0.999
MAX_SENSOR_WARMUP_S = 0.125
RESET_TOPIC = prepare_dataset.RESET_COMMAND_TOPIC


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(array)),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
            "max": float(np.max(array))}


def _active_packet_alignment(path: Path, start_ns: int,
                             end_ns: int) -> dict[str, Any]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = body.analysis._topic_map(connection)
        packet_topic = body.analysis.PACKET_TIMING
        odom_topic = body.analysis.ODOM
        if packet_topic not in topics or odom_topic not in topics:
            raise ValueError("active packet audit requires packet timing and odometry")
        packet_by_stamp: dict[int, int] = {}
        for _, message in body.analysis._messages(connection, topics, packet_topic):
            try:
                item = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            stamp = item.get("bridge_receive_ros_stamp_ns")
            sequence = item.get("packet_sequence")
            if isinstance(stamp, int) and isinstance(sequence, int):
                packet_by_stamp[stamp] = sequence

        rows: list[tuple[int, int | None, float]] = []
        for receipt_ns, message in body.analysis._messages(
                connection, topics, odom_topic):
            if not start_ns <= receipt_ns <= end_ns:
                continue
            source_ns = body.analysis._stamp_ns(message.header.stamp)
            packet = packet_by_stamp.get(source_ns)
            twist = message.twist.twist
            speed = math.hypot(float(twist.linear.x), float(twist.linear.y))
            rows.append((receipt_ns, packet, speed))
        return _classify_active_packet_rows(rows, start_ns)
    finally:
        connection.close()


def _classify_active_packet_rows(
        rows: list[tuple[int, int | None, float]],
        lap_zero_start_ns: int) -> dict[str, Any]:
    first_matched = next((index for index, (_, packet, _) in enumerate(rows)
                          if packet is not None), None)
    if first_matched is None:
        raise ValueError("no packet-aligned odometry in active interval")
    startup_rows = rows[:first_matched]
    scored_rows = rows[first_matched:]
    if any(packet is None for _, packet, _ in scored_rows):
        raise ValueError("unmatched odometry occurred after packet stream start")
    packet_ids = [int(packet) for _, packet, _ in scored_rows]
    if any(right != left + 1
           for left, right in zip(packet_ids, packet_ids[1:])):
        raise ValueError("packet sequence gap/duplicate after stream start")
    matched = len(scored_rows)
    total = len(scored_rows)
    first_receipt_ns = scored_rows[0][0]
    return {
            "matched_samples": matched,
            "total_samples": total,
            "fraction": matched / total if total else 0.0,
            "minimum_fraction": PACKET_MATCH_MIN,
            "first_matched_odom_receipt_ns": first_receipt_ns,
            "first_last_packet_id": [packet_ids[0], packet_ids[-1]],
            "pre_packet_startup": {
                "samples_excluded_from_scoring": len(startup_rows),
                "duration_after_lap_zero_s": (
                    (first_receipt_ns - lap_zero_start_ns) / 1e9),
                "body_speed_mps": _percentiles(
                    [speed for _, _, speed in startup_rows]),
            },
        }


def _sensor_ready_index(valid: list[bool], receipt_ns: list[int]) -> int:
    if len(valid) != len(receipt_ns) or not valid:
        raise ValueError("sensor readiness requires aligned nonempty samples")
    first_ready = next((index for index, ready in enumerate(valid) if ready), None)
    if first_ready is None:
        raise ValueError("no complete sensor-only sample in active interval")
    warmup_s = (receipt_ns[first_ready] - receipt_ns[0]) / 1e9
    if warmup_s > MAX_SENSOR_WARMUP_S:
        raise ValueError(
            f"required sensor inputs became complete only after {warmup_s:.3f} s")
    if not all(valid[first_ready:]):
        raise ValueError("required sensor input is missing after sensor startup")
    return first_ready


def _validate(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": path.parents[1].name,
        "bag": str(path.resolve()),
        "bag_sha256": _sha256(path),
        "role": "validation",
        "training_allowed": False,
        "expected_laps": EXPECTED_LAPS,
        "race_speed_domain_mps": [0.0, MAX_RACE_SPEED_MPS],
        "admitted": False,
        "failures": [],
    }

    receipt = practice._receipt_gates(
        path, expected_laps=EXPECTED_LAPS, allow_post_run_disconnect=False)
    report["receipt_gates"] = receipt
    active_start = int(receipt["active_start_receipt_ns"])
    active_end = int(receipt["active_end_receipt_ns"])

    capture = body.load_capture(path, include_nonvalid_phases=True)
    report["parser_status"] = {
        "has_phase_markers": capture.has_phase_markers,
        "final_lap_count": capture.final_lap_count,
        "parser_aborted": capture.aborted,
        "parser_reason": capture.reason,
    }
    if capture.has_phase_markers:
        report["failures"].append("expected one unphased whole-run capture")
    if capture.final_lap_count != EXPECTED_LAPS:
        report["failures"].append("parser final lap count differs from six")
    # The generic phase parser intentionally considers any unphased capture
    # shorter than its legacy 12-lap default aborted. The independent exact
    # 0..6 receipt gate above is authoritative for this validation-only path.
    expected_legacy_reason = f"unphased whole run; final lap count={EXPECTED_LAPS}"
    if capture.aborted and capture.reason != expected_legacy_reason:
        report["failures"].append("capture has a non-parser abort condition")
    if capture.collision_count_start != 0 or capture.collision_count_end != 0:
        report["failures"].append("collision count changed or was nonzero")
    if capture.timing_faults:
        report["failures"].append("capture contains bridge timing faults")

    active_alignment = _active_packet_alignment(path, active_start, active_end)
    scored_start = int(active_alignment["first_matched_odom_receipt_ns"])
    report["packet_alignment"] = {
        "active_interval": active_alignment,
        "whole_bag_diagnostic": {
            "matched_samples": capture.packet_sequence_matched_samples,
            "total_samples": capture.packet_sequence_total_samples,
        },
    }
    if active_alignment["fraction"] < PACKET_MATCH_MIN:
        report["failures"].append(
            "active packet/odometry alignment below 99.9 percent")

    reset_topic_present, reset_starts = prepare_dataset._reset_epoch_starts(path)
    resets_during_active = [stamp for stamp in reset_starts
                            if active_start <= stamp <= active_end]
    report["reset_audit"] = {
        "topic": RESET_TOPIC,
        "topic_present_in_bag": reset_topic_present,
        "reset_edges_in_active_interval": resets_during_active,
        "no_reset_evidence": "exact lap transitions 0..6 and contiguous simulator packet IDs",
    }
    if resets_during_active:
        report["failures"].append("reset command occurred during scored interval")

    active = sorted(
        (sample for sequence in capture.sequences for sample in sequence
         if scored_start <= sample.receipt_ns <= active_end),
        key=lambda sample: sample.packet_sequence,
    )
    if not active:
        report["failures"].append("no parsed motion samples in the active interval")
        report["admitted"] = False
        return report

    initial_sensor_valid = [prepare_dataset._sensor_frame(sample) is not None
                            for sample in active]
    sensor_start = _sensor_ready_index(
        initial_sensor_valid, [sample.receipt_ns for sample in active])
    sensor_warmup = active[:sensor_start]
    active = active[sensor_start:]
    sensor_valid = initial_sensor_valid[sensor_start:]
    scored_start = active[0].receipt_ns

    packet_ids = [sample.packet_sequence for sample in active]
    continuity = (all(packet >= 0 for packet in packet_ids)
                  and all(right == left + 1
                          for left, right in zip(packet_ids, packet_ids[1:])))
    report["active_interval"] = {
        "lap_zero_transition_receipt_ns": active_start,
        "start_receipt_ns": scored_start,
        "end_receipt_ns": active_end,
        "duration_s": (active_end - scored_start) / 1e9,
        "aligned_samples": len(active),
        "first_last_packet_id": [packet_ids[0], packet_ids[-1]],
        "packet_ids_contiguous": continuity,
    }
    if not continuity:
        report["failures"].append("active interval has a packet gap or duplicate")

    truth_valid = [
        sample.simulator_pose_xyyaw is not None
        and sample.simulator_rigid_state is not None
        and sample.simulator_linear_acceleration is not None
        and np.isfinite(sample.simulator_pose_xyyaw).all()
        and np.isfinite(sample.simulator_rigid_state).all()
        and np.isfinite(sample.simulator_linear_acceleration).all()
        and sample.pose_xyyaw is not None
        and np.isfinite(sample.pose_xyyaw).all()
        for sample in active
    ]
    if not all(sensor_valid):
        report["failures"].append("one or more required legal sensor inputs are missing")
    if not all(truth_valid):
        report["failures"].append("one or more simulator/odom pose labels are missing")

    speed = [float(math.hypot(sample.state[0], sample.state[1]))
             for sample in active]
    if not np.isfinite(speed).all():
        report["failures"].append("non-finite body-speed samples")
    if max(speed) > MAX_RACE_SPEED_MPS + 1e-6:
        report["failures"].append("active interval exceeds the 12 m/s race domain")
    lap_samples: dict[str, int] = {}
    for sample in active:
        key = str(sample.lap_count)
        lap_samples[key] = lap_samples.get(key, 0) + 1
    if any(lap_samples.get(str(lap), 0) == 0 for lap in range(EXPECTED_LAPS)):
        report["failures"].append("one or more completed-lap intervals have no samples")

    steering = [float(sample.actuators[0]) for sample in active]
    throttle = [float(sample.actuators[1]) for sample in active]
    report["coverage"] = {
        "body_speed_mps": _percentiles(speed),
        "steering_feedback_rad": _percentiles(steering),
        "throttle_feedback": _percentiles(throttle),
        "samples_by_lap_count": lap_samples,
        "sensor_complete_fraction": float(np.mean(sensor_valid)),
        "simulator_and_odom_label_complete_fraction": float(np.mean(truth_valid)),
        "encoder_history_warmup": {
            "samples_excluded_before_sensor_ready": len(sensor_warmup),
            "duration_s": (
                (active[0].receipt_ns - sensor_warmup[0].receipt_ns) / 1e9
                if sensor_warmup else 0.0),
            "maximum_allowed_s": MAX_SENSOR_WARMUP_S,
            "reason": "rear encoder speed uses a causal 100 ms angle window",
        },
        "nonoverlapping_2s_windows": sum(
            sum(scored_start <= sample.receipt_ns <= active_end
                for sample in sequence) // 80
            for sequence in capture.sequences),
    }
    report["admitted"] = not report["failures"]
    return report


def validate_capture(path: Path, output: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"bag does not exist: {path}")
    output = output or path.parents[1] / "practice_capture_validation.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {output}")

    try:
        report = _validate(path)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "run_id": path.parents[1].name,
            "bag": str(path),
            "bag_sha256": _sha256(path),
            "role": "validation",
            "training_allowed": False,
            "expected_laps": EXPECTED_LAPS,
            "admitted": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="closed ROS 2 sqlite bag file")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_capture(args.bag, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
