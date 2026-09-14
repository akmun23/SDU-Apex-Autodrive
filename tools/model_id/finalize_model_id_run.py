#!/usr/bin/env python3
"""Finalize a model-ID event log after an unclean ROS launch shutdown.

The live recorder writes events.csv incrementally.  This tool only reconstructs
the derived CSV partitions and timing report from that canonical event stream;
it never adds simulator data or changes the runtime experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


EVENT_FIELDS = (
    "event_index", "arrival_monotonic_ns", "topic", "message_type",
    "header_stamp_ns", "simulation_time_s", "payload_json",
)
BRIDGE_TOPIC = "/autodrive/roboracer_1/bridge_packet_timing"
FAULT_TOPIC = "/autodrive/roboracer_1/bridge_timing_fault"
FAULT_DETAIL_TOPIC = "/autodrive/roboracer_1/bridge_timing_fault_detail"
EVENT_FILE_TOPICS = {
    "bridge_timing_fault.csv": (FAULT_TOPIC,),
    "bridge_timing_fault_detail.csv": (FAULT_DETAIL_TOPIC,),
    "imu.csv": ("/autodrive/roboracer_1/imu",),
    "encoders.csv": (
        "/autodrive/roboracer_1/left_encoder",
        "/autodrive/roboracer_1/right_encoder",
    ),
    "actuator_feedback.csv": (
        "/autodrive/roboracer_1/steering",
        "/autodrive/roboracer_1/throttle",
    ),
    "runtime_state.csv": (
        "/odom", "/ekf_odom", "/amcl_pose", "/current_map_pose",
    ),
}
TRUTH_KEYS = (
    "simulator_position_x", "simulator_position_y", "simulator_position_z",
    "simulator_orientation_quaternion_x",
    "simulator_orientation_quaternion_y",
    "simulator_orientation_quaternion_z",
    "simulator_orientation_quaternion_w",
    "simulator_orientation_euler_x", "simulator_orientation_euler_y",
    "simulator_orientation_euler_z", "simulator_linear_velocity_x",
    "simulator_linear_velocity_y", "simulator_linear_velocity_z",
    "simulator_angular_velocity_x", "simulator_angular_velocity_y",
    "simulator_angular_velocity_z", "simulator_linear_acceleration_x",
    "simulator_linear_acceleration_y", "simulator_linear_acceleration_z",
    "simulator_encoder_angles_left", "simulator_encoder_angles_right",
)


def _write(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _timing_rows(run_id: str, events: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("topic") != BRIDGE_TOPIC:
            continue
        try:
            payload = json.loads(event["payload_json"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        row: dict[str, Any] = {
            "run_id": run_id,
            "event_index": event.get("event_index"),
            "bridge_arrival_monotonic_ns": payload.get(
                "bridge_arrival_monotonic_ns", event.get("arrival_monotonic_ns")),
            "request_monotonic_ns": payload.get("request_monotonic_ns"),
            "request_sequence": payload.get("request_sequence"),
            "packet_sequence": payload.get("packet_sequence"),
            "telemetry_sequence": payload.get("telemetry_sequence"),
            "simulation_time_s": payload.get("simulation_time_s"),
            "simulation_physics_step": payload.get("simulation_physics_step"),
            "simulation_render_frame": payload.get("simulation_render_frame"),
            "sent_throttle_norm": payload.get("sent_throttle_norm"),
            "sent_steering_norm": payload.get("sent_steering_norm"),
            "sent_reset": payload.get("sent_reset"),
            "applied_command_sequence": payload.get("applied_command_sequence"),
            "commanded_throttle_norm": payload.get("commanded_throttle_norm"),
            "commanded_steering_norm": payload.get("commanded_steering_norm"),
            "applied_throttle_norm": payload.get("applied_throttle_norm"),
            "applied_steering_norm": payload.get("applied_steering_norm"),
            "response_received": True,
            "associated_packet_sequence": payload.get("packet_sequence"),
        }
        row.update({key: value for key, value in payload.items()
                    if key.startswith("simulator_")})
        rows.append(row)
    return rows


def _summary(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _report(events: list[dict[str, str]], timing: list[dict[str, Any]]) -> dict[str, Any]:
    def numbers(key: str) -> list[float]:
        return [float(row[key]) for row in timing if row.get(key) not in (None, "")]

    source = numbers("simulation_time_s")
    physics = numbers("simulation_physics_step")
    requests = [int(row["request_sequence"]) for row in timing
                if row.get("request_sequence") not in (None, "")]
    telemetry = [int(row["telemetry_sequence"]) for row in timing
                 if row.get("telemetry_sequence") not in (None, "")]
    source_dt = [b - a for a, b in zip(source, source[1:])]
    physics_dt = [b - a for a, b in zip(physics, physics[1:])]
    applied_lag = [int(row["request_sequence"]) - int(row["applied_command_sequence"])
                   for row in timing
                   if row.get("request_sequence") not in (None, "") and
                   row.get("applied_command_sequence") not in (None, "")]
    return {
        "schema_version": 1,
        "finalized_from_events": True,
        "bridge_packet_count": len(timing),
        "event_count": len(events),
        "source_time_s": _summary(source_dt),
        "physics_step_delta": _summary(physics_dt),
        "request_sequence": {
            "count": len(requests),
            "first": requests[0] if requests else None,
            "last": requests[-1] if requests else None,
            "gaps": sum(b - a != 1 for a, b in zip(requests, requests[1:])),
        },
        "telemetry_sequence": {
            "count": len(telemetry),
            "first": telemetry[0] if telemetry else None,
            "last": telemetry[-1] if telemetry else None,
            "gaps": sum(b - a != 1 for a, b in zip(telemetry, telemetry[1:])),
        },
        "request_to_applied_command_lag": _summary(applied_lag),
        "source_duplicate_or_reverse_count": sum(value <= 0.0 for value in source_dt),
        "source_gaps_over_30ms": sum(value > 0.030 for value in source_dt),
        "physics_step_present_count": len(physics),
        "applied_command_present_count": sum(
            row.get("applied_command_sequence") not in (None, "") for row in timing),
    }


def finalize(run_dir: Path, duration_sec: float) -> dict[str, Any]:
    with (run_dir / "events.csv").open(newline="", encoding="utf-8") as stream:
        events = list(csv.DictReader(stream))
    timing = _timing_rows(run_dir.name, events)
    base_fields = (
        "run_id", "event_index", "request_sequence", "request_monotonic_ns",
        "packet_sequence", "associated_packet_sequence", "response_received",
        "bridge_arrival_monotonic_ns", "telemetry_sequence", "simulation_time_s",
        "simulation_physics_step", "simulation_render_frame", "sent_throttle_norm",
        "sent_steering_norm", "sent_reset", "commanded_throttle_norm",
        "commanded_steering_norm", "applied_command_sequence",
        "applied_throttle_norm", "applied_steering_norm",
    )
    extras = sorted({key for row in timing for key in row if key.startswith("simulator_")})
    _write(run_dir / "bridge_requests.csv", timing, (*base_fields, *extras))
    _write(run_dir / "simulator_packets.csv", timing, (*base_fields, *extras))
    for file_name, topics in EVENT_FILE_TOPICS.items():
        _write(run_dir / file_name,
               [row for row in events if row.get("topic") in topics], EVENT_FIELDS)
    selected = [row for row in timing if any(row.get(key) not in (None, "")
                                             for key in TRUTH_KEYS)]
    if selected:
        fields = (
            "run_id", "packet_sequence", "request_sequence", "telemetry_sequence",
            "simulation_time_s", "simulation_physics_step", "simulation_render_frame",
            *TRUTH_KEYS,
        )
        _write(run_dir / "gt_odom.csv", selected, fields)
    else:
        _write(run_dir / "gt_odom.csv",
               [{"status": "not_recorded", "reason": "no simulator packet fields present"}],
               ("status", "reason"))
    _write(run_dir / "experiment_schedule.csv", [{
        "phase": "configured_capture_window", "mode": "track_validation",
        "duration_sec": duration_sec, "source": "model_identification.launch.py",
        "status": "offline_finalized",
    }], ("phase", "mode", "duration_sec", "source", "status"))
    report = _report(events, timing)
    (run_dir / "timing_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    args = parser.parse_args()
    if not (args.run_dir / "events.csv").is_file():
        raise SystemExit(f"missing events.csv: {args.run_dir}")
    report = finalize(args.run_dir, args.duration_sec)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
