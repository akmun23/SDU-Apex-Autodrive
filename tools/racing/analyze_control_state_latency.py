#!/usr/bin/env python3
"""Summarize bridge cadence and MPC state/solve latency from a closed bag."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - ROS image dependency
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc


LIDAR = "/autodrive/roboracer_1/lidar"
BRIDGE_TIMING = "/autodrive/roboracer_1/bridge_packet_timing"
MPC_DIAGNOSTICS = "/mpc/diagnostics"


def percentile(values: list[float], probability: float) -> float | None:
    selected = sorted(value for value in values if math.isfinite(value))
    if not selected:
        return None
    position = (len(selected) - 1) * probability / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    return selected[lower] + (selected[upper] - selected[lower]) * (position - lower)


def numeric_summary(values: list[float], unit: str) -> dict[str, Any]:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(selected),
        "unit": unit,
        "p50": percentile(selected, 50),
        "p95": percentile(selected, 95),
        "p99": percentile(selected, 99),
        "max": max(selected) if selected else None,
    }


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def format_value(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def rows_for(
    connection: sqlite3.Connection,
    topics: dict[str, tuple[int, str]],
    name: str,
) -> list[tuple[int, Any]]:
    if name not in topics:
        return []
    topic_id, type_name = topics[name]
    message_type = get_message(type_name)
    return [
        (int(timestamp), deserialize_message(bytes(payload), message_type))
        for timestamp, payload in connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
            (topic_id,),
        )
    ]


def intervals_seconds(stamps_ns: list[int]) -> list[float]:
    return [
        (later - earlier) / 1e9
        for earlier, later in zip(stamps_ns, stamps_ns[1:])
        if later > earlier
    ]


def analyze(bag: Path, output: Path) -> dict[str, Any]:
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = {
            name: (int(topic_id), type_name)
            for topic_id, name, type_name in connection.execute(
                "SELECT id,name,type FROM topics"
            )
        }
        missing_topics = [
            name for name in (LIDAR, BRIDGE_TIMING, MPC_DIAGNOSTICS)
            if name not in topics
        ]
        lidar = rows_for(connection, topics, LIDAR)
        bridge_rows = rows_for(connection, topics, BRIDGE_TIMING)
        diagnostics = rows_for(connection, topics, MPC_DIAGNOSTICS)

        lidar_receipt = [timestamp for timestamp, _ in lidar]
        lidar_header = [
            int(message.header.stamp.sec) * 1_000_000_000 +
            int(message.header.stamp.nanosec)
            for _, message in lidar
            if int(message.header.stamp.sec) or int(message.header.stamp.nanosec)
        ]
        receipt_dt = intervals_seconds(lidar_receipt)
        header_dt = intervals_seconds(lidar_header)
        lidar_report = {
            "message_count": len(lidar),
            "receipt_rate_hz": (len(lidar_receipt) - 1) / ((lidar_receipt[-1] - lidar_receipt[0]) / 1e9)
            if len(lidar_receipt) > 1 and lidar_receipt[-1] > lidar_receipt[0] else None,
            "header_rate_hz": (len(lidar_header) - 1) / ((lidar_header[-1] - lidar_header[0]) / 1e9)
            if len(lidar_header) > 1 and lidar_header[-1] > lidar_header[0] else None,
            "receipt_interval_ms": numeric_summary([value * 1000 for value in receipt_dt], "ms"),
            "header_interval_ms": numeric_summary([value * 1000 for value in header_dt], "ms"),
        }

        mpc_samples: list[dict[str, Any]] = []
        status_counts: Counter[str] = Counter()
        for receipt_ns, message in diagnostics:
            try:
                data = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            prediction = data.get("control_time_prediction") or {}
            timing = data.get("host_timing_ns") or {}
            callback_ns = timing.get("callback")
            synchronize_ns = timing.get("synchronize")
            publish_ns = data.get("diagnostic_publish_steady_ns")
            pipeline_us = (
                (publish_ns - callback_ns) / 1000.0
                if isinstance(callback_ns, (int, float)) and
                isinstance(publish_ns, (int, float)) and publish_ns >= callback_ns
                else math.nan
            )
            synchronization_us = (
                (synchronize_ns - callback_ns) / 1000.0
                if isinstance(callback_ns, (int, float)) and
                isinstance(synchronize_ns, (int, float)) and synchronize_ns >= callback_ns
                else math.nan
            )
            solver = data.get("solver") or {}
            status = str(data.get("status", "unknown"))
            status_counts[status] += 1
            state = data.get("state") or []
            mpc_samples.append({
                "receipt_time_ns": receipt_ns,
                "status": status,
                "source_dt_ms": as_float(data.get("source_dt_s")) * 1000.0,
                "source_age_ms": as_float(data.get("source_age_s")) * 1000.0,
                "pose_odom_skew_ms": as_float(data.get("pose_odom_skew_s")) * 1000.0,
                "prediction_compute_us": as_float(prediction.get("elapsed_us")),
                "prediction_command_changes": int(prediction.get("command_changes_used", 0)),
                "prediction_command_fallback": bool(prediction.get("command_fallback", False)),
                "prediction_time_fallback": bool(prediction.get("time_fallback", False)),
                "synchronization_us": synchronization_us,
                "pipeline_wall_us": pipeline_us,
                "solver_us": as_float(solver.get("solve_us")),
                "total_rti_us": as_float(data.get("total_rti_us")),
                "solver_iterations": int(solver.get("iterations", 0)),
                "progress_m": as_float(data.get("progress_m")),
                "speed_mps": as_float(state[2]) if len(state) > 2 else math.nan,
                "steering_command_rad": as_float(state[5]) if len(state) > 5 else math.nan,
                "steering_feedback_rad": as_float(state[6]) if len(state) > 6 else math.nan,
            })

        mpc_report: dict[str, Any] = {
            "diagnostic_count": len(mpc_samples),
            "status_counts": dict(status_counts),
        }
        sample_metrics = (
            ("source_dt_ms", "ms"),
            ("source_age_ms", "ms"),
            ("pose_odom_skew_ms", "ms"),
            ("prediction_compute_us", "us"),
            ("synchronization_us", "us"),
            ("pipeline_wall_us", "us"),
            ("solver_us", "us"),
            ("total_rti_us", "us"),
        )
        for field, unit in sample_metrics:
            mpc_report[field] = numeric_summary(
                [float(row[field]) for row in mpc_samples], unit
            )
        mpc_report["prediction_command_fallback_count"] = sum(
            row["prediction_command_fallback"] for row in mpc_samples
        )
        mpc_report["prediction_time_fallback_count"] = sum(
            row["prediction_time_fallback"] for row in mpc_samples
        )
        mpc_report["solver_over_25ms_count"] = sum(
            float(row["solver_us"]) > 25_000.0 for row in mpc_samples
        )

        bridge_samples = []
        for _, message in bridge_rows:
            try:
                data = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            request_ns = data.get("request_monotonic_ns")
            arrival_ns = data.get("bridge_arrival_monotonic_ns")
            if not isinstance(request_ns, (int, float)) or not isinstance(arrival_ns, (int, float)):
                continue
            bridge_samples.append({
                "packet_sequence": int(data.get("packet_sequence", -1)),
                "request_monotonic_ns": int(request_ns),
                "arrival_monotonic_ns": int(arrival_ns),
                "request_response_ms": (arrival_ns - request_ns) / 1e6,
                "steering_update_age_ms": as_float(data.get("steering_command_update_age_ms")),
                "applied_command_sequence": data.get("applied_command_sequence"),
                "simulator_frame": data.get("simulation_frame"),
                "simulator_physics_step": data.get("simulation_physics_step"),
            })
        bridge_arrivals = [row["arrival_monotonic_ns"] for row in bridge_samples]
        bridge_intervals_ms = [value * 1000 for value in intervals_seconds(bridge_arrivals)]
        bridge_report = {
            "packet_count": len(bridge_samples),
            "request_response_ms": numeric_summary(
                [row["request_response_ms"] for row in bridge_samples], "ms"
            ),
            "arrival_interval_ms": numeric_summary(bridge_intervals_ms, "ms"),
            "steering_update_age_ms": numeric_summary(
                [row["steering_update_age_ms"] for row in bridge_samples], "ms"
            ),
            "sequence_gaps": sum(
                max(0, later["packet_sequence"] - earlier["packet_sequence"] - 1)
                for earlier, later in zip(bridge_samples, bridge_samples[1:])
                if later["packet_sequence"] >= earlier["packet_sequence"]
            ),
            "simulator_physics_step_observed_count": sum(
                row["simulator_physics_step"] is not None for row in bridge_samples
            ),
        }

        report = {
            "schema_version": 1,
            "bag": str(bag),
            "missing_topics": missing_topics,
            "lidar": lidar_report,
            "bridge": bridge_report,
            "mpc": mpc_report,
            "interpretation": {
                "sensor_rate_is_header_interval_rate": True,
                "state_extrapolation_tuned": False,
                "source_data_is_read_only": True,
            },
        }
        output.mkdir(parents=True, exist_ok=True)
        summary_path = output / "control_state_latency_summary.json"
        samples_path = output / "latency_samples.csv"
        summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        fields = list(mpc_samples[0]) if mpc_samples else []
        with samples_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(mpc_samples)
        print(f"LiDAR header rate: {format_value(lidar_report['header_rate_hz'], 3)} Hz")
        print(
            "Bridge request→response p50/p95/max: "
            f"{format_value(bridge_report['request_response_ms']['p50'])}/"
            f"{format_value(bridge_report['request_response_ms']['p95'])}/"
            f"{format_value(bridge_report['request_response_ms']['max'])} ms"
        )
        print(
            "MPC source-age p95, prediction p95, solver p95, full pipeline p95: "
            f"{format_value(mpc_report['source_age_ms']['p95'])} ms, "
            f"{format_value(mpc_report['prediction_compute_us']['p95'], 1)} us, "
            f"{format_value(mpc_report['solver_us']['p95'], 1)} us, "
            f"{format_value((mpc_report['pipeline_wall_us']['p95'] or 0.0) / 1000.0)} ms"
        )
        print("MPC statuses:", dict(status_counts))
        print(f"Wrote {summary_path} and {samples_path}")
        return report
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.bag.is_file():
        parser.error(f"bag does not exist: {args.bag}")
    analyze(args.bag, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
