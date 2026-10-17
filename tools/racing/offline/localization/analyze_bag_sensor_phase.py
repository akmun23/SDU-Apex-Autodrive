#!/usr/bin/env python3
"""Audit source stamps and receive ordering for the odometry sensor bag."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


ROOT = Path(__file__).resolve().parents[4]
TOPICS = {
    "imu": "/autodrive/roboracer_1/imu",
    "truth": "/autodrive/roboracer_1/odom",
    "production_odom": "/odom",
    "left_encoder": "/autodrive/roboracer_1/left_encoder",
    "right_encoder": "/autodrive/roboracer_1/right_encoder",
    "steering_feedback": "/autodrive/roboracer_1/steering",
    "throttle_feedback": "/autodrive/roboracer_1/throttle",
    "steering_command": "/autodrive/roboracer_1/steering_command",
    "throttle_command": "/autodrive/roboracer_1/throttle_command",
}
LABEL_FOR_TOPIC = {topic: label for label, topic in TOPICS.items()}


def header_stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def distribution(values_ns: list[int]) -> dict[str, float | int] | None:
    if len(values_ns) < 2:
        return None
    gaps_ms = np.diff(np.asarray(values_ns, dtype=np.int64)) / 1.0e6
    return {
        "samples": int(len(values_ns)),
        "gap_median_ms": float(np.median(gaps_ms)),
        "gap_p95_ms": float(np.percentile(gaps_ms, 95)),
        "gap_max_ms": float(np.max(gaps_ms)),
        "gap_min_ms": float(np.min(gaps_ms)),
    }


def signed_distribution(values_ms: list[float]) -> dict[str, float | int] | None:
    if not values_ms:
        return None
    values = np.asarray(values_ms, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    absolute = np.abs(values)
    return {
        "samples": int(values.size),
        "median_ms": float(np.median(values)),
        "p05_ms": float(np.percentile(values, 5)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_abs_ms": float(np.max(absolute)),
        "exact_within_1us_fraction": float(np.mean(absolute <= 0.001)),
    }


def analyze(bag: Path, output: Path, nominal_period_ms: float) -> dict[str, Any]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {entry.name: get_message(entry.type)
             for entry in reader.get_all_topics_and_types()}
    absent = sorted(set(TOPICS.values()) - set(types))
    if absent:
        raise ValueError(f"bag lacks required timing topics: {absent}")

    source_ns: dict[str, list[int]] = defaultdict(list)
    receive_ns: dict[str, list[int]] = defaultdict(list)
    latest_source: dict[str, int] = {}
    latest_receive: dict[str, int] = {}
    event_source_skew: dict[str, list[float]] = defaultdict(list)
    event_receive_skew: dict[str, list[float]] = defaultdict(list)
    event_latest_source_age: dict[str, list[float]] = defaultdict(list)
    events = 0

    while reader.has_next():
        topic, serialized, received = reader.read_next()
        label = LABEL_FOR_TOPIC.get(topic)
        if label is None:
            continue
        received = int(received)
        message = deserialize_message(serialized, types[topic])
        source = header_stamp_ns(message)
        receive_ns[label].append(received)
        latest_receive[label] = received
        if source is not None:
            source_ns[label].append(source)
            latest_source[label] = source

        if label not in {"imu", "truth", "production_odom"} or source is None:
            continue
        events += 1
        for other in ("imu", "truth", "left_encoder", "right_encoder"):
            if other == label or other not in latest_receive:
                continue
            event_source_skew[f"{label}_minus_{other}"].append(
                (source - latest_source[other]) / 1.0e6
                if other in latest_source else float("nan"))
            event_receive_skew[f"{label}_receive_minus_{other}"].append(
                (received - latest_receive[other]) / 1.0e6)
            if other in latest_source:
                event_latest_source_age[f"{label}_latest_{other}_age_at_receive"].append(
                    (source - latest_source[other]) / 1.0e6)

    topic_summary: dict[str, Any] = {}
    for label in TOPICS:
        source_values = source_ns[label]
        receive_values = receive_ns[label]
        topic_summary[label] = {
            "source_header_available": bool(source_values),
            "source_cadence": distribution(source_values),
            "receive_cadence": distribution(receive_values),
            "count": len(receive_values),
        }
        if source_values and len(source_values) == len(receive_values):
            # Compare only within a topic: absolute source and bag receive clocks
            # can have different epochs, so the absolute offset is not meaningful.
            source_gaps = np.diff(np.asarray(source_values, dtype=np.int64))
            receive_gaps = np.diff(np.asarray(receive_values, dtype=np.int64))
            topic_summary[label]["source_receive_gap_jitter_ms"] = (
                float(np.percentile(np.abs(source_gaps - receive_gaps), 95) / 1e6)
                if len(source_gaps) else 0.0)

    result = {
        "bag": str(bag.resolve()),
        "nominal_period_ms_for_packet_equivalence": nominal_period_ms,
        "event_count": events,
        "topic_timing": topic_summary,
        "same_receive_order_latest_sample_skew": {
            key: signed_distribution(values)
            for key, values in event_source_skew.items()
        },
        "receive_order_delta_ms": {
            key: signed_distribution(values)
            for key, values in event_receive_skew.items()
        },
        "latest_header_age_at_event_source_time_ms": {
            key: signed_distribution(values)
            for key, values in event_latest_source_age.items()
        },
        "interpretation": (
            "Header stamps reveal source-time skew only for stamped topics. "
            "Float feedback topics have receive timestamps only. Receive order "
            "is reported directly; this audit does not shift or rewrite samples."),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-period-ms", type=float, default=25.0)
    args = parser.parse_args()
    analyze(args.bag.resolve(), args.output.resolve(), args.nominal_period_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
