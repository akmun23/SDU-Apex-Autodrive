#!/usr/bin/env python3
"""Shift rosbag receive timestamps onto a sensor header's ROS epoch.

Use only for offline production-node replay when the bag recorder's clock and
the sensor message header clock have a constant offset. The shift is applied
to bag event times (including headerless Float32 commands); serialized message
headers and payloads are not modified.

Example:
  python3 tools/rebase_rosbag2_time_to_sensor_epoch.py INPUT_BAG OUTPUT_BAG
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, SequentialWriter
from rosbag2_py import StorageOptions
from rosidl_runtime_py.utilities import get_message


DEFAULT_REFERENCE_TOPIC = "/autodrive/roboracer_1/odom"


def open_reader(path: Path, storage_id: str) -> SequentialReader:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(path), storage_id=storage_id),
        ConverterOptions("cdr", "cdr"),
    )
    return reader


def header_time_ns(message: object) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_bag", type=Path)
    parser.add_argument("output_bag", type=Path)
    parser.add_argument("--reference-topic", default=DEFAULT_REFERENCE_TOPIC)
    parser.add_argument(
        "--maximum-p95-offset-jitter-ms", type=float, default=2.0,
        help="refuse rebasing if reference bag/header offsets are not nearly constant",
    )
    args = parser.parse_args()

    if not args.input_bag.is_dir():
        parser.error(f"input bag directory does not exist: {args.input_bag}")
    if args.output_bag.exists():
        parser.error(f"output already exists; refusing to overwrite: {args.output_bag}")

    metadata_path = args.input_bag / "metadata.yaml"
    if not metadata_path.is_file():
        parser.error(f"input is not a rosbag2 directory: {args.input_bag}")
    storage_id = "mcap" if any(args.input_bag.glob("*.mcap")) else "sqlite3"
    reference_reader = open_reader(args.input_bag, storage_id)
    topics = reference_reader.get_all_topics_and_types()
    types = {topic.name: topic.type for topic in topics}
    if args.reference_topic not in types:
        parser.error(f"reference topic is absent: {args.reference_topic}")
    reference_type = get_message(types[args.reference_topic])

    offsets_ns: list[int] = []
    while reference_reader.has_next():
        topic_name, serialized, bag_time_ns = reference_reader.read_next()
        if topic_name == args.reference_topic:
            message = deserialize_message(serialized, reference_type)
            offsets_ns.append(bag_time_ns - header_time_ns(message))
    if len(offsets_ns) < 3:
        parser.error("need at least three reference messages to estimate the offset")

    offset_ns = int(statistics.median(offsets_ns))
    deviations_ms = [abs(value - offset_ns) * 1.0e-6 for value in offsets_ns]
    deviations_ms.sort()
    p95_index = min(len(deviations_ms) - 1, int(0.95 * len(deviations_ms)))
    p95_jitter_ms = deviations_ms[p95_index]
    if p95_jitter_ms > args.maximum_p95_offset_jitter_ms:
        parser.error(
            f"bag/header offset is not constant enough: p95 deviation "
            f"{p95_jitter_ms:.3f}ms exceeds "
            f"{args.maximum_p95_offset_jitter_ms:.3f}ms"
        )

    args.output_bag.parent.mkdir(parents=True, exist_ok=True)
    writer = SequentialWriter()
    writer.open(
        StorageOptions(uri=str(args.output_bag), storage_id="mcap"),
        ConverterOptions("cdr", "cdr"),
    )
    for topic in topics:
        writer.create_topic(topic)

    source_reader = open_reader(args.input_bag, storage_id)
    message_count = 0
    while source_reader.has_next():
        topic_name, serialized, bag_time_ns = source_reader.read_next()
        rebased_time_ns = bag_time_ns - offset_ns
        if rebased_time_ns < 0:
            parser.error("rebasing produced a negative event timestamp")
        writer.write(topic_name, serialized, rebased_time_ns)
        message_count += 1

    print(
        f"Rebased {message_count} events by {-offset_ns * 1.0e-9:+.9f}s; "
        f"reference offset p95 deviation={p95_jitter_ms:.3f}ms. "
        "Serialized message headers were preserved."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
