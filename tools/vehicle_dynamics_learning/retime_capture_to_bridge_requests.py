#!/usr/bin/env python3
"""Create a derived ROS bag on matched Bridge-request timestamps.

Only the named sensor/odometry streams are retimed. Samples without a matched
request are omitted from those streams, while bridge timing rows are retained
for matched packets only and assigned a contiguous derived packet index. The
input bag is read-only; all non-target topics are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String


PACKET_TIMING = "/autodrive/roboracer_1/bridge_packet_timing"
RETIMED_TOPICS = {
    "/autodrive/roboracer_1/imu",
    "/autodrive/roboracer_1/left_encoder",
    "/autodrive/roboracer_1/right_encoder",
    "/autodrive/roboracer_1/odom",
    "/odom",
    "/odom/diagnostics",
}


def _ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _set_ns(stamp: Any, value: int) -> None:
    stamp.sec, stamp.nanosec = divmod(int(value), 1_000_000_000)


def _topic_metadata(reader: Any) -> list[Any]:
    return list(reader.get_all_topics_and_types())


def transform(input_bag: Path, output_bag_dir: Path) -> dict[str, Any]:
    input_bag = input_bag.resolve()
    output_bag_dir = output_bag_dir.resolve()
    if input_bag.name != "run_0.db3" or not (input_bag.parent / "metadata.yaml").is_file():
        raise ValueError("--input-bag must be a closed ROS 2 run_0.db3 bag")
    if output_bag_dir.exists():
        raise ValueError(f"refusing to overwrite {output_bag_dir}")

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(
        uri=str(input_bag.parent), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"))
    metadata = _topic_metadata(reader)
    source_metadata = yaml.safe_load(
        (input_bag.parent / "metadata.yaml").read_text(encoding="utf-8"))
    source_qos = {
        item["topic_metadata"]["name"]:
        item["topic_metadata"].get("offered_qos_profiles", "")
        for item in source_metadata["rosbag2_bagfile_information"][
            "topics_with_message_count"]}
    by_name = {topic.name: topic for topic in metadata}
    required = RETIMED_TOPICS | {PACKET_TIMING}
    missing = sorted(required - by_name.keys())
    if missing:
        raise ValueError("input bag lacks required topic(s): " + ", ".join(missing))

    timing_type = get_message(by_name[PACKET_TIMING].type)
    if timing_type is not String:
        raise ValueError("bridge_packet_timing has unexpected message type")
    valid_packets: list[dict[str, Any]] = []
    timing_rows = 0
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic != PACKET_TIMING:
            continue
        timing_rows += 1
        event = json.loads(deserialize_message(serialized, timing_type).data)
        receive = event.get("bridge_receive_ros_stamp_ns")
        arrival = event.get("bridge_arrival_monotonic_ns")
        request = event.get("request_monotonic_ns")
        request_id = event.get("request_sequence")
        if (isinstance(receive, int) and isinstance(arrival, int)
                and isinstance(request, int) and isinstance(request_id, int)
                and request_id >= 0):
            valid_packets.append(event)

    if len(valid_packets) < 1000:
        raise ValueError("too few matched Bridge responses for a practice replay")
    clock_offset_ns = int(np.median([
        int(event["bridge_receive_ros_stamp_ns"])
        - int(event["bridge_arrival_monotonic_ns"])
        for event in valid_packets]))
    receive_to_request: dict[int, tuple[int, int, dict[str, Any]]] = {}
    seen_requests: set[int] = set()
    for index, event in enumerate(valid_packets, start=1):
        receive = int(event["bridge_receive_ros_stamp_ns"])
        request_id = int(event["request_sequence"])
        if receive in receive_to_request:
            raise ValueError(f"duplicate receive timestamp {receive}")
        if request_id in seen_requests:
            raise ValueError(f"duplicate request sequence {request_id}")
        seen_requests.add(request_id)
        explicit = event.get("bridge_request_ros_stamp_ns")
        request_ros = (int(explicit) if isinstance(explicit, int) else
                       int(event["request_monotonic_ns"]) + clock_offset_ns)
        receive_to_request[receive] = (request_ros, index, event)

    request_ids = [int(event["request_sequence"]) for event in valid_packets]
    request_steps = np.diff(np.asarray(request_ids, dtype=np.int64))
    if len(request_steps) and np.any(request_steps != 1):
        raise ValueError("matched request sequence has gaps; packet association is ambiguous")
    request_stamps = np.asarray([
        receive_to_request[int(event["bridge_receive_ros_stamp_ns"])][0]
        for event in valid_packets], dtype=np.int64)
    request_intervals_ms = np.diff(request_stamps).astype(np.float64) / 1e6
    if len(request_intervals_ms) and np.any(request_intervals_ms <= 0.0):
        raise ValueError("request timestamps are not strictly increasing")
    request_id_to_time = {
        int(event["request_sequence"]): receive_to_request[
            int(event["bridge_receive_ros_stamp_ns"])][0]
        for event in valid_packets}
    receive_stamps_sorted = sorted(receive_to_request)

    def packet_for_source_stamp(source_ns: int, *, rounded_float: bool = False):
        packet = receive_to_request.get(source_ns)
        if packet is not None or not rounded_float:
            return packet
        position = bisect.bisect_left(receive_stamps_sorted, source_ns)
        candidates = [receive_stamps_sorted[index]
                      for index in (position - 1, position)
                      if 0 <= index < len(receive_stamps_sorted)]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda stamp: abs(stamp - source_ns))
        return (receive_to_request[nearest]
                if abs(nearest - source_ns) <= 2_000 else None)

    output_bag_dir.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(
        uri=str(output_bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"))
    for topic in metadata:
        writer.create_topic(rosbag2_py.TopicMetadata(
            0, topic.name, topic.type, topic.serialization_format,
            topic.offered_qos_profiles, topic.type_description_hash))

    classes = {topic.name: get_message(topic.type) for topic in metadata}
    copied_messages = 0
    retimed_counts = {topic: 0 for topic in sorted(RETIMED_TOPICS)}
    dropped_counts = {topic: 0 for topic in sorted(RETIMED_TOPICS)}
    timing_output_count = 0
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(
        uri=str(input_bag.parent), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"))
    while reader.has_next():
        topic, serialized, receive_time = reader.read_next()
        if topic == PACKET_TIMING:
            event = json.loads(deserialize_message(serialized, timing_type).data)
            receive = event.get("bridge_receive_ros_stamp_ns")
            matched = receive_to_request.get(receive) if isinstance(receive, int) else None
            if matched is None:
                continue
            request_ros, derived_index, source_event = matched
            event["source_packet_sequence"] = event.get("packet_sequence")
            event["packet_sequence"] = derived_index
            event["bridge_request_ros_stamp_ns"] = request_ros
            message = String()
            message.data = json.dumps(event, separators=(",", ":"))
            writer.write(topic, serialize_message(message), receive_time)
            timing_output_count += 1
            continue
        if topic in RETIMED_TOPICS:
            message = deserialize_message(serialized, classes[topic])
            if topic == "/odom/diagnostics":
                label = message.layout.dim[0].label if message.layout.dim else ""
                if label == "deterministic_odometry_v7" and len(message.data) == 42:
                    source_index = 28
                elif label == "deterministic_odometry_v5" and len(message.data) == 28:
                    source_index = 1
                else:
                    raise ValueError("unexpected /odom/diagnostics schema")
                source_receive = int(round(float(message.data[source_index]) * 1e9))
            else:
                source_receive = _ns(message.header.stamp)
                matched = packet_for_source_stamp(
                    source_receive, rounded_float=(topic == "/odom/diagnostics"))
            if matched is None:
                dropped_counts[topic] += 1
                continue
            if topic == "/odom/diagnostics":
                # Keep source-time diagnostics on the same grid as the
                # associated odometry and truth samples. Index 28 is the
                # packet's source stamp; 29 is its predecessor; 32/33 are
                # packet-assembler source-time watermarks.
                for index in ((1, 28, 32, 33) if source_index == 28 else (1,)):
                    prior_ns = int(round(float(message.data[index]) * 1e9))
                    mapped = packet_for_source_stamp(prior_ns, rounded_float=True)
                    if mapped is not None:
                        message.data[index] = mapped[0] * 1e-9
                if source_index == 28:
                    previous_ns = int(round(float(message.data[29]) * 1e9))
                    previous = packet_for_source_stamp(
                        previous_ns, rounded_float=True)
                    if previous is not None:
                        message.data[29] = previous[0] * 1e-9
                    else:
                        request_id = int(matched[2]["request_sequence"])
                        message.data[29] = request_id_to_time.get(
                            request_id - 1, 0) * 1e-9
            else:
                _set_ns(message.header.stamp, matched[0])
            serialized = serialize_message(message)
            retimed_counts[topic] += 1
        writer.write(topic, serialized, receive_time)
        copied_messages += 1

    # Finalize the SQLite database before validating its presence.
    del writer
    if not (output_bag_dir / "metadata.yaml").is_file():
        raise RuntimeError("rosbag writer did not finalize metadata.yaml")
    # Jazzy serializes QoS enums into a metadata-v9 YAML structure that Humble
    # cannot parse. Restore the source bag's Humble-compatible QoS strings and
    # schema version, while retaining the new bag's actual topic counts/times.
    output_metadata_path = output_bag_dir / "metadata.yaml"
    output_metadata = yaml.safe_load(
        output_metadata_path.read_text(encoding="utf-8"))
    information = output_metadata["rosbag2_bagfile_information"]
    information["version"] = 5
    for item in information["topics_with_message_count"]:
        topic = item["topic_metadata"]
        topic.pop("type_description_hash", None)
        topic["offered_qos_profiles"] = source_qos.get(topic["name"], "")
    with sqlite3.connect(output_bag_dir / "run_0.db3") as connection:
        actual_message_count = int(connection.execute(
            "SELECT COUNT(*) FROM messages").fetchone()[0])
    information["message_count"] = actual_message_count
    if information.get("files"):
        information["files"][0]["message_count"] = actual_message_count
    output_metadata_path.write_text(
        yaml.safe_dump(output_metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    return {
        "input_bag": str(input_bag),
        "input_size_bytes": input_bag.stat().st_size,
        "output_bag_dir": str(output_bag_dir),
        "bridge_timing_rows_input": timing_rows,
        "matched_requests": len(valid_packets),
        "unmatched_or_unassociated_timing_rows_dropped": (
            timing_rows - len(valid_packets)),
        "request_sequence_first_last": [request_ids[0], request_ids[-1]],
        "request_interval_ms_p01_p50_p99_min_max": [
            float(np.quantile(request_intervals_ms, q))
            for q in (0.01, 0.50, 0.99)] + [
                float(np.min(request_intervals_ms)),
                float(np.max(request_intervals_ms))],
        "ros_minus_monotonic_offset_ns": clock_offset_ns,
        "retimed_topics": retimed_counts,
        "unmatched_or_unmapped_samples_dropped": dropped_counts,
        "bridge_timing_rows_output": timing_output_count,
        "other_messages_copied": copied_messages,
        "output_message_count": actual_message_count,
        "source_packets_renumbered_in_derived_timing_topic": True,
        "source_data_modified": False,
        "interpretation": (
            "Derived analysis bag only: target sensor/truth/odom headers use "
            "matched Bridge request times. Unmatched samples are removed; "
            "all non-target topic payloads and bag receipt timestamps are copied."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bag", type=Path, required=True)
    parser.add_argument("--output-bag-dir", type=Path, required=True,
                        help="new directory whose bag file will be run_0.db3")
    args = parser.parse_args()
    report_path = args.output_bag_dir.resolve().parent / "retime_report.json"
    if report_path.exists():
        parser.error(f"refusing to overwrite {report_path}")
    try:
        report = transform(args.input_bag, args.output_bag_dir)
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        parser.exit(2, f"request-time bag conversion failed: {exc}\n")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
