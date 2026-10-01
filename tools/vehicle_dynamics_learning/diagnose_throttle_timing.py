#!/usr/bin/env python3
"""Inspect high-acceleration samples around sub-10 ms throttle-sweep bursts.

This is a read-only bag diagnostic. It compares bag receipt intervals with
odometry header-stamp intervals and prints neighboring odometry, IMU, and
bridge packet-timing messages to distinguish delivery bursts from model
nonlinearity or timestamp misalignment.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_open_plane_dynamics as common  # noqa: E402
from tools.vehicle_dynamics_learning.structured_body_models import (  # noqa: E402
    generalized_accelerations,
)


PACKET_TIMING = common.PACKET_TIMING
WINDOW_NS = 120_000_000


def _sample_payloads(connection: sqlite3.Connection, topic_id: int,
                     message_class: type, center_ns: int) -> list[dict[str, Any]]:
    from rclpy.serialization import deserialize_message

    output = []
    for receipt_ns, payload in connection.execute(
            "SELECT timestamp, data FROM messages "
            "WHERE topic_id=? AND timestamp BETWEEN ? AND ? "
            "ORDER BY timestamp, id",
            (topic_id, center_ns - WINDOW_NS, center_ns + WINDOW_NS)):
        message = deserialize_message(bytes(payload), message_class)
        row: dict[str, Any] = {"receipt_ns": int(receipt_ns)}
        if hasattr(message, "header"):
            stamp = common._stamp_ns(message.header.stamp)
            row["source_ns"] = stamp
        if hasattr(message, "twist") and hasattr(message, "pose"):
            twist = message.twist.twist
            pose = message.pose.pose
            q = pose.orientation
            row["body_twist"] = [float(twist.linear.x), float(twist.linear.y),
                                 float(twist.angular.z)]
            row["pose_xy_yaw"] = [float(pose.position.x),
                                  float(pose.position.y),
                                  math.atan2(2.0 * (float(q.w) * float(q.z)
                                                    + float(q.x) * float(q.y)),
                                             1.0 - 2.0 * (float(q.y)**2
                                                          + float(q.z)**2))]
        elif hasattr(message, "linear_acceleration"):
            row["imu_accel_xyz"] = [float(message.linear_acceleration.x),
                                    float(message.linear_acceleration.y),
                                    float(message.linear_acceleration.z)]
            row["imu_gyro_xyz"] = [float(message.angular_velocity.x),
                                   float(message.angular_velocity.y),
                                   float(message.angular_velocity.z)]
        elif hasattr(message, "data"):
            text = str(message.data)
            try:
                row["data"] = json.loads(text)
            except json.JSONDecodeError:
                row["data"] = text
        output.append(row)
    return output


def _decode_events(path: Path, centers: list[int]) -> dict[str, list[dict[str, Any]]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from std_msgs.msg import String

        topic_classes = ((common.ODOM, Odometry), (common.IMU, Imu),
                         (PACKET_TIMING, String))
        decoded: dict[str, list[dict[str, Any]]] = {}
        for topic, message_class in topic_classes:
            if topic not in topics:
                decoded[topic] = []
                continue
            message_class_from_bag = common.get_message(topics[topic][1])
            decoded[topic] = []
            for center in centers:
                decoded[topic].append({
                    "center_ns": int(center),
                    "samples": _sample_payloads(connection, topics[topic][0],
                                                 message_class_from_bag, center),
                })
        return decoded
    finally:
        connection.close()


def inspect(dataset_path: Path, output: Path | None) -> dict[str, Any]:
    archive = np.load(dataset_path, allow_pickle=False)
    bounds = archive["sequence_bounds"]
    run_ids = archive["sequence_run_id"].astype(str)
    frames_by_name = {name: archive[name] for name in (
        "body_state", "receipt_time_ns", "source_stamp_offset_ms",
        "sequence_index")}
    manifest = json.loads((dataset_path.parent / "manifest.json").read_text())
    candidates = []
    timing_intervals = {"receipt_ms": [], "header_ms": []}
    for sequence_id, (start, end) in enumerate(bounds):
        start, end = int(start), int(end)
        receipt = frames_by_name["receipt_time_ns"][start:end, 0]
        offsets = frames_by_name["source_stamp_offset_ms"][start:end, 0]
        source = receipt + np.rint(offsets * 1e6).astype(np.int64)
        state = frames_by_name["body_state"][start:end]
        receipt_dt = np.diff(receipt).astype(np.float64) / 1e9
        source_dt = np.diff(source).astype(np.float64) / 1e9
        timing_intervals["receipt_ms"].extend((receipt_dt * 1000.0).tolist())
        timing_intervals["header_ms"].extend((source_dt * 1000.0).tolist())
        accelerations = generalized_accelerations(
            state[:-1], state[1:], source_dt)
        sub10ms = np.flatnonzero((source_dt > 0.0) & (source_dt < 0.010))
        for local in sub10ms:
            acc_norm = float(np.linalg.norm(accelerations[local]))
            candidates.append({
                "sequence_index": sequence_id,
                "run_id": str(run_ids[sequence_id]),
                "receipt_ns": int(receipt[local + 1]),
                "receipt_dt_ms": float(receipt_dt[local] * 1000.0),
                "header_dt_ms": float(source_dt[local] * 1000.0),
                "acceleration_norm_mps2_equivalent": acc_norm,
                "acceleration": accelerations[local].astype(float).tolist(),
                "previous_body_state": state[local].astype(float).tolist(),
                "next_body_state": state[local + 1].astype(float).tolist(),
            })
    # Limit raw-bag decoding to the most informative distinct events.
    candidates.sort(key=lambda row: row["acceleration_norm_mps2_equivalent"],
                    reverse=True)
    selected = []
    used: dict[int, list[int]] = {}
    for row in candidates:
        offsets = used.setdefault(row["sequence_index"], [])
        if any(abs(row["receipt_ns"] - prior) < 500_000_000
               for prior in offsets):
            continue
        selected.append(row)
        offsets.append(row["receipt_ns"])
        if len(selected) >= 16:
            break
    bag_by_run = {row["run_id"]: Path(row["bag"])
                  for row in manifest["sequences"]}
    decoded_by_bag = {}
    for row in selected:
        bag = bag_by_run[row["run_id"]]
        key = str(bag)
        if key not in decoded_by_bag:
            decoded_by_bag[key] = {"path": bag, "events": []}
        decoded_by_bag[key]["events"].append(row["receipt_ns"])
    for bag_info in decoded_by_bag.values():
        bag_path = bag_info["path"]
        if not bag_path.is_absolute():
            bag_path = REPO_ROOT / bag_path
        bag_info["decoded"] = _decode_events(bag_path, bag_info["events"])
    receipt_dt = np.asarray(timing_intervals["receipt_ms"])
    source_dt = np.asarray(timing_intervals["header_ms"])
    report = {
        "dataset": str(dataset_path.resolve()),
        "sequence_count": int(len(bounds)),
        "sample_interval_ms": {
            "receipt": {str(q): float(np.quantile(receipt_dt, q))
                        for q in (0.0, 0.01, 0.5, 0.99, 1.0)},
            "header_stamp": {str(q): float(np.quantile(source_dt, q))
                             for q in (0.0, 0.01, 0.5, 0.99, 1.0)},
            "receipt_under_10ms_fraction": float(np.mean(receipt_dt < 10.0)),
            "header_under_10ms_fraction": float(np.mean(source_dt < 10.0)),
        },
        "sub_10ms_transition_count": int(len(candidates)),
        "selected_events": selected,
        "raw_neighboring_messages": {
            key: value["decoded"] for key, value in decoded_by_bag.items()},
        "interpretation": (
            "Short intervals with compensating long intervals can preserve "
            "an average 40 Hz while corrupting finite-difference acceleration. "
            "Use the raw neighboring timestamps and bridge packet-timing "
            "payloads above to determine whether the simulator, bridge, or "
            "recording path creates the burst."),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.dataset.resolve(),
                     args.output.resolve() if args.output else None)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
