#!/usr/bin/env python3
"""Certify Bridge -> odometry exact applied-input association from a ROS 2 bag.

Only evaluates signals captured at the same integer-nanosecond Bridge request
epoch. Never matches a nearest time or uses future command samples. Intended
for the accuracy gate, not for real-time controller input.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path


def score_epochs(
    requests: dict[int, tuple[float, float]],
    odometry: dict[int, tuple[float, float]],
    imu_stamps: set[int],
    tolerance: float = 1e-6,
) -> dict:
    """Return exact-stamp/physical input parity; no receipt-time interpolation."""
    missing = sorted(set(odometry) - set(requests))
    matched = set(odometry) & set(requests)
    mismatches = []
    max_steering = 0.0
    max_throttle = 0.0
    for stamp in sorted(matched):
        actual = requests[stamp]
        used = odometry[stamp]
        if not all(math.isfinite(v) for v in (*actual, *used)):
            mismatches.append(stamp)
            continue
        ds = abs(actual[0] - used[0])
        dt = abs(actual[1] - used[1])
        max_steering = max(max_steering, ds)
        max_throttle = max(max_throttle, dt)
        if ds > tolerance or dt > tolerance:
            mismatches.append(stamp)
    return {
        "request_commands": len(requests),
        "odometry_epochs": len(odometry),
        "imu_epochs": len(imu_stamps),
        "odometry_exact_stamp_matches": len(matched),
        "odometry_exact_stamp_missing": len(missing),
        "odometry_command_mismatch_count": len(mismatches),
        "imu_exact_stamp_missing": len(imu_stamps - set(requests)),
        "maximum_steering_normalized_error": max_steering,
        "maximum_throttle_normalized_error": max_throttle,
        "missing_epoch_ns": missing[:32],
        "mismatched_epoch_ns": mismatches[:32],
        "certified": (len(odometry) > 0 and len(matched) == len(odometry) and
                      not mismatches and not (imu_stamps - set(requests))),
    }


def analyze(bag_path: Path) -> dict:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    if bag_path.is_dir():
        files = sorted(bag_path.glob("*.db3"))
        if not files:
            raise ValueError("bag directory does not contain .db3 storage")
    else:
        files = [bag_path]
    topic_by_name: dict[str, tuple[str, int]] = {}
    requests: dict[int, tuple[float, float]] = {}
    odometry: dict[int, tuple[float, float]] = {}
    feedback: dict[int, tuple[float, float]] = {}
    odometry_feedback: dict[int, tuple[float, float]] = {}
    imu_stamps: set[int] = set()
    for path in files:
        with sqlite3.connect(str(path)) as db:
            topic_by_name = {
                name: (message_type, tid)
                for tid, name, message_type in db.execute(
                    "SELECT id, name, type FROM topics")}
            target = {
                "/odom/bridge_request_command": "request",
                "/odom/bridge_actuator_feedback": "feedback",
                "/odom/unity_plant_state": "odometry",
                "/autodrive/roboracer_1/imu": "imu",
            }
            for name, role in target.items():
                if name not in topic_by_name:
                    if role == "feedback":
                        continue
                    raise ValueError(f"bag missing required topic {name}")
                message_type, tid = topic_by_name[name]
                msg_type = get_message(message_type)
                for (cdr,) in db.execute(
                        "SELECT data FROM messages WHERE topic_id=?", (tid,)):
                    msg = deserialize_message(cdr, msg_type)
                    stamp = int(msg.header.stamp.sec) * 1_000_000_000 +                             int(msg.header.stamp.nanosec)
                    if role == "feedback":
                        feedback[stamp] = (
                            float(msg.steering_angle_rad),
                            float(msg.throttle_normalized),
                        )
                    elif role == "request":
                        requests[stamp] = (
                            float(msg.steering_angle_rad) / (math.pi / 6),
                            float(msg.throttle_normalized),
                        )
                    elif role == "odometry":
                        odometry[stamp] = (
                            float(msg.delayed_steering_command_2_normalized),
                            float(msg.delayed_throttle_command_2_normalized),
                        )
                        if msg.actuator_feedback_valid:
                            odometry_feedback[stamp] = (
                                float(msg.steering_feedback_rad),
                                float(msg.throttle_feedback_normalized),
                            )
                    else:
                        imu_stamps.add(stamp)
    result = score_epochs(requests, odometry, imu_stamps)
    feedback_result = score_epochs(feedback, odometry_feedback, set(odometry))
    result["feedback_epoch_audit"] = feedback_result
    result["certified"] = result["certified"] and feedback_result["certified"]
    result["scope"] = "host request association only; no Unity apply-tick proof"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = analyze(args.bag)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0 if summary["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
