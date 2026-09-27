#!/usr/bin/env python3
"""Offline-only odometry/AMCL accuracy analysis against recorded simulator truth.

This reads a closed ROS bag; it creates no ROS subscriptions. IPS and the
bridge odometry topic are development ground truth and must never be used by
the competition controller/localization nodes.
"""

from __future__ import annotations

import argparse
import bisect
import math
import sqlite3
import statistics
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - depends on the ROS installation
    raise SystemExit(f"ROS 2 Python modules are required: {exc}")


IPS = "/autodrive/roboracer_1/ips"
TRUTH_ODOM = "/autodrive/roboracer_1/odom"
AMCL_POSE = "/amcl_pose"
MAP_POSE = "/current_map_pose"
ODOM = "/odom"
EKF_ODOM = "/ekf_odom"
LAP_COUNT = "/autodrive/roboracer_1/lap_count"
COM_X_M = 0.15532
MAX_ALIGN_NS = 20_000_000


def stamp_ns(stamp: Any) -> int | None:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def read_topic(
    connection: sqlite3.Connection,
    topics: dict[str, tuple[int, str]],
    name: str,
) -> list[tuple[int, Any]]:
    if name not in topics:
        return []
    topic_id, msg_type = topics[name]
    message_class = get_message(msg_type)
    return [
        (int(receipt_ns), deserialize_message(bytes(payload), message_class))
        for receipt_ns, payload in connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
            (topic_id,),
        )
    ]


def pair_nearest(
    left: list[tuple[int, Any]],
    right: list[tuple[int, Any]],
    left_time,
    right_time,
) -> list[tuple[Any, Any, int]]:
    right_times = [right_time(row) for row in right]
    pairs: list[tuple[Any, Any, int]] = []
    for left_row in left:
        query = left_time(left_row)
        index = bisect.bisect_left(right_times, query)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(right)]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(right_times[i] - query))
        offset = right_times[best] - query
        if abs(offset) <= MAX_ALIGN_NS:
            pairs.append((left_row, right[best], offset))
    return pairs


def position_error_report(label: str, errors: list[tuple[float, float]], offsets_ms: list[float]) -> None:
    distances = [math.hypot(dx, dy) for dx, dy in errors]
    dxs = [item[0] for item in errors]
    dys = [item[1] for item in errors]
    print(
        f"{label}: n={len(errors)}, position error m p50/p95/p99/max="
        f"{fmt(percentile(distances, 50))}/{fmt(percentile(distances, 95))}/"
        f"{fmt(percentile(distances, 99))}/{fmt(max(distances) if distances else None)}; "
        f"x/y bias={fmt(statistics.fmean(dxs) if dxs else None)}/"
        f"{fmt(statistics.fmean(dys) if dys else None)} m; "
        f"truth-time offset ms p50/p95 abs-p95="
        f"{fmt(percentile(offsets_ms, 50), 2)}/{fmt(percentile(offsets_ms, 95), 2)}/"
        f"{fmt(percentile([abs(x) for x in offsets_ms], 95), 2)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="practice-run run_0.db3")
    args = parser.parse_args()
    if not args.bag.is_file():
        parser.error(f"bag does not exist: {args.bag}")

    connection = sqlite3.connect(args.bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = {
            name: (int(topic_id), msg_type)
            for topic_id, name, msg_type in connection.execute(
                "SELECT id,name,type FROM topics"
            )
        }
        required = (IPS, TRUTH_ODOM, AMCL_POSE, MAP_POSE, ODOM, LAP_COUNT)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("bag missing required topic(s): " + ", ".join(missing))

        ips = read_topic(connection, topics, IPS)
        truth_odom = read_topic(connection, topics, TRUTH_ODOM)
        amcl = read_topic(connection, topics, AMCL_POSE)
        map_pose = read_topic(connection, topics, MAP_POSE)
        odom = read_topic(connection, topics, ODOM)
        ekf_odom = read_topic(connection, topics, EKF_ODOM)
        lap_counts = read_topic(connection, topics, LAP_COUNT)

        if not all((ips, truth_odom, amcl, map_pose, odom, lap_counts)):
            raise ValueError("one or more required state streams contain no messages")

        ips_pairs = [(receipt, message) for receipt, message in ips]
        truth_by_source = [
            (stamp_ns(message.header.stamp), (receipt, message))
            for receipt, message in truth_odom
        ]
        truth_by_source = sorted(
            (int(source), row) for source, row in truth_by_source if source is not None
        )
        truth_source_times = [source for source, _ in truth_by_source]

        print(f"Bag: {args.bag}")
        print("Offline ground-truth comparison only; no runtime node consumes IPS.")

        # The controller's map pose is compared at receipt time because its
        # header is refreshed after AMCL correction. Raw AMCL keeps its scan
        # source stamp and is compared at that source time.
        map_errors: list[tuple[float, float]] = []
        map_offsets: list[float] = []
        for (receipt, message), (_, truth) , offset in pair_nearest(
            map_pose, ips_pairs, lambda row: row[0], lambda row: row[0]
        ):
            point = truth
            estimated = message.pose.pose.position
            map_errors.append((float(estimated.x - point.x), float(estimated.y - point.y)))
            map_offsets.append(offset / 1e6)
        position_error_report("MPC input /current_map_pose vs IPS", map_errors, map_offsets)

        amcl_errors: list[tuple[float, float]] = []
        amcl_offsets: list[float] = []
        amcl_yaw_errors: list[float] = []
        for receipt, message in amcl:
            source = stamp_ns(message.header.stamp)
            if source is None:
                continue
            index = bisect.bisect_left(truth_source_times, source)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(truth_by_source)]
            if not candidates:
                continue
            best = min(candidates, key=lambda i: abs(truth_source_times[i] - source))
            truth_time, (_, truth) = truth_by_source[best]
            offset = truth_time - source
            if abs(offset) > MAX_ALIGN_NS:
                continue
            truth_pos = truth.pose.pose.position
            amcl_pos = message.pose.pose.position
            amcl_errors.append((float(amcl_pos.x - truth_pos.x), float(amcl_pos.y - truth_pos.y)))
            amcl_offsets.append(offset / 1e6)
            amcl_yaw_errors.append(wrap_angle(
                yaw_from_quaternion(message.pose.pose.orientation) -
                yaw_from_quaternion(truth.pose.pose.orientation)))
        position_error_report("Raw /amcl_pose vs bridge truth", amcl_errors, amcl_offsets)
        yaw_abs = [abs(value) for value in amcl_yaw_errors]
        print(
            "Raw AMCL yaw error rad p50/p95/p99/max="
            f"{fmt(percentile(yaw_abs, 50))}/{fmt(percentile(yaw_abs, 95))}/"
            f"{fmt(percentile(yaw_abs, 99))}/{fmt(max(yaw_abs) if yaw_abs else None)}"
        )

        # Integrate-frame odometry starts at an arbitrary origin and heading.
        # Align that frame once at the first sample, then report the residual
        # drift against IPS without refitting the transform around the lap.
        initial_pair = None
        for row in odom:
            source = stamp_ns(row[1].header.stamp)
            if source is None:
                continue
            index = bisect.bisect_left(truth_source_times, source)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(truth_by_source)]
            if not candidates:
                continue
            best = min(candidates, key=lambda i: abs(truth_source_times[i] - source))
            if abs(truth_source_times[best] - source) <= MAX_ALIGN_NS:
                ips_index = bisect.bisect_left([stamp for stamp, _ in ips], row[0])
                ips_candidates = [i for i in (ips_index - 1, ips_index) if 0 <= i < len(ips)]
                if ips_candidates:
                    ips_best = min(ips_candidates, key=lambda i: abs(ips[i][0] - row[0]))
                    initial_pair = (row, truth_by_source[best][1][1], ips[ips_best][1])
                    break
        if initial_pair is None:
            raise ValueError("could not align initial odom, truth odom, and IPS samples")

        first_odom = initial_pair[0][1]
        first_truth = initial_pair[1]
        first_ips = initial_pair[2]
        rotation = wrap_angle(
            yaw_from_quaternion(first_truth.pose.pose.orientation) -
            yaw_from_quaternion(first_odom.pose.pose.orientation))
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        odom_origin = first_odom.pose.pose.position
        ips_origin_x, ips_origin_y = float(first_ips.x), float(first_ips.y)

        odom_position_errors: list[tuple[float, float]] = []
        odom_offsets: list[float] = []
        odom_error_by_lap: dict[int, list[float]] = {}
        lap_times = [stamp for stamp, _ in lap_counts]
        lap_values = [int(message.data) for _, message in lap_counts]
        odom_source_times = [stamp_ns(message.header.stamp) for _, message in odom]
        for (receipt, estimate), source in zip(odom, odom_source_times):
            if source is None:
                continue
            index = bisect.bisect_left(truth_source_times, source)
            candidates = [i for i in (index - 1, index) if 0 <= i < len(truth_by_source)]
            if not candidates:
                continue
            best = min(candidates, key=lambda i: abs(truth_source_times[i] - source))
            truth_time, (_, truth) = truth_by_source[best]
            source_offset = truth_time - source
            if abs(source_offset) > MAX_ALIGN_NS:
                continue
            ips_index = bisect.bisect_left([stamp for stamp, _ in ips], receipt)
            ips_candidates = [i for i in (ips_index - 1, ips_index) if 0 <= i < len(ips)]
            if not ips_candidates:
                continue
            ips_best = min(ips_candidates, key=lambda i: abs(ips[i][0] - receipt))
            ips_time, truth_point = ips[ips_best]
            if abs(ips_time - receipt) > MAX_ALIGN_NS:
                continue

            position = estimate.pose.pose.position
            dx = float(position.x - odom_origin.x)
            dy = float(position.y - odom_origin.y)
            aligned_x = ips_origin_x + cos_r * dx - sin_r * dy
            aligned_y = ips_origin_y + sin_r * dx + cos_r * dy
            error = (aligned_x - float(truth_point.x), aligned_y - float(truth_point.y))
            odom_position_errors.append(error)
            odom_offsets.append((ips_time - receipt) / 1e6)
            lap_index = bisect.bisect_right(lap_times, receipt) - 1
            lap = lap_values[max(0, lap_index)]
            odom_error_by_lap.setdefault(lap, []).append(math.hypot(*error))

        position_error_report("Integrated /odom vs IPS after one initial SE(2) alignment",
                              odom_position_errors, odom_offsets)
        print("Integrated odom position error p95 by lap (initial alignment held fixed):")
        for lap, errors in sorted(odom_error_by_lap.items()):
            print(f"  lap_count={lap}: n={len(errors)}, p95={fmt(percentile(errors, 95))} m, "
                  f"max={fmt(max(errors))} m")

        for topic, stream in ((ODOM, odom), (EKF_ODOM, ekf_odom)):
            if not stream:
                continue
            errors_u: list[float] = []
            errors_v: list[float] = []
            errors_r: list[float] = []
            offsets: list[float] = []
            for receipt, estimate in stream:
                source = stamp_ns(estimate.header.stamp)
                if source is None:
                    continue
                index = bisect.bisect_left(truth_source_times, source)
                candidates = [i for i in (index - 1, index) if 0 <= i < len(truth_by_source)]
                if not candidates:
                    continue
                best = min(candidates, key=lambda i: abs(truth_source_times[i] - source))
                truth_time, (_, truth) = truth_by_source[best]
                offset = truth_time - source
                if abs(offset) > MAX_ALIGN_NS:
                    continue
                estimate_twist = estimate.twist.twist
                truth_twist = truth.twist.twist
                truth_vy_rear = float(truth_twist.linear.y) - float(
                    truth_twist.angular.z) * COM_X_M
                errors_u.append(float(estimate_twist.linear.x) - float(truth_twist.linear.x))
                errors_v.append(float(estimate_twist.linear.y) - truth_vy_rear)
                errors_r.append(float(estimate_twist.angular.z) - float(truth_twist.angular.z))
                offsets.append(offset / 1e6)
            if errors_u:
                print(
                    f"{topic} body-twist error vs truth at rear axle: n={len(errors_u)}, "
                    f"u RMSE={math.sqrt(statistics.fmean(x*x for x in errors_u)):.4f} m/s, "
                    f"v RMSE={math.sqrt(statistics.fmean(x*x for x in errors_v)):.4f} m/s, "
                    f"r RMSE={math.sqrt(statistics.fmean(x*x for x in errors_r)):.4f} rad/s; "
                    f"offset abs-p95={fmt(percentile([abs(x) for x in offsets], 95), 2)} ms"
                )
    finally:
        connection.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}")
