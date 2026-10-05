#!/usr/bin/env python3
"""Read-only summary of lap, collision, MPC and LiDAR evidence in a ROS 2 bag."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - depends on the ROS installation
    raise SystemExit(
        "ROS 2 Python message modules are required (rclpy and rosidl_runtime_py): "
        f"{exc}"
    ) from exc


LAP_COUNT = "/autodrive/roboracer_1/lap_count"
LAST_LAP_TIME = "/autodrive/roboracer_1/last_lap_time"
COLLISION_COUNT = "/autodrive/roboracer_1/collision_count"
LIDAR = "/autodrive/roboracer_1/lidar"
MPC_DIAGNOSTICS = "/mpc/diagnostics"
REQUIRED_TOPICS = (LAP_COUNT, LAST_LAP_TIME, COLLISION_COUNT, LIDAR)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def topic_rows(connection: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    return {
        name: (int(topic_id), msg_type)
        for topic_id, name, msg_type in connection.execute(
            "SELECT id, name, type FROM topics"
        )
    }


def messages(
    connection: sqlite3.Connection,
    topic_id: int,
    msg_type: str,
) -> list[tuple[int, Any]]:
    message_class = get_message(msg_type)
    decoded: list[tuple[int, Any]] = []
    for timestamp, serialized in connection.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp, id",
        (topic_id,),
    ):
        decoded.append(
            (int(timestamp), deserialize_message(bytes(serialized), message_class))
        )
    return decoded


def stamp_ns(stamp: Any) -> int | None:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def cadence_report(timestamps_ns: list[int], *, preserve_order: bool = False) -> dict[str, Any]:
    ordered = list(timestamps_ns) if preserve_order else sorted(timestamps_ns)
    gaps_ms = [
        (later - earlier) / 1_000_000.0
        for earlier, later in zip(ordered, ordered[1:])
    ]
    positive_gaps_ms = [gap for gap in gaps_ms if gap > 0.0]
    span_s = (ordered[-1] - ordered[0]) / 1_000_000_000.0 if len(ordered) > 1 else 0.0
    return {
        "count": len(ordered),
        # Use N-1 intervals over elapsed time; N/span overstates the measured rate.
        "rate_hz": (len(ordered) - 1) / span_s if span_s > 0 else None,
        "p50_ms": percentile(positive_gaps_ms, 50),
        "p95_ms": percentile(positive_gaps_ms, 95),
        "p99_ms": percentile(positive_gaps_ms, 99),
        "max_ms": max(positive_gaps_ms) if positive_gaps_ms else None,
        "over_30_ms": sum(gap > 30.0 for gap in positive_gaps_ms),
        "over_35_ms": sum(gap > 35.0 for gap in positive_gaps_ms),
        "duplicate_stamps": sum(gap == 0.0 for gap in gaps_ms),
        "nonmonotonic_stamps": sum(gap < 0.0 for gap in gaps_ms),
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def analyze_structured(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")

    # URI mode=ro makes accidental writes impossible. No ROS bag APIs or simulator
    # connections are used; only the supplied SQLite file is read.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = topic_rows(connection)
        missing = [name for name in REQUIRED_TOPICS if name not in topics]
        if missing:
            raise ValueError("bag is missing required topics: " + ", ".join(missing))

        bag_start_value = connection.execute(
            "SELECT MIN(timestamp) FROM messages"
        ).fetchone()[0]
        if bag_start_value is None:
            raise ValueError("bag contains no messages")
        bag_start_ns = int(bag_start_value)

        def read(topic: str) -> list[tuple[int, Any]]:
            topic_id, msg_type = topics[topic]
            return messages(connection, topic_id, msg_type)

        lap_counts = read(LAP_COUNT)
        lap_times = read(LAST_LAP_TIME)
        collision_counts = read(COLLISION_COUNT)

        transitions: list[tuple[int, int]] = []
        last_count: int | None = None
        for timestamp, message in lap_counts:
            count = int(message.data)
            if last_count is None or count != last_count:
                transitions.append((timestamp, count))
                last_count = count

        finite_lap_times = [
            (timestamp, float(message.data))
            for timestamp, message in lap_times
            if math.isfinite(float(message.data))
        ]
        lap_time_by_transition: dict[int, float] = {}
        unused_times = set(range(len(finite_lap_times)))
        for timestamp, count in transitions:
            if count <= 0 or not unused_times:
                continue
            nearest = min(unused_times, key=lambda i: abs(finite_lap_times[i][0] - timestamp))
            event_ns, value = finite_lap_times[nearest]
            # Lap count and last_lap_time are published by separate callbacks;
            # allow their small transport/order skew while rejecting unrelated data.
            if abs(event_ns - timestamp) <= 500_000_000:
                lap_time_by_transition[count] = value
                unused_times.remove(nearest)

        final_lap_count = transitions[-1][1] if transitions else 0
        warmup = lap_time_by_transition.get(1)
        scored = [lap_time_by_transition[count] for count in range(2, 12)
                  if count in lap_time_by_transition]
        extra = lap_time_by_transition.get(12)
        scored_mean = sum(scored) / len(scored) if scored else None

        initial_collision = int(collision_counts[0][1].data) if collision_counts else 0
        first_collision_ns: int | None = bag_start_ns if initial_collision > 0 else None
        final_collision = initial_collision
        for timestamp, message in collision_counts:
            count = int(message.data)
            final_collision = count
            if first_collision_ns is None and count > initial_collision:
                first_collision_ns = timestamp

        lidar = read(LIDAR)
        if not lidar:
            raise ValueError(f"bag topic {LIDAR} contains no messages")
        receipt_ns = [timestamp for timestamp, _ in lidar]
        header_ns = [
            header_stamp
            for _, message in lidar
            if (header_stamp := stamp_ns(message.header.stamp)) is not None
        ]

        # The failed-run status summary stops at the first collision event. The
        # full safe-run status summary uses the whole run.
        mpc_counts: dict[str, int] = {}
        if MPC_DIAGNOSTICS in topics:
            for timestamp, message in read(MPC_DIAGNOSTICS):
                if first_collision_ns is not None and timestamp >= first_collision_ns:
                    continue
                try:
                    status = json.loads(message.data).get("status", "<missing status>")
                except (TypeError, json.JSONDecodeError):
                    status = "<invalid JSON>"
                mpc_counts[str(status)] = mpc_counts.get(str(status), 0) + 1

        return {
            "bag": str(path),
            "final_lap_count": final_lap_count,
            "lap_count_transitions": [
                {
                    "count": count,
                    "seconds_from_first_lidar": (timestamp - receipt_ns[0]) / 1e9,
                }
                for timestamp, count in transitions
            ],
            "warmup_lap_s": warmup,
            "scored_laps_s": scored,
            "scored_lap_count": len(scored),
            "scored_mean_s": scored_mean,
            "scored_median_s": statistics.median(scored) if scored else None,
            "scored_stdev_s": statistics.stdev(scored) if len(scored) > 1 else None,
            "scored_best_s": min(scored) if scored else None,
            "extra_lap_s": extra,
            "collision_initial": initial_collision,
            "collision_final": final_collision,
            "collision_delta": final_collision - initial_collision,
            "first_collision_s_from_first_lidar": (
                None if first_collision_ns is None or initial_collision > 0
                else (first_collision_ns - receipt_ns[0]) / 1e9
            ),
            "lidar_bag_receipt": cadence_report(receipt_ns),
            "lidar_header_timestamp": cadence_report(header_ns, preserve_order=True),
            "mpc_status_counts": mpc_counts,
        }
    finally:
        connection.close()


def format_report(report: dict[str, Any]) -> str:
    scored = report["scored_laps_s"]
    collision_delta = report["collision_delta"]
    first_collision_s = report["first_collision_s_from_first_lidar"]
    output = [
        f"Bag: {report['bag']}",
        f"Final lap count: {report['final_lap_count']}",
        "Lap-count transitions (count@seconds from first LiDAR): "
        + ", ".join(
            f"{item['count']}@{item['seconds_from_first_lidar']:.3f}"
            for item in report["lap_count_transitions"]
        ),
        f"Warmup lap (count 1): {fmt(report['warmup_lap_s'], 4)} s",
        f"Scored laps (counts 2-11): {len(scored)}/10",
    ]
    output.extend(
        f"  scored {index:02d}: {lap_time:.4f} s"
        for index, lap_time in enumerate(scored, start=1)
    )
    output.extend(
        [
            f"Scored mean ({'complete' if len(scored) == 10 else 'partial'}, "
            f"{len(scored)}/10): {fmt(report['scored_mean_s'], 4)} s",
            f"Extra lap (count 12): {fmt(report['extra_lap_s'], 4)} s",
            f"Collision count: {report['collision_initial']} -> {report['collision_final']}",
            "First collision: "
            + (f"already {report['collision_initial']} at bag start"
               if report["collision_initial"] > 0 else
               ("none observed" if collision_delta <= 0 else
                f"{first_collision_s:.6f} s from first LiDAR receipt")),
        ]
    )
    for label, stats in (
        ("bag receipt", report["lidar_bag_receipt"]),
        ("header timestamp", report["lidar_header_timestamp"]),
    ):
        output.append(
            f"LiDAR {label}: {stats['count']} messages, "
            f"{fmt(stats['rate_hz'])} Hz; gaps ms p50/p95/p99/max "
            f"{fmt(stats['p50_ms'])}/{fmt(stats['p95_ms'])}/"
            f"{fmt(stats['p99_ms'])}/{fmt(stats['max_ms'])}; "
            f">30 ms {stats['over_30_ms']}, >35 ms {stats['over_35_ms']}"
            + (f"; duplicate stamps {stats['duplicate_stamps']}, "
               f"nonmonotonic stamps {stats['nonmonotonic_stamps']}"
               if label == "header timestamp" else "")
        )
    mpc_scope = (
        "before first collision"
        if collision_delta > 0 or report["collision_initial"] > 0
        else "recorded run"
    )
    output.append(f"MPC diagnostic statuses ({mpc_scope}):")
    mpc_counts = report["mpc_status_counts"]
    if mpc_counts:
        output.extend(f"  {status}: {count}" for status, count in sorted(mpc_counts.items()))
    else:
        output.append("  no diagnostic messages")
    return "\n".join(output)


def analyze(path: Path) -> str:
    return format_report(analyze_structured(path))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report measured lap, collision, MPC, and LiDAR evidence from a ROS 2 .db3 bag."
    )
    parser.add_argument("bag", type=Path, help="ROS 2 SQLite .db3 file")
    parser.add_argument("--json", action="store_true", help="emit structured JSON")
    args = parser.parse_args()
    try:
        report = analyze_structured(args.bag)
        print(json.dumps(report, indent=2, sort_keys=True) if args.json
              else format_report(report))
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
