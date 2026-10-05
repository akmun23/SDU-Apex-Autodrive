#!/usr/bin/env python3
"""Score localization errors in raceline tangent/normal coordinates.

This is a read-only post-run analysis. Simulator truth is loaded from the bag
only for scoring; no ROS nodes are started and no runtime component consumes
truth. Odom streams are aligned to truth once at the start, then scored without
refitting that transform.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - ROS image dependency
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc

from tools.racing.track_projection import TrackProjection, wrap_angle


TRUTH_ODOM = "/autodrive/roboracer_1/odom"
LAP_COUNT = "/autodrive/roboracer_1/lap_count"
AMCL_HEALTH = "/amcl_localization_health"
AMCL_ALIGNMENT = "/amcl_scan_alignment"
POSE_STREAMS = {
    "/amcl_pose": "map",
    "/current_map_pose": "map",
    "/odom": "odom",
    "/ekf_odom": "odom",
}
MAX_CAUSAL_AGE_NS = 30_000_000


def message_rows(
    connection: sqlite3.Connection,
    topics: dict[str, tuple[int, str]],
    name: str,
) -> list[tuple[int, Any]]:
    if name not in topics:
        return []
    topic_id, type_name = topics[name]
    message_type = get_message(type_name)
    return [
        (int(receipt_ns), deserialize_message(bytes(payload), message_type))
        for receipt_ns, payload in connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
            (topic_id,),
        )
    ]


def stamp_ns(message: Any) -> int | None:
    stamp = message.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def pose_of(message: Any) -> Any:
    return message.pose.pose


def nearest_causal(
    target_ns: int,
    rows: list[tuple[int, Any]],
    source_stamps: list[int],
) -> tuple[Any, int, int] | None:
    index = bisect.bisect_right(source_stamps, target_ns) - 1
    if index < 0:
        return None
    age_ns = target_ns - source_stamps[index]
    if age_ns > MAX_CAUSAL_AGE_NS:
        return None
    return rows[index][1], age_ns, index


def lap_at(receipt_ns: int, lap_times: list[int], lap_values: list[int]) -> int:
    index = bisect.bisect_right(lap_times, receipt_ns) - 1
    return int(lap_values[index]) if index >= 0 else -1


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(field: str) -> dict[str, float | None]:
        values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
        absolute = [abs(value) for value in values]
        return {
            "count": len(values),
            "bias": statistics.fmean(values) if values else None,
            "abs_p50": percentile(absolute, 50),
            "abs_p95": percentile(absolute, 95),
            "abs_p99": percentile(absolute, 99),
            "abs_max": max(absolute) if absolute else None,
        }

    return {
        "sample_count": len(rows),
        "normal_error_m": stats("normal_error_m"),
        "tangential_error_m": stats("tangential_error_m"),
        "yaw_error_rad": stats("yaw_error_rad"),
        "truth_lateral_offset_m": stats("truth_lateral_offset_m"),
        "causal_truth_age_ms": stats("truth_age_ms"),
    }


def summarize_scan_corrections(
    connection: sqlite3.Connection,
    topics: dict[str, tuple[int, str]],
    truth_messages: list[tuple[int, Any]],
    truth_stamps: list[int],
    truth_projections: list[Any],
) -> dict[str, Any]:
    health_messages = message_rows(connection, topics, AMCL_HEALTH)
    alignment_messages = message_rows(connection, topics, AMCL_ALIGNMENT)
    health_rows = []
    for _, message in health_messages:
        data = list(message.data)
        if len(data) < 15 or not math.isfinite(float(data[14])):
            continue
        health_rows.append((int(round(float(data[14]) * 1e9)), data))
    health_rows.sort(key=lambda item: item[0])
    health_stamps = [stamp for stamp, _ in health_rows]

    accepted = 0
    rejected = 0
    aligned: list[dict[str, float]] = []
    for _, message in alignment_messages:
        data = list(message.data)
        if len(data) < 4 or not math.isfinite(float(data[0])):
            continue
        scan_stamp = int(round(float(data[0]) * 1e9))
        is_accepted = float(data[3]) >= 0.5
        accepted += int(is_accepted)
        rejected += int(not is_accepted)
        if not is_accepted:
            continue
        health_index = bisect.bisect_right(health_stamps, scan_stamp) - 1
        truth_match = nearest_causal(scan_stamp, truth_messages, truth_stamps)
        if health_index < 0 or truth_match is None:
            continue
        if scan_stamp - health_stamps[health_index] > MAX_CAUSAL_AGE_NS:
            continue
        _, _, truth_index = truth_match
        truth = truth_messages[truth_index][1]
        truth_pose = pose_of(truth)
        truth_yaw = yaw_from_quaternion(truth_pose.orientation)
        projection = truth_projections[truth_index]
        path_heading = wrap_angle(truth_yaw - projection.heading_error_rad)
        tangent = (math.cos(path_heading), math.sin(path_heading))
        normal = (-tangent[1], tangent[0])
        health = health_rows[health_index][1]
        applied_x, applied_y = float(health[12]), float(health[13])
        aligned.append({
            "along_track_m": applied_x * tangent[0] + applied_y * tangent[1],
            "normal_m": applied_x * normal[0] + applied_y * normal[1],
            "magnitude_m": math.hypot(applied_x, applied_y),
        })

    def values(field: str) -> list[float]:
        return [row[field] for row in aligned]

    def metric(field: str) -> dict[str, float | int | None]:
        selected = values(field)
        return {
            "count": len(selected),
            "mean_signed_m": statistics.fmean(selected) if selected else None,
            "abs_p50_m": percentile([abs(value) for value in selected], 50),
            "abs_p95_m": percentile([abs(value) for value in selected], 95),
            "sum_signed_m": sum(selected) if selected else None,
        }

    return {
        "alignment_reports": len(alignment_messages),
        "accepted_reports": accepted,
        "rejected_reports": rejected,
        "accepted_reports_with_causal_truth_and_health": len(aligned),
        "applied_along_track_correction": metric("along_track_m"),
        "applied_normal_correction": metric("normal_m"),
        "applied_xy_correction_magnitude": metric("magnitude_m"),
        "sign_convention": "positive along-track is forward; positive normal is left",
    }


def align_odom_once(estimate: Any, truth: Any) -> tuple[float, float, float]:
    estimate_pose = pose_of(estimate)
    truth_pose = pose_of(truth)
    rotation = wrap_angle(
        yaw_from_quaternion(truth_pose.orientation) -
        yaw_from_quaternion(estimate_pose.orientation)
    )
    cosine, sine = math.cos(rotation), math.sin(rotation)
    ex = float(estimate_pose.position.x)
    ey = float(estimate_pose.position.y)
    tx = float(truth_pose.position.x)
    ty = float(truth_pose.position.y)
    return tx - cosine * ex + sine * ey, ty - sine * ex - cosine * ey, rotation


def load_stream(
    name: str,
    rows: list[tuple[int, Any]],
    truth_rows: list[tuple[int, Any]],
    truth_stamps: list[int],
    truth_projections: list[Any],
    lap_times: list[int],
    lap_values: list[int],
    run_start_ns: int,
) -> list[dict[str, Any]]:
    valid_rows = [(receipt, msg, stamp_ns(msg)) for receipt, msg in rows]
    valid_rows = [(receipt, msg, stamp) for receipt, msg, stamp in valid_rows if stamp is not None]
    if not valid_rows:
        return []

    transform = (0.0, 0.0, 0.0)
    if POSE_STREAMS[name] == "odom":
        first_match = next(
            (
                (message, matched[0])
                for _, message, stamp in valid_rows
                if (matched := nearest_causal(stamp, truth_rows, truth_stamps)) is not None
            ),
            None,
        )
        if first_match is None:
            return []
        transform = align_odom_once(*first_match)

    result: list[dict[str, Any]] = []
    for receipt_ns, estimate, source_ns in valid_rows:
        matched = nearest_causal(source_ns, truth_rows, truth_stamps)
        if matched is None:
            continue
        truth, truth_age_ns, truth_index = matched
        estimate_pose = pose_of(estimate)
        truth_pose = pose_of(truth)
        estimate_x = float(estimate_pose.position.x)
        estimate_y = float(estimate_pose.position.y)
        estimate_yaw = yaw_from_quaternion(estimate_pose.orientation)
        if POSE_STREAMS[name] == "odom":
            offset_x, offset_y, rotation = transform
            cosine, sine = math.cos(rotation), math.sin(rotation)
            estimate_x, estimate_y = (
                offset_x + cosine * estimate_x - sine * estimate_y,
                offset_y + sine * estimate_x + cosine * estimate_y,
            )
            estimate_yaw = wrap_angle(estimate_yaw + rotation)

        truth_x = float(truth_pose.position.x)
        truth_y = float(truth_pose.position.y)
        truth_yaw = yaw_from_quaternion(truth_pose.orientation)
        projection = truth_projections[truth_index]
        error_x, error_y = estimate_x - truth_x, estimate_y - truth_y
        tangent_x = math.cos(truth_yaw - projection.heading_error_rad)
        tangent_y = math.sin(truth_yaw - projection.heading_error_rad)
        normal_x, normal_y = -tangent_y, tangent_x
        result.append({
            "run_id": "",
            "stream": name,
            "receipt_time_s": (receipt_ns - run_start_ns) / 1e9,
            "source_time_s": source_ns / 1e9,
            "lap_count": lap_at(receipt_ns, lap_times, lap_values),
            "truth_progress_m": projection.s_m,
            "track_segment": projection.segment_index,
            "truth_lateral_offset_m": projection.lateral_offset_m,
            "normal_error_m": error_x * normal_x + error_y * normal_y,
            "tangential_error_m": error_x * tangent_x + error_y * tangent_y,
            "position_error_m": math.hypot(error_x, error_y),
            "yaw_error_rad": wrap_angle(estimate_yaw - truth_yaw),
            "truth_age_ms": truth_age_ns / 1e6,
            "truth_x_m": truth_x,
            "truth_y_m": truth_y,
            "estimate_x_m": estimate_x,
            "estimate_y_m": estimate_y,
            "truth_yaw_rad": truth_yaw,
            "estimate_yaw_rad": estimate_yaw,
        })
    return result


def analyze(bag: Path, trajectory: Path, run_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    track = TrackProjection.from_csv(trajectory, closed=True)
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = {
            name: (int(topic_id), type_name)
            for topic_id, name, type_name in connection.execute(
                "SELECT id,name,type FROM topics"
            )
        }
        required = [TRUTH_ODOM, LAP_COUNT]
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError("bag missing required topic(s): " + ", ".join(missing))

        truth_rows = message_rows(connection, topics, TRUTH_ODOM)
        truth_rows = [
            (stamp_ns(message), (receipt, message))
            for receipt, message in truth_rows
            if stamp_ns(message) is not None
        ]
        truth_rows.sort(key=lambda item: item[0])
        if not truth_rows:
            raise ValueError("simulator truth odometry has no source timestamps")
        truth_stamps = [stamp for stamp, _ in truth_rows]
        truth_messages = [row for _, row in truth_rows]
        truth_projections = []
        previous_segment = None
        for _, (_, truth) in truth_rows:
            truth_pose = pose_of(truth)
            truth_x = float(truth_pose.position.x)
            truth_y = float(truth_pose.position.y)
            truth_yaw = yaw_from_quaternion(truth_pose.orientation)
            if previous_segment is None:
                projection = track.project(truth_x, truth_y, truth_yaw)
            else:
                projection = track.project(
                    truth_x, truth_y, truth_yaw,
                    previous_segment=previous_segment,
                    local_search_radius=64,
                )
                if projection.distance_m > 0.75:
                    projection = track.project(truth_x, truth_y, truth_yaw)
            truth_projections.append(projection)
            previous_segment = projection.segment_index

        lap_rows = message_rows(connection, topics, LAP_COUNT)
        lap_times: list[int] = []
        lap_values: list[int] = []
        last_value: int | None = None
        for receipt, message in lap_rows:
            value = int(message.data)
            if value != last_value:
                lap_times.append(receipt)
                lap_values.append(value)
                last_value = value

        run_start_ns = truth_rows[0][1][0]
        combined: list[dict[str, Any]] = []
        per_stream: dict[str, dict[str, Any]] = {}
        for name in POSE_STREAMS:
            stream = message_rows(connection, topics, name)
            if not stream:
                continue
            scored = load_stream(
                name, stream, truth_messages, truth_stamps, truth_projections,
                lap_times, lap_values, run_start_ns,
            )
            for row in scored:
                row["run_id"] = run_id
            combined.extend(scored)
            per_stream[name] = summarize(scored)

        scan_corrections = summarize_scan_corrections(
            connection, topics, truth_messages, truth_stamps, truth_projections
        )

        by_lap: dict[str, Any] = {}
        for name in POSE_STREAMS:
            stream_rows = [row for row in combined if row["stream"] == name]
            laps = sorted({int(row["lap_count"]) for row in stream_rows})
            by_lap[name] = {
                str(lap): summarize([row for row in stream_rows if row["lap_count"] == lap])
                for lap in laps
            }

        map_stream = per_stream.get("/current_map_pose", {})
        normal_p95 = (map_stream.get("normal_error_m") or {}).get("abs_p95")
        tangent_p95 = (map_stream.get("tangential_error_m") or {}).get("abs_p95")
        if normal_p95 is None or tangent_p95 is None:
            decision = "insufficient data"
        elif normal_p95 > tangent_p95 * 1.25:
            decision = "normal error dominates; localization is a track-width/safety priority"
        elif tangent_p95 > normal_p95 * 1.25:
            decision = "tangential error dominates; along-track localization is a progress/control priority"
        else:
            decision = "normal and tangential localization errors are comparable"

        summary = {
            "schema_version": 1,
            "run_id": run_id,
            "bag": str(bag),
            "trajectory": str(trajectory),
            "trajectory_total_length_m": track.total_length,
            "scoring": {
                "truth_topic": TRUTH_ODOM,
                "truth_matching": "latest source-time truth sample at or before estimate stamp",
                "maximum_truth_age_ms": MAX_CAUSAL_AGE_NS / 1e6,
                "odom_alignment": "single initial SE(2) alignment, held fixed for the run",
            },
            "streams": per_stream,
            "amcl_scan_corrections": scan_corrections,
            "per_lap": by_lap,
            "localization_priority_decision": decision,
            "row_count": len(combined),
        }
        return combined, summary
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    if not args.bag.is_file():
        parser.error(f"bag does not exist: {args.bag}")
    if not args.trajectory.is_file():
        parser.error(f"trajectory does not exist: {args.trajectory}")
    run_id = args.run_id or args.bag.parent.parent.name
    rows, summary = analyze(args.bag, args.trajectory, run_id)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "localization_error.csv"
    json_path = args.output / "localization_error_summary.json"
    fields = [
        "run_id", "stream", "receipt_time_s", "source_time_s", "lap_count",
        "truth_progress_m", "track_segment", "truth_lateral_offset_m",
        "normal_error_m", "tangential_error_m", "position_error_m",
        "yaw_error_rad", "truth_age_ms", "truth_x_m", "truth_y_m",
        "estimate_x_m", "estimate_y_m", "truth_yaw_rad", "estimate_yaw_rad",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"run: {run_id}; trajectory length: {summary['trajectory_total_length_m']:.3f} m")
    for name, metrics in summary["streams"].items():
        normal = metrics["normal_error_m"]["abs_p95"]
        tangent = metrics["tangential_error_m"]["abs_p95"]
        yaw = metrics["yaw_error_rad"]["abs_p95"]
        print(
            f"{name}: n={metrics['sample_count']} normal p95={normal:.4f} m, "
            f"tangential p95={tangent:.4f} m, yaw p95={yaw:.4f} rad"
        )
    print("Decision:", summary["localization_priority_decision"])
    corrections = summary["amcl_scan_corrections"]
    along = corrections["applied_along_track_correction"]
    print(
        "AMCL scan corrections: "
        f"{corrections['accepted_reports_with_causal_truth_and_health']}/"
        f"{corrections['alignment_reports']} aligned accepted scans; "
        f"along-track mean={along['mean_signed_m']} m/scan, "
        f"sum={along['sum_signed_m']} m"
    )
    print(f"Wrote {csv_path} and {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
