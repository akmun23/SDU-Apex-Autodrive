#!/usr/bin/env python3
"""Build a spatial race-engineer report from one saved ROS 2 simulator bag.

The bag is opened read-only. Simulator truth is used only for offline scoring;
this program is not imported or called by any runtime controller component.
The report uses SVG and a small embedded PNG generated with Python's standard
library so it also works in the project ROS image without plotting packages.
"""

from __future__ import annotations

import argparse
import base64
import bisect
import csv
import json
import math
import sqlite3
import statistics
import struct
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - depends on the ROS image
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc

from tools.racing.track_projection import Projection, TrackProjection, wrap_angle


TRUTH = "/autodrive/roboracer_1/odom"
LAP_COUNT = "/autodrive/roboracer_1/lap_count"
LAP_TIME = "/autodrive/roboracer_1/last_lap_time"
COLLISION_COUNT = "/autodrive/roboracer_1/collision_count"
MAP = "/map"
MPC = "/mpc/diagnostics"
POSE_TOPICS = {
    "/amcl_pose": "map",
    "/current_map_pose": "map",
    "/ekf_pose": "map",
    "/odom": "odom",
    "/ekf_odom": "odom",
}
SCALAR_TOPICS = {
    "steering_command": "/autodrive/roboracer_1/steering_command",
    "steering_feedback": "/autodrive/roboracer_1/steering",
    "throttle_command": "/autodrive/roboracer_1/throttle_command",
    "throttle_feedback": "/autodrive/roboracer_1/throttle",
}
PREDICTION_DT_S = 0.025  # Production MPC contract: 40 Hz prediction nodes.
MAX_JOIN_AGE_NS = 60_000_000
SECTOR_COUNT = 8
SPATIAL_BIN_M = 0.10
CAR_HALF_WIDTH_M = 0.1365


@dataclass
class BagRow:
    receipt_ns: int
    stamp_ns: int
    message: Any


def topic_index(connection: sqlite3.Connection) -> dict[str, tuple[int, str]]:
    return {
        name: (int(topic_id), kind)
        for topic_id, name, kind in connection.execute(
            "SELECT id,name,type FROM topics"
        )
    }


def message_stamp(message: Any) -> int | None:
    header = getattr(message, "header", None)
    if header is None:
        return None
    stamp = header.stamp
    result = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return result if result > 0 else None


def read_topic(
    connection: sqlite3.Connection,
    topics: dict[str, tuple[int, str]],
    name: str,
) -> list[BagRow]:
    if name not in topics:
        return []
    topic_id, type_name = topics[name]
    message_type = get_message(type_name)
    result = []
    for receipt, raw in connection.execute(
        "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
        (topic_id,),
    ):
        message = deserialize_message(bytes(raw), message_type)
        result.append(BagRow(int(receipt), message_stamp(message) or int(receipt), message))
    return result


def percentile(values: Iterable[float], p: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = (len(ordered) - 1) * p / 100.0
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "p50": percentile(finite, 50),
        "p95": percentile(finite, 95),
        "max": max(finite) if finite else None,
    }


def yaw_of(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def pose_of(message: Any) -> Any:
    return message.pose.pose


def projected_samples(track: TrackProjection, rows: list[BagRow]) -> list[Projection]:
    projections: list[Projection] = []
    previous: int | None = None
    for row in rows:
        pose = pose_of(row.message)
        x, y, yaw = float(pose.position.x), float(pose.position.y), yaw_of(pose.orientation)
        if previous is None:
            projection = track.project(x, y, yaw)
        else:
            projection = track.project(
                x, y, yaw, previous_segment=previous, local_search_radius=16
            )
            if projection.distance_m > 0.75:
                projection = track.project(x, y, yaw)
        projections.append(projection)
        previous = projection.segment_index
    return projections


def nearest_prior(rows: list[BagRow], stamps: list[int], target_ns: int) -> tuple[BagRow, int] | None:
    index = bisect.bisect_right(stamps, target_ns) - 1
    if index < 0 or target_ns - stamps[index] > MAX_JOIN_AGE_NS:
        return None
    return rows[index], index


def prior_value(rows: list[BagRow], stamps: list[int], target_ns: int) -> float | None:
    found = nearest_prior(rows, stamps, target_ns)
    return float(found[0].message.data) if found else None


def steering_command_rad_at(
    streams: dict[str, list[BagRow]],
    scalar_stamps: dict[str, list[int]],
    target_ns: int,
    steering_limit_rad: float,
) -> float | None:
    """The ROS actuator command is normalized; return its physical angle."""
    normalized = prior_value(
        streams[SCALAR_TOPICS["steering_command"]],
        scalar_stamps["steering_command"],
        target_ns,
    )
    return None if normalized is None else normalized * steering_limit_rad


def reference_speed(projection: Projection, values: list[float], point_count: int) -> float:
    if len(values) != point_count or not values:
        return math.nan
    index = projection.segment_index
    following = (index + 1) % point_count
    return values[index] + projection.segment_fraction * (values[following] - values[index])


def shortest_delta(after: float, before: float, length: float) -> float:
    return (after - before + 0.5 * length) % length - 0.5 * length


def transitions(rows: list[BagRow]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    previous: int | None = None
    for row in rows:
        count = int(row.message.data)
        if count != previous:
            result.append((row.receipt_ns, count))
            previous = count
    return result


def lap_times_by_count(
    lap_events: list[tuple[int, int]], time_rows: list[BagRow]
) -> dict[int, float]:
    unused = set(range(len(time_rows)))
    result: dict[int, float] = {}
    for stamp, count in lap_events:
        if count <= 0 or not unused:
            continue
        index = min(unused, key=lambda candidate: abs(time_rows[candidate].receipt_ns - stamp))
        if abs(time_rows[index].receipt_ns - stamp) <= 500_000_000:
            value = float(time_rows[index].message.data)
            if math.isfinite(value):
                result[count] = value
                unused.remove(index)
    return result


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_map_png(map_message: Any, output: Path) -> None:
    info = map_message.info
    width, height = int(info.width), int(info.height)
    values = list(map_message.data)
    if len(values) != width * height:
        raise ValueError("OccupancyGrid data size does not match its dimensions")
    scanlines = bytearray()
    # OccupancyGrid row zero is the map's bottom row. The SVG scene is rendered
    # in a y-up group, so preserving row order gives the correct orientation.
    for y in range(height):
        scanlines.append(0)
        for value in values[y * width:(y + 1) * width]:
            occupancy = int(value)
            if occupancy < 0:
                color = (150, 150, 150)
            elif occupancy == 0:
                color = (248, 248, 248)
            else:
                shade = max(18, 115 - int(occupancy * 0.85))
                color = (shade, shade, shade)
            scanlines.extend(color)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(bytes(scanlines), 6))
        + png_chunk(b"IEND", b"")
    )
    output.write_bytes(png)


def csv_write(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_points(series: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(row["x"]), float(row["y"]))
        for row in series
        if row.get("x") is not None and row.get("y") is not None
        and math.isfinite(float(row["x"])) and math.isfinite(float(row["y"]))
    ]


def chart_svg(
    output: Path,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    *,
    y_zero: bool = False,
    note: str = "",
) -> None:
    width, height = 1200, 670
    left, right, top, bottom = 94, 1160, 65, 590
    points_by_series = [finite_points(item["points"]) for item in series]
    all_points = [point for points in points_by_series for point in points]
    if not all_points:
        all_points = [(0.0, 0.0), (1.0, 1.0)]
    xmin, xmax = min(p[0] for p in all_points), max(p[0] for p in all_points)
    ymin, ymax = min(p[1] for p in all_points), max(p[1] for p in all_points)
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymax = ymin + 1.0
    if y_zero:
        ymin, ymax = min(0.0, ymin), max(0.0, ymax)
    xpad, ypad = max((xmax - xmin) * 0.025, 1e-9), max((ymax - ymin) * 0.08, 1e-9)
    xmin, xmax, ymin, ymax = xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad
    sx = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    sy = lambda y: bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" font-weight="bold">{title}</text>',
    ]
    for tick in range(6):
        x = xmin + (xmax - xmin) * tick / 5
        y = ymin + (ymax - ymin) * tick / 5
        px, py = sx(x), sy(y)
        out.append(f'<path d="M{px:.2f},{top} V{bottom} M{left},{py:.2f} H{right}" stroke="#e4e7eb"/>')
        out.append(f'<text x="{px:.2f}" y="{bottom + 23}" text-anchor="middle" font-family="sans-serif" font-size="12">{x:.3g}</text>')
        out.append(f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y:.3g}</text>')
    out.append(f'<path d="M{left},{top} V{bottom} H{right}" fill="none" stroke="#333" stroke-width="1.5"/>')
    for item, points in zip(series, points_by_series):
        if not points:
            continue
        stride = max(1, len(points) // 2200)
        selected = points[::stride]
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in selected)
        out.append(f'<polyline points="{coords}" fill="none" stroke="{item["color"]}" stroke-width="1.4" opacity="0.78"/>')
    legend_y = top + 17
    for index, item in enumerate(series):
        if index >= 7:
            out.append(f'<text x="{right - 205}" y="{legend_y + index * 17:.1f}" font-family="sans-serif" font-size="12">+ {len(series) - 7} more runs</text>')
            break
        out.append(f'<path d="M{right - 205},{legend_y + index * 17:.1f} h22" stroke="{item["color"]}" stroke-width="3"/>')
        out.append(f'<text x="{right - 178}" y="{legend_y + 4 + index * 17:.1f}" font-family="sans-serif" font-size="12">{item["label"]}</text>')
    out.append(f'<text x="{(left + right) / 2}" y="645" text-anchor="middle" font-family="sans-serif" font-size="15">{x_label}</text>')
    out.append(f'<text x="20" y="{(top + bottom) / 2}" transform="rotate(-90 20 {(top + bottom) / 2})" text-anchor="middle" font-family="sans-serif" font-size="15">{y_label}</text>')
    if note:
        out.append(f'<text x="{left}" y="{height - 8}" font-family="sans-serif" font-size="12" fill="#555">{note}</text>')
    out.append("</svg>")
    output.write_text("\n".join(out), encoding="utf-8")


def bar_chart_svg(output: Path, title: str, bars: list[tuple[str, float]]) -> None:
    width, height = 1200, 670
    left, right, top, bottom = 280, 1150, 65, 600
    maximum = max((value for _, value in bars), default=1.0)
    maximum = max(maximum, 0.01) * 1.08
    row_height = (bottom - top) / max(1, len(bars))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" font-weight="bold">{title}</text>',
    ]
    for tick in range(6):
        value = maximum * tick / 5
        x = left + (right - left) * tick / 5
        out.append(f'<path d="M{x:.2f},{top} V{bottom}" stroke="#e4e7eb"/>')
        out.append(f'<text x="{x:.2f}" y="{bottom + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.3f}s</text>')
    for index, (label, value) in enumerate(bars):
        y = top + index * row_height + row_height * 0.15
        height_bar = max(2.0, row_height * 0.68)
        bar_width = max(0.0, value / maximum * (right - left))
        out.append(f'<text x="{left - 10}" y="{y + height_bar * 0.76:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>')
        out.append(f'<rect x="{left}" y="{y:.2f}" width="{bar_width:.2f}" height="{height_bar:.2f}" fill="#d94b3d"/>')
    out.append(f'<text x="{(left + right) / 2}" y="650" text-anchor="middle" font-family="sans-serif" font-size="15">Excess sector time over its best observed scored lap</text>')
    out.append("</svg>")
    output.write_text("\n".join(out), encoding="utf-8")


def map_svg(
    output: Path,
    map_png: Path,
    map_message: Any,
    centerline: TrackProjection,
    trajectory: TrackProjection,
    truth_rows: list[BagRow],
    map_pose_rows: list[BagRow],
) -> None:
    info = map_message.info
    ox, oy, yaw, resolution = (
        float(info.origin.position.x), float(info.origin.position.y),
        yaw_of(info.origin.orientation), float(info.resolution),
    )
    map_w, map_h = int(info.width) * resolution, int(info.height) * resolution
    x_values = [ox, ox + map_w] + centerline.x + trajectory.x
    y_values = [oy, oy + map_h] + centerline.y + trajectory.y
    for rows in (truth_rows, map_pose_rows):
        for row in rows:
            pose = pose_of(row.message)
            x_values.append(float(pose.position.x))
            y_values.append(float(pose.position.y))
    xmin, xmax = min(x_values), max(x_values)
    ymin, ymax = min(y_values), max(y_values)
    pad = max(xmax - xmin, ymax - ymin) * 0.05
    xmin, xmax, ymin, ymax = xmin - pad, xmax + pad, ymin - pad, ymax + pad
    total_y = ymin + ymax
    embedded = base64.b64encode(map_png.read_bytes()).decode("ascii")
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="{xmin:.6f} {ymin:.6f} {xmax - xmin:.6f} {ymax - ymin:.6f}">',
        '<rect width="100%" height="100%" fill="#d8dce2"/>',
        f'<g transform="translate(0 {total_y:.8f}) scale(1 -1)">',
        f'<image x="{ox:.8f}" y="{oy:.8f}" width="{map_w:.8f}" height="{map_h:.8f}" href="data:image/png;base64,{embedded}" opacity="0.93"/>',
    ]
    def polyline(xs: list[float], ys: list[float], color: str, width: float, close: bool = False) -> None:
        coordinates = " ".join(f"{x:.5f},{y:.5f}" for x, y in zip(xs, ys))
        if close and xs:
            coordinates += f" {xs[0]:.5f},{ys[0]:.5f}"
        out.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="{width:.4f}"/>')
    polyline(centerline.x, centerline.y, "#fb923c", 0.025, centerline.closed)
    polyline(trajectory.x, trajectory.y, "#1463d9", 0.012, trajectory.closed)
    for rows, color, width in ((truth_rows, "#cf2637", 0.012), (map_pose_rows, "#16a36a", 0.010)):
        polyline([float(pose_of(r.message).position.x) for r in rows],
                 [float(pose_of(r.message).position.y) for r in rows], color, width)
    out.append("</g>")
    out.append(f'<text x="{xmin + pad * 0.5}" y="{ymin + pad * 0.6}" font-family="sans-serif" font-size="0.12">centerline (orange), reference (blue), simulator truth (red), map estimate (green)</text>')
    out.append("</svg>")
    output.write_text("\n".join(out), encoding="utf-8")


def body_values(message: Any) -> tuple[float, float, float]:
    twist = message.twist.twist
    return float(twist.linear.x), float(twist.linear.y), float(twist.angular.z)


def interpolated_truth(
    target_ns: int,
    rows: list[BagRow],
    stamps: list[int],
    projections: list[Projection],
    track: TrackProjection,
) -> dict[str, float] | None:
    upper = bisect.bisect_left(stamps, target_ns)
    if upper == 0 or upper >= len(rows):
        return None
    lower = upper - 1
    before, after = stamps[lower], stamps[upper]
    if target_ns - before > MAX_JOIN_AGE_NS or after - target_ns > MAX_JOIN_AGE_NS or after <= before:
        return None
    ratio = (target_ns - before) / (after - before)
    first, second = rows[lower].message, rows[upper].message
    u0, v0, r0 = body_values(first)
    u1, v1, r1 = body_values(second)
    u, v, r = (u0 + ratio * (u1 - u0), v0 + ratio * (v1 - v0), r0 + ratio * (r1 - r0))
    before_projection, after_projection = projections[lower], projections[upper]
    progress = (
        before_projection.s_m + ratio * shortest_delta(
            after_projection.s_m, before_projection.s_m, track.total_length
        )
    ) % track.total_length
    lateral = before_projection.lateral_offset_m + ratio * (
        after_projection.lateral_offset_m - before_projection.lateral_offset_m
    )
    return {
        "u_mps": u, "v_mps": v, "r_radps": r,
        "progress_m": progress, "lateral_m": lateral,
    }


def align_odom_once(estimate: Any, truth: Any) -> tuple[float, float, float]:
    ep, tp = pose_of(estimate), pose_of(truth)
    rotation = wrap_angle(yaw_of(tp.orientation) - yaw_of(ep.orientation))
    c, s = math.cos(rotation), math.sin(rotation)
    ex, ey = float(ep.position.x), float(ep.position.y)
    tx, ty = float(tp.position.x), float(tp.position.y)
    return tx - c * ex + s * ey, ty - s * ex - c * ey, rotation


def localization_rows(
    pose_streams: dict[str, list[BagRow]],
    truth_rows: list[BagRow],
    truth_stamps: list[int],
    truth_projections: list[Projection],
    run_start_ns: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, rows in pose_streams.items():
        if not rows:
            continue
        source_stamps = [row.stamp_ns for row in rows]
        transform = None
        if POSE_TOPICS[name] == "odom":
            for row in rows:
                match = nearest_prior(truth_rows, truth_stamps, row.stamp_ns)
                if match:
                    transform = align_odom_once(row.message, match[0].message)
                    break
            if transform is None:
                continue
        for row in rows:
            match = nearest_prior(truth_rows, truth_stamps, row.stamp_ns)
            if match is None:
                continue
            truth_row, truth_index = match
            truth_pose = pose_of(truth_row.message)
            estimate_pose = pose_of(row.message)
            ex, ey, eyaw = (float(estimate_pose.position.x), float(estimate_pose.position.y), yaw_of(estimate_pose.orientation))
            if transform is not None:
                tx, ty, rotation = transform
                c, s = math.cos(rotation), math.sin(rotation)
                ex, ey = tx + c * ex - s * ey, ty + s * ex + c * ey
                eyaw = wrap_angle(eyaw + rotation)
            truth_yaw = yaw_of(truth_pose.orientation)
            error_x = ex - float(truth_pose.position.x)
            error_y = ey - float(truth_pose.position.y)
            tangent_yaw = truth_yaw - truth_projections[truth_index].heading_error_rad
            tangent = (math.cos(tangent_yaw), math.sin(tangent_yaw))
            normal = (-tangent[1], tangent[0])
            output.append({
                "stream": name, "time_s": (row.receipt_ns - run_start_ns) / 1e9,
                "source_stamp_ns": row.stamp_ns,
                "truth_progress_m": truth_projections[truth_index].s_m,
                "normal_error_m": error_x * normal[0] + error_y * normal[1],
                "tangential_error_m": error_x * tangent[0] + error_y * tangent[1],
                "position_error_m": math.hypot(error_x, error_y),
                "yaw_error_rad": wrap_angle(eyaw - truth_yaw),
                "truth_age_ms": (row.stamp_ns - truth_stamps[truth_index]) / 1e6,
            })
    return output


def write_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    trajectory = TrackProjection.from_csv(args.trajectory, closed=True)
    if len(args.reference_speed) == len(trajectory.x) + 1:
        args.reference_speed = args.reference_speed[:-1]
    centerline = TrackProjection.from_csv(args.track, closed=True)
    connection = sqlite3.connect(args.bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = topic_index(connection)
        missing = [name for name in (TRUTH, LAP_COUNT) if name not in topics]
        if missing:
            raise ValueError("bag lacks required topic(s): " + ", ".join(missing))
        all_names = {TRUTH, LAP_COUNT, LAP_TIME, COLLISION_COUNT, MAP, MPC, *SCALAR_TOPICS.values(), *POSE_TOPICS}
        streams = {name: read_topic(connection, topics, name) for name in all_names}
    finally:
        connection.close()
    print(f"decoded requested bag topics in {time.monotonic() - started:.1f}s", flush=True)

    truth_rows = streams[TRUTH]
    truth_rows = [row for row in truth_rows if message_stamp(row.message) is not None]
    truth_rows.sort(key=lambda row: row.stamp_ns)
    for rows in streams.values():
        rows.sort(key=lambda row: row.stamp_ns)
    if len(truth_rows) < 2:
        raise ValueError("bag needs at least two truth odometry messages")
    truth_stamps = [row.stamp_ns for row in truth_rows]
    truth_projection = projected_samples(trajectory, truth_rows)
    center_projection = projected_samples(centerline, truth_rows)
    print(f"projected {len(truth_rows)} truth samples", flush=True)
    lap_events = transitions(streams[LAP_COUNT])
    scored_times = lap_times_by_count(lap_events, streams[LAP_TIME])
    collision_values = [int(row.message.data) for row in streams[COLLISION_COUNT]]
    collision_delta = max(collision_values) - min(collision_values) if collision_values else None
    map_message = streams[MAP][0].message if streams[MAP] else None

    scalar_stamps: dict[str, list[int]] = {}
    for key, topic in SCALAR_TOPICS.items():
        scalar_stamps[key] = [row.stamp_ns for row in streams[topic]]
    map_pose_rows = streams["/current_map_pose"]
    map_pose_stamps = [row.stamp_ns for row in map_pose_rows]
    diagnostics = []
    for row in streams[MPC]:
        try:
            data = json.loads(row.message.data)
        except (TypeError, json.JSONDecodeError):
            continue
        source_ns = int(data.get("control_ros_stamp_ns") or data.get("source_stamp_ns") or row.stamp_ns)
        diagnostics.append((row, source_ns, data))

    run_start_ns = truth_rows[0].receipt_ns
    progress_unwrapped = [truth_projection[0].s_m]
    for before, after in zip(truth_projection, truth_projection[1:]):
        progress_unwrapped.append(
            progress_unwrapped[-1] + shortest_delta(after.s_m, before.s_m, trajectory.total_length)
        )
    receipt_times = [row.receipt_ns for row in truth_rows]
    lap_event_stamps = [event[0] for event in lap_events]
    lap_intervals: list[tuple[int, int, int]] = []
    previous_lap_event_ns = run_start_ns
    for event_ns, completed_lap_count in lap_events:
        if completed_lap_count > 0:
            lap_intervals.append((completed_lap_count, previous_lap_event_ns, event_ns))
        previous_lap_event_ns = event_ns
    lap_interval_ends = [interval[2] for interval in lap_intervals]

    def unwrapped_at(receipt_ns: int) -> float:
        i = bisect.bisect_right(receipt_times, receipt_ns) - 1
        return progress_unwrapped[max(0, min(i, len(progress_unwrapped) - 1))]

    events_with_origin = [(stamp, count, unwrapped_at(stamp)) for stamp, count in lap_events]
    origins_by_count = {count: origin for _, count, origin in events_with_origin}
    def lap_at(receipt_ns: int) -> int:
        i = bisect.bisect_right(lap_event_stamps, receipt_ns) - 1
        return lap_events[i][1] if i >= 0 else -1

    def completed_lap_at(receipt_ns: int) -> int | None:
        """Return the lap whose start/end transitions contain this sample."""
        i = bisect.bisect_right(lap_interval_ends, receipt_ns)
        if i >= len(lap_intervals):
            return None
        count, start_ns, end_ns = lap_intervals[i]
        return count if start_ns <= receipt_ns < end_ns else None

    truth_metrics: list[dict[str, Any]] = []
    for index, (row, ref, center, unwrapped) in enumerate(zip(truth_rows, truth_projection, center_projection, progress_unwrapped)):
        u, v, yaw_rate = body_values(row.message)
        reference_speed_mps = reference_speed(ref, args.reference_speed, len(trajectory.x))
        reference_curvature = ref.curvature_inv_m
        following = (ref.segment_index + 1) % len(trajectory.x)
        ref_x = trajectory.x[ref.segment_index] + ref.segment_fraction * (trajectory.x[following] - trajectory.x[ref.segment_index])
        ref_y = trajectory.y[ref.segment_index] + ref.segment_fraction * (trajectory.y[following] - trajectory.y[ref.segment_index])
        ref_heading = math.atan2(
            trajectory.y[following] - trajectory.y[ref.segment_index],
            trajectory.x[following] - trajectory.x[ref.segment_index],
        )
        reference_center_projection = centerline.project(
            ref_x, ref_y, ref_heading,
            previous_segment=center.segment_index, local_search_radius=16,
        )
        if reference_center_projection.distance_m > 0.75:
            reference_center_projection = centerline.project(ref_x, ref_y, ref_heading)
        left_clearance = center.left_width_m - center.lateral_offset_m - CAR_HALF_WIDTH_M
        right_clearance = center.right_width_m + center.lateral_offset_m - CAR_HALF_WIDTH_M
        wall_clearance = min(left_clearance, right_clearance) if math.isfinite(left_clearance) and math.isfinite(right_clearance) else math.nan
        if 0 < index < len(truth_rows) - 1:
            prev_dt = (row.stamp_ns - truth_rows[index - 1].stamp_ns) / 1e9
            next_dt = (truth_rows[index + 1].stamp_ns - row.stamp_ns) / 1e9
            if 0 < prev_dt <= 0.1 and 0 < next_dt <= 0.1:
                u0, v0, _ = body_values(truth_rows[index - 1].message)
                u1, v1, _ = body_values(truth_rows[index + 1].message)
                dt = prev_dt + next_dt
                ax = (u1 - u0) / dt - v * yaw_rate
                ay_body = (v1 - v0) / dt + u * yaw_rate
            else:
                ax = ay_body = math.nan
        else:
            ax = ay_body = math.nan
        ay_proxy = u * yaw_rate
        lap_count = lap_at(row.receipt_ns)
        lap_origin = origins_by_count.get(lap_count)
        lap_progress = unwrapped - lap_origin if lap_origin is not None else math.nan
        ref_error = ref.lateral_offset_m
        estimate_cte = None
        joined_map = nearest_prior(map_pose_rows, map_pose_stamps, row.stamp_ns) if map_pose_rows else None
        if joined_map:
            estimate_pose = pose_of(joined_map[0].message)
            estimate_projection = trajectory.project(
                float(estimate_pose.position.x), float(estimate_pose.position.y),
                yaw_of(estimate_pose.orientation), previous_segment=ref.segment_index,
                local_search_radius=16,
            )
            if estimate_projection.distance_m > 0.75:
                estimate_projection = trajectory.project(
                    float(estimate_pose.position.x), float(estimate_pose.position.y),
                    yaw_of(estimate_pose.orientation),
                )
            estimate_cte = estimate_projection.lateral_offset_m
        truth_metrics.append({
            "run_id": args.run_id, "time_s": (row.receipt_ns - run_start_ns) / 1e9,
            "source_time_s": row.stamp_ns / 1e9, "lap_count": lap_count,
            "lap_number": completed_lap_at(row.receipt_ns),
            "lap_progress_m": lap_progress, "s_m": ref.s_m,
            "x_m": float(pose_of(row.message).position.x), "y_m": float(pose_of(row.message).position.y),
            "truth_speed_mps": u, "truth_lateral_speed_mps": v,
            "truth_yaw_rad": yaw_of(pose_of(row.message).orientation),
            "truth_heading_error_rad": ref.heading_error_rad,
            "reference_speed_mps": reference_speed_mps, "speed_error_mps": u - reference_speed_mps if math.isfinite(reference_speed_mps) else math.nan,
            "ax_mps2": ax, "ay_body_mps2": ay_body, "ay_proxy_mps2": ay_proxy,
            "yaw_rate_radps": yaw_rate, "actual_curvature_inv_m": yaw_rate / max(abs(u), 0.25),
            "reference_curvature_inv_m": reference_curvature,
            "truth_lateral_offset_m": ref.lateral_offset_m,
            "truth_track_lateral_offset_m": center.lateral_offset_m,
            "reference_lateral_offset_m": reference_center_projection.lateral_offset_m,
            "estimated_lateral_offset_m": estimate_cte,
            "tracking_error_m": ref_error,
            "localization_normal_error_m": None,
            "steering_command_rad": steering_command_rad_at(
                streams, scalar_stamps, row.stamp_ns,
                args.steering_limit_rad),
            "steering_feedback_rad": prior_value(streams[SCALAR_TOPICS["steering_feedback"]], scalar_stamps["steering_feedback"], row.stamp_ns),
            "throttle_command": prior_value(streams[SCALAR_TOPICS["throttle_command"]], scalar_stamps["throttle_command"], row.stamp_ns),
            "throttle_feedback": prior_value(streams[SCALAR_TOPICS["throttle_feedback"]], scalar_stamps["throttle_feedback"], row.stamp_ns),
            "left_wall_clearance_m": left_clearance, "right_wall_clearance_m": right_clearance,
            "minimum_wall_clearance_m": wall_clearance,
            "track_segment": ref.segment_index,
        })

    # Localization is evaluated against truth at the estimator source stamp;
    # odometry-only frames receive one fixed initial SE(2) alignment.
    localization = localization_rows(
        {name: streams[name] for name in POSE_TOPICS},
        truth_rows, truth_stamps, truth_projection, run_start_ns,
    )
    map_errors = [row for row in localization if row["stream"] == "/current_map_pose"]
    map_errors.sort(key=lambda row: int(row["source_stamp_ns"]))
    map_error_stamps = [int(row["source_stamp_ns"]) for row in map_errors]
    for row in truth_metrics:
        target_stamp = int(row["source_time_s"] * 1e9)
        match_index = bisect.bisect_right(map_error_stamps, target_stamp) - 1
        if match_index >= 0 and target_stamp - map_error_stamps[match_index] <= MAX_JOIN_AGE_NS:
            row["localization_normal_error_m"] = map_errors[match_index]["normal_error_m"]
    print(f"scored state and localization streams in {time.monotonic() - started:.1f}s", flush=True)

    # First prediction node is exactly one documented MPC step (25 ms). Compare
    # it to interpolated simulator truth at that future stamp, never to a later
    # estimator measurement. This is offline model scoring only.
    model_rows: list[dict[str, Any]] = []
    for _, start_ns, data in diagnostics:
        prediction_list = data.get("predictions") or []
        first = next((item for item in prediction_list if int(item.get("n", -1)) == 1), None)
        if not first:
            continue
        future_ns = start_ns + int(PREDICTION_DT_S * 1e9)
        actual = interpolated_truth(future_ns, truth_rows, truth_stamps, truth_projection, trajectory)
        state = first.get("state") or []
        if actual is None or len(state) < 5:
            continue
        predicted_s = float(first.get("s_m", math.nan))
        predicted_ey = float(state[0])
        model_rows.append({
            "run_id": args.run_id, "time_s": (start_ns - run_start_ns) / 1e9,
            "progress_m": float(data.get("progress_m", math.nan)),
            "predicted_s_m": predicted_s, "actual_s_m": actual["progress_m"],
            "progress_error_m": shortest_delta(predicted_s, actual["progress_m"], trajectory.total_length),
            "predicted_e_y_m": predicted_ey, "actual_e_y_m": actual["lateral_m"],
            "lateral_error_m": predicted_ey - actual["lateral_m"],
            "predicted_u_mps": float(state[2]), "actual_u_mps": actual["u_mps"],
            "speed_error_mps": float(state[2]) - actual["u_mps"],
            "predicted_r_radps": float(state[4]), "actual_r_radps": actual["r_radps"],
            "yaw_rate_error_radps": float(state[4]) - actual["r_radps"],
            "status": data.get("status", ""),
        })

    sector_rows: list[dict[str, Any]] = []
    sector_length = trajectory.total_length / SECTOR_COUNT
    for lap_count, start_ns, end_ns in lap_intervals:
        if not 2 <= lap_count <= 11:
            continue
        origin = unwrapped_at(start_ns)
        projected_lap_length = unwrapped_at(end_ns) - origin
        if projected_lap_length <= 0.5 * trajectory.total_length:
            continue
        lap_samples = [
            row for row in truth_metrics
            if row["lap_number"] == lap_count
            and start_ns <= run_start_ns + int(row["time_s"] * 1e9) < end_ns
        ]
        if not lap_samples:
            continue
        # Add lap-start as an interpolation point. Captured telemetry is 40 Hz;
        # each sector crossing is linearly interpolated between adjacent samples.
        points = [(start_ns, 0.0, None)]
        for row in lap_samples:
            stamp = run_start_ns + int(row["time_s"] * 1e9)
            progress = (unwrapped_at(stamp) - origin) * trajectory.total_length / projected_lap_length
            progress = min(trajectory.total_length, max(0.0, progress))
            points.append((stamp, progress, row))
        # The lap counter transition defines the timing endpoint. Include it as
        # the exact loop-length endpoint so a few centimetres of projection
        # offset cannot drop the final spatial sector.
        points.append((end_ns, trajectory.total_length, None))
        points.sort(key=lambda point: point[0])
        crossings: list[tuple[int, dict[str, Any] | None, dict[str, Any] | None]] = []
        for sector in range(SECTOR_COUNT + 1):
            target = sector * sector_length
            crossing = None
            for first, second in zip(points, points[1:]):
                if first[1] <= target <= second[1] and second[1] > first[1]:
                    fraction = (target - first[1]) / (second[1] - first[1])
                    crossing = (int(first[0] + fraction * (second[0] - first[0])), first[2], second[2])
                    break
            if crossing:
                crossings.append(crossing)
            else:
                crossings.append((-1, None, None))
        for sector in range(SECTOR_COUNT):
            ta, _, _ = crossings[sector]
            tb, _, _ = crossings[sector + 1]
            if ta < 0 or tb <= ta:
                continue
            samples = [point[2] for point in points if ta <= point[0] <= tb and point[2] is not None]
            if not samples:
                continue
            valid_speed = [float(row["truth_speed_mps"]) for row in samples]
            clearance = [float(row["minimum_wall_clearance_m"]) for row in samples if math.isfinite(float(row["minimum_wall_clearance_m"]))]
            lateral = [float(row["tracking_error_m"]) for row in samples]
            speed_errors = [abs(float(row["speed_error_mps"])) for row in samples if math.isfinite(float(row["speed_error_mps"]))]
            loc = [abs(float(row["localization_normal_error_m"])) for row in samples if row["localization_normal_error_m"] is not None]
            sector_rows.append({
                "run_id": args.run_id, "lap_count": lap_count, "sector": sector + 1,
                "s_start_m": sector * sector_length, "s_end_m": (sector + 1) * sector_length,
                "sector_time_s": (tb - ta) / 1e9, "time_loss_vs_best_s": None,
                "projected_lap_length_m": projected_lap_length,
                "entry_speed_mps": valid_speed[0], "minimum_speed_mps": min(valid_speed),
                "exit_speed_mps": valid_speed[-1], "peak_abs_steering_rad": max(
                    [abs(float(row["steering_feedback_rad"])) for row in samples if row["steering_feedback_rad"] is not None] or [math.nan]
                ), "peak_lateral_accel_mps2": max(abs(float(row["ay_proxy_mps2"])) for row in samples),
                "max_abs_cte_m": max(abs(value) for value in lateral),
                "minimum_wall_clearance_m": min(clearance) if clearance else math.nan,
                "localization_normal_p95_m": percentile(loc, 95),
                "speed_error_p95_mps": percentile(speed_errors, 95),
                "samples": len(samples),
            })
    best_sector: dict[int, float] = {}
    for row in sector_rows:
        if 2 <= int(row["lap_count"]) <= 11:
            sector = int(row["sector"])
            best_sector[sector] = min(best_sector.get(sector, math.inf), float(row["sector_time_s"]))
    for row in sector_rows:
        if int(row["sector"]) in best_sector:
            row["time_loss_vs_best_s"] = float(row["sector_time_s"]) - best_sector[int(row["sector"])]
    sector_sums = {
        lap: sum(float(row["sector_time_s"]) for row in sector_rows if int(row["lap_count"]) == lap)
        for lap in range(2, 12)
        if sum(int(row["lap_count"]) == lap for row in sector_rows) == SECTOR_COUNT
    }
    sector_timer_deltas = [abs(sector_sums[lap] - scored_times[lap]) for lap in sector_sums if lap in scored_times]

    spatial_buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in truth_metrics:
        if row["lap_number"] is not None and math.isfinite(float(row["lap_progress_m"])):
            bin_index = int(max(0.0, float(row["lap_progress_m"])) / SPATIAL_BIN_M)
            spatial_buckets[(int(row["lap_number"]), bin_index)].append(row)
    spatial_rows = []
    for (lap, bin_index), samples in sorted(spatial_buckets.items()):
        def mean(field: str) -> float:
            vals = [float(item[field]) for item in samples if item[field] is not None and math.isfinite(float(item[field]))]
            return statistics.fmean(vals) if vals else math.nan
        spatial_rows.append({
            "run_id": args.run_id, "lap_count": lap, "s_bin_m": bin_index * SPATIAL_BIN_M,
            "sample_count": len(samples),
            "time_spent_s": max(float(item["time_s"]) for item in samples) - min(float(item["time_s"]) for item in samples),
            "speed_mps": mean("truth_speed_mps"), "reference_speed_mps": mean("reference_speed_mps"),
            "speed_error_mps": mean("speed_error_mps"), "ax_mps2": mean("ax_mps2"),
            "ay_proxy_mps2": mean("ay_proxy_mps2"), "ay_body_mps2": mean("ay_body_mps2"),
            "reference_curvature_inv_m": mean("reference_curvature_inv_m"),
            "actual_yaw_rate_radps": mean("yaw_rate_radps"),
            "actual_curvature_inv_m": mean("actual_curvature_inv_m"),
            "tracking_error_m": mean("tracking_error_m"),
            "truth_track_lateral_offset_m": mean("truth_track_lateral_offset_m"),
            "reference_lateral_offset_m": mean("reference_lateral_offset_m"),
            "localization_normal_error_m": mean("localization_normal_error_m"),
            "steering_command_rad": mean("steering_command_rad"),
            "steering_feedback_rad": mean("steering_feedback_rad"),
            "minimum_wall_clearance_m": min((float(item["minimum_wall_clearance_m"]) for item in samples if math.isfinite(float(item["minimum_wall_clearance_m"]))), default=math.nan),
        })

    saturation_rows = []
    for row, source_ns, data in diagnostics:
        action = data.get("first_action") or []
        state = data.get("state") or []
        command_stamp = int(data.get("control_ros_stamp_ns") or source_ns)
        steering_cmd = steering_command_rad_at(
            streams, scalar_stamps, command_stamp, args.steering_limit_rad)
        steering_actual = prior_value(streams[SCALAR_TOPICS["steering_feedback"]], scalar_stamps["steering_feedback"], command_stamp)
        throttle_cmd = prior_value(streams[SCALAR_TOPICS["throttle_command"]], scalar_stamps["throttle_command"], command_stamp)
        throttle_actual = prior_value(streams[SCALAR_TOPICS["throttle_feedback"]], scalar_stamps["throttle_feedback"], command_stamp)
        saturation_rows.append({
            "run_id": args.run_id, "time_s": (row.receipt_ns - run_start_ns) / 1e9,
            "lap_number": completed_lap_at(row.receipt_ns),
            "progress_m": data.get("progress_m"), "status": data.get("status"),
            "solver_iterations": (data.get("solver") or {}).get("iterations"),
            "solver_us": (data.get("solver") or {}).get("solve_us"),
            "first_action_steering_rad": action[0] if len(action) > 0 else None,
            "first_action_speed_mps": action[1] if len(action) > 1 else None,
            "first_control_steering_rate_radps": action[2] if len(action) > 2 else None,
            "first_control_target_speed_rate_mps2": action[3] if len(action) > 3 else None,
            "state_target_speed_mps": state[5] if len(state) > 5 else None,
            "state_steering_command_rad": state[6] if len(state) > 6 else None,
            "state_delayed_steering_command_1_rad": state[7] if len(state) > 7 else None,
            "state_delayed_steering_command_2_rad": state[8] if len(state) > 8 else None,
            "steering_command_rad": steering_cmd, "steering_feedback_rad": steering_actual,
            "steering_limit_fraction": abs(steering_actual) / args.steering_limit_rad if steering_actual is not None else None,
            "steering_near_configured_limit": bool(steering_actual is not None and abs(steering_actual) >= 0.98 * args.steering_limit_rad),
            "first_action_steering_near_limit": bool(len(action) > 0 and abs(float(action[0])) >= 0.98 * args.steering_limit_rad),
            "steering_command_feedback_delta_rad": (steering_cmd - steering_actual) if steering_cmd is not None and steering_actual is not None else None,
            "throttle_command": throttle_cmd, "throttle_feedback": throttle_actual,
            "throttle_near_full_command": bool(throttle_cmd is not None and 0.98 <= abs(throttle_cmd) <= 1.02),
            "rejection_speed_guard": data.get("rejection_speed_guard_applied"),
            "corridor_repair": data.get("corridor_repair_used"),
            "best_effort_action": data.get("best_effort_action_published"),
            "rti2_triggered": data.get("rti2_triggered"),
            "rti2_budget_skipped": data.get("rti2_budget_skipped"),
            "corridor_slack_m": data.get("minimum_predicted_corridor_slack_m"),
            "recovery_active": data.get("recovery_active"),
        })

    outdir = args.output
    outdir.mkdir(parents=True, exist_ok=True)
    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    fields = list(truth_metrics[0].keys()) if truth_metrics else []
    csv_write(outdir / "tracking_error.csv", fields, truth_metrics)
    csv_write(outdir / "localization_error.csv", list(localization[0].keys()) if localization else ["stream"], localization)
    csv_write(outdir / "controller_saturation.csv", list(saturation_rows[0].keys()) if saturation_rows else ["run_id"], saturation_rows)
    csv_write(outdir / "sector_metrics.csv", list(sector_rows[0].keys()) if sector_rows else ["run_id"], sector_rows)
    csv_write(outdir / "spatial_metrics.csv", list(spatial_rows[0].keys()) if spatial_rows else ["run_id"], spatial_rows)
    csv_write(outdir / "model_prediction_error.csv", list(model_rows[0].keys()) if model_rows else ["run_id"], model_rows)

    lap_rows = []
    for event_index, (stamp, count) in enumerate(lap_events):
        lap_rows.append({
            "run_id": args.run_id, "lap_count": count,
            "lap_role": "warmup" if count == 1 else "scored" if 2 <= count <= 11 else "extra" if count == 12 else "other",
            "lap_time_s": scored_times.get(count),
            "transition_time_s": (stamp - run_start_ns) / 1e9,
            "complete": count in scored_times,
            "collision_delta": collision_delta,
        })
    csv_write(outdir / "lap_times.csv", ["run_id", "lap_count", "lap_role", "lap_time_s", "transition_time_s", "complete", "collision_delta"], lap_rows)

    # Reports and all ten handoff plots. Lines are kept run-separated so the
    # spread between racing laps remains visible instead of being averaged away.
    lap_series: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in truth_metrics:
        if row["lap_number"] is not None:
            lap_series[int(row["lap_number"])].append(row)
    colors = ["#1769aa", "#d94b3d", "#168a5b", "#8a56ac", "#db8c14", "#177e89", "#bb4777", "#657b36", "#735b4a", "#405a9a"]
    def by_lap(yfield: str, *, xfield: str = "lap_progress_m", laps: Iterable[int] | None = None) -> list[dict[str, Any]]:
        selected = sorted(laps if laps is not None else lap_series)
        return [{"label": f"lap {lap}", "color": colors[i % len(colors)], "points": [
            {"x": item[xfield], "y": item[yfield]} for item in lap_series[lap]
        ]} for i, lap in enumerate(selected)]

    if map_message:
        png_path = plots / "map.png"
        write_map_png(map_message, png_path)
        map_svg(plots / "01_map_paths.svg", png_path, map_message, centerline, trajectory, truth_rows, map_pose_rows)
    else:
        chart_svg(plots / "01_map_paths.svg", "Map and paths (map topic missing)", "x", "y", [])
    chart_svg(plots / "02_speed_vs_s.svg", "Actual and reference speed vs path distance", "Lap-relative distance (m)", "Speed (m/s)", by_lap("truth_speed_mps") + by_lap("reference_speed_mps"))
    chart_svg(plots / "03_cte_vs_s.svg", "Truth cross-track error vs path distance", "Lap-relative distance (m)", "Cross-track error (m)", by_lap("tracking_error_m"), y_zero=True)
    loc_series = []
    for index, name in enumerate(POSE_TOPICS):
        samples = [item for item in localization if item["stream"] == name]
        if samples:
            loc_series.append({"label": name, "color": colors[index], "points": [{"x": item["truth_progress_m"], "y": item["normal_error_m"]} for item in samples]})
    chart_svg(plots / "04_localization_error.svg", "Localization normal error vs track distance", "Track distance s (m)", "Normal error (m)", loc_series, y_zero=True)
    steering_cmd_rows = streams[SCALAR_TOPICS["steering_command"]]
    steering_actual_rows = streams[SCALAR_TOPICS["steering_feedback"]]
    steering_series = []
    for label, rows, color, scale in (
        ("command", steering_cmd_rows, "#1769aa", args.steering_limit_rad),
        ("feedback", steering_actual_rows, "#d94b3d", 1.0),
    ):
        points = []
        for row in rows:
            i = bisect.bisect_right(truth_stamps, row.stamp_ns) - 1
            if i >= 0:
                points.append({"x": truth_projection[i].s_m,
                               "y": float(row.message.data) * scale})
        steering_series.append({"label": label, "color": color, "points": points})
    chart_svg(plots / "05_steering_vs_s.svg", "Steering command and feedback", "Track distance s (m)", "Steering (rad)", steering_series, y_zero=True)
    yaw_series = [
        {"label": "sim yaw rate", "color": "#1769aa", "points": [{"x": row["progress_m"], "y": row["actual_r_radps"]} for row in model_rows]},
        {"label": "MPC 25 ms prediction", "color": "#d94b3d", "points": [{"x": row["progress_m"], "y": row["predicted_r_radps"]} for row in model_rows]},
        {"label": "prediction error", "color": "#8a56ac", "points": [{"x": row["progress_m"], "y": row["yaw_rate_error_radps"]} for row in model_rows]},
    ]
    chart_svg(plots / "06_yaw_model_residual.svg", "One-step yaw-rate prediction vs simulator", "MPC progress (m)", "Yaw rate (rad/s)", yaw_series, y_zero=True, note="Only accepted MPC rows with a causally matched +25 ms truth sample are included.")
    chart_svg(plots / "07_lateral_accel_vs_envelope.svg", "Lateral acceleration observations by speed", "Truth speed (m/s)", "u·r (m/s²)", [{"label": f"lap {lap}", "color": colors[i % len(colors)], "points": [{"x": item["truth_speed_mps"], "y": item["ay_proxy_mps2"]} for item in rows]} for i, (lap, rows) in enumerate(sorted(lap_series.items()))], y_zero=True, note="Observation-only: no validated empirical capability envelope has been fitted yet.")
    chart_svg(plots / "08_ggv_scatter.svg", "Longitudinal/lateral acceleration observations", "Body longitudinal acceleration ax (m/s²)", "Lateral acceleration u·r (m/s²)", [{"label": f"lap {lap}", "color": colors[i % len(colors)], "points": [{"x": item["ax_mps2"], "y": item["ay_proxy_mps2"]} for item in rows]} for i, (lap, rows) in enumerate(sorted(lap_series.items()))], y_zero=True, note="Acceleration derivatives are omitted at gaps over 100 ms; no fitted boundary is shown.")
    chart_svg(plots / "09_wall_clearance.svg", "Estimated vehicle-body clearance to track edges", "Lap-relative distance (m)", "Clearance (m)", by_lap("minimum_wall_clearance_m"), y_zero=True, note=f"Centerline widths minus {CAR_HALF_WIDTH_M:.4f} m nominal half-width; negative means edge overlap.")
    loss_bars = sorted([row for row in sector_rows if row["time_loss_vs_best_s"] is not None], key=lambda row: float(row["time_loss_vs_best_s"]), reverse=True)
    bar_chart_svg(
        plots / "10_sector_time_loss.svg",
        "Largest per-sector time losses vs best observed scored lap",
        [(f"lap {row['lap_count']} / sector {row['sector']}", float(row["time_loss_vs_best_s"])) for row in loss_bars[:20]],
    )

    scored_lap_values = [value for count, value in scored_times.items() if 2 <= count <= 11]
    mpc_status = defaultdict(int)
    for _, _, data in diagnostics:
        mpc_status[str(data.get("status", "unknown"))] += 1
    scored_truth = [
        row for row in truth_metrics
        if row["lap_number"] is not None and 2 <= int(row["lap_number"]) <= 11
    ]
    scored_localization = [
        row for row in localization
        if (completed_lap_at(run_start_ns + int(float(row["time_s"]) * 1e9)) is not None
            and 2 <= completed_lap_at(run_start_ns + int(float(row["time_s"]) * 1e9)) <= 11)
    ]
    actual_speeds = [float(row["truth_speed_mps"]) for row in truth_metrics]
    scored_speeds = [float(row["truth_speed_mps"]) for row in scored_truth]
    ctes = [abs(float(row["tracking_error_m"])) for row in truth_metrics]
    scored_ctes = [abs(float(row["tracking_error_m"])) for row in scored_truth]
    clearances = [float(row["minimum_wall_clearance_m"]) for row in truth_metrics if math.isfinite(float(row["minimum_wall_clearance_m"]))]
    scored_clearances = [float(row["minimum_wall_clearance_m"]) for row in scored_truth if math.isfinite(float(row["minimum_wall_clearance_m"]))]
    normal_errors = [abs(float(row["normal_error_m"])) for row in localization if row["stream"] == "/current_map_pose"]
    scored_map_localization = [row for row in scored_localization if row["stream"] == "/current_map_pose"]
    yaw_residuals = [abs(float(row["yaw_rate_error_radps"])) for row in model_rows]
    speed_residuals = [abs(float(row["speed_error_mps"])) for row in model_rows]
    summary = {
        "schema_version": 1, "run_id": args.run_id, "bag": str(args.bag),
        "trajectory": str(args.trajectory), "track": str(args.track),
        "scoring": {"truth_topic": TRUTH, "truth_used_only_offline": True,
                    "mpc_one_step_horizon_s": PREDICTION_DT_S,
                    "lap_sector_count": SECTOR_COUNT, "spatial_bin_m": SPATIAL_BIN_M},
        "lap_count_final": lap_events[-1][1] if lap_events else 0,
        "lap_times_s": {str(count): value for count, value in scored_times.items()},
        "scored_laps": len(scored_lap_values),
        "best_lap_s": min(scored_lap_values) if scored_lap_values else None,
        "median_lap_s": statistics.median(scored_lap_values) if scored_lap_values else None,
        "mean_lap_s": statistics.fmean(scored_lap_values) if scored_lap_values else None,
        "lap_stdev_s": statistics.stdev(scored_lap_values) if len(scored_lap_values) > 1 else None,
        "collision_delta": collision_delta,
        "lap_length_m": trajectory.total_length,
        "truth_speed_mps": stats(actual_speeds),
        "absolute_tracking_error_m": stats(ctes),
        "map_localization_normal_error_m": stats(normal_errors),
        "map_localization_tangential_error_m": stats(
            abs(float(row["tangential_error_m"])) for row in localization if row["stream"] == "/current_map_pose"
        ),
        "map_localization_yaw_error_rad": stats(
            abs(float(row["yaw_error_rad"])) for row in localization if row["stream"] == "/current_map_pose"
        ),
        "absolute_speed_tracking_error_mps": stats(
            abs(float(row["speed_error_mps"])) for row in truth_metrics if math.isfinite(float(row["speed_error_mps"]))
        ),
        "minimum_wall_clearance_m": min(clearances) if clearances else None,
        "sector_sum_vs_reported_lap_time_abs_error_s": stats(sector_timer_deltas),
        "scored_lap_metrics": {
            "truth_speed_mps": stats(scored_speeds),
            "absolute_tracking_error_m": stats(scored_ctes),
            "absolute_speed_error_mps": stats(
                abs(float(row["speed_error_mps"])) for row in scored_truth
                if math.isfinite(float(row["speed_error_mps"]))
            ),
            "map_localization_normal_error_m": stats(
                abs(float(row["normal_error_m"])) for row in scored_map_localization
            ),
            "map_localization_tangential_error_m": stats(
                abs(float(row["tangential_error_m"])) for row in scored_map_localization
            ),
            "map_localization_yaw_error_rad": stats(
                abs(float(row["yaw_error_rad"])) for row in scored_map_localization
            ),
            "minimum_wall_clearance_m": min(scored_clearances) if scored_clearances else None,
            "truth_sample_count": len(scored_truth),
        },
        "one_step_model_error": {
            "sample_count": len(model_rows), "abs_speed_p95_mps": percentile(speed_residuals, 95),
            "abs_yaw_rate_p95_radps": percentile(yaw_residuals, 95),
            "abs_lateral_error_p95_m": percentile([abs(float(row["lateral_error_m"])) for row in model_rows], 95),
            "abs_progress_error_p95_m": percentile([abs(float(row["progress_error_m"])) for row in model_rows], 95),
        },
        "mpc_status_counts": dict(mpc_status),
        "data_coverage": {
            "truth_samples": len(truth_rows), "map_pose_samples": len(streams["/current_map_pose"]),
            "mpc_diagnostics": len(diagnostics), "one_step_model_matches": len(model_rows),
            "localization_streams": sorted({row["stream"] for row in localization}),
            "track_width_available": all(math.isfinite(value) for value in centerline.left_width + centerline.right_width),
        },
        "sector_count": len(sector_rows), "spatial_bin_count": len(spatial_rows),
        "collision_topics_recorded": bool(streams[COLLISION_COUNT]),
        "report_files": ["summary.md", "summary.json", "lap_times.csv", "sector_metrics.csv",
                         "localization_error.csv", "tracking_error.csv", "controller_saturation.csv",
                         "spatial_metrics.csv", "model_prediction_error.csv", "plots/"],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    top_losses = loss_bars[:5]
    top_safety = sorted(sector_rows, key=lambda row: float(row["minimum_wall_clearance_m"]) if row["minimum_wall_clearance_m"] is not None else math.inf)[:5]
    top_model = sorted(model_rows, key=lambda row: abs(float(row["yaw_rate_error_radps"])) + abs(float(row["speed_error_mps"])) if math.isfinite(float(row["yaw_rate_error_radps"])) else -math.inf, reverse=True)[:5]
    lines = [
        f"# Race run analysis: {args.run_id}", "",
        f"Bag: `{args.bag}`  ", f"Trajectory: `{args.trajectory}`  ", f"Track: `{args.track}`", "",
        "## Result", "",
        f"Scored laps: {len(scored_lap_values)}/10; best {summary['best_lap_s']!s} s; median {summary['median_lap_s']!s} s; mean {summary['mean_lap_s']!s} s; collision delta {collision_delta!s}.",
        f"Scored-lap truth samples: {len(scored_truth)}; speed p95 {summary['scored_lap_metrics']['truth_speed_mps']['p95']!s} m/s; |CTE| p95 {summary['scored_lap_metrics']['absolute_tracking_error_m']['p95']!s} m; map normal/tangential/yaw-error p95 {summary['scored_lap_metrics']['map_localization_normal_error_m']['p95']!s} m / {summary['scored_lap_metrics']['map_localization_tangential_error_m']['p95']!s} m / {summary['scored_lap_metrics']['map_localization_yaw_error_rad']['p95']!s} rad.",
        f"One-step MPC/truth matches: {len(model_rows)}; speed-error p95 {summary['one_step_model_error']['abs_speed_p95_mps']!s} m/s; yaw-rate-error p95 {summary['one_step_model_error']['abs_yaw_rate_p95_radps']!s} rad/s.",
        "", "## Lap times", "", "| Lap count | Role | Time (s) |", "|---:|---|---:|",
    ]
    for row in lap_rows:
        lines.append(f"| {row['lap_count']} | {row['lap_role']} | {row['lap_time_s'] if row['lap_time_s'] is not None else 'incomplete'} |")
    def list_section(title: str, rows: list[dict[str, Any]], fmt: Any) -> None:
        lines.extend(["", f"## {title}", ""])
        lines.extend([f"- {fmt(row)}" for row in rows] or ["- No supported rows."])
    list_section("TOP 5 LAP-TIME LOSSES", top_losses, lambda row: f"lap {row['lap_count']}, sector {row['sector']}: {float(row['time_loss_vs_best_s']):.3f} s slower than that sector's best observed lap")
    list_section("TOP 5 SAFETY BOTTLENECKS", top_safety, lambda row: f"lap {row['lap_count']}, sector {row['sector']}: minimum body clearance {row['minimum_wall_clearance_m']}")
    list_section("TOP 5 MODEL MISMATCHES", top_model, lambda row: f"at t={row['time_s']:.2f}s/s={row['progress_m']:.2f}m: |Δr|={abs(row['yaw_rate_error_radps']):.4f} rad/s, |Δu|={abs(row['speed_error_mps']):.4f} m/s, |Δey|={abs(row['lateral_error_m']):.4f} m")
    lines.extend(["", "## Plots", "", *[f"- [Plot {index:02d}: {path.stem}]({path.relative_to(outdir)})" for index, path in enumerate(sorted(plots.glob("*.svg")), 1)], "",
                  "Acceleration capability in plots 7–8 is observational only until the multi-run empirical envelope passes its support/validation gates.",
                  "Simulator truth appears only in this offline scorer and is never fed into a runtime node."])
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def load_reference_speed(path: Path) -> list[float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next((line.strip()[1:].strip() for line in lines if line.lstrip().startswith("#")), None)
    data = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not data:
        raise ValueError(f"trajectory contains no samples: {path}")
    if header:
        fields = [item.strip() for item in header.split(",")]
    else:
        fields = [item.strip() for item in data.pop(0).split(",")]
    reader = csv.DictReader(data, fieldnames=fields)
    rows = list(reader)
    speed_key = next((key for key in (
        "vx_mps", "velocity_mps", "speed_mps", "speed") if key in fields), None)
    if speed_key is None:
        return []
    return [float(row[speed_key]) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--track", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--steering-limit-rad", type=float, default=0.5236)
    args = parser.parse_args()
    if not args.bag.is_file() or not args.trajectory.is_file() or not args.track.is_file():
        parser.error("bag, trajectory, and track files must all exist")
    if not math.isfinite(args.steering_limit_rad) or args.steering_limit_rad <= 0:
        parser.error("--steering-limit-rad must be positive and finite")
    args.run_id = args.run_id or args.bag.parent.parent.name
    args.reference_speed = load_reference_speed(args.trajectory)
    try:
        summary = write_report(args)
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"run: {args.run_id}; scored={summary['scored_laps']}/10; mean={summary['mean_lap_s']}; CTE p95={summary['absolute_tracking_error_m']['p95']}; one-step model matches={summary['one_step_model_error']['sample_count']}")
    print(f"report: {args.output / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
