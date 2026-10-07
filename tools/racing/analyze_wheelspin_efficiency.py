#!/usr/bin/env python3
"""Measure wheel/body mismatch against simulator-truth acceleration offline."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - provided by project ROS image
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.racing.track_projection import TrackProjection, wrap_angle  # noqa: E402

TRUTH = "/autodrive/roboracer_1/odom"
LAP_COUNT = "/autodrive/roboracer_1/lap_count"
COLLISION = "/autodrive/roboracer_1/collision_count"
ODOM_DIAG = "/odom/diagnostics"
MAP_POSE = "/current_map_pose"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
THROTTLE_FEEDBACK = "/autodrive/roboracer_1/throttle"
STEERING = "/autodrive/roboracer_1/steering"
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
MAX_JOIN_AGE_NS = 60_000_000
PREDICTION_HORIZONS_S = (0.25, 0.50)
DEFAULT_STEERING_LIMIT_RAD = 0.5236
SPEED_BANDS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, math.inf))
STEERING_BANDS = ((0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, math.inf))
DIAG = {
    "wheel_mapped_mps": 4,
    "speed_pred_mps": 5,
    "wheel_update_used": 12,
    "timing_degraded": 15,
    "wheel_burst_rejected": 26,
    "wheel_packet_mps": 25,
}


def pose_of(message: Any) -> Any:
    pose = message.pose
    return pose.pose if hasattr(pose, "pose") else pose


def yaw_of(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def read_topic(connection: sqlite3.Connection, topics: dict[str, tuple[int, str]], name: str):
    if name not in topics:
        return []
    topic_id, type_name = topics[name]
    message_type = get_message(type_name)
    return [
        (int(receipt_ns), deserialize_message(bytes(raw), message_type))
        for receipt_ns, raw in connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id",
            (topic_id,),
        )
    ]


def nearest_prior(rows: list[tuple[int, Any]], stamps: list[int], target_ns: int):
    index = bisect.bisect_right(stamps, target_ns) - 1
    if index < 0 or target_ns - stamps[index] > MAX_JOIN_AGE_NS:
        return None
    return rows[index]


def nearest_at(rows: list[tuple[int, Any]], stamps: list[int], target_ns: int):
    index = bisect.bisect_left(stamps, target_ns)
    choices = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    if not choices:
        return None
    best = min(choices, key=lambda i: abs(stamps[i] - target_ns))
    if abs(stamps[best] - target_ns) > MAX_JOIN_AGE_NS:
        return None
    return rows[best]


def percentile(values: list[float], p: float) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    position = (len(values) - 1) * p
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def spearman_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        ranked = [0.0] * len(values)
        first = 0
        while first < len(order):
            after = first + 1
            while after < len(order) and values[order[after]] == values[order[first]]:
                after += 1
            average_rank = 0.5 * (first + after - 1) + 1.0
            for position in range(first, after):
                ranked[order[position]] = average_rank
            first = after
        return ranked

    x_rank, y_rank = ranks(x_values), ranks(y_values)
    x_mean, y_mean = statistics.fmean(x_rank), statistics.fmean(y_rank)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_rank, y_rank))
    x_energy = sum((x - x_mean) ** 2 for x in x_rank)
    y_energy = sum((y - y_mean) ** 2 for y in y_rank)
    if x_energy <= 0.0 or y_energy <= 0.0:
        return None
    return covariance / math.sqrt(x_energy * y_energy)


def summarize(rows: list[dict[str, Any]], *, acceleration_only: bool) -> dict[str, Any]:
    chosen = [row for row in rows if not acceleration_only or row["valid_for_acceleration"]]
    metrics = (
        "slip_excess_mps", "slip_ratio", "truth_accel_250ms_mps2",
        "truth_speed_accel_250ms_mps2", "wheel_minus_truth_u_mps",
        "wheel_minus_truth_speed_mps", "s1_proxy_error_vs_truth_u_mps",
        "speed_gain_250ms_mps", "speed_gain_500ms_mps",
        "localization_normal_growth_500ms_m",
        "localization_tangential_growth_500ms_m",
    )
    result: dict[str, Any] = {"sample_count": len(rows), "valid_acceleration_count": len(chosen)}
    for key in metrics:
        values = [float(row[key]) for row in chosen if row.get(key) is not None]
        result[key] = {
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p95_abs": percentile([abs(value) for value in values], 0.95),
        }
    result["wheel_burst_rate"] = (
        statistics.fmean(float(row["wheel_burst_rejected"]) for row in rows)
        if rows else None
    )
    result["wheel_update_rate"] = (
        statistics.fmean(float(row["wheel_update_used"]) for row in rows)
        if rows else None
    )
    return result


def speed_band(speed: float) -> str:
    for lower, upper in SPEED_BANDS:
        if lower <= speed < upper:
            return f"{lower:g}-{upper:g}" if math.isfinite(upper) else f">={lower:g}"
    return "out_of_range"


def steering_band(steering: float) -> str:
    absolute = abs(steering)
    for lower, upper in STEERING_BANDS:
        if lower <= absolute < upper:
            return f"{lower:g}-{upper:g}" if math.isfinite(upper) else f">={lower:g}"
    return "out_of_range"


def throttle_band(throttle: float) -> str:
    if throttle < 0.25:
        return "0-0.25"
    if throttle < 0.50:
        return "0.25-0.50"
    if throttle < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def throttle_feedback_band(throttle: float | None) -> str:
    if throttle is None:
        return "missing"
    if throttle < 0.10:
        return "0-0.10"
    if throttle < 0.20:
        return "0.10-0.20"
    if throttle < 0.30:
        return "0.20-0.30"
    return ">=0.30"


def run_samples(
    run_dir: Path,
    role: str,
    centerline: TrackProjection,
    steering_limit_rad: float = DEFAULT_STEERING_LIMIT_RAD,
    slip_edges: list[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bag = run_dir / "run" / "run_0.db3"
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = {
            name: (int(topic_id), message_type)
            for topic_id, name, message_type in connection.execute("SELECT id,name,type FROM topics")
        }
        required = (TRUTH, ODOM_DIAG, THROTTLE_COMMAND, STEERING, LAP_COUNT, COLLISION)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError(f"{run_dir.name}: missing bag topics {missing}")
        truth = read_topic(connection, topics, TRUTH)
        odom_diag = read_topic(connection, topics, ODOM_DIAG)
        map_pose = read_topic(connection, topics, MAP_POSE)
        throttle_command = read_topic(connection, topics, THROTTLE_COMMAND)
        throttle_feedback = read_topic(connection, topics, THROTTLE_FEEDBACK)
        steering = read_topic(connection, topics, STEERING)
        steering_command = read_topic(connection, topics, STEERING_COMMAND)
        lap_count = read_topic(connection, topics, LAP_COUNT)
        collisions = read_topic(connection, topics, COLLISION)
    finally:
        connection.close()

    initial_collision = int(collisions[0][1].data) if collisions else 0
    collision_event_ns = next(
        (stamp for stamp, message in collisions if int(message.data) > initial_collision), None
    )
    truth_stamps = [stamp for stamp, _ in truth]
    map_stamps = [stamp for stamp, _ in map_pose]
    throttle_stamps = [stamp for stamp, _ in throttle_command]
    feedback_stamps = [stamp for stamp, _ in throttle_feedback]
    steering_stamps = [stamp for stamp, _ in steering]
    steering_command_stamps = [stamp for stamp, _ in steering_command]
    lap_stamps = [stamp for stamp, _ in lap_count]

    truth_projection: list[tuple[float, float, float, int]] = []
    previous_segment: int | None = None
    for _, message in truth:
        pose = pose_of(message)
        projection = centerline.project(
            float(pose.position.x), float(pose.position.y), yaw_of(pose.orientation),
            previous_segment=previous_segment, local_search_radius=16,
        ) if previous_segment is not None else centerline.project(
            float(pose.position.x), float(pose.position.y), yaw_of(pose.orientation)
        )
        if projection.distance_m > 0.75:
            projection = centerline.project(
                float(pose.position.x), float(pose.position.y), yaw_of(pose.orientation)
            )
        previous_segment = projection.segment_index
        truth_projection.append((
            projection.s_m, projection.lateral_offset_m,
            yaw_of(pose.orientation), projection.segment_index,
        ))

    def localization_error(target_ns: int, truth_index: int) -> tuple[float | None, float | None]:
        map_row = nearest_prior(map_pose, map_stamps, target_ns)
        if map_row is None:
            return None, None
        estimate = pose_of(map_row[1])
        truth_s, truth_lateral, _, truth_segment = truth_projection[truth_index]
        projected = centerline.project(
            float(estimate.position.x), float(estimate.position.y), yaw_of(estimate.orientation),
            previous_segment=truth_segment, local_search_radius=16,
        )
        if projected.distance_m > 0.75:
            return None, None
        normal = projected.lateral_offset_m - truth_lateral
        tangent = (projected.s_m - truth_s + centerline.total_length / 2.0) % centerline.total_length - centerline.total_length / 2.0
        return normal, tangent

    samples: list[dict[str, Any]] = []
    unmatched_truth = 0
    for receipt_ns, message in odom_diag:
        if collision_event_ns is not None and receipt_ns >= collision_event_ns:
            break
        values = list(message.data)
        if len(values) <= max(DIAG.values()):
            continue
        command_row = nearest_prior(throttle_command, throttle_stamps, receipt_ns)
        if command_row is None:
            continue
        command = float(command_row[1].data)
        if command <= 0.0:
            continue
        lap_row = nearest_prior(lap_count, lap_stamps, receipt_ns)
        lap = int(lap_row[1].data) if lap_row else -1
        if lap < 2 or lap > 11:
            continue
        truth_row = nearest_at(truth, truth_stamps, receipt_ns)
        if truth_row is None:
            unmatched_truth += 1
            continue
        truth_index = bisect.bisect_left(truth_stamps, truth_row[0])
        truth_now = truth_row[1]
        u_now = float(truth_now.twist.twist.linear.x)
        v_now = float(truth_now.twist.twist.linear.y)
        speed_now = math.hypot(u_now, v_now)
        future: dict[float, tuple[Any, int]] = {}
        for horizon in PREDICTION_HORIZONS_S:
            row = nearest_at(truth, truth_stamps, receipt_ns + int(horizon * 1e9))
            if row is not None:
                index = bisect.bisect_left(truth_stamps, row[0])
                future[horizon] = (row[1], index)
        if any(horizon not in future for horizon in PREDICTION_HORIZONS_S):
            continue

        wheel_mapped = float(values[DIAG["wheel_mapped_mps"]])
        speed_pred = float(values[DIAG["speed_pred_mps"]])
        pre_wheel_valid = (
            len(values) > 41 and values[0] >= 7.0 and values[41] >= 0.5
        )
        pre_wheel_speed = float(values[40]) if pre_wheel_valid else None
        slip_excess = wheel_mapped - pre_wheel_speed if pre_wheel_valid else None
        slip_ratio = (
            slip_excess / max(pre_wheel_speed, 1.0)
            if slip_excess is not None and pre_wheel_speed is not None else None
        )
        wheel_used = int(values[DIAG["wheel_update_used"]] >= 0.5)
        timing_degraded = int(values[DIAG["timing_degraded"]] >= 0.5)
        burst_rejected = int(values[DIAG["wheel_burst_rejected"]] >= 0.5)
        steer_row = nearest_prior(steering, steering_stamps, receipt_ns)
        steer = float(steer_row[1].data) if steer_row else 0.0
        steer_command_row = nearest_prior(steering_command, steering_command_stamps, receipt_ns)
        steer_command = float(steer_command_row[1].data) if steer_command_row else None
        feedback_row = nearest_prior(throttle_feedback, feedback_stamps, receipt_ns)
        throttle_actual = float(feedback_row[1].data) if feedback_row else None

        u_250 = float(future[0.25][0].twist.twist.linear.x)
        u_500 = float(future[0.50][0].twist.twist.linear.x)
        speed_250 = math.hypot(u_250, float(future[0.25][0].twist.twist.linear.y))
        speed_500 = math.hypot(u_500, float(future[0.50][0].twist.twist.linear.y))
        dt_250 = (future[0.25][1] - truth_index) / 40.0
        dt_500 = (future[0.50][1] - truth_index) / 40.0
        if dt_250 <= 0.0 or dt_500 <= 0.0:
            continue
        normal_now, tangent_now = localization_error(receipt_ns, truth_index)
        normal_future, tangent_future = localization_error(
            receipt_ns + 500_000_000, future[0.50][1]
        )
        row = {
            "run_id": run_dir.name,
            "role": role,
            "lap_count": lap,
            "receipt_time_s": receipt_ns / 1e9,
            "track_s_m": truth_projection[truth_index][0],
            "truth_speed_mps": speed_now,
            "truth_u_mps": u_now,
            "steering_feedback_rad": steer,
            "steering_command_normalized": steer_command,
            "steering_command_rad": (
                None if steer_command is None
                else steer_command * steering_limit_rad
            ),
            "throttle_command": command,
            "throttle_feedback": throttle_actual,
            "wheel_mapped_mps": wheel_mapped,
            "legacy_speed_pred_diagnostic_mps": speed_pred,
            "pre_wheel_update_speed_mps": pre_wheel_speed,
            "pre_wheel_update_speed_valid": pre_wheel_valid,
            "slip_excess_mps": slip_excess,
            "slip_ratio": slip_ratio,
            "wheel_minus_truth_u_mps": wheel_mapped - u_now,
            "wheel_minus_truth_speed_mps": wheel_mapped - speed_now,
            "s1_proxy_error_vs_truth_u_mps": (
                None if slip_excess is None else slip_excess - (wheel_mapped - u_now)
            ),
            "wheel_update_used": wheel_used,
            "timing_degraded": timing_degraded,
            "wheel_burst_rejected": burst_rejected,
            "valid_for_acceleration": bool(wheel_used and not timing_degraded and not burst_rejected),
            "valid_for_s1_proxy": bool(
                wheel_used and not timing_degraded and not burst_rejected and pre_wheel_valid
            ),
            "truth_accel_250ms_mps2": (u_250 - u_now) / dt_250,
            "speed_gain_250ms_mps": speed_250 - speed_now,
            "speed_gain_500ms_mps": speed_500 - speed_now,
            "truth_speed_accel_250ms_mps2": (speed_250 - speed_now) / dt_250,
            "effective_horizon_250ms_s": dt_250,
            "effective_horizon_500ms_s": dt_500,
            "localization_normal_error_m": normal_now,
            "localization_tangential_error_m": tangent_now,
            "localization_normal_growth_500ms_m": (
                None if normal_now is None or normal_future is None
                else abs(normal_future) - abs(normal_now)
            ),
            "localization_tangential_growth_500ms_m": (
                None if tangent_now is None or tangent_future is None
                else abs(tangent_future) - abs(tangent_now)
            ),
        }
        samples.append(row)

    detail = {
        "run_id": run_dir.name,
        "role": role,
        "bag": str(bag.relative_to(ROOT) if bag.is_relative_to(ROOT) else bag),
        "collision_count_start": initial_collision,
        "first_collision_receipt_ns": collision_event_ns,
        "samples_positive_throttle_scored_laps": len(samples),
        "truth_alignment_misses": unmatched_truth,
        "valid_for_acceleration": sum(row["valid_for_acceleration"] for row in samples),
    }
    return samples, detail


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-run-dir", action="append", required=True, type=Path)
    parser.add_argument("--additional-run-dir", action="append", default=[], type=Path)
    parser.add_argument("--centerline", type=Path, default=ROOT / (
        "live_runs/raceline_candidates/practice_exact_84cap_clearance035_20261004/"
        "track_prep/SmoothCenterline.csv"))
    parser.add_argument(
        "--steering-limit-rad", type=float,
        default=DEFAULT_STEERING_LIMIT_RAD,
        help="physical steering angle at normalized command magnitude 1")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.steering_limit_rad) or args.steering_limit_rad <= 0.0:
        parser.error("--steering-limit-rad must be finite and positive")
    centerline_path = args.centerline if args.centerline.is_absolute() else ROOT / args.centerline
    if not centerline_path.is_file():
        parser.error(f"centerline not found: {centerline_path}")
    centerline = TrackProjection.from_csv(centerline_path, closed=True)

    all_samples: list[dict[str, Any]] = []
    run_details: list[dict[str, Any]] = []
    p0_samples: list[dict[str, Any]] = []
    for role, run_dirs in (("P0", args.p0_run_dir), ("additional", args.additional_run_dir)):
        for run_dir in run_dirs:
            run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
            samples, detail = run_samples(
                run_dir, role, centerline, args.steering_limit_rad)
            all_samples.extend(samples)
            run_details.append(detail)
            if role == "P0":
                p0_samples.extend(row for row in samples if row["valid_for_acceleration"])

    if not p0_samples:
        parser.error("P0 bags contain no valid positive-throttle samples on scored laps")
    p0_truth_slip = sorted(float(row["wheel_minus_truth_u_mps"]) for row in p0_samples)
    slip_edges = [percentile(p0_truth_slip, q) for q in (0.20, 0.40, 0.60, 0.80)]
    edges = [float(value) for value in slip_edges if value is not None]
    for row in all_samples:
        row["p0_truth_slip_quantile_bin"] = (
            f"Q{bisect.bisect_right(edges, float(row['wheel_minus_truth_u_mps'])) + 1}"
        )
        row["speed_band_mps"] = speed_band(float(row["truth_speed_mps"]))
        row["steering_band_abs_rad"] = steering_band(float(row["steering_feedback_rad"]))
        row["throttle_command_band"] = throttle_band(float(row["throttle_command"]))
        row["throttle_feedback_band"] = throttle_feedback_band(
            None if row["throttle_feedback"] is None else float(row["throttle_feedback"])
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_samples:
        grouped[(row["run_id"], "all", "all", row["p0_truth_slip_quantile_bin"])].append(row)
        grouped[(row["run_id"], row["speed_band_mps"], row["steering_band_abs_rad"], row["p0_truth_slip_quantile_bin"])].append(row)
    summary_rows = []
    for key, rows in sorted(grouped.items()):
        run_id, speed, steering, slip_bin = key
        metrics = summarize(rows, acceleration_only=True)
        summary = {
            "run_id": run_id,
            "speed_band_mps": speed,
            "steering_band_abs_rad": steering,
            "p0_slip_quantile_bin": slip_bin,
        }
        for metric_name, values in metrics.items():
            if isinstance(values, dict):
                for statistic, value in values.items():
                    summary[f"{metric_name}_{statistic}"] = value
            else:
                summary[metric_name] = values
        summary_rows.append(summary)

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sample_features.csv", all_samples)
    write_csv(output_dir / "slip_efficiency_bins.csv", summary_rows)

    matched_groups: dict[tuple[str, str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"low": [], "high": []}
    )
    for row in all_samples:
        if not row["valid_for_acceleration"]:
            continue
        # Compare simulator-truth wheel/body mismatch within the same coarse
        # operating regime. S1 is tested as a candidate observer signal below.
        actual_slip = float(row["wheel_minus_truth_u_mps"])
        if actual_slip < 0.0:
            continue
        key = (
            row["run_id"], row["speed_band_mps"],
            row["steering_band_abs_rad"], row["throttle_command_band"],
            row["throttle_feedback_band"],
        )
        if actual_slip <= 0.10:
            matched_groups[key]["low"].append(row)
        elif actual_slip >= 0.25:
            matched_groups[key]["high"].append(row)
    matched_rows: list[dict[str, Any]] = []
    for (run_id, speed, steering_name, command_band, feedback_band), pair in sorted(matched_groups.items()):
        low, high = pair["low"], pair["high"]
        if len(low) < 5 or len(high) < 5:
            continue
        low_accel = [float(row["truth_accel_250ms_mps2"]) for row in low]
        high_accel = [float(row["truth_accel_250ms_mps2"]) for row in high]
        low_gain = [float(row["speed_gain_500ms_mps"]) for row in low]
        high_gain = [float(row["speed_gain_500ms_mps"]) for row in high]
        low_proxy = [float(row["slip_excess_mps"]) for row in low if row["slip_excess_mps"] is not None]
        high_proxy = [float(row["slip_excess_mps"]) for row in high if row["slip_excess_mps"] is not None]
        matched_rows.append({
            "run_id": run_id,
            "speed_band_mps": speed,
            "steering_band_abs_rad": steering_name,
            "throttle_command_band": command_band,
            "throttle_feedback_band": feedback_band,
            "low_slip_n": len(low),
            "high_slip_n": len(high),
            "low_truth_wheel_body_slip_mps_p50": percentile(
                [float(row["wheel_minus_truth_u_mps"]) for row in low], 0.5
            ),
            "high_truth_wheel_body_slip_mps_p50": percentile(
                [float(row["wheel_minus_truth_u_mps"]) for row in high], 0.5
            ),
            "low_s1_proxy_mps_p50": percentile(
                low_proxy, 0.5
            ),
            "high_s1_proxy_mps_p50": percentile(
                high_proxy, 0.5
            ),
            "low_slip_longitudinal_accel_p50_mps2": percentile(low_accel, 0.5),
            "high_slip_longitudinal_accel_p50_mps2": percentile(high_accel, 0.5),
            "high_minus_low_longitudinal_accel_p50_mps2": (
                percentile(high_accel, 0.5) - percentile(low_accel, 0.5)
            ),
            "low_slip_speed_gain_500ms_p50_mps": percentile(low_gain, 0.5),
            "high_slip_speed_gain_500ms_p50_mps": percentile(high_gain, 0.5),
            "high_minus_low_speed_gain_500ms_p50_mps": (
                percentile(high_gain, 0.5) - percentile(low_gain, 0.5)
            ),
        })
    write_csv(output_dir / "matched_slip_contrasts.csv", matched_rows)

    actual_slip_bins = (
        ("underread", -math.inf, 0.0), ("0-0.10", 0.0, 0.10),
        ("0.10-0.25", 0.10, 0.25), ("0.25-0.50", 0.25, 0.50),
        (">=0.50", 0.50, math.inf),
    )
    actual_slip_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_samples:
        mismatch = float(row["wheel_minus_truth_u_mps"])
        for label, lower, upper in actual_slip_bins:
            if lower <= mismatch < upper:
                actual_slip_groups[(row["run_id"], label)].append(row)
                break
    actual_slip_rows: list[dict[str, Any]] = []
    for (run_id, label), rows in sorted(actual_slip_groups.items()):
        metrics = summarize(rows, acceleration_only=True)
        actual_slip_rows.append({
            "run_id": run_id,
            "truth_wheel_body_slip_bin_mps": label,
            "sample_count": metrics["sample_count"],
            "valid_acceleration_count": metrics["valid_acceleration_count"],
            "truth_longitudinal_accel_250ms_p50_mps2": metrics["truth_accel_250ms_mps2"]["p50"],
            "truth_speed_accel_250ms_p50_mps2": metrics["truth_speed_accel_250ms_mps2"]["p50"],
            "speed_gain_500ms_p50_mps": metrics["speed_gain_500ms_mps"]["p50"],
            "speed_gain_500ms_p95_abs_mps": metrics["speed_gain_500ms_mps"]["p95_abs"],
            "localization_normal_growth_500ms_p50_m": metrics["localization_normal_growth_500ms_m"]["p50"],
        })
    write_csv(output_dir / "truth_slip_efficiency_bins.csv", actual_slip_rows)

    proxy_agreement: dict[str, Any] = {}
    for run_id in sorted({row["run_id"] for row in all_samples}):
        rows = [
            row for row in all_samples
            if row["run_id"] == run_id and row["valid_for_s1_proxy"]
        ]
        proxy_agreement[run_id] = {
            "sample_count": len(rows),
            "s1_vs_truth_wheel_body_slip_spearman": spearman_correlation(
                [float(row["slip_excess_mps"]) for row in rows],
                [float(row["wheel_minus_truth_u_mps"]) for row in rows],
            ),
            "s1_proxy_abs_error_vs_truth_u_mps_p50": percentile(
                [abs(float(row["s1_proxy_error_vs_truth_u_mps"])) for row in rows], 0.5
            ),
            "s1_proxy_abs_error_vs_truth_u_mps_p95": percentile(
                [abs(float(row["s1_proxy_error_vs_truth_u_mps"])) for row in rows], 0.95
            ),
        }
    report = {
        "schema_version": 4,
        "label_source": "simulator truth, offline only",
        "truth_slip_label_definition": "wheel_mapped_mps - simulator-truth body-forward speed; offline label only",
        "steering_command_conversion": (
            "normalized /autodrive/.../steering_command multiplied by "
            f"{args.steering_limit_rad:.9g} rad; /steering feedback is already radians"
        ),
        "s1_slip_excess_definition": "wheel_mapped_mps - pre_wheel_update_speed_mps from odometry diagnostic v7; unavailable on earlier diagnostic schemas",
        "s1_slip_ratio_definition": "s1_slip_excess_mps / max(pre_wheel_update_speed_mps, 1.0)",
        "legacy_speed_pred_note": "diagnostic index 5 is retained for comparison only; observer code overwrites it after wheel fusion in turn mode, so it is not a consistent S1 reference",
        "acceleration_sample_gate": "positive throttle command, wheel update used, no timing degradation, no burst rejection",
        "truth_acceleration_definition": "change in simulator truth body-forward linear velocity divided by actual aligned sample interval; separate from speed-magnitude gain",
        "matched_contrast_definition": "within run, coarse speed, absolute steering, throttle-command, and throttle-feedback bins; compare truth wheel-minus-body mismatch <=0.10 m/s against >=0.25 m/s; bins with fewer than five samples per side omitted",
        "lap_gate": "lap_count 2 through 11 (scored laps only)",
        "p0_truth_wheel_body_slip_quantile_edges_mps": edges,
        "s1_candidate_proxy_agreement_by_run": proxy_agreement,
        "runs": run_details,
        "p0_valid_sample_count": len(p0_samples),
        "p0_valid_s1_sample_count": sum(row["valid_for_s1_proxy"] for row in p0_samples),
        "additional_valid_sample_count": sum(
            detail["valid_for_acceleration"] for detail in run_details if detail["role"] == "additional"
        ),
        "additional_valid_s1_sample_count": sum(
            row["valid_for_s1_proxy"] for row in all_samples if row["role"] == "additional"
        ),
        "overall_p0": summarize(p0_samples, acceleration_only=False),
        "outputs": [
            "sample_features.csv", "slip_efficiency_bins.csv", "matched_slip_contrasts.csv",
            "truth_slip_efficiency_bins.csv",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"P0 valid samples: {len(p0_samples)}")
    print(f"P0 truth wheel/body slip quantile edges (m/s): {edges}")
    print(f"Wrote traction analysis: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
