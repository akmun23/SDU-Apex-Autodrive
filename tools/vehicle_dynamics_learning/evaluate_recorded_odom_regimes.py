#!/usr/bin/env python3
"""Score a recorded sensor odometry stream against simulator truth by regime.

Development-only offline bag analysis. Ground-truth topics are read only from
closed bags; no runtime node subscribes to them. The default estimate topic is
the isolated Explore odometry sidecar, not the competition output topic.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TRUTH_ODOM = "/autodrive/roboracer_1/odom"
STEERING = "/autodrive/roboracer_1/steering"
COLLISIONS = "/autodrive/roboracer_1/collision_count"
TIMING_FAULT = "/autodrive/roboracer_1/bridge_timing_fault"
COM_X_M = 0.15532
MAX_SOURCE_OFFSET_NS = 20_000_000
MAX_RECEIPT_OFFSET_NS = 20_000_000
SPEED_EDGES_MPS = (0.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.01)
STEERING_EDGES_RAD = (0.0, 0.10, 0.20, 0.30, 0.40, 0.5241)


def _stamp_ns(stamp: Any) -> int | None:
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def _rmse(values: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _source_interval_summary(source_stamps_ns: list[int]) -> dict[str, Any]:
    stamps = np.asarray(source_stamps_ns, dtype=np.int64)
    intervals_ms = np.diff(stamps).astype(np.float64) / 1e6
    positive = intervals_ms[intervals_ms > 0.0]
    within_run = positive[positive <= 60.0]
    return {
        "within_run_intervals": int(len(within_run)),
        "long_intervals_gt_60ms": int(np.sum(positive > 60.0)),
        "duplicate_source_stamps": int(np.sum(intervals_ms == 0.0)),
        "backward_source_steps": int(np.sum(intervals_ms < 0.0)),
        "median_within_run_ms": (
            float(np.median(within_run)) if len(within_run) else None),
        "p95_within_run_ms": (
            float(np.percentile(within_run, 95)) if len(within_run) else None),
        "mean_within_run_rate_hz": (
            float(1000.0 / np.mean(within_run)) if len(within_run) else None),
    }


def _error_summary(truth: np.ndarray, estimate: np.ndarray) -> dict[str, Any]:
    error = estimate - truth
    return {
        "samples": int(len(error)),
        "u_rmse_mps": _rmse(error[:, 0].tolist()),
        "v_rmse_mps": _rmse(error[:, 1].tolist()),
        "yaw_rate_rmse_rps": _rmse(error[:, 2].tolist()),
        "u_bias_mps": float(np.mean(error[:, 0])) if len(error) else None,
        "v_bias_mps": float(np.mean(error[:, 1])) if len(error) else None,
        "yaw_rate_bias_rps": float(np.mean(error[:, 2])) if len(error) else None,
    }


def _score_regimes(
    truth: np.ndarray,
    estimate: np.ndarray,
    speed_mps: np.ndarray,
    steering_rad: np.ndarray,
) -> list[dict[str, Any]]:
    speed_bin = np.searchsorted(SPEED_EDGES_MPS, speed_mps, side="right") - 1
    steering_bin = np.searchsorted(
        STEERING_EDGES_RAD, np.abs(steering_rad), side="right") - 1
    rows = []
    for speed_index in range(len(SPEED_EDGES_MPS) - 1):
        for steer_index in range(len(STEERING_EDGES_RAD) - 1):
            mask = (speed_bin == speed_index) & (steering_bin == steer_index)
            if not np.any(mask):
                continue
            row = _error_summary(truth[mask], estimate[mask])
            row.update({
                "speed_bin_mps": [SPEED_EDGES_MPS[speed_index],
                                  SPEED_EDGES_MPS[speed_index + 1]],
                "abs_steering_bin_rad": [STEERING_EDGES_RAD[steer_index],
                                         STEERING_EDGES_RAD[steer_index + 1]],
            })
            rows.append(row)
    return rows


def _read_bag(path: Path, odom_topic: str) -> dict[str, Any]:
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as exc:  # pragma: no cover - depends on the ROS installation
        raise RuntimeError(f"ROS 2 Python modules are required: {exc}") from exc
    if not path.is_file():
        raise ValueError(f"bag does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = {
            name: (int(topic_id), msg_type)
            for topic_id, name, msg_type in connection.execute(
                "SELECT id,name,type FROM topics")
        }
        for name in (TRUTH_ODOM, odom_topic, STEERING):
            if name not in topics:
                raise ValueError(f"{path} is missing required topic {name}")

        def messages(topic: str) -> list[tuple[int, Any]]:
            topic_id, msg_type = topics[topic]
            msg_type_class = get_message(msg_type)
            return [
                (int(receipt), deserialize_message(bytes(payload), msg_type_class))
                for receipt, payload in connection.execute(
                    "SELECT timestamp,data FROM messages WHERE topic_id=? "
                    "ORDER BY timestamp,id", (topic_id,))
            ]

        truth_rows = messages(TRUTH_ODOM)
        estimate_rows = messages(odom_topic)
        steering_rows = messages(STEERING)
        truth_by_source = []
        for receipt, message in truth_rows:
            source = _stamp_ns(message.header.stamp)
            if source is None:
                continue
            twist = message.twist.twist
            truth_by_source.append((source, receipt, np.asarray((
                float(twist.linear.x),
                float(twist.linear.y) - COM_X_M * float(twist.angular.z),
                float(twist.angular.z),
            ), dtype=np.float64)))
        truth_by_source.sort(key=lambda row: row[0])
        truth_source_times = [row[0] for row in truth_by_source]

        steering_times = [receipt for receipt, _ in steering_rows]
        # `/steering` is already the measured physical angle in radians. Only
        # `/steering_command` is normalized; scaling this feedback again halves
        # the assigned regime and hides the high-angle cells.
        steering_values = [float(message.data) for _, message in steering_rows]
        joined_truth: list[np.ndarray] = []
        joined_estimate: list[np.ndarray] = []
        speeds: list[float] = []
        steering_values_joined: list[float] = []
        source_offsets_ms: list[float] = []
        estimate_source_stamps: list[int] = []
        unmatched_estimates = 0
        for receipt, message in estimate_rows:
            source = _stamp_ns(message.header.stamp)
            if source is None:
                unmatched_estimates += 1
                continue
            estimate_source_stamps.append(source)
            index = bisect.bisect_left(truth_source_times, source)
            candidates = [i for i in (index - 1, index)
                          if 0 <= i < len(truth_by_source)]
            if not candidates:
                unmatched_estimates += 1
                continue
            best = min(candidates, key=lambda i: abs(truth_source_times[i] - source))
            truth_source, truth_receipt, truth_state = truth_by_source[best]
            offset = truth_source - source
            if abs(offset) > MAX_SOURCE_OFFSET_NS:
                unmatched_estimates += 1
                continue
            steer_index = bisect.bisect_left(steering_times, truth_receipt)
            steer_candidates = [i for i in (steer_index - 1, steer_index)
                                if 0 <= i < len(steering_times)]
            if not steer_candidates:
                unmatched_estimates += 1
                continue
            steer_best = min(steer_candidates,
                             key=lambda i: abs(steering_times[i] - truth_receipt))
            if abs(steering_times[steer_best] - truth_receipt) > MAX_RECEIPT_OFFSET_NS:
                unmatched_estimates += 1
                continue
            twist = message.twist.twist
            estimate_state = np.asarray((
                float(twist.linear.x), float(twist.linear.y),
                float(twist.angular.z)), dtype=np.float64)
            if not np.isfinite(estimate_state).all():
                unmatched_estimates += 1
                continue
            joined_truth.append(truth_state)
            joined_estimate.append(estimate_state)
            speeds.append(abs(float(truth_state[0])))
            steering_values_joined.append(steering_values[steer_best])
            source_offsets_ms.append(offset / 1e6)

        truth_array = np.asarray(joined_truth, dtype=np.float64).reshape((-1, 3))
        estimate_array = np.asarray(joined_estimate, dtype=np.float64).reshape((-1, 3))
        speed_array = np.asarray(speeds, dtype=np.float64)
        steering_array = np.asarray(steering_values_joined, dtype=np.float64)
        collision_values = ([int(message.data) for _, message in messages(COLLISIONS)]
                            if COLLISIONS in topics else [])
        timing_values = ([bool(message.data) for _, message in messages(TIMING_FAULT)]
                         if TIMING_FAULT in topics else [])
        run_id = path.parents[1].name if path.parent.name == "run" else path.stem
        return {
            "run_id": run_id,
            "bag": str(path.resolve()),
            "estimate_topic": odom_topic,
            "truth_topic": TRUTH_ODOM,
            "steering_feedback_topic": STEERING,
            "matched_samples": len(joined_truth),
            "estimate_samples": len(estimate_rows),
            "unmatched_estimate_samples": unmatched_estimates,
            "estimate_source_stamp_intervals": _source_interval_summary(
                estimate_source_stamps),
            "source_stamp_offset_abs_p95_ms": (
                float(np.percentile(np.abs(source_offsets_ms), 95))
                if source_offsets_ms else None),
            "collision_count_start_end": (
                [collision_values[0], collision_values[-1]] if collision_values else None),
            "bridge_timing_fault_count": sum(timing_values),
            "overall": _error_summary(truth_array, estimate_array),
            "regimes": _score_regimes(
                truth_array, estimate_array, speed_array, steering_array),
        }
    finally:
        connection.close()


def analyze(bags: list[Path], odom_topic: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")
    runs = [_read_bag(path, odom_topic) for path in bags]
    by_cell: dict[tuple[float, float, float, float], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for run in runs:
        for cell in run["regimes"]:
            key = (*cell["speed_bin_mps"], *cell["abs_steering_bin_rad"])
            by_cell[key].append((run["run_id"], cell))
    cells = []
    for key, observations in sorted(by_cell.items()):
        rmse_names = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps")
        bias_names = ("u_bias_mps", "v_bias_mps", "yaw_rate_bias_rps")
        per_run = {run_id: {metric: cell[metric]
                            for metric in rmse_names + bias_names}
                   | {"samples": cell["samples"]}
                   for run_id, cell in observations}
        cells.append({
            "speed_bin_mps": list(key[:2]),
            "abs_steering_bin_rad": list(key[2:]),
            "independent_runs": len(observations),
            "mean_run_rmse": {
                metric: float(np.mean([cell[metric] for _, cell in observations]))
                for metric in rmse_names
            },
            "mean_run_bias": {
                metric: float(np.mean([cell[metric] for _, cell in observations]))
                for metric in bias_names
            },
            "per_run": per_run,
        })
    report = {
        "schema_version": 1,
        "analysis": "sensor odometry twist error against packet-stamped simulator truth, grouped by measured speed and steering",
        "timebase": "source message stamps are used for truth pairing and estimate cadence; cadence excludes long gaps at reset boundaries; bag receipt times only pair headerless steering feedback; receipt jitter is never integrated as dt",
        "lateral_velocity_convention": "both estimate and truth are rear-axle lateral velocity; simulator COM truth is converted by v_rear = v_com - 0.15532*r",
        "run_count": len(runs),
        "runs": runs,
        "run_mean_regime_summary": cells,
        "interpretation_limit": "Offline comparison only. Runtime estimator inputs must remain compliant; simulator truth is used only for bag scoring.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", type=Path, nargs="+",
                        help="closed rosbag2 sqlite files (run_0.db3)")
    parser.add_argument("--odom-topic", default="/explore_sensor_odom",
                        help="recorded sensor odometry estimate topic")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.bags, args.odom_topic, args.output)
    print(json.dumps({
        "runs": [{"run_id": run["run_id"], "matched": run["matched_samples"],
                  "overall": run["overall"]} for run in report["runs"]],
        "report": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
