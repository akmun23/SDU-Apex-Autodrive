#!/usr/bin/env python3
"""Compare ROS rear-encoder angle rates with same-run native WheelCollider RPM.

The internal RPM capture is from a source-built player and is not valid for
pinned-player tire-force identification. This paired-run test only evaluates
the encoder-angle-to-wheel-angular-rate measurement path.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sqlite3
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis


MIN_OVERLAP_S = 45.0
ALIGNMENT_MAX_RMSE_MPS = 0.10
ALIGNMENT_MIN_CORRELATION = 0.98
STEERING_LIMIT_RAD = 0.5236
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
STEERING_ALIGNMENT_MAX_RMSE_RAD = 0.03
STEERING_ALIGNMENT_MAX_P95_RAD = 0.05
THROTTLE_ALIGNMENT_MAX_RMSE = 0.03
THROTTLE_ALIGNMENT_MAX_P95 = 0.05
MIN_BAG_COVERAGE = 0.95
WINDOWS_S = (0.025, 0.050, 0.100, 0.200)
ACCEPT_WINDOW_S = 0.100
MAX_ACCEPT_BIAS_MPS = 0.05
MAX_ACCEPT_P95_MPS = 0.15
WHEEL_PAIRS = (("left", "rl"), ("right", "rr"))


def _load_capture(path: Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    metadata: dict[str, str] = {}
    columns: dict[str, list[float] | list[int]] = {
        "sim_time_s": [], "vel_body_z_mps": [],
        "capture_unix_time_ns": [],
        "steering_command": [], "throttle_command": [],
        "left_rpm": [], "right_rpm": [],
        "left_grounded": [], "right_grounded": [],
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line in stream:
            if line.startswith("# "):
                if "=" in line:
                    key, value = line[2:].strip().split("=", 1)
                    metadata[key] = value
                continue
            reader = csv.DictReader(itertools.chain((line,), stream))
            required = set(columns)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("wheel capture lacks shared-clock timestamps; "
                                 "recapture with the updated Unity recorder")
            for row in reader:
                columns["sim_time_s"].append(float(row["sim_time_s"]))
                columns["vel_body_z_mps"].append(float(row["vel_body_z_mps"]))
                columns["capture_unix_time_ns"].append(
                    int(row["capture_unix_time_ns"]))
                columns["steering_command"].append(float(row["steering_command"]))
                columns["throttle_command"].append(float(row["throttle_command"]))
                for wheel, prefix in WHEEL_PAIRS:
                    columns[f"{wheel}_rpm"].append(float(row[f"{prefix}_rpm"]))
                    columns[f"{wheel}_grounded"].append(
                        float(row[f"{prefix}_grounded"]))
            break
    arrays = {
        name: np.asarray(values, dtype=(np.int64 if name == "capture_unix_time_ns"
                                        else float))
        for name, values in columns.items()
    }
    if len(arrays["sim_time_s"]) < 1000:
        raise ValueError("internal wheel capture is too short")
    if not np.all(np.diff(arrays["sim_time_s"]) > 0.0):
        raise ValueError("internal simulation timestamps are not strictly increasing")
    if not np.all(np.diff(arrays["capture_unix_time_ns"]) > 0):
        raise ValueError("internal capture wall-clock timestamps are not strictly increasing")
    for name in ("vel_body_z_mps", "steering_command", "throttle_command",
                 "left_rpm", "right_rpm", "left_grounded", "right_grounded"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"internal capture contains non-finite values: {name}")
    if int(metadata.get("overwritten_samples", "-1")) != 0:
        raise ValueError("internal capture overwrote samples; full-run pairing is invalid")
    return metadata, arrays


def _load_ros(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, STEERING_COMMAND, analysis.THROTTLE_COMMAND,
                    analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError("bag missing required topics: " + ", ".join(missing))
        odom_t: list[int] = []
        odom_v: list[float] = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM):
            odom_t.append(receipt_ns)
            odom_v.append(float(message.twist.twist.linear.x))

        steering_t: list[int] = []
        steering_command: list[float] = []
        for receipt_ns, message in analysis._messages(
                connection, topics, STEERING_COMMAND):
            steering_t.append(receipt_ns)
            steering_command.append(float(message.data))
        if len(steering_t) < 10:
            raise ValueError("too few steering-command samples")

        throttle_t: list[int] = []
        throttle_command: list[float] = []
        for receipt_ns, message in analysis._messages(
                connection, topics, analysis.THROTTLE_COMMAND):
            throttle_t.append(receipt_ns)
            throttle_command.append(float(message.data))
        if len(throttle_t) < 10:
            raise ValueError("too few throttle-command samples")

        encoder_data: dict[str, np.ndarray] = {}
        encoder_names: list[str] = []
        for side, topic in (("left", analysis.LEFT_ENCODER),
                            ("right", analysis.RIGHT_ENCODER)):
            times: list[int] = []
            angles: list[float] = []
            for receipt_ns, message in analysis._messages(connection, topics, topic):
                if not message.position:
                    continue
                times.append(receipt_ns)
                angles.append(float(message.position[0]))
                if message.name:
                    encoder_names.append(f"{side}:{message.name[0]}")
            if len(times) < 10:
                raise ValueError(f"too few {side} encoder samples")
            encoder_data[f"{side}_unix_time_ns"] = np.asarray(times, dtype=np.int64)
            encoder_data[f"{side}_angle_rad"] = np.asarray(angles)
        encoder_data["odom_unix_time_ns"] = np.asarray(odom_t, dtype=np.int64)
        encoder_data["odom_vx_mps"] = np.asarray(odom_v)
        encoder_data["steering_command_unix_time_ns"] = np.asarray(
            steering_t, dtype=np.int64)
        encoder_data["steering_command_norm"] = np.asarray(steering_command)
        encoder_data["throttle_command_unix_time_ns"] = np.asarray(
            throttle_t, dtype=np.int64)
        encoder_data["throttle_command_norm"] = np.asarray(throttle_command)
        for time_key in ("odom_unix_time_ns", "steering_command_unix_time_ns",
                         "throttle_command_unix_time_ns",
                         "left_unix_time_ns", "right_unix_time_ns"):
            if not np.all(np.diff(encoder_data[time_key]) > 0):
                raise ValueError(f"ROS receipt timestamps are not strictly increasing: {time_key}")
        return encoder_data, sorted(set(encoder_names))
    finally:
        connection.close()


def _native_window_mean(times: np.ndarray, values: np.ndarray,
                        centers: np.ndarray, width_s: float) -> np.ndarray:
    integral = np.zeros_like(values)
    integral[1:] = np.cumsum(0.5 * (values[1:] + values[:-1])
                             * np.diff(times))
    starts = centers - width_s / 2.0
    ends = centers + width_s / 2.0
    return (np.interp(ends, times, integral) - np.interp(starts, times, integral)) / width_s


def _evaluate_side(side: str, capture_t: np.ndarray, capture: dict[str, np.ndarray],
                   ros: dict[str, np.ndarray],
                   radius_m: float) -> list[tuple[float, int, float, float, float, float]]:
    centers_ros = ros[f"{side}_time_s"]
    angles = ros[f"{side}_angle_rad"]
    encoder_times = centers_ros
    centers_source = centers_ros
    odom_v = np.interp(centers_ros, ros["odom_time_s"], ros["odom_vx_mps"])
    native_omega = (capture[f"{side}_rpm"] * (2.0 * math.pi / 60.0))
    grounded = capture[f"{side}_grounded"]
    outputs = []
    for width in WINDOWS_S:
        half = width / 2.0
        valid = ((centers_source - half >= capture_t[0])
                 & (centers_source + half <= capture_t[-1])
                 & (centers_ros - half >= encoder_times[0])
                 & (centers_ros + half <= encoder_times[-1])
                 & (centers_ros >= ros["odom_time_s"][0])
                 & (centers_ros <= ros["odom_time_s"][-1])
                 & (np.abs(odom_v) >= 0.8))
        sample_centers = centers_source[valid]
        encoder_rate = (
            np.interp(centers_ros[valid] + half, centers_ros, angles)
            - np.interp(centers_ros[valid] - half, centers_ros, angles)
        ) / width
        native_rate = _native_window_mean(capture_t, native_omega,
                                          sample_centers, width)
        grounded_fraction = _native_window_mean(capture_t, grounded,
                                                 sample_centers, width)
        retained = grounded_fraction >= 0.99
        errors_mps = radius_m * (encoder_rate[retained] - native_rate[retained])
        if len(errors_mps) < 100:
            raise ValueError(f"insufficient grounded samples for {side}, {width:.3f}s")
        outputs.append((width, len(errors_mps),
                        float(np.mean(errors_mps)),
                        float(np.sqrt(np.mean(errors_mps**2))),
                        float(np.percentile(np.abs(errors_mps), 95)),
                        float(np.max(np.abs(errors_mps)))) )
    return outputs


def evaluate(capture_path: Path, bag_path: Path) -> bool:
    metadata, capture = _load_capture(capture_path)
    ros, encoder_names = _load_ros(bag_path)
    radius = float(metadata["rear_left_radius_m"])
    if (radius <= 0.0
            or abs(radius - float(metadata["rear_right_radius_m"])) > 1e-6):
        raise ValueError("rear wheel radii are missing or asymmetric")
    epoch_ns = max(int(capture["capture_unix_time_ns"][0]),
                   int(ros["odom_unix_time_ns"][0]))
    capture_t = (capture["capture_unix_time_ns"] - epoch_ns) / 1e9
    ros_odom_t = (ros["odom_unix_time_ns"] - epoch_ns) / 1e9
    ros["odom_time_s"] = ros_odom_t
    ros_steering_t = (
        ros["steering_command_unix_time_ns"] - epoch_ns) / 1e9
    ros_throttle_t = (
        ros["throttle_command_unix_time_ns"] - epoch_ns) / 1e9
    for side in ("left", "right"):
        ros[f"{side}_time_s"] = (
            ros[f"{side}_unix_time_ns"] - epoch_ns) / 1e9
    overlap_start = max(float(capture_t[0]), float(ros_odom_t[0]))
    overlap_end = min(float(capture_t[-1]), float(ros_odom_t[-1]))
    overlap = overlap_end - overlap_start
    bag_span = float(ros_odom_t[-1] - ros_odom_t[0])
    coverage = overlap / bag_span if bag_span > 0.0 else 0.0
    valid_odom = ((ros_odom_t >= overlap_start) & (ros_odom_t <= overlap_end))
    if (overlap < MIN_OVERLAP_S or coverage < MIN_BAG_COVERAGE
            or np.count_nonzero(valid_odom) < 1000):
        print(f"shared-clock overlap={overlap:.2f} s, bag coverage={coverage:.1%} "
              f"(required >= {MIN_BAG_COVERAGE:.0%})")
        print("decision: REJECT pairing; captures do not cover the same interval")
        return False
    expected_v = np.interp(ros_odom_t[valid_odom], capture_t,
                           capture["vel_body_z_mps"])
    measured_v = ros["odom_vx_mps"][valid_odom]
    align_rmse = float(np.sqrt(np.mean((expected_v - measured_v) ** 2)))
    correlation = float(np.corrcoef(expected_v, measured_v)[0, 1])
    steering_overlap = ((ros_steering_t >= overlap_start)
                        & (ros_steering_t <= overlap_end))
    throttle_overlap = ((ros_throttle_t >= overlap_start)
                        & (ros_throttle_t <= overlap_end))
    if (np.count_nonzero(steering_overlap) < 1000
            or np.count_nonzero(throttle_overlap) < 1000):
        print("decision: REJECT pairing; fewer than 1000 shared-interval "
              "actuator-command samples")
        return False
    steering_times = ros_steering_t[steering_overlap]
    expected_steering = np.interp(
        steering_times, capture_t, capture["steering_command"])
    actual_steering = ros["steering_command_norm"][steering_overlap]
    steering_error_rad = (actual_steering - expected_steering) * STEERING_LIMIT_RAD
    steering_rmse = float(np.sqrt(np.mean(steering_error_rad ** 2)))
    steering_p95 = float(np.percentile(np.abs(steering_error_rad), 95))
    throttle_times = ros_throttle_t[throttle_overlap]
    expected_throttle = np.interp(
        throttle_times, capture_t, capture["throttle_command"])
    actual_throttle = ros["throttle_command_norm"][throttle_overlap]
    throttle_error = actual_throttle - expected_throttle
    throttle_rmse = float(np.sqrt(np.mean(throttle_error ** 2)))
    throttle_p95 = float(np.percentile(np.abs(throttle_error), 95))
    print(f"same-run capture: {capture_path}")
    print(f"bag: {bag_path}")
    print(f"native samples={len(capture['sim_time_s'])}, dt={metadata['fixed_delta_time_s']} s; "
          f"internal RPM is captured for all four wheels")
    print(f"bag encoder names: {', '.join(encoder_names) or '(not published)'}")
    print(f"shared-clock body-speed comparison: overlap={overlap:.2f} s, "
          f"bag coverage={coverage:.1%}, RMSE={align_rmse:.5f} m/s, "
          f"correlation={correlation:.6f}")
    print(f"shared-clock steering-command comparison: "
          f"RMSE={steering_rmse:.5f} rad, p95={steering_p95:.5f} rad")
    print(f"shared-clock throttle-command comparison: "
          f"RMSE={throttle_rmse:.5f}, p95={throttle_p95:.5f} normalized")
    if (overlap < MIN_OVERLAP_S or align_rmse > ALIGNMENT_MAX_RMSE_MPS
            or correlation < ALIGNMENT_MIN_CORRELATION
            or steering_rmse > STEERING_ALIGNMENT_MAX_RMSE_RAD
            or steering_p95 > STEERING_ALIGNMENT_MAX_P95_RAD
            or throttle_rmse > THROTTLE_ALIGNMENT_MAX_RMSE
            or throttle_p95 > THROTTLE_ALIGNMENT_MAX_P95):
        print("decision: REJECT pairing; shared-clock body-speed or actuator "
              "command alignment did not pass")
        return False

    results = {}
    for side in ("left", "right"):
        results[side] = _evaluate_side(
            side, capture_t, capture, ros, radius)
        print(f"{side} rear encoder versus WheelCollider.rpm, converted with r={radius:.4f} m:")
        print("  window  samples  bias[m/s]  RMSE[m/s]  p95|error|[m/s]  max|error|[m/s]")
        for width, count, bias, rmse, p95, maximum in results[side]:
            print(f"  {width:0.3f}   {count:6d}  {bias:+.5f}    {rmse:.5f}    "
                  f"{p95:.5f}           {maximum:.5f}")

    selected = [row for side in ("left", "right")
                for row in results[side] if abs(row[0] - ACCEPT_WINDOW_S) < 1e-9]
    passed = all(abs(row[2]) <= MAX_ACCEPT_BIAS_MPS
                 and row[4] <= MAX_ACCEPT_P95_MPS for row in selected)
    print("predeclared 100 ms acceptance: |bias|<=0.05 m/s and p95<=0.15 m/s "
          f"for both rear wheels: {'PASS' if passed else 'FAIL'}")
    print("scope: validates encoder-derived rear wheel angular rate in this paired "
          "source build only; it does not validate tire forces or pinned-player behavior")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_capture_csv", type=Path)
    parser.add_argument("rosbag_db3", type=Path)
    args = parser.parse_args()
    try:
        return 0 if evaluate(args.wheel_capture_csv, args.rosbag_db3) else 1
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"cannot evaluate paired wheel capture: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
