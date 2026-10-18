#!/usr/bin/env python3
"""Measure production odometry and MPC rollout errors against Unity bag truth.

Run in the ROS 2 environment that can deserialize the bag's message types:
  python3 tools/evaluate_mpc_bag.py live_runs/<run>/run [--trajectory path.csv]

The trajectory is optional; when supplied, diagnostic path progress and
cross-track error are converted back to a local XY/yaw rollout for comparison.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


GT = "/autodrive/roboracer_1/odom"
ODOM = "/odom"
EKF = "/ekf_odom"
PLANT = "/odom/unity_plant_state"
DIAGNOSTICS = "/mpc/diagnostics"
IMU = "/autodrive/roboracer_1/imu"
LAP = "/autodrive/roboracer_1/lap_count"
COLLISION = "/autodrive/roboracer_1/collision_count"
LAST_LAP = "/autodrive/roboracer_1/last_lap_time"
BEST_LAP = "/autodrive/roboracer_1/best_lap_time"
THROTTLE = "/autodrive/roboracer_1/throttle"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
STEERING = "/autodrive/roboracer_1/steering"
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
BRIDGE_FAULT = "/autodrive/roboracer_1/bridge_timing_fault_detail"
DRIVE_COMMAND = "/mpc/drive_command"
LEFT_ENCODER = "/autodrive/roboracer_1/left_encoder"
RIGHT_ENCODER = "/autodrive/roboracer_1/right_encoder"
ODOM_DIAGNOSTICS = "/odom/diagnostics"
COM_TO_REAR_AXLE_M = 0.15532
WHEEL_RADIUS_M = 0.059
REAR_HALF_TRACK_M = 0.118
ENCODER_SAMPLE_PERIOD_S = 0.025

TOPICS = {
    GT, ODOM, EKF, PLANT, DIAGNOSTICS, IMU, LAP, COLLISION, LAST_LAP,
    BEST_LAP, THROTTLE, THROTTLE_COMMAND, STEERING, STEERING_COMMAND,
    BRIDGE_FAULT, DRIVE_COMMAND, LEFT_ENCODER, RIGHT_ENCODER, ODOM_DIAGNOSTICS,
}


def stamp(message: Any) -> float:
    return message.header.stamp.sec + message.header.stamp.nanosec * 1e-9


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def normalized_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def read_bag(uri: Path) -> dict[str, list[dict[str, Any]]]:
    storage_id = "mcap" if any(uri.glob("*.mcap")) else "sqlite3"
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(uri), storage_id=storage_id),
        ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    type_names = {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
        if topic.name in TOPICS
    }
    message_types = {}
    for name, kind in type_names.items():
        try:
            message_types[name] = get_message(kind)
        except (AttributeError, ModuleNotFoundError):
            # The plant-state message is diagnostic-only here; the 35-value
            # odom diagnostic topic already contains its raw pre-correction
            # plant state. This lets the evaluator run on a stock ROS install.
            if name not in (PLANT, DRIVE_COMMAND):
                raise
            print(f"note: skipping {name} (custom message package not sourced)")
    data = {topic: [] for topic in type_names}
    skipped_plant_schema = False

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        if topic not in message_types:
            continue
        try:
            message = deserialize_message(serialized, message_types[topic])
        except Exception as error:
            # Older runtime captures used earlier UnityPlantState layouts.
            # That diagnostic topic is not needed for ground-truth odom or
            # MPC forecast scoring; keep the rest of the bag usable while
            # making the schema mismatch explicit.
            if topic != PLANT:
                raise
            if not skipped_plant_schema:
                print(f"note: skipping incompatible {PLANT} payloads: {error}")
                skipped_plant_schema = True
            continue
        row: dict[str, Any] = {"bag_time": bag_time_ns * 1e-9}
        if topic in (GT, ODOM, EKF):
            row.update(
                stamp=stamp(message),
                frame=message.header.frame_id,
                child=message.child_frame_id,
                x=message.pose.pose.position.x,
                y=message.pose.pose.position.y,
                yaw=yaw_from_quaternion(message.pose.pose.orientation),
                u=message.twist.twist.linear.x,
                v=message.twist.twist.linear.y,
                r=message.twist.twist.angular.z,
            )
        elif topic == PLANT:
            row.update(
                stamp=stamp(message),
                schema=message.state_schema_version,
                u=message.longitudinal_velocity_mps,
                v_com=message.lateral_velocity_com_mps,
                v_rear=message.lateral_velocity_rear_axle_mps,
                r=message.yaw_rate_radps,
                wheel_omega=list(message.wheel_omega_radps),
                wheel_angle=list(message.wheel_rotation_angle_rad),
                long_slip=list(message.longitudinal_slip),
                lat_slip=list(message.lateral_slip),
                normal_load=list(message.normal_load_n),
                tire_fx=list(message.tire_longitudinal_force_n),
                tire_fy=list(message.tire_lateral_force_n),
                ax=message.longitudinal_acceleration_mps2,
                ay=message.lateral_acceleration_mps2,
                yaw_accel=message.yaw_acceleration_radps2,
            )
        elif topic == DIAGNOSTICS:
            try:
                row["json"] = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                row["json"] = {}
        elif topic == IMU:
            row.update(
                stamp=stamp(message),
                ax=message.linear_acceleration.x,
                ay=message.linear_acceleration.y,
                yaw_rate=message.angular_velocity.z,
            )
        elif topic == DRIVE_COMMAND:
            row.update(
                stamp=stamp(message),
                steering=message.steering_angle_rad,
                throttle=message.throttle_normalized,
            )
        elif topic in (LEFT_ENCODER, RIGHT_ENCODER):
            row.update(
                stamp=stamp(message),
                angle=message.position[0] if message.position else math.nan,
                velocity=message.velocity[0] if message.velocity else math.nan,
            )
        elif topic == ODOM_DIAGNOSTICS:
            row.update(label=message.layout.dim[0].label if message.layout.dim else "",
                       values=list(message.data))
        elif topic == BRIDGE_FAULT:
            row["value"] = message.data
        else:
            row["value"] = message.data
        data[topic].append(row)
    return data


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * (len(ordered) - 1)))]


def print_interval(topic: str, rows: list[dict[str, Any]]) -> None:
    times = [row.get("stamp", row["bag_time"]) for row in rows]
    intervals = [b - a for a, b in zip(times, times[1:]) if 0.0 < b - a < 0.2]
    if intervals:
        print(
            f"rate {topic}: n={len(rows)} mean={statistics.mean(intervals)*1000:.2f}ms "
            f"p95={percentile(intervals, 0.95)*1000:.2f}ms "
            f"span={times[-1]-times[0]:.2f}s"
        )


def matched_pairs(
    truth: list[dict[str, Any]], estimate: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    lookup = {round(row["stamp"], 6): row for row in estimate}
    return [
        (row, lookup[round(row["stamp"], 6)])
        for row in truth
        if round(row["stamp"], 6) in lookup
    ]


def report_state_errors(
    truth: list[dict[str, Any]], estimate: list[dict[str, Any]], label: str,
) -> None:
    pairs = matched_pairs(truth, estimate)
    print(f"{label}: matched={len(pairs)}")
    if not pairs:
        return
    for field in ("u", "v", "r"):
        # Simulator /odom twist is at the rigid-body COM; production odom is
        # at base_link (rear axle). Rotate the COM velocity to the rear axle
        # before scoring its lateral component.
        if field == "v" and label in ("production odom", "EKF odom"):
            errors = [estimated[field] - (actual["v"] - actual["r"] * COM_TO_REAR_AXLE_M)
                      for actual, estimated in pairs]
        else:
            errors = [estimated[field] - actual[field] for actual, estimated in pairs]
        absolute = [abs(error) for error in errors]
        print(
            f"  {field}: bias={statistics.mean(errors):.4f} "
            f"RMSE={math.sqrt(statistics.mean(e*e for e in errors)):.4f} "
            f"p95_abs={percentile(absolute, 0.95):.4f} "
            f"max_abs={max(absolute):.4f}"
        )

    actual0, estimated0 = pairs[0]
    yaw_alignment = estimated0["yaw"] - actual0["yaw"]
    position_errors = []
    yaw_errors = []
    c, s = math.cos(yaw_alignment), math.sin(yaw_alignment)
    for actual, estimated in pairs:
        dx, dy = actual["x"] - actual0["x"], actual["y"] - actual0["y"]
        actual_dx, actual_dy = c * dx - s * dy, s * dx + c * dy
        estimated_dx = estimated["x"] - estimated0["x"]
        estimated_dy = estimated["y"] - estimated0["y"]
        position_errors.append(math.hypot(
            estimated_dx - actual_dx, estimated_dy - actual_dy,
        ))
        yaw_errors.append(normalized_angle(
            (estimated["yaw"] - estimated0["yaw"])
            - (actual["yaw"] - actual0["yaw"])
        ))
    print(
        f"  relative_pose: xy_end={position_errors[-1]:.3f}m "
        f"xy_p95={percentile(position_errors, 0.95):.3f}m "
        f"xy_max={max(position_errors):.3f}m "
        f"yaw_end={yaw_errors[-1]:.3f}rad "
        f"yaw_p95_abs={percentile([abs(x) for x in yaw_errors], 0.95):.3f}rad"
    )


def nearest_by_stamp(rows: list[dict[str, Any]], target: float) -> dict[str, Any] | None:
    if not rows:
        return None
    times = [row.get("stamp", row["bag_time"]) for row in rows]
    index = bisect.bisect_left(times, target)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    return min((rows[i] for i in candidates), key=lambda row: abs(
        row.get("stamp", row["bag_time"]) - target))


def report_odom_diagnostics(
    truth: list[dict[str, Any]], odom: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    steering_rows: list[dict[str, Any]], throttle_rows: list[dict[str, Any]],
) -> None:
    """Compare observer output and its uncorrected plant at the same epochs."""
    if not diagnostics:
        return
    print("\nOdometry source breakdown (Unity truth is evaluation-only):")
    print(f"  diagnostic label: {diagnostics[0].get('label', '<missing>')}")
    print("  fields: raw source-plant u/v/r, gyro-corrected yaw forecast when available, encoder speed")
    # In this node diagnostics and odom are emitted once per synchronized
    # sensor packet. Use sequence pairing when counts agree; the diagnostics
    # array's internal stamp is a relative 25 ms clock, not the bag ROS stamp.
    pair_by_index = len(truth) == len(diagnostics) == len(odom)
    samples: list[dict[str, float]] = []
    if pair_by_index:
        rows = zip(truth, odom, diagnostics)
    else:
        # Fallback to recorder time for bags that contain extra startup rows.
        aligned = []
        for diag in diagnostics:
            t = diag["bag_time"]
            gt = nearest_by_stamp(truth, t)
            est = nearest_by_stamp(odom, t)
            if gt is not None and est is not None:
                aligned.append((gt, est, diag))
        rows = aligned
    for sample_index, (gt, est, diag) in enumerate(rows):
        values = diag.get("values", [])
        if len(values) < 35:
            continue
        gt_rear_v = gt["v"] - gt["r"] * COM_TO_REAR_AXLE_M
        speed = math.hypot(gt["u"], gt_rear_v)
        # Bridge actuator feedback are Float32 and headerless, so nearest bag
        # time is the correct temporal key for those streams.
        feedback_steer = nearest_by_stamp(steering_rows, gt["bag_time"])
        feedback_throttle = nearest_by_stamp(throttle_rows, gt["bag_time"])
        samples.append({
            "sample_index": float(sample_index),
            "speed": speed,
            "steering": abs(float(feedback_steer["value"])) if feedback_steer else 0.0,
            "throttle": float(feedback_throttle["value"]) if feedback_throttle else 0.0,
            "u_error": est["u"] - gt["u"],
            "raw_u_error": values[10] - gt["u"],
            "v_error": est["v"] - gt_rear_v,
            "raw_v_error": values[11] - (gt["v"]),
            "yaw_error": values[12] - gt["r"],
            "corrected_yaw_error": (
                values[39] - gt["r"] if len(values) >= 41 and
                any(version in diag.get("label", "") for version in (
                    "source_derived_odom_v2", "source_derived_odom_v3"))
                else math.nan
            ),
            "gt_yaw_rate": gt["r"],
            "raw_yaw_prediction": values[12],
            "corrected_yaw_prediction": (
                values[39] if len(values) >= 41 and
                any(version in diag.get("label", "") for version in (
                    "source_derived_odom_v2", "source_derived_odom_v3"))
                else math.nan
            ),
            "yaw_prediction_mode": values[41] if len(values) >= 42 else -1.0,
            "encoder_speed": values[7],
            "encoder_u_error": values[7] - gt["u"],
            "wheel_body_mismatch": 0.059 * 0.5 * (values[22] + values[23]) - values[10],
            "model_ax_error": values[32] - values[8],
            "model_ay_error": values[33] - values[9],
            "process_us": values[21],
        })
    if not samples:
        print("  no matched samples")
        return
    def show(label: str, selected: list[dict[str, float]]) -> None:
        if not selected:
            return
        print(f"  {label}: n={len(selected)}")
        for key, unit in (("u_error", "m/s"), ("raw_u_error", "m/s"),
                          ("encoder_u_error", "m/s"), ("v_error", "m/s"),
                          ("wheel_body_mismatch", "m/s")):
            errors = [sample[key] for sample in selected]
            print(f"    {key}: bias={statistics.mean(errors):+.4f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in errors)):.4f} {unit}")
        for label, key, prediction_key in (
            ("raw_yaw_forecast", "yaw_error", "raw_yaw_prediction"),
            ("corrected_yaw_forecast", "corrected_yaw_error",
             "corrected_yaw_prediction"),
        ):
            valid = [s for s in selected if math.isfinite(s[key])]
            errors = [s[key] for s in valid]
            if not errors:
                continue
            worst = max(valid, key=lambda s: abs(s[key]))
            absolute = [abs(error) for error in errors]
            print(f"    {label}: bias={statistics.mean(errors):+.4f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
                  f"p95={percentile(absolute, 0.95):.4f} "
                  f"p99={percentile(absolute, 0.99):.4f} "
                  f"max={abs(worst[key]):.4f} rad/s at sample "
                  f"{int(worst['sample_index'])} "
                  f"(GT={worst['gt_yaw_rate']:+.3f}, "
                  f"prediction={worst[prediction_key]:+.3f}, "
                  f"speed={worst['speed']:.2f}m/s, "
                  f"|steer|={worst['steering']:.3f}rad, "
                  f"throttle={worst['throttle']:.3f}, "
                  f"mode={int(worst['yaw_prediction_mode'])})")
        print(f"    raw_model_vs_IMU accel RMSE: ax={math.sqrt(statistics.mean(x['model_ax_error']**2 for x in selected)):.3f}, "
              f"ay={math.sqrt(statistics.mean(x['model_ay_error']**2 for x in selected)):.3f} m/s2")
    show("all", samples)
    for low, high in ((0, 2), (2, 4), (4, 6), (6, 8), (8, math.inf)):
        show(f"speed [{low},{high:g}) m/s", [s for s in samples if low <= s["speed"] < high])
    for low, high in ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, math.inf)):
        show(f"abs steering [{low:.2f},{high:g}) rad", [s for s in samples if low <= s["steering"] < high])
    for low, high in ((-math.inf, 0.01), (0.01, 0.25), (0.25, 0.75), (0.75, math.inf)):
        show(f"throttle [{low:g},{high:g})", [s for s in samples if low <= s["throttle"] < high])
    mismatch = [s for s in samples if abs(s["wheel_body_mismatch"]) > 0.5]
    show("wheel peripheral / raw-body mismatch >0.5 m/s", mismatch)


def report_encoder_phase(
    left_encoder: list[dict[str, Any]], right_encoder: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]], throttle_rows: list[dict[str, Any]],
) -> None:
    """Compare the 1 kHz pre-solve RPM integral with Unity encoder angles."""
    if not diagnostics or len(left_encoder) != len(right_encoder) or len(left_encoder) != len(diagnostics):
        return
    predicted = [row.get("values", [])[35:39] for row in diagnostics]
    if any(len(row) != 4 for row in predicted):
        return
    errors = [[], []]
    brake_transition_errors = [[], []]
    for index in range(1, len(diagnostics)):
        measured = (
            left_encoder[index]["angle"] - left_encoder[index - 1]["angle"],
            right_encoder[index]["angle"] - right_encoder[index - 1]["angle"],
        )
        for side in range(2):
            residual = measured[side] - (
                predicted[index][side + 2] - predicted[index - 1][side + 2]
            )
            errors[side].append(residual)
            q_now = nearest_by_stamp(throttle_rows, diagnostics[index]["bag_time"])
            q_prev = nearest_by_stamp(throttle_rows, diagnostics[index - 1]["bag_time"])
            if q_now and q_prev and float(q_now["value"]) <= 0.005 and float(q_prev["value"]) > 0.01:
                brake_transition_errors[side].append(residual)
    print("\nUnity encoder sample-phase check (observed encoder angle minus pre-solve RPM prediction):")
    for side, label in enumerate(("left rear", "right rear")):
        values = errors[side]
        print(f"  {label}: n={len(values)} RMSE={math.sqrt(statistics.mean(x*x for x in values)):.5f} rad")
        transition = brake_transition_errors[side]
        if transition:
            print(f"    drive-to-brake transitions: n={len(transition)} "
                  f"mean={statistics.mean(transition):+.5f} rad "
                  f"max_abs={max(abs(x) for x in transition):.5f} rad")


def report_encoder_slip(
    truth: list[dict[str, Any]], plant: list[dict[str, Any]],
    left_encoder: list[dict[str, Any]], right_encoder: list[dict[str, Any]],
    throttle_rows: list[dict[str, Any]],
) -> None:
    """Estimate rear longitudinal slip from measured wheel angle and GT body motion.

    This is an interval-average kinematic proxy, not Unity's internal contact
    slip: encoders are 40 Hz, body twist is at the COM, and each interval uses
    the known 25 ms simulator sampling period. Plant-reported slip is shown as
    a model-vs-proxy comparison, never labelled independent truth.
    """
    if min(len(truth), len(left_encoder), len(right_encoder)) < 3:
        return

    def keyed(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
        return {round(row["stamp"], 6): row for row in rows}

    gt_by_t = keyed(truth)
    left_by_t, right_by_t = keyed(left_encoder), keyed(right_encoder)
    plant_by_t = keyed(plant)
    common = sorted(set(gt_by_t) & set(left_by_t) & set(right_by_t))
    rows: list[dict[str, float]] = []
    for previous_t, current_t in zip(common, common[1:]):
        if current_t - previous_t > 0.1:
            continue
        previous_gt, current_gt = gt_by_t[previous_t], gt_by_t[current_t]
        dt = ENCODER_SAMPLE_PERIOD_S
        enc_w = []
        for enc_by_t in (left_by_t, right_by_t):
            delta = enc_by_t[current_t]["angle"] - enc_by_t[previous_t]["angle"]
            enc_w.append(delta / dt)
        u_mid = 0.5 * (previous_gt["u"] + current_gt["u"])
        r_mid = 0.5 * (previous_gt["r"] + current_gt["r"])
        # Body coordinates: u forward, y left, positive r counter-clockwise.
        # The rear wheel centers are +/-0.118 m from the vehicle centerline.
        wheel_forward = (u_mid - r_mid * REAR_HALF_TRACK_M,
                         u_mid + r_mid * REAR_HALF_TRACK_M)
        interval: dict[str, float] = {
            "speed": abs(u_mid),
            "throttle": float((nearest_by_stamp(
                throttle_rows, current_gt["bag_time"]) or {}).get("value", math.nan)),
        }
        for side, index, vx in (("left", 2, wheel_forward[0]),
                                ("right", 3, wheel_forward[1])):
            surface = WHEEL_RADIUS_M * enc_w[0 if side == "left" else 1]
            slip_guide = ((surface - vx) / abs(vx)) if abs(vx) >= 0.5 else math.nan
            # PhysX's active-torque branch uses max(|v|, |R*omega|)+0.1m/s.
            slip_physx_active = (surface - vx) / (max(abs(vx), abs(surface)) + 0.1)
            slip_physx_coast = (surface - vx) / max(4.0, abs(vx), abs(surface))
            interval[f"{side}_surface_minus_body"] = surface - vx
            interval[f"{side}_slip_guide"] = slip_guide
            interval[f"{side}_slip_physx_active"] = slip_physx_active
            interval[f"{side}_slip_physx_coast"] = slip_physx_coast
            p = plant_by_t.get(current_t)
            if p:
                interval[f"{side}_reported_slip"] = p["long_slip"][index]
                interval[f"{side}_model_omega"] = p["wheel_omega"][index]
                interval[f"{side}_encoder_omega"] = enc_w[0 if side == "left" else 1]
                previous_plant = plant_by_t.get(previous_t)
                if previous_plant:
                    plant_delta = math.remainder(
                        p["wheel_angle"][index] - previous_plant["wheel_angle"][index],
                        10.0 * math.pi)
                    interval[f"{side}_model_interval_omega"] = plant_delta / dt
                    interval[f"{side}_encoder_minus_model_omega"] = (
                        enc_w[0 if side == "left" else 1] - plant_delta / dt)
                interval[f"{side}_model_slip_minus_active"] = (
                    p["long_slip"][index] - slip_physx_active)
        rows.append(interval)

    if not rows:
        return
    print("\nRear-wheel longitudinal-slip diagnostic (40 Hz encoder/GT proxy):")
    print("  This is not direct Unity contact-slip truth; wheel/body mismatch is interval-averaged.")
    print(f"  matched intervals={len(rows)}; encoder omega uses fixed {ENCODER_SAMPLE_PERIOD_S*1000:.0f} ms")

    def summarize(label: str, selected: list[dict[str, float]]) -> None:
        if not selected:
            return
        print(f"  {label}: n={len(selected)}")
        for side in ("left", "right"):
            prefix = f"{side}_"
            surface = [x[prefix + "surface_minus_body"] for x in selected]
            print(f"    {side}: Rω−v_wheel bias={statistics.mean(surface):+.3f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in surface)):.3f} m/s")
            for suffix in ("slip_guide", "slip_physx_active", "slip_physx_coast"):
                values = [x[prefix + suffix] for x in selected
                          if math.isfinite(x[prefix + suffix])]
                if values:
                    print(f"      {suffix}: median={statistics.median(values):+.3f} "
                          f"p95_abs={percentile([abs(v) for v in values], 0.95):.3f}")
            residual_key = prefix + "model_slip_minus_active"
            residuals = [x[residual_key] for x in selected if residual_key in x]
            if residuals:
                print(f"      source-plant slip minus encoder/GT active-slip proxy: "
                      f"bias={statistics.mean(residuals):+.3f} "
                      f"RMSE={math.sqrt(statistics.mean(x*x for x in residuals)):.3f}")
            omega_residual_key = prefix + "encoder_minus_model_omega"
            omega_residuals = [x[omega_residual_key] for x in selected
                               if omega_residual_key in x]
            if omega_residuals:
                print(f"      encoderΔθ/25ms minus source-plant Δθ/25ms: "
                      f"bias={statistics.mean(omega_residuals):+.2f} "
                      f"RMSE={math.sqrt(statistics.mean(x*x for x in omega_residuals)):.2f} rad/s")

    summarize("all", rows)
    summarize("zero-throttle/brake command", [x for x in rows if x["throttle"] < 0.01])
    summarize("positive throttle", [x for x in rows if x["throttle"] >= 0.01])
    summarize("wheel/body speed mismatch >0.5 m/s", [
        x for x in rows if max(abs(x["left_surface_minus_body"]),
                               abs(x["right_surface_minus_body"])) > 0.5])


def report_imu_velocity_propagation(
    truth: list[dict[str, Any]], odom: list[dict[str, Any]],
    imu: list[dict[str, Any]], odom_diagnostics: list[dict[str, Any]],
    throttle_feedback: list[dict[str, Any]],
) -> None:
    """Replay explicit Unity body-frame kinematics with sampled IMU accel."""
    gt_by_stamp = {round(row["stamp"], 6): row for row in truth}
    odom_by_stamp = {round(row["stamp"], 6): row for row in odom}
    imu_by_stamp = {round(row["stamp"], 6): row for row in imu}
    common = sorted(set(gt_by_stamp) & set(odom_by_stamp) & set(imu_by_stamp))
    if len(common) < 3:
        return
    # The body-frame velocity equations follow rotating-frame kinematics:
    # du/dt = a_x + r*v, dv/dt = a_y - r*u. IMU.cs publishes Rigidbody
    # acceleration rotated into the vehicle frame. Midpoint integration uses
    # adjacent 40 Hz samples; it uses no truth after initial alignment.
    first_odom = odom_by_stamp[common[0]]
    first_imu = imu_by_stamp[common[0]]
    u_hat = first_odom["u"]
    r0 = first_imu["yaw_rate"]
    v_com_hat = first_odom["v"] + r0 * COM_TO_REAR_AXLE_M
    estimates = [(u_hat, v_com_hat - r0 * COM_TO_REAR_AXLE_M)]
    one_step_errors: list[tuple[float, float]] = []
    plant_increment_errors: list[float] = []
    accel_increment_errors: list[float] = []
    fused_rollouts = {gain: [odom_by_stamp[common[0]]["u"]]
                      for gain in (0.0, 0.25, 0.5, 0.75, 1.0)}
    state_corrections = {gain: [odom_by_stamp[common[0]]["u"]]
                         for gain in (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)}
    speed_conditioned_correction = [odom_by_stamp[common[0]]["u"]]
    gated_rollouts = {name: [odom_by_stamp[common[0]]["u"]]
                      for name in ("brake", "encoder_slip_0p5", "brake_or_slip")}
    gated_counts = Counter()
    for i in range(1, len(common)):
        previous, current = (imu_by_stamp[common[i - 1]], imu_by_stamp[common[i]])
        dt = (common[i] - common[i - 1])
        if not 0.0 < dt <= 0.1:
            estimates.append((u_hat, v_com_hat - current["yaw_rate"] * COM_TO_REAR_AXLE_M))
            continue
        gt_prev, gt_now = gt_by_stamp[common[i - 1]], gt_by_stamp[common[i]]
        odom_prev = odom_by_stamp[common[i - 1]]
        odom_now = odom_by_stamp[common[i]]
        r_mid = 0.5 * (previous["yaw_rate"] + current["yaw_rate"])
        ax_mid = 0.5 * (previous["ax"] + current["ax"])
        ay_mid = 0.5 * (previous["ay"] + current["ay"])
        u_stage = u_hat + 0.5 * dt * (previous["ax"] + previous["yaw_rate"] * v_com_hat)
        v_stage = v_com_hat + 0.5 * dt * (previous["ay"] - previous["yaw_rate"] * u_hat)
        u_hat += dt * (ax_mid + r_mid * v_stage)
        v_com_hat += dt * (ay_mid - r_mid * u_stage)
        rear_hat = v_com_hat - current["yaw_rate"] * COM_TO_REAR_AXLE_M
        estimates.append((u_hat, rear_hat))
        true_delta_u = gt_now["u"] - gt_prev["u"]
        imu_delta_u = dt * (ax_mid + r_mid * (
            odom_prev["v"] + previous["yaw_rate"] * COM_TO_REAR_AXLE_M))
        # Current raw source-plant output is stored in the diagnostic array,
        # sequence-paired with this 40 Hz packet. Its increment starts from
        # the previous sensor-corrected production state.
        current_diag = odom_diagnostics[i] if len(odom_diagnostics) == len(common) else {}
        has_raw_plant = (
            ("odom_v6" in current_diag.get("label", "") or
             "sensor_corrected_source_derived_odom_v" in current_diag.get("label", "")) and
            len(current_diag.get("values", [])) >= 35
        )
        raw_model_u = current_diag["values"][10] if has_raw_plant else odom_now["u"]
        model_delta_u = raw_model_u - odom_prev["u"]
        diag_values = current_diag.get("values", [])
        encoder_speed = diag_values[7] if has_raw_plant else math.nan
        feedback = nearest_by_stamp(throttle_feedback, gt_now["bag_time"])
        braking = feedback is not None and float(feedback["value"]) <= 0.005
        encoder_slip = math.isfinite(encoder_speed) and abs(encoder_speed - raw_model_u) > 0.5
        gates = {
            "brake": braking,
            "encoder_slip_0p5": encoder_slip,
            "brake_or_slip": braking or encoder_slip,
        }
        for name, active in gates.items():
            if active:
                gated_counts[name] += 1
            gated_rollouts[name].append(gated_rollouts[name][-1] +
                (imu_delta_u if active else model_delta_u))
        plant_increment_errors.append(model_delta_u - true_delta_u)
        accel_increment_errors.append(imu_delta_u - true_delta_u)
        for gain, rollout in fused_rollouts.items():
            rollout.append(rollout[-1] + (1.0 - gain) *
                           model_delta_u + gain * imu_delta_u)
        for gain, rollout in state_corrections.items():
            accel_prediction = rollout[-1] + imu_delta_u
            rollout.append(raw_model_u + gain * (accel_prediction - raw_model_u))
        # Candidate only: the model is trusted more at low speed, where the
        # current practice capture shows lower incremental model error; use a
        # smoothly increasing IMU correction as speed rises.  The two scale
        # constants are exposed here for paired-run evaluation, not hidden in
        # production code.
        speed_gain = 0.25 + 0.5 * min(abs(odom_prev["u"]) / 4.0, 1.0)
        accel_prediction = speed_conditioned_correction[-1] + imu_delta_u
        speed_conditioned_correction.append(
            raw_model_u + speed_gain * (accel_prediction - raw_model_u))
        # Oracle-initialized one-step error isolates sampled-acceleration
        # integration from accumulated state/initialization error.
        gt_u0, gt_v0 = gt_prev["u"], gt_prev["v"]
        gt_u_stage = gt_u0 + 0.5 * dt * (previous["ax"] + previous["yaw_rate"] * gt_v0)
        gt_v_stage = gt_v0 + 0.5 * dt * (previous["ay"] - previous["yaw_rate"] * gt_u0)
        pred_u = gt_u0 + dt * (ax_mid + r_mid * gt_v_stage)
        pred_v = gt_v0 + dt * (ay_mid - r_mid * gt_u_stage)
        one_step_errors.append((pred_u - gt_now["u"],
                                (pred_v - gt_now["v"]) - current["yaw_rate"] * COM_TO_REAR_AXLE_M))
    print("\nIMU-only velocity propagation candidate (midpoint, 25 ms samples):")
    for label, errors in (
        ("one-step, initialized from each prior GT state", one_step_errors),
        ("recursive rollout from first production odom state", [
            (estimate[0] - gt_by_stamp[key]["u"],
             estimate[1] - (gt_by_stamp[key]["v"] - gt_by_stamp[key]["r"] * COM_TO_REAR_AXLE_M))
            for key, estimate in zip(common[1:], estimates[1:])]),
    ):
        if not errors:
            continue
        print(f"  {label}: n={len(errors)}")
        for index, field in ((0, "u"), (1, "v_rear")):
            values = [error[index] for error in errors]
            print(f"    {field}: bias={statistics.mean(values):+.4f} "
                  f"RMSE={math.sqrt(statistics.mean(x*x for x in values)):.4f} "
                  f"p95_abs={percentile([abs(x) for x in values], 0.95):.4f} "
                  f"max_abs={max(abs(x) for x in values):.4f} m/s")
    print("\nOne-step forward-velocity increment comparison:")
    for label, errors in (("native PhysX increment", plant_increment_errors),
                          ("IMU kinematic increment", accel_increment_errors)):
        print(f"  {label}: bias={statistics.mean(errors):+.4f} "
              f"RMSE={math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"p95_abs={percentile([abs(x) for x in errors], 0.95):.4f} m/s")
    print("  recursive increment blend (diagnostic gain sweep, not a fitted model):")
    for gain, rollout in fused_rollouts.items():
        errors = [value - gt_by_stamp[key]["u"]
                  for value, key in zip(rollout[1:], common[1:])]
        print(f"    IMU delta weight={gain:.2f}: RMSE="
              f"{math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"end_error={errors[-1]:+.3f} m/s")
    print("  plant-state prediction corrected toward integrated IMU delta:")
    for gain, rollout in state_corrections.items():
        errors = [value - gt_by_stamp[key]["u"]
                  for value, key in zip(rollout[1:], common[1:])]
        print(f"    correction gain={gain:.2f}: RMSE="
              f"{math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"end_error={errors[-1]:+.3f} m/s")
    conditioned_errors = [value - gt_by_stamp[key]["u"]
                          for value, key in zip(
                              speed_conditioned_correction[1:], common[1:])]
    print("    speed-conditioned correction (0.25 + 0.50*min(|u|/4,1)): "
          f"RMSE={math.sqrt(statistics.mean(x*x for x in conditioned_errors)):.4f} "
          f"end_error={conditioned_errors[-1]:+.3f} m/s")
    print("  causal IMU delta substitution only in physically flagged intervals:")
    for name, rollout in gated_rollouts.items():
        errors = [value - gt_by_stamp[key]["u"]
                  for value, key in zip(rollout[1:], common[1:])]
        print(f"    {name}: flagged={gated_counts[name]} RMSE="
              f"{math.sqrt(statistics.mean(x*x for x in errors)):.4f} "
              f"end_error={errors[-1]:+.3f} m/s")


def load_trajectory(path: Path | None) -> tuple[list[dict[str, float]], float] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty trajectory: {path}")
    first = rows[0]
    s_name = "# s_m" if "# s_m" in first else "s_m"
    points = [
        {
            "s": float(row[s_name]),
            "x": float(row["x_m"]),
            "y": float(row["y_m"]),
            "psi": float(row["psi_rad"]),
        }
        for row in rows
    ]
    return points, points[-1]["s"] + points[1]["s"] - points[0]["s"]


def trajectory_pose(
    trajectory: tuple[list[dict[str, float]], float], progress: float,
) -> tuple[float, float, float]:
    points, lap_length = trajectory
    s_value = progress % lap_length
    positions = [point["s"] for point in points]
    index = max(0, min(len(points) - 2, bisect.bisect_right(positions, s_value) - 1))
    a, b = points[index], points[index + 1]
    ratio = (s_value - a["s"]) / max(1e-9, b["s"] - a["s"])
    dpsi = normalized_angle(b["psi"] - a["psi"])
    psi = a["psi"] + ratio * dpsi
    x = a["x"] + ratio * (b["x"] - a["x"])
    y = a["y"] + ratio * (b["y"] - a["y"])
    return x, y, psi


def report_horizon_errors(
    truth: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
    trajectory: tuple[list[dict[str, float]], float] | None,
) -> None:
    print(
        "horizon comparison note: closed-loop logs replan controls every 25 ms; "
        "beyond already-queued commands these errors are not fixed-input plant accuracy."
    )
    truth_times = [row["stamp"] for row in truth]
    errors: dict[int, list[dict[str, float]]] = {}
    for row in diagnostics:
        diag = row.get("json", {})
        control_ns = diag.get("control_ros_stamp_ns")
        if not isinstance(control_ns, int):
            continue
        start_time = control_ns * 1e-9
        control_prediction = diag.get("control_time_prediction", {})
        map_start = control_prediction.get("predicted_map_pose")
        start_index = bisect.bisect_left(truth_times, start_time)
        candidates = [i for i in (start_index - 1, start_index) if 0 <= i < len(truth)]
        if not candidates:
            continue
        start_truth = min(candidates, key=lambda i: abs(truth_times[i] - start_time))
        actual_start = truth[start_truth]
        for prediction in diag.get("predictions", []):
            n = prediction.get("n")
            state = prediction.get("state")
            if not isinstance(n, int) or not isinstance(state, list) or len(state) < 5:
                continue
            target_time = start_time + n * 0.025
            idx = bisect.bisect_left(truth_times, target_time)
            candidates = [i for i in (idx - 1, idx) if 0 <= i < len(truth)]
            if not candidates:
                continue
            target_index = min(candidates, key=lambda i: abs(truth_times[i] - target_time))
            if abs(truth_times[target_index] - target_time) > 0.018:
                continue
            actual = truth[target_index]
            item = {
                "dt": abs(truth_times[target_index] - target_time),
                "u": state[2] - actual["u"],
                "v": state[3] - actual["v"],
                "r": state[4] - actual["r"],
                "start_speed": math.hypot(actual_start["u"], actual_start["v"]),
            }
            if trajectory and isinstance(map_start, list) and len(map_start) == 3:
                rx, ry, rpsi = trajectory_pose(trajectory, prediction["s_m"])
                predicted_map_x = rx - state[0] * math.sin(rpsi)
                predicted_map_y = ry + state[0] * math.cos(rpsi)
                delta_yaw = normalized_angle(actual_start["yaw"] - map_start[2])
                dx, dy = predicted_map_x - map_start[0], predicted_map_y - map_start[1]
                predicted_x = actual_start["x"] + math.cos(delta_yaw) * dx - math.sin(delta_yaw) * dy
                predicted_y = actual_start["y"] + math.sin(delta_yaw) * dx + math.cos(delta_yaw) * dy
                item["xy"] = math.hypot(
                    predicted_x - actual["x"], predicted_y - actual["y"],
                )
                predicted_yaw = actual_start["yaw"] + normalized_angle(
                    rpsi + state[1] - map_start[2]
                )
                item["yaw"] = normalized_angle(predicted_yaw - actual["yaw"])
            errors.setdefault(n, []).append(item)

    for n, rows in sorted(errors.items()):
        print(
            f"horizon {n} ({n*25}ms): n={len(rows)} "
            f"time_align_p95={percentile([x['dt'] for x in rows], 0.95)*1000:.2f}ms"
        )
        for field in ("u", "v", "r", "xy", "yaw"):
            values = [row[field] for row in rows if field in row]
            if not values:
                continue
            absolute = [abs(value) for value in values]
            print(
                f"  {field}: RMSE={math.sqrt(statistics.mean(x*x for x in values)):.4f} "
                f"p95_abs={percentile(absolute, 0.95):.4f} max_abs={max(absolute):.4f}"
            )
        for lower, upper in ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, math.inf)):
            subset = [row["u"] for row in rows if lower <= row["start_speed"] < upper]
            if subset:
                rmse = math.sqrt(statistics.mean(value * value for value in subset))
                print(f"  speed_bin [{lower:g},{upper:g}): u_RMSE={rmse:.4f} n={len(subset)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="rosbag directory")
    parser.add_argument("--trajectory", type=Path, help="raceline CSV used for the run")
    args = parser.parse_args()
    data = read_bag(args.bag)
    truth = data.get(GT, [])
    if not truth:
        parser.error(f"bag has no simulator truth stream: {GT}")

    print(f"truth: frame={truth[0]['frame']} child={truth[0]['child']} count={len(truth)}")
    for topic in (GT, ODOM, ODOM_DIAGNOSTICS, DIAGNOSTICS, THROTTLE_COMMAND):
        print_interval(topic, data.get(topic, []))
    for topic, label in ((ODOM, "production odom"), (EKF, "EKF odom")):
        if data.get(topic):
            report_state_errors(truth, data[topic], label)

    speeds = [math.hypot(row["u"], row["v"]) for row in truth]
    print(
        f"GT speed: median={statistics.median(speeds):.3f}m/s "
        f"p95={percentile(speeds, 0.95):.3f}m/s max={max(speeds):.3f}m/s"
    )
    for topic, label in ((LAP, "laps"), (COLLISION, "collisions"),
                         (LAST_LAP, "last_lap_s"), (BEST_LAP, "best_lap_s")):
        values = [row["value"] for row in data.get(topic, [])]
        if values:
            print(f"{label}: first={values[0]} last={values[-1]} max={max(values)}")
    for topic in (THROTTLE, THROTTLE_COMMAND, STEERING, STEERING_COMMAND):
        values = [row["value"] for row in data.get(topic, [])]
        if values:
            print(
                f"{topic}: min={min(values):.4f} max={max(values):.4f} "
                f"mean={statistics.mean(values):.4f}"
            )
    report_odom_diagnostics(
        truth, data.get(ODOM, []), data.get(ODOM_DIAGNOSTICS, []),
        data.get(STEERING, []), data.get(THROTTLE, []),
    )
    report_encoder_phase(
        data.get(LEFT_ENCODER, []), data.get(RIGHT_ENCODER, []),
        data.get(ODOM_DIAGNOSTICS, []), data.get(THROTTLE, []))
    if data.get(ODOM):
        report_imu_velocity_propagation(
            truth, data[ODOM], data.get(IMU, []), data.get(ODOM_DIAGNOSTICS, []),
            data.get(THROTTLE, []))
    plant_rows = data.get(PLANT, [])
    if plant_rows:
        print(f"plant schema versions: {sorted({row['schema'] for row in plant_rows})}")
        print("wheel omega has no independent Unity truth channel; do not score it as truth.")
        report_encoder_slip(
            truth, plant_rows, data.get(LEFT_ENCODER, []),
            data.get(RIGHT_ENCODER, []), data.get(THROTTLE, []))
    diagnostics = [row for row in data.get(DIAGNOSTICS, []) if row.get("json")]
    if diagnostics:
        payloads = [row["json"] for row in diagnostics]
        print(f"MPC status counts: {dict(Counter(x.get('status') for x in payloads))}")
        print(f"MPC plant-state source counts: {dict(Counter(x.get('plant_state_source') for x in payloads))}")
        report_horizon_errors(truth, diagnostics, load_trajectory(args.trajectory))
    for row in data.get(BRIDGE_FAULT, []):
        if row["value"]:
            print(f"bridge timing fault detail: {row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
