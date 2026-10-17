#!/usr/bin/env python3
"""Score a recursive PhysX velocity rollout with the permitted ROS IMU yaw.

This is an offline evaluation tool. Ground-truth capture fields are labels
only; runtime inputs are the legal ROS command stream used to make the plant
rollout and the ROS IMU orientation sampled causally from the bag. It never
reads bridge packet-timing diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


ROOT = Path(__file__).resolve().parents[2]
IMU_TOPIC = "/autodrive/roboracer_1/imu"


def wrap(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def yaw_from_xyzw(q: np.ndarray) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))


def rotate_xyzw(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = q[:3]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + q[3] * twice_cross + np.cross(xyz, twice_cross)


def stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"samples": 0}
    absolute = np.abs(finite)
    return {
        "samples": int(finite.size),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def read_imu(bag: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    database = bag if bag.is_file() else next(bag.glob("*.db3"))
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        start_ns = int(connection.execute(
            "SELECT MIN(timestamp) FROM messages").fetchone()[0])
        topic = connection.execute(
            "SELECT id, type FROM topics WHERE name=?", (IMU_TOPIC,)).fetchone()
        if topic is None:
            raise ValueError(f"bag lacks permitted sensor topic {IMU_TOPIC}")
        topic_id, type_name = topic
        message_type = get_message(type_name)
        times: list[float] = []
        yaw: list[float] = []
        for timestamp, payload in connection.execute(
                "SELECT timestamp, data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp, id", (topic_id,)):
            message = deserialize_message(bytes(payload), message_type)
            q = message.orientation
            times.append((int(timestamp) - start_ns) * 1.0e-9)
            yaw.append(math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
    finally:
        connection.close()
    if len(times) < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("IMU receive-time sequence is invalid")
    return start_ns, np.asarray(times), np.unwrap(np.asarray(yaw)), np.asarray(yaw)


def read_predictions(path: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if len(row) != 6:
                raise ValueError(f"{path}:{line_number}: expected six columns")
            rows.append((
                int(row[0]),
                np.asarray([float(value) for value in row[4].split()]),
                np.asarray([float(value) for value in row[5].split()])))
    if len(rows) < 2 or any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("prediction capture rows must be strictly increasing")
    return rows


def evaluate(capture: Path, bag: Path, prediction_csv: Path,
             command_input: Path, start_row: int,
             source_realtime_minus_bag_s: float) -> dict[str, Any]:
    rollout = json.loads(command_input.read_text(encoding="utf-8"))
    command = rollout.get("command_input", {})
    initialization = rollout.get("input_contract", {})
    if initialization.get("future_state_measurements_included") is not False:
        raise ValueError("rollout must explicitly exclude future state measurements")
    if initialization.get("truth_or_sensor_values_used_to_initialize_state") is not False:
        raise ValueError(
            "this evaluator requires the documented reset-state input; "
            "sensor- or truth-initialized rollouts need a separately labeled evaluation")
    if command.get("offline_only_restricted_packet_debug"):
        raise ValueError("the evaluation input must not use packet debug data")
    if command.get("source") != "ROS bag zero-order-held steering/throttle command topics":
        raise ValueError("rollout must use only the ordinary ROS command topics")
    expected_topics = {
        "/autodrive/roboracer_1/steering_command",
        "/autodrive/roboracer_1/throttle_command",
    }
    if not expected_topics.issubset(set(command.get("topics", []))):
        raise ValueError("rollout metadata does not identify both legal command topics")
    if not math.isfinite(source_realtime_minus_bag_s):
        raise ValueError("clock offset must be finite")

    source = pd.read_csv(capture, comment="#")
    predictions = read_predictions(prediction_csv)
    first_prediction_row = predictions[0][0]
    if start_row >= first_prediction_row:
        raise ValueError("start row must precede the first rollout prediction")
    if predictions[-1][0] >= len(source):
        raise ValueError("prediction capture row exceeds source capture length")
    start = source.iloc[start_row]
    start_position_x = float(start.pos_world_x_m)
    start_position_z = float(start.pos_world_z_m)
    start_yaw_unity = yaw_from_xyzw(np.asarray([
        start.rot_world_x, start.rot_world_y, start.rot_world_z,
        start.rot_world_w], dtype=np.float64))

    bag_start_ns, imu_times, imu_yaw_unwrapped, imu_yaw_raw = read_imu(bag)
    del bag_start_ns  # The bag-relative receive-time axis is used below.
    rows = np.asarray([row for row, _, _ in predictions], dtype=np.int64)
    capture_rows = source.iloc[rows]
    bag_times = (capture_rows.capture_realtime_s.to_numpy(dtype=np.float64)
                 - source_realtime_minus_bag_s)
    imu_indices = np.searchsorted(imu_times, bag_times, side="right") - 1
    if np.any(imu_indices < 0):
        raise ValueError("IMU bag does not cover the first rollout sample")
    imu_age_ms = (bag_times - imu_times[imu_indices]) * 1000.0

    model_yaw = np.empty(len(predictions), dtype=np.float64)
    truth_yaw = np.empty_like(model_yaw)
    imu_yaw = imu_yaw_unwrapped[imu_indices]
    model_uv = np.empty((len(predictions), 2), dtype=np.float64)
    truth_uv = np.empty_like(model_uv)
    truth_position = np.empty_like(model_uv)
    for index, (row, q, world_velocity) in enumerate(predictions):
        sample = source.iloc[row]
        # In the Unity frame, z is forward and x is right. Convert into ROS
        # base_link convention (x forward, y left) after rotating to body axes.
        body_velocity = rotate_xyzw(
            np.asarray([-q[0], -q[1], -q[2], q[3]]), world_velocity)
        model_uv[index] = [body_velocity[2], -body_velocity[0]]
        source_q = np.asarray([
            sample.rot_world_x, sample.rot_world_y,
            sample.rot_world_z, sample.rot_world_w], dtype=np.float64)
        truth_yaw[index] = -wrap(yaw_from_xyzw(source_q) - start_yaw_unity)
        model_yaw[index] = -yaw_from_xyzw(q)
        truth_uv[index] = [float(sample.vel_body_z_mps),
                           -float(sample.vel_body_x_mps)]
        dx = float(sample.pos_world_x_m) - start_position_x
        dz = float(sample.pos_world_z_m) - start_position_z
        truth_position[index] = [
            dx * math.sin(start_yaw_unity) + dz * math.cos(start_yaw_unity),
            -(dx * math.cos(start_yaw_unity) - dz * math.sin(start_yaw_unity)),
        ]

    def integrate(yaw: np.ndarray) -> np.ndarray:
        position = np.zeros(2, dtype=np.float64)
        previous_uv = np.zeros(2, dtype=np.float64)
        previous_yaw = 0.0
        result = np.empty_like(model_uv)
        dt = 0.025
        for index, current_yaw in enumerate(yaw):
            delta_yaw = float(wrap(current_yaw - previous_yaw))
            midpoint_yaw = previous_yaw + 0.5 * delta_yaw
            midpoint_uv = 0.5 * (previous_uv + model_uv[index])
            position += dt * np.asarray([
                midpoint_uv[0] * math.cos(midpoint_yaw)
                - midpoint_uv[1] * math.sin(midpoint_yaw),
                midpoint_uv[0] * math.sin(midpoint_yaw)
                + midpoint_uv[1] * math.cos(midpoint_yaw),
            ])
            result[index] = position
            previous_uv = model_uv[index]
            previous_yaw = float(current_yaw)
        return result

    model_yaw = np.unwrap(model_yaw)
    truth_yaw = np.unwrap(truth_yaw)
    fusion_gains = (0.0, 0.25, 0.50, 0.75, 1.0)
    position_by_yaw = {
        "source_truth_yaw_offline_ceiling": integrate(truth_yaw),
        "native_plant_yaw": integrate(model_yaw),
        # Preserve the IMU's actual receive-time yaw origin. In this run it
        # reports a stale heading immediately after reset, then catches up
        # while the vehicle is stationary; this is intentionally not hidden.
        "permitted_ros_imu_yaw": integrate(imu_yaw),
    }
    for gain in fusion_gains:
        fused_yaw = model_yaw + gain * np.asarray(wrap(imu_yaw - model_yaw))
        position_by_yaw[f"model_imu_blend_gain_{gain:.2f}"] = integrate(fused_yaw)

    # Pick one convex fusion gain on the first temporal half only, then keep
    # it fixed over the complete reset-to-finish rollout. The second half is
    # therefore an untouched temporal holdout for this diagnostic.
    training_count = len(predictions) // 2
    model_error = np.asarray(wrap(model_yaw[:training_count] - truth_yaw[:training_count]))
    yaw_innovation = np.asarray(wrap(imu_yaw[:training_count] - model_yaw[:training_count]))
    denominator = float(np.dot(yaw_innovation, yaw_innovation))
    selected_gain = (float(np.clip(
        -np.dot(model_error, yaw_innovation) / denominator, 0.0, 1.0))
        if denominator > 1.0e-12 else 0.0)
    selected_yaw = model_yaw + selected_gain * np.asarray(
        wrap(imu_yaw - model_yaw))
    position_by_yaw["first_half_selected_blend_full_run"] = integrate(selected_yaw)

    # Separately choose the convex blend by integrated position on the first
    # half. This is a deliberately small scalar observer calibration, not a
    # vehicle-force fit; the second half remains held out for the score.
    position_gain_grid = np.linspace(0.0, 1.0, 101)
    position_gain_scores: list[tuple[float, float]] = []
    for gain in position_gain_grid:
        fused_yaw = model_yaw + gain * np.asarray(wrap(imu_yaw - model_yaw))
        candidate_position = integrate(fused_yaw)
        train_error = np.linalg.norm(
            candidate_position[:training_count] - truth_position[:training_count],
            axis=1)
        position_gain_scores.append((
            float(np.sqrt(np.mean(train_error * train_error))), float(gain)))
    position_selected_gain = min(position_gain_scores)[1]
    position_selected_yaw = model_yaw + position_selected_gain * np.asarray(
        wrap(imu_yaw - model_yaw))
    position_by_yaw["first_half_position_selected_blend_full_run"] = integrate(
        position_selected_yaw)

    # A held IMU orientation is not a measurement of current heading. Test a
    # freshness-gated correction: fit one convex gain using fresh IMU samples
    # from the first half, ignore stale IMU orientation, and leave the second
    # temporal half untouched.
    fresh_imu_age_threshold_ms = 50.0
    fresh_imu = imu_age_ms <= fresh_imu_age_threshold_ms
    fresh_position_scores: list[tuple[float, float]] = []
    for gain in position_gain_grid:
        fused_yaw = model_yaw + np.where(
            fresh_imu, gain * np.asarray(wrap(imu_yaw - model_yaw)), 0.0)
        candidate_position = integrate(fused_yaw)
        train_error = np.linalg.norm(
            candidate_position[:training_count]
            - truth_position[:training_count], axis=1)
        fresh_position_scores.append((
            float(np.sqrt(np.mean(train_error * train_error))), float(gain)))
    fresh_selected_gain = min(fresh_position_scores)[1]
    fresh_gated_yaw = model_yaw + np.where(
        fresh_imu,
        fresh_selected_gain * np.asarray(wrap(imu_yaw - model_yaw)), 0.0)
    fresh_gated_position = integrate(fresh_gated_yaw)
    fresh_gated_position_error = np.linalg.norm(
        fresh_gated_position - truth_position, axis=1)
    position_by_yaw["fresh_imu_gated_model_blend"] = fresh_gated_position
    report: dict[str, Any] = {
        "capture": str(capture.resolve().relative_to(ROOT)),
        "bag": str(bag.resolve().relative_to(ROOT)),
        "command_rollout_input": str(command_input.resolve().relative_to(ROOT)),
        "rollout_pose_csv": str(prediction_csv.resolve().relative_to(ROOT)),
        "runtime_input_policy": {
            "commands": sorted(expected_topics),
            "yaw_anchor": IMU_TOPIC,
            "uses_packet_debug": False,
            "uses_ground_truth_as_input": False,
            "truth_role": "offline scoring only",
        },
        "initialization_provenance": {
            "class": "documented reset state, not sensor-initialized production Odom",
            "description": initialization.get("initialization"),
            "future_state_measurements_included": False,
            "truth_or_sensor_values_used_to_initialize_state": False,
        },
        "samples": int(len(predictions)),
        "duration_s": float((rows[-1] - start_row) * 0.001),
        "command_to_source_clock_offset_s_offline_pairing": source_realtime_minus_bag_s,
        "causal_imu_age_ms": {
            "median": float(np.median(imu_age_ms)),
            "p95": float(np.percentile(imu_age_ms, 95)),
            "max": float(np.max(imu_age_ms)),
            "over_50ms_samples": int(np.count_nonzero(imu_age_ms > 50.0)),
            "over_100ms_samples": int(np.count_nonzero(imu_age_ms > 100.0)),
        },
        "forward_speed_model_vs_truth_mps": stats(model_uv[:, 0] - truth_uv[:, 0]),
        "lateral_speed_model_vs_truth_mps": stats(model_uv[:, 1] - truth_uv[:, 1]),
        "native_plant_yaw_vs_truth_rad": stats(
            np.asarray(wrap(model_yaw - truth_yaw))),
        "permitted_imu_yaw_vs_truth_rad": stats(
            np.asarray(wrap(imu_yaw - truth_yaw))),
        "largest_permitted_imu_yaw_error_samples": [
            {
                "capture_row": int(row),
                "capture_realtime_s": float(source.iloc[row].capture_realtime_s),
                "imu_age_ms": float(age),
                "truth_yaw_rad": float(truth),
                "imu_yaw_rad_unwrapped": float(measured),
                "signed_error_rad": float(wrap(measured - truth)),
            }
            for index in np.argsort(
                np.abs(wrap(imu_yaw - truth_yaw)))[-10:][::-1]
            for row, age, truth, measured in [(
                rows[index], imu_age_ms[index], truth_yaw[index], imu_yaw[index])]
        ],
        "temporal_holdout_fusion": {
            "training_fraction": 0.5,
            "convex_gain_selected_on_first_half": selected_gain,
            "selected_yaw_error_first_half_rad": stats(
                np.asarray(wrap(selected_yaw[:training_count]
                                - truth_yaw[:training_count]))),
            "selected_yaw_error_second_half_rad": stats(
                np.asarray(wrap(selected_yaw[training_count:]
                                - truth_yaw[training_count:]))),
            "position_gain_selected_on_first_half": position_selected_gain,
            "position_gain_train_rmse_m": min(position_gain_scores)[0],
        },
        "freshness_gated_fusion": {
            "imu_freshness_threshold_ms": fresh_imu_age_threshold_ms,
            "fresh_sample_count": int(np.count_nonzero(fresh_imu)),
            "stale_sample_count": int(np.count_nonzero(~fresh_imu)),
            "gain_selected_on_fresh_first_half_samples": fresh_selected_gain,
            "training_position_rmse_m": min(fresh_position_scores)[0],
            "second_half_position_rmse_m": float(np.sqrt(np.mean(
                np.square(fresh_gated_position_error[training_count:])))),
            "full_run_position_rmse_m": float(np.sqrt(np.mean(
                np.square(fresh_gated_position_error)))),
            "full_run_position_p95_abs_m": float(np.percentile(
                fresh_gated_position_error, 95)),
            "second_half_yaw_error_rad": stats(np.asarray(wrap(
                fresh_gated_yaw[training_count:] - truth_yaw[training_count:]))),
        },
        "integrated_position_by_yaw_source": {},
    }
    for label, prediction in position_by_yaw.items():
        position_error = np.linalg.norm(prediction - truth_position, axis=1)
        report["integrated_position_by_yaw_source"][label] = {
            **stats(position_error),
            "final_error_m": float(position_error[-1]),
        }
        if label == "first_half_position_selected_blend_full_run":
            report["temporal_holdout_fusion"][
                "selected_position_error_first_half_m"] = stats(
                    position_error[:training_count])
            report["temporal_holdout_fusion"][
                "selected_position_error_second_half_m"] = stats(
                    position_error[training_count:])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--command-input", type=Path, required=True)
    parser.add_argument("--start-row", type=int, required=True)
    parser.add_argument("--source-realtime-minus-bag-s", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.capture.resolve(), args.bag.resolve(), args.prediction_csv.resolve(),
        args.command_input.resolve(), args.start_row,
        args.source_realtime_minus_bag_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
