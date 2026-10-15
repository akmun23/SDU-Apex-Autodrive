#!/usr/bin/env python3
"""Check source-derived Unity IMU rigid-body propagation on sealed run bags.

This is a diagnostic, not a production observer. Simulator truth supplies only
the initial body velocity and offline scoring labels. Each subsequent update
uses the current 40 Hz IMU acceleration and gyro sample on the fixed 25 ms
physics grid, with the source-derived COM-to-rear-axle acceleration shift.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_open_plane_body_dynamics import MotionSample, load_capture
from tools import analyze_open_plane_dynamics as bag_analysis


DT_S = 0.025
COM_TO_REAR_AXLE_M = 0.15532


def _metrics(errors: np.ndarray) -> dict[str, Any]:
    if errors.size == 0:
        return {"samples": 0, "bias": [None, None], "rmse": [None, None],
                "p95_abs": [None, None]}
    return {
        "samples": int(errors.shape[0]),
        "bias": np.mean(errors, axis=0).tolist(),
        "rmse": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
        "p95_abs": np.quantile(np.abs(errors), 0.95, axis=0).tolist(),
    }


def _valid_samples(path: Path) -> list[MotionSample]:
    capture = load_capture(
        path, include_nonvalid_phases=True, continuous_phased_run=True)
    samples = sorted(
        (sample for sequence in capture.sequences for sample in sequence
         if sample.packet_sequence >= 0
         and sample.imu_acceleration_mps2 is not None
         and sample.imu_yaw_rate_rps is not None
         and np.isfinite(sample.state).all()
         and np.isfinite(sample.imu_acceleration_mps2).all()),
        key=lambda sample: sample.packet_sequence)
    if len(samples) < 2:
        raise ValueError(f"not enough aligned IMU/truth samples: {path}")
    return samples


def _read_imu_xyz(path: Path) -> tuple[list[int], list[np.ndarray]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = bag_analysis._topic_map(connection)
        if bag_analysis.IMU not in topics:
            raise ValueError(f"bag has no {bag_analysis.IMU}: {path}")
        rows = []
        for receipt_ns, message in bag_analysis._messages(
                connection, topics, bag_analysis.IMU):
            values = np.asarray((
                message.linear_acceleration.x,
                message.linear_acceleration.y,
                message.linear_acceleration.z,
            ), dtype=np.float64)
            if np.isfinite(values).all():
                rows.append((int(receipt_ns), values))
    finally:
        connection.close()
    rows.sort(key=lambda row: row[0])
    return ([row[0] for row in rows], [row[1] for row in rows])


def _causal_imu_z(sample: MotionSample, times: list[int], values: list[np.ndarray]
                  ) -> float | None:
    index = bisect.bisect_right(times, sample.receipt_ns) - 1
    if index < 0 or sample.receipt_ns - times[index] > 35_000_000:
        return None
    return float(values[index][2])


def _rigid_body_derivative(state: np.ndarray, acceleration: np.ndarray,
                           angular_velocity: np.ndarray) -> np.ndarray:
    u, v, w = state
    p, q, r = angular_velocity
    ax, ay, az = acceleration
    return np.asarray((ax + r * v - q * w,
                       ay + p * w - r * u,
                       az + q * u - p * v))


def _accel_at_rear(sample: MotionSample, yaw_accel: float) -> np.ndarray:
    ax_com, ay_com = np.asarray(sample.imu_acceleration_mps2, dtype=np.float64)
    yaw_rate = float(sample.imu_yaw_rate_rps)
    return np.asarray((
        ax_com + COM_TO_REAR_AXLE_M * yaw_rate ** 2,
        ay_com - COM_TO_REAR_AXLE_M * yaw_accel,
    ))


def analyze_run(path: Path) -> dict[str, Any]:
    samples = _valid_samples(path)
    imu_times, imu_values = _read_imu_xyz(path)
    one_step: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "previous_sample", "current_sample", "current_sample_rk2",
            "trapezoid", "oracle_full_3d_cross_terms")}
    recursive_errors: list[np.ndarray] = []
    recursive_3d_errors: list[np.ndarray] = []
    step_ids: list[int] = []
    segment_count = 0
    predicted = np.asarray(samples[0].state[:2], dtype=np.float64)
    initial_r = float(samples[0].imu_yaw_rate_rps)
    predicted_3d = np.asarray((
        float(samples[0].state[0]),
        float(samples[0].state[1]) + COM_TO_REAR_AXLE_M * initial_r,
        0.0,
    ))
    previous = samples[0]
    segment_count = 1

    for current in samples[1:]:
        if current.packet_sequence != previous.packet_sequence + 1:
            predicted = np.asarray(current.state[:2], dtype=np.float64)
            current_r = float(current.imu_yaw_rate_rps)
            predicted_3d = np.asarray((
                float(current.state[0]),
                float(current.state[1]) + COM_TO_REAR_AXLE_M * current_r,
                0.0,
            ))
            previous = current
            segment_count += 1
            continue

        r0 = float(previous.imu_yaw_rate_rps)
        r1 = float(current.imu_yaw_rate_rps)
        yaw_accel = (r1 - r0) / DT_S
        a0 = _accel_at_rear(previous, yaw_accel)
        a1 = _accel_at_rear(current, yaw_accel)
        u0, v0 = np.asarray(previous.state[:2], dtype=np.float64)
        u1, v1 = np.asarray(current.state[:2], dtype=np.float64)

        # Evaluate the body-frame equations with a teacher-forced initial state.
        # "current_sample" is the causal endpoint sample consumed by Odom.
        derivative_previous = np.asarray((
            a0[0] + r0 * v0, a0[1] - r0 * u0))
        derivative_current = np.asarray((
            a1[0] + r1 * v0, a1[1] - r1 * u0))
        derivative_trapezoid = 0.5 * (derivative_previous + derivative_current)
        truth_delta = np.asarray((u1 - u0, v1 - v0))
        mid_u = u0 + 0.5 * DT_S * derivative_current[0]
        mid_v = v0 + 0.5 * DT_S * derivative_current[1]
        rk2_next = np.asarray((
            u0 + DT_S * (a1[0] + r1 * mid_v),
            v0 + DT_S * (a1[1] - r1 * mid_u),
        ))
        one_step["previous_sample"].append(
            DT_S * derivative_previous - truth_delta)
        one_step["current_sample"].append(
            DT_S * derivative_current - truth_delta)
        one_step["current_sample_rk2"].append(rk2_next - (u1, v1))
        one_step["trapezoid"].append(
            DT_S * derivative_trapezoid - truth_delta)
        rigid = current.simulator_rigid_state
        roll_pitch_rate = current.imu_roll_pitch_rate_rps
        if rigid is not None and roll_pitch_rate is not None:
            vertical_speed = float(rigid[9])
            roll_rate, pitch_rate = np.asarray(roll_pitch_rate, dtype=np.float64)
            v_com = v1 + COM_TO_REAR_AXLE_M * r1
            oracle_full_3d = np.asarray((
                u0 + DT_S * (float(current.imu_acceleration_mps2[0]) +
                             r1 * v_com - pitch_rate * vertical_speed) - u1,
                v0 + DT_S * (float(current.imu_acceleration_mps2[1]) +
                             roll_rate * vertical_speed - r1 * u1 -
                             COM_TO_REAR_AXLE_M * yaw_accel) - v1,
            ))
            one_step["oracle_full_3d_cross_terms"].append(oracle_full_3d)

        # Sensor-only causal propagation; truth is not used after initialization.
        du0 = a1[0] + r1 * predicted[1]
        dv0 = a1[1] - r1 * predicted[0]
        mid_u = predicted[0] + 0.5 * DT_S * du0
        mid_v = predicted[1] + 0.5 * DT_S * dv0
        predicted = predicted + DT_S * np.asarray((
            a1[0] + r1 * mid_v,
            a1[1] - r1 * mid_u,
        ))
        recursive_errors.append(predicted - np.asarray(current.state[:2]))

        imu_z = _causal_imu_z(current, imu_times, imu_values)
        if imu_z is not None and current.imu_roll_pitch_rate_rps is not None:
            acceleration_3d = np.asarray((
                current.imu_acceleration_mps2[0],
                current.imu_acceleration_mps2[1],
                imu_z,
            ), dtype=np.float64)
            angular_3d = np.asarray((
                current.imu_roll_pitch_rate_rps[0],
                current.imu_roll_pitch_rate_rps[1],
                r1,
            ), dtype=np.float64)
            k1 = _rigid_body_derivative(predicted_3d, acceleration_3d, angular_3d)
            midpoint_3d = predicted_3d + 0.5 * DT_S * k1
            predicted_3d += DT_S * _rigid_body_derivative(
                midpoint_3d, acceleration_3d, angular_3d)
            predicted_rear_v = predicted_3d[1] - COM_TO_REAR_AXLE_M * r1
            recursive_3d_errors.append(np.asarray((
                predicted_3d[0] - current.state[0],
                predicted_rear_v - current.state[1],
            )))
        step_ids.append(int(current.packet_sequence))
        previous = current

    one_step_metrics = {
        name: _metrics(np.asarray(rows, dtype=np.float64).reshape((-1, 2)))
        for name, rows in one_step.items()
    }
    errors = np.asarray(recursive_errors, dtype=np.float64).reshape((-1, 2))
    errors_3d = np.asarray(recursive_3d_errors, dtype=np.float64).reshape((-1, 2))
    elapsed = max(0.0, (len(errors)) * DT_S)
    return {
        "run": path.parent.parent.name,
        "bag": str(path),
        "bag_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "samples": len(samples),
        "continuous_segments": segment_count,
        "physics_dt_s": DT_S,
        "com_to_rear_axle_m": COM_TO_REAR_AXLE_M,
        "one_step_increment_error": one_step_metrics,
        "recursive_sensor_propagation": {
            "interpretation": (
                "IMU-only causal velocity integration from one simulator-truth "
                "initial velocity per packet-contiguous segment; no later truth "
                "or wheel updates are used. This is not a command-only future "
                "plant rollout"),
            "elapsed_s": elapsed,
            "body_u_v_error": _metrics(errors),
            "endpoint_error_mps": errors[-1].tolist() if len(errors) else [None, None],
        },
        "recursive_full_3d_sensor_propagation": {
            "interpretation": (
                "Rigid-body Newton-Euler velocity equations in the COM frame, "
                "with x/y/z IMU acceleration and p/q/r gyro. Initial u/v use "
                "simulator truth for scoring; initial vertical speed is set to "
                "zero. No truth or wheel updates are used afterward."),
            "samples": int(len(errors_3d)),
            "elapsed_s": len(errors_3d) * DT_S,
            "body_u_rear_v_error": _metrics(errors_3d),
            "endpoint_error_mps": (
                errors_3d[-1].tolist() if len(errors_3d) else [None, None]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [analyze_run(path) for path in args.bags]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": "Unity IMU rigid-body equations with fixed 25 ms packets",
        "truth_policy": "offline initialization and scoring only",
        "runs": reports,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "runs": [{
            "run": report["run"],
            "one_step_current_rmse_mps": report["one_step_increment_error"][
                "current_sample"]["rmse"],
        "oracle_full_3d_rmse_mps": report["one_step_increment_error"][
            "oracle_full_3d_cross_terms"]["rmse"],
            "recursive_u_v_rmse_mps": report["recursive_sensor_propagation"][
                "body_u_v_error"]["rmse"],
            "recursive_endpoint_error_mps": report[
                "recursive_sensor_propagation"]["endpoint_error_mps"],
        } for report in reports],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
