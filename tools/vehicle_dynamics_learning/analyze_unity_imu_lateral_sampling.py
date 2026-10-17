#!/usr/bin/env python3
"""Measure lateral-velocity observability under Unity's 1 kHz IMU law.

The capture stores 1 kHz Rigidbody pose/twist, not a separate IMU topic. This
script reconstructs the IMU script's finite-difference world acceleration,
rotates it into the Rigidbody frame, and evaluates the full body-frame
equation dv/dt = ay - r*u + p*w, including the COM-to-rear-axle
yaw-acceleration term. It compares 1 kHz integration with candidate 25 ms
sampling/aggregation rules. Ground truth is used only to synthesize the
documented IMU output and initialize each diagnostic integration; no future
truth is fed back.

This is a sampling/observability diagnostic, not an odometry candidate or a
claim that the competition sensor path is identical to this capture phase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FIELDS = (
    "sim_time_s",
    "pos_world_x_m", "pos_world_y_m", "pos_world_z_m",
    "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
    "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
    "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps",
    "fl_steer_deg", "fr_steer_deg",
)


def rotate(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply xyzw quaternion rotation row-wise."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    xyz, w = q[:, :3], q[:, 3:4]
    return (2.0 * np.sum(xyz * vector, axis=1, keepdims=True) * xyz
            + (w * w - np.sum(xyz * xyz, axis=1, keepdims=True)) * vector
            + 2.0 * w * np.cross(xyz, vector))


def describe(error: np.ndarray) -> dict[str, float | int]:
    error = error[np.isfinite(error)]
    absolute = np.abs(error)
    return {
        "samples": int(error.size),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
        "endpoint_signed_error": float(error[-1]),
    }


def sample_distribution(error: np.ndarray) -> dict[str, float | int]:
    error = error[np.isfinite(error)]
    absolute = np.abs(error)
    return {
        "samples": int(error.size),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "p95_abs": float(np.percentile(absolute, 95)),
        "max_abs": float(np.max(absolute)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-ms", type=float, default=25.0)
    parser.add_argument("--reference-forward-offset-m", type=float,
                        default=0.15532,
                        help="COM-to-rear-axle longitudinal offset")
    parser.add_argument("--yaw-rate-gain-m", type=float, default=0.181939)
    parser.add_argument("--speed-yaw-rate-gain-s", type=float, default=-0.009656)
    parser.add_argument("--lateral-velocity-limit-mps", type=float, default=0.35)
    args = parser.parse_args()
    if args.grid_ms <= 0.0 or not math.isfinite(args.grid_ms):
        parser.error("--grid-ms must be finite and positive")

    with args.capture.open(newline="") as stream:
        metadata: dict[str, str] = {}
        for line in stream:
            if line.startswith("#"):
                key, separator, value = line[1:].strip().partition("=")
                if separator:
                    metadata[key.strip()] = value.strip()
                continue
            header = next(csv.reader([line]))
            break
        missing = sorted(set(FIELDS) - set(header))
        if missing:
            raise ValueError(f"capture is missing fields: {missing}")
        column_indices = [header.index(name) for name in FIELDS]
        rows = []
        for row in csv.reader(stream):
            if row:
                rows.append([float(row[index]) for index in column_indices])

    data = np.asarray(rows, dtype=np.float64)
    if len(data) < 26:
        raise ValueError("capture is too short for the requested sample grid")
    (time_s, *rest) = data.T
    position = np.column_stack(rest[:3])
    quaternion = np.column_stack(rest[3:7])
    velocity_body = np.column_stack(rest[7:10])
    omega_body = np.column_stack(rest[10:13])
    front_steering_rad = np.deg2rad(np.column_stack(rest[13:15]))
    # This older capture schema omits explicit world velocity. Reconstruct it
    # from the captured Rigidbody-local velocity and Rigidbody orientation.
    velocity_world = rotate(quaternion, velocity_body)
    dt = np.diff(time_s)
    fixed_dt = float(np.median(dt))
    if not np.all(np.isfinite(dt)) or np.max(np.abs(dt - fixed_dt)) > 1e-6:
        raise ValueError("capture is not a uniform fixed-step sequence")
    # IMU.FixedUpdate computes (worldVelocity[n]-worldVelocity[n-1])/dt,
    # then expresses that inertial acceleration in the current body frame.
    accel_world = np.diff(velocity_world, axis=0) / dt[:, None]
    u = velocity_body[:, 2]
    v = -velocity_body[:, 0]
    vertical_speed = velocity_body[:, 1]
    yaw_rate = -omega_body[:, 1]
    roll_rate = -omega_body[:, 2]
    nominal_dt = float(metadata.get("fixed_delta_time_s", fixed_dt))
    grid_steps = int(round(args.grid_ms / (nominal_dt * 1000.0)))
    if not math.isclose(grid_steps * nominal_dt * 1000.0, args.grid_ms,
                        abs_tol=1e-6):
        raise ValueError("requested grid must be an integer multiple of capture dt")
    # Respawns or abrupt pose resets are not physical acceleration. Split at
    # those boundaries so a reset impulse cannot contaminate adjacent tests.
    q_norm = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    q_dot = np.abs(np.sum(q_norm[:-1] * q_norm[1:], axis=1))
    rotation_step = 2.0 * np.arccos(np.clip(q_dot, -1.0, 1.0))
    position_step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    boundary = (position_step > 0.1) | (rotation_step > 0.1)
    segment_starts = np.concatenate(([0], np.flatnonzero(boundary) + 1))
    segment_ends = np.concatenate((np.flatnonzero(boundary) + 1, [len(data)]))
    segments = [(int(start), int(end)) for start, end in
                zip(segment_starts, segment_ends) if end - start > grid_steps + 1]
    if not segments:
        raise ValueError("capture contains no continuous segment long enough")

    speed_edges = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 20.0)
    steering_edges = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
    continuous_indices = np.concatenate([
        np.arange(start, end, dtype=np.int64) for start, end in segments])
    continuous_u = np.maximum(0.0, u[continuous_indices])
    continuous_r = yaw_rate[continuous_indices]
    continuous_v_rear = (
        v[continuous_indices]
        - continuous_r * args.reference_forward_offset_m)
    continuous_steering = np.max(
        np.abs(front_steering_rad[continuous_indices]), axis=1)
    gain_all = (args.yaw_rate_gain_m
                + args.speed_yaw_rate_gain_s * continuous_u)
    kinematic_all = (
        np.clip(continuous_r * gain_all,
                -args.lateral_velocity_limit_mps,
                args.lateral_velocity_limit_mps)
        - continuous_r * args.reference_forward_offset_m)
    kinematic_error = kinematic_all - continuous_v_rear
    regime_table: dict[str, dict[str, float | int]] = {}
    for speed_low, speed_high in zip(speed_edges[:-1], speed_edges[1:]):
        for steer_low, steer_high in zip(steering_edges[:-1],
                                         steering_edges[1:]):
            mask = ((continuous_u >= speed_low)
                    & (continuous_u < speed_high)
                    & (continuous_steering >= steer_low)
                    & (continuous_steering < steer_high))
            if not np.any(mask):
                continue
            key = (f"speed_{speed_low:g}_{speed_high:g}_mps__"
                   f"steer_{steer_low:g}_{steer_high:g}_rad")
            regime_table[key] = sample_distribution(kinematic_error[mask])

    phases: list[dict[str, object]] = []
    fine_errors: list[np.ndarray] = []
    fine_errors_3d: list[np.ndarray] = []
    for segment_id, (start, end) in enumerate(segments):
        segment_slice = slice(start, end)
        segment_dt = np.diff(time_s[segment_slice])
        segment_accel_world = np.diff(velocity_world[segment_slice], axis=0) \
            / segment_dt[:, None]
        inverse_q = quaternion[segment_slice][1:].copy()
        inverse_q[:, :3] *= -1.0
        accel_body_xyz = rotate(inverse_q, segment_accel_world)
        # Unity->ROS mapping: forward=+Unity Z, left=-Unity X, yaw=-Unity Y.
        segment_ay_com = np.empty(end - start, dtype=np.float64)
        segment_ay_com[1:] = -accel_body_xyz[:, 0]
        segment_ay_com[0] = segment_ay_com[1]
        segment_u = u[segment_slice]
        segment_w = vertical_speed[segment_slice]
        segment_roll_rate = roll_rate[segment_slice]
        segment_yaw_rate = yaw_rate[segment_slice]
        segment_v = (v[segment_slice]
                     - segment_yaw_rate * args.reference_forward_offset_m)
        segment_yaw_alpha = np.empty(end - start, dtype=np.float64)
        segment_yaw_alpha[1:] = np.diff(segment_yaw_rate) / segment_dt
        segment_yaw_alpha[0] = segment_yaw_alpha[1]
        segment_ay_reference = (
            segment_ay_com
            - segment_yaw_alpha * args.reference_forward_offset_m)
        derivative_input = segment_ay_reference - segment_yaw_rate * segment_u
        derivative_input_3d = (
            derivative_input + segment_roll_rate * segment_w)

        # Fine-grid propagation is a consistency control. Only its initial
        # truth value seeds the integrator; no later truth is fed back.
        fine_v = np.empty_like(segment_v)
        fine_v[0] = segment_v[0]
        for local_index in range(1, len(segment_v)):
            fine_v[local_index] = fine_v[local_index - 1] + 0.5 * (
                derivative_input[local_index - 1]
                + derivative_input[local_index]) * segment_dt[local_index - 1]
        fine_errors.append(fine_v - segment_v)
        fine_v_3d = np.empty_like(segment_v)
        fine_v_3d[0] = segment_v[0]
        for local_index in range(1, len(segment_v)):
            fine_v_3d[local_index] = fine_v_3d[local_index - 1] + 0.5 * (
                derivative_input_3d[local_index - 1]
                + derivative_input_3d[local_index]) * segment_dt[local_index - 1]
        fine_errors_3d.append(fine_v_3d - segment_v)

        for phase in range(grid_steps):
            local_indices = np.arange(phase, end - start, grid_steps,
                                      dtype=np.int64)
            if len(local_indices) < 2:
                continue
            sample_times = time_s[start + local_indices]
            steps = np.diff(sample_times)
            sampled_yaw_rate = segment_yaw_rate[local_indices]
            sampled_u = segment_u[local_indices]
            sampled_ay = segment_ay_com[local_indices]
            sampled_alpha = np.empty(len(local_indices), dtype=np.float64)
            sampled_alpha[1:] = np.diff(sampled_yaw_rate) / steps
            # The initial reference value is only used by a left-hold
            # diagnostic. Seed it with the first interval's causal estimate;
            # right-hold/trapezoid results use the measured interval endpoint.
            sampled_alpha[0] = sampled_alpha[1]
            sampled_h = (sampled_ay
                         - sampled_alpha * args.reference_forward_offset_m
                         - sampled_yaw_rate * sampled_u)
            errors_by_method: dict[str, dict[str, float | int]] = {}
            for method in ("left_hold", "right_hold", "trapezoid"):
                estimate = np.empty(len(local_indices), dtype=np.float64)
                estimate[0] = segment_v[local_indices[0]]
                for j, h in enumerate(steps, start=1):
                    if method == "left_hold":
                        rate = sampled_h[j - 1]
                    elif method == "right_hold":
                        rate = sampled_h[j]
                    else:
                        rate = 0.5 * (sampled_h[j - 1] + sampled_h[j])
                    estimate[j] = estimate[j - 1] + h * rate
                errors_by_method[method] = describe(
                    estimate - segment_v[local_indices])
            estimate_full_3d = np.empty(len(local_indices), dtype=np.float64)
            estimate_full_3d[0] = segment_v[local_indices[0]]
            sampled_h_full_3d = derivative_input_3d[local_indices]
            for j, h in enumerate(steps, start=1):
                rate = 0.5 * (sampled_h_full_3d[j - 1]
                              + sampled_h_full_3d[j])
                estimate_full_3d[j] = estimate_full_3d[j - 1] + h * rate
            errors_by_method["trapezoid_full_3d"] = describe(
                estimate_full_3d - segment_v[local_indices])
            averaged_imu = np.empty(len(local_indices), dtype=np.float64)
            averaged_imu_full_3d = np.empty(len(local_indices), dtype=np.float64)
            exact_interval = np.empty(len(local_indices), dtype=np.float64)
            exact_interval_full_3d = np.empty(len(local_indices), dtype=np.float64)
            averaged_imu[0] = segment_v[local_indices[0]]
            averaged_imu_full_3d[0] = segment_v[local_indices[0]]
            exact_interval[0] = segment_v[local_indices[0]]
            exact_interval_full_3d[0] = segment_v[local_indices[0]]
            for j, (left, right) in enumerate(
                    zip(local_indices[:-1], local_indices[1:]), start=1):
                interval_dt = sample_times[j] - sample_times[j - 1]
                # The Unity IMU computes an instantaneous 1 kHz derivative.
                # Averaging those source samples over the completed 25 ms
                # interval is the causal anti-alias alternative being tested.
                average_ay = float(np.mean(segment_ay_com[left + 1:right + 1]))
                average_ay_full_3d = float(np.mean(
                    segment_ay_com[left + 1:right + 1]
                    + segment_roll_rate[left + 1:right + 1]
                    * segment_w[left + 1:right + 1]))
                average_ru_from_endpoints = 0.5 * (
                    segment_yaw_rate[left] * segment_u[left]
                    + segment_yaw_rate[right] * segment_u[right])
                averaged_imu[j] = (
                    averaged_imu[j - 1]
                    + interval_dt * (average_ay - average_ru_from_endpoints)
                    - args.reference_forward_offset_m
                    * (segment_yaw_rate[right] - segment_yaw_rate[left]))
                averaged_imu_full_3d[j] = (
                    averaged_imu_full_3d[j - 1]
                    + interval_dt * (average_ay_full_3d
                                     - average_ru_from_endpoints)
                    - args.reference_forward_offset_m
                    * (segment_yaw_rate[right] - segment_yaw_rate[left]))
                # This is an oracle lower bound: integrate the actual 1 kHz
                # body-equation derivative over the interval, not a value
                # available in the 40 Hz observer.
                interval_h = float(np.mean(
                    derivative_input[left + 1:right + 1]))
                interval_h_full_3d = float(np.mean(
                    derivative_input_3d[left + 1:right + 1]))
                exact_interval[j] = exact_interval[j - 1] + interval_dt * interval_h
                exact_interval_full_3d[j] = (
                    exact_interval_full_3d[j - 1]
                    + interval_dt * interval_h_full_3d)
            errors_by_method["interval_averaged_imu_with_40hz_u_r"] = describe(
                averaged_imu - segment_v[local_indices])
            errors_by_method["interval_averaged_imu_full_3d_with_40hz_u_r"] = describe(
                averaged_imu_full_3d - segment_v[local_indices])
            errors_by_method["1khz_integrated_equation_oracle"] = describe(
                exact_interval - segment_v[local_indices])
            errors_by_method["1khz_integrated_full_3d_oracle"] = describe(
                exact_interval_full_3d - segment_v[local_indices])
            # Existing production kinematic base, evaluated as a sensor-only
            # algebraic comparator with true u/r so the lateral model alone is
            # isolated. This is not the full observer (which has other gates
            # and high-yaw experts).
            gain = (args.yaw_rate_gain_m
                    + args.speed_yaw_rate_gain_s
                    * np.maximum(0.0, segment_u[local_indices]))
            velocity_at_reference = np.clip(
                sampled_yaw_rate * gain,
                -args.lateral_velocity_limit_mps,
                args.lateral_velocity_limit_mps)
            kinematic_rear_v = (
                velocity_at_reference
                - sampled_yaw_rate * args.reference_forward_offset_m)
            errors_by_method["production_kinematic_base"] = sample_distribution(
                kinematic_rear_v - segment_v[local_indices])
            phases.append({
                "segment": segment_id,
                "phase_offset_ms": float(phase * fixed_dt * 1000.0),
                "sample_count": int(len(local_indices)),
                "duration_s": float(sample_times[-1] - sample_times[0]),
                "methods": errors_by_method,
            })

    methods = ("left_hold", "right_hold", "trapezoid", "trapezoid_full_3d",
               "interval_averaged_imu_with_40hz_u_r",
               "interval_averaged_imu_full_3d_with_40hz_u_r",
               "1khz_integrated_equation_oracle",
               "1khz_integrated_full_3d_oracle",
               "production_kinematic_base")
    aggregate: dict[str, dict[str, float]] = {}
    for method in methods:
        method_results = [phase["methods"][method] for phase in phases]
        rmses = np.asarray([item["rmse"] for item in method_results],
                           dtype=np.float64)
        p95s = np.asarray([item["p95_abs"] for item in method_results],
                          dtype=np.float64)
        summary = {
            "median_phase_rmse_mps": float(np.median(rmses)),
            "worst_phase_rmse_mps": float(np.max(rmses)),
            "median_phase_p95_abs_mps": float(np.median(p95s)),
            "worst_phase_p95_abs_mps": float(np.max(p95s)),
        }
        endpoints = [item["endpoint_signed_error"] for item in method_results
                     if "endpoint_signed_error" in item]
        if endpoints:
            absolute_endpoints = np.abs(np.asarray(endpoints, dtype=np.float64))
            summary["median_abs_endpoint_error_mps"] = float(
                np.median(absolute_endpoints))
            summary["worst_abs_endpoint_error_mps"] = float(
                np.max(absolute_endpoints))
        aggregate[method] = summary

    result = {
        "capture": str(args.capture.resolve()),
        "unity_version": metadata.get("unity_version"),
        "fixed_dt_s_median": fixed_dt,
        "fixed_dt_s_declared": nominal_dt,
        "samples": int(len(data)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "forward_speed_range_mps": [float(np.min(u)), float(np.max(u))],
        "max_abs_lateral_speed_com_mps": float(np.max(np.abs(v))),
        "yaw_rate_range_radps": [float(np.min(yaw_rate)),
                                  float(np.max(yaw_rate))],
        "max_abs_front_wheel_steering_rad": float(
            np.max(np.abs(front_steering_rad))),
        "roll_rate_vertical_speed_lateral_term_mps2": sample_distribution(
            np.concatenate([roll_rate[start:end] * vertical_speed[start:end]
                            for start, end in segments])),
        "reset_or_pose_discontinuity_boundaries": int(np.count_nonzero(boundary)),
        "continuous_segments": [
            {"segment": index, "samples": end - start,
             "start_time_s": float(time_s[start]),
             "duration_s": float(time_s[end - 1] - time_s[start])}
            for index, (start, end) in enumerate(segments)],
        "sample_grid_ms": args.grid_ms,
        "reference_forward_offset_m": args.reference_forward_offset_m,
        "production_kinematic_base_parameters": {
            "yaw_rate_gain_m": args.yaw_rate_gain_m,
            "speed_yaw_rate_gain_s": args.speed_yaw_rate_gain_s,
            "lateral_velocity_limit_mps": args.lateral_velocity_limit_mps,
            "source": "f1tenth_localization/config/sensor_odometry.yaml",
        },
        "production_kinematic_base_regime_errors": {
            "overall": sample_distribution(kinematic_error),
            "by_joint_speed_and_physical_front_steering": regime_table,
            "speed_bin_edges_mps": list(speed_edges),
            "steering_bin_edges_rad": list(steering_edges),
            "evaluation_input_note": (
                "Uses captured truth forward speed and yaw rate to isolate the "
                "lateral formula; this is not an end-to-end observer score."),
        },
        "sample_phases_tested": len(phases),
        "equation": (
            "dv_rear/dt = ay_com - yaw_accel*COM_to_rear_offset "
            "- yaw_rate*forward_speed; v_rear=v_com-yaw_rate*offset"),
        "full_3d_equation_extension": (
            "add roll_rate_body*vertical_velocity_body to lateral "
            "acceleration before rear-reference conversion"),
        "imu_reconstruction": (
            "Unity IMU.cs velocity finite difference in world (world velocity "
            "reconstructed from captured Rigidbody-local twist and orientation), "
            "rotated into the current Rigidbody frame; no sensor noise was added"),
        "initialization": "captured lateral velocity at each phase start; diagnostic only",
        "anti_alias_test": (
            "interval_averaged_imu_with_40hz_u_r averages the documented 1 kHz "
            "IMU acceleration over each completed 25 ms interval, uses endpoint "
            "yaw/speed for the r*u term, and exact delta-yaw reference conversion. "
            "The 1khz oracle is not deployable and is only a lower bound."),
        "fine_grid_trapezoid_check": describe(np.concatenate(fine_errors)),
        "fine_grid_full_3d_trapezoid_check": describe(
            np.concatenate(fine_errors_3d)),
        "sampled_grid_aggregate": aggregate,
        "phase_results": phases,
        "interpretation_limit": (
            "This isolates sampling/phase error using a deterministic 1 kHz "
            "capture. It does not reproduce the competition transport or "
            "validate a deployable odometry update."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"phase_results"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
