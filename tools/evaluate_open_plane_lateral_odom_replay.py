#!/usr/bin/env python3
"""Replay lateral odometry models on the real 40 Hz transition-sweep bag.

The current observer estimates lateral velocity kinematically from yaw rate;
its optional alternative integrates the measured IMU lateral acceleration in
turns. This development-only comparison scores the untouched third repetition.
The dynamic candidate uses true simulator forward speed as an optimistic input
to isolate the lateral model. It is not a competition-runtime implementation.

Acceptance criteria were preregistered in
docs/development/ENGINEERING_STATE.md; a pass permits a later exact
encoder/observer replay, not immediate deployment.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import sqlite3
import statistics
from pathlib import Path

try:
    import evaluate_open_plane_transient_sweep as sweep
except ImportError as exc:  # pragma: no cover - ROS 2 runtime dependency
    raise SystemExit(f"ROS 2 analysis modules are required: {exc}") from exc


HOLDOUT_REPETITION = 3
COM_X_M = 0.15532
IMU_ACCELERATION_REFERENCE_X_M = 0.15532
LATERAL_VELOCITY_YAW_GAIN_M = 0.167
LATERAL_VELOCITY_SPEED_YAW_GAIN_S = -0.0063
LATERAL_VELOCITY_LIMIT_MPS = 0.35
TURN_ENTER_YAW_RATE_RADPS = 0.6
TURN_ENTER_LATERAL_ACCEL_MPS2 = 6.0
TURN_EXIT_YAW_RATE_RADPS = 0.1
TURN_EXIT_LATERAL_ACCEL_MPS2 = 0.5
TURN_EXIT_HOLD_S = 0.5
PHASE_RE = sweep.PHASE_RE


def _rmse(errors: list[float]) -> float:
    if not errors:
        raise ValueError("no holdout samples were scored")
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def _rear_slip(vx: float, vy: float, yaw_rate: float) -> float:
    half_track = sweep.common.TRACK_WIDTH_M * 0.5
    left = sweep.common._wheel_slip(vx, vy, yaw_rate, 0.0, half_track, 0.0)
    right = sweep.common._wheel_slip(vx, vy, yaw_rate, 0.0, -half_track, 0.0)
    if left is None or right is None:
        raise ValueError("rear slip undefined for recorded forward speed")
    return 0.5 * (abs(left) + abs(right))


def _load_motion(path: Path, phases):
    database = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = sweep.common._topic_map(database)
        missing = [name for name in (sweep.ODOM, sweep.common.IMU)
                   if name not in topics]
        if missing:
            raise ValueError("missing motion topic(s): " + ", ".join(missing))
        odom_id, odom_type = topics[sweep.ODOM]
        imu_id, imu_type = topics[sweep.common.IMU]
        odometry = list(sweep._decode(database, odom_id, odom_type))
        imu = list(sweep._decode(database, imu_id, imu_type))
    finally:
        database.close()

    imu_times = [timestamp for timestamp, _ in imu]
    rows = []
    for timestamp, message in odometry:
        index = bisect.bisect_left(imu_times, timestamp)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(imu)]
        if not candidates:
            continue
        selected = min(candidates, key=lambda i: abs(imu_times[i] - timestamp))
        if abs(imu_times[selected] - timestamp) > 20_000_000:
            continue
        imu_message = imu[selected][1]
        twist = message.twist.twist
        vx_com = float(twist.linear.x)
        vy_com = float(twist.linear.y)
        yaw_rate = float(imu_message.angular_velocity.z)
        vx, vy = sweep.common._rear_axle_velocity(vx_com, vy_com, yaw_rate)
        values = (
            vx, vy,
            yaw_rate,
            float(imu_message.linear_acceleration.x),
            float(imu_message.linear_acceleration.y),
        )
        if all(math.isfinite(value) for value in values):
            rows.append((timestamp, *values))
    rows.sort(key=lambda row: row[0])

    holdout_windows = [phase for phase in phases
                       if phase.valid is True
                       and (match := PHASE_RE.match(phase.label))
                       and int(match.group("rep")) == HOLDOUT_REPETITION]
    if not holdout_windows:
        raise ValueError("no valid repetition-3 transition phases")
    return rows, holdout_windows


def evaluate(path: Path) -> bool:
    _, quality, phases = sweep.load(path)
    expected_per_rep = 2 * (2 * len(sweep.TRANSITION_LEVELS) - 1)
    rates_ok = all(
        metric["hz"] is not None and metric["hz"] >= 38.0
        and metric["p95_gap_ms"] is not None and metric["p95_gap_ms"] <= 35.0
        and metric["max_gap_ms"] is not None and metric["max_gap_ms"] <= 60.0
        for metric in quality["rates"].values())
    if (quality["valid_steering_phases"] != 3 * expected_per_rep
            or quality["matched_sweep_starts"] != 6
            or quality["collision_initial"] != 0
            or quality["collision_final"] != 0
            or quality["bridge_timing_faults"] != 0
            or quality["aborted"] or not rates_ok):
        raise ValueError(f"capture failed its predeclared quality gate: {quality}")

    rows, holdout_windows = _load_motion(path, phases)
    dynamic_vy = 0.0
    turn_mode = False
    turn_calm_s = 0.0
    previous_stamp_ns = None
    previous_yaw_rate = None
    previous_u = 0.0
    scored = []
    matched_imu = 0

    for timestamp, vx, truth_vy, yaw_rate, imu_ax, imu_ay in rows:
        matched_imu += 1
        if previous_stamp_ns is None:
            previous_stamp_ns = timestamp
            previous_yaw_rate = yaw_rate
            previous_u = max(0.0, vx)
            continue
        dt = (timestamp - previous_stamp_ns) / 1.0e9
        if dt <= 0.0 or dt > 0.075:
            raise ValueError(f"invalid synchronized odom/IMU interval {dt:.4f}s")

        if (not turn_mode and
                (abs(yaw_rate) >= TURN_ENTER_YAW_RATE_RADPS
                 or abs(imu_ay) >= TURN_ENTER_LATERAL_ACCEL_MPS2)):
            turn_mode = True
            turn_calm_s = 0.0
            dynamic_vy = 0.0

        yaw_alpha = (yaw_rate - previous_yaw_rate) / dt
        ax_origin = imu_ax + yaw_rate * yaw_rate * IMU_ACCELERATION_REFERENCE_X_M
        ay_origin = imu_ay - yaw_alpha * IMU_ACCELERATION_REFERENCE_X_M
        if turn_mode:
            du = ax_origin + yaw_rate * dynamic_vy
            u_mid = previous_u + 0.5 * dt * du
            dynamic_vy += dt * (ay_origin - yaw_rate * u_mid)
            dynamic_vy = max(-30.0, min(30.0, dynamic_vy))
            calm = (abs(yaw_rate) < TURN_EXIT_YAW_RATE_RADPS
                    and abs(imu_ay) < TURN_EXIT_LATERAL_ACCEL_MPS2)
            turn_calm_s = turn_calm_s + dt if calm else 0.0
            if turn_calm_s >= TURN_EXIT_HOLD_S:
                dynamic_vy = 0.0
                turn_mode = False
                turn_calm_s = 0.0
        else:
            dynamic_vy = 0.0

        current_formula_vcom = max(-LATERAL_VELOCITY_LIMIT_MPS, min(
            LATERAL_VELOCITY_LIMIT_MPS,
            yaw_rate * (LATERAL_VELOCITY_YAW_GAIN_M
                        + LATERAL_VELOCITY_SPEED_YAW_GAIN_S * max(vx, 0.0))))
        current_formula_vy = current_formula_vcom - yaw_rate * COM_X_M
        if any(phase.start_ns <= timestamp <= phase.end_ns
               for phase in holdout_windows):
            scored.append((vx, truth_vy, yaw_rate, current_formula_vy, dynamic_vy))

        previous_stamp_ns = timestamp
        previous_yaw_rate = yaw_rate
        previous_u = max(0.0, vx)

    if not scored:
        raise ValueError("holdout phase windows contain no synchronized motion samples")

    baseline_errors = [row[3] - row[1] for row in scored]
    dynamic_errors = [row[4] - row[1] for row in scored]
    baseline_slip_errors = [
        _rear_slip(row[0], row[3], row[2]) - _rear_slip(row[0], row[1], row[2])
        for row in scored]
    dynamic_slip_errors = [
        _rear_slip(row[0], row[4], row[2]) - _rear_slip(row[0], row[1], row[2])
        for row in scored]
    baseline_rmse = _rmse(baseline_errors)
    dynamic_rmse = _rmse(dynamic_errors)
    dynamic_bias = statistics.mean(dynamic_errors)
    baseline_slip_rmse = _rmse(baseline_slip_errors)
    dynamic_slip_rmse = _rmse(dynamic_slip_errors)
    passed = (
        dynamic_rmse <= 0.08
        and dynamic_rmse <= 0.50 * baseline_rmse
        and abs(dynamic_bias) <= 0.05
        and dynamic_slip_rmse <= 0.02)

    print(f"bag: {path}")
    print("capture PASS: 126/126 phases; 6/6 matched starts; "
          "zero collisions/timing faults; 38+ Hz streams")
    print(f"synchronized odom/IMU samples={matched_imu}; holdout samples={len(scored)}")
    print("holdout lateral state RMSE (m/s): "
          f"current kinematic={baseline_rmse:.5f}, "
          f"IMU-integrated optimistic={dynamic_rmse:.5f} "
          f"(gain={1-dynamic_rmse/baseline_rmse:+.1%})")
    print(f"IMU-integrated v_y bias={dynamic_bias:+.5f} m/s "
          f"(gate |bias|<=0.05); rear |Sy| RMSE current/integrated="
          f"{baseline_slip_rmse:.5f}/{dynamic_slip_rmse:.5f} "
          "(gate integrated<=0.02)")
    print(f"decision: {'PASS; exact encoder/observer replay is justified' if passed else 'REJECT the current IMU-integration branch for this response'}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
