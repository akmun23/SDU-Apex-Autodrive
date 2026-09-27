#!/usr/bin/env python3
"""Held-out validation of the 4.5--7.5 m/s nonlinear yaw response surface.

Fit steering PCHIPs from repetitions 1--2 at 4.5 and 7.5 m/s; interpolate
signed lateral acceleration to the completely held-out 6.5 m/s repetition 3.
For an angle holdout, remove that steering knot at both source speeds. This is
an offline identification test; ground truth is never a model input.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as common
from tools import evaluate_open_plane_rear_slip_effect as rear_slip
from tools import evaluate_open_plane_yaw_spline as yaw_spline


SOURCE_SPEEDS_MPS = (4.5, 7.5)
HOLDOUT_SPEED_MPS = 6.5
STEERING_LEVELS_RAD = (0.15, 0.20, 0.21, 0.22, 0.23,
                       0.25, 0.30, 0.35, 0.42, 0.50)
INTERIOR_HOLDOUT_ANGLES_RAD = (0.20, 0.21, 0.22, 0.23,
                               0.25, 0.30, 0.35, 0.42)
TRAIN_REPETITIONS = (1, 2)
HOLDOUT_REPETITION = 3
MAX_MIDDLE_RMSE_PER_M = 0.08
MAX_MIDDLE_ABS_ERROR_PER_M = 0.16
MIN_CONFIGURED_REDUCTION = 0.50
MAX_SOURCE_RMSE_PER_M = 0.05
MIN_YAW_SENSITIVITY_PER_S = 0.05


def _angle_curves(blocks, source_speed: float, sign: int,
                  heldout_angle: float | None):
    grouped: dict[float, list[float]] = defaultdict(list)
    for block in blocks:
        if (abs(block.target_speed_mps - source_speed) > 1e-6
                or (1 if block.steering_rad > 0.0 else -1) != sign
                or block.repetition not in TRAIN_REPETITIONS):
            continue
        angle = round(abs(block.steering_rad), 2)
        if heldout_angle is not None and abs(angle - heldout_angle) < 1e-6:
            continue
        grouped[angle].append(block.yaw_gain_per_m)
    angles = np.asarray(sorted(grouped), dtype=float)
    gains = np.asarray([float(np.mean(grouped[angle])) for angle in angles])
    expected = len(STEERING_LEVELS_RAD) - int(heldout_angle is not None)
    if len(angles) != expected:
        raise ValueError(f"source {source_speed:.1f}m/s sign={sign:+d} angle="
                         f"{heldout_angle}: expected {expected} knots, "
                         f"found {angles.tolist()}")
    return angles, gains


def _predict(speed: float, steering: float,
             curves: dict[float, tuple[np.ndarray, np.ndarray]],
             source_speeds: tuple[float, ...]) -> tuple[float, float]:
    """Return yaw gain and d(yaw-rate)/d(steering), blending lateral accel."""
    sign = 1 if steering > 0.0 else -1
    source: list[tuple[float, float]] = []
    for source_speed in source_speeds:
        angles, gains = curves[source_speed]
        gain, d_gain_d_abs = yaw_spline._pchip_value_and_slope(
            angles, gains, abs(steering))
        d_gain_d_steer = d_gain_d_abs * sign
        tangent = math.tan(steering)
        ay = source_speed**2 * tangent * gain
        day_d_steer = source_speed**2 * (
            gain / math.cos(steering)**2 + tangent * d_gain_d_steer)
        source.append((ay, day_d_steer))

    if len(source_speeds) == 1:
        ay, day_d_steer = source[0]
    else:
        low_speed, high_speed = source_speeds
        if not low_speed <= speed <= high_speed:
            raise ValueError("speed interpolation requested outside measured knots")
        high_weight = (speed - low_speed) / (high_speed - low_speed)
        low_weight = 1.0 - high_weight
        ay = low_weight * source[0][0] + high_weight * source[1][0]
        day_d_steer = (low_weight * source[0][1]
                       + high_weight * source[1][1])
    gain = ay / (speed**2 * math.tan(steering))
    yaw_sensitivity = day_d_steer / speed
    return float(gain), float(yaw_sensitivity)


def _capture_gate(path: Path) -> tuple[int, dict[str, object]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        required = (common.ODOM, common.STEERING, common.LEFT_ENCODER,
                    common.RIGHT_ENCODER, common.IMU, common.COLLISIONS,
                    common.TIMING_FAULT, common.PACKET_TIMING, common.PHASE)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("bag is missing required topic(s): " + ", ".join(missing))
        phases, end = common._phase_events(connection, topics)
        collisions = [int(msg.data) for _, msg in common._messages(
            connection, topics, common.COLLISIONS)]
        faults = [bool(msg.data) for _, msg in common._messages(
            connection, topics, common.TIMING_FAULT)]
        phase_probes = [phase for phase in phases
                        if phase.label.startswith("isolated_r")]
        if len(phase_probes) != 180 or any(phase.valid is not True
                                          for phase in phase_probes):
            raise ValueError(f"expected 180 valid matched probes; found "
                             f"{len(phase_probes)}")
        if sorted({int(phase.label.split("_", 2)[1][1:])
                   for phase in phase_probes}) != [1, 2, 3]:
            raise ValueError("expected repetitions 1, 2, and 3")
        for phase in phase_probes:
            start = (phase.initial_window_speed_mps,
                     phase.initial_window_abs_vy_mps,
                     phase.initial_window_abs_yaw_rate_rps,
                     phase.initial_steering_rad)
            if any(value is None for value in start):
                raise ValueError(f"missing matched-start telemetry: {phase.label}")
            if (abs(start[0] - phase.target_speed_mps) > 0.20
                    or start[1] > 0.08 or start[2] > 0.12 or abs(start[3]) > 0.02):
                raise ValueError(f"unmatched start: {phase.label}: {start}")
        if (end.get("aborted") is not False or end.get("quality_failures")
                or len(phases) != int(end.get("phase_count", -1))
                or not collisions or max(collisions) != 0 or any(faults)):
            raise ValueError(f"capture/vehicle safety gate failed: {end}")

        rates = {}
        for name in (common.ODOM, common.STEERING, common.LEFT_ENCODER,
                     common.RIGHT_ENCODER, common.IMU, common.PACKET_TIMING):
            topic_id = topics[name][0]
            receipts = [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp, id",
                (topic_id,),
            )]
            hz, p95_ms, max_ms = common._rate(receipts)
            rates[name] = (hz, p95_ms, max_ms)
            if (hz is None or hz < 38.0 or p95_ms is None or p95_ms > 35.0
                    or max_ms is None or max_ms > 60.0):
                raise ValueError(f"stream-quality gate failed for {name}: {rates[name]}")
    finally:
        connection.close()
    return len(phase_probes), rates


def evaluate(path: Path) -> bool:
    probe_count, rates = _capture_gate(path)
    blocks = rear_slip._load_blocks(path)
    expected_speeds = set((*SOURCE_SPEEDS_MPS, HOLDOUT_SPEED_MPS))
    actual_speeds = {round(block.target_speed_mps, 1) for block in blocks}
    if actual_speeds != expected_speeds or len(blocks) != 180:
        raise ValueError(f"unexpected speed/condition coverage: {actual_speeds}, "
                         f"{len(blocks)} blocks")

    middle_actual: list[float] = []
    middle_prediction: list[float] = []
    middle_baseline: list[float] = []
    source_errors: dict[float, list[float]] = {speed: [] for speed in SOURCE_SPEEDS_MPS}
    for block in blocks:
        if block.repetition != HOLDOUT_REPETITION:
            continue
        if block.measured_forward_speed_mps is None:
            raise ValueError(f"missing measured forward speed for {block.phase}")
        speed = round(block.target_speed_mps, 1)
        angle = round(abs(block.steering_rad), 2)
        if angle not in INTERIOR_HOLDOUT_ANGLES_RAD:
            continue
        sign = 1 if block.steering_rad > 0.0 else -1
        curves = {
            source_speed: _angle_curves(blocks, source_speed, sign, angle)
            for source_speed in SOURCE_SPEEDS_MPS
        }
        prediction, _ = _predict(block.measured_forward_speed_mps,
                                 block.steering_rad, curves, SOURCE_SPEEDS_MPS)
        if speed == HOLDOUT_SPEED_MPS:
            middle_actual.append(block.yaw_gain_per_m)
            middle_prediction.append(prediction)
            middle_baseline.append(yaw_spline._configured_gain(block.steering_rad))
        else:
            single_curve = {speed: curves[speed]}
            source_prediction, _ = _predict(
                block.measured_forward_speed_mps, block.steering_rad,
                single_curve, (speed,))
            source_errors[speed].append(source_prediction - block.yaw_gain_per_m)

    if len(middle_actual) != 16 or any(len(errors) != 16
                                       for errors in source_errors.values()):
        raise ValueError(f"incomplete held-out scoring: middle={len(middle_actual)}, "
                         f"source={ {k: len(v) for k, v in source_errors.items()} }")
    actual = np.asarray(middle_actual)
    predicted = np.asarray(middle_prediction)
    baseline = np.asarray(middle_baseline)
    middle_rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
    middle_max = float(np.max(np.abs(predicted - actual)))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline - actual))))
    reduction = 1.0 - middle_rmse / baseline_rmse if baseline_rmse > 0.0 else 0.0
    source_rmse = {
        speed: float(np.sqrt(np.mean(np.square(errors))))
        for speed, errors in source_errors.items()
    }

    sensitivity_values = []
    for sign in (-1, 1):
        curves = {
            source_speed: _angle_curves(blocks, source_speed, sign, None)
            for source_speed in SOURCE_SPEEDS_MPS
        }
        for angle in np.linspace(STEERING_LEVELS_RAD[0], STEERING_LEVELS_RAD[-1], 141):
            _, sensitivity = _predict(
                HOLDOUT_SPEED_MPS, sign * float(angle), curves, SOURCE_SPEEDS_MPS)
            sensitivity_values.append(sensitivity)
    sensitivity_ok = all(math.isfinite(value) and value > MIN_YAW_SENSITIVITY_PER_S
                         for value in sensitivity_values)

    print(f"bag: {path}")
    print(f"capture PASS: {probe_count}/180 matched probes; 3 speeds; repetitions "
          "1--2 train, repetition 3 holdout; zero collisions/timing faults")
    for name, (hz, p95, max_gap) in rates.items():
        print(f"  {name}: {hz:.3f} Hz; p95/max gap={p95:.2f}/{max_gap:.2f} ms")
    print("model: source-speed PCHIP in steering, signed lateral-acceleration "
          "interpolation in speed; internal steering knot withheld")
    print(f"6.5 m/s repetition-3 predictions={len(actual)}: yaw-gain RMSE="
          f"{middle_rmse:.5f} 1/m; max error={middle_max:.5f} 1/m; "
          f"configured RMSE={baseline_rmse:.5f} 1/m; reduction={reduction:+.1%}")
    for speed in SOURCE_SPEEDS_MPS:
        print(f"{speed:.1f} m/s repetition-3 angle-holdout RMSE="
              f"{source_rmse[speed]:.5f} 1/m")
    print(f"6.5 m/s d(yaw rate)/d(steer) range="
          f"{min(sensitivity_values):.4f}..{max(sensitivity_values):.4f} 1/s; "
          f"all finite and positive={sensitivity_ok}")
    passed = (middle_rmse <= MAX_MIDDLE_RMSE_PER_M
              and middle_max <= MAX_MIDDLE_ABS_ERROR_PER_M
              and reduction >= MIN_CONFIGURED_REDUCTION
              and all(value <= MAX_SOURCE_RMSE_PER_M
                      for value in source_rmse.values())
              and sensitivity_ok)
    print("decision: " + ("ACCEPT steady high-speed map for later dynamic-rollout testing"
                          if passed else
                          "REJECT this speed-steering interpolation; retain measured tables"))
    return passed


def _self_test() -> None:
    angles = np.asarray(STEERING_LEVELS_RAD)
    gains = 1.1 - 0.7 * angles
    curves = {speed: (angles, gains.copy()) for speed in SOURCE_SPEEDS_MPS}
    query_speed = 6.51
    gain, sensitivity = _predict(query_speed, 0.30, curves, SOURCE_SPEEDS_MPS)
    expected = (((query_speed - 4.5) / (7.5 - 4.5) * 7.5**2
                 + (7.5 - query_speed) / (7.5 - 4.5) * 4.5**2)
                / query_speed**2) * (1.1 - 0.7 * 0.30)
    assert math.isclose(gain, expected, rel_tol=1e-12)
    assert sensitivity > MIN_YAW_SENSITIVITY_PER_S
    for sign in (-1, 1):
        signed_gain, signed_sensitivity = _predict(
            query_speed, sign * 0.30, curves, SOURCE_SPEEDS_MPS)
        assert math.isclose(signed_gain, gain, rel_tol=1e-12)
        assert math.isclose(signed_sensitivity, sensitivity, rel_tol=1e-12)
    print("high-speed response math: PASS (lateral-acceleration blend, signed Jacobian)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.bag is None:
        parser.error("provide a high-speed-surface bag or use --self-test")
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
