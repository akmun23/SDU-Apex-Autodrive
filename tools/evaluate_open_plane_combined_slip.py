#!/usr/bin/env python3
"""Held-out test of rear longitudinal slip during matched high-angle probes.

Compares an effective yaw-gain model using steering condition and measured
speed against the same model augmented by rear-encoder longitudinal slip.
Repetition 3 is held out; this is an offline development diagnostic only.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import analyze_open_plane_dynamics as analysis


PHASE_LABEL = re.compile(r"^drive_r([123])_([+-]\d+\.\d+)_(low|high)$")
MIN_BLOCK_SAMPLES = 30
MIN_SLIP_SEPARATION = 0.01
MIN_RMSE_IMPROVEMENT = 0.10


@dataclass(frozen=True)
class Block:
    repetition: int
    steering_rad: float
    drive_level: str
    speed_mps: float
    yaw_gain_per_m: float
    rear_abs_longitudinal_slip: float
    lateral_velocity_mps: float
    throttle_feedback: float

    @property
    def condition(self) -> tuple[int, int]:
        return (round(abs(self.steering_rad) * 100),
                1 if self.steering_rad > 0.0 else -1)


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def _load_blocks(path: Path) -> list[Block]:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, analysis.STEERING, analysis.THROTTLE_FEEDBACK,
                    analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER,
                    analysis.PHASE, analysis.COLLISIONS)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError("bag is missing required topic(s): " + ", ".join(missing))

        odometry = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM):
            twist = message.twist.twist
            vx_com, vy_com, yaw_rate = (float(twist.linear.x),
                                        float(twist.linear.y),
                                        float(twist.angular.z))
            vx, vy = analysis._rear_axle_velocity(vx_com, vy_com, yaw_rate)
            values = (vx, vy, yaw_rate)
            if all(math.isfinite(value) for value in values):
                odometry.append((receipt_ns, *values))

        def scalar_rows(topic: str) -> list[analysis.ScalarRow]:
            return [analysis.ScalarRow(receipt, receipt, float(message.data))
                    for receipt, message in analysis._messages(connection, topics, topic)
                    if math.isfinite(float(message.data))]

        steering = scalar_rows(analysis.STEERING)
        throttle = scalar_rows(analysis.THROTTLE_FEEDBACK)
        steering_times = [row.receipt_ns for row in steering]
        throttle_times = [row.receipt_ns for row in throttle]

        def encoder_rows(topic: str) -> list[analysis.EncoderRow]:
            rows = []
            for receipt, message in analysis._messages(connection, topics, topic):
                if message.position and math.isfinite(float(message.position[0])):
                    rows.append(analysis.EncoderRow(
                        receipt, analysis._stamp_ns(message.header.stamp),
                        float(message.position[0])))
            return rows

        left = encoder_rows(analysis.LEFT_ENCODER)
        right = encoder_rows(analysis.RIGHT_ENCODER)
        left_times = [row.receipt_ns for row in left]
        right_times = [row.receipt_ns for row in right]
        collision_values = [int(message.data) for _, message in analysis._messages(
            connection, topics, analysis.COLLISIONS)]
        phases, end = analysis._phase_events(connection, topics)
    finally:
        connection.close()

    if end.get("aborted") is not False or end.get("quality_failures"):
        raise ValueError(f"capture did not complete cleanly: {end}")
    if collision_values and max(collision_values) != 0:
        raise ValueError("capture contains a collision-count increase")

    odom_times = [row[0] for row in odometry]
    half_track = analysis.TRACK_WIDTH_M / 2.0
    blocks: list[Block] = []
    for phase in phases:
        match = PHASE_LABEL.fullmatch(phase.label)
        if match is None:
            continue
        if phase.valid is not True or phase.quality_failures:
            raise ValueError(f"invalid probe phase {phase.label}: {phase.quality_failures}")
        repetition = int(match.group(1))
        drive_level = match.group(3)
        start_ns = phase.start_ns + analysis.PHASE_SETTLE_NS
        begin = int(np.searchsorted(odom_times, start_ns, side="left"))
        stop = int(np.searchsorted(odom_times, phase.end_ns, side="left"))
        speeds: list[float] = []
        yaw_gains: list[float] = []
        slips: list[float] = []
        lateral_velocities: list[float] = []
        throttle_values: list[float] = []
        for receipt_ns, vx, vy, yaw_rate in odometry[begin:stop]:
            actual_steering = analysis._nearest_scalar(
                steering, steering_times, receipt_ns)
            actual_throttle = analysis._nearest_scalar(
                throttle, throttle_times, receipt_ns)
            if actual_steering is None or actual_throttle is None or vx < 1.0:
                continue
            delta = actual_steering.value
            denominator = vx * math.tan(delta)
            if abs(denominator) < 0.1:
                continue
            left_slip = analysis._encoder_slip(
                left, left_times, receipt_ns, vx - yaw_rate * half_track, 1.0)
            right_slip = analysis._encoder_slip(
                right, right_times, receipt_ns, vx + yaw_rate * half_track, 1.0)
            if left_slip is None or right_slip is None:
                continue
            rear_slip = 0.5 * (abs(left_slip) + abs(right_slip))
            yaw_gain = yaw_rate / denominator
            if all(math.isfinite(value) for value in
                   (rear_slip, yaw_gain, vx, vy, actual_throttle.value)):
                speeds.append(vx)
                yaw_gains.append(yaw_gain)
                slips.append(rear_slip)
                lateral_velocities.append(abs(vy))
                throttle_values.append(actual_throttle.value)
        if len(yaw_gains) < MIN_BLOCK_SAMPLES:
            raise ValueError(f"too few aligned samples in {phase.label}: {len(yaw_gains)}")
        blocks.append(Block(
            repetition=repetition,
            steering_rad=phase.commanded_steering_rad,
            drive_level=drive_level,
            speed_mps=_median(speeds),
            yaw_gain_per_m=_median(yaw_gains),
            rear_abs_longitudinal_slip=_median(slips),
            lateral_velocity_mps=_median(lateral_velocities),
            throttle_feedback=_median(throttle_values),
        ))

    if len(blocks) != 24:
        raise ValueError(f"expected 24 valid matched probes, found {len(blocks)}")
    keys = {(b.repetition, b.condition, b.drive_level) for b in blocks}
    expected = {(rep, (angle, sign), drive)
                for rep in (1, 2, 3)
                for angle in (42, 50)
                for sign in (-1, 1)
                for drive in ("low", "high")}
    if keys != expected:
        raise ValueError("probe capture does not contain the complete repeated factorial")
    return blocks


def _design(blocks: list[Block], conditions: list[tuple[int, int]],
            slip_means: dict[tuple[int, int], float] | None = None) -> np.ndarray:
    columns = len(conditions) + 1 + int(slip_means is not None)
    matrix = np.zeros((len(blocks), columns), dtype=float)
    condition_index = {condition: index for index, condition in enumerate(conditions)}
    for row, block in enumerate(blocks):
        matrix[row, condition_index[block.condition]] = 1.0
        matrix[row, len(conditions)] = block.speed_mps - 5.0
        if slip_means is not None:
            matrix[row, len(conditions) + 1] = (
                block.rear_abs_longitudinal_slip - slip_means[block.condition])
    return matrix


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def evaluate(path: Path) -> bool:
    blocks = _load_blocks(path)
    conditions = sorted({block.condition for block in blocks})
    training = [block for block in blocks if block.repetition in (1, 2)]
    heldout = [block for block in blocks if block.repetition == 3]
    train_slip_means = {
        condition: _median([block.rear_abs_longitudinal_slip for block in training
                            if block.condition == condition])
        for condition in conditions
    }
    train_y = np.asarray([block.yaw_gain_per_m for block in training])
    heldout_y = np.asarray([block.yaw_gain_per_m for block in heldout])
    base_fit, _, base_rank, _ = np.linalg.lstsq(
        _design(training, conditions), train_y, rcond=None)
    augmented_fit, _, augmented_rank, _ = np.linalg.lstsq(
        _design(training, conditions, train_slip_means), train_y, rcond=None)
    augmented_condition_number = float(np.linalg.cond(
        _design(training, conditions, train_slip_means)))
    base_prediction = _design(heldout, conditions) @ base_fit
    augmented_prediction = _design(heldout, conditions, train_slip_means) @ augmented_fit
    base_rmse = _rmse(heldout_y, base_prediction)
    augmented_rmse = _rmse(heldout_y, augmented_prediction)
    improvement = 1.0 - augmented_rmse / base_rmse if base_rmse > 1e-12 else 0.0
    slip_coefficient = float(augmented_fit[-1])

    by_pair: dict[tuple[int, tuple[int, int]], dict[str, Block]] = defaultdict(dict)
    for block in blocks:
        by_pair[(block.repetition, block.condition)][block.drive_level] = block
    heldout_pairs = [by_pair[(3, condition)] for condition in conditions]
    slip_separation = _median([
        abs(pair["high"].rear_abs_longitudinal_slip
            - pair["low"].rear_abs_longitudinal_slip)
        for pair in heldout_pairs
    ])
    aligned_pairs = 0
    for pair in heldout_pairs:
        low, high = pair["low"], pair["high"]
        observed_speed_corrected_delta = (
            high.yaw_gain_per_m - low.yaw_gain_per_m
            - float(augmented_fit[len(conditions)]) * (high.speed_mps - low.speed_mps)
        )
        predicted_slip_delta = slip_coefficient * (
            high.rear_abs_longitudinal_slip - low.rear_abs_longitudinal_slip)
        if observed_speed_corrected_delta * predicted_slip_delta > 0.0:
            aligned_pairs += 1

    print(f"{path}: matched factorial probes={len(blocks)}; train={len(training)}, "
          f"held out={len(heldout)}")
    print(f"  held-out yaw-gain RMSE: speed+steering={base_rmse:.5f} 1/m; "
          f"plus rear |Sx|={augmented_rmse:.5f} 1/m ({improvement * 100:+.1f}%)")
    print(f"  rear-slip coefficient={slip_coefficient:+.4f} (yaw-gain units per slip ratio); "
          f"median paired |Sx| separation={slip_separation:.4f}; "
          f"held-out effect direction={aligned_pairs}/4; "
          f"design condition={augmented_condition_number:.1f}")
    for condition, pair in zip(conditions, heldout_pairs):
        low, high = pair["low"], pair["high"]
        print(f"  steer={condition[1] * condition[0] / 100:+.2f}rad: "
              f"low/high |Sx|={low.rear_abs_longitudinal_slip:.4f}/"
              f"{high.rear_abs_longitudinal_slip:.4f}, "
              f"speed={low.speed_mps:.3f}/{high.speed_mps:.3f}m/s, "
              f"yaw gain={low.yaw_gain_per_m:.3f}/{high.yaw_gain_per_m:.3f} 1/m, "
              f"median |vy|={low.lateral_velocity_mps:.3f}/"
              f"{high.lateral_velocity_mps:.3f}m/s, "
              f"throttle feedback={low.throttle_feedback:.3f}/"
              f"{high.throttle_feedback:.3f}")

    accepted = (
        base_rank == len(conditions) + 1
        and augmented_rank == len(conditions) + 2
        and augmented_condition_number <= 1000.0
        and math.isfinite(slip_coefficient)
        and slip_separation >= MIN_SLIP_SEPARATION
        and improvement >= MIN_RMSE_IMPROVEMENT
        and aligned_pairs >= 3
    )
    print("  decision: " + ("ACCEPT rear-slip feature for further model validation"
                          if accepted else
                          "REJECT rear-slip feature for this high-speed model"))
    return accepted


def _self_test() -> None:
    blocks = []
    for steering_index, steering in enumerate((-0.50, -0.42, 0.42, 0.50)):
        low_slip = 0.008 + 0.003 * steering_index
        high_slip = low_slip + (0.03 + 0.01 * (steering_index % 3))
        low_speed = 4.94 + 0.01 * (steering_index % 2)
        high_speed = 5.02 + 0.015 * ((steering_index + 1) % 3)
        for repetition in (1, 2, 3):
            condition = (round(abs(steering) * 100), 1 if steering > 0 else -1)
            condition_offset = 0.15 * condition[0] / 100 + 0.03 * condition[1]
            for level, slip, speed in (("low", low_slip, low_speed),
                                       ("high", high_slip, high_speed)):
                yaw_gain = (0.8 + condition_offset + 0.5 * (speed - 5.0)
                            - 2.0 * (slip - 0.04) + 0.002 * repetition)
                blocks.append(Block(repetition, steering, level, speed, yaw_gain,
                                    slip, 0.0, 0.2))
    conditions = sorted({block.condition for block in blocks})
    training = [block for block in blocks if block.repetition < 3]
    heldout = [block for block in blocks if block.repetition == 3]
    slip_means = {condition: _median([block.rear_abs_longitudinal_slip
                                      for block in training
                                      if block.condition == condition])
                  for condition in conditions}
    train_y = np.asarray([block.yaw_gain_per_m for block in training])
    test_y = np.asarray([block.yaw_gain_per_m for block in heldout])
    base_fit, *_ = np.linalg.lstsq(_design(training, conditions), train_y, rcond=None)
    model_fit, *_ = np.linalg.lstsq(
        _design(training, conditions, slip_means), train_y, rcond=None)
    condition_number = np.linalg.cond(_design(training, conditions, slip_means))
    assert abs(float(model_fit[-1]) + 2.0) < 1e-10
    assert condition_number <= 1000.0
    assert _rmse(test_y, _design(heldout, conditions, slip_means) @ model_fit) < 0.004
    assert _rmse(test_y, _design(heldout, conditions) @ base_fit) > 0.01
    print("combined-slip model math: PASS (speed nuisance, condition holdout, slip effect)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.bags:
        parser.error("provide the combined_slip_5mps bag or use --self-test")
    accepted = [evaluate(path) for path in args.bags]
    return 0 if all(accepted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
