#!/usr/bin/env python3
"""Test whether rear-wheel longitudinal slip explains repeated yaw branches.

This offline diagnostic compares a condition-matched yaw-gain predictor
(speed, steering magnitude and turn direction) with the same predictor plus
the measured mean absolute rear-wheel longitudinal-slip proxy. Repetitions 1
and 2 fit the within-condition slip coefficient; repetition 3 is held out.
It does not identify tire forces or enter either driving runtime.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis


SETTLE_NS = analysis.PHASE_SETTLE_NS
MIN_BLOCK_SAMPLES = 30


@dataclass(frozen=True)
class Block:
    phase: str
    repetition: int
    target_speed_mps: float
    steering_rad: float
    yaw_gain_per_m: float
    rear_abs_longitudinal_slip: float
    measured_forward_speed_mps: float | None = None

    @property
    def condition(self) -> tuple[int, int, int]:
        return (round(self.target_speed_mps * 10), round(abs(self.steering_rad) * 100),
                1 if self.steering_rad > 0.0 else -1)


def _load_blocks(path: Path) -> list[Block]:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, analysis.STEERING, analysis.LEFT_ENCODER,
                    analysis.RIGHT_ENCODER, analysis.PHASE, analysis.COLLISIONS)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError("bag is missing required topic(s): " + ", ".join(missing))

        odometry = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM):
            twist = message.twist.twist
            odometry.append((receipt_ns, float(twist.linear.x),
                             float(twist.linear.y), float(twist.angular.z)))
        steering = [
            analysis.ScalarRow(receipt_ns, receipt_ns, float(message.data))
            for receipt_ns, message in analysis._messages(
                connection, topics, analysis.STEERING)
        ]
        steering_times = [row.receipt_ns for row in steering]

        def encoder_rows(topic: str) -> list[analysis.EncoderRow]:
            rows = []
            for receipt_ns, message in analysis._messages(connection, topics, topic):
                if message.position and math.isfinite(float(message.position[0])):
                    rows.append(analysis.EncoderRow(
                        receipt_ns, analysis._stamp_ns(message.header.stamp),
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
    blocks: list[Block] = []
    half_track = analysis.TRACK_WIDTH_M / 2.0
    for phase in phases:
        if not phase.label.startswith("isolated_r") or phase.valid is not True:
            continue
        repetition = int(phase.label.split("_", 2)[1][1:])
        start_ns = phase.start_ns + SETTLE_NS
        begin = int(np.searchsorted(odom_times, start_ns, side="left"))
        end_index = int(np.searchsorted(odom_times, phase.end_ns, side="left"))
        yaw_gains: list[float] = []
        rear_slips: list[float] = []
        measured_speeds: list[float] = []

        for receipt_ns, u_mps, v_mps, yaw_rate in odometry[begin:end_index]:
            actual_steering = analysis._nearest_scalar(
                steering, steering_times, receipt_ns)
            if actual_steering is None or abs(u_mps) < 0.5:
                continue
            delta = actual_steering.value
            denom = u_mps * math.tan(delta)
            if abs(denom) < 0.1:
                continue
            left_slip = analysis._encoder_slip(
                left, left_times, receipt_ns, u_mps - yaw_rate * half_track, 1.0)
            right_slip = analysis._encoder_slip(
                right, right_times, receipt_ns, u_mps + yaw_rate * half_track, 1.0)
            if left_slip is None or right_slip is None:
                continue
            yaw_gain = yaw_rate / denom
            if all(math.isfinite(value) for value in (yaw_gain, left_slip, right_slip)):
                yaw_gains.append(yaw_gain)
                rear_slips.append(0.5 * (abs(left_slip) + abs(right_slip)))
                measured_speeds.append(abs(u_mps))

        if len(yaw_gains) < MIN_BLOCK_SAMPLES:
            continue
        blocks.append(Block(
            phase=phase.label,
            repetition=repetition,
            target_speed_mps=phase.target_speed_mps,
            steering_rad=phase.commanded_steering_rad,
            yaw_gain_per_m=float(np.median(yaw_gains)),
            rear_abs_longitudinal_slip=float(np.median(rear_slips)),
            measured_forward_speed_mps=float(np.median(measured_speeds)),
        ))

    if not blocks:
        raise ValueError(f"no valid isolated steering blocks found in {path}")
    return blocks


def _fit_condition_means(blocks: list[Block]) -> tuple[dict, dict, float]:
    by_condition: dict[tuple[int, int, int], list[Block]] = defaultdict(list)
    for block in blocks:
        by_condition[block.condition].append(block)
    yaw_means = {key: float(np.mean([b.yaw_gain_per_m for b in rows]))
                 for key, rows in by_condition.items()}
    slip_means = {key: float(np.mean([b.rear_abs_longitudinal_slip for b in rows]))
                  for key, rows in by_condition.items()}

    numerator = 0.0
    denominator = 0.0
    for key, rows in by_condition.items():
        if len(rows) < 2:
            continue
        for block in rows:
            dx = block.rear_abs_longitudinal_slip - slip_means[key]
            dy = block.yaw_gain_per_m - yaw_means[key]
            numerator += dx * dy
            denominator += dx * dx
    coefficient = numerator / denominator if denominator > 1e-12 else 0.0
    return yaw_means, slip_means, coefficient


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _self_test() -> None:
    training = [
        Block("a", 1, 2.2, 0.30, 1.0, 0.04),
        Block("a", 2, 2.2, 0.30, 1.2, 0.06),
        Block("b", 1, 2.2, 0.42, 1.5, 0.05),
        Block("b", 2, 2.2, 0.42, 1.7, 0.07),
    ]
    yaw_means, slip_means, coefficient = _fit_condition_means(training)
    assert math.isclose(coefficient, 10.0, rel_tol=1e-12)
    key = training[0].condition
    predicted = yaw_means[key] + coefficient * (0.08 - slip_means[key])
    assert math.isclose(predicted, 1.4, rel_tol=1e-12)
    print("rear-slip response math: PASS (condition centering, held-out correction)")


def evaluate(path: Path) -> tuple[float, float, float, bool]:
    blocks = _load_blocks(path)
    repetitions = sorted({block.repetition for block in blocks})
    if repetitions != [1, 2, 3]:
        raise ValueError(f"expected repetitions 1,2,3; found {repetitions} in {path}")
    training = [block for block in blocks if block.repetition in (1, 2)]
    heldout = [block for block in blocks if block.repetition == 3]
    expected_conditions = {block.condition for block in heldout}
    train_conditions = {block.condition for block in training}
    if expected_conditions != train_conditions:
        raise ValueError("training and held-out condition sets differ")

    yaw_means, slip_means, coefficient = _fit_condition_means(training)
    actual = np.array([block.yaw_gain_per_m for block in heldout])
    baseline = np.array([yaw_means[block.condition] for block in heldout])
    augmented = np.array([
        yaw_means[block.condition] + coefficient * (
            block.rear_abs_longitudinal_slip - slip_means[block.condition])
        for block in heldout
    ])
    base_rmse = _rmse(actual, baseline)
    augmented_rmse = _rmse(actual, augmented)
    improvement = 1.0 - augmented_rmse / base_rmse if base_rmse > 0.0 else 0.0
    print(f"{path}: {len(training)} train / {len(heldout)} held-out blocks; "
          f"rear-slip coefficient={coefficient:+.3f} (yaw-gain units per slip ratio)")
    print(f"  condition-only held-out RMSE={base_rmse:.4f} 1/m; "
          f"condition+rear-slip={augmented_rmse:.4f} 1/m "
          f"({improvement * 100:+.1f}% improvement)")
    for block, truth, pred_base, pred_aug in zip(heldout, actual, baseline, augmented):
        print(f"  {block.phase}: measured={truth:.3f}, base={pred_base:.3f}, "
              f"slip-model={pred_aug:.3f} 1/m; mean|Sx_rear|="
              f"{block.rear_abs_longitudinal_slip:.4f}")
    return base_rmse, augmented_rmse, coefficient, improvement >= 0.10


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.bags:
        parser.error("provide matched-start bags or use --self-test")
    results = [evaluate(path) for path in args.bags]
    accepted = all(result[3] for result in results)
    signs = {math.copysign(1.0, result[2]) for result in results if result[2] != 0.0}
    stable_sign = len(signs) <= 1
    print("predeclared test: rear-slip feature must reduce rep-3 yaw-gain RMSE by "
          "at least 10% in each speed capture and keep the fitted coefficient sign stable")
    print(f"result: {'PASS for a state-feature candidate' if accepted and stable_sign else 'REJECTED'}")
    return 0 if accepted and stable_sign else 1


if __name__ == "__main__":
    raise SystemExit(main())
