#!/usr/bin/env python3
"""Held-out test of a speed-dependent lateral-acceleration/yaw envelope.

Development-only offline analysis. It compares the configured steering-only
yaw-gain model against the grey-box limit

    |r_ss| <= a_y,max / |u|

using matched-start isolated probes at 2.2 and 3.0 m/s. It is not a plant
model, MPC input, or runtime node; it only decides whether this reduced model
is worth implementing and validating on-track.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis


# Current mpc_competition.yaml response parameters, mirrored for offline A/B.
GAIN = 2.95
GAIN_REDUCTION_PER_RAD = 35.6
GAIN_START_RAD = 0.41
GAIN_END_RAD = 0.46
SETTLE_NS = analysis.PHASE_SETTLE_NS

# Acceptance criteria fixed before evaluating the captured bags.
MIN_POOLED_RMSE_REDUCTION = 0.20
MAX_PER_SPEED_REGRESSION = 0.10
PLAUSIBLE_LATERAL_ACCEL_MPS2 = (3.5, 5.5)


@dataclass(frozen=True)
class Block:
    bag: str
    phase: str
    repetition: int
    target_speed_mps: float
    speed_mps: float
    steering_rad: float
    measured_yaw_rate_rps: float


def _configured_gain(steering_rad: float) -> float:
    active_interval = min(max(abs(steering_rad) - GAIN_START_RAD, 0.0),
                          GAIN_END_RAD - GAIN_START_RAD)
    return GAIN - GAIN_REDUCTION_PER_RAD * active_interval


def _envelope_yaw_rate(speed_mps: float, steering_rad: float,
                       lateral_limit_mps2: float) -> float:
    kinematic = speed_mps * math.tan(steering_rad) * GAIN
    if speed_mps <= 0.0:
        raise ValueError("the lateral envelope requires forward speed")
    limit = lateral_limit_mps2 / speed_mps
    return math.copysign(min(abs(kinematic), limit), kinematic)


def _envelope_steering_jacobian(speed_mps: float, steering_rad: float,
                                lateral_limit_mps2: float) -> float:
    """Derivative of the hard-clipped yaw map with respect to steering."""
    kinematic = speed_mps * math.tan(steering_rad) * GAIN
    if speed_mps <= 0.0:
        raise ValueError("the lateral envelope requires forward speed")
    if abs(kinematic) >= lateral_limit_mps2 / speed_mps:
        return 0.0
    return speed_mps * GAIN / math.cos(steering_rad) ** 2


def _load_blocks(path: Path) -> tuple[list[Block], float | None, int, int]:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, analysis.STEERING, analysis.PHASE,
                    analysis.COLLISIONS, analysis.PACKET_TIMING)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError("bag is missing required topic(s): " + ", ".join(missing))

        odometry = []
        for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM):
            twist = message.twist.twist
            odometry.append((receipt_ns, float(twist.linear.x),
                             float(twist.angular.z)))
        steering = [
            analysis.ScalarRow(receipt_ns, receipt_ns, float(message.data))
            for receipt_ns, message in analysis._messages(connection, topics, analysis.STEERING)
        ]
        steering_times = [row.receipt_ns for row in steering]
        collision_values = [int(message.data)
                            for _, message in analysis._messages(
                                connection, topics, analysis.COLLISIONS)]
        timing_receipts = [receipt_ns for receipt_ns, _ in analysis._messages(
            connection, topics, analysis.PACKET_TIMING)]
        rate_hz, _, _ = analysis._rate(timing_receipts)
        phases, end = analysis._phase_events(connection, topics)
    finally:
        connection.close()

    odom_times = [row[0] for row in odometry]
    blocks: list[Block] = []
    for phase in phases:
        if not phase.label.startswith("isolated_r") or phase.valid is not True:
            continue
        repetition = int(phase.label.split("_", 2)[1][1:])
        start_ns = phase.start_ns + SETTLE_NS
        begin = int(np.searchsorted(odom_times, start_ns, side="left"))
        end_index = int(np.searchsorted(odom_times, phase.end_ns, side="left"))
        speeds: list[float] = []
        yaw_rates: list[float] = []
        steer_values: list[float] = []
        for receipt_ns, speed, yaw_rate in odometry[begin:end_index]:
            steer = analysis._nearest_scalar(steering, steering_times, receipt_ns)
            if steer is None:
                continue
            if not all(math.isfinite(value) for value in (speed, yaw_rate, steer.value)):
                continue
            speeds.append(abs(speed))
            yaw_rates.append(yaw_rate)
            steer_values.append(steer.value)
        if len(speeds) < 30 or not steer_values:
            continue
        blocks.append(Block(
            bag=path.name,
            phase=phase.label,
            repetition=repetition,
            target_speed_mps=phase.target_speed_mps,
            speed_mps=float(np.median(speeds)),
            steering_rad=float(np.median(steer_values)),
            measured_yaw_rate_rps=float(np.median(yaw_rates)),
        ))

    if not blocks:
        raise ValueError(f"no valid isolated steering probes found in {path}")
    if end.get("aborted") is not False or end.get("quality_failures"):
        raise ValueError(f"capture did not complete cleanly: {end}")
    if collision_values and max(collision_values) != 0:
        raise ValueError(f"capture contains collision-count increase: {max(collision_values)}")
    return blocks, rate_hz, len(collision_values), sum(collision_values)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _self_test() -> None:
    cap = 4.4
    low = _envelope_yaw_rate(2.2, 0.30, cap)
    high = _envelope_yaw_rate(3.0, 0.30, cap)
    reverse = _envelope_yaw_rate(3.0, -0.50, cap)
    assert low > 0.0 and high > 0.0 and reverse < 0.0
    assert math.isclose(high, cap / 3.0, rel_tol=0.02)
    assert abs(reverse) < abs(_envelope_yaw_rate(2.2, -0.50, cap))
    assert _envelope_steering_jacobian(3.0, 0.30, cap) == 0.0
    assert _envelope_steering_jacobian(2.2, 0.20, cap) > 0.0
    assert _configured_gain(0.50) < _configured_gain(0.30)
    print("lateral-envelope mathematics: PASS (speed saturation, sign symmetry, steering Jacobian)")


def evaluate(paths: list[Path]) -> int:
    blocks: list[Block] = []
    for path in paths:
        capture, rate, collision_samples, collision_max = _load_blocks(path)
        blocks.extend(capture)
        print(f"capture {path}: blocks={len(capture)}, packet_rate={rate:.3f} Hz, "
              f"collision_samples={collision_samples}, collision_max={collision_max}")

    repetitions = sorted({block.repetition for block in blocks})
    if len(repetitions) < 3:
        raise ValueError(f"three repetitions are required for held-out testing; got {repetitions}")
    heldout_repetition = repetitions[-1]
    training = [block for block in blocks if block.repetition != heldout_repetition]
    heldout = [block for block in blocks if block.repetition == heldout_repetition]
    if not training or not heldout:
        raise ValueError("empty train or held-out split")

    def predict(block: Block, cap: float, candidate: bool) -> float:
        if candidate:
            return _envelope_yaw_rate(block.speed_mps, block.steering_rad, cap)
        return (block.speed_mps * math.tan(block.steering_rad)
                * _configured_gain(block.steering_rad))

    best_cap = math.nan
    best_error = math.inf
    for cap in np.linspace(2.0, 8.0, 601):
        predicted = np.array([predict(block, float(cap), True) for block in training])
        actual = np.array([block.measured_yaw_rate_rps for block in training])
        error = _rmse(actual, predicted)
        if error < best_error:
            best_error, best_cap = error, float(cap)

    actual = np.array([block.measured_yaw_rate_rps for block in heldout])
    old = np.array([predict(block, best_cap, False) for block in heldout])
    candidate = np.array([predict(block, best_cap, True) for block in heldout])
    old_rmse = _rmse(actual, old)
    candidate_rmse = _rmse(actual, candidate)
    pooled_improvement = 1.0 - candidate_rmse / old_rmse if old_rmse > 0 else 0.0

    print(f"train repetitions={','.join(map(str, repetitions[:-1]))}, "
          f"held-out repetition={heldout_repetition}; blocks={len(training)}/{len(heldout)}")
    print(f"fitted lateral acceleration limit={best_cap:.3f} m/s^2 "
          f"({best_cap / analysis.GRAVITY_MPS2:.3f} g); train yaw RMSE="
          f"{best_error:.4f} rad/s")
    print(f"held-out yaw-rate RMSE: configured steering-only={old_rmse:.4f}, "
          f"speed-dependent envelope={candidate_rmse:.4f} rad/s "
          f"({pooled_improvement * 100:+.1f}% improvement)")

    flat_gradient = sum(
        _envelope_steering_jacobian(block.speed_mps, block.steering_rad, best_cap) == 0.0
        for block in heldout
    )
    print(f"control-sensitivity diagnostic: hard envelope has zero d(yaw_rate)/d(steer) "
          f"in {flat_gradient}/{len(heldout)} held-out operating points")

    per_speed_ok = True
    for speed in sorted({block.target_speed_mps for block in heldout}):
        subset = [block for block in heldout if block.target_speed_mps == speed]
        y = np.array([block.measured_yaw_rate_rps for block in subset])
        old_speed = _rmse(y, np.array([predict(block, best_cap, False) for block in subset]))
        new_speed = _rmse(y, np.array([predict(block, best_cap, True) for block in subset]))
        regression = new_speed / old_speed - 1.0 if old_speed > 0.0 else math.inf
        per_speed_ok &= regression <= MAX_PER_SPEED_REGRESSION
        print(f"  target={speed:.1f} m/s: {len(subset)} holdout blocks, "
              f"RMSE {old_speed:.4f}→{new_speed:.4f} rad/s ({regression * 100:+.1f}%)")

    accepted = (
        pooled_improvement >= MIN_POOLED_RMSE_REDUCTION
        and per_speed_ok
        and PLAUSIBLE_LATERAL_ACCEL_MPS2[0] <= best_cap <= PLAUSIBLE_LATERAL_ACCEL_MPS2[1]
    )
    print("predeclared screen: ≥20% pooled held-out improvement, ≤10% regression at "
          "either speed, and fitted limit 3.5–5.5 m/s²")
    print(f"offline prediction screen: {'PASS' if accepted else 'REJECTED'}")
    print("MPC promotion: NOT APPROVED by this screen alone; the hard cap's local "
          "steering Jacobian and prior high-curvature closed-loop failure must be addressed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="*", type=Path,
                        help="matched-start isolated open-plane bags at different speeds")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if len(args.bags) < 2:
        parser.error("provide both 2.2 m/s and 3.0 m/s isolated bags")
    return evaluate(args.bags)


if __name__ == "__main__":
    raise SystemExit(main())
