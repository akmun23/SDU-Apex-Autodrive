#!/usr/bin/env python3
"""Hold out the sharp 4.5 m/s steering transition within one simulator bag.

Training uses repetitions 1--2 and the measured steering levels except
0.21--0.23 rad. Repetition 3 at those three levels (both signs) is untouched.
The capture itself may have aborted later, but the loader admits only the
complete, quality-gated 4.5 m/s group.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools import evaluate_open_plane_speed_steering_surface as speed_surface
from tools import evaluate_open_plane_yaw_spline as yaw_spline


TARGET_SPEED_MPS = 4.5
WITHHELD_ANGLES_RAD = (0.21, 0.22, 0.23)
TRANSITION_KNOTS_RAD = (0.20, 0.21, 0.22, 0.23, 0.25)
MAX_YAW_GAIN_RMSE_PER_M = 0.10
MIN_CONFIGURED_RMSE_REDUCTION = 0.50
MIN_SLOPE_FOR_SIGN_CHECK_PER_S = 0.05


def evaluate(path: Path) -> bool:
    blocks = speed_surface._load_complete_speed_group_from_partial_run(
        path, TARGET_SPEED_MPS)
    training = [block for block in blocks if block.repetition in (1, 2)]
    holdout = [block for block in blocks
               if block.repetition == 3
               and any(abs(abs(block.steering_rad) - angle) < 1e-6
                       for angle in WITHHELD_ANGLES_RAD)]
    expected = 2 * len(WITHHELD_ANGLES_RAD)
    if len(holdout) != expected:
        raise ValueError(f"expected {expected} held-out transition points, "
                         f"found {len(holdout)}")

    knots_by_sign: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sign in (-1, 1):
        grouped: dict[float, list[float]] = defaultdict(list)
        for block in training:
            if (1 if block.steering_rad > 0.0 else -1) != sign:
                continue
            angle = round(abs(block.steering_rad), 2)
            if angle not in WITHHELD_ANGLES_RAD:
                grouped[angle].append(block.yaw_gain_per_m)
        angles = np.asarray(sorted(grouped), dtype=float)
        gains = np.asarray([np.mean(grouped[angle]) for angle in angles], dtype=float)
        if len(angles) < 4 or any(angle not in grouped
                                  for angle in (0.15, 0.20, 0.25)):
            raise ValueError(f"insufficient local training knots for sign {sign:+d}")
        knots_by_sign[sign] = (angles, gains)

    actual: list[float] = []
    predicted: list[float] = []
    configured: list[float] = []
    for block in holdout:
        sign = 1 if block.steering_rad > 0.0 else -1
        angle = abs(block.steering_rad)
        x, y = knots_by_sign[sign]
        estimate, _ = yaw_spline._pchip_value_and_slope(x, y, angle)
        actual.append(block.yaw_gain_per_m)
        predicted.append(estimate)
        configured.append(yaw_spline._configured_gain(block.steering_rad))

    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    configured_array = np.asarray(configured)
    rmse = float(np.sqrt(np.mean((predicted_array - actual_array) ** 2)))
    max_error = float(np.max(np.abs(predicted_array - actual_array)))
    configured_rmse = float(np.sqrt(np.mean((configured_array - actual_array) ** 2)))
    reduction = (1.0 - rmse / configured_rmse
                 if configured_rmse > 1e-12 else 0.0)

    # Compare analytic model sensitivity with held-out finite differences of
    # yaw rate over the independently measured transition knots.
    heldout_means: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    for block in blocks:
        if block.repetition != 3:
            continue
        sign = 1 if block.steering_rad > 0.0 else -1
        angle = round(abs(block.steering_rad), 2)
        if angle in TRANSITION_KNOTS_RAD:
            yaw_rate = (block.measured_forward_speed_mps
                        * math.tan(angle) * block.yaw_gain_per_m)
            heldout_means[(sign, angle)].append(
                (block.measured_forward_speed_mps, yaw_rate))

    slopes: list[tuple[int, float, float, float, bool]] = []
    for sign in (-1, 1):
        x, y = knots_by_sign[sign]
        for left, right in zip(TRANSITION_KNOTS_RAD, TRANSITION_KNOTS_RAD[1:]):
            left_rows = heldout_means[(sign, left)]
            right_rows = heldout_means[(sign, right)]
            if not left_rows or not right_rows:
                raise ValueError(f"missing repetition-3 sensitivity knot: "
                                 f"sign={sign:+d}, {left:.2f}..{right:.2f}")
            left_rate = float(np.mean([row[1] for row in left_rows]))
            right_rate = float(np.mean([row[1] for row in right_rows]))
            observed = (right_rate - left_rate) / (right - left)
            midpoint = 0.5 * (left + right)
            gain, gain_slope = yaw_spline._pchip_value_and_slope(x, y, midpoint)
            speed = float(np.mean([row[0] for row in left_rows + right_rows]))
            predicted_slope = speed * (
                gain / math.cos(midpoint) ** 2
                + math.tan(midpoint) * gain_slope)
            checked = abs(observed) > MIN_SLOPE_FOR_SIGN_CHECK_PER_S
            agrees = (not checked or observed * predicted_slope > 0.0)
            slopes.append((sign, midpoint, observed, predicted_slope, agrees))

    slope_pass = all(row[4] for row in slopes)
    passed = (rmse <= MAX_YAW_GAIN_RMSE_PER_M
              and reduction >= MIN_CONFIGURED_RMSE_REDUCTION
              and slope_pass)
    print(f"bag: {path}")
    print("train: 4.5 m/s repetitions 1--2; steering knots exclude 0.21/0.22/0.23 rad")
    print("holdout: repetition 3 at 0.21/0.22/0.23 rad, both signs")
    print(f"  PCHIP yaw-gain RMSE={rmse:.5f} 1/m; max error={max_error:.5f} 1/m; "
          f"configured RMSE={configured_rmse:.5f} 1/m; reduction={reduction:+.1%}")
    print("  held-out yaw-gain rows (sign, steer, measured, predicted):")
    for block, estimate in zip(holdout, predicted):
        print(f"    {block.steering_rad:+.2f} {block.yaw_gain_per_m:.4f} "
              f"{estimate:.4f} 1/m")
    print("  local yaw-rate slopes (measured / model; rad/s per rad):")
    for sign, midpoint, observed, candidate, agrees in slopes:
        print(f"    sign={sign:+d} near {midpoint:.3f}: {observed:+.4f} / "
              f"{candidate:+.4f}; sign gate={'PASS' if agrees else 'FAIL'}")
    print("  decision: " + ("ACCEPT exact-speed local response map for dynamic-rollout testing"
                          if passed else
                          "REJECT steering-only local map for this transition"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
