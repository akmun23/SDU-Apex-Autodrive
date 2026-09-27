#!/usr/bin/env python3
"""Held-out fine steering-transition test at 6.5 m/s on the open plane."""

from __future__ import annotations

import argparse
import math
import statistics
import sqlite3
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as common
from tools import evaluate_open_plane_highspeed_crossfactor as crossfactor
from tools import evaluate_open_plane_yaw_spline as yaw_spline


SPEED_MPS = 6.5
ANGLES = tuple(round(0.145 + 0.005 * index, 3) for index in range(8))
TRAIN_REPETITIONS = (1, 2)
HOLDOUT_REPETITION = 3
MIN_DERIVATIVE_MAGNITUDE = 0.10  # s^-1; below this, a sign is noise-sensitive.
MIN_RESOLVED_INTERVALS = 4
MAX_GAIN_RMSE = 0.03  # 1/m


def evaluate(path: Path, quality_ok: bool, speed_mps: float = SPEED_MPS,
             angles: tuple[float, ...] = ANGLES) -> bool:
    rows, run_end = crossfactor.load_blocks(path)
    expected = {(rep, sign, angle)
                for rep in (1, 2, 3)
                for sign in (-1, 1)
                for angle in angles}
    indexed = {
        (row["repetition"],
         1 if row["steering_command"] > 0.0 else -1,
         round(abs(row["steering_command"]), 4)): row
        for row in rows
        if row["mode"] == "speed_hold"
        and abs(row["speed_target"] - speed_mps) < 1e-6
    }
    complete = set(indexed) == expected
    phase_quality = complete and all(
        row["valid"] is True
        and math.isfinite(row["start_speed"])
        and abs(row["start_speed"] - speed_mps) <= 0.20
        and abs(row["speed"] - speed_mps) <= 0.20
        for row in indexed.values()
    )
    print(f"bag: {path}")
    print(f"run_end: aborted={run_end.get('aborted')}, reason={run_end.get('reason')!r}")
    print(f"target speed={speed_mps:.3f} m/s; probe coverage: "
          f"{len(indexed)}/{len(expected)}; "
          f"phase quality={'PASS' if phase_quality else 'FAIL'}")

    predicted: list[float] = []
    actual: list[float] = []
    direct_predicted: list[float] = []
    direct_actual: list[float] = []
    measured_slopes: list[float] = []
    predicted_slopes: list[float] = []
    direct_predicted_slopes: list[float] = []
    resolved_matches = 0
    direct_resolved_matches = 0
    resolved_intervals = 0
    training_complete = True
    for sign in (-1, 1):
        train_by_angle = {
            angle: [indexed[(rep, sign, angle)]["yaw_gain"]
                    for rep in TRAIN_REPETITIONS if (rep, sign, angle) in indexed]
            for angle in angles
        }
        if any(len(values) != 2 or not all(map(math.isfinite, values))
               for values in train_by_angle.values()):
            training_complete = False
            continue
        x = np.asarray(angles)
        y = np.asarray([statistics.mean(train_by_angle[angle])
                        for angle in angles])
        train_yaw_rate = {
            angle: [sign * indexed[(rep, sign, angle)]["yaw_rate"]
                    for rep in TRAIN_REPETITIONS
                    if (rep, sign, angle) in indexed]
            for angle in angles
        }
        if any(len(values) != 2 or not all(map(math.isfinite, values))
               for values in train_yaw_rate.values()):
            training_complete = False
            continue
        direct_y = np.asarray([statistics.mean(train_yaw_rate[angle])
                               for angle in angles])
        for angle in angles:
            held = indexed.get((HOLDOUT_REPETITION, sign, angle))
            if held is None or not math.isfinite(held["yaw_gain"]):
                training_complete = False
                continue
            gain, _ = yaw_spline._pchip_value_and_slope(x, y, angle)
            predicted.append(gain)
            actual.append(held["yaw_gain"])
            rate, _ = yaw_spline._pchip_value_and_slope(x, direct_y, angle)
            direct_predicted.append(rate)
            direct_actual.append(sign * held["yaw_rate"])

        for left_angle, right_angle in zip(angles, angles[1:]):
            left = indexed.get((HOLDOUT_REPETITION, sign, left_angle))
            right = indexed.get((HOLDOUT_REPETITION, sign, right_angle))
            if left is None or right is None:
                training_complete = False
                continue
            measured = (sign * right["yaw_rate"] - sign * left["yaw_rate"])
            measured /= right_angle - left_angle
            middle = 0.5 * (left_angle + right_angle)
            gain, gain_slope = yaw_spline._pchip_value_and_slope(x, y, middle)
            _rate, rate_slope = yaw_spline._pchip_value_and_slope(
                x, direct_y, middle)
            speed = 0.5 * (left["forward_speed"] + right["forward_speed"])
            model = speed * (gain / math.cos(middle) ** 2
                             + math.tan(middle) * gain_slope)
            measured_slopes.append(measured)
            predicted_slopes.append(model)
            if abs(measured) >= MIN_DERIVATIVE_MAGNITUDE:
                resolved_intervals += 1
                resolved_matches += measured * model > 0.0
                direct_resolved_matches += measured * rate_slope > 0.0
            direct_predicted_slopes.append(rate_slope)

    gain_rmse = (math.sqrt(statistics.mean((a - p) ** 2
                                           for a, p in zip(actual, predicted)))
                 if actual and len(actual) == len(predicted) else math.inf)
    direct_rmse = (math.sqrt(statistics.mean((a - p) ** 2
                                             for a, p in zip(direct_actual,
                                                             direct_predicted)))
                   if direct_actual and len(direct_actual) == len(direct_predicted)
                   else math.inf)
    jacobian_ok = (resolved_intervals >= MIN_RESOLVED_INTERVALS
                   and resolved_matches == resolved_intervals)
    print(f"rep-3 yaw-gain RMSE (PCHIP trained on reps 1–2): "
          f"{gain_rmse:.4f} 1/m (limit {MAX_GAIN_RMSE:.2f})")
    print(f"held-out d|r|/d|steer| measured: "
          f"{[round(v, 3) for v in measured_slopes]}")
    print(f"PCHIP local derivative: {[round(v, 3) for v in predicted_slopes]}")
    print(f"resolved Jacobian signs: {resolved_matches}/{resolved_intervals} "
          f"(need at least {MIN_RESOLVED_INTERVALS}, all matching)")
    print("post-hoc diagnostic only (direct |yaw_rate| surface; same holdout, "
          "not an independent acceptance):")
    print(f"  direct yaw-rate RMSE: {direct_rmse:.5f} rad/s")
    print(f"  direct-response slopes: "
          f"{[round(v, 3) for v in direct_predicted_slopes]}")
    print(f"  direct-response Jacobian signs: "
          f"{direct_resolved_matches}/{resolved_intervals}")

    accepted = (quality_ok and not run_end.get("aborted", True)
                and phase_quality and training_complete
                and gain_rmse <= MAX_GAIN_RMSE and jacobian_ok)
    print("decision: " + ("ACCEPT local surface for transient-model validation only"
                          if accepted else
                          "REJECT for model promotion; retain as diagnostic data"))
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--speed", type=float, default=SPEED_MPS,
                        help="target speed in m/s (default: 6.5)")
    parser.add_argument("--angles", nargs="+", type=float, default=ANGLES,
                        help="strictly increasing steering magnitudes in radians")
    args = parser.parse_args()
    if (not math.isfinite(args.speed) or args.speed <= 0.0
            or len(args.angles) < 3
            or any(not math.isfinite(angle) or angle <= 0.0 or angle > 0.50
                   for angle in args.angles)
            or any(left >= right for left, right in
                   zip(args.angles, args.angles[1:]))):
        parser.error("speed must be positive and angles must be strictly increasing in (0, 0.50]")
    try:
        quality_ok = common.analyze(args.bag) == 0
        return 0 if evaluate(args.bag, quality_ok, args.speed,
                             tuple(args.angles)) else 1
    except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
