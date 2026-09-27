#!/usr/bin/env python3
"""Test cross-speed prediction through the measured high-steer response cliff.

Training uses the clean 4.0 m/s full-angle sweep and matched repetitions 1--2
from the later-aborted 6.5 m/s capture. The complete 4.5 m/s matched group is
the unseen-speed holdout. This is offline identification only; it never feeds
ground truth into a driving controller.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import evaluate_open_plane_speed_steering_surface as speed_surface
from tools import evaluate_open_plane_transient_rollout as transient
from tools import evaluate_open_plane_yaw_spline as yaw_spline


SOURCE_SPEEDS_MPS = (4.0, 6.5)
HELDOUT_SPEED_MPS = 4.5
TRANSITION_ANGLES_RAD = (0.15, 0.20, 0.21, 0.22, 0.23, 0.25)
FULL_ANGLE_LEVELS_RAD = (0.05, 0.10, 0.15, 0.20, 0.25,
                         0.30, 0.35, 0.42, 0.46, 0.50)
SOURCE_4_ANGLE_KNOTS_RAD = tuple(angle for angle in FULL_ANGLE_LEVELS_RAD
                                 if angle >= 0.15)
MIN_RMSE_REDUCTION = 0.50
MAX_YAW_GAIN_RMSE_PER_M = 0.10
MIN_SLOPE_FOR_SIGN_CHECK_PER_S = 0.05
SETTLED_START_S = 0.55


@dataclass(frozen=True)
class ResponseBlock:
    phase: str
    repetition: int
    target_speed_mps: float
    steering_rad: float
    yaw_gain_per_m: float
    measured_forward_speed_mps: float


def _load_4mps_source(path: Path) -> list[ResponseBlock]:
    probes, quality = transient.load(path)
    expected_per_rep = 2 * (2 * len(FULL_ANGLE_LEVELS_RAD) - 1)
    expected_quality = {
        "valid_steps": 3 * expected_per_rep,
        "matched_starts": 6,
        "collisions": (0, 0),
        "timing_faults": 0,
        "aborted": False,
        "probe_counts": {1: expected_per_rep, 2: expected_per_rep,
                         3: expected_per_rep, 4: 0},
    }
    if quality != expected_quality:
        raise ValueError(f"4.0 m/s source capture failed its quality gate: {quality}")
    angles = sorted({round(abs(probe.commanded_steering_rad), 2)
                     for probe in probes})
    if tuple(angles) != FULL_ANGLE_LEVELS_RAD:
        raise ValueError(f"unexpected 4.0 m/s source steering grid: {angles}")

    blocks: list[ResponseBlock] = []
    for probe in probes:
        if abs(probe.commanded_steering_rad) < SOURCE_4_ANGLE_KNOTS_RAD[0]:
            continue
        settled = [sample for sample in probe.samples
                   if sample.time_s >= SETTLED_START_S
                   and sample.speed_mps > 1.0
                   and abs(sample.steering_rad) > 0.10]
        gains = [sample.yaw_rate_rps /
                 (sample.speed_mps * math.tan(sample.steering_rad))
                 for sample in settled
                 if abs(sample.speed_mps * math.tan(sample.steering_rad)) > 0.1]
        speeds = [sample.speed_mps for sample in settled
                  if abs(sample.speed_mps * math.tan(sample.steering_rad)) > 0.1]
        if len(gains) < 7:
            raise ValueError(f"too few settled source samples: "
                             f"r{probe.repetition} {probe.commanded_steering_rad:+.2f}")
        blocks.append(ResponseBlock(
            phase=f"fullsteer_r{probe.repetition}_{probe.commanded_steering_rad:+.2f}",
            repetition=probe.repetition,
            target_speed_mps=4.0,
            steering_rad=probe.commanded_steering_rad,
            yaw_gain_per_m=float(np.median(gains)),
            measured_forward_speed_mps=float(np.median(speeds)),
        ))
    return blocks


def _source_curve(
        blocks: list[ResponseBlock], source_speed: float, sign: int
        ) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for block in blocks:
        if (abs(block.target_speed_mps - source_speed) < 1e-6
                and (1 if block.steering_rad > 0.0 else -1) == sign):
            grouped[round(abs(block.steering_rad), 2)].append(block.yaw_gain_per_m)
    expected = (SOURCE_4_ANGLE_KNOTS_RAD if source_speed == 4.0
                else speed_surface.PARTIAL_GROUP_ANGLES_RAD)
    if tuple(sorted(grouped)) != expected:
        raise ValueError(f"source {source_speed:.1f} m/s sign={sign:+d} has knots "
                         f"{sorted(grouped)}, expected {expected}")
    angles = np.asarray(sorted(grouped), dtype=float)
    gains = np.asarray([float(np.mean(grouped[angle])) for angle in angles])
    if not np.all(np.isfinite(gains)):
        raise ValueError(f"non-finite source curve at {source_speed:.1f} m/s")
    return angles, gains


def evaluate(source_4mps: Path, mixed_65mps: Path,
             holdout_45mps: Path) -> bool:
    source_4 = _load_4mps_source(source_4mps)
    source_65_raw = speed_surface._load_complete_speed_group_from_partial_run(
        mixed_65mps, 6.5, repetitions=(1, 2))
    source_65 = [ResponseBlock(
        block.phase, block.repetition, block.target_speed_mps,
        block.steering_rad, block.yaw_gain_per_m,
        block.measured_forward_speed_mps,
    ) for block in source_65_raw]
    holdout_raw = speed_surface._load_complete_speed_group_from_partial_run(
        holdout_45mps, HELDOUT_SPEED_MPS)
    holdout = [ResponseBlock(
        block.phase, block.repetition, block.target_speed_mps,
        block.steering_rad, block.yaw_gain_per_m,
        block.measured_forward_speed_mps,
    ) for block in holdout_raw
        if abs(block.steering_rad) in TRANSITION_ANGLES_RAD]

    expected_source_65 = 2 * len(speed_surface.PARTIAL_GROUP_ANGLES_RAD) * 2
    if len(source_65) != expected_source_65:
        raise ValueError(f"6.5 m/s source must have 40 complete matched blocks; "
                         f"found {len(source_65)}")
    if (len(holdout) != 36
            or sorted({block.repetition for block in holdout}) != [1, 2, 3]
            or {round(abs(block.steering_rad), 2) for block in holdout}
            != set(TRANSITION_ANGLES_RAD)):
        raise ValueError(f"4.5 m/s transition holdout coverage is incomplete: "
                         f"{len(holdout)} blocks")

    source_blocks = source_4 + source_65
    curves = {
        (speed, sign, angle): _source_curve(source_blocks, speed, sign)
        for speed in SOURCE_SPEEDS_MPS
        for sign in (-1, 1)
        for angle in TRANSITION_ANGLES_RAD
    }
    measured: list[float] = []
    acceleration_predictions: list[float] = []
    linear_gain_predictions: list[float] = []
    configured: list[float] = []
    yaw_rate_errors: list[float] = []
    sensitivity_by_sign_angle: dict[tuple[int, float], list[float]] = defaultdict(list)

    for block in holdout:
        angle = round(abs(block.steering_rad), 2)
        sign = 1 if block.steering_rad > 0.0 else -1
        model_curves = {
            (speed, sign, angle): curves[(speed, sign, angle)]
            for speed in SOURCE_SPEEDS_MPS
        }
        acceleration_gain, _ = speed_surface._predict_lateral_acceleration(
            block.measured_forward_speed_mps, block.steering_rad,
            SOURCE_SPEEDS_MPS, model_curves)
        linear_gain, _ = speed_surface._predict(
            block.measured_forward_speed_mps, block.steering_rad,
            SOURCE_SPEEDS_MPS, model_curves)
        measured.append(block.yaw_gain_per_m)
        acceleration_predictions.append(acceleration_gain)
        linear_gain_predictions.append(linear_gain)
        configured.append(yaw_spline._configured_gain(block.steering_rad))
        yaw_rate_errors.append(
            block.measured_forward_speed_mps
            * math.tan(block.steering_rad)
            * (acceleration_gain - block.yaw_gain_per_m))
        sign_corrected_yaw_rate = (
            block.measured_forward_speed_mps
            * math.tan(abs(block.steering_rad))
            * block.yaw_gain_per_m
        )
        sensitivity_by_sign_angle[(sign, angle)].append(sign_corrected_yaw_rate)

    measured_array = np.asarray(measured)
    acceleration_array = np.asarray(acceleration_predictions)
    linear_array = np.asarray(linear_gain_predictions)
    configured_array = np.asarray(configured)
    acceleration_rmse = float(np.sqrt(np.mean((acceleration_array - measured_array) ** 2)))
    linear_rmse = float(np.sqrt(np.mean((linear_array - measured_array) ** 2)))
    configured_rmse = float(np.sqrt(np.mean((configured_array - measured_array) ** 2)))
    max_error = float(np.max(np.abs(acceleration_array - measured_array)))
    yaw_rate_rmse = float(np.sqrt(np.mean(np.square(yaw_rate_errors))))
    reduction = (1.0 - acceleration_rmse / configured_rmse
                 if configured_rmse > 1e-12 else 0.0)

    slope_checks: list[tuple[int, float, float, float, bool]] = []
    for sign in (-1, 1):
        for left, right in zip(TRANSITION_ANGLES_RAD, TRANSITION_ANGLES_RAD[1:]):
            left_rate = float(np.mean(sensitivity_by_sign_angle[(sign, left)]))
            right_rate = float(np.mean(sensitivity_by_sign_angle[(sign, right)]))
            measured_slope = (right_rate - left_rate) / (right - left)
            midpoint = 0.5 * (left + right)
            query_speed = float(np.mean([block.measured_forward_speed_mps
                                         for block in holdout]))
            sensitivity_curves = {
                (speed, sign, midpoint): curves[(speed, sign, left)]
                for speed in SOURCE_SPEEDS_MPS
            }
            _, predicted_slope = speed_surface._predict_lateral_acceleration(
                query_speed, sign * midpoint, SOURCE_SPEEDS_MPS,
                sensitivity_curves)
            checked = abs(measured_slope) > MIN_SLOPE_FOR_SIGN_CHECK_PER_S
            agrees = (not checked or
                      (math.isfinite(predicted_slope)
                       and predicted_slope * measured_slope > 0.0))
            slope_checks.append((sign, midpoint, measured_slope,
                                 predicted_slope, agrees))

    slope_pass = all(item[4] for item in slope_checks)
    passed = (acceleration_rmse <= MAX_YAW_GAIN_RMSE_PER_M
              and reduction >= MIN_RMSE_REDUCTION
              and slope_pass)

    print("training: clean 4.0 m/s full-angle sweep + 6.5 m/s matched reps 1--2")
    print("holdout: complete 4.5 m/s matched group; 0.15--0.25 rad; "
          "36 speed-unseen response blocks")
    print("model compares direct K_yaw blending with signed lateral-acceleration "
          "blending; all queries are inside measured steering/speed support")
    print(f"  acceleration blend RMSE={acceleration_rmse:.5f} 1/m; "
          f"max error={max_error:.5f} 1/m; configured RMSE={configured_rmse:.5f} 1/m; "
          f"reduction={reduction:+.1%}; yaw-rate RMSE={yaw_rate_rmse:.5f} rad/s")
    print(f"  direct yaw-gain blend RMSE={linear_rmse:.5f} 1/m")
    print("  angle   measured-K  acceleration-K  direct-K  configured-K")
    for angle in TRANSITION_ANGLES_RAD:
        indexes = [index for index, block in enumerate(holdout)
                   if round(abs(block.steering_rad), 2) == angle]
        print(f"  {angle:.2f}    "
              f"{np.mean(measured_array[indexes]):.4f}       "
              f"{np.mean(acceleration_array[indexes]):.4f}          "
              f"{np.mean(linear_array[indexes]):.4f}     "
              f"{np.mean(configured_array[indexes]):.4f} 1/m")
    print("  local sign-corrected yaw-rate slopes (measured / predicted, 1/s):")
    for sign, midpoint, observed, predicted, agrees in slope_checks:
        print(f"    sign={sign:+d} at {midpoint:.3f} rad: {observed:+.4f} / "
              f"{predicted:+.4f}; sign gate={'PASS' if agrees else 'FAIL'}")
    print("  decision: " + ("ACCEPT within this tested transition band"
                          if passed else
                          "REJECT speed-only interpolation for this transition band"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_4mps", type=Path)
    parser.add_argument("mixed_65mps", type=Path,
                        help="partial capture; only fully matched repetitions 1--2 are used")
    parser.add_argument("holdout_45mps", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.source_4mps, args.mixed_65mps,
                         args.holdout_45mps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
