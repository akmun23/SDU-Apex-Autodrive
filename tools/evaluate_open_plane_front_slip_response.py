#!/usr/bin/env python3
"""Cross-speed test of a guide-shaped front-slip/lateral-response model.

Fit an empirical PCHIP from the reconstructed mean front-wheel lateral-slip
proxy to sign-corrected chassis lateral acceleration using only 4.0 and
6.5 m/s source data. Score the complete 4.5 m/s open-plane group in the
0.15--0.25 rad transition. The proxy comes from body kinematics and Ackermann
geometry; it is not Unity's native WheelHit slip or a measured tire force.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools import evaluate_open_plane_speed_steering_surface as speed_surface
from tools import evaluate_open_plane_transient_rollout as transient
from tools import evaluate_open_plane_yaw_spline as yaw_spline


TRANSITION_ANGLES_RAD = (0.15, 0.20, 0.21, 0.22, 0.23, 0.25)
FULL_ANGLE_LEVELS_RAD = (0.05, 0.10, 0.15, 0.20, 0.25,
                         0.30, 0.35, 0.42, 0.46, 0.50)
SETTLED_START_S = 0.55
MIN_RMSE_REDUCTION = 0.50
MAX_YAW_GAIN_RMSE_PER_M = 0.10
MIN_SLOPE_FOR_SIGN_CHECK_PER_S = 0.05


@dataclass(frozen=True)
class SlipResponseBlock:
    phase: str
    repetition: int
    target_speed_mps: float
    steering_rad: float
    front_abs_slip_proxy: float
    signed_lateral_acceleration_mps2: float
    measured_forward_speed_mps: float


def _front_slip(vx: float, vy: float, yaw_rate: float,
                steering_rad: float) -> float:
    left_angle, right_angle = analysis._ackermann_angles(steering_rad)
    half_track = analysis.TRACK_WIDTH_M / 2.0
    left = analysis._wheel_slip(
        vx, vy, yaw_rate, analysis.WHEELBASE_M, half_track, left_angle)
    right = analysis._wheel_slip(
        vx, vy, yaw_rate, analysis.WHEELBASE_M, -half_track, right_angle)
    if left is None or right is None:
        raise ValueError("front-wheel slip is undefined at this operating point")
    return 0.5 * (abs(left) + abs(right))


def _load_4mps_source(path: Path) -> list[SlipResponseBlock]:
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
        raise ValueError(f"4.0 m/s source capture failed quality gate: {quality}")
    angles = sorted({round(abs(probe.commanded_steering_rad), 2)
                     for probe in probes})
    if tuple(angles) != FULL_ANGLE_LEVELS_RAD:
        raise ValueError(f"unexpected 4.0 m/s steering grid: {angles}")

    blocks: list[SlipResponseBlock] = []
    for probe in probes:
        settled = [sample for sample in probe.samples
                   if sample.time_s >= SETTLED_START_S and sample.speed_mps > 1.0]
        sign = 1.0 if probe.commanded_steering_rad > 0.0 else -1.0
        slips = [_front_slip(sample.speed_mps, sample.vy_mps,
                             sample.yaw_rate_rps, sample.steering_rad)
                 for sample in settled]
        lateral_accels = [sign * sample.speed_mps * sample.yaw_rate_rps
                          for sample in settled]
        if len(slips) < 7:
            raise ValueError(f"too few settled source samples in {probe}")
        blocks.append(SlipResponseBlock(
            phase=f"fullsteer_r{probe.repetition}_{probe.commanded_steering_rad:+.2f}",
            repetition=probe.repetition,
            target_speed_mps=4.0,
            steering_rad=probe.commanded_steering_rad,
            front_abs_slip_proxy=float(np.median(slips)),
            signed_lateral_acceleration_mps2=float(np.median(lateral_accels)),
            measured_forward_speed_mps=float(np.median(
                [sample.speed_mps for sample in settled])),
        ))
    return blocks


def _collapse_slip_knots(
        points: list[SlipResponseBlock]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for point in points:
        grouped[round(point.front_abs_slip_proxy, 4)].append(
            point.signed_lateral_acceleration_mps2)
    slips = np.asarray(sorted(grouped), dtype=float)
    accelerations = np.asarray([
        float(np.mean(grouped[slip])) for slip in slips
    ], dtype=float)
    if len(slips) < 5 or not np.all(np.diff(slips) > 0.0):
        raise ValueError("front-slip curve has insufficient distinct support")
    if not np.all(np.isfinite(accelerations)):
        raise ValueError("front-slip curve contains a non-finite response")
    return slips, accelerations


def _predict_acceleration(slip: float, curve: tuple[np.ndarray, np.ndarray]) -> float:
    slips, accelerations = curve
    if slip < slips[0] or slip > slips[-1]:
        raise ValueError(f"front-slip query {slip:.5f} outside measured source support "
                         f"[{slips[0]:.5f}, {slips[-1]:.5f}]")
    value, _ = yaw_spline._pchip_value_and_slope(slips, accelerations, slip)
    return value


def evaluate(source_4mps: Path, mixed_65mps: Path,
             holdout_45mps: Path) -> bool:
    source_4 = _load_4mps_source(source_4mps)
    source_65_raw = speed_surface._load_complete_speed_group_from_partial_run(
        mixed_65mps, 6.5, repetitions=(1, 2))
    source_65 = []
    for block in source_65_raw:
        if (block.front_lateral_slip_proxy is None
                or block.lateral_acceleration_mps2 is None):
            raise ValueError(f"missing 6.5 m/s slip/acceleration in {block.phase}")
        sign = 1.0 if block.steering_rad > 0.0 else -1.0
        source_65.append(SlipResponseBlock(
            block.phase, block.repetition, block.target_speed_mps,
            block.steering_rad, block.front_lateral_slip_proxy,
            sign * block.lateral_acceleration_mps2,
            block.measured_forward_speed_mps,
        ))

    holdout_raw = speed_surface._load_complete_speed_group_from_partial_run(
        holdout_45mps, 4.5)
    holdout: list[SlipResponseBlock] = []
    for block in holdout_raw:
        if round(abs(block.steering_rad), 2) not in TRANSITION_ANGLES_RAD:
            continue
        if (block.front_lateral_slip_proxy is None
                or block.lateral_acceleration_mps2 is None):
            raise ValueError(f"missing 4.5 m/s slip/acceleration in {block.phase}")
        sign = 1.0 if block.steering_rad > 0.0 else -1.0
        holdout.append(SlipResponseBlock(
            block.phase, block.repetition, block.target_speed_mps,
            block.steering_rad, block.front_lateral_slip_proxy,
            sign * block.lateral_acceleration_mps2,
            block.measured_forward_speed_mps,
        ))
    if len(source_65) != 40 or len(holdout) != 36:
        raise ValueError(f"incomplete matched data: 6.5m/s={len(source_65)}, "
                         f"4.5m/s holdout={len(holdout)}")

    curve = _collapse_slip_knots(source_4 + source_65)
    holdout_slips = [block.front_abs_slip_proxy for block in holdout]
    if min(holdout_slips) < curve[0][0] or max(holdout_slips) > curve[0][-1]:
        raise ValueError("4.5 m/s front-slip holdout requires extrapolation: "
                         f"target=[{min(holdout_slips):.5f}, {max(holdout_slips):.5f}], "
                         f"source=[{curve[0][0]:.5f}, {curve[0][-1]:.5f}]")

    measured_gain: list[float] = []
    predicted_gain: list[float] = []
    configured_gain: list[float] = []
    yaw_rate_errors: list[float] = []
    predicted_yaw_rate: dict[tuple[int, float], list[float]] = defaultdict(list)
    measured_yaw_rate: dict[tuple[int, float], list[float]] = defaultdict(list)

    for block in holdout:
        sign = 1 if block.steering_rad > 0.0 else -1
        magnitude = abs(block.steering_rad)
        predicted_accel = _predict_acceleration(
            block.front_abs_slip_proxy, curve)
        denominator = block.measured_forward_speed_mps ** 2 * math.tan(magnitude)
        predicted_k = predicted_accel / denominator
        measured_k = (block.signed_lateral_acceleration_mps2 / denominator)
        measured_gain.append(measured_k)
        predicted_gain.append(predicted_k)
        configured_gain.append(yaw_spline._configured_gain(block.steering_rad))
        yaw_rate_errors.append(
            (predicted_accel - block.signed_lateral_acceleration_mps2)
            / block.measured_forward_speed_mps)
        angle = round(magnitude, 2)
        predicted_yaw_rate[(sign, angle)].append(
            predicted_accel / block.measured_forward_speed_mps)
        measured_yaw_rate[(sign, angle)].append(
            block.signed_lateral_acceleration_mps2
            / block.measured_forward_speed_mps)

    actual = np.asarray(measured_gain)
    predicted = np.asarray(predicted_gain)
    baseline = np.asarray(configured_gain)
    rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
    configured_rmse = float(np.sqrt(np.mean(np.square(baseline - actual))))
    max_error = float(np.max(np.abs(predicted - actual)))
    reduction = 1.0 - rmse / configured_rmse if configured_rmse > 1e-12 else 0.0
    yaw_rate_rmse = float(np.sqrt(np.mean(np.square(yaw_rate_errors))))

    slope_checks = []
    for sign in (-1, 1):
        for left, right in zip(TRANSITION_ANGLES_RAD, TRANSITION_ANGLES_RAD[1:]):
            measured_left = float(np.mean(measured_yaw_rate[(sign, left)]))
            measured_right = float(np.mean(measured_yaw_rate[(sign, right)]))
            predicted_left = float(np.mean(predicted_yaw_rate[(sign, left)]))
            predicted_right = float(np.mean(predicted_yaw_rate[(sign, right)]))
            observed_slope = (measured_right - measured_left) / (right - left)
            candidate_slope = (predicted_right - predicted_left) / (right - left)
            checked = abs(observed_slope) > MIN_SLOPE_FOR_SIGN_CHECK_PER_S
            agrees = (not checked or
                      candidate_slope * observed_slope > 0.0)
            slope_checks.append((sign, 0.5 * (left + right), observed_slope,
                                 candidate_slope, agrees))
    slope_pass = all(item[4] for item in slope_checks)
    passed = (rmse <= MAX_YAW_GAIN_RMSE_PER_M
              and reduction >= MIN_RMSE_REDUCTION and slope_pass)

    print("source: 4.0 m/s full-angle sweep + matched 6.5 m/s reps 1--2")
    print("holdout: complete 4.5 m/s group, 0.15--0.25 rad; "
          f"{len(holdout)} independent phase blocks")
    print(f"slip support: train=[{curve[0][0]:.5f},{curve[0][-1]:.5f}], "
          f"holdout=[{min(holdout_slips):.5f},{max(holdout_slips):.5f}] (no extrapolation)")
    print(f"front-slip PCHIP: yaw-gain RMSE={rmse:.5f} 1/m; "
          f"max error={max_error:.5f} 1/m; configured RMSE="
          f"{configured_rmse:.5f} 1/m; reduction={reduction:+.1%}; "
          f"yaw-rate RMSE={yaw_rate_rmse:.5f} rad/s")
    print("  angle  mean-front-|Sy|  measured-a_y/g  predicted-a_y/g  measured-K  predicted-K")
    for angle in TRANSITION_ANGLES_RAD:
        rows = [block for block in holdout
                if round(abs(block.steering_rad), 2) == angle]
        predictions_at_angle = [
            _predict_acceleration(block.front_abs_slip_proxy, curve)
            for block in rows
        ]
        denominator = [block.measured_forward_speed_mps ** 2
                       * math.tan(abs(block.steering_rad)) for block in rows]
        mean_k = float(np.mean([
            block.signed_lateral_acceleration_mps2 / value
            for block, value in zip(rows, denominator)
        ]))
        pred_k = float(np.mean([value / den
                                for value, den in zip(predictions_at_angle, denominator)]))
        print(f"  {angle:.2f}      "
              f"{np.mean([b.front_abs_slip_proxy for b in rows]):.4f}          "
              f"{np.mean([b.signed_lateral_acceleration_mps2 for b in rows]) / analysis.GRAVITY_MPS2:.3f}          "
              f"{np.mean(predictions_at_angle) / analysis.GRAVITY_MPS2:.3f}          "
              f"{mean_k:.3f}      {pred_k:.3f} 1/m")
    print("  local sign-corrected yaw-rate slopes (measured / predicted, 1/s):")
    for sign, midpoint, observed, candidate, agrees in slope_checks:
        print(f"    sign={sign:+d} at {midpoint:.3f} rad: "
              f"{observed:+.4f} / {candidate:+.4f}; "
              f"sign gate={'PASS' if agrees else 'FAIL'}")
    print("  decision: " + ("ACCEPT as a steady-state slip-response candidate only"
                          if passed else
                          "REJECT front-slip-proxy-only force law"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_4mps", type=Path)
    parser.add_argument("mixed_65mps", type=Path)
    parser.add_argument("holdout_45mps", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.source_4mps, args.mixed_65mps,
                         args.holdout_45mps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
