#!/usr/bin/env python3
"""Leave-one-repetition-out 6.5 m/s nonlinear yaw and position replay.

The lateral-acceleration/steering curve is trained on repetitions 1--2 from
two independent transition captures plus bridge and wide-angle captures.
Repetition 3 is held out throughout. Speed and physical steering are measured
inputs; this is a conditional lateral-model test, not a full vehicle rollout.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools import evaluate_open_plane_demand_rollout as rollout
from tools import evaluate_open_plane_yaw_spline as spline
from tools.fit_open_plane_tire_model import _load_samples


SPEED_TARGET_MPS = 6.5
MIN_STEERING_RAD = 0.125
MAX_STEERING_RAD = 0.50
MAX_YAW_RMSE_RPS = 0.05
MIN_JACOBIAN_MATCH = 0.90
MIN_RESOLVED_SLOPE_RPS_PER_RAD = 0.10


def _sequences(paths: list[Path], repetitions: set[int]) -> list[list]:
    grouped: dict[tuple[Path, str], list] = defaultdict(list)
    for path in paths:
        for sample in _load_samples(path):
            if sample.repetition in repetitions:
                grouped[(path, sample.phase)].append(sample)
    return [sorted(rows, key=lambda row: row.time_s)
            for rows in grouped.values() if len(rows) >= 8]


def _fit_acceleration_curves(training_sequences: list[list]
                             ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    values: dict[int, dict[float, list[float]]] = {
        -1: defaultdict(list), 1: defaultdict(list)}
    for rows in training_sequences:
        sign = 1 if statistics.median(row.steering_rad for row in rows) > 0 else -1
        angle = round(statistics.median(abs(row.steering_rad) for row in rows), 4)
        if not MIN_STEERING_RAD <= angle <= MAX_STEERING_RAD:
            continue
        lateral_acceleration = statistics.median(
            sign * abs(row.u_mps) * row.yaw_rate_rps for row in rows)
        if not math.isfinite(lateral_acceleration):
            raise ValueError(f"nonfinite response at steering {angle:.4f}")
        values[sign][angle].append(lateral_acceleration)

    curves = {}
    for sign, by_angle in values.items():
        angles = sorted(by_angle)
        response = [statistics.mean(by_angle[angle]) for angle in angles]
        if len(angles) < 12 or angles[0] > MIN_STEERING_RAD + 0.005 \
                or angles[-1] < MAX_STEERING_RAD - 0.005:
            raise ValueError(f"incomplete steering support for sign {sign:+d}: "
                             f"{angles}")
        curves[sign] = (np.asarray(angles, dtype=float),
                        np.asarray(response, dtype=float))
    return curves


def _model_acceleration(curves, sign: int, steering: float) -> float:
    angles, values = curves[sign]
    acceleration, _ = spline._pchip_value_and_slope(
        angles, values, abs(steering))
    return acceleration


def _prepare(sequences: list[list], curves) -> list[dict]:
    prepared = []
    for rows in sequences:
        times, actual, candidate, configured, angles, signs = [], [], [], [], [], []
        lateral_velocity, speeds, longitudinal, accelerations = [], [], [], []

        def save_segment() -> None:
            if len(times) >= 8:
                prepared.append({
                    "time": times.copy(), "actual": actual.copy(),
                    "demand": candidate.copy(), "configured": configured.copy(),
                    "vy": lateral_velocity.copy(), "speed": speeds.copy(),
                    "u": longitudinal.copy(), "ay": accelerations.copy(),
                    "angle": angles.copy(), "sign": signs.copy(),
                })
            for values in (times, actual, candidate, configured,
                           lateral_velocity, speeds, longitudinal, accelerations,
                           angles, signs):
                values.clear()

        for row in rows:
            speed = abs(row.u_mps)
            steering = row.steering_rad
            angle = abs(steering)
            if not 5.5 <= speed <= 7.5 or not MIN_STEERING_RAD <= angle <= MAX_STEERING_RAD:
                if times and row.time_s - times[-1] > 0.10:
                    save_segment()
                continue
            sign = 1 if steering > 0 else -1
            try:
                acceleration = _model_acceleration(curves, sign, steering)
            except ValueError:
                if times and row.time_s - times[-1] > 0.10:
                    save_segment()
                continue

            gain = 2.95 - 35.6 * min(max(angle - 0.41, 0.0), 0.05)
            times.append(row.time_s)
            actual.append(row.yaw_rate_rps)
            candidate.append(sign * acceleration / speed)
            configured.append(speed * math.tan(steering) * gain)
            angles.append(angle)
            signs.append(sign)
            lateral_velocity.append(row.vy_mps)
            speeds.append(speed)
            longitudinal.append(row.u_mps)
            accelerations.append(sign * acceleration)
        save_segment()
    return prepared


def _jacobian_agreement(sequences: list[dict]) -> tuple[int, int]:
    grouped: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    for sequence in sequences:
        if not sequence["time"]:
            continue
        # Each excitation phase holds one signed steering target.
        sign = int(statistics.median(sequence["sign"]))
        angle = round(statistics.median(sequence["angle"]), 4)
        actual_oriented = statistics.mean(sign * value
                                          for value in sequence["actual"])
        predicted_oriented = statistics.mean(
            sign * value for value in sequence["demand"])
        grouped[(sign, round(angle, 4))].append(
            (actual_oriented, predicted_oriented))

    matches = count = 0
    for sign in (-1, 1):
        angles = sorted(angle for row_sign, angle in grouped if row_sign == sign)
        for left, right in zip(angles, angles[1:]):
            left_actual = statistics.mean(value[0] for value in grouped[(sign, left)])
            right_actual = statistics.mean(value[0] for value in grouped[(sign, right)])
            left_predicted = statistics.mean(value[1] for value in grouped[(sign, left)])
            right_predicted = statistics.mean(value[1] for value in grouped[(sign, right)])
            measured_slope = (right_actual - left_actual) / (right - left)
            predicted_slope = (right_predicted - left_predicted) / (right - left)
            if abs(measured_slope) >= MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                count += 1
                matches += measured_slope * predicted_slope > 0.0
    return matches, count


def evaluate(transition_paths: list[Path], bridge_path: Path,
             support_path: Path) -> bool:
    paths = [*transition_paths, bridge_path, support_path]
    for path in paths:
        with contextlib.redirect_stdout(io.StringIO()):
            if analysis.analyze(path) != 0:
                raise ValueError(f"data-quality analysis failed: {path}")

    training_sequences = _sequences(paths, {1, 2})
    holdout_sequences = _sequences(paths, {3})
    curves = _fit_acceleration_curves(training_sequences)
    training = _prepare(training_sequences, curves)
    holdout = _prepare(holdout_sequences, curves)
    if len(training) < 80 or len(holdout) < 40:
        raise ValueError(f"incomplete sequences: train={len(training)}, "
                         f"holdout={len(holdout)}")

    tau = rollout._fit_tau(training)
    candidate = rollout._horizon_rmse(holdout, tau, "demand")
    baseline = rollout._horizon_rmse(
        holdout, rollout.BASELINE_TAU_S, "configured")
    candidate_position = rollout._position_horizon_rmse(
        holdout, tau, "demand")
    baseline_position = rollout._position_horizon_rmse(
        holdout, rollout.BASELINE_TAU_S, "configured")
    ratios = {horizon: candidate[horizon] / baseline[horizon]
              for horizon in rollout.HORIZONS_S}
    position_ratios = {
        horizon: candidate_position[horizon] / baseline_position[horizon]
        for horizon in rollout.HORIZONS_S
    }
    matches, count = _jacobian_agreement(holdout)
    jacobian_fraction = matches / count if count else 0.0
    yaw_pass = (sum(value <= 0.80 for value in ratios.values()) >= 3
                and all(value <= 1.10 for value in ratios.values())
                and max(candidate.values()) <= MAX_YAW_RMSE_RPS)
    position_pass = (sum(value <= 0.80 for value in position_ratios.values()) >= 3
                     and all(value <= 1.10 for value in position_ratios.values()))
    passed = (yaw_pass and position_pass and count >= 12
              and jacobian_fraction >= MIN_JACOBIAN_MATCH)

    print("training: repetitions 1--2 from two independent transition captures, "
          "plus bridge and wide-angle captures at 6.5 m/s")
    print("held-out: repetition 3 from all four captures; steering 0.125..0.50 rad")
    print(f"conditional sequences: train={len(training)}, holdout={len(holdout)}")
    print(f"fitted yaw-response time constant: {tau:.3f} s")
    print("recursive yaw-rate RMSE (candidate / configured MPC):")
    for horizon in rollout.HORIZONS_S:
        print(f"  {horizon * 1000:4.0f} ms: {candidate[horizon]:.4f} / "
              f"{baseline[horizon]:.4f} rad/s (ratio={ratios[horizon]:.3f})")
    print("recursive position RMSE (candidate yaw + held lateral velocity / MPC):")
    for horizon in rollout.HORIZONS_S:
        print(f"  {horizon * 1000:4.0f} ms: {candidate_position[horizon]:.4f} / "
              f"{baseline_position[horizon]:.4f} m "
              f"(ratio={position_ratios[horizon]:.3f})")
    print(f"held-out local yaw Jacobian signs: {matches}/{count} "
          f"({jacobian_fraction:.1%})")
    print("decision: " + ("PASS conditional model at 6.5 m/s; not yet a "
                          "cross-speed or complete vehicle validation" if passed
                          else "REJECT; do not promote to the MPC"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transition_run_1", type=Path)
    parser.add_argument("transition_run_2", type=Path)
    parser.add_argument("bridge_run", type=Path)
    parser.add_argument("wide_support_run", type=Path)
    args = parser.parse_args()
    try:
        return 0 if evaluate([args.transition_run_1, args.transition_run_2],
                             args.bridge_run, args.wide_support_run) else 1
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
