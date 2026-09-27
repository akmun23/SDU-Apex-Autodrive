#!/usr/bin/env python3
"""Cross-speed held-out yaw rollout for the measured high-angle response.

Training uses repetitions 1--2 at 4.0, 4.25, 4.75 and 6.5 m/s; the complete
5.0 m/s high-angle support run is held out. Each speed has its own measured
lateral-acceleration-versus-steering curve, blended shape-preservingly in
speed. Recorded speed and physical steering remain conditional inputs.
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


SOURCE_SPEEDS_MPS = (4.0, 4.25, 4.75, 6.5)
MIN_STEERING_RAD = 0.25
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


def _fit_curves(training_sequences: list[list]):
    values: dict[tuple[float, int], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for rows in training_sequences:
        speed = statistics.median(row.target_speed_mps for row in rows)
        sign = 1 if statistics.median(row.steering_rad for row in rows) > 0 else -1
        angle = round(statistics.median(abs(row.steering_rad) for row in rows), 4)
        if not MIN_STEERING_RAD - 0.005 <= angle <= MAX_STEERING_RAD + 0.005:
            continue
        response = statistics.median(
            sign * abs(row.u_mps) * row.yaw_rate_rps for row in rows)
        if not math.isfinite(response):
            raise ValueError(f"nonfinite response at {speed:.2f} m/s, "
                             f"{angle:.4f} rad")
        values[(speed, sign)][angle].append(response)

    curves = {}
    for speed in SOURCE_SPEEDS_MPS:
        for sign in (-1, 1):
            by_angle = values[(speed, sign)]
            angles = sorted(by_angle)
            if (len(angles) < 6 or angles[0] > MIN_STEERING_RAD + 0.005
                    or angles[-1] < MAX_STEERING_RAD - 0.005):
                raise ValueError(f"incomplete {speed:.2f} m/s, sign {sign:+d} "
                                 f"high-angle curve: {angles}")
            response = [statistics.mean(by_angle[angle]) for angle in angles]
            curves[(speed, sign)] = (np.asarray(angles),
                                     np.asarray(response, dtype=float))
    return curves


def _model_acceleration(curves, speed: float, sign: int,
                        steering: float) -> float:
    source_responses = []
    for source_speed in SOURCE_SPEEDS_MPS:
        angles, values = curves[(source_speed, sign)]
        query_angle = abs(steering)
        if query_angle < angles[0] and angles[0] - query_angle <= 0.005:
            query_angle = float(angles[0])
        elif query_angle > angles[-1] and query_angle - angles[-1] <= 0.005:
            query_angle = float(angles[-1])
        response, _ = spline._pchip_value_and_slope(
            angles, values, query_angle)
        source_responses.append(response)
    acceleration, _ = spline._pchip_value_and_slope(
        np.asarray(SOURCE_SPEEDS_MPS), np.asarray(source_responses), speed)
    return acceleration


def _prepare(sequences: list[list], curves) -> list[dict]:
    prepared = []
    for rows in sequences:
        times, actual, candidate, configured = [], [], [], []
        lateral_velocity, speeds, longitudinal, accelerations = [], [], [], []
        angles, signs = [], []

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
            if times and row.time_s - times[-1] > 0.075:
                save_segment()
            speed = abs(row.u_mps)
            steering = row.steering_rad
            angle = abs(steering)
            if (not SOURCE_SPEEDS_MPS[0] - 0.20 <= speed
                    <= SOURCE_SPEEDS_MPS[-1] + 0.20
                    or not MIN_STEERING_RAD - 0.005 <= angle
                    <= MAX_STEERING_RAD + 0.005):
                if times and row.time_s - times[-1] > 0.10:
                    save_segment()
                continue
            sign = 1 if steering > 0 else -1
            try:
                acceleration = _model_acceleration(
                    curves, min(max(speed, SOURCE_SPEEDS_MPS[0]),
                                SOURCE_SPEEDS_MPS[-1]), sign, steering)
            except ValueError:
                if times and row.time_s - times[-1] > 0.10:
                    save_segment()
                continue

            gain = 2.95 - 35.6 * min(max(angle - 0.41, 0.0), 0.05)
            times.append(row.time_s)
            actual.append(row.yaw_rate_rps)
            candidate.append(sign * acceleration / speed)
            configured.append(speed * math.tan(steering) * gain)
            lateral_velocity.append(row.vy_mps)
            speeds.append(speed)
            longitudinal.append(row.u_mps)
            accelerations.append(sign * acceleration)
            angles.append(angle)
            signs.append(sign)
        save_segment()
    return prepared


def _jacobian_agreement(sequences: list[dict]) -> tuple[int, int]:
    grouped: dict[tuple[int, float], list[tuple[float, float]]] = defaultdict(list)
    for sequence in sequences:
        if not sequence["time"]:
            continue
        sign = int(statistics.median(sequence["sign"]))
        angle = round(statistics.median(sequence["angle"]), 4)
        actual = statistics.mean(sign * value for value in sequence["actual"])
        predicted = statistics.mean(sign * value for value in sequence["demand"])
        grouped[(sign, angle)].append((actual, predicted))

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


def evaluate(source_paths: dict[float, list[Path]], holdout_paths: list[Path],
             holdout_speed: float) -> bool:
    if set(source_paths) != set(SOURCE_SPEEDS_MPS):
        raise ValueError(f"source captures must cover {SOURCE_SPEEDS_MPS}")
    if not holdout_paths:
        raise ValueError("at least one holdout bag is required")
    all_paths = [path for paths in source_paths.values() for path in paths]
    all_paths.extend(holdout_paths)
    for path in all_paths:
        with contextlib.redirect_stdout(io.StringIO()):
            if analysis.analyze(path) != 0:
                raise ValueError(f"data-quality analysis failed: {path}")

    training_paths = [path for paths in source_paths.values() for path in paths]
    training_sequences = _sequences(training_paths, {1, 2})
    holdout_sequences = _sequences(holdout_paths, {1, 2, 3})
    curves = _fit_curves(training_sequences)
    training = _prepare(training_sequences, curves)
    holdout = _prepare(holdout_sequences, curves)
    if len(training) < 40 or len(holdout) < 30:
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
    passed = (yaw_pass and position_pass and count >= 8
              and jacobian_fraction >= MIN_JACOBIAN_MATCH)

    print("training speeds: 4.00, 4.25, 4.75, 6.50 m/s; repetitions 1--2")
    print(f"held-out: {', '.join(str(path) for path in holdout_paths)}; "
          f"{holdout_speed:.2f} m/s, repetitions 1--3, "
          "steering 0.25..0.50 rad")
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
    print("decision: " + (f"PASS high-angle interpolation from 4.0--6.5 m/s "
                          f"for {holdout_speed:.2f} m/s rollout only" if passed else
                          "REJECT; do not promote the cross-speed map"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bag", nargs=2, action="append", required=True,
                        metavar=("SPEED_MPS", "BAG"),
                        help="repeat for each training capture; include all bags "
                             "for the 6.5 m/s knot")
    parser.add_argument("--holdout-bag", type=Path, action="append", required=True,
                        help="repeat to combine transition and wide-angle bags")
    parser.add_argument("--holdout-speed-mps", type=float, default=5.0,
                        help="holdout speed (default: 5.0 m/s)")
    args = parser.parse_args()
    source_paths: dict[float, list[Path]] = defaultdict(list)
    try:
        for speed_text, bag_text in args.source_bag:
            speed = float(speed_text)
            if speed not in SOURCE_SPEEDS_MPS:
                raise ValueError(f"unsupported source speed: {speed_text}")
            source_paths[speed].append(Path(bag_text))
        return 0 if evaluate(dict(source_paths), args.holdout_bag,
                             args.holdout_speed_mps) else 1
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
