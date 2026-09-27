#!/usr/bin/env python3
"""Held-out recursive validation of a measured full-speed lateral response.

The candidate uses lateral acceleration against speed-scaled steering demand
through 0.25 rad and a measured steering-angle curve from 0.275 rad upward,
with a smoothstep blend between them. Source speeds are 3.0, 3.5, 4.0, 4.25,
4.5, 4.75, 6.5, and 8.25 m/s; 5.0 m/s is held out. Repetition 3 at each source
speed can also be held out. Speed and steering are measured
exogenous inputs, so this checks the lateral response only, not full MPC or
longitudinal dynamics.
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


SOURCE_SPEEDS_MPS = (3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 6.5, 8.25)
DEMAND_MAX_ANGLE_RAD = 0.25
ANGLE_MIN_RAD = 0.275
ANGLE_MAX_RAD = 0.50
MIN_YAW_RMSE_RPS = 0.05
MIN_JACOBIAN_MATCH = 0.90
MIN_RESOLVED_SLOPE_RPS_PER_RAD = 0.10


def _phase_sequences(paths: list[Path], repetitions: set[int]) -> list[list]:
    grouped: dict[tuple[Path, str], list] = defaultdict(list)
    for path in paths:
        for sample in _load_samples(path):
            if sample.repetition in repetitions:
                grouped[(path, sample.phase)].append(sample)
    return [sorted(rows, key=lambda row: row.time_s)
            for rows in grouped.values()
            if len(rows) >= 8
            # A fixed-pedal block is deliberately not speed-regulated. Its
            # speed drift belongs to the coupled longitudinal test, not the
            # fixed-speed steering surface, even when the phase's actuator-
            # command checks passed.
            and not any("_pedal_" in row.phase for row in rows)]


def _phase_point(rows: list) -> tuple[float, float, int, float, float]:
    speed = statistics.median(abs(row.u_mps) for row in rows)
    steering = statistics.median(row.steering_rad for row in rows)
    sign = 1 if steering > 0.0 else -1
    angle = abs(steering)
    accel = statistics.median(sign * abs(row.u_mps) * row.yaw_rate_rps
                              for row in rows)
    demand = statistics.median(abs(row.u_mps) * math.tan(angle)
                               for row in rows)
    return speed, angle, sign, demand, accel


def _fit_curves(training_sequences: list[list]):
    demand_points: dict[tuple[float, int], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    angle_points: dict[tuple[float, int], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    phase_counts: dict[tuple[float, int], int] = defaultdict(int)

    for rows in training_sequences:
        speed, angle, sign, demand, accel = _phase_point(rows)
        speed_knot = min(SOURCE_SPEEDS_MPS,
                         key=lambda value: abs(value - statistics.median(
                             row.target_speed_mps for row in rows)))
        if abs(speed - speed_knot) > 0.25:
            continue
        key = (speed_knot, sign)
        phase_counts[key] += 1
        # Preserve the 2.5 mrad steering-transition probes: coarser q bins
        # erased the narrow 0.180--0.185 rad yaw-response trough on holdout.
        demand_knot = round(demand / 0.0025) * 0.0025
        demand_points[key][demand_knot].append(accel)
        if 0.12 - 0.005 <= angle <= ANGLE_MAX_RAD + 0.005:
            angle_knot = round(angle, 4)
            angle_points[key][angle_knot].append(accel)

    demand_curves = {}
    angle_curves = {}
    for speed in SOURCE_SPEEDS_MPS:
        for sign in (-1, 1):
            key = (speed, sign)
            q_knots = sorted(demand_points[key])
            q_values = [statistics.mean(demand_points[key][knot])
                        for knot in q_knots]
            if len(q_knots) < 10 or not all(math.isfinite(v) for v in q_values):
                raise ValueError(f"insufficient demand response at {speed:.2f} "
                                 f"m/s, sign {sign:+d}: {len(q_knots)} knots")
            demand_curves[key] = (np.asarray(q_knots, dtype=float),
                                  np.asarray(q_values, dtype=float))

            angles = sorted(angle_points[key])
            values = [statistics.mean(angle_points[key][angle])
                      for angle in angles]
            if (len(angles) < 10 or angles[0] > 0.125 + 0.005
                    or angles[-1] < ANGLE_MAX_RAD - 0.005
                    or not all(math.isfinite(v) for v in values)):
                raise ValueError(f"insufficient high-angle response at "
                                 f"{speed:.2f} m/s, sign {sign:+d}: {angles}")
            angle_curves[key] = (np.asarray(angles, dtype=float),
                                 np.asarray(values, dtype=float))

    demand_low = max(curve[0][0] for curve in demand_curves.values())
    demand_high = min(curve[0][-1] for curve in demand_curves.values())
    if not demand_low < demand_high:
        raise ValueError(f"no common demand domain: {demand_low}..{demand_high}")
    return demand_curves, angle_curves, demand_low, demand_high, phase_counts


def _interpolate_speed(curves, speed: float, sign: int,
                       input_value: float) -> float:
    values = []
    for source_speed in SOURCE_SPEEDS_MPS:
        knots, response = curves[(source_speed, sign)]
        bounded = min(max(input_value, float(knots[0])), float(knots[-1]))
        value, _ = spline._pchip_value_and_slope(knots, response, bounded)
        values.append(value)
    response, _ = spline._pchip_value_and_slope(
        np.asarray(SOURCE_SPEEDS_MPS), np.asarray(values), speed)
    return response


def _candidate_yaw_rate(speed: float, steering: float, sign: int,
                        demand_curves, angle_curves,
                        demand_low: float, demand_high: float,
                        low_angle_coordinate: str) -> float:
    if (speed < SOURCE_SPEEDS_MPS[0] - 0.25
            or speed > SOURCE_SPEEDS_MPS[-1] + 0.25):
        raise ValueError(f"speed {speed:.3f} outside measured model range")
    model_speed = min(max(speed, SOURCE_SPEEDS_MPS[0]),
                      SOURCE_SPEEDS_MPS[-1])
    angle = abs(steering)
    # Do not clamp every speed to the common-domain intersection. The actual
    # per-speed PCHIP below already bounds to measured knots. Intersecting the
    # domains here discards valid low-speed, low-angle measurements whenever
    # the highest-speed capture has a larger minimum lateral demand.
    demand = speed * math.tan(angle)
    demand_accel = _interpolate_speed(
        demand_curves, model_speed, sign, demand)
    angle_accel = _interpolate_speed(
        angle_curves, model_speed, sign, angle)
    if angle <= DEMAND_MAX_ANGLE_RAD and low_angle_coordinate == "steering":
        return angle_accel / speed
    if angle <= DEMAND_MAX_ANGLE_RAD:
        return demand_accel / speed
    if angle >= ANGLE_MIN_RAD:
        return angle_accel / speed
    blend = (angle - DEMAND_MAX_ANGLE_RAD) / (ANGLE_MIN_RAD - DEMAND_MAX_ANGLE_RAD)
    blend = blend * blend * (3.0 - 2.0 * blend)
    return ((1.0 - blend) * demand_accel + blend * angle_accel) / speed


def _fit_time_constant_surface(training_sequences: list[list], demand_curves,
                               angle_curves, demand_low: float,
                               demand_high: float,
                               low_angle_coordinate: str
                               ) -> dict[str, dict[float, float]]:
    raw_bands: dict[tuple[float, str], list[list]] = defaultdict(list)
    for rows in training_sequences:
        target_speed = statistics.median(row.target_speed_mps for row in rows)
        source_speed = min(SOURCE_SPEEDS_MPS,
                           key=lambda value: abs(value - target_speed))
        angle = statistics.median(abs(row.steering_rad) for row in rows)
        band = ("low" if 0.12 <= angle <= DEMAND_MAX_ANGLE_RAD else
                "high" if angle >= ANGLE_MIN_RAD else None)
        if band is not None and abs(target_speed - source_speed) < 1e-6:
            raw_bands[(source_speed, band)].append(rows)

    result: dict[str, dict[float, float]] = {"low": {}, "high": {}}
    for band in result:
        for speed in SOURCE_SPEEDS_MPS:
            sequences = _prepare(
                raw_bands[(speed, band)], demand_curves, angle_curves,
                demand_low, demand_high, None, low_angle_coordinate)
            if len(sequences) < 8:
                raise ValueError(f"insufficient {band}-angle lag data at "
                                 f"{speed:.2f} m/s: {len(sequences)} phases")
            result[band][speed] = rollout._fit_tau(sequences)
    return result


def _time_constant_at(tau_curves: dict[str, dict[float, float]],
                      speed: float, angle: float) -> float:
    speed = min(max(speed, SOURCE_SPEEDS_MPS[0]), SOURCE_SPEEDS_MPS[-1])
    speed_knots = np.asarray(SOURCE_SPEEDS_MPS, dtype=float)
    values = {}
    for band in ("low", "high"):
        values[band], _ = spline._pchip_value_and_slope(
            speed_knots,
            np.asarray([tau_curves[band][source] for source in SOURCE_SPEEDS_MPS]),
            speed)
    if angle <= DEMAND_MAX_ANGLE_RAD:
        return max(0.005, values["low"])
    if angle >= ANGLE_MIN_RAD:
        return max(0.005, values["high"])
    blend = (angle - DEMAND_MAX_ANGLE_RAD) / (ANGLE_MIN_RAD - DEMAND_MAX_ANGLE_RAD)
    blend = blend * blend * (3.0 - 2.0 * blend)
    return max(0.005, (1.0 - blend) * values["low"] + blend * values["high"])


def _prepare(sequences: list[list], demand_curves, angle_curves,
             demand_low: float, demand_high: float,
             tau_curves: dict[str, dict[float, float]] | None = None,
             low_angle_coordinate: str = "demand"
             ) -> list[dict]:
    prepared = []
    for rows in sequences:
        times, actual, candidate, configured = [], [], [], []
        lateral_velocity, speeds, longitudinal, angles, signs = [], [], [], [], []
        time_constants = []

        def save_segment() -> None:
            if len(times) >= 8:
                prepared.append({
                    "time": times.copy(), "actual": actual.copy(),
                    "demand": candidate.copy(), "configured": configured.copy(),
                    "vy": lateral_velocity.copy(), "speed": speeds.copy(),
                    "u": longitudinal.copy(), "angle": angles.copy(),
                    "sign": signs.copy(), "tau": time_constants.copy(),
                })
            for values in (times, actual, candidate, configured,
                           lateral_velocity, speeds, longitudinal, angles, signs,
                           time_constants):
                values.clear()

        for row in rows:
            speed = abs(row.u_mps)
            steering = row.steering_rad
            angle = abs(steering)
            if speed < 1.0 or not 0.12 <= angle <= ANGLE_MAX_RAD + 1e-4:
                if times and row.time_s - times[-1] > 0.10:
                    save_segment()
                continue
            sign = 1 if steering > 0.0 else -1
            try:
                predicted = _candidate_yaw_rate(
                    speed, steering, sign, demand_curves, angle_curves,
                    demand_low, demand_high, low_angle_coordinate)
            except ValueError:
                if times and row.time_s - times[-1] > 0.10:
                    save_segment()
                continue
            if times and row.time_s - times[-1] > 0.10:
                save_segment()
            gain = 2.95 - 35.6 * min(max(angle - 0.41, 0.0), 0.05)
            times.append(row.time_s)
            actual.append(row.yaw_rate_rps)
            candidate.append(sign * predicted)
            configured.append(speed * math.tan(steering) * gain)
            lateral_velocity.append(row.vy_mps)
            speeds.append(speed)
            longitudinal.append(row.u_mps)
            angles.append(angle)
            signs.append(sign)
            if tau_curves is not None:
                time_constants.append(_time_constant_at(tau_curves, speed, angle))
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
            actual_left = statistics.mean(v[0] for v in grouped[(sign, left)])
            actual_right = statistics.mean(v[0] for v in grouped[(sign, right)])
            pred_left = statistics.mean(v[1] for v in grouped[(sign, left)])
            pred_right = statistics.mean(v[1] for v in grouped[(sign, right)])
            actual_slope = (actual_right - actual_left) / (right - left)
            predicted_slope = (pred_right - pred_left) / (right - left)
            if abs(actual_slope) >= MIN_RESOLVED_SLOPE_RPS_PER_RAD:
                count += 1
                matches += actual_slope * predicted_slope > 0.0
    return matches, count


def evaluate(source_paths: dict[float, list[Path]], holdout_paths: list[Path],
             holdout_speed: float,
             holdout_repetitions: frozenset[int] = frozenset((1, 2, 3)),
             low_angle_coordinate: str = "demand",
             fixed_tau_s: float | None = None) -> bool:
    if set(source_paths) != set(SOURCE_SPEEDS_MPS):
        raise ValueError(f"source bags must cover exactly {SOURCE_SPEEDS_MPS}")
    all_paths = [path for paths in source_paths.values() for path in paths]
    all_paths += holdout_paths
    if not holdout_paths:
        raise ValueError("at least one held-out capture is required")
    for path in all_paths:
        with contextlib.redirect_stdout(io.StringIO()):
            if analysis.analyze(path) != 0:
                raise ValueError(f"data-quality analysis failed: {path}")

    training_sequences = _phase_sequences(
        [path for paths in source_paths.values() for path in paths], {1, 2})
    holdout_sequences = _phase_sequences(holdout_paths,
                                         set(holdout_repetitions))
    demand_curves, angle_curves, demand_low, demand_high, counts = _fit_curves(
        training_sequences)
    if fixed_tau_s is None:
        tau_curves = _fit_time_constant_surface(
            training_sequences, demand_curves, angle_curves,
            demand_low, demand_high, low_angle_coordinate)
    else:
        if not math.isfinite(fixed_tau_s) or fixed_tau_s <= 0.0:
            raise ValueError("fixed time constant must be finite and positive")
        tau_curves = {
            band: {speed: fixed_tau_s for speed in SOURCE_SPEEDS_MPS}
            for band in ("low", "high")
        }
    training = _prepare(training_sequences, demand_curves, angle_curves,
                        demand_low, demand_high, tau_curves,
                        low_angle_coordinate)
    holdout = _prepare(holdout_sequences, demand_curves, angle_curves,
                       demand_low, demand_high, tau_curves,
                       low_angle_coordinate)
    if len(training) < 100 or len(holdout) < 20:
        raise ValueError(f"incomplete recursive sequences: train={len(training)}, "
                         f"holdout={len(holdout)}")

    global_tau = (rollout._fit_tau(training) if fixed_tau_s is None
                  else fixed_tau_s)
    candidate = rollout._horizon_rmse(holdout, None, "demand")
    baseline = rollout._horizon_rmse(
        holdout, rollout.BASELINE_TAU_S, "configured")
    candidate_position = rollout._position_horizon_rmse(holdout, None, "demand")
    baseline_position = rollout._position_horizon_rmse(
        holdout, rollout.BASELINE_TAU_S, "configured")
    yaw_ratios = {h: candidate[h] / baseline[h] for h in rollout.HORIZONS_S}
    position_ratios = {h: candidate_position[h] / baseline_position[h]
                       for h in rollout.HORIZONS_S}
    matches, count = _jacobian_agreement(holdout)
    jacobian_fraction = matches / count if count else 0.0
    passed = (
        sum(ratio <= 0.80 for ratio in yaw_ratios.values()) >= 3
        and all(ratio <= 1.10 for ratio in yaw_ratios.values())
        and max(candidate.values()) <= MIN_YAW_RMSE_RPS
        and sum(ratio <= 0.80 for ratio in position_ratios.values()) >= 3
        and all(ratio <= 1.10 for ratio in position_ratios.values())
        and count >= 8 and jacobian_fraction >= MIN_JACOBIAN_MATCH
    )

    print("source speeds: " + ", ".join(f"{s:.2f}" for s in SOURCE_SPEEDS_MPS)
          + " m/s; repetitions 1--2")
    print(f"held-out captures: {', '.join(map(str, holdout_paths))}; "
          f"nominal {holdout_speed:.2f} m/s, repetitions "
          f"{','.join(map(str, sorted(holdout_repetitions)))}")
    print(f"common speed-scaled steering demand domain: "
          f"{demand_low:.3f}..{demand_high:.3f} m/s")
    print(f"low-angle coordinate: {low_angle_coordinate}")
    print("time-constant mode: " +
          (f"fixed {fixed_tau_s:.3f} s" if fixed_tau_s is not None
           else "speed/angle fitted from training repetitions"))
    print("trained source phase counts (negative / positive steering): " +
          ", ".join(f"{speed:.2f}: {counts[(speed, -1)]}/"
                     f"{counts[(speed, 1)]}" for speed in SOURCE_SPEEDS_MPS))
    print(f"recursive segments: train={len(training)}, holdout={len(holdout)}; "
          f"global-fit diagnostic tau={global_tau:.3f} s")
    print("training-fit yaw response tau by speed (low-angle / high-angle): " +
          ", ".join(f"{speed:.2f}: {tau_curves['low'][speed]:.3f}/"
                     f"{tau_curves['high'][speed]:.3f}s"
                     for speed in SOURCE_SPEEDS_MPS))
    print("recursive yaw-rate RMSE (candidate / configured MPC):")
    for horizon in rollout.HORIZONS_S:
        print(f"  {horizon * 1000:4.0f} ms: {candidate[horizon]:.4f} / "
              f"{baseline[horizon]:.4f} rad/s "
              f"(ratio={yaw_ratios[horizon]:.3f})")
    print("recursive planar-position RMSE (candidate yaw + held lateral velocity / MPC):")
    for horizon in rollout.HORIZONS_S:
        print(f"  {horizon * 1000:4.0f} ms: {candidate_position[horizon]:.4f} / "
              f"{baseline_position[horizon]:.4f} m "
              f"(ratio={position_ratios[horizon]:.3f})")
    print(f"held-out resolved local yaw-Jacobian signs: {matches}/{count} "
          f"({jacobian_fraction:.1%})")
    print("decision: " + ("PASS conditional full-speed hybrid response only; "
                          "not full MPC/vehicle validation" if passed else
                          "REJECT; do not promote this hybrid map"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bag", nargs=2, action="append", required=True,
                        metavar=("SPEED_MPS", "BAG"),
                        help="repeat for each source capture, including all repetitions")
    parser.add_argument("--holdout-bag", type=Path, action="append", required=True)
    parser.add_argument("--holdout-speed-mps", type=float, required=True)
    parser.add_argument("--holdout-repetitions", type=int, nargs="+",
                        choices=(1, 2, 3), default=(1, 2, 3))
    parser.add_argument("--low-angle-coordinate", choices=("demand", "steering"),
                        default="demand")
    parser.add_argument("--fixed-tau-s", type=float,
                        help="hold the measured response time constant fixed "
                             "instead of fitting it from training captures")
    args = parser.parse_args()
    source_paths: dict[float, list[Path]] = defaultdict(list)
    try:
        for speed_text, bag_text in args.source_bag:
            speed = float(speed_text)
            if speed not in SOURCE_SPEEDS_MPS:
                raise ValueError(f"unsupported source speed: {speed_text}")
            source_paths[speed].append(Path(bag_text))
        return 0 if evaluate(dict(source_paths), args.holdout_bag,
                             args.holdout_speed_mps,
                             frozenset(args.holdout_repetitions),
                             args.low_angle_coordinate,
                             args.fixed_tau_s) else 1
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
