#!/usr/bin/env python3
"""Held-out recursive yaw-rate replay for the piecewise lateral response surface.

The surface is trained on 4.0/4.25/4.75/5.0 m/s repetitions 1--2. A single
first yaw-rate state initializes each phase; subsequent yaw-rate predictions
are recursive. Measured speed and steering are exogenous inputs, so this is a
conditional lateral-model check, not a full vehicle or MPC rollout.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

from tools import analyze_open_plane_dynamics as analysis
from tools import evaluate_open_plane_fine_speed_surface as surface
from tools.fit_open_plane_tire_model import _load_samples


HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
FIT_TAUS_S = (0.0, *(step / 1000.0 for step in range(5, 301, 5)))
BASELINE_TAU_S = 0.015


def _sample_sequences(paths: list[Path], repetitions: set[int]
                      ) -> list[list]:
    grouped: dict[tuple[Path, str], list] = defaultdict(list)
    for path in paths:
        for sample in _load_samples(path):
            if sample.repetition in repetitions:
                grouped[(path, sample.phase)].append(sample)
    return [sorted(samples, key=lambda sample: sample.time_s)
            for samples in grouped.values() if len(samples) >= 8]


def _prepare_sequences(
    sequences: list[list],
    steering_curves: dict[tuple[float, int, str], tuple],
    demand_curves: dict[int, dict[float, tuple]],
    demand_low: float,
    demand_high: float,
) -> list[dict]:
    prepared = []
    for samples in sequences:
        times, actual, demand_target = [], [], []
        configured_target, lateral_velocity, speed_values, longitudinal = [], [], [], []
        ay_values = []

        def save_segment() -> None:
            if len(times) >= 8:
                prepared.append({"time": times.copy(), "actual": actual.copy(),
                                 "demand": demand_target.copy(),
                                 "configured": configured_target.copy(),
                                 "vy": lateral_velocity.copy(),
                                 "speed": speed_values.copy(),
                                 "u": longitudinal.copy(),
                                 "ay": ay_values.copy()})
            times.clear()
            actual.clear()
            demand_target.clear()
            configured_target.clear()
            lateral_velocity.clear()
            speed_values.clear()
            longitudinal.clear()
            ay_values.clear()

        for sample in samples:
            speed = abs(sample.u_mps)
            steering = sample.steering_rad
            angle = abs(steering)
            if speed < 1.0 or not 0.15 <= angle <= 0.50:
                if times and sample.time_s - times[-1] > 0.10:
                    save_segment()
                continue
            sign = 1 if steering > 0.0 else -1
            try:
                oriented_yaw_rate = surface._hybrid_yaw_rate(
                    speed, steering, sign, steering_curves, demand_curves,
                    demand_low, demand_high)
            except ValueError:
                if times and sample.time_s - times[-1] > 0.10:
                    save_segment()
                continue
            if times and sample.time_s - times[-1] > 0.10:
                save_segment()
            times.append(sample.time_s)
            actual.append(sample.yaw_rate_rps)
            demand_target.append(sign * oriented_yaw_rate)
            configured_target.append(
                speed * math.tan(steering)
                * surface.CONFIGURED_GAIN_PER_M)
            lateral_velocity.append(sample.vy_mps)
            speed_values.append(speed)
            longitudinal.append(sample.u_mps)
            ay_values.append(sign * oriented_yaw_rate * speed)
        save_segment()
    return prepared


def _rollout(sequence: dict, response: str,
             tau_s: float | None) -> list[float]:
    prediction = [sequence["actual"][0]]
    for index in range(1, len(sequence["time"])):
        dt = sequence["time"][index] - sequence["time"][index - 1]
        if not 0.0 < dt <= 0.10:
            raise ValueError(f"invalid sample interval in replay: {dt:.4f}s")
        time_constant = (sequence["tau"][index]
                         if tau_s is None else tau_s)
        retention = (math.exp(-dt / time_constant)
                     if time_constant > 0.0 else 0.0)
        target = sequence[response][index]
        prediction.append(retention * prediction[-1]
                          + (1.0 - retention) * target)
    return prediction


def _fit_tau(training: list[dict]) -> float:
    best_tau, best_sse = None, math.inf
    for tau_s in FIT_TAUS_S:
        squared_error = 0.0
        count = 0
        for sequence in training:
            prediction = _rollout(sequence, "demand", tau_s)
            squared_error += sum(
                (actual - predicted) ** 2
                for actual, predicted in zip(sequence["actual"][1:],
                                             prediction[1:]))
            count += len(prediction) - 1
        mean_squared_error = squared_error / count if count else math.inf
        if mean_squared_error < best_sse:
            best_tau, best_sse = tau_s, mean_squared_error
    if best_tau is None or not math.isfinite(best_sse):
        raise ValueError("no finite training fit for response time constant")
    return best_tau


def _horizon_rmse(sequences: list[dict], tau_s: float | None,
                  response: str) -> dict[float, float]:
    errors: dict[float, list[float]] = {horizon: [] for horizon in HORIZONS_S}
    for sequence in sequences:
        prediction = _rollout(sequence, response, tau_s)
        start = sequence["time"][0]
        for horizon in HORIZONS_S:
            target_time = start + horizon
            index = min(range(len(sequence["time"])),
                        key=lambda item: abs(sequence["time"][item] - target_time))
            if abs(sequence["time"][index] - target_time) <= 0.04:
                errors[horizon].append(
                    sequence["actual"][index] - prediction[index])
    return {horizon: (math.sqrt(statistics.mean(value * value for value in rows))
                      if rows else math.inf)
            for horizon, rows in errors.items()}


def _lateral_velocity_horizon_rmse(
    sequences: list[dict], tau_s: float, integrate_force: bool,
) -> dict[float, float]:
    errors: dict[float, list[float]] = {horizon: [] for horizon in HORIZONS_S}
    for sequence in sequences:
        yaw_prediction = [sequence["actual"][0]]
        vy_prediction = [sequence["vy"][0]]
        for index in range(1, len(sequence["time"])):
            dt = sequence["time"][index] - sequence["time"][index - 1]
            retention = math.exp(-dt / tau_s) if tau_s > 0.0 else 0.0
            next_yaw = (retention * yaw_prediction[-1]
                        + (1.0 - retention) * sequence["demand"][index])
            if integrate_force:
                yaw_acceleration = (next_yaw - yaw_prediction[-1]) / dt
                mean_yaw = 0.5 * (yaw_prediction[-1] + next_yaw)
                mean_speed = 0.5 * (sequence["speed"][index - 1]
                                    + sequence["speed"][index])
                # Odom stores COM velocity while the model state uses the
                # rear-axle lateral velocity: v_rear_dot = Fy/m - u*r - a*r_dot.
                next_vy = vy_prediction[-1] + dt * (
                    sequence["ay"][index] - mean_speed * mean_yaw
                    - analysis.COM_X_M * yaw_acceleration)
            else:
                next_vy = vy_prediction[-1]
            yaw_prediction.append(next_yaw)
            vy_prediction.append(next_vy)
        start = sequence["time"][0]
        for horizon in HORIZONS_S:
            target_time = start + horizon
            index = min(range(len(sequence["time"])),
                        key=lambda item: abs(sequence["time"][item] - target_time))
            if abs(sequence["time"][index] - target_time) <= 0.04:
                errors[horizon].append(
                    sequence["vy"][index] - vy_prediction[index])
    return {horizon: (math.sqrt(statistics.mean(value * value for value in rows))
                      if rows else math.inf)
            for horizon, rows in errors.items()}


def _position_horizon_rmse(sequences: list[dict], tau_s: float | None,
                           response: str) -> dict[float, float]:
    errors: dict[float, list[float]] = {horizon: [] for horizon in HORIZONS_S}
    for sequence in sequences:
        yaw_prediction = _rollout(sequence, response, tau_s)
        ref_x = ref_y = ref_heading = 0.0
        pred_x = pred_y = pred_heading = 0.0
        ref_positions = [(ref_x, ref_y)]
        pred_positions = [(pred_x, pred_y)]
        for index in range(1, len(sequence["time"])):
            dt = sequence["time"][index] - sequence["time"][index - 1]
            ref_yaw = 0.5 * (sequence["actual"][index - 1]
                             + sequence["actual"][index])
            pred_yaw = 0.5 * (yaw_prediction[index - 1]
                              + yaw_prediction[index])
            speed = 0.5 * (sequence["u"][index - 1]
                           + sequence["u"][index])
            ref_vy = 0.5 * (sequence["vy"][index - 1]
                            + sequence["vy"][index])
            pred_vy = sequence["vy"][0]
            ref_mid_heading = ref_heading + 0.5 * dt * ref_yaw
            pred_mid_heading = pred_heading + 0.5 * dt * pred_yaw
            ref_x += dt * (speed * math.cos(ref_mid_heading)
                           - ref_vy * math.sin(ref_mid_heading))
            ref_y += dt * (speed * math.sin(ref_mid_heading)
                           + ref_vy * math.cos(ref_mid_heading))
            pred_x += dt * (speed * math.cos(pred_mid_heading)
                            - pred_vy * math.sin(pred_mid_heading))
            pred_y += dt * (speed * math.sin(pred_mid_heading)
                            + pred_vy * math.cos(pred_mid_heading))
            ref_heading += dt * ref_yaw
            pred_heading += dt * pred_yaw
            ref_positions.append((ref_x, ref_y))
            pred_positions.append((pred_x, pred_y))

        start = sequence["time"][0]
        for horizon in HORIZONS_S:
            target_time = start + horizon
            index = min(range(len(sequence["time"])),
                        key=lambda item: abs(sequence["time"][item] - target_time))
            if abs(sequence["time"][index] - target_time) <= 0.04:
                dx = ref_positions[index][0] - pred_positions[index][0]
                dy = ref_positions[index][1] - pred_positions[index][1]
                errors[horizon].append(dx * dx + dy * dy)
    return {horizon: (math.sqrt(statistics.mean(rows)) if rows else math.inf)
            for horizon, rows in errors.items()}


def evaluate(source_paths: dict[float, Path], holdout_path: Path,
             support_paths: dict[float, Path],
             holdout_support_path: Path) -> bool:
    if set(source_paths) != set(surface.SOURCE_SPEEDS_MPS):
        raise ValueError("one source bag is required for each identified speed")
    if set(support_paths) != set(surface.SOURCE_SPEEDS_MPS):
        raise ValueError("one endpoint-support bag is required for each source speed")

    for path in (*source_paths.values(), *support_paths.values(), holdout_path,
                 holdout_support_path):
        with contextlib.redirect_stdout(io.StringIO()):
            if analysis.analyze(path) != 0:
                raise ValueError(f"data-quality analysis failed for {path}")

    rows_by_speed = {}
    for speed in surface.SOURCE_SPEEDS_MPS:
        rows_by_speed[speed], _ = surface._validated_blocks(
            source_paths[speed], speed)
        rows_by_speed[speed].extend(surface._validated_support_blocks(
            support_paths[speed], speed))
    steering_curves = {
        (speed, sign, response): surface._source_curves(
            rows_by_speed[speed], speed, sign, response)
        for speed in surface.SOURCE_SPEEDS_MPS
        for sign in (-1, 1)
        for response in ("yaw_rate", "lateral_acceleration")
    }
    demand_curves = {
        sign: surface._demand_speed_curves(rows_by_speed, sign)
        for sign in (-1, 1)
    }
    demand_low = max(curves[speed][0][0]
                     for curves in demand_curves.values()
                     for speed in surface.SOURCE_SPEEDS_MPS)
    demand_high = min(curves[speed][0][-1]
                      for curves in demand_curves.values()
                      for speed in surface.SOURCE_SPEEDS_MPS)

    training_paths = [*source_paths.values(), *support_paths.values()]
    training = _prepare_sequences(
        _sample_sequences(training_paths, {1, 2}), steering_curves,
        demand_curves, demand_low, demand_high)
    holdout = _prepare_sequences(
        _sample_sequences([holdout_path, holdout_support_path], {1, 2, 3}),
        steering_curves, demand_curves, demand_low, demand_high)
    if len(training) < 100 or len(holdout) < 100:
        raise ValueError(f"incomplete recursive sequences: train={len(training)}, "
                         f"holdout={len(holdout)}")

    fitted_tau = _fit_tau(training)
    candidate = _horizon_rmse(holdout, fitted_tau, "demand")
    baseline = _horizon_rmse(holdout, BASELINE_TAU_S, "configured")
    candidate_vy = _lateral_velocity_horizon_rmse(holdout, fitted_tau, True)
    baseline_vy = _lateral_velocity_horizon_rmse(holdout, BASELINE_TAU_S, False)
    candidate_position = _position_horizon_rmse(holdout, fitted_tau, "demand")
    baseline_position = _position_horizon_rmse(
        holdout, BASELINE_TAU_S, "configured")
    relative_change = {
        horizon: (candidate[horizon] / baseline[horizon]
                  if baseline[horizon] > 0.0 else math.inf)
        for horizon in HORIZONS_S
    }
    useful_horizons = sum(ratio <= 0.80 for ratio in relative_change.values())
    no_large_regression = all(ratio <= 1.10 for ratio in relative_change.values())
    position_ratio = {
        horizon: (candidate_position[horizon] / baseline_position[horizon]
                  if baseline_position[horizon] > 0.0 else math.inf)
        for horizon in HORIZONS_S
    }
    useful_position_horizons = sum(
        ratio <= 0.80 for ratio in position_ratio.values())
    no_large_position_regression = all(
        ratio <= 1.10 for ratio in position_ratio.values())
    passed = (useful_horizons >= 3 and no_large_regression
              and useful_position_horizons >= 3
              and no_large_position_regression)

    print("training: source speeds 4.00, 4.25, 4.75, 5.00 m/s; repetitions 1--2")
    print(f"held-out: {holdout_path} plus {holdout_support_path}; "
          "4.50 m/s, all repetitions, steering 0.15..0.50 rad")
    print(f"conditional sequences: train={len(training)}, holdout={len(holdout)}")
    print(f"fitted yaw-response time constant: {fitted_tau:.3f} s; "
          f"configured comparison tau={BASELINE_TAU_S:.3f} s")
    print("recursive yaw-rate RMSE by horizon (candidate / current configured model):")
    for horizon in HORIZONS_S:
        print(f"  {horizon*1000:4.0f} ms: {candidate[horizon]:.4f} / "
              f"{baseline[horizon]:.4f} rad/s "
          f"(ratio={relative_change[horizon]:.3f})")
    print("recursive rear-axle lateral-velocity RMSE (candidate / current hold):")
    for horizon in HORIZONS_S:
        print(f"  {horizon*1000:4.0f} ms: {candidate_vy[horizon]:.4f} / "
              f"{baseline_vy[horizon]:.4f} m/s")
    print("recursive planar position RMSE (candidate yaw + held v / current model):")
    for horizon in HORIZONS_S:
        print(f"  {horizon*1000:4.0f} ms: {candidate_position[horizon]:.4f} / "
              f"{baseline_position[horizon]:.4f} m "
              f"(ratio={position_ratio[horizon]:.3f})")
    print("decision: " + ("PASS conditional yaw-rate recursion for full vehicle "
                          "rollout testing only" if passed else
                          "REJECT; do not promote to vehicle rollout or MPC"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_4mps", type=Path)
    parser.add_argument("source_425mps", type=Path)
    parser.add_argument("source_475mps", type=Path)
    parser.add_argument("source_5mps", type=Path)
    parser.add_argument("holdout_45mps", type=Path)
    parser.add_argument("--support-bags", nargs=4, type=Path,
                        metavar=("SUPPORT_4", "SUPPORT_425",
                                 "SUPPORT_475", "SUPPORT_5"), required=True)
    parser.add_argument("--holdout-support-bag", type=Path, required=True,
                        help="4.50 m/s high-angle holdout support capture")
    args = parser.parse_args()
    source_paths = dict(zip(surface.SOURCE_SPEEDS_MPS,
                            (args.source_4mps, args.source_425mps,
                             args.source_475mps, args.source_5mps)))
    support_paths = dict(zip(surface.SOURCE_SPEEDS_MPS, args.support_bags))
    try:
        return 0 if evaluate(source_paths, args.holdout_45mps,
                             support_paths,
                             args.holdout_support_bag) else 1
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
