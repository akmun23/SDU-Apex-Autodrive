#!/usr/bin/env python3
"""Validate a full-angle 4 m/s yaw surface in an MPC-like kinematic rollout.

Training uses only repetitions 1--2 of transient_fullsteer_4mps; repetition 3
is the untouched holdout. A shape-preserving cubic interpolates the measured
steady yaw-rate-per-speed knots from 0 through 0.50 rad. A first-order yaw
state and integer-sample input delay are fitted on training transients. The
rollout then propagates heading and XY with the MPC midpoint kinematics, using
recorded speed, lateral velocity and actual steering as conditional inputs.

Predeclared model gate: held-out steady yaw RMSE <=0.05 rad/s; finite local
sensitivities with matching sign wherever held-out finite differences exceed
0.05 s^-1; and >=20% yaw-RMSE improvement at three or more 25--750 ms horizons
with no horizon >10% worse. Position/heading rollout errors are reported as
additional stage-level evidence; no controller code is modified here.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for item in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

try:
    import evaluate_open_plane_transient_rollout as transient
    from tools import evaluate_open_plane_yaw_spline as yaw_spline
except ImportError as exc:  # pragma: no cover - ROS 2 / repo imports
    raise SystemExit(f"required offline-analysis modules unavailable: {exc}") from exc


HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
TRAIN_REPETITIONS = (1, 2)
HOLDOUT_REPETITION = 3
STEADY_START_S = 0.55
MIN_SENSITIVITY_S_INV = 0.05
TAUS_S = tuple(i / 1000.0 for i in range(5, 201, 5))
DELAYS = (0, 1, 2)
FULL_ANGLE_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25,
                     0.30, 0.35, 0.42, 0.46, 0.50)
TRANSITION_LEVELS = tuple(round(0.18 + 0.01 * index, 2) for index in range(11))


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _angle_sign(steering: float) -> tuple[float, float]:
    return abs(steering), (1.0 if steering >= 0.0 else -1.0)


def _training_curve(probes, angles: list[float]):
    by_angle: dict[float, list[float]] = {angle: [] for angle in angles}
    for probe in probes:
        if probe.repetition not in TRAIN_REPETITIONS:
            continue
        target_angle = round(abs(probe.commanded_steering_rad), 2)
        sign = 1.0 if probe.commanded_steering_rad > 0.0 else -1.0
        for sample in probe.samples:
            if sample.time_s < STEADY_START_S or sample.speed_mps <= 1.0:
                continue
            by_angle[target_angle].append(sign * sample.yaw_rate_rps / sample.speed_mps)
    if any(not values for values in by_angle.values()):
        raise ValueError("training repetitions do not cover every steering level")
    curve_x = np.asarray([0.0, *angles], dtype=float)
    curve_y = np.asarray([0.0, *(_median(by_angle[angle]) for angle in angles)],
                         dtype=float)
    slopes = yaw_spline._pchip_slopes(curve_x, curve_y)
    return tuple(map(float, curve_x)), tuple(map(float, curve_y)), tuple(map(float, slopes))


def _curve_value_slope(curve, query: float) -> tuple[float, float]:
    x, y, slopes = curve
    # Feedback can differ from a boundary command by a few 1e-4 rad even
    # when the phase-quality gate passes; tolerate that measurement residual
    # without extrapolating the identified surface.
    endpoint_tolerance = 0.002
    if query < x[0] - endpoint_tolerance or query > x[-1] + endpoint_tolerance:
        raise ValueError(f"steering {query:.4f} rad outside measured map [0,{x[-1]:.2f}]")
    query = min(max(query, x[0]), x[-1])
    index = min(max(bisect_right(x, query) - 1, 0), len(x) - 2)
    width = x[index + 1] - x[index]
    t = (query - x[index]) / width
    y0, y1 = y[index], y[index + 1]
    m0, m1 = slopes[index], slopes[index + 1]
    value = ((2*t**3 - 3*t**2 + 1)*y0 + (t**3 - 2*t**2 + t)*width*m0
             + (-2*t**3 + 3*t**2)*y1 + (t**3 - t**2)*width*m1)
    derivative = ((6*t**2 - 6*t)*y0/width + (3*t**2 - 4*t + 1)*m0
                  + (-6*t**2 + 6*t)*y1/width + (3*t**2 - 2*t)*m1)
    return value, derivative


def bisect_right(values: tuple[float, ...], target: float) -> int:
    low, high = 0, len(values)
    while low < high:
        middle = (low + high) // 2
        if target < values[middle]:
            high = middle
        else:
            low = middle + 1
    return low


def _target_yaw(speed: float, steering: float, curve) -> float:
    magnitude, sign = _angle_sign(steering)
    response_per_speed, _ = _curve_value_slope(curve, magnitude)
    return sign * speed * response_per_speed


def _configured_target(speed: float, steering: float) -> float:
    gain = 2.95 - 35.6 * min(max(abs(steering) - 0.41, 0.0), 0.05)
    return speed * math.tan(steering) * gain


def _fit_dynamics(probes, curve) -> tuple[float, int]:
    best = (math.inf, 0.015, 0)
    for tau in TAUS_S:
        for delay in DELAYS:
            squared = 0.0
            count = 0
            for probe in probes:
                if probe.repetition not in TRAIN_REPETITIONS:
                    continue
                state = probe.samples[0].yaw_rate_rps
                for index in range(1, len(probe.samples)):
                    previous, sample = probe.samples[index - 1:index + 1]
                    if sample.time_s > 0.75:
                        break
                    dt = sample.time_s - previous.time_s
                    input_index = index - delay
                    steering = (probe.initial_steering_rad if input_index < 0
                                else probe.samples[input_index].steering_rad)
                    target = _target_yaw(sample.speed_mps, steering, curve)
                    retention = math.exp(-dt / tau)
                    state = retention * state + (1.0 - retention) * target
                    squared += (state - sample.yaw_rate_rps) ** 2
                    count += 1
            score = squared / count if count else math.inf
            if score < best[0]:
                best = score, tau, delay
    return best[1], best[2]


def _rollout(probe, end_index: int, curve, tau: float, delay: int,
             configured: bool = False):
    yaw_rate = probe.samples[0].yaw_rate_rps
    x = probe.samples[0].pose_x_m
    y = probe.samples[0].pose_y_m
    heading = probe.samples[0].pose_yaw_rad
    for index in range(1, end_index + 1):
        previous, sample = probe.samples[index - 1:index + 1]
        dt = sample.time_s - previous.time_s
        if not 0.0 < dt <= 0.075:
            raise ValueError(f"bad odometry interval {dt:.4f}s")
        input_index = index - delay
        steering = (probe.initial_steering_rad if input_index < 0
                    else probe.samples[input_index].steering_rad)
        target = (_configured_target(sample.speed_mps, steering) if configured
                  else _target_yaw(sample.speed_mps, steering, curve))
        retention = math.exp(-dt / tau)
        yaw_next = retention * yaw_rate + (1.0 - retention) * target
        yaw_mid = 0.5 * (yaw_rate + yaw_next)
        heading_mid = heading + 0.5 * dt * yaw_mid
        x += dt * (sample.speed_mps * math.cos(heading_mid)
                   - sample.vy_mps * math.sin(heading_mid))
        y += dt * (sample.speed_mps * math.sin(heading_mid)
                   + sample.vy_mps * math.cos(heading_mid))
        heading += dt * yaw_mid
        yaw_rate = yaw_next
    return yaw_rate, x, y, heading


def _wrapped(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _horizon_score(probes, curve, tau, delay, configured=False):
    output = {}
    holdout = [probe for probe in probes if probe.repetition == HOLDOUT_REPETITION]
    for horizon in HORIZONS_S:
        yaw_errors: list[float] = []
        position_errors: list[float] = []
        heading_errors: list[float] = []
        for probe in holdout:
            index = min(range(len(probe.samples)),
                        key=lambda i: abs(probe.samples[i].time_s - horizon))
            if abs(probe.samples[index].time_s - horizon) > 0.035:
                continue
            r, x, y, heading = _rollout(
                probe, index, curve, tau, delay, configured=configured)
            actual = probe.samples[index]
            yaw_errors.append(r - actual.yaw_rate_rps)
            position_errors.append(math.hypot(x - actual.pose_x_m, y - actual.pose_y_m))
            heading_errors.append(_wrapped(heading - actual.pose_yaw_rad))
        output[horizon] = (
            math.sqrt(sum(e*e for e in yaw_errors) / len(yaw_errors)),
            math.sqrt(sum(e*e for e in position_errors) / len(position_errors)),
            math.sqrt(sum(e*e for e in heading_errors) / len(heading_errors)),
            len(yaw_errors))
    return output


def _steady_holdout_rmse(probes, curve) -> float:
    errors: list[float] = []
    for probe in probes:
        if probe.repetition != HOLDOUT_REPETITION:
            continue
        sign = 1.0 if probe.commanded_steering_rad > 0.0 else -1.0
        for sample in probe.samples:
            if sample.time_s < STEADY_START_S:
                continue
            prediction = _target_yaw(sample.speed_mps, sample.steering_rad, curve)
            errors.append(prediction - sample.yaw_rate_rps)
    return math.sqrt(sum(error*error for error in errors) / len(errors))


def _sensitivity_check(probes, curve, angles: list[float]) -> tuple[bool, list[tuple[float, float, float]]]:
    holdout_means: dict[tuple[int, float], list[float]] = {}
    for probe in probes:
        if probe.repetition != HOLDOUT_REPETITION:
            continue
        sign = 1 if probe.commanded_steering_rad > 0.0 else -1
        angle = round(abs(probe.commanded_steering_rad), 2)
        steady = [sign * sample.yaw_rate_rps for sample in probe.samples
                  if sample.time_s >= STEADY_START_S]
        holdout_means.setdefault((sign, angle), []).append(statistics.median(steady))

    checks = []
    passed = True
    for left, right in zip(angles, angles[1:]):
        midpoint = 0.5 * (left + right)
        _, model_slope_per_speed = _curve_value_slope(curve, midpoint)
        model_slope = 4.0 * model_slope_per_speed
        for sign in (-1, 1):
            left_value = statistics.mean(holdout_means[(sign, left)])
            right_value = statistics.mean(holdout_means[(sign, right)])
            observed_slope = (right_value - left_value) / (right - left)
            agrees = (abs(observed_slope) < MIN_SENSITIVITY_S_INV
                      or model_slope * observed_slope > 0.0)
            passed &= agrees and math.isfinite(model_slope)
            checks.append((midpoint, observed_slope, model_slope))
    return passed, checks


def evaluate(path: Path) -> bool:
    probes, quality = transient.load(path)
    angles = sorted({round(abs(probe.commanded_steering_rad), 2) for probe in probes})
    if tuple(angles) not in (FULL_ANGLE_LEVELS, TRANSITION_LEVELS):
        raise ValueError(f"unexpected steering grid in holdout data: {tuple(angles)}")
    expected_per_repetition = 2 * (2 * len(angles) - 1)
    expected_valid_steps = 3 * expected_per_repetition
    expected = {1: expected_per_repetition, 2: expected_per_repetition,
                3: expected_per_repetition, 4: 0}
    if (quality["valid_steps"] != expected_valid_steps or quality["matched_starts"] != 6
            or quality["collisions"] != (0, 0) or quality["timing_faults"] != 0
            or quality["aborted"] or quality["probe_counts"] != expected):
        raise ValueError(f"capture failed predeclared data gate: {quality}")

    curve = _training_curve(probes, angles)
    tau, delay = _fit_dynamics(probes, curve)
    steady_rmse = _steady_holdout_rmse(probes, curve)
    sensitivity_pass, sensitivities = _sensitivity_check(probes, curve, angles)
    baseline = _horizon_score(probes, curve, 0.015, 0, configured=True)
    candidate = _horizon_score(probes, curve, tau, delay)

    yaw_gains: list[float] = []
    position_gains: list[float] = []
    heading_gains: list[float] = []
    print(f"bag: {path}")
    print(f"capture PASS: {expected_valid_steps}/{expected_valid_steps} valid steps; "
          "6/6 matched starts; "
          "zero collisions/timing faults; repetitions 1--2 train, 3 holdout")
    print("training steady yaw-rate-per-speed knots (1/m):")
    for angle, value in zip(curve[0][1:], curve[1][1:]):
        print(f"  |steer|={angle:.2f}: {value:.5f}")
    print(f"fitted yaw state: tau={tau:.3f}s, input delay={delay*25}ms")
    print(f"holdout steady yaw RMSE={steady_rmse:.5f} rad/s "
          f"(gate <=0.05); sensitivity sign={'PASS' if sensitivity_pass else 'FAIL'}")
    for midpoint, observed, model in sensitivities:
        print(f"  interval midpoint={midpoint:.3f}: held-out d(sign*r)/d|delta|="
              f"{observed:+.4f}, model derivative={model:+.4f} s^-1")

    print("horizon_ms yaw_RMSE_config/candidate(rad/s) improvement "
          "position_RMSE_config/candidate(m) heading_RMSE_config/candidate(rad)")
    for horizon in HORIZONS_S:
        base_r, base_xy, base_psi, n = baseline[horizon]
        fit_r, fit_xy, fit_psi, fit_n = candidate[horizon]
        if n != expected_per_repetition or fit_n != expected_per_repetition:
            raise ValueError(f"incomplete horizon {horizon:.3f}s: {n}/{fit_n}")
        yaw_gain = 1.0 - fit_r / base_r if base_r > 0.0 else 0.0
        xy_gain = 1.0 - fit_xy / base_xy if base_xy > 0.0 else 0.0
        psi_gain = 1.0 - fit_psi / base_psi if base_psi > 0.0 else 0.0
        yaw_gains.append(yaw_gain)
        position_gains.append(xy_gain)
        heading_gains.append(psi_gain)
        print(f"{horizon*1000:3.0f} {base_r:.5f}/{fit_r:.5f} {yaw_gain:+.1%} "
              f"{base_xy:.5f}/{fit_xy:.5f} {base_psi:.5f}/{fit_psi:.5f}")

    yaw_horizon_pass = (sum(x >= 0.20 for x in yaw_gains) >= 3
                        and min(yaw_gains) >= -0.10)
    map_pass = steady_rmse <= 0.05 and sensitivity_pass
    finite = all(math.isfinite(value) for value in (*curve[1], *curve[2], tau))
    print(f"acceptance: steady_map={'PASS' if map_pass else 'FAIL'}, "
          f"yaw_horizon={'PASS' if yaw_horizon_pass else 'FAIL'}, finite={finite}")
    return map_pass and yaw_horizon_pass and finite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
