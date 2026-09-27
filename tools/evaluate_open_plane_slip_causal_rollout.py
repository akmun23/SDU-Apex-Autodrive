#!/usr/bin/env python3
"""Test a causal front-tire-slip yaw model on held-out real-sim data.

The fitted tire-slip feature is computed from the model's previous predicted
rear-axle state and prior steering input. Repetitions 1--2 fit the model;
repetition 3 is untouched. Speed is a measured conditional input so this
isolates whether the nonlinear steering/slip response improves yaw prediction.
The lateral state is propagated with the independently validated odometry
kinematic law. This is an offline plant-model test, not a runtime change.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import evaluate_open_plane_fullsteer_rollout as yaw_model
import evaluate_open_plane_transient_rollout as transient


COM_X_M = 0.15532
LATERAL_VELOCITY_YAW_GAIN_M = 0.167
LATERAL_VELOCITY_SPEED_YAW_GAIN_S = -0.0063
LATERAL_VELOCITY_LIMIT_MPS = 0.35
MIN_TRAIN_SAMPLES = 500


def _front_slip(vx: float, vy: float, yaw_rate: float,
                steering: float) -> float:
    common = transient.common
    half_track = common.TRACK_WIDTH_M * 0.5
    left_angle, right_angle = common._ackermann_angles(steering)
    left = common._wheel_slip(
        vx, vy, yaw_rate, common.WHEELBASE_M, half_track, left_angle)
    right = common._wheel_slip(
        vx, vy, yaw_rate, common.WHEELBASE_M, -half_track, right_angle)
    if left is None or right is None:
        raise ValueError("front slip is undefined at this predicted speed")
    return 0.5 * (abs(left) + abs(right))


def _center_at(angle: float, centers: dict[float, float]) -> float:
    knots = sorted(centers)
    if angle <= knots[0]:
        return centers[knots[0]]
    if angle >= knots[-1]:
        return centers[knots[-1]]
    for left, right in zip(knots, knots[1:]):
        if left <= angle <= right:
            fraction = (angle - left) / (right - left)
            return centers[left] + fraction * (centers[right] - centers[left])
    raise AssertionError("unreachable slip-center interpolation")


def _fit_slip_effect(probes, curve, angles: list[float]):
    features: dict[float, list[float]] = {angle: [] for angle in angles}
    pairs: list[tuple[float, float, float]] = []
    for probe in probes:
        if probe.repetition not in yaw_model.TRAIN_REPETITIONS:
            continue
        angle = round(abs(probe.commanded_steering_rad), 2)
        sign = 1.0 if probe.commanded_steering_rad > 0.0 else -1.0
        for previous, current in zip(probe.samples, probe.samples[1:]):
            if current.time_s < yaw_model.STEADY_START_S:
                continue
            delta = current.steering_rad
            if current.speed_mps <= 1.0 or abs(delta) < 0.05:
                continue
            feature = _front_slip(
                previous.speed_mps, previous.vy_mps,
                previous.yaw_rate_rps, previous.steering_rad)
            response, _ = yaw_model._curve_value_slope(curve, abs(delta))
            base_gain = response / math.tan(abs(delta))
            observed_gain = (sign * current.yaw_rate_rps
                             / (current.speed_mps * math.tan(abs(delta))))
            features[angle].append(feature)
            pairs.append((angle, feature, observed_gain - base_gain))
    if len(pairs) < MIN_TRAIN_SAMPLES or any(not values for values in features.values()):
        raise ValueError(f"insufficient causal training pairs: {len(pairs)}")
    centers = {angle: statistics.median(values)
               for angle, values in features.items()}
    numerator = denominator = 0.0
    for angle, feature, residual_gain in pairs:
        centered = feature - centers[angle]
        numerator += centered * residual_gain
        denominator += centered * centered
    if denominator <= 1.0e-12:
        raise ValueError("prior-state front-slip feature has no training variation")
    beta = numerator / denominator
    return centers, beta, len(pairs)


def _target_yaw(speed: float, steering: float, curve,
                slip: float | None = None,
                centers: dict[float, float] | None = None,
                beta: float = 0.0) -> float:
    target = yaw_model._target_yaw(speed, steering, curve)
    if slip is None or centers is None or abs(steering) < 0.05:
        return target
    sign = 1.0 if steering > 0.0 else -1.0
    correction = (sign * speed * math.tan(abs(steering)) * beta
                  * (slip - _center_at(abs(steering), centers)))
    return target + correction


def _lateral_from_yaw(yaw_rate: float, speed: float) -> float:
    reference = yaw_rate * (
        LATERAL_VELOCITY_YAW_GAIN_M
        + LATERAL_VELOCITY_SPEED_YAW_GAIN_S * max(speed, 0.0))
    reference = max(-LATERAL_VELOCITY_LIMIT_MPS,
                    min(LATERAL_VELOCITY_LIMIT_MPS, reference))
    return reference - yaw_rate * COM_X_M


def _rollout(probe, end_index: int, curve, tau: float, delay: int,
             *, centers: dict[float, float] | None = None,
             beta: float = 0.0):
    samples = probe.samples
    yaw_rate = samples[0].yaw_rate_rps
    vy = samples[0].vy_mps
    x, y, heading = (samples[0].pose_x_m, samples[0].pose_y_m,
                     samples[0].pose_yaw_rad)
    for index in range(1, end_index + 1):
        previous, sample = samples[index - 1:index + 1]
        dt = sample.time_s - previous.time_s
        if not 0.0 < dt <= 0.075:
            raise ValueError(f"bad odometry interval {dt:.4f}s")
        state_slip = None
        if centers is not None:
            state_slip = _front_slip(
                previous.speed_mps, vy, yaw_rate, previous.steering_rad)
        input_index = index - delay
        steering = (probe.initial_steering_rad if input_index < 0
                    else samples[input_index].steering_rad)
        target = _target_yaw(sample.speed_mps, steering, curve,
                             state_slip, centers, beta)
        retention = math.exp(-dt / tau)
        yaw_next = retention * yaw_rate + (1.0 - retention) * target
        vy_next = _lateral_from_yaw(yaw_next, sample.speed_mps)
        yaw_mid = 0.5 * (yaw_rate + yaw_next)
        vy_mid = 0.5 * (vy + vy_next)
        heading_mid = heading + 0.5 * dt * yaw_mid
        x += dt * (sample.speed_mps * math.cos(heading_mid)
                   - vy_mid * math.sin(heading_mid))
        y += dt * (sample.speed_mps * math.sin(heading_mid)
                   + vy_mid * math.cos(heading_mid))
        heading += dt * yaw_mid
        yaw_rate, vy = yaw_next, vy_next
    return yaw_rate, x, y, heading


def _score(probes, curve, tau, delay, *, centers=None, beta=0.0):
    output = {}
    holdout = [probe for probe in probes
               if probe.repetition == yaw_model.HOLDOUT_REPETITION]
    for horizon in yaw_model.HORIZONS_S:
        yaw_errors, position_errors, heading_errors = [], [], []
        for probe in holdout:
            index = min(range(len(probe.samples)),
                        key=lambda i: abs(probe.samples[i].time_s - horizon))
            if abs(probe.samples[index].time_s - horizon) > 0.035:
                continue
            result = _rollout(probe, index, curve, tau, delay,
                              centers=centers, beta=beta)
            yaw, x, y, heading = result
            actual = probe.samples[index]
            yaw_errors.append(yaw - actual.yaw_rate_rps)
            position_errors.append(math.hypot(
                x - actual.pose_x_m, y - actual.pose_y_m))
            heading_errors.append(math.atan2(
                math.sin(heading - actual.pose_yaw_rad),
                math.cos(heading - actual.pose_yaw_rad)))
        output[horizon] = (
            math.sqrt(sum(value * value for value in yaw_errors) / len(yaw_errors)),
            math.sqrt(sum(value * value for value in position_errors) / len(position_errors)),
            math.sqrt(sum(value * value for value in heading_errors) / len(heading_errors)),
            len(yaw_errors))
    return output


def evaluate(path: Path) -> bool:
    probes, quality = transient.load(path)
    angles = sorted({round(abs(probe.commanded_steering_rad), 2)
                     for probe in probes})
    expected_per_rep = 2 * (2 * len(angles) - 1)
    expected = {1: expected_per_rep, 2: expected_per_rep,
                3: expected_per_rep, 4: 0}
    if (quality["valid_steps"] != 3 * expected_per_rep
            or quality["matched_starts"] != 6
            or quality["collisions"] != (0, 0)
            or quality["timing_faults"] != 0 or quality["aborted"]
            or quality["probe_counts"] != expected):
        raise ValueError(f"capture failed quality gate: {quality}")

    curve = yaw_model._training_curve(probes, angles)
    tau, delay = yaw_model._fit_dynamics(probes, curve)
    centers, beta, pair_count = _fit_slip_effect(probes, curve, angles)
    baseline = _score(probes, curve, tau, delay)
    candidate = _score(probes, curve, tau, delay,
                       centers=centers, beta=beta)
    gains = []
    max_regression = -math.inf
    print(f"bag: {path}")
    print("capture PASS: 126/126 phases; repetitions 1-2 train, 3 holdout")
    print(f"causal training pairs={pair_count}; beta={beta:+.4f} "
          "(yaw-gain units per unit front |Sy|); "
          f"yaw tau={tau:.3f}s, delay={delay*25}ms")
    print("The feature is computed from the previous predicted rear-axle "
          "state; speed and steering are conditional simulator inputs.")
    print("horizon_ms yaw_RMSE_yawonly/causal-slip(rad/s) gain "
          "position_RMSE_yawonly/slip(m)")
    for horizon in yaw_model.HORIZONS_S:
        base_yaw, base_xy, _, base_n = baseline[horizon]
        fit_yaw, fit_xy, _, fit_n = candidate[horizon]
        if base_n != expected_per_rep or fit_n != expected_per_rep:
            raise ValueError(f"incomplete horizon {horizon:.3f}s")
        gain = 1.0 - fit_yaw / base_yaw if base_yaw > 0.0 else 0.0
        gains.append(gain)
        max_regression = max(max_regression, -gain)
        print(f"{horizon*1000:3.0f} {base_yaw:.5f}/{fit_yaw:.5f} "
              f"{gain:+.1%} {base_xy:.5f}/{fit_xy:.5f}")
    passed = sum(gain >= 0.20 for gain in gains) >= 3 and min(gains) >= -0.10
    print(f"decision: {'PASS causal slip-state model' if passed else 'REJECT causal slip-state model'}; "
          f"gains={','.join(f'{gain:+.1%}' for gain in gains)}")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
