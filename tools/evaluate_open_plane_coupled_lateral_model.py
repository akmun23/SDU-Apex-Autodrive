#!/usr/bin/env python3
"""Test a nonlinear, coupled lateral/yaw state model on an untouched run.

This is a development-only identification experiment. Training labels use the
simulator odometry stream, but model features contain only body speed, lateral
velocity, yaw rate and measured steering. Speed and steering are replayed from
the held-out run so this isolates the lateral/yaw plant; velocity and yaw are
propagated recursively by the candidate. It is not a full offline simulator.

The model uses a smooth radial basis over the measured steering interval,
interacted with the current lateral/yaw state. This preserves the measured
0.18--0.28 rad response shape instead of imposing a linear bicycle gain.
Repetitions 1--2 train; repetition 3 is untouched.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for entry in (SCRIPT_DIR, REPO_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_open_plane_fullsteer_rollout as yaw_model
import evaluate_open_plane_transient_rollout as transient


TRAIN_REPETITIONS = (1, 2)
HOLDOUT_REPETITION = 3
HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750)
STEERING_CENTERS_RAD = np.arange(0.16, 0.301, 0.01)
STEERING_WIDTH_RAD = 0.018
RIDGE = 10.0


def _features(speed: float, vy: float, yaw_rate: float,
              steering: float) -> np.ndarray:
    """Turn-sign canonicalized nonlinear basis over steering and state."""
    sign = 1.0 if steering >= 0.0 else -1.0
    magnitude = abs(steering)
    lateral = sign * vy
    yaw = sign * yaw_rate
    speed_feature = (speed - 4.0) / 0.5
    lateral_feature = lateral / 0.25
    yaw_feature = yaw / 2.0
    state = np.asarray((
        1.0,
        speed_feature,
        lateral_feature,
        yaw_feature,
        lateral_feature * yaw_feature,
    ))
    radial = np.exp(-0.5 * np.square(
        (magnitude - STEERING_CENTERS_RAD) / STEERING_WIDTH_RAD))
    return (radial[:, None] * state[None, :]).reshape(-1)


def _fit(probes):
    features: list[np.ndarray] = []
    targets: list[tuple[float, float]] = []
    for probe in probes:
        if probe.repetition not in TRAIN_REPETITIONS:
            continue
        samples = probe.samples
        for index in range(1, len(samples) - 1):
            previous, current, following = samples[index - 1:index + 2]
            dt = following.time_s - previous.time_s
            if not 0.025 <= dt <= 0.15:
                continue
            sign = 1.0 if current.steering_rad >= 0.0 else -1.0
            features.append(_features(
                current.speed_mps, current.vy_mps,
                current.yaw_rate_rps, current.steering_rad))
            targets.append((
                sign * (following.vy_mps - previous.vy_mps) / dt,
                sign * (following.yaw_rate_rps - previous.yaw_rate_rps) / dt,
            ))
    if len(features) < 500:
        raise ValueError(f"too few training transitions: {len(features)}")

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1.0e-9] = 1.0
    x = (x - x_mean) / x_scale
    y_mean = y.mean(axis=0)
    y_scale = y.std(axis=0)
    y_scale[y_scale < 1.0e-9] = 1.0
    y = (y - y_mean) / y_scale
    coefficients = np.linalg.solve(
        x.T @ x + RIDGE * np.eye(x.shape[1]), x.T @ y)
    return x_mean, x_scale, y_mean, y_scale, coefficients, len(features)


def _predict(model, speed: float, vy: float, yaw_rate: float,
             steering: float) -> tuple[float, float]:
    x_mean, x_scale, y_mean, y_scale, coefficients, _ = model
    sign = 1.0 if steering >= 0.0 else -1.0
    x = (_features(speed, vy, yaw_rate, steering) - x_mean) / x_scale
    derivative = (x @ coefficients) * y_scale + y_mean
    return sign * float(derivative[0]), sign * float(derivative[1])


def _baseline_step(speed: float, yaw_rate: float, steering: float,
                   dt: float, curve, tau: float) -> float:
    target = yaw_model._target_yaw(speed, steering, curve)
    retention = math.exp(-dt / tau)
    return retention * yaw_rate + (1.0 - retention) * target


def _rollout(probe, end_index: int, model, curve, tau: float, delay: int,
             *, coupled: bool, state_limits: tuple[float, float]):
    samples = probe.samples
    vy = samples[0].vy_mps
    yaw_rate = samples[0].yaw_rate_rps
    x, y, heading = samples[0].pose_x_m, samples[0].pose_y_m, samples[0].pose_yaw_rad
    for index in range(1, end_index + 1):
        previous, sample = samples[index - 1:index + 1]
        dt = sample.time_s - previous.time_s
        if not 0.0 < dt <= 0.075:
            raise ValueError(f"bad odometry interval {dt:.4f}s")
        input_index = index - delay
        steering = (probe.initial_steering_rad if input_index < 0
                    else samples[input_index].steering_rad)
        if coupled:
            lateral_dot, yaw_dot = _predict(
                model, sample.speed_mps, vy, yaw_rate, steering)
            vy_next = vy + dt * lateral_dot
            yaw_next = yaw_rate + dt * yaw_dot
        else:
            vy_next = vy
            yaw_next = _baseline_step(
                sample.speed_mps, yaw_rate, steering, dt, curve, tau)
        if (not math.isfinite(vy_next) or not math.isfinite(yaw_next)
                or abs(vy_next) > state_limits[0]
                or abs(yaw_next) > state_limits[1]):
            return None
        yaw_mid = 0.5 * (yaw_rate + yaw_next)
        vy_mid = 0.5 * (vy + vy_next)
        heading_mid = heading + 0.5 * dt * yaw_mid
        x += dt * (sample.speed_mps * math.cos(heading_mid)
                   - vy_mid * math.sin(heading_mid))
        y += dt * (sample.speed_mps * math.sin(heading_mid)
                   + vy_mid * math.cos(heading_mid))
        heading += dt * yaw_mid
        vy, yaw_rate = vy_next, yaw_next
    return vy, yaw_rate, x, y, heading


def _score(probes, model, curve, tau: float, delay: int,
           state_limits: tuple[float, float]):
    holdout = [probe for probe in probes
               if probe.repetition == HOLDOUT_REPETITION]
    output: dict[float, dict[str, tuple[float, int]]] = {}
    for horizon in HORIZONS_S:
        errors = {name: [] for name in (
            "coupled_vy", "baseline_vy", "coupled_yaw", "baseline_yaw",
            "coupled_xy", "baseline_xy", "coupled_heading", "baseline_heading")}
        for probe in holdout:
            index = min(range(len(probe.samples)),
                        key=lambda i: abs(probe.samples[i].time_s - horizon))
            if abs(probe.samples[index].time_s - horizon) > 0.035:
                continue
            actual = probe.samples[index]
            for coupled, prefix in ((True, "coupled"), (False, "baseline")):
                result = _rollout(
                    probe, index, model, curve, tau, delay, coupled=coupled,
                    state_limits=state_limits)
                if result is None:
                    for name in errors:
                        if name.startswith(prefix):
                            errors[name].append(math.inf)
                    continue
                vy, yaw, x, y, heading = result
                errors[f"{prefix}_vy"].append(vy - actual.vy_mps)
                errors[f"{prefix}_yaw"].append(yaw - actual.yaw_rate_rps)
                errors[f"{prefix}_xy"].append(math.hypot(
                    x - actual.pose_x_m, y - actual.pose_y_m))
                errors[f"{prefix}_heading"].append(math.atan2(
                    math.sin(heading - actual.pose_yaw_rad),
                    math.cos(heading - actual.pose_yaw_rad)))
        output[horizon] = {
            name: (math.sqrt(float(np.mean(np.square(values)))), len(values))
            for name, values in errors.items()
        }
    return output


def evaluate(path: Path) -> bool:
    probes, quality = transient.load(path)
    angles = sorted({round(abs(p.commanded_steering_rad), 2) for p in probes})
    if tuple(angles) != yaw_model.TRANSITION_LEVELS:
        raise ValueError(f"expected focused 0.18--0.28 rad capture, got {angles}")
    expected_per_rep = 2 * (2 * len(angles) - 1)
    if (quality["valid_steps"] != 3 * expected_per_rep
            or quality["probe_counts"] != {1: expected_per_rep, 2: expected_per_rep,
                                            3: expected_per_rep, 4: 0}
            or quality["matched_starts"] != 6
            or quality["collisions"] != (0, 0)
            or quality["timing_faults"] != 0 or quality["aborted"]):
        raise ValueError(f"capture failed quality gate: {quality}")

    curve = yaw_model._training_curve(probes, angles)
    tau, delay = yaw_model._fit_dynamics(probes, curve)
    model = _fit(probes)
    training_samples = [sample for probe in probes
                        if probe.repetition in TRAIN_REPETITIONS
                        for sample in probe.samples]
    state_limits = (
        3.0 * max(abs(sample.vy_mps) for sample in training_samples),
        3.0 * max(abs(sample.yaw_rate_rps) for sample in training_samples),
    )
    scores = _score(probes, model, curve, tau, delay, state_limits)
    yaw_gains = []
    vy_gains = []
    print(f"bag: {path}")
    print(f"capture PASS: {quality['valid_steps']}/{quality['valid_steps']} probes; "
          "repetitions 1--2 train, 3 holdout; zero collisions/timing faults")
    print(f"coupled model transitions={model[-1]}, steering centers="
          f"{len(STEERING_CENTERS_RAD)}, ridge={RIDGE:g}; baseline yaw tau={tau:.3f}s, "
          f"steering delay={delay * 25}ms; rollout limits from 3x training state range")
    print("horizon_ms  vy_RMSE baseline/coupled (m/s)  yaw_RMSE baseline/coupled (rad/s) "
          "position_RMSE baseline/coupled (m)")
    for horizon in HORIZONS_S:
        row = scores[horizon]
        b_vy, n = row["baseline_vy"]
        c_vy, nc = row["coupled_vy"]
        b_yaw, _ = row["baseline_yaw"]
        c_yaw, _ = row["coupled_yaw"]
        b_xy, _ = row["baseline_xy"]
        c_xy, _ = row["coupled_xy"]
        if n != expected_per_rep or nc != expected_per_rep:
            raise ValueError(f"incomplete horizon {horizon:.3f}: {n}/{nc}")
        if not all(math.isfinite(value) for value in (c_vy, c_yaw, c_xy)):
            vy_gains.append(-math.inf)
            yaw_gains.append(-math.inf)
            print(f"{horizon * 1000:3.0f}         {b_vy:.5f}/DIVERGED "
                  f"                         {b_yaw:.5f}/DIVERGED "
                  f"                      {b_xy:.5f}/DIVERGED")
            continue
        vy_gain = 1.0 - c_vy / b_vy if b_vy > 1e-12 else 0.0
        yaw_gain = 1.0 - c_yaw / b_yaw if b_yaw > 1e-12 else 0.0
        vy_gains.append(vy_gain)
        yaw_gains.append(yaw_gain)
        print(f"{horizon * 1000:3.0f}         {b_vy:.5f}/{c_vy:.5f} "
              f"({vy_gain:+.1%})                 {b_yaw:.5f}/{c_yaw:.5f} "
              f"({yaw_gain:+.1%})             {b_xy:.5f}/{c_xy:.5f}")

    # Predeclared promotion screen: both propagated states improve at >=3/5
    # horizons and neither state regresses by >10% at any horizon.
    passed = (sum(gain >= 0.20 for gain in vy_gains) >= 3
              and sum(gain >= 0.20 for gain in yaw_gains) >= 3
              and min(vy_gains) >= -0.10 and min(yaw_gains) >= -0.10)
    print("decision: " + ("PASS coupled lateral/yaw model screening" if passed
                         else "REJECT coupled lateral/yaw model screening"))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
