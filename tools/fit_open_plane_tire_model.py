#!/usr/bin/env python3
"""Fit and validate a guide-informed four-wheel lateral-force hypothesis.

This is an offline development tool, not part of either driving runtime. It
uses rear-axle odometry, measured steering, and the open-plane phase labels to
test whether the published tire-slip landmarks explain held-out motion. The
guide does not specify the spline tangents or dynamic wheel normal loads; the
script makes those assumptions explicit and does not claim per-wheel force
identification from body motion alone.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools import analyze_open_plane_dynamics as analysis


MASS_KG = analysis.TOTAL_MASS_KG
GRAVITY_MPS2 = analysis.GRAVITY_MPS2
WHEELBASE_M = analysis.WHEELBASE_M
TRACK_M = analysis.TRACK_WIDTH_M
WHEEL_RADIUS_M = analysis.WHEEL_RADIUS_M
COM_X_M = analysis.COM_X_M
SETTLE_NS = analysis.PHASE_SETTLE_NS

# Current competition MPC yaw-only baseline in mpc_competition.yaml.
MPC_GAIN = 2.95
MPC_GAIN_REDUCTION = 35.6
MPC_GAIN_START_RAD = 0.41
MPC_GAIN_END_RAD = 0.46
MPC_YAW_TAU_S = 0.015

FRONT_LOAD_N = MASS_KG * GRAVITY_MPS2 * COM_X_M / WHEELBASE_M / 2.0
REAR_LOAD_N = MASS_KG * GRAVITY_MPS2 * (WHEELBASE_M - COM_X_M) / WHEELBASE_M / 2.0
STATIC_WHEEL_LOADS_N = np.array(
    [FRONT_LOAD_N, FRONT_LOAD_N, REAR_LOAD_N, REAR_LOAD_N], dtype=float)
MIN_YAW_INERTIA_KGM2 = 0.005
MAX_YAW_INERTIA_KGM2 = 0.08
WHEEL_X_M = np.array([WHEELBASE_M, WHEELBASE_M, 0.0, 0.0], dtype=float)
HALF_TRACK_M = TRACK_M / 2.0
WHEEL_Y_M = np.array([HALF_TRACK_M, -HALF_TRACK_M,
                      HALF_TRACK_M, -HALF_TRACK_M], dtype=float)


@dataclass(frozen=True)
class Sample:
    time_s: float
    phase: str
    repetition: int
    target_speed_mps: float
    steering_rad: float
    u_mps: float
    vy_mps: float
    yaw_rate_rps: float
    yaw_accel_rps2: float
    lateral_force_n: float
    slips: tuple[float, float, float, float]


ACKERMANN_ORDER = "coordinate"


def _ackermann(steering: float) -> tuple[float, float]:
    """Return front-left/right angles; allow testing the guide's side labels."""
    tangent = math.tan(steering)
    numerator = 2.0 * WHEELBASE_M * tangent
    left = math.atan2(numerator, 2.0 * WHEELBASE_M + TRACK_M * tangent)
    right = math.atan2(numerator, 2.0 * WHEELBASE_M - TRACK_M * tangent)
    if ACKERMANN_ORDER == "coordinate":
        # With x forward/y left and left wheels at y=+track/2, the left
        # (inside) wheel must take the larger angle in a positive left turn.
        return right, left
    return left, right


def _wheel_slips(u: float, vy: float, yaw_rate: float,
                 steering: float) -> tuple[float, float, float, float]:
    left, right = _ackermann(steering)
    wheel_angles = (left, right, 0.0, 0.0)
    result = []
    for x, y, angle in zip(WHEEL_X_M, WHEEL_Y_M, wheel_angles):
        vx_body = u - yaw_rate * y
        vy_body = vy + yaw_rate * x
        cosine, sine = math.cos(angle), math.sin(angle)
        vx_wheel = cosine * vx_body + sine * vy_body
        vy_wheel = -sine * vx_body + cosine * vy_body
        result.append(vy_wheel / abs(vx_wheel) if abs(vx_wheel) >= 0.5 else math.nan)
    return tuple(result)  # type: ignore[return-value]


def _mu(abs_slip: np.ndarray, initial_tangent: float) -> np.ndarray:
    """Monotone Hermite interpretation of the published tire-force knots.

    Values are fixed at (0, 0), (0.01, 1), and (0.10, 0.5). The peak and
    asymptote have zero tangent; the unreported initial tangent is the sole
    spline-shape parameter. Force is held at the stated asymptote beyond 0.10.
    """
    slip = np.asarray(abs_slip, dtype=float)
    result = np.empty_like(slip)
    first = slip < 0.01
    t = np.clip(slip[first] / 0.01, 0.0, 1.0)
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    result[first] = initial_tangent * h10 + h01

    second = (slip >= 0.01) & (slip < 0.10)
    t = np.clip((slip[second] - 0.01) / 0.09, 0.0, 1.0)
    smoothstep = 3.0 * t**2 - 2.0 * t**3
    result[second] = 1.0 - 0.5 * smoothstep
    result[slip >= 0.10] = 0.5
    return result


def _local_derivative(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Five-point centered derivative with the measured local sample period."""
    result = np.full_like(values, np.nan, dtype=float)
    for index in range(2, len(values) - 2):
        local_step = (times_s[index + 2] - times_s[index - 2]) / 4.0
        if local_step <= 0.0:
            continue
        result[index] = (
            values[index - 2] - 8.0 * values[index - 1]
            + 8.0 * values[index + 1] - values[index + 2]
        ) / (12.0 * local_step)
    return result


def _load_samples(path: Path) -> list[Sample]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (analysis.ODOM, analysis.STEERING, analysis.PHASE)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError("bag is missing required topic(s): " + ", ".join(missing))

        odometry = [
            (receipt_ns, message)
            for receipt_ns, message in analysis._messages(connection, topics, analysis.ODOM)
        ]
        if len(odometry) < 10:
            raise ValueError("bag has too few odometry samples")
        if any((message.header.frame_id, message.child_frame_id) != ("world", "roboracer_1")
               for _, message in odometry):
            raise ValueError("unexpected odometry frames; expected world -> roboracer_1")

        steering = [
            analysis.ScalarRow(receipt_ns, receipt_ns, float(message.data))
            for receipt_ns, message in analysis._messages(connection, topics, analysis.STEERING)
            if math.isfinite(float(message.data))
        ]
        steering_times = [row.receipt_ns for row in steering]
        phases, _ = analysis._phase_events(connection, topics)
    finally:
        connection.close()

    receipts_ns = np.array([stamp for stamp, _ in odometry], dtype=np.int64)
    times_s = (receipts_ns - receipts_ns[0]).astype(float) / 1e9
    u = np.array([message.twist.twist.linear.x for _, message in odometry], dtype=float)
    vy_com_packet = np.array(
        [message.twist.twist.linear.y for _, message in odometry], dtype=float)
    yaw_rate = np.array([message.twist.twist.angular.z for _, message in odometry], dtype=float)
    # The bridge copies Unity's COM linear velocity into twist unchanged,
    # although pose/TF is at the rear-axle child-frame origin. Shift it before
    # reconstructing wheel-center velocities or fitting a rear-axle model.
    vy = vy_com_packet - COM_X_M * yaw_rate
    vy_com = vy + COM_X_M * yaw_rate
    vy_com_dot = _local_derivative(vy_com, times_s)
    yaw_accel = _local_derivative(yaw_rate, times_s)
    measured_lateral_force = MASS_KG * (vy_com_dot + u * yaw_rate)

    samples: list[Sample] = []
    for phase in phases:
        if not phase.label.startswith(("sweep_r", "isolated_r", "transient_r")) \
                or phase.valid is not True:
            continue
        repetition = int(phase.label.split("_", 2)[1][1:])
        start_ns = phase.start_ns + SETTLE_NS
        left_index = int(np.searchsorted(receipts_ns, start_ns, side="left"))
        right_index = int(np.searchsorted(receipts_ns, phase.end_ns, side="left"))
        for index in range(left_index, right_index):
            actual_steering = analysis._nearest_scalar(
                steering, steering_times, int(receipts_ns[index]))
            if actual_steering is None:
                continue
            slips = _wheel_slips(u[index], vy[index], yaw_rate[index], actual_steering.value)
            if (not all(math.isfinite(value) for value in slips)
                    or not math.isfinite(measured_lateral_force[index])
                    or not math.isfinite(yaw_accel[index])):
                continue
            samples.append(Sample(
                time_s=float(times_s[index]),
                phase=phase.label,
                repetition=repetition,
                target_speed_mps=phase.target_speed_mps,
                steering_rad=actual_steering.value,
                u_mps=u[index],
                vy_mps=vy[index],
                yaw_rate_rps=yaw_rate[index],
                yaw_accel_rps2=yaw_accel[index],
                lateral_force_n=measured_lateral_force[index],
                slips=slips,
            ))
    if not samples:
        raise ValueError("bag contains no valid speed-sweep samples")
    return samples


def _wheel_forces(slips: np.ndarray, scale: float,
                  initial_tangent: float, polarity: float) -> np.ndarray:
    return (polarity * scale * STATIC_WHEEL_LOADS_N
            * np.sign(slips) * _mu(np.abs(slips), initial_tangent))


def _body_forces_and_moment(sample: Sample, vy: float, yaw_rate: float,
                            scale: float, initial_tangent: float,
                            polarity: float) -> tuple[float, float]:
    slips = np.asarray(_wheel_slips(sample.u_mps, vy, yaw_rate, sample.steering_rad))
    local_forces = _wheel_forces(slips, scale,
                                 initial_tangent, polarity)
    front_angles = _ackermann(sample.steering_rad)
    angles = np.array([*front_angles, 0.0, 0.0])
    # The guide describes tire forces along wheel-forward and wheel-lateral
    # axes. This hypothesis models lateral forces only; drive/brake Fx is not
    # observable per wheel in this capture and remains a stated omission.
    body_fx = -local_forces * np.sin(angles)
    body_fy = local_forces * np.cos(angles)
    moment = np.sum((WHEEL_X_M - COM_X_M) * body_fy - WHEEL_Y_M * body_fx)
    return float(np.sum(body_fy)), float(moment)


def _mpc_yaw_gain(steering_rad: float) -> float:
    active_interval = min(max(abs(steering_rad) - MPC_GAIN_START_RAD, 0.0),
                          MPC_GAIN_END_RAD - MPC_GAIN_START_RAD)
    return MPC_GAIN - MPC_GAIN_REDUCTION * active_interval


def _fit_curve(training: list[Sample]) -> tuple[float, float, float, float]:
    force = np.array([sample.lateral_force_n for sample in training])
    slips = np.array([sample.slips for sample in training])
    best = (math.inf, math.nan, math.nan, math.nan)
    for polarity in (-1.0, 1.0):
        signed = np.sign(slips)
        for initial_tangent in np.linspace(0.0, 3.0, 121):
            base = polarity * np.sum(
                STATIC_WHEEL_LOADS_N * signed * _mu(np.abs(slips), initial_tangent), axis=1)
            denominator = float(base @ base)
            if denominator <= 1e-9:
                continue
            scale = max(0.0, float(base @ force) / denominator)
            error = force - scale * base
            rmse = float(np.sqrt(np.mean(error**2)))
            if rmse < best[0]:
                best = rmse, initial_tangent, scale, polarity
    return best[1], best[2], best[3], best[0]


def _fit_yaw_moment_coefficients(yaw_accel: np.ndarray, yaw_rate: np.ndarray,
                                 tire_moment: np.ndarray) -> tuple[float, float]:
    """Fit M_tire = Iz*r_dot + C_yaw*r with physical coefficient bounds."""
    inertia_min = MIN_YAW_INERTIA_KGM2
    inertia_max = MAX_YAW_INERTIA_KGM2
    design = np.column_stack((yaw_accel, yaw_rate))
    unconstrained = np.linalg.lstsq(design, tire_moment, rcond=None)[0]
    candidates: list[tuple[float, float]] = []
    if (inertia_min <= unconstrained[0] <= inertia_max
            and unconstrained[1] >= 0.0):
        candidates.append((float(unconstrained[0]), float(unconstrained[1])))

    yaw_rate_energy = float(yaw_rate @ yaw_rate)
    for inertia in (inertia_min, inertia_max):
        damping = (max(0.0, float(yaw_rate @ (tire_moment - inertia * yaw_accel))
                       / yaw_rate_energy) if yaw_rate_energy > 1e-12 else 0.0)
        candidates.append((inertia, damping))

    accel_energy = float(yaw_accel @ yaw_accel)
    inertia = (float(yaw_accel @ tire_moment) / accel_energy
               if accel_energy > 1e-12 else inertia_min)
    candidates.append((min(inertia_max, max(inertia_min, inertia)), 0.0))
    return min(candidates, key=lambda pair: float(np.mean(
        (design @ np.asarray(pair) - tire_moment) ** 2)))


def _tire_moments(samples: list[Sample], scale: float,
                  initial_tangent: float, polarity: float) -> np.ndarray:
    return np.asarray([
        _body_forces_and_moment(sample, sample.vy_mps, sample.yaw_rate_rps,
                                scale, initial_tangent, polarity)[1]
        for sample in samples
    ])


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    return float(np.sqrt(np.mean((actual[valid] - predicted[valid]) ** 2)))


def _evaluate_rollouts(samples: list[Sample], repetition: int,
                       scale: float, initial_tangent: float,
                       polarity: float, yaw_inertia: float) -> tuple[float, float, float, float]:
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        if sample.repetition == repetition:
            grouped.setdefault(sample.phase, []).append(sample)

    baseline_vy_error: list[float] = []
    baseline_r_error: list[float] = []
    tire_vy_error: list[float] = []
    tire_r_error: list[float] = []
    for phase_samples in grouped.values():
        phase_samples.sort(key=lambda sample: sample.time_s)
        if len(phase_samples) < 4:
            continue
        base_vy = phase_samples[0].vy_mps
        base_r = phase_samples[0].yaw_rate_rps
        tire_vy = base_vy
        tire_r = base_r
        previous_t = phase_samples[0].time_s
        for sample in phase_samples[1:]:
            dt = sample.time_s - previous_t
            previous_t = sample.time_s
            if not 0.01 <= dt <= 0.06:
                continue

            gain = _mpc_yaw_gain(sample.steering_rad)
            steady_r = sample.u_mps * math.tan(sample.steering_rad) * gain
            retention = math.exp(-dt / MPC_YAW_TAU_S)
            base_r_next = retention * base_r + (1.0 - retention) * steady_r
            base_vy_next = base_vy

            force, moment = _body_forces_and_moment(
                sample, tire_vy, tire_r, scale, initial_tangent, polarity)
            tire_r_dot = moment / yaw_inertia
            tire_vcom_dot = force / MASS_KG - sample.u_mps * tire_r
            tire_vy_dot = tire_vcom_dot - COM_X_M * tire_r_dot
            tire_vy_next = tire_vy + dt * tire_vy_dot
            tire_r_next = tire_r + dt * tire_r_dot

            baseline_vy_error.append(base_vy_next - sample.vy_mps)
            baseline_r_error.append(base_r_next - sample.yaw_rate_rps)
            tire_vy_error.append(tire_vy_next - sample.vy_mps)
            tire_r_error.append(tire_r_next - sample.yaw_rate_rps)
            base_vy, base_r = base_vy_next, base_r_next
            tire_vy, tire_r = tire_vy_next, tire_r_next

    return (
        _rmse(np.asarray(baseline_vy_error), np.zeros(len(baseline_vy_error))),
        _rmse(np.asarray(baseline_r_error), np.zeros(len(baseline_r_error))),
        _rmse(np.asarray(tire_vy_error), np.zeros(len(tire_vy_error))),
        _rmse(np.asarray(tire_r_error), np.zeros(len(tire_r_error))),
    )


def _self_test() -> None:
    for tangent in (0.0, 1.5, 3.0):
        knots = _mu(np.array([0.0, 0.01, 0.10, 0.20]), tangent)
        assert np.allclose(knots, [0.0, 1.0, 0.5, 0.5], atol=1e-12)
        dense = _mu(np.linspace(0.0, 0.01, 101), tangent)
        assert np.all(np.diff(dense) >= -1e-12)
    assert math.isclose(float(STATIC_WHEEL_LOADS_N.sum()), MASS_KG * GRAVITY_MPS2)
    assert math.isclose(FRONT_LOAD_N * 2.0 * WHEELBASE_M,
                        MASS_KG * GRAVITY_MPS2 * COM_X_M)
    assert math.isclose(_wheel_slips(2.0, 0.0, 0.0, 0.0)[0], 0.0, abs_tol=1e-12)
    left, right = _ackermann(0.2)
    assert left > right > 0.0
    left_negative, right_negative = _ackermann(-0.2)
    assert math.isclose(left_negative, -right, abs_tol=1e-12)
    assert math.isclose(right_negative, -left, abs_tol=1e-12)
    rear_vy = analysis._rear_axle_velocity(3.0, 0.45, 2.0)[1]
    assert math.isclose(rear_vy + 2.0 * COM_X_M, 0.45, abs_tol=1e-12)
    assert math.isclose(_mpc_yaw_gain(0.41), 2.95, abs_tol=1e-12)
    assert math.isclose(_mpc_yaw_gain(0.46), 1.17, abs_tol=1e-12)
    assert math.isclose(_mpc_yaw_gain(0.50), 1.17, abs_tol=1e-12)
    synthetic_accel = np.array([-3.0, -0.5, 0.25, 2.0])
    synthetic_yaw = np.array([-1.0, -0.2, 0.1, 0.7])
    synthetic_moment = 0.035 * synthetic_accel + 0.08 * synthetic_yaw
    fit_inertia, fit_damping = _fit_yaw_moment_coefficients(
        synthetic_accel, synthetic_yaw, synthetic_moment)
    assert math.isclose(fit_inertia, 0.035, abs_tol=1e-12)
    assert math.isclose(fit_damping, 0.08, abs_tol=1e-12)
    print("tire-model mathematics: PASS (spline landmarks/monotonicity, "
          "static load balance, zero slip, COM shift, Ackermann side order, "
          "bounded yaw inertia/damping fit)")


def analyze(path: Path) -> int:
    samples = _load_samples(path)
    repetitions = sorted({sample.repetition for sample in samples})
    if len(repetitions) >= 3:
        heldout_repetition = repetitions[-1]
        training = [sample for sample in samples if sample.repetition != heldout_repetition]
    else:
        heldout_repetition = repetitions[-1]
        training = [sample for sample in samples if sample.repetition != heldout_repetition]
    heldout = [sample for sample in samples if sample.repetition == heldout_repetition]
    if len(training) < 100 or len(heldout) < 100:
        raise ValueError(f"inadequate train/heldout coverage: {len(training)}/{len(heldout)} samples")

    initial_tangent, force_scale, polarity, train_rmse = _fit_curve(training)
    train_force = np.array([sample.lateral_force_n for sample in training])
    test_force = np.array([sample.lateral_force_n for sample in heldout])
    def predict_force(group: list[Sample], tangent: float, scale: float,
                      sign: float) -> np.ndarray:
        slips = np.array([sample.slips for sample in group])
        return sign * scale * np.sum(
            STATIC_WHEEL_LOADS_N * np.sign(slips)
            * _mu(np.abs(slips), tangent), axis=1)

    guide_train_prediction = predict_force(training, initial_tangent, 1.0, polarity)
    fitted_train_prediction = predict_force(training, initial_tangent, force_scale, polarity)
    heldout_prediction = predict_force(heldout, initial_tangent, force_scale, polarity)
    all_slips = np.abs(np.array([sample.slips for sample in samples]))
    front_slips = all_slips[:, :2]
    rear_slips = all_slips[:, 2:]
    training_label = ",".join(str(repetition) for repetition in repetitions
                               if repetition != heldout_repetition)
    print(f"bag: {path}")
    print(f"samples: train repetitions {training_label}={len(training)}, "
          f"held-out repetition {heldout_repetition}={len(heldout)}")
    print("state reference: rear-axle body frame; bridge COM twist shifted "
          "using the published x_COM before wheel kinematics")
    print(f"Ackermann wheel assignment tested: {ACKERMANN_ORDER}")
    print(f"static tire loads: front={FRONT_LOAD_N:.3f} N/wheel, "
          f"rear={REAR_LOAD_N:.3f} N/wheel; total={STATIC_WHEEL_LOADS_N.sum():.3f} N")
    print(f"abs(Sy) coverage: front={front_slips.min():.4f}..{front_slips.max():.4f}, "
          f"rear={rear_slips.min():.4f}..{rear_slips.max():.4f}; "
          f"samples below peak={np.sum(all_slips < 0.01)}, "
          f"between peak/asymptote={np.sum((all_slips >= 0.01) & (all_slips < 0.10))}, "
          f"at/above asymptote={np.sum(all_slips >= 0.10)}")
    print("published lateral tire landmarks: peak (|Sy|=0.010, mu=1.0), "
          "asymptote (|Sy|=0.100, mu=0.5); static loads and spline tangents are assumptions")
    print(f"fitted force sign={polarity:+.0f}, initial Hermite tangent={initial_tangent:.3f}, "
          f"common tire-force scale={force_scale:.3f} (guide nominal=1.0)")
    print(f"net lateral-force RMSE: guide scale=1.0 train={_rmse(train_force, guide_train_prediction):.3f} N, "
          f"fitted train={_rmse(train_force, fitted_train_prediction):.3f} N, "
          f"held-out={_rmse(test_force, heldout_prediction):.3f} N")
    print(f"Ackermann wheel assignment tested: {ACKERMANN_ORDER}")
    print(f"abs(Sy) coverage: front={np.abs(np.array([s.slips for s in samples])[:, :2]).min():.4f}.."
          f"{np.abs(np.array([s.slips for s in samples])[:, :2]).max():.4f}, "
          f"rear={np.abs(np.array([s.slips for s in samples])[:, 2:]).min():.4f}.."
          f"{np.abs(np.array([s.slips for s in samples])[:, 2:]).max():.4f}")
    print(f"best force fit: sign={polarity:+.0f}, tangent={initial_tangent:.3f}, "
          f"scale={force_scale:.3f}; guide-scale RMSE="
          f"{_rmse(train_force, guide_train_prediction):.3f} N train, "
          f"fitted={_rmse(train_force, fitted_train_prediction):.3f} N train / "
          f"{_rmse(test_force, heldout_prediction):.3f} N held-out")

    training_moments = np.array([
        _body_forces_and_moment(sample, sample.vy_mps, sample.yaw_rate_rps,
                                force_scale, initial_tangent, polarity)[1]
        for sample in training
    ])
    test_moments = np.array([
        _body_forces_and_moment(sample, sample.vy_mps, sample.yaw_rate_rps,
                                force_scale, initial_tangent, polarity)[1]
        for sample in heldout
    ])
    training_yaw_accel = np.array([sample.yaw_accel_rps2 for sample in training])
    training_yaw_rate = np.array([sample.yaw_rate_rps for sample in training])
    denominator = float(training_yaw_accel @ training_yaw_accel)
    yaw_inertia = (float(training_moments @ training_yaw_accel) / denominator
                   if denominator > 1e-9 else math.nan)

    damped_inertia, yaw_damping = _fit_yaw_moment_coefficients(
        training_yaw_accel, training_yaw_rate, training_moments)
    unconstrained_moment_fit = np.linalg.lstsq(
        np.column_stack((training_yaw_accel, training_yaw_rate)),
        training_moments, rcond=None)[0]
    damped_test_accel = (test_moments - yaw_damping * np.array(
        [sample.yaw_rate_rps for sample in heldout])) / damped_inertia
    heldout_yaw_accel = np.array([sample.yaw_accel_rps2 for sample in heldout])
    no_accel_rmse = _rmse(heldout_yaw_accel, np.zeros_like(heldout_yaw_accel))
    damped_accel_rmse = _rmse(heldout_yaw_accel, damped_test_accel)
    repetition_fits = []
    for repetition in sorted({sample.repetition for sample in training}):
        group = [sample for sample in training if sample.repetition == repetition]
        rep_moments = _tire_moments(
            group, force_scale, initial_tangent, polarity)
        rep_inertia, rep_damping = _fit_yaw_moment_coefficients(
            np.asarray([sample.yaw_accel_rps2 for sample in group]),
            np.asarray([sample.yaw_rate_rps for sample in group]),
            rep_moments)
        repetition_fits.append((rep_inertia, rep_damping))
    inertia_mean = (sum(pair[0] for pair in repetition_fits) / len(repetition_fits)
                    if repetition_fits else math.nan)
    inertia_spread = ((max(pair[0] for pair in repetition_fits)
                       - min(pair[0] for pair in repetition_fits)) / inertia_mean
                      if inertia_mean > 0.0 and len(repetition_fits) >= 2
                      else math.inf)
    interior_inertia = (MIN_YAW_INERTIA_KGM2 < damped_inertia
                        < MAX_YAW_INERTIA_KGM2)
    stable_repetitions = inertia_spread <= 0.30
    holdout_improvement = (1.0 - damped_accel_rmse / no_accel_rmse
                           if no_accel_rmse > 0.0 else -math.inf)
    drag_model_accepted = (interior_inertia and stable_repetitions
                           and holdout_improvement >= 0.20)
    print("effective linear yaw-drag check (M_tire = Iz*r_dot + C_yaw*r):")
    print(f"  unconstrained training fit: Iz={unconstrained_moment_fit[0]:+.5f} kg m^2, "
          f"C_yaw={unconstrained_moment_fit[1]:+.5f} N m s/rad")
    print(f"  bounded fit: Iz={damped_inertia:.5f} kg m^2, "
          f"C_yaw={yaw_damping:.5f} N m s/rad; per-repetition Iz spread="
          f"{inertia_spread:.1%}")
    print(f"  holdout yaw-acceleration RMSE: zero-acceleration={no_accel_rmse:.3f}, "
          f"tire+drag={damped_accel_rmse:.3f} rad/s^2 "
          f"({holdout_improvement:+.1%})")
    print(f"  decision: {'PASS' if drag_model_accepted else 'REJECT'}; requires "
          "interior plausible Iz, <=30% repetition spread, and >=20% "
          "held-out acceleration improvement")
    if not math.isfinite(yaw_inertia) or yaw_inertia <= 0.0:
        print(f"yaw inertia inferred from training net moment: {yaw_inertia:.5f} kg m^2")
        print(f"guide-force moment fit implies non-positive yaw inertia ({yaw_inertia:.5f} kg m^2); "
              "moments cannot yet support a forward rollout")
        print("result: REJECTED for MPC promotion; no yaw-inertia fallback is applied")
        return 0
    heldout_moment_yaw_accel = test_moments / yaw_inertia

    baseline_vy, baseline_r, tire_vy, tire_r = _evaluate_rollouts(
        samples, heldout_repetition, force_scale, initial_tangent, polarity, yaw_inertia)

    print(f"yaw inertia inferred from training net moment: {yaw_inertia:.5f} kg m^2 "
          f"(rectangle scale m(L^2+W^2)/12="
          f"{MASS_KG * (0.5**2 + 0.27**2) / 12.0:.5f})")
    print(f"held-out yaw-acceleration RMSE from guide-force moments: "
          f"{_rmse(heldout_yaw_accel, heldout_moment_yaw_accel):.3f} rad/s^2")
    print("held-out one-step state RMSE (current yaw-only MPC vs four-wheel force rollout):")
    print(f"  rear-axle vy: {baseline_vy:.4f} vs {tire_vy:.4f} m/s")
    print(f"  yaw rate:     {baseline_r:.4f} vs {tire_r:.4f} rad/s")

    improves_vy = tire_vy <= 0.8 * baseline_vy
    improves_yaw = tire_r <= 0.8 * baseline_r
    plausible_inertia = 0.005 <= yaw_inertia <= 0.08
    nominal_force = 0.5 <= force_scale <= 1.5
    accepted = improves_vy and improves_yaw and plausible_inertia and nominal_force
    print("predeclared promotion screen: ≥20% held-out RMSE improvement in both vy and yaw rate, "
          "0.005≤Iz≤0.08 kg m², and 0.5≤force scale≤1.5")
    print(f"result: {'PASS' if accepted else 'REJECTED for MPC promotion'}")
    if not accepted:
        print("Interpretation: this fit is diagnostic only; retain the current MPC model and "
              "do not tune raceline limits from this fit.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", nargs="?", type=Path,
                        help="speed-sweep run_0.db3 bag")
    parser.add_argument("--self-test", action="store_true",
                        help="run focused spline and force-balance checks")
    parser.add_argument("--ackermann-order", choices=("guide", "coordinate"),
                        default="coordinate",
                        help="map the guide's printed left/right equations directly, or swap them "
                             "to match x-forward/y-left wheel locations")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.bag is None:
        parser.error("provide a bag or use --self-test")
    global ACKERMANN_ORDER
    ACKERMANN_ORDER = args.ackermann_order
    return analyze(args.bag)


if __name__ == "__main__":
    raise SystemExit(main())
