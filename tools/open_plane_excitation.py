#!/usr/bin/env python3
"""Development-only direct actuator excitation for the wall-free explore sim.

Run only against the pinned explore simulator with other actuator publishers
stopped. This script intentionally consumes simulator odometry and collision
truth for experiment safety; it is not part of any competition launch/runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import random
import signal
import statistics
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace

from race_domain_experiment_plan import (
    HIGH_STEER_VALIDATION_SPEED_MPS,
    RACE_DOMAIN_BOUNDARY_SPEED_MPS,
    RACE_DOMAIN_GOVERNOR_MPS,
    RACE_DOMAIN_HARD_LIMIT_MPS,
    RACE_DOMAIN_THROTTLE_SPEED_ANCHORS,
    RACE_DOMAIN_STEERING_FRONTIER_SPEEDS_MPS,
    build_race_domain_steering_frontier_plan,
    build_race_domain_boundary_plan,
    build_high_steer_validation_plan,
    build_race_domain_moderate_braking_plan,
    build_race_domain_plan,
    plan_duration_s,
    race_domain_feedforward,
    race_domain_moderate_steering_limit,
    steering_limit_for_speed,
)
from race_domain_dynamic_coupled_plan import (
    DYNAMIC_STEERING_FREQUENCIES_HZ,
    FRONTIER_11MPS_ANGLES_RAD,
    FRONTIER_SWEEP_ANGLES_RAD,
    PRBS_MIN_DWELL_S,
    build_dynamic_coupled_plan,
    plan_as_dicts as dynamic_coupled_plan_as_dicts,
)

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, Float32, Int32, String


RATE_HZ = 40.0
PERIOD_SEC = 1.0 / RATE_HZ
STEERING_LIMIT_RAD = 0.5236
COM_X_M = 0.15532
MAX_STEERING_RAD = STEERING_LIMIT_RAD
MAX_THROTTLE = 0.5
EMERGENCY_SPEED_MPS = 9.0
EXCITATION_SPEED_GOVERNOR_MPS = 8.0
# Completed full-input captures stayed below 6.2 deg; the partial run then
# crossed 8.5 deg and exceeded 50 deg within 0.1 s before the speed cutoff.
MAX_EXPERIMENT_TILT_RAD = math.radians(8.0)
ODOM_TIMEOUT_SEC = 0.25
COLLISION_TIMEOUT_SEC = 0.50
STEERING_FEEDBACK_TIMEOUT_SEC = 0.15
PHASE_SETTLE_SEC = 0.35
PROBE_START_SETTLE_SEC = 0.50
PROBE_START_TIMEOUT_SEC = 4.0
PROBE_START_MAX_VY_MPS = 0.08
PROBE_START_MAX_YAW_RATE_RPS = 0.12
PROBE_START_MAX_SPEED_ERROR_MPS = 0.20
PROBE_START_MAX_STEERING_RAD = 0.02
PROBE_START_WINDOW_SEC = 0.50
PROBE_START_MIN_SAMPLES = 15
MIN_PHASE_SAMPLES = 30
MAX_SPEED_MEDIAN_ERROR_MPS = 0.12
MAX_SPEED_P95_ERROR_MPS = 0.20
MAX_STEERING_P95_ERROR_RAD = 0.02
STEERING_AMPLITUDES_RAD = (0.15, 0.25, 0.30, 0.35, 0.42)
SPEED_TARGETS_MPS = (2.5, 4.5, 6.5, 7.5, 8.5)
FULL_INPUT_STEERING_LEVELS_RAD = (
    0.0, -0.05, 0.05, -0.10, 0.10, -0.15, 0.15, -0.20, 0.20,
    -0.25, 0.25, -0.30, 0.30, -0.35, 0.35, -0.42, 0.42,
    -0.46, 0.46, -0.50, 0.50, -0.5236, 0.5236,
)
FULL_INPUT_THROTTLE_LEVELS = (0.0, 0.08, 0.14, 0.20, 0.28, 0.36, 0.44, 0.50)
THROTTLE_FEEDFORWARD = (0.10, 0.18, 0.27, 0.31, 0.34)
THROTTLE_SLEW_STEERING_RAD = (0.30, 0.42)
THROTTLE_SLEW_DELTA_NORM = 0.08
THROTTLE_SLEW_STIMULUS_DELAY_S = 0.60
THROTTLE_SLEW_RAMP_S = 0.30
THROTTLE_SLEW_PHASE_S = 1.80
THROTTLE_START_FEEDBACK_TOLERANCE = 0.03
ODOM_TOPIC = "/autodrive/roboracer_1/odom"
STEERING_TOPIC = "/autodrive/roboracer_1/steering"
THROTTLE_FEEDBACK_TOPIC = "/autodrive/roboracer_1/throttle"
COLLISION_TOPIC = "/autodrive/roboracer_1/collision_count"
RESET_COMMAND_TOPIC = "/autodrive/reset_command"
STEERING_COMMAND_TOPIC = "/autodrive/roboracer_1/steering_command"
THROTTLE_COMMAND_TOPIC = "/autodrive/roboracer_1/throttle_command"
PHASE_TOPIC = "/open_plane_experiment/phase"
SIM_RESET_HOLD_SEC = 0.90
SIM_RESET_TIMEOUT_SEC = 4.0
SIM_RESET_STABLE_SEC = 0.50
SIM_SPAWN_TOLERANCE_M = 0.25
DYNAMIC_COUPLED_PROFILES = (
    "race_domain_dynamic_coupled_train",
    "race_domain_dynamic_coupled_validation",
    "race_domain_dynamic_coupled_final",
)
SWERVE_THROTTLE_SLEW_PROFILES = (
    "race_domain_swerve_throttle_slew_train",
    "race_domain_swerve_throttle_slew_validation",
)
SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE = (
    "race_domain_swerve_throttle_slew_frontier_validation")
SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_PROFILE = (
    "race_domain_swerve_throttle_slew_11mps_replication")
SWERVE_THROTTLE_SLEW_MODERATE_PROFILE = (
    "race_domain_swerve_throttle_slew_moderate_validation")
SWERVE_THROTTLE_SLEW_LOWSTEER_PROFILE = (
    "race_domain_swerve_throttle_slew_lowsteer_validation")
YAW_FRONTIER_THROTTLE_SLEW_PROFILE = "yaw_frontier_throttle_slew_train"
YAW_FRONTIER_THROTTLE_SLEW_SPEED_STEERING_RAD = (
    (9.5, (0.14, 0.18, 0.20)),
)
YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE = (
    "yaw_frontier_lowangle_unwind_train")
YAW_FRONTIER_LOWANGLE_UNWIND_REPEATS = 4
YAW_FRONTIER_LOWANGLE_UNWIND_SPEED_STEERING_RAD = (
    (9.5, (0.14, 0.18, 0.20)),
)
SWERVE_THROTTLE_RATE_SWEEP_PROFILE = (
    "race_domain_swerve_throttle_rate_sweep_validation")
SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE = (
    "race_domain_swerve_throttle_rate_sweep_highsteer_validation")
SWERVE_THROTTLE_RATE_FACTORIAL_PROFILE = (
    "race_domain_swerve_throttle_rate_factorial_validation")
SWERVE_THROTTLE_RATE_FACTORIAL_4P5_PROFILE = (
    "race_domain_swerve_throttle_rate_factorial_4p5_validation")
SWERVE_THROTTLE_RATE_FACTORIAL_6P5_PROFILE = (
    "race_domain_swerve_throttle_rate_factorial_6p5_validation")
SWERVE_THROTTLE_RATE_FACTORIAL_7P5_PROFILE = (
    "race_domain_swerve_throttle_rate_factorial_7p5_validation")
SWERVE_THROTTLE_RATE_FRONTIER_UP_TRAIN_PROFILE = (
    "race_domain_swerve_throttle_slew_up_frontier_train")
SWERVE_THROTTLE_RATE_FRONTIER_UP_VALIDATION_PROFILE = (
    "race_domain_swerve_throttle_slew_up_frontier_validation")
SWERVE_THROTTLE_RATE_FRONTIER_UP_PROFILES = (
    SWERVE_THROTTLE_RATE_FRONTIER_UP_TRAIN_PROFILE,
    SWERVE_THROTTLE_RATE_FRONTIER_UP_VALIDATION_PROFILE,
)
SWERVE_THROTTLE_RATE_FRONTIER_UP_PROFILE = (
    SWERVE_THROTTLE_RATE_FRONTIER_UP_VALIDATION_PROFILE)
SWERVE_THROTTLE_RATE_RACE_DOMAIN_TRAIN_PROFILE = (
    "race_domain_swerve_throttle_rate_race_domain_train")
SWERVE_THROTTLE_RATE_RACE_DOMAIN_VALIDATION_PROFILE = (
    "race_domain_swerve_throttle_rate_race_domain_validation")
SWERVE_THROTTLE_RATE_RACE_DOMAIN_PROFILES = (
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_TRAIN_PROFILE,
    SWERVE_THROTTLE_RATE_RACE_DOMAIN_VALIDATION_PROFILE,
)
SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES = (
    *SWERVE_THROTTLE_SLEW_PROFILES,
    SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE,
    SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_PROFILE,
    SWERVE_THROTTLE_SLEW_MODERATE_PROFILE,
    SWERVE_THROTTLE_SLEW_LOWSTEER_PROFILE,
    YAW_FRONTIER_THROTTLE_SLEW_PROFILE,
    YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE,
    SWERVE_THROTTLE_RATE_SWEEP_PROFILE,
    SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_4P5_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_6P5_PROFILE,
    SWERVE_THROTTLE_RATE_FACTORIAL_7P5_PROFILE,
    *SWERVE_THROTTLE_RATE_FRONTIER_UP_PROFILES,
    *SWERVE_THROTTLE_RATE_RACE_DOMAIN_PROFILES,
)
LOW_SPEED_TRANSIENT_PROFILE = "race_domain_low_speed_highsteer_transients"
LOW_SPEED_TRANSIENT_SPEEDS_MPS = (2.5, 3.0, 3.5)
LOW_SPEED_TRANSIENT_STEERING_RAD = (0.30, 0.35, 0.40, 0.42)
YAW_TRANSIENT_PROFILE = "race_domain_yaw_transition_transients"
YAW_TRANSIENT_SPEEDS_MPS = (3.5, 3.75, 4.0)
YAW_TRANSIENT_STEERING_RAD = (0.30, 0.35, 0.42)
YAW_ATLAS_INTERPOLATION_PROFILE = "yaw_atlas_interpolation_validation"
YAW_ATLAS_INTERPOLATION_POINTS = (
    (4.25, 0.0875), (4.25, 0.2625), (6.75, 0.1625),
    (7.75, 0.1625), (8.75, 0.0125), (10.25, 0.0375),
)
YAW_ATLAS_OFFGRID_FINAL_PROFILE = "yaw_atlas_offgrid_final"
YAW_ATLAS_OFFGRID_FINAL_POINTS = (
    (4.75, 0.0625), (4.75, 0.1125), (6.75, 0.0875),
    (8.75, 0.0625), (8.75, 0.1125), (10.75, 0.0875),
)
YAW_ATLAS_EXTRATREES_FINAL_PROFILE = "yaw_atlas_extratrees_final"
YAW_ATLAS_EXTRATREES_FINAL_POINTS = (
    (4.62, 0.108), (6.62, 0.083), (7.62, 0.133),
    (8.12, 0.058), (8.62, 0.058), (10.62, 0.033),
)
YAW_ATLAS_INTERPOLATION_REPEATS = 2
YAW_FULLBAND_GAPFILL_PROFILE = "yaw_fullband_gapfill_train"
YAW_FULLBAND_GAPFILL_POINTS = (
    (4.75, 0.125),
    (8.75, 0.05), (8.75, 0.075), (8.75, 0.10), (8.75, 0.125),
    (10.75, 0.05), (10.75, 0.075), (10.75, 0.10), (10.75, 0.125),
)
YAW_FULLBAND_GAPFILL_REPEATS = 2
YAW_UNWIND_THROTTLE_PROFILE = "yaw_unwind_throttle_slew_train"
YAW_UNWIND_THROTTLE_SPEED_MPS = 8.75
YAW_UNWIND_THROTTLE_STEERING_RAD = 0.075
# Prior clean captures at this speed measured a steady throttle feedback near
# 0.361; pairing the probe to 0.35 settled below the intended speed cell.
YAW_UNWIND_THROTTLE_START_NORM = 0.361
YAW_UNWIND_THROTTLE_END_NORM = 0.241
YAW_UNWIND_THROTTLE_REPEATS = 2
YAW_UNWIND_THROTTLE_MODES = ("step", "ramp")
YAW_LOW_ANGLE_RATE_PROFILE = "yaw_low_angle_rate_surface"
YAW_LOW_ANGLE_RATE_SPEEDS_MPS = (4.25, 6.25, 8.25, 10.25)
YAW_LOW_ANGLE_RATE_STEERING_RAD = (0.05, 0.075, 0.10, 0.125)
YAW_LOW_ANGLE_RATE_MODES = (("step", 0.025), ("ramp", 0.30))
YAW_LOW_ANGLE_RATE_REPEATS = 2
YAW_TRANSIENT_PROFILES = (
    LOW_SPEED_TRANSIENT_PROFILE, YAW_TRANSIENT_PROFILE,
    YAW_ATLAS_INTERPOLATION_PROFILE, YAW_ATLAS_OFFGRID_FINAL_PROFILE,
    YAW_ATLAS_EXTRATREES_FINAL_PROFILE,
    YAW_FULLBAND_GAPFILL_PROFILE, YAW_UNWIND_THROTTLE_PROFILE,
    YAW_LOW_ANGLE_RATE_PROFILE,
)


def _is_yaw_transient_approach(label: str) -> bool:
    return label.startswith(("approach_lowdyn_", "approach_yawdyn_",
                             "approach_atlas_", "approach_yawgap_",
                             "approach_yawbrake_",
                             "approach_lowyaw_"))


RACE_DOMAIN_SPEED_GOVERNED_PROFILES = (
    *DYNAMIC_COUPLED_PROFILES,
    *SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES,
    *YAW_TRANSIENT_PROFILES,
)
SWERVE_THROTTLE_SLEW_SPEED_STEERING_RAD = (
    (4.5, (0.30, 0.42)),
    (6.5, (0.30, 0.42)),
    (7.5, (0.30, 0.42)),
    (9.5, (0.08, 0.14)),
    (11.1, (0.06, 0.12)),
)
SWERVE_THROTTLE_SLEW_APPROACH_S = {
    4.5: 8.0,
    6.5: 9.0,
    7.5: 10.0,
    9.0: 11.0,
    9.5: 11.0,
    10.0: 12.0,
    10.5: 12.0,
    11.1: 12.0,
}
SWERVE_THROTTLE_SLEW_FRONTIER_SPEED_STEERING_RAD = (
    (9.5, (0.14, 0.18, 0.20)),
    (10.5, (0.14, 0.18, 0.20)),
    (11.1, (0.12, 0.16, 0.20)),
)
SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_SPEED_STEERING_RAD = (
    (11.1, (0.12, 0.16, 0.20)),
)
SWERVE_THROTTLE_SLEW_MODERATE_SPEED_STEERING_RAD = (
    (4.5, (0.12, 0.20)),
    (6.5, (0.12, 0.20)),
    (7.5, (0.12, 0.20)),
)
SWERVE_THROTTLE_SLEW_LOWSTEER_SPEED_STEERING_RAD = (
    (4.5, (0.08, 0.10)),
    (6.5, (0.08, 0.10)),
    (7.5, (0.08, 0.10)),
)
SWERVE_THROTTLE_SLEW_DELTA_NORM = 0.08
SWERVE_THROTTLE_SLEW_PHASE_S = 1.85
SWERVE_THROTTLE_RATE_SWEEP_SPEED_MPS = 8.0
SWERVE_THROTTLE_RATE_SWEEP_STEERING_RAD = (0.30, 0.42)
SWERVE_THROTTLE_RATE_SWEEP_RAMP_DURATIONS_S = (0.15, 0.30, 0.60)
SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_RAMP_DURATIONS_S = (0.15, 0.30)
SWERVE_THROTTLE_RATE_FACTORIAL_DELTAS_NORM = (0.08, 0.12)
SWERVE_THROTTLE_RATE_FACTORIAL_4P5_SPEED_MPS = 4.5
SWERVE_THROTTLE_RATE_FACTORIAL_6P5_SPEED_MPS = 6.5
SWERVE_THROTTLE_RATE_FACTORIAL_7P5_SPEED_MPS = 7.5
SWERVE_THROTTLE_RATE_FACTORIAL_RATES_NORM_PER_SEC = (
    0.13333333333333333, 0.26666666666666666, 0.5333333333333333)
SWERVE_THROTTLE_RATE_FRONTIER_UP_SPEED_STEERING_RAD = (
    (9.0, (0.12, 0.20)),
    (10.0, (0.08, 0.14)),
)
SWERVE_THROTTLE_RATE_FRONTIER_UP_DELTA_NORM = 0.04
SWERVE_THROTTLE_RATE_FRONTIER_UP_RAMP_DURATIONS_S = (0.15, 0.30)
SWERVE_THROTTLE_RATE_RACE_DOMAIN_STEERING_RAD = (0.08, 0.14, 0.20)
SWERVE_THROTTLE_RATE_RACE_DOMAIN_DELTAS_NORM = (0.04, 0.08)
SWERVE_THROTTLE_RATE_RACE_DOMAIN_RAMP_DURATIONS_S = (0.15, 0.30)
SWERVE_THROTTLE_RATE_RACE_DOMAIN_SPEEDS_MPS = (4.5, 6.5, 7.5)
SUBNET_TRANSIENT_PROFILE = "subnet_highsteer_transients"
SUBNET_TRANSIENT_SPEED_MPS = HIGH_STEER_VALIDATION_SPEED_MPS
SUBNET_TRANSIENT_STEERING_RAD = (0.30, 0.42)
SUBNET_TRANSIENT_MANOEUVRES = (
    "turnin_acceleration",
    "turnin_throttle_reduction",
    "turnin_active_braking",
    "unwind_acceleration",
    "unwind_active_braking",
)
SUBNET_TRANSIENT_DURATION_S = 2.0
SUBNET_TRANSIENT_MAX_SPEED_MPS = 8.5


@dataclass(frozen=True)
class Phase:
    label: str
    duration_s: float
    speed_target_mps: float
    steering_rad: float = 0.0
    throttle_mode: str = "speed_hold"
    throttle_norm: float | None = None
    reach_speed_target: bool = False
    validate_samples: bool = False
    validate_speed: bool = True
    validate_steering: bool = True
    settle_before_probe: bool = False
    probe_race_domain: bool = False
    throttle_profile: str | None = None
    throttle_start_norm: float | None = None
    throttle_end_norm: float | None = None
    throttle_stimulus_delay_s: float = 0.0
    throttle_ramp_duration_s: float = 0.0
    condition_pair_id: str | None = None
    steering_profile: str | None = None
    steering_amplitude_rad: float | None = None
    steering_frequency_hz: float | None = None
    steering_frequencies_hz: tuple[float, ...] = ()
    steering_phases_rad: tuple[float, ...] = ()
    steering_waypoints: tuple[tuple[float, float], ...] = ()
    steering_prbs_levels_normalized: tuple[float, ...] = ()
    steering_dwell_s: float = 0.0
    require_speed_target_match: bool = False


def _slew_probe_command(phase: Phase, elapsed_s: float) -> float:
    if phase.throttle_start_norm is None or phase.throttle_end_norm is None:
        raise ValueError("slew probe requires start and end throttle commands")
    if phase.throttle_profile not in ("ramp", "step"):
        raise ValueError("slew probe profile must be ramp or step")
    if elapsed_s < phase.throttle_stimulus_delay_s:
        return phase.throttle_start_norm
    stimulus_elapsed = elapsed_s - phase.throttle_stimulus_delay_s
    if phase.throttle_profile == "step":
        return (phase.throttle_start_norm
                if stimulus_elapsed < PERIOD_SEC else phase.throttle_end_norm)
    if phase.throttle_ramp_duration_s <= 0.0:
        raise ValueError("ramp duration must be positive")
    fraction = min(1.0, stimulus_elapsed / phase.throttle_ramp_duration_s)
    return (phase.throttle_start_norm + fraction
            * (phase.throttle_end_norm - phase.throttle_start_norm))


def _phase_steering_command(phase: Phase, elapsed_s: float) -> float:
    """Evaluate the stored deterministic dynamic steering command."""
    profile = phase.steering_profile
    if profile is None:
        return phase.steering_rad
    amplitude = phase.steering_amplitude_rad
    if amplitude is None or not 0.0 <= amplitude <= MAX_STEERING_RAD:
        raise ValueError("dynamic steering requires an in-range amplitude")
    elapsed = max(0.0, elapsed_s)
    if profile == "multisine":
        if (not phase.steering_frequencies_hz
                or len(phase.steering_frequencies_hz)
                != len(phase.steering_phases_rad)):
            raise ValueError("multisine steering requires matched frequencies/phases")
        wave = sum(math.sin(math.tau * frequency * elapsed + phase_offset)
                   for frequency, phase_offset in zip(
                       phase.steering_frequencies_hz,
                       phase.steering_phases_rad))
        wave /= len(phase.steering_frequencies_hz)
        wave *= min(1.0, elapsed / 0.25)
    elif profile == "triangle":
        if phase.steering_frequency_hz is None or phase.steering_frequency_hz <= 0.0:
            raise ValueError("triangle steering requires a positive frequency")
        cycle = (elapsed * phase.steering_frequency_hz + 0.25) % 1.0
        wave = (1.0 - 4.0 * abs(cycle - 0.5)) * min(1.0, elapsed / 0.25)
    elif profile == "prbs":
        if phase.steering_dwell_s < PRBS_MIN_DWELL_S:
            raise ValueError("piecewise steering dwell is too short")
        levels = phase.steering_prbs_levels_normalized
        if not levels:
            raise ValueError("PRBS steering requires stored levels")
        ramp_s = 0.25
        active_s = elapsed - ramp_s
        if active_s < 0.0:
            wave = levels[0] * min(1.0, elapsed / ramp_s)
        elif active_s < len(levels) * phase.steering_dwell_s:
            index = min(int(active_s / phase.steering_dwell_s), len(levels) - 1)
            wave = levels[index]
        else:
            tail_s = active_s - len(levels) * phase.steering_dwell_s
            wave = levels[-1] * max(0.0, 1.0 - tail_s / ramp_s)
    elif profile == "waypoints":
        points = phase.steering_waypoints
        if len(points) < 2:
            raise ValueError("waypoint steering requires at least two points")
        if elapsed <= points[0][0]:
            return points[0][1]
        for (t0, v0), (t1, v1) in zip(points, points[1:]):
            if elapsed <= t1:
                fraction = (elapsed - t0) / (t1 - t0)
                return v0 + fraction * (v1 - v0)
        return points[-1][1]
    else:
        raise ValueError(f"unknown dynamic steering profile: {profile}")
    return amplitude * max(-1.0, min(1.0, wave))


def _nominal_feedforward(speed_mps: float) -> float:
    if speed_mps <= SPEED_TARGETS_MPS[0]:
        return THROTTLE_FEEDFORWARD[0]
    for index in range(1, len(SPEED_TARGETS_MPS)):
        if speed_mps <= SPEED_TARGETS_MPS[index]:
            low_speed, high_speed = SPEED_TARGETS_MPS[index - 1:index + 1]
            low_throttle, high_throttle = THROTTLE_FEEDFORWARD[index - 1:index + 1]
            fraction = (speed_mps - low_speed) / (high_speed - low_speed)
            return low_throttle + fraction * (high_throttle - low_throttle)
    return THROTTLE_FEEDFORWARD[-1]


def _steering_waypoints(
        values: tuple[float, ...], sign: int, transition_s: float = 0.15,
        dwell_s: float = 0.35) -> tuple[tuple[float, float], ...]:
    points = [(0.0, 0.0)]
    elapsed = 0.0
    for value in values:
        elapsed += transition_s
        points.append((elapsed, sign * value))
        elapsed += dwell_s
        points.append((elapsed, sign * value))
    return tuple(points)


def _dynamic_coupled_manoeuvre_phases(condition) -> list[Phase]:
    speed = condition.target_speed_mps
    magnitude = condition.max_steering_rad
    sign = condition.first_turn_sign
    nominal = race_domain_feedforward(speed, _nominal_feedforward(speed))
    throttle_up = min(MAX_THROTTLE, nominal + 0.03)
    throttle_down = max(0.0, nominal - 0.05)
    condition_id = condition.condition_id

    def dynamic(label: str, duration: float, *, profile: str,
                amplitude: float, frequency: float | None = None,
                frequencies: tuple[float, ...] = (),
                phases: tuple[float, ...] = (),
                waypoints: tuple[tuple[float, float], ...] = (),
                prbs_levels: tuple[float, ...] = (),
                dwell_s: float = 0.0) -> Phase:
        return Phase(
            f"coupled_{condition_id}_{label}", duration, speed,
            throttle_mode="race_domain_hold", validate_samples=True,
            validate_speed=False, validate_steering=False,
            condition_pair_id=condition_id,
            steering_profile=profile,
            steering_amplitude_rad=amplitude,
            steering_frequency_hz=frequency,
            steering_frequencies_hz=frequencies,
            steering_phases_rad=phases,
            steering_waypoints=waypoints,
            steering_prbs_levels_normalized=prbs_levels,
            steering_dwell_s=dwell_s,
        )

    def throttle_ramp(label: str, start: float, end: float,
                      steering_amplitude: float) -> Phase:
        return Phase(
            f"coupled_{condition_id}_{label}", 1.50, speed,
            throttle_mode="slew_probe", validate_samples=True,
            validate_speed=False, validate_steering=False,
            throttle_profile="ramp", throttle_start_norm=start,
            throttle_end_norm=end, throttle_stimulus_delay_s=0.15,
            throttle_ramp_duration_s=0.60,
            condition_pair_id=condition_id,
            steering_profile="triangle",
            steering_amplitude_rad=steering_amplitude,
            steering_frequency_hz=condition.triangle_frequency_hz,
        )

    triangle_values = (0.0, 0.35 * magnitude, 0.75 * magnitude,
                       magnitude, 0.60 * magnitude, 0.0,
                       -0.35 * magnitude, -0.75 * magnitude,
                       -magnitude, -0.60 * magnitude, 0.0)
    turn_waypoints = _steering_waypoints(
        triangle_values, sign, transition_s=0.20, dwell_s=0.40)
    turn_phase = dynamic(
        "turn_in_unwind_reversal", turn_waypoints[-1][0],
        profile="waypoints", amplitude=magnitude,
        waypoints=turn_waypoints)
    multisine = dynamic(
        "multisine_steering", 12.0, profile="multisine",
        amplitude=magnitude,
        frequencies=DYNAMIC_STEERING_FREQUENCIES_HZ,
        phases=condition.multisine_phase_rad)
    triangle = dynamic(
        "triangular_steering", 8.0, profile="triangle",
        amplitude=0.80 * magnitude,
        frequency=condition.triangle_frequency_hz)
    prbs = dynamic(
        "random_piecewise_steering",
        0.25 + len(condition.prbs_levels_normalized) * PRBS_MIN_DWELL_S + 0.25,
        profile="prbs", amplitude=magnitude,
        prbs_levels=condition.prbs_levels_normalized,
        dwell_s=PRBS_MIN_DWELL_S)
    pickup = throttle_ramp(
        "steering_throttle_pickup", nominal, throttle_up,
        0.55 * magnitude)
    reduction = throttle_ramp(
        "steering_throttle_reduction", throttle_up, throttle_down,
        0.55 * magnitude)
    brake_waypoints = _steering_waypoints(
        (0.0, sign * min(0.10, 0.40 * magnitude), 0.0),
        1, transition_s=0.25, dwell_s=0.25)
    active_brake = Phase(
        f"coupled_{condition_id}_steering_active_brake",
        brake_waypoints[-1][0], speed,
        throttle_mode="fixed", throttle_norm=0.0,
        validate_samples=True, validate_speed=False,
        validate_steering=False, condition_pair_id=condition_id,
        steering_profile="waypoints",
        steering_amplitude_rad=max(abs(value) for _, value in brake_waypoints),
        steering_waypoints=brake_waypoints)
    release_waypoints = ((0.0, sign * min(0.08, 0.30 * magnitude)),
                         (0.30, 0.0), (1.50, 0.0))
    brake_release = Phase(
        f"coupled_{condition_id}_brake_release_unwind", 1.50, speed,
        throttle_mode="slew_probe", validate_samples=True,
        validate_speed=False, validate_steering=False,
        throttle_profile="ramp", throttle_start_norm=0.0,
        throttle_end_norm=nominal, throttle_stimulus_delay_s=0.15,
        throttle_ramp_duration_s=0.60,
        condition_pair_id=condition_id,
        steering_profile="waypoints",
        steering_amplitude_rad=max(abs(value) for _, value in release_waypoints),
        steering_waypoints=release_waypoints)

    if condition.speed_band == "5-7":
        return [multisine, triangle, prbs, turn_phase, pickup, reduction,
                active_brake, brake_release]
    if condition.speed_band == "7-9":
        return [multisine, triangle, prbs, turn_phase, pickup, reduction,
                active_brake, brake_release]

    steering_levels = (
        FRONTIER_11MPS_ANGLES_RAD
        if math.isclose(speed, 11.1) else FRONTIER_SWEEP_ANGLES_RAD)
    sweep_values = ((0.0, *steering_levels, 0.0,
                     *tuple(-value for value in steering_levels), 0.0))
    sweep_waypoints = _steering_waypoints(
        sweep_values, sign, transition_s=0.12, dwell_s=0.28)
    mixed_order = (0.0, 0.12, 0.04, 0.16, 0.0,
                   -0.12, -0.04, -0.16, 0.0)
    mixed_waypoints = _steering_waypoints(
        mixed_order, sign, transition_s=0.14, dwell_s=0.36)
    reversal_waypoints = _steering_waypoints(
        (0.0, 0.08, -0.08, 0.08, 0.0), sign,
        transition_s=0.15, dwell_s=0.35)
    return [
        dynamic("frontier_sweep_both_signs", sweep_waypoints[-1][0],
                profile="waypoints", amplitude=magnitude,
                waypoints=sweep_waypoints),
        dynamic("frontier_mixed_order", mixed_waypoints[-1][0],
                profile="waypoints", amplitude=magnitude,
                waypoints=mixed_waypoints),
        dynamic("frontier_reversal_008", reversal_waypoints[-1][0],
                profile="waypoints", amplitude=0.08,
                waypoints=reversal_waypoints),
        throttle_ramp("frontier_turnin_throttle_reduction",
                      nominal, throttle_down, min(0.12, magnitude)),
        active_brake,
        brake_release,
    ]


def _subnet_highsteer_transient_phases(seed: int) -> list[Phase]:
    """Build reset-isolated C4 transients in the measured 7.5 m/s envelope."""
    rng = random.Random(seed)
    speed = SUBNET_TRANSIENT_SPEED_MPS
    baseline = _nominal_feedforward(speed)
    # A prior live probe at +0.05 crossed the unchanged 8.5 m/s test guard.
    # Keep an independent acceleration stimulus, but reduce its amplitude.
    pickup = min(MAX_THROTTLE, baseline + 0.02)
    reduced = max(0.06, baseline - 0.10)
    conditions = [
        (angle, sign, manoeuvre)
        for angle in SUBNET_TRANSIENT_STEERING_RAD
        for sign in (-1, 1)
        for manoeuvre in SUBNET_TRANSIENT_MANOEUVRES
    ]
    rng.shuffle(conditions)

    phases: list[Phase] = []
    for index, (angle, sign, manoeuvre) in enumerate(conditions):
        condition_id = f"c{index:02d}_s{speed:.1f}_d{sign:+d}_a{angle:.2f}_{manoeuvre}"
        phases.extend((
            Phase(
                f"approach_subnet_{condition_id}", 12.0, speed,
                throttle_mode="race_domain_approach",
                reach_speed_target=True,
                validate_samples=True,
                validate_speed=False, validate_steering=False,
                condition_pair_id=condition_id,
            ),
            Phase(
                f"settle_subnet_{condition_id}", 1.25, speed,
                throttle_mode="race_domain_hold",
                validate_samples=True,
                validate_speed=False, validate_steering=False,
                condition_pair_id=condition_id,
            ),
        ))

        is_unwind = manoeuvre.startswith("unwind_")
        steering = sign * angle
        if is_unwind:
            # Establish the matched high-steer state before the unwind. This
            # is a measured precondition, not an assumed instantaneous state.
            phases.append(Phase(
                f"precondition_subnet_{condition_id}", 1.25, speed,
                steering_rad=steering,
                throttle_mode="fixed", throttle_norm=baseline,
                validate_samples=True,
                validate_speed=False, validate_steering=False,
                condition_pair_id=condition_id,
            ))

        if manoeuvre.endswith("acceleration"):
            throttle_end = pickup
        elif manoeuvre.endswith("throttle_reduction"):
            # Zero throttle is active brake torque in this simulator, so this
            # distinct reduction condition remains at positive throttle.
            throttle_end = reduced
        else:
            # The only available brake command is zero throttle; there is no
            # separate coast input in this actuator interface.
            throttle_end = 0.0

        if is_unwind:
            waypoints = ((0.0, steering), (0.30, 0.0),
                         (SUBNET_TRANSIENT_DURATION_S, 0.0))
        else:
            waypoints = ((0.0, 0.0), (0.30, steering),
                         (SUBNET_TRANSIENT_DURATION_S, steering))
        phases.append(Phase(
            f"subnet_{condition_id}", SUBNET_TRANSIENT_DURATION_S, speed,
            throttle_mode="slew_probe",
            validate_samples=True,
            validate_speed=False, validate_steering=False,
            throttle_profile="ramp",
            throttle_start_norm=baseline,
            throttle_end_norm=throttle_end,
            throttle_stimulus_delay_s=0.15,
            throttle_ramp_duration_s=0.60,
            condition_pair_id=condition_id,
            steering_profile="waypoints",
            steering_amplitude_rad=angle,
            steering_waypoints=waypoints,
        ))
    return phases


def _race_domain_swerve_throttle_slew_phases(
        seed: int,
        speed_steering_rad: tuple[tuple[float, tuple[float, ...]], ...]
        = SWERVE_THROTTLE_SLEW_SPEED_STEERING_RAD,
        reset_before_each_probe: bool = False,
        repetitions: int = 1,
        align_lowangle_unwind: bool = False) -> list[Phase]:
    """Pair throttle shapes during matched bidirectional s-curves.

    ``reset_before_each_probe`` gives both members the same spawn and speed
    approach, avoiding carry-over when a swerve leaves lateral/yaw motion.
    """
    if repetitions < 1:
        raise ValueError("throttle-slew repetitions must be positive")
    rng = random.Random(seed)
    speeds = list(speed_steering_rad)
    rng.shuffle(speeds)
    phases: list[Phase] = []
    for speed, steering_levels in speeds:
        nominal = race_domain_feedforward(speed, _nominal_feedforward(speed))
        delta = min(
            SWERVE_THROTTLE_SLEW_DELTA_NORM,
            nominal,
            MAX_THROTTLE - nominal,
        )
        if delta <= 0.0:
            raise ValueError(f"no symmetric throttle headroom at {speed:g} m/s")
        conditions = [
            (steering, turn_sign, throttle_sign, repetition)
            for steering in steering_levels
            for turn_sign in (-1.0, 1.0)
            # At the 9.5/11.1 m/s frontier, an upward step immediately reaches
            # the established 11.2 m/s governor and would compare the safety
            # override rather than the requested throttle shape. Existing
            # throttle-surface captures already cover the high-throttle side;
            # this swerve study tests bounded reductions there.
            for throttle_sign in ((-1.0,) if speed >= 9.5 else (-1.0, 1.0))
            for repetition in range(1, repetitions + 1)
        ]
        rng.shuffle(conditions)
        for steering, turn_sign, throttle_sign, repetition in conditions:
            repetition_suffix = (f"_rep{repetition:02d}"
                                 if repetitions > 1 else "")
            pair_id = (
                f"v{speed:.1f}_a{steering:.3f}_"
                f"turn{turn_sign:+.0f}_throttle{throttle_sign:+.0f}"
                f"{repetition_suffix}")
            if not reset_before_each_probe:
                phases.append(Phase(
                    f"approach_swerve_pair_{pair_id}",
                    SWERVE_THROTTLE_SLEW_APPROACH_S[speed],
                    speed,
                    throttle_mode="race_domain_approach",
                    reach_speed_target=True,
                    condition_pair_id=pair_id,
                ))
            shapes = ["ramp", "step"]
            rng.shuffle(shapes)
            throttle_end = nominal + throttle_sign * delta
            if align_lowangle_unwind:
                # In the observed step response, speed enters 8.5–9.0 m/s
                # around 0.73 s after the throttle cut. Hold the unwind near
                # +/-0.025 rad during that exact interval so the training
                # sequences contain the previously missing speed/angle/phase
                # cell, rather than passing it only after speed has fallen.
                waypoints = (
                    (0.00, 0.00),
                    (0.30, turn_sign * steering),
                    (0.60, turn_sign * steering),
                    (0.70, turn_sign * 0.030),
                    (0.80, turn_sign * 0.015),
                    (0.90, 0.00),
                    (1.15, 0.00),
                    (1.40, -turn_sign * steering),
                    (1.60, -turn_sign * steering),
                    (SWERVE_THROTTLE_SLEW_PHASE_S, 0.00),
                )
            else:
                waypoints = (
                    (0.00, 0.00),
                    (0.30, turn_sign * steering),
                    (0.60, turn_sign * steering),
                    (0.95, 0.00),
                    (1.25, -turn_sign * steering),
                    (1.60, -turn_sign * steering),
                    (SWERVE_THROTTLE_SLEW_PHASE_S, 0.00),
                )
            for shape in shapes:
                if reset_before_each_probe:
                    # The legacy profiles collect the two paired treatments
                    # back-to-back and require the prior swerve to settle.
                    # The low-steer validation is specifically intended to
                    # characterize potentially unrecoverable transients, so
                    # reset to the same spawn and rebuild target speed before
                    # each member of the ramp/step pair.
                    phases.append(Phase(
                        f"approach_swerve_pair_{pair_id}_{shape}",
                        SWERVE_THROTTLE_SLEW_APPROACH_S[speed],
                        speed,
                        throttle_mode="race_domain_approach",
                        reach_speed_target=True,
                        condition_pair_id=pair_id,
                    ))
                phases.append(Phase(
                    f"swerve_slew_{pair_id}_{shape}",
                    SWERVE_THROTTLE_SLEW_PHASE_S,
                    speed,
                    throttle_mode="slew_probe",
                    validate_samples=True,
                    validate_speed=False,
                    validate_steering=False,
                    settle_before_probe=True,
                    throttle_profile=shape,
                    throttle_start_norm=nominal,
                    throttle_end_norm=throttle_end,
                    throttle_stimulus_delay_s=0.60,
                    throttle_ramp_duration_s=0.30,
                    condition_pair_id=pair_id,
                    steering_profile="waypoints",
                    steering_amplitude_rad=steering,
                    steering_waypoints=waypoints,
                ))
    return phases


def _highsteer_throttle_rate_sweep_phases(
        seed: int, throttle_delta_norm: float,
        highsteer_validation: bool = False) -> list[Phase]:
    """Compare positive-throttle ramps with reset-matched steps at 8 m/s."""
    speed = SWERVE_THROTTLE_RATE_SWEEP_SPEED_MPS
    nominal = race_domain_feedforward(speed, _nominal_feedforward(speed))
    if (not math.isfinite(throttle_delta_norm)
            or throttle_delta_norm <= 0.0
            or throttle_delta_norm > MAX_THROTTLE - nominal):
        raise ValueError(
            f"requested throttle delta exceeds available headroom at {speed:g} m/s")
    delta = throttle_delta_norm

    rng = random.Random(seed)
    steering_angles = (
        (0.42,) if highsteer_validation
        else SWERVE_THROTTLE_RATE_SWEEP_STEERING_RAD)
    ramp_durations = (
        SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_RAMP_DURATIONS_S
        if highsteer_validation
        else SWERVE_THROTTLE_RATE_SWEEP_RAMP_DURATIONS_S)
    conditions = [
        (angle, turn, ramp_duration)
        for angle in steering_angles
        for turn in (-1.0, 1.0)
        for ramp_duration in ramp_durations
    ]
    rng.shuffle(conditions)
    phases: list[Phase] = []
    for angle, turn, ramp_duration in conditions:
        pair_id = (
            f"v{speed:.1f}_a{angle:.3f}_turn{turn:+.0f}_up_"
            f"d{delta:.3f}_"
            f"ramp{ramp_duration:.2f}")
        waypoints = (
            (0.00, 0.00),
            (0.30, turn * angle),
            (0.60, turn * angle),
            (0.95, 0.00),
            (1.25, -turn * angle),
            (1.60, -turn * angle),
            (SWERVE_THROTTLE_SLEW_PHASE_S, 0.00),
        )
        shapes = ["ramp", "step"]
        rng.shuffle(shapes)
        for shape in shapes:
            # Reset and rebuild the same speed/state for each paired member.
            phases.append(Phase(
                f"approach_swerve_pair_{pair_id}_{shape}",
                10.0,
                speed,
                throttle_mode="race_domain_approach",
                reach_speed_target=True,
                condition_pair_id=pair_id,
            ))
            phases.append(Phase(
                f"swerve_slew_{pair_id}_{shape}",
                SWERVE_THROTTLE_SLEW_PHASE_S,
                speed,
                throttle_mode="slew_probe",
                validate_samples=True,
                validate_speed=False,
                validate_steering=False,
                settle_before_probe=True,
                throttle_profile=shape,
                throttle_start_norm=nominal,
                throttle_end_norm=nominal + delta,
                throttle_stimulus_delay_s=0.60,
                throttle_ramp_duration_s=ramp_duration,
                condition_pair_id=pair_id,
                steering_profile="waypoints",
                steering_amplitude_rad=angle,
                steering_waypoints=waypoints,
            ))
    return phases


def _highsteer_throttle_rate_factorial_phases(
        seed: int, speed: float = SWERVE_THROTTLE_RATE_SWEEP_SPEED_MPS
        ) -> list[Phase]:
    """Cross throttle-change size with matched rise rates at high steering."""
    nominal = race_domain_feedforward(speed, _nominal_feedforward(speed))
    conditions = [
        (angle, turn, delta, delta / rise_rate)
        for angle in SWERVE_THROTTLE_RATE_SWEEP_STEERING_RAD
        for turn in (-1.0, 1.0)
        for delta in SWERVE_THROTTLE_RATE_FACTORIAL_DELTAS_NORM
        for rise_rate in SWERVE_THROTTLE_RATE_FACTORIAL_RATES_NORM_PER_SEC
    ]
    if any(nominal + delta > MAX_THROTTLE for _angle, _turn, delta, _duration
           in conditions):
        raise ValueError("factorial throttle target exceeds the command limit")

    rng = random.Random(seed)
    rng.shuffle(conditions)
    phases: list[Phase] = []
    for angle, turn, delta, ramp_duration in conditions:
        rise_rate = delta / ramp_duration
        pair_id = (
            f"v{speed:.1f}_a{angle:.3f}_turn{turn:+.0f}_up_"
            f"d{delta:.3f}_rate{rise_rate:.3f}")
        waypoints = (
            (0.00, 0.00),
            (0.30, turn * angle),
            (0.60, turn * angle),
            (0.95, 0.00),
            (1.25, -turn * angle),
            (1.60, -turn * angle),
            (SWERVE_THROTTLE_SLEW_PHASE_S, 0.00),
        )
        shapes = ["ramp", "step"]
        rng.shuffle(shapes)
        for shape in shapes:
            phases.append(Phase(
                f"approach_swerve_pair_{pair_id}_{shape}",
                10.0,
                speed,
                throttle_mode="race_domain_approach",
                reach_speed_target=True,
                condition_pair_id=pair_id,
            ))
            phases.append(Phase(
                f"swerve_slew_{pair_id}_{shape}",
                SWERVE_THROTTLE_SLEW_PHASE_S,
                speed,
                throttle_mode="slew_probe",
                validate_samples=True,
                validate_speed=False,
                validate_steering=False,
                settle_before_probe=True,
                throttle_profile=shape,
                throttle_start_norm=nominal,
                throttle_end_norm=nominal + delta,
                throttle_stimulus_delay_s=0.60,
                throttle_ramp_duration_s=ramp_duration,
                condition_pair_id=pair_id,
                steering_profile="waypoints",
                steering_amplitude_rad=angle,
                steering_waypoints=waypoints,
            ))
    return phases


def _frontier_throttle_up_swerve_phases(seed: int) -> list[Phase]:
    """Pair positive throttle steps/ramps in the missing 9–10 m/s swerve cells."""
    rng = random.Random(seed)
    conditions = [
        (speed, steering, turn, duration)
        for speed, steering_levels in SWERVE_THROTTLE_RATE_FRONTIER_UP_SPEED_STEERING_RAD
        for steering in steering_levels
        for turn in (-1.0, 1.0)
        for duration in SWERVE_THROTTLE_RATE_FRONTIER_UP_RAMP_DURATIONS_S
    ]
    rng.shuffle(conditions)
    phases: list[Phase] = []
    for speed, steering, turn, ramp_duration in conditions:
        nominal = race_domain_feedforward(speed, _nominal_feedforward(speed))
        final_throttle = nominal + SWERVE_THROTTLE_RATE_FRONTIER_UP_DELTA_NORM
        if final_throttle >= MAX_THROTTLE:
            raise ValueError(
                f"frontier throttle target exceeds headroom at {speed:g} m/s")
        rate = SWERVE_THROTTLE_RATE_FRONTIER_UP_DELTA_NORM / ramp_duration
        pair_id = (
            f"v{speed:.1f}_a{steering:.3f}_turn{turn:+.0f}_up_"
            f"d{SWERVE_THROTTLE_RATE_FRONTIER_UP_DELTA_NORM:.3f}_"
            f"rate{rate:.3f}")
        waypoints = (
            (0.00, 0.00),
            (0.30, turn * steering),
            (0.60, turn * steering),
            (0.95, 0.00),
            (1.25, -turn * steering),
            (1.60, -turn * steering),
            (SWERVE_THROTTLE_SLEW_PHASE_S, 0.00),
        )
        shapes = ["ramp", "step"]
        rng.shuffle(shapes)
        for shape in shapes:
            # Each pair member starts from a fresh reset and the same measured
            # speed approach. This prevents the first swerve contaminating the
            # matched initial wheel/body slip state of the second.
            phases.append(Phase(
                f"approach_swerve_pair_{pair_id}_{shape}",
                SWERVE_THROTTLE_SLEW_APPROACH_S[speed],
                speed,
                throttle_mode="race_domain_approach",
                reach_speed_target=True,
                condition_pair_id=pair_id,
            ))
            phases.append(Phase(
                f"swerve_slew_{pair_id}_{shape}",
                SWERVE_THROTTLE_SLEW_PHASE_S,
                speed,
                throttle_mode="slew_probe",
                validate_samples=True,
                validate_speed=False,
                validate_steering=False,
                settle_before_probe=True,
                throttle_profile=shape,
                throttle_start_norm=nominal,
                throttle_end_norm=final_throttle,
                throttle_stimulus_delay_s=0.60,
                throttle_ramp_duration_s=ramp_duration,
                condition_pair_id=pair_id,
                steering_profile="waypoints",
                steering_amplitude_rad=steering,
                steering_waypoints=waypoints,
            ))
    return phases


def _race_domain_throttle_rate_phases(seed: int, speed: float) -> list[Phase]:
    """Measure matched throttle slew in the practiced speed/steering range."""
    if speed not in SWERVE_THROTTLE_RATE_RACE_DOMAIN_SPEEDS_MPS:
        raise ValueError(
            "race-domain throttle-rate speed must be one of "
            f"{SWERVE_THROTTLE_RATE_RACE_DOMAIN_SPEEDS_MPS}")
    nominal = race_domain_feedforward(speed, _nominal_feedforward(speed))
    conditions = [
        (steering, turn, delta, duration)
        for steering in SWERVE_THROTTLE_RATE_RACE_DOMAIN_STEERING_RAD
        for turn in (-1.0, 1.0)
        for delta in SWERVE_THROTTLE_RATE_RACE_DOMAIN_DELTAS_NORM
        for duration in SWERVE_THROTTLE_RATE_RACE_DOMAIN_RAMP_DURATIONS_S
    ]
    if any(nominal + delta > MAX_THROTTLE
           for _steering, _turn, delta, _duration in conditions):
        raise ValueError(f"race-domain throttle target exceeds limit at {speed:g} m/s")

    rng = random.Random(seed)
    rng.shuffle(conditions)
    phases: list[Phase] = []
    for steering, turn, delta, ramp_duration in conditions:
        rate = delta / ramp_duration
        pair_id = (
            f"v{speed:.1f}_a{steering:.3f}_turn{turn:+.0f}_up_"
            f"d{delta:.3f}_rate{rate:.3f}")
        waypoints = (
            (0.00, 0.00),
            (0.30, turn * steering),
            (0.60, turn * steering),
            (0.95, 0.00),
            (1.25, -turn * steering),
            (1.60, -turn * steering),
            (SWERVE_THROTTLE_SLEW_PHASE_S, 0.00),
        )
        shapes = ["ramp", "step"]
        rng.shuffle(shapes)
        for shape in shapes:
            phases.append(Phase(
                f"approach_swerve_pair_{pair_id}_{shape}",
                SWERVE_THROTTLE_SLEW_APPROACH_S[speed],
                speed,
                throttle_mode="race_domain_approach",
                reach_speed_target=True,
                condition_pair_id=pair_id,
            ))
            phases.append(Phase(
                f"swerve_slew_{pair_id}_{shape}",
                SWERVE_THROTTLE_SLEW_PHASE_S,
                speed,
                throttle_mode="slew_probe",
                validate_samples=True,
                validate_speed=False,
                validate_steering=False,
                settle_before_probe=True,
                throttle_profile=shape,
                throttle_start_norm=nominal,
                throttle_end_norm=nominal + delta,
                throttle_stimulus_delay_s=0.60,
                throttle_ramp_duration_s=ramp_duration,
                condition_pair_id=pair_id,
                steering_profile="waypoints",
                steering_amplitude_rad=steering,
                steering_waypoints=waypoints,
            ))
    return phases


def build_schedule(seed: int, profile: str = "high_angle_boundary",
                  transition_speed_mps: float = 4.5,
                  throttle_rate_sweep_delta_norm: float =
                  SWERVE_THROTTLE_SLEW_DELTA_NORM) -> list[Phase]:
    """Build repeatable speed blocks and signed steering probes."""
    rng = random.Random(seed)
    phases: list[Phase] = []
    if profile == "high_angle_boundary":
        target_speed = 2.2
        phases.extend((
            Phase("approach_2.2mps", 8.0, target_speed, throttle_mode="approach",
                  reach_speed_target=True),
            Phase("settle_2.2mps", 1.0, target_speed),
        ))
        conditions = [
            (sign * amplitude, amplitude)
            for amplitude in (0.42, 0.46, 0.50)
            for sign in (-1.0, 1.0)
        ]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering, amplitude in ordered:
                phases.append(Phase(
                    f"boundary_r{repetition}_{steering:+.2f}rad",
                    2.0,
                    target_speed,
                    steering_rad=steering,
                    validate_samples=True,
                ))
        return phases
    if profile == "throttle_slew_pair":
        target_speed = transition_speed_mps
        if target_speed not in (4.5, 6.5):
            raise ValueError("throttle_slew_pair supports only 4.5 or 6.5 m/s")
        phases.extend((
            Phase(f"approach_{target_speed:.1f}mps", 8.0, target_speed,
                  throttle_mode="approach", reach_speed_target=True),
            Phase(f"settle_{target_speed:.1f}mps", 1.0, target_speed),
        ))
        base_throttle = THROTTLE_FEEDFORWARD[SPEED_TARGETS_MPS.index(target_speed)]
        conditions = [
            (sign * steering, delta_sign)
            for steering in THROTTLE_SLEW_STEERING_RAD
            for sign in (-1.0, 1.0)
            for delta_sign in (-1.0, 1.0)
        ]
        for repetition in range(1, 4):
            ordered_conditions = conditions.copy()
            rng.shuffle(ordered_conditions)
            for steering_rad, delta_sign in ordered_conditions:
                pair_id = (
                    f"r{repetition}_v{target_speed:.1f}_"
                    f"s{steering_rad:+.2f}_d{delta_sign:+.0f}")
                shapes = ["ramp", "step"]
                rng.shuffle(shapes)
                final_throttle = base_throttle + delta_sign * THROTTLE_SLEW_DELTA_NORM
                for shape in shapes:
                    phases.append(Phase(
                        f"throttle_slew_{pair_id}_{shape}",
                        THROTTLE_SLEW_PHASE_S,
                        target_speed,
                        steering_rad=steering_rad,
                        throttle_mode="slew_probe",
                        validate_samples=True,
                        validate_speed=False,
                        settle_before_probe=True,
                        throttle_profile=shape,
                        throttle_start_norm=base_throttle,
                        throttle_end_norm=final_throttle,
                        throttle_stimulus_delay_s=THROTTLE_SLEW_STIMULUS_DELAY_S,
                        throttle_ramp_duration_s=THROTTLE_SLEW_RAMP_S,
                        condition_pair_id=pair_id,
                    ))
        return phases
    if profile == "full_input_excitation":
        # Randomized factorial actuator excitation. Each replicate contains
        # every signed steering level (including exact zero) crossed with all
        # throttle/braking commands; varied dwell time and order expose both
        # static nonlinear surfaces and transient/history effects. Speed is
        # allowed to evolve freely, subject only to the safety governor.
        dwell_s = (1.3, 1.8, 2.5)
        conditions = [
            (steering, throttle)
            for steering in FULL_INPUT_STEERING_LEVELS_RAD
            for throttle in FULL_INPUT_THROTTLE_LEVELS
        ]
        for repetition, duration in enumerate(dwell_s, start=1):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for index, (steering, throttle) in enumerate(ordered, start=1):
                phases.append(Phase(
                    f"excite_r{repetition}_{index:03d}_"
                    f"steer_{steering:+.2f}_throttle_{throttle:.2f}",
                    duration,
                    0.0,
                    steering_rad=steering,
                    throttle_mode="excitation",
                    throttle_norm=throttle,
                    validate_samples=True,
                    validate_speed=False,
                ))
        return phases
    if profile == "isolated_highsteer_75_long":
        # The prior validation run measured 7.7–7.9 m/s with steering up to
        # 0.524 rad, but each high-steer condition lasted only ~2.5 s and came
        # from one run. Hold the empirically observed feasible cells long
        # enough to support a 2 s context plus 5 s command-only evaluation.
        phases.extend((
            Phase("approach_7.5mps", 12.0, HIGH_STEER_VALIDATION_SPEED_MPS,
                  throttle_mode="approach", reach_speed_target=True),
            Phase("settle_7.5mps", 1.0, HIGH_STEER_VALIDATION_SPEED_MPS),
        ))
        phases.extend(Phase(
            block.label, block.duration_s, block.target_speed_mps,
            steering_rad=block.steering_rad,
            validate_samples=True,
            # Steering-induced speed loss and speed-hold throttle response are
            # measurements here, not reasons to discard a valid run.
            validate_speed=False,
            settle_before_probe=True,
        ) for block in build_high_steer_validation_plan(seed))
        return phases
    if profile == "isolated_highsteer_multispeed":
        # The single 7.5 m/s high-steer domain exposed a large yaw-model
        # mismatch, while the resulting global residual failed P0 transfer.
        # Cover the same steering grid at low, middle, and upper racing speed
        # so a regime-conditioned fit is supported by whole speed bands rather
        # than one operating point. Each probe is settled before measurement;
        # speed and signed-angle order are independently randomized.
        speeds = [2.5, 4.5, 6.5]
        rng.shuffle(speeds)
        angles = (0.20, 0.25, 0.30, 0.35, 0.42, 0.46, 0.50, 0.5236)
        for target_speed in speeds:
            phases.extend((
                Phase(f"approach_multispeed_v{target_speed:.1f}", 12.0,
                      target_speed, throttle_mode="approach",
                      reach_speed_target=True,
                      require_speed_target_match=True),
                Phase(f"settle_multispeed_v{target_speed:.1f}", 1.0,
                      target_speed),
            ))
            conditions = [0.0]
            conditions.extend(sign * angle for angle in angles
                              for sign in (-1.0, 1.0))
            rng.shuffle(conditions)
            phases.extend(Phase(
                f"multispeed_v{target_speed:.1f}_r1_steer_{steering:+.4f}",
                7.5,
                target_speed,
                steering_rad=steering,
                validate_samples=True,
                validate_speed=False,
                settle_before_probe=True,
            ) for steering in conditions)
        return phases
    if profile == "race_domain_dynamic_steering":
        # The static speed/steering surfaces already cover steady response.
        # This compact whole-run sequence adds matched-speed steering
        # reversals so the plant sees the transients it must recursively
        # predict when following a high-curvature raceline. The 7.5 m/s
        # maximum is the repeated full-lock condition established by the
        # held-out high-steer captures; this profile does not extrapolate
        # that steering demand to higher speed.
        steering_magnitudes = (0.15, 0.30, 0.42, 0.50, 0.5236)
        for target_speed in (4.5, 6.5, HIGH_STEER_VALIDATION_SPEED_MPS):
            phases.extend((
                Phase(f"approach_{target_speed:.1f}mps", 10.0,
                      target_speed, throttle_mode="approach",
                      reach_speed_target=True),
                Phase(f"settle_{target_speed:.1f}mps", 1.0, target_speed),
            ))
            first_sign = 1.0 if rng.getrandbits(1) else -1.0
            steering_trace = [0.0]
            steering_trace.extend(first_sign * angle
                                  for angle in steering_magnitudes)
            steering_trace.append(0.0)
            steering_trace.extend(-first_sign * angle
                                  for angle in steering_magnitudes)
            for index, steering in enumerate(steering_trace):
                phases.append(Phase(
                    f"dynamic_v{target_speed:.1f}_step{index:02d}_"
                    f"steer_{steering:+.4f}",
                    1.25,
                    target_speed,
                    steering_rad=steering,
                    validate_samples=True,
                    validate_steering=False,
                    # Speed-controller effort and speed loss under transient
                    # lateral demand are observed plant behavior, not a
                    # reason to discard a correctly recorded transition.
                    # Steering feedback is itself a predicted plant state;
                    # its lag is retained and checked in offline response
                    # analysis instead of a steady-state command-error gate.
                    validate_speed=False,
                ))
        return phases
    if profile in YAW_TRANSIENT_PROFILES:
        # Static holds already cover these angles. This fills the transient
        # turn-in/unwind/reversal gap exposed by the practice collision, with
        # a fresh spawn and matched speed for every signed condition. The new
        # yaw profile isolates the 3.5-4.0 m/s gap while leaving the established
        # low-speed schedule byte-for-byte unchanged.
        rng = random.Random(seed)
        if profile == YAW_FULLBAND_GAPFILL_PROFILE:
            conditions = [
                (repeat, speed, angle, sign)
                for repeat in range(1, YAW_FULLBAND_GAPFILL_REPEATS + 1)
                for speed, angle in YAW_FULLBAND_GAPFILL_POINTS
                for sign in (-1.0, 1.0)
            ]
            rng.shuffle(conditions)
            for repeat, speed, angle, sign in conditions:
                condition = (
                    f"r{repeat:02d}_v{speed:.2f}_a{angle:.4f}_turn{sign:+.0f}")
                phases.extend((
                    Phase(
                        f"approach_yawgap_{condition}", 10.0, speed,
                        throttle_mode="race_domain_approach",
                        reach_speed_target=True,
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"settle_yawgap_{condition}", 0.75, speed,
                        throttle_mode="race_domain_hold",
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"yawgap_{condition}", 2.75, speed,
                        throttle_mode="race_domain_hold",
                        validate_samples=True,
                        validate_speed=False,
                        validate_steering=False,
                        condition_pair_id=condition,
                        steering_profile="waypoints",
                        steering_amplitude_rad=angle,
                        steering_waypoints=(
                            (0.00, 0.0),
                            (0.20, sign * angle),
                            (0.70, sign * angle),
                            (0.95, 0.0),
                            (1.30, 0.0),
                            (1.45, -sign * angle),
                            (2.10, -sign * angle),
                            (2.35, 0.0),
                            (2.75, 0.0),
                        ),
                    ),
                ))
            return phases
        if profile == YAW_UNWIND_THROTTLE_PROFILE:
            conditions = [
                (repeat, mode, sign)
                for repeat in range(1, YAW_UNWIND_THROTTLE_REPEATS + 1)
                for mode in YAW_UNWIND_THROTTLE_MODES
                for sign in (-1.0, 1.0)
            ]
            rng.shuffle(conditions)
            for repeat, mode, sign in conditions:
                condition = f"r{repeat:02d}_{mode}_turn{sign:+.0f}"
                phases.extend((
                    Phase(
                        f"approach_yawbrake_{condition}", 10.0,
                        YAW_UNWIND_THROTTLE_SPEED_MPS,
                        throttle_mode="race_domain_approach",
                        reach_speed_target=True,
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"settle_yawbrake_{condition}", 0.75,
                        YAW_UNWIND_THROTTLE_SPEED_MPS,
                        throttle_mode="race_domain_hold",
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"yawbrake_{condition}", 4.0,
                        YAW_UNWIND_THROTTLE_SPEED_MPS,
                        throttle_mode="slew_probe",
                        throttle_start_norm=YAW_UNWIND_THROTTLE_START_NORM,
                        throttle_end_norm=YAW_UNWIND_THROTTLE_END_NORM,
                        throttle_profile=mode,
                        throttle_stimulus_delay_s=0.70,
                        throttle_ramp_duration_s=(0.30 if mode == "ramp" else 0.0),
                        validate_samples=True,
                        validate_speed=False,
                        validate_steering=False,
                        settle_before_probe=True,
                        probe_race_domain=True,
                        condition_pair_id=condition,
                        steering_profile="waypoints",
                        steering_amplitude_rad=YAW_UNWIND_THROTTLE_STEERING_RAD,
                        steering_waypoints=(
                            (0.00, 0.0),
                            (0.20, sign * YAW_UNWIND_THROTTLE_STEERING_RAD),
                            (0.45, sign * YAW_UNWIND_THROTTLE_STEERING_RAD),
                            (0.70, sign * 0.025),
                            (1.30, sign * 0.025),
                            (1.50, 0.0),
                            (1.70, 0.0),
                            (1.95, -sign * YAW_UNWIND_THROTTLE_STEERING_RAD),
                            (2.20, -sign * YAW_UNWIND_THROTTLE_STEERING_RAD),
                            (2.45, -sign * 0.025),
                            (3.05, -sign * 0.025),
                            (3.25, 0.0),
                            (4.00, 0.0),
                        ),
                    ),
                ))
            return phases
        if profile == YAW_LOW_ANGLE_RATE_PROFILE:
            if transition_speed_mps not in YAW_LOW_ANGLE_RATE_SPEEDS_MPS:
                raise ValueError(
                    f"{profile} supports only {YAW_LOW_ANGLE_RATE_SPEEDS_MPS}")
            conditions = [
                (repeat, angle, rate_name, ramp_s, sign)
                for repeat in range(1, YAW_LOW_ANGLE_RATE_REPEATS + 1)
                for angle in YAW_LOW_ANGLE_RATE_STEERING_RAD
                for rate_name, ramp_s in YAW_LOW_ANGLE_RATE_MODES
                for sign in (-1.0, 1.0)
            ]
            rng.shuffle(conditions)
            speed = transition_speed_mps
            for repeat, angle, rate_name, ramp_s, sign in conditions:
                condition = (
                    f"r{repeat:02d}_v{speed:.2f}_a{angle:.4f}_"
                    f"rate{rate_name}_turn{sign:+.0f}")
                if rate_name == "step":
                    turn_in, unwind, reversal = (
                        PERIOD_SEC, PERIOD_SEC, PERIOD_SEC)
                else:
                    turn_in, unwind, reversal = ramp_s, ramp_s, ramp_s
                turn_in_end = turn_in
                hold_end = 0.70
                unwind_end = hold_end + unwind
                neutral_end = 1.20
                reversal_end = neutral_end + reversal
                reverse_hold_end = 1.95
                return_end = reverse_hold_end + unwind
                phases.extend((
                    Phase(
                        f"approach_lowyaw_{condition}", 10.0, speed,
                        throttle_mode="race_domain_approach",
                        reach_speed_target=True,
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"settle_lowyaw_{condition}", 0.75, speed,
                        throttle_mode="race_domain_hold",
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"lowyaw_{condition}", 2.75, speed,
                        throttle_mode="race_domain_hold",
                        validate_samples=True,
                        validate_speed=False,
                        validate_steering=False,
                        condition_pair_id=condition,
                        steering_profile="waypoints",
                        steering_amplitude_rad=angle,
                        steering_waypoints=(
                            (0.00, 0.0),
                            (turn_in_end, sign * angle),
                            (hold_end, sign * angle),
                            (unwind_end, 0.0),
                            (neutral_end, 0.0),
                            (reversal_end, -sign * angle),
                            (reverse_hold_end, -sign * angle),
                            (return_end, 0.0),
                            (2.75, 0.0),
                        ),
                    ),
                ))
            return phases
        if profile in (YAW_ATLAS_INTERPOLATION_PROFILE,
                       YAW_ATLAS_OFFGRID_FINAL_PROFILE,
                       YAW_ATLAS_EXTRATREES_FINAL_PROFILE):
            points = {
                YAW_ATLAS_INTERPOLATION_PROFILE:
                    YAW_ATLAS_INTERPOLATION_POINTS,
                YAW_ATLAS_OFFGRID_FINAL_PROFILE:
                    YAW_ATLAS_OFFGRID_FINAL_POINTS,
                YAW_ATLAS_EXTRATREES_FINAL_PROFILE:
                    YAW_ATLAS_EXTRATREES_FINAL_POINTS,
            }[profile]
            conditions = [
                (repeat, speed, angle, sign)
                for repeat in range(1, YAW_ATLAS_INTERPOLATION_REPEATS + 1)
                for speed, angle in points
                for sign in (-1.0, 1.0)
            ]
            rng.shuffle(conditions)
            for repeat, speed, angle, sign in conditions:
                condition = (
                    f"r{repeat:02d}_v{speed:.2f}_a{angle:.4f}_turn{sign:+.0f}")
                phases.extend((
                    Phase(
                        f"approach_atlas_{condition}", 10.0, speed,
                        throttle_mode="race_domain_approach",
                        reach_speed_target=True,
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"settle_atlas_{condition}", 0.75, speed,
                        throttle_mode="race_domain_hold",
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"atlas_{condition}", 2.75, speed,
                        throttle_mode="race_domain_hold",
                        validate_samples=True,
                        validate_speed=False,
                        validate_steering=False,
                        condition_pair_id=condition,
                        steering_profile="waypoints",
                        steering_amplitude_rad=angle,
                        steering_waypoints=(
                            (0.00, 0.0),
                            (0.20, sign * angle),
                            (0.70, sign * angle),
                            (0.95, 0.0),
                            (1.30, 0.0),
                            (1.45, -sign * angle),
                            (2.10, -sign * angle),
                            (2.35, 0.0),
                            (2.75, 0.0),
                        ),
                    ),
                ))
            return phases
        yaw_targeted = profile == YAW_TRANSIENT_PROFILE
        speeds = list(YAW_TRANSIENT_SPEEDS_MPS if yaw_targeted
                      else LOW_SPEED_TRANSIENT_SPEEDS_MPS)
        steering_angles = (YAW_TRANSIENT_STEERING_RAD if yaw_targeted
                           else LOW_SPEED_TRANSIENT_STEERING_RAD)
        label_prefix = "yawdyn" if yaw_targeted else "lowdyn"
        rng.shuffle(speeds)
        for speed in speeds:
            conditions = [
                (angle, sign)
                for angle in steering_angles
                for sign in (-1, 1)
            ]
            rng.shuffle(conditions)
            for angle, sign in conditions:
                speed_label = f"{speed:.2f}" if yaw_targeted else f"{speed:.1f}"
                condition = f"v{speed_label}_a{angle:.2f}_turn{sign:+d}"
                steering_waypoints = (
                    (
                        (0.00, 0.0),
                        (0.15, sign * angle),
                        (0.75, sign * angle),
                        (1.00, 0.0),
                        (1.60, 0.0),
                        (1.75, -sign * min(angle, 0.30)),
                        (2.35, -sign * min(angle, 0.30)),
                        (2.60, 0.0),
                    ) if yaw_targeted else (
                        (0.00, 0.0),
                        (0.15, sign * angle),
                        (0.50, sign * angle),
                        (0.80, 0.0),
                        (1.00, -sign * 0.30),
                        (1.25, -sign * 0.30),
                        (1.45, 0.0),
                    )
                )
                phases.extend((
                    Phase(
                        f"approach_{label_prefix}_{condition}", 10.0, speed,
                        throttle_mode="race_domain_approach",
                        reach_speed_target=True,
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"settle_{label_prefix}_{condition}", 0.75, speed,
                        throttle_mode="race_domain_hold",
                        condition_pair_id=condition,
                    ),
                    Phase(
                        f"{label_prefix}_{condition}",
                        2.75 if yaw_targeted else 1.50, speed,
                        throttle_mode="race_domain_hold",
                        validate_samples=True,
                        validate_speed=False,
                        validate_steering=False,
                        condition_pair_id=condition,
                        steering_profile="waypoints",
                        steering_amplitude_rad=angle,
                        steering_waypoints=steering_waypoints,
                    ),
                ))
        return phases
    if profile == SUBNET_TRANSIENT_PROFILE:
        return _subnet_highsteer_transient_phases(seed)
    if profile in (SWERVE_THROTTLE_RATE_SWEEP_PROFILE,
                   SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE):
        return _highsteer_throttle_rate_sweep_phases(
            seed, throttle_rate_sweep_delta_norm,
            highsteer_validation=(
                profile == SWERVE_THROTTLE_RATE_SWEEP_HIGHSTEER_PROFILE))
    if profile == SWERVE_THROTTLE_RATE_FACTORIAL_PROFILE:
        return _highsteer_throttle_rate_factorial_phases(seed)
    if profile == SWERVE_THROTTLE_RATE_FACTORIAL_4P5_PROFILE:
        return _highsteer_throttle_rate_factorial_phases(
            seed, SWERVE_THROTTLE_RATE_FACTORIAL_4P5_SPEED_MPS)
    if profile == SWERVE_THROTTLE_RATE_FACTORIAL_6P5_PROFILE:
        return _highsteer_throttle_rate_factorial_phases(
            seed, SWERVE_THROTTLE_RATE_FACTORIAL_6P5_SPEED_MPS)
    if profile == SWERVE_THROTTLE_RATE_FACTORIAL_7P5_PROFILE:
        return _highsteer_throttle_rate_factorial_phases(
            seed, SWERVE_THROTTLE_RATE_FACTORIAL_7P5_SPEED_MPS)
    if profile in SWERVE_THROTTLE_RATE_FRONTIER_UP_PROFILES:
        return _frontier_throttle_up_swerve_phases(seed)
    if profile in SWERVE_THROTTLE_RATE_RACE_DOMAIN_PROFILES:
        return _race_domain_throttle_rate_phases(
            seed, transition_speed_mps)
    if profile in SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES:
        speed_steering = (
            YAW_FRONTIER_THROTTLE_SLEW_SPEED_STEERING_RAD
            if profile in (YAW_FRONTIER_THROTTLE_SLEW_PROFILE,
                           YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE) else
            SWERVE_THROTTLE_SLEW_FRONTIER_SPEED_STEERING_RAD
            if profile == SWERVE_THROTTLE_SLEW_FRONTIER_PROFILE else
            SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_SPEED_STEERING_RAD
            if profile == SWERVE_THROTTLE_SLEW_11MPS_REPLICATION_PROFILE else
            SWERVE_THROTTLE_SLEW_MODERATE_SPEED_STEERING_RAD
            if profile == SWERVE_THROTTLE_SLEW_MODERATE_PROFILE else
            SWERVE_THROTTLE_SLEW_LOWSTEER_SPEED_STEERING_RAD
            if profile == SWERVE_THROTTLE_SLEW_LOWSTEER_PROFILE else
            SWERVE_THROTTLE_SLEW_SPEED_STEERING_RAD)
        return _race_domain_swerve_throttle_slew_phases(
            seed, speed_steering,
            reset_before_each_probe=(
                profile == SWERVE_THROTTLE_SLEW_LOWSTEER_PROFILE),
            repetitions=(YAW_FRONTIER_LOWANGLE_UNWIND_REPEATS
                         if profile == YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE
                         else 1),
            align_lowangle_unwind=(
                profile == YAW_FRONTIER_LOWANGLE_UNWIND_PROFILE))
    if profile in DYNAMIC_COUPLED_PROFILES:
        for condition in build_dynamic_coupled_plan(seed):
            condition_id = condition.condition_id
            speed = condition.target_speed_mps
            phases.extend((
                Phase(
                    f"approach_coupled_{condition_id}", 12.0, speed,
                    throttle_mode="race_domain_approach",
                    reach_speed_target=True,
                    condition_pair_id=condition_id,
                ),
                Phase(
                    f"settle_coupled_{condition_id}", 1.0, speed,
                    throttle_mode="race_domain_hold",
                    condition_pair_id=condition_id,
                ),
            ))
            phases.extend(_dynamic_coupled_manoeuvre_phases(condition))
        return phases
    if profile == "race_domain_steering_frontier":
        # Existing runs showed reproducible high-speed yaw/ay roll-off beyond
        # 0.10–0.14 rad, but their steering conditions followed one another
        # without matched initial lateral/yaw state. Give every condition a
        # fresh spawn reset, speed approach, and matched-state settling period;
        # this also prevents reaching the finite edge of the Explore plane.
        plan = build_race_domain_steering_frontier_plan(seed)
        for target_speed in RACE_DOMAIN_STEERING_FRONTIER_SPEEDS_MPS:
            for block in (item for item in plan
                          if item.target_speed_mps == target_speed):
                phases.append(Phase(
                    f"approach_{block.label}", 6.0, target_speed,
                    throttle_mode="race_domain_approach",
                    reach_speed_target=True))
                phases.append(Phase(
                    block.label, block.duration_s, block.target_speed_mps,
                    steering_rad=block.steering_rad,
                    throttle_mode="race_domain_hold",
                    validate_samples=True,
                    # Cornering speed loss is measured plant behavior.
                    validate_speed=False,
                    settle_before_probe=True,
                ))
        return phases
    if profile in ("race_domain_continuous", "race_domain_brake_boundary",
                   "race_domain_moderate_braking"):
        plan = (build_race_domain_moderate_braking_plan(seed)
                if profile == "race_domain_moderate_braking" else
                build_race_domain_boundary_plan(seed)
                if profile == "race_domain_brake_boundary" else
                build_race_domain_plan(
                    seed, boundary_speed_mps=RACE_DOMAIN_BOUNDARY_SPEED_MPS))
        phases.append(Phase(
            profile,
            plan_duration_s(plan),
            0.0,
            throttle_mode="race_domain_continuous",
        ))
        return phases
    if profile == "isolated_boundary":
        target_speed = 2.2
        phases.extend((
            Phase("approach_2.2mps", 8.0, target_speed, throttle_mode="approach",
                  reach_speed_target=True),
            Phase("settle_2.2mps", 1.0, target_speed),
        ))
        conditions = [
            sign * amplitude
            for amplitude in (0.30, 0.42, 0.46, 0.50)
            for sign in (-1.0, 1.0)
        ]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_{steering:+.2f}rad",
                    2.0,
                    target_speed,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_speed_sweep":
        steering_levels = (0.30, 0.34, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50)
        for speed in (2.2, 3.0):
            phases.extend((
                Phase(f"approach_{speed:.1f}mps", 8.0, speed,
                      throttle_mode="approach", reach_speed_target=True),
                Phase(f"settle_{speed:.1f}mps", 1.0, speed),
            ))
            for repetition in (1, 2):
                signs = (-1.0, 1.0) if repetition == 1 else (1.0, -1.0)
                for sign in signs:
                    directions = ("up", "down") if repetition == 1 else ("down", "up")
                    for direction in directions:
                        if direction == "down":
                            phases.append(Phase(
                                f"prep_r{repetition}_{sign:+.0f}_{speed:.1f}mps",
                                1.2, speed, steering_rad=sign * steering_levels[-1],
                            ))
                        levels = (steering_levels if direction == "up"
                                  else tuple(reversed(steering_levels)))
                        for magnitude in levels:
                            phases.append(Phase(
                                f"sweep_r{repetition}_{sign:+.0f}_{direction}_"
                                f"{magnitude:.2f}_{speed:.1f}mps",
                                1.2, speed, steering_rad=sign * magnitude,
                                validate_samples=True, validate_speed=False,
                            ))
        return phases
    if profile in ("isolated_force_3mps", "isolated_force_4mps",
                   "isolated_force_5mps"):
        # Matched-start repeated probes characterize the high-angle yaw/slip
        # branch at a fixed speed. Repetition 3 is reserved for holdout.
        target_speed = {
            "isolated_force_3mps": 3.0,
            "isolated_force_4mps": 4.0,
            "isolated_force_5mps": 5.0,
        }[profile]
        steering_levels = (0.30, 0.42, 0.46, 0.50)
        phases.extend((
            Phase(f"approach_{target_speed:.1f}mps", 8.0, target_speed,
                  throttle_mode="approach", reach_speed_target=True),
            Phase(f"settle_{target_speed:.1f}mps", 1.0, target_speed),
        ))
        conditions = [sign * amplitude
                      for amplitude in steering_levels
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_{steering:+.2f}rad_"
                    f"{target_speed:.1f}mps",
                    2.0,
                    target_speed,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_highspeed_surface":
        # Matched-start response identification beyond the currently validated
        # 3--5 m/s band. Repetition 3 is a complete-run holdout. Dense steering
        # levels resolve the known 0.20--0.23 rad transition while the remaining
        # knots preserve the full high-angle response shape.
        steering_levels = (0.15, 0.20, 0.21, 0.22, 0.23,
                           0.25, 0.30, 0.35, 0.42, 0.50)
        for target_speed in (4.5, 6.5, 7.5):
            phases.extend((
                Phase(f"approach_{target_speed:.1f}mps", 8.0, target_speed,
                      throttle_mode="approach", reach_speed_target=True),
                Phase(f"settle_{target_speed:.1f}mps", 1.0, target_speed),
            ))
            conditions = [sign * angle for angle in steering_levels
                          for sign in (-1.0, 1.0)]
            for repetition in range(1, 4):
                ordered = conditions.copy()
                rng.shuffle(ordered)
                for steering in ordered:
                    phases.append(Phase(
                        f"isolated_r{repetition}_{steering:+.2f}rad_"
                        f"{target_speed:.1f}mps",
                        1.4,
                        target_speed,
                        steering_rad=steering,
                        validate_samples=True,
                        settle_before_probe=True,
                    ))
        return phases
    if profile == "isolated_3to5_response_surface":
        # Dense, matched-start speed/steering surface for recursive MPC model
        # validation. Train at 3 and 5 m/s; keep every 4 m/s phase available
        # as a complete unseen-speed holdout. The steering knots resolve the
        # measured high-speed authority trough without changing simulator
        # physics or using ground truth as a controller input.
        steering_levels = (0.15, 0.20, 0.21, 0.22, 0.23,
                           0.25, 0.30, 0.35, 0.42, 0.50)
        for target_speed in (3.0, 4.0, 5.0):
            phases.extend((
                Phase(f"approach_{target_speed:.1f}mps", 8.0, target_speed,
                      throttle_mode="approach", reach_speed_target=True),
                Phase(f"settle_{target_speed:.1f}mps", 1.0, target_speed),
            ))
            conditions = [sign * angle for angle in steering_levels
                          for sign in (-1.0, 1.0)]
            for repetition in range(1, 4):
                ordered = conditions.copy()
                rng.shuffle(ordered)
                for steering in ordered:
                    phases.append(Phase(
                        f"isolated_r{repetition}_{steering:+.2f}rad_"
                        f"{target_speed:.1f}mps",
                        1.4,
                        target_speed,
                        steering_rad=steering,
                        validate_samples=True,
                        settle_before_probe=True,
                    ))
        return phases
    if profile == "isolated_highspeed_crossfactor":
        # Resolve the high-speed steering onset and independently perturb
        # throttle at matched speed/state. Repetition 3 is a held-out run.
        # The fixed-pedal probes are short and remain below the global 9 m/s
        # cutoff; they do not alter the simulator's physics.
        steering_levels = (0.05, 0.075, 0.10, 0.125, 0.15,
                           0.175, 0.20, 0.225, 0.25)
        throttle_levels = (0.10, 0.15, 0.20)
        for target_speed, feedforward in ((6.5, 0.27), (7.5, 0.31)):
            phases.extend((
                Phase(f"approach_{target_speed:.1f}mps", 8.0, target_speed,
                      throttle_mode="approach", reach_speed_target=True),
                Phase(f"settle_{target_speed:.1f}mps", 1.0, target_speed),
            ))
            for repetition in range(1, 4):
                conditions = [
                    (sign * angle, "speed_hold", None, "hold")
                    for angle in steering_levels
                    for sign in (-1.0, 1.0)
                ]
                conditions.extend(
                    (sign * angle, "fixed", feedforward + offset,
                     f"pedal_{offset:+.3f}")
                    for angle in throttle_levels
                    for sign in (-1.0, 1.0)
                    for offset in (-0.003, 0.003)
                )
                rng.shuffle(conditions)
                for steering, mode, throttle, condition in conditions:
                    throttle_args = ({"throttle_mode": mode,
                                      "throttle_norm": throttle}
                                     if mode == "fixed" else {})
                    label = (f"isolated_r{repetition}_{target_speed:.1f}mps_"
                             f"steer_{steering:+.3f}rad_{condition}")
                    phases.append(Phase(
                        label,
                        1.2,
                        target_speed,
                        steering_rad=steering,
                        validate_samples=True,
                        validate_speed=(mode != "fixed"),
                        settle_before_probe=True,
                        **throttle_args,
                    ))
        return phases
    if profile == "isolated_highspeed_tail":
        # Extend the clean training domain to its empirically supported upper
        # edge. A 2026-09-29 run completed the 8.0/8.3 m/s surfaces, but at
        # 8.6 m/s developed a delayed roll/pitch impulse after steering had
        # returned to zero (tilt >120 deg, no collision-count change). Do not
        # repeat that unsafe region or relax the existing 8 deg / 9 m/s stops.
        speed_steering = (
            (8.0, (0.025, 0.040, 0.055, 0.070)),
            (8.3, (0.022, 0.035, 0.050, 0.065)),
        )
        for target_speed, steering_levels in speed_steering:
            phases.extend((
                Phase(f"approach_{target_speed:.1f}mps", 8.0,
                      target_speed, throttle_mode="approach",
                      reach_speed_target=True),
                Phase(f"settle_{target_speed:.1f}mps", 1.0, target_speed),
            ))
            conditions = [0.0] + [sign * angle for angle in steering_levels
                                  for sign in (-1.0, 1.0)]
            for repetition in range(1, 4):
                ordered = conditions.copy()
                rng.shuffle(ordered)
                for steering in ordered:
                    phases.append(Phase(
                        f"tail_r{repetition}_{target_speed:.1f}mps_"
                        f"steer_{steering:+.4f}rad",
                        1.2,
                        target_speed,
                        steering_rad=steering,
                        validate_samples=True,
                        settle_before_probe=True,
                    ))
        return phases
    if profile == "isolated_transition_65mps":
        # Resolve the held-out Jacobian mismatch around the response minimum.
        # Keep this narrow profile at the already repeatable 6.5 m/s operating
        # point; repetition 3 is a complete held-out steering surface.
        target_speed = 6.5
        steering_levels = tuple(round(0.145 + 0.005 * index, 3)
                                 for index in range(8))
        phases.extend((
            Phase("approach_6.5mps", 8.0, target_speed,
                  throttle_mode="approach", reach_speed_target=True),
            Phase("settle_6.5mps", 1.0, target_speed),
        ))
        conditions = [sign * angle for angle in steering_levels
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_6.5mps_steer_{steering:+.3f}rad",
                    1.2,
                    target_speed,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_transition_45mps":
        # Resolve the speed-dependent front-saturation cliff at 4.5 m/s.
        # Existing 0.01-rad data places the response trough inside 0.18--0.21;
        # this 0.0025-rad grid is intended to validate its local MPC Jacobian.
        target_speed = 4.5
        steering_levels = (0.15, 0.175) + tuple(
            round(0.18 + 0.0025 * index, 4) for index in range(19)) + (0.23, 0.25)
        phases.extend((
            Phase("approach_4.5mps", 8.0, target_speed,
                  throttle_mode="approach", reach_speed_target=True),
            Phase("settle_4.5mps", 1.0, target_speed),
        ))
        conditions = [sign * angle for angle in steering_levels
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_4.5mps_steer_{steering:+.4f}rad",
                    1.4,
                    target_speed,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_transition_speed_surface":
        # The measured transition shifts approximately with inverse speed:
        # its center is near 0.8 / v rad in the measured 4.0--5.0 m/s captures.
        # Resolve that region densely while retaining low/high-angle anchors.
        if not 3.0 <= transition_speed_mps <= 8.0:
            raise ValueError("transition speed surface requires 3 to 8 m/s")
        center = 0.80 / transition_speed_mps
        dense_levels = {
            round(center - 0.025 + 0.0025 * index, 4)
            for index in range(21)
        }
        steering_levels = tuple(sorted(dense_levels | {0.15, 0.20, 0.25}))
        phases.extend((
            Phase(f"approach_{transition_speed_mps:.2f}mps", 8.0,
                  transition_speed_mps, throttle_mode="approach",
                  reach_speed_target=True),
            Phase(f"settle_{transition_speed_mps:.2f}mps", 1.0,
                  transition_speed_mps),
        ))
        conditions = [sign * angle for angle in steering_levels
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_{transition_speed_mps:.2f}mps_"
                    f"steer_{steering:+.4f}rad",
                    1.4,
                    transition_speed_mps,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_transition_support":
        # Extend both the low-demand and high-steering ends of the dense
        # transition captures so one held-out surface covers the full turn.
        if not 3.0 <= transition_speed_mps <= 8.0:
            raise ValueError("transition support requires 3 to 8 m/s")
        phases.extend((
            Phase(f"approach_{transition_speed_mps:.2f}mps", 8.0,
                  transition_speed_mps, throttle_mode="approach",
                  reach_speed_target=True),
            Phase(f"settle_{transition_speed_mps:.2f}mps", 1.0,
                  transition_speed_mps),
        ))
        conditions = [sign * angle for angle in
                      (0.125, 0.275, 0.30, 0.35, 0.42, 0.46, 0.50)
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_{transition_speed_mps:.2f}mps_"
                    f"support_steer_{steering:+.4f}rad",
                    1.4,
                    transition_speed_mps,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_transition_bridge":
        # Fill the measured gap between the fine response-transition probes
        # and the wide-angle support anchors without repeating either sweep.
        if not 3.0 <= transition_speed_mps <= 8.0:
            raise ValueError("transition bridge requires 3 to 8 m/s")
        phases.extend((
            Phase(f"approach_{transition_speed_mps:.2f}mps", 8.0,
                  transition_speed_mps, throttle_mode="approach",
                  reach_speed_target=True),
            Phase(f"settle_{transition_speed_mps:.2f}mps", 1.0,
                  transition_speed_mps),
        ))
        conditions = [sign * angle for angle in (0.20, 0.225, 0.25)
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_{transition_speed_mps:.2f}mps_"
                    f"bridge_steer_{steering:+.4f}rad",
                    1.4,
                    transition_speed_mps,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_transition_low_support":
        # Extend the low-q endpoint of the 6.5 m/s demand surface so the
        # cross-speed interpolation remains inside measured support.
        if not 3.0 <= transition_speed_mps <= 8.0:
            raise ValueError("low-demand support requires 3 to 8 m/s")
        phases.extend((
            Phase(f"approach_{transition_speed_mps:.2f}mps", 8.0,
                  transition_speed_mps, throttle_mode="approach",
                  reach_speed_target=True),
            Phase(f"settle_{transition_speed_mps:.2f}mps", 1.0,
                  transition_speed_mps),
        ))
        conditions = [sign * angle for angle in (0.08, 0.10)
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_{transition_speed_mps:.2f}mps_"
                    f"low_support_steer_{steering:+.4f}rad",
                    1.4,
                    transition_speed_mps,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "isolated_transition_full_surface":
        # A single fixed-speed calibration spanning the response knee through
        # full lock. Keep the 14 levels sparse enough to finish with three
        # repetitions while resolving the steep low-angle authority drop.
        if not 3.0 <= transition_speed_mps <= 9.0:
            raise ValueError("full response surface requires 3 to 9 m/s")
        phases.extend((
            Phase(f"approach_{transition_speed_mps:.2f}mps", 8.0,
                  transition_speed_mps, throttle_mode="approach",
                  reach_speed_target=True),
            Phase(f"settle_{transition_speed_mps:.2f}mps", 1.0,
                  transition_speed_mps),
        ))
        steering_levels = (0.08, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225,
                           0.25, 0.275, 0.30, 0.35, 0.42, 0.46, 0.50)
        conditions = [sign * angle for angle in steering_levels
                      for sign in (-1.0, 1.0)]
        for repetition in range(1, 4):
            ordered = conditions.copy()
            rng.shuffle(ordered)
            for steering in ordered:
                phases.append(Phase(
                    f"isolated_r{repetition}_{transition_speed_mps:.2f}mps_"
                    f"full_steer_{steering:+.4f}rad",
                    1.4,
                    transition_speed_mps,
                    steering_rad=steering,
                    validate_samples=True,
                    settle_before_probe=True,
                ))
        return phases
    if profile == "transient_4mps":
        # Repeated up/down steering sweeps expose whether the high-angle yaw
        # curve depends only on current steering or also on recent tire/body
        # state. Repetition four is reserved for a fresh sequence holdout.
        target_speed = 4.0
        phases.extend((
            Phase("approach_4.0mps", 8.0, target_speed, throttle_mode="approach",
                  reach_speed_target=True),
            Phase("settle_4.0mps", 1.0, target_speed),
        ))
        ascending = (0.30, 0.42, 0.46, 0.50)
        descending = (0.46, 0.42, 0.30)
        for repetition in range(1, 5):
            signs = [-1.0, 1.0]
            rng.shuffle(signs)
            for sign in signs:
                for index, amplitude in enumerate(ascending):
                    phases.append(Phase(
                        f"transient_r{repetition}_{sign:+.0f}_up_{amplitude:.2f}_4.0mps",
                        1.25, target_speed, steering_rad=sign * amplitude,
                        validate_samples=True,
                        settle_before_probe=(index == 0),
                    ))
                for amplitude in descending:
                    phases.append(Phase(
                        f"transient_r{repetition}_{sign:+.0f}_down_{amplitude:.2f}_4.0mps",
                        1.25, target_speed, steering_rad=sign * amplitude,
                        validate_samples=True,
                    ))
                phases.append(Phase(
                    f"transient_r{repetition}_{sign:+.0f}_return_zero_4.0mps",
                    1.25, target_speed, steering_rad=0.0,
                ))
        return phases
    if profile in ("transient_fullsteer_4mps", "transient_transition_4mps",
                   "transient_transition_4mps_fixedthrottle",
                   "transient_transition_dwell_4mps_fixedthrottle"):
        # Repetitions one and two train the steering-response map; repetition
        # three is held out. The focused transition grid resolves the observed
        # 0.20--0.25 rad front-slip/yaw-authority collapse at 0.01 rad spacing.
        # Fixed-throttle variants hold 0.164 through each probe to separate
        # speed-loop throttle changes from steering-history effects. The dwell
        # profile narrows the sweep and holds each level for four seconds.
        target_speed = 4.0
        phases.extend((
            Phase("approach_4.0mps", 8.0, target_speed, throttle_mode="approach",
                  reach_speed_target=True),
            Phase("settle_4.0mps", 1.0, target_speed),
        ))
        fixed_throttle_test = profile in (
            "transient_transition_4mps_fixedthrottle",
            "transient_transition_dwell_4mps_fixedthrottle",
        )
        probe_throttle = ({"throttle_mode": "fixed", "throttle_norm": 0.164}
                          if fixed_throttle_test else {})
        dwell_test = profile == "transient_transition_dwell_4mps_fixedthrottle"
        if dwell_test:
            steering_levels = (0.20, 0.21, 0.22, 0.23)
        elif profile in ("transient_transition_4mps",
                         "transient_transition_4mps_fixedthrottle"):
            steering_levels = tuple(round(0.18 + 0.01 * index, 2)
                                    for index in range(11))
        else:
            steering_levels = (0.05, 0.10, 0.15, 0.20, 0.25,
                               0.30, 0.35, 0.42, 0.46, 0.50)
        phase_duration_s = 4.0 if dwell_test else 1.20
        for repetition in range(1, 4):
            signs = [-1.0, 1.0]
            rng.shuffle(signs)
            for sign in signs:
                for index, amplitude in enumerate(steering_levels):
                    phases.append(Phase(
                        f"transient_r{repetition}_{sign:+.0f}_up_{amplitude:.2f}_4.0mps",
                        phase_duration_s, target_speed, steering_rad=sign * amplitude,
                        validate_samples=True,
                        settle_before_probe=(index == 0),
                        **probe_throttle,
                    ))
                for amplitude in reversed(steering_levels[:-1]):
                    phases.append(Phase(
                        f"transient_r{repetition}_{sign:+.0f}_down_{amplitude:.2f}_4.0mps",
                        phase_duration_s, target_speed, steering_rad=sign * amplitude,
                        validate_samples=True,
                        **probe_throttle,
                    ))
                phases.append(Phase(
                    f"transient_r{repetition}_{sign:+.0f}_return_zero_4.0mps",
                    phase_duration_s, target_speed, steering_rad=0.0,
                    **probe_throttle,
                ))
        return phases
    if profile != "grid":
        raise ValueError(f"unknown profile: {profile}")

    for speed in SPEED_TARGETS_MPS:
        phases.append(Phase(f"approach_{speed:.1f}mps", 8.0, speed,
                            throttle_mode="approach", reach_speed_target=True))
        phases.append(Phase(f"settle_{speed:.1f}mps", 1.0, speed))
        steering_steps = [
            sign * amplitude
            for amplitude in STEERING_AMPLITUDES_RAD
            for sign in (-1.0, 1.0)
        ]
        rng.shuffle(steering_steps)
        for index, steering in enumerate(steering_steps, start=1):
            phases.append(Phase(
                f"steer_{index:02d}_{speed:.1f}mps_{steering:+.2f}rad",
                1.6, speed, steering_rad=steering, validate_samples=True,
            ))
    return phases


class OpenPlaneExcitation:
    def __init__(self, seed: int, timeout_s: float, profile: str,
                 transition_speed_mps: float = 4.5,
                 probe_dwell_s: float = 0.0,
                 speed_hold_kp: float = 0.04,
                 speed_hold_ki: float = 0.0,
                 speed_median_gate_mps: float = MAX_SPEED_MEDIAN_ERROR_MPS,
                 speed_p95_gate_mps: float = MAX_SPEED_P95_ERROR_MPS,
                 throttle_rate_sweep_delta_norm: float =
                 SWERVE_THROTTLE_SLEW_DELTA_NORM) -> None:
        if not math.isfinite(speed_hold_kp) or speed_hold_kp < 0.0:
            raise ValueError("speed-hold proportional gain must be finite and nonnegative")
        if not math.isfinite(speed_hold_ki) or speed_hold_ki < 0.0:
            raise ValueError("speed-hold integral gain must be finite and nonnegative")
        if (not math.isfinite(speed_median_gate_mps) or
                speed_median_gate_mps <= 0.0 or
                not math.isfinite(speed_p95_gate_mps) or
                speed_p95_gate_mps <= 0.0):
            raise ValueError("speed-error gates must be finite and positive")
        if (not math.isfinite(throttle_rate_sweep_delta_norm)
                or throttle_rate_sweep_delta_norm <= 0.0
                or throttle_rate_sweep_delta_norm > MAX_THROTTLE):
            raise ValueError("rate-sweep throttle delta must be in (0, 0.50]")
        if not math.isfinite(probe_dwell_s) or not 0.0 <= probe_dwell_s <= 15.0:
            raise ValueError("probe dwell must be finite and in [0, 15] seconds")
        if probe_dwell_s > 0.0 and profile in (
                "full_input_excitation", "grid", *DYNAMIC_COUPLED_PROFILES,
                SUBNET_TRANSIENT_PROFILE, *SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES):
            raise ValueError("probe dwell override is unsupported for this fixed capture protocol")
        self.dynamic_coupled_plan = (
            build_dynamic_coupled_plan(seed)
            if profile in DYNAMIC_COUPLED_PROFILES else ())
        self.race_domain_plan = (
            build_race_domain_moderate_braking_plan(seed)
            if profile == "race_domain_moderate_braking" else
            build_race_domain_boundary_plan(seed)
            if profile == "race_domain_brake_boundary" else
            build_race_domain_plan(
                seed, boundary_speed_mps=RACE_DOMAIN_BOUNDARY_SPEED_MPS)
            if profile == "race_domain_continuous" else ())
        phases = build_schedule(
            seed, profile, transition_speed_mps,
            throttle_rate_sweep_delta_norm)
        self.subnet_transient_reset_count = (
            sum(phase.label.startswith("approach_subnet_")
                for phase in phases)
            if profile == SUBNET_TRANSIENT_PROFILE else 0)
        self.race_domain_plan_version = (
            4 if profile == "race_domain_moderate_braking" else
            3 if profile == "race_domain_brake_boundary" else
            2 if profile == "race_domain_continuous" else None)
        if (profile in ("race_domain_continuous", "race_domain_brake_boundary",
                        "race_domain_moderate_braking")
                and timeout_s < plan_duration_s(self.race_domain_plan) + 5.0):
            raise ValueError(
                "race-domain capture timeout must exceed its plan by 5 seconds")
        if profile in DYNAMIC_COUPLED_PROFILES:
            required = (
                sum(phase.duration_s for phase in phases)
                + len(self.dynamic_coupled_plan)
                * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
                + 5.0
            )
            if timeout_s < required:
                raise ValueError(
                    f"{profile} requires --timeout-s >= {required:g}")
        if profile == SUBNET_TRANSIENT_PROFILE:
            required = (
                sum(phase.duration_s for phase in phases)
                + self.subnet_transient_reset_count
                * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
                + 5.0
            )
            if timeout_s < required:
                raise ValueError(
                    f"{profile} requires --timeout-s >= {required:g}")
        if profile in YAW_TRANSIENT_PROFILES:
            reset_cycles = sum(
                _is_yaw_transient_approach(phase.label)
                for phase in phases)
            required = (
                sum(phase.duration_s for phase in phases)
                + reset_cycles * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
                + 5.0
            )
            if timeout_s < required:
                raise ValueError(
                    f"{profile} requires --timeout-s >= {required:g}")
        if profile in SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES:
            required = (
                sum(phase.duration_s for phase in phases)
                + sum(phase.settle_before_probe for phase in phases)
                * PROBE_START_TIMEOUT_SEC
                + sum(phase.label.startswith("approach_swerve_pair_")
                      for phase in phases)
                * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
                + 5.0
            )
            if timeout_s < required:
                raise ValueError(
                    f"{profile} requires --timeout-s >= {required:g}")
        if profile == "isolated_highsteer_multispeed":
            required = (
                sum(phase.duration_s for phase in phases)
                + sum(phase.settle_before_probe for phase in phases)
                * PROBE_START_TIMEOUT_SEC
                + 5.0
            )
            if timeout_s < required:
                raise ValueError(
                    f"{profile} requires --timeout-s >= {required:g}")
        self.node = rclpy.create_node("open_plane_excitation")
        self.seed = seed
        self.profile = profile
        self.phases = phases
        if probe_dwell_s > 0.0:
            self.phases = [
                replace(phase, duration_s=probe_dwell_s)
                if phase.validate_samples else phase
                for phase in self.phases
            ]
        self.probe_dwell_s = probe_dwell_s
        self.timeout_s = timeout_s
        self.speed_hold_kp = speed_hold_kp
        self.speed_hold_ki = speed_hold_ki
        self.speed_integral_throttle = 0.0
        self.speed_integral_target_mps: float | None = None
        self.speed_integral_updated_at: float | None = None
        self.speed_median_gate_mps = speed_median_gate_mps
        self.speed_p95_gate_mps = speed_p95_gate_mps
        self.created_at = time.monotonic()
        self.started_at: float | None = None
        self.phase_started_at: float | None = None
        self.probe_wait_started_at: float | None = None
        self.probe_stable_since: float | None = None
        self.ready_since: float | None = None
        self.phase_index = 0
        self.last_race_domain_block_index = -1
        self.speed_mps: float | None = None
        self.vx_mps: float | None = None
        self.vy_mps: float | None = None
        self.yaw_rate_rps: float | None = None
        self.tilt_rad: float | None = None
        self.position_xy: tuple[float, float] | None = None
        self.state_history: deque[tuple[float, float, float, float]] = deque(maxlen=64)
        self.last_odom_at: float | None = None
        self.steering_feedback_rad: float | None = None
        self.last_steering_at: float | None = None
        self.throttle_feedback_norm: float | None = None
        self.last_throttle_at: float | None = None
        self.collision_initial: int | None = None
        self.last_collision_at: float | None = None
        self.collision_baseline_safe = False
        self.aborted = False
        self.done = False
        self.quality_failures: list[str] = []
        self.phase_samples: list[tuple[float, float]] = []
        self.phase_max_speed_mps = 0.0
        self.phase_max_tilt_rad = 0.0
        self.phase_governor_ticks = 0
        self.phase_start_published = False
        self.phase_stimulus_published = False
        self.published_count = 0
        self.last_status_log = 0.0
        self.neutral_ticks_remaining = 0
        self.shutdown_sent = False
        self.finish_reason = ""
        self.reset_state: str | None = None
        self.reset_started_at: float | None = None
        self.reset_released_at: float | None = None
        self.reset_stable_since: float | None = None
        self.reset_reason = ""
        self.reset_count = 0
        self.spawn_xy: tuple[float, float] | None = None

        sensor_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.steering_pub = self.node.create_publisher(
            Float32, STEERING_COMMAND_TOPIC, 10)
        self.throttle_pub = self.node.create_publisher(
            Float32, THROTTLE_COMMAND_TOPIC, 10)
        self.phase_pub = self.node.create_publisher(String, PHASE_TOPIC, 10)
        self.phase_event_queue: queue.Queue[str | None] = queue.Queue()
        self.phase_event_failures = 0
        self.phase_event_worker = threading.Thread(
            target=self._publish_phase_events,
            name="open-plane-phase-events",
            daemon=True,
        )
        self.phase_event_worker.start()
        self.odom_sub = self.node.create_subscription(
            Odometry, ODOM_TOPIC, self._on_odom, sensor_qos)
        self.steering_sub = self.node.create_subscription(
            Float32, STEERING_TOPIC, self._on_steering, sensor_qos)
        self.throttle_sub = self.node.create_subscription(
            Float32, THROTTLE_FEEDBACK_TOPIC, self._on_throttle_feedback,
            sensor_qos)
        self.reset_pub = (
            self.node.create_publisher(Bool, RESET_COMMAND_TOPIC, 1)
            if (profile == "race_domain_steering_frontier"
                or profile == SUBNET_TRANSIENT_PROFILE
                or profile in RACE_DOMAIN_SPEED_GOVERNED_PROFILES) else None
        )
        self.collision_sub = self.node.create_subscription(
            Int32, COLLISION_TOPIC, self._on_collision, sensor_qos)
        self.timer = self.node.create_timer(PERIOD_SEC, self._tick)
        nominal_schedule_s = sum(phase.duration_s for phase in self.phases)
        settle_budget_s = (
            sum(phase.settle_before_probe for phase in self.phases)
            * PROBE_START_TIMEOUT_SEC
        )
        reset_budget_s = (
            sum(phase.label.startswith("approach_frontier_")
                for phase in self.phases)
            + (len(self.dynamic_coupled_plan)
               if self.dynamic_coupled_plan else 0)
            + self.subnet_transient_reset_count
            + sum(_is_yaw_transient_approach(phase.label)
                  for phase in self.phases)
            + sum(phase.label.startswith("approach_swerve_pair_")
                  for phase in self.phases)
        ) * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
        self.node.get_logger().info(
            f"waiting for source odom and zero collision count; profile={profile}, seed={seed}, "
            f"phases={len(self.phases)}, "
            f"schedule_max={nominal_schedule_s + settle_budget_s + reset_budget_s:.1f}s "
            f"(nominal={nominal_schedule_s:.1f}s, settle_budget={settle_budget_s:.1f}s, "
            f"reset_budget={reset_budget_s:.1f}s), "
            f"probe_dwell_override={probe_dwell_s:.1f}s, "
            f"throttle_command_cap={MAX_THROTTLE:.2f}, "
            f"speed_hold_kp={speed_hold_kp:.3f}, "
            f"speed_hold_ki={speed_hold_ki:.3f}, "
            f"speed_gates[p50/p95]={speed_median_gate_mps:.3f}/"
            f"{speed_p95_gate_mps:.3f}m/s, "
            f"timeout={timeout_s:.1f}s")

    def _publish(self, steering_rad: float, throttle_norm: float) -> None:
        steering_rad = max(-MAX_STEERING_RAD, min(MAX_STEERING_RAD, steering_rad))
        throttle_norm = max(0.0, min(MAX_THROTTLE, throttle_norm))
        # Simulator steering input is normalized; excitation limits are specified in radians.
        self.steering_pub.publish(Float32(data=steering_rad / STEERING_LIMIT_RAD))
        self.throttle_pub.publish(Float32(data=throttle_norm))
        self.published_count += 1

    def _neutral(self, reason: str) -> None:
        self._publish(0.0, 0.0)
        self.node.get_logger().warn(f"neutral output: {reason}")

    def _finish(self, reason: str, aborted: bool) -> None:
        if self.done:
            return
        if self.reset_pub is not None:
            self.reset_pub.publish(Bool(data=False))
        if self.phase_start_published and self.phase_index < len(self.phases):
            self._close_phase("aborted" if aborted else "complete", reason)
        self.done = True
        self.aborted = aborted
        self.finish_reason = reason
        self._publish_event({
            "event": "experiment_end",
            "profile": self.profile,
            "seed": self.seed,
            "phase_count": len(self.phases),
            "sim_reset_count": self.reset_count,
            "reason": reason,
            "aborted": aborted,
            "quality_failures": self.quality_failures,
            "monotonic_ns": time.monotonic_ns(),
        })
        if self.last_odom_at is not None:
            self._neutral(reason)
            self.neutral_ticks_remaining = 3
        else:
            self.node.get_logger().warn(f"no actuator output before source odom: {reason}")
            self._shutdown_after_neutral()

    def _shutdown_after_neutral(self) -> None:
        if self.shutdown_sent:
            return
        self.shutdown_sent = True
        now = time.monotonic()
        elapsed = max(0.0, now - self.started_at) if self.started_at is not None else 0.0
        rate = (self.published_count - 1) / elapsed if elapsed > 0.0 else 0.0
        self.node.get_logger().info(
            f"finished: reason={self.finish_reason}, aborted={self.aborted}, "
            f"phases={self.phase_index}/{len(self.phases)}, "
            f"quality_failures={len(self.quality_failures)}, "
            f"command_count={self.published_count}, command_rate={rate:.2f}Hz")

    def _on_odom(self, message: Odometry) -> None:
        vx = float(message.twist.twist.linear.x)
        vy_com = float(message.twist.twist.linear.y)
        yaw_rate = float(message.twist.twist.angular.z)
        vy = vy_com - yaw_rate * COM_X_M
        speed = math.hypot(vx, vy)
        orientation = message.pose.pose.orientation
        position = message.pose.pose.position
        x = float(position.x)
        y = float(position.y)
        tilt_cosine = 1.0 - 2.0 * (
            float(orientation.x) ** 2 + float(orientation.y) ** 2)
        tilt = math.acos(max(-1.0, min(1.0, tilt_cosine)))
        if not all(map(math.isfinite, (vx, vy, yaw_rate, speed, tilt, x, y))):
            return
        now = time.monotonic()
        self.speed_mps = speed
        self.vx_mps = vx
        self.vy_mps = vy
        self.yaw_rate_rps = yaw_rate
        self.tilt_rad = tilt
        self.position_xy = (x, y)
        self.last_odom_at = now
        self.state_history.append((now, speed, vy, yaw_rate))
        if (self.started_at is not None and self.phase_started_at is not None and
                self.phase_index < len(self.phases)):
            self.phase_max_speed_mps = max(self.phase_max_speed_mps, speed)
            self.phase_max_tilt_rad = max(self.phase_max_tilt_rad, tilt)
            phase = self.phases[self.phase_index]
            if (phase.validate_samples and
                    now - self.phase_started_at >= PHASE_SETTLE_SEC and
                    self.steering_feedback_rad is not None and
                    self.last_steering_at is not None and
                    now - self.last_steering_at <= STEERING_FEEDBACK_TIMEOUT_SEC):
                self.phase_samples.append((
                    speed,
                    abs(self.steering_feedback_rad - phase.steering_rad),
                ))
        self._maybe_start()

    def _on_steering(self, message: Float32) -> None:
        value = float(message.data)
        if math.isfinite(value):
            self.steering_feedback_rad = value
            self.last_steering_at = time.monotonic()

    def _on_throttle_feedback(self, message: Float32) -> None:
        value = float(message.data)
        if math.isfinite(value):
            self.throttle_feedback_norm = value
            self.last_throttle_at = time.monotonic()

    def _on_collision(self, message: Int32) -> None:
        self.last_collision_at = time.monotonic()
        count = int(message.data)
        if self.collision_initial is None:
            self.collision_initial = count
            if count != 0:
                self._finish(f"unsafe initial collision count={count}", aborted=True)
                return
            self.collision_baseline_safe = True
            self._maybe_start()
        elif count > self.collision_initial:
            self._finish(f"collision count increased to {count}", aborted=True)

    def _start_phase(self, now: float) -> None:
        phase = self.phases[self.phase_index]
        self.phase_started_at = now
        self.probe_wait_started_at = now if phase.settle_before_probe else None
        self.probe_stable_since = None
        self.phase_samples.clear()
        self.phase_max_speed_mps = self.speed_mps or 0.0
        self.phase_max_tilt_rad = self.tilt_rad or 0.0
        self.phase_governor_ticks = 0
        self.phase_start_published = False
        self.phase_stimulus_published = False

    def _begin_sim_reset(self, now: float, reason: str) -> None:
        if self.reset_pub is None:
            self._finish("built-in simulator reset publisher unavailable", aborted=True)
            return
        self.reset_state = "hold"
        self.reset_started_at = now
        self.reset_released_at = None
        self.reset_stable_since = None
        self.reset_reason = reason
        self.phase_started_at = None
        self.phase_start_published = False
        self._publish(0.0, 0.0)
        self.reset_pub.publish(Bool(data=True))
        self._publish_event({
            "event": "sim_reset_start",
            "profile": self.profile,
            "reason": reason,
            "next_phase_index": self.phase_index,
            "position_xy": self.position_xy,
            "speed_mps": self.speed_mps,
            "monotonic_ns": time.monotonic_ns(),
        })

    def _tick_sim_reset(self, now: float) -> None:
        assert self.reset_pub is not None and self.reset_started_at is not None
        self._publish(0.0, 0.0)
        if self.reset_state == "hold":
            self.reset_pub.publish(Bool(data=True))
            if now - self.reset_started_at >= SIM_RESET_HOLD_SEC:
                self.reset_pub.publish(Bool(data=False))
                self.reset_state = "wait"
                self.reset_released_at = now
                self.reset_stable_since = None
                self._publish_event({
                    "event": "sim_reset_release",
                    "profile": self.profile,
                    "next_phase_index": self.phase_index,
                    "reset_command": False,
                    "monotonic_ns": time.monotonic_ns(),
                })
            return

        self.reset_pub.publish(Bool(data=False))
        if now - self.reset_started_at > SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC:
            self._finish("simulator did not return to spawn after reset", aborted=True)
            return
        position_ok = (
            self.position_xy is not None
            and (self.spawn_xy is None or math.dist(
                self.position_xy, self.spawn_xy) <= SIM_SPAWN_TOLERANCE_M)
        )
        state_ready = (
            self.reset_released_at is not None
            and self.last_odom_at is not None
            and self.last_odom_at >= self.reset_released_at
            and self.last_steering_at is not None
            and self.last_steering_at >= self.reset_released_at
            and self.last_throttle_at is not None
            and self.last_throttle_at >= self.reset_released_at
            and self.speed_mps is not None and self.speed_mps <= 0.20
            and self.steering_feedback_rad is not None
            and abs(self.steering_feedback_rad) <= 0.02
            and self.throttle_feedback_norm is not None
            and abs(self.throttle_feedback_norm) <= 0.02
            and position_ok
        )
        if not state_ready:
            self.reset_stable_since = None
            return
        if self.reset_stable_since is None:
            self.reset_stable_since = now
            return
        if now - self.reset_stable_since < SIM_RESET_STABLE_SEC:
            return

        if self.spawn_xy is None:
            self.spawn_xy = self.position_xy
        assert self.spawn_xy is not None and self.position_xy is not None
        self.reset_count += 1
        self._publish_event({
            "event": "sim_reset_recovered",
            "profile": self.profile,
            "reason": self.reset_reason,
            "next_phase_index": self.phase_index,
            "reset_index": self.reset_count,
            "spawn_xy": self.spawn_xy,
            "position_xy": self.position_xy,
            "position_error_m": math.dist(self.position_xy, self.spawn_xy),
            "speed_mps": self.speed_mps,
            "monotonic_ns": time.monotonic_ns(),
        })
        self.reset_state = None
        self._start_phase(now)

    def _maybe_start(self) -> None:
        if (self.started_at is not None or self.done or self.speed_mps is None or
                not self.collision_baseline_safe or self.last_collision_at is None):
            return
        now = time.monotonic()
        if self.ready_since is None:
            self.ready_since = now
            return
        if now - self.ready_since < 0.75:
            return
        for topic, publisher in (
            (STEERING_COMMAND_TOPIC, self.steering_pub),
            (THROTTLE_COMMAND_TOPIC, self.throttle_pub),
        ):
            infos = self.node.get_publishers_info_by_topic(topic)
            if len(infos) > 1:
                self._finish(f"conflicting command publishers on {topic}", aborted=True)
                return
            if publisher.get_subscription_count() == 0:
                return
        if ((self.profile == "race_domain_steering_frontier"
             or self.profile == SUBNET_TRANSIENT_PROFILE
             or self.profile in RACE_DOMAIN_SPEED_GOVERNED_PROFILES)
                and (self.reset_pub is None
                     or self.reset_pub.get_subscription_count() == 0)):
            return
        self.started_at = now
        if (self.profile == "race_domain_steering_frontier"
                or self.profile == SUBNET_TRANSIENT_PROFILE
                or self.profile in RACE_DOMAIN_SPEED_GOVERNED_PROFILES):
            self._begin_sim_reset(now, "initial reset to spawn")
        else:
            self._start_phase(now)
            self._neutral("source odom and zero collision baseline ready")
        self.node.get_logger().info(
            f"experiment started at measured speed={self.speed_mps:.3f}m/s")

    @staticmethod
    def _feedforward(speed_target: float) -> float:
        return _nominal_feedforward(speed_target)

    def _phase_command(self, phase: Phase,
                       phase_elapsed_s: float | None = None) -> float:
        assert self.speed_mps is not None
        if phase.throttle_mode in ("fixed", "excitation"):
            return float(phase.throttle_norm or 0.0)
        if phase.throttle_mode == "slew_probe":
            if phase_elapsed_s is None:
                assert self.phase_started_at is not None
                phase_elapsed_s = max(
                    0.0, time.monotonic() - self.phase_started_at)
            return _slew_probe_command(phase, phase_elapsed_s)
        error = phase.speed_target_mps - self.speed_mps
        feedforward = self._feedforward(phase.speed_target_mps)
        if phase.throttle_mode == "race_domain_approach":
            feedforward = race_domain_feedforward(
                phase.speed_target_mps, feedforward)
        if phase.throttle_mode in ("approach", "race_domain_approach"):
            return max(0.0, min(MAX_THROTTLE, feedforward + 0.14 * error))
        return self._speed_hold_command(
            phase.speed_target_mps,
            race_domain=phase.throttle_mode == "race_domain_hold")

    def _speed_hold_command(self, target_speed_mps: float,
                            race_domain: bool = False) -> float:
        assert self.speed_mps is not None
        now = time.monotonic()
        error = target_speed_mps - self.speed_mps
        feedforward = self._feedforward(target_speed_mps)
        if race_domain:
            feedforward = race_domain_feedforward(
                target_speed_mps, feedforward)
        if self.speed_integral_target_mps != target_speed_mps:
            self.speed_integral_throttle = 0.0
            self.speed_integral_target_mps = target_speed_mps
            self.speed_integral_updated_at = now
        dt = (0.0 if self.speed_integral_updated_at is None else
              max(0.0, min(0.10, now - self.speed_integral_updated_at)))
        self.speed_integral_updated_at = now
        proportional = self.speed_hold_kp * error
        unconstrained = feedforward + proportional + self.speed_integral_throttle
        # Conditional integration prevents accumulating error against either
        # throttle limit. Its small bounded trim addresses persistent speed
        # error without making the proportional response more aggressive.
        if (self.speed_hold_ki > 0.0 and
                ((0.0 < unconstrained < MAX_THROTTLE)
                 or (unconstrained <= 0.0 and error > 0.0)
                 or (unconstrained >= MAX_THROTTLE and error < 0.0))):
            self.speed_integral_throttle = max(
                -0.05,
                min(0.05, self.speed_integral_throttle
                    + self.speed_hold_ki * error * dt),
            )
        # Keep the speed loop continuous near its setpoint. Simulator zero
        # throttle applies active braking and caused a repeatable limit cycle
        # around the target when used as a modest-overspeed branch.
        return max(0.0, min(MAX_THROTTLE,
                            feedforward + proportional
                            + self.speed_integral_throttle))

    def _probe_state_metrics(self, now: float) -> tuple[float, float, float] | None:
        recent = [sample for sample in self.state_history
                  if now - sample[0] <= PROBE_START_WINDOW_SEC]
        if len(recent) < PROBE_START_MIN_SAMPLES:
            return None
        return (
            statistics.median(sample[1] for sample in recent),
            statistics.median(abs(sample[2]) for sample in recent),
            statistics.median(abs(sample[3]) for sample in recent),
        )

    def _tick(self) -> None:
        if self.done:
            if self.neutral_ticks_remaining > 0 and self.last_odom_at is not None:
                self._neutral(self.finish_reason)
                self.neutral_ticks_remaining -= 1
            if self.neutral_ticks_remaining == 0:
                self._shutdown_after_neutral()
            return
        now = time.monotonic()
        self._maybe_start()
        if now - self.created_at >= self.timeout_s:
            self._finish("total timeout reached before readiness", aborted=True)
            return
        if self.started_at is None:
            if now - self.last_status_log >= 2.0:
                self.node.get_logger().info("no actuator output until source odom arrives")
                self.last_status_log = now
            return
        elapsed = now - self.started_at
        if elapsed >= self.timeout_s:
            self._finish("total timeout reached", aborted=True)
            return
        # The 40 Hz bridge intentionally pauses packet publication while the
        # simulator reset level is asserted. Reset recovery has its own bounded
        # timeout and must not be preempted by the normal stream-staleness gates.
        if self.reset_state is not None:
            self._tick_sim_reset(now)
            return
        if (self.last_collision_at is not None and
                now - self.last_collision_at > COLLISION_TIMEOUT_SEC):
            self._finish("collision telemetry timeout", aborted=True)
            return
        if self.speed_mps is None:
            if now - self.last_status_log >= 2.0:
                self.node.get_logger().info("no actuator output until source odom arrives")
                self.last_status_log = now
            return
        if self.last_odom_at is None or now - self.last_odom_at > ODOM_TIMEOUT_SEC:
            self._finish("source odometry timeout", aborted=True)
            return
        if (self.tilt_rad is not None
                and self.tilt_rad >= MAX_EXPERIMENT_TILT_RAD):
            self._finish(
                f"vehicle tilt limit: {math.degrees(self.tilt_rad):.1f}deg "
                f">= {math.degrees(MAX_EXPERIMENT_TILT_RAD):.1f}deg",
                aborted=True,
            )
            return
        speed_limit = (RACE_DOMAIN_HARD_LIMIT_MPS
                       if self.profile in ("race_domain_continuous",
                                           "race_domain_brake_boundary",
                                           "race_domain_moderate_braking",
                                           "race_domain_steering_frontier",
                                           *RACE_DOMAIN_SPEED_GOVERNED_PROFILES)
                       else SUBNET_TRANSIENT_MAX_SPEED_MPS
                       if self.profile == SUBNET_TRANSIENT_PROFILE
                       else EMERGENCY_SPEED_MPS)
        if self.speed_mps > speed_limit:
            self._finish(f"emergency speed cutoff: {self.speed_mps:.3f}m/s "
                         f"> {speed_limit:.2f}m/s", aborted=True)
            return
        assert self.phase_started_at is not None
        phase = self.phases[self.phase_index]
        if (phase.settle_before_probe and not self.phase_start_published
                and self.probe_wait_started_at is not None):
            state_metrics = self._probe_state_metrics(now)
            state_ready = (
                state_metrics is not None
                and self.steering_feedback_rad is not None
                and self.last_steering_at is not None
                and now - self.last_steering_at <= STEERING_FEEDBACK_TIMEOUT_SEC
                and abs(state_metrics[0] - phase.speed_target_mps)
                    <= PROBE_START_MAX_SPEED_ERROR_MPS
                and state_metrics[1] <= PROBE_START_MAX_VY_MPS
                and state_metrics[2] <= PROBE_START_MAX_YAW_RATE_RPS
                and abs(self.steering_feedback_rad) <= PROBE_START_MAX_STEERING_RAD
                and (phase.throttle_mode != "slew_probe" or (
                    self.throttle_feedback_norm is not None
                    and self.last_throttle_at is not None
                    and now - self.last_throttle_at <= ODOM_TIMEOUT_SEC
                    and abs(self.throttle_feedback_norm
                            - float(phase.throttle_start_norm))
                        <= THROTTLE_START_FEEDBACK_TOLERANCE))
            )
            if state_ready:
                if self.probe_stable_since is None:
                    self.probe_stable_since = now
                elif now - self.probe_stable_since >= PROBE_START_SETTLE_SEC:
                    self.phase_started_at = now
                    self.probe_wait_started_at = None
                    self.probe_stable_since = None
            else:
                self.probe_stable_since = None

            if self.probe_wait_started_at is not None:
                if now - self.probe_wait_started_at > PROBE_START_TIMEOUT_SEC:
                    self._finish(
                        f"failed to recover matched probe state for {phase.label}",
                        aborted=True,
                    )
                    return
                # Restore a common speed before each fixed-throttle probe;
                # the probe's throttle level begins only after this state is
                # stable, so its starting conditions remain comparable.
                self._publish(0.0, self._speed_hold_command(
                    phase.speed_target_mps,
                    race_domain=(
                        phase.throttle_mode == "race_domain_hold"
                        or phase.probe_race_domain
                        or self.profile in SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES)))
                return

        phase_elapsed = now - self.phase_started_at
        if not self.phase_start_published:
            self.phase_samples.clear()
            self.phase_start_published = True
            state_metrics = self._probe_state_metrics(now)
            self._publish_event({
                "event": "phase_start",
                "profile": self.profile,
                "seed": self.seed,
                "phase_index": self.phase_index,
                "phase_count": len(self.phases),
                "phase_elapsed_s": phase_elapsed,
                "phase_duration_s": phase.duration_s,
                "label": phase.label,
                "target_speed_mps": phase.speed_target_mps,
                "steering_command_rad": phase.steering_rad,
                "steering_profile": phase.steering_profile,
                "steering_amplitude_rad": phase.steering_amplitude_rad,
                "steering_frequency_hz": phase.steering_frequency_hz,
                "steering_frequencies_hz": phase.steering_frequencies_hz,
                "steering_phases_rad": phase.steering_phases_rad,
                "steering_waypoints": phase.steering_waypoints,
                "steering_prbs_levels_normalized": (
                    phase.steering_prbs_levels_normalized),
                "steering_dwell_s": phase.steering_dwell_s,
                "throttle_mode": phase.throttle_mode,
                "speed_hold_kp": self.speed_hold_kp,
                "speed_hold_ki": self.speed_hold_ki,
                "speed_median_gate_mps": self.speed_median_gate_mps,
                "speed_p95_gate_mps": self.speed_p95_gate_mps,
                "fixed_throttle_command_norm": (
                    phase.throttle_norm if phase.throttle_mode == "fixed" else None
                ),
                "requested_throttle_command_norm": (
                    phase.throttle_norm
                    if phase.throttle_mode in ("fixed", "excitation") else None
                ),
                "initial_vx_mps": self.vx_mps,
                "initial_vy_mps": self.vy_mps,
                "initial_yaw_rate_rps": self.yaw_rate_rps,
                "initial_tilt_deg": (
                    math.degrees(self.tilt_rad)
                    if self.tilt_rad is not None else None
                ),
                "initial_steering_rad": self.steering_feedback_rad,
                "initial_throttle_feedback_norm": self.throttle_feedback_norm,
                "initial_window_speed_mps": state_metrics[0] if state_metrics else None,
                "initial_window_abs_vy_mps": state_metrics[1] if state_metrics else None,
                "initial_window_abs_yaw_rate_rps": state_metrics[2] if state_metrics else None,
                "throttle_profile": phase.throttle_profile,
                "throttle_start_norm": phase.throttle_start_norm,
                "throttle_end_norm": phase.throttle_end_norm,
                "throttle_stimulus_delay_s": phase.throttle_stimulus_delay_s,
                "throttle_ramp_duration_s": phase.throttle_ramp_duration_s,
                "condition_pair_id": phase.condition_pair_id,
                "dynamic_coupled_plan_version": (
                    1 if self.profile in DYNAMIC_COUPLED_PROFILES else None),
                "dynamic_coupled_conditions": (
                    dynamic_coupled_plan_as_dicts(self.seed)
                    if (self.profile in DYNAMIC_COUPLED_PROFILES
                        and self.phase_index == 0) else None),
                "dynamic_coupled_command_plan": (
                    [asdict(item) for item in self.phases]
                    if (self.profile in DYNAMIC_COUPLED_PROFILES
                        and self.phase_index == 0) else None),
                "subnet_transient_command_plan": (
                    [asdict(item) for item in self.phases]
                    if (self.profile == SUBNET_TRANSIENT_PROFILE
                        and self.phase_index == 0) else None),
                "subnet_transient_zero_throttle_semantics": (
                    "active_brake_torque; no separate coast command"
                    if self.profile == SUBNET_TRANSIENT_PROFILE else None),
                "subnet_transient_speed_limit_mps": (
                    SUBNET_TRANSIENT_MAX_SPEED_MPS
                    if self.profile == SUBNET_TRANSIENT_PROFILE else None),
                "race_domain_command_plan": [
                    {
                        "target_speed_mps": block.target_speed_mps,
                        "steering_rad": block.steering_rad,
                        "duration_s": block.duration_s,
                        "label": block.label,
                    }
                    for block in self.race_domain_plan
                ],
                "race_domain_zero_throttle_semantics": (
                    "active_brake_torque; no separate coast command"),
                "race_domain_speed_governor_mps": RACE_DOMAIN_GOVERNOR_MPS,
                "race_domain_hard_limit_mps": RACE_DOMAIN_HARD_LIMIT_MPS,
                "race_domain_plan_version": self.race_domain_plan_version,
                "race_domain_boundary_target_mps": RACE_DOMAIN_BOUNDARY_SPEED_MPS,
                "race_domain_throttle_speed_anchors": [
                    {"speed_mps": speed, "throttle_norm": throttle}
                    for speed, throttle in RACE_DOMAIN_THROTTLE_SPEED_ANCHORS
                ],
                "monotonic_ns": time.monotonic_ns(),
            })
        if (phase.throttle_mode == "slew_probe"
                and not self.phase_stimulus_published
                and phase_elapsed >= phase.throttle_stimulus_delay_s):
            self._publish_event({
                "event": "throttle_slew_stimulus",
                "profile": self.profile,
                "seed": self.seed,
                "phase_index": self.phase_index,
                "label": phase.label,
                "condition_pair_id": phase.condition_pair_id,
                "throttle_profile": phase.throttle_profile,
                "throttle_start_norm": phase.throttle_start_norm,
                "throttle_end_norm": phase.throttle_end_norm,
                "steering_command_rad": _phase_steering_command(
                    phase, phase_elapsed),
                "speed_mps": self.speed_mps,
                "vx_mps": self.vx_mps,
                "vy_mps": self.vy_mps,
                "yaw_rate_rps": self.yaw_rate_rps,
                "steering_feedback_rad": self.steering_feedback_rad,
                "throttle_feedback_norm": self.throttle_feedback_norm,
                "phase_elapsed_s": phase_elapsed,
                "monotonic_ns": time.monotonic_ns(),
            })
            self.phase_stimulus_published = True
        speed_target_reached = (
            abs(self.speed_mps - phase.speed_target_mps) <= 0.10
            if phase.require_speed_target_match
            else self.speed_mps >= phase.speed_target_mps - 0.10
        )
        if phase.reach_speed_target and speed_target_reached:
            self.node.get_logger().info(
                f"phase complete: {phase.label}, measured_speed={self.speed_mps:.3f}m/s")
            self._next_phase(now)
            return
        if phase_elapsed >= phase.duration_s:
            if phase.reach_speed_target:
                self._finish(
                    f"failed to reach {phase.speed_target_mps:.1f}m/s before approach deadline "
                    f"(measured {self.speed_mps:.3f}m/s)",
                    aborted=True,
                )
                return
            self._next_phase(now)
            return

        if self.profile in ("race_domain_continuous",
                            "race_domain_brake_boundary",
                            "race_domain_moderate_braking"):
            block_index = min(
                int(phase_elapsed // self.race_domain_plan[0].duration_s),
                len(self.race_domain_plan) - 1)
            block = self.race_domain_plan[block_index]
            if (self.profile in ("race_domain_brake_boundary",
                                 "race_domain_moderate_braking")
                    and block_index != self.last_race_domain_block_index):
                self._publish_event({
                    "event": "race_domain_block_start",
                    "profile": self.profile,
                    "seed": self.seed,
                    "phase_index": self.phase_index,
                    "block_index": block_index,
                    "label": block.label,
                    "target_speed_mps": block.target_speed_mps,
                    "steering_command_rad": block.steering_rad,
                    "measured_speed_mps": self.speed_mps,
                    "steering_feedback_rad": self.steering_feedback_rad,
                    "throttle_feedback_norm": self.throttle_feedback_norm,
                    "tilt_rad": self.tilt_rad,
                    "monotonic_ns": time.monotonic_ns(),
                })
                self.last_race_domain_block_index = block_index
            throttle = self._speed_hold_command(
                block.target_speed_mps, race_domain=True)
            if self.speed_mps >= block.target_speed_mps + 0.50:
                # In this simulator zero throttle is active braking.
                throttle = 0.0
            steering_limit = (
                race_domain_moderate_steering_limit(self.speed_mps)
                if self.profile == "race_domain_moderate_braking" else
                steering_limit_for_speed(self.speed_mps))
            steering = max(-steering_limit,
                           min(steering_limit, block.steering_rad))
            if self.speed_mps >= RACE_DOMAIN_GOVERNOR_MPS:
                throttle = 0.0
                steering = 0.0
                self.phase_governor_ticks += 1
            self._publish(steering, throttle)
            return

        throttle = self._phase_command(phase, phase_elapsed)
        # Speed-regulated phases get a target-speed guard. Fixed-input probes
        # remain fixed except for explicit excitation-mode safety governing.
        if (phase.throttle_mode not in ("fixed", "excitation", "slew_probe") and
                self.speed_mps >= phase.speed_target_mps + 0.50):
            throttle = 0.0
        if (phase.throttle_mode == "excitation" and
                self.speed_mps >= EXCITATION_SPEED_GOVERNOR_MPS and throttle > 0.0):
            throttle = 0.0
            self.phase_governor_ticks += 1
        if (self.profile in RACE_DOMAIN_SPEED_GOVERNED_PROFILES
                and self.speed_mps >= RACE_DOMAIN_GOVERNOR_MPS
                and throttle > 0.0):
            # Keep the captured motion inside the handoff's 11.2 m/s support;
            # retain the requested steering so the state at the boundary is
            # still observed rather than replaced with an artificial straight.
            throttle = 0.0
            self.phase_governor_ticks += 1
        if (self.profile == SUBNET_TRANSIENT_PROFILE
                and self.speed_mps >= SUBNET_TRANSIENT_MAX_SPEED_MPS
                and throttle > 0.0):
            throttle = 0.0
            self.phase_governor_ticks += 1
        steering = _phase_steering_command(phase, phase_elapsed)
        self._publish(steering, throttle)

    def _next_phase(self, now: float) -> None:
        previous_phase = self.phases[self.phase_index]
        self._close_phase("complete")
        self.phase_index += 1
        if self.phase_index >= len(self.phases):
            self._finish("schedule complete", aborted=False)
            return
        next_phase = self.phases[self.phase_index]
        if (self.profile == "race_domain_steering_frontier"
                and previous_phase.label.startswith("frontier_")):
            self._begin_sim_reset(now, f"completed {previous_phase.label}")
        elif (self.profile in DYNAMIC_COUPLED_PROFILES
              and next_phase.label.startswith("approach_coupled_")):
            self._begin_sim_reset(
                now, f"completed {previous_phase.condition_pair_id}")
        elif (self.profile == SUBNET_TRANSIENT_PROFILE
              and next_phase.label.startswith("approach_subnet_")):
            self._begin_sim_reset(
                now, f"completed {previous_phase.condition_pair_id}")
        elif (self.profile in YAW_TRANSIENT_PROFILES
              and _is_yaw_transient_approach(next_phase.label)):
            self._begin_sim_reset(
                now, f"completed {previous_phase.condition_pair_id}")
        elif (self.profile in SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES
              and next_phase.label.startswith("approach_swerve_pair_")):
            self._begin_sim_reset(
                now, f"completed {previous_phase.condition_pair_id}")
        else:
            self._start_phase(now)

    def _publish_event(self, event: dict[str, object]) -> None:
        event.setdefault("wall_time_ns", time.time_ns())
        self.phase_event_queue.put(
            json.dumps(event, separators=(",", ":"), sort_keys=True))

    def _publish_phase_events(self) -> None:
        while True:
            serialized = self.phase_event_queue.get()
            try:
                if serialized is None:
                    return
                self.phase_pub.publish(String(data=serialized))
            except Exception as exc:
                self.phase_event_failures += 1
                self.node.get_logger().error(
                    f"phase diagnostic publish failed: {exc}")
            finally:
                self.phase_event_queue.task_done()

    def _close_phase_event_worker(self) -> None:
        flush_deadline = time.monotonic() + 2.0
        while (self.phase_event_queue.unfinished_tasks > 0
               and time.monotonic() < flush_deadline):
            time.sleep(0.005)
        if self.phase_event_queue.unfinished_tasks > 0:
            self.phase_event_failures += 1
        self.phase_event_queue.put(None)
        self.phase_event_worker.join(timeout=2.0)
        if self.phase_event_worker.is_alive():
            self.phase_event_failures += 1

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    def _close_phase(self, status: str, reason: str = "") -> None:
        phase = self.phases[self.phase_index]
        speed_errors = [abs(speed - phase.speed_target_mps)
                        for speed, _ in self.phase_samples]
        steering_errors = [error for _, error in self.phase_samples]
        metrics: dict[str, object] = {
            "event": "phase_end",
            "profile": self.profile,
            "seed": self.seed,
            "phase_index": self.phase_index,
            "label": phase.label,
            "status": status,
            "reason": reason,
            "samples": len(self.phase_samples),
            "measured_speed_max_mps": self.phase_max_speed_mps,
            "measured_tilt_max_deg": math.degrees(self.phase_max_tilt_rad),
            "speed_governor_ticks": self.phase_governor_ticks,
            "requested_throttle_command_norm": (
                phase.throttle_norm
                if phase.throttle_mode in ("fixed", "excitation") else None
            ),
            "throttle_profile": phase.throttle_profile,
            "throttle_start_norm": phase.throttle_start_norm,
            "throttle_end_norm": phase.throttle_end_norm,
            "throttle_stimulus_delay_s": phase.throttle_stimulus_delay_s,
            "throttle_ramp_duration_s": phase.throttle_ramp_duration_s,
            "condition_pair_id": phase.condition_pair_id,
            "speed_error_median_mps": self._percentile(speed_errors, 0.5),
            "speed_error_p95_mps": self._percentile(speed_errors, 0.95),
            "steering_error_p95_rad": self._percentile(steering_errors, 0.95),
            "speed_validation_enabled": phase.validate_speed,
            "steering_validation_enabled": phase.validate_steering,
            "monotonic_ns": time.monotonic_ns(),
        }
        if phase.validate_samples:
            failures = []
            if status == "aborted":
                failures.append("phase_aborted")
            if len(self.phase_samples) < MIN_PHASE_SAMPLES:
                failures.append(f"samples<{MIN_PHASE_SAMPLES}")
            median_error = metrics["speed_error_median_mps"]
            p95_error = metrics["speed_error_p95_mps"]
            steering_error = metrics["steering_error_p95_rad"]
            if (phase.validate_speed and
                    (median_error is None or median_error > self.speed_median_gate_mps)):
                failures.append("speed_median")
            if (phase.validate_speed and
                    (p95_error is None or p95_error > self.speed_p95_gate_mps)):
                failures.append("speed_p95")
            if (phase.validate_steering
                    and (steering_error is None
                         or steering_error > MAX_STEERING_P95_ERROR_RAD)):
                failures.append("steering_p95")
            metrics["valid"] = not failures
            metrics["quality_failures"] = failures
            if failures:
                failure = f"{phase.label}:{','.join(failures)}"
                self.quality_failures.append(failure)
        else:
            metrics["valid"] = None
        self._publish_event(metrics)
        self.phase_start_published = False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Development-only open-plane actuator excitation (explore simulator only).")
    parser.add_argument("--seed", type=int, default=20260926,
                        help="deterministic steering phase order (default: 20260926)")
    parser.add_argument("--profile", choices=("high_angle_boundary", "isolated_boundary",
                                               "isolated_speed_sweep", "isolated_force_3mps",
                                               "isolated_force_4mps",
                                               "isolated_force_5mps",
                                               "isolated_highspeed_surface",
                                               "isolated_highsteer_75_long",
                                               "isolated_highsteer_multispeed",
                                               "race_domain_dynamic_steering",
                                               LOW_SPEED_TRANSIENT_PROFILE,
                                               YAW_TRANSIENT_PROFILE,
                                               YAW_ATLAS_INTERPOLATION_PROFILE,
                                               YAW_ATLAS_OFFGRID_FINAL_PROFILE,
                                               YAW_ATLAS_EXTRATREES_FINAL_PROFILE,
                                               YAW_FULLBAND_GAPFILL_PROFILE,
                                               YAW_UNWIND_THROTTLE_PROFILE,
                                               YAW_LOW_ANGLE_RATE_PROFILE,
                                               "isolated_3to5_response_surface",
                                               "isolated_highspeed_crossfactor",
                                               "isolated_highspeed_tail",
                                               "isolated_transition_65mps",
                                               "isolated_transition_45mps",
                                               "isolated_transition_speed_surface",
                                               "isolated_transition_support",
                                               "isolated_transition_bridge",
                                               "isolated_transition_low_support",
                                               "isolated_transition_full_surface",
                                               "transient_4mps",
                                               "transient_fullsteer_4mps",
                                               "transient_transition_4mps",
                                               "transient_transition_4mps_fixedthrottle",
                                               "transient_transition_dwell_4mps_fixedthrottle",
                                               "throttle_slew_pair",
                                               *SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES,
                                               "full_input_excitation",
                                               "race_domain_continuous",
                                               "race_domain_brake_boundary",
                                               "race_domain_moderate_braking",
                                               "race_domain_steering_frontier",
                                               *DYNAMIC_COUPLED_PROFILES,
                                               SUBNET_TRANSIENT_PROFILE, "grid"),
                        default="high_angle_boundary",
                        help="isolated profiles recover near-straight speed/yaw/lateral-velocity state before each probe")
    parser.add_argument("--timeout-s", type=float, default=150.0,
                        help="whole experiment timeout, bounded to (0, 1200] seconds")
    parser.add_argument("--transition-speed-mps", type=float, default=4.5,
                        help="speed for transition surface/support profiles (3–8 m/s)")
    parser.add_argument("--probe-dwell-s", type=float, default=0.0,
                        help="override valid probe-phase dwell only (0 keeps profile defaults; max 15 s)")
    parser.add_argument("--speed-hold-kp", type=float, default=0.04,
                        help="proportional throttle correction per m/s speed error")
    parser.add_argument("--speed-hold-ki", type=float, default=0.0,
                        help="integral throttle correction per m/s-second error")
    parser.add_argument("--speed-median-gate-mps", type=float,
                        default=MAX_SPEED_MEDIAN_ERROR_MPS,
                        help="maximum per-phase median speed error")
    parser.add_argument("--speed-p95-gate-mps", type=float,
                        default=MAX_SPEED_P95_ERROR_MPS,
                        help="maximum per-phase p95 speed error")
    parser.add_argument("--throttle-rate-sweep-delta-norm", type=float,
                        default=SWERVE_THROTTLE_SLEW_DELTA_NORM,
                        help="positive throttle increment for the high-steer rate-sweep profile")
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or not 0.0 < args.timeout_s <= 1200.0:
        parser.error("--timeout-s must be greater than 0 and no more than 1200")
    if args.profile == "race_domain_continuous" and args.timeout_s < 265.0:
        parser.error("race_domain_continuous requires --timeout-s >= 265")
    if args.profile == "race_domain_brake_boundary":
        required = plan_duration_s(build_race_domain_boundary_plan(args.seed)) + 5.0
        if args.timeout_s < required:
            parser.error(
                f"race_domain_brake_boundary requires --timeout-s >= {required:g}")
    if args.profile == "race_domain_moderate_braking":
        required = plan_duration_s(
            build_race_domain_moderate_braking_plan(args.seed)) + 5.0
        if args.timeout_s < required:
            parser.error(
                f"race_domain_moderate_braking requires --timeout-s >= {required:g}")
    if args.profile == "race_domain_steering_frontier":
        schedule = build_schedule(args.seed, args.profile,
                                  args.transition_speed_mps)
        reset_cycles = sum(
            phase.label.startswith("approach_frontier_")
            for phase in schedule)
        required = (
            sum(phase.duration_s for phase in schedule)
            + sum(phase.settle_before_probe for phase in schedule)
            * PROBE_START_TIMEOUT_SEC
            + reset_cycles * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        if args.timeout_s < required:
            parser.error(
                f"race_domain_steering_frontier requires --timeout-s >= {required:g}")
    if args.profile in DYNAMIC_COUPLED_PROFILES:
        schedule = build_schedule(args.seed, args.profile,
                                  args.transition_speed_mps)
        reset_cycles = len(build_dynamic_coupled_plan(args.seed))
        required = (
            sum(phase.duration_s for phase in schedule)
            + reset_cycles * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        if args.timeout_s < required:
            parser.error(
                f"{args.profile} requires --timeout-s >= {required:g}")
    if args.profile in SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES:
        schedule = build_schedule(args.seed, args.profile,
                                  args.transition_speed_mps,
                                  args.throttle_rate_sweep_delta_norm)
        reset_cycles = sum(
            phase.label.startswith("approach_swerve_pair_")
            for phase in schedule)
        required = (
            sum(phase.duration_s for phase in schedule)
            + sum(phase.settle_before_probe for phase in schedule)
            * PROBE_START_TIMEOUT_SEC
            + reset_cycles * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        if args.timeout_s < required:
            parser.error(
                f"{args.profile} requires --timeout-s >= {required:g}")
    if args.profile == "isolated_highsteer_75_long":
        schedule = build_schedule(args.seed, args.profile,
                                  args.transition_speed_mps)
        required = (sum(phase.duration_s for phase in schedule)
                    + sum(phase.settle_before_probe for phase in schedule)
                    * PROBE_START_TIMEOUT_SEC + 5.0)
        if args.timeout_s < required:
            parser.error(
                f"isolated_highsteer_75_long requires --timeout-s >= {required:g}")
    if args.profile == "isolated_highsteer_multispeed":
        schedule = build_schedule(args.seed, args.profile,
                                  args.transition_speed_mps)
        required = (sum(phase.duration_s for phase in schedule)
                    + sum(phase.settle_before_probe for phase in schedule)
                    * PROBE_START_TIMEOUT_SEC + 5.0)
        if args.timeout_s < required:
            parser.error(
                "isolated_highsteer_multispeed requires "
                f"--timeout-s >= {required:g}")
    if args.profile == "race_domain_dynamic_steering":
        required = sum(phase.duration_s for phase in build_schedule(
            args.seed, args.profile, args.transition_speed_mps)) + 5.0
        if args.timeout_s < required:
            parser.error(
                f"race_domain_dynamic_steering requires --timeout-s >= {required:g}")
    if args.profile in YAW_TRANSIENT_PROFILES:
        schedule = build_schedule(args.seed, args.profile,
                                  args.transition_speed_mps)
        reset_cycles = sum(
            _is_yaw_transient_approach(phase.label)
            for phase in schedule)
        required = (
            sum(phase.duration_s for phase in schedule)
            + reset_cycles * (SIM_RESET_HOLD_SEC + SIM_RESET_TIMEOUT_SEC)
            + 5.0
        )
        if args.timeout_s < required:
            parser.error(
                f"{args.profile} requires --timeout-s >= {required:g}")

    rclpy.init()
    if (not math.isfinite(args.transition_speed_mps)
            or (args.profile in ("isolated_transition_speed_surface",
                                 "isolated_transition_support",
                                 "isolated_transition_bridge",
                                 "isolated_transition_low_support",
                                 "isolated_transition_full_surface",
                                 "throttle_slew_pair")
                and not 3.0 <= args.transition_speed_mps <= 9.0)):
        parser.error("--transition-speed-mps must be in [3, 9] for transition profiles")
    if (args.profile == "throttle_slew_pair"
            and args.transition_speed_mps not in (4.5, 6.5)):
        parser.error("throttle_slew_pair supports only 4.5 or 6.5 m/s")
    if (not math.isfinite(args.speed_hold_kp) or args.speed_hold_kp < 0.0 or
            not math.isfinite(args.speed_hold_ki) or args.speed_hold_ki < 0.0 or
            not math.isfinite(args.speed_median_gate_mps) or
            args.speed_median_gate_mps <= 0.0 or
            not math.isfinite(args.speed_p95_gate_mps) or
            args.speed_p95_gate_mps <= 0.0):
        parser.error("speed controller gains must be nonnegative and speed gates positive")
    if (not math.isfinite(args.throttle_rate_sweep_delta_norm)
            or args.throttle_rate_sweep_delta_norm <= 0.0
            or args.throttle_rate_sweep_delta_norm > MAX_THROTTLE):
        parser.error("--throttle-rate-sweep-delta-norm must be in (0, 0.50]")
    if (not math.isfinite(args.probe_dwell_s) or
            not 0.0 <= args.probe_dwell_s <= 15.0):
        parser.error("--probe-dwell-s must be in [0, 15]")
    if (args.probe_dwell_s > 0.0 and
            args.profile in ("full_input_excitation", "grid",
                             *DYNAMIC_COUPLED_PROFILES,
                             SUBNET_TRANSIENT_PROFILE,
                             *YAW_TRANSIENT_PROFILES,
                             *SWERVE_THROTTLE_SLEW_CAPTURE_PROFILES)):
        parser.error("--probe-dwell-s is unsupported for this fixed capture profile")
    experiment = OpenPlaneExcitation(args.seed, args.timeout_s, args.profile,
                                    args.transition_speed_mps,
                                    args.probe_dwell_s,
                                    args.speed_hold_kp,
                                    args.speed_hold_ki,
                                    args.speed_median_gate_mps,
                                    args.speed_p95_gate_mps,
                                    args.throttle_rate_sweep_delta_norm)
    interrupted = False

    def stop_on_signal(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        experiment._finish("interrupted", aborted=True)

    previous_sigint = signal.signal(signal.SIGINT, stop_on_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, stop_on_signal)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(experiment.node)
    try:
        while rclpy.ok() and not experiment.shutdown_sent:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if not experiment.done:
            experiment._finish("interrupted" if interrupted else "executor stopped", aborted=True)
        experiment._close_phase_event_worker()
        executor.remove_node(experiment.node)
        executor.shutdown(timeout_sec=1.0)
        experiment.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 1 if (experiment.aborted or experiment.quality_failures
                 or experiment.phase_event_failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
