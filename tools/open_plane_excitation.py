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
from dataclasses import dataclass

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32, Int32, String


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
ODOM_TOPIC = "/autodrive/roboracer_1/odom"
STEERING_TOPIC = "/autodrive/roboracer_1/steering"
COLLISION_TOPIC = "/autodrive/roboracer_1/collision_count"
STEERING_COMMAND_TOPIC = "/autodrive/roboracer_1/steering_command"
THROTTLE_COMMAND_TOPIC = "/autodrive/roboracer_1/throttle_command"
PHASE_TOPIC = "/open_plane_experiment/phase"


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
    settle_before_probe: bool = False


def build_schedule(seed: int, profile: str = "high_angle_boundary",
                  transition_speed_mps: float = 4.5) -> list[Phase]:
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
                 speed_hold_kp: float = 0.04,
                 speed_hold_ki: float = 0.0,
                 speed_median_gate_mps: float = MAX_SPEED_MEDIAN_ERROR_MPS,
                 speed_p95_gate_mps: float = MAX_SPEED_P95_ERROR_MPS) -> None:
        if not math.isfinite(speed_hold_kp) or speed_hold_kp < 0.0:
            raise ValueError("speed-hold proportional gain must be finite and nonnegative")
        if not math.isfinite(speed_hold_ki) or speed_hold_ki < 0.0:
            raise ValueError("speed-hold integral gain must be finite and nonnegative")
        if (not math.isfinite(speed_median_gate_mps) or
                speed_median_gate_mps <= 0.0 or
                not math.isfinite(speed_p95_gate_mps) or
                speed_p95_gate_mps <= 0.0):
            raise ValueError("speed-error gates must be finite and positive")
        self.node = rclpy.create_node("open_plane_excitation")
        self.seed = seed
        self.profile = profile
        self.phases = build_schedule(seed, profile, transition_speed_mps)
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
        self.speed_mps: float | None = None
        self.vx_mps: float | None = None
        self.vy_mps: float | None = None
        self.yaw_rate_rps: float | None = None
        self.tilt_rad: float | None = None
        self.state_history: deque[tuple[float, float, float, float]] = deque(maxlen=64)
        self.last_odom_at: float | None = None
        self.steering_feedback_rad: float | None = None
        self.last_steering_at: float | None = None
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
        self.published_count = 0
        self.last_status_log = 0.0
        self.neutral_ticks_remaining = 0
        self.shutdown_sent = False
        self.finish_reason = ""

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
        self.collision_sub = self.node.create_subscription(
            Int32, COLLISION_TOPIC, self._on_collision, sensor_qos)
        self.timer = self.node.create_timer(PERIOD_SEC, self._tick)
        nominal_schedule_s = sum(phase.duration_s for phase in self.phases)
        reset_budget_s = (
            sum(phase.settle_before_probe for phase in self.phases)
            * PROBE_START_TIMEOUT_SEC
        )
        self.node.get_logger().info(
            f"waiting for source odom and zero collision count; profile={profile}, seed={seed}, "
            f"phases={len(self.phases)}, "
            f"schedule_max={nominal_schedule_s + reset_budget_s:.1f}s "
            f"(nominal={nominal_schedule_s:.1f}s, reset_budget={reset_budget_s:.1f}s), "
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
        tilt_cosine = 1.0 - 2.0 * (
            float(orientation.x) ** 2 + float(orientation.y) ** 2)
        tilt = math.acos(max(-1.0, min(1.0, tilt_cosine)))
        if not all(map(math.isfinite, (vx, vy, yaw_rate, speed, tilt))):
            return
        now = time.monotonic()
        self.speed_mps = speed
        self.vx_mps = vx
        self.vy_mps = vy
        self.yaw_rate_rps = yaw_rate
        self.tilt_rad = tilt
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
        self.started_at = now
        self.phase_started_at = now
        self.probe_wait_started_at = (
            now if self.phases[0].settle_before_probe else None
        )
        self._neutral("source odom and zero collision baseline ready")
        self.node.get_logger().info(
            f"experiment started at measured speed={self.speed_mps:.3f}m/s")

    @staticmethod
    def _feedforward(speed_target: float) -> float:
        if speed_target <= SPEED_TARGETS_MPS[0]:
            return THROTTLE_FEEDFORWARD[0]
        for index in range(1, len(SPEED_TARGETS_MPS)):
            if speed_target <= SPEED_TARGETS_MPS[index]:
                low_speed, high_speed = SPEED_TARGETS_MPS[index - 1:index + 1]
                low_throttle, high_throttle = THROTTLE_FEEDFORWARD[index - 1:index + 1]
                ratio = (speed_target - low_speed) / (high_speed - low_speed)
                return low_throttle + ratio * (high_throttle - low_throttle)
        return THROTTLE_FEEDFORWARD[-1]

    def _phase_command(self, phase: Phase) -> float:
        assert self.speed_mps is not None
        if phase.throttle_mode in ("fixed", "excitation"):
            return float(phase.throttle_norm or 0.0)
        error = phase.speed_target_mps - self.speed_mps
        feedforward = self._feedforward(phase.speed_target_mps)
        if phase.throttle_mode == "approach":
            return max(0.0, min(MAX_THROTTLE, feedforward + 0.14 * error))
        return self._speed_hold_command(phase.speed_target_mps)

    def _speed_hold_command(self, target_speed_mps: float) -> float:
        assert self.speed_mps is not None
        now = time.monotonic()
        error = target_speed_mps - self.speed_mps
        feedforward = self._feedforward(target_speed_mps)
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
        if (self.last_collision_at is not None and
                now - self.last_collision_at > COLLISION_TIMEOUT_SEC):
            self._finish("collision telemetry timeout", aborted=True)
            return
        if self.started_at is None or self.speed_mps is None:
            if now - self.last_status_log >= 2.0:
                self.node.get_logger().info("no actuator output until source odom arrives")
                self.last_status_log = now
            return
        if self.last_odom_at is None or now - self.last_odom_at > ODOM_TIMEOUT_SEC:
            self._finish("source odometry timeout", aborted=True)
            return
        elapsed = now - self.started_at
        if elapsed >= self.timeout_s:
            self._finish("total timeout reached", aborted=True)
            return
        if (self.tilt_rad is not None
                and self.tilt_rad >= MAX_EXPERIMENT_TILT_RAD):
            self._finish(
                f"vehicle tilt limit: {math.degrees(self.tilt_rad):.1f}deg "
                f">= {math.degrees(MAX_EXPERIMENT_TILT_RAD):.1f}deg",
                aborted=True,
            )
            return
        if self.speed_mps > EMERGENCY_SPEED_MPS:
            self._finish(f"emergency speed cutoff: {self.speed_mps:.3f}m/s", aborted=True)
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
                self._publish(0.0, self._speed_hold_command(phase.speed_target_mps))
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
                "label": phase.label,
                "target_speed_mps": phase.speed_target_mps,
                "steering_command_rad": phase.steering_rad,
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
                "initial_window_speed_mps": state_metrics[0] if state_metrics else None,
                "initial_window_abs_vy_mps": state_metrics[1] if state_metrics else None,
                "initial_window_abs_yaw_rate_rps": state_metrics[2] if state_metrics else None,
                "monotonic_ns": time.monotonic_ns(),
            })
        if phase.reach_speed_target and self.speed_mps >= phase.speed_target_mps - 0.10:
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

        throttle = self._phase_command(phase)
        # Speed-regulated phases get a target-speed guard. Fixed-input probes
        # must remain fixed so their actuator command is an identified input;
        # the global emergency-speed cutoff above still bounds the experiment.
        if (phase.throttle_mode not in ("fixed", "excitation") and
                self.speed_mps >= phase.speed_target_mps + 0.50):
            throttle = 0.0
        if (phase.throttle_mode == "excitation" and
                self.speed_mps >= EXCITATION_SPEED_GOVERNOR_MPS and throttle > 0.0):
            throttle = 0.0
            self.phase_governor_ticks += 1
        steering = phase.steering_rad
        self._publish(steering, throttle)

    def _next_phase(self, now: float) -> None:
        self._close_phase("complete")
        self.phase_index += 1
        if self.phase_index >= len(self.phases):
            self._finish("schedule complete", aborted=False)
            return
        self.phase_started_at = now
        next_phase = self.phases[self.phase_index]
        self.probe_wait_started_at = now if next_phase.settle_before_probe else None
        self.probe_stable_since = None
        self.phase_samples.clear()
        self.phase_max_speed_mps = self.speed_mps or 0.0
        self.phase_max_tilt_rad = self.tilt_rad or 0.0
        self.phase_governor_ticks = 0
        self.phase_start_published = False

    def _publish_event(self, event: dict[str, object]) -> None:
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
            "speed_error_median_mps": self._percentile(speed_errors, 0.5),
            "speed_error_p95_mps": self._percentile(speed_errors, 0.95),
            "steering_error_p95_rad": self._percentile(steering_errors, 0.95),
            "speed_validation_enabled": phase.validate_speed,
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
            if steering_error is None or steering_error > MAX_STEERING_P95_ERROR_RAD:
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
                                               "isolated_3to5_response_surface",
                                               "isolated_highspeed_crossfactor",
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
                                               "full_input_excitation", "grid"),
                        default="high_angle_boundary",
                        help="isolated profiles recover near-straight speed/yaw/lateral-velocity state before each probe")
    parser.add_argument("--timeout-s", type=float, default=150.0,
                        help="whole experiment timeout, bounded to (0, 1200] seconds")
    parser.add_argument("--transition-speed-mps", type=float, default=4.5,
                        help="speed for transition surface/support profiles (3–8 m/s)")
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
    args = parser.parse_args()
    if not math.isfinite(args.timeout_s) or not 0.0 < args.timeout_s <= 1200.0:
        parser.error("--timeout-s must be greater than 0 and no more than 1200")

    rclpy.init()
    if (not math.isfinite(args.transition_speed_mps)
            or (args.profile in ("isolated_transition_speed_surface",
                                 "isolated_transition_support",
                                 "isolated_transition_bridge",
                                 "isolated_transition_low_support",
                                 "isolated_transition_full_surface")
                and not 3.0 <= args.transition_speed_mps <= 9.0)):
        parser.error("--transition-speed-mps must be in [3, 9] for transition profiles")
    if (not math.isfinite(args.speed_hold_kp) or args.speed_hold_kp < 0.0 or
            not math.isfinite(args.speed_hold_ki) or args.speed_hold_ki < 0.0 or
            not math.isfinite(args.speed_median_gate_mps) or
            args.speed_median_gate_mps <= 0.0 or
            not math.isfinite(args.speed_p95_gate_mps) or
            args.speed_p95_gate_mps <= 0.0):
        parser.error("speed controller gains must be nonnegative and speed gates positive")
    experiment = OpenPlaneExcitation(args.seed, args.timeout_s, args.profile,
                                    args.transition_speed_mps,
                                    args.speed_hold_kp,
                                    args.speed_hold_ki,
                                    args.speed_median_gate_mps,
                                    args.speed_p95_gate_mps)
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
