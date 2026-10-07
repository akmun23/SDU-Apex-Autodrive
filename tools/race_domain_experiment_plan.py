"""Deterministic, speed-conditioned commands for race-domain plant captures."""

from __future__ import annotations

from dataclasses import dataclass
import random


BLOCK_DURATION_S = 4.0
RACE_DOMAIN_GOVERNOR_MPS = 11.2
RACE_DOMAIN_HARD_LIMIT_MPS = 11.9
RACE_DOMAIN_SPEEDS_MPS = (2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 10.5, 11.0)
RACE_DOMAIN_BOUNDARY_SPEED_MPS = 11.1
RACE_DOMAIN_COMBINED_STEER_LIMITS = {
    9.5: 0.09,
    10.5: 0.09,
    11.1: 0.05,
}
RACE_DOMAIN_MODERATE_STEER_LIMITS = {
    9.5: 0.14,
    10.5: 0.14,
    11.1: 0.12,
}
HIGH_STEER_VALIDATION_SPEED_MPS = 7.5
HIGH_STEER_VALIDATION_DWELL_S = 7.5
HIGH_STEER_VALIDATION_REPETITIONS = 1
HIGH_STEER_VALIDATION_ANGLES_RAD = (0.20, 0.25, 0.30, 0.35,
                                     0.42, 0.46, 0.50, 0.5236)
RACE_DOMAIN_STEERING_FRONTIER_SPEEDS_MPS = (9.5, 10.5, 11.1)
RACE_DOMAIN_STEERING_FRONTIER_ANGLES_RAD = {
    9.5: (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
    10.5: (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20),
    11.1: (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18),
}
RACE_DOMAIN_STEERING_FRONTIER_DWELL_S = 3.0
# Preliminary only: stable, near-straight throttle-sweep plateaus in
# openplane_throttle_5pct_5deg_20260930_r05_resume yielded approximately
# (8.49 m/s, 0.35), (9.66 m/s, 0.40), and (10.82 m/s, 0.45). This one-run
# inverse map is used only to excite the missing 9–12 m/s capture region; it
# is not a learned production control law and must be rechecked in new runs.
RACE_DOMAIN_THROTTLE_SPEED_ANCHORS = (
    (8.49, 0.35),
    (9.66, 0.40),
    (10.82, 0.45),
)


def race_domain_feedforward(target_speed_mps: float,
                            nominal_feedforward: float) -> float:
    """Use measured near-straight throttle plateaus above the old 8.5 m/s cap."""
    if target_speed_mps <= RACE_DOMAIN_THROTTLE_SPEED_ANCHORS[0][0]:
        return nominal_feedforward
    lower_speed, lower_throttle = RACE_DOMAIN_THROTTLE_SPEED_ANCHORS[-2]
    upper_speed, upper_throttle = RACE_DOMAIN_THROTTLE_SPEED_ANCHORS[-1]
    slope = (upper_throttle - lower_throttle) / (upper_speed - lower_speed)
    throttle = upper_throttle + slope * (target_speed_mps - upper_speed)
    return max(0.0, min(0.5, throttle))


@dataclass(frozen=True)
class CommandBlock:
    target_speed_mps: float
    steering_rad: float
    duration_s: float = BLOCK_DURATION_S
    label: str = ""


_STEERING_BY_SPEED = {
    2.0: (0.0, -0.25, 0.25, -0.42, 0.42, -0.5236, 0.5236),
    3.5: (0.0, -0.15, 0.15, -0.30, 0.30, -0.42, 0.42),
    5.0: (0.0, -0.10, 0.10, -0.20, 0.20, -0.30, 0.30),
    6.5: (0.0, -0.05, 0.05, -0.10, 0.10, -0.20, 0.20),
    8.0: (0.0, -0.03, 0.03, -0.06, 0.06, -0.10, 0.10),
    9.5: (0.0, -0.02, 0.02, -0.04, 0.04, -0.08, 0.08),
    10.5: (0.0, -0.02, 0.02, -0.04, 0.04, -0.08, 0.08),
    11.0: (0.0, -0.01, 0.01, -0.02, 0.02, -0.04, 0.04),
}


def steering_limit_for_speed(speed_mps: float) -> float:
    """Conservative test-input envelope, grounded in existing clean samples."""
    if speed_mps < 0.0:
        raise ValueError("speed must be nonnegative")
    if speed_mps <= 3.0:
        return 0.5236
    if speed_mps <= 5.0:
        return 0.42
    if speed_mps <= 7.0:
        return 0.30
    if speed_mps <= 9.0:
        return 0.20
    return 0.10


def build_race_domain_plan(
        seed: int, boundary_speed_mps: float = 11.0
        ) -> tuple[CommandBlock, ...]:
    """Cover feasible speed/steering cells, exits and throttle-reduction phases.

    The primary grid is speed-ordered to avoid demanding abrupt, unrealistic
    speed jumps. Steering sign/order within each speed band is randomized.
    Speed targets above 9 m/s use only small steering angles because existing
    captures contain no supported evidence for larger angles at those speeds.
    """
    rng = random.Random(seed)
    if not 10.5 < boundary_speed_mps < RACE_DOMAIN_GOVERNOR_MPS:
        raise ValueError("boundary speed must be above 10.5 and below the governor")
    speeds = (*RACE_DOMAIN_SPEEDS_MPS[:-1], boundary_speed_mps)
    blocks: list[CommandBlock] = []
    for speed in speeds:
        steering_values = list(_STEERING_BY_SPEED.get(
            speed, _STEERING_BY_SPEED[11.0]))
        rng.shuffle(steering_values)
        blocks.extend(
            CommandBlock(speed, steering, label=f"grid_{speed:g}_{steering:+.3f}")
            for steering in steering_values)

    # A rising speed with decreasing steering represents turn exit/unwind and
    # renewed drive; the following descending targets exercise cuts/braking
    # while retaining small, racing-relevant steering demand.
    corner_exit = ((5.0, 0.30), (6.5, 0.20), (8.0, 0.10), (9.5, 0.06))
    for index, (speed, steering) in enumerate(corner_exit):
        sign = -1.0 if rng.getrandbits(1) else 1.0
        blocks.append(CommandBlock(
            speed, sign * steering, label=f"corner_exit_{index}_{sign:+.0f}"))
    braking = ((10.5, 0.06), (9.0, 0.08), (7.0, 0.12),
               (5.0, 0.16), (3.5, 0.20))
    for index, (speed, steering) in enumerate(braking):
        sign = -1.0 if rng.getrandbits(1) else 1.0
        blocks.append(CommandBlock(
            speed, sign * steering, label=f"brake_transition_{index}_{sign:+.0f}"))
    return tuple(blocks)


def build_race_domain_boundary_plan(seed: int) -> tuple[CommandBlock, ...]:
    """Add matched steering/braking trials just beyond existing clean support.

    At 9.5 and 10.5 m/s the largest tested steering grows by 0.01 rad from
    the existing clean profile; at 11.1 m/s it grows from 0.04 to 0.05 rad.
    The existing tilt, collision, governor, and hard-speed aborts remain active.
    """
    rng = random.Random(seed)
    blocks: list[CommandBlock] = []
    magnitudes = {
        9.5: (0.0, 0.04, 0.06, 0.08, 0.09),
        10.5: (0.0, 0.04, 0.06, 0.08, 0.09),
        11.1: (0.0, 0.02, 0.04, 0.05),
    }
    for speed in (9.5, 10.5, 11.1):
        steering_values = []
        for magnitude in magnitudes[speed]:
            if magnitude == 0.0:
                steering_values.append(0.0)
                continue
            signs = [-1.0, 1.0]
            rng.shuffle(signs)
            steering_values.extend(sign * magnitude for sign in signs)
        limit = RACE_DOMAIN_COMBINED_STEER_LIMITS[speed]
        if any(abs(value) > limit + 1.0e-9 for value in steering_values):
            raise AssertionError("boundary plan exceeds its tested steering step")
        blocks.extend(CommandBlock(
            speed, steering,
            label=f"boundary_grid_{speed:g}_{steering:+.3f}")
            for steering in steering_values)

    blocks.append(CommandBlock(
        3.0, 0.0, label="brake_test_reposition_low"))
    brake_groups = (
        (9.5, 7.0, (0.0, 0.09, -0.09)),
        (10.5, 8.0, (0.09, -0.09)),
        (11.1, 9.5, (0.0, 0.05, -0.05)),
    )
    trial_index = 0
    for start_speed, brake_speed, steering_values in brake_groups:
        steering_order = list(steering_values)
        rng.shuffle(steering_order)
        for steering in steering_order:
            sign = ("L" if steering > 0.0 else
                    "R" if steering < 0.0 else "straight")
            blocks.append(CommandBlock(
                start_speed, steering,
                label=f"brake_approach_{trial_index}_{start_speed:g}_{sign}"))
            blocks.append(CommandBlock(
                brake_speed, steering,
                label=f"brake_pulse_{trial_index}_{start_speed:g}_{sign}"))
            trial_index += 1
    blocks.append(CommandBlock(
        3.0, 0.0, label="corner_exit_reposition_low"))

    exit_signs = [-1.0, 1.0]
    rng.shuffle(exit_signs)
    for sign in exit_signs:
        for index, (speed, steering) in enumerate(
                ((5.0, 0.30), (6.5, 0.20), (8.0, 0.10), (9.5, 0.06))):
            blocks.append(CommandBlock(
                speed, sign * steering,
                label=f"corner_exit_{'L' if sign > 0 else 'R'}_{index}"))
    return tuple(blocks)


def race_domain_moderate_steering_limit(speed_mps: float) -> float:
    """Permit a guarded, modest extension above 9 m/s for boundary discovery."""
    if speed_mps < 0.0:
        raise ValueError("speed must be nonnegative")
    if speed_mps <= 9.0:
        return steering_limit_for_speed(speed_mps)
    if speed_mps <= 10.8:
        return RACE_DOMAIN_MODERATE_STEER_LIMITS[10.5]
    return RACE_DOMAIN_MODERATE_STEER_LIMITS[11.1]


def build_race_domain_moderate_braking_plan(
        seed: int) -> tuple[CommandBlock, ...]:
    """Probe 9–12 m/s turn-in, throttle reduction, braking and release.

    Every brake sequence starts from a low-speed reposition, approaches its
    target straight, then changes one factor at a time: steering turn-in,
    positive-throttle reduction, zero-throttle active braking, and steering
    release. The modest steering extension stays under the existing 8-degree
    tilt abort, 11.2 m/s governor and 11.9 m/s hard cutoff.
    """
    rng = random.Random(seed)
    dwell = 6.0
    blocks: list[CommandBlock] = [
        CommandBlock(3.0, 0.0, dwell, "initial_reposition_low")
    ]
    grid_steering = {
        9.5: (0.10, 0.12, 0.14),
        10.5: (0.10, 0.12, 0.14),
        11.1: (0.10, 0.12),
    }
    for speed, magnitudes in grid_steering.items():
        blocks.append(CommandBlock(
            speed, 0.0, dwell, f"boundary_approach_{speed:g}_straight"))
        signed = [sign * magnitude for magnitude in magnitudes
                  for sign in (-1.0, 1.0)]
        rng.shuffle(signed)
        if any(abs(value) > RACE_DOMAIN_MODERATE_STEER_LIMITS[speed] + 1e-9
               for value in signed):
            raise AssertionError("moderate plan exceeds its steering envelope")
        blocks.extend(CommandBlock(
            speed, steering, dwell,
            f"boundary_grid_{speed:g}_{steering:+.3f}")
            for steering in signed)

    trials = [
        (9.5, 7.0, steering)
        for steering in (-0.14, 0.14)
    ] + [
        (10.5, 8.0, steering)
        for steering in (-0.14, 0.14)
    ] + [
        (11.1, 9.5, steering)
        for steering in (-0.12, 0.12)
    ]
    rng.shuffle(trials)
    for index, (start_speed, reduced_speed, steering) in enumerate(trials):
        sign = "L" if steering > 0.0 else "R"
        blocks.extend((
            CommandBlock(3.0, 0.0, dwell,
                         f"transition_reposition_{index}_{sign}"),
            CommandBlock(start_speed, 0.0, dwell,
                         f"transition_approach_{index}_{start_speed:g}_{sign}"),
            CommandBlock(start_speed, steering, dwell,
                         f"turn_in_{index}_{start_speed:g}_{sign}"),
            CommandBlock(reduced_speed, steering, dwell,
                         f"throttle_reduce_{index}_{start_speed:g}_{sign}"),
            CommandBlock(3.0, steering, dwell,
                         f"brake_onset_{index}_{start_speed:g}_{sign}"),
            CommandBlock(3.0, 0.0, dwell,
                         f"brake_release_{index}_{start_speed:g}_{sign}"),
        ))

    exit_signs = [-1.0, 1.0]
    rng.shuffle(exit_signs)
    for sign in exit_signs:
        for index, (speed, steering) in enumerate(
                ((5.0, 0.30), (6.5, 0.20), (8.0, 0.10), (9.5, 0.06))):
            blocks.append(CommandBlock(
                speed, sign * steering, dwell,
                f"corner_exit_{'L' if sign > 0 else 'R'}_{index}"))
    return tuple(blocks)


def build_high_steer_validation_plan(
        seed: int,
        target_speed_mps: float = HIGH_STEER_VALIDATION_SPEED_MPS,
) -> tuple[CommandBlock, ...]:
    """Replicate the observed 7.7–7.9 m/s high-steer cells with long holds.

    Each capture performs one randomized 7.5 s sweep of the observed angles
    at the requested speed; independent whole-run captures provide
    replication. The default preserves the established 7.5 m/s profile.
    """
    rng = random.Random(seed)
    conditions = [CommandBlock(
        target_speed_mps, 0.0,
        HIGH_STEER_VALIDATION_DWELL_S, "baseline_zero_steer")]
    conditions.extend(
        CommandBlock(
            target_speed_mps, sign * angle,
            HIGH_STEER_VALIDATION_DWELL_S,
            f"steer_{sign:+.0f}_{angle:.4f}")
        for angle in HIGH_STEER_VALIDATION_ANGLES_RAD
        for sign in (-1.0, 1.0)
    )
    blocks: list[CommandBlock] = []
    for repetition in range(1, HIGH_STEER_VALIDATION_REPETITIONS + 1):
        ordered = conditions.copy()
        rng.shuffle(ordered)
        blocks.extend(CommandBlock(
            block.target_speed_mps, block.steering_rad, block.duration_s,
            f"r{repetition}_{block.label}") for block in ordered)
    return tuple(blocks)


def build_race_domain_steering_frontier_plan(
        seed: int) -> tuple[CommandBlock, ...]:
    """Probe matched-start high-speed steering response around the known edge.

    Earlier 9.5–11.1 m/s captures stopped at 0.12–0.14 rad and showed a
    reproducible yaw/lateral-acceleration roll-off. This plan samples the
    response more densely through 0.20 rad, in both turn directions. The
    caller settles to the same near-straight state before every block and
    keeps the simulator's tilt and hard-speed aborts active.
    """
    rng = random.Random(seed)
    blocks: list[CommandBlock] = []
    for speed in RACE_DOMAIN_STEERING_FRONTIER_SPEEDS_MPS:
        blocks.append(CommandBlock(
            speed, 0.0, RACE_DOMAIN_STEERING_FRONTIER_DWELL_S,
            f"frontier_baseline_{speed:g}"))
        signed = [sign * magnitude
                  for magnitude in RACE_DOMAIN_STEERING_FRONTIER_ANGLES_RAD[speed]
                  for sign in (-1.0, 1.0)]
        rng.shuffle(signed)
        blocks.extend(CommandBlock(
            speed, steering, RACE_DOMAIN_STEERING_FRONTIER_DWELL_S,
            f"frontier_{speed:g}_{steering:+.3f}")
            for steering in signed)
    return tuple(blocks)


def plan_duration_s(plan: tuple[CommandBlock, ...]) -> float:
    return sum(block.duration_s for block in plan)
