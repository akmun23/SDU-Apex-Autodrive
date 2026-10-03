"""Seeded dynamic-identification conditions within the measured envelope."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random


DYNAMIC_STEERING_FREQUENCIES_HZ = (0.3, 0.6, 1.0, 1.5)
FRONTIER_SWEEP_ANGLES_RAD = (0.04, 0.08, 0.12, 0.16, 0.20)
FRONTIER_11MPS_ANGLES_RAD = (0.04, 0.08, 0.12, 0.16, 0.18)
PRBS_MIN_DWELL_S = 0.50


@dataclass(frozen=True)
class DynamicCoupledCondition:
    condition_id: str
    speed_band: str
    target_speed_mps: float
    max_steering_rad: float
    first_turn_sign: int
    triangle_frequency_hz: float
    multisine_phase_rad: tuple[float, ...]
    prbs_levels_normalized: tuple[float, ...]


# Steering bounds are deliberately the largest directly demonstrated values
# at or immediately below each speed. The 7.5 m/s full-lock condition is the
# existing held-out high-steer validation; 9.5/10.5 and 11.1 m/s limits come
# from the reset-isolated measured frontier.
_CONDITION_BASE = (
    ("5-7", 5.0, 0.30),
    ("5-7", 6.5, 0.20),
    ("7-9", 7.0, 0.20),
    ("7-9", 7.5, 0.5236),
    ("7-9", 8.0, 0.10),
    ("7-9", 8.5, 0.10),
    ("9-11.2", 9.5, 0.20),
    ("9-11.2", 10.5, 0.20),
    ("9-11.2", 11.1, 0.18),
)


def _stratified_levels(rng: random.Random, count: int = 12) -> tuple[float, ...]:
    """Return a seeded Latin-hypercube draw over normalized steering [-1, 1]."""
    levels = [(-1.0 + 2.0 * (index + rng.random()) / count)
              for index in range(count)]
    rng.shuffle(levels)
    return tuple(levels)


def build_dynamic_coupled_plan(seed: int) -> tuple[DynamicCoupledCondition, ...]:
    """Return the 5–11.2 m/s dynamic plan in seeded randomized order."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("dynamic-coupled seed must be an integer")
    rng = random.Random(seed)
    rows = list(_CONDITION_BASE)
    rng.shuffle(rows)
    result = []
    for band, speed, steering_limit in rows:
        speed_tag = f"{speed:.1f}".replace(".", "p")
        result.append(DynamicCoupledCondition(
            condition_id=f"band_{band.replace('.', 'p')}_v{speed_tag}",
            speed_band=band,
            target_speed_mps=speed,
            max_steering_rad=steering_limit,
            first_turn_sign=1 if rng.getrandbits(1) else -1,
            triangle_frequency_hz=rng.choice(DYNAMIC_STEERING_FREQUENCIES_HZ),
            multisine_phase_rad=tuple(
                rng.uniform(0.0, math.tau)
                for _ in DYNAMIC_STEERING_FREQUENCIES_HZ),
            prbs_levels_normalized=_stratified_levels(rng),
        ))
    return tuple(result)


def plan_as_dicts(seed: int) -> list[dict[str, object]]:
    """JSON-ready plan carrying every random waveform parameter."""
    return [asdict(condition) for condition in build_dynamic_coupled_plan(seed)]
