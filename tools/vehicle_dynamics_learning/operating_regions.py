"""Canonical WP18/WP20 operating-region bins from the offline-plant handoff."""

from __future__ import annotations

import math

import numpy as np


SPEED_EDGES_MPS = (0.0, 3.0, 5.0, 7.0, 9.0, 10.5, 12.0)
STEERING_EDGES_RAD = (0.0, 0.10, 0.20, 0.30, 0.40, 0.524)
STEERING_RATE_BINS_RADPS = (("R0", 0.0, 0.5), ("R1", 0.5, 2.0),
                            ("R2", 2.0, 5.0), ("R3", 5.0, 10.0))
THROTTLE_EDGES = (-1.001, -0.05, 0.05, 0.20, 0.40, 0.60, 0.80, 1.001)
THROTTLE_SLEW_BINS_PER_S = (("T0", 0.0, 0.25), ("T1", 0.25, 2.0),
                            ("T2", 2.0, 8.0), ("T3", 8.0, 22.0))
MISMATCH_EDGES_MPS = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, math.inf)


def region_masks(frames: np.ndarray, reset_index: np.ndarray,
                 packet_sequence: np.ndarray, valid: np.ndarray,
                 dt_s: float = 0.025) -> dict[str, np.ndarray]:
    """Return canonical pointwise operating masks for a 9-channel row array.

    `frames` uses the established layout `[u_rear,v_rear,r,steer_fb,
    throttle_fb,wheel_l,wheel_r,steer_cmd,throttle_cmd]`. Derived rates are
    causal backward differences. Transition-derived labels are disabled at
    resets and packet gaps.
    """
    values = np.asarray(frames, dtype=np.float64)
    reset = np.asarray(reset_index, dtype=np.int64)
    packet = np.asarray(packet_sequence, dtype=np.int64)
    base_valid = np.asarray(valid, dtype=bool)
    if (values.ndim != 2 or values.shape[1] != 9
            or reset.shape != (len(values),) or packet.shape != reset.shape
            or base_valid.shape != reset.shape
            or not np.isfinite(dt_s) or dt_s <= 0.0):
        raise ValueError("operating-region inputs must be aligned nine-channel rows")

    u, v = values[:, 0], values[:, 1]
    speed = np.hypot(u, v)
    steering = values[:, 3]
    throttle = values[:, 8]
    steering_rate = np.zeros(len(values), dtype=np.float64)
    throttle_slew = np.zeros(len(values), dtype=np.float64)
    contiguous = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        contiguous[1:] = ((reset[1:] == reset[:-1])
                          & (packet[1:] == packet[:-1] + 1))
        steering_rate[1:] = np.diff(steering) / dt_s
        throttle_slew[1:] = np.diff(throttle) / dt_s

    masks: dict[str, np.ndarray] = {}
    for index, (low, high) in enumerate(zip(SPEED_EDGES_MPS[:-1],
                                            SPEED_EDGES_MPS[1:])):
        masks[f"S{index}"] = ((speed >= low)
            & (speed < high if index < len(SPEED_EDGES_MPS) - 2 else speed <= high))
    absolute_steering = np.abs(steering)
    for index, (low, high) in enumerate(zip(STEERING_EDGES_RAD[:-1],
                                            STEERING_EDGES_RAD[1:])):
        masks[f"D{index}"] = ((absolute_steering >= low)
            & (absolute_steering < high if index < len(STEERING_EDGES_RAD) - 2
               else absolute_steering <= high))
    absolute_steering_rate = np.abs(steering_rate)
    for name, low, high in STEERING_RATE_BINS_RADPS:
        masks[name] = ((absolute_steering_rate >= low)
            & (absolute_steering_rate <= high if name == "R3"
               else absolute_steering_rate < high))

    for index, (low, high) in enumerate(zip(THROTTLE_EDGES[:-1],
                                            THROTTLE_EDGES[1:])):
        masks[f"throttle_{index}"] = (throttle >= low) & (throttle < high)
    masks.update({
        "negative_command_braking": throttle < -0.05,
        "near_zero_throttle_command": np.abs(throttle) <= 0.05,
        "low_positive_throttle_command": (throttle > 0.05) & (throttle < 0.20),
        "ordinary_racing_throttle_command": (throttle >= 0.20) & (throttle < 0.60),
        "high_throttle_command": (throttle >= 0.60) & (throttle <= 1.0),
    })

    absolute_throttle_slew = np.abs(throttle_slew)
    for name, low, high in THROTTLE_SLEW_BINS_PER_S:
        masks[name] = ((absolute_throttle_slew >= low)
            & (absolute_throttle_slew <= high if name == "T3"
               else absolute_throttle_slew < high))
    mismatch = np.abs(np.mean(values[:, 5:7], axis=1) - u)
    for index, (low, high) in enumerate(zip(MISMATCH_EDGES_MPS[:-1],
                                            MISMATCH_EDGES_MPS[1:])):
        masks[f"M{index}"] = ((mismatch >= low)
            & (mismatch < high if math.isfinite(high) else mismatch >= low))

    previous_absolute_steering = np.abs(np.r_[steering[0], steering[:-1]]) \
        if len(values) else np.empty(0, dtype=np.float64)
    masks.update({
        "high_speed_near_straight": (speed >= 9.0) & (absolute_steering < 0.10),
        "high_speed_moderate_steering": (speed >= 9.0)
            & (absolute_steering >= 0.10) & (absolute_steering < 0.30),
        "7_to_9mps_high_steering": (speed >= 7.0) & (speed < 9.0)
            & (absolute_steering >= 0.30),
        "low_speed_high_steering": (speed < 3.0) & (absolute_steering >= 0.30),
        "braking_release": (throttle < -0.05) & (throttle_slew >= 0.25),
        "throttle_pickup": (throttle > 0.05) & (throttle_slew >= 0.25),
        "steering_turn_in": (absolute_steering >= 0.10)
            & (absolute_steering_rate >= 0.5)
            & (absolute_steering - previous_absolute_steering > 0.0),
        "steering_unwind": (absolute_steering >= 0.10)
            & (absolute_steering_rate >= 0.5)
            & (absolute_steering - previous_absolute_steering < 0.0),
        "simultaneous_steering_throttle_transition":
            (absolute_steering_rate >= 0.5) & (absolute_throttle_slew >= 0.25),
        "large_wheel_body_mismatch": mismatch >= 2.0,
        "ordinary_practice_track_region": np.ones(len(values), dtype=bool),
    })
    for name, mask in masks.items():
        mask &= base_valid & np.isfinite(values).all(axis=1)
        if name not in {"S0", "D0", "R0", "T0", "M0"}:
            mask &= contiguous | (np.arange(len(values)) == 0)
    return masks


def steering_sign(steering_feedback_rad: np.ndarray) -> np.ndarray:
    """Return -1/0/+1 separately from absolute steering-bin membership."""
    values = np.asarray(steering_feedback_rad, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("steering feedback must be finite")
    return np.sign(values).astype(np.int8)
