"""Float32 ports of the active Unity ``VehicleController`` actuator code.

These functions mirror ``Assets/Scripts/VehicleController.cs`` and the
serialized active F1TENTH parameters. They do not model the WheelCollider or
PhysX contact solver.
"""

from __future__ import annotations

import math

import numpy as np


STEERING_RATE_DEG_S = np.float32(183.346)
STEERING_LIMIT_DEG = np.float32(30.0)
WHEELBASE_MM = np.float32(324.0)
TRACK_WIDTH_MM = np.float32(236.0)


def step_autonomous_steering_deg(
    steering_angle_deg: float,
    last_steering_angle_deg: float,
    steering_command_norm: float,
    dt_s: float,
) -> float:
    """Apply one exact source-ordered autonomous ``Steer()`` update.

    Unity's fields and arithmetic here are single precision. The command is
    the normalized ``AutonomousSteering`` field, and returned angle is the
    internal physical ``SteeringAngle`` in degrees. The source's comparison
    is intentionally preserved even though its sign convention makes this
    unlike a conventional symmetric rate limiter.
    """
    angle = np.float32(steering_angle_deg)
    last_angle = np.float32(last_steering_angle_deg)
    command = np.float32(np.clip(np.float32(steering_command_norm), -1.0, 1.0))
    dt = np.float32(dt_s)
    if not (np.isfinite(angle) and np.isfinite(last_angle)
            and np.isfinite(command) and np.isfinite(dt)) or dt <= 0.0:
        raise ValueError("steering state, command, and positive dt must be finite")

    threshold = np.float32(last_angle / STEERING_LIMIT_DEG)
    increment = np.float32(STEERING_RATE_DEG_S * dt)
    target = np.float32(-command * STEERING_LIMIT_DEG)
    if command > threshold:
        angle = np.float32(angle - increment)
        if angle <= target:
            angle = target
    else:
        angle = np.float32(angle + increment)
        if angle >= target:
            angle = target
    return float(angle)


def front_ackermann_angles_deg(center_angle_deg: float) -> tuple[float, float]:
    """Unity ``VehicleController.Steer`` front-wheel Ackermann angles."""
    angle = np.float32(center_angle_deg)
    radians_per_degree = np.float32(np.float32(math.pi) / np.float32(180.0))
    angle_rad = np.float32(angle * radians_per_degree)
    tangent = np.float32(np.tan(angle_rad))
    numerator = np.float32(
        np.float32(2.0) * WHEELBASE_MM * tangent)
    left_denominator = np.float32(
        np.float32(2.0) * WHEELBASE_MM
        + TRACK_WIDTH_MM * tangent)
    right_denominator = np.float32(
        np.float32(2.0) * WHEELBASE_MM
        - TRACK_WIDTH_MM * tangent)
    left_rad = np.float32(np.arctan(np.float32(numerator / left_denominator)))
    right_rad = np.float32(np.arctan(np.float32(numerator / right_denominator)))
    degrees_per_radian = np.float32(np.float32(180.0) / np.float32(math.pi))
    return (float(np.float32(left_rad * degrees_per_radian)),
            float(np.float32(right_rad * degrees_per_radian)))


def center_angle_from_front_ackermann_deg(
    left_angle_deg: float, right_angle_deg: float
) -> float:
    """Invert the configured Ackermann relation for capture initialization."""
    left_tangent = math.tan(math.radians(left_angle_deg))
    right_tangent = math.tan(math.radians(right_angle_deg))
    twice_wheelbase = 2.0 * float(WHEELBASE_MM)
    left_center_tangent = (
        twice_wheelbase * left_tangent
        / (twice_wheelbase - float(TRACK_WIDTH_MM) * left_tangent))
    right_center_tangent = (
        twice_wheelbase * right_tangent
        / (twice_wheelbase + float(TRACK_WIDTH_MM) * right_tangent))
    return math.degrees(math.atan(
        0.5 * (left_center_tangent + right_center_tangent)))
