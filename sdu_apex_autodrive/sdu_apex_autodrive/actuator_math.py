"""Pure AutoDRIVE actuator-boundary conversions."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ActuatorLimits:
    """Source-backed AutoDRIVE RoboRacer command limits."""

    max_steering_angle_rad: float = 0.5236
    steering_min: float = -1.0
    steering_max: float = 1.0
    throttle_min: float = -1.0
    throttle_max: float = 1.0
    max_target_speed_mps: float = 4.0


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a finite scalar to an inclusive range."""
    return min(max(value, low), high)


def validate_limits(limits: ActuatorLimits) -> None:
    values = (
        limits.max_steering_angle_rad,
        limits.steering_min,
        limits.steering_max,
        limits.throttle_min,
        limits.throttle_max,
        limits.max_target_speed_mps,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('actuator limits must be finite')
    if limits.max_steering_angle_rad <= 0.0:
        raise ValueError('max_steering_angle_rad must be > 0')
    if limits.steering_min >= limits.steering_max:
        raise ValueError('steering_min must be below steering_max')
    if not limits.steering_min <= 0.0 <= limits.steering_max:
        raise ValueError('steering range must contain zero')
    if limits.throttle_min >= limits.throttle_max:
        raise ValueError('throttle_min must be below throttle_max')
    if not limits.throttle_min <= 0.0 <= limits.throttle_max:
        raise ValueError('throttle range must contain zero')
    if limits.max_target_speed_mps <= 0.0:
        raise ValueError('max_target_speed_mps must be > 0')


def steering_angle_to_normalized(
    steering_angle_rad: float,
    limits: ActuatorLimits,
) -> float:
    """Convert centre-wheel steering radians to native normalized input."""
    validate_limits(limits)
    if not math.isfinite(steering_angle_rad):
        raise ValueError('steering angle must be finite')
    bounded = clamp(
        steering_angle_rad,
        -limits.max_steering_angle_rad,
        limits.max_steering_angle_rad,
    )
    return clamp(
        bounded / limits.max_steering_angle_rad,
        limits.steering_min,
        limits.steering_max,
    )


def bound_target_speed(
    target_speed_mps: float,
    limits: ActuatorLimits,
) -> float:
    """Bound Ackermann's forward target speed."""
    validate_limits(limits)
    if not math.isfinite(target_speed_mps):
        raise ValueError('target speed must be finite')
    return clamp(target_speed_mps, 0.0, limits.max_target_speed_mps)
