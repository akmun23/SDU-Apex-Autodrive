"""Source-backed AutoDRIVE actuator conversion and speed control."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class InterfaceConfig:
    """AutoDRIVE RoboRacer interface limits.

    The 2026 technical guide defines normalized actuator commands in [-1, 1]
    and a steering-angle range of [-0.5236, 0.5236] rad.
    """

    max_steering_angle_rad: float = 0.5236
    steering_command_min: float = -1.0
    steering_command_max: float = 1.0
    throttle_command_min: float = -1.0
    throttle_command_max: float = 1.0
    max_target_speed_mps: float = 2.5


@dataclass(frozen=True)
class SpeedControllerConfig:
    """Forward-only target-speed controller settings.

    These are controller tuning values, not actuator calibration constants.
    The output cap can deliberately be lower than AutoDRIVE's native limit.
    """

    feedforward_gain: float = 0.04
    proportional_gain: float = 0.02
    integral_gain: float = 0.005
    integral_limit: float = 1.0
    maximum_forward_throttle: float = 0.07
    throttle_rise_rate_per_sec: float = 0.10
    throttle_fall_rate_per_sec: float = 0.20
    stop_speed_threshold_mps: float = 0.01


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def validate_interface(config: InterfaceConfig) -> None:
    values = (
        config.max_steering_angle_rad,
        config.steering_command_min,
        config.steering_command_max,
        config.throttle_command_min,
        config.throttle_command_max,
        config.max_target_speed_mps,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('AutoDRIVE interface parameters must be finite')
    if config.max_steering_angle_rad <= 0.0:
        raise ValueError('max_steering_angle_rad must be > 0')
    if config.steering_command_min >= config.steering_command_max:
        raise ValueError('steering command minimum must be below maximum')
    if not config.steering_command_min <= 0.0 <= config.steering_command_max:
        raise ValueError('steering command range must include neutral zero')
    if config.throttle_command_min >= config.throttle_command_max:
        raise ValueError('throttle command minimum must be below maximum')
    if not config.throttle_command_min <= 0.0 <= config.throttle_command_max:
        raise ValueError('throttle command range must include neutral zero')
    if config.max_target_speed_mps <= 0.0:
        raise ValueError('max_target_speed_mps must be > 0')


def validate_speed_controller(
    config: SpeedControllerConfig,
    interface: InterfaceConfig,
) -> None:
    values = (
        config.feedforward_gain,
        config.proportional_gain,
        config.integral_gain,
        config.integral_limit,
        config.maximum_forward_throttle,
        config.throttle_rise_rate_per_sec,
        config.throttle_fall_rate_per_sec,
        config.stop_speed_threshold_mps,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('speed-controller parameters must be finite')
    if (
        config.feedforward_gain < 0.0
        or config.proportional_gain < 0.0
        or config.integral_gain < 0.0
    ):
        raise ValueError('speed-controller gains must be >= 0')
    if config.integral_limit < 0.0:
        raise ValueError('integral_limit must be >= 0')
    if config.stop_speed_threshold_mps < 0.0:
        raise ValueError('stop_speed_threshold_mps must be >= 0')
    if (
        config.throttle_rise_rate_per_sec <= 0.0
        or config.throttle_fall_rate_per_sec <= 0.0
    ):
        raise ValueError('throttle slew rates must be > 0')
    if not (
        0.0
        < config.maximum_forward_throttle
        <= interface.throttle_command_max
    ):
        raise ValueError(
            'maximum_forward_throttle must be positive and within the '
            'native limit')


def steering_angle_to_command(
    steering_angle_rad: float,
    config: InterfaceConfig,
) -> float:
    """Convert centre steering angle radians to native normalized input."""
    validate_interface(config)
    if not math.isfinite(steering_angle_rad):
        raise ValueError('steering angle must be finite')
    bounded_angle = _clamp(
        steering_angle_rad,
        -config.max_steering_angle_rad,
        config.max_steering_angle_rad,
    )
    normalized = bounded_angle / config.max_steering_angle_rad
    return _clamp(
        normalized,
        config.steering_command_min,
        config.steering_command_max,
    )


def bound_target_speed(
    target_speed_mps: float,
    config: InterfaceConfig,
) -> float:
    """Bound the forward target speed carried by AckermannDrive.speed."""
    validate_interface(config)
    if not math.isfinite(target_speed_mps):
        raise ValueError('target speed must be finite')
    return _clamp(target_speed_mps, 0.0, config.max_target_speed_mps)


class ForwardSpeedController:
    """PI target-speed controller producing native forward throttle input."""

    def __init__(
        self,
        config: SpeedControllerConfig,
        interface: InterfaceConfig,
    ) -> None:
        validate_interface(interface)
        validate_speed_controller(config, interface)
        self._config = config
        self._interface = interface
        self._integral = 0.0
        self._last_output = 0.0

    @property
    def integral(self) -> float:
        return self._integral

    def reset(self) -> None:
        self._integral = 0.0
        self._last_output = 0.0

    def update(
        self,
        target_speed_mps: float,
        current_speed_mps: float,
        dt_seconds: float,
    ) -> float:
        values = (target_speed_mps, current_speed_mps, dt_seconds)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('speed-controller inputs must be finite')
        if dt_seconds <= 0.0:
            raise ValueError('dt_seconds must be > 0')

        target = bound_target_speed(target_speed_mps, self._interface)
        current = max(0.0, current_speed_mps)
        if target <= self._config.stop_speed_threshold_mps:
            self.reset()
            return 0.0

        error = target - current
        candidate_integral = _clamp(
            self._integral + error * dt_seconds,
            -self._config.integral_limit,
            self._config.integral_limit,
        )
        feedforward = self._config.feedforward_gain * target
        proportional = self._config.proportional_gain * error
        candidate_output = (
            feedforward
            + proportional
            + self._config.integral_gain * candidate_integral
        )
        desired_output = _clamp(
            candidate_output,
            0.0,
            self._config.maximum_forward_throttle,
        )

        # Integrate in the unsaturated region, or when error would move a
        # saturated output back toward the usable range.
        if (
            0.0 < candidate_output < self._config.maximum_forward_throttle
            or (
                candidate_output >= self._config.maximum_forward_throttle
                and error < 0.0
            )
            or (candidate_output <= 0.0 and error > 0.0)
        ):
            self._integral = candidate_integral

        rate = (
            self._config.throttle_rise_rate_per_sec
            if desired_output >= self._last_output
            else self._config.throttle_fall_rate_per_sec
        )
        maximum_step = rate * dt_seconds
        output = _clamp(
            desired_output,
            self._last_output - maximum_step,
            self._last_output + maximum_step,
        )
        self._last_output = output
        return _clamp(
            output,
            max(0.0, self._interface.throttle_command_min),
            self._interface.throttle_command_max,
        )
