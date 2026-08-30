"""Measured-feedforward target-speed controller for AutoDRIVE."""

from dataclasses import dataclass
import math
from typing import Sequence

from .actuator_math import clamp


@dataclass(frozen=True)
class SpeedControllerConfig:
    kp: float = 0.015
    ki: float = 0.003
    ka: float = 0.0
    integral_limit: float = 1.0
    throttle_min_forward: float = 0.0
    throttle_max_forward: float = 0.10
    throttle_rise_rate_per_sec: float = 0.20
    throttle_fall_rate_per_sec: float = 0.50
    stop_speed_threshold_mps: float = 0.02
    feedforward_speed_mps: tuple[float, ...] = (
        0.0, 0.6, 1.0, 1.5, 1.8, 2.5)
    feedforward_throttle: tuple[float, ...] = (
        0.0, 0.025, 0.035, 0.055, 0.065, 0.085)


def _finite_sequence(values: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def validate_feedforward_table(
    speed_points: Sequence[float],
    throttle_points: Sequence[float],
) -> None:
    """Validate a piecewise-linear table or zero-only placeholder."""
    if len(speed_points) != len(throttle_points):
        raise ValueError('feedforward arrays must have equal length')
    if not speed_points:
        raise ValueError('feedforward table must not be empty')
    if not _finite_sequence(speed_points) or not _finite_sequence(throttle_points):
        raise ValueError('feedforward table values must be finite')
    if len(speed_points) == 1:
        if float(speed_points[0]) != 0.0 or float(throttle_points[0]) != 0.0:
            raise ValueError('single-point placeholder must be (0.0, 0.0)')
        return
    if any(
        float(speed_points[index]) >= float(speed_points[index + 1])
        for index in range(len(speed_points) - 1)
    ):
        raise ValueError('feedforward speeds must be strictly increasing')


def interpolate_feedforward(
    target_speed_mps: float,
    speed_points: Sequence[float],
    throttle_points: Sequence[float],
) -> float:
    """Interpolate within measured data; clamp beyond measured endpoints."""
    validate_feedforward_table(speed_points, throttle_points)
    if not math.isfinite(target_speed_mps):
        raise ValueError('target speed must be finite')
    if len(speed_points) == 1:
        return float(throttle_points[0])
    if target_speed_mps <= float(speed_points[0]):
        return float(throttle_points[0])
    if target_speed_mps >= float(speed_points[-1]):
        return float(throttle_points[-1])
    for index in range(len(speed_points) - 1):
        low_speed = float(speed_points[index])
        high_speed = float(speed_points[index + 1])
        if low_speed <= target_speed_mps <= high_speed:
            ratio = (target_speed_mps - low_speed) / (high_speed - low_speed)
            low_throttle = float(throttle_points[index])
            high_throttle = float(throttle_points[index + 1])
            return low_throttle + ratio * (high_throttle - low_throttle)
    raise RuntimeError('feedforward interval search failed')


def validate_speed_controller(config: SpeedControllerConfig) -> None:
    values = (
        config.kp,
        config.ki,
        config.ka,
        config.integral_limit,
        config.throttle_min_forward,
        config.throttle_max_forward,
        config.throttle_rise_rate_per_sec,
        config.throttle_fall_rate_per_sec,
        config.stop_speed_threshold_mps,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError('speed-controller parameters must be finite')
    if config.kp < 0.0 or config.ki < 0.0:
        raise ValueError('kp and ki must be >= 0')
    if config.integral_limit < 0.0:
        raise ValueError('integral_limit must be >= 0')
    if config.throttle_min_forward < 0.0:
        raise ValueError('throttle_min_forward must be >= 0')
    if config.throttle_min_forward >= config.throttle_max_forward:
        raise ValueError('forward throttle minimum must be below maximum')
    if (
        config.throttle_rise_rate_per_sec <= 0.0
        or config.throttle_fall_rate_per_sec <= 0.0
    ):
        raise ValueError('throttle slew rates must be > 0')
    if config.stop_speed_threshold_mps < 0.0:
        raise ValueError('stop_speed_threshold_mps must be >= 0')
    validate_feedforward_table(
        config.feedforward_speed_mps,
        config.feedforward_throttle,
    )
    if any(
        throttle < config.throttle_min_forward
        or throttle > config.throttle_max_forward
        for throttle in config.feedforward_throttle
    ):
        raise ValueError('feedforward throttle lies outside forward limits')


class TargetSpeedController:
    """Feedforward + acceleration feedforward + signed PI with slew limits."""

    def __init__(self, config: SpeedControllerConfig) -> None:
        validate_speed_controller(config)
        self._config = config
        self._integral = 0.0
        self._last_output = 0.0

    @property
    def config(self) -> SpeedControllerConfig:
        return self._config

    @property
    def integral(self) -> float:
        return self._integral

    @property
    def last_output(self) -> float:
        return self._last_output

    def reconfigure(
        self,
        config: SpeedControllerConfig,
        *,
        reset_integral: bool = True,
    ) -> None:
        validate_speed_controller(config)
        self._config = config
        if reset_integral:
            self.reset()
        else:
            self._integral = clamp(
                self._integral,
                -config.integral_limit,
                config.integral_limit,
            )
            self._last_output = clamp(
                self._last_output,
                config.throttle_min_forward,
                config.throttle_max_forward,
            )

    def reset(self) -> None:
        self._integral = 0.0
        self._last_output = 0.0

    def update(
        self,
        target_speed_mps: float,
        measured_speed_mps: float,
        requested_accel_mps2: float,
        dt_seconds: float,
    ) -> float:
        values = (
            target_speed_mps,
            measured_speed_mps,
            requested_accel_mps2,
            dt_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError('speed-controller inputs must be finite')
        if dt_seconds <= 0.0:
            raise ValueError('dt_seconds must be > 0')

        target = max(0.0, target_speed_mps)
        measured = max(0.0, measured_speed_mps)
        if target <= self._config.stop_speed_threshold_mps:
            self.reset()
            return 0.0

        error = target - measured
        candidate_integral = clamp(
            self._integral + error * dt_seconds,
            -self._config.integral_limit,
            self._config.integral_limit,
        )
        feedforward = interpolate_feedforward(
            target,
            self._config.feedforward_speed_mps,
            self._config.feedforward_throttle,
        )
        unsaturated = (
            feedforward
            + self._config.ka * requested_accel_mps2
            + self._config.kp * error
            + self._config.ki * candidate_integral
        )
        desired = clamp(
            unsaturated,
            self._config.throttle_min_forward,
            self._config.throttle_max_forward,
        )

        # Conditional integration: accept only when unsaturated, or when error
        # drives a saturated output back toward its usable interval.
        if (
            self._config.throttle_min_forward < unsaturated
            < self._config.throttle_max_forward
            or (
                unsaturated >= self._config.throttle_max_forward
                and error < 0.0
            )
            or (
                unsaturated <= self._config.throttle_min_forward
                and error > 0.0
            )
        ):
            self._integral = candidate_integral

        rate = (
            self._config.throttle_rise_rate_per_sec
            if desired >= self._last_output
            else self._config.throttle_fall_rate_per_sec
        )
        maximum_step = rate * dt_seconds
        output = clamp(
            desired,
            self._last_output - maximum_step,
            self._last_output + maximum_step,
        )
        output = clamp(
            output,
            self._config.throttle_min_forward,
            self._config.throttle_max_forward,
        )
        self._last_output = output
        return output
