"""Replay the production speed controller and actuator command boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from sdu_apex_autodrive.sdu_apex_autodrive.speed_controller import (
    LongitudinalMode,
    LongitudinalStateEstimator,
    SpeedControllerConfig,
    TargetSpeedController,
)


@dataclass(frozen=True)
class ActuatorOutput:
    steering_normalized: float
    throttle_normalized: float
    mode: LongitudinalMode


class ProductionActuator:
    """Production Python speed controller with sensor-only odom/IMU inputs."""

    def __init__(self, config_yaml: Path,
                 parameter_overlay_yaml: Path | None = None) -> None:
        raw = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
        params = raw["autodrive_actuator_interface"]["ros__parameters"]
        if parameter_overlay_yaml is not None:
            overlay = yaml.safe_load(
                parameter_overlay_yaml.read_text(encoding="utf-8"))
            params.update(
                overlay["autodrive_actuator_interface"]["ros__parameters"])
        tuple_fields = {
            "feedforward_speed_mps", "feedforward_throttle",
            "max_acceleration_speed_mps", "max_acceleration_envelope_mps2",
            "acceleration_speed_mps", "acceleration_throttle_per_mps2",
        }
        names = (
            "kp", "ki", "ka", "integral_limit", "throttle_max_forward",
            "throttle_rise_rate_per_sec", "throttle_fall_rate_per_sec",
            "stop_speed_threshold_mps", "overspeed_coast_threshold_mps",
            "speed_hold_error_deadband_mps", "speed_hold_recovery_error_mps",
            "speed_hold_acceleration_deadband_mps2", "speed_boost_error_mps",
            "speed_hold_prediction_horizon_sec", "speed_hold_entry_margin_mps",
            "speed_downshift_stable_sec", "speed_downshift_band_mps",
            "speed_overspeed_confirmation_sec", "hard_overspeed_cutoff_mps",
            "feedforward_speed_mps", "feedforward_throttle",
            "speed_error_to_accel_gain", "speed_error_integral_to_accel_gain",
            "max_acceleration_mps2", "max_acceleration_speed_mps",
            "max_acceleration_envelope_mps2", "max_deceleration_mps2",
            "acceleration_feedback_gain", "acceleration_integral_gain",
            "acceleration_integral_limit",
            "acceleration_throttle_rise_rate_per_sec",
            "acceleration_throttle_fall_rate_per_sec",
            "acceleration_speed_mps", "acceleration_throttle_per_mps2",
        )
        values = {}
        for name in names:
            value = params[name]
            values[name] = tuple(float(x) for x in value) if name in tuple_fields else float(value)
        optional_regime_defaults = {
            "throttle_rise_regime_rate_per_sec": 10.0,
            "throttle_rise_regime_speed_min_mps": 0.0,
            "throttle_rise_regime_speed_max_mps": 0.0,
            "throttle_rise_regime_min_abs_steering_rad": math.pi,
            "throttle_rise_regime_max_abs_steering_rad": math.pi,
        }
        for name, default in optional_regime_defaults.items():
            values[name] = float(params.get(name, default))
        values["throttle_rise_event_enabled"] = bool(
            params.get("throttle_rise_event_enabled", False))
        optional_event_defaults = {
            "throttle_rise_event_rate_per_sec": 10.0,
            "throttle_rise_event_speed_min_mps": 0.0,
            "throttle_rise_event_speed_max_mps": 0.0,
            "throttle_rise_event_min_abs_steering_rad": 0.0,
            "throttle_rise_event_max_abs_steering_rad": 0.0,
            "throttle_rise_event_increment_min": 0.0,
            "throttle_rise_event_increment_max": 0.0,
        }
        for name, default in optional_event_defaults.items():
            values[name] = float(params.get(name, default))
        self.config = SpeedControllerConfig(**values)
        self.config.validate()
        self.speed_controller = TargetSpeedController(self.config)
        self.speed_estimator = LongitudinalStateEstimator(
            acceleration_filter_alpha=float(params["acceleration_filter_alpha"]),
            slip_threshold_mps=float(params["slip_threshold_mps"]),
            slip_ratio=float(params["slip_ratio"]),
            odom_correction_gain=float(params["odom_correction_gain"]),
            speed_measurement_filter_alpha=float(
                params["speed_measurement_filter_alpha"]),
        )
        self.max_steering_rad = float(params["max_steering_angle_rad"])
        self.max_target_speed_mps = float(params["max_target_speed_mps"])
        self.last_controller_odom_stamp_ns: int | None = None
        self.speed_mps: float | None = None
        self.raw_speed_mps: float | None = None
        self.acceleration_mps2 = 0.0

    def reset(self) -> None:
        self.speed_controller.reset()
        self.speed_estimator.reset()
        self.last_controller_odom_stamp_ns = None
        self.speed_mps = None
        self.raw_speed_mps = None
        self.acceleration_mps2 = 0.0

    def observe(self, stamp_ns: int, odom_speed_mps: float,
                imu_acceleration_x_mps2: float) -> float:
        """Apply the production odom/IMU estimator callback logic in source time."""
        stamp_s = int(stamp_ns) * 1.0e-9
        self.speed_estimator.update_acceleration(
            float(imu_acceleration_x_mps2), stamp_s)
        self.raw_speed_mps = max(0.0, float(odom_speed_mps))
        self.speed_mps = self.speed_estimator.update_odometry(
            self.raw_speed_mps, stamp_s)
        self.acceleration_mps2 = self.speed_estimator.acceleration_mps2
        return self.speed_mps

    def tick(self, steering_angle_rad: float, target_speed_mps: float,
             dt_s: float = 0.025, measurement_fresh: bool = True
             ) -> ActuatorOutput:
        if self.speed_mps is None or self.raw_speed_mps is None:
            raise RuntimeError("production actuator needs a valid odometry sample")
        steering_norm = min(
            max(float(steering_angle_rad), -self.max_steering_rad),
            self.max_steering_rad) / self.max_steering_rad
        target = min(max(float(target_speed_mps), 0.0),
                     self.max_target_speed_mps)
        if self.raw_speed_mps - target >= self.config.hard_overspeed_cutoff_mps:
            self.speed_controller.reset()
            throttle = 0.0
            mode = LongitudinalMode.BRAKE
        else:
            command = self.speed_controller.update_command(
                target, self.speed_mps, 0.0, dt_s,
                self.acceleration_mps2, measurement_fresh,
                steering_angle_rad)
            throttle = command.throttle_normalized
            mode = command.mode
        return ActuatorOutput(float(steering_norm), float(throttle), mode)
