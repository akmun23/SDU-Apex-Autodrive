"""Numerically equivalent Python reference for the deterministic observer."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


@dataclass
class Estimate:
    stamp_s: float = 0.0
    dt_s: float = 0.0
    speed_pred_mps: float = 0.0
    speed_mps: float = 0.0
    body_u_mps: float = 0.0
    body_v_mps: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    wheel_raw_mps: float = 0.0
    wheel_mapped_mps: float = 0.0
    wheel_packet_mps: float = 0.0
    turn_speed_bias_mps: float = 0.0
    wheel_update_used: bool = False
    wheel_burst_rejected: bool = False
    turn_mode: bool = False
    reset_epoch: bool = False
    timing_degraded: bool = False
    sensor_outlier: bool = False


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ReferenceObserver:
    def __init__(self, normal_packet_dt_max_s: float = 0.035,
                 degraded_packet_dt_max_s: float = 0.050,
                 wheel_radius_m: float = 0.059,
                 reset_encoder_jump_rad: float = 50.0,
                 wheel_speed_window_s: float = 0.10,
                 max_integratable_gap_s: float = 0.250,
                 integrate_lateral_acceleration_in_turn: bool = False,
                 wheel_speed_scale: float = 0.968,
                 wheel_burst_disagreement_mps: float = 1.0,
                 wheel_speed_slew_limit_mps2: float = 40.0,
                 wheel_speed_scale_speeds_mps: tuple[float, ...] = (
                     0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.0,
                     10.0, 12.0, 14.0),
                 wheel_speed_scale_values: tuple[float, ...] = (
                     1.0, 0.998, 0.994, 0.992, 0.990, 0.9815,
                     0.9735, 0.9658, 0.9587, 0.9520, 0.9445),
                 allow_turn_current_packet_recovery: bool = True,
                 turn_current_packet_max_increase_mps: float = 0.20,
                 use_turn_speed_bias_model: bool = True,
                 turn_speed_bias_constant_mps: float = -0.03,
                 turn_speed_bias_speed_mps: float = 0.0,
                 turn_speed_bias_speed_squared_mps: float = 0.0,
                 turn_speed_bias_yaw_rate_abs_mps: float = 0.0,
                 turn_speed_bias_yaw_rate_squared_mps: float = 0.0,
                 turn_speed_bias_speed_yaw_rate_abs_mps: float = 0.0,
                 turn_speed_bias_max_mps: float = 0.03,
                 use_coherent_packet_velocity_for_pose: bool = True,
                 coherent_packet_pose_blend: float = 1.0,
                 use_kinematic_lateral_slip_model: bool = True,
                 lateral_slip_ratio: float = 0.012,
                 lateral_slip_yaw_rate_scale_radps: float = 0.15,
                 lateral_slip_max_mps: float = 0.30,
                 decel_detect_ax_mps2: float = -0.5,
                 decel_ax_scale: float = 1.005,
                 decel_ax_offset_mps2: float = 0.020,
                 imu_x_offset_m: float = 0.08,
                 wheel_update_ax_abs_max_mps2: float = 6.5,
                 wheel_freeze_speed_mps: float = 0.15,
                 wheel_innovation_max_mps: float = 1.50,
                 wheel_recovery_launch_speed_mps: float = 2.0,
                 wheel_recovery_launch_innovation_mps: float = 2.0,
                 wheel_recovery_launch_wheel_speed_mps: float = 4.0,
                 stationary_speed_threshold_mps: float = 0.03,
                 wheel_update_beta: float = 0.85,
                 stationary_hold_s: float = 0.10,
                 stationary_ax_abs_max_mps2: float = 0.25,
                 stationary_ay_abs_max_mps2: float = 0.75,
                 stationary_yaw_rate_abs_radps: float = 0.15,
                 turn_enter_yaw_rate_radps: float = 0.6,
                 turn_enter_abs_ay_mps2: float = 6.0,
                 turn_exit_yaw_rate_radps: float = 0.1,
                 turn_exit_abs_ay_mps2: float = 0.5,
                 turn_exit_hold_s: float = 0.5,
                 turn_wheel_braking_ax_mps2: float = -1.0,
                 max_imu_ax_abs_mps2: float = 30.0) -> None:
        self.normal_packet_dt_max_s = float(normal_packet_dt_max_s)
        self.degraded_packet_dt_max_s = float(degraded_packet_dt_max_s)
        self.wheel_radius_m = max(0.0, float(wheel_radius_m))
        self.reset_encoder_jump_rad = max(0.0, float(reset_encoder_jump_rad))
        self.wheel_speed_window_s = max(0.0, float(wheel_speed_window_s))
        self.max_integratable_gap_s = max(0.0, float(max_integratable_gap_s))
        self.wheel_speed_scale = max(0.0, float(wheel_speed_scale))
        self.wheel_speed_scale_speeds_mps = tuple(
            float(value) for value in wheel_speed_scale_speeds_mps)
        self.wheel_speed_scale_values = tuple(
            float(value) for value in wheel_speed_scale_values)
        self.wheel_burst_disagreement_mps = float(wheel_burst_disagreement_mps)
        self.wheel_speed_slew_limit_mps2 = max(0.0, float(wheel_speed_slew_limit_mps2))
        self.allow_turn_current_packet_recovery = bool(
            allow_turn_current_packet_recovery)
        self.turn_current_packet_max_increase_mps = max(
            0.0, float(turn_current_packet_max_increase_mps))
        self.use_turn_speed_bias_model = bool(use_turn_speed_bias_model)
        self.turn_speed_bias_constant_mps = float(turn_speed_bias_constant_mps)
        self.turn_speed_bias_speed_mps = float(turn_speed_bias_speed_mps)
        self.turn_speed_bias_speed_squared_mps = float(turn_speed_bias_speed_squared_mps)
        self.turn_speed_bias_yaw_rate_abs_mps = float(turn_speed_bias_yaw_rate_abs_mps)
        self.turn_speed_bias_yaw_rate_squared_mps = float(
            turn_speed_bias_yaw_rate_squared_mps)
        self.turn_speed_bias_speed_yaw_rate_abs_mps = float(
            turn_speed_bias_speed_yaw_rate_abs_mps)
        self.turn_speed_bias_max_mps = max(0.0, float(turn_speed_bias_max_mps))
        self.use_coherent_packet_velocity_for_pose = bool(
            use_coherent_packet_velocity_for_pose)
        self.coherent_packet_pose_blend = float(np.clip(
            coherent_packet_pose_blend, 0.0, 1.0))
        self.use_kinematic_lateral_slip_model = bool(
            use_kinematic_lateral_slip_model)
        self.lateral_slip_ratio = max(0.0, float(lateral_slip_ratio))
        self.lateral_slip_yaw_rate_scale_radps = max(
            0.0, float(lateral_slip_yaw_rate_scale_radps))
        self.lateral_slip_max_mps = max(0.0, float(lateral_slip_max_mps))
        self.decel_detect_ax_mps2 = float(decel_detect_ax_mps2)
        self.decel_ax_scale = float(decel_ax_scale)
        self.decel_ax_offset_mps2 = float(decel_ax_offset_mps2)
        self.imu_x_offset_m = max(0.0, float(imu_x_offset_m))
        # Keep the offline replay numerically aligned with the deployed
        # observer.  A stale replay gate can make a valid runtime change look
        # ineffective during offline validation.
        self.wheel_update_ax_abs_max_mps2 = max(
            0.0, float(wheel_update_ax_abs_max_mps2))
        self.wheel_freeze_speed_mps = max(0.0, float(wheel_freeze_speed_mps))
        self.wheel_innovation_max_mps = max(0.0, float(wheel_innovation_max_mps))
        self.wheel_recovery_launch_speed_mps = max(
            0.0, float(wheel_recovery_launch_speed_mps))
        self.wheel_recovery_launch_innovation_mps = max(
            0.0, float(wheel_recovery_launch_innovation_mps))
        self.wheel_recovery_launch_wheel_speed_mps = max(
            0.0, float(wheel_recovery_launch_wheel_speed_mps))
        self.stationary_speed_threshold_mps = max(
            0.0, float(stationary_speed_threshold_mps))
        self.wheel_update_beta = float(np.clip(wheel_update_beta, 0.0, 1.0))
        self.stationary_hold_s = max(0.0, float(stationary_hold_s))
        self.stationary_ax_abs_max_mps2 = max(0.0, float(stationary_ax_abs_max_mps2))
        self.stationary_ay_abs_max_mps2 = max(0.0, float(stationary_ay_abs_max_mps2))
        self.stationary_yaw_rate_abs_radps = max(
            0.0, float(stationary_yaw_rate_abs_radps))
        self.turn_enter_yaw_rate_radps = max(0.0, float(turn_enter_yaw_rate_radps))
        self.turn_enter_abs_ay_mps2 = max(0.0, float(turn_enter_abs_ay_mps2))
        self.turn_exit_yaw_rate_radps = max(0.0, float(turn_exit_yaw_rate_radps))
        self.turn_exit_abs_ay_mps2 = max(0.0, float(turn_exit_abs_ay_mps2))
        self.turn_exit_hold_s = max(0.0, float(turn_exit_hold_s))
        self.turn_wheel_braking_ax_mps2 = float(turn_wheel_braking_ax_mps2)
        self.max_imu_ax_abs_mps2 = max(0.0, float(max_imu_ax_abs_mps2))
        self.integrate_lateral_acceleration_in_turn = bool(
            integrate_lateral_acceleration_in_turn)
        self.reset()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ReferenceObserver":
        """Construct the offline mirror from the deployed observer YAML."""
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        parameters = config.get("sensor_odometry", {}).get("ros__parameters", {})
        required = {
            "normal_packet_dt_max_s", "degraded_packet_dt_max_s",
            "wheel_radius_m", "reset_encoder_jump_rad", "wheel_speed_window_s",
            "max_integratable_gap_s", "wheel_speed_scale", "wheel_burst_disagreement_mps",
            "wheel_speed_slew_limit_mps2", "allow_turn_current_packet_recovery",
            "turn_current_packet_max_increase_mps", "use_turn_speed_bias_model",
            "turn_speed_bias_constant_mps", "turn_speed_bias_speed_mps",
            "turn_speed_bias_speed_squared_mps", "turn_speed_bias_yaw_rate_abs_mps",
            "turn_speed_bias_yaw_rate_squared_mps", "turn_speed_bias_speed_yaw_rate_abs_mps",
            "turn_speed_bias_max_mps", "use_coherent_packet_velocity_for_pose",
            "coherent_packet_pose_blend", "integrate_lateral_acceleration_in_turn",
            "use_kinematic_lateral_slip_model", "lateral_slip_ratio",
            "lateral_slip_yaw_rate_scale_radps", "lateral_slip_max_mps",
            "decel_detect_ax_mps2", "decel_ax_scale", "decel_ax_offset_mps2",
            "imu_x_m", "wheel_update_ax_abs_max_mps2", "wheel_freeze_speed_mps",
            "wheel_innovation_max_mps", "wheel_recovery_launch_speed_mps",
            "wheel_recovery_launch_innovation_mps", "wheel_recovery_launch_wheel_speed_mps",
            "stationary_speed_threshold_mps", "wheel_update_beta", "stationary_hold_s",
            "stationary_ax_abs_max_mps2", "stationary_ay_abs_max_mps2",
            "stationary_yaw_rate_abs_max_radps", "turn_enter_yaw_rate_radps",
            "turn_enter_abs_ay_mps2", "turn_exit_yaw_rate_radps", "turn_exit_abs_ay_mps2",
            "turn_exit_hold_s", "turn_wheel_braking_ax_mps2", "max_imu_ax_abs_mps2",
        }
        missing = sorted(required.difference(parameters))
        if missing:
            raise ValueError(f"{path} is missing observer parameters: {missing}")
        values = dict(parameters)
        values["wheel_speed_scale_speeds_mps"] = tuple(
            values.get("wheel_speed_scale_speeds_mps", ()))
        values["wheel_speed_scale_values"] = tuple(
            values.get("wheel_speed_scale_values", ()))
        values["stationary_yaw_rate_abs_radps"] = values.pop(
            "stationary_yaw_rate_abs_max_radps")
        values["imu_x_offset_m"] = values.pop("imu_x_m")
        constructor_keys = (required - {
            "stationary_yaw_rate_abs_max_radps", "imu_x_m"}) | {
            "stationary_yaw_rate_abs_radps", "imu_x_offset_m",
            "wheel_speed_scale_speeds_mps", "wheel_speed_scale_values",
        }
        return cls(**{key: values[key] for key in constructor_keys})

    def reset(self) -> None:
        self.initialized = False
        self.turn = False
        self.calm = 0.0
        self.previous_stamp = 0.0
        self.previous_left = 0.0
        self.previous_right = 0.0
        self.previous_yaw = 0.0
        self.previous_yaw_rate = 0.0
        self.speed = 0.0
        self.u = 0.0
        self.v = 0.0
        self.previous_pose_u = 0.0
        self.previous_pose_v = 0.0
        self.x = 0.0
        self.y = 0.0
        self.last_pred = 0.0
        self.last_raw = 0.0
        self.last_mapped = 0.0
        self.last_packet = 0.0
        self.last_turn_speed_bias = 0.0
        self.stationary_time = 0.0
        self.encoder_history = deque()
        self.wheel_dropout_active = False
        self.wheel_burst_rejected = False
        self.wheel_burst_recovery_pending = False

    def wheel_scale_for_speed(self, raw_speed_mps: float) -> float:
        speeds = self.wheel_speed_scale_speeds_mps
        values = self.wheel_speed_scale_values
        if len(speeds) < 2 or len(speeds) != len(values) or not np.isfinite(raw_speed_mps):
            return self.wheel_speed_scale
        if any(not np.isfinite(speed) or not np.isfinite(value) or value < 0.0
               for speed, value in zip(speeds, values)):
            return self.wheel_speed_scale
        if any(upper <= lower for lower, upper in zip(speeds, speeds[1:])):
            return self.wheel_speed_scale
        speed = max(0.0, float(raw_speed_mps))
        if speed <= speeds[0]:
            return values[0]
        if speed >= speeds[-1]:
            return values[-1]
        upper = int(np.searchsorted(speeds, speed, side="right"))
        fraction = (speed - speeds[upper - 1]) / (speeds[upper] - speeds[upper - 1])
        return values[upper - 1] + fraction * (values[upper] - values[upper - 1])

    def map_wheel(self, value: float) -> float:
        if not np.isfinite(value) or value <= 0.0:
            return 0.0
        return float(value) * self.wheel_scale_for_speed(float(value))

    def turn_speed_bias(self, wheel_mapped: float, yaw_rate: float) -> float:
        if (not self.use_turn_speed_bias_model or
                not np.isfinite(wheel_mapped) or not np.isfinite(yaw_rate) or
                self.turn_speed_bias_max_mps <= 0.0):
            return 0.0
        speed = max(0.0, float(wheel_mapped))
        yaw_abs = abs(float(yaw_rate))
        bias = (
            self.turn_speed_bias_constant_mps +
            self.turn_speed_bias_speed_mps * speed +
            self.turn_speed_bias_speed_squared_mps * speed * speed +
            self.turn_speed_bias_yaw_rate_abs_mps * yaw_abs +
            self.turn_speed_bias_yaw_rate_squared_mps * yaw_abs * yaw_abs +
            self.turn_speed_bias_speed_yaw_rate_abs_mps * speed * yaw_abs)
        return float(np.clip(
            bias, -self.turn_speed_bias_max_mps, self.turn_speed_bias_max_mps))

    def kinematic_lateral_velocity(
            self, yaw_rate_radps: float, longitudinal_speed_mps: float) -> float:
        if (not self.use_kinematic_lateral_slip_model or
                not np.isfinite(yaw_rate_radps) or
                not np.isfinite(longitudinal_speed_mps) or
                self.lateral_slip_ratio <= 0.0 or
                self.lateral_slip_max_mps <= 0.0):
            return 0.0
        transition = max(1.0e-3, self.lateral_slip_yaw_rate_scale_radps)
        direction = math.tanh(yaw_rate_radps / transition)
        return float(np.clip(
            -self.lateral_slip_ratio * direction * max(0.0, longitudinal_speed_mps),
            -self.lateral_slip_max_mps, self.lateral_slip_max_mps))

    def _result(self, row: pd.Series, dt: float = 0.0) -> Estimate:
        return Estimate(
            stamp_s=float(row.stamp_s), dt_s=dt,
            speed_pred_mps=self.last_pred, speed_mps=self.speed,
            body_u_mps=self.u, body_v_mps=self.v, x_m=self.x, y_m=self.y,
            wheel_raw_mps=self.last_raw, wheel_mapped_mps=self.last_mapped,
            wheel_packet_mps=self.last_packet,
            turn_speed_bias_mps=self.last_turn_speed_bias,
            wheel_burst_rejected=self.wheel_burst_rejected,
            turn_mode=self.turn, sensor_outlier=False)

    def update(self, row: pd.Series) -> Estimate:
        values = row[[
            "stamp_s", "left_angle_rad", "right_angle_rad", "ax_mps2",
            "ay_mps2", "yaw_rate_radps", "yaw_rad"]].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            result = self._result(row)
            result.timing_degraded = True
            return result
        stamp, left, right, ax, ay, yaw_rate, yaw = values
        if not self.initialized:
            self.initialized = True
            self.previous_stamp = stamp
            self.previous_left = left
            self.previous_right = right
            self.previous_yaw = yaw
            self.previous_yaw_rate = yaw_rate
            self.encoder_history.clear()
            self.encoder_history.append((stamp, left, right))
            return self._result(row)

        dt = stamp - self.previous_stamp
        if dt <= 0.0:
            result = self._result(row, dt)
            result.timing_degraded = True
            return result
        dl = left - self.previous_left
        dr = right - self.previous_right
        if (abs(dl) > self.reset_encoder_jump_rad or
                abs(dr) > self.reset_encoder_jump_rad):
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.turn = False
            self.calm = 0.0
            self.speed = self.u = self.v = 0.0
            self.previous_pose_u = self.previous_pose_v = 0.0
            self.last_pred = self.last_raw = self.last_mapped = 0.0
            self.last_packet = 0.0
            self.last_turn_speed_bias = 0.0
            self.wheel_dropout_active = False
            self.wheel_burst_recovery_pending = False
            result = self._result(row, dt)
            result.reset_epoch = True
            return result
        # Match the deployed observer: a short source gap still has valid
        # synchronized encoder endpoints and is integrated. Only a gap longer
        # than the configured integratable horizon is re-baselined.
        if dt > self.max_integratable_gap_s:
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.last_pred = self.speed
            self.last_raw = self.last_mapped = 0.0
            self.last_packet = 0.0
            self.wheel_dropout_active = False
            self.wheel_burst_recovery_pending = False
            self.previous_pose_u = self.u
            self.previous_pose_v = self.v
            self.encoder_history.clear()
            self.encoder_history.append((stamp, left, right))
            result = self._result(row, dt)
            result.timing_degraded = True
            return result

        self.encoder_history.append((stamp, left, right))
        if self.wheel_speed_window_s > 0.0:
            while (len(self.encoder_history) > 1 and
                   stamp - self.encoder_history[1][0] >= self.wheel_speed_window_s):
                self.encoder_history.popleft()
            wheel_stamp, wheel_left, wheel_right = self.encoder_history[0]
        else:
            wheel_stamp, wheel_left, wheel_right = (
                self.previous_stamp, self.previous_left, self.previous_right)
        wheel_dt = stamp - wheel_stamp
        raw = (abs(self.wheel_radius_m * 0.5 * ((left - wheel_left) + (right - wheel_right)) /
                   wheel_dt) if wheel_dt > 0.0 else 0.0)
        packet = abs(self.wheel_radius_m * 0.5 * (dl + dr) / dt)
        mapped = self.map_wheel(raw)
        packet_mapped = packet * self.wheel_scale_for_speed(packet)
        wheel_slew_rejected = (
            self.wheel_speed_slew_limit_mps2 > 0.0 and
            self.speed > max(2.0, self.wheel_recovery_launch_speed_mps) and
            self.last_mapped > 0.0 and
            abs(mapped - self.last_mapped) / dt > self.wheel_speed_slew_limit_mps2 and
            abs(mapped - self.speed) > self.wheel_innovation_max_mps)
        self.last_raw, self.last_mapped, self.last_packet = raw, mapped, packet
        self.last_turn_speed_bias = 0.0
        self.wheel_burst_rejected = wheel_slew_rejected or (
            self.wheel_burst_disagreement_mps > 0.0 and
            mapped > self.speed + self.wheel_burst_disagreement_mps and
            packet_mapped > mapped + self.wheel_burst_disagreement_mps)
        near_zero_packet_limit = max(0.5, 0.25 * self.speed)
        if self.speed > 0.5 and packet_mapped < near_zero_packet_limit:
            self.wheel_dropout_active = True
        if packet < self.wheel_freeze_speed_mps and self.speed > 0.5:
            self.wheel_dropout_active = True
        if self.wheel_burst_rejected:
            self.wheel_dropout_active = True
            self.wheel_burst_recovery_pending = True
        if abs(ax) > self.max_imu_ax_abs_mps2:
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.last_pred = self.speed
            result = self._result(row, dt)
            result.timing_degraded = True
            result.sensor_outlier = True
            return result
        calm_stationary_sample = (
            raw < self.stationary_speed_threshold_mps and
            abs(ax) <= self.stationary_ax_abs_max_mps2 and
            abs(ay) <= self.stationary_ay_abs_max_mps2 and
            abs(yaw_rate) <= self.stationary_yaw_rate_abs_radps)
        self.stationary_time = (
            self.stationary_time + dt if calm_stationary_sample else 0.0)
        if self.stationary_time >= self.stationary_hold_s:
            self.speed = self.u = self.v = 0.0
            self.turn = False
            self.calm = 0.0
            self.last_pred = 0.0
            self.wheel_dropout_active = False
            self.wheel_burst_recovery_pending = False
            dyaw = _wrap(yaw - self.previous_yaw)
            yaw_mid = _wrap(self.previous_yaw + 0.5 * dyaw)
            u_mid = 0.5 * (self.previous_pose_u + self.u)
            v_mid = 0.5 * (self.previous_pose_v + self.v)
            self.x += (u_mid * math.cos(yaw_mid) -
                       v_mid * math.sin(yaw_mid)) * dt
            self.y += (u_mid * math.sin(yaw_mid) +
                       v_mid * math.cos(yaw_mid)) * dt
            self.previous_stamp, self.previous_left, self.previous_right = (
                stamp, left, right)
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.previous_pose_u = self.u
            self.previous_pose_v = self.v
            return self._result(row, dt)
        yaw_alpha = (yaw_rate - self.previous_yaw_rate) / dt
        ax_origin = ax + yaw_rate * yaw_rate * self.imu_x_offset_m
        ay_origin = ay - yaw_alpha * self.imu_x_offset_m
        def launch_wheel_spin(predicted: float) -> bool:
            return (predicted < self.wheel_recovery_launch_speed_mps and
                    packet_mapped > self.wheel_recovery_launch_wheel_speed_mps and
                    packet_mapped > predicted + self.wheel_recovery_launch_innovation_mps)
        if not self.turn and (abs(yaw_rate) >= self.turn_enter_yaw_rate_radps or
                              abs(ay) >= self.turn_enter_abs_ay_mps2):
            turn_bias = self.turn_speed_bias(mapped, yaw_rate)
            self.turn = True
            self.calm = 0.0
            if (not self.wheel_dropout_active and
                    packet >= self.wheel_freeze_speed_mps and
                    np.isfinite(mapped)):
                self.u = self.speed = max(0.0, mapped + turn_bias)
            else:
                self.u = self.speed
            self.v = 0.0
        wheel_used = False
        pose_u = self.u
        pose_v = self.v
        pred = self.speed
        integrate_lateral_dynamics = (
            self.integrate_lateral_acceleration_in_turn or
            self.wheel_dropout_active)
        def wheel_speed_is_valid(predicted: float) -> bool:
            if (dt > self.normal_packet_dt_max_s and
                    dt > self.max_integratable_gap_s) or not np.isfinite(mapped):
                return False
            if raw < self.wheel_freeze_speed_mps and predicted > 0.5:
                return False
            if launch_wheel_spin(predicted):
                return False
            if wheel_slew_rejected:
                return False
            return (abs(mapped - predicted) <= self.wheel_innovation_max_mps or
                    (predicted < self.wheel_freeze_speed_mps and
                     mapped >= self.stationary_speed_threshold_mps))

        if self.turn:
            turn_bias = self.turn_speed_bias(mapped, yaw_rate)
            turn_mapped = max(0.0, mapped + turn_bias)
            turn_packet_mapped = max(
                0.0, packet_mapped + self.turn_speed_bias(packet_mapped, yaw_rate))
            self.last_turn_speed_bias = turn_bias
            normal_wheel_recovery = (self.wheel_dropout_active and
                              packet >= self.wheel_freeze_speed_mps and
                              not self.wheel_burst_rejected and
                              not launch_wheel_spin(self.speed) and
                              (not self.wheel_burst_recovery_pending or
                               (self.wheel_burst_disagreement_mps > 0.0 and
                                abs(turn_packet_mapped - turn_mapped) <=
                                self.wheel_burst_disagreement_mps)) and
                               (abs(turn_packet_mapped - self.speed) <=
                               self.wheel_innovation_max_mps) and
                               turn_packet_mapped <= self.speed +
                               self.turn_current_packet_max_increase_mps)
            turn_current_packet_recovery = (
                self.allow_turn_current_packet_recovery and
                self.wheel_dropout_active and self.wheel_burst_recovery_pending and
                not self.wheel_burst_rejected and not launch_wheel_spin(self.speed) and
                packet >= self.wheel_freeze_speed_mps and
                turn_packet_mapped >= self.speed and
                turn_packet_mapped <= self.speed + self.turn_current_packet_max_increase_mps and
                self.wheel_burst_disagreement_mps > 0.0 and
                turn_packet_mapped > turn_mapped + self.wheel_burst_disagreement_mps)
            wheel_recovery = normal_wheel_recovery or turn_current_packet_recovery
            wheel_coherent = (not self.wheel_burst_rejected and
                              packet >= self.wheel_freeze_speed_mps and
                              not launch_wheel_spin(self.speed) and
                              self.wheel_burst_disagreement_mps > 0.0 and
                              not wheel_slew_rejected and
                              abs(turn_packet_mapped - turn_mapped) <=
                              self.wheel_burst_disagreement_mps)
            wheel_ok = (not self.wheel_burst_rejected and
                        not self.wheel_dropout_active and
                        (wheel_speed_is_valid(self.speed) or wheel_coherent))
            if wheel_ok:
                self.u = turn_mapped
                wheel_used = True
            elif wheel_recovery:
                self.u = turn_packet_mapped
                self.wheel_dropout_active = False
                self.wheel_burst_recovery_pending = False
                self.encoder_history.clear()
                self.encoder_history.append((stamp, left, right))
                wheel_used = True
            elif self.wheel_dropout_active and not integrate_lateral_dynamics:
                # Repeated cumulative encoder samples are missing motion, not
                # zero vehicle speed. Propagate the causal longitudinal speed
                # with the synchronized IMU until a coherent packet returns.
                braking_ax = ax
                if ax < self.decel_detect_ax_mps2:
                    braking_ax = (self.decel_ax_scale * ax +
                                  self.decel_ax_offset_mps2)
                braking_ax += yaw_rate * yaw_rate * self.imu_x_offset_m
                if braking_ax < 0.0:
                    self.u = max(0.0, self.u + braking_ax * dt)
            elif (not integrate_lateral_dynamics and
                  raw < self.wheel_freeze_speed_mps and
                  ax <= self.turn_wheel_braking_ax_mps2):
                braking_ax = ax
                if ax < self.decel_detect_ax_mps2:
                    braking_ax = (self.decel_ax_scale * ax +
                                  self.decel_ax_offset_mps2)
                braking_ax += yaw_rate * yaw_rate * self.imu_x_offset_m
                if braking_ax < 0.0:
                    self.u = max(0.0, self.u + braking_ax * dt)
            if integrate_lateral_dynamics:
                # Keep the calibrated longitudinal braking correction active
                # during a turn dropout. The RK2 update remains necessary for
                # lateral propagation, but raw ax here would bypass the
                # sensor-only dropout calibration.
                turn_ax_origin = ax_origin
                if self.wheel_dropout_active and ax < self.decel_detect_ax_mps2:
                    turn_ax_origin = (
                        self.decel_ax_scale * ax + self.decel_ax_offset_mps2 +
                        yaw_rate * yaw_rate * self.imu_x_offset_m)
                du = turn_ax_origin + yaw_rate * self.v
                dv = ay_origin - yaw_rate * self.u
                u_mid = self.u + 0.5 * dt * du
                v_mid = self.v + 0.5 * dt * dv
                self.u += dt * (turn_ax_origin + yaw_rate * v_mid)
                self.v += dt * (ay_origin - yaw_rate * u_mid)
                self.u = float(np.clip(self.u, -30.0, 30.0))
                self.v = float(np.clip(self.v, -30.0, 30.0))
                if wheel_ok or wheel_recovery:
                    # Keep a fresh synchronized recovery packet as the
                    # longitudinal anchor. The dynamic update still computes
                    # the lateral state, but applying its ax correction after
                    # recovery would create an artificial under-speed step.
                    self.u = (turn_packet_mapped if wheel_recovery else
                              turn_mapped)
            else:
                self.v = self.kinematic_lateral_velocity(yaw_rate, self.u)
            coherent_current_packet = (
                wheel_used and not self.wheel_burst_rejected and
                packet >= self.wheel_freeze_speed_mps and
                self.wheel_burst_disagreement_mps > 0.0 and
                abs(packet_mapped - mapped) <= self.wheel_burst_disagreement_mps)
            if (self.use_coherent_packet_velocity_for_pose and
                    coherent_current_packet and np.isfinite(packet_mapped)):
                pose_u = ((1.0 - self.coherent_packet_pose_blend) * self.u +
                          self.coherent_packet_pose_blend * turn_packet_mapped)
                pose_v = self.v
            else:
                pose_u, pose_v = self.u, self.v
            self.speed = math.hypot(self.u, self.v)
            pred = self.speed
            calm = (abs(yaw_rate) < self.turn_exit_yaw_rate_radps and
                    abs(ay) < self.turn_exit_abs_ay_mps2)
            self.calm = self.calm + dt if calm else 0.0
            if self.calm >= self.turn_exit_hold_s:
                self.speed = math.hypot(self.u, self.v)
                self.u, self.v = self.speed, 0.0
                self.turn = False
                self.calm = 0.0
                pose_u, pose_v = self.u, self.v
        else:
            ax_effective = ax
            if ax < self.decel_detect_ax_mps2:
                ax_effective = self.decel_ax_scale * ax + self.decel_ax_offset_mps2
            pred = max(0.0, self.speed + ax_effective * dt)
            self.speed = pred
            # Keep short degraded gaps usable: synchronized encoder endpoints
            # still describe displacement inside the integratable horizon.
            if dt <= self.max_integratable_gap_s:
                wheel_recovery = (self.wheel_dropout_active and
                                  packet >= self.wheel_freeze_speed_mps and
                                  not self.wheel_burst_rejected and
                                  not launch_wheel_spin(pred) and
                                  (not self.wheel_burst_recovery_pending or
                                   (self.wheel_burst_disagreement_mps > 0.0 and
                                    abs(packet_mapped - mapped) <=
                                    self.wheel_burst_disagreement_mps)) and
                                  (abs(packet_mapped - pred) <=
                                   self.wheel_innovation_max_mps))
                wheel_ok = abs(ax) < self.wheel_update_ax_abs_max_mps2 and (
                    (wheel_recovery if self.wheel_dropout_active else
                     (wheel_speed_is_valid(pred) or
                     (not self.wheel_burst_rejected and
                      packet >= self.wheel_freeze_speed_mps and
                       not launch_wheel_spin(pred) and
                       not wheel_slew_rejected and
                       self.wheel_burst_disagreement_mps > 0.0 and
                       abs(packet_mapped - mapped) <=
                       self.wheel_burst_disagreement_mps))))
                if wheel_ok:
                    if wheel_recovery:
                        self.speed = packet_mapped
                        self.wheel_dropout_active = False
                        self.wheel_burst_recovery_pending = False
                        self.encoder_history.clear()
                        self.encoder_history.append((stamp, left, right))
                    else:
                        self.speed = mapped if pred < self.wheel_freeze_speed_mps else (
                            (1.0 - self.wheel_update_beta) * pred
                            + self.wheel_update_beta * mapped)
                    wheel_used = True
            self.u, self.v = self.speed, 0.0
            pose_u, pose_v = self.u, self.v

        dyaw = _wrap(yaw - self.previous_yaw)
        yaw_mid = _wrap(self.previous_yaw + 0.5 * dyaw)
        u_mid = 0.5 * (self.previous_pose_u + pose_u)
        v_mid = 0.5 * (self.previous_pose_v + pose_v)
        self.x += (u_mid * math.cos(yaw_mid) - v_mid * math.sin(yaw_mid)) * dt
        self.y += (u_mid * math.sin(yaw_mid) + v_mid * math.cos(yaw_mid)) * dt
        self.previous_pose_u, self.previous_pose_v = pose_u, pose_v
        self.last_pred = pred
        self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
        self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
        result = self._result(row, dt)
        result.wheel_update_used = wheel_used
        result.timing_degraded = dt > self.normal_packet_dt_max_s
        return result
