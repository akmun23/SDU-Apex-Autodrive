"""Numerically equivalent Python reference for the deterministic observer."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
import numpy as np
import pandas as pd


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
    wheel_update_used: bool = False
    wheel_burst_rejected: bool = False
    turn_mode: bool = False
    reset_epoch: bool = False
    timing_degraded: bool = False
    sensor_outlier: bool = False


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ReferenceObserver:
    def __init__(self, normal_packet_dt_max_s: float = 0.080,
                 integrate_lateral_acceleration_in_turn: bool = False,
                 wheel_speed_scale: float = 0.982,
                 wheel_burst_disagreement_mps: float = 1.0) -> None:
        self.normal_packet_dt_max_s = float(normal_packet_dt_max_s)
        self.wheel_speed_window_s = 0.10
        self.wheel_speed_scale = max(0.0, float(wheel_speed_scale))
        self.wheel_burst_disagreement_mps = float(wheel_burst_disagreement_mps)
        # Keep the offline replay numerically aligned with the deployed
        # observer.  A stale replay gate can make a valid runtime change look
        # ineffective during offline validation.
        self.wheel_innovation_max_mps = 1.50
        self.wheel_update_beta = 0.85
        self.integrate_lateral_acceleration_in_turn = bool(
            integrate_lateral_acceleration_in_turn)
        self.turn_wheel_braking_ax_mps2 = -1.0
        self.reset()

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
        self.x = 0.0
        self.y = 0.0
        self.last_pred = 0.0
        self.last_raw = 0.0
        self.last_mapped = 0.0
        self.last_packet = 0.0
        self.stationary_time = 0.0
        self.encoder_history = deque()
        self.wheel_dropout_active = False
        self.wheel_burst_rejected = False
        self.wheel_burst_recovery_pending = False

    def map_wheel(self, value: float) -> float:
        return float(value) if np.isfinite(value) and value > 0.0 else 0.0

    def _result(self, row: pd.Series, dt: float = 0.0) -> Estimate:
        return Estimate(
            stamp_s=float(row.stamp_s), dt_s=dt,
            speed_pred_mps=self.last_pred, speed_mps=self.speed,
            body_u_mps=self.u, body_v_mps=self.v, x_m=self.x, y_m=self.y,
            wheel_raw_mps=self.last_raw, wheel_mapped_mps=self.last_mapped,
            wheel_packet_mps=self.last_packet,
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
        if abs(dl) > 50.0 or abs(dr) > 50.0:
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.turn = False
            self.calm = 0.0
            self.speed = self.u = self.v = 0.0
            self.last_pred = self.last_raw = self.last_mapped = 0.0
            self.last_packet = 0.0
            self.wheel_dropout_active = False
            self.wheel_burst_recovery_pending = False
            result = self._result(row, dt)
            result.reset_epoch = True
            return result
        # Match the deployed observer: a short source gap still has valid
        # synchronized encoder endpoints and is integrated. Only a gap longer
        # than the configured integratable horizon is re-baselined.
        if dt > 0.250:
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.last_pred = self.speed
            self.last_raw = self.last_mapped = 0.0
            self.last_packet = 0.0
            self.wheel_dropout_active = False
            self.wheel_burst_recovery_pending = False
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
        raw = (abs(0.059 * 0.5 * ((left - wheel_left) + (right - wheel_right)) /
                   wheel_dt) if wheel_dt > 0.0 else 0.0)
        packet = abs(0.059 * 0.5 * (dl + dr) / dt)
        mapped = self.map_wheel(raw * self.wheel_speed_scale)
        packet_mapped = packet * self.wheel_speed_scale
        self.last_raw, self.last_mapped, self.last_packet = raw, mapped, packet
        self.wheel_burst_rejected = (
            self.wheel_burst_disagreement_mps > 0.0 and
            mapped > self.speed + self.wheel_burst_disagreement_mps and
            packet_mapped > mapped + self.wheel_burst_disagreement_mps)
        if packet < 0.15 and self.speed > 0.5:
            self.wheel_dropout_active = True
        if self.wheel_burst_rejected:
            self.wheel_dropout_active = True
            self.wheel_burst_recovery_pending = True
        if abs(ax) > 30.0:
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.last_pred = self.speed
            result = self._result(row, dt)
            result.timing_degraded = True
            result.sensor_outlier = True
            return result
        calm_stationary_sample = (
            raw < 0.03 and abs(ax) <= 0.25 and abs(ay) <= 0.75 and
            abs(yaw_rate) <= 0.15)
        self.stationary_time = (
            self.stationary_time + dt if calm_stationary_sample else 0.0)
        if self.stationary_time >= 0.10:
            self.speed = self.u = self.v = 0.0
            self.turn = False
            self.calm = 0.0
            self.last_pred = 0.0
            self.wheel_dropout_active = False
            self.wheel_burst_recovery_pending = False
            dyaw = _wrap(yaw - self.previous_yaw)
            yaw_mid = _wrap(self.previous_yaw + 0.5 * dyaw)
            self.x += (self.u * math.cos(yaw_mid) -
                       self.v * math.sin(yaw_mid)) * dt
            self.y += (self.u * math.sin(yaw_mid) +
                       self.v * math.cos(yaw_mid)) * dt
            self.previous_stamp, self.previous_left, self.previous_right = (
                stamp, left, right)
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            return self._result(row, dt)
        yaw_alpha = (yaw_rate - self.previous_yaw_rate) / dt
        ax_origin = ax + yaw_rate * yaw_rate * 0.08
        ay_origin = ay - yaw_alpha * 0.08
        if not self.turn and (abs(yaw_rate) >= 0.6 or abs(ay) >= 6.0):
            self.turn = True
            self.calm = 0.0
            if (not self.wheel_dropout_active and packet >= 0.15 and
                    np.isfinite(mapped)):
                self.u = self.speed = mapped
            else:
                self.u = self.speed
            self.v = 0.0
        wheel_used = False
        pred = self.speed
        def wheel_speed_is_valid(predicted: float) -> bool:
            if (dt > self.normal_packet_dt_max_s and dt > 0.250) or not np.isfinite(mapped):
                return False
            if raw < 0.15 and predicted > 0.5:
                return False
            return (abs(mapped - predicted) <= self.wheel_innovation_max_mps or
                    (predicted < 0.15 and mapped >= 0.03))

        if self.turn:
            wheel_recovery = (self.wheel_dropout_active and packet >= 0.15 and
                              not self.wheel_burst_rejected and
                              (not self.wheel_burst_recovery_pending or
                               (self.wheel_burst_disagreement_mps > 0.0 and
                                abs(packet_mapped - mapped) <=
                                self.wheel_burst_disagreement_mps)) and
                              (abs(packet_mapped - self.speed) <=
                               self.wheel_innovation_max_mps or
                               (self.wheel_dropout_active and
                                self.wheel_burst_disagreement_mps > 0.0 and
                                abs(packet_mapped - mapped) <=
                                self.wheel_burst_disagreement_mps)))
            wheel_coherent = (not self.wheel_burst_rejected and packet >= 0.15 and
                              self.wheel_burst_disagreement_mps > 0.0 and
                              abs(packet_mapped - mapped) <=
                              self.wheel_burst_disagreement_mps)
            wheel_ok = (not self.wheel_burst_rejected and
                        not self.wheel_dropout_active and
                        (wheel_speed_is_valid(self.speed) or wheel_coherent))
            if wheel_ok:
                self.u = mapped
                wheel_used = True
            elif wheel_recovery:
                self.u = packet_mapped
                self.wheel_dropout_active = False
                self.wheel_burst_recovery_pending = False
                self.encoder_history.clear()
                self.encoder_history.append((stamp, left, right))
                wheel_used = True
            elif self.wheel_dropout_active:
                # Repeated cumulative encoder samples are missing motion, not
                # zero vehicle speed. Propagate the causal longitudinal speed
                # with the synchronized IMU until a coherent packet returns.
                braking_ax = ax
                if ax < -0.5:
                    braking_ax = 1.005 * ax + 0.020
                braking_ax += yaw_rate * yaw_rate * 0.08
                self.u = max(0.0, self.u + braking_ax * dt)
            elif raw < 0.15 and ax <= self.turn_wheel_braking_ax_mps2:
                braking_ax = ax
                if ax < -0.5:
                    braking_ax = 1.005 * ax + 0.020
                braking_ax += yaw_rate * yaw_rate * 0.08
                self.u = max(0.0, self.u + braking_ax * dt)
            if self.integrate_lateral_acceleration_in_turn:
                du = ax_origin + yaw_rate * self.v
                dv = ay_origin - yaw_rate * self.u
                u_mid = self.u + 0.5 * dt * du
                v_mid = self.v + 0.5 * dt * dv
                self.u += dt * (ax_origin + yaw_rate * v_mid)
                self.v += dt * (ay_origin - yaw_rate * u_mid)
                self.u = float(np.clip(self.u, -30.0, 30.0))
                self.v = float(np.clip(self.v, -30.0, 30.0))
                if wheel_ok:
                    self.u = mapped
            else:
                self.v = 0.0
            self.speed = math.hypot(self.u, self.v)
            pred = self.speed
            calm = abs(yaw_rate) < 0.1 and abs(ay) < 0.5
            self.calm = self.calm + dt if calm else 0.0
            if self.calm >= 0.5:
                self.speed = math.hypot(self.u, self.v)
                self.u, self.v = self.speed, 0.0
                self.turn = False
                self.calm = 0.0
        else:
            ax_effective = 1.005 * ax + 0.020 if ax < -0.5 else ax
            pred = max(0.0, self.speed + ax_effective * dt)
            self.speed = pred
            if dt <= self.normal_packet_dt_max_s:
                wheel_recovery = (self.wheel_dropout_active and packet >= 0.15 and
                                  not self.wheel_burst_rejected and
                                  (not self.wheel_burst_recovery_pending or
                                   (self.wheel_burst_disagreement_mps > 0.0 and
                                    abs(packet_mapped - mapped) <=
                                    self.wheel_burst_disagreement_mps)) and
                                  (abs(packet_mapped - pred) <=
                                   self.wheel_innovation_max_mps or
                                   (self.wheel_dropout_active and
                                    self.wheel_burst_disagreement_mps > 0.0 and
                                    abs(packet_mapped - mapped) <=
                                    self.wheel_burst_disagreement_mps)))
                wheel_ok = abs(ax) < 6.5 and (
                    (wheel_recovery if self.wheel_dropout_active else
                     (wheel_speed_is_valid(pred) or
                      (not self.wheel_burst_rejected and packet >= 0.15 and
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
                        self.speed = mapped if pred < 0.15 else (
                            (1.0 - self.wheel_update_beta) * pred
                            + self.wheel_update_beta * mapped)
                    wheel_used = True
            self.u, self.v = self.speed, 0.0

        dyaw = _wrap(yaw - self.previous_yaw)
        yaw_mid = _wrap(self.previous_yaw + 0.5 * dyaw)
        self.x += (self.u * math.cos(yaw_mid) - self.v * math.sin(yaw_mid)) * dt
        self.y += (self.u * math.sin(yaw_mid) + self.v * math.cos(yaw_mid)) * dt
        self.last_pred = pred
        self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
        self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
        result = self._result(row, dt)
        result.wheel_update_used = wheel_used
        result.timing_degraded = dt > self.normal_packet_dt_max_s
        return result
