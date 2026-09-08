"""Numerically equivalent Python reference for the deterministic observer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

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
    wheel_update_used: bool = False
    turn_mode: bool = False
    reset_epoch: bool = False
    timing_degraded: bool = False
    sensor_outlier: bool = False


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ReferenceObserver:
    def __init__(self, map_path: str | Path | None = None,
                 normal_packet_dt_max_s: float = 0.080,
                 integrate_lateral_acceleration_in_turn: bool = False) -> None:
        if map_path is None:
            map_path = Path(__file__).parents[2] / "config" / "wheel_speed_map.csv"
        table = pd.read_csv(map_path)
        self.wheel = table.wheel_speed_mps.to_numpy(dtype=float)
        self.body = table.body_speed_mps.to_numpy(dtype=float)
        self.normal_packet_dt_max_s = float(normal_packet_dt_max_s)
        self.integrate_lateral_acceleration_in_turn = bool(
            integrate_lateral_acceleration_in_turn)
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

    def map_wheel(self, value: float) -> float:
        if not np.isfinite(value) or value <= 0.0:
            return 0.0
        if value < self.wheel[0]:
            # The identified table starts at 0.703 m/s.  Preserve a valid
            # low-speed encoder measurement instead of mapping it to zero.
            return float(value)
        if value >= self.wheel[-1]:
            return float(self.body[-1])
        return float(np.interp(value, self.wheel, self.body))

    def _result(self, row: pd.Series, dt: float = 0.0) -> Estimate:
        return Estimate(
            stamp_s=float(row.stamp_s), dt_s=dt,
            speed_pred_mps=self.last_pred, speed_mps=self.speed,
            body_u_mps=self.u, body_v_mps=self.v, x_m=self.x, y_m=self.y,
            wheel_raw_mps=self.last_raw, wheel_mapped_mps=self.last_mapped,
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
            result = self._result(row, dt)
            result.reset_epoch = True
            return result
        if dt > 0.100:
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.last_pred = self.speed
            self.last_raw = self.last_mapped = 0.0
            result = self._result(row, dt)
            result.timing_degraded = True
            return result

        raw = abs(0.059 * 0.5 * (dl + dr) / dt)
        mapped = self.map_wheel(raw)
        self.last_raw, self.last_mapped = raw, mapped
        if abs(ax) > 30.0:
            self.previous_stamp, self.previous_left, self.previous_right = stamp, left, right
            self.previous_yaw, self.previous_yaw_rate = yaw, yaw_rate
            self.last_pred = self.speed
            result = self._result(row, dt)
            result.timing_degraded = True
            result.sensor_outlier = True
            return result
        yaw_alpha = (yaw_rate - self.previous_yaw_rate) / dt
        ax_origin = ax + yaw_rate * yaw_rate * 0.08
        ay_origin = ay - yaw_alpha * 0.08
        if not self.turn and (abs(yaw_rate) >= 0.6 or abs(ay) >= 6.0):
            self.turn = True
            self.calm = 0.0
            self.u, self.v = self.speed, 0.0
        wheel_used = False
        pred = self.speed
        def wheel_speed_is_valid(predicted: float) -> bool:
            if dt > self.normal_packet_dt_max_s or not np.isfinite(mapped):
                return False
            if raw < 0.15 and predicted > 0.5:
                return False
            return (abs(mapped - predicted) <= 0.30 or
                    (predicted < 0.15 and mapped >= 0.15))

        if self.turn:
            wheel_ok = wheel_speed_is_valid(self.speed)
            if wheel_ok:
                self.u = mapped
                wheel_used = True
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
                wheel_ok = abs(ax) < 0.6 and wheel_speed_is_valid(pred)
                if wheel_ok:
                    self.speed = mapped if pred < 0.15 else 0.8 * pred + 0.2 * mapped
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
