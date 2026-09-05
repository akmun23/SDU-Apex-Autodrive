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


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ReferenceObserver:
    def __init__(self, map_path: str | Path | None = None) -> None:
        if map_path is None:
            map_path = Path(__file__).parents[2] / "config" / "wheel_speed_map.csv"
        table = pd.read_csv(map_path)
        self.wheel = table.wheel_speed_mps.to_numpy(dtype=float)
        self.body = table.body_speed_mps.to_numpy(dtype=float)
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
        if not np.isfinite(value) or value <= 0.0 or value < self.wheel[0]:
            return 0.0
        if value >= self.wheel[-1]:
            return float(self.body[-1])
        return float(np.interp(value, self.wheel, self.body))

    def _result(self, row: pd.Series, dt: float = 0.0) -> Estimate:
        return Estimate(
            stamp_s=float(row.stamp_s), dt_s=dt,
            speed_pred_mps=self.last_pred, speed_mps=self.speed,
            body_u_mps=self.u, body_v_mps=self.v, x_m=self.x, y_m=self.y,
            wheel_raw_mps=self.last_raw, wheel_mapped_mps=self.last_mapped,
            turn_mode=self.turn)

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
        yaw_alpha = (yaw_rate - self.previous_yaw_rate) / dt
        ax_origin = ax + yaw_rate * yaw_rate * 0.08
        ay_origin = ay - yaw_alpha * 0.08
        if not self.turn and (abs(yaw_rate) >= 0.6 or abs(ay) >= 6.0):
            self.turn = True
            self.calm = 0.0
            self.u, self.v = self.speed, 0.0
        wheel_used = False
        pred = self.speed
        if self.turn:
            du = ax_origin + yaw_rate * self.v
            dv = ay_origin - yaw_rate * self.u
            u_mid = self.u + 0.5 * dt * du
            v_mid = self.v + 0.5 * dt * dv
            self.u += dt * (ax_origin + yaw_rate * v_mid)
            self.v += dt * (ay_origin - yaw_rate * u_mid)
            self.u = float(np.clip(self.u, -30.0, 30.0))
            self.v = float(np.clip(self.v, -30.0, 30.0))
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
            if dt <= 0.040:
                wheel_ok = (abs(ax) < 0.6 and not (raw < 0.15 and pred > 0.5)
                            and abs(mapped - pred) <= 0.30)
                if wheel_ok:
                    self.speed = 0.8 * pred + 0.2 * mapped
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
        result.timing_degraded = dt > 0.040
        return result
