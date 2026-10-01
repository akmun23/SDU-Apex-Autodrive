"""Quantized rear-wheel encoder measurements from predicted wheel motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncoderSample:
    stamp_s: float
    left_angle_rad: float
    right_angle_rad: float
    left_count: int
    right_count: int


class SyntheticRearEncoders:
    """Integrate wheel surface speed, then quantize at the published encoder resolution."""

    def __init__(self, wheel_radius_m: float = 0.059,
                 pulses_per_revolution: int = 16,
                 conversion_ratio: int = 120) -> None:
        if wheel_radius_m <= 0.0 or pulses_per_revolution < 1 or conversion_ratio < 1:
            raise ValueError("encoder geometry and resolution must be positive")
        self.wheel_radius_m = float(wheel_radius_m)
        self.counts_per_wheel_revolution = int(
            pulses_per_revolution * conversion_ratio)
        self.reset()

    def reset(self, stamp_s: float = 0.0,
              wheel_angles_rad: tuple[float, float] = (0.0, 0.0)) -> None:
        if not np.isfinite(stamp_s) or not np.isfinite(wheel_angles_rad).all():
            raise ValueError("encoder reset values must be finite")
        self._stamp_s = float(stamp_s)
        counts = np.rint(
            np.asarray(wheel_angles_rad, dtype=np.float64)
            * self.counts_per_wheel_revolution / (2.0 * np.pi)
        ).astype(np.int64)
        self._counts = counts
        self._angles = counts.astype(np.float64) * (
            2.0 * np.pi / self.counts_per_wheel_revolution)

    def step(self, left_surface_speed_mps: float,
             right_surface_speed_mps: float, dt_s: float) -> EncoderSample:
        values = np.asarray((left_surface_speed_mps,
                             right_surface_speed_mps, dt_s), dtype=np.float64)
        if not np.isfinite(values).all() or dt_s <= 0.0:
            raise ValueError("wheel speeds and positive dt must be finite")
        wheel_delta = values[:2] * dt_s / self.wheel_radius_m
        exact_angle = self._angles + wheel_delta
        counts = np.rint(
            exact_angle * self.counts_per_wheel_revolution / (2.0 * np.pi)
        ).astype(np.int64)
        self._angles = counts.astype(np.float64) * (
            2.0 * np.pi / self.counts_per_wheel_revolution)
        self._counts = counts
        self._stamp_s += float(dt_s)
        return EncoderSample(
            self._stamp_s, float(self._angles[0]), float(self._angles[1]),
            int(counts[0]), int(counts[1]))
