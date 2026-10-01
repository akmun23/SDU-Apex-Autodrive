"""Python binding to the exact production sensor odometry implementation."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_OUTPUT_NAMES = (
    "stamp_s", "dt_s", "speed_pred_mps", "speed_mps", "body_u_mps",
    "body_v_mps", "x_m", "y_m", "yaw_rad", "wheel_raw_mps",
    "wheel_mapped_mps", "wheel_packet_mps", "turn_speed_bias_mps",
    "ax_mps2", "ay_mps2", "yaw_rate_radps", "wheel_update_used",
    "wheel_burst_rejected", "turn_mode", "reset_epoch", "timing_degraded",
    "sensor_outlier", "valid",
)


@dataclass(frozen=True)
class OdometryState:
    """Named snapshot of the production observer's diagnostic output."""

    values: dict[str, float]

    def __getattr__(self, name: str) -> float:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_array(self) -> np.ndarray:
        return np.asarray([self.values[name] for name in _OUTPUT_NAMES],
                          dtype=np.float64)


class ProductionOdometry:
    """Run production packet assembly, yaw handling, and odometry offline.

    Inputs must be the sensor-only channels available to deployment. The class
    accepts an output library compiled from the repository's production C++
    sources and reads the same deployment YAML as the ROS node.
    """

    def __init__(self, library: Path, config_yaml: Path) -> None:
        self.library = ctypes.CDLL(str(library.resolve()))
        self._configure_abi()
        message = ctypes.create_string_buffer(2048)
        self._handle = self.library.offline_odom_create(
            str(config_yaml.resolve()).encode(), message, len(message))
        if not self._handle:
            raise ValueError(message.value.decode(errors="replace"))

    def _configure_abi(self) -> None:
        lib = self.library
        lib.offline_odom_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                             ctypes.c_size_t]
        lib.offline_odom_create.restype = ctypes.c_void_p
        lib.offline_odom_destroy.argtypes = [ctypes.c_void_p]
        lib.offline_odom_reset.argtypes = [ctypes.c_void_p]
        lib.offline_odom_add_left.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                               ctypes.c_double]
        lib.offline_odom_add_right.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                                ctypes.c_double]
        lib.offline_odom_add_imu.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                              ctypes.c_double, ctypes.c_double,
                                              ctypes.c_double, ctypes.c_double]
        lib.offline_odom_get_output.argtypes = [ctypes.c_void_p,
                                                 ctypes.POINTER(ctypes.c_double),
                                                 ctypes.c_size_t]
        lib.offline_odom_get_output.restype = ctypes.c_int

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.library.offline_odom_destroy(self._handle)
            self._handle = None

    def reset(self) -> None:
        self.library.offline_odom_reset(self._handle)

    def add_left_encoder(self, stamp_ns: int, cumulative_angle_rad: float) -> None:
        self.library.offline_odom_add_left(self._handle, int(stamp_ns),
                                           float(cumulative_angle_rad))

    def add_right_encoder(self, stamp_ns: int, cumulative_angle_rad: float) -> None:
        self.library.offline_odom_add_right(self._handle, int(stamp_ns),
                                            float(cumulative_angle_rad))

    def add_imu(self, stamp_ns: int, ax_mps2: float, ay_mps2: float,
                yaw_rate_radps: float, yaw_rad: float) -> None:
        self.library.offline_odom_add_imu(
            self._handle, int(stamp_ns), float(ax_mps2), float(ay_mps2),
            float(yaw_rate_radps), float(yaw_rad))

    def snapshot(self) -> OdometryState | None:
        output = (ctypes.c_double * len(_OUTPUT_NAMES))()
        ready = self.library.offline_odom_get_output(
            self._handle, output, len(_OUTPUT_NAMES))
        if not ready:
            return None
        values = dict(zip(_OUTPUT_NAMES, output))
        return OdometryState(values)

    def __enter__(self) -> "ProductionOdometry":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

