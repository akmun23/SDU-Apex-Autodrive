"""Causal synthetic IMU generation and train-run sensor residual calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import COM_X_M


@dataclass(frozen=True)
class ImuNoiseProfile:
    residual_rows: np.ndarray
    training_run_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residual_rows)
        if residuals.ndim != 2 or residuals.shape[1] != 3 or len(residuals) < 2:
            raise ValueError("IMU residual profile must be Nx3 with N >= 2")
        if not np.isfinite(residuals).all():
            raise ValueError("IMU residual profile contains non-finite samples")

    @property
    def bias(self) -> np.ndarray:
        return np.mean(self.residual_rows, axis=0)

    @property
    def standard_deviation(self) -> np.ndarray:
        return np.std(self.residual_rows, axis=0, ddof=1)

    @classmethod
    def from_training_data(cls, data: dict[str, Any],
                           samples_per_run: int = 5000,
                           seed: int = 20261012) -> "ImuNoiseProfile":
        """Resample joint IMU-minus-truth residual rows from train runs only."""
        sensors = data.get("sensor_frames")
        rigid = data.get("simulator_rigid_state")
        acceleration = data.get("simulator_linear_acceleration")
        valid = data.get("sensor_valid")
        if sensors is None or rigid is None or acceleration is None or valid is None:
            raise ValueError("dataset lacks aligned sensors/truth required for IMU calibration")
        rng = np.random.default_rng(seed)
        residual_rows, source_ids = [], []
        for run_index, (run_id, split) in enumerate(zip(
                data["run_ids"], data["splits"])):
            if split != "train":
                continue
            row_ids = []
            for seq_id, (start_value, end_value) in enumerate(data["bounds"]):
                if int(data["seq_run"][seq_id]) != run_index:
                    continue
                start, end = int(start_value), int(end_value)
                ids = np.arange(start, end, dtype=np.int64)
                finite = (valid[ids]
                          & np.isfinite(sensors[ids, 4:7]).all(axis=1)
                          & np.isfinite(acceleration[ids, :2]).all(axis=1)
                          & np.isfinite(rigid[ids, 12]))
                row_ids.extend(ids[finite].tolist())
            if not row_ids:
                continue
            ids = np.asarray(row_ids, dtype=np.int64)
            if len(ids) > samples_per_run:
                ids = np.sort(rng.choice(ids, size=samples_per_run,
                                         replace=False))
            truth = np.column_stack((acceleration[ids, :2], rigid[ids, 12]))
            residual_rows.append(sensors[ids, 4:7] - truth)
            source_ids.append(str(run_id))
        if not residual_rows:
            raise ValueError("no valid train-run IMU calibration samples")
        return cls(np.concatenate(residual_rows).astype(np.float32),
                   tuple(source_ids))


@dataclass(frozen=True)
class ImuSample:
    stamp_s: float
    linear_acceleration_x_mps2: float
    linear_acceleration_y_mps2: float
    angular_velocity_z_radps: float
    orientation_yaw_rad: float


class SyntheticImu:
    """Synthesize body-frame IMU signals from causal rear-axle motion states."""

    def __init__(self, noise_profile: ImuNoiseProfile | None = None,
                 seed: int = 20261013) -> None:
        self.noise_profile = noise_profile
        self.rng = np.random.default_rng(seed)
        self._stamp_s = 0.0
        self._state = np.zeros(3, dtype=np.float64)
        self._pose = np.zeros(3, dtype=np.float64)
        self._initialized = False

    def reset(self, stamp_s: float, state: np.ndarray,
              pose_xyyaw: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        pose = np.asarray(pose_xyyaw, dtype=np.float64)
        if state.shape[0] < 3 or pose.shape != (3,):
            raise ValueError("IMU reset requires u/v/r and x/y/yaw")
        if not np.isfinite(state[:3]).all() or not np.isfinite(pose).all():
            raise ValueError("IMU reset state must be finite")
        self._stamp_s = float(stamp_s)
        self._state = state[:3].copy()
        self._pose = pose.copy()
        self._initialized = True

    def step(self, state: np.ndarray, pose_xyyaw: np.ndarray,
             dt_s: float) -> ImuSample:
        if not self._initialized:
            raise RuntimeError("IMU must be reset before stepping")
        current = np.asarray(state, dtype=np.float64)
        pose = np.asarray(pose_xyyaw, dtype=np.float64)
        if current.shape[0] < 3 or pose.shape != (3,) or dt_s <= 0.0:
            raise ValueError("IMU step requires u/v/r, x/y/yaw and positive dt")
        if not np.isfinite(current[:3]).all() or not np.isfinite(pose).all():
            raise ValueError("IMU step state must be finite")
        previous_u, previous_v, previous_r = self._state
        following_u, following_v, following_r = current[:3]
        u_mid = 0.5 * (previous_u + following_u)
        v_mid = 0.5 * (previous_v + following_v)
        r_mid = 0.5 * (previous_r + following_r)
        alpha = (following_r - previous_r) / dt_s
        rear_ax = (following_u - previous_u) / dt_s - r_mid * v_mid
        rear_ay = (following_v - previous_v) / dt_s + r_mid * u_mid
        com_ax = rear_ax - r_mid * r_mid * COM_X_M
        com_ay = rear_ay + alpha * COM_X_M
        measured = np.asarray((com_ax, com_ay, following_r), dtype=np.float64)
        if self.noise_profile is not None:
            residuals = np.asarray(self.noise_profile.residual_rows,
                                   dtype=np.float64)
            index = int(self.rng.integers(0, len(residuals)))
            measured += residuals[index]
        self._stamp_s += float(dt_s)
        self._state = current[:3].copy()
        self._pose = pose.copy()
        return ImuSample(self._stamp_s, float(measured[0]),
                         float(measured[1]), float(measured[2]),
                         float(pose[2]))
