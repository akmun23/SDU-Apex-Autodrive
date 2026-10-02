"""Explicit race-domain nominal plus history-conditioned dynamics residual."""

from __future__ import annotations

import math

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    COM_X_M,
    DT_S,
    WHEEL_RADIUS_M,
)


HISTORY_STEPS = 80
LATENT_SIZE = 32
RESIDUAL_NAMES = (
    "delta_ax_mps2", "delta_ay_mps2", "delta_yaw_accel_rps2",
    "delta_rear_left_surface_accel_mps2",
    "delta_rear_right_surface_accel_mps2",
)
HISTORY_FEATURE_SIZE = 19
RESIDUAL_LIMITS = (6.0, 10.0, 50.0, 150.0, 150.0)
STEP_FEATURE_SCALE = np.asarray((
    8.0, 4.0, 5.0, 20.0, 20.0, 0.5236, 1.0, 0.5236, 1.0,
    10.0, 10.0, 50.0,
), dtype=np.float32)


def local_pose_targets(poses: np.ndarray) -> np.ndarray:
    """Represent simulator pose samples in the rollout's initial frame."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("pose window must have shape (N>=2, 3)")
    if not np.isfinite(values).all():
        raise ValueError("pose window contains non-finite values")
    dx = values[1:, 0] - values[0, 0]
    dy = values[1:, 1] - values[0, 1]
    cosine, sine = math.cos(values[0, 2]), math.sin(values[0, 2])
    local = np.empty((len(values) - 1, 3), dtype=np.float32)
    local[:, 0] = cosine * dx + sine * dy
    local[:, 1] = -sine * dx + cosine * dy
    delta_yaw = values[1:, 2] - values[0, 2]
    local[:, 2] = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
    return local


def integrate_pose(torch, predicted_states, initial_state,
                   detach_every_steps: int = 0):
    """Integrate predicted body states into rear-axle-relative pose."""
    previous = initial_state[:, :3]
    pose = torch.zeros((len(previous), 3), dtype=previous.dtype,
                       device=previous.device)
    trajectory = []
    for index in range(predicted_states.shape[1]):
        following = predicted_states[:, index, :3]
        u = 0.5 * (previous[:, 0] + following[:, 0])
        r = 0.5 * (previous[:, 2] + following[:, 2])
        v_rear = 0.5 * (
            previous[:, 1] - COM_X_M * previous[:, 2]
            + following[:, 1] - COM_X_M * following[:, 2])
        yaw_mid = pose[:, 2] + 0.5 * r * DT_S
        dx = (u * torch.cos(yaw_mid) - v_rear * torch.sin(yaw_mid)) * DT_S
        dy = (u * torch.sin(yaw_mid) + v_rear * torch.cos(yaw_mid)) * DT_S
        pose = torch.stack((pose[:, 0] + dx, pose[:, 1] + dy,
                            pose[:, 2] + r * DT_S), dim=-1)
        trajectory.append(pose)
        previous = following
        if (detach_every_steps and (index + 1) % detach_every_steps == 0
                and index + 1 < predicted_states.shape[1]):
            pose = pose.detach()
            previous = previous.detach()
    return torch.stack(trajectory, dim=1)


def hybrid_model_type(torch, nn, nominal_model, history_mean: np.ndarray,
                      history_scale: np.ndarray, latent_size: int = LATENT_SIZE):
    """Build the deterministic nominal + 2 s encoder + residual teacher."""
    if latent_size not in (16, 32, 64):
        raise ValueError("latent size must be 16, 32, or 64")
    mean = np.asarray(history_mean, dtype=np.float32)
    scale = np.asarray(history_scale, dtype=np.float32)
    if (mean.shape != (HISTORY_FEATURE_SIZE,)
            or scale.shape != mean.shape or np.any(scale <= 0.0)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()):
        raise ValueError("invalid legal-sensor history normalizers")

    class HybridRaceTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.nominal = nominal_model
            self.latent_size = latent_size
            self.register_buffer("history_mean", torch.as_tensor(mean))
            self.register_buffer("history_scale", torch.as_tensor(scale))
            self.register_buffer(
                "step_feature_scale", torch.as_tensor(STEP_FEATURE_SCALE))
            self.register_buffer(
                "residual_limits", torch.as_tensor(RESIDUAL_LIMITS))
            self.history_encoder = nn.GRU(
                HISTORY_FEATURE_SIZE, 64, batch_first=True)
            self.history_projection = nn.Sequential(
                nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, latent_size),
                nn.Tanh(),
            )
            self.residual_head = nn.Sequential(
                nn.Linear(12 + latent_size, 128), nn.SiLU(),
                nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 5),
            )
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
            self.latent_update = nn.GRUCell(17, latent_size)

        def encode_history(self, history):
            normalized = (history - self.history_mean) / self.history_scale
            _, hidden = self.history_encoder(normalized)
            return self.history_projection(hidden[-1])

        def step(self, state, latent, command, dt: float = DT_S):
            nominal_next, nominal_acceleration = self.nominal.step(
                state, command, dt)
            features = torch.cat((
                state[:, 0:3], state[:, 5:7] * WHEEL_RADIUS_M,
                state[:, 17:19], command, nominal_acceleration,
            ), dim=-1) / self.step_feature_scale
            residual = torch.tanh(self.residual_head(torch.cat(
                (features, latent), dim=-1))) * self.residual_limits
            following = nominal_next.clone()
            following[:, :3] = following[:, :3] + dt * residual[:, :3]
            following[:, 5:7] = following[:, 5:7] + (
                dt * residual[:, 3:5] / WHEEL_RADIUS_M)
            latent_next = self.latent_update(
                torch.cat((features, residual / self.residual_limits), dim=-1),
                latent)
            acceleration = nominal_acceleration + residual[:, :3]
            nominal_wheel_accel = (
                (nominal_next[:, 5:7] - state[:, 5:7])
                * WHEEL_RADIUS_M / dt)
            wheel_acceleration = nominal_wheel_accel + residual[:, 3:5]
            return following, latent_next, acceleration, wheel_acceleration, residual

        def rollout(self, initial_state, history, commands,
                    detach_every_steps: int = 0):
            if (history.ndim != 3 or history.shape[1] != HISTORY_STEPS
                    or history.shape[2] != HISTORY_FEATURE_SIZE):
                raise ValueError("rollout requires a causal 2 s legal-sensor history")
            if detach_every_steps < 0:
                raise ValueError("detach interval must be nonnegative")
            state = initial_state
            latent = self.encode_history(history)
            states, accelerations, wheel_accelerations, residuals = [], [], [], []
            for index in range(commands.shape[1]):
                state, latent, acceleration, wheel_acceleration, residual = self.step(
                    state, latent, commands[:, index])
                states.append(state)
                accelerations.append(acceleration)
                wheel_accelerations.append(wheel_acceleration)
                residuals.append(residual)
                if (detach_every_steps
                        and (index + 1) % detach_every_steps == 0
                        and index + 1 < commands.shape[1]):
                    state = state.detach()
                    latent = latent.detach()
            return (
                torch.stack(states, dim=1),
                torch.stack(accelerations, dim=1),
                torch.stack(wheel_accelerations, dim=1),
                torch.stack(residuals, dim=1),
            )

    return HybridRaceTeacher
