"""History-conditioned offline plant with explicit planar rigid-body kinematics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    FEATURES,
    HIDDEN_SIZE,
    MAX_CONTEXT_STEPS,
)
from tools.vehicle_dynamics_learning.signal_semantics import (
    REAR_AXLE_TO_COM_X_M,
)


def midpoint_acceleration_from_transition(
        current_body: np.ndarray, next_body: np.ndarray, dt_s: float,
        rear_axle_to_com_x_m: float = REAR_AXLE_TO_COM_X_M) -> np.ndarray:
    """Infer interval-midpoint body acceleration from adjacent rigid states."""
    current = np.asarray(current_body, dtype=np.float64)
    following = np.asarray(next_body, dtype=np.float64)
    if (current.shape[-1] != 3 or following.shape[-1] != 3
            or current.shape != following.shape
            or not np.isfinite(current).all() or not np.isfinite(following).all()
            or not np.isfinite(dt_s) or dt_s <= 0.0):
        raise ValueError("body states and timestep must be finite and aligned")
    u, v_rear, yaw_rate = np.moveaxis(current, -1, 0)
    u_next, v_rear_next, yaw_rate_next = np.moveaxis(following, -1, 0)
    v_com = v_rear + rear_axle_to_com_x_m * yaw_rate
    v_com_next = v_rear_next + rear_axle_to_com_x_m * yaw_rate_next
    yaw_mid = 0.5 * (yaw_rate + yaw_rate_next)
    u_mid = 0.5 * (u + u_next)
    v_com_mid = 0.5 * (v_com + v_com_next)
    a_x = (u_next - u) / dt_s - yaw_mid * v_com_mid
    a_y = (v_com_next - v_com) / dt_s + yaw_mid * u_mid
    yaw_accel = (yaw_rate_next - yaw_rate) / dt_s
    return np.stack((a_x, a_y, yaw_accel), axis=-1)


def integrate_planar_body(current_body: np.ndarray, acceleration: np.ndarray,
                          dt_s: float,
                          rear_axle_to_com_x_m: float = REAR_AXLE_TO_COM_X_M
                          ) -> np.ndarray:
    """Implicit-midpoint rigid-body integration for interval accelerations."""
    current = np.asarray(current_body, dtype=np.float64)
    accel = np.asarray(acceleration, dtype=np.float64)
    if (current.shape[-1] != 3 or accel.shape[-1] != 3
            or current.shape != accel.shape
            or not np.isfinite(current).all() or not np.isfinite(accel).all()
            or not np.isfinite(dt_s) or dt_s <= 0.0
            or not np.isfinite(rear_axle_to_com_x_m)):
        raise ValueError("body state, acceleration, and timestep must be finite")
    u, v_rear, yaw_rate = np.moveaxis(current, -1, 0)
    a_x, a_y, yaw_accel = np.moveaxis(accel, -1, 0)
    yaw_rate_next = yaw_rate + yaw_accel * dt_s
    yaw_mid = 0.5 * (yaw_rate + yaw_rate_next)
    v_com = v_rear + rear_axle_to_com_x_m * yaw_rate
    half_coupling = 0.5 * dt_s * yaw_mid
    determinant = 1.0 + half_coupling * half_coupling
    rhs_u = u + half_coupling * v_com + dt_s * a_x
    rhs_v = v_com - half_coupling * u + dt_s * a_y
    u_next = (rhs_u + half_coupling * rhs_v) / determinant
    v_com_next = (-half_coupling * rhs_u + rhs_v) / determinant
    v_rear_next = v_com_next - rear_axle_to_com_x_m * yaw_rate_next
    return np.stack((u_next, v_rear_next, yaw_rate_next), axis=-1)


class RigidAccelerationHistoryTransition(nn.Module):
    """Predict body acceleration, then apply implicit-midpoint kinematics.

    Output acceleration labels are physical body-frame `[a_x, a_y, yaw_ddot]`.
    Steering/throttle feedback increments retain WP28's normalized-state units.
    The module remains an offline research model; it does not alter production.
    """

    def __init__(self, state_mean: np.ndarray, state_scale: np.ndarray,
                 acceleration_mean: np.ndarray, acceleration_scale: np.ndarray,
                 actuator_delta_mean: np.ndarray,
                 actuator_delta_scale: np.ndarray, dt_s: float = 0.025,
                 rear_axle_to_com_x_m: float = REAR_AXLE_TO_COM_X_M) -> None:
        super().__init__()
        state_mean = np.asarray(state_mean, dtype=np.float32)
        state_scale = np.asarray(state_scale, dtype=np.float32)
        acceleration_mean = np.asarray(acceleration_mean, dtype=np.float32)
        acceleration_scale = np.asarray(acceleration_scale, dtype=np.float32)
        actuator_delta_mean = np.asarray(actuator_delta_mean, dtype=np.float32)
        actuator_delta_scale = np.asarray(actuator_delta_scale, dtype=np.float32)
        if (state_mean.shape != (5,) or state_scale.shape != (5,)
                or acceleration_mean.shape != (3,)
                or acceleration_scale.shape != (3,)
                or actuator_delta_mean.shape != (2,)
                or actuator_delta_scale.shape != (2,)
                or not all(np.isfinite(value).all() for value in (
                    state_mean, state_scale, acceleration_mean,
                    acceleration_scale, actuator_delta_mean,
                    actuator_delta_scale))
                or np.any(state_scale <= 0.0)
                or np.any(acceleration_scale <= 0.0)
                or np.any(actuator_delta_scale <= 0.0)
                or not np.isfinite(dt_s) or dt_s <= 0.0
                or not np.isfinite(rear_axle_to_com_x_m)):
            raise ValueError("rigid-body model scales or geometry are invalid")
        input_size = MAX_CONTEXT_STEPS * FEATURES + MAX_CONTEXT_STEPS + 5 + 2
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, 5),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("state_mean", torch.as_tensor(state_mean))
        self.register_buffer("state_scale", torch.as_tensor(state_scale))
        self.register_buffer("acceleration_mean",
                             torch.as_tensor(acceleration_mean))
        self.register_buffer("acceleration_scale",
                             torch.as_tensor(acceleration_scale))
        self.register_buffer("actuator_delta_mean",
                             torch.as_tensor(actuator_delta_mean))
        self.register_buffer("actuator_delta_scale",
                             torch.as_tensor(actuator_delta_scale))
        self.dt_s = float(dt_s)
        self.rear_axle_to_com_x_m = float(rear_axle_to_com_x_m)
        self.acceleration_supervision_weight = 1.0

    def forward(self, history: torch.Tensor, mask: torch.Tensor,
                state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        delta, _ = self.forward_with_acceleration(history, mask, state, command)
        return delta

    def forward_with_acceleration(
            self, history: torch.Tensor, mask: torch.Tensor,
            state: torch.Tensor, command: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if (history.ndim != 3
                or history.shape[1:] != (MAX_CONTEXT_STEPS, FEATURES)
                or mask.shape != history.shape[:2]
                or state.shape[-1] != 5 or command.shape[-1] != 2):
            raise ValueError("rigid acceleration transition input mismatch")
        features = torch.cat((
            (history * mask.unsqueeze(-1)).flatten(start_dim=1),
            mask, state, command), dim=-1)
        output = self.net(features)
        acceleration = (self.acceleration_mean
                        + self.acceleration_scale * output[:, :3])
        physical = state * self.state_scale + self.state_mean
        u, v_rear, yaw_rate = physical[:, :3].unbind(dim=-1)
        a_x, a_y, yaw_accel = acceleration.unbind(dim=-1)
        v_com = v_rear + self.rear_axle_to_com_x_m * yaw_rate
        yaw_rate_next = yaw_rate + yaw_accel * self.dt_s
        yaw_mid = 0.5 * (yaw_rate + yaw_rate_next)
        half_coupling = 0.5 * self.dt_s * yaw_mid
        determinant = 1.0 + half_coupling.square()
        rhs_u = u + half_coupling * v_com + a_x * self.dt_s
        rhs_v = v_com - half_coupling * u + a_y * self.dt_s
        u_next = (rhs_u + half_coupling * rhs_v) / determinant
        v_com_next = (-half_coupling * rhs_u + rhs_v) / determinant
        v_rear_next = (v_com_next
                       - self.rear_axle_to_com_x_m * yaw_rate_next)
        body_next = torch.stack((u_next, v_rear_next, yaw_rate_next), dim=-1)
        body_next = (body_next - self.state_mean[:3]) / self.state_scale[:3]
        actuator_delta = (self.actuator_delta_mean
                          + self.actuator_delta_scale * output[:, 3:])
        next_state = torch.cat((body_next, state[:, 3:] + actuator_delta), dim=-1)
        return next_state - state, acceleration

    def midpoint_acceleration_label(self, current_state: torch.Tensor,
                                    next_state: torch.Tensor) -> torch.Tensor:
        """Compute a causal-transition target for acceleration supervision."""
        if (current_state.shape != next_state.shape
                or current_state.shape[-1] != 5):
            raise ValueError("midpoint acceleration states are misaligned")
        current = current_state * self.state_scale + self.state_mean
        following = next_state * self.state_scale + self.state_mean
        u, v_rear, yaw_rate = current[:, :3].unbind(dim=-1)
        u_next, v_rear_next, yaw_rate_next = following[:, :3].unbind(dim=-1)
        v_com = v_rear + self.rear_axle_to_com_x_m * yaw_rate
        v_com_next = v_rear_next + self.rear_axle_to_com_x_m * yaw_rate_next
        yaw_mid = 0.5 * (yaw_rate + yaw_rate_next)
        u_mid = 0.5 * (u + u_next)
        v_com_mid = 0.5 * (v_com + v_com_next)
        return torch.stack((
            (u_next - u) / self.dt_s - yaw_mid * v_com_mid,
            (v_com_next - v_com) / self.dt_s + yaw_mid * u_mid,
            (yaw_rate_next - yaw_rate) / self.dt_s,
        ), dim=-1)
