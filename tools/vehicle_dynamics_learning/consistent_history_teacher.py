"""History-consistent 40 Hz teacher plant for offline vehicle simulation.

The same GRU encodes the measured initialization history and advances from
predicted state/command rows during rollout. This removes the separately
parameterized latent transition used by the previous EDSSM family. Rigid-body
transport, actuator lag, rear-wheel contact slip, and roll integration remain
explicit; a bounded network predicts the effective accelerations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    REAR_AXLE_TO_COM_M,
    REAR_TRACK_WIDTH_M,
)


def make_model(torch, nn, *, history_mean: np.ndarray,
               history_scale: np.ndarray, state_mean: np.ndarray,
               state_scale: np.ndarray, command_mean: np.ndarray,
               command_scale: np.ndarray, acceleration_bounds: np.ndarray,
               roll_oscillator_coefficients: np.ndarray,
               steering_delay_steps: int, steering_alpha: float,
               throttle_delay_steps: int, throttle_alpha: float,
               hidden_size: int = 128, latent_size: int = 32):
    """Create the model class lazily so system Python need not import torch."""
    arrays = {
        "history_mean": (history_mean, (11,)),
        "history_scale": (history_scale, (11,)),
        "state_mean": (state_mean, (9,)),
        "state_scale": (state_scale, (9,)),
        "command_mean": (command_mean, (2,)),
        "command_scale": (command_scale, (2,)),
        "acceleration_bounds": (acceleration_bounds, (5,)),
        "roll_oscillator_coefficients": (roll_oscillator_coefficients, (3,)),
    }
    checked: dict[str, np.ndarray] = {}
    for name, (raw, shape) in arrays.items():
        value = np.asarray(raw, dtype=np.float32)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {name} for history-consistent plant")
        if name.endswith("scale") or name == "acceleration_bounds":
            if np.any(value <= 0.0):
                raise ValueError(f"{name} must be strictly positive")
        checked[name] = value
    if steering_delay_steps not in (0, 1) or throttle_delay_steps not in (0, 1):
        raise ValueError("actuator delay must be zero or one sample")
    if not (np.isfinite(steering_alpha) and 0.0 <= steering_alpha <= 1.0
            and np.isfinite(throttle_alpha) and 0.0 <= throttle_alpha <= 1.0):
        raise ValueError("actuator alpha must be finite and in [0, 1]")
    if hidden_size < 16 or latent_size < 8:
        raise ValueError("history model dimensions are unexpectedly small")

    class ConsistentHistoryTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name, value in checked.items():
                self.register_buffer(name, torch.as_tensor(value.copy()))
            self.history_steps = 80
            self.history_state_size = 9
            self.state_size = 9
            self.include_raw_encoder_history = False
            self.include_wheel_innovation_history = False
            self.hidden_size = hidden_size
            self.latent_size = latent_size
            self.steering_delay_steps = steering_delay_steps
            self.throttle_delay_steps = throttle_delay_steps
            self.steering_alpha = float(steering_alpha)
            self.throttle_alpha = float(throttle_alpha)

            # One recurrent operator is shared by initial history encoding and
            # every predicted state/command update. Thus the rollout hidden
            # state is exactly the encoder state for the generated sequence.
            self.history_gru = nn.GRU(11, hidden_size, batch_first=True)
            self.history_projection = nn.Sequential(
                nn.Linear(hidden_size, 64), nn.SiLU(),
                nn.Linear(64, latent_size), nn.Tanh())
            feature_size = 9 + 2 + latent_size + 4
            self.acceleration_net = nn.Sequential(
                nn.Linear(feature_size, 256), nn.SiLU(),
                nn.Linear(256, 256), nn.SiLU(),
                nn.Linear(256, 192), nn.SiLU(),
                nn.Linear(192, 5))
            nn.init.zeros_(self.acceleration_net[-1].weight)
            nn.init.zeros_(self.acceleration_net[-1].bias)

        def _history_row(self, state, command):
            row = torch.cat((state, command), dim=-1)
            return ((row - self.history_mean) / self.history_scale)[:, None, :]

        def _acceleration(self, state, command, hidden):
            normalized_state = (state - self.state_mean) / self.state_scale
            normalized_command = (command - self.command_mean) / self.command_scale
            context = self.history_projection(hidden[-1])

            u, v, yaw_rate = state[:, 0], state[:, 1], state[:, 2]
            half_track = REAR_TRACK_WIDTH_M / 2.0
            left_contact = u - half_track * yaw_rate
            right_contact = u + half_track * yaw_rate
            wheel_slip = torch.stack((
                state[:, 5] - left_contact,
                state[:, 6] - right_contact,
            ), dim=-1)
            rear_lateral_velocity = v + REAR_AXLE_TO_COM_M * yaw_rate
            slip_angle = torch.stack((
                torch.atan2(rear_lateral_velocity,
                            torch.abs(left_contact) + 0.20),
                torch.atan2(rear_lateral_velocity,
                            torch.abs(right_contact) + 0.20),
            ), dim=-1)
            slip_features = torch.cat((wheel_slip, slip_angle), dim=-1)
            slip_features = slip_features * slip_features.new_tensor(
                (1.0, 1.0, 5.0, 5.0))
            features = torch.cat((normalized_state, normalized_command,
                                  context, slip_features), dim=-1)
            acceleration = self.acceleration_bounds * torch.tanh(
                self.acceleration_net(features))
            return acceleration, context

        def rollout(self, initial_state, delayed_command, history, commands):
            if (initial_state.ndim != 2 or initial_state.shape[1] != 9
                    or delayed_command.shape != (len(initial_state), 2)
                    or history.shape != (len(initial_state), 80, 11)
                    or commands.ndim != 3 or commands.shape[0] != len(initial_state)
                    or commands.shape[2] != 2 or commands.shape[1] < 1):
                raise ValueError("history-consistent rollout inputs have invalid shapes")
            if not (torch.isfinite(initial_state).all()
                    and torch.isfinite(delayed_command).all()
                    and torch.isfinite(history).all()
                    and torch.isfinite(commands).all()):
                raise ValueError("history-consistent rollout inputs must be finite")

            normalized_history = ((history - self.history_mean)
                                  / self.history_scale)
            _, hidden = self.history_gru(normalized_history)
            state = initial_state
            previous_command = delayed_command
            states, accelerations, latents = [], [], []
            gates = []
            dt = DT_S
            for index in range(commands.shape[1]):
                command = commands[:, index]
                acceleration, context = self._acceleration(state, command, hidden)
                u, v, yaw_rate = state[:, 0], state[:, 1], state[:, 2]
                ax, ay, yaw_accel = acceleration[:, 0], acceleration[:, 1], acceleration[:, 2]
                u_next = u + dt * (ax + yaw_rate * v)
                v_next = v + dt * (ay - yaw_rate * u)
                yaw_rate_next = yaw_rate + dt * yaw_accel

                steering_target = (command[:, 0] if self.steering_delay_steps == 0
                                   else previous_command[:, 0])
                throttle_target = (command[:, 1] if self.throttle_delay_steps == 0
                                   else previous_command[:, 1])
                steering_next = state[:, 3] + self.steering_alpha * (
                    steering_target - state[:, 3])
                throttle_next = state[:, 4] + self.throttle_alpha * (
                    throttle_target - state[:, 4])
                wheels_next = state[:, 5:7] + dt * acceleration[:, 3:5]
                c_ay, c_roll, c_rate = self.roll_oscillator_coefficients
                roll_rate_dot = (c_ay * ay + c_roll * state[:, 7]
                                 + c_rate * state[:, 8])
                roll_rate_next = state[:, 8] + dt * roll_rate_dot
                roll_next = state[:, 7] + 0.5 * dt * (state[:, 8] + roll_rate_next)
                next_state = torch.stack((
                    u_next, v_next, yaw_rate_next, steering_next, throttle_next,
                    wheels_next[:, 0], wheels_next[:, 1], roll_next,
                    roll_rate_next), dim=-1)
                if not torch.isfinite(next_state).all():
                    raise FloatingPointError(
                        f"non-finite state at rollout step {index + 1}")
                states.append(next_state)
                accelerations.append(acceleration)
                latents.append(context)
                gates.append(torch.ones(
                    (len(state), 1), dtype=state.dtype, device=state.device))

                if index + 1 < commands.shape[1]:
                    # The generated state is paired with the next known
                    # command, just as each observed history row is.
                    row = self._history_row(next_state, commands[:, index + 1])
                    _, hidden = self.history_gru(row, hidden)
                previous_command = command
                state = next_state

            return (torch.stack(states, dim=1),
                    torch.stack(accelerations, dim=1),
                    torch.stack(latents, dim=1),
                    torch.stack(gates, dim=1))

    return ConsistentHistoryTeacher


def actuator_metadata(actuator_fit: Any) -> dict[str, Any]:
    return {
        "steering": {
            "delay_steps": int(actuator_fit.steering.delay_steps),
            "alpha": float(actuator_fit.steering.alpha),
        },
        "throttle": {
            "delay_steps": int(actuator_fit.throttle.delay_steps),
            "alpha": float(actuator_fit.throttle.alpha),
        },
        "diagnostics": actuator_fit.diagnostics,
    }
