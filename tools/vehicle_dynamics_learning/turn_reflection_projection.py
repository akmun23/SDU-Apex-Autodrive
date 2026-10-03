"""Trajectory-level left/right symmetry projection for offline evaluations."""

from __future__ import annotations

from typing import Any

import torch

from tools.vehicle_dynamics_learning.turn_reflection import (
    reflect_acceleration_targets,
    reflect_commands,
    reflect_history_features,
    reflect_state_channels,
)


class TurnReflectionProjectedModel:
    """Average a command-only rollout with its reflected counterpart.

    The two model branches recursively evolve independently; their reflected
    state and acceleration trajectories are averaged at the output. This is
    an offline physics-informed ensemble, not a change to the production plant.
    """

    def __init__(self, model: Any):
        self.model = model
        self.history_state_size = model.history_state_size
        self.include_roll_state = model.include_roll_state
        self.include_raw_encoder_history = model.include_raw_encoder_history
        self.include_wheel_innovation_history = (
            model.include_wheel_innovation_history)

    def rollout(self, initial_state, delayed_command, history, commands):
        original = self.model.rollout(
            initial_state, delayed_command, history, commands)
        reflected = self.model.rollout(
            reflect_state_channels(initial_state),
            reflect_commands(delayed_command),
            reflect_history_features(history, self.history_state_size),
            reflect_commands(commands))
        projected_state = 0.5 * (
            original[0] + reflect_state_channels(reflected[0]))
        projected_acceleration = 0.5 * (
            original[1] + reflect_acceleration_targets(reflected[1]))
        projected_gates = 0.5 * (original[3] + reflected[3])
        # Latents are branch-specific internal coordinates, not a physical
        # observable. The primary scorer ignores this slot after the rollout.
        return projected_state, projected_acceleration, original[2], projected_gates


class HighSteerStepwiseReflectionModel:
    """Apply reflection projection only to high-steering transitions.

    Unlike trajectory averaging, this projects each transition and feeds the
    projected physical state into the next step. The two causal latent states
    remain separate; on projected steps they are advanced using the projected
    acceleration and their respective reflected physical coordinates. The
    threshold is the existing 0.30 rad steering boundary used by the held-out
    signed-steering diagnosis, not a threshold selected on practice transfer.
    """

    def __init__(self, model: Any, steering_threshold_rad: float = 0.30):
        if not 0.0 < steering_threshold_rad <= 0.524:
            raise ValueError("steering threshold must be in (0, 0.524] rad")
        required = ("transition", "encode_history", "latent_transition",
                    "state_mean", "state_scale", "command_mean",
                    "command_scale", "acceleration_bounds",
                    "dynamics_state_size")
        if any(not hasattr(model, name) for name in required):
            raise TypeError("stepwise projection needs the EDSSM transition interface")
        self.model = model
        self.steering_threshold_rad = float(steering_threshold_rad)
        self.history_state_size = model.history_state_size
        self.include_roll_state = model.include_roll_state
        self.include_raw_encoder_history = model.include_raw_encoder_history
        self.include_wheel_innovation_history = (
            model.include_wheel_innovation_history)

    def _advance_latent(self, physical_state, command, acceleration, latent):
        normalized_state = (
            (physical_state - self.model.state_mean)
            / self.model.state_scale)[:, :self.model.dynamics_state_size]
        normalized_command = (
            (command - self.model.command_mean) / self.model.command_scale)
        normalized_acceleration = (
            acceleration / self.model.acceleration_bounds)
        latent_input = torch.cat((
            normalized_state, normalized_command, normalized_acceleration),
            dim=-1)
        return self.model.latent_transition(latent_input, latent)

    def rollout(self, initial_state, delayed_command, history, commands):
        model = self.model
        if commands.ndim != 3 or commands.shape[-1] != 2:
            raise ValueError("future commands must have shape (B,T,2)")
        state = initial_state
        mirror_state = reflect_state_channels(state)
        delay = delayed_command
        mirror_delay = reflect_commands(delay)
        latent = model.encode_history(history)
        mirror_latent = model.encode_history(
            reflect_history_features(history, self.history_state_size))
        states, accelerations, latents, gates = [], [], [], []
        for index in range(commands.shape[1]):
            command = commands[:, index]
            mirror_command = reflect_commands(command)
            current_state = state
            current_mirror_state = mirror_state
            direct = model.transition(state, delay, latent, command)
            mirrored = model.transition(
                mirror_state, mirror_delay, mirror_latent, mirror_command)
            reflected_state = reflect_state_channels(mirrored[0])
            reflected_acceleration = reflect_acceleration_targets(mirrored[3])
            projected_state = 0.5 * (direct[0] + reflected_state)
            projected_acceleration = 0.5 * (
                direct[3] + reflected_acceleration)
            projected_gates = 0.5 * (direct[4] + mirrored[4])
            active = ((state[:, 3].abs() >= self.steering_threshold_rad)
                      | (command[:, 0].abs()
                         >= self.steering_threshold_rad))
            state = torch.where(active[:, None], projected_state, direct[0])
            acceleration = torch.where(
                active[:, None], projected_acceleration, direct[3])
            latent_projected = self._advance_latent(
                physical_state=current_state,
                command=command, acceleration=acceleration, latent=latent)
            mirror_acceleration = reflect_acceleration_targets(acceleration)
            mirror_latent_projected = self._advance_latent(
                physical_state=current_mirror_state, command=mirror_command,
                acceleration=mirror_acceleration, latent=mirror_latent)
            latent = torch.where(active[:, None], latent_projected, direct[2])
            mirror_latent = torch.where(
                active[:, None], mirror_latent_projected, mirrored[2])
            mirror_state = reflect_state_channels(state)
            delay = command
            mirror_delay = mirror_command
            states.append(state)
            accelerations.append(acceleration)
            latents.append(latent)
            gates.append(torch.where(active[:, None], projected_gates,
                                     direct[4]))
        if not states:
            raise ValueError("stepwise reflection rollout requires a command")
        return (torch.stack(states, dim=1),
                torch.stack(accelerations, dim=1),
                torch.stack(latents, dim=1),
                torch.stack(gates, dim=1))
