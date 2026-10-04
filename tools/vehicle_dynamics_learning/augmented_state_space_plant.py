#!/usr/bin/env python3
"""Causal latent state-space plant structure for black-box AutoDRIVE modeling.

This module defines the WP21 model and API only.  It does not alter the
production simulator, odometry, localization, MPC, or actuator path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree


DT_S = 0.025
WHEEL_RADIUS_M = 0.0325
HISTORY_STEPS = 80
HISTORY_FEATURE_NAMES = (
    "odom_u_mps", "odom_v_mps", "odom_yaw_rate_rps",
    "steering_feedback_rad", "throttle_feedback_norm",
    "steering_command_rad", "throttle_command_norm",
    "left_encoder_angle_increment_rad",
    "right_encoder_angle_increment_rad", "encoder_valid",
)
SUPPORT_FEATURE_NAMES = (
    "speed_mps", "steering_feedback_rad", "steering_rate_rps",
    "throttle_feedback_norm", "throttle_slew_per_s", "yaw_rate_rps",
    "abs_filtered_wheel_body_mismatch_mps",
)


def _as_tuple(values: Any, size: int, name: str, positive: bool = False
              ) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


@dataclass(frozen=True)
class PlantConfig:
    """Serializable, train-split-only model and support configuration."""

    history_steps: int = HISTORY_STEPS
    history_feature_names: tuple[str, ...] = HISTORY_FEATURE_NAMES
    latent_size: int = 8
    latent_enabled: bool = True
    latent_measurement_feedback: bool = True
    body_transition_mode: str = "anchored_increment"
    dt_s: float = DT_S
    wheel_radius_m: float = WHEEL_RADIUS_M
    history_mean: tuple[float, ...] = (0.0,) * 10
    history_scale: tuple[float, ...] = (1.0,) * 10
    state_mean: tuple[float, ...] = (0.0,) * 5
    state_scale: tuple[float, ...] = (1.0,) * 5
    command_mean: tuple[float, ...] = (0.0, 0.0)
    command_scale: tuple[float, ...] = (1.0, 1.0)
    body_increment_limit: tuple[float, ...] = (0.25, 0.10, 0.20)
    body_state_normalized_limit: tuple[float, ...] = (3.0, 3.0, 3.0)
    encoder_increment_limit: tuple[float, ...] = (0.25, 0.25)
    steering_delay_steps: int = 0
    throttle_delay_steps: int = 0
    steering_alpha: float = 1.0
    throttle_alpha: float = 1.0
    support_mean: tuple[float, ...] = (0.0,) * 7
    support_scale: tuple[float, ...] = (1.0,) * 7
    support_bank: tuple[tuple[float, ...], ...] = ()
    support_run_index: tuple[int, ...] = ()
    support_run_ids: tuple[str, ...] = ()
    support_supported_upper: float = 0.0417017919022596
    support_weak_upper: float = 0.16416021114474064
    support_calibrated: bool = False
    support_calibration_scope: str = ""

    def __post_init__(self) -> None:
        expected_latent = 8 if self.latent_enabled else 0
        if self.history_steps < 1 or self.latent_size != expected_latent:
            raise ValueError("enabled latent state requires size 8; disabled requires size 0")
        if not isinstance(self.latent_measurement_feedback, bool):
            raise ValueError("latent_measurement_feedback must be boolean")
        if self.body_transition_mode not in ("anchored_increment", "direct_state"):
            raise ValueError("body transition mode must be anchored_increment or direct_state")
        if tuple(self.history_feature_names) != HISTORY_FEATURE_NAMES:
            raise ValueError("history feature ordering does not match WP21")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not np.isfinite(self.wheel_radius_m) or self.wheel_radius_m <= 0:
            raise ValueError("wheel radius must be finite and positive")
        for name, size in (("history_mean", 10), ("history_scale", 10),
                           ("state_mean", 5), ("state_scale", 5),
                           ("command_mean", 2), ("command_scale", 2),
                           ("body_increment_limit", 3),
                           ("body_state_normalized_limit", 3),
                           ("encoder_increment_limit", 2),
                           ("support_mean", 7), ("support_scale", 7)):
            _as_tuple(getattr(self, name), size, name,
                      positive=name.endswith("scale") or name.endswith("limit"))
        if self.steering_delay_steps not in (0, 1) or \
                self.throttle_delay_steps not in (0, 1):
            raise ValueError("actuator delays must be zero or one 25 ms sample")
        for alpha in (self.steering_alpha, self.throttle_alpha):
            if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
                raise ValueError("actuator alpha must be finite and in [0,1]")
        if (not np.isfinite(self.support_supported_upper)
                or not np.isfinite(self.support_weak_upper)
                or self.support_supported_upper < 0.0
                or self.support_weak_upper <= self.support_supported_upper):
            raise ValueError("support thresholds must be finite and ordered")
        bank = np.asarray(self.support_bank, dtype=np.float64)
        run_index = np.asarray(self.support_run_index, dtype=np.int64)
        if bank.size:
            if bank.ndim != 2 or bank.shape[1] != 7 or not np.isfinite(bank).all():
                raise ValueError("support bank must be finite N x 7 normalized data")
            if run_index.shape != (len(bank),):
                raise ValueError("support run indexes must align with bank rows")
            if (not self.support_run_ids or np.any(run_index < 0)
                    or np.any(run_index >= len(self.support_run_ids))):
                raise ValueError("support bank run indexes are invalid")
        elif len(run_index):
            raise ValueError("support run indexes require a nonempty bank")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PlantConfig":
        tuple_fields = {
            "history_feature_names", "history_mean", "history_scale",
            "state_mean", "state_scale", "command_mean", "command_scale",
            "body_increment_limit", "encoder_increment_limit", "support_mean",
            "body_state_normalized_limit", "support_scale", "support_run_index",
            "support_run_ids",
        }
        converted = dict(raw)
        for key in tuple_fields:
            if key in converted:
                converted[key] = tuple(converted[key])
        if "support_bank" in converted:
            converted["support_bank"] = tuple(
                tuple(row) for row in converted["support_bank"])
        return cls(**converted)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoryStateEncoder(nn.Module):
    """GRU that initializes latent state from past-only sensor/command history."""

    def __init__(self, input_size: int = 10, hidden_size: int = 32,
                 latent_size: int = 8) -> None:
        super().__init__()
        self.input_size = input_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.project = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                     nn.SiLU(), nn.Linear(hidden_size, latent_size),
                                     nn.Tanh())

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1] < 1 \
                or history.shape[2] != self.input_size:
            raise ValueError(
                f"history must have shape (batch, nonempty, {self.input_size})")
        if not torch.isfinite(history).all():
            raise ValueError("history must be finite")
        _, hidden = self.gru(history)
        return self.project(hidden[-1])


class ActuatorTransition(nn.Module):
    """Fixed train-identified first-order actuator response with 0/1-step delay."""

    def __init__(self, steering_alpha: float, throttle_alpha: float,
                 steering_delay_steps: int, throttle_delay_steps: int) -> None:
        super().__init__()
        if steering_delay_steps not in (0, 1) or throttle_delay_steps not in (0, 1):
            raise ValueError("actuator delay must be zero or one sample")
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0
               for value in (steering_alpha, throttle_alpha)):
            raise ValueError("actuator response alpha must be in [0,1]")
        self.register_buffer("alpha", torch.tensor(
            [steering_alpha, throttle_alpha], dtype=torch.float32))
        self.register_buffer("delay", torch.tensor(
            [steering_delay_steps, throttle_delay_steps], dtype=torch.bool))

    def forward(self, feedback: torch.Tensor, command: torch.Tensor,
                previous_command: torch.Tensor) -> torch.Tensor:
        if feedback.shape[-1] != 2 or command.shape != feedback.shape \
                or previous_command.shape != feedback.shape:
            raise ValueError("actuator tensors must have matching (...,2) shapes")
        target = torch.where(self.delay, previous_command, command)
        return feedback + self.alpha * (target - feedback)


class StableBodyNominal(nn.Module):
    """Neutral, bounded nominal: hold body velocity over one 25 ms transition.

    The explicit SE(2) integration is handled by PoseIntegrator.  No guessed
    wheelbase, tire parameter, or four-wheel force law is embedded here.
    """

    def forward(self, body_state: torch.Tensor) -> torch.Tensor:
        if body_state.shape[-1] != 3 or not torch.isfinite(body_state).all():
            raise ValueError("body state must be finite (...,3)")
        return body_state


class LatentResidualTransition(nn.Module):
    """Bounded learned body-transition output for the selected WP23 variant."""

    def __init__(self, feature_size: int, body_increment_limit: tuple[float, ...],
                 hidden_size: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.SiLU(),
                                   nn.Linear(hidden_size, hidden_size), nn.SiLU())
        self.head = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("limit", torch.as_tensor(
            body_increment_limit, dtype=torch.float32))

    def forward(self, normalized_features: torch.Tensor) -> torch.Tensor:
        return self.limit * torch.tanh(self.head(self.trunk(normalized_features)))


class LatentStateTransition(nn.Module):
    """One-step causal latent update; its output is the next internal state."""

    def __init__(self, input_size: int, latent_size: int = 8) -> None:
        super().__init__()
        self.cell = nn.GRUCell(input_size, latent_size)

    def forward(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or latent.ndim != 2 or len(features) != len(latent):
            raise ValueError("latent transition expects aligned rank-2 batches")
        return self.cell(features, latent)


class MeasurementHead(nn.Module):
    """Generates left/right rear encoder angle increments, in radians/sample."""

    def __init__(self, feature_size: int,
                 encoder_increment_limit: tuple[float, ...],
                 hidden_size: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("limit", torch.as_tensor(
            encoder_increment_limit, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.limit * torch.tanh(self.net(features))


class PoseIntegrator(nn.Module):
    """Exact constant body-twist SE(2) integration for rear-axle pose."""

    def __init__(self, dt_s: float = DT_S) -> None:
        super().__init__()
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("pose integration period must be positive")
        self.dt_s = float(dt_s)

    def forward(self, pose: torch.Tensor, body_state: torch.Tensor) -> torch.Tensor:
        if pose.shape[-1] != 3 or body_state.shape[-1] != 3 \
                or pose.shape[:-1] != body_state.shape[:-1]:
            raise ValueError("pose/body arrays must have matching (...,3) shapes")
        if not torch.isfinite(pose).all() or not torch.isfinite(body_state).all():
            raise ValueError("pose integration inputs must be finite")
        u, v, r = body_state.unbind(dim=-1)
        dtheta = r * self.dt_s
        safe_r = torch.where(r.abs() > 1e-7, r, torch.ones_like(r))
        dx_turn = (u * torch.sin(dtheta) + v * (torch.cos(dtheta) - 1.0)) \
            / safe_r
        dy_turn = (u * (1.0 - torch.cos(dtheta)) + v * torch.sin(dtheta)) \
            / safe_r
        dx = torch.where(r.abs() > 1e-7, dx_turn, u * self.dt_s)
        dy = torch.where(r.abs() > 1e-7, dy_turn, v * self.dt_s)
        x, y, heading = pose.unbind(dim=-1)
        world_x = x + torch.cos(heading) * dx - torch.sin(heading) * dy
        world_y = y + torch.sin(heading) * dx + torch.cos(heading) * dy
        heading_next = torch.atan2(torch.sin(heading + dtheta),
                                   torch.cos(heading + dtheta))
        return torch.stack((world_x, world_y, heading_next), dim=-1)


class SupportEstimatorAdapter(nn.Module):
    """Frozen run-balanced kNN support score calibrated in WP20.

    Confidence is only calibrated against WP20's composite 2 s body error;
    it is not a per-channel error bound or proof of plant accuracy.
    """

    def __init__(self, mean: tuple[float, ...], scale: tuple[float, ...],
                 bank: tuple[tuple[float, ...], ...],
                 run_index: tuple[int, ...], run_ids: tuple[str, ...],
                 supported_upper: float, weak_upper: float,
                 calibrated: bool) -> None:
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("bank", torch.as_tensor(bank, dtype=torch.float32)
                             .reshape(-1, 7))
        self.register_buffer("run_index", torch.as_tensor(run_index,
                                                           dtype=torch.long))
        self.run_ids = tuple(run_ids)
        self.supported_upper = float(supported_upper)
        self.weak_upper = float(weak_upper)
        self.calibrated = bool(calibrated and len(run_ids) >= 3 and len(bank) > 0)
        self._trees: list[cKDTree] | None = None

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.shape[-1] != 7:
            raise ValueError("support feature vector must end in seven channels")
        original_shape = features.shape[:-1]
        flat = features.reshape(-1, 7)
        if not torch.isfinite(flat).all():
            raise ValueError("support features must be finite")
        if self.bank.shape[0] == 0:
            score = torch.full((len(flat),), self.weak_upper + 1.0,
                               dtype=flat.dtype, device=flat.device)
        else:
            query_np = ((flat.detach() - self.mean) / self.scale).cpu().numpy()
            if self._trees is None:
                bank_np = self.bank.detach().cpu().numpy()
                run_np = self.run_index.detach().cpu().numpy()
                self._trees = [cKDTree(bank_np[run_np == run])
                               for run in range(len(self.run_ids))
                               if np.any(run_np == run)]
            if len(self._trees) < 3:
                score = torch.full((len(flat),), self.weak_upper + 1.0,
                                   dtype=flat.dtype, device=flat.device)
            else:
                distances = np.column_stack([
                    tree.query(query_np, k=1, workers=1)[0]
                    for tree in self._trees])
                scores_np = np.partition(distances, 2, axis=1)[:, :3].mean(axis=1)
                score = torch.as_tensor(scores_np, dtype=flat.dtype,
                                        device=flat.device)
        span = max(self.weak_upper - self.supported_upper, 1e-12)
        confidence = torch.where(
            score <= self.supported_upper, torch.ones_like(score),
            torch.where(score < self.weak_upper,
                        (self.weak_upper - score) / span,
                        torch.zeros_like(score)))
        if not self.calibrated:
            confidence = torch.zeros_like(confidence)
        confidence = confidence.clamp(0.0, 1.0)
        return {"score": score.reshape(original_shape),
                "confidence": confidence.reshape(original_shape)}


class AugmentedStateSpacePlant(nn.Module):
    """Batch-capable plant with stateful reset/step/rollout convenience API."""

    def __init__(self, config: PlantConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("history_mean", torch.tensor(config.history_mean))
        self.register_buffer("history_scale", torch.tensor(config.history_scale))
        self.register_buffer("state_mean", torch.tensor(config.state_mean))
        self.register_buffer("state_scale", torch.tensor(config.state_scale))
        self.register_buffer("command_mean", torch.tensor(config.command_mean))
        self.register_buffer("command_scale", torch.tensor(config.command_scale))
        self.history_encoder = (HistoryStateEncoder(latent_size=config.latent_size)
                                if config.latent_enabled else None)
        self.actuator_transition = ActuatorTransition(
            config.steering_alpha, config.throttle_alpha,
            config.steering_delay_steps, config.throttle_delay_steps)
        self.body_nominal = StableBodyNominal()
        # Features: normalized observable state, latent, normalized command.
        latent_input_size = config.latent_size if config.latent_enabled else 0
        residual_limit = (config.body_state_normalized_limit
                          if config.body_transition_mode == "direct_state"
                          else config.body_increment_limit)
        self.body_residual = LatentResidualTransition(
            5 + latent_input_size + 2, residual_limit)
        # Inputs: next state, command, body increment and generated encoder output.
        latent_measurement_size = 2 if config.latent_measurement_feedback else 0
        self.latent_transition = (LatentStateTransition(
            5 + 2 + 3 + latent_measurement_size, config.latent_size)
            if config.latent_enabled else None)
        # Inputs: state(k), state(k+1), command(k), latent(k), Delta body.
        self.measurement_head = MeasurementHead(
            5 + 5 + 2 + latent_input_size + 3,
            config.encoder_increment_limit)
        self.pose_integrator = PoseIntegrator(config.dt_s)
        self.support_estimator = SupportEstimatorAdapter(
            config.support_mean, config.support_scale, config.support_bank,
            config.support_run_index, config.support_run_ids,
            config.support_supported_upper, config.support_weak_upper,
            config.support_calibrated)
        self._state: torch.Tensor | None = None
        self._pose: torch.Tensor | None = None
        self._latent: torch.Tensor | None = None
        self._previous_command: torch.Tensor | None = None
        self._encoder_history: torch.Tensor | None = None
        self._encoder_valid_history: torch.Tensor | None = None
        self._measurement: torch.Tensor | None = None
        self._body_increment: torch.Tensor | None = None
        self._support: dict[str, torch.Tensor] | None = None
        self._single = False

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        return self.state_mean.device, self.state_mean.dtype

    def reset(self, history: torch.Tensor | np.ndarray,
              initial_state: torch.Tensor | np.ndarray,
              initial_pose: torch.Tensor | np.ndarray) -> None:
        device, dtype = self._device_dtype()
        history = torch.as_tensor(history, dtype=dtype, device=device)
        state = torch.as_tensor(initial_state, dtype=dtype, device=device)
        pose = torch.as_tensor(initial_pose, dtype=dtype, device=device)
        self._single = history.ndim == 2
        if self._single:
            history, state, pose = history[None], state[None], pose[None]
        if (history.ndim != 3 or history.shape[1:] !=
                (self.config.history_steps, len(HISTORY_FEATURE_NAMES))):
            raise ValueError("reset history must match configured 80 x 10 layout")
        if state.shape != (len(history), 5) or pose.shape != (len(history), 3):
            raise ValueError("initial state/pose must align with history batch")
        if not torch.isfinite(history).all() or not torch.isfinite(state).all() \
                or not torch.isfinite(pose).all():
            raise ValueError("reset inputs must be finite")
        self._state = state
        self._pose = pose
        normalized_history = (history - self.history_mean) / self.history_scale
        self._latent = (self.history_encoder(normalized_history)
                        if self.history_encoder is not None else
                        state.new_empty((len(state), 0)))
        self._previous_command = history[:, -2, 5:7] if len(history[0]) > 1 \
            else history[:, -1, 5:7]
        self._encoder_history = history[:, -4:, 7:9]
        self._encoder_valid_history = history[:, -4:, 9] > 0.5
        self._measurement = history[:, -1, 7:9]
        self._body_increment = torch.zeros_like(state[:, :3])
        self._support = self._estimate_support(
            self._state, self._measurement, self._encoder_history,
            self._encoder_valid_history,
            history[:, -2, 3:5] if history.shape[1] > 1 else history[:, -1, 3:5])

    def _require_reset(self) -> None:
        if self._state is None or self._pose is None or self._latent is None \
                or self._previous_command is None or self._encoder_history is None \
                or self._encoder_valid_history is None:
            raise RuntimeError("call reset(history, initial_state, initial_pose) first")

    def _estimate_support(self, state: torch.Tensor, measurement: torch.Tensor,
                          encoder_history: torch.Tensor,
                          encoder_valid_history: torch.Tensor,
                          previous_actuator: torch.Tensor
                          ) -> dict[str, torch.Tensor]:
        if encoder_history.shape[1] < 4:
            pad = encoder_history[:, :1].expand(-1, 4-encoder_history.shape[1], -1)
            encoder_history = torch.cat((pad, encoder_history), dim=1)
        wheel_rates = encoder_history * (self.config.wheel_radius_m / self.config.dt_s)
        filtered = wheel_rates.mean(dim=1)
        feature = torch.stack((
            torch.linalg.vector_norm(state[:, :2], dim=1),
            state[:, 3], (state[:, 3] - previous_actuator[:, 0]) / self.config.dt_s,
            state[:, 4], (state[:, 4] - previous_actuator[:, 1]) / self.config.dt_s,
            state[:, 2], torch.abs(filtered.mean(dim=1) - state[:, 0]),
        ), dim=1)
        result = self.support_estimator(feature)
        # A history with invalid encoder samples cannot claim support from a
        # mismatch feature reconstructed using placeholder zero increments.
        enough_encoder_history = encoder_valid_history.all(dim=1)
        result["confidence"] = torch.where(
            enough_encoder_history, result["confidence"],
            torch.zeros_like(result["confidence"]))
        return result

    def step(self, command: torch.Tensor | np.ndarray
             ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Advance one 25 ms step from the current predicted state and command."""
        self._require_reset()
        assert self._state is not None and self._pose is not None
        assert self._latent is not None and self._previous_command is not None
        assert self._encoder_history is not None and self._encoder_valid_history is not None
        device, dtype = self._device_dtype()
        command = torch.as_tensor(command, dtype=dtype, device=device)
        if command.ndim == 1:
            command = command[None]
        if command.shape != (len(self._state), 2) or not torch.isfinite(command).all():
            raise ValueError("command must be finite with shape (batch,2)")
        state, latent = self._state, self._latent
        body = state[:, :3]
        nominal = (self.body_nominal(body)
                   if self.config.body_transition_mode == "anchored_increment"
                   else torch.zeros_like(body))
        normalized_state = (state - self.state_mean) / self.state_scale
        normalized_command = (command - self.command_mean) / self.command_scale
        residual_features = torch.cat((normalized_state, latent,
                                       normalized_command), dim=1)
        proposal = self.body_residual(residual_features)
        if self.config.body_transition_mode == "anchored_increment":
            body_next = nominal + proposal
        else:
            body_next = (self.state_mean[:3]
                         + self.state_scale[:3] * proposal)
        transition_delta = body_next - body
        actuator_next = self.actuator_transition(
            state[:, 3:5], command, self._previous_command)
        state_next = torch.cat((body_next, actuator_next), dim=1)
        measurement_features = torch.cat((
            normalized_state, (state_next - self.state_mean) / self.state_scale,
            normalized_command, latent, transition_delta), dim=1)
        measurement = self.measurement_head(measurement_features)
        latent_feature_parts = [
            (state_next - self.state_mean) / self.state_scale,
            normalized_command, transition_delta]
        if self.config.latent_measurement_feedback:
            latent_feature_parts.append(measurement)
        latent_features = torch.cat(latent_feature_parts, dim=1)
        latent_next = (self.latent_transition(latent_features, latent)
                       if self.latent_transition is not None else latent)
        pose_next = self.pose_integrator(self._pose, body)
        encoder_history = torch.cat((self._encoder_history[:, 1:],
                                     measurement[:, None, :]), dim=1)
        encoder_valid_history = torch.cat((self._encoder_valid_history[:, 1:],
                                           torch.ones_like(
                                               self._encoder_valid_history[:, :1])),
                                          dim=1)
        support = self._estimate_support(state_next, measurement,
                                         encoder_history, encoder_valid_history,
                                         state[:, 3:5])
        self._state, self._pose, self._latent = state_next, pose_next, latent_next
        self._previous_command = command
        self._encoder_history = encoder_history
        self._encoder_valid_history = encoder_valid_history
        self._measurement = measurement
        self._body_increment = body_next - body
        self._support = support
        if self._single:
            return state_next[0], measurement[0], {
                key: value[0] for key, value in support.items()}
        return state_next, measurement, support

    def rollout(self, commands: torch.Tensor | np.ndarray
                ) -> dict[str, torch.Tensor]:
        self._require_reset()
        commands = torch.as_tensor(commands, dtype=self.state_mean.dtype,
                                   device=self.state_mean.device)
        if commands.ndim == 2:
            commands = commands[None]
        if commands.ndim != 3 or commands.shape[0] != len(self._state) \
                or commands.shape[2] != 2 or commands.shape[1] == 0:
            raise ValueError("commands must have nonempty shape (batch,steps,2)")
        if not torch.isfinite(commands).all():
            raise ValueError("commands must be finite")
        states, poses, measurements = [], [], []
        body_increments, latents, support_scores, support_confidence = [], [], [], []
        for index in range(commands.shape[1]):
            state, measurement, support = self.step(commands[:, index])
            if self._single:
                states.append(state)
                measurements.append(measurement)
                assert self._latent is not None and self._body_increment is not None
                latents.append(self._latent[0])
                body_increments.append(self._body_increment[0])
                support_scores.append(support["score"])
                support_confidence.append(support["confidence"])
                assert self._pose is not None
                poses.append(self._pose[0])
            else:
                states.append(state)
                measurements.append(measurement)
                assert self._latent is not None and self._body_increment is not None
                latents.append(self._latent)
                body_increments.append(self._body_increment)
                support_scores.append(support["score"])
                support_confidence.append(support["confidence"])
                assert self._pose is not None
                poses.append(self._pose)
        result = {
            "states": torch.stack(states, dim=0 if self._single else 1),
            "poses": torch.stack(poses, dim=0 if self._single else 1),
            "measurements": torch.stack(measurements, dim=0 if self._single else 1),
            "body_increments": torch.stack(body_increments,
                                            dim=0 if self._single else 1),
            "latents": torch.stack(latents, dim=0 if self._single else 1),
            "support_score": torch.stack(support_scores,
                                         dim=0 if self._single else 1),
            "support_confidence": torch.stack(support_confidence,
                                              dim=0 if self._single else 1),
        }
        return result

    def get_state(self) -> torch.Tensor:
        self._require_reset()
        assert self._state is not None
        return self._state[0] if self._single else self._state

    def get_measurement(self) -> torch.Tensor:
        self._require_reset()
        assert self._measurement is not None
        return self._measurement[0] if self._single else self._measurement

    def get_support(self) -> dict[str, torch.Tensor]:
        self._require_reset()
        assert self._support is not None
        return ({key: value[0] for key, value in self._support.items()}
                if self._single else self._support)


def mirror_state(state: torch.Tensor) -> torch.Tensor:
    """Reflect lateral/yaw/steering signs; forward quantities are unchanged."""
    if state.shape[-1] != 5:
        raise ValueError("observable state must end in five channels")
    sign = state.new_tensor([1.0, -1.0, -1.0, -1.0, 1.0])
    return state * sign


def mirror_command(command: torch.Tensor) -> torch.Tensor:
    if command.shape[-1] != 2:
        raise ValueError("command must end in steering/throttle")
    return command * command.new_tensor([-1.0, 1.0])


def mirror_history(history: torch.Tensor) -> torch.Tensor:
    if history.shape[-1] != 10:
        raise ValueError("WP21 history must end in ten channels")
    sign = history.new_tensor([1.0, -1.0, -1.0, -1.0, 1.0,
                               -1.0, 1.0, 1.0, 1.0, 1.0])
    reflected = history * sign
    return torch.cat((reflected[..., :7], reflected[..., 8:9],
                      reflected[..., 7:8], reflected[..., 9:10]), dim=-1)


def mirror_measurement(measurement: torch.Tensor) -> torch.Tensor:
    if measurement.shape[-1] != 2:
        raise ValueError("encoder measurement must end in left/right channels")
    return measurement.flip(dims=(-1,))


def mirror_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 3:
        raise ValueError("pose must end in x/y/heading")
    return pose * pose.new_tensor([1.0, -1.0, -1.0])


def symmetry_equivariance_loss(state: torch.Tensor,
                              mirrored_state_prediction: torch.Tensor,
                              measurement: torch.Tensor,
                              mirrored_measurement_prediction: torch.Tensor
                              ) -> torch.Tensor:
    """Optional small equivariance penalty; never projects a trajectory."""
    if state.shape != mirrored_state_prediction.shape \
            or measurement.shape != mirrored_measurement_prediction.shape:
        raise ValueError("reflection predictions must match original shapes")
    return (torch.nn.functional.smooth_l1_loss(
        mirrored_state_prediction, mirror_state(state))
        + torch.nn.functional.smooth_l1_loss(
            mirrored_measurement_prediction, mirror_measurement(measurement)))


def encoder_increment_to_rate(increment_rad: torch.Tensor,
                              dt_s: float = DT_S,
                              wheel_radius_m: float = WHEEL_RADIUS_M
                              ) -> torch.Tensor:
    if dt_s <= 0.0 or wheel_radius_m <= 0.0 or increment_rad.shape[-1] != 2:
        raise ValueError("encoder conversion requires (...,2) and positive constants")
    return increment_rad * (wheel_radius_m / dt_s)


def encoder_rate_to_increment(rate_mps: torch.Tensor,
                              dt_s: float = DT_S,
                              wheel_radius_m: float = WHEEL_RADIUS_M
                              ) -> torch.Tensor:
    if dt_s <= 0.0 or wheel_radius_m <= 0.0 or rate_mps.shape[-1] != 2:
        raise ValueError("encoder conversion requires (...,2) and positive constants")
    return rate_mps * (dt_s / wheel_radius_m)


def save_checkpoint(path: str | Path, model: AugmentedStateSpacePlant,
                    extra: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "sdu_apex_augmented_state_space_wp21_v1",
                "config": model.config.to_dict(),
                "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"
                    ) -> tuple[AugmentedStateSpacePlant, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("format") != "sdu_apex_augmented_state_space_wp21_v1":
        raise ValueError("not a WP21 augmented state-space checkpoint")
    config = PlantConfig.from_dict(checkpoint["config"])
    model = AugmentedStateSpacePlant(config).to(map_location)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint.get("extra", {})
