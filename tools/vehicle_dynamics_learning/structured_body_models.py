#!/usr/bin/env python3
"""Physics-structured body-transition models for offline identification.

All neural models predict aggregate effective body accelerations.  The known
body-frame transport terms are integrated explicitly; this is not a model of
individual tire forces and is never used by the competition runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BODY_CHANNELS = 3
BODY_NAMES = ("u_mps", "v_mps", "yaw_rate_rps")
BODY_HORIZONS = (1, 5, 10, 20, 30)
BODY_HORIZON_SECONDS = (0.025, 0.125, 0.250, 0.500, 0.750)
DT_REFERENCE_S = 0.025


@dataclass(frozen=True)
class AccelerationStatistics:
    mean: np.ndarray
    scale: np.ndarray
    sample_count: int


def generalized_accelerations(
    current: np.ndarray, following: np.ndarray, dt_s: np.ndarray
) -> np.ndarray:
    """Convert measured body-state differences into effective accelerations.

    With x-forward/y-left and positive yaw, body-frame kinematics give
    u_dot = a_x + r*v and v_dot = a_y - r*u.  The learned channels are therefore
    a_x_eff = u_dot - r*v, a_y_eff = v_dot + r*u, and alpha_z = r_dot.
    They aggregate all contact and actuator effects; they are not tire forces.
    """
    current = np.asarray(current, dtype=np.float64)
    following = np.asarray(following, dtype=np.float64)
    dt_s = np.asarray(dt_s, dtype=np.float64)
    if current.ndim != 2 or following.shape != current.shape:
        raise ValueError("current/following states must be equally shaped matrices")
    if current.shape[1] < BODY_CHANNELS or dt_s.shape != (len(current),):
        raise ValueError("body-state or dt shape is invalid")
    if not np.isfinite(current[:, :3]).all() or not np.isfinite(following[:, :3]).all():
        raise ValueError("body-state data must be finite")
    if not np.isfinite(dt_s).all() or np.any(dt_s <= 0.0):
        raise ValueError("sample intervals must be finite and positive")
    derivative = (following[:, :3] - current[:, :3]) / dt_s[:, None]
    u, v, r = current[:, 0], current[:, 1], current[:, 2]
    return np.column_stack((derivative[:, 0] - r * v,
                            derivative[:, 1] + r * u,
                            derivative[:, 2])).astype(np.float32)


def training_transition_rows(
    data: dict[str, Any], train_run_indices: set[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return current features, generalized acceleration targets, and dt.

    Only transitions from the supplied whole-run training IDs are included.
    A sequence boundary is always respected; no finite differences cross it.
    """
    feature_rows: list[np.ndarray] = []
    acceleration_rows: list[np.ndarray] = []
    interval_rows: list[np.ndarray] = []
    frames = data["frames"]
    dts = data["dt_s"]
    for (start, end), run in zip(data["bounds"], data["seq_run"]):
        if int(run) not in train_run_indices or int(end - start) < 2:
            continue
        start, end = int(start), int(end)
        dt = dts[start + 1:end].astype(np.float64, copy=False)
        valid = np.isfinite(dt) & (dt > 0.0) & (dt <= 0.25)
        if not np.any(valid):
            continue
        current = frames[start:end - 1][valid]
        following = frames[start + 1:end][valid]
        dt = dt[valid]
        feature_rows.append(current.astype(np.float32, copy=False))
        acceleration_rows.append(generalized_accelerations(current, following, dt))
        interval_rows.append(dt.astype(np.float32, copy=False))
    if not feature_rows:
        raise ValueError("training runs contain no valid adjacent body-state samples")
    return (np.concatenate(feature_rows), np.concatenate(acceleration_rows),
            np.concatenate(interval_rows))


def acceleration_statistics(targets: np.ndarray) -> AccelerationStatistics:
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != BODY_CHANNELS or not len(targets):
        raise ValueError("acceleration targets must have shape [N,3]")
    mean = np.mean(targets, axis=0)
    # Floors keep nearly constant channels numerically trainable without
    # pretending that sub-resolution variation is informative.
    scale = np.maximum(np.std(targets, axis=0), np.asarray([0.50, 0.50, 1.00]))
    return AccelerationStatistics(mean.astype(np.float32), scale.astype(np.float32),
                                  int(len(targets)))


def fit_linear_acceleration_baseline(
    features: np.ndarray,
    dt_s: np.ndarray,
    targets: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    acceleration_mean: np.ndarray,
    acceleration_scale: np.ndarray,
    ridge: float = 1.0e-2,
) -> np.ndarray:
    """Fit a fold-local regularized linear acceleration prior.

    The known rigid-body coupling is integrated separately.  This regression
    is only a low-capacity empirical generalized-acceleration baseline; the
    residual network can represent the remaining nonlinear response.
    """
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")
    normalized = (features - feature_mean[None, :]) / feature_scale[None, :]
    dt_code = (dt_s / DT_REFERENCE_S - 1.0)[:, None]
    design = np.column_stack((normalized, dt_code,
                              np.ones(len(normalized), dtype=np.float32)))
    response = (targets - acceleration_mean[None, :]) / acceleration_scale[None, :]
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, design.T @ response)
    return coefficients.astype(np.float32)


def make_structured_model(
    torch,
    nn,
    architecture: str,
    latent_dim: int,
    feature_count: int,
    hidden_size: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    acceleration_mean: np.ndarray,
    acceleration_scale: np.ndarray,
    linear_coefficients: np.ndarray | None = None,
):
    """Construct an acceleration model with registered fold-local scalers."""
    if architecture not in ("accel_mlp", "latent", "linear_residual"):
        raise ValueError(f"unsupported structured architecture: {architecture}")
    if architecture == "latent" and latent_dim not in (2, 4, 8):
        raise ValueError("latent dimension must be 2, 4, or 8")
    if architecture != "latent" and latent_dim != 0:
        raise ValueError("only the latent architecture accepts a latent state")
    if architecture == "linear_residual" and linear_coefficients is None:
        raise ValueError("linear-residual architecture needs a train-fold baseline")

    context_size = feature_count + 1  # normalized features plus relative dt

    class StructuredBodyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.architecture = architecture
            self.latent_dim = latent_dim
            self.feature_count = feature_count
            self.encoder = None
            self.latent_projection = None
            self.latent_transition = None
            if latent_dim:
                self.encoder = nn.GRU(feature_count, hidden_size,
                                      batch_first=True)
                self.latent_projection = nn.Sequential(
                    nn.Linear(hidden_size, latent_dim), nn.Tanh())
                self.latent_transition = nn.GRUCell(context_size, latent_dim)
            head_input = context_size + latent_dim
            self.acceleration_head = nn.Sequential(
                nn.Linear(head_input, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, BODY_CHANNELS))
            if linear_coefficients is None:
                coefficients = np.zeros((context_size + 1, BODY_CHANNELS),
                                        dtype=np.float32)
            else:
                coefficients = np.asarray(linear_coefficients, dtype=np.float32)
                if coefficients.shape != (context_size + 1, BODY_CHANNELS):
                    raise ValueError("linear baseline coefficient shape is invalid")
            self.register_buffer("linear_coefficients",
                                 torch.as_tensor(coefficients))
            self.register_buffer("feature_mean",
                                 torch.as_tensor(feature_mean, dtype=torch.float32))
            self.register_buffer("feature_scale",
                                 torch.as_tensor(feature_scale, dtype=torch.float32))
            self.register_buffer("acceleration_mean",
                                 torch.as_tensor(acceleration_mean,
                                                 dtype=torch.float32))
            self.register_buffer("acceleration_scale",
                                 torch.as_tensor(acceleration_scale,
                                                 dtype=torch.float32))

        def initial_latent(self, history):
            if not self.latent_dim:
                return None
            _, hidden = self.encoder(history)
            return self.latent_projection(hidden[-1])

        def predict_acceleration(self, feature, latent, dt_s):
            dt_code = dt_s[:, None] / DT_REFERENCE_S - 1.0
            context = torch.cat((feature, dt_code), dim=1)
            head_input = (torch.cat((context, latent), dim=1)
                          if self.latent_dim else context)
            residual = self.acceleration_head(head_input)
            if self.architecture == "linear_residual":
                design = torch.cat((context,
                                    torch.ones_like(context[:, :1])), dim=1)
                prior = design @ self.linear_coefficients
                residual = prior + residual
            next_latent = (self.latent_transition(context, latent)
                           if self.latent_dim else None)
            return residual, next_latent

    return StructuredBodyModel()


def integrate_body_state(torch, body, acceleration, dt_s):
    """Apply the known continuous body-frame coupling over one Euler step."""
    u, v, yaw_rate = body.unbind(dim=1)
    ax_eff, ay_eff, yaw_accel = acceleration.unbind(dim=1)
    return torch.stack((
        u + dt_s * (ax_eff + yaw_rate * v),
        v + dt_s * (ay_eff - yaw_rate * u),
        yaw_rate + dt_s * yaw_accel,
    ), dim=1)


def rollout_structured_body(model, history, future, dts,
                            held_input_channels: tuple[int, ...] = ()):
    """Recursively integrate u/v/r; future non-body channels are measured.

    `history`, `future` are normalized with the fold's training-only feature
    statistics. Future steering/throttle feedback, wheel speed and commands
    are teacher-forced to isolate the body-state transition. Future body truth
    is never injected.
    """
    import torch

    current_feature = history[:, -1, :]
    body = current_feature[:, :BODY_CHANNELS]
    latent = model.initial_latent(history)
    feature_mean = model.feature_mean[:BODY_CHANNELS]
    feature_scale = model.feature_scale[:BODY_CHANNELS]
    predictions = []
    for step in range(future.shape[1]):
        dt = dts[:, step]
        acceleration_normalized, next_latent = model.predict_acceleration(
            current_feature, latent, dt)
        acceleration = (acceleration_normalized * model.acceleration_scale
                        + model.acceleration_mean)
        physical_body = body * feature_scale + feature_mean
        # Exact stated body-frame coupling, integrated with the recorded dt.
        next_physical = integrate_body_state(torch, physical_body,
                                             acceleration, dt)
        body = (next_physical - feature_mean) / feature_scale
        predictions.append(body)
        exogenous = future[:, step, BODY_CHANNELS:]
        if held_input_channels:
            exogenous = exogenous.clone()
            for channel in held_input_channels:
                if channel < BODY_CHANNELS or channel >= current_feature.shape[1]:
                    raise ValueError("held input channel is outside the exogenous frame")
                exogenous[:, channel - BODY_CHANNELS] = current_feature[:, channel]
        current_feature = torch.cat((body, exogenous), dim=1)
        latent = next_latent
    if not predictions:
        raise ValueError("rollout must contain at least one future sample")
    return torch.stack(predictions, dim=1)
