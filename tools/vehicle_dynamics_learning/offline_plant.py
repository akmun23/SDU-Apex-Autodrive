#!/usr/bin/env python3
"""Shared causal step/reset interface for offline plant candidates.

The first adapter is the frozen historical GRU. It is intentionally a
development-only interface: state history at reset is an initial condition,
and after reset only the commanded steering/throttle are exogenous inputs.
Future measured actuator, encoder, odometry, or simulator-truth values are
never read by ``step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.evaluate_free_running_plant import (
    _load_models,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import (
    COM_X_M,
    _rssm_model,
)


BODY_STATE_NAMES = (
    "u_rear_mps", "v_rear_mps", "yaw_rate_rps", "steering_feedback_rad",
    "throttle_feedback_norm", "rear_left_surface_mps",
    "rear_right_surface_mps",
)
STATE_NAMES = ("x_m", "y_m", "yaw_rad", *BODY_STATE_NAMES)
COMMAND_NAMES = ("steering_command_rad", "throttle_command_norm")
FRAME_FEATURE_COUNT = 9


@dataclass(frozen=True)
class PlantEstimate:
    state: np.ndarray
    uncertainty: np.ndarray
    support_distance: float | None


def _integrate_pose(previous: np.ndarray, current: np.ndarray,
                    pose: np.ndarray, dt_s: float) -> np.ndarray:
    u = 0.5 * (float(previous[0]) + float(current[0]))
    v = 0.5 * (float(previous[1]) + float(current[1]))
    r = 0.5 * (float(previous[2]) + float(current[2]))
    yaw_mid = float(pose[2]) + 0.5 * r * dt_s
    return np.asarray((
        pose[0] + (u * np.cos(yaw_mid) - v * np.sin(yaw_mid)) * dt_s,
        pose[1] + (u * np.sin(yaw_mid) + v * np.cos(yaw_mid)) * dt_s,
        pose[2] + r * dt_s,
    ), dtype=np.float32)


def _summarize_ensemble(member_states: np.ndarray,
                        member_poses: np.ndarray) -> PlantEstimate:
    mean_state = np.mean(member_states, axis=0)
    state_spread = np.std(member_states, axis=0, ddof=0)
    mean_pose = np.mean(member_poses[:, :2], axis=0)
    mean_yaw = np.arctan2(np.mean(np.sin(member_poses[:, 2])),
                          np.mean(np.cos(member_poses[:, 2])))
    yaw_errors = np.arctan2(np.sin(member_poses[:, 2] - mean_yaw),
                            np.cos(member_poses[:, 2] - mean_yaw))
    pose_spread = np.asarray((
        np.std(member_poses[:, 0], ddof=0),
        np.std(member_poses[:, 1], ddof=0),
        np.sqrt(np.mean(yaw_errors ** 2)),
    ), dtype=np.float32)
    state = np.concatenate((mean_pose, np.asarray([mean_yaw]), mean_state))
    uncertainty = np.concatenate((pose_spread, state_spread)).astype(np.float32)
    return PlantEstimate(state.astype(np.float32), uncertainty, None)


class HistoricalGruPlant:
    """Step a frozen ensemble of command-driven recurrent plant models."""

    def __init__(self, torch, models: list[Any], feature_mean: np.ndarray,
                 feature_scale: np.ndarray, history_steps: int,
                 support_tree=None) -> None:
        if not models:
            raise ValueError("at least one frozen model is required")
        if feature_mean.shape != (FRAME_FEATURE_COUNT,):
            raise ValueError("feature_mean must have nine channels")
        if feature_scale.shape != (FRAME_FEATURE_COUNT,):
            raise ValueError("feature_scale must have nine channels")
        if not np.isfinite(feature_mean).all() or not np.isfinite(feature_scale).all():
            raise ValueError("model normalizers must be finite")
        if np.any(feature_scale <= 0.0):
            raise ValueError("model scales must be positive")
        self.torch = torch
        self.models = models
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float32)
        self.history_steps = int(history_steps)
        if self.history_steps < 2:
            raise ValueError("historical GRU needs at least two context frames")
        self.support_tree = support_tree
        self._features: list[Any] = []
        self._hidden: list[Any] = []
        self._member_states: np.ndarray | None = None
        self._member_poses: np.ndarray | None = None
        self._estimate: PlantEstimate | None = None

    def reset(self, observed_history: np.ndarray,
              initial_pose_xyyaw: np.ndarray | None = None) -> PlantEstimate:
        """Initialize from exactly one past history ending at the plant state."""
        history = np.asarray(observed_history, dtype=np.float32)
        if history.shape != (self.history_steps, FRAME_FEATURE_COUNT):
            raise ValueError(
                f"history must have shape ({self.history_steps}, 9)")
        if not np.isfinite(history).all():
            raise ValueError("initial history contains non-finite values")
        pose = (np.zeros(3, dtype=np.float32) if initial_pose_xyyaw is None
                else np.asarray(initial_pose_xyyaw, dtype=np.float32))
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("initial pose must contain finite x, y, yaw")
        normalized = (history - self.feature_mean) / self.feature_scale
        self._features = []
        self._hidden = []
        with self.torch.no_grad():
            for model in self.models:
                device = next(model.parameters()).device
                sequence = self.torch.as_tensor(
                    normalized, dtype=self.torch.float32, device=device)
                hidden = self.torch.zeros(
                    (1, model.cell.hidden_size), dtype=sequence.dtype,
                    device=device)
                for index in range(self.history_steps - 1):
                    hidden = model.cell(sequence[index:index + 1], hidden)
                self._features.append(sequence[-1:].clone())
                self._hidden.append(hidden)
        self._member_states = np.broadcast_to(
            history[-1, :7], (len(self.models), 7)).copy()
        self._member_poses = np.broadcast_to(
            pose, (len(self.models), 3)).copy()
        self._estimate = self._make_estimate()
        return self._estimate

    def step(self, steering_command_rad: float, throttle_command_norm: float,
             dt_s: float = 0.025) -> PlantEstimate:
        """Advance once using commands only; measured future channels are absent."""
        if not self._features:
            raise RuntimeError("reset must be called before step")
        if not np.isfinite((steering_command_rad, throttle_command_norm, dt_s)).all():
            raise ValueError("commands and dt must be finite")
        if dt_s <= 0.0:
            raise ValueError("dt must be positive")
        torch = self.torch
        command_physical = np.asarray(
            [steering_command_rad, throttle_command_norm], dtype=np.float32)
        command_normalized = (command_physical - self.feature_mean[7:9]) / self.feature_scale[7:9]
        next_states = []
        next_features = []
        next_hidden = []
        previous_states = self._member_states
        with torch.no_grad():
            for model, feature, hidden in zip(
                    self.models, self._features, self._hidden):
                step = torch.as_tensor(
                    [dt_s], dtype=feature.dtype, device=feature.device)
                predicted, updated_hidden = model.advance(feature, hidden, step)
                physical = (predicted[0].cpu().numpy().astype(np.float32)
                            * self.feature_scale[:7]
                            + self.feature_mean[:7])
                command = torch.as_tensor(
                    command_normalized[None, :], dtype=feature.dtype,
                    device=feature.device)
                next_feature = torch.cat((predicted, command), dim=1)
                next_states.append(physical)
                next_features.append(next_feature)
                next_hidden.append(updated_hidden)
        self._features = next_features
        self._hidden = next_hidden
        self._member_states = np.stack(next_states)
        self._member_poses = np.stack([
            _integrate_pose(previous, following, pose, dt_s)
            for previous, following, pose in zip(
                previous_states, self._member_states, self._member_poses)
        ])
        self._estimate = self._make_estimate()
        return self._estimate

    def get_state(self) -> np.ndarray:
        if self._estimate is None:
            raise RuntimeError("reset must be called before get_state")
        return self._estimate.state.copy()

    def get_uncertainty(self) -> np.ndarray:
        if self._estimate is None:
            raise RuntimeError("reset must be called before get_uncertainty")
        return self._estimate.uncertainty.copy()

    def get_support(self) -> float | None:
        if self._estimate is None:
            raise RuntimeError("reset must be called before get_support")
        return self._estimate.support_distance

    def _make_estimate(self) -> PlantEstimate:
        mean_state = np.mean(self._member_states, axis=0)
        support = None
        if self.support_tree is not None:
            normalized = ((mean_state - self.feature_mean[:7])
                          / self.feature_scale[:7])
            support = float(self.support_tree.query(normalized, k=1)[0])
        result = _summarize_ensemble(self._member_states, self._member_poses)
        return PlantEstimate(result.state, result.uncertainty, support)


def build_training_support_tree(data: dict[str, Any],
                                feature_mean: np.ndarray,
                                feature_scale: np.ndarray,
                                stride: int = 4):
    """Create a compact train-only nearest-state support index.

    The returned distance is a state-space coverage diagnostic, not a
    calibrated probability or a proof that the full hidden state is in
    distribution.
    """
    if stride < 1:
        raise ValueError("support stride must be positive")
    from scipy.spatial import cKDTree

    rows = []
    for sequence_id, (start_value, end_value) in enumerate(data["bounds"]):
        run_id = int(data["seq_run"][sequence_id])
        if data["splits"][run_id] != "train":
            continue
        start, end = int(start_value), int(end_value)
        local = data["frames"][start:end:stride, :7]
        if len(local):
            rows.append(local)
    if not rows:
        raise ValueError("no training states available for support index")
    training = np.concatenate(rows).astype(np.float32)
    normalized = (training - feature_mean[None, :7]) / feature_scale[None, :7]
    return cKDTree(normalized, compact_nodes=True, balanced_tree=True)


def load_historical_gru_plant(run_dir: Path, dataset_path: Path,
                              device: str = "cpu") -> HistoricalGruPlant:
    """Load the frozen baseline and construct its train-only support index."""
    torch, _ = _torch()
    if device == "cpu":
        torch.set_num_threads(1)
    data = _load_dataset(dataset_path)
    torch, models, payload, metadata, _ = _load_models(run_dir, data, device)
    support_tree = build_training_support_tree(
        data, payload["feature_mean"], payload["feature_scale"])
    return HistoricalGruPlant(
        torch, models, payload["feature_mean"], payload["feature_scale"],
        int(metadata["history_steps"]), support_tree)


class RssmTeacherPlant:
    """Prior-mean RSSM adapter with the same reset/step/query surface.

    Its reported uncertainty is ensemble spread only. The latent prior
    variance is not converted to state uncertainty without a separate
    calibration pass.
    """

    def __init__(self, torch, models: list[Any], x_mean: np.ndarray,
                 x_scale: np.ndarray, y_mean: np.ndarray,
                 y_scale: np.ndarray, context_steps: int,
                 support_tree=None) -> None:
        if not models:
            raise ValueError("at least one frozen RSSM model is required")
        if x_mean.shape != (9,) or x_scale.shape != (9,):
            raise ValueError("RSSM feature normalizers must have nine channels")
        if y_mean.shape != (10,) or y_scale.shape != (10,):
            raise ValueError("RSSM target normalizers must have ten channels")
        self.torch = torch
        self.models = models
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_scale = np.asarray(x_scale, dtype=np.float32)
        self.y_mean = np.asarray(y_mean, dtype=np.float32)
        self.y_scale = np.asarray(y_scale, dtype=np.float32)
        self.context_steps = int(context_steps)
        self.support_tree = support_tree
        self._hidden: list[Any] = []
        self._states: list[Any] = []
        self._member_states: np.ndarray | None = None
        self._member_poses: np.ndarray | None = None
        self._estimate: PlantEstimate | None = None

    @property
    def history_steps(self) -> int:
        """Context length expected by the shared branch-evaluation API."""
        return self.context_steps

    def reset(self, observed_history: np.ndarray,
              initial_pose_xyyaw: np.ndarray | None = None) -> PlantEstimate:
        history = np.asarray(observed_history, dtype=np.float32)
        if history.shape != (self.context_steps, FRAME_FEATURE_COUNT):
            raise ValueError(
                f"history must have shape ({self.context_steps}, 9)")
        if not np.isfinite(history).all():
            raise ValueError("initial history contains non-finite values")
        pose = (np.zeros(3, dtype=np.float32) if initial_pose_xyyaw is None
                else np.asarray(initial_pose_xyyaw, dtype=np.float32))
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("initial pose must contain finite x, y, yaw")
        normalized = (history - self.x_mean) / self.x_scale
        self._hidden = []
        self._states = []
        with self.torch.no_grad():
            for model in self.models:
                device = next(model.parameters()).device
                context = self.torch.as_tensor(
                    normalized[None], dtype=self.torch.float32, device=device)
                self._hidden.append(model.encode(context))
                self._states.append(model.initial_state_from_context(context))
        self._member_states = self._decode_member_states()
        self._member_poses = np.broadcast_to(
            pose, (len(self.models), 3)).copy()
        self._estimate = self._make_estimate()
        return self._estimate

    def step(self, steering_command_rad: float, throttle_command_norm: float,
             dt_s: float = 0.025) -> PlantEstimate:
        if not self._states:
            raise RuntimeError("reset must be called before step")
        if not np.isfinite((steering_command_rad, throttle_command_norm, dt_s)).all():
            raise ValueError("commands and dt must be finite")
        if dt_s <= 0.0:
            raise ValueError("dt must be positive")
        command_physical = np.asarray(
            [steering_command_rad, throttle_command_norm], dtype=np.float32)
        command_normalized = (
            command_physical - self.x_mean[7:9]) / self.x_scale[7:9]
        following_states, following_hidden = [], []
        previous_states = self._member_states
        with self.torch.no_grad():
            for model, hidden, current_state in zip(
                    self.models, self._hidden, self._states):
                device = current_state.device
                command = self.torch.as_tensor(
                    command_normalized[None], dtype=current_state.dtype,
                    device=device)
                prior_mean, _ = model.prior_distribution(
                    hidden, command)
                latent = prior_mean
                prediction, _ = model.decode(
                    hidden, latent, command, current_state)
                updated_hidden = model.advance_memory(
                    hidden, prediction, command, latent)
                following_states.append(prediction)
                following_hidden.append(updated_hidden)
        self._states = following_states
        self._hidden = following_hidden
        self._member_states = self._decode_member_states()
        self._member_poses = np.stack([
            _integrate_pose(previous, following, pose, dt_s)
            for previous, following, pose in zip(
                previous_states, self._member_states, self._member_poses)
        ])
        self._estimate = self._make_estimate()
        return self._estimate

    def get_state(self) -> np.ndarray:
        if self._estimate is None:
            raise RuntimeError("reset must be called before get_state")
        return self._estimate.state.copy()

    def get_uncertainty(self) -> np.ndarray:
        if self._estimate is None:
            raise RuntimeError("reset must be called before get_uncertainty")
        return self._estimate.uncertainty.copy()

    def get_support(self) -> float | None:
        if self._estimate is None:
            raise RuntimeError("reset must be called before get_support")
        return self._estimate.support_distance

    def _decode_member_states(self) -> np.ndarray:
        physical_states = np.stack([
            (state[0].cpu().numpy() * self.y_scale[:7]
             + self.y_mean[:7]) for state in self._states
        ])
        # The RSSM target is at the Unity COM. The common offline API and
        # production odometry are rear-axle referenced.
        physical_states[:, 1] -= COM_X_M * physical_states[:, 2]
        return physical_states.astype(np.float32)

    def _make_estimate(self) -> PlantEstimate:
        mean_state = np.mean(self._member_states, axis=0)
        support = None
        if self.support_tree is not None:
            normalized = ((mean_state - self.x_mean[:7])
                          / self.x_scale[:7])
            support = float(self.support_tree.query(normalized, k=1)[0])
        result = _summarize_ensemble(self._member_states, self._member_poses)
        return PlantEstimate(result.state, result.uncertainty, support)


def load_rssm_teacher_plant(checkpoints: list[Path], dataset_path: Path,
                            device: str = "cpu") -> RssmTeacherPlant:
    """Load one or more fixed-split RSSM checkpoints for prior-only stepping."""
    if not checkpoints:
        raise ValueError("at least one RSSM checkpoint is required")
    torch, nn = _torch()
    if device == "cpu":
        torch.set_num_threads(1)
    data = _load_dataset(dataset_path)
    payloads = [torch.load(path, map_location="cpu", weights_only=False)
                for path in checkpoints]
    first = payloads[0]
    metadata = first["metadata"]
    if metadata["feature_names"] != data["feature_names"]:
        raise ValueError("RSSM and dataset feature layouts differ")
    x_mean, x_scale = first["x_mean"], first["x_scale"]
    y_mean, y_scale = first["y_mean"], first["y_scale"]
    model_type = _rssm_model(
        torch, nn, int(metadata["hidden_size"]),
        int(metadata["latent_size"]), x_mean, x_scale, y_mean, y_scale)
    models = []
    for payload in payloads:
        if payload["metadata"] != metadata:
            raise ValueError("RSSM ensemble metadata differs between checkpoints")
        model = model_type().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    support_tree = build_training_support_tree(data, x_mean, x_scale)
    return RssmTeacherPlant(
        torch, models, x_mean, x_scale, y_mean, y_scale,
        int(metadata["context_steps"]), support_tree)
