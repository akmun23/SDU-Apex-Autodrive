#!/usr/bin/env python3
"""Train a posterior/prior recurrent state-space teacher for vehicle motion.

Posterior latents may use the observed next physical state during training.
Every validation/free rollout uses the command-conditioned prior only. The
teacher is offline research code and never consumes future truth at inference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    ACCELERATION_NAMES,
    COM_X_M,
    DT_S,
    STATE_COUNT,
    STATE_NAMES,
    _eligible_sequences,
    _evaluate,
    _fit_normalizers,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.family_condition_sampler import (
    build_sequence_sampler,
)
from tools.vehicle_dynamics_learning.experiment_artifacts import (
    write_standard_artifacts,
)
from tools.vehicle_dynamics_learning.race_domain_objectives import (
    RACE_SPEED_BIN_EDGES_MPS,
    race_speed_bin_weights,
    weighted_smooth_l1,
)


POSE_LOSS_HORIZONS_S = (0.25, 0.75, 2.0)
POSE_POSITION_SCALE_M = 0.25
POSE_HEADING_SCALE_RAD = 0.10


def _rssm_model(torch, nn, hidden_size: int, latent_size: int,
                x_mean: np.ndarray, x_scale: np.ndarray,
                y_mean: np.ndarray, y_scale: np.ndarray):
    class RSSMTeacher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.latent_size = latent_size
            self.register_buffer("x_mean", torch.as_tensor(x_mean))
            self.register_buffer("x_scale", torch.as_tensor(x_scale))
            self.register_buffer("y_mean", torch.as_tensor(y_mean))
            self.register_buffer("y_scale", torch.as_tensor(y_scale))
            self.context_encoder = nn.GRU(
                input_size=9, hidden_size=hidden_size, num_layers=2,
                batch_first=True)
            self.prior = nn.Sequential(
                nn.Linear(hidden_size + 2, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, 2 * latent_size))
            self.posterior = nn.Sequential(
                nn.Linear(hidden_size + 2 + STATE_COUNT, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, 2 * latent_size))
            self.delta_decoder = nn.Sequential(
                nn.Linear(hidden_size + latent_size + 2 + STATE_COUNT,
                          hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                nn.Linear(hidden_size, STATE_COUNT))
            self.acceleration_decoder = nn.Sequential(
                nn.Linear(hidden_size + latent_size + 2, hidden_size),
                nn.SiLU(), nn.Linear(hidden_size, 3))
            self.memory_transition = nn.GRUCell(
                STATE_COUNT + 2 + latent_size, hidden_size)

        def encode(self, context):
            _, hidden = self.context_encoder(context)
            return hidden[-1]

        def _distribution(self, network, inputs):
            mean, log_std = network(inputs).chunk(2, dim=-1)
            return mean, torch.clamp(log_std, min=-5.0, max=1.5)

        def prior_distribution(self, hidden, command):
            return self._distribution(
                self.prior, torch.cat((hidden, command), dim=-1))

        def posterior_distribution(self, hidden, command, next_state):
            return self._distribution(
                self.posterior,
                torch.cat((hidden, command, next_state), dim=-1))

        def decode(self, hidden, latent, command, current_state):
            decoder_input = torch.cat((hidden, latent, command,
                                       current_state), dim=-1)
            delta = self.delta_decoder(decoder_input)
            acceleration = self.acceleration_decoder(
                torch.cat((hidden, latent, command), dim=-1))
            return current_state + delta, acceleration

        def advance_memory(self, hidden, state, command, latent):
            return self.memory_transition(
                torch.cat((state, command, latent), dim=-1), hidden)

        def initial_state_from_context(self, context):
            raw = context[:, -1, :7] * self.x_scale[:7] + self.x_mean[:7]
            u_rear, v_rear, yaw_rate = raw[:, 0], raw[:, 1], raw[:, 2]
            initial_physical = torch.stack((
                u_rear, v_rear + COM_X_M * yaw_rate, yaw_rate,
                raw[:, 3], raw[:, 4], raw[:, 5], raw[:, 6]), dim=-1)
            return (initial_physical - self.y_mean[:STATE_COUNT]) / (
                self.y_scale[:STATE_COUNT])

        def forward(self, context, future_commands,
                    initial_state=None, sample_prior=False):
            hidden = self.encode(context)
            current_state = (self.initial_state_from_context(context)
                             if initial_state is None else initial_state)
            predictions = []
            for step in range(future_commands.shape[1]):
                command = future_commands[:, step]
                prior_mean, prior_log_std = self.prior_distribution(
                    hidden, command)
                if sample_prior:
                    latent = prior_mean + torch.exp(prior_log_std) * torch.randn_like(
                        prior_mean)
                else:
                    latent = prior_mean
                next_state, acceleration = self.decode(
                    hidden, latent, command, current_state)
                predictions.append(torch.cat((next_state, acceleration), dim=-1))
                hidden = self.advance_memory(
                    hidden, next_state, command, latent)
                current_state = next_state
            return torch.stack(predictions, dim=1)

    return RSSMTeacher


def _mirror_normalized_features(values: np.ndarray, mean: np.ndarray,
                                scale: np.ndarray) -> np.ndarray:
    """Reflect a left-turn sample into the equivalent right-turn frame."""
    physical = np.asarray(values) * scale + mean
    mirrored = physical.copy()
    mirrored[..., 1:4] *= -1.0  # lateral speed, yaw rate, steering feedback
    mirrored[..., 5], mirrored[..., 6] = (
        physical[..., 6].copy(), physical[..., 5].copy())
    mirrored[..., 7] *= -1.0  # steering command
    return ((mirrored - mean) / scale).astype(values.dtype, copy=False)


def _mirror_normalized_commands(values: np.ndarray, mean: np.ndarray,
                                scale: np.ndarray) -> np.ndarray:
    physical = np.asarray(values) * scale + mean
    mirrored = physical.copy()
    mirrored[..., 0] *= -1.0
    return ((mirrored - mean) / scale).astype(values.dtype, copy=False)


def _mirror_normalized_state(values: np.ndarray, mean: np.ndarray,
                             scale: np.ndarray) -> np.ndarray:
    physical = np.asarray(values) * scale + mean
    mirrored = _mirror_physical_state(physical)
    return ((mirrored - mean) / scale).astype(values.dtype, copy=False)


def _mirror_physical_state(values: np.ndarray) -> np.ndarray:
    mirrored = np.asarray(values).copy()
    mirrored[..., 1:4] *= -1.0
    mirrored[..., 5], mirrored[..., 6] = (
        values[..., 6].copy(), values[..., 5].copy())
    return mirrored


def _mirror_normalized_targets(values: np.ndarray, mean: np.ndarray,
                               scale: np.ndarray) -> np.ndarray:
    physical = np.asarray(values) * scale + mean
    mirrored = physical.copy()
    mirrored[..., :STATE_COUNT] = _mirror_physical_state(
        physical[..., :STATE_COUNT])
    mirrored[..., STATE_COUNT + 1] *= -1.0  # lateral acceleration
    mirrored[..., STATE_COUNT + 2] *= -1.0  # yaw acceleration
    return ((mirrored - mean) / scale).astype(values.dtype, copy=False)


def _local_pose_targets(poses: np.ndarray) -> np.ndarray:
    """Express future simulator poses relative to the rollout's initial pose."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("pose window must have shape (N>=2, 3)")
    if not np.isfinite(values).all():
        raise ValueError("pose window contains non-finite values")
    origin = values[0]
    dx = values[1:, 0] - origin[0]
    dy = values[1:, 1] - origin[1]
    cosine, sine = math.cos(origin[2]), math.sin(origin[2])
    local = np.empty((len(values) - 1, 3), dtype=np.float32)
    local[:, 0] = cosine * dx + sine * dy
    local[:, 1] = -sine * dx + cosine * dy
    delta_yaw = values[1:, 2] - origin[2]
    local[:, 2] = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
    return local


def _integrate_pose_from_com_states(torch, predicted_states,
                                    initial_state_normalized,
                                    y_mean: np.ndarray,
                                    y_scale: np.ndarray):
    """Integrate predicted COM states into rear-axle-relative pose."""
    mean = torch.as_tensor(y_mean[:STATE_COUNT], dtype=predicted_states.dtype,
                           device=predicted_states.device)
    scale = torch.as_tensor(y_scale[:STATE_COUNT],
                            dtype=predicted_states.dtype,
                            device=predicted_states.device)
    physical = predicted_states[..., :STATE_COUNT] * scale + mean
    previous = initial_state_normalized * scale + mean
    pose = torch.zeros((physical.shape[0], 3), dtype=physical.dtype,
                       device=physical.device)
    trajectory = []
    for index in range(physical.shape[1]):
        following = physical[:, index, :3]
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
    return torch.stack(trajectory, dim=1)


def _integrated_pose_loss(torch, nn, predicted_states, target_pose,
                          initial_state, y_mean: np.ndarray,
                          y_scale: np.ndarray) -> Any:
    predicted_pose = _integrate_pose_from_com_states(
        torch, predicted_states, initial_state, y_mean, y_scale)
    losses = []
    for seconds in POSE_LOSS_HORIZONS_S:
        index = round(seconds / DT_S) - 1
        if index >= predicted_pose.shape[1]:
            continue
        position_error = (
            predicted_pose[:, index, :2] - target_pose[:, index, :2]
        ) / POSE_POSITION_SCALE_M
        heading_error = torch.atan2(
            torch.sin(predicted_pose[:, index, 2] - target_pose[:, index, 2]),
            torch.cos(predicted_pose[:, index, 2] - target_pose[:, index, 2]),
        ) / POSE_HEADING_SCALE_RAD
        zero_position = torch.zeros_like(position_error)
        zero_heading = torch.zeros_like(heading_error)
        position_loss = nn.functional.smooth_l1_loss(
            position_error, zero_position, beta=1.0)
        heading_loss = nn.functional.smooth_l1_loss(
            heading_error, zero_heading, beta=1.0)
        losses.append(0.5 * (position_loss + heading_loss))
    if not losses:
        raise ValueError("rollout is shorter than every pose-loss horizon")
    return torch.stack(losses).mean()


def _sample_batch(data: dict[str, Any], targets: np.ndarray,
                  groups: dict[int, list[tuple[int, int]]], batch_size: int,
                  context_steps: int, rollout_steps: int,
                  x_mean: np.ndarray, x_scale: np.ndarray,
                  y_mean: np.ndarray, y_scale: np.ndarray,
                  rng: np.random.Generator,
                  family_sampler=None,
                  symmetry_augmentation: bool = False
                  ) -> tuple[np.ndarray, ...]:
    run_ids = np.asarray(sorted(groups), dtype=np.int32)
    contexts, commands, labels, initial_states, pose_targets = [], [], [], [], []
    for _ in range(batch_size):
        if family_sampler is None:
            run_id = int(rng.choice(run_ids))
            start, end = groups[run_id][
                int(rng.integers(0, len(groups[run_id])))]
        else:
            run_id, (start, end), _, _ = family_sampler.sample(rng)
        last_start = end - context_steps - rollout_steps - 1
        index = int(rng.integers(start, last_start + 1))
        future_start = index + context_steps
        feature_history = data["frames"][index:future_start]
        current_feature = feature_history[-1, :7]
        u_rear, v_rear, yaw_rate = current_feature[:3]
        initial_physical = np.asarray((
            u_rear, v_rear + COM_X_M * yaw_rate, yaw_rate,
            current_feature[3], current_feature[4], current_feature[5],
            current_feature[6]), dtype=np.float32)
        contexts.append((feature_history - x_mean) / x_scale)
        commands.append((data["frames"][future_start:future_start + rollout_steps,
                                           7:9] - x_mean[7:9]) / x_scale[7:9])
        labels.append((targets[future_start:future_start + rollout_steps]
                       - y_mean) / y_scale)
        initial_states.append((initial_physical - y_mean[:STATE_COUNT])
                              / y_scale[:STATE_COUNT])
        simulator_pose = data.get("simulator_pose_xyyaw")
        if simulator_pose is None:
            raise ValueError("integrated pose training requires simulator poses")
        pose_window = simulator_pose[
            future_start - 1:future_start + rollout_steps]
        pose_targets.append(_local_pose_targets(pose_window))
    batch = [np.stack(contexts), np.stack(commands), np.stack(labels),
             np.stack(initial_states), np.stack(pose_targets)]
    mirrored_count = 0
    if symmetry_augmentation:
        mirrored = rng.random(batch_size) < 0.5
        mirrored_count = int(np.count_nonzero(mirrored))
        if mirrored_count:
            batch[0][mirrored] = _mirror_normalized_features(
                batch[0][mirrored], x_mean, x_scale)
            batch[1][mirrored] = _mirror_normalized_commands(
                batch[1][mirrored], x_mean[7:9], x_scale[7:9])
            batch[2][mirrored] = _mirror_normalized_targets(
                batch[2][mirrored], y_mean, y_scale)
            batch[3][mirrored] = _mirror_normalized_state(
                batch[3][mirrored], y_mean[:STATE_COUNT],
                y_scale[:STATE_COUNT])
            batch[4][mirrored, :, 1] *= -1.0
            batch[4][mirrored, :, 2] *= -1.0
    return (*batch, mirrored_count)


def _batch_loss(torch, nn, model, context, commands, labels, initial_state,
                kl_weight: float, free_rollout_weight: float,
                acceleration_weight: float, speed_weights=None,
                pose_targets=None, pose_loss_weight: float = 0.1,
                y_mean: np.ndarray | None = None,
                y_scale: np.ndarray | None = None):
    batch_size, horizon = labels.shape[:2]
    hidden = model.encode(context)
    current_state = initial_state
    reconstruction = torch.zeros((), device=context.device)
    kl_total = torch.zeros((), device=context.device)
    acceleration_loss = torch.zeros((), device=context.device)
    for step in range(horizon):
        command = commands[:, step]
        next_target = labels[:, step, :STATE_COUNT]
        prior_mean, prior_log_std = model.prior_distribution(hidden, command)
        post_mean, post_log_std = model.posterior_distribution(
            hidden, command, next_target)
        latent = post_mean + torch.exp(post_log_std) * torch.randn_like(post_mean)
        prediction, acceleration = model.decode(
            hidden, latent, command, current_state)
        if speed_weights is None:
            reconstruction_step = nn.functional.smooth_l1_loss(
                prediction, next_target, beta=0.5)
            acceleration_step = nn.functional.smooth_l1_loss(
                acceleration, labels[:, step, STATE_COUNT:], beta=0.5)
        else:
            reconstruction_step = weighted_smooth_l1(
                torch, prediction, next_target, speed_weights[:, step], beta=0.5)
            acceleration_step = weighted_smooth_l1(
                torch, acceleration, labels[:, step, STATE_COUNT:],
                speed_weights[:, step], beta=0.5)
        reconstruction = reconstruction + reconstruction_step / horizon
        acceleration_loss = acceleration_loss + acceleration_step / horizon
        variance_ratio = torch.exp(2.0 * post_log_std - 2.0 * prior_log_std)
        mean_distance = (post_mean - prior_mean) ** 2 * torch.exp(
            -2.0 * prior_log_std)
        kl = (prior_log_std - post_log_std
              + 0.5 * (variance_ratio + mean_distance - 1.0))
        kl_total = kl_total + kl.sum(dim=-1).mean() / horizon
        hidden = model.advance_memory(hidden, next_target, command, latent)
        current_state = next_target

    # The offline plant uses the prior mean at inference; train its recursive
    # rollout against that same deterministic transition.
    prior_rollout = model(context, commands, initial_state=initial_state,
                          sample_prior=False)
    if speed_weights is None:
        free_rollout_loss = nn.functional.smooth_l1_loss(
            prior_rollout[:, :, :STATE_COUNT], labels[:, :, :STATE_COUNT],
            beta=0.5)
    else:
        free_rollout_loss = weighted_smooth_l1(
            torch, prior_rollout[:, :, :STATE_COUNT],
            labels[:, :, :STATE_COUNT], speed_weights, beta=0.5)
    loss = (free_rollout_weight * free_rollout_loss
            + 0.25 * reconstruction + kl_weight * kl_total
            + acceleration_weight * acceleration_loss)
    pose_loss = torch.zeros((), device=context.device)
    if pose_loss_weight > 0.0:
        if pose_targets is None or y_mean is None or y_scale is None:
            raise ValueError("pose loss requires pose targets and state scales")
        pose_loss = _integrated_pose_loss(
            torch, nn, prior_rollout, pose_targets, initial_state,
            y_mean, y_scale)
        loss = loss + pose_loss_weight * pose_loss
    return loss, {
        "free_rollout": free_rollout_loss,
        "posterior_reconstruction": reconstruction,
        "kl": kl_total,
        "acceleration": acceleration_loss,
        "integrated_pose": pose_loss,
    }


def _pose_aware_selection_score(validation: dict[str, Any],
                                pose_loss_weight: float) -> tuple[float | None,
                                                                  float | None]:
    """Add run-macro short-pose error to the existing speed-balanced score."""
    state_score = validation.get("checkpoint_selection_score")
    if state_score is None:
        return None, None
    rows = []
    for run in validation.get("per_run", {}).values():
        horizons = run.get("horizons", {})
        for horizon in ("0.25s", "0.75s", "2s"):
            metrics = horizons.get(horizon)
            if metrics is None:
                continue
            position_xy = metrics.get("position_xy_rmse_m")
            heading = metrics.get("heading_rmse_rad")
            if (position_xy is None or heading is None
                    or not np.isfinite(position_xy).all()
                    or not np.isfinite(heading)):
                continue
            radial_rmse = float(np.hypot(*position_xy))
            rows.append(0.5 * (
                radial_rmse / POSE_POSITION_SCALE_M
                + float(heading) / POSE_HEADING_SCALE_RAD))
    if not rows:
        if pose_loss_weight == 0.0:
            return float(state_score), None
        return None, None
    pose_score = float(np.mean(rows))
    return float(state_score + pose_loss_weight * pose_score), pose_score


def train(dataset_path: Path, output_dir: Path, *, device_name: str,
          context_seconds: float, rollout_seconds: float,
          hidden_size: int, latent_size: int, batch_size: int,
          max_steps: int, eval_every: int, patience: int,
          max_eval_windows_per_run: int, seed: int,
          score_test: bool,
          symmetry_augmentation: bool = False,
          pose_loss_weight: float = 0.1) -> dict[str, Any]:
    if not np.isfinite(pose_loss_weight) or pose_loss_weight < 0.0:
        raise ValueError("pose loss weight must be finite and nonnegative")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, nn = _torch()
    torch.set_num_threads(1)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    context_steps = round(context_seconds / DT_S)
    rollout_steps = round(rollout_seconds / DT_S)
    data = _load_dataset(dataset_path)
    targets = _targets(data)
    groups = {
        split: _eligible_sequences(data, targets, split, context_steps,
                                   rollout_steps)
        for split in ("train", "validation", "test", "final_test")
    }
    if not groups["train"] or not groups["validation"]:
        raise ValueError("need eligible whole-run train and validation data")
    family_sampler = build_sequence_sampler(data, groups["train"])
    x_mean, x_scale, y_mean, y_scale = _fit_normalizers(
        data, targets, groups["train"])
    Model = _rssm_model(torch, nn, hidden_size, latent_size,
                        x_mean, x_scale, y_mean, y_scale)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4,
                                  weight_decay=1.0e-5)
    rng = np.random.default_rng(seed)
    eval_seed = seed + 65041
    best_score = math.inf
    best_state_score = None
    best_pose_score = None
    best_step = 0
    patience_count = 0
    history = []
    start_time = time.perf_counter()
    steps_completed = 0
    symmetry_augmented_samples = 0
    for step in range(1, max_steps + 1):
        steps_completed = step
        batch = _sample_batch(
            data, targets, groups["train"], batch_size,
            context_steps, rollout_steps, x_mean, x_scale,
            y_mean, y_scale, rng, family_sampler,
            symmetry_augmentation=symmetry_augmentation)
        symmetry_augmented_samples += batch[5]
        speed_weights = None
        if int(data.get("schema_version", 0)) >= 8:
            physical_labels = batch[2] * y_scale + y_mean
            target_speed = np.hypot(
                physical_labels[:, :, 0], physical_labels[:, :, 1])
            speed_weights = torch.as_tensor(
                race_speed_bin_weights(target_speed), dtype=torch.float32,
                device=device)
        context = torch.as_tensor(batch[0], dtype=torch.float32, device=device)
        commands = torch.as_tensor(batch[1], dtype=torch.float32, device=device)
        labels = torch.as_tensor(batch[2], dtype=torch.float32, device=device)
        initial = torch.as_tensor(batch[3], dtype=torch.float32, device=device)
        pose_targets = torch.as_tensor(batch[4], dtype=torch.float32,
                                       device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = _batch_loss(
            torch, nn, model, context, commands, labels, initial,
            kl_weight=0.01, free_rollout_weight=1.0,
            acceleration_weight=0.05, speed_weights=speed_weights,
            pose_targets=pose_targets, pose_loss_weight=pose_loss_weight,
            y_mean=y_mean, y_scale=y_scale)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite RSSM loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == max_steps:
            validation = _evaluate(
                torch, model, data, targets, groups["validation"],
                x_mean, x_scale, y_mean, y_scale, device, context_steps,
                rollout_steps, eval_seed, max_eval_windows_per_run)
            score, pose_score = _pose_aware_selection_score(
                validation, pose_loss_weight=pose_loss_weight)
            event = {
                "step": step,
                "training_loss": float(loss.detach().cpu()),
                "loss_parts": {name: float(value.detach().cpu())
                               for name, value in parts.items()},
                "validation_score": score,
                "validation_state_score": validation[
                    "checkpoint_selection_score"],
                "validation_pose_score": pose_score,
                "validation": validation,
            }
            history.append(event)
            print(json.dumps({"step": step, "loss": event["training_loss"],
                              "validation_score": score}), flush=True)
            if score is not None and score < best_score:
                best_score = score
                best_state_score = validation["checkpoint_selection_score"]
                best_pose_score = pose_score
                best_step = step
                patience_count = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "x_mean": x_mean, "x_scale": x_scale,
                    "y_mean": y_mean, "y_scale": y_scale,
                    "metadata": {
                        "architecture": "rssm_posterior_prior_teacher",
                        "feature_names": data["feature_names"],
                        "target_state_names": list(STATE_NAMES),
                        "acceleration_target_names": list(ACCELERATION_NAMES),
                        "context_steps": context_steps,
                        "rollout_steps": rollout_steps,
                        "hidden_size": hidden_size,
                        "latent_size": latent_size,
                        "training_objective": {
                            "prior_rollout": "deterministic prior mean",
                            "integrated_pose_loss_weight": pose_loss_weight,
                            "integrated_pose_horizons_s": list(
                                POSE_LOSS_HORIZONS_S),
                            "position_scale_m": POSE_POSITION_SCALE_M,
                            "heading_scale_rad": POSE_HEADING_SCALE_RAD,
                        },
                        "training_runs": [str(data["run_ids"][i])
                                          for i in sorted(groups["train"])],
                        "posterior_uses_future_target_during_training_only": True,
                        "free_rollout_uses_prior_only": True,
                    },
                }, output_dir / "best_rssm_teacher.pt")
            else:
                patience_count += 1
            (output_dir / "validation_history.json").write_text(
                json.dumps(history, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            if patience_count >= patience:
                break

    best_payload = torch.load(output_dir / "best_rssm_teacher.pt",
                              map_location=device, weights_only=False)
    model.load_state_dict(best_payload["state_dict"])
    model.eval()
    validation = _evaluate(
        torch, model, data, targets, groups["validation"],
        x_mean, x_scale, y_mean, y_scale, device, context_steps,
        rollout_steps, eval_seed, max_eval_windows_per_run,
        stratify=int(data.get("schema_version", 0)) >= 8)
    test = (_evaluate(
        torch, model, data, targets, groups["test"],
        x_mean, x_scale, y_mean, y_scale, device, context_steps,
        rollout_steps, eval_seed + 1, max_eval_windows_per_run,
        stratify=int(data.get("schema_version", 0)) >= 8)
        if score_test and groups["test"] else None)
    report = {
        "schema_version": 1,
        "architecture": "rssm_posterior_prior_teacher",
        "dataset": str(dataset_path.resolve()),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else None),
        "context_seconds": context_steps * DT_S,
        "rollout_seconds": rollout_steps * DT_S,
        "hidden_size": hidden_size,
        "latent_size": latent_size,
        "batch_size": batch_size,
        "steps_completed": steps_completed,
        "optimizer_steps_per_second": steps_completed / max(
            time.perf_counter() - start_time, 1e-9),
        "best_step": best_step,
        "best_validation_score": best_score,
        "best_pose_aware_validation_score": best_score,
        "best_validation_state_score": best_state_score,
        "best_validation_pose_score": best_pose_score,
        "eligible_runs_by_split": {
            split: [str(data["run_ids"][i]) for i in sorted(local_groups)]
            for split, local_groups in groups.items()},
        "eligible_sequences_by_split": {
            split: sum(map(len, local_groups.values()))
            for split, local_groups in groups.items()},
        "training_sampler": family_sampler.metadata,
        "symmetry_augmentation": symmetry_augmentation,
        "symmetry_augmented_sample_fraction": (
            symmetry_augmented_samples / (steps_completed * batch_size)),
        "training_objective": {
            "prior_rollout": "deterministic prior mean to match plant inference",
            "integrated_pose_loss_weight": pose_loss_weight,
            "integrated_pose_horizons_s": list(POSE_LOSS_HORIZONS_S),
            "position_scale_m": POSE_POSITION_SCALE_M,
            "heading_scale_rad": POSE_HEADING_SCALE_RAD,
            "checkpoint_selection": (
                "existing speed/run-balanced body-state score + pose-loss-weight times "
                "run-macro normalized position/heading RMSE at 0.25/0.75/2 s"),
        },
        "validation": validation,
        "test_scored_once": bool(score_test),
        "test": test,
        "posterior_uses_future_target_during_training_only": True,
        "free_rollout_uses_prior_only": True,
        "race_domain_loss_weighting": ({
            "speed_bins_mps": list(RACE_SPEED_BIN_EDGES_MPS),
            "method": "equal total contribution per populated true future-speed bin in each batch",
            "checkpoint_selection": "equal core 0-9 / fast boundary 9-12 macro over independent validation runs at 0.25, 0.75, and 2 seconds",
        } if int(data.get("schema_version", 0)) >= 8 else None),
        "checkpoint": str(output_dir / "best_rssm_teacher.pt"),
        "history": history,
    }
    write_standard_artifacts(
        output_dir, dataset_path, data,
        {"architecture": "rssm_posterior_prior_teacher",
         "context_seconds": context_steps * DT_S,
         "rollout_seconds": rollout_steps * DT_S,
         "hidden_size": hidden_size, "latent_size": latent_size,
         "batch_size": batch_size, "max_steps": max_steps,
         "eval_every": eval_every, "patience": patience,
         "max_eval_windows_per_run": max_eval_windows_per_run,
         "pose_loss_weight": pose_loss_weight,
         "training_sampler": family_sampler.metadata,
         "symmetry_augmentation": symmetry_augmentation,
         "symmetry_augmented_sample_fraction": (
             symmetry_augmented_samples / (steps_completed * batch_size)),
         "score_test": score_test}, seed, report,
        ("tools/vehicle_dynamics_learning/train_rssm_teacher.py",
         "tools/vehicle_dynamics_learning/train_direct_sequence_teacher.py",
         "tools/vehicle_dynamics_learning/train_nssm.py",
         "tools/vehicle_dynamics_learning/experiment_artifacts.py"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-seconds", type=float, default=1.0)
    parser.add_argument("--rollout-seconds", type=float, default=2.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-eval-windows-per-run", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--pose-loss-weight", type=float, default=0.1)
    parser.add_argument("--score-test", action="store_true")
    parser.add_argument("--symmetry-augmentation", action="store_true",
                        help="randomly reflect half of sampled training windows")
    args = parser.parse_args()
    train(
        args.dataset, args.output_dir, device_name=args.device,
        context_seconds=args.context_seconds,
        rollout_seconds=args.rollout_seconds,
        hidden_size=args.hidden_size, latent_size=args.latent_size,
        batch_size=args.batch_size, max_steps=args.max_steps,
        eval_every=args.eval_every, patience=args.patience,
        max_eval_windows_per_run=args.max_eval_windows_per_run,
        seed=args.seed, score_test=args.score_test,
        symmetry_augmentation=args.symmetry_augmentation,
        pose_loss_weight=args.pose_loss_weight)
    print(f"wrote {args.output_dir / 'training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
