#!/usr/bin/env python3
"""Train one deterministic EDSSM encoder/seed on frozen schema-9 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    ACCELERATION_NAMES,
    COMMAND_NAMES,
    DT_S,
    HISTORY_STEPS,
    LATENT_SIZE,
    PHYSICAL_STATE_NAMES,
    ROLL_STATE_NAMES,
    RAW_ENCODER_HISTORY_NAMES,
    WHEEL_INNOVATION_HISTORY_NAMES,
    WHEEL_DYNAMICS_MODES,
    acceleration_target_names,
    acceleration_targets_from_dataset,
    append_roll_state,
    ActuatorChannel,
    ActuatorFit,
    effective_model_type,
    fit_roll_oscillator,
    fit_actuator_dynamics,
    generalized_acceleration_targets,
    integrate_pose,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.encoder_sidecar_compatibility import (
    is_fixed_cadence_variant,
)
from tools.vehicle_dynamics_learning.family_condition_sampler import (
    FamilyConditionSampler,
    run_family_labels,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    DEFAULT_DATASET,
    _fixed_validation_windows,
    _robust_normalizers,
    evaluate_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.vehicle_dynamics_learning.turn_reflection import (
    reflect_acceleration_targets,
    reflect_commands,
    reflect_history_features,
    reflect_pose_sequence,
    reflect_state_channels,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
             "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002")
DEFAULT_OUTPUT_ROOT = TASK_ROOT / "edssm_training_20261002"
CURRICULUM = (
    ("0.75s", (30,)),
    ("0.75s_plus_2s", (30, 80)),
    ("0.75s_plus_2s_plus_5s", (30, 80, 200)),
)
SPEED_EDGES = np.asarray((0.0, 3.0, 5.0, 7.0, 9.0, 12.000001))
LOSS_WEIGHTS = {
    "body_state": 1.0,
    "rear_wheel_state": 0.50,
    "roll_state": 0.25,
    "effective_acceleration": 0.25,
    "pose_secondary": 0.10,
    "latent_l2": 1e-4,
}


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for EDSSM training") from exc
    return torch, nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_append_only_dataset_extension(source_metadata: dict[str, Any],
                                         target_dataset: Path,
                                         target_data: dict[str, Any]
                                         ) -> dict[str, Any]:
    """Prove that a new training view only appends whole-run data.

    Warm-start transfer across dataset hashes is safe only when every old
    archive array is an exact prefix of its counterpart and the whole-run
    validation membership is unchanged. This intentionally rejects reordering,
    edits, changed labels, and altered validation/test rows.
    """
    source_value = source_metadata.get("dataset_path")
    if not source_value:
        raise ValueError("checkpoint has no source dataset path for prefix proof")
    source_dataset = Path(str(source_value))
    if str(source_dataset).startswith("/workspace/src/"):
        source_dataset = REPO_ROOT / str(source_dataset).removeprefix(
            "/workspace/src/")
    elif not source_dataset.is_absolute():
        source_dataset = REPO_ROOT / source_dataset
    source_dataset = source_dataset.resolve()
    if (not source_dataset.is_file()
            or _sha256(source_dataset) != source_metadata.get("dataset_sha256")):
        raise ValueError("checkpoint source dataset is missing or its hash changed")

    source_train = [str(value) for value in source_metadata.get("training_runs", ())]
    source_validation = [str(value) for value in
                         source_metadata.get("validation_runs", ())]
    target_train = [str(run) for run, split in
                    zip(target_data["run_ids"], target_data["splits"])
                    if str(split) == "train"]
    target_validation = [str(run) for run, split in
                         zip(target_data["run_ids"], target_data["splits"])
                         if str(split) == "validation"]
    if (not source_train or not set(source_train).issubset(target_train)
            or source_validation != target_validation):
        raise ValueError(
            "append-only transfer requires source training runs to remain in "
            "training and identical whole-run validation membership")

    prefix_arrays = 0
    appended_rows: dict[str, int] = {}
    with np.load(source_dataset, allow_pickle=False) as source, \
            np.load(target_dataset, allow_pickle=False) as target:
        if set(source.files) != set(target.files):
            raise ValueError("append-only transfer found changed dataset fields")
        for name in source.files:
            old, new = source[name], target[name]
            if old.shape == new.shape:
                candidate = new
            elif (old.ndim > 0 and old.ndim == new.ndim
                  and old.shape[1:] == new.shape[1:]
                  and old.shape[0] <= new.shape[0]):
                candidate = new[:old.shape[0]]
                appended_rows[name] = int(new.shape[0] - old.shape[0])
            else:
                raise ValueError(
                    f"append-only transfer found incompatible shape for {name}")
            equal = (np.array_equal(old, candidate, equal_nan=True)
                     if old.dtype.kind in "fc" else
                     np.array_equal(old, candidate))
            if not equal:
                raise ValueError(
                    f"append-only transfer found modified prefix in {name}")
            prefix_arrays += 1
    return {
        "verified": True,
        "source_dataset": str(source_dataset),
        "source_dataset_sha256": str(source_metadata["dataset_sha256"]),
        "target_dataset_sha256": _sha256(target_dataset),
        "exact_prefix_array_count": prefix_arrays,
        "source_training_runs_preserved": len(source_train),
        "new_training_run_count": len(target_train) - len(source_train),
        "validation_runs_unchanged": len(source_validation),
        "appended_rows_by_array": appended_rows,
    }


def _inherit_checkpoint_calibration(model: Any,
                                    normalizers: dict[str, np.ndarray],
                                    arrays: dict[str, np.ndarray],
                                    source_metadata: dict[str, Any]
                                    ) -> ActuatorFit:
    """Retain a checkpoint's scales and actuator response during extension."""
    for name in arrays:
        inherited = np.asarray(source_metadata[name], dtype=np.float32)
        arrays[name] = inherited
        normalizers[name] = inherited
    actuator = source_metadata["actuator_fit"]
    fit = ActuatorFit(
        ActuatorChannel(**actuator["steering"]),
        ActuatorChannel(**actuator["throttle"]),
        actuator.get("diagnostics", {}))
    model.steering_delay_steps = fit.steering.delay_steps
    model.throttle_delay_steps = fit.throttle.delay_steps
    model.steering_alpha = float(fit.steering.alpha)
    model.throttle_alpha = float(fit.throttle.alpha)
    return fit


def _run_indices(data: dict[str, Any]) -> np.ndarray:
    result = np.full(len(data["frames"]), -1, dtype=np.int32)
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        result[start:end] = run
    if np.any(result < 0):
        raise ValueError("sequence bounds do not cover the dataset")
    return result


def _build_window_sampler(data: dict[str, Any], state: np.ndarray,
                          horizon_steps: int, split: str,
                          mismatch_edges: tuple[float, float] | None = None,
                          stride_steps: int | None = None,
                          stratify_signed_wheel_mismatch: bool = False,
                          signed_mismatch_edges: tuple[float, float] = (-0.5, 0.5),
                          require_raw_encoder_targets: bool = False,
                          max_throttle_command: float = 0.50,
                          ) -> tuple[FamilyConditionSampler, dict[str, Any]]:
    if split not in ("train", "validation"):
        raise ValueError("EDSSM sampler supports only train/validation")
    if horizon_steps not in (30, 80, 200, 400):
        raise ValueError("EDSSM windows must use 0.75/2/5/10 second horizons")
    if (not np.isfinite(max_throttle_command)
            or not 0.0 <= max_throttle_command <= 1.0):
        raise ValueError("maximum throttle command must be in [0, 1]")
    frames = np.asarray(data["frames"], dtype=np.float64)
    pose = data.get("simulator_pose_xyyaw")
    if pose is None:
        raise ValueError("EDSSM requires simulator pose labels for local pose loss")
    signed_mismatch = 0.5 * (state[:, 5] + state[:, 6]) - state[:, 0]
    mismatch = np.abs(signed_mismatch)
    signed_mismatch_edges = tuple(map(float, signed_mismatch_edges))
    if (len(signed_mismatch_edges) != 2
            or not np.isfinite(signed_mismatch_edges).all()
            or signed_mismatch_edges[0] >= signed_mismatch_edges[1]):
        raise ValueError("signed wheel/body mismatch edges must be finite and increasing")
    speed = np.hypot(state[:, 0], state[:, 1])
    run_indices = np.asarray(data["splits"]).astype(str)
    families = run_family_labels(data)
    sequence_conditions = data.get("sequence_condition_id")
    if sequence_conditions is None:
        sequence_conditions = np.arange(len(data["bounds"]), dtype=np.int32)
    sequence_conditions = np.asarray(sequence_conditions, dtype=np.int64)
    raw_wheel_valid = data.get("encoder_raw_valid")
    if require_raw_encoder_targets and raw_wheel_valid is None:
        raise ValueError("raw wheel targets require the verified encoder sidecar")
    if require_raw_encoder_targets and raw_wheel_valid is not None:
        raw_wheel_valid = np.asarray(raw_wheel_valid, dtype=bool)
        if raw_wheel_valid.shape != (len(frames),):
            raise ValueError("raw encoder validity mask does not align with frames")
    step = (max(1, horizon_steps // 4) if stride_steps is None
            else int(stride_steps))
    items_by_run: dict[int, list[tuple[int, int]]] = {}
    item_condition: dict[tuple[int, int], tuple[int, ...]] = {}
    item_stats: dict[tuple[int, int], tuple[float, float, float]] = {}
    train_mismatches: list[float] = []
    for sequence_id, ((start_raw, end_raw), run_raw) in enumerate(
            zip(data["bounds"], data["seq_run"])):
        seq_start, seq_end, run = int(start_raw), int(end_raw), int(run_raw)
        if run_indices[run] != split:
            continue
        low = seq_start + HISTORY_STEPS - 1
        high = seq_end - horizon_steps - 1
        if high < low:
            continue
        starts = list(range(low, high + 1, step))
        if starts[-1] != high:
            starts.append(high)
        for start in starts:
            future = np.arange(start + 1, start + horizon_steps + 1)
            if require_raw_encoder_targets:
                # Measured history may contain a causal filtered-odometry
                # fill at encoder dropouts. Every future wheel target must be
                # a true packet-aligned raw encoder rate.
                if not np.all(raw_wheel_valid[future]):
                    continue
            if (np.max(speed[future]) > 12.0
                    or np.max(frames[future, 8]) > max_throttle_command
                    or not np.isfinite(pose[np.concatenate(([start], future))]).all()):
                continue
            key_item = (sequence_id, start)
            items_by_run.setdefault(run, []).append(key_item)
            local_speed = float(speed[start])
            local_mismatch = float(mismatch[start])
            local_signed_mismatch = float(signed_mismatch[start])
            item_stats[key_item] = (local_speed, local_mismatch,
                                    local_signed_mismatch)
            if split == "train":
                train_mismatches.append(local_mismatch)

    if split == "train":
        if not train_mismatches:
            raise ValueError("no supported train windows for EDSSM sampler")
        mismatch_edges = tuple(map(float, np.quantile(
            train_mismatches, (0.50, 0.90))))
    elif mismatch_edges is None:
        raise ValueError("validation sampler needs training-only mismatch cutpoints")
    assert mismatch_edges is not None
    for item, (local_speed, local_mismatch, local_signed_mismatch) in item_stats.items():
        speed_bin = int(np.searchsorted(SPEED_EDGES[1:], local_speed, side="right"))
        mismatch_bin = int(np.searchsorted(
            np.asarray(mismatch_edges), local_mismatch, side="right"))
        sequence_id = item[0]
        condition = int(sequence_conditions[sequence_id])
        if stratify_signed_wheel_mismatch:
            signed_mismatch_bin = int(np.searchsorted(
                np.asarray(signed_mismatch_edges), local_signed_mismatch,
                side="right"))
            item_condition[item] = (
                condition, speed_bin, mismatch_bin, signed_mismatch_bin)
        else:
            item_condition[item] = (condition, speed_bin, mismatch_bin)
    family_weights = None
    if (data.get("training_family_names") is not None
            and data.get("training_family_probabilities") is not None):
        family_weights = dict(zip(
            np.asarray(data["training_family_names"]).astype(str).tolist(),
            np.asarray(data["training_family_probabilities"],
                       dtype=np.float64).tolist()))
    sampler = FamilyConditionSampler(
        items_by_run, families,
        condition_for=lambda _run, item: item_condition[item],
        family_weights=family_weights,
    )
    report = {
        "split": split,
        "horizon_steps": horizon_steps,
        "window_stride_steps": step,
        "eligible_windows": sum(map(len, items_by_run.values())),
        "runs": {str(data["run_ids"][run]): len(items)
                 for run, items in sorted(items_by_run.items())},
        "mismatch_bin_edges_train_only_mps": list(mismatch_edges),
        "signed_wheel_body_mismatch_stratification": bool(
            stratify_signed_wheel_mismatch),
        "requires_valid_raw_encoder_future_targets": bool(
            require_raw_encoder_targets),
        "signed_mismatch_bin_edges_mps": (
            list(signed_mismatch_edges)
            if stratify_signed_wheel_mismatch else None),
        "hierarchy": ["registered family weight", "whole run", "source condition",
                      "start-speed bin", "wheel/body-mismatch magnitude bin",
                      *( ["signed wheel/body-mismatch bin"]
                         if stratify_signed_wheel_mismatch else []),
                      "window"],
        "command_domain": f"future throttle command <={max_throttle_command:.6g}",
        "body_speed_domain_mps": [0.0, 12.0],
        "speed_bin_edges_mps": SPEED_EDGES.tolist(),
        "family_condition_sampler": sampler.metadata,
    }
    return sampler, report


def _sample_batch(data: dict[str, Any], state: np.ndarray,
                  accelerations: np.ndarray, sampler: FamilyConditionSampler,
                  batch_size: int, horizon_steps: int,
                  rng: np.random.Generator,
                  history_state_size: int,
                  raw_history_features: np.ndarray | None = None
                  ) -> tuple[np.ndarray, ...]:
    frames = data["frames"]
    histories, initial, delayed, commands, states, accel, poses = (
        [], [], [], [], [], [], [])
    for _ in range(batch_size):
        _, item, _, _ = sampler.sample(rng)
        sequence_id, start = item
        history_indices = np.arange(start - HISTORY_STEPS + 1, start + 1)
        future_indices = np.arange(start + 1, start + horizon_steps + 1)
        history = np.column_stack((state[history_indices, :history_state_size],
                                   frames[history_indices, 7:9]))
        if raw_history_features is not None:
            history = np.column_stack((history,
                                       raw_history_features[history_indices]))
        histories.append(history)
        initial.append(state[start])
        delayed.append(frames[history_indices[-2], 7:9])
        commands.append(frames[future_indices, 7:9])
        states.append(state[future_indices])
        accel_labels = accelerations[start:start + horizon_steps]
        if not np.isfinite(accel_labels).all():
            raise ValueError("a sampled training window has incomplete acceleration labels")
        accel.append(accel_labels)
        poses.append(data["simulator_pose_xyyaw"][
            np.concatenate(([start], future_indices))])
    return tuple(np.asarray(values, dtype=np.float32) for values in (
        histories, initial, delayed, commands, states, accel, poses))


def _reflect_training_batch(batch: tuple[np.ndarray, ...],
                            history_state_size: int
                            ) -> tuple[np.ndarray, ...]:
    """Reflect all vehicle channels across its longitudinal body axis."""
    histories, initial, delayed, commands, states, accel, poses = (
        np.asarray(value).copy() for value in batch)
    state_size = initial.shape[-1]
    if state_size not in (7, 9) or states.shape[-1] != state_size:
        raise ValueError("reflection requires the seven- or nine-state EDSSM layout")
    if not 7 <= history_state_size <= state_size:
        raise ValueError("history state width is incompatible with rollout state")
    if histories.shape[-1] < history_state_size + 2:
        raise ValueError("history is missing its steering/throttle commands")
    if delayed.shape[-1] != 2 or commands.shape[-1] != 2:
        raise ValueError("reflection requires steering/throttle command pairs")
    if accel.shape[-1] != 5 or poses.shape[-1] != 3:
        raise ValueError("reflection received an unsupported acceleration/pose layout")

    initial = reflect_state_channels(initial)
    states = reflect_state_channels(states)
    histories = reflect_history_features(histories, history_state_size)
    delayed = reflect_commands(delayed)
    commands = reflect_commands(commands)
    accel = reflect_acceleration_targets(accel)
    poses = reflect_pose_sequence(poses)
    return histories, initial, delayed, commands, states, accel, poses


def _random_reflection_augmentation(batch: tuple[np.ndarray, ...],
                                    history_state_size: int,
                                    rng: np.random.Generator
                                    ) -> tuple[np.ndarray, ...]:
    """Randomly mirror half the batch, preserving the original batch size."""
    mask = rng.random(len(batch[1])) < 0.5
    if not np.any(mask):
        return batch
    reflected = _reflect_training_batch(batch, history_state_size)
    result = []
    for original, mirrored in zip(batch, reflected):
        updated = np.asarray(original).copy()
        updated[mask] = mirrored[mask]
        result.append(updated)
    return tuple(result)


def _tensor(torch, value: np.ndarray, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _weighted_longitudinal_loss(torch, nn, state_error, target_state,
                                commands, low_throttle_weight: float):
    point_loss = nn.functional.smooth_l1_loss(
        state_error[:, :, 0], torch.zeros_like(state_error[:, :, 0]),
        beta=1.0, reduction="none")
    if low_throttle_weight == 1.0:
        return point_loss.mean()
    target_speed = torch.linalg.vector_norm(target_state[:, :, :2], dim=-1)
    low_throttle_moving = ((target_speed >= 3.0) & (target_speed <= 8.0)
                           & (commands[:, :, 1] <= 0.05))
    weights = torch.where(
        low_throttle_moving,
        torch.full_like(point_loss, low_throttle_weight),
        torch.ones_like(point_loss))
    return torch.sum(point_loss * weights) / torch.sum(weights)


def _weighted_body_state_loss(torch, nn, state_error, target_state,
                              commands, low_throttle_weight: float):
    forward_loss = _weighted_longitudinal_loss(
        torch, nn, state_error, target_state, commands,
        low_throttle_weight)
    other_state_loss = nn.functional.smooth_l1_loss(
        state_error[:, :, 1:5], torch.zeros_like(state_error[:, :, 1:5]),
        beta=1.0)
    return (forward_loss + 4.0 * other_state_loss) / 5.0


def _state_distillation_loss(torch, nn, student_state, teacher_state,
                             state_scale, channels=(2, 5, 6)):
    """Keep selected recursively predicted states near a frozen parent model."""
    channel_index = torch.as_tensor(channels, dtype=torch.long,
                                    device=student_state.device)
    scale = state_scale.index_select(0, channel_index)
    student = student_state.index_select(-1, channel_index) / scale
    teacher = teacher_state.index_select(-1, channel_index) / scale
    return nn.functional.smooth_l1_loss(
        student, teacher, beta=1.0)


def _state_distillation_channels(output_head_only: str | None,
                                 include_roll_state: bool = False):
    """Select parent states that isolate a targeted head's coupled effects."""
    if output_head_only == "wheel":
        return (0, 1, 2)  # forward speed, lateral speed, yaw rate
    if output_head_only in ("longitudinal", "longitudinal_pose"):
        channels = (1, 2, 5, 6)  # preserve cornering and wheel-state dynamics
        return channels + ((7, 8) if include_roll_state else ())
    if output_head_only == "yaw_pose":
        channels = (0, 1, 5, 6)  # preserve speed and wheel-state dynamics
        return channels + ((7, 8) if include_roll_state else ())
    if output_head_only == "longitudinal_wheel":
        return (1, 2)
    if output_head_only == "body_wheel":
        return (2,)
    return (2, 5, 6)


def _loss(torch, nn, model, batch: tuple[np.ndarray, ...], normalizers: dict[str, np.ndarray],
          device, wheel_acceleration_weight: float = 1.0,
          wheel_state_weight: float = LOSS_WEIGHTS["rear_wheel_state"],
          output_head_only: str | None = None,
          yaw_terminal_position_weight: float = 0.0,
          longitudinal_low_throttle_weight: float = 1.0,
          teacher_state=None, state_distillation_weight: float = 0.0,
          state_distillation_channels=(2, 5, 6),
          terminal_heading_loss_weight: float = 0.0,
          pose_secondary_weight: float = LOSS_WEIGHTS["pose_secondary"]):
    if not np.isfinite(wheel_acceleration_weight) or wheel_acceleration_weight < 0.0:
        raise ValueError("wheel acceleration loss weight must be finite and nonnegative")
    if not np.isfinite(wheel_state_weight) or wheel_state_weight < 0.0:
        raise ValueError("rear wheel state loss weight must be finite and nonnegative")
    if (not np.isfinite(yaw_terminal_position_weight)
            or yaw_terminal_position_weight < 0.0):
        raise ValueError("yaw terminal position loss weight must be finite and nonnegative")
    if (not np.isfinite(longitudinal_low_throttle_weight)
            or longitudinal_low_throttle_weight < 1.0):
        raise ValueError("low-throttle longitudinal loss weight must be finite and >=1")
    if (not np.isfinite(state_distillation_weight)
            or state_distillation_weight < 0.0
            or (state_distillation_weight > 0.0 and teacher_state is None)):
        raise ValueError("state distillation needs a teacher rollout and nonnegative weight")
    if (not np.isfinite(terminal_heading_loss_weight)
            or terminal_heading_loss_weight < 0.0
            or (terminal_heading_loss_weight > 0.0
                and output_head_only not in (
                    "longitudinal_pose", "longitudinal_wheel"))):
        raise ValueError("terminal heading loss is restricted to longitudinal pose-aware head training")
    if (not np.isfinite(pose_secondary_weight)
            or pose_secondary_weight < 0.0):
        raise ValueError("pose secondary loss weight must be finite and nonnegative")
    histories, initial, delayed, commands, targets, acceleration_targets, pose_targets = batch
    history_tensor = _tensor(torch, histories, device)
    initial_tensor = _tensor(torch, initial, device)
    delayed_tensor = _tensor(torch, delayed, device)
    command_tensor = _tensor(torch, commands, device)
    target_tensor = _tensor(torch, targets, device)
    accel_target_tensor = _tensor(torch, acceleration_targets, device)
    pose_target_tensor = _tensor(torch, pose_targets, device)
    predicted, predicted_acceleration, predicted_latent, _ = model.rollout(
        initial_tensor, delayed_tensor, history_tensor, command_tensor)
    state_scale = _tensor(torch, normalizers["state_scale"], device)
    distillation_loss = predicted.new_zeros(())
    if teacher_state is not None:
        distillation_loss = _state_distillation_loss(
            torch, nn, predicted, teacher_state, state_scale,
            state_distillation_channels)
    acceleration_bounds = _tensor(torch, normalizers["acceleration_bounds"], device)
    state_error = (predicted - target_tensor) / state_scale
    body_loss = _weighted_body_state_loss(
        torch, nn, state_error, target_tensor, command_tensor,
        longitudinal_low_throttle_weight)
    wheel_loss = nn.functional.smooth_l1_loss(
        state_error[:, :, 5:7], torch.zeros_like(state_error[:, :, 5:7]), beta=1.0)
    roll_state_loss = (nn.functional.smooth_l1_loss(
        state_error[:, :, 7:9], torch.zeros_like(state_error[:, :, 7:9]),
        beta=1.0) if targets.shape[-1] == 9 else state_error.new_zeros(()))
    acceleration_error = ((predicted_acceleration - accel_target_tensor)
                          / acceleration_bounds)
    body_acceleration_loss = nn.functional.smooth_l1_loss(
        acceleration_error[:, :, :3],
        torch.zeros_like(acceleration_error[:, :, :3]), beta=1.0)
    wheel_acceleration_loss = nn.functional.smooth_l1_loss(
        acceleration_error[:, :, 3:],
        torch.zeros_like(acceleration_error[:, :, 3:]), beta=1.0)
    # Preserve the original five-channel mean exactly at weight=1. The
    # optional zero setting is for encoder-rate labels: each wheel-speed
    # observation is already a 100 ms angle difference, so its 25 ms finite
    # difference is a noisy second-difference rather than direct wheel torque
    # evidence. Recursive wheel-state and whole-body losses remain active.
    acceleration_loss = (
        3.0 * body_acceleration_loss
        + 2.0 * wheel_acceleration_weight * wheel_acceleration_loss
    ) / (3.0 + 2.0 * wheel_acceleration_weight)
    pose = integrate_pose(
        torch, predicted, pose_target_tensor[:, 0], initial_tensor)
    pose_position_error = (pose[:, :, :2] - pose_target_tensor[:, 1:, :2]) / 0.20
    pose_heading_error = torch.atan2(
        torch.sin(pose[:, :, 2] - pose_target_tensor[:, 1:, 2]),
        torch.cos(pose[:, :, 2] - pose_target_tensor[:, 1:, 2])) / 0.10
    terminal_heading_loss = nn.functional.smooth_l1_loss(
        pose_heading_error[:, -1],
        torch.zeros_like(pose_heading_error[:, -1]), beta=1.0)
    pose_loss = (
        nn.functional.smooth_l1_loss(
            pose_position_error, torch.zeros_like(pose_position_error), beta=1.0)
        + nn.functional.smooth_l1_loss(
            pose_heading_error, torch.zeros_like(pose_heading_error), beta=1.0))
    latent_loss = torch.mean(predicted_latent ** 2)
    total = (LOSS_WEIGHTS["body_state"] * body_loss
             + wheel_state_weight * wheel_loss
             + (LOSS_WEIGHTS["roll_state"] * roll_state_loss
                if targets.shape[-1] == 9 else 0.0)
             + LOSS_WEIGHTS["effective_acceleration"] * acceleration_loss
             + pose_secondary_weight * pose_loss
             + LOSS_WEIGHTS["latent_l2"] * latent_loss
             + state_distillation_weight * distillation_loss)
    if output_head_only == "wheel":
        # Isolate encoder-wheel fitting from the shared body/latent representation.
        total = (wheel_state_weight * wheel_loss
                 + state_distillation_weight * distillation_loss)
    elif output_head_only == "yaw":
        yaw_state_loss = nn.functional.smooth_l1_loss(
            state_error[:, :, 2], torch.zeros_like(state_error[:, :, 2]),
            beta=1.0)
        heading_loss = nn.functional.smooth_l1_loss(
            pose_heading_error, torch.zeros_like(pose_heading_error), beta=1.0)
        terminal_position_loss = nn.functional.smooth_l1_loss(
            pose_position_error[:, -1],
            torch.zeros_like(pose_position_error[:, -1]), beta=1.0)
        # Preserve the wheel improvement from a prior wheel-head stage while
        # fitting yaw/heading through only the yaw-acceleration output head.
        total = (yaw_state_loss + 0.5 * heading_loss
                 + yaw_terminal_position_weight * terminal_position_loss
                 + wheel_state_weight * wheel_loss)
    elif output_head_only == "yaw_pose":
        yaw_state_loss = nn.functional.smooth_l1_loss(
            state_error[:, :, 2], torch.zeros_like(state_error[:, :, 2]),
            beta=1.0)
        heading_loss = nn.functional.smooth_l1_loss(
            pose_heading_error, torch.zeros_like(pose_heading_error), beta=1.0)
        terminal_position_loss = nn.functional.smooth_l1_loss(
            pose_position_error[:, -1],
            torch.zeros_like(pose_position_error[:, -1]), beta=1.0)
        total = (yaw_state_loss + 0.5 * heading_loss
                 + pose_secondary_weight * pose_loss
                 + yaw_terminal_position_weight * terminal_position_loss
                 + state_distillation_weight * distillation_loss)
    elif output_head_only == "longitudinal":
        # Isolate the forward-motion equation after the residual attribution
        # found a repeatable low-throttle practice bias. Only the ax output
        # head is trainable in this mode; the loss is the recursively predicted
        # COM forward-velocity trajectory, not measured future feedback.
        # Optional parent distillation constrains the coupled v/yaw/wheel
        # rollout without giving those target heads trainable parameters.
        total = (_weighted_longitudinal_loss(
                    torch, nn, state_error, target_tensor, command_tensor,
                    longitudinal_low_throttle_weight)
                 + state_distillation_weight * distillation_loss)
    elif output_head_only == "longitudinal_pose":
        # Tune only forward acceleration against recursive speed and pose.
        # Distillation keeps lateral/yaw, wheel and internally predicted roll
        # states near the stronger parent so an axial correction cannot
        # silently trade away its cornering behavior.
        total = (_weighted_longitudinal_loss(
                    torch, nn, state_error, target_tensor, command_tensor,
                    longitudinal_low_throttle_weight)
                 + pose_secondary_weight * pose_loss
                 + terminal_heading_loss_weight * terminal_heading_loss
                 + state_distillation_weight * distillation_loss)
    elif output_head_only == "longitudinal_wheel":
        total = (_weighted_longitudinal_loss(
                    torch, nn, state_error, target_tensor, command_tensor,
                    longitudinal_low_throttle_weight)
                 + wheel_state_weight * wheel_loss
                 + pose_secondary_weight * pose_loss
                 + LOSS_WEIGHTS["latent_l2"] * latent_loss
                 + state_distillation_weight * distillation_loss
                 + terminal_heading_loss_weight * terminal_heading_loss)
    elif output_head_only in ("body", "body_wheel"):
        # Refine COM/yaw dynamics without changing the shared representation
        # or wheel-output heads. Keep recursive wheel-state and pose losses
        # active because the latent state couples later wheel predictions to
        # earlier body accelerations.
        total = (body_loss
                 + wheel_state_weight * wheel_loss
                 + pose_secondary_weight * pose_loss
                 + LOSS_WEIGHTS["latent_l2"] * latent_loss
                 + state_distillation_weight * distillation_loss)
    if (output_head_only not in (
            None, "body_wheel", "wheel", "longitudinal",
            "longitudinal_pose", "longitudinal_wheel", "yaw_pose")
            and state_distillation_weight > 0.0):
        raise ValueError("state distillation requires full-model or a supported isolated-head mode")
    components = {
        "body_state": body_loss.detach(),
        "rear_wheel_state": wheel_loss.detach(),
        "roll_state": roll_state_loss.detach(),
        "effective_acceleration": acceleration_loss.detach(),
        "body_effective_acceleration": body_acceleration_loss.detach(),
        "wheel_effective_acceleration": wheel_acceleration_loss.detach(),
        "pose_secondary": pose_loss.detach(),
        "latent_l2": latent_loss.detach(),
        "state_distillation": distillation_loss.detach(),
        "terminal_heading": terminal_heading_loss.detach(),
    }
    return total, components


def _selection_score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    horizons = metrics["horizons"]
    long_horizon = "10s" in horizons
    recursive_horizons = ("2s", "5s", "10s") if long_horizon else ("2s", "5s")
    body_key = "body_recursive_2_10s" if long_horizon else "body_recursive_2_5s"
    wheel_key = "wheel_recursive_2_10s" if long_horizon else "wheel_recursive_2_5s"
    pose_key = "pose_recursive_2_10s" if long_horizon else "pose_recursive_2_5s"
    components = {
        "body_0.75s": horizons["0.75s"]["body"]["macro_run_mean"],
        body_key: float(np.mean([
            horizons[horizon]["body"]["macro_run_mean"]
            for horizon in recursive_horizons])),
        wheel_key: float(np.mean([
            horizons[horizon]["wheel"]["macro_run_mean"]
            for horizon in recursive_horizons])),
        pose_key: (1.0 / (2.0 * len(recursive_horizons))) * sum(
            horizons[horizon][metric]["macro_run_mean"] / scale
            for horizon in recursive_horizons
            for metric, scale in (("position_m", 0.20),
                                  ("heading_rad", 0.10))),
    }
    hard_values = []
    for regime in ("high_steering", "zero_throttle",
                   "low_throttle_moving", "wheel_mismatch_proxy"):
        entry = metrics["hard_regimes"].get(regime, {}).get("horizons", {})
        if ("0.75s" in entry and entry["0.75s"]["body"]["macro_run_mean"] is not None
                and entry["2s"]["body"]["macro_run_mean"] is not None):
            hard_values.append(0.5 * (
                entry["0.75s"]["body"]["macro_run_mean"]
                + entry["2s"]["body"]["macro_run_mean"]))
    components["hard_regime_macro"] = (
        float(np.mean(hard_values)) if hard_values
        else components[body_key])
    weights = {
        "body_0.75s": 0.30,
        body_key: 0.25,
        wheel_key: 0.15,
        pose_key: 0.15,
        "hard_regime_macro": 0.15,
    }
    return float(sum(weights[name] * components[name] for name in weights)), components


def _checkpoint_payload(model, metadata: dict[str, Any], optimizer,
                        step: int, score_value: float,
                        torch) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metadata": metadata,
        "model_state_dict": {name: value.detach().cpu()
                             for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": int(step),
        "validation_selection_score": float(score_value),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }


def train(dataset_path: Path, output_dir: Path, encoder: str, seed: int,
          sampling_seed: int = 20261009, stage_steps: int = 400,
          batch_size: int = 32, eval_every: int = 200,
          learning_rate: float = 3e-4, device_name: str = "cuda",
          latent_size: int = LATENT_SIZE, expert_count: int = 1,
          wheel_acceleration_weight: float = 1.0,
          include_roll_state: bool = False,
          couple_roll_acceleration: bool = False,
          roll_residual_mode: bool = False,
          fit_roll_oscillator_state: bool = False,
          body_acceleration_source: str = "state_derivative",
          init_checkpoint: Path | None = None,
          wheel_state_weight: float = LOSS_WEIGHTS["rear_wheel_state"],
          output_head_only: str | None = None,
          yaw_terminal_position_weight: float = 0.0,
          stratify_signed_wheel_mismatch: bool = False,
          wheel_state_source: str = "filtered_odometry",
          include_raw_encoder_history: bool = False,
          include_wheel_innovation_history: bool = False,
          max_rollout_steps: int = 200,
          longitudinal_low_throttle_weight: float = 1.0,
          state_distillation_weight: float = 0.0,
          terminal_heading_loss_weight: float = 0.0,
          max_throttle_command: float = 0.50,
          distillation_teacher_checkpoint: Path | None = None,
          pose_secondary_weight: float = LOSS_WEIGHTS["pose_secondary"],
          wheel_dynamics_mode: str = "surface_acceleration",
          turn_reflection_augmentation: bool = False,
          allow_append_only_dataset_extension: bool = False
          ) -> dict[str, Any]:
    dataset_path, output_dir = dataset_path.resolve(), output_dir.resolve()
    if init_checkpoint is not None:
        init_checkpoint = init_checkpoint.resolve()
    if distillation_teacher_checkpoint is not None:
        distillation_teacher_checkpoint = distillation_teacher_checkpoint.resolve()
    roll_residual_head_finetune = (
        output_head_only in ("body", "body_wheel", "wheel", "yaw",
                             "longitudinal", "longitudinal_pose",
                             "longitudinal_wheel", "yaw_pose")
        and include_roll_state and roll_residual_mode)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse EDSSM training directory: {output_dir}")
    if (encoder not in ("gru", "tcn") or latent_size not in (16, 32, 64)
            or expert_count not in (1, 2, 3)
            or stage_steps <= 0 or batch_size <= 0
            or max_rollout_steps not in (200, 400)
            or not np.isfinite(wheel_acceleration_weight)
            or wheel_acceleration_weight < 0.0
            or not np.isfinite(wheel_state_weight)
            or wheel_state_weight < 0.0
            or not np.isfinite(yaw_terminal_position_weight)
            or yaw_terminal_position_weight < 0.0
            or (yaw_terminal_position_weight > 0.0
                and output_head_only not in ("yaw", "yaw_pose"))
            or not np.isfinite(longitudinal_low_throttle_weight)
            or not 1.0 <= longitudinal_low_throttle_weight <= 10.0
            or (longitudinal_low_throttle_weight > 1.0
                and output_head_only not in (
                     None, "longitudinal", "body_wheel",
                    "longitudinal_pose", "longitudinal_wheel"))
            or not np.isfinite(state_distillation_weight)
            or not 0.0 <= state_distillation_weight <= 10.0
            or not np.isfinite(terminal_heading_loss_weight)
            or not 0.0 <= terminal_heading_loss_weight <= 10.0
            or not np.isfinite(pose_secondary_weight)
            or not 0.0 <= pose_secondary_weight <= 10.0
            or not np.isfinite(max_throttle_command)
            or not 0.0 <= max_throttle_command <= 1.0
            or (terminal_heading_loss_weight > 0.0
                and output_head_only not in (
                    "longitudinal_pose", "longitudinal_wheel"))
            or (state_distillation_weight > 0.0
                and (init_checkpoint is None
                     or output_head_only not in (
                         None, "body_wheel", "wheel", "longitudinal",
                         "longitudinal_pose", "longitudinal_wheel",
                         "yaw_pose")))
            or (distillation_teacher_checkpoint is not None
                and state_distillation_weight <= 0.0)
            or (couple_roll_acceleration and not include_roll_state)
            or (roll_residual_mode and not include_roll_state)
            or (fit_roll_oscillator_state and not include_roll_state)
            or (roll_residual_mode and init_checkpoint is None)
            or (allow_append_only_dataset_extension
                and (init_checkpoint is None or not roll_residual_mode))
            or (init_checkpoint is not None
                and not (roll_residual_mode or output_head_only is not None
                         or stratify_signed_wheel_mismatch))
            or (stratify_signed_wheel_mismatch and init_checkpoint is None)
            or (output_head_only not in (
                None, "body", "body_wheel", "wheel", "yaw",
                "longitudinal", "longitudinal_pose",
                "longitudinal_wheel", "yaw_pose", "roll_residual"))
            or (output_head_only is not None
                and init_checkpoint is None)
            or (output_head_only in (
                    "body", "body_wheel", "wheel", "yaw", "longitudinal",
                    "longitudinal_pose", "longitudinal_wheel", "yaw_pose")
                and (include_roll_state or roll_residual_mode)
                and not roll_residual_head_finetune)
            or (output_head_only == "roll_residual"
                and not (roll_residual_mode and fit_roll_oscillator_state))
            or body_acceleration_source not in (
                "state_derivative", "packet_interval_mean")
            or wheel_dynamics_mode not in WHEEL_DYNAMICS_MODES
            or wheel_state_source not in ("filtered_odometry", "raw_encoder")
            or (include_raw_encoder_history
                and include_wheel_innovation_history)):
        raise ValueError("invalid EDSSM encoder/training dimensions")
    torch, nn = _torch()
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    data = _load_dataset(dataset_path)
    dataset_extension = None
    if allow_append_only_dataset_extension:
        source_checkpoint = torch.load(
            init_checkpoint, map_location="cpu", weights_only=False)
        dataset_extension = _verify_append_only_dataset_extension(
            source_checkpoint.get("metadata", {}), dataset_path, data)
    if device_name == "cpu":
        torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if (int(data["schema_version"]) != 9
            or np.any(~np.isin(data["splits"], ("train", "validation", "test", "final_test")))):
        raise ValueError("EDSSM requires the frozen schema-9 whole-run dataset")
    curriculum = CURRICULUM
    if max_rollout_steps == 400:
        curriculum += (("0.75s_plus_2s_plus_5s_plus_10s",
                        (30, 80, 200, 400)),)
    physical_state = physical_state_from_dataset(
        data, wheel_state_source=wheel_state_source).astype(np.float32)
    if wheel_state_source == "raw_encoder" and data.get("encoder_raw_valid") is None:
        raise ValueError("raw_encoder training requires the verified angle sidecar")
    raw_history_features = (
        raw_encoder_history_features(data) if include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if include_wheel_innovation_history else None)
    state = (append_roll_state(data, physical_state)
             if include_roll_state else physical_state)
    roll_oscillator_coefficients = None
    roll_oscillator_report = None
    if fit_roll_oscillator_state:
        (roll_oscillator_coefficients,
         roll_oscillator_report) = fit_roll_oscillator(data, physical_state)
    acceleration_targets = acceleration_targets_from_dataset(
        data, physical_state, body_acceleration_source,
        wheel_dynamics_mode).astype(np.float32)
    normalizers = _robust_normalizers(
        data, state, acceleration_targets,
        history_state=(physical_state if roll_residual_mode else state),
        history_extra=raw_history_features,
        require_raw_encoder_valid=(wheel_state_source == "raw_encoder"),
        max_throttle_command=max_throttle_command)
    actuator_fit = fit_actuator_dynamics(data)
    training_runs = np.asarray(data["run_ids"])[
        np.asarray(data["splits"]).astype(str) == "train"].astype(str).tolist()
    validation_runs = np.asarray(data["run_ids"])[
        np.asarray(data["splits"]).astype(str) == "validation"].astype(str).tolist()
    if set(training_runs).intersection(validation_runs):
        raise ValueError("whole-run training and validation sets overlap")

    max_horizon = max_rollout_steps
    _, sampler_metadata = _build_window_sampler(
        data, state, max_horizon, "train",
        stratify_signed_wheel_mismatch=stratify_signed_wheel_mismatch,
        require_raw_encoder_targets=(wheel_state_source == "raw_encoder"),
        max_throttle_command=max_throttle_command)
    mismatch_edges = tuple(sampler_metadata[
        "mismatch_bin_edges_train_only_mps"])
    samplers: dict[int, tuple[FamilyConditionSampler, dict[str, Any]]] = {
        max_steps: _build_window_sampler(
            data, state, max_steps, "train", mismatch_edges=mismatch_edges,
            stratify_signed_wheel_mismatch=stratify_signed_wheel_mismatch,
            require_raw_encoder_targets=(wheel_state_source == "raw_encoder"),
            max_throttle_command=max_throttle_command)
        for max_steps in sorted({steps for _, horizons in curriculum
                                 for steps in horizons})
    }
    validation_horizons = {name: steps for name, steps in (
        ("0.75s", 30), ("2s", 80), ("5s", 200))}
    if max_rollout_steps > 200:
        validation_horizons["10s"] = max_rollout_steps
    validation_windows = _fixed_validation_windows(
        data, state, horizon_steps=max_horizon,
        max_throttle_command=max_throttle_command)
    validation_report_sampler = {
        horizon_name: {
            "training_windows": samplers[steps][1],
        }
        for horizon_name, steps in validation_horizons.items()
    }

    arrays = {name: normalizers[name] for name in (
        "history_mean", "history_scale", "state_mean", "state_scale",
        "command_mean", "command_scale", "acceleration_bounds")}
    model_class = effective_model_type(
        torch, nn, arrays["history_mean"], arrays["history_scale"],
        arrays["state_mean"], arrays["state_scale"],
        arrays["command_mean"], arrays["command_scale"],
        arrays["acceleration_bounds"], actuator_fit,
        encoder=encoder, latent_size=latent_size,
        expert_count=expert_count, include_roll_state=include_roll_state,
        couple_roll_acceleration=couple_roll_acceleration,
        roll_residual_mode=roll_residual_mode,
        roll_oscillator_coefficients=roll_oscillator_coefficients,
        include_raw_encoder_history=include_raw_encoder_history,
        include_wheel_innovation_history=include_wheel_innovation_history,
        wheel_dynamics_mode=wheel_dynamics_mode)
    model = model_class().to(device_name)
    warm_start = None
    if output_head_only in (
            "body", "body_wheel", "wheel", "yaw", "longitudinal",
            "longitudinal_pose", "longitudinal_wheel", "yaw_pose"):
        source = torch.load(init_checkpoint, map_location="cpu",
                            weights_only=False)
        source_metadata = source.get("metadata", {})
        same_training_dataset = (
            source_metadata.get("dataset_sha256") == _sha256(dataset_path))
        fixed_cadence_transfer = (
            output_head_only == "wheel"
            and is_fixed_cadence_variant(source_metadata, dataset_path))
        if (not (same_training_dataset or fixed_cadence_transfer
                or dataset_extension is not None)
                or source_metadata.get("encoder") != encoder
                or int(source_metadata.get("latent_size", -1)) != latent_size
                or int(source_metadata.get("expert_count", 1)) != expert_count
                or bool(source_metadata.get("include_roll_state", False))
                    != include_roll_state
                or bool(source_metadata.get("roll_residual_mode", False))
                    != roll_residual_mode
                or bool(source_metadata.get("couple_roll_acceleration", False))
                    != couple_roll_acceleration
                or ((source_metadata.get("roll_oscillator_fit") is not None)
                    != fit_roll_oscillator_state)
                or source_metadata.get("wheel_state_source",
                                       "filtered_odometry") != wheel_state_source
                or source_metadata.get("wheel_dynamics_mode",
                                       "surface_acceleration") != wheel_dynamics_mode
                or bool(source_metadata.get("include_raw_encoder_history", False))
                    != include_raw_encoder_history
                or bool(source_metadata.get(
                    "include_wheel_innovation_history", False))
                    != include_wheel_innovation_history
                or float(source_metadata.get(
                    "wheel_acceleration_loss_weight", 1.0))
                    != wheel_acceleration_weight
                or source_metadata.get("acceleration_target_source",
                                       "state_derivative")
                    != body_acceleration_source
                or source_metadata.get("wheel_dynamics_mode",
                                       "surface_acceleration") != wheel_dynamics_mode):
            raise ValueError("output-head checkpoint does not match the frozen model configuration")
        model.load_state_dict(source["model_state_dict"], strict=True)
        if dataset_extension is not None:
            actuator_fit = _inherit_checkpoint_calibration(
                model, normalizers, arrays, source_metadata)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        output_indices = {
            "body": (0, 1, 2),
            "body_wheel": (0, 1, 3, 4),
            "wheel": (3, 4),
            "yaw": (2,),
            "yaw_pose": (2,),
            "longitudinal": (0,),
            "longitudinal_pose": (0,),
            "longitudinal_wheel": (0, 3, 4),
        }[output_head_only]
        if expert_count == 1:
            selected_heads = [model.acceleration_heads[index]
                              for index in output_indices]
        else:
            selected_heads = [head
                              for expert_heads in model.expert_acceleration_heads
                              for index, head in enumerate(expert_heads)
                              if index in output_indices]
        for head in selected_heads:
            for parameter in head.parameters():
                parameter.requires_grad_(True)
        if not all(parameter.requires_grad
                   for head in selected_heads for parameter in head.parameters()):
            raise ValueError("selected acceleration heads were not enabled for fine-tuning")
        warm_start = {
            "checkpoint": str(init_checkpoint),
            "sha256": _sha256(init_checkpoint),
            "mode": (f"{output_head_only} acceleration head(s) only; "
                     "all non-selected parameters frozen"),
            "dataset_compatibility": (
                "exact hash match" if same_training_dataset else
                "verified append-only whole-run extension" if
                dataset_extension is not None else
                "verified fixed-25ms raw encoder-rate variant; all source "
                "arrays and validity masks match"),
            "dataset_extension_proof": dataset_extension,
            "normalizer_policy": (
                "retain verified parent training-only normalizers and "
                "actuator response" if dataset_extension is not None else None),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad),
        }
    elif roll_residual_mode:
        source = torch.load(init_checkpoint, map_location="cpu",
                            weights_only=False)
        source_metadata = source.get("metadata", {})
        same_training_dataset = (
            source_metadata.get("dataset_sha256") == _sha256(dataset_path))
        if (not (same_training_dataset or dataset_extension is not None)
                or source_metadata.get("encoder") != encoder
                or int(source_metadata.get("latent_size", -1)) != latent_size
                or int(source_metadata.get("expert_count", 1)) != expert_count
                or float(source_metadata.get(
                    "wheel_acceleration_loss_weight", 1.0)) != wheel_acceleration_weight
                or source_metadata.get("acceleration_target_source", "state_derivative")
                    != body_acceleration_source
                or source_metadata.get("wheel_state_source",
                                       "filtered_odometry") != wheel_state_source
                or source_metadata.get("wheel_dynamics_mode",
                                       "surface_acceleration") != wheel_dynamics_mode
                or bool(source_metadata.get("include_raw_encoder_history", False))
                    != include_raw_encoder_history
                or bool(source_metadata.get(
                    "include_wheel_innovation_history", False))
                    != include_wheel_innovation_history):
            raise ValueError("warm-start checkpoint uses different data/model targets")
        if (source_metadata.get("include_roll_state", False)
                and source_metadata.get("roll_residual_mode", False)):
            if (not include_roll_state
                    or (source_metadata.get("couple_roll_acceleration", False)
                        != couple_roll_acceleration)
                    or source_metadata.get("roll_oscillator_fit") is not None):
                raise ValueError("roll-state continuation settings do not match checkpoint")
            model.load_state_dict(source["model_state_dict"], strict=True)
            if dataset_extension is not None:
                actuator_fit = _inherit_checkpoint_calibration(
                    model, normalizers, arrays, source_metadata)
            warm_start = {
                "checkpoint": str(init_checkpoint),
                "sha256": _sha256(init_checkpoint),
                "mode": "full-model continuation of the matching predicted-roll residual teacher",
                "dataset_extension_proof": dataset_extension,
                "normalizer_policy": (
                    "retain verified parent training-only normalizers and "
                    "actuator response; update learned dynamics on appended run"
                    if dataset_extension is not None else "unchanged"),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad),
            }
        else:
            if source_metadata.get("include_roll_state", False):
                raise ValueError("roll residual can only be added to a matching 7-state lead")
            source_state = source["model_state_dict"]
            target_state = model.state_dict()
            shape_mismatch = sorted(
                name for name, value in source_state.items()
                if name not in target_state or target_state[name].shape != value.shape)
            if shape_mismatch != ["state_mean", "state_scale"]:
                raise ValueError(
                    f"warm-start model architecture mismatch: {shape_mismatch}")
            transferable = {
                name: value for name, value in source_state.items()
                if name in target_state and target_state[name].shape == value.shape
            }
            load_result = model.load_state_dict(transferable, strict=False)
            allowed_missing = {
                "state_mean", "state_scale",
                "roll_force_residual.0.weight", "roll_force_residual.0.bias",
                "roll_force_residual.2.weight", "roll_force_residual.2.bias",
                "roll_oscillator_coefficients",
            }
            if set(load_result.missing_keys) != allowed_missing or load_result.unexpected_keys:
                raise ValueError("warm-start transfer did not preserve the complete base model")
            warm_start = {
                "checkpoint": str(init_checkpoint),
                "sha256": _sha256(init_checkpoint),
                "transferred_parameter_and_buffer_count": len(transferable),
                "newly_initialized_keys": sorted(load_result.missing_keys),
                "mode": "add internal-roll residual to matching frozen 7-state lead",
            }
    elif init_checkpoint is not None:
        source = torch.load(init_checkpoint, map_location="cpu",
                            weights_only=False)
        source_metadata = source.get("metadata", {})
        if (source_metadata.get("dataset_sha256") != _sha256(dataset_path)
                or source_metadata.get("include_roll_state", False)
                or source_metadata.get("encoder") != encoder
                or int(source_metadata.get("latent_size", -1)) != latent_size
                or int(source_metadata.get("expert_count", 1)) != expert_count
                or float(source_metadata.get(
                    "wheel_acceleration_loss_weight", 1.0))
                    != wheel_acceleration_weight
                or source_metadata.get("acceleration_target_source",
                                       "state_derivative")
                    != body_acceleration_source):
            raise ValueError("full-model warm start is not the matching frozen 7-state model")
        model.load_state_dict(source["model_state_dict"], strict=True)
        warm_start = {
            "checkpoint": str(init_checkpoint),
            "sha256": _sha256(init_checkpoint),
            "mode": "full-model fine-tuning with signed wheel/body-mismatch-stratified windows",
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad),
        }
    if output_head_only == "roll_residual":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.roll_force_residual.parameters():
            parameter.requires_grad_(True)
        warm_start = {
            **(warm_start or {}),
            "mode": "roll acceleration residual only; base/actuator/latent frozen",
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad),
        }
    teacher_model = None
    teacher_checkpoint = None
    if state_distillation_weight > 0.0:
        teacher_path = (distillation_teacher_checkpoint or init_checkpoint)
        state_distillation_channels = _state_distillation_channels(
            output_head_only, include_roll_state)
        state_names = PHYSICAL_STATE_NAMES + ROLL_STATE_NAMES
        teacher_checkpoint = {
            "path": str(teacher_path),
            "sha256": _sha256(teacher_path),
            "channels": [state_names[index]
                         for index in state_distillation_channels],
        }
        teacher_model = model_class().to(device_name)
        teacher_checkpoint_data = torch.load(
            teacher_path, map_location="cpu", weights_only=False)
        teacher_metadata = teacher_checkpoint_data.get("metadata", {})
        if (teacher_metadata.get("dataset_sha256") != _sha256(dataset_path)
                or teacher_metadata.get("encoder") != encoder
                or int(teacher_metadata.get("latent_size", -1)) != latent_size
                or int(teacher_metadata.get("expert_count", 1)) != expert_count
                or bool(teacher_metadata.get("include_roll_state", False))
                    != include_roll_state
                or bool(teacher_metadata.get("roll_residual_mode", False))
                    != roll_residual_mode
                or teacher_metadata.get("wheel_state_source",
                                        "filtered_odometry") != wheel_state_source
                or teacher_metadata.get("wheel_dynamics_mode",
                                        "surface_acceleration") != wheel_dynamics_mode
                or bool(teacher_metadata.get("include_raw_encoder_history", False))
                    != include_raw_encoder_history
                or bool(teacher_metadata.get(
                    "include_wheel_innovation_history", False))
                    != include_wheel_innovation_history):
            raise ValueError("distillation checkpoint does not match the frozen model architecture and dataset")
        teacher_model.load_state_dict(
            teacher_checkpoint_data["model_state_dict"], strict=True)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
    trainable_parameters = [parameter for parameter in model.parameters()
                            if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("EDSSM has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate,
                                  weight_decay=1e-5)
    sample_rng = np.random.default_rng(sampling_seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training_log.jsonl"
    metadata = {
        "model": "deterministic_effective_race_teacher",
        "encoder": encoder,
        "latent_size": int(latent_size),
        "expert_count": int(expert_count),
        "expert_gate_inputs": (
            "normalized predicted physical state, normalized command, latent"
            + (", rear longitudinal slip in m/s"
               if wheel_dynamics_mode == "contact_slip" else "")
            if expert_count > 1 else None),
        "history_steps": HISTORY_STEPS,
        "history_seconds": HISTORY_STEPS * DT_S,
        "future_inputs": list(COMMAND_NAMES),
        "physical_state_names": list(PHYSICAL_STATE_NAMES + (
            ROLL_STATE_NAMES if include_roll_state else ())),
        "wheel_state_source": wheel_state_source,
        "wheel_dynamics_mode": wheel_dynamics_mode,
        "include_raw_encoder_history": bool(include_raw_encoder_history),
        "include_wheel_innovation_history": bool(
            include_wheel_innovation_history),
        "include_roll_state": bool(include_roll_state),
        "couple_roll_acceleration": bool(couple_roll_acceleration),
        "roll_residual_mode": bool(roll_residual_mode),
        "roll_oscillator_fit": roll_oscillator_report,
        "warm_start": warm_start,
        "training_objective_mode": (
            ("longitudinal_pose_head_only_recursive_path" if
             output_head_only == "longitudinal_pose" else
             "yaw_pose_head_only_recursive_path" if
             output_head_only == "yaw_pose" else
             f"{output_head_only}_head_only" if output_head_only else
            "joint_state_pose_with_low_throttle_forward_weight_and_yaw_wheel_distillation"
            if state_distillation_weight > 0.0 else
             "joint_state_pose_with_low_throttle_forward_weight"
             if longitudinal_low_throttle_weight > 1.0 else
             "joint_state_and_pose")
            ),
        "history_feature_names": list(
            (PHYSICAL_STATE_NAMES if roll_residual_mode else
             PHYSICAL_STATE_NAMES + (ROLL_STATE_NAMES if include_roll_state else ()))
            + COMMAND_NAMES
            + (RAW_ENCODER_HISTORY_NAMES
               if include_raw_encoder_history else ())
            + (WHEEL_INNOVATION_HISTORY_NAMES
               if include_wheel_innovation_history else ())),
        "acceleration_target_names": list(
            acceleration_target_names(wheel_dynamics_mode)),
        "acceleration_target_source": body_acceleration_source,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_schema_version": int(data["schema_version"]),
        "dataset_role": "replacement_teacher_dataset_v1",
        "training_runs": training_runs,
        "validation_runs": validation_runs,
        "seed": int(seed),
        "sampling_seed": int(sampling_seed),
        "curriculum": [{"stage": name, "horizons_steps": list(steps),
                        "steps": stage_steps}
                       for name, steps in curriculum],
        "max_recursive_rollout_steps": int(max_rollout_steps),
        "max_throttle_command_norm": float(max_throttle_command),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "optimizer": "AdamW(weight_decay=1e-5)",
        "loss_weights": {**LOSS_WEIGHTS,
                         "rear_wheel_state": float(wheel_state_weight),
                         "pose_secondary": float(pose_secondary_weight)},
        "wheel_acceleration_loss_weight": float(wheel_acceleration_weight),
        "wheel_dynamics_mode": wheel_dynamics_mode,
        "yaw_terminal_position_loss_weight": float(yaw_terminal_position_weight),
        "longitudinal_low_throttle_loss_weight": float(
            longitudinal_low_throttle_weight),
        "terminal_heading_loss_weight": float(terminal_heading_loss_weight),
        "state_distillation_loss_weight": float(state_distillation_weight),
        "pose_secondary_loss_weight": float(pose_secondary_weight),
        "state_distillation_teacher": teacher_checkpoint,
        "stratify_signed_wheel_mismatch": bool(
            stratify_signed_wheel_mismatch),
        "turn_reflection_augmentation": bool(turn_reflection_augmentation),
        "turn_reflection_augmentation_policy": (
            "samplewise 50% longitudinal-axis reflection; lateral/yaw/steering and roll signs reverse; rear wheels swap; world pose reflected about each window's initial body axis"
            if turn_reflection_augmentation else None),
        "roll_state_equation": (
            "roll_next = roll + 0.5*dt*(roll_rate + predicted_roll_rate_next)"
            if include_roll_state else None),
        "roll_rate_equation": (
            roll_oscillator_report["equation"]
            if roll_oscillator_report is not None else None),
        "roll_rate_acceleration_coupling": (
            "predicted body-frame lateral acceleration ay enters roll-rate transition"
            if couple_roll_acceleration else None),
        "roll_training_inputs": (
            "initial measured roll/rate; future rollout uses only predicted roll/rate"
            if include_roll_state else None),
        "selection_score_weights": {
            "body_0.75s": 0.30,
            ("body_recursive_2_10s" if max_rollout_steps > 200
             else "body_recursive_2_5s"): 0.25,
            ("wheel_recursive_2_10s" if max_rollout_steps > 200
             else "wheel_recursive_2_5s"): 0.15,
            ("pose_recursive_2_10s" if max_rollout_steps > 200
             else "pose_recursive_2_5s"): 0.15,
            "hard_regime_macro": 0.15,
        },
        "selection_score_pose_scales": {"position_m": 0.20, "heading_rad": 0.10},
        "selection_policy": "lowest registered validation macro-run composite; no test/final-test access",
        "sampler": {
            "base": sampler_metadata,
            "by_curriculum_horizon": validation_report_sampler,
        },
        "training_domain_row_count": int(normalizers["training_domain_rows"][0]),
        "training_domain": (
            f"training whole runs; body speed <=12 m/s; future throttle command "
            f"<={max_throttle_command:.6g}"),
        "normalization_fit": (
            "inherited training-only parent median/interquartile range; "
            "append-only extension proof recorded in warm_start"
            if dataset_extension is not None else
            "training-only median and interquartile range in declared racing domain"),
        "acceleration_bound_fit": "training-only 99.9th absolute percentile plus 15% margin in declared racing domain",
        "actuator_fit": {
            "steering": {"delay_steps": actuator_fit.steering.delay_steps,
                         "alpha": actuator_fit.steering.alpha},
            "throttle": {"delay_steps": actuator_fit.throttle.delay_steps,
                         "alpha": actuator_fit.throttle.alpha},
            "diagnostics": actuator_fit.diagnostics,
        },
        **{name: np.asarray(value).tolist() for name, value in arrays.items()},
    }
    best_score = float("inf")
    best_checkpoint = output_dir / "best.pt"
    last_checkpoint = output_dir / "last.pt"
    best_metrics_path = output_dir / "best_validation.json"
    global_step = 0
    history_rows: list[dict[str, Any]] = []

    def validate_and_checkpoint(stage_name: str, stage_step: int):
        nonlocal best_score
        model.eval()
        metrics = evaluate_model(
            model, data, state, arrays["state_scale"], device_name,
            windows=validation_windows,
            wheel_state_source=wheel_state_source,
            horizon_steps=validation_horizons)
        score_value, score_components = _selection_score(metrics)
        row = {
            "global_step": global_step,
            "stage": stage_name,
            "stage_step": stage_step,
            "validation_selection_score": score_value,
            "selection_components": score_components,
            "validation_metrics": metrics,
        }
        history_rows.append(row)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        payload = _checkpoint_payload(
            model, metadata, optimizer, global_step, score_value, torch)
        payload["sampling_rng_state"] = sample_rng.bit_generator.state
        torch.save(payload, last_checkpoint)
        if score_value < best_score:
            best_score = score_value
            torch.save(payload, best_checkpoint)
            best_metrics_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "checkpoint": str(best_checkpoint),
                    "checkpoint_score": score_value,
                    "selection_components": score_components,
                    "validation_metrics": metrics,
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        model.train()
        return score_value, score_components

    start_time = time.time()
    validate_and_checkpoint("initial_untrained", 0)
    for stage_name, horizons in curriculum:
        for stage_step in range(1, stage_steps + 1):
            if len(horizons) == 1:
                horizon = horizons[0]
            else:
                horizon = int(sample_rng.choice(horizons))
            sampler, _ = samplers[horizon]
            batch = _sample_batch(data, state, acceleration_targets,
                                  sampler, batch_size, horizon, sample_rng,
                                  model.history_state_size,
                                  raw_history_features)
            if turn_reflection_augmentation:
                batch = _random_reflection_augmentation(
                    batch, model.history_state_size, sample_rng)
            teacher_states = None
            if teacher_model is not None:
                histories, initial, delayed, commands = (
                    _tensor(torch, batch[index], device_name)
                    for index in range(4))
                with torch.no_grad():
                    teacher_states = teacher_model.rollout(
                        initial, delayed, histories, commands)[0]
            optimizer.zero_grad(set_to_none=True)
            loss_value, _ = _loss(torch, nn, model, batch, normalizers,
                                  device_name, wheel_acceleration_weight,
                                  wheel_state_weight, output_head_only,
                                  yaw_terminal_position_weight,
                                  longitudinal_low_throttle_weight,
                                  teacher_states, state_distillation_weight,
                                  state_distillation_channels=(
                                      _state_distillation_channels(
                                          output_head_only,
                                          include_roll_state)),
                                  terminal_heading_loss_weight=(
                                      terminal_heading_loss_weight),
                                  pose_secondary_weight=pose_secondary_weight)
            if not torch.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite EDSSM loss at global step {global_step}")
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            if stage_step % eval_every == 0 or stage_step == stage_steps:
                score_value, components = validate_and_checkpoint(
                    stage_name, stage_step)
                print(json.dumps({
                    "encoder": encoder,
                    "seed": seed,
                    "stage": stage_name,
                    "step": stage_step,
                    "global_step": global_step,
                    "train_loss": float(loss_value.detach().cpu()),
                    "validation_selection_score": score_value,
                    "selection_components": components,
                    "best_score": best_score,
                    "elapsed_s": time.time() - start_time,
                }), flush=True)
        # The architecture and optimizer are continuous across curriculum stages;
        # only sampled rollout horizons change.
    final_report = {
        "schema_version": 1,
        "encoder": encoder,
        "latent_size": int(latent_size),
        "expert_count": int(expert_count),
        "wheel_acceleration_loss_weight": float(wheel_acceleration_weight),
        "wheel_dynamics_mode": wheel_dynamics_mode,
        "wheel_state_loss_weight": float(wheel_state_weight),
        "yaw_terminal_position_loss_weight": float(yaw_terminal_position_weight),
        "longitudinal_low_throttle_loss_weight": float(
            longitudinal_low_throttle_weight),
        "terminal_heading_loss_weight": float(terminal_heading_loss_weight),
        "pose_secondary_loss_weight": float(pose_secondary_weight),
        "state_distillation_loss_weight": float(state_distillation_weight),
        "state_distillation_teacher": teacher_checkpoint,
        "stratify_signed_wheel_mismatch": bool(
            stratify_signed_wheel_mismatch),
        "include_roll_state": bool(include_roll_state),
        "couple_roll_acceleration": bool(couple_roll_acceleration),
        "roll_residual_mode": bool(roll_residual_mode),
        "roll_oscillator_fit": roll_oscillator_report,
        "warm_start": warm_start,
        "training_objective_mode": (
            f"{output_head_only}_head_only" if output_head_only
            else "joint_state_pose_with_low_throttle_forward_weight_and_yaw_wheel_distillation"
            if state_distillation_weight > 0.0
            else "joint_state_pose_with_low_throttle_forward_weight"
            if longitudinal_low_throttle_weight > 1.0
            else "joint_state_and_pose"),
        "acceleration_target_source": body_acceleration_source,
        "seed": int(seed),
        "sampling_seed": int(sampling_seed),
        "global_steps": global_step,
        "best_validation_selection_score": best_score,
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "training_log": str(log_path),
        "dataset_sha256": metadata["dataset_sha256"],
        "wall_time_s": time.time() - start_time,
        "test_and_final_test_used": False,
        "validation_run_ids": validation_runs,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return final_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder", choices=("gru", "tcn"), required=True)
    parser.add_argument("--latent-size", choices=(16, 32, 64), type=int,
                        default=LATENT_SIZE)
    parser.add_argument("--experts", choices=(1, 2, 3), type=int, default=1)
    parser.add_argument("--wheel-acceleration-weight", type=float, default=1.0,
                        help="weight on finite-difference rear-encoder acceleration labels")
    parser.add_argument("--wheel-state-weight", type=float,
                        default=LOSS_WEIGHTS["rear_wheel_state"],
                        help="weight on recursively predicted rear-wheel state loss")
    parser.add_argument("--include-roll-state", action="store_true",
                        help="predict roll rate recursively and integrate roll internally")
    parser.add_argument("--couple-roll-acceleration", action="store_true",
                        help="feed predicted lateral acceleration to the internal roll-rate transition")
    parser.add_argument("--roll-residual-mode", action="store_true",
                        help="keep the base 7-state dynamics branch and add a zero-initialized internal-roll residual")
    parser.add_argument("--fit-roll-oscillator", action="store_true",
                        help="fit a train-only roll spring/damping relation driven by predicted lateral acceleration")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="same-data base checkpoint for targeted residual fine-tuning")
    parser.add_argument("--output-head-only",
                        choices=("body", "body_wheel", "wheel", "yaw",
                                 "longitudinal", "longitudinal_pose",
                                 "longitudinal_wheel", "yaw_pose",
                                 "roll_residual"),
                        help="freeze the base model and train only the selected dynamics output branch")
    parser.add_argument("--yaw-terminal-position-weight", type=float, default=0.0,
                        help="terminal position loss weight during isolated yaw-head training")
    parser.add_argument("--longitudinal-low-throttle-weight", type=float,
                        default=1.0,
                        help="upweight recursive forward-speed loss when target speed is 3–8 m/s and future throttle command <=0.05")
    parser.add_argument("--state-distillation-weight", type=float, default=0.0,
                        help="retain selected frozen-parent motion states during targeted residual fine-tuning")
    parser.add_argument("--distillation-teacher-checkpoint", type=Path,
                        help="optional same-data teacher checkpoint, separate from the model initialization checkpoint")
    parser.add_argument("--pose-secondary-weight", type=float,
                        default=LOSS_WEIGHTS["pose_secondary"],
                        help="weight on the recursively integrated full-trajectory position and heading loss")
    parser.add_argument("--terminal-heading-loss-weight", type=float, default=0.0,
                        help="additional final integrated-heading loss; restricted to pose-aware longitudinal-head fine-tuning")
    parser.add_argument("--stratify-signed-wheel-mismatch", action="store_true",
                        help="balance negative, near-zero, and positive wheel/body speed-mismatch regimes")
    parser.add_argument("--body-acceleration-source",
                        choices=("state_derivative", "packet_interval_mean"),
                        default="state_derivative",
                        help="offline label source; packet acceleration is never a rollout input")
    parser.add_argument("--wheel-state-source",
                        choices=("filtered_odometry", "raw_encoder"),
                        default="filtered_odometry",
                        help="use the existing 100 ms encoder estimate or packet-aligned 25 ms raw-angle rates")
    parser.add_argument("--wheel-dynamics-mode",
                        choices=WHEEL_DYNAMICS_MODES,
                        default="surface_acceleration",
                        help="predict independent wheel acceleration or rear slip acceleration around exact contact kinematics")
    parser.add_argument("--include-raw-encoder-history", action="store_true",
                        help="add causal 40 Hz wheel rates and validity to initial history while retaining filtered wheel state targets")
    parser.add_argument("--include-wheel-innovation-history", action="store_true",
                        help="add causal raw-minus-filtered wheel-rate innovations and validity to initial history")
    parser.add_argument("--turn-reflection-augmentation", action="store_true",
                        help="randomly mirror half of training windows using empirically checked left/right vehicle symmetry")
    parser.add_argument("--allow-append-only-dataset-extension", action="store_true",
                        help=("allow full predicted-roll continuation only after "
                              "verifying every source archive array is an exact "
                              "prefix and validation-run membership is unchanged"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, default=20261009)
    parser.add_argument("--stage-steps", type=int, default=400)
    parser.add_argument("--max-rollout-steps", type=int, choices=(200, 400),
                        default=200,
                        help="maximum recursive training/validation horizon in 25 ms steps")
    parser.add_argument("--max-throttle-command", type=float, default=0.50,
                        help="maximum normalized future throttle command included in fitting and validation windows")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    expert_suffix = f"_e{args.experts}" if args.experts > 1 else ""
    output = args.output or (
        DEFAULT_OUTPUT_ROOT
        / f"edssm_{args.encoder}_z{args.latent_size}{expert_suffix}_seed{args.seed}")
    try:
        report = train(args.dataset, output, args.encoder, args.seed,
                       args.sampling_seed, args.stage_steps, args.batch_size,
                       args.eval_every, args.learning_rate, args.device,
                       args.latent_size, args.experts,
                       args.wheel_acceleration_weight,
                       args.include_roll_state,
                       args.couple_roll_acceleration,
                       args.roll_residual_mode,
                       args.fit_roll_oscillator,
                       args.body_acceleration_source,
                       args.init_checkpoint,
                       args.wheel_state_weight,
                       args.output_head_only,
                       args.yaw_terminal_position_weight,
                       args.stratify_signed_wheel_mismatch,
                       args.wheel_state_source,
                       args.include_raw_encoder_history,
                       args.include_wheel_innovation_history,
                       args.max_rollout_steps,
                       args.longitudinal_low_throttle_weight,
                       args.state_distillation_weight,
                       args.terminal_heading_loss_weight,
                       args.max_throttle_command,
                       args.distillation_teacher_checkpoint,
                       args.pose_secondary_weight,
                       args.wheel_dynamics_mode,
                       args.turn_reflection_augmentation,
                       args.allow_append_only_dataset_extension)
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            FloatingPointError, RuntimeError) as exc:
        print(f"EDSSM training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
