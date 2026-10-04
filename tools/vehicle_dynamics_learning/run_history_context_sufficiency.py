#!/usr/bin/env python3
"""Compare causal context lengths for the fixed-cadence offline plant.

This is a handoff WP28 research experiment. Every candidate has the same
160-row input capacity, network width, run/start draws, optimizer schedule,
and recursive transition; only the number of recent rows admitted by its
mask changes. Rollouts consume their own predicted state/history and the
supplied command sequence. No production stack or simulator is changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.operating_regions import region_masks
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _normalization,
    _state,
    _write_json,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    FROZEN_EVAL_STARTS,
    HistoryTransition,
    _metrics,
    _predict_run,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    ROOT,
    SOURCE_COMMIT,
    TASK_ROOT,
    WP19_CHECKPOINT,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _load_data,
    _collect_horizon_windows,
    _predict_wp19_baseline,
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)


OUTPUT_ROOT = (TASK_ROOT / "next_phase_after_2129427"
               / "history_context_sufficiency_v1")
L3_CHECKPOINT = (TASK_ROOT / "next_phase_after_2129427"
                 / "truncated_history_detached_long_rollout_v1/checkpoint.pt")
CONTEXT_STEPS = {"0.25s": 10, "0.5s": 20, "1.0s": 40,
                 "2.0s": 80, "4.0s": 160}
MAX_CONTEXT_STEPS = 160
FEATURES = 7
HIDDEN_SIZE = 256
SEED = 20261028
STAGES = (("L0", (1,), 1000, 32),
          ("L1", (4, 10, 20), 600, 16),
          ("L2", (20, 40, 80), 600, 8))
EVAL_EVERY = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
MIN_LEARNING_RATE = 3e-5
BOOTSTRAP_REPLICATES = 5000
METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)
REGION_NAMES = (
    "7_to_9mps_high_steering",
    "simultaneous_steering_throttle_transition",
    "large_wheel_body_mismatch",
    "steering_left",
    "steering_right",
)


class FixedCapacityHistoryTransition(nn.Module):
    """Explicit-history 25 ms transition with a fixed 4 s input interface."""

    def __init__(self, delta_mean: np.ndarray, delta_scale: np.ndarray) -> None:
        super().__init__()
        input_size = MAX_CONTEXT_STEPS * FEATURES + MAX_CONTEXT_STEPS + 5 + 2
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, 5),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("delta_mean", torch.as_tensor(
            delta_mean, dtype=torch.float32))
        self.register_buffer("delta_scale", torch.as_tensor(
            delta_scale, dtype=torch.float32))

    def forward(self, history: torch.Tensor, mask: torch.Tensor,
                state: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        if (history.ndim != 3
                or history.shape[1:] != (MAX_CONTEXT_STEPS, FEATURES)
                or mask.shape != history.shape[:2]
                or state.shape[-1] != 5 or command.shape[-1] != 2):
            raise ValueError("history-context transition input shape mismatch")
        values = torch.cat((
            (history * mask.unsqueeze(-1)).flatten(start_dim=1),
            mask, state, command), dim=-1)
        return self.delta_mean + self.delta_scale * self.net(values)


def pack_history(raw_history: np.ndarray, context_steps: int,
                 mean: np.ndarray, scale: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the recent context and left-pad to the common 4 s capacity."""
    values = np.asarray(raw_history, dtype=np.float32)
    if (values.shape != (MAX_CONTEXT_STEPS, FEATURES)
            or context_steps not in CONTEXT_STEPS.values()
            or not np.isfinite(values).all()):
        raise ValueError("raw history or requested context is invalid")
    normalized = (values[-context_steps:] - mean) / scale
    packed = np.zeros((MAX_CONTEXT_STEPS, FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_CONTEXT_STEPS, dtype=np.float32)
    packed[-context_steps:] = normalized
    mask[-context_steps:] = 1.0
    return packed, mask


def _history_window(capture, sequence_index: int, source_row: int,
                    context_steps: int) -> np.ndarray:
    """Return fixed-capacity history, padding only rows outside admitted context."""
    if context_steps < 1 or context_steps > MAX_CONTEXT_STEPS:
        raise ValueError("context length is outside the fixed history capacity")
    begin, end = map(int, capture.bounds[sequence_index])
    row = int(source_row)
    available = min(MAX_CONTEXT_STEPS, row + 1)
    if row < 0 or begin + row >= end or available < context_steps:
        raise ValueError("history crosses a sequence boundary or lacks context")
    history = np.zeros((MAX_CONTEXT_STEPS, FEATURES), dtype=np.float32)
    first = begin + row - available + 1
    history[-available:] = capture.input_features[
        first:begin + row + 1, :FEATURES]
    return history


def advance_context(history: torch.Tensor, mask: torch.Tensor,
                    row: torch.Tensor, context_steps: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one predicted row while retaining exactly the selected window."""
    if (history.ndim != 3
            or history.shape[1:] != (MAX_CONTEXT_STEPS, FEATURES)
            or mask.shape != history.shape[:2]
            or row.shape != (history.shape[0], FEATURES)
            or not 1 <= context_steps <= MAX_CONTEXT_STEPS):
        raise ValueError("context advance inputs are misaligned")
    retained = torch.cat((history[:, -(context_steps - 1):]
                          if context_steps > 1 else history[:, :0],
                          row[:, None, :]), dim=1)
    pad_rows = MAX_CONTEXT_STEPS - context_steps
    if pad_rows:
        retained = torch.cat((history.new_zeros(
            history.shape[0], pad_rows, FEATURES), retained), dim=1)
    next_mask = history.new_zeros(history.shape[0], MAX_CONTEXT_STEPS)
    next_mask[:, -context_steps:] = 1.0
    return retained, next_mask


def _batch_arrays(data, refs, horizon: int, context_steps: int,
                  norm_np: dict[str, np.ndarray], config,
                  device: torch.device) -> tuple[torch.Tensor, ...]:
    histories, masks, states, poses, commands, targets, target_poses = (
        [] for _ in range(7))
    history_mean = np.asarray(config.history_mean[:FEATURES], dtype=np.float32)
    history_scale = np.asarray(config.history_scale[:FEATURES], dtype=np.float32)
    for capture_index, sequence_index, source_row in refs:
        capture = data.captures[capture_index]
        sequence_begin, sequence_end = map(int, capture.bounds[sequence_index])
        begin = sequence_begin + int(source_row)
        if begin + horizon >= sequence_end:
            raise ValueError("prediction crosses a sequence boundary")
        raw_history = _history_window(
            capture, sequence_index, source_row, context_steps)
        packed, mask = pack_history(raw_history, context_steps,
                                    history_mean, history_scale)
        histories.append(packed)
        masks.append(mask)
        states.append(_state(capture, begin))
        poses.append(data.poses[capture_index][begin])
        commands.append(capture.frames[begin:begin + horizon, 7:9])
        targets.append(np.stack([
            _state(capture, begin + step)
            for step in range(1, horizon + 1)]))
        target_poses.append(data.poses[capture_index][begin + 1:begin + horizon + 1])

    state = (np.asarray(states, dtype=np.float32) - norm_np["state_mean"]) \
        / norm_np["state_scale"]
    command = (np.asarray(commands, dtype=np.float32) - norm_np["command_mean"]) \
        / norm_np["command_scale"]
    target = (np.asarray(targets, dtype=np.float32) - norm_np["state_mean"]) \
        / norm_np["state_scale"]
    arrays = (np.asarray(histories, dtype=np.float32),
              np.asarray(masks, dtype=np.float32), state,
              np.asarray(poses, dtype=np.float32), command, target,
              np.asarray(target_poses, dtype=np.float32))
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device)
                 for value in arrays)


def _rollout(model: FixedCapacityHistoryTransition,
             arrays: tuple[torch.Tensor, ...],
             norm: dict[str, torch.Tensor], horizon: int,
             context_steps: int, integrator: PoseIntegrator,
             *, training_loss: bool = False, local_only: bool = False
             ) -> dict[str, torch.Tensor]:
    history, mask, state, pose, commands, target, target_pose = arrays
    predicted_states, predicted_poses, losses = [], [], []
    for step in range(horizon):
        if training_loss and hasattr(model, "forward_with_acceleration"):
            delta, predicted_acceleration = model.forward_with_acceleration(
                history, mask, state, commands[:, step])
        else:
            delta = model(history, mask, state, commands[:, step])
            predicted_acceleration = None
        next_state = state + delta
        current_body = state[:, :3] * norm["state_scale"][:3] \
            + norm["state_mean"][:3]
        next_body = next_state[:, :3] * norm["state_scale"][:3] \
            + norm["state_mean"][:3]
        if not torch.isfinite(current_body).all() or not torch.isfinite(next_body).all():
            current_peak = float(torch.nan_to_num(
                current_body.detach().abs(), nan=0.0, posinf=float("inf")
            ).max())
            next_peak = float(torch.nan_to_num(
                next_body.detach().abs(), nan=0.0, posinf=float("inf")
            ).max())
            raise FloatingPointError(
                f"non-finite body prediction at step {step + 1}; "
                f"current_absmax={current_peak}, next_absmax={next_peak}")
        next_pose = integrator(pose, 0.5 * (current_body + next_body))
        if not torch.isfinite(next_state).all() or not torch.isfinite(next_pose).all():
            raise FloatingPointError(f"non-finite rollout at step {step + 1}")
        predicted_states.append(next_state)
        predicted_poses.append(next_pose)
        if training_loss:
            state_loss = F.smooth_l1_loss(next_state, target[:, step])
            if predicted_acceleration is not None:
                target_previous_state = (state if step == 0
                                         else target[:, step - 1])
                target_acceleration = model.midpoint_acceleration_label(
                    target_previous_state, target[:, step])
                target_acceleration = (
                    target_acceleration - model.acceleration_mean
                ) / model.acceleration_scale
                predicted_acceleration = (
                    predicted_acceleration - model.acceleration_mean
                ) / model.acceleration_scale
                acceleration_loss = F.smooth_l1_loss(
                    predicted_acceleration, target_acceleration)
                state_loss = state_loss + float(
                    getattr(model, "acceleration_supervision_weight", 1.0)
                ) * acceleration_loss
            if local_only:
                losses.append(state_loss)
            else:
                position_error = torch.linalg.vector_norm(
                    next_pose[:, :2] - target_pose[:, step, :2], dim=-1) / 0.5
                heading_error = next_pose[:, 2] - target_pose[:, step, 2]
                heading_error = torch.atan2(torch.sin(heading_error),
                                            torch.cos(heading_error)) / 0.1
                position_loss = F.smooth_l1_loss(
                    position_error, torch.zeros_like(position_error))
                heading_loss = F.smooth_l1_loss(
                    heading_error, torch.zeros_like(heading_error))
                losses.append(state_loss + 0.25 * position_loss
                              + 0.5 * heading_loss)
        if step + 1 < horizon:
            next_command = commands[:, step + 1]
            physical_state = (next_state * norm["state_scale"]
                              + norm["state_mean"])
            physical_command = (next_command * norm["command_scale"]
                                + norm["command_mean"])
            raw_history_row = torch.cat((physical_state, physical_command), dim=-1)
            normalized_row = (raw_history_row - norm["history_mean"]) \
                / norm["history_scale"]
            history, mask = advance_context(
                history, mask, normalized_row, context_steps)
        state, pose = next_state, next_pose
    return {
        "state": torch.stack(predicted_states, dim=1),
        "pose": torch.stack(predicted_poses, dim=1),
        "loss": (torch.stack(losses).mean() if losses else state.new_zeros(())),
    }


def _eligible_refs(by_run: dict[str, list[tuple[int, int, int]]]
                   ) -> dict[str, list[tuple[int, int, int]]]:
    result = {}
    for run_id, refs in sorted(by_run.items()):
        kept = [ref for ref in refs if int(ref[2]) >= MAX_CONTEXT_STEPS - 1]
        if kept:
            result[run_id] = kept
    return result


def _training_pools(data, horizon_steps: int = 80
                    ) -> tuple[dict[str, dict[str, list]], dict[str, str]]:
    if len(data.captures) != len(data.raw_sources):
        raise ValueError("training captures and sequence metadata are misaligned")
    family_for_run: dict[str, str] = {}
    for capture, source in zip(data.captures, data.raw_sources):
        run_families = source.get("training_families", source["run_families"])
        if len(run_families) != len(capture.run_ids):
            raise ValueError("training run-family metadata is misaligned")
        family_for_run.update({str(run_id): str(run_families[index])
                               for index, run_id in enumerate(capture.run_ids)})
    pools: dict[str, dict[str, list]] = {}
    for run_id in data.training_runs:
        conditions: dict[str, list] = defaultdict(list)
        refs = data.train_windows_by_horizon[horizon_steps].get(run_id, [])
        for ref in refs:
            capture_index, sequence_index, source_row = ref
            if int(source_row) < MAX_CONTEXT_STEPS - 1:
                continue
            condition_by_sequence = data.raw_sources[
                int(capture_index)]["sequence_condition_id"]
            condition = str(condition_by_sequence[int(sequence_index)])
            conditions[condition].append(ref)
        if conditions:
            pools[run_id] = dict(conditions)
    if len(pools) < 5:
        raise RuntimeError(
            f"WP28 requires five independent runs with {horizon_steps}-step targets")
    return pools, family_for_run


def _draw_plan(pools: dict[str, dict[str, list]], seed: int,
               stages=STAGES,
               pools_by_stage: dict[str, dict[str, dict[str, list]]] | None = None
               ) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    rng = np.random.default_rng(seed)
    plan: list[dict[str, Any]] = []
    counters: dict[str, Counter] = {
        "run": Counter(), "condition": Counter(), "horizon": Counter()}
    for stage, choices, updates, batch_size in stages:
        stage_pools = (pools_by_stage or {}).get(stage, pools)
        stage_run_ids = sorted(stage_pools)
        for update in range(1, updates + 1):
            horizon = int(choices[int(rng.integers(len(choices)))])
            batch = []
            for _ in range(batch_size):
                run_id = stage_run_ids[int(rng.integers(len(stage_run_ids)))]
                conditions = sorted(stage_pools[run_id])
                condition = conditions[int(rng.integers(len(conditions)))]
                refs = stage_pools[run_id][condition]
                ref = refs[int(rng.integers(len(refs)))]
                batch.append([int(value) for value in ref])
                counters["run"][run_id] += 1
                counters["condition"][f"{run_id}:{condition}"] += 1
                counters["horizon"][f"{stage}:{horizon}"] += 1
            plan.append({"stage": stage, "update": update,
                         "horizon_steps": horizon, "refs": batch})
    return plan, {key: dict(value) for key, value in counters.items()}


def _eval_metrics(data, refs_by_run: dict[str, list], model, norm_np,
                  config, context_steps: int, device: torch.device,
                  horizon: int) -> dict[str, dict[str, float]]:
    norm = _torch_norm(norm_np, config, device)
    result: dict[str, dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for run_id, refs in sorted(refs_by_run.items()):
            capture_index = int(refs[0][0])
            arrays = _batch_arrays(data, refs, horizon, context_steps,
                                   norm_np, config, device)
            rollout = _rollout(model, arrays, norm, horizon, context_steps,
                               PoseIntegrator(DT_S).to(device))
            state = rollout["state"].cpu().numpy() * norm_np["state_scale"] \
                + norm_np["state_mean"]
            truth = arrays[5].cpu().numpy() * norm_np["state_scale"] \
                + norm_np["state_mean"]
            result[run_id] = _metrics(
                state, rollout["pose"].cpu().numpy(), truth,
                arrays[6].cpu().numpy(), horizon)
    return result


def _eval_l3_metrics(data, refs_by_run: dict[str, list], model, norm_np,
                     config, device: torch.device,
                     horizon: int) -> dict[str, dict[str, float]]:
    result = {}
    for run_id, refs in sorted(refs_by_run.items()):
        prediction = _predict_run(data, refs, model, norm_np, config,
                                  horizon, device)
        result[run_id] = _metrics(
            prediction[0], prediction[1], prediction[2], prediction[3], horizon)
    return result


def _torch_norm(norm_np: dict[str, np.ndarray], config,
                device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(norm_np[key], dtype=torch.float32, device=device)
        for key in ("state_mean", "state_scale", "command_mean", "command_scale")
    } | {
        "history_mean": torch.as_tensor(config.history_mean[:FEATURES],
                                        dtype=torch.float32, device=device),
        "history_scale": torch.as_tensor(config.history_scale[:FEATURES],
                                         dtype=torch.float32, device=device),
    }


def _score(per_run: dict[str, dict[str, float]],
           parent: dict[str, dict[str, float]]) -> tuple[float, dict[str, float]]:
    macro = {name: float(np.mean([values[name] for values in per_run.values()]))
             for name in METRICS}
    parent_macro = {name: float(np.mean([values[name]
                                        for values in parent.values()]))
                    for name in METRICS}
    ratios = {name: macro[name] / max(parent_macro[name], 1e-8)
              for name in METRICS}
    return float(np.mean(list(ratios.values()))), ratios


def _bootstrap_delta(candidate: dict[str, dict[str, float]],
                     reference: dict[str, dict[str, float]],
                     metric: str, seed: int) -> dict[str, Any]:
    run_ids = sorted(set(candidate) & set(reference))
    deltas = np.asarray([candidate[run][metric] - reference[run][metric]
                         for run in run_ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(deltas) < 2:
        interval = None
    else:
        indexes = rng.integers(0, len(deltas),
                               size=(BOOTSTRAP_REPLICATES, len(deltas)))
        interval = np.quantile(deltas[indexes].mean(axis=1), [0.025, 0.975])
        interval = [float(value) for value in interval]
    return {"candidate_minus_reference_macro_delta": float(deltas.mean()),
            "run_cluster_bootstrap_95pct_ci": interval,
            "independent_runs": len(run_ids),
            "per_run_delta": dict(zip(run_ids, deltas.tolist()))}


def _region_metrics(data, refs_by_run: dict[str, list], model,
                    norm_np, config, context_steps: int,
                    device: torch.device,
                    horizon_steps: int = 80) -> dict[str, Any]:
    norm = _torch_norm(norm_np, config, device)
    report: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for run_id, refs in sorted(refs_by_run.items()):
            capture_index = int(refs[0][0])
            capture = data.captures[capture_index]
            reset_rows = np.full(len(capture.frames), -1, dtype=np.int32)
            for index, (begin, end) in enumerate(capture.bounds):
                reset_rows[int(begin):int(end)] = capture.sequence_reset[index]
            valid = (np.isfinite(capture.frames).all(axis=1)
                     & np.isfinite(capture.body).all(axis=1)
                     & np.isclose(capture.dt_s, DT_S, rtol=0.0, atol=1e-7))
            masks = region_masks(capture.frames, reset_rows,
                                 capture.packet, valid, DT_S)
            memberships: dict[str, list[int]] = defaultdict(list)
            for i, (_, sequence_index, source_row) in enumerate(refs):
                row = int(capture.bounds[sequence_index, 0]) + int(source_row)
                for name in REGION_NAMES[:-2]:
                    if name in masks and bool(masks[name][row]):
                        memberships[name].append(i)
                memberships["steering_left" if capture.frames[row, 3] > 0.0
                            else "steering_right" if capture.frames[row, 3] < 0.0
                            else "steering_zero"].append(i)
            full = _batch_arrays(data, refs, horizon_steps, context_steps,
                                 norm_np, config, device)
            predicted = _rollout(model, full, norm, horizon_steps, context_steps,
                                 PoseIntegrator(DT_S).to(device))
            p_state = predicted["state"].cpu().numpy() * norm_np["state_scale"] \
                + norm_np["state_mean"]
            t_state = full[5].cpu().numpy() * norm_np["state_scale"] \
                + norm_np["state_mean"]
            p_pose = predicted["pose"].cpu().numpy()
            t_pose = full[6].cpu().numpy()
            report[run_id] = {}
            for region in REGION_NAMES:
                indexes = memberships.get(region, [])
                if not indexes:
                    continue
                idx = np.asarray(indexes, dtype=np.int64)
                report[run_id][region] = {
                    "start_count": int(len(indexes)),
                    **_metrics(p_state[idx], p_pose[idx], t_state[idx],
                               t_pose[idx], horizon_steps),
                }
    return report


def _region_macro(region_data: dict[str, Any]) -> dict[str, Any]:
    regions = sorted({region for runs in region_data.values()
                      for region in runs})
    output = {}
    for region in regions:
        runs = {run: values[region] for run, values in region_data.items()
                if region in values}
        if len(runs) < 2:
            continue
        metrics = [name for name in METRICS if name in next(iter(runs.values()))]
        output[region] = {
            "independent_runs": len(runs),
            "start_count": int(sum(values["start_count"]
                                   for values in runs.values())),
            "macro_run_metrics": {
                metric: float(np.mean([values[metric]
                                       for values in runs.values()])
                              ) for metric in metrics},
            "per_run": runs,
        }
    return output


def _checkpoint_eval(data, refs, model, norm_np, config, context_steps,
                     device, stage: str, horizon: int,
                     parent_metrics, metrics=None) -> dict[str, Any]:
    if metrics is None:
        metrics = _eval_metrics(data, refs, model, norm_np, config,
                                context_steps, device, horizon)
    if stage == "L0":
        scale = np.asarray(norm_np["state_scale"], dtype=np.float64)
        channels = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps")
        denominator = scale[:3]
        value = np.mean([np.mean([run[name] for run in metrics.values()])
                         / denominator[index]
                         for index, name in enumerate(channels)])
        ratios = {name: float(np.mean([run[name] for run in metrics.values()]))
                  for name in channels}
    else:
        value, ratios = _score(metrics, parent_metrics)
    return {"score": float(value), "metrics": metrics,
            "score_channels": ratios, "horizon_steps": horizon,
            "independent_run_count": len(metrics)}


def _gradient_norm(tensors) -> float:
    values = [torch.sum(tensor.detach().square()) for tensor in tensors
              if tensor is not None]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu())


def _gradient_components(model) -> dict[str, float]:
    layers = list(model.net.children())
    output = layers[-1]
    if not isinstance(output, nn.Linear) or output.out_features < 5:
        raise TypeError("WP27 gradient groups require a five-channel linear head")
    trunk_parameters = [parameter for layer in layers[:-1]
                        for parameter in layer.parameters()]
    return {
        "shared_trunk": _gradient_norm(
            [parameter.grad for parameter in trunk_parameters]),
        "primary_dynamics_output": _gradient_norm(
            [output.weight.grad[:3],
             None if output.bias.grad is None else output.bias.grad[:3]]),
        "auxiliary_actuator_output": _gradient_norm(
            [output.weight.grad[3:5],
             None if output.bias.grad is None else output.bias.grad[3:5]]),
    }


def _gradient_summary(rows: list[dict[str, float]], threshold: float
                      ) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    output: dict[str, Any] = {"sample_count": len(rows)}
    for name in ("raw_total", "post_clip_total", "shared_trunk",
                 "primary_dynamics_output", "auxiliary_actuator_output"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        output[name] = {
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        }
    output["clip_threshold"] = float(threshold)
    output["fraction_clipped"] = float(np.mean([
        row["raw_total"] > threshold for row in rows]))
    return output


def _train_one(data, context_name: str, context_steps: int,
               plan: list[dict[str, Any]], eval_refs: dict[str, list],
               norm_np, config, parent_metrics, device: torch.device,
               stages=STAGES,
               initial_state_dict: dict[str, torch.Tensor] | None = None,
               model_factory=None, wp27_policy: bool = False,
               wp27_a2_yaw_reference: float | None = None
               ) -> tuple[nn.Module, dict[str, Any]]:
    if wp27_policy:
        expected = {
            "L0": (1,),
            "L1": (4, 10, 20),
            "L2": (20, 40, 80),
        }
        actual = {stage: tuple(choices) for stage, choices, _, _ in stages}
        if actual != expected:
            raise ValueError(f"WP27 requires the prescribed horizons, got {actual}")
        if (wp27_a2_yaw_reference is None
                or not np.isfinite(wp27_a2_yaw_reference)
                or wp27_a2_yaw_reference <= 0.0):
            raise ValueError("WP27 requires the frozen A2 2-second yaw reference")
    torch.manual_seed(SEED)
    model = (model_factory() if model_factory is not None else
             FixedCapacityHistoryTransition(
                 norm_np["delta_mean"], norm_np["delta_scale"])).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    norm = _torch_norm(norm_np, config, device)
    integrator = PoseIntegrator(DT_S).to(device)
    reports = []
    total_updates = 0
    plan_cursor = 0
    gradient_history_by_stage: dict[str, list[float]] = {}
    best_one_step_score = float("inf")
    previous_eval_loss: float | None = None
    one_step_worse_evaluations = 0
    a2_yaw_worse_evaluations = 0
    stop_reason = None
    started = time.perf_counter()
    for stage, choices, max_updates, _batch_size in stages:
        if wp27_policy:
            while (plan_cursor < len(plan)
                   and plan[plan_cursor]["stage"] != stage):
                plan_cursor += 1
            if plan_cursor >= len(plan):
                raise RuntimeError(f"WP27 draw plan has no {stage} samples")
            clip_threshold = 10.0
            clip_policy_source = "fixed_10_for_L0_L1"
            if stage == "L2":
                l1_norms = gradient_history_by_stage.get("L1", [])[:100]
                if len(l1_norms) < 100:
                    raise RuntimeError(
                        "WP27 needs the first 100 L1 gradients to set the L2 clip")
                l1_p95 = float(np.quantile(l1_norms, 0.95))
                clip_threshold = min(10.0, max(1.0, l1_p95))
                clip_policy_source = "clamped_p95_of_first_100_L1_raw_norms"
        best_score = float("inf")
        best_state = None
        best_optimizer = None
        best_result = None
        stale = 0
        stale_for_lr = 0
        reductions = 0
        stage_stop_reason = None
        gradients = []
        gradient_rows = []
        gradient_rows_since_eval = []
        gradient_checkpoints = []
        trace = []
        training_losses_since_eval = []
        stage_updates = 0
        for local_update in range(1, max_updates + 1):
            if plan_cursor >= len(plan):
                raise RuntimeError("training draw plan exhausted")
            draw = plan[plan_cursor]
            plan_cursor += 1
            if draw["stage"] != stage:
                raise RuntimeError("paired WP28 draw plan stage mismatch")
            refs = [tuple(row) for row in draw["refs"]]
            horizon = int(draw["horizon_steps"])
            batch = _batch_arrays(data, refs, horizon, context_steps,
                                 norm_np, config, device)
            optimizer.zero_grad(set_to_none=True)
            try:
                prediction = _rollout(
                    model, batch, norm, horizon, context_steps, integrator,
                    training_loss=True, local_only=(stage == "L0"))
            except (FloatingPointError, ValueError) as exc:
                raise FloatingPointError(
                    f"{context_name} {stage} update {local_update} "
                    f"horizon={horizon} refs={draw['refs']}: {exc}") from exc
            loss = prediction["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{context_name} {stage} non-finite loss")
            loss.backward()
            component_norms = (_gradient_components(model)
                               if wp27_policy else {})
            threshold = clip_threshold if wp27_policy else 10.0
            raw_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), threshold))
            if not np.isfinite(raw_norm):
                raise FloatingPointError(f"{context_name} {stage} non-finite gradient")
            optimizer.step()
            post_clip_norm = min(raw_norm, threshold)
            gradients.append(raw_norm)
            stage_updates += 1
            total_updates += 1
            train_loss_value = float(loss.detach())
            training_losses_since_eval.append(train_loss_value)
            if wp27_policy:
                row = {
                    "raw_total": raw_norm,
                    "post_clip_total": post_clip_norm,
                    **component_norms,
                }
                gradient_rows.append(row)
                gradient_rows_since_eval.append(row)
                if stage == "L1" and local_update <= 100:
                    gradient_history_by_stage.setdefault(stage, []).append(raw_norm)
            if local_update % 20 == 0:
                trace.append({"update": local_update, "loss": train_loss_value,
                              "raw_gradient_norm": raw_norm})
            if local_update % EVAL_EVERY:
                continue
            eval_horizon = 1 if stage == "L0" else max(choices)
            validation_by_horizon = None
            if wp27_policy:
                validation_by_horizon = {
                    horizon: _eval_metrics(
                        data, eval_refs, model, norm_np, config, context_steps,
                        device, horizon)
                    for horizon in (1, 10, 20, 40, 80, 200)
                }
            scored = _checkpoint_eval(
                data, eval_refs, model, norm_np, config, context_steps,
                device, stage, eval_horizon, parent_metrics[eval_horizon],
                metrics=(validation_by_horizon[eval_horizon]
                         if validation_by_horizon is not None else None))
            scored.update({"stage": stage, "local_update": local_update,
                           "total_updates": total_updates,
                           "learning_rate": optimizer.param_groups[0]["lr"]})
            current_train_loss = (float(np.mean(training_losses_since_eval))
                                  if training_losses_since_eval else train_loss_value)
            scored["training_loss_since_previous_evaluation"] = current_train_loss
            if wp27_policy:
                scored["validation_by_horizon_macro_run"] = {
                    str(horizon): {
                        metric: float(np.mean([
                            values[metric] for values in run_values.values()]))
                        for metric in next(iter(run_values.values()))
                    }
                    for horizon, run_values in validation_by_horizon.items()
                }
                one_step_score, _ = _score(
                    validation_by_horizon[1], parent_metrics[1])
                previous_best_one_step = best_one_step_score
                best_one_step_score = min(best_one_step_score, one_step_score)
                if (previous_best_one_step < float("inf")
                        and previous_eval_loss is not None
                        and current_train_loss < previous_eval_loss
                        and one_step_score > previous_best_one_step * 1.01):
                    one_step_worse_evaluations += 1
                else:
                    one_step_worse_evaluations = 0
                scored["one_step_validation_composite_score"] = one_step_score
                scored["best_one_step_validation_composite_score"] = (
                    best_one_step_score)
                scored["gradient_diagnostics_since_previous_evaluation"] = (
                    _gradient_summary(gradient_rows_since_eval, threshold))
                gradient_checkpoints.append({
                    "local_update": local_update,
                    "diagnostics": scored[
                        "gradient_diagnostics_since_previous_evaluation"],
                })
                gradient_rows_since_eval = []
                training_losses_since_eval = []

            improved = scored["score"] < best_score - 1e-7
            if improved:
                best_score = scored["score"]
                best_state = copy.deepcopy(model.state_dict())
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                best_result = scored
                stale = 0
                stale_for_lr = 0
            else:
                stale += 1
                stale_for_lr += 1
            if stale_for_lr >= 2 and reductions < 3:
                new_lr = max(MIN_LEARNING_RATE,
                             optimizer.param_groups[0]["lr"] * 0.5)
                if new_lr < optimizer.param_groups[0]["lr"]:
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr
                    reductions += 1
                stale_for_lr = 0
            if wp27_policy:
                previous_eval_loss = current_train_loss
                l2_clipped_fraction = (float(np.mean([
                    row["raw_total"] > clip_threshold for row in gradient_rows]))
                    if gradient_rows else 0.0)
                if stage == "L0" and stale >= 3:
                    stage_stop_reason = (
                        "L0 one-step validation patience reached 3 evaluations")
                elif (stage in {"L1", "L2"}
                      and one_step_worse_evaluations >= 2):
                    stop_reason = (
                        "one-step validation worsened persistently while training loss fell")
                elif (stage == "L2" and l2_clipped_fraction > 0.95
                      and stage_updates >= EVAL_EVERY):
                    stop_reason = "more than 95% of L2 updates were clipped"
                if stage == "L2":
                    yaw_2s = float(np.mean([
                        values["yaw_rate_rmse_rps"]
                        for values in validation_by_horizon[80].values()]))
                    if yaw_2s > 1.20 * wp27_a2_yaw_reference:
                        a2_yaw_worse_evaluations += 1
                    else:
                        a2_yaw_worse_evaluations = 0
                    scored["two_second_yaw_rmse_macro_run"] = yaw_2s
                    scored["two_second_yaw_reference_A2"] = (
                        float(wp27_a2_yaw_reference))
                    if a2_yaw_worse_evaluations >= 2:
                        stop_reason = (
                            "2-second yaw RMSE exceeded frozen A2 by >20% twice")
            print(f"{context_name} {stage} {local_update}/{max_updates} "
                  f"score={scored['score']:.5f} best={best_score:.5f} "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
            if wp27_policy and (stage_stop_reason is not None
                                or stop_reason is not None):
                break
        if best_state is None or best_optimizer is None or best_result is None:
            raise RuntimeError(f"{context_name} {stage} produced no checkpoint")
        model.load_state_dict(best_state, strict=True)
        optimizer.load_state_dict(best_optimizer)
        stage_report = {
            "stage": stage,
            "max_updates": max_updates,
            "completed_updates": stage_updates,
            "horizon_choices_steps": list(choices),
            "best_validation": best_result,
            "raw_gradient_norm_p50": float(np.quantile(gradients, 0.50)),
            "raw_gradient_norm_p95": float(np.quantile(gradients, 0.95)),
            "learning_rate_reductions": reductions,
            "training_trace_every_20_updates": trace,
        }
        if wp27_policy:
            stage_report.update({
                "clip_threshold": float(clip_threshold),
                "clip_threshold_source": clip_policy_source,
                "gradient_summary_over_stage": _gradient_summary(
                    gradient_rows, clip_threshold),
                "gradient_diagnostics_at_each_evaluation": gradient_checkpoints,
                "early_stop_reason": (
                    stage_stop_reason or stop_reason
                    if stage_updates < max_updates else None),
            })
        reports.append(stage_report)
        if wp27_policy and stop_reason is not None:
            break
    return model, {"stages": reports, "updates": total_updates,
                   "elapsed_seconds": time.perf_counter() - started,
                   "unused_draw_count": len(plan) - plan_cursor,
                   "early_stop_reason": stop_reason}


def _run_id(refs: list[tuple[int, int, int]], data) -> str:
    capture_index, sequence_index, _ = refs[0]
    capture = data.captures[capture_index]
    return str(capture.run_ids[int(capture.sequence_run[sequence_index])])


def _fixed_eval_refs(data) -> tuple[dict[str, list], dict[str, list]]:
    # WP28 scores only through 2 s, so collect 2 s-safe starts before applying
    # the 4 s history requirement instead of inheriting WP24's 10 s horizon.
    development_all = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"validation"}, 80, capture_indices={0})
    practice_all = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"unseen_practice"}, 80, capture_indices={1})
    development = _select_eval_windows(_eligible_refs(development_all), 64)
    practice = _select_eval_windows(_eligible_refs(practice_all), 64)
    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    if set(development) != expected_validation or set(practice) != expected_practice:
        raise RuntimeError("WP28 usable whole-run set differs from frozen WP24 runs")
    return development, practice


def run(device_name: str = "cuda", output_root: Path = OUTPUT_ROOT
        ) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    train_pools, family_for_run = _training_pools(data)
    plan, sampler_counts = _draw_plan(train_pools, SEED + 1)
    validation_refs, practice_refs = _fixed_eval_refs(data)
    if any(len(refs) < 8 for refs in validation_refs.values()):
        raise RuntimeError("too few common reset-safe validation starts after 4 s")
    if any(len(refs) < 8 for refs in practice_refs.values()):
        raise RuntimeError("too few common reset-safe practice starts after 4 s")

    # Fixed development starts are kept byte-for-byte in the output package.
    validation_starts = {
        run_id: [[int(seq), int(row)] for _, seq, row in refs]
        for run_id, refs in sorted(validation_refs.items())}
    starts_path = output_root / "validation_starts.json"
    if output_root.exists():
        existing = {path.name for path in output_root.iterdir()}
        if existing != {"validation_starts.json"}:
            raise FileExistsError(
                f"refusing to overwrite non-preflight WP28 artifacts in {output_root}")
        previous_starts = json.loads(starts_path.read_text(encoding="utf-8"))
        if previous_starts != validation_starts:
            raise RuntimeError("existing WP28 preflight starts differ from current data")
    else:
        _write_json(starts_path, validation_starts)

    parent_wp19_checkpoint = torch.load(WP19_CHECKPOINT, map_location=device,
                                       weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(parent_wp19_checkpoint["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    l3_saved = torch.load(L3_CHECKPOINT, map_location=device, weights_only=True)
    l3_model = HistoryTransition(norm_np["delta_mean"],
                                 norm_np["delta_scale"]).to(device)
    l3_model.load_state_dict(l3_saved["state_dict"], strict=True)

    baseline_metrics: dict[int, dict[str, dict[str, float]]] = {
        1: {}, 20: {}, 80: {}}
    l3_baseline_metrics: dict[int, dict[str, dict[str, float]]] = {
        1: {}, 20: {}, 80: {}}
    for horizon in (1, 20, 80):
        for run_id, refs in sorted(validation_refs.items()):
            capture_index = int(refs[0][0])
            capture = data.captures[capture_index]
            parent = _predict_wp19_baseline(
                capture, data.poses[capture_index], refs, horizon,
                wp19_stats, wp19, device)
            truth_arrays = _batch_arrays(data, refs, horizon, 80,
                                         norm_np, config, device)
            truth_state = truth_arrays[5].cpu().numpy() * norm_np["state_scale"] \
                + norm_np["state_mean"]
            truth_pose = truth_arrays[6].cpu().numpy()
            baseline_metrics[horizon][run_id] = _metrics(
                np.concatenate((parent["body"], truth_state[..., 3:5]), axis=-1),
                parent["pose"], truth_state, truth_pose, horizon,
                include_actuator=False)
        # L3 is evaluated on the same eligible starts, using its own 2 s history.
        if horizon == 1:
            l3_horizon = 1
        else:
            l3_horizon = horizon
        l3_metrics = _eval_l3_metrics(
            data, validation_refs, l3_model, norm_np, config,
            device, l3_horizon)
        for run_id, metrics in l3_metrics.items():
            l3_baseline_metrics[horizon][run_id] = metrics

    history_stats = {
        "mean": np.asarray(config.history_mean[:FEATURES]).tolist(),
        "scale": np.asarray(config.history_scale[:FEATURES]).tolist(),
    }
    candidate_reports: dict[str, Any] = {}
    selected_models: dict[str, FixedCapacityHistoryTransition] = {}
    started = time.perf_counter()
    for context_name, context_steps in CONTEXT_STEPS.items():
        print(f"WP28 training {context_name} context ({context_steps} samples)",
              flush=True)
        model, training = _train_one(
            data, context_name, context_steps, plan, validation_refs,
            norm_np, config, baseline_metrics, device)
        selected_models[context_name] = model
        final_validation = {
            horizon: _eval_metrics(
                data, validation_refs, model, norm_np, config,
                context_steps, device, horizon)
            for horizon in (10, 30, 80)}
        final_practice = _eval_metrics(
            data, practice_refs, model, norm_np, config,
            context_steps, device, 80)
        region_detail = _region_metrics(
            data, validation_refs, model, norm_np, config,
            context_steps, device)
        training_manifest = {
            "training_runs": sorted(train_pools),
            "families_by_run": {run: family_for_run.get(run, "unclassified")
                                for run in sorted(train_pools)},
            "condition_count_by_run": {run: len(values)
                                       for run, values in sorted(train_pools.items())},
            "sampler_counts": sampler_counts,
            "draw_plan_sha256": hashlib.sha256(json.dumps(
                plan, separators=(",", ":")).encode()).hexdigest(),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        directory = output_root / context_name
        directory.mkdir(parents=False, exist_ok=False)
        checkpoint = directory / "checkpoint.pt"
        metadata = {
            "model": "fixed-capacity explicit-history nonlinear transition",
            "context_name": context_name,
            "context_steps": context_steps,
            "context_seconds": context_steps * DT_S,
            "max_history_steps": MAX_CONTEXT_STEPS,
            "history_features": ["odom_u", "odom_v", "odom_yaw_rate",
                                 "steering_feedback", "throttle_feedback",
                                 "steering_command", "throttle_command"],
            "history_masked_left_padding": True,
            "future_measured_sensor_or_truth_in_rollout": False,
            "control_and_data_rate_hz": 40.0,
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                capture_output=True, text=True).stdout.strip(),
            "git_worktree_dirty": bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, check=True,
                capture_output=True, text=True).stdout.strip()),
            "seed": SEED,
            "fixed_capacity_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()),
            "history_normalization": history_stats,
        }
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
        checkpoint_sha = sha256_file(checkpoint)
        (directory / "checkpoint.sha256").write_text(
            checkpoint_sha + "\n", encoding="utf-8")
        _write_json(directory / "checkpoint_metadata.json",
                    {**metadata, "checkpoint_sha256": checkpoint_sha})
        _write_json(directory / "training_report.json", training)
        candidate_reports[context_name] = {
            "context_steps": context_steps,
            "checkpoint_sha256": checkpoint_sha,
            "training": training,
            "validation": final_validation,
            "practice_diagnostic": final_practice,
            "validation_regions_2s": _region_macro(region_detail),
            "validation_region_per_run": region_detail,
        }
        print(f"WP28 complete {context_name} elapsed="
              f"{training['elapsed_seconds']:.1f}s", flush=True)

    candidate_scores: dict[str, float] = {}
    candidate_macro: dict[str, dict[str, dict[str, float]]] = {}
    for name, report in candidate_reports.items():
        candidate_macro[name] = {}
        per_run_2s = report["validation"][80]
        score, ratios = _score(per_run_2s, baseline_metrics[80])
        candidate_scores[name] = score
        candidate_macro[name] = {
            metric: {"macro_run_mean": float(np.mean(
                [values[metric] for values in per_run_2s.values()])),
                     "per_run": {run: values[metric]
                                 for run, values in per_run_2s.items()}}
            for metric in METRICS}
        report["validation_vs_WP19_2s"] = {
            "equal_weight_relative_score": score,
            "metric_error_ratios": ratios,
            "paired_parent_minus_candidate": {
                metric: _bootstrap_delta(
                    baseline_metrics[80], per_run_2s, metric, SEED + i)
                for i, metric in enumerate(METRICS)},
        }

    best_name = min(candidate_scores, key=candidate_scores.get)
    best_metrics = candidate_reports[best_name]["validation"][80]
    selection_audit = {}
    selected_name = None
    for name, report in candidate_reports.items():
        metrics = report["validation"][80]
        ratios = {metric: float(np.mean([values[metric]
                                         for values in metrics.values()])
                                / max(np.mean([values[metric]
                                               for values in best_metrics.values()]),
                                      1e-9))
                  for metric in METRICS}
        intervals = {metric: _bootstrap_delta(
            metrics, best_metrics, metric, SEED + 100 + index)
            for index, metric in enumerate(METRICS)}
        no_material_regression = all(
            interval["run_cluster_bootstrap_95pct_ci"] is not None
            and interval["run_cluster_bootstrap_95pct_ci"][1]
                <= 0.03 * max(float(np.mean([run[metric]
                                             for run in best_metrics.values()])), 1e-9)
            for metric, interval in intervals.items())
        within_three = all(value <= 1.03 for value in ratios.values())
        selection_audit[name] = {
            "ratios_to_best_history_by_metric": ratios,
            "paired_metric_bootstrap_vs_best": intervals,
            "within_3pct_all_2s_body_metrics": within_three,
            "no_material_regression_bootstrap_rule": no_material_regression,
        }
        if (selected_name is None and within_three and no_material_regression):
            selected_name = name
    selection_status = ("shortest history passes the 3% and run-bootstrap rule"
                        if selected_name else
                        "no history length meets the preregistered shortest-context rule")

    report = {
        "study": "WP28 fixed-capacity causal history sufficiency",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        "source_commit_expected": SOURCE_COMMIT,
        "git_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip()),
        "control_and_data_rate_hz": 40.0,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "future_measurements_or_truth_used_as_rollout_inputs": False,
        "test_or_final_test_opened": False,
        "training_runs": sorted(train_pools),
        "training_runs_without_4s_context": sorted(
            set(data.training_runs) - set(train_pools)),
        "validation_runs": sorted(validation_refs),
        "practice_runs_diagnostic_only": sorted(practice_refs),
        "draw_plan_sha256": hashlib.sha256(json.dumps(
            plan, separators=(",", ":")).encode()).hexdigest(),
        "artifact_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "wp19_parent_checkpoint": sha256_file(WP19_CHECKPOINT),
            "l3_5s_comparator_checkpoint": sha256_file(L3_CHECKPOINT),
            "openplane_source_dataset": sha256_file(DEFAULT_DYNAMIC),
            "openplane_fixed_encoder_sidecar": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice_source_dataset": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed_encoder_sidecar": sha256_file(DEFAULT_PRACTICE_FIXED),
        },
        "draw_plan": plan,
        "sampler_counts": sampler_counts,
        "WP19_parent_same_validation_starts": baseline_metrics,
        "L3_5s_same_validation_starts": l3_baseline_metrics,
        "candidate_reports": candidate_reports,
        "candidate_2s_relative_score_vs_WP19": candidate_scores,
        "best_history_by_validation_composite": best_name,
        "selection_audit_vs_best": selection_audit,
        "selected_history": selected_name,
        "selection_status": selection_status,
        "limits": [
            "History lengths share one fixed 160-row by 7-feature input and one network capacity; inactive rows are masked.",
            "The 4 s eligibility requirement removes two short training captures; all five candidates use the same remaining runs and windows.",
            "Validation runs have been used for prior development and this selection; results are not pristine confirmation.",
            "Two practice runs are transfer diagnostics only.",
            "No 1 kHz vehicle dynamics are inferred from the 40 Hz labels.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "output_directory": output_root.relative_to(ROOT).as_posix(),
    }
    _write_json(output_root / "wp28_history_context_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result = run(args.device, output)
    print(json.dumps({
        "output": result["output_directory"],
        "best_history": result["best_history_by_validation_composite"],
        "selected_history": result["selected_history"],
        "selection_status": result["selection_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
