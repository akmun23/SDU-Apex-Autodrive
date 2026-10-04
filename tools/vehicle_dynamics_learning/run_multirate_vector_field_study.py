#!/usr/bin/env python3
"""Compare 40 Hz discrete and substepped continuous-time offline plants.

The only supervision is the existing 40 Hz simulator capture. Commands are
held at each captured value for the duration of a 25 ms control interval.
Nothing here changes AutoDRIVE, production odometry, or the MPC.
"""

from __future__ import annotations

import argparse
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
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    TASK_ROOT,
    _load_data,
    _sample_windows,
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = TASK_ROOT / "next_phase_after_2129427" / "multirate_vector_field_v1"
SEED = 20261003
BATCH_SIZE = 64
DEFAULT_UPDATES = 300
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 128
TRAINING_INTEGRATION_SUBSTEPS = 25
EVAL_INTEGRATION_SUBSTEPS = (1, 5, 25)
EVAL_HORIZON_STEPS = 80  # 2 s; same frozen whole-run development/practice starts.
BOOTSTRAP_REPLICATES = 5000
POSITION_SCALE_M = 0.5
HEADING_SCALE_RAD = 0.1
SOURCE_COMMIT = "2129427838101eee277e17e91727810bbe2df687"


class StateDeltaModel(nn.Module):
    """Direct 25 ms endpoint map, used as the matched discrete baseline."""

    def __init__(self, delta_mean: np.ndarray, delta_scale: np.ndarray) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, 5),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("delta_mean", torch.as_tensor(delta_mean, dtype=torch.float32))
        self.register_buffer("delta_scale", torch.as_tensor(delta_scale, dtype=torch.float32))

    def forward(self, current: torch.Tensor, previous: torch.Tensor,
                command: torch.Tensor, previous_command: torch.Tensor) -> torch.Tensor:
        features = torch.cat((current, previous, command, previous_command), dim=-1)
        return self.delta_mean + self.delta_scale * self.net(features)


class ContinuousStateVectorField(nn.Module):
    """Learn dx/dt in normalized state coordinates; no autonomous latent state."""

    def __init__(self, rate_mean: np.ndarray, rate_scale: np.ndarray) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, 5),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("rate_mean", torch.as_tensor(rate_mean, dtype=torch.float32))
        self.register_buffer("rate_scale", torch.as_tensor(rate_scale, dtype=torch.float32))

    def forward(self, current: torch.Tensor, previous: torch.Tensor,
                command: torch.Tensor, previous_command: torch.Tensor) -> torch.Tensor:
        features = torch.cat((current, previous, command, previous_command), dim=-1)
        return self.rate_mean + self.rate_scale * self.net(features)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _state(capture, absolute_row: int) -> np.ndarray:
    return np.concatenate((capture.body[absolute_row],
                           capture.frames[absolute_row, 3:5])).astype(np.float32)


def _transition_arrays(data, refs, state_mean: np.ndarray, state_scale: np.ndarray,
                       command_mean: np.ndarray, command_scale: np.ndarray,
                       device: torch.device) -> tuple[torch.Tensor, ...]:
    current, previous, command, previous_command, target, pose, target_pose = (
        [] for _ in range(7))
    for capture_index, sequence_index, source_row in refs:
        capture = data.captures[capture_index]
        begin = int(capture.bounds[sequence_index, 0]) + int(source_row)
        current.append(_state(capture, begin))
        previous.append(_state(capture, begin - 1))
        command.append(capture.frames[begin, 7:9])
        previous_command.append(capture.frames[begin - 1, 7:9])
        target.append(_state(capture, begin + 1))
        pose.append(data.poses[capture_index][begin])
        target_pose.append(data.poses[capture_index][begin + 1])
    current = (np.asarray(current) - state_mean) / state_scale
    previous = (np.asarray(previous) - state_mean) / state_scale
    command = (np.asarray(command) - command_mean) / command_scale
    previous_command = (np.asarray(previous_command) - command_mean) / command_scale
    target = (np.asarray(target) - state_mean) / state_scale
    values = (current, previous, command, previous_command, target,
              np.asarray(pose), np.asarray(target_pose))
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device)
                 for value in values)


def _normalization(data, config) -> dict[str, np.ndarray]:
    state_mean = np.asarray(config.state_mean, dtype=np.float32)
    state_scale = np.asarray(config.state_scale, dtype=np.float32)
    command_mean = np.asarray(config.command_mean, dtype=np.float32)
    command_scale = np.asarray(config.command_scale, dtype=np.float32)
    delta_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    rate_by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    for run_id, refs in sorted(data.train_windows_by_horizon[20].items()):
        if run_id not in data.training_runs or not refs:
            continue
        pick = np.linspace(0, len(refs) - 1, min(4096, len(refs)), dtype=np.int64)
        for ref_index in pick:
            capture_index, sequence_index, source_row = refs[int(ref_index)]
            capture = data.captures[capture_index]
            begin = int(capture.bounds[sequence_index, 0]) + int(source_row)
            state_now = (_state(capture, begin) - state_mean) / state_scale
            state_next = (_state(capture, begin + 1) - state_mean) / state_scale
            delta = state_next - state_now
            delta_by_run[run_id].append(delta)
            rate_by_run[run_id].append(delta / DT_S)

    def moments(groups: dict[str, list[np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        per_run = [np.stack(rows) for _, rows in sorted(groups.items()) if rows]
        if len(per_run) < 5:
            raise RuntimeError("multirate normalization requires >=5 training runs")
        mean = np.mean([values.mean(axis=0) for values in per_run], axis=0)
        variance = np.mean([
            np.mean((values - mean) ** 2, axis=0) for values in per_run], axis=0)
        scale = np.maximum(np.sqrt(variance), 1e-3)
        return mean.astype(np.float32), scale.astype(np.float32)

    delta_mean, delta_scale = moments(delta_by_run)
    rate_mean, rate_scale = moments(rate_by_run)
    return {
        "state_mean": state_mean, "state_scale": state_scale,
        "command_mean": command_mean, "command_scale": command_scale,
        "delta_mean": delta_mean, "delta_scale": delta_scale,
        "rate_mean": rate_mean, "rate_scale": rate_scale,
    }


def _pose_midpoint(pose: torch.Tensor, body_normalized: torch.Tensor,
                   state_mean: torch.Tensor, state_scale: torch.Tensor,
                   dt: float) -> torch.Tensor:
    body = body_normalized[:, :3] * state_scale[:3] + state_mean[:3]
    u, v, r = body.unbind(dim=-1)
    x, y, heading = pose.unbind(dim=-1)
    mid_heading = heading + 0.5 * dt * r
    dx = dt * (torch.cos(mid_heading) * u - torch.sin(mid_heading) * v)
    dy = dt * (torch.sin(mid_heading) * u + torch.cos(mid_heading) * v)
    return torch.stack((x + dx, y + dy, heading + dt * r), dim=-1)


def _ode_advance(model: ContinuousStateVectorField, current: torch.Tensor,
                 previous: torch.Tensor, command: torch.Tensor,
                 previous_command: torch.Tensor, pose: torch.Tensor,
                 state_mean: torch.Tensor, state_scale: torch.Tensor,
                 substeps: int) -> tuple[torch.Tensor, torch.Tensor]:
    h = DT_S / substeps
    state = current
    current_pose = pose
    for _ in range(substeps):
        first_rate = model(state, previous, command, previous_command)
        midpoint_state = state + (0.5 * h) * first_rate
        second_rate = model(midpoint_state, previous, command, previous_command)
        current_pose = _pose_midpoint(current_pose, midpoint_state,
                                      state_mean, state_scale, h)
        state = state + h * second_rate
    return state, current_pose


def _discrete_advance(model: StateDeltaModel, current: torch.Tensor,
                      previous: torch.Tensor, command: torch.Tensor,
                      previous_command: torch.Tensor, pose: torch.Tensor,
                      state_mean: torch.Tensor, state_scale: torch.Tensor,
                      pose_integrator: PoseIntegrator
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    state = current + model(current, previous, command, previous_command)
    body_midpoint = 0.5 * (current[:, :3] + state[:, :3])
    body_midpoint = body_midpoint * state_scale[:3] + state_mean[:3]
    return state, pose_integrator(pose, body_midpoint)


def _loss(state_prediction: torch.Tensor, pose_prediction: torch.Tensor,
          target_state: torch.Tensor, target_pose: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    state_loss = F.smooth_l1_loss(state_prediction, target_state)
    position_error = torch.linalg.vector_norm(
        pose_prediction[..., :2] - target_pose[..., :2], dim=-1) / POSITION_SCALE_M
    heading_delta = pose_prediction[..., 2] - target_pose[..., 2]
    heading_error = torch.atan2(torch.sin(heading_delta), torch.cos(heading_delta)) \
        / HEADING_SCALE_RAD
    position_loss = F.smooth_l1_loss(position_error, torch.zeros_like(position_error))
    heading_loss = F.smooth_l1_loss(heading_error, torch.zeros_like(heading_error))
    total = state_loss + 0.25 * position_loss + 0.5 * heading_loss
    return total, {
        "state": float(state_loss.detach()),
        "position": float(position_loss.detach()),
        "heading": float(heading_loss.detach()),
        "total": float(total.detach()),
    }


def _train_model(name: str, model: nn.Module, data, draw_batches,
                 norm: dict[str, np.ndarray], device: torch.device,
                 substeps: int | None) -> dict[str, Any]:
    tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=device)
               for key, value in norm.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    pose_integrator = PoseIntegrator(DT_S).to(device)
    losses: list[dict[str, float]] = []
    gradient_norms: list[float] = []
    started = time.perf_counter()
    model.train()
    for update, refs in enumerate(draw_batches, 1):
        arrays = _transition_arrays(
            data, refs, norm["state_mean"], norm["state_scale"],
            norm["command_mean"], norm["command_scale"], device)
        current, previous, command, previous_command, target, pose, target_pose = arrays
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, ContinuousStateVectorField):
            state_prediction, pose_prediction = _ode_advance(
                model, current, previous, command, previous_command, pose,
                tensors["state_mean"], tensors["state_scale"],
                TRAINING_INTEGRATION_SUBSTEPS)
        else:
            state_prediction, pose_prediction = _discrete_advance(
                model, current, previous, command, previous_command, pose,
                tensors["state_mean"], tensors["state_scale"], pose_integrator)
        loss, parts = _loss(state_prediction, pose_prediction, target, target_pose)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: non-finite loss at update {update}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not np.isfinite(gradient_norm):
            raise RuntimeError(f"{name}: non-finite gradient at update {update}")
        optimizer.step()
        losses.append(parts)
        gradient_norms.append(gradient_norm)
        if update % 50 == 0 or update == len(draw_batches):
            print(f"{name} update {update}/{len(draw_batches)} "
                  f"loss={parts['total']:.5f} raw_grad={gradient_norm:.3f} "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    model.eval()
    return {
        "model": name,
        "updates": len(draw_batches),
        "batch_size": BATCH_SIZE,
        "sampled_transition_count": len(draw_batches) * BATCH_SIZE,
        "mean_last_25_losses": {
            key: float(np.mean([row[key] for row in losses[-25:]]))
            for key in losses[-1]
        },
        "mean_gradient_norm": float(np.mean(gradient_norms)),
        "gradient_norm_p95": float(np.quantile(gradient_norms, 0.95)),
        "fraction_gradient_clipped_at_10": float(np.mean(
            np.asarray(gradient_norms) > 10.0)),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _wrap_np(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _metrics(prediction: dict[str, np.ndarray], truth_state: np.ndarray,
             truth_pose: np.ndarray, horizon: int) -> dict[str, float]:
    state_error = prediction["state"][:, :horizon] - truth_state[:, :horizon]
    pose_error = prediction["pose"][:, :horizon] - truth_pose[:, :horizon]
    return {
        "position_radial_trajectory_rmse_m": float(np.sqrt(np.mean(
            np.sum(pose_error[..., :2] ** 2, axis=-1)))),
        "heading_trajectory_rmse_rad": float(np.sqrt(np.mean(
            _wrap_np(pose_error[..., 2]) ** 2))),
        "u_rmse_mps": float(np.sqrt(np.mean(state_error[..., 0] ** 2))),
        "v_rmse_mps": float(np.sqrt(np.mean(state_error[..., 1] ** 2))),
        "yaw_rate_rmse_rps": float(np.sqrt(np.mean(state_error[..., 2] ** 2))),
        "steering_feedback_rmse_rad": float(np.sqrt(np.mean(state_error[..., 3] ** 2))),
        "throttle_feedback_rmse": float(np.sqrt(np.mean(state_error[..., 4] ** 2))),
    }


def _rollout_run(data, run_id: str, refs, model: nn.Module, model_kind: str,
                 substeps: int, norm: dict[str, np.ndarray], device: torch.device,
                 horizon: int) -> dict[str, Any]:
    sequence_groups: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for ref in refs:
        sequence_groups[(int(ref[0]), int(ref[1]))].append(ref)
    if len(sequence_groups) > 1:
        parts = [
            _rollout_run(data, run_id, group, model, model_kind, substeps,
                         norm, device, horizon)
            for _, group in sorted(sequence_groups.items())
        ]
        prediction = {
            key: np.concatenate([part["prediction"][key] for part in parts], axis=0)
            for key in ("state", "pose")
        }
        truth_state = np.concatenate([part["truth_state"] for part in parts], axis=0)
        truth_pose = np.concatenate([part["truth_pose"] for part in parts], axis=0)
        return {
            "run_id": run_id,
            "start_count": sum(part["start_count"] for part in parts),
            "horizons": {
                str(steps): _metrics(prediction, truth_state, truth_pose, steps)
                for steps in (1, 10, 30, 80) if steps <= horizon
            },
            "prediction": prediction,
            "truth_state": truth_state,
            "truth_pose": truth_pose,
        }
    capture_index, sequence_index, _ = refs[0]
    capture = data.captures[capture_index]
    starts = []
    for ci, si, source_row in refs:
        if ci != capture_index or si != sequence_index:
            raise RuntimeError("evaluation run group crossed a sequence unexpectedly")
        begin = int(capture.bounds[si, 0]) + int(source_row)
        starts.append(begin)
    if any(begin + horizon >= int(capture.bounds[sequence_index, 1]) for begin in starts):
        raise RuntimeError(f"{run_id}: evaluation start lacks full {horizon}-step target")

    current_raw = np.asarray([_state(capture, begin) for begin in starts])
    previous_raw = np.asarray([_state(capture, begin - 1) for begin in starts])
    current = torch.as_tensor((current_raw - norm["state_mean"]) / norm["state_scale"],
                              dtype=torch.float32, device=device)
    previous = torch.as_tensor((previous_raw - norm["state_mean"]) / norm["state_scale"],
                               dtype=torch.float32, device=device)
    initial_commands = np.asarray([capture.frames[begin - 1, 7:9] for begin in starts])
    previous_command = torch.as_tensor(
        (initial_commands - norm["command_mean"]) / norm["command_scale"],
        dtype=torch.float32, device=device)
    current_pose = torch.as_tensor(
        np.asarray([data.poses[capture_index][begin] for begin in starts]),
        dtype=torch.float32, device=device)
    commands = np.stack([
        capture.frames[begin:begin + horizon, 7:9] for begin in starts])
    truth_state = np.stack([
        np.asarray([_state(capture, begin + step + 1) for step in range(horizon)])
        for begin in starts])
    truth_pose = np.stack([
        data.poses[capture_index][begin + 1:begin + horizon + 1] for begin in starts])
    commands_t = torch.as_tensor(
        (commands - norm["command_mean"]) / norm["command_scale"],
        dtype=torch.float32, device=device)
    state_predictions, pose_predictions = [], []
    pose_integrator = PoseIntegrator(DT_S).to(device)
    model.eval()
    with torch.no_grad():
        for step in range(horizon):
            command = commands_t[:, step]
            if model_kind == "ode":
                next_state, next_pose = _ode_advance(
                    model, current, previous, command, previous_command, current_pose,
                    torch.as_tensor(norm["state_mean"], dtype=torch.float32, device=device),
                    torch.as_tensor(norm["state_scale"], dtype=torch.float32, device=device),
                    substeps)
            else:
                next_state, next_pose = _discrete_advance(
                    model, current, previous, command, previous_command, current_pose,
                    torch.as_tensor(norm["state_mean"], dtype=torch.float32, device=device),
                    torch.as_tensor(norm["state_scale"], dtype=torch.float32, device=device),
                    pose_integrator)
            if not torch.isfinite(next_state).all() or not torch.isfinite(next_pose).all():
                raise RuntimeError(f"{run_id}: non-finite recursive rollout at step {step + 1}")
            state_predictions.append(next_state.cpu().numpy()
                                    * norm["state_scale"] + norm["state_mean"])
            pose_predictions.append(next_pose.cpu().numpy())
            previous, current = current, next_state
            previous_command = command
            current_pose = next_pose
    prediction = {
        "state": np.stack(state_predictions, axis=1),
        "pose": np.stack(pose_predictions, axis=1),
    }
    by_horizon = {
        str(steps): _metrics(prediction, truth_state, truth_pose, steps)
        for steps in (1, 10, 30, 80) if steps <= horizon
    }
    return {
        "run_id": run_id,
        "start_count": len(starts),
        "horizons": by_horizon,
        "prediction": prediction,
        "truth_state": truth_state,
        "truth_pose": truth_pose,
    }


def _bootstrap_mean(values: list[float], seed: int) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(array), size=(BOOTSTRAP_REPLICATES, len(array)))
    means = array[indexes].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _evaluate(data, models: dict[str, tuple[nn.Module, str]], norm,
              device: torch.device, horizon: int) -> dict[str, Any]:
    selected = {
        "development_validation": _select_eval_windows(data.validation_windows, 64),
        "practice_diagnostic": _select_eval_windows(data.practice_windows, 64),
    }
    results: dict[str, Any] = {}
    raw_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for variant, (model, kind) in models.items():
        results[variant] = {}
        raw_metrics[variant] = {}
        for role, runs in selected.items():
            results[variant][role] = {}
            raw_metrics[variant][role] = {}
            for run_id, refs in sorted(runs.items()):
                print(f"evaluate {variant} {role} {run_id} ({len(refs)} starts)",
                      flush=True)
                integration_steps = 25 if variant == "ode_1khz" else (
                    5 if variant == "ode_200hz" else 1)
                run_result = _rollout_run(data, run_id, refs, model, kind,
                                          integration_steps, norm, device, horizon)
                raw_metrics[variant][role][run_id] = run_result["horizons"]
                results[variant][role][run_id] = {
                    "start_count": run_result["start_count"],
                    "horizons": run_result["horizons"],
                }
    comparisons = {}
    baseline_name = "direct_40hz"
    metrics = tuple(next(iter(raw_metrics[baseline_name]["development_validation"].values()))["80"])
    for variant in ("ode_40hz", "ode_200hz", "ode_1khz"):
        comparisons[variant] = {}
        for role in ("development_validation", "practice_diagnostic"):
            comparisons[variant][role] = {}
            common_runs = sorted(set(raw_metrics[baseline_name][role])
                                 & set(raw_metrics[variant][role]))
            for horizon_key in ("1", "10", "30", "80"):
                if int(horizon_key) > horizon:
                    continue
                comparisons[variant][role][horizon_key] = {}
                for metric_index, metric in enumerate(metrics):
                    deltas = {
                        run_id: (raw_metrics[variant][role][run_id][horizon_key][metric]
                                 - raw_metrics[baseline_name][role][run_id][horizon_key][metric])
                        for run_id in common_runs
                    }
                    comparisons[variant][role][horizon_key][metric] = {
                        "direction": "candidate minus discrete baseline; negative favors candidate",
                        "run_macro_delta": float(np.mean(list(deltas.values()))),
                        "run_cluster_bootstrap_95pct_ci": _bootstrap_mean(
                            list(deltas.values()), SEED + metric_index + int(horizon_key)),
                        "independent_run_count": len(deltas),
                        "per_run_delta": deltas,
                    }
    return {
        "independent_unit": "run; windows are averaged within run before comparisons",
        "primary_split": "development_validation",
        "practice_role": "diagnostic only; not used for model selection",
        "horizon_steps": [1, 10, 30, 80],
        "horizon_seconds": [steps * DT_S for steps in (1, 10, 30, 80)],
        "per_run": results,
        "paired_vs_direct_40hz": comparisons,
    }


def run(updates: int = DEFAULT_UPDATES, device_name: str = "cpu",
        output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if _git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"expected frozen base {SOURCE_COMMIT}; found {_git_head()}")
    if updates < 1:
        raise ValueError("updates must be positive")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite multirate study: {output_root}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm = _normalization(data, config)
    # Verify the input supervision truly is fixed 40 Hz in the eligible plant data.
    dt_summary = {}
    for capture in data.captures:
        valid_dt = capture.dt_s[np.isfinite(capture.dt_s)]
        dt_summary[capture.name] = {
            "rows": int(len(capture.frames)),
            "unique_dt_seconds": np.unique(valid_dt).tolist(),
            "nominal_frequency_hz": float(1.0 / np.median(valid_dt)),
        }
    if any(not np.allclose(capture.dt_s, DT_S, rtol=0.0, atol=1e-7)
           for capture in data.captures):
        raise RuntimeError("loaded capture contains non-25-ms intervals; segment timing first")

    rng = np.random.default_rng(SEED)
    counters = {"family": Counter(), "run": Counter(), "condition": Counter()}
    draw_batches = [
        _sample_windows(data, 20, BATCH_SIZE, rng, counters)
        for _ in range(updates)
    ]
    draw_payload = [[[int(item) for item in ref] for ref in batch]
                    for batch in draw_batches]
    draw_hash = hashlib.sha256(json.dumps(
        draw_payload, separators=(",", ":")).encode()).hexdigest()

    models: dict[str, tuple[nn.Module, str]] = {}
    training_reports = {}
    for name, model_class, kind, substeps in (
        ("direct_40hz", StateDeltaModel, "discrete", None),
        ("ode_1khz", ContinuousStateVectorField, "ode", TRAINING_INTEGRATION_SUBSTEPS),
    ):
        torch.manual_seed(SEED)
        if model_class is StateDeltaModel:
            model = model_class(norm["delta_mean"], norm["delta_scale"]).to(device)
        else:
            model = model_class(norm["rate_mean"], norm["rate_scale"]).to(device)
        print(f"train {name}: {updates} updates, batch={BATCH_SIZE}, "
              f"shared_draw_sha256={draw_hash}", flush=True)
        training_reports[name] = _train_model(name, model, data, draw_batches,
                                               norm, device, substeps)
        models[name] = (model, kind)

    # Evaluate one fixed vector field at 40 Hz, 200 Hz, and 1 kHz. This isolates
    # numerical integration resolution from learned parameters and data inputs.
    ode_model = models["ode_1khz"][0]
    models["ode_40hz"] = (ode_model, "ode")
    models["ode_200hz"] = (ode_model, "ode")
    models["ode_1khz"] = (ode_model, "ode")
    evaluation = _evaluate(data, models, norm, device, EVAL_HORIZON_STEPS)

    output_root.mkdir(parents=True, exist_ok=False)
    source_files = {
        "openplane_dataset": DEFAULT_DYNAMIC,
        "openplane_fixed_encoder_sidecar": DEFAULT_DYNAMIC_FIXED,
        "practice_dataset": DEFAULT_PRACTICE,
        "practice_fixed_encoder_sidecar": DEFAULT_PRACTICE_FIXED,
    }
    dataset_manifest = {
        "split_policy": "train only for optimization; validation for primary selection; unseen practice diagnostic; test/final-test excluded",
        "training_runs": list(data.training_runs),
        "validation_runs": sorted(data.validation_windows),
        "practice_diagnostic_runs": sorted(data.practice_windows),
        "training_draw_count": len(draw_payload) * BATCH_SIZE,
        "training_draws_sha256": draw_hash,
        "files": {name: {"path": path.relative_to(ROOT).as_posix(),
                          "sha256": sha256_file(path)}
                  for name, path in source_files.items()},
    }
    checkpoint_metadata = {
        "source_commit": SOURCE_COMMIT,
        "seed": SEED,
        "history": "no autonomous learned latent; current/previous observable state and current/previous command",
        "control_period_seconds": DT_S,
        "training_ode_step_seconds": DT_S / TRAINING_INTEGRATION_SUBSTEPS,
        "training_ode_substeps_per_control": TRAINING_INTEGRATION_SUBSTEPS,
        "command_interpolation": "zero-order hold over each recorded 25 ms control interval",
        "state_order": ["u_rear_mps", "v_rear_mps", "yaw_rate_rps",
                        "steering_feedback_rad", "throttle_feedback_norm"],
        "normalization": {key: value.tolist() for key, value in norm.items()},
    }
    variant_dirs = {}
    for name, (model, kind) in models.items():
        if name not in ("direct_40hz", "ode_1khz"):
            continue
        variant_dir = output_root / name
        variant_dir.mkdir(parents=True, exist_ok=False)
        checkpoint_path = variant_dir / "checkpoint.pt"
        torch.save({"state_dict": model.state_dict(), "metadata": checkpoint_metadata,
                    "variant": name, "kind": kind}, checkpoint_path)
        checkpoint_hash = sha256_file(checkpoint_path)
        (variant_dir / "checkpoint.sha256").write_text(
            checkpoint_hash + "\n", encoding="utf-8")
        (variant_dir / "source_commit.txt").write_text(
            SOURCE_COMMIT + "\n", encoding="utf-8")
        _write_json(variant_dir / "metadata.json", {
            **checkpoint_metadata, "variant": name, "kind": kind,
            "checkpoint_sha256": checkpoint_hash,
        })
        _write_json(variant_dir / "training_report.json", training_reports[name])
        _write_json(variant_dir / "dataset_manifest.json", dataset_manifest)
        variant_dirs[name] = variant_dir

    _write_json(output_root / "evaluation_report.json", evaluation)
    _write_json(output_root / "dataset_manifest.json", dataset_manifest)
    report = {
        "study": "40Hz discrete vs continuous vector field integrated at multiple rates",
        "source_commit": SOURCE_COMMIT,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "optimizer_steps": updates * 2,
        "dataset_timing": dt_summary,
        "available_intermediate_state_labels": False,
        "control_sample_period_seconds": DT_S,
        "training_integration_substeps": TRAINING_INTEGRATION_SUBSTEPS,
        "training_internal_step_seconds": DT_S / TRAINING_INTEGRATION_SUBSTEPS,
        "evaluation_internal_step_seconds": {
            "ode_40hz": DT_S,
            "ode_200hz": DT_S / 5,
            "ode_1khz": DT_S / 25,
        },
        "input_hold_assumption": "command held constant between 40 Hz samples",
        "training_draws_sha256": draw_hash,
        "training_draws_shared_between_models": True,
        "training": training_reports,
        "evaluation": evaluation,
        "limits": [
            "Current captured state labels are exactly 40 Hz; 1 kHz substep states are latent model predictions, not measurements.",
            "This controlled study tests a memoryless continuous vector field conditioned on current/previous observable state and commands; it does not establish that hidden tire dynamics are Markov in this representation.",
            "Practice captures are diagnostics only and were not used for checkpoint selection.",
            "No test/final-test runs, future sensors, or future ground truth are used as rollout inputs.",
            "A successful numerical integration comparison is necessary but not sufficient for full-lap plant accuracy.",
        ],
    }
    report["output_directory"] = output_root.relative_to(ROOT).as_posix()
    _write_json(output_root / "multirate_study_report.json", report)
    _write_json(output_root / "training_sampler_draws.json", {
        "sha256": draw_hash, "draw_batches": draw_payload,
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = run(args.updates, args.device, output_root)
    print(json.dumps({
        "output": str(output_root),
        "training_steps": report["optimizer_steps"],
        "primary_2s_validation": {
            name: {
                run_id: metrics["horizons"]["80"]
                for run_id, metrics in report["evaluation"]["per_run"][name][
                    "development_validation"].items()
            } for name in ("direct_40hz", "ode_40hz", "ode_200hz", "ode_1khz")
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
