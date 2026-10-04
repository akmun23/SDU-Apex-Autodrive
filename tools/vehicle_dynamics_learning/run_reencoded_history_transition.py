#!/usr/bin/env python3
"""Train a causal history-conditioned discrete plant on fixed 40 Hz data.

This is a research-only model. It uses the previous two seconds of observable
history for its initial state, then rolls forward only its own predicted
vehicle/actuator state and the supplied command sequence. It changes no
simulator, production odometry, localization, MPC, or actuator code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    HISTORY_STEPS,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _normalization,
    _state,
    _write_json,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _load_data,
    _predict_wp19_baseline,
    _sample_windows,
    _select_eval_windows,
    _training_windows_and_stats,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
             / "full_modeling_reset_20261001"
             / "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1")
OUTPUT_ROOT = TASK_ROOT / "next_phase_after_2129427/reencoded_history_transition_v1"
FROZEN_EVAL_STARTS = (TASK_ROOT / "next_phase_after_2129427/frozen_eval_starts.json")
WP19_CHECKPOINT = (TASK_ROOT / "wp19_target_ablation_v2_common_encoder_mask"
                   / "07_body_state_increment__encoder_angle_increment/model.pt")
SOURCE_COMMIT = "2129427838101eee277e17e91727810bbe2df687"
SEED = 20261003
HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
BOOTSTRAP_REPLICATES = 5000
STAGES = ((20, 40, 8), (80, 60, 8), (200, 40, 4))
EVAL_HORIZONS = (1, 10, 30, 80, 200, 400)


class HistoryTransition(nn.Module):
    """Direct 25 ms delta map conditioned on a bounded, explicit history."""

    def __init__(self, delta_mean: np.ndarray, delta_scale: np.ndarray) -> None:
        super().__init__()
        input_size = HISTORY_STEPS * 7 + 5 + 2
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

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                command: torch.Tensor) -> torch.Tensor:
        features = torch.cat((history.flatten(start_dim=1), state, command), dim=1)
        return self.delta_mean + self.delta_scale * self.net(features)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()


def _batch_arrays(data, refs, horizon: int, norm: dict[str, np.ndarray],
                  config, device: torch.device) -> tuple[torch.Tensor, ...]:
    histories, states, poses, commands, targets, target_poses = [], [], [], [], [], []
    history_mean = np.asarray(config.history_mean[:7], dtype=np.float32)
    history_scale = np.asarray(config.history_scale[:7], dtype=np.float32)
    for capture_index, sequence_index, source_row in refs:
        capture = data.captures[capture_index]
        begin = int(capture.bounds[sequence_index, 0]) + int(source_row)
        if begin - HISTORY_STEPS + 1 < int(capture.bounds[sequence_index, 0]):
            raise ValueError("history crosses a sequence/reset boundary")
        if begin + horizon >= int(capture.bounds[sequence_index, 1]):
            raise ValueError("training/evaluation window lacks the requested target horizon")
        histories.append(capture.input_features[
            begin - HISTORY_STEPS + 1:begin + 1, :7])
        states.append(_state(capture, begin))
        poses.append(data.poses[capture_index][begin])
        commands.append(capture.frames[begin:begin + horizon, 7:9])
        targets.append(np.stack([
            _state(capture, row)
            for row in range(begin + 1, begin + horizon + 1)]))
        target_poses.append(data.poses[capture_index][begin + 1:begin + horizon + 1])
    history = (np.asarray(histories, dtype=np.float32) - history_mean) / history_scale
    state = (np.asarray(states, dtype=np.float32) - norm["state_mean"]) / norm["state_scale"]
    command = (np.asarray(commands, dtype=np.float32) - norm["command_mean"]) \
        / norm["command_scale"]
    target = (np.asarray(targets, dtype=np.float32) - norm["state_mean"]) \
        / norm["state_scale"]
    values = (history, state, np.asarray(poses, dtype=np.float32), command,
              target, np.asarray(target_poses, dtype=np.float32))
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device)
                 for value in values)


def _predicted_history_row(state: torch.Tensor, next_command: torch.Tensor,
                           norm: dict[str, torch.Tensor]) -> torch.Tensor:
    physical = state * norm["state_scale"] + norm["state_mean"]
    raw = torch.cat((physical, next_command * norm["command_scale"]
                     + norm["command_mean"]), dim=-1)
    return (raw - norm["history_mean"]) / norm["history_scale"]


def _shift_history(history: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    if history.ndim != 3 or history.shape[1] != HISTORY_STEPS \
            or row.shape != (history.shape[0], history.shape[2]):
        raise ValueError("history shift requires aligned batch x 80 x feature tensors")
    return torch.cat((history[:, 1:], row[:, None, :]), dim=1)


def _rollout(model: HistoryTransition, arrays: tuple[torch.Tensor, ...],
             norm: dict[str, torch.Tensor], horizon: int,
             pose_integrator: PoseIntegrator,
             collect_loss: bool = False,
             detach_every_steps: int | None = None,
             backward_segments: bool = False) -> dict[str, torch.Tensor]:
    if backward_segments and (not collect_loss or detach_every_steps is None):
        raise ValueError("segmented backward requires a loss and segment length")
    if detach_every_steps is not None and detach_every_steps < 1:
        raise ValueError("gradient segment length must be positive")
    history, state, pose, commands, target, target_pose = arrays
    states, poses, losses = [], [], []
    accumulated_loss = state.new_zeros(())
    for step in range(horizon):
        next_state = state + model(history, state, commands[:, step])
        current_body = state[:, :3] * norm["state_scale"][:3] \
            + norm["state_mean"][:3]
        next_body = next_state[:, :3] * norm["state_scale"][:3] \
            + norm["state_mean"][:3]
        next_pose = pose_integrator(pose, 0.5 * (current_body + next_body))
        if not torch.isfinite(next_state).all() or not torch.isfinite(next_pose).all():
            raise FloatingPointError(f"non-finite recursive output at step {step + 1}")
        states.append(next_state.detach() if backward_segments else next_state)
        poses.append(next_pose.detach() if backward_segments else next_pose)
        if collect_loss:
            state_loss = F.smooth_l1_loss(
                (next_state - target[:, step]) , torch.zeros_like(next_state))
            position_error = torch.linalg.vector_norm(
                next_pose[:, :2] - target_pose[:, step, :2], dim=-1) / 0.5
            heading_error = next_pose[:, 2] - target_pose[:, step, 2]
            heading_error = torch.atan2(torch.sin(heading_error),
                                        torch.cos(heading_error)) / 0.1
            position_loss = F.smooth_l1_loss(
                position_error, torch.zeros_like(position_error))
            heading_loss = F.smooth_l1_loss(
                heading_error, torch.zeros_like(heading_error))
            step_loss = state_loss + 0.25 * position_loss + 0.5 * heading_loss
            if backward_segments:
                losses.append(step_loss)
                segment_end = ((step + 1) % detach_every_steps == 0
                               or step + 1 == horizon)
                if segment_end:
                    segment_loss = torch.stack(losses).sum() / horizon
                    segment_loss.backward()
                    accumulated_loss = accumulated_loss + segment_loss.detach()
                    losses.clear()
        if step + 1 < horizon:
            if (backward_segments and detach_every_steps is not None
                    and (step + 1) % detach_every_steps == 0):
                history = history.detach()
                next_state = next_state.detach()
                next_pose = next_pose.detach()
            history_row = _predicted_history_row(
                next_state, commands[:, step + 1], norm)
            history = _shift_history(history, history_row)
        state, pose = next_state, next_pose
    return {"state": torch.stack(states, dim=1),
            "pose": torch.stack(poses, dim=1),
            "loss": (accumulated_loss if backward_segments else
                     torch.stack(losses).mean() if losses else state.new_zeros(()))}


def _train(data, model: HistoryTransition, norm_np: dict[str, np.ndarray],
           config, device: torch.device) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    norm = {"state_mean": torch.as_tensor(norm_np["state_mean"], device=device),
            "state_scale": torch.as_tensor(norm_np["state_scale"], device=device),
            "command_mean": torch.as_tensor(norm_np["command_mean"], device=device),
            "command_scale": torch.as_tensor(norm_np["command_scale"], device=device),
            "history_mean": torch.as_tensor(config.history_mean[:7], device=device),
            "history_scale": torch.as_tensor(config.history_scale[:7], device=device)}
    pose_integrator = PoseIntegrator(DT_S).to(device)
    rng = np.random.default_rng(SEED)
    counters = {key: Counter() for key in ("family", "run", "condition")}
    records, all_draws = [], []
    update = 0
    started = time.perf_counter()
    for horizon, updates, batch_size in STAGES:
        stage_losses, stage_gradients, stage_draws = [], [], []
        model.train()
        for _ in range(updates):
            update += 1
            refs = _sample_windows(data, horizon, batch_size, rng, counters)
            arrays = _batch_arrays(data, refs, horizon, norm_np, config, device)
            optimizer.zero_grad(set_to_none=True)
            result = _rollout(model, arrays, norm, horizon, pose_integrator,
                              collect_loss=True)
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {update}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at update {update}")
            optimizer.step()
            stage_losses.append(float(loss.detach()))
            stage_gradients.append(grad_norm)
            stage_draws.append([[int(value) for value in ref] for ref in refs])
            if update % 20 == 0:
                print(f"update {update} horizon={horizon} loss={stage_losses[-1]:.5f} "
                      f"grad={grad_norm:.3f} elapsed={time.perf_counter()-started:.1f}s",
                      flush=True)
        records.append({
            "horizon_steps": horizon,
            "updates": updates,
            "batch_size": batch_size,
            "mean_loss_last_10": float(np.mean(stage_losses[-10:])),
            "mean_gradient_norm": float(np.mean(stage_gradients)),
            "gradient_norm_p95": float(np.quantile(stage_gradients, 0.95)),
            "draws": stage_draws,
        })
        all_draws.extend(stage_draws)
    return {
        "stages": [{key: value for key, value in row.items() if key != "draws"}
                   for row in records],
        "updates": update,
        "draw_batches": all_draws,
        "sampler_counts": {key: dict(value) for key, value in counters.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }


def _wrap(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _metrics(predicted_state: np.ndarray, predicted_pose: np.ndarray,
             truth_state: np.ndarray, truth_pose: np.ndarray,
             horizon: int, include_actuator: bool = True) -> dict[str, float]:
    state_error = predicted_state[:, :horizon] - truth_state[:, :horizon]
    pose_error = predicted_pose[:, :horizon] - truth_pose[:, :horizon]
    heading_error = _wrap(pose_error[..., 2])
    position = np.linalg.norm(pose_error[..., :2], axis=-1)
    result = {
        "position_radial_trajectory_rmse_m": float(np.mean(np.sqrt(
            np.mean(position ** 2, axis=1)))),
        "position_endpoint_error_m": float(np.mean(position[:, -1])),
        "heading_trajectory_rmse_rad": float(np.mean(np.sqrt(
            np.mean(heading_error ** 2, axis=1)))),
        "heading_endpoint_error_rad": float(np.mean(np.abs(heading_error[:, -1]))),
        "u_rmse_mps": float(np.mean(np.sqrt(
            np.mean(state_error[..., 0] ** 2, axis=1)))),
        "v_rmse_mps": float(np.mean(np.sqrt(
            np.mean(state_error[..., 1] ** 2, axis=1)))),
        "yaw_rate_rmse_rps": float(np.mean(np.sqrt(
            np.mean(state_error[..., 2] ** 2, axis=1)))),
    }
    if include_actuator:
        result["steering_feedback_rmse_rad"] = float(np.mean(np.sqrt(
            np.mean(state_error[..., 3] ** 2, axis=1))))
        result["throttle_feedback_rmse"] = float(np.mean(np.sqrt(
            np.mean(state_error[..., 4] ** 2, axis=1))))
    return result


def _predict_run(data, refs, model, norm_np, config, horizon, device):
    arrays = _batch_arrays(data, refs, horizon, norm_np, config, device)
    history, state, pose, commands, truth_state, truth_pose = arrays
    tensors = {key: torch.as_tensor(value, dtype=torch.float32, device=device)
               for key, value in norm_np.items()}
    tensors["history_mean"] = torch.as_tensor(
        config.history_mean[:7], dtype=torch.float32, device=device)
    tensors["history_scale"] = torch.as_tensor(
        config.history_scale[:7], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        result = _rollout(model, (history, state, pose, commands,
                                  truth_state, truth_pose), tensors, horizon,
                          PoseIntegrator(DT_S).to(device))
    predicted_state = (result["state"].cpu().numpy() * norm_np["state_scale"]
                       + norm_np["state_mean"])
    return (predicted_state, result["pose"].cpu().numpy(),
            truth_state.cpu().numpy() * norm_np["state_scale"] + norm_np["state_mean"],
            truth_pose.cpu().numpy())


def _bootstrap(deltas: dict[str, float], seed: int) -> list[float] | None:
    values = np.asarray(list(deltas.values()), dtype=np.float64)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values),
                         size=(BOOTSTRAP_REPLICATES, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def _pose_step_numpy(pose: np.ndarray, body: np.ndarray, dt: float) -> np.ndarray:
    x, y, heading = np.moveaxis(pose, -1, 0)
    u, v, yaw_rate = np.moveaxis(body, -1, 0)
    angle = yaw_rate * dt
    safe_rate = np.where(np.abs(yaw_rate) > 1e-7, yaw_rate, 1.0)
    dx = np.where(np.abs(yaw_rate) > 1e-7,
                  (u * np.sin(angle) + v * (np.cos(angle) - 1.0)) / safe_rate,
                  u * dt)
    dy = np.where(np.abs(yaw_rate) > 1e-7,
                  (u * (1.0 - np.cos(angle)) + v * np.sin(angle)) / safe_rate,
                  v * dt)
    return np.stack((x + np.cos(heading) * dx - np.sin(heading) * dy,
                     y + np.sin(heading) * dx + np.cos(heading) * dy,
                     heading + angle), axis=-1)


def _pose_integration_diagnostic(data) -> dict[str, Any]:
    """Compare 25 ms pose updates using held, midpoint, and interpolated truth."""
    collected: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        "development_validation": {}, "practice_diagnostic": {}}
    for capture_index, capture in enumerate(data.captures):
        for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
                zip(capture.bounds, capture.sequence_run)):
            begin, end, run_index = int(begin_raw), int(end_raw), int(run_raw)
            split = str(capture.splits[run_index])
            if split == "validation":
                role = "development_validation"
            elif split == "unseen_practice":
                role = "practice_diagnostic"
            else:
                continue
            body = capture.body[begin:end]
            pose = data.poses[capture_index][begin:end]
            valid = (len(body) > 1 and np.isfinite(body).all()
                     and np.isfinite(pose).all())
            if not valid:
                continue
            held = _pose_step_numpy(pose[:-1], body[:-1], DT_S)
            midpoint = _pose_step_numpy(
                pose[:-1], 0.5 * (body[:-1] + body[1:]), DT_S)
            fine = pose[:-1].copy()
            for substep in range(25):
                fraction = (substep + 0.5) / 25.0
                body_at_midpoint = body[:-1] + fraction * (body[1:] - body[:-1])
                fine = _pose_step_numpy(fine, body_at_midpoint, DT_S / 25.0)
            truth = pose[1:]
            run_id = str(capture.run_ids[run_index])
            run_methods = collected[role].setdefault(run_id, [])
            run_methods.append((truth, held, midpoint, fine))

    report: dict[str, Any] = {
        "purpose": "isolate sampled-pose integration from learned vehicle dynamics",
        "data_rate_hz": 1.0 / DT_S,
        "fine_rate_hz": 1000,
        "fine_method": "25 exact SE(2) updates with linearly interpolated measured endpoint body states",
        "fine_method_is_causal_or_deployable": False,
        "fine_method_role": "acausal interpolation oracle; uses the next 40 Hz truth state only for this diagnostic",
        "per_role": {},
    }
    for role, runs in collected.items():
        per_run: dict[str, dict[str, float]] = {}
        for run_id, sequences in runs.items():
            truth = np.concatenate([item[0] for item in sequences])
            methods = {"held_40hz": np.concatenate([item[1] for item in sequences]),
                       "midpoint_40hz": np.concatenate([item[2] for item in sequences]),
                       "interpolated_1khz_oracle": np.concatenate(
                           [item[3] for item in sequences])}
            per_run[run_id] = {}
            for name, prediction in methods.items():
                position_error = prediction[:, :2] - truth[:, :2]
                heading_error = _wrap(prediction[:, 2] - truth[:, 2])
                per_run[run_id][name + "_position_rmse_m"] = float(np.sqrt(
                    np.mean(np.sum(position_error ** 2, axis=1))))
                per_run[run_id][name + "_heading_rmse_rad"] = float(np.sqrt(
                    np.mean(heading_error ** 2)))
        comparisons = {}
        for candidate, baseline in (("midpoint_40hz", "held_40hz"),
                                    ("interpolated_1khz_oracle", "midpoint_40hz")):
            comparisons[candidate + "_minus_" + baseline] = {}
            for metric in ("position_rmse_m", "heading_rmse_rad"):
                deltas = {
                    run_id: values[candidate + "_" + metric]
                    - values[baseline + "_" + metric]
                    for run_id, values in per_run.items()}
                comparisons[candidate + "_minus_" + baseline][metric] = {
                    "run_macro_delta": float(np.mean(list(deltas.values()))),
                    "run_cluster_bootstrap_95pct_ci": _bootstrap(
                        deltas, SEED + len(deltas) + len(metric)),
                    "independent_run_count": len(deltas),
                    "per_run_delta": deltas,
                }
        report["per_role"][role] = {
            "per_run": per_run,
            "run_macro_mean": {
                method: {
                    metric: float(np.mean([values[method + "_" + metric]
                                           for values in per_run.values()]))
                    for metric in ("position_rmse_m", "heading_rmse_rad")}
                for method in ("held_40hz", "midpoint_40hz",
                               "interpolated_1khz_oracle")},
            "paired_run_comparisons": comparisons,
        }
    return report


def _evaluate(data, model, norm_np, config, device) -> dict[str, Any]:
    selected = {
        "development_validation": _select_eval_windows(data.validation_windows, 64),
        "practice_diagnostic": _select_eval_windows(data.practice_windows, 64),
    }
    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    for role, runs in selected.items():
        expected_runs = frozen["split_roles"][role]
        if set(runs) != set(expected_runs):
            raise RuntimeError(f"{role}: run set differs from frozen WP24 evaluation")
        for run_id, refs in runs.items():
            actual = {(int(sequence), int(row)) for _, sequence, row in refs}
            expected = {(int(item["sequence_index"]), int(item["source_row"]))
                        for item in expected_runs[run_id]}
            if actual != expected:
                raise RuntimeError(f"{run_id}: start rows differ from frozen WP24 evaluation")
    wp19_checkpoint = torch.load(WP19_CHECKPOINT, map_location=device,
                                 weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_checkpoint["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    per_run: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for role, runs in selected.items():
        per_run[role] = {}
        for run_id, refs in sorted(runs.items()):
            capture_index = int(refs[0][0])
            horizon = 200 if role == "practice_diagnostic" else 400
            print(f"evaluate {role} {run_id}, {len(refs)} starts, {horizon*DT_S:.1f}s",
                  flush=True)
            candidate = _predict_run(data, refs, model, norm_np, config,
                                     horizon, device)
            baseline = _predict_wp19_baseline(
                data.captures[capture_index], data.poses[capture_index], refs,
                horizon, wp19_stats, wp19, device)
            truth_state = candidate[2]
            truth_pose = candidate[3]
            horizons = [value for value in EVAL_HORIZONS if value <= horizon]
            per_run[role][run_id] = {"starts": len(refs), "horizons": {}}
            for steps in horizons:
                candidate_metrics = _metrics(candidate[0], candidate[1],
                                             truth_state, truth_pose, steps)
                baseline_metrics = _metrics(
                    np.concatenate((baseline["body"], truth_state[..., 3:5]), axis=-1),
                    baseline["pose"], truth_state, truth_pose, steps,
                    include_actuator=False)
                per_run[role][run_id]["horizons"][str(steps)] = {
                    "history_transition": candidate_metrics,
                    "wp19_parent": baseline_metrics,
                }
        paired[role] = {}
        horizon_keys = [str(value) for value in EVAL_HORIZONS
                        if value <= (200 if role == "practice_diagnostic" else 400)]
        for horizon_key in horizon_keys:
            paired[role][horizon_key] = {}
            for metric in ("position_radial_trajectory_rmse_m", "position_endpoint_error_m",
                           "heading_trajectory_rmse_rad", "u_rmse_mps",
                           "v_rmse_mps", "yaw_rate_rmse_rps"):
                deltas = {
                    run_id: (values["horizons"][horizon_key]["wp19_parent"][metric]
                             - values["horizons"][horizon_key]["history_transition"][metric])
                    for run_id, values in per_run[role].items()
                }
                paired[role][horizon_key][metric] = {
                    "direction": "WP19 parent minus candidate; positive favors candidate",
                    "run_macro_delta": float(np.mean(list(deltas.values()))),
                    "run_cluster_bootstrap_95pct_ci": _bootstrap(
                        deltas, SEED + int(horizon_key) + len(metric)),
                    "independent_run_count": len(deltas),
                    "per_run_delta": deltas,
                }
    return {
        "independent_unit": "whole run; windows averaged within run",
        "development_validation_role": "primary model comparison",
        "practice_role": "unseen diagnostic only",
        "evaluation_start_source": FROZEN_EVAL_STARTS.relative_to(ROOT).as_posix(),
        "evaluation_starts_match_frozen_wp24_manifest": True,
        "training_or_selection_use_of_evaluation_runs": False,
        "per_run": per_run,
        "paired_wp19_parent_minus_candidate": paired,
    }


def run(device_name: str = "cpu", output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    if _git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"expected frozen source {SOURCE_COMMIT}; found {_git_head()}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_root}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    for capture in data.captures:
        if not np.allclose(capture.dt_s, DT_S, rtol=0.0, atol=1e-7):
            raise RuntimeError("eligible capture is not fixed 40 Hz")
    model = HistoryTransition(norm_np["delta_mean"],
                              norm_np["delta_scale"]).to(device)
    pose_integration = _pose_integration_diagnostic(data)
    training = _train(data, model, norm_np, config, device)
    evaluation = _evaluate(data, model, norm_np, config, device)

    draw_payload = training.pop("draw_batches")
    draw_hash = hashlib.sha256(json.dumps(
        draw_payload, separators=(",", ":")).encode()).hexdigest()
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_paths = {
        "openplane": DEFAULT_DYNAMIC,
        "openplane_fixed40hz_sidecar": DEFAULT_DYNAMIC_FIXED,
        "practice": DEFAULT_PRACTICE,
        "practice_fixed40hz_sidecar": DEFAULT_PRACTICE_FIXED,
    }
    from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import sha256_file
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "training_runs": list(data.training_runs),
        "validation_runs": sorted(data.validation_windows),
        "practice_diagnostic_runs": sorted(data.practice_windows),
        "sampler_draw_batches_sha256": draw_hash,
        "files": {key: {"path": path.relative_to(ROOT).as_posix(),
                         "sha256": sha256_file(path)}
                  for key, path in manifest_paths.items()},
        "split_policy": "train only for optimizer updates; held-out whole runs for evaluation; test/final-test excluded",
    }
    checkpoint_metadata = {
        "model": "bounded explicit 2-second history-conditioned 25-ms discrete transition",
        "source_commit": SOURCE_COMMIT,
        "seed": SEED,
        "history_steps": HISTORY_STEPS,
        "history_seconds": HISTORY_STEPS * DT_S,
        "history_feature_order": ["odom_u", "odom_v", "odom_yaw_rate",
                                  "steering_feedback", "throttle_feedback",
                                  "steering_command", "throttle_command"],
        "state_order": ["u_rear_mps", "v_rear_mps", "yaw_rate_rps",
                         "steering_feedback_rad", "throttle_feedback_norm"],
        "transition_target": "normalized 25 ms state increment",
        "pose_integration": "exact SE(2) constant-twist update using midpoint body state",
        "training_rollout_horizons_steps": [20, 80, 200],
        "future_truth_or_sensor_input": False,
        "normalization_train_only": {key: value.tolist() for key, value in norm_np.items()},
        "history_mean_first_7": list(config.history_mean[:7]),
        "history_scale_first_7": list(config.history_scale[:7]),
    }
    checkpoint_path = output_root / "checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": checkpoint_metadata},
               checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    (output_root / "checkpoint.sha256").write_text(checkpoint_hash + "\n",
                                                   encoding="utf-8")
    _write_json(output_root / "checkpoint_metadata.json", {
        **checkpoint_metadata, "checkpoint_sha256": checkpoint_hash})
    _write_json(output_root / "dataset_manifest.json", manifest)
    _write_json(output_root / "training_report.json", training)
    _write_json(output_root / "training_sampler_draws.json", {
        "sha256": draw_hash, "draw_batches": draw_payload})
    _write_json(output_root / "evaluation_report.json", evaluation)
    _write_json(output_root / "pose_integration_diagnostic.json", pose_integration)
    report = {
        "study": "autoregressive bounded-history transition with multistep free rollout",
        "source_commit": SOURCE_COMMIT,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "available_intermediate_state_labels": False,
        "data_control_rate_hz": 1.0 / DT_S,
        "internal_step_rate_hz": None,
        "training_updates": training["updates"],
        "training_stages": STAGES,
        "training_draw_batches_sha256": draw_hash,
        "training": training,
        "pose_integration_diagnostic": pose_integration,
        "evaluation": evaluation,
        "limits": [
            "Dataset supplies only 40 Hz state labels; this predicts one 25 ms transition and does not claim validated 1 kHz force dynamics.",
            "The 2 s sensor history initializes each rollout; later history rows use only predicted body/actuator state and the supplied command sequence.",
            "Practice runs are diagnostics only; validation runs are the primary comparison; test/final-test are not opened.",
            "The model is research-only and has not been accepted as an accurate offline simulator.",
        ],
        "output_directory": output_root.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
    }
    _write_json(output_root / "study_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    report = run(args.device, output)
    print(json.dumps({
        "output": report["output_directory"],
        "primary_5s_validation_position": {
            run_id: metrics["horizons"]["200"]["history_transition"][
                "position_radial_trajectory_rmse_m"]
            for run_id, metrics in report["evaluation"]["per_run"][
                "development_validation"].items()
        },
        "paired_5s_position_gain": report["evaluation"][
            "paired_wp19_parent_minus_candidate"]["development_validation"][
                "200"]["position_radial_trajectory_rmse_m"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
