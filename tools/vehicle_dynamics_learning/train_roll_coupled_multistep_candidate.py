#!/usr/bin/env python3
"""Train a recurrent roll-aware lateral residual against multi-step plant loss.

This follows the diagnostic that teacher-forced roll residual correction did
not transfer to recursive motion. The frozen body plant remains unchanged; a
small zero-initialized residual head is trained through 5 s rollouts, with roll
and roll-rate internally propagated and supervised from training-only labels.
Held-out runs are evaluated only after the fixed training budget completes.
"""

from __future__ import annotations

import json
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools.vehicle_dynamics_learning.evaluate_internal_roll_state_prediction import (
    CHECKPOINT as PARENT_CHECKPOINT,
    DT_S,
    _balanced_fit,
)
from tools.vehicle_dynamics_learning.diagnose_rigid_acceleration_sensor_residual_value import (
    _transition_rows,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    _batch_arrays,
    _normalization,
    _rollout,
    advance_context,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    _metrics,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_PRACTICE,
    _collect_horizon_windows,
    _load_data,
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    PoseIntegrator,
)


OUTPUT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/next_phase_after_2129427"
    / "history_context_sufficiency_v1"
    / "roll_coupled_multistep_candidate_v2_20261004")
EVAL_REPORT = OUTPUT / "evaluation.json"
TRAIN_REPORT = OUTPUT / "training.json"
DT_HORIZON = 200
TRAIN_UPDATES = 120
EVAL_EVERY = 10
BATCH_RUNS = 4
TRAIN_SEED = 20261005
COM_X_M = 0.15532
MAX_STARTS_PER_RUN = 16
EVAL_HORIZONS = (("2s", 80), ("5s", 200), ("10s", 400))


class LateralRollResidual(nn.Module):
    """Learn a bounded-scale lateral-only acceleration residual."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(-1)


def _body_step(state_norm: torch.Tensor, delta_norm: torch.Tensor,
               acceleration: torch.Tensor, norm_np: dict[str, np.ndarray]
               ) -> torch.Tensor:
    mean = torch.as_tensor(norm_np["state_mean"][:3], dtype=state_norm.dtype,
                           device=state_norm.device)
    scale = torch.as_tensor(norm_np["state_scale"][:3], dtype=state_norm.dtype,
                            device=state_norm.device)
    physical = state_norm[:, :3] * scale + mean
    u, v_rear, yaw_rate = physical.unbind(dim=-1)
    ax, ay, yaw_accel = acceleration.unbind(dim=-1)
    yaw_rate_next = yaw_rate + yaw_accel * DT_S
    yaw_mid = 0.5 * (yaw_rate + yaw_rate_next)
    v_com = v_rear + COM_X_M * yaw_rate
    half = 0.5 * DT_S * yaw_mid
    determinant = 1.0 + half.square()
    rhs_u = u + half * v_com + DT_S * ax
    rhs_v = v_com - half * u + DT_S * ay
    u_next = (rhs_u + half * rhs_v) / determinant
    v_com_next = (-half * rhs_u + rhs_v) / determinant
    v_rear_next = v_com_next - COM_X_M * yaw_rate_next
    body_next = torch.stack((u_next, v_rear_next, yaw_rate_next), dim=-1)
    body_norm = (body_next - mean) / scale
    return torch.cat((body_norm, state_norm[:, 3:5] + delta_norm[:, 3:5]),
                     dim=-1)


def _roll_alpha(roll: torch.Tensor, rate: torch.Tensor,
                physical_state: torch.Tensor, command: torch.Tensor,
                lateral_accel: torch.Tensor, roll_model: dict[str, Any]
                ) -> torch.Tensor:
    features = torch.cat((roll[:, None], rate[:, None], physical_state,
                          command, lateral_accel[:, None]), dim=-1)
    mean = torch.as_tensor(roll_model["feature_mean"], dtype=features.dtype,
                           device=features.device)
    scale = torch.as_tensor(roll_model["feature_scale"], dtype=features.dtype,
                            device=features.device)
    coefficient = torch.as_tensor(
        roll_model["physical_coefficients_roll_accel"],
        dtype=features.dtype, device=features.device)
    intercept = torch.as_tensor(
        roll_model["physical_intercept_roll_accel"],
        dtype=features.dtype, device=features.device)
    del mean, scale
    return intercept + features @ coefficient


def _torch_rollout(plant, residual_model, roll_model, arrays, roll_initial,
                   roll_targets, norm_np, config, horizon: int,
                   training: bool) -> dict[str, Any]:
    history, mask, state, pose, commands, target, target_pose = arrays
    device = state.device
    dt = DT_S
    roll = roll_initial[:, 0]
    rate = roll_initial[:, 1]
    history_mean = torch.as_tensor(config.history_mean[:7], device=device)
    history_scale = torch.as_tensor(config.history_scale[:7], device=device)
    state_mean = torch.as_tensor(norm_np["state_mean"], device=device)
    state_scale = torch.as_tensor(norm_np["state_scale"], device=device)
    command_mean = torch.as_tensor(norm_np["command_mean"], device=device)
    command_scale = torch.as_tensor(norm_np["command_scale"], device=device)
    integrator = PoseIntegrator(dt).to(device)
    states, poses, rolls, rates, corrections = [], [], [], [], []
    losses = []
    for step in range(horizon):
        delta, base_acceleration = plant.forward_with_acceleration(
            history, mask, state, commands[:, step])
        physical_state = state * state_scale + state_mean
        physical_command = commands[:, step] * command_scale + command_mean
        correction_features = torch.cat((
            roll[:, None], rate[:, None], physical_state,
            physical_command, base_acceleration), dim=-1)
        correction = residual_model(correction_features)
        acceleration = torch.stack((
            base_acceleration[:, 0],
            base_acceleration[:, 1] - correction,
            base_acceleration[:, 2]), dim=-1)
        next_state = _body_step(state, delta, acceleration, norm_np)
        current_body = physical_state[:, :3]
        next_body = next_state[:, :3] * state_scale[:3] + state_mean[:3]
        next_pose = integrator(pose, 0.5 * (current_body + next_body))

        roll_acceleration = _roll_alpha(
            roll, rate, physical_state, physical_command,
            acceleration[:, 1], roll_model)
        next_rate = rate + dt * roll_acceleration
        next_roll = roll + 0.5 * dt * (rate + next_rate)

        states.append(next_state)
        poses.append(next_pose)
        rolls.append(next_roll)
        rates.append(next_rate)
        corrections.append(correction)
        if training:
            state_loss = F.smooth_l1_loss(next_state, target[:, step])
            position_error = (next_pose[:, :2] - target_pose[:, step, :2]) / 0.5
            heading_error = next_pose[:, 2] - target_pose[:, step, 2]
            heading_error = torch.atan2(torch.sin(heading_error),
                                        torch.cos(heading_error)) / 0.1
            position_loss = F.smooth_l1_loss(
                position_error, torch.zeros_like(position_error))
            heading_loss = F.smooth_l1_loss(
                heading_error, torch.zeros_like(heading_error))
            roll_error = torch.atan2(
                torch.sin(next_roll - roll_targets[:, step, 0]),
                torch.cos(next_roll - roll_targets[:, step, 0])) / 0.03
            rate_error = (next_rate - roll_targets[:, step, 1]) / 0.1
            roll_loss = (F.smooth_l1_loss(roll_error, torch.zeros_like(roll_error))
                         + F.smooth_l1_loss(rate_error, torch.zeros_like(rate_error)))
            correction_loss = correction.square().mean()
            losses.append(state_loss + 0.25 * position_loss
                          + 0.5 * heading_loss + 0.05 * roll_loss
                          + 0.001 * correction_loss)

        if step + 1 < horizon:
            next_command = commands[:, step + 1]
            next_physical_state = next_state * state_scale + state_mean
            next_physical_command = next_command * command_scale + command_mean
            row = (torch.cat((next_physical_state, next_physical_command), dim=-1)
                   - history_mean) / history_scale
            history, mask = advance_context(
                history, mask, row, CONTEXT_STEPS["2.0s"])
        state, pose, roll, rate = next_state, next_pose, next_roll, next_rate

    return {
        "state": torch.stack(states, dim=1),
        "pose": torch.stack(poses, dim=1),
        "roll": torch.stack(rolls, dim=1),
        "roll_rate": torch.stack(rates, dim=1),
        "correction": torch.stack(corrections, dim=1),
        "loss": torch.stack(losses).mean() if losses else state.new_zeros(()),
    }


def _roll_targets(capture, refs, attitude: np.ndarray, horizon: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    starts = [int(capture.bounds[sequence, 0]) + row
              for _, sequence, row in refs]
    initial = attitude[starts][:, [0, 2]].astype(np.float32)
    targets = np.stack([
        attitude[start + 1:start + horizon + 1][:, [0, 2]]
        for start in starts]).astype(np.float32)
    return initial, targets


def _filtered_refs(by_run, capture, valid, horizon, allowed_runs):
    output = {}
    for run_id, refs in by_run.items():
        if run_id not in allowed_runs:
            continue
        kept = []
        for ref in refs:
            _, sequence, row = ref
            start = int(capture.bounds[sequence, 0]) + int(row)
            if valid[start:start + horizon + 1].all():
                kept.append(ref)
        if kept:
            output[run_id] = kept
    return output


def _eval_group(data, capture_index, split, horizon, plant, residual_model,
                roll_model, norm_np, config, attitude, valid, device,
                max_starts=MAX_STARTS_PER_RUN,
                allowed_runs: set[str] | None = None,
                start_refs: dict[str, list] | None = None):
    capture = data.captures[capture_index]
    if start_refs is None:
        by_run = _collect_horizon_windows(
            data.captures, data.raw_sources, data.poses, {split}, horizon,
            capture_indices={capture_index})
        if allowed_runs is not None:
            by_run = {run: refs for run, refs in by_run.items()
                      if run in allowed_runs}
        refs = _select_eval_windows(by_run, max_starts)
    else:
        refs = start_refs
    refs = _filtered_refs(refs, capture, valid, horizon, set(refs))
    candidate_metrics, parent_metrics, roll_metrics = {}, {}, {}
    integrator = PoseIntegrator(DT_S).to(device)
    norm = {
        "state_mean": torch.as_tensor(norm_np["state_mean"], device=device),
        "state_scale": torch.as_tensor(norm_np["state_scale"], device=device),
        "command_mean": torch.as_tensor(norm_np["command_mean"], device=device),
        "command_scale": torch.as_tensor(norm_np["command_scale"], device=device),
        "history_mean": torch.as_tensor(config.history_mean[:7], device=device),
        "history_scale": torch.as_tensor(config.history_scale[:7], device=device),
    }
    for run_id, run_refs in sorted(refs.items()):
        arrays = _batch_arrays(
            data, run_refs, horizon, CONTEXT_STEPS["2.0s"],
            norm_np, config, device)
        roll_initial_np, roll_targets_np = _roll_targets(
            capture, run_refs, attitude, horizon)
        roll_initial = torch.as_tensor(roll_initial_np, device=device)
        roll_targets = torch.as_tensor(roll_targets_np, device=device)
        with torch.no_grad():
            candidate = _torch_rollout(
                plant, residual_model, roll_model, arrays, roll_initial,
                roll_targets, norm_np, config, horizon, training=False)
            parent = _rollout(
                plant, arrays, norm, horizon, CONTEXT_STEPS["2.0s"], integrator)
        candidate_state = (candidate["state"].cpu().numpy()
                           * norm_np["state_scale"] + norm_np["state_mean"])
        target_state = (arrays[5].cpu().numpy() * norm_np["state_scale"]
                        + norm_np["state_mean"])
        target_pose = arrays[6].cpu().numpy()
        candidate_pose = candidate["pose"].cpu().numpy()
        parent_state = (parent["state"].cpu().numpy()
                        * norm_np["state_scale"] + norm_np["state_mean"])
        candidate_metrics[run_id] = _metrics(
            candidate_state, candidate_pose, target_state, target_pose, horizon)
        parent_metrics[run_id] = _metrics(
            parent_state, parent["pose"].cpu().numpy(),
            target_state, target_pose, horizon)
        roll_error = np.arctan2(
            np.sin(candidate["roll"].cpu().numpy() - roll_targets_np[:, :, 0]),
            np.cos(candidate["roll"].cpu().numpy() - roll_targets_np[:, :, 0]))
        rate_error = (candidate["roll_rate"].cpu().numpy()
                      - roll_targets_np[:, :, 1])
        roll_metrics[run_id] = {
            "roll_rmse_rad": float(np.mean(np.sqrt(
                np.mean(roll_error ** 2, axis=1)))),
            "roll_rate_rmse_radps": float(np.mean(np.sqrt(
                np.mean(rate_error ** 2, axis=1)))),
            "mean_absolute_correction_mps2": float(np.mean(
                np.abs(candidate["correction"].cpu().numpy()))),
            "starts": len(run_refs),
        }
    return candidate_metrics, parent_metrics, roll_metrics


def _balanced_run_summary(candidate, parent) -> dict[str, Any]:
    metrics = sorted(set.intersection(*[set(value) for value in candidate.values()]))
    summary = {}
    for index, metric in enumerate(metrics):
        runs = sorted(set(candidate) & set(parent))
        delta = np.asarray([candidate[run][metric] - parent[run][metric]
                            for run in runs], dtype=np.float64)
        rng = np.random.default_rng(TRAIN_SEED + index)
        sample = rng.integers(0, len(delta), size=(5000, len(delta)))
        ci = np.quantile(delta[sample].mean(axis=1), [0.025, 0.975])
        summary[metric] = {
            "parent_macro_run_mean": float(np.mean([
                parent[run][metric] for run in runs])),
            "candidate_macro_run_mean": float(np.mean([
                candidate[run][metric] for run in runs])),
            "candidate_minus_parent_macro_delta": float(delta.mean()),
            "run_cluster_bootstrap_95pct_ci": [float(ci[0]), float(ci[1])],
            "independent_run_count": len(runs),
            "per_run_delta": dict(zip(runs, delta.tolist())),
        }
    return summary


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    torch.set_num_threads(1)
    torch.manual_seed(TRAIN_SEED)
    rng = np.random.default_rng(TRAIN_SEED)
    device = torch.device("cpu")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    saved = torch.load(PARENT_CHECKPOINT, map_location=device, weights_only=True)
    metadata = saved["metadata"]
    plant = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"])).to(device)
    plant.load_state_dict(saved["state_dict"], strict=True)
    plant.eval()
    for parameter in plant.parameters():
        parameter.requires_grad_(False)

    train_rows = _transition_rows(
        data, 0, "train", DEFAULT_DYNAMIC, config,
        allowed_run_ids=set(data.training_runs))
    with np.load(DEFAULT_DYNAMIC, allow_pickle=False) as archive:
        dynamic_attitude = np.asarray(archive["imu_attitude_frames"],
                                      dtype=np.float32)
        dynamic_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
    residual_model = LateralRollResidual().to(device)
    optimizer = torch.optim.AdamW(residual_model.parameters(), lr=5e-4,
                                  weight_decay=1e-4)
    internal_holdout_runs = {
        "openplane_dyn_coupled_train_r03_20261002",
        "practice_filter_none_12lap_20260925",
    }
    if not internal_holdout_runs.issubset(set(data.training_runs)):
        raise RuntimeError("frozen internal whole-run holdout roster changed")
    training_rows = {run: rows for run, rows in train_rows.items()
                     if run not in internal_holdout_runs}
    roll_model = _balanced_fit(training_rows)
    training_refs_by_run = {
        run: refs for run, refs in data.train_windows_by_horizon[DT_HORIZON].items()
        if run in set(data.training_runs) - internal_holdout_runs and refs}
    if len(training_refs_by_run) < 8:
        raise RuntimeError("multi-step roll candidate requires >=8 training runs")
    training_runs = sorted(training_refs_by_run)
    holdout_candidate, holdout_parent, _ = _eval_group(
        data, 0, "train", DT_HORIZON, plant, residual_model,
        roll_model, norm_np, config, dynamic_attitude, dynamic_valid, device,
        allowed_runs=internal_holdout_runs)
    if set(holdout_candidate) != internal_holdout_runs:
        raise RuntimeError("internal whole-run candidate holdout is incomplete")
    best_validation_score = float(np.mean([
        values["position_radial_trajectory_rmse_m"]
        for values in holdout_candidate.values()]))
    best_state = {name: tensor.detach().clone()
                  for name, tensor in residual_model.state_dict().items()}
    best_update = 0
    validation_trace = [{
        "update": 0,
        "candidate_position_radial_rmse_macro_m": best_validation_score,
        "parent_position_radial_rmse_macro_m": float(np.mean([
            values["position_radial_trajectory_rmse_m"]
            for values in holdout_parent.values()])),
    }]
    train_records = []
    started = time.perf_counter()
    for update in range(1, TRAIN_UPDATES + 1):
        selected_runs = rng.choice(training_runs, size=BATCH_RUNS, replace=False)
        refs = []
        for run in selected_runs:
            pool = training_refs_by_run[str(run)]
            eligible = []
            for ref in pool:
                _, sequence, row = ref
                start = int(data.captures[0].bounds[sequence, 0]) + int(row)
                if dynamic_valid[start:start + DT_HORIZON + 1].all():
                    eligible.append(ref)
            if not eligible:
                raise RuntimeError(f"training run {run} lacks valid roll windows")
            refs.append(eligible[int(rng.integers(0, len(eligible)))])
        arrays = _batch_arrays(
            data, refs, DT_HORIZON, CONTEXT_STEPS["2.0s"],
            norm_np, config, device)
        roll_initial_np, roll_targets_np = _roll_targets(
            data.captures[0], refs, dynamic_attitude, DT_HORIZON)
        roll_initial = torch.as_tensor(roll_initial_np, device=device)
        roll_targets = torch.as_tensor(roll_targets_np, device=device)
        optimizer.zero_grad(set_to_none=True)
        prediction = _torch_rollout(
            plant, residual_model, roll_model, arrays, roll_initial,
            roll_targets, norm_np, config, DT_HORIZON, training=True)
        if not torch.isfinite(prediction["loss"]):
            raise FloatingPointError(f"non-finite roll candidate loss at {update}")
        prediction["loss"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            residual_model.parameters(), 1.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite roll candidate gradient at {update}")
        optimizer.step()
        train_records.append({
            "update": update,
            "loss": float(prediction["loss"].detach()),
            "gradient_norm": grad_norm,
            "mean_abs_correction_mps2": float(torch.mean(
                torch.abs(prediction["correction"].detach()))),
        })
        if update % EVAL_EVERY == 0:
            candidate_holdout, parent_holdout, _ = _eval_group(
                data, 0, "train", DT_HORIZON, plant, residual_model,
                roll_model, norm_np, config, dynamic_attitude, dynamic_valid,
                device, allowed_runs=internal_holdout_runs)
            validation_score = float(np.mean([
                values["position_radial_trajectory_rmse_m"]
                for values in candidate_holdout.values()]))
            validation_trace.append({
                "update": update,
                "candidate_position_radial_rmse_macro_m": validation_score,
                "parent_position_radial_rmse_macro_m": float(np.mean([
                    values["position_radial_trajectory_rmse_m"]
                    for values in parent_holdout.values()])),
            })
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                best_update = update
                best_state = {name: tensor.detach().clone()
                              for name, tensor in residual_model.state_dict().items()}

    residual_model.load_state_dict(best_state, strict=True)

    OUTPUT.mkdir(parents=True)
    checkpoint_path = OUTPUT / "checkpoint.pt"
    torch.save({
        "state_dict": {name: tensor.detach().cpu()
                       for name, tensor in best_state.items()},
        "metadata": {
            "model": "WP29 roll-conditioned lateral acceleration residual trained through 5s rollout",
            "parent_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
            "training_run_ids": training_runs,
            "internal_validation_run_ids": sorted(internal_holdout_runs),
            "best_internal_validation_update": best_update,
            "dt_s": DT_S,
            "rollout_horizon_steps": DT_HORIZON,
            "future_sensor_or_truth_inputs": False,
            "production_integration": False,
        },
    }, checkpoint_path)
    train_summary = {
        "parent_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
        "dynamic_source_sha256": sha256_file(DEFAULT_DYNAMIC),
        "training_run_ids": training_runs,
        "internal_validation_run_ids": sorted(internal_holdout_runs),
        "best_internal_validation_update": best_update,
        "best_internal_validation_position_rmse_macro_m":
            best_validation_score,
        "internal_validation_trace": validation_trace,
        "training_updates": TRAIN_UPDATES,
        "batch_run_count": BATCH_RUNS,
        "horizon_steps": DT_HORIZON,
        "horizon_seconds": DT_HORIZON * DT_S,
        "roll_fit_training_runs": roll_model["train_runs"],
        "roll_fit_spectral_radius": roll_model[
            "zero_input_discrete_roll_rate_spectral_radius"],
        "loss_policy": {
            "state_smooth_l1_normalized": 1.0,
            "position_smooth_l1_normalized_by_0_5m": 0.25,
            "heading_smooth_l1_normalized_by_0_1rad": 0.5,
            "roll_and_rate_smooth_l1_normalized_by_0_03rad_0_1radps": 0.05,
            "lateral_correction_l2": 0.001,
        },
        "update_trace": train_records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    TRAIN_REPORT.write_text(json.dumps(train_summary, indent=2) + "\n",
                            encoding="utf-8")

    with np.load(DEFAULT_PRACTICE, allow_pickle=False) as archive:
        practice_attitude = np.asarray(archive["imu_attitude_frames"],
                                       dtype=np.float32)
        practice_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
    evaluations = {}
    for horizon_name, horizon in EVAL_HORIZONS:
        groups = {}
        for capture_index, split, attitude, valid in (
                (0, "validation", dynamic_attitude, dynamic_valid),
                (1, "unseen_practice", practice_attitude, practice_valid)):
            if capture_index == 1 and horizon > 200:
                groups["practice_diagnostic"] = {
                    "omitted": "no 10-second unseen-practice sequences"}
                continue
            candidate, parent, roll_metrics = _eval_group(
                data, capture_index, split, horizon, plant, residual_model,
                roll_model, norm_np, config, attitude, valid, device)
            groups["dynamic_validation" if capture_index == 0
                   else "practice_diagnostic"] = {
                "paired_macro_summary": _balanced_run_summary(candidate, parent),
                "candidate_per_run": candidate,
                "parent_per_run": parent,
                "internal_roll_per_run": roll_metrics,
            }
        evaluations[horizon_name] = groups

    report = {
        "candidate": "roll-conditioned lateral correction optimized through recursive 5s loss",
        "parent_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
        "evaluation_data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "practice": sha256_file(DEFAULT_PRACTICE),
        },
        "future_truth_or_sensor_used_after_initialization": False,
        "future_command_sequence_used": True,
        "candidate_roll_state_source": "training-only fitted dynamics; initialized once from measured roll/rate",
        "candidate_correction_only_changes": "body lateral acceleration",
        "training": train_summary,
        "recursive_whole_run_evaluations": evaluations,
        "final_test_touched": False,
        "production_integration": False,
    }
    EVAL_REPORT.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": EVAL_REPORT.relative_to(ROOT).as_posix(),
        "training": result["training"],
        "horizon_groups": {
            name: {group: value.get("paired_macro_summary")
                   for group, value in groups.items()}
            for name, groups in result[
                "recursive_whole_run_evaluations"].items()},
    }, indent=2))
