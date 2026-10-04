#!/usr/bin/env python3
"""Test an internally propagated roll state as a lateral-acceleration input.

Training is restricted to the frozen plant's training runs. The candidate
corrects only lateral acceleration; roll/rate are initialized from the starting
sample and then recursively predicted. Held-out validation/practice evaluation
uses only the future command sequence, not future measured body or IMU data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.vehicle_dynamics_learning.diagnose_rigid_acceleration_sensor_residual_value import (
    _predict_residuals,
    _transition_rows,
)
from tools.vehicle_dynamics_learning.evaluate_internal_roll_state_prediction import (
    CHECKPOINT,
    DT_S,
    _balanced_fit,
    _predict_alpha_batch,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
    integrate_planar_body,
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
    / "roll_conditioned_lateral_acceleration_candidate_units_fixed_20261004.json")
EVAL_HORIZONS = (("2s", 80), ("5s", 200), ("10s", 400))
PRACTICE_MAX_HORIZON = 200
MAX_STARTS_PER_RUN = 16
MAX_TRAIN_ROWS_PER_RUN = 15000
BOOTSTRAP_REPLICATES = 5000
SEED = 20261004
COM_X_M = 0.15532


def _sensor_features(rows: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack((rows["base_features"], rows["roll_features"]))


def _fit_lateral_corrector(train_rows, baseline_predictions):
    features, targets = [], []
    selected_rows: dict[str, list[int]] = {}
    for run_index, run_id in enumerate(sorted(train_rows)):
        count = len(train_rows[run_id]["target"])
        n = min(count, MAX_TRAIN_ROWS_PER_RUN)
        rng = np.random.default_rng(SEED + run_index)
        selected = np.sort(rng.choice(count, n, replace=False))
        selected_rows[run_id] = selected.tolist()
        residual = baseline_predictions[run_id][:, 1] \
            - train_rows[run_id]["target"][:, 1]
        features.append(_sensor_features(train_rows[run_id])[selected])
        targets.append(residual[selected])
    if len(features) < 5:
        raise RuntimeError("lateral corrector needs five independent train runs")
    x, y = np.concatenate(features), np.concatenate(targets)
    model = HistGradientBoostingRegressor(
        max_iter=40, max_leaf_nodes=7, min_samples_leaf=100,
        learning_rate=0.08, l2_regularization=5.0,
        random_state=SEED)
    model.fit(x, y)
    return model, {"selected_rows_per_run": selected_rows,
                   "fit_row_count": int(len(y)),
                   "feature_names": [
                       "current body/actuator/command/difference/speed features",
                       "current roll and roll rate"]}


def _load_model(data, config, device):
    saved = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    metadata = saved["metadata"]
    norm_np = _normalization(data, config)
    model = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"])).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, norm_np, sha256_file(CHECKPOINT)


def _physical_base_features(state: np.ndarray, previous: np.ndarray,
                            command: np.ndarray) -> np.ndarray:
    body, previous_body = state[:, :3], previous[:, :3]
    actuators, previous_actuators = state[:, 3:5], previous[:, 3:5]
    return np.column_stack((
        body, actuators, command,
        (actuators - previous_actuators) / DT_S,
        (body - previous_body) / DT_S,
        np.hypot(body[:, 0], body[:, 1]),
    )).astype(np.float32)


def _body_step_torch(state_norm: torch.Tensor, delta_norm: torch.Tensor,
                     acceleration: torch.Tensor, norm_np: dict[str, np.ndarray]
                     ) -> torch.Tensor:
    current = (state_norm[:, :3].detach().cpu().numpy() * norm_np["state_scale"][:3]
               + norm_np["state_mean"][:3])
    following = integrate_planar_body(
        current, acceleration.detach().cpu().numpy(), DT_S, COM_X_M)
    body_norm = ((following - norm_np["state_mean"][:3])
                 / norm_np["state_scale"][:3])
    body_tensor = torch.as_tensor(body_norm, dtype=torch.float32,
                                  device=state_norm.device)
    actuator_next = state_norm[:, 3:5] + delta_norm[:, 3:5]
    return torch.cat((body_tensor, actuator_next), dim=-1)


def _run_rollout(data, capture_index: int, split: str, attitude: np.ndarray,
                 valid: np.ndarray, refs, horizon: int, plant, corrector,
                 roll_model, norm_np, config, device
                 ) -> tuple[dict[str, dict[str, float]],
                            dict[str, dict[str, float]]]:
    capture = data.captures[capture_index]
    score_candidate, score_parent = {}, {}
    history_mean = np.asarray(config.history_mean[:7], dtype=np.float32)
    history_scale = np.asarray(config.history_scale[:7], dtype=np.float32)
    plant.eval()
    pose_integrator = PoseIntegrator(DT_S).to(device)
    for run_id, all_refs in sorted(refs.items()):
        run_refs = []
        for ref in all_refs:
            _, sequence_index, source_row = ref
            begin = int(capture.bounds[sequence_index, 0]) + int(source_row)
            if valid[begin:begin + horizon + 1].all():
                run_refs.append(ref)
        if not run_refs:
            continue
        arrays = _batch_arrays(
            data, run_refs, horizon, CONTEXT_STEPS["2.0s"],
            norm_np, config, device)
        history, mask, state, pose, commands, target, target_pose = arrays
        initial_rows = [int(capture.bounds[ref[1], 0]) + int(ref[2])
                        for ref in run_refs]
        roll = attitude[initial_rows, 0].astype(np.float64)
        roll_rate = attitude[initial_rows, 2].astype(np.float64)
        previous_states = np.asarray([
            np.concatenate((capture.body[row - 1],
                            capture.frames[row - 1, 3:5]))
            for row in initial_rows], dtype=np.float32)
        predicted_states, predicted_poses = [], []
        with torch.no_grad():
            for step in range(horizon):
                delta, base_acceleration = plant.forward_with_acceleration(
                    history, mask, state, commands[:, step])
                physical_state = (
                    state.cpu().numpy() * norm_np["state_scale"]
                    + norm_np["state_mean"])
                command_physical = (
                    commands[:, step].cpu().numpy()
                    * norm_np["command_scale"] + norm_np["command_mean"])
                base_features = _physical_base_features(
                    physical_state, previous_states, command_physical)
                correction_features = np.column_stack((
                    base_features, roll, roll_rate))
                residual = corrector.predict(correction_features)
                corrected_acceleration = base_acceleration.clone()
                corrected_acceleration[:, 1] -= torch.as_tensor(
                    residual, dtype=torch.float32, device=device)

                current_body = (
                    state[:, :3] * norm_np["state_scale"][:3]
                    + norm_np["state_mean"][:3])
                next_state = _body_step_torch(
                    state, delta, corrected_acceleration, norm_np)
                next_body = (
                    next_state[:, :3] * norm_np["state_scale"][:3]
                    + norm_np["state_mean"][:3])
                next_pose = pose_integrator(
                    pose, 0.5 * (current_body + next_body))

                roll_features = np.column_stack((
                    roll, roll_rate, physical_state[:, :5], command_physical,
                    corrected_acceleration[:, 1].cpu().numpy()))
                roll_acceleration = _predict_alpha_batch(
                    roll_model, roll_features)
                next_roll_rate = roll_rate + DT_S * roll_acceleration
                next_roll = roll + 0.5 * DT_S * (roll_rate + next_roll_rate)

                predicted_states.append(next_state)
                predicted_poses.append(next_pose)
                if step + 1 < horizon:
                    next_command = commands[:, step + 1]
                    next_physical = (
                        next_state.cpu().numpy() * norm_np["state_scale"]
                        + norm_np["state_mean"])
                    next_command_physical = (
                        next_command.cpu().numpy()
                        * norm_np["command_scale"] + norm_np["command_mean"])
                    raw_history = np.column_stack((next_physical,
                                                   next_command_physical))
                    normalized = ((raw_history - history_mean)
                                  / history_scale)
                    row_tensor = torch.as_tensor(
                        normalized, dtype=torch.float32, device=device)
                    history, mask = advance_context(
                        history, mask, row_tensor, CONTEXT_STEPS["2.0s"])
                previous_states = physical_state
                state, pose = next_state, next_pose
                roll, roll_rate = next_roll, next_roll_rate

        candidate_state = torch.stack(predicted_states, dim=1).cpu().numpy()
        candidate_pose = torch.stack(predicted_poses, dim=1).cpu().numpy()
        candidate_state = (candidate_state * norm_np["state_scale"]
                           + norm_np["state_mean"])
        target_state = (target.cpu().numpy() * norm_np["state_scale"]
                        + norm_np["state_mean"])
        target_pose_np = target_pose.cpu().numpy()
        score_candidate[run_id] = _metrics(
            candidate_state, candidate_pose, target_state, target_pose_np,
            horizon)
        with torch.no_grad():
            baseline = _rollout(
                plant, arrays, {
                    "state_mean": torch.as_tensor(
                        norm_np["state_mean"], device=device),
                    "state_scale": torch.as_tensor(
                        norm_np["state_scale"], device=device),
                    "command_mean": torch.as_tensor(
                        norm_np["command_mean"], device=device),
                    "command_scale": torch.as_tensor(
                        norm_np["command_scale"], device=device),
                    "history_mean": torch.as_tensor(
                        config.history_mean[:7], device=device),
                    "history_scale": torch.as_tensor(
                        config.history_scale[:7], device=device),
                }, horizon, CONTEXT_STEPS["2.0s"], pose_integrator)
        baseline_state = (baseline["state"].cpu().numpy()
                          * norm_np["state_scale"] + norm_np["state_mean"])
        score_parent[run_id] = _metrics(
            baseline_state, baseline["pose"].cpu().numpy(),
            target_state, target_pose_np, horizon)
    return score_candidate, score_parent


def _paired_summary(candidate, parent, seed: int) -> dict[str, Any]:
    metrics = sorted(set.intersection(*[
        set(values) for values in candidate.values()]))
    output = {}
    for metric_index, metric in enumerate(metrics):
        run_ids = sorted(set(candidate) & set(parent))
        deltas = np.asarray([candidate[run][metric] - parent[run][metric]
                             for run in run_ids], dtype=np.float64)
        rng = np.random.default_rng(seed + metric_index)
        picks = rng.integers(0, len(deltas),
                             size=(BOOTSTRAP_REPLICATES, len(deltas)))
        ci = np.quantile(deltas[picks].mean(axis=1), [0.025, 0.975])
        output[metric] = {
            "parent_run_macro_mean": float(np.mean([
                parent[run][metric] for run in run_ids])),
            "candidate_run_macro_mean": float(np.mean([
                candidate[run][metric] for run in run_ids])),
            "candidate_minus_parent_run_macro_delta": float(deltas.mean()),
            "run_cluster_bootstrap_95pct_ci": [float(ci[0]), float(ci[1])],
            "independent_runs": len(run_ids),
            "per_run_delta": dict(zip(run_ids, deltas.tolist())),
        }
    return output


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    plant, norm_np, checkpoint_sha = _load_model(data, config, device)
    train_rows = _transition_rows(
        data, 0, "train", DEFAULT_DYNAMIC, config,
        allowed_run_ids=set(data.training_runs))
    validation_rows = _transition_rows(data, 0, "validation",
                                       DEFAULT_DYNAMIC, config)
    practice_rows = _transition_rows(data, 1, "unseen_practice",
                                     DEFAULT_PRACTICE, config)
    train_predictions = _predict_residuals(train_rows, plant, norm_np, device)
    corrector, corrector_meta = _fit_lateral_corrector(
        train_rows, train_predictions)
    roll_model = _balanced_fit(train_rows)

    def attitude_arrays(path):
        with np.load(path, allow_pickle=False) as archive:
            return (np.asarray(archive["imu_attitude_frames"], dtype=np.float32),
                    np.asarray(archive["imu_attitude_valid"], dtype=bool))

    dynamic_attitude, dynamic_valid = attitude_arrays(DEFAULT_DYNAMIC)
    practice_attitude, practice_valid = attitude_arrays(DEFAULT_PRACTICE)
    data_reports = {
        "dynamic_validation": {"path": DEFAULT_DYNAMIC.relative_to(ROOT).as_posix(),
                               "sha256": sha256_file(DEFAULT_DYNAMIC)},
        "practice_diagnostic": {"path": DEFAULT_PRACTICE.relative_to(ROOT).as_posix(),
                                "sha256": sha256_file(DEFAULT_PRACTICE)},
    }
    horizons = {}
    one_step = {}
    for group, rows in (("dynamic_validation", validation_rows),
                        ("practice_diagnostic", practice_rows)):
        sensor_group = {}
        # Use the frozen plant predictions already obtained on each held-out run.
        heldout_predictions = _predict_residuals(rows, plant, norm_np, device)
        for run_id, row in rows.items():
            base_error = heldout_predictions[run_id][:, 1] - row["target"][:, 1]
            residual_estimate = corrector.predict(_sensor_features(row))
            corrected_error = base_error - residual_estimate
            sensor_group[run_id] = {
                "parent_lateral_acceleration_rmse_mps2": float(np.sqrt(
                    np.mean(base_error ** 2))),
                "roll_corrected_lateral_acceleration_rmse_mps2": float(np.sqrt(
                    np.mean(corrected_error ** 2))),
                "transition_count": int(len(base_error)),
            }
        one_step[group] = sensor_group

    capture_sets = ((0, "validation", dynamic_attitude, dynamic_valid,
                     data.validation_windows),
                    (1, "unseen_practice", practice_attitude, practice_valid,
                     data.practice_windows))
    for horizon_name, horizon in EVAL_HORIZONS:
        groups = {}
        for capture_index, split, attitude, valid, _ in capture_sets:
            if capture_index == 1 and horizon > PRACTICE_MAX_HORIZON:
                groups["practice_diagnostic"] = {
                    "omitted": "no 10 s unseen-practice sequence is available"}
                continue
            by_run = _collect_horizon_windows(
                data.captures, data.raw_sources, data.poses, {split}, horizon,
                capture_indices={capture_index})
            refs = _select_eval_windows(by_run, MAX_STARTS_PER_RUN)
            candidate, parent = _run_rollout(
                data, capture_index, split, attitude, valid, refs, horizon,
                plant, corrector, roll_model, norm_np, config, device)
            groups["dynamic_validation" if capture_index == 0
                   else "practice_diagnostic"] = {
                "candidate": candidate,
                "parent": parent,
                "paired": _paired_summary(candidate, parent,
                                           SEED + horizon + capture_index),
            }
        horizons[horizon_name] = groups

    report = {
        "study": "internally propagated roll with lateral-only acceleration correction",
        "frozen_parent_checkpoint_sha256": checkpoint_sha,
        "data": data_reports,
        "training_run_ids": sorted(data.training_runs),
        "training_transition_counts": {
            run: int(len(rows["target"])) for run, rows in train_rows.items()},
        "corrector": {
            "target": "parent predicted minus truth body lateral acceleration",
            "features": corrector_meta["feature_names"],
            "fit_rows": corrector_meta["fit_row_count"],
            "maximum_rows_per_training_run": MAX_TRAIN_ROWS_PER_RUN,
            "algorithm": "HistGradientBoostingRegressor(max_iter=40,max_leaf_nodes=7,min_samples_leaf=100)",
            "one_step_teacher_forced": one_step,
        },
        "roll_model": {
            "training_run_ids": roll_model["train_runs"],
            "training_transition_count": roll_model["train_row_count"],
            "zero_input_discrete_spectral_radius": roll_model[
                "zero_input_discrete_roll_rate_spectral_radius"],
            "inputs_at_inference": [
                "predicted roll and roll rate",
                "predicted current body state and actuator feedback",
                "current command and corrected predicted lateral acceleration"],
            "future_measured_imu_used": False,
        },
        "recursive_body_and_pose_metrics": horizons,
        "future_measured_body_or_roll_used_after_initialization": False,
        "future_commands_used_as_known_open_loop_inputs": True,
        "production_integration": False,
        "final_test_touched": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "horizons": {
            name: {group: values.get("paired")
                   for group, values in groups.items()}
            for name, groups in result["recursive_body_and_pose_metrics"].items()},
    }, indent=2))
