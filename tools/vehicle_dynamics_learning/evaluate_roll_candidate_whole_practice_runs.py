#!/usr/bin/env python3
"""Compare recursive plant candidates over complete held-out practice runs.

Each rollout starts from one measured 80-frame history/state/pose. The model
then receives only recorded commands and its own recursively predicted state;
truth and future sensors are read only for scoring. Adjacent archive sequences
are joined only when packet sequence and reset epoch prove continuity.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.diagnose_rigid_acceleration_sensor_residual_value import (
    _transition_rows,
)
from tools.vehicle_dynamics_learning.evaluate_internal_roll_state_prediction import (
    _balanced_fit,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    _normalization,
    _rollout,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _state,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_PRACTICE,
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.train_roll_coupled_multistep_candidate import (
    COM_X_M,
    LateralRollResidual,
    OUTPUT as ROLL_CANDIDATE_DIR,
    _torch_rollout,
)
from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    PoseIntegrator,
)


OUTPUT = ROLL_CANDIDATE_DIR / "whole_practice_run_transfer_20261004.json"
RUNS = (
    "practice_unseen_model_validation_20261001_r02",
    "practice_unseen_model_validation_20261001_r03",
)
DT_S = 0.025
PARENT_CHECKPOINT = (ROLL_CANDIDATE_DIR.parent
    / "rigid_acceleration_history_direct_supervision_5s_v1/checkpoint.pt")
WP27_CHECKPOINT = (ROLL_CANDIDATE_DIR.parent
    / "rigid_acceleration_highsteer_transfer_wp27_v1/checkpoint.pt")
INTERNAL_HOLDOUT_RUNS = {
    "openplane_dyn_coupled_train_r03_20261002",
    "practice_filter_none_12lap_20260925",
}


def _load_plant(path: Path, norm_np: dict[str, np.ndarray],
                device: torch.device) -> tuple[RigidAccelerationHistoryTransition, dict]:
    saved = torch.load(path, map_location=device, weights_only=True)
    metadata = saved["metadata"]
    model = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"]),
    ).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    return model, metadata


def _continuous_run(capture, run_id: str) -> tuple[int, int, int, list[int]]:
    run_index = list(capture.run_ids.astype(str)).index(run_id)
    sequences = np.flatnonzero(capture.sequence_run == run_index)
    if not len(sequences):
        raise ValueError(f"no sequence for {run_id}")
    ordered = sorted((int(capture.bounds[i, 0]), int(capture.bounds[i, 1]), i)
                     for i in sequences)
    if any(left[1] != right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError(f"{run_id} contains a data gap between sequences")
    sequence_ids = [item[2] for item in ordered]
    epochs = {int(capture.sequence_reset[i]) for i in sequence_ids}
    if len(epochs) != 1:
        raise ValueError(f"{run_id} crosses a simulator reset")
    begin, end = ordered[0][0], ordered[-1][1]
    packets = capture.packet[begin:end]
    if len(packets) != end - begin or not np.all(np.diff(packets) == 1):
        raise ValueError(f"{run_id} packet stream is not continuous")
    return run_index, begin, end, sequence_ids


def _arrays(data, capture_index: int, begin: int, end: int,
            norm_np: dict[str, np.ndarray], config, device: torch.device
            ) -> tuple[torch.Tensor, ...]:
    capture = data.captures[capture_index]
    horizon = end - begin - 1
    context = CONTEXT_STEPS["2.0s"]
    if begin < context - 1 or horizon < 1:
        raise ValueError("continuous practice run lacks initialization context")
    history = np.zeros((1, 160, 7), dtype=np.float32)
    history[0, -context:] = (
        capture.input_features[begin - context + 1:begin + 1, :7]
        - config.history_mean[:7]) / config.history_scale[:7]
    mask = np.zeros((1, 160), dtype=np.float32)
    mask[0, -context:] = 1.0
    states = np.asarray([_state(capture, row)
                         for row in range(begin, end)], dtype=np.float32)
    commands = capture.frames[begin:end - 1, 7:9].astype(np.float32)
    target_state = states[1:]
    target_pose = data.poses[capture_index][begin + 1:end]
    state_norm = ((states[:1] - norm_np["state_mean"])
                  / norm_np["state_scale"])
    command_norm = ((commands - norm_np["command_mean"])
                    / norm_np["command_scale"])[None]
    target_norm = ((target_state - norm_np["state_mean"])
                   / norm_np["state_scale"])[None]
    values = (
        history, mask, state_norm,
        data.poses[capture_index][begin:begin + 1].astype(np.float32),
        command_norm, target_norm, target_pose[None].astype(np.float32),
    )
    return tuple(torch.as_tensor(value, dtype=torch.float32, device=device)
                 for value in values)


def _metrics(pred_state: np.ndarray, pred_pose: np.ndarray,
             truth_state: np.ndarray, truth_pose: np.ndarray) -> dict[str, float]:
    position = np.linalg.norm(pred_pose[:, :2] - truth_pose[:, :2], axis=1)
    heading = np.arctan2(np.sin(pred_pose[:, 2] - truth_pose[:, 2]),
                         np.cos(pred_pose[:, 2] - truth_pose[:, 2]))
    error = pred_state - truth_state
    return {
        "position_radial_rmse_m": float(np.sqrt(np.mean(position ** 2))),
        "position_endpoint_m": float(position[-1]),
        "position_p95_m": float(np.quantile(position, 0.95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading ** 2))),
        "heading_endpoint_abs_rad": float(abs(heading[-1])),
        "u_rmse_mps": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "v_rmse_mps": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "yaw_rate_rmse_rps": float(np.sqrt(np.mean(error[:, 2] ** 2))),
    }


def _rollout_models(parent, wp27, candidate, roll_model, arrays,
                    roll_initial, roll_targets, norm_np, config, horizon,
                    device):
    norms = {
        "state_mean": torch.as_tensor(norm_np["state_mean"], device=device),
        "state_scale": torch.as_tensor(norm_np["state_scale"], device=device),
        "command_mean": torch.as_tensor(norm_np["command_mean"], device=device),
        "command_scale": torch.as_tensor(norm_np["command_scale"], device=device),
        "history_mean": torch.as_tensor(config.history_mean[:7], device=device),
        "history_scale": torch.as_tensor(config.history_scale[:7], device=device),
    }
    with torch.no_grad():
        return {
            "frozen_parent": _rollout(
                parent, arrays, norms, horizon, CONTEXT_STEPS["2.0s"],
                PoseIntegrator(DT_S).to(device)),
            "highsteer_augmented_wp27": _rollout(
                wp27, arrays, norms, horizon, CONTEXT_STEPS["2.0s"],
                PoseIntegrator(DT_S).to(device)),
            "roll_coupled_multistep_v2": _torch_rollout(
                parent, candidate, roll_model, arrays, roll_initial,
                roll_targets, norm_np, config, horizon, training=False),
        }


def _physical_metrics(output, arrays, norm_np):
    predicted_state = (output["state"].cpu().numpy()[0]
                       * norm_np["state_scale"] + norm_np["state_mean"])
    predicted_pose = output["pose"].cpu().numpy()[0]
    truth_state = (arrays[5].cpu().numpy()[0] * norm_np["state_scale"]
                   + norm_np["state_mean"])
    truth_pose = arrays[6].cpu().numpy()[0]
    return _metrics(predicted_state, predicted_pose, truth_state, truth_pose)


def _evaluate_one(data, capture_index: int, run_id: str, begin: int, end: int,
                  norm_np, config, attitude, attitude_valid, lap_count,
                  parent, wp27, candidate, roll_model, device
                  ) -> dict[str, Any]:
    capture = data.captures[capture_index]
    initialization = begin + CONTEXT_STEPS["2.0s"] - 1
    horizon = end - initialization - 1
    arrays = _arrays(data, capture_index, initialization, end,
                     norm_np, config, device)
    command_rows = np.arange(initialization, end - 1)
    truth_state = arrays[5].cpu().numpy()[0] * norm_np["state_scale"] \
        + norm_np["state_mean"]
    truth_pose = arrays[6].cpu().numpy()[0]
    roll_initial = torch.as_tensor(
        attitude[initialization:initialization + 1, [0, 2]],
        dtype=torch.float32, device=device)
    roll_targets = torch.as_tensor(
        attitude[initialization + 1:end, [0, 2]][None],
        dtype=torch.float32, device=device)
    outputs = _rollout_models(
        parent, wp27, candidate, roll_model, arrays, roll_initial,
        roll_targets, norm_np, config, horizon, device)

    model_outputs = {}
    for name, output in outputs.items():
        state = output["state"].cpu().numpy()[0] * norm_np["state_scale"] \
            + norm_np["state_mean"]
        pose = output["pose"].cpu().numpy()[0]
        model_outputs[name] = {
            "full_continuous_run_metrics": _metrics(
                state, pose, truth_state, truth_pose),
            "per_lap": {},
        }
        growth = {}
        for steps in (1, 2, 4, 8, 20, 40, 80, 160, 200, 400):
            if steps > len(state):
                continue
            delta_state = state[steps - 1] - truth_state[steps - 1]
            delta_pose = pose[steps - 1] - truth_pose[steps - 1]
            delta_pose[2] = np.arctan2(np.sin(delta_pose[2]),
                                       np.cos(delta_pose[2]))
            growth[f"{steps}_steps_{steps * DT_S:g}s"] = {
                "position_error_at_horizon_m": float(
                    np.linalg.norm(delta_pose[:2])),
                "heading_abs_error_at_horizon_rad": float(abs(delta_pose[2])),
                "u_abs_error_at_horizon_mps": float(abs(delta_state[0])),
                "v_abs_error_at_horizon_mps": float(abs(delta_state[1])),
                "yaw_rate_abs_error_at_horizon_rps": float(abs(delta_state[2])),
                "position_rmse_from_initialization_m": _metrics(
                    state[:steps], pose[:steps], truth_state[:steps],
                    truth_pose[:steps])["position_radial_rmse_m"],
            }
        model_outputs[name]["error_growth_by_horizon"] = growth
        run_lap_labels = lap_count[command_rows + 1]
        for lap in sorted(int(item) for item in np.unique(run_lap_labels)):
            select = np.flatnonzero(run_lap_labels == lap)
            if len(select) < 20:
                continue
            index = select
            model_outputs[name]["per_lap"][str(lap)] = {
                "samples": int(len(index)),
                "duration_s": float(len(index) * DT_S),
                **_metrics(state[index], pose[index],
                           truth_state[index], truth_pose[index]),
            }
        model_outputs[name]["prediction_steps"] = int(horizon)
        model_outputs[name]["initialization_frame"] = int(initialization)

    transitions = (np.flatnonzero(
        np.diff(lap_count[begin:end]) == 1) + begin + 1)
    isolated = {name: [] for name in model_outputs}
    for lap_start, next_crossing in zip(transitions, transitions[1:]):
        lap_start, next_crossing = int(lap_start), int(next_crossing)
        if lap_start - begin < CONTEXT_STEPS["2.0s"] - 1:
            continue
        lap_end = next_crossing + 1
        lap_arrays = _arrays(
            data, capture_index, lap_start, lap_end, norm_np, config, device)
        lap_horizon = lap_end - lap_start - 1
        lap_roll_initial = torch.as_tensor(
            attitude[lap_start:lap_start + 1, [0, 2]],
            dtype=torch.float32, device=device)
        lap_roll_targets = torch.as_tensor(
            attitude[lap_start + 1:lap_end, [0, 2]][None],
            dtype=torch.float32, device=device)
        lap_outputs = _rollout_models(
            parent, wp27, candidate, roll_model, lap_arrays,
            lap_roll_initial, lap_roll_targets, norm_np, config,
            lap_horizon, device)
        for name, output in lap_outputs.items():
            isolated[name].append({
                "lap_counter_at_initialization": int(lap_count[lap_start]),
                "initialization_frame": lap_start,
                "samples": int(lap_horizon),
                "duration_s": float(lap_horizon * DT_S),
                **_physical_metrics(output, lap_arrays, norm_np),
            })

    for name, episodes in isolated.items():
        model_outputs[name]["isolated_lap_replays"] = episodes
        metric_names = ("position_radial_rmse_m", "position_endpoint_m",
                        "heading_rmse_rad", "u_rmse_mps", "v_rmse_mps",
                        "yaw_rate_rmse_rps")
        model_outputs[name]["isolated_lap_macro_metrics"] = {
            metric: float(np.mean([episode[metric] for episode in episodes]))
            for metric in metric_names
        } if episodes else {}
    return {
        "run_id": run_id,
        "run_duration_s": float((end - begin - 1) * DT_S),
        "initialization_contract": (
            "80 measured history rows, current measured body/actuator state, and "
            "current simulator pose; afterward commands only and recursive state"),
        "future_truth_or_sensor_feedback_used": False,
        "packet_continuity_verified": True,
        "reset_epoch_count": 1,
        "attitude_valid_fraction_after_initialization": float(
            attitude_valid[initialization + 1:end].mean()),
        "models": model_outputs,
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    parent, parent_meta = _load_plant(PARENT_CHECKPOINT, norm_np, device)
    wp27, wp27_meta = _load_plant(WP27_CHECKPOINT, norm_np, device)
    candidate_saved = torch.load(
        ROLL_CANDIDATE_DIR / "checkpoint.pt", map_location=device,
        weights_only=True)
    candidate = LateralRollResidual().to(device)
    candidate.load_state_dict(candidate_saved["state_dict"], strict=True)
    candidate.eval()
    train_rows = _transition_rows(
        data, 0, "train", DEFAULT_DYNAMIC, config,
        allowed_run_ids=set(data.training_runs) - INTERNAL_HOLDOUT_RUNS)
    roll_model = _balanced_fit(train_rows)

    capture_index = 1
    capture = data.captures[capture_index]
    with np.load(DEFAULT_PRACTICE, allow_pickle=False) as archive:
        if not np.array_equal(archive["packet_sequence"], capture.packet):
            raise ValueError("practice labels are not aligned with plant capture")
        attitude = np.asarray(archive["imu_attitude_frames"], dtype=np.float32)
        attitude_valid = np.asarray(archive["imu_attitude_valid"], dtype=bool)
        run_labels = archive["lap_count"]
    capture.lap_count = np.asarray(run_labels, dtype=np.int32)
    reports = {}
    for run_id in RUNS:
        run_index, begin, end, sequence_ids = _continuous_run(capture, run_id)
        if begin < 79:
            raise ValueError("held-out run lacks the required measured context")
        if not attitude_valid[begin + 79:end].all():
            raise ValueError(f"attitude labels invalid for {run_id}")
        reports[run_id] = _evaluate_one(
            data, capture_index, run_id, begin, end, norm_np, config,
            attitude, attitude_valid, run_labels, parent, wp27, candidate,
            roll_model, device)
        reports[run_id]["sequence_count_joined"] = len(sequence_ids)
        reports[run_id]["lap_count_transitions"] = int(np.count_nonzero(
            np.diff(run_labels[begin:end]) == 1))

    report = {
        "study": "recursive full-capture practice replay, including per-lap error accumulation",
        "dataset": str(DEFAULT_PRACTICE.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DEFAULT_PRACTICE),
        "run_ids": list(RUNS),
        "candidate_checkpoint_sha256": sha256_file(
            ROLL_CANDIDATE_DIR / "checkpoint.pt"),
        "parent_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
        "highsteer_augmented_checkpoint_sha256": sha256_file(WP27_CHECKPOINT),
        "candidate_training_runs": candidate_saved["metadata"]["training_run_ids"],
        "candidate_checkpoint_selection_runs": sorted(INTERNAL_HOLDOUT_RUNS),
        "note_on_holdout": (
            "These two practice runs were not used to fit or select roll-coupled "
            "candidate v2; they were previously used for its short-horizon "
            "diagnostic report, so this is a longer follow-up, not a pristine "
            "unseen test."),
        "wp27_training_runs": wp27_meta["training_run_ids"],
        "future_truth_or_sensor_feedback_used": False,
        "commands_are_recorded_inputs": True,
        "production_integration": False,
        "results": reports,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    compact = {}
    for run_id, run_result in result["results"].items():
        compact[run_id] = {
            "run_duration_s": run_result["run_duration_s"],
            "lap_count_transitions": run_result["lap_count_transitions"],
        "models": {
                name: {
                    "full_run": detail["full_continuous_run_metrics"],
                    "isolated_lap_macro": detail["isolated_lap_macro_metrics"],
                    "lap_position_rmse_m": {
                        lap: metrics["position_radial_rmse_m"]
                        for lap, metrics in detail["per_lap"].items()},
                }
                for name, detail in run_result["models"].items()},
        }
    print(json.dumps({"report": str(OUTPUT), "results": compact}, indent=2))
