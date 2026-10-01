#!/usr/bin/env python3
"""Compare a hybrid planar-GRU / 3D-attitude teacher on complete bag runs.

The planar GRU predicts rear-axle u/v/yaw-rate plus actuator/wheel states.
The 3D teacher contributes only vertical velocity and body roll/pitch rates.
The hybrid advances rear-axle position and a quaternion explicitly. After the
initial history it uses predicted states and the logged command trace only.
This is an offline architecture diagnostic; it does not modify runtime code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tools.vehicle_dynamics_learning.evaluate_free_running_plant import (
    _free_rollout,
    _integrate_pose,
    _load_models,
)
from tools.vehicle_dynamics_learning.structured_body_models import (
    REAR_AXLE_TO_COM_X_M,
)
from tools.vehicle_dynamics_learning.train_nssm import (
    SIMULATOR_DT_S,
    _load_dataset,
)
from tools.vehicle_dynamics_learning.train_rigid_body_teacher import (
    _state_arrays,
    _torch_model,
)


def _run_metrics(data: dict[str, Any], run_index: int, sequence_index: int,
                 planar_model, planar_payload, planar_metadata,
                 rigid_model, state_all: np.ndarray) -> dict[str, Any] | None:
    start, end = map(int, data["bounds"][sequence_index])
    frames = data["frames"][start:end]
    dt = data["dt_s"][start:end]
    history_steps = int(planar_metadata["history_steps"])
    if len(frames) <= history_steps:
        return None
    rigid = data["simulator_rigid_state"][start:end]
    if not np.isfinite(rigid).all():
        raise ValueError("hybrid evaluation requires complete rigid-state labels")
    state_truth = state_all[start:end]
    truth_position = rigid[:, :3]
    truth_quaternion = rigid[:, 3:7]

    planar = _free_rollout(
        torch, planar_model, frames, dt, history_steps,
        planar_payload["feature_mean"], planar_payload["feature_scale"],
    ).astype(np.float32, copy=False)

    hidden = torch.zeros(1, rigid_model.cell.hidden_size, dtype=torch.float32)
    for index in range(history_steps - 1):
        hidden = rigid_model.update_hidden(
            torch.as_tensor(state_truth[index:index + 1], dtype=torch.float32),
            torch.as_tensor(truth_quaternion[index:index + 1], dtype=torch.float32),
            torch.as_tensor(frames[index, 7:9][None, :], dtype=torch.float32),
            hidden,
        )

    state = torch.as_tensor(state_truth[history_steps - 1:history_steps],
                            dtype=torch.float32)
    quaternion = torch.as_tensor(
        truth_quaternion[history_steps - 1:history_steps], dtype=torch.float32)
    position = torch.as_tensor(
        truth_position[history_steps - 1:history_steps], dtype=torch.float32)
    predicted_positions = [position[0].numpy().copy()]
    predicted_quaternions = [quaternion[0].numpy().copy()]

    with torch.no_grad():
        for index in range(history_steps - 1, len(frames) - 1):
            command = torch.as_tensor(frames[index, 7:9][None, :],
                                      dtype=torch.float32)
            step = torch.as_tensor([float(dt[index + 1])], dtype=torch.float32)
            rigid_next, _, _, hidden, _ = rigid_model.transition(
                state, quaternion, position, command, hidden, step)

            planar_current = torch.as_tensor(planar[index:index + 1, :7],
                                             dtype=torch.float32)
            planar_next = torch.as_tensor(planar[index + 1:index + 2, :7],
                                          dtype=torch.float32)
            # Preserve the empirically stronger planar model. Convert its
            # rear-axle lateral speed to COM speed for rigid-body state, while
            # keeping its predicted yaw rate and actuator/wheel outputs.
            hybrid_next = rigid_next.clone()
            hybrid_next[:, 0] = planar_next[:, 0]
            hybrid_next[:, 1] = (planar_next[:, 1]
                                 + REAR_AXLE_TO_COM_X_M * planar_next[:, 2])
            hybrid_next[:, 5] = planar_next[:, 2]
            hybrid_next[:, 6:10] = planar_next[:, 3:7]

            omega_mid = torch.stack((
                0.5 * (state[:, 3] + hybrid_next[:, 3]),
                0.5 * (state[:, 4] + hybrid_next[:, 4]),
                0.5 * (planar_current[:, 2] + planar_next[:, 2]),
            ), dim=-1)
            q_mid_delta = rigid_model._rotation_vector_quaternion(
                omega_mid * (0.5 * step[:, None]))
            q_mid = rigid_model._quat_multiply(quaternion, q_mid_delta)
            q_delta = rigid_model._rotation_vector_quaternion(
                omega_mid * step[:, None])
            quaternion_next = rigid_model._quat_multiply(quaternion, q_delta)
            quaternion_next = quaternion_next / torch.clamp(
                torch.linalg.vector_norm(quaternion_next, dim=-1, keepdim=True),
                min=1.0e-8,
            )

            velocity_com_mid = 0.5 * (state[:, :3] + hybrid_next[:, :3])
            rear_offset = torch.zeros_like(velocity_com_mid)
            rear_offset[:, 0] = -REAR_AXLE_TO_COM_X_M
            velocity_rear_mid = velocity_com_mid + torch.linalg.cross(
                omega_mid, rear_offset, dim=-1)
            position_next = (position
                             + rigid_model._rotate(q_mid, velocity_rear_mid)
                             * step[:, None])

            state = hybrid_next
            quaternion = quaternion_next
            position = position_next
            predicted_positions.append(position[0].numpy().copy())
            predicted_quaternions.append(quaternion[0].numpy().copy())

    predicted_positions = np.asarray(predicted_positions)
    predicted_quaternions = np.asarray(predicted_quaternions)
    first = history_steps - 1
    target_positions = truth_position[first:]
    target_quaternions = truth_quaternion[first:]
    xy_error = predicted_positions[:, :2] - target_positions[:, :2]
    radial_error = np.linalg.norm(xy_error, axis=1)
    orientation_error = (
        Rotation.from_quat(target_quaternions).inv()
        * Rotation.from_quat(predicted_quaternions)
    ).magnitude()

    packet_pose = data["simulator_pose_xyyaw"][start:end]
    planar_pose = _integrate_pose(
        planar[first:], dt[first:], packet_pose[first], lateral_offset_m=0.0)
    planar_xy_error = planar_pose[:, :2] - packet_pose[first:, :2]
    planar_radial = np.linalg.norm(planar_xy_error, axis=1)
    planar_yaw_error = np.arctan2(
        np.sin(planar_pose[:, 2] - packet_pose[first:, 2]),
        np.cos(planar_pose[:, 2] - packet_pose[first:, 2]),
    )
    lap_count = data["lap_count"][start:end]
    lap_boundary_errors = []
    for sample_index in range(1, len(lap_count)):
        completed_laps = int(lap_count[sample_index])
        if completed_laps <= int(lap_count[sample_index - 1]):
            continue
        rollout_index = sample_index - first
        if not 0 <= rollout_index < len(radial_error):
            continue
        lap_boundary_errors.append({
            "completed_laps": completed_laps,
            "hybrid_xy_error_m": float(radial_error[rollout_index]),
            "hybrid_orientation_error_rad": float(
                orientation_error[rollout_index]),
            "planar_xy_error_m": float(planar_radial[rollout_index]),
            "planar_heading_error_rad": float(
                planar_yaw_error[rollout_index]),
        })

    def pose_summary(radial, heading=None):
        result = {
            "xy_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
            "xy_p95_m": float(np.quantile(radial, 0.95)),
            "xy_endpoint_m": float(radial[-1]),
        }
        if heading is not None:
            result["heading_rmse_rad"] = float(
                np.sqrt(np.mean(heading ** 2)))
        return result

    horizons = {}
    duration = (len(radial_error) - 1) * SIMULATOR_DT_S
    for seconds in (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
        index = int(round(seconds / SIMULATOR_DT_S))
        if index < len(radial_error):
            horizons[f"{seconds:g}s"] = {
                "hybrid_xy_error_m": float(radial_error[index]),
                "hybrid_orientation_error_rad": float(orientation_error[index]),
                "planar_xy_error_m": float(planar_radial[index]),
                "planar_heading_error_rad": float(planar_yaw_error[index]),
            }
    return {
        "run_id": str(data["run_ids"][run_index]),
        "split": str(data["splits"][run_index]),
        "sequence_index": int(sequence_index),
        "forecast_duration_s": duration,
        "samples_scored": int(len(radial_error) - 1),
        "planar_model": pose_summary(planar_radial, planar_yaw_error),
        "hybrid": {
            **pose_summary(radial_error),
            "orientation_rmse_rad": float(
                np.sqrt(np.mean(orientation_error ** 2))),
            "orientation_p95_rad": float(
                np.quantile(orientation_error, 0.95)),
            "orientation_endpoint_rad": float(orientation_error[-1]),
        },
        "lap_boundary_errors": lap_boundary_errors,
        "horizons": horizons,
    }


def evaluate(dataset_path: Path, planar_model_dir: Path,
             rigid_checkpoint: Path, run_ids: list[str]) -> dict[str, Any]:
    data = _load_dataset(dataset_path)
    if data["schema_version"] < 6:
        raise ValueError("hybrid evaluator requires schema-6 3D data")
    if not np.allclose(data["dt_s"], SIMULATOR_DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("hybrid evaluator requires exact 25 ms simulator time")
    state_all, _, _ = _state_arrays(data)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    _, planar_models, planar_payload, planar_metadata, _ = _load_models(
        planar_model_dir, data, "cpu")
    if not planar_models:
        raise ValueError("planar model directory contains no checkpoints")

    checkpoint = torch.load(rigid_checkpoint, map_location="cpu",
                            weights_only=False)
    rigid_metadata = checkpoint["metadata"]
    required_metadata = ("hidden_size", "state_mean", "state_scale",
                         "command_mean", "command_scale", "rate_mean",
                         "rate_scale")
    missing = [key for key in required_metadata if key not in rigid_metadata]
    if missing:
        raise ValueError(f"rigid checkpoint lacks metadata: {missing}")
    rigid_type = _torch_model(
        torch, torch.nn, int(rigid_metadata["hidden_size"]),
        rigid_metadata["state_mean"], rigid_metadata["state_scale"],
        rigid_metadata["command_mean"], rigid_metadata["command_scale"],
        rigid_metadata["rate_mean"], rigid_metadata["rate_scale"],
    )
    rigid_model = rigid_type()
    rigid_model.load_state_dict(checkpoint["state_dict"], strict=True)
    rigid_model.eval()

    run_indices = {str(run_id): index
                   for index, run_id in enumerate(data["run_ids"])}
    missing_runs = sorted(set(run_ids) - set(run_indices))
    if missing_runs:
        raise ValueError(f"run IDs absent from dataset: {missing_runs}")
    reports = []
    for run_id in run_ids:
        run_index = run_indices[run_id]
        for sequence_index, sequence_run in enumerate(data["seq_run"]):
            if int(sequence_run) != run_index:
                continue
            report = _run_metrics(
                data, run_index, sequence_index, planar_models[0],
                planar_payload, planar_metadata, rigid_model, state_all)
            if report is not None:
                reports.append(report)
    return {
        "dataset": str(dataset_path),
        "planar_model_dir": str(planar_model_dir),
        "rigid_checkpoint": str(rigid_checkpoint),
        "future_truth_or_sensor_inputs": False,
        "initial_truth_history_steps": int(planar_metadata["history_steps"]),
        "hybrid_contract": (
            "2D predicted rear-axle u/v/yaw-rate, actuator and wheel states; "
            "3D predicted vertical velocity and body roll/pitch rates; "
            "explicit COM-to-rear-axle velocity offset and quaternion pose"),
        "runs": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--planar-model-dir", type=Path, required=True)
    parser.add_argument("--rigid-checkpoint", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.dataset, args.planar_model_dir,
                      args.rigid_checkpoint, args.run_ids)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
