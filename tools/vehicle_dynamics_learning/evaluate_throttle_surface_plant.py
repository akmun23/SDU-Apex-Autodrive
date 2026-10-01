#!/usr/bin/env python3
"""Free-run plant checkpoints over complete held-out throttle sequences.

Only the first measured history initializes each rollout. Future controls are
the recorded command trace; measured actuator, wheel, body, IMU, and position
signals are used for scoring only. Position scoring is explicitly against the
bridge pose, which the current audit finds to duplicate IPS rather than an
independent external truth source.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.train_nssm import (
    STATE_COUNT,
    _model_type,
    _rollout,
    _torch,
)


HORIZONS_S = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 11.0)
STATE_NAMES = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps",
               "steering_feedback_rad", "throttle_feedback_norm",
               "rear_left_surface_mps", "rear_right_surface_mps")
def _load_export(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    archive = np.load(path, allow_pickle=False)
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {"sequence_bounds", "sequence_run_id", "packet_sequence",
                "body_state", "actuator_feedback", "encoder_surface_mps_100ms",
                "plant_commands", "odom_pose", "ips_position"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"export lacks arrays: {sorted(missing)}")
    expected_count = int(manifest["sequence_count"])
    if len(archive["sequence_bounds"]) != expected_count:
        raise ValueError("sequence bounds do not match the export manifest")
    run_names = archive["sequence_run_id"].astype(str)
    record_ids = [row.get("run_id") for row in manifest["captures"]]
    if not record_ids or not set(run_names).issubset(record_ids):
        raise ValueError("sequence run IDs are inconsistent with manifest")
    arrays = {name: archive[name] for name in archive.files}
    return arrays, manifest


def _checkpoint(torch, nn, checkpoint: Path, feature_names: list[str], device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata.get("feature_names") != feature_names:
        raise ValueError(f"feature layout mismatch in {checkpoint}")
    architecture = metadata.get("architecture", "gru")
    feature_mean = payload["feature_mean"].astype(np.float32)
    feature_scale = payload["feature_scale"].astype(np.float32)
    factory = _model_type(
        torch, nn, int(metadata["hidden_size"]), architecture,
        int(metadata.get("expert_count", 1)), int(metadata["history_steps"]),
        len(feature_names), feature_mean, feature_scale,
        metadata.get("body_acceleration_mean"),
        metadata.get("body_acceleration_scale"),
        metadata.get("integration_method", "euler"),
        metadata.get("rear_axle_to_com_x_m", 0.0))
    model = factory().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload, metadata


def _body_pose_rollout(states: np.ndarray, dts: np.ndarray,
                       initial_pose: np.ndarray) -> np.ndarray:
    pose = np.empty((len(states), 3), dtype=np.float64)
    pose[0] = initial_pose[:3]
    for i in range(1, len(states)):
        dt = float(dts[i])
        u = 0.5 * (states[i - 1, 0] + states[i, 0])
        v = 0.5 * (states[i - 1, 1] + states[i, 1])
        yaw_rate = 0.5 * (states[i - 1, 2] + states[i, 2])
        yaw_mid = pose[i - 1, 2] + 0.5 * yaw_rate * dt
        pose[i, 0] = pose[i - 1, 0] + (
            u * math.cos(yaw_mid) - v * math.sin(yaw_mid)) * dt
        pose[i, 1] = pose[i - 1, 1] + (
            u * math.sin(yaw_mid) + v * math.cos(yaw_mid)) * dt
        pose[i, 2] = pose[i - 1, 2] + yaw_rate * dt
    return pose


def _prepare_sequence(arrays: dict[str, np.ndarray], sequence_id: int,
                      start: int, end: int) -> dict[str, np.ndarray] | None:
    frames = np.column_stack((
        arrays["body_state"][start:end],
        arrays["actuator_feedback"][start:end],
        arrays["encoder_surface_mps_100ms"][start:end],
        arrays["plant_commands"][start:end],
    )).astype(np.float32)
    packet_sequence = arrays["packet_sequence"][start:end, 0]
    pose = arrays["odom_pose"][start:end]
    ips = arrays["ips_position"][start:end]
    valid = np.isfinite(frames).all(axis=1)
    good = np.flatnonzero(valid)
    if not len(good):
        return None
    first = int(good[0])
    frames, packet_sequence, pose, ips = (item[first:] for item in
                                         (frames, packet_sequence, pose, ips))
    valid = np.isfinite(frames).all(axis=1)
    frames, packet_sequence, pose, ips = (item[valid] for item in
                                         (frames, packet_sequence, pose, ips))
    if len(frames) < 220:
        return None
    if np.any(np.diff(packet_sequence) != 1):
        return None
    dts = np.full(len(frames), 0.025, dtype=np.float64)
    return {"frames": frames, "dts": dts,
            "pose": pose, "ips": ips}


def _angle_cluster_bootstrap(values: np.ndarray, angles: np.ndarray,
                             seed: int = 20260930) -> dict[str, Any]:
    """Bootstrap steering-level cluster means, not correlated 40 Hz rows."""
    valid = np.isfinite(values)
    unique = np.unique(angles[valid])
    if len(unique) < 2:
        return {"clusters": int(len(unique)), "rmse": None,
                "rmse_ci95": None}
    cluster_mse = np.asarray([
        np.mean(np.square(values[valid & (angles == angle)]))
        for angle in unique], dtype=np.float64)
    rmse = float(math.sqrt(float(np.mean(cluster_mse))))
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(cluster_mse), size=(3000, len(cluster_mse)))
    boot = np.sqrt(cluster_mse[sample].mean(axis=1))
    return {"clusters": int(len(unique)), "rmse": rmse,
            "rmse_ci95": [float(np.quantile(boot, 0.025)),
                          float(np.quantile(boot, 0.975))]}


def _score_checkpoint(torch, model, payload, metadata, sequences,
                      sequence_metadata, feature_names, batch_size: int,
                      device) -> dict[str, Any]:
    mean = payload["feature_mean"].astype(np.float32)
    scale = payload["feature_scale"].astype(np.float32)
    history_steps = int(metadata["history_steps"])
    min_length = min(len(sequence["frames"]) for sequence in sequences)
    if min_length <= history_steps:
        raise ValueError("held-out sequences do not fit model history")
    scored_rows: list[dict[str, Any]] = []
    horizon_squared: dict[float, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_S}
    horizon_angles: dict[float, list[float]] = {
        horizon: [] for horizon in HORIZONS_S}
    horizon_pose_squared: dict[float, list[float]] = {
        horizon: [] for horizon in HORIZONS_S}
    horizon_pose_angles: dict[float, list[float]] = {
        horizon: [] for horizon in HORIZONS_S}
    all_time_squared: list[np.ndarray] = []
    all_time_angles: list[float] = []
    full_pose_rmse: list[float] = []
    full_pose_angles: list[float] = []
    torch_mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
    torch_scale = torch.as_tensor(scale, dtype=torch.float32, device=device)

    # Truncate only the final few unmatched rows so each batch is rectangular;
    # every condition retains >11 s after its 0.375 s measured initialization.
    for batch_start in range(0, len(sequences), batch_size):
        batch_sequences = sequences[batch_start:batch_start + batch_size]
        batch_rows = sequence_metadata[batch_start:batch_start + batch_size]
        truth_frames = np.stack([s["frames"][:min_length]
                                 for s in batch_sequences])
        dts = np.stack([s["dts"][:min_length] for s in batch_sequences])
        normalized = (truth_frames - mean[None, None, :]) / scale[None, None, :]
        history = torch.as_tensor(normalized[:, :history_steps],
                                  dtype=torch.float32, device=device)
        future = torch.as_tensor(normalized[:, history_steps:],
                                 dtype=torch.float32, device=device)
        future_dt = torch.as_tensor(dts[:, history_steps:],
                                    dtype=torch.float32, device=device)
        with torch.no_grad():
            predicted_normalized = _rollout(
                model, history, future, future_dt, history_steps)
        predicted = (predicted_normalized * torch_scale[:STATE_COUNT]
                     + torch_mean[:STATE_COUNT]).cpu().numpy().astype(np.float64)
        truth = truth_frames[:, history_steps:, :STATE_COUNT].astype(np.float64)
        errors = predicted - truth
        elapsed = np.broadcast_to(
            np.arange(1, min_length - history_steps + 1, dtype=np.float64)[None, :]
            * 0.025,
            (len(batch_sequences), min_length - history_steps)).copy()
        for local_index, (seq, row) in enumerate(zip(batch_sequences, batch_rows)):
            angle = float(row["steering_command_rad"])
            local_error = errors[local_index]
            local_elapsed = elapsed[local_index]
            all_time_squared.append(np.mean(local_error[:, :3] ** 2, axis=0))
            all_time_angles.append(angle)
            for horizon in HORIZONS_S:
                sample_index = int(np.searchsorted(local_elapsed, horizon,
                                                   side="left"))
                if sample_index >= len(local_elapsed):
                    continue
                horizon_squared[horizon].append(
                    local_error[sample_index, :].copy() ** 2)
                horizon_angles[horizon].append(angle)
                anchor = history_steps - 1
                local_states = np.concatenate((
                    truth_frames[local_index, anchor:anchor + 1, :3],
                    predicted[local_index, :sample_index + 1, :3]), axis=0)
                local_dt = np.concatenate((
                    np.asarray([0.0]),
                    dts[local_index, history_steps:history_steps
                        + sample_index + 1]))
                integrated = _body_pose_rollout(
                    local_states, local_dt,
                    batch_sequences[local_index]["pose"][anchor])
                target_pose = batch_sequences[local_index]["pose"][
                    history_steps + sample_index]
                pose_delta = integrated[-1] - target_pose[[0, 1, 3]]
                pose_delta[2] = math.atan2(math.sin(pose_delta[2]),
                                           math.cos(pose_delta[2]))
                horizon_pose_squared[horizon].append(
                    float(np.dot(pose_delta[:2], pose_delta[:2])))
                horizon_pose_angles[horizon].append(angle)
            anchor = history_steps - 1
            local_states = np.concatenate((
                truth_frames[local_index, anchor:anchor + 1, :3],
                predicted[local_index, :, :3]), axis=0)
            local_dt = np.concatenate((np.asarray([0.0]),
                                       dts[local_index, history_steps:min_length]))
            integrated = _body_pose_rollout(
                local_states, local_dt, seq["pose"][anchor])
            target = seq["pose"][history_steps:min_length]
            delta = integrated[1:] - target[:, [0, 1, 3]]
            delta[:, 2] = np.arctan2(np.sin(delta[:, 2]),
                                     np.cos(delta[:, 2]))
            radial = np.linalg.norm(delta[:, :2], axis=1)
            full_pose_rmse.append(float(np.sqrt(np.mean(radial ** 2))))
            full_pose_angles.append(angle)
            scored_rows.append({
                "sequence_index": int(row["sequence_index"]),
                "phase_index": int(row["phase_index"]),
                "steering_command_rad": angle,
                "throttle_start_norm": float(row["throttle_start_norm"]),
                "throttle_end_norm": float(row["throttle_end_norm"]),
                "replicate_index": int(row["replicate_index"]),
                "samples_scored": int(len(local_error)),
                "free_run_duration_s": float(local_elapsed[-1]),
                "full_run_state_rmse": np.sqrt(np.mean(
                    local_error ** 2, axis=0)).tolist(),
                "full_run_body_pose_radial_rmse_m": full_pose_rmse[-1],
            })

    horizon_report: dict[str, Any] = {}
    for horizon in HORIZONS_S:
        if not horizon_squared[horizon]:
            continue
        squared = np.stack(horizon_squared[horizon])
        angles = np.asarray(horizon_angles[horizon])
        channels: dict[str, Any] = {}
        for index, name in enumerate(STATE_NAMES):
            channels[name] = _angle_cluster_bootstrap(
                np.sqrt(squared[:, index]), angles,
                seed=20260930 + int(horizon * 1000) + index)
        pose_sq = np.asarray(horizon_pose_squared[horizon])
        pose_angle = np.asarray(horizon_pose_angles[horizon])
        horizon_report[f"{horizon:g}s"] = {
            "n_condition_replicates": int(len(squared)),
            "body_state_rmse_cluster_bootstrap": channels,
            "position_radial_rmse_cluster_bootstrap": _angle_cluster_bootstrap(
                np.sqrt(pose_sq), pose_angle,
                seed=20261000 + int(horizon * 1000)),
        }
    all_sq = np.stack(all_time_squared)
    all_angles_array = np.asarray(all_time_angles)
    full_pose = np.asarray(full_pose_rmse)
    full_pose_angle_array = np.asarray(full_pose_angles)
    return {
        "initialization": {
            "history_steps": history_steps,
            "median_dt_s": float(np.median([np.median(s["dts"])
                                             for s in sequences])),
            "future_inputs": ["steering_command_rad",
                              "throttle_command_norm"],
            "future_recorded_sensors_or_truth_used_as_inputs": False,
            "position_target_independent_of_odom": False,
        },
        "sequence_count": len(scored_rows),
        "common_scored_duration_s": float(np.median([
            row["free_run_duration_s"] for row in scored_rows])),
        "body_state_full_trajectory_rmse_cluster_bootstrap": {
            name: _angle_cluster_bootstrap(
                np.sqrt(all_sq[:, index]), all_angles_array,
                seed=20262000 + index)
            for index, name in enumerate(STATE_NAMES[:3])},
        "position_full_trajectory_radial_rmse_cluster_bootstrap":
            _angle_cluster_bootstrap(full_pose, full_pose_angle_array,
                                     seed=20263000),
        "horizons": horizon_report,
        "per_sequence": scored_rows,
    }


def evaluate(dataset_path: Path, model_specs: list[tuple[str, Path]],
             output: Path, device_name: str, batch_size: int) -> dict[str, Any]:
    arrays, manifest = _load_export(dataset_path)
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required for rollout scoring: {exc}") from exc
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cpu":
        torch.set_num_threads(1)
    device = torch.device(device_name)
    run_id = "openplane_throttle_5pct_5deg_20260930_r05_resume"
    sequences = []
    sequence_rows = []
    excluded_sequences = []
    for seq_id, (start, end) in enumerate(arrays["sequence_bounds"]):
        if str(arrays["sequence_run_id"][seq_id]) != run_id:
            continue
        packet_steps = np.diff(arrays["packet_sequence"][int(start):int(end), 0])
        if np.any(packet_steps != 1):
            excluded_sequences.append({
                "sequence_index": seq_id,
                "phase_index": manifest["sequences"][seq_id]["phase_index"],
                "reason": "noncontiguous_simulator_packet_sequence",
                "nonunit_packet_steps": int(np.count_nonzero(packet_steps != 1)),
            })
            continue
        seq = _prepare_sequence(arrays, seq_id, int(start), int(end))
        if seq is None:
            excluded_sequences.append({
                "sequence_index": seq_id,
                "phase_index": manifest["sequences"][seq_id]["phase_index"],
                "reason": "incomplete_model_observation_sequence",
            })
            continue
        sequences.append(seq)
        row = manifest["sequences"][seq_id].copy()
        row["sequence_index"] = seq_id
        sequence_rows.append(row)
    if not sequences:
        raise ValueError("no external r05 sequences were selected")
    if len(sequences) + len(excluded_sequences) != 738:
        raise ValueError("r05 sequence accounting does not match its audited 738 phases")
    if any(len(seq["frames"]) < 216 for seq in sequences):
        raise ValueError("a test sequence is too short for the required rollout")
    feature_names = [
        "u_rear_mps", "v_rear_mps", "yaw_rate_rps",
        "steering_feedback_rad", "throttle_feedback_norm",
        "rear_left_surface_mps", "rear_right_surface_mps",
        "steering_command_rad", "throttle_command_norm"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "held_out_capture": run_id,
        "capture_was_used_for_training_or_checkpoint_selection": False,
        "capture_quality": next(row for row in manifest["captures"]
                                 if row["run_id"] == run_id),
        "independence_note": (
            "The r05 bag is a separate repeat capture. IPS is not an "
            "independent position target here: its x/y samples match the "
            "position component of /odom to within recorded quantization."),
        "angle_cluster_uncertainty": (
            "95% intervals resample the 13 steering-command angle clusters; "
            "they do not pretend that 725k serial samples are independent."),
        "candidate_models": {},
        "simulator_timebase": (
            "exactly 0.025 s per consecutive simulator packet; request and "
            "receive clocks are diagnostic only"),
        "excluded_sequences": excluded_sequences,
    }
    models_to_score = []
    for name, path in model_specs:
        if path.is_dir():
            training = json.loads((path / "training_report.json").read_text())
            checkpoint_name = next((member["checkpoint"] for member in
                                    training["members"] if member.get("checkpoint")),
                                   None)
            if checkpoint_name is None:
                raise ValueError(f"no checkpoint in {path}")
            checkpoint = path / checkpoint_name
        else:
            checkpoint = path
        model, payload, metadata = _checkpoint(
            torch, nn, checkpoint, feature_names, device)
        models_to_score.append((name, model, payload, metadata))
    for name, model, payload, metadata in models_to_score:
        print(f"scoring {name}: {metadata.get('architecture')} on "
              f"{len(sequences)} external conditions", flush=True)
        result["candidate_models"][name] = _score_checkpoint(
            torch, model, payload, metadata, sequences, sequence_rows,
            feature_names, batch_size, device)
    result["models"] = {
        name: {"checkpoint": str(path.resolve())}
        for name, path in model_specs}
    result["sequence_count_expected"] = 738
    result["sequence_count_scored"] = len(sequences)
    result["sequence_count_excluded"] = len(excluded_sequences)
    result["bootstrap_group_count"] = 13
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", action="append", required=True,
                        metavar="NAME=CHECKPOINT_OR_DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    specs = []
    for specification in args.model:
        if "=" not in specification:
            parser.error("--model must be NAME=CHECKPOINT_OR_DIR")
        name, path = specification.split("=", 1)
        specs.append((name, Path(path)))
    evaluate(args.dataset, specs, args.output, args.device, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
