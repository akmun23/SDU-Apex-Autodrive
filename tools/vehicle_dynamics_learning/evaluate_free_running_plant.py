#!/usr/bin/env python3
"""Evaluate command-driven plant rollouts on complete, untouched captures.

Only the initial history is initialized from recorded state. Thereafter the
model receives its own predicted body/actuator/wheel state and the recorded
steering/throttle command trace; future measured state is used only for scores.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools.vehicle_dynamics_learning.train_nssm import (
    _load_dataset,
    _model_type,
    _torch,
)


STATE_NAMES = (
    "u_rear_mps", "v_rear_mps", "yaw_rate_rps", "steering_feedback_rad",
    "throttle_feedback_norm", "rear_left_surface_mps",
    "rear_right_surface_mps",
)
SCORE_HORIZONS_S = (0.025, 0.125, 0.250, 0.500, 0.750, 1.0, 2.0, 5.0)
POSE_ORIGIN_OFFSET_M = 0.0  # /odom pose and exported plant twist are rear-axle referenced.


def _pose_rows(bag: Path) -> dict[int, np.ndarray]:
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        if analysis.ODOM not in topics:
            raise ValueError(f"pose truth topic missing from {bag}")
        result = {}
        for receipt_ns, message in analysis._messages(
                connection, topics, analysis.ODOM):
            pose = message.pose.pose
            q = pose.orientation
            yaw = math.atan2(
                2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
                1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
            )
            values = np.asarray((pose.position.x, pose.position.y, yaw),
                                dtype=np.float64)
            if np.isfinite(values).all():
                result[int(receipt_ns)] = values
        return result
    finally:
        connection.close()


def _lap_transitions(bag: Path) -> list[tuple[int, int]]:
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        topic = "/autodrive/roboracer_1/lap_count"
        if topic not in topics:
            return []
        transitions = []
        previous = None
        for receipt_ns, message in analysis._messages(connection, topics, topic):
            count = int(message.data)
            if count != previous:
                transitions.append((int(receipt_ns), count))
                previous = count
        return transitions
    finally:
        connection.close()


def _metric_rows(error: np.ndarray) -> dict[str, Any]:
    return {
        "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "p95_abs": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
        "max_abs": np.max(np.abs(error), axis=0).tolist(),
    }


def _integrate_pose(states: np.ndarray, dt: np.ndarray, initial_pose: np.ndarray,
                    lateral_offset_m: float) -> np.ndarray:
    """Midpoint-integrate rear-axle twist at a fixed body longitudinal offset."""
    pose = np.empty((len(states), 3), dtype=np.float64)
    pose[0] = initial_pose
    for index in range(1, len(states)):
        step = float(dt[index])
        u = 0.5 * (states[index - 1, 0] + states[index, 0])
        r = 0.5 * (states[index - 1, 2] + states[index, 2])
        v_rear = 0.5 * (states[index - 1, 1] + states[index, 1])
        v_origin = v_rear + lateral_offset_m * r
        yaw_mid = pose[index - 1, 2] + 0.5 * r * step
        pose[index, 0] = pose[index - 1, 0] + (
            u * math.cos(yaw_mid) - v_origin * math.sin(yaw_mid)) * step
        pose[index, 1] = pose[index - 1, 1] + (
            u * math.sin(yaw_mid) + v_origin * math.cos(yaw_mid)) * step
        pose[index, 2] = pose[index - 1, 2] + r * step
    return pose


def _load_models(run_dir: Path, dataset: dict[str, Any], device: str):
    torch, nn = _torch()
    if device == "cpu":
        # The plant advances one scalar trajectory step at a time. Large
        # intra-op pools add overhead to each small GRUCell call.
        torch.set_num_threads(1)
    training_report_path = run_dir / "training_report.json"
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    first = torch.load(run_dir / "member_00.pt", map_location="cpu",
                       weights_only=False)
    metadata = first["metadata"]
    if metadata["feature_names"] != dataset["feature_names"]:
        raise ValueError("model and dataset feature layouts do not match")
    if len(dataset["feature_names"]) != 9:
        raise ValueError("whole-run plant evaluator requires the 9-channel command-only feature layout")
    architecture = metadata.get("architecture", "gru")
    factory = _model_type(
        torch, nn, int(metadata["hidden_size"]), architecture,
        int(metadata.get("expert_count", 1)), int(metadata["history_steps"]),
        len(metadata["feature_names"]), first["feature_mean"],
        first["feature_scale"], metadata.get("body_acceleration_mean"),
        metadata.get("body_acceleration_scale"),
        metadata.get("integration_method", "euler"))
    models = []
    members = [row for row in report.get("members", [])
               if row.get("checkpoint")]
    if not members:
        members = [{"checkpoint": path.name}
                   for path in sorted(run_dir.glob("member_*.pt"))]
    for member in members:
        payload = torch.load(run_dir / member["checkpoint"],
                             map_location=device, weights_only=False)
        model = factory().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    if not models:
        raise ValueError(f"no checkpoints in {run_dir}")
    return torch, models, first, metadata, report


def _free_rollout(torch, model, frames: np.ndarray, dt_s: np.ndarray,
                  history_steps: int, mean: np.ndarray,
                  scale: np.ndarray) -> np.ndarray:
    if len(frames) <= history_steps:
        raise ValueError("sequence is shorter than its truth-initialized history")
    normalized = (frames - mean[None, :]) / scale[None, :]
    history = torch.as_tensor(normalized[:history_steps][None, :, :],
                              dtype=torch.float32,
                              device=next(model.parameters()).device)
    hidden = torch.zeros(1, model.cell.hidden_size, dtype=history.dtype,
                         device=history.device)
    with torch.no_grad():
        for index in range(history_steps - 1):
            hidden = model.cell(history[:, index, :], hidden)
        feature = history[:, -1, :]
        states = frames[:, :7].astype(np.float64, copy=True)
        for index in range(history_steps, len(frames)):
            dt = torch.as_tensor([dt_s[index]], dtype=torch.float32,
                                 device=history.device)
            predicted, hidden = model.advance(feature, hidden, dt)
            physical = (predicted[0].cpu().numpy().astype(np.float64)
                        * scale[:7] + mean[:7])
            states[index] = physical
            # The only exogenous channels are the recorded commands. The
            # measured actuator and wheel channels at this future row are not
            # supplied to the recurrent model.
            next_input = torch.as_tensor(
                normalized[index, 7:9][None, :], dtype=torch.float32,
                device=history.device)
            feature = torch.cat((predicted, next_input), dim=1)
    return states


def _sequence_score(states: np.ndarray, truth: np.ndarray, dt: np.ndarray,
                    sample_times_ns: np.ndarray, pose_by_receipt: dict[int, np.ndarray],
                    lap_transitions: list[tuple[int, int]],
                    history_steps: int, com_x_m: float) -> dict[str, Any]:
    pred = states[history_steps:]
    target = truth[history_steps:, :7]
    error = pred - target
    elapsed = (sample_times_ns[history_steps:] -
               sample_times_ns[history_steps - 1]) / 1e9
    horizons: dict[str, Any] = {}
    for horizon in SCORE_HORIZONS_S:
        index = int(np.searchsorted(elapsed, horizon, side="left"))
        if index >= len(elapsed):
            continue
        horizons[f"{horizon:g}s"] = {
            "actual_elapsed_s": float(elapsed[index]),
            "absolute_error": np.abs(error[index]).tolist(),
        }

    pose_coverage = float(np.mean([int(t) in pose_by_receipt
                                   for t in sample_times_ns]))
    pose_result = {"coverage_fraction": pose_coverage}
    if pose_coverage == 1.0:
        true_pose = np.stack([pose_by_receipt[int(t)] for t in sample_times_ns])
        offsets = (0.0, com_x_m)
        calibration = {}
        for offset in offsets:
            integrated_truth = _integrate_pose(
                truth[history_steps - 1:, :3],
                dt[history_steps - 1:], true_pose[history_steps - 1], offset)
            delta = (integrated_truth
                     - true_pose[history_steps - 1:])
            delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
            calibration[str(offset)] = float(np.sqrt(np.mean(delta[:, :2] ** 2)))
        predicted_pose = _integrate_pose(
            states[history_steps - 1:], dt[history_steps - 1:],
            true_pose[history_steps - 1], POSE_ORIGIN_OFFSET_M)
        pose_error = predicted_pose - true_pose[history_steps - 1:]
        pose_error[:, 2] = np.arctan2(np.sin(pose_error[:, 2]),
                                      np.cos(pose_error[:, 2]))
        radial = np.linalg.norm(pose_error[:, :2], axis=1)
        pose_result.update({
            "pose_origin_offset_m": POSE_ORIGIN_OFFSET_M,
            "pose_origin_reference": "rear axle per bridge-frame audit; no per-run offset selected from score labels",
            "ground_truth_twist_integration_xy_rmse_m_by_offset": calibration,
            "unanchored_after_initial_history": True,
            "position_xy_rmse_m": np.sqrt(np.mean(pose_error[:, :2] ** 2,
                                                    axis=0)).tolist(),
            "position_radial_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
            "position_radial_p95_m": float(np.quantile(radial, 0.95)),
            "position_endpoint_m": float(radial[-1]),
            "position_max_m": float(radial.max()),
            "heading_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
            "heading_endpoint_abs_rad": float(abs(pose_error[-1, 2])),
        })
        lap_errors = []
        for receipt_ns, lap_count in lap_transitions:
            if lap_count < 1:
                continue
            sample_index = int(np.searchsorted(sample_times_ns, receipt_ns,
                                               side="left"))
            if not history_steps <= sample_index < len(sample_times_ns):
                continue
            rollout_index = sample_index - history_steps + 1
            local_error = pose_error[rollout_index]
            lap_errors.append({
                "completed_laps": lap_count,
                "sample_alignment_error_ms": float(
                    (sample_times_ns[sample_index] - receipt_ns) / 1e6),
                "position_error_m": float(np.linalg.norm(local_error[:2])),
                "heading_error_rad": float(local_error[2]),
            })
        pose_result["lap_boundary_errors"] = lap_errors
    return {
        "samples_scored": int(len(pred)),
        "free_run_duration_s": float(elapsed[-1]) if len(elapsed) else 0.0,
        "state_metrics": {name: _metric_rows(error[:, index:index + 1])
                          for index, name in enumerate(STATE_NAMES)},
        "horizon_absolute_errors": horizons,
        "pose": pose_result,
    }


def evaluate(dataset_path: Path, run_dir: Path, run_ids: list[str],
             device_name: str, output: Path) -> dict[str, Any]:
    data = _load_dataset(dataset_path)
    device = device_name
    torch, models, payload, metadata, training_report = _load_models(
        run_dir, data, device)
    training_ids = set(metadata.get("training_runs", []))
    overlap = sorted(training_ids.intersection(run_ids))
    if overlap:
        raise ValueError(f"refusing to score training runs: {overlap}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        bounds = archive["sequence_bounds"]
        run_indices = archive["sequence_run_index"]
        run_names = archive["run_ids"].astype(str)
        sample_times = archive["sample_time_ns"].astype(np.int64)
        frames = archive["frames"].astype(np.float32)
        dt_s = archive["dt_s"].astype(np.float32)
    manifest = json.loads((dataset_path.parent / "manifest.json").read_text(
        encoding="utf-8"))
    records = {row["run_id"]: row for row in manifest.get("runs", [])}
    mean = payload["feature_mean"].astype(np.float32)
    scale = payload["feature_scale"].astype(np.float32)
    history_steps = int(metadata["history_steps"])
    run_report = []
    for run_id in run_ids:
        matches = np.flatnonzero(run_names == run_id)
        if len(matches) != 1:
            raise ValueError(f"run ID must occur exactly once in dataset: {run_id}")
        run_index = int(matches[0])
        record = records.get(run_id, {})
        if not record.get("clean_stream_and_collision_gate", False):
            raise ValueError(f"refusing low-quality run {run_id}: {record.get('quality_failures')}")
        bag = Path(record["bag"])
        if not bag.is_absolute():
            bag = Path.cwd() / bag
        pose_by_receipt = _pose_rows(bag)
        lap_transitions = _lap_transitions(bag)
        sequence_reports = []
        run_sequences = [index for index, value in enumerate(run_indices)
                         if int(value) == run_index]
        for sequence_id in run_sequences:
            begin, end = map(int, bounds[sequence_id])
            if end - begin <= history_steps:
                continue
            sequence_frames = frames[begin:end]
            sequence_dt = dt_s[begin:end]
            sequence_times = sample_times[begin:end]
            member_states = [
                _free_rollout(torch, model, sequence_frames, sequence_dt,
                              history_steps, mean, scale)
                for model in models
            ]
            ensemble_state = np.mean(member_states, axis=0)
            truth = sequence_frames.astype(np.float64)
            score = _sequence_score(
                ensemble_state, truth, sequence_dt, sequence_times,
                pose_by_receipt, lap_transitions, history_steps,
                analysis.COM_X_M)
            if len(member_states) > 1:
                spread = np.std(np.stack(member_states, axis=0), axis=0,
                                ddof=0)[history_steps:]
                score["ensemble_mean_member_spread_by_state"] = {
                    name: float(np.mean(spread[:, index]))
                    for index, name in enumerate(STATE_NAMES)
                }
            score["sequence_id"] = sequence_id
            score["sample_count_total"] = int(end - begin)
            sequence_reports.append(score)
        if not sequence_reports:
            raise ValueError(f"no continuous sequence long enough in {run_id}")
        run_report.append({"run_id": run_id, "bag": str(bag),
                           "quality_failures": record.get("quality_failures", []),
                           "sequences": sequence_reports})
    result = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "training_run_dir": str(run_dir.resolve()),
        "architecture": metadata.get("architecture", "gru"),
        "integration_method": metadata.get("integration_method", "euler"),
        "member_count": len(models),
        "initial_truth_history_steps": history_steps,
        "initial_truth_history_s": float((history_steps - 1) * np.median(dt_s)),
        "future_inputs_after_initialization": ["steering_command_rad",
                                               "throttle_command_norm"],
        "future_truth_or_measured_actuator_wheel_feedback_used": False,
        "global_pose_integration": "midpoint integration of predicted rear-axle body twist; one initial pose anchor, no later corrections",
        "runs": run_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        evaluate(args.dataset, args.run_dir, args.run_id, args.device,
                 args.output)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"free-running plant evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
