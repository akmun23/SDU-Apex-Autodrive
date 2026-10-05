#!/usr/bin/env python3
"""Evaluate SUBNET free rollouts on untouched contiguous practice captures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from subnet_body_plant import SubnetBodyPlant


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "live_runs/derived_dynamics_learning_20260928/practice_transfer_validation_20261001_r03/openplane_dynamics.npz"
ACTUATOR_REPORT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004/actuator_model_report.json"
DT_S = 0.025
REAR_AXLE_TO_COM_X_M = 0.15532
# Include one-step and subsecond horizons so the same unseen-lap evaluation
# separates local transition error from error that accumulates recursively.
HORIZONS_S = (0.025, 0.25, 0.5, 1.0, 2.0, 3.75, 5.0, 10.0, 20.0, 30.0)
CHANNELS = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_sequence(data: Any, run_id: str) -> tuple[np.ndarray, ...]:
    run_ids = data["run_ids"].astype(str)
    matches = np.flatnonzero(run_ids == run_id)
    if len(matches) != 1:
        raise ValueError(f"expected one whole-run entry for {run_id}")
    run_index = int(matches[0])
    if str(data["run_splits"][run_index]) != "unseen_practice":
        raise ValueError(f"practice run is not held out: {run_id}")
    seq_run = np.asarray(data["sequence_run_index"], dtype=np.int64)
    seq_indices = np.flatnonzero(seq_run == run_index)
    bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
    if len(seq_indices) < 2:
        raise ValueError(f"multiple contiguous lap segments are required: {run_id}")
    run_bounds = bounds[seq_indices]
    if run_bounds[0, 0] < 0:
        raise ValueError("invalid source frame start")
    if np.any(run_bounds[1:, 0] != run_bounds[:-1, 1]):
        raise ValueError(f"practice segments are not contiguous: {run_id}")
    start, end = int(run_bounds[0, 0]), int(run_bounds[-1, 1])
    if not np.all(np.asarray(data["frame_run_index"])[start:end] == run_index):
        raise ValueError(f"selected interval crosses run boundary: {run_id}")
    resets = np.asarray(data["frame_reset_index"])[start:end]
    if len(np.unique(resets)) != 1:
        raise ValueError(f"practice run contains simulator resets: {run_id}")
    dt = np.asarray(data["dt_s"], dtype=np.float64)[start:end]
    if not np.allclose(dt, DT_S, rtol=0.0, atol=1e-7):
        raise ValueError(f"practice model timebase is not 25 ms: {run_id}")
    packet = np.asarray(data["packet_sequence"], dtype=np.int64)[start:end]
    if len(packet) != end - start or np.any(np.diff(packet) != 1):
        raise ValueError(f"practice packet sequence is discontinuous: {run_id}")
    lap_values = np.unique(np.asarray(data["lap_count"])[start:end])
    if len(lap_values) not in (3, 6):
        raise ValueError(f"expected a three- or six-lap capture, found {len(lap_values)}: {run_id}")
    frames = np.asarray(data["frames"], dtype=np.float32)[start:end]
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)[start:end]
    pose = np.asarray(data["simulator_pose_xyyaw"], dtype=np.float64)[start:end]
    if (not np.isfinite(frames).all() or not np.isfinite(rigid).all()
            or not np.isfinite(pose).all()):
        raise ValueError(f"non-finite practice data: {run_id}")
    inputs = frames[:, 3:5]
    outputs = np.column_stack((
        rigid[:, 7],
        rigid[:, 8] - REAR_AXLE_TO_COM_X_M * rigid[:, 12],
        rigid[:, 12]))
    return inputs, outputs, pose, np.asarray(data["lap_count"])[start:end]


def _integrate(body: np.ndarray, initial_body: np.ndarray,
               initial_pose: np.ndarray) -> np.ndarray:
    pose = initial_pose.astype(np.float64, copy=True)
    previous = initial_body.astype(np.float64, copy=True)
    path = np.empty((len(body), 3), dtype=np.float64)
    for index, state in enumerate(body):
        u, v, yaw_rate = previous
        angle = yaw_rate * DT_S
        if abs(yaw_rate) > 1.0e-7:
            dx = (u * np.sin(angle) + v * (np.cos(angle) - 1.0)) / yaw_rate
            dy = (u * (1.0 - np.cos(angle)) + v * np.sin(angle)) / yaw_rate
        else:
            dx, dy = u * DT_S, v * DT_S
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        pose[0] += cosine * dx - sine * dy
        pose[1] += sine * dx + cosine * dy
        pose[2] = np.arctan2(np.sin(pose[2] + angle),
                             np.cos(pose[2] + angle))
        path[index] = pose
        previous = state
    return path


def _metrics(prediction: np.ndarray, truth: np.ndarray,
             predicted_pose: np.ndarray, truth_pose: np.ndarray) -> dict[str, Any]:
    body_error = prediction - truth
    rmse = np.sqrt(np.mean(body_error ** 2, axis=0))
    pose_error = predicted_pose - truth_pose
    pose_error[:, 2] = np.arctan2(np.sin(pose_error[:, 2]),
                                  np.cos(pose_error[:, 2]))
    position = np.linalg.norm(pose_error[:, :2], axis=1)
    return {
        "samples": len(prediction),
        "body_rmse": dict(zip(CHANNELS, rmse.astype(float).tolist())),
        "body_bias": dict(zip(CHANNELS, body_error.mean(axis=0).astype(float).tolist())),
        "position_trajectory_rmse_m": float(np.sqrt(np.mean(position ** 2))),
        "position_endpoint_error_m": float(position[-1]),
        "position_endpoint_dx_m": float(pose_error[-1, 0]),
        "position_endpoint_dy_m": float(pose_error[-1, 1]),
        "heading_trajectory_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
        "heading_endpoint_error_rad": float(pose_error[-1, 2]),
    }


def _rollout(model: SubnetBodyPlant, inputs: np.ndarray,
             outputs: np.ndarray, device: torch.device) -> np.ndarray:
    model = model.to(device).eval()
    with torch.no_grad():
        history_u = torch.as_tensor(inputs[:model.history_steps],
                                    dtype=torch.float32, device=device)
        history_y = torch.as_tensor(outputs[:model.history_steps],
                                    dtype=torch.float32, device=device)
        latent = model.encode_history(history_u, history_y)
        future = torch.as_tensor(inputs[model.history_steps:],
                                  dtype=torch.float32, device=device)
        prediction = model.rollout(latent, future).cpu().numpy().astype(np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("SUBNET generated a non-finite practice rollout")
    return prediction


def _command_driven_feedback(inputs: np.ndarray, commands: np.ndarray,
                             fit_report: dict[str, Any],
                             history_steps: int) -> tuple[np.ndarray, dict[str, Any]]:
    predicted = inputs.astype(np.float64, copy=True)
    channel_names = ("steering", "throttle")
    metrics = {}
    for channel, name in enumerate(channel_names):
        fit = fit_report["fit"][name]
        alpha = float(fit["alpha"])
        delay = int(fit["selected_delay_steps"])
        if delay not in (0, 1):
            raise ValueError(f"unsupported actuator delay for {name}: {delay}")
        for sample in range(history_steps, len(predicted)):
            command_sample = sample - delay
            if command_sample < 0:
                raise ValueError("actuator command history does not cover delay")
            previous = predicted[sample - 1, channel]
            command = commands[command_sample, channel]
            predicted[sample, channel] = previous + alpha * (command - previous)
        error = predicted[history_steps:, channel] - inputs[history_steps:, channel]
        metrics[name] = {
            "feedback_rmse": float(np.sqrt(np.mean(error ** 2))),
            "feedback_abs_error_p95": float(np.quantile(np.abs(error), 0.95)),
            "alpha": alpha,
            "delay_steps": delay,
        }
    return predicted, metrics


def evaluate(checkpoints: dict[str, Path], source_path: Path,
             output_path: Path, device_name: str,
             run_ids: tuple[str, ...],
             actuator_report_path: Path = ACTUATOR_REPORT) -> dict[str, Any]:
    source_path, output_path = source_path.resolve(), output_path.resolve()
    actuator_report_path = actuator_report_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not actuator_report_path.is_file():
        raise FileNotFoundError(actuator_report_path)
    actuator_report = json.loads(
        actuator_report_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    report: dict[str, Any] = {
        "schema_version": 1,
        "study": "whole contiguous practice free-rollout; one truth-seeded state then recursive body prediction",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "actuator_model_report_sha256": _sha256(actuator_report_path),
        "run_count": len(run_ids),
        "modes": {
            "recorded_feedback_oracle": (
                "uses measured future actuator feedback; diagnostic upper bound only"),
            "command_driven": (
                "uses fitted first-order actuator dynamics driven by future commands; "
                "no future measured feedback"),
        },
        "lap_horizons_s": list(HORIZONS_S),
        "checkpoints": {},
    }
    with np.load(source_path, allow_pickle=False) as data:
        for run_id in run_ids:
            inputs, truth, truth_pose, lap_count = _run_sequence(data, run_id)
            if len(inputs) <= 12:
                raise ValueError(f"practice run too short for SUBNET history: {run_id}")
            sequence_index = np.flatnonzero(
                np.asarray(data["run_ids"]).astype(str) == run_id)
            sequence_run = np.asarray(data["sequence_run_index"], dtype=np.int64)
            selected_sequences = np.flatnonzero(sequence_run == int(sequence_index[0]))
            sample_start = int(np.asarray(data["sequence_bounds"])[
                selected_sequences[0], 0])
            sample_end = sample_start + len(inputs)
            commands = np.asarray(data["frames"], dtype=np.float32)[
                sample_start:sample_end, 7:9]
            command_feedback, actuator_metrics = _command_driven_feedback(
                inputs, commands, actuator_report, 12)
            anchor = 11
            truth_integrated_pose = _integrate(
                truth[anchor + 1:], truth[anchor], truth_pose[anchor])
            truth_pose_floor = _metrics(
                truth[anchor + 1:], truth[anchor + 1:],
                truth_integrated_pose, truth_pose[anchor + 1:])
            run_report: dict[str, Any] = {
                "source_samples": len(inputs),
                "evaluated_samples": len(inputs) - 12,
                "duration_s": (len(inputs) - 12) * DT_S,
                "lap_count_values": np.unique(lap_count).astype(int).tolist(),
                "expected_laps": len(np.unique(lap_count)),
                "command_driven_actuator_prediction": actuator_metrics,
                "truth_body_pose_integration_floor": {
                    key: truth_pose_floor[key] for key in (
                        "position_trajectory_rmse_m", "position_endpoint_error_m",
                        "heading_trajectory_rmse_rad", "heading_endpoint_error_rad")
                },
                "models": {},
            }
            for model_index, (name, path) in enumerate(sorted(checkpoints.items())):
                if not path.is_file():
                    raise FileNotFoundError(path)
                model = SubnetBodyPlant.load_npz(path)
                anchor = model.history_steps - 1
                expected = truth[anchor + 1:]
                initial_pose = truth_pose[anchor]
                expected_pose = truth_pose[anchor + 1:]
                model_modes = {}
                for mode, model_inputs in (
                        ("recorded_feedback_oracle", inputs),
                        ("command_driven", command_feedback)):
                    prediction = _rollout(model, model_inputs, truth, device)
                    predicted_pose = _integrate(
                        prediction, truth[anchor], initial_pose)
                    horizon_metrics = {}
                    for horizon_s in HORIZONS_S:
                        horizon = min(int(round(horizon_s / DT_S)), len(prediction))
                        if horizon < 1:
                            continue
                        metrics = _metrics(
                            prediction[:horizon], expected[:horizon],
                            predicted_pose[:horizon], expected_pose[:horizon])
                        metrics["duration_s"] = horizon * DT_S
                        horizon_metrics[str(horizon_s)] = metrics
                    full_run_metrics = _metrics(
                        prediction, expected, predicted_pose, expected_pose)
                    full_run_metrics["duration_s"] = len(prediction) * DT_S
                    horizon_metrics["full_run"] = full_run_metrics

                    lap_metrics = {}
                    predicted_lap = lap_count[anchor + 1:]
                    for lap in np.unique(predicted_lap):
                        selected = np.flatnonzero(predicted_lap == lap)
                        if len(selected):
                            lap_metrics[str(int(lap))] = _metrics(
                                prediction[selected], expected[selected],
                                predicted_pose[selected], expected_pose[selected])
                    model_modes[mode] = {
                        "horizons": horizon_metrics,
                        "per_lap": lap_metrics,
                    }
                run_report["models"][name] = {
                    "checkpoint": str(path.resolve()),
                    "checkpoint_sha256": _sha256(path),
                    "modes": model_modes,
                }
            report["checkpoints"][run_id] = run_report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--actuator-report", type=Path, default=ACTUATOR_REPORT)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoints = {"candidate": args.candidate}
    if args.baseline:
        checkpoints["baseline"] = args.baseline
    report = evaluate(checkpoints, args.source, args.output, args.device,
                      tuple(args.run_id), args.actuator_report)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "runs": {run: {
            name: {h: {
                "position_rmse_m": value["position_trajectory_rmse_m"],
                "position_endpoint_m": value["position_endpoint_error_m"],
                "speed_rmse_mps": value["body_rmse"]["u_rear_mps"],
                "yaw_rate_rmse_rps": value["body_rmse"]["yaw_rate_rps"],
            } for h, value in model["modes"]["command_driven"]["horizons"].items()}
            for name, model in row["models"].items()}
            for run, row in report["checkpoints"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
