#!/usr/bin/env python3
"""Score SUBNET free rollouts on randomized held-out high-steer captures."""

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
ACTUATOR_REPORT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004/"
    "actuator_model_report.json")
DT_S = 0.025
HISTORY = 12
REAR_AXLE_TO_COM_X_M = 0.15532
HORIZONS = (1, 10, 20, 40, 80, 150, 200)
CHANNELS = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integrate(body: np.ndarray, initial_body: np.ndarray,
               initial_pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(initial_pose, dtype=np.float64).copy()
    previous = np.asarray(initial_body, dtype=np.float64).copy()
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
    body_rmse = np.sqrt(np.mean(np.square(body_error), axis=0))
    pose_error = predicted_pose - truth_pose
    pose_error[:, 2] = np.arctan2(np.sin(pose_error[:, 2]),
                                  np.cos(pose_error[:, 2]))
    position_error = np.linalg.norm(pose_error[:, :2], axis=1)
    return {
        "samples": len(prediction),
        "body_rmse": dict(zip(CHANNELS, body_rmse.astype(float).tolist())),
        "body_bias": dict(zip(CHANNELS, body_error.mean(axis=0).astype(float).tolist())),
        "position_trajectory_rmse_m": float(
            np.sqrt(np.mean(position_error ** 2))),
        "position_endpoint_error_m": float(position_error[-1]),
        "position_endpoint_dx_m": float(pose_error[-1, 0]),
        "position_endpoint_dy_m": float(pose_error[-1, 1]),
        "heading_trajectory_rmse_rad": float(
            np.sqrt(np.mean(pose_error[:, 2] ** 2))),
        "heading_endpoint_error_rad": float(pose_error[-1, 2]),
    }


def _prediction_summary(prediction: np.ndarray, truth: np.ndarray,
                        initial_body: np.ndarray, initial_pose: np.ndarray,
                        truth_pose: np.ndarray) -> dict[str, Any]:
    predicted_pose = _integrate(prediction, initial_body, initial_pose)
    horizons = {}
    for horizon in HORIZONS:
        count = min(horizon, len(prediction))
        horizons[str(horizon)] = {
            **_metrics(prediction[:count], truth[:count],
                       predicted_pose[:count], truth_pose[:count]),
            "duration_s": count * DT_S,
        }
    full = _metrics(prediction, truth, predicted_pose, truth_pose)
    full["duration_s"] = len(prediction) * DT_S
    return {"horizons": horizons, "full_condition": full}


def _regime_rollout(base_model: SubnetBodyPlant,
                    aggressive_model: SubnetBodyPlant,
                    initial_inputs: np.ndarray, initial_outputs: np.ndarray,
                    future_inputs: np.ndarray, steering_threshold_rad: float,
                    steering_exit_threshold_rad: float, dwell_steps: int,
                    device: torch.device, latch_after_entry: bool,
                    speed_threshold_mps: float | None = None
                    ) -> tuple[np.ndarray, float]:
    """Run both input-driven latent models and causally select an output."""
    if (base_model.history_steps != aggressive_model.history_steps
            or base_model.state_order != aggressive_model.state_order):
        raise ValueError("hybrid experts must have matching latent dimensions/history")
    with torch.no_grad():
        base_state = base_model.encode_history(
            torch.as_tensor(initial_inputs, dtype=torch.float32, device=device),
            torch.as_tensor(initial_outputs, dtype=torch.float32, device=device))
        aggressive_state = aggressive_model.encode_history(
            torch.as_tensor(initial_inputs, dtype=torch.float32, device=device),
            torch.as_tensor(initial_outputs, dtype=torch.float32, device=device))
        prediction = []
        regime_active = False
        highsteer_steps = 0
        aggressive_steps = 0
        fused_body = np.asarray(initial_outputs[-1], dtype=np.float64).copy()
        for values in future_inputs:
            actuator = torch.as_tensor(values, dtype=torch.float32, device=device)
            base_output, base_state = base_model.step(base_state, actuator)
            aggressive_output, aggressive_state = aggressive_model.step(
                aggressive_state, actuator)
            steering_abs = abs(float(values[0]))
            if speed_threshold_mps is not None:
                # The gate is causal: speed is the previously selected plant
                # prediction; steering feedback is the current available input.
                regime_active = (
                    float(np.hypot(fused_body[0], fused_body[1]))
                    >= speed_threshold_mps
                    and steering_abs >= steering_threshold_rad)
                eligible = regime_active
            else:
                if not regime_active and steering_abs >= steering_threshold_rad:
                    regime_active = True
                    highsteer_steps = 0
                elif (regime_active and not latch_after_entry
                      and steering_abs < steering_exit_threshold_rad):
                    regime_active = False
                    highsteer_steps = 0
                if regime_active:
                    highsteer_steps += 1
                else:
                    highsteer_steps = 0
                eligible = regime_active and highsteer_steps >= dwell_steps
            selected = aggressive_output if eligible else base_output
            fused_body = selected.detach().cpu().numpy().astype(np.float64)
            aggressive_steps += int(eligible)
            prediction.append(fused_body)
    result = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("regime hybrid generated a non-finite rollout")
    return result, aggressive_steps / max(1, len(future_inputs))


def _quality_gate(source_path: Path, run_id: str) -> tuple[dict[str, Any], Path]:
    manifest_path = source_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("runs", [])
    if len(rows) != 1:
        raise ValueError(f"expected one whole-capture run in {manifest_path}")
    run = rows[0]
    alignment = run.get("packet_sequence_alignment", {})
    if (run.get("run_id") != run_id
            or run.get("effective_split") != "validation"
            or run.get("aborted")
            or run.get("reason") != "schedule complete"
            or not run.get("clean_stream_and_collision_gate")
            or run.get("whole_bag_quality_failures")
            or run.get("quality_failures")
            or int(run.get("timing_faults", -1)) != 0
            or any(int(value) != 0 for value in run.get("collisions", []))
            or float(alignment.get("match_fraction", 0.0)) < 0.999):
        raise ValueError(f"high-steering validation capture failed quality gates: {run_id}")
    if any(float(stats.get("hz", 0.0)) < 38.0
           for stats in run.get("streams", {}).values()):
        raise ValueError(f"capture has a stream below 38 Hz: {run_id}")
    return run, manifest_path


def _command_feedback(inputs: np.ndarray, commands: np.ndarray,
                      actuator_report: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    predicted = inputs.astype(np.float64, copy=True)
    errors: dict[str, float] = {}
    for column, name in enumerate(("steering", "throttle")):
        fit = actuator_report["fit"][name]
        alpha = float(fit["alpha"])
        delay = int(fit["selected_delay_steps"])
        if delay not in (0, 1):
            raise ValueError(f"unsupported {name} actuator delay: {delay}")
        for sample in range(HISTORY, len(predicted)):
            command_index = sample - delay
            predicted[sample, column] = predicted[sample - 1, column] + alpha * (
                commands[command_index, column] - predicted[sample - 1, column])
        errors[name] = float(np.sqrt(np.mean(np.square(
            predicted[HISTORY:, column] - inputs[HISTORY:, column]))))
    return predicted, errors


def evaluate(sources: list[Path], checkpoints: dict[str, Path],
             output: Path, actuator_report_path: Path = ACTUATOR_REPORT,
             device_name: str = "cpu",
             regime_hybrid: tuple[str, str] | None = None,
             steering_threshold_rad: float = 0.30,
             steering_exit_threshold_rad: float = 0.30,
             dwell_s: float = 1.0,
             latch_after_entry: bool = False,
             speed_threshold_mps: float | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    actuator_report = json.loads(actuator_report_path.read_text(encoding="utf-8"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    models = {}
    checkpoint_hashes = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        models[name] = SubnetBodyPlant.load_npz(path).to(device).eval()
        checkpoint_hashes[name] = _sha256(path)
    hybrid_name = None
    dwell_steps = int(round(dwell_s / DT_S))
    if regime_hybrid is not None:
        base_name, aggressive_name = regime_hybrid
        if base_name not in models or aggressive_name not in models:
            raise ValueError("hybrid expert names must match loaded checkpoints")
        if (not np.isfinite(steering_threshold_rad)
                or steering_threshold_rad <= 0.0
                or not np.isfinite(steering_exit_threshold_rad)
                or steering_exit_threshold_rad <= 0.0
                or steering_exit_threshold_rad > steering_threshold_rad
                or not np.isfinite(dwell_s) or dwell_s <= 0.0
                or dwell_steps <= 0
                or not np.isclose(dwell_steps * DT_S, dwell_s, rtol=0.0, atol=1e-9)):
            raise ValueError("hybrid threshold must be positive and dwell a 25 ms multiple")
        if (speed_threshold_mps is not None
                and (not np.isfinite(speed_threshold_mps)
                     or speed_threshold_mps < 0.0)):
            raise ValueError("predicted-speed threshold must be finite and non-negative")
        hybrid_name = (
            f"{base_name}_to_{aggressive_name}_speed{speed_threshold_mps:g}_"
            f"steer{steering_threshold_rad:g}"
            if speed_threshold_mps is not None else
            f"{base_name}_to_{aggressive_name}_dwell_{dwell_steps}steps")

    report: dict[str, Any] = {
        "study": "free recursive SUBNET rollouts on held-out randomized high-steering transients",
        "input_policy": "measured actuator feedback for oracle mode; training-only first-order command-to-feedback model for command-driven mode",
        "sample_period_s": DT_S,
        "horizons_steps": list(HORIZONS),
        "actuator_model_report_sha256": _sha256(actuator_report_path),
        "checkpoint_sha256": checkpoint_hashes,
        "regime_hybrid": ({
            "name": hybrid_name,
            "base_expert": regime_hybrid[0],
            "aggressive_expert": regime_hybrid[1],
            "steering_threshold_rad": steering_threshold_rad,
            "steering_exit_threshold_rad": steering_exit_threshold_rad,
            "predicted_speed_threshold_mps": speed_threshold_mps,
            "continuous_highsteer_dwell_s": (None if speed_threshold_mps is not None
                                               else dwell_s),
            "switch_rule": (
                "select aggressive iff previous selected predicted body speed "
                "meets threshold and current actuator steering feedback meets "
                "threshold" if speed_threshold_mps is not None else
                "enter at the steering threshold and remain active through the prediction segment" if latch_after_entry else
                "enter at the steering threshold, remain active until steering falls below the exit threshold"),
            "latch_after_entry": latch_after_entry,
            "latent_policy": "both experts advance every sample from their own history/input-driven latent state; selection affects only emitted body output",
        } if regime_hybrid is not None else None),
        "captures": {},
    }
    for source_path in sources:
        source_path = source_path.resolve()
        with np.load(source_path, allow_pickle=False) as data:
            if int(data["schema_version"][0]) != 7:
                raise ValueError(f"unsupported capture schema: {source_path}")
            run_ids = data["run_ids"].astype(str)
            if len(run_ids) != 1:
                raise ValueError(f"expected one run per source archive: {source_path}")
            run_id = str(run_ids[0])
            run_quality, manifest_path = _quality_gate(source_path, run_id)
            features = data["feature_names"].astype(str).tolist()
            required = ("steering_feedback_rad", "throttle_feedback_norm",
                        "steering_command_rad", "throttle_command_norm")
            if set(required) - set(features):
                raise ValueError(f"missing actuator features in {source_path}")
            feedback_indices = [features.index(required[0]), features.index(required[1])]
            command_indices = [features.index(required[2]), features.index(required[3])]
            frames = np.asarray(data["frames"], dtype=np.float32)
            rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
            pose = np.asarray(data["simulator_pose_xyyaw"], dtype=np.float64)
            bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
            reset_indices = np.asarray(data["sequence_reset_index"], dtype=np.int64)
            dt = np.asarray(data["dt_s"], dtype=np.float64)
            if (len(bounds) != 20 or not np.allclose(dt, DT_S, rtol=0.0, atol=1e-7)
                    or rigid.shape != (len(frames), 13)
                    or pose.shape != (len(frames), 3)):
                raise ValueError(f"capture data shape/cadence mismatch: {source_path}")
            condition_report_path = source_path.with_name(
                "throttle_following_report.json")
            condition_report = json.loads(
                condition_report_path.read_text(encoding="utf-8"))
            condition_by_reset = {
                int(row["reset_index"]): row
                for row in condition_report.get("conditions", [])
            }
            if any(int(reset) not in condition_by_reset for reset in reset_indices):
                raise ValueError(f"condition labels do not map to reset sequences: {run_id}")
            inputs = frames[:, feedback_indices]
            commands = frames[:, command_indices]
            outputs = np.column_stack((
                rigid[:, 7],
                rigid[:, 8] - REAR_AXLE_TO_COM_X_M * rigid[:, 12],
                rigid[:, 12]))
            if not (np.isfinite(inputs).all() and np.isfinite(commands).all()
                    and np.isfinite(outputs).all() and np.isfinite(pose).all()):
                raise ValueError(f"non-finite state or actuator data: {source_path}")
            capture: dict[str, Any] = {
                "source": str(source_path),
                "source_sha256": _sha256(source_path),
                "manifest_sha256": _sha256(manifest_path),
                "condition_report_sha256": _sha256(condition_report_path),
                "whole_capture_quality": {
                    "timing_faults": int(run_quality["timing_faults"]),
                    "collisions": run_quality["collisions"],
                    "packet_match_fraction": float(
                        run_quality["packet_sequence_alignment"]["match_fraction"]),
                    "stream_hz": {key: float(value["hz"])
                                  for key, value in run_quality["streams"].items()},
                },
                "model_modes": {},
            }
            actuator_errors_by_condition = {"steering": [], "throttle": []}
            predicted_feedback_by_sequence = []
            for left_value, right_value in bounds:
                left, right = int(left_value), int(right_value)
                predicted_feedback, _ = _command_feedback(
                    inputs[left:right], commands[left:right], actuator_report)
                predicted_feedback_by_sequence.append(predicted_feedback)
                for channel, name in enumerate(("steering", "throttle")):
                    actuator_errors_by_condition[name].append(float(np.sqrt(
                        np.mean(np.square(
                            predicted_feedback[HISTORY:, channel]
                            - inputs[left + HISTORY:right, channel])))))
            for model_name, model in models.items():
                mode_rows: dict[str, list[dict[str, Any]]] = {
                    "command_driven": [], "recorded_feedback_oracle": []}
                for condition_index, (left_value, right_value) in enumerate(bounds):
                    left, right = int(left_value), int(right_value)
                    if right - left <= HISTORY + 1:
                        raise ValueError(f"transient sequence too short in {run_id}")
                    local_inputs = inputs[left:right]
                    local_truth = outputs[left:right]
                    local_pose = pose[left:right]
                    with torch.no_grad():
                        latent = model.encode_history(
                            torch.as_tensor(local_inputs[:HISTORY], dtype=torch.float32,
                                            device=device),
                            torch.as_tensor(local_truth[:HISTORY], dtype=torch.float32,
                                            device=device))
                    predicted_feedback = predicted_feedback_by_sequence[
                        condition_index]
                    future_by_mode = {
                        "command_driven": predicted_feedback[HISTORY:],
                        "recorded_feedback_oracle": local_inputs[HISTORY:],
                    }
                    for mode, future_inputs in future_by_mode.items():
                        with torch.no_grad():
                            prediction = model.rollout(
                                latent, torch.as_tensor(future_inputs,
                                    dtype=torch.float32, device=device)
                            ).cpu().numpy().astype(np.float64)
                        if not np.isfinite(prediction).all():
                            raise FloatingPointError(
                                f"non-finite rollout: {model_name}/{run_id}/{mode}")
                        truth = local_truth[HISTORY:]
                        truth_pose = local_pose[HISTORY:]
                        summary = _prediction_summary(
                            prediction, truth, local_truth[HISTORY - 1],
                            local_pose[HISTORY - 1], truth_pose)
                        max_abs_steer = float(np.max(np.abs(
                            local_inputs[HISTORY:, 0])))
                        max_speed = float(np.max(local_truth[HISTORY:, 0]))
                        mode_rows[mode].append({
                            "condition_id": condition_by_reset[
                                int(reset_indices[condition_index])][
                                    "condition_pair_id"],
                            "sequence_label": condition_by_reset[
                                int(reset_indices[condition_index])]["label"],
                            "max_abs_steering_rad": max_abs_steer,
                            "max_speed_mps": max_speed,
                            **summary,
                        })
                capture["model_modes"][model_name] = mode_rows
            if regime_hybrid is not None:
                base_name, aggressive_name = regime_hybrid
                hybrid_rows: dict[str, list[dict[str, Any]]] = {
                    "command_driven": [], "recorded_feedback_oracle": []}
                for condition_index, (left_value, right_value) in enumerate(bounds):
                    left, right = int(left_value), int(right_value)
                    local_inputs = inputs[left:right]
                    local_truth = outputs[left:right]
                    local_pose = pose[left:right]
                    predicted_feedback = predicted_feedback_by_sequence[
                        condition_index]
                    future_by_mode = {
                        "command_driven": predicted_feedback[HISTORY:],
                        "recorded_feedback_oracle": local_inputs[HISTORY:],
                    }
                    for mode, future_inputs in future_by_mode.items():
                        prediction, aggressive_fraction = _regime_rollout(
                            models[base_name], models[aggressive_name],
                            local_inputs[:HISTORY], local_truth[:HISTORY],
                            future_inputs, steering_threshold_rad,
                            steering_exit_threshold_rad, dwell_steps, device,
                            latch_after_entry, speed_threshold_mps)
                        summary = _prediction_summary(
                            prediction, local_truth[HISTORY:],
                            local_truth[HISTORY - 1], local_pose[HISTORY - 1],
                            local_pose[HISTORY:])
                        hybrid_rows[mode].append({
                            "condition_id": condition_by_reset[
                                int(reset_indices[condition_index])][
                                    "condition_pair_id"],
                            "sequence_label": condition_by_reset[
                                int(reset_indices[condition_index])]["label"],
                            "max_abs_steering_rad": float(np.max(
                                np.abs(future_inputs[:, 0]))),
                            "max_speed_mps": float(np.max(
                                local_truth[HISTORY:, 0])),
                            "aggressive_expert_selected_fraction": aggressive_fraction,
                            **summary,
                        })
                capture["model_modes"][hybrid_name] = hybrid_rows
            capture["command_to_feedback_rmse_macro_condition"] = {
                name: float(np.mean(values))
                for name, values in actuator_errors_by_condition.items()
            }
            report["captures"][run_id] = capture
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="named checkpoint as NAME=PATH; repeatable")
    parser.add_argument("--actuator-report", type=Path, default=ACTUATOR_REPORT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--regime-hybrid", nargs=2, metavar=("BASE", "AGGRESSIVE"),
                        help="evaluate an offline switched model from two named checkpoints")
    parser.add_argument("--hybrid-steering-threshold-rad", type=float, default=0.30)
    parser.add_argument("--hybrid-steering-exit-threshold-rad", type=float,
                        default=0.30,
                        help="deactivate aggressive expert below this absolute steering")
    parser.add_argument("--hybrid-dwell-s", type=float, default=1.0,
                        help="continuous high-steering duration before expert switch")
    parser.add_argument("--hybrid-latch-after-entry", action="store_true",
                        help="keep the aggressive expert selected until the prediction segment ends")
    parser.add_argument("--hybrid-speed-threshold-mps", type=float,
                        help="causal gate on the previous selected predicted speed")
    args = parser.parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        if "=" not in value:
            parser.error("--checkpoint must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or name in checkpoints:
            parser.error("checkpoint names must be non-empty and unique")
        checkpoints[name] = Path(path)
    result = evaluate(args.source, checkpoints, args.output,
                      args.actuator_report, args.device,
                      tuple(args.regime_hybrid) if args.regime_hybrid else None,
                      args.hybrid_steering_threshold_rad,
                      args.hybrid_steering_exit_threshold_rad,
                      args.hybrid_dwell_s, args.hybrid_latch_after_entry,
                      args.hybrid_speed_threshold_mps)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "captures": {
            run: {
                model: {
                    mode: {
                        horizon: {
                            "position_rmse_m": float(np.mean([
                                item["horizons"][horizon][
                                    "position_trajectory_rmse_m"]
                                for item in modes]))
                        } for horizon in ("40", "80", "150", "200")
                    } for mode, modes in modes_by_model.items()
                } for model, modes_by_model in row["model_modes"].items()
            } for run, row in result["captures"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
