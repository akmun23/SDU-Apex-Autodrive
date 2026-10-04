#!/usr/bin/env python3
"""Train and validate a shared-history recurrent vehicle plant.

Research-only: the model advances at the captured 40 Hz cadence, predicts its
own body/wheel/actuator/roll state after initialization, and consumes commands
only during free rollout. Dynamic validation selects checkpoints; test and
final-test splits remain closed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.consistent_history_teacher import (
    actuator_metadata,
    make_model,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    HISTORY_STEPS,
    append_roll_state,
    fit_actuator_dynamics,
    fit_roll_oscillator,
    generalized_acceleration_targets,
    integrate_pose,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.family_condition_sampler import (
    FamilyConditionSampler,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    DEFAULT_DATASET,
    _fixed_validation_windows,
    _load_model,
    _robust_normalizers,
    evaluate_model,
)
from tools.vehicle_dynamics_learning.train_effective_race_teacher import (
    _build_window_sampler,
    _selection_score,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
             / "full_modeling_reset_20261001"
             / "replacement_offline_sim_raceline_20261003"
             / "full_throttle_domain_v1/next_phase_after_2129427")
OUTPUT_DIR = TASK_ROOT / "consistent_history_gru_v1_20261004"
PARENT_CHECKPOINT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002"
    / "encoder_raw_state_teacher_v1"
    / "edssm_gru_z32_e2_rollresidual_10s_yawonly_longitudinal_seed101/last.pt")
PARENT_SHA256 = "8c3b50980e3f7d5d0fa1b946542672be25e032bf84f37d6ad13e5d8325cc2636"
SEED = 20261004
MAX_THROTTLE = 0.50
STAGES = ((30, 120, 12), (80, 120, 8), (200, 120, 4), (400, 80, 2))
EVAL_INTERVAL = 40
EVAL_HORIZONS = {"0.75s": 30, "2s": 80, "5s": 200}


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for offline plant training") from exc
    return torch, nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _state_and_targets(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    physical = physical_state_from_dataset(data, "filtered_odometry")
    state = append_roll_state(data, physical).astype(np.float32)
    acceleration = generalized_acceleration_targets(
        physical, data["bounds"], data["dt_s"])
    if state.shape != (len(data["frames"]), 9):
        raise ValueError("unexpected nine-state physical/roll layout")
    return state, acceleration


def _sample_batch(data: dict[str, Any], state: np.ndarray,
                  acceleration: np.ndarray,
                  sampler: FamilyConditionSampler, batch_size: int,
                  horizon: int, rng: np.random.Generator
                  ) -> tuple[np.ndarray, ...]:
    frames = np.asarray(data["frames"], dtype=np.float32)
    histories, initial, delayed, commands, targets, accel, poses = (
        [], [], [], [], [], [], [])
    for _ in range(batch_size):
        _, (sequence_id, start), _, _ = sampler.sample(rng)
        sequence_start = int(data["bounds"][sequence_id, 0])
        if start - HISTORY_STEPS + 1 < sequence_start or start < sequence_start + 1:
            raise ValueError("sampler selected a history crossing a reset")
        hist_indices = np.arange(start - HISTORY_STEPS + 1, start + 1)
        target_indices = np.arange(start + 1, start + horizon + 1)
        histories.append(np.column_stack((
            state[hist_indices], frames[hist_indices, 7:9])))
        initial.append(state[start])
        delayed.append(frames[start - 1, 7:9])
        # The command at source row k drives the fixed 25 ms transition k->k+1.
        commands.append(frames[start:start + horizon, 7:9])
        targets.append(state[target_indices])
        local_accel = acceleration[start:start + horizon]
        if local_accel.shape != (horizon, 5) or not np.isfinite(local_accel).all():
            raise ValueError("sampled acceleration target crosses a sequence end")
        accel.append(local_accel)
        pose_rows = np.concatenate(([start], target_indices))
        poses.append(data["simulator_pose_xyyaw"][pose_rows])
    return tuple(np.asarray(items, dtype=np.float32) for items in (
        histories, initial, delayed, commands, targets, accel, poses))


def _tensor(torch, value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _loss(torch, nn, model, batch, normalizers, device):
    histories, initial, delayed, commands, targets, accel_targets, poses = batch
    histories = _tensor(torch, histories, device)
    initial = _tensor(torch, initial, device)
    delayed = _tensor(torch, delayed, device)
    commands = _tensor(torch, commands, device)
    targets = _tensor(torch, targets, device)
    accel_targets = _tensor(torch, accel_targets, device)
    poses = _tensor(torch, poses, device)

    predicted, predicted_accel, _, _ = model.rollout(
        initial, delayed, histories, commands)
    state_scale = _tensor(torch, normalizers["state_scale"], device)
    accel_scale = _tensor(torch, normalizers["acceleration_bounds"], device)
    state_error = (predicted - targets) / state_scale
    body_loss = nn.functional.smooth_l1_loss(
        state_error[:, :, :3], torch.zeros_like(state_error[:, :, :3]))
    actuator_loss = nn.functional.smooth_l1_loss(
        state_error[:, :, 3:5], torch.zeros_like(state_error[:, :, 3:5]))
    wheel_loss = nn.functional.smooth_l1_loss(
        state_error[:, :, 5:7], torch.zeros_like(state_error[:, :, 5:7]))
    roll_loss = nn.functional.smooth_l1_loss(
        state_error[:, :, 7:9], torch.zeros_like(state_error[:, :, 7:9]))
    acceleration_error = (predicted_accel - accel_targets) / accel_scale
    body_accel_loss = nn.functional.smooth_l1_loss(
        acceleration_error[:, :, :3],
        torch.zeros_like(acceleration_error[:, :, :3]))
    wheel_accel_loss = nn.functional.smooth_l1_loss(
        acceleration_error[:, :, 3:5],
        torch.zeros_like(acceleration_error[:, :, 3:5]))

    predicted_pose = integrate_pose(torch, predicted, poses[:, 0], initial)
    position_error = (predicted_pose[:, :, :2] - poses[:, 1:, :2]) / 0.20
    heading_error = torch.atan2(
        torch.sin(predicted_pose[:, :, 2] - poses[:, 1:, 2]),
        torch.cos(predicted_pose[:, :, 2] - poses[:, 1:, 2])) / 0.10
    pose_loss = (nn.functional.smooth_l1_loss(
        position_error, torch.zeros_like(position_error))
        + nn.functional.smooth_l1_loss(
            heading_error, torch.zeros_like(heading_error)))
    total = (body_loss + 0.15 * actuator_loss + 0.60 * wheel_loss
             + 0.20 * roll_loss + 0.20 * body_accel_loss
             + 0.10 * wheel_accel_loss + 0.40 * pose_loss)
    components = {
        "body_state": body_loss,
        "actuator_state": actuator_loss,
        "wheel_state": wheel_loss,
        "roll_state": roll_loss,
        "body_acceleration": body_accel_loss,
        "wheel_acceleration": wheel_accel_loss,
        "pose": pose_loss,
    }
    return total, components


def _evaluate(torch, model, data, state, normalizers, windows, device,
              *, command_offset=-1):
    return evaluate_model(
        model, data, state, normalizers["state_scale"], str(device),
        windows=windows, batch_size=8, wheel_state_source="filtered_odometry",
        horizon_steps=EVAL_HORIZONS, command_offset_frames=command_offset)


def _training_runs(data: dict[str, Any]) -> list[str]:
    return sorted(np.asarray(data["run_ids"]).astype(str)[
        np.asarray(data["splits"]).astype(str) == "train"].tolist())


def train(output_dir: Path, device_name: str = "cpu") -> dict[str, Any]:
    torch, nn = _torch()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment: {output_dir}")
    if _sha256(PARENT_CHECKPOINT) != PARENT_SHA256:
        raise ValueError("frozen parent checksum changed; refuse mismatched comparison")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(2)
    device = torch.device(device_name)

    data = _load_dataset(DEFAULT_DATASET)
    if int(data["schema_version"]) != 9:
        raise ValueError("shared-history plant expects frozen schema 9")
    source_hash = _sha256(DEFAULT_DATASET)
    state, acceleration = _state_and_targets(data)
    actuator_fit = fit_actuator_dynamics(data)
    roll_coefficients, roll_report = fit_roll_oscillator(
        data, physical_state_from_dataset(data, "filtered_odometry"))
    normalizers = _robust_normalizers(
        data, state, acceleration, history_state=state,
        max_throttle_command=MAX_THROTTLE)
    # Keep bounds strictly training-only and add no synthetic intermediate
    # labels: the five output channels are derivatives of captured 25 ms states.
    run_index = np.full(len(data["frames"]), -1, dtype=np.int32)
    for (start, end), run in zip(data["bounds"], data["seq_run"]):
        run_index[int(start):int(end)] = int(run)
    train_mask = (np.asarray(data["splits"]).astype(str)[run_index] == "train")
    support = (train_mask & (np.hypot(state[:, 0], state[:, 1]) <= 12.0)
               & (np.asarray(data["frames"][:, 8]) <= MAX_THROTTLE)
               & np.isfinite(acceleration).all(axis=1))
    accel_bounds = np.quantile(np.abs(acceleration[support]), 0.999, axis=0) * 1.15
    accel_bounds = np.maximum(accel_bounds, np.asarray((1.0, 1.0, 0.5, 1.0, 1.0)))
    normalizers["acceleration_bounds"] = accel_bounds.astype(np.float32)

    Model = make_model(
        torch, nn,
        history_mean=normalizers["history_mean"],
        history_scale=normalizers["history_scale"],
        state_mean=normalizers["state_mean"],
        state_scale=normalizers["state_scale"],
        command_mean=normalizers["command_mean"],
        command_scale=normalizers["command_scale"],
        acceleration_bounds=normalizers["acceleration_bounds"],
        roll_oscillator_coefficients=roll_coefficients,
        steering_delay_steps=actuator_fit.steering.delay_steps,
        steering_alpha=actuator_fit.steering.alpha,
        throttle_delay_steps=actuator_fit.throttle.delay_steps,
        throttle_alpha=actuator_fit.throttle.alpha,
    )
    model = Model().to(device)

    # All checkpoints are selected with the same six whole-run validation
    # captures and frozen starts; practice and test splits are not consulted.
    validation_windows = _fixed_validation_windows(
        data, state, max_windows_per_run=8, horizon_steps=200,
        max_throttle_command=MAX_THROTTLE, split="validation")
    if len({int(row["run"]) for row in validation_windows}) != 6:
        raise ValueError("expected six independent dynamic validation captures")
    torch_parent, parent, parent_metadata = _load_model(PARENT_CHECKPOINT, str(device))
    if parent_metadata["dataset_sha256"] != source_hash:
        raise ValueError("parent checkpoint uses a different frozen dataset")
    parent.eval()
    with torch.no_grad():
        parent_metrics = _evaluate(
            torch, parent, data, state, parent_metadata, validation_windows,
            device, command_offset=-1)
    parent_score, parent_components = _selection_score(parent_metrics)
    del parent, torch_parent

    samplers: dict[int, tuple[FamilyConditionSampler, dict[str, Any]]] = {}
    for horizon, _, _ in STAGES:
        samplers[horizon] = _build_window_sampler(
            data, state, horizon, "train", max_throttle_command=MAX_THROTTLE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    output_dir.mkdir(parents=True, exist_ok=False)
    best_score = float("inf")
    best_state: dict[str, Any] | None = None
    best_validation: dict[str, Any] | None = None
    update = 0
    validation_history: list[dict[str, Any]] = []
    stage_history: list[dict[str, Any]] = []
    began = time.perf_counter()

    for horizon, updates, batch_size in STAGES:
        sampler, sampler_report = samplers[horizon]
        stage_losses = []
        stage_started = time.perf_counter()
        for local_step in range(1, updates + 1):
            update += 1
            batch = _sample_batch(
                data, state, acceleration, sampler, batch_size, horizon, rng)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, components = _loss(torch, nn, model, batch, normalizers, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {update}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0))
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at update {update}")
            optimizer.step()
            stage_losses.append(float(loss.detach()))

            if local_step % EVAL_INTERVAL == 0 or local_step == updates:
                model.eval()
                with torch.no_grad():
                    metrics = _evaluate(
                        torch, model, data, state, normalizers,
                        validation_windows, device, command_offset=-1)
                score, components_score = _selection_score(metrics)
                row = {
                    "update": update,
                    "stage_horizon_steps": horizon,
                    "stage_step": local_step,
                    "selection_score": score,
                    "selection_components": components_score,
                    "metrics": metrics,
                }
                validation_history.append(row)
                if score < best_score:
                    best_score = score
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()}
                    best_validation = row
                    model_payload = {
                        "schema_version": 1,
                        "metadata": {
                            "model": "shared_history_gru_effective_acceleration_teacher",
                            "architecture": (
                                "single GRU encodes measured 80-frame context and is "
                                "then advanced by its own predicted state/next-command rows"),
                            "dataset_path": str(DEFAULT_DATASET),
                            "dataset_sha256": source_hash,
                            "dataset_schema_version": int(data["schema_version"]),
                            "training_runs": _training_runs(data),
                            "validation_runs": sorted(
                                np.asarray(data["run_ids"]).astype(str)[
                                    np.asarray(data["splits"]).astype(str)
                                    == "validation"].tolist()),
                            "future_inputs": [
                                "steering_command_rad", "throttle_command_norm"],
                            "future_truth_or_sensor_feedback_used": False,
                            "control_dt_s": 0.025,
                            "control_rate_hz": 40.0,
                            "internal_simulation_rate_hz": 40.0,
                            "history_steps": HISTORY_STEPS,
                            "state_names": [
                                "u_com_mps", "v_com_mps", "yaw_rate_rps",
                                "steering_feedback_rad", "throttle_feedback_norm",
                                "rear_left_surface_speed_mps",
                                "rear_right_surface_speed_mps",
                                "imu_roll_rad", "imu_roll_rate_rps"],
                            "history_feature_names": [
                                "u_com_mps", "v_com_mps", "yaw_rate_rps",
                                "steering_feedback_rad", "throttle_feedback_norm",
                                "rear_left_surface_speed_mps",
                                "rear_right_surface_speed_mps", "imu_roll_rad",
                                "imu_roll_rate_rps", "steering_command_rad",
                                "throttle_command_norm"],
                            "acceleration_names": [
                                "ax_effective_mps2", "ay_effective_mps2",
                                "yaw_accel_rps2", "rear_left_surface_accel_mps2",
                                "rear_right_surface_accel_mps2"],
                            "command_alignment_offset_frames": -1,
                            "normalizers": {
                                name: np.asarray(normalizers[name]).tolist()
                                for name in ("history_mean", "history_scale",
                                             "state_mean", "state_scale",
                                             "command_mean", "command_scale",
                                             "acceleration_bounds")},
                            "actuator_fit": actuator_metadata(actuator_fit),
                            "roll_oscillator_fit": roll_report,
                            "roll_oscillator_coefficients": roll_coefficients.tolist(),
                            "parent_checkpoint_sha256": PARENT_SHA256,
                            "checkpoint_selection": (
                                "macro whole-run dynamic validation only; practice/test/final-test excluded"),
                            "selection_score": score,
                            "selection_components": components_score,
                        },
                        "state_dict": best_state,
                        "global_step": update,
                    }
                    torch.save(model_payload, output_dir / "checkpoint.pt")
                    (output_dir / "checkpoint.sha256").write_text(
                        _sha256(output_dir / "checkpoint.pt") + "\n",
                        encoding="utf-8")
                print(
                    f"update={update} horizon={horizon} train_loss="
                    f"{np.mean(stage_losses[-min(20, len(stage_losses)):]):.5f} "
                    f"val_score={score:.5f} best={best_score:.5f} "
                    f"elapsed={time.perf_counter()-began:.1f}s",
                    flush=True)

        stage_history.append({
            "horizon_steps": horizon,
            "updates": updates,
            "batch_size": batch_size,
            "last_20_mean_training_loss": float(np.mean(stage_losses[-20:])),
            "elapsed_seconds": time.perf_counter() - stage_started,
            "sampler": sampler_report,
        })

    if best_state is None or best_validation is None:
        raise RuntimeError("training produced no validation-selected checkpoint")
    report = {
        "study": "shared history/rollout GRU plus explicit rigid-body and rear-wheel dynamics",
        "dataset_path": str(DEFAULT_DATASET),
        "dataset_sha256": source_hash,
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "parent_checkpoint_sha256": PARENT_SHA256,
        "output_checkpoint_sha256": _sha256(output_dir / "checkpoint.pt"),
        "training_runs": _training_runs(data),
        "validation_run_count": len({int(row["run"]) for row in validation_windows}),
        "test_and_final_test_used": False,
        "practice_used_for_selection": False,
        "future_truth_or_sensor_feedback_used": False,
        "command_alignment_offset_frames": -1,
        "training_stages": stage_history,
        "training_updates": update,
        "elapsed_seconds": time.perf_counter() - began,
        "parent_validation_selection_score": parent_score,
        "parent_validation_selection_components": parent_components,
        "best_candidate_validation_selection_score": best_score,
        "best_candidate_validation_selection_components": best_validation[
            "selection_components"],
        "candidate_to_parent_selection_score_ratio": best_score / parent_score,
        "candidate_improved_validation_selection_score": best_score < parent_score,
        "best_checkpoint_validation_metrics": best_validation["metrics"],
        "validation_checkpoint_history": validation_history,
        "decision": (
            "retain as research candidate only if later whole-practice and held-out "
            "test runs confirm; no production integration"),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train(args.output.resolve(), args.device)
    print(json.dumps({
        "checkpoint": str(args.output.resolve() / "checkpoint.pt"),
        "checkpoint_sha256": result["output_checkpoint_sha256"],
        "parent_score": result["parent_validation_selection_score"],
        "candidate_score": result["best_candidate_validation_selection_score"],
        "ratio": result["candidate_to_parent_selection_score_ratio"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
