#!/usr/bin/env python3
"""Paired EDSSM/RSSM recursive scores on identical whole-run starts.

The comparison is validation-only: one open-plane dynamic validation set plus
the frozen 135-start production-practice benchmark. No test/final-test rows are
loaded or used for model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    REAR_AXLE_TO_COM_M,
    append_roll_state,
    physical_state_from_dataset,
    integrate_pose,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.encoder_sidecar_compatibility import (
    is_fixed_cadence_variant,
)
from tools.vehicle_dynamics_learning.offline_plant import load_rssm_teacher_plant
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    DEFAULT_DATASET,
    _fixed_validation_windows,
    _load_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_REGISTRY = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_sim_baseline_registry_20261002.json"
)
PRACTICE_BENCHMARK = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/practice_transfer_benchmark_v1.json"
)
OUTPUT_DIR = (
    REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "edssm_training_20261002/paired_comparison"
)
HORIZONS = {"0.75s": 30, "2s": 80, "5s": 200, "10s": 400}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for paired EDSSM comparison") from exc
    return torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_data_path(value: str) -> Path:
    raw = Path(value)
    if str(raw).startswith("/workspace/"):
        raw = REPO_ROOT / str(raw).removeprefix("/workspace/")
    elif not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw.resolve()


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _build_inputs(data: dict[str, Any], state: np.ndarray,
                  windows: list[dict[str, Any]],
                  history_state_size: int | None = None,
                  raw_history_features: np.ndarray | None = None,
                  rollout_steps: int = max(HORIZONS.values())
                  ) -> tuple[np.ndarray, ...]:
    history_state_size = (state.shape[1] if history_state_size is None
                          else int(history_state_size))
    if not 0 < history_state_size <= state.shape[1]:
        raise ValueError("invalid EDSSM history state size")
    frames = data["frames"]
    histories, initial_states, delayed, commands, truth_states, truth_poses = (
        [], [], [], [], [], [])
    for window in windows:
        start = int(window["start"])
        context_indices = np.arange(start - HISTORY_STEPS + 1, start + 1)
        future_indices = np.arange(start + 1, start + rollout_steps + 1)
        histories.append(np.column_stack((state[context_indices,
                                                  :history_state_size],
                                          frames[context_indices, 7:9])))
        if raw_history_features is not None:
            histories[-1] = np.column_stack((
                histories[-1], raw_history_features[context_indices]))
        initial_states.append(state[start])
        delayed.append(frames[start - 1, 7:9])
        commands.append(frames[future_indices, 7:9])
        # Plant state is simulator-truth COM motion plus actuator and encoder
        # channels; frames[:, :7] is the separate odometry rear-axle feature
        # vector. Returning frames here silently mixed reference frames.
        truth_states.append(state[future_indices])
        truth_poses.append(data["simulator_pose_xyyaw"][
            np.concatenate(([start], future_indices))])
    return tuple(np.asarray(values, dtype=np.float32) for values in (
        histories, initial_states, delayed, commands, truth_states, truth_poses))


def _edssm_rollouts(torch, model, data: dict[str, Any], state: np.ndarray,
                    windows: list[dict[str, Any]], device: str,
                    batch_size: int = 24,
                    rollout_steps: int = max(HORIZONS.values())
                    ) -> tuple[np.ndarray, np.ndarray]:
    if model.include_roll_state and state.shape[1] == 7:
        state = append_roll_state(data, state)
    predicted_states, predicted_poses = [], []
    raw_history = (
        raw_encoder_history_features(data) if model.include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if model.include_wheel_innovation_history else None)
    for offset in range(0, len(windows), batch_size):
        rows = windows[offset:offset + batch_size]
        history, initial, delayed, commands, _, poses = _build_inputs(
            data, state, rows, model.history_state_size, raw_history,
            rollout_steps=rollout_steps)
        with torch.no_grad():
            prediction, _, _, _ = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(commands, dtype=torch.float32, device=device))
            pose = integrate_pose(
                torch, prediction,
                torch.as_tensor(poses[:, 0], dtype=torch.float32, device=device),
                torch.as_tensor(initial, dtype=torch.float32, device=device))
        predicted_states.append(prediction.cpu().numpy().astype(np.float64))
        predicted_poses.append(pose.cpu().numpy().astype(np.float64))
    return np.concatenate(predicted_states), np.concatenate(predicted_poses)


def _rssm_rollouts(checkpoint: Path, dataset_path: Path,
                   data: dict[str, Any], windows: list[dict[str, Any]],
                   device: str,
                   rollout_steps: int = max(HORIZONS.values())
                   ) -> tuple[np.ndarray, np.ndarray]:
    plant = load_rssm_teacher_plant(
        [checkpoint], dataset_path, device=device,
        max_supported_speed_mps=12.0)
    state_out = np.empty((len(windows), rollout_steps, 7), dtype=np.float64)
    pose_out = np.empty((len(windows), rollout_steps, 3), dtype=np.float64)
    for row_index, window in enumerate(windows):
        start = int(window["start"])
        context_indices = np.arange(start - HISTORY_STEPS + 1, start + 1)
        future_indices = np.arange(start + 1, start + rollout_steps + 1)
        history = data["frames"][context_indices]
        truth_poses = data["simulator_pose_xyyaw"]
        estimate = plant.reset(history, truth_poses[start])
        for step, frame_index in enumerate(future_indices):
            command = data["frames"][frame_index, 7:9]
            estimate = plant.step(float(command[0]), float(command[1]), DT_S)
            state_out[row_index, step] = estimate.state[3:]
            pose_out[row_index, step] = estimate.state[:3]
        if not np.isfinite(state_out[row_index]).all() or not np.isfinite(
                pose_out[row_index]).all():
            raise FloatingPointError("RSSM produced a non-finite paired rollout")
    return state_out, pose_out


def _per_window_errors(pred_state: np.ndarray, pred_pose: np.ndarray,
                       truth_state: np.ndarray, truth_pose: np.ndarray,
                       prediction_reference: str,
                       wheel_valid_mask: np.ndarray | None = None
                       ) -> dict[str, dict[str, float]]:
    if prediction_reference not in ("com", "rear_axle"):
        raise ValueError("prediction state reference must be COM or rear axle")
    # Report all body-state comparisons in the rigid-body COM frame. EDSSM
    # predicts COM directly; the production RSSM adapter exposes rear-axle
    # lateral velocity, so recover COM with v_com = v_rear + L*r.
    com_state = pred_state.copy()
    if prediction_reference == "rear_axle":
        com_state[:, :, 1] += REAR_AXLE_TO_COM_M * com_state[:, :, 2]
    position = pred_pose[:, :, :2] - truth_pose[:, 1:, :2]
    heading = _wrap(pred_pose[:, :, 2] - truth_pose[:, 1:, 2])
    speed = np.hypot(com_state[:, :, 0], com_state[:, :, 1]) - np.hypot(
        truth_state[:, :, 0], truth_state[:, :, 1])
    result: dict[str, dict[str, float]] = {}
    for name, steps in HORIZONS.items():
        if steps > pred_state.shape[1]:
            continue
        body = com_state[:, :steps, :3] - truth_state[:, :steps, :3]
        wheel = com_state[:, :steps, 5:7] - truth_state[:, :steps, 5:7]
        if wheel_valid_mask is None:
            wheel_rmse = np.sqrt(np.mean(wheel ** 2, axis=(1, 2)))
        else:
            mask = np.asarray(wheel_valid_mask[:, :steps], dtype=bool)
            count = mask.sum(axis=1) * wheel.shape[2]
            if np.any(count == 0):
                raise ValueError("a score window has no valid raw-wheel labels")
            wheel_rmse = np.sqrt(
                np.sum(wheel ** 2 * mask[:, :, None], axis=(1, 2)) / count)
        radial_position = np.linalg.norm(position[:, :steps], axis=2)
        local = {
            "u_com_rmse_mps": np.sqrt(np.mean(body[:, :, 0] ** 2, axis=1)),
            "v_com_rmse_mps": np.sqrt(np.mean(body[:, :, 1] ** 2, axis=1)),
            "yaw_rate_rmse_rps": np.sqrt(np.mean(body[:, :, 2] ** 2, axis=1)),
            "body_uvr_rmse": np.sqrt(np.mean(body ** 2, axis=(1, 2))),
            "speed_rmse_mps": np.sqrt(np.mean(speed[:, :steps] ** 2, axis=1)),
            "wheel_rmse_mps": wheel_rmse,
            "position_radial_rmse_m": np.sqrt(
                np.mean(radial_position ** 2, axis=1)),
            "heading_rmse_rad": np.sqrt(np.mean(heading[:, :steps] ** 2, axis=1)),
            "position_endpoint_m": radial_position[:, -1],
            "heading_endpoint_abs_rad": np.abs(heading[:, steps - 1]),
        }
        if pred_state.shape[2] == 9 and truth_state.shape[2] == 9:
            attitude_error = pred_state[:, :steps, 7:9] - truth_state[:, :steps, 7:9]
            local["roll_angle_rmse_rad"] = np.sqrt(np.mean(
                attitude_error[:, :, 0] ** 2, axis=1))
            local["roll_rate_rmse_rps"] = np.sqrt(np.mean(
                attitude_error[:, :, 1] ** 2, axis=1))
        result[name] = {key: values for key, values in local.items()}
    return result


def _run_macro(values: np.ndarray, run_indices: np.ndarray,
               run_ids: np.ndarray, seed: int = 20261002
               ) -> dict[str, Any]:
    per_run = {}
    for run in sorted(np.unique(run_indices)):
        selected = values[run_indices == run]
        per_run[str(run_ids[run])] = float(np.mean(selected))
    means = np.asarray(list(per_run.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(means), size=(10000, len(means)))
    ci = np.quantile(means[draws].mean(axis=1), (0.025, 0.975))
    return {
        "independent_runs": len(per_run),
        "macro_run_mean": float(np.mean(means)),
        "run_cluster_bootstrap_95pct_ci": ci.tolist(),
        "per_run": per_run,
    }


def _paired_run_difference(left: dict[str, Any], right: dict[str, Any]
                           ) -> dict[str, Any]:
    """Paired whole-run bootstrap for left-minus-right error."""
    left_runs = left["per_run"]
    right_runs = right["per_run"]
    if set(left_runs) != set(right_runs):
        raise ValueError("paired model summaries have different whole-run sets")
    run_ids = sorted(left_runs)
    differences = np.asarray(
        [left_runs[run] - right_runs[run] for run in run_ids], dtype=np.float64)
    baseline = np.asarray([right_runs[run] for run in run_ids], dtype=np.float64)
    rng = np.random.default_rng(20261002)
    draws = rng.integers(0, len(run_ids), size=(10000, len(run_ids)))
    sampled_delta = differences[draws].mean(axis=1)
    sampled_base = baseline[draws].mean(axis=1)
    percent = 100.0 * sampled_delta / np.maximum(sampled_base, 1e-12)
    return {
        "independent_paired_runs": len(run_ids),
        "macro_run_difference_left_minus_right": float(differences.mean()),
        "paired_run_difference_left_minus_right_95pct_ci": np.quantile(
            sampled_delta, (0.025, 0.975)).tolist(),
        "left_minus_right_percent_of_right": float(
            100.0 * differences.mean() / max(float(baseline.mean()), 1e-12)),
        "left_minus_right_percent_of_right_95pct_ci": np.quantile(
            percent, (0.025, 0.975)).tolist(),
        "per_run_difference": dict(zip(run_ids, differences.tolist())),
    }


def _candidate_pair_differences(left: dict[str, Any], right: dict[str, Any]
                               ) -> dict[str, Any]:
    """Paired whole-run differences between two EDSSM candidate reports."""
    result: dict[str, Any] = {"horizons": {}, "hard_regimes": {}}
    for horizon in sorted(set(left["horizons"]) & set(right["horizons"])):
        result["horizons"][horizon] = {}
        for metric in sorted(set(left["horizons"][horizon])
                             & set(right["horizons"][horizon])):
            a = left["horizons"][horizon][metric].get("edssm")
            b = right["horizons"][horizon][metric].get("edssm")
            if a is not None and b is not None:
                result["horizons"][horizon][metric] = _paired_run_difference(a, b)
    common_regimes = set(left["hard_regimes"]) & set(right["hard_regimes"])
    for regime in sorted(common_regimes):
        result["hard_regimes"][regime] = {"horizons": {}}
        left_horizons = left["hard_regimes"][regime]["horizons"]
        right_horizons = right["hard_regimes"][regime]["horizons"]
        for horizon in sorted(set(left_horizons) & set(right_horizons)):
            result["hard_regimes"][regime]["horizons"][horizon] = {}
            for metric in sorted(set(left_horizons[horizon])
                                 & set(right_horizons[horizon])):
                a = left_horizons[horizon][metric].get("edssm")
                b = right_horizons[horizon][metric].get("edssm")
                if a is not None and b is not None:
                    result["hard_regimes"][regime]["horizons"][horizon][metric] = (
                        _paired_run_difference(a, b))
    return result


def _summarize_pair(ed_state: np.ndarray, ed_pose: np.ndarray,
                    rssm_state: np.ndarray, rssm_pose: np.ndarray,
                    data: dict[str, Any], state: np.ndarray,
                    windows: list[dict[str, Any]], label: str,
                    rssm_truth_state: np.ndarray | None = None,
                    wheel_state_source: str = "filtered_odometry"
                    ) -> dict[str, Any]:
    rollout_steps = int(ed_state.shape[1])
    ed_truth_state = np.stack([
        state[int(row["start"]) + 1:
              int(row["start"]) + rollout_steps + 1]
        for row in windows])
    if rssm_truth_state is None:
        rssm_truth_state = state
    rssm_truth = np.stack([
        rssm_truth_state[int(row["start"]) + 1:
                         int(row["start"]) + rollout_steps + 1]
        for row in windows])
    if ed_state.shape[2] == 9 and state.shape[1] == 7:
        roll_state = append_roll_state(data, state)
        ed_truth_state = np.stack([
            roll_state[int(row["start"]) + 1:
                       int(row["start"]) + rollout_steps + 1]
            for row in windows])
    truth_pose = np.stack([
        data["simulator_pose_xyyaw"][
            int(row["start"]):int(row["start"]) + rollout_steps + 1]
        for row in windows])
    wheel_valid_mask = None
    if wheel_state_source == "raw_encoder":
        raw_wheel_valid = data.get("encoder_raw_valid")
        if raw_wheel_valid is None:
            raise ValueError("raw-wheel comparison requires validity labels")
        wheel_valid_mask = np.stack([
            raw_wheel_valid[int(row["start"]) + 1:
                            int(row["start"]) + rollout_steps + 1]
            for row in windows])
    ed_errors = _per_window_errors(
        ed_state, ed_pose, ed_truth_state, truth_pose, "com",
        wheel_valid_mask=wheel_valid_mask)
    rssm_errors = _per_window_errors(
        rssm_state, rssm_pose, rssm_truth, truth_pose, "rear_axle")
    run_indices = np.asarray([int(row["run"]) for row in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    regimes = sorted(set(tag for row in windows for tag in row.get("regimes", [])))
    output: dict[str, Any] = {
        "evaluation": label,
        "whole_run_count": int(len(np.unique(run_indices))),
        "window_count": int(len(windows)),
        "primary_unit": "independent whole validation run; windows nested within run",
        "state_error_reference": "body center of mass",
        "future_inputs_only": ["steering_command_rad", "throttle_command_norm"],
        "future_truth_or_sensor_feedback_used": False,
        "horizons": {},
        "hard_regimes": {},
    }
    for horizon in HORIZONS:
        if horizon not in ed_errors:
            continue
        output["horizons"][horizon] = {}
        for metric in ed_errors[horizon]:
            ed_summary = _run_macro(ed_errors[horizon][metric], run_indices, run_ids)
            if metric not in rssm_errors[horizon]:
                output["horizons"][horizon][metric] = {
                    "edssm": ed_summary,
                    "lead_rssm_comparable": False,
                }
            else:
                rssm_summary = _run_macro(
                    rssm_errors[horizon][metric], run_indices, run_ids)
                output["horizons"][horizon][metric] = {
                    "edssm": ed_summary,
                    "lead_rssm": rssm_summary,
                    "paired_run_difference": _paired_run_difference(
                        ed_summary, rssm_summary),
                }
    for regime in regimes:
        selected = np.asarray([regime in row.get("regimes", []) for row in windows])
        if not np.any(selected):
            continue
        output["hard_regimes"][regime] = {"horizons": {}}
        local_runs = run_indices[selected]
        for horizon in HORIZONS:
            if horizon not in ed_errors:
                continue
            output["hard_regimes"][regime]["horizons"][horizon] = {}
            for metric in ed_errors[horizon]:
                ed_summary = _run_macro(
                    ed_errors[horizon][metric][selected], local_runs, run_ids)
                if metric not in rssm_errors[horizon]:
                    output["hard_regimes"][regime]["horizons"][horizon][metric] = {
                        "edssm": ed_summary,
                        "lead_rssm_comparable": False,
                    }
                else:
                    rssm_summary = _run_macro(
                        rssm_errors[horizon][metric][selected], local_runs, run_ids)
                    output["hard_regimes"][regime]["horizons"][horizon][metric] = {
                        "edssm": ed_summary,
                        "lead_rssm": rssm_summary,
                        "paired_run_difference": _paired_run_difference(
                            ed_summary, rssm_summary),
                    }
    output["window_manifest"] = [
        {"run_id": str(run_ids[int(row["run"])]),
         "sequence_id": int(row["sequence_id"]), "start": int(row["start"]),
         "regimes": row.get("regimes", [])}
        for row in windows
    ]
    del state
    return output


def _dynamic_windows(data: dict[str, Any], state: np.ndarray
                     ) -> list[dict[str, Any]]:
    windows = _fixed_validation_windows(
        data, state, max_windows_per_run=24,
        horizon_steps=max(HORIZONS.values()))
    # Keep the already frozen validation starts used for checkpoint selection;
    # add a post-selection frontier label without changing those starts.
    for row in windows:
        start = int(row["start"])
        future = slice(start + 1, start + 201)
        frontier = (
            (np.hypot(state[start, 0], state[start, 1]) >= 9.0
             and abs(state[start, 3]) >= 0.10)
            or np.any((np.hypot(state[future, 0], state[future, 1]) >= 9.0)
                      & (np.abs(state[future, 3]) >= 0.10)))
        if frontier:
            row["regimes"] = sorted(set(row.get("regimes", []))
                                     | {"steering_frontier"})
    return windows


def _practice_windows(benchmark: dict[str, Any], data: dict[str, Any],
                      horizon_steps: int = max(HORIZONS.values())
                      ) -> list[dict[str, Any]]:
    by_run = {str(run_id): index for index, run_id in
              enumerate(np.asarray(data["run_ids"]).astype(str))}
    sequence_end = np.full(len(data["frames"]), -1, dtype=np.int64)
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if str(data["splits"][run]) == "unseen_practice":
            sequence_end[start:end] = end
    rows = []
    for item in benchmark["windows"]:
        run_id = str(item["run_id"])
        if run_id not in by_run or data["splits"][by_run[run_id]] != "unseen_practice":
            raise ValueError(f"practice benchmark references non-validation run {run_id}")
        start = int(item["global_start_index"])
        if (start < 0 or start >= len(sequence_end)
                or sequence_end[start] < 0
                or start + horizon_steps >= sequence_end[start]):
            continue
        rows.append({
            "run": by_run[run_id],
            "sequence_id": -1,
            "start": start,
            "regimes": list(item["categories"]),
            "run_id": run_id,
        })
    return rows


def _registry_rssm_path(registry_path: Path) -> Path:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    historical = registry["historical_baseline"][
        "split_policy_and_pre-handoff_artifacts"]["model_checkpoints"]
    rows = [row for row in historical
            if row.get("role", "").startswith("lead race-domain RSSM")]
    if len(rows) != 1:
        raise ValueError("frozen registry does not identify one lead RSSM")
    path = (REPO_ROOT / rows[0]["path"]).resolve()
    if _sha256(path) != rows[0]["sha256"]:
        raise ValueError("lead RSSM checkpoint hash differs from WP0 registry")
    return path


def compare(encoder_checkpoints: dict[str, Path], dataset_path: Path,
            practice_benchmark_path: Path, registry_path: Path,
            output_dir: Path, device: str = "cuda") -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    practice_benchmark_path = practice_benchmark_path.resolve()
    registry_path = registry_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite paired comparison: {output_dir}")
    for name, checkpoint in encoder_checkpoints.items():
        if (not name or any(char not in "abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name)
                or not checkpoint.is_file()):
            raise ValueError(f"invalid {name} EDSSM checkpoint: {checkpoint}")
    if len(encoder_checkpoints) != 2:
        raise ValueError("paired comparison requires exactly two EDSSM candidates")
    lead_rssm = _registry_rssm_path(registry_path)
    data = _load_dataset(dataset_path)
    if int(data["schema_version"]) != 9:
        raise ValueError("dynamic paired score requires frozen schema-9 data")
    filtered_state = physical_state_from_dataset(
        data, wheel_state_source="filtered_odometry").astype(np.float32)
    raw_available = data.get("encoder_raw_valid") is not None
    raw_state = (physical_state_from_dataset(
        data, wheel_state_source="raw_encoder").astype(np.float32)
        if raw_available else filtered_state)
    dynamic_windows = _dynamic_windows(data, filtered_state)
    dataset_hash = _sha256(dataset_path)
    if len(dynamic_windows) == 0:
        raise ValueError("dynamic validation has no common ten-second starts")
    models: dict[str, tuple[Any, Any]] = {}
    candidate_metadata: dict[str, dict[str, Any]] = {}
    for name, checkpoint in encoder_checkpoints.items():
        torch, model, metadata = _load_model(checkpoint.resolve(), device)
        accepted_hashes = {dataset_hash}
        source_hash = data.get("encoder_raw_source_dataset_sha256")
        if source_hash:
            accepted_hashes.add(str(source_hash))
        dataset_compatibility = "exact trained dataset or authenticated source dataset"
        if metadata["dataset_sha256"] not in accepted_hashes:
            if not is_fixed_cadence_variant(metadata, dataset_path):
                raise ValueError(f"{name} checkpoint was trained from a different dataset")
            dataset_compatibility = (
                "authenticated fixed-25ms encoder-rate variant; all source arrays "
                "and raw-wheel validity are identical, only valid raw-wheel rates differ")
        if set(metadata["training_runs"]).intersection(
                np.asarray(data["run_ids"])[np.asarray(data["splits"]).astype(str)
                                            == "validation"].astype(str)):
            raise ValueError(f"{name} checkpoint overlaps validation runs")
        models[name] = (torch, model)
        candidate_metadata[name] = {
            "encoder": metadata["encoder"],
            "latent_size": int(metadata["latent_size"]),
            "expert_count": int(metadata.get("expert_count", 1)),
            "seed": int(metadata["seed"]),
            "wheel_state_source": str(metadata.get(
                "wheel_state_source", "filtered_odometry")),
            "evaluation_dataset_compatibility": dataset_compatibility,
            "include_raw_encoder_history": bool(metadata.get(
                "include_raw_encoder_history", False)),
            "include_wheel_innovation_history": bool(metadata.get(
                "include_wheel_innovation_history", False)),
            "validation_run_ids": metadata["validation_runs"],
        }
    if (not raw_available and any(
            row["wheel_state_source"] == "raw_encoder"
            for row in candidate_metadata.values())):
        raise ValueError("raw_encoder candidate requires a verified encoder-state view")
    dynamic_states = {
        name: (raw_state if candidate_metadata[name]["wheel_state_source"]
               == "raw_encoder" else filtered_state)
        for name in encoder_checkpoints
    }
    dynamic_rssm_state, dynamic_rssm_pose = _rssm_rollouts(
        lead_rssm, dataset_path, data, dynamic_windows, device)
    dynamic_report = {}
    for name, checkpoint in encoder_checkpoints.items():
        torch, model = models[name]
        state = dynamic_states[name]
        ed_state, ed_pose = _edssm_rollouts(
            torch, model, data, state, dynamic_windows, device)
        dynamic_report[name] = _summarize_pair(
            ed_state, ed_pose, dynamic_rssm_state, dynamic_rssm_pose,
            data, state, dynamic_windows, "open-plane dynamic validation",
            rssm_truth_state=filtered_state,
            wheel_state_source=candidate_metadata[name]["wheel_state_source"])

    benchmark = json.loads(practice_benchmark_path.read_text(encoding="utf-8"))
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or benchmark.get("training_or_checkpoint_selection_use") is not False):
        raise ValueError("practice benchmark is not frozen validation-only data")
    practice_path = _repo_data_path(str(benchmark["dataset"]))
    if _sha256(practice_path) != benchmark["dataset_sha256"]:
        raise ValueError("practice benchmark dataset hash changed")
    practice_data = _load_dataset(practice_path)
    practice_filtered_state = physical_state_from_dataset(
        practice_data, wheel_state_source="filtered_odometry").astype(np.float32)
    practice_raw_available = practice_data.get("encoder_raw_valid") is not None
    practice_raw_state = (physical_state_from_dataset(
        practice_data, wheel_state_source="raw_encoder").astype(np.float32)
        if practice_raw_available else practice_filtered_state)
    practice_rollout_steps = 200
    practice_windows = _practice_windows(
        benchmark, practice_data, horizon_steps=practice_rollout_steps)
    if not practice_windows:
        raise ValueError(
            "practice benchmark has no starts with a complete five-second continuation")
    practice_rssm_state, practice_rssm_pose = _rssm_rollouts(
        lead_rssm, practice_path, practice_data, practice_windows, device,
        rollout_steps=practice_rollout_steps)
    practice_report = {}
    for name, checkpoint in encoder_checkpoints.items():
        torch, model = models[name]
        practice_state = (practice_raw_state
                          if candidate_metadata[name]["wheel_state_source"]
                          == "raw_encoder" else practice_filtered_state)
        if (candidate_metadata[name]["wheel_state_source"] == "raw_encoder"
                and not practice_raw_available):
            raise ValueError("raw_encoder candidate requires a verified practice encoder view")
        ed_state, ed_pose = _edssm_rollouts(
            torch, model, practice_data, practice_state,
            practice_windows, device,
            rollout_steps=practice_rollout_steps)
        practice_report[name] = _summarize_pair(
            ed_state, ed_pose, practice_rssm_state, practice_rssm_pose,
            practice_data, practice_state, practice_windows,
            "frozen production-practice starts with a complete five-second continuation",
            rssm_truth_state=practice_filtered_state,
            wheel_state_source=candidate_metadata[name]["wheel_state_source"])

    candidate_names = list(encoder_checkpoints)
    candidate_differences = {
        "candidate_a": candidate_names[0],
        "candidate_b": candidate_names[1],
        "difference_convention": (
            "candidate_a minus candidate_b; negative means A has lower error"),
        "dynamic_validation": _candidate_pair_differences(
            dynamic_report[candidate_names[0]],
            dynamic_report[candidate_names[1]]),
        "production_practice_transfer": _candidate_pair_differences(
            practice_report[candidate_names[0]],
            practice_report[candidate_names[1]]),
    }
    wheel_sources_match = (
        candidate_metadata[candidate_names[0]]["wheel_state_source"]
        == candidate_metadata[candidate_names[1]]["wheel_state_source"])
    if not wheel_sources_match:
        def remove_incomparable_wheel_deltas(value):
            if isinstance(value, dict):
                value.pop("wheel_rmse_mps", None)
                for child in value.values():
                    remove_incomparable_wheel_deltas(child)
        remove_incomparable_wheel_deltas(candidate_differences)
    for name in candidate_names:
        dynamic_report[name]["wheel_state_source"] = candidate_metadata[name][
            "wheel_state_source"]
        practice_report[name]["wheel_state_source"] = candidate_metadata[name][
            "wheel_state_source"]
    candidate_differences["wheel_metric_comparable"] = wheel_sources_match
    candidate_differences["wheel_metric_note"] = (
        "paired wheel deltas omitted because candidates predict different wheel-state signals"
        if not wheel_sources_match else "same wheel-state target semantics")

    report = {
        "schema_version": 2,
        "purpose": "paired same-start recursive model comparison through a ten-second horizon before any optimizer integration",
        "horizon_steps": HORIZONS,
        "dynamic_dataset": str(dataset_path),
        "dynamic_dataset_sha256": dataset_hash,
        "practice_benchmark": str(practice_benchmark_path),
        "practice_benchmark_sha256": _sha256(practice_benchmark_path),
        "practice_benchmark_window_count": len(benchmark["windows"]),
        "practice_rollout_steps": practice_rollout_steps,
        "practice_rollout_window_count": len(practice_windows),
        "practice_dataset": str(practice_path),
        "practice_dataset_sha256": _sha256(practice_path),
        "lead_rssm_checkpoint": str(lead_rssm),
        "lead_rssm_checkpoint_sha256": _sha256(lead_rssm),
        "edssm_checkpoints": {
            name: {"path": str(path.resolve()),
                   "sha256": _sha256(path.resolve()),
                   **candidate_metadata[name]}
            for name, path in encoder_checkpoints.items()
        },
        "test_and_final_test_used": False,
        "selection_was_frozen_before_practice_transfer": True,
        "future_truth_or_feedback_used": False,
        "identical_starts_and_commands_within_each_domain": True,
        "dynamic_validation": dynamic_report,
        "production_practice_transfer": practice_report,
        "paired_candidate_differences": candidate_differences,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gru-checkpoint", type=Path)
    parser.add_argument("--tcn-checkpoint", type=Path)
    parser.add_argument("--candidate-a-name")
    parser.add_argument("--candidate-a-checkpoint", type=Path)
    parser.add_argument("--candidate-b-name")
    parser.add_argument("--candidate-b-checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-benchmark", type=Path,
                        default=PRACTICE_BENCHMARK)
    parser.add_argument("--registry", type=Path, default=BASELINE_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    candidate_args = (args.candidate_a_name, args.candidate_a_checkpoint,
                      args.candidate_b_name, args.candidate_b_checkpoint)
    if any(value is not None for value in candidate_args):
        if not all((args.candidate_a_name, args.candidate_a_checkpoint,
                    args.candidate_b_name, args.candidate_b_checkpoint)):
            parser.error("candidate A/B each require both a name and checkpoint")
        candidates = {
            args.candidate_a_name: args.candidate_a_checkpoint,
            args.candidate_b_name: args.candidate_b_checkpoint,
        }
    else:
        if not args.gru_checkpoint or not args.tcn_checkpoint:
            parser.error("supply both --gru-checkpoint and --tcn-checkpoint")
        candidates = {"gru": args.gru_checkpoint, "tcn": args.tcn_checkpoint}
    try:
        report = compare(
            candidates,
            args.dataset, args.practice_benchmark, args.registry,
            args.output_dir, args.device)
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            FloatingPointError) as exc:
        print(f"paired EDSSM/RSSM comparison failed: {exc}", file=sys.stderr)
        return 2
    summary = {}
    for domain_key, domain in (("dynamic", report["dynamic_validation"]),
                               ("practice", report["production_practice_transfer"])):
        summary[domain_key] = {}
        for encoder, result in domain.items():
            summary[domain_key][encoder] = {
                horizon: {
                    metric: values["edssm"]["macro_run_mean"]
                    for metric, values in metrics.items()
                }
                for horizon, metrics in result["horizons"].items()
            }
    print(json.dumps({"summary": summary,
                      "report": str(args.output_dir.resolve()
                                    / "paired_comparison.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
