#!/usr/bin/env python3
"""Train a non-recursive future-trajectory predictor as a plant ceiling test.

The model sees a fixed history of measured state/input features and the whole
future steering/throttle command sequence. Future physical states, actuator
feedback, wheel speeds, and simulator truth are targets only. This diagnostic
predictor is not a step-able plant and is not eligible for runtime use.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.family_condition_sampler import (
    build_sequence_sampler,
)
from tools.vehicle_dynamics_learning.experiment_artifacts import (
    write_standard_artifacts,
)
from tools.vehicle_dynamics_learning.race_domain_objectives import (
    RACE_SPEED_DOMAINS,
    RACE_SPEED_BIN_EDGES_MPS,
    race_speed_bin_weights,
    race_speed_domain_labels,
    weighted_smooth_l1,
)


DT_S = 0.025
COM_X_M = 0.15532
STATE_COUNT = 7
TARGET_COUNT = 10
STATE_NAMES = (
    "u_com_mps", "v_com_mps", "yaw_rate_rps", "steering_feedback_rad",
    "throttle_feedback_norm", "rear_left_surface_mps",
    "rear_right_surface_mps",
)
ACCELERATION_NAMES = ("ax_body_mps2", "ay_body_mps2", "yaw_acceleration_rps2")


def _torch_model(torch, nn, context_steps: int, future_steps: int,
                 width: int, layer_count: int, head_count: int,
                 dropout: float):
    class DirectTrajectoryTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_steps = context_steps
            self.future_steps = future_steps
            self.context_projection = nn.Linear(9, width)
            self.command_projection = nn.Linear(2, width)
            self.segment_embedding = nn.Parameter(torch.empty(1, 2, width))
            self.position_embedding = nn.Parameter(
                torch.empty(1, context_steps + future_steps, width))
            nn.init.normal_(self.segment_embedding, std=0.02)
            nn.init.normal_(self.position_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=head_count, dim_feedforward=4 * width,
                dropout=dropout, batch_first=True, norm_first=False,
                activation="gelu")
            self.sequence = nn.TransformerEncoder(
                layer, num_layers=layer_count, enable_nested_tensor=False)
            self.output = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, TARGET_COUNT))

        def forward(self, context, future_commands):
            if context.shape[1] != self.context_steps:
                raise ValueError("context length does not match model config")
            if future_commands.shape[1] != self.future_steps:
                raise ValueError("future-command length does not match model config")
            context_tokens = (self.context_projection(context)
                              + self.segment_embedding[:, 0:1])
            command_tokens = (self.command_projection(future_commands)
                              + self.segment_embedding[:, 1:2])
            tokens = torch.cat((context_tokens, command_tokens), dim=1)
            encoded = self.sequence(tokens + self.position_embedding)
            return self.output(encoded[:, self.context_steps:])

    return DirectTrajectoryTransformer


def _targets(data: dict[str, Any]) -> np.ndarray:
    rigid = data["simulator_rigid_state"]
    acceleration = data["simulator_linear_acceleration"]
    if rigid is None or acceleration is None:
        raise ValueError("direct teacher requires schema-6 simulator labels")
    frames = data["frames"]
    state = np.column_stack((
        rigid[:, 7], rigid[:, 8], rigid[:, 12], frames[:, 3:7],
    )).astype(np.float32, copy=False)
    yaw_acceleration = np.full(len(frames), np.nan, dtype=np.float32)
    for start_value, end_value in data["bounds"]:
        start, end = int(start_value), int(end_value)
        if end - start > 1:
            yaw_acceleration[start:end - 1] = (
                np.diff(rigid[start:end, 12]) / DT_S)
    return np.column_stack((state, acceleration[:, :2],
                            yaw_acceleration)).astype(np.float32)


def _eligible_sequences(data: dict[str, Any], targets: np.ndarray,
                        split: str, context_steps: int,
                        future_steps: int) -> dict[int, list[tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    minimum = context_steps + future_steps + 1
    for seq_index, (start_value, end_value) in enumerate(data["bounds"]):
        run_index = int(data["seq_run"][seq_index])
        if data["splits"][run_index] != split:
            continue
        start, end = int(start_value), int(end_value)
        if end - start < minimum:
            continue
        if (not np.isfinite(data["frames"][start:end]).all()
                or not np.isfinite(targets[start:end - 1]).all()):
            continue
        groups[run_index].append((start, end))
    return dict(groups)


def _fit_normalizers(data: dict[str, Any], targets: np.ndarray,
                     train_groups: dict[int, list[tuple[int, int]]]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = np.zeros(len(data["frames"]), dtype=bool)
    for sequences in train_groups.values():
        for start, end in sequences:
            # The final yaw-acceleration label in a sequence is undefined.
            selected[start:end - 1] = True
    features = data["frames"][selected].astype(np.float64)
    labels = targets[selected].astype(np.float64)
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("training normalizers received non-finite rows")
    x_mean = features.mean(axis=0).astype(np.float32)
    x_scale = np.maximum(features.std(axis=0), np.asarray(
        [0.25, 0.10, 0.10, 0.02, 0.02, 0.25, 0.25, 0.02, 0.05],
        dtype=np.float64)).astype(np.float32)
    y_mean = labels.mean(axis=0).astype(np.float32)
    y_scale = np.maximum(labels.std(axis=0), np.asarray(
        [0.5, 0.25, 0.25, 0.02, 0.02, 0.5, 0.5, 0.5, 0.5, 0.5],
        dtype=np.float64)).astype(np.float32)
    return x_mean, x_scale, y_mean, y_scale


def _sample_windows(data: dict[str, Any], groups: dict[int, list[tuple[int, int]]],
                    count: int, context_steps: int, future_steps: int,
                    rng: np.random.Generator) -> list[tuple[int, int]]:
    run_ids = np.asarray(sorted(groups), dtype=np.int32)
    if not len(run_ids):
        return []
    windows = []
    per_run: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for run_id in run_ids:
        sequences = groups[int(run_id)]
        for index in range(count):
            start, end = sequences[index % len(sequences)]
            last_start = end - context_steps - future_steps - 1
            offset = int(rng.integers(0, last_start - start + 1))
            per_run[int(run_id)].append((start + offset, end))
    rounds = max((len(items) for items in per_run.values()), default=0)
    for round_index in range(rounds):
        for run_id in run_ids:
            local = per_run[int(run_id)]
            if round_index < len(local):
                windows.append(local[round_index])
    return windows


def _training_batch(data: dict[str, Any], targets: np.ndarray,
                    groups: dict[int, list[tuple[int, int]]],
                    batch_size: int, context_steps: int, future_steps: int,
                    rng: np.random.Generator,
                    family_sampler=None) -> tuple[np.ndarray, ...]:
    run_ids = np.asarray(sorted(groups), dtype=np.int32)
    context_rows, command_rows, target_rows = [], [], []
    for _ in range(batch_size):
        if family_sampler is None:
            run_id = int(rng.choice(run_ids))
            sequences = groups[run_id]
            start, end = sequences[int(rng.integers(0, len(sequences)))]
        else:
            run_id, (start, end), _, _ = family_sampler.sample(rng)
        last_start = end - context_steps - future_steps - 1
        index = int(rng.integers(start, last_start + 1))
        future_start = index + context_steps
        context_rows.append(data["frames"][index:future_start])
        command_rows.append(data["frames"][future_start:future_start + future_steps,
                                             7:9])
        target_rows.append(targets[future_start:future_start + future_steps])
    return (np.stack(context_rows), np.stack(command_rows),
            np.stack(target_rows))


def _windows_by_run(data: dict[str, Any], targets: np.ndarray,
                    groups: dict[int, list[tuple[int, int]]],
                    max_windows_per_run: int, context_steps: int,
                    future_steps: int, seed: int
                    ) -> dict[int, list[tuple[int, int]]]:
    result = {}
    for run_id, sequences in sorted(groups.items()):
        rng = np.random.default_rng(seed + int(run_id))
        windows = _sample_windows(
            data, {run_id: sequences}, max_windows_per_run,
            context_steps, future_steps, rng)
        result[run_id] = windows
    return result


def _pose_rollout(predicted_state: np.ndarray, initial_state: np.ndarray,
                  initial_pose: np.ndarray) -> np.ndarray:
    pose = np.empty((len(predicted_state), 3), dtype=np.float64)
    x, y, yaw = map(float, initial_pose)
    previous = np.asarray(initial_state[:3], dtype=np.float64)
    for index, following in enumerate(predicted_state[:, :3]):
        u = 0.5 * (previous[0] + following[0])
        r = 0.5 * (previous[2] + following[2])
        v_rear = 0.5 * ((previous[1] - COM_X_M * previous[2])
                        + (following[1] - COM_X_M * following[2]))
        yaw_mid = yaw + 0.5 * r * DT_S
        x += (u * math.cos(yaw_mid) - v_rear * math.sin(yaw_mid)) * DT_S
        y += (u * math.sin(yaw_mid) + v_rear * math.cos(yaw_mid)) * DT_S
        yaw += r * DT_S
        pose[index] = (x, y, yaw)
        previous = following
    return pose


def _evaluate(torch, model, data: dict[str, Any], targets: np.ndarray,
              groups: dict[int, list[tuple[int, int]]], x_mean: np.ndarray,
              x_scale: np.ndarray, y_mean: np.ndarray, y_scale: np.ndarray,
              device, context_steps: int, future_steps: int, seed: int,
              max_windows_per_run: int, *, stratify: bool = False,
              support_data: dict[str, Any] | None = None
              ) -> dict[str, Any]:
    windows = _windows_by_run(
        data, targets, groups, max_windows_per_run, context_steps,
        future_steps, seed)
    model.eval()
    per_run = {}
    regime_run_scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "speed_mps": defaultdict(lambda: defaultdict(dict)),
        "absolute_steering_rad": defaultdict(lambda: defaultdict(dict)),
        "wheel_body_mismatch": defaultdict(lambda: defaultdict(dict)),
    } if stratify else {}
    mismatch_thresholds = None
    race_domain = int(data.get("schema_version", 0)) >= 8
    race_domain_speed_run_scores: dict[str, dict[str, dict[str, float]]] = {
        domain: defaultdict(dict) for domain in RACE_SPEED_DOMAINS
    }
    if stratify:
        training_data = data if support_data is None else support_data
        if training_data["feature_names"] != data["feature_names"]:
            raise ValueError("support and evaluation feature layouts differ")
        training_mismatch = []
        for sequence_index, (start_value, end_value) in enumerate(
                training_data["bounds"]):
            run_index = int(training_data["seq_run"][sequence_index])
            if training_data["splits"][run_index] != "train":
                continue
            start, end = int(start_value), int(end_value)
            frame = training_data["frames"][start:end]
            local = np.abs(0.5 * (frame[:, 5] + frame[:, 6]) - frame[:, 0])
            training_mismatch.append(local[np.isfinite(local)])
        if not training_mismatch or not sum(map(len, training_mismatch)):
            raise ValueError("cannot define mismatch strata without train rows")
        mismatch_thresholds = np.quantile(
            np.concatenate(training_mismatch), (0.50, 0.90))
    speed_edges = (0.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.0)
    steering_edges = (0.0, 0.10, 0.20, 0.30, 0.40, 0.5240001)

    def interval_labels(values: np.ndarray, edges: tuple[float, ...],
                        suffix: str) -> np.ndarray:
        indices = np.searchsorted(np.asarray(edges), values, side="right") - 1
        labels = np.full(len(values), "outside", dtype="U32")
        valid = (indices >= 0) & (indices < len(edges) - 1)
        for index, bin_index in enumerate(indices):
            if valid[index]:
                labels[index] = (
                    f"{edges[bin_index]:g}-{edges[bin_index + 1]:g}{suffix}")
        labels[values == edges[-1]] = f"{edges[-2]:g}-{edges[-1]:g}{suffix}"
        return labels
    horizon_steps = sorted(set(step for step in (
        round(0.25 / DT_S), round(0.75 / DT_S), round(2.0 / DT_S),
        round(5.0 / DT_S), future_steps) if step <= future_steps))
    for run_id, run_windows in windows.items():
        batch_predictions, batch_targets, window_pose_errors = [], [], []
        window_initial_states = []
        initial_speed, initial_steering, initial_mismatch = [], [], []
        with torch.no_grad():
            for batch_start in range(0, len(run_windows), 16):
                local = run_windows[batch_start:batch_start + 16]
                context_array, command_array, target_array = [], [], []
                initial_states, initial_poses, ground_truth_poses = [], [], []
                for start, end in local:
                    future_start = start + context_steps
                    future_end = future_start + future_steps
                    context_array.append(data["frames"][start:future_start])
                    command_array.append(data["frames"][future_start:future_end, 7:9])
                    target_array.append(targets[future_start:future_end])
                    initial_states.append(targets[future_start - 1, :STATE_COUNT])
                    window_initial_states.append(
                        targets[future_start - 1, :STATE_COUNT])
                    initial_poses.append(data["simulator_pose_xyyaw"][future_start - 1])
                    ground_truth_poses.append(
                        data["simulator_pose_xyyaw"][future_start:future_end])
                    rigid_initial = data["simulator_rigid_state"][future_start - 1]
                    feature_initial = data["frames"][future_start - 1]
                    initial_speed.append(float(np.hypot(
                        rigid_initial[7], rigid_initial[8])))
                    if stratify:
                        initial_steering.append(abs(float(feature_initial[3])))
                        initial_mismatch.append(abs(float(
                            0.5 * (feature_initial[5] + feature_initial[6])
                            - feature_initial[0])))
                context = (np.stack(context_array) - x_mean) / x_scale
                commands = (np.stack(command_array) - x_mean[7:9]) / x_scale[7:9]
                context_t = torch.as_tensor(context, dtype=torch.float32,
                                            device=device)
                commands_t = torch.as_tensor(commands, dtype=torch.float32,
                                             device=device)
                prediction = model(context_t, commands_t).cpu().numpy()
                physical_prediction = prediction * y_scale + y_mean
                batch_predictions.append(physical_prediction)
                batch_targets.append(np.stack(target_array))
                for pred, initial, pose0, true_pose in zip(
                        physical_prediction, initial_states, initial_poses,
                        ground_truth_poses):
                    if not (np.isfinite(pose0).all()
                            and np.isfinite(true_pose).all()):
                        continue
                    pred_pose = _pose_rollout(pred, initial, pose0)
                    error = pred_pose - true_pose
                    error[:, 2] = np.arctan2(np.sin(error[:, 2]),
                                             np.cos(error[:, 2]))
                    window_pose_errors.append(error)
        pred_all = np.concatenate(batch_predictions)
        target_all = np.concatenate(batch_targets)
        state_error = pred_all[:, :, :STATE_COUNT] - target_all[:, :, :STATE_COUNT]
        norm_error = state_error / y_scale[:STATE_COUNT]
        initial_state_all = np.stack(window_initial_states)
        persistence_error = (
            initial_state_all[:, None, :] - target_all[:, :, :STATE_COUNT]
        ) / y_scale[:STATE_COUNT]
        run_regime_labels = {}
        run_speed_domain_labels = None
        if race_domain:
            run_speed_domain_labels = race_speed_domain_labels(
                np.asarray(initial_speed, dtype=np.float64))
        if stratify:
            speed_values = np.asarray(initial_speed)
            steering_values = np.asarray(initial_steering)
            mismatch_values = np.asarray(initial_mismatch)
            mismatch_low, mismatch_high = map(float, mismatch_thresholds)
            run_regime_labels = {
                "speed_mps": interval_labels(speed_values, speed_edges, "mps"),
                "absolute_steering_rad": interval_labels(
                    steering_values, steering_edges, "rad"),
                "wheel_body_mismatch": np.where(
                    mismatch_values <= mismatch_low, "low",
                    np.where(mismatch_values <= mismatch_high,
                             "moderate", "high")),
            }
        horizon_result = {}
        for step in horizon_steps:
            error_at_step = state_error[:, step - 1]
            normalized_at_step = norm_error[:, step - 1]
            metric = {
                "actual_seconds": float(step * DT_S),
                "state_rmse": {
                    name: float(value) for name, value in zip(
                        STATE_NAMES, np.sqrt(np.mean(error_at_step ** 2, axis=0)))},
                "normalized_body_state_rmse": float(np.sqrt(np.mean(
                    normalized_at_step[:, :3] ** 2))),
                "normalized_full_state_rmse": float(np.sqrt(np.mean(
                    normalized_at_step ** 2))),
                "persistence_normalized_body_state_rmse": float(np.sqrt(
                    np.mean(persistence_error[:, step - 1, :3] ** 2))),
            }
            metric["direct_to_persistence_rmse_ratio"] = (
                metric["normalized_body_state_rmse"]
                / max(metric["persistence_normalized_body_state_rmse"], 1e-8))
            if window_pose_errors:
                local_pose = np.stack(window_pose_errors)[:, :step]
                metric["position_xy_rmse_m"] = np.sqrt(np.mean(
                    local_pose[:, :, :2] ** 2, axis=(0, 1))).tolist()
                metric["heading_rmse_rad"] = float(np.sqrt(np.mean(
                    local_pose[:, :, 2] ** 2)))
                metric["pose_windows"] = int(len(window_pose_errors))
            horizon_result[f"{step * DT_S:g}s"] = metric
            if race_domain:
                horizon_key = f"{step * DT_S:g}s"
                run_name = str(data["run_ids"][run_id])
                for domain in RACE_SPEED_DOMAINS:
                    selected = run_speed_domain_labels == domain
                    if np.any(selected):
                        race_domain_speed_run_scores[domain][run_name][horizon_key] = float(
                            np.sqrt(np.mean(
                                norm_error[selected, step - 1, :3] ** 2)))
            if stratify:
                horizon_key = f"{step * DT_S:g}s"
                for axis, labels in run_regime_labels.items():
                    for label in np.unique(labels):
                        selected = labels == label
                        if label == "outside" or not np.any(selected):
                            continue
                        regime_score = float(np.sqrt(np.mean(
                            norm_error[selected, step - 1, :3] ** 2)))
                        regime_run_scores[axis][str(label)][
                            str(data["run_ids"][run_id])][horizon_key] = regime_score
        per_run[str(data["run_ids"][run_id])] = {
            "window_count": int(len(run_windows)),
            "horizons": horizon_result,
        }
    macro = {}
    for horizon in horizon_steps:
        key = f"{horizon * DT_S:g}s"
        values = [row["horizons"][key]["normalized_body_state_rmse"]
                  for row in per_run.values() if key in row["horizons"]]
        macro[key] = {
            "independent_run_count": len(values),
            "macro_run_mean_normalized_body_state_rmse": (
                float(np.mean(values)) if values else None),
            "run_min": float(np.min(values)) if values else None,
            "run_max": float(np.max(values)) if values else None,
        }
        persistence_values = [
            row["horizons"][key]["persistence_normalized_body_state_rmse"]
            for row in per_run.values() if key in row["horizons"]]
        macro[key]["macro_run_mean_persistence_normalized_body_state_rmse"] = (
            float(np.mean(persistence_values)) if persistence_values else None)
        macro[key]["direct_to_persistence_rmse_ratio"] = (
            float(np.mean(values) / max(np.mean(persistence_values), 1e-8))
            if values and persistence_values else None)
    by_regime = {}
    if stratify:
        rng = np.random.default_rng(seed + 271828)
        for axis, bins in regime_run_scores.items():
            by_regime[axis] = {}
            for label, runs in bins.items():
                by_regime[axis][label] = {}
                for horizon in sorted({
                        key for values in runs.values() for key in values}):
                    run_values = np.asarray([
                        values[horizon] for values in runs.values()
                        if horizon in values], dtype=np.float64)
                    if len(run_values) >= 2:
                        draws = rng.integers(
                            0, len(run_values), size=(1000, len(run_values)))
                        ci = np.quantile(np.mean(run_values[draws], axis=1),
                                         (0.025, 0.975)).tolist()
                    else:
                        ci = None
                    by_regime[axis][label][horizon] = {
                        "independent_run_count": int(len(run_values)),
                        "macro_run_mean_normalized_body_state_rmse": (
                            float(np.mean(run_values)) if len(run_values) else None),
                        "run_min": float(np.min(run_values)) if len(run_values) else None,
                        "run_max": float(np.max(run_values)) if len(run_values) else None,
                        "run_cluster_bootstrap_95pct_ci": ci,
                        "per_run": {
                            run_id: float(values[horizon])
                            for run_id, values in sorted(runs.items())
                            if horizon in values},
                    }
    race_domain_speed_macro = {}
    for domain, run_scores in race_domain_speed_run_scores.items():
        race_domain_speed_macro[domain] = {}
        for horizon in horizon_steps:
            key = f"{horizon * DT_S:g}s"
            values = np.asarray([scores[key] for scores in run_scores.values()
                                 if key in scores], dtype=np.float64)
            if len(values):
                rng = np.random.default_rng(seed + horizon + len(domain))
                draws = rng.integers(0, len(values), size=(1000, len(values)))
                interval = np.quantile(np.mean(values[draws], axis=1),
                                       (0.025, 0.975)).tolist()
            else:
                interval = None
            race_domain_speed_macro[domain][key] = {
                "independent_run_count": int(len(values)),
                "macro_run_mean_normalized_body_state_rmse": (
                    float(np.mean(values)) if len(values) else None),
                "run_cluster_bootstrap_95pct_ci": interval,
                "per_run": {
                    run_id: float(scores[key])
                    for run_id, scores in sorted(run_scores.items())
                    if key in scores},
            }
    primary_keys = ("0.25s", "0.75s", "2s")
    if race_domain:
        score_values = [
            race_domain_speed_macro[domain][key][
                "macro_run_mean_normalized_body_state_rmse"]
            for domain in RACE_SPEED_DOMAINS for key in primary_keys
            if key in race_domain_speed_macro[domain]
            and race_domain_speed_macro[domain][key]["independent_run_count"] >= 2
            and race_domain_speed_macro[domain][key][
                "macro_run_mean_normalized_body_state_rmse"] is not None]
        if len(score_values) != len(RACE_SPEED_DOMAINS) * len(primary_keys):
            checkpoint_score = None
        else:
            checkpoint_score = float(np.mean(score_values))
    else:
        score_values = [macro[key]["macro_run_mean_normalized_body_state_rmse"]
                        for key in ("0.25s", "0.75s", "2s", "5s")
                        if key in macro
                        and macro[key]["macro_run_mean_normalized_body_state_rmse"]
                        is not None]
        checkpoint_score = float(np.mean(score_values)) if score_values else None
    return {
        "eligible_run_count": len(groups),
        "eligible_sequence_count": sum(map(len, groups.values())),
        "window_count": sum(map(len, windows.values())),
        "macro_run_horizons": macro,
        "checkpoint_selection_score": checkpoint_score,
        "checkpoint_selection_basis": (
            "equal macro over core 0-9 and fast-boundary 9-12 m/s, each averaged across independent runs at 0.25/0.75/2 s; each cell requires at least two validation runs"
            if race_domain else "legacy macro run mean at available 0.25/0.75/2/5 s horizons"),
        "race_domain_speed_macro": race_domain_speed_macro,
        "per_run": per_run,
        "by_initial_regime": by_regime,
        "wheel_body_mismatch_train_thresholds_mps": (
            {"low_upper_p50": float(mismatch_thresholds[0]),
             "moderate_upper_p90": float(mismatch_thresholds[1])}
            if mismatch_thresholds is not None else None),
    }


def train(dataset_path: Path, output_dir: Path, *, device_name: str,
          context_seconds: float, future_seconds: float, width: int,
          layer_count: int, head_count: int, batch_size: int,
          max_steps: int, eval_every: int, patience: int,
          max_eval_windows_per_run: int, seed: int,
          score_test: bool) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch, nn = _torch()
    torch.set_num_threads(1)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    context_steps = round(context_seconds / DT_S)
    future_steps = round(future_seconds / DT_S)
    if context_steps < 1 or future_steps < 1:
        raise ValueError("context and future horizons must be positive")
    data = _load_dataset(dataset_path)
    targets = _targets(data)
    groups = {
        split: _eligible_sequences(data, targets, split, context_steps,
                                   future_steps)
        for split in ("train", "validation", "test", "final_test")
    }
    if not groups["train"] or not groups["validation"]:
        raise ValueError("need eligible whole-run train and validation sequences")
    family_sampler = build_sequence_sampler(data, groups["train"])
    x_mean, x_scale, y_mean, y_scale = _fit_normalizers(
        data, targets, groups["train"])
    Model = _torch_model(torch, nn, context_steps, future_steps,
                         width, layer_count, head_count, 0.05)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4,
                                  weight_decay=1.0e-4)
    rng = np.random.default_rng(seed)
    eval_rng_seed = seed + 91017
    best_score = math.inf
    best_step = 0
    no_improvement = 0
    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    steps_completed = 0
    for step in range(1, max_steps + 1):
        steps_completed = step
        context, commands, labels = _training_batch(
            data, targets, groups["train"], batch_size, context_steps,
            future_steps, rng, family_sampler)
        context = (context - x_mean) / x_scale
        commands = (commands - x_mean[7:9]) / x_scale[7:9]
        labels = (labels - y_mean) / y_scale
        context_t = torch.as_tensor(context, dtype=torch.float32, device=device)
        commands_t = torch.as_tensor(commands, dtype=torch.float32, device=device)
        labels_t = torch.as_tensor(labels, dtype=torch.float32, device=device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(context_t, commands_t)
        if int(data.get("schema_version", 0)) >= 8:
            speeds = np.hypot(labels[:, :, 0], labels[:, :, 1])
            speed_weights = race_speed_bin_weights(speeds)
            state_loss = weighted_smooth_l1(
                torch, prediction[:, :, :STATE_COUNT],
                labels_t[:, :, :STATE_COUNT], speed_weights, beta=0.5)
            acceleration_loss = weighted_smooth_l1(
                torch, prediction[:, :, STATE_COUNT:],
                labels_t[:, :, STATE_COUNT:], speed_weights, beta=0.5)
        else:
            state_loss = nn.functional.smooth_l1_loss(
                prediction[:, :, :STATE_COUNT], labels_t[:, :, :STATE_COUNT],
                beta=0.5)
            acceleration_loss = nn.functional.smooth_l1_loss(
                prediction[:, :, STATE_COUNT:], labels_t[:, :, STATE_COUNT:],
                beta=0.5)
        loss = state_loss + 0.10 * acceleration_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_every == 0 or step == max_steps:
            validation = _evaluate(
                torch, model, data, targets, groups["validation"],
                x_mean, x_scale, y_mean, y_scale, device, context_steps,
                future_steps, eval_rng_seed, max_eval_windows_per_run)
            score = validation["checkpoint_selection_score"]
            event = {
                "step": step,
                "training_loss": float(loss.detach().cpu()),
                "training_state_loss": float(state_loss.detach().cpu()),
                "training_acceleration_loss": float(acceleration_loss.detach().cpu()),
                "validation_score": score,
                "validation": validation,
            }
            history.append(event)
            print(json.dumps({
                "step": step,
                "loss": event["training_loss"],
                "validation_score": score,
            }), flush=True)
            if score is not None and score < best_score:
                best_score = score
                best_step = step
                no_improvement = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "x_mean": x_mean,
                    "x_scale": x_scale,
                    "y_mean": y_mean,
                    "y_scale": y_scale,
                    "metadata": {
                        "architecture": "direct_sequence_transformer",
                        "feature_names": data["feature_names"],
                        "target_state_names": list(STATE_NAMES),
                        "acceleration_target_names": list(ACCELERATION_NAMES),
                        "context_steps": context_steps,
                        "future_steps": future_steps,
                        "width": width,
                        "layer_count": layer_count,
                        "head_count": head_count,
                        "training_runs": [str(data["run_ids"][i])
                                          for i in sorted(groups["train"])],
                        "future_inputs": ["steering_command_rad",
                                          "throttle_command_norm"],
                        "future_measurements_or_truth_as_inputs": False,
                    },
                }, output_dir / "best_direct_sequence.pt")
            else:
                no_improvement += 1
            (output_dir / "validation_history.json").write_text(
                json.dumps(history, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            if no_improvement >= patience:
                break

    best_payload = torch.load(output_dir / "best_direct_sequence.pt",
                              map_location=device, weights_only=False)
    model.load_state_dict(best_payload["state_dict"])
    model.eval()
    final_validation = _evaluate(
        torch, model, data, targets, groups["validation"],
        x_mean, x_scale, y_mean, y_scale, device, context_steps,
        future_steps, eval_rng_seed, max_eval_windows_per_run,
        stratify=int(data.get("schema_version", 0)) >= 8)
    test_metrics = None
    if score_test and groups["test"]:
        test_metrics = _evaluate(
            torch, model, data, targets, groups["test"],
            x_mean, x_scale, y_mean, y_scale, device, context_steps,
            future_steps, eval_rng_seed + 1, max_eval_windows_per_run,
            stratify=int(data.get("schema_version", 0)) >= 8)
    report = {
        "schema_version": 1,
        "architecture": "direct_sequence_transformer",
        "dataset": str(dataset_path.resolve()),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else None),
        "fixed_dt_s": DT_S,
        "context_seconds": context_steps * DT_S,
        "future_seconds": future_steps * DT_S,
        "width": width,
        "layers": layer_count,
        "heads": head_count,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "steps_completed": steps_completed,
        "best_step": best_step,
        "best_validation_score": best_score,
        "optimizer_steps_per_second": steps_completed / max(
            time.perf_counter() - start_time, 1e-9),
        "eligible_runs_by_split": {
            split: [str(data["run_ids"][i]) for i in sorted(split_groups)]
            for split, split_groups in groups.items()},
        "eligible_sequences_by_split": {
            split: sum(map(len, split_groups.values()))
            for split, split_groups in groups.items()},
        "training_sampler": family_sampler.metadata,
        "validation": final_validation,
        "test_scored_once": bool(score_test),
        "test": test_metrics,
        "future_truth_or_sensors_used_as_inputs": False,
        "free_running_plant": False,
        "race_domain_loss_weighting": ({
            "speed_bins_mps": list(RACE_SPEED_BIN_EDGES_MPS),
            "method": "equal total contribution per populated true future-speed bin in each batch",
            "checkpoint_selection": "equal core 0-9 / fast boundary 9-12 macro over independent validation runs at 0.25, 0.75, and 2 seconds",
        } if int(data.get("schema_version", 0)) >= 8 else None),
        "checkpoint": str(output_dir / "best_direct_sequence.pt"),
        "history": history,
    }
    write_standard_artifacts(
        output_dir, dataset_path, data,
        {"architecture": "direct_sequence_transformer",
         "context_seconds": context_steps * DT_S,
         "future_seconds": future_steps * DT_S,
         "width": width, "layers": layer_count, "heads": head_count,
         "batch_size": batch_size, "max_steps": max_steps,
         "eval_every": eval_every, "patience": patience,
         "max_eval_windows_per_run": max_eval_windows_per_run,
         "training_sampler": family_sampler.metadata,
         "score_test": score_test}, seed, report,
        ("tools/vehicle_dynamics_learning/train_direct_sequence_teacher.py",
         "tools/vehicle_dynamics_learning/train_nssm.py",
         "tools/vehicle_dynamics_learning/experiment_artifacts.py"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--future-seconds", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-eval-windows-per-run", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--score-test", action="store_true",
                        help="evaluate the selected checkpoint on whole test runs once")
    args = parser.parse_args()
    train(
        args.dataset, args.output_dir, device_name=args.device,
        context_seconds=args.context_seconds,
        future_seconds=args.future_seconds, width=args.width,
        layer_count=args.layers, head_count=args.heads,
        batch_size=args.batch_size, max_steps=args.max_steps,
        eval_every=args.eval_every, patience=args.patience,
        max_eval_windows_per_run=args.max_eval_windows_per_run,
        seed=args.seed, score_test=args.score_test)
    print(f"wrote {args.output_dir / 'training_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
