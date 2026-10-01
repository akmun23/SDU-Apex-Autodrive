#!/usr/bin/env python3
"""Grouped whole-run comparison of structured nonlinear body models.

The experiment is offline-only.  It uses the existing oracle-plant dataset to
isolate the body transition while replaying measured actuator/wheel channels.
It does not alter production odometry, MPC, planner, simulator physics, or ROS
topic policy.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from train_nssm import (
    STATE_COUNT,
    _batch_from_windows,
    _fixed_eval_windows,
    _load_dataset,
    _model_type,
    _normalizers,
    _rollout,
    _sample_batch,
    _sequence_groups,
    _tensor_batch,
    _torch,
)
from structured_body_models import (
    BODY_CHANNELS,
    BODY_NAMES,
    BODY_HORIZON_SECONDS,
    BODY_HORIZONS,
    DT_REFERENCE_S,
    acceleration_statistics,
    fit_linear_acceleration_baseline,
    make_structured_model,
    rollout_structured_body,
    training_transition_rows,
)


SPEED_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, float("inf"))
STEERING_EDGES = (0.0, 0.15, 0.30, 0.42, float("inf"))
DEFAULT_ARCHITECTURES = ("gru", "accel_mlp", "latent2", "latent4",
                         "latent8", "linear_residual")
BODY_TEACHER_FORCE_CHANNELS = (3, 4, 5, 6)
ATTITUDE_INPUT_CHANNELS = (9, 10, 11, 12)
def _family_key(run_id: str) -> str:
    """Group related experiment runs without splitting speed variants apart."""
    value = re.sub(r"_20\d{6}(?:_\d+)?$", "", run_id)
    value = re.sub(r"_20\d{6}_(?:train|holdout|validation)\d*$", "", value)
    if value.startswith("openplane_rootless_"):
        tail = value[len("openplane_rootless_"):]
        if tail.startswith("isolated"):
            return "rootless_isolated"
        tail = re.sub(r"^[0-9]+(?:to[0-9]+)?_", "", tail)
        return "rootless_" + tail
    if value.startswith("openplane_full_input_excitation_"):
        return "full_input_excitation"
    if value.startswith("openplane_isolated_force_"):
        return "isolated_force"
    if value.startswith("openplane_speedhold_"):
        value = re.sub(r"_\d+(?:\.\d+)?mps.*$", "", value)
        return value
    if value.startswith("openplane_transition_"):
        value = re.sub(r"_\d+(?:\.\d+)?mps.*$", "", value)
        return value
    if value.startswith("openplane_highspeed_crossfactor_"):
        return "highspeed_crossfactor"
    return value


def _manifest_runs(path: Path) -> dict[str, dict[str, Any]]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is required beside {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(row["run_id"]): row for row in manifest.get("runs", [])}


def _run_rows(data: dict[str, Any], run_index: int) -> np.ndarray:
    rows = []
    for (start, end), seq_run in zip(data["bounds"], data["seq_run"]):
        if int(seq_run) == run_index:
            rows.append(data["frames"][int(start):int(end)])
    return np.concatenate(rows) if rows else np.empty((0, data["frames"].shape[1]))


def _coverage_profile(data: dict[str, Any], run_indices: list[int]
                      ) -> dict[str, Any]:
    frames = [_run_rows(data, run) for run in run_indices]
    frames = [rows for rows in frames if len(rows)]
    if not frames:
        return {"runs": 0, "samples": 0, "max_measured_speed_mps": None,
                "speed_bins": {}, "steering_bins": {},
                "speed_steering_cells": {}}
    samples = np.concatenate(frames)
    speed = np.hypot(samples[:, 0], samples[:, 1])
    steering = np.abs(samples[:, 3])
    speed_counts, _ = np.histogram(speed, bins=SPEED_EDGES)
    steer_counts, _ = np.histogram(steering, bins=STEERING_EDGES)
    joint, _, _ = np.histogram2d(speed, steering,
                                 bins=(SPEED_EDGES, STEERING_EDGES))
    speed_names = [
        (f"{SPEED_EDGES[i]:g}-{SPEED_EDGES[i + 1]:g}mps"
         if math.isfinite(SPEED_EDGES[i + 1])
         else f"{SPEED_EDGES[i]:g}+mps")
        for i in range(len(SPEED_EDGES) - 1)]
    steering_names = [
        (f"{STEERING_EDGES[i]:g}-{STEERING_EDGES[i + 1]:g}rad"
         if math.isfinite(STEERING_EDGES[i + 1])
         else f"{STEERING_EDGES[i]:g}+rad")
        for i in range(len(STEERING_EDGES) - 1)]
    cells = {}
    for i, speed_name in enumerate(speed_names):
        for j, steering_name in enumerate(steering_names):
            cells[f"{speed_name}__{steering_name}"] = int(joint[i, j])
    return {
        "runs": len(frames),
        "samples": int(len(samples)),
        "max_measured_speed_mps": float(speed.max()),
        "speed_bins": dict(zip(speed_names, map(int, speed_counts))),
        "steering_bins": dict(zip(steering_names, map(int, steer_counts))),
        "speed_steering_cells": cells,
    }


def _grouped_folds(
    data: dict[str, Any],
    eligible_runs: list[int],
    manifest: dict[str, dict[str, Any]],
    fold_count: int,
    seed: int,
) -> tuple[dict[int, int], dict[str, list[int]], dict[str, str]]:
    """Assign whole runs to stratified folds; exact duplicates stay grouped."""
    if fold_count < 2:
        raise ValueError("at least two folds are required")
    names = {int(i): str(data["run_ids"][i]) for i in eligible_runs}
    family_by_run = {i: _family_key(name) for i, name in names.items()}
    group_by_run = {}
    for run, name in names.items():
        fingerprint = str(manifest.get(name, {}).get("fingerprint", ""))
        # Experiment family is used for stratification, not held out as a
        # monolith; this preserves speed coverage in every fold. Exact capture
        # duplicates are still indivisible across training and evaluation.
        group_by_run[run] = (f"duplicate:{fingerprint}" if fingerprint
                             else f"run:{name}")

    speed_edges = np.asarray(SPEED_EDGES, dtype=np.float64)
    steer_edges = np.asarray(STEERING_EDGES, dtype=np.float64)
    family_names = sorted(set(family_by_run.values()))
    family_index = {name: index for index, name in enumerate(family_names)}
    base_profile_size = 1 + (len(speed_edges) - 1) * (len(steer_edges) - 1)
    group_profiles: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(base_profile_size + len(family_names), dtype=np.float64))
    group_runs: dict[str, list[int]] = defaultdict(list)
    for run in eligible_runs:
        group = group_by_run[run]
        frames = _run_rows(data, run)
        if len(frames):
            speed = np.hypot(frames[:, 0], frames[:, 1])
            steering = np.abs(frames[:, 3])
            histogram, _, _ = np.histogram2d(speed, steering,
                                             bins=(speed_edges, steer_edges))
            profile = np.zeros(base_profile_size + len(family_names),
                               dtype=np.float64)
            profile[0] = len(frames)
            profile[1:base_profile_size] = histogram.ravel()
            profile[base_profile_size + family_index[family_by_run[run]]] = 1.0
            group_profiles[group] += profile
        else:
            group_profiles[group][base_profile_size +
                                   family_index[family_by_run[run]]] += 1.0
        group_runs[group].append(run)

    if len(group_runs) < fold_count:
        raise ValueError(
            f"only {len(group_runs)} independent run/duplicate groups for "
            f"{fold_count} folds")

    groups = sorted(group_runs)
    rng = np.random.default_rng(seed)
    tie_break = {group: float(rng.random()) for group in groups}
    groups.sort(key=lambda group: (-group_profiles[group][0], tie_break[group]))
    totals = np.sum([group_profiles[group] for group in groups], axis=0)
    target = totals / fold_count
    scale = np.maximum(target, 1.0)
    assigned_profiles = np.zeros((fold_count, len(target)), dtype=np.float64)
    fold_runs = np.zeros(fold_count, dtype=np.float64)
    fold_by_group: dict[str, int] = {}
    for group in groups:
        scores = []
        for fold in range(fold_count):
            candidate = assigned_profiles.copy()
            candidate[fold] += group_profiles[group]
            cost = np.sum(((candidate - target[None, :]) / scale[None, :]) ** 2)
            run_target = len(eligible_runs) / fold_count
            candidate_run_counts = fold_runs.copy()
            candidate_run_counts[fold] += len(group_runs[group])
            run_cost = np.sum(((candidate_run_counts - run_target) /
                               max(run_target, 1.0)) ** 2)
            scores.append(float(cost + 0.25 * run_cost))
        best = int(np.argmin(np.asarray(scores) + rng.random(fold_count) * 1e-10))
        fold_by_group[group] = best
        assigned_profiles[best] += group_profiles[group]
        fold_runs[best] += len(group_runs[group])

    run_fold = {run: fold_by_group[group_by_run[run]] for run in eligible_runs}
    family_folds = {
        family: sorted({run_fold[run] for run in eligible_runs
                        if family_by_run[run] == family})
        for family in family_names
    }
    return run_fold, family_folds, {names[run]: family_by_run[run]
                                    for run in eligible_runs}


def _yaw_parameters(config_path: Path) -> dict[str, float]:
    text = config_path.read_text(encoding="utf-8")
    keys = (
        "yaw_rate_response_time_constant_s",
        "yaw_rate_steering_gain_per_m",
        "yaw_rate_steering_gain_reduction_per_rad",
        "yaw_rate_steering_gain_start_rad",
        "yaw_rate_steering_gain_end_rad",
        "yaw_rate_low_speed_response_time_constant_s",
        "yaw_rate_low_speed_transition_speed_mps",
    )
    values: dict[str, float] = {}
    for key in keys:
        match = re.search(rf"^\s*{re.escape(key)}:\s*([-+0-9.eE]+)",
                          text, re.MULTILINE)
        if not match:
            raise ValueError(f"missing {key} in {config_path}")
        values[key] = float(match.group(1))
    return values


def _mpc_yaw_next(current: np.ndarray, dt_s: np.ndarray,
                  parameters: dict[str, float]) -> np.ndarray:
    steering = current[..., 3]
    magnitude = np.abs(steering)
    gain = parameters["yaw_rate_steering_gain_per_m"] - (
        parameters["yaw_rate_steering_gain_reduction_per_rad"] *
        np.clip(magnitude - parameters["yaw_rate_steering_gain_start_rad"],
                0.0,
                parameters["yaw_rate_steering_gain_end_rad"] -
                parameters["yaw_rate_steering_gain_start_rad"]))
    low_tau = parameters["yaw_rate_low_speed_response_time_constant_s"]
    base_tau = parameters["yaw_rate_response_time_constant_s"]
    transition_speed = parameters["yaw_rate_low_speed_transition_speed_mps"]
    if low_tau > 0.0:
        speed = np.maximum(current[..., 0], 0.0)
        tau = base_tau + (low_tau - base_tau) * np.exp(-speed / transition_speed)
    else:
        tau = np.full_like(dt_s, base_tau, dtype=np.float64)
    retention = np.exp(-dt_s / tau)
    steady = current[..., 0] * np.tan(steering) * gain
    return retention * current[..., 2] + (1.0 - retention) * steady


def _fold_regime_thresholds(
    train_features: np.ndarray,
    acceleration_targets: np.ndarray,
    dt_s: np.ndarray,
    yaw_parameters: dict[str, float],
) -> dict[str, float]:
    mismatch = np.abs(0.5 * (train_features[:, 5] + train_features[:, 6]) -
                      train_features[:, 0])
    next_yaw = train_features[:, 2] + dt_s * acceleration_targets[:, 2]
    yaw_residual = np.abs(next_yaw - _mpc_yaw_next(
        train_features, dt_s, yaw_parameters))
    return {
        "encoder_body_mismatch_q90_mps": float(np.quantile(mismatch, 0.90)),
        "mpc_yaw_one_step_residual_q90_radps": float(
            max(np.quantile(yaw_residual, 0.90), 0.05)),
        "source": "outer-training-run samples only",
    }


def _window_regimes(history: np.ndarray, future: np.ndarray, dt_s: np.ndarray,
                    thresholds: dict[str, float],
                    yaw_parameters: dict[str, float]
                    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    current = history[:, -1, :]
    body_sequence = np.concatenate((current[:, None, :3], future[:, :, :3]),
                                   axis=1)
    speed_sequence = np.linalg.norm(body_sequence[:, :, :2], axis=2)
    steering_sequence = np.concatenate((current[:, None, 3], future[:, :, 3]),
                                       axis=1)
    mismatch_sequence = np.abs(
        0.5 * (np.concatenate((current[:, None, 5], future[:, :, 5]), axis=1) +
               np.concatenate((current[:, None, 6], future[:, :, 6]), axis=1)) -
        body_sequence[:, :, 0])
    initial_throttle = current[:, 8]
    command_sequence = np.concatenate((initial_throttle[:, None],
                                       future[:, :, 8]), axis=1)
    max_steering = np.max(np.abs(steering_sequence), axis=1)
    min_command = np.min(command_sequence, axis=1)
    throttle_cut = initial_throttle - min_command >= 0.20
    reversal = np.zeros(len(history), dtype=bool)
    for index in range(steering_sequence.shape[1] - 1):
        left, right = steering_sequence[:, index], steering_sequence[:, index + 1]
        reversal |= ((left * right < 0.0) & (np.abs(left) >= 0.08) &
                     (np.abs(right) >= 0.08))

    actual_current = np.concatenate(
        (current[:, None, :], future[:, :-1, :]), axis=1)
    yaw_prediction = _mpc_yaw_next(actual_current, dt_s, yaw_parameters)
    yaw_residual = np.abs(future[:, :, 2] - yaw_prediction)
    max_yaw_residual = np.max(yaw_residual, axis=1)
    speed_at_horizon = speed_sequence[:, -1]
    max_speed = np.max(speed_sequence, axis=1)
    max_mismatch = np.max(mismatch_sequence, axis=1)
    high_yaw = (max_yaw_residual >=
                thresholds["mpc_yaw_one_step_residual_q90_radps"])
    high_mismatch = (max_mismatch >=
                     thresholds["encoder_body_mismatch_q90_mps"])
    masks = {
        "normal_driving": ((speed_sequence[:, 0] >= 2.0) &
                           (speed_sequence[:, 0] < 7.0) &
                           (max_steering < 0.30) & ~high_mismatch &
                           ~throttle_cut & ~high_yaw),
        "high_steering": max_steering >= 0.42,
        "high_speed": max_speed >= 6.0,
        "large_encoder_body_mismatch": high_mismatch,
        "throttle_cut": throttle_cut,
        "braking": min_command < -0.05,
        "steering_reversal": reversal,
        "nonlinear_yaw_transition": high_yaw,
    }
    return masks, speed_at_horizon, max_steering


def _metric(errors: np.ndarray, normalized_scale: np.ndarray) -> dict[str, Any]:
    if not len(errors):
        return {"windows": 0, "rmse": None, "bias": None,
                "p95_absolute_error": None, "normalized_rmse": None}
    rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    bias = np.mean(errors, axis=0)
    p95 = np.quantile(np.abs(errors), 0.95, axis=0)
    normalized = errors / normalized_scale[None, :]
    return {
        "windows": int(len(errors)),
        "rmse": dict(zip(BODY_NAMES, map(float, rmse))),
        "bias": dict(zip(BODY_NAMES, map(float, bias))),
        "p95_absolute_error": dict(zip(BODY_NAMES, map(float, p95))),
        "normalized_rmse": float(np.sqrt(np.mean(normalized ** 2))),
    }


def _horizon_windows(data: dict[str, Any], run: int,
                     history_steps: int, horizon: int,
                     maximum_windows: int) -> list[tuple[int, int]]:
    sequence_ids = [seq for seq, ((start, end), run_id) in enumerate(
        zip(data["bounds"], data["seq_run"]))
        if int(run_id) == run and int(end - start) >= history_steps + horizon + 1]
    if not sequence_ids:
        return []
    return _fixed_eval_windows(data, {run: sequence_ids}, history_steps,
                               horizon, maximum_windows)


def _run_window_sets(data: dict[str, Any], run: int, history_steps: int,
                     maximum_windows: int,
                     horizons: tuple[int, ...] = BODY_HORIZONS
                     ) -> dict[int, list[tuple[int, int]]]:
    return {horizon: _horizon_windows(data, run, history_steps, horizon,
                                      maximum_windows)
            for horizon in horizons}


def _score_run(
    torch,
    model,
    legacy: bool,
    data: dict[str, Any],
    run: int,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    thresholds: dict[str, float],
    yaw_parameters: dict[str, float],
    history_steps: int,
    max_windows: int,
    device: str,
    batch_size: int,
    windows_by_horizon: dict[int, list[tuple[int, int]]] | None = None,
    horizons: tuple[int, ...] = BODY_HORIZONS,
    held_input_channels: tuple[int, ...] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        seconds = BODY_HORIZON_SECONDS[BODY_HORIZONS.index(horizon)]
        windows = (windows_by_horizon[horizon] if windows_by_horizon is not None
                   else _horizon_windows(data, run, history_steps, horizon,
                                         max_windows))
        if not windows:
            result[str(horizon)] = {"seconds_nominal": seconds,
                                    "windows": 0, "all": _metric(
                                        np.empty((0, 3)), feature_scale[:3]),
                                    "regimes": {}, "speed_steering_cells": {}}
            continue
        error_rows: list[np.ndarray] = []
        history_rows: list[np.ndarray] = []
        future_rows: list[np.ndarray] = []
        dt_rows: list[np.ndarray] = []
        actual_durations: list[np.ndarray] = []
        for offset in range(0, len(windows), batch_size):
            rows = windows[offset:offset + batch_size]
            arrays = _batch_from_windows(data, rows, history_steps, horizon)
            raw_history, raw_future, raw_dt = arrays
            history, future, dts = _tensor_batch(
                torch, arrays, feature_mean, feature_scale, device)
            with torch.no_grad():
                if legacy:
                    prediction = _rollout(
                        model, history, future, dts, history_steps,
                        teacher_force_channels=BODY_TEACHER_FORCE_CHANNELS,
                        held_input_channels=held_input_channels,
                    )[:, :, :3]
                else:
                    prediction = rollout_structured_body(model, history,
                                                         future, dts,
                                                         held_input_channels)
            prediction_raw = (prediction.cpu().numpy() * feature_scale[None, None, :3]
                              + feature_mean[None, None, :3])
            error_rows.append(prediction_raw[:, -1, :] - raw_future[:, -1, :3])
            history_rows.append(raw_history)
            future_rows.append(raw_future)
            dt_rows.append(raw_dt)
            actual_durations.append(np.sum(raw_dt, axis=1))
        errors = np.concatenate(error_rows)
        histories = np.concatenate(history_rows)
        futures = np.concatenate(future_rows)
        intervals = np.concatenate(dt_rows)
        masks, speed_at_horizon, max_steering = _window_regimes(
            histories, futures, intervals, thresholds, yaw_parameters)
        horizon_result = {
            "seconds_nominal": seconds,
            "seconds_median_actual": float(np.median(
                np.concatenate(actual_durations))),
            "windows": int(len(errors)),
            "all": _metric(errors, feature_scale[:3]),
            "regimes": {},
            "speed_steering_cells": {},
        }
        for regime, selected in masks.items():
            if np.any(selected):
                horizon_result["regimes"][regime] = _metric(
                    errors[selected], feature_scale[:3])
        speed_bin = np.digitize(speed_at_horizon,
                                np.asarray(SPEED_EDGES[1:-1]), right=False)
        steer_bin = np.digitize(max_steering,
                                np.asarray(STEERING_EDGES[1:-1]), right=False)
        for speed_index in range(len(SPEED_EDGES) - 1):
            for steer_index in range(len(STEERING_EDGES) - 1):
                selected = (speed_bin == speed_index) & (steer_bin == steer_index)
                if np.any(selected):
                    speed_name = (
                        f"{SPEED_EDGES[speed_index]:g}-"
                        f"{SPEED_EDGES[speed_index + 1]:g}mps"
                        if math.isfinite(SPEED_EDGES[speed_index + 1])
                        else f"{SPEED_EDGES[speed_index]:g}+mps")
                    steer_name = (
                        f"{STEERING_EDGES[steer_index]:g}-"
                        f"{STEERING_EDGES[steer_index + 1]:g}rad"
                        if math.isfinite(STEERING_EDGES[steer_index + 1])
                        else f"{STEERING_EDGES[steer_index]:g}+rad")
                    horizon_result["speed_steering_cells"][
                        f"{speed_name}__{steer_name}"] = _metric(
                            errors[selected], feature_scale[:3])
        result[str(horizon)] = horizon_result
    return result


def _macro_score(run_results: dict[str, Any], horizon: int) -> float | None:
    values = []
    for result in run_results.values():
        metric = result.get(str(horizon), {}).get("all", {})
        if metric.get("windows", 0) > 0:
            values.append(metric["normalized_rmse"])
    return float(np.mean(values)) if values else None


def _region_metric(record: dict[str, Any], region: str) -> dict[str, Any]:
    if region == "all":
        return record.get("all", {})
    if region.startswith("regime:"):
        return record.get("regimes", {}).get(region.removeprefix("regime:"), {})
    if region.startswith("cell:"):
        return record.get("speed_steering_cells", {}).get(
            region.removeprefix("cell:"), {})
    return {}


def _save_checkpoint(torch, path: Path, model, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata},
               temporary)
    temporary.replace(path)


def _make_model(torch, nn, architecture: str, feature_count: int,
                feature_mean: np.ndarray, feature_scale: np.ndarray,
                acceleration_mean: np.ndarray,
                acceleration_scale: np.ndarray,
                linear_coefficients: np.ndarray,
                hidden_size: int):
    if architecture == "gru":
        return _model_type(torch, nn, 128, "gru", 1, 16,
                           feature_count)()
    latent_dim = int(architecture.removeprefix("latent")) if architecture.startswith("latent") else 0
    structured_arch = ("latent" if latent_dim else architecture)
    return make_structured_model(
        torch, nn, structured_arch, latent_dim, feature_count, hidden_size,
        feature_mean, feature_scale, acceleration_mean, acceleration_scale,
        linear_coefficients if architecture == "linear_residual" else None)


def _train_loss(torch, nn, model, architecture: str, history, future, dts,
                history_steps: int,
                held_input_channels: tuple[int, ...] = ()):
    if architecture == "gru":
        # Match the structured candidates' offline system-identification
        # protocol: measured actuator feedback and wheel speeds are exogenous
        # at every step, and only body transition quality is optimized.
        prediction = _rollout(
            model, history, future, dts, history_steps,
            teacher_force_channels=BODY_TEACHER_FORCE_CHANNELS,
            held_input_channels=held_input_channels)
        return nn.functional.smooth_l1_loss(
            prediction[:, :, :BODY_CHANNELS],
            future[:, :, :BODY_CHANNELS], beta=0.05)
    prediction = rollout_structured_body(model, history, future, dts,
                                         held_input_channels)
    target = future[:, :, :BODY_CHANNELS]
    horizon_indices = [index - 1 for index in BODY_HORIZONS
                       if index <= prediction.shape[1]]
    horizon_loss = torch.stack([
        nn.functional.smooth_l1_loss(prediction[:, index], target[:, index],
                                     beta=0.05)
        for index in horizon_indices]).mean()
    trajectory_loss = nn.functional.smooth_l1_loss(prediction, target, beta=0.05)
    return 0.75 * horizon_loss + 0.25 * trajectory_loss


def _curriculum_length(step: int, step_budget: int) -> int:
    fraction = step / max(step_budget, 1)
    if fraction <= 0.15:
        return 5
    if fraction <= 0.35:
        return 10
    if fraction <= 0.60:
        return 20
    return 32


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _train_candidate(
    torch,
    nn,
    architecture: str,
    data: dict[str, Any],
    train_groups: dict[int, list[int]],
    validation_groups: dict[int, list[int]],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    acceleration_mean: np.ndarray,
    acceleration_scale: np.ndarray,
    linear_coefficients: np.ndarray,
    args: argparse.Namespace,
    fold: int,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    seed = args.seed + fold * 10007 + DEFAULT_ARCHITECTURES.index(architecture) * 101
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np_rng = np.random.default_rng(seed)
    model = _make_model(
        torch, nn, architecture, data["frames"].shape[1], feature_mean,
        feature_scale, acceleration_mean, acceleration_scale,
        linear_coefficients, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    validation_runs = sorted(validation_groups)
    checkpoint = output_dir / f"{architecture}.pt"
    validation_windows = {
        run: _run_window_sets(data, run, args.history_steps,
                              args.validation_windows_per_run, (30,))[30]
        for run in validation_runs
    }
    validation_windows = {run: windows for run, windows in validation_windows.items()
                          if windows}
    if not validation_windows:
        raise ValueError(f"fold {fold} has no 750 ms inner-validation windows")
    restart_this_model = f"{fold}:{architecture}" in set(args.retrain_model)
    if checkpoint.is_file() and args.resume and not restart_this_model:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        metadata = payload.get("metadata", {})
        expected_training_ids = [str(data["run_ids"][i])
                                 for i in sorted(train_groups)]
        expected_validation_ids = [
            str(data["run_ids"][i]) for i in validation_windows]
        if (metadata.get("architecture") != architecture
                or metadata.get("seed") != seed
                or metadata.get("history_steps") != args.history_steps
                or metadata.get("rollout_steps") != args.rollout_steps
                or metadata.get("validation_windows_per_run",
                                args.validation_windows_per_run)
                != args.validation_windows_per_run
                or metadata.get("steps_per_model", args.steps_per_model)
                != args.steps_per_model
                or metadata.get("batch_size", args.batch_size) != args.batch_size
                or metadata.get("hidden_size", 128 if architecture == "gru"
                                else args.hidden_size)
                != (128 if architecture == "gru" else args.hidden_size)
                or metadata.get("training_run_ids") != expected_training_ids
                or metadata.get("inner_validation_run_ids")
                != expected_validation_ids
                or metadata.get("feature_names") != data["feature_names"]):
            raise ValueError(
                f"resume checkpoint metadata does not match this fold/model: "
                f"{checkpoint}")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        validation_result = {
            str(data["run_ids"][run]): _score_run(
                torch, model, architecture == "gru", data, run,
                feature_mean, feature_scale, args.regime_thresholds,
                args.yaw_parameters, args.history_steps,
                args.validation_windows_per_run, device,
                args.inference_batch_size, {30: validation_windows[run]}, (30,),
                args.held_input_channels)
            for run in validation_windows}
        resumed_score = _macro_score(validation_result, 30)
        if resumed_score is None:
            raise ValueError(f"resume checkpoint has no validation score: {checkpoint}")
        print(f"fold={fold + 1}/{args.folds} model={architecture} "
              f"reused_checkpoint_step={metadata.get('step')} "
              f"val_macro_750ms={resumed_score:.5f}", flush=True)
        return {
            "model": model,
            "checkpoint": checkpoint.name,
            "seed": seed,
            "steps_completed": int(metadata.get("steps_completed",
                                                metadata.get("step", 0))),
            "best_step": int(metadata.get("step", 0)),
            "best_inner_validation_macro_750ms_normalized_rmse": resumed_score,
            "last_training_loss": metadata.get("last_training_loss"),
            "validation_history": metadata.get("validation_history", []),
            "resumed_from_checkpoint": True,
        }
    best_score = math.inf
    best_step = 0
    stale = 0
    history: list[dict[str, Any]] = []
    last_loss = float("nan")
    for step in range(1, args.steps_per_model + 1):
        rollout_steps = _curriculum_length(step, args.steps_per_model)
        arrays = _sample_batch(data, train_groups, args.batch_size,
                               args.history_steps, rollout_steps, np_rng)
        hist, future, dts = _tensor_batch(
            torch, arrays, feature_mean, feature_scale, device)
        model.train()
        loss = _train_loss(torch, nn, model, architecture, hist, future, dts,
                           args.history_steps, args.held_input_channels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step % args.eval_every != 0 and step != 1:
            continue
        model.eval()
        validation_result = {}
        for run in validation_windows:
            validation_result[str(data["run_ids"][run])] = _score_run(
                torch, model, architecture == "gru", data, run,
                feature_mean, feature_scale,
                args.regime_thresholds, args.yaw_parameters,
                args.history_steps, args.validation_windows_per_run,
                device, args.inference_batch_size,
                {30: validation_windows[run]}, (30,),
                args.held_input_channels)
        validation_score = _macro_score(validation_result, 30)
        if validation_score is None:
            raise ValueError("inner validation has no valid 750 ms score")
        history.append({"step": step, "rollout_steps": rollout_steps,
                        "train_loss": last_loss,
                        "macro_run_balanced_750ms_normalized_rmse": validation_score})
        print(f"fold={fold + 1}/{args.folds} model={architecture} step={step} "
              f"rollout={rollout_steps} train={last_loss:.5f} "
              f"val_macro_750ms={validation_score:.5f}", flush=True)
        if validation_score < best_score:
            best_score = validation_score
            best_step = step
            stale = 0
            _save_checkpoint(torch, checkpoint, model, {
                "architecture": architecture,
                "latent_dim": int(architecture.removeprefix("latent"))
                    if architecture.startswith("latent") else 0,
                "seed": seed,
                "step": step,
                "steps_completed": step,
                "hidden_size": (128 if architecture == "gru"
                                else args.hidden_size),
                "best_inner_validation_macro_750ms_normalized_rmse": best_score,
                "last_training_loss": last_loss,
                "validation_history": history.copy(),
                "history_steps": args.history_steps,
                "rollout_steps": args.rollout_steps,
                "steps_per_model": args.steps_per_model,
                "batch_size": args.batch_size,
                "validation_windows_per_run": args.validation_windows_per_run,
                "feature_names": data["feature_names"],
                "feature_mean": feature_mean.tolist(),
                "feature_scale": feature_scale.tolist(),
                "acceleration_mean": acceleration_mean.tolist(),
                "acceleration_scale": acceleration_scale.tolist(),
                "training_run_ids": [str(data["run_ids"][i])
                                     for i in sorted(train_groups)],
                "inner_validation_run_ids": [
                    str(data["run_ids"][i]) for i in validation_windows],
            })
        else:
            stale += 1
            if stale >= args.patience:
                break
    if not checkpoint.is_file():
        raise RuntimeError(f"no checkpoint was selected for {architecture}")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return {
        "model": model,
        "checkpoint": checkpoint.name,
        "seed": seed,
        "steps_completed": step,
        "best_step": best_step,
        "best_inner_validation_macro_750ms_normalized_rmse": best_score,
        "last_training_loss": last_loss,
        "validation_history": history,
        "resumed_from_checkpoint": False,
    }


def _fold_data(data: dict[str, Any], train_runs: list[int],
               validation_runs: list[int], test_runs: list[int]
               ) -> dict[str, Any]:
    copied = dict(data)
    splits = np.full(data["splits"].shape, "unused", dtype=data["splits"].dtype)
    splits[train_runs] = "train"
    splits[validation_runs] = "validation"
    splits[test_runs] = "cv_test"
    copied["splits"] = splits
    return copied


def _aggregate_cv(results: dict[str, dict[str, Any]],
                  bootstrap_count: int, seed: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for architecture, run_results in results.items():
        horizon_rows: dict[str, Any] = {}
        for horizon in BODY_HORIZONS:
            regions = {"all"}
            for result in run_results.values():
                record = result.get(str(horizon), {})
                regions.update(f"regime:{name}"
                               for name in record.get("regimes", {}))
                regions.update(f"cell:{name}"
                               for name in record.get("speed_steering_cells", {}))
            region_rows = {}
            for region in sorted(regions):
                channel_rows = {}
                for channel in BODY_NAMES:
                    per_run = []
                    for run_id, result in run_results.items():
                        record = result.get(str(horizon), {})
                        metric = _region_metric(record, region)
                        if metric.get("windows", 0) >= 5:
                            per_run.append((run_id, metric["rmse"][channel],
                                            int(metric["windows"])))
                    if not per_run:
                        channel_rows[channel] = {"independent_runs": 0,
                                                 "macro_run_rmse": None,
                                                 "median_run_rmse": None,
                                                 "worst_run": None,
                                                 "bootstrap_95pct_ci": None}
                        continue
                    values = np.asarray([row[1] for row in per_run],
                                        dtype=np.float64)
                    rng = np.random.default_rng(seed + horizon + len(channel))
                    if len(values) >= 2:
                        indices = rng.integers(0, len(values),
                                               size=(bootstrap_count, len(values)))
                        samples = values[indices].mean(axis=1)
                        interval = np.quantile(samples, [0.025, 0.975]).tolist()
                    else:
                        interval = None
                    worst = max(per_run, key=lambda row: row[1])
                    channel_rows[channel] = {
                        "independent_runs": len(per_run),
                        "windows_across_runs": int(sum(row[2] for row in per_run)),
                        "macro_run_rmse": float(np.mean(values)),
                        "median_run_rmse": float(np.median(values)),
                        "worst_run": {"run_id": worst[0], "rmse": float(worst[1]),
                                      "windows": worst[2]},
                        "bootstrap_95pct_ci": interval,
                    }
                region_rows[region] = channel_rows
            horizon_rows[str(horizon)] = {"seconds_nominal": BODY_HORIZON_SECONDS[
                BODY_HORIZONS.index(horizon)], "regions": region_rows}
        output[architecture] = horizon_rows
    return output


def _paired_comparisons(results: dict[str, dict[str, Any]],
                        bootstrap_count: int, seed: int) -> dict[str, Any]:
    baseline = results.get("gru", {})
    comparisons: dict[str, Any] = {}
    for architecture, run_results in results.items():
        if architecture == "gru":
            continue
        horizons = {}
        for horizon in BODY_HORIZONS:
            by_region = {}
            regions = {"all"}
            for result in run_results.values():
                record = result.get(str(horizon), {})
                regions.update(f"regime:{name}"
                               for name in record.get("regimes", {}))
                regions.update(f"cell:{name}"
                               for name in record.get("speed_steering_cells", {}))
            for region in sorted(regions):
                for channel in BODY_NAMES:
                    paired = []
                    for run_id, candidate_result in run_results.items():
                        base_result = baseline.get(run_id)
                        if base_result is None:
                            continue
                        candidate_record = candidate_result.get(str(horizon), {})
                        base_record = base_result.get(str(horizon), {})
                        cm = _region_metric(candidate_record, region)
                        bm = _region_metric(base_record, region)
                        if cm.get("windows", 0) < 5 or bm.get("windows", 0) < 5:
                            continue
                        paired.append((run_id, float(cm["rmse"][channel]),
                                       float(bm["rmse"][channel])))
                    if not paired:
                        by_region.setdefault(region, {})[channel] = {
                            "paired_runs": 0, "relative_rmse_change": None,
                            "bootstrap_95pct_ci": None}
                        continue
                    cand = np.asarray([row[1] for row in paired])
                    base = np.asarray([row[2] for row in paired])
                    change = float(cand.mean() / max(base.mean(), 1e-9) - 1.0)
                    rng = np.random.default_rng(seed + horizon + len(architecture) + len(channel))
                    if len(paired) >= 2:
                        indices = rng.integers(0, len(paired),
                                               size=(bootstrap_count, len(paired)))
                        boot = cand[indices].mean(axis=1) / np.maximum(
                            base[indices].mean(axis=1), 1e-9) - 1.0
                        interval = np.quantile(boot, [0.025, 0.975]).tolist()
                    else:
                        interval = None
                    by_region.setdefault(region, {})[channel] = {
                        "paired_runs": len(paired),
                        "relative_rmse_change": change,
                        "bootstrap_95pct_ci": interval,
                    }
            horizons[str(horizon)] = by_region
        comparisons[architecture] = horizons
    return comparisons


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch, nn = _torch()
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    for value in args.retrain_model:
        try:
            fold_text, architecture = value.split(":", maxsplit=1)
            fold_index = int(fold_text)
        except ValueError as exc:
            raise ValueError(
                f"invalid --retrain-model {value!r}; expected FOLD:ARCH") from exc
        if (fold_index < 0 or fold_index >= args.folds
                or architecture not in args.architectures):
            raise ValueError(
                f"--retrain-model {value!r} is outside the selected CV run")
    data = _load_dataset(args.dataset)
    excluded_incomplete_attitude_sequences = 0
    if data["imu_attitude_valid"] is not None:
        complete = np.asarray([
            bool(np.all(data["imu_attitude_valid"][int(start):int(end)]))
            for start, end in data["bounds"]
        ], dtype=bool)
        excluded_incomplete_attitude_sequences = int(np.count_nonzero(~complete))
        if excluded_incomplete_attitude_sequences:
            data["bounds"] = data["bounds"][complete]
            data["seq_run"] = data["seq_run"][complete]
    held_input_channels: tuple[int, ...] = ()
    if args.condition_on_attitude:
        attitude = data["imu_attitude_frames"]
        valid = data["imu_attitude_valid"]
        if attitude is None or valid is None:
            raise ValueError("--condition-on-attitude requires schema-3 IMU attitude data")
        eligible_rows = np.zeros(len(data["frames"]), dtype=bool)
        for start, end in data["bounds"]:
            eligible_rows[int(start):int(end)] = True
        if not np.all(valid[eligible_rows]):
            raise ValueError("attitude-conditioned CV requires aligned roll/pitch and gyro on every selected sample")
        data["frames"] = np.column_stack((data["frames"], attitude)).astype(
            np.float32, copy=False)
        data["feature_names"].extend(data["imu_attitude_feature_names"])
        held_input_channels = ATTITUDE_INPUT_CHANNELS
    args.held_input_channels = held_input_channels
    manifest = _manifest_runs(args.dataset)
    runs_with_sequences = set(map(int, data["seq_run"]))
    eligible_runs = [int(i) for i, split in enumerate(data["splits"])
                     if split == "train" and int(i) in runs_with_sequences]
    if not eligible_runs:
        raise ValueError("dataset contains no original training runs")
    run_fold, family_fold, family_map = _grouped_folds(
        data, eligible_runs, manifest, args.folds, args.fold_seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but this PyTorch build has no CUDA support")
    if args.output_dir.exists() and not args.resume:
        raise ValueError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    args.yaw_parameters = _yaw_parameters(args.mpc_config)

    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "experiment": "physics-structured aggregate effective-acceleration body model",
        "runtime_policy": "offline only; no MPC/odom/planner/runtime topic changes",
        "body_equations": {
            "u_rear_dot": "a_x_eff + r*(v_rear + L*r)",
            "v_rear_dot": "a_y_eff - r*u_rear - L*alpha_z_eff",
            "yaw_rate_dot": "alpha_z_eff",
            "rear_axle_to_com_x_m": 0.15532,
            "learned_outputs": ["a_x_eff_mps2", "a_y_eff_mps2",
                                "alpha_z_eff_radps2"],
        },
        "plant_protocol": {
            "inputs": ("body-state history plus measured steering/throttle feedback, rear-wheel speed and logged commands"
                       + (" plus IMU roll/pitch and roll/pitch rates from the observed history"
                          if args.condition_on_attitude else "")),
            "future_forcing": (
                "measured actuator and wheel channels are fed during rollout; "
                "IMU roll/pitch and rates are held at their last observed value "
                "and never read from the future; future body truth is never injected"
                if args.condition_on_attitude else
                "measured actuator and wheel channels are fed during rollout; "
                "future body truth is never injected"),
            "gru_control": "the established GRU architecture is re-trained independently inside each outer fold with measured actuator/wheel teacher forcing and body-only loss to match the structured candidates; using its existing all-run checkpoint would leak held-out runs",
            "metric_unit": "whole simulator run; overlapping windows are not treated as independent trials",
        },
        "split_policy": "five grouped folds over original train runs only; original named test, validation, and final-test runs are excluded",
        "speed_edges_mps": list(SPEED_EDGES[:-1]),
        "speed_last_bin_open_ended": True,
        "absolute_steering_edges_rad": list(STEERING_EDGES[:-1]),
        "steering_last_bin_open_ended": True,
        "fold_count": args.folds,
        "eligible_run_count": len(eligible_runs),
        "sequences_excluded_for_incomplete_attitude":
            excluded_incomplete_attitude_sequences,
        "family_count": len(set(family_map.values())),
        "overall_cv_data_coverage": _coverage_profile(data, eligible_runs),
        "run_to_fold": {str(data["run_ids"][run]): fold
                        for run, fold in run_fold.items()},
        "run_to_family": family_map,
        "family_to_folds": family_fold,
        "configuration": {
            "architectures": list(args.architectures),
            "condition_on_measured_attitude_history": args.condition_on_attitude,
            "held_input_channels": list(held_input_channels),
            "retrained_checkpoints_on_resume": list(args.retrain_model),
            "steps_per_model": args.steps_per_model,
            "batch_size": args.batch_size,
            "history_steps": args.history_steps,
            "maximum_rollout_steps": args.rollout_steps,
            "rollout_curriculum": [[0.15, 5], [0.35, 10],
                                   [0.60, 20], [1.0, 32]],
            "hidden_size": args.hidden_size,
            "plain_gru_hidden_size": 128,
            "eval_every": args.eval_every,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "fold_seed": args.fold_seed,
            "seed": args.seed,
            "device": device,
            "cuda_name": torch.cuda.get_device_name(0)
                if device.startswith("cuda") else None,
            "mpc_yaw_parameters_source": str(args.mpc_config.resolve()),
            "mpc_yaw_parameters": args.yaw_parameters,
            "evaluation_windows_per_run_per_horizon": args.max_windows_per_run,
            "minimum_windows_per_run_for_paired_summary": 5,
        },
        "folds": [],
        "per_model_per_run": {},
    }
    all_results: dict[str, dict[str, Any]] = {
        architecture: {} for architecture in args.architectures}
    for fold in range(args.folds):
        test_runs = sorted(run for run in eligible_runs if run_fold[run] == fold)
        train_pool = sorted(run for run in eligible_runs if run_fold[run] != fold)
        inner_fold, _, _ = _grouped_folds(
            data, train_pool, manifest, min(args.folds, len(set(
                family_map[str(data["run_ids"][r])] for r in train_pool))),
            args.fold_seed + 101 + fold)
        inner_buckets: dict[int, list[int]] = defaultdict(list)
        for run, bucket in inner_fold.items():
            inner_buckets[bucket].append(run)
        validation_bucket = fold % len(inner_buckets)
        validation_runs = sorted(inner_buckets[validation_bucket])
        training_runs = sorted(set(train_pool) - set(validation_runs))
        fold_data = _fold_data(data, training_runs, validation_runs, test_runs)
        train_groups = _sequence_groups(fold_data, "train", args.history_steps,
                                        args.rollout_steps)
        validation_groups = _sequence_groups(
            fold_data, "validation", args.history_steps, args.rollout_steps)
        if not train_groups or not validation_groups:
            raise ValueError(f"fold {fold} has no train or inner-validation sequences")
        feature_mean, feature_scale = _normalizers(fold_data, train_groups)
        train_features, train_accels, train_dt = training_transition_rows(
            fold_data, set(train_groups))
        accel_stats = acceleration_statistics(train_accels)
        linear_coefficients = fit_linear_acceleration_baseline(
            train_features, train_dt, train_accels, feature_mean, feature_scale,
            accel_stats.mean, accel_stats.scale)
        args.regime_thresholds = _fold_regime_thresholds(
            train_features, train_accels, train_dt, args.yaw_parameters)
        fold_path = args.output_dir / f"fold_{fold:02d}"
        fold_path.mkdir(exist_ok=args.resume)
        fold_report = {
            "fold": fold,
            "training_runs": [str(data["run_ids"][r]) for r in training_runs],
            "inner_validation_runs": [str(data["run_ids"][r])
                                       for r in validation_runs],
            "test_runs": [str(data["run_ids"][r]) for r in test_runs],
            "train_sequences": sum(map(len, train_groups.values())),
            "validation_sequences": sum(map(len, validation_groups.values())),
            "training_transition_samples": accel_stats.sample_count,
            "acceleration_mean": accel_stats.mean.tolist(),
            "acceleration_scale": accel_stats.scale.tolist(),
            "regime_thresholds": args.regime_thresholds,
            "data_coverage": {
                "model_training": _coverage_profile(fold_data, training_runs),
                "inner_validation": _coverage_profile(fold_data, validation_runs),
                "outer_test": _coverage_profile(fold_data, test_runs),
            },
            "models": {},
        }
        report["folds"].append(fold_report)
        test_window_sets = {
            run: _run_window_sets(fold_data, run, args.history_steps,
                                  args.max_windows_per_run)
            for run in test_runs
        }
        print(f"fold={fold + 1}/{args.folds} train_runs={len(training_runs)} "
              f"inner_val_runs={len(validation_runs)} test_runs={len(test_runs)} "
              f"transition_samples={accel_stats.sample_count}", flush=True)
        for architecture in args.architectures:
            model_dir = fold_path / architecture
            model_dir.mkdir(exist_ok=args.resume)
            fit = _train_candidate(
                torch, nn, architecture, fold_data, train_groups,
                validation_groups, feature_mean, feature_scale,
                accel_stats.mean, accel_stats.scale, linear_coefficients,
                args, fold, model_dir, device)
            per_run = {}
            for run in test_runs:
                run_result = _score_run(
                    torch, fit["model"], architecture == "gru", fold_data,
                    run, feature_mean, feature_scale, args.regime_thresholds,
                    args.yaw_parameters, args.history_steps,
                    args.max_windows_per_run, device,
                    args.inference_batch_size, test_window_sets[run],
                    held_input_channels=args.held_input_channels)
                run_id = str(data["run_ids"][run])
                per_run[run_id] = run_result
                all_results[architecture][run_id] = run_result
            fit_report = {key: value for key, value in fit.items()
                          if key != "model"}
            fit_report["test_run_count"] = len(per_run)
            fit_report["test_750ms_macro_normalized_rmse"] = _macro_score(
                per_run, 30)
            fold_report["models"][architecture] = fit_report
            report["per_model_per_run"].setdefault(architecture, {}).update(per_run)
            print(f"fold={fold + 1} model={architecture} "
                  f"best_step={fit['best_step']} "
                  f"test_macro_750ms={fit_report['test_750ms_macro_normalized_rmse']}",
                  flush=True)
            _write_json(args.output_dir / "cv_progress.json", report)
        if not args.architectures:
            raise ValueError("no architectures were selected")
    report["cross_validation_summary"] = _aggregate_cv(
        all_results, args.bootstrap_count, args.seed + 51)
    report["paired_change_vs_gru"] = _paired_comparisons(
        all_results, args.bootstrap_count, args.seed + 71)
    _write_json(args.output_dir / "cv_report.json", report)
    print(f"wrote {args.output_dir / 'cv_report.json'}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mpc-config", type=Path,
                        default=Path("f1tenth_mpc/config/mpc_competition.yaml"))
    parser.add_argument("--architectures", nargs="+",
                        choices=DEFAULT_ARCHITECTURES,
                        default=list(DEFAULT_ARCHITECTURES))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps-per-model", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-windows-per-run", type=int, default=128)
    parser.add_argument("--validation-windows-per-run", type=int, default=48)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--fold-seed", type=int, default=20260929)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--condition-on-attitude", action="store_true",
                        help="use measured IMU roll/pitch and roll/pitch rates from history; hold them at their last observed value during rollout")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="reuse only checkpoints whose fold metadata matches")
    parser.add_argument("--retrain-model", action="append", default=[],
                        metavar="FOLD:ARCH",
                        help="retrain one zero-based fold/model instead of reusing its checkpoint")
    args = parser.parse_args()
    if (args.folds < 2 or args.steps_per_model < 1 or args.batch_size < 1
            or args.history_steps < 2 or args.rollout_steps < 30
            or args.hidden_size < 8 or args.eval_every < 1
            or args.patience < 1 or args.max_windows_per_run < 1
            or args.validation_windows_per_run < 1
            or args.inference_batch_size < 1 or args.bootstrap_count < 100):
        parser.error("fold, training, evaluation and bootstrap settings are invalid")
    if len(set(args.architectures)) != len(args.architectures):
        parser.error("architecture names must be unique")
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"structured body CV failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
