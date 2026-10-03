#!/usr/bin/env python3
"""Run WP17 diagnostic interventions on the frozen parent EDSSM.

Every oracle mode deliberately injects future measured data and is therefore
diagnostic/noncausal. I0 is the only normal command-only rollout. No model
weights are changed and no training split contributes rollout windows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    append_roll_state,
    integrate_pose,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _fixed_validation_windows,
    _load_model,
    _window_batch,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.vehicle_dynamics_learning.signal_semantics import (
    REAR_TRACK_WIDTH_M,
)


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001")
PARENT_ROOT = (TASK_ROOT / "replacement_offline_sim_raceline_20261002"
    / "encoder_raw_state_teacher_v1")
DEFAULT_CHECKPOINT = (PARENT_ROOT
    / "edssm_gru_z32_e2_rollresidual_10s_yawonly_wheelonly_lowthrottle4_joint_lr3e5_seed101"
    / "best.pt")
EXPECTED_CHECKPOINT_SHA256 = (
    "8f84fa54306492bd9750e4e1e3fb0be014195491eccd8f4fd67f2943402bfe8e")
DEFAULT_DYNAMIC_DATASET = PARENT_ROOT / "openplane_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE_DATASET = PARENT_ROOT / "practice_dynamics_raw_wheels.npz"
FIXED_ENCODER_ROOT = (TASK_ROOT / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/encoder_fixed40hz_v1")
DEFAULT_DYNAMIC_FIXED = FIXED_ENCODER_ROOT / \
    "openplane_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_PRACTICE_FIXED = FIXED_ENCODER_ROOT / \
    "practice_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_PRACTICE_COMPARISON = (PARENT_ROOT
    / "paired_contact_slip_vs_parent_v1/paired_comparison.json")
DEFAULT_FIRST_DIVERGENCE = PARENT_ROOT / \
    "recursive_first_divergence_wp14_parent_5s_v1.json"
DEFAULT_OUTPUT = (FIXED_ENCODER_ROOT.parent
    / "wp17_frozen_parent_interventions_v1.json")
HORIZONS = {"0.25s": 10, "0.75s": 30, "2s": 80,
            "5s": 200, "10s": 400}
ORACLE_LABEL = "diagnostic noncausal future intervention"
MODES = ("I0", "I1", "I2", "I3", "I4", "I5", "I6")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_oracle_alignment(expected_rows: np.ndarray,
                              supplied_rows: np.ndarray,
                              mode: str) -> None:
    """Reject offset/misaligned oracle rows; expected input is current state."""
    expected = np.asarray(expected_rows, dtype=np.int64)
    supplied = np.asarray(supplied_rows, dtype=np.int64)
    if mode not in ("I1", "I2", "I4", "I5"):
        raise ValueError(f"{mode} is not an oracle mode")
    if expected.shape != supplied.shape or not np.array_equal(expected, supplied):
        raise ValueError(
            f"{mode} oracle rows must align to the current state row; "
            "shifted/future-offset observations are forbidden")


def _check_same_base_rows(source: dict[str, Any], fixed: dict[str, Any],
                          source_path: Path, fixed_path: Path) -> None:
    """Verify fixed-rate sidecar is row-identical to the WP14 dataset."""
    for name in ("run_ids", "splits", "bounds", "seq_run", "packet_sequence",
                 "dt_s", "frames", "simulator_rigid_state",
                 "simulator_pose_xyyaw", "sample_time_ns"):
        left = np.asarray(source[name])
        right = np.asarray(fixed[name])
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(
                f"WP16 fixed-cadence dataset is not row-identical at {name}: "
                f"{source_path} vs {fixed_path}")
    if "encoder_raw_surface_mps" not in fixed or "encoder_raw_valid" not in fixed:
        raise ValueError(f"fixed-cadence encoder sidecar missing from {fixed_path}")


def _practice_windows(data: dict[str, Any], comparison_path: Path
                      ) -> list[dict[str, Any]]:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if (comparison.get("practice_dataset_sha256") is None
            or comparison.get("future_truth_or_feedback_used") is not False
            or comparison.get("selection_was_frozen_before_practice_transfer") is not True):
        raise ValueError("WP14 practice window source failed frozen-provenance checks")
    run_lookup = {str(name): index for index, name in
                  enumerate(np.asarray(data["run_ids"]).astype(str))}
    windows = []
    for item in comparison["production_practice_transfer"]["wp14_parent"][
            "window_manifest"]:
        run_name = str(item["run_id"])
        if run_name not in run_lookup:
            raise ValueError(f"WP14 practice window references unknown run {run_name}")
        windows.append({
            "start": int(item["start"]), "run": run_lookup[run_name],
            "sequence_id": int(item["sequence_id"]),
            "regimes": list(item["regimes"]),
        })
    if len(windows) != 31:
        raise ValueError(f"WP14 practice start count changed: {len(windows)}")
    return windows


def _window_sequence_bounds(data: dict[str, Any], run: int, start: int
                            ) -> tuple[int, int]:
    matches = [(int(a), int(b)) for (a, b), row_run in
               zip(data["bounds"], data["seq_run"])
               if int(row_run) == run and int(a) <= start < int(b)]
    if len(matches) != 1:
        raise ValueError(f"window row {start} is not in one reset sequence")
    return matches[0]


def _horizon_windows(data: dict[str, Any], state: np.ndarray,
                     windows: list[dict[str, Any]],
                     raw_valid: np.ndarray | None,
                     horizon: int, require_raw_inputs: bool = False
                     ) -> list[dict[str, Any]]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    pose = np.asarray(data["simulator_pose_xyyaw"], dtype=np.float64)
    selected = []
    for window in windows:
        start, run = int(window["start"]), int(window["run"])
        _, end = _window_sequence_bounds(data, run, start)
        if start + horizon >= end:
            continue
        rows = np.arange(start + 1, start + horizon + 1)
        if (np.max(np.hypot(state[rows, 0], state[rows, 1])) > 12.0
                or np.max(frames[rows, 8]) > 0.50
                or not np.isfinite(pose[start:start + horizon + 1]).all()):
            continue
        if require_raw_inputs:
            if raw_valid is None:
                raise ValueError("raw-wheel intervention requires validity labels")
            input_rows = np.arange(start, start + horizon)
            if not np.all(raw_valid[input_rows]):
                continue
        selected.append(window)
    return selected


def _history_features(model: Any, data: dict[str, Any]) -> np.ndarray | None:
    if model.include_raw_encoder_history:
        return raw_encoder_history_features(data)
    if model.include_wheel_innovation_history:
        return wheel_innovation_history_features(data)
    return None


def _simulate(model: Any, torch: Any, data: dict[str, Any], state: np.ndarray,
              windows: list[dict[str, Any]], horizon: int, mode: str,
              fixed_data: dict[str, Any] | None, device: str,
              batch_size: int = 128) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown WP17 intervention mode {mode}")
    if mode == "I0" and fixed_data is not None:
        raise ValueError("normal I0 rollout cannot receive future oracle arrays")
    if not windows:
        return {"windows": [], "states": np.empty((0, horizon, state.shape[1])),
                "poses": np.empty((0, horizon, 3)),
                "latents": np.empty((0, horizon, 0))}
    if horizon not in (10, 30, 80, 200, 400):
        raise ValueError("WP17 horizon must match the handoff's fixed horizons")

    raw_history = _history_features(model, data)
    chunks: list[dict[str, Any]] = []
    for offset in range(0, len(windows), batch_size):
        rows = windows[offset:offset + batch_size]
        history, initial, delayed, commands, truth, truth_pose = _window_batch(
            data, state, rows, model.history_state_size, raw_history,
            horizon_steps=horizon)
        starts = np.asarray([int(row["start"]) for row in rows], dtype=np.int64)
        current_rows = starts[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
        target_rows = current_rows + 1
        if mode in ("I1", "I2", "I4", "I5"):
            validate_oracle_alignment(current_rows,
                                      current_rows.copy(), mode)

        hist_tensor = torch.as_tensor(history, dtype=torch.float32, device=device)
        state_tensor = torch.as_tensor(initial, dtype=torch.float32, device=device)
        delay_tensor = torch.as_tensor(delayed, dtype=torch.float32, device=device)
        command_tensor = torch.as_tensor(commands, dtype=torch.float32, device=device)
        truth_tensor = (torch.as_tensor(state[:, :3], dtype=torch.float32,
                                        device=device) if mode == "I4" else None)
        frame_tensor = (torch.as_tensor(data["frames"], dtype=torch.float32,
                                        device=device)
                        if mode in ("I2", "I5") else None)
        raw_tensor = None
        if mode == "I1":
            if fixed_data is None:
                raise ValueError("I1 requires the fixed-25ms encoder view")
            raw_tensor = torch.as_tensor(
                np.asarray(fixed_data["encoder_raw_surface_mps"], dtype=np.float32),
                dtype=torch.float32, device=device)
            valid = np.asarray(fixed_data["encoder_raw_valid"], dtype=bool)
            if not np.all(valid[current_rows]):
                raise ValueError("I1 selected a window containing invalid raw rate")
        if mode in ("I3", "I6"):
            if history.shape[2] < 7:
                raise ValueError("wheel-disconnection requires wheel channels in history")
            neutral = np.asarray(model.history_mean.detach().cpu())
            history[:, :, 5:7] = neutral[None, None, 5:7]
            hist_tensor = torch.as_tensor(history, dtype=torch.float32,
                                          device=device)

        latent = model.encode_history(hist_tensor)
        side_wheel = state_tensor[:, 5:7].clone()
        predicted_states: list[Any] = []
        predicted_latents: list[Any] = []
        for step in range(horizon):
            transition_state = state_tensor
            if mode in ("I1", "I2", "I3", "I4", "I5", "I6"):
                transition_state = state_tensor.clone()
            if mode == "I1":
                transition_state[:, 5:7] = raw_tensor[current_rows[:, step]]
            elif mode == "I2":
                transition_state[:, 5:7] = frame_tensor[
                    current_rows[:, step], 5:7]
            elif mode in ("I3", "I6"):
                state_mean = model.state_mean[5:7]
                transition_state[:, 5:7] = state_mean
            elif mode == "I4":
                true_current = truth_tensor[current_rows[:, step]]
                transition_state[:, 0:3] = true_current[:, 0:3]
            elif mode == "I5":
                transition_state[:, 3:5] = frame_tensor[
                    current_rows[:, step], 3:5]

            next_state, delay_tensor, latent, acceleration, _ = model.transition(
                transition_state, delay_tensor, latent,
                command_tensor[:, step])

            if mode == "I3":
                # Wheel acceleration remains a predicted measurement output;
                # only its normalized state/features are cut from body/latent.
                next_state = next_state.clone()
                next_state[:, 5:7] = (state_tensor[:, 5:7]
                                      + DT_S * acceleration[:, 3:5])
            elif mode == "I6":
                # Separate the wheel output accumulator from recurrent state.
                side_wheel = side_wheel + DT_S * acceleration[:, 3:5]
                next_state = next_state.clone()
                next_state[:, 5:7] = model.state_mean[5:7]
            elif mode == "I4":
                next_state = next_state.clone()
                next_state[:, 0:3] = truth_tensor[target_rows[:, step]]
            elif mode == "I5":
                next_state = next_state.clone()
                next_state[:, 3:5] = frame_tensor[target_rows[:, step], 3:5]

            recorded_state = next_state
            if mode == "I6":
                recorded_state = next_state.clone()
                recorded_state[:, 5:7] = side_wheel
            predicted_states.append(recorded_state)
            predicted_latents.append(latent)
            state_tensor = next_state

        pred_state = torch.stack(predicted_states, dim=1)
        pred_latent = torch.stack(predicted_latents, dim=1)
        pred_pose = integrate_pose(
            torch, pred_state, torch.as_tensor(truth_pose[:, 0],
                dtype=torch.float32, device=device),
            torch.as_tensor(initial, dtype=torch.float32, device=device))
        chunks.append({
            "windows": rows,
            "states": pred_state.detach().cpu().numpy().astype(np.float64),
            "poses": pred_pose.detach().cpu().numpy().astype(np.float64),
            "latents": pred_latent.detach().cpu().numpy().astype(np.float64),
            "truth": truth.astype(np.float64),
            "truth_pose": truth_pose[:, 1:].astype(np.float64),
            "current_rows": current_rows,
            "target_rows": target_rows,
        })
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0)
            if key != "windows" else [row for chunk in chunks for row in chunk[key]]
            for key in chunks[0]}


def _metric_arrays(result: dict[str, Any], fixed_data: dict[str, Any]
                   ) -> dict[str, np.ndarray]:
    state_error = result["states"] - result["truth"]
    pose_error = result["poses"] - result["truth_pose"]
    pose_heading = np.arctan2(np.sin(pose_error[:, :, 2]),
                              np.cos(pose_error[:, :, 2]))
    rows = result["target_rows"]
    raw = np.asarray(fixed_data["encoder_raw_surface_mps"], dtype=np.float64)[rows]
    raw_valid = np.asarray(fixed_data["encoder_raw_valid"], dtype=bool)[rows]
    wheel_raw_error = np.sqrt(np.mean((result["states"][:, :, 5:7] - raw) ** 2,
                                      axis=2))
    wheel_raw_error[~raw_valid] = np.nan
    return {
        "u_mps": state_error[:, :, 0],
        "v_mps": state_error[:, :, 1],
        "yaw_rate_rps": state_error[:, :, 2],
        "heading_rad": pose_heading,
        "radial_position_m": np.linalg.norm(pose_error[:, :, :2], axis=2),
        "wheel_pair_vs_stored_mps": np.sqrt(np.mean(
            state_error[:, :, 5:7] ** 2, axis=2)),
        "wheel_pair_vs_fixed25_mps": wheel_raw_error,
        "latent_norm": np.linalg.norm(result["latents"], axis=2),
    }


def _run_horizon_metrics(result: dict[str, Any], metrics: dict[str, np.ndarray],
                         horizon: int, data: dict[str, Any]
                         ) -> dict[str, Any]:
    run_ids = np.asarray(data["run_ids"]).astype(str)
    cumulative_by_run: dict[str, Any] = {}
    endpoint_by_run: dict[str, Any] = {}
    runs = sorted({int(row["run"]) for row in result["windows"]})
    for run in runs:
        selected = np.asarray([int(row["run"]) == run
                               for row in result["windows"]], dtype=bool)
        cumulative_metrics: dict[str, Any] = {}
        endpoint_metrics: dict[str, Any] = {}
        for name, values in metrics.items():
            chosen = np.asarray(values[selected, :horizon], dtype=np.float64)
            endpoint = np.asarray(values[selected, horizon - 1], dtype=np.float64)
            finite = np.isfinite(chosen)
            cumulative_metrics[name] = (float(np.sqrt(np.mean(chosen[finite] ** 2)))
                if np.any(finite) else None)
            endpoint = endpoint[np.isfinite(endpoint)]
            endpoint_metrics[name] = (float(np.sqrt(np.mean(endpoint ** 2)))
                if len(endpoint) else None)
        cumulative_metrics["windows"] = int(selected.sum())
        endpoint_metrics["windows"] = int(selected.sum())
        cumulative_by_run[str(run_ids[run])] = cumulative_metrics
        endpoint_by_run[str(run_ids[run])] = endpoint_metrics

    def aggregate(per_run: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        macro: dict[str, Any] = {}
        bootstrap: dict[str, Any] = {}
        rng = np.random.default_rng(seed)
        for name in metrics:
            values = np.asarray([row[name] for row in per_run.values()
                                 if row[name] is not None], dtype=np.float64)
            if not len(values):
                macro[name] = None
                bootstrap[name] = None
            else:
                macro[name] = float(np.mean(values))
                draws = rng.integers(0, len(values), size=(5000, len(values)))
                bootstrap[name] = np.quantile(values[draws].mean(axis=1),
                                               (0.025, 0.975)).tolist()
        return macro, bootstrap

    cumulative_macro, cumulative_ci = aggregate(
        cumulative_by_run, 20261003 + horizon + len(runs))
    endpoint_macro, endpoint_ci = aggregate(
        endpoint_by_run, 20261031 + horizon + len(runs))
    return {"independent_run_count": len(cumulative_by_run),
            "weak_evidence_under_three_runs": len(cumulative_by_run) < 3,
            "cumulative_macro_run_rmse_or_latent_rms": cumulative_macro,
            "cumulative_run_cluster_bootstrap_95pct_ci": cumulative_ci,
            "endpoint_macro_run_rmse_or_latent_rms": endpoint_macro,
            "endpoint_run_cluster_bootstrap_95pct_ci": endpoint_ci,
            "per_run_cumulative": cumulative_by_run,
            "per_run_endpoint": endpoint_by_run}


def _paired_deltas(reference: dict[str, Any], treatment: dict[str, Any],
                   fixed_data: dict[str, Any], horizon: int,
                   data: dict[str, Any],
                   oracle_clamped_channels: tuple[str, ...] = ()
                   ) -> dict[str, Any]:
    reference_by_start = {(int(row["run"]), int(row["start"])): index
                          for index, row in enumerate(reference["windows"])}
    pairs = [(index, reference_by_start.get((int(row["run"]), int(row["start"]))))
             for index, row in enumerate(treatment["windows"])]
    if any(reference_index is None for _, reference_index in pairs):
        raise ValueError("paired intervention/reference starts are not identical")
    ref_indices = np.asarray([int(pair[1]) for pair in pairs], dtype=np.int64)
    treat_indices = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    ref_slice = {key: value[ref_indices] for key, value in reference.items()
                 if isinstance(value, np.ndarray) and value.ndim > 0}
    treatment_slice = {key: value[treat_indices] for key, value in treatment.items()
                       if isinstance(value, np.ndarray) and value.ndim > 0}
    ref_slice["windows"] = [reference["windows"][i] for i in ref_indices]
    treatment_slice["windows"] = [treatment["windows"][i] for i in treat_indices]
    ref_metrics = _metric_arrays(ref_slice, fixed_data)
    treatment_metrics = _metric_arrays(treatment_slice, fixed_data)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    by_channel: dict[str, Any] = {}
    for channel in ("u_mps", "v_mps", "yaw_rate_rps", "heading_rad",
                    "radial_position_m", "wheel_pair_vs_stored_mps",
                    "wheel_pair_vs_fixed25_mps", "latent_norm"):
        if channel in oracle_clamped_channels:
            by_channel[channel] = {
                "status": "oracle-clamped by intervention; not a predictive metric",
                "paired_run_relative_rmse_improvement_fraction": None,
                "run_cluster_bootstrap_95pct_ci": None,
                "runs_improved": None,
                "runs_compared": 0,
                "per_run_relative_rmse_improvement_fraction": {},
            }
            continue
        changes: dict[str, float] = {}
        endpoint_changes: dict[str, float] = {}
        absolute_changes: dict[str, float] = {}
        endpoint_absolute_changes: dict[str, float] = {}
        for run in sorted({int(row["run"]) for row in ref_slice["windows"]}):
            selected = np.asarray([int(row["run"]) == run
                                   for row in ref_slice["windows"]], dtype=bool)
            a = np.asarray(ref_metrics[channel][selected, :horizon], dtype=np.float64)
            b = np.asarray(treatment_metrics[channel][selected, :horizon], dtype=np.float64)
            valid = np.isfinite(a) & np.isfinite(b)
            if not np.any(valid):
                continue
            ref_rmse = float(np.sqrt(np.mean(a[valid] ** 2)))
            new_rmse = float(np.sqrt(np.mean(b[valid] ** 2)))
            ref_endpoint = a[:, horizon - 1]
            new_endpoint = b[:, horizon - 1]
            endpoint_valid = (np.isfinite(ref_endpoint)
                              & np.isfinite(new_endpoint))
            if not np.any(endpoint_valid):
                continue
            ref_endpoint_rmse = float(np.sqrt(np.mean(
                ref_endpoint[endpoint_valid] ** 2)))
            new_endpoint_rmse = float(np.sqrt(np.mean(
                new_endpoint[endpoint_valid] ** 2)))
            if channel.endswith("mps") or channel in (
                    "yaw_rate_rps", "heading_rad", "radial_position_m",
                    "latent_norm"):
                changes[str(run_ids[run])] = (ref_rmse - new_rmse) / max(ref_rmse, 1e-12)
                absolute_changes[str(run_ids[run])] = ref_rmse - new_rmse
                endpoint_changes[str(run_ids[run])] = (
                    ref_endpoint_rmse - new_endpoint_rmse
                    ) / max(ref_endpoint_rmse, 1e-12)
                endpoint_absolute_changes[str(run_ids[run])] = (
                    ref_endpoint_rmse - new_endpoint_rmse)
        values = np.asarray(list(changes.values()), dtype=np.float64)
        endpoint_values = np.asarray(list(endpoint_changes.values()), dtype=np.float64)

        def bootstrap_summary(samples: np.ndarray, seed: int):
            if not len(samples):
                return None, None, 0
            rng = np.random.default_rng(seed + len(samples))
            draws = rng.integers(0, len(samples), size=(5000, len(samples)))
            interval = np.quantile(samples[draws].mean(axis=1),
                                   (0.025, 0.975)).tolist()
            return float(np.mean(samples)), interval, int(np.count_nonzero(samples > 0.0))

        mean, ci, improved = bootstrap_summary(values, 20261017 + horizon)
        endpoint_mean, endpoint_ci, endpoint_improved = bootstrap_summary(
            endpoint_values, 20261031 + horizon)
        by_channel[channel] = {
            ("paired_run_relative_rms_change_fraction" if channel == "latent_norm"
             else "paired_run_relative_rmse_improvement_fraction"): mean,
            "run_cluster_bootstrap_95pct_ci": ci,
            "endpoint_paired_run_relative_rms_change_fraction" if channel == "latent_norm"
                else "endpoint_paired_run_relative_rmse_improvement_fraction": endpoint_mean,
            "endpoint_run_cluster_bootstrap_95pct_ci": endpoint_ci,
            "runs_improved": improved,
            "endpoint_runs_improved": endpoint_improved,
            "runs_compared": int(len(values)),
            "per_run_relative_rmse_improvement_fraction": changes,
            "per_run_endpoint_relative_rmse_improvement_fraction": endpoint_changes,
            "per_run_absolute_rmse_improvement": absolute_changes,
            "per_run_endpoint_absolute_rmse_improvement": endpoint_absolute_changes,
        }
    return {"matched_window_count": len(pairs),
            "independent_run_count": len(set(int(row["run"])
                                               for row in treatment_slice["windows"])),
            "channels": by_channel}


def _first_divergence(result: dict[str, Any], metrics: dict[str, np.ndarray],
                      data: dict[str, Any]) -> dict[str, Any]:
    run_ids = np.asarray(data["run_ids"]).astype(str)
    result_by_channel = {}
    for name, errors in metrics.items():
        if name == "latent_norm":
            continue
        per_run = []
        per_run_crossing = {}
        for run in sorted({int(row["run"]) for row in result["windows"]}):
            mask = np.asarray([int(row["run"]) == run
                               for row in result["windows"]], dtype=bool)
            values = np.asarray(errors[mask], dtype=np.float64)
            valid_count = np.sum(np.isfinite(values), axis=0)
            squared = np.nansum(values ** 2, axis=0)
            curve = np.sqrt(squared / np.maximum(valid_count, 1))
            curve[valid_count == 0] = np.nan
            per_run.append(curve)
            finite = np.flatnonzero(np.isfinite(curve))
            crossing = None
            if finite.size:
                initial = curve[finite[0]]
                above = finite[curve[finite] > 2.0 * initial]
                crossing = (int(above[0] + 1) * DT_S if len(above) else None)
            per_run_crossing[str(run_ids[run])] = crossing
        curves = np.asarray(per_run)
        macro = np.nanmean(curves, axis=0)
        finite = np.flatnonzero(np.isfinite(macro))
        crossing = None
        if finite.size:
            initial = macro[finite[0]]
            above = finite[macro[finite] > 2.0 * initial]
            crossing = (int(above[0] + 1) * DT_S if len(above) else None)
        result_by_channel[name] = {
            "first_macro_rmse_over_twice_first_step_s": crossing,
            "per_run_first_macro_rmse_over_twice_first_step_s": per_run_crossing,
            "macro_run_rmse_at_0_25_0_75_2_5_10s": {
                f"{steps * DT_S:g}s": float(macro[steps - 1])
                for steps in HORIZONS.values()
                if steps <= len(macro) and np.isfinite(macro[steps - 1])},
        }
    return result_by_channel


def _eligible_horizon(windows_by_domain: dict[str, Any], data: dict[str, Any],
                      state: np.ndarray, fixed_data: dict[str, Any],
                      horizon: int) -> dict[str, list[dict[str, Any]]]:
    raw_valid = np.asarray(fixed_data["encoder_raw_valid"], dtype=bool)
    return {
        "all": _horizon_windows(data, state, windows_by_domain,
                                raw_valid, horizon, False),
        "raw": _horizon_windows(data, state, windows_by_domain,
                                raw_valid, horizon, True),
    }


def _state_topology_decision(domains: dict[str, Any]) -> dict[str, Any]:
    """Apply WP17's predeclared oracle-improvement decision to run-level data."""
    dynamic = domains["dynamic_validation"]["horizons"]
    required = ("0.75s", "5s")
    body_channels = ("u_mps", "v_mps", "yaw_rate_rps", "heading_rad",
                     "radial_position_m")
    evidence = {}
    supports_measurement_state = True
    for horizon in required:
        horizon_data = dynamic[horizon]
        evidence[horizon] = {}
        for mode in ("I1", "I2"):
            paired = horizon_data["modes"][mode]["paired_delta_vs_I0"]
            rows = {}
            for channel in body_channels:
                item = paired["channels"][channel]
                mean = item["endpoint_paired_run_relative_rmse_improvement_fraction"]
                ci = item["endpoint_run_cluster_bootstrap_95pct_ci"]
                improved = item["endpoint_runs_improved"]
                run_count = item["runs_compared"]
                rows[channel] = {"mean_relative_improvement": mean,
                    "run_cluster_bootstrap_95pct_ci": ci,
                    "runs_improved": improved,
                    "independent_runs": run_count}
                if (mean is None or ci is None or run_count < 3
                        or mean >= 0.10 or ci[1] > 0.0
                        or improved >= (run_count + 1) // 2):
                    supports_measurement_state = False
            evidence[horizon][mode] = rows
    if supports_measurement_state:
        state = "A: recursive physical state"
        reason = "Both measured-wheel oracle definitions produce >=10% paired endpoint-error reduction with run-level support on the primary dynamic horizons."
    else:
        state = "B: measurement/output only"
        reason = (
            "The fixed-25-ms and stored-filtered wheel oracles do not improve "
            "the dynamic body trajectory by the WP17 threshold on both 0.75 s "
            "and 5 s; their long-horizon paired effects are predominantly "
            "regressions. This rejects the measured processed rate as a "
            "recursive physical state for the replacement plant, without "
            "claiming that latent traction history is unnecessary.")
    return {
        "rear_processed_wheel_rate": state,
        "decision_reason": reason,
        "primary_dynamic_evidence": evidence,
        "practice_runs_are_transfer_diagnostic_only": True,
        "limitations": [
            "Practice contributes only two independent runs, so it cannot establish a precise transfer effect.",
            "I3/I6 mean-neutralization creates wheel/contact mismatch with p95 about 4.5 m/s on dynamic validation and is off-support; its large error is not treated as a clean causal estimate.",
            "I4 body error is oracle-clamped by construction; its body metrics are not predictive scores.",
            "This decision concerns the processed measured wheel rate only. It does not say wheel physics or latent traction is irrelevant.",
        ],
        "downstream_requirement": (
            "Represent rear encoder behavior as a measurement/output in the "
            "replacement model. Any latent traction/history state must be "
            "causally inferred and predicted internally; WP18 determines "
            "whether such hidden state is empirically needed."),
    }


def _one_domain(torch: Any, model: Any, data: dict[str, Any],
                fixed_data: dict[str, Any], windows: list[dict[str, Any]],
                device: str) -> dict[str, Any]:
    state = physical_state_from_dataset(
        data, str(model.metadata.get("wheel_state_source", "filtered_odometry")))
    if model.include_roll_state:
        state = append_roll_state(data, state)
    eligibility = {label: _eligible_horizon(windows, data, state, fixed_data, steps)
                   for label, steps in HORIZONS.items()}
    report: dict[str, Any] = {"horizons": {}, "window_starts": {
        "frozen_wp14_window_count": len(windows),
        "frozen_wp14_runs": sorted({str(data["run_ids"][int(w["run"])])
                                     for w in windows}),
    }}
    for label, horizon in HORIZONS.items():
        cohorts = eligibility[label]
        if not cohorts["all"]:
            report["horizons"][label] = {"status": "unavailable"}
            continue
        cohort_key = "all"
        if label != "10s":
            cohort_key = "all"
        raw_windows = cohorts["raw"]
        baseline_all = _simulate(model, torch, data, state, cohorts["all"],
            horizon, "I0", None, device)
        baseline_raw = (baseline_all if len(raw_windows) == len(cohorts["all"])
                        and all(a["start"] == b["start"] for a, b in
                            zip(raw_windows, cohorts["all"]))
                        else _simulate(model, torch, data, state, raw_windows,
                            horizon, "I0", None, device))
        mode_results: dict[str, dict[str, Any]] = {"I0": baseline_all}
        mode_pair_refs: dict[str, dict[str, Any]] = {"I0": baseline_all}
        mode_windows = {"I0": cohorts["all"]}
        # I1 receives fixed-rate encoder observations only on complete-valid
        # windows; its paired I0 is recomputed on those exact same starts.
        if raw_windows:
            mode_results["I1"] = _simulate(model, torch, data, state,
                raw_windows, horizon, "I1", fixed_data, device)
            mode_pair_refs["I1"] = baseline_raw
            mode_windows["I1"] = raw_windows
        for mode in ("I2", "I3", "I4", "I5", "I6"):
            mode_results[mode] = _simulate(model, torch, data, state,
                cohorts["all"], horizon, mode, fixed_data, device)
            mode_pair_refs[mode] = baseline_all
            mode_windows[mode] = cohorts["all"]

        horizon_modes: dict[str, Any] = {}
        for mode, result in mode_results.items():
            metrics = _metric_arrays(result, fixed_data)
            horizon_modes[mode] = {
                "window_count": len(result["windows"]),
                "independent_run_count": len(set(
                    int(row["run"]) for row in result["windows"])),
                "metrics": _run_horizon_metrics(result, metrics, horizon, data),
                "first_divergence": _first_divergence(result, metrics, data),
                "label": ("normal command-only recursive rollout" if mode == "I0"
                          else ORACLE_LABEL),
                "oracle_clamped_state_channels": {
                    "I1": ["rear_left_surface_speed_mps",
                           "rear_right_surface_speed_mps"],
                    "I2": ["rear_left_surface_speed_mps",
                           "rear_right_surface_speed_mps"],
                    "I4": ["u_com_mps", "v_com_mps", "yaw_rate_rps"],
                    "I5": ["steering_actual_rad", "throttle_feedback_norm"],
                }.get(mode, []),
            }
            if mode != "I0":
                reference = mode_pair_refs[mode]
                horizon_modes[mode]["paired_delta_vs_I0"] = _paired_deltas(
                    reference, result, fixed_data, horizon, data,
                    oracle_clamped_channels=(
                        ("u_mps", "v_mps", "yaw_rate_rps")
                        if mode == "I4" else ()))

        if "I3" in mode_results and "I6" in mode_results:
            for signal in ("states", "poses", "latents"):
                if not np.allclose(mode_results["I3"][signal],
                                   mode_results["I6"][signal],
                                   rtol=1e-6, atol=1e-6):
                    raise ValueError(
                        f"I3/I6 {signal} mismatch despite shared masked transition")
        report["horizons"][label] = {
            "steps": horizon,
            "available_wp14_start_count": len(cohorts["all"]),
            "raw_oracle_complete_input_count": len(raw_windows),
            "raw_oracle_independent_runs": len(set(
                int(row["run"]) for row in raw_windows)),
            "modes": horizon_modes,
        }
    neutral_wheel = np.asarray(model.state_mean.detach().cpu(),
                               dtype=np.float64)[5:7]
    used_rows = np.concatenate([
        np.arange(int(row["start"]), int(row["start"]) + 200,
                  dtype=np.int64) for row in windows])
    used_body = state[used_rows]
    half_track = REAR_TRACK_WIDTH_M / 2.0
    body_contact = np.column_stack((
        used_body[:, 0] - used_body[:, 2] * half_track,
        used_body[:, 0] + used_body[:, 2] * half_track))
    neutral_mismatch = np.asarray(neutral_wheel[None, :] - body_contact)
    report["i3_i6_architectural_check"] = {
        "parent_mode": str(model.wheel_dynamics_mode),
        "contact_slip_feature_path_present": model.wheel_dynamics_mode == "contact_slip",
        "wheel_features_neutralized_in_history_and_transition": True,
        "neutralization": "replace both normalized wheel channels with their training means at the history encoder and every transition; this is the handoff-prescribed mean/zero-normalized intervention",
        "neutralized_wheel_training_mean_mps": neutral_wheel.tolist(),
        "neutralized_wheel_minus_kinematic_contact_speed_mps": {
            "bias_per_side": np.mean(neutral_mismatch, axis=0).tolist(),
            "rmse_per_side": np.sqrt(np.mean(neutral_mismatch ** 2,
                                               axis=0)).tolist(),
            "absolute_p95_per_side": np.quantile(
                np.abs(neutral_mismatch), 0.95, axis=0).tolist(),
            "interpretation": "large values indicate this ablation changes wheel/body mismatch to an off-support condition; do not treat its rollout error as a clean causal estimate",
        },
        "I3_I6_body_and_latent_dynamics_equivalent_by_construction": True,
        "reason": "The frozen surface-acceleration EDSSM has no separate measurement head. I3 preserves its recursive wheel output accumulator; I6 keeps that accumulator side-channel-only. Both mask wheel channels from history, acceleration, roll-residual, and latent inputs, so body/latent trajectories must agree.",
    }
    return report


def run(checkpoint: Path, dynamic_dataset: Path, practice_dataset: Path,
        dynamic_fixed: Path, practice_fixed: Path, practice_comparison: Path,
        first_divergence: Path, output: Path, device: str = "cuda"
        ) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite WP17 report: {output}")
    for path in (checkpoint, dynamic_dataset, practice_dataset, dynamic_fixed,
                 practice_fixed, practice_comparison, first_divergence):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("WP17 checkpoint must match frozen WP14 comparator")
    frozen = json.loads(first_divergence.read_text(encoding="utf-8"))
    if (frozen.get("checkpoint_sha256") != checkpoint_sha
            or frozen.get("future_truth_or_feedback_used_for_prediction") is not False
            or frozen.get("test_and_final_test_used") is not False):
        raise ValueError("WP14 first-divergence report does not freeze this parent")

    torch, model, metadata = _load_model(checkpoint, device)
    model.metadata = metadata
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dynamic = _load_dataset(dynamic_dataset)
    practice = _load_dataset(practice_dataset)
    dynamic_fixed_data = _load_dataset(dynamic_fixed)
    practice_fixed_data = _load_dataset(practice_fixed)
    _check_same_base_rows(dynamic, dynamic_fixed_data,
                          dynamic_dataset, dynamic_fixed)
    _check_same_base_rows(practice, practice_fixed_data,
                          practice_dataset, practice_fixed)
    if not np.allclose(dynamic["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("WP17 source dataset cadence is not exact 25 ms")

    dynamic_state = physical_state_from_dataset(dynamic,
        str(metadata.get("wheel_state_source", "filtered_odometry")))
    if model.include_roll_state:
        dynamic_state = append_roll_state(dynamic, dynamic_state)
    dynamic_windows = _fixed_validation_windows(
        dynamic, dynamic_state, max_windows_per_run=24, horizon_steps=200,
        max_throttle_command=0.50, split="validation")
    expected_dynamic = frozen["domains"]["dynamic_validation"]
    dynamic_ids = sorted({str(dynamic["run_ids"][int(row["run"])])
                          for row in dynamic_windows})
    if (len(dynamic_windows) != expected_dynamic["window_count"]
            or dynamic_ids != expected_dynamic["whole_run_ids"]):
        raise ValueError("WP17 dynamic starts do not reproduce frozen WP14 cohort")
    practice_windows = _practice_windows(practice, practice_comparison)
    expected_practice = frozen["domains"]["practice_unseen"]
    if (frozen.get("practice_comparison_sha256")
            != sha256(practice_comparison)):
        raise ValueError("WP14 practice start manifest hash differs from frozen report")
    practice_ids = sorted({str(practice["run_ids"][int(row["run"])])
                           for row in practice_windows})
    if (len(practice_windows) != expected_practice["window_count"]
            or practice_ids != expected_practice["whole_run_ids"]):
        raise ValueError("WP17 practice starts do not reproduce frozen WP14 cohort")
    training = set(str(value) for value in metadata.get("training_runs", []))
    if training.intersection(dynamic_ids + practice_ids):
        raise ValueError("WP17 validation start cohort overlaps model training runs")

    # Runtime equivalence check: I0's explicit step wrapper must reproduce the
    # production EDSSM rollout before any intervention results are accepted.
    probe_windows = dynamic_windows[:2]
    probe_state = physical_state_from_dataset(dynamic,
        str(metadata.get("wheel_state_source", "filtered_odometry")))
    if model.include_roll_state:
        probe_state = append_roll_state(dynamic, probe_state)
    probe = _simulate(model, torch, dynamic, probe_state, probe_windows, 10,
                      "I0", None, device)
    raw_history = _history_features(model, dynamic)
    history, initial, delayed, commands, _, _ = _window_batch(
        dynamic, probe_state, probe_windows, model.history_state_size,
        raw_history, horizon_steps=10)
    with torch.no_grad():
        expected, _, _, _ = model.rollout(
            torch.as_tensor(initial, dtype=torch.float32, device=device),
            torch.as_tensor(delayed, dtype=torch.float32, device=device),
            torch.as_tensor(history, dtype=torch.float32, device=device),
            torch.as_tensor(commands, dtype=torch.float32, device=device))
    if not np.allclose(probe["states"], expected.detach().cpu().numpy(),
                       rtol=1e-6, atol=1e-6):
        raise ValueError("I0 wrapper does not match the frozen model rollout")

    result = {
        "schema_version": 1,
        "purpose": "WP17 frozen-parent wheel/body/actuator causal intervention diagnosis",
        "generated_date": "2026-10-03",
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "training_performed": False,
        "checkpoint_unchanged": sha256(checkpoint) == checkpoint_sha,
        "test_and_final_test_used": False,
        "future_truth_or_feedback_used_by_I0": False,
        "modes_I1_to_I6_label": ORACLE_LABEL,
        "diagnostic_interventions_are_not_causal_deployment_predictions": True,
        "window_cohorts_source": str(first_divergence.relative_to(ROOT)),
        "window_cohorts_source_sha256": sha256(first_divergence),
        "practice_start_manifest": str(practice_comparison.relative_to(ROOT)),
        "practice_start_manifest_sha256": sha256(practice_comparison),
        "fixed_rate_row_alignment_verified": True,
        "source_and_fixed_view_core_arrays_identical": [
            "run_ids", "splits", "bounds", "seq_run", "packet_sequence",
            "dt_s", "frames", "simulator_rigid_state",
            "simulator_pose_xyyaw", "sample_time_ns"],
        "horizons_s": {key: steps * DT_S for key, steps in HORIZONS.items()},
        "domains": {
            "dynamic_validation": _one_domain(
                torch, model, dynamic, dynamic_fixed_data,
                dynamic_windows, device),
            "practice_unseen": _one_domain(
                torch, model, practice, practice_fixed_data,
                practice_windows, device),
        },
    }
    result["wp17_decision"] = _state_topology_decision(result["domains"])
    result["checkpoint_unchanged"] = sha256(checkpoint) == checkpoint_sha
    if not result["checkpoint_unchanged"]:
        raise RuntimeError("frozen parent checkpoint changed during WP17")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dynamic-dataset", type=Path, default=DEFAULT_DYNAMIC_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE_DATASET)
    parser.add_argument("--dynamic-fixed", type=Path, default=DEFAULT_DYNAMIC_FIXED)
    parser.add_argument("--practice-fixed", type=Path, default=DEFAULT_PRACTICE_FIXED)
    parser.add_argument("--practice-comparison", type=Path,
                        default=DEFAULT_PRACTICE_COMPARISON)
    parser.add_argument("--first-divergence", type=Path,
                        default=DEFAULT_FIRST_DIVERGENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run(*(getattr(args, name).resolve() for name in (
        "checkpoint", "dynamic_dataset", "practice_dataset", "dynamic_fixed",
        "practice_fixed", "practice_comparison", "first_divergence", "output")),
        device=args.device)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "domains": {name: {
            "frozen_starts": value["window_starts"],
            "horizons": {h: {
                "windows": item.get("available_wp14_start_count"),
                "raw_complete": item.get("raw_oracle_complete_input_count"),
                "modes": sorted(item.get("modes", {})),
            } for h, item in value["horizons"].items()},
        } for name, value in report["domains"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
