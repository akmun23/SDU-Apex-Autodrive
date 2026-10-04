#!/usr/bin/env python3
"""Attribute EDSSM rollout error on frozen dynamic/practice validation runs.

This is a diagnostic only: it does not train models, read the blind WP14 bag,
or use future measurements as rollout inputs. Truth/IMU values are used only
after prediction to score and stratify residuals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.compare_effective_race_teacher import (
    _build_inputs,
    _registry_rssm_path,
    _rssm_rollouts,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    REAR_AXLE_TO_COM_M,
    append_roll_state,
    generalized_acceleration_targets,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _fixed_validation_windows,
    _load_model,
    _support_limits,
    _window_regimes,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928/"
             "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002")
TRAIN_DATASET = TASK_ROOT / "replacement_teacher_dataset_v1/openplane_dynamics.npz"
CANDIDATE = (TASK_ROOT / "edssm_training_20261002/"
             "edssm_gru_z32_e2_seed101/best.pt")
PRACTICE_BENCHMARK = ROOT / "live_runs/derived_dynamics_learning_20260928/"
PRACTICE_BENCHMARK = PRACTICE_BENCHMARK / "full_modeling_reset_20261001/practice_transfer_benchmark_v1.json"
REGISTRY = ROOT / "live_runs/derived_dynamics_learning_20260928/"
REGISTRY = REGISTRY / "full_modeling_reset_20261001/replacement_sim_baseline_registry_20261002.json"

HORIZON_MARKS = (1, 2, 4, 10, 20, 30, 40, 80, 120, 160, 200)
STATE_METRICS = {
    "u_com_mps": (0,),
    "v_com_mps": (1,),
    "yaw_rate_rps": (2,),
    "steering_feedback_rad": (3,),
    "throttle_feedback_norm": (4,),
    "rear_left_wheel_mps": (5,),
    "rear_right_wheel_mps": (6,),
}
BUCKETS = {
    "speed_mps": (np.asarray((0.0, 3.0, 5.0, 7.0, 9.0, 12.0001)),
                  ("0-3", "3-5", "5-7", "7-9", "9-12")),
    "abs_steering_rad": (np.asarray((0.0, 0.10, 0.20, 0.30, 0.40, 0.52361)),
                          ("0-.10", ".10-.20", ".20-.30", ".30-.40", ".40-.524")),
    "abs_steering_rate_rps": (np.asarray((0.0, 0.5, 2.0, 5.0, 10.0, 40.0)),
                               ("0-.5", ".5-2", "2-5", "5-10", "10-40")),
    "throttle_command": (
        np.asarray((0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00001)),
        ("0-.05", ".05-.10", ".10-.20", ".20-.35", ".35-.50",
         ".50-.75", ".75-1.00")),
    "abs_throttle_slew_per_s": (
        np.asarray((0.0, 0.25, 1.0, 3.0, 8.0, 22.0, 40.0001)),
        ("0-.25", ".25-1", "1-3", "3-8", "8-22", "22-40")),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _torch():
    import torch
    return torch


def _command_rows(source_rows: np.ndarray,
                  command_offset_frames: int) -> np.ndarray:
    """Map a source transition row to its aligned command row.

    Offset is defined relative to the target-state row: -1 selects the
    preceding command row; 0 selects the command row matching the target.
    """
    if command_offset_frames not in (-1, 0):
        raise ValueError("command alignment offset must be -1 or 0 frames")
    return np.asarray(source_rows, dtype=np.int64) + 1 + command_offset_frames


def _source_roll_states(initial_roll: np.ndarray,
                        predicted_roll: np.ndarray) -> np.ndarray:
    """Align causal internal roll state with each transition's source row."""
    initial_roll = np.asarray(initial_roll, dtype=np.float64)
    predicted_roll = np.asarray(predicted_roll, dtype=np.float64)
    if (predicted_roll.ndim != 3 or predicted_roll.shape[2] != 2
            or initial_roll.shape != (predicted_roll.shape[0], 2)
            or predicted_roll.shape[1] < 1):
        raise ValueError("roll states must align as (batch,time,2) with an initial state")
    source = np.empty_like(predicted_roll)
    source[:, 0] = initial_roll
    source[:, 1:] = predicted_roll[:, :-1]
    return source


def _practice_windows(benchmark: dict[str, Any], data: dict[str, Any],
                      horizon_steps: int = 200) -> list[dict[str, Any]]:
    """Use frozen starts that remain inside verified contiguous practice runs.

    Practice captures are split into packet-sequence bounds. A window may
    cross those bookkeeping boundaries only when run/reset identity, packet
    sequence, and the 25 ms sample interval prove the join is continuous.
    """
    if horizon_steps <= 0:
        raise ValueError("practice rollout horizon must be positive")
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    seq_runs = np.asarray(data["seq_run"], dtype=np.int32)
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    resets = data.get("sequence_reset_index")
    resets = (None if resets is None else np.asarray(resets, dtype=np.int32))
    packet = np.asarray(data["packet_sequence"], dtype=np.int64)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    groups_by_run: dict[int, list[tuple[int, int]]] = {}
    for run_index, split in enumerate(splits):
        if split != "unseen_practice":
            continue
        sequence_indices = np.flatnonzero(seq_runs == run_index)
        if not sequence_indices.size:
            continue
        sequence_indices = sorted(
            sequence_indices.tolist(), key=lambda idx: int(bounds[idx, 0]))
        group_start = int(bounds[sequence_indices[0], 0])
        previous_index = sequence_indices[0]
        previous_end = int(bounds[previous_index, 1])
        groups = []
        for sequence_index in sequence_indices[1:]:
            next_start, next_end = map(int, bounds[sequence_index])
            same_reset = (resets is None or
                          resets[previous_index] == resets[sequence_index])
            joined = (
                previous_end == next_start
                and seq_runs[previous_index] == seq_runs[sequence_index]
                and same_reset
                and packet[previous_end] - packet[previous_end - 1] == 1
                and np.isclose(dt[previous_end], DT_S, rtol=0.0, atol=1e-7))
            if not joined:
                groups.append((group_start, previous_end))
                group_start = next_start
            previous_index = sequence_index
            previous_end = next_end
        groups.append((group_start, previous_end))
        groups_by_run[run_index] = groups

    by_run_id = {str(run_id): index for index, run_id in enumerate(run_ids)}
    rows = []
    for item in benchmark["windows"]:
        run_id = str(item["run_id"])
        if run_id not in by_run_id:
            raise ValueError(f"practice benchmark references unknown run {run_id}")
        run_index = by_run_id[run_id]
        if splits[run_index] != "unseen_practice":
            raise ValueError(f"practice benchmark references non-practice run {run_id}")
        start = int(item["global_start_index"])
        context_start = start - HISTORY_STEPS + 1
        if any(context_start >= left and start + horizon_steps < right
               for left, right in groups_by_run.get(run_index, ())):
            rows.append({
                "run": run_index,
                "sequence_id": -1,
                "start": start,
                "regimes": list(item["categories"]),
                "run_id": run_id,
            })
    return rows


def _candidate_rollouts(torch, model, data: dict[str, Any], state: np.ndarray,
                        windows: list[dict[str, Any]], device: str,
                        command_offset_frames: int = 0,
                        batch_size: int = 24):
    pred_state, pred_accel, truth_state, source_indices, source_roll = (
        [], [], [], [], [])
    for offset in range(0, len(windows), batch_size):
        rows = windows[offset:offset + batch_size]
        rollout_state = (append_roll_state(data, state)
                         if model.include_roll_state and state.shape[1] == 7
                         else state)
        history, initial, delayed, commands, targets, _ = _build_inputs(
            data, rollout_state, rows, model.history_state_size,
            rollout_steps=200)
        source_rows = np.stack([
            np.arange(int(row["start"]), int(row["start"]) + 200,
                      dtype=np.int64)
            for row in rows])
        command_rows = _command_rows(source_rows, command_offset_frames)
        if np.any(command_rows < 0) or np.any(command_rows >= len(data["frames"])):
            raise ValueError("command alignment leaves the dataset")
        commands = np.asarray(data["frames"][command_rows, 7:9],
                              dtype=np.float32)
        with torch.no_grad():
            prediction, acceleration, _, _ = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(commands, dtype=torch.float32, device=device))
        prediction_np = prediction.cpu().numpy().astype(np.float64)
        pred_state.append(prediction_np)
        pred_accel.append(acceleration.cpu().numpy().astype(np.float64))
        truth_state.append(targets.astype(np.float64))
        if model.include_roll_state and prediction_np.shape[2] >= 9:
            starts = np.asarray([int(row["start"]) for row in rows],
                                dtype=np.int64)
            initial_roll = rollout_state[starts, 7:9].astype(np.float64)
            source_roll.append(_source_roll_states(
                initial_roll, prediction_np[:, :, 7:9]))
        else:
            source_roll.append(np.full((len(rows), 200, 2), np.nan))
        source_indices.extend(
            np.arange(int(row["start"]), int(row["start"]) + 200,
                      dtype=np.int64) for row in rows)
    return (np.concatenate(pred_state), np.concatenate(pred_accel),
            np.concatenate(truth_state), np.stack(source_indices),
            np.concatenate(source_roll))


def _macro_error(error: np.ndarray, run_index: np.ndarray,
                 run_ids: np.ndarray, selected: np.ndarray,
                 include_signed_bias: bool = True) -> dict[str, Any]:
    per_run = {}
    per_run_bias = {}
    for run in sorted(set(run_index[selected].tolist())):
        mask = selected & (run_index == run)
        values = error[mask]
        if values.size:
            per_run[str(run_ids[run])] = float(np.sqrt(np.mean(values ** 2)))
            if include_signed_bias:
                per_run_bias[str(run_ids[run])] = float(np.mean(values))
    result = {
        "independent_runs": len(per_run),
        "macro_run_rmse": float(np.mean(list(per_run.values()))) if per_run else None,
        "per_run_rmse": per_run,
        "transition_evaluations": int(selected.sum()),
    }
    if include_signed_bias:
        # Signed error is prediction minus truth. Macro-average by run so a
        # long capture cannot dominate the bias estimate.
        result.update({
            "macro_run_bias": (float(np.mean(list(per_run_bias.values())))
                               if per_run_bias else None),
            "per_run_bias": per_run_bias,
        })
    return result


def _curve_summary(pred: np.ndarray, truth: np.ndarray, windows,
                   run_ids: np.ndarray, domain: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    window_runs = np.asarray([int(row["run"]) for row in windows], dtype=np.int32)
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    if (pred.ndim != 3 or truth.ndim != 3
            or pred.shape[:2] != truth.shape[:2]
            or pred.shape[2] > truth.shape[2]):
        raise ValueError("residual curves need aligned rollout/time dimensions")
    # The production RSSM exposes seven motion/actuator channels; an EDSSM
    # with internal roll adds two targets. Compare only common channels for
    # the production baseline, while retaining all nine for the EDSSM score.
    errors = pred - truth[:, :, :pred.shape[2]]
    absolute_metrics = {
        **{name: errors[:, :, index] for name, (index,) in STATE_METRICS.items()},
        "rear_wheel_pair_rms_mps": np.sqrt(np.mean(errors[:, :, 5:7] ** 2, axis=2)),
        "body_speed_mps": (np.hypot(pred[:, :, 0], pred[:, :, 1])
                           - np.hypot(truth[:, :, 0], truth[:, :, 1])),
    }
    if pred.shape[2] >= 9:
        absolute_metrics["roll_angle_rad"] = errors[:, :, 7]
        absolute_metrics["roll_rate_rps"] = errors[:, :, 8]
    curves: dict[str, Any] = {}
    curve_rows = []
    for metric, residual in absolute_metrics.items():
        rows = []
        for step in range(residual.shape[1]):
            per_run = {}
            for run in sorted(set(window_runs.tolist())):
                selected_windows = window_runs == run
                values = residual[selected_windows, step]
                if values.size:
                    per_run[str(run_ids[run])] = float(
                        np.sqrt(np.mean(values ** 2)))
            macro = float(np.mean(list(per_run.values()))) if per_run else None
            rows.append({"step": step + 1, "time_s": (step + 1) * DT_S,
                         "macro_run_rmse": macro, "per_run_rmse": per_run})
            curve_rows.append({"domain": domain, "metric": metric,
                               "step": step + 1, "time_s": (step + 1) * DT_S,
                               "macro_run_rmse": macro})
        curves[metric] = {
            "sampled_steps": [rows[index - 1] for index in HORIZON_MARKS],
            "full_200_step_curve": rows,
        }
    return curves, curve_rows


def _source_covariates(data: dict[str, Any], state: np.ndarray,
                       source_indices: np.ndarray,
                       command_offset_frames: int = 0,
                       internal_roll: np.ndarray | None = None
                       ) -> dict[str, np.ndarray]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    attitude = data.get("imu_attitude_frames")
    attitude_valid = data.get("imu_attitude_valid")
    if attitude is None or attitude_valid is None:
        raise ValueError("held-out roll analysis requires IMU attitude labels")
    attitude = np.asarray(attitude, dtype=np.float64)
    attitude_valid = np.asarray(attitude_valid, dtype=bool)
    steering_rate = np.full(len(frames), np.nan)
    throttle_slew = np.full(len(frames), np.nan)
    for start_raw, end_raw in data["bounds"]:
        start, end = int(start_raw), int(end_raw)
        steering_rate[start + 1:end] = np.diff(state[start:end, 3]) / DT_S
        throttle_slew[start + 1:end] = np.diff(frames[start:end, 8]) / DT_S
    for boundary in _continuous_sequence_joins(data):
        steering_rate[boundary + 1] = (
            state[boundary + 1, 3] - state[boundary, 3]) / DT_S
        throttle_slew[boundary + 1] = (
            frames[boundary + 1, 8] - frames[boundary, 8]) / DT_S
    speed = np.hypot(state[:, 0], state[:, 1])
    signed_mismatch = 0.5 * (state[:, 5] + state[:, 6]) - state[:, 0]
    mismatch = np.abs(signed_mismatch)
    command_rows = _command_rows(source_indices, command_offset_frames)
    if np.any(command_rows < 0) or np.any(command_rows >= len(frames)):
        raise ValueError("command alignment leaves the dataset")
    if internal_roll is None:
        internal_roll = np.full((*source_indices.shape, 2), np.nan)
    else:
        internal_roll = np.asarray(internal_roll, dtype=np.float64)
        if internal_roll.shape != (*source_indices.shape, 2):
            raise ValueError("internal roll states do not align with transitions")
    return {
        "speed_mps": speed[source_indices],
        "abs_steering_rad": np.abs(state[source_indices, 3]),
        "abs_steering_rate_rps": np.abs(steering_rate[source_indices]),
        "throttle_command": frames[command_rows, 8],
        "abs_throttle_slew_per_s": np.abs(throttle_slew[command_rows]),
        "wheel_body_mismatch_mps": mismatch[source_indices],
        "signed_wheel_body_mismatch_mps": signed_mismatch[source_indices],
        "abs_roll_rad": np.abs(attitude[source_indices, 0]),
        "abs_roll_rate_rps": np.abs(attitude[source_indices, 2]),
        "abs_internal_roll_rad": np.abs(internal_roll[..., 0]),
        "abs_internal_roll_rate_rps": np.abs(internal_roll[..., 1]),
        "attitude_valid": attitude_valid[source_indices],
    }


def _continuous_sequence_joins(data: dict[str, Any]) -> list[int]:
    """Return source rows whose following frame is a verified same-run join."""
    joins = []
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    seq_runs = np.asarray(data["seq_run"], dtype=np.int32)
    seq_resets = data.get("sequence_reset_index")
    seq_resets = (None if seq_resets is None else
                  np.asarray(seq_resets, dtype=np.int32))
    packet = np.asarray(data["packet_sequence"], dtype=np.int64)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    for index in range(len(bounds) - 1):
        end = int(bounds[index, 1])
        following_start = int(bounds[index + 1, 0])
        same_reset = (seq_resets is None
                      or seq_resets[index] == seq_resets[index + 1])
        if (end == following_start
                and seq_runs[index] == seq_runs[index + 1]
                and same_reset
                and packet[end] - packet[end - 1] == 1
                and np.isclose(dt[end], DT_S, rtol=0.0, atol=1e-7)):
            joins.append(end - 1)
    return joins


def _acceleration_targets_with_contiguous_joins(
        data: dict[str, Any], state: np.ndarray) -> np.ndarray:
    targets = generalized_acceleration_targets(
        state, data["bounds"], data["dt_s"])
    for source in _continuous_sequence_joins(data):
        current, following = state[source], state[source + 1]
        derivative = (following[:3] - current[:3]) / DT_S
        targets[source, 0] = derivative[0] - current[2] * current[1]
        targets[source, 1] = derivative[1] + current[2] * current[0]
        targets[source, 2] = derivative[2]
        targets[source, 3:5] = (following[5:7] - current[5:7]) / DT_S
    return targets


def _train_cutpoints(train_data: dict[str, Any], train_state: np.ndarray):
    indices = []
    splits = np.asarray(train_data["splits"]).astype(str)
    for (start_raw, end_raw), run_raw in zip(train_data["bounds"],
                                              train_data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] == "train":
            indices.append(np.arange(start, end - 1, dtype=np.int64))
    if not indices:
        raise ValueError("no training transitions for diagnostic cutpoints")
    cov = _source_covariates(train_data, train_state,
                             np.concatenate(indices))
    result = {}
    for key in ("wheel_body_mismatch_mps",
                "signed_wheel_body_mismatch_mps",
                "abs_roll_rad", "abs_roll_rate_rps"):
        valid = np.isfinite(cov[key])
        if key.startswith("abs_roll"):
            valid &= cov["attitude_valid"]
        quantiles = np.quantile(
            cov[key][valid], ((0.10, 0.50, 0.90)
                              if key == "signed_wheel_body_mismatch_mps"
                              else (0.50, 0.90, 0.95)))
        edges = np.unique(quantiles)
        if len(edges) < 1:
            raise ValueError(f"no training cutpoints for {key}")
        if key == "signed_wheel_body_mismatch_mps":
            labels = [f"<={edges[0]:.6g}"]
            labels.extend(
                f"{edges[index]:.6g}..{edges[index + 1]:.6g}"
                for index in range(len(edges) - 1))
            labels.append(f">{edges[-1]:.6g}")
        else:
            labels = []
            previous = 0.0
            for edge in edges:
                labels.append(f"{previous:.6g}-{edge:.6g}")
                previous = float(edge)
            labels.append(f">={previous:.6g}")
        result[key] = (np.concatenate(([-np.inf], edges, [np.inf])), tuple(labels))
    return result


def _make_group_report(data, state, pred, truth, windows, source_indices,
                       domain, train_cutpoints):
    error = pred - truth
    run_index = np.asarray([int(row["run"]) for row in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    cov = _source_covariates(data, state, source_indices)
    covariates = {}
    for key, (edges, labels) in {**BUCKETS, **train_cutpoints}.items():
        matrix = cov[key]
        if matrix.shape != (len(windows), 200):
            raise ValueError(f"{key} covariates do not align with rollout transitions")
        groups = {}
        for index, label in enumerate(labels):
            lo, hi = edges[index], edges[index + 1]
            selected = np.isfinite(matrix) & (matrix >= lo) & (matrix < hi)
            if key.startswith("abs_roll"):
                selected &= cov["attitude_valid"]
            if not np.any(selected):
                continue
            wheel_error = np.sqrt(np.mean(error[:, :, 5:7] ** 2, axis=2))
            fields = {
                "u_com_mps": error[:, :, 0],
                "steering_feedback_rad": error[:, :, 3],
                "throttle_feedback_norm": error[:, :, 4],
                "rear_left_wheel_mps": error[:, :, 5],
                "rear_right_wheel_mps": error[:, :, 6],
                "wheel_pair_rms_mps": wheel_error,
                "yaw_rate_rps": error[:, :, 2],
                "body_speed_mps": (np.hypot(pred[:, :, 0], pred[:, :, 1])
                                    - np.hypot(truth[:, :, 0], truth[:, :, 1])),
            }
            groups[label] = {
                "transition_evaluations": int(selected.sum()),
                "independent_whole_runs": int(len(set(run_index[
                    np.any(selected, axis=1)].tolist()))),
                "metrics": {
                    metric: _macro_error(
                        values, np.repeat(run_index, 200), run_ids,
                        selected.reshape(-1),
                        include_signed_bias=metric != "wheel_pair_rms_mps")
                    for metric, values in ((name, values.reshape(-1))
                                           for name, values in fields.items())
                },
                "at_rollout_time": {},
            }
            # Compare the same factor-conditioned state errors at selected
            # recursive rollout steps. Pooling every rollout time together
            # obscures whether an error is present immediately or accumulates.
            for step in HORIZON_MARKS:
                keep = selected[:, step - 1]
                groups[label]["at_rollout_time"][f"{step * DT_S:.3f}s"] = {
                    metric: _macro_error(
                        values[:, step - 1], run_index, run_ids, keep,
                        include_signed_bias=metric != "wheel_pair_rms_mps")
                    for metric, values in fields.items()
                }
        covariates[key] = groups
    # The large practice speed deficit is associated with both low throttle
    # and wheel/body mismatch. Preserve interactions with both mismatch
    # magnitude and direction so marginal bins do not hide drive/coast regimes.
    throttle_edges, throttle_labels = BUCKETS["throttle_command"]
    body_speed_error = (np.hypot(pred[:, :, 0], pred[:, :, 1])
                        - np.hypot(truth[:, :, 0], truth[:, :, 1]))
    interaction_fields = {
        "body_speed_mps": body_speed_error,
        "u_com_mps": error[:, :, 0],
        "rear_left_wheel_mps": error[:, :, 5],
        "rear_right_wheel_mps": error[:, :, 6],
        "throttle_feedback_norm": error[:, :, 4],
    }
    throttle = cov["throttle_command"]

    def throttle_mismatch_interaction(mismatch_key: str) -> dict[str, Any]:
        mismatch_edges, mismatch_labels = train_cutpoints[mismatch_key]
        mismatch = cov[mismatch_key]
        interaction = {}
        for throttle_index, throttle_label in enumerate(throttle_labels):
            throttle_selected = (
                (throttle >= throttle_edges[throttle_index])
                & (throttle < throttle_edges[throttle_index + 1]))
            throttle_groups = {}
            for mismatch_index, mismatch_label in enumerate(mismatch_labels):
                selected = (throttle_selected
                            & (mismatch >= mismatch_edges[mismatch_index])
                            & (mismatch < mismatch_edges[mismatch_index + 1]))
                if not np.any(selected):
                    continue
                values_by_time = {}
                for step in HORIZON_MARKS:
                    keep = selected[:, step - 1]
                    values_by_time[f"{step * DT_S:.3f}s"] = {
                        "transition_evaluations": int(keep.sum()),
                        "independent_whole_runs": int(len(set(
                            run_index[keep].tolist()))),
                        "metrics": {
                            name: _macro_error(
                                values[:, step - 1], run_index, run_ids, keep)
                            for name, values in interaction_fields.items()},
                    }
                throttle_groups[mismatch_label] = {
                    "transition_evaluations": int(selected.sum()),
                    "at_rollout_time": values_by_time,
                }
            if throttle_groups:
                interaction[throttle_label] = throttle_groups
        return interaction

    return {
        "single_factor": covariates,
        "throttle_x_wheel_body_mismatch": throttle_mismatch_interaction(
            "wheel_body_mismatch_mps"),
        "throttle_x_signed_wheel_body_mismatch": throttle_mismatch_interaction(
            "signed_wheel_body_mismatch_mps"),
        "interaction_threshold_source": (
            "training-run quantiles; magnitude p50/p90/p95, signed p10/p50/p90"),
    }


def _roll_incremental_report(acceleration_residual: np.ndarray,
                             covariates: dict[str, np.ndarray], windows,
                             run_ids: np.ndarray) -> dict[str, Any]:
    """Compare measured versus causal model-predicted roll as residual features.

    Neither residual regressor feeds measurements back into the plant. The
    measured-roll arm is post-hoc explanatory only; the internal-roll arm uses
    the model's own source-state roll at each step, initialized from the
    measured history and predicted thereafter.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for roll attribution") from exc
    required = (
        "speed_mps", "abs_steering_rad", "abs_steering_rate_rps",
        "throttle_command", "abs_throttle_slew_per_s",
        "wheel_body_mismatch_mps", "abs_roll_rad", "abs_roll_rate_rps",
        "abs_internal_roll_rad", "abs_internal_roll_rate_rps",
        "attitude_valid")
    if any(key not in covariates for key in required):
        raise ValueError("roll attribution is missing aligned covariates")
    window_runs = np.asarray([int(row["run"]) for row in windows],
                             dtype=np.int32)
    horizon = np.broadcast_to(
        (np.arange(1, acceleration_residual.shape[1] + 1, dtype=np.float64)
         * DT_S)[None, :], acceleration_residual.shape[:2])
    base = np.stack((covariates["speed_mps"],
                     covariates["abs_steering_rad"],
                     covariates["abs_steering_rate_rps"],
                     covariates["throttle_command"],
                     covariates["abs_throttle_slew_per_s"],
                     covariates["wheel_body_mismatch_mps"],
                     horizon), axis=-1).reshape(-1, 7)
    feature_sets = {
        "measured_roll_posthoc": np.stack((
            covariates["abs_roll_rad"],
            covariates["abs_roll_rate_rps"]), axis=-1).reshape(-1, 2),
        "causal_internal_roll": np.stack((
            covariates["abs_internal_roll_rad"],
            covariates["abs_internal_roll_rate_rps"]), axis=-1).reshape(-1, 2),
    }
    base_valid = (np.isfinite(base).all(axis=1)
                  & covariates["attitude_valid"].reshape(-1))
    run_rows = np.repeat(window_runs, acceleration_residual.shape[1])
    targets = {
        "abs_ax_error_log1p": np.log1p(np.abs(
            acceleration_residual[:, :, 0])).reshape(-1),
        "abs_ay_error_log1p": np.log1p(np.abs(
            acceleration_residual[:, :, 1])).reshape(-1),
        "abs_yaw_accel_error_log1p": np.log1p(np.abs(
            acceleration_residual[:, :, 2])).reshape(-1),
        "wheel_accel_error_log1p": np.log1p(np.sqrt(np.mean(
            acceleration_residual[:, :, 3:5] ** 2, axis=2))).reshape(-1),
    }

    def model():
        return HistGradientBoostingRegressor(
            max_iter=40, max_leaf_nodes=7, min_samples_leaf=100,
            learning_rate=0.08, l2_regularization=5.0,
            random_state=20261002)

    report: dict[str, Any] = {
        "method": "leave-one-whole-run-out HistGradientBoostingRegressor on log1p absolute acceleration residuals",
        "base_features": [
            "speed", "abs steering", "abs steering rate", "throttle command",
            "abs throttle slew", "rear-wheel/body-speed mismatch", "rollout horizon"],
        "feature_arms": {
            "measured_roll_posthoc": ["abs measured roll", "abs measured roll rate"],
            "causal_internal_roll": ["abs model-predicted source roll",
                                     "abs model-predicted source roll rate"],
        },
        "measured_attitude_used_as_vehicle_rollout_input": False,
        "internal_roll_source": (
            "measured roll in the initial 80-frame history; thereafter the "
            "model's own predicted state at each transition source"),
        "valid_transition_evaluations": int(base_valid.sum()),
        "targets": {},
    }
    for target_name, target in targets.items():
        arms = {}
        for arm_name, roll in feature_sets.items():
            valid = base_valid & np.isfinite(roll).all(axis=1)
            arm_base, arm_roll, arm_target = base[valid], roll[valid], target[valid]
            arm_runs = run_rows[valid]
            per_run: dict[str, dict[str, float]] = {}
            for run in sorted(np.unique(arm_runs)):
                test = arm_runs == run
                train = ~test
                if (np.count_nonzero(train) < 500
                        or np.count_nonzero(test) < 100):
                    continue
                baseline, augmented = model(), model()
                baseline.fit(arm_base[train], arm_target[train])
                augmented.fit(np.column_stack((arm_base, arm_roll))[train],
                              arm_target[train])
                base_rmse = float(np.sqrt(np.mean(
                    (baseline.predict(arm_base[test]) - arm_target[test]) ** 2)))
                roll_rmse = float(np.sqrt(np.mean(
                    (augmented.predict(np.column_stack((arm_base, arm_roll))[test])
                     - arm_target[test]) ** 2)))
                per_run[str(run_ids[run])] = {
                    "base_rmse_log1p": base_rmse,
                    "with_roll_rmse_log1p": roll_rmse,
                    "relative_change": roll_rmse / max(base_rmse, 1e-12) - 1.0,
                    "heldout_transitions": int(np.count_nonzero(test)),
                }
            changes = [value["relative_change"] for value in per_run.values()]
            arms[arm_name] = {
                "per_run": per_run,
                "independent_runs": len(per_run),
                "macro_relative_change": (float(np.mean(changes))
                                          if changes else None),
            }
        report["targets"][target_name] = {
            **arms,
            "interpretation": (
                "negative held-out change supports residual information from that roll source; "
                "only causal_internal_roll is available to a free-running plant"),
        }
    return report


def _evaluate_domain(name, data, state, windows, torch, model, device,
                     rssm_checkpoint, train_dataset_path, train_cutpoints,
                     command_offset_frames: int):
    if int(data["schema_version"]) < 5:
        raise ValueError(f"{name} dataset lacks simulator pose targets")
    if not windows:
        raise ValueError(f"no scoring windows for {name}")
    state = np.asarray(state, dtype=np.float32)
    predicted, predicted_acceleration, truth, source_indices, internal_roll = _candidate_rollouts(
        torch, model, data, state, windows, device,
        command_offset_frames=command_offset_frames)
    rssm_state, _ = _rssm_rollouts(
        rssm_checkpoint, train_dataset_path, data, windows, device,
        rollout_steps=200,
        command_offset_frames=command_offset_frames)
    # Convert the frozen RSSM's rear-axle lateral velocity to COM before
    # scoring it alongside the candidate and rigid-state labels.
    rssm_state[:, :, 1] += REAR_AXLE_TO_COM_M * rssm_state[:, :, 2]
    window_runs = np.asarray([int(row["run"]) for row in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    curves, curve_rows = _curve_summary(predicted, truth, windows, run_ids, name)
    base_curves, base_rows = _curve_summary(rssm_state, truth, windows, run_ids,
                                             name + "_rssm")
    acceleration_labels = _acceleration_targets_with_contiguous_joins(
        data, state)[source_indices]
    if not np.isfinite(acceleration_labels).all():
        raise ValueError("scored windows contain incomplete acceleration labels")
    acceleration_residual = predicted_acceleration - acceleration_labels
    domain_covariates = _source_covariates(
        data, state, source_indices, command_offset_frames, internal_roll)
    accel_names = (
        "ax_effective_mps2", "ay_effective_mps2", "yaw_accel_rps2",
        "rear_left_surface_accel_mps2", "rear_right_surface_accel_mps2")
    acceleration_metrics = {}
    for name_index, metric in enumerate(accel_names):
        per_horizon = {}
        for label, steps in (("0.75s", 30), ("2s", 80), ("5s", 200)):
            per_horizon[label] = _macro_error(
                acceleration_residual[:, :steps, name_index].reshape(-1),
                np.repeat(window_runs, steps), run_ids,
                np.ones(len(windows) * steps, dtype=bool))
        acceleration_metrics[metric] = per_horizon
    acceleration_at_rollout_step = {}
    for name_index, metric in enumerate(accel_names):
        acceleration_at_rollout_step[metric] = {
            f"{step * DT_S:.3f}s": _macro_error(
                acceleration_residual[:, step - 1, name_index],
                window_runs, run_ids,
                np.ones(len(windows), dtype=bool))
            for step in HORIZON_MARKS
        }
    acceleration_by_source = {
        "ax_effective_mps2": acceleration_residual[:, :, 0],
        "ay_effective_mps2": acceleration_residual[:, :, 1],
        "yaw_accel_rps2": acceleration_residual[:, :, 2],
        "rear_wheel_pair_accel_rms_mps2": np.sqrt(np.mean(
            acceleration_residual[:, :, 3:5] ** 2, axis=2)),
    }
    accel_strata = {}
    for covariate, (edges, labels) in {**BUCKETS, **train_cutpoints}.items():
        values = domain_covariates[covariate]
        buckets = {}
        for index, label in enumerate(labels):
            selected = np.isfinite(values) & (values >= edges[index]) & (
                values < edges[index + 1])
            if covariate.startswith("abs_roll"):
                selected &= domain_covariates["attitude_valid"]
            if not np.any(selected):
                continue
            buckets[label] = {
                "transition_evaluations": int(selected.sum()),
                "metrics": {
                    metric: _macro_error(
                        residual.reshape(-1), np.repeat(window_runs, 200),
                        run_ids, selected.reshape(-1),
                        include_signed_bias=(
                            metric != "rear_wheel_pair_accel_rms_mps2"))
                    for metric, residual in acceleration_by_source.items()},
                "at_rollout_time": {
                    f"{step * DT_S:.3f}s": {
                        metric: _macro_error(
                            residual[:, step - 1], window_runs, run_ids,
                            selected[:, step - 1],
                            include_signed_bias=(
                                metric != "rear_wheel_pair_accel_rms_mps2"))
                        for metric, residual in acceleration_by_source.items()
                    }
                    for step in HORIZON_MARKS
                },
            }
        accel_strata[covariate] = buckets
    return {
        "domain": name,
        "schema_version": int(data["schema_version"]),
        "run_ids": sorted(set(str(run_ids[index]) for index in window_runs)),
        "window_count": len(windows),
        "independent_run_count": int(len(set(window_runs.tolist()))),
        "windows": [{"run_id": str(run_ids[int(row["run"])]),
                     "sequence_id": int(row["sequence_id"]),
                     "start": int(row["start"]),
                     "regimes": row.get("regimes", [])} for row in windows],
        "candidate_per_step": curves,
        "lead_rssm_per_step": base_curves,
        "stratified_candidate_errors": _make_group_report(
            data, state, predicted, truth, windows, source_indices, name,
            train_cutpoints),
        "candidate_acceleration_error": acceleration_metrics,
        "candidate_acceleration_error_at_rollout_step":
            acceleration_at_rollout_step,
        "candidate_acceleration_error_by_condition": accel_strata,
        "roll_incremental_predictive_value": _roll_incremental_report(
            acceleration_residual, domain_covariates, windows, run_ids),
        "acceleration_label_rows": int(np.isfinite(acceleration_labels).all(axis=2).sum()),
        "per_step_csv_rows": curve_rows + base_rows,
    }


def diagnose(candidate_path: Path, output_dir: Path, device: str,
             dynamic_windows_per_run: int = 24,
             dataset_path: Path = TRAIN_DATASET,
             practice_benchmark_path: Path = PRACTICE_BENCHMARK,
             max_throttle_command: float | None = None,
             command_offset_frames: int = 0
             ) -> dict[str, Any]:
    if command_offset_frames not in (-1, 0):
        raise ValueError("command alignment offset must be -1 or 0 frames")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite diagnostics: {output_dir}")
    if dynamic_windows_per_run <= 0:
        raise ValueError("dynamic validation window limit must be positive")
    dataset_path = dataset_path.resolve()
    practice_benchmark_path = practice_benchmark_path.resolve()
    train_data = _load_dataset(dataset_path)
    if int(train_data["schema_version"]) != 9:
        raise ValueError("diagnostics require frozen schema-9 teacher data")
    torch, model, metadata = _load_model(candidate_path, device)
    if metadata["dataset_sha256"] != _sha256(dataset_path):
        raise ValueError("candidate and immutable training dataset hashes disagree")
    if set(metadata["training_runs"]).intersection(
            np.asarray(train_data["run_ids"])[
                np.asarray(train_data["splits"]).astype(str) == "validation"].astype(str)):
        raise ValueError("candidate training runs overlap dynamic validation")
    train_state = physical_state_from_dataset(
        train_data, wheel_state_source=str(metadata.get(
            "wheel_state_source", "filtered_odometry"))).astype(np.float32)
    cutpoints = _train_cutpoints(train_data, train_state)
    val_windows = _fixed_validation_windows(train_data, train_state,
                                           max_windows_per_run=dynamic_windows_per_run,
                                           max_throttle_command=(
                                               float(metadata.get(
                                                   "max_throttle_command_norm", 0.50))
                                               if max_throttle_command is None
                                               else max_throttle_command))
    dynamic = _evaluate_domain(
        "dynamic_validation", train_data, train_state, val_windows, torch, model, device,
        _registry_rssm_path(REGISTRY), TRAIN_DATASET, cutpoints,
        command_offset_frames)

    benchmark = json.loads(
        practice_benchmark_path.read_text(encoding="utf-8"))
    practice_path = Path(str(benchmark["dataset"]).replace(
        "/workspace/", str(ROOT) + "/"))
    if _sha256(practice_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen practice benchmark dataset hash changed")
    practice_data = _load_dataset(practice_path)
    practice_state = physical_state_from_dataset(
        practice_data, wheel_state_source=str(metadata.get(
            "wheel_state_source", "filtered_odometry"))).astype(np.float32)
    practice_windows = _practice_windows(benchmark, practice_data)
    practice = _evaluate_domain(
        "practice_transfer", practice_data, practice_state, practice_windows, torch, model,
        device, _registry_rssm_path(REGISTRY), TRAIN_DATASET, cutpoints,
        command_offset_frames)
    report = {
        "schema_version": 1,
        "purpose": "held-out rollout residual attribution; diagnostic only",
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": _sha256(candidate_path),
        "candidate_training_runs": metadata["training_runs"],
        "candidate_validation_runs": metadata["validation_runs"],
        "training_dataset": str(dataset_path),
        "training_dataset_sha256": _sha256(dataset_path),
        "practice_benchmark": str(practice_benchmark_path),
        "practice_benchmark_sha256": _sha256(practice_benchmark_path),
        "test_and_final_test_used": False,
        "future_truth_or_sensor_feedback_used_in_rollout": False,
        "roll_and_rates_used_only_for_posthoc_stratification": True,
        "max_throttle_command_norm": float(
            metadata.get("max_throttle_command_norm", 0.50)
            if max_throttle_command is None else max_throttle_command),
        "command_offset_frames_from_target_state_row": int(
            command_offset_frames),
        "command_alignment_note": (
            "-1 selects the command stored with the preceding frame; 0 selects "
            "the command stored with the target-state frame"),
        "dynamic_windows_per_run_limit": int(dynamic_windows_per_run),
        "training_derived_cutpoints": {
            key: {"thresholds": edges[1:-1].tolist(),
                  "labels": list(labels)}
            for key, (edges, labels) in cutpoints.items()},
        "domains": {"dynamic_validation": dynamic,
                    "practice_transfer": practice},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "residual_attribution.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    with (output_dir / "per_step_rmse.csv").open("w", newline="",
                                                   encoding="utf-8") as stream:
        rows = dynamic["per_step_csv_rows"] + practice["per_step_csv_rows"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE,
                        help="schema-9 EDSSM trained on the frozen training runs")
    parser.add_argument("--dataset", type=Path, default=TRAIN_DATASET,
                        help="frozen schema-9 training/validation data matching the candidate hash")
    parser.add_argument("--practice-benchmark", type=Path,
                        default=PRACTICE_BENCHMARK,
                        help="held-out whole-practice-run window manifest")
    parser.add_argument("--max-throttle-command", type=float,
                        help="override checkpoint throttle cap for a broader diagnostic validation domain")
    parser.add_argument("--command-offset-frames", type=int, choices=(-1, 0),
                        default=0,
                        help="command row offset relative to target state")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dynamic-windows-per-run", type=int, default=24,
                        help="1 s spaced starts per whole dynamic validation run; use a large value for all supported starts")
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available()
                        else "cpu")
    args = parser.parse_args()
    result = diagnose(args.candidate.resolve(), args.output_dir.resolve(),
                      args.device, args.dynamic_windows_per_run,
                      args.dataset, args.practice_benchmark,
                      args.max_throttle_command,
                      args.command_offset_frames)
    print(json.dumps({"domains": {
        name: {"windows": item["window_count"],
               "runs": item["independent_run_count"]}
        for name, item in result["domains"].items()},
        "output": str(args.output_dir.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
