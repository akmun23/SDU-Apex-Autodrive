#!/usr/bin/env python3
"""Locate how held-out state error grows during command-only rollouts.

Each window is initialized from measured state and causal history once. Future
inputs are only the recorded steering/throttle commands. Truth is used solely
for scoring. Errors are summarized per independent source run at each 25 ms
step, rather than treating correlated frames as independent samples.
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


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/" \
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002"
DATA_ROOT = TASK_ROOT / "encoder_raw_state_teacher_v1"
DEFAULT_CHECKPOINT = DATA_ROOT / \
    "edssm_gru_z32_e2_rollresidual_10s_yawonly_wheelonly_lowthrottle4_joint_lr3e5_seed101/best.pt"
DEFAULT_DYNAMIC_DATASET = DATA_ROOT / "openplane_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE_DATASET = DATA_ROOT / "practice_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE_COMPARISON = DATA_ROOT / \
    "paired_contact_slip_vs_parent_v1/paired_comparison.json"
DEFAULT_OUTPUT = DATA_ROOT / "recursive_first_divergence_wp14_parent_v1.json"
HORIZON_SNAPSHOTS = (1, 4, 10, 20, 30, 40, 80, 120, 200, 400)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_crossing_step(curve: np.ndarray, threshold: float) -> int | None:
    """Return the 1-based first step above a diagnostic threshold."""
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("crossing input must be a finite threshold and 1-D curve")
    crossings = np.flatnonzero(values > threshold)
    return int(crossings[0] + 1) if crossings.size else None


def _per_run_curves(error: np.ndarray, run_indices: np.ndarray,
                    run_ids: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """Compute per-run pointwise RMSE/bias and equally weighted run macros."""
    error = np.asarray(error, dtype=np.float64)
    run_indices = np.asarray(run_indices, dtype=np.int64)
    if error.ndim != 2 or len(error) != len(run_indices):
        raise ValueError("error must have shape (windows, steps) with run IDs")
    rmse: dict[str, np.ndarray] = {}
    bias: dict[str, np.ndarray] = {}
    for run in sorted(set(run_indices.tolist())):
        values = error[run_indices == run]
        if len(values):
            rmse[str(run_ids[run])] = np.sqrt(np.mean(values ** 2, axis=0))
            bias[str(run_ids[run])] = np.mean(values, axis=0)
    if not rmse:
        raise ValueError("no run-level errors to summarize")
    run_rmse = np.stack(list(rmse.values()))
    run_bias = np.stack(list(bias.values()))
    return {
        "per_run_rmse": {name: values.tolist()
                         for name, values in sorted(rmse.items())},
        "per_run_bias": {name: values.tolist()
                         for name, values in sorted(bias.items())},
        "macro_run_rmse": np.mean(run_rmse, axis=0).tolist(),
        "macro_run_bias": np.mean(run_bias, axis=0).tolist(),
        "run_min_rmse": np.min(run_rmse, axis=0).tolist(),
        "run_max_rmse": np.max(run_rmse, axis=0).tolist(),
    }


def _score_domain(torch, model, metadata: dict[str, Any], data: dict[str, Any],
                  split: str, device: str, max_windows_per_run: int,
                  batch_size: int, horizon_steps: int,
                  fixed_windows: list[dict[str, Any]] | None = None
                  ) -> dict[str, Any]:
    if not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("recursive divergence analysis requires exact 25 ms data")
    wheel_source = str(metadata.get("wheel_state_source", "filtered_odometry"))
    base_state = physical_state_from_dataset(data, wheel_state_source=wheel_source)
    state = append_roll_state(data, base_state) if model.include_roll_state else base_state
    windows = fixed_windows or _fixed_validation_windows(
        data, state, max_windows_per_run=max_windows_per_run,
        horizon_steps=horizon_steps, max_throttle_command=0.50, split=split)
    if any(int(row["start"]) < HISTORY_STEPS - 1
           or int(row["start"]) + horizon_steps >= len(state)
           for row in windows):
        raise ValueError("frozen evaluation window is outside dataset bounds")
    raw_history = (
        raw_encoder_history_features(data) if model.include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if model.include_wheel_innovation_history else None)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    metric_names = ("forward_speed_mps", "steering_feedback_rad",
                    "throttle_feedback_norm", "rear_left_wheel_mps",
                    "rear_right_wheel_mps", "rear_wheel_pair_rms_mps",
                    "yaw_rate_rps", "lateral_speed_mps",
                    "position_radial_m", "heading_rad")
    if model.include_roll_state:
        metric_names += ("roll_rad", "roll_rate_rps")
    metrics_by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in metric_names}
    run_indices: list[int] = []
    eligible_windows: list[dict[str, Any]] = []
    predicted_count = 0
    for offset in range(0, len(windows), batch_size):
        rows = windows[offset:offset + batch_size]
        history, initial, delayed, commands, truth, truth_pose = _window_batch(
            data, state, rows, model.history_state_size, raw_history,
            horizon_steps=horizon_steps)
        with torch.no_grad():
            predicted, _, _, _ = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(commands, dtype=torch.float32, device=device))
            predicted_pose = integrate_pose(
                torch, predicted,
                torch.as_tensor(truth_pose[:, 0], dtype=torch.float32,
                                device=device),
                torch.as_tensor(initial, dtype=torch.float32, device=device))
        pred = predicted.cpu().numpy().astype(np.float64)
        pred_pose = predicted_pose.cpu().numpy().astype(np.float64)
        truth = truth.astype(np.float64)
        target_pose = truth_pose[:, 1:].astype(np.float64)
        full_delta = pred - truth
        delta = full_delta[:, :, :7]
        position_error = pred_pose[:, :, :2] - target_pose[:, :, :2]
        heading_error = np.arctan2(
            np.sin(pred_pose[:, :, 2] - target_pose[:, :, 2]),
            np.cos(pred_pose[:, :, 2] - target_pose[:, :, 2]))
        batch_values = {
            "forward_speed_mps": delta[:, :, 0],
            "steering_feedback_rad": delta[:, :, 3],
            "throttle_feedback_norm": delta[:, :, 4],
            "rear_left_wheel_mps": delta[:, :, 5],
            "rear_right_wheel_mps": delta[:, :, 6],
            "rear_wheel_pair_rms_mps": np.sqrt(
                np.mean(delta[:, :, 5:7] ** 2, axis=2)),
            "yaw_rate_rps": delta[:, :, 2],
            "lateral_speed_mps": delta[:, :, 1],
            "position_radial_m": np.linalg.norm(position_error, axis=2),
            "heading_rad": heading_error,
        }
        if model.include_roll_state:
            batch_values["roll_rad"] = full_delta[:, :, 7]
            batch_values["roll_rate_rps"] = full_delta[:, :, 8]
        for name, values in batch_values.items():
            metrics_by_name[name].append(values)
        run_indices.extend(int(row["run"]) for row in rows)
        eligible_windows.extend(rows)
        predicted_count += int(pred.shape[0] * pred.shape[1])

    run_index_array = np.asarray(run_indices, dtype=np.int64)
    curves: dict[str, Any] = {}
    snapshots = sorted({step for step in HORIZON_SNAPSHOTS
                        if step <= horizon_steps})
    for name, batches in metrics_by_name.items():
        all_errors = np.concatenate(batches, axis=0)
        aggregate = _per_run_curves(
            all_errors, run_index_array, run_ids)
        macro = np.asarray(aggregate["macro_run_rmse"], dtype=np.float64)
        first_rmse = float(macro[0])
        crossing = _first_crossing_step(macro, 2.0 * first_rmse)
        aggregate["first_step_rmse"] = first_rmse
        aggregate["first_2x_rmse_step"] = crossing
        aggregate["first_2x_rmse_time_s"] = (
            None if crossing is None else crossing * DT_S)
        aggregate["snapshots"] = {
            f"{step * DT_S:.3f}s": {
                "step": step,
                "macro_run_rmse": float(macro[step - 1]),
                "per_run_rmse": {
                    run: values[step - 1]
                    for run, values in aggregate["per_run_rmse"].items()},
            }
            for step in snapshots
        }
        cumulative: dict[str, Any] = {}
        for step in snapshots:
            per_run_horizon = {}
            for run in sorted(set(run_index_array.tolist())):
                selected = run_index_array == run
                per_run_horizon[str(run_ids[run])] = float(np.sqrt(np.mean(
                    all_errors[selected, :step] ** 2)))
            cumulative[f"{step * DT_S:.3f}s"] = {
                "step": step,
                "macro_run_rmse": float(np.mean(list(per_run_horizon.values()))),
                "per_run_rmse": per_run_horizon,
            }
        aggregate["cumulative_horizon_rmse"] = cumulative
        curves[name] = aggregate
    return {
        "whole_run_ids": sorted({str(run_ids[index])
                                  for index in run_index_array}),
        "whole_run_count": int(len(set(run_index_array.tolist()))),
        "window_count": len(eligible_windows),
        "horizon_steps": horizon_steps,
        "horizon_seconds": horizon_steps * DT_S,
        "predicted_state_steps": predicted_count,
        "future_inputs_only": ["steering_command_rad", "throttle_command_norm"],
        "future_truth_or_sensor_feedback_used": False,
        "initialization": "measured state plus causal 80-frame history once per window; no future measured-state feedback",
        "error_curves": curves,
    }


def diagnose(checkpoint: Path, dynamic_dataset: Path, practice_dataset: Path,
             output: Path, practice_comparison: Path = DEFAULT_PRACTICE_COMPARISON,
             device: str = "cpu", max_windows_per_run: int = 24,
             batch_size: int = 8, horizon_steps: int = 200) -> dict[str, Any]:
    paths = tuple(path.resolve() for path in
                  (checkpoint, dynamic_dataset, practice_dataset, output,
                   practice_comparison))
    (checkpoint, dynamic_dataset, practice_dataset, output,
     practice_comparison) = paths
    if max_windows_per_run <= 0 or batch_size <= 0 or horizon_steps != 200:
        raise ValueError("positive window/batch sizes and the frozen 200-step horizon are required")
    torch, model, metadata = _load_model(checkpoint, device)
    practice_report = json.loads(practice_comparison.read_text(encoding="utf-8"))
    if (practice_report.get("practice_dataset_sha256")
            != _sha256(practice_dataset)
            or practice_report.get("future_truth_or_feedback_used") is not False
            or not practice_report.get("selection_was_frozen_before_practice_transfer")):
        raise ValueError("practice start manifest does not match the frozen benchmark")
    practice_data = _load_dataset(practice_dataset)
    run_lookup = {str(name): index for index, name in
                  enumerate(np.asarray(practice_data["run_ids"]).astype(str))}
    practice_windows = []
    for row in practice_report["production_practice_transfer"]["wp14_parent"][
            "window_manifest"]:
        run_name = str(row["run_id"])
        if run_name not in run_lookup:
            raise ValueError(f"frozen practice manifest has unknown run {run_name}")
        practice_windows.append({
            "start": int(row["start"]),
            "run": run_lookup[run_name],
            "sequence_id": int(row["sequence_id"]),
            "regimes": list(row["regimes"]),
        })
    if not practice_windows:
        raise ValueError("frozen practice start manifest is empty")
    results = {}
    for name, dataset_path, data, split, fixed_windows in (
            ("dynamic_validation", dynamic_dataset,
             _load_dataset(dynamic_dataset), "validation", None),
            ("practice_unseen", practice_dataset, practice_data,
             "unseen_practice", practice_windows)):
        results[name] = {
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            **_score_domain(torch, model, metadata, data, split, device,
                            max_windows_per_run, batch_size, horizon_steps,
                            fixed_windows),
        }
    result = {
        "schema_version": 1,
        "purpose": "pointwise recursive error growth at every fixed 25 ms step",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_runs": metadata["training_runs"],
        "practice_comparison": str(practice_comparison),
        "practice_comparison_sha256": _sha256(practice_comparison),
        "cadence_s": DT_S,
        "test_and_final_test_used": False,
        "future_truth_or_feedback_used_for_prediction": False,
        "crossing_definition": "first pointwise macro-run RMSE greater than twice the 25 ms RMSE; descriptive only, not an acceptance gate",
        "domains": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dynamic-dataset", type=Path,
                        default=DEFAULT_DYNAMIC_DATASET)
    parser.add_argument("--practice-dataset", type=Path,
                        default=DEFAULT_PRACTICE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--practice-comparison", type=Path,
                        default=DEFAULT_PRACTICE_COMPARISON)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-windows-per-run", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, choices=(200,), default=200)
    args = parser.parse_args()
    result = diagnose(args.checkpoint, args.dynamic_dataset,
                      args.practice_dataset, args.output,
                      args.practice_comparison, args.device,
                      args.max_windows_per_run, args.batch_size,
                      args.horizon_steps)
    summary = {}
    for domain, report in result["domains"].items():
        summary[domain] = {
            "window_count": report["window_count"],
            "whole_run_ids": report["whole_run_ids"],
            "metrics": {name: {
                "first_step_rmse": curve["first_step_rmse"],
                "first_2x_rmse_time_s": curve["first_2x_rmse_time_s"],
                "rmse_at_0.1_0.5_1_2_5s": {
                    time: curve["snapshots"][time]["macro_run_rmse"]
                    for time in ("0.100s", "0.500s", "1.000s",
                                 "2.000s", "5.000s")
                    if time in curve["snapshots"]},
            } for name, curve in report["error_curves"].items()},
        }
    print(json.dumps({"output": str(args.output.resolve()),
                      "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
