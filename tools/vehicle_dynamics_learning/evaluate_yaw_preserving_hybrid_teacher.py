#!/usr/bin/env python3
"""Evaluate a diagnostic parent-yaw / candidate-body parallel teacher blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    append_roll_state,
    integrate_pose,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.evaluate_effective_teacher_highsteer_transfer import (
    RUNS as HIGHSTEER_RUNS,
    _matched_windows as _matched_highsteer_windows,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _cluster_summary,
    _window_batch,
)
from tools.vehicle_dynamics_learning.score_full_practice_replay import (
    _contiguous_run_indices,
    _finish_gate,
    _lap_transition_indices,
    _predicted_crossings,
    _rollout,
    _score_one_run,
    _sha256,
)
from tools.vehicle_dynamics_learning.score_full_practice_replay import _metrics
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/practice_transfer_benchmark_v1.json")
DEFAULT_OUTPUT_DIR = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
    "next_phase_after_2129427/append_only_dynamic_extension_v1_20261004")
DEFAULT_HIGHSTEER_DATASET = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/highsteer_75_heldout_validation_plus_r03_20261004/"
    "openplane_dynamics.npz")


def _evaluate_highsteer(dataset_path: Path, parent_checkpoint: Path,
                        candidate_checkpoint: Path, device: str
                        ) -> dict[str, Any]:
    data, windows, _, _, _ = _matched_highsteer_windows(dataset_path)
    torch, parent_model, parent_metadata = _load_model(parent_checkpoint, device)
    _, candidate_model, candidate_metadata = _load_model(candidate_checkpoint, device)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    if (set(parent_metadata["training_runs"]).intersection(HIGHSTEER_RUNS)
            or set(candidate_metadata["training_runs"]).intersection(
                HIGHSTEER_RUNS)):
        raise ValueError("a teacher was trained on a high-steer holdout run")

    state_source = physical_state_from_dataset(
        data, str(parent_metadata.get("wheel_state_source", "filtered_odometry")))
    state = (append_roll_state(data, state_source).astype(np.float32)
             if parent_model.include_roll_state
             else state_source.astype(np.float32))
    raw_history = None
    if parent_model.include_raw_encoder_history:
        from tools.vehicle_dynamics_learning.effective_race_teacher import raw_encoder_history_features
        raw_history = raw_encoder_history_features(data)
    elif parent_model.include_wheel_innovation_history:
        from tools.vehicle_dynamics_learning.effective_race_teacher import wheel_innovation_history_features
        raw_history = wheel_innovation_history_features(data)

    run_names = sorted(HIGHSTEER_RUNS)
    metric_names = (
        "position_radial_rmse_m", "heading_rmse_rad", "speed_rmse_mps",
        "forward_speed_rmse_mps", "lateral_speed_rmse_mps",
        "yaw_rate_rmse_rps", "rear_wheel_pair_rmse_mps")
    horizons = {"0.75s": 30, "2s": 80, "5s": 200}
    per_run: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        str(offset): {
            name: {
                f"{horizon}/{metric}": {run: [] for run in run_names}
                for horizon in horizons for metric in metric_names
            }
            for name in ("parent", "full_candidate", "hybrid")
        }
        for offset in (-1, 0)
    }
    for offset in (-1, 0):
        histories, initial, delayed, commands, truth, poses = _window_batch(
            data, state, windows, parent_model.history_state_size, raw_history,
            max(horizons.values()), offset)
        device_object = torch.device(device)
        tensors = [torch.as_tensor(value, dtype=torch.float32,
                                   device=device_object)
                  for value in (histories, initial, delayed, commands)]
        with torch.no_grad():
            parent_state = parent_model.rollout(
                tensors[1], tensors[2], tensors[0], tensors[3])[0]
            candidate_state = candidate_model.rollout(
                tensors[1], tensors[2], tensors[0], tensors[3])[0]
            hybrid_state = candidate_state.clone()
            hybrid_state[:, :, 2] = parent_state[:, :, 2]
            start_pose = torch.as_tensor(
                poses[:, 0], dtype=torch.float32, device=device_object)
            parent_pose = integrate_pose(torch, parent_state, start_pose,
                                         tensors[1])
            candidate_pose = integrate_pose(torch, candidate_state,
                                            start_pose, tensors[1])
            hybrid_pose = integrate_pose(torch, hybrid_state, start_pose,
                                         tensors[1])
        predictions = {
            "parent": (parent_state.cpu().numpy(), parent_pose.cpu().numpy()),
            "full_candidate": (candidate_state.cpu().numpy(),
                               candidate_pose.cpu().numpy()),
            "hybrid": (hybrid_state.cpu().numpy(), hybrid_pose.cpu().numpy()),
        }
        for horizon_name, count in horizons.items():
            for index, window in enumerate(windows):
                run_id = str(run_ids[int(window["run"])])
                for model_name, (pred_state, pred_pose) in predictions.items():
                    metrics = _metrics(
                        pred_state[index, :count], pred_pose[index, :count],
                        truth[index, :count], poses[index, 1:count + 1])
                    for metric in metric_names:
                        per_run[str(offset)][model_name][
                            f"{horizon_name}/{metric}"].setdefault(
                                run_id, []).append(float(metrics[metric]))
    summary: dict[str, Any] = {}
    for offset, models in per_run.items():
        summary[offset] = {}
        for model_name, metric_runs in models.items():
            summary[offset][model_name] = {}
            for metric, run_values in metric_runs.items():
                run_means = {run: float(np.mean(values))
                             for run, values in run_values.items() if values}
                summary[offset][model_name][metric] = _cluster_summary(
                    run_means)
    return {
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "heldout_runs": run_names,
        "window_count": len(windows),
        "independent_run_count": len(run_names),
        "command_offsets_from_target_row": [-1, 0],
        "metrics": summary,
    }


def _hybrid_rollout(torch: Any, parent_checkpoint: Path,
                    candidate_checkpoint: Path, data: dict[str, Any],
                    run_index: int, device: str,
                    start_frame: int | None = None,
                    end_frame: int | None = None) -> dict[str, Any]:
    parent = _rollout(parent_checkpoint, data, run_index, device,
                      start_frame, end_frame, 0)
    candidate = _rollout(candidate_checkpoint, data, run_index, device,
                         start_frame, end_frame, 0)
    if (parent["initial_frame"] != candidate["initial_frame"]
            or not np.array_equal(parent["future_frames"],
                                  candidate["future_frames"])):
        raise ValueError("parent and candidate used different replay frames")

    state = candidate["state"].copy()
    state[:, 2] = parent["state"][:, 2]
    metadata = candidate["metadata"]
    base_state = physical_state_from_dataset(
        data, str(metadata.get("wheel_state_source", "filtered_odometry")))
    if bool(metadata.get("include_roll_state", False)):
        base_state = append_roll_state(data, base_state)
    start = int(candidate["initial_frame"])
    initial_state = torch.as_tensor(
        base_state[start:start + 1], dtype=torch.float32, device=device)
    initial_pose = torch.as_tensor(
        candidate["initial_pose"][None], dtype=torch.float32, device=device)
    physical_states = torch.as_tensor(
        state[None], dtype=torch.float32, device=device)
    with torch.no_grad():
        pose = integrate_pose(
            torch, physical_states, initial_pose, initial_state)[0]
    return {
        "initial_frame": start,
        "run_start": candidate["run_start"],
        "run_end": candidate["run_end"],
        "initial_pose": candidate["initial_pose"],
        "future_frames": candidate["future_frames"],
        "state": state,
        "pose": pose.cpu().numpy().astype(np.float64),
        "metadata": metadata,
        "initialization_contract": (
            "parallel parent and candidate rollouts share one measured start; "
            "recorded commands only; output uses candidate body/actuator/wheel/"
            "roll channels and parent yaw rate"),
    }


def evaluate(dataset_path: Path, benchmark_path: Path,
             parent_checkpoint: Path, candidate_checkpoint: Path,
             output_path: Path, device: str = "cuda",
             highsteer_dataset: Path = DEFAULT_HIGHSTEER_DATASET
             ) -> dict[str, Any]:
    paths = tuple(path.resolve() for path in (
        dataset_path, benchmark_path, parent_checkpoint,
        candidate_checkpoint, output_path))
    dataset_path, benchmark_path, parent_checkpoint, candidate_checkpoint, output_path = paths
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (benchmark.get("benchmark_id") != "practice_transfer_benchmark_v1"
            or benchmark.get("frozen") is not True
            or benchmark.get("training_or_checkpoint_selection_use") is not False
            or _sha256(dataset_path) != benchmark.get("dataset_sha256")):
        raise ValueError("practice replay data differ from the frozen benchmark")

    torch, parent_model, parent_metadata = _load_model(parent_checkpoint, device)
    _, candidate_model, candidate_metadata = _load_model(candidate_checkpoint, device)
    if (parent_model.history_state_size != candidate_model.history_state_size
            or parent_model.include_roll_state
            != candidate_model.include_roll_state
            or parent_metadata.get("wheel_state_source", "filtered_odometry")
            != candidate_metadata.get("wheel_state_source",
                                      "filtered_odometry")):
        raise ValueError("parent and candidate plant-state contracts differ")
    del parent_model, candidate_model
    data = _load_dataset(dataset_path)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    run_indices = [index for index, split in enumerate(data["splits"])
                   if str(split) == "unseen_practice"]
    if not run_indices:
        raise ValueError("frozen benchmark contains no unseen practice runs")

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "diagnostic split-output hybrid; not a production model",
        "future_truth_or_sensor_feedback_used": False,
        "test_and_final_test_used": False,
        "training_or_checkpoint_selection_use": False,
        "position_integration": (
            "candidate u/v with parent yaw-rate; candidate actuator, wheel, "
            "and internally predicted roll channels"),
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": _sha256(parent_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": _sha256(candidate_checkpoint),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "parent_training_runs": parent_metadata["training_runs"],
        "candidate_training_runs": candidate_metadata["training_runs"],
        "runs": {},
        "highsteer_holdout": _evaluate_highsteer(
            highsteer_dataset.resolve(), parent_checkpoint,
            candidate_checkpoint, device),
    }
    for run_index in run_indices:
        run_id = str(run_ids[run_index])
        rows = _contiguous_run_indices(data, run_index)
        run_start, run_end = int(rows[0]), int(rows[-1] + 1)
        truth_pose = data["simulator_pose_xyyaw"]
        gate, tangent, actual_crossings = _finish_gate(
            data["lap_count"], truth_pose, run_start, run_end)
        parent = _rollout(parent_checkpoint, data, run_index, device,
                          command_offset_frames=0)
        candidate = _rollout(candidate_checkpoint, data, run_index, device,
                             command_offset_frames=0)
        hybrid = _hybrid_rollout(torch, parent_checkpoint, candidate_checkpoint,
                                 data, run_index, device)
        scores = {
            "parent": _score_one_run(data, parent),
            "full_candidate": _score_one_run(data, candidate),
            "hybrid": _score_one_run(data, hybrid),
        }
        crossings = _predicted_crossings(
            hybrid["initial_pose"], hybrid["pose"], gate, tangent)
        actual = [time for time in actual_crossings
                  if time > hybrid["initial_frame"] * 0.025]
        actual_laps = np.diff(actual)
        predicted_laps = np.diff(crossings)
        paired = min(len(actual_laps), len(predicted_laps))
        scores["hybrid"]["lap_time_mae_s"] = (
            float(np.mean(np.abs(actual_laps[:paired]
                                 - predicted_laps[:paired])))
            if paired else None)

        transitions = _lap_transition_indices(
            np.asarray(data["lap_count"], dtype=np.int32), run_start, run_end)
        isolated = []
        for lap_index, (start, next_crossing) in enumerate(
                zip(transitions, transitions[1:])):
            result = _hybrid_rollout(
                torch, parent_checkpoint, candidate_checkpoint, data, run_index,
                device, int(start), int(next_crossing + 1))
            isolated_score = _score_one_run(data, result)
            isolated.append({
                "lap_index": lap_index,
                "metrics": isolated_score["full_recursive_metrics"],
            })
        report["runs"][run_id] = {
            "parent": scores["parent"]["full_recursive_metrics"],
            "full_candidate": scores["full_candidate"]["full_recursive_metrics"],
            "hybrid": scores["hybrid"]["full_recursive_metrics"],
            "hybrid_lap_time_mae_s": scores["hybrid"]["lap_time_mae_s"],
            "hybrid_isolated_laps": isolated,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--highsteer-dataset", type=Path,
                        default=DEFAULT_HIGHSTEER_DATASET)
    parser.add_argument("--output", type=Path, default=(
        DEFAULT_OUTPUT_DIR / "yaw_preserving_parallel_hybrid_practice.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate(args.dataset, args.benchmark, args.parent,
                      args.candidate, args.output, args.device,
                      args.highsteer_dataset)
    print(json.dumps({
        run: {name: {key: value.get(key) for key in (
            "position_radial_rmse_m", "heading_rmse_rad", "speed_rmse_mps",
            "yaw_rate_rmse_rps", "yaw_rate_bias_rps",
            "rear_wheel_pair_rmse_mps")}
              for name, value in values.items()
              if name in ("parent", "full_candidate", "hybrid")}
        for run, values in result["runs"].items()
    }, indent=2))
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
