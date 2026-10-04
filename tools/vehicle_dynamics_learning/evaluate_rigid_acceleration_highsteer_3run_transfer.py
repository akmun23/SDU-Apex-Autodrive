#!/usr/bin/env python3
"""Evaluate frozen rigid-acceleration checkpoints on three whole-run holds.

The candidate and its no-added-highsteer control are evaluated with their
own training-only normalization, on identical starts from independent
high-steering captures. This is an offline diagnostic; it does not train,
launch the simulator, or alter production components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    _capture_from_archive,
    _macro,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    _batch_arrays,
    _bootstrap_delta,
    _eval_metrics,
    _normalization,
    _rollout,
    _torch_norm,
)
from tools.vehicle_dynamics_learning.run_long_horizon_context_candidate import (
    _eval_refs,
)
from tools.vehicle_dynamics_learning.run_rigid_acceleration_highsteer_transfer import (
    HIGHSTEER_TRAIN_DATASET,
    HIGHSTEER_TRAIN_RUNS,
    _append_training_capture,
    _load_acceleration_model,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    _metrics,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)


TASK_ROOT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
             / "full_modeling_reset_20261001"
             / "replacement_offline_sim_raceline_20261003"
             / "full_throttle_domain_v1/next_phase_after_2129427"
             / "history_context_sufficiency_v1")
DEFAULT_CANDIDATE = TASK_ROOT / "rigid_acceleration_highsteer_longrollout_v1/checkpoint.pt"
DEFAULT_CONTROL = TASK_ROOT / "rigid_acceleration_longrollout_no_highsteer_control_v1/checkpoint.pt"
DEFAULT_HOLDOUT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
                   / "full_modeling_reset_20261001"
                   / "highsteer_75_heldout_validation_plus_r03_20261004"
                   / "openplane_dynamics.npz")
DEFAULT_OUTPUT = TASK_ROOT / "rigid_acceleration_highsteer_3run_transfer_with_steering_bins_20261004.json"
HOLDOUT_RUNS = {
    "openplane_highsteer_75_validation_r01",
    "openplane_highsteer_75_validation_r02",
    "openplane_highsteer_75_validation_r03_20261004",
}
HORIZONS = (80, 200)
STEERING_BINS = (
    ("abs_steer_le_0p25_rad", 0.0, 0.25),
    ("abs_steer_0p25_to_0p35_rad", 0.25, 0.35),
    ("abs_steer_0p35_to_0p46_rad", 0.35, 0.46),
    ("abs_steer_gt_0p46_rad", 0.46, float("inf")),
)
METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _refs(capture, horizon: int, history: int):
    refs = _sample_all_valid_context_refs(
        capture, horizon, history, expected_runs=HOLDOUT_RUNS)
    return refs


def _metrics_by_steering_bin(data, refs_by_run, model, norm_np, config,
                             context_steps: int, device: torch.device,
                             horizon: int
                             ) -> tuple[dict[str, Any], dict[str, Any]]:
    norm = _torch_norm(norm_np, config, device)
    overall: dict[str, Any] = {}
    report: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for run_id, refs in sorted(refs_by_run.items()):
            capture_index = int(refs[0][0])
            capture = data.captures[capture_index]
            arrays = _batch_arrays(
                data, refs, horizon, context_steps,
                norm_np, config, device)
            rollout = _rollout(
                model, arrays, norm, horizon, context_steps,
                PoseIntegrator(DT_S).to(device))
            predicted_state = (rollout["state"].cpu().numpy()
                               * norm_np["state_scale"]
                               + norm_np["state_mean"])
            target_state = (arrays[5].cpu().numpy()
                            * norm_np["state_scale"]
                            + norm_np["state_mean"])
            predicted_pose = rollout["pose"].cpu().numpy()
            target_pose = arrays[6].cpu().numpy()
            starts = np.asarray([
                abs(float(capture.frames[
                    int(capture.bounds[sequence_index, 0]) + int(source_row), 7]))
                for _, sequence_index, source_row in refs], dtype=np.float64)
            overall[run_id] = _metrics(
                predicted_state, predicted_pose, target_state,
                target_pose, horizon)
            report[run_id] = {}
            for name, low, high in STEERING_BINS:
                selected = ((starts >= low) & (starts <= high)
                            if low == 0.0 else
                            (starts > low) & (starts <= high))
                if not np.any(selected):
                    continue
                mask = np.flatnonzero(selected)
                report[run_id][name] = {
                    "start_count": int(mask.size),
                    **_metrics(predicted_state[mask], predicted_pose[mask],
                               target_state[mask], target_pose[mask], horizon),
                }
    return overall, report


def _paired_binned(candidate, control) -> dict[str, Any]:
    bin_names = sorted({name for runs in candidate.values() for name in runs}
                       | {name for runs in control.values() for name in runs})
    paired: dict[str, Any] = {}
    for bin_index, name in enumerate(bin_names):
        candidate_metrics = {
            run: values[name] for run, values in candidate.items()
            if name in values}
        control_metrics = {
            run: values[name] for run, values in control.items()
            if name in values}
        common = set(candidate_metrics) & set(control_metrics)
        if len(common) < 2:
            continue
        paired[name] = {
            metric: _bootstrap_delta(
                candidate_metrics, control_metrics, metric,
                20261044 + bin_index + metric_index)
            for metric_index, metric in enumerate(METRICS)}
    return paired


def _training_context(add_highsteer: bool, device: torch.device):
    data, wp20 = _load_data()
    if add_highsteer:
        _append_training_capture(data, HIGHSTEER_TRAIN_DATASET)
        if not HIGHSTEER_TRAIN_RUNS.issubset(set(data.training_runs)):
            raise RuntimeError("candidate training captures are missing")
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm = _normalization(data, config)
    return data, config, norm


def run(device_name: str = "cpu", candidate_path: Path = DEFAULT_CANDIDATE,
        control_path: Path = DEFAULT_CONTROL,
        holdout_path: Path = DEFAULT_HOLDOUT,
        output_path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)

    control_data, control_config, control_norm = _training_context(False, device)
    candidate_data, candidate_config, candidate_norm = _training_context(True, device)
    candidate, candidate_sha = _load_acceleration_model(
        candidate_path, candidate_norm, device)
    control, control_sha = _load_acceleration_model(
        control_path, control_norm, device)

    capture, pose = _capture_from_archive(
        holdout_path, HOLDOUT_RUNS, expected_split="validation")
    if (set(capture.run_ids.tolist()) != HOLDOUT_RUNS
            or HOLDOUT_RUNS & set(candidate_data.training_runs)
            or HOLDOUT_RUNS & set(control_data.training_runs)):
        raise RuntimeError("high-steering holdout overlaps training or has wrong roster")
    holdout = SimpleNamespace(captures=[capture], poses=[pose])
    refs_by_horizon = {
        horizon: _refs(capture, horizon, CONTEXT_STEPS["2.0s"])
        for horizon in HORIZONS}

    results: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    steering_bins: dict[str, Any] = {}
    paired_steering_bins: dict[str, Any] = {}
    for horizon in HORIZONS:
        key = str(horizon)
        if horizon == 200:
            candidate_metrics, candidate_bins = _metrics_by_steering_bin(
                holdout, refs_by_horizon[horizon], candidate, candidate_norm,
                candidate_config, CONTEXT_STEPS["2.0s"], device, horizon)
            control_metrics, control_bins = _metrics_by_steering_bin(
                holdout, refs_by_horizon[horizon], control, control_norm,
                control_config, CONTEXT_STEPS["2.0s"], device, horizon)
        else:
            candidate_metrics = _eval_metrics(
                holdout, refs_by_horizon[horizon], candidate, candidate_norm,
                candidate_config, CONTEXT_STEPS["2.0s"], device, horizon)
            control_metrics = _eval_metrics(
                holdout, refs_by_horizon[horizon], control, control_norm,
                control_config, CONTEXT_STEPS["2.0s"], device, horizon)
        results[key] = {
            "candidate_per_run": candidate_metrics,
            "candidate_run_macro": _macro(candidate_metrics),
            "matched_control_per_run": control_metrics,
            "matched_control_run_macro": _macro(control_metrics),
        }
        paired[key] = {
            metric: _bootstrap_delta(
                candidate_metrics, control_metrics, metric,
                20261004 + horizon + index)
            for index, metric in enumerate(METRICS)}
        if horizon == 200:
            steering_bins = {
                "candidate_per_run": candidate_bins,
                "matched_control_per_run": control_bins,
            }
            paired_steering_bins = _paired_binned(candidate_bins, control_bins)

    report: dict[str, Any] = {
        "study": "frozen rigid-acceleration candidate vs matched no-extra-data control on three high-steer validation runs",
        "candidate_checkpoint_sha256": candidate_sha,
        "matched_control_checkpoint_sha256": control_sha,
        "heldout_dataset_sha256": sha256_file(holdout_path),
        "heldout_whole_run_ids": sorted(HOLDOUT_RUNS),
        "heldout_split": "validation; excluded from fitting and checkpoint selection",
        "candidate_training_run_ids": sorted(candidate_data.training_runs),
        "control_training_run_ids": sorted(control_data.training_runs),
        "highsteer_added_training_run_ids": sorted(HIGHSTEER_TRAIN_RUNS),
        "fixed_dt_s": 0.025,
        "history_steps": CONTEXT_STEPS["2.0s"],
        "input_policy": "causal recorded actuator/encoder history and known future commands; no future truth/sensors",
        "start_count_by_run": {
            run_id: len(refs)
            for run_id, refs in sorted(refs_by_horizon[200].items())},
        "results": results,
        "five_second_steering_command_magnitude_bins": steering_bins,
        "five_second_steering_bin_paired_candidate_minus_control": paired_steering_bins,
        "paired_candidate_minus_control_run_cluster_ci": paired,
        "simulator_launched": False,
        "production_integration": False,
        "limits": [
            "Only three independent whole-run captures cover 7.5 m/s high-steering holds.",
            "This does not validate the full 0-12 m/s operating surface or a full-lap command replay.",
        ],
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = (args.candidate, args.control, args.holdout, args.output)
    paths = tuple(path if path.is_absolute() else ROOT / path for path in paths)
    report = run(args.device, *paths)
    print(json.dumps({
        "output": paths[3].relative_to(ROOT).as_posix(),
        "start_count_by_run": report["start_count_by_run"],
        "five_second_candidate": report["results"]["200"]["candidate_run_macro"],
        "five_second_control": report["results"]["200"]["matched_control_run_macro"],
        "five_second_position_delta": report[
            "paired_candidate_minus_control_run_cluster_ci"]["200"][
                "position_radial_trajectory_rmse_m"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
