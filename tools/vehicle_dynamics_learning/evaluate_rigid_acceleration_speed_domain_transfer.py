#!/usr/bin/env python3
"""Stratify the matched acceleration-plant comparison by starting speed.

Uses the frozen whole-run dynamic validation and practice diagnostics to test
whether the 7.5 m/s high-steering augmentation's transfer tradeoff is confined
to particular speed bands. It evaluates frozen checkpoints only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    _batch_arrays,
    _bootstrap_delta,
    _metrics,
    _region_metrics,
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
from tools.vehicle_dynamics_learning.evaluate_rigid_acceleration_highsteer_3run_transfer import (
    DEFAULT_CANDIDATE,
    DEFAULT_CONTROL,
    _training_context,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    sha256_file,
)


TASK_ROOT = DEFAULT_CANDIDATE.parent.parent
DEFAULT_OUTPUT = TASK_ROOT / "rigid_acceleration_speed_domain_and_regions_20261004.json"
HORIZON = 200
SPEED_BINS = (
    ("0_to_4mps", 0.0, 4.0),
    ("4_to_6mps", 4.0, 6.0),
    ("6_to_8mps", 6.0, 8.0),
    ("8_to_10mps", 8.0, 10.0),
    ("10_to_12mps", 10.0, 12.0),
)
METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _metrics_by_speed_bin(data, refs_by_run, model, norm_np, config,
                          device: torch.device) -> dict[str, Any]:
    norm = _torch_norm(norm_np, config, device)
    result: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for run_id, refs in sorted(refs_by_run.items()):
            capture = data.captures[int(refs[0][0])]
            arrays = _batch_arrays(
                data, refs, HORIZON, CONTEXT_STEPS["2.0s"],
                norm_np, config, device)
            rollout = _rollout(
                model, arrays, norm, HORIZON, CONTEXT_STEPS["2.0s"],
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
                np.hypot(*capture.body[
                    int(capture.bounds[sequence_index, 0])
                    + int(source_row), :2])
                for _, sequence_index, source_row in refs], dtype=np.float64)
            result[run_id] = {}
            for name, low, high in SPEED_BINS:
                selected = ((starts >= low) & (starts <= high)
                            if low == 0.0 else
                            (starts > low) & (starts <= high))
                if not np.any(selected):
                    continue
                indexes = np.flatnonzero(selected)
                result[run_id][name] = {
                    "start_count": int(indexes.size),
                    **_metrics(predicted_state[indexes], predicted_pose[indexes],
                               target_state[indexes], target_pose[indexes], HORIZON),
                }
    return result


def _paired_bins(candidate, control) -> dict[str, Any]:
    names = sorted({name for runs in candidate.values() for name in runs}
                   | {name for runs in control.values() for name in runs})
    paired: dict[str, Any] = {}
    for bin_index, name in enumerate(names):
        cand = {run: values[name] for run, values in candidate.items()
                if name in values}
        base = {run: values[name] for run, values in control.items()
                if name in values}
        if len(set(cand) & set(base)) < 2:
            continue
        paired[name] = {
            metric: _bootstrap_delta(
                cand, base, metric, 20261045 + bin_index + index)
            for index, metric in enumerate(METRICS)}
    return paired


def run(device_name: str = "cpu", candidate_path: Path = DEFAULT_CANDIDATE,
        control_path: Path = DEFAULT_CONTROL,
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

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    validation_ids = set(frozen["split_roles"]["development_validation"])
    practice_ids = set(frozen["split_roles"]["practice_diagnostic"])
    groups = {
        "development_validation": (
            0, "validation", validation_ids),
        "unseen_practice_diagnostic": (
            1, "unseen_practice", practice_ids),
    }
    results: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for group_name, (capture_index, split, expected_runs) in groups.items():
        refs = _eval_refs(control_data, split, capture_index, expected_runs)
        if expected_runs & set(control_data.training_runs):
            raise RuntimeError(f"{group_name} overlaps the control training split")
        if expected_runs & set(candidate_data.training_runs):
            raise RuntimeError(f"{group_name} overlaps the candidate training split")
        candidate_bins = _metrics_by_speed_bin(
            candidate_data, refs, candidate, candidate_norm,
            candidate_config, device)
        control_bins = _metrics_by_speed_bin(
            control_data, refs, control, control_norm,
            control_config, device)
        results[group_name] = {
            "run_ids": sorted(refs),
            "start_count_by_run": {run: len(rows) for run, rows in refs.items()},
            "candidate_per_run": candidate_bins,
            "matched_control_per_run": control_bins,
        }
        paired[group_name] = _paired_bins(candidate_bins, control_bins)
        if group_name == "development_validation":
            candidate_regions = _region_metrics(
                candidate_data, refs, candidate, candidate_norm,
                candidate_config, CONTEXT_STEPS["2.0s"], device, HORIZON)
            control_regions = _region_metrics(
                control_data, refs, control, control_norm,
                control_config, CONTEXT_STEPS["2.0s"], device, HORIZON)
            results[group_name]["causal_input_region_per_run"] = {
                "candidate": candidate_regions,
                "matched_control": control_regions,
            }
            paired[group_name]["causal_input_region"] = _paired_bins(
                candidate_regions, control_regions)

    report: dict[str, Any] = {
        "study": "5 s rigid-acceleration candidate/control errors stratified by initial body-speed band",
        "candidate_checkpoint_sha256": sha256_file(candidate_path),
        "matched_control_checkpoint_sha256": sha256_file(control_path),
        "highsteer_training_dataset_sha256": sha256_file(HIGHSTEER_TRAIN_DATASET),
        "highsteer_added_training_runs": sorted(HIGHSTEER_TRAIN_RUNS),
        "horizon_steps": HORIZON,
        "horizon_seconds": HORIZON * DT_S,
        "speed_bins_mps": [
            {"name": name, "minimum": low, "maximum": high}
            for name, low, high in SPEED_BINS],
        "causal_input_regions": [
            "7_to_9mps_high_steering",
            "simultaneous_steering_throttle_transition",
            "large_wheel_body_mismatch",
            "steering_left",
            "steering_right",
        ],
        "results": results,
        "paired_candidate_minus_control_run_cluster_ci": paired,
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "limits": [
            "The dynamic comparison uses 64 fixed starts per whole run; bins are reported only where at least two independent runs contain starts.",
            "The candidate's added data is localized at 7.5 m/s; this report diagnoses speed transfer but does not select a gate or establish full-lap accuracy.",
        ],
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = (args.candidate, args.control, args.output)
    paths = tuple(path if path.is_absolute() else ROOT / path for path in paths)
    report = run(args.device, *paths)
    print(json.dumps({
        "output": paths[2].relative_to(ROOT).as_posix(),
        "groups": {
            group: {
                "run_ids": values["run_ids"],
                "bins": sorted({name for run in values["candidate_per_run"].values()
                                for name in run}),
            }
            for group, values in report["results"].items()},
        "position_deltas": {
            group: {
                name: metrics["position_radial_trajectory_rmse_m"][
                    "candidate_minus_reference_macro_delta"]
                for name, metrics in bins.items()
                if name != "causal_input_region"}
            for group, bins in report[
                "paired_candidate_minus_control_run_cluster_ci"].items()},
        "region_position_deltas": {
            region: metrics["position_radial_trajectory_rmse_m"][
                "candidate_minus_reference_macro_delta"]
            for region, metrics in report[
                "paired_candidate_minus_control_run_cluster_ci"
                    ]["development_validation"][
                        "causal_input_region"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
