#!/usr/bin/env python3
"""Extend the rigid-acceleration plant's recursive objective to ten seconds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    EXPECTED_RUNS as HIGHSTEER_RUNS,
    HIGHSTEER_SOURCE,
    _capture_from_archive,
    _macro,
    _paired,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    OUTPUT_ROOT as WP28_ROOT,
    SEED,
    FixedCapacityHistoryTransition,
    _draw_plan,
    _eval_metrics,
    _normalization,
    _train_one,
)
from tools.vehicle_dynamics_learning.run_long_horizon_context_candidate import (
    _curve,
    _eval_refs,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _collect_horizon_windows,
    _load_data,
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)


PARENT_CHECKPOINT = (WP28_ROOT / "rigid_acceleration_history_direct_supervision_5s_v1"
                     / "checkpoint.pt")
BASELINE_CHECKPOINT = (WP28_ROOT / "long_rollout_5s_selected_context_v1"
                       / "checkpoint.pt")
OUTPUT_ROOT = (WP28_ROOT / "rigid_acceleration_history_direct_supervision_10s_v1")
STAGES = (("L2", (200, 300, 400), 300, 8),)
METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _ten_second_pool(data) -> dict[str, dict[str, list]]:
    by_run = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"train"}, 400, capture_indices={0})
    condition_by_sequence = data.raw_sources[0]["sequence_condition_id"]
    capture = data.captures[0]
    pools = {}
    for run_id in data.training_runs:
        conditions: dict[str, list] = defaultdict(list)
        for ref in by_run.get(run_id, []):
            _, sequence_index, source_row = ref
            if int(source_row) < 159:
                continue
            condition = str(int(condition_by_sequence[int(sequence_index)]))
            conditions[condition].append(ref)
        if conditions:
            pools[run_id] = dict(conditions)
    if len(pools) < 5:
        raise RuntimeError("10 s training requires at least five independent runs")
    if any(ref[0] != 0 for conditions in pools.values()
           for refs in conditions.values() for ref in refs):
        raise RuntimeError("10 s pool includes a non-training capture")
    return pools


def _load_candidate(path: Path, norm_np, device: torch.device):
    saved = torch.load(path, map_location=device, weights_only=True)
    meta = saved["metadata"]
    model = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(meta["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(meta["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(meta["dt_s"]),
        rear_axle_to_com_x_m=float(meta["rear_axle_to_com_x_m"]),
    ).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model, saved


def run(device_name: str = "cpu", output_root: Path = OUTPUT_ROOT
        ) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    training_pools = _ten_second_pool(data)
    plan, sampler_counts = _draw_plan(
        training_pools, SEED + 1000, stages=STAGES)

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    validation_all = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"validation"}, 400, capture_indices={0})
    validation_refs = _select_eval_windows(validation_all, 64)
    if set(validation_refs) != expected_validation:
        raise RuntimeError("10 s validation run roster differs from frozen split")
    practice_refs = _eval_refs(data, "unseen_practice", 1, expected_practice)
    if HIGHSTEER_RUNS & (set(data.training_runs) | set(validation_refs)
                         | set(practice_refs)):
        raise RuntimeError("high-steer holdout overlaps train/dev/practice")

    parent, parent_saved = _load_candidate(PARENT_CHECKPOINT, norm_np, device)
    wp28_saved = torch.load(BASELINE_CHECKPOINT, map_location=device,
                            weights_only=True)
    wp28 = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    wp28.load_state_dict(wp28_saved["state_dict"], strict=True)
    parent_metrics = {
        400: _eval_metrics(data, validation_refs, parent, norm_np, config,
                           CONTEXT_STEPS["2.0s"], device, 400)}

    candidate, training = _train_one(
        data, "rigid_acceleration_direct_supervision_10s",
        CONTEXT_STEPS["2.0s"], plan, validation_refs, norm_np, config,
        parent_metrics, device, stages=STAGES,
        initial_state_dict=parent_saved["state_dict"],
        model_factory=lambda: RigidAccelerationHistoryTransition(
            norm_np["state_mean"], norm_np["state_scale"],
            np.asarray(parent_saved["metadata"]["acceleration_mean_train_only"],
                       dtype=np.float32),
            np.asarray(parent_saved["metadata"]["acceleration_scale_train_only"],
                       dtype=np.float32),
            norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
            dt_s=float(parent_saved["metadata"]["dt_s"]),
            rear_axle_to_com_x_m=float(
                parent_saved["metadata"]["rear_axle_to_com_x_m"])))

    output_root.mkdir(parents=True)
    checkpoint_path = output_root / "checkpoint.pt"
    torch.save({
        "state_dict": {name: value.detach().cpu()
                       for name, value in candidate.state_dict().items()},
        "metadata": {
            **parent_saved["metadata"],
            "model": "WP28 causal history + midpoint acceleration with implicit rigid integration",
            "training_horizons_steps": [200, 300, 400],
            "training_horizon_seconds": [5.0, 7.5, 10.0],
            "ten_second_finetune_run_ids": sorted(training_pools),
            "validation_run_ids": sorted(validation_refs),
            "future_sensor_or_truth_inputs": False,
            "production_integration": False,
        },
    }, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    (output_root / "checkpoint.sha256").write_text(
        checkpoint_sha + "\n", encoding="utf-8")

    validation_10s = _eval_metrics(
        data, validation_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 400)
    parent_10s = parent_metrics[400]
    wp28_10s = _eval_metrics(
        data, validation_refs, wp28, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 400)
    validation_5s = _curve(
        data, validation_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    parent_5s = _curve(
        data, validation_refs, parent, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    practice_5s = _curve(
        data, practice_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    parent_practice_5s = _curve(
        data, practice_refs, parent, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    high_5s = _curve(high_data, high_refs, candidate, norm_np, config,
                     CONTEXT_STEPS["2.0s"], device)
    parent_high_5s = _curve(
        high_data, high_refs, parent, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)

    report = {
        "study": "ten-second direct acceleration-supervised recursive finetune",
        "checkpoint_sha256": checkpoint_sha,
        "parent_5s_checkpoint_sha256": sha256_file(PARENT_CHECKPOINT),
        "wp28_baseline_checkpoint_sha256": sha256_file(BASELINE_CHECKPOINT),
        "data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer": sha256_file(HIGHSTEER_SOURCE),
        },
        "training_runs": sorted(training_pools),
        "training_windows_by_run": {
            run_id: sum(map(len, conditions.values()))
            for run_id, conditions in training_pools.items()},
        "sampler_counts": sampler_counts,
        "training": training,
        "validation_runs": sorted(validation_refs),
        "validation_10s_candidate_per_run": validation_10s,
        "validation_10s_parent_5s_candidate_per_run": parent_10s,
        "validation_10s_WP28_per_run": wp28_10s,
        "validation_10s_macro": {
            "candidate": _macro(validation_10s),
            "parent_5s_candidate": _macro(parent_10s),
            "WP28": _macro(wp28_10s),
        },
        "validation_10s_paired_candidate_minus_parent": {
            metric: _paired(validation_10s, parent_10s, metric)
            for metric in METRICS},
        "validation_5s_candidate_macro": {
            horizon: _macro(values) for horizon, values in validation_5s.items()},
        "validation_5s_parent_macro": {
            horizon: _macro(values) for horizon, values in parent_5s.items()},
        "practice_5s_candidate_macro": {
            horizon: _macro(values) for horizon, values in practice_5s.items()},
        "practice_5s_parent_macro": {
            horizon: _macro(values) for horizon, values in parent_practice_5s.items()},
        "highsteer_5s_candidate_macro": {
            horizon: _macro(values) for horizon, values in high_5s.items()},
        "highsteer_5s_parent_macro": {
            horizon: _macro(values) for horizon, values in parent_high_5s.items()},
        "practice_and_highsteer_used_for_training_or_selection": False,
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "The six 10 s development runs select checkpoints. Practice and the "
            "two high-steer runs are transfer diagnostics only; they are not used "
            "for training or selection. No full-lap or global accuracy claim is made."),
    }
    _write_json(output_root / "ten_second_finetune_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(args.device, output)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "training_seconds": report["training"]["elapsed_seconds"],
        "validation_10s": report["validation_10s_macro"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
