#!/usr/bin/env python3
"""Stress-test rigid-acceleration candidate beyond its 5 s fit horizon."""

from __future__ import annotations

import argparse
import json
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
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    OUTPUT_ROOT as WP28_ROOT,
    FixedCapacityHistoryTransition,
    _bootstrap_delta,
    _eval_metrics,
    _normalization,
)
from tools.vehicle_dynamics_learning.run_long_horizon_context_candidate import (
    _eval_refs,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
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


ROOT_OUTPUT = (WP28_ROOT / "rigid_acceleration_history_5s_v1"
               / "transfer_stress_10s.json")
CANDIDATE_CHECKPOINT = (WP28_ROOT / "rigid_acceleration_history_5s_v1"
                        / "checkpoint.pt")
WP28_CHECKPOINT = (WP28_ROOT / "long_rollout_5s_selected_context_v1"
                   / "checkpoint.pt")
METRICS = (
    "position_radial_trajectory_rmse_m",
    "position_endpoint_error_m",
    "heading_trajectory_rmse_rad",
    "heading_endpoint_error_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def evaluate(device_name: str = "cpu", output_path: Path = ROOT_OUTPUT,
             candidate_checkpoint: Path = CANDIDATE_CHECKPOINT
             ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    validation_5s = _eval_refs(data, "validation", 0, expected_validation)
    practice_5s = _eval_refs(data, "unseen_practice", 1, expected_practice)
    validation_10s_all = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"validation"}, 400, capture_indices={0})
    validation_10s = _select_eval_windows(validation_10s_all, 64)
    if set(validation_10s) != expected_validation:
        raise RuntimeError("10 s validation run roster changed")

    saved = torch.load(candidate_checkpoint, map_location=device,
                       weights_only=True)
    meta = saved["metadata"]
    candidate = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(meta["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(meta["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(meta["dt_s"]),
        rear_axle_to_com_x_m=float(meta["rear_axle_to_com_x_m"]),
    ).to(device)
    candidate.load_state_dict(saved["state_dict"], strict=True)
    base_saved = torch.load(WP28_CHECKPOINT, map_location=device,
                            weights_only=True)
    baseline = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    baseline.load_state_dict(base_saved["state_dict"], strict=True)

    val_candidate_10s = _eval_metrics(
        data, validation_10s, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 400)
    val_base_10s = _eval_metrics(
        data, validation_10s, baseline, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 400)
    val_candidate_5s = _eval_metrics(
        data, validation_5s, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)
    val_base_5s = _eval_metrics(
        data, validation_5s, baseline, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)
    practice_candidate_5s = _eval_metrics(
        data, practice_5s, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)
    practice_base_5s = _eval_metrics(
        data, practice_5s, baseline, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)

    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    high_candidate_5s = _eval_metrics(
        high_data, high_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)
    high_base_5s = _eval_metrics(
        high_data, high_refs, baseline, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)

    comparisons = {
        "validation_10s": (val_candidate_10s, val_base_10s),
        "validation_5s": (val_candidate_5s, val_base_5s),
        "practice_5s": (practice_candidate_5s, practice_base_5s),
        "highsteer_5s": (high_candidate_5s, high_base_5s),
    }
    paired = {
        name: {
            metric: _bootstrap_delta(
                values[0], values[1], metric,
                20261004 + len(name) + len(metric))
            for metric in METRICS}
        for name, values in comparisons.items()}
    report = {
        "study": "held-out transfer and 10 s stress of acceleration-integrated plant",
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "WP28_5s_checkpoint_sha256": sha256_file(WP28_CHECKPOINT),
        "data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer": sha256_file(HIGHSTEER_SOURCE),
        },
        "independent_unit": "whole simulator run",
        "horizons": {"validation_10s_steps": 400, "other_steps": 200},
        "per_run_metrics": {
            "validation_10s_candidate": val_candidate_10s,
            "validation_10s_WP28": val_base_10s,
            "validation_5s_candidate": val_candidate_5s,
            "validation_5s_WP28": val_base_5s,
            "practice_5s_candidate": practice_candidate_5s,
            "practice_5s_WP28": practice_base_5s,
            "highsteer_5s_candidate": high_candidate_5s,
            "highsteer_5s_WP28": high_base_5s,
        },
        "macro_metrics": {
            name: {"candidate": _macro(values[0]),
                  "WP28": _macro(values[1])}
            for name, values in comparisons.items()},
        "paired_candidate_minus_WP28_with_run_cluster_ci": paired,
        "validation_10s_participated_in_5s_checkpoint_selection": True,
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "The two high-steering runs and practice diagnostics are independent "
            "of candidate training. The six 10 s development runs participated in "
            "5 s checkpoint selection, so the 10 s test is stress evaluation, not "
            "an untouched final test."),
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT_OUTPUT)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_CHECKPOINT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    candidate = (args.candidate if args.candidate.is_absolute()
                 else ROOT / args.candidate)
    report = evaluate(args.device, output, candidate)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "macro_metrics": report["macro_metrics"],
        "paired_position_deltas": {
            name: values["position_radial_trajectory_rmse_m"]
            for name, values in report[
                "paired_candidate_minus_WP28_with_run_cluster_ci"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
