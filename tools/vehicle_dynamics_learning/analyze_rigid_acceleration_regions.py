#!/usr/bin/env python3
"""Break down matched 5 s body/pose errors by rollout-start regime."""

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
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    OUTPUT_ROOT as WP28_ROOT,
    FixedCapacityHistoryTransition,
    _normalization,
    _region_metrics,
)
from tools.vehicle_dynamics_learning.run_long_horizon_context_candidate import (
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
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)


DEFAULT_CANDIDATE = (WP28_ROOT / "rigid_acceleration_history_direct_supervision_5s_v1"
                     / "checkpoint.pt")
DEFAULT_BASELINE = (WP28_ROOT / "long_rollout_5s_selected_context_v1"
                    / "checkpoint.pt")
DEFAULT_OUTPUT = (WP28_ROOT / "rigid_acceleration_history_direct_supervision_5s_v1"
                  / "regime_breakdown_5s.json")


def _load_models(candidate_path: Path, baseline_path: Path,
                 data, norm_np, device: torch.device):
    candidate_saved = torch.load(candidate_path, map_location=device,
                                 weights_only=True)
    metadata = candidate_saved["metadata"]
    candidate = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"]),
    ).to(device)
    candidate.load_state_dict(candidate_saved["state_dict"], strict=True)
    baseline_saved = torch.load(baseline_path, map_location=device,
                                weights_only=True)
    baseline = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    baseline.load_state_dict(baseline_saved["state_dict"], strict=True)
    return candidate, baseline


def analyze(candidate_path: Path = DEFAULT_CANDIDATE,
            baseline_path: Path = DEFAULT_BASELINE,
            output_path: Path = DEFAULT_OUTPUT,
            device_name: str = "cpu") -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    candidate, baseline = _load_models(
        candidate_path, baseline_path, data, norm_np, device)

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    validation_refs = _eval_refs(
        data, "validation", 0,
        set(frozen["split_roles"]["development_validation"]))
    practice_refs = _eval_refs(
        data, "unseen_practice", 1,
        set(frozen["split_roles"]["practice_diagnostic"]))
    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    groups = {
        "whole_run_development": (data, validation_refs),
        "unseen_practice_diagnostic": (data, practice_refs),
        "independent_highsteer": (high_data, high_refs),
    }
    metrics = {}
    for name, (group_data, refs) in groups.items():
        print(f"scoring {name}: {sum(map(len, refs.values()))} starts", flush=True)
        metrics[name] = {
            "candidate_per_run": _region_metrics(
                group_data, refs, candidate, norm_np, config,
                CONTEXT_STEPS["2.0s"], device, 200),
            "WP28_per_run": _region_metrics(
                group_data, refs, baseline, norm_np, config,
                CONTEXT_STEPS["2.0s"], device, 200),
        }
    report = {
        "study": "paired five-second recursive errors stratified by initial operating regime",
        "candidate_checkpoint_sha256": sha256_file(candidate_path),
        "WP28_checkpoint_sha256": sha256_file(baseline_path),
        "data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer": sha256_file(HIGHSTEER_SOURCE),
        },
        "horizon_steps": 200,
        "horizon_seconds": 5.0,
        "start_regions": [
            "7_to_9mps_high_steering",
            "simultaneous_steering_throttle_transition",
            "large_wheel_body_mismatch",
            "steering_left",
            "steering_right",
        ],
        "per_run_region_metrics": metrics,
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "Each start is grouped by its initial state/command regime; the full "
            "five-second trajectory is scored. Regions with fewer than two runs "
            "are descriptive only."),
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    candidate = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    baseline = args.baseline if args.baseline.is_absolute() else ROOT / args.baseline
    output = args.output if args.output.is_absolute() else ROOT / args.output
    analyze(candidate, baseline, output, args.device)
    print(output.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
