#!/usr/bin/env python3
"""Evaluate the selected roll-coupled candidate on independent high-steer runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    EXPECTED_RUNS,
    HIGHSTEER_SOURCE,
    _capture_from_archive,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.evaluate_rigid_acceleration_highsteer_residual_gate import (
    HIGHSTEER_VALIDATION_RUNS,
    HIGHSTEER_VALIDATION_SOURCE,
)
from tools.vehicle_dynamics_learning.evaluate_internal_roll_state_prediction import (
    _balanced_fit,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    _load_data,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    _normalization,
)
from tools.vehicle_dynamics_learning.diagnose_rigid_acceleration_sensor_residual_value import (
    _transition_rows,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
)
from tools.vehicle_dynamics_learning.train_roll_coupled_multistep_candidate import (
    LateralRollResidual,
    _balanced_run_summary,
    _eval_group,
    COM_X_M,
    DT_HORIZON,
    MAX_STARTS_PER_RUN,
    OUTPUT as CANDIDATE_DIR,
)


OUTPUT = CANDIDATE_DIR / "independent_highsteer_transfer_plus_r03_20261004.json"
INTERNAL_HOLDOUT_RUNS = {
    "openplane_dyn_coupled_train_r03_20261002",
    "practice_filter_none_12lap_20260925",
}
MAX_REFS_PER_RUN = 16


def _sample(refs: dict[str, list], capture, expected_runs: set[str],
            filter_highsteer: bool
            ) -> dict[str, list]:
    selected = {}
    for run_id, values in sorted(refs.items()):
        kept = []
        for ref in values:
            _, sequence, row = ref
            start = int(capture.bounds[sequence, 0]) + int(row)
            speed = float(np.hypot(*capture.body[start, :2]))
            steering = abs(float(capture.frames[start, 3]))
            if not filter_highsteer or (7.0 <= speed <= 9.0 and steering >= 0.30):
                kept.append(ref)
        if len(kept) > MAX_REFS_PER_RUN:
            indexes = np.linspace(0, len(kept) - 1, MAX_REFS_PER_RUN,
                                  dtype=np.int64)
            kept = [kept[int(index)] for index in indexes]
        if kept:
            selected[run_id] = kept
    if set(selected) != expected_runs:
        raise RuntimeError(
            f"high-steering transfer run roster changed: {sorted(selected)}")
    return selected


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    parent_path = (CANDIDATE_DIR.parent
                   / "rigid_acceleration_history_direct_supervision_5s_v1"
                   / "checkpoint.pt")
    parent_saved = torch.load(parent_path, map_location=device, weights_only=True)
    metadata = parent_saved["metadata"]
    parent = RigidAccelerationHistoryTransition(
        norm_np["state_mean"], norm_np["state_scale"],
        np.asarray(metadata["acceleration_mean_train_only"], dtype=np.float32),
        np.asarray(metadata["acceleration_scale_train_only"], dtype=np.float32),
        norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
        dt_s=float(metadata["dt_s"]),
        rear_axle_to_com_x_m=float(metadata["rear_axle_to_com_x_m"])).to(device)
    parent.load_state_dict(parent_saved["state_dict"], strict=True)
    candidate_saved = torch.load(CANDIDATE_DIR / "checkpoint.pt",
                                 map_location=device, weights_only=True)
    candidate = LateralRollResidual().to(device)
    candidate.load_state_dict(candidate_saved["state_dict"], strict=True)

    train_rows = _transition_rows(
        data, 0, "train", DEFAULT_DYNAMIC, config,
        allowed_run_ids=set(data.training_runs) - INTERNAL_HOLDOUT_RUNS)
    roll_model = _balanced_fit(train_rows)
    results = {}
    for source_name, source_path, expected_runs in (
            ("preserved_two_run_highsteer", HIGHSTEER_SOURCE, EXPECTED_RUNS),
            ("new_three_run_highsteer", HIGHSTEER_VALIDATION_SOURCE,
             HIGHSTEER_VALIDATION_RUNS)):
        high_capture, high_pose = _capture_from_archive(
            source_path, expected_runs)
        high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
        with np.load(source_path, allow_pickle=False) as archive:
            attitude = np.asarray(archive["imu_attitude_frames"],
                                  dtype=np.float32)
            attitude_valid = np.asarray(archive["imu_attitude_valid"],
                                        dtype=bool)
        all_refs = _sample_all_valid_context_refs(
            high_capture, DT_HORIZON, history_steps=80,
            expected_runs=expected_runs)
        samples = {
            "all_feasible_highsteer_holdout_starts": _sample(
                all_refs, high_capture, expected_runs, filter_highsteer=False),
            "7_to_9mps_abs_steer_at_least_0p30rad": _sample(
                all_refs, high_capture, expected_runs, filter_highsteer=True),
        }
        results[source_name] = {}
        for region, refs in samples.items():
            cand, base, roll = _eval_group(
                high_data, 0, "validation", DT_HORIZON, parent, candidate,
                roll_model, norm_np, config, attitude, attitude_valid, device,
                start_refs=refs)
            results[source_name][region] = {
                "candidate_minus_parent_run_macro_summary":
                    _balanced_run_summary(cand, base),
                "candidate_per_run": cand,
                "parent_per_run": base,
                "roll_per_run": roll,
                "starts_per_run": {run: len(rows) for run, rows in refs.items()},
            }
    report = {
        "study": "5-second recursive transfer to whole high-steering holdout runs",
        "candidate_checkpoint_sha256": sha256_file(CANDIDATE_DIR / "checkpoint.pt"),
        "frozen_parent_checkpoint_sha256": sha256_file(parent_path),
        "highsteer_sources": {
            "preserved_two_run": {
                "path": HIGHSTEER_SOURCE.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(HIGHSTEER_SOURCE),
                "run_ids": sorted(EXPECTED_RUNS),
            },
            "new_three_run": {
                "path": HIGHSTEER_VALIDATION_SOURCE.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(HIGHSTEER_VALIDATION_SOURCE),
                "run_ids": sorted(HIGHSTEER_VALIDATION_RUNS),
            },
        },
        "training_runs_for_roll_model": sorted(
            set(data.training_runs) - INTERNAL_HOLDOUT_RUNS),
        "prediction_policy": (
            "roll/rate initialized from the first held-out sample, then both "
            "roll and body are recursively predicted; only future command input "
            "is read"),
        "future_measured_body_or_imu_used": False,
        "regions_by_source": results,
        "final_test_touched": False,
        "production_integration": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "regions": {
            source: {region: values[
                "candidate_minus_parent_run_macro_summary"]
                for region, values in regions.items()}
            for source, regions in report["regions_by_source"].items()},
    }, indent=2))
