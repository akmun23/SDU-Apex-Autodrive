#!/usr/bin/env python3
"""Check whether 40 Hz rigid-body truth integrates to simulator truth pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    PoseIntegrator,
)
from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    EXPECTED_RUNS as HIGHSTEER_RUNS,
    HIGHSTEER_SOURCE,
    _capture_from_archive,
    _sample_all_valid_context_refs,
    _macro,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    _batch_arrays,
    _normalization,
    _metrics,
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


OUTPUT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
          / "full_modeling_reset_20261001"
          / "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1"
          / "next_phase_after_2129427/history_context_sufficiency_v1"
          / "truth_pose_integration_diagnostic_v1.json")
HORIZONS = (1, 10, 20, 40, 80, 120, 160, 200)


def _oracle_metrics(data, refs, norm_np, config, device
                    ) -> dict[str, dict[str, float]]:
    arrays = _batch_arrays(
        data, refs, max(HORIZONS), CONTEXT_STEPS["2.0s"],
        norm_np, config, device)
    _, _, start_state, start_pose, _, targets, target_poses = arrays
    state_scale = torch.as_tensor(norm_np["state_scale"], dtype=torch.float32,
                                  device=device)
    state_mean = torch.as_tensor(norm_np["state_mean"], dtype=torch.float32,
                                 device=device)
    current_state = start_state * state_scale + state_mean
    truth_states = targets * state_scale + state_mean
    pose = start_pose
    integrator = PoseIntegrator(DT_S).to(device)
    output = {}
    for step in range(max(HORIZONS)):
        next_state = truth_states[:, step]
        midpoint_body = 0.5 * (current_state[:, :3] + next_state[:, :3])
        pose = integrator(pose, midpoint_body)
        current_state = next_state
        if step == 0:
            predicted_poses = [pose]
        else:
            predicted_poses.append(pose)
        if step + 1 in HORIZONS:
            horizon = step + 1
            prediction = torch.stack(predicted_poses, dim=1)
            truth = truth_states[:, :horizon].cpu().numpy()
            output[str(horizon)] = _metrics(
                truth, prediction.cpu().numpy(), truth,
                target_poses[:, :horizon].cpu().numpy(), horizon)
    return output


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    validation = _eval_refs(
        data, "validation", 0,
        set(frozen["split_roles"]["development_validation"]))
    practice = _eval_refs(
        data, "unseen_practice", 1,
        set(frozen["split_roles"]["practice_diagnostic"]))
    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, max(HORIZONS), history_steps=80)
    high = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    groups = {"whole_run_development": (data, validation),
              "unseen_practice": (data, practice),
              "independent_highsteer": (high, high_refs)}
    metrics = {
        name: {
            run_id: _oracle_metrics(group_data, run_refs, norm_np, config, device)
            for run_id, run_refs in sorted(refs.items())}
        for name, (group_data, refs) in groups.items()}
    report = {
        "study": "oracle 40 Hz body-state integration against simulator pose",
        "sample_period_s": DT_S,
        "sample_rate_hz": 1.0 / DT_S,
        "body_state_channels": ["u_rear_mps", "v_rear_mps", "yaw_rate_rps"],
        "integrator": "midpoint truth body state + exact constant-twist SE(2) pose step",
        "horizons_steps": list(HORIZONS),
        "independent_runs_by_group": {
            name: sorted(refs) for name, (_, refs) in groups.items()},
        "per_run_metrics": metrics,
        "run_macro_metrics": {
            name: {
                str(horizon): _macro({
                    run_id: values[str(horizon)]
                    for run_id, values in horizons.items()})
                for horizon in HORIZONS}
            for name, horizons in metrics.items()},
        "data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer": sha256_file(HIGHSTEER_SOURCE),
        },
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "This tests only whether the current 40 Hz rigid body state and pose "
            "conventions are mutually consistent under the existing pose integrator. "
            "It does not test acceleration prediction or sub-25-ms dynamics."),
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = run(output)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "position_by_group": {
            name: {h: values["position_radial_trajectory_rmse_m"]
                  for h, values in horizons.items()}
            for name, horizons in report["run_macro_metrics"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
