#!/usr/bin/env python3
"""Test a physics-integrated acceleration transition against WP28 baselines.

Research-only candidate: a causal fixed 2 s history model predicts planar
body-frame acceleration, which is integrated through the exact rigid-body
kinematics used to define the transition labels. No runtime stack is changed.
"""

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
    _parent_metrics,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    RigidAccelerationHistoryTransition,
    midpoint_acceleration_from_transition,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    L3_CHECKPOINT,
    OUTPUT_ROOT as WP28_ROOT,
    SEED,
    _draw_plan,
    _eval_l3_metrics,
    _normalization,
    _training_pools,
    _train_one,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)
from tools.vehicle_dynamics_learning.run_long_horizon_context_candidate import (
    LONG_STAGES,
    METRICS,
    _curve,
    _eval_refs,
    _model_load,
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


OUTPUT_ROOT = (WP28_ROOT / "rigid_acceleration_history_5s_v1")
EVAL_HORIZONS = (1, 10, 20, 40, 80, 120, 160, 200)
COM_X_M = 0.15532


def _acceleration_normalization(data) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Run-balanced acceleration moments using training transitions only."""
    by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    for run_id in sorted(data.training_runs):
        refs = data.train_windows_by_horizon[20].get(run_id, [])
        if not refs:
            continue
        selected = np.linspace(0, len(refs) - 1,
                               min(4096, len(refs)), dtype=np.int64)
        for index in selected:
            capture_index, sequence_index, source_row = refs[int(index)]
            capture = data.captures[capture_index]
            begin = int(capture.bounds[sequence_index, 0]) + int(source_row)
            value = midpoint_acceleration_from_transition(
                capture.body[begin], capture.body[begin + 1], 0.025, COM_X_M)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"invalid acceleration label in training run {run_id}")
            by_run[run_id].append(value)
    groups = {run: np.stack(rows) for run, rows in by_run.items() if rows}
    if len(groups) < 5:
        raise RuntimeError("acceleration normalization requires five training runs")
    means = np.stack([values.mean(axis=0) for values in groups.values()])
    mean = means.mean(axis=0)
    variance = np.mean([
        np.mean((values - mean) ** 2, axis=0) for values in groups.values()],
        axis=0)
    scale = np.maximum(np.sqrt(variance), 0.1)
    return mean.astype(np.float32), scale.astype(np.float32), {
        run: len(values) for run, values in sorted(groups.items())}


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
    acceleration_mean, acceleration_scale, acceleration_rows = (
        _acceleration_normalization(data))
    short_pools, _ = _training_pools(data, horizon_steps=80)
    long_pools, _ = _training_pools(data, horizon_steps=200)
    plan, sampler_counts = _draw_plan(
        short_pools, SEED + 1, stages=LONG_STAGES,
        pools_by_stage={"L2": long_pools})

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    validation_refs = _eval_refs(data, "validation", 0, expected_validation)
    practice_refs = _eval_refs(data, "unseen_practice", 1, expected_practice)
    if HIGHSTEER_RUNS & (set(data.training_runs) | set(validation_refs)
                         | set(practice_refs)):
        raise RuntimeError("high-steering holdout overlaps train/dev/practice")

    wp19_path = (ROOT / "live_runs/derived_dynamics_learning_20260928"
                 / "full_modeling_reset_20261001"
                 / "replacement_offline_sim_raceline_20261003"
                 / "full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
                 / "07_body_state_increment__encoder_angle_increment/model.pt")
    wp19_saved = torch.load(wp19_path, map_location=device, weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_saved["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)

    l3_saved = torch.load(L3_CHECKPOINT, map_location=device, weights_only=True)
    from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
        HistoryTransition,
    )
    from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
        FixedCapacityHistoryTransition,
    )

    l3 = HistoryTransition(norm_np["delta_mean"],
                           norm_np["delta_scale"]).to(device)
    l3.load_state_dict(l3_saved["state_dict"], strict=True)
    parent_by_horizon = {
        horizon: _parent_metrics(
            data, data.poses[0], validation_refs, wp19, wp19_stats,
            horizon, device)
        for horizon in (1, 20, 200)}

    base_path = WP28_ROOT / "long_rollout_5s_selected_context_v1/checkpoint.pt"
    base_model = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    base_sha = _model_load(base_path, base_model, device)
    l3_sha = sha256_file(L3_CHECKPOINT)

    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])

    output_root.mkdir(parents=True)
    candidate, training = _train_one(
        data, "rigid_acceleration_2s_context_5s_rollout",
        CONTEXT_STEPS["2.0s"], plan, validation_refs, norm_np, config,
        parent_by_horizon, device, stages=LONG_STAGES,
        model_factory=lambda: RigidAccelerationHistoryTransition(
            norm_np["state_mean"], norm_np["state_scale"],
            acceleration_mean, acceleration_scale,
            norm_np["delta_mean"][3:], norm_np["delta_scale"][3:],
            dt_s=0.025, rear_axle_to_com_x_m=COM_X_M))

    checkpoint_path = output_root / "checkpoint.pt"
    torch.save({
        "state_dict": {name: value.detach().cpu()
                       for name, value in candidate.state_dict().items()},
        "metadata": {
            "model": "WP28 history-conditioned acceleration with explicit planar integration",
            "context_steps": CONTEXT_STEPS["2.0s"],
            "control_and_data_rate_hz": 40.0,
            "dt_s": 0.025,
            "rear_axle_to_com_x_m": COM_X_M,
            "acceleration_channels": ["body_com_ax_mps2",
                                      "body_com_ay_mps2", "yaw_accel_rps2"],
            "acceleration_mean_train_only": acceleration_mean.tolist(),
            "acceleration_scale_train_only": acceleration_scale.tolist(),
            "acceleration_supervision_weight": 1.0,
            "training_run_ids": sorted(data.training_runs),
            "validation_run_ids": sorted(validation_refs),
            "future_sensor_or_truth_inputs": False,
            "production_integration": False,
        },
    }, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    (output_root / "checkpoint.sha256").write_text(
        checkpoint_sha + "\n", encoding="utf-8")

    candidate_validation = _curve(
        data, validation_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    baseline_validation = _curve(
        data, validation_refs, base_model, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    candidate_practice = _curve(
        data, practice_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    candidate_high = _curve(
        high_data, high_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    baseline_high = _curve(
        high_data, high_refs, base_model, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)

    parent_validation = {
        str(horizon): _parent_metrics(
            data, data.poses[0], validation_refs, wp19, wp19_stats,
            horizon, device)
        for horizon in EVAL_HORIZONS}
    l3_validation = {
        str(horizon): _eval_l3_metrics(
            data, validation_refs, l3, norm_np, config, device, horizon)
        for horizon in EVAL_HORIZONS}
    l3_high = {
        str(horizon): _eval_l3_metrics(
            high_data, high_refs, l3, norm_np, config, device, horizon)
        for horizon in EVAL_HORIZONS}
    parent_high = {
        str(horizon): _parent_metrics(
            high_data, high_pose, high_refs, wp19, wp19_stats,
            horizon, device)
        for horizon in EVAL_HORIZONS}

    report: dict[str, Any] = {
        "study": "explicit planar rigid-body acceleration-integrated recursive plant",
        "checkpoint_sha256": checkpoint_sha,
        "baseline_wp28_checkpoint_sha256": base_sha,
        "l3_checkpoint_sha256": l3_sha,
        "data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer": sha256_file(HIGHSTEER_SOURCE),
        },
        "sample_rate_hz": 40.0,
        "dt_s": 0.025,
        "planar_geometry_rear_axle_to_com_x_m": COM_X_M,
        "midpoint_acceleration_normalization_train_rows_by_run": acceleration_rows,
        "sampler_counts": sampler_counts,
        "training": training,
        "validation_runs": sorted(validation_refs),
        "validation_candidate_per_run": candidate_validation,
        "validation_candidate_macro": {
            horizon: _macro(values)
            for horizon, values in candidate_validation.items()},
        "validation_baselines": {
            "WP28_5s_per_run": baseline_validation,
            "WP28_5s_macro": {
                horizon: _macro(values)
                for horizon, values in baseline_validation.items()},
            "L3_per_run": l3_validation,
            "L3_macro": {h: _macro(v) for h, v in l3_validation.items()},
            "WP19_per_run": parent_validation,
            "WP19_macro": {h: _macro(v) for h, v in parent_validation.items()},
        },
        "validation_paired_delta_candidate_minus_wp28": {
            horizon: {metric: _paired(
                baseline_validation[horizon], candidate_validation[horizon], metric)
                for metric in METRICS}
            for horizon in candidate_validation},
        "practice_unseen_run_diagnostic_per_run": candidate_practice,
        "highsteer_independent_runs": sorted(HIGHSTEER_RUNS),
        "highsteer_candidate_per_run": candidate_high,
        "highsteer_candidate_macro": {
            horizon: _macro(values) for horizon, values in candidate_high.items()},
        "highsteer_baselines": {
            "WP28_5s_per_run": baseline_high,
            "WP28_5s_macro": {h: _macro(v) for h, v in baseline_high.items()},
            "L3_per_run": l3_high,
            "L3_macro": {h: _macro(v) for h, v in l3_high.items()},
            "WP19_per_run": parent_high,
            "WP19_macro": {h: _macro(v) for h, v in parent_high.items()},
        },
        "highsteer_paired_delta_candidate_minus_wp28": {
            horizon: {metric: _paired(
                baseline_high[horizon], candidate_high[horizon], metric)
                for metric in METRICS}
            for horizon in candidate_high},
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "This compares 5 s free-recursive body/pose prediction against held-out "
            "whole runs and two independent high-steering runs. It is not a full-lap "
            "validation and does not establish a production odometry or MPC model."),
    }
    _write_json(output_root / "rigid_acceleration_candidate_report.json", report)
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
        "checkpoint_sha256": report["checkpoint_sha256"],
        "training_seconds": report["training"]["elapsed_seconds"],
        "validation_candidate_5s": report["validation_candidate_macro"]["200"],
        "validation_wp28_5s": report["validation_baselines"]["WP28_5s_macro"]["200"],
        "highsteer_candidate_5s": report["highsteer_candidate_macro"]["200"],
        "highsteer_wp28_5s": report["highsteer_baselines"]["WP28_5s_macro"]["200"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
