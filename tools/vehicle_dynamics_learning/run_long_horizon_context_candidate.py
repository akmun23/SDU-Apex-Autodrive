#!/usr/bin/env python3
"""Train one WP28 2 s-context plant with an explicit 5 s rollout objective.

This is a targeted recursive-error experiment. It retains the WP28 model,
normalization, optimizer, seed, short stages, and run-balanced sampler. Only
the final recursive horizon is extended from at most 2 s to 2–5 s. It does
not modify production odometry, MPC, localization, or simulator physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    L3_CHECKPOINT,
    OUTPUT_ROOT as WP28_ROOT,
    SEED,
    STAGES,
    FixedCapacityHistoryTransition,
    _draw_plan,
    _eval_l3_metrics,
    _eval_metrics,
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


OUTPUT_ROOT = WP28_ROOT / "long_rollout_5s_selected_context_v1"
LONG_STAGES = (
    STAGES[0],
    STAGES[1],
    ("L2", (80, 120, 160, 200), STAGES[2][2], STAGES[2][3]),
)
EVAL_HORIZONS = (1, 10, 20, 40, 80, 120, 160, 200)
METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _eval_refs(data, split: str, capture_index: int,
               expected_runs: set[str]) -> dict[str, list[tuple[int, int, int]]]:
    by_run = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses, {split}, 200,
        capture_indices={capture_index})
    selected = _select_eval_windows(by_run, 64)
    if set(selected) != expected_runs:
        raise RuntimeError(
            f"{split} 5 s evaluation run roster changed: "
            f"expected={sorted(expected_runs)} actual={sorted(selected)}")
    if any(not refs for refs in selected.values()):
        raise RuntimeError(f"{split} contains an empty 5 s evaluation run")
    return selected


def _model_load(path: Path, model: FixedCapacityHistoryTransition,
                device: torch.device) -> str:
    saved = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    return sha256_file(path)


def _curve(data, refs, model, norm_np, config, context_steps,
           device) -> dict[str, dict[str, dict[str, float]]]:
    return {
        str(horizon): _eval_metrics(
            data, refs, model, norm_np, config, context_steps, device, horizon)
        for horizon in EVAL_HORIZONS}


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
    if HIGHSTEER_RUNS & set(data.training_runs):
        raise RuntimeError("high-steering evaluation runs overlap training")
    if HIGHSTEER_RUNS & (expected_validation | expected_practice):
        raise RuntimeError("high-steering evaluation overlaps checkpoint selection")

    wp19_saved = torch.load(
        (ROOT / "live_runs/derived_dynamics_learning_20260928"
         / "full_modeling_reset_20261001"
         / "replacement_offline_sim_raceline_20261003"
         / "full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
         / "07_body_state_increment__encoder_angle_increment/model.pt"),
        map_location=device, weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_saved["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    l3_saved = torch.load(L3_CHECKPOINT, map_location=device, weights_only=True)
    from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
        HistoryTransition,
    )
    l3 = HistoryTransition(norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    l3.load_state_dict(l3_saved["state_dict"], strict=True)

    parents: dict[int, dict[str, dict[str, float]]] = {}
    l3_metrics: dict[int, dict[str, dict[str, float]]] = {}
    for horizon in (1, 20, 200):
        parents[horizon] = _parent_metrics(
            data, data.poses[0], validation_refs,
            wp19, wp19_stats, horizon, device)
        l3_metrics[horizon] = _eval_l3_metrics(
            data, validation_refs, l3, norm_np, config, device, horizon)

    old_checkpoint = WP28_ROOT / "2.0s" / "checkpoint.pt"
    old_wp28 = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    old_wp28_sha = _model_load(old_checkpoint, old_wp28, device)

    output_root.mkdir(parents=True)
    validation_manifest = {
        run_id: [[int(seq), int(row)] for _, seq, row in refs]
        for run_id, refs in sorted(validation_refs.items())}
    practice_manifest = {
        run_id: [[int(seq), int(row)] for _, seq, row in refs]
        for run_id, refs in sorted(practice_refs.items())}
    _write_json(output_root / "validation_starts.json", validation_manifest)
    _write_json(output_root / "practice_diagnostic_starts.json", practice_manifest)
    _write_json(output_root / "training_draw_plan.json", plan)

    parent_by_horizon = {h: parents[h] for h in (1, 20, 200)}
    candidate, training = _train_one(
        data, "2.0s_long_rollout", CONTEXT_STEPS["2.0s"], plan,
        validation_refs, norm_np, config, parent_by_horizon, device,
        stages=LONG_STAGES)

    checkpoint_path = output_root / "checkpoint.pt"
    checkpoint = {
        "state_dict": {name: value.detach().cpu()
                       for name, value in candidate.state_dict().items()},
        "metadata": {
            "model": "WP28 fixed-capacity history transition",
            "context_steps": CONTEXT_STEPS["2.0s"],
            "training_horizons_steps": [1, 4, 10, 20, 40, 80, 120, 160, 200],
            "seed": SEED,
            "training_run_ids": sorted(data.training_runs),
            "validation_run_ids": sorted(validation_refs),
            "future_sensor_or_truth_inputs": False,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    (output_root / "checkpoint.sha256").write_text(
        checkpoint_sha + "\n", encoding="utf-8")

    validation_curve = _curve(
        data, validation_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    old_curve = _curve(
        data, validation_refs, old_wp28, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    l3_curve = {
        str(h): _eval_l3_metrics(
            data, validation_refs, l3, norm_np, config, device, h)
        for h in EVAL_HORIZONS}
    parent_curve = {
        str(h): _parent_metrics(
            data, data.poses[0], validation_refs, wp19, wp19_stats, h, device)
        for h in EVAL_HORIZONS}
    practice_curve = _curve(
        data, practice_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)

    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    high_curve = _curve(
        high_data, high_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device)
    high_baselines = {
        "WP19_parent": {
            str(h): _parent_metrics(
                high_data, high_pose, high_refs, wp19, wp19_stats, h, device)
            for h in EVAL_HORIZONS},
        "L3_5s": {
            str(h): _eval_l3_metrics(
                high_data, high_refs, l3, norm_np, config, device, h)
            for h in EVAL_HORIZONS},
        "WP28_2s_context": _curve(
            high_data, high_refs, old_wp28, norm_np, config,
            CONTEXT_STEPS["2.0s"], device),
    }

    report = {
        "study": "WP28 selected 2 s context with 5 s recursive training objective",
        "checkpoint_sha256": checkpoint_sha,
        "parent_wp28_checkpoint_sha256": old_wp28_sha,
        "source_data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer_capture": sha256_file(HIGHSTEER_SOURCE),
        },
        "control_and_data_rate_hz": 40.0,
        "dt_s": 0.025,
        "selected_context_steps": CONTEXT_STEPS["2.0s"],
        "stage_schedule": [
            {"stage": name, "horizon_choices_steps": list(choices),
             "updates": updates, "batch_size": batch}
            for name, choices, updates, batch in LONG_STAGES],
        "training_pool_run_counts": {
            "short_stages_eligible_runs": len(short_pools),
            "long_stage_eligible_runs": len(long_pools),
            "sampler_counts": sampler_counts,
        },
        "training": training,
        "validation_run_ids": sorted(validation_refs),
        "validation_start_count_by_run": {
            run: len(refs) for run, refs in validation_refs.items()},
        "test_or_final_test_opened": False,
        "highsteer_runs_used_for_training_or_selection": False,
        "simulator_launched": False,
        "production_integration": False,
        "validation_curve": {
            "long_objective_candidate_per_run": validation_curve,
            "long_objective_candidate_run_macro": {
                h: _macro(values) for h, values in validation_curve.items()},
            "previous_WP28_per_run": old_curve,
            "previous_WP28_run_macro": {
                h: _macro(values) for h, values in old_curve.items()},
            "L3_per_run": l3_curve,
            "L3_run_macro": {h: _macro(values) for h, values in l3_curve.items()},
            "WP19_per_run": parent_curve,
            "WP19_run_macro": {h: _macro(values) for h, values in parent_curve.items()},
            "paired_run_deltas_candidate_minus_baseline": {
                baseline: {
                    h: {metric: _paired(
                        baseline_curve[h], validation_curve[h], metric)
                        for metric in METRICS}
                    for h in validation_curve}
                for baseline, baseline_curve in (
                    ("previous_WP28", old_curve), ("L3_5s", l3_curve),
                    ("WP19", parent_curve))},
        },
        "practice_transfer_diagnostic_run_macro": {
            h: _macro(values) for h, values in practice_curve.items()},
        "highsteer_holdout": {
            "run_ids": sorted(HIGHSTEER_RUNS),
            "start_count_by_run": {
                run: len(refs) for run, refs in high_refs.items()},
            "candidate_per_run": high_curve,
            "candidate_run_macro": {
                h: _macro(values) for h, values in high_curve.items()},
            "baselines": high_baselines,
            "paired_candidate_minus_baseline": {
                baseline: {
                    h: {metric: _paired(
                        values[h], high_curve[h], metric)
                        for metric in METRICS}
                    for h in map(str, EVAL_HORIZONS)}
                for baseline, values in high_baselines.items()},
        },
        "interpretation_limit": (
            "The development validation set chooses checkpoints and is not an "
            "independent final test. Practice has two runs. The preserved high-"
            "steer capture is independent of this candidate's training and "
            "checkpoint selection but contains only two runs. No full-lap claim "
            "is made."),
    }
    _write_json(output_root / "long_rollout_candidate_report.json", report)
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
        "validation_5s_candidate": report["validation_curve"][
            "long_objective_candidate_run_macro"]["200"],
        "highsteer_5s_candidate": report["highsteer_holdout"][
            "candidate_run_macro"]["200"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
