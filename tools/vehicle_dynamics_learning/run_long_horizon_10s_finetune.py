#!/usr/bin/env python3
"""Warm-start the 2 s-context plant and train against 5–10 s rollouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    FixedCapacityHistoryTransition,
    _draw_plan,
    _eval_l3_metrics,
    _eval_metrics,
    _normalization,
    _train_one,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    FROZEN_EVAL_STARTS,
    ROOT,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    HistoryTransition,
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


BASE_CHECKPOINT = (WP28_ROOT / "long_rollout_5s_selected_context_v1"
                   / "checkpoint.pt")
OUTPUT_ROOT = WP28_ROOT / "long_rollout_10s_warmstart_v1"
WP19_CHECKPOINT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
    / "07_body_state_increment__encoder_angle_increment/model.pt")
LONG_STAGE = (("L2", (200, 300, 400), 300, 4),)
VALIDATION_HORIZONS = (1, 20, 80, 200, 400)
HIGHSTEER_HORIZONS = (1, 10, 20, 40, 80, 120, 160, 200)
METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _training_pools_10s(data) -> dict[str, dict[str, list]]:
    by_run = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"train"}, 400, capture_indices={0})
    condition_by_sequence = data.raw_sources[0]["sequence_condition_id"]
    pools: dict[str, dict[str, list]] = {}
    for run_id in data.training_runs:
        conditions: dict[str, list] = defaultdict(list)
        for ref in by_run.get(run_id, []):
            _, sequence_index, source_row = ref
            if int(source_row) < CONTEXT_STEPS["2.0s"] - 1:
                continue
            condition = str(condition_by_sequence[int(sequence_index)])
            conditions[condition].append(ref)
        if conditions:
            pools[run_id] = dict(conditions)
    if len(pools) < 5:
        raise RuntimeError("10 s objective needs five independent training runs")
    return pools


def _evaluation_refs(data, split: str, capture_index: int,
                     expected_runs: set[str], horizon: int
                     ) -> dict[str, list[tuple[int, int, int]]]:
    by_run = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {split}, horizon, capture_indices={capture_index})
    selected = _select_eval_windows(by_run, 64)
    if set(selected) != expected_runs or any(not refs for refs in selected.values()):
        raise RuntimeError(f"{split}: eligible run roster changed at {horizon} steps")
    return selected


def _curve(data, refs, model, norm_np, config, device,
           horizons: tuple[int, ...]) -> dict[str, dict[str, dict[str, float]]]:
    return {
        str(horizon): _eval_metrics(
            data, refs, model, norm_np, config,
            CONTEXT_STEPS["2.0s"], device, horizon)
        for horizon in horizons}


def _mean_curve(curve):
    return {horizon: _macro(metrics) for horizon, metrics in curve.items()}


def run(device_name: str = "cpu", output_root: Path = OUTPUT_ROOT
        ) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)

    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    pools = _training_pools_10s(data)
    plan, sampler_counts = _draw_plan(
        pools, SEED + 400, stages=LONG_STAGE)

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    validation_refs = _evaluation_refs(
        data, "validation", 0, expected_validation, 400)
    practice_refs = _evaluation_refs(
        data, "unseen_practice", 1, expected_practice, 200)
    if HIGHSTEER_RUNS & (set(pools) | set(validation_refs) | set(practice_refs)):
        raise RuntimeError("high-steering transfer runs overlap train/selection")

    wp19_saved = torch.load(WP19_CHECKPOINT, map_location=device, weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_saved["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    l3_saved = torch.load(L3_CHECKPOINT, map_location=device, weights_only=True)
    l3 = HistoryTransition(norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    l3.load_state_dict(l3_saved["state_dict"], strict=True)

    base_saved = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=True)
    base_model = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    base_model.load_state_dict(base_saved["state_dict"], strict=True)
    base_sha = sha256_file(BASE_CHECKPOINT)
    base_expected_sha = (BASE_CHECKPOINT.parent / "checkpoint.sha256").read_text(
        encoding="utf-8").strip()
    if base_sha != base_expected_sha:
        raise RuntimeError("5 s warm-start checkpoint does not match its hash")

    base_parent_10s = _parent_metrics(
        data, data.poses[0], validation_refs, wp19, wp19_stats, 400, device)
    output_root.mkdir(parents=True)
    _write_json(output_root / "training_draw_plan.json", plan)
    _write_json(output_root / "validation_starts.json", {
        run_id: [[int(seq), int(row)] for _, seq, row in refs]
        for run_id, refs in sorted(validation_refs.items())})
    _write_json(output_root / "practice_diagnostic_starts.json", {
        run_id: [[int(seq), int(row)] for _, seq, row in refs]
        for run_id, refs in sorted(practice_refs.items())})

    candidate, training = _train_one(
        data, "2.0s_context_10s_objective", CONTEXT_STEPS["2.0s"],
        plan, validation_refs, norm_np, config, {400: base_parent_10s},
        device, stages=LONG_STAGE,
        initial_state_dict=base_saved["state_dict"])

    checkpoint_path = output_root / "checkpoint.pt"
    torch.save({
        "state_dict": {key: value.detach().cpu()
                       for key, value in candidate.state_dict().items()},
        "metadata": {
            "model": "WP28 fixed-capacity history transition",
            "context_steps": CONTEXT_STEPS["2.0s"],
            "warm_start_checkpoint_sha256": base_sha,
            "training_horizons_steps": list(LONG_STAGE[0][1]),
            "seed": SEED,
            "training_run_ids": sorted(pools),
            "validation_run_ids": sorted(validation_refs),
            "future_sensor_or_truth_inputs": False,
        },
    }, checkpoint_path)
    candidate_sha = sha256_file(checkpoint_path)
    (output_root / "checkpoint.sha256").write_text(
        candidate_sha + "\n", encoding="utf-8")

    validation_curve = _curve(
        data, validation_refs, candidate, norm_np, config, device,
        VALIDATION_HORIZONS)
    base_curve = _curve(
        data, validation_refs, base_model, norm_np, config, device,
        VALIDATION_HORIZONS)
    l3_curve = {
        str(h): _eval_l3_metrics(data, validation_refs, l3, norm_np,
                                config, device, h)
        for h in VALIDATION_HORIZONS}
    parent_curve = {
        str(h): _parent_metrics(data, data.poses[0], validation_refs,
                                wp19, wp19_stats, h, device)
        for h in VALIDATION_HORIZONS}
    practice_curve = _curve(
        data, practice_refs, candidate, norm_np, config, device, (200,))
    practice_base = _curve(
        data, practice_refs, base_model, norm_np, config, device, (200,))

    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    high_candidate = _curve(
        high_data, high_refs, candidate, norm_np, config, device,
        HIGHSTEER_HORIZONS)
    high_base = _curve(
        high_data, high_refs, base_model, norm_np, config, device,
        HIGHSTEER_HORIZONS)

    report = {
        "study": "10 s warm-start recursive objective, selected 2 s context",
        "checkpoint_sha256": candidate_sha,
        "warm_start_checkpoint_sha256": base_sha,
        "source_data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer": sha256_file(HIGHSTEER_SOURCE),
        },
        "control_and_data_rate_hz": 40.0,
        "dt_s": 0.025,
        "training_stage": {
            "horizon_choices_steps": list(LONG_STAGE[0][1]),
            "updates": LONG_STAGE[0][2],
            "batch_size": LONG_STAGE[0][3],
            "eligible_training_runs": len(pools),
            "sampler_counts": sampler_counts,
            "training": training,
        },
        "validation_run_ids": sorted(validation_refs),
        "validation_start_count_by_run": {
            run: len(refs) for run, refs in validation_refs.items()},
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "validation": {
            "candidate_per_run": validation_curve,
            "candidate_run_macro": _mean_curve(validation_curve),
            "warm_start_per_run": base_curve,
            "warm_start_run_macro": _mean_curve(base_curve),
            "L3_per_run": l3_curve,
            "L3_run_macro": _mean_curve(l3_curve),
            "WP19_per_run": parent_curve,
            "WP19_run_macro": _mean_curve(parent_curve),
            "paired_candidate_minus_warm_start_10s": {
                metric: _paired(
                    validation_curve["400"], base_curve["400"], metric)
                for metric in METRICS},
        },
        "practice_5s_diagnostic": {
            "candidate_per_run": practice_curve["200"],
            "candidate_run_macro": _macro(practice_curve["200"]),
            "warm_start_per_run": practice_base["200"],
            "warm_start_run_macro": _macro(practice_base["200"]),
        },
        "highsteer_5s_diagnostic": {
            "run_ids": sorted(HIGHSTEER_RUNS),
            "start_count_by_run": {run: len(refs)
                                    for run, refs in high_refs.items()},
            "candidate_per_run": high_candidate,
            "candidate_run_macro": _mean_curve(high_candidate),
            "warm_start_per_run": high_base,
            "warm_start_run_macro": _mean_curve(high_base),
            "paired_candidate_minus_warm_start_5s": {
                metric: _paired(
                    high_candidate["200"], high_base["200"], metric)
                for metric in METRICS},
        },
        "interpretation_limit": (
            "The six 10 s validation runs select the final checkpoint and are "
            "development evidence, not unseen final-test proof. Practice has "
            "only two 5 s runs; the independent high-steering transfer set has "
            "two runs and only supports a 5 s horizon. No full-lap accuracy "
            "claim is made."),
    }
    _write_json(output_root / "long_rollout_10s_candidate_report.json", report)
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
        "training_seconds": report["training_stage"]["training"][
            "elapsed_seconds"],
        "validation_10s_candidate": report["validation"][
            "candidate_run_macro"]["400"],
        "highsteer_5s_candidate": report["highsteer_5s_diagnostic"][
            "candidate_run_macro"]["200"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
