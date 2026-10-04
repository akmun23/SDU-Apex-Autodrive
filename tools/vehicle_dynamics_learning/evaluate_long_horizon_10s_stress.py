#!/usr/bin/env python3
"""Score frozen 2 s-history candidates on 10 s validation rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    _macro,
    _parent_metrics,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    L3_CHECKPOINT,
    OUTPUT_ROOT as WP28_ROOT,
    FixedCapacityHistoryTransition,
    _bootstrap_delta,
    _eval_l3_metrics,
    _eval_metrics,
    _normalization,
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


LONG_CANDIDATE = (WP28_ROOT / "long_rollout_5s_selected_context_v1"
                  / "checkpoint.pt")
OLD_WP28 = WP28_ROOT / "2.0s" / "checkpoint.pt"
WP19_CHECKPOINT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
    / "07_body_state_increment__encoder_angle_increment/model.pt")
OUTPUT = (WP28_ROOT / "long_rollout_5s_selected_context_v1"
          / "ten_second_recursive_stress_v1.json")
METRICS = (
    "position_radial_trajectory_rmse_m",
    "position_endpoint_error_m",
    "heading_trajectory_rmse_rad",
    "heading_endpoint_error_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def evaluate(device_name: str = "cpu", output_path: Path = OUTPUT):
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    by_run = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"validation"}, 400, capture_indices={0})
    refs = _select_eval_windows(by_run, 64)
    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected = set(frozen["split_roles"]["development_validation"])
    if set(refs) != expected or any(not run_refs for run_refs in refs.values()):
        raise RuntimeError("10 s validation runs differ from the frozen roster")

    def load_context_checkpoint(path: Path) -> tuple[FixedCapacityHistoryTransition, str]:
        saved = torch.load(path, map_location=device, weights_only=True)
        model = FixedCapacityHistoryTransition(
            norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        return model, sha256_file(path)

    candidate, candidate_sha = load_context_checkpoint(LONG_CANDIDATE)
    previous, previous_sha = load_context_checkpoint(OLD_WP28)

    wp19_saved = torch.load(WP19_CHECKPOINT, map_location=device, weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_saved["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    l3_saved = torch.load(L3_CHECKPOINT, map_location=device, weights_only=True)
    l3 = HistoryTransition(norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    l3.load_state_dict(l3_saved["state_dict"], strict=True)

    models = {
        "long_objective_WP28": _eval_metrics(
            data, refs, candidate, norm_np, config,
            CONTEXT_STEPS["2.0s"], device, 400),
        "previous_WP28": _eval_metrics(
            data, refs, previous, norm_np, config,
            CONTEXT_STEPS["2.0s"], device, 400),
        "L3_5s": _eval_l3_metrics(data, refs, l3, norm_np, config, device, 400),
        "WP19_parent": _parent_metrics(
            data, data.poses[0], refs, wp19, wp19_stats, 400, device),
    }
    deltas = {}
    for baseline in ("previous_WP28", "L3_5s", "WP19_parent"):
        deltas[baseline] = {
            metric: _bootstrap_delta(
                models["long_objective_WP28"], models[baseline], metric,
                20261029 + len(metric) + len(baseline))
            for metric in METRICS}
    report = {
        "study": "10 s recursive validation stress; 40 Hz model/control cadence",
        "horizon_steps": 400,
        "horizon_seconds": 10.0,
        "dt_s": 0.025,
        "validation_run_ids": sorted(refs),
        "start_count_by_run": {run: len(run_refs)
                                for run, run_refs in refs.items()},
        "independent_unit": "whole simulator capture run",
        "source_data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
        },
        "checkpoint_sha256": {
            "long_objective_WP28": candidate_sha,
            "previous_WP28": previous_sha,
            "L3_5s": sha256_file(L3_CHECKPOINT),
            "WP19_parent": sha256_file(WP19_CHECKPOINT),
        },
        "per_run_metrics": models,
        "run_macro_metrics": {name: _macro(values)
                              for name, values in models.items()},
        "paired_long_objective_candidate_minus_baseline": deltas,
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "These whole-run validation captures also participated in 5 s "
            "checkpoint selection. The 10 s horizon was not a selection target, "
            "so treat this as an extended development stress test, not final "
            "unseen-run proof."),
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = evaluate(args.device, output)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "run_macro_metrics": report["run_macro_metrics"],
        "paired_candidate_vs_previous_WP28": {
            metric: values["candidate_minus_reference_macro_delta"]
            for metric, values in report[
                "paired_long_objective_candidate_minus_baseline"][
                    "previous_WP28"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
