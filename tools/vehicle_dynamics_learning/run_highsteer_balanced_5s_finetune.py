#!/usr/bin/env python3
"""Test whether evidence-backed high-steer oversampling improves transfer."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

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
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    OUTPUT_ROOT as WP28_ROOT,
    SEED,
    FixedCapacityHistoryTransition,
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
from tools.vehicle_dynamics_learning.operating_regions import region_masks


BASE_CHECKPOINT = (WP28_ROOT / "long_rollout_5s_selected_context_v1"
                   / "checkpoint.pt")
OUTPUT_ROOT = WP28_ROOT / "highsteer_balanced_5s_finetune_v1"
WP19_CHECKPOINT = (
    ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001"
    / "replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
    / "07_body_state_increment__encoder_angle_increment/model.pt")
STAGE = (("L2", (80, 120, 160, 200), 300, 8),)
EVAL_HORIZONS = (20, 80, 200)
METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _highsteer_pool(data, global_pools):
    capture = data.captures[0]
    reset = np.full(len(capture.frames), -1, dtype=np.int32)
    for sequence_index, (begin, end) in enumerate(capture.bounds):
        reset[int(begin):int(end)] = capture.sequence_reset[sequence_index]
    valid = (np.isfinite(capture.frames).all(axis=1)
             & np.isfinite(capture.body).all(axis=1)
             & np.isclose(capture.dt_s, 0.025, rtol=0.0, atol=1e-7))
    masks = region_masks(capture.frames, reset, capture.packet, valid, 0.025)
    selected: dict[str, dict[str, list]] = {}
    for run_id, conditions in global_pools.items():
        run_conditions = {}
        for condition, refs in conditions.items():
            rows = []
            for ref in refs:
                _, sequence_index, source_row = ref
                absolute_row = int(capture.bounds[sequence_index, 0]) + int(source_row)
                if bool(masks["7_to_9mps_high_steering"][absolute_row]):
                    rows.append(ref)
            if rows:
                run_conditions[condition] = rows
        if run_conditions:
            selected[run_id] = run_conditions
    if len(selected) < 2:
        raise RuntimeError("fewer than two training runs cover 7–9 m/s high steering")
    return selected


def _balanced_plan(global_pool, high_pool, seed: int):
    rng = np.random.default_rng(seed)
    stage, horizons, updates, batch_size = STAGE[0]
    if batch_size % 2:
        raise ValueError("balanced sampler requires even batch size")
    plan = []
    counts = {"global": Counter(), "high_steer": Counter()}
    for update in range(1, updates + 1):
        horizon = int(horizons[int(rng.integers(len(horizons)))])
        batch = []
        for source, pools in (("high_steer", high_pool),
                              ("global", global_pool)):
            run_ids = sorted(pools)
            for _ in range(batch_size // 2):
                run_id = run_ids[int(rng.integers(len(run_ids)))]
                conditions = sorted(pools[run_id])
                condition = conditions[int(rng.integers(len(conditions)))]
                refs = pools[run_id][condition]
                ref = refs[int(rng.integers(len(refs)))]
                batch.append([int(value) for value in ref])
                counts[source][f"{run_id}:{condition}"] += 1
        rng.shuffle(batch)
        plan.append({"stage": stage, "update": update,
                     "horizon_steps": horizon,
                     "refs": batch,
                     "samples_per_update": {
                         "high_steer": batch_size // 2,
                         "global": batch_size // 2}})
    return plan, {key: dict(value) for key, value in counts.items()}


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
    global_pool, _ = _training_pools(data, horizon_steps=200)
    high_pool = _highsteer_pool(data, global_pool)
    plan, sampler_counts = _balanced_plan(global_pool, high_pool, SEED + 520)

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    expected_validation = set(frozen["split_roles"]["development_validation"])
    expected_practice = set(frozen["split_roles"]["practice_diagnostic"])
    validation_all = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"validation"}, 200, capture_indices={0})
    validation_refs = _select_eval_windows(validation_all, 64)
    practice_all = _collect_horizon_windows(
        data.captures, data.raw_sources, data.poses,
        {"unseen_practice"}, 200, capture_indices={1})
    practice_refs = _select_eval_windows(practice_all, 64)
    if set(validation_refs) != expected_validation or set(practice_refs) != expected_practice:
        raise RuntimeError("5 s balanced-finetune evaluation run roster changed")
    if HIGHSTEER_RUNS & (set(data.training_runs) | set(validation_refs) | set(practice_refs)):
        raise RuntimeError("high-steering transfer runs overlap training/selection")

    base_expected = (BASE_CHECKPOINT.parent / "checkpoint.sha256").read_text(
        encoding="utf-8").strip()
    base_sha = sha256_file(BASE_CHECKPOINT)
    if base_sha != base_expected:
        raise RuntimeError("5 s warm-start checkpoint does not match its hash")
    base_saved = torch.load(BASE_CHECKPOINT, map_location=device, weights_only=True)
    base_model = FixedCapacityHistoryTransition(
        norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
    base_model.load_state_dict(base_saved["state_dict"], strict=True)

    wp19_saved = torch.load(WP19_CHECKPOINT, map_location=device, weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_saved["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    parent_metrics = _parent_metrics(
        data, data.poses[0], validation_refs, wp19, wp19_stats, 200, device)

    output_root.mkdir(parents=True)
    _write_json(output_root / "training_draw_plan.json", plan)
    _write_json(output_root / "validation_starts.json", {
        run: [[int(seq), int(row)] for _, seq, row in refs]
        for run, refs in sorted(validation_refs.items())})
    _write_json(output_root / "practice_diagnostic_starts.json", {
        run: [[int(seq), int(row)] for _, seq, row in refs]
        for run, refs in sorted(practice_refs.items())})

    candidate, training = _train_one(
        data, "2.0s_context_highsteer_balanced", CONTEXT_STEPS["2.0s"],
        plan, validation_refs, norm_np, config, {200: parent_metrics},
        device, stages=STAGE,
        initial_state_dict=base_saved["state_dict"])

    checkpoint_path = output_root / "checkpoint.pt"
    torch.save({
        "state_dict": {name: value.detach().cpu()
                       for name, value in candidate.state_dict().items()},
        "metadata": {
            "model": "WP28 fixed-capacity history transition",
            "context_steps": CONTEXT_STEPS["2.0s"],
            "warm_start_checkpoint_sha256": base_sha,
            "training_horizons_steps": list(STAGE[0][1]),
            "highsteer_samples_per_batch": 4,
            "global_samples_per_batch": 4,
            "highsteer_training_run_ids": sorted(high_pool),
            "validation_run_ids": sorted(validation_refs),
            "future_sensor_or_truth_inputs": False,
        },
    }, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    (output_root / "checkpoint.sha256").write_text(
        checkpoint_sha + "\n", encoding="utf-8")

    candidate_validation = {
        str(h): _eval_metrics(
            data, validation_refs, candidate, norm_np, config,
            CONTEXT_STEPS["2.0s"], device, h)
        for h in EVAL_HORIZONS}
    base_validation = {
        str(h): _eval_metrics(
            data, validation_refs, base_model, norm_np, config,
            CONTEXT_STEPS["2.0s"], device, h)
        for h in EVAL_HORIZONS}
    candidate_practice = _eval_metrics(
        data, practice_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)
    base_practice = _eval_metrics(
        data, practice_refs, base_model, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)

    high_capture, high_pose = _capture_from_archive(
        HIGHSTEER_SOURCE, HIGHSTEER_RUNS)
    high_refs = _sample_all_valid_context_refs(
        high_capture, 200, history_steps=80)
    high_data = SimpleNamespace(captures=[high_capture], poses=[high_pose])
    candidate_high = _eval_metrics(
        high_data, high_refs, candidate, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)
    base_high = _eval_metrics(
        high_data, high_refs, base_model, norm_np, config,
        CONTEXT_STEPS["2.0s"], device, 200)

    report = {
        "study": "WP28 5 s objective with high-steer regime balanced in final-stage sampling",
        "checkpoint_sha256": checkpoint_sha,
        "warm_start_checkpoint_sha256": base_sha,
        "source_data_sha256": {
            "dynamic": sha256_file(DEFAULT_DYNAMIC),
            "dynamic_fixed40hz": sha256_file(DEFAULT_DYNAMIC_FIXED),
            "practice": sha256_file(DEFAULT_PRACTICE),
            "practice_fixed40hz": sha256_file(DEFAULT_PRACTICE_FIXED),
            "highsteer_holdout": sha256_file(HIGHSTEER_SOURCE),
        },
        "control_and_data_rate_hz": 40.0,
        "dt_s": 0.025,
        "training": {
            "stage": {"horizons_steps": list(STAGE[0][1]),
                      "updates": STAGE[0][2], "batch_size": STAGE[0][3]},
            "eligible_global_runs": sorted(global_pool),
            "eligible_highsteer_runs": sorted(high_pool),
            "highsteer_reference_counts_by_run": {
                run: sum(len(rows) for rows in conditions.values())
                for run, conditions in high_pool.items()},
            "sampling_counts": sampler_counts,
            "optimizer": training,
        },
        "validation_run_ids": sorted(validation_refs),
        "validation_start_count_by_run": {
            run: len(refs) for run, refs in validation_refs.items()},
        "candidate_validation_per_run": candidate_validation,
        "candidate_validation_run_macro": {
            h: _macro(values) for h, values in candidate_validation.items()},
        "warm_start_validation_per_run": base_validation,
        "warm_start_validation_run_macro": {
            h: _macro(values) for h, values in base_validation.items()},
        "candidate_minus_warm_start_validation": {
            h: {metric: _paired(candidate_validation[h], base_validation[h], metric)
                for metric in METRICS}
            for h in candidate_validation},
        "practice_5s_diagnostic": {
            "candidate_per_run": candidate_practice,
            "candidate_run_macro": _macro(candidate_practice),
            "warm_start_per_run": base_practice,
            "warm_start_run_macro": _macro(base_practice),
        },
        "highsteer_5s_transfer": {
            "independent_run_ids": sorted(HIGHSTEER_RUNS),
            "start_count_by_run": {run: len(refs)
                                    for run, refs in high_refs.items()},
            "candidate_per_run": candidate_high,
            "candidate_run_macro": _macro(candidate_high),
            "warm_start_per_run": base_high,
            "warm_start_run_macro": _macro(base_high),
            "candidate_minus_warm_start": {
                metric: _paired(candidate_high, base_high, metric)
                for metric in METRICS},
        },
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_integration": False,
        "interpretation_limit": (
            "High-steer oversampling is based on only three independent training "
            "runs; the two transfer runs are not used for training/checkpoint "
            "selection. Validation selects this fine-tune, so it is development "
            "evidence, not a final unseen-run proof."),
    }
    _write_json(output_root / "highsteer_balanced_candidate_report.json", report)
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
        "training_seconds": report["training"]["optimizer"][
            "elapsed_seconds"],
        "validation_5s_candidate": report[
            "candidate_validation_run_macro"]["200"],
        "highsteer_5s_candidate": report[
            "highsteer_5s_transfer"]["candidate_run_macro"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
