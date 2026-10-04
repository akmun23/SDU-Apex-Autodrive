#!/usr/bin/env python3
"""Score WP28 context candidates on preserved whole-run high-steer captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    CONTEXT_STEPS,
    FixedCapacityHistoryTransition,
    _eval_l3_metrics,
    _eval_metrics,
    _region_macro,
    _region_metrics,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    HistoryTransition,
    _metrics,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    _load_data,
    _predict_wp19_baseline,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.run_history_context_sufficiency import (
    L3_CHECKPOINT,
    OUTPUT_ROOT,
    ROOT,
    TASK_ROOT,
    _normalization,
)
from tools.vehicle_dynamics_learning.evaluate_truncated_history_highsteer_holdout import (
    EXPECTED_RUNS,
    HIGHSTEER_SOURCE,
    _capture_from_archive,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _write_json,
)


DEFAULT_OUTPUT = OUTPUT_ROOT / "highsteer_preserved_whole_run_evaluation_5s_v5.json"
LONG_CURVE_HORIZONS = (1, 2, 4, 8, 10, 20, 40, 80, 120, 160, 200)


def _sample_all_valid_context_refs(capture, horizon_steps: int,
                                  history_steps: int,
                                  expected_runs: set[str] | None = None):
    expected_runs = EXPECTED_RUNS if expected_runs is None else set(expected_runs)
    filtered: dict[str, list[tuple[int, int, int]]] = {
        run_id: [] for run_id in expected_runs}
    for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
            zip(capture.bounds, capture.sequence_run)):
        run_id = str(capture.run_ids[int(run_raw)])
        if (run_id not in expected_runs
                or str(capture.splits[int(run_raw)]) != "validation"):
            continue
        begin, end = int(begin_raw), int(end_raw)
        length = end - begin
        if (length < history_steps + horizon_steps
                or not np.isfinite(capture.input_features[begin:end]).all()
                or not np.isfinite(capture.body[begin:end]).all()
                or not np.isfinite(capture.frames[begin:end]).all()
                or np.any(np.diff(capture.packet[begin:end]) != 1)):
            continue
        filtered[run_id].extend(
            (0, sequence_index, row)
            for row in range(history_steps - 1, length - horizon_steps))
    if not expected_runs or any(not rows for rows in filtered.values()):
        raise RuntimeError("high-steering captures lack requested reset-safe windows")
    return filtered


def _sample_teacher_forced_one_step_refs(capture):
    refs: dict[str, list[tuple[int, int, int]]] = {
        run_id: [] for run_id in EXPECTED_RUNS}
    sequence_counts = {run_id: 0 for run_id in EXPECTED_RUNS}
    for sequence_index, ((begin_raw, end_raw), run_raw) in enumerate(
            zip(capture.bounds, capture.sequence_run)):
        run_id = str(capture.run_ids[int(run_raw)])
        if (run_id not in EXPECTED_RUNS
                or str(capture.splits[int(run_raw)]) != "validation"):
            continue
        begin, end = int(begin_raw), int(end_raw)
        length = end - begin
        if (length <= 80
                or not np.isfinite(capture.input_features[begin:end]).all()
                or not np.isfinite(capture.body[begin:end]).all()
                or not np.isfinite(capture.frames[begin:end]).all()
                or np.any(np.diff(capture.packet[begin:end]) != 1)):
            continue
        rows = np.linspace(79, length - 2,
                           min(64, length - 80), dtype=np.int64)
        refs[run_id].extend((0, sequence_index, int(row)) for row in rows)
        sequence_counts[run_id] += 1
    if set(refs) != EXPECTED_RUNS or any(not values for values in refs.values()):
        raise RuntimeError("high-steering captures lack one-step teacher-forced rows")
    return refs, sequence_counts


def _parent_metrics(data, pose, refs, model, stats, horizon, device):
    result = {}
    for run_id, run_refs in sorted(refs.items()):
        prediction = _predict_wp19_baseline(
            data.captures[0], pose, run_refs, horizon, stats, model, device)
        capture = data.captures[0]
        state_rows, pose_rows = [], []
        for _, sequence_index, source_row in run_refs:
            begin = int(capture.bounds[sequence_index, 0]) + int(source_row)
            state_rows.append(np.stack([
                np.concatenate((capture.body[row], capture.frames[row, 3:5]))
                for row in range(begin + 1, begin + horizon + 1)]))
            pose_rows.append(pose[begin + 1:begin + horizon + 1])
        truth_state = np.asarray(state_rows, dtype=np.float32)
        truth_pose = np.asarray(pose_rows, dtype=np.float32)
        result[run_id] = _metrics(
            np.concatenate((prediction["body"], truth_state[..., 3:5]), axis=-1),
            prediction["pose"], truth_state, truth_pose, horizon,
            include_actuator=False)
    return result


def _macro(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    names = next(iter(metrics.values())).keys()
    return {name: float(np.mean([values[name] for values in metrics.values()]))
            for name in names}


def _paired(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]],
            name: str) -> dict[str, Any]:
    runs = sorted(set(a) & set(b))
    return {"first_minus_second_run_macro": float(np.mean([
                a[run][name] - b[run][name] for run in runs])),
            "per_run_first_minus_second": {
                run: float(a[run][name] - b[run][name]) for run in runs},
            "independent_run_count": len(runs)}


def evaluate(device_name: str = "cuda", output_path: Path = DEFAULT_OUTPUT
             ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(20261028)

    training_data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(training_data, wp20)
    norm_np = _normalization(training_data, config)
    capture, pose = _capture_from_archive(HIGHSTEER_SOURCE, EXPECTED_RUNS)
    horizons = (80, 200)
    history_steps_by_horizon = {80: 160, 200: 80}
    refs_by_horizon = {
        horizon: _sample_all_valid_context_refs(
            capture, horizon, history_steps_by_horizon[horizon])
        for horizon in horizons}
    holdout_data = SimpleNamespace(captures=[capture], poses=[pose])

    source_training_runs = set(training_data.training_runs)
    frozen = json.loads((TASK_ROOT / "next_phase_after_2129427"
                         / "frozen_eval_starts.json").read_text(encoding="utf-8"))
    main_selection_runs = (set(frozen["split_roles"]["development_validation"])
                           | set(frozen["split_roles"]["practice_diagnostic"]))
    if EXPECTED_RUNS & source_training_runs or EXPECTED_RUNS & main_selection_runs:
        raise RuntimeError("high-steering runs overlap training or WP28 selection")

    wp19_saved = torch.load(
        (TASK_ROOT / "wp19_target_ablation_v2_common_encoder_mask"
         / "07_body_state_increment__encoder_angle_increment/model.pt"),
        map_location=device, weights_only=True)
    wp19 = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE).to(device)
    wp19.load_state_dict(wp19_saved["state_dict"], strict=True)
    wp19_stats = training_statistics(
        training_data.captures, training_data.training_windows_80)
    l3_saved = torch.load(L3_CHECKPOINT, map_location=device, weights_only=True)
    l3 = HistoryTransition(norm_np["delta_mean"],
                           norm_np["delta_scale"]).to(device)
    l3.load_state_dict(l3_saved["state_dict"], strict=True)

    reports: dict[str, Any] = {}
    for horizon in horizons:
        refs = refs_by_horizon[horizon]
        reports.setdefault("WP19_parent", {})[str(horizon)] = _parent_metrics(
            holdout_data, pose, refs, wp19, wp19_stats, horizon, device)
        reports.setdefault("L3_5s", {})[str(horizon)] = _eval_l3_metrics(
            holdout_data, refs, l3, norm_np, config, device, horizon)
    long_refs = refs_by_horizon[200]
    for baseline, model, statistics in (
            ("WP19_parent", wp19, wp19_stats),):
        reports.setdefault(baseline, {})["long_curve"] = {
            str(horizon): _parent_metrics(
                holdout_data, pose, long_refs, model, statistics,
                horizon, device)
            for horizon in LONG_CURVE_HORIZONS}
    reports["L3_5s"]["long_curve"] = {
        str(horizon): _eval_l3_metrics(
            holdout_data, long_refs, l3, norm_np, config, device, horizon)
        for horizon in LONG_CURVE_HORIZONS}

    candidates = {}
    candidate_models = {}
    for name, context_steps in CONTEXT_STEPS.items():
        directory = OUTPUT_ROOT / name
        checkpoint_path = directory / "checkpoint.pt"
        expected_sha = (directory / "checkpoint.sha256").read_text(
            encoding="utf-8").strip()
        actual_sha = sha256_file(checkpoint_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"{name}: checkpoint hash does not match sidecar")
        saved = torch.load(checkpoint_path, map_location=device,
                           weights_only=True)
        model = FixedCapacityHistoryTransition(
            norm_np["delta_mean"], norm_np["delta_scale"]).to(device)
        model.load_state_dict(saved["state_dict"], strict=True)
        candidate_models[name] = model
        per_horizon = {}
        region_details = {}
        for horizon in horizons:
            if CONTEXT_STEPS[name] > history_steps_by_horizon[horizon]:
                continue
            refs = refs_by_horizon[horizon]
            per_horizon[str(horizon)] = _eval_metrics(
                holdout_data, refs, model, norm_np, config, context_steps,
                device, horizon)
            region_details[str(horizon)] = _region_metrics(
                holdout_data, refs, model, norm_np, config,
                context_steps, device, horizon_steps=horizon)
        long_curve = {}
        long_regions = {}
        if context_steps <= history_steps_by_horizon[200]:
            for horizon in LONG_CURVE_HORIZONS:
                long_curve[str(horizon)] = _eval_metrics(
                    holdout_data, long_refs, model, norm_np, config,
                    context_steps, device, horizon)
                long_regions[str(horizon)] = _region_metrics(
                    holdout_data, long_refs, model, norm_np, config,
                    context_steps, device, horizon_steps=horizon)
        candidates[name] = {
            "context_steps": context_steps,
            "scored_horizons_steps": sorted(
                int(horizon) for horizon in per_horizon),
            "checkpoint_sha256": actual_sha,
            "per_run_horizons": per_horizon,
            "macro_run_horizons": {
                horizon: _macro(metrics)
                for horizon, metrics in per_horizon.items()},
            "per_run_long_curve": long_curve,
            "macro_run_long_curve": {
                horizon: _macro(metrics)
                for horizon, metrics in long_curve.items()},
            "target_region_macro_horizons": {
                horizon: _region_macro(metrics)
                for horizon, metrics in region_details.items()},
            "target_region_macro_long_curve": {
                horizon: _region_macro(metrics)
                for horizon, metrics in long_regions.items()},
            "target_region_per_run_horizons": region_details,
            "target_region_per_run_long_curve": long_regions,
        }

    teacher_refs, teacher_sequence_counts = _sample_teacher_forced_one_step_refs(
        capture)
    teacher_forced = {
        "start_count_by_run": {run: len(refs)
                               for run, refs in teacher_refs.items()},
        "sequence_count_by_run": teacher_sequence_counts,
        "WP19_parent": _parent_metrics(
            holdout_data, pose, teacher_refs, wp19, wp19_stats, 1, device),
        "L3_5s": _eval_l3_metrics(
            holdout_data, teacher_refs, l3, norm_np, config, device, 1),
        "WP28_2.0s": _eval_metrics(
            holdout_data, teacher_refs, candidate_models["2.0s"],
            norm_np, config, CONTEXT_STEPS["2.0s"], device, 1),
        "WP28_2.0s_target_regions": _region_metrics(
            holdout_data, teacher_refs, candidate_models["2.0s"],
            norm_np, config, CONTEXT_STEPS["2.0s"], device,
            horizon_steps=1),
        "interpretation": (
            "Each sampled transition starts from simulator-truth current body/"
            "actuator state and recorded causal observable history, then predicts "
            "only the next 25 ms. This is a teacher-forced one-step diagnostic, "
            "not a recursive rollout or an online input policy."),
    }

    for name, candidate in candidates.items():
        candidate["paired_vs_baselines"] = {
            baseline: {
                horizon: {metric: _paired(
                    reports[baseline][horizon],
                    candidate["per_run_horizons"][horizon], metric)
                          for metric in (
                              "position_radial_trajectory_rmse_m",
                              "heading_trajectory_rmse_rad",
                              "u_rmse_mps", "v_rmse_mps",
                              "yaw_rate_rmse_rps")}
                for horizon in map(str, horizons)
                if horizon in candidate["per_run_horizons"]}
            for baseline in ("WP19_parent", "L3_5s")}
        candidate["paired_long_curve_vs_baselines"] = {
            baseline: {
                horizon: {metric: _paired(
                    reports[baseline]["long_curve"][horizon],
                    candidate["per_run_long_curve"][horizon], metric)
                          for metric in (
                              "position_radial_trajectory_rmse_m",
                              "heading_trajectory_rmse_rad",
                              "u_rmse_mps", "v_rmse_mps",
                              "yaw_rate_rmse_rps")}
                for horizon in candidate["per_run_long_curve"]}
            for baseline in ("WP19_parent", "L3_5s")}

    report = {
        "study": "WP28 context candidates on preserved 7.5 m/s high-steering simulator runs",
        "source_capture": HIGHSTEER_SOURCE.relative_to(ROOT).as_posix(),
        "source_capture_sha256": sha256_file(HIGHSTEER_SOURCE),
        "whole_run_ids": sorted(EXPECTED_RUNS),
        "training_run_overlap": False,
        "checkpoint_selection_run_overlap": False,
        "all_reset_safe_starts_used": True,
        "valid_start_count_by_run_and_horizon": {
            str(horizon): {run: len(run_refs)
                           for run, run_refs in refs.items()}
            for horizon, refs in refs_by_horizon.items()},
        "history_context_rows_required_by_horizon": {
            str(horizon): history_steps
            for horizon, history_steps in history_steps_by_horizon.items()},
        "control_and_data_rate_hz": 40.0,
        "horizons_steps": list(horizons),
        "long_curve_horizons_steps": list(LONG_CURVE_HORIZONS),
        "long_curve_uses_same_5s_reset_safe_starts": True,
        "history_context_steps": 160,
        "independent_unit": "whole run; start windows are averaged within each run",
        "test_or_final_test_opened": False,
        "simulator_launched": False,
        "production_or_mpc_integration": False,
        "interpretation_limit": (
            "Two independent high-steering runs provide focused transfer evidence, "
            "not a broad run-level generalization guarantee. Five-second scores "
            "use 2-second-or-shorter contexts; the 4-second candidate cannot be "
            "scored at 5 seconds because preserved source sequences are only "
            "7.5 seconds and windows may not cross sequence gaps."),
        "baselines_per_run": reports,
        "candidates": candidates,
        "teacher_forced_one_step_all_segment_rows": teacher_forced,
    }
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = evaluate(args.device, output)
    print(json.dumps({
        "output": output.relative_to(ROOT).as_posix(),
        "valid_start_count_by_run_and_horizon": (
            result["valid_start_count_by_run_and_horizon"]),
        "target_regions_5s": list(
            result["candidates"]["2.0s"]["target_region_macro_horizons"]["200"]),
        "teacher_forced_1step_selected_candidate_run_macro": _macro(
            result["teacher_forced_one_step_all_segment_rows"]["WP28_2.0s"]),
        "selected_2s_context_trajectory_error_by_horizon": {
            horizon: {
                key: result["candidates"]["2.0s"][
                    "macro_run_long_curve"][horizon][key]
                for key in (
                    "position_radial_trajectory_rmse_m",
                    "heading_trajectory_rmse_rad", "u_rmse_mps",
                    "v_rmse_mps", "yaw_rate_rmse_rps")}
            for horizon in result["candidates"]["2.0s"]["macro_run_long_curve"]},
        "candidate_2s_context_5s_metrics": (
            result["candidates"]["2.0s"]["macro_run_horizons"]["200"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
