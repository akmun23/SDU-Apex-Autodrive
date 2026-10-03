#!/usr/bin/env python3
"""Offline, held-out error diagnosis for the failed WP22 smoke candidate.

This is scoring only: it performs no fitting, checkpoint selection, simulator
launch, or data collection. Results reuse WP22 development validation runs and
are explicitly diagnostic, not independent confirmation.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    load_checkpoint,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    training_statistics,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_OUTPUT as WP22_OUTPUT,
    ROOT,
    SEED,
    _load_data,
    _per_window_metrics,
    _predict_candidate,
    _predict_wp19_baseline,
    sha256_file,
)
from tools.vehicle_dynamics_learning.operating_regions import (
    MISMATCH_EDGES_MPS,
    SPEED_EDGES_MPS,
    STEERING_EDGES_RAD,
    STEERING_RATE_BINS_RADPS,
    THROTTLE_EDGES,
    THROTTLE_SLEW_BINS_PER_S,
)


DEFAULT_REPORT = WP22_OUTPUT / "wp22_failure_diagnosis.json"
HORIZONS = (("250ms", 10), ("750ms", 30), ("2s", 80), ("5s", 200))


def _bin(edges: tuple[float, ...], value: float, prefix: str) -> str:
    index = int(np.searchsorted(edges, value, side="right") - 1)
    return f"{prefix}{min(max(index, 0), len(edges) - 2)}"


def _range_bin(ranges, value: float, prefix: str) -> str:
    for name, low, high in ranges:
        inside = low <= value <= high if name == ranges[-1][0] \
            else low <= value < high
        if inside:
            return name
    return f"{prefix}_outside"


def _start_tags(capture, sequence_index: int, source_row: int
                ) -> tuple[dict[str, str], np.ndarray]:
    begin = int(capture.bounds[sequence_index, 0]) + source_row
    current, previous = capture.frames[begin], capture.frames[begin - 1]
    features = np.asarray((
        np.hypot(current[0], current[1]), current[3],
        (current[3] - previous[3]) / 0.025, current[4],
        (current[4] - previous[4]) / 0.025, current[2],
        abs(float(np.mean(current[5:7]) - current[0])),
    ), dtype=np.float64)
    speed, steering, steer_rate, throttle_fb, throttle_slew, _, mismatch = features
    steer_abs, steer_rate_abs, throttle_cmd = abs(steering), abs(steer_rate), current[8]
    throttle_cmd_slew = (current[8] - previous[8]) / 0.025
    tags = {
        "speed_bin": _bin(SPEED_EDGES_MPS, speed, "S"),
        "steering_bin": _bin(STEERING_EDGES_RAD, steer_abs, "D"),
        "steering_sign": ("negative" if steering < -1e-6 else
                          "positive" if steering > 1e-6 else "near_zero"),
        "steering_rate_bin": _range_bin(STEERING_RATE_BINS_RADPS,
                                         steer_rate_abs, "R"),
        "throttle_feedback_bin": _bin(THROTTLE_EDGES, throttle_fb, "FB"),
        "throttle_command_bin": _bin(THROTTLE_EDGES, throttle_cmd, "C"),
        "throttle_slew_bin": _range_bin(THROTTLE_SLEW_BINS_PER_S,
                                         abs(throttle_cmd_slew), "T"),
        "mismatch_bin": _bin(MISMATCH_EDGES_MPS, mismatch, "M"),
    }
    if speed >= 9.0 and steer_abs < 0.10:
        tags["combined"] = "high_speed_near_straight"
    if speed >= 9.0 and 0.10 <= steer_abs < 0.30:
        tags["combined"] = "high_speed_moderate_steering"
    if 7.0 <= speed < 9.0 and steer_abs >= 0.30:
        tags["combined"] = "7_to_9mps_high_steering"
    if speed < 3.0 and steer_abs >= 0.30:
        tags["combined"] = "low_speed_high_steering"
    if throttle_cmd < -0.05:
        tags["combined"] = "negative_command_braking"
    if throttle_cmd < -0.05 and throttle_cmd_slew >= 0.25:
        tags["combined"] = "braking_release"
    if throttle_cmd > 0.05 and throttle_cmd_slew >= 0.25:
        tags["combined"] = "throttle_pickup"
    prev_abs_steer = abs(previous[3])
    if (steer_abs >= 0.10 and steer_rate_abs >= 0.5
            and steer_abs > prev_abs_steer):
        tags["combined"] = "steering_turn_in"
    if (steer_abs >= 0.10 and steer_rate_abs >= 0.5
            and steer_abs < prev_abs_steer):
        tags["combined"] = "steering_unwind"
    if steer_rate_abs >= 0.5 and abs(throttle_cmd_slew) >= 0.25:
        tags["combined"] = "simultaneous_steering_throttle_transition"
    if mismatch >= 2.0:
        tags["combined"] = "large_wheel_body_mismatch"
    return tags, features


def run_diagnosis(report_path: Path = DEFAULT_REPORT) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite WP22 diagnosis: {report_path}")
    data, _ = _load_data()
    checkpoint_path = WP22_OUTPUT / "seed101_candidate.pt"
    model, checkpoint_extra = load_checkpoint(checkpoint_path, "cpu")
    baseline_checkpoint = torch.load(
        ROOT / "live_runs/derived_dynamics_learning_20260928"
        "/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
        "/full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
        "/07_body_state_increment__encoder_angle_increment/model.pt",
        map_location="cpu", weights_only=True)
    from tools.vehicle_dynamics_learning.run_wp19_target_ablation import make_model
    baseline = make_model(torch.nn)
    baseline.load_state_dict(baseline_checkpoint["state_dict"], strict=True)
    baseline_stats = training_statistics(data.captures, data.training_windows_80)
    region_accumulator = defaultdict(
        lambda: defaultdict(lambda: defaultdict(
            lambda: defaultdict(lambda: {
                "candidate": [], "wp19_parent": []}))))
    support_rows: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"score": [], "body_error": []})
    run_summaries = {}
    all_runs = {"openplane_validation": data.validation_windows,
                "practice_transfer_diagnostic": data.practice_windows}
    for split, windows_by_run in all_runs.items():
        split_runs = {}
        for run_id, refs in sorted(windows_by_run.items()):
            max_horizon = 200
            capture_index = refs[0][0]
            capture = data.captures[capture_index]
            candidate = _predict_candidate(model, data, refs, max_horizon,
                                           torch.device("cpu"))
            parent = _predict_wp19_baseline(
                capture, data.poses[capture_index], refs, max_horizon,
                baseline_stats, baseline, torch.device("cpu"))
            per_run_horizons = {}
            for horizon_name, horizon in HORIZONS:
                candidate_metrics = _per_window_metrics(candidate, horizon)
                parent_metrics = _per_window_metrics(parent, horizon)
                metric_names = tuple(candidate_metrics[0])
                per_run_horizons[horizon_name] = {
                    metric: {
                        "candidate": float(np.mean([row[metric]
                                                    for row in candidate_metrics])),
                        "wp19_parent": float(np.mean([row[metric]
                                                       for row in parent_metrics])),
                    }
                    for metric in metric_names}
                if horizon_name not in ("2s", "5s"):
                    continue
                for index, ref in enumerate(refs):
                    cap_i, seq_i, row = ref
                    tags, _ = _start_tags(
                        data.captures[cap_i], seq_i, row)
                    cmetrics, pmetrics = candidate_metrics[index], parent_metrics[index]
                    # Composite body error is diagnostic only; it is not the
                    # WP20 calibration target or a per-channel confidence bound.
                    body_scale = np.asarray(model.config.state_scale[:3])
                    body_diff = ((candidate["body"][index, :horizon]
                                  - candidate["truth_body"][index, :horizon])
                                 / body_scale)
                    normalized_composite = float(np.sqrt(np.mean(body_diff**2)))
                    if horizon_name == "2s":
                        score = float(np.mean(candidate["support_score"][index, :horizon]))
                        support_rows[run_id]["score"].append(score)
                        support_rows[run_id]["body_error"].append(normalized_composite)
                    for axis, group in tags.items():
                        for metric, value in cmetrics.items():
                            slot = region_accumulator[
                                f"{horizon_name}/{axis}"][group][run_id][metric]
                            slot["candidate"].append(float(value))
                            slot["wp19_parent"].append(float(pmetrics[metric]))
            split_runs[run_id] = {
                "windows": len(refs), "horizons": per_run_horizons}
        run_summaries[split] = {"independent_runs": len(split_runs),
                                "per_run": split_runs}

    # Convert per-window paired records to run-balanced region metrics.
    region_summary: dict[str, Any] = {}
    for axis, groups in sorted(region_accumulator.items()):
        region_summary[axis] = {}
        for group, by_run in sorted(groups.items()):
            region_summary[axis][group] = {
                "independent_runs": len(by_run),
                "evidence": "weak" if len(by_run) < 3 else "run_level",
                "metrics": {
                    metric: {
                        "candidate_macro_run_mean": float(np.mean([
                            np.mean(run_values[metric]["candidate"])
                            for run_values in by_run.values()])),
                        "wp19_parent_macro_run_mean": float(np.mean([
                            np.mean(run_values[metric]["wp19_parent"])
                            for run_values in by_run.values()])),
                        "windows": sum(len(run_values[metric]["candidate"])
                                       for run_values in by_run.values()),
                        "per_run": {
                            run_id: {
                                "candidate": float(np.mean(values[metric]["candidate"])),
                                "wp19_parent": float(np.mean(values[metric]["wp19_parent"])),
                                "windows": len(values[metric]["candidate"]),
                            }
                            for run_id, values in sorted(by_run.items())},
                    }
                    for metric in next(iter(by_run.values()))},
            }

    support_summary = {}
    run_corr = {}
    for run_id, values in sorted(support_rows.items()):
        score = np.asarray(values["score"], dtype=np.float64)
        error = np.asarray(values["body_error"], dtype=np.float64)
        rho = float(spearmanr(score, error).statistic) if len(score) >= 3 else np.nan
        correlation = rho if np.isfinite(rho) else None
        run_corr[run_id] = correlation
        support_summary[run_id] = {
            "scored_windows": len(score),
            "support_score_median": float(np.median(score)) if len(score) else None,
            "normalized_body_error_median": float(np.median(error)) if len(error) else None,
            "support_error_spearman": correlation,
        }
    openplane_corr = [value for run, value in run_corr.items()
                      if run in data.validation_windows and value is not None]
    report = {
        "schema_version": 1,
        "work_package": "WP22 failure diagnosis",
        "purpose": "locate recursive-error onset and operating-regime/support patterns after the fixed candidate failed its material gate",
        "diagnostic_only": True,
        "reuses_wp20_validation_runs": True,
        "not_independent_confirmation": True,
        "training_performed": False,
        "simulator_launched": False,
        "candidate_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "candidate_checkpoint_sha256": sha256_file(checkpoint_path),
        "candidate_seed": checkpoint_extra.get("seed"),
        "baseline_checkpoint_sha256": sha256_file(
            ROOT / "live_runs/derived_dynamics_learning_20260928"
            "/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
            "/full_throttle_domain_v1/wp19_target_ablation_v2_common_encoder_mask"
            "/07_body_state_increment__encoder_angle_increment/model.pt"),
        "horizon_errors_by_run": run_summaries,
        "operating_region_errors": region_summary,
        "support_vs_composite_error_by_run": support_summary,
        "support_correlation_positive_openplane_runs": sum(
            value > 0.0 for value in openplane_corr),
        "support_correlation_openplane_runs": len(openplane_corr),
        "support_correlation_macro_run_mean": (
            float(np.mean(openplane_corr)) if openplane_corr else None),
        "interpretation_limits": [
            "WP20 support was calibrated against WP19, not this candidate.",
            "WP22 validation runs have already been used for development gates.",
            "Regions with fewer than three independent runs are weak evidence.",
            "No regime-level finding from two practice runs is treated as confirmation.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_diagnosis(args.report)
    print(json.dumps({
        "report": str(args.report),
        "candidate_sha256": report["candidate_checkpoint_sha256"],
        "support_positive_runs": report["support_correlation_positive_openplane_runs"],
        "support_run_count": report["support_correlation_openplane_runs"],
        "operating_region_axes": list(report["operating_region_errors"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
