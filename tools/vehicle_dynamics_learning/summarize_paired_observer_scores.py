#!/usr/bin/env python3
"""Run-level paired bootstrap for held-out observer-versus-odometry scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _paired_summary(reports: list[dict[str, Any]], region: str,
                    estimator: str, reference: str, axis: int,
                    seed: int, resamples: int) -> dict[str, Any]:
    candidate = np.asarray([
        report["estimators"][estimator]["regions"][region]["rmse"][axis]
        for report in reports], dtype=np.float64)
    baseline = np.asarray([
        report["estimators"][reference]["regions"][region]["rmse"][axis]
        for report in reports], dtype=np.float64)
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(reports), size=(resamples, len(reports)))
    boot_candidate = candidate[index].mean(axis=1)
    boot_baseline = baseline[index].mean(axis=1)
    boot_delta = boot_candidate - boot_baseline
    boot_ratio = boot_candidate / boot_baseline
    return {
        "observer": estimator,
        "reference": reference,
        "state_axis": ("u_rear_mps", "v_rear_mps")[axis],
        "run_count": len(reports),
        "observer_macro_run_rmse": float(candidate.mean()),
        "reference_macro_run_rmse": float(baseline.mean()),
        "delta_observer_minus_reference": float(delta.mean()),
        "paired_run_bootstrap_95pct_ci_delta": np.quantile(
            boot_delta, [0.025, 0.975]).tolist(),
        "observer_to_reference_rmse_ratio": float(candidate.mean() / baseline.mean()),
        "paired_run_bootstrap_95pct_ci_ratio": np.quantile(
            boot_ratio, [0.025, 0.975]).tolist(),
        "runs_improved": int(np.count_nonzero(delta < 0.0)),
        "run_ids": [report["run_id"] for report in reports],
    }


def _paired_scalar_summary(reports: list[dict[str, Any]],
                           estimator: str, reference: str, metric: str,
                           seed: int, resamples: int) -> dict[str, Any]:
    candidate = np.asarray([
        report["relative_motion"]["estimators"][estimator][metric]
        for report in reports], dtype=np.float64)
    baseline = np.asarray([
        report["relative_motion"]["estimators"][reference][metric]
        for report in reports], dtype=np.float64)
    delta = candidate - baseline
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(reports), size=(resamples, len(reports)))
    boot_candidate = candidate[index].mean(axis=1)
    boot_baseline = baseline[index].mean(axis=1)
    boot_delta = boot_candidate - boot_baseline
    boot_ratio = boot_candidate / boot_baseline
    return {
        "observer": estimator,
        "reference": reference,
        "metric": metric,
        "run_count": len(reports),
        "observer_macro_run_mean_m": float(candidate.mean()),
        "reference_macro_run_mean_m": float(baseline.mean()),
        "delta_observer_minus_reference_m": float(delta.mean()),
        "paired_run_bootstrap_95pct_ci_delta_m": np.quantile(
            boot_delta, [0.025, 0.975]).tolist(),
        "observer_to_reference_ratio": float(candidate.mean() / baseline.mean()),
        "paired_run_bootstrap_95pct_ci_ratio": np.quantile(
            boot_ratio, [0.025, 0.975]).tolist(),
        "runs_improved": int(np.count_nonzero(delta < 0.0)),
        "run_ids": [report["run_id"] for report in reports],
    }


def summarize(score_paths: list[Path], output: Path, seed: int = 20260929,
              resamples: int = 20000) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite summary: {output}")
    if not score_paths:
        raise ValueError("at least one paired observer score is required")
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in score_paths]
    run_ids = [report["run_id"] for report in reports]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("each whole-run capture may appear only once")
    for report in reports:
        quality = report["capture_quality"]
        exact = report["evaluation"]["exact_replay_timestamp_matches"]
        if (quality["aborted"] or quality["timing_faults"]
                or quality["collisions_start_end"] != [0, 0]
                or exact["production"] != report["evaluation"]["scored_samples"]
                or (exact["candidate"] is not None and
                    exact["candidate"] != report["evaluation"]["scored_samples"])):
            raise ValueError(f"invalid or unmatched held-out data in {report['run_id']}")
    estimators = set.intersection(*(
        set(report["estimators"]) for report in reports))
    if "sensor_only_gru" not in estimators:
        raise ValueError("all scores must include the sensor-only GRU")
    references = [name for name in ("production_odom", "candidate_odom")
                  if name in estimators]
    regions = set.intersection(*(
        set(report["estimators"]["sensor_only_gru"]["regions"])
        for report in reports))
    results = {}
    for region in sorted(regions):
        results[region] = {
            f"gru_vs_{reference}": {
                axis_name: _paired_summary(
                    reports, region, "sensor_only_gru", reference, axis,
                    seed + axis, resamples)
                for axis, axis_name in enumerate(("u", "v"))
            }
            for reference in references
        }
    motion_references = [name for name in ("production_odom", "candidate_odom")
                         if all(name in report["relative_motion"]["estimators"]
                                for report in reports)]
    relative_motion = {
        f"gru_vs_{reference}": {
            metric: _paired_scalar_summary(
                reports, "sensor_only_gru", reference, metric,
                seed + 100 + index, resamples)
            for index, metric in enumerate((
                "endpoint_error_rmse_m", "position_error_radial_rmse_m"))
        }
        for reference in motion_references
    }
    summary = {
        "comparison": "frozen sensor-only GRU versus production replay(s); score samples match exactly by source timestamp",
        "uncertainty": "paired bootstrap resamples whole simulator runs; 40 Hz samples are not treated as independent trials",
        "score_files": [str(path.resolve()) for path in score_paths],
        "run_ids": run_ids,
        "resamples": resamples,
        "seed": seed,
        "results": results,
        "relative_motion": relative_motion,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    result = summarize(args.scores, args.output, args.seed, args.resamples)
    compact = {region: {reference: {
        axis: {"observer": stats["observer_macro_run_rmse"],
               "reference": stats["reference_macro_run_rmse"],
               "ci_delta": stats["paired_run_bootstrap_95pct_ci_delta"],
               "wins": stats["runs_improved"], "n": stats["run_count"]}
        for axis, stats in axes.items()}
    for reference, axes in block.items()}
        for region, block in result["results"].items()}
    compact_motion = {reference: {
        metric: {"observer_mean": stats["observer_macro_run_mean_m"],
                 "reference_mean": stats["reference_macro_run_mean_m"],
                 "ci_delta_m": stats["paired_run_bootstrap_95pct_ci_delta_m"],
                 "wins": stats["runs_improved"], "n": stats["run_count"]}
        for metric, stats in metrics.items()}
        for reference, metrics in result["relative_motion"].items()}
    print(json.dumps({"run_ids": result["run_ids"],
                      "results": compact,
                      "relative_motion": compact_motion,
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
