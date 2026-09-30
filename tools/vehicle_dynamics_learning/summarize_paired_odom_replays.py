#!/usr/bin/env python3
"""Aggregate paired production-odometry replay scores by independent run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_pair(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if baseline["run_id"] != candidate["run_id"]:
        raise ValueError(f"paired run IDs differ: {baseline_path}, {candidate_path}")
    if (baseline["raw_bag"] != candidate["raw_bag"] or
            baseline["unique_source_stamped_samples"] !=
            candidate["unique_source_stamped_samples"]):
        raise ValueError(f"paired scores do not cover the same capture: {baseline['run_id']}")
    for result in (baseline, candidate):
        quality = result["capture_quality"]
        if (quality["aborted"] or quality["timing_faults"]
                or quality["collisions_start_end"] != [0, 0]
                or result["unique_source_stamped_samples"] <= 0):
            raise ValueError(f"quality failure in {result['run_id']}")
    return {"run_id": baseline["run_id"],
            "baseline": baseline, "candidate": candidate}


def _summarize(pairs: list[dict[str, Any]], region: str,
               axis: int, seed: int, bootstrap_samples: int
               ) -> dict[str, Any]:
    base = np.asarray([
        pair["baseline"]["twist_by_regime"][region]["rmse"][axis]
        for pair in pairs], dtype=np.float64)
    cand = np.asarray([
        pair["candidate"]["twist_by_regime"][region]["rmse"][axis]
        for pair in pairs], dtype=np.float64)
    delta = cand - base
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(pairs), size=(bootstrap_samples, len(pairs)))
    boot_base = base[indices].mean(axis=1)
    boot_cand = cand[indices].mean(axis=1)
    boot_delta = boot_cand - boot_base
    ratio = (boot_cand / boot_base) if np.all(boot_base > 0) else np.full_like(boot_base, np.nan)
    return {
        "region": region,
        "axis": ("u_mps", "v_rear_mps", "yaw_rate_radps")[axis],
        "run_count": len(pairs),
        "baseline_macro_run_rmse": float(base.mean()),
        "candidate_macro_run_rmse": float(cand.mean()),
        "macro_run_rmse_delta_candidate_minus_baseline": float(delta.mean()),
        "paired_run_bootstrap_95pct_ci_delta": np.quantile(
            boot_delta, [0.025, 0.975]).tolist(),
        "candidate_to_baseline_macro_rmse_ratio": float(cand.mean() / base.mean()),
        "paired_run_bootstrap_95pct_ci_ratio": np.quantile(
            ratio, [0.025, 0.975]).tolist(),
        "runs_improved": int(np.count_nonzero(delta < 0.0)),
        "run_ids": [pair["run_id"] for pair in pairs],
        "interpretation": "bootstrap resamples whole runs; the runs, not 40 Hz samples, are the independent units",
    }


def summarize(baseline_paths: list[Path], candidate_paths: list[Path],
              output: Path, seed: int = 20260929,
              bootstrap_samples: int = 20000) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite summary: {output}")
    if len(baseline_paths) != len(candidate_paths) or not baseline_paths:
        raise ValueError("provide equally sized, non-empty paired score lists")
    pairs = [_load_pair(base, cand)
             for base, cand in zip(baseline_paths, candidate_paths)]
    if len({pair["run_id"] for pair in pairs}) != len(pairs):
        raise ValueError("each independent run must appear only once")
    regions = ("all_valid_phase_samples", "high_speed_and_steering")
    results = {}
    for region in regions:
        if not all(region in pair["baseline"]["twist_by_regime"] and
                   region in pair["candidate"]["twist_by_regime"]
                   for pair in pairs):
            continue
        results[region] = {
            axis: _summarize(pairs, region, index, seed + index,
                             bootstrap_samples)
            for index, axis in enumerate(("u_mps", "v_rear_mps"))
        }
    report = {
        "comparison": "paired exact production-node replay; per-run RMSE first, then run-level bootstrap",
        "baseline_scores": [str(path.resolve()) for path in baseline_paths],
        "candidate_scores": [str(path.resolve()) for path in candidate_paths],
        "run_ids": [pair["run_id"] for pair in pairs],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    args = parser.parse_args()
    report = summarize(args.baseline, args.candidate, args.output,
                       args.seed, args.bootstrap_samples)
    print(json.dumps({"runs": report["run_ids"],
                      "results": report["results"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
