#!/usr/bin/env python3
"""Paired run-cluster comparison of two saved models on fresh-run reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


STATE_NAMES = ("u_mps", "v_mps", "yaw_rate_rps")
HORIZONS = ("1", "5", "10", "20", "30")
SEED = 20260928


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_map(path: Path) -> dict[str, dict[str, Any]]:
    report = _read(path)
    scores = {}
    for row in report["runs"]:
        run_id = str(row["run_id"])
        ensemble = row.get("ensemble")
        horizons = ensemble.get("horizons", {}) if ensemble else {}
        scores[run_id] = {
            "windows": int(row.get("windows", 0)),
            "scores": {
                horizon: {
                    axis: horizons.get(horizon, {}).get(axis, {}).get(
                        "ensemble_mean_rmse")
                    for axis in STATE_NAMES
                }
                for horizon in HORIZONS
            },
        }
    return scores


def _speed_group(run_id: str) -> str:
    match = re.search(r"holdout_(45|65)_r\d+", run_id)
    return f"{match.group(1)}" if match else "all"


def _paired_stats(rows: list[dict[str, Any]], bootstrap_count: int,
                  rng: np.random.Generator) -> dict[str, Any]:
    available = [row for row in rows if row.get("baseline") is not None
                 and row.get("candidate") is not None and row["baseline"] > 0.0]
    if not available:
        return {"runs": 0, "mean_delta": None, "mean_relative_change_pct": None,
                "relative_change_95pct_run_bootstrap_ci": None}
    deltas = np.asarray([row["candidate"] - row["baseline"] for row in available])
    relative = 100.0 * deltas / np.asarray([row["baseline"] for row in available])
    draws = rng.integers(0, len(available), size=(bootstrap_count, len(available)))
    boot = relative[draws].mean(axis=1)
    return {
        "runs": len(available),
        "mean_delta": float(deltas.mean()),
        "mean_relative_change_pct": float(relative.mean()),
        "relative_change_95pct_run_bootstrap_ci": [
            float(x) for x in np.quantile(boot, (0.025, 0.975))],
        "run_results": {
            row["run_id"]: {
                "windows": row["windows"],
                "baseline": row["baseline"],
                "candidate": row["candidate"],
                "relative_change_pct": 100.0 * (
                    row["candidate"] - row["baseline"]) / row["baseline"],
            }
            for row in available
        },
    }


def _scope(baseline_path: Path, candidate_path: Path,
           bootstrap_count: int, rng: np.random.Generator) -> dict[str, Any]:
    baseline = _score_map(baseline_path)
    candidate = _score_map(candidate_path)
    shared = sorted(set(baseline) & set(candidate))
    missing_baseline = sorted(set(candidate) - set(baseline))
    missing_candidate = sorted(set(baseline) - set(candidate))
    result: dict[str, Any] = {
        "baseline_report": str(baseline_path.resolve()),
        "candidate_report": str(candidate_path.resolve()),
        "shared_run_ids": shared,
        "candidate_runs_missing_from_baseline": missing_baseline,
        "baseline_runs_missing_from_candidate": missing_candidate,
        "metrics": {},
    }
    for horizon in HORIZONS:
        for axis in STATE_NAMES:
            rows = []
            for run_id in shared:
                rows.append({
                    "run_id": run_id,
                    "windows": min(baseline[run_id]["windows"],
                                   candidate[run_id]["windows"]),
                    "baseline": baseline[run_id]["scores"][horizon][axis],
                    "candidate": candidate[run_id]["scores"][horizon][axis],
                })
            key = f"{horizon}_steps__{axis}"
            metric_result = _paired_stats(rows, bootstrap_count, rng)
            groups = {}
            for group in ("45", "65"):
                group_rows = [row for row in rows
                              if _speed_group(row["run_id"]) == group]
                if group_rows:
                    groups[f"{float(group) / 10.0:.1f}_mps"] = _paired_stats(
                        group_rows, bootstrap_count, rng)
            metric_result["by_speed"] = groups
            result["metrics"][key] = metric_result
    return result


def compare(plain_full: Path, candidate_full: Path,
            plain_high: Path, candidate_high: Path,
            output: Path, bootstrap_count: int = 10000,
            plain_label: str = "plain GRU",
            candidate_label: str = "candidate") -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    result = {
        "schema_version": 1,
        "comparison": (
            f"{candidate_label} minus {plain_label}; "
            "negative RMSE change is better"),
        "uncertainty": "paired bootstrap resamples independent run IDs; windows are never treated as independent samples",
        "high_steering_threshold_rad": 0.30,
        "horizon_primary": "30_steps__* (750 ms for 40 Hz data)",
        "full_run": _scope(plain_full, candidate_full, bootstrap_count, rng),
        "high_steering": _scope(plain_high, candidate_high, bootstrap_count, rng),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-full", type=Path, required=True)
    parser.add_argument("--candidate-full", type=Path, required=True)
    parser.add_argument("--plain-high-steering", type=Path, required=True)
    parser.add_argument("--candidate-high-steering", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    parser.add_argument("--plain-label", default="plain GRU")
    parser.add_argument("--candidate-label", default="candidate")
    args = parser.parse_args()
    if args.bootstrap_count < 100:
        parser.error("bootstrap-count must be at least 100")
    try:
        compare(args.plain_full, args.candidate_full,
                args.plain_high_steering, args.candidate_high_steering,
                args.output, args.bootstrap_count,
                args.plain_label, args.candidate_label)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"comparison failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
