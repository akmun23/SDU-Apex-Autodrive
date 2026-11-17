#!/usr/bin/env python3
"""Summarize a recorded trial's outcome, laps, deferrals and solver timing.

Usage:
    source /opt/ros/<distro>/setup.bash
    tools/analysis/summarize_trial.py live_runs/RUN_ID [--keep-export DIR]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def percentiles(values, points=(50, 95, 99, 100)):
    if len(values) == 0:
        return "n/a"
    return " ".join(f"P{p}={np.percentile(values, p):.4g}" for p in points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("--keep-export", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text())
    runtime = manifest["runtime"]
    trajectory = ROOT / runtime["trajectory"]["path"]
    export = args.keep_export or Path(tempfile.mkdtemp(prefix="trial_"))
    export.mkdir(parents=True, exist_ok=True)
    report, cycles, diag = (export / "report.json", export / "cycles.csv",
                            export / "diag.jsonl")
    subprocess.run(
        [sys.executable, str(ROOT / "tools/analysis/export_mpc_run_report.py"),
         str(run / "run"), "--trajectory", str(trajectory),
         "--experiment-regime", "live_closed_loop",
         "--output-json", str(report), "--output-csv", str(cycles),
         "--output-diagnostics-jsonl", str(diag)],
        check=True, capture_output=True)
    summary = json.loads(report.read_text())["summary"]
    watcher = (run / "watcher.log").read_text().splitlines()
    outcome = next((line for line in reversed(watcher)
                    if line.startswith(("SCREEN_", "FAULT", "COLLISION"))),
                   watcher[-1] if watcher else "no watcher output")
    print(f"run {run.name}: {outcome}")
    print(f"duration {summary['bag_duration_s']:.1f} s, "
          f"cutoff={summary.get('score_cutoff_reason')}")
    laps = summary.get("recorded_lap_times_s") or []
    print("lap times s (first is warm-up)", [round(t, 4) for t in laps])
    if len(laps) > 1:
        print(f"scored lap mean {np.mean(laps[1:]):.4f} s")
    print("status counts", summary["status_counts"])
    gate = summary.get("gate_ms") or {}
    print("publication gate ms", {k: round(v, 2) for k, v in gate.items()
                                  if k in ("p50", "p95", "p99", "max")})

    cycles_seen, timeline = [], ""
    racing_errors, racing_relative, restoration_errors, iteration_times = [], [], [], []
    racing_steps = fallback_steps = 0
    for line in diag.read_text().splitlines():
        payload = json.loads(line)["payload"]
        status = payload.get("status")
        if status in ("waiting_for_localization", "state_stale_wait"):
            continue
        timeline += {"accepted_feasible": ".", "plan_continuation": "c",
                     "waiting_for_solver_budget": "W"}.get(status, "?")
        cycles_seen.append(payload)
        solver = payload.get("solver") or {}
        for it in solver.get("source_iterations") or []:
            if it.get("full_step_evaluated"):
                # Error is in the iteration's own merit: racing cost while the
                # iterate is feasible, wall violation while it restores.
                error = it.get("full_step_cost_prediction_error")
                if error is not None and it.get("accepted"):
                    if it.get("feasible_before", True):
                        racing_errors.append(abs(error))
                        racing_steps += 1
                        fallback_steps += not it.get("accurate", True)
                        if it.get("cost_before", 0.) > 0.:
                            racing_relative.append(100. * abs(error) / it["cost_before"])
                    else:
                        restoration_errors.append(abs(error))
            if it.get("total_ms") is not None:
                iteration_times.append(it["total_ms"])
    solve_ms = [p["solve_us"] / 1000. for p in cycles_seen if p.get("solve_us")]
    print("cycle timeline (.=published c=plan continuation W=deferred ?=other):")
    for start in range(0, len(timeline), 100):
        print("  " + timeline[start:start + 100])
    print("solve ms", percentiles(solve_ms))
    print("iteration ms", percentiles(iteration_times))
    print("accepted racing steps |predicted - exact| cost", percentiles(racing_errors))
    print("accepted racing steps |predicted - exact| / cost %", percentiles(racing_relative))
    fallback_share = 100. * fallback_steps / racing_steps if racing_steps else 0.
    print(f"accepted racing steps {racing_steps}, exact-decrease fallbacks (outside tolerance) "
          f"{fallback_steps} ({fallback_share:.4g}%)")
    print("restoration iterations |predicted - exact| L1 merit", percentiles(restoration_errors))
    published = [p for p in cycles_seen if p.get("command_published")]
    physical_mm = [1000. * p["solver"]["physical_minimum_corner_clearance_m"]
                   for p in published if (p.get("solver") or {}).get(
                       "physical_minimum_corner_clearance_m") is not None]
    if physical_mm:
        validated_publications = [p for p in published if p.get("solver")]
        reserve_loss = sum((p.get("solver") or {}).get(
            "desired_clearance_reserve_satisfied") is False for p in validated_publications)
        print(f"published plans below desired reserve {reserve_loss}/{len(validated_publications)}")
        print("published full-horizon physical clearance mm",
              percentiles(physical_mm, points=(0, 1, 50, 99)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
