#!/usr/bin/env python3
"""One-screen summary of a live trial: outcome, laps, deferrals, timing.

    source /opt/ros/<distro>/setup.bash
    tools/summarize_trial.py live_runs/RUN_ID [--keep-export DIR]

Wraps tools/export_mpc_run_report.py (which needs the ROS message types) and
condenses its output into the quantities that decide whether a controller
change helped: did it finish, how many cycles published, how long did each
cycle take, and how well did the QP predict the exact source cost.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = {
    "practice": "f1tenth_planning/trajectories/autodrive_practice_20260924_b/"
                "source_command_4p800.csv",
    "competition": "f1tenth_planning/trajectories/autodrive_iros_2026_compete/"
                   "ipopt_best_6p925_source_witness.csv",
}


def percentiles(values, points=(50, 95, 99, 100)):
    if len(values) == 0:
        return "n/a"
    return " ".join(f"P{p}={np.percentile(values, p):.4g}" for p in points)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--keep-export", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = json.loads((run / "run_manifest.json").read_text())
    track = manifest.get("track", {})
    track = track if isinstance(track, str) else track.get("name", "practice")
    trajectory = ROOT / TRAJECTORIES["competition" if "comp" in str(track)
                                     or "iros" in str(track) else "practice"]
    export = args.keep_export or Path(tempfile.mkdtemp(prefix="trial_"))
    export.mkdir(parents=True, exist_ok=True)
    report, cycles, diag = (export / "report.json", export / "cycles.csv",
                            export / "diag.jsonl")
    subprocess.run(
        [sys.executable, str(ROOT / "tools/export_mpc_run_report.py"),
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
    print("status counts", summary["status_counts"])
    gate = summary.get("gate_ms") or {}
    print("publication gate ms", {k: round(v, 2) for k, v in gate.items()
                                  if k in ("p50", "p95", "p99", "max")})

    cycles_seen, timeline = [], ""
    iteration_errors, iteration_times = [], []
    for line in diag.read_text().splitlines():
        payload = json.loads(line)["payload"]
        status = payload.get("status")
        if status in ("waiting_for_localization", "state_stale_wait"):
            continue
        timeline += {"accepted_feasible": ".", "accepted_degraded": "d",
                     "waiting_for_solver_budget": "W"}.get(status, "?")
        cycles_seen.append(payload)
        solver = payload.get("solver") or {}
        for it in solver.get("source_iterations") or []:
            if it.get("full_step_evaluated"):
                error = it.get("full_step_cost_prediction_error")
                if error is not None:
                    iteration_errors.append(abs(error))
            if it.get("total_ms") is not None:
                iteration_times.append(it["total_ms"])
    solve_ms = [p["solve_us"] / 1000. for p in cycles_seen if p.get("solve_us")]
    print("cycle timeline (.=published W=deferred d=degraded ?=other):")
    for start in range(0, len(timeline), 100):
        print("  " + timeline[start:start + 100])
    print("solve ms", percentiles(solve_ms))
    print("iteration ms", percentiles(iteration_times))
    print("|QP predicted - exact| cost", percentiles(iteration_errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
