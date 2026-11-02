#!/usr/bin/env python3
"""Strict five-run, ten-scored-lap MPC qualification from Unity/ROS bags.

Requires complete supervisor logs and versioned epoch-aware bag reports.
No extrapolation from optimizer scores or a single clean simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "tools/export_mpc_run_report.py"
RUN_SCHEMA = "sdu_apex_mpc_run_report/v4"

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_run(run: Path, report_dir: Path, trajectory: Path | None,
             scored_laps: int, max_deadline_ms: float) -> dict:
    watcher = run / "watcher.log"
    controller = run / "controller.log"
    bag = run / "run"
    if not watcher.is_file() or not controller.is_file() or not (bag / "metadata.yaml").is_file():
        raise RuntimeError(f"{run}: missing watcher, controller or bag")
    report_dir.mkdir(parents=True, exist_ok=True)
    output_json = report_dir / f"{run.name}.json"
    output_csv = report_dir / f"{run.name}.csv"
    if not output_json.is_file():
        args = [sys.executable, str(EXPORT), str(bag),
                "--output-json", str(output_json), "--output-csv", str(output_csv),
                "--experiment-regime", "live_closed_loop"]
        if trajectory:
            args.extend(("--trajectory", str(trajectory)))
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError(f"{run}: report export failed: {p.stderr[-700:]}")
    report = json.loads(output_json.read_text())
    if report["schema"] != RUN_SCHEMA:
        raise RuntimeError(f"{run}: unexpected report schema")
    summary = report["summary"]
    watcher_text = watcher.read_text(errors="replace")
    controller_text = controller.read_text(errors="replace")
    initial = re.findall(r"^lap_count_start=(\d+)$", watcher_text, re.M)
    milestones = [int(n) for n in re.findall(r"^lap_count=(\d+)$", watcher_text, re.M)]
    declared_collisions = [int(n) for n in re.findall(r"^collision_count=(\d+)$", watcher_text, re.M)]
    complete = "SCREEN_COMPLETE: lap limit" in watcher_text
    rejection = "MPC_CYCLE_REJECTED:" in watcher_text
    fatal = "SCREEN_ABORT:" in watcher_text
    final_stats = re.findall(r"MPC health \(shutdown\): ([^\n]+)", controller_text)
    deadline_misses = None
    if final_stats:
        match = re.search(r"deadline_miss_cycles=(\d+)", final_stats[-1])
        if match:
            deadline_misses = int(match.group(1))
    status = summary["status_counts"]
    accepted_prior = int(summary.get("prior_plan_accepts", 0))
    short_prefix = int(summary.get("short_prefix_accepts", 0))
    viability = int(status.get("accepted_viability_recovery", 0))
    observed = list(map(float, summary.get("recorded_lap_times_s", [])))
    scoring = observed[1:scored_laps + 1]
    max_gate = float(summary.get("gate_ms", {}).get("max") or float("inf"))
    gate_fail = int(summary.get("gate_ms", {}).get("at_or_over_budget") or 0)
    driver_lap_count = summary.get("lap_count_max")
    records = {
        "run": str(run),
        "complete": complete,
        "lap_count_max": driver_lap_count,
        "scored_lap_times_s": scoring,
        "ten_lap_time_s": sum(scoring),
        "mean_scored_lap_s": statistics.mean(scoring) if scoring else None,
        "best_scored_lap_s": min(scoring) if scoring else None,
        "worst_scored_lap_s": max(scoring) if scoring else None,
        "scored_lap_std_s": statistics.pstdev(scoring) if len(scoring) > 1 else None,
        "unpublished_mpc_rejections": rejection,
        "first_unpublished_mpc_rejection_bag_epoch": summary.get(
            "first_unpublished_mpc_rejection_bag_receipt_epoch_s"),
        "collision_bag_epoch": summary.get("first_collision_bag_receipt_epoch_s"),
        "bridge_fault_bag_epoch": summary.get("first_bridge_fault_bag_receipt_epoch_s"),
        "deadline_miss_cycles": deadline_misses,
        "max_gate_ms": max_gate,
        "gate_violation_count": gate_fail,
        "status_counts": status,
        "accepted_retained_plan_actions": accepted_prior,
        "accepted_short_prefix_actions": short_prefix,
        "accepted_viability_recovery_actions": viability,
        "mpc_success_cycles": int(sum(int(v) for k,v in status.items()
                                       if k.startswith("accepted_"))),
        "source_record_sha256": digest(bag / "metadata.yaml"),
        "controller_image_sha256": (run / "controller_image_id.txt").read_text().strip()
            if (run / "controller_image_id.txt").is_file() else None,
        "configuration_sha256": {
            f: digest(run / f) for f in
            ("mpc_base.yaml", "mpc_overlay.yaml", "mpc_parameters.yaml")
            if (run / f).is_file()},
        "trajectory_sha256": digest(run / "raceline.csv")
            if (run / "raceline.csv").is_file() else
            report["metadata"].get("trajectory_sha256"),
    }
    records["pass"] = all((
        complete, not fatal, not rejection,
        initial == ["0"],
        milestones and milestones[-1] >= scored_laps + 1,
        len(scoring) == scored_laps,
        driver_lap_count is not None and driver_lap_count >= scored_laps + 1,
        all(n == 0 for n in declared_collisions),
        records["collision_bag_epoch"] is None,
        records["first_unpublished_mpc_rejection_bag_epoch"] is None,
        records["bridge_fault_bag_epoch"] is None,
        deadline_misses == 0,
        max_gate < max_deadline_ms,
        gate_fail == 0,
        records["mpc_success_cycles"] > 0,
    ))
    return records

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scored-laps", type=int, default=10)
    p.add_argument("--max-gate-ms", type=float, default=20.0)
    p.add_argument("--required-runs", type=int, default=5)
    a = p.parse_args()
    if a.scored_laps < 1 or a.required_runs < 1 or a.max_gate_ms <= 0:
        p.error("invalid qualification thresholds")
    if len(a.runs) != a.required_runs:
        p.error(f"exactly {a.required_runs} runs are required")
    records = [read_run(r.resolve(), a.report_dir, a.trajectory,
                        a.scored_laps, a.max_gate_ms) for r in a.runs]
    # A profile/raceline change halfway through does not constitute a single
    # qualified configuration, even if all individually completed.
    signature = [{
        "controller_image_sha256": r["controller_image_sha256"],
        "configuration_sha256": r["configuration_sha256"],
        "trajectory_sha256": r["trajectory_sha256"],
    } for r in records]
    homogeneous = all(s == signature[0] for s in signature)
    clean = all(r["pass"] for r in records) and homogeneous
    result = {
        "schema": "sdu_apex_mpc_race_qualification/v1",
        "required_runs": a.required_runs, "scored_laps_per_run": a.scored_laps,
        "strict_max_gate_ms": a.max_gate_ms,
        "identical_image_profile_and_trajectory": homogeneous,
        "all_qualification_gates_pass": clean,
        "total_scored_laps": sum(len(r["scored_lap_times_s"]) for r in records),
        "ten_lap_totals_s": [r["ten_lap_time_s"] for r in records],
        "mean_ten_lap_total_s": statistics.mean(r["ten_lap_time_s"] for r in records),
        "max_callback_to_gate_ms": max(r["max_gate_ms"] for r in records),
        "sum_retained_plan_actions": sum(r["accepted_retained_plan_actions"]
                                         for r in records),
        "sum_short_prefix_actions": sum(r["accepted_short_prefix_actions"]
                                        for r in records),
        "sum_viability_recovery_actions": sum(r["accepted_viability_recovery_actions"]
                                              for r in records),
        "runs": records,
        "qualification_limit": (
            "Evidence is for this exact map, car physics, machine/image, "
            "raceline, MPC profile and simulator. Does not establish "
            "high-speed limit handling, global source-model accuracy, "
            "proof of recursive feasibility or numerical QP optimality."),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"},
                     indent=2))
    for r in records:
        print(r["run"], "pass", r["pass"],
              "total", round(r["ten_lap_time_s"],3),
              "max_gate_ms",round(r["max_gate_ms"],3),
              "prior",r["accepted_retained_plan_actions"])
    if not clean:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
