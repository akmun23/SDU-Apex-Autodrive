#!/usr/bin/env python3
"""Rebuild a sealed frozen-QP corpus from archived practice-run states.

This is a saved-state reconstruction, not recovery of live ADMM memory. Each
captured fixture is nevertheless an exact immutable QP/seed snapshot for
solver-only same-QP experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.mpc_qp_fixture import sha256_file, sha256_path

ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY = ROOT / (
    "f1tenth_planning/trajectories/autodrive_practice_20260924_b/"
    "autodrive_practice_20260924_b_mintime_raceline.csv")

# One consistent production-stage profile is used to reconstruct saved-state
# QPs. Replay command plus QP hash are sealed per fixture; this is not claimed
# to recover the exact configuration or ADMM memory of the live callback.
REPLAY_ARGS = [
    "7", "0", "0", "1.6", "0.001", "0", "0", "0", "32", "10",
    "0.02", "sequence", "warm", "8", "100", "0.25", "1", "1",
    "0.10", "0.10", "0.05", "0.05", "7", "2000", "10", "1", "0",
    "1", "4000", "1500", "7.5", "60", "25", "5", "7.8125", "3",
    "5", "0.1", "2000", "10",
]

CASES = [
    {
        "id": "success_priority_worker_tid_5lap",
        "outcome": "success",
        "input": "live_runs/mpc_priority_worker_tid_20261021/solver_inputs.csv",
        "bag": "live_runs/mpc_priority_worker_tid_20261021/run",
        "report": "live_runs/mpc_qp_frozen_corpus_20261021/source_reports/success_priority.json",
        "count": 25,
        "tags": ["near_limit_accepted_lap", "steady_straight", "powered_turn"],
        "observed": "Five practice laps, zero collisions; one successful session.",
    },
    {
        "id": "success_native_slip25_5lap",
        "outcome": "success",
        "input": "live_runs/mpc_native_speed_slip25_throttle10_20261021/solver_inputs_replay.csv",
        "bag": "live_runs/mpc_native_speed_slip25_throttle10_20261021/run",
        "report": "live_runs/mpc_qp_frozen_corpus_20261021/source_reports/success_native.json",
        "count": 25,
        "tags": ["near_limit_accepted_lap", "high_speed_turn", "wheel_slip"],
        "observed": "Five practice laps, zero collisions; uncapped native speed profile.",
    },
    {
        "id": "failure_workers8_terminal_collision",
        "outcome": "failure",
        "input": "live_runs/mpc_priority_workers8_supervised_20261021/terminal_solver_inputs.csv",
        "bag": "live_runs/mpc_priority_workers8_supervised_20261021/run",
        "report": "live_runs/mpc_qp_frozen_corpus_20261021/source_reports/workers8.json",
        "count": 25,
        "tags": ["final_wall_approach", "negative_curvature_reversal", "high_speed_turn"],
        "observed": "Two laps then collision; first nonlinear corridor rejection at stage 2; max prefix chain 6.",
    },
    {
        "id": "failure_recovery_policy_collision",
        "outcome": "failure",
        "input": "live_runs/mpc_recovery_policy_live2_20261021/solver_inputs.csv",
        "bag": "live_runs/mpc_recovery_policy_live2_20261021/run",
        "report": "live_runs/mpc_qp_frozen_corpus_20261021/source_reports/recovery.json",
        "count": 25,
        "tags": ["negative_curvature_reversal", "final_wall_approach"],
        "observed": "No completed laps; one collision; 38-action prefix chain around progress 28.9–31.0 m.",
    },
    {
        "id": "failure_slew500_mpc_rejection",
        "outcome": "failure",
        "input": "live_runs/mpc_slew500_practice_lap1_20261021/solver_inputs.csv",
        "bag": "live_runs/mpc_slew500_practice_lap1_20261021/run",
        "report": "live_runs/mpc_qp_frozen_corpus_20261021/source_reports/slew500.json",
        "count": 25,
        "tags": ["solver_rejection", "normal_straight", "powered_turn"],
        "observed": "No completed lap; an MPC rejection stopped the run; no collision counted in bag.",
    },
]


def data_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as stream:
        first = stream.readline()
        if not first.startswith("#"):
            stream.seek(0)
        return [row for row in csv.reader(stream)
                if row and not row[0].startswith("#")]


def sample_indices(length: int, count: int) -> list[int]:
    if length < 1:
        return []
    count = min(count, length)
    if count == 1:
        return [length - 1]
    indices = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    return sorted(set(indices))


def report_context(report_path: Path) -> dict:
    report = json.loads(report_path.read_text())
    summary = report["summary"]
    detail_path = Path(report["cycle_detail_csv"])
    with detail_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    first_rejection = next(
        (row for row in rows if row.get("status") == "rejected_nonlinear_rollout"),
        None)
    prefix_rows = []
    for row in rows:
        try:
            length = int(row.get("candidate_prefix_chain_length") or 0)
        except ValueError:
            length = 0
        if length:
            prefix_rows.append((length, row))
    maximum_prefix = max((length for length, _ in prefix_rows), default=0)
    maximum_prefix_event = next(
        (row for length, row in prefix_rows if length == maximum_prefix), None)

    def compact_event(row: dict | None) -> dict | None:
        if row is None:
            return None
        keys = (
            "bag_time_from_start_s", "progress_m", "u_mps", "v_mps",
            "r_radps", "status", "solver_status", "solver_stop_reason",
            "iterations", "primal_residual", "dual_residual",
            "full_horizon_failure_stage", "prefix_validation_horizon",
            "candidate_prefix_chain_length", "corridor_slack_m",
            "issued_command_steering_rad", "issued_command_throttle",
            "actual_published_command_steering_rad",
            "actual_published_command_throttle")
        return {key: row.get(key) for key in keys if key in row}

    return {
        "run_manifest_state": "historical_unsealed_after_the_fact",
        "source_report": str(report_path.resolve()),
        "source_report_sha256": sha256_file(report_path),
        "source_report_cycle_detail_csv": str(detail_path.resolve()),
        "source_report_cycle_detail_csv_sha256": sha256_file(detail_path),
        "source_report_missing_run_metadata":
            report.get("metadata", {}).get("run_metadata_missing", []),
        "cutoff_reason": summary.get("score_cutoff_reason"),
        "scored_through_first_fault_from_start_s":
            summary.get("scored_through_first_fault_from_start_s"),
        "first_unpublished_rejection_from_start_s":
            summary.get("first_unpublished_mpc_rejection_from_start_s"),
        "first_collision_from_start_s": summary.get("first_collision_from_start_s"),
        "first_bridge_fault_from_start_s": summary.get("first_bridge_fault_from_start_s"),
        "first_rejection": compact_event(first_rejection),
        "maximum_candidate_prefix_chain": maximum_prefix,
        "maximum_prefix_event": compact_event(maximum_prefix_event),
        "lap_count_max": summary.get("lap_count_max"),
        "recorded_lap_times_s": summary.get("recorded_lap_times_s"),
        "status_counts": summary.get("status_counts"),
        "unity_command_effect_epochs_known":
            summary.get("unity_command_effect_epochs_known"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "live_runs/mpc_qp_frozen_corpus_20261021")
    args = parser.parse_args()
    binary = args.replay_binary.resolve()
    output = args.output.resolve()
    if not binary.is_file() or not TRAJECTORY.is_file():
        parser.error("replay binary or practice trajectory is missing")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "sdu-apex-frozen-qp-regression-corpus-v1",
        "source_regime": "saved_state_reconstruction",
        "solver_experiment_regime": "identical_qp after frozen snapshot",
        "live_solver_memory_restored": False,
        "trajectory": str(TRAJECTORY.resolve()),
        "trajectory_sha256": sha256_file(TRAJECTORY),
        "replay_binary": str(binary),
        "replay_binary_sha256": sha256_file(binary),
        "replay_args": REPLAY_ARGS,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [],
    }
    total_fixtures = 0
    for case in CASES:
        input_csv = ROOT / case["input"]
        bag = ROOT / case["bag"]
        report_path = ROOT / case["report"]
        if not input_csv.is_file() or not bag.is_dir() or not report_path.is_file():
            raise RuntimeError(f"missing corpus input or source bag: {case['id']}")
        case_context = report_context(report_path)
        rows = data_rows(input_csv)
        indices = sample_indices(len(rows), case["count"])
        case_dir = output / case["id"]
        fixture_dir = case_dir / "fixtures"
        case_dir.mkdir(parents=True, exist_ok=True)
        case_context_path = case_dir / "case_context.json"
        case_context_path.write_text(
            json.dumps(case_context, indent=2, sort_keys=True) + "\n")
        command = [sys.executable, str(ROOT / "tools/mpc_qp_fixture.py"),
                   "capture", "--replay-binary", str(binary),
                   "--trajectory", str(TRAJECTORY), "--input-csv",
                   str(input_csv), "--output-dir", str(fixture_dir),
                   "--rows", ",".join(map(str, indices)), "--source-bag",
                   str(bag), "--case-label", case["id"], "--outcome",
                   case["outcome"], "--case-context-json",
                   str(case_context_path), "--root", str(ROOT), "--",
                   *REPLAY_ARGS]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        fixture_files = sorted(fixture_dir.glob("qp_*.fqp"))
        record = {
            "id": case["id"],
            "outcome": case["outcome"],
            "observed_run_result": case["observed"],
            "diagnostic_tags": case["tags"],
            "source_bag": str(bag.resolve()),
            "source_bag_sha256": sha256_path(bag),
            "input_csv": str(input_csv.resolve()),
            "input_csv_sha256": sha256_file(input_csv),
            "source_report": str(report_path.resolve()),
            "source_report_sha256": sha256_file(report_path),
            "source_event_context": case_context,
            "requested_saved_state_rows": indices,
            "captured_fixture_count": len(fixture_files),
            "fixture_payloads": [path.name for path in fixture_files],
            "capture_return_code": completed.returncode,
        }
        manifest["cases"].append(record)
        total_fixtures += len(fixture_files)
        (output / "corpus_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if completed.returncode != 0:
            raise RuntimeError(f"capture failed for {case['id']}")
        print(f"{case['id']}: {len(fixture_files)}/{len(indices)} QPs captured")

    manifest["total_fixtures"] = total_fixtures
    manifest["failure_groups"] = sum(c["outcome"] == "failure"
                                     and c["captured_fixture_count"] > 0
                                     for c in manifest["cases"])
    manifest["success_groups"] = sum(c["outcome"] == "success"
                                     and c["captured_fixture_count"] > 0
                                     for c in manifest["cases"])
    (output / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if total_fixtures < 100 or manifest["failure_groups"] < 3 or \
            manifest["success_groups"] < 2:
        raise RuntimeError("corpus gate failed: require >=100 fixtures, "
                           ">=3 failure groups, and >=2 success groups")
    print(f"corpus gate passed: {total_fixtures} fixtures, "
          f"{manifest['failure_groups']} failure groups, "
          f"{manifest['success_groups']} success groups")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"corpus capture failed: {error}", file=sys.stderr)
        raise SystemExit(2)
