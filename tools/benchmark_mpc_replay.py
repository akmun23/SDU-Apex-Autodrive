#!/usr/bin/env python3
"""Run the production MPC core on a fixed saved-state corpus and report timing.

The replay is offline: it does not include ROS/DDS callback or simulator time.
Pass a separate label for each operating-regime fixture so reports stay
comparable and do not hide hard cases inside an aggregate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ACCEPTED_STATUSES = {0, 1, 2, 8}
TIMING_FIELDS = {
    "cycle_ms": ("cycle_us", 1.0e-3),
    "nominal_build_ms": ("nominal_build_us", 1.0e-3),
    "queue_seed_ms": ("cold_start_queue_seed_us", 1.0e-3),
    "nominal_rollout_a_ms": ("rollout_a_us", 1.0e-3),
    "reference_resampling_ms": ("progress_reference_resampling_us", 1.0e-3),
    "nominal_rollout_b_ms": ("rollout_b_us", 1.0e-3),
    "qp_build_ms": ("qp_build_us", 1.0e-3),
    "jacobian_wall_ms": ("jacobian_wall_us", 1.0e-3),
    "jacobian_thread_cpu_ms": ("jacobian_thread_cpu_us", 1.0e-3),
    "jacobian_slowest_stage_ms": ("jacobian_slowest_stage_us", 1.0e-3),
    "admm_ms": ("admm_us", 1.0e-3),
    "cold_admm_init_ms": ("solver_cold_start_initialization_ns", 1.0e-6),
    "riccati_factorization_ms": ("riccati_factorization_ns", 1.0e-6),
    "riccati_rhs_ms": ("riccati_rhs_ns", 1.0e-6),
    "admm_projection_ms": ("admm_projection_ns", 1.0e-6),
    "rho_update_ms": ("rho_update_ns", 1.0e-6),
    "qp_scale_ms": ("scaling_transform_ns", 1.0e-6),
    "qp_unscale_ms": ("output_unscale_ns", 1.0e-6),
    "candidate_rollout_ms": ("rollout_us", 1.0e-3),
    "nonlinear_validation_ms": ("nonlinear_validation_us", 1.0e-3),
    "line_search_ms": ("line_search_us", 1.0e-3),
}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))]


def distribution(values: list[float]) -> dict[str, float | int] | None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return {
        "count": len(finite),
        "p50": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "max": max(finite),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()

    status = git("status", "--porcelain")
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    return {
        "head": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def cmake_metadata(build_dir: Path) -> dict[str, Any]:
    cache_path = build_dir / "CMakeCache.txt"
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(errors="replace").splitlines():
            match = re.match(r"([^:#]+):[^=]+=([\s\S]*)$", line)
            if match:
                cache[match.group(1)] = match.group(2)

    def compiler_version(key: str) -> str | None:
        compiler = cache.get(key)
        if not compiler:
            return None
        try:
            return subprocess.run(
                [compiler, "--version"], check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            ).stdout.splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError):
            return compiler

    compile_commands_path = build_dir / "compile_commands.json"
    compile_commands: list[dict[str, Any]] = []
    if compile_commands_path.exists():
        try:
            compile_commands = json.loads(compile_commands_path.read_text())
        except json.JSONDecodeError:
            compile_commands = []
    relevant_flags = {}
    for item in compile_commands:
        source = Path(item.get("file", "")).name
        if source in {"mpc_rti.c", "riccati_solver.c", "unity_source_plant.cpp"}:
            relevant_flags[source] = item.get("command") or item.get("arguments")
    return {
        "build_dir": str(build_dir.resolve()),
        "build_type": cache.get("CMAKE_BUILD_TYPE"),
        "c_compiler": cache.get("CMAKE_C_COMPILER"),
        "c_compiler_version": compiler_version("CMAKE_C_COMPILER"),
        "cxx_compiler": cache.get("CMAKE_CXX_COMPILER"),
        "cxx_compiler_version": compiler_version("CMAKE_CXX_COMPILER"),
        "c_flags": cache.get("CMAKE_C_FLAGS"),
        "c_release_flags": cache.get("CMAKE_C_FLAGS_RELEASE"),
        "cxx_flags": cache.get("CMAKE_CXX_FLAGS"),
        "cxx_release_flags": cache.get("CMAKE_CXX_FLAGS_RELEASE"),
        "compile_commands": relevant_flags,
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "missing")
        status_counts[status] = status_counts.get(status, 0) + 1
    admm_termination_counts: dict[str, int] = {}
    admm_iterations: list[float] = []
    line_search_trials = 0
    line_search_accepted = 0
    line_search_skipped_budget = 0
    line_search_skipped_command_epoch = 0
    for row in rows:
        try:
            status = int(row.get("admm_status", "-1"))
            iterations = float(row.get("iterations", "nan"))
            if math.isfinite(iterations):
                admm_iterations.append(iterations)
        except (TypeError, ValueError):
            status = -1
        if status == 0:
            termination = "converged"
        elif status == 1:
            time_limited = row.get("time_budget_exhausted")
            termination = ("wall_time_budget" if time_limited == "1" else
                           "iteration_cap" if time_limited == "0" else
                           "limit_cause_unrecorded")
        elif status == -1:
            termination = "not_run"
        else:
            termination = "solver_error"
        admm_termination_counts[termination] = (
            admm_termination_counts.get(termination, 0) + 1)
        line_search_trials += int(row.get("line_search_trials", 0) or 0)
        line_search_accepted += int(row.get("line_search_accepted", 0) or 0)
        line_search_skipped_budget += int(
            row.get("line_search_skipped_budget", 0) or 0)
        line_search_skipped_command_epoch += int(
            row.get("line_search_skipped_command_epoch", 0) or 0)
    accepted = [row for row in rows
                if int(row.get("status", -1)) in ACCEPTED_STATUSES]
    timings: dict[str, Any] = {}
    for output_name, (column, scale) in TIMING_FIELDS.items():
        values = []
        for row in rows:
            try:
                values.append(float(row[column]) * scale)
            except (KeyError, TypeError, ValueError):
                continue
        timings[output_name] = distribution(values)
    operating = {}
    for name, column in (
        ("forward_speed_mps", "input_u_mps"),
        ("lateral_speed_mps", "input_v_mps"),
        ("yaw_rate_radps", "input_r_radps"),
        ("absolute_steering_rad", "input_abs_steering_rad"),
        ("reference_speed_mps", "reference_speed_mps"),
        ("reference_curvature_per_m", "reference_curvature_per_m"),
        ("minimum_corridor_slack_m", "min_corridor_slack_m"),
    ):
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                continue
        operating[name] = distribution(values)
    sqp_iterations = {
        key: sum(int(row.get(column, 0)) for row in rows)
        for key, column in (
            ("attempted", "sqp_iterations_attempted"),
            ("completed", "sqp_iterations_completed"),
            ("accepted", "sqp_iterations_accepted"),
        )
    }
    worker_utilization = []
    for row in rows:
        try:
            wall_ms = float(row["jacobian_wall_us"]) * 1.0e-3
            cpu_ms = float(row["jacobian_thread_cpu_us"]) * 1.0e-3
            threads = int(row["jacobian_threads_used"])
        except (KeyError, TypeError, ValueError):
            continue
        if wall_ms > 0.0 and threads > 0:
            worker_utilization.append(cpu_ms / (wall_ms * threads))
    jacobian_counts = {}
    for key, column in (
        ("stages_completed", "jacobian_stages_completed"),
        ("source_sensitivity_calls", "jacobian_source_sensitivity_calls"),
        ("finite_difference_columns", "jacobian_finite_difference_columns"),
    ):
        jacobian_counts[key] = sum(int(row.get(column, 0)) for row in rows)
    workers_used: dict[str, int] = {}
    for row in rows:
        value = row.get("jacobian_threads_used")
        if value is not None:
            workers_used[value] = workers_used.get(value, 0) + 1
    cycle_total_us = sum(
        float(row.get("cycle_us", 0.0)) for row in rows
        if math.isfinite(float(row.get("cycle_us", 0.0))))
    return {
        "rows": len(rows),
        "statuses": status_counts,
        "admm_termination_counts": admm_termination_counts,
        "line_search": {
            "trials": line_search_trials,
            "accepted": line_search_accepted,
            "skipped_budget": line_search_skipped_budget,
            "skipped_command_epoch": line_search_skipped_command_epoch,
        },
        "admm_iterations": distribution(admm_iterations),
        "exact_accepted": len(accepted),
        "acceptance_fraction": len(accepted) / len(rows) if rows else None,
        "success_weighted_ms_per_exact_accepted_action":
            cycle_total_us / len(accepted) * 1.0e-3 if accepted else None,
        "sqp_iteration_totals": sqp_iterations,
        "jacobian_work_totals": jacobian_counts,
        "jacobian_threads_used_histogram": workers_used,
        "timing_ms": timings,
        "mean_jacobian_worker_utilization":
            sum(worker_utilization) / len(worker_utilization)
            if worker_utilization else None,
        "operating_point_distributions": operating,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--label", required=True,
                        help="fixture/regime label; keep separate files per regime")
    parser.add_argument("--profile", required=True,
                        help="production or named development configuration")
    parser.add_argument("--ros-workload", default="offline replay; no concurrent ROS load")
    parser.add_argument("--replay-args", nargs=argparse.REMAINDER, default=[],
                        help="remaining positional options passed to mpc_bag_solver_replay")
    args = parser.parse_args()

    if args.raw_csv.exists() or args.output_json.exists():
        parser.error("refusing to overwrite an existing benchmark artifact")
    for path in (args.binary, args.trajectory, args.input_csv):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    root = Path(__file__).resolve().parents[1]
    replay_args = args.replay_args[1:] if args.replay_args[:1] == ["--"] \
        else args.replay_args
    if len(replay_args) > 13:
        try:
            requested_threads = int(replay_args[13])
        except ValueError:
            parser.error("Jacobian worker count must be an integer")
        if requested_threads < 0 or requested_threads > 8:
            parser.error("Jacobian workers must be between 0 and 8")
    replay_mode = replay_args[11] if len(replay_args) > 11 else "sequence"
    solver_start = replay_args[12] if len(replay_args) > 12 else "warm"
    command = [str(args.binary.resolve()), str(args.trajectory.resolve()),
               str(args.input_csv.resolve()), *replay_args]
    load_before = os.getloadavg() if hasattr(os, "getloadavg") else None
    start = time.monotonic()
    try:
        with args.raw_csv.open("x") as output, args.raw_csv.with_suffix(
                args.raw_csv.suffix + ".stderr.log").open("x") as errors:
            process = subprocess.run(command, stdout=output, stderr=errors,
                                     check=False, text=True)
    except OSError as error:
        parser.error(f"could not run replay: {error}")
    elapsed_s = time.monotonic() - start
    if process.returncode != 0:
        print(f"replay failed ({process.returncode}); see "
              f"{args.raw_csv}.stderr.log", file=sys.stderr)
        return process.returncode

    rows = load_rows(args.raw_csv)
    if not rows:
        print("replay produced no rows", file=sys.stderr)
        return 1
    try:
        fixture_rows = load_rows(args.input_csv)
    except (OSError, csv.Error) as error:
        print(f"input fixture could not be read: {error}", file=sys.stderr)
        return 1
    if len(fixture_rows) != len(rows):
        print(f"replay row count {len(rows)} does not match saved-state "
              f"fixture row count {len(fixture_rows)}", file=sys.stderr)
        return 1

    load_after = os.getloadavg() if hasattr(os, "getloadavg") else None
    report = {
        "schema": "sdu-apex-mpc-replay-benchmark-v1",
        "label": args.label,
        "configuration_profile": args.profile,
        "mode": "offline fixed-state solver replay (not live end-to-end timing)",
        "sequence_mode": replay_mode,
        "admm_start": solver_start,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "elapsed_wall_s": elapsed_s,
        "workload_label": args.ros_workload,
        "git": git_metadata(root),
        "machine": {
            "cpu_model": cpu_model(),
            "architecture": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "affinity_cpus": len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else None,
            "load_average_1_5_15": load_before,
            "load_average_after_1_5_15": load_after,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "build": cmake_metadata(args.build_dir),
        "artifacts": {
            "binary": {"path": str(args.binary.resolve()),
                       "sha256": sha256_file(args.binary)},
            "trajectory": {"path": str(args.trajectory.resolve()),
                           "sha256": sha256_file(args.trajectory)},
            "saved_state_fixture": {"path": str(args.input_csv.resolve()),
                                    "sha256": sha256_file(args.input_csv)},
            "replay_csv": str(args.raw_csv.resolve()),
            "stderr_log": str(args.raw_csv.with_suffix(
                args.raw_csv.suffix + ".stderr.log").resolve()),
        },
        "summary": summarize_rows(rows),
        "startup_split": {
            "first_saved_cycle": summarize_rows(rows[:1]),
            "remaining_saved_cycles": summarize_rows(rows[1:]),
            "interpretation": (
                "first-vs-rest replay only; this is not 40 Hz sustained-load "
                "measurement. Use the live bag evaluator for callback timing."
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(f"benchmark metadata: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
