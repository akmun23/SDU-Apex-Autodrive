#!/usr/bin/env python3
"""Export versioned, epoch-aware MPC bag reports for reproducible A/B runs.

Run in the ROS 2 environment used by ``evaluate_mpc_bag.py``. Bridge joins are
exact request-stamp joins only; this tool never infers the Unity physics-step
at which a command was applied from arrival time.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tools import evaluate_mpc_bag as evaluator
except ModuleNotFoundError:
    # Preserve the established `python tools/export_mpc_run_report.py ...`
    # invocation where the script directory, rather than the repo root, is
    # sys.path[0].
    import evaluate_mpc_bag as evaluator


REPORT_SCHEMA = "sdu_apex_mpc_run_report/v4"
COMMAND_FLOAT_MATCH_TOL = 5.0e-8
ACCEPTED_STATUSES = {
    "accepted_optimal", "accepted_feasible", "accepted_line_search",
    "accepted_candidate_prefix", "accepted_prior_plan",
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_metadata(
    root: Path, bag: Path, trajectory: Path | None,
    supplied_metadata: Path | None,
) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain")
    diff = None
    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pass
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            pass

    cmake_cache = root / "build" / "f1tenth_mpc" / "CMakeCache.txt"
    build_values: dict[str, str] = {}
    if cmake_cache.is_file():
        for line in cmake_cache.read_text(errors="replace").splitlines():
            if ":" not in line or "=" not in line or line.startswith("//"):
                continue
            key, value = line.split("=", 1)
            if key.startswith((
                    "CMAKE_BUILD_TYPE:", "CMAKE_C_COMPILER:",
                    "CMAKE_CXX_COMPILER:", "CMAKE_C_FLAGS:",
                    "CMAKE_CXX_FLAGS:", "SDU_APEX_MPC_NATIVE_TUNING:",
                    "OpenMP_C_FLAGS:")):
                build_values[key.split(":", 1)[0]] = value

    supplied: dict[str, Any] = {}
    if supplied_metadata is not None:
        supplied = json.loads(supplied_metadata.read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("--run-metadata must contain a JSON object")

    cpu_model = platform.processor() or None
    if cpu_model is None:
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    return {
        "report_generation_environment": {
        "head": _git(root, "rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "worktree_status": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()
        if diff is not None else None,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity": affinity,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "build_cache": str(cmake_cache) if cmake_cache.is_file() else None,
        "build_values": build_values,
        },
        "trajectory": (str(trajectory) if trajectory else None),
        "trajectory_sha256": sha256_file(trajectory) if trajectory else None,
        "bag": str(bag),
        "bag_metadata_sha256": sha256_file(bag / "metadata.yaml"),
        "run_metadata_missing": [
            key for key in (
                "commit_sha", "compiler", "build_flags", "cpu_model",
                "cpu_topology", "ros_executor_threads", "openmp_settings",
                "container_limits", "unity_mode", "trajectory_sha256",
                "loaded_yaml_hashes", "initial_pose", "map_hash",
                "supervisor_version") if key not in supplied
        ],
        "operator_supplied_run_metadata": supplied,
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _finite_list(values: Any) -> list[float | None]:
    if not isinstance(values, list):
        return []
    return [_finite_number(value) for value in values]


def _normalized_solver_category(reason: Any) -> str:
    """Keep solver termination causes distinct for handoff-level counts."""
    if not isinstance(reason, str):
        return "unknown"
    if reason == "converged":
        return "optimal"
    if reason == "iteration_cap":
        return "iteration_cap"
    if reason in {"time_budget", "wall_time_budget"}:
        return "admm_time_budget"
    if reason == "deadline_before_admm":
        return "qp_build_deadline_skip"
    if reason == "numerical_or_solver_error":
        return "numerical_or_solver_error"
    if reason == "not_run":
        return "not_run"
    return "unknown"


def _first_collision_time(rows: list[dict[str, Any]]) -> float | None:
    ordered = sorted(rows, key=lambda row: float(row.get("bag_time", 0.0)))
    if not ordered:
        return None
    initial = _finite_number(ordered[0].get("value"))
    if initial is None:
        return None
    for row in ordered[1:]:
        value = _finite_number(row.get("value"))
        if value is not None and value > initial:
            return float(row["bag_time"])
    return None


def _first_bridge_fault_time(rows: list[dict[str, Any]]) -> float | None:
    for row in sorted(rows, key=lambda item: float(item.get("bag_time", 0.0))):
        value = row.get("value")
        if isinstance(value, str) and value:
            return float(row["bag_time"])
    return None


def _first_unpublished_rejection_time(
    rows: list[dict[str, Any]],
) -> float | None:
    for row in sorted(rows, key=lambda item: float(item.get("bag_time", 0.0))):
        payload = row.get("json")
        status = payload.get("status") if isinstance(payload, dict) else None
        if (isinstance(status, str) and status.startswith("rejected") and
                payload.get("command_published") is False):
            return float(row["bag_time"])
    return None


def _distinct_lap_times(
    rows: list[dict[str, Any]], cutoff_s: float | None = None,
    lap_count_rows: list[dict[str, Any]] | None = None,
) -> list[float]:
    """Return lap-completion events, not one repeated topic sample per tick."""
    if lap_count_rows:
        # Consecutive laps can have exactly equal quantized durations. A value
        # change is not a completion event: use the independent lap counter.
        # These headerless topics are separate DDS publications. Select the
        # first last-lap sample at/after the counter change (at most 3 ticks),
        # never an old duration from the preceding lap or after a fault cutoff.
        samples = sorted((float(row["bag_time"]), value)
                         for row in rows
                         if (value := _finite_number(row.get("value"))) is not None
                         and value > 0.0 and "bag_time" in row
                         and (cutoff_s is None or row["bag_time"] <= cutoff_s))
        times = [sample[0] for sample in samples]
        result = []
        previous_count = None
        for row in sorted(lap_count_rows,
                          key=lambda item: float(item.get("bag_time", 0.0))):
            stamp = _finite_number(row.get("bag_time"))
            count = _finite_number(row.get("value"))
            if stamp is None or count is None or count < 0:
                continue
            if cutoff_s is not None and stamp > cutoff_s:
                break
            if previous_count is not None and count > previous_count:
                index = bisect.bisect_left(times, stamp)
                if index < len(samples) and times[index] - stamp <= 0.075:
                    result.append(samples[index][1])
            previous_count = count
        return result

    result: list[float] = []
    previous: float | None = None
    for row in sorted(rows, key=lambda item: float(item.get("bag_time", 0.0))):
        bag_time = _finite_number(row.get("bag_time"))
        if cutoff_s is not None and bag_time is not None and bag_time > cutoff_s:
            continue
        value = _finite_number(row.get("value"))
        if value is None or value <= 0.0:
            continue
        if previous is None or abs(value - previous) > 1.0e-4:
            result.append(value)
        previous = value
    return result


def _bridge_by_request_stamp(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        payload = row.get("json")
        if not isinstance(payload, dict):
            continue
        request_stamp = payload.get("bridge_request_ros_stamp_ns")
        sequence = payload.get("request_sequence")
        if request_stamp is None or sequence is None:
            continue
        try:
            indexed[int(request_stamp)] = payload
        except (TypeError, ValueError):
            continue
    return indexed


def _drive_command_by_stamp(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        exact_stamp = row.get("stamp_ns")
        if exact_stamp is not None:
            try:
                indexed[int(exact_stamp)] = row
                continue
            except (TypeError, ValueError):
                pass
        stamp = _finite_number(row.get("stamp"))
        if stamp is not None:
            indexed[round(stamp * 1.0e9)] = row
    return indexed


def _cycle_row(
    bag_receipt_epoch_s: float, bag_time_origin_epoch_s: float,
    payload: dict[str, Any],
    bridge_by_stamp: dict[int, dict[str, Any]],
    drive_by_stamp: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    prediction = payload.get("control_time_prediction")
    prediction = prediction if isinstance(prediction, dict) else {}
    solver = payload.get("solver")
    solver = solver if isinstance(solver, dict) else {}
    worker_cpu = _finite_list(solver.get("jacobian_worker_cpu_us"))
    worker_wall = _finite_list(solver.get("jacobian_worker_stage_wall_us"))
    worker_counts = solver.get("jacobian_worker_stage_count")
    worker_tids = solver.get("jacobian_worker_os_tid")
    if not isinstance(worker_counts, list):
        worker_counts = []
    if not isinstance(worker_tids, list):
        worker_tids = []
    active_worker_indices = [
        index for index in range(max(len(worker_cpu), len(worker_wall),
                                     len(worker_counts)))
        if index < len(worker_counts) and worker_counts[index] > 0]
    slowest_worker_cpu_index = max(
        active_worker_indices,
        key=lambda index: worker_cpu[index]
        if index < len(worker_cpu) and worker_cpu[index] is not None else -1.0,
        default=None)
    slowest_worker_wall_index = max(
        active_worker_indices,
        key=lambda index: worker_wall[index]
        if index < len(worker_wall) and worker_wall[index] is not None else -1.0,
        default=None)
    state = payload.get("state")
    state = state if isinstance(state, list) else []
    action = payload.get("first_action")
    action = action if isinstance(action, list) else []
    nominal_action = payload.get("nominal_first_action")
    nominal_action = nominal_action if isinstance(nominal_action, list) else []
    residual_hotspots = solver.get("residual_hotspots")
    residual_hotspots = residual_hotspots if isinstance(residual_hotspots, dict) else {}
    candidate_controls = payload.get("candidate_controls")
    if not isinstance(candidate_controls, list):
        candidate_controls = payload.get("rejected_candidate_controls")
    nominal_controls = payload.get("nominal_controls")
    control_delta_inf = None
    if (isinstance(candidate_controls, list) and
            isinstance(nominal_controls, list) and candidate_controls and
            len(candidate_controls) == len(nominal_controls)):
        deltas = []
        for candidate, nominal in zip(candidate_controls, nominal_controls):
            if (isinstance(candidate, list) and len(candidate) >= 2 and
                    isinstance(nominal, list) and len(nominal) >= 2):
                values = [_finite_number(candidate[i]) for i in range(2)]
                refs = [_finite_number(nominal[i]) for i in range(2)]
                if all(value is not None for value in values + refs):
                    deltas.extend(abs(values[i] - refs[i]) for i in range(2))
        if deltas:
            control_delta_inf = max(deltas)

    source_epoch = prediction.get("plant_source_stamp_ns")
    if source_epoch is None:
        source_epoch = payload.get("source_stamp_ns")
    try:
        source_epoch_ns = int(source_epoch) if source_epoch is not None else None
    except (TypeError, ValueError):
        source_epoch_ns = None
    bridge = bridge_by_stamp.get(source_epoch_ns, {}) if source_epoch_ns else {}
    bridge_receive_stamp = _finite_number(
        bridge.get("bridge_receive_ros_stamp_ns"))
    bridge_request_to_receive_ms = (
        (bridge_receive_stamp - source_epoch_ns) * 1.0e-6
        if bridge_receive_stamp is not None and source_epoch_ns is not None
        else None)
    command_published = payload.get("command_published") is True
    command_stamp = payload.get("candidate_command_stamp_ns")
    try:
        scheduled_command_stamp_ns = (
            int(command_stamp) if command_stamp is not None else None)
    except (TypeError, ValueError):
        scheduled_command_stamp_ns = None
    issued_stamp = payload.get("issued_command_stamp_ns")
    try:
        issued_command_stamp_ns = int(issued_stamp) if issued_stamp is not None else None
    except (TypeError, ValueError):
        issued_command_stamp_ns = None
    scheduled_bridge = (bridge_by_stamp.get(scheduled_command_stamp_ns, {})
                        if scheduled_command_stamp_ns is not None else {})
    drive = drive_by_stamp.get(issued_command_stamp_ns, {}) \
        if issued_command_stamp_ns is not None else {}

    try:
        published_steering, published_throttle = action[:2]
    except (TypeError, ValueError):
        published_steering = published_throttle = None
    try:
        nominal_steering, nominal_throttle = nominal_action[:2]
    except (TypeError, ValueError):
        nominal_steering = nominal_throttle = None

    budget = _finite_number(payload.get("end_to_end_cycle_budget_ms"))
    gate = _finite_number(payload.get("publication_gate_elapsed_ms"))
    gate_violation = (gate >= budget) if gate is not None and budget is not None else None
    previous_diag = payload.get("previous_diagnostics_timing")
    previous_diag = previous_diag if isinstance(previous_diag, dict) else {}
    async_publisher = payload.get("diagnostic_async_publisher")
    async_publisher = async_publisher if isinstance(async_publisher, dict) else {}
    candidate_prefix = payload.get("candidate_prefix_revalidation")
    candidate_prefix = candidate_prefix if isinstance(candidate_prefix, dict) else {}
    prior_plan = payload.get("prior_plan_revalidation")
    prior_plan = prior_plan if isinstance(prior_plan, dict) else {}
    full_prior_suffix = payload.get("full_prior_plan_suffix_revalidation")
    full_prior_suffix = (full_prior_suffix
                         if isinstance(full_prior_suffix, dict) else {})
    quality_prefix = payload.get("quality_aware_prefix_selection")
    quality_prefix = quality_prefix if isinstance(quality_prefix, dict) else {}
    quality_candidate_controls = quality_prefix.get(
        "candidate_prefix_controls")
    quality_prior_controls = quality_prefix.get("prior_prefix_controls")
    return {
        "bag_receipt_epoch_s": bag_receipt_epoch_s,
        "bag_time_from_start_s": bag_receipt_epoch_s - bag_time_origin_epoch_s,
        "bridge_request_epoch_ns": source_epoch_ns,
        "bridge_request_sequence": bridge.get("request_sequence"),
        "bridge_received_epoch_ns": bridge.get("bridge_receive_ros_stamp_ns"),
        "bridge_request_to_receive_ms": bridge_request_to_receive_ms,
        "scheduled_command_stamp_ns": scheduled_command_stamp_ns,
        "issued_command_stamp_ns": issued_command_stamp_ns,
        "scheduled_bridge_request_sequence": scheduled_bridge.get("request_sequence"),
        "actual_published_command_steering_rad": _finite_number(drive.get("steering")),
        "actual_published_command_throttle": _finite_number(drive.get("throttle")),
        "published_steering_message_matches_candidate": (
            abs(float(drive["steering"]) - float(published_steering)) <
            COMMAND_FLOAT_MATCH_TOL
            if command_published and drive.get("steering") is not None and
            _finite_number(published_steering) is not None else None),
        "published_throttle_message_matches_candidate": (
            abs(float(drive["throttle"]) - float(published_throttle)) <
            COMMAND_FLOAT_MATCH_TOL
            if command_published and drive.get("throttle") is not None and
            _finite_number(published_throttle) is not None else None),
        "actual_unity_effect_epoch": None,
        "status": payload.get("status"),
        "cycle_reason": payload.get("reason"),
        "input_failure_reason": payload.get("input_failure_reason"),
        "nominal_build_failure_stage": payload.get(
            "nominal_build_failure_stage"),
        "nominal_build_failure_constraint": payload.get(
            "nominal_build_failure_constraint"),
        "nominal_build_failure_index": payload.get(
            "nominal_build_failure_index"),
        "nonlinear_failure_reason": payload.get("nonlinear_failure_reason"),
        "nonlinear_failure_stage": payload.get("nonlinear_failure_stage"),
        "nonlinear_failure_detail": payload.get("nonlinear_failure"),
        "action_source": payload.get("action_source"),
        "command_published": payload.get("command_published"),
        "solver_skipped_for_deadline": payload.get(
            "solver_skipped_for_deadline"),
        "progress_m": _finite_number(payload.get("progress_m")),
        "e_y_m": _finite_number(state[0]) if len(state) > 0 else None,
        "e_psi_rad": _finite_number(state[1]) if len(state) > 1 else None,
        "u_mps": _finite_number(state[2]) if len(state) > 2 else None,
        "v_mps": _finite_number(state[3]) if len(state) > 3 else None,
        "r_radps": _finite_number(state[4]) if len(state) > 4 else None,
        "actual_steering_rad": _finite_number(state[7]) if len(state) > 7 else None,
        "steering_limit_rad": _finite_number(payload.get("steering_limit_rad")),
        "target_u_mps": _finite_number(payload.get("reference", {}).get(
            "speed_reference_mps")) if isinstance(payload.get("reference"), dict) else None,
        "curvature_inv_m": _finite_number(payload.get("path_curvature_per_m")),
        "candidate_first_steering_rad": _finite_number(published_steering),
        "candidate_first_throttle": _finite_number(published_throttle),
        "issued_command_steering_rad": (
            _finite_number(published_steering) if command_published else None),
        "issued_command_throttle": (
            _finite_number(published_throttle) if command_published else None),
        "nominal_steering_rad": _finite_number(nominal_steering),
        "nominal_throttle": _finite_number(nominal_throttle),
        "steering_command_delta_rad": (
            _finite_number(published_steering) - _finite_number(nominal_steering)
            if _finite_number(published_steering) is not None and
            _finite_number(nominal_steering) is not None else None),
        "throttle_command_delta": (
            _finite_number(published_throttle) - _finite_number(nominal_throttle)
            if _finite_number(published_throttle) is not None and
            _finite_number(nominal_throttle) is not None else None),
        "candidate_prefix_chain_length": payload.get(
            "consecutive_candidate_prefix_actions"),
        "prior_plan_chain_length": payload.get("consecutive_prior_plan_actions"),
        "solver_status": solver.get("admm_status"),
        "solver_stop_reason": (
            solver.get("admm_stop_reason") or
            ("not_run" if payload.get("status") == "waiting_for_localization" else
             "wall_time_budget" if solver.get("time_budget_exhausted") is True else
            "iteration_cap" if solver.get("admm_status") == 1 and
                solver.get("time_budget_exhausted") is False else
            "converged" if solver.get("admm_status") == 0 else
            "not_run_or_unknown")),
        "iterations": solver.get("iterations"),
        "sqp_iterations_attempted": solver.get("sqp_iterations_attempted"),
        "sqp_iterations_completed": solver.get("sqp_iterations_completed"),
        "sqp_iterations_accepted": solver.get("sqp_iterations_accepted"),
        "line_search_selected_alpha": _finite_number(solver.get("line_search_selected_alpha")),
        "line_search_merit_triggered": solver.get("line_search_merit_triggered"),
        "refinement_prefix_selected_pass": solver.get("refinement_prefix_selected_pass"),
        "refinement_first_prefix_objective": _finite_number(
            solver.get("refinement_first_prefix_objective")),
        "refinement_second_prefix_objective": _finite_number(
            solver.get("refinement_second_prefix_objective")),
        "refinement_prefix_validation_us": _finite_number(
            solver.get("refinement_prefix_validation_us")),
        "primal_residual": _finite_number(solver.get("primal_residual")),
        "dual_residual": _finite_number(solver.get("dual_residual")),
        "state_primal_residual": _finite_number(
            residual_hotspots.get("state_primal", {}).get("value")
            if isinstance(residual_hotspots.get("state_primal"), dict) else None),
        "state_primal_stage": residual_hotspots.get(
            "state_primal", {}).get("stage")
            if isinstance(residual_hotspots.get("state_primal"), dict) else None,
        "state_primal_coordinate": residual_hotspots.get(
            "state_primal", {}).get("coordinate")
            if isinstance(residual_hotspots.get("state_primal"), dict) else None,
        "state_dual_residual": _finite_number(
            residual_hotspots.get("state_dual", {}).get("value")
            if isinstance(residual_hotspots.get("state_dual"), dict) else None),
        "state_dual_stage": residual_hotspots.get(
            "state_dual", {}).get("stage")
            if isinstance(residual_hotspots.get("state_dual"), dict) else None,
        "state_dual_coordinate": residual_hotspots.get(
            "state_dual", {}).get("coordinate")
            if isinstance(residual_hotspots.get("state_dual"), dict) else None,
        "input_primal_residual": _finite_number(
            residual_hotspots.get("input_primal", {}).get("value")
            if isinstance(residual_hotspots.get("input_primal"), dict) else None),
        "input_primal_stage": residual_hotspots.get(
            "input_primal", {}).get("stage")
            if isinstance(residual_hotspots.get("input_primal"), dict) else None,
        "input_primal_coordinate": residual_hotspots.get(
            "input_primal", {}).get("coordinate")
            if isinstance(residual_hotspots.get("input_primal"), dict) else None,
        "input_dual_residual": _finite_number(
            residual_hotspots.get("input_dual", {}).get("value")
            if isinstance(residual_hotspots.get("input_dual"), dict) else None),
        "input_dual_stage": residual_hotspots.get(
            "input_dual", {}).get("stage")
            if isinstance(residual_hotspots.get("input_dual"), dict) else None,
        "input_dual_coordinate": residual_hotspots.get(
            "input_dual", {}).get("coordinate")
            if isinstance(residual_hotspots.get("input_dual"), dict) else None,
        "objective": _finite_number(solver.get("objective")),
        "rho_change_count": solver.get("rho_change_count"),
        "rho_start": _finite_number(solver.get("rho_start")),
        "rho_final": _finite_number(solver.get("rho_final")),
        "rho_u_start": _finite_number(solver.get("rho_u_start")),
        "rho_u_final": _finite_number(solver.get("rho_u_final")),
        "factorization_count": solver.get("factorization_count"),
        "regularization_count": solver.get("regularization_count"),
        "max_regularization": _finite_number(solver.get("max_regularization")),
        "nominal_candidate_control_delta_inf": control_delta_inf,
        "nominal_vs_candidate_curvature_error_max_per_m": _finite_number(
            payload.get("nominal_vs_candidate_curvature_error_max_per_m")),
        "nominal_vs_candidate_progress_error_max_m": _finite_number(
            payload.get("nominal_vs_candidate_progress_error_max_m")),
        "nominal_vs_candidate_left_bound_error_max_m": _finite_number(
            payload.get("nominal_vs_candidate_left_bound_error_max_m")),
        "nominal_vs_candidate_right_bound_error_max_m": _finite_number(
            payload.get("nominal_vs_candidate_right_bound_error_max_m")),
        "full_horizon_failure_stage": payload.get("nonlinear_failure_stage"),
        "candidate_prefix_validation_attempted": candidate_prefix.get(
            "attempted"),
        "candidate_prefix_exact_valid": candidate_prefix.get("exact_valid"),
        "candidate_prefix_failure_stage": candidate_prefix.get(
            "failure_stage"),
        "candidate_prefix_validation_status": candidate_prefix.get(
            "validation_status"),
        "candidate_prefix_validation_horizon": candidate_prefix.get(
            "validation_horizon"),
        "candidate_prefix_timing_rejected": candidate_prefix.get(
            "timing_rejected"),
        "candidate_prefix_solver_converged": candidate_prefix.get(
            "solver_converged"),
        "candidate_prefix_solver_iteration_limited": candidate_prefix.get(
            "solver_iteration_limited"),
        "prior_plan_validation_attempted": prior_plan.get("attempted"),
        "prior_plan_exact_valid": prior_plan.get("exact_valid"),
        "prior_plan_failure_stage": prior_plan.get("failure_stage"),
        "prior_plan_validation_status": prior_plan.get("validation_status"),
        "prior_plan_validation_horizon": prior_plan.get(
            "validation_horizon"),
        "prior_plan_timing_rejected": prior_plan.get("timing_rejected"),
        "full_prior_suffix_attempted": full_prior_suffix.get("attempted"),
        "full_prior_suffix_exact_valid": full_prior_suffix.get("exact_valid"),
        "full_prior_suffix_failure_stage": full_prior_suffix.get(
            "failure_stage"),
        "full_prior_suffix_validation_status": full_prior_suffix.get(
            "validation_status"),
        "full_prior_suffix_validation_horizon": full_prior_suffix.get(
            "validation_horizon"),
        "full_prior_suffix_timing_rejected": full_prior_suffix.get(
            "timing_rejected"),
        "full_prior_suffix_validation_us": _finite_number(
            full_prior_suffix.get("validation_us")),
        "quality_prefix_policy_enabled": payload.get(
            "quality_aware_prefix_selection_on_rejection"),
        "quality_prefix_attempted": quality_prefix.get("attempted"),
        "quality_prefix_choice": quality_prefix.get("choice"),
        "quality_candidate_prefix_exact_valid": quality_prefix.get(
            "candidate_valid"),
        "quality_prior_prefix_exact_valid": quality_prefix.get("prior_valid"),
        "quality_candidate_prefix_objective": _finite_number(
            quality_prefix.get("candidate_objective")),
        "quality_prior_prefix_objective": _finite_number(
            quality_prefix.get("prior_objective")),
        "quality_candidate_prefix_min_slack_m": _finite_number(
            quality_prefix.get("candidate_minimum_corridor_slack_m")),
        "quality_prior_prefix_min_slack_m": _finite_number(
            quality_prefix.get("prior_minimum_corridor_slack_m")),
        "quality_prefix_validation_us": _finite_number(
            quality_prefix.get("validation_us")),
        "quality_candidate_prefix_controls_json": (
            json.dumps(quality_candidate_controls, separators=(",", ":"))
            if isinstance(quality_candidate_controls, list) else None),
        "quality_prior_prefix_controls_json": (
            json.dumps(quality_prior_controls, separators=(",", ":"))
            if isinstance(quality_prior_controls, list) else None),
        "prefix_validation_horizon": candidate_prefix.get(
            "validation_horizon"),
        "corridor_slack_m": _finite_number(
            solver.get("minimum_predicted_corridor_slack_m",
                       payload.get("minimum_predicted_corridor_slack_m"))),
        "source_age_s": _finite_number(payload.get("source_age_s")),
        "callback_to_gate_ms": gate,
        "callback_to_publish_return_ms": _finite_number(
            payload.get("end_to_end_control_ms")),
        "middleware_publish_us": _finite_number(payload.get("middleware_publish_us")),
        "command_history_us": _finite_number(payload.get("command_history_us")),
        "budget_ms": budget,
        "publication_gate_budget_violation": gate_violation,
        "prior_plan_published_while_over_budget": (
            bool(payload.get("action_source") == "revalidated_prior_plan" and
                 gate_violation is True) if gate_violation is not None else None),
        "command_start_lateness_ms": _finite_number(
            payload.get("candidate_command_lateness_ms")),
        "control_callback_interval_ms": _finite_number(
            payload.get("control_callback_interval_ms")),
        "previous_diagnostic_work_wall_us": _finite_number(
            previous_diag.get("wall_us")),
        "previous_diagnostic_work_thread_cpu_us": _finite_number(
            previous_diag.get("thread_cpu_us")),
        "previous_diagnostic_plant_wall_us": _finite_number(
            previous_diag.get("plant_wall_us")),
        "previous_diagnostic_plant_thread_cpu_us": _finite_number(
            previous_diag.get("plant_thread_cpu_us")),
        "previous_diagnostic_queue_enqueue_wall_us": _finite_number(
            previous_diag.get("queue_enqueue_wall_us")),
        "previous_diagnostic_queue_enqueue_thread_cpu_us": _finite_number(
            previous_diag.get("queue_enqueue_thread_cpu_us")),
        "previous_diagnostic_plant_steps": previous_diag.get(
            "reconstructed_plant_steps"),
        "previous_diagnostic_json_build_wall_us": _finite_number(
            previous_diag.get("json_build_wall_us")),
        "previous_diagnostic_json_build_thread_cpu_us": _finite_number(
            previous_diag.get("json_build_thread_cpu_us")),
        "diagnostic_worker_published_frames": async_publisher.get(
            "published_frames"),
        "diagnostic_worker_replaced_frames": async_publisher.get(
            "replaced_frames"),
        "diagnostic_worker_publish_failures": async_publisher.get(
            "publish_failures"),
        "diagnostic_worker_lower_priority": async_publisher.get(
            "lower_priority"),
        "async_last_worker_queue_wait_us": _finite_number(
            async_publisher.get("last_queue_wait_us")),
        "async_last_worker_serialization_wall_us": _finite_number(
            async_publisher.get("last_serialization_wall_us")),
        "async_last_worker_serialization_thread_cpu_us": _finite_number(
            async_publisher.get("last_serialization_thread_cpu_us")),
        "async_last_worker_publish_wall_us": _finite_number(
            async_publisher.get("last_publish_wall_us")),
        "async_last_worker_publish_thread_cpu_us": _finite_number(
            async_publisher.get("last_publish_thread_cpu_us")),
        "qp_build_ms": (
            _finite_number(solver.get("qp_build_us")) / 1000.0
            if _finite_number(solver.get("qp_build_us")) is not None else None),
        "node_solve_ms": (
            _finite_number(solver.get("node_solve_us")) / 1000.0
            if _finite_number(solver.get("node_solve_us")) is not None else None),
        "nominal_build_ms": (
            _finite_number(solver.get("nominal_build_us")) / 1000.0
            if _finite_number(solver.get("nominal_build_us")) is not None else None),
        "jacobian_wall_ms": (
            _finite_number(solver.get("jacobian_wall_us")) / 1000.0
            if _finite_number(solver.get("jacobian_wall_us")) is not None else None),
        "jacobian_worker_cpu_sum_ms": (
            _finite_number(solver.get("jacobian_thread_cpu_us")) / 1000.0
            if _finite_number(solver.get("jacobian_thread_cpu_us")) is not None
            else None),
        "jacobian_workers_used": solver.get("jacobian_threads_used"),
        "jacobian_worker_cpu_us": worker_cpu,
        "jacobian_worker_stage_wall_us": worker_wall,
        "jacobian_worker_stage_count": worker_counts,
        "jacobian_worker_os_tid": worker_tids,
        "jacobian_slowest_worker_cpu_index": slowest_worker_cpu_index,
        "jacobian_slowest_worker_cpu_tid": (
            worker_tids[slowest_worker_cpu_index]
            if slowest_worker_cpu_index is not None and
            slowest_worker_cpu_index < len(worker_tids) else None),
        "jacobian_slowest_worker_cpu_us": (
            worker_cpu[slowest_worker_cpu_index]
            if slowest_worker_cpu_index is not None and
            slowest_worker_cpu_index < len(worker_cpu) else None),
        "jacobian_slowest_worker_wall_index": slowest_worker_wall_index,
        "jacobian_slowest_worker_wall_tid": (
            worker_tids[slowest_worker_wall_index]
            if slowest_worker_wall_index is not None and
            slowest_worker_wall_index < len(worker_tids) else None),
        "jacobian_slowest_worker_wall_us": (
            worker_wall[slowest_worker_wall_index]
            if slowest_worker_wall_index is not None and
            slowest_worker_wall_index < len(worker_wall) else None),
        "admm_ms": (
            _finite_number(solver.get("admm_us")) / 1000.0
            if _finite_number(solver.get("admm_us")) is not None else None),
        "nonlinear_validation_ms": (
            _finite_number(solver.get("nonlinear_validation_us")) / 1000.0
            if _finite_number(solver.get("nonlinear_validation_us")) is not None
            else None),
        "exact_rollout_ms": (
            _finite_number(solver.get("exact_rollout_us")) / 1000.0
            if _finite_number(solver.get("exact_rollout_us")) is not None
            else None),
        "nominal_seed_source": solver.get("nominal_control_seed_source"),
        "diagnostic_recomputed_plant_steps": payload.get(
            "diagnostic_recomputed_plant_steps"),
        "diagnostic_cached_source_output_count": payload.get(
            "diagnostic_cached_source_output_count"),
    }


def _progress_bins(
    rows: list[dict[str, Any]], bin_width_m: float,
) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        progress = _finite_number(row.get("progress_m"))
        if progress is not None:
            groups[int(progress // bin_width_m)].append(row)
    result = []
    for index, group in sorted(groups.items()):
        gates = [row["callback_to_gate_ms"] for row in group
                 if row.get("callback_to_gate_ms") is not None]
        result.append({
            "start_m": index * bin_width_m,
            "end_m": (index + 1) * bin_width_m,
            "cycles": len(group),
            "status_counts": dict(Counter(row.get("status", "unknown")
                                           for row in group)),
            "full_horizon_exact_accepts": sum(
                row.get("status") in {"accepted_optimal", "accepted_feasible",
                                      "accepted_line_search"} for row in group),
            "short_prefix_accepts": sum(
                row.get("status") == "accepted_candidate_prefix" for row in group),
            "gate_ms_p50": evaluator.percentile(gates, 0.50) if gates else None,
            "gate_ms_p95": evaluator.percentile(gates, 0.95) if gates else None,
            "gate_ms_max": max(gates) if gates else None,
            "gate_ms_p999": evaluator.percentile(gates, 0.999) if gates else None,
            "gate_violations": sum(
                row.get("publication_gate_budget_violation") is True
                for row in group),
            "diagnostic_json_build_wall_us_p95": _percentile_or_none(
                [row["previous_diagnostic_json_build_wall_us"] for row in group
                 if row.get("previous_diagnostic_json_build_wall_us") is not None],
                0.95),
            "async_last_worker_queue_wait_us_p95": _percentile_or_none(
                [row["async_last_worker_queue_wait_us"] for row in group
                 if row.get("async_last_worker_queue_wait_us") is not None], 0.95),
            "max_abs_cross_track_m": max(
                (abs(row["e_y_m"]) for row in group if row.get("e_y_m") is not None),
                default=None),
            "max_abs_heading_error_rad": max(
                (abs(row["e_psi_rad"]) for row in group
                 if row.get("e_psi_rad") is not None), default=None),
            "speed_error_rmse_mps": (
                sum(row["speed_error_mps"] ** 2 for row in group
                    if row.get("speed_error_mps") is not None) /
                sum(row.get("speed_error_mps") is not None for row in group)
            ) ** 0.5 if any(row.get("speed_error_mps") is not None
                            for row in group) else None,
        })
    return result


def _bag_time_origin_epoch_s(data: dict[str, list[dict[str, Any]]]) -> float:
    timestamps = [float(row["bag_time"]) for rows in data.values()
                  for row in rows if row.get("bag_time") is not None]
    return min(timestamps) if timestamps else 0.0


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "p50": evaluator.percentile(values, 0.50),
        "p95": evaluator.percentile(values, 0.95),
        "p99": evaluator.percentile(values, 0.99),
        "p999": evaluator.percentile(values, 0.999),
        "max": max(values),
    }


def _rmse(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return math.sqrt(sum(value * value for value in finite) / len(finite)) \
        if finite else None


def _duration_above(
    rows: list[dict[str, Any]], field: str, threshold: float,
    maximum_sample_gap_s: float = 0.1,
) -> float:
    ordered = sorted(rows, key=lambda row: row["bag_receipt_epoch_s"])
    duration = 0.0
    for current, following in zip(ordered, ordered[1:]):
        value = _finite_number(current.get(field))
        dt = following["bag_receipt_epoch_s"] - current["bag_receipt_epoch_s"]
        if value is not None and abs(value) > threshold and 0.0 < dt <= maximum_sample_gap_s:
            duration += dt
    return duration


def _percentile_or_none(values: list[float], probability: float) -> float | None:
    return evaluator.percentile(values, probability) if values else None


def make_report(
    root: Path, bag: Path, data: dict[str, list[dict[str, Any]]],
    trajectory: Path | None, supplied_metadata: Path | None,
    progress_bin_m: float, experiment_regime: str = "unspecified",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diag_topic = (evaluator.CANDIDATE_DIAGNOSTICS
                  if data.get(evaluator.CANDIDATE_DIAGNOSTICS)
                  else evaluator.DIAGNOSTICS)
    diagnostic_rows = [row for row in data.get(diag_topic, [])
                       if isinstance(row.get("json"), dict) and
                       "_decode_error" not in row["json"]]
    bridge_index = _bridge_by_request_stamp(data.get(evaluator.BRIDGE_TIMING, []))
    drive_index = _drive_command_by_stamp(data.get(evaluator.DRIVE_COMMAND, []))
    bag_time_origin_epoch_s = _bag_time_origin_epoch_s(data)
    cycles = [_cycle_row(float(row.get("bag_time", 0.0)),
                         bag_time_origin_epoch_s, row["json"],
                         bridge_index, drive_index)
              for row in diagnostic_rows]
    cycles.sort(key=lambda row: (row["bag_receipt_epoch_s"],
                                row["bridge_request_epoch_ns"] or 0))
    last_steering = None
    for row in cycles:
        speed = row.get("u_mps")
        target_speed = row.get("target_u_mps")
        row["speed_error_mps"] = (
            speed - target_speed
            if speed is not None and target_speed is not None else None)
        steering = row.get("issued_command_steering_rad")
        row["cross_cycle_steering_sign_reversal"] = (
            bool(last_steering * steering < 0.0)
            if last_steering is not None and steering is not None else None)
        limit = row.get("steering_limit_rad")
        row["steering_saturated"] = (
            abs(steering) >= limit - 1.0e-5
            if steering is not None and limit is not None else None)
        interval = row.get("control_callback_interval_ms")
        row["callback_interval_exceeds_25ms_period"] = (
            interval > 25.0 if interval is not None else None)
        last_steering = steering

    collision_time = _first_collision_time(data.get(evaluator.COLLISION, []))
    bridge_fault_time = _first_bridge_fault_time(data.get(evaluator.BRIDGE_FAULT, []))
    controller_fault_time = _first_unpublished_rejection_time(
        data.get(diag_topic, []))
    cutoff_events = [
        ("collision", collision_time),
        ("bridge_fault", bridge_fault_time),
        ("unpublished_mpc_rejection", controller_fault_time),
    ]
    observed_cutoffs = [value for _, value in cutoff_events
                        if value is not None]
    cutoff = min(observed_cutoffs) if observed_cutoffs else None
    cutoff_reasons = [name for name, value in cutoff_events
                      if cutoff is not None and value == cutoff]
    for row in cycles:
        row["score_eligible"] = (
            cutoff is None or row["bag_receipt_epoch_s"] <= cutoff)
    scored = [row for row in cycles if row["score_eligible"]]

    status_counts = Counter(row.get("status", "unknown") for row in scored)
    cycle_reason_counts = Counter(
        str(row["cycle_reason"]) for row in scored
        if isinstance(row.get("cycle_reason"), str) and row["cycle_reason"])
    nonlinear_failure_counts = Counter(
        str(row["nonlinear_failure_reason"]) for row in scored
        if isinstance(row.get("nonlinear_failure_reason"), str) and
        row["nonlinear_failure_reason"] not in ("", "none", "not_applicable"))
    candidate_prefix_failure_stage_counts = Counter(
        str(row["candidate_prefix_failure_stage"]) for row in scored
        if row.get("candidate_prefix_validation_attempted") is True and
        row.get("candidate_prefix_exact_valid") is False and
        isinstance(row.get("candidate_prefix_failure_stage"), (int, float)) and
        row["candidate_prefix_failure_stage"] >= 0)
    quality_prefix_rows = [row for row in scored
                           if row.get("quality_prefix_attempted") is True]
    quality_prefix_choice_counts = Counter(
        str(row.get("quality_prefix_choice") or "unknown")
        for row in quality_prefix_rows)
    gate_values = [row["callback_to_gate_ms"] for row in scored
                   if row.get("callback_to_gate_ms") is not None]
    errors = {
        name: [abs(float(row[field])) for row in scored
               if row.get(field) is not None]
        for name, field in (("cross_track_abs_m", "e_y_m"),
                            ("heading_abs_rad", "e_psi_rad"),
                            ("speed_abs_error_mps", "speed_error_mps"))
    }
    by_status: dict[str, dict[str, Any]] = {}
    for status in sorted({str(row.get("status", "unknown")) for row in scored}):
        status_rows = [row for row in scored if row.get("status") == status]
        by_status[status] = {
            "cycles": len(status_rows),
            "iterations": _distribution([
                float(row["iterations"]) for row in status_rows
                if row.get("iterations") is not None]),
            "primal_residual": _distribution([
                float(row["primal_residual"]) for row in status_rows
                if row.get("primal_residual") is not None]),
            "dual_residual": _distribution([
                float(row["dual_residual"]) for row in status_rows
                if row.get("dual_residual") is not None]),
            "gate_ms": _distribution([
                float(row["callback_to_gate_ms"]) for row in status_rows
                if row.get("callback_to_gate_ms") is not None]),
        }
    raw_lap_time_samples = data.get(evaluator.LAST_LAP, [])
    lap_times = _distinct_lap_times(
        raw_lap_time_samples, cutoff, data.get(evaluator.LAP))
    lap_counts = [_finite_number(row.get("value"))
                  for row in data.get(evaluator.LAP, [])
                  if cutoff is None or float(row.get("bag_time", 0.0)) <= cutoff]
    lap_counts = [int(value) for value in lap_counts if value is not None]
    summary = {
        "bag_duration_s": (float(data[evaluator.GT][-1]["bag_time"] -
                                  data[evaluator.GT][0]["bag_time"])
                           if data.get(evaluator.GT) else None),
        "diagnostic_topic": diag_topic,
        "diagnostic_messages_decoded": len(cycles),
        "bag_time_origin_epoch_s": bag_time_origin_epoch_s,
        "score_cutoff_reason": "+".join(cutoff_reasons) or None,
        "scored_through_first_fault_epoch_s": cutoff,
        "scored_through_first_fault_from_start_s": (
            cutoff - bag_time_origin_epoch_s if cutoff is not None else None),
        "first_collision_bag_receipt_epoch_s": collision_time,
        "first_collision_from_start_s": (
            collision_time - bag_time_origin_epoch_s
            if collision_time is not None else None),
        "first_bridge_fault_bag_receipt_epoch_s": bridge_fault_time,
        "first_bridge_fault_from_start_s": (
            bridge_fault_time - bag_time_origin_epoch_s
            if bridge_fault_time is not None else None),
        "first_unpublished_mpc_rejection_bag_receipt_epoch_s": controller_fault_time,
        "first_unpublished_mpc_rejection_from_start_s": (
            controller_fault_time - bag_time_origin_epoch_s
            if controller_fault_time is not None else None),
        "cycles_scored": len(scored),
        "status_counts": dict(status_counts),
        "cycle_reason_counts": dict(cycle_reason_counts),
        "nonlinear_failure_reason_counts": dict(nonlinear_failure_counts),
        "candidate_prefix_failure_stage_counts": dict(
            candidate_prefix_failure_stage_counts),
        "quality_prefix_policy_enabled": any(
            row.get("quality_prefix_policy_enabled") is True for row in scored),
        "quality_prefix_attempt_count": len(quality_prefix_rows),
        "quality_prefix_choice_counts": dict(quality_prefix_choice_counts),
        "quality_prefix_both_exact_valid_count": sum(
            row.get("quality_candidate_prefix_exact_valid") is True and
            row.get("quality_prior_prefix_exact_valid") is True
            for row in quality_prefix_rows),
        "quality_prefix_validation_us": _distribution([
            float(row["quality_prefix_validation_us"])
            for row in quality_prefix_rows
            if row.get("quality_prefix_validation_us") is not None]),
        "status_percentages": {
            status: 100.0 * count / len(scored)
            for status, count in status_counts.items()} if scored else {},
        "full_horizon_exact_accepts": sum(
            row.get("status") in {"accepted_optimal", "accepted_feasible",
                                  "accepted_line_search"} for row in scored),
        "short_prefix_accepts": sum(
            row.get("status") == "accepted_candidate_prefix" for row in scored),
        "prior_plan_accepts": sum(
            row.get("status") == "accepted_prior_plan" for row in scored),
        "published_commands": sum(row.get("command_published") is True
                                   for row in scored),
        "middleware_publish_us": _distribution([
            row["middleware_publish_us"] for row in scored
            if row.get("middleware_publish_us") is not None]),
        "command_history_us": _distribution([
            row["command_history_us"] for row in scored
            if row.get("command_history_us") is not None]),
        "gate_ms": ({
            "count": len(gate_values),
            "p50": evaluator.percentile(gate_values, 0.50),
            "p95": evaluator.percentile(gate_values, 0.95),
            "p99": evaluator.percentile(gate_values, 0.99),
            "p999": evaluator.percentile(gate_values, 0.999),
            "max": max(gate_values),
            "at_or_over_budget": sum(
                row.get("publication_gate_budget_violation") is True
                for row in scored),
        } if gate_values else None),
        "bridge_exact_request_stamp_matches": sum(
            row.get("bridge_request_sequence") is not None for row in cycles),
        "published_drive_command_exact_stamp_matches": sum(
            row.get("actual_published_command_steering_rad") is not None
            for row in cycles),
        "published_steering_messages_matching_candidate": sum(
            row.get("published_steering_message_matches_candidate") is True
            for row in cycles),
        "published_throttle_messages_matching_candidate": sum(
            row.get("published_throttle_message_matches_candidate") is True
            for row in cycles),
        "published_command_mismatches": sum(
            row.get("published_steering_message_matches_candidate") is False or
            row.get("published_throttle_message_matches_candidate") is False
            for row in cycles),
        "unity_command_effect_epochs_known": 0,
        "callback_intervals_over_25ms": sum(
            row.get("callback_interval_exceeds_25ms_period") is True
            for row in scored),
        "steering_sign_reversals_between_cycles": sum(
            row.get("cross_cycle_steering_sign_reversal") is True
            for row in scored),
        "steering_saturation_fraction": (
            sum(row.get("steering_saturated") is True for row in scored) /
            sum(row.get("steering_saturated") is not None for row in scored)
            if any(row.get("steering_saturated") is not None for row in scored)
            else None),
        "solver_termination_counts": dict(Counter(
            _normalized_solver_category(row.get("solver_stop_reason"))
            for row in scored)),
        "solver_stop_reason_counts": dict(Counter(
            row.get("solver_stop_reason") or "not_run_or_unknown"
            for row in scored)),
        "qp_build_deadline_skip_count": sum(
            row.get("solver_skipped_for_deadline") is True for row in scored),
        "qp_build_timeout_count": None,
        "qp_build_timeout_count_unknown_reason": (
            "the current MPC diagnostic schema has no distinct QP-build-timeout field"),
        "solver_by_mpc_outcome": by_status,
        "jacobian_profile": {
            "wall_us": _distribution([
                float(row["jacobian_wall_ms"]) * 1000.0 for row in scored
                if row.get("jacobian_wall_ms") is not None]),
            "sum_worker_cpu_us": _distribution([
                float(row["jacobian_worker_cpu_sum_ms"]) * 1000.0
                for row in scored
                if row.get("jacobian_worker_cpu_sum_ms") is not None]),
            "slowest_worker_cpu_us": _distribution([
                float(row["jacobian_slowest_worker_cpu_us"])
                for row in scored
                if row.get("jacobian_slowest_worker_cpu_us") is not None]),
            "slowest_worker_wall_us": _distribution([
                float(row["jacobian_slowest_worker_wall_us"])
                for row in scored
                if row.get("jacobian_slowest_worker_wall_us") is not None]),
            "workers_used_counts": dict(Counter(
                row.get("jacobian_workers_used") for row in scored
                if row.get("jacobian_workers_used") is not None)),
            "observed_worker_thread_ids": sorted({
                int(tid) for row in scored
                for tid in row.get("jacobian_worker_os_tid", [])
                if tid is not None and tid > 0}),
            "worker_stage_count_totals": [
                sum(row.get("jacobian_worker_stage_count", [])[index]
                    for row in scored
                    if index < len(row.get("jacobian_worker_stage_count", [])) and
                    row["jacobian_worker_stage_count"][index] is not None)
                for index in range(max((len(row.get(
                    "jacobian_worker_stage_count", [])) for row in scored),
                    default=0))],
        },
        "solver_profile": {
            "rho_changes": _distribution([
                float(row["rho_change_count"]) for row in scored
                if row.get("rho_change_count") is not None]),
            "factorizations": _distribution([
                float(row["factorization_count"]) for row in scored
                if row.get("factorization_count") is not None]),
            "regularizations": _distribution([
                float(row["regularization_count"]) for row in scored
                if row.get("regularization_count") is not None]),
            "max_regularization": _distribution([
                float(row["max_regularization"]) for row in scored
                if row.get("max_regularization") is not None]),
        },
        "diagnostic_profile": {
            "async_worker_sample_semantics": (
                "latest worker counters sampled before enqueuing this diagnostic; "
                "not correlated to this frame because no completion-frame ID exists"),
            "control_thread_total_wall_us": _distribution([
                float(row["previous_diagnostic_work_wall_us"])
                for row in scored
                if row.get("previous_diagnostic_work_wall_us") is not None]),
            "control_thread_total_cpu_us": _distribution([
                float(row["previous_diagnostic_work_thread_cpu_us"])
                for row in scored
                if row.get("previous_diagnostic_work_thread_cpu_us") is not None]),
            "json_build_wall_us": _distribution([
                float(row["previous_diagnostic_json_build_wall_us"])
                for row in scored
                if row.get("previous_diagnostic_json_build_wall_us") is not None]),
            "queue_enqueue_wall_us": _distribution([
                float(row["previous_diagnostic_queue_enqueue_wall_us"])
                for row in scored
                if row.get("previous_diagnostic_queue_enqueue_wall_us") is not None]),
            "async_last_worker_queue_wait_us": _distribution([
                float(row["async_last_worker_queue_wait_us"])
                for row in scored
                if row.get("async_last_worker_queue_wait_us") is not None]),
            "async_last_worker_serialization_wall_us": _distribution([
                float(row["async_last_worker_serialization_wall_us"])
                for row in scored
                if row.get("async_last_worker_serialization_wall_us") is not None]),
            "async_last_worker_publish_wall_us": _distribution([
                float(row["async_last_worker_publish_wall_us"])
                for row in scored
                if row.get("async_last_worker_publish_wall_us") is not None]),
            "replaced_frame_total_max": max(
                (int(row["diagnostic_worker_replaced_frames"])
                 for row in scored
                 if row.get("diagnostic_worker_replaced_frames") is not None),
                default=None),
            "serialization_worker_low_priority": (
                all(row.get("diagnostic_worker_lower_priority") is True
                    for row in scored
                    if row.get("diagnostic_worker_lower_priority") is not None)
                if any(row.get("diagnostic_worker_lower_priority") is not None
                       for row in scored) else None),
            "additional_plant_steps": sum(
                int(row.get("diagnostic_recomputed_plant_steps") or 0)
                for row in scored),
        },
        "absolute_error_distributions": {
            name: _distribution(values) for name, values in errors.items()},
        "controller_tracking": {
            "cross_track_rmse_m": _rmse([
                float(row["e_y_m"]) for row in scored
                if row.get("e_y_m") is not None]),
            "heading_rmse_rad": _rmse([
                float(row["e_psi_rad"]) for row in scored
                if row.get("e_psi_rad") is not None]),
            "speed_error_rmse_mps": _rmse([
                float(row["speed_error_mps"]) for row in scored
                if row.get("speed_error_mps") is not None]),
            "estimated_seconds_abs_cross_track_over_5cm": _duration_above(
                scored, "e_y_m", 0.05),
            "off_path_duration_method": (
                "sum observed diagnostic intervals <=100ms whose preceding "
                "sample has |e_y|>0.05m; controller estimate, not truth"),
        },
        "bridge_request_to_receive_ms": _distribution([
            float(row["bridge_request_to_receive_ms"]) for row in cycles
            if row.get("bridge_request_to_receive_ms") is not None]),
        "command_start_lateness_ms": _distribution([
            float(row["command_start_lateness_ms"]) for row in scored
            if row.get("command_start_lateness_ms") is not None]),
        "lap_count_max": max(lap_counts) if lap_counts else None,
        "lap_time_topic_samples": len(raw_lap_time_samples),
        "recorded_lap_times_s": lap_times,
        "lap_time_event_source": ("lap_counter_transition"
                                  if data.get(evaluator.LAP)
                                  else "duration_value_change_only"),
        "recorded_lap_time_s": _distribution(lap_times),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_regime": experiment_regime,
        "metadata": build_metadata(root, bag, trajectory, supplied_metadata),
        "summary": summary,
        "progress_bin_width_m": progress_bin_m,
        "per_progress_bin": _progress_bins(scored, progress_bin_m),
        "cycle_detail_format": "companion CSV; one row per MPC diagnostic cycle",
        "interpretation_limits": [
            "A bridge request stamp/sequence is not a Unity FixedUpdate applied-command ID.",
            "A candidate command stamp is a scheduled ROS command stamp, not proof of physical effect.",
            "Cross-topic cutoff uses bag receipt order; it is not a physical collision epoch.",
            "Short-prefix acceptance is not full-horizon feasibility.",
            "Unity truth is retained as a separate stream and is not substituted for odom.",
            "Async publisher timing samples are latest-completed snapshots, not frame-correlated.",
            "The current schema does not expose a distinct QP-build-timeout stop reason.",
        ],
    }
    return report, cycles


def write_cycle_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "bag_receipt_epoch_s", "bag_time_from_start_s", "score_eligible",
        "bridge_request_epoch_ns", "bridge_request_sequence",
        "bridge_received_epoch_ns", "bridge_request_to_receive_ms",
        "scheduled_command_stamp_ns",
        "issued_command_stamp_ns",
        "actual_published_command_steering_rad", "actual_published_command_throttle",
        "published_steering_message_matches_candidate",
        "published_throttle_message_matches_candidate",
        "scheduled_bridge_request_sequence",
        "actual_unity_effect_epoch", "status", "cycle_reason",
        "input_failure_reason", "nominal_build_failure_stage",
        "nominal_build_failure_constraint", "nominal_build_failure_index",
        "nonlinear_failure_reason", "nonlinear_failure_stage",
        "nonlinear_failure_detail", "action_source", "command_published",
        "solver_skipped_for_deadline",
        "progress_m", "e_y_m", "e_psi_rad", "u_mps", "v_mps", "r_radps",
        "actual_steering_rad", "target_u_mps", "curvature_inv_m",
        "candidate_first_steering_rad", "candidate_first_throttle",
        "issued_command_steering_rad", "issued_command_throttle",
        "nominal_steering_rad", "nominal_throttle",
        "steering_command_delta_rad", "throttle_command_delta",
        "candidate_prefix_chain_length", "prior_plan_chain_length", "solver_status",
        "solver_stop_reason", "iterations", "primal_residual", "dual_residual",
        "sqp_iterations_attempted", "sqp_iterations_completed", "sqp_iterations_accepted",
        "line_search_selected_alpha", "line_search_merit_triggered",
        "refinement_prefix_selected_pass", "refinement_first_prefix_objective",
        "refinement_second_prefix_objective", "refinement_prefix_validation_us",
        "state_primal_residual", "state_primal_stage", "state_primal_coordinate",
        "state_dual_residual", "state_dual_stage", "state_dual_coordinate",
        "input_primal_residual", "input_primal_stage", "input_primal_coordinate",
        "input_dual_residual", "input_dual_stage", "input_dual_coordinate",
        "objective", "full_horizon_failure_stage",
        "candidate_prefix_validation_attempted",
        "candidate_prefix_exact_valid", "candidate_prefix_failure_stage",
        "candidate_prefix_validation_status",
        "candidate_prefix_validation_horizon",
        "candidate_prefix_timing_rejected",
        "candidate_prefix_solver_converged",
        "candidate_prefix_solver_iteration_limited",
        "prior_plan_validation_attempted", "prior_plan_exact_valid",
        "prior_plan_failure_stage", "prior_plan_validation_status",
        "prior_plan_validation_horizon", "prior_plan_timing_rejected",
        "full_prior_suffix_attempted", "full_prior_suffix_exact_valid",
        "full_prior_suffix_failure_stage",
        "full_prior_suffix_validation_status",
        "full_prior_suffix_validation_horizon",
        "full_prior_suffix_timing_rejected",
        "full_prior_suffix_validation_us",
        "quality_prefix_policy_enabled", "quality_prefix_attempted",
        "quality_prefix_choice", "quality_candidate_prefix_exact_valid",
        "quality_prior_prefix_exact_valid",
        "quality_candidate_prefix_objective",
        "quality_prior_prefix_objective",
        "quality_candidate_prefix_min_slack_m",
        "quality_prior_prefix_min_slack_m", "quality_prefix_validation_us",
        "quality_candidate_prefix_controls_json",
        "quality_prior_prefix_controls_json",
        "prefix_validation_horizon",
        "nominal_candidate_control_delta_inf", "rho_change_count", "rho_start",
        "rho_final", "rho_u_start", "rho_u_final", "factorization_count",
        "regularization_count", "max_regularization",
        "nominal_vs_candidate_curvature_error_max_per_m",
        "nominal_vs_candidate_progress_error_max_m",
        "nominal_vs_candidate_left_bound_error_max_m",
        "nominal_vs_candidate_right_bound_error_max_m",
        "corridor_slack_m", "source_age_s", "callback_to_gate_ms",
        "callback_to_publish_return_ms", "middleware_publish_us", "command_history_us",
        "budget_ms", "publication_gate_budget_violation",
        "prior_plan_published_while_over_budget", "command_start_lateness_ms",
        "qp_build_ms", "node_solve_ms", "nominal_build_ms", "jacobian_wall_ms",
        "jacobian_worker_cpu_sum_ms", "admm_ms", "nonlinear_validation_ms",
        "jacobian_workers_used", "jacobian_worker_cpu_us",
        "jacobian_worker_stage_wall_us", "jacobian_worker_stage_count",
        "jacobian_worker_os_tid", "jacobian_slowest_worker_cpu_index",
        "jacobian_slowest_worker_cpu_tid", "jacobian_slowest_worker_cpu_us",
        "jacobian_slowest_worker_wall_index", "jacobian_slowest_worker_wall_tid",
        "jacobian_slowest_worker_wall_us",
        "exact_rollout_ms", "nominal_seed_source",
        "steering_limit_rad", "cross_cycle_steering_sign_reversal",
        "steering_saturated", "speed_error_mps",
        "callback_interval_exceeds_25ms_period",
        "control_callback_interval_ms", "previous_diagnostic_work_wall_us",
        "previous_diagnostic_work_thread_cpu_us",
        "previous_diagnostic_plant_wall_us", "previous_diagnostic_plant_thread_cpu_us",
        "previous_diagnostic_queue_enqueue_wall_us",
        "previous_diagnostic_queue_enqueue_thread_cpu_us",
        "previous_diagnostic_json_build_wall_us",
        "previous_diagnostic_json_build_thread_cpu_us",
        "diagnostic_worker_published_frames",
        "diagnostic_worker_replaced_frames",
        "diagnostic_worker_publish_failures",
        "diagnostic_worker_lower_priority",
        "async_last_worker_queue_wait_us",
        "async_last_worker_serialization_wall_us",
        "async_last_worker_serialization_thread_cpu_us",
        "async_last_worker_publish_wall_us",
        "async_last_worker_publish_thread_cpu_us",
        "previous_diagnostic_plant_steps", "diagnostic_recomputed_plant_steps",
        "diagnostic_cached_source_output_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, separators=(",", ":"))
                if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def write_diagnostic_sequence_sidecar(
        path: Path, rows: list[dict[str, Any]]) -> int:
    """Preserve complete decoded MPC diagnostics for offline branch replay.

    The cycle CSV is intentionally a compact scalar summary. This JSONL keeps
    the full per-cycle payload, including candidate/nominal controls and any
    certified-plan snapshot, so later analysis does not reconstruct a missing
    branch from published commands.
    """
    ordered = sorted(rows, key=lambda row: float(row.get("bag_time", 0.0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in ordered:
            payload = row.get("json")
            if not isinstance(payload, dict) or "_decode_error" in payload:
                continue
            record = {
                "bag_receipt_epoch_s": float(row.get("bag_time", 0.0)),
                "source_stamp_ns": payload.get("source_stamp_ns"),
                "control_ros_stamp_ns": payload.get("control_ros_stamp_ns"),
                "candidate_command_stamp_ns": payload.get(
                    "candidate_command_stamp_ns"),
                "issued_command_stamp_ns": payload.get("issued_command_stamp_ns"),
                "status": payload.get("status"),
                "action_source": payload.get("action_source"),
                "command_published": payload.get("command_published"),
                "payload": payload,
            }
            stream.write(json.dumps(record, separators=(",", ":"),
                                  allow_nan=False) + "\n")
    return sum(
        isinstance(row.get("json"), dict) and
        "_decode_error" not in row["json"] for row in ordered)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--run-metadata", type=Path,
                        help="optional JSON with Unity/container/YAML/reset details")
    parser.add_argument("--progress-bin-m", type=float, default=5.0)
    parser.add_argument(
        "--experiment-regime",
        choices=("live_closed_loop", "unspecified"),
        default="unspecified",
        help="explicitly set live_closed_loop only when MPC commands controlled the run")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--output-diagnostics-jsonl", type=Path,
        help="full per-cycle diagnostic payload sidecar (default: sibling of output CSV)")
    args = parser.parse_args()
    if args.progress_bin_m <= 0.0:
        parser.error("--progress-bin-m must be positive")
    data = evaluator.read_bag(args.bag)
    report, cycles = make_report(
        args.root.resolve(), args.bag.resolve(), data,
        args.trajectory.resolve() if args.trajectory else None,
        args.run_metadata.resolve() if args.run_metadata else None,
        args.progress_bin_m, args.experiment_regime)
    report["cycle_detail_csv"] = str(args.output_csv.resolve())
    diag_topic = (evaluator.CANDIDATE_DIAGNOSTICS
                  if data.get(evaluator.CANDIDATE_DIAGNOSTICS)
                  else evaluator.DIAGNOSTICS)
    diagnostic_sidecar = args.output_diagnostics_jsonl or args.output_csv.with_name(
        f"{args.output_csv.stem}_diagnostics.jsonl")
    sidecar_count = write_diagnostic_sequence_sidecar(
        diagnostic_sidecar, data.get(diag_topic, []))
    report["diagnostic_sequence_sidecar"] = {
        "path": str(diagnostic_sidecar.resolve()),
        "format": "JSONL; one record per decoded MPC diagnostic with full payload",
        "records": sidecar_count,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_cycle_csv(args.output_csv, cycles)
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(
        f"wrote schema={REPORT_SCHEMA} JSON={args.output_json} "
        f"CSV={args.output_csv} diagnostics={diagnostic_sidecar} "
        f"({sidecar_count} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
