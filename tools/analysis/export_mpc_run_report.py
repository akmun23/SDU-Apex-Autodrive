#!/usr/bin/env python3
"""Export MPC bag reports joined by exact bridge request stamps in a ROS 2 environment."""

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
    from tools.analysis import evaluate_mpc_bag as evaluator
except ModuleNotFoundError:
    # Preserve the established `python tools/analysis/export_mpc_run_report.py ...`
    # invocation where the script directory, rather than the repo root, is
    # sys.path[0].
    import evaluate_mpc_bag as evaluator


REPORT_SCHEMA = "sdu_apex_mpc_run_report/v4"
COMMAND_FLOAT_MATCH_TOL = 5.0e-8


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

    return []


def _per_lap_tracking(
    lap_rows: list[dict[str, Any]], cycles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-official-lap MPC diagnostics delimited by ROS bag lap-count epochs.

    Uses the same bag receipt clock for lap counter and MPC diagnostics, not
    inferred Unity applied-command epochs. The lap duration is taken separately
    from the official last_lap_time publication, not these receipt deltas.
    """
    transitions: list[tuple[int, float]] = []
    last_count = None
    for row in sorted(lap_rows, key=lambda r: r.get("bag_time", 0.0)):
        count = _finite_number(row.get("value"))
        epoch = _finite_number(row.get("bag_time"))
        if count is None or epoch is None or count < 0:
            continue
        count = int(count)
        if last_count is None or count > last_count:
            transitions.append((count, epoch))
            last_count = count
    result: list[dict[str, Any]] = []
    if not transitions:
        return result
    # A lap starts at the first observed baseline count; each +1 is one
    # observed completion. Missing counter transitions are never invented.
    for (previous, started), (current, ended) in zip(transitions, transitions[1:]):
        if current != previous + 1 or ended <= started:
            continue
        sample = [row for row in cycles if row.get("score_eligible") and
                  started <= (row.get("bag_receipt_epoch_s") or -1) < ended]
        cross = [float(row["e_y_m"]) for row in sample if row.get("e_y_m") is not None]
        heading = [float(row["e_psi_rad"]) for row in sample if row.get("e_psi_rad") is not None]
        speed = [float(row["speed_error_mps"]) for row in sample if row.get("speed_error_mps") is not None]
        gate = [float(row["callback_to_gate_ms"]) for row in sample if row.get("callback_to_gate_ms") is not None]
        result.append(dict(official_lap_index=current - transitions[0][0],
                           lap_counter_value=current,
                           bag_receipt_start_s=started, bag_receipt_end_s=ended,
                           diagnostic_cycles=len(sample),
                           cross_track_rmse_m=_rmse(cross),
                           cross_track_max_abs_m=max(map(abs, cross), default=None),
                           heading_rmse_rad=_rmse(heading),
                           speed_error_rmse_mps=_rmse(speed),
                           callback_to_gate_ms=_distribution(gate),
                           status_counts=dict(Counter(row.get("status") for row in sample))))
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
    return indexed


def _cycle_row(
    bag_receipt_epoch_s: float, bag_time_origin_epoch_s: float,
    payload: dict[str, Any],
    bridge_by_stamp: dict[int, dict[str, Any]],
    drive_by_stamp: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    def object_field(name):
        value = payload.get(name)
        return value if isinstance(value, dict) else {}

    prediction = object_field("control_time_prediction")
    solver = object_field("solver")
    state = payload.get("state")
    action = payload.get("first_action")
    nominal = payload.get("nominal_first_action")
    reference = object_field("reference")

    def coordinate(values, index):
        return _finite_number(values[index]) if isinstance(values, list) and len(values) > index else None

    def epoch(value):
        return value if isinstance(value, int) and value > 0 else None

    source_epoch = epoch(prediction.get("plant_source_stamp_ns")) or epoch(
        payload.get("source_stamp_ns"))
    scheduled_epoch = epoch(payload.get("candidate_command_stamp_ns"))
    issued_epoch = epoch(payload.get("issued_command_stamp_ns"))
    bridge = bridge_by_stamp.get(source_epoch, {})
    scheduled_bridge = bridge_by_stamp.get(scheduled_epoch, {})
    drive = drive_by_stamp.get(issued_epoch, {})
    received_epoch = bridge.get("bridge_receive_ros_stamp_ns")
    published = payload.get("command_published") is True
    steering, throttle = coordinate(action, 0), coordinate(action, 1)
    nominal_steering, nominal_throttle = coordinate(nominal, 0), coordinate(nominal, 1)
    budget = _finite_number(payload.get("end_to_end_cycle_budget_ms"))
    gate = _finite_number(payload.get("publication_gate_elapsed_ms"))
    termination = solver.get("termination")
    termination = termination if isinstance(termination, str) else "not_run"
    candidate_controls = payload.get("candidate_controls")
    nominal_controls = payload.get("nominal_controls")
    deltas = []
    if (isinstance(candidate_controls, list) and isinstance(nominal_controls, list) and
            len(candidate_controls) == len(nominal_controls)):
        for candidate, seed in zip(candidate_controls, nominal_controls):
            if not isinstance(candidate, list) or not isinstance(seed, list):
                continue
            for index in range(2):
                value, initial = coordinate(candidate, index), coordinate(seed, index)
                if value is not None and initial is not None:
                    deltas.append(abs(value - initial))

    def matches(field, value):
        actual = _finite_number(drive.get(field))
        return abs(actual - value) < COMMAND_FLOAT_MATCH_TOL if (
            published and actual is not None and value is not None) else None

    return {
        "bag_receipt_epoch_s": bag_receipt_epoch_s,
        "bag_time_from_start_s": bag_receipt_epoch_s - bag_time_origin_epoch_s,
        "bridge_request_epoch_ns": source_epoch,
        "bridge_request_sequence": bridge.get("request_sequence"),
        "bridge_received_epoch_ns": received_epoch,
        "bridge_request_to_receive_ms": (
            (received_epoch - source_epoch) * 1.e-6
            if isinstance(received_epoch, int) and source_epoch is not None else None),
        "scheduled_command_stamp_ns": scheduled_epoch,
        "issued_command_stamp_ns": issued_epoch,
        "scheduled_bridge_request_sequence": scheduled_bridge.get("request_sequence"),
        "actual_published_command_steering_rad": _finite_number(drive.get("steering")),
        "actual_published_command_throttle": _finite_number(drive.get("throttle")),
        "published_steering_message_matches_candidate": matches("steering", steering),
        "published_throttle_message_matches_candidate": matches("throttle", throttle),
        "actual_unity_effect_epoch": None,
        "status": payload.get("status"),
        "cycle_reason": payload.get("reason"),
        "input_failure_reason": payload.get("input_failure_reason"),
        "nonlinear_failure_reason": payload.get("nonlinear_failure_reason"),
        "nonlinear_failure_stage": payload.get("nonlinear_failure_stage"),
        "command_published": payload.get("command_published"),
        "progress_m": _finite_number(payload.get("progress_m")),
        "e_y_m": coordinate(state, 0),
        "e_psi_rad": coordinate(state, 1),
        "u_mps": coordinate(state, 2),
        "v_mps": coordinate(state, 3),
        "r_radps": coordinate(state, 4),
        "actual_steering_rad": coordinate(state, 7),
        "target_u_mps": _finite_number(reference.get("speed_reference_mps")),
        "candidate_first_steering_rad": steering,
        "candidate_first_throttle": throttle,
        "issued_command_steering_rad": steering if published else None,
        "issued_command_throttle": throttle if published else None,
        "nominal_steering_rad": nominal_steering,
        "nominal_throttle": nominal_throttle,
        "steering_command_delta_rad": (steering - nominal_steering
            if steering is not None and nominal_steering is not None else None),
        "throttle_command_delta": (throttle - nominal_throttle
            if throttle is not None and nominal_throttle is not None else None),
        "solver_status": termination,
        "solver_stop_reason": termination,
        "sqp_iterations_attempted": solver.get("sqp_iterations_attempted"),
        "sqp_iterations_accepted": solver.get("sqp_iterations_accepted"),
        "source_trials": solver.get("source_exact_nlp_source_trials"),
        "kkt_converged": solver.get("source_exact_nlp_kkt_converged"),
        "objective": _finite_number(solver.get("source_exact_nlp_original_cost_final")),
        "nominal_candidate_control_delta_inf": max(deltas, default=None),
        "corridor_slack_m": _finite_number(
            solver.get("source_exact_nlp_original_minimum_corner_margin_m")),
        "physical_clearance_m": _finite_number(
            solver.get("physical_minimum_corner_clearance_m")),
        "source_age_s": _finite_number(prediction.get("age_s")),
        "callback_to_gate_ms": gate,
        "callback_to_publish_return_ms": _finite_number(payload.get("end_to_end_control_ms")),
        "budget_ms": budget,
        "publication_gate_budget_violation": (gate >= budget
            if gate is not None and budget is not None else None),
        "node_solve_ms": (_finite_number(payload.get("solve_us")) / 1000.
            if _finite_number(payload.get("solve_us")) is not None else None),
        "nominal_build_ms": (_finite_number(payload.get("nominal_build_us")) / 1000.
            if _finite_number(payload.get("nominal_build_us")) is not None else None),
        "sqp_ms": (_finite_number(payload.get("sqp_us")) / 1000.
            if _finite_number(payload.get("sqp_us")) is not None else None),
        "nonlinear_validation_ms": (_finite_number(payload.get("validation_us")) / 1000.
            if _finite_number(payload.get("validation_us")) is not None else None),
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
            "status_counts": dict(Counter(row.get("status", "unknown") for row in group)),
            "full_horizon_exact_accepts": sum(
                row.get("status") == "accepted_feasible" for row in group),
            "gate_ms_p50": evaluator.percentile(gates, 0.50) if gates else None,
            "gate_ms_p95": evaluator.percentile(gates, 0.95) if gates else None,
            "gate_ms_max": max(gates) if gates else None,
            "gate_ms_p999": evaluator.percentile(gates, 0.999) if gates else None,
            "gate_violations": sum(
                row.get("publication_gate_budget_violation") is True for row in group),
            "max_abs_cross_track_m": max(
                (abs(row["e_y_m"]) for row in group if row.get("e_y_m") is not None),
                default=None),
            "max_abs_heading_error_rad": max(
                (abs(row["e_psi_rad"]) for row in group if row.get("e_psi_rad") is not None),
                default=None),
            "speed_error_rmse_mps": _rmse([
                row["speed_error_mps"] for row in group
                if row.get("speed_error_mps") is not None]),
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


def make_report(
    root: Path, bag: Path, data: dict[str, list[dict[str, Any]]],
    trajectory: Path | None, supplied_metadata: Path | None,
    progress_bin_m: float, experiment_regime: str = "unspecified",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diag_topic = (evaluator.CANDIDATE_DIAGNOSTICS
                  if data.get(evaluator.CANDIDATE_DIAGNOSTICS) else evaluator.DIAGNOSTICS)
    diagnostic_rows = [row for row in data.get(diag_topic, [])
                       if isinstance(row.get("json"), dict) and "_decode_error" not in row["json"]]
    bridge_index = _bridge_by_request_stamp(data.get(evaluator.BRIDGE_TIMING, []))
    drive_index = _drive_command_by_stamp(data.get(evaluator.DRIVE_COMMAND, []))
    bag_time_origin_epoch_s = _bag_time_origin_epoch_s(data)
    cycles = [_cycle_row(float(row.get("bag_time", 0.0)), bag_time_origin_epoch_s,
                         row["json"], bridge_index, drive_index) for row in diagnostic_rows]
    cycles.sort(key=lambda row: (row["bag_receipt_epoch_s"], row["bridge_request_epoch_ns"] or 0))
    last_steering = None
    for row in cycles:
        speed, target = row.get("u_mps"), row.get("target_u_mps")
        row["speed_error_mps"] = speed - target if speed is not None and target is not None else None
        steering = row.get("issued_command_steering_rad")
        row["cross_cycle_steering_sign_reversal"] = (
            bool(last_steering * steering < 0.0)
            if last_steering is not None and steering is not None else None)
        last_steering = steering

    collision_time = _first_collision_time(data.get(evaluator.COLLISION, []))
    bridge_fault_time = _first_bridge_fault_time(data.get(evaluator.BRIDGE_FAULT, []))
    controller_fault_time = _first_unpublished_rejection_time(data.get(diag_topic, []))
    cutoff_events = [("collision", collision_time), ("bridge_fault", bridge_fault_time),
                     ("unpublished_mpc_rejection", controller_fault_time)]
    cutoff = min((value for _, value in cutoff_events if value is not None), default=None)
    cutoff_reasons = [name for name, value in cutoff_events if cutoff is not None and value == cutoff]
    for row in cycles:
        row["score_eligible"] = cutoff is None or row["bag_receipt_epoch_s"] <= cutoff
    scored = [row for row in cycles if row["score_eligible"]]
    status_counts = Counter(row.get("status", "unknown") for row in scored)
    reasons = Counter(str(row["cycle_reason"]) for row in scored if row.get("cycle_reason"))
    nonlinear_failures = Counter(str(row["nonlinear_failure_reason"]) for row in scored
        if row.get("nonlinear_failure_reason") not in (None, "", "none", "not_applicable"))
    gate_values = [row["callback_to_gate_ms"] for row in scored if row.get("callback_to_gate_ms") is not None]
    errors = {name: [abs(row[field]) for row in scored if row.get(field) is not None]
              for name, field in (("cross_track_abs_m", "e_y_m"), ("heading_abs_rad", "e_psi_rad"),
                                  ("speed_abs_error_mps", "speed_error_mps"))}
    by_status = {}
    for status in sorted({str(row.get("status", "unknown")) for row in scored}):
        samples = [row for row in scored if row.get("status") == status]
        by_status[status] = {
            "cycles": len(samples),
            "sqp_iterations_attempted": _distribution([row["sqp_iterations_attempted"] for row in samples
                if row.get("sqp_iterations_attempted") is not None]),
            "gate_ms": _distribution([row["callback_to_gate_ms"] for row in samples
                if row.get("callback_to_gate_ms") is not None]),
        }
    raw_lap_time_samples = data.get(evaluator.LAST_LAP, [])
    lap_times = _distinct_lap_times(raw_lap_time_samples, cutoff, data.get(evaluator.LAP))
    lap_counts = [_finite_number(row.get("value")) for row in data.get(evaluator.LAP, [])
                  if cutoff is None or float(row.get("bag_time", 0.0)) <= cutoff]
    lap_counts = [int(value) for value in lap_counts if value is not None]
    gate_summary = _distribution(gate_values)
    if gate_summary is not None:
        gate_summary["at_or_over_budget"] = sum(
            row.get("publication_gate_budget_violation") is True for row in scored)
    terminations = dict(Counter(row["solver_stop_reason"] for row in scored))
    summary = {
        "bag_duration_s": (float(data[evaluator.GT][-1]["bag_time"] - data[evaluator.GT][0]["bag_time"])
                           if data.get(evaluator.GT) else None),
        "diagnostic_topic": diag_topic,
        "diagnostic_messages_decoded": len(cycles),
        "bag_time_origin_epoch_s": bag_time_origin_epoch_s,
        "score_cutoff_reason": "+".join(cutoff_reasons) or None,
        "scored_through_first_fault_epoch_s": cutoff,
        "scored_through_first_fault_from_start_s": cutoff - bag_time_origin_epoch_s if cutoff is not None else None,
        "first_collision_bag_receipt_epoch_s": collision_time,
        "first_collision_from_start_s": collision_time - bag_time_origin_epoch_s if collision_time is not None else None,
        "first_bridge_fault_bag_receipt_epoch_s": bridge_fault_time,
        "first_bridge_fault_from_start_s": bridge_fault_time - bag_time_origin_epoch_s if bridge_fault_time is not None else None,
        "first_unpublished_mpc_rejection_bag_receipt_epoch_s": controller_fault_time,
        "first_unpublished_mpc_rejection_from_start_s": controller_fault_time - bag_time_origin_epoch_s if controller_fault_time is not None else None,
        "cycles_scored": len(scored),
        "status_counts": dict(status_counts),
        "cycle_reason_counts": dict(reasons),
        "nonlinear_failure_reason_counts": dict(nonlinear_failures),
        "status_percentages": {status: 100.0 * count / len(scored) for status, count in status_counts.items()} if scored else {},
        "full_horizon_exact_accepts": sum(row.get("status") == "accepted_feasible" for row in scored),
        "published_commands": sum(row.get("command_published") is True for row in scored),
        "gate_ms": gate_summary,
        "bridge_exact_request_stamp_matches": sum(row.get("bridge_request_sequence") is not None for row in cycles),
        "published_drive_command_exact_stamp_matches": sum(row.get("actual_published_command_steering_rad") is not None for row in cycles),
        "published_steering_messages_matching_candidate": sum(row.get("published_steering_message_matches_candidate") is True for row in cycles),
        "published_throttle_messages_matching_candidate": sum(row.get("published_throttle_message_matches_candidate") is True for row in cycles),
        "published_command_mismatches": sum(row.get("published_steering_message_matches_candidate") is False or
            row.get("published_throttle_message_matches_candidate") is False for row in cycles),
        "unity_command_effect_epochs_known": 0,
        "steering_sign_reversals_between_cycles": sum(row.get("cross_cycle_steering_sign_reversal") is True for row in scored),
        "solver_termination_counts": terminations,
        "solver_stop_reason_counts": terminations,
        "solver_by_mpc_outcome": by_status,
        "absolute_error_distributions": {name: _distribution(values) for name, values in errors.items()},
        "controller_tracking": {
            "cross_track_rmse_m": _rmse([row["e_y_m"] for row in scored if row.get("e_y_m") is not None]),
            "heading_rmse_rad": _rmse([row["e_psi_rad"] for row in scored if row.get("e_psi_rad") is not None]),
            "speed_error_rmse_mps": _rmse([row["speed_error_mps"] for row in scored if row.get("speed_error_mps") is not None]),
            "estimated_seconds_abs_cross_track_over_5cm": _duration_above(scored, "e_y_m", 0.05),
            "off_path_duration_method": "sum observed diagnostic intervals <=100ms whose preceding sample has |e_y|>0.05m; controller estimate, not truth",
        },
        "bridge_request_to_receive_ms": _distribution([row["bridge_request_to_receive_ms"] for row in cycles
            if row.get("bridge_request_to_receive_ms") is not None]),
        "lap_count_max": max(lap_counts) if lap_counts else None,
        "lap_time_topic_samples": len(raw_lap_time_samples),
        "recorded_lap_times_s": lap_times,
        "lap_time_event_source": "lap_counter_transition" if data.get(evaluator.LAP) else "unavailable",
        "recorded_lap_time_s": _distribution(lap_times),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_regime": experiment_regime,
        "metadata": build_metadata(root, bag, trajectory, supplied_metadata),
        "summary": summary,
        "per_official_lap_tracking": _per_lap_tracking(data.get(evaluator.LAP, []), cycles),
        "progress_bin_width_m": progress_bin_m,
        "per_progress_bin": _progress_bins(scored, progress_bin_m),
        "cycle_detail_format": "companion CSV; one row per MPC diagnostic cycle",
        "interpretation_limits": [
            "A bridge request stamp/sequence is not a Unity FixedUpdate applied-command ID.",
            "A candidate command stamp is a scheduled ROS command stamp, not proof of physical effect.",
            "Cross-topic cutoff uses bag receipt order; it is not a physical collision epoch.",
            "Unity truth is retained as a separate stream and is not substituted for odom.",
        ],
    }
    return report, cycles


def write_cycle_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(_cycle_row(0., 0., {}, {}, {})) + [
        "speed_error_mps", "cross_cycle_steering_sign_reversal", "score_eligible"]
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
    """Preserve complete decoded MPC diagnostics for offline replay.

    The cycle CSV is intentionally a compact scalar summary. This JSONL keeps
    the full per-cycle payload, including candidate, nominal and source-optimized
    controls, so analysis can reproduce the frozen source problem.
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
                "command_published": payload.get("command_published"),
                "payload": payload,
            }
            stream.write(json.dumps(record, separators=(",", ":"),
                                  allow_nan=False) + "\n")
    return sum(
        isinstance(row.get("json"), dict) and
        "_decode_error" not in row["json"] for row in ordered)


def summarize_fixed_command_rollout(windows):
    """Retrospective dynamics check, never unexecuted-plan tracking accuracy."""
    valid = [row for row in windows if isinstance(row.get('xy_m'), (int, float))
             and math.isfinite(row['xy_m'])]
    if not valid:
        return dict(status='unavailable', windows=0, error='No aligned pre-fault 750 ms windows')
    worst = max(valid, key=lambda row: row['path_max_xy_m'])
    return dict(status='scored', windows=len(valid), horizon_ms=750,
        endpoint_xy_rmse_m=_rmse([row['xy_m'] for row in valid]),
        endpoint_xy_max_m=max(row['xy_m'] for row in valid),
        full_path_xy_max_m=max(row['path_max_xy_m'] for row in valid),
        paths_over_5cm=sum(row['path_max_xy_m'] > .05 for row in valid),
        forward_speed_rmse_mps=_rmse([row['end_u_error_mps'] for row in valid]),
        lateral_speed_rmse_mps=_rmse([row['end_v_error_mps'] for row in valid]),
        yaw_rate_rmse_radps=_rmse([row['end_r_error_radps'] for row in valid]),
        worst_window={key: (worst.get(key) if isinstance(worst.get(key), (int, float))
                           and math.isfinite(worst[key]) else None) for key in (
            'start_stamp_ns', 'start_bag_time_s', 'start_speed_mps',
            'start_steering_rad', 'progress_m', 'path_max_xy_m', 'path_max_step')},
        interpretation='Logged start plus actual issued commands; no future state corrections. '
                       'Overlapping windows are not independent trials. Not an unexecuted MPC-plan test.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2])
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
    parser.add_argument('--source-plant-library', type=Path,
                        help='also score fixed-issued-command 750 ms replay against truth')
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
    if args.source_plant_library:
        try:
            cutoff = report['summary'].get('scored_through_first_fault_epoch_s')
            before_fault = lambda topic: [row for row in data.get(topic, [])
                                         if cutoff is None or row.get('bag_time', math.inf) <= cutoff]
            result = evaluator.report_source_plant_rollout(
                before_fault(evaluator.GT), before_fault(evaluator.PLANT),
                before_fault(evaluator.ODOM_DIAGNOSTICS), args.source_plant_library,
                before_fault(evaluator.IMU), before_fault(evaluator.LEFT_ENCODER),
                before_fault(evaluator.RIGHT_ENCODER))
            windows = (result or {}).get('valid_750ms_windows', [])
            report['summary']['fixed_command_rollout'] = summarize_fixed_command_rollout(windows)
            report['summary']['fixed_command_rollout']['library_sha256'] = sha256_file(args.source_plant_library)
        except (OSError, ValueError, RuntimeError) as error:
            report['summary']['fixed_command_rollout'] = dict(status='unavailable', error=str(error))
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
