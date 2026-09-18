#!/usr/bin/env python3
"""Summarize per-cycle MPC diagnostics captured by the allowed telemetry recorder.

This tool is offline-only. It reads controller diagnostics and legal state
fields already published by MPC; it does not read simulator truth or alter
controller decisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import gzip
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


CYCLE_FIELDS = [
    "event_index", "source_stamp_ns", "cycle_status", "solver_iterations",
    "primal_residual", "dual_residual", "max_residual",
    "max_regularization", "regularization_count",
    "nonsmooth_jacobian_columns", "nonlinear_failure_stage",
    "nonlinear_failure_reason", "r1_nonlinear_failure_stage",
    "r1_nonlinear_failure_reason", "r2_nonlinear_failure_stage",
    "r2_nonlinear_failure_reason", "reason", "state_age_s", "control_time_age_s",
    "control_time_mode", "command_changes_used", "command_fallback",
    "current_progress_m", "map_x_m", "map_y_m", "map_yaw_rad",
    "path_curvature_per_m", "curvature_source", "u_mps", "v_mps", "r_radps", "e_y_m",
    "e_psi_rad", "target_speed_mps", "steering_command_rad",
    "previous_steering_rate_radps", "previous_target_speed_rate_mps2",
    "first_steering_rate_radps", "first_target_speed_rate_mps2",
    "published_steering_rad", "published_target_speed_mps",
    "abs_curvature_per_m", "abs_steering_rad", "abs_steering_rate_radps",
    "abs_yaw_rate_radps", "lateral_accel_proxy_mps2",
    "nominal_vs_candidate_progress_error_max_m",
    "nominal_vs_candidate_curvature_error_max_per_m",
    "nominal_vs_candidate_left_bound_error_max_m",
    "nominal_vs_candidate_right_bound_error_max_m",
    "minimum_predicted_corridor_slack_m", "lateral_accel_proxy_stage",
    "rti_iterations_used", "rti2_triggered", "rti2_trigger_reason_mask",
    "rti2_budget_skipped", "r1_status", "r2_status",
    "r1_nonlinear_objective", "r2_nonlinear_objective",
    "r1_min_corridor_slack", "r2_min_corridor_slack",
    "r1_solver_iterations", "r2_solver_iterations", "r1_solve_us",
    "r2_solve_us", "total_rti_us", "selected_candidate",
    "rho_start", "rho_final", "rho_u_start", "rho_u_final",
    "rho_change_count", "factorization_count", "factorization_time_ns",
    "operating_mode", "operating_mode_source", "steering_reversal",
]


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested_payload(raw: str) -> dict[str, Any] | None:
    """Decode recorder JSON, including std_msgs/String's nested value string."""
    try:
        value: Any = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    for _ in range(4):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
            continue
        if isinstance(value, dict):
            nested = next((value[key] for key in ("value", "data", "payload")
                           if isinstance(value.get(key), str)), None)
            if nested is not None:
                try:
                    value = json.loads(nested)
                except json.JSONDecodeError:
                    return None
                continue
            return value
        return None
    return value if isinstance(value, dict) else None


def _array_number(values: Any, index: int) -> float | None:
    if not isinstance(values, list) or len(values) <= index:
        return None
    return _finite_number(values[index])


def flatten_diagnostic(event: dict[str, str], payload: dict[str, Any],
                       path_curve: float | None = None) -> dict[str, Any]:
    state = payload.get("state")
    if not isinstance(state, list):
        state = []
    first_action = payload.get("first_action")
    if not isinstance(first_action, list):
        first_action = []
    control_time = payload.get("control_time_prediction")
    if not isinstance(control_time, dict):
        control_time = {}
    solver = payload.get("solver")
    if not isinstance(solver, dict):
        solver = {}
    source_pose = control_time.get("source_map_pose")
    if not isinstance(source_pose, list):
        source_pose = []

    curvature = _finite_number(payload.get("path_curvature_per_m"))
    curvature_source = "diagnostic" if curvature is not None else None
    if curvature is None:
        # Backward compatibility for older traces: infer local curvature from
        # the first logged reference yaw-rate / speed, and mark that it is an
        # approximation rather than the current exact path sample.
        predictions = payload.get("predictions")
        if isinstance(predictions, list) and predictions:
            reference = predictions[0].get("reference") if isinstance(
                predictions[0], dict) else None
            ref_u = _array_number(reference, 2)
            ref_r = _array_number(reference, 4)
            if ref_u is not None and ref_r is not None and abs(ref_u) > 1.0e-6:
                curvature = ref_r / ref_u
                curvature_source = "first_horizon_reference"
    if curvature is None and path_curve is not None:
        curvature = path_curve
        curvature_source = "configured_trajectory_at_progress"

    u = _array_number(state, 2)
    v = _array_number(state, 3)
    yaw_rate = _array_number(state, 4)
    steering = _array_number(state, 6)
    previous_steering_rate = _array_number(state, 7)
    first_steering_rate = _array_number(first_action, 2)
    first_target_rate = _array_number(first_action, 3)
    control_stamp = _finite_number(payload.get("control_ros_stamp_ns"))
    source_stamp = _finite_number(payload.get("source_stamp_ns"))
    if source_stamp is None:
        source_stamp = _finite_number(event.get("header_stamp_ns"))
    age = _finite_number(payload.get("source_age_s"))
    if age is None and source_stamp is not None and control_stamp is not None:
        age = max(0.0, (control_stamp - source_stamp) * 1.0e-9)
    lateral_load = (abs(u * yaw_rate)
                    if u is not None and yaw_rate is not None else None)

    accepted_status = str(payload.get("status", "")).startswith("accepted")
    operating_rate = first_target_rate if accepted_status else _array_number(state, 8)
    operating_mode_source = "candidate_action" if accepted_status else "last_action_state"
    mode = "unknown"
    if operating_rate is not None:
        if operating_rate < -1.0e-3:
            mode = "braking_command"
        elif operating_rate > 1.0e-3:
            mode = "powered_acceleration_command"
        else:
            mode = "near_zero_speed_slew"
    else:
        operating_mode_source = "unavailable"
    reversal = None
    if previous_steering_rate is not None and first_steering_rate is not None:
        reversal = (previous_steering_rate * first_steering_rate < 0.0 and
                    abs(previous_steering_rate) > 0.05 and
                    abs(first_steering_rate) > 0.05)

    flat: dict[str, Any] = {field: None for field in CYCLE_FIELDS}
    flat.update({
        "event_index": event.get("event_index"),
        "source_stamp_ns": int(source_stamp) if source_stamp is not None else None,
        "cycle_status": payload.get("status", "unknown"),
        "solver_iterations": solver.get("iterations"),
        "primal_residual": solver.get("primal_residual"),
        "dual_residual": solver.get("dual_residual"),
        "max_regularization": solver.get("max_regularization"),
        "regularization_count": solver.get("regularization_count"),
        "nonsmooth_jacobian_columns": solver.get("nonsmooth_columns"),
        "nonlinear_failure_stage": payload.get("nonlinear_failure_stage"),
        "nonlinear_failure_reason": payload.get("nonlinear_failure_reason"),
        "r1_nonlinear_failure_stage": payload.get("r1_nonlinear_failure_stage"),
        "r1_nonlinear_failure_reason": payload.get("r1_nonlinear_failure_reason"),
        "r2_nonlinear_failure_stage": payload.get("r2_nonlinear_failure_stage"),
        "r2_nonlinear_failure_reason": payload.get("r2_nonlinear_failure_reason"),
        "reason": payload.get("reason"),
        "state_age_s": age,
        "control_time_age_s": control_time.get("age_s"),
        "control_time_mode": control_time.get("mode"),
        "command_changes_used": control_time.get("command_changes_used"),
        "command_fallback": control_time.get("command_fallback"),
        "current_progress_m": payload.get("progress_m"),
        "map_x_m": _array_number(source_pose, 0),
        "map_y_m": _array_number(source_pose, 1),
        "map_yaw_rad": _array_number(source_pose, 2),
        "path_curvature_per_m": curvature,
        "curvature_source": curvature_source,
        "u_mps": u,
        "v_mps": v,
        "r_radps": yaw_rate,
        "e_y_m": _array_number(state, 0),
        "e_psi_rad": _array_number(state, 1),
        "target_speed_mps": _array_number(state, 5),
        "steering_command_rad": steering,
        "previous_steering_rate_radps": previous_steering_rate,
        "previous_target_speed_rate_mps2": _array_number(state, 8),
        "first_steering_rate_radps": first_steering_rate,
        "first_target_speed_rate_mps2": first_target_rate,
        "published_steering_rad": _array_number(first_action, 0),
        "published_target_speed_mps": _array_number(first_action, 1),
        "abs_curvature_per_m": abs(curvature) if curvature is not None else None,
        "abs_steering_rad": abs(steering) if steering is not None else None,
        "abs_steering_rate_radps": (abs(first_steering_rate)
                                    if first_steering_rate is not None else None),
        "abs_yaw_rate_radps": abs(yaw_rate) if yaw_rate is not None else None,
        "lateral_accel_proxy_mps2": payload.get(
            "lateral_accel_proxy_mps2", lateral_load),
        "nominal_vs_candidate_progress_error_max_m": payload.get(
            "nominal_vs_candidate_progress_error_max_m"),
        "nominal_vs_candidate_curvature_error_max_per_m": payload.get(
            "nominal_vs_candidate_curvature_error_max_per_m"),
        "nominal_vs_candidate_left_bound_error_max_m": payload.get(
            "nominal_vs_candidate_left_bound_error_max_m"),
        "nominal_vs_candidate_right_bound_error_max_m": payload.get(
            "nominal_vs_candidate_right_bound_error_max_m"),
        "minimum_predicted_corridor_slack_m": payload.get(
            "minimum_predicted_corridor_slack_m"),
        "lateral_accel_proxy_stage": payload.get("lateral_accel_proxy_stage"),
        "rti_iterations_used": payload.get("rti_iterations_used"),
        "rti2_triggered": payload.get("rti2_triggered"),
        "rti2_trigger_reason_mask": payload.get("rti2_trigger_reason_mask"),
        "rti2_budget_skipped": payload.get("rti2_budget_skipped"),
        "r1_status": payload.get("r1_status"),
        "r2_status": payload.get("r2_status"),
        "r1_nonlinear_objective": payload.get("r1_nonlinear_objective"),
        "r2_nonlinear_objective": payload.get("r2_nonlinear_objective"),
        "r1_min_corridor_slack": payload.get("r1_min_corridor_slack"),
        "r2_min_corridor_slack": payload.get("r2_min_corridor_slack"),
        "r1_solver_iterations": payload.get("r1_solver_iterations"),
        "r2_solver_iterations": payload.get("r2_solver_iterations"),
        "r1_solve_us": payload.get("r1_solve_us"),
        "r2_solve_us": payload.get("r2_solve_us"),
        "total_rti_us": payload.get("total_rti_us", solver.get("solve_us")),
        "selected_candidate": payload.get("selected_candidate"),
        "rho_start": solver.get("rho_start"),
        "rho_final": solver.get("rho_final"),
        "rho_u_start": solver.get("rho_u_start"),
        "rho_u_final": solver.get("rho_u_final"),
        "rho_change_count": solver.get("rho_change_count"),
        "factorization_count": solver.get("factorization_count"),
        "factorization_time_ns": solver.get("factorization_time_ns"),
        "operating_mode": mode,
        "operating_mode_source": operating_mode_source,
        "steering_reversal": reversal,
    })
    primal = _finite_number(flat["primal_residual"])
    dual = _finite_number(flat["dual_residual"])
    flat["max_residual"] = max(primal, dual) if primal is not None and dual is not None else None
    return flat


def rejection_category(row: dict[str, Any]) -> str:
    status = str(row.get("cycle_status") or "unknown").lower()
    if status.startswith("accepted"):
        return "accepted"
    if status in {"rejected_input", "rejected_state", "rejected_projection"}:
        return "invalid_input_or_state"
    if status == "rejected_residual":
        return "solver_residual"
    if status == "rejected_solver":
        return "solver_failure"
    if status == "rejected_regularization":
        return "regularization_limit"
    if status == "rejected_nonlinear_rollout":
        reason = str(row.get("nonlinear_failure_reason") or "unknown").lower()
        return f"nonlinear_{reason}"
    if status == "rejected":
        reason = str(row.get("reason") or "").lower()
        if "state unavailable" in reason or "map_pose" in reason:
            return "missing_synchronized_state"
        if reason:
            return "input_gate:" + "_".join(reason.split())
    if status.startswith("rejected_"):
        return status.removeprefix("rejected_")
    return "unclassified_status"


def _numeric_bin(value: Any, cuts: Iterable[float], labels: list[str]) -> str:
    number = _finite_number(value)
    if number is None:
        return "missing"
    for limit, label in zip(cuts, labels):
        if number < limit:
            return label
    return labels[-1]


def _strata(row: dict[str, Any]) -> dict[str, str]:
    return {
        "speed_mps": _numeric_bin(row.get("u_mps"), [2, 4, 6, 8],
                                  ["<2", "2-4", "4-6", "6-8", ">=8"]),
        "abs_curvature_per_m": _numeric_bin(row.get("abs_curvature_per_m"),
            [0.05, 0.10, 0.20, 0.40], ["<0.05", "0.05-0.10", "0.10-0.20",
                                     "0.20-0.40", ">=0.40"]),
        "abs_steering_rad": _numeric_bin(row.get("abs_steering_rad"),
            [0.05, 0.10, 0.20, 0.35], ["<0.05", "0.05-0.10", "0.10-0.20",
                                      "0.20-0.35", ">=0.35"]),
        "abs_steering_rate_radps": _numeric_bin(row.get("abs_steering_rate_radps"),
            [0.25, 0.75, 1.5, 2.5], ["<0.25", "0.25-0.75", "0.75-1.5",
                                    "1.5-2.5", ">=2.5"]),
        "abs_yaw_rate_radps": _numeric_bin(row.get("abs_yaw_rate_radps"),
            [0.25, 0.5, 1.0, 2.0], ["<0.25", "0.25-0.5", "0.5-1.0",
                                   "1.0-2.0", ">=2.0"]),
        "lateral_accel_proxy_mps2": _numeric_bin(row.get("lateral_accel_proxy_mps2"),
            [1, 2, 4, 8], ["<1", "1-2", "2-4", "4-8", ">=8"]),
        "operating_mode": str(row.get("operating_mode") or "unknown"),
        "steering_reversal": str(row.get("steering_reversal")
                                 if row.get("steering_reversal") is not None
                                 else "unknown"),
        "source_age_ms": _numeric_bin(
            (_finite_number(row.get("state_age_s")) or 0.0) * 1000.0
            if row.get("state_age_s") is not None else None,
            [25, 40, 60, 90, 120],
            ["0-25", "25-40", "40-60", "60-90", "90-120", ">120"]),
        "progress_schedule_error_m": _numeric_bin(
            row.get("nominal_vs_candidate_progress_error_max_m"),
            [0.01, 0.025, 0.05, 0.10], ["<0.01", "0.01-0.025",
                "0.025-0.05", "0.05-0.10", ">=0.10"]),
        "curvature_schedule_error_per_m": _numeric_bin(
            row.get("nominal_vs_candidate_curvature_error_max_per_m"),
            [0.01, 0.025, 0.05, 0.10], ["<0.01", "0.01-0.025",
                "0.025-0.05", "0.05-0.10", ">=0.10"]),
        "corridor_slack_m": _numeric_bin(row.get("minimum_predicted_corridor_slack_m"),
            [0, 0.10, 0.25, 0.50], ["<0", "0-0.10", "0.10-0.25",
                                    "0.25-0.50", ">=0.50"]),
        "nonsmooth_columns": _numeric_bin(row.get("nonsmooth_jacobian_columns"),
            [10, 25, 50], ["<10", "10-25", "25-50", ">=50"]),
        "solver_iterations": _numeric_bin(row.get("solver_iterations"),
            [25, 50, 75, 100], ["<25", "25-50", "50-75", "75-100", ">=100"]),
        "rho_changes": _numeric_bin(row.get("rho_change_count"),
            [1, 2, 4], ["0", "1", "2-3", ">=4"]),
    }


@lru_cache(maxsize=8)
def _load_path_table(path_text: str) -> tuple[list[tuple[float, float]], float] | None:
    points: list[tuple[float, float, float, float]] = []
    try:
        with Path(path_text).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                fields = next(csv.reader([line]))
                if len(fields) < 5:
                    continue
                vals = [float(fields[index]) for index in (0, 1, 2, 4)]
                if all(math.isfinite(value) for value in vals):
                    points.append((vals[0], vals[1], vals[2], vals[3]))
    except (OSError, ValueError, csv.Error):
        return None
    if len(points) < 3:
        return None
    closing = math.hypot(points[0][1] - points[-1][1],
                         points[0][2] - points[-1][2])
    if closing < 1.0e-4 and len(points) > 3:
        points.pop()
    lap_length = points[-1][0] - points[0][0] + math.hypot(
        points[0][1] - points[-1][1], points[0][2] - points[-1][2])
    if lap_length <= 0.0:
        return None
    return [(point[0], point[3]) for point in points], lap_length


def _read_path_curve(path: Path, progress_m: Any) -> float | None:
    progress = _finite_number(progress_m)
    if progress is None:
        return None
    loaded = _load_path_table(str(path.resolve()))
    if loaded is None:
        return None
    points, lap_length = loaded
    wrapped = (progress - points[0][0]) % lap_length + points[0][0]
    for index, lower in enumerate(points):
        upper = points[index + 1] if index + 1 < len(points) else points[0]
        s0 = lower[0]
        s1 = upper[0] if index + 1 < len(points) else points[0][0] + lap_length
        query = wrapped if wrapped >= s0 else wrapped + lap_length
        if query <= s1:
            fraction = max(0.0, min(1.0, (query - s0) / (s1 - s0)))
            return lower[1] + fraction * (upper[1] - lower[1])
    return points[0][1]


def read_cycles(path: Path, trajectory_path: Path | None = None) -> tuple[list[dict[str, Any]], int]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    malformed = 0
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        for event in csv.DictReader(stream):
            topic = event.get("topic", "")
            if "mpc" not in topic and "diagnostics_json" not in event:
                continue
            if topic and not (topic.endswith("/mpc/diagnostics") or
                              topic.endswith("/mpc_shadow/diagnostics")):
                continue
            payload = _nested_payload(event.get("payload_json", ""))
            if payload is None and event.get("diagnostics_json"):
                payload = _nested_payload(event["diagnostics_json"])
            if payload is None or "status" not in payload:
                malformed += 1
                continue
            curve = (_read_path_curve(trajectory_path, payload.get("progress_m"))
                     if trajectory_path else None)
            rows.append(flatten_diagnostic(event, payload, curve))
    return rows, malformed


def write_outputs(rows: list[dict[str, Any]], malformed: int, output_dir: Path,
                  source_path: Path | None = None,
                  trajectory_path: Path | None = None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mpc_cycles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CYCLE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    categories = Counter(rejection_category(row) for row in rows)
    rejected = [row for row in rows if rejection_category(row) != "accepted"]
    classified = [row for row in rejected
                  if rejection_category(row) != "unclassified_status"]
    with (output_dir / "rejection_categories.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "count", "fraction_of_rejected_pct"])
        for category, count in categories.most_common():
            if category == "accepted":
                continue
            writer.writerow([category, count,
                100.0 * count / len(rejected) if rejected else 0.0])

    strata_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    feature_coverage: Counter[str] = Counter()
    for row in rows:
        accepted = rejection_category(row) == "accepted"
        for feature, label in _strata(row).items():
            feature_coverage[feature] += label != "missing"
            count = strata_counts[(feature, label)]
            count["total"] += 1
            count["accepted" if accepted else "rejected"] += 1
    with (output_dir / "stratified_rates.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", "bin", "cycles", "accepted", "rejected",
                         "rejection_rate_pct", "feature_coverage_pct"])
        for (feature, label), count in sorted(strata_counts.items()):
            writer.writerow([feature, label, count["total"], count["accepted"],
                count["rejected"],
                (100.0 * count["rejected"] / count["total"]
                 if count["total"] and label != "missing" else ""),
                100.0 * feature_coverage[feature] / len(rows) if rows else 0.0])

    coverage = 100.0 * len(classified) / len(rejected) if rejected else 100.0
    summary = {
        "cycles": len(rows),
        "accepted": categories.get("accepted", 0),
        "rejected": len(rejected),
        "malformed_diagnostic_events": malformed,
        "rejection_classification_coverage_pct": coverage,
        "meets_95pct_classification_gate": coverage >= 95.0,
        "rejection_categories": {key: val for key, val in categories.items()
                                 if key != "accepted"},
        "accepted_status_count": categories.get("accepted", 0),
        "input_trace": str(source_path) if source_path else None,
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
        "trajectory_sha256": (hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
                              if trajectory_path else None),
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frequent = sorted(((key, val) for key, val in categories.items()
                       if key != "accepted"), key=lambda item: (-item[1], item[0]))
    lines = ["# MPC live rejection analysis", "",
        f"- Diagnostic cycles: {len(rows)}",
        f"- Accepted: {summary['accepted']}",
        f"- Rejected: {len(rejected)}",
        f"- Malformed diagnostic events: {malformed}",
        f"- Rejection classification coverage: {coverage:.1f}% "
        f"({'meets' if coverage >= 95.0 else 'does not meet'} the 95% gate)", "",
        "Missing-feature bins are reported as coverage gaps, not physical "
        "operating regimes.", "",
        "## Rejection categories", ""]
    if frequent:
        lines += [f"- `{name}`: {count} "
                  f"({100.0 * count / len(rejected):.1f}% of rejected)"
                  for name, count in frequent]
    else:
        lines.append("No rejected MPC diagnostic cycles were found.")
    lines += ["", "## Interpretation", "",
        "Rates are associations within recorded diagnostic cycles, not causal "
        "claims. Source/command timing is reported as data and is never used "
        "to reject or discard a cycle. Simulator truth is not read.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="recorder controller_trace.csv[.gz]")
    parser.add_argument("output_dir", type=Path,
                        help="directory for cycle/category/rate CSV and reports")
    parser.add_argument("--trajectory", type=Path,
                        help="optional exact run raceline for backfilling curvature "
                             "when older diagnostics omit it; use the run's recorded path")
    args = parser.parse_args()
    rows, malformed = read_cycles(args.input, args.trajectory)
    if not rows:
        parser.error("no MPC diagnostic events found; include /mpc/diagnostics "
                     "or /mpc_shadow/diagnostics in the recorded controller trace")
    summary = write_outputs(rows, malformed, args.output_dir,
                            args.input, args.trajectory)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
