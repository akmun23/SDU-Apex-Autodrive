#!/usr/bin/env python3
"""Offline geometry and source-plant feasibility audit for an MPC raceline.

The static part follows the production trajectory's periodic, piecewise-linear
arc-length interpolation.  The dynamic part replays full-horizon plans already
captured by the MPC diagnostic stream through the source-plant C ABI, including
the optimizer's command-epoch input semantics, 25 ms command stages, 1 ms
plant substeps, and Frenet tracking update used by production. It writes a
row-wise witness so
the replay is reproducible.

This is a diagnostic, not a global infeasibility proof: a failed local MPC
solve is classified as such and does not imply that no feasible control
sequence exists.  Unity truth is never used as a rollout input.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import ctypes
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

try:
    from tools import evaluate_mpc_bag as evaluator
    from tools import vehicle_footprint
except ModuleNotFoundError:
    import evaluate_mpc_bag as evaluator
    import vehicle_footprint


SCHEMA = "sdu_apex_mpc_trajectory_feasibility_audit/v1"
WHEELBASE_M = 0.324
STEERING_LIMIT_RAD = math.pi / 6.0
STEERING_RATE_RADPS = math.radians(183.345993)
VEHICLE_WIDTH_M = vehicle_footprint.WIDTH_M
REAR_AXLE_TO_FRONT_M = vehicle_footprint.BODY_X[1]
REAR_OVERHANG_M = -vehicle_footprint.BODY_X[0]
COM_FORWARD_FROM_REAR_M = 0.15532
MIN_POSITIVE_THROTTLE = 0.001


def read_mpc_command_delay_steps() -> int:
    """Read the production optimizer's compile-time command-delay setting."""
    header = (Path(__file__).resolve().parents[1] /
              "f1tenth_mpc/include/mpc_types.h")
    match = re.search(
        r"^#define\s+MPC_COMMAND_DELAY_STEPS\s+(\d+)\b",
        header.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"cannot read MPC_COMMAND_DELAY_STEPS from {header}")
    return int(match.group(1))


MPC_COMMAND_DELAY_STEPS = read_mpc_command_delay_steps()


def source_rollout_command(control: list[float]) -> tuple[float, float]:
    """Return the optimizer-stage command applied at the command epoch."""
    if MPC_COMMAND_DELAY_STEPS != 0:
        raise RuntimeError(
            "source rollout audit supports only the production direct-input "
            "horizon; update command chronology before changing the MPC delay")
    return float(control[0]), float(control[1])
MPC_FULL_HORIZON_STATUSES = {
    "accepted_optimal", "accepted_feasible", "accepted_line_search",
    "accepted_viability_recovery",
}
PUBLISHED_CONTROL_STATUSES = {
    "accepted_candidate_prefix", "accepted_prior_plan",
}
CANDIDATE_PRIOR_FIELDS = (
    "bag_time_s", "status", "candidate_prefix_chain_length", "progress_m",
    "prior_snapshot_source", "candidate_horizon",
    "candidate_first_corridor_violation_stage",
    "candidate_minimum_physical_corridor_slack_m", "prior_cursor",
    "prior_suffix_start_index", "prior_remaining_horizon",
    "prior_first_corridor_violation_stage",
    "prior_minimum_physical_corridor_slack_m",
    "candidate_issued_action_steering_rad",
    "candidate_issued_action_throttle", "prior_action_steering_rad",
    "prior_action_throttle", "absolute_first_steering_difference_rad",
    "absolute_first_throttle_difference", "candidate_solver_primal_residual",
    "candidate_solver_dual_residual",
    "candidate_survival_steps_before_violation",
    "prior_survival_steps_before_violation", "prior_survives_longer",
    "prior_valid_for_full_remaining_suffix",
    "prefix_comparison_horizon_steps",
    "candidate_prefix_source_plant_footprint_valid",
    "prior_prefix_source_plant_footprint_valid",
    "candidate_prefix_minimum_physical_corridor_slack_m",
    "prior_prefix_minimum_physical_corridor_slack_m",
    "candidate_prefix_speed_error_rmse_mps",
    "candidate_prefix_speed_error_bias_mps",
    "candidate_prefix_speed_error_max_abs_mps",
    "prior_prefix_speed_error_rmse_mps",
    "prior_prefix_speed_error_bias_mps",
    "prior_prefix_speed_error_max_abs_mps",
    "candidate_prefix_objective", "prior_prefix_objective",
    "candidate_prefix_objective_terms_json",
    "prior_prefix_objective_terms_json",
)
BLEND_CONTINUATION_FIELDS = (
    "bag_time_s", "progress_m", "prior_remaining_horizon",
    "candidate_weight", "command_bounds_valid", "source_steps",
    "first_corridor_violation_stage", "minimum_corridor_slack_m",
    "speed_error_rmse_mps", "objective_total", "full_horizon_source_valid",
    "first_steering_rad", "first_throttle", "controls_json",
)
DYNAMIC_FEASIBILITY_BIN_FIELDS = (
    "progress_start_m", "progress_end_m", "saved_state_count",
    "blend_trial_count", "source_valid_trial_count",
    "strict_positive_slack_witness_count",
    "by_candidate_weight_valid_counts_json",
    "minimum_positive_witness_slack_m", "maximum_positive_witness_slack_m",
    "classification",
)


def summarize_dynamic_feasibility_by_progress(
        comparisons: list[dict[str, Any]], bin_width_m: float = 0.5,
        blend_weights: tuple[float, ...] = (0.25, 0.50, 0.75)
        ) -> list[dict[str, Any]]:
    """Aggregate tested source-plant witnesses by unwrapped start progress.

    A bin with no positive-slack witness means only that the finite recorded
    candidate/prior blend set did not produce one at its sampled states. It is
    not evidence of global physical infeasibility.
    """
    if not math.isfinite(bin_width_m) or bin_width_m <= 0.0:
        raise ValueError("dynamic feasibility bin width must be positive")
    buckets: dict[int, dict[str, Any]] = {}
    for comparison in comparisons:
        progress = float(comparison["progress_m"])
        if not math.isfinite(progress):
            raise ValueError("comparison progress must be finite")
        index = math.floor(progress / bin_width_m)
        bucket = buckets.setdefault(index, {
            "states": set(), "trials": [],
        })
        bucket["states"].add(progress)
        bucket["trials"].extend(
            comparison.get("blended_continuation_trials", []))

    rows = []
    for index, bucket in sorted(buckets.items()):
        trials = bucket["trials"]
        valid = [trial for trial in trials
                 if trial.get("full_horizon_source_valid")]
        positive = [trial for trial in valid
                    if trial.get("minimum_corridor_slack_m") is not None
                    and float(trial["minimum_corridor_slack_m"]) > 0.0]
        by_weight = {
            f"{weight:.2f}": sum(
                trial.get("candidate_weight") == weight
                and trial.get("full_horizon_source_valid")
                for trial in trials)
            for weight in blend_weights
        }
        rows.append({
            "progress_start_m": index * bin_width_m,
            "progress_end_m": (index + 1) * bin_width_m,
            "saved_state_count": len(bucket["states"]),
            "blend_trial_count": len(trials),
            "source_valid_trial_count": len(valid),
            "strict_positive_slack_witness_count": len(positive),
            "by_candidate_weight_valid_counts": by_weight,
            "minimum_positive_witness_slack_m": min((
                float(trial["minimum_corridor_slack_m"])
                for trial in positive), default=None),
            "maximum_positive_witness_slack_m": max((
                float(trial["minimum_corridor_slack_m"])
                for trial in positive), default=None),
            "classification": (
                "tested_source_plant_feasible_witness_found"
                if positive else
                "no_positive_witness_in_tested_blends_not_global_infeasibility"),
        })
    return rows


def blend_candidate_and_prior_controls(
    candidate: list[list[float]], prior_suffix: list[list[float]],
    candidate_weight: float,
) -> list[list[float]]:
    """Build an offline continuation trial, holding the prior tail if needed.

    This is only a deterministic interpolation experiment. It does not make a
    blended sequence a certified plan and it must never be used for runtime
    publication without a new exact validation from the current state.
    """
    if len(candidate) != 30 or not 1 <= len(prior_suffix) <= 30:
        raise ValueError("blend requires a 30-step candidate and prior suffix")
    if not math.isfinite(candidate_weight) or not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate weight must be finite and in [0, 1]")
    if any(not isinstance(control, list) or len(control) != 2
           for control in candidate + prior_suffix):
        raise ValueError("malformed steering/throttle control pair")
    prior_tail = prior_suffix[-1]
    output: list[list[float]] = []
    for stage, candidate_control in enumerate(candidate):
        prior_control = (prior_suffix[stage]
                         if stage < len(prior_suffix) else prior_tail)
        output.append([
            candidate_weight * float(candidate_control[index]) +
            (1.0 - candidate_weight) * float(prior_control[index])
            for index in range(2)
        ])
    return output


def validate_blend_weight_grid(values: list[float]) -> tuple[float, ...]:
    """Validate a deterministic grid for offline candidate/prior blends."""
    if not values or any(not math.isfinite(value) or value < 0.0 or value > 1.0
                         for value in values):
        raise ValueError("blend weights must be a nonempty grid in [0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("blend weights must not contain duplicates")
    return tuple(sorted(values))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_runtime_corridor_profile(path: Path) -> dict[str, float]:
    """Read runtime corridor settings without silently using C defaults."""
    text = path.read_text()
    result: dict[str, float] = {}
    for key in ("corridor_margin_m", "first_prediction_corridor_margin_m",
                "nonlinear_corridor_tolerance_m",
                "corridor_preview_halfwidth_m"):
        matches = re.findall(
            rf"^\s*{re.escape(key)}\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(?:#.*)?$",
            text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one numeric {key} in {path}, found {len(matches)}")
        result[key] = float(matches[0])
    return result


def read_runtime_objective_profile(path: Path) -> dict[str, float]:
    """Read the scalar nonlinear-validation objective weights from runtime YAML."""
    keys = (
        "weight_e_y", "weight_e_psi", "weight_u", "weight_u_overspeed",
        "weight_longitudinal_slip", "weight_v", "weight_r",
        "weight_steering_command", "weight_steering_target",
        "weight_throttle_command", "weight_steering_slew",
        "weight_throttle_slew", "terminal_multiplier", "max_steering_rad",
    )
    text = path.read_text()
    result: dict[str, float] = {}
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    for key in keys:
        matches = re.findall(
            rf"^\s*{re.escape(key)}\s*:\s*{number}\s*(?:#.*)?$",
            text, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one numeric {key} in {path}, found {len(matches)}")
        result[key] = float(matches[0])
    return result


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def metric(values: list[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return {"n": 0, "min": None, "max": None, "p95_abs": None,
                "rmse": None}
    absolute = sorted(abs(value) for value in finite)
    return {
        "n": len(finite),
        "min": min(finite),
        "max": max(finite),
        "max_abs": max(abs(value) for value in finite),
        "p95_abs": absolute[min(len(absolute) - 1,
                                math.ceil(0.95 * len(absolute)) - 1)],
        "rmse": math.sqrt(statistics.mean(value * value for value in finite)),
    }


def load_trajectory(path: Path) -> tuple[list[dict[str, float]], float]:
    points: list[dict[str, float]] = []
    with path.open(newline="") as stream:
        rows = csv.reader(line for line in stream
                          if line.strip() and not line.lstrip().startswith("#"))
        for row in rows:
            if len(row) < 9:
                continue
            points.append(dict(zip(
                ("s", "x", "y", "heading", "curvature", "speed",
                 "acceleration", "left", "right"),
                map(float, row[:9]),
            )))
    if len(points) < 3:
        raise ValueError(f"trajectory has fewer than three points: {path}")
    if len(points) > 3 and math.hypot(
            points[0]["x"] - points[-1]["x"],
            points[0]["y"] - points[-1]["y"]) < 1.0e-4:
        # Match mpc_trajectory_prepare(): a repeated closing row is removed,
        # then the true final-to-first closing segment defines lap length.
        points.pop()
    if any(not all(math.isfinite(value) for value in row.values())
           for row in points):
        raise ValueError("trajectory contains a non-finite value")
    if any(points[i + 1]["s"] <= points[i]["s"]
           for i in range(len(points) - 1)):
        raise ValueError("trajectory arc length must increase strictly")
    if any(point["speed"] < 0.0 or point["left"] <= 0.0 or
           point["right"] <= 0.0 for point in points):
        raise ValueError("trajectory speed/bounds are outside the MPC contract")
    close = math.hypot(points[0]["x"] - points[-1]["x"],
                       points[0]["y"] - points[-1]["y"])
    length = points[-1]["s"] - points[0]["s"] + close
    if not math.isfinite(length) or length <= points[-1]["s"] - points[0]["s"]:
        raise ValueError("trajectory does not have a valid closing segment")
    return points, length


def sample_path(points: list[dict[str, float]], length: float,
                progress: float) -> dict[str, float]:
    first_s = points[0]["s"]
    wrapped_s = (progress - first_s) % length + first_s
    stamps = [point["s"] for point in points]
    upper = bisect.bisect_right(stamps, wrapped_s)
    if upper == 0:
        lower, upper = len(points) - 1, 0
        s0, s1 = points[lower]["s"], points[0]["s"] + length
        query = wrapped_s + length
    elif upper >= len(points):
        lower, upper = len(points) - 1, 0
        s0, s1, query = points[lower]["s"], points[0]["s"] + length, wrapped_s
    else:
        lower, s0 = upper - 1, points[upper - 1]["s"]
        s1, query = points[upper]["s"], wrapped_s
    span = s1 - s0
    ratio = max(0.0, min(1.0, (query - s0) / span))
    a, b = points[lower], points[upper]
    b_heading = a["heading"] + wrap(b["heading"] - a["heading"])
    result = {key: a[key] + ratio * (b[key] - a[key])
              for key in ("x", "y", "curvature", "speed", "acceleration",
                          "left", "right")}
    result["heading"] = a["heading"] + ratio * (b_heading - a["heading"])
    result["s"] = wrapped_s
    return result


def sample_corridor_bounds(points: list[dict[str, float]], length: float,
                           progress: float,
                           preview_halfwidth_m: float) -> tuple[float, float]:
    """Match mpc_rti.c's periodic minimum-bound preview exactly.

    Bounds are piecewise linear, so the minimum over the preview interval is
    attained at an endpoint or at a trajectory knot. Evaluate the endpoints,
    then include every knot whose periodic copy lies within the closed window.
    """
    if not math.isfinite(preview_halfwidth_m) or preview_halfwidth_m < 0.0:
        raise ValueError("corridor preview halfwidth must be finite and >= 0")
    center = sample_path(points, length, progress)["s"]
    lower = sample_path(points, length, center - preview_halfwidth_m)
    upper = sample_path(points, length, center + preview_halfwidth_m)
    left_values = [lower["left"], upper["left"]]
    right_values = [lower["right"], upper["right"]]
    if preview_halfwidth_m > 0.0:
        half_lap = 0.5 * length
        for point in points:
            delta = (point["s"] - center + half_lap) % length - half_lap
            if abs(delta) <= preview_halfwidth_m:
                left_values.append(point["left"])
                right_values.append(point["right"])
    return min(left_values), min(right_values)


def footprint_extents(e_psi: float) -> tuple[float, float]:
    width = 0.5 * VEHICLE_WIDTH_M * abs(math.cos(e_psi))
    sine = math.sin(e_psi)
    return (width + max(REAR_AXLE_TO_FRONT_M * sine, -REAR_OVERHANG_M * sine),
            width + max(-REAR_AXLE_TO_FRONT_M * sine, REAR_OVERHANG_M * sine))


def geometry_audit(points: list[dict[str, float]], length: float,
                   bin_width_m: float, wall_clearance_m: float,
                   tolerance_m: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    step_m = min(0.05, bin_width_m / 4.0)
    count = math.ceil(length / step_m)
    samples: list[dict[str, float]] = []
    for i in range(count):
        s = points[0]["s"] + i * length / count
        item = sample_path(points, length, s)
        item["steering_seed"] = math.atan(WHEELBASE_M * item["curvature"])
        item["yaw_rate_demand"] = item["speed"] * item["curvature"]
        item["lateral_accel_demand"] = item["speed"] ** 2 * item["curvature"]
        item["aligned_footprint_extent"] = footprint_extents(0.0)[0]
        item["left_slack"] = (item["left"] - wall_clearance_m -
                               item["aligned_footprint_extent"])
        item["right_slack"] = (item["right"] - wall_clearance_m -
                                item["aligned_footprint_extent"])
        samples.append(item)

    segment_heading_errors: list[float] = []
    kinematic_accel_errors: list[float] = []
    steering_rates: list[float] = []
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        ds = ((nxt["s"] - point["s"]) if i + 1 < len(points) else
              length - (point["s"] - points[0]["s"]))
        delta_heading = wrap(nxt["heading"] - point["heading"])
        segment_heading_errors.append(delta_heading / ds - point["curvature"])
        kinematic_acceleration = point["speed"] * (nxt["speed"] - point["speed"]) / ds
        kinematic_accel_errors.append(kinematic_acceleration - point["acceleration"])
        current_steering = math.atan(WHEELBASE_M * point["curvature"])
        next_steering = math.atan(WHEELBASE_M * nxt["curvature"])
        dt = ds / max(0.01, 0.5 * (point["speed"] + nxt["speed"]))
        steering_rates.append((next_steering - current_steering) /
                              max(dt, 1.0e-9))

    bins: list[dict[str, Any]] = []
    for lower in range(0, math.ceil(length / bin_width_m)):
        s0 = points[0]["s"] + lower * bin_width_m
        s1 = min(points[0]["s"] + length, s0 + bin_width_m)
        selected = [row for row in samples if s0 <= row["s"] < s1]
        if not selected:
            continue
        bins.append({
            "s_start_m": s0,
            "s_end_m": s1,
            "sample_count": len(selected),
            "speed_max_mps": max(row["speed"] for row in selected),
            "abs_curvature_max_per_m": max(abs(row["curvature"])
                                             for row in selected),
            "abs_yaw_rate_demand_max_radps": max(abs(row["yaw_rate_demand"])
                                                   for row in selected),
            "abs_lateral_accel_demand_max_mps2": max(
                abs(row["lateral_accel_demand"]) for row in selected),
            "steering_seed_abs_max_rad": max(abs(row["steering_seed"])
                                              for row in selected),
            "left_slack_min_m": min(row["left_slack"] for row in selected),
            "right_slack_min_m": min(row["right_slack"] for row in selected),
            "speed_acceleration_min_mps2": min(row["acceleration"]
                                                 for row in selected),
            "speed_acceleration_max_mps2": max(row["acceleration"]
                                                 for row in selected),
        })

    close = math.hypot(points[0]["x"] - points[-1]["x"],
                       points[0]["y"] - points[-1]["y"])
    seam_ds = length - (points[-1]["s"] - points[0]["s"])
    seam_tangent = math.atan2(points[0]["y"] - points[-1]["y"],
                              points[0]["x"] - points[-1]["x"])
    seam = {
        "closing_segment_length_m": seam_ds,
        "endpoint_gap_m": close,
        "heading_last_to_first_delta_rad": wrap(
            points[0]["heading"] - points[-1]["heading"]),
        "last_heading_to_closing_chord_rad": wrap(
            seam_tangent - points[-1]["heading"]),
        "closing_chord_to_first_heading_rad": wrap(
            points[0]["heading"] - seam_tangent),
        "curvature_first_minus_last_per_m": (
            points[0]["curvature"] - points[-1]["curvature"]),
        "speed_first_minus_last_mps": points[0]["speed"] - points[-1]["speed"],
        "acceleration_first_minus_last_mps2": (
            points[0]["acceleration"] - points[-1]["acceleration"]),
        "left_bound_first_minus_last_m": points[0]["left"] - points[-1]["left"],
        "right_bound_first_minus_last_m": points[0]["right"] - points[-1]["right"],
    }
    all_slacks = [min(row["left_slack"], row["right_slack"])
                  for row in samples]
    steering_seed_max = max(abs(math.atan(WHEELBASE_M * p["curvature"]))
                            for p in points)
    steering_slew_abs = [abs(value) for value in steering_rates]
    summary = {
        "point_count": len(points),
        "lap_length_m": length,
        "sample_spacing_m": length / count,
        "seam": seam,
        "stored_curvature_vs_heading_secant_error_per_m": metric(
            segment_heading_errors),
        "kinematic_acceleration_minus_csv_acceleration_mps2": metric(
            kinematic_accel_errors),
        "speed_max_mps": max(row["speed"] for row in samples),
        "abs_curvature_max_per_m": max(abs(row["curvature"]) for row in samples),
        "abs_yaw_rate_demand_max_radps": max(
            abs(row["yaw_rate_demand"]) for row in samples),
        "abs_lateral_accel_demand_max_mps2": max(
            abs(row["lateral_accel_demand"]) for row in samples),
        "geometric_steering_seed_max_rad": steering_seed_max,
        "steering_limit_rad": STEERING_LIMIT_RAD,
        "steering_seed_saturation_points": sum(
            abs(math.atan(WHEELBASE_M * p["curvature"])) > STEERING_LIMIT_RAD
            for p in points),
        "geometric_steering_seed_slew_radps": metric(steering_rates),
        "geometric_steering_seed_slew_limit_exceedances": sum(
            abs(value) > STEERING_RATE_RADPS for value in steering_rates),
        "source_steering_rate_limit_radps": STEERING_RATE_RADPS,
        "minimum_aligned_footprint_clearance_m": min(
            min(row["left"], row["right"]) - footprint_extents(0.0)[0]
            for row in samples),
        "wall_clearance_requirement_m": wall_clearance_m,
        "minimum_centerline_corridor_slack_after_footprint_m": min(all_slacks),
        "geometric_centerline_feasible_at_requested_clearance":
            min(all_slacks) >= -tolerance_m,
        "heading_aware_envelope_formula": (
            "left/right support of four rear-axle-origin rectangle corners; "
            f"front={REAR_AXLE_TO_FRONT_M:.3f} m, rear={REAR_OVERHANG_M:.3f} m, "
            f"width={VEHICLE_WIDTH_M:.3f} m"),
        "bin_width_m": bin_width_m,
    }
    return summary, bins


def _plant_state_from_diag(diag: dict[str, Any]) -> tuple[list[float], float, int]:
    state = diag.get("state")
    hidden = diag.get("plant_internal_state")
    flags = diag.get("plant_state_flags")
    if not isinstance(state, list) or len(state) != 10:
        raise ValueError("diagnostic state must contain 10 values")
    if not isinstance(hidden, list) or len(hidden) != 22:
        raise ValueError("diagnostic hidden plant state must contain 22 values")
    if not isinstance(flags, list) or len(flags) != 3:
        raise ValueError("diagnostic plant state flags must contain 3 values")
    # Source plant order: u, COM-v, r, wheel omega[4], sprung[6], steering,
    # jounce[4], low-forward timers[4], low-side timers[4]. The MPC carries the
    # last 22 values in its compact internal-state order.
    source = ([float(state[2]),
               float(state[3]) + float(state[4]) * COM_FORWARD_FROM_REAR_M,
               float(state[4])]
              + list(map(float, hidden[0:10]))
              + [float(state[7])]
              + list(map(float, hidden[10:22])))
    return source, float(state[0]), int(flags[1])


def _source_rollout(diag: dict[str, Any], points: list[dict[str, float]],
                    length: float, step_function: Any,
                    output_csv: csv.writer,
                    wall_clearance_m: float,
                    first_prediction_clearance_m: float,
                    corridor_preview_halfwidth_m: float,
                    corridor_tolerance_m: float,
                    throttle_override: float | None = None,
                    scenario_label: str | None = None,
                    objective_profile: dict[str, float] | None = None,
                    stop_at_corridor_violation: bool = True,
                    collect_stage_trace: bool = False) -> dict[str, Any]:
    state, e_y, suspension_initialized = _plant_state_from_diag(diag)
    e_psi = float(diag["state"][1])
    progress = float(diag["progress_m"])
    # The controller logs both the shifted nominal warm-start and the current
    # solver candidate. Only candidate_controls is the sequence whose exact
    # nonlinear rollout was accepted on a full-horizon status.
    controls = diag.get("candidate_controls")
    if not isinstance(controls, list) or not 1 <= len(controls) <= 30:
        raise ValueError("plan must contain 1 to 30 controls")
    if any(not isinstance(control, list) or len(control) != 2
           for control in controls):
        raise ValueError("malformed nominal control sequence")
    if MPC_COMMAND_DELAY_STEPS != 0:
        raise RuntimeError(
            "source rollout audit must be updated before evaluating a "
            "nonzero MPC_COMMAND_DELAY_STEPS")

    delayed_steer_1 = float(diag["state"][5])
    delayed_steer_2 = float(diag["state"][6])
    delayed_q_1 = float(diag["state"][8])
    delayed_q_2 = float(diag["state"][9])
    wheel_angles = [0.0] * 4
    errors: list[float] = []
    clearance_slack: list[float] = []
    speed_errors: list[float] = []
    throttle_values: list[float] = []
    peak_lateral_accel: list[float] = []
    longitudinal_accels: list[float] = []
    u_interval_accels: list[float] = []
    peak_abs_slip: list[float] = []
    speed_accel_errors: list[float] = []
    rear_lateral_speeds: list[float] = []
    yaw_rates: list[float] = []
    actual_steering: list[float] = []
    steering_commands: list[float] = []
    steering_rate_25ms: list[float] = []
    wheel_omega: list[float] = []
    normal_loads: list[float] = []
    previous_actual_steering: float | None = None
    first_corridor_violation_stage: int | None = None
    replay_states: dict[int, list[float]] = {}
    stage_trace: list[dict[str, float | int]] = []
    predictions = diag.get("predictions", [])
    prediction_by_stage = {
        int(item["n"]): item for item in predictions
        if isinstance(item, dict) and isinstance(item.get("n"), int)
    }
    failed_step = None
    objective_terms = {
        "cross_track": 0.0, "heading": 0.0, "speed": 0.0,
        "lateral_speed": 0.0, "yaw_rate": 0.0,
        "steering_state": 0.0, "slip": 0.0,
        "steering_effort": 0.0, "throttle_effort": 0.0,
        "steering_slew": 0.0, "throttle_slew": 0.0,
    }
    objective_speed_errors: list[float] = []
    objective_complete = objective_profile is not None
    active_speed_ceiling = float(diag.get("active_speed_ceiling_mps", 16.0))
    previous_objective_steering = float(diag["state"][6])
    previous_objective_throttle = float(diag["state"][9])

    def add_state_objective(stage: int, at_progress: float, ey: float,
                            epsi: float, speed: float, lateral_speed: float,
                            yaw_rate: float, steering_command: float,
                            actual_steering: float,
                            incoming_slip_excess: list[float] | None) -> None:
        if objective_profile is None:
            return
        reference = sample_path(points, length, at_progress)
        target_speed = (speed if stage == 0 else
                        min(reference["speed"], active_speed_ceiling))
        target_yaw_rate = reference["curvature"] * target_speed
        target_steering = max(-objective_profile["max_steering_rad"],
                              min(objective_profile["max_steering_rad"],
                                  math.atan(WHEELBASE_M *
                                            reference["curvature"])))
        scale = (objective_profile["terminal_multiplier"]
                 if stage == len(controls) else 1.0)
        speed_weight = (objective_profile["weight_u_overspeed"]
                        if speed > target_speed else
                        objective_profile["weight_u"])
        objective_terms["cross_track"] += scale * objective_profile[
            "weight_e_y"] * ey * ey
        objective_terms["heading"] += scale * objective_profile[
            "weight_e_psi"] * epsi * epsi
        objective_terms["speed"] += scale * speed_weight * (speed - target_speed) ** 2
        objective_speed_errors.append(speed - target_speed)
        objective_terms["lateral_speed"] += scale * objective_profile[
            "weight_v"] * lateral_speed * lateral_speed
        objective_terms["yaw_rate"] += scale * objective_profile[
            "weight_r"] * (yaw_rate - target_yaw_rate) ** 2
        objective_terms["steering_state"] += scale * objective_profile[
            "weight_steering_command"] * (
                (steering_command - target_steering) ** 2 +
                (actual_steering - target_steering) ** 2)
        if stage > 0 and incoming_slip_excess is not None:
            objective_terms["slip"] += scale * objective_profile[
                "weight_longitudinal_slip"] * sum(
                    value * value for value in incoming_slip_excess)

    if objective_profile is not None:
        add_state_objective(
            0, progress, float(diag["state"][0]), float(diag["state"][1]),
            float(state[0]), float(diag["state"][3]), float(diag["state"][4]),
            float(diag["state"][6]), float(diag["state"][7]), None)

    for index in range(len(controls)):
        control_steer, plan_q = source_rollout_command(controls[index])
        control_q = plan_q if throttle_override is None else throttle_override
        reference = sample_path(points, length, progress)
        plant_input = evaluator.UnitySourcePlantInputC()
        plant_input.state[:] = state
        plant_input.wheel_rotation_angle_rad[:] = wheel_angles
        # Match vehicle_model.c's exact source-boundary clamp. The optimizer
        # stores the legal float32 representation of 30 degrees, which is a
        # few ulps above the source double constant.
        plant_input.steering_target_rad = max(
            -STEERING_LIMIT_RAD, min(STEERING_LIMIT_RAD, control_steer))
        plant_input.throttle_normalized = control_q
        plant_input.suspension_history_initialized = suspension_initialized
        plant_output = evaluator.UnitySourcePlantOutputC()
        if not step_function(ctypes.byref(plant_input), 0.025,
                             ctypes.byref(plant_output)):
            failed_step = index + 1
            break

        ds_body = float(plant_output.integrated_body_forward_displacement_m)
        dy_body = float(plant_output.integrated_body_lateral_displacement_m)
        dyaw = float(plant_output.integrated_yaw_delta_rad)
        c, s = math.cos(e_psi), math.sin(e_psi)
        path_forward = ds_body * c - dy_body * s
        path_lateral = ds_body * s + dy_body * c
        denominator = 1.0 - reference["curvature"] * e_y
        if abs(denominator) < 0.05:
            failed_step = index + 1
            break
        delta_s = path_forward / denominator
        next_e_y = (e_y + path_lateral -
                    0.5 * reference["curvature"] * delta_s * delta_s)
        next_e_psi = wrap(e_psi + dyaw - reference["curvature"] * delta_s)
        progress += delta_s
        output_state = list(plant_output.state)
        u = output_state[0]
        u_interval_accels.append((u - float(state[0])) / 0.025)
        r = output_state[2]
        v_rear = output_state[1] - r * COM_FORWARD_FROM_REAR_M
        q = control_q
        if objective_profile is not None:
            steering_delta = control_steer - previous_objective_steering
            throttle_delta = control_q - previous_objective_throttle
            objective_terms["steering_effort"] += objective_profile[
                "weight_steering_target"] * control_steer * control_steer
            objective_terms["throttle_effort"] += objective_profile[
                "weight_throttle_command"] * control_q * control_q
            objective_terms["steering_slew"] += objective_profile[
                "weight_steering_slew"] * steering_delta * steering_delta
            objective_terms["throttle_slew"] += objective_profile[
                "weight_throttle_slew"] * throttle_delta * throttle_delta
            previous_objective_steering = control_steer
            previous_objective_throttle = control_q
        # The source state is passed to the next production stage through the
        # MPC's float u/v/r and physical-steering coordinates, while hidden
        # wheel/suspension states remain double precision.
        next_mpc_u = ctypes.c_float(u).value
        next_mpc_r = ctypes.c_float(r).value
        next_mpc_v = ctypes.c_float(v_rear).value
        state = ([next_mpc_u,
                  next_mpc_v + next_mpc_r * COM_FORWARD_FROM_REAR_M,
                  next_mpc_r]
                 + output_state[3:13]
                 + [ctypes.c_float(output_state[13]).value]
                 + output_state[14:26])
        if objective_profile is not None:
            e_y = ctypes.c_float(next_e_y).value
            e_psi = ctypes.c_float(next_e_psi).value
            add_state_objective(
                index + 1, progress, e_y, e_psi, next_mpc_u, next_mpc_v,
                next_mpc_r, control_steer,
                ctypes.c_float(output_state[13]).value,
                list(map(float,
                         plant_output.maximum_longitudinal_slip_excess_mps)))
        suspension_initialized = int(plant_output.suspension_history_initialized)
        wheel_angles = [0.0] * 4

        after_ref = sample_path(points, length, progress)
        left_extent, right_extent = footprint_extents(next_e_psi)
        margin = (first_prediction_clearance_m if index == 0
                  else wall_clearance_m)
        preview_left, preview_right = sample_corridor_bounds(
            points, length, progress, corridor_preview_halfwidth_m)
        left_slack = preview_left - margin - left_extent - next_e_y
        right_slack = preview_right - margin - right_extent + next_e_y
        slack = min(left_slack, right_slack)
        speed_error = u - after_ref["speed"]
        speed_accel_errors.append(u_interval_accels[-1] -
                                  after_ref["acceleration"])
        errors.append(speed_error)
        speed_errors.append(speed_error)
        clearance_slack.append(slack)
        throttle_values.append(q)
        peak_lateral_accel.append(float(plant_output.lateral_acceleration_mps2))
        longitudinal_accels.append(
            float(plant_output.longitudinal_acceleration_mps2))
        peak_abs_slip.append(max(abs(float(value))
                                 for value in plant_output.longitudinal_slip))
        rear_lateral_speeds.append(v_rear)
        yaw_rates.append(r)
        actual_steering.append(float(output_state[13]))
        steering_commands.append(control_steer)
        if previous_actual_steering is not None:
            steering_rate_25ms.append(
                (float(output_state[13]) - previous_actual_steering) / 0.025)
        previous_actual_steering = float(output_state[13])
        wheel_omega.extend(map(float, output_state[3:7]))
        normal_loads.extend(map(float, plant_output.normal_load_n))
        if slack < -corridor_tolerance_m and \
                first_corridor_violation_stage is None:
            first_corridor_violation_stage = index + 1
        replay_states[index + 1] = [
            next_e_y, next_e_psi, u, v_rear, r,
            ctypes.c_float(output_state[13]).value,
        ]
        if collect_stage_trace:
            next_delayed_steer_1 = delayed_steer_2
            next_delayed_q_1 = delayed_q_2
            stage_trace.append({
                "stage": index + 1,
                "progress_m": progress,
                "e_y_m": next_e_y,
                "e_psi_rad": next_e_psi,
                "u_mps": u,
                "v_rear_mps": v_rear,
                "r_radps": r,
                "actual_steering_rad": ctypes.c_float(output_state[13]).value,
                "command_steering_rad": control_steer,
                "command_throttle": control_q,
                "applied_throttle": q,
                "corridor_slack_m": slack,
                "mpc_state": [
                    next_e_y, next_e_psi, u, v_rear, r,
                    next_delayed_steer_1, control_steer,
                    float(output_state[13]), next_delayed_q_1, control_q,
                ],
                "plant_internal_state": (
                    output_state[3:13] + output_state[14:26]),
                "suspension_history_initialized": int(
                    plant_output.suspension_history_initialized),
            })
        output_csv.writerow([
            diag.get("source_stamp_ns", 0), diag.get("status", ""),
            scenario_label if scenario_label is not None else
            (("captured_plan" if diag.get("status") in
              MPC_FULL_HORIZON_STATUSES else "unaccepted_candidate_sequence")
             if throttle_override is None else
             f"fixed_q_{throttle_override:.2f}"),
            index + 1, progress, next_e_y, next_e_psi, u, v_rear, r,
            output_state[13], control_steer, control_q,
            control_steer, control_q, q, after_ref["speed"], speed_error,
            after_ref["curvature"], after_ref["curvature"] * u,
            u * u * after_ref["curvature"],
            after_ref["acceleration"], u_interval_accels[-1],
            plant_output.longitudinal_acceleration_mps2,
            plant_output.lateral_acceleration_mps2,
            plant_output.yaw_acceleration_radps2,
            *plant_output.longitudinal_slip, *plant_output.lateral_slip,
            *plant_output.normal_load_n, margin, left_slack, right_slack,
        ])
        e_y, e_psi = next_e_y, next_e_psi
        if (first_corridor_violation_stage is not None and
                stop_at_corridor_violation):
            break
        delayed_steer_1, delayed_steer_2 = delayed_steer_2, control_steer
        delayed_q_1, delayed_q_2 = delayed_q_2, control_q

    prediction_errors: list[float] = []
    prediction_state_errors: dict[str, list[float]] = {
        "e_y_m": [], "e_psi_rad": [], "u_mps": [], "v_rear_mps": [],
        "r_radps": [], "actual_steering_rad": [],
    }


    field_indices = {"e_y_m": 0, "e_psi_rad": 1, "u_mps": 2,
                     "v_rear_mps": 3, "r_radps": 4,
                     "actual_steering_rad": 5}
    for stage, expected in prediction_by_stage.items():
        if throttle_override is not None:
            break
        replayed = replay_states.get(stage)
        observed = expected.get("state")
        if replayed is None or not isinstance(observed, list) or len(observed) < 8:
            continue
        observed_values = [float(observed[i]) for i in range(5)] + [
            float(observed[7])]
        for name, index in field_indices.items():
            prediction_state_errors[name].append(replayed[index] -
                                                 observed_values[index])
    all_prediction_diffs = [value for values in prediction_state_errors.values()
                            for value in values]
    controls_q = ([float(control[1]) for control in controls]
                  if throttle_override is None else
                  [float(throttle_override)] * len(controls))
    full_horizon_accepted = diag.get("status") in MPC_FULL_HORIZON_STATUSES
    return {
        "source_stamp_ns": int(diag.get("source_stamp_ns", 0)),
        "bag_time_s": diag.get("bag_time"),
        "regime_match": diag.get("regime_match"),
        "progress_start_m": float(diag["progress_m"]),
        "status": diag.get("status"),
        "scenario": (("captured_plan" if full_horizon_accepted else
                      "unaccepted_candidate_sequence")
                     if throttle_override is None else
                     f"fixed_q_{throttle_override:.3f}; steering schedule held "
                     "from captured plan"),
        "captured_status_was_full_horizon_accepted": full_horizon_accepted,
        "is_original_validated_plan": throttle_override is None and
            full_horizon_accepted,
        "wall_clearance_requirement_m": wall_clearance_m,
        "first_prediction_wall_clearance_requirement_m":
            first_prediction_clearance_m,
        "source_plant_replay_steps": len(replay_states),
        "stage_trace": stage_trace if collect_stage_trace else None,
        "replay_failure_stage": failed_step,
        "first_corridor_violation_stage": first_corridor_violation_stage,
        "all_tested_throttle_commands_at_least_0p001":
            min(controls_q) >= MIN_POSITIVE_THROTTLE - 1.0e-7,
        "tested_throttle_min": min(controls_q),
        "tested_throttle_max": max(controls_q),
        "commanded_q_during_replay_min": min(throttle_values)
            if throttle_values else None,
        "min_corridor_slack_m": min(clearance_slack)
            if clearance_slack else None,
        "max_abs_speed_tracking_error_mps": max(map(abs, speed_errors))
            if speed_errors else None,
        "speed_tracking_error_mps": {
            **metric(speed_errors),
            "bias": statistics.mean(speed_errors) if speed_errors else None,
        },
        "max_abs_lateral_acceleration_mps2": max(map(abs, peak_lateral_accel))
            if peak_lateral_accel else None,
        "max_abs_rear_lateral_speed_mps": max(map(abs, rear_lateral_speeds))
            if rear_lateral_speeds else None,
        "max_abs_yaw_rate_radps": max(map(abs, yaw_rates))
            if yaw_rates else None,
        "max_abs_actual_steering_rad": max(map(abs, actual_steering))
            if actual_steering else None,
        "max_abs_command_steering_rad": max(map(abs, steering_commands))
            if steering_commands else None,
        "max_abs_actual_steering_rate_25ms_radps":
            max(map(abs, steering_rate_25ms)) if steering_rate_25ms else None,
        "max_abs_wheel_omega_radps": max(map(abs, wheel_omega))
            if wheel_omega else None,
        "minimum_wheel_normal_load_n": min(normal_loads)
            if normal_loads else None,
        "source_interval_acceleration_minus_reference_acceleration_mps2":
            metric(speed_accel_errors),
        "source_longitudinal_acceleration_min_mps2": min(longitudinal_accels)
            if longitudinal_accels else None,
        "source_longitudinal_acceleration_max_mps2": max(longitudinal_accels)
            if longitudinal_accels else None,
        "source_forward_speed_interval_acceleration_min_mps2": min(u_interval_accels)
            if u_interval_accels else None,
        "source_forward_speed_interval_acceleration_max_mps2": max(u_interval_accels)
            if u_interval_accels else None,
        "source_forward_speed_end_mps": replay_states[max(replay_states)][2]
            if replay_states else None,
        "reference_speed_end_mps": sample_path(
            points, length, progress)["speed"] if replay_states else None,
        "max_abs_longitudinal_slip": max(peak_abs_slip)
            if peak_abs_slip else None,
        "scalar_nonlinear_objective": (
            {"total": sum(objective_terms.values()),
             "terms": objective_terms,
             "speed_error_mps": {
                 **metric(objective_speed_errors),
                 "bias": statistics.mean(objective_speed_errors),
             },
             "corridor_valid": first_corridor_violation_stage is None,
             "matches_production_formula": True}
            if objective_complete and len(replay_states) == len(controls) and
            failed_step is None
            else None),
        "logged_prediction_sample_count": len(all_prediction_diffs),
        "source_replay_minus_logged_prediction": {
            name: metric(values) for name, values in prediction_state_errors.items()
        },
        "max_abs_replay_vs_logged_prediction": max(map(abs, all_prediction_diffs))
            if all_prediction_diffs else None,
        "diagnostic_reported_minimum_corridor_slack_m": (
            diag.get("minimum_predicted_corridor_slack_m")),
        "classification": (
            (("source-plant feasible witness over the captured 750 ms plan"
              if failed_step is None and clearance_slack and
              min(clearance_slack) >= -corridor_tolerance_m else
              "captured accepted full-horizon plan fails replayed corridor/plant "
              "checks; not proof of global trajectory infeasibility")
             if full_horizon_accepted else
             ("replayed unaccepted candidate sequence; local source replay "
              "diagnoses its first violation only and is not a feasibility "
              "claim about other sequences"))
            if throttle_override is None else
            ("one-factor source-plant counterfactual with captured steering; "
             "not an optimized or accepted control plan")),
    }


def compare_candidate_and_prior_plan(
    diagnostics: list[dict[str, Any]],
    points: list[dict[str, float]],
    length: float,
    step_function: Any,
    witness_csv: csv.writer,
    comparison_csv: csv.writer,
    wall_clearance_m: float,
    first_prediction_clearance_m: float,
    corridor_preview_halfwidth_m: float,
    corridor_tolerance_m: float,
    objective_profile: dict[str, float] | None = None,
    blend_writer: csv.writer | None = None,
    blend_weights: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> list[dict[str, Any]]:
    """Replay both full suffixes and their actual 75 ms prefix decision."""
    comparisons: list[dict[str, Any]] = []
    for diag in diagnostics:
        if diag.get("status") not in {
                "accepted_candidate_prefix", "rejected_nonlinear_rollout"}:
            continue
        snapshot = diag.get("certified_plan_snapshot")
        snapshot_source = "recorded"
        if not isinstance(snapshot, dict):
            snapshot = diag.get("reconstructed_certified_plan_snapshot")
            snapshot_source = "reconstructed_from_published_control_history"
        if not isinstance(snapshot, dict) or not snapshot.get("valid"):
            continue
        candidate_controls = diag.get("candidate_controls")
        if not isinstance(candidate_controls, list):
            candidate_controls = diag.get("rejected_candidate_controls")
        prior_controls = snapshot.get("controls")
        if (not isinstance(candidate_controls, list) or
                not isinstance(prior_controls, list) or
                len(candidate_controls) != 30 or
                len(prior_controls) != int(snapshot.get("horizon", -1))):
            continue
        # The diagnostic snapshot is emitted after the current cycle's
        # publication decision. A candidate-prefix publication advances the
        # retained-plan cursor in try_publish_candidate_prefix(), thereby
        # skipping the old-plan action replaced by the candidate. For that
        # same-state counterfactual, the replaced action is now at `cursor`.
        # A rejected cycle publishes nothing and leaves the cursor unchanged,
        # so its next prior-plan action remains `cursor + 1`.
        cursor = int(snapshot.get("cursor", -1))
        prior_start = (cursor if diag.get("status") ==
                       "accepted_candidate_prefix" else cursor + 1)
        if not 0 <= prior_start < len(prior_controls):
            continue
        prior_suffix = prior_controls[prior_start:]
        candidate = dict(diag)
        candidate["candidate_controls"] = candidate_controls
        prior = dict(diag)
        prior["candidate_controls"] = prior_suffix
        common = (points, length, step_function, witness_csv,
                  wall_clearance_m, first_prediction_clearance_m,
                  corridor_preview_halfwidth_m, corridor_tolerance_m)
        candidate_result = _source_rollout(
            candidate, *common, scenario_label="candidate_full_suffix")
        prior_result = _source_rollout(
            prior, *common, scenario_label="retained_certified_plan_suffix")
        prefix_steps = min(3, len(candidate_controls), len(prior_suffix))
        candidate_prefix = dict(diag)
        candidate_prefix["candidate_controls"] = candidate_controls[:prefix_steps]
        prior_prefix = dict(diag)
        prior_prefix["candidate_controls"] = prior_suffix[:prefix_steps]
        candidate_prefix_result = _source_rollout(
            candidate_prefix, *common,
            scenario_label="candidate_75ms_prefix",
            objective_profile=objective_profile)
        prior_prefix_result = _source_rollout(
            prior_prefix, *common,
            scenario_label="prior_75ms_prefix",
            objective_profile=objective_profile)
        blend_trials: list[dict[str, Any]] = []
        for candidate_weight in blend_weights:
            blended_controls = blend_candidate_and_prior_controls(
                candidate_controls, prior_suffix, candidate_weight)
            command_bounds_valid = all(
                math.isfinite(float(action[0])) and
                abs(float(action[0])) <= STEERING_LIMIT_RAD + 1.0e-7 and
                math.isfinite(float(action[1])) and
                MIN_POSITIVE_THROTTLE - 1.0e-7 <= float(action[1]) <= 1.0 + 1.0e-7
                for action in blended_controls)
            blended_diag = dict(diag)
            blended_diag["candidate_controls"] = blended_controls
            blend_result = _source_rollout(
                blended_diag, *common,
                scenario_label=(f"candidate_prior_blend_"
                                f"{candidate_weight:.2f}"),
                objective_profile=objective_profile)
            speed_error = blend_result.get("speed_tracking_error_mps") or {}
            objective = blend_result.get("scalar_nonlinear_objective") or {}
            full_horizon_valid = (
                command_bounds_valid and
                blend_result["source_plant_replay_steps"] == 30 and
                blend_result["first_corridor_violation_stage"] is None and
                blend_result.get("replay_failure_stage") is None)
            blend_trial = {
                "candidate_weight": candidate_weight,
                "command_bounds_valid": command_bounds_valid,
                "source_steps": blend_result["source_plant_replay_steps"],
                "first_corridor_violation_stage":
                    blend_result["first_corridor_violation_stage"],
                "minimum_corridor_slack_m":
                    blend_result["min_corridor_slack_m"],
                "speed_error_rmse_mps": speed_error.get("rmse"),
                "objective_total": objective.get("total"),
                "full_horizon_source_valid": full_horizon_valid,
                "first_steering_rad": blended_controls[0][0],
                "first_throttle": blended_controls[0][1],
                "controls": blended_controls,
                "tail_extension": "hold_last_retained_action",
                "interpretation": (
                    "offline candidate/prior interpolation screened by the exact "
                    "source plant; not an MPC solution or certified action plan"),
            }
            blend_trials.append(blend_trial)
            if blend_writer is not None:
                blend_writer.writerow([
                    diag.get("bag_time"), float(diag["progress_m"]),
                    len(prior_suffix), candidate_weight, command_bounds_valid,
                    blend_result["source_plant_replay_steps"],
                    blend_result["first_corridor_violation_stage"],
                    blend_result["min_corridor_slack_m"], speed_error.get("rmse"),
                    objective.get("total"), full_horizon_valid,
                    blended_controls[0][0], blended_controls[0][1],
                    json.dumps(blended_controls, separators=(",", ":")),
                ])
        candidate_objective = candidate_prefix_result.get(
            "scalar_nonlinear_objective") or {}
        prior_objective = prior_prefix_result.get(
            "scalar_nonlinear_objective") or {}
        candidate_speed_error = candidate_prefix_result.get(
            "speed_tracking_error_mps") or {}
        prior_speed_error = prior_prefix_result.get(
            "speed_tracking_error_mps") or {}
        candidate_action = candidate_controls[0]
        prior_action = prior_suffix[0]
        candidate_fail = candidate_result["first_corridor_violation_stage"]
        prior_fail = prior_result["first_corridor_violation_stage"]
        candidate_survival = (
            candidate_fail - 1 if candidate_fail is not None else
            candidate_result["source_plant_replay_steps"])
        prior_survival = (
            prior_fail - 1 if prior_fail is not None else
            prior_result["source_plant_replay_steps"])
        comparison = {
            "bag_time_s": diag.get("bag_time"),
            "status": diag.get("status"),
            "candidate_prefix_chain_length": diag.get(
                "consecutive_candidate_prefix_actions"),
            "progress_m": float(diag["progress_m"]),
            "prior_snapshot_source": snapshot_source,
            "candidate_horizon": len(candidate_controls),
            "candidate_first_corridor_violation_stage": candidate_fail,
            "candidate_minimum_physical_corridor_slack_m":
                candidate_result["min_corridor_slack_m"],
            "prior_cursor": int(snapshot["cursor"]),
            "prior_suffix_start_index": prior_start,
            "prior_remaining_horizon": len(prior_suffix),
            "prior_first_corridor_violation_stage": prior_fail,
            "prior_minimum_physical_corridor_slack_m":
                prior_result["min_corridor_slack_m"],
            "candidate_issued_action_steering_rad": float(candidate_action[0]),
            "candidate_issued_action_throttle": float(candidate_action[1]),
            "prior_action_steering_rad": float(prior_action[0]),
            "prior_action_throttle": float(prior_action[1]),
            "absolute_first_steering_difference_rad": abs(
                float(candidate_action[0]) - float(prior_action[0])),
            "absolute_first_throttle_difference": abs(
                float(candidate_action[1]) - float(prior_action[1])),
            "candidate_solver_primal_residual": diag.get("primal_residual"),
            "candidate_solver_dual_residual": diag.get("dual_residual"),
            "candidate_survival_steps_before_violation": candidate_survival,
            "prior_survival_steps_before_violation": prior_survival,
            "prior_survives_longer": prior_survival > candidate_survival,
            "prior_valid_for_full_remaining_suffix": prior_fail is None,
            "prefix_comparison_horizon_steps": prefix_steps,
            "candidate_prefix_source_plant_footprint_valid": (
                prefix_steps == 3 and
                candidate_prefix_result["source_plant_replay_steps"] == 3 and
                candidate_prefix_result["first_corridor_violation_stage"] is None and
                candidate_prefix_result.get("replay_failure_stage") is None and
                min(float(action[1]) for action in candidate_controls[:3]) >=
                    MIN_POSITIVE_THROTTLE - 1.0e-7 and
                max(float(action[1]) for action in candidate_controls[:3]) <=
                    1.0 + 1.0e-7 and
                max(abs(float(action[0])) for action in candidate_controls[:3]) <=
                    STEERING_LIMIT_RAD + 1.0e-7),
            "prior_prefix_source_plant_footprint_valid": (
                prefix_steps == 3 and
                prior_prefix_result["source_plant_replay_steps"] == 3 and
                prior_prefix_result["first_corridor_violation_stage"] is None and
                prior_prefix_result.get("replay_failure_stage") is None and
                min(float(action[1]) for action in prior_suffix[:3]) >=
                    MIN_POSITIVE_THROTTLE - 1.0e-7 and
                max(float(action[1]) for action in prior_suffix[:3]) <=
                    1.0 + 1.0e-7 and
                max(abs(float(action[0])) for action in prior_suffix[:3]) <=
                    STEERING_LIMIT_RAD + 1.0e-7),
            "candidate_prefix_minimum_physical_corridor_slack_m":
                candidate_prefix_result["min_corridor_slack_m"],
            "prior_prefix_minimum_physical_corridor_slack_m":
                prior_prefix_result["min_corridor_slack_m"],
            "candidate_prefix_speed_error_rmse_mps":
                candidate_speed_error.get("rmse"),
            "candidate_prefix_speed_error_bias_mps":
                candidate_speed_error.get("bias"),
            "candidate_prefix_speed_error_max_abs_mps":
                candidate_speed_error.get("max_abs"),
            "prior_prefix_speed_error_rmse_mps":
                prior_speed_error.get("rmse"),
            "prior_prefix_speed_error_bias_mps":
                prior_speed_error.get("bias"),
            "prior_prefix_speed_error_max_abs_mps":
                prior_speed_error.get("max_abs"),
            "candidate_prefix_objective": candidate_objective.get("total"),
            "prior_prefix_objective": prior_objective.get("total"),
            "candidate_prefix_objective_terms_json": json.dumps(
                candidate_objective.get("terms"), sort_keys=True),
            "prior_prefix_objective_terms_json": json.dumps(
                prior_objective.get("terms"), sort_keys=True),
            "blended_continuation_trials": blend_trials,
        }
        comparisons.append(comparison)
        comparison_csv.writerow([comparison.get(field)
                                 for field in CANDIDATE_PRIOR_FIELDS])
    return comparisons


def select_terminal_predecessors(diagnostics: list[dict[str, Any]],
                                 count: int) -> list[dict[str, Any]]:
    if not diagnostics or count <= 0:
        return []
    terminal_index = next((i for i in range(len(diagnostics) - 1, -1, -1)
                           if diagnostics[i].get("status") ==
                           "rejected_nonlinear_rollout"), len(diagnostics) - 1)
    prefix_index = terminal_index - 1
    while prefix_index >= 0 and diagnostics[prefix_index].get("status") == \
            "accepted_candidate_prefix":
        prefix_index -= 1
    start = max(0, prefix_index - count + 1)
    selected = [row for row in diagnostics[start:prefix_index + 1]
                if row.get("status") in MPC_FULL_HORIZON_STATUSES and
                isinstance(row.get("candidate_controls"), list) and
                len(row["candidate_controls"]) == 30 and
                isinstance(row.get("plant_internal_state"), list) and
                len(row["plant_internal_state"]) == 22]
    return selected[-count:]


def select_nearest_regime(diagnostics: list[dict[str, Any]], count: int,
                          target_speed_mps: float,
                          target_abs_curvature_per_m: float) -> list[dict[str, Any]]:
    """Choose separated accepted snapshots nearest a speed/curvature target."""
    eligible = []
    for row in diagnostics:
        state = row.get("state")
        curvature = row.get("path_curvature_per_m")
        if (row.get("status") not in MPC_FULL_HORIZON_STATUSES or
                not isinstance(row.get("candidate_controls"), list) or
                len(row["candidate_controls"]) != 30 or
                not isinstance(row.get("plant_internal_state"), list) or
                len(row["plant_internal_state"]) != 22 or
                not isinstance(state, list) or len(state) != 10 or
                not isinstance(curvature, (int, float))):
            continue
        speed_error = abs(float(state[2]) - target_speed_mps)
        curvature_error = abs(abs(float(curvature)) -
                              target_abs_curvature_per_m)
        score = speed_error / 1.0 + curvature_error / 0.15
        eligible.append((score, row, speed_error, curvature_error))
    eligible.sort(key=lambda item: (item[0], float(item[1].get("bag_time", 0.0))))
    selected: list[dict[str, Any]] = []
    selected_times: list[float] = []
    for _, row, speed_error, curvature_error in eligible:
        stamp = float(row.get("bag_time", 0.0))
        if any(abs(stamp - previous) < 0.5 for previous in selected_times):
            continue
        copy = dict(row)
        copy["regime_match"] = {
            "target_speed_mps": target_speed_mps,
            "actual_speed_mps": float(row["state"][2]),
            "absolute_speed_error_mps": speed_error,
            "target_abs_curvature_per_m": target_abs_curvature_per_m,
            "actual_abs_curvature_per_m": abs(float(row["path_curvature_per_m"])),
            "absolute_curvature_error_per_m": curvature_error,
        }
        selected.append(copy)
        selected_times.append(stamp)
        if len(selected) == count:
            break
    return selected


def validate_fixed_throttle_grid(values: list[float]) -> None:
    """Keep offline drive-mode probes inside the production MPC command set."""
    if any(not math.isfinite(q) or q < MIN_POSITIVE_THROTTLE or q > 1.0
           for q in values):
        raise ValueError("fixed throttle probes must lie in [0.001, 1.0]")


def select_last_candidate_prefix_before_terminal_failure(
        diagnostics: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the last logged short-prefix action and its unaccepted full sequence."""
    terminal_index = next((i for i in range(len(diagnostics) - 1, -1, -1)
                           if diagnostics[i].get("status") ==
                           "rejected_nonlinear_rollout"), None)
    if terminal_index is None or terminal_index == 0:
        return None
    candidate = diagnostics[terminal_index - 1]
    if candidate.get("status") != "accepted_candidate_prefix" or \
            not isinstance(candidate.get("candidate_controls"), list) or \
            len(candidate["candidate_controls"]) != 30 or \
            not isinstance(candidate.get("plant_internal_state"), list) or \
            len(candidate["plant_internal_state"]) != 22:
        return None
    return candidate


def reconstruct_certified_plan_history(
        diagnostics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct retained controls from published full-plan/action history.

    Newer bags carry an explicit post-publication snapshot. Older bags can be
    reconstructed only when every relevant control publication is recorded:
    accepted full plans replace the retained sequence, while published
    candidate/prior actions advance its consumed cursor. Explicit snapshots
    are used as an oracle to validate this state machine where available.
    """
    reconstructed: list[dict[str, Any]] = []
    controls: list[list[float]] | None = None
    cursor = 0
    checked = matched = mismatches = 0
    missing_publication_flags = 0

    def snapshot() -> dict[str, Any] | None:
        if controls is None:
            return None
        return {"valid": True, "horizon": len(controls), "cursor": cursor,
                "controls": controls}

    for diagnostic in diagnostics:
        row = dict(diagnostic)
        status = row.get("status")
        publish_flag = row.get("command_published")
        if (status in MPC_FULL_HORIZON_STATUSES or
                status in PUBLISHED_CONTROL_STATUSES) and publish_flag is None:
            missing_publication_flags += 1
        published = publish_flag is True

        if status in MPC_FULL_HORIZON_STATUSES and published:
            candidate = row.get("candidate_controls")
            if isinstance(candidate, list) and len(candidate) == 30:
                controls = [[float(action[0]), float(action[1])]
                            for action in candidate]
                cursor = 0
        elif status in PUBLISHED_CONTROL_STATUSES and published:
            if controls is not None and cursor < len(controls) - 1:
                cursor += 1

        expected = snapshot()
        recorded = row.get("certified_plan_snapshot")
        if isinstance(recorded, dict):
            checked += 1
            if expected is None:
                same = not bool(recorded.get("valid"))
            else:
                recorded_controls = recorded.get("controls", [])
                same = (
                    bool(recorded.get("valid")) and
                    int(recorded.get("cursor", -1)) == cursor and
                    len(recorded_controls) == len(controls) and
                    all(abs(float(a[index]) - float(b[index])) <= 1.0e-6
                        for a, b in zip(recorded_controls, controls)
                        for index in (0, 1)))
            if same:
                matched += 1
            else:
                mismatches += 1
                # An explicit snapshot is authoritative; resynchronize after
                # recording the discrepancy rather than propagating drift.
                if recorded.get("valid"):
                    controls = [[float(action[0]), float(action[1])]
                                for action in recorded.get("controls", [])]
                    cursor = int(recorded.get("cursor", 0))
                else:
                    controls = None
                    cursor = 0
            row["retained_plan_snapshot_source"] = "recorded"
        elif expected is not None:
            row["reconstructed_certified_plan_snapshot"] = expected
            row["retained_plan_snapshot_source"] = (
                "reconstructed_from_published_control_history")
        else:
            row["retained_plan_snapshot_source"] = "unavailable"
        reconstructed.append(row)

    return reconstructed, {
        "recorded_snapshot_count": checked,
        "reconstruction_matches_recorded_count": matched,
        "reconstruction_mismatch_count": mismatches,
        "missing_command_published_flag_count": missing_publication_flags,
        "interpretation": (
            "reconstruction is checked against explicit snapshots when present; "
            "without snapshots it is conditional on complete command-published "
            "flags and the logged accepted-plan/prefix/prior status sequence"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("bag_run", type=Path, nargs="?",
                        help="optional ROS 2 bag directory containing MPC diagnostics")
    parser.add_argument("--library", type=Path,
                        help="matching libunity_source_plant_eval.so; required with bag_run")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--runtime-mpc-profile", type=Path,
                        default=Path("f1tenth_mpc/config/mpc_competition.yaml"),
                        help="runtime YAML whose corridor margins define audit defaults")
    parser.add_argument("--bin-width-m", type=float, default=0.5)
    parser.add_argument("--wall-clearance-m", type=float,
                        help="override runtime corridor_margin_m (meters)")
    parser.add_argument("--corridor-tolerance-m", type=float,
                        help="override runtime nonlinear corridor tolerance")
    parser.add_argument("--plans-before-terminal-chain", type=int, default=4)
    parser.add_argument("--plan-selection",
                        choices=("terminal_predecessors", "nearest_regime"),
                        default="terminal_predecessors")
    parser.add_argument("--target-speed-mps", type=float)
    parser.add_argument("--target-abs-curvature-per-m", type=float)
    parser.add_argument(
        "--fixed-throttle-grid", type=float, nargs="+", default=[],
        help="additional offline source-plant counterfactuals. Each value is "
             "held across the horizon while the captured steering sequence "
             "and in-flight command queue remain unchanged; values must be "
             "in the ordinary MPC drive range [0.001, 1.0]. This is a local "
             "response sweep, not a feasible-policy search.",
    )
    parser.add_argument(
        "--blend-weight-grid", type=float, nargs="+",
        default=[0.25, 0.50, 0.75],
        help="candidate-control blend weights for offline source-plant search; "
             "0 is retained plan and 1 is candidate")
    args = parser.parse_args()
    if args.bin_width_m <= 0.0 or \
            (args.wall_clearance_m is not None and args.wall_clearance_m < 0.0) or \
            (args.corridor_tolerance_m is not None and
             args.corridor_tolerance_m < 0.0) or \
            args.plans_before_terminal_chain < 0:
        parser.error("bin, clearance, tolerance or plan count is invalid")
    try:
        validate_fixed_throttle_grid(args.fixed_throttle_grid)
        blend_weights = validate_blend_weight_grid(args.blend_weight_grid)
    except ValueError as error:
        parser.error(str(error))
    if not args.runtime_mpc_profile.is_file():
        parser.error("runtime MPC profile does not exist")
    try:
        runtime_profile = read_runtime_corridor_profile(args.runtime_mpc_profile)
        objective_profile = read_runtime_objective_profile(
            args.runtime_mpc_profile)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    wall_clearance_m = (runtime_profile["corridor_margin_m"]
                        if args.wall_clearance_m is None
                        else args.wall_clearance_m)
    first_prediction_clearance_m = runtime_profile[
        "first_prediction_corridor_margin_m"]
    corridor_preview_halfwidth_m = runtime_profile[
        "corridor_preview_halfwidth_m"]
    corridor_tolerance_m = (
        runtime_profile["nonlinear_corridor_tolerance_m"]
        if args.corridor_tolerance_m is None else args.corridor_tolerance_m)
    if args.bag_run and (args.library is None or not args.library.is_file()):
        parser.error("bag_run requires an existing --library")
    if args.plan_selection == "nearest_regime" and (
            args.target_speed_mps is None or
            args.target_abs_curvature_per_m is None or
            args.target_speed_mps < 0.0 or
            args.target_abs_curvature_per_m < 0.0):
        parser.error("nearest_regime requires non-negative speed and curvature targets")

    points, length = load_trajectory(args.trajectory)
    geometry, bins = geometry_audit(
        points, length, args.bin_width_m, wall_clearance_m,
        corridor_tolerance_m)
    diagnostics: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    terminal_prefix_candidate: dict[str, Any] | None = None
    step_function = None
    retained_plan_reconstruction = {
        "recorded_snapshot_count": 0,
        "reconstruction_matches_recorded_count": 0,
        "reconstruction_mismatch_count": 0,
        "missing_command_published_flag_count": 0,
        "interpretation": "no bag supplied",
    }
    if args.bag_run:
        data = evaluator.read_bag(args.bag_run)
        diagnostics = [dict(record.get("json", {}),
                            bag_time=record.get("bag_time"))
                       for record in data.get(evaluator.DIAGNOSTICS, [])]
        diagnostics, retained_plan_reconstruction = \
            reconstruct_certified_plan_history(diagnostics)
        if args.plan_selection == "nearest_regime":
            selected = select_nearest_regime(
                diagnostics, args.plans_before_terminal_chain,
                args.target_speed_mps, args.target_abs_curvature_per_m)
        else:
            selected = select_terminal_predecessors(
                diagnostics, args.plans_before_terminal_chain)
            terminal_prefix_candidate = \
                select_last_candidate_prefix_before_terminal_failure(diagnostics)
        model = ctypes.CDLL(str(args.library.resolve()))
        step_function = model.unity_source_plant_step_c
        step_function.argtypes = [
            ctypes.POINTER(evaluator.UnitySourcePlantInputC), ctypes.c_double,
            ctypes.POINTER(evaluator.UnitySourcePlantOutputC)]
        step_function.restype = ctypes.c_int

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = args.output_prefix.with_name(args.output_prefix.name + "_bins.csv")
    witness_path = args.output_prefix.with_name(
        args.output_prefix.name + "_source_witness.csv")
    candidate_prior_path = args.output_prefix.with_name(
        args.output_prefix.name + "_candidate_prior_ab.csv")
    blend_continuation_path = args.output_prefix.with_name(
        args.output_prefix.name + "_blend_continuation_search.csv")
    dynamic_bins_path = args.output_prefix.with_name(
        args.output_prefix.name + "_dynamic_feasibility_bins.csv")
    report_path = args.output_prefix.with_suffix(".json")
    with bin_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bins[0].keys()))
        writer.writeheader()
        writer.writerows(bins)

    witness_header = [
        "plan_source_stamp_ns", "plan_status", "scenario", "stage", "progress_m",
        "e_y_m", "e_psi_rad", "u_mps", "v_rear_mps", "r_radps",
        "actual_steering_rad", "command_steering_rad", "command_throttle",
        "source_plant_input_throttle", "reference_speed_mps",
        "speed_error_mps", "curvature_per_m", "reference_yaw_rate_radps",
        "reference_lateral_accel_mps2", "reference_acceleration_mps2",
        "source_interval_acceleration_mps2", "source_ax_mps2", "source_ay_mps2",
        "source_yaw_accel_radps2", "slip_fl", "slip_fr", "slip_rl",
        "slip_rr", "lateral_slip_fl", "lateral_slip_fr", "lateral_slip_rl",
        "lateral_slip_rr", "normal_fl_n", "normal_fr_n", "normal_rl_n",
        "normal_rr_n", "wall_clearance_requirement_m",
        "left_corridor_slack_m", "right_corridor_slack_m",
    ]
    plan_results: list[dict[str, Any]] = []
    terminal_prefix_replay: dict[str, Any] | None = None
    terminal_throttle_counterfactuals: dict[str, Any] = {}
    with witness_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(witness_header)
        if terminal_prefix_candidate is not None:
            assert step_function is not None
            terminal_prefix_replay = _source_rollout(
                terminal_prefix_candidate, points, length, step_function, writer,
                wall_clearance_m, first_prediction_clearance_m,
                corridor_preview_halfwidth_m, corridor_tolerance_m)
            if terminal_prefix_replay is not None:
                terminal_throttle_counterfactuals = {
                    "fixed_q_0p001": _source_rollout(
                        terminal_prefix_candidate, points, length,
                        step_function, writer, wall_clearance_m,
                        first_prediction_clearance_m,
                        corridor_preview_halfwidth_m, corridor_tolerance_m,
                        MIN_POSITIVE_THROTTLE,
                        scenario_label="terminal_fixed_q_0.001"),
                    "fixed_q_0p01": _source_rollout(
                        terminal_prefix_candidate, points, length, step_function,
                        writer, wall_clearance_m, first_prediction_clearance_m,
                        corridor_preview_halfwidth_m, corridor_tolerance_m,
                        0.01, scenario_label="terminal_fixed_q_0.010"),
                    "fixed_q_1p00": _source_rollout(
                        terminal_prefix_candidate, points, length, step_function,
                        writer, wall_clearance_m, first_prediction_clearance_m,
                        corridor_preview_halfwidth_m, corridor_tolerance_m,
                        1.0, scenario_label="terminal_fixed_q_1.000"),
                }
                for fixed_q in sorted(set(args.fixed_throttle_grid)):
                    if fixed_q <= MIN_POSITIVE_THROTTLE + 1.0e-7 or fixed_q >= 0.9999999:
                        continue
                    terminal_throttle_counterfactuals[
                        f"fixed_q_{fixed_q:.3f}"] = _source_rollout(
                            terminal_prefix_candidate, points, length,
                            step_function, writer, wall_clearance_m,
                            first_prediction_clearance_m,
                            corridor_preview_halfwidth_m,
                            corridor_tolerance_m, fixed_q,
                            scenario_label=f"terminal_fixed_q_{fixed_q:.3f}")
        for diag in selected:
            assert step_function is not None
            captured = _source_rollout(
                diag, points, length, step_function, writer, wall_clearance_m,
                first_prediction_clearance_m, corridor_preview_halfwidth_m,
                corridor_tolerance_m, objective_profile=objective_profile)
            logged_objective = diag.get("solver", {}).get("objective")
            captured["runtime_reported_nonlinear_objective"] = logged_objective
            offline_objective = captured.get("scalar_nonlinear_objective")
            captured["offline_minus_runtime_objective"] = (
                offline_objective["total"] - float(logged_objective)
                if offline_objective is not None and
                isinstance(logged_objective, (float, int)) and
                math.isfinite(float(logged_objective)) else None)
            minimum = _source_rollout(
                diag, points, length, step_function, writer, wall_clearance_m,
                first_prediction_clearance_m, corridor_preview_halfwidth_m,
                corridor_tolerance_m, MIN_POSITIVE_THROTTLE,
                scenario_label="captured_fixed_q_0.001")
            fixed_q_0p01 = _source_rollout(
                diag, points, length, step_function, writer, wall_clearance_m,
                first_prediction_clearance_m, corridor_preview_halfwidth_m,
                corridor_tolerance_m, 0.01,
                scenario_label="captured_fixed_q_0.010")
            maximum = _source_rollout(
                diag, points, length, step_function, writer, wall_clearance_m,
                first_prediction_clearance_m, corridor_preview_halfwidth_m,
                corridor_tolerance_m, 1.0)
            captured["throttle_counterfactuals"] = {
                "fixed_q_0p001": minimum,
                "fixed_q_0p01": fixed_q_0p01,
                "fixed_q_1p00": maximum,
                "interpretation": (
                    "one-factor local source-plant response with the captured "
                    "steering plan and in-flight two-command queue preserved; "
                    "not an optimized feasible schedule"),
            }
            for fixed_q in sorted(set(args.fixed_throttle_grid)):
                if fixed_q <= MIN_POSITIVE_THROTTLE + 1.0e-7 or fixed_q >= 0.9999999:
                    continue
                q_result = _source_rollout(
                    diag, points, length, step_function, writer,
                    wall_clearance_m, first_prediction_clearance_m,
                    corridor_preview_halfwidth_m, corridor_tolerance_m,
                    fixed_q, scenario_label=f"fixed_q_{fixed_q:.3f}")
                captured["throttle_counterfactuals"][
                    f"fixed_q_{fixed_q:.3f}"] = q_result
            plan_results.append(captured)

    candidate_prior_results: list[dict[str, Any]] = []
    with candidate_prior_path.open("w", newline="") as stream:
        comparison_writer = csv.writer(stream)
        comparison_writer.writerow(CANDIDATE_PRIOR_FIELDS)
        with blend_continuation_path.open("w", newline="") as blend_stream:
            blend_writer = csv.writer(blend_stream)
            blend_writer.writerow(BLEND_CONTINUATION_FIELDS)
            if step_function is not None:
                with witness_path.open("a", newline="") as witness_stream:
                    witness_writer = csv.writer(witness_stream)
                    candidate_prior_results = compare_candidate_and_prior_plan(
                        diagnostics, points, length, step_function, witness_writer,
                        comparison_writer, wall_clearance_m,
                        first_prediction_clearance_m,
                        corridor_preview_halfwidth_m, corridor_tolerance_m,
                        objective_profile, blend_writer, blend_weights)

    blend_trials = [trial for row in candidate_prior_results
                    for trial in row.get("blended_continuation_trials", [])]
    blend_summary = {
        "trial_count": len(blend_trials),
        "full_horizon_source_valid_count": sum(
            trial["full_horizon_source_valid"] for trial in blend_trials),
        "command_bounds_invalid_count": sum(
            not trial["command_bounds_valid"] for trial in blend_trials),
        "by_candidate_weight": {
            f"{weight:.2f}": {
                "count": sum(trial["candidate_weight"] == weight
                             for trial in blend_trials),
                "full_horizon_source_valid": sum(
                    trial["candidate_weight"] == weight and
                    trial["full_horizon_source_valid"]
                    for trial in blend_trials),
                "minimum_slack_m": min((
                    trial["minimum_corridor_slack_m"]
                    for trial in blend_trials
                    if trial["candidate_weight"] == weight and
                    trial["minimum_corridor_slack_m"] is not None),
                    default=None),
            }
            for weight in blend_weights
        },
        "candidate_weight_grid": list(blend_weights),
        "interpretation": (
            "offline deterministic blend search over same-state candidate and "
            "retained-plan controls; short retained suffix is extended by "
            "holding its last action. A passing sequence is a source-model "
            "witness only, not an MPC solution or certified action plan."),
    }
    dynamic_bins = summarize_dynamic_feasibility_by_progress(
        candidate_prior_results, args.bin_width_m, blend_weights)
    with dynamic_bins_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(DYNAMIC_FEASIBILITY_BIN_FIELDS))
        writer.writeheader()
        for row in dynamic_bins:
            csv_row = {
                field: row.get(field)
                for field in DYNAMIC_FEASIBILITY_BIN_FIELDS
            }
            csv_row["by_candidate_weight_valid_counts_json"] = json.dumps(
                row["by_candidate_weight_valid_counts"], sort_keys=True)
            writer.writerow(csv_row)

    prefix_pairs = [row for row in candidate_prior_results
                    if row.get("prefix_comparison_horizon_steps") == 3]
    jointly_valid_prefixes = [
        row for row in prefix_pairs
        if row.get("candidate_prefix_source_plant_footprint_valid") and
        row.get("prior_prefix_source_plant_footprint_valid")]
    objective_pairs = [
        row for row in jointly_valid_prefixes
        if row.get("candidate_prefix_objective") is not None and
        row.get("prior_prefix_objective") is not None]
    prefix_quality_summary = {
        "comparison_count": len(prefix_pairs),
        "candidate_source_plant_footprint_valid_count": sum(
            bool(row.get("candidate_prefix_source_plant_footprint_valid"))
            for row in prefix_pairs),
        "prior_source_plant_footprint_valid_count": sum(
            bool(row.get("prior_prefix_source_plant_footprint_valid"))
            for row in prefix_pairs),
        "both_source_plant_footprint_valid_count": len(jointly_valid_prefixes),
        "objective_comparison_count": len(objective_pairs),
        "candidate_lower_objective_count": sum(
            row["candidate_prefix_objective"] < row["prior_prefix_objective"]
            for row in objective_pairs),
        "prior_lower_objective_count": sum(
            row["prior_prefix_objective"] < row["candidate_prefix_objective"]
            for row in objective_pairs),
        "median_candidate_objective": statistics.median(
            row["candidate_prefix_objective"] for row in objective_pairs)
            if objective_pairs else None,
        "median_prior_objective": statistics.median(
            row["prior_prefix_objective"] for row in objective_pairs)
            if objective_pairs else None,
        "median_candidate_speed_error_rmse_mps": statistics.median(
            row["candidate_prefix_speed_error_rmse_mps"]
            for row in jointly_valid_prefixes
            if row.get("candidate_prefix_speed_error_rmse_mps") is not None)
            if jointly_valid_prefixes else None,
        "median_prior_speed_error_rmse_mps": statistics.median(
            row["prior_prefix_speed_error_rmse_mps"]
            for row in jointly_valid_prefixes
            if row.get("prior_prefix_speed_error_rmse_mps") is not None)
            if jointly_valid_prefixes else None,
        "interpretation": (
            "same-state offline 75 ms source-plant/footprint A/B; not closed-loop "
            "evidence and not proof either continuation remains feasible later"),
    }

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "global_infeasibility_claim": False,
        "artifacts": {
            "trajectory_path": str(args.trajectory),
            "trajectory_sha256": sha256(args.trajectory),
            "runtime_mpc_profile_path": str(args.runtime_mpc_profile),
            "runtime_mpc_profile_sha256": sha256(args.runtime_mpc_profile),
            "runtime_corridor_margins": runtime_profile,
            "runtime_objective_weights": objective_profile,
            "audit_wall_clearance_m": wall_clearance_m,
            "audit_first_prediction_wall_clearance_m":
                first_prediction_clearance_m,
            "audit_nonlinear_corridor_tolerance_m": corridor_tolerance_m,
            "bag_run_path": str(args.bag_run) if args.bag_run else None,
            "source_plant_library_path": str(args.library) if args.library else None,
            "source_plant_library_sha256": sha256(args.library)
                if args.library else None,
            "source_plant_cpp_sha256": sha256(
                Path("f1tenth_mpc/src/unity_source_plant.cpp")),
            "geometry_bins_csv": str(bin_path),
            "source_witness_csv": str(witness_path),
            "candidate_prior_counterfactual_csv": str(candidate_prior_path),
            "blend_continuation_search_csv": str(blend_continuation_path),
            "dynamic_feasibility_bins_csv": str(dynamic_bins_path),
            "bag_db3_sha256": (sha256(next(args.bag_run.glob("*.db3")))
                if args.bag_run and list(args.bag_run.glob("*.db3")) else None),
        },
        "geometry": geometry,
        "dynamic_replay": {
            "bag_diagnostic_count": len(diagnostics),
            "retained_plan_snapshot_reconstruction":
                retained_plan_reconstruction,
            "selected_pre_terminal_full_horizon_plans": len(selected),
            "terminal_unaccepted_candidate_sequence_replayed":
                terminal_prefix_replay is not None,
            "terminal_unaccepted_candidate_sequence": terminal_prefix_replay,
            "terminal_candidate_throttle_counterfactuals":
                terminal_throttle_counterfactuals,
            "selection": (
                "nearest accepted full-horizon snapshots in current speed and "
                "absolute steering curvature"
                if args.plan_selection == "nearest_regime" else
                "last full-horizon-accepted candidate-control plans immediately "
                "preceding the terminal candidate-prefix chain"),
            "plan_selection": args.plan_selection,
            "target_speed_mps": args.target_speed_mps,
            "target_abs_curvature_per_m": args.target_abs_curvature_per_m,
            "source_plant_stage_dt_s": 0.025 if args.bag_run else None,
            "source_plant_internal_substep_s": 0.001,
            "expected_action_delay_packet_stages": 2,
            "plans": plan_results,
            "candidate_vs_retained_plan_counterfactual": {
                "comparison_count": len(candidate_prior_results),
                "candidate_suffix_corridor_failures": sum(
                    row["candidate_first_corridor_violation_stage"] is not None
                    for row in candidate_prior_results),
                "prior_suffixes_valid_through_all_remaining_controls": sum(
                    row["prior_valid_for_full_remaining_suffix"]
                    for row in candidate_prior_results),
                "prior_suffix_survives_more_stages": sum(
                    row["prior_survives_longer"]
                    for row in candidate_prior_results),
                "interpretation": (
                    "same-state offline source-plant counterfactual; retained-plan "
                    "suffixes have only their remaining horizon, and this does "
                    "not establish closed-loop policy benefit or global path "
                    "feasibility"),
            },
            "candidate_prior_blend_continuation_search": blend_summary,
            "dynamic_feasibility_by_progress": {
                "bin_width_m": args.bin_width_m,
                "candidate_weight_grid": list(blend_weights),
                "sampled_bin_count": len(dynamic_bins),
                "bins_with_positive_slack_witness": sum(
                    row["strict_positive_slack_witness_count"] > 0
                    for row in dynamic_bins),
                "bins_without_positive_slack_witness": sum(
                    row["strict_positive_slack_witness_count"] == 0
                    for row in dynamic_bins),
                "interpretation": (
                    "Bins summarize only saved states and the finite candidate/prior "
                    "blend grid. A bin without a tested positive-slack witness is "
                    "unsupported by this search, not globally physically infeasible."),
                "bins": dynamic_bins,
            },
            "three_step_prefix_quality_ab": prefix_quality_summary,
        },
        "failure_classification_policy": {
            "geometry": "computed from CSV path and stored left/right corridor bounds",
            "speed_and_input_demand": (
                "reported from v(s), kappa(s), acceleration(s); not itself a "
                "proof of plant infeasibility"),
            "solver": "only the captured accepted full-horizon plans are witnesses",
            "local_failure": (
                "failed local solve or nonlinear rollout is not a global "
                "dynamical infeasibility proof"),
            "throttle_envelope": "captured MPC controls must satisfy q >= 0.001",
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(report_path),
        "geometry_bins": str(bin_path),
        "source_witness": str(witness_path),
        "candidate_prior_counterfactuals": str(candidate_prior_path),
        "blend_continuation_search": str(blend_continuation_path),
        "geometry": {
            "length_m": geometry["lap_length_m"],
            "max_speed_mps": geometry["speed_max_mps"],
            "max_abs_lateral_accel_mps2": geometry[
                "abs_lateral_accel_demand_max_mps2"],
            "minimum_aligned_corridor_slack_m": geometry[
                "minimum_centerline_corridor_slack_after_footprint_m"],
        },
        "source_plans_replayed": len(plan_results),
        "candidate_prior_comparisons": len(candidate_prior_results),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
