#!/usr/bin/env python3
"""Assemble causal simulator transitions and gate model-identification data.

This is an offline-only tool.  It consumes the packet table written by
``model_id_timing_recorder`` and never creates runtime ROS messages.  Repeated
bridge observations of one simulator state are collapsed, not counted as
extra samples.  Source gaps are retained in the output and reported.

The simulator telemetry is emitted after the fixed-step update. Therefore the
applied command recorded in the current packet is the command that generated
the transition from the previous packet state to the current packet state.
The assembler preserves that causal alignment explicitly; it does not mix
controls from adjacent packets.

The tool intentionally refuses to fit a model when the measured source rate
does not meet the requested timing contract.  Use ``--allow-non-target-rate``
only for an explicitly labelled exploratory artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

try:
    from sdu_apex_autodrive.model_id_frames import (
        PacketFrameError,
        body_velocity_to_world,
        decode_packet,
    )
except ModuleNotFoundError:
    # Keep direct invocation from a source checkout usable without requiring
    # the ROS package to have been installed first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdu_apex_autodrive"))
    from sdu_apex_autodrive.model_id_frames import (  # type: ignore[no-redef]
        PacketFrameError,
        body_velocity_to_world,
        decode_packet,
    )


MAX_STEERING_RAD = 0.5236


def _read_packets(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no packet rows")
    required = {
        "simulation_time_s", "simulation_physics_step",
        "simulator_position_x", "simulator_position_y",
        "simulator_orientation_euler_z",
        "simulator_orientation_quaternion_x",
        "simulator_orientation_quaternion_y",
        "simulator_orientation_quaternion_z",
        "simulator_orientation_quaternion_w",
        "simulator_linear_velocity_x", "simulator_linear_velocity_y",
        "simulator_angular_velocity_z",
        "applied_throttle_norm",
        "applied_steering_norm",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    return rows


def _deduplicate(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, float]], dict[str, int]]:
    """Keep one row per source physics step and retain ordering diagnostics."""
    unique: list[dict[str, float]] = []
    seen_steps: set[int] = set()
    duplicates = 0
    malformed = 0
    source_order_violations = 0
    source_time_reverse_count = 0
    physics_step_reverse_count = 0
    source_step_gap_count = 0
    previous_time: float | None = None
    previous_step: int | None = None
    for row in rows:
        try:
            packet = decode_packet(row)
        except PacketFrameError:
            malformed += 1
            continue
        step = packet.physics_step
        if step in seen_steps:
            duplicates += 1
            continue
        if previous_time is not None and packet.time_s <= previous_time:
            source_order_violations += 1
            source_time_reverse_count += 1
        if previous_step is not None:
            if step <= previous_step:
                source_order_violations += 1
                physics_step_reverse_count += 1
            elif step > previous_step + 1:
                source_step_gap_count += 1
        seen_steps.add(step)
        current = {
            "time": packet.time_s,
            "physics_step": float(packet.physics_step),
            "x": packet.position_x_m,
            "y": packet.position_y_m,
            "z": packet.position_z_m,
            "yaw": packet.yaw_quaternion_rad,
            "yaw_euler": packet.yaw_euler_rad,
            "vx": packet.body_u_mps,
            "vy": packet.body_v_mps,
            "vz": packet.body_vertical_velocity_mps,
            # IMU.cs maps Unity yaw about its vertical axis to API angular z.
            "yaw_rate": packet.yaw_rate_radps,
            "angular_x": packet.angular_velocity_x_radps,
            "angular_y": packet.angular_velocity_y_radps,
            "angular_z": packet.angular_velocity_z_radps,
            "throttle": packet.throttle_norm,
            "steering": packet.steering_norm,
            "applied_command_sequence": packet.applied_command_sequence,
        }
        unique.append(current)
        previous_time = packet.time_s
        previous_step = step
    return unique, {
        "duplicate_rows": duplicates,
        "malformed_rows": malformed,
        "source_order_violations": source_order_violations,
        "source_time_reverse_count": source_time_reverse_count,
        "physics_step_reverse_count": physics_step_reverse_count,
        "source_step_gap_count": source_step_gap_count,
    }


def _unwrap(previous: float, current: float) -> float:
    delta = (current - previous + math.pi) % (2.0 * math.pi) - math.pi
    return previous + delta


def _body_velocity(row: dict[str, float], twist_frame: str) -> tuple[float, float]:
    if twist_frame == "body":
        return row["vx"], row["vy"]
    c = math.cos(row["yaw"])
    s = math.sin(row["yaw"])
    return c * row["vx"] + s * row["vy"], -s * row["vx"] + c * row["vy"]


def _world_velocity(row: dict[str, float], twist_frame: str) -> tuple[float, float]:
    if twist_frame == "world":
        return row["vx"], row["vy"]
    return body_velocity_to_world(row["yaw"], row["vx"], row["vy"])


def _frame_error(rows: list[dict[str, float]], frame: str) -> float:
    errors: list[float] = []
    for previous, current in zip(rows, rows[1:]):
        dt = current["time"] - previous["time"]
        if not (dt > 1e-6 and dt < 2.0):
            continue
        dx = (current["x"] - previous["x"]) / dt
        dy = (current["y"] - previous["y"]) / dt
        predicted_x, predicted_y = _world_velocity(previous, frame)
        errors.append(math.hypot(dx - predicted_x, dy - predicted_y))
    return float(np.median(errors)) if errors else float("inf")


def _kinematic_diagnostics(rows: list[dict[str, float]]) -> dict[str, object]:
    """Compare raw pose against raw body velocity and raw yaw-rate fields."""
    displacement_errors: list[float] = []
    yaw_rate_errors: list[float] = []
    yaw_rate_axis_errors: dict[str, dict[str, list[float]]] = {
        axis: {"same_sign": [], "negated": []} for axis in ("x", "y", "z")
    }
    quaternion_euler_errors = [
        abs(_unwrap(row["yaw_euler"], row["yaw"]) - row["yaw_euler"])
        for row in rows
    ]
    for previous, current in zip(rows, rows[1:]):
        dt = current["time"] - previous["time"]
        step_delta = int(round(current["physics_step"] - previous["physics_step"]))
        if not (dt > 1.0e-6 and dt < 2.0 and step_delta > 0):
            continue
        midpoint_yaw = previous["yaw"] + 0.5 * (
            _unwrap(previous["yaw"], current["yaw"]) - previous["yaw"])
        midpoint_u = 0.5 * (previous["vx"] + current["vx"])
        midpoint_v = 0.5 * (previous["vy"] + current["vy"])
        predicted_x, predicted_y = body_velocity_to_world(
            midpoint_yaw, midpoint_u, midpoint_v)
        observed_x = (current["x"] - previous["x"]) / dt
        observed_y = (current["y"] - previous["y"]) / dt
        displacement_errors.append(math.hypot(
            observed_x - predicted_x, observed_y - predicted_y))
        yaw_delta = _unwrap(previous["yaw"], current["yaw"]) - previous["yaw"]
        measured_rate = 0.5 * (previous["yaw_rate"] + current["yaw_rate"])
        yaw_rate_errors.append(abs(yaw_delta / dt - measured_rate))
        yaw_rate_derivative = yaw_delta / dt
        for axis in ("x", "y", "z"):
            axis_rate = 0.5 * (
                previous[f"angular_{axis}"] + current[f"angular_{axis}"])
            yaw_rate_axis_errors[axis]["same_sign"].append(
                abs(yaw_rate_derivative - axis_rate))
            yaw_rate_axis_errors[axis]["negated"].append(
                abs(yaw_rate_derivative + axis_rate))

    def stats(values: list[float]) -> dict[str, float | int | None]:
        return {
            "count": len(values),
            "median": float(np.median(values)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": max(values, default=None),
        }

    axis_stats = {
        axis: {sign: stats(values) for sign, values in signs.items()}
        for axis, signs in yaw_rate_axis_errors.items()
    }

    return {
        "transition_count": len(displacement_errors),
        "position_vs_body_velocity_error_mps": stats(displacement_errors),
        "yaw_derivative_vs_angular_z_error_radps": stats(yaw_rate_errors),
        "yaw_derivative_vs_each_angular_axis_error_radps": axis_stats,
        "quaternion_vs_euler_yaw_error_rad": stats(quaternion_euler_errors),
        "yaw_rate_source": "simulator_angular_velocity_z",
        "position_velocity_comparison": "trapezoidal_midpoint_over_source_transition",
    }


def _transitions(rows: list[dict[str, float]], twist_frame: str) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    if not rows:
        return output
    unwrapped_yaw = rows[0]["yaw"]
    rows[0]["yaw_unwrapped"] = unwrapped_yaw
    for previous, current in zip(rows, rows[1:]):
        current["yaw_unwrapped"] = _unwrap(previous["yaw_unwrapped"], current["yaw"])
        dt = current["time"] - previous["time"]
        step_delta = int(round(current["physics_step"] - previous["physics_step"]))
        if not (dt > 1e-6 and step_delta > 0):
            continue
        u0, v0 = _body_velocity(previous, twist_frame)
        u1, v1 = _body_velocity(current, twist_frame)
        applied_command_sequence = current["applied_command_sequence"]
        if applied_command_sequence is None:
            # Gate 0 requires a known command for every fitting transition.
            # Do not infer it from request order or callback arrival time.
            continue
        output.append({
            "simulation_time_k_s": previous["time"],
            "simulation_time_k1_s": current["time"],
            "simulation_physics_step_k": previous["physics_step"],
            "simulation_physics_step_k1": current["physics_step"],
            "physics_step_delta": float(step_delta),
            "dt_sim_s": dt,
            "x_k_m": previous["x"], "y_k_m": previous["y"],
            "yaw_k_rad": previous["yaw_unwrapped"],
            "u_k_mps": u0, "v_k_mps": v0, "r_k_radps": previous["yaw_rate"],
            # The current packet is the post-step state and carries the
            # command consumed for previous -> current.
            "applied_command_sequence_k1": float(applied_command_sequence),
            "throttle_k_norm": current["throttle"],
            "steering_k_rad": current["steering"] * MAX_STEERING_RAD,
            "x_k1_m": current["x"], "y_k1_m": current["y"],
            "yaw_k1_rad": current["yaw_unwrapped"],
            "u_k1_mps": u1, "v_k1_mps": v1, "r_k1_radps": current["yaw_rate"],
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fit_exploratory_longitudinal(rows: list[dict[str, float]]) -> dict[str, object]:
    """Fit a direct-throttle acceleration surface for exploration only."""
    features = np.asarray([
        [1.0, r["u_k_mps"], r["u_k_mps"] ** 2, r["throttle_k_norm"],
         r["u_k_mps"] * r["throttle_k_norm"], r["throttle_k_norm"] ** 2,
         r["steering_k_rad"] ** 2]
        for r in rows
    ])
    target = np.asarray([
        (r["u_k1_mps"] - r["u_k_mps"]) / r["dt_sim_s"] for r in rows
    ])
    if len(rows) < features.shape[1] + 2 or np.linalg.matrix_rank(features) < features.shape[1]:
        raise ValueError("exploratory fit is rank deficient; collect richer excitation")
    # Small ridge penalty limits ill-conditioned extrapolation without hiding
    # the fact that this is a candidate model.
    regularization = 1e-6 * np.eye(features.shape[1])
    coefficients = np.linalg.solve(
        features.T @ features + regularization, features.T @ target)
    prediction = features @ coefficients
    errors = np.abs(prediction - target)
    return {
        "schema_version": 1,
        "status": "exploratory_candidate_not_runtime_validated",
        "boundary": "applied_throttle_to_body_u",
        "feature_names": [
            "bias", "u", "u_squared", "throttle", "u_times_throttle",
            "throttle_squared", "steering_squared",
        ],
        "coefficients": [float(value) for value in coefficients],
        "samples": len(rows),
        "one_step_acceleration_mae_mps2": float(np.mean(errors)),
        "one_step_acceleration_p95_mps2": float(np.percentile(errors, 95)),
    }


def assemble(run_dir: Path, twist_frame: str, allow_non_target_rate: bool,
             fit_exploratory: bool, max_kinematic_error_mps: float,
             max_yaw_rate_error_radps: float = 0.75) -> dict[str, object]:
    packets = _read_packets(run_dir / "simulator_packets.csv")
    unique, counts = _deduplicate(packets)
    if len(unique) < 3:
        raise ValueError("not enough unique source packets")
    dt = np.diff([row["time"] for row in unique])
    positive_dt = dt[dt > 1e-6]
    source_hz = 1.0 / float(np.median(positive_dt)) if len(positive_dt) else 0.0
    frame_errors = {frame: _frame_error(unique, frame) for frame in ("body", "world")}
    selected_frame = twist_frame
    if twist_frame == "auto":
        # The packet contract is known from IMU.cs: x/y are body velocity.
        # Retain the old CLI value for compatibility, but never infer a frame
        # from the smallest residual.
        selected_frame = "body"
    kinematic_diagnostics = _kinematic_diagnostics(unique)
    transitions = _transitions(unique, selected_frame)
    step_deltas = [int(round(row["physics_step_delta"])) for row in transitions]
    source_order_pass = counts["source_order_violations"] == 0
    timing_pass = (36.0 <= source_hz <= 44.0 and source_order_pass and
                   all(delta >= 1 for delta in step_deltas))
    kinematics_error = frame_errors[selected_frame]
    yaw_rate_stats = kinematic_diagnostics[
        "yaw_derivative_vs_angular_z_error_radps"]
    yaw_rate_error = yaw_rate_stats["median"]
    yaw_rate_pass = (yaw_rate_error is not None and
                     yaw_rate_error <= max_yaw_rate_error_radps)
    kinematics_pass = (kinematics_error <= max_kinematic_error_mps and
                       yaw_rate_pass)
    gate_failures: list[str] = []
    if not source_order_pass:
        gate_failures.append("source_order")
    if not timing_pass:
        gate_failures.append("source_rate")
    if kinematics_error > max_kinematic_error_mps:
        gate_failures.append("position_velocity_consistency")
    if not yaw_rate_pass:
        gate_failures.append("yaw_rate_consistency")
    if not source_order_pass:
        status = "rejected_source_order"
    elif not timing_pass and not allow_non_target_rate:
        status = "rejected_non_target_source_rate"
    elif not kinematics_pass:
        status = "rejected_kinematic_consistency"
    else:
        status = "exploratory_only_non_target_rate" if not timing_pass else "timing_and_kinematics_gate_passed"
    output_dir = run_dir / "assembled"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "model_transition_v3.csv", transitions)
    report: dict[str, object] = {
        "schema_version": 3,
        "run_dir": str(run_dir),
        "status": status,
        "quality_gate_failures": gate_failures,
        "raw_packet_count": len(packets),
        "unique_source_packet_count": len(unique),
        "transition_count": len(transitions),
        "duplicate_rows_discarded": counts["duplicate_rows"],
        "malformed_rows_discarded": counts["malformed_rows"],
        "source_rate_hz_median": source_hz,
        "source_dt_s_median": float(np.median(positive_dt)) if len(positive_dt) else None,
        "source_dt_s_p95": float(np.percentile(positive_dt, 95)) if len(positive_dt) else None,
        "physics_step_delta_min": min(step_deltas, default=None),
        "physics_step_delta_max": max(step_deltas, default=None),
        "twist_frame_selected": selected_frame,
        "twist_frame_median_displacement_error_mps": frame_errors,
        "kinematic_consistency": {
            "selected_frame_median_error_mps": kinematics_error,
            "maximum_allowed_median_error_mps": max_kinematic_error_mps,
            "yaw_rate_median_error_radps": yaw_rate_error,
            "maximum_allowed_yaw_rate_median_error_radps": max_yaw_rate_error_radps,
            "yaw_rate_pass": yaw_rate_pass,
            "pass": kinematics_pass,
            "yaw_rate_source": "simulator_angular_velocity_z",
            "diagnostics": kinematic_diagnostics,
        },
        "frame_convention": {
            "position": "raw AutoDRIVE API x/y, copied unchanged by official bridge",
            "linear_velocity": "raw AutoDRIVE API body x/y; x forward, y lateral",
            "yaw": "normalized raw API quaternion, not Euler-only",
            "yaw_rate": "raw API angular velocity z",
            "frame_selection": "declared contract; no residual-based auto-selection",
        },
        "source_order": {
            "pass": source_order_pass,
            "violations": counts["source_order_violations"],
            "time_reverse_count": counts["source_time_reverse_count"],
            "physics_step_reverse_count": counts["physics_step_reverse_count"],
            "physics_step_gap_count": counts["source_step_gap_count"],
        },
        "timing_contract": {
            "expected_source_rate_hz": 40.0,
            "pass": timing_pass,
            "allow_non_target_rate": allow_non_target_rate,
            "raw_rows_are_never_upsampled": True,
        },
        "quality_gate_pass": timing_pass and kinematics_pass,
        "model_boundary": "applied_throttle_to_body_u",
        "transition_schema": {
            "file": "model_transition_v3.csv",
            "control_alignment": (
                "current_packet_applied_command_generates_previous_to_current"
            ),
            "required_control_field": "applied_command_sequence_k1",
            "ambiguous_transitions_discarded": len(transitions) < len(step_deltas),
        },
        "ground_truth_use": "offline_transition_target_only",
    }
    if fit_exploratory:
        if not timing_pass and not allow_non_target_rate:
            raise ValueError(
                "refusing exploratory fit because timing gate failed; "
                "pass --allow-non-target-rate to label it explicitly")
        if not source_order_pass:
            raise ValueError("refusing fit because source packet order is invalid")
        if not kinematics_pass:
            raise ValueError(
                "refusing fit because position and body-velocity fields are "
                "kinematically inconsistent")
        model = _fit_exploratory_longitudinal(transitions)
        model["timing_report"] = report
        (output_dir / "longitudinal_model_candidate.json").write_text(
            json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["fit_status"] = model["status"]
    (output_dir / "timing_quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--twist-frame", choices=("auto", "body", "world"), default="body")
    parser.add_argument("--allow-non-target-rate", action="store_true")
    parser.add_argument("--fit-exploratory", action="store_true")
    parser.add_argument(
        "--max-kinematic-error-mps", type=float, default=0.75,
        help="maximum median position/velocity consistency error for a fit gate")
    parser.add_argument(
        "--max-yaw-rate-error-radps", type=float, default=0.75,
        help="maximum median yaw-derivative/angular-z error for a fit gate")
    args = parser.parse_args()
    print(json.dumps(assemble(
        args.run_dir, args.twist_frame, args.allow_non_target_rate,
        args.fit_exploratory, args.max_kinematic_error_mps,
        args.max_yaw_rate_error_radps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
