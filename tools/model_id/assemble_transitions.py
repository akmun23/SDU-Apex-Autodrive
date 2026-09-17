#!/usr/bin/env python3
"""Assemble causal simulator transitions and gate model-identification data.

This is an offline-only tool.  It consumes the packet table written by
``model_id_timing_recorder`` and never creates runtime ROS messages.  Repeated
bridge observations of one simulator state are collapsed, not counted as
extra samples.  Source gaps are retained in the output and reported.

The simulator telemetry is emitted after the fixed-step update. Therefore the
``applied_command_sequence`` recorded in the current packet identifies the
command that generated the transition from the previous packet state to the
current packet state.  The current packet's ``commanded_*`` fields are the
new command received at the packet boundary and are consumed by the next
transition.  The assembler preserves both facts explicitly; it does not mix
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

# When invoked directly from the repository root, Python can otherwise see
# the outer ROS package directory as a namespace package before it sees the
# actual import package one level below it.
_SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "sdu_apex_autodrive"
if _SOURCE_PACKAGE_ROOT.is_dir():
    sys.path.insert(0, str(_SOURCE_PACKAGE_ROOT))

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
# A 40 Hz source transition should be approximately 25 ms.  Keep the same
# cadence window as the runtime bridge: a slow or over-rate stream must be
# rejected rather than hidden by a median-rate calculation.
MIN_SOURCE_TRANSITION_DT_S = 0.015
MAX_SOURCE_TRANSITION_DT_S = 0.035
POSITION_DISCONTINUITY_THRESHOLD_M = 1.5
YAW_DISCONTINUITY_THRESHOLD_RAD = 1.0
ENCODER_WHEEL_RADIUS_M = 0.059
# The model-ID player is an open-ground diagnostic scene.  A car leaving the
# plane can still produce perfectly regular source packets, so cadence and
# kinematic checks alone are not sufficient to accept its falling tail.
MAX_OFF_PLANE_Z_DEVIATION_M = 1.0


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
        "commanded_throttle_norm", "commanded_steering_norm",
        "simulator_feedback_steering_norm", "sent_reset",
        "simulator_encoder_angles_left", "simulator_encoder_angles_right",
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
            "commanded_throttle": _optional_float(
                row, "commanded_throttle_norm"),
            "commanded_steering": _optional_float(
                row, "commanded_steering_norm"),
            "feedback_steering": _optional_float(
                row, "simulator_feedback_steering_norm"),
            "encoder_left": _optional_float(
                row, "simulator_encoder_angles_left"),
            "encoder_right": _optional_float(
                row, "simulator_encoder_angles_right"),
            "sent_reset": _parse_bool(row.get("sent_reset")),
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


def _optional_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _parse_bool(raw: object) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _mark_segments(rows: list[dict[str, float]]) -> dict[str, object]:
    """Mark reset/teleport boundaries without inventing source transitions."""
    reset_epoch = 0
    segment_id = 0
    boundary_reasons: list[dict[str, object]] = []
    previous_reset = False
    last_reset_boundary_index: int | None = None
    for index, row in enumerate(rows):
        reason: str | None = None
        reset_level = bool(row.get("sent_reset", False))
        if index > 0:
            previous = rows[index - 1]
            if reset_level and not previous_reset:
                reason = "reset_command_rising_edge"
            else:
                position_jump = math.hypot(
                    row["x"] - previous["x"], row["y"] - previous["y"])
                yaw_jump = abs(_unwrap(previous["yaw"], row["yaw"]) -
                               previous["yaw"])
                if position_jump > POSITION_DISCONTINUITY_THRESHOLD_M:
                    if (last_reset_boundary_index is not None and
                            index - last_reset_boundary_index <= 2):
                        # A reset command and the following teleport can be
                        # reported as two source packets. The reset already
                        # created the new segment; only suppress the second
                        # crossing transition as a boundary marker.
                        reason = "reset_teleport"
                    else:
                        reason = "position_discontinuity"
                elif yaw_jump > YAW_DISCONTINUITY_THRESHOLD_RAD:
                    reason = "yaw_discontinuity"
        if reason is not None and reason != "reset_teleport":
            reset_epoch += 1
            segment_id += 1
            boundary_reasons.append({
                "packet_index": index,
                "physics_step": row["physics_step"],
                "reason": reason,
            })
            if reason == "reset_command_rising_edge":
                last_reset_boundary_index = index
        row["reset_epoch"] = float(reset_epoch)
        row["segment_id"] = float(segment_id)
        row["segment_boundary_before"] = reason
        previous_reset = reset_level
    return {
        "reset_epoch_count": reset_epoch,
        "segment_count": segment_id + (1 if rows else 0),
        "boundary_count": len(boundary_reasons),
        "boundaries": boundary_reasons,
        "position_threshold_m": POSITION_DISCONTINUITY_THRESHOLD_M,
        "yaw_threshold_rad": YAW_DISCONTINUITY_THRESHOLD_RAD,
    }


def _derive_wheel_speeds(rows: list[dict[str, float]]) -> None:
    """Derive source-time wheel surface speed without crossing boundaries."""
    previous: dict[str, float] | None = None
    for row in rows:
        row["wheel_speed_mps"] = None
        if previous is not None and previous["segment_id"] == row["segment_id"]:
            dt = row["time"] - previous["time"]
            if (MIN_SOURCE_TRANSITION_DT_S <= dt <= MAX_SOURCE_TRANSITION_DT_S and
                    row["encoder_left"] is not None and
                    row["encoder_right"] is not None and
                    previous["encoder_left"] is not None and
                    previous["encoder_right"] is not None):
                left_rate = (row["encoder_left"] - previous["encoder_left"]) / dt
                right_rate = (row["encoder_right"] - previous["encoder_right"]) / dt
                row["wheel_speed_mps"] = ENCODER_WHEEL_RADIUS_M * 0.5 * (
                    left_rate + right_rate)
        previous = row


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
        if (previous.get("segment_id") != current.get("segment_id") or
                current.get("segment_boundary_before") is not None):
            continue
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
        if (previous.get("segment_id") != current.get("segment_id") or
                current.get("segment_boundary_before") is not None):
            continue
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


def _scene_validity_diagnostics(rows: list[dict[str, float]]) -> dict[str, object]:
    """Reject an open-plane capture after the vehicle leaves the ground.

    This is an offline data-quality gate.  It does not infer or alter vehicle
    physics; it prevents a regular 40 Hz stream from hiding a rollover or an
    edge departure inside the identification dataset.
    """
    z_values = [float(row["z"]) for row in rows if math.isfinite(row["z"])]
    if not z_values:
        return {
            "pass": False,
            "reason": "no_finite_ground_truth_z",
            "initial_z_m": None,
            "maximum_abs_z_deviation_m": None,
            "maximum_abs_vertical_velocity_mps": None,
            "maximum_allowed_abs_z_deviation_m": MAX_OFF_PLANE_Z_DEVIATION_M,
        }
    initial_z = z_values[0]
    deviations = [abs(value - initial_z) for value in z_values]
    vertical_speeds = [abs(float(row["vz"])) for row in rows
                       if math.isfinite(row["vz"])]
    maximum_deviation = max(deviations)
    return {
        "pass": maximum_deviation <= MAX_OFF_PLANE_Z_DEVIATION_M,
        "reason": None if maximum_deviation <= MAX_OFF_PLANE_Z_DEVIATION_M
        else "ground_truth_left_open_plane",
        "initial_z_m": initial_z,
        "maximum_abs_z_deviation_m": maximum_deviation,
        "maximum_abs_vertical_velocity_mps": max(vertical_speeds, default=None),
        "maximum_allowed_abs_z_deviation_m": MAX_OFF_PLANE_Z_DEVIATION_M,
    }


def _transitions(rows: list[dict[str, float]], twist_frame: str,
                 max_transition_dt_s: float) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    if not rows:
        return output
    unwrapped_yaw = rows[0]["yaw"]
    rows[0]["yaw_unwrapped"] = unwrapped_yaw
    for previous, current in zip(rows, rows[1:]):
        current["yaw_unwrapped"] = _unwrap(previous["yaw_unwrapped"], current["yaw"])
        dt = current["time"] - previous["time"]
        step_delta = int(round(current["physics_step"] - previous["physics_step"]))
        if not (MIN_SOURCE_TRANSITION_DT_S <= dt <= max_transition_dt_s and
                step_delta > 0):
            continue
        # A reset/teleport is a hard rollout boundary. Retain the packet on
        # either side, but never fit the transition crossing the boundary.
        if (previous["segment_id"] != current["segment_id"] or
                current["segment_boundary_before"] is not None):
            continue
        u0, v0 = _body_velocity(previous, twist_frame)
        u1, v1 = _body_velocity(current, twist_frame)
        applied_command_sequence = current["applied_command_sequence"]
        if (applied_command_sequence is None or
                previous["commanded_throttle"] is None or
                previous["commanded_steering"] is None or
                current["commanded_throttle"] is None or
                current["commanded_steering"] is None or
                current["feedback_steering"] is None):
            # Gate 0 requires a known command for every fitting transition.
            # Do not infer it from request order or callback arrival time, and
            # do not silently recover an ambiguous steering unit.
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
            # ``commanded_*_k1`` is retained as the raw post-step command
            # for provenance.  The preceding packet command is the command
            # consumed by the fixed-step interval represented by this row.
            # ``applied_steering_rad_k1`` is the post-controller effective
            # angle in the current packet; it is the correct state endpoint.
            # The feedback field is retained separately because the source
            # GUI getter is sampled before the controller's steering update.
            "applied_command_sequence_k1": float(applied_command_sequence),
            "transition_throttle_norm_k": previous["commanded_throttle"],
            "transition_steering_norm_k": previous["commanded_steering"],
            "commanded_throttle_norm_k1": current["commanded_throttle"],
            "commanded_steering_norm_k1": current["commanded_steering"],
            "applied_throttle_norm_k1": current["throttle"],
            "applied_steering_rad_k1": current["steering"],
            "simulator_feedback_steering_rad_k1": current["feedback_steering"],
            "steering_state_rad_k": previous["steering"],
            "steering_feedback_rad_k": previous["feedback_steering"],
            "wheel_speed_mps_k1": current["wheel_speed_mps"],
            "encoder_left_rad_k1": current["encoder_left"],
            "encoder_right_rad_k1": current["encoder_right"],
            "reset_epoch": current["reset_epoch"],
            "segment_id": current["segment_id"],
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


def assemble(run_dir: Path, twist_frame: str, allow_non_target_rate: bool,
             max_kinematic_error_mps: float,
             max_yaw_rate_error_radps: float = 0.75) -> dict[str, object]:
    packets = _read_packets(run_dir / "simulator_packets.csv")
    unique, counts = _deduplicate(packets)
    if len(unique) < 3:
        raise ValueError("not enough unique source packets")
    segment_report = _mark_segments(unique)
    _derive_wheel_speeds(unique)
    dt = np.diff([row["time"] for row in unique])
    positive_dt = dt[dt > 1e-6]
    source_dt_gap_count = int(np.count_nonzero(
        positive_dt > MAX_SOURCE_TRANSITION_DT_S))
    source_dt_cadence_violation_count = int(np.count_nonzero(
        (positive_dt < MIN_SOURCE_TRANSITION_DT_S) |
        (positive_dt > MAX_SOURCE_TRANSITION_DT_S)))
    source_dt_max = float(np.max(positive_dt)) if len(positive_dt) else None
    source_hz = 1.0 / float(np.median(positive_dt)) if len(positive_dt) else 0.0
    frame_errors = {frame: _frame_error(unique, frame) for frame in ("body", "world")}
    selected_frame = twist_frame
    if twist_frame == "auto":
        # The packet contract is known from IMU.cs: x/y are body velocity.
        # Retain the old CLI value for compatibility, but never infer a frame
        # from the smallest residual.
        selected_frame = "body"
    kinematic_diagnostics = _kinematic_diagnostics(unique)
    scene_validity = _scene_validity_diagnostics(unique)
    transitions = _transitions(
        unique, selected_frame, MAX_SOURCE_TRANSITION_DT_S)
    step_deltas = [int(round(row["physics_step_delta"])) for row in transitions]
    source_order_pass = counts["source_order_violations"] == 0
    timing_pass = (36.0 <= source_hz <= 44.0 and source_order_pass and
                   source_dt_cadence_violation_count == 0 and
                   all(delta >= 1 for delta in step_deltas))
    kinematics_error = frame_errors[selected_frame]
    yaw_rate_stats = kinematic_diagnostics[
        "yaw_derivative_vs_angular_z_error_radps"]
    yaw_rate_error = yaw_rate_stats["median"]
    yaw_rate_pass = (yaw_rate_error is not None and
                     yaw_rate_error <= max_yaw_rate_error_radps)
    kinematics_pass = (kinematics_error <= max_kinematic_error_mps and
                       yaw_rate_pass)
    scene_validity_pass = bool(scene_validity["pass"])
    gate_failures: list[str] = []
    if not source_order_pass:
        gate_failures.append("source_order")
    if not timing_pass:
        gate_failures.append("source_rate")
    if source_dt_gap_count:
        gate_failures.append("source_time_gap")
    if source_dt_cadence_violation_count:
        gate_failures.append("source_cadence")
    if kinematics_error > max_kinematic_error_mps:
        gate_failures.append("position_velocity_consistency")
    if not yaw_rate_pass:
        gate_failures.append("yaw_rate_consistency")
    if not scene_validity_pass:
        gate_failures.append("off_plane_motion")
    if not source_order_pass:
        status = "rejected_source_order"
    elif not timing_pass and not allow_non_target_rate:
        status = "rejected_non_target_source_rate"
    elif not kinematics_pass:
        status = "rejected_kinematic_consistency"
    elif not scene_validity_pass:
        status = "rejected_off_plane_motion"
    else:
        status = "exploratory_only_non_target_rate" if not timing_pass else "timing_and_kinematics_gate_passed"
    output_dir = run_dir / "assembled"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "model_transition_v4.csv", transitions)
    report: dict[str, object] = {
        "schema_version": 4,
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
        "source_dt_s_max": source_dt_max,
        "source_dt_gap_count": source_dt_gap_count,
        "source_dt_cadence_violation_count": source_dt_cadence_violation_count,
        "minimum_allowed_source_transition_dt_s": MIN_SOURCE_TRANSITION_DT_S,
        "maximum_allowed_source_transition_dt_s": MAX_SOURCE_TRANSITION_DT_S,
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
        "scene_validity": scene_validity,
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
        "segments": segment_report,
        "timing_contract": {
            "expected_source_rate_hz": 40.0,
            "pass": timing_pass,
            "allow_non_target_rate": allow_non_target_rate,
            "raw_rows_are_never_upsampled": True,
            "source_cadence_window_s": [
                MIN_SOURCE_TRANSITION_DT_S, MAX_SOURCE_TRANSITION_DT_S],
            "source_cadence_violations_rejected": True,
        },
        "quality_gate_pass": timing_pass and kinematics_pass and
        scene_validity_pass,
        "model_boundary": "applied_throttle_to_body_u",
        "transition_schema": {
            "file": "model_transition_v4.csv",
            "version": 4,
            "fields": [
                "transition_throttle_norm_k",
                "transition_steering_norm_k",
                "commanded_throttle_norm_k1",
                "commanded_steering_norm_k1",
                "applied_throttle_norm_k1",
                "applied_steering_rad_k1",
                "simulator_feedback_steering_rad_k1",
                "steering_state_rad_k",
                "steering_feedback_rad_k",
                "wheel_speed_mps_k1",
                "reset_epoch",
                "segment_id",
            ],
            "steering_contract": (
                "commanded_steering_norm_k1 is normalized input; "
                "applied_steering_rad_k1 and "
                "simulator_feedback_steering_rad_k1 are already physical "
                "radians copied from the source packet"
            ),
            "control_alignment": (
                "current_packet_applied_command_generates_previous_to_current; "
                "previous_packet_command_is_the_transition_input"
            ),
            "transition_input_fields": {
                "steering": "transition_steering_norm_k",
                "throttle": "transition_throttle_norm_k",
                "steering_state_k": "steering_state_rad_k",
                "steering_state_k1": "applied_steering_rad_k1",
                "pre_update_feedback_k": "steering_feedback_rad_k",
            },
            "required_control_field": "applied_command_sequence_k1",
            "ambiguous_transitions_discarded": False,
            "discarded_transition_count": (
                len(unique) - 1 - len(transitions)),
        },
        "ground_truth_use": "offline_transition_target_only",
    }
    (output_dir / "timing_quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--twist-frame", choices=("auto", "body", "world"), default="body")
    parser.add_argument("--allow-non-target-rate", action="store_true")
    parser.add_argument(
        "--max-kinematic-error-mps", type=float, default=0.75,
        help="maximum median position/velocity consistency error for a fit gate")
    parser.add_argument(
        "--max-yaw-rate-error-radps", type=float, default=0.75,
        help="maximum median yaw-derivative/angular-z error for a fit gate")
    args = parser.parse_args()
    print(json.dumps(assemble(
        args.run_dir, args.twist_frame, args.allow_non_target_rate,
        args.max_kinematic_error_mps,
        args.max_yaw_rate_error_radps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
