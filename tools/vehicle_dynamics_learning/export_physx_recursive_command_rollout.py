#!/usr/bin/env python3
"""Export one initial state and a command-only sequence for recursive PhysX.

The JSON passed to the native plant contains no future pose, body velocity,
wheel speed, slip, or contact measurements. Recorded WheelCollider steering,
motor-torque, and brake-torque properties are the known inputs; CSV truth is
kept outside the plant input and is used only by the offline scorer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning import evaluate_physx_no_drive_plant_one_step as replay
from tools.vehicle_dynamics_learning.physx_vehicle_source import (
    physx_jounce_from_wheel_pose_actor_body,
    physx_wheel_rotation_angle_from_pose_actor_body,
)
from tools.vehicle_dynamics_learning.unity_vehicle_controller_source import (
    center_angle_from_front_ackermann_deg,
    front_ackermann_angles_deg,
    step_autonomous_steering_deg,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    wheel_actuation,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_mechanics_dynamic_validation_20261013_r12g/"
    "unity_source_capture/wheel_dynamics_unity_mechanics_dynamic_validation_r12g.csv")
DEFAULT_SHAPES = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "vehicle_compound_shapes_clearance_r07_20261016.json")
FIXED_DT_S = replay.FIXED_DT_S


def _wheel_pose(row: pd.Series, wheel: str) -> dict:
    return {
        "position_body_m": [
            float(getattr(row, f"{wheel}_wheel_pose_body_{axis}"))
            for axis in "xyz"],
        "rotation_body_xyzw": [
            float(getattr(row, f"{wheel}_wheel_pose_q{axis}"))
            for axis in "xyzw"],
    }


def _controller_boundary_wheel_commands(
        rows: pd.DataFrame, previous: pd.Series,
        ticks_per_command: int) -> tuple[list[list[dict]], dict]:
    """Expand held 40 Hz VehicleController commands through source code."""
    if (ticks_per_command <= 0 or len(rows) == 0
            or len(rows) % ticks_per_command != 0):
        raise ValueError(
            "controller-boundary export requires whole command intervals")
    center_angle_deg = center_angle_from_front_ackermann_deg(
        float(previous["fl_steer_deg"]), float(previous["fr_steer_deg"]))
    last_angle_deg = center_angle_deg
    commands: list[list[dict]] = []
    steering_errors: list[float] = []
    drive_errors: list[float] = []
    brake_errors: list[float] = []
    for block_start in range(0, len(rows), ticks_per_command):
        block = rows.iloc[block_start:block_start + ticks_per_command]
        steering_values = block["steering_command"].to_numpy(dtype=np.float64)
        throttle_values = block["throttle_command"].to_numpy(dtype=np.float64)
        if (not np.isfinite(steering_values).all()
                or not np.isfinite(throttle_values).all()
                or np.ptp(steering_values) > 1.0e-7
                or np.ptp(throttle_values) > 1.0e-7):
            raise ValueError(
                "captured controller commands are not held over the declared "
                "physics-tick interval")
        steering_command = float(steering_values[0])
        throttle_command = float(throttle_values[0])
        actuation = wheel_actuation(throttle_command)
        for row in block.itertuples(index=False):
            center_angle_deg = step_autonomous_steering_deg(
                center_angle_deg, last_angle_deg, steering_command, FIXED_DT_S)
            last_angle_deg = center_angle_deg
            front_left, front_right = front_ackermann_angles_deg(
                center_angle_deg)
            predicted_steer_deg = (front_left, front_right, 0.0, 0.0)
            wheel_commands = []
            for index, wheel in enumerate(replay.WHEELS):
                observed_steer = float(getattr(row, f"{wheel}_steer_deg"))
                observed_drive = float(getattr(
                    row, f"{wheel}_motor_torque_nm"))
                observed_brake = float(getattr(
                    row, f"{wheel}_brake_torque_nm"))
                steering_errors.append(
                    predicted_steer_deg[index] - observed_steer)
                drive_errors.append(
                    actuation.motor_torque_nm[index] - observed_drive)
                brake_errors.append(
                    actuation.brake_torque_nm[index] - observed_brake)
                wheel_commands.append({
                    "steer_angle_rad": math.radians(
                        predicted_steer_deg[index]),
                    "drive_torque_nm": actuation.motor_torque_nm[index],
                    "brake_torque_nm": actuation.brake_torque_nm[index],
                })
            commands.append(wheel_commands)

    summary = {
        "control_rate_hz": 1.0 / (ticks_per_command * FIXED_DT_S),
        "physics_ticks_per_command": ticks_per_command,
        "held_command_intervals": len(rows) // ticks_per_command,
        "initial_center_steering_deg_from_prior_wheel_feedback": (
            center_angle_from_front_ackermann_deg(
                float(previous["fl_steer_deg"]),
                float(previous["fr_steer_deg"]))),
        "max_abs_wheel_steering_reconstruction_error_deg": float(
            np.max(np.abs(steering_errors))),
        "max_abs_wheel_drive_torque_reconstruction_error_nm": float(
            np.max(np.abs(drive_errors))),
        "max_abs_wheel_brake_torque_reconstruction_error_nm": float(
            np.max(np.abs(brake_errors))),
        "source_equations": [
            "VehicleController.Steer() at 1 kHz with captured 40 Hz-held AutonomousSteering",
            "VehicleController front Ackermann equations",
            "VehicleController.Drive(): CAWD quarter torque and CAWB exact-zero brake branch",
        ],
    }
    if (summary["max_abs_wheel_steering_reconstruction_error_deg"] > 0.02
            or summary[
                "max_abs_wheel_drive_torque_reconstruction_error_nm"] > 0.02
            or summary[
                "max_abs_wheel_brake_torque_reconstruction_error_nm"] > 0.02):
        raise ValueError(
            "Unity controller equations do not reproduce captured per-wheel "
            f"actuation: {summary}")
    return commands, summary


def export_rollout(capture: Path, row_index: int, count: int,
                   shape_audit: Path, *,
                   controller_command_ticks: int | None = None) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    metadata = replay._metadata(capture)
    frame = pd.read_csv(capture, comment="#", usecols=replay._columns())
    if row_index < 1 or row_index + count >= len(frame):
        raise ValueError("rollout needs an existing prior sample and next-state labels")
    selected_with_previous = frame.iloc[row_index - 1:row_index + count + 1]
    selected_times = selected_with_previous["sim_time_s"].to_numpy(
        dtype=np.float64)
    if np.any(np.abs(np.diff(selected_times) - FIXED_DT_S) > 5.0e-5):
        raise ValueError("selected rows are not contiguous 1 ms physics samples")
    selected = selected_with_previous.iloc[1:]
    times = selected["sim_time_s"].to_numpy(dtype=np.float64)
    assert len(times) == count + 1

    previous = selected_with_previous.iloc[0]
    first = selected.iloc[0]
    previous_position = np.asarray(
        [previous[f"pos_world_{axis}_m"] for axis in "xyz"], dtype=float)
    first_position = np.asarray(
        [first[f"pos_world_{axis}_m"] for axis in "xyz"], dtype=float)
    if float(np.linalg.norm(first_position - previous_position)) > 0.05:
        raise ValueError("initial state begins across a reset/teleport boundary")
    previous_rotation = np.asarray(
        [previous[f"rot_world_{axis}"] for axis in "xyzw"], dtype=float)
    first_rotation = np.asarray(
        [first[f"rot_world_{axis}"] for axis in "xyzw"], dtype=float)
    previous_rotation /= np.linalg.norm(previous_rotation)
    first_rotation /= np.linalg.norm(first_rotation)
    pose_dot = min(1.0, abs(float(np.dot(previous_rotation, first_rotation))))
    if 2.0 * math.acos(pose_dot) > 0.1:
        raise ValueError("initial state begins across a reset rotation jump")

    state, _ = replay._state_and_wheels(first, metadata)
    initial_wheels = []
    for index, wheel in enumerate(replay.WHEELS):
        steer = math.radians(float(first[f"{wheel}_steer_deg"]))
        pose = _wheel_pose(first, wheel)
        angle, orientation_residual = physx_wheel_rotation_angle_from_pose_actor_body(
            tuple(pose["rotation_body_xyzw"]), steer)
        config = replay._wheel_configuration(metadata, index, steer, 0.0, 0.0)
        jounce = physx_jounce_from_wheel_pose_actor_body(
            tuple(pose["position_body_m"]), replay.RIGIDBODY_CENTER_OF_MASS_BODY_M,
            config.wheel_center_offset_com_body_m,
            config.suspension_direction_body)
        if orientation_residual > 0.01:
            raise ValueError(
                f"{wheel} initial wheel-pose factorization residual is "
                f"{orientation_residual:.6g} rad")
        initial_wheels.append({
            "radius_m": config.wheel_radius_m,
            "width_m": 0.1,
            "mass_kg": config.wheel_mass_kg,
            "moment_of_inertia_kgm2": config.wheel_inertia_kgm2,
            "wheel_damping_rate_nms": config.wheel_damping_rate_nms,
            "max_compression_m": config.max_compression_m,
            "max_droop_m": config.max_droop_m,
            "spring_rate_npm": config.spring_rate_npm,
            "damper_rate_nspm": config.damper_rate_nspm,
            "sprung_mass_kg": config.sprung_mass_kg,
            "wheel_center_offset_com_body_m":
                list(config.wheel_center_offset_com_body_m),
            "force_application_offset_com_body_m":
                list(config.force_application_offset_com_body_m),
            "suspension_direction_body": list(config.suspension_direction_body),
            "forward_curve": list(config.forward_curve),
            "sideways_curve": list(config.sideways_curve),
            "wheel_omega_radps": state.wheel_omega_radps[index],
            "wheel_rotation_angle_rad": angle,
            "initial_pose_jounce_m": jounce,
            "initial_pose_factorization_residual_rad": orientation_residual,
        })

    if controller_command_ticks is None:
        wheel_commands = [[{
            "steer_angle_rad": math.radians(
                float(getattr(row, f"{wheel}_steer_deg"))),
            "drive_torque_nm": float(
                getattr(row, f"{wheel}_motor_torque_nm")),
            "brake_torque_nm": float(
                getattr(row, f"{wheel}_brake_torque_nm")),
        } for wheel in replay.WHEELS]
            for row in selected.iloc[:-1].itertuples(index=False)]
        command_summary = None
    else:
        wheel_commands, command_summary = _controller_boundary_wheel_commands(
            selected.iloc[:-1], previous, controller_command_ticks)

    steps = [{
            "capture_row": row_index + offset,
            "time_s": float(row.sim_time_s),
            "wheel_commands": wheel_commands[offset],
        }
        for offset, row in enumerate(selected.iloc[:-1].itertuples(index=False))]

    rotation = [float(first[f"rot_world_{axis}"]) for axis in "xyzw"]
    rotation_norm = math.sqrt(sum(value * value for value in rotation))
    rotation = [value / rotation_norm for value in rotation]
    initial = {
        "capture_row": row_index,
        "time_s": float(first.sim_time_s),
        "fixed_dt_s": FIXED_DT_S,
        "body": {
            "position_world_m": [float(first[f"pos_world_{axis}_m"])
                                 for axis in "xyz"],
            "rotation_world_xyzw": rotation,
            "linear_velocity_world_mps": list(state.linear_velocity_world_mps),
            "angular_velocity_world_radps": list(state.angular_velocity_world_radps),
            "mass_kg": float(metadata["mass_kg"]),
            "inertia_diagonal_kgm2": replay._vector(
                metadata, "inertia_tensor_body", 3),
            "inertia_rotation_xyzw": replay._vector(
                metadata, "inertia_tensor_rotation_body", 4),
            "center_of_mass_body_m": list(replay.RIGIDBODY_CENTER_OF_MASS_BODY_M),
        },
        "wheel_shapes": [_wheel_pose(first, wheel) for wheel in replay.WHEELS],
        "wheel_inputs": initial_wheels,
    }
    return {
        "mode": "recursive_command_only",
        "capture": str(capture.resolve().relative_to(ROOT)),
        "shape_audit": str(shape_audit.resolve().relative_to(ROOT)),
        "initial": initial,
        "steps": steps,
        "controller_boundary_reconstruction": command_summary,
        "input_contract": {
            "future_state_measurements_included": False,
            "known_inputs": (
                "recorded per-wheel WheelCollider outputs"
                if command_summary is None else
                "recorded 40 Hz-held VehicleController steering/throttle commands"),
            "input_note": (
                "Recorded 1 kHz per-wheel WheelCollider outputs isolate plant rollout from upstream controller/actuator mapping."
                if command_summary is None else
                "The native input contains only 40 Hz-held steering/throttle commands expanded through public VehicleController source equations; no future captured per-wheel angles or torques are included."),
            "label_rows": [row_index + 1, row_index + count],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--row", type=int, required=True,
                        help="initial-state row; prior row is checked for continuity")
    parser.add_argument("--count", type=int, default=500,
                        help="number of 1 ms command steps")
    parser.add_argument("--shape-audit", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument(
        "--controller-command-ticks", type=int,
        help="expand held controller commands through VehicleController source (e.g. 25 ticks at 40 Hz)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = export_rollout(
        args.capture, args.row, args.count, args.shape_audit,
        controller_command_ticks=args.controller_command_ticks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {args.count} commands from row {args.row}; "
          "no future state measurements are in plant input")


if __name__ == "__main__":
    main()
