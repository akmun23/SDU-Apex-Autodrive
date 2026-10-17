#!/usr/bin/env python3
"""Export a recursive source-plant run from the documented reset state.

Unlike the teacher-forced parity exporter, this input contains no captured
body/wheel state. It starts from the open-plane reset pose, zero rigid-body
and wheel velocity, static Unity vehicle parameters, and the captured held
controller commands expanded through the source VehicleController equations.
Capture states are used only by the command-equation consistency check and
later scoring; they are never copied into the native plant input.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from tools.vehicle_dynamics_learning import evaluate_physx_no_drive_plant_one_step as replay
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    RIGIDBODY_CENTER_OF_MASS_BODY_M,
    wheel_actuation,
)
from tools.vehicle_dynamics_learning.unity_vehicle_controller_source import (
    front_ackermann_angles_deg,
    step_autonomous_steering_deg,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
DEFAULT_SHAPES = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "vehicle_compound_shapes_clearance_r07_20261016.json")
RESET_POSITION_WORLD_M = (0.0, 0.01, 0.0)
CONTROL_TICKS = 25
OUTPUT_STRIDE_TICKS = 25
def export_reset_rollout(capture: Path, count: int, shape_audit: Path,
                         start_row: int = 0) -> dict:
    if count < CONTROL_TICKS or count % CONTROL_TICKS:
        raise ValueError("count must be a positive multiple of 25 physics ticks")
    if start_row < 0:
        raise ValueError("start_row must be nonnegative")
    metadata = replay._metadata(capture)
    frame = pd.read_csv(capture, comment="#", usecols=replay._columns(),
                        nrows=start_row + count + 1)
    if len(frame) < start_row + count + 1:
        raise ValueError("capture does not contain the requested rollout span")
    frame = frame.iloc[start_row:start_row + count + 1].reset_index(drop=True)
    time = frame["sim_time_s"].to_numpy(dtype=float)
    if any(abs((time[i + 1] - time[i]) - replay.FIXED_DT_S) > 5.0e-5
           for i in range(count)):
        raise ValueError("selected capture rows are not contiguous 1 ms ticks")

    # Reconstruct at every 1 ms source step. This preserves the capture's
    # observed command-update phase instead of assuming updates begin on a
    # multiple of 25 ms after reset. Captured per-wheel actuator values are
    # used only to audit the source equations; they do not enter commands.
    commands = []
    center_angle_deg = 0.0
    last_angle_deg = 0.0
    steer_errors = []
    drive_errors = []
    brake_errors = []
    for row in frame.iloc[:count].itertuples(index=False):
        center_angle_deg = step_autonomous_steering_deg(
            center_angle_deg, last_angle_deg, float(row.steering_command),
            replay.FIXED_DT_S)
        last_angle_deg = center_angle_deg
        front_left, front_right = front_ackermann_angles_deg(center_angle_deg)
        actuation = wheel_actuation(float(row.throttle_command))
        expected_steer = (front_left, front_right, 0.0, 0.0)
        wheel_commands = []
        for index, wheel in enumerate(replay.WHEELS):
            steer_errors.append(expected_steer[index] - float(
                getattr(row, f"{wheel}_steer_deg")))
            drive_errors.append(actuation.motor_torque_nm[index] - float(
                getattr(row, f"{wheel}_motor_torque_nm")))
            brake_errors.append(actuation.brake_torque_nm[index] - float(
                getattr(row, f"{wheel}_brake_torque_nm")))
            wheel_commands.append({
                "steer_angle_rad": math.radians(expected_steer[index]),
                "drive_torque_nm": actuation.motor_torque_nm[index],
                "brake_torque_nm": actuation.brake_torque_nm[index],
            })
        commands.append(wheel_commands)
    reconstruction = {
        "simulation_step_s": replay.FIXED_DT_S,
        "command_update_phase": "captured source command values, stepped each 1 ms",
        "max_abs_wheel_steering_reconstruction_error_deg": max(
            abs(value) for value in steer_errors),
        "max_abs_wheel_drive_torque_reconstruction_error_nm": max(
            abs(value) for value in drive_errors),
        "max_abs_wheel_brake_torque_reconstruction_error_nm": max(
            abs(value) for value in brake_errors),
    }
    if (reconstruction["max_abs_wheel_steering_reconstruction_error_deg"]
            > 0.02
            or reconstruction[
                "max_abs_wheel_drive_torque_reconstruction_error_nm"] > 0.02
            or reconstruction[
                "max_abs_wheel_brake_torque_reconstruction_error_nm"] > 0.02):
        raise ValueError(
            "Unity controller equations do not reproduce captured per-wheel "
            f"actuation: {reconstruction}")

    wheel_inputs = []
    wheel_shapes = []
    for index in range(4):
        config = replay._wheel_configuration(metadata, index, 0.0, 0.0, 0.0)
        wheel_inputs.append({
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
            "wheel_omega_radps": 0.0,
            "wheel_rotation_angle_rad": 0.0,
            "initial_pose_jounce_m": 0.0,
        })
        offset = config.wheel_center_offset_com_body_m
        wheel_shapes.append({
            "position_body_m": [
                RIGIDBODY_CENTER_OF_MASS_BODY_M[axis] + offset[axis]
                for axis in range(3)],
            "rotation_body_xyzw": [0.0, 0.0, 0.0, 1.0],
        })

    body = {
        "position_world_m": list(RESET_POSITION_WORLD_M),
        "rotation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity_world_mps": [0.0, 0.0, 0.0],
        "angular_velocity_world_radps": [0.0, 0.0, 0.0],
        "mass_kg": float(metadata["mass_kg"]),
        "inertia_diagonal_kgm2": replay._vector(
            metadata, "inertia_tensor_body", 3),
        "inertia_rotation_xyzw": replay._vector(
            metadata, "inertia_tensor_rotation_body", 4),
        "center_of_mass_body_m": list(RIGIDBODY_CENTER_OF_MASS_BODY_M),
    }
    return {
        "mode": "recursive_command_only",
        "output_every_n_steps": OUTPUT_STRIDE_TICKS,
        "capture": str(capture.resolve().relative_to(ROOT)),
        "start_capture_row": start_row,
        "shape_audit": str(shape_audit.resolve().relative_to(ROOT)),
        "initial": {
            "capture_row": 0,
            "time_s": 0.0,
            "fixed_dt_s": replay.FIXED_DT_S,
            "body": body,
            "wheel_shapes": wheel_shapes,
            "wheel_inputs": wheel_inputs,
        },
        "steps": [{
            "capture_row": start_row + index,
            "time_s": float(time[index]),
            "wheel_commands": commands[index],
        } for index in range(count)],
        "controller_boundary_reconstruction": reconstruction,
        "input_contract": {
            "initialization": (
                "documented open-plane reset pose; zero body and wheel rates; "
                "serialized vehicle parameters; zero suspension jounce"),
            "future_state_measurements_included": False,
            "known_inputs": (
                "captured 40 Hz-held steering/throttle commands expanded through "
                "public Unity VehicleController source equations"),
            "scoring_output_rate_hz": 40,
            "truth_or_sensor_values_used_to_initialize_state": False,
            "scoring_labels": "not included; read separately by the evaluator",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--count", type=int, required=True,
                        help="1 ms physics ticks; must be divisible by 25")
    parser.add_argument("--start-row", type=int, default=0,
                        help="capture row at which the documented reset state begins")
    parser.add_argument("--shape-audit", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = export_reset_rollout(args.capture, args.count, args.shape_audit,
                                start_row=args.start_row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n",
                           encoding="utf-8")
    print(f"wrote reset-initialized command-only rollout: "
          f"{args.count} ticks; no captured state enters plant input")


if __name__ == "__main__":
    main()
