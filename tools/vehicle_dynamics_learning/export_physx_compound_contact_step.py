#!/usr/bin/env python3
"""Export one truth-initialized WheelCollider step for the native contact replay.

The resulting input contains only tick-start body/wheel state, current commands,
and the source-model vehicle-stage velocity. The native PhysX scene then handles
gravity, damping, compound-shape contacts, and the final 1 ms body step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_physx_no_drive_plant_one_step as replay
from tools.vehicle_dynamics_learning.physx_no_drive_plant import (
    step_physx_no_drive_static_planes,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_mechanics_dynamic_validation_20261013_r12g/"
    "unity_source_capture/wheel_dynamics_unity_mechanics_dynamic_validation_r12g.csv")
DEFAULT_SHAPES = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "vehicle_compound_shapes_clearance_r07_20261016.json")


def export_step(capture: Path, row_index: int, shape_audit: Path,
                frame: pd.DataFrame | None = None,
                metadata: dict[str, str] | None = None) -> dict:
    if metadata is None:
        metadata = replay._metadata(capture)
    if frame is None:
        frame = pd.read_csv(capture, comment="#", usecols=replay._columns())
    if row_index < 0 or row_index + 1 >= len(frame):
        raise ValueError(f"capture row {row_index} needs a following row")
    frame = frame.iloc[row_index:row_index + 2]
    current, following = frame.iloc[0], frame.iloc[1]
    state, wheels = replay._state_and_wheels(current, metadata)
    mass = float(metadata["mass_kg"])
    inertia = replay._vector(metadata, "inertia_tensor_body", 3)
    result = step_physx_no_drive_static_planes(
        state, wheels, replay.FIXED_DT_S, mass,
        tuple(1.0 / value for value in inertia),
        replay.RIGIDBODY_CENTER_OF_MASS_BODY_M,
        replay._vector(metadata, "inertia_tensor_rotation_body", 4))
    wheel_poses = []
    wheel_inputs = []
    for index, wheel in enumerate(replay.WHEELS):
        wheel_config = replay._wheel_configuration(
            metadata, index,
            float(np.deg2rad(current[f"{wheel}_steer_deg"])),
            float(current[f"{wheel}_motor_torque_nm"]),
            float(current[f"{wheel}_brake_torque_nm"]),
        )
        wheel_poses.append({
            "position_body_m": [
                float(getattr(current, f"{wheel}_wheel_pose_body_{axis}"))
                for axis in "xyz"],
            "rotation_body_xyzw": [
                float(getattr(current, f"{wheel}_wheel_pose_q{axis}"))
                for axis in "xyzw"],
        })
        wheel_inputs.append({
            "wheel_center_offset_com_body_m":
                list(wheel_config.wheel_center_offset_com_body_m),
            "force_application_offset_com_body_m":
                list(wheel_config.force_application_offset_com_body_m),
            "suspension_direction_body":
                list(wheel_config.suspension_direction_body),
            "radius_m": wheel_config.wheel_radius_m,
            "width_m": 0.1,
            "mass_kg": wheel_config.wheel_mass_kg,
            "moment_of_inertia_kgm2": wheel_config.wheel_inertia_kgm2,
            "wheel_damping_rate_nms": wheel_config.wheel_damping_rate_nms,
            "max_compression_m": wheel_config.max_compression_m,
            "max_droop_m": wheel_config.max_droop_m,
            "spring_rate_npm": wheel_config.spring_rate_npm,
            "damper_rate_nspm": wheel_config.damper_rate_nspm,
            "sprung_mass_kg": wheel_config.sprung_mass_kg,
            "steer_angle_rad": wheel_config.steer_angle_rad,
            "drive_torque_nm": wheel_config.drive_torque_nm,
            "brake_torque_nm": wheel_config.brake_torque_nm,
            "wheel_omega_radps": float(current[f"{wheel}_rpm"])
                * (2.0 * np.pi / 60.0),
            "forward_curve": list(wheel_config.forward_curve),
            "sideways_curve": list(wheel_config.sideways_curve),
        })
    truth = {
        "position_world_m": [float(current[f"pos_world_{axis}_m"])
                             for axis in "xyz"],
        "rotation_world_xyzw": [float(current[f"rot_world_{axis}"])
                                for axis in "xyzw"],
        "linear_velocity_world_mps": list(state.linear_velocity_world_mps),
        "angular_velocity_world_radps": list(state.angular_velocity_world_radps),
    }
    next_rotation = [float(following[f"rot_world_{axis}"])
                     for axis in "xyzw"]
    expected_linear_world = replay._rotate_xyzw(
        np.asarray(next_rotation, dtype=np.float32),
        np.asarray([
            float(following[f"vel_body_{axis}_mps"]) for axis in "xyz"],
            dtype=np.float32))
    expected_angular_world = replay._rotate_xyzw(
        np.asarray(next_rotation, dtype=np.float32),
        np.asarray([
            float(following[f"omega_body_{axis}_rps"]) for axis in "xyz"],
            dtype=np.float32))
    expected = {
        "position_world_m": [float(following[f"pos_world_{axis}_m"])
                             for axis in "xyz"],
        "rotation_world_xyzw": next_rotation,
        "linear_velocity_world_mps": [float(value)
                                       for value in expected_linear_world],
        "angular_velocity_world_radps": [float(value)
                                          for value in expected_angular_world],
        "linear_velocity_body_mps": [float(following[f"vel_body_{axis}_mps"])
                                     for axis in "xyz"],
        "angular_velocity_body_radps": [float(following[f"omega_body_{axis}_rps"])
                                        for axis in "xyz"],
    }
    return {
        "capture": str(capture.resolve().relative_to(ROOT)),
        "capture_row": row_index,
        "time_s": float(current.sim_time_s),
        "fixed_dt_s": replay.FIXED_DT_S,
        "body": {
            **truth,
            "vehicle_stage_linear_velocity_world_mps":
                list(result.vehicle_stage_linear_velocity_world_mps),
            "vehicle_stage_angular_velocity_world_radps":
                list(result.vehicle_stage_angular_velocity_world_radps),
            "mass_kg": mass,
            "inertia_diagonal_kgm2": list(inertia),
            "inertia_rotation_xyzw": replay._vector(
                metadata, "inertia_tensor_rotation_body", 4),
            "center_of_mass_body_m": list(replay.RIGIDBODY_CENTER_OF_MASS_BODY_M),
        },
        "wheel_shapes": wheel_poses,
        "wheel_inputs": wheel_inputs,
        "expected_next": expected,
        "source_only_next": {
            "position_world_m": list(result.state.actor_position_world_m),
            "rotation_world_xyzw": list(result.state.actor_rotation_world_xyzw),
            "linear_velocity_world_mps": list(result.state.linear_velocity_world_mps),
            "angular_velocity_world_radps": list(result.state.angular_velocity_world_radps),
        },
        "plant_diagnostics": {
            "vehicle_substeps": result.substep_count,
            "suspension_force_n": list(result.last_suspension_force_n),
            "tire_force_world_n": [list(force) for force in result.last_tire_force_world_n],
            "model_slip_used": {
                wheel: {
                    "longitudinal": result.last_longitudinal_slip[index],
                    "lateral": result.last_lateral_slip[index],
                }
                for index, wheel in enumerate(replay.WHEELS)
            },
            "captured_wheel_hit_slip_from_latest_completed_vehicle_update": {
                wheel: {
                    "longitudinal": float(current[f"{wheel}_forward_slip"]),
                    "lateral": float(current[f"{wheel}_sideways_slip"]),
                }
                for wheel in replay.WHEELS
            },
        },
        "shape_audit": str(shape_audit.resolve().relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--count", type=int, default=1,
                        help="contiguous 1 ms samples for warm-contact replay")
    parser.add_argument("--shape-audit", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    metadata = replay._metadata(args.capture)
    frame = pd.read_csv(args.capture, comment="#", usecols=replay._columns())
    if args.row < 0 or args.row + args.count >= len(frame):
        parser.error("requested row window must include a following truth row")
    steps = [export_step(args.capture, row, args.shape_audit, frame, metadata)
             for row in range(args.row, args.row + args.count)]
    result = steps[0] if args.count == 1 else {
        "capture": str(args.capture.resolve().relative_to(ROOT)),
        "start_row": args.row,
        "count": args.count,
        "steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "start_row": args.row,
                      "count": args.count,
                      "first_time_s": steps[0]["time_s"],
                      "last_time_s": steps[-1]["time_s"]}, indent=2))


if __name__ == "__main__":
    main()
