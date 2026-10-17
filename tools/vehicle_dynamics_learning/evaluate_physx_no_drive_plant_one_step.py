#!/usr/bin/env python3
"""No-fit one-step replay of the source-composed PhysX NoDrive plant.

The replay resets the *observed* rigid-body and wheel-speed state at each
valid 1 ms sample, then predicts exactly one physics tick. Current wheel
torques and steer angles are treated as the measured controller output. The
open-plane is represented by its known static y=0 plane. Current WheelCollider
GetWorldPose positions initialize the hidden jounce history through the
source `computeWheelLocalPoses` equation. No future state, WheelHit slip/load,
or fitted parameter is used as a predictor input; the next capture row is
scoring truth only.

This is a teacher-forced one-step physics diagnostic, not a recursive rollout
or a production odometry implementation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools.vehicle_dynamics_learning.physx_no_drive_plant import (
    PhysXNoDriveState,
    PhysXNoDriveWheel,
    step_physx_no_drive_static_planes,
)
from tools.vehicle_dynamics_learning.physx_vehicle_source import (
    physx_jounce_from_wheel_pose_actor_body,
    unity_wheel_collider_offsets_com_body,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    RIGIDBODY_CENTER_OF_MASS_BODY_M,
    WHEEL_COLLIDER_CENTER_Y_M,
    WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
    WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW,
    WHEEL_RADIUS_M,
    WHEEL_SUSPENSION_DISTANCE_M,
    WHEEL_SUSPENSION_TARGET_POSITION,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTURE = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
WHEELS = ("fl", "fr", "rl", "rr")
METADATA_WHEELS = ("front_left", "front_right", "rear_left", "rear_right")
FIXED_DT_S = 0.001
ROAD_PLANE_WORLD_XYZD = (0.0, 1.0, 0.0, 0.0)
# The existing r12g open-plane capture repeatedly shows Unity's built-in reset
# target at (0, 0.01, 0) with identity orientation and zero body velocity.
OPEN_PLANE_RESET_POSITION_WORLD_M = np.asarray((0.0, 0.01, 0.0))


def _metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, separator, value = line[2:].strip().partition("=")
            if separator:
                result[key] = value
    return result


def _vector(metadata: dict[str, str], key: str, size: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in metadata[key].split(","))
    if len(values) != size or not all(math.isfinite(value) for value in values):
        raise ValueError(f"invalid {key} metadata")
    return values


def _rotate_xyzw(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = rotation[:3]
    twice_cross = np.asarray(2.0 * np.cross(xyz, vector), dtype=np.float32)
    return np.asarray(vector + rotation[3] * twice_cross
                      + np.cross(xyz, twice_cross), dtype=np.float32)


def _pose_yaw(rotation_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = rotation_xyzw
    return math.atan2(2.0 * (w * y + x * z),
                      1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _columns() -> list[str]:
    columns = [
        "sim_time_s", "pos_world_x_m", "pos_world_y_m", "pos_world_z_m",
        "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
        "vel_body_x_mps", "vel_body_y_mps", "vel_body_z_mps",
        "omega_body_x_rps", "omega_body_y_rps", "omega_body_z_rps",
        "throttle_command", "steering_command",
    ]
    for wheel in WHEELS:
        columns.extend((
            f"{wheel}_rpm", f"{wheel}_steer_deg",
            f"{wheel}_motor_torque_nm", f"{wheel}_brake_torque_nm",
            f"{wheel}_grounded", f"{wheel}_forward_slip",
            f"{wheel}_sideways_slip", f"{wheel}_contact_force_n",
            f"{wheel}_wheel_pose_body_x", f"{wheel}_wheel_pose_body_y",
            f"{wheel}_wheel_pose_body_z",
            f"{wheel}_wheel_pose_qx", f"{wheel}_wheel_pose_qy",
            f"{wheel}_wheel_pose_qz", f"{wheel}_wheel_pose_qw",
        ))
    return columns


def _wheel_configuration(
    metadata: dict[str, str], index: int, steer_angle_rad: float,
    drive_torque_nm: float, brake_torque_nm: float,
) -> PhysXNoDriveWheel:
    source_name = METADATA_WHEELS[index]
    center_offset, application_offset = unity_wheel_collider_offsets_com_body(
        WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M[index],
        WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW[index],
        (0.0, WHEEL_COLLIDER_CENTER_Y_M, 0.0),
        RIGIDBODY_CENTER_OF_MASS_BODY_M,
        float(metadata[f"{source_name}_target_position"]),
        float(metadata[f"{source_name}_suspension_distance_m"]),
        float(metadata[f"{source_name}_radius_m"]),
        float(metadata[f"{source_name}_force_app_point_distance_m"]),
    )
    radius = float(metadata[f"{source_name}_radius_m"])
    wheel_mass = float(metadata[f"{source_name}_wheel_mass_kg"])
    return PhysXNoDriveWheel(
        wheel_center_offset_com_body_m=center_offset,
        suspension_direction_body=(0.0, -1.0, 0.0),
        force_application_offset_com_body_m=application_offset,
        wheel_radius_m=radius,
        max_compression_m=(1.0 - float(metadata[
            f"{source_name}_target_position"]))
            * float(metadata[f"{source_name}_suspension_distance_m"]),
        max_droop_m=float(metadata[f"{source_name}_target_position"])
        * float(metadata[f"{source_name}_suspension_distance_m"]),
        wheel_mass_kg=wheel_mass,
        sprung_mass_kg=float(metadata[f"{source_name}_sprung_mass_kg"]),
        spring_rate_npm=float(metadata[f"{source_name}_spring_n_per_m"]),
        damper_rate_nspm=float(metadata[f"{source_name}_damper_ns_per_m"]),
        wheel_inertia_kgm2=0.5 * wheel_mass * radius**2,
        wheel_damping_rate_nms=float(metadata[
            f"{source_name}_wheel_damping_rate"]),
        forward_curve=_vector(metadata, f"{source_name}_forward_curve", 5),
        sideways_curve=_vector(metadata, f"{source_name}_sideways_curve", 5),
        steer_angle_rad=steer_angle_rad,
        drive_torque_nm=drive_torque_nm,
        brake_torque_nm=brake_torque_nm,
        cached_hit_plane_world_xyzd=ROAD_PLANE_WORLD_XYZD,
    )


def _state_and_wheels(
    row: pd.Series, metadata: dict[str, str]
) -> tuple[PhysXNoDriveState,
           tuple[PhysXNoDriveWheel, PhysXNoDriveWheel,
                 PhysXNoDriveWheel, PhysXNoDriveWheel]]:
    rotation = np.asarray((row.rot_world_x, row.rot_world_y,
                           row.rot_world_z, row.rot_world_w), dtype=np.float32)
    rotation_norm = np.float32(np.sqrt(np.dot(rotation, rotation)))
    rotation = np.asarray(rotation / rotation_norm, dtype=np.float32)
    linear_body = np.asarray((row.vel_body_x_mps, row.vel_body_y_mps,
                              row.vel_body_z_mps), dtype=np.float32)
    angular_body = np.asarray((row.omega_body_x_rps, row.omega_body_y_rps,
                               row.omega_body_z_rps), dtype=np.float32)
    linear_world = _rotate_xyzw(rotation, linear_body)
    angular_world = _rotate_xyzw(rotation, angular_body)
    wheel_omegas: list[float] = []
    jounces: list[float | None] = []
    wheel_models: list[PhysXNoDriveWheel] = []

    for index, wheel in enumerate(WHEELS):
        source_name = METADATA_WHEELS[index]
        radius = float(metadata[f"{source_name}_radius_m"])
        wheel_mass = float(metadata[f"{source_name}_wheel_mass_kg"])
        center_offset, _ = unity_wheel_collider_offsets_com_body(
            WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M[index],
            WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW[index],
            (0.0, WHEEL_COLLIDER_CENTER_Y_M, 0.0),
            RIGIDBODY_CENTER_OF_MASS_BODY_M,
            float(metadata[f"{source_name}_target_position"]),
            float(metadata[f"{source_name}_suspension_distance_m"]),
            radius,
            float(metadata[f"{source_name}_force_app_point_distance_m"]),
        )
        wheel_pose = (float(getattr(row, f"{wheel}_wheel_pose_body_x")),
                      float(getattr(row, f"{wheel}_wheel_pose_body_y")),
                      float(getattr(row, f"{wheel}_wheel_pose_body_z")))
        if float(row.sim_time_s) == 0.0:
            # PxVehicleWheels4DynData initializes mJounces to PX_MAX_F32.
            # The first update maps that sentinel to zero jounce speed; the
            # pre-physics rendered wheel pose is not a valid jounce history.
            jounces.append(None)
        else:
            jounces.append(physx_jounce_from_wheel_pose_actor_body(
                wheel_pose, RIGIDBODY_CENTER_OF_MASS_BODY_M, center_offset,
                (0.0, -1.0, 0.0)))
        wheel_omegas.append(float(getattr(row, f"{wheel}_rpm"))
                            * (2.0 * math.pi / 60.0))
        wheel_models.append(_wheel_configuration(
            metadata, index,
            math.radians(float(getattr(row, f"{wheel}_steer_deg"))),
            float(getattr(row, f"{wheel}_motor_torque_nm")),
            float(getattr(row, f"{wheel}_brake_torque_nm"))))

    state = PhysXNoDriveState(
        actor_position_world_m=(float(row.pos_world_x_m),
                                float(row.pos_world_y_m),
                                float(row.pos_world_z_m)),
        actor_rotation_world_xyzw=tuple(float(value) for value in rotation),
        linear_velocity_world_mps=tuple(float(value) for value in linear_world),
        angular_velocity_world_radps=tuple(float(value) for value in angular_world),
        wheel_omega_radps=tuple(wheel_omegas),
        previous_jounce_m=tuple(jounces),
    )
    return state, tuple(wheel_models)  # type: ignore[return-value]


def _statistics(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "rmse": float(np.sqrt(np.mean(array * array))),
        "p95_abs": float(np.quantile(np.abs(array), 0.95)),
        "max_abs": float(np.max(np.abs(array))),
        "mean": float(np.mean(array)),
    }


def _is_open_plane_reset_transition(current: pd.Series,
                                    following: pd.Series) -> bool:
    """Identify the observed explicit teleport-to-spawn transition only."""
    target_position = np.asarray((following.pos_world_x_m,
                                  following.pos_world_y_m,
                                  following.pos_world_z_m))
    target_rotation = np.asarray((following.rot_world_x,
                                  following.rot_world_y,
                                  following.rot_world_z,
                                  following.rot_world_w))
    target_velocity = np.asarray((following.vel_body_x_mps,
                                  following.vel_body_y_mps,
                                  following.vel_body_z_mps))
    target_is_reset_state = (
        np.linalg.norm(target_position - OPEN_PLANE_RESET_POSITION_WORLD_M)
        <= 1.0e-5
        and np.linalg.norm(target_rotation - np.asarray((0.0, 0.0, 0.0, 1.0))
                            ) <= 1.0e-5
        and np.linalg.norm(target_velocity) <= 1.0e-6
    )
    if not target_is_reset_state:
        return False

    current_position = np.asarray((current.pos_world_x_m,
                                   current.pos_world_y_m,
                                   current.pos_world_z_m))
    current_rotation = np.asarray((current.rot_world_x,
                                   current.rot_world_y,
                                   current.rot_world_z,
                                   current.rot_world_w))
    current_velocity = np.asarray((current.vel_body_x_mps,
                                   current.vel_body_y_mps,
                                   current.vel_body_z_mps))
    return (
        np.linalg.norm(current_position - OPEN_PLANE_RESET_POSITION_WORLD_M)
        > 1.0e-4
        or np.linalg.norm(current_rotation - np.asarray((0.0, 0.0, 0.0, 1.0)))
        > 1.0e-4
        or np.linalg.norm(current_velocity) > 1.0e-4
    )


def evaluate_capture(path: Path, max_transitions: int | None = None,
                     start_index: int = 0, stride: int = 1
                     ) -> dict[str, Any]:
    metadata = _metadata(path)
    if not math.isclose(float(metadata["fixed_delta_time_s"]), FIXED_DT_S,
                        rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError("capture is not on Unity's configured 1 ms grid")
    frame = pd.read_csv(path, comment="#", usecols=_columns())
    if start_index < 0 or stride < 1:
        raise ValueError("start index must be nonnegative and stride positive")
    frame = frame.iloc[start_index:]
    if len(frame) < 2:
        raise ValueError("capture needs at least two rows")

    mass = float(metadata["mass_kg"])
    inertia_diagonal = _vector(metadata, "inertia_tensor_body", 3)
    inverse_inertia = tuple(1.0 / value for value in inertia_diagonal)
    com_rotation = _vector(metadata, "inertia_tensor_rotation_body", 4)
    fixed_delta = np.diff(frame.sim_time_s.to_numpy(dtype=np.float64))
    metric_names = ("position_m", "linear_velocity_mps",
                    "angular_velocity_radps", "yaw_rad",
                    *(f"wheel_omega_{wheel}_radps" for wheel in WHEELS))
    errors: dict[str, dict[str, list[float]]] = {
        band: {key: [] for key in metric_names}
        for band in ("below_5mps", "at_or_above_5mps")
    }
    worst: dict[str, dict[str, dict[str, Any]]] = {
        band: {} for band in errors
    }
    skipped_dt = 0
    skipped_reset = 0
    reset_rows: list[int] = []
    candidate_indices = list(range(0, len(frame) - 1, stride))
    if max_transitions is not None:
        candidate_indices = candidate_indices[:max_transitions]

    for index in candidate_indices:
        if not math.isclose(float(fixed_delta[index]), FIXED_DT_S,
                            rel_tol=0.0, abs_tol=2.0e-7):
            skipped_dt += 1
            continue
        current = frame.iloc[index]
        following = frame.iloc[index + 1]
        if _is_open_plane_reset_transition(current, following):
            skipped_reset += 1
            reset_rows.append(start_index + index)
            continue
        state, wheel_models = _state_and_wheels(current, metadata)
        result = step_physx_no_drive_static_planes(
            state, wheel_models, FIXED_DT_S, mass, inverse_inertia,
            RIGIDBODY_CENTER_OF_MASS_BODY_M, com_rotation)
        speed = abs(float(current.vel_body_z_mps))
        band = "below_5mps" if speed < 5.0 else "at_or_above_5mps"
        bucket = errors[band]

        sample_context = {
            "capture_row": start_index + index,
            "time_s": float(current.sim_time_s),
            "forward_speed_mps": speed,
            "throttle_command": float(current.throttle_command),
            "steering_command": float(current.steering_command),
        }

        def record_error(name: str, value: float,
                         components: tuple[float, ...] | None = None) -> None:
            bucket[name].append(value)
            if (name not in worst[band]
                    or abs(value) > worst[band][name]["abs_error"]):
                worst[band][name] = {
                    **sample_context,
                    "abs_error": abs(value),
                    "signed_error": value,
                    "components": components,
                }

        actual_position = np.asarray((following.pos_world_x_m,
                                      following.pos_world_y_m,
                                      following.pos_world_z_m), dtype=np.float64)
        predicted_position = np.asarray(
            result.state.actor_position_world_m, dtype=np.float64)
        position_delta = predicted_position - actual_position
        record_error("position_m", float(np.linalg.norm(position_delta)),
                     tuple(float(value) for value in position_delta))

        next_rotation = np.asarray((following.rot_world_x,
                                    following.rot_world_y,
                                    following.rot_world_z,
                                    following.rot_world_w), dtype=np.float32)
        next_rotation /= np.float32(np.sqrt(np.dot(next_rotation,
                                                   next_rotation)))
        actual_linear = _rotate_xyzw(
            next_rotation,
            np.asarray((following.vel_body_x_mps, following.vel_body_y_mps,
                        following.vel_body_z_mps), dtype=np.float32))
        actual_angular = _rotate_xyzw(
            next_rotation,
            np.asarray((following.omega_body_x_rps, following.omega_body_y_rps,
                        following.omega_body_z_rps), dtype=np.float32))
        linear_delta = (np.asarray(result.state.linear_velocity_world_mps)
                        - actual_linear)
        angular_delta = (np.asarray(result.state.angular_velocity_world_radps)
                         - actual_angular)
        record_error("linear_velocity_mps", float(np.linalg.norm(linear_delta)),
                     tuple(float(value) for value in linear_delta))
        record_error("angular_velocity_radps",
                     float(np.linalg.norm(angular_delta)),
                     tuple(float(value) for value in angular_delta))
        yaw_delta = _wrap_angle(
            _pose_yaw(result.state.actor_rotation_world_xyzw)
            - _pose_yaw(tuple(float(value) for value in next_rotation)))
        record_error("yaw_rad", abs(yaw_delta), (yaw_delta,))

        for wheel_index, wheel in enumerate(WHEELS):
            predicted_omega = result.state.wheel_omega_radps[wheel_index]
            actual_omega = (float(getattr(following, f"{wheel}_rpm"))
                            * (2.0 * math.pi / 60.0))
            record_error(f"wheel_omega_{wheel}_radps",
                         predicted_omega - actual_omega)

    summary: dict[str, Any] = {}
    for band, metrics in errors.items():
        summary[band] = {
            key: _statistics(values) if values else {"samples": 0}
            for key, values in metrics.items()
        }
    return {
        "capture": str(path.resolve().relative_to(ROOT)),
        "rows_read": len(frame),
        "start_row": start_index,
        "stride_rows": stride,
        "candidate_transitions": len(candidate_indices),
        "transitions_scored": len(candidate_indices) - skipped_dt,
        "invalid_dt_transitions_skipped": skipped_dt,
        "explicit_reset_transitions_skipped": skipped_reset,
        "explicit_reset_transition_start_rows": reset_rows,
        "reset_signature": (
            "only transitions whose next captured pose is the observed exact "
            "open-plane spawn (0, 0.01, 0), identity rotation, zero body "
            "velocity, while the preceding state differs; no collision or "
            "large-error samples are otherwise excluded"),
        "physics_step_s": FIXED_DT_S,
        "predictor_inputs": [
            "current captured rigid-body pose and velocities",
            "current captured wheel RPM, steer angles, motor/brake torques",
            "current WheelCollider wheel-center pose only to initialize jounce",
            "fixed open-plane y=0 scene plane",
            "serialized Unity/PhysX parameters and source equations",
        ],
        "not_used_as_inputs": [
            "future vehicle state",
            "WheelHit forward/side slip",
            "WheelHit suspension force",
            "fitted or learned coefficients",
        ],
        "validation_scope": (
            "teacher-forced one-step source plant replay; not recursive, "
            "not production odometry, and not evidence of full Unity parity"),
        "bands": summary,
        "worst_samples": worst,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_capture(args.capture, args.max_transitions,
                              args.start_index, args.stride)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
