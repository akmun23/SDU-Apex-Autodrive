"""Source-translated PhysX 4.1 NoDrive vehicle update for static ray-hit planes.

This composes the independently ported vehicle-source operations into one
fixed physics tick. It expects scene-query hit planes and Unity-controller
wheel torques/steer angles as inputs. Dynamic hit actors, sticky-tire and
suspension-limit solver constraints, and ordinary scene contacts are not
simulated here; the final scene step is explicitly the free-body branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tools.vehicle_dynamics_learning.physx_no_drive_source import (
    no_drive_substep_count,
    no_drive_wheel_speed_step,
)
from tools.vehicle_dynamics_learning.physx_rigidbody_source import (
    integrate_free_solver_pose,
    integrate_vehicle_body_substep,
    unconstrained_velocity_step,
)
from tools.vehicle_dynamics_learning.physx_vehicle_source import (
    _rotate_xyzw,
    physx_filtered_tire_load_n,
    physx_tire_contact_frame,
    physx_tire_slips,
    physx_wheel_bottom_velocity_mps,
    suspension_force_magnitude_n,
    suspension_jounce_from_cached_plane,
    unity_tire_force_callback,
)


@dataclass(frozen=True)
class PhysXNoDriveState:
    actor_position_world_m: tuple[float, float, float]
    actor_rotation_world_xyzw: tuple[float, float, float, float]
    linear_velocity_world_mps: tuple[float, float, float]
    angular_velocity_world_radps: tuple[float, float, float]
    wheel_omega_radps: tuple[float, float, float, float]
    previous_jounce_m: tuple[float | None, float | None,
                              float | None, float | None]


@dataclass(frozen=True)
class PhysXNoDriveWheel:
    """One WheelCollider's source-configured state and current step inputs."""

    wheel_center_offset_com_body_m: tuple[float, float, float]
    suspension_direction_body: tuple[float, float, float]
    force_application_offset_com_body_m: tuple[float, float, float]
    wheel_radius_m: float
    max_compression_m: float
    max_droop_m: float
    wheel_mass_kg: float
    sprung_mass_kg: float
    spring_rate_npm: float
    damper_rate_nspm: float
    wheel_inertia_kgm2: float
    wheel_damping_rate_nms: float
    forward_curve: tuple[float, float, float, float, float]
    sideways_curve: tuple[float, float, float, float, float]
    steer_angle_rad: float
    drive_torque_nm: float
    brake_torque_nm: float
    cached_hit_plane_world_xyzd: tuple[float, float, float, float] | None
    surface_velocity_world_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    surface_friction_multiplier: float = 1.0


@dataclass(frozen=True)
class PhysXNoDriveStepResult:
    state: PhysXNoDriveState
    substep_count: int
    last_jounce_m: tuple[float | None, float | None,
                         float | None, float | None]
    last_suspension_force_n: tuple[float, float, float, float]
    last_tire_force_world_n: tuple[tuple[float, float, float], ...]
    vehicle_stage_linear_velocity_world_mps: tuple[float, float, float]
    vehicle_stage_angular_velocity_world_radps: tuple[float, float, float]
    last_longitudinal_slip: tuple[float, float, float, float]
    last_lateral_slip: tuple[float, float, float, float]


def _vec3(values: tuple[float, float, float], name: str) -> np.ndarray:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a finite 3-vector")
    return np.asarray(values, dtype=np.float32)


def _quat(values: tuple[float, float, float, float], name: str) -> np.ndarray:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a finite xyzw quaternion")
    result = np.asarray(values, dtype=np.float32)
    norm_sq = np.float32(np.dot(result, result))
    if norm_sq <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return np.asarray(result / np.float32(np.sqrt(norm_sq)), dtype=np.float32)


def step_physx_no_drive_static_planes(
    state: PhysXNoDriveState,
    wheels: tuple[PhysXNoDriveWheel, PhysXNoDriveWheel,
                  PhysXNoDriveWheel, PhysXNoDriveWheel],
    fixed_dt_s: float,
    mass_kg: float,
    inverse_inertia_diagonal_kgm2: tuple[float, float, float],
    center_of_mass_local_position_m: tuple[float, float, float],
    center_of_mass_local_rotation_xyzw: tuple[float, float, float, float],
    gravity_world_mps2: tuple[float, float, float] = (0.0, -9.81, 0.0),
    linear_damping_per_s: float = 0.273,
    angular_damping_per_s: float = 0.1,
) -> PhysXNoDriveStepResult:
    """Advance one Unity 1 ms tick from four already-selected static hit planes.

    The NoDrive substep count is selected once from the initial chassis-forward
    speed, as in PhysX. Input torques and wheel angles are held through that
    vehicle update. The vehicle's temporary COM transform advances inside its
    substeps, but the actual actor pose remains at the tick-start pose until
    the ordinary scene solver's final free-pose integration.
    """
    if len(wheels) != 4:
        raise ValueError("the active F1TENTH model requires four wheels")
    scalars = (fixed_dt_s, mass_kg, linear_damping_per_s,
               angular_damping_per_s)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("step parameters must be finite")
    if fixed_dt_s <= 0.0 or mass_kg <= 0.0:
        raise ValueError("fixed timestep and mass must be positive")

    actor_position = _vec3(state.actor_position_world_m, "actor position")
    actor_rotation = _quat(state.actor_rotation_world_xyzw, "actor rotation")
    linear_velocity = _vec3(state.linear_velocity_world_mps,
                            "linear velocity")
    angular_velocity = _vec3(state.angular_velocity_world_radps,
                             "angular velocity")
    com_local_position = _vec3(center_of_mass_local_position_m,
                               "local COM position")
    com_local_rotation = _quat(center_of_mass_local_rotation_xyzw,
                               "local COM rotation")
    inverse_inertia = _vec3(inverse_inertia_diagonal_kgm2,
                            "inverse inertia diagonal")
    if np.any(inverse_inertia < 0.0):
        raise ValueError("inverse inertia cannot be negative")
    wheel_omegas = np.asarray(state.wheel_omega_radps, dtype=np.float32)
    if (wheel_omegas.shape != (4,) or not np.all(np.isfinite(wheel_omegas))
            or len(state.previous_jounce_m) != 4):
        raise ValueError("wheel states must contain four finite entries")

    actor_rotation_initial = actor_rotation.copy()
    com_position_initial = np.asarray(
        actor_position + _rotate_xyzw(actor_rotation, com_local_position),
        dtype=np.float32)
    # PxVehicleUpdate copies getCMassLocalPose(), then sets its quaternion to
    # identity before forming carChassisTransform: its vehicle substep frame
    # has actor orientation at the COM translation.
    vehicle_com_position = com_position_initial.copy()
    vehicle_chassis_rotation = actor_rotation.copy()
    chassis_forward = _rotate_xyzw(
        vehicle_chassis_rotation,
        np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
    forward_speed = float(np.dot(linear_velocity, chassis_forward))
    substeps = no_drive_substep_count(forward_speed)
    substep_dt = fixed_dt_s / substeps
    previous_jounces = list(state.previous_jounce_m)
    last_jounces: list[float | None] = [None] * 4
    last_suspension_forces = np.zeros(4, dtype=np.float32)
    last_tire_forces = np.zeros((4, 3), dtype=np.float32)
    last_longitudinal_slips = np.zeros(4, dtype=np.float32)
    last_lateral_slips = np.zeros(4, dtype=np.float32)
    intention_to_accelerate = (
        any(abs(wheel.drive_torque_nm) > 0.0 for wheel in wheels)
        and all(abs(wheel.brake_torque_nm) == 0.0 for wheel in wheels))

    for _ in range(substeps):
        chassis_force = np.zeros(3, dtype=np.float32)
        chassis_torque = np.zeros(3, dtype=np.float32)
        for index, wheel in enumerate(wheels):
            plane = wheel.cached_hit_plane_world_xyzd
            if plane is None:
                no_contact_jounce = -wheel.max_droop_m
                last_jounces[index] = no_contact_jounce
                last_suspension_forces[index] = 0.0
                last_tire_forces[index] = 0.0
                wheel_omegas[index] = no_drive_wheel_speed_step(
                    float(wheel_omegas[index]), wheel.drive_torque_nm,
                    wheel.brake_torque_nm, 0.0, wheel.wheel_inertia_kgm2,
                    wheel.wheel_damping_rate_nms, substep_dt)
                previous_jounces[index] = no_contact_jounce
                continue

            intersection = suspension_jounce_from_cached_plane(
                tuple(float(value) for value in vehicle_com_position),
                tuple(float(value) for value in vehicle_chassis_rotation),
                wheel.wheel_center_offset_com_body_m,
                wheel.suspension_direction_body,
                wheel.wheel_radius_m,
                wheel.max_compression_m,
                plane)
            if intersection is None:
                no_contact_jounce = -wheel.max_droop_m
                last_jounces[index] = no_contact_jounce
                last_suspension_forces[index] = 0.0
                last_tire_forces[index] = 0.0
                wheel_omegas[index] = no_drive_wheel_speed_step(
                    float(wheel_omegas[index]), wheel.drive_torque_nm,
                    wheel.brake_torque_nm, 0.0, wheel.wheel_inertia_kgm2,
                    wheel.wheel_damping_rate_nms, substep_dt)
                previous_jounces[index] = no_contact_jounce
                continue

            raw_jounce, wheel_bottom = intersection
            if raw_jounce <= -wheel.max_droop_m:
                no_contact_jounce = -wheel.max_droop_m
                last_jounces[index] = no_contact_jounce
                last_suspension_forces[index] = 0.0
                last_tire_forces[index] = 0.0
                wheel_omegas[index] = no_drive_wheel_speed_step(
                    float(wheel_omegas[index]), wheel.drive_torque_nm,
                    wheel.brake_torque_nm, 0.0, wheel.wheel_inertia_kgm2,
                    wheel.wheel_damping_rate_nms, substep_dt)
                previous_jounces[index] = no_contact_jounce
                continue

            jounce = min(raw_jounce, wheel.max_compression_m)
            if previous_jounces[index] is None:
                jounce_speed = 0.0
            else:
                jounce_speed = (jounce - previous_jounces[index]) / substep_dt
            previous_jounces[index] = jounce
            last_jounces[index] = jounce

            normal = _vec3(plane[:3], "cached hit normal")
            normal_norm_sq = np.float32(np.dot(normal, normal))
            if abs(float(normal_norm_sq) - 1.0) > 1.0e-5:
                raise ValueError("PhysX cached hit normal must be unit length")
            suspension_direction_world = _rotate_xyzw(
                vehicle_chassis_rotation,
                _vec3(wheel.suspension_direction_body,
                      "suspension direction"))
            suspension_force = suspension_force_magnitude_n(
                gravity_world_mps2,
                tuple(float(value) for value in suspension_direction_world),
                tuple(float(value) for value in normal),
                wheel.sprung_mass_kg,
                wheel.spring_rate_npm,
                wheel.damper_rate_nspm,
                jounce,
                jounce_speed)
            last_suspension_forces[index] = suspension_force
            suspension_vector = np.asarray(
                normal * np.float32(suspension_force), dtype=np.float32)
            application_offset = _rotate_xyzw(
                vehicle_chassis_rotation,
                _vec3(wheel.force_application_offset_com_body_m,
                      "force application offset"))
            chassis_force = np.asarray(chassis_force + suspension_vector,
                                       dtype=np.float32)
            chassis_torque = np.asarray(
                chassis_torque + np.cross(application_offset,
                                          suspension_vector),
                dtype=np.float32)

            wheel_velocity = physx_wheel_bottom_velocity_mps(
                tuple(float(value) for value in linear_velocity),
                tuple(float(value) for value in angular_velocity),
                tuple(float(value) for value in vehicle_com_position),
                wheel_bottom,
                wheel.surface_velocity_world_mps)
            longitudinal_axis, lateral_axis = physx_tire_contact_frame(
                tuple(float(value) for value in vehicle_chassis_rotation),
                tuple(float(value) for value in normal),
                wheel.steer_angle_rad)
            longitudinal_speed = float(np.dot(
                np.asarray(wheel_velocity, dtype=np.float32),
                np.asarray(longitudinal_axis, dtype=np.float32)))
            lateral_speed = float(np.dot(
                np.asarray(wheel_velocity, dtype=np.float32),
                np.asarray(lateral_axis, dtype=np.float32)))
            torque_active = (
                wheel.brake_torque_nm != 0.0
                or (intention_to_accelerate
                    and wheel.drive_torque_nm != 0.0))
            long_slip, side_slip = physx_tire_slips(
                longitudinal_speed, lateral_speed,
                float(wheel_omegas[index]), wheel.wheel_radius_m,
                torque_active)
            last_longitudinal_slips[index] = np.float32(long_slip)
            last_lateral_slips[index] = np.float32(side_slip)
            tire_load = physx_filtered_tire_load_n(
                suspension_force, wheel.sprung_mass_kg,
                wheel.wheel_mass_kg,
                gravity_mps2=float(np.linalg.norm(
                    np.asarray(gravity_world_mps2, dtype=np.float32))))
            tire_force = np.zeros(3, dtype=np.float32)
            tire_torque = 0.0
            if tire_load * wheel.surface_friction_multiplier > 0.0:
                tire_torque, force_long, force_lateral, _ = (
                    unity_tire_force_callback(
                        long_slip, side_slip, tire_load,
                        wheel.wheel_radius_m,
                        wheel.forward_curve, wheel.sideways_curve,
                        wheel.surface_friction_multiplier))
                tire_force = np.asarray(
                    np.asarray(longitudinal_axis, dtype=np.float32)
                    * np.float32(force_long)
                    + np.asarray(lateral_axis, dtype=np.float32)
                    * np.float32(force_lateral), dtype=np.float32)
                chassis_force = np.asarray(chassis_force + tire_force,
                                           dtype=np.float32)
                chassis_torque = np.asarray(
                    chassis_torque + np.cross(application_offset, tire_force),
                    dtype=np.float32)
            last_tire_forces[index] = tire_force
            wheel_omegas[index] = no_drive_wheel_speed_step(
                float(wheel_omegas[index]), wheel.drive_torque_nm,
                wheel.brake_torque_nm, tire_torque,
                wheel.wheel_inertia_kgm2,
                wheel.wheel_damping_rate_nms, substep_dt)

        vehicle_com_position_tuple, vehicle_rotation_tuple, linear_tuple, angular_tuple = (
            integrate_vehicle_body_substep(
                tuple(float(value) for value in vehicle_com_position),
                tuple(float(value) for value in vehicle_chassis_rotation),
                tuple(float(value) for value in linear_velocity),
                tuple(float(value) for value in angular_velocity),
                tuple(float(value) for value in chassis_force),
                tuple(float(value) for value in chassis_torque),
                1.0 / mass_kg,
                tuple(float(value) for value in inverse_inertia),
                substep_dt))
        vehicle_com_position = np.asarray(vehicle_com_position_tuple,
                                         dtype=np.float32)
        vehicle_chassis_rotation = np.asarray(vehicle_rotation_tuple,
                                              dtype=np.float32)
        linear_velocity = np.asarray(linear_tuple, dtype=np.float32)
        angular_velocity = np.asarray(angular_tuple, dtype=np.float32)

    # PxVehicleUpdates has now written wheel/body velocities, but not the
    # actor pose. The ordinary PhysX body step starts from the tick-start actor
    # pose and advances the COM/mass-frame pose after gravity and drag.
    vehicle_stage_linear_velocity = tuple(float(value) for value in linear_velocity)
    vehicle_stage_angular_velocity = tuple(float(value) for value in angular_velocity)
    linear_tuple, angular_tuple = unconstrained_velocity_step(
        tuple(float(value) for value in linear_velocity),
        tuple(float(value) for value in angular_velocity),
        gravity_world_mps2, fixed_dt_s,
        linear_damping_per_s, angular_damping_per_s)
    mass_rotation_initial = np.asarray(
        _quat_multiply(actor_rotation_initial, com_local_rotation),
        dtype=np.float32)
    mass_com_position_tuple, mass_rotation_final_tuple = integrate_free_solver_pose(
        tuple(float(value) for value in com_position_initial),
        tuple(float(value) for value in mass_rotation_initial),
        linear_tuple, angular_tuple, fixed_dt_s)
    actor_rotation_final = _quat_multiply(
        np.asarray(mass_rotation_final_tuple, dtype=np.float32),
        _quat_conjugate(com_local_rotation))
    actor_position_final = np.asarray(
        np.asarray(mass_com_position_tuple, dtype=np.float32)
        - _rotate_xyzw(actor_rotation_final, com_local_position),
        dtype=np.float32)

    result_state = PhysXNoDriveState(
        tuple(float(value) for value in actor_position_final),
        tuple(float(value) for value in actor_rotation_final),
        tuple(float(value) for value in linear_tuple),
        tuple(float(value) for value in angular_tuple),
        tuple(float(value) for value in wheel_omegas),
        tuple(last_jounces))
    return PhysXNoDriveStepResult(
        result_state,
        substeps,
        tuple(last_jounces),
        tuple(float(value) for value in last_suspension_forces),
        tuple(tuple(float(value) for value in force)
              for force in last_tire_forces),
        vehicle_stage_linear_velocity,
        vehicle_stage_angular_velocity,
        tuple(float(value) for value in last_longitudinal_slips),
        tuple(float(value) for value in last_lateral_slips))


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), dtype=np.float32)


def _quat_conjugate(rotation: np.ndarray) -> np.ndarray:
    return np.asarray((-rotation[0], -rotation[1], -rotation[2], rotation[3]),
                      dtype=np.float32)
