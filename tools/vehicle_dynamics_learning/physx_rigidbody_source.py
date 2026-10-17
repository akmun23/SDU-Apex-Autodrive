"""Float32 port of PhysX 4.1's unconstrained rigid-body velocity stage.

This is the velocity update before constraint/contact impulses. It does not
integrate pose, solve contacts, or model the WheelCollider raycast/tire solve.
"""

from __future__ import annotations

import math

import numpy as np


# Sc::BodyCore initializes dynamic actors with this squared linear limit.
PHYSX_DYNAMIC_MAX_LINEAR_VELOCITY_SQ = 1.0e32
# The Unity project overrides PhysX's default angular limit in DynamicsManager.
UNITY_MAX_ANGULAR_SPEED_RADPS = 7.0
PHYSX_SOLVER_QUATERNION_RATE_LIMIT_RADPS = 1.0e7


def _vec3(values: tuple[float, float, float], name: str) -> np.ndarray:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a finite 3-vector")
    return np.asarray(values, dtype=np.float32)


def _magnitude_squared(vector: np.ndarray) -> np.float32:
    # Match PxVec3's float32 multiply/add sequence rather than accumulating in
    # Python's float64 or delegating to a potentially widened norm operation.
    xx = np.float32(vector[0] * vector[0])
    yy = np.float32(vector[1] * vector[1])
    zz = np.float32(vector[2] * vector[2])
    return np.float32(np.float32(xx + yy) + zz)


def _clamp_magnitude(vector: np.ndarray, max_magnitude_sq: np.float32
                     ) -> np.ndarray:
    magnitude_sq = _magnitude_squared(vector)
    if magnitude_sq > max_magnitude_sq:
        ratio = np.float32(max_magnitude_sq / magnitude_sq)
        scale = np.float32(np.sqrt(ratio))
        return np.asarray(vector * scale, dtype=np.float32)
    return vector


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), dtype=np.float32)


def integrate_vehicle_body_substep(
    position_world_m: tuple[float, float, float],
    rotation_world_xyzw: tuple[float, float, float, float],
    linear_velocity_world_mps: tuple[float, float, float],
    angular_velocity_world_radps: tuple[float, float, float],
    force_world_n: tuple[float, float, float],
    torque_world_nm: tuple[float, float, float],
    inverse_mass_per_kg: float,
    inverse_inertia_diagonal_kgm2: tuple[float, float, float],
    dt_s: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float],
           tuple[float, float, float], tuple[float, float, float]]:
    """Port PhysX Vehicles' `integrateBody` one internal substep.

    Forces and torques are world-frame sums from the four suspension/tire
    contacts. The inverse inertia is diagonal in the body's mass frame.
    PhysX applies semi-implicit linear integration and a first-order normalized
    quaternion update. This is the vehicle SDK substep, before Unity's
    subsequent scene contact solve.
    """
    position = _vec3(position_world_m, "position")
    rotation = np.asarray(rotation_world_xyzw, dtype=np.float32)
    if (len(rotation_world_xyzw) != 4
            or not all(math.isfinite(value) for value in rotation_world_xyzw)):
        raise ValueError("rotation must be a finite xyzw quaternion")
    velocity = _vec3(linear_velocity_world_mps, "linear velocity")
    angular_velocity = _vec3(angular_velocity_world_radps,
                             "angular velocity")
    force = _vec3(force_world_n, "force")
    torque = _vec3(torque_world_nm, "torque")
    inverse_inertia = _vec3(inverse_inertia_diagonal_kgm2,
                            "inverse inertia")
    scalar_values = (inverse_mass_per_kg, dt_s)
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("inverse mass and timestep must be finite")
    if (inverse_mass_per_kg <= 0.0 or dt_s <= 0.0
            or np.any(inverse_inertia < 0.0)):
        raise ValueError("inverse mass, inertia and timestep are invalid")
    q_norm_sq = np.float32(np.sum(rotation * rotation, dtype=np.float32))
    if q_norm_sq <= 0.0:
        raise ValueError("rotation quaternion must be nonzero")
    rotation = np.asarray(rotation / np.float32(np.sqrt(q_norm_sq)),
                          dtype=np.float32)
    dt = np.float32(dt_s)

    # PxVehicleUpdate.cpp::integrateBody: v += F * (inverseMass * dt).
    inverse_mass_dt = np.float32(np.float32(inverse_mass_per_kg) * dt)
    velocity_delta = np.asarray(force * inverse_mass_dt, dtype=np.float32)
    velocity = np.asarray(velocity + velocity_delta, dtype=np.float32)

    # Cm::transformInertiaTensor(invD, PxMat33(q), worldInvI) = R D R^T.
    x, y, z, w = rotation
    two = np.float32(2.0)
    rotation_matrix = np.asarray((
        (1.0 - two * (y*y + z*z), two * (x*y - z*w),
         two * (x*z + y*w)),
        (two * (x*y + z*w), 1.0 - two * (x*x + z*z),
         two * (y*z - x*w)),
        (two * (x*z - y*w), two * (y*z + x*w),
         1.0 - two * (x*x + y*y)),
    ), dtype=np.float32)
    world_inverse_inertia = np.asarray(
        (rotation_matrix * inverse_inertia[np.newaxis, :])
        @ rotation_matrix.T, dtype=np.float32)
    torque_dt = np.asarray(torque * dt, dtype=np.float32)
    angular_delta = np.asarray(world_inverse_inertia @ torque_dt,
                               dtype=np.float32)
    angular_velocity = np.asarray(angular_velocity + angular_delta,
                                  dtype=np.float32)

    # Position uses the newly updated linear velocity.
    position = np.asarray(position + velocity * dt, dtype=np.float32)

    # PxQuat qdot = (omega_world * q) * (dt/2); q = normalize(q + qdot).
    omega_quaternion = np.asarray((angular_velocity[0], angular_velocity[1],
                                   angular_velocity[2], 0.0),
                                  dtype=np.float32)
    qdot = np.asarray(
        _quat_multiply(omega_quaternion, rotation)
        * np.float32(dt * np.float32(0.5)), dtype=np.float32)
    rotation = np.asarray(rotation + qdot, dtype=np.float32)
    rotation_norm_sq = np.float32(np.sum(rotation * rotation,
                                         dtype=np.float32))
    rotation = np.asarray(rotation / np.float32(np.sqrt(rotation_norm_sq)),
                          dtype=np.float32)

    return (tuple(float(value) for value in position),
            tuple(float(value) for value in rotation),
            tuple(float(value) for value in velocity),
            tuple(float(value) for value in angular_velocity))


def integrate_free_solver_pose(
    position_world_m: tuple[float, float, float],
    rotation_world_xyzw: tuple[float, float, float, float],
    linear_velocity_world_mps: tuple[float, float, float],
    angular_velocity_world_radps: tuple[float, float, float],
    dt_s: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Port ``Dy::integrateCore`` pose integration with no solver impulses.

    This is the scene-solver pose step after unconstrained velocity prep. It
    uses the PhysX closed-form quaternion integrator, unlike the first-order
    normalized quaternion update in PxVehicle's internal substep. It applies
    only to the no-contact/no-lock case; contact solver velocity corrections
    must be supplied by the real solver and are deliberately not fabricated.
    """
    position = _vec3(position_world_m, "position")
    rotation = np.asarray(rotation_world_xyzw, dtype=np.float32)
    if (len(rotation_world_xyzw) != 4
            or not all(math.isfinite(value) for value in rotation_world_xyzw)):
        raise ValueError("rotation must be a finite xyzw quaternion")
    velocity = _vec3(linear_velocity_world_mps, "linear velocity")
    angular_velocity = _vec3(angular_velocity_world_radps,
                             "angular velocity")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("timestep must be positive and finite")
    q_norm_sq = np.float32(np.sum(rotation * rotation, dtype=np.float32))
    if q_norm_sq <= 0.0:
        raise ValueError("rotation quaternion must be nonzero")
    rotation = np.asarray(rotation / np.float32(np.sqrt(q_norm_sq)),
                          dtype=np.float32)
    dt = np.float32(dt_s)
    position = np.asarray(position + velocity * dt, dtype=np.float32)

    angular_norm_sq = _magnitude_squared(angular_velocity)
    if angular_norm_sq != np.float32(0.0):
        angular_speed = np.float32(np.sqrt(angular_norm_sq))
        if angular_speed > PHYSX_SOLVER_QUATERNION_RATE_LIMIT_RADPS:
            angular_velocity = np.asarray(
                angular_velocity * np.float32(
                    PHYSX_SOLVER_QUATERNION_RATE_LIMIT_RADPS / angular_speed),
                dtype=np.float32)
            angular_speed = np.float32(
                PHYSX_SOLVER_QUATERNION_RATE_LIMIT_RADPS)
        half_angle = np.float32(dt * angular_speed * np.float32(0.5))
        sine = np.float32(np.sin(half_angle))
        cosine = np.float32(np.cos(half_angle))
        vector_scale = np.float32(sine / angular_speed)
        q_vector = np.asarray(angular_velocity * vector_scale,
                              dtype=np.float32)
        quaternion_delta = _quat_multiply(
            np.asarray((q_vector[0], q_vector[1], q_vector[2], 0.0),
                       dtype=np.float32), rotation)
        rotation = np.asarray(
            quaternion_delta + rotation * cosine, dtype=np.float32)
        result_norm_sq = np.float32(np.sum(rotation * rotation,
                                           dtype=np.float32))
        rotation = np.asarray(rotation / np.float32(np.sqrt(result_norm_sq)),
                              dtype=np.float32)
    return (tuple(float(value) for value in position),
            tuple(float(value) for value in rotation))


def unconstrained_velocity_step(
    linear_velocity_mps: tuple[float, float, float],
    angular_velocity_radps: tuple[float, float, float],
    gravity_mps2: tuple[float, float, float],
    dt_s: float,
    linear_damping_per_s: float,
    angular_damping_per_s: float,
    disable_gravity: bool = False,
    accel_scale: float = 1.0,
    max_linear_velocity_sq: float = PHYSX_DYNAMIC_MAX_LINEAR_VELOCITY_SQ,
    max_angular_velocity_radps: float = UNITY_MAX_ANGULAR_SPEED_RADPS,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Advance the source `bodyCoreComputeUnconstrainedVelocity` operation.

    The source adds gravity first, applies ``max(1 - damping * dt, 0)`` to
    each velocity vector, then clamps each vector's magnitude. Maximum linear
    speed is represented as a squared threshold exactly as in PhysX.
    """
    linear = _vec3(linear_velocity_mps, "linear velocity")
    angular = _vec3(angular_velocity_radps, "angular velocity")
    gravity = _vec3(gravity_mps2, "gravity")
    scalar_values = (dt_s, linear_damping_per_s, angular_damping_per_s,
                     accel_scale, max_linear_velocity_sq,
                     max_angular_velocity_radps)
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("rigid-body step parameters must be finite")
    if (dt_s <= 0.0 or linear_damping_per_s < 0.0
            or angular_damping_per_s < 0.0 or max_linear_velocity_sq < 0.0
            or max_angular_velocity_radps < 0.0):
        raise ValueError("rigid-body damping, limits and timestep are invalid")

    dt = np.float32(dt_s)
    if not disable_gravity:
        scale = np.float32(accel_scale)
        gravity_delta = np.asarray(
            np.asarray(gravity * dt, dtype=np.float32) * scale,
            dtype=np.float32)
        linear = np.asarray(linear + gravity_delta, dtype=np.float32)

    linear_multiplier = np.float32(max(
        np.float32(1.0) - np.float32(linear_damping_per_s) * dt,
        np.float32(0.0)))
    angular_multiplier = np.float32(max(
        np.float32(1.0) - np.float32(angular_damping_per_s) * dt,
        np.float32(0.0)))
    linear = np.asarray(linear * linear_multiplier, dtype=np.float32)
    angular = np.asarray(angular * angular_multiplier, dtype=np.float32)

    linear_limit_sq = np.float32(max_linear_velocity_sq)
    angular_limit = np.float32(max_angular_velocity_radps)
    angular_limit_sq = np.float32(angular_limit * angular_limit)
    linear = _clamp_magnitude(linear, linear_limit_sq)
    angular = _clamp_magnitude(angular, angular_limit_sq)
    return (tuple(float(value) for value in linear),
            tuple(float(value) for value in angular))
