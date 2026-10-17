"""Source-translated WheelCollider suspension operations from PhysX 4.1."""

from __future__ import annotations

import math

import numpy as np


def _finite_vec3(values: tuple[float, float, float], name: str) -> np.ndarray:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a finite 3-vector")
    return np.asarray(values, dtype=np.float32)


def _rotate_xyzw(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = rotation[:3]
    twice_cross = np.asarray(2.0 * np.cross(xyz, vector), dtype=np.float32)
    return np.asarray(vector + rotation[3] * twice_cross
                      + np.cross(xyz, twice_cross), dtype=np.float32)


def _unit_vec3(values: tuple[float, float, float], name: str) -> np.ndarray:
    vector = _finite_vec3(values, name)
    magnitude_sq = np.float32(np.dot(vector, vector))
    if magnitude_sq <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return np.asarray(vector / np.float32(np.sqrt(magnitude_sq)),
                      dtype=np.float32)


def unity_wheel_collider_offsets_com_body(
    wheel_transform_position_body_m: tuple[float, float, float],
    wheel_transform_rotation_body_xyzw: tuple[float, float, float, float],
    collider_center_local_m: tuple[float, float, float],
    center_of_mass_body_m: tuple[float, float, float],
    target_position: float,
    suspension_distance_m: float,
    wheel_radius_m: float,
    force_app_point_distance_m: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build Unity's COM-relative wheel-center and force-point offsets.

    The WheelCollider transform rotates its local ``center``. Unity's active
    wrapper sets suspension travel to chassis-down, so the target-position
    offset is along body ``-Y`` regardless of wheel camber. The force point is
    the target-shifted center minus the wheel radius, then offset along body
    up by ``forceAppPointDistance``. The player passes this same force-point
    vector to PhysX for tire and suspension application.
    """
    position = _finite_vec3(wheel_transform_position_body_m,
                            "wheel transform position")
    centre = _finite_vec3(collider_center_local_m, "WheelCollider center")
    com = _finite_vec3(center_of_mass_body_m, "Rigidbody COM")
    rotation = _quat_xyzw(wheel_transform_rotation_body_xyzw,
                          "wheel transform rotation")
    values = (target_position, suspension_distance_m, wheel_radius_m,
              force_app_point_distance_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("WheelCollider geometry parameters must be finite")
    if (not 0.0 <= target_position <= 1.0 or suspension_distance_m < 0.0
            or wheel_radius_m <= 0.0):
        raise ValueError("WheelCollider travel, target, or radius is invalid")

    rotated_centre = _rotate_xyzw(rotation, centre)
    wheel_centre = np.asarray(
        position + rotated_centre - com
        + np.asarray((0.0, -(1.0 - target_position) * suspension_distance_m,
                      0.0), dtype=np.float32), dtype=np.float32)
    application = np.asarray(
        wheel_centre + np.asarray((0.0,
                                   -wheel_radius_m
                                   + force_app_point_distance_m,
                                   0.0), dtype=np.float32), dtype=np.float32)
    return (tuple(float(value) for value in wheel_centre),
            tuple(float(value) for value in application))


def _quat_xyzw(values: tuple[float, float, float, float], name: str
               ) -> np.ndarray:
    rotation = np.asarray(values, dtype=np.float32)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be a finite xyzw quaternion")
    norm_sq = np.float32(np.dot(rotation, rotation))
    if norm_sq <= 0.0:
        raise ValueError(f"{name} must be nonzero")
    return np.asarray(rotation / np.float32(np.sqrt(norm_sq)),
                      dtype=np.float32)


def unity_cylinder_collider_plane_clearance_m(
    parent_position_world_m: tuple[float, float, float],
    parent_rotation_world_xyzw: tuple[float, float, float, float],
    child_position_local_m: tuple[float, float, float],
    child_rotation_local_xyzw: tuple[float, float, float, float],
    cylinder_radius_m: float,
    cylinder_half_height_m: float,
    plane_world_xyzd: tuple[float, float, float, float],
) -> float:
    """Signed support clearance for a transformed circular cylinder/plane.

    Unity's F1TENTH prefab uses the built-in Y-axis Cylinder mesh (builtin
    fileID 10206), scaled to a nominal 0.055 m radius and 0.022 m half-height
    under each WheelCollider visual. The actual built-in mesh is polygonal;
    this uses its circular-cylinder support envelope until the exact builtin
    vertex hull is extracted. It computes only nominal geometric separation;
    it does not emulate PhysX contact generation, friction, or solver impulses.
    """
    parent_position = _finite_vec3(parent_position_world_m,
                                   "collider parent position")
    parent_rotation = _quat_xyzw(parent_rotation_world_xyzw,
                                 "collider parent rotation")
    child_position = _finite_vec3(child_position_local_m,
                                  "collider child position")
    child_rotation = _quat_xyzw(child_rotation_local_xyzw,
                                "collider child rotation")
    plane = _finite_vec3(plane_world_xyzd[:3], "plane normal")
    if len(plane_world_xyzd) != 4 or not math.isfinite(plane_world_xyzd[3]):
        raise ValueError("plane must contain a finite normal and offset")
    normal_norm_sq = np.float32(np.dot(plane, plane))
    if abs(float(normal_norm_sq) - 1.0) > 1.0e-5:
        raise ValueError("plane normal must be unit length")
    if (not math.isfinite(cylinder_radius_m)
            or not math.isfinite(cylinder_half_height_m)
            or cylinder_radius_m <= 0.0 or cylinder_half_height_m <= 0.0):
        raise ValueError("cylinder dimensions must be positive and finite")

    child_offset_world = _rotate_xyzw(parent_rotation, child_position)
    centre = np.asarray(parent_position + child_offset_world,
                        dtype=np.float32)
    cylinder_rotation = _quat_xyzw(
        tuple(float(value) for value in _rotate_quaternions_xyzw(
            parent_rotation, child_rotation)), "cylinder world rotation")
    cylinder_axis = _rotate_xyzw(
        cylinder_rotation, np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
    axis_normal_dot = np.float32(np.dot(cylinder_axis, plane))
    radial_support = np.float32(cylinder_radius_m) * np.float32(np.sqrt(
        max(0.0, float(np.float32(1.0 - axis_normal_dot * axis_normal_dot)))))
    axial_support = (np.float32(cylinder_half_height_m)
                     * np.float32(abs(float(axis_normal_dot))))
    signed_centre_distance = np.float32(
        np.dot(centre, plane) + np.float32(plane_world_xyzd[3]))
    return float(np.float32(signed_centre_distance
                            - radial_support - axial_support))


def unity_convex_mesh_plane_clearance_m(
    center_world_m: np.ndarray,
    rotation_world_xyzw: np.ndarray,
    mesh_vertices_local_xyz: np.ndarray,
    mesh_scale_xyz: tuple[float, float, float],
    plane_world_xyzd: tuple[float, float, float, float],
    chunk_size: int = 8192,
) -> np.ndarray:
    """Compute exact source-vertex support clearance for a convex mesh/plane.

    The support of the convex hull in a direction is the support of its source
    vertex set. This computes that geometric quantity, not Unity's cooked
    PhysX shape, contact offsets, friction, or solver impulse. Batching is
    chunked to keep long 1 kHz captures memory-bounded.
    """
    centers = np.asarray(center_world_m, dtype=np.float64)
    rotations = np.asarray(rotation_world_xyzw, dtype=np.float64)
    vertices = np.asarray(mesh_vertices_local_xyz, dtype=np.float64)
    scale = np.asarray(mesh_scale_xyz, dtype=np.float64)
    plane = np.asarray(plane_world_xyzd, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
        raise ValueError("collider centers must have shape (N, 3)")
    if rotations.shape != (len(centers), 4):
        raise ValueError("collider rotations must have shape (N, 4)")
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("mesh vertices must have shape (M, 3)")
    if scale.shape != (3,) or np.any(scale <= 0.0):
        raise ValueError("mesh scale must be a positive 3-vector")
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError("plane must be a finite xyzd 4-vector")
    if (not np.all(np.isfinite(centers)) or not np.all(np.isfinite(rotations))
            or not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(scale))):
        raise ValueError("mesh and collider geometry must be finite")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    normal = plane[:3]
    normal_norm = np.linalg.norm(normal)
    if abs(float(normal_norm) - 1.0) > 1.0e-5:
        raise ValueError("plane normal must be unit length")
    rotation_norms = np.linalg.norm(rotations, axis=1)
    if np.any(rotation_norms <= 0.0):
        raise ValueError("collider rotations must be nonzero quaternions")
    rotations = rotations / rotation_norms[:, None]

    # Transform the world plane normal into each collider's local frame;
    # project the scaled source vertices there instead of materializing every
    # transformed mesh vertex for every capture row.
    inverse = rotations.copy()
    inverse[:, :3] *= -1.0
    local_normal = np.broadcast_to(normal, (len(centers), 3)).copy()
    twice_cross = 2.0 * np.cross(inverse[:, :3], local_normal)
    local_normal += inverse[:, 3, None] * twice_cross
    local_normal += np.cross(inverse[:, :3], twice_cross)
    scaled_vertices = vertices * scale
    result = np.empty(len(centers), dtype=np.float64)
    for start in range(0, len(centers), chunk_size):
        stop = min(start + chunk_size, len(centers))
        projected = local_normal[start:stop] @ scaled_vertices.T
        support = np.min(projected, axis=1)
        result[start:stop] = (centers[start:stop] @ normal + plane[3]
                              + support)
    return result


def _rotate_quaternions_xyzw(
    parent: np.ndarray, child: np.ndarray
) -> np.ndarray:
    """Compose normalized xyzw rotations as parent * child."""
    px, py, pz, pw = parent
    cx, cy, cz, cw = child
    result = np.asarray((
        pw * cx + px * cw + py * cz - pz * cy,
        pw * cy - px * cz + py * cw + pz * cx,
        pw * cz + px * cy - py * cx + pz * cw,
        pw * cw - px * cx - py * cy - pz * cz,
    ), dtype=np.float32)
    norm_sq = np.float32(np.dot(result, result))
    return np.asarray(result / np.float32(np.sqrt(norm_sq)),
                      dtype=np.float32)


def physx_wheel_pose_position_actor_body(
    center_of_mass_body_m: tuple[float, float, float],
    wheel_centre_offset_com_body_m: tuple[float, float, float],
    suspension_direction_body: tuple[float, float, float],
    jounce_m: float,
) -> tuple[float, float, float]:
    """Port `computeWheelLocalPoses` wheel-center position in actor space."""
    center_of_mass = _finite_vec3(center_of_mass_body_m, "body COM")
    wheel_offset = _finite_vec3(wheel_centre_offset_com_body_m,
                                "wheel centre offset")
    suspension_direction = _finite_vec3(suspension_direction_body,
                                         "suspension direction")
    if not math.isfinite(jounce_m):
        raise ValueError("jounce must be finite")
    direction_norm_sq = np.float32(np.dot(suspension_direction,
                                          suspension_direction))
    if abs(float(direction_norm_sq) - 1.0) > 1.0e-5:
        raise ValueError("PhysX suspension direction must be unit length")
    position = np.asarray(
        center_of_mass + wheel_offset
        - suspension_direction * np.float32(jounce_m), dtype=np.float32)
    return tuple(float(value) for value in position)


def physx_jounce_from_wheel_pose_actor_body(
    wheel_pose_position_actor_body_m: tuple[float, float, float],
    center_of_mass_body_m: tuple[float, float, float],
    wheel_centre_offset_com_body_m: tuple[float, float, float],
    suspension_direction_body: tuple[float, float, float],
) -> float:
    """Recover PhysX jounce from the wheel-center pose and configured offsets.

    This is the inverse of `computeWheelLocalPoses`' position equation. It is
    useful for initializing the native jounce-history state from Unity's
    WheelCollider `GetWorldPose` telemetry without feeding in future truth.
    """
    wheel_pose = _finite_vec3(wheel_pose_position_actor_body_m,
                              "wheel pose position")
    center_of_mass = _finite_vec3(center_of_mass_body_m, "body COM")
    wheel_offset = _finite_vec3(wheel_centre_offset_com_body_m,
                                "wheel centre offset")
    suspension_direction = _finite_vec3(suspension_direction_body,
                                         "suspension direction")
    direction_norm_sq = np.float32(np.dot(suspension_direction,
                                          suspension_direction))
    if abs(float(direction_norm_sq) - 1.0) > 1.0e-5:
        raise ValueError("PhysX suspension direction must be unit length")
    baseline = np.asarray(center_of_mass + wheel_offset, dtype=np.float32)
    displacement = np.asarray(wheel_pose - baseline, dtype=np.float32)
    jounce = np.float32(-np.dot(displacement, suspension_direction))
    return float(jounce)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply normalized Unity/PhysX xyzw quaternions, left * right."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    result = np.asarray((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), dtype=np.float32)
    norm_sq = np.float32(np.dot(result, result))
    if norm_sq <= 0.0:
        raise ValueError("quaternion product must be nonzero")
    return np.asarray(result / np.float32(np.sqrt(norm_sq)), dtype=np.float32)


def physx_wheel_pose_rotation_actor_body(
    steer_angle_rad: float, wheel_rotation_angle_rad: float
) -> tuple[float, float, float, float]:
    """Port WheelCollider::GetLocalPose's wheel orientation.

    The matching Unity player converts WheelCollider steering degrees to
    radians before storing them in PxVehicleNoDrive. GetLocalPose then forms
    ``q_y(steer) * q_x(rotationAngle)`` using half-angle sine/cosine.
    """
    if not all(math.isfinite(value) for value in
               (steer_angle_rad, wheel_rotation_angle_rad)):
        raise ValueError("wheel steering and rotation angles must be finite")
    half_steer = np.float32(np.float32(steer_angle_rad) * np.float32(0.5))
    half_roll = np.float32(
        np.float32(wheel_rotation_angle_rad) * np.float32(0.5))
    steer = np.asarray((0.0, np.sin(half_steer), 0.0,
                        np.cos(half_steer)), dtype=np.float32)
    roll = np.asarray((np.sin(half_roll), 0.0, 0.0,
                       np.cos(half_roll)), dtype=np.float32)
    result = _quat_multiply_xyzw(steer, roll)
    return tuple(float(value) for value in result)


def physx_wheel_rotation_angle_from_pose_actor_body(
    wheel_pose_rotation_actor_body_xyzw: tuple[float, float, float, float],
    steer_angle_rad: float,
) -> tuple[float, float]:
    """Recover the initial PhysX wheel angle and report pose closure error.

    The wheel angle is only needed modulo 2*pi to reproduce the visual/collider
    pose. The second return value is the shortest quaternion angular residual;
    it reveals any orientation component not represented by steering and roll.
    """
    measured = _quat_xyzw(wheel_pose_rotation_actor_body_xyzw,
                          "wheel pose rotation")
    steering_pose = np.asarray(physx_wheel_pose_rotation_actor_body(
        steer_angle_rad, 0.0), dtype=np.float32)
    inverse_steering = steering_pose.copy()
    inverse_steering[:3] *= -1.0
    roll = _quat_multiply_xyzw(inverse_steering, measured)
    angle = float(np.float32(2.0) * np.arctan2(roll[0], roll[3]))
    reconstructed = np.asarray(physx_wheel_pose_rotation_actor_body(
        steer_angle_rad, angle), dtype=np.float32)
    dot = min(1.0, max(-1.0, abs(float(np.dot(measured, reconstructed)))))
    residual = 2.0 * math.acos(dot)
    return angle, residual


def suspension_jounce_from_cached_plane(
    chassis_com_position_world_m: tuple[float, float, float],
    chassis_com_rotation_world_xyzw: tuple[float, float, float, float],
    wheel_centre_offset_com_body_m: tuple[float, float, float],
    suspension_direction_body: tuple[float, float, float],
    wheel_radius_m: float,
    max_compression_m: float,
    hit_plane_world_xyzw: tuple[float, float, float, float],
) -> tuple[float, tuple[float, float, float]] | None:
    """Recompute PhysX's ray/plane jounce from a cached scene-query hit.

    The cached plane comes from the single WheelCollider scene query issued
    for the Unity physics step. PhysX reprojects its ray from the evolving
    temporary chassis pose on each internal vehicle substep; it does not
    issue another scene query there. The hit plane is ``(nx, ny, nz, d)``
    with ``n dot x + d = 0``. The wheel-centre offset is the COM-relative
    value configured in ``PxVehicleWheelsSimData`` (already target-shifted).

    Returns ``None`` when PhysX rejects a ray start below/inside the cached
    hit plane. A parallel ray is also returned as no usable intersection
    rather than reproducing the source's undefined divide-by-zero result.
    Otherwise returns ``(jounce, wheel_bottom_world_position)``.
    """
    position = _finite_vec3(chassis_com_position_world_m,
                            "chassis COM position")
    centre_offset = _finite_vec3(wheel_centre_offset_com_body_m,
                                 "wheel centre offset")
    suspension_direction = _finite_vec3(suspension_direction_body,
                                         "suspension direction")
    plane_normal = _finite_vec3(hit_plane_world_xyzw[:3], "hit plane normal")
    rotation = np.asarray(chassis_com_rotation_world_xyzw, dtype=np.float32)
    if (len(chassis_com_rotation_world_xyzw) != 4
            or not all(math.isfinite(value)
                       for value in chassis_com_rotation_world_xyzw)):
        raise ValueError("chassis rotation must be a finite xyzw quaternion")
    if (len(hit_plane_world_xyzw) != 4
            or not math.isfinite(hit_plane_world_xyzw[3])):
        raise ValueError("hit plane must contain finite (normal, d)")
    if (not math.isfinite(wheel_radius_m) or wheel_radius_m <= 0.0
            or not math.isfinite(max_compression_m)
            or max_compression_m < 0.0):
        raise ValueError("wheel radius/compression are invalid")
    q_norm_sq = np.float32(np.sum(rotation * rotation, dtype=np.float32))
    if q_norm_sq <= 0.0:
        raise ValueError("chassis quaternion must be nonzero")
    rotation = np.asarray(rotation / np.float32(np.sqrt(q_norm_sq)),
                          dtype=np.float32)
    suspension_direction_norm_sq = np.float32(
        np.sum(suspension_direction * suspension_direction,
               dtype=np.float32))
    if abs(float(suspension_direction_norm_sq) - 1.0) > 1.0e-5:
        raise ValueError("PhysX suspension direction must be unit length")

    ray_direction = _rotate_xyzw(rotation, suspension_direction)
    wheel_centre_world = np.asarray(
        position + _rotate_xyzw(rotation, centre_offset), dtype=np.float32)
    start = np.asarray(
        wheel_centre_world
        - ray_direction * np.float32(wheel_radius_m + max_compression_m),
        dtype=np.float32)
    plane_d = np.float32(hit_plane_world_xyzw[3])
    start_signed_distance = np.float32(
        np.float32(np.dot(plane_normal, start)) + plane_d)
    if start_signed_distance < 0.0:
        return None

    normal_dot_direction = np.float32(
        np.dot(plane_normal, ray_direction))
    if normal_dot_direction == 0.0:
        return None
    ray_parameter = np.float32(
        -start_signed_distance / normal_dot_direction)
    rest_length = np.float32(2.0 * wheel_radius_m + max_compression_m)
    jounce = np.float32(rest_length - ray_parameter)
    wheel_bottom = np.asarray(
        start + ray_direction * np.float32(rest_length - jounce),
        dtype=np.float32)
    return (float(jounce), tuple(float(value) for value in wheel_bottom))


def suspension_jounce_speed_mps(jounce_m: float,
                                previous_jounce_m: float | None,
                                substep_dt_s: float) -> float:
    """PhysX's jounce derivative; the uninitialized first sample is zero."""
    values = (jounce_m, substep_dt_s)
    if (not all(math.isfinite(value) for value in values)
            or (previous_jounce_m is not None
                and not math.isfinite(previous_jounce_m))):
        raise ValueError("jounce and timestep must be finite")
    if substep_dt_s <= 0.0:
        raise ValueError("substep timestep must be positive")
    if previous_jounce_m is None:
        return 0.0
    return (jounce_m - previous_jounce_m) / substep_dt_s


def suspension_force_magnitude_n(
    gravity_world_mps2: tuple[float, float, float],
    suspension_direction_world: tuple[float, float, float],
    hit_normal_world: tuple[float, float, float],
    sprung_mass_kg: float,
    spring_rate_npm: float,
    damper_rate_nspm: float,
    jounce_m: float,
    jounce_speed_mps: float,
    disable_force_projection: bool = False,
) -> float:
    """Port PhysX's suspension spring/gravity projection for a static hit.

    ``disable_force_projection=False`` is the PhysX default. The caller must
    supply the current substep jounce and its derivative and the selected
    scene-query normal. Dynamic hit actors have an additional acceleration
    clamp in PhysX and are outside this static-surface helper.
    """
    gravity = _finite_vec3(gravity_world_mps2, "gravity")
    direction = _finite_vec3(suspension_direction_world,
                             "suspension direction")
    normal = _finite_vec3(hit_normal_world, "hit normal")
    values = (sprung_mass_kg, spring_rate_npm, damper_rate_nspm,
              jounce_m, jounce_speed_mps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("suspension parameters must be finite")
    if (sprung_mass_kg < 0.0 or spring_rate_npm < 0.0
            or damper_rate_nspm < 0.0):
        raise ValueError("suspension mass and rates must be nonnegative")

    gravity_suspension = np.float32(np.dot(gravity, direction))
    alpha = np.float32(max(0.0, float(gravity_suspension)))
    if alpha != 0.0:
        tangent_gravity = np.asarray(
            gravity - direction * alpha, dtype=np.float32)
    else:
        tangent_gravity = np.zeros(3, dtype=np.float32)
    spring_force = np.float32(max(
        0.0,
        float(np.float32(sprung_mass_kg) * alpha
              + np.float32(spring_rate_npm) * np.float32(jounce_m))))
    spring_force = np.float32(
        spring_force + np.float32(jounce_speed_mps)
        * np.float32(damper_rate_nspm))
    if disable_force_projection:
        normal_component = np.float32(
            np.dot(normal, tangent_gravity * np.float32(sprung_mass_kg)))
        result = np.float32(spring_force - normal_component)
    else:
        force_vector = np.asarray(
            -direction * spring_force
            - tangent_gravity * np.float32(sprung_mass_kg), dtype=np.float32)
        result = np.float32(np.dot(normal, force_vector))
    return float(result)


def physx_tire_contact_frame(
    chassis_rotation_world_xyzw: tuple[float, float, float, float],
    hit_normal_world: tuple[float, float, float],
    wheel_steer_angle_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Build the source `computeTireDirs` axes in the contact plane."""
    if not math.isfinite(wheel_steer_angle_rad):
        raise ValueError("wheel steering angle must be finite")
    rotation = np.asarray(chassis_rotation_world_xyzw, dtype=np.float32)
    if (len(chassis_rotation_world_xyzw) != 4
            or not all(math.isfinite(value)
                       for value in chassis_rotation_world_xyzw)):
        raise ValueError("chassis rotation must be a finite xyzw quaternion")
    norm_sq = np.float32(np.dot(rotation, rotation))
    if norm_sq <= 0.0:
        raise ValueError("chassis quaternion must be nonzero")
    rotation = np.asarray(rotation / np.float32(np.sqrt(norm_sq)),
                          dtype=np.float32)
    hit_normal = _unit_vec3(hit_normal_world, "hit normal")
    chassis_lateral = _rotate_xyzw(
        rotation, np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
    raw_longitudinal = _unit_vec3(
        tuple(float(value) for value in np.cross(chassis_lateral, hit_normal)),
        "unsteered tire longitudinal axis")
    raw_lateral = _unit_vec3(
        tuple(float(value) for value in np.cross(hit_normal,
                                                  raw_longitudinal)),
        "unsteered tire lateral axis")
    cosine = np.float32(np.cos(np.float32(wheel_steer_angle_rad)))
    sine = np.float32(np.sin(np.float32(wheel_steer_angle_rad)))
    longitudinal = np.asarray(raw_longitudinal * cosine
                              + raw_lateral * sine, dtype=np.float32)
    lateral = np.asarray(raw_lateral * cosine
                         - raw_longitudinal * sine, dtype=np.float32)
    return (tuple(float(value) for value in longitudinal),
            tuple(float(value) for value in lateral))


def physx_wheel_bottom_velocity_mps(
    chassis_linear_velocity_world_mps: tuple[float, float, float],
    chassis_angular_velocity_world_radps: tuple[float, float, float],
    chassis_com_position_world_m: tuple[float, float, float],
    wheel_bottom_position_world_m: tuple[float, float, float],
    hit_actor_velocity_world_mps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float]:
    """Evaluate PhysX's rigid contact-point velocity relative to the hit actor."""
    velocity = _finite_vec3(chassis_linear_velocity_world_mps,
                            "chassis linear velocity")
    angular = _finite_vec3(chassis_angular_velocity_world_radps,
                           "chassis angular velocity")
    com_position = _finite_vec3(chassis_com_position_world_m,
                                "chassis COM position")
    wheel_position = _finite_vec3(wheel_bottom_position_world_m,
                                  "wheel-bottom position")
    surface_velocity = _finite_vec3(hit_actor_velocity_world_mps,
                                    "hit actor velocity")
    lever = np.asarray(wheel_position - com_position, dtype=np.float32)
    point_velocity = np.asarray(
        velocity + np.cross(angular, lever), dtype=np.float32)
    point_velocity = np.asarray(point_velocity - surface_velocity,
                                dtype=np.float32)
    return tuple(float(value) for value in point_velocity)


def physx_tire_slips(
    longitudinal_speed_mps: float,
    lateral_speed_mps: float,
    wheel_omega_radps: float,
    wheel_radius_m: float,
    torque_active: bool,
    minimum_long_slip_denominator_mps: float = 4.0,
) -> tuple[float, float]:
    """PhysX 4.1 NoDrive longitudinal/lateral slip equations."""
    values = (longitudinal_speed_mps, lateral_speed_mps,
              wheel_omega_radps, wheel_radius_m,
              minimum_long_slip_denominator_mps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("tire slip inputs must be finite")
    if wheel_radius_m <= 0.0 or minimum_long_slip_denominator_mps <= 0.0:
        raise ValueError("wheel radius and minimum denominator must be positive")
    long_speed = np.float32(longitudinal_speed_mps)
    lateral_speed = np.float32(lateral_speed_mps)
    wheel_speed = np.float32(np.float32(wheel_omega_radps)
                             * np.float32(wheel_radius_m))
    long_abs = np.float32(abs(float(long_speed)))
    wheel_abs = np.float32(abs(float(wheel_speed)))
    lateral_slip = np.float32(np.arctan(
        lateral_speed / np.float32(long_abs + np.float32(1.0))))
    if long_speed == 0.0 and wheel_speed == 0.0:
        long_slip = np.float32(0.0)
    elif torque_active:
        denominator = np.float32(max(float(long_abs), float(wheel_abs)) + 0.1)
        long_slip = np.float32((wheel_speed - long_speed) / denominator)
    else:
        denominator = np.float32(max(
            minimum_long_slip_denominator_mps,
            float(long_abs), float(wheel_abs)))
        long_slip = np.float32((wheel_speed - long_speed) / denominator)
    return float(long_slip), float(lateral_slip)


def _unity_wheel_friction_curve_base_response(
    slip: float, curve: tuple[float, float, float, float, float]
) -> np.float32:
    """Evaluate the signed native spline before its separate stiffness scale."""
    if len(curve) != 5 or not math.isfinite(slip) or not all(
            math.isfinite(value) for value in curve):
        raise ValueError("slip and five friction-curve values must be finite")
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, _ = (
        (np.float32(value) for value in curve))
    if extremum_slip <= 0.0 or asymptote_slip <= extremum_slip:
        raise ValueError("friction curve slip landmarks are invalid")
    value = np.float32(slip)
    magnitude = np.float32(abs(float(value)))
    if magnitude < extremum_slip:
        t = np.float32(magnitude / extremum_slip)
        response = np.float32(extremum_value * np.float32(
            t + np.float32(t * t) - np.float32(t * t * t)))
    elif magnitude < asymptote_slip:
        t = np.float32((magnitude - extremum_slip)
                       / (asymptote_slip - extremum_slip))
        smooth = np.float32(np.float32(3.0) * t * t
                             - np.float32(2.0) * t * t * t)
        response = np.float32(
            extremum_value + (asymptote_value - extremum_value) * smooth)
    else:
        response = asymptote_value
    sign = np.float32(1.0 if value >= 0.0 else -1.0)
    return np.float32(sign * response)


def unity_wheel_friction_curve_response(
    slip: float, curve: tuple[float, float, float, float, float]
) -> float:
    """Evaluate Unity's signed spline response including its stiffness scale."""
    base = _unity_wheel_friction_curve_base_response(slip, curve)
    return float(np.float32(base * np.float32(curve[4])))


def unity_tire_force_callback(
    longitudinal_slip: float,
    lateral_slip: float,
    filtered_tire_load_n: float,
    wheel_radius_m: float,
    forward_curve: tuple[float, float, float, float, float],
    sideways_curve: tuple[float, float, float, float, float],
    surface_tire_friction: float = 1.0,
) -> tuple[float, float, float, float]:
    """UnityPlayer's WheelCollider callback: wheel torque, Fx, Fy, aligning M."""
    values = (longitudinal_slip, lateral_slip, filtered_tire_load_n,
              wheel_radius_m, surface_tire_friction)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("tire callback inputs must be finite")
    if (filtered_tire_load_n < 0.0 or wheel_radius_m <= 0.0
            or surface_tire_friction < 0.0):
        raise ValueError("tire load, radius, and surface friction are invalid")
    long_response = _unity_wheel_friction_curve_base_response(
        longitudinal_slip, forward_curve)
    side_response = _unity_wheel_friction_curve_base_response(
        lateral_slip, sideways_curve)
    utilization_sq = np.float32(
        (long_response / np.float32(forward_curve[1])) ** 2
        + (side_response / np.float32(sideways_curve[1])) ** 2)
    scale = np.float32(1.0 / np.sqrt(max(1.0, float(utilization_sq))))
    force_long = np.float32(
        np.float32(filtered_tire_load_n) * np.float32(surface_tire_friction)
        * long_response * scale * np.float32(forward_curve[4]))
    force_lateral = np.float32(
        -np.float32(filtered_tire_load_n) * np.float32(surface_tire_friction)
        * side_response * scale * np.float32(sideways_curve[4]))
    wheel_torque = np.float32(-force_long * np.float32(wheel_radius_m))
    return (float(wheel_torque), float(force_long), float(force_lateral), 0.0)


def physx_filtered_tire_load_n(
    suspension_force_n: float,
    sprung_mass_kg: float,
    wheel_mass_kg: float,
    gravity_mps2: float = 9.81,
    in_contact: bool = True,
    filter_points: tuple[float, float, float, float] = (0.0, 0.2308, 3.0, 3.0),
) -> float:
    """Apply PhysX's normalized load filter and wheel-plus-sprung rest load."""
    values = (suspension_force_n, sprung_mass_kg, wheel_mass_kg, gravity_mps2,
              *filter_points)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("tire-load inputs must be finite")
    if (sprung_mass_kg < 0.0 or wheel_mass_kg < 0.0 or gravity_mps2 <= 0.0
            or len(filter_points) != 4):
        raise ValueError("tire-load parameters are invalid")
    if not in_contact:
        return 0.0
    min_load, min_filtered, max_load, max_filtered = (
        np.float32(value) for value in filter_points)
    if max_load <= min_load:
        raise ValueError("tire-load filter interval must be positive")
    rest_load = np.float32(
        np.float32(gravity_mps2)
        * np.float32(np.float32(sprung_mass_kg) + np.float32(wheel_mass_kg)))
    normalized = np.float32(np.float32(suspension_force_n) / rest_load)
    if normalized <= min_load:
        filtered = min_filtered
    elif normalized >= max_load:
        filtered = max_filtered
    else:
        filtered = np.float32(
            min_filtered + (normalized - min_load)
            * (max_filtered - min_filtered) / (max_load - min_load))
    return float(np.float32(filtered * rest_load))
