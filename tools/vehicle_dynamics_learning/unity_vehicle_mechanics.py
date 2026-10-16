"""Source-confirmed kinematic and actuator equations for the AutoDRIVE car.

These equations reproduce public ``VehicleController`` behavior and rigid-body
wheel-point kinematics. They do not emulate Unity's native WheelCollider tire
solver. Keep simulator-only labels out of runtime odometry/controller inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from tools.vehicle_dynamics_learning.unity_vehicle_controller_source import (
    step_autonomous_steering_deg,
)


WHEEL_NAMES = ("front_left", "front_right", "rear_left", "rear_right")
STEERING_LIMIT_RAD = math.radians(30.0)
STEERING_RATE_RADPS = math.radians(183.345993)
MOTOR_TORQUE_NM = 428.0
WHEEL_RADIUS_M = 0.059
WHEELBASE_ACKERMANN_M = 0.324
TRACK_M = 0.236
RIGIDBODY_MASS_KG = 3.47
RIGIDBODY_DRAG_PER_S = 0.273
RIGIDBODY_INERTIA_DIAGONAL_KGM2 = (
    0.0935173556, 0.10022112, 0.0227794666)
RIGIDBODY_INERTIA_ROTATION_XYZW = (
    0.121077552, -0.0044240444, 0.0175926872, 0.9924773)
RIGIDBODY_YAW_INERTIA_KGM2 = 0.0957346
WHEEL_DAMPING_RATE_NMS = 0.25
WHEEL_MASS_KG = 0.109
WHEELHIT_SLIP_SPEED_REGULARIZER_MPS = 0.1
WHEEL_SUSPENSION_DISTANCE_M = 0.05
WHEEL_SUSPENSION_TARGET_POSITION = 0.5
WHEEL_SUSPENSION_SPRING_NPM = 500.0
WHEEL_SUSPENSION_DAMPER_NSPM = 100.0
WHEEL_COLLIDER_CENTER_Y_M = 0.025
WHEEL_SPRUNG_MASSES_KG = (0.817168832, 0.8160193,
                           0.9189912, 0.9178206)
WHEEL_COLLIDER_TARGET_TRAVEL_M = (
    WHEEL_SUSPENSION_TARGET_POSITION * WHEEL_SUSPENSION_DISTANCE_M)
WHEEL_FORWARD_FRICTION_CURVE = (0.15, 0.9, 0.25, 0.58, 0.8)
WHEEL_SIDEWAYS_FRICTION_CURVE = (0.01, 1.0, 0.1, 0.5, 1.0)
# F1TENTH.prefab WheelCollider transform mounts and their static local
# orientations. Quaternions are xyzw; front toe/camber is +/-5 degrees.
WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M = (
    (-0.118, 0.06, 0.17), (0.118, 0.06, 0.17),
    (-0.118, 0.05, -0.16), (0.118, 0.05, -0.16))
RIGIDBODY_CENTER_OF_MASS_BODY_M = (-0.00008, 0.06434, -0.00468)
WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW = (
    (0.0, 0.0, -math.sin(math.radians(2.5)), math.cos(math.radians(2.5))),
    (0.0, 0.0, math.sin(math.radians(2.5)), math.cos(math.radians(2.5))),
    (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
# Derived from the serialized COM height and the mean captured tire contact
# height on the flat open-plane player. This is not a serialized Unity field.
CAPTURED_COM_HEIGHT_ABOVE_ROAD_M = 0.07434
GRAVITY_MPS2 = 9.80665
UNITY_PHYSICS_GRAVITY_MPS2 = 9.81

# Coordinates are x-forward/y-left. These moment arms use the serialized
# WheelCollider centers relative to the configured Rigidbody COM, not the
# slightly different Ackermann wheelbase.
WHEEL_X_M = (0.17468, 0.17468, -0.15532, -0.15532)
WHEEL_Y_M = (0.118, -0.118, 0.118, -0.118)


@dataclass(frozen=True)
class WheelActuation:
    motor_torque_nm: tuple[float, float, float, float]
    brake_torque_nm: tuple[float, float, float, float]


def step_steering(steering_feedback_rad: float,
                  steering_command_rad: float,
                  dt_s: float) -> float:
    """Advance center steering one Unity physics step.

    The public command and feedback use positive-left radians. The bridge
    normalizes the command, while ``VehicleController.Steer()`` compares that
    normalized command to its sign-inverted internal angle before applying its
    asymmetric increment/clamp branches. Consequently this is not a symmetric
    rate limiter: some command changes snap directly to the target.
    """
    values = (steering_feedback_rad, steering_command_rad, dt_s)
    if not all(math.isfinite(value) for value in values) or dt_s <= 0.0:
        raise ValueError("steering state, command and positive dt must be finite")
    command_norm = steering_command_rad / STEERING_LIMIT_RAD
    internal_angle = -math.degrees(steering_feedback_rad)
    next_internal_angle = step_autonomous_steering_deg(
        internal_angle, internal_angle, command_norm, dt_s)
    return -math.radians(next_internal_angle)


def ackermann_angles(center_steering_rad: float
                     ) -> tuple[float, float, float, float]:
    """Return physical left/right front-wheel angles; rear wheels are fixed."""
    if not math.isfinite(center_steering_rad):
        raise ValueError("steering angle must be finite")
    delta = max(-STEERING_LIMIT_RAD,
                min(STEERING_LIMIT_RAD, center_steering_rad))
    tangent = math.tan(delta)
    numerator = 2.0 * WHEELBASE_ACKERMANN_M * tangent
    left = math.atan2(
        numerator, 2.0 * WHEELBASE_ACKERMANN_M - TRACK_M * tangent)
    right = math.atan2(
        numerator, 2.0 * WHEELBASE_ACKERMANN_M + TRACK_M * tangent)
    return left, right, 0.0, 0.0


def ackermann_center_from_wheel_angles(left_rad: float,
                                       right_rad: float) -> float:
    """Invert the source Ackermann equations for the center steering angle.

    The returned sign follows the center angle passed to
    :func:`ackermann_angles`. Unity's VehicleController internally negates the
    normalized autonomous command, so negate both source-captured wheel angles
    before passing them here to obtain the positive-left state convention.
    """
    if not math.isfinite(left_rad) or not math.isfinite(right_rad):
        raise ValueError("wheel steering angles must be finite")
    left_tangent = math.tan(left_rad)
    right_tangent = math.tan(right_rad)
    twice_wheelbase = 2.0 * WHEELBASE_ACKERMANN_M
    tangent_left = (twice_wheelbase * left_tangent
                    / (twice_wheelbase + TRACK_M * left_tangent))
    tangent_right = (twice_wheelbase * right_tangent
                     / (twice_wheelbase - TRACK_M * right_tangent))
    return math.atan(0.5 * (tangent_left + tangent_right))


def wheel_actuation(throttle_command_norm: float) -> WheelActuation:
    """Map one normalized command to the exact active CAWD/CAWB wheel torques.

    Unity branches on exact zero: zero requests 428 N m of brake torque at
    each wheel; any nonzero command requests equal quarter-share motor torque
    and clears brake torque. Negative throttle is negative motor torque.
    """
    if not math.isfinite(throttle_command_norm):
        raise ValueError("throttle command must be finite")
    throttle = max(-1.0, min(1.0, throttle_command_norm))
    if throttle == 0.0:
        return WheelActuation((0.0,) * 4, (MOTOR_TORQUE_NM,) * 4)
    per_wheel = MOTOR_TORQUE_NM * throttle / 4.0
    return WheelActuation((per_wheel,) * 4, (0.0,) * 4)


def wheel_contact_velocity(u_mps: float, v_mps: float, yaw_rate_radps: float,
                           wheel_index: int, steering_rad: float = 0.0
                           ) -> tuple[float, float]:
    """Return longitudinal/lateral ground speed in an individual tire frame.

    State and geometry use x-forward/y-left. ``wheel_index`` follows
    ``WHEEL_NAMES``. This is rigid-body kinematics only; tire slip and forces
    remain separate physical quantities.
    """
    values = (u_mps, v_mps, yaw_rate_radps, steering_rad)
    if (not all(math.isfinite(value) for value in values)
            or not 0 <= wheel_index < len(WHEEL_NAMES)):
        raise ValueError("invalid body state, steering angle or wheel index")
    vx_body = u_mps - yaw_rate_radps * WHEEL_Y_M[wheel_index]
    vy_body = v_mps + yaw_rate_radps * WHEEL_X_M[wheel_index]
    cosine = math.cos(steering_rad)
    sine = math.sin(steering_rad)
    return (cosine * vx_body + sine * vy_body,
            -sine * vx_body + cosine * vy_body)


def wheel_surface_speed(rpm: float, radius_m: float = WHEEL_RADIUS_M) -> float:
    """Convert signed WheelCollider RPM to circumferential speed."""
    if not math.isfinite(rpm) or not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("RPM and positive wheel radius must be finite")
    return rpm * (2.0 * math.pi / 60.0) * radius_m


def wheel_suspension_travel_from_body_pose(
    body_position_world_m: tuple[float, float, float],
    body_rotation_world_xyzw: tuple[float, float, float, float],
    wheel_transform_position_body_m: tuple[float, float, float],
    wheel_transform_rotation_body_xyzw: tuple[float, float, float, float],
    wheel_collider_center_local_m: tuple[float, float, float],
    ground_point_world_m: tuple[float, float, float],
    ground_normal_world: tuple[float, float, float],
    wheel_radius_m: float = WHEEL_RADIUS_M,
) -> float:
    """Intersect a WheelCollider suspension ray with a local ground plane.

    Returns distance down the wheel's local suspension axis from the
    transformed collider center to the position where the tire center is one
    radius above the plane. This is rigid geometry, not a learned state. It
    requires the ground point/normal and is valid only while the wheel is in
    contact and the ray approaches the plane.
    """
    vectors = (body_position_world_m, wheel_transform_position_body_m,
               wheel_collider_center_local_m, ground_point_world_m,
               ground_normal_world)
    quaternions = (body_rotation_world_xyzw,
                   wheel_transform_rotation_body_xyzw)
    if any(len(vector) != 3 or not all(math.isfinite(value) for value in vector)
           for vector in vectors):
        raise ValueError("wheel suspension geometry requires finite 3-vectors")
    if any(len(quaternion) != 4
           or not all(math.isfinite(value) for value in quaternion)
           for quaternion in quaternions):
        raise ValueError("wheel suspension geometry requires finite quaternions")
    if not math.isfinite(wheel_radius_m) or wheel_radius_m <= 0.0:
        raise ValueError("wheel radius must be positive")
    normal_norm = math.sqrt(sum(value * value for value in ground_normal_world))
    if normal_norm <= 0.0:
        raise ValueError("ground normal must be nonzero")
    normal = tuple(value / normal_norm for value in ground_normal_world)

    def rotate(quaternion: tuple[float, float, float, float],
               vector: tuple[float, float, float]) -> tuple[float, float, float]:
        qx, qy, qz, qw = quaternion
        qnorm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if qnorm <= 0.0:
            raise ValueError("wheel suspension quaternion must be nonzero")
        qx, qy, qz, qw = qx/qnorm, qy/qnorm, qz/qnorm, qw/qnorm
        vx, vy, vz = vector
        tx = 2.0 * (qy*vz - qz*vy)
        ty = 2.0 * (qz*vx - qx*vz)
        tz = 2.0 * (qx*vy - qy*vx)
        return (vx + qw*tx + qy*tz - qz*ty,
                vy + qw*ty + qz*tx - qx*tz,
                vz + qw*tz + qx*ty - qy*tx)

    center_body = rotate(wheel_transform_rotation_body_xyzw,
                         wheel_collider_center_local_m)
    down_wheel = rotate(wheel_transform_rotation_body_xyzw,
                        (0.0, -1.0, 0.0))
    base_body = tuple(wheel_transform_position_body_m[i] + center_body[i]
                      for i in range(3))
    base_world_offset = rotate(body_rotation_world_xyzw, base_body)
    down_world = rotate(body_rotation_world_xyzw, down_wheel)
    base_world = tuple(body_position_world_m[i] + base_world_offset[i]
                       for i in range(3))
    denominator = sum(down_world[i] * normal[i] for i in range(3))
    if abs(denominator) <= 1.0e-8:
        raise ValueError("suspension axis is parallel to the ground plane")
    signed_height = sum((base_world[i] - ground_point_world_m[i]) * normal[i]
                        for i in range(3))
    return (wheel_radius_m - signed_height) / denominator


def wheel_suspension_compression_m(
    travel_m: float,
    target_position: float = WHEEL_SUSPENSION_TARGET_POSITION,
    suspension_distance_m: float = WHEEL_SUSPENSION_DISTANCE_M,
) -> float:
    """Convert down-axis travel into compression from Unity's target point."""
    values = (travel_m, target_position, suspension_distance_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("suspension travel and settings must be finite")
    if not 0.0 <= target_position <= 1.0 or suspension_distance_m <= 0.0:
        raise ValueError("invalid suspension target or distance")
    target_travel = target_position * suspension_distance_m
    return target_travel - travel_m


def wheel_suspension_load_n(
    sprung_mass_kg: float,
    compression_from_target_m: float,
    compression_rate_mps: float,
    grounded: bool = True,
    spring_n_per_m: float = WHEEL_SUSPENSION_SPRING_NPM,
    damper_ns_per_m: float = WHEEL_SUSPENSION_DAMPER_NSPM,
) -> float:
    """Estimate WheelCollider normal load from spring travel and rate.

    The spring displacement is relative to the calibrated sprung-mass support
    position implied by the serialized target and wheel geometry. Positive
    compression means the wheel center has moved upward relative to the body;
    positive rate means it is compressing. The static sprung-mass load plus
    linear spring/damper increment matches the active Unity prefab and was
    checked against 1 kHz WheelHit loads on two source runs. Contact is
    unilateral, so an airborne wheel has no load.
    """
    values = (sprung_mass_kg, compression_from_target_m, compression_rate_mps,
              spring_n_per_m, damper_ns_per_m)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("suspension state and coefficients must be finite")
    if sprung_mass_kg < 0.0 or spring_n_per_m < 0.0 or damper_ns_per_m < 0.0:
        raise ValueError("suspension mass, spring and damping must be nonnegative")
    if not grounded:
        return 0.0
    return max(0.0, sprung_mass_kg * GRAVITY_MPS2
               + spring_n_per_m * compression_from_target_m
               + damper_ns_per_m * compression_rate_mps)


def wheel_contact_velocity_3d(
    body_velocity: tuple[float, float, float],
    body_angular_velocity: tuple[float, float, float],
    contact_position_from_body_origin: tuple[float, float, float],
    center_of_mass_from_body_origin: tuple[float, float, float],
    contact_forward: tuple[float, float, float],
    contact_sideways: tuple[float, float, float],
) -> tuple[float, float]:
    """Project rigid-body contact-point velocity into a WheelHit frame.

    Inputs use Unity's local (right, up, forward) axes.  The Rigidbody's linear
    velocity is specified at its center of mass, so the angular term must use
    the WheelHit point relative to the configured COM.  This rigid kinematics
    calculation is exact; it does not model tire forces or Unity's native
    WheelCollider slip solver.
    """
    vectors = (body_velocity, body_angular_velocity,
               contact_position_from_body_origin,
               center_of_mass_from_body_origin, contact_forward,
               contact_sideways)
    if any(len(vector) != 3 or not all(math.isfinite(value) for value in vector)
           for vector in vectors):
        raise ValueError("all contact-velocity vectors must be finite 3-vectors")

    relative = tuple(contact_position_from_body_origin[i] -
                     center_of_mass_from_body_origin[i] for i in range(3))
    wx, wy, wz = body_angular_velocity
    rx, ry, rz = relative
    rotational = (wy * rz - wz * ry,
                  wz * rx - wx * rz,
                  wx * ry - wy * rx)
    velocity = tuple(body_velocity[i] + rotational[i] for i in range(3))
    longitudinal = sum(velocity[i] * contact_forward[i] for i in range(3))
    lateral = sum(velocity[i] * contact_sideways[i] for i in range(3))
    return longitudinal, lateral


def wheel_hit_slip_candidate(
    longitudinal_contact_mps: float,
    lateral_contact_mps: float,
    wheel_rpm: float,
    radius_m: float = WHEEL_RADIUS_M,
    speed_regularization_mps: float = 1.0,
) -> tuple[float, float]:
    """Return a capture-calibrated candidate for Unity WheelHit slip labels.

    The longitudinal term is circumferential wheel speed minus contact speed,
    normalized by regularized forward speed.  The lateral term is an angle,
    not a linear ratio.  The 1 m/s regularizer and signs are empirically
    supported by the pinned Unity player captures in
    ``analyze_unity_contact_slip.py``; they are not claimed as the private
    native WheelCollider implementation.
    """
    values = (longitudinal_contact_mps, lateral_contact_mps, wheel_rpm,
              radius_m, speed_regularization_mps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("slip inputs must be finite")
    if radius_m <= 0.0 or speed_regularization_mps <= 0.0:
        raise ValueError("wheel radius and speed regularization must be positive")
    surface_speed = wheel_surface_speed(wheel_rpm, radius_m)
    denominator = abs(longitudinal_contact_mps) + speed_regularization_mps
    longitudinal_slip = (surface_speed - longitudinal_contact_mps) / denominator
    lateral_slip = math.atan2(lateral_contact_mps, denominator)
    return longitudinal_slip, lateral_slip


def wheel_hit_longitudinal_slip_from_state(
    longitudinal_contact_mps: float,
    wheel_rpm: float,
    radius_m: float = WHEEL_RADIUS_M,
    speed_regularizer_mps: float = WHEELHIT_SLIP_SPEED_REGULARIZER_MPS,
) -> float:
    """Reconstruct longitudinal WheelHit slip from current body/wheel state.

    The denominator is ``max(abs(R*omega), abs(v_contact)) + epsilon``. The
    0.1 m/s floor is identified from pinned Unity captures; it is not exposed
    by WheelCollider's public C# API. This candidate matched both the high-
    speed and observed brake-lock slip transitions across r07 and held-out
    r11, but its inputs must still be estimated from runtime odometry.
    """
    values = (longitudinal_contact_mps, wheel_rpm, radius_m,
              speed_regularizer_mps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("longitudinal slip inputs must be finite")
    if radius_m <= 0.0 or speed_regularizer_mps <= 0.0:
        raise ValueError("wheel radius and slip regularizer must be positive")
    surface_speed = wheel_surface_speed(wheel_rpm, radius_m)
    denominator = max(abs(surface_speed), abs(longitudinal_contact_mps)) \
        + speed_regularizer_mps
    return (surface_speed - longitudinal_contact_mps) / denominator


def guide_longitudinal_slip(
    contact_forward_mps: float,
    wheel_rpm: float,
    radius_m: float = WHEEL_RADIUS_M,
) -> float:
    """Evaluate AutoDRIVE's documented longitudinal slip equation.

    This ratio is singular at zero forward speed; the caller must use a
    separately identified near-rest contact model rather than silently
    regularizing the documented equation.
    """
    if not all(math.isfinite(value) for value in
               (contact_forward_mps, wheel_rpm, radius_m)):
        raise ValueError("longitudinal slip inputs must be finite")
    if radius_m <= 0.0:
        raise ValueError("wheel radius must be positive")
    if abs(contact_forward_mps) <= 1.0e-12:
        raise ValueError("documented longitudinal slip is singular at zero speed")
    surface_speed = wheel_surface_speed(wheel_rpm, radius_m)
    return (surface_speed - contact_forward_mps) / contact_forward_mps


def guide_lateral_slip(
    contact_forward_mps: float,
    contact_lateral_mps: float,
) -> float:
    """Evaluate AutoDRIVE's documented ``v_y / abs(v_x)`` tire slip."""
    if not all(math.isfinite(value) for value in
               (contact_forward_mps, contact_lateral_mps)):
        raise ValueError("lateral slip inputs must be finite")
    if abs(contact_forward_mps) <= 1.0e-12:
        raise ValueError("documented lateral slip is singular at zero speed")
    return contact_lateral_mps / abs(contact_forward_mps)


def wheel_friction_curve_value(
    slip: float,
    extremum_slip: float,
    extremum_value: float,
    asymptote_slip: float,
    asymptote_value: float,
    stiffness: float,
    zero_slip_tangent: float,
) -> float:
    """Evaluate the guide's two-segment cubic tire curve candidate.

    The first Hermite segment has the fitted zero-slip tangent and zero slope
    at the extremum.  The extremum-to-asymptote segment uses a cubic smoothstep
    (zero slope at both ends), then holds the asymptote.  Unity's serialized
    curve supplies the landmarks and stiffness; its native spline tangents
    remain private, so the tangent is an identified model parameter.
    """
    values = (slip, extremum_slip, extremum_value, asymptote_slip,
              asymptote_value, stiffness, zero_slip_tangent)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("friction-curve inputs must be finite")
    if (extremum_slip <= 0.0 or asymptote_slip <= extremum_slip
            or stiffness < 0.0 or zero_slip_tangent < 0.0):
        raise ValueError("invalid friction-curve landmarks or scale")

    magnitude = abs(slip)
    if magnitude < extremum_slip:
        t = magnitude / extremum_slip
        h10 = t**3 - 2.0 * t**2 + t
        h01 = -2.0 * t**3 + 3.0 * t**2
        value = (zero_slip_tangent * extremum_slip * h10
                 + extremum_value * h01)
    elif magnitude < asymptote_slip:
        t = ((magnitude - extremum_slip)
             / (asymptote_slip - extremum_slip))
        smooth = 3.0 * t**2 - 2.0 * t**3
        value = extremum_value + (asymptote_value - extremum_value) * smooth
    else:
        value = asymptote_value
    return math.copysign(value * stiffness, slip) if slip != 0.0 else 0.0


def planar_body_velocity_derivative(
    forward_speed_mps: float,
    lateral_speed_mps: float,
    yaw_rate_radps: float,
    force_forward_n: float,
    force_lateral_n: float,
    yaw_moment_nm: float,
    mass_kg: float,
    yaw_inertia_kgm2: float,
) -> tuple[float, float, float]:
    """Newton-Euler derivative of planar body-frame velocity at the COM."""
    values = (forward_speed_mps, lateral_speed_mps, yaw_rate_radps,
              force_forward_n, force_lateral_n, yaw_moment_nm, mass_kg,
              yaw_inertia_kgm2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("rigid-body dynamics inputs must be finite")
    if mass_kg <= 0.0 or yaw_inertia_kgm2 <= 0.0:
        raise ValueError("mass and yaw inertia must be positive")
    u_dot = force_forward_n / mass_kg + yaw_rate_radps * lateral_speed_mps
    v_dot = force_lateral_n / mass_kg - yaw_rate_radps * forward_speed_mps
    yaw_acceleration = yaw_moment_nm / yaw_inertia_kgm2
    return u_dot, v_dot, yaw_acceleration


def wheel_angular_acceleration(
    motor_torque_nm: float,
    brake_torque_nm: float,
    wheel_angular_speed_radps: float,
    tire_force_forward_n: float,
    radius_m: float,
    wheel_inertia_kgm2: float,
    damping_nms: float = 0.0,
) -> float:
    """Wheel-spin derivative for the explicit torque-balance teacher model."""
    values = (motor_torque_nm, brake_torque_nm, wheel_angular_speed_radps,
              tire_force_forward_n, radius_m, wheel_inertia_kgm2, damping_nms)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wheel rotational dynamics inputs must be finite")
    if radius_m <= 0.0 or wheel_inertia_kgm2 <= 0.0 or damping_nms < 0.0:
        raise ValueError("invalid wheel radius, inertia, or damping")
    brake_sign = math.copysign(1.0, wheel_angular_speed_radps) \
        if wheel_angular_speed_radps != 0.0 else 0.0
    net_torque = (motor_torque_nm - brake_torque_nm * brake_sign
                  - radius_m * tire_force_forward_n
                  - damping_nms * wheel_angular_speed_radps)
    return net_torque / wheel_inertia_kgm2


def wheel_angular_velocity_step(
    wheel_angular_speed_radps: float,
    motor_torque_nm: float,
    brake_torque_nm: float,
    tire_force_forward_n: float,
    radius_m: float,
    wheel_inertia_kgm2: float,
    damping_nms: float,
    dt_s: float,
) -> float:
    """Advance wheel spin with a unilateral brake-lock constraint.

    The free step follows ``I*domega = motor - r*Fx - c*omega``. Brake torque
    is then applied as an opposing angular impulse, but cannot reverse a wheel
    that it stops. This matters for the active Unity controller's large brake
    torque at exactly zero throttle. The inertia is an identified effective
    WheelCollider parameter when used with captured RPM, not one inferred
    directly from serialized wheel mass.
    """
    values = (wheel_angular_speed_radps, motor_torque_nm, brake_torque_nm,
              tire_force_forward_n, radius_m, wheel_inertia_kgm2,
              damping_nms, dt_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("wheel state, torques and parameters must be finite")
    if (brake_torque_nm < 0.0 or radius_m <= 0.0
            or wheel_inertia_kgm2 <= 0.0 or damping_nms < 0.0 or dt_s <= 0.0):
        raise ValueError("invalid brake, radius, inertia, damping or timestep")

    free_speed = wheel_angular_speed_radps + dt_s * (
        motor_torque_nm - radius_m * tire_force_forward_n
        - damping_nms * wheel_angular_speed_radps) / wheel_inertia_kgm2
    brake_impulse_speed = brake_torque_nm * dt_s / wheel_inertia_kgm2
    if free_speed > 0.0:
        return max(0.0, free_speed - brake_impulse_speed)
    if free_speed < 0.0:
        return min(0.0, free_speed + brake_impulse_speed)
    return 0.0


def inertia_about_body_axis(
    principal_inertia_kgm2: tuple[float, float, float],
    principal_axes_rotation_xyzw: tuple[float, float, float, float],
    body_axis_xyz: tuple[float, float, float],
) -> float:
    """Project Unity's rotated principal inertia tensor onto a body axis.

    Unity serializes Rigidbody inertia as a principal diagonal plus a
    quaternion locating those principal axes in the body frame. A component
    of the diagonal is not generally the yaw inertia. The supplied axis must
    be unit length; this function rotates it into principal coordinates and
    evaluates ``sum(I_i * axis_i**2)``.
    """
    values = (*principal_inertia_kgm2,
              *principal_axes_rotation_xyzw, *body_axis_xyz)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("inertia tensor, rotation, and axis must be finite")
    if any(value <= 0.0 for value in principal_inertia_kgm2):
        raise ValueError("principal moments must be positive")
    qx, qy, qz, qw = principal_axes_rotation_xyzw
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm <= 0.0:
        raise ValueError("principal-axes quaternion must be nonzero")
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    ax, ay, az = body_axis_xyz
    axis_norm = math.sqrt(ax*ax + ay*ay + az*az)
    if abs(axis_norm - 1.0) > 1.0e-6:
        raise ValueError("body axis must be unit length")
    # q^{-1} * axis * q rotates a body-frame vector into the principal frame.
    tx = 2.0 * (qy*az - qz*ay)
    ty = 2.0 * (qz*ax - qx*az)
    tz = 2.0 * (qx*ay - qy*ax)
    px = ax - qw*tx + (qy*tz - qz*ty)
    py = ay - qw*ty + (qz*tx - qx*tz)
    pz = az - qw*tz + (qx*ty - qy*tx)
    return (principal_inertia_kgm2[0] * px*px
            + principal_inertia_kgm2[1] * py*py
            + principal_inertia_kgm2[2] * pz*pz)


def inertia_tensor_body(
    principal_inertia_kgm2: tuple[float, float, float],
    principal_axes_rotation_xyzw: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """Rotate Unity's principal inertia diagonal into Rigidbody body axes."""
    values = (*principal_inertia_kgm2, *principal_axes_rotation_xyzw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("inertia tensor and rotation must be finite")
    if any(value <= 0.0 for value in principal_inertia_kgm2):
        raise ValueError("principal moments must be positive")
    qx, qy, qz, qw = principal_axes_rotation_xyzw
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm <= 0.0:
        raise ValueError("principal-axes quaternion must be nonzero")
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    rotation = (
        (1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),
         2.0*(qx*qz + qy*qw)),
        (2.0*(qx*qy + qz*qw), 1.0 - 2.0*(qx*qx + qz*qz),
         2.0*(qy*qz - qx*qw)),
        (2.0*(qx*qz - qy*qw), 2.0*(qy*qz + qx*qw),
         1.0 - 2.0*(qx*qx + qy*qy)),
    )
    return tuple(tuple(sum(rotation[row][axis]
                           * principal_inertia_kgm2[axis]
                           * rotation[column][axis]
                           for axis in range(3))
                       for column in range(3))
                 for row in range(3))


def rigid_body_angular_torque(
    angular_velocity_body_radps: tuple[float, float, float],
    angular_acceleration_body_radps2: tuple[float, float, float],
    inertia_body_kgm2: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    """Compute body torque ``I*alpha + omega x (I*omega)``.

    This is the full 3-D Newton–Euler equation for a rigid body with a
    body-fixed inertia tensor; it retains off-diagonal inertia and
    gyroscopic coupling instead of treating yaw as an isolated scalar state.
    """
    vectors = (angular_velocity_body_radps, angular_acceleration_body_radps2)
    if any(len(vector) != 3 or not all(math.isfinite(value) for value in vector)
           for vector in vectors):
        raise ValueError("angular velocity and acceleration must be finite 3-vectors")
    if (len(inertia_body_kgm2) != 3
            or any(len(row) != 3 for row in inertia_body_kgm2)
            or not all(math.isfinite(value) for row in inertia_body_kgm2
                       for value in row)):
        raise ValueError("inertia tensor must be a finite 3x3 matrix")
    omega = angular_velocity_body_radps
    alpha = angular_acceleration_body_radps2
    inertia_omega = tuple(sum(inertia_body_kgm2[i][j] * omega[j]
                              for j in range(3)) for i in range(3))
    inertia_alpha = tuple(sum(inertia_body_kgm2[i][j] * alpha[j]
                              for j in range(3)) for i in range(3))
    cross = (omega[1] * inertia_omega[2] - omega[2] * inertia_omega[1],
             omega[2] * inertia_omega[0] - omega[0] * inertia_omega[2],
             omega[0] * inertia_omega[1] - omega[1] * inertia_omega[0])
    return tuple(inertia_alpha[i] + cross[i] for i in range(3))
