from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Any

import casadi as ca
import numpy as np

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
    UNITY_PHYSICS_GRAVITY_MPS2,
    MOTOR_TORQUE_NM,
    RIGIDBODY_DRAG_PER_S,
    RIGIDBODY_CENTER_OF_MASS_BODY_M,
    RIGIDBODY_MASS_KG,
    RIGIDBODY_YAW_INERTIA_KGM2,
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
    WHEEL_COLLIDER_CENTER_Y_M,
    WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
    WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW,
    WHEEL_DAMPING_RATE_NMS,
    WHEEL_FORWARD_FRICTION_CURVE,
    WHEEL_MASS_KG,
    WHEEL_RADIUS_M,
    WHEELBASE_ACKERMANN_M,
    TRACK_M,
    WHEEL_SUSPENSION_DISTANCE_M,
    WHEEL_SUSPENSION_DAMPER_NSPM,
    WHEEL_SUSPENSION_SPRING_NPM,
    WHEEL_SUSPENSION_TARGET_POSITION,
    WHEEL_SIDEWAYS_FRICTION_CURVE,
    WHEEL_X_M,
    WHEEL_Y_M,
)


ROOT = Path(__file__).resolve().parents[3]
UNITY_SOURCE_CAPTURE = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
_WHEEL_KEYS = ("front_left", "front_right", "rear_left", "rear_right")
PHYSX_TOLERANCE_SCALE_LENGTH_M = 1.0
PHYSX_APPLIED_TORQUE_SLIP_OFFSET_M = 0.1
PHYSX_LATERAL_SLIP_SPEED_OFFSET_MPS = 1.0
PHYSX_TIRE_LOAD_FILTER = (0.0, 0.2308, 3.0, 3.0)


def _unity_wheel_inertia_kgm2(mass_kg: float, effective_radius_m: float
                              ) -> float:
    """Unity WheelCollider wrapper's source-derived solid-wheel inertia."""
    return 0.5 * mass_kg * effective_radius_m**2


def _physx_compute_sprung_masses(wheel_centres_body_m,
                                 centre_of_mass_body_m,
                                 total_mass_kg: float
                                 ) -> tuple[float, float, float, float]:
    """Port PhysX 4.1's four-wheel minimum-variance setup solve."""
    centres = np.asarray(wheel_centres_body_m, dtype=np.float64)
    com = np.asarray(centre_of_mass_body_m, dtype=np.float64)
    offsets = centres - com
    # PhysX gravityDirection=1 (Unity +Y up) uses d0=Z, d1=X.
    constraints = np.vstack((offsets[:, (2, 0)].T,
                             np.ones((1, len(offsets)))))
    count = len(offsets)
    system = np.zeros((count + 3, count + 3), dtype=np.float64)
    system[:3, :count] = constraints
    system[3:, :count] = 2.0 * np.eye(count)
    system[3:, count:] = constraints.T
    rhs = np.concatenate((
        np.asarray((0.0, 0.0, total_mass_kg)),
        np.full(count, 2.0 * total_mass_kg / count)))
    return tuple(float(value) for value in np.linalg.solve(system, rhs)[:4])


def _capture_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("# "):
                break
            key, separator, value = line[2:].strip().partition("=")
            if separator:
                result[key] = value
    return result


def _rotation_matrix_xyzw(quaternion: tuple[float, ...]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.asarray((
        (1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),
         2.0 * (x*z + y*w)),
        (2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z),
         2.0 * (y*z - x*w)),
        (2.0 * (x*z - y*w), 2.0 * (y*z + x*w),
         1.0 - 2.0 * (x*x + y*y)),
    ), dtype=np.float64)


def _capture_body_height_median(capture: Path, sample_count: int = 1000) -> float:
    with capture.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith("#"))
        values = []
        for index, row in enumerate(rows):
            if index >= sample_count:
                break
            values.append(float(row["pos_world_y_m"]))
    if not values:
        raise ValueError("Unity capture has no body-height samples")
    return float(np.median(values))


@dataclass(frozen=True)
class UnityWheelModel:
    mass_kg: float
    yaw_inertia_kgm2: float
    drag_per_s: float
    wheel_radius_m: float
    wheel_radius_scale: float
    wheel_inertia_kgm2: float
    wheel_damping_nms: float
    steering_limit_rad: float
    steering_rate_radps: float
    wheelbase_m: float
    track_m: float
    motor_torque_nm: float
    body_com_height_m: float
    sprung_mass_kg: tuple[float, float, float, float]
    forward_curve: tuple[tuple[float, ...], ...]
    lateral_curve: tuple[tuple[float, ...], ...]
    static_normal_load_n: tuple[float, float, float, float]
    suspension_target_travel_m: tuple[float, float, float, float]
    suspension_spring_n_per_m: tuple[float, float, float, float]
    suspension_damper_ns_per_m: tuple[float, float, float, float]
    suspension_distance_m: float
    suspension_base_xyz_body_m: tuple[tuple[float, float, float], ...]
    suspension_camber_rad: tuple[float, float, float, float]
    suspension_ground_y_m: float
    suspension_reference_body_height_m: float
    inertia_pitch_kgm2: float
    inertia_roll_kgm2: float

    @classmethod
    def from_source_capture(cls, capture: Path = UNITY_SOURCE_CAPTURE
                            ) -> "UnityWheelModel":
        """Build configured wheel data from Unity source, not fitted values."""
        required = (capture,)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing Unity model source artifacts: "
                                    + ", ".join(missing))
        metadata = _capture_metadata(capture)
        captured_forward = tuple(tuple(float(value) for value in metadata[
            f"{name}_forward_curve"].split(",")) for name in _WHEEL_KEYS)
        captured_lateral = tuple(tuple(float(value) for value in metadata[
            f"{name}_sideways_curve"].split(",")) for name in _WHEEL_KEYS)
        if any(len(curve) != 5
               for curve in (*captured_forward, *captured_lateral)):
            raise ValueError("Unity capture friction-curve metadata is malformed")
        forward = (WHEEL_FORWARD_FRICTION_CURVE,) * 4
        lateral = (WHEEL_SIDEWAYS_FRICTION_CURVE,) * 4
        if (not np.allclose(captured_forward, forward, rtol=0.0, atol=1e-6)
                or not np.allclose(captured_lateral, lateral,
                                   rtol=0.0, atol=1e-6)):
            raise ValueError("capture curves differ from active Unity config")
        radii = tuple(float(metadata[f"{name}_radius_m"])
                      for name in _WHEEL_KEYS)
        if max(radii) - min(radii) > 1.0e-6:
            raise ValueError("the four Unity wheel radii are not equal")
        if abs(radii[0] - WHEEL_RADIUS_M) > 1.0e-6:
            raise ValueError("capture wheel radius differs from active prefab")

        collider_center = np.asarray((0.0, WHEEL_COLLIDER_CENTER_Y_M, 0.0))
        wheel_centres = tuple(
            np.asarray(position, dtype=np.float64)
            + _rotation_matrix_xyzw(rotation) @ collider_center
            for position, rotation in zip(
                WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
                WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW))
        sprung = _physx_compute_sprung_masses(
            wheel_centres, RIGIDBODY_CENTER_OF_MASS_BODY_M,
            RIGIDBODY_MASS_KG)
        static_load = tuple(mass * UNITY_PHYSICS_GRAVITY_MPS2
                             for mass in sprung)
        target_travel = (WHEEL_SUSPENSION_TARGET_POSITION
                         * WHEEL_SUSPENSION_DISTANCE_M,) * 4
        captured_spring = tuple(float(metadata[f"{name}_spring_n_per_m"])
                                for name in _WHEEL_KEYS)
        captured_damper = tuple(float(metadata[f"{name}_damper_ns_per_m"])
                                for name in _WHEEL_KEYS)
        spring = (WHEEL_SUSPENSION_SPRING_NPM,) * 4
        damper = (WHEEL_SUSPENSION_DAMPER_NSPM,) * 4
        if (not np.allclose(captured_spring, spring, rtol=0.0, atol=1e-6)
                or not np.allclose(captured_damper, damper,
                                   rtol=0.0, atol=1e-6)):
            raise ValueError("capture suspension differs from active config")
        base_xyz = []
        camber = []
        for position, rotation in zip(
                WHEEL_COLLIDER_TRANSFORM_POSITION_BODY_M,
                WHEEL_COLLIDER_TRANSFORM_ROTATION_BODY_XYZW):
            rotation_matrix = _rotation_matrix_xyzw(rotation)
            base_xyz.append(tuple(np.asarray(position, dtype=np.float64)
                                  + rotation_matrix @ collider_center))
            down_axis = rotation_matrix @ np.asarray((0.0, -1.0, 0.0))
            camber.append(float(np.arctan2(down_axis[0], -down_axis[1])))
        inertia_diagonal = np.asarray([
            float(value) for value in metadata[
                "inertia_tensor_body"].split(",")], dtype=np.float64)
        inertia_rotation = tuple(float(value) for value in metadata[
            "inertia_tensor_rotation_body"].split(","))
        inertia_rotation_matrix = _rotation_matrix_xyzw(inertia_rotation)
        inertia_body = (inertia_rotation_matrix @ np.diag(inertia_diagonal)
                        @ inertia_rotation_matrix.T)
        return cls(
            mass_kg=RIGIDBODY_MASS_KG,
            yaw_inertia_kgm2=RIGIDBODY_YAW_INERTIA_KGM2,
            drag_per_s=RIGIDBODY_DRAG_PER_S,
            wheel_radius_m=WHEEL_RADIUS_M,
            wheel_radius_scale=1.0,
            wheel_inertia_kgm2=_unity_wheel_inertia_kgm2(
                WHEEL_MASS_KG, WHEEL_RADIUS_M),
            wheel_damping_nms=WHEEL_DAMPING_RATE_NMS,
            steering_limit_rad=STEERING_LIMIT_RAD,
            steering_rate_radps=STEERING_RATE_RADPS,
            wheelbase_m=WHEELBASE_ACKERMANN_M,
            track_m=TRACK_M,
            motor_torque_nm=MOTOR_TORQUE_NM,
            body_com_height_m=CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
            sprung_mass_kg=sprung,
            forward_curve=forward,
            lateral_curve=lateral,
            static_normal_load_n=static_load,
            suspension_target_travel_m=target_travel,
            suspension_spring_n_per_m=spring,
            suspension_damper_ns_per_m=damper,
            suspension_distance_m=WHEEL_SUSPENSION_DISTANCE_M,
            suspension_base_xyz_body_m=tuple(base_xyz),
            suspension_camber_rad=tuple(camber),
            suspension_ground_y_m=0.0,
            suspension_reference_body_height_m=_capture_body_height_median(
                capture),
            inertia_pitch_kgm2=float(inertia_body[0, 0]),
            inertia_roll_kgm2=float(inertia_body[2, 2]),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source_capture": str(UNITY_SOURCE_CAPTURE.relative_to(ROOT)),
            "source_config": {
                "active_prefab": (
                    "external/autodrive-unity-source/Assets/Prefabs/"
                    "F1TENTH/F1TENTH.prefab"),
                "physx_source_commit": (
                    "a2c0428acab643e60618c681b501e86f7fd558cc"),
            },
            "mass_kg": self.mass_kg,
            "yaw_inertia_kgm2": self.yaw_inertia_kgm2,
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_radius_scale": self.wheel_radius_scale,
            "wheel_inertia_kgm2": self.wheel_inertia_kgm2,
            "wheel_inertia_derivation": (
                "Unity WheelCollider::AddWheelToExistingVehicle sets PhysX "
                "mMOI=0.5*mMass*(effectiveRadius)^2; active F1TENTH prefab "
                "has mass 0.109 kg, radius 0.059 m, unit transform scale"),
            "wheel_damping_nms": self.wheel_damping_nms,
            "wheel_mass_kg": WHEEL_MASS_KG,
            "sprung_masses_kg": self.sprung_mass_kg,
            "sprung_mass_derivation": (
                "Unity WheelCollider::UpdateVehicleSpringCMOffsetsAndMasses "
                "passes transformed wheel-centre offsets, Rigidbody COM and "
                "mass to PxVehicleComputeSprungMasses(gravityDirection=1)"),
            "suspension_target_position": WHEEL_SUSPENSION_TARGET_POSITION,
            "wheel_spin_damping_time_constant_s": (
                self.wheel_inertia_kgm2 / self.wheel_damping_nms),
            "steering_limit_rad": self.steering_limit_rad,
            "steering_rate_radps": self.steering_rate_radps,
            "motor_torque_total_nm": self.motor_torque_nm,
            "forward_curves": self.forward_curve,
            "lateral_curves": self.lateral_curve,
            "tire_force_model": (
                "Unity VehicleComputeTireForce from matching UnityPlayer "
                "symbol-bearing build; curve response and combined-slip "
                "ellipse are direct native-code translations"),
            "tire_load_filter": {
                "points": PHYSX_TIRE_LOAD_FILTER,
                "rest_load_kg": tuple(
                    sprung_mass + WHEEL_MASS_KG
                    for sprung_mass in self.sprung_mass_kg),
                "source": "PhysX 4.1 PxVehicleTireLoadFilterData default; "
                          "rest load is wheel mass plus sprung mass",
            },
            "load_model": (
                "PhysX 4.1 source spring/damper force for a flat horizontal "
                "contact plane; uses source ray direction, jounce, gravity "
                "projection, and contact-normal projection"),
            "load_model_id": "unity_sprung_mass_spring_damper_heave_roll_pitch_v1",
            "suspension": {
                "target_travel_m": self.suspension_target_travel_m,
                "spring_n_per_m": self.suspension_spring_n_per_m,
                "damper_ns_per_m": self.suspension_damper_ns_per_m,
                "travel_distance_m": self.suspension_distance_m,
                "wheel_camber_rad": self.suspension_camber_rad,
                "reference_body_height_m": (
                    self.suspension_reference_body_height_m),
                "inertia_pitch_kgm2": self.inertia_pitch_kgm2,
                "inertia_roll_kgm2": self.inertia_roll_kgm2,
                "force_application_point_distance_m": 0.0,
            },
            "model_boundary": (
                "The Unity tire-force callback is reverse-engineered from the "
                "matching native binary. Slip generation, PhysX wheel-state "
                "integration and raycast contact selection are not yet "
                "reproduced end-to-end; the flat-plane suspension force and "
                "default tire-load filter are now source-translated. Chassis "
                "contact integration is not reproduced. "
                "Wheel inertia/damping, sprung masses, and rest travel now "
                "come from the Unity wrapper, PhysX 4.1 source, and prefab. "
                "The capture supplies runtime Rigidbody inertia and nominal "
                "body-height initialization, not fitted coefficients."),
            "tire_slip_coordinates": (
                "PhysX 4.1 applied-torque branch (Unity NoDrive has motor or "
                "brake torque at each optimizer node): longSlip=(R*omega-vx)/"
                "(max(abs(vx),abs(R*omega))+0.1m); lateralSlip=atan(vy/"
                "(abs(vx)+1m)). The remaining approximation is the planar "
                "contact-point/tire-axis velocity reconstruction; low-speed "
                "sticky constraints are not modeled."),
        }


def _unity_friction_curve_response(slip, curve: tuple[float, ...]):
    """Translate Unity's native WheelFrictionCurve callback, not a fit."""
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = curve
    magnitude = ca.fabs(slip)
    first_t = magnitude / extremum_slip
    first = extremum_value * (first_t + first_t**2 - first_t**3)
    decay_t = (magnitude - extremum_slip) / (asymptote_slip - extremum_slip)
    smooth = 3.0 * decay_t**2 - 2.0 * decay_t**3
    sign = ca.if_else(slip >= 0.0, 1.0, -1.0)
    second = extremum_value + (asymptote_value - extremum_value) * smooth
    return stiffness * ca.if_else(
        magnitude < extremum_slip, sign * first,
        sign * ca.if_else(magnitude < asymptote_slip,
                          second, asymptote_value))


def _unity_tire_force_shader_casadi(long_slip, lateral_slip, tire_load,
                                    wheel_radius, forward_curve,
                                    lateral_curve):
    """Unity native callback outputs: wheel torque, Fx, Fy and aligning M."""
    long_response = _unity_friction_curve_response(long_slip, forward_curve)
    lateral_response = _unity_friction_curve_response(lateral_slip,
                                                       lateral_curve)
    # Native unityCurveTireForce: normalize un-stiffened curve responses,
    # THEN apply each WheelFrictionCurve.stiffness to the final force. The
    # helper above already multiplies stiffness, so divide it out here.
    raw_long_response = long_response / forward_curve[4]
    raw_lateral_response = lateral_response / lateral_curve[4]
    utilization_sq = (
        (raw_long_response / forward_curve[1])**2
        + (raw_lateral_response / lateral_curve[1])**2)
    combined_slip_scale = 1.0 / ca.sqrt(ca.fmax(1.0, utilization_sq))
    force_long = tire_load * raw_long_response * combined_slip_scale * forward_curve[4]
    force_lateral = -tire_load * raw_lateral_response * combined_slip_scale * lateral_curve[4]
    wheel_torque = -force_long * wheel_radius
    return wheel_torque, force_long, force_lateral, 0.0


def _physx_applied_torque_slips_casadi(long_speed, lateral_speed,
                                       wheel_surface_speed):
    """PhysX 4.1 non-tank slip equations used when torque/brake is active."""
    long_speed_abs = ca.fabs(long_speed)
    wheel_speed_abs = ca.fabs(wheel_surface_speed)
    long_denominator = (
        ca.fmax(long_speed_abs, wheel_speed_abs)
        + PHYSX_APPLIED_TORQUE_SLIP_OFFSET_M
        * PHYSX_TOLERANCE_SCALE_LENGTH_M)
    long_slip = (wheel_surface_speed - long_speed) / long_denominator
    # Optimizer wheel-axis lateral_speed is LEFT-positive. PhysX/Unity
    # reports applied-torque lateral slip in the RIGHT-positive native basis.
    # A native 2.4 m/s 481-sample flight confirms the sign inversion on both
    # steered front wheels; no tire friction coefficients are changed.
    lateral_slip = -ca.atan(
        lateral_speed
        / (long_speed_abs + PHYSX_LATERAL_SLIP_SPEED_OFFSET_MPS
           * PHYSX_TOLERANCE_SCALE_LENGTH_M))
    return long_slip, lateral_slip


def physx_no_drive_wheel_speed_interval_casadi(
        omega, drive_torque, raw_brake_torque, tire_torque,
        moment_of_inertia, damping_rate, duration_s,
        physics_step_s: float = 0.001,
        substeps_per_physics_step: int = 1):
    """Advance wheel speed with PhysX NoDrive's implicit Euler rule.

    For constant torque over an interval this is the closed-form composition
    of PhysX ``integrateNoDriveWheelSpeeds`` at the internal vehicle
    substep. ``physics_step_s`` is Unity's outer fixed step; PhysX divides it
    by ``substeps_per_physics_step``. Integer internal-step counts are
    source-equivalent; the fractional final step is smoothly extended so
    CasADi can differentiate through the optimizer's variable segment
    duration. Tire torque is held at the supplied interval average.

    PhysX applies the raw brake torque against the current wheel direction,
    then locks the wheel if the integrated speed changes sign while braking.
    """
    if physics_step_s <= 0.0 or substeps_per_physics_step < 1:
        raise ValueError("physics step and vehicle substep count must be positive")
    internal_step_s = physics_step_s / substeps_per_physics_step
    dt_over_inertia = duration_s / moment_of_inertia
    signed_brake = ca.if_else(
        omega > 0.0, -raw_brake_torque,
        ca.if_else(omega < 0.0, raw_brake_torque, 0.0))
    total_torque = drive_torque + tire_torque + signed_brake
    damping_step = damping_rate * internal_step_s / moment_of_inertia
    step_ratio = 1.0 / (1.0 + damping_step)
    step_count = duration_s / internal_step_s
    decay = ca.exp(step_count * ca.log(step_ratio))
    equilibrium = total_torque / damping_rate
    undamped_result = omega + dt_over_inertia * total_torque
    result = ca.if_else(
        damping_rate > 0.0,
        decay * omega + (1.0 - decay) * equilibrium,
        undamped_result)
    braking_crossed_zero = ca.logic_and(
        raw_brake_torque > 0.0, omega * result <= 0.0)
    return ca.if_else(braking_crossed_zero, 0.0, result)


def _wheel_surface_speed_from_physx_long_slip(long_speed, long_slip):
    """Invert the applied-torque slip equation for forward-driving seeds."""
    speed = np.asarray(long_speed, dtype=np.float64)
    slip = np.asarray(long_slip, dtype=np.float64)
    offset = (PHYSX_APPLIED_TORQUE_SLIP_OFFSET_M
              * PHYSX_TOLERANCE_SCALE_LENGTH_M)
    # For the forward-running wheel speeds used by the raceline OCP, positive
    # slip has |R*omega| >= |v|; negative slip has |R*omega| <= |v|.
    return np.where(
        slip >= 0.0,
        (speed + offset * slip) / (1.0 - slip),
        speed + slip * (np.abs(speed) + offset))


def _wheel_surface_speed_from_autodrive_guide_slip(long_speed, long_slip):
    """Invert the legacy guide coordinate Sx=(R*omega-vx)/max(vx, 0.5)."""
    speed = np.asarray(long_speed, dtype=np.float64)
    slip = np.asarray(long_slip, dtype=np.float64)
    return speed + slip * np.maximum(speed, 0.5)


def tire_forces_casadi(model: UnityWheelModel, u, v, yaw_rate, steering,
                       wheel_omega, suspension_forces, roll_rate=0.0,
                       pitch_rate=0.0):
    """Four-wheel force/slip equations from source config and planar state."""
    tangent = ca.tan(steering)
    numerator = 2.0 * model.wheelbase_m * tangent
    steer_left = ca.atan2(
        numerator, 2.0 * model.wheelbase_m - model.track_m * tangent)
    steer_right = ca.atan2(
        numerator, 2.0 * model.wheelbase_m + model.track_m * tangent)
    wheel_steering = (steer_left, steer_right, 0.0, 0.0)
    fx_wheels = []
    fy_wheels = []
    body_fx = []
    body_fy = []
    slip_x_values = []
    slip_y_values = []
    for index, angle in enumerate(wheel_steering):
        # Native tire slip uses contact-point velocity = COM velocity +
        # angular_velocity x contact_arm, not the previous planar COM speed.
        # The tire-force application point is one radius below its wheel
        # centre; both mount Y and Rigidbody COM Y are Unity up-positive.
        contact_arm_down_m = (RIGIDBODY_CENTER_OF_MASS_BODY_M[1]
                              + model.wheel_radius_m
                              - model.suspension_base_xyz_body_m[index][1])
        # Body axes are forward, LEFT, up. A downward contact arm yields
        # forward correction -pitch_rate*depth and lateral +roll_rate*depth.
        vx_body = (u - yaw_rate * WHEEL_Y_M[index]
                   - pitch_rate * contact_arm_down_m)
        vy_body = (v + yaw_rate * WHEEL_X_M[index]
                   + roll_rate * contact_arm_down_m)
        forward_speed = ca.cos(angle) * vx_body + ca.sin(angle) * vy_body
        lateral_speed = -ca.sin(angle) * vx_body + ca.cos(angle) * vy_body
        wheel_surface = (wheel_omega[index] * model.wheel_radius_m
                         * model.wheel_radius_scale)
        slip_x, slip_y = _physx_applied_torque_slips_casadi(
            forward_speed, lateral_speed, wheel_surface)
        # Native callback tireLoad equals PxVehicleWheelQueryResult.suspSpringForce
        # in the source-matched 1 ms replay (all 1,924 measured wheel samples).
        # The older symbolic filter inflated that load by ~1.49 N. Do not
        # apply it again: native callback already receives the effective load.
        tire_load = suspension_forces[index]
        _, fx, fy, _ = _unity_tire_force_shader_casadi(
            slip_x, slip_y, tire_load, model.wheel_radius_m,
            model.forward_curve[index], model.lateral_curve[index])
        # The Unity/PhysX callback returns lateral force in wheel RIGHT,
        # but IPOPT chassis axes are forward/LEFT/up. Ackermann steer angle
        # is LEFT-positive. Right unit vector=(sin(angle), -cos(angle)).
        # Preserve callback raw fy in fy_wheels for contact parity checks.
        fx_body = fx * ca.cos(angle) + fy * ca.sin(angle)
        fy_body = fx * ca.sin(angle) - fy * ca.cos(angle)
        fx_wheels.append(fx)
        fy_wheels.append(fy)
        body_fx.append(fx_body)
        body_fy.append(fy_body)
        slip_x_values.append(slip_x)
        slip_y_values.append(slip_y)
    total_fx = sum(body_fx)
    total_fy = sum(body_fy)
    yaw_moment = sum(WHEEL_X_M[i] * body_fy[i]
                     - WHEEL_Y_M[i] * body_fx[i] for i in range(4))
    return (total_fx, total_fy, yaw_moment, tuple(fx_wheels), tuple(fy_wheels),
            tuple(slip_x_values), tuple(slip_y_values))


def normal_loads_casadi(model: UnityWheelModel, ax, ay):
    """Quasi-static spring-pair load transfer; no learned lateral cap."""
    longitudinal = (model.mass_kg * model.body_com_height_m * ax
                    / (2.0 * model.wheelbase_m))
    lateral = (model.mass_kg * model.body_com_height_m * ay
               / (4.0 * model.track_m))
    front_sign = (-1.0, -1.0, 1.0, 1.0)
    side_sign = (-1.0, 1.0, -1.0, 1.0)
    return tuple(
        model.static_normal_load_n[i]
        + front_sign[i] * longitudinal
        + side_sign[i] * lateral
        for i in range(4)
    )


def _physx_filtered_tire_load_casadi(suspension_force, sprung_mass_kg,
                                     wheel_mass_kg, gravity_mps2):
    """PhysX 4.1 normalized-load filter and tire rest-load conversion."""
    minimum_load, minimum_filtered, maximum_load, maximum_filtered = (
        PHYSX_TIRE_LOAD_FILTER)
    rest_load_n = gravity_mps2 * (sprung_mass_kg + wheel_mass_kg)
    normalized_load = suspension_force / rest_load_n
    filtered_normalized = ca.if_else(
        normalized_load <= minimum_load,
        minimum_filtered,
        ca.if_else(
            normalized_load >= maximum_load,
            maximum_filtered,
            minimum_filtered + (normalized_load - minimum_load)
            * (maximum_filtered - minimum_filtered)
            / (maximum_load - minimum_load)))
    return filtered_normalized * rest_load_n


def suspension_travel_casadi(model: UnityWheelModel, body_height,
                             body_height_rate, roll, roll_rate, pitch,
                             pitch_rate):
    """PhysX raycast suspension travel and rate on a flat ground plane.

    PhysX casts along the chassis-local down axis, not the cambered wheel axis.
    ``travel`` is the ray-derived jounce coordinate expressed from max
    compression: max compression is zero and max droop is suspensionDistance.
    Coordinates and mount geometry follow F1TENTH.prefab (Unity x-right,
    y-up, z-forward); body yaw does not affect height over a flat plane.
    """
    sin_roll, cos_roll = ca.sin(roll), ca.cos(roll)
    sin_pitch, cos_pitch = ca.sin(pitch), ca.cos(pitch)
    travels = []
    travel_rates = []
    for index in range(4):
        base_x, base_y, base_z = model.suspension_base_xyz_body_m[index]
        rotated_y = (sin_roll * base_x + cos_roll * base_y)
        base_height = cos_pitch * rotated_y - sin_pitch * base_z
        denominator = cos_pitch * cos_roll
        numerator = (body_height + base_height
                     - model.suspension_ground_y_m)
        travel = numerator / denominator - model.wheel_radius_m

        base_roll_derivative = cos_pitch * (
            cos_roll * base_x - sin_roll * base_y)
        base_pitch_derivative = (
            -sin_pitch * rotated_y - cos_pitch * base_z)
        denominator_roll_derivative = -cos_pitch * ca.sin(roll)
        denominator_pitch_derivative = -sin_pitch * cos_roll
        travel_height_derivative = 1.0 / denominator
        travel_roll_derivative = (
            base_roll_derivative * denominator
            - numerator * denominator_roll_derivative) / denominator**2
        travel_pitch_derivative = (
            base_pitch_derivative * denominator
            - numerator * denominator_pitch_derivative) / denominator**2
        travel_rate = (travel_height_derivative * body_height_rate
                       + travel_roll_derivative * roll_rate
                       + travel_pitch_derivative * pitch_rate)
        travels.append(travel)
        travel_rates.append(travel_rate)
    return tuple(travels), tuple(travel_rates)


def suspension_loads_casadi(model: UnityWheelModel, body_height,
                            body_height_rate, roll, roll_rate, pitch,
                            pitch_rate):
    """PhysX 4.1 suspension force projected onto a flat-ground normal."""
    travels, travel_rates = suspension_travel_casadi(
        model, body_height, body_height_rate, roll, roll_rate, pitch,
        pitch_rate)
    loads = []
    suspension_axis_up_projection = ca.cos(pitch) * ca.cos(roll)
    gravity_along_suspension = ca.fmax(
        0.0, UNITY_PHYSICS_GRAVITY_MPS2 * suspension_axis_up_projection)
    projected_gravity_remainder = ca.if_else(
        suspension_axis_up_projection > 0.0,
        UNITY_PHYSICS_GRAVITY_MPS2
        * (1.0 - suspension_axis_up_projection**2),
        0.0)
    for index, (travel, travel_rate) in enumerate(zip(travels, travel_rates)):
        jounce = model.suspension_target_travel_m[index] - travel
        jounce_rate = -travel_rate
        spring_and_weight_force = ca.fmax(
            0.0,
            model.sprung_mass_kg[index] * gravity_along_suspension
            + model.suspension_spring_n_per_m[index] * jounce)
        suspension_force = (spring_and_weight_force
                            + model.suspension_damper_ns_per_m[index]
                            * jounce_rate)
        # PxVehicleUpdate projects the spring/weight force onto the raycast
        # contact normal; the active plane is horizontal, so this is the
        # vertical component of -w*F - T*beta*m.
        loads.append(
            suspension_axis_up_projection * suspension_force
            + projected_gravity_remainder * model.sprung_mass_kg[index])
    return tuple(loads)


def suspension_loads_numpy(model: UnityWheelModel, body_height: float,
                           body_height_rate: float, roll: float,
                           roll_rate: float, pitch: float,
                           pitch_rate: float) -> np.ndarray:
    """Numeric counterpart used to initialize the research optimizer."""
    sin_roll, cos_roll = math.sin(roll), math.cos(roll)
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    loads = []
    for index in range(4):
        base_x, base_y, base_z = model.suspension_base_xyz_body_m[index]
        rotated_y = sin_roll * base_x + cos_roll * base_y
        base_height = cos_pitch * rotated_y - sin_pitch * base_z
        denominator = cos_pitch * cos_roll
        numerator = body_height + base_height - model.suspension_ground_y_m
        travel = numerator / denominator - model.wheel_radius_m
        base_roll_derivative = cos_pitch * (
            cos_roll * base_x - sin_roll * base_y)
        base_pitch_derivative = (
            -sin_pitch * rotated_y - cos_pitch * base_z)
        denominator_roll_derivative = -cos_pitch * sin_roll
        denominator_pitch_derivative = -sin_pitch * cos_roll
        travel_height_derivative = 1.0 / denominator
        travel_roll_derivative = (
            base_roll_derivative * denominator
            - numerator * denominator_roll_derivative) / denominator**2
        travel_pitch_derivative = (
            base_pitch_derivative * denominator
            - numerator * denominator_pitch_derivative) / denominator**2
        travel_rate = (travel_height_derivative * body_height_rate
                       + travel_roll_derivative * roll_rate
                       + travel_pitch_derivative * pitch_rate)
        jounce = model.suspension_target_travel_m[index] - travel
        jounce_rate = -travel_rate
        alpha = max(0.0, UNITY_PHYSICS_GRAVITY_MPS2 * denominator)
        spring_and_weight = max(
            0.0, model.sprung_mass_kg[index] * alpha
            + model.suspension_spring_n_per_m[index] * jounce)
        suspension_force = (spring_and_weight
                            + model.suspension_damper_ns_per_m[index]
                            * jounce_rate)
        projected_gravity = (UNITY_PHYSICS_GRAVITY_MPS2
                             * max(0.0, 1.0 - denominator**2)
                             if denominator > 0.0 else 0.0)
        loads.append(denominator * suspension_force
                     + projected_gravity * model.sprung_mass_kg[index])
    return np.asarray(loads, dtype=np.float64)


def wheel_torque_casadi(model: UnityWheelModel, throttle):
    """Common-actuator AWD: source MotorTorque split equally across wheels."""
    per_wheel = model.motor_torque_nm * throttle / 4.0
    return (per_wheel,) * 4

