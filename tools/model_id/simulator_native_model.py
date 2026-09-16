"""Offline four-wheel simulator-native candidate.

The compatibility plant retains the historical dimensionless slip guide, but
the explicit Unity-wheel path uses the coordinates reconstructed from the
diagnostic traces: forward slip is based on wheel-surface and ground speed,
sideways slip is ``-wheel_lateral_velocity / max(abs(wheel_forward_velocity),
0.5 m/s)``, and the serialized WheelFrictionCurve is evaluated piecewise
linearly.  This module is not imported by runtime ROS nodes and is not the
production MPC model until true open-loop and blind acceptance passes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from wheel_friction_curve import (
        GUIDE_LATERAL_CURVE,
        GUIDE_LONGITUDINAL_CURVE,
        TireCurveParameters,
        friction_value,
    )
except ModuleNotFoundError:
    from tools.model_id.wheel_friction_curve import (
        GUIDE_LATERAL_CURVE,
        GUIDE_LONGITUDINAL_CURVE,
        TireCurveParameters,
        friction_value,
    )


GRAVITY_MPS2 = 9.81
MIN_SLIP_SPEED_MPS = 0.25
# The exact Unity WheelCollider diagnostics identify different denominator
# guards for the two slip coordinates.  Keep these separate from the older
# guide-model floor above; using one generic floor would silently change the
# Unity contact law.
UNITY_LATERAL_SLIP_SPEED_FLOOR_MPS = 0.5
INTEGRATION_SUBSTEP_S = 0.002
STEERING_RATE_RADPS = 3.2
MAX_STEERING_RAD = 0.5236

# These are not WheelCollider.mass-derived inertias.  They are the three
# effective rotational-response values recovered from the exact Unity
# diagnostic trace.  Keeping them named and separate prevents the wheel
# rotational solver response from being hidden inside a tire or longitudinal
# gain.  They are only used by the explicit Unity-wheel offline plant below.
UNITY_EFFECTIVE_WHEEL_INERTIA_REGIMES_KGM2 = (
    0.000366, 0.000441, 0.000436)
UNITY_EFFECTIVE_WHEEL_INERTIA_BREAKPOINTS_MPS = (6.0, 12.0)
UNITY_WHEEL_DAMPING_NMS = 0.25
UNITY_ACTIVE_MOTOR_TORQUE_NM = 428.0
UNITY_ACTIVE_BRAKE_TORQUE_NM = 428.0
UNITY_COM_HEIGHT_M = 0.06434
UNITY_STEERING_SOURCE_FIXED_DT_S = 0.001


def _prefab_curve(
        extremum_slip: float, extremum_value: float,
        asymptote_slip: float, asymptote_value: float,
        stiffness: float) -> TireCurveParameters:
    """Build a curve whose provenance is the checked-in F1TENTH prefab.

    The prefab is an offline source reference only.  It is not a simulator
    runtime input and this function does not modify the Unity project.
    """
    return TireCurveParameters(
        extremum_slip=extremum_slip,
        extremum_value=extremum_value,
        asymptote_slip=asymptote_slip,
        asymptote_value=asymptote_value,
        stiffness=stiffness,
        source="unity_prefab:F1TENTH.prefab",
    )


@dataclass(frozen=True)
class WheelGeometry:
    """Wheel positions in the API body frame: x forward, y left."""

    x_m: float
    y_m: float
    driven: bool = True


@dataclass(frozen=True)
class DriveStateParameters:
    """Small causal wheel-speed transition retained from the prior discovery."""

    coefficients: tuple[float, ...] = (
        -0.0003516271, 0.0419576436, 23.9652144, 0.0251136087,
        -0.4763574048, 0.0006345492,
    )

    def next_wheel_speed(self, body_u: float, wheel_speed: float,
                         throttle: float) -> float:
        values = np.asarray((
            1.0, wheel_speed, throttle, wheel_speed * throttle,
            throttle * throttle, body_u,
        ), dtype=float)
        return max(0.0, float(values @ np.asarray(self.coefficients)))


@dataclass(frozen=True)
class UnityWheelDynamicsParameters:
    """Explicit offline reconstruction of the active Unity wheel branch.

    ``effective_inertia_regimes_kgm2`` is deliberately not called a physical
    wheel inertia.  Unity's WheelCollider solver does not serialize the
    rotational inertia used by the observed response; these values are a
    separately identified effective state transition from the exact trace.
    Every other value in this record is a source-side controller or
    WheelCollider value.
    """

    motor_torque_nm: float = UNITY_ACTIVE_MOTOR_TORQUE_NM
    brake_torque_nm: float = UNITY_ACTIVE_BRAKE_TORQUE_NM
    throttle_limit: float = 1.0
    wheel_damping_nms: tuple[float, ...] = (UNITY_WHEEL_DAMPING_NMS,) * 4
    effective_inertia_regimes_kgm2: tuple[float, ...] = (
        *UNITY_EFFECTIVE_WHEEL_INERTIA_REGIMES_KGM2,)
    inertia_breakpoints_mps: tuple[float, ...] = (
        *UNITY_EFFECTIVE_WHEEL_INERTIA_BREAKPOINTS_MPS,)
    drive_torque_fractions: tuple[float, ...] = (0.25,) * 4
    brake_torque_fractions: tuple[float, ...] = (1.0,) * 4
    provenance: str = (
        "unity_vehiclecontroller:CAWD_CAWB_motor_torque_428Nm;"
        "unity_wheelcollider:radius_and_damping;"
        "exact_trace:effective_rotational_inertia_regimes")

    def __post_init__(self) -> None:
        if len(self.wheel_damping_nms) != 4:
            raise ValueError("Unity wheel damping must contain four values")
        if len(self.drive_torque_fractions) != 4:
            raise ValueError("Unity drive fractions must contain four values")
        if len(self.brake_torque_fractions) != 4:
            raise ValueError("Unity brake fractions must contain four values")
        if len(self.effective_inertia_regimes_kgm2) != (
                len(self.inertia_breakpoints_mps) + 1):
            raise ValueError(
                "Unity inertia regimes must have one more value than breakpoints")
        if any(value <= 0.0 or not math.isfinite(value)
               for value in self.effective_inertia_regimes_kgm2):
            raise ValueError("Unity effective wheel inertias must be positive")
        if any(value < 0.0 or not math.isfinite(value)
               for value in self.inertia_breakpoints_mps):
            raise ValueError("Unity inertia breakpoints must be non-negative")
        if tuple(sorted(self.inertia_breakpoints_mps)) != self.inertia_breakpoints_mps:
            raise ValueError("Unity inertia breakpoints must be sorted")

    def inertia_for_speed(self, speed_mps: float) -> float:
        """Return the identified effective rotational inertia regime."""
        index = int(np.searchsorted(self.inertia_breakpoints_mps,
                                    max(0.0, abs(float(speed_mps))),
                                    side="right"))
        return float(self.effective_inertia_regimes_kgm2[index])


@dataclass(frozen=True)
class NativeModelParameters:
    """Structural parameters with provenance kept outside the flat fit vector."""

    # Unity Rigidbody.mass is the chassis/body mass used for body translation
    # and static support loads. WheelCollider.mass is kept separately below;
    # the additive sum is not assumed to be Unity's rigid-body mass.
    mass_kg: float = 3.470
    yaw_inertia_kgm2: float | None = None
    com_x_from_rear_axle_m: float = 0.155320086
    # Position point relative to the point at which [u, v] is reported.  The
    # accepted replay data currently uses the source convention directly;
    # an offline Unity dump may establish a non-zero lever arm explicitly.
    position_offset_from_velocity_point_x_m: float = 0.0
    # Contact geometry is measured from WheelCollider positions.  The
    # controller's Ackermann geometry is kept separately below.
    wheelbase_m: float = 0.33000004
    # The source controller can use a steering-geometry wheelbase that is
    # distinct from the measured WheelCollider axle spacing.  Keep both
    # values so the offline plant follows the actual contact points and the
    # actual Ackermann command calculation.
    steering_geometry_wheelbase_m: float | None = 0.324
    track_m: float = 0.236
    wheel_radius_m: float = 0.059
    # VehicleController.WheelRadius is a legacy controller parameter. It is
    # distinct from WheelCollider.radius and is only used by the skid-steer
    # branch in the competition controller.
    controller_wheel_radius_m: float | None = None
    drive_type: str | None = None
    steering_limit_rad: float = MAX_STEERING_RAD
    steering_rate_radps: float = STEERING_RATE_RADPS
    # VehicleController.Steer() is evaluated in Unity's FixedUpdate loop.
    # Keep its measured source cadence explicit instead of approximating the
    # controller with one 25 ms MoveTowards step.
    steering_source_fixed_dt_s: float = UNITY_STEERING_SOURCE_FIXED_DT_S
    # Keep the effective API-frame to model-frame conversion explicit. The
    # raw Unity WheelCollider sign is recorded separately by the diagnostic
    # experiment; it is not interchangeable with the API-frame replay sign.
    steering_to_wheel_angle_sign: float = 1.0
    longitudinal_curve: TireCurveParameters = GUIDE_LONGITUDINAL_CURVE
    lateral_curve: TireCurveParameters = GUIDE_LATERAL_CURVE
    # A dump may contain per-wheel overrides.  The scalar curves above remain
    # the compatibility/default path used by the fitted candidates.
    wheel_longitudinal_curves: tuple[TireCurveParameters, ...] | None = None
    wheel_lateral_curves: tuple[TireCurveParameters, ...] | None = None
    wheel_radii_m: tuple[float, ...] | None = None
    wheel_masses_kg: tuple[float, ...] | None = None
    wheel_sprung_masses_kg: tuple[float, ...] | None = None
    # The Unity Rigidbody COM height is kept as a structural load-transfer
    # input.  It is not a fitted tire/load coefficient.  The default is None
    # so a model loaded from a dump must explicitly opt in to the mechanical
    # load path rather than silently inventing a suspension state.
    unity_com_height_m: float | None = None
    # ``static_sprung_mass`` is the canonical current plant.  The optional
    # ``mechanical_longitudinal_cg_transfer`` and
    # ``mechanical_cg_transfer`` paths apply exact rigid-body CG moment
    # balance to the predicted tire wrench.  They are offline screens only;
    # they are not substitutes for the still-missing WheelCollider
    # suspension/roll state and do not alter the production MPC model.
    unity_normal_load_mode: str = "static_sprung_mass"
    # Unity's body-frame pitch/roll inertias are separate from the required
    # body-y yaw inertia.  They are required only by the offline suspension
    # state path; leaving them unset prevents a guessed vertical model.
    pitch_inertia_kgm2: float | None = None
    roll_inertia_kgm2: float | None = None
    suspension_distances_m: tuple[float, ...] | None = None
    suspension_spring_rates_n_per_m: tuple[float, ...] | None = None
    suspension_damper_rates_ns_per_m: tuple[float, ...] | None = None
    suspension_target_positions: tuple[float, ...] | None = None
    wheel_damping_rates_per_s: tuple[float, ...] | None = None
    force_app_point_distances_m: tuple[float, ...] | None = None
    rigid_body_drag_per_s: float | None = None
    rigid_body_angular_drag_per_s: float | None = None
    rigid_body_max_angular_velocity_radps: float | None = None
    # Optional explicit wheel-rotational branch.  It is kept separate from
    # the legacy 8/9-state candidates so adding this diagnostic model cannot
    # silently alter their fitted behavior.
    unity_wheel_dynamics: UnityWheelDynamicsParameters | None = None
    drive_state: DriveStateParameters = DriveStateParameters()
    longitudinal_gain: float = 1.0
    lateral_gain: float = 1.0
    drag_linear_n_per_mps: float = 0.0
    drag_quadratic_n_per_mps2: float = 0.0
    # Inertia-free effective gains. These are used only when no independent
    # Unity yaw-inertia dump exists; they are not relabelled as physical tire
    # stiffness or inertia.
    effective_longitudinal_gain_mps2: float | None = None
    effective_drag_linear_per_s: float | None = None
    effective_drag_quadratic_per_m: float | None = None
    effective_lateral_front_mps2: float | None = None
    effective_lateral_rear_mps2: float | None = None
    effective_yaw_front_per_s2: float | None = None
    effective_yaw_rear_per_s2: float | None = None
    # VS2.5 uses the complete four-contact normalized force/moment basis. The
    # coefficients remain effective simulator gains until a diagnostic dump
    # establishes physical loads and inertia.
    effective_force_x_gain_mps2: float | None = None
    effective_force_y_gain_mps2: float | None = None
    effective_yaw_moment_gain_per_s2: float | None = None
    effective_force_y_front_gain_mps2: float | None = None
    effective_force_y_rear_gain_mps2: float | None = None
    effective_yaw_lateral_front_gain_per_s2: float | None = None
    effective_yaw_lateral_rear_gain_per_s2: float | None = None
    effective_yaw_longitudinal_gain_per_s2: float | None = None
    # Optional effective residual damping. This is not the Rigidbody
    # angularDrag field and is only enabled when identified from replay data.
    effective_yaw_damping_per_s: float = 0.0
    # Optional lateral-load-transfer height for offline structural screening.
    # It is deliberately zero by default and is not inferred as the physical
    # center-of-mass height from a simulator dump.
    effective_load_transfer_height_m: float = 0.0
    contact_model: str = "four_wheel"
    parameter_provenance: str = "official_2026_guide_until_diagnostic_dump"


    @property
    def lf_m(self) -> float:
        return self.wheelbase_m - self.com_x_from_rear_axle_m

    @property
    def lr_m(self) -> float:
        return self.com_x_from_rear_axle_m

    @property
    def wheels(self) -> tuple[WheelGeometry, ...]:
        half_track = 0.5 * self.track_m
        return (
            WheelGeometry(self.lf_m, half_track),
            WheelGeometry(self.lf_m, -half_track),
            WheelGeometry(-self.lr_m, half_track),
            WheelGeometry(-self.lr_m, -half_track),
        )

    @property
    def has_effective_gains(self) -> bool:
        return all(value is not None for value in (
            self.effective_longitudinal_gain_mps2,
            self.effective_drag_linear_per_s,
            self.effective_drag_quadratic_per_m,
            self.effective_lateral_front_mps2,
            self.effective_lateral_rear_mps2,
            self.effective_yaw_front_per_s2,
            self.effective_yaw_rear_per_s2,
        ))

    @property
    def has_moment_basis_gains(self) -> bool:
        return all(value is not None for value in (
            self.effective_force_x_gain_mps2,
            self.effective_drag_linear_per_s,
            self.effective_drag_quadratic_per_m,
            self.effective_force_y_gain_mps2,
            self.effective_yaw_moment_gain_per_s2,
        ))

    @property
    def has_split_moment_basis_gains(self) -> bool:
        return all(value is not None for value in (
            self.effective_force_x_gain_mps2,
            self.effective_drag_linear_per_s,
            self.effective_drag_quadratic_per_m,
            self.effective_force_y_front_gain_mps2,
            self.effective_force_y_rear_gain_mps2,
            self.effective_yaw_lateral_front_gain_per_s2,
            self.effective_yaw_lateral_rear_gain_per_s2,
            self.effective_yaw_longitudinal_gain_per_s2,
        ))

    def longitudinal_curve_for_wheel(self, index: int) -> TireCurveParameters:
        if self.wheel_longitudinal_curves is None:
            return self.longitudinal_curve
        return self.wheel_longitudinal_curves[index]

    def lateral_curve_for_wheel(self, index: int) -> TireCurveParameters:
        if self.wheel_lateral_curves is None:
            return self.lateral_curve
        return self.wheel_lateral_curves[index]

    @property
    def wheel_mass_total_kg(self) -> float:
        """Return the recorded WheelCollider masses, when available."""
        return (sum(self.wheel_masses_kg)
                if self.wheel_masses_kg is not None else 0.0)

    @property
    def additive_system_mass_kg(self) -> float:
        """Return an accounting sum, not a claim about Unity dynamics."""
        return self.mass_kg + self.wheel_mass_total_kg


def f1tenth_prefab_parameters(
        contact_model: str = "four_wheel") -> NativeModelParameters:
    """Return static parameters read from the F1TENTH competition prefab.

    This records the source-side configuration currently used for offline
    replay.  Rigidbody inertia is intentionally not fabricated: the prefab
    does not serialize Unity's automatically generated inertia tensor.
    """
    longitudinal = _prefab_curve(0.15, 0.9, 0.25, 0.58, 0.8)
    lateral = _prefab_curve(0.01, 1.0, 0.1, 0.5, 1.0)
    return NativeModelParameters(
        mass_kg=3.470,
        com_x_from_rear_axle_m=0.15532,
        position_offset_from_velocity_point_x_m=-0.15532,
        wheelbase_m=0.33,
        steering_geometry_wheelbase_m=0.324,
        track_m=0.236,
        wheel_radius_m=0.059,
        controller_wheel_radius_m=0.0325,
        drive_type="CAWD",
        steering_limit_rad=math.radians(30.0),
        steering_rate_radps=math.radians(183.346),
        steering_to_wheel_angle_sign=1.0,
        longitudinal_curve=longitudinal,
        lateral_curve=lateral,
        wheel_masses_kg=(0.109,) * 4,
        unity_com_height_m=UNITY_COM_HEIGHT_M,
        unity_normal_load_mode="static_sprung_mass",
        parameter_provenance=(
            "unity_prefab:F1TENTH.prefab;"
            "inertia=not_serialized;"
            "position_frame=GPS_Rear_Axle_Center_vs_IMU_Rigidbody_COM"),
        contact_model=contact_model,
    )


def body_axis_inertia(
        principal_moments: dict[str, float],
        principal_rotation: dict[str, float],
        axis: str | tuple[float, float, float]) -> float:
    """Project Unity's principal inertia tensor onto an explicit body axis.

    ``inertiaTensorRotation`` rotates the principal-inertia frame into the
    Rigidbody body frame.  Unity's vehicle forward axis is body ``z`` and its
    vertical/yaw axis is body ``y``; therefore callers must select the axis
    explicitly instead of relying on a function named "yaw" with a hidden
    projection convention.
    """
    x, y, z, w = (float(principal_rotation[key])
                  for key in ("x", "y", "z", "w"))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1.0e-12:
        raise ValueError("inertia tensor rotation has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if isinstance(axis, str):
        if axis not in ("x", "y", "z"):
            raise ValueError("body inertia axis must be x, y, or z")
        selected_axis = {
            "x": np.asarray((1.0, 0.0, 0.0), dtype=float),
            "y": np.asarray((0.0, 1.0, 0.0), dtype=float),
            "z": np.asarray((0.0, 0.0, 1.0), dtype=float),
        }[axis]
    else:
        selected_axis = np.asarray(axis, dtype=float)
        if selected_axis.shape != (3,):
            raise ValueError("body inertia axis vector must have three entries")
        axis_norm = float(np.linalg.norm(selected_axis))
        if axis_norm <= 1.0e-12:
            raise ValueError("body inertia axis vector has zero norm")
        selected_axis = selected_axis / axis_norm

    # Columns are the principal axes expressed in the Rigidbody frame.
    rotation_matrix = np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])
    tensor = rotation_matrix @ np.diag([
        float(principal_moments["x"]),
        float(principal_moments["y"]),
        float(principal_moments["z"]),
    ]) @ rotation_matrix.T
    return float(selected_axis @ tensor @ selected_axis)


def body_frame_inertias(
        principal_moments: dict[str, float],
        principal_rotation: dict[str, float]) -> dict[str, float]:
    """Return all principal-tensor projections in the Rigidbody frame."""
    return {
        axis: body_axis_inertia(principal_moments, principal_rotation, axis)
        for axis in ("x", "y", "z")
    }


def ackermann_angles(command_angle: float, wheelbase_m: float,
                     track_m: float) -> tuple[float, float, float, float]:
    """Return API-frame FL, FR, RL, RR steering angles for front steering."""
    if abs(command_angle) < 1.0e-12:
        return 0.0, 0.0, 0.0, 0.0
    tangent = math.tan(command_angle)
    # This is the established API-frame effective convention used by the
    # accepted replay data. Raw Unity WheelCollider angles use a different
    # component/axis convention and are exposed below for diagnostics.
    left = math.atan2(2.0 * wheelbase_m * tangent,
                      2.0 * wheelbase_m - track_m * tangent)
    right = math.atan2(2.0 * wheelbase_m * tangent,
                       2.0 * wheelbase_m + track_m * tangent)
    return left, right, 0.0, 0.0


def unity_wheel_collider_ackermann_angles(
        wheel_angle: float, wheelbase_m: float,
        track_m: float) -> tuple[float, float, float, float]:
    """Return raw FL/FR angles using VehicleController.cs ordering.

    This is a diagnostic reference only. A caller must establish the axis and
    sign transform before using it in an API-frame replay model.
    """
    if abs(wheel_angle) < 1.0e-12:
        return 0.0, 0.0, 0.0, 0.0
    tangent = math.tan(wheel_angle)
    left = math.atan((2.0 * wheelbase_m * tangent) /
                     (2.0 * wheelbase_m + track_m * tangent))
    right = math.atan((2.0 * wheelbase_m * tangent) /
                      (2.0 * wheelbase_m - track_m * tangent))
    return left, right, 0.0, 0.0


def contact_kinematics(state: np.ndarray, steering_angle: float,
                       parameters: NativeModelParameters) -> list[dict[str, float]]:
    """Return per-wheel body/tire velocities and documented slip coordinates."""
    if state.shape not in ((8,), (9,)):
        raise ValueError(f"expected state shape (8,) or (9,), got {state.shape}")
    # The model state stores wheel *surface speed* in m/s.  The recorder has
    # already converted encoder angular rate using the configured radius, so
    # multiplying by wheel_radius here would apply that conversion twice.
    _, _, _, u, v, r, _ = (float(value) for value in state[:7])
    if state.shape == (8,):
        wheel_surface_speeds = (float(state[7]),) * 4
    else:
        # The available recorder exposes left/right encoder states.  Use each
        # side for its front and rear contact instead of pretending the
        # inside and outside wheels have the same surface speed.
        wheel_surface_speeds = (
            float(state[7]), float(state[8]),
            float(state[7]), float(state[8]))
    angles = ackermann_angles(
        steering_angle * parameters.steering_to_wheel_angle_sign,
        (parameters.steering_geometry_wheelbase_m
         if parameters.steering_geometry_wheelbase_m is not None
         else parameters.wheelbase_m),
        parameters.track_m)
    output: list[dict[str, float]] = []
    for index, (geometry, angle) in enumerate(zip(parameters.wheels, angles)):
        vx_body = u - r * geometry.y_m
        vy_body = v + r * geometry.x_m
        cosine = math.cos(angle)
        sine = math.sin(angle)
        vx_tire = cosine * vx_body + sine * vy_body
        vy_tire = -sine * vx_body + cosine * vy_body
        denominator = max(abs(vx_tire), MIN_SLIP_SPEED_MPS)
        sx = (wheel_surface_speeds[index] - vx_tire) / denominator
        sy = vy_tire / denominator
        output.append({
            "x_m": geometry.x_m,
            "y_m": geometry.y_m,
            "steering_rad": angle,
            "vx_body_mps": vx_body,
            "vy_body_mps": vy_body,
            "vx_tire_mps": vx_tire,
            "vy_tire_mps": vy_tire,
            "wheel_surface_speed_mps": wheel_surface_speeds[index],
            "sx": sx,
            "sy": sy,
        })
    return output


def position_point_body_velocity(
        state: np.ndarray, parameters: NativeModelParameters) -> tuple[float, float]:
    """Return body velocity at the recorded pose point.

    A positive x offset is forward of the point at which body velocity is
    reported.  The lever-arm relation is ``v_point = v + r*offset_x``.
    For a rear-axle point relative to a COM velocity, ``offset_x`` is
    ``-com_x``.
    This is an offline reference-point correction; it does not change the
    runtime odometry frame.
    """
    if state.shape not in ((8,), (9,), (11,)):
        raise ValueError(
            f"expected state shape (8,), (9,), or (11,), got {state.shape}")
    u = float(state[3])
    v = float(state[4])
    r = float(state[5])
    return u, v + parameters.position_offset_from_velocity_point_x_m * r


def static_normal_loads(parameters: NativeModelParameters) -> tuple[float, ...]:
    front_axle = parameters.mass_kg * GRAVITY_MPS2 * parameters.lr_m / parameters.wheelbase_m
    rear_axle = parameters.mass_kg * GRAVITY_MPS2 * parameters.lf_m / parameters.wheelbase_m
    return front_axle / 2.0, front_axle / 2.0, rear_axle / 2.0, rear_axle / 2.0


def unity_normal_loads(parameters: NativeModelParameters) -> tuple[float, ...]:
    """Return the explicit support-load source used by the Unity plant.

    A diagnostic dump's ``WheelHit.sprungMass`` is a per-contact runtime
    quantity.  It is used directly when present; the geometry-derived static
    load is only the fallback for a prefab/reference profile.  Dynamic
    suspension load transfer is intentionally not synthesized here.
    """
    if parameters.wheel_sprung_masses_kg is not None:
        if len(parameters.wheel_sprung_masses_kg) != 4:
            raise ValueError("Unity sprung-mass data must contain four values")
        return tuple(
            float(mass) * GRAVITY_MPS2
            for mass in parameters.wheel_sprung_masses_kg)
    return static_normal_loads(parameters)


def unity_normal_loads_from_acceleration(
        parameters: NativeModelParameters, longitudinal_accel_mps2: float,
        lateral_accel_mps2: float) -> tuple[float, ...]:
    """Return structural per-wheel loads for an explicit CG-transfer screen.

    The transfer is derived from the measured Unity body mass, COM height,
    contact wheelbase, and track width.  Positive body acceleration transfers
    support rearward/rightward in the ``x-forward, y-left`` model frame.  The
    base sprung masses are retained as the static distribution; the transfer
    conserves their total load.  This is intentionally an algebraic rigid
    body screen, not a claim that it reproduces the WheelCollider's spring,
    damper, and contact solver transitions.
    """
    if parameters.unity_normal_load_mode == "static_sprung_mass":
        return unity_normal_loads(parameters)
    if parameters.unity_normal_load_mode not in {
            "mechanical_longitudinal_cg_transfer", "mechanical_cg_transfer"}:
        raise ValueError(
            "unsupported Unity normal-load mode: "
            f"{parameters.unity_normal_load_mode}")
    height = parameters.unity_com_height_m
    if height is None or not math.isfinite(height) or height <= 0.0:
        raise ValueError(
            "mechanical Unity load transfer requires dumped COM height")
    values = (longitudinal_accel_mps2, lateral_accel_mps2)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Unity load-transfer acceleration must be finite")
    if parameters.wheelbase_m <= 0.0 or parameters.track_m <= 0.0:
        raise ValueError("Unity contact wheelbase and track must be positive")
    if parameters.wheel_sprung_masses_kg is not None:
        if len(parameters.wheel_sprung_masses_kg) != 4:
            raise ValueError("Unity sprung-mass data must contain four values")
        base = np.asarray(tuple(
            float(mass) * GRAVITY_MPS2
            for mass in parameters.wheel_sprung_masses_kg), dtype=float)
    else:
        base = np.asarray(static_normal_loads(parameters), dtype=float)
    # The front/rear axle shift is m*h/L*ax.  It is added to each front wheel
    # as half of the axle shift and subtracted from each rear wheel.  The
    # left/right shift is distributed in the same way across the two axles.
    front_shift_per_wheel = (
        -parameters.mass_kg * height / parameters.wheelbase_m *
        longitudinal_accel_mps2 / 2.0)
    left_shift_per_wheel = 0.0
    if parameters.unity_normal_load_mode == "mechanical_cg_transfer":
        left_shift_per_wheel = (
            -parameters.mass_kg * height / parameters.track_m *
            lateral_accel_mps2 / 2.0)
    loads = base + np.asarray((
        front_shift_per_wheel + left_shift_per_wheel,
        front_shift_per_wheel - left_shift_per_wheel,
        -front_shift_per_wheel + left_shift_per_wheel,
        -front_shift_per_wheel - left_shift_per_wheel,
    ), dtype=float)
    if np.any(loads < 0.0):
        # The planar operating envelope should normally remain four-wheel
        # grounded.  Keep the unilateral contact boundary explicit instead
        # of allowing a negative normal load to become a hidden tire gain.
        loads = np.maximum(loads, 0.0)
    return tuple(float(value) for value in loads)


def unity_wheel_travel_loads(
        parameters: NativeModelParameters,
        wheel_travel: tuple[float, ...] | list[float] | np.ndarray,
        wheel_travel_rates: tuple[float, ...] | list[float] | np.ndarray
        ) -> tuple[float, ...]:
    """Evaluate the measured native WheelCollider travel/load law.

    Unity's serialized spring and damper are expressed in N/m and N s/m,
    while ``WheelCollider`` travel is normalized by ``suspensionDistance``.
    The exact contact-point audit therefore uses

        ``Fz = max(0, sprungMass*g - spring*distance*travel
                         - damper*distance*travel_rate)``.

    ``targetPosition`` remains a source configuration value, but it is not a
    free force coefficient and is not inserted into this measured local
    travel law.  This function is deliberately separate so the source-side
    mechanism cannot be replaced by an optimizer gain.
    """
    if parameters.wheel_sprung_masses_kg is None:
        raise ValueError("Unity travel loads require captured sprung masses")
    if (parameters.suspension_distances_m is None or
            parameters.suspension_spring_rates_n_per_m is None or
            parameters.suspension_damper_rates_ns_per_m is None):
        raise ValueError(
            "Unity travel loads require captured distance/spring/damper values")
    arrays = (
        parameters.wheel_sprung_masses_kg,
        parameters.suspension_distances_m,
        parameters.suspension_spring_rates_n_per_m,
        parameters.suspension_damper_rates_ns_per_m,
        tuple(float(value) for value in wheel_travel),
        tuple(float(value) for value in wheel_travel_rates),
    )
    if any(len(values) != 4 for values in arrays):
        raise ValueError("Unity travel-load arrays must contain four values")
    if not all(math.isfinite(value) for values in arrays for value in values):
        raise ValueError("Unity travel-load values must be finite")
    loads: list[float] = []
    for mass, distance, spring, damper, travel, travel_rate in zip(*arrays):
        if mass < 0.0 or distance <= 0.0 or spring < 0.0 or damper < 0.0:
            raise ValueError("Unity travel-load source values are invalid")
        load = (mass * GRAVITY_MPS2 - spring * distance * travel -
                damper * distance * travel_rate)
        loads.append(max(0.0, float(load)))
    return tuple(loads)


def unity_suspension_loads(
        parameters: NativeModelParameters, heave_m: float,
        pitch_rad: float, roll_rad: float, heave_rate_mps: float,
        pitch_rate_radps: float, roll_rate_radps: float
        ) -> tuple[float, ...]:
    """Evaluate the explicit four-WheelCollider spring/damper load state.

    The suspension coordinates are deviations from the settled, four-wheel
    grounded configuration.  For wheel ``i`` the body-side vertical
    displacement is ``heave - pitch*x_i + roll*y_i`` in the model frame.  A
    positive upward displacement unloads that wheel, so the serialized Unity
    spring and damper produce ``Fz = Fz_static - k*q - c*q_dot``.  The static
    ``Fz`` is taken from the captured runtime ``sprungMass`` rather than a
    fitted load or tire parameter.

    This is the vertical suspension submodel used by the offline plant.  It
    does not use simulator truth during rollout and it treats negative load as
    an explicit unilateral-contact boundary rather than allowing a negative
    tire force.
    """
    values = (heave_m, pitch_rad, roll_rad, heave_rate_mps,
              pitch_rate_radps, roll_rate_radps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Unity suspension state must be finite")
    if parameters.wheel_sprung_masses_kg is None:
        raise ValueError("Unity suspension requires captured sprung masses")
    if (parameters.suspension_spring_rates_n_per_m is None or
            parameters.suspension_damper_rates_ns_per_m is None):
        raise ValueError("Unity suspension requires captured spring/damper rates")
    if (len(parameters.wheel_sprung_masses_kg) != 4 or
            len(parameters.suspension_spring_rates_n_per_m) != 4 or
            len(parameters.suspension_damper_rates_ns_per_m) != 4):
        raise ValueError("Unity suspension arrays must contain four values")
    if parameters.suspension_distances_m is None:
        raise ValueError("Unity suspension requires captured distances")
    travel: list[float] = []
    travel_rate: list[float] = []
    for index, wheel in enumerate(parameters.wheels):
        displacement = (heave_m - pitch_rad * wheel.x_m +
                        roll_rad * wheel.y_m)
        velocity = (heave_rate_mps - pitch_rate_radps * wheel.x_m +
                    roll_rate_radps * wheel.y_m)
        distance = float(parameters.suspension_distances_m[index])
        travel.append(displacement / distance)
        travel_rate.append(velocity / distance)
    return unity_wheel_travel_loads(parameters, travel, travel_rate)


def unity_contact_kinematics(
        state: np.ndarray, steering_angle: float,
        parameters: NativeModelParameters) -> list[dict[str, float]]:
    """Return four WheelCollider contact velocities from wheel angular state.

    The state is ``[X, Y, yaw, u, v, r, steering, omega_FL, omega_FR,
    omega_RL, omega_RR]``.  Unlike the compatibility path, the wheel state is
    angular rate and the physical WheelCollider radius is applied exactly
    once when forming the contact-patch surface speed.
    """
    if state.shape != (11,):
        raise ValueError(
            f"expected Unity wheel state shape (11,), got {state.shape}")
    radii = (parameters.wheel_radii_m
             if parameters.wheel_radii_m is not None
             else (parameters.wheel_radius_m,) * 4)
    if len(radii) != 4:
        raise ValueError("Unity wheel radii must contain four values")
    _, _, _, u, v, r, _ = (float(value) for value in state[:7])
    angles = ackermann_angles(
        steering_angle * parameters.steering_to_wheel_angle_sign,
        (parameters.steering_geometry_wheelbase_m
         if parameters.steering_geometry_wheelbase_m is not None
         else parameters.wheelbase_m),
        parameters.track_m)
    output: list[dict[str, float]] = []
    for index, (geometry, angle) in enumerate(zip(parameters.wheels, angles)):
        vx_body = u - r * geometry.y_m
        vy_body = v + r * geometry.x_m
        cosine = math.cos(angle)
        sine = math.sin(angle)
        vx_tire = cosine * vx_body + sine * vy_body
        vy_tire = -sine * vx_body + cosine * vy_body
        omega = float(state[7 + index])
        surface_speed = radii[index] * omega
        # These are the simulator coordinates, not real-vehicle slip
        # definitions.  Forward slip uses the larger of wheel-surface and
        # ground speed and saturates at +/-1.  The exact trace reconstructs
        # WheelHit.sidewaysSlip as the negative wheel-frame lateral velocity
        # over the forward-speed floor.
        forward_denominator = max(
            abs(surface_speed), abs(vx_tire), MIN_SLIP_SPEED_MPS)
        lateral_denominator = max(
            abs(vx_tire), UNITY_LATERAL_SLIP_SPEED_FLOOR_MPS)
        forward_slip = max(-1.0, min(1.0, (
            (surface_speed - vx_tire) / forward_denominator)))
        sideways_slip = -vy_tire / lateral_denominator
        output.append({
            "x_m": geometry.x_m,
            "y_m": geometry.y_m,
            "steering_rad": angle,
            "vx_body_mps": vx_body,
            "vy_body_mps": vy_body,
            "vx_tire_mps": vx_tire,
            "vy_tire_mps": vy_tire,
            "omega_radps": omega,
            "wheel_surface_speed_mps": surface_speed,
            "sx": forward_slip,
            "sy": sideways_slip,
        })
    return output


def _unity_wheel_torques(
        throttle_norm: float, parameters: NativeModelParameters
        ) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Reproduce the active VehicleController CAWD/CAWB torque branch."""
    dynamics = parameters.unity_wheel_dynamics
    if dynamics is None:
        raise ValueError("Unity wheel dynamics are required for this plant")
    throttle = max(-1.0, min(1.0, float(throttle_norm)))
    drive_torque = dynamics.throttle_limit * throttle * dynamics.motor_torque_nm
    if abs(drive_torque) <= 1.0e-12:
        motor = (0.0,) * 4
        brake = tuple(
            dynamics.brake_torque_nm * fraction
            for fraction in dynamics.brake_torque_fractions)
    else:
        motor = tuple(
            drive_torque * fraction
            for fraction in dynamics.drive_torque_fractions)
        brake = (0.0,) * 4
    return motor, brake


def _unity_contact_force_components(
        state: np.ndarray, steering_angle: float,
        parameters: NativeModelParameters,
        normal_loads: tuple[float, ...]
        ) -> tuple[tuple[float, float, float], ...]:
    """Return per-wheel ``(Fx_body, Fy_body, Fx_tire)`` contact forces."""
    if len(normal_loads) != 4 or not all(
            math.isfinite(float(load)) and float(load) >= 0.0
            for load in normal_loads):
        raise ValueError("Unity normal loads must contain four finite values")
    contacts = unity_contact_kinematics(state, steering_angle, parameters)
    components: list[tuple[float, float, float]] = []
    for index, (contact, normal_load) in enumerate(
            zip(contacts, normal_loads)):
        # Unity's serialized WheelFrictionCurve is piecewise linear between
        # the origin, extremum, and asymptote, then constant.  The generic
        # cubic Hermite guide surrogate is intentionally not used here.
        fx_tire = normal_load * _unity_wheel_friction_value(
            contact["sx"], parameters.longitudinal_curve_for_wheel(index))
        # ``sy`` is the exact Unity sideways-slip sign, so the contact force
        # is positive in the corresponding body-left direction.
        fy_tire = normal_load * _unity_wheel_friction_value(
            contact["sy"], parameters.lateral_curve_for_wheel(index))
        cosine = math.cos(contact["steering_rad"])
        sine = math.sin(contact["steering_rad"])
        components.append((
            cosine * fx_tire - sine * fy_tire,
            sine * fx_tire + cosine * fy_tire,
            fx_tire,
        ))
    return tuple(components)


def _unity_wheel_friction_value(
        slip: float, curve: TireCurveParameters) -> float:
    """Evaluate a dumped Unity WheelFrictionCurve without a fitted shape.

    Unity exposes the three curve points and stiffness, not a real-tire law.
    The native WheelCollider model uses the simulator's documented linear
    segments: origin to extremum, extremum to asymptote, then the asymptote
    plateau.  This function is deliberately separate from ``friction_value``
    because the latter is the historical cubic guide surrogate used by the
    older effective candidates.
    """
    if not math.isfinite(slip):
        raise ValueError("Unity slip must be finite")
    magnitude = abs(slip)
    if magnitude <= curve.extremum_slip:
        value = curve.extremum_value * magnitude / curve.extremum_slip
    elif magnitude <= curve.asymptote_slip:
        fraction = ((magnitude - curve.extremum_slip) /
                    (curve.asymptote_slip - curve.extremum_slip))
        value = curve.extremum_value + fraction * (
            curve.asymptote_value - curve.extremum_value)
    else:
        value = curve.asymptote_value
    signed = math.copysign(curve.stiffness * value, slip) if slip else 0.0
    return float(signed)


def _unity_contact_wrench(
        state: np.ndarray, steering_angle: float,
        throttle_norm: float, parameters: NativeModelParameters,
        normal_loads: tuple[float, ...] | None = None
        ) -> tuple[float, float, float, tuple[float, ...]]:
    """Return body wrench and each wheel's tire-frame longitudinal force."""
    dynamics = parameters.unity_wheel_dynamics
    if dynamics is None:
        raise ValueError("Unity wheel dynamics are required for this plant")
    loads = (unity_normal_loads(parameters)
             if normal_loads is None else normal_loads)
    if len(loads) != 4 or not all(
            math.isfinite(float(load)) and float(load) >= 0.0
            for load in loads):
        raise ValueError("Unity normal loads must contain four finite values")
    motor_torques, brake_torques = _unity_wheel_torques(
        throttle_norm, parameters)
    components = _unity_contact_force_components(
        state, steering_angle, parameters, loads)
    force_x = sum(component[0] for component in components)
    force_y = sum(component[1] for component in components)
    moment_z = sum(
        wheel.x_m * component[1] - wheel.y_m * component[0]
        for wheel, component in zip(parameters.wheels, components))
    tire_longitudinal_forces = tuple(component[2] for component in components)

    # Rigidbody drag is a body-force coefficient, not a fitted tire term.
    if parameters.rigid_body_drag_per_s is not None:
        force_x -= parameters.rigid_body_drag_per_s * float(state[3])
        force_y -= parameters.rigid_body_drag_per_s * float(state[4])
    if (parameters.rigid_body_angular_drag_per_s is not None and
            parameters.yaw_inertia_kgm2 is not None):
        moment_z -= (parameters.rigid_body_angular_drag_per_s *
                     parameters.yaw_inertia_kgm2 * float(state[5]))
    return force_x, force_y, moment_z, tire_longitudinal_forces


def step_unity_wheel_collider(
        state: np.ndarray, steering_target_norm: float,
        throttle_norm: float, dt: float,
        parameters: NativeModelParameters) -> np.ndarray:
    """Advance the explicit Unity WheelCollider offline reconstruction.

    This path is intentionally separate from :func:`step`.  It is a
    diagnostic/replay model only: it uses the checked-in Unity structure and
    exact diagnostic snapshot, but it does not receive simulator truth at
    runtime and is not imported by ROS or production MPC code.
    """
    if state.shape != (11,):
        raise ValueError(
            f"expected Unity wheel state shape (11,), got {state.shape}")
    if parameters.unity_wheel_dynamics is None:
        raise ValueError("step_unity_wheel_collider requires Unity wheel dynamics")
    if parameters.yaw_inertia_kgm2 is None:
        raise ValueError(
            "step_unity_wheel_collider requires dumped body yaw inertia")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    target = (max(-1.0, min(1.0, float(steering_target_norm))) *
              parameters.steering_limit_rad)
    x, y, yaw, u, v, r, steering = (
        float(value) for value in state[:7])
    wheel_omegas = [float(value) for value in state[7:11]]
    steering_segments = _unity_vehicle_controller_steering_segments(
        steering, steering_target_norm, dt, parameters)
    for segment_dt, segment_start, segment_end in steering_segments:
        count = max(1, int(math.ceil(segment_dt / INTEGRATION_SUBSTEP_S)))
        sub_dt = segment_dt / count
        for sub_index in range(count):
            fraction = (sub_index + 0.5) / count
            delta = segment_start + fraction * (segment_end - segment_start)
            local_state = np.asarray(
                [x, y, yaw, u, v, r, delta, *wheel_omegas], dtype=float)
            normal_loads = unity_normal_loads(parameters)
        # The dynamic path is deliberately a short fixed-point solve.  Tire
        # force is linear in the supplied normal loads, while the CG transfer
        # is derived from the resulting contact acceleration.  Two iterations
        # are enough to remove the load/force circularity at this model's
        # substep scale without introducing a fitted relaxation constant.
            iterations = (3 if parameters.unity_normal_load_mode in {
                "mechanical_longitudinal_cg_transfer", "mechanical_cg_transfer"
            } else 1)
            force_x = force_y = moment_z = 0.0
            tire_forces: tuple[float, ...] = (0.0,) * 4
            for _ in range(iterations):
                force_x, force_y, moment_z, tire_forces = _unity_contact_wrench(
                    local_state, delta, throttle_norm, parameters,
                    normal_loads=normal_loads)
                if parameters.unity_normal_load_mode not in {
                        "mechanical_longitudinal_cg_transfer",
                        "mechanical_cg_transfer"}:
                    break
                # Rigidbody drag acts at the body COM and therefore does not
                # create the wheel-support transfer being identified here.
                drag_x = ((parameters.rigid_body_drag_per_s or 0.0) * float(u))
                drag_y = ((parameters.rigid_body_drag_per_s or 0.0) * float(v))
                tire_ax = (force_x + drag_x) / parameters.mass_kg + r * v
                tire_ay = (force_y + drag_y) / parameters.mass_kg - r * u
                normal_loads = unity_normal_loads_from_acceleration(
                    parameters, tire_ax, tire_ay)
            motor_torques, brake_torques = _unity_wheel_torques(
                throttle_norm, parameters)
            speed = abs(u)
            dynamics = parameters.unity_wheel_dynamics
            wheel_damping = dynamics.wheel_damping_nms
            omega_dot = []
            for index, omega in enumerate(wheel_omegas):
                radius = (parameters.wheel_radii_m[index]
                          if parameters.wheel_radii_m is not None
                          else parameters.wheel_radius_m)
                unbraked_torque = (
                    motor_torques[index] - tire_forces[index] * radius -
                    wheel_damping[index] * omega)
                brake = brake_torques[index]
                if brake > 0.0:
                    # A Coulomb brake must not be integrated through zero with
                    # an explicit Euler sign flip.  When the brake can hold the
                    # wheel, use the static-brake solution; otherwise oppose
                    # the current/free-motion direction and stop exactly at
                    # zero if the next substep would cross it.
                    if abs(omega) <= 1.0e-9 and abs(unbraked_torque) <= brake:
                        wheel_torque = 0.0
                    else:
                        direction = (math.copysign(1.0, omega)
                                     if abs(omega) > 1.0e-9 else
                                     math.copysign(1.0, unbraked_torque))
                        wheel_torque = unbraked_torque - direction * brake
                        inertia = dynamics.inertia_for_speed(speed)
                        if (abs(omega) > 1.0e-9 and
                                omega * (omega + sub_dt * wheel_torque / inertia)
                                < 0.0):
                            wheel_torque = -omega * inertia / sub_dt
                else:
                    wheel_torque = unbraked_torque
                omega_dot.append(
                    wheel_torque / dynamics.inertia_for_speed(speed))
            u_dot = force_x / parameters.mass_kg + r * v
            v_dot = force_y / parameters.mass_kg - r * u
            r_dot = moment_z / parameters.yaw_inertia_kgm2
            pose_u, pose_v = position_point_body_velocity(
                local_state, parameters)
            x += sub_dt * (pose_u * math.cos(yaw) - pose_v * math.sin(yaw))
            y += sub_dt * (pose_u * math.sin(yaw) + pose_v * math.cos(yaw))
            yaw += sub_dt * r
            u += sub_dt * u_dot
            v += sub_dt * v_dot
            r += sub_dt * r_dot
            wheel_omegas = [omega + sub_dt * derivative
                            for omega, derivative in zip(wheel_omegas, omega_dot)]
            wheel_omegas = [0.0 if abs(omega) < 1.0e-9 else omega
                            for omega in wheel_omegas]
    steering_end = steering_segments[-1][2]
    return np.asarray([x, y, yaw, u, v, r, steering_end, *wheel_omegas],
                      dtype=float)


def step_unity_wheel_collider_with_suspension(
        state: np.ndarray, steering_target_norm: float,
        throttle_norm: float, dt: float,
        parameters: NativeModelParameters) -> np.ndarray:
    """Advance the offline Unity plant with explicit sprung-body states.

    The state extends the 11-state wheel plant with six deviations from the
    settled grounded pose: heave, heave rate, pitch, pitch rate, roll, and
    roll rate.  Each wheel load is generated by the captured WheelCollider
    spring/damper values, while horizontal tire forces use the serialized
    Unity curves.  Horizontal contact forces act below the exact COM and
    therefore excite the explicit pitch/roll states.

    This is a diagnostic model path.  It does not consume simulator truth
    during rollout, does not change Unity, and is not connected to ROS,
    odometry, or production MPC.
    """
    if state.shape != (17,):
        raise ValueError(
            f"expected Unity suspension state shape (17,), got {state.shape}")
    if parameters.unity_wheel_dynamics is None:
        raise ValueError(
            "step_unity_wheel_collider_with_suspension requires Unity wheel dynamics")
    if parameters.yaw_inertia_kgm2 is None:
        raise ValueError("Unity suspension plant requires dumped yaw inertia")
    if parameters.pitch_inertia_kgm2 is None:
        raise ValueError("Unity suspension plant requires dumped pitch inertia")
    if parameters.roll_inertia_kgm2 is None:
        raise ValueError("Unity suspension plant requires dumped roll inertia")
    if parameters.unity_com_height_m is None:
        raise ValueError("Unity suspension plant requires dumped COM height")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    target = (max(-1.0, min(1.0, float(steering_target_norm))) *
              parameters.steering_limit_rad)
    x, y, yaw, u, v, r, steering = (
        float(value) for value in state[:7])
    wheel_omegas = [float(value) for value in state[7:11]]
    heave, heave_rate, pitch, pitch_rate, roll, roll_rate = (
        float(value) for value in state[11:17])
    steering_segments = _unity_vehicle_controller_steering_segments(
        steering, steering_target_norm, dt, parameters)
    base_loads = unity_normal_loads(parameters)
    base_load_sum = sum(base_loads)
    base_pitch_moment = sum(
        -wheel.x_m * load
        for wheel, load in zip(parameters.wheels, base_loads))
    base_roll_moment = sum(
        wheel.y_m * load
        for wheel, load in zip(parameters.wheels, base_loads))
    dynamics = parameters.unity_wheel_dynamics
    wheel_damping = dynamics.wheel_damping_nms
    com_height = float(parameters.unity_com_height_m)

    for segment_dt, segment_start, segment_end in steering_segments:
        count = max(1, int(math.ceil(
            segment_dt / (INTEGRATION_SUBSTEP_S / 2.0))))
        sub_dt = segment_dt / count
        for sub_index in range(count):
            fraction = (sub_index + 0.5) / count
            delta = segment_start + fraction * (segment_end - segment_start)
            local_state = np.asarray(
                [x, y, yaw, u, v, r, delta, *wheel_omegas], dtype=float)
            loads = unity_suspension_loads(
                parameters, heave, pitch, roll, heave_rate, pitch_rate,
                roll_rate)
            components = _unity_contact_force_components(
                local_state, delta, parameters, loads)
            force_x = sum(component[0] for component in components)
            force_y = sum(component[1] for component in components)
            moment_z = sum(
                wheel.x_m * component[1] - wheel.y_m * component[0]
                for wheel, component in zip(parameters.wheels, components))
            if parameters.rigid_body_drag_per_s is not None:
                force_x -= parameters.rigid_body_drag_per_s * u
                force_y -= parameters.rigid_body_drag_per_s * v
            if parameters.rigid_body_angular_drag_per_s is not None:
                moment_z -= (parameters.rigid_body_angular_drag_per_s *
                             parameters.yaw_inertia_kgm2 * r)

            # WheelCollider contact forces act at the ground below the COM.
            # The vertical support forces also act at measured wheel points.
            pitch_moment = sum(
                -com_height * component[0] - wheel.x_m * load
                for wheel, component, load in zip(
                    parameters.wheels, components, loads))
            pitch_moment -= base_pitch_moment
            roll_moment = sum(
                wheel.y_m * load + com_height * component[1]
                for wheel, component, load in zip(
                    parameters.wheels, components, loads))
            roll_moment -= base_roll_moment
            heave_accel = (sum(loads) - base_load_sum) / parameters.mass_kg
            pitch_accel = pitch_moment / parameters.pitch_inertia_kgm2
            roll_accel = roll_moment / parameters.roll_inertia_kgm2
            if parameters.rigid_body_angular_drag_per_s is not None:
                pitch_accel -= (parameters.rigid_body_angular_drag_per_s *
                                pitch_rate)
                roll_accel -= (parameters.rigid_body_angular_drag_per_s *
                               roll_rate)

            motor_torques, brake_torques = _unity_wheel_torques(
                throttle_norm, parameters)
            omega_dot = []
            for index, omega in enumerate(wheel_omegas):
                radius = (parameters.wheel_radii_m[index]
                          if parameters.wheel_radii_m is not None
                          else parameters.wheel_radius_m)
                unbraked_torque = (
                    motor_torques[index] - components[index][2] * radius -
                    wheel_damping[index] * omega)
                brake = brake_torques[index]
                if brake > 0.0:
                    if abs(omega) <= 1.0e-9 and abs(unbraked_torque) <= brake:
                        wheel_torque = 0.0
                    else:
                        direction = (math.copysign(1.0, omega)
                                     if abs(omega) > 1.0e-9 else
                                     math.copysign(1.0, unbraked_torque))
                        wheel_torque = unbraked_torque - direction * brake
                        inertia = dynamics.inertia_for_speed(abs(u))
                        if (abs(omega) > 1.0e-9 and
                                omega * (omega + sub_dt * wheel_torque / inertia)
                                < 0.0):
                            wheel_torque = -omega * inertia / sub_dt
                else:
                    wheel_torque = unbraked_torque
                omega_dot.append(
                    wheel_torque / dynamics.inertia_for_speed(abs(u)))

            u_dot = force_x / parameters.mass_kg + r * v
            v_dot = force_y / parameters.mass_kg - r * u
            r_dot = moment_z / parameters.yaw_inertia_kgm2
            pose_u, pose_v = position_point_body_velocity(
                local_state, parameters)
            x += sub_dt * (pose_u * math.cos(yaw) - pose_v * math.sin(yaw))
            y += sub_dt * (pose_u * math.sin(yaw) + pose_v * math.cos(yaw))
            yaw += sub_dt * r
            u += sub_dt * u_dot
            v += sub_dt * v_dot
            r += sub_dt * r_dot
            heave_rate += sub_dt * heave_accel
            heave += sub_dt * heave_rate
            pitch_rate += sub_dt * pitch_accel
            pitch += sub_dt * pitch_rate
            roll_rate += sub_dt * roll_accel
            roll += sub_dt * roll_rate
            wheel_omegas = [omega + sub_dt * derivative
                            for omega, derivative in zip(wheel_omegas, omega_dot)]
            wheel_omegas = [0.0 if abs(omega) < 1.0e-9 else omega
                            for omega in wheel_omegas]
    steering_end = steering_segments[-1][2]
    return np.asarray([
        x, y, yaw, u, v, r, steering_end, *wheel_omegas,
        heave, heave_rate, pitch, pitch_rate, roll, roll_rate,
    ], dtype=float)


def _move_towards(current: float, target: float, maximum_delta: float) -> float:
    return current + max(-maximum_delta, min(maximum_delta, target - current))


def _unity_vehicle_controller_source_step(
        current_delta: float, target_norm: float, dt: float,
        parameters: NativeModelParameters) -> float:
    """Replay one Unity ``VehicleController.Steer`` source update.

    Unity stores its internal ``SteeringAngle`` with the opposite sign to the
    API-positive effective steering state.  The comparison and asymmetric
    clamp below mirror the source branch in that internal convention.
    """
    if not all(math.isfinite(value) for value in (
            current_delta, target_norm, dt)) or dt <= 0.0:
        raise ValueError("Unity steering source inputs must be finite and positive")
    command = max(-1.0, min(1.0, target_norm))
    internal_angle = -current_delta
    target_internal_angle = -command * parameters.steering_limit_rad
    if command > internal_angle / parameters.steering_limit_rad:
        internal_angle -= parameters.steering_rate_radps * dt
        if internal_angle <= target_internal_angle:
            internal_angle = target_internal_angle
    else:
        internal_angle += parameters.steering_rate_radps * dt
        if internal_angle >= target_internal_angle:
            internal_angle = target_internal_angle
    return -internal_angle


def _unity_vehicle_controller_steering_next(
        current_delta: float, target_norm: float, dt: float,
        parameters: NativeModelParameters) -> float:
    source_dt = parameters.steering_source_fixed_dt_s
    if not math.isfinite(source_dt) or source_dt <= 0.0:
        raise ValueError("Unity steering source timestep must be positive")
    count = max(1, int(round(dt / source_dt)))
    steering = current_delta
    elapsed = 0.0
    for index in range(count):
        duration = dt - elapsed if index == count - 1 else source_dt
        steering = _unity_vehicle_controller_source_step(
            steering, target_norm, max(duration, np.finfo(float).eps), parameters)
        elapsed += duration
    return steering


def _unity_vehicle_controller_steering_segments(
        current_delta: float, target_norm: float, dt: float,
        parameters: NativeModelParameters
        ) -> list[tuple[float, float, float]]:
    """Return ``(duration, start, end)`` segments at Unity source cadence."""
    source_dt = parameters.steering_source_fixed_dt_s
    count = max(1, int(round(dt / source_dt)))
    segments: list[tuple[float, float, float]] = []
    steering = current_delta
    elapsed = 0.0
    for index in range(count):
        duration = dt - elapsed if index == count - 1 else source_dt
        duration = max(duration, np.finfo(float).eps)
        next_steering = _unity_vehicle_controller_source_step(
            steering, target_norm, duration, parameters)
        segments.append((duration, steering, next_steering))
        steering = next_steering
        elapsed += duration
    return segments


def _contact_forces(state: np.ndarray, steering_angle: float,
                    parameters: NativeModelParameters) -> tuple[float, float, float, float]:
    loads = static_normal_loads(parameters)
    contacts = contact_kinematics(state, steering_angle, parameters)
    force_x = force_y = moment_z = 0.0
    for index, (contact, normal_load) in enumerate(zip(contacts, loads)):
        fx_tire = (parameters.longitudinal_gain * normal_load *
                   friction_value(
                       contact["sx"],
                       parameters.longitudinal_curve_for_wheel(index)))
        # Positive Sy means the contact patch moves to the left; tire force
        # opposes that motion, hence the minus sign.
        fy_tire = (-parameters.lateral_gain * normal_load *
                   friction_value(
                       contact["sy"],
                       parameters.lateral_curve_for_wheel(index)))
        cosine = math.cos(contact["steering_rad"])
        sine = math.sin(contact["steering_rad"])
        fx = cosine * fx_tire - sine * fy_tire
        fy = sine * fx_tire + cosine * fy_tire
        force_x += fx
        force_y += fy
        moment_z += contact["x_m"] * fy - contact["y_m"] * fx
    speed = abs(float(state[3]))
    drag = (parameters.drag_linear_n_per_mps * speed +
            parameters.drag_quadratic_n_per_mps2 * speed * speed)
    force_x -= math.copysign(drag, float(state[3]) or 1.0)
    return force_x, force_y, moment_z, float(sum(loads))


def _effective_coordinates(state: np.ndarray, steering_angle: float,
                           parameters: NativeModelParameters) -> tuple[float, float, float]:
    """Return qx, qy_front, qy_rear for the inertia-free formulation."""
    if parameters.contact_model == "axle":
        _, _, _, u, v, r, _ = (float(value) for value in state[:7])
        wheel_surface_speed = (float(state[7]) if state.shape == (8,)
                               else 0.5 * (float(state[7]) + float(state[8])))
        front_vx = u
        front_vy = v + parameters.lf_m * r
        rear_vx = u
        rear_vy = v - parameters.lr_m * r
        cosine = math.cos(steering_angle)
        sine = math.sin(steering_angle)
        front_vx_tire = cosine * front_vx + sine * front_vy
        front_vy_tire = -sine * front_vx + cosine * front_vy
        denominator_front = max(abs(front_vx_tire), MIN_SLIP_SPEED_MPS)
        denominator_rear = max(abs(rear_vx), MIN_SLIP_SPEED_MPS)
        front_sx = (wheel_surface_speed - front_vx_tire) / denominator_front
        rear_sx = (wheel_surface_speed - rear_vx) / denominator_rear
        front_sy = front_vy_tire / denominator_front
        rear_sy = rear_vy / denominator_rear
        return (
            friction_value(front_sx, parameters.longitudinal_curve_for_wheel(0)) +
            friction_value(rear_sx, parameters.longitudinal_curve_for_wheel(2)),
            -friction_value(front_sy, parameters.lateral_curve_for_wheel(0)),
            -friction_value(rear_sy, parameters.lateral_curve_for_wheel(2)),
        )
    if parameters.contact_model != "four_wheel":
        raise ValueError(f"unsupported contact model: {parameters.contact_model}")
    contacts = contact_kinematics(state, steering_angle, parameters)
    qx = sum(
        friction_value(contact["sx"],
                       parameters.longitudinal_curve_for_wheel(index))
        for index, (contact, wheel) in enumerate(zip(contacts, parameters.wheels))
        if wheel.driven)
    # qy is the normalized body lateral force shape. The tire force opposes
    # contact lateral velocity, so negate the signed friction value.
    q_front = sum(
        -friction_value(contact["sy"],
                        parameters.lateral_curve_for_wheel(index))
        for index, contact in enumerate(contacts[:2])) / 2.0
    q_rear = sum(
        -friction_value(contact["sy"],
                        parameters.lateral_curve_for_wheel(index))
        for index, contact in enumerate(contacts[2:], start=2)) / 2.0
    return qx, q_front, q_rear


def _moment_basis_coordinates(
        state: np.ndarray, steering_angle: float,
        parameters: NativeModelParameters) -> tuple[float, float, float]:
    """Return normalized body force and yaw-moment basis for VS2.5.

    Each wheel's documented friction response is evaluated in its tire frame,
    rotated into the body frame, and accumulated at the actual contact point.
    The result is deliberately dimensionless.  Fitted coefficients therefore
    remain effective force/moment gains rather than claims about tire force or
    rigid-body inertia.
    """
    if parameters.contact_model != "moment_basis":
        raise ValueError(
            "moment-basis coordinates require contact_model='moment_basis'")
    components = _split_moment_basis_coordinates(
        state, steering_angle, parameters)
    return components[0], components[1] + components[2], \
        components[3] + components[4] + components[5]


def _split_moment_basis_coordinates(
        state: np.ndarray, steering_angle: float,
        parameters: NativeModelParameters) -> tuple[float, float, float, float,
                                                       float, float]:
    """Return split four-contact force/moment bases for offline screening.

    The six values are total longitudinal body force, front/rear lateral body
    force, front/rear ``x*Fy`` moment, and the complete longitudinal
    ``-y*Fx`` moment. They remain normalized coordinates; fitted coefficients
    are effective simulator gains rather than physical stiffness or inertia.
    """
    if parameters.contact_model not in ("moment_basis", "moment_basis_split"):
        raise ValueError(
            "split moment basis requires a moment-basis contact model")
    contacts = contact_kinematics(state, steering_angle, parameters)
    loads = static_normal_loads(parameters)
    load_factors = [1.0] * len(contacts)
    if parameters.effective_load_transfer_height_m > 0.0:
        # Use the causal steady-turn approximation ay ~= r*u. The correction
        # preserves total normal load and is an effective screening term, not
        # a claim that the dump exposes the true roll-center/CoM height.
        _, _, _, u, _, r, _ = (float(value) for value in state[:7])
        lateral_acceleration = r * u
        correction_per_wheel = (
            parameters.mass_kg * lateral_acceleration *
            parameters.effective_load_transfer_height_m /
            (2.0 * parameters.track_m))
        load_factors = [
            max(0.0, load + math.copysign(
                correction_per_wheel, wheel.y_m or 1.0)) / static_load
            for load, static_load, wheel in zip(
                loads, loads, parameters.wheels)
        ]
    qx = qy_front = qy_rear = 0.0
    qmoment_front = qmoment_rear = qmoment_longitudinal = 0.0
    for index, (contact, wheel) in enumerate(zip(contacts, parameters.wheels)):
        if not wheel.driven:
            continue
        fx_tire = friction_value(
            contact["sx"], parameters.longitudinal_curve_for_wheel(index))
        # Positive Sy is contact-patch motion to the left. Tire force opposes
        # it, matching the sign convention used by the physical path.
        fy_tire = -friction_value(
            contact["sy"], parameters.lateral_curve_for_wheel(index))
        cosine = math.cos(contact["steering_rad"])
        sine = math.sin(contact["steering_rad"])
        fx_body = cosine * fx_tire - sine * fy_tire
        fy_body = sine * fx_tire + cosine * fy_tire
        load_factor = load_factors[index]
        qx += load_factor * fx_body
        if index < 2:
            qy_front += load_factor * fy_body
            qmoment_front += load_factor * contact["x_m"] * fy_body
        else:
            qy_rear += load_factor * fy_body
            qmoment_rear += load_factor * contact["x_m"] * fy_body
        qmoment_longitudinal -= load_factor * contact["y_m"] * fx_body
    return (qx, qy_front, qy_rear, qmoment_front, qmoment_rear,
            qmoment_longitudinal)


def step(state: np.ndarray, steering_target_norm: float,
         throttle_norm: float, dt: float,
         parameters: NativeModelParameters) -> np.ndarray:
    """Advance the causal pose/body/actuator state.

    The state is ``[X, Y, yaw, u, v, r, steering, wheel_speed]`` for the
    compatibility mean-wheel form, or
    ``[X, Y, yaw, u, v, r, steering, wheel_left_speed,
    wheel_right_speed]`` for the legal two-encoder form.
    """
    if state.shape not in ((8,), (9,)):
        raise ValueError(f"expected state shape (8,) or (9,), got {state.shape}")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    throttle = max(0.0, min(1.0, float(throttle_norm)))
    target = max(-1.0, min(1.0, float(steering_target_norm))) * parameters.steering_limit_rad
    x, y, yaw, u, v, r, steering = (float(value) for value in state[:7])
    if state.shape == (8,):
        wheel_starts = (float(state[7]),)
    else:
        wheel_starts = (float(state[7]), float(state[8]))
    steering_end = _move_towards(
        steering, target, parameters.steering_rate_radps * dt)
    wheel_ends = tuple(
        parameters.drive_state.next_wheel_speed(u, wheel, throttle)
        for wheel in wheel_starts)
    count = max(1, int(math.ceil(dt / INTEGRATION_SUBSTEP_S)))
    sub_dt = dt / count
    for index in range(count):
        fraction = (index + 0.5) / count
        delta = steering + fraction * (steering_end - steering)
        wheel_mids = tuple(
            start + fraction * (end - start)
            for start, end in zip(wheel_starts, wheel_ends))
        local_state = np.asarray(
            [x, y, yaw, u, v, r, delta, *wheel_mids[:len(wheel_starts)]])
        if parameters.yaw_inertia_kgm2 is not None:
            force_x, force_y, moment_z, _ = _contact_forces(
                local_state, delta, parameters)
            u_dot = force_x / parameters.mass_kg + r * v
            v_dot = force_y / parameters.mass_kg - r * u
            r_dot = moment_z / parameters.yaw_inertia_kgm2
        elif parameters.contact_model == "moment_basis_split":
            if not parameters.has_split_moment_basis_gains:
                raise ValueError(
                    "split moment basis requires effective force/moment gains")
            (qx, qy_front, qy_rear, qmoment_front, qmoment_rear,
             qmoment_longitudinal) = _split_moment_basis_coordinates(
                local_state, delta, parameters)
            u_dot = (
                parameters.effective_force_x_gain_mps2 * qx -
                parameters.effective_drag_linear_per_s * u -
                parameters.effective_drag_quadratic_per_m * u * abs(u) + r * v)
            v_dot = (
                parameters.effective_force_y_front_gain_mps2 * qy_front +
                parameters.effective_force_y_rear_gain_mps2 * qy_rear - r * u)
            r_dot = (
                parameters.effective_yaw_lateral_front_gain_per_s2 *
                qmoment_front +
                parameters.effective_yaw_lateral_rear_gain_per_s2 *
                qmoment_rear +
                parameters.effective_yaw_longitudinal_gain_per_s2 *
                qmoment_longitudinal)
        elif parameters.contact_model == "moment_basis":
            if not parameters.has_moment_basis_gains:
                raise ValueError(
                    "moment_basis step requires effective force/moment gains")
            qx, qy, qmoment = _moment_basis_coordinates(
                local_state, delta, parameters)
            u_dot = (
                parameters.effective_force_x_gain_mps2 * qx -
                parameters.effective_drag_linear_per_s * u -
                parameters.effective_drag_quadratic_per_m * u * abs(u) + r * v)
            v_dot = (
                parameters.effective_force_y_gain_mps2 * qy - r * u)
            r_dot = parameters.effective_yaw_moment_gain_per_s2 * qmoment
            r_dot -= parameters.effective_yaw_damping_per_s * r
        else:
            if not parameters.has_effective_gains:
                raise ValueError(
                    "step requires dumped yaw inertia or effective lateral/yaw gains")
            qx, q_front, q_rear = _effective_coordinates(
                local_state, delta, parameters)
            u_dot = (
                parameters.effective_longitudinal_gain_mps2 * qx -
                parameters.effective_drag_linear_per_s * u -
                parameters.effective_drag_quadratic_per_m * u * abs(u) + r * v)
            v_dot = (
                parameters.effective_lateral_front_mps2 * q_front +
                parameters.effective_lateral_rear_mps2 * q_rear - r * u)
            r_dot = (
                parameters.effective_yaw_front_per_s2 * q_front -
                parameters.effective_yaw_rear_per_s2 * q_rear)
        pose_u, pose_v = position_point_body_velocity(local_state, parameters)
        x += sub_dt * (pose_u * math.cos(yaw) - pose_v * math.sin(yaw))
        y += sub_dt * (pose_u * math.sin(yaw) + pose_v * math.cos(yaw))
        yaw += sub_dt * r
        u += sub_dt * u_dot
        v += sub_dt * v_dot
        r += sub_dt * r_dot
        u = max(0.0, u)
    return np.asarray(
        [x, y, yaw, u, v, r, steering_end, *wheel_ends], dtype=float)


def parameters_from_dump(path: Path) -> NativeModelParameters:
    """Load fixed structure from the Unity diagnostic JSON with provenance."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    source_fixed_dt = float(payload.get(
        "fixedDeltaTime", UNITY_STEERING_SOURCE_FIXED_DT_S))
    if not math.isfinite(source_fixed_dt) or source_fixed_dt <= 0.0:
        raise ValueError("diagnostic fixedDeltaTime must be positive and finite")
    vehicle = payload["vehicle"]
    rigid_body = vehicle["rigidBody"]
    yaw_axis = rigid_body.get("yawAxis")
    if yaw_axis != "body_y":
        raise ValueError(
            "diagnostic dump must explicitly declare Unity body Y as yawAxis; "
            "the legacy local-Z yaw field is not accepted")
    wheels = vehicle["wheels"]
    if len(wheels) != 4:
        raise ValueError("diagnostic dump must contain four wheels")

    def vector(value: dict[str, float], axis: str) -> float:
        return float(value[axis])

    com = rigid_body["centerOfMass"]
    wheel_positions = [wheel["positionVehicleFrame"] for wheel in wheels]
    # Unity frame is x-right/z-forward; API body frame is x-forward/y-left.
    com_api_x = vector(com, "z")
    com_api_y = -vector(com, "x")
    positions = [
        (vector(position, "z") - com_api_x,
         -vector(position, "x") - com_api_y)
        for position in wheel_positions
    ]
    front = positions[:2]
    rear = positions[2:]
    wheelbase = sum(position[0] for position in front) / 2.0 - \
        sum(position[0] for position in rear) / 2.0
    track = abs(front[0][1] - front[1][1])
    # Keep Rigidbody.mass separate from WheelCollider.mass. Unity's dump does
    # not establish additive total-mass semantics; wheel masses remain
    # available as explicit per-wheel structural parameters below.
    rigidbody_mass = float(rigid_body["mass"])

    def curve(raw: dict[str, Any]) -> TireCurveParameters:
        return TireCurveParameters(
            extremum_slip=float(raw["extremumSlip"]),
            extremum_value=float(raw["extremumValue"]),
            asymptote_slip=float(raw["asymptoteSlip"]),
            asymptote_value=float(raw["asymptoteValue"]),
            stiffness=float(raw["stiffness"]),
            source="unity_diagnostic_dump",
        )

    longitudinal_curves = tuple(
        curve(wheel["forwardFriction"]) for wheel in wheels)
    lateral_curves = tuple(
        curve(wheel["sidewaysFriction"]) for wheel in wheels)
    longitudinal_curve = longitudinal_curves[0]
    lateral_curve = lateral_curves[0]
    sprung_masses = (
        tuple(float(wheel["sprungMass"]) for wheel in wheels)
        if all("sprungMass" in wheel for wheel in wheels) else None)
    suspension_springs = (
        tuple(float(wheel["suspensionSpring"]["spring"]) for wheel in wheels)
        if all(isinstance(wheel.get("suspensionSpring"), dict) and
               "spring" in wheel["suspensionSpring"] for wheel in wheels)
        else None)
    suspension_dampers = (
        tuple(float(wheel["suspensionSpring"]["damper"]) for wheel in wheels)
        if all(isinstance(wheel.get("suspensionSpring"), dict) and
               "damper" in wheel["suspensionSpring"] for wheel in wheels)
        else None)
    suspension_targets = (
        tuple(float(wheel["suspensionSpring"]["targetPosition"])
              for wheel in wheels)
        if all(isinstance(wheel.get("suspensionSpring"), dict) and
               "targetPosition" in wheel["suspensionSpring"]
               for wheel in wheels)
        else None)
    body_inertias = body_frame_inertias(
        rigid_body["inertiaTensor"], rigid_body["inertiaTensorRotation"])
    for axis in ("x", "y", "z"):
        field = f"bodyInertia{axis.upper()}"
        if field not in rigid_body:
            raise ValueError(
                f"diagnostic dump is missing corrected {field} field")
        reported = float(rigid_body[field])
        if not math.isfinite(reported) or not math.isclose(
                body_inertias[axis], reported, rel_tol=1.0e-4,
                abs_tol=2.0e-6):
            raise ValueError(
                f"diagnostic {field} disagrees with inertia tensor projection: "
                f"derived={body_inertias[axis]} reported={reported}")
    derived_yaw_inertia = body_inertias["y"]
    reported_yaw_inertia = rigid_body.get("yawInertiaBodyFrame")
    if reported_yaw_inertia is not None:
        reported = float(reported_yaw_inertia)
        if not math.isfinite(reported) or not math.isclose(
                derived_yaw_inertia, reported, rel_tol=1.0e-4,
                abs_tol=2.0e-6):
            raise ValueError(
                "diagnostic yawInertiaBodyFrame disagrees with body-Y inertia "
                f"projection: derived={derived_yaw_inertia} reported={reported}")
    com_x_from_rear_axle = -sum(position[0] for position in rear) / 2.0
    unity_wheel_dynamics = None
    if "motorTorqueNm" in vehicle:
        drive_type = str(vehicle.get("driveType", ""))
        brake_type = str(vehicle.get("brakeType", ""))
        drive_fractions = ((0.25,) * 4 if drive_type == "CAWD" else
                           (0.0,) * 4)
        brake_fractions = ((1.0,) * 4 if brake_type == "CAWB" else
                           (0.0,) * 4)
        unity_wheel_dynamics = UnityWheelDynamicsParameters(
            motor_torque_nm=float(vehicle["motorTorqueNm"]),
            # VehicleController sets BrakeTorque = MotorTorque.
            brake_torque_nm=float(vehicle["motorTorqueNm"]),
            throttle_limit=float(vehicle.get("throttleLimit", 1.0)),
            wheel_damping_nms=tuple(
                float(wheel["wheelDampingRate"]) for wheel in wheels),
            drive_torque_fractions=drive_fractions,
            brake_torque_fractions=brake_fractions,
            provenance=(
                f"unity_diagnostic_dump:{path};"
                "vehiclecontroller:active_drive_and_brake_branch;"
                "exact_trace:effective_rotational_inertia_regimes"))
    return NativeModelParameters(
        mass_kg=rigidbody_mass,
        yaw_inertia_kgm2=derived_yaw_inertia,
        # API-frame pitch is Unity body-x; API-frame roll is Unity body-z.
        pitch_inertia_kgm2=body_inertias["x"],
        roll_inertia_kgm2=body_inertias["z"],
        com_x_from_rear_axle_m=com_x_from_rear_axle,
        position_offset_from_velocity_point_x_m=-com_x_from_rear_axle,
        wheelbase_m=wheelbase,
        steering_geometry_wheelbase_m=float(
            vehicle.get("wheelbaseM", wheelbase)),
        track_m=track,
        wheel_radius_m=sum(float(wheel["radius"]) for wheel in wheels) / 4.0,
        controller_wheel_radius_m=(
            float(vehicle["wheelRadiusControllerM"])
            if "wheelRadiusControllerM" in vehicle else None),
        drive_type=(str(vehicle["driveType"])
                    if vehicle.get("driveType") is not None else None),
        steering_limit_rad=float(vehicle["steeringLimitRad"]),
        steering_rate_radps=float(vehicle["steeringRateRadPerSecond"]),
        steering_source_fixed_dt_s=source_fixed_dt,
        # The source controller stores SteeringAngle = -AppliedSteering in
        # Unity's x-right frame.  The x-right -> y-left model-frame
        # conversion reverses that axis as well, so the effective model
        # wheel angle is positive for positive AppliedSteering.
        steering_to_wheel_angle_sign=1.0,
        longitudinal_curve=longitudinal_curve,
        lateral_curve=lateral_curve,
        wheel_longitudinal_curves=longitudinal_curves,
        wheel_lateral_curves=lateral_curves,
        wheel_radii_m=tuple(float(wheel["radius"]) for wheel in wheels),
        wheel_masses_kg=tuple(float(wheel["mass"]) for wheel in wheels),
        wheel_sprung_masses_kg=sprung_masses,
        unity_com_height_m=float(com["y"]),
        unity_normal_load_mode="static_sprung_mass",
        suspension_distances_m=tuple(
            float(wheel["suspensionDistance"]) for wheel in wheels),
        suspension_spring_rates_n_per_m=suspension_springs,
        suspension_damper_rates_ns_per_m=suspension_dampers,
        suspension_target_positions=suspension_targets,
        wheel_damping_rates_per_s=tuple(
            float(wheel["wheelDampingRate"]) for wheel in wheels),
        force_app_point_distances_m=tuple(
            float(wheel["forceAppPointDistance"]) for wheel in wheels),
        rigid_body_drag_per_s=float(rigid_body.get("drag", 0.0)),
        rigid_body_angular_drag_per_s=float(
            rigid_body.get("angularDrag", 0.0)),
        rigid_body_max_angular_velocity_radps=(
            float(rigid_body["maxAngularVelocity"])
            if "maxAngularVelocity" in rigid_body else None),
        unity_wheel_dynamics=unity_wheel_dynamics,
        parameter_provenance="unity_diagnostic_dump:" + str(path),
    )
