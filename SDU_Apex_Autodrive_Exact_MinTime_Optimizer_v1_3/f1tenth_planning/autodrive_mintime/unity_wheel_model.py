from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import casadi as ca

from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
    MOTOR_TORQUE_NM,
    RIGIDBODY_DRAG_PER_S,
    RIGIDBODY_MASS_KG,
    RIGIDBODY_YAW_INERTIA_KGM2,
    STEERING_LIMIT_RAD,
    STEERING_RATE_RADPS,
    WHEEL_DAMPING_RATE_NMS,
    WHEEL_RADIUS_M,
    WHEEL_X_M,
    WHEEL_Y_M,
)


ROOT = Path(__file__).resolve().parents[3]
TRAIN_CAPTURE = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "dynamics_source_parity_20261012_r07/"
    "wheel_dynamics_unity_vehicle_model_parity_source_20261012_r07.csv")
WHEEL_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "wheel_spin_state_inputs_v10_20261013.json")
GUIDE_TIRE_FIT = ROOT / (
    "live_runs/unity_exact_binary_audit_20261012/"
    "tire_curve_force_balance_guide_slips_20261013.json")

_WHEEL_KEYS = ("front_left", "front_right", "rear_left", "rear_right")


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


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    slip_speed_floor_mps: float
    longitudinal_scale: float
    longitudinal_zero_slip_tangent: float
    lateral_scale: float
    lateral_zero_slip_tangent: float
    forward_curve: tuple[tuple[float, ...], ...]
    lateral_curve: tuple[tuple[float, ...], ...]
    static_normal_load_n: tuple[float, float, float, float]

    @classmethod
    def from_training_capture(cls, capture: Path = TRAIN_CAPTURE
                              ) -> "UnityWheelModel":
        required = (capture, GUIDE_TIRE_FIT, WHEEL_FIT)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing Unity model source artifacts: "
                                    + ", ".join(missing))
        metadata = _capture_metadata(capture)
        tire_fit = _json(GUIDE_TIRE_FIT)["curve_model"]["training_fit"]
        wheel_fit = _json(WHEEL_FIT)
        lateral_fit = (tire_fit["lateral_scale"],
                       tire_fit["lateral_zero_slip_tangent"])
        wheel_balance = wheel_fit["training_fit_state_reconstructed_inputs"]

        sprung = tuple(float(metadata[f"{name}_sprung_mass_kg"])
                       for name in _WHEEL_KEYS)
        forward = tuple(tuple(float(value) for value in metadata[
            f"{name}_forward_curve"].split(",")) for name in _WHEEL_KEYS)
        lateral = tuple(tuple(float(value) for value in metadata[
            f"{name}_sideways_curve"].split(",")) for name in _WHEEL_KEYS)
        if any(len(curve) != 5 for curve in (*forward, *lateral)):
            raise ValueError("Unity capture friction-curve metadata is malformed")
        radii = tuple(float(metadata[f"{name}_radius_m"])
                      for name in _WHEEL_KEYS)
        if max(radii) - min(radii) > 1.0e-6:
            raise ValueError("the four Unity wheel radii are not equal")

        # This quasi-static load transfer splits roll load by the two equal
        # front/rear spring pairs in the serialized Unity vehicle. The static
        # wheel loads and fitted tire response come from the 1 kHz training
        # capture; r11 remains held out for transfer scoring.
        static_load = tuple(mass * 9.80665 for mass in sprung)
        return cls(
            mass_kg=RIGIDBODY_MASS_KG,
            yaw_inertia_kgm2=RIGIDBODY_YAW_INERTIA_KGM2,
            drag_per_s=RIGIDBODY_DRAG_PER_S,
            wheel_radius_m=radii[0],
            wheel_radius_scale=1.0,
            wheel_inertia_kgm2=float(wheel_balance["effective_inertia_kgm2"]),
            wheel_damping_nms=WHEEL_DAMPING_RATE_NMS,
            steering_limit_rad=STEERING_LIMIT_RAD,
            steering_rate_radps=STEERING_RATE_RADPS,
            wheelbase_m=0.324,
            track_m=0.236,
            motor_torque_nm=MOTOR_TORQUE_NM,
            body_com_height_m=CAPTURED_COM_HEIGHT_ABOVE_ROAD_M,
            sprung_mass_kg=sprung,
            # The guide defines Sx=(R*w-vx)/vx and Sy=vy/abs(vx).
            # The 0.5 m/s floor is the documented greybox regularization used
            # by the prior recursive comparisons; WheelHit telemetry fits are
            # deliberately not used as tire-force slip coordinates here.
            slip_speed_floor_mps=0.5,
            longitudinal_scale=float(tire_fit["longitudinal_scale"]),
            longitudinal_zero_slip_tangent=float(
                tire_fit["longitudinal_zero_slip_tangent"]),
            lateral_scale=float(lateral_fit[0]),
            lateral_zero_slip_tangent=float(lateral_fit[1]),
            forward_curve=forward,
            lateral_curve=lateral,
            static_normal_load_n=static_load,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source_capture": str(TRAIN_CAPTURE.relative_to(ROOT)),
            "training_only_fits": {
                "guide_slip_tire_force_response": str(
                    GUIDE_TIRE_FIT.relative_to(ROOT)),
                "effective_wheel_spin_inertia": str(
                    WHEEL_FIT.relative_to(ROOT)),
            },
            "mass_kg": self.mass_kg,
            "yaw_inertia_kgm2": self.yaw_inertia_kgm2,
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_radius_scale": self.wheel_radius_scale,
            "wheel_inertia_kgm2": self.wheel_inertia_kgm2,
            "steering_limit_rad": self.steering_limit_rad,
            "steering_rate_radps": self.steering_rate_radps,
            "motor_torque_total_nm": self.motor_torque_nm,
            "forward_curves": self.forward_curve,
            "lateral_curves": self.lateral_curve,
            "tire_fit": {
                "longitudinal_scale": self.longitudinal_scale,
                "longitudinal_zero_slip_tangent": self.longitudinal_zero_slip_tangent,
                "lateral_scale": self.lateral_scale,
                "lateral_zero_slip_tangent": self.lateral_zero_slip_tangent,
            },
            "load_model": (
                "source sprung-mass static loads plus Newtonian longitudinal "
                "and lateral transfer; equal front/rear spring-pair roll split"),
            "model_boundary": (
                "Unity WheelCollider native solver is closed; configured curve "
                "landmarks and source-derived rigid-body equations are explicit, "
                "unpublished spline/combined-slip details are fitted from r07"),
            "tire_slip_coordinates": (
                "AutoDRIVE guide Sx=(R*omega-vx)/max(vx,0.5), "
                "Sy=vy/max(abs(vx),0.5); WheelHit diagnostic slips are not "
                "used as force-law inputs"),
        }


def _odd_cubic_curve(slip, curve: tuple[float, ...], tangent: float):
    extremum_slip, extremum_value, asymptote_slip, asymptote_value, stiffness = curve
    signed_t = slip / extremum_slip
    abs_t = ca.fabs(signed_t)
    first = (
        tangent * extremum_slip
        * (signed_t**3 - 2.0 * signed_t * abs_t + signed_t)
        + extremum_value * (-2.0 * signed_t**3 + 3.0 * signed_t * abs_t)
    )
    magnitude = ca.fabs(slip)
    decay_t = (magnitude - extremum_slip) / (asymptote_slip - extremum_slip)
    smooth = 3.0 * decay_t**2 - 2.0 * decay_t**3
    sign = ca.if_else(slip >= 0.0, 1.0, -1.0)
    second = sign * (extremum_value
                     + (asymptote_value - extremum_value) * smooth)
    plateau = sign * asymptote_value
    return stiffness * ca.if_else(
        magnitude < extremum_slip, first,
        ca.if_else(magnitude < asymptote_slip, second, plateau))


def tire_forces_casadi(model: UnityWheelModel, u, v, yaw_rate, steering,
                       wheel_omega, normal_loads):
    """Four-wheel force/slip equations in the Unity capture's fitted frame."""
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
        vx_body = u - yaw_rate * WHEEL_Y_M[index]
        vy_body = v + yaw_rate * WHEEL_X_M[index]
        forward_speed = ca.cos(angle) * vx_body + ca.sin(angle) * vy_body
        lateral_speed = -ca.sin(angle) * vx_body + ca.cos(angle) * vy_body
        wheel_surface = (wheel_omega[index] * model.wheel_radius_m
                         * model.wheel_radius_scale)
        slip_den = ca.fmax(ca.fabs(forward_speed), model.slip_speed_floor_mps)
        slip_x = (wheel_surface - forward_speed) / slip_den
        slip_y = lateral_speed / slip_den
        fx = (model.longitudinal_scale * normal_loads[index]
              * _odd_cubic_curve(slip_x, model.forward_curve[index],
                                 model.longitudinal_zero_slip_tangent))
        fy = (-model.lateral_scale * normal_loads[index]
              * _odd_cubic_curve(slip_y, model.lateral_curve[index],
                                 model.lateral_zero_slip_tangent))
        fx_body = fx * ca.cos(angle) - fy * ca.sin(angle)
        fy_body = fx * ca.sin(angle) + fy * ca.cos(angle)
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


def wheel_torque_casadi(model: UnityWheelModel, throttle):
    """Common-actuator AWD: source MotorTorque split equally across wheels."""
    per_wheel = model.motor_torque_nm * throttle / 4.0
    return (per_wheel,) * 4

