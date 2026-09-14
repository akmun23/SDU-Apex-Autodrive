"""Reference implementation of the identified physical vehicle plant.

This module is offline model-identification code.  It deliberately has no
ROS, simulator, or runtime-controller dependencies.  The same equations are
mirrored by ``f1tenth_mpc/src/vehicle_plant.c`` for native replay; production
MPC parameters are not changed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np


"""Measured Unity structural anchors used by the offline plant candidate.

These are not fitted from the old replay data.  The tire/drive coefficients
below remain candidate parameters and must still pass the current handoff's
blind plant gates before any runtime use.
"""

MASS_KG = 3.470
LF_M = 0.174679914
LR_M = 0.155320086
IZ_KGM2 = 0.0276985662
MAX_STEERING_RAD = 0.5236
STEERING_RATE_RADPS = 3.2
WHEEL_RADIUS_M = 0.059
GRAVITY_MPS2 = 9.81
MAX_SPEED_MPS = 22.88
MIN_SLIP_SPEED_MPS = 0.5
INTEGRATION_SUBSTEP_S = 0.002


@dataclass(frozen=True)
class PlantParameters:
    """Physical parameters shared by Python fitting and native replay."""

    mass_kg: float = MASS_KG
    lf_m: float = LF_M
    lr_m: float = LR_M
    iz_kgm2: float = IZ_KGM2
    max_steering_rad: float = MAX_STEERING_RAD
    steering_rate_radps: float = STEERING_RATE_RADPS
    max_speed_mps: float = MAX_SPEED_MPS
    force_max_n: float = 17.7166
    slip_gain_per_mps: float = 2.2574
    coast_speed_drag_n_per_mps: float = 0.8910
    cf_n_per_rad: float = 51.40
    cr_n_per_rad: float = 43.10
    df_n: float = 11.50
    dr_n: float = 10.60
    wheel_coefficients: tuple[float, ...] = (
        -0.0003516271, 0.0419576436, 23.9652144, 0.0251136087,
        -0.4763574048, 0.0006345492,
    )
    tire_model: str = "tanh"

    @classmethod
    def from_longitudinal_parameters(
            cls, longitudinal: dict[str, Any], **overrides: Any
    ) -> "PlantParameters":
        dynamics = longitudinal.get("wheel_dynamics", {})
        values = {
            "force_max_n": float(longitudinal.get("force_max_n", cls.force_max_n)),
            "slip_gain_per_mps": float(longitudinal.get(
                "slip_gain_per_mps", cls.slip_gain_per_mps)),
            "coast_speed_drag_n_per_mps": float(longitudinal.get(
                "coast_speed_drag_n_per_mps", cls.coast_speed_drag_n_per_mps)),
        }
        if dynamics.get("coefficients"):
            values["wheel_coefficients"] = tuple(
                float(value) for value in dynamics["coefficients"])
        values.update(overrides)
        return cls(**values)

    def with_lateral(self, lateral: dict[str, float]) -> "PlantParameters":
        return replace(
            self,
            iz_kgm2=float(lateral.get("iz_kgm2", self.iz_kgm2)),
            cf_n_per_rad=float(lateral["cf_n_per_rad"]),
            cr_n_per_rad=float(lateral["cr_n_per_rad"]),
            df_n=float(lateral["df_n"]),
            dr_n=float(lateral["dr_n"]),
            tire_model=str(lateral.get("tire_model", self.tire_model)),
        )


def _move_towards(current: float, target: float, maximum_delta: float) -> float:
    difference = target - current
    return current + max(-maximum_delta, min(maximum_delta, difference))


def _wheel_next(body_u: float, wheel: float, throttle: float,
                parameters: PlantParameters) -> float:
    coeff = np.asarray(parameters.wheel_coefficients, dtype=float)
    features = np.asarray([
        1.0, wheel, throttle, wheel * throttle,
        throttle * throttle, body_u,
    ])
    return max(0.0, float(features @ coeff))


def _tire_force(alpha: float, stiffness: float, peak: float,
                tire_model: str) -> float:
    if tire_model == "linear_saturated":
        return max(-peak, min(peak, stiffness * alpha))
    if tire_model != "tanh":
        raise ValueError(f"unsupported tire model: {tire_model}")
    return peak * math.tanh(stiffness * alpha / max(peak, 1.0e-9))


def lateral_forces(u: float, v: float, r: float, delta: float,
                   parameters: PlantParameters) -> tuple[float, float]:
    """Return front/rear lateral force for one body-state sample."""
    safe_u = math.copysign(max(abs(u), MIN_SLIP_SPEED_MPS), u or 1.0)
    alpha_f = delta - math.atan2(v + parameters.lf_m * r, safe_u)
    alpha_r = -math.atan2(v - parameters.lr_m * r, safe_u)
    return (
        _tire_force(alpha_f, parameters.cf_n_per_rad, parameters.df_n,
                    parameters.tire_model),
        _tire_force(alpha_r, parameters.cr_n_per_rad, parameters.dr_n,
                    parameters.tire_model),
    )


def _body_derivative(u: float, v: float, r: float, delta: float,
                     wheel: float, parameters: PlantParameters,
                     include_longitudinal: bool = True) -> tuple[float, float, float]:
    front_force, rear_force = lateral_forces(u, v, r, delta, parameters)
    if include_longitudinal:
        longitudinal_force = (
            parameters.force_max_n * math.tanh(
                parameters.slip_gain_per_mps * (wheel - u)) -
            parameters.coast_speed_drag_n_per_mps * u)
        u_dot = (longitudinal_force - front_force * math.sin(delta)) / parameters.mass_kg + r * v
    else:
        u_dot = 0.0
    v_dot = ((front_force * math.cos(delta) + rear_force) /
             parameters.mass_kg - r * u)
    r_dot = ((parameters.lf_m * front_force * math.cos(delta) -
              parameters.lr_m * rear_force) / parameters.iz_kgm2)
    return u_dot, v_dot, r_dot


def lateral_body_step(u: float, v: float, r: float, delta_start: float,
                      delta_target: float, dt: float,
                      parameters: PlantParameters) -> tuple[float, float]:
    """Integrate only ``v,r`` with a continuously rate-limited steering ramp.

    Longitudinal speed is held at the measured value for lateral parameter
    fitting.  This keeps lateral fitting independent of an unaccepted
    longitudinal model; the complete plant uses :func:`step` below.
    """
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    delta_end = _move_towards(
        delta_start, delta_target,
        parameters.steering_rate_radps * dt)
    count = max(1, int(math.ceil(dt / INTEGRATION_SUBSTEP_S)))
    sub_dt = dt / count
    current_v, current_r = v, r
    for index in range(count):
        fraction = (index + 0.5) / count
        delta = delta_start + fraction * (delta_end - delta_start)
        _, v_dot, r_dot = _body_derivative(
            u, current_v, current_r, delta, 0.0, parameters,
            include_longitudinal=False)
        current_v += sub_dt * v_dot
        current_r += sub_dt * r_dot
    return current_v, current_r


def step(state: np.ndarray, steering_target_norm: float,
         throttle_norm: float, dt: float,
         parameters: PlantParameters) -> np.ndarray:
    """Advance ``[X,Y,yaw,u,v,r,delta,wheel]`` one source-time step."""
    if state.shape != (8,):
        raise ValueError(f"expected state shape (8,), got {state.shape}")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    steering_target = max(-1.0, min(1.0, steering_target_norm)) * parameters.max_steering_rad
    throttle = max(0.0, min(1.0, throttle_norm))
    x, y, yaw, u, v, r, delta, wheel = (float(value) for value in state)
    delta_end = _move_towards(
        delta, steering_target, parameters.steering_rate_radps * dt)
    wheel_end = _wheel_next(u, wheel, throttle, parameters)
    count = max(1, int(math.ceil(dt / INTEGRATION_SUBSTEP_S)))
    sub_dt = dt / count
    start_delta, start_wheel = delta, wheel
    for index in range(count):
        fraction = (index + 0.5) / count
        delta_mid = start_delta + fraction * (delta_end - start_delta)
        wheel_mid = start_wheel + fraction * (wheel_end - start_wheel)
        u_dot, v_dot, r_dot = _body_derivative(
            u, v, r, delta_mid, wheel_mid, parameters)
        x += sub_dt * (u * math.cos(yaw) - v * math.sin(yaw))
        y += sub_dt * (u * math.sin(yaw) + v * math.cos(yaw))
        yaw += sub_dt * r
        u = max(0.0, min(parameters.max_speed_mps, u + sub_dt * u_dot))
        v += sub_dt * v_dot
        r += sub_dt * r_dot
    return np.asarray([x, y, yaw, u, v, r, delta_end, wheel_end], dtype=float)


def lateral_candidate_parameters(kind: str, values: np.ndarray) -> dict[str, float | str]:
    if kind == "linear_saturated":
        return {
            "tire_model": kind,
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "df_n": 11.50,
            "dr_n": 10.60,
        }
    if kind == "tanh":
        return {
            "tire_model": kind,
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "df_n": float(values[2]),
            "dr_n": float(values[3]),
        }
    raise ValueError(f"unknown lateral candidate: {kind}")
