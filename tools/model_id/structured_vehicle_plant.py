"""Reference implementation of the identified physical vehicle plant.

This module is offline model-identification code.  It deliberately has no
ROS, simulator, or runtime-controller dependencies.  The same equations are
mirrored by ``f1tenth_mpc/src/vehicle_plant.c`` for native replay; production
MPC parameters are not changed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
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
IZ_KGM2 = 0.0961908
POSITION_OFFSET_FROM_VELOCITY_POINT_X_M = -0.155320086
MAX_STEERING_RAD = 0.5236
STEERING_RATE_RADPS = 3.2
STEERING_DYNAMICS_KINDS = ("rate_limited", "instantaneous", "first_order")
REGIME_SPEEDS_MPS = (5.0, 10.0, 14.0)
REGIME_TRANSITION_WIDTH_MPS = 2.0
WHEEL_RADIUS_M = 0.059
GRAVITY_MPS2 = 9.81
MAX_SPEED_MPS = 16.0
MIN_SLIP_SPEED_MPS = 0.5
INTEGRATION_SUBSTEP_S = 0.002
UNITY_RIGID_BODY_DRAG_PER_S = 0.273
UNITY_RIGID_BODY_ANGULAR_DRAG_PER_S = 0.1
DEFAULT_MANIFEST_PATH = (Path(__file__).resolve().parents[2] /
                         "f1tenth_mpc/config/vehicle_model_manifest_v1.json")


@dataclass(frozen=True)
class PlantParameters:
    """Physical parameters shared by Python fitting and native replay."""

    mass_kg: float = MASS_KG
    lf_m: float = LF_M
    lr_m: float = LR_M
    iz_kgm2: float = IZ_KGM2
    position_offset_from_velocity_point_x_m: float = POSITION_OFFSET_FROM_VELOCITY_POINT_X_M
    max_steering_rad: float = MAX_STEERING_RAD
    steering_rate_radps: float = STEERING_RATE_RADPS
    steering_dynamics_kind: str = "rate_limited"
    # The steering state is the effective wheel angle used by the tire model.
    # ``first_order`` is an offline-identification option; it is not enabled
    # by the production MPC profile unless a causal holdout accepts it.
    steering_lag_time_constant_s: float = 0.0
    max_speed_mps: float = MAX_SPEED_MPS
    regime_transition_width_mps: float = REGIME_TRANSITION_WIDTH_MPS
    linear_damping_per_s: float = UNITY_RIGID_BODY_DRAG_PER_S
    angular_damping_per_s: float = UNITY_RIGID_BODY_ANGULAR_DRAG_PER_S
    force_max_n: float = 19.893600647379536
    hard_brake_force_n: float = 18.575806999393215
    slip_gain_per_mps: float = 0.9715058770352094
    # Optional smooth low/mid/high speed profiles.  A scalar remains the
    # canonical fallback until a cross-regime fit is accepted offline.
    force_max_regimes_n: tuple[float, ...] | None = None
    slip_gain_regimes_per_mps: tuple[float, ...] | None = None
    # The measured Unity Rigidbody drag is represented explicitly above.
    # Keeping this fitted force term at zero avoids counting the same drag
    # twice.  A legacy fitted-drag profile remains available to replay tools.
    coast_speed_drag_n_per_mps: float = 0.0
    cf_n_per_rad: float = 2674.084021452388
    cr_n_per_rad: float = 4853.771529131131
    df_n: float = 12.63696205350674
    dr_n: float = 14.123309383979418
    lateral_cf_regimes_n_per_rad: tuple[float, ...] | None = None
    lateral_cr_regimes_n_per_rad: tuple[float, ...] | None = None
    lateral_df_regimes_n: tuple[float, ...] | None = None
    lateral_dr_regimes_n: tuple[float, ...] | None = None
    # These gains are zero for the current canonical tanh candidate.  The
    # non-zero values are reserved for an offline combined-slip experiment:
    # lateral force capacity/stiffness vary with forward speed and measured
    # longitudinal wheel slip.  They are deliberately not part of the native
    # runtime plant until a blind replay and parity gate accepts them.
    lateral_speed_stiffness_gain: float = 0.0
    lateral_speed_peak_gain: float = 0.0
    combined_slip_gain: float = 0.0
    # Optional empirical residual for the steering transition itself.  This
    # is an offline identification term, not a simulator or runtime-MPC
    # change.  It captures a repeatable force/moment impulse while the
    # effective steering state is moving; zero preserves the canonical plant.
    steering_rate_force_gain_n_per_radps: float = 0.0
    steering_rate_moment_gain_nm_per_radps: float = 0.0
    wheel_coefficients: tuple[float, ...] = (
        -0.055034041731618855, -38.44226600983389, 967.4757492136804,
        1.425533941018024, -37.16345280265934, 0.027050887976534637,
    )
    # The continuous candidate maps wheel-speed derivative and multiplies it
    # by the measured transition dt.  This is the current offline candidate;
    # it is not production-MPC approval.
    wheel_dynamics_kind: str = "continuous"
    tire_model: str = "tanh"

    @classmethod
    def from_manifest(cls, manifest_path: Path | None = None) -> "PlantParameters":
        """Resolve the canonical offline candidate from the model manifest.

        The manifest is the authority for the resolved replay plant. This
        keeps report-time fitting overrides and native replay defaults from
        quietly selecting different vehicle parameters. The production MPC
        profile is intentionally not reachable through this method.
        """
        if manifest_path is None:
            manifest_path = DEFAULT_MANIFEST_PATH
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        profile_name = manifest["canonical_candidate_profile"]
        profile = manifest["profiles"][profile_name]
        if profile.get("status") != "not_promoted":
            raise ValueError("canonical offline candidate must remain not_promoted")
        contract = profile["model_contract"]
        if contract["state_order"] != [
                "x_m", "y_m", "yaw_rad", "u_mps", "v_mps", "r_radps",
                "steering_rad", "wheel_speed_mps"]:
            raise ValueError("manifest candidate state order is not the plant state")
        if contract["input_order"] != ["steering_target_norm", "throttle_norm"]:
            raise ValueError("manifest candidate input order is not the plant input")
        values = profile["parameters"]
        pose_reference = profile["pose_reference"]
        return cls(
            mass_kg=float(values["mass_kg"]),
            lf_m=float(values["lf_m"]),
            lr_m=float(values["lr_m"]),
            iz_kgm2=float(values["iz_kgm2"]),
            position_offset_from_velocity_point_x_m=float(
                pose_reference["position_offset_from_velocity_point_x_m"]),
            max_steering_rad=float(values["max_steering_rad"]),
            steering_rate_radps=float(values["steering_rate_radps"]),
            steering_dynamics_kind=str(values.get(
                "steering_dynamics_kind", "rate_limited")),
            steering_lag_time_constant_s=float(values.get(
                "steering_lag_time_constant_s", 0.0)),
            max_speed_mps=float(values["max_speed_mps"]),
            regime_transition_width_mps=float(values.get(
                "regime_transition_width_mps", REGIME_TRANSITION_WIDTH_MPS)),
            linear_damping_per_s=float(values["linear_damping_per_s"]),
            angular_damping_per_s=float(values["angular_damping_per_s"]),
            force_max_n=float(values["force_max_n"]),
            hard_brake_force_n=float(values["hard_brake_force_n"]),
            slip_gain_per_mps=float(values["slip_gain_per_mps"]),
            force_max_regimes_n=tuple(float(value) for value in values[
                "force_max_regimes_n"]) if values.get(
                    "force_max_regimes_n") is not None else None,
            slip_gain_regimes_per_mps=tuple(float(value) for value in values[
                "slip_gain_regimes_per_mps"]) if values.get(
                    "slip_gain_regimes_per_mps") is not None else None,
            coast_speed_drag_n_per_mps=float(
                values["coast_speed_drag_n_per_mps"]),
            cf_n_per_rad=float(values["cf_n_per_rad"]),
            cr_n_per_rad=float(values["cr_n_per_rad"]),
            df_n=float(values["df_n"]),
            dr_n=float(values["dr_n"]),
            lateral_cf_regimes_n_per_rad=tuple(float(value) for value in values[
                "lateral_cf_regimes_n_per_rad"]) if values.get(
                    "lateral_cf_regimes_n_per_rad") is not None else None,
            lateral_cr_regimes_n_per_rad=tuple(float(value) for value in values[
                "lateral_cr_regimes_n_per_rad"]) if values.get(
                    "lateral_cr_regimes_n_per_rad") is not None else None,
            lateral_df_regimes_n=tuple(float(value) for value in values[
                "lateral_df_regimes_n"]) if values.get(
                    "lateral_df_regimes_n") is not None else None,
            lateral_dr_regimes_n=tuple(float(value) for value in values[
                "lateral_dr_regimes_n"]) if values.get(
                    "lateral_dr_regimes_n") is not None else None,
            lateral_speed_stiffness_gain=float(values.get(
                "lateral_speed_stiffness_gain", 0.0)),
            lateral_speed_peak_gain=float(values.get(
                "lateral_speed_peak_gain", 0.0)),
            combined_slip_gain=float(values.get("combined_slip_gain", 0.0)),
            steering_rate_force_gain_n_per_radps=float(values.get(
                "steering_rate_force_gain_n_per_radps", 0.0)),
            steering_rate_moment_gain_nm_per_radps=float(values.get(
                "steering_rate_moment_gain_nm_per_radps", 0.0)),
            wheel_coefficients=tuple(float(value)
                                     for value in values["wheel_coefficients"]),
            wheel_dynamics_kind=("continuous"
                                 if values["wheel_dynamics"] == "continuous"
                                 else "discrete"),
            tire_model=str(values["tire_model"]),
        )

    @classmethod
    def from_longitudinal_parameters(
            cls, longitudinal: dict[str, Any], **overrides: Any
    ) -> "PlantParameters":
        dynamics = longitudinal.get("wheel_dynamics", {})
        values = {
            "force_max_n": float(longitudinal.get("force_max_n", cls.force_max_n)),
            "hard_brake_force_n": float(longitudinal.get(
                "hard_brake_force_n", cls.hard_brake_force_n)),
            "slip_gain_per_mps": float(longitudinal.get(
                "slip_gain_per_mps", cls.slip_gain_per_mps)),
            "coast_speed_drag_n_per_mps": float(longitudinal.get(
                "coast_speed_drag_n_per_mps", cls.coast_speed_drag_n_per_mps)),
        }
        if longitudinal.get("force_max_regimes_n") is not None:
            values["force_max_regimes_n"] = tuple(float(value) for value in
                                                   longitudinal["force_max_regimes_n"])
        if longitudinal.get("slip_gain_regimes_per_mps") is not None:
            values["slip_gain_regimes_per_mps"] = tuple(
                float(value) for value in longitudinal["slip_gain_regimes_per_mps"])
        if longitudinal.get("regime_transition_width_mps") is not None:
            values["regime_transition_width_mps"] = float(
                longitudinal["regime_transition_width_mps"])
        if dynamics.get("coefficients"):
            values["wheel_coefficients"] = tuple(
                float(value) for value in dynamics["coefficients"])
        dynamics_kind = dynamics.get("kind")
        if dynamics_kind == "identified_continuous_wheel_speed_derivative":
            values["wheel_dynamics_kind"] = "continuous"
        elif dynamics_kind == "identified_discrete_wheel_speed_state":
            # The discrete fit predicts wheel[k+1] directly.  It must not be
            # integrated as a derivative; doing so adds roughly one full
            # wheel-speed state per second and destabilizes recursive pose
            # prediction.
            values["wheel_dynamics_kind"] = "discrete"
        elif dynamics_kind is not None:
            raise ValueError(
                "unsupported identified wheel dynamics kind: "
                f"{dynamics_kind}")
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
            lateral_speed_stiffness_gain=float(lateral.get(
                "lateral_speed_stiffness_gain", self.lateral_speed_stiffness_gain)),
            lateral_speed_peak_gain=float(lateral.get(
                "lateral_speed_peak_gain", self.lateral_speed_peak_gain)),
            combined_slip_gain=float(lateral.get(
                "combined_slip_gain", self.combined_slip_gain)),
            steering_rate_force_gain_n_per_radps=float(lateral.get(
                "steering_rate_force_gain_n_per_radps",
                self.steering_rate_force_gain_n_per_radps)),
            steering_rate_moment_gain_nm_per_radps=float(lateral.get(
                "steering_rate_moment_gain_nm_per_radps",
                self.steering_rate_moment_gain_nm_per_radps)),
            lateral_cf_regimes_n_per_rad=(tuple(float(value) for value in lateral[
                "lateral_cf_regimes_n_per_rad"]) if lateral.get(
                    "lateral_cf_regimes_n_per_rad") is not None else
                self.lateral_cf_regimes_n_per_rad),
            lateral_cr_regimes_n_per_rad=(tuple(float(value) for value in lateral[
                "lateral_cr_regimes_n_per_rad"]) if lateral.get(
                    "lateral_cr_regimes_n_per_rad") is not None else
                self.lateral_cr_regimes_n_per_rad),
            lateral_df_regimes_n=(tuple(float(value) for value in lateral[
                "lateral_df_regimes_n"]) if lateral.get(
                    "lateral_df_regimes_n") is not None else
                self.lateral_df_regimes_n),
            lateral_dr_regimes_n=(tuple(float(value) for value in lateral[
                "lateral_dr_regimes_n"]) if lateral.get(
                    "lateral_dr_regimes_n") is not None else
                self.lateral_dr_regimes_n),
            tire_model=str(lateral.get("tire_model", self.tire_model)),
        )


def _move_towards(current: float, target: float, maximum_delta: float) -> float:
    difference = target - current
    return current + max(-maximum_delta, min(maximum_delta, difference))


def _steering_next(current: float, target: float, dt: float,
                   parameters: PlantParameters) -> float:
    """Advance the identified steering state using an explicit contract."""
    if parameters.steering_dynamics_kind == "instantaneous":
        return target
    if parameters.steering_dynamics_kind == "rate_limited":
        return _move_towards(
            current, target, parameters.steering_rate_radps * dt)
    if parameters.steering_dynamics_kind == "first_order":
        if (not math.isfinite(parameters.steering_lag_time_constant_s) or
                parameters.steering_lag_time_constant_s <= 0.0):
            raise ValueError(
                "first_order steering requires a positive lag time constant")
        alpha = 1.0 - math.exp(
            -dt / parameters.steering_lag_time_constant_s)
        return current + alpha * (target - current)
    raise ValueError(
        "unsupported steering dynamics: "
        f"{parameters.steering_dynamics_kind}")


def _speed_regime_value(values: tuple[float, ...] | None, speed_mps: float,
                        transition_width_mps: float) -> float | None:
    """Blend low/mid/high values without a discontinuity at regime edges."""
    if values is None:
        return None
    if len(values) != len(REGIME_SPEEDS_MPS):
        raise ValueError(
            f"expected {len(REGIME_SPEEDS_MPS)} regime values, got {len(values)}")
    width = max(float(transition_width_mps), 1.0e-6)
    speed = abs(float(speed_mps))

    def smoothstep(value: float) -> float:
        clipped = max(0.0, min(1.0, value))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    # Full low regime through 7 m/s, smooth low/mid blend over 7--9 m/s,
    # full mid through 11 m/s, smooth mid/high blend over 11--13 m/s.
    half_width = width * 0.5
    low_mid = smoothstep((speed - (REGIME_SPEEDS_MPS[1] - half_width)) /
                         width)
    mid_high = smoothstep((speed - (REGIME_SPEEDS_MPS[2] - half_width)) /
                          width)
    low = 1.0 - low_mid
    high = mid_high
    mid = max(0.0, 1.0 - low - high)
    return float(low * values[0] + mid * values[1] + high * values[2])


def _wheel_next(body_u: float, wheel: float, throttle: float, dt: float,
                parameters: PlantParameters) -> float:
    coeff = np.asarray(parameters.wheel_coefficients, dtype=float)
    features = np.asarray([
        1.0, wheel, throttle, wheel * throttle,
        throttle * throttle, body_u,
    ])
    wheel_prediction = float(features @ coeff)
    if parameters.wheel_dynamics_kind == "continuous":
        wheel_prediction = wheel + dt * wheel_prediction
    elif parameters.wheel_dynamics_kind != "discrete":
        raise ValueError(
            f"unsupported wheel dynamics: {parameters.wheel_dynamics_kind}")
    return max(0.0, wheel_prediction)


def _tire_force(alpha: float, stiffness: float, peak: float,
                tire_model: str, speed_mps: float, wheel_mps: float,
                parameters: PlantParameters) -> float:
    speed_ratio = min(1.0, abs(speed_mps) / max(parameters.max_speed_mps, 1.0e-9))
    speed_stiffness = max(
        0.2, 1.0 + parameters.lateral_speed_stiffness_gain * speed_ratio)
    speed_peak = max(
        0.2, 1.0 + parameters.lateral_speed_peak_gain * speed_ratio)
    safe_speed = max(abs(speed_mps), MIN_SLIP_SPEED_MPS)
    longitudinal_slip = (wheel_mps - speed_mps) / safe_speed
    combined_scale = 1.0 / math.sqrt(
        1.0 + (parameters.combined_slip_gain * longitudinal_slip) ** 2)
    stiffness *= speed_stiffness * combined_scale
    peak *= speed_peak * combined_scale
    if tire_model == "linear_saturated":
        return max(-peak, min(peak, stiffness * alpha))
    if tire_model not in ("tanh", "speed_combined_tanh",
                          "regime_speed_combined_tanh"):
        raise ValueError(f"unsupported tire model: {tire_model}")
    return peak * math.tanh(stiffness * alpha / max(peak, 1.0e-9))


def lateral_forces(u: float, v: float, r: float, delta: float,
                   parameters: PlantParameters, wheel: float = 0.0,
                   throttle: float = 0.0) -> tuple[float, float]:
    """Return front/rear lateral force for one body-state sample."""
    safe_u = math.copysign(max(abs(u), MIN_SLIP_SPEED_MPS), u or 1.0)
    alpha_f = delta - math.atan2(v + parameters.lf_m * r, safe_u)
    alpha_r = -math.atan2(v - parameters.lr_m * r, safe_u)
    cf = _speed_regime_value(
        parameters.lateral_cf_regimes_n_per_rad, u,
        parameters.regime_transition_width_mps)
    cr = _speed_regime_value(
        parameters.lateral_cr_regimes_n_per_rad, u,
        parameters.regime_transition_width_mps)
    df = _speed_regime_value(
        parameters.lateral_df_regimes_n, u,
        parameters.regime_transition_width_mps)
    dr = _speed_regime_value(
        parameters.lateral_dr_regimes_n, u,
        parameters.regime_transition_width_mps)
    return (
        _tire_force(alpha_f, cf if cf is not None else parameters.cf_n_per_rad,
                    df if df is not None else parameters.df_n,
                    parameters.tire_model, u, wheel, parameters),
        _tire_force(alpha_r, cr if cr is not None else parameters.cr_n_per_rad,
                    dr if dr is not None else parameters.dr_n,
                    parameters.tire_model, u, wheel, parameters),
    )


def _body_derivative(u: float, v: float, r: float, delta: float,
                     wheel: float, parameters: PlantParameters,
                     throttle: float = 1.0,
                     include_longitudinal: bool = True,
                     steering_rate_radps: float = 0.0) -> tuple[float, float, float]:
    front_force, rear_force = lateral_forces(
        u, v, r, delta, parameters, wheel=wheel, throttle=throttle)
    if include_longitudinal:
        if throttle > 1.0e-5:
            force_max = _speed_regime_value(
                parameters.force_max_regimes_n, u,
                parameters.regime_transition_width_mps)
            slip_gain = _speed_regime_value(
                parameters.slip_gain_regimes_per_mps, u,
                parameters.regime_transition_width_mps)
            longitudinal_force = (
                (force_max if force_max is not None else parameters.force_max_n) *
                math.tanh((slip_gain if slip_gain is not None else
                           parameters.slip_gain_per_mps) * (wheel - u)) -
                parameters.coast_speed_drag_n_per_mps * u)
        else:
            # Zero throttle is an active all-wheel brake command in the
            # simulator, not a passive-coast input.
            longitudinal_force = (
                -max(0.0, parameters.hard_brake_force_n) -
                parameters.coast_speed_drag_n_per_mps * u)
        u_dot = (longitudinal_force - front_force * math.sin(delta)) / parameters.mass_kg + r * v
    else:
        u_dot = 0.0
    v_dot = ((front_force * math.cos(delta) + rear_force) /
             parameters.mass_kg - r * u)
    r_dot = ((parameters.lf_m * front_force * math.cos(delta) -
              parameters.lr_m * rear_force) / parameters.iz_kgm2 -
             parameters.angular_damping_per_s * r)
    v_dot += (parameters.steering_rate_force_gain_n_per_radps *
              steering_rate_radps / parameters.mass_kg)
    r_dot += (parameters.steering_rate_moment_gain_nm_per_radps *
              steering_rate_radps / parameters.iz_kgm2)
    u_dot -= parameters.linear_damping_per_s * u
    v_dot -= parameters.linear_damping_per_s * v
    return u_dot, v_dot, r_dot


def lateral_body_step(u: float, v: float, r: float, delta_start: float,
                      delta_target: float, dt: float,
                      parameters: PlantParameters, wheel: float = 0.0,
                      throttle: float = 0.0) -> tuple[float, float]:
    """Integrate only ``v,r`` with the selected steering transition model.

    Longitudinal speed is held at the measured value for lateral parameter
    fitting.  This keeps lateral fitting independent of an unaccepted
    longitudinal model; the complete plant uses :func:`step` below.
    """
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    delta_end = _steering_next(
        delta_start, delta_target, dt, parameters)
    count = max(1, int(math.ceil(dt / INTEGRATION_SUBSTEP_S)))
    sub_dt = dt / count
    current_v, current_r = v, r
    for index in range(count):
        fraction = (index + 0.5) / count
        delta = delta_start + fraction * (delta_end - delta_start)
        _, v_dot, r_dot = _body_derivative(
            u, current_v, current_r, delta, wheel, parameters,
            throttle=throttle, include_longitudinal=False,
            steering_rate_radps=(delta_end - delta_start) / dt)
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
    delta_end = _steering_next(delta, steering_target, dt, parameters)
    wheel_end = _wheel_next(u, wheel, throttle, dt, parameters)
    count = max(1, int(math.ceil(dt / INTEGRATION_SUBSTEP_S)))
    sub_dt = dt / count
    start_delta, start_wheel = delta, wheel
    for index in range(count):
        fraction = (index + 0.5) / count
        delta_mid = start_delta + fraction * (delta_end - start_delta)
        wheel_mid = start_wheel + fraction * (wheel_end - start_wheel)
        u_dot, v_dot, r_dot = _body_derivative(
            u, v, r, delta_mid, wheel_mid, parameters,
            throttle=throttle,
            steering_rate_radps=(delta_end - delta) / dt)
        pose_v = v + parameters.position_offset_from_velocity_point_x_m * r
        x += sub_dt * (u * math.cos(yaw) - pose_v * math.sin(yaw))
        y += sub_dt * (u * math.sin(yaw) + pose_v * math.cos(yaw))
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
    if kind == "speed_combined_tanh":
        return {
            "tire_model": kind,
            "cf_n_per_rad": float(values[0]),
            "cr_n_per_rad": float(values[1]),
            "df_n": float(values[2]),
            "dr_n": float(values[3]),
            "lateral_speed_stiffness_gain": float(values[4]),
            "lateral_speed_peak_gain": float(values[5]),
            "combined_slip_gain": float(values[6]),
        }
    raise ValueError(f"unknown lateral candidate: {kind}")
