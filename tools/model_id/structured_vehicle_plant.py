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
# Direct Unity F1TENTH WheelCollider anchors.  These are simulator
# parameters, not identified real-car tire coefficients.  The active RoboRacer
# scene uses the prefab's WheelCollider geometry/friction curves and overrides
# the controller motor torque; the longitudinal torque path remains separate
# until its WheelCollider rotational behavior is identified.
UNITY_ACKERMANN_WHEELBASE_M = 0.324
UNITY_TRACK_WIDTH_M = 0.236
UNITY_COM_HEIGHT_M = 0.06434
# WheelCollider locations expressed in the plant frame (x forward, y left)
# after applying the serialized Rigidbody center-of-mass offset.  These are
# deliberately separate from the VehicleController Ackermann wheelbase:
# Unity uses 324 mm in the steering equations, while the collider locations
# in the prefab are 330 mm apart around the COM.
UNITY_FRONT_WHEEL_X_M = 0.174679914
UNITY_REAR_WHEEL_X_M = -0.155320086
# WheelCollider.sprungMass is calculated by Unity at runtime; it is not
# serialized in the prefab. These values are now confirmed by the exact
# competition-scene diagnostic capture and remain structural references, not
# fitted tire/load-transfer parameters.
UNITY_WHEEL_SPRUNG_MASSES_KG = (
    0.8171949982643127,
    0.8160193562507629,
    0.9189916849136353,
    0.9177939891815186,
)
UNITY_WHEEL_MASS_KG = 0.109
UNITY_WHEEL_DAMPING_RATE = 0.25
UNITY_SIDEWAYS_EXTREMUM_SLIP = 0.01
UNITY_SIDEWAYS_EXTREMUM_VALUE = 1.0
UNITY_SIDEWAYS_ASYMPTOTE_SLIP = 0.1
UNITY_SIDEWAYS_ASYMPTOTE_VALUE = 0.5
UNITY_SIDEWAYS_STIFFNESS = 1.0
UNITY_FORWARD_EXTREMUM_SLIP = 0.15
UNITY_FORWARD_EXTREMUM_VALUE = 0.9
UNITY_FORWARD_ASYMPTOTE_SLIP = 0.25
UNITY_FORWARD_ASYMPTOTE_VALUE = 0.58
UNITY_FORWARD_STIFFNESS = 0.8
UNITY_COMPETITION_MOTOR_TORQUE_NM = 428.0
UNITY_COMPETITION_BRAKE_TORQUE_NM = 428.0
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
    # The direct Unity screen can select the serialized forward curve while
    # retaining the identified wheel-state transition. The default preserves
    # the canonical offline candidate until the explicit wheel rotational
    # model passes its own holdout.
    unity_longitudinal_model: str = "identified_force"
    # ``unity_wheel_collider`` uses the direct serialized sideways curve and
    # per-wheel sprung masses above.  It intentionally has no fitted peak,
    # cornering stiffness, friction coefficient, or speed gain.
    unity_ackermann_wheelbase_m: float = UNITY_ACKERMANN_WHEELBASE_M
    unity_track_width_m: float = UNITY_TRACK_WIDTH_M
    unity_wheel_sprung_masses_kg: tuple[float, ...] = UNITY_WHEEL_SPRUNG_MASSES_KG
    unity_normal_load_mode: str = "static_sprung_mass"
    unity_sideways_slip_scale: float = 1.0
    # Offline-only load-distribution screen.  These coefficients describe the
    # measured WheelHit contact-load proxy, not a tire parameter.  Defaults
    # remain zero because the eight-state plant has no causal suspension/load
    # states and the coefficients require a fresh exact-scene confirmation.
    unity_front_rear_load_transfer_bias_n: float = 0.0
    unity_front_rear_load_transfer_gain_n_per_mps2: float = 0.0
    unity_left_right_load_transfer_bias_n: float = 0.0
    unity_left_right_load_transfer_gain_n_per_mps2: float = 0.0

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


def unity_forward_slip(wheel_surface_speed_mps: float,
                       wheel_longitudinal_speed_mps: float) -> float:
    """Return the simulator's signed forward-slip coordinate.

    The exact open-scene trace shows that Unity normalizes the wheel/ground
    speed difference by the larger speed magnitude, not by ground speed alone:
    ``(wheel_speed-ground_speed) / max(abs(wheel_speed), abs(ground_speed))``.
    The lower bound only prevents a zero-speed singularity and the final clip
    matches the observed full-brake saturation at ``-1``. This is a simulator
    coordinate inferred from the trace, not a real-tire slip-ratio definition.
    """
    values = (wheel_surface_speed_mps, wheel_longitudinal_speed_mps)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Unity forward-slip inputs must be finite")
    denominator = max(abs(wheel_surface_speed_mps),
                      abs(wheel_longitudinal_speed_mps),
                      MIN_SLIP_SPEED_MPS)
    slip = ((wheel_surface_speed_mps - wheel_longitudinal_speed_mps) /
            denominator)
    return max(-1.0, min(1.0, slip))


def unity_forward_wheel_force(normal_load_n: float,
                              wheel_surface_speed_mps: float,
                              wheel_longitudinal_speed_mps: float) -> float:
    """Evaluate one Unity forward WheelFrictionCurve force.

    ``normal_load_n`` is intentionally an explicit input. The function does
    not replace dynamic WheelHit load with a fitted force capacity; callers
    must supply either a measured load state or a separately validated load
    model.
    """
    if not math.isfinite(normal_load_n) or normal_load_n < 0.0:
        raise ValueError("Unity normal load must be finite and non-negative")
    slip = unity_forward_slip(wheel_surface_speed_mps,
                              wheel_longitudinal_speed_mps)
    return normal_load_n * unity_wheel_friction_value(
        slip, UNITY_FORWARD_EXTREMUM_SLIP, UNITY_FORWARD_EXTREMUM_VALUE,
        UNITY_FORWARD_ASYMPTOTE_SLIP, UNITY_FORWARD_ASYMPTOTE_VALUE,
        UNITY_FORWARD_STIFFNESS)


def unity_cawd_motor_torque_per_wheel(throttle_norm: float) -> float:
    """Return the active RoboRacer CAWD motor torque for one wheel."""
    if not math.isfinite(throttle_norm):
        raise ValueError("throttle must be finite")
    throttle = max(0.0, min(1.0, throttle_norm))
    return UNITY_COMPETITION_MOTOR_TORQUE_NM * throttle / 4.0


def unity_competition_brake_torque_per_wheel(throttle_norm: float) -> float:
    """Return the active CAWB brake torque for a zero-throttle command."""
    if not math.isfinite(throttle_norm):
        raise ValueError("throttle must be finite")
    return (UNITY_COMPETITION_BRAKE_TORQUE_NM
            if abs(throttle_norm) <= 1.0e-5 else 0.0)


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


def unity_wheel_friction_value(
        slip: float, extremum_slip: float, extremum_value: float,
        asymptote_slip: float, asymptote_value: float,
        stiffness: float) -> float:
    """Evaluate Unity's serialized piecewise WheelFrictionCurve.

    This is the simulator's slip law, not a physical tire-law assumption.
    Unity's curve rises linearly from the origin to the extremum, then
    linearly to the asymptote and remains constant beyond it.  The sign is
    retained so the caller can use the result as a signed body force.
    """
    values = (slip, extremum_slip, extremum_value, asymptote_slip,
              asymptote_value, stiffness)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Unity friction curve inputs must be finite")
    if extremum_slip <= 0.0 or asymptote_slip <= extremum_slip:
        raise ValueError("Unity friction slip breakpoints must be ordered")
    if stiffness < 0.0:
        raise ValueError("Unity friction stiffness must be non-negative")
    magnitude = abs(slip)
    if magnitude <= extremum_slip:
        value = extremum_value * magnitude / extremum_slip
    elif magnitude <= asymptote_slip:
        fraction = ((magnitude - extremum_slip) /
                    (asymptote_slip - extremum_slip))
        value = extremum_value + fraction * (asymptote_value - extremum_value)
    else:
        value = asymptote_value
    return math.copysign(stiffness * value, slip) if slip != 0.0 else 0.0


def unity_sideways_friction_value(slip: float) -> float:
    """Evaluate the current F1TENTH prefab sideways curve."""
    return unity_wheel_friction_value(
        slip, UNITY_SIDEWAYS_EXTREMUM_SLIP,
        UNITY_SIDEWAYS_EXTREMUM_VALUE, UNITY_SIDEWAYS_ASYMPTOTE_SLIP,
        UNITY_SIDEWAYS_ASYMPTOTE_VALUE, UNITY_SIDEWAYS_STIFFNESS)


def _unity_ackermann_angles(delta: float,
                            parameters: PlantParameters) -> tuple[float, float]:
    """Return model-positive left/right angles from VehicleController.Steer.

    ``VehicleController`` stores ``SteeringAngle`` with the opposite sign to
    its published ``AppliedSteering`` value.  Therefore the model's positive
    delta maps to the controller's negative internal angle, which swaps the
    two denominators when expressed as positive model wheel angles.
    """
    wheelbase = parameters.unity_ackermann_wheelbase_m
    track = parameters.unity_track_width_m
    tangent = math.tan(delta)
    denominator_left = 2.0 * wheelbase - track * tangent
    denominator_right = 2.0 * wheelbase + track * tangent
    if (abs(denominator_left) <= 1.0e-9 or
            abs(denominator_right) <= 1.0e-9):
        raise ValueError("Ackermann denominator is singular")
    return (
        math.atan((2.0 * wheelbase * tangent) / denominator_left),
        math.atan((2.0 * wheelbase * tangent) / denominator_right),
    )


def _unity_wheel_lateral_wrench(
        u: float, v: float, r: float, delta: float,
        parameters: PlantParameters,
        longitudinal_accel_mps2: float = 0.0,
        lateral_accel_mps2: float | None = None) -> tuple[float, float, float]:
    """Return body-frame Fx, Fy and yaw moment from four Unity wheel curves.

    The model uses the measured planar wheel locations from the F1TENTH
    prefab. Static normal loads use the exact competition-scene runtime
    ``WheelCollider.sprungMass`` values multiplied by Unity gravity.
    ``sprungMass`` is calculated by Unity and is not serialized in the prefab;
    the captured values are therefore tied to this exact scene/configuration.
    No load-transfer term is hidden in a tire peak; dynamic load transfer is a
    separate identification problem because the current eight-state model has
    no suspension states.
    """
    if parameters.unity_normal_load_mode not in {
            "static_sprung_mass", "causal_linear_transfer",
            "mechanical_cg_transfer"}:
        raise ValueError(
            "unsupported Unity normal-load mode: "
            f"{parameters.unity_normal_load_mode}")
    sprung_masses = parameters.unity_wheel_sprung_masses_kg
    if len(sprung_masses) != 4:
        raise ValueError("Unity model requires four wheel sprung masses")
    if (not math.isfinite(parameters.unity_sideways_slip_scale) or
            parameters.unity_sideways_slip_scale <= 0.0):
        raise ValueError("Unity sideways slip scale must be positive")
    left_angle, right_angle = _unity_ackermann_angles(delta, parameters)
    # Model convention: x forward, y left; Unity prefab x is lateral and z is
    # longitudinal.  Wheel order follows ModelIdentificationDiagnostics:
    # front-left, front-right, rear-left, rear-right.
    if lateral_accel_mps2 is None:
        # The algebraic screen uses the centripetal component available from
        # the eight-state body state.  A suspension-state implementation must
        # replace this approximation before runtime promotion.
        lateral_accel_mps2 = u * r
    front_rear_transfer = 0.0
    left_right_transfer = 0.0
    if parameters.unity_normal_load_mode == "causal_linear_transfer":
        front_rear_transfer = (
            parameters.unity_front_rear_load_transfer_bias_n +
            parameters.unity_front_rear_load_transfer_gain_n_per_mps2 *
            longitudinal_accel_mps2)
        left_right_transfer = (
            parameters.unity_left_right_load_transfer_bias_n +
            parameters.unity_left_right_load_transfer_gain_n_per_mps2 *
            lateral_accel_mps2)
    elif parameters.unity_normal_load_mode == "mechanical_cg_transfer":
        # Exact-geometry diagnostic screen. Positive body acceleration shifts
        # load rearward/rightward in the x-forward/y-left convention, so the
        # signed front-minus-rear and left-minus-right transfers are negative.
        # This uses only Unity's measured mass, COM height, contact geometry,
        # and configured wheel track; no fitted load or tire coefficient is
        # introduced. It remains offline until blind trace parity accepts it.
        geometric_wheelbase = UNITY_FRONT_WHEEL_X_M - UNITY_REAR_WHEEL_X_M
        front_rear_transfer = (
            -parameters.mass_kg * UNITY_COM_HEIGHT_M /
            geometric_wheelbase * longitudinal_accel_mps2)
        left_right_transfer = (
            -parameters.mass_kg * UNITY_COM_HEIGHT_M /
            parameters.unity_track_width_m * lateral_accel_mps2)
    # The transfer definition matches the diagnostic fit: add front/rear
    # transfer to each front wheel and subtract it from each rear wheel; add
    # left/right transfer to each left wheel and subtract it from each right
    # wheel.  The total supported load remains conserved.
    wheel_data = (
        (UNITY_FRONT_WHEEL_X_M, +parameters.unity_track_width_m * 0.5,
         left_angle, sprung_masses[0] +
         front_rear_transfer / (4.0 * GRAVITY_MPS2) +
         left_right_transfer / (4.0 * GRAVITY_MPS2)),
        (UNITY_FRONT_WHEEL_X_M, -parameters.unity_track_width_m * 0.5,
         right_angle, sprung_masses[1] +
         front_rear_transfer / (4.0 * GRAVITY_MPS2) -
         left_right_transfer / (4.0 * GRAVITY_MPS2)),
        (UNITY_REAR_WHEEL_X_M, +parameters.unity_track_width_m * 0.5,
         0.0, sprung_masses[2] -
         front_rear_transfer / (4.0 * GRAVITY_MPS2) +
         left_right_transfer / (4.0 * GRAVITY_MPS2)),
        (UNITY_REAR_WHEEL_X_M, -parameters.unity_track_width_m * 0.5,
         0.0, sprung_masses[3] -
         front_rear_transfer / (4.0 * GRAVITY_MPS2) -
         left_right_transfer / (4.0 * GRAVITY_MPS2)),
    )
    body_fx = 0.0
    body_fy = 0.0
    yaw_moment = 0.0
    for x_position, y_position, wheel_angle, sprung_mass in wheel_data:
        wheel_vx = u - r * y_position
        wheel_vy = v + r * x_position
        cosine = math.cos(wheel_angle)
        sine = math.sin(wheel_angle)
        wheel_forward = wheel_vx * cosine + wheel_vy * sine
        wheel_sideways = -wheel_vx * sine + wheel_vy * cosine
        # Unity's WheelHit sidewaysSlip is represented as the signed lateral
        # velocity ratio in the wheel frame.  At very low speed the ratio is
        # numerically undefined; the guard only prevents a singular offline
        # prediction and is outside the raceline fitting envelope.
        slip = (parameters.unity_sideways_slip_scale * -wheel_sideways /
                max(abs(wheel_forward), MIN_SLIP_SPEED_MPS))
        wheel_force = (sprung_mass * GRAVITY_MPS2 *
                       unity_sideways_friction_value(slip))
        force_x = -wheel_force * sine
        force_y = wheel_force * cosine
        body_fx += force_x
        body_fy += force_y
        yaw_moment += x_position * force_y - y_position * force_x
    return body_fx, body_fy, yaw_moment


def _unity_wheel_longitudinal_force(
        u: float, v: float, r: float, wheel_surface_speed: float,
        delta: float, parameters: PlantParameters) -> float:
    """Return total longitudinal force from the serialized Unity curve.

    This is an explicit four-wheel screen. It uses the current wheel surface
    speed state and exact static runtime loads; it does not add a fitted force
    capacity. Dynamic normal load and wheel rotational dynamics remain separate
    states and are not silently folded into this force.
    """
    left_angle, right_angle = _unity_ackermann_angles(delta, parameters)
    wheel_data = (
        (UNITY_FRONT_WHEEL_X_M, +parameters.unity_track_width_m * 0.5,
         left_angle, parameters.unity_wheel_sprung_masses_kg[0]),
        (UNITY_FRONT_WHEEL_X_M, -parameters.unity_track_width_m * 0.5,
         right_angle, parameters.unity_wheel_sprung_masses_kg[1]),
        (UNITY_REAR_WHEEL_X_M, +parameters.unity_track_width_m * 0.5,
         0.0, parameters.unity_wheel_sprung_masses_kg[2]),
        (UNITY_REAR_WHEEL_X_M, -parameters.unity_track_width_m * 0.5,
         0.0, parameters.unity_wheel_sprung_masses_kg[3]),
    )
    total_force = 0.0
    for x_position, y_position, wheel_angle, sprung_mass in wheel_data:
        wheel_vx = u - r * y_position
        wheel_vy = v + r * x_position
        wheel_longitudinal_speed = (
            wheel_vx * math.cos(wheel_angle) +
            wheel_vy * math.sin(wheel_angle))
        total_force += unity_forward_wheel_force(
            sprung_mass * GRAVITY_MPS2, wheel_surface_speed,
            wheel_longitudinal_speed)
    return total_force


def lateral_forces(u: float, v: float, r: float, delta: float,
                   parameters: PlantParameters, wheel: float = 0.0,
                   throttle: float = 0.0) -> tuple[float, float]:
    """Return front/rear lateral force for one body-state sample."""
    if parameters.tire_model == "unity_wheel_collider":
        _, body_fy, _ = _unity_wheel_lateral_wrench(
            u, v, r, delta, parameters)
        # Preserve this function's historical axle-force API for callers that
        # only need total lateral force.  The exact yaw moment is retained in
        # _body_derivative through the full wheel wrench.
        return body_fy * 0.5, body_fy * 0.5
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
    if parameters.tire_model == "unity_wheel_collider":
        wheel_lateral_fx, wheel_lateral_fy, wheel_yaw_moment = (
            _unity_wheel_lateral_wrench(u, v, r, delta, parameters))
        longitudinal_force = 0.0
        if include_longitudinal:
            if parameters.unity_longitudinal_model == "unity_forward_curve":
                longitudinal_force = _unity_wheel_longitudinal_force(
                    u, v, r, wheel, delta, parameters)
            elif parameters.unity_longitudinal_model == "identified_force":
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
                    longitudinal_force = (
                        -max(0.0, parameters.hard_brake_force_n) -
                        parameters.coast_speed_drag_n_per_mps * u)
            else:
                raise ValueError(
                    "unsupported Unity longitudinal model: "
                    f"{parameters.unity_longitudinal_model}")
        if parameters.unity_normal_load_mode == "causal_linear_transfer":
            longitudinal_accel = (longitudinal_force / parameters.mass_kg + r * v -
                                  parameters.linear_damping_per_s * u)
            wheel_lateral_fx, wheel_lateral_fy, wheel_yaw_moment = (
                _unity_wheel_lateral_wrench(
                    u, v, r, delta, parameters,
                    longitudinal_accel_mps2=longitudinal_accel,
                    lateral_accel_mps2=u * r))
        u_dot = (longitudinal_force + wheel_lateral_fx) / parameters.mass_kg + r * v
        v_dot = wheel_lateral_fy / parameters.mass_kg - r * u
        r_dot = (wheel_yaw_moment / parameters.iz_kgm2 -
                 parameters.angular_damping_per_s * r)
        v_dot += (parameters.steering_rate_force_gain_n_per_radps *
                  steering_rate_radps / parameters.mass_kg)
        r_dot += (parameters.steering_rate_moment_gain_nm_per_radps *
                  steering_rate_radps / parameters.iz_kgm2)
        u_dot -= parameters.linear_damping_per_s * u
        v_dot -= parameters.linear_damping_per_s * v
        return u_dot, v_dot, r_dot
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
         parameters: PlantParameters,
         internal_substep_s: float | None = None) -> np.ndarray:
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
    integration_substep = (INTEGRATION_SUBSTEP_S if internal_substep_s is None
                            else internal_substep_s)
    if (not math.isfinite(integration_substep) or
            integration_substep <= 0.0):
        raise ValueError("internal_substep_s must be positive and finite")
    count = max(1, int(math.ceil(dt / integration_substep)))
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
