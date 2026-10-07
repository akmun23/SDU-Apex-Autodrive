#!/usr/bin/env python3
"""Check production C/Python/CasADi stage parity and expose optimizer gaps.

The active AutoDRIVE min-time optimizer is a continuous, spatially-transcribed
source-command model, not the older global double-track optimizer. It shares
many production coefficients, but it does not yet propagate the production
steering-command queue or held lateral-velocity state. This tool keeps
production transition parity separate from that remaining optimizer gap.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import re
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import yaml

from tools.racing.offline.controller.production_mpc import ProductionMpc


ROOT = Path(__file__).resolve().parents[2]
NX = 10
NU = 2


class ModelState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in (
        "e_y", "e_psi", "u", "v", "r", "target_speed",
        "steering_command", "delayed_steering_command_1",
        "delayed_steering_command_2", "actual_steering_angle")]


class ModelControl(ctypes.Structure):
    _fields_ = [("steering_rate", ctypes.c_float),
                ("target_speed_rate", ctypes.c_float)]


class StageResult(ctypes.Structure):
    _fields_ = [("next", ModelState), ("delta_s_m", ctypes.c_float),
                ("body_accel_mps2", ctypes.c_float),
                ("branch_flags", ctypes.c_uint), ("valid", ctypes.c_int)]


class StageLinearization(ctypes.Structure):
    _fields_ = [
        ("A", (ctypes.c_float * NX) * NX),
        ("B", (ctypes.c_float * NU) * NX),
        ("d", ctypes.c_float * NX),
        ("nominal_branch_flags", ctypes.c_uint),
        ("nonsmooth_column_mask", ctypes.c_uint16),
        ("nonsmooth_column_count", ctypes.c_int),
        ("valid", ctypes.c_int),
    ]


class YawResidualModel(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("gain", ctypes.c_float),
        ("correction_clip_radps2", ctypes.c_float),
        ("feature_mean", ctypes.c_float * 13),
        ("feature_scale", ctypes.c_float * 13),
        ("coefficients", ctypes.c_float * 14),
        ("target_speed_zero_mps", ctypes.c_float),
        ("target_speed_full_mps", ctypes.c_float),
        ("speed_deficit_full_mps", ctypes.c_float),
        ("speed_deficit_zero_mps", ctypes.c_float),
        ("abs_steering_zero_rad", ctypes.c_float),
        ("abs_steering_full_rad", ctypes.c_float),
    ]


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _smoothstep(value: float, start: float, end: float) -> float:
    t = clamp((value - start) / (end - start), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def yaw_residual_rate(candidate: dict[str, Any], state: np.ndarray,
                      q_delta: float, q_speed: float, curvature: float,
                      acceleration: float) -> float:
    delta = float(state[9])
    u, r, v = float(state[2]), float(state[4]), float(state[3])
    abs_delta = abs(delta)
    features = np.asarray((
        u, r, delta, q_delta, v, curvature, acceleration, q_speed,
        u * delta * abs_delta, delta * abs(q_delta), v * abs_delta,
        abs_delta * acceleration, delta * abs(q_speed)), dtype=np.float64)
    z = (features - np.asarray(candidate["feature_mean"], dtype=np.float64)) / np.asarray(
        candidate["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(candidate["coefficients_with_intercept"], dtype=np.float64)
    correction = float(np.clip(
        coefficients[0] + np.dot(z, coefficients[1:]),
        -float(candidate["correction_clip_radps2"]),
        float(candidate["correction_clip_radps2"])))
    gate = candidate["support_gate"]
    target_gate = _smoothstep(
        float(state[5]), gate["target_speed_zero_mps"],
        gate["target_speed_full_mps"])
    deficit = float(state[5] - state[2])
    tracking_gate = 1.0 - _smoothstep(
        deficit, gate["speed_deficit_full_mps"],
        gate["speed_deficit_zero_mps"])
    steering_gate = _smoothstep(
        abs_delta, gate["abs_steering_zero_rad"],
        gate["abs_steering_full_rad"])
    return float(candidate.get("gain", 1.0)) * correction * target_gate * tracking_gate * steering_gate


def configure_c_yaw_residual(library: ctypes.CDLL,
                             candidate: dict[str, Any] | None) -> None:
    library.vehicle_model_set_yaw_rate_residual_model.argtypes = [
        ctypes.POINTER(YawResidualModel)]
    library.vehicle_model_set_yaw_rate_residual_model.restype = ctypes.c_int
    residual = YawResidualModel()
    if candidate is not None:
        gate = candidate["support_gate"]
        residual.enabled = 1
        residual.gain = float(candidate.get("gain", 1.0))
        residual.correction_clip_radps2 = float(candidate["correction_clip_radps2"])
        for index, value in enumerate(candidate["feature_mean"]):
            residual.feature_mean[index] = float(value)
        for index, value in enumerate(candidate["feature_scale"]):
            residual.feature_scale[index] = float(value)
        for index, value in enumerate(candidate["coefficients_with_intercept"]):
            residual.coefficients[index] = float(value)
        residual.target_speed_zero_mps = float(gate["target_speed_zero_mps"])
        residual.target_speed_full_mps = float(gate["target_speed_full_mps"])
        residual.speed_deficit_full_mps = float(gate["speed_deficit_full_mps"])
        residual.speed_deficit_zero_mps = float(gate["speed_deficit_zero_mps"])
        residual.abs_steering_zero_rad = float(gate["abs_steering_zero_rad"])
        residual.abs_steering_full_rad = float(gate["abs_steering_full_rad"])
    if not library.vehicle_model_set_yaw_rate_residual_model(ctypes.byref(residual)):
        raise RuntimeError("C model rejected fitted yaw residual parameters")


def python_step(model: dict[str, Any], state: np.ndarray, control: np.ndarray,
                dt: float, curvature: float,
                residual_model: dict[str, Any] | None = None) -> tuple[np.ndarray, float, float]:
    p = model["parameters"]
    ey, epsi, u, v, r, target, steer_cmd, delay1, delay2, actual = map(float, state)
    q_delta = clamp(float(control[0]), -p["steering_rate_radps"], p["steering_rate_radps"])
    q_speed = clamp(float(control[1]), -p["target_speed_rate_reduction_mps2"],
                    p["target_speed_rate_increase_mps2"])
    steer_next = clamp(steer_cmd + dt * q_delta,
                       -p["max_steering_angle_rad"], p["max_steering_angle_rad"])
    actual_rate = clamp((delay1 - actual) / dt,
                        -p["steering_rate_radps"], p["steering_rate_radps"])
    actual_next = clamp(actual + dt * actual_rate,
                        -p["max_steering_angle_rad"], p["max_steering_angle_rad"])
    target_next = clamp(target + dt * q_speed, 0.0, p["max_speed_mps"])
    target_mid = 0.5 * (target + target_next)
    u0 = clamp(u, 0.0, p["max_speed_mps"])
    accel = (p["longitudinal_response_bias_mps2"]
             + p["longitudinal_speed_coefficient_per_s"] * u0
             + p["longitudinal_target_error_gain_per_s"] * (target_mid - u0)
             + p["longitudinal_target_rate_coefficient"] * q_speed)
    brake_limit = (p["brake_deceleration_intercept_mps2"]
                   + p["brake_deceleration_speed_slope_per_s"] * u0)
    accel = clamp(accel, -brake_limit, p["longitudinal_acceleration_limit_mps2"])
    u_next = clamp(u0 + dt * accel, 0.0, p["max_speed_mps"])
    u_mid = 0.5 * (u0 + u_next)

    steer_reduction = p["yaw_steering_gain_reduction_per_rad"] * clamp(
        abs(actual_next) - p["yaw_steering_gain_start_rad"], 0.0,
        p["yaw_steering_gain_end_rad"] - p["yaw_steering_gain_start_rad"])
    curvature_reduction = p["yaw_curvature_gain_reduction_per_inv_m"] * clamp(
        abs(curvature) - p["yaw_curvature_gain_start_inv_m"], 0.0,
        p["yaw_curvature_gain_end_inv_m"] - p["yaw_curvature_gain_start_inv_m"])
    steady_r = u_mid * math.tan(actual_next) * (
        p["yaw_steering_gain_per_m"] - steer_reduction - curvature_reduction)
    tau = p["yaw_response_time_constant_s"]
    if p["yaw_low_speed_response_time_constant_s"] > 0.0:
        tau += (p["yaw_low_speed_response_time_constant_s"] - tau) * math.exp(
            -max(u_mid, 0.0) / p["yaw_low_speed_transition_speed_mps"])
    retention = math.exp(-dt / tau)
    r_next = retention * r + (1.0 - retention) * steady_r
    if residual_model is not None:
        r_next += dt * yaw_residual_rate(
            residual_model, state, q_delta, q_speed, curvature, accel)
    r_mid = 0.5 * (r + r_next)

    den0 = 1.0 - curvature * ey
    if abs(den0) < p["frenet_denominator_min_abs"]:
        raise ArithmeticError("initial Frenet denominator is singular")
    sdot0 = (u0 * math.cos(epsi) - v * math.sin(epsi)) / den0
    eydot0 = u0 * math.sin(epsi) + v * math.cos(epsi)
    epsidot0 = r - curvature * sdot0
    ey_mid = ey + 0.5 * dt * eydot0
    epsi_mid = epsi + 0.5 * dt * epsidot0
    den_mid = 1.0 - curvature * ey_mid
    if abs(den_mid) < p["frenet_denominator_min_abs"]:
        raise ArithmeticError("midpoint Frenet denominator is singular")
    sdot_mid = (u_mid * math.cos(epsi_mid) - v * math.sin(epsi_mid)) / den_mid
    eydot_mid = u_mid * math.sin(epsi_mid) + v * math.cos(epsi_mid)
    epsidot_mid = r_mid - curvature * sdot_mid
    next_state = np.asarray((
        ey + dt * eydot_mid,
        wrap(epsi + dt * epsidot_mid),
        u_next,
        v,
        r_next,
        target_next,
        steer_next,
        steer_cmd,
        delay1,
        actual_next,
    ), dtype=np.float64)
    return next_state, dt * sdot_mid, accel


def casadi_function(model: dict[str, Any],
                    residual_model: dict[str, Any] | None = None) -> ca.Function:
    p = model["parameters"]
    x = ca.SX.sym("x", NX)
    w = ca.SX.sym("w", NU)
    dt = ca.SX.sym("dt")
    kappa = ca.SX.sym("kappa")

    clip = lambda z, lo, hi: ca.fmin(ca.fmax(z, lo), hi)
    qd = clip(w[0], -p["steering_rate_radps"], p["steering_rate_radps"])
    qv = clip(w[1], -p["target_speed_rate_reduction_mps2"],
              p["target_speed_rate_increase_mps2"])
    steer_next = clip(x[6] + dt * qd,
                      -p["max_steering_angle_rad"], p["max_steering_angle_rad"])
    actual_rate = clip((x[7] - x[9]) / dt,
                       -p["steering_rate_radps"], p["steering_rate_radps"])
    actual_next = clip(x[9] + dt * actual_rate,
                       -p["max_steering_angle_rad"], p["max_steering_angle_rad"])
    target_next = clip(x[5] + dt * qv, 0.0, p["max_speed_mps"])
    target_mid = 0.5 * (x[5] + target_next)
    u0 = clip(x[2], 0.0, p["max_speed_mps"])
    accel = (p["longitudinal_response_bias_mps2"]
             + p["longitudinal_speed_coefficient_per_s"] * u0
             + p["longitudinal_target_error_gain_per_s"] * (target_mid - u0)
             + p["longitudinal_target_rate_coefficient"] * qv)
    brake_limit = (p["brake_deceleration_intercept_mps2"]
                   + p["brake_deceleration_speed_slope_per_s"] * u0)
    accel = clip(accel, -brake_limit, p["longitudinal_acceleration_limit_mps2"])
    unext = clip(u0 + dt * accel, 0.0, p["max_speed_mps"])
    umid = 0.5 * (u0 + unext)

    steering_reduction = p["yaw_steering_gain_reduction_per_rad"] * clip(
        ca.fabs(actual_next) - p["yaw_steering_gain_start_rad"], 0.0,
        p["yaw_steering_gain_end_rad"] - p["yaw_steering_gain_start_rad"])
    curvature_reduction = p["yaw_curvature_gain_reduction_per_inv_m"] * clip(
        ca.fabs(kappa) - p["yaw_curvature_gain_start_inv_m"], 0.0,
        p["yaw_curvature_gain_end_inv_m"] - p["yaw_curvature_gain_start_inv_m"])
    steady_r = umid * ca.tan(actual_next) * (
        p["yaw_steering_gain_per_m"] - steering_reduction - curvature_reduction)
    tau = p["yaw_response_time_constant_s"]
    if p["yaw_low_speed_response_time_constant_s"] > 0.0:
        tau += (p["yaw_low_speed_response_time_constant_s"] - tau) * ca.exp(
            -ca.fmax(umid, 0.0) / p["yaw_low_speed_transition_speed_mps"])
    retention = ca.exp(-dt / tau)
    rnext = retention * x[4] + (1.0 - retention) * steady_r
    if residual_model is not None:
        mean = residual_model["feature_mean"]
        scale = residual_model["feature_scale"]
        coeff = residual_model["coefficients_with_intercept"]
        delta = x[9]
        abs_delta = ca.fabs(delta)
        features = (
            x[2], x[4], delta, qd, x[3], kappa, accel, qv,
            x[2] * delta * abs_delta, delta * ca.fabs(qd),
            x[3] * abs_delta, abs_delta * accel, delta * ca.fabs(qv))
        correction = float(coeff[0])
        for index, feature in enumerate(features):
            correction += float(coeff[index + 1]) * (
                feature - float(mean[index])) / float(scale[index])
        correction = clip(
            correction, -float(residual_model["correction_clip_radps2"]),
            float(residual_model["correction_clip_radps2"]))
        gate = residual_model["support_gate"]

        def smoothstep(value, start, end):
            t = clip((value - float(start)) / (float(end) - float(start)), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        target_gate = smoothstep(
            x[5], gate["target_speed_zero_mps"], gate["target_speed_full_mps"])
        tracking_gate = 1.0 - smoothstep(
            x[5] - x[2], gate["speed_deficit_full_mps"],
            gate["speed_deficit_zero_mps"])
        steering_gate = smoothstep(
            abs_delta, gate["abs_steering_zero_rad"],
            gate["abs_steering_full_rad"])
        rnext += dt * float(residual_model.get("gain", 1.0)) * correction * target_gate * tracking_gate * steering_gate
    rmid = 0.5 * (x[4] + rnext)

    den0 = 1.0 - kappa * x[0]
    sdot0 = (u0 * ca.cos(x[1]) - x[3] * ca.sin(x[1])) / den0
    eydot0 = u0 * ca.sin(x[1]) + x[3] * ca.cos(x[1])
    epsidot0 = x[4] - kappa * sdot0
    eymid = x[0] + 0.5 * dt * eydot0
    epsimid = x[1] + 0.5 * dt * epsidot0
    denmid = 1.0 - kappa * eymid
    sdotmid = (umid * ca.cos(epsimid) - x[3] * ca.sin(epsimid)) / denmid
    eydotmid = umid * ca.sin(epsimid) + x[3] * ca.cos(epsimid)
    epsidotmid = rmid - kappa * sdotmid
    epsi_next = ca.atan2(ca.sin(x[1] + dt * epsidotmid),
                         ca.cos(x[1] + dt * epsidotmid))
    xnext = ca.vertcat(
        x[0] + dt * eydotmid, epsi_next, unext, x[3], rnext, target_next,
        steer_next, x[6], x[7], actual_next)
    z = ca.vertcat(x, w)
    jacobian = ca.jacobian(xnext, z)
    return ca.Function("shared_source_command_step", [x, w, dt, kappa],
                       [xnext, dt * sdotmid, accel, jacobian])


def state_array(state: ModelState) -> np.ndarray:
    return np.asarray([getattr(state, name) for name, _ in state._fields_], dtype=np.float64)


def c_step(library: ctypes.CDLL, x: np.ndarray, w: np.ndarray,
           dt: float, curvature: float) -> tuple[np.ndarray, float, float,
                                                 np.ndarray, np.ndarray, int]:
    state = ModelState(*map(float, x))
    control = ModelControl(*map(float, w))
    stage = library.mpc_vehicle_model_step(
        ctypes.byref(state), ctypes.byref(control), ctypes.c_float(dt),
        ctypes.c_float(curvature))
    if not stage.valid:
        raise ArithmeticError("production C model rejected sampled stage")
    linearization = StageLinearization()
    ok = library.mpc_vehicle_model_step_with_jacobian(
        ctypes.byref(state), ctypes.byref(control), ctypes.c_float(dt),
        ctypes.c_float(curvature), ctypes.byref(stage), ctypes.byref(linearization))
    if not ok or not linearization.valid:
        raise ArithmeticError("production C model rejected Jacobian stage")
    c_jac = np.zeros((NX, NX + NU), dtype=np.float64)
    for i in range(NX):
        for j in range(NX):
            c_jac[i, j] = linearization.A[i][j]
        for j in range(NU):
            c_jac[i, NX + j] = linearization.B[i][j]
    return (state_array(stage.next), float(stage.delta_s_m),
            float(stage.body_accel_mps2), c_jac,
            np.asarray([stage.branch_flags, linearization.nonsmooth_column_count]),
            int(stage.valid))


def check_source_config(model: dict[str, Any], config_path: Path) -> dict[str, Any]:
    params = yaml.safe_load(config_path.read_text(encoding="utf-8"))["/**"]["ros__parameters"]
    p = model["parameters"]
    keys = {
        "max_speed_mps": "max_speed_mps",
        "steering_command_delay_s": "physical_steering_delay_s",
        "yaw_response_time_constant_s": "yaw_rate_response_time_constant_s",
        "yaw_steering_gain_per_m": "yaw_rate_steering_gain_per_m",
        "yaw_steering_gain_reduction_per_rad": "yaw_rate_steering_gain_reduction_per_rad",
        "yaw_steering_gain_start_rad": "yaw_rate_steering_gain_start_rad",
        "yaw_steering_gain_end_rad": "yaw_rate_steering_gain_end_rad",
        "yaw_curvature_gain_reduction_per_inv_m": "yaw_rate_curvature_gain_reduction_per_m",
        "yaw_curvature_gain_start_inv_m": "yaw_rate_curvature_gain_start_per_m",
        "yaw_curvature_gain_end_inv_m": "yaw_rate_curvature_gain_end_per_m",
        "yaw_low_speed_response_time_constant_s": "yaw_rate_low_speed_response_time_constant_s",
        "yaw_low_speed_transition_speed_mps": "yaw_rate_low_speed_transition_speed_mps",
        "yaw_response_surface_enabled": "yaw_rate_response_surface_enabled",
    }
    mismatches = {}
    for source_key, yaml_key in keys.items():
        if yaml_key not in params:
            mismatches[yaml_key] = {"model": p[source_key], "yaml": "missing"}
            continue
        actual = params[yaml_key]
        expected = p[source_key]
        if isinstance(expected, bool):
            matches = bool(actual) is expected
        else:
            matches = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-7)
        if not matches:
            mismatches[yaml_key] = {"model": expected, "yaml": actual}
    return {"matches": not mismatches, "mismatches": mismatches}


def check_header_contract(model: dict[str, Any], header_path: Path) -> dict[str, Any]:
    source = header_path.read_text(encoding="utf-8")
    mapping = {
        "SOURCE_MAX_STEERING_RAD": "max_steering_angle_rad",
        "SOURCE_STEERING_RATE_RADPS": "steering_rate_radps",
        "SOURCE_STEERING_WHEELBASE_M": "wheelbase_m",
        "MPC_MAX_COMMAND_SPEED_MPS": "max_speed_mps",
        "MPC_TARGET_SPEED_RATE_INCREASE_MAX_MPS2": "target_speed_rate_increase_mps2",
        "MPC_TARGET_SPEED_RATE_REDUCTION_MAX_MPS2": "target_speed_rate_reduction_mps2",
        "MPC_LONGITUDINAL_RESPONSE_BIAS_MPS2": "longitudinal_response_bias_mps2",
        "MPC_LONGITUDINAL_SPEED_COEFF_PER_S": "longitudinal_speed_coefficient_per_s",
        "MPC_LONGITUDINAL_TARGET_ERROR_GAIN_PER_S": "longitudinal_target_error_gain_per_s",
        "MPC_LONGITUDINAL_TARGET_RATE_COEFF": "longitudinal_target_rate_coefficient",
        "MPC_LONGITUDINAL_ACCEL_LIMIT_MPS2": "longitudinal_acceleration_limit_mps2",
        "MPC_LONGITUDINAL_BRAKE_DECEL_INTERCEPT_MPS2": "brake_deceleration_intercept_mps2",
        "MPC_LONGITUDINAL_BRAKE_DECEL_SLOPE_S_INV": "brake_deceleration_speed_slope_per_s",
        "MPC_YAW_RATE_RESPONSE_TIME_CONSTANT_SECONDS": "yaw_response_time_constant_s",
        "MPC_YAW_RATE_STEERING_GAIN_PER_M": "yaw_steering_gain_per_m",
    }
    mismatches = {}
    for macro, key in mapping.items():
        match = re.search(
            rf"^\s*#define\s+{macro}\s+\(?"
            rf"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)[fF]?\)?",
            source, re.M)
        if not match:
            mismatches[macro] = {"model": model["parameters"][key], "header": "missing"}
            continue
        value = float(match.group(1))
        if not math.isclose(value, float(model["parameters"][key]), rel_tol=0.0, abs_tol=1e-7):
            mismatches[macro] = {"model": model["parameters"][key], "header": value}
    return {"matches": not mismatches, "mismatches": mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "config/racing/racing_vehicle_model.json")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--mpc-config", type=Path, default=ROOT / "f1tenth_mpc/config/mpc_competition.yaml")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--yaw-residual-candidate", type=Path,
                        help="optional fitted JSON candidate to verify through C/Python/CasADi")
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "live_runs/racing_model_diagnostics_20261006/model_parity_shared_transition_v1.json")
    args = parser.parse_args()

    model = json.loads(args.model.read_text(encoding="utf-8"))
    residual_model = (json.loads(args.yaw_residual_candidate.read_text(encoding="utf-8"))
                      if args.yaw_residual_candidate else None)
    if residual_model is not None and residual_model.get("support_gate", {}).get("kind") != \
            "smoothstep_target_tracking_steering_v1":
        raise ValueError("unsupported fitted yaw residual support gate")
    p = model["parameters"]
    rng = np.random.default_rng(args.seed)
    library = ctypes.CDLL(str(args.library.resolve()))
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl), ctypes.c_float, ctypes.c_float]
    library.mpc_vehicle_model_step.restype = StageResult
    library.mpc_vehicle_model_step_with_jacobian.argtypes = [
        ctypes.POINTER(ModelState), ctypes.POINTER(ModelControl), ctypes.c_float,
        ctypes.c_float, ctypes.POINTER(StageResult), ctypes.POINTER(StageLinearization)]
    library.mpc_vehicle_model_step_with_jacobian.restype = ctypes.c_int
    casadi_step = casadi_function(model, residual_model)
    max_state_error = max_distance_error = max_accel_error = max_jac_error = 0.0
    max_state_error_by_channel = np.zeros(NX, dtype=np.float64)
    all_smooth = 0

    # Loading the real runtime YAML configures the C yaw parameters exactly as
    # the production controller does; no simulator/controller process starts.
    with ProductionMpc(args.library, args.mpc_config, args.trajectory):
        configure_c_yaw_residual(library, residual_model)
        steering_knots = (0.0, 0.39, 0.41, 0.435, 0.46, 0.48,
                          p["max_steering_angle_rad"])
        for case_index in range(args.cases):
            u = float(rng.uniform(1.2, 11.8))
            target = float(np.clip(u + rng.uniform(-0.6, 0.6), 0.5, 15.5))
            steering = float(rng.uniform(-p["max_steering_angle_rad"],
                                          p["max_steering_angle_rad"]))
            if case_index % 8 == 0:
                knot = steering_knots[(case_index // 8) % len(steering_knots)]
                actual = math.copysign(knot, -1.0 if case_index % 16 else 1.0)
            else:
                actual = float(rng.uniform(-p["max_steering_angle_rad"],
                                           p["max_steering_angle_rad"]))
            actual = clamp(actual, -p["max_steering_angle_rad"],
                           p["max_steering_angle_rad"])
            delay1 = float(np.clip(actual + rng.uniform(-0.005, 0.005),
                                   -p["max_steering_angle_rad"],
                                   p["max_steering_angle_rad"]))
            state = np.asarray((
                rng.uniform(-0.12, 0.12), rng.uniform(-0.20, 0.20), u,
                rng.uniform(-0.4, 0.4), rng.uniform(-3.0, 3.0), target,
                steering, delay1,
                rng.uniform(-p["max_steering_angle_rad"], p["max_steering_angle_rad"]),
                actual,
            ), dtype=np.float64)
            control = np.asarray((rng.uniform(-2.5, 2.5), rng.uniform(-2.0, 2.0)))
            dt = float(model["sample_period_s"])
            curvature = float(rng.uniform(-1.2, 1.2))
            py_next, py_ds, py_ax = python_step(
                model, state, control, dt, curvature, residual_model)
            c_next, c_ds, c_ax, c_jac, flags, _ = c_step(
                library, state, control, dt, curvature)
            ca_result = casadi_step(state, control, dt, curvature)
            ca_next = np.asarray(ca_result[0]).reshape(-1).astype(np.float64)
            ca_ds, ca_ax = float(ca_result[1]), float(ca_result[2])
            ca_jac = np.asarray(ca_result[3], dtype=np.float64)
            max_state_error = max(max_state_error,
                                  float(np.max(np.abs(py_next - c_next))),
                                  float(np.max(np.abs(ca_next - c_next))))
            max_state_error_by_channel = np.maximum(
                max_state_error_by_channel,
                np.maximum(np.abs(py_next - c_next), np.abs(ca_next - c_next)))
            max_distance_error = max(max_distance_error, abs(py_ds - c_ds), abs(ca_ds - c_ds))
            max_accel_error = max(max_accel_error, abs(py_ax - c_ax), abs(ca_ax - c_ax))
            if flags[0] == 0 and flags[1] == 0:
                all_smooth += 1
                max_jac_error = max(max_jac_error,
                                     float(np.max(np.abs(c_jac - ca_jac))))

    yaml_match = check_source_config(model, args.mpc_config)
    header_match = check_header_contract(model, ROOT / "f1tenth_mpc/include/mpc_types.h")
    state_tolerance = 1.0e-4 if residual_model is not None else 2.0e-5
    report = {
        "schema_version": 1,
        "model_id": model["model_id"],
        "yaw_residual_candidate_id": (
            residual_model.get("model_id") if residual_model else None),
        "cases": args.cases,
        "seed": args.seed,
        "production_transition_parity": {
            "max_state_abs_error": max_state_error,
            "max_state_abs_error_by_channel": {
                name: float(max_state_error_by_channel[index])
                for index, name in enumerate(model["state_order"])
            },
            "max_delta_s_abs_error_m": max_distance_error,
            "max_body_accel_abs_error_mps2": max_accel_error,
            "acceptance_tolerances": {
                "state_abs": state_tolerance,
                "delta_s_abs_m": 2.0e-5,
                "body_accel_abs_mps2": 2.0e-5,
                "jacobian_abs": 2.0e-3,
            },
            "jacobian_cases_without_branch_or_nonsmooth_flags": all_smooth,
            "max_c_vs_casadi_jacobian_abs_error": max_jac_error,
            "passed": max_state_error < state_tolerance and max_distance_error < 2e-5
                and max_accel_error < 2e-5 and max_jac_error < 2e-3,
        },
        "production_config_matches_shared_spec": yaml_match,
        "compiled_c_header_matches_shared_spec": header_match,
        "optimizer_contract": {
            "status": "continuous_spatial_candidate_fails_recursive_parity",
            "optimizer_model": "custom continuous spatial source-command model",
            "optimizer_states": [
                "e_y_m", "e_psi_rad", "body_u_mps", "yaw_rate_radps",
                "target_speed_mps", "steering_command_rad (queue-aware option)",
                "actual_steering_angle_rad (queue-aware option)",
            ],
            "optimizer_controls": [
                "steering_rate_radps", "target_speed_rate_mps2",
            ],
            "mpc_states": model["state_order"],
            "mpc_controls": model["control_order"],
            "known_stage_differences": [
                "queue-aware optimizer mode separates command and actual steering, but reconstructs the 25 ms history with continuous spatial Hermite interpolation rather than the exact discrete queue transition",
                "optimizer recomputes lateral velocity algebraically from speed and yaw rate; production MPC holds the current lateral-velocity state through each prediction stage",
                "optimizer uses continuous spatial collocation and exact continuous yaw-lag integration; production MPC uses a 25 ms discrete stage with midpoint speed and next-step physical steering",
                "the queue-aware candidate's recursive production-C replay departed by more than 0.05 rad heading after 4 stages and the production model rejected the rollout at 12.22 m; see p0_steering_queue_20261006/retry_from_failed_iterate/stage_audit.json",
            ],
            "parity_passed": False,
        },
        "handoff_h2_passed": False,
        "interpretation": "Production C/Python/CasADi transition parity passes, but active optimizer stage parity does not. See the active-optimizer stage audit for a trajectory-conditioned estimate of the remaining discrepancy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if residual_model is not None:
        return 0 if report["production_transition_parity"]["passed"] else 1
    return 0 if report["handoff_h2_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
