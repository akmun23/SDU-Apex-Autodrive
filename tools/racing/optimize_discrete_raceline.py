#!/usr/bin/env python3
"""Offline speed-profile optimization on the frozen P0 path.

This is a model-contract experiment, not a runtime controller. It uses the
verified production 25 ms source-command transition in direct multiple
shooting, keeps P0 geometry fixed, and accepts a horizon only if an independent
production-C replay completes the lap with negligible state error.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import yaml

from tools.racing.verify_optimizer_mpc_model_parity import (
    ModelControl,
    ModelState,
    StageResult,
    casadi_function,
    check_header_contract,
    check_source_config,
    configure_c_yaw_residual,
    python_step,
)
from tools.racing.offline.controller.production_mpc import ProductionMpc


ROOT = Path(__file__).resolve().parents[2]
NX = 10
NU = 2
DT = 0.025
DEFAULT_MODEL = ROOT / "config/racing/racing_vehicle_model.json"
DEFAULT_MPC_CONFIG = ROOT / "f1tenth_mpc/config/mpc_competition.yaml"
DEFAULT_TRAJECTORY = ROOT / (
    "live_runs/raceline_candidates/"
    "practice_9g_runtime_matched_wallmargin010_20261005/output/"
    "autodrive_mintime_raceline.csv"
)
DEFAULT_OPTIMIZER_CONFIG = ROOT / (
    "live_runs/raceline_candidates/p0_steering_queue_20261006/"
    "retry_from_failed_iterate/resolved_config.yaml"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trajectory(path: Path) -> dict[str, np.ndarray | float]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(value) for value in line.split(",")])
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 9 or len(values) < 20:
        raise ValueError(f"Unsupported or incomplete trajectory CSV: {path}")
    if not np.isfinite(values).all():
        raise ValueError("Trajectory contains non-finite values")
    if np.linalg.norm(values[-1, 1:3] - values[0, 1:3]) < 1.0e-4:
        values = values[:-1]
    s = values[:, 0]
    if abs(s[0]) > 1.0e-8 or np.any(np.diff(s) <= 0.0):
        raise ValueError("Trajectory arc length must start at zero and increase")
    closing_distance = float(np.linalg.norm(values[0, 1:3] - values[-1, 1:3]))
    length = float(s[-1] + closing_distance)
    if length <= s[-1] or closing_distance <= 0.0:
        raise ValueError("Trajectory has invalid closing segment")
    return {
        "rows": values,
        "s": s,
        "x": values[:, 1],
        "y": values[:, 2],
        "heading": values[:, 3],
        "kappa": values[:, 4],
        "speed": values[:, 5],
        "accel": values[:, 6],
        "left": values[:, 7],
        "right": values[:, 8],
        "length": length,
    }


def periodic_linear(query: float | np.ndarray, s: np.ndarray,
                    values: np.ndarray, length: float) -> np.ndarray:
    q = np.mod(np.asarray(query, dtype=np.float64), length)
    x = np.r_[s, length]
    y = np.r_[values, values[0]]
    return np.interp(q, x, y)


def build_lookup(name: str, s: np.ndarray, values: np.ndarray,
                 length: float) -> ca.Function:
    grid = np.r_[s, length]
    data = np.r_[values, values[0]]
    return ca.interpolant(name, "linear", [grid], data)


def desired_steady_yaw(speed: float, steering: float, curvature: float,
                       p: dict[str, Any]) -> float:
    steer_reduction = p["yaw_steering_gain_reduction_per_rad"] * min(
        max(abs(steering) - p["yaw_steering_gain_start_rad"], 0.0),
        p["yaw_steering_gain_end_rad"] - p["yaw_steering_gain_start_rad"],
    )
    curvature_reduction = p["yaw_curvature_gain_reduction_per_inv_m"] * min(
        max(abs(curvature) - p["yaw_curvature_gain_start_inv_m"], 0.0),
        p["yaw_curvature_gain_end_inv_m"] - p["yaw_curvature_gain_start_inv_m"],
    )
    gain = p["yaw_steering_gain_per_m"] - steer_reduction - curvature_reduction
    return speed * math.tan(steering) * gain


def steering_for_yaw(speed: float, yaw_rate: float, curvature: float,
                     p: dict[str, Any]) -> float:
    limit = float(p["max_steering_angle_rad"])
    target = float(yaw_rate)
    lo, hi = -limit, limit
    lo_value = desired_steady_yaw(speed, lo, curvature, p)
    hi_value = desired_steady_yaw(speed, hi, curvature, p)
    if target <= lo_value:
        return lo
    if target >= hi_value:
        return hi
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if desired_steady_yaw(speed, mid, curvature, p) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def setup_c_library(library: Any) -> None:
    library.mpc_vehicle_model_step.argtypes = [
        ctypes.POINTER(ModelState),
        ctypes.POINTER(ModelControl),
        ctypes.c_float,
        ctypes.c_float,
    ]
    library.mpc_vehicle_model_step.restype = StageResult


def production_c_step(library: Any, state: np.ndarray, control: np.ndarray,
                      curvature: float) -> tuple[np.ndarray, float, float]:
    c_state = ModelState(*map(float, state))
    c_control = ModelControl(*map(float, control))
    stage = library.mpc_vehicle_model_step(
        ctypes.byref(c_state), ctypes.byref(c_control),
        ctypes.c_float(DT), ctypes.c_float(curvature),
    )
    if not stage.valid:
        raise ArithmeticError("production C transition rejected optimized stage")
    x_next = np.asarray(
        [getattr(stage.next, name) for name, _ in ModelState._fields_],
        dtype=np.float64,
    )
    return x_next, float(stage.delta_s_m), float(stage.body_accel_mps2)


def initial_guess(track: dict[str, Any], model: dict[str, Any],
                  steps: int,
                  yaw_residual_model: dict[str, Any] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = model["parameters"]
    length = float(track["length"])
    source_s = np.asarray(track["s"])
    source_speed = np.asarray(track["speed"])
    source_kappa = np.asarray(track["kappa"])
    source_target_time = float(np.sum(
        np.diff(np.r_[source_s, length]) /
        (0.5 * (source_speed + np.roll(source_speed, -1)))
    ))
    def rollout(speed_scale: float):
        desired_speed = lambda query: float(np.clip(
            periodic_linear(query, source_s, source_speed, length) * speed_scale,
            0.8, float(p["max_speed_mps"]),
        ))
        initial_speed = desired_speed(0.0)
        initial_curvature = float(periodic_linear(
            0.0, source_s, source_kappa, length))
        initial_delta = steering_for_yaw(
            initial_speed, initial_speed * initial_curvature, initial_curvature, p)
        initial_target = np.clip(
            initial_speed - (p["longitudinal_response_bias_mps2"]
                             + p["longitudinal_speed_coefficient_per_s"] * initial_speed)
            / p["longitudinal_target_error_gain_per_s"],
            0.0, float(p["max_speed_mps"]),
        )
        state = np.asarray((
            0.0, 0.0, initial_speed, 0.0,
            desired_steady_yaw(initial_speed, initial_delta, initial_curvature, p),
            initial_target, initial_delta, initial_delta, initial_delta, initial_delta,
        ), dtype=np.float64)
        progress = np.zeros(steps + 1, dtype=np.float64)
        states = np.zeros((NX, steps + 1), dtype=np.float64)
        controls = np.zeros((NU, steps), dtype=np.float64)
        states[:, 0] = state

        # This feedback policy only builds a dynamically feasible NLP seed;
        # the optimized schedule determines the final candidate.
        for k in range(steps):
            s_now = float(progress[k])
            u_ref = desired_speed(s_now)
            target_equilibrium = u_ref - (
                p["longitudinal_response_bias_mps2"]
                + p["longitudinal_speed_coefficient_per_s"] * u_ref
            ) / p["longitudinal_target_error_gain_per_s"]
            qv = float(np.clip(
                (target_equilibrium - state[5]) / 0.15,
                -p["target_speed_rate_reduction_mps2"],
                p["target_speed_rate_increase_mps2"],
            ))
            future_s = s_now + 2.0 * DT * max(float(state[2]), 0.0)
            future_kappa = float(periodic_linear(
                future_s, source_s, source_kappa, length))
            current_kappa = float(periodic_linear(
                s_now, source_s, source_kappa, length))
            denominator = max(0.25, 1.0 - current_kappa * float(state[0]))
            desired_r = (
                future_kappa * max(float(state[2]), 0.0)
                * math.cos(float(state[1])) / denominator
                - 4.0 * float(state[1]) - 2.0 * float(state[0])
            )
            desired_delta = steering_for_yaw(
                max(float(state[2]), 0.8), desired_r, future_kappa, p)
            qdelta = float(np.clip(
                (desired_delta - state[6]) / DT,
                -p["steering_rate_radps"], p["steering_rate_radps"],
            ))
            controls[:, k] = (qdelta, qv)
            state_next, delta_s, _ = python_step(
                model, state, controls[:, k], DT, current_kappa,
                yaw_residual_model)
            progress[k + 1] = s_now + delta_s
            states[:, k + 1] = state_next
            state = state_next
        return progress, states, controls

    # Match the seed's final distance to one lap rather than assuming its
    # target-speed tracking is instantaneous. This only calibrates the initial
    # guess; IPOPT still enforces the exact terminal-progress constraint.
    nominal_scale = source_target_time / (steps * DT)
    low, high = 0.70 * nominal_scale, 1.45 * nominal_scale
    low_rollout = rollout(low)
    high_rollout = rollout(high)
    if low_rollout[0][-1] <= length <= high_rollout[0][-1]:
        for _ in range(12):
            mid = 0.5 * (low + high)
            mid_rollout = rollout(mid)
            if mid_rollout[0][-1] < length:
                low = mid
            else:
                high = mid
        return rollout(0.5 * (low + high))
    return low_rollout if abs(low_rollout[0][-1] - length) < abs(
        high_rollout[0][-1] - length) else high_rollout


def interpolate_warm_start(old: dict[str, np.ndarray], steps: int
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_steps = old["controls"].shape[1]
    new_tau = np.linspace(0.0, 1.0, steps + 1)
    old_tau_x = np.linspace(0.0, 1.0, old_steps + 1)
    old_tau_u = (np.arange(old_steps, dtype=np.float64) + 0.5) / old_steps
    old_tau_new_u = (np.arange(steps, dtype=np.float64) + 0.5) / steps
    progress = np.interp(new_tau, old_tau_x, old["progress"])
    states = np.vstack([
        np.interp(new_tau, old_tau_x, old["states"][i]) for i in range(NX)
    ])
    controls = np.vstack([
        np.interp(old_tau_new_u, old_tau_u, old["controls"][i])
        for i in range(NU)
    ])
    return progress, states, controls


def solver_diagnostics(stats: dict[str, Any]) -> dict[str, Any]:
    iterations = stats.get("iterations", {})

    def last_value(key: str) -> float | None:
        values = iterations.get(key)
        if not isinstance(values, (list, tuple)) or not values:
            return None
        try:
            return float(values[-1])
        except (TypeError, ValueError):
            return None

    return {
        "return_status": stats.get("return_status"),
        "success": bool(stats.get("success", False)),
        "iterations": int(stats.get("iter_count", 0) or 0),
        "final_primal_infeasibility": last_value("inf_pr"),
        "final_dual_infeasibility": last_value("inf_du"),
        "final_objective": last_value("obj"),
    }


def solve_horizon(steps: int, track: dict[str, Any], model: dict[str, Any],
                  optimizer_config: dict[str, Any],
                  dynamic: ca.Function,
                  warm: dict[str, np.ndarray] | None,
                  max_iter: int,
                  hessian_approximation: str,
                  yaw_residual_model: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    p = model["parameters"]
    length = float(track["length"])
    s_ref = np.asarray(track["s"])
    kappa_ref = np.asarray(track["kappa"])
    left_ref = np.asarray(track["left"])
    right_ref = np.asarray(track["right"])
    kappa_at = build_lookup(f"kappa_{steps}", s_ref, kappa_ref, length)
    left_at = build_lookup(f"left_{steps}", s_ref, left_ref, length)
    right_at = build_lookup(f"right_{steps}", s_ref, right_ref, length)

    # Normalize decision variables to keep IPOPT's primal scales comparable.
    sx = np.asarray([0.5, 0.5, 10.0, 0.35, 10.0, 10.0, 0.5, 0.5, 0.5, 0.5])
    su = np.asarray([p["steering_rate_radps"],
                     p["target_speed_rate_reduction_mps2"]])
    opti = ca.Opti()
    z = opti.variable(NX, steps + 1)
    w = opti.variable(NU, steps)
    progress_fraction = opti.variable(1, steps + 1)
    X = ca.diag(ca.DM(sx)) @ z
    U = ca.diag(ca.DM(su)) @ w
    S = length * progress_fraction

    if warm is None:
        s_guess, x_guess, u_guess = initial_guess(
            track, model, steps, yaw_residual_model)
    else:
        s_guess, x_guess, u_guess = interpolate_warm_start(warm, steps)
    opti.set_initial(z, x_guess / sx[:, None])
    opti.set_initial(w, u_guess / su[:, None])
    opti.set_initial(progress_fraction, (s_guess / length)[None, :])

    raw_optimizer_config = optimizer_config.get("config", optimizer_config)
    vehicle = optimizer_config.get("vehicle_model", {})
    required_geometry = (
        "required_wall_clearance_m", "planning_footprint_width_m", "car_width_m",
        "rear_axle_to_front_bumper_m", "rear_overhang_m",
    )
    missing_geometry = [name for name in required_geometry if name not in vehicle]
    if missing_geometry:
        raise ValueError(
            "resolved optimizer config lacks vehicle geometry: "
            + ", ".join(missing_geometry))
    limits = raw_optimizer_config.get("limits", {})
    track_cfg = raw_optimizer_config.get("track", {})
    numerics = raw_optimizer_config.get("numerics", {})
    envelope_cfg = raw_optimizer_config.get("lateral_envelope", {})
    combined_cfg = raw_optimizer_config.get("combined_acceleration", {})
    clearance = (float(vehicle["required_wall_clearance_m"])
                 + float(track_cfg.get("extra_wall_clearance_m", 0.10)))
    min_speed = float(limits.get("min_body_speed_mps", 0.8))
    max_heading = float(limits.get("max_heading_error_rad", 0.75))
    min_denominator = float(numerics.get("min_frenet_denominator", 0.25))
    min_progress_mps = float(numerics.get("min_progress_mps", 0.3))
    planning_half_width = 0.5 * float(vehicle["planning_footprint_width_m"])
    front_extent = max(float(vehicle["rear_axle_to_front_bumper_m"]),
                       float(vehicle["rear_overhang_m"]))
    ay_base = max(float(value) for row in envelope_cfg.get("ay_max_mps2", [[8.0]])
                  for value in (row if isinstance(row, list) else [row]))
    ay_cap = ay_base * float(envelope_cfg.get("scale", 1.0))
    combined_enabled = bool(combined_cfg.get("enabled", False))
    combined_exponent = float(combined_cfg.get("exponent", 2.0))

    opti.subject_to(S[0] == 0.0)
    opti.subject_to(S[steps] == length)
    # v is an identity state in the production transition. Fixing v[0]=0
    # propagates it exactly; including its terminal periodic equality would
    # duplicate an already implied equality and degrade IPOPT's rank test.
    periodic_indices = (0, 1, 2, 4, 5, 6, 7, 8, 9)
    for state_index in periodic_indices:
        opti.subject_to(X[state_index, steps] == X[state_index, 0])
    opti.subject_to(X[3, 0] == 0.0)  # Production holds this lateral state.
    objective = 0
    state_scale = ca.DM(sx)
    for k in range(steps + 1):
        xk = X[:, k]
        sk = S[k]
        kappa = kappa_at(sk)
        left = left_at(sk)
        right = right_at(sk)
        den = 1.0 - kappa * xk[0]
        physical_footprint = (
            0.5 * float(vehicle["car_width_m"]) * ca.sqrt(ca.cos(xk[1]) ** 2 + 1e-10)
            + front_extent * ca.sqrt(ca.sin(xk[1]) ** 2 + 1e-10)
        )
        footprint = ca.fmax(planning_half_width, physical_footprint)
        opti.subject_to(opti.bounded(min_speed, xk[2], p["max_speed_mps"]))
        opti.subject_to(opti.bounded(-max_heading, xk[1], max_heading))
        opti.subject_to(opti.bounded(-p["max_steering_angle_rad"], xk[6],
                                    p["max_steering_angle_rad"]))
        opti.subject_to(opti.bounded(-p["max_steering_angle_rad"], xk[9],
                                    p["max_steering_angle_rad"]))
        opti.subject_to(xk[5] >= 0.0)
        opti.subject_to(xk[5] <= p["max_speed_mps"])
        opti.subject_to(den >= min_denominator)
        opti.subject_to(xk[0] <= left - clearance - footprint)
        opti.subject_to(xk[0] >= -(right - clearance - footprint))
        if k < steps:
            opti.subject_to(opti.bounded(
                -p["steering_rate_radps"], U[0, k], p["steering_rate_radps"]))
            opti.subject_to(opti.bounded(
                -p["target_speed_rate_reduction_mps2"], U[1, k],
                p["target_speed_rate_increase_mps2"]))
            x_next, delta_s, acceleration, _ = dynamic(
                xk, U[:, k], DT, kappa)
            opti.subject_to((X[:, k + 1] - x_next) / state_scale == 0.0)
            opti.subject_to(delta_s >= DT * min_progress_mps)
            opti.subject_to((S[k + 1] - S[k] - delta_s) / 0.10 == 0.0)
            ay = xk[2] * xk[4]
            if combined_enabled:
                brake = (p["brake_deceleration_intercept_mps2"]
                         + p["brake_deceleration_speed_slope_per_s"] * xk[2])
                ax_norm = ca.if_else(
                    acceleration >= 0.0,
                    acceleration / p["longitudinal_acceleration_limit_mps2"],
                    -acceleration / brake,
                )
                ay_norm = ay / ay_cap
                opti.subject_to(
                    ca.power(ax_norm, combined_exponent)
                    + ca.power(ay_norm, combined_exponent) <= 1.0
                )
            else:
                opti.subject_to(ca.fabs(ay) <= ay_cap)
            # The horizon is fixed-step, so this cost selects a smooth
            # center-tracking schedule among feasible solutions. Lap time is
            # compared only as steps * DT, never inferred from this objective.
            objective += (
                2.0 * ca.power(xk[0] / 0.5, 2)
                + 1.0 * ca.power(xk[1] / 0.5, 2)
                + 0.03 * ca.power(U[0, k] / p["steering_rate_radps"], 2)
                + 0.02 * ca.power(U[1, k] / p["target_speed_rate_reduction_mps2"], 2)
            )
    opti.minimize(objective)
    solver_options = {
        "max_iter": int(max_iter),
        "tol": 2.0e-6,
        "acceptable_tol": 2.0e-5,
        "acceptable_iter": 12,
        "acceptable_constr_viol_tol": 2.0e-5,
        "print_level": 0,
        "sb": "yes",
        "nlp_scaling_method": "gradient-based",
        "mu_strategy": "adaptive",
        "linear_solver": "mumps",
    }
    if hessian_approximation == "limited-memory":
        solver_options["hessian_approximation"] = "limited-memory"
    opti.solver("ipopt", {"expand": True}, solver_options)
    try:
        solution = opti.solve_limited()
        stats = opti.stats()
        values = solution.value
    except RuntimeError as exc:
        stats = opti.stats()
        message = str(exc)
        try:
            z_value = opti.debug.value(z)
            w_value = opti.debug.value(w)
            s_value = opti.debug.value(progress_fraction)
            warm_out = {
                "states": np.asarray(z_value) * sx[:, None],
                "controls": np.asarray(w_value) * su[:, None],
                "progress": np.asarray(s_value).reshape(-1) * length,
            }
        except RuntimeError:
            warm_out = None
        return {"success": False, "stats": stats, "message": message,
                "warm": warm_out}

    states = np.asarray(values(X), dtype=np.float64)
    controls = np.asarray(values(U), dtype=np.float64)
    progress = np.asarray(values(S), dtype=np.float64).reshape(-1)
    return {"success": bool(stats.get("success", False)), "stats": stats,
            "states": states, "controls": controls, "progress": progress,
            "objective": float(values(objective)), "warm": {
                "states": states, "controls": controls, "progress": progress,
            }}


def replay_with_production_c(library: Any, result: dict[str, Any],
                             track: dict[str, Any], model: dict[str, Any],
                             vehicle: dict[str, Any],
                             extra_wall_clearance_m: float,
                             yaw_residual_model: dict[str, Any] | None = None
                             ) -> dict[str, Any]:
    configure_c_yaw_residual(library, yaw_residual_model)
    states = result["states"]
    controls = result["controls"]
    progress_nodes = result["progress"]
    s_ref = np.asarray(track["s"])
    kappa_ref = np.asarray(track["kappa"])
    left_ref = np.asarray(track["left"])
    right_ref = np.asarray(track["right"])
    length = float(track["length"])
    c_state = states[:, 0].copy()
    c_progress = 0.0
    maximum_state_error = np.zeros(NX, dtype=np.float64)
    maximum_progress_error = 0.0
    min_body_clearance = math.inf
    min_progress_step = math.inf
    invalid_stage: int | None = None
    p = model["parameters"]
    for k in range(controls.shape[1]):
        kappa = float(periodic_linear(c_progress, s_ref, kappa_ref, length))
        c_next, c_ds, _ = production_c_step(library, c_state, controls[:, k], kappa)
        c_progress += c_ds
        expected_state = states[:, k + 1]
        error = c_next - expected_state
        error[1] = math.atan2(math.sin(error[1]), math.cos(error[1]))
        maximum_state_error = np.maximum(maximum_state_error, np.abs(error))
        maximum_progress_error = max(
            maximum_progress_error, abs(c_progress - progress_nodes[k + 1]))
        min_progress_step = min(min_progress_step, c_ds / DT)
        sampled_left = float(periodic_linear(c_progress, s_ref, left_ref, length))
        sampled_right = float(periodic_linear(c_progress, s_ref, right_ref, length))
        sampled_kappa = float(periodic_linear(c_progress, s_ref, kappa_ref, length))
        epsi = c_next[1]
        footprint = max(
            0.5 * float(vehicle["planning_footprint_width_m"]),
            0.5 * float(vehicle["car_width_m"]) * abs(math.cos(epsi))
            + max(float(vehicle["rear_axle_to_front_bumper_m"]),
                  float(vehicle["rear_overhang_m"])) * abs(math.sin(epsi)),
        )
        center_to_wall = min(
            sampled_left - c_next[0], sampled_right + c_next[0])
        min_body_clearance = min(
            min_body_clearance, center_to_wall - footprint
            - float(vehicle["required_wall_clearance_m"])
            - extra_wall_clearance_m)
        if not math.isfinite(c_progress) or c_progress <= progress_nodes[k]:
            invalid_stage = k + 1
            break
        c_state = c_next

    completed = (invalid_stage is None
                 and abs(c_progress - length) <= 2.0e-4
                 and np.max(maximum_state_error) <= 2.0e-4
                 and maximum_progress_error <= 2.0e-4
                 and min_body_clearance >= -2.0e-4)
    return {
        "status": "passed" if completed else "failed",
        "is_simulator_truth_validation": False,
        "steps": int(controls.shape[1]),
        "dt_s": DT,
        "production_c_progress_m": float(c_progress),
        "required_progress_m": length,
        "maximum_progress_error_m": float(maximum_progress_error),
        "minimum_progress_rate_mps": float(min_progress_step),
        "maximum_state_abs_error": {
            name: float(maximum_state_error[index])
            for index, name in enumerate((
                "e_y_m", "e_psi_rad", "u_mps", "v_mps", "r_radps",
                "target_speed_mps", "steering_command_rad",
                "delayed_command_1_rad", "delayed_command_2_rad",
                "actual_steering_rad",
            ))
        },
        "minimum_body_clearance_from_reference_bounds_m": float(min_body_clearance),
        "invalid_stage": invalid_stage,
        "pass_criteria": {
            "full_lap_progress_error_m": 2.0e-4,
            "max_state_error": 2.0e-4,
            "max_progress_error_m": 2.0e-4,
            "minimum_extra_clearance_m": -2.0e-4,
        },
    }


def export_result(out_dir: Path, track: dict[str, Any],
                  result: dict[str, Any],
                  replay: dict[str, Any], steps: int,
                  source_trajectory: Path,
                  yaw_residual_model: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = np.asarray(track["rows"], dtype=np.float64).copy()
    progress = result["progress"]
    states = result["states"]
    path_speed = np.interp(path[:, 0], progress, states[2, :])
    path[:, 5] = path_speed
    s_periodic = np.r_[path[:, 0] - float(track["length"]),
                       path[:, 0], path[:, 0] + float(track["length"])]
    speed_periodic = np.r_[path_speed, path_speed, path_speed]
    derivative = np.gradient(speed_periodic, s_periodic)[len(path):2 * len(path)]
    path[:, 6] = path_speed * derivative
    output_csv = out_dir / "autodrive_discrete25ms_speed_candidate.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# s_m,x_m,y_m,psi_rad,kappa_radpm,velocity_mps,acceleration_mps2,d_left_m,d_right_m\n")
        writer = csv.writer(stream, lineterminator="\n")
        for row in path:
            writer.writerow([f"{value:.9f}" for value in row[:9]])

    nodes_csv = out_dir / "discrete25ms_nodes.csv"
    with nodes_csv.open("w", encoding="utf-8", newline="") as stream:
        fields = ("s_ref_m", "kappa_ref", "ey_m", "epsi_rad", "u_mps", "v_mps",
                  "r_radps", "target_mps", "delta_rad", "steering_command_rad",
                  "delayed_command_1_rad", "delayed_command_2_rad",
                  "qdelta_radps", "qv_mps2")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for k in range(steps + 1):
            kappa = float(periodic_linear(
                result["progress"][k], np.asarray(track["s"]),
                np.asarray(track["kappa"]), float(track["length"])))
            qdelta = (result["controls"][0, k]
                      if k < steps else result["controls"][0, 0])
            qv = (result["controls"][1, k]
                  if k < steps else result["controls"][1, 0])
            x = result["states"][:, k]
            writer.writerow([f"{value:.10f}" for value in (
                result["progress"][k], kappa, x[0], x[1], x[2], x[3], x[4], x[5],
                x[9], x[6], x[7], x[8], qdelta, qv,
            )])

    intervals = np.diff(np.r_[np.asarray(track["s"]), float(track["length"])])
    ref_speed = np.asarray(track["speed"])
    ref_time = float(np.sum(intervals / (0.5 * (ref_speed + np.roll(ref_speed, -1)))))
    candidate_time = float(np.sum(
        intervals / (0.5 * (path_speed + np.roll(path_speed, -1)))))
    report = {
        "schema_version": 1,
        "candidate_kind": "offline fixed-geometry production-discrete speed profile",
        "yaw_residual_candidate_id": (
            yaw_residual_model.get("model_id")
            if yaw_residual_model is not None else None),
        "acceptance": "accepted_for_next_offline_gate" if replay["status"] == "passed"
            else "rejected",
        "simulator_run_authorized": False,
        "source_trajectory": str(source_trajectory.resolve()),
        "source_trajectory_sha256": sha256(source_trajectory),
        "path_geometry_changed": False,
        "length_m": float(track["length"]),
        "fixed_step_s": DT,
        "steps": steps,
        "discrete_model_lap_time_s": steps * DT,
        "source_reference_integrated_speed_time_s": ref_time,
        "candidate_export_integrated_speed_time_s": candidate_time,
        "candidate_minus_source_integrated_time_s": candidate_time - ref_time,
        "speed_mps": {
            "min": float(np.min(path_speed)),
            "max": float(np.max(path_speed)),
            "mean_by_distance_mps": float(float(track["length"]) / candidate_time),
        },
        "recursive_production_c_replay": replay,
        "artifacts": {
            "trajectory_csv": output_csv.name,
            "optimizer_nodes_csv": nodes_csv.name,
        },
        "interpretation": (
            "The C replay checks transcription/model parity only. It is not a "
            "simulator-truth or closed-loop MPC lap-time result. The exported "
            "speed schedule is not approved for racing unless separately "
            "validated through the established batch simulator procedure."
        ),
    }
    (out_dir / "optimization_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True,
                        help="fixed 25 ms horizon(s), attempted in order")
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--yaw-residual-candidate", type=Path,
                        help="optional fitted, support-gated yaw residual JSON")
    parser.add_argument("--mpc-config", type=Path, default=DEFAULT_MPC_CONFIG)
    parser.add_argument("--optimizer-config", type=Path,
                        default=DEFAULT_OPTIMIZER_CONFIG)
    parser.add_argument("--max-iter", type=int, default=650)
    parser.add_argument("--hessian-approximation",
                        choices=("exact", "limited-memory"), default="exact")
    parser.add_argument("--warm-start", type=Path,
                        help="saved rejected_iterate.npz from the same model/path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(step < 100 or step > 400 for step in args.steps):
        raise ValueError("each horizon must be between 100 and 400 stages")
    args.output.mkdir(parents=True, exist_ok=True)
    track = load_trajectory(args.trajectory)
    model = json.loads(args.model.read_text(encoding="utf-8"))
    yaw_residual_model = (
        json.loads(args.yaw_residual_candidate.read_text(encoding="utf-8"))
        if args.yaw_residual_candidate else None)
    optimizer_config = yaml.safe_load(args.optimizer_config.read_text(encoding="utf-8"))
    yaml_match = check_source_config(model, args.mpc_config)
    header_match = check_header_contract(model, ROOT / "f1tenth_mpc/include/mpc_types.h")
    if not yaml_match["matches"] or not header_match["matches"]:
        raise ValueError(f"Production transition parameter mismatch: {yaml_match}, {header_match}")
    dynamic = casadi_function(model, yaw_residual_model)
    c_library = ctypes.CDLL(str(args.library.resolve()))
    setup_c_library(c_library)
    warm: dict[str, np.ndarray] | None = None
    if args.warm_start:
        with np.load(args.warm_start) as saved:
            warm = {name: np.asarray(saved[name], dtype=np.float64)
                    for name in ("states", "controls", "progress")}
    all_reports = []
    with ProductionMpc(args.library, args.mpc_config, args.trajectory):
        configure_c_yaw_residual(c_library, yaw_residual_model)
        for steps in args.steps:
            horizon_dir = args.output / f"N{steps:03d}"
            result = solve_horizon(
                steps, track, model, optimizer_config, dynamic, warm,
                args.max_iter, args.hessian_approximation,
                yaw_residual_model)
            if not result["success"]:
                failure = {
                    "steps": steps,
                    "solver": solver_diagnostics(result["stats"]),
                    "message": result.get("message"),
                    "production_parameters_match": True,
                    "hessian_approximation": args.hessian_approximation,
                }
                horizon_dir.mkdir(parents=True, exist_ok=True)
                if result.get("warm") is not None:
                    np.savez_compressed(
                        horizon_dir / "rejected_iterate.npz", **result["warm"])
                (horizon_dir / "rejected_solver.json").write_text(
                    json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                all_reports.append(failure)
                break
            extra_wall_clearance = float(
                optimizer_config.get("config", optimizer_config)
                .get("track", {}).get("extra_wall_clearance_m", 0.10))
            vehicle = optimizer_config.get("vehicle_model", {})
            replay = replay_with_production_c(
                c_library, result, track, model, vehicle, extra_wall_clearance,
                yaw_residual_model)
            if replay["status"] != "passed":
                failure = {
                    "steps": steps,
                    "solver_status": result["stats"].get("return_status"),
                    "iterations": result["stats"].get("iter_count"),
                    "recursive_production_c_replay": replay,
                    "status": "rejected_recursive_parity",
                }
                horizon_dir.mkdir(parents=True, exist_ok=True)
                (horizon_dir / "rejected_replay.json").write_text(
                    json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                all_reports.append(failure)
                break
            report = export_result(
                horizon_dir, track, result, replay, steps, args.trajectory,
                yaw_residual_model)
            report["solver"] = {
                **solver_diagnostics(result["stats"]),
                "objective": result["objective"],
                "hessian_approximation": args.hessian_approximation,
            }
            (horizon_dir / "optimization_report.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            all_reports.append(report)
            warm = result["warm"]
            print(json.dumps({
                "steps": steps,
                "model_time_s": steps * DT,
                "export_integrated_time_s": report["candidate_export_integrated_time_s"],
                "solver_iterations": report["solver"]["iterations"],
                "recursive_replay": replay["status"],
                "output": str(horizon_dir),
            }), flush=True)
    summary = {
        "trajectory_sha256": sha256(args.trajectory),
        "model_id": model["model_id"],
        "yaw_residual_candidate_id": (
            yaw_residual_model.get("model_id")
            if yaw_residual_model is not None else None),
        "production_config_matches": yaml_match["matches"],
        "production_header_matches": header_match["matches"],
        "horizons": all_reports,
        "simulator_started": False,
    }
    (args.output / "sequence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if all_reports and all(
        report.get("acceptance") == "accepted_for_next_offline_gate"
        for report in all_reports
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
