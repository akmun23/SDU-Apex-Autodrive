from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import time
from typing import Any

import casadi as ca
import numpy as np

from .track import Track, load_centerline_csv
from .unity_wheel_model import (
    UnityWheelModel,
    suspension_loads_casadi,
    suspension_loads_numpy,
    suspension_travel_casadi,
    tire_forces_casadi,
    physx_no_drive_wheel_speed_interval_casadi,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    GRAVITY_MPS2,
    WHEEL_RADIUS_M,
    WHEEL_X_M,
    WHEEL_Y_M,
    ackermann_angles,
)
from tools.vehicle_dynamics_learning.physx_no_drive_source import (
    UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS,
    UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS,
    UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS,
)
from tools.racing.mpc_vehicle_geometry import (
    footprint_half_width_casadi,
    footprint_half_width_numpy,
)

FEASIBILITY_MAX_ITERATIONS = 600
TIME_OPTIMIZATION_MAX_ITERATIONS = 2000
NLP_FEASIBILITY_TOLERANCE = 1.0e-3
NEAR_FEASIBLE_WARM_START_MAX_VIOLATION = 5.0e-3
DIAGNOSTIC_ITERATE_CHECKPOINT_MAX_VIOLATION = 0.1
DEFAULT_TIME_TRUST_REGION_FACTORS = (4.0, 8.0, 16.0, 1.0e6)


def should_retain_diagnostic_iterate(
        finite: bool, feasible_incumbent: bool,
        max_constraint_violation: float | None) -> bool:
    """Retain a bounded failed iterate only for explicit solver continuation."""
    return bool(
        finite and not feasible_incumbent
        and max_constraint_violation is not None
        and math.isfinite(float(max_constraint_violation))
        and float(max_constraint_violation)
        <= DIAGNOSTIC_ITERATE_CHECKPOINT_MAX_VIOLATION)


def validate_time_trust_region_factors(
        factors: tuple[float, ...] | list[float]
        ) -> tuple[float, ...]:
    """Validate a strictly increasing sequence of time-stage trust radii."""
    values = tuple(float(value) for value in factors)
    if (not values or not np.isfinite(values).all()
            or any(value <= 1.0 for value in values)
            or any(right <= left for left, right in zip(values, values[1:]))):
        raise ValueError(
            "time trust-region factors must be finite, greater than one, "
            "and strictly increasing")
    return values


def validate_minimum_wheel_normal_load(
        minimum_load_n: float, static_wheel_loads_n: tuple[float, ...]
        ) -> float:
    """Keep each wheel loaded above the chosen floor and below static load."""
    static_loads = np.asarray(static_wheel_loads_n, dtype=np.float64)
    if (not math.isfinite(minimum_load_n) or minimum_load_n < 0.05
            or static_loads.shape != (4,)
            or not np.isfinite(static_loads).all()
            or minimum_load_n >= float(np.min(static_loads))):
        raise ValueError(
            "minimum wheel normal load must be finite, at least 0.05 N, "
            "and below every static wheel load")
    return float(minimum_load_n)


STATE_NAMES = (
    "ey_m", "epsi_rad", "u_mps", "v_mps", "yaw_rate_radps",
    "steering_rad", "body_height_m", "body_height_rate_mps",
    "roll_rad", "roll_rate_radps", "pitch_rad", "pitch_rate_radps",
)
WHEEL_STATE_NAMES = (
    "wheel_omega_fl_radps", "wheel_omega_fr_radps",
    "wheel_omega_rl_radps", "wheel_omega_rr_radps",
)


def _footprint_half_width(epsi, model: UnityWheelModel):
    del model  # The production MPC envelope is the shared geometry contract.
    return footprint_half_width_casadi(epsi, ca)


def _path_geometry(track: Track, ey: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = -np.sin(track.psi)
    ny = np.cos(track.psi)
    x = track.x + ey * nx
    y = track.y + ey * ny
    xp = np.roll(x, -1)
    yp = np.roll(y, -1)
    xm = np.roll(x, 1)
    ym = np.roll(y, 1)
    dx = xp - xm
    dy = yp - ym
    d2x = xp - 2.0 * x + xm
    d2y = yp - 2.0 * y + ym
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1.0e-10)
    curvature = 4.0 * (dx * d2y - dy * d2x) / denom
    segment = np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))
    return x, y, curvature


def _solver_status(stats: dict[str, Any]) -> dict[str, Any]:
    return_status = str(stats.get("return_status", "unknown"))
    return {
        "success": bool(stats.get("success", False)),
        "return_status": return_status,
        "ipopt_converged": return_status in {
            "Solve_Succeeded", "Solved_To_Acceptable_Level"},
        "iterations": int(stats.get("iter_count", 0)),
    }


def _periodic_suspension_seed(model: UnityWheelModel, ds: np.ndarray,
                              sdot: np.ndarray, forward_speed: np.ndarray,
                              lateral_speed: np.ndarray, ax: np.ndarray,
                              ay: np.ndarray, initial: np.ndarray
                              ) -> tuple[np.ndarray, int, float]:
    """Make a periodic 1 kHz RK4 seed for heave, roll, pitch and their rates.

    The forcing is the current optimizer seed's planar acceleration. This is
    only an NLP initializer; the final OCP still predicts all states jointly.
    """
    n = len(ds)
    interval_s = ds * 0.5 * (1.0 / sdot + 1.0 / np.roll(sdot, -1))

    def derivative(state: np.ndarray, inputs: tuple[float, float, float, float]
                    ) -> np.ndarray:
        ax_i, ay_i, u_i, v_i = inputs
        height, height_rate, roll, roll_rate, pitch, pitch_rate = state
        loads = suspension_loads_numpy(
            model, height, height_rate, roll, roll_rate, pitch, pitch_rate)
        fx = model.mass_kg * (ax_i + model.drag_per_s * u_i)
        fy = model.mass_kg * (ay_i + model.drag_per_s * v_i)
        roll_moment = sum(-WHEEL_Y_M[i] * loads[i] for i in range(4)) \
            - model.body_com_height_m * fy
        pitch_moment = sum(-WHEEL_X_M[i] * loads[i] for i in range(4)) \
            - model.body_com_height_m * fx
        return np.asarray((
            height_rate,
            float(np.sum(loads)) / model.mass_kg - GRAVITY_MPS2
            - model.drag_per_s * height_rate,
            roll_rate,
            roll_moment / model.inertia_roll_kgm2 - 0.1 * roll_rate,
            pitch_rate,
            pitch_moment / model.inertia_pitch_kgm2 - 0.1 * pitch_rate,
        ), dtype=np.float64)

    def run_lap(start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = start.copy()
        node_states = np.empty((6, n), dtype=np.float64)
        for k in range(n):
            kp = (k + 1) % n
            node_states[:, k] = state
            duration = float(interval_s[k])
            steps = max(1, int(math.ceil(duration / 0.001)))
            h = duration / steps
            input_start = np.asarray((ax[k], ay[k], forward_speed[k],
                                      lateral_speed[k]), dtype=np.float64)
            input_delta = (np.asarray((ax[kp], ay[kp], forward_speed[kp],
                                       lateral_speed[kp]), dtype=np.float64)
                           - input_start)
            for j in range(steps):
                alpha0 = j / steps
                alpham = (j + 0.5) / steps
                alpha1 = (j + 1.0) / steps
                f0 = input_start + alpha0 * input_delta
                fm = input_start + alpham * input_delta
                f1 = input_start + alpha1 * input_delta
                k1 = derivative(state, tuple(f0))
                k2 = derivative(state + 0.5 * h * k1, tuple(fm))
                k3 = derivative(state + 0.5 * h * k2, tuple(fm))
                k4 = derivative(state + h * k3, tuple(f1))
                state += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state, node_states

    state = np.asarray(initial, dtype=np.float64).copy()
    scales = np.asarray((0.01, 0.5, 0.1, 2.0, 0.1, 2.0))
    node_states = np.empty((6, n), dtype=np.float64)
    residual = math.inf
    for lap in range(1, 41):
        end_state, node_states = run_lap(state)
        residual = float(np.max(np.abs((end_state - state) / scales)))
        state = end_state
        if residual < 1.0e-4:
            return node_states, lap, residual
    return node_states, 40, residual


def _constraint_diagnostics(g_values: np.ndarray, lower: np.ndarray,
                            upper: np.ndarray,
                            labels: list[str]) -> dict[str, Any]:
    g_values = np.asarray(g_values, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (g_values.shape == lower.shape == upper.shape
            and len(labels) == len(g_values)):
        raise ValueError("constraint diagnostic vectors do not align")
    violation = np.maximum(np.maximum(lower - g_values,
                                      g_values - upper), 0.0)
    violation[~np.isfinite(g_values)] = np.inf
    worst = np.argsort(violation)[-20:][::-1]
    return {
        "max_constraint_violation": float(np.max(violation, initial=0.0)),
        "nonfinite_constraint_count": int(np.count_nonzero(~np.isfinite(g_values))),
        "worst_constraints": [
            {"label": labels[int(i)], "index": int(i),
             "violation": float(violation[i]),
             "value": (float(g_values[i]) if np.isfinite(g_values[i]) else None),
             "lower": (float(lower[i]) if np.isfinite(lower[i]) else None),
             "upper": (float(upper[i]) if np.isfinite(upper[i]) else None)}
            for i in worst if violation[i] > 0.0],
    }


def _brake_phases(track: Track, brake_seed: np.ndarray,
                  minimum_nodes: int = 6
                  ) -> tuple[list[bool], list[int], np.ndarray, list[tuple[int, int]]]:
    """Turn a measured mode seed into movable, continuous-length phases.

    Only the alternating drive/brake phase count and order are seeded. IPOPT
    optimizes every phase length; no captured switch location is fixed.
    """
    mask = np.asarray(brake_seed, dtype=bool)
    n = track.count
    if mask.shape != (n,) or mask[0] or mask[-1]:
        raise ValueError("the gate must lie in a drive phase and brake seed must match the track")
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    bounds = np.r_[0, changes, n]
    modes: list[bool] = []
    nodes: list[int] = []
    lengths: list[float] = []
    ranges: list[tuple[int, int]] = []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        modes.append(bool(mask[begin]))
        nodes.append(int(end - begin))
        lengths.append(float(np.sum(track.ds[begin:end])))
        ranges.append((int(begin), int(end)))
    if any(modes[i] == modes[i - 1] for i in range(1, len(modes))):
        raise ValueError("brake seed did not produce alternating phases")

    # Preserve the full spatial mesh while giving brief measured phases enough
    # knots to move and resize continuously without creating a coarse segment.
    deficits = 0
    for i, count in enumerate(nodes):
        if count < minimum_nodes:
            deficits += minimum_nodes - count
            nodes[i] = minimum_nodes
    drive_donors = sorted(
        (i for i, mode in enumerate(modes) if not mode),
        key=lambda i: nodes[i], reverse=True)
    for donor in drive_donors:
        available = max(0, nodes[donor] - minimum_nodes)
        take = min(available, deficits)
        nodes[donor] -= take
        deficits -= take
        if deficits == 0:
            break
    if deficits:
        raise ValueError("insufficient drive-phase knots to resolve brake phases")
    if sum(nodes) != n:
        raise AssertionError("phase mesh reallocation changed total knot count")
    return modes, nodes, np.asarray(lengths, dtype=np.float64), ranges


def _periodic_interpolants(
        track: Track,
        tracking_bias_m: np.ndarray | None = None,
        path_reference_offset_m: np.ndarray | None = None
        ) -> dict[str, Any]:
    grid = np.r_[np.asarray(track.s, dtype=np.float64), track.length]
    bias = (np.zeros(track.count, dtype=np.float64)
            if tracking_bias_m is None
            else np.asarray(tracking_bias_m, dtype=np.float64))
    if bias.shape != (track.count,) or not np.isfinite(bias).all():
        raise ValueError("tracking path bias must be one finite value per track node")
    reference_offset = (
        np.zeros(track.count, dtype=np.float64)
        if path_reference_offset_m is None
        else np.asarray(path_reference_offset_m, dtype=np.float64))
    if (reference_offset.shape != (track.count,)
            or not np.isfinite(reference_offset).all()):
        raise ValueError(
            "path-reference offsets must be finite, one per track node")
    periodic = {
        "x": track.x,
        "y": track.y,
        "sin_psi": np.sin(track.psi),
        "cos_psi": np.cos(track.psi),
        "kappa": track.kappa,
        "left": track.left,
        "right": track.right,
        "tracking_bias": bias,
        "path_reference_offset": reference_offset,
    }
    suffix = str(id(track))
    return {
        key: ca.interpolant(
            f"free_path_{key}_{suffix}", "linear", [grid],
            np.r_[np.asarray(values, dtype=np.float64), float(values[0])])
        for key, values in periodic.items()
    }


def _path_geometry_at(track: Track, station: np.ndarray, ey: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray]:
    station = np.asarray(station, dtype=np.float64)
    extended_s = np.r_[track.s, track.length]

    def periodic(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.interp(station, extended_s, np.r_[values, values[0]])

    px = periodic(track.x) - periodic(np.sin(track.psi)) * ey
    py = periodic(track.y) + periodic(np.cos(track.psi)) * ey
    # Calculate curvature from the actual variable-station path coordinates.
    ax = px - np.roll(px, 1)
    ay = py - np.roll(py, 1)
    bx = np.roll(px, -1) - px
    by = np.roll(py, -1) - py
    chord_product = np.maximum(
        np.hypot(ax, ay) * np.hypot(bx, by)
        * np.hypot(ax + bx, ay + by), 1.0e-10)
    curvature = 2.0 * (ax * by - ay * bx) / chord_product
    route_arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
    return px, py, curvature, route_arc


def _normalize_gate_state_anchor(anchor: dict[str, Any] | None
                                 ) -> dict[str, Any] | None:
    if anchor is None:
        return None
    scalar_names = ("ey_m", "epsi_rad", "u_mps", "v_mps",
                    "yaw_rate_radps", "steering_rad")
    missing = [name for name in scalar_names + ("wheel_omega_radps",)
               if name not in anchor]
    if missing:
        raise ValueError(f"gate-state anchor is missing fields: {missing}")
    normalized: dict[str, Any] = {}
    for name in scalar_names:
        value = float(anchor[name])
        if not math.isfinite(value):
            raise ValueError(f"gate-state anchor {name} must be finite")
        normalized[name] = value
    wheel_omega = np.asarray(anchor["wheel_omega_radps"], dtype=np.float64)
    if wheel_omega.shape != (4,) or not np.isfinite(wheel_omega).all():
        raise ValueError("gate-state wheel speeds must be four finite values")
    normalized["wheel_omega_radps"] = wheel_omega.tolist()
    return normalized


def solve_bottom_up(track: Track, model: UnityWheelModel,
                    *, wall_clearance_m: float = 0.15,
                    max_speed_mps: float = 12.0,
                    min_speed_mps: float = 0.35,
                    brake_mode_seed: np.ndarray | None = None,
                    brake_mode_seed_source: str = "drive-mode initialization",
                    initial_state_seed: dict[str, np.ndarray] | None = None,
                    initial_state_seed_s: np.ndarray | None = None,
                    initial_state_seed_is_model_consistent: bool = False,
                    minimum_trust_region_reference: float = 1.0,
                    phase_lengths_initial: np.ndarray | None = None,
                    gate_state_anchor: dict[str, Any] | None = None,
                    gate_lateral_tolerance_m: float = 0.0,
                    gate_steering_tolerance_rad: float = 0.0,
                    max_body_path_heading_error_rad: float | None = None,
                    max_abs_lateral_speed_mps: float | None = None,
                    minimum_wheel_normal_load_n: float = 0.05,
                    path_reference_offset_m: np.ndarray | None = None,
                    max_path_reference_deviation_m: float | None = None,
                    feasibility_trust_region: float = 25.0,
                    time_trust_region_factors: tuple[float, ...] = (
                        DEFAULT_TIME_TRUST_REGION_FACTORS),
                    tracking_bias_m: np.ndarray | None = None,
                    feasibility_max_iterations: int = FEASIBILITY_MAX_ITERATIONS,
                    time_optimization_max_iterations: int = TIME_OPTIMIZATION_MAX_ITERATIONS,
                    time_hessian_approximation: str = "limited-memory",
                    throttle_fall_rate_limit_norm_per_s: float | None = None,
                    diagnose_initial_only: bool = False,
                    print_level: int = 4) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Jointly optimize a free path, motion states, throttle, and brake phases.

    Unity changes discontinuously between CAWB full braking at zero throttle
    and CAWD positive drive. Instead of 257 independent integer decisions,
    measured training-only mode data initializes the count/order of contiguous
    phases. IPOPT optimizes their lengths and all path/state/control variables;
    captured phase locations are not constraints. The vehicle force model is
    the Unity-source-calibrated model described in ``UnityWheelModel``.
    """
    n = track.count
    gate_state_anchor = _normalize_gate_state_anchor(gate_state_anchor)
    if (not math.isfinite(gate_lateral_tolerance_m)
            or gate_lateral_tolerance_m < 0.0):
        raise ValueError("gate lateral tolerance must be finite and non-negative")
    if (not math.isfinite(gate_steering_tolerance_rad)
            or gate_steering_tolerance_rad < 0.0):
        raise ValueError(
            "gate steering tolerance must be finite and non-negative")
    if (max_body_path_heading_error_rad is not None
            and (not math.isfinite(max_body_path_heading_error_rad)
                 or max_body_path_heading_error_rad <= 0.0
                 or max_body_path_heading_error_rad > 0.75)):
        raise ValueError(
            "body/path heading-error limit must be in (0, 0.75] rad")
    if (max_abs_lateral_speed_mps is not None
            and (not math.isfinite(max_abs_lateral_speed_mps)
                 or max_abs_lateral_speed_mps <= 0.0
                 or max_abs_lateral_speed_mps > 2.0)):
        raise ValueError(
            "absolute lateral-speed limit must be in (0, 2] m/s")
    if (not math.isfinite(feasibility_trust_region)
            or feasibility_trust_region <= 0.0):
        raise ValueError("feasibility trust region must be finite and positive")
    minimum_wheel_normal_load_n = validate_minimum_wheel_normal_load(
        minimum_wheel_normal_load_n, model.static_normal_load_n)
    if ((path_reference_offset_m is None)
            != (max_path_reference_deviation_m is None)):
        raise ValueError(
            "path reference offsets and maximum deviation must be provided together")
    if max_path_reference_deviation_m is not None:
        if (not math.isfinite(max_path_reference_deviation_m)
                or max_path_reference_deviation_m <= 0.0
                or max_path_reference_deviation_m > 2.0):
            raise ValueError(
                "maximum path-reference deviation must be in (0, 2] m")
        path_reference_offset_m = np.asarray(
            path_reference_offset_m, dtype=np.float64)
        if (path_reference_offset_m.shape != (track.count,)
                or not np.isfinite(path_reference_offset_m).all()):
            raise ValueError(
                "path-reference offsets must be finite, one per track node")
    time_trust_region_factors = validate_time_trust_region_factors(
        time_trust_region_factors)
    if (not isinstance(feasibility_max_iterations, int)
            or feasibility_max_iterations <= 0):
        raise ValueError("feasibility iteration budget must be a positive integer")
    if (not isinstance(time_optimization_max_iterations, int)
            or time_optimization_max_iterations <= 0):
        raise ValueError("time-optimization iteration budget must be a positive integer")
    if time_hessian_approximation not in {"limited-memory", "exact"}:
        raise ValueError("time Hessian approximation must be limited-memory or exact")
    if brake_mode_seed is None:
        brake_mode_seed = np.zeros(n, dtype=bool)
    brake_mode_seed = np.asarray(brake_mode_seed, dtype=bool)
    if (not math.isfinite(minimum_trust_region_reference)
            or minimum_trust_region_reference <= 0.0):
        raise ValueError("minimum trust-region reference must be positive")
    if (throttle_fall_rate_limit_norm_per_s is not None
            and (not math.isfinite(throttle_fall_rate_limit_norm_per_s)
                 or throttle_fall_rate_limit_norm_per_s <= 0.0)):
        raise ValueError("throttle fall-rate limit must be finite and positive")
    phase_modes, phase_nodes, phase_seed_lengths, phase_ranges = _brake_phases(
        track, brake_mode_seed)
    phase_count = len(phase_modes)
    if phase_lengths_initial is None:
        phase_initial_lengths = phase_seed_lengths.copy()
    else:
        phase_initial_lengths = np.asarray(phase_lengths_initial,
                                           dtype=np.float64)
        if (phase_initial_lengths.shape != (phase_count,)
                or not np.isfinite(phase_initial_lengths).all()
                or np.any(phase_initial_lengths <= 0.0)
                or not np.isclose(np.sum(phase_initial_lengths), track.length,
                                  rtol=0.0, atol=1.0e-5)):
            raise ValueError("invalid initial brake/drive phase lengths")
    node_counts = np.asarray(phase_nodes, dtype=np.int64)
    n = int(np.sum(node_counts))
    if tracking_bias_m is None:
        tracking_bias_values = np.zeros(track.count, dtype=np.float64)
    else:
        tracking_bias_values = np.asarray(tracking_bias_m, dtype=np.float64)
        if (tracking_bias_values.shape != (track.count,)
                or not np.isfinite(tracking_bias_values).all()):
            raise ValueError(
                "tracking path bias must be one finite value per track node")
    interp = _periodic_interpolants(
        track, tracking_bias_values, path_reference_offset_m)
    opti = ca.Opti()
    x = opti.variable(16, n)
    qdelta = opti.variable(n)
    throttle = opti.variable(n)
    phase_lengths = opti.variable(phase_count)
    normal_load = opti.variable(4, n)
    accel_xy = opti.variable(2, n)
    constraint_labels: list[str] = []

    def add_constraint(expression, label: str | list[str]) -> None:
        count = int(expression.numel())
        opti.subject_to(expression)
        if isinstance(label, str):
            constraint_labels.extend([label] * count)
        elif len(label) == count:
            constraint_labels.extend(label)
        else:
            raise ValueError("constraint labels do not match expression size")

    # Each phase keeps at least a 5 mm interval and at most 250 mm between
    # knots. This is a numerical mesh bound, not a vehicle speed/steering cap.
    for phase, count in enumerate(node_counts):
        add_constraint(phase_lengths[phase] >= 0.005 * int(count),
                       f"phase_length_min_{phase}")
        add_constraint(phase_lengths[phase] <= 0.25 * int(count),
                       f"phase_length_max_{phase}")
    add_constraint(ca.sum1(phase_lengths) == track.length,
                   "periodic_phase_lengths_sum")
    opti.set_initial(phase_lengths, phase_initial_lengths)

    station_nodes = []
    step_nodes = []
    step_nodes_seed = []
    brake_modes = []
    station_seed = []
    station_start = 0
    seed_start = 0.0
    for phase, (mode, count) in enumerate(zip(phase_modes, node_counts)):
        count = int(count)
        for local in range(count):
            fraction = local / count
            station_nodes.append(station_start + phase_lengths[phase] * fraction)
            step_nodes.append(phase_lengths[phase] / count)
            step_nodes_seed.append(float(phase_initial_lengths[phase]) / count)
            brake_modes.append(mode)
            station_seed.append(seed_start + phase_initial_lengths[phase] * fraction)
        station_start += phase_lengths[phase]
        seed_start += float(phase_initial_lengths[phase])
    station_seed = np.asarray(station_seed, dtype=np.float64)
    if station_seed.shape != (n,) or not np.all(np.diff(station_seed) > 0.0):
        raise ValueError("phase seed does not create an increasing spatial mesh")
    brake_modes_array = np.asarray(brake_modes, dtype=bool)

    seed_sample_s = (track.s if initial_state_seed_s is None
                     else np.asarray(initial_state_seed_s, dtype=np.float64))
    if (seed_sample_s.ndim != 1 or not np.isfinite(seed_sample_s).all()
            or len(seed_sample_s) < 2 or np.any(np.diff(seed_sample_s) <= 0.0)
            or seed_sample_s[0] < -1.0e-8
            or seed_sample_s[-1] >= track.length):
        raise ValueError("initial state seed stations must span [0, track.length)")

    def sample_periodic(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            return np.interp(station_seed, np.r_[seed_sample_s, track.length],
                             np.r_[values, values[0]])
        return np.vstack([sample_periodic(row) for row in values])

    ey, epsi, u, v, yaw_rate = (x[i, :] for i in range(5))
    wheel_omega = [x[i, :] for i in range(5, 9)]
    steering = x[9, :]
    body_height, body_height_rate, roll, roll_rate, pitch, pitch_rate = (
        x[i, :] for i in range(10, 16))

    # Initial guesses use the measured source run where available; otherwise
    # they are generated from centerline curvature and configured tire peaks.
    if initial_state_seed is not None:
        expected = {
            "ey_m": (len(seed_sample_s),),
            "epsi_rad": (len(seed_sample_s),),
            "u_mps": (len(seed_sample_s),),
            "v_mps": (len(seed_sample_s),),
            "yaw_rate_radps": (len(seed_sample_s),),
            "wheel_omega_radps": (4, len(seed_sample_s)),
            "steering_rad": (len(seed_sample_s),),
            "steering_rate_radps": (len(seed_sample_s),),
            "throttle_norm": (len(seed_sample_s),),
            "ax_mps2": (len(seed_sample_s),),
            "ay_mps2": (len(seed_sample_s),),
            "normal_load_n": (4, len(seed_sample_s)),
            "body_height_m": (len(seed_sample_s),),
            "body_height_rate_mps": (len(seed_sample_s),),
            "roll_rad": (len(seed_sample_s),),
            "roll_rate_radps": (len(seed_sample_s),),
            "pitch_rad": (len(seed_sample_s),),
            "pitch_rate_radps": (len(seed_sample_s),),
        }
        optional_body_state = {
            "body_height_m": np.full(
                len(seed_sample_s), model.suspension_reference_body_height_m),
            "body_height_rate_mps": np.zeros(len(seed_sample_s)),
            "roll_rad": np.zeros(len(seed_sample_s)),
            "roll_rate_radps": np.zeros(len(seed_sample_s)),
            "pitch_rad": np.zeros(len(seed_sample_s)),
            "pitch_rate_radps": np.zeros(len(seed_sample_s)),
        }
        seed_values = {}
        for name, shape in expected.items():
            raw = initial_state_seed.get(name, optional_body_state.get(name))
            if raw is None:
                raise ValueError(f"captured initial guess lacks {name}")
            values = np.asarray(raw, dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"invalid captured initial guess for {name}")
            seed_values[name] = values
        seed_ey = sample_periodic(seed_values["ey_m"])
        seed_epsi = sample_periodic(seed_values["epsi_rad"])
        seed_u = sample_periodic(seed_values["u_mps"])
        seed_v = sample_periodic(seed_values["v_mps"])
        seed_yaw = sample_periodic(seed_values["yaw_rate_radps"])
        seed_omega = sample_periodic(seed_values["wheel_omega_radps"])
        seed_steering = sample_periodic(seed_values["steering_rad"])
        seed_qdelta = sample_periodic(seed_values["steering_rate_radps"])
        seed_throttle = sample_periodic(seed_values["throttle_norm"])
        seed_ax = sample_periodic(seed_values["ax_mps2"])
        seed_ay = sample_periodic(seed_values["ay_mps2"])
        seed_body_height = sample_periodic(seed_values["body_height_m"])
        seed_body_height_rate = sample_periodic(
            seed_values["body_height_rate_mps"])
        seed_roll = sample_periodic(seed_values["roll_rad"])
        seed_roll_rate = sample_periodic(seed_values["roll_rate_radps"])
        seed_pitch = sample_periodic(seed_values["pitch_rad"])
        seed_pitch_rate = sample_periodic(seed_values["pitch_rate_radps"])
    else:
        kappa_seed = sample_periodic(np.asarray(track.kappa))
        curve_peak = [max(curve[1], curve[3]) * curve[4]
                      for curve in model.lateral_curve]
        seed_lateral_accel = sum(
            model.static_normal_load_n[i] * curve_peak[i]
            for i in range(4)) / model.mass_kg
        seed_u = np.clip(np.minimum(
            min(5.5, max_speed_mps - 0.2),
            np.sqrt(seed_lateral_accel * 0.72 /
                    np.maximum(np.abs(kappa_seed), 1.0e-5))),
            min_speed_mps, max_speed_mps)
        seed_steering = np.clip(
            np.arctan(model.wheelbase_m * kappa_seed),
            -0.95 * model.steering_limit_rad,
            0.95 * model.steering_limit_rad)
        seed_v = seed_u * (0.15532 * kappa_seed
                           - 0.025 * np.sign(kappa_seed))
        course_speed = np.hypot(seed_u, seed_v)
        seed_yaw = course_speed * kappa_seed
        seed_epsi = -np.arctan2(seed_v, seed_u)
        seed_ey = np.zeros(n)
        seed_body_height = np.full(
            n, model.suspension_reference_body_height_m)
        seed_body_height_rate = np.zeros(n)
        seed_roll = np.zeros(n)
        seed_roll_rate = np.zeros(n)
        seed_pitch = np.zeros(n)
        seed_pitch_rate = np.zeros(n)
        seed_omega = np.empty((4, n), dtype=np.float64)
        for k in range(n):
            steer = ackermann_angles(float(seed_steering[k]))
            for i in range(4):
                vx_wheel = seed_u[k] - seed_yaw[k] * WHEEL_Y_M[i]
                vy_wheel = seed_v[k] + seed_yaw[k] * WHEEL_X_M[i]
                forward = (np.cos(steer[i]) * vx_wheel
                           + np.sin(steer[i]) * vy_wheel)
                seed_omega[i, k] = forward / model.wheel_radius_m
        seed_qdelta = (np.gradient(seed_steering, station_seed)
                       * course_speed)
        seed_throttle = np.clip(
            4.0 * model.wheel_damping_nms * np.mean(seed_omega, axis=0)
            / model.motor_torque_nm, 0.015, 0.95)
        seed_ax = -model.drag_per_s * seed_u
        seed_ay = seed_u * seed_yaw

    computed_seed_load = np.column_stack([
        suspension_loads_numpy(
            model, seed_body_height[k], seed_body_height_rate[k],
            seed_roll[k], seed_roll_rate[k], seed_pitch[k],
            seed_pitch_rate[k])
        for k in range(n)])
    if (initial_state_seed is not None
            and initial_state_seed_is_model_consistent):
        # Preserve the algebraic load variables when resuming a solver
        # iterate. Recomputing them from the body states changes the saved
        # point whenever its spring/damper equalities retain a small residual.
        seed_load = sample_periodic(seed_values["normal_load_n"])
        if (seed_load.shape != (4, n) or not np.isfinite(seed_load).all()
                or np.any(seed_load <= 0.0)):
            raise ValueError("model-consistent normal-load seed is invalid")
    else:
        seed_load = computed_seed_load
    suspension_seed_summary = {
        "method": "warm-start body state",
        "periodic_laps": 0,
        "fixed_point_residual_scaled": None,
    }

    # For a fresh guess, initialize locked brake/transition wheel states at
    # zero. A model-consistent solver checkpoint instead retains its exact
    # wheel and throttle variables, including any small current NLP defect.
    mode_transition_nodes = brake_modes_array != np.roll(brake_modes_array, 1)
    if not initial_state_seed_is_model_consistent:
        seed_omega[:, brake_modes_array | mode_transition_nodes] = 0.0
        seed_throttle[brake_modes_array] = 0.0
        seed_throttle[~brake_modes_array] = np.maximum(
            0.015, seed_throttle[~brake_modes_array])
    if gate_state_anchor is not None:
        anchored_wheels = np.asarray(
            gate_state_anchor["wheel_omega_radps"], dtype=np.float64)
        if ((brake_modes_array[0] or mode_transition_nodes[0])
                and np.any(np.abs(anchored_wheels) > 1.0e-4)):
            raise ValueError(
                "measured gate wheel speeds conflict with a locked-wheel "
                "brake-mode boundary")
        seed_ey[0] = gate_state_anchor["ey_m"]
        seed_epsi[0] = gate_state_anchor["epsi_rad"]
        seed_u[0] = gate_state_anchor["u_mps"]
        seed_v[0] = gate_state_anchor["v_mps"]
        seed_yaw[0] = gate_state_anchor["yaw_rate_radps"]
        seed_omega[:, 0] = anchored_wheels
        seed_steering[0] = np.clip(
            seed_steering[0],
            gate_state_anchor["steering_rad"] - gate_steering_tolerance_rad,
            gate_state_anchor["steering_rad"] + gate_steering_tolerance_rad)
    opti.set_initial(ey, seed_ey)
    opti.set_initial(epsi, seed_epsi)
    opti.set_initial(u, seed_u)
    opti.set_initial(v, seed_v)
    opti.set_initial(yaw_rate, seed_yaw)
    for i in range(4):
        opti.set_initial(wheel_omega[i], seed_omega[i])
    opti.set_initial(steering, seed_steering)
    opti.set_initial(body_height, seed_body_height)
    opti.set_initial(body_height_rate, seed_body_height_rate)
    opti.set_initial(roll, seed_roll)
    opti.set_initial(roll_rate, seed_roll_rate)
    opti.set_initial(pitch, seed_pitch)
    opti.set_initial(pitch_rate, seed_pitch_rate)
    opti.set_initial(qdelta, seed_qdelta)
    opti.set_initial(throttle, seed_throttle)
    opti.set_initial(normal_load, seed_load)
    opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))
    # With no measured gate anchor, all states—including Frenet offset and
    # heading—are periodic decision variables. A warm-up gate equality is
    # applied only when the caller explicitly supplies a measured anchor.
    if gate_state_anchor is not None:
        if gate_lateral_tolerance_m == 0.0:
            add_constraint(ey[0] == gate_state_anchor["ey_m"],
                           "gate_lateral_offset")
        else:
            add_constraint(
                ey[0] >= gate_state_anchor["ey_m"] - gate_lateral_tolerance_m,
                "gate_lateral_offset_lower")
            add_constraint(
                ey[0] <= gate_state_anchor["ey_m"] + gate_lateral_tolerance_m,
                "gate_lateral_offset_upper")
        add_constraint(epsi[0] == gate_state_anchor["epsi_rad"],
                       "gate_heading_error")
        add_constraint(u[0] == gate_state_anchor["u_mps"],
                       "gate_forward_speed")
        add_constraint(v[0] == gate_state_anchor["v_mps"],
                       "gate_lateral_speed")
        add_constraint(yaw_rate[0] == gate_state_anchor["yaw_rate_radps"],
                       "gate_yaw_rate")
        if gate_steering_tolerance_rad == 0.0:
            add_constraint(steering[0] == gate_state_anchor["steering_rad"],
                           "gate_steering_angle")
        else:
            add_constraint(
                steering[0] >= (gate_state_anchor["steering_rad"]
                                 - gate_steering_tolerance_rad),
                "gate_steering_angle_lower")
            add_constraint(
                steering[0] <= (gate_state_anchor["steering_rad"]
                                 + gate_steering_tolerance_rad),
                "gate_steering_angle_upper")
        for wheel in range(4):
            add_constraint(
                wheel_omega[wheel][0]
                == gate_state_anchor["wheel_omega_radps"][wheel],
                f"gate_wheel_omega_{wheel}")

    spatial_rhs = []
    time_density = []
    slip_x_nodes = []
    slip_y_nodes = []
    force_nodes = []
    wheel_tire_torque_nodes = []
    body_path_heading_error_nodes = []
    for k in range(n):
        station = station_nodes[k]
        curvature = interp["kappa"](station)
        left = interp["left"](station)
        right = interp["right"](station)
        predicted_actual_ey = ey[k] + interp["tracking_bias"](station)
        den = 1.0 - curvature * ey[k]
        sdot = (u[k] * ca.cos(epsi[k]) - v[k] * ca.sin(epsi[k])) / den
        add_constraint(den >= 0.25, "frenet_denominator")
        add_constraint(sdot >= 0.30, "path_speed")
        add_constraint(opti.bounded(min_speed_mps, u[k], max_speed_mps),
                       "forward_speed")
        add_constraint(opti.bounded(-0.75, epsi[k], 0.75), "heading_error")
        if max_abs_lateral_speed_mps is not None:
            add_constraint(opti.bounded(
                -max_abs_lateral_speed_mps, v[k], max_abs_lateral_speed_mps),
                f"lateral_velocity_trackability_k{k}")
        if max_path_reference_deviation_m is not None:
            reference_ey = interp["path_reference_offset"](station)
            add_constraint(
                ey[k] - reference_ey <= max_path_reference_deviation_m,
                f"path_reference_deviation_upper_k{k}")
            add_constraint(
                ey[k] - reference_ey >= -max_path_reference_deviation_m,
                f"path_reference_deviation_lower_k{k}")
        add_constraint(opti.bounded(-model.steering_limit_rad,
                                    steering[k], model.steering_limit_rad),
                       "steering_angle")
        add_constraint(opti.bounded(-model.steering_rate_radps,
                                    qdelta[k], model.steering_rate_radps),
                       "steering_rate")
        footprint = _footprint_half_width(epsi[k], model)
        add_constraint(predicted_actual_ey <= left - wall_clearance_m - footprint,
                       f"left_track_boundary_k{k}")
        add_constraint(predicted_actual_ey >= -right + wall_clearance_m + footprint,
                       f"right_track_boundary_k{k}")
        if brake_modes_array[k]:
            add_constraint(throttle[k] == 0.0, "unity_exact_zero_cawb_brake")
        else:
            add_constraint(opti.bounded(0.015, throttle[k], 1.0),
                           "unity_positive_cawd_drive")

        fz = [normal_load[i, k] for i in range(4)]
        for i in range(4):
            add_constraint(normal_load[i, k] >= minimum_wheel_normal_load_n,
                           f"normal_load_w{i}")
            add_constraint(wheel_omega[i][k] >= 0.0,
                           f"forward_wheel_rotation_w{i}")
        fx, fy, mz, fx_w, _, slip_x, slip_y = tire_forces_casadi(
            model, u[k], v[k], yaw_rate[k], steering[k],
            [wheel_omega[i][k] for i in range(4)], fz)
        ax, ay = accel_xy[0, k], accel_xy[1, k]
        add_constraint(ax - (fx / model.mass_kg
                             - model.drag_per_s * u[k]) == 0,
                       f"longitudinal_force_balance_k{k}")
        add_constraint(ay - (fy / model.mass_kg
                             - model.drag_per_s * v[k]) == 0,
                       f"lateral_force_balance_k{k}")
        predicted_load = suspension_loads_casadi(
            model, body_height[k], body_height_rate[k], roll[k],
            roll_rate[k], pitch[k], pitch_rate[k])
        suspension_travel, _ = suspension_travel_casadi(
            model, body_height[k], body_height_rate[k], roll[k],
            roll_rate[k], pitch[k], pitch_rate[k])
        for i in range(4):
            add_constraint((normal_load[i, k] - predicted_load[i]) / 10.0 == 0,
                           f"unity_spring_damper_load_w{i}")
            add_constraint(opti.bounded(
                0.0, suspension_travel[i], model.suspension_distance_m),
                f"unity_suspension_travel_w{i}")

        # Unity's tire callback returns wheel reaction torque as -Fx*R.
        # Retain it at each collocation node for the source-derived wheel
        # angular-momentum update below.
        wheel_tire_torque_nodes.append(tuple(
            -model.wheel_radius_m * fx_w[i] for i in range(4)))

        longitudinal_dot = ax + yaw_rate[k] * v[k]
        lateral_dot = ay - yaw_rate[k] * u[k]
        yaw_dot = mz / model.yaw_inertia_kgm2 - 0.1 * yaw_rate[k]
        lateral_path_rate = (
            u[k] * ca.sin(epsi[k]) + v[k] * ca.cos(epsi[k])) / sdot
        path_heading_offset = ca.atan2(lateral_path_rate, den)
        body_path_heading_error = epsi[k] - path_heading_offset
        body_path_heading_error_nodes.append(body_path_heading_error)
        if max_body_path_heading_error_rad is not None:
            add_constraint(opti.bounded(
                -max_body_path_heading_error_rad,
                body_path_heading_error,
                max_body_path_heading_error_rad),
                f"body_path_heading_trackability_k{k}")
        heave_acceleration = (
            sum(fz) / model.mass_kg - 9.80665
            - model.drag_per_s * body_height_rate[k])
        roll_moment = (
            sum(-WHEEL_Y_M[i] * fz[i] for i in range(4))
            - model.body_com_height_m * fy)
        pitch_moment = (
            sum(-WHEEL_X_M[i] * fz[i] for i in range(4))
            - model.body_com_height_m * fx)
        roll_acceleration = (roll_moment / model.inertia_roll_kgm2
                             - 0.1 * roll_rate[k])
        pitch_acceleration = (pitch_moment / model.inertia_pitch_kgm2
                              - 0.1 * pitch_rate[k])
        spatial_rhs.append(ca.vertcat(
            (u[k] * ca.sin(epsi[k]) + v[k] * ca.cos(epsi[k])) / sdot,
            yaw_rate[k] / sdot - curvature,
            longitudinal_dot / sdot,
            lateral_dot / sdot,
            yaw_dot / sdot,
            qdelta[k] / sdot,
            body_height_rate[k] / sdot,
            heave_acceleration / sdot,
            roll_rate[k] / sdot,
            roll_acceleration / sdot,
            pitch_rate[k] / sdot,
            pitch_acceleration / sdot))
        time_density.append(1.0 / sdot)
        slip_x_nodes.append(slip_x)
        slip_y_nodes.append(slip_y)
        force_nodes.append((fx, fy, mz))

    # Check both the footprint corridor and optional controller-trackability
    # envelope at segment midpoints. Node-only bounds leave a gap in the
    # spatial mesh where a curved/free path can violate either condition.
    for k in range(n):
        kp = (k + 1) % n
        next_station = station_nodes[kp]
        if kp == 0:
            next_station += track.length
        midpoint_station = 0.5 * (station_nodes[k] + next_station)
        midpoint_ey = 0.5 * (ey[k] + ey[kp])
        midpoint_epsi = 0.5 * (epsi[k] + epsi[kp])
        if max_abs_lateral_speed_mps is not None:
            add_constraint(opti.bounded(
                -max_abs_lateral_speed_mps,
                0.5 * (v[k] + v[kp]),
                max_abs_lateral_speed_mps),
                f"lateral_velocity_trackability_midpoint_k{k}")
        if max_path_reference_deviation_m is not None:
            midpoint_reference_ey = interp["path_reference_offset"](
                midpoint_station)
            midpoint_reference_deviation = midpoint_ey - midpoint_reference_ey
            add_constraint(
                midpoint_reference_deviation
                <= max_path_reference_deviation_m,
                f"path_reference_deviation_midpoint_upper_k{k}")
            add_constraint(
                midpoint_reference_deviation
                >= -max_path_reference_deviation_m,
                f"path_reference_deviation_midpoint_lower_k{k}")
        if max_body_path_heading_error_rad is not None:
            midpoint_curvature = interp["kappa"](midpoint_station)
            midpoint_den = 1.0 - midpoint_curvature * midpoint_ey
            add_constraint(midpoint_den >= 0.25,
                           f"frenet_midpoint_denominator_k{k}")
            midpoint_ey_rate = 0.5 * (
                spatial_rhs[k][0] + spatial_rhs[kp][0])
            midpoint_path_angle = ca.atan2(
                midpoint_ey_rate, midpoint_den)
            midpoint_heading_error = midpoint_epsi - midpoint_path_angle
            add_constraint(opti.bounded(
                -max_body_path_heading_error_rad,
                midpoint_heading_error,
                max_body_path_heading_error_rad),
                f"body_path_heading_trackability_midpoint_k{k}")
        predicted_midpoint_ey = (
            midpoint_ey + interp["tracking_bias"](midpoint_station))
        midpoint_footprint = _footprint_half_width(midpoint_epsi, model)
        add_constraint(
            predicted_midpoint_ey <= (
                interp["left"](midpoint_station) - wall_clearance_m
                - midpoint_footprint),
            f"left_track_boundary_midpoint_k{k}")
        add_constraint(
            predicted_midpoint_ey >= (
                -interp["right"](midpoint_station) + wall_clearance_m
                + midpoint_footprint),
            f"right_track_boundary_midpoint_k{k}")

    lap_time = 0.0
    dynamic_indices = tuple(range(5)) + (9, 10, 11, 12, 13, 14, 15)
    spatial_state_scales = ca.DM((
        0.25, 0.35, 2.0, 1.0, 2.0, 0.35,
        0.02, 1.0, 0.12, 2.0, 0.10, 2.0))

    def source_wheel_update(omega, drive_torque, brake_torque, tire_torque,
                            duration_s, forward_speed):
        low_speed_result = physx_no_drive_wheel_speed_interval_casadi(
            omega, drive_torque, brake_torque, tire_torque,
            model.wheel_inertia_kgm2, model.wheel_damping_nms, duration_s,
            substeps_per_physics_step=UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS)
        high_speed_result = physx_no_drive_wheel_speed_interval_casadi(
            omega, drive_torque, brake_torque, tire_torque,
            model.wheel_inertia_kgm2, model.wheel_damping_nms, duration_s,
            substeps_per_physics_step=UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS)
        return ca.if_else(
            ca.fabs(forward_speed) < UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS,
            low_speed_result, high_speed_result)

    for k in range(n):
        kp = (k + 1) % n
        xk = ca.vertcat(*[x[index, k] for index in dynamic_indices])
        xkp = ca.vertcat(*[x[index, kp] for index in dynamic_indices])
        ds = step_nodes[k]
        add_constraint(
            (xkp - xk - 0.5 * ds * (spatial_rhs[k] + spatial_rhs[kp]))
            / spatial_state_scales == 0,
            [f"periodic_collocation_state_{name}" for name in STATE_NAMES])
        if (throttle_fall_rate_limit_norm_per_s is not None
                and not brake_modes_array[k]
                and not brake_modes_array[kp]):
            segment_time = 0.5 * ds * (time_density[k] + time_density[kp])
            add_constraint(
                throttle[kp] - throttle[k]
                >= -throttle_fall_rate_limit_norm_per_s * segment_time,
                f"positive_throttle_fall_rate_k{k}")
        segment_time = 0.5 * ds * (time_density[k] + time_density[kp])
        interval_start_forward_speed = x[0, k]
        for i in range(4):
            # Unity's controller chooses CAWD quarter-torque for any nonzero
            # throttle and CAWB's full configured brake torque per wheel at
            # exact zero. Reproduce PhysX's speed-dependent 5/1 internal
            # substeps, implicit-Euler wheel update, finite brake torque, and
            # zero-crossing lock. PhysX selects its 5/1 count once from the
            # outer tick's start-of-step longitudinal speed; the collocation
            # interval's start state is the closest represented equivalent.
            # A spatial interval spanning the 5 m/s threshold remains a
            # discretization approximation. A phase boundary is split into
            # drive then brake (or vice versa), not snapped to zero wheel RPM.
            wheel_state = wheel_omega[i][k]
            tire_torque = 0.5 * (
                wheel_tire_torque_nodes[k][i]
                + wheel_tire_torque_nodes[kp][i])
            if brake_modes_array[k] != brake_modes_array[kp]:
                half_time = 0.5 * segment_time
                if brake_modes_array[k]:
                    wheel_state = source_wheel_update(
                        wheel_state, 0.0, model.motor_torque_nm,
                        tire_torque, half_time, interval_start_forward_speed)
                    wheel_state = source_wheel_update(
                        wheel_state,
                        model.motor_torque_nm * throttle[kp] / 4.0, 0.0,
                        tire_torque, half_time, interval_start_forward_speed)
                else:
                    wheel_state = source_wheel_update(
                        wheel_state,
                        model.motor_torque_nm * throttle[k] / 4.0, 0.0,
                        tire_torque, half_time, interval_start_forward_speed)
                    wheel_state = source_wheel_update(
                        wheel_state, 0.0, model.motor_torque_nm,
                        tire_torque, half_time, interval_start_forward_speed)
            elif brake_modes_array[k]:
                wheel_state = source_wheel_update(
                    wheel_state, 0.0, model.motor_torque_nm, tire_torque,
                    segment_time, interval_start_forward_speed)
            else:
                average_drive_torque = (
                    model.motor_torque_nm * 0.5
                    * (throttle[k] + throttle[kp]) / 4.0)
                wheel_state = source_wheel_update(
                    wheel_state, average_drive_torque, 0.0, tire_torque,
                    segment_time, interval_start_forward_speed)
            add_constraint(
                (wheel_omega[i][kp] - wheel_state) / 100.0 == 0,
                f"physx_implicit_wheel_update_{WHEEL_STATE_NAMES[i]}_k{k}")
        lap_time += 0.5 * ds * (time_density[k] + time_density[kp])

    # Reconcile tire force, acceleration, and drive torque at the initial
    # trajectory. Suspension loads come from its seeded heave/attitude states,
    # never from the former quasi-static acceleration split.
    seed_force_state = ca.MX.sym(f"seed_force_state_{id(opti)}", 12, n)
    seed_fx, seed_fy, _, seed_fx_w, _, _, _ = tire_forces_casadi(
        model, seed_force_state[0, :], seed_force_state[1, :],
        seed_force_state[2, :], seed_force_state[3, :],
        [seed_force_state[4 + i, :] for i in range(4)],
        [seed_force_state[8 + i, :] for i in range(4)])
    seed_force_map = ca.Function(
        f"seed_force_map_{id(opti)}", [seed_force_state],
        [ca.vertcat(seed_fx, seed_fy, *seed_fx_w)])
    initialization_consistency_iterations = 0
    for iteration in range(0 if initial_state_seed_is_model_consistent else 5):
        force_state = np.vstack((seed_u, seed_v, seed_yaw, seed_steering,
                                 seed_omega, seed_load))
        force_values = np.asarray(seed_force_map(force_state),
                                  dtype=np.float64)
        ax_values = (force_values[0] / model.mass_kg
                     - model.drag_per_s * seed_u)
        ay_values = (force_values[1] / model.mass_kg
                     - model.drag_per_s * seed_v)
        seed_kappa = sample_periodic(np.asarray(track.kappa))
        seed_sdot = ((seed_u * np.cos(seed_epsi)
                      - seed_v * np.sin(seed_epsi))
                     / (1.0 - seed_kappa * seed_ey))
        if np.any(seed_sdot <= 0.0) or not np.isfinite(seed_sdot).all():
            raise ValueError("suspension seed requires positive path speed")
        old_body_state = np.vstack((
            seed_body_height, seed_body_height_rate, seed_roll, seed_roll_rate,
            seed_pitch, seed_pitch_rate))
        body_state_values, periodic_laps, periodic_residual = (
            _periodic_suspension_seed(
                model, np.asarray(step_nodes_seed), seed_sdot, seed_u, seed_v,
                ax_values, ay_values, old_body_state[:, 0]))
        seed_body_height, seed_body_height_rate = body_state_values[0:2]
        seed_roll, seed_roll_rate = body_state_values[2:4]
        seed_pitch, seed_pitch_rate = body_state_values[4:6]
        seed_load = np.column_stack([
            suspension_loads_numpy(
                model, seed_body_height[k], seed_body_height_rate[k],
                seed_roll[k], seed_roll_rate[k], seed_pitch[k],
                seed_pitch_rate[k])
            for k in range(n)])
        required_per_wheel = (
            model.wheel_radius_m * force_values[2:6]
            + model.wheel_damping_nms * seed_omega)
        throttle_values = np.clip(
            4.0 * np.mean(required_per_wheel, axis=0)
            / model.motor_torque_nm, 0.015, 1.0)
        throttle_values[brake_modes_array] = 0.0
        max_change = max(
            float(np.max(np.abs(seed_load - np.asarray(opti.value(
                normal_load, opti.initial()))))),
            float(np.max(np.abs(ax_values - seed_ax))),
            float(np.max(np.abs(ay_values - seed_ay))),
            float(np.max(np.abs(throttle_values - seed_throttle))),
            float(np.max(np.abs(body_state_values - old_body_state))))
        seed_ax = ax_values
        seed_ay = ay_values
        seed_throttle = throttle_values
        opti.set_initial(normal_load, seed_load)
        opti.set_initial(body_height, seed_body_height)
        opti.set_initial(body_height_rate, seed_body_height_rate)
        opti.set_initial(roll, seed_roll)
        opti.set_initial(roll_rate, seed_roll_rate)
        opti.set_initial(pitch, seed_pitch)
        opti.set_initial(pitch_rate, seed_pitch_rate)
        opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))
        opti.set_initial(throttle, seed_throttle)
        initialization_consistency_iterations = iteration + 1
        suspension_seed_summary = {
            "method": "periodic 1 kHz RK4 using current tire-model planar forcing",
            "outer_force_load_iterations": iteration + 1,
            "periodic_laps": periodic_laps,
            "fixed_point_residual_scaled": periodic_residual,
        }
        if max_change < 1.0e-4:
            break

    # First find a model-consistent lap near the measured motion. Starting the
    # minimum-time objective from an infeasible capture made IPOPT trade path
    # speed against large dynamics defects and stall at a non-feasible point.
    # This feasibility stage is only an initializer; it does not constrain the
    # subsequent free-path minimum-time solve.
    seed_x_values = np.vstack((
        seed_ey, seed_epsi, seed_u, seed_v, seed_yaw, seed_omega,
        seed_steering, seed_body_height, seed_body_height_rate, seed_roll,
        seed_roll_rate, seed_pitch, seed_pitch_rate))
    state_scales = ca.repmat(ca.DM((
        0.25, 0.35, 2.0, 1.0, 2.0,
        150.0, 150.0, 150.0, 150.0, 0.35,
        0.02, 1.0, 0.12, 2.0, 0.10, 2.0)),
                             1, n)
    feasibility_objective = (
        ca.sumsqr((x - ca.DM(seed_x_values)) / state_scales)
        + ca.sumsqr((qdelta - ca.DM(seed_qdelta)) / 3.2)
        + ca.sumsqr(throttle - ca.DM(seed_throttle))
        + ca.sumsqr((phase_lengths - ca.DM(phase_seed_lengths))
                    / ca.DM(np.maximum(phase_seed_lengths, 1.0)))
        + ca.sumsqr((normal_load - ca.DM(seed_load)) / 10.0)
        + ca.sumsqr((accel_xy - ca.DM(np.vstack((seed_ax, seed_ay)))) / 10.0)
    )
    trust_region_bound = opti.parameter()
    # r18/r19 reached a fully feasible captured-motion point with deviation
    # objective 19.4153. Keep phase-I inside a slightly wider, evidenced
    # neighborhood instead of allowing numerically extreme iterates; the cap
    # is not a vehicle constraint and is widened from each feasible incumbent
    # during the subsequent minimum-time continuation.
    opti.set_value(trust_region_bound, feasibility_trust_region)
    add_constraint(feasibility_objective <= trust_region_bound,
                   "captured_motion_trust_region")
    opti.minimize(feasibility_objective)
    try:
        initial_constraints = _constraint_diagnostics(
            opti.value(opti.g, opti.initial()),
            opti.value(opti.lbg, opti.initial()),
            opti.value(opti.ubg, opti.initial()), constraint_labels)
    except (RuntimeError, ValueError) as error:
        initial_constraints = {
            "max_constraint_violation": None,
            "worst_constraints": [],
            "diagnostic_error": str(error),
        }
    if diagnose_initial_only:
        return {}, {
            "optimizer": "CasADi/IPOPT phase-switched spatial direct collocation",
            "track": track.to_dict(),
            "unity_vehicle_model": model.metadata(),
            "solver": {"success": False,
                       "return_status": "initial_guess_diagnostic_only",
                       "iterations": 0, "feasible_incumbent": False,
                       "ipopt_converged": False},
            "nlp_diagnostics": {"initial_guess": initial_constraints},
            "phase_seed": {
                "modes": ["brake" if mode else "drive" for mode in phase_modes],
                "nodes": phase_nodes,
                "lengths_m": phase_seed_lengths.tolist(),
            },
            "model_consistent_initialization_iterations": (
                initialization_consistency_iterations),
        }
    opti.solver("ipopt", {
        "expand": True,
        "ipopt": {
            # A finite budget prevents an infeasible initialization from
            # consuming hours without yielding a usable incumbent.
            "max_iter": feasibility_max_iterations,
            "tol": 1.0e-6,
            "acceptable_tol": 1.0e-5,
            "print_level": int(print_level),
        },
    }, {})
    started = time.perf_counter()
    solver_error = None
    stage_reports: list[dict[str, Any]] = []
    initial_is_feasible = (
        initial_constraints.get("max_constraint_violation") is not None
        and initial_constraints["max_constraint_violation"]
        <= NLP_FEASIBILITY_TOLERANCE)
    if initial_is_feasible:
        # A saved model-consistent candidate is already a feasible incumbent;
        # do not spend thousands of iterations minimizing distance to itself.
        stage1_solution = None
        stage1_stats = {
            "success": True,
            "return_status": "Initial_Feasible_Incumbent",
            "iter_count": 0,
            "_initial_snapshot": opti.initial(),
        }
        stage1_diagnostics = initial_constraints
    else:
        try:
            stage1_solution = opti.solve_limited()
        except RuntimeError as error:
            stage1_solution = None
            solver_error = str(error)
        stage1_stats = opti.stats()

    def stage_values(solution, stats, expression):
        if solution is not None:
            try:
                return np.asarray(solution.value(expression), dtype=np.float64)
            except RuntimeError:
                pass
        if stats.get("_initial_snapshot") is not None:
            return np.asarray(opti.value(
                expression, stats["_initial_snapshot"]), dtype=np.float64)
        return np.asarray(opti.debug.value(expression), dtype=np.float64)

    def stage_constraint_diagnostics(solution, stats):
        try:
            values = stage_values(solution, stats, opti.g).reshape(-1)
            lower = stage_values(solution, stats, opti.lbg).reshape(-1)
            upper = stage_values(solution, stats, opti.ubg).reshape(-1)
            return _constraint_diagnostics(values, lower, upper,
                                           constraint_labels)
        except (RuntimeError, ValueError) as error:
            return {"max_constraint_violation": None,
                    "worst_constraints": [], "diagnostic_error": str(error)}

    if not initial_is_feasible:
        stage1_diagnostics = stage_constraint_diagnostics(
            stage1_solution, stage1_stats)
    stage_reports.append({
        "name": "captured-motion feasibility",
        **_solver_status(stage1_stats),
        "max_constraint_violation": stage1_diagnostics.get(
            "max_constraint_violation"),
    })
    solution = stage1_solution
    stats = stage1_stats
    final_constraint_diagnostics = stage1_diagnostics
    selected_stage = "captured-motion feasibility"
    best_trust_region_factor = None

    if (stage1_diagnostics.get("max_constraint_violation") is not None
            and stage1_diagnostics["max_constraint_violation"]
            <= NLP_FEASIBILITY_TOLERANCE):
        # A feasible incumbent is required before time minimization. Preserve
        # all primal values as the warm start even if IPOPT's KKT test is not
        # yet satisfied.
        for variable in (x, qdelta, throttle, phase_lengths,
                         normal_load, accel_xy):
            opti.set_initial(variable,
                             stage_values(stage1_solution, stage1_stats,
                                          variable))
        stage2_initial_diagnostics = _constraint_diagnostics(
            opti.value(opti.g, opti.initial()),
            opti.value(opti.lbg, opti.initial()),
            opti.value(opti.ubg, opti.initial()), constraint_labels)
        stage_reports.append({
            "name": "minimum-time warm-start check",
            "max_constraint_violation": stage2_initial_diagnostics[
                "max_constraint_violation"],
        })
        # Pure minimum-time optimization repeatedly left the feasible region.
        # Minimize lap time inside successively larger trust regions around
        # the feasible captured-motion solution, accepting only full-NLP
        # feasible iterates. The final large-radius pass approximates an
        # unconstrained minimum-time solve, but remains a local IPOPT result.
        feasibility_reference = max(float(stage_values(
            stage1_solution, stage1_stats, feasibility_objective)),
            minimum_trust_region_reference)
        stage_reports[0]["feasibility_objective_value"] = (
            feasibility_reference)
        best_solution = stage1_solution
        best_stats = stage1_stats
        best_diagnostics = stage1_diagnostics
        best_lap_time = float(stage_values(
            stage1_solution, stage1_stats, lap_time))
        # Keep the best-time result separately from the latest feasible
        # continuation point. A wider trust stage can reach a feasible but
        # temporarily slower path that is a better starting basin for the
        # next stage; restarting only from best_lap_time discards that progress.
        restart_solution = stage1_solution
        restart_stats = stage1_stats
        solver_error = None
        # A 1% expansion of the squared-distance budget left only a 2.6%
        # normalized Euclidean step at the r22 feasible point and IPOPT
        # diverged far outside the bound. Start with a useful trust radius;
        # each accepted stage is still required to satisfy every NLP constraint.
        # Earlier code stopped when dynamics residuals were also high, so it
        # never reached the near-unconstrained time pass. Advance through all
        # larger radii after each finite failed solve, restarting from the
        # latest feasible continuation point rather than the failed iterate or
        # reverting to an earlier, lower-time basin.
        trust_region_factors = time_trust_region_factors
        for trust_factor in trust_region_factors:
            for variable in (x, qdelta, throttle, phase_lengths,
                             normal_load, accel_xy):
                opti.set_initial(variable, stage_values(
                    restart_solution, restart_stats, variable))
            trust_bound_value = feasibility_reference * trust_factor
            opti.set_value(trust_region_bound, trust_bound_value)
            opti.minimize(lap_time)
            time_ipopt_options = {
                "max_iter": time_optimization_max_iterations,
                "tol": 1.0e-6,
                "acceptable_tol": 1.0e-5,
                "print_level": int(print_level),
            }
            if time_hessian_approximation == "limited-memory":
                time_ipopt_options.update({
                    "hessian_approximation": "limited-memory",
                    "limited_memory_max_history": 20,
                })
            opti.solver("ipopt", {
                "expand": True,
                "ipopt": time_ipopt_options,
            }, {})
            try:
                continuation_solution = opti.solve_limited()
                continuation_error = None
            except RuntimeError as error:
                continuation_solution = None
                continuation_error = str(error)
            continuation_stats = opti.stats()
            continuation_diagnostics = stage_constraint_diagnostics(
                continuation_solution, continuation_stats)
            continuation_time = float(stage_values(
                continuation_solution, continuation_stats, lap_time))
            continuation_violation = continuation_diagnostics.get(
                "max_constraint_violation")
            continuation_feasible = (
                continuation_violation is not None
                and continuation_violation <= NLP_FEASIBILITY_TOLERANCE)
            continuation_restartable = (
                continuation_violation is not None
                and math.isfinite(float(continuation_violation))
                and continuation_violation
                <= NEAR_FEASIBLE_WARM_START_MAX_VIOLATION)
            non_trust_violations = [
                float(item["violation"])
                for item in continuation_diagnostics.get(
                    "worst_constraints", [])
                if item["label"] != "captured_motion_trust_region"]
            max_non_trust_violation = max(non_trust_violations, default=0.0)
            retry_at_larger_radius = (
                not continuation_feasible
                and continuation_violation is not None
                and math.isfinite(float(continuation_violation))
                and trust_factor < trust_region_factors[-1])
            stage_reports.append({
                "name": ("free-geometry minimum time" if trust_factor >= 1.0e6
                         else "free-geometry trust-region time"),
                "trust_region_factor": trust_factor,
                "trust_region_bound": trust_bound_value,
                **_solver_status(continuation_stats),
                "exception": continuation_error,
                "objective_lap_time_s": continuation_time,
                "max_constraint_violation": continuation_violation,
                "max_non_trust_constraint_violation": (
                    max_non_trust_violation),
                "continued_after_failed_time_stage": retry_at_larger_radius,
                "restart_point_updated": continuation_restartable,
                "restart_point_kind": (
                    "feasible" if continuation_feasible else
                    "near_feasible" if continuation_restartable else
                    "previous_feasible_or_near_feasible"),
                "last_primal_infeasibility": (
                    float(continuation_stats["iterations"]["inf_pr"][-1])
                    if continuation_stats.get("iterations", {}).get("inf_pr")
                    else None),
                "last_dual_infeasibility": (
                    float(continuation_stats["iterations"]["inf_du"][-1])
                    if continuation_stats.get("iterations", {}).get("inf_du")
                    else None),
                "worst_constraints": continuation_diagnostics.get(
                    "worst_constraints", [])[:12],
            })
            # A near-feasible point is allowed only as the initial point for
            # the next, wider IPOPT trust stage. It is never selected as the
            # output candidate unless it separately meets the strict NLP
            # feasibility tolerance above.
            if continuation_restartable:
                restart_solution = continuation_solution
                restart_stats = continuation_stats
            if not continuation_feasible:
                solver_error = continuation_error
                if retry_at_larger_radius:
                    continue
                break
            solver_error = continuation_error
            if continuation_time < best_lap_time:
                best_solution = continuation_solution
                best_stats = continuation_stats
                best_diagnostics = continuation_diagnostics
                best_lap_time = continuation_time
                best_trust_region_factor = trust_factor

        solution = best_solution
        stats = best_stats
        final_constraint_diagnostics = best_diagnostics
        if best_trust_region_factor is None:
            selected_stage = "captured-motion feasibility fallback"
        elif best_trust_region_factor >= 1.0e6:
            selected_stage = "free-geometry minimum time"
        else:
            selected_stage = "free-geometry trust-region time"

    elapsed = time.perf_counter() - started

    def value(expression):
        evaluated = stage_values(solution, stats, expression)
        return np.asarray(evaluated, dtype=np.float64)

    try:
        g_values = value(opti.g).reshape(-1)
        lower = value(opti.lbg).reshape(-1)
        upper = value(opti.ubg).reshape(-1)
    except (RuntimeError, ValueError) as error:
        return {}, {
            "optimizer": "CasADi/IPOPT phase-switched spatial direct collocation",
            "track": track.to_dict(), "unity_vehicle_model": model.metadata(),
            "elapsed_s": elapsed,
            "solver": {**_solver_status(stats),
                       "exception": solver_error or str(error),
                       "feasible_incumbent": False,
                       "ipopt_converged": False},
            "nlp_diagnostics": {"max_constraint_violation": None,
                                "worst_constraints": [],
                                "solver_error": solver_error or str(error)},
        }
    finite = np.isfinite(g_values).all()
    final_constraint_diagnostics = _constraint_diagnostics(
        g_values, lower, upper, constraint_labels)
    violation = np.maximum(np.maximum(lower - g_values,
                                      g_values - upper), 0.0)
    violation[~np.isfinite(g_values)] = np.inf
    max_violation = final_constraint_diagnostics["max_constraint_violation"]
    feasible_incumbent = bool(
        finite and max_violation <= NLP_FEASIBILITY_TOLERANCE)
    nlp_diagnostics = {
        **final_constraint_diagnostics,
        "initial_guess": initial_constraints,
        "last_primal_infeasibility": (
            float(stats["iterations"]["inf_pr"][-1])
            if stats.get("iterations", {}).get("inf_pr") else None),
        "last_dual_infeasibility": (
            float(stats["iterations"]["inf_du"][-1])
            if stats.get("iterations", {}).get("inf_du") else None),
        "feasible_incumbent": feasible_incumbent,
        "solver_error": solver_error,
    }
    solver_report = _solver_status(stats)
    solver_report.update({
        "exception": solver_error,
        "feasible_incumbent": feasible_incumbent,
        "ipopt_converged": _solver_status(stats)["ipopt_converged"],
    })
    report = {
        "optimizer": "CasADi/IPOPT phase-switched spatial direct collocation",
        "path_initialization": (
            "free geometry; initialized from the supplied lateral/heading "
            "state seed, then optimized without a hard reference-path constraint"),
        "decision_variables": {
            "differential_states": list(STATE_NAMES),
            "dynamic_states": [*STATE_NAMES, *WHEEL_STATE_NAMES],
            "algebraic_states": ["four spring/damper loads and planar accelerations"],
            "controls": ["physical steering rate", "positive normalized motor torque", "exact-zero CAWB brake phases"],
            "path_geometry": "independent lateral offset at every optimized spatial node",
            "brake_phase_lengths_are_optimized": True,
            "brake_phase_order_and_count": "seeded from measured training run; switch locations and durations are free",
        },
        "wheel_spin_discretization": {
            "method": (
                "closed-form composition of PhysX implicit-Euler internal wheel "
                "substeps; 5 x 0.2 ms below 5 m/s, 1 x 1 ms at/above 5 m/s"),
            "time_constant_s": (
                model.wheel_inertia_kgm2 / model.wheel_damping_nms),
            "reason": (
                "integer internal-step damping exactly composes "
                "integrateNoDriveWheelSpeeds for held torque; the substep "
                "branch is selected from interval-start forward speed, matching "
                "the outer vehicle tick as represented by the spatial mesh; "
                "a spatial interval crossing 5 m/s cannot reproduce per-tick "
                "branch changes. Variable intervals use a smooth fractional-"
                "step extension and trapezoidal tire-reaction torque"),
            "physx_source_status": (
                "source wheel torque split, speed-dependent 5/1 internal "
                "substeps, finite CAWB brake torque, implicit damping, and "
                "zero-crossing wheel lock are connected to the spatial OCP; "
                "wheel reaction torque is averaged over each spatial interval"),
            "body_state_discretization": "trapezoidal spatial collocation",
            "spatial_state_defect_scales": [
                0.25, 0.35, 2.0, 1.0, 2.0, 0.35,
                0.02, 1.0, 0.12, 2.0, 0.10, 2.0],
            "suspension_load_model": (
                "per-wheel WheelCollider ray travel, calibrated source spring "
                "and damper forces; heave/roll/pitch integrate rigid-body "
                "forces and moments with Unity source inertia and drag"),
        },
        "unity_vehicle_model": model.metadata(),
        "track": track.to_dict(),
        "gate_state_anchor": gate_state_anchor,
        "gate_lateral_tolerance_m": gate_lateral_tolerance_m,
        "gate_steering_tolerance_rad": gate_steering_tolerance_rad,
        "controller_trackability_limits": {
            "max_body_to_path_heading_error_rad": (
                max_body_path_heading_error_rad),
            "max_abs_lateral_speed_mps": max_abs_lateral_speed_mps,
            "body_to_path_heading_definition": (
                "body heading relative to the optimized path tangent; path "
                "tangent offset is atan2(d(ey)/ds, 1-kappa_centerline*ey)"),
        },
        "path_reference_tube": {
            "enabled": max_path_reference_deviation_m is not None,
            "maximum_lateral_deviation_m": max_path_reference_deviation_m,
            "reference_offset_min_m": (
                None if path_reference_offset_m is None
                else float(np.min(path_reference_offset_m))),
            "reference_offset_max_m": (
                None if path_reference_offset_m is None
                else float(np.max(path_reference_offset_m))),
        },
        "minimum_wheel_normal_load_constraint_n": (
            minimum_wheel_normal_load_n),
        "feasibility_trust_region": feasibility_trust_region,
        "ipopt_iteration_budgets": {
            "captured_motion_feasibility": feasibility_max_iterations,
            "per_time_optimization_stage": time_optimization_max_iterations,
        },
        "time_hessian_approximation": time_hessian_approximation,
        "time_trust_region_factors": list(time_trust_region_factors),
        "nlp_feasibility_tolerance": NLP_FEASIBILITY_TOLERANCE,
        "constraint_residual_scaling": {
            "longitudinal_force_balance": "unscaled physical acceleration residual (m/s^2)",
            "lateral_force_balance": "unscaled physical acceleration residual (m/s^2)",
            "physical_equations_changed": False,
        },
        "wall_clearance_m": wall_clearance_m,
        "supported_speed_domain_mps": [min_speed_mps, max_speed_mps],
        "brake_mode_seed_source": brake_mode_seed_source,
        "throttle_fall_rate_limit_norm_per_s": (
            throttle_fall_rate_limit_norm_per_s),
        "brake_phase_seed_ranges_and_modes": [
            {"seed_start_node": begin, "seed_end_exclusive_node": end,
             "mode": "brake" if mode else "drive",
             "optimized_mesh_nodes": int(count),
             "seed_length_m": float(length)}
            for (begin, end), mode, count, length in zip(
                phase_ranges, phase_modes, phase_nodes, phase_seed_lengths)],
        "model_consistent_initialization_iterations": (
            initialization_consistency_iterations),
        "suspension_seed": suspension_seed_summary,
        "optimized_phase_lengths_m": value(phase_lengths).reshape(-1).tolist()
            if finite else None,
        "optimized_brake_nodes": int(np.count_nonzero(brake_modes_array)),
        "predicted_lap_time_s": float(value(lap_time)) if finite else None,
        "elapsed_s": elapsed,
        "solve_stages": stage_reports,
        "selected_stage": selected_stage,
        "selected_trust_region_factor": best_trust_region_factor,
        "feasibility_objective_reference": (
            stage_reports[0].get("feasibility_objective_value")),
        "solver": solver_report,
        "nlp_diagnostics": nlp_diagnostics,
        "limits_and_caveats": [
            "The optimizer is a new free-geometry path-and-speed OCP, not a speed-scaled reference trajectory.",
            "The measured capture seeds only the number/order of drive and exact-zero brake phases; IPOPT optimizes every phase length.",
            "Public Unity VehicleController, active serialized vehicle values, source capture friction curves, and source-derived rigid-body equations are used; the native WheelCollider contact solver is not embedded in the differentiable OCP.",
            "The configured Unity friction curves and native tire-force callback equations are source-translated, but contact velocity, changing raycast hit/load, surface friction pair, and full body/contact solve are not reproduced end-to-end. The horizontal plane and spring/damper states remain optimizer assumptions.",
            "Wheel spin uses the closed-form PhysX implicit-Euler composition with the source 5/1 substep branch selected from interval-start forward speed, matching the outer vehicle tick represented by the spatial mesh; spatial intervals crossing 5 m/s, fractional durations, interval-averaged tire reaction, and exact native 1 ms contact/tire recomputation remain OCP discretization approximations. Zero-throttle CAWB phases use finite configured brake torque and the PhysX zero-crossing wheel lock.",
            "IPOPT establishes a local solution only; the objective is not a global-minimum or Unity lap-time guarantee.",
        ],
    }
    warm_startable_iterate = bool(
        finite and max_violation <= NEAR_FEASIBLE_WARM_START_MAX_VIOLATION)
    diagnostic_iterate_only = should_retain_diagnostic_iterate(
        finite, feasible_incumbent, max_violation)
    if (not feasible_incumbent and not warm_startable_iterate
            and not diagnostic_iterate_only):
        return {}, report
    if warm_startable_iterate and not feasible_incumbent:
        report["near_feasible_warm_start_only"] = True
    elif diagnostic_iterate_only:
        report["diagnostic_iterate_only"] = True
        report["diagnostic_iterate_warning"] = (
            "This infeasible iterate may only be used for bounded solver "
            "continuation. It is not a raceline or accepted warm start.")

    states = value(x)
    loads = value(normal_load)
    acceleration = value(accel_xy)
    phase_length_values = value(phase_lengths).reshape(-1)
    station_values = []
    for phase, count in enumerate(node_counts):
        start_s = float(np.sum(phase_length_values[:phase]))
        station_values.extend(
            start_s + phase_length_values[phase]
            * (np.arange(int(count), dtype=np.float64) / int(count)))
    station_values = np.asarray(station_values)
    path_x, path_y, path_curvature, route_arc = _path_geometry_at(
        track, station_values, states[0])
    time_values = np.asarray([float(value(expr)) for expr in time_density])
    periodic_track_s = np.r_[track.s, track.length]
    midpoint_s = np.mod(
        0.5 * (station_values + np.r_[station_values[1:],
                                      track.length + station_values[0]]),
        track.length)
    midpoint_ey_values = 0.5 * (states[0] + np.roll(states[0], -1))
    midpoint_epsi_values = 0.5 * (states[1] + np.roll(states[1], -1))
    node_bias = np.interp(
        station_values, periodic_track_s,
        np.r_[tracking_bias_values, tracking_bias_values[0]])
    midpoint_bias = np.interp(
        midpoint_s, periodic_track_s,
        np.r_[tracking_bias_values, tracking_bias_values[0]])
    node_left_clearance = (
        np.interp(station_values, periodic_track_s,
                  np.r_[track.left, track.left[0]])
        - wall_clearance_m - _physical_footprint(states[1])
        - states[0] - node_bias)
    node_right_clearance = (
        np.interp(station_values, periodic_track_s,
                  np.r_[track.right, track.right[0]])
        - wall_clearance_m - _physical_footprint(states[1])
        + states[0] + node_bias)
    midpoint_left_clearance = (
        np.interp(midpoint_s, periodic_track_s,
                  np.r_[track.left, track.left[0]])
        - wall_clearance_m - _physical_footprint(midpoint_epsi_values)
        - midpoint_ey_values - midpoint_bias)
    midpoint_right_clearance = (
        np.interp(midpoint_s, periodic_track_s,
                  np.r_[track.right, track.right[0]])
        - wall_clearance_m - _physical_footprint(midpoint_epsi_values)
        + midpoint_ey_values + midpoint_bias)
    force_x_values = np.asarray(
        [float(value(force_nodes[k][0])) for k in range(n)])
    force_y_values = np.asarray(
        [float(value(force_nodes[k][1])) for k in range(n)])
    ax_residual_mps2 = acceleration[0] - (
        force_x_values / model.mass_kg - model.drag_per_s * states[2])
    ay_residual_mps2 = acceleration[1] - (
        force_y_values / model.mass_kg - model.drag_per_s * states[3])
    report["constraint_diagnostics"] = {
        "minimum_centerline_side_clearance_after_footprint_m": float(min(
            np.min(node_left_clearance), np.min(node_right_clearance),
            np.min(midpoint_left_clearance),
            np.min(midpoint_right_clearance))),
        "track_boundary_sampling": "nodes_and_segment_midpoints",
        "max_abs_longitudinal_force_balance_residual_mps2": float(
            np.max(np.abs(ax_residual_mps2))),
        "max_abs_lateral_force_balance_residual_mps2": float(
            np.max(np.abs(ay_residual_mps2))),
        "minimum_wheel_normal_load_n": float(np.min(loads)),
        "max_abs_longitudinal_slip_by_wheel": np.max(
            np.abs(np.asarray([[float(value(slip_x_nodes[k][i]))
                                for k in range(n)] for i in range(4)])),
            axis=1).tolist(),
        "max_abs_lateral_slip_by_wheel": np.max(
            np.abs(np.asarray([[float(value(slip_y_nodes[k][i]))
                                for k in range(n)] for i in range(4)])),
            axis=1).tolist(),
        "peak_speed_mps": float(np.max(states[2])),
        "peak_abs_steering_rad": float(np.max(np.abs(states[9]))),
        "peak_abs_lateral_velocity_mps": float(np.max(np.abs(states[3]))),
        "peak_abs_body_to_path_heading_error_rad": float(np.max(np.abs(
            np.asarray([float(value(item))
                        for item in body_path_heading_error_nodes])))),
        "peak_abs_lateral_acceleration_mps2": float(np.max(np.abs(acceleration[1]))),
        "peak_abs_body_roll_rad": float(np.max(np.abs(states[12]))),
        "peak_abs_body_pitch_rad": float(np.max(np.abs(states[14]))),
        "body_height_range_m": [float(np.min(states[10])),
                                 float(np.max(states[10]))],
        "path_offset_rms_m": float(np.sqrt(np.mean(states[0] ** 2))),
        "path_offset_peak_abs_m": float(np.max(np.abs(states[0]))),
        "path_length_m": float(route_arc[-1] + np.hypot(
            path_x[0] - path_x[-1], path_y[0] - path_y[-1])),
        "path_curvature_abs_max_per_m": float(np.max(np.abs(path_curvature))),
        "time_density_integral_check_s": float(np.sum(
            0.5 * np.asarray([float(value(expr)) for expr in step_nodes])
            * (time_values + np.roll(time_values, -1)))),
    }
    candidate = {
        "x_m": path_x, "y_m": path_y, "s_ref_m": station_values,
        "route_arc_m": route_arc,
        "ey_m": states[0], "epsi_rad": states[1],
        "speed_mps": states[2], "lateral_speed_mps": states[3],
        "yaw_rate_radps": states[4], "wheel_omega_radps": states[5:9],
        "steering_rad": states[9], "steering_rate_radps": value(qdelta).reshape(-1),
        "body_height_m": states[10],
        "body_height_rate_mps": states[11],
        "roll_rad": states[12], "roll_rate_radps": states[13],
        "pitch_rad": states[14], "pitch_rate_radps": states[15],
        "throttle_norm": value(throttle).reshape(-1),
        "brake_mode": brake_modes_array.astype(np.float64),
        "normal_load_n": loads, "ax_mps2": acceleration[0],
        "ay_mps2": acceleration[1],
        "slip_x": np.asarray([[float(value(slip_x_nodes[k][i]))
                               for k in range(n)] for i in range(4)]),
        "slip_y": np.asarray([[float(value(slip_y_nodes[k][i]))
                               for k in range(n)] for i in range(4)]),
        "force_summary": np.asarray([[float(value(force_nodes[k][axis]))
                                     for k in range(n)] for axis in range(3)]),
        "path_curvature_per_m": path_curvature,
    }
    if path_reference_offset_m is not None:
        reference_values = np.interp(
            station_values,
            np.r_[track.s, track.length],
            np.r_[path_reference_offset_m, path_reference_offset_m[0]])
        path_deviation = states[0] - reference_values
        candidate["path_reference_offset_m"] = reference_values
        candidate["path_reference_deviation_m"] = path_deviation
        report["constraint_diagnostics"]["max_abs_path_reference_deviation_m"] = (
            float(np.max(np.abs(path_deviation))))
    return candidate, report


def _physical_footprint(epsi: np.ndarray) -> np.ndarray:
    return footprint_half_width_numpy(epsi)


def write_candidate(candidate: dict[str, np.ndarray], report: dict[str, Any],
                    track: Track, centerline_path: Path,
                    output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    n = track.count
    raceline = output_dir / "unity_wheel_hybrid_raceline.csv"
    with raceline.open("w", encoding="utf-8") as stream:
        stream.write("# x_m,y_m,s_ref_m,speed_mps,steering_rad,throttle_norm,"
                     "path_curvature_per_m,ey_m,epsi_rad\n")
        for k in range(n):
            stream.write(
                f"{candidate['x_m'][k]:.9f},{candidate['y_m'][k]:.9f},"
                f"{candidate['s_ref_m'][k]:.9f},"
                f"{candidate['speed_mps'][k]:.9f},"
                f"{candidate['steering_rad'][k]:.9f},"
                f"{candidate['throttle_norm'][k]:.9f},"
                f"{candidate['path_curvature_per_m'][k]:.9f},"
                f"{candidate['ey_m'][k]:.9f},"
                f"{candidate['epsi_rad'][k]:.9f}\n")
    state_path = output_dir / "unity_wheel_hybrid_states.csv"
    state_columns = [
        "s_ref_m", "x_m", "y_m", "ey_m", "epsi_rad", "speed_mps",
        "lateral_speed_mps", "yaw_rate_radps", "steering_rad",
        "body_height_m", "body_height_rate_mps", "roll_rad",
        "roll_rate_radps", "pitch_rad", "pitch_rate_radps",
        "steering_rate_radps", "throttle_norm", "ax_mps2", "ay_mps2",
        "normal_load_fl_n", "normal_load_fr_n", "normal_load_rl_n",
        "normal_load_rr_n", "slipx_fl", "slipx_fr", "slipx_rl",
        "slipx_rr", "slipy_fl", "slipy_fr", "slipy_rl", "slipy_rr",
        "kappa_path_per_m",
    ]
    if "path_reference_offset_m" in candidate:
        state_columns += ["path_reference_offset_m",
                          "path_reference_deviation_m"]
    with state_path.open("w", encoding="utf-8") as stream:
        stream.write(",".join(state_columns) + "\n")
        for k in range(n):
            values = [candidate["s_ref_m"][k], candidate["x_m"][k],
                      candidate["y_m"][k], candidate["ey_m"][k],
                      candidate["epsi_rad"][k], candidate["speed_mps"][k],
                      candidate["lateral_speed_mps"][k],
                      candidate["yaw_rate_radps"][k],
                      candidate["steering_rad"][k],
                      candidate["body_height_m"][k],
                      candidate["body_height_rate_mps"][k],
                      candidate["roll_rad"][k],
                      candidate["roll_rate_radps"][k],
                      candidate["pitch_rad"][k],
                      candidate["pitch_rate_radps"][k],
                      candidate["steering_rate_radps"][k],
                      candidate["throttle_norm"][k], candidate["ax_mps2"][k],
                      candidate["ay_mps2"][k],
                      *candidate["normal_load_n"][:, k],
                      *candidate["slip_x"][:, k], *candidate["slip_y"][:, k],
                      candidate["path_curvature_per_m"][k]]
            if "path_reference_offset_m" in candidate:
                values += [candidate["path_reference_offset_m"][k],
                           candidate["path_reference_deviation_m"][k]]
            stream.write(",".join(f"{float(value):.9f}" for value in values)
                         + "\n")
    report.update({
        "centerline_source": str(centerline_path),
        "centerline_sha256": hashlib.sha256(
            centerline_path.read_bytes()).hexdigest(),
        "trajectory_csv": str(raceline),
        "state_csv": str(state_path),
    })
    (output_dir / "optimizer_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

