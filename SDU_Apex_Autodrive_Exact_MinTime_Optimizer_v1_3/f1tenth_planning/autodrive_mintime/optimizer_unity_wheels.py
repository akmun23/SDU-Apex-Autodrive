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
    normal_loads_casadi,
    tire_forces_casadi,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    WHEEL_RADIUS_M,
    WHEEL_X_M,
    WHEEL_Y_M,
    ackermann_angles,
)


STATE_NAMES = (
    "ey_m", "epsi_rad", "u_mps", "v_mps", "yaw_rate_radps",
    "steering_rad",
)
WHEEL_STATE_NAMES = (
    "wheel_omega_fl_radps", "wheel_omega_fr_radps",
    "wheel_omega_rl_radps", "wheel_omega_rr_radps",
)


def _footprint_half_width(epsi, model: UnityWheelModel):
    physical = (0.1365 * ca.fabs(ca.cos(epsi))
                + 0.255 * ca.fabs(ca.sin(epsi)))
    return ca.fmax(0.15, physical)


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
    return {
        "success": bool(stats.get("success", False)),
        "return_status": str(stats.get("return_status", "unknown")),
        "iterations": int(stats.get("iter_count", 0)),
    }


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


def _periodic_interpolants(track: Track) -> dict[str, Any]:
    grid = np.r_[np.asarray(track.s, dtype=np.float64), track.length]
    periodic = {
        "x": track.x,
        "y": track.y,
        "sin_psi": np.sin(track.psi),
        "cos_psi": np.cos(track.psi),
        "kappa": track.kappa,
        "left": track.left,
        "right": track.right,
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


def solve_bottom_up(track: Track, model: UnityWheelModel,
                    *, wall_clearance_m: float = 0.15,
                    max_speed_mps: float = 12.0,
                    min_speed_mps: float = 0.35,
                    brake_mode_seed: np.ndarray | None = None,
                    brake_mode_seed_source: str = "drive-mode initialization",
                    initial_state_seed: dict[str, np.ndarray] | None = None,
                    max_solver_seconds: float = 300.0,
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
    if brake_mode_seed is None:
        brake_mode_seed = np.zeros(n, dtype=bool)
    brake_mode_seed = np.asarray(brake_mode_seed, dtype=bool)
    if not math.isfinite(max_solver_seconds) or max_solver_seconds <= 0.0:
        raise ValueError("solver time limit must be finite and positive")
    phase_modes, phase_nodes, phase_seed_lengths, phase_ranges = _brake_phases(
        track, brake_mode_seed)
    phase_count = len(phase_modes)
    node_counts = np.asarray(phase_nodes, dtype=np.int64)
    n = int(np.sum(node_counts))
    interp = _periodic_interpolants(track)
    opti = ca.Opti()
    x = opti.variable(10, n)
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
    opti.set_initial(phase_lengths, phase_seed_lengths)

    station_nodes = []
    step_nodes = []
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
            brake_modes.append(mode)
            station_seed.append(seed_start + phase_seed_lengths[phase] * fraction)
        station_start += phase_lengths[phase]
        seed_start += float(phase_seed_lengths[phase])
    station_seed = np.asarray(station_seed, dtype=np.float64)
    if station_seed.shape != (n,) or not np.all(np.diff(station_seed) > 0.0):
        raise ValueError("phase seed does not create an increasing spatial mesh")
    brake_modes_array = np.asarray(brake_modes, dtype=bool)

    def sample_periodic(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            return np.interp(station_seed, np.r_[track.s, track.length],
                             np.r_[values, values[0]])
        return np.vstack([sample_periodic(row) for row in values])

    ey, epsi, u, v, yaw_rate = (x[i, :] for i in range(5))
    wheel_omega = [x[i, :] for i in range(5, 9)]
    steering = x[9, :]

    # Initial guesses use the measured source run where available; otherwise
    # they are generated from centerline curvature and configured tire peaks.
    if initial_state_seed is not None:
        expected = {
            "ey_m": (track.count,), "epsi_rad": (track.count,),
            "u_mps": (track.count,), "v_mps": (track.count,),
            "yaw_rate_radps": (track.count,),
            "wheel_omega_radps": (4, track.count),
            "steering_rad": (track.count,),
            "steering_rate_radps": (track.count,),
            "throttle_norm": (track.count,), "ax_mps2": (track.count,),
            "ay_mps2": (track.count,), "normal_load_n": (4, track.count),
        }
        for name, shape in expected.items():
            values = np.asarray(initial_state_seed[name], dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"invalid captured initial guess for {name}")
        seed_ey = sample_periodic(initial_state_seed["ey_m"])
        seed_epsi = sample_periodic(initial_state_seed["epsi_rad"])
        seed_u = sample_periodic(initial_state_seed["u_mps"])
        seed_v = sample_periodic(initial_state_seed["v_mps"])
        seed_yaw = sample_periodic(initial_state_seed["yaw_rate_radps"])
        seed_omega = sample_periodic(initial_state_seed["wheel_omega_radps"])
        seed_steering = sample_periodic(initial_state_seed["steering_rad"])
        seed_qdelta = sample_periodic(initial_state_seed["steering_rate_radps"])
        seed_throttle = sample_periodic(initial_state_seed["throttle_norm"])
        seed_ax = sample_periodic(initial_state_seed["ax_mps2"])
        seed_ay = sample_periodic(initial_state_seed["ay_mps2"])
        seed_load = sample_periodic(initial_state_seed["normal_load_n"])
    else:
        kappa_seed = sample_periodic(np.asarray(track.kappa))
        curve_peak = [max(curve[1], curve[3]) * curve[4]
                      for curve in model.lateral_curve]
        seed_lateral_accel = (model.lateral_scale * sum(
            model.static_normal_load_n[i] * curve_peak[i]
            for i in range(4)) / model.mass_kg)
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
        seed_load = np.empty((4, n), dtype=np.float64)
        for i in range(4):
            front_sign = -1.0 if i < 2 else 1.0
            side_sign = -1.0 if i in (0, 2) else 1.0
            longitudinal = (model.mass_kg * model.body_com_height_m
                            * seed_ax / (2.0 * model.wheelbase_m))
            lateral = (model.mass_kg * model.body_com_height_m
                       * seed_ay / (4.0 * model.track_m))
            seed_load[i] = (model.static_normal_load_n[i]
                            + front_sign * longitudinal
                            + side_sign * lateral)

    seed_omega[:, brake_modes_array] = 0.0
    seed_throttle[brake_modes_array] = 0.0
    seed_throttle[~brake_modes_array] = np.maximum(
        0.015, seed_throttle[~brake_modes_array])
    opti.set_initial(ey, seed_ey)
    opti.set_initial(epsi, seed_epsi)
    opti.set_initial(u, seed_u)
    opti.set_initial(v, seed_v)
    opti.set_initial(yaw_rate, seed_yaw)
    for i in range(4):
        opti.set_initial(wheel_omega[i], seed_omega[i])
    opti.set_initial(steering, seed_steering)
    opti.set_initial(qdelta, seed_qdelta)
    opti.set_initial(throttle, seed_throttle)
    opti.set_initial(normal_load, seed_load)
    opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))
    # Keep the optimized periodic line at the actual map gate; other path
    # offsets and the full speed/control schedule remain free.
    add_constraint(ey[0] == float(seed_ey[0]), "gate_lateral_offset")
    add_constraint(epsi[0] == float(seed_epsi[0]), "gate_heading_error")

    spatial_rhs = []
    time_density = []
    slip_x_nodes = []
    slip_y_nodes = []
    force_nodes = []
    wheel_spin_spatial_rhs = []
    for k in range(n):
        station = station_nodes[k]
        curvature = interp["kappa"](station)
        left = interp["left"](station)
        right = interp["right"](station)
        den = 1.0 - curvature * ey[k]
        sdot = (u[k] * ca.cos(epsi[k]) - v[k] * ca.sin(epsi[k])) / den
        add_constraint(den >= 0.25, "frenet_denominator")
        add_constraint(sdot >= 0.30, "path_speed")
        add_constraint(opti.bounded(min_speed_mps, u[k], max_speed_mps),
                       "forward_speed")
        add_constraint(opti.bounded(-0.75, epsi[k], 0.75), "heading_error")
        add_constraint(opti.bounded(-model.steering_limit_rad,
                                    steering[k], model.steering_limit_rad),
                       "steering_angle")
        add_constraint(opti.bounded(-model.steering_rate_radps,
                                    qdelta[k], model.steering_rate_radps),
                       "steering_rate")
        footprint = _footprint_half_width(epsi[k], model)
        add_constraint(ey[k] <= left - wall_clearance_m - footprint,
                       "left_track_boundary")
        add_constraint(ey[k] >= -right + wall_clearance_m + footprint,
                       "right_track_boundary")
        if brake_modes_array[k]:
            add_constraint(throttle[k] == 0.0, "unity_exact_zero_cawb_brake")
            for i in range(4):
                add_constraint(wheel_omega[i][k] == 0.0,
                               f"unity_brake_locked_wheel_w{i}")
        else:
            add_constraint(opti.bounded(0.015, throttle[k], 1.0),
                           "unity_positive_cawd_drive")

        fz = [normal_load[i, k] for i in range(4)]
        for i in range(4):
            add_constraint(normal_load[i, k] >= 0.05,
                           f"normal_load_w{i}")
            add_constraint(wheel_omega[i][k] >= 0.0,
                           f"forward_wheel_rotation_w{i}")
        fx, fy, mz, fx_w, _, slip_x, slip_y = tire_forces_casadi(
            model, u[k], v[k], yaw_rate[k], steering[k],
            [wheel_omega[i][k] for i in range(4)], fz)
        ax, ay = accel_xy[0, k], accel_xy[1, k]
        add_constraint(ax == fx / model.mass_kg - model.drag_per_s * u[k],
                       f"longitudinal_force_balance_k{k}")
        add_constraint(ay == fy / model.mass_kg - model.drag_per_s * v[k],
                       f"lateral_force_balance_k{k}")
        load_prediction = normal_loads_casadi(model, ax, ay)
        for i in range(4):
            add_constraint(normal_load[i, k] == load_prediction[i],
                           f"normal_load_balance_w{i}")

        # VehicleController assigns one quarter of common CAWD motor torque
        # to each wheel. Wheel angular speed is dynamic: motor torque, tire
        # reaction torque, and measured effective wheel inertia determine its
        # acceleration. The old algebraic torque-balance constraint erased
        # wheelspin transients and made throttle/slip physics impossible.
        motor_torque_per_wheel = model.motor_torque_nm * throttle[k] / 4.0
        wheel_spin_spatial_rhs.append(tuple(
            (motor_torque_per_wheel
             - model.wheel_radius_m * fx_w[i]
             - model.wheel_damping_nms * wheel_omega[i][k])
            / (model.wheel_inertia_kgm2 * sdot)
            for i in range(4)))

        longitudinal_dot = ax + yaw_rate[k] * v[k]
        lateral_dot = ay - yaw_rate[k] * u[k]
        yaw_dot = mz / model.yaw_inertia_kgm2 - 0.1 * yaw_rate[k]
        spatial_rhs.append(ca.vertcat(
            (u[k] * ca.sin(epsi[k]) + v[k] * ca.cos(epsi[k])) / sdot,
            yaw_rate[k] / sdot - curvature,
            longitudinal_dot / sdot,
            lateral_dot / sdot,
            yaw_dot / sdot,
            qdelta[k] / sdot))
        time_density.append(1.0 / sdot)
        slip_x_nodes.append(slip_x)
        slip_y_nodes.append(slip_y)
        force_nodes.append((fx, fy, mz))

    lap_time = 0.0
    dynamic_indices = (0, 1, 2, 3, 4, 9)
    for k in range(n):
        kp = (k + 1) % n
        xk = ca.vertcat(*[x[index, k] for index in dynamic_indices])
        xkp = ca.vertcat(*[x[index, kp] for index in dynamic_indices])
        ds = step_nodes[k]
        add_constraint(
            xkp - xk == 0.5 * ds * (spatial_rhs[k] + spatial_rhs[kp]),
            [f"periodic_collocation_state_{name}" for name in STATE_NAMES])
        if not brake_modes_array[k] and not brake_modes_array[kp]:
            for i in range(4):
                add_constraint(
                    wheel_omega[i][kp] - wheel_omega[i][k]
                    == 0.5 * ds * (wheel_spin_spatial_rhs[k][i]
                                   + wheel_spin_spatial_rhs[kp][i]),
                    f"periodic_collocation_{WHEEL_STATE_NAMES[i]}_k{k}")
        elif brake_modes_array[k] != brake_modes_array[kp]:
            # The source controller applies CAWB at 428 N m and source
            # captures show all four WheelCollider RPM values reach exactly
            # zero by the following 1 ms sample. Preserve that measured mode
            # switch instead of inventing an unobserved brake transient.
            for i in range(4):
                add_constraint(wheel_omega[i][kp] == 0.0,
                               f"unity_brake_mode_transition_wheel_{i}_k{k}")
        lap_time += 0.5 * ds * (time_density[k] + time_density[kp])

    # The captured accelerations are ground truth, not guaranteed to satisfy
    # this calibrated tire model. Reconcile only the algebraic seed values
    # (loads, acceleration, and common drive torque) with the model while
    # keeping captured motion states and wheel speeds as the initial trajectory.
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
    for iteration in range(8):
        force_state = np.vstack((seed_u, seed_v, seed_yaw, seed_steering,
                                 seed_omega, seed_load))
        force_values = np.asarray(seed_force_map(force_state),
                                  dtype=np.float64)
        ax_values = (force_values[0] / model.mass_kg
                     - model.drag_per_s * seed_u)
        ay_values = (force_values[1] / model.mass_kg
                     - model.drag_per_s * seed_v)
        longitudinal = (model.mass_kg * model.body_com_height_m * ax_values
                        / (2.0 * model.wheelbase_m))
        lateral = (model.mass_kg * model.body_com_height_m * ay_values
                   / (4.0 * model.track_m))
        front_sign = np.asarray((-1.0, -1.0, 1.0, 1.0))[:, None]
        side_sign = np.asarray((-1.0, 1.0, -1.0, 1.0))[:, None]
        load_values = (np.asarray(model.static_normal_load_n)[:, None]
                       + front_sign * longitudinal[None, :]
                       + side_sign * lateral[None, :])
        load_values = np.maximum(load_values, 0.05)
        required_per_wheel = (
            model.wheel_radius_m * force_values[2:6]
            + model.wheel_damping_nms * seed_omega)
        throttle_values = np.clip(
            4.0 * np.mean(required_per_wheel, axis=0)
            / model.motor_torque_nm, 0.015, 1.0)
        throttle_values[brake_modes_array] = 0.0
        max_change = max(
            float(np.max(np.abs(load_values - seed_load))),
            float(np.max(np.abs(ax_values - seed_ax))),
            float(np.max(np.abs(ay_values - seed_ay))),
            float(np.max(np.abs(throttle_values - seed_throttle))))
        seed_load = load_values
        seed_ax = ax_values
        seed_ay = ay_values
        seed_throttle = throttle_values
        opti.set_initial(normal_load, seed_load)
        opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))
        opti.set_initial(throttle, seed_throttle)
        initialization_consistency_iterations = iteration + 1
        if max_change < 1.0e-5:
            break

    # First find a model-consistent lap near the measured motion. Starting the
    # minimum-time objective from an infeasible capture made IPOPT trade path
    # speed against large dynamics defects and stall at a non-feasible point.
    # This feasibility stage is only an initializer; it does not constrain the
    # subsequent free-path minimum-time solve.
    seed_x_values = np.vstack((seed_ey, seed_epsi, seed_u, seed_v, seed_yaw,
                               seed_omega, seed_steering))
    state_scales = ca.repmat(ca.DM((0.25, 0.35, 2.0, 1.0, 2.0,
                                   150.0, 150.0, 150.0, 150.0, 0.35)),
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
                       "optimality_proven": False},
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
            "max_iter": 8000,
            "max_cpu_time": float(max_solver_seconds),
            "tol": 1.0e-6,
            "acceptable_tol": 1.0e-5,
            "print_level": int(print_level),
        },
    }, {})
    started = time.perf_counter()
    solver_error = None
    stage_reports: list[dict[str, Any]] = []
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

    if (stage1_diagnostics.get("max_constraint_violation") is not None
            and stage1_diagnostics["max_constraint_violation"] <= 5.0e-4):
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
        opti.minimize(lap_time)
        opti.solver("ipopt", {
            "expand": True,
            "ipopt": {
                "max_iter": 8000,
                "max_cpu_time": float(max_solver_seconds),
                "tol": 1.0e-6,
                "acceptable_tol": 1.0e-5,
                "hessian_approximation": "limited-memory",
                "limited_memory_max_history": 20,
                "print_level": int(print_level),
            },
        }, {})
        try:
            stage2_solution = opti.solve_limited()
            stage2_error = None
        except RuntimeError as error:
            stage2_solution = None
            stage2_error = str(error)
        stage2_stats = opti.stats()
        stage2_diagnostics = stage_constraint_diagnostics(
            stage2_solution, stage2_stats)
        stage_reports.append({
            "name": "free-geometry minimum time",
            **_solver_status(stage2_stats),
            "exception": stage2_error,
            "objective_lap_time_s": float(stage_values(
                stage2_solution, stage2_stats, lap_time)),
            "max_constraint_violation": stage2_diagnostics.get(
                "max_constraint_violation"),
            "last_primal_infeasibility": (
                float(stage2_stats["iterations"]["inf_pr"][-1])
                if stage2_stats.get("iterations", {}).get("inf_pr") else None),
            "last_dual_infeasibility": (
                float(stage2_stats["iterations"]["inf_du"][-1])
                if stage2_stats.get("iterations", {}).get("inf_du") else None),
            "worst_constraints": stage2_diagnostics.get(
                "worst_constraints", [])[:12],
        })
        if (stage2_diagnostics.get("max_constraint_violation") is not None
                and stage2_diagnostics["max_constraint_violation"] <= 5.0e-4):
            solution = stage2_solution
            stats = stage2_stats
            final_constraint_diagnostics = stage2_diagnostics
            solver_error = stage2_error
            selected_stage = "free-geometry minimum time"
        else:
            # Never discard a feasible initializer because the faster stage
            # failed. It remains a separate free path candidate, but is not
            # represented as a minimum-time result.
            solution = stage1_solution
            stats = stage1_stats
            final_constraint_diagnostics = stage1_diagnostics
            solver_error = stage2_error
            selected_stage = "captured-motion feasibility fallback"

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
                       "optimality_proven": False},
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
    feasible_incumbent = bool(finite and max_violation <= 5.0e-4)
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
        "optimality_proven": bool(stats.get("success", False)),
    })
    report = {
        "optimizer": "CasADi/IPOPT phase-switched spatial direct collocation",
        "path_initialization": "map-derived centerline only; reference path is not constrained",
        "decision_variables": {
            "differential_states": list(STATE_NAMES),
            "dynamic_states": [*STATE_NAMES, *WHEEL_STATE_NAMES],
            "algebraic_states": ["four normal loads and planar accelerations"],
            "controls": ["physical steering rate", "positive normalized motor torque", "exact-zero CAWB brake phases"],
            "path_geometry": "independent lateral offset at every optimized spatial node",
            "brake_phase_lengths_are_optimized": True,
            "brake_phase_order_and_count": "seeded from measured training run; switch locations and durations are free",
        },
        "unity_vehicle_model": model.metadata(),
        "track": track.to_dict(),
        "wall_clearance_m": wall_clearance_m,
        "supported_speed_domain_mps": [min_speed_mps, max_speed_mps],
        "brake_mode_seed_source": brake_mode_seed_source,
        "brake_phase_seed_ranges_and_modes": [
            {"seed_start_node": begin, "seed_end_exclusive_node": end,
             "mode": "brake" if mode else "drive",
             "optimized_mesh_nodes": int(count),
             "seed_length_m": float(length)}
            for (begin, end), mode, count, length in zip(
                phase_ranges, phase_modes, phase_nodes, phase_seed_lengths)],
        "model_consistent_initialization_iterations": (
            initialization_consistency_iterations),
        "optimized_phase_lengths_m": value(phase_lengths).reshape(-1).tolist()
            if finite else None,
        "optimized_brake_nodes": int(np.count_nonzero(brake_modes_array)),
        "predicted_lap_time_s": float(value(lap_time)) if finite else None,
        "elapsed_s": elapsed,
        "solve_stages": stage_reports,
        "selected_stage": selected_stage,
        "solver": solver_report,
        "nlp_diagnostics": nlp_diagnostics,
        "limits_and_caveats": [
            "The optimizer is a new free-geometry path-and-speed OCP, not a speed-scaled reference trajectory.",
            "The measured capture seeds only the number/order of drive and exact-zero brake phases; IPOPT optimizes every phase length.",
            "Public Unity VehicleController, active serialized vehicle values, source capture friction curves, and source-derived rigid-body equations are used; the native WheelCollider solver remains unavailable.",
            "Tire-force response and normal-load transfer are calibrated/quasi-static approximations; Unity replay remains the truth gate.",
            "Drive-mode wheel spin is integrated as a dynamic state using source CAWD torque split, tire reaction torque, fitted effective wheel inertia, and damping. Exact-zero CAWB mode locks wheels at zero as observed in the 1 kHz source capture.",
            "IPOPT establishes a local solution only; the objective is not a global-minimum or Unity lap-time guarantee.",
        ],
    }
    if not feasible_incumbent:
        return {}, report

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
    report["constraint_diagnostics"] = {
        "minimum_centerline_side_clearance_after_footprint_m": float(min(
            np.min(np.interp(station_values, np.r_[track.s, track.length],
                             np.r_[track.left, track.left[0]])
                   - wall_clearance_m - _physical_footprint(states[1])
                   - states[0]),
            np.min(np.interp(station_values, np.r_[track.s, track.length],
                             np.r_[track.right, track.right[0]])
                   - wall_clearance_m - _physical_footprint(states[1])
                   + states[0]))),
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
        "peak_abs_lateral_acceleration_mps2": float(np.max(np.abs(acceleration[1]))),
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
    return candidate, report


def _physical_footprint(epsi: np.ndarray) -> np.ndarray:
    return np.maximum(0.15, 0.1365 * np.abs(np.cos(epsi))
                      + 0.255 * np.abs(np.sin(epsi)))


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
    columns = list(candidate)
    with state_path.open("w", encoding="utf-8") as stream:
        stream.write(",".join([
            "s_ref_m", "x_m", "y_m", "ey_m", "epsi_rad", "speed_mps",
            "lateral_speed_mps", "yaw_rate_radps", "steering_rad",
            "steering_rate_radps", "throttle_norm", "ax_mps2", "ay_mps2",
            "normal_load_fl_n", "normal_load_fr_n", "normal_load_rl_n",
            "normal_load_rr_n", "slipx_fl", "slipx_fr", "slipx_rl",
            "slipx_rr", "slipy_fl", "slipy_fr", "slipy_rl", "slipy_rr",
            "kappa_path_per_m",
        ]) + "\n")
        for k in range(n):
            values = [candidate["s_ref_m"][k], candidate["x_m"][k],
                      candidate["y_m"][k], candidate["ey_m"][k],
                      candidate["epsi_rad"][k], candidate["speed_mps"][k],
                      candidate["lateral_speed_mps"][k],
                      candidate["yaw_rate_radps"][k],
                      candidate["steering_rad"][k],
                      candidate["steering_rate_radps"][k],
                      candidate["throttle_norm"][k], candidate["ax_mps2"][k],
                      candidate["ay_mps2"][k],
                      *candidate["normal_load_n"][:, k],
                      *candidate["slip_x"][:, k], *candidate["slip_y"][:, k],
                      candidate["path_curvature_per_m"][k]]
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

