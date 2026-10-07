from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import time
from typing import Any

import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline

from .model import VehicleModel, LateralEnvelope, smooth_abs, wrap_angle_np
from .track import Track, build_track_from_xy_widths, load_centerline_csv


@dataclass
class Guess:
    ey: np.ndarray
    epsi: np.ndarray
    u: np.ndarray
    r: np.ndarray
    vt: np.ndarray
    delta: np.ndarray
    qdelta: np.ndarray
    qv: np.ndarray
    label: str
    actual_delta: np.ndarray | None = None


@dataclass
class Solution:
    track: Track
    ey: np.ndarray
    epsi: np.ndarray
    u: np.ndarray
    r: np.ndarray
    vt: np.ndarray
    delta: np.ndarray
    qdelta: np.ndarray
    qv: np.ndarray
    lap_time_s: float
    objective: float
    solver_stats: dict[str, Any]
    elapsed_s: float
    label: str
    converged: bool = True
    # When present, ``delta`` is physical steering and this stores the
    # rate-controlled command. Legacy solutions keep the original single
    # steering state and leave this unset.
    steering_command: np.ndarray | None = None


class SolveTrackFailure(RuntimeError):
    """IPOPT returned a usable iterate, but not an accepted optimum."""

    def __init__(self, message: str, candidate: Solution):
        super().__init__(message)
        self.candidate = candidate


def _optimizer_speed_limit(model: VehicleModel, envelope: LateralEnvelope) -> float:
    return min(model.max_body_speed_mps, envelope.max_supported_speed_mps)


def _yaw_lag_interval_numeric(r0: float, target0: float, target1: float,
                              dt: float, tau: float) -> tuple[float, float]:
    """Exact first-order yaw-lag step for a linearly varying steady response.

    Returns the interval-end yaw rate and its time integral.  The latter is
    used by the Frenet heading equation, so yaw lag is not re-discretized by a
    coarse spatial trapezoid.
    """
    h = dt / tau
    decay = math.exp(-h)
    one_minus_decay = -math.expm1(-h)
    target1_weight = 1.0 - one_minus_decay / h
    r1 = (decay * r0 + one_minus_decay * target0
          + target1_weight * (target1 - target0))
    yaw_integral = 0.5 * dt * (target0 + target1) - tau * (r1 - r0)
    return r1, yaw_integral


def _yaw_lag_interval_casadi(r0, target0, target1, dt, tau):
    """CasADi equivalent of :func:`_yaw_lag_interval_numeric`."""
    h = dt / tau
    decay = ca.exp(-h)
    one_minus_decay = 1.0 - decay
    target1_weight = 1.0 - one_minus_decay / h
    r1 = (decay * r0 + one_minus_decay * target0
          + target1_weight * (target1 - target0))
    yaw_integral = 0.5 * dt * (target0 + target1) - tau * (r1 - r0)
    return r1, yaw_integral


def _smooth_abs_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.sqrt(x * x + eps * eps)


def _vehicle_lateral_extent(epsi: np.ndarray, model: VehicleModel) -> np.ndarray:
    """Minimum side footprint used by the planner.

    Match the existing optimize_trajectory.py safety semantics by never using
    less than half of mintime.optimizer_width_m (0.30 m in the current repo).
    When the vehicle is angled relative to the path, additionally protect the
    actual 0.510 x 0.273 m AutoDRIVE rectangular body.
    """
    physical = (
        0.5 * model.car_width_m * np.abs(np.cos(epsi))
        + max(model.rear_axle_to_front_bumper_m, model.rear_overhang_m)
          * np.abs(np.sin(epsi))
    )
    return np.maximum(0.5 * model.planning_footprint_width_m, physical)


def _wall_clearance(config: dict[str, Any], model: VehicleModel) -> float:
    track = config.get("track", {})
    return (model.required_wall_clearance_m
            + float(track.get("extra_wall_clearance_m", 0.0))
            + float(track.get("optimizer_geometry_buffer_m", 0.0)))


def _steering_actuator_config(config: dict[str, Any]) -> tuple[bool, float]:
    section = config.get("steering_actuator", {})
    if not isinstance(section, dict):
        raise ValueError("steering_actuator must be a mapping")
    enabled = bool(section.get("model_command_queue", False))
    delay_s = float(section.get("command_delay_s", 0.025))
    if enabled and (not math.isfinite(delay_s) or delay_s <= 0.0 or delay_s > 0.1):
        raise ValueError("steering_actuator.command_delay_s must be in (0, 0.1]")
    return enabled, delay_s


def _delayed_steering_command_casadi(
    track: Track,
    command,
    qdelta,
    sdot: list[Any],
    delay_s: float,
    max_progress_mps: float,
) -> list[Any]:
    """Interpolate the command one fixed-time interval behind each OCP node.

    The command is represented on the spatial mesh. A cubic Hermite segment
    uses the OCP command state and its time-rate-derived spatial slope, so the
    delayed signal is C1 at mesh boundaries. The lag distance uses the
    trapezoidal progress-rate estimate over the command delay.
    """
    n = track.count
    min_ds = float(np.min(track.ds))
    max_back_m = delay_s * max_progress_mps
    segments = max(1, int(math.ceil(max_back_m / min_ds)) + 1)
    if segments >= n:
        raise ValueError("steering delay window spans the entire OCP lap")
    spatial_slopes = [qdelta[i] / sdot[i] for i in range(n)]
    result = []
    for k in range(n):
        previous = (k - 1) % n
        query = track.s[k] - 0.5 * delay_s * (sdot[k] + sdot[previous])
        candidates: list[tuple[float, Any]] = []
        travelled = 0.0
        for offset in range(1, segments + 1):
            lo = (k - offset) % n
            hi = (lo + 1) % n
            width = float(track.ds[lo])
            s_hi = float(track.s[k]) - travelled
            s_lo = s_hi - width
            t = (query - s_lo) / width
            t2, t3 = t * t, t * t * t
            h00 = 2.0 * t3 - 3.0 * t2 + 1.0
            h10 = t3 - 2.0 * t2 + t
            h01 = -2.0 * t3 + 3.0 * t2
            h11 = t3 - t2
            value = (
                h00 * command[lo] + h10 * width * spatial_slopes[lo]
                + h01 * command[hi] + h11 * width * spatial_slopes[hi]
            )
            candidates.append((s_lo, value))
            travelled += width
        delayed = command[(k - segments) % n]
        for s_lo, value in reversed(candidates):
            delayed = ca.if_else(query >= s_lo, value, delayed)
        result.append(delayed)
    return result


def _optimizer_steering_rate_limit(model: VehicleModel,
                                  config: dict[str, Any]) -> float:
    constraints = config.get("optimizer_constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("optimizer_constraints must be a mapping")
    limit = float(constraints.get(
        "max_steering_rate_radps", model.max_steering_rate_radps))
    if (not math.isfinite(limit) or limit <= 0.0 or
            limit > model.max_steering_rate_radps):
        raise ValueError(
            "optimizer max_steering_rate_radps must be positive and no "
            "greater than the physical model limit")
    return limit


def _optimizer_steering_limit(model: VehicleModel,
                              config: dict[str, Any]) -> float:
    constraints = config.get("optimizer_constraints", {})
    if not isinstance(constraints, dict):
        raise ValueError("optimizer_constraints must be a mapping")
    limit = float(constraints.get("max_steering_rad", model.max_steering_rad))
    if (not math.isfinite(limit) or limit <= 0.0 or
            limit > model.max_steering_rad):
        raise ValueError(
            "optimizer max_steering_rad must be positive and no greater "
            "than the physical model limit")
    return limit


def _repair_speed_periodic(v: np.ndarray, ds: np.ndarray, model: VehicleModel,
                           iterations: int = 30) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    n = len(v)
    for _ in range(iterations):
        changed = False
        # forward acceleration limit
        for k in range(n):
            kp = (k + 1) % n
            vmax = math.sqrt(max(model.min_speed_mps ** 2,
                                 v[k] ** 2 + 2.0 * model.accel_limit_mps2 * ds[k]))
            if v[kp] > vmax:
                v[kp] = vmax
                changed = True
        # backward braking limit
        for k in range(n - 1, -1, -1):
            kp = (k + 1) % n
            brake = model.brake_intercept_mps2 + model.brake_slope_s_inv * max(v[kp], 0.0)
            vmax = math.sqrt(max(model.min_speed_mps ** 2,
                                 v[kp] ** 2 + 2.0 * brake * ds[k]))
            if v[k] > vmax:
                v[k] = vmax
                changed = True
        if not changed:
            break
    return np.clip(v, model.min_speed_mps, model.max_body_speed_mps)


def _geometry_from_offset(track: Track, ey: np.ndarray):
    x = track.x - ey * np.sin(track.psi)
    y = track.y + ey * np.cos(track.psi)
    # Differentiate w.r.t. reference s. This gives correct geometric heading/curvature.
    s_closed = np.r_[track.s, track.length]
    x_closed = np.r_[x, x[0]]
    y_closed = np.r_[y, y[0]]
    sx = CubicSpline(s_closed, x_closed, bc_type="periodic")
    sy = CubicSpline(s_closed, y_closed, bc_type="periodic")
    dx = sx(track.s, 1)
    dy = sy(track.s, 1)
    ddx = sx(track.s, 2)
    ddy = sy(track.s, 2)
    psi = np.unwrap(np.arctan2(dy, dx))
    denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12)
    kappa = (dx * ddy - dy * ddx) / denom
    epsi = wrap_angle_np(psi - track.psi)
    return x, y, epsi, kappa



def _clip_offset_to_safe_corridor(track: Track, ey: np.ndarray,
                                  model: VehicleModel,
                                  config: dict[str, Any],
                                  iterations: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project an offset guess into the same safe corridor enforced by the NLP."""
    ey = np.asarray(ey, dtype=float).copy()
    margin = _wall_clearance(config, model)
    epsi = np.zeros_like(ey)
    kappa_path = track.kappa.copy()
    for _ in range(iterations):
        _, _, epsi, kappa_path = _geometry_from_offset(track, ey)
        footprint = _vehicle_lateral_extent(epsi, model)
        left_max = track.left - margin - footprint - 1.0e-3
        right_min = -(track.right - margin - footprint - 1.0e-3)
        if np.any(left_max < right_min):
            idx = int(np.argmin(left_max - right_min))
            raise ValueError(
                "Track is too narrow for the required AutoDRIVE planning footprint + "
                f"wall clearance at s={track.s[idx]:.3f} m: "
                f"left={track.left[idx]:.3f} right={track.right[idx]:.3f} "
                f"required_aligned_center_to_wall="
                f"{0.5 * model.planning_footprint_width_m + margin:.3f} m")
        next_ey = np.minimum(np.maximum(ey, right_min), left_max)
        if np.max(np.abs(next_ey - ey)) < 1.0e-6:
            ey = next_ey
            break
        ey = next_ey
    _, _, epsi, kappa_path = _geometry_from_offset(track, ey)
    return ey, epsi, kappa_path

def build_guess(track: Track, model: VehicleModel, envelope: LateralEnvelope,
                config: dict[str, Any], mode: str = "center") -> Guess:
    steering_rate_limit = _optimizer_steering_rate_limit(model, config)
    steering_limit = _optimizer_steering_limit(model, config)
    margin = _wall_clearance(config, model)
    speed_limit = _optimizer_speed_limit(model, envelope)
    nominal_extent = 0.5 * model.planning_footprint_width_m
    left_room = np.maximum(track.left - margin - nominal_extent, 0.01)
    right_room = np.maximum(track.right - margin - nominal_extent, 0.01)

    if mode == "center":
        ey = np.zeros(track.count)
    elif mode == "left":
        ey = 0.25 * left_room
    elif mode == "right":
        ey = -0.25 * right_room
    elif mode.startswith("wave"):
        phase = 0.0
        if ":" in mode:
            phase = float(mode.split(":", 1)[1])
        amp = 0.18 * np.minimum(left_room, right_room)
        ey = amp * np.sin(2.0 * math.pi * track.s / track.length + phase)
    else:
        raise ValueError(f"Unknown initial-guess mode {mode}")

    ey, epsi, kappa_path = _clip_offset_to_safe_corridor(
        track, ey, model, config)
    # Fixed-point speed estimate because the measured envelope depends on both
    # speed and steering demand.
    u = np.full(track.count, min(6.0, speed_limit))
    for _ in range(10):
        delta = np.asarray([
            model.steering_for_yaw_rate(float(speed), float(curvature * speed),
                                        float(curvature))
            for speed, curvature in zip(u, kappa_path)
        ])
        ay_cap = envelope.numpy(u, delta)
        curve_speed = np.sqrt(np.maximum(ay_cap / np.maximum(np.abs(kappa_path), 1e-4),
                                         model.min_speed_mps ** 2))
        u_new = np.minimum(curve_speed, speed_limit)
        u_new = np.maximum(u_new, model.min_speed_mps)
        u_new = np.minimum(
            _repair_speed_periodic(u_new, track.ds, model), speed_limit)
        if np.max(np.abs(u_new - u)) < 1e-3:
            u = u_new
            break
        u = u_new

    delta = np.asarray([
        model.steering_for_yaw_rate(float(speed), float(curvature * speed),
                                    float(curvature))
        for speed, curvature in zip(u, kappa_path)
    ])
    delta = np.clip(delta, -steering_limit * 0.98, steering_limit * 0.98)
    r = kappa_path * u
    vt = np.asarray(model.steady_target_speed(u), dtype=float)
    vt = np.clip(vt, 0.0, min(model.max_command_speed_mps, speed_limit))

    # Convert spatial changes into time rates using local ds/dt ≈ u.
    def periodic_rate(z, clip_lo, clip_hi):
        dz = np.roll(z, -1) - z
        # wrap angle-like delta differences only where useful
        rate = dz / np.maximum(track.ds, 1e-6) * u
        return np.clip(rate, clip_lo, clip_hi)

    qdelta = periodic_rate(delta, -steering_rate_limit, steering_rate_limit)
    qv = periodic_rate(vt, -model.max_target_speed_rate_reduction_mps2,
                       model.max_target_speed_rate_increase_mps2)
    return Guess(ey=ey, epsi=epsi, u=u, r=r, vt=vt, delta=delta,
                 qdelta=qdelta, qv=qv, label=mode)


def _project_warm_raceline_to_track(track: Track, warm_csv: str | Path,
                                    model: VehicleModel, envelope: LateralEnvelope,
                                    config: dict[str, Any]) -> Guess:
    steering_rate_limit = _optimizer_steering_rate_limit(model, config)
    steering_limit = _optimizer_steering_limit(model, config)
    raw = []
    with open(warm_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                vals = [float(v) for v in line.split(",")]
            except ValueError:
                continue
            if len(vals) >= 6:
                raw.append(vals)
    if len(raw) < 20:
        raise ValueError(f"Warm raceline {warm_csv} has too few usable rows")
    a = np.asarray(raw, dtype=float)
    warm_xy = a[:, 1:3]
    warm_speed = a[:, 5]
    warm_heading = a[:, 3]

    # Dense nearest reference is robust enough for a closed track of this size.
    ref_xy = np.column_stack([track.x, track.y])
    samples_s = []
    samples_ey = []
    samples_speed = []
    samples_epsi = []
    for p, speed, heading in zip(warm_xy, warm_speed, warm_heading):
        d2 = np.sum((ref_xy - p) ** 2, axis=1)
        i = int(np.argmin(d2))
        left_n = np.array([-math.sin(track.psi[i]), math.cos(track.psi[i])])
        ey = float(np.dot(p - ref_xy[i], left_n))
        samples_s.append(track.s[i])
        samples_ey.append(ey)
        samples_speed.append(speed)
        samples_epsi.append(math.atan2(math.sin(heading - track.psi[i]),
                                       math.cos(heading - track.psi[i])))

    order = np.argsort(samples_s)
    ss = np.asarray(samples_s)[order]
    eyv = np.asarray(samples_ey)[order]
    uv = np.asarray(samples_speed)[order]
    epv = np.asarray(samples_epsi)[order]
    # Collapse duplicate reference indices by simple averaging.
    uniq, inv = np.unique(np.round(ss, 6), return_inverse=True)
    def avg(vals):
        out = np.zeros(len(uniq))
        cnt = np.zeros(len(uniq))
        for j, idx in enumerate(inv):
            out[idx] += vals[j]
            cnt[idx] += 1
        return out / np.maximum(cnt, 1)
    eyu = avg(eyv)
    uu = avg(uv)
    epu = avg(epv)

    def pinterp(vals):
        s_ext = np.r_[uniq - track.length, uniq, uniq + track.length]
        v_ext = np.r_[vals, vals, vals]
        return np.interp(track.s, s_ext, v_ext)

    ey = pinterp(eyu)
    speed_limit = _optimizer_speed_limit(model, envelope)
    u = np.clip(pinterp(uu), model.min_speed_mps, speed_limit)

    # The production line is only an initializer. Re-project it into V1.3's
    # stricter repo-derived wall corridor and rebuild all dependent states so
    # IPOPT does not begin from a path/dynamics mismatch.
    ey, epsi, kappa_path = _clip_offset_to_safe_corridor(
        track, ey, model, config)
    for _ in range(8):
        delta = np.asarray([
            model.steering_for_yaw_rate(float(speed), float(curvature * speed),
                                        float(curvature))
            for speed, curvature in zip(u, kappa_path)
        ])
        delta = np.clip(delta, -steering_limit * 0.98,
                        steering_limit * 0.98)
        ay_cap = envelope.numpy(u, delta)
        curve_cap = np.sqrt(np.maximum(
            ay_cap / np.maximum(np.abs(kappa_path), 1.0e-5),
            model.min_speed_mps ** 2))
        u_new = np.minimum(u, np.minimum(0.995 * curve_cap, speed_limit))
        u_new = _repair_speed_periodic(u_new, track.ds, model, iterations=20)
        u_new = np.minimum(u_new, speed_limit)
        if np.max(np.abs(u_new - u)) < 1.0e-4:
            u = u_new
            break
        u = u_new

    delta = np.asarray([
        model.steering_for_yaw_rate(float(speed), float(curvature * speed),
                                    float(curvature))
        for speed, curvature in zip(u, kappa_path)
    ])
    delta = np.clip(delta, -steering_limit * 0.98, steering_limit * 0.98)
    r = kappa_path * u
    vt = np.clip(model.steady_target_speed(u), 0.0,
                 min(model.max_command_speed_mps, speed_limit))
    qdelta = np.clip((np.roll(delta, -1) - delta) / np.maximum(track.ds, 1e-6) * u,
                     -steering_rate_limit, steering_rate_limit)
    qv = np.clip((np.roll(vt, -1) - vt) / np.maximum(track.ds, 1e-6) * u,
                 -model.max_target_speed_rate_reduction_mps2,
                 model.max_target_speed_rate_increase_mps2)
    return Guess(ey, epsi, u, r, vt, delta, qdelta, qv, "warm_raceline")


def _resample_solution_guess(sol: Solution, new_track: Track) -> Guess:
    old_s = sol.track.s
    L_old = sol.track.length
    q = new_track.s / new_track.length * L_old
    def interp(v):
        s_ext = np.r_[old_s - L_old, old_s, old_s + L_old]
        v_ext = np.r_[v, v, v]
        return np.interp(q, s_ext, v_ext)
    steering_seed = (sol.steering_command
                     if sol.steering_command is not None else sol.delta)
    return Guess(
        interp(sol.ey), interp(sol.epsi), interp(sol.u), interp(sol.r),
        interp(sol.vt), interp(steering_seed), interp(sol.qdelta),
        interp(sol.qv), f"refined_from_{sol.label}",
        actual_delta=interp(sol.delta) if sol.steering_command is not None else None)


def _load_solution_nodes_guess(track: Track, nodes_csv: str | Path) -> Guess:
    """Resample a previous OCP solution, preserving its dynamic-state seed."""
    required = ("s_ref_m", "ey_m", "epsi_rad", "u_mps", "r_radps",
                "target_mps", "delta_rad", "qdelta_radps", "qv_mps2")
    rows: list[dict[str, float]] = []
    with Path(nodes_csv).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"{nodes_csv} does not contain optimizer state columns")
        fields = list(required)
        if "steering_command_rad" in reader.fieldnames:
            fields.append("steering_command_rad")
        for row in reader:
            rows.append({name: float(row[name]) for name in fields})
    if len(rows) < 20:
        raise ValueError(f"Optimizer state seed {nodes_csv} has too few rows")
    data = {name: np.asarray([row[name] for row in rows], dtype=float)
            for name in required}
    # New queue-aware solutions export command and physical angle separately.
    # Older production seeds only have delta_rad, which remains a valid
    # initialization for both states.
    old_s = data["s_ref_m"]
    if (not all(np.all(np.isfinite(values)) for values in data.values()) or
            np.any(np.diff(old_s) <= 0.0) or old_s[0] < -1.0e-8):
        raise ValueError(f"Optimizer state seed {nodes_csv} has invalid values/order")
    old_length = float(old_s[-1] + np.median(np.diff(old_s)))
    if old_length <= old_s[-1] or track.length <= 0.0:
        raise ValueError(f"Optimizer state seed {nodes_csv} has invalid lap length")
    query = track.s / track.length * old_length

    def periodic(values: np.ndarray) -> np.ndarray:
        return np.interp(query,
                         np.r_[old_s - old_length, old_s, old_s + old_length],
                         np.r_[values, values, values])

    return Guess(
        ey=periodic(data["ey_m"]), epsi=periodic(data["epsi_rad"]),
        u=periodic(data["u_mps"]), r=periodic(data["r_radps"]),
        vt=periodic(data["target_mps"]),
        delta=periodic(data.get("steering_command_rad", data["delta_rad"])),
        qdelta=periodic(data["qdelta_radps"]), qv=periodic(data["qv_mps2"]),
        label="warm_solution_nodes",
        actual_delta=(periodic(data["delta_rad"])
                      if "steering_command_rad" in data else None))


def _periodic_spatial_derivative(z: np.ndarray, track: Track) -> np.ndarray:
    """Second-order-ish periodic derivative on the nearly uniform optimizer mesh."""
    z = np.asarray(z, dtype=float)
    ds_f = np.maximum(track.ds, 1.0e-9)
    ds_b = np.maximum(np.roll(track.ds, 1), 1.0e-9)
    return (np.roll(z, -1) - np.roll(z, 1)) / (ds_f + ds_b)


def _solver_summary(stats: dict[str, Any]) -> dict[str, Any]:
    iterations = stats.get("iterations") or {}
    def _last(name):
        values = iterations.get(name)
        if isinstance(values, (list, tuple)) and values:
            try:
                return float(values[-1])
            except Exception:
                return None
        return None
    return {
        "return_status": stats.get("return_status"),
        "success": bool(stats.get("success", False)),
        "unified_return_status": stats.get("unified_return_status"),
        "iter_count": int(stats.get("iter_count", 0) or 0),
        "final_inf_pr": _last("inf_pr"),
        "final_inf_du": _last("inf_du"),
        "final_mu": _last("mu"),
        "final_objective": _last("obj"),
        "t_wall_total_s": float(stats.get("t_wall_total", 0.0) or 0.0),
    }


def diagnose_solution(solution: Solution, model: VehicleModel,
                      envelope: LateralEnvelope, config: dict[str, Any]) -> dict[str, Any]:
    """Independent numerical diagnostics for solved or failed IPOPT iterates."""
    tr = solution.track
    ey = np.asarray(solution.ey)
    epsi = np.asarray(solution.epsi)
    u = np.asarray(solution.u)
    r = np.asarray(solution.r)
    vt = np.asarray(solution.vt)
    delta = np.asarray(solution.delta)
    actuator_enabled, steering_delay_s = _steering_actuator_config(config)
    command_delta = (np.asarray(solution.steering_command)
                     if solution.steering_command is not None else delta)
    qdelta = np.asarray(solution.qdelta)
    qv = np.asarray(solution.qv)
    steering_rate_limit = _optimizer_steering_rate_limit(model, config)
    steering_limit = _optimizer_steering_limit(model, config)
    speed_limit = _optimizer_speed_limit(model, envelope)
    wall_margin = _wall_clearance(config, model)
    min_den = float(config.get("numerics", {}).get("min_frenet_denominator", 0.25))
    min_progress = float(config.get("numerics", {}).get("min_progress_mps", 0.30))
    use_lateral_velocity = bool(config.get("lateral_dynamics", {}).get(
        "use_odometry_lateral_velocity", False))
    max_heading = float(config.get("limits", {}).get("max_heading_error_rad", 0.75))
    scales = config.get("scaling", {})
    sx = np.asarray([
        float(scales.get("ey_m", 0.5)),
        float(scales.get("epsi_rad", 0.5)),
        float(scales.get("u_mps", 10.0)),
        float(scales.get("r_radps", 10.0)),
        float(scales.get("target_mps", 10.0)),
        float(scales.get("delta_rad", 0.5)),
    ], dtype=float)

    den = 1.0 - tr.kappa * ey
    body_v = (model.lateral_velocity_numeric(u, r) if use_lateral_velocity
              else np.zeros_like(u))
    sdot = (u * np.cos(epsi) - body_v * np.sin(epsi)) / np.maximum(den, 1.0e-9)
    raw_a = np.asarray(model.raw_longitudinal_accel(u, vt, qv), dtype=float)
    if actuator_enabled:
        delay_progress_rate = 0.5 * (sdot + np.roll(sdot, 1))
        delay_query = tr.s - steering_delay_s * delay_progress_rate
        s_extended = np.r_[tr.s - 2.0 * tr.length, tr.s - tr.length,
                           tr.s, tr.s + tr.length, tr.s + 2.0 * tr.length]
        command_extended = np.tile(command_delta, 5)
        slope_extended = np.tile(qdelta / np.maximum(sdot, 1.0e-8), 5)
        delayed_command = np.empty(tr.count, dtype=float)
        for k, query in enumerate(delay_query):
            interval = int(np.searchsorted(s_extended, query, side="right") - 1)
            interval = min(max(interval, 0), len(s_extended) - 2)
            width = s_extended[interval + 1] - s_extended[interval]
            t = (query - s_extended[interval]) / width
            t2, t3 = t * t, t * t * t
            delayed_command[k] = (
                (2.0 * t3 - 3.0 * t2 + 1.0) * command_extended[interval]
                + (t3 - 2.0 * t2 + t) * width * slope_extended[interval]
                + (-2.0 * t3 + 3.0 * t2) * command_extended[interval + 1]
                + (t3 - t2) * width * slope_extended[interval + 1]
            )
        actual_rate = np.clip(
            (delayed_command - delta) / steering_delay_s,
            -steering_rate_limit, steering_rate_limit)
    else:
        actual_rate = np.zeros_like(delta)
    yaw_residual_rate = np.asarray([
        model.yaw_residual_rate_numeric(
            float(speed), float(yaw), float(steering), float(rate),
            float(model.lateral_velocity_numeric(speed, yaw)), float(curvature),
            float(ax), float(speed_rate), float(target))
        for speed, yaw, steering, rate, curvature, ax, speed_rate, target in zip(
            u, r, delta, qdelta, tr.kappa, raw_a, qv, vt)
    ])
    r_ss = np.asarray([
        model.steady_yaw_rate(float(speed), float(steering), float(curvature))
        + model.yaw_tau_s * float(residual)
        for speed, steering, curvature, residual in zip(
            u, delta, tr.kappa, yaw_residual_rate)
    ])
    safe_sdot = np.maximum(sdot, 1.0e-8)
    state_fields = [ey, epsi, u, r, vt]
    derivative_fields = [
        (u * np.sin(epsi) + body_v * np.cos(epsi)) / safe_sdot,
        np.zeros_like(r),
        raw_a / safe_sdot,
        np.zeros_like(r),
        qv / safe_sdot,
    ]
    if actuator_enabled:
        state_fields.extend([command_delta, delta])
        derivative_fields.extend([qdelta / safe_sdot, actual_rate / safe_sdot])
        state_scales = np.r_[sx, float(scales.get("actual_delta_rad",
                                                scales.get("delta_rad", 0.5)))]
        state_names = ("ey", "epsi", "u", "r", "target",
                       "steering_command", "actual_steering")
    else:
        state_fields.append(delta)
        derivative_fields.append(qdelta / safe_sdot)
        state_scales = sx
        state_names = ("ey", "epsi", "u", "r", "target", "delta")
    f = np.vstack(derivative_fields)
    X = np.vstack(state_fields)
    defects = np.zeros_like(X)
    for k in range(tr.count):
        kp = (k + 1) % tr.count
        defects[:, k] = X[:, kp] - X[:, k] - 0.5 * tr.ds[k] * (f[:, k] + f[:, kp])
        interval_dt = tr.ds[k] / max(0.5 * (safe_sdot[k] + safe_sdot[kp]), 1.0e-8)
        yaw_end, yaw_integral = _yaw_lag_interval_numeric(
            r[k], r_ss[k], r_ss[kp], interval_dt, model.yaw_tau_s)
        defects[1, k] = (
            epsi[kp] - epsi[k] - yaw_integral
            + 0.5 * (tr.kappa[k] + tr.kappa[kp]) * tr.ds[k])
        defects[3, k] = r[kp] - yaw_end
    scaled = defects / state_scales[:, None]

    footprint = _vehicle_lateral_extent(epsi, model)
    ay = u * r
    ay_cap = np.asarray(envelope.numpy(u, delta), dtype=float)
    brake = np.asarray(model.brake_limit(u), dtype=float)
    combined = config.get("combined_acceleration", {})
    if bool(combined.get("enabled", False)):
        exponent = float(combined.get("exponent", 2.0))
        ax_norm = np.where(raw_a >= 0.0,
                           raw_a / model.accel_limit_mps2,
                           -raw_a / brake)
        ay_norm = _smooth_abs_np(ay) / ay_cap
        combined_slack_min: float | None = float(np.nanmin(
            1.0 - np.abs(ax_norm) ** exponent - np.abs(ay_norm) ** exponent))
    else:
        combined_slack_min = None
    return {
        "solver": _solver_summary(solution.solver_stats),
        "lap_time_s": float(solution.lap_time_s),
        "objective": float(solution.objective),
        "converged": bool(solution.converged),
        "yaw_residual_candidate_id": (
            model.yaw_residual_candidate.get("model_id")
            if model.yaw_residual_candidate is not None else None),
        "yaw_residual_rate_radps2": {
            "active_node_count": int(np.count_nonzero(np.abs(yaw_residual_rate) > 1e-9)),
            "max_abs": float(np.max(np.abs(yaw_residual_rate))),
            "p95_abs": float(np.percentile(np.abs(yaw_residual_rate), 95)),
        },
        "max_abs_scaled_collocation_defect": float(np.nanmax(np.abs(scaled))),
        "scaled_collocation_defect_by_state_max": {
            name: float(np.nanmax(np.abs(scaled[i])))
            for i, name in enumerate(state_names)
        },
        "raw_collocation_defect_by_state_max": {
            name: float(np.nanmax(np.abs(defects[i])))
            for i, name in enumerate(state_names)
        },
        "minimum_slacks": {
            "frenet_denominator": float(np.nanmin(den - min_den)),
            "progress_mps": float(np.nanmin(sdot - min_progress)),
            "body_speed_upper_mps": float(speed_limit - np.nanmax(u)),
            "target_speed_upper_mps": float(
                min(model.max_command_speed_mps, speed_limit) - np.nanmax(vt)),
            "lateral_accel_mps2": float(np.nanmin(ay_cap - np.abs(ay))),
            "combined_acceleration": combined_slack_min,
            "accel_upper_mps2": float(np.nanmin(model.accel_limit_mps2 - raw_a)),
            "brake_lower_mps2": float(np.nanmin(raw_a + brake)),
            "left_body_clearance_m": float(np.nanmin(tr.left - wall_margin - footprint - ey)),
            "right_body_clearance_m": float(np.nanmin(tr.right - wall_margin - footprint + ey)),
            "heading_error_rad": float(max_heading - np.nanmax(np.abs(epsi))),
            "steering_rad": float(steering_limit - np.nanmax(np.abs(delta))),
            "steering_command_rad": float(
                steering_limit - np.nanmax(np.abs(command_delta))),
            "steering_rate_radps": float(steering_rate_limit - np.nanmax(np.abs(qdelta))),
            "target_rate_increase_mps2": float(model.max_target_speed_rate_increase_mps2 - np.nanmax(qv)),
            "target_rate_reduction_mps2": float(model.max_target_speed_rate_reduction_mps2 + np.nanmin(qv)),
        },
    }


def save_attempt_artifacts(solution: Solution, output_dir: str | Path,
                           model: VehicleModel, envelope: LateralEnvelope,
                           config: dict[str, Any], prefix: str = "attempt") -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = solution.track
    node_path = out / f"{prefix}_nodes.csv"
    fields = ["s_ref_m", "x_ref_m", "y_ref_m", "kappa_ref", "ey_m",
              "epsi_rad", "u_mps", "r_radps", "target_mps", "delta_rad"]
    if solution.steering_command is not None:
        fields.append("steering_command_rad")
    fields.extend(["qdelta_radps", "qv_mps2", "left_m", "right_m"])
    with node_path.open("w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for i in range(tr.count):
            values = [tr.s[i], tr.x[i], tr.y[i], tr.kappa[i], solution.ey[i],
                      solution.epsi[i], solution.u[i], solution.r[i],
                      solution.vt[i], solution.delta[i]]
            if solution.steering_command is not None:
                values.append(solution.steering_command[i])
            values.extend([solution.qdelta[i], solution.qv[i],
                           tr.left[i], tr.right[i]])
            f.write(",".join(f"{value:.9f}" for value in values) + "\n")
    diag = diagnose_solution(solution, model, envelope, config)
    diag["solution_label"] = solution.label
    diag["nodes_file"] = str(node_path)
    (out / f"{prefix}_report.json").write_text(
        json.dumps(diag, indent=2, default=str), encoding="utf-8")
    return diag


def solve_track(track: Track, model: VehicleModel, envelope: LateralEnvelope,
                config: dict[str, Any], guess: Guess,
                max_iter_override: int | None = None) -> Solution:
    n = track.count
    solver = config.get("solver", {})
    yaw_lag_exact_fraction = float(solver.get("yaw_lag_exact_fraction", 1.0))
    if (not math.isfinite(yaw_lag_exact_fraction) or
            not 0.0 <= yaw_lag_exact_fraction <= 1.0):
        raise ValueError("solver.yaw_lag_exact_fraction must be within [0, 1]")
    opti = ca.Opti()
    steering_rate_limit = _optimizer_steering_rate_limit(model, config)
    steering_limit = _optimizer_steering_limit(model, config)
    actuator_enabled, steering_delay_s = _steering_actuator_config(config)

    scales = config.get("scaling", {})
    sx_values = [
        float(scales.get("ey_m", 0.5)),
        float(scales.get("epsi_rad", 0.5)),
        float(scales.get("u_mps", 10.0)),
        float(scales.get("r_radps", 10.0)),
        float(scales.get("target_mps", 10.0)),
        float(scales.get("delta_rad", 0.5)),
    ]
    if actuator_enabled:
        sx_values.append(float(scales.get("actual_delta_rad",
                                         scales.get("delta_rad", 0.5))))
    sx = np.asarray(sx_values, dtype=float)
    su = np.asarray([
        float(scales.get("qdelta_radps", steering_rate_limit)),
        float(scales.get("qv_mps2", model.max_target_speed_rate_reduction_mps2)),
    ], dtype=float)

    state_count = 7 if actuator_enabled else 6
    Z = opti.variable(state_count, n)
    W = opti.variable(2, n)
    X = ca.diag(ca.DM(sx)) @ Z
    U = ca.diag(ca.DM(su)) @ W
    ey, epsi, body_u, yaw_r, target_v, delta_command = [X[i, :] for i in range(6)]
    delta_actual = X[6, :] if actuator_enabled else delta_command
    qdelta, qv = U[0, :], U[1, :]

    initial_rows = [
        guess.ey / sx[0], guess.epsi / sx[1], guess.u / sx[2], guess.r / sx[3],
        guess.vt / sx[4], guess.delta / sx[5],
    ]
    if actuator_enabled:
        actual_seed = (np.asarray(guess.actual_delta, dtype=float)
                       if guess.actual_delta is not None else
                       guess.delta - 2.0 * steering_delay_s * guess.qdelta)
        actual_seed = np.clip(actual_seed, -steering_limit, steering_limit)
        initial_rows.append(actual_seed / sx[6])
    opti.set_initial(Z, np.vstack(initial_rows))
    opti.set_initial(W, np.vstack([guess.qdelta / su[0], guess.qv / su[1]]))

    limits = config.get("limits", {})
    track_cfg = config.get("track", {})
    numerics = config.get("numerics", {})
    reg = config.get("regularization", {})
    wall_margin = _wall_clearance(config, model)
    min_progress = float(numerics.get("min_progress_mps", 0.3))
    min_frenet_den = float(numerics.get("min_frenet_denominator", 0.25))
    max_heading_error = float(limits.get("max_heading_error_rad", 0.75))
    ay_fun = envelope.casadi_function()
    combined = config.get("combined_acceleration", {})
    combined_enabled = bool(combined.get("enabled", False))
    combined_p = float(combined.get("exponent", 2.0))
    use_lateral_velocity = bool(config.get("lateral_dynamics", {}).get(
        "use_odometry_lateral_velocity", False))

    long_extent = max(model.rear_axle_to_front_bumper_m, model.rear_overhang_m)

    speed_limit = _optimizer_speed_limit(model, envelope)
    command_speed_limit = min(model.max_command_speed_mps, speed_limit)
    body_v_nodes = []
    sdot_nodes = []
    for k in range(n):
        den = 1.0 - float(track.kappa[k]) * ey[k]
        body_v = (model.lateral_velocity_casadi(body_u[k], yaw_r[k])
                  if use_lateral_velocity else 0.0)
        sdot = (body_u[k] * ca.cos(epsi[k]) - body_v * ca.sin(epsi[k])) / den
        body_v_nodes.append(body_v)
        sdot_nodes.append(sdot)
    delayed_command_nodes = (
        _delayed_steering_command_casadi(
            track, delta_command, qdelta, sdot_nodes, steering_delay_s,
            (model.max_body_speed_mps + model.lateral_velocity_max_mps)
            / min_frenet_den)
        if actuator_enabled else None
    )

    f_nodes = []
    time_density = []
    accel_nodes = []
    yaw_response_nodes = []
    ay_nodes = []
    aycap_nodes = []

    for k in range(n):
        den = 1.0 - float(track.kappa[k]) * ey[k]
        body_v = body_v_nodes[k]
        sdot = sdot_nodes[k]
        raw_a = model.raw_longitudinal_accel(body_u[k], target_v[k], qv[k])
        r_ss_base = model.steady_yaw_rate_casadi(
            body_u[k], delta_actual[k], float(track.kappa[k]))
        residual_lateral_v = model.lateral_velocity_casadi(body_u[k], yaw_r[k])
        yaw_residual = model.yaw_residual_rate_casadi(
            body_u[k], yaw_r[k], delta_actual[k], qdelta[k], residual_lateral_v,
            float(track.kappa[k]), raw_a, qv[k], target_v[k])
        r_ss = r_ss_base + model.yaw_tau_s * yaw_residual
        r_dot = (r_ss - yaw_r[k]) / model.yaw_tau_s
        yaw_response_nodes.append(r_ss)
        f_components = [
            (body_u[k] * ca.sin(epsi[k]) + body_v * ca.cos(epsi[k])) / sdot,
            (yaw_r[k] - float(track.kappa[k]) * sdot) / sdot,
            raw_a / sdot,
            r_dot / sdot,
            qv[k] / sdot,
            qdelta[k] / sdot,
        ]
        if actuator_enabled:
            actual_rate = ca.fmin(ca.fmax(
                (delayed_command_nodes[k] - delta_actual[k]) / steering_delay_s,
                -steering_rate_limit), steering_rate_limit)
            f_components.append(actual_rate / sdot)
        f = ca.vertcat(*f_components)
        f_nodes.append(f)
        time_density.append(1.0 / sdot)
        accel_nodes.append(raw_a)
        ay = body_u[k] * yaw_r[k]
        aycap = ay_fun(body_u[k], delta_actual[k])
        ay_nodes.append(ay)
        aycap_nodes.append(aycap)

        # State/input limits.
        opti.subject_to(opti.bounded(model.min_speed_mps, body_u[k], speed_limit))
        opti.subject_to(opti.bounded(0.0, target_v[k], command_speed_limit))
        opti.subject_to(opti.bounded(-steering_limit, delta_command[k], steering_limit))
        if actuator_enabled:
            opti.subject_to(opti.bounded(-steering_limit, delta_actual[k], steering_limit))
        opti.subject_to(opti.bounded(-steering_rate_limit, qdelta[k], steering_rate_limit))
        opti.subject_to(opti.bounded(-model.max_target_speed_rate_reduction_mps2,
                                     qv[k], model.max_target_speed_rate_increase_mps2))
        opti.subject_to(opti.bounded(-max_heading_error, epsi[k], max_heading_error))
        opti.subject_to(den >= min_frenet_den)
        opti.subject_to(sdot >= min_progress)

        brake_cap = model.brake_limit(body_u[k])
        opti.subject_to(raw_a <= model.accel_limit_mps2)
        opti.subject_to(raw_a >= -brake_cap)
        opti.subject_to(smooth_abs(ay) <= aycap)

        # Match existing optimize_trajectory.py: at least optimizer_width/2
        # plus the repo's 0.15 m wall clearance.  Also protect the true
        # rectangular AutoDRIVE body when heading error makes it wider.
        physical_footprint = (
            0.5 * model.car_width_m * smooth_abs(ca.cos(epsi[k]))
            + long_extent * smooth_abs(ca.sin(epsi[k]))
        )
        footprint = ca.fmax(0.5 * model.planning_footprint_width_m, physical_footprint)
        left_limit = float(track.left[k]) - wall_margin - footprint
        right_limit = float(track.right[k]) - wall_margin - footprint
        opti.subject_to(ey[k] <= left_limit)
        opti.subject_to(ey[k] >= -right_limit)

        if combined_enabled:
            # Optional empirical coupling. Disabled by default until track data calibrates it.
            ax_pos = model.accel_limit_mps2
            ax_neg = brake_cap
            ax_norm = ca.if_else(raw_a >= 0.0, raw_a / ax_pos, -raw_a / ax_neg)
            ay_norm = smooth_abs(ay) / aycap
            opti.subject_to(ca.power(ax_norm, combined_p) + ca.power(ay_norm, combined_p) <= 1.0)

    # Trapezoidal direct collocation in centerline arc length; last interval wraps to node 0.
    lap_time = 0
    reg_cost = 0
    for k in range(n):
        kp = (k + 1) % n
        ds = float(track.ds[k])
        # The 15 ms yaw response is much faster than a 0.25 m spatial interval
        # (about 60 ms at 4 m/s).  Trapezoidal integration is stable here but
        # oscillatory when dt/tau > 2, which gives IPOPT a false alternating
        # yaw mode.  Keep trapezoidal integration for the slower states and
        # integrate yaw exactly for a linearly varying steady-response input.
        trapezoid_defect = (
            X[:, kp] - X[:, k] - 0.5 * ds * (f_nodes[k] + f_nodes[kp])
        ) / ca.DM(sx)
        state_indices = (0, 2, 4, 5, 6) if actuator_enabled else (0, 2, 4, 5)
        for state_index in state_indices:
            opti.subject_to(trapezoid_defect[state_index] == 0.0)

        interval_sdot = 0.5 * (sdot_nodes[k] + sdot_nodes[kp])
        interval_dt = ds / interval_sdot
        yaw_end, yaw_integral = _yaw_lag_interval_casadi(
            yaw_r[k], yaw_response_nodes[k], yaw_response_nodes[kp],
            interval_dt, model.yaw_tau_s)
        curvature_mid = 0.5 * (float(track.kappa[k]) + float(track.kappa[kp]))
        exact_epsi_defect = (
            epsi[kp] - epsi[k] - yaw_integral + curvature_mid * ds
        ) / sx[1]
        exact_yaw_defect = (yaw_r[kp] - yaw_end) / sx[3]
        # Continue from the proven trapezoidal transcription to the exact
        # first-order yaw-lag step. Intermediate fractions are homotopy
        # constraints only; production candidate acceptance always uses 1.0.
        opti.subject_to(
            (1.0 - yaw_lag_exact_fraction) * trapezoid_defect[1]
            + yaw_lag_exact_fraction * exact_epsi_defect == 0.0)
        opti.subject_to(
            (1.0 - yaw_lag_exact_fraction) * trapezoid_defect[3]
            + yaw_lag_exact_fraction * exact_yaw_defect == 0.0)
        lap_time += 0.5 * ds * (time_density[k] + time_density[kp])
        reg_cost += 0.5 * ds * (
            float(reg.get("steering_rate", 1e-5)) * (qdelta[k] ** 2 * time_density[k] + qdelta[kp] ** 2 * time_density[kp])
            + float(reg.get("target_speed_rate", 1e-6)) * (qv[k] ** 2 * time_density[k] + qv[kp] ** 2 * time_density[kp])
            + float(reg.get("heading_error", 1e-6)) * (epsi[k] ** 2 * time_density[k] + epsi[kp] ** 2 * time_density[kp])
        )

    objective = lap_time + reg_cost
    opti.minimize(objective)

    p_opts = {"expand": bool(solver.get("expand", True))}
    s_opts = {
        "max_iter": int(max_iter_override if max_iter_override is not None
                        else solver.get("refinement_max_iter", 450)),
        "tol": float(solver.get("tol", 1e-7)),
        "acceptable_tol": float(solver.get("acceptable_tol", 1e-5)),
        "acceptable_iter": int(solver.get("acceptable_iter", 15)),
        "acceptable_constr_viol_tol": float(
            solver.get("acceptable_constr_viol_tol", 1e-2)),
        "print_level": int(solver.get("print_level", 5)),
        "sb": "yes",
        "nlp_scaling_method": str(solver.get(
            "nlp_scaling_method", "gradient-based")),
        "mu_strategy": str(solver.get("mu_strategy", "adaptive")),
        "linear_solver": str(solver.get("linear_solver", "mumps")),
    }
    hessian_approximation = solver.get("hessian_approximation")
    if hessian_approximation is not None:
        s_opts["hessian_approximation"] = str(hessian_approximation)
    opti.solver("ipopt", p_opts, s_opts)

    start = time.perf_counter()
    # solve_limited returns the best IPOPT iterate even on iteration/time limits.
    # This lets the continuation driver save diagnostics instead of losing a
    # four-minute failed refinement with no files.
    sol = opti.solve_limited()
    elapsed = time.perf_counter() - start
    stats = opti.stats()
    values = lambda expr: np.asarray(sol.value(expr), dtype=float).reshape(-1)
    converged = bool(stats.get("success", False))

    result = Solution(
        track=track,
        ey=values(ey), epsi=values(epsi), u=values(body_u), r=values(yaw_r),
        vt=values(target_v), delta=values(delta_actual), qdelta=values(qdelta), qv=values(qv),
        lap_time_s=float(sol.value(lap_time)), objective=float(sol.value(objective)),
        solver_stats=stats, elapsed_s=elapsed, label=guess.label, converged=converged,
        steering_command=(values(delta_command) if actuator_enabled else None),
    )
    diagnostic_error = None
    if converged and yaw_lag_exact_fraction >= 1.0:
        diagnostic = diagnose_solution(result, model, envelope, config)
        defect = diagnostic["max_abs_scaled_collocation_defect"]
        bad_slacks = {
            name: value for name, value in diagnostic["minimum_slacks"].items()
            if value is not None and value < -1.0e-3
        }
        if defect > 1.0e-4 or bad_slacks:
            result.converged = False
            converged = False
            diagnostic_error = (
                f"independent post-solve gate failed: scaled_defect={defect:.3g}, "
                f"negative_slacks={bad_slacks}"
            )
    if not converged:
        summary = _solver_summary(stats)
        raise SolveTrackFailure(
            f"IPOPT did not converge: status={summary['return_status']} "
            f"iter={summary['iter_count']} inf_pr={summary['final_inf_pr']} "
            f"inf_du={summary['final_inf_du']}"
            + (f"; {diagnostic_error}" if diagnostic_error else ""),
            result,
        )
    return result


def run_continuation(base_track_path: str | Path, repo_root: str | Path,
                     model: VehicleModel, envelope: LateralEnvelope,
                     config: dict[str, Any], warm_raceline: str | Path | None = None,
                     checkpoint_dir: str | Path | None = None,
                     warm_solution_nodes: str | Path | None = None) -> tuple[Solution, dict[str, Any]]:
    """Solve a robust mesh continuation without burning 3000-iteration starts.

    The real V1.2 run established two strong empirical facts on this track:
      * coarse wave:pi/2 reached the same optimum as the successful right start
        in ~49 iterations (right needed ~1041), and
      * the production raceline solved 0.18/0.14/0.11 in 91/85/57 iterations,
        while repaired/raw continuation guesses repeatedly hit 3000.

    V1.3 therefore orders starts by measured success, uses separate bounded
    iteration budgets, and stops as soon as a refinement mesh converges.
    """
    solver_cfg = config.get("solver", {})
    schedule = [float(x) for x in solver_cfg.get(
        "mesh_schedule_m", [0.25, 0.18, 0.14, 0.11, 0.08])]
    if not schedule:
        raise ValueError("solver.mesh_schedule_m must not be empty")

    coarse_modes = list(solver_cfg.get(
        "coarse_multistart", ["wave:1.57079632679", "right", "warm", "center"]))
    refinement_modes = list(solver_cfg.get(
        "refinement_starts", ["warm", "previous", "center"]))
    stop_coarse = bool(solver_cfg.get("stop_after_first_coarse_success", True))
    stop_refine = bool(solver_cfg.get("stop_after_first_refinement_success", True))
    allow_fallback = bool(solver_cfg.get("allow_best_coarse_fallback", True))
    coarse_max_iter = int(solver_cfg.get("coarse_max_iter", 1400))
    refinement_max_iter = int(solver_cfg.get("refinement_max_iter", 450))

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    continuation: dict[str, Any] = {
        "requested_mesh_schedule_m": schedule,
        "coarse_max_iter": coarse_max_iter,
        "refinement_max_iter": refinement_max_iter,
        "levels": [],
        "status": "running",
    }
    previous: Solution | None = None

    for level, spacing in enumerate(schedule):
        track = load_centerline_csv(base_track_path, spacing)
        mesh_name = f"mesh_{spacing:.3f}".replace(".", "p")
        mesh_dir = checkpoint_root / mesh_name if checkpoint_root is not None else None
        if mesh_dir is not None:
            mesh_dir.mkdir(parents=True, exist_ok=True)

        candidates: list[Guess] = []
        modes = coarse_modes if previous is None else refinement_modes
        for mode in modes:
            try:
                if mode == "warm":
                    if warm_raceline and Path(warm_raceline).exists():
                        candidates.append(_project_warm_raceline_to_track(
                            track, warm_raceline, model, envelope, config))
                elif mode == "solution_nodes":
                    if warm_solution_nodes and Path(warm_solution_nodes).exists():
                        candidates.append(_load_solution_nodes_guess(
                            track, warm_solution_nodes))
                elif mode == "previous":
                    if previous is not None:
                        candidates.append(_resample_solution_guess(previous, track))
                elif mode in {"center", "left", "right"} or mode.startswith("wave"):
                    candidates.append(build_guess(track, model, envelope, config, mode))
                else:
                    raise ValueError(f"Unknown continuation start mode {mode}")
            except Exception as exc:
                print(f"SKIP start {mode}: {type(exc).__name__}: {exc}")

        if not candidates:
            candidates = [build_guess(track, model, envelope, config, "center")]

        # De-duplicate labels while preserving the deliberately chosen order.
        unique: list[Guess] = []
        seen: set[str] = set()
        for guess in candidates:
            if guess.label not in seen:
                unique.append(guess)
                seen.add(guess.label)
        candidates = unique

        attempts: list[dict[str, Any]] = []
        solved: list[Solution] = []
        errors: list[str] = []
        max_iter = coarse_max_iter if previous is None else refinement_max_iter

        for attempt_index, guess in enumerate(candidates):
            print(f"\n=== mesh {spacing:.3f} m, start {guess.label}, max_iter={max_iter} ===")
            try:
                sol = solve_track(
                    track, model, envelope, config, guess,
                    max_iter_override=max_iter)
                diag = diagnose_solution(sol, model, envelope, config)
                attempts.append({"label": guess.label, "status": "solved", **diag})
                print(
                    f"SOLVED {guess.label}: lap={sol.lap_time_s:.6f}s "
                    f"iter={sol.solver_stats.get('iter_count')} wall={sol.elapsed_s:.2f}s")
                solved.append(sol)
                if (previous is None and stop_coarse) or (previous is not None and stop_refine):
                    break
            except SolveTrackFailure as exc:
                candidate = exc.candidate
                diag = diagnose_solution(candidate, model, envelope, config)
                attempts.append({"label": guess.label, "status": "failed_iterate", **diag})
                errors.append(f"{guess.label}: {exc}")
                print(f"FAILED {guess.label}: {exc}")
                if mesh_dir is not None:
                    safe_label = guess.label.replace(':', '_').replace('/', '_')
                    save_attempt_artifacts(
                        candidate, mesh_dir, model, envelope, config,
                        prefix=f"failed_{attempt_index:02d}_{safe_label}")
            except Exception as exc:
                errors.append(f"{guess.label}: {type(exc).__name__}: {exc}")
                attempts.append({
                    "label": guess.label,
                    "status": "exception",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                })
                print(f"FAILED {guess.label}: {type(exc).__name__}: {exc}")

        level_report = {
            "mesh_spacing_m": spacing,
            "node_count": track.count,
            "max_iter_per_start": max_iter,
            "attempts": attempts,
        }
        continuation["levels"].append(level_report)

        if not solved:
            level_report["status"] = "failed"
            level_report["errors"] = errors[:16]
            continuation["failed_mesh_spacing_m"] = spacing
            if previous is not None and allow_fallback:
                continuation["status"] = "partial_success_best_coarse_fallback"
                continuation["completed_mesh_spacing_m"] = float(
                    previous.track.length / previous.track.count)
                continuation["message"] = (
                    f"Refinement failed at {spacing:.3f} m; returning the last "
                    "converged mesh instead of discarding it.")
                if checkpoint_root is not None:
                    (checkpoint_root / "continuation_report.json").write_text(
                        json.dumps(continuation, indent=2, default=str), encoding="utf-8")
                return previous, continuation
            continuation["status"] = "failed_no_converged_mesh"
            if checkpoint_root is not None:
                (checkpoint_root / "continuation_report.json").write_text(
                    json.dumps(continuation, indent=2, default=str), encoding="utf-8")
            raise RuntimeError(
                f"All starts failed at mesh {spacing}. " + " | ".join(errors[:8]))

        previous = min(solved, key=lambda candidate: candidate.lap_time_s)
        previous.label = f"mesh{spacing:.3f}_{previous.label}"
        level_report["status"] = "solved"
        level_report["best_lap_time_s"] = previous.lap_time_s
        level_report["best_label"] = previous.label
        print(f"BEST mesh {spacing:.3f}: {previous.lap_time_s:.6f}s ({previous.label})")

        if mesh_dir is not None:
            checkpoint_report = export_solution(previous, model, envelope, config, mesh_dir)
            checkpoint_report["checkpoint_mesh_spacing_m"] = spacing
            checkpoint_report["continuation_complete"] = (level == len(schedule) - 1)
            (mesh_dir / "report.json").write_text(
                json.dumps(checkpoint_report, indent=2, default=str), encoding="utf-8")
            (checkpoint_root / "continuation_report.json").write_text(
                json.dumps(continuation, indent=2, default=str), encoding="utf-8")

    assert previous is not None
    continuation["status"] = "complete"
    continuation["completed_mesh_spacing_m"] = schedule[-1]
    if checkpoint_root is not None:
        (checkpoint_root / "continuation_report.json").write_text(
            json.dumps(continuation, indent=2, default=str), encoding="utf-8")
    return previous, continuation

def _interp_periodic(sq: np.ndarray, s: np.ndarray, v: np.ndarray, length: float):
    se = np.r_[s - length, s, s + length]
    ve = np.r_[v, v, v]
    return np.interp(np.mod(sq, length), se, ve)


def export_solution(solution: Solution, model: VehicleModel, envelope: LateralEnvelope,
                    config: dict[str, Any], output_dir: str | Path,
                    output_spacing_m: float | None = None) -> dict[str, Any]:
    """Export the solved path, optionally at a denser controller-facing spacing.

    V1.2 linearly interpolated e_y to 0.02 m and then fitted another periodic
    cubic spline through the dense points.  That created artificial 20--100 1/m
    curvature spikes which were absent from the OCP.  V1.3 exports the solved
    mesh itself.  The controller-facing export may now be linearly densified
    in the solved path arclength.  This preserves the exact piecewise-linear
    OCP geometry and fields; it does not fit a new spline or invent curvature.
    A dense export is useful for projection and wall checking at 40 Hz because
    the controller then sees several trajectory points during one fast control
    step instead of a single coarse optimizer node.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = solution.track

    ey = np.asarray(solution.ey, dtype=float)
    epsi = np.asarray(solution.epsi, dtype=float)
    u = np.asarray(solution.u, dtype=float)
    r = np.asarray(solution.r, dtype=float)
    vt = np.asarray(solution.vt, dtype=float)
    delta = np.asarray(solution.delta, dtype=float)
    qdelta = np.asarray(solution.qdelta, dtype=float)
    qv = np.asarray(solution.qv, dtype=float)

    # Actual solved path coordinates at optimizer nodes.
    x = tr.x - ey * np.sin(tr.psi)
    y = tr.y + ey * np.cos(tr.psi)
    xy = np.column_stack([x, y])
    closed = np.vstack([xy, xy[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s_actual_closed = np.r_[0.0, np.cumsum(seg)]
    s_actual = s_actual_closed[:-1]
    L_actual = float(s_actual_closed[-1])

    # Keep the optimizer's body heading/yaw-derived curvature for independent
    # diagnostics, but do not label those states as the geometry of the path.
    psi_opt = tr.psi + epsi
    kappa_dynamic = r / np.maximum(u, 1.0e-6)
    accel = np.asarray(model.raw_longitudinal_accel(u, vt, qv), dtype=float)

    # Independent geometric validation at the same optimizer nodes. This is a
    # validator only; it is not used to mutate the optimized reference.
    sxg = CubicSpline(s_actual_closed, np.r_[x, x[0]], bc_type="periodic")
    syg = CubicSpline(s_actual_closed, np.r_[y, y[0]], bc_type="periodic")
    dx = sxg(s_actual, 1); dy = syg(s_actual, 1)
    ddx = sxg(s_actual, 2); ddy = syg(s_actual, 2)
    psi_geom = np.unwrap(np.arctan2(dy, dx))
    kappa_geom = (dx * ddy - dy * ddx) / np.maximum((dx*dx + dy*dy)**1.5, 1e-12)
    psi_opt_unwrapped = np.unwrap(psi_opt)
    heading_err = np.arctan2(np.sin(psi_geom - psi_opt_unwrapped),
                             np.cos(psi_geom - psi_opt_unwrapped))
    kappa_err = kappa_geom - kappa_dynamic

    # Approximate raw wall distances based on the reference-normal offset. The
    # top-level runner replaces these with map ray-cast distances and hard-
    # validates the same 0.30 m center-to-wall requirement used by the existing
    # optimize_trajectory.py pipeline.
    left_approx = np.maximum(tr.left - ey, 0.0)
    right_approx = np.maximum(tr.right + ey, 0.0)

    # Keep the OCP node solution for the report and solution_nodes.csv, but
    # optionally export a uniform controller-facing view of that same
    # piecewise-linear path.  Periodic linear interpolation is intentional:
    # it cannot change the solved geometry or create the curvature spikes that
    # caused the old dense cubic export to be rejected.
    export_s = s_actual
    export_parameter_s = s_actual
    export_x = x
    export_y = y
    export_heading = np.arctan2(np.sin(psi_geom), np.cos(psi_geom))
    export_curvature = kappa_geom
    export_speed = u
    export_accel = accel
    export_left = left_approx
    export_right = right_approx
    if output_spacing_m is not None:
        spacing = float(output_spacing_m)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("output_spacing_m must be finite and positive")
        if spacing < float(np.mean(seg)):
            export_count = max(3, int(np.ceil(L_actual / spacing)))
            export_parameter_s = (
                np.arange(export_count, dtype=float) * L_actual / export_count)
            source_s = np.r_[s_actual, L_actual]

            def periodic_linear(values: np.ndarray, closure_value: float | None = None) -> np.ndarray:
                # A wrapped angular field cannot close by interpolating the
                # raw first value.  ``psi_opt_unwrapped`` is continuous along
                # the solved lap, so choose the 2*pi-equivalent of its first
                # value that is nearest the last value before interpolating
                # the closing segment.  Without this, a dense export can
                # contain a false multi-radian heading jump at the lap seam.
                if closure_value is None:
                    closure_value = float(values[0])
                return np.interp(
                    export_parameter_s, source_s,
                    np.r_[values, closure_value])

            # Geometry fields must describe the same periodic path as x/y.
            # Previously the CSV combined linearly densified x/y with heading
            # states and r/u as "path curvature". The production MPC consumes
            # curvature directly in its Frenet propagation, so that mismatch
            # can manufacture horizon corridor failures even when the path
            # points themselves are unchanged.
            export_x = np.asarray(sxg(export_parameter_s), dtype=float)
            export_y = np.asarray(syg(export_parameter_s), dtype=float)
            dx_export = sxg(export_parameter_s, 1)
            dy_export = syg(export_parameter_s, 1)
            ddx_export = sxg(export_parameter_s, 2)
            ddy_export = syg(export_parameter_s, 2)
            export_heading = np.arctan2(dy_export, dx_export)
            export_curvature = (
                dx_export * ddy_export - dy_export * ddx_export
            ) / np.maximum(
                (dx_export * dx_export + dy_export * dy_export) ** 1.5,
                1.0e-12)

            export_speed = periodic_linear(u)
            export_accel = periodic_linear(accel)
            export_left = periodic_linear(left_approx)
            export_right = periodic_linear(right_approx)

    # CSV station follows the emitted geometry's actual sampled arc length,
    # including any shape change from periodic-spline densification.
    export_xy_closed = np.vstack([
        np.column_stack([export_x, export_y]),
        [export_x[0], export_y[0]],
    ])
    export_segment_lengths = np.linalg.norm(
        np.diff(export_xy_closed, axis=0), axis=1)
    export_path_length = float(np.sum(export_segment_lengths))
    export_s = np.r_[0.0, np.cumsum(export_segment_lengths)[:-1]]

    traj_path = out / "autodrive_mintime_raceline.csv"
    with traj_path.open("w", encoding="utf-8") as f:
        f.write("# s_m,x_m,y_m,psi_rad,kappa_radpm,velocity_mps,acceleration_mps2,d_left_m,d_right_m\n")
        for i in range(len(export_s)):
            f.write(
                f"{export_s[i]:.7f},{export_x[i]:.7f},{export_y[i]:.7f},"
                f"{np.arctan2(np.sin(export_heading[i]), np.cos(export_heading[i])):.8f},"
                f"{export_curvature[i]:.8f},{export_speed[i]:.7f},{export_accel[i]:.7f},"
                f"{export_left[i]:.6f},{export_right[i]:.6f}\n")

    node_path = out / "solution_nodes.csv"
    node_fields = ["s_ref_m", "x_ref_m", "y_ref_m", "kappa_ref", "ey_m",
                   "epsi_rad", "u_mps", "r_radps", "target_mps", "delta_rad"]
    if solution.steering_command is not None:
        node_fields.append("steering_command_rad")
    node_fields.extend(["qdelta_radps", "qv_mps2", "left_m", "right_m"])
    with node_path.open("w", encoding="utf-8") as f:
        f.write(",".join(node_fields) + "\n")
        for i in range(tr.count):
            values = [tr.s[i], tr.x[i], tr.y[i], tr.kappa[i], ey[i], epsi[i],
                      u[i], r[i], vt[i], delta[i]]
            if solution.steering_command is not None:
                values.append(solution.steering_command[i])
            values.extend([qdelta[i], qv[i], tr.left[i], tr.right[i]])
            f.write(",".join(f"{value:.9f}" for value in values) + "\n")

    time_recomputed = float(np.sum(seg / np.maximum(0.5 * (u + np.roll(u, -1)), 1e-3)))
    ay = u * r
    aycap = envelope.numpy(u, delta)
    footprint = _vehicle_lateral_extent(epsi, model)
    wall_margin = _wall_clearance(config, model)
    left_slack = tr.left - ey - footprint - wall_margin
    right_slack = tr.right + ey - footprint - wall_margin

    q = lambda a, p: float(np.quantile(np.asarray(a, dtype=float), p))
    report = {
        "status": "solved",
        "optimizer": "AutoDRIVE-specific Frenet spatial minimum-time OCP",
        "solution_label": solution.label,
        "predicted_lap_time_s": solution.lap_time_s,
        "export_recomputed_lap_time_s": time_recomputed,
        "objective": solution.objective,
        "optimizer_wall_time_s": solution.elapsed_s,
        "solver_status": solution.solver_stats.get("return_status"),
        "solver_iterations": solution.solver_stats.get("iter_count"),
        "track": {
            **tr.to_dict(),
            "optimized_path_length_m": L_actual,
            "exported_path_length_m": export_path_length,
            "output_points": int(len(export_s)),
            "export_spacing_mean_m": float(export_path_length / len(export_s)),
            "export_mode": (
                "optimizer_nodes"
                if len(export_s) == tr.count else
                "periodic_cubic_geometry_controller_densification"),
        },
        "vehicle_model": model.to_dict(),
        "vehicle_provenance": model.provenance(),
        "optimizer_constraints": {
            "max_steering_rad": _optimizer_steering_limit(model, config),
            "physical_model_max_steering_rad": model.max_steering_rad,
            "max_steering_rate_radps": _optimizer_steering_rate_limit(
                model, config),
            "physical_model_steering_rate_radps": model.max_steering_rate_radps,
        },
        "lateral_envelope": envelope.to_dict(),
        "wall_safety": {
            "actual_car_width_m": model.car_width_m,
            "planning_footprint_width_m": model.planning_footprint_width_m,
            "repo_wall_clearance_m": model.required_wall_clearance_m,
            "extra_wall_clearance_m": float(config.get("track", {}).get("extra_wall_clearance_m", 0.0)),
            "optimizer_geometry_buffer_m": float(
                config.get("track", {}).get("optimizer_geometry_buffer_m", 0.0)),
            "aligned_required_center_to_wall_m": (
                0.5 * model.planning_footprint_width_m + wall_margin),
            "heading_aware_physical_footprint": True,
            "minimum_internal_left_slack_m": float(np.min(left_slack)),
            "minimum_internal_right_slack_m": float(np.min(right_slack)),
            "wall_distance_columns": "reference-normal approximation; top-level runner raycasts map and overwrites final file",
        },
        "geometry_consistency": {
            "trajectory_heading_source": "periodic cubic spline tangent of exported x/y",
            "trajectory_curvature_source": "periodic cubic spline curvature of exported x/y",
            "heading_abs_error_p95_rad": q(np.abs(heading_err), 0.95),
            "heading_abs_error_max_rad": float(np.max(np.abs(heading_err))),
            "dynamic_curvature_abs_max_m_inv": float(np.max(np.abs(kappa_dynamic))),
            "geometric_curvature_abs_p95_m_inv": q(np.abs(kappa_geom), 0.95),
            "geometric_curvature_abs_max_m_inv": float(np.max(np.abs(kappa_geom))),
            "curvature_difference_abs_p95_m_inv": q(np.abs(kappa_err), 0.95),
            "curvature_difference_abs_max_m_inv": float(np.max(np.abs(kappa_err))),
        },
        "limits_and_usage": {
            "speed_min_mps": float(np.min(u)),
            "speed_max_mps": float(np.max(u)),
            "target_speed_min_mps": float(np.min(vt)),
            "target_speed_max_mps": float(np.max(vt)),
            "steering_abs_max_rad": float(np.max(np.abs(delta))),
            "steering_command_abs_max_rad": float(np.max(np.abs(
                solution.steering_command if solution.steering_command is not None
                else delta))),
            "steering_rate_abs_max_radps": float(np.max(np.abs(qdelta))),
            "target_rate_min_mps2": float(np.min(qv)),
            "target_rate_max_mps2": float(np.max(qv)),
            "accel_min_mps2": float(np.min(accel)),
            "accel_max_mps2": float(np.max(accel)),
            "lateral_accel_abs_max_mps2": float(np.max(np.abs(ay))),
            "lateral_envelope_min_slack_mps2": float(np.min(aycap - np.abs(ay))),
            "left_safety_slack_min_m": float(np.min(left_slack)),
            "right_safety_slack_min_m": float(np.min(right_slack)),
            "heading_error_abs_max_rad": float(np.max(np.abs(epsi))),
            "yaw_rate_abs_max_radps": float(np.max(np.abs(r))),
        },
        "files": {
            "trajectory": str(traj_path),
            "nodes": str(node_path),
        },
    }
    return report
