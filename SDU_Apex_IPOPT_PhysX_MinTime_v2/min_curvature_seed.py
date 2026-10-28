#!/usr/bin/env python3
"""Generate a map-checked curvature/length geometry seed for IPOPT.

The geometric objective is discrete curvature energy plus optional path
length, reference-path, and straight-section terms. This only generates an
initialization; IPOPT still optimizes free path, speed, steering, throttle,
and vehicle states.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import casadi as ca
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline

from map_track import MapGrid, _FOOTPRINT_LOCAL, _FOOTPRINT_UNSAMPLED_CORRECTION


def _read_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 9 or len(rows) < 12:
        raise ValueError("Reference CSV must have 9+ columns and at least 12 rows")
    if not np.isfinite(rows).all():
        raise ValueError("Reference CSV contains non-finite values")
    xy = rows[:, 1:3]
    speed = rows[:, 5]
    # IPOPT's Track loader treats these as raw left/right map distances.
    widths = rows[:, 7:9]
    if np.any(widths <= 0.0) or np.any(speed <= 0.0):
        raise ValueError("Reference speed and corridor widths must be positive")
    return xy, speed, widths


def _resample_reference(xy: np.ndarray, speed: np.ndarray,
                        widths: np.ndarray, spacing: float):
    if np.linalg.norm(xy[0] - xy[-1]) < 1.0e-5:
        xy, speed, widths = xy[:-1], speed[:-1], widths[:-1]
    keep = np.ones(len(xy), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1.0e-7
    xy, speed, widths = xy[keep], speed[keep], widths[keep]
    closed = np.vstack((xy, xy[0]))
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(segment <= 1.0e-9):
        raise ValueError("Reference has a zero-length segment")
    source_s = np.r_[0.0, np.cumsum(segment)]
    length = float(source_s[-1])
    count = max(40, int(math.ceil(length / spacing)))
    stations = np.linspace(0.0, length, count, endpoint=False)
    sx = CubicSpline(source_s, closed[:, 0], bc_type="periodic")
    sy = CubicSpline(source_s, closed[:, 1], bc_type="periodic")
    point = np.column_stack((sx(stations), sy(stations)))
    dx, dy = sx(stations, 1), sy(stations, 1)
    ddx, ddy = sx(stations, 2), sy(stations, 2)
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = (dx * ddy - dy * ddx) / np.maximum(
        (dx * dx + dy * dy) ** 1.5, 1.0e-12)

    # Match the optimizer track loader's periodic linear interpolation of
    # width channels, and use the same normalized lap stations for speed.
    source_open_s = source_s[:-1]
    length_ratio = stations / length
    source_query = length_ratio * length

    def periodic_interp(values):
        return np.interp(
            source_query,
            np.r_[source_open_s - length, source_open_s, source_open_s + length],
            np.tile(values, 3))

    sampled_speed = periodic_interp(speed)
    sampled_widths = np.column_stack(
        (periodic_interp(widths[:, 0]), periodic_interp(widths[:, 1])))
    ds = length / count
    return stations, point, heading, curvature, sampled_speed, sampled_widths, ds, length


def _footprint_clearance_grid(grid: MapGrid, xy: np.ndarray, heading: np.ndarray,
                              offsets: np.ndarray, margin: float,
                              chunk_size: int = 24) -> np.ndarray:
    """Clearance for all normal offsets, using the mapped vehicle perimeter."""
    n = len(xy)
    result = np.empty((n, len(offsets)), dtype=np.float64)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    local = _FOOTPRINT_LOCAL
    local_x = cos_h[:, None] * local[None, :, 0] - sin_h[:, None] * local[None, :, 1]
    local_y = sin_h[:, None] * local[None, :, 0] + cos_h[:, None] * local[None, :, 1]
    normal = np.column_stack((-sin_h, cos_h))
    for begin in range(0, n, chunk_size):
        end = min(n, begin + chunk_size)
        base = xy[begin:end, None, None, :] + (
            offsets[None, :, None, None] * normal[begin:end, None, None, :])
        body = np.stack((local_x[begin:end], local_y[begin:end]), axis=-1)
        probes = base + body[:, None, :, :]
        values = np.asarray(grid.distance(probes.reshape(-1, 2))).reshape(
            end - begin, len(offsets), len(local))
        result[begin:end] = (np.min(values, axis=2)
                             - _FOOTPRINT_UNSAMPLED_CORRECTION - margin)
    return result


def _safe_offset_bounds(clearance: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zero = int(np.argmin(np.abs(offsets)))
    if np.max(np.abs(offsets[zero])) > 1.0e-10:
        raise ValueError("Offset grid must include zero")
    safe = clearance >= 0.0
    if not np.all(safe[:, zero]):
        index = int(np.flatnonzero(~safe[:, zero])[0])
        raise ValueError(f"Reference footprint is unsafe at station {index}")
    low = np.empty(len(clearance))
    high = np.empty(len(clearance))
    for i, row in enumerate(safe):
        lo = zero
        while lo > 0 and row[lo - 1]:
            lo -= 1
        hi = zero
        while hi + 1 < len(row) and row[hi + 1]:
            hi += 1
        # Stay one scan cell inside the first invalid footprint placement.
        low[i] = offsets[lo] + (offsets[1] - offsets[0])
        high[i] = offsets[hi] - (offsets[1] - offsets[0])
        if low[i] >= 0.0 or high[i] <= 0.0 or low[i] >= high[i]:
            raise ValueError(f"No two-sided safe offset corridor at station {i}")
    return low, high


def _minimum_curvature_offset(xy: np.ndarray, heading: np.ndarray,
                              lower: np.ndarray, upper: np.ndarray,
                              harmonics: int,
                              length_penalty_per_m2: float = 0.0,
                              reference_xy: np.ndarray | None = None,
                              reference_attraction_per_m4: float = 0.0,
                              entry_weights: np.ndarray | None = None,
                              entry_reference_xy: np.ndarray | None = None,
                              entry_reference_attraction_per_m4: float = 0.0,
                              entry_curvature_multiplier: float = 0.0,
                              exit_weights: np.ndarray | None = None,
                              exit_reference_attraction_per_m4: float = 0.0,
                              straight_weights: np.ndarray | None = None,
                              straight_target_x: float | None = None,
                              straight_attraction_per_m4: float = 0.0,
                              straight_curvature_multiplier: float = 0.0,
                              outer_wall_weights: np.ndarray | None = None,
                              outer_wall_target_offset: np.ndarray | None = None,
                              outer_wall_attraction_per_m4: float = 0.0) -> dict:
    """Minimize curvature, path length, and optional reference deviation.

    The length weight has units 1/m^2. The optional reference weight has units
    1/m^4 and penalizes integrated squared XY deviation from an aligned,
    periodic reference path. A periodic Fourier basis keeps the path smooth
    while allowing map-checked offsets.
    """
    n = len(xy)
    if n < 12 or heading.shape != (n,) or lower.shape != (n,) or upper.shape != (n,):
        raise ValueError("Minimum-curvature mesh or corridor is invalid")
    if not 1 <= harmonics <= n // 4:
        raise ValueError("Fourier harmonic count is outside the stable mesh range")
    if not math.isfinite(length_penalty_per_m2) or length_penalty_per_m2 < 0.0:
        raise ValueError("Path-length penalty must be finite and nonnegative")
    if (not math.isfinite(reference_attraction_per_m4)
            or reference_attraction_per_m4 < 0.0):
        raise ValueError("Reference-attraction weight must be finite and nonnegative")
    if reference_xy is not None and reference_xy.shape != (n, 2):
        raise ValueError("Aligned reference path must have shape (mesh_nodes, 2)")
    if entry_reference_xy is not None and entry_reference_xy.shape != (n, 2):
        raise ValueError("Aligned entry reference must have shape (mesh_nodes, 2)")
    for name, weights in (("entry", entry_weights), ("exit", exit_weights)):
        if weights is not None and weights.shape != (n,):
            raise ValueError(f"{name.title()} region weights must match the path mesh")
    for name, value in (("entry reference", entry_reference_attraction_per_m4),
                        ("entry curvature", entry_curvature_multiplier),
                        ("exit reference", exit_reference_attraction_per_m4)):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name.title()} weight must be finite and nonnegative")
    if ((entry_reference_attraction_per_m4 > 0.0
         and reference_xy is None and entry_reference_xy is None)
            or (exit_reference_attraction_per_m4 > 0.0 and reference_xy is None)):
        raise ValueError("Local reference attraction requires an aligned reference path")
    if not math.isfinite(straight_attraction_per_m4) or straight_attraction_per_m4 < 0.0:
        raise ValueError("Straightness weight must be finite and nonnegative")
    if (not math.isfinite(straight_curvature_multiplier)
            or straight_curvature_multiplier < 0.0):
        raise ValueError("Straight-section curvature multiplier must be finite and nonnegative")
    if straight_weights is not None and straight_weights.shape != (n,):
        raise ValueError("Straightness weights must match the path mesh")
    if outer_wall_weights is not None and outer_wall_weights.shape != (n,):
        raise ValueError("Outer-wall weights must match the path mesh")
    if outer_wall_target_offset is not None and outer_wall_target_offset.shape != (n,):
        raise ValueError("Outer-wall target offsets must match the path mesh")
    if (not math.isfinite(outer_wall_attraction_per_m4)
            or outer_wall_attraction_per_m4 < 0.0):
        raise ValueError("Outer-wall attraction must be finite and nonnegative")
    if ((outer_wall_attraction_per_m4 > 0.0)
            != (outer_wall_weights is not None
                and outer_wall_target_offset is not None)):
        raise ValueError("Outer-wall attraction needs a region and target offsets")
    if (straight_attraction_per_m4 > 0.0 or straight_curvature_multiplier > 0.0) and (
            straight_weights is None or straight_target_x is None):
        raise ValueError("Straightness penalty requires a selected segment and target x")

    station_phase = np.arange(n, dtype=np.float64) / n
    columns = [np.ones(n, dtype=np.float64)]
    for harmonic in range(1, harmonics + 1):
        angle = 2.0 * math.pi * harmonic * station_phase
        columns.extend((np.cos(angle), np.sin(angle)))
    basis = np.column_stack(columns)
    coefficients = ca.MX.sym("periodic_offset_coefficients", basis.shape[1])
    offset = ca.DM(basis) @ coefficients
    normal_x = -np.sin(heading)
    normal_y = np.cos(heading)
    path_x = ca.DM(xy[:, 0]) + ca.DM(normal_x) * offset
    path_y = ca.DM(xy[:, 1]) + ca.DM(normal_y) * offset

    previous_x, previous_y = ca.vertcat(path_x[-1], path_x[:-1]), ca.vertcat(path_y[-1], path_y[:-1])
    next_x, next_y = ca.vertcat(path_x[1:], path_x[0]), ca.vertcat(path_y[1:], path_y[0])
    a_x, a_y = path_x - previous_x, path_y - previous_y
    b_x, b_y = next_x - path_x, next_y - path_y
    c_x, c_y = next_x - previous_x, next_y - previous_y
    length_a = ca.sqrt(a_x * a_x + a_y * a_y + 1.0e-14)
    length_b = ca.sqrt(b_x * b_x + b_y * b_y + 1.0e-14)
    length_c = ca.sqrt(c_x * c_x + c_y * c_y + 1.0e-14)
    curvature = 2.0 * (a_x * b_y - a_y * b_x) / (length_a * length_b * length_c)
    curvature_energy = ca.sum1(curvature * curvature * 0.5 * (length_a + length_b))
    entry_curvature_energy = ca.MX(0.0)
    if entry_weights is not None:
        entry_curvature_energy = ca.sum1(
            ca.DM(entry_weights) * curvature * curvature * 0.5 * (length_a + length_b))
    straight_curvature_energy = ca.MX(0.0)
    if straight_weights is not None:
        straight_curvature_energy = ca.sum1(
            ca.DM(straight_weights) * curvature * curvature
            * 0.5 * (length_a + length_b))
    path_length = ca.sum1(length_a)
    reference_energy = ca.MX(0.0)
    entry_reference_energy = ca.MX(0.0)
    exit_reference_energy = ca.MX(0.0)
    if reference_xy is not None:
        reference_x = ca.DM(reference_xy[:, 0])
        reference_y = ca.DM(reference_xy[:, 1])
        error_squared = ((path_x - reference_x) ** 2
                         + (path_y - reference_y) ** 2)
        reference_energy = ca.sum1(error_squared * 0.5 * (length_a + length_b))
        if entry_weights is not None:
            entry_target = (entry_reference_xy if entry_reference_xy is not None
                            else reference_xy)
            entry_error_squared = ((path_x - ca.DM(entry_target[:, 0])) ** 2
                                   + (path_y - ca.DM(entry_target[:, 1])) ** 2)
            entry_reference_energy = ca.sum1(
                ca.DM(entry_weights) * entry_error_squared
                * 0.5 * (length_a + length_b))
        if exit_weights is not None:
            exit_reference_energy = ca.sum1(
                ca.DM(exit_weights) * error_squared * 0.5 * (length_a + length_b))
    straight_energy = ca.MX(0.0)
    if straight_weights is not None and straight_target_x is not None:
        straight_error_squared = (path_x - straight_target_x) ** 2
        straight_energy = ca.sum1(
            ca.DM(straight_weights) * straight_error_squared
            * 0.5 * (length_a + length_b))
    outer_wall_energy = ca.MX(0.0)
    if outer_wall_weights is not None and outer_wall_target_offset is not None:
        wall_error_squared = (offset - ca.DM(outer_wall_target_offset)) ** 2
        outer_wall_energy = ca.sum1(
            ca.DM(outer_wall_weights) * wall_error_squared
            * 0.5 * (length_a + length_b))
    objective = (curvature_energy
                 + entry_curvature_multiplier * entry_curvature_energy
                 + straight_curvature_multiplier * straight_curvature_energy
                 + length_penalty_per_m2 * path_length
                 + reference_attraction_per_m4 * reference_energy
                 + entry_reference_attraction_per_m4 * entry_reference_energy
                 + exit_reference_attraction_per_m4 * exit_reference_energy
                 + straight_attraction_per_m4 * straight_energy
                 + outer_wall_attraction_per_m4 * outer_wall_energy)

    solver = ca.nlpsol(
        "periodic_min_curvature", "ipopt",
        {"x": coefficients, "f": objective, "g": offset},
        {"expand": True, "print_time": False,
         "ipopt": {"print_level": 0, "max_iter": 1200,
                   "tol": 1.0e-8, "acceptable_tol": 1.0e-6}})
    target_reference = reference_xy
    if entry_weights is not None and entry_reference_xy is not None:
        blend = np.clip(entry_weights, 0.0, 1.0)[:, None]
        target_reference = (reference_xy * (1.0 - blend)
                            + entry_reference_xy * blend)
    if target_reference is None:
        initial_coefficients = np.zeros(basis.shape[1])
    else:
        target_offset = np.sum(
            (target_reference - xy) * np.column_stack((normal_x, normal_y)), axis=1)
        initial_coefficients = np.linalg.lstsq(basis, target_offset, rcond=None)[0]
    result = solver(x0=initial_coefficients, lbx=-ca.inf,
                    ubx=ca.inf, lbg=lower, ubg=upper)
    stats = solver.stats()
    solution = np.asarray(result["x"], dtype=np.float64).reshape(-1)
    if (not stats.get("success") or not np.isfinite(solution).all()):
        raise RuntimeError("Periodic minimum-curvature solve failed: "
                           + str(stats.get("return_status")))
    offset_values = basis @ solution
    path = xy + np.column_stack((normal_x, normal_y)) * offset_values[:, None]
    path_previous, path_next = np.roll(path, 1, axis=0), np.roll(path, -1, axis=0)
    edge_a = path - path_previous
    edge_b = path_next - path
    chord = path_next - path_previous
    len_a = np.maximum(np.linalg.norm(edge_a, axis=1), 1.0e-12)
    len_b = np.maximum(np.linalg.norm(edge_b, axis=1), 1.0e-12)
    len_c = np.maximum(np.linalg.norm(chord, axis=1), 1.0e-12)
    discrete_kappa = 2.0 * (edge_a[:, 0] * edge_b[:, 1]
                            - edge_a[:, 1] * edge_b[:, 0]) / (len_a * len_b * len_c)
    discrete_curvature_energy = float(np.sum(
        discrete_kappa * discrete_kappa * 0.5 * (len_a + len_b)))
    discrete_path_length = float(np.sum(len_a))
    if reference_xy is None:
        discrete_reference_energy = 0.0
    else:
        error_squared = np.sum((path - reference_xy) ** 2, axis=1)
        discrete_reference_energy = float(np.sum(
            error_squared * 0.5 * (len_a + len_b)))
    if entry_weights is None or reference_xy is None:
        discrete_entry_reference_energy = 0.0
        discrete_entry_curvature_energy = 0.0
    else:
        entry_target = (entry_reference_xy if entry_reference_xy is not None
                        else reference_xy)
        discrete_entry_reference_energy = float(np.sum(
            entry_weights * np.sum((path - entry_target) ** 2, axis=1)
            * 0.5 * (len_a + len_b)))
        discrete_entry_curvature_energy = float(np.sum(
            entry_weights * discrete_kappa * discrete_kappa
            * 0.5 * (len_a + len_b)))
    if exit_weights is None or reference_xy is None:
        discrete_exit_reference_energy = 0.0
    else:
        discrete_exit_reference_energy = float(np.sum(
            exit_weights * np.sum((path - reference_xy) ** 2, axis=1)
            * 0.5 * (len_a + len_b)))
    if straight_weights is None or straight_target_x is None:
        discrete_straight_energy = 0.0
        discrete_straight_curvature_energy = 0.0
    else:
        discrete_straight_energy = float(np.sum(
            straight_weights * (path[:, 0] - straight_target_x) ** 2
            * 0.5 * (len_a + len_b)))
        discrete_straight_curvature_energy = float(np.sum(
            straight_weights * discrete_kappa * discrete_kappa
            * 0.5 * (len_a + len_b)))
    if outer_wall_weights is None or outer_wall_target_offset is None:
        discrete_outer_wall_energy = 0.0
    else:
        discrete_outer_wall_energy = float(np.sum(
            outer_wall_weights * (offset_values - outer_wall_target_offset) ** 2
            * 0.5 * (len_a + len_b)))
    active_tolerance = 0.025 + 1.0e-8
    active = ((offset_values - lower) <= active_tolerance) | (
        (upper - offset_values) <= active_tolerance)
    return {"offset_m": offset_values,
            "status": str(stats.get("return_status")),
            "iterations": int(stats.get("iter_count", 0)),
            "objective": float(result["f"]),
            "curvature_energy_per_m": discrete_curvature_energy,
            "path_length_m": discrete_path_length,
            "length_penalty_per_m2": float(length_penalty_per_m2),
            "length_penalty_term_per_m": float(
                length_penalty_per_m2 * discrete_path_length),
            "reference_attraction_per_m4": float(reference_attraction_per_m4),
            "reference_deviation_energy_m3": discrete_reference_energy,
            "reference_attraction_term_per_m": float(
                reference_attraction_per_m4 * discrete_reference_energy),
            "entry_reference_attraction_per_m4": float(
                entry_reference_attraction_per_m4),
            "entry_reference_deviation_energy_m3": discrete_entry_reference_energy,
            "entry_reference_attraction_term_per_m": float(
                entry_reference_attraction_per_m4 * discrete_entry_reference_energy),
            "entry_curvature_energy_per_m": discrete_entry_curvature_energy,
            "entry_curvature_multiplier": float(entry_curvature_multiplier),
            "exit_reference_attraction_per_m4": float(
                exit_reference_attraction_per_m4),
            "exit_reference_deviation_energy_m3": discrete_exit_reference_energy,
            "exit_reference_attraction_term_per_m": float(
                exit_reference_attraction_per_m4 * discrete_exit_reference_energy),
            "straightness_energy_m3": discrete_straight_energy,
            "straightness_attraction_per_m4": float(
                straight_attraction_per_m4),
            "straightness_attraction_term_per_m": float(
                straight_attraction_per_m4 * discrete_straight_energy),
            "straight_section_curvature_energy_per_m":
                discrete_straight_curvature_energy,
            "straight_section_curvature_multiplier": float(
                straight_curvature_multiplier),
            "outer_wall_attraction_per_m4": float(outer_wall_attraction_per_m4),
            "outer_wall_deviation_energy_m3": discrete_outer_wall_energy,
            "outer_wall_attraction_term_per_m": float(
                outer_wall_attraction_per_m4 * discrete_outer_wall_energy),
            "harmonics": int(harmonics),
            "active_corridor_fraction": float(np.mean(active))}


def _geometry(xy: np.ndarray, spacing: float):
    closed = np.vstack((xy, xy[0]))
    ds_line = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    station = np.r_[0.0, np.cumsum(ds_line)]
    if np.any(ds_line <= 1.0e-8):
        raise ValueError("Generated path has a zero-length segment")
    length = float(station[-1])
    count = max(len(xy), int(math.ceil(length / spacing)))
    query = np.linspace(0.0, length, count, endpoint=False)
    sx = CubicSpline(station, closed[:, 0], bc_type="periodic")
    sy = CubicSpline(station, closed[:, 1], bc_type="periodic")
    p = np.column_stack((sx(query), sy(query)))
    dx, dy = sx(query, 1), sy(query, 1)
    ddx, ddy = sx(query, 2), sy(query, 2)
    psi = np.unwrap(np.arctan2(dy, dx))
    kappa = (dx * ddy - dy * ddx) / np.maximum(
        (dx * dx + dy * dy) ** 1.5, 1.0e-12)
    return query, p, psi, kappa, length, sx, sy


def _map_widths(grid: MapGrid, xy: np.ndarray, heading: np.ndarray,
                fallback: np.ndarray, max_width: float = 2.5,
                step: float = 0.01) -> np.ndarray:
    distances = np.arange(0.0, max_width + step, step)
    normal = np.column_stack((-np.sin(heading), np.cos(heading)))
    widths = np.empty((len(xy), 2), dtype=np.float64)
    for side, sign in enumerate((1.0, -1.0)):
        points = xy[:, None, :] + sign * distances[None, :, None] * normal[:, None, :]
        clearance = np.asarray(grid.distance(points.reshape(-1, 2))).reshape(
            len(xy), len(distances))
        blocked = clearance <= 0.025
        has_boundary = np.any(blocked, axis=1)
        first = np.argmax(blocked, axis=1)
        scanned = np.where(has_boundary,
                           np.maximum(0.0, distances[first] - step),
                           fallback[:, side])
        # At a detected boundary the current map scan is authoritative. Only
        # use the inherited conservative width where a ray finds no boundary
        # (e.g. an open map seam); capping every ray by the old path's widths
        # incorrectly rejected valid wall-side offsets of the new geometry.
        if np.any(~has_boundary & (fallback[:, side] <= 0.0)):
            raise ValueError("Open map ray has no positive inherited corridor")
        widths[:, side] = scanned
    if np.min(widths) <= 0.0:
        raise ValueError("Map-derived corridor contains a nonpositive width")
    return widths


def _minimum_clearance(grid: MapGrid, xy: np.ndarray, heading: np.ndarray,
                       margin: float, chunk_size: int = 128) -> float:
    values = []
    for begin in range(0, len(xy), chunk_size):
        end = min(len(xy), begin + chunk_size)
        offsets = np.array([0.0])
        clearance = _footprint_clearance_grid(
            grid, xy[begin:end], heading[begin:end], offsets, margin)
        values.append(float(np.min(clearance)))
    return min(values)


def _smooth_region_mask(reference_xy: np.ndarray, x_bounds: tuple[float, float],
                        y_bounds: tuple[float, float], taper_m: float = 0.25) -> np.ndarray:
    """Smooth rectangular spatial mask, fixed to the aligned reference geometry."""
    if (x_bounds[0] >= x_bounds[1] or y_bounds[0] >= y_bounds[1]
            or taper_m <= 0.0):
        raise ValueError("Local region bounds and taper must be positive")

    def interval_weight(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
        inward = np.clip(np.minimum(value - lower, upper - value) / taper_m, 0.0, 1.0)
        # Cubic smoothstep avoids a derivative jump where local weighting begins/ends.
        return inward * inward * (3.0 - 2.0 * inward)

    return (interval_weight(reference_xy[:, 0], *x_bounds)
            * interval_weight(reference_xy[:, 1], *y_bounds))


def _safe_wall_target_offsets(lower: np.ndarray, upper: np.ndarray,
                              side: str, setback_m: float,
                              smoothing_sigma_nodes: float) -> np.ndarray:
    """Target a swept-footprint-safe offset toward one side of the path.

    Positive offsets follow the path's left normal; negative offsets point to
    its right. ``lower`` and ``upper`` are already clearance-checked bounds.
    """
    if side not in ("left", "right"):
        raise ValueError("Wall-target side must be 'left' or 'right'")
    if (lower.ndim != 1 or upper.shape != lower.shape
            or np.any(lower >= upper) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()):
        raise ValueError("Wall-target offset bounds must be finite ordered vectors")
    if (not math.isfinite(setback_m) or setback_m < 0.05
            or not math.isfinite(smoothing_sigma_nodes)
            or smoothing_sigma_nodes < 0.0):
        raise ValueError("Wall-target setback must be at least 5 cm and smoothing nonnegative")

    if side == "left":
        smoothed_limit = gaussian_filter1d(
            upper, smoothing_sigma_nodes, mode="wrap")
        target = np.minimum(smoothed_limit - setback_m, upper - 0.05)
        return np.maximum(target, lower + 0.02)

    smoothed_limit = gaussian_filter1d(
        lower, smoothing_sigma_nodes, mode="wrap")
    target = np.maximum(smoothed_limit + setback_m, lower + 0.05)
    return np.minimum(target, upper - 0.02)


def _align_reference(path: Path, base_xy: np.ndarray, spacing: float) -> np.ndarray:
    source_xy, speed, widths = _read_reference(path)
    _, samples, _, _, _, _, _, _ = _resample_reference(
        source_xy, speed, widths, spacing)
    seam_index = int(np.argmin(np.linalg.norm(samples - base_xy[0], axis=1)))
    samples = np.roll(samples, -seam_index, axis=0)
    seam_distance = float(np.linalg.norm(samples[0] - base_xy[0]))
    if seam_distance > 1.0:
        raise ValueError(f"Reference path {path} does not share the source track seam")
    if np.dot(samples[1] - samples[0], base_xy[1] - base_xy[0]) <= 0.0:
        raise ValueError(f"Reference path {path} direction does not match source geometry")
    reference_phase = np.arange(len(samples), dtype=np.float64) / len(samples)
    base_phase = np.arange(len(base_xy), dtype=np.float64) / len(base_xy)
    extended_phase = np.r_[reference_phase - 1.0, reference_phase,
                           reference_phase + 1.0]
    return np.column_stack((
        np.interp(base_phase, extended_phase, np.tile(samples[:, 0], 3)),
        np.interp(base_phase, extended_phase, np.tile(samples[:, 1], 3))))


def generate(map_path: Path, source_path: Path, output_path: Path,
             spacing: float, margin: float, max_offset: float,
             offset_step: float, harmonics: int,
             midline_smoothing_m: float,
             length_penalty_per_m2: float = 0.0,
             reference_path: Path | None = None,
             reference_attraction_per_m4: float = 0.0,
             entry_reference_path: Path | None = None,
             entry_region_x_m: tuple[float, float] | None = None,
             entry_region_y_m: tuple[float, float] | None = None,
             entry_reference_attraction_per_m4: float = 0.0,
             entry_curvature_multiplier: float = 0.0,
             exit_region_x_m: tuple[float, float] | None = None,
             exit_region_y_m: tuple[float, float] | None = None,
             exit_reference_attraction_per_m4: float = 0.0,
             local_region_taper_m: float = 0.25,
             straight_y_min_m: float | None = None,
             straight_y_max_m: float | None = None,
             straight_x_min_m: float = 0.0,
             straight_taper_m: float = 0.4,
             straight_attraction_per_m4: float = 0.0,
             straight_curvature_multiplier: float = 0.0,
             outer_wall_region_x_m: tuple[float, float] | None = None,
             outer_wall_region_y_m: tuple[float, float] | None = None,
             outer_wall_taper_m: float = 0.35,
             outer_wall_setback_m: float = 0.10,
             outer_wall_smoothing_m: float = 0.30,
             outer_wall_attraction_per_m4: float = 0.0,
             straight_x_max_m: float | None = None,
             preserve_source_geometry: bool = False,
             straight_target_x_m: float | None = None,
             outer_wall_side: str = "left") -> dict:
    grid = MapGrid.load(map_path)
    source_xy, source_speed, source_widths = _read_reference(source_path)
    _, input_xy, input_psi, input_kappa, input_speed, input_widths, input_ds, input_length = (
        _resample_reference(source_xy, source_speed, source_widths, spacing))
    # By default recover a geometric road midline from verified left/right
    # widths. For a local refinement pass, optionally preserve the supplied
    # candidate geometry exactly as the base so unrelated sections do not
    # move while a spatially limited objective is applied.
    smoothing_sigma_nodes = midline_smoothing_m / input_ds
    if preserve_source_geometry:
        centerline_offset = np.zeros(len(input_xy), dtype=np.float64)
        midpoint_xy = input_xy.copy()
    else:
        input_normal = np.column_stack((-np.sin(input_psi), np.cos(input_psi)))
        centerline_offset = 0.5 * (input_widths[:, 0] - input_widths[:, 1])
        smooth_centerline_offset = gaussian_filter1d(
            centerline_offset, smoothing_sigma_nodes, mode="wrap")
        midpoint_xy = input_xy + smooth_centerline_offset[:, None] * input_normal
    (_, base_xy, base_psi, base_kappa, base_length, _, _) = _geometry(
        midpoint_xy, spacing)
    ds = base_length / len(base_xy)
    base_progress_fraction = np.linspace(0.0, 1.0, len(base_xy), endpoint=False)
    input_progress_fraction = np.linspace(0.0, 1.0, len(input_speed), endpoint=False)
    base_speed = np.interp(base_progress_fraction, input_progress_fraction,
                           input_speed, period=1.0)
    half_corridor = gaussian_filter1d(
        0.5 * (input_widths[:, 0] + input_widths[:, 1]),
        smoothing_sigma_nodes, mode="wrap")
    half_corridor = np.interp(base_progress_fraction, input_progress_fraction,
                              half_corridor, period=1.0)
    base_widths = np.column_stack((half_corridor, half_corridor))
    aligned_reference = None
    if reference_path is not None:
        aligned_reference = _align_reference(reference_path, base_xy, spacing)
    aligned_entry_reference = (None if entry_reference_path is None else
                               _align_reference(entry_reference_path, base_xy, spacing))
    if aligned_entry_reference is None:
        aligned_entry_reference = aligned_reference
    if reference_attraction_per_m4 > 0.0 and aligned_reference is None:
        raise ValueError("Reference attraction requires --reference-path")
    entry_enabled = (entry_reference_attraction_per_m4 > 0.0
                     or entry_curvature_multiplier > 0.0)
    exit_enabled = exit_reference_attraction_per_m4 > 0.0
    if entry_enabled and (aligned_reference is None or entry_region_x_m is None
                          or entry_region_y_m is None):
        raise ValueError("Entry terms require a reference path and entry region bounds")
    if exit_enabled and (aligned_reference is None or exit_region_x_m is None
                         or exit_region_y_m is None):
        raise ValueError("Exit attraction requires a reference path and exit region bounds")
    entry_weights = (_smooth_region_mask(
        aligned_reference, entry_region_x_m, entry_region_y_m, local_region_taper_m)
        if entry_enabled else None)
    exit_weights = (_smooth_region_mask(
        aligned_reference, exit_region_x_m, exit_region_y_m, local_region_taper_m)
        if exit_enabled else None)
    if entry_weights is not None and np.count_nonzero(entry_weights > 0.05) < 8:
        raise ValueError("Entry bounds select too few reference samples")
    if exit_weights is not None and np.count_nonzero(exit_weights > 0.05) < 8:
        raise ValueError("Exit bounds select too few reference samples")
    if (straight_attraction_per_m4 > 0.0
            or straight_curvature_multiplier > 0.0):
        if (aligned_reference is None or straight_y_min_m is None
                or straight_y_max_m is None or straight_y_min_m >= straight_y_max_m):
            raise ValueError("Straight-section attraction needs a reference and valid y bounds")
        if not math.isfinite(straight_x_min_m):
            raise ValueError("Straight-section x threshold must be finite")
        straight_mask = ((aligned_reference[:, 1] >= straight_y_min_m)
                         & (aligned_reference[:, 1] <= straight_y_max_m)
                         & (aligned_reference[:, 0] >= straight_x_min_m))
        if straight_x_max_m is not None:
            if not math.isfinite(straight_x_max_m) or straight_x_max_m <= straight_x_min_m:
                raise ValueError("Straight-section x bounds must be finite and ordered")
            straight_mask &= aligned_reference[:, 0] <= straight_x_max_m
        if int(np.count_nonzero(straight_mask)) < 8:
            raise ValueError("Straight-section bounds select too few reference samples")
        straight_target_x = float(np.mean(aligned_reference[straight_mask, 0]))
        if straight_target_x_m is not None:
            if not math.isfinite(straight_target_x_m):
                raise ValueError("Straight-section target x must be finite")
            straight_target_x = float(straight_target_x_m)
        # Smoothly taper the straight preference before each turn. A longer
        # transition avoids an abrupt curvature change at the straight/turn join.
        if not math.isfinite(straight_taper_m) or straight_taper_m <= 0.0:
            raise ValueError("Straight-section blend length must be positive")
        y = aligned_reference[:, 1]
        def smoothstep(values):
            values = np.clip(values, 0.0, 1.0)
            return values * values * (3.0 - 2.0 * values)
        lower_weight = smoothstep((y - straight_y_min_m) / straight_taper_m)
        upper_weight = smoothstep((straight_y_max_m - y) / straight_taper_m)
        selected_weight = smoothstep(
            (aligned_reference[:, 0] - straight_x_min_m) / 0.25)
        if straight_x_max_m is not None:
            selected_weight *= smoothstep(
                (straight_x_max_m - aligned_reference[:, 0]) / 0.25)
        straight_weights = np.minimum(lower_weight, upper_weight) * selected_weight
    else:
        straight_target_x = None
        straight_weights = None
    offsets = np.arange(-max_offset, max_offset + 0.5 * offset_step, offset_step)
    if not np.any(np.isclose(offsets, 0.0, atol=1.0e-10)):
        offsets = np.sort(np.r_[offsets, 0.0])
    clearance_grid = _footprint_clearance_grid(
        grid, base_xy, base_psi, offsets, margin)
    lower, upper = _safe_offset_bounds(clearance_grid, offsets)
    if outer_wall_attraction_per_m4 > 0.0:
        if (aligned_reference is None or outer_wall_region_x_m is None
                or outer_wall_region_y_m is None):
            raise ValueError("Outer-wall objective needs a reference and local region bounds")
        if (not math.isfinite(outer_wall_taper_m) or outer_wall_taper_m <= 0.0
                or not math.isfinite(outer_wall_setback_m)
                or outer_wall_setback_m < 0.05
                or not math.isfinite(outer_wall_smoothing_m)
                or outer_wall_smoothing_m < 0.0):
            raise ValueError("Outer-wall taper/smoothing must be nonnegative and setback at least 5 cm")
        outer_wall_weights = _smooth_region_mask(
            aligned_reference, outer_wall_region_x_m, outer_wall_region_y_m,
            outer_wall_taper_m)
        if np.count_nonzero(outer_wall_weights > 0.05) < 8:
            raise ValueError("Outer-wall bounds select too few reference samples")
        # Select the wall side explicitly: positive path-normal offsets are
        # left, negative offsets are right. Keep the target inside the exact
        # swept-footprint corridor after raster-bound smoothing.
        outer_wall_target_offset = _safe_wall_target_offsets(
            lower, upper, outer_wall_side, outer_wall_setback_m,
            outer_wall_smoothing_m / ds)
    else:
        outer_wall_weights = None
        outer_wall_target_offset = None
    fit = _minimum_curvature_offset(
        base_xy, base_psi, lower, upper, harmonics,
        length_penalty_per_m2, aligned_reference,
        reference_attraction_per_m4, entry_weights, aligned_entry_reference,
        entry_reference_attraction_per_m4, entry_curvature_multiplier,
        exit_weights, exit_reference_attraction_per_m4, straight_weights,
        straight_target_x, straight_attraction_per_m4,
        straight_curvature_multiplier, outer_wall_weights,
        outer_wall_target_offset, outer_wall_attraction_per_m4)
    offset = fit["offset_m"]
    normal = np.column_stack((-np.sin(base_psi), np.cos(base_psi)))

    # Check the exact periodic-spline geometry against the mapped vehicle
    # footprint; if needed, retain the largest safe fraction of the solved
    # offset field.
    selected = None
    alpha = 1.0
    dense_spacing = min(0.025, grid.resolution / 2.0)
    for _ in range(24):
        candidate_nodes = base_xy + alpha * offset[:, None] * normal
        q, dense_xy, dense_psi, dense_kappa, path_length, sx, sy = _geometry(
            candidate_nodes, dense_spacing)
        margin_actual = _minimum_clearance(
            grid, dense_xy, dense_psi, margin)
        raw_center_clearance = float(np.min(grid.distance(dense_xy)))
        if margin_actual >= 0.0 and raw_center_clearance >= margin + 0.1365:
            selected = (alpha, candidate_nodes, q, dense_xy, dense_psi,
                        dense_kappa, path_length, sx, sy, margin_actual,
                        raw_center_clearance)
            break
        alpha *= 0.5
    if selected is None:
        raise ValueError("Minimum-curvature path failed exact map/footprint checks")

    (alpha, candidate_nodes, node_s, dense_xy, dense_psi, dense_kappa,
     path_length, sx, sy, footprint_clearance, center_clearance) = selected
    # Use IPOPT's own spatial spacing for the exported path geometry.
    node_count = max(len(base_xy), int(math.ceil(path_length / spacing)))
    node_s = np.linspace(0.0, path_length, node_count, endpoint=False)
    path_xy = np.column_stack((sx(node_s), sy(node_s)))
    dx, dy = sx(node_s, 1), sy(node_s, 1)
    ddx, ddy = sx(node_s, 2), sy(node_s, 2)
    path_psi = np.unwrap(np.arctan2(dy, dx))
    path_kappa = (dx * ddy - dy * ddx) / np.maximum(
        (dx * dx + dy * dy) ** 1.5, 1.0e-12)
    base_progress = np.mod(node_s / path_length * base_length, base_length)
    base_stations = np.arange(len(base_xy), dtype=np.float64) * ds
    offset_on_output = np.interp(
        base_progress,
        np.r_[base_stations - base_length, base_stations,
              base_stations + base_length],
        np.tile(alpha * offset, 3))
    base_widths_on_output = np.column_stack((
        np.interp(base_progress,
                  np.r_[base_stations - base_length, base_stations,
                        base_stations + base_length],
                  np.tile(base_widths[:, 0], 3)),
        np.interp(base_progress,
                  np.r_[base_stations - base_length, base_stations,
                        base_stations + base_length],
                  np.tile(base_widths[:, 1], 3))))
    fallback_widths = np.column_stack((
        base_widths_on_output[:, 0] - offset_on_output,
        base_widths_on_output[:, 1] + offset_on_output))
    map_widths = _map_widths(grid, path_xy, path_psi, fallback_widths)

    # Carry the prior speed profile only as an optional IPOPT initialization;
    # it is not a constraint and does not affect the geometric objective.
    seed_speed = np.interp(
        node_s / path_length,
        np.linspace(0.0, 1.0, len(base_speed), endpoint=False),
        base_speed, period=1.0)
    accel_seed = np.gradient(seed_speed, node_s, edge_order=1)
    accel_seed = np.clip(accel_seed, -20.0, 20.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_left_m,d_right_m\n")
        stream.write("# geometry: map-checked curvature-length seed; speed is initialization only\n")
        writer = csv.writer(stream, lineterminator="\n")
        for i in range(node_count):
            writer.writerow((f"{node_s[i]:.8f}", f"{path_xy[i,0]:.8f}",
                             f"{path_xy[i,1]:.8f}", f"{path_psi[i]:.9f}",
                             f"{path_kappa[i]:.9f}", f"{seed_speed[i]:.6f}",
                             f"{accel_seed[i]:.6f}",
                             f"{map_widths[i,0]:.6f}",
                             f"{map_widths[i,1]:.6f}"))

    input_rms = float(np.sqrt(np.mean(input_kappa * input_kappa)))
    midpoint_rms = float(np.sqrt(np.mean(base_kappa * base_kappa)))
    seed_rms = float(np.sqrt(np.mean(path_kappa * path_kappa)))
    report = {
        "status": "minimum_curvature_geometry_seed_generated",
        "map": str(map_path.resolve()),
        "source_geometry": str(source_path.resolve()),
        "output_csv": str(output_path.resolve()),
        "objective": ("exact discrete curvature energy plus weighted path length, "
                      "optional reference-path attraction, and a local straight-section term"),
        "length_penalty_per_m2": float(length_penalty_per_m2),
        "reference_path": (str(reference_path.resolve())
                           if reference_path is not None else None),
        "reference_attraction_per_m4": float(reference_attraction_per_m4),
        "entry_reference_path": (str(entry_reference_path.resolve())
                                  if entry_reference_path is not None else None),
        "local_turn_sections": {
            "entry": ({
                "reference_x_bounds_m": list(entry_region_x_m),
                "reference_y_bounds_m": list(entry_region_y_m),
                "smooth_taper_m": float(local_region_taper_m),
                "target_path": (str(entry_reference_path.resolve())
                                if entry_reference_path is not None else
                                (str(reference_path.resolve())
                                 if reference_path is not None else None)),
                "reference_attraction_per_m4": float(
                    entry_reference_attraction_per_m4),
                "curvature_multiplier": float(entry_curvature_multiplier),
            } if entry_weights is not None else None),
            "exit": ({
                "reference_x_bounds_m": list(exit_region_x_m),
                "reference_y_bounds_m": list(exit_region_y_m),
                "smooth_taper_m": float(local_region_taper_m),
                "reference_attraction_per_m4": float(
                    exit_reference_attraction_per_m4),
            } if exit_weights is not None else None),
        },
        "straight_section": ({
            "y_min_m": float(straight_y_min_m),
            "y_max_m": float(straight_y_max_m),
            "reference_x_min_m": float(straight_x_min_m),
            "reference_x_max_m": (None if straight_x_max_m is None
                                   else float(straight_x_max_m)),
            "blend_length_m": float(straight_taper_m),
            "target_x_m": straight_target_x,
            "target_x_override_m": straight_target_x_m,
            "attraction_per_m4": float(straight_attraction_per_m4),
            "curvature_multiplier": float(straight_curvature_multiplier),
        } if straight_weights is not None else None),
        "outer_wall_section": ({
            "reference_x_bounds_m": list(outer_wall_region_x_m),
            "reference_y_bounds_m": list(outer_wall_region_y_m),
            "target_side_of_path": outer_wall_side,
            "positive_offset_is_path_left": True,
            "smooth_taper_m": float(outer_wall_taper_m),
            "safe_bound_smoothing_sigma_m": float(outer_wall_smoothing_m),
            "setback_from_swept_footprint_safe_limit_m": float(outer_wall_setback_m),
            "attraction_per_m4": float(outer_wall_attraction_per_m4),
            "target_offset_min_m": float(np.min(outer_wall_target_offset)),
            "target_offset_max_m": float(np.max(outer_wall_target_offset)),
        } if outer_wall_weights is not None else None),
        "role": "IPOPT initial geometry; IPOPT remains free to change path and optimize vehicle dynamics/speed",
        "starting_geometry": {
            "method": ("preserved supplied source path for local refinement"
                       if preserve_source_geometry else
                       "road midpoint estimated from verified per-station left/right map widths"),
            "input_geometry_shifted_to_midpoint_max_m": float(np.max(np.abs(centerline_offset))),
            "input_geometry_shifted_to_midpoint_rms_m": float(np.sqrt(np.mean(centerline_offset**2))),
            "midline_offset_smoothing_sigma_m": float(midline_smoothing_m),
        },
        "map_footprint_check": {
            "extra_wall_margin_m": margin,
            "minimum_vehicle_perimeter_clearance_after_margin_m": footprint_clearance,
            "minimum_centerline_distance_to_occupied_cells_m": center_clearance,
            "clearance_test_spacing_m": dense_spacing,
            "pass": footprint_clearance >= 0.0 and center_clearance >= margin + 0.1365,
        },
        "mesh": {"nodes": int(node_count), "spacing_m": float(path_length / node_count),
                 "length_m": float(path_length), "source_length_m": float(base_length)},
        "offset_solve": {"method": "CasADi/IPOPT exact discrete curvature-length objective",
                         "status": fit["status"],
                         "iterations": fit["iterations"],
                         "objective": fit["objective"],
                         "discrete_curvature_energy_per_m": fit["curvature_energy_per_m"],
                         "discrete_path_length_m": fit["path_length_m"],
                         "length_penalty_term_per_m": fit["length_penalty_term_per_m"],
                         "reference_deviation_energy_m3": fit[
                             "reference_deviation_energy_m3"],
                         "reference_attraction_term_per_m": fit[
                             "reference_attraction_term_per_m"],
                         "entry_reference_deviation_energy_m3": fit[
                             "entry_reference_deviation_energy_m3"],
                         "entry_reference_attraction_term_per_m": fit[
                             "entry_reference_attraction_term_per_m"],
                         "entry_curvature_energy_per_m": fit[
                             "entry_curvature_energy_per_m"],
                         "entry_curvature_multiplier": fit[
                             "entry_curvature_multiplier"],
                         "exit_reference_deviation_energy_m3": fit[
                             "exit_reference_deviation_energy_m3"],
                         "exit_reference_attraction_term_per_m": fit[
                             "exit_reference_attraction_term_per_m"],
                         "straightness_energy_m3": fit["straightness_energy_m3"],
                         "straightness_attraction_term_per_m": fit[
                             "straightness_attraction_term_per_m"],
                         "straight_section_curvature_energy_per_m": fit[
                             "straight_section_curvature_energy_per_m"],
                         "outer_wall_deviation_energy_m3": fit[
                             "outer_wall_deviation_energy_m3"],
                         "outer_wall_attraction_per_m4": float(
                             outer_wall_attraction_per_m4),
                         "outer_wall_attraction_term_per_m": fit[
                             "outer_wall_attraction_term_per_m"],
                         "fourier_harmonics": fit["harmonics"],
                         "fraction_within_2_5cm_of_corridor_bound": fit["active_corridor_fraction"],
                         "largest_offset_m": float(np.max(np.abs(alpha * offset))),
                         "safe_offset_min_m": float(np.min(lower)),
                         "safe_offset_max_m": float(np.max(upper)),
                         "exact_geometry_safety_scale": float(alpha)},
        "start_pose": {"fixed_world_pose": False,
                       "periodic_csv_seam_is_only_an_index_origin": True},
        "curvature": {
            "input_raceline_rms_abs_per_m": input_rms,
            "road_midpoint_rms_abs_per_m": midpoint_rms,
            "seed_rms_abs_per_m": seed_rms,
            "input_raceline_peak_abs_per_m": float(np.max(np.abs(input_kappa))),
            "road_midpoint_peak_abs_per_m": float(np.max(np.abs(base_kappa))),
            "seed_peak_abs_per_m": float(np.max(np.abs(path_kappa))),
            "rms_reduction_from_road_midpoint_percent": 100.0 * (midpoint_rms - seed_rms) / max(midpoint_rms, 1.0e-12),
            "rms_reduction_from_input_raceline_percent": 100.0 * (input_rms - seed_rms) / max(input_rms, 1.0e-12),
        },
        "speed_seed": {"min_mps": float(np.min(seed_speed)),
                       "mean_mps": float(np.mean(seed_speed)),
                       "max_mps": float(np.max(seed_speed)),
                       "used_as_constraint": False},
        "corridor_widths": {"left_min_m": float(np.min(map_widths[:, 0])),
                            "right_min_m": float(np.min(map_widths[:, 1])),
                            "source": "practice-map occupancy ray scan at 1 cm increments"},
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # A pure minimum-curvature solve must improve its own curvature metric.
    # With explicit path/reference/wall terms, however, a deliberate local
    # geometry tradeoff may increase global RMS while satisfying the requested
    # weighted objective; do not reject those otherwise map-safe candidates.
    pure_min_curvature = (
        length_penalty_per_m2 == 0.0
        and reference_attraction_per_m4 == 0.0
        and entry_reference_attraction_per_m4 == 0.0
        and entry_curvature_multiplier == 0.0
        and exit_reference_attraction_per_m4 == 0.0
        and straight_attraction_per_m4 == 0.0
        and straight_curvature_multiplier == 0.0
        and outer_wall_attraction_per_m4 == 0.0)
    if pure_min_curvature and seed_rms >= midpoint_rms:
        raise RuntimeError("Exact spline curvature did not improve over the recovered road midpoint")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--map", type=Path,
                        default=root / "f1tenth_planning/maps/autodrive_practice_20260924_b.pgm")
    parser.add_argument("--centerline", type=Path,
                        default=root / "f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv")
    parser.add_argument("--output", type=Path,
                        default=root / "f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_min_curvature_seed.csv")
    parser.add_argument("--spacing", type=float, default=0.06)
    parser.add_argument("--wall-margin", type=float, default=0.05)
    parser.add_argument("--max-offset", type=float, default=2.5)
    parser.add_argument("--offset-step", type=float, default=0.025)
    parser.add_argument("--harmonics", type=int, default=16,
                        help="number of periodic Fourier modes for the smooth path offset")
    parser.add_argument("--midline-smoothing-m", type=float, default=0.24,
                        help="Gaussian sigma used to smooth quantized map-width midpoint offsets")
    parser.add_argument("--length-penalty-per-m2", type=float, default=0.0,
                        help=("weight for path length in curvature-energy objective; "
                              "0 is pure minimum curvature, larger values favor shorter paths"))
    parser.add_argument("--reference-path", type=Path, default=None,
                        help="optional closed path to stay near while optimizing curvature/length")
    parser.add_argument("--reference-attraction-per-m4", type=float, default=0.0,
                        help="weight for integrated squared XY deviation from --reference-path")
    parser.add_argument("--entry-reference-path", type=Path, default=None,
                        help="optional path to follow locally in the entry/apex zone")
    parser.add_argument("--entry-x-min", type=float, default=None,
                        help="lower x bound for a local entry-zone objective, in map metres")
    parser.add_argument("--entry-x-max", type=float, default=None,
                        help="upper x bound for a local entry-zone objective, in map metres")
    parser.add_argument("--entry-y-min", type=float, default=None,
                        help="lower y bound for a local entry-zone objective, in map metres")
    parser.add_argument("--entry-y-max", type=float, default=None,
                        help="upper y bound for a local entry-zone objective, in map metres")
    parser.add_argument("--entry-reference-attraction-per-m4", type=float, default=0.0,
                        help="additional local attraction to the reference line in the entry zone")
    parser.add_argument("--entry-curvature-multiplier", type=float, default=0.0,
                        help="additional local curvature-energy multiplier in the entry zone")
    parser.add_argument("--exit-x-min", type=float, default=None,
                        help="lower x bound for local exit-zone reference attraction")
    parser.add_argument("--exit-x-max", type=float, default=None,
                        help="upper x bound for local exit-zone reference attraction")
    parser.add_argument("--exit-y-min", type=float, default=None,
                        help="lower y bound for local exit-zone reference attraction")
    parser.add_argument("--exit-y-max", type=float, default=None,
                        help="upper y bound for local exit-zone reference attraction")
    parser.add_argument("--exit-reference-attraction-per-m4", type=float, default=0.0,
                        help="additional local attraction to the reference line in the exit zone")
    parser.add_argument("--local-region-taper-m", type=float, default=0.25,
                        help="smooth taper distance at local entry/exit region boundaries")
    parser.add_argument("--straight-y-min", type=float, default=None,
                        help="lower map-y bound for an optional straight-section penalty")
    parser.add_argument("--straight-y-max", type=float, default=None,
                        help="upper map-y bound for an optional straight-section penalty")
    parser.add_argument("--straight-x-min", type=float, default=0.0,
                        help="minimum reference x used to distinguish the straight corridor")
    parser.add_argument("--straight-x-max", type=float, default=None,
                        help="optional maximum reference x to restrict the straight-section term")
    parser.add_argument("--straight-taper-m", type=float, default=0.4,
                        help="distance over which the straightness preference blends into each turn")
    parser.add_argument("--straight-attraction-per-m4", type=float, default=0.0,
                        help="weight penalizing lateral x deviation from a straight line in the selected section")
    parser.add_argument("--straight-curvature-multiplier", type=float, default=0.0,
                        help="additional multiplier on curvature energy in the selected straight section")
    parser.add_argument("--straight-target-x", type=float, default=None,
                        help="optional explicit map-x target for the selected local section")
    parser.add_argument("--outer-wall-x-min", type=float, default=None,
                        help="lower x bound for a local outside-wall preference")
    parser.add_argument("--outer-wall-x-max", type=float, default=None,
                        help="upper x bound for a local outside-wall preference")
    parser.add_argument("--outer-wall-y-min", type=float, default=None,
                        help="lower y bound for a local outside-wall preference")
    parser.add_argument("--outer-wall-y-max", type=float, default=None,
                        help="upper y bound for a local outside-wall preference")
    parser.add_argument("--outer-wall-side", choices=("left", "right"), default="left",
                        help="which side of the path-normal corridor to target")
    parser.add_argument("--outer-wall-taper-m", type=float, default=0.35,
                        help="smooth spatial taper at outside-wall region boundaries")
    parser.add_argument("--outer-wall-setback-m", type=float, default=0.10,
                        help="target setback inside the swept-footprint-safe offset boundary")
    parser.add_argument("--outer-wall-smoothing-m", type=float, default=0.30,
                        help="Gaussian sigma used to smooth raster-derived safe offsets")
    parser.add_argument("--outer-wall-attraction-per-m4", type=float, default=0.0,
                        help="weight attracting the path to the smoothed safe outside offset")
    parser.add_argument("--preserve-source-geometry", action="store_true",
                        help="use the supplied path as the base instead of reconstructing the road midpoint")
    args = parser.parse_args()
    if (args.spacing <= 0.0 or args.wall_margin < 0.0 or args.max_offset <= 0.0
            or args.offset_step <= 0.0 or args.harmonics < 1
            or args.midline_smoothing_m < 0.0
            or not math.isfinite(args.length_penalty_per_m2)
            or args.length_penalty_per_m2 < 0.0
            or not math.isfinite(args.reference_attraction_per_m4)
            or args.reference_attraction_per_m4 < 0.0
            or not math.isfinite(args.entry_reference_attraction_per_m4)
            or args.entry_reference_attraction_per_m4 < 0.0
            or not math.isfinite(args.entry_curvature_multiplier)
            or args.entry_curvature_multiplier < 0.0
            or not math.isfinite(args.exit_reference_attraction_per_m4)
            or args.exit_reference_attraction_per_m4 < 0.0
            or not math.isfinite(args.local_region_taper_m)
            or args.local_region_taper_m <= 0.0
            or not math.isfinite(args.straight_taper_m)
            or args.straight_taper_m <= 0.0
            or not math.isfinite(args.straight_attraction_per_m4)
            or args.straight_attraction_per_m4 < 0.0
            or not math.isfinite(args.straight_curvature_multiplier)
            or args.straight_curvature_multiplier < 0.0
            or (args.straight_target_x is not None
                and not math.isfinite(args.straight_target_x))
            or not math.isfinite(args.outer_wall_taper_m)
            or args.outer_wall_taper_m <= 0.0
            or not math.isfinite(args.outer_wall_setback_m)
            or args.outer_wall_setback_m < 0.05
            or not math.isfinite(args.outer_wall_smoothing_m)
            or args.outer_wall_smoothing_m < 0.0
            or not math.isfinite(args.outer_wall_attraction_per_m4)
            or args.outer_wall_attraction_per_m4 < 0.0
            or (args.reference_attraction_per_m4 > 0.0
                and args.reference_path is None)
            or (args.straight_x_max is not None
                and (not math.isfinite(args.straight_x_max)
                     or args.straight_x_max <= args.straight_x_min))
            or ((args.straight_attraction_per_m4 > 0.0
                 or args.straight_curvature_multiplier > 0.0)
                and (args.reference_path is None
                     or args.straight_y_min is None
                     or args.straight_y_max is None
                     or args.straight_y_min >= args.straight_y_max))):
        parser.error("spacing, offset ranges, and harmonics must be positive; margin nonnegative")
    outer_wall_enabled = args.outer_wall_attraction_per_m4 > 0.0
    if outer_wall_enabled and any(value is None for value in (
            args.outer_wall_x_min, args.outer_wall_x_max,
            args.outer_wall_y_min, args.outer_wall_y_max)):
        parser.error("outer-wall objective requires --outer-wall-x/y-min/max bounds")
    entry_enabled = (args.entry_reference_attraction_per_m4 > 0.0
                     or args.entry_curvature_multiplier > 0.0)
    if entry_enabled and any(value is None for value in (
            args.entry_x_min, args.entry_x_max, args.entry_y_min, args.entry_y_max)):
        parser.error("entry objective requires --entry-x/y-min/max bounds")
    if args.exit_reference_attraction_per_m4 > 0.0 and any(value is None for value in (
            args.exit_x_min, args.exit_x_max, args.exit_y_min, args.exit_y_max)):
        parser.error("exit attraction requires --exit-x/y-min/max bounds")
    report = generate(args.map, args.centerline, args.output, args.spacing,
                      args.wall_margin, args.max_offset, args.offset_step,
                      args.harmonics, args.midline_smoothing_m,
                      args.length_penalty_per_m2, args.reference_path,
                      args.reference_attraction_per_m4, args.entry_reference_path,
                      ((args.entry_x_min, args.entry_x_max) if entry_enabled else None),
                      ((args.entry_y_min, args.entry_y_max) if entry_enabled else None),
                      args.entry_reference_attraction_per_m4,
                      args.entry_curvature_multiplier,
                      ((args.exit_x_min, args.exit_x_max)
                       if args.exit_reference_attraction_per_m4 > 0.0 else None),
                      ((args.exit_y_min, args.exit_y_max)
                       if args.exit_reference_attraction_per_m4 > 0.0 else None),
                      args.exit_reference_attraction_per_m4,
                      args.local_region_taper_m, args.straight_y_min,
                      args.straight_y_max, args.straight_x_min,
                      args.straight_taper_m,
                      args.straight_attraction_per_m4,
                      args.straight_curvature_multiplier,
                      ((args.outer_wall_x_min, args.outer_wall_x_max)
                       if outer_wall_enabled else None),
                      ((args.outer_wall_y_min, args.outer_wall_y_max)
                       if outer_wall_enabled else None),
                      args.outer_wall_taper_m,
                      args.outer_wall_setback_m,
                      args.outer_wall_smoothing_m,
                      args.outer_wall_attraction_per_m4,
                      args.straight_x_max,
                      args.preserve_source_geometry,
                      args.straight_target_x,
                      args.outer_wall_side)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
