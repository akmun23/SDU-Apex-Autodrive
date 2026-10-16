#!/usr/bin/env python3
"""Search free practice-track trajectories against native Unity physics.

The map centerline is only the spatial coordinate used to select 40 Hz
commands. It is not a path-tracking target or a constraint on the car's actual
trajectory. Steering and throttle are optimized as smooth periodic control
profiles; the candidate raceline is the resulting 1 kHz Unity trajectory.
Every objective evaluation runs the unchanged source-built Unity player and
its native WheelCollider/PhysX solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_unity_source_raceline import (  # noqa: E402
    FIXED_DT_S, TICKS_PER_COMMAND,
    _finish_gate, _read_capture_xz, _read_json, _trial_score,
)
from optimize_unity_spatial_raceline import (  # noqa: E402
    MIN_TRACK_CLEARANCE_M, PRACTICE_MAP, PRACTICE_MAP_ORIGIN_XY,
    PRACTICE_MAP_RESOLUTION_M, _read_pgm8, _track_clearance_summary,
)
from optimize_unity_sector_raceline import DEFAULT_OPTIMIZER_PLAYER  # noqa: E402


DEFAULT_SEED_PROFILE = REPO / (
    "live_runs/unity_open_loop_replay_r115_20261014/trials/trial_001/"
    "command_profile.json")
DEFAULT_REFERENCE_CAPTURE = REPO / (
    "live_runs/unity_source_raceline_feedback_mintime_20261014_r115/"
    "dynamics/wheel_dynamics_feedback_mintime_20261014_r115.csv")
DEFAULT_CENTERLINE = REPO / (
    "live_runs/racelines/icra_practice_20260924_b/SmoothCenterline.csv")
DEFAULT_OUTPUT = REPO / (
    "live_runs/raceline_candidates/unity_centerline_direct_control_20261015_r01")

# Four broad lap sectors keep the Unity search identifiable with the number of
# real evaluations available. The earlier eight-sector sweep was under-sampled
# by the 16-D GP and proposed extreme controls without a valid lap.
KNOTS = 4
STEERING_NAMES = tuple(f"steering_delta_knot_{i:02d}" for i in range(KNOTS))
THROTTLE_NAMES = tuple(f"throttle_delta_knot_{i:02d}" for i in range(KNOTS))
PARAMETER_NAMES = STEERING_NAMES + THROTTLE_NAMES
LOWER = np.asarray([-0.20] * KNOTS + [-0.18] * KNOTS, dtype=np.float64)
UPPER = np.asarray([0.20] * KNOTS + [0.18] * KNOTS, dtype=np.float64)
KNOT_CENTERS = (np.arange(KNOTS, dtype=np.float64) + 0.5) / KNOTS
TARGET_S = 5.0
INVALID_MAP_SCORE_S = 22.0
UNITY_TO_MAP_OFFSET_XY = (0.24, 0.05)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_seed_trial(seed_path: Path) -> dict[str, Any]:
    """Do not optimize from a profile whose own Unity lap was invalid."""
    trial_path = seed_path.parent / "trial.json"
    if not trial_path.is_file():
        raise ValueError(f"seed has no adjacent Unity trial record: {trial_path}")
    trial = json.loads(trial_path.read_text(encoding="utf-8"))
    clearance = trial.get("track_clearance")
    if not isinstance(clearance, dict):
        clearance = {}
    minimum_clearance = trial.get("track_clearance_min_m")
    if minimum_clearance is None:
        minimum_clearance = clearance.get("track_clearance_min_m")
    if (trial.get("lap_valid") is not True or
            trial.get("track_valid") is not True or
            int(trial.get("collisions_observed", -1)) != 0 or
            minimum_clearance is None or
            not np.isfinite(float(minimum_clearance)) or
            float(minimum_clearance) < MIN_TRACK_CLEARANCE_M):
        raise ValueError(
            "seed must be a complete, collision-free Unity lap with at least "
            f"{MIN_TRACK_CLEARANCE_M:.2f} m map clearance; see {trial_path}")
    return {
        "trial_record": str(trial_path),
        "lap_time_s": float(trial["unity_lap_time_s"]),
        "minimum_map_clearance_m": float(minimum_clearance),
        "collisions_observed": 0,
    }


def _read_centerline(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = ("# s_m", "x_m", "y_m")
    if len(rows) < 100 or any(name not in rows[0] for name in required):
        raise ValueError("centerline needs at least 100 rows with s_m/x_m/y_m")
    station = np.asarray([float(row["# s_m"]) for row in rows])
    xy = np.asarray([[float(row["x_m"]), float(row["y_m"])] for row in rows])
    if (not np.isfinite(station).all() or not np.isfinite(xy).all() or
            np.any(np.diff(station) <= 0.0)):
        raise ValueError("centerline stations or coordinates are invalid")
    closing_length = float(np.linalg.norm(xy[0] - xy[-1]))
    length = float(station[-1] + closing_length)
    if length < 5.0 or closing_length < 1.0e-4:
        raise ValueError("centerline is not a usable closed path")
    return station, xy, length


def _map_from_unity(points_xz: np.ndarray) -> np.ndarray:
    points_xz = np.asarray(points_xz, dtype=np.float64)
    return np.column_stack((points_xz[:, 1] + UNITY_TO_MAP_OFFSET_XY[0],
                            -points_xz[:, 0] + UNITY_TO_MAP_OFFSET_XY[1]))


def _unity_from_map(points_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    return np.column_stack((UNITY_TO_MAP_OFFSET_XY[1] - points_xy[:, 1],
                            points_xy[:, 0] - UNITY_TO_MAP_OFFSET_XY[0]))


def _closed_centerline_for_gate(centerline_path: Path, capture_path: Path,
                                warmup_count: int, route_count: int
                                ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    station, xy, length = _read_centerline(centerline_path)
    spline = CubicSpline(
        np.r_[station, length], np.vstack((xy, xy[0])), bc_type="periodic")
    gate_origin_xz, _, _ = _finish_gate(capture_path, warmup_count)
    gate_map_xy = _map_from_unity(gate_origin_xz[None, :])[0]
    nearest = int(np.argmin(np.linalg.norm(xy - gate_map_xy, axis=1)))
    start_station = float(station[nearest])
    route_progress = np.arange(route_count, dtype=np.float64) / route_count
    query_station = (start_station + route_progress * length) % length
    route_map_xy = np.asarray(spline(query_station), dtype=np.float64)
    route_unity_xz = _unity_from_map(route_map_xy)
    gate_distance = float(np.linalg.norm(xy[nearest] - gate_map_xy))
    if gate_distance > 0.75:
        raise ValueError(
            f"centerline is not aligned with the scored gate ({gate_distance:.3f} m)")
    return route_unity_xz, route_progress, {
        "source": str(centerline_path),
        "source_sha256": _sha256(centerline_path),
        "length_m": length,
        "route_command_count": route_count,
        "start_nearest_centerline_index": nearest,
        "gate_to_centerline_distance_m": gate_distance,
        "map_origin_xy": list(PRACTICE_MAP_ORIGIN_XY),
        "map_resolution_m": PRACTICE_MAP_RESOLUTION_M,
        "reference_only": True,
    }


def _seed_route(seed: dict[str, Any], seed_path: Path
                ) -> tuple[np.ndarray, dict[str, Any]]:
    warmup = int(seed["warmup_command_count"])
    commands = seed["commands"][warmup:]
    if len(commands) < 100 or not all(
            row.get("has_reference_position") is True for row in commands):
        raise ValueError("seed must contain a complete spatial route")
    route = np.asarray([[float(row["reference_x_m"]),
                         float(row["reference_z_m"])] for row in commands],
                       dtype=np.float64)
    if not np.isfinite(route).all():
        raise ValueError("seed route contains non-finite coordinates")
    lengths = np.linalg.norm(np.diff(route, axis=0), axis=1)
    closing = float(np.linalg.norm(route[0] - route[-1]))
    length = float(lengths.sum() + closing)
    if length < 5.0 or np.any(lengths < 1.0e-6) or closing < 1.0e-6:
        raise ValueError("seed route is degenerate or not closed")
    return route, {
        "source": "spatial path in the supplied command profile",
        "source_sha256": _sha256(seed_path),
        "length_m": length,
        "route_command_count": len(route),
        "reference_only": False,
    }


def _route_progress(points_xz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xz, dtype=np.float64)
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("spatial route has too few finite coordinates")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    closing = float(np.linalg.norm(points[0] - points[-1]))
    total = float(lengths.sum() + closing)
    if total < 5.0 or np.any(lengths < 1.0e-6):
        raise ValueError("spatial route is degenerate")
    return np.r_[0.0, np.cumsum(lengths)] / total


def _periodic_knots(values: np.ndarray, progress: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    progress = np.asarray(progress, dtype=np.float64)
    if values.shape != (KNOTS,) or not np.isfinite(values).all():
        raise ValueError(f"expected {KNOTS} finite control knots")
    spline = CubicSpline(
        np.r_[KNOT_CENTERS, KNOT_CENTERS[0] + 1.0],
        np.r_[values, values[0]], bc_type="periodic")
    query = progress.copy()
    query[query < KNOT_CENTERS[0]] += 1.0
    return np.asarray(spline(query), dtype=np.float64)


def _smooth_window_weights(progress: np.ndarray, start: float,
                           end: float) -> np.ndarray:
    """Compact raised-cosine window over normalized route distance."""
    progress = np.asarray(progress, dtype=np.float64)
    if (not 0.0 <= start < end <= 1.0 or not np.isfinite(progress).all()):
        raise ValueError("local route window must be finite with 0 <= start < end <= 1")
    phase = (progress - start) / (end - start)
    weights = np.zeros_like(progress)
    inside = (phase >= 0.0) & (phase <= 1.0)
    fade = min(0.15, 0.5)
    rising = inside & (phase < fade)
    falling = inside & (phase > 1.0 - fade)
    plateau = inside & ~rising & ~falling
    weights[rising] = 0.5 - 0.5 * np.cos(np.pi * phase[rising] / fade)
    weights[falling] = 0.5 - 0.5 * np.cos(
        np.pi * (1.0 - phase[falling]) / fade)
    weights[plateau] = 1.0
    return weights


def _seed_parameters(seed: dict[str, Any], route_unity_xz: np.ndarray
                     ) -> np.ndarray:
    """Zero adjustment means preserve the seed's measured 40 Hz controls."""
    warmup = int(seed["warmup_command_count"])
    commands = seed["commands"][warmup:]
    if len(commands) < 32 or not all(
            row.get("has_reference_position") is True for row in commands):
        raise ValueError("seed must have a spatially indexed scored command lap")
    controls = np.asarray([[float(row["steering_norm"]),
                            float(row["throttle_norm"])] for row in commands])
    if (route_unity_xz.ndim != 2 or route_unity_xz.shape[1] != 2 or
            not np.isfinite(route_unity_xz).all() or
            not np.isfinite(controls).all() or
            np.any(np.abs(controls[:, 0]) > 1.0) or
            np.any((controls[:, 1] < 0.0) | (controls[:, 1] > 1.0))):
        raise ValueError("seed controls contain non-finite values")
    return np.zeros(2 * KNOTS, dtype=np.float64)


def _resample_seed_controls(seed: dict[str, Any], route_unity_xz: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transfer seed commands onto route arc progress, not command index."""
    warmup = int(seed["warmup_command_count"])
    source = seed["commands"][warmup:]
    source_xz = np.asarray([[float(row["reference_x_m"]),
                             float(row["reference_z_m"])] for row in source])
    source_controls = np.asarray([[float(row["steering_norm"]),
                                   float(row["throttle_norm"])]
                                  for row in source], dtype=np.float64)
    if (source_xz.shape == route_unity_xz.shape and
            np.allclose(source_xz, route_unity_xz, rtol=0.0, atol=1.0e-9)):
        return source_controls[:, 0], source_controls[:, 1], {
            "source_commands": len(source),
            "route_points": len(route_unity_xz),
            "transfer": "exact seed route and control samples",
        }
    target_map = _map_from_unity(route_unity_xz)
    source_map = _map_from_unity(source_xz)
    nearest = cKDTree(target_map).query(source_map, k=1)[1].astype(np.int64)
    progress = nearest.astype(np.float64) / len(route_unity_xz)
    for index in range(1, len(progress)):
        while progress[index] < progress[index - 1] - 0.5:
            progress[index] += 1.0
    progress = np.maximum.accumulate(progress)
    if progress[-1] - progress[0] < 0.80:
        raise ValueError("seed command references do not cover most of the lap")
    progress -= progress[0]
    # The original capture has small nearest-waypoint reversals and duplicate
    # spatial samples; average only exact duplicate progress bins, then perform
    # monotonic interpolation at the new uniform route stations.
    unique_progress, inverse = np.unique(progress, return_inverse=True)
    base = []
    for key in ("steering_norm", "throttle_norm"):
        values = np.asarray([float(row[key]) for row in source])
        sums = np.bincount(inverse, weights=values)
        means = sums / np.bincount(inverse)
        query = np.arange(len(route_unity_xz), dtype=np.float64) / len(route_unity_xz)
        base.append(np.interp(query, unique_progress, means))
    controls = np.column_stack(base)
    if not np.isfinite(controls).all():
        raise ValueError("resampled seed controls are non-finite")
    return controls[:, 0], controls[:, 1], {
        "source_commands": len(source),
        "route_points": len(route_unity_xz),
        "source_progress_start": float(progress[0]),
        "source_progress_end": float(progress[-1]),
        "nearest_projection_monotonic": True,
    }


def make_candidate_profile(seed: dict[str, Any], route_unity_xz: np.ndarray,
                           base_controls: np.ndarray,
                           parameters: np.ndarray,
                           zero_throttle_floor: float = 0.0,
                           local_window: tuple[float, float] | None = None,
                           local_offsets: tuple[float, float] = (0.0, 0.0)
                           ) -> dict[str, Any]:
    parameters = np.asarray(parameters, dtype=np.float64)
    if (parameters.shape != (2 * KNOTS,) or not np.isfinite(parameters).all()
            or np.any(parameters < LOWER) or np.any(parameters > UPPER)):
        raise ValueError("control knots are outside their finite bounds")
    if not np.isfinite(zero_throttle_floor) or not 0.0 <= zero_throttle_floor <= 1.0:
        raise ValueError("zero-throttle floor must be in [0, 1]")
    local_offsets = tuple(float(value) for value in local_offsets)
    if len(local_offsets) != 2 or not np.isfinite(local_offsets).all():
        raise ValueError("local steering/throttle offsets must be finite")
    if local_window is None:
        if any(value != 0.0 for value in local_offsets):
            raise ValueError("localized offsets require a route window")
        local_weights = np.zeros(len(route_unity_xz), dtype=np.float64)
    else:
        local_weights = _smooth_window_weights(
            _route_progress(route_unity_xz), *local_window)
    warmup_count = int(seed["warmup_command_count"])
    route_progress = np.arange(len(route_unity_xz), dtype=np.float64) / len(route_unity_xz)
    steering_delta = _periodic_knots(parameters[:KNOTS], route_progress)
    throttle_delta = _periodic_knots(parameters[KNOTS:], route_progress)
    commands = [dict(row) for row in seed["commands"][:warmup_count]]
    base_controls = np.asarray(base_controls, dtype=np.float64)
    if (base_controls.shape != (len(route_unity_xz), 2) or
            not np.isfinite(base_controls).all()):
        raise ValueError("route-resampled seed controls have invalid shape/data")
    for index, point in enumerate(route_unity_xz):
        throttle = float(np.clip(
            float(base_controls[index, 1]) + throttle_delta[index] +
            local_offsets[1] * local_weights[index], 0.0, 1.0))
        if throttle == 0.0 and zero_throttle_floor > 0.0:
            throttle = float(zero_throttle_floor)
        commands.append({
            "steering_norm": float(np.clip(
                float(base_controls[index, 0]) + steering_delta[index] +
                local_offsets[0] * local_weights[index],
                -1.0, 1.0)),
            "throttle_norm": throttle,
            "has_reference_position": True,
            "reference_x_m": float(point[0]),
            "reference_z_m": float(point[1]),
        })
    profile = dict(seed)
    profile["commands"] = commands
    profile["spatial_command_lookup"] = True
    profile["warmup_command_count"] = warmup_count
    profile["scored_horizon_commands"] = int(seed["scored_horizon_commands"])
    profile["raceline_feedback_control"] = False
    profile["zero_throttle_floor"] = float(zero_throttle_floor)
    profile["localized_control_window"] = (
        None if local_window is None else {
            "route_progress_start": float(local_window[0]),
            "route_progress_end": float(local_window[1]),
            "steering_delta": local_offsets[0],
            "throttle_delta": local_offsets[1],
        })
    return profile


def _initial_design(seed_parameters: np.ndarray,
                    focus_knot: int | None = None) -> list[np.ndarray]:
    design = [seed_parameters.copy()]
    if focus_knot is not None:
        if not 0 <= focus_knot < KNOTS:
            raise ValueError(f"focus knot must be in [0, {KNOTS - 1}]")
        steering_index = focus_knot
        throttle_index = KNOTS + focus_knot
        for index, delta in ((steering_index, -0.10),
                             (steering_index, 0.10),
                             (throttle_index, -0.09),
                             (throttle_index, 0.09)):
            point = seed_parameters.copy()
            point[index] = delta
            design.append(point)
        for steering_delta, throttle_delta in (
                (-0.05, -0.05), (-0.05, 0.05),
                (0.05, -0.05), (0.05, 0.05)):
            point = seed_parameters.copy()
            point[steering_index] = steering_delta
            point[throttle_index] = throttle_delta
            design.append(point)
        return design

    for dimension in range(2 * KNOTS):
        delta = 0.10 if dimension < KNOTS else 0.09
        for sign in (-1.0, 1.0):
            point = seed_parameters.copy()
            point[dimension] = np.clip(
                point[dimension] + sign * delta,
                LOWER[dimension], UPPER[dimension])
            if not any(np.allclose(point, old, atol=1.0e-10)
                       for old in design):
                design.append(point)
    # Low-frequency coupled probes move the path/speed policy beyond its seed
    # while keeping the first exact Unity evaluations interpretable.
    for steering_shift, throttle_shift in (
            (0.0, 0.08), (0.0, 0.16), (0.06, 0.08), (-0.06, 0.08)):
        point = seed_parameters.copy()
        point[:KNOTS] = np.clip(
            point[:KNOTS] + steering_shift, LOWER[:KNOTS], UPPER[:KNOTS])
        point[KNOTS:] = np.clip(
            point[KNOTS:] + throttle_shift, LOWER[KNOTS:], UPPER[KNOTS:])
        if not any(np.allclose(point, old, atol=1.0e-10)
                   for old in design):
            design.append(point)
    return design


def _scaled(points: np.ndarray) -> np.ndarray:
    return (points - LOWER) / (UPPER - LOWER)


def _propose(rows: list[dict[str, Any]], seed: int, index: int
             ) -> tuple[np.ndarray, dict[str, Any]]:
    observed = np.asarray([
        [float(row["parameters"][name]) for name in PARAMETER_NAMES]
        for row in rows], dtype=np.float64)
    scores = np.asarray([float(row["objective_s"]) for row in rows])
    if len(rows) < 3 or not np.isfinite(scores).all():
        raise ValueError("Bayesian search needs at least three finite observations")
    kernel = (ConstantKernel(1.0, (0.1, 10.0)) *
              Matern(length_scale=np.full(2 * KNOTS, 0.35),
                     length_scale_bounds=(0.08, 2.0), nu=2.5) +
              WhiteKernel(noise_level=0.02,
                          noise_level_bounds=(1.0e-4, 0.2)))
    model = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
        random_state=seed + index, alpha=1.0e-6)
    model.fit(_scaled(observed), scores)
    candidate_scaled = qmc.Sobol(
        d=2 * KNOTS, scramble=True, seed=seed + index
    ).random_base2(m=13)
    mean, std = model.predict(candidate_scaled, return_std=True)
    improvement = float(np.min(scores)) - mean - 0.005
    z = improvement / np.maximum(std, 1.0e-9)
    acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
    for point in _scaled(observed):
        acquisition[np.linalg.norm(candidate_scaled - point, axis=1) < 0.04] = 0.0
    chosen = int(np.argmax(acquisition))
    return LOWER + candidate_scaled[chosen] * (UPPER - LOWER), {
        "method": "exact_unity_constrained_expected_improvement",
        "expected_improvement_s": float(acquisition[chosen]),
        "observation_count": len(rows),
        "kernel": str(model.kernel_),
    }


def _score_trial(run_dir: Path, parameters: np.ndarray,
                 gate_origin: np.ndarray, gate_direction: np.ndarray,
                 warmup_count: int, route_unity_xz: np.ndarray,
                 player_dir: Path, timeout_s: float,
                 map_clearance: np.ndarray) -> dict[str, Any]:
    result = _trial_score(
        run_dir, parameters, gate_origin, gate_direction, warmup_count,
        route_unity_xz, player_dir, timeout_s,
        parameter_names=PARAMETER_NAMES)
    capture_path = Path(result["capture_path"]) if result.get("capture_path") else None
    clearance_summary: dict[str, Any] = {}
    track_valid = False
    if capture_path is not None and result.get("unity_lap_time_s") is not None:
        crossings = result.get("forward_gate_crossings_s", [])
        expected_start = warmup_count * TICKS_PER_COMMAND * FIXED_DT_S
        possible_starts = [time_s for time_s in crossings
                           if abs(float(time_s) - expected_start) <= 0.25]
        if possible_starts:
            lap_start = min(possible_starts,
                            key=lambda time_s: abs(float(time_s) - expected_start))
            lap_start = float(lap_start)
            lap_stop = lap_start + float(result["unity_lap_time_s"])
            clearance_summary = _track_clearance_summary(
                capture_path, lap_start, lap_stop, map_clearance,
                route_unity_xz)
            track_valid = (
                float(clearance_summary["track_clearance_min_m"]) >=
                MIN_TRACK_CLEARANCE_M)
    result.update(clearance_summary)
    result["track_valid"] = bool(track_valid)
    result["lap_valid"] = bool(result.get("lap_valid") and track_valid)
    result["objective_s"] = _candidate_objective(result)
    return result


def _candidate_objective(result: dict[str, Any]) -> float:
    """Keep time and map-margin signal for complete, but invalid, Unity laps."""
    if result["lap_valid"]:
        return float(result["unity_lap_time_s"])
    progress = float(result.get("reference_lap_progress_fraction", 0.0))
    lap_time = result.get("unity_lap_time_s")
    margin_fraction = result.get("track_samples_with_vehicle_margin_fraction")
    if lap_time is not None and margin_fraction is not None:
        clearance = float(result.get("track_clearance_min_m") or 0.0)
        return (
            INVALID_MAP_SCORE_S + 0.1 * float(lap_time) +
            3.0 * max(0.0, MIN_TRACK_CLEARANCE_M - clearance) +
            2.0 * (1.0 - float(margin_fraction)))
    return INVALID_MAP_SCORE_S + 10.0 + 10.0 * (1.0 - progress)


def _retain_compact_capture_set(rows: list[dict[str, Any]],
                                output: Path) -> None:
    """Keep baseline, fastest valid/invalid laps and furthest-progress capture."""
    valid = [(index, row) for index, row in enumerate(rows)
             if row.get("lap_valid") and row.get("unity_lap_time_s") is not None]
    best_index = min(
        valid, key=lambda pair: float(pair[1]["unity_lap_time_s"]),
        default=(0, rows[0] if rows else {}))[0]
    invalid = [(index, row) for index, row in enumerate(rows)
               if not row.get("lap_valid")]
    diagnostic_index = max(
        invalid,
        key=lambda pair: float(pair[1].get("reference_lap_progress_fraction", 0.0)),
        default=(-1, {}))[0]
    completed_invalid = [(index, row) for index, row in invalid
                         if row.get("unity_lap_time_s") is not None]
    fastest_invalid_index = min(
        completed_invalid,
        key=lambda pair: float(pair[1]["unity_lap_time_s"]),
        default=(-1, {}))[0]
    keep_indices = {0, best_index, diagnostic_index, fastest_invalid_index}
    for index, row in enumerate(rows):
        raw_path = row.get("capture_path")
        if not raw_path:
            row["dynamics_capture_retained"] = False
            continue
        capture = Path(raw_path)
        retain = index in keep_indices
        if not retain and capture.is_file() and capture.is_relative_to(output):
            capture.unlink()
            row["capture_path"] = None
        row["dynamics_capture_retained"] = bool(retain and capture.is_file())
        trial_path = output / "trials" / str(row["trial"]) / "trial.json"
        if trial_path.is_file():
            trial_path.write_text(
                json.dumps(row, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    inputs = (args.seed_profile.resolve(), args.reference_capture.resolve(),
              args.centerline.resolve(), PRACTICE_MAP)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    seed_validation = _validate_seed_trial(inputs[0])
    seed_profile = _read_json(inputs[0])
    warmup_count = int(seed_profile["warmup_command_count"])
    if args.route_source == "seed":
        route_unity_xz, centerline_info = _seed_route(seed_profile, inputs[0])
    else:
        route_count = int(seed_profile.get("scored_horizon_commands", 0))
        if route_count < 100:
            raise ValueError("seed profile does not contain a complete scored route")
        route_unity_xz, _, centerline_info = _closed_centerline_for_gate(
            inputs[2], inputs[1], warmup_count, route_count)
    if (not seed_profile.get("spatial_command_lookup") or
            int(seed_profile.get("physics_ticks_per_command", -1)) != 25 or
            abs(float(seed_profile.get("fixed_delta_time_s", 0.0)) - 0.001) > 1.0e-8):
        raise ValueError("seed must use the verified 1 kHz / 40 Hz spatial runner")
    if not 0.0 <= args.zero_throttle_floor <= 1.0:
        raise ValueError("zero-throttle floor must be in [0, 1]")
    local_offsets = (float(args.local_steering_delta),
                     float(args.local_throttle_delta))
    if args.local_window is None and any(value != 0.0 for value in local_offsets):
        raise ValueError("localized control deltas require --local-window")
    if any(value != 0.0 for value in local_offsets) and args.max_evaluations != 1:
        raise ValueError("a fixed localized correction is one Unity evaluation")
    if args.local_window is not None:
        _smooth_window_weights(np.asarray([args.local_window[0], args.local_window[1]]),
                                *args.local_window)
    player_dir = args.player_dir.resolve()
    if not (player_dir / "AutoDRIVE Simulator.x86_64").is_file():
        raise FileNotFoundError(f"source Unity player missing: {player_dir}")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite campaign: {args.output}")
    if not 1 <= args.max_evaluations <= 1000 or args.timeout_s <= 0.0:
        raise ValueError("max evaluations or timeout is outside supported bounds")

    gate_origin, gate_direction, _ = _finish_gate(inputs[1], warmup_count)
    map_pixels = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(map_pixels == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M
    seed_parameters = _seed_parameters(seed_profile, route_unity_xz)
    base_steering, base_throttle, control_transfer = _resample_seed_controls(
        seed_profile, route_unity_xz)
    base_controls = np.column_stack((base_steering, base_throttle))
    if args.focus_knot is not None and not 0 <= args.focus_knot < KNOTS:
        raise ValueError(f"focus knot must be in [0, {KNOTS - 1}]")
    initial_design = _initial_design(seed_parameters, args.focus_knot)
    output = args.output.resolve()
    seed_hash = _sha256(inputs[0])
    if args.resume:
        if not output.is_dir() or not (output / "campaign.json").is_file():
            raise FileNotFoundError("cannot resume a missing route-control campaign")
        manifest = _read_json(output / "campaign.json")
        expected_route_hash = (seed_hash if args.route_source == "seed"
                               else _sha256(inputs[2]))
        if (manifest.get("seed_profile_sha256") != seed_hash or
                manifest.get("reference_capture_sha256") != _sha256(inputs[1]) or
                manifest.get("centerline", {}).get("source_sha256") != expected_route_hash or
                manifest.get("route_source", "centerline") != args.route_source or
                manifest.get("focus_knot") != args.focus_knot or
                float(manifest.get("zero_throttle_floor", 0.0)) !=
                float(args.zero_throttle_floor) or
                manifest.get("local_window") != args.local_window or
                manifest.get("local_offsets", [0.0, 0.0]) != list(local_offsets) or
                manifest.get("player_gameassembly_sha256") !=
                _sha256(player_dir / "GameAssembly.so") or
                manifest.get("map_sha256") != _sha256(PRACTICE_MAP)):
            raise ValueError("campaign inputs differ from the saved search")
        rows = [json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((output / "trials").glob("trial_*/trial.json"))]
        for row in rows:
            row["objective_s"] = _candidate_objective(row)
        if len(rows) != int(manifest.get("completed_evaluations", -1)):
            raise ValueError("campaign manifest and saved trial count disagree")
    else:
        output.mkdir(parents=True)
        (output / "trials").mkdir()
        rows = []
        manifest = {
            "status": "running",
            "optimizer": (
                f"Bayesian optimization over {KNOTS}-knot additive corrections "
                "to the exact seed's 40 Hz steering and throttle stream"),
            "physics_evaluator": "unchanged source Unity player; native WheelCollider/PhysX at 1 kHz",
            "control_period_s": 0.025,
            "physics_step_s": 0.001,
            "objective": (
                "Unity gate-to-gate lap time; valid laps require zero collision "
                "and map-derived vehicle-center clearance >= 0.20 m. Incomplete "
                "trials receive a progress penalty; completed but off-margin laps "
                "retain time and clearance-fraction ranking for Bayesian learning."),
            "centerline": centerline_info,
            "route_source": args.route_source,
            "focus_knot": args.focus_knot,
            "zero_throttle_floor": float(args.zero_throttle_floor),
            "local_window": (None if args.local_window is None
                             else list(args.local_window)),
            "local_offsets": list(local_offsets),
            "route_role": (
                "candidate path seed; body motion remains free in Unity"
                if args.route_source == "seed" else
                "spatial command index only; not a tracked or optimized-path constraint"),
            "actual_candidate_geometry": (
                "the 1 kHz Unity body trajectory generated by each steering profile; "
                "path seed coordinates are used only for spatial command selection"),
            "seed_profile": str(inputs[0]),
            "seed_trial_validation": seed_validation,
            "seed_profile_sha256": _sha256(inputs[0]),
            "seed_reference_capture": str(inputs[1]),
            "reference_capture_sha256": _sha256(inputs[1]),
            "player": str(player_dir),
            "player_gameassembly_sha256": _sha256(player_dir / "GameAssembly.so"),
            "map": str(PRACTICE_MAP),
            "map_sha256": _sha256(PRACTICE_MAP),
            "parameters": {
                "names": list(PARAMETER_NAMES),
                "lower": LOWER.tolist(),
                "upper": UPPER.tolist(),
                "baseline_adjustments": dict(zip(PARAMETER_NAMES, seed_parameters.tolist())),
            },
            "seed_control_transfer": control_transfer,
            "reference_trajectory_changed": False,
            "runtime_physics_changed": False,
            "evaluations": [],
        }
    manifest["objective"] = (
        "Unity gate-to-gate lap time; valid laps require zero collision and "
        "map-derived vehicle-center clearance >= 0.20 m. Incomplete trials "
        "receive a progress penalty; completed but off-margin laps retain "
        "time and clearance-fraction ranking for Bayesian learning.")
    manifest["status"] = "running"
    if not args.resume:
        (output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    while len(rows) < args.max_evaluations:
        index = len(rows)
        if index < len(initial_design):
            parameters = initial_design[index]
            proposal: dict[str, Any] = {
                "method": ("focused_steering_throttle_pair_design"
                           if args.focus_knot is not None else
                           "one_factor_at_a_time_route_control_design"),
                "design_index": index,
                "design_size": len(initial_design),
            }
        else:
            parameters, proposal = _propose(rows, args.seed, index)
        trial_dir = output / "trials" / f"trial_{index:03d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        profile = make_candidate_profile(
            seed_profile, route_unity_xz, base_controls, parameters,
            args.zero_throttle_floor, args.local_window, local_offsets)
        (trial_dir / "command_profile.json").write_text(
            json.dumps(profile, separators=(",", ":")) + "\n",
            encoding="utf-8")
        result = _score_trial(
            trial_dir, parameters, gate_origin, gate_direction, warmup_count,
            route_unity_xz, player_dir, args.timeout_s, map_clearance)
        result["trial"] = trial_dir.name
        result["parameters"] = dict(zip(PARAMETER_NAMES, parameters.tolist()))
        if args.local_window is not None:
            result["localized_control_window"] = {
                "route_progress_start": float(args.local_window[0]),
                "route_progress_end": float(args.local_window[1]),
                "steering_delta": local_offsets[0],
                "throttle_delta": local_offsets[1],
            }
        result["proposal"] = proposal
        (trial_dir / "trial.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(result)
        _retain_compact_capture_set(rows, output)
        valid_times = [float(row["unity_lap_time_s"]) for row in rows
                       if row.get("lap_valid")]
        best_time = min(valid_times, default=None)
        manifest["completed_evaluations"] = len(rows)
        manifest["best_valid_lap_time_s"] = best_time
        manifest["best_valid_trial"] = min(
            (row for row in rows if row.get("lap_valid")),
            key=lambda row: float(row["unity_lap_time_s"]), default=None)
        manifest["evaluations"] = [{
            "trial": row["trial"],
            "parameters": row["parameters"],
            "proposal": row["proposal"],
            "unity_status": row.get("unity_status"),
            "collisions_observed": row.get("collisions_observed"),
            "lap_valid": row.get("lap_valid"),
            "lap_time_s": row.get("unity_lap_time_s"),
            "objective_s": row.get("objective_s"),
            "progress": row.get("reference_lap_progress_fraction"),
            "track_clearance_min_m": row.get("track_clearance_min_m"),
            "peak_body_speed_mps": row.get("peak_body_speed_mps"),
            "p95_abs_wheel_slip": row.get("p95_abs_wheel_slip"),
            "dynamics_capture_retained": row.get("dynamics_capture_retained"),
            "capture_path": row.get("capture_path"),
        } for row in rows]
        manifest["status"] = (
            "target_reached" if best_time is not None and best_time < args.target_s
            else "running")
        (output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "trial": trial_dir.name,
            "parameters": result["parameters"],
            "status": result.get("unity_status"),
            "collision_count": result.get("collisions_observed"),
            "lap_valid": result.get("lap_valid"),
            "lap_time_s": result.get("unity_lap_time_s"),
            "progress": result.get("reference_lap_progress_fraction"),
            "clearance_min_m": result.get("track_clearance_min_m"),
            "peak_speed_mps": result.get("peak_body_speed_mps"),
            "best_so_far_s": best_time,
            "wall_time_s": result.get("wall_time_s"),
        }, separators=(",", ":")), flush=True)
        if manifest["status"] == "target_reached":
            break

    if manifest.get("status") != "target_reached":
        manifest["status"] = (
            "budget_exhausted_without_valid_lap"
            if manifest.get("best_valid_lap_time_s") is None
            else "budget_exhausted_above_target")
        (output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if manifest.get("status") == "target_reached" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-profile", type=Path, default=DEFAULT_SEED_PROFILE)
    parser.add_argument("--reference-capture", type=Path,
                        default=DEFAULT_REFERENCE_CAPTURE)
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--route-source", choices=("centerline", "seed"),
                        default="centerline",
                        help="Use the map centerline or the seed profile's spatial path.")
    parser.add_argument(
        "--zero-throttle-floor", type=float, default=0.0,
        help=("Replace exact-zero scored throttle commands with this positive "
              "value; warmup remains unchanged."))
    parser.add_argument(
        "--focus-knot", type=int, default=None,
        help=("Run a compact local steering/throttle grid at one route knot; "
              "useful when Unity traces localize the remaining error."))
    parser.add_argument(
        "--local-window", type=float, nargs=2, metavar=("START", "END"),
        default=None, help="Apply fixed control deltas only over this normalized route-distance window.")
    parser.add_argument("--local-steering-delta", type=float, default=0.0)
    parser.add_argument("--local-throttle-delta", type=float, default=0.0)
    parser.add_argument("--player-dir", type=Path,
                        default=DEFAULT_OPTIMIZER_PLAYER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-evaluations", type=int, default=64)
    parser.add_argument("--target-s", type=float, default=TARGET_S)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_s < 20.0:
        parser.error("target lap time must be positive")
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
