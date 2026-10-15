#!/usr/bin/env python3
"""Jointly search two turn geometries and their speed plans in source Unity.

The unmodified Unity WheelCollider plant is the objective evaluator. The
search changes only the development raceline profile: outward turn offsets in
S6/S12 and smooth target-speed scales across the coupled turn/exit pairs
S6-S7 and S11-S12. A lap is accepted only when Unity completes it without a
collision and the captured trajectory retains the map-derived vehicle margin.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from evaluate_unity_geometry_ablation import (
    DEFAULT_PLAYER,
    MIN_CLEARANCE_M,
    _evaluate,
    _read_profile,
    _sha256,
)
from build_unity_raceline_feedback_profile import smooth_closed_curvature
from optimize_unity_spatial_raceline import (
    PRACTICE_MAP,
    PRACTICE_MAP_ORIGIN_XY,
    PRACTICE_MAP_RESOLUTION_M,
    REPO,
    _read_pgm8,
)


PARAMETER_NAMES = (
    "outward_offset_s6_m",
    "outward_offset_s12_m",
    "target_speed_scale_s6_s7",
    "target_speed_scale_s11_s12",
)
LOWER = np.asarray((0.0, 0.0, 1.0, 1.0), dtype=np.float64)
UPPER = np.asarray((0.12, 0.12, 1.10, 1.10), dtype=np.float64)
INITIAL_DESIGN = np.asarray((
    (0.0, 0.0, 1.0, 1.0),
    (0.08, 0.0, 1.04, 1.0),
    (0.0, 0.08, 1.0, 1.04),
    (0.08, 0.08, 1.04, 1.04),
    (0.12, 0.0, 1.08, 1.0),
    (0.0, 0.12, 1.0, 1.08),
    (0.12, 0.12, 1.08, 1.08),
    (0.04, 0.04, 1.02, 1.02),
), dtype=np.float64)
INVALID_SCORE_S = 12.0
DEFAULT_SEED_PROFILE = REPO / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02/"
    "raceline_feedback_profile.json")
DEFAULT_BASELINE_RUN = REPO / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02")


def _periodic_distance(progress: np.ndarray, center: float) -> np.ndarray:
    distance = np.abs(progress - center)
    return np.minimum(distance, 1.0 - distance)


def _sector_window(progress: np.ndarray, sectors: tuple[int, ...],
                   radius_sectors: float) -> np.ndarray:
    result = np.zeros_like(progress)
    for sector in sectors:
        center = (sector - 0.5) / 12.0
        distance = _periodic_distance(progress, center) * 12.0
        weight = np.where(
            distance <= radius_sectors,
            0.5 * (1.0 + np.cos(np.pi * distance / radius_sectors)), 0.0)
        result = np.maximum(result, weight)
    return result


def _map_clearance(points: np.ndarray, map_clearance: np.ndarray) -> np.ndarray:
    map_x = points[:, 1] + 0.24
    map_y = -points[:, 0] + 0.05
    col = np.floor((map_x - PRACTICE_MAP_ORIGIN_XY[0]) /
                   PRACTICE_MAP_RESOLUTION_M).astype(int)
    row_from_bottom = np.floor((map_y - PRACTICE_MAP_ORIGIN_XY[1]) /
                               PRACTICE_MAP_RESOLUTION_M).astype(int)
    row = map_clearance.shape[0] - 1 - row_from_bottom
    inside = ((row >= 0) & (row < map_clearance.shape[0]) &
              (col >= 0) & (col < map_clearance.shape[1]))
    clearance = np.zeros(len(points), dtype=np.float64)
    clearance[inside] = map_clearance[row[inside], col[inside]]
    return clearance


def make_candidate(seed: dict[str, Any], parameters: np.ndarray
                   ) -> tuple[dict[str, Any], dict[str, float]]:
    if parameters.shape != (len(PARAMETER_NAMES),) or not np.isfinite(parameters).all():
        raise ValueError("joint path/speed parameter vector is invalid")
    if np.any(parameters < LOWER) or np.any(parameters > UPPER):
        raise ValueError("joint path/speed parameter is outside search bounds")

    profile = copy.deepcopy(seed)
    warmup = int(profile["warmup_command_count"])
    route = profile["commands"][warmup:]
    points = np.asarray([[row["reference_x_m"], row["reference_z_m"]]
                         for row in route], dtype=np.float64)
    if points.shape[0] < 100 or not np.isfinite(points).all():
        raise ValueError("seed route is incomplete or non-finite")

    progress = np.arange(len(points), dtype=np.float64) / len(points)
    curvature, _ = smooth_closed_curvature(points, 0.30)
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1.0e-9)
    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    outward = -np.tanh(curvature / 0.15)[:, None] * left_normal
    offset6 = _sector_window(progress, (6,), 1.5)
    offset12 = _sector_window(progress, (12,), 1.5)
    candidate_points = points + (
        parameters[0] * offset6[:, None] +
        parameters[1] * offset12[:, None]) * outward

    lengths = np.linalg.norm(
        np.roll(candidate_points, -1, axis=0) - candidate_points, axis=1)
    if np.any(lengths <= 1.0e-6):
        raise ValueError("candidate path contains duplicate adjacent points")
    route_arc = np.r_[0.0, np.cumsum(lengths[:-1])]
    speed6 = _sector_window(progress, (6, 7), 1.0)
    speed12 = _sector_window(progress, (11, 12), 1.0)
    for index, row in enumerate(route):
        row["reference_x_m"] = float(candidate_points[index, 0])
        row["reference_z_m"] = float(candidate_points[index, 1])
        row["route_arc_m"] = float(route_arc[index])
        row["target_speed_mps"] = float(
            row["target_speed_mps"] *
            (1.0 + (parameters[2] - 1.0) * speed6[index]) *
            (1.0 + (parameters[3] - 1.0) * speed12[index]))
    profile["raceline_path_length_m"] = float(np.sum(lengths))
    profile["joint_path_speed_search"] = {
        "method": "native_1khz_unity_plant_bayesian_search",
        "parameters": dict(zip(PARAMETER_NAMES, map(float, parameters))),
        "path_curvature_used_for_speed_retiming": False,
        "geometry_sectors_1_based": [6, 12],
        "speed_groups_1_based": [[6, 7], [11, 12]],
    }
    return profile, {
        "path_length_m": float(np.sum(lengths)),
        "offset_s6_m": float(parameters[0]),
        "offset_s12_m": float(parameters[1]),
        "speed_scale_s6_s7": float(parameters[2]),
        "speed_scale_s11_s12": float(parameters[3]),
    }


def _score(parameters: np.ndarray, row: dict[str, Any] | None = None) -> float:
    if row is None:
        return INVALID_SCORE_S
    if row.get("valid_lap") and row.get("lap_time_s") is not None:
        return float(row["lap_time_s"])
    clearance = float(row.get("track_clearance_min_m") or 0.0)
    return INVALID_SCORE_S + max(0.0, MIN_CLEARANCE_M - clearance)


def _propose(parameters: np.ndarray, scores: np.ndarray,
             seed: int, index: int) -> tuple[np.ndarray, dict[str, Any]]:
    scaled = (parameters - LOWER) / (UPPER - LOWER)
    kernel = (ConstantKernel(1.0, (0.1, 10.0))
              * Matern(length_scale=np.full(4, 0.4),
                       length_scale_bounds=(0.08, 2.0), nu=2.5)
              + WhiteKernel(noise_level=1.0e-5,
                            noise_level_bounds=(1.0e-7, 0.05)))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
        random_state=seed + index, alpha=1.0e-8)
    gp.fit(scaled, scores)
    candidates = qmc.Sobol(d=4, scramble=True,
                           seed=seed + index).random_base2(12)
    mean, std = gp.predict(candidates, return_std=True)
    improvement = float(np.min(scores)) - mean - 0.005
    z = improvement / np.maximum(std, 1.0e-9)
    acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
    for point in scaled:
        acquisition[np.linalg.norm(candidates - point, axis=1) < 0.035] = -np.inf
    chosen = int(np.argmax(acquisition))
    return LOWER + candidates[chosen] * (UPPER - LOWER), {
        "method": "4d_gaussian_process_expected_improvement",
        "expected_improvement_s": float(acquisition[chosen]),
        "observation_count": int(len(scores)),
        "kernel": str(gp.kernel_),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_path = args.seed_profile.resolve()
    baseline_dir = args.baseline_run.resolve()
    player = args.player.resolve()
    output = args.output.resolve()
    baseline_result_path = baseline_dir / "dynamics/offline_result.json"
    for path in (seed_path, PRACTICE_MAP, baseline_result_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (player / "GameAssembly.so").is_file():
        raise FileNotFoundError(f"Unity player is missing: {player}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.evaluations < 2 or args.timeout_s <= 0.0:
        raise ValueError("evaluations and timeout must be positive")

    seed_profile = _read_profile(seed_path)
    baseline_result = json.loads(baseline_result_path.read_text(encoding="utf-8"))
    if baseline_result.get("status") != "complete" or not baseline_result.get("raceline_lap_completed"):
        raise ValueError("seed run is not a completed Unity lap")
    map_pixels = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(map_pixels == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M
    seed_hash = _sha256(seed_path)
    player_hash = _sha256(player / "GameAssembly.so")
    output.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "status": "running",
        "objective": "same source Unity 1 kHz plant lap time subject to completion, zero collisions, and >=0.20 m map margin",
        "search_space": {
            name: [float(LOWER[i]), float(UPPER[i])]
            for i, name in enumerate(PARAMETER_NAMES)},
        "parameter_names": list(PARAMETER_NAMES),
        "initial_design": INITIAL_DESIGN.tolist(),
        "seed_profile": str(seed_path),
        "seed_profile_sha256": seed_hash,
        "same_player_baseline": str(baseline_dir),
        "baseline_lap_time_s": float(baseline_result["scored_duration_s"]),
        "player": str(player),
        "player_gameassembly_sha256": player_hash,
        "physics_changed": False,
        "reference_trajectory_changed": False,
        "evaluations": [],
    }
    (output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    params_rows = [INITIAL_DESIGN[0].copy()]
    score_rows = [float(baseline_result["scored_duration_s"])]
    result_rows: list[dict[str, Any]] = [{
        "evaluation": 0,
        "parameters": dict(zip(PARAMETER_NAMES, map(float, INITIAL_DESIGN[0]))),
        "method": "verified_seed_baseline",
        "lap_time_s": score_rows[0],
        "valid_lap": True,
        "track_clearance_min_m": None,
        "run_dir": str(baseline_dir),
    }]

    for index in range(1, args.evaluations):
        if index < len(INITIAL_DESIGN):
            parameters = INITIAL_DESIGN[index].copy()
            proposal = {"method": "physical_factorial_initial_design"}
        else:
            parameters, proposal = _propose(
                np.asarray(params_rows), np.asarray(score_rows), args.seed, index)
        candidate, geometry = make_candidate(seed_profile, parameters)
        target_clearance = _map_clearance(
            np.asarray([[command["reference_x_m"], command["reference_z_m"]]
                        for command in candidate["commands"][
                            candidate["warmup_command_count"]:]],
                       dtype=np.float64), map_clearance)
        preflight = {
            **geometry,
            "target_path_clearance_min_m": float(target_clearance.min()),
            "target_path_clearance_p05_m": float(np.quantile(target_clearance, 0.05)),
        }
        if preflight["target_path_clearance_min_m"] < MIN_CLEARANCE_M:
            row: dict[str, Any] = {
                "candidate": f"trial_{index:03d}",
                "valid_lap": False,
                "preflight_rejected": True,
                "preflight": preflight,
                "lap_time_s": None,
                "track_clearance_min_m": preflight["target_path_clearance_min_m"],
            }
        else:
            row = _evaluate(
                f"trial_{index:03d}", candidate, str(seed_path), player,
                output, map_clearance, args.timeout_s)
            row["preflight"] = preflight
        score = _score(parameters, row)
        summary = {
            "evaluation": index,
            "parameters": dict(zip(PARAMETER_NAMES, map(float, parameters))),
            "proposal": proposal,
            "lap_time_s": row.get("lap_time_s"),
            "objective_s": score,
            "valid_lap": bool(row.get("valid_lap")),
            "unity_status": row.get("unity_status", "preflight_rejected"),
            "collisions_observed": row.get("collisions_observed", 0),
            "track_clearance_min_m": row.get("track_clearance_min_m"),
            "track_samples_with_vehicle_margin_fraction": row.get(
                "track_samples_with_vehicle_margin_fraction"),
            "tracking_rms_m": row.get("tracking_rms_m"),
            "peak_speed_mps": row.get("peak_speed_mps"),
            "capture": row.get("capture"),
            "preflight": preflight,
        }
        result_rows.append(summary)
        params_rows.append(parameters)
        score_rows.append(score)
        manifest["evaluations"] = result_rows
        valid_times = [float(item["lap_time_s"]) for item in result_rows
                       if item.get("valid_lap") and item.get("lap_time_s") is not None]
        manifest["best_valid_lap_s"] = min(valid_times)
        manifest["status"] = "running"
        (output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, separators=(",", ":")), flush=True)

    manifest["evaluations"] = result_rows
    manifest["status"] = "complete"
    valid = [item for item in result_rows
             if item.get("valid_lap") and item.get("lap_time_s") is not None]
    manifest["best_valid"] = min(valid, key=lambda item: float(item["lap_time_s"]))
    (output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-profile", type=Path, default=DEFAULT_SEED_PROFILE)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--player", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluations", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.evaluations < 2 or args.timeout_s <= 0.0:
        parser.error("evaluations must be >=2 and timeout must be positive")
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
