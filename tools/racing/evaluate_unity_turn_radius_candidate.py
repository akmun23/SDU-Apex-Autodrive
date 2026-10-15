#!/usr/bin/env python3
"""Evaluate a curvature-reducing path and matched speed plan in source Unity.

The two slow corners are displaced toward the outside of their measured
curvature. Their speed targets are retimed from the exact kinematic relation
ay = v^2 * |curvature|, preserving the seed's lateral-acceleration demand.
The modified route is pre-screened against the practice occupancy map, then
scored by the unchanged 1 kHz Unity plant in the established batch launcher.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

from build_unity_raceline_feedback_profile import smooth_closed_curvature
from evaluate_unity_geometry_ablation import (
    DEFAULT_PLAYER,
    MIN_CLEARANCE_M,
    _evaluate,
    _read_profile,
    _sha256,
)
from optimize_unity_spatial_raceline import (
    PRACTICE_MAP,
    PRACTICE_MAP_ORIGIN_XY,
    PRACTICE_MAP_RESOLUTION_M,
    REPO,
    _read_pgm8,
)


DEFAULT_SEED = REPO / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02/"
    "raceline_feedback_profile.json")
DEFAULT_BASELINE_RUN = REPO / (
    "live_runs/unity_geometry_candidate_validation_20261014_r01/"
    "combined_candidate")


def _map_clearance(points_world_xz: np.ndarray, map_clearance: np.ndarray
                   ) -> np.ndarray:
    map_x = points_world_xz[:, 1] + 0.24
    map_y = -points_world_xz[:, 0] + 0.05
    col = np.floor((map_x - PRACTICE_MAP_ORIGIN_XY[0]) /
                   PRACTICE_MAP_RESOLUTION_M).astype(int)
    row_from_bottom = np.floor((map_y - PRACTICE_MAP_ORIGIN_XY[1]) /
                               PRACTICE_MAP_RESOLUTION_M).astype(int)
    row = map_clearance.shape[0] - 1 - row_from_bottom
    inside = ((row >= 0) & (row < map_clearance.shape[0]) &
              (col >= 0) & (col < map_clearance.shape[1]))
    clearance = np.zeros(len(row), dtype=np.float64)
    clearance[inside] = map_clearance[row[inside], col[inside]]
    return clearance


def make_candidate(seed: dict[str, Any], amplitude_m: float,
                   curvature_sigma_m: float,
                   sectors: tuple[int, ...] = (6, 11)
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = copy.deepcopy(seed)
    warmup = int(profile["warmup_command_count"])
    commands = profile["commands"]
    route = commands[warmup:]
    points = np.asarray([[row["reference_x_m"], row["reference_z_m"]]
                         for row in route], dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 100 or not np.isfinite(points).all():
        raise ValueError("seed raceline is incomplete or invalid")
    curvature, _ = smooth_closed_curvature(points, curvature_sigma_m)
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1.0e-9)
    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    if not sectors or any(sector < 1 or sector > 12 for sector in sectors):
        raise ValueError("sectors must be 1-based values in [1, 12]")
    route_sector = np.arange(len(points), dtype=np.float64) * 12.0 / len(points)
    window = np.zeros(len(points), dtype=np.float64)
    for sector in sectors:
        distance = np.abs(route_sector - (sector - 0.5))
        distance = np.minimum(distance, 12.0 - distance)
        window = np.maximum(window, np.where(
            distance <= 1.5,
            0.5 * (1.0 + np.cos(np.pi * distance / 1.5)), 0.0))
    # Signed curvature points toward the turn center. Move in the opposite
    # normal direction, with a smooth zero-crossing for near-straight samples.
    outward_sign = np.tanh(curvature / 0.15)
    candidate_points = points - (amplitude_m * window * outward_sign)[:, None] * left_normal
    candidate_curvature, _ = smooth_closed_curvature(
        candidate_points, curvature_sigma_m)
    base_abs = np.abs(curvature)
    candidate_abs = np.abs(candidate_curvature)
    speed_ratio = np.ones(len(points), dtype=np.float64)
    supported = base_abs >= 0.05
    speed_ratio[supported] = np.sqrt(
        base_abs[supported] / np.maximum(candidate_abs[supported], 0.05))

    segment_lengths = np.linalg.norm(
        np.roll(candidate_points, -1, axis=0) - candidate_points, axis=1)
    route_arc = np.r_[0.0, np.cumsum(segment_lengths[:-1])]
    for index, row in enumerate(route):
        row["reference_x_m"] = float(candidate_points[index, 0])
        row["reference_z_m"] = float(candidate_points[index, 1])
        row["route_arc_m"] = float(route_arc[index])
        row["target_speed_mps"] = float(row["target_speed_mps"] * speed_ratio[index])
    profile["raceline_path_length_m"] = float(segment_lengths.sum())
    profile["turn_radius_candidate"] = {
        "method": "outward offset in S6/S11 with cosine shoulders",
        "amplitude_m": amplitude_m,
        "curvature_smoothing_sigma_m": curvature_sigma_m,
        "speed_retiming": "seed_target * sqrt(abs(kappa_seed)/abs(kappa_candidate))",
        "affected_sectors_1_based": list(sectors),
    }
    preflight = {
        "seed_path_length_m": float(np.linalg.norm(
            np.roll(points, -1, axis=0) - points, axis=1).sum()),
        "candidate_path_length_m": float(segment_lengths.sum()),
        "seed_abs_curvature_p95_per_m": float(np.quantile(base_abs, 0.95)),
        "candidate_abs_curvature_p95_per_m": float(
            np.quantile(candidate_abs, 0.95)),
        "speed_ratio_min": float(speed_ratio.min()),
        "speed_ratio_max": float(speed_ratio.max()),
    }
    return profile, {"points_world_xz": candidate_points, **preflight}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-profile", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--player", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amplitude-m", type=float, default=0.05)
    parser.add_argument("--curvature-sigma-m", type=float, default=0.30)
    parser.add_argument("--sectors", type=int, nargs="+", default=(6, 11),
                        help="1-based turn sectors to move outward")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    seed_path = args.seed_profile.resolve()
    baseline_dir = args.baseline_run.resolve()
    player = args.player.resolve()
    output = args.output.resolve()
    for path in (seed_path, PRACTICE_MAP):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (baseline_dir / "dynamics/offline_result.json").is_file():
        raise FileNotFoundError("same-player baseline result is missing")
    if not (player / "GameAssembly.so").is_file():
        raise FileNotFoundError(f"Unity player is missing: {player}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate run: {output}")
    if not (0.0 < args.amplitude_m <= 0.20 and args.curvature_sigma_m > 0.0):
        raise ValueError("invalid outward offset or curvature smoothing")

    seed = _read_profile(seed_path)
    previous = json.loads((baseline_dir / "dynamics/offline_result.json")
                          .read_text(encoding="utf-8"))
    candidate, geometry = make_candidate(
        seed, args.amplitude_m, args.curvature_sigma_m, tuple(args.sectors))
    output.mkdir(parents=True)
    map_pixels = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(map_pixels == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M
    target_clearance = _map_clearance(geometry.pop("points_world_xz"), map_clearance)
    geometry["target_path_clearance_min_m"] = float(target_clearance.min())
    geometry["target_path_clearance_p05_m"] = float(np.quantile(target_clearance, 0.05))
    if geometry["target_path_clearance_min_m"] < MIN_CLEARANCE_M:
        (output / "preflight_rejected.json").write_text(
            json.dumps(geometry, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "preflight_rejected", **geometry}, indent=2))
        return 2

    manifest: dict[str, Any] = {
        "status": "running",
        "track": "practice",
        "simulator": "tools/start_simulator.sh, official image, batchmode",
        "player": str(player),
        "player_gameassembly_sha256": _sha256(player / "GameAssembly.so"),
        "seed_profile": str(seed_path),
        "seed_profile_sha256": _sha256(seed_path),
        "same_player_baseline": str(baseline_dir),
        "baseline_lap_time_s": previous["scored_duration_s"],
        "geometry_preflight": geometry,
        "runs": [],
    }
    (output / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")
    row = _evaluate("outward_radius_005", candidate, str(seed_path), player,
                    output, map_clearance, args.timeout_s)
    row["lap_time_change_vs_same_player_baseline_s"] = (
        None if row["lap_time_s"] is None else
        float(row["lap_time_s"]) - float(previous["scored_duration_s"]))
    manifest["runs"].append(row)
    manifest["best_valid_lap_s"] = min(
        [float(previous["scored_duration_s"])] +
        ([float(row["lap_time_s"])] if row["valid_lap"] else []))
    manifest["status"] = "complete"
    (output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
