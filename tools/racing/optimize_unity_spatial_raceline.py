#!/usr/bin/env python3
"""Black-box raceline search with commands indexed by Unity vehicle progress.

The unchanged Unity source plant runs the car at 1 kHz. During the scored lap,
it selects the recorded 40 Hz steering/throttle sample whose captured Unity
position is nearest the simulated car, with forward-only local progress. This
prevents a faster candidate from receiving a corner command at the old
wall-clock time. A Gaussian-process expected-improvement search adjusts the
scored lap's throttle and steering gains; each score is the actual gate-to-gate
time from Unity, invalidated by any collision before the finish.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from optimize_unity_source_raceline import (
    DEFAULT_CAPTURE,
    DEFAULT_PARITY,
    DEFAULT_PLAYER,
    DEFAULT_PROFILE,
    REPO,
    TICKS_PER_COMMAND,
    FIXED_DT_S,
    _finish_gate,
    _profile_parts,
    _read_capture_xz,
    _read_json,
    _trial_score,
)


NAMES = ("throttle_scale", "steering_scale")
BOUNDS = np.asarray([[1.0, 3.0], [0.80, 1.20]], dtype=np.float64)
DOE = (
    (1.00, 1.00),
    (1.60, 0.90),
    (1.90, 1.00),
    (2.50, 1.00),
    (1.60, 0.90),
    (1.60, 1.10),
    (2.20, 0.90),
    (2.20, 1.10),
)
SPATIAL_HORIZON_COMMANDS = 360
PRACTICE_MAP = REPO / "f1tenth_planning/maps/autodrive_practice_20260924_b.pgm"
PRACTICE_MAP_ORIGIN_XY = (-3.68, -8.49)
PRACTICE_MAP_RESOLUTION_M = 0.05
# The practice map uses map-x ~= Unity-z and map-y ~= -Unity-x. These fixed
# offsets were calibrated against the full source-Unity reference lap; on the
# held-out part of that lap the reference remains within one map cell of free
# space. This geometry check is an optimizer constraint, not a runtime input.
UNITY_TO_MAP_OFFSET_XY = (0.24, 0.05)
MIN_TRACK_CLEARANCE_M = 0.20
INVALID_SCORE_S = 20.0


def make_spatial_profile(warmup: list[dict[str, float]],
                         lap: list[dict[str, float]],
                         warmup_count: int,
                         reference_times: np.ndarray,
                         reference_xz: np.ndarray,
                         throttle_scale: float,
                         steering_scale: float) -> dict[str, object]:
    commands: list[dict[str, object]] = []
    for command in warmup:
        commands.append({
            "steering_norm": command["steering_norm"],
            "throttle_norm": command["throttle_norm"],
            "has_reference_position": False,
            "reference_x_m": 0.0,
            "reference_z_m": 0.0,
        })
    for index, command in enumerate(lap):
        reference_time = (warmup_count + index) * TICKS_PER_COMMAND * FIXED_DT_S
        nearest = int(np.argmin(np.abs(reference_times - reference_time)))
        commands.append({
            "steering_norm": float(np.clip(
                command["steering_norm"] * steering_scale, -1.0, 1.0)),
            "throttle_norm": float(np.clip(
                command["throttle_norm"] * throttle_scale, -1.0, 1.0)),
            "has_reference_position": True,
            "reference_x_m": float(reference_xz[nearest, 0]),
            "reference_z_m": float(reference_xz[nearest, 1]),
        })
    return {
        "schema_version": 1,
        "fixed_delta_time_s": FIXED_DT_S,
        "physics_ticks_per_command": TICKS_PER_COMMAND,
        "command_phase_fixed_steps": 0,
        "spatial_command_lookup": True,
        "warmup_command_count": warmup_count,
        "scored_horizon_commands": SPATIAL_HORIZON_COMMANDS,
        "commands": commands,
    }


def _read_pgm8(path: Path) -> np.ndarray:
    """Read the repository's binary P5 8-bit occupancy image."""
    raw = path.read_bytes()
    cursor = 0
    tokens: list[bytes] = []
    while len(tokens) < 4:
        while cursor < len(raw) and raw[cursor] in b" \t\r\n":
            cursor += 1
        if cursor < len(raw) and raw[cursor] == ord("#"):
            newline = raw.find(b"\n", cursor)
            if newline < 0:
                raise ValueError(f"unterminated PGM comment in {path}")
            cursor = newline + 1
            continue
        start = cursor
        while cursor < len(raw) and raw[cursor] not in b" \t\r\n#":
            cursor += 1
        if start == cursor:
            raise ValueError(f"invalid PGM header in {path}")
        tokens.append(raw[start:cursor])
    magic, width_token, height_token, maxval_token = tokens
    width, height, maxval = map(int, (width_token, height_token, maxval_token))
    if magic != b"P5" or maxval != 255 or width < 1 or height < 1:
        raise ValueError(f"expected an 8-bit binary PGM map, got {path}")
    if cursor >= len(raw) or raw[cursor] not in b" \t\r\n":
        raise ValueError(f"missing PGM pixel separator in {path}")
    if raw[cursor:cursor + 2] == b"\r\n":
        cursor += 2
    else:
        cursor += 1
    pixels = np.frombuffer(raw, dtype=np.uint8, count=width * height,
                           offset=cursor)
    if pixels.size != width * height:
        raise ValueError(f"truncated PGM image in {path}")
    return pixels.reshape(height, width)


def _track_clearance_summary(capture_path: Path, start_s: float,
                             stop_s: float,
                             map_clearance: np.ndarray,
                             reference_path: np.ndarray | None = None,
                             sector_count: int = 12
                             ) -> dict[str, object]:
    times, xz = _read_capture_xz(capture_path)
    selection = (times >= start_s) & (times <= stop_s)
    if np.count_nonzero(selection) < 2:
        raise ValueError("Unity capture has too few samples for track validation")
    # Practice-map axes are rotated/reflected relative to Unity's X/Z plane.
    map_x = xz[selection, 1] + UNITY_TO_MAP_OFFSET_XY[0]
    map_y = -xz[selection, 0] + UNITY_TO_MAP_OFFSET_XY[1]
    origin_x, origin_y = PRACTICE_MAP_ORIGIN_XY
    col = np.floor((map_x - origin_x) / PRACTICE_MAP_RESOLUTION_M).astype(int)
    row_from_bottom = np.floor(
        (map_y - origin_y) / PRACTICE_MAP_RESOLUTION_M).astype(int)
    row = map_clearance.shape[0] - 1 - row_from_bottom
    inside = ((row >= 0) & (row < map_clearance.shape[0]) &
              (col >= 0) & (col < map_clearance.shape[1]))
    clearance = np.zeros(len(row), dtype=np.float64)
    clearance[inside] = map_clearance[row[inside], col[inside]]
    result: dict[str, object] = {
        "track_clearance_min_m": float(clearance.min()),
        "track_clearance_p01_m": float(np.quantile(clearance, 0.01)),
        "track_clearance_p05_m": float(np.quantile(clearance, 0.05)),
        "track_samples_with_vehicle_margin_fraction": float(
            np.mean(clearance >= MIN_TRACK_CLEARANCE_M)),
    }
    if reference_path is not None and len(reference_path) > 1:
        nearest = cKDTree(reference_path).query(xz[selection], k=1)[1]
        nearest = _unwrap_closed_route_indices(nearest, len(reference_path))
        sectors = np.minimum(
            ((nearest - nearest[0]) * sector_count /
             len(reference_path)).astype(int),
            sector_count - 1)
        result["track_clearance_sector_min_m"] = [
            (float(np.min(clearance[sectors == sector]))
             if np.any(sectors == sector) else None)
            for sector in range(sector_count)]
        result["track_clearance_sector_below_margin_fraction"] = [
            (float(np.mean(clearance[sectors == sector] < MIN_TRACK_CLEARANCE_M))
             if np.any(sectors == sector) else None)
            for sector in range(sector_count)]
    return result


def _unwrap_closed_route_indices(indices: np.ndarray, route_count: int
                                 ) -> np.ndarray:
    """Unwrap nearest-point indices across the route seam in lap order."""
    indices = np.asarray(indices, dtype=np.int64)
    if (indices.ndim != 1 or len(indices) == 0 or route_count < 2 or
            np.any(indices < 0) or np.any(indices >= route_count)):
        raise ValueError("route indices must be a non-empty in-range vector")
    delta = np.diff(indices)
    wraps = np.cumsum(np.where(
        delta < -route_count / 2.0, 1,
        np.where(delta > route_count / 2.0, -1, 0)))
    unwrapped = np.r_[indices[0], indices[1:] + wraps * route_count]
    return np.maximum.accumulate(unwrapped)


def propose(rows: list[dict[str, object]], seed: int,
            index: int) -> tuple[np.ndarray, dict[str, object]]:
    x = np.asarray([[row["parameters"][name] for name in NAMES]
                    for row in rows], dtype=np.float64)
    y = np.asarray([float(row["objective_s"]) for row in rows])
    lower, upper = BOUNDS[:, 0], BOUNDS[:, 1]
    scaled = (x - lower) / (upper - lower)
    kernel = (ConstantKernel(1.0, (0.1, 10.0))
              * Matern(length_scale=np.full(2, 0.4),
                       length_scale_bounds=(0.08, 2.0), nu=2.5)
              + WhiteKernel(noise_level=1.0e-5,
                            noise_level_bounds=(1.0e-7, 0.05)))
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2,
        random_state=seed + index, alpha=1.0e-8)
    gp.fit(scaled, y)
    candidates = qmc.Sobol(d=2, scramble=True, seed=seed + index).random_base2(12)
    mean, std = gp.predict(candidates, return_std=True)
    improvement = float(np.min(y)) - mean - 0.01
    z = improvement / np.maximum(std, 1.0e-9)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    for point in scaled:
        ei[np.linalg.norm(candidates - point, axis=1) < 0.02] = -np.inf
    chosen = candidates[int(np.argmax(ei))]
    return lower + chosen * (upper - lower), {
        "method": "gaussian_process_expected_improvement",
        "expected_improvement_s": float(np.max(ei)),
        "observations": len(rows),
        "kernel": str(gp.kernel_),
    }


def run(args: argparse.Namespace) -> int:
    profile_path = args.source_profile.resolve()
    parity_path = args.parity_report.resolve()
    capture_path = args.reference_capture.resolve()
    player_dir = args.player_dir.resolve()
    for path in (profile_path, parity_path, capture_path, PRACTICE_MAP):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not player_dir.is_dir():
        raise FileNotFoundError(player_dir)

    occupancy = _read_pgm8(PRACTICE_MAP)
    if occupancy.shape != (312, 105):
        raise ValueError(f"unexpected practice map dimensions: {occupancy.shape}")
    map_clearance = distance_transform_edt(
        np.pad(occupancy == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M

    source = _read_json(profile_path)
    parity = _read_json(parity_path)
    warmup, lap, warmup_count = _profile_parts(source, parity)
    gate_origin, gate_direction, _ = _finish_gate(capture_path, warmup_count)
    reference_times, reference_xz = _read_capture_xz(capture_path)
    boundaries = parity["lap_count_boundaries"]
    lap_start_s = int(boundaries[0]["packet_index"]) * 0.025
    lap_end_s = int(boundaries[1]["packet_index"]) * 0.025
    lap_path = reference_xz[(reference_times >= lap_start_s)
                            & (reference_times <= lap_end_s)]
    if len(lap_path) < 100:
        raise ValueError("reference command replay has no complete lap path")

    source_hash = __import__("hashlib").sha256(profile_path.read_bytes()).hexdigest()
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError(f"cannot resume missing campaign: {args.output}")
        manifest = _read_json(args.output / "campaign.json")
        if manifest.get("source_profile_sha256") != source_hash:
            raise ValueError("resume source command profile differs from campaign")
        rows = [_read_json(path) for path in sorted(
            (args.output / "trials").glob("trial_*/trial.json"))]
        if [r.get("trial") for r in rows] != [f"trial_{i:03d}" for i in range(len(rows))]:
            raise ValueError("campaign trial records are not contiguous")
        best = [row for row in rows if row.get("lap_valid")]
        best_capture = (Path(min(best, key=lambda row: row["objective_s"])
                             ["best_capture_path"]) if best else None)
        diagnostic = [row for row in rows if row.get("diagnostic_capture_path")]
        diagnostic_capture = (Path(max(
            diagnostic,
            key=lambda row: row.get("reference_lap_progress_fraction", 0.0)
        )["diagnostic_capture_path"]) if diagnostic else None)
        for saved in (best_capture, diagnostic_capture):
            if saved is not None and not saved.is_file():
                raise FileNotFoundError(f"preserved capture is missing: {saved}")
        manifest["status"] = "running"
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite campaign: {args.output}")
        args.output.mkdir(parents=True)
        (args.output / "trials").mkdir()
        rows = []
        best_capture = None
        diagnostic_capture = None
        manifest = {
            "status": "running",
            "objective": "Unity source plant gate-to-gate lap time; pre-finish collision invalidates trial",
            "simulator": "source-built practice race scene, native WheelCollider, 1 kHz",
            "command_selection": "nearest captured Unity path position, monotonic forward local search at 40 Hz",
            "source_profile": str(profile_path),
            "source_profile_sha256": source_hash,
            "reference_capture": str(capture_path),
            "warmup_command_count": warmup_count,
            "scored_command_count": len(lap),
            "scored_horizon_commands": SPATIAL_HORIZON_COMMANDS,
            "parameter_names": list(NAMES),
            "parameter_bounds": {name: BOUNDS[i].tolist()
                                 for i, name in enumerate(NAMES)},
            "reference_gate_origin_unity_xz_m": gate_origin.tolist(),
            "reference_gate_forward_unity_xz": gate_direction.tolist(),
            "trials": [],
        }
        (args.output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.max_evaluations <= len(rows):
        raise ValueError("max-evaluations must exceed existing trial count")
    for index in range(len(rows), args.max_evaluations):
        if index < len(DOE):
            parameters = np.asarray(DOE[index], dtype=np.float64)
            proposal: dict[str, object] = {"method": "empirical_initial_design"}
        else:
            parameters, proposal = propose(rows, args.seed, index)
        name = f"trial_{index:03d}"
        trial_dir = args.output / "trials" / name
        trial_dir.mkdir()
        command_profile = make_spatial_profile(
            warmup, lap, warmup_count, reference_times, reference_xz,
            float(parameters[0]), float(parameters[1]))
        (trial_dir / "command_profile.json").write_text(
            json.dumps(command_profile, separators=(",", ":")) + "\n",
            encoding="utf-8")
        result = _trial_score(
            trial_dir, parameters, gate_origin, gate_direction,
            warmup_count, lap_path, player_dir, args.timeout_s,
            parameter_names=NAMES)
        result["proposal"] = proposal

        capture = Path(result["capture_path"]) if result.get("capture_path") else None
        if capture is not None and result.get("unity_lap_time_s") is not None:
            lap_start = result["forward_gate_crossings_s"][0]
            lap_stop = lap_start + float(result["unity_lap_time_s"])
            track_metrics = _track_clearance_summary(
                capture, lap_start, lap_stop, map_clearance)
            result.update(track_metrics)
            result["track_valid"] = (
                track_metrics["track_clearance_min_m"] >= MIN_TRACK_CLEARANCE_M)
        else:
            result["track_valid"] = False
            result["track_clearance_min_m"] = 0.0
            result["track_clearance_p01_m"] = 0.0
            result["track_clearance_p05_m"] = 0.0
            result["track_samples_with_vehicle_margin_fraction"] = 0.0
        result["lap_valid"] = bool(result["lap_valid"] and result["track_valid"])
        if not result["lap_valid"]:
            result["objective_s"] = INVALID_SCORE_S + 5.0 * (
                1.0 - float(result.get("reference_lap_progress_fraction", 0.0)))
            result["best_capture_path"] = None

        prior_valid = [row for row in rows if row.get("lap_valid")]
        better = result["lap_valid"] and (
            not prior_valid or result["objective_s"] < min(
                float(row["objective_s"]) for row in prior_valid))
        if capture is not None:
            if better:
                if best_capture is not None and best_capture.exists():
                    best_capture.unlink()
                best_capture = capture
                result["best_capture_path"] = str(capture)
            elif result["lap_valid"]:
                result["best_capture_path"] = None
            elif not result["lap_valid"] and (
                    diagnostic_capture is None or
                    result.get("reference_lap_progress_fraction", 0.0) >
                    max((float(row.get("reference_lap_progress_fraction", 0.0))
                         for row in rows if row.get("diagnostic_capture_path")),
                        default=-1.0)):
                if diagnostic_capture is not None and diagnostic_capture.exists():
                    diagnostic_capture.unlink()
                diagnostic_capture = capture
                result["diagnostic_capture_path"] = str(capture)
            elif capture.exists():
                capture.unlink()

        (trial_dir / "trial.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(result)
        manifest["trials"] = [{
            "trial": row["trial"],
            "parameters": row["parameters"],
            "proposal": row["proposal"],
            "lap_valid": row["lap_valid"],
            "unity_status": row["unity_status"],
            "collisions_observed": row["collisions_observed"],
            "unity_lap_time_s": row.get("unity_lap_time_s"),
            "objective_s": row["objective_s"],
            "progress": row.get("reference_lap_progress_fraction"),
            "track_valid": row.get("track_valid"),
            "track_clearance_min_m": row.get("track_clearance_min_m"),
            "track_samples_with_vehicle_margin_fraction": row.get(
                "track_samples_with_vehicle_margin_fraction"),
            "peak_speed_mps": row.get("peak_body_speed_mps"),
            "p95_abs_wheel_slip": row.get("p95_abs_wheel_slip"),
            "best_capture_path": row.get("best_capture_path"),
            "diagnostic_capture_path": row.get("diagnostic_capture_path"),
        } for row in rows]
        best_time = min((float(row["unity_lap_time_s"]) for row in rows
                         if row["lap_valid"]), default=None)
        manifest["best_lap_time_s"] = best_time
        manifest["status"] = (
            "target_reached" if best_time is not None and best_time < args.target_s
            else "running")
        (args.output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "trial": name,
            "parameters": result["parameters"],
            "lap_valid": result["lap_valid"],
            "unity_status": result["unity_status"],
            "collisions": result["collisions_observed"],
            "lap_time_s": result.get("unity_lap_time_s"),
            "progress": result.get("reference_lap_progress_fraction"),
            "peak_speed_mps": result.get("peak_body_speed_mps"),
            "p95_abs_wheel_slip": result.get("p95_abs_wheel_slip"),
            "track_valid": result.get("track_valid"),
            "track_clearance_min_m": result.get("track_clearance_min_m"),
            "track_samples_with_vehicle_margin_fraction": result.get(
                "track_samples_with_vehicle_margin_fraction"),
            "best_so_far_s": best_time,
            "wall_time_s": result["wall_time_s"],
        }, separators=(",", ":")), flush=True)
        if manifest["status"] == "target_reached":
            break

    best_time = manifest.get("best_lap_time_s")
    if best_time is None:
        manifest["status"] = "budget_exhausted_without_valid_lap"
    elif float(best_time) >= args.target_s:
        manifest["status"] = "budget_exhausted_above_target"
    (args.output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0 if manifest["status"] == "target_reached" else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--reference-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--player-dir", type=Path, default=DEFAULT_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-s", type=float, default=5.0)
    parser.add_argument("--max-evaluations", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args()
    if args.max_evaluations < 1 or args.timeout_s <= 0.0:
        parser.error("evaluation count and timeout must be positive")
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
