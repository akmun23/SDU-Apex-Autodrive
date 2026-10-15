#!/usr/bin/env python3
"""Bayesian-search smooth open-loop controls against the exact Unity plant.

The seed is a replay of controls previously emitted by the Unity raceline
controller, but candidate scoring has no path-following or slip-feedback
controller: Unity only uses position to select the next point in the recorded
40 Hz command schedule. Four periodic cubic knots independently scale throttle
and steering, giving an eight-dimensional, smooth control basis. Native 1 kHz
Unity WheelCollider physics determines lap time; collisions and less than
0.20 m map clearance are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_unity_source_raceline import (  # noqa: E402
    FIXED_DT_S, INVALID_SCORE_S,
    TICKS_PER_COMMAND, _finish_gate, _read_capture_xz, _read_json,
    _trial_score,
)
from optimize_unity_spatial_raceline import (  # noqa: E402
    MIN_TRACK_CLEARANCE_M, PRACTICE_MAP, PRACTICE_MAP_RESOLUTION_M,
    _read_pgm8, _track_clearance_summary,
)
from optimize_unity_sector_raceline import (  # noqa: E402
    DEFAULT_OPTIMIZER_PLAYER,
)


DEFAULT_SEED_RUN = REPO / (
    "live_runs/unity_open_loop_replay_r115_20261014/trials/trial_001")
DEFAULT_REFERENCE_CAPTURE = REPO / (
    "live_runs/unity_source_raceline_feedback_mintime_20261014_r115/dynamics/"
    "wheel_dynamics_feedback_mintime_20261014_r115.csv")
KNOTS = 4
PARAMETER_NAMES = tuple(
    [f"throttle_gain_knot_{i}" for i in range(KNOTS)] +
    [f"steering_scale_knot_{i}" for i in range(KNOTS)])
LOWER = np.asarray([0.75] * KNOTS + [0.80] * KNOTS, dtype=np.float64)
UPPER = np.asarray([1.60] * KNOTS + [1.20] * KNOTS, dtype=np.float64)
SEED_PARAMETERS = np.ones(2 * KNOTS, dtype=np.float64)
TARGET_S = 5.0
INVALID_TRACK_SCORE_S = 20.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_progress(commands: list[dict[str, Any]]) -> np.ndarray:
    points = np.asarray([[float(row["reference_x_m"]),
                          float(row["reference_z_m"])] for row in commands],
                        dtype=np.float64)
    if len(points) < 8 or not np.isfinite(points).all():
        raise ValueError("spatial seed needs at least eight finite route points")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    closing_length = float(np.linalg.norm(points[0] - points[-1]))
    total_length = float(segment_lengths.sum() + closing_length)
    if total_length < 5.0 or np.any(segment_lengths < 1.0e-5):
        raise ValueError("spatial seed route is degenerate")
    progress = np.r_[0.0, np.cumsum(segment_lengths)] / total_length
    if np.any(np.diff(progress) <= 0.0):
        raise ValueError("spatial seed route progress is not strictly increasing")
    return progress


def _periodic_knots(values: np.ndarray, progress: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (KNOTS,) or not np.isfinite(values).all():
        raise ValueError(f"expected {KNOTS} finite periodic knot values")
    centers = (np.arange(KNOTS, dtype=np.float64) + 0.5) / KNOTS
    x = np.r_[centers, centers[0] + 1.0]
    y = np.r_[values, values[0]]
    query = np.asarray(progress, dtype=np.float64).copy()
    query[query < centers[0]] += 1.0
    return np.asarray(CubicSpline(x, y, bc_type="periodic")(query),
                      dtype=np.float64)


def make_candidate_profile(seed: dict[str, Any], parameters: np.ndarray
                           ) -> dict[str, Any]:
    parameters = np.asarray(parameters, dtype=np.float64)
    if (parameters.shape != (2 * KNOTS,) or not np.isfinite(parameters).all()
            or np.any(parameters < LOWER) or np.any(parameters > UPPER)):
        raise ValueError("candidate parameters are outside the recorded bounds")
    warmup_count = int(seed["warmup_command_count"])
    commands = [dict(row) for row in seed["commands"]]
    route_commands = commands[warmup_count:]
    if not route_commands or not all(
            row.get("has_reference_position") is True for row in route_commands):
        raise ValueError("seed must contain spatially indexed open-loop commands")
    progress = _route_progress(route_commands)
    throttle_gain = _periodic_knots(parameters[:KNOTS], progress)
    steering_scale = _periodic_knots(parameters[KNOTS:], progress)
    for index, command in enumerate(route_commands):
        command["throttle_norm"] = float(np.clip(
            float(command["throttle_norm"]) * throttle_gain[index], -1.0, 1.0))
        command["steering_norm"] = float(np.clip(
            float(command["steering_norm"]) * steering_scale[index], -1.0, 1.0))
    profile = dict(seed)
    profile["commands"] = commands
    return profile


def _horizon_check(profile: dict[str, Any], result: dict[str, Any]
                   ) -> tuple[int, bool | None]:
    expected = int(profile["warmup_command_count"] +
                   profile["scored_horizon_commands"]) * int(
                       profile["physics_ticks_per_command"])
    if result.get("unity_status") != "complete":
        return expected, None
    return expected, result.get("fixed_steps_applied") == expected


def _initial_design() -> list[np.ndarray]:
    center = SEED_PARAMETERS.copy()
    design = []
    for throttle, steering in (
            (1.10, 1.00), (1.20, 1.00), (0.90, 1.00),
            (1.00, 0.95), (1.00, 1.05), (1.10, 0.95),
            (1.10, 1.05), (1.20, 0.95), (1.20, 1.05),
            (0.95, 1.05), (1.05, 0.90), (1.05, 1.10)):
        point = center.copy()
        point[:KNOTS] = throttle
        point[KNOTS:] = steering
        design.append(point)
    # Add smooth, localized changes after the global direction checks.
    for knot in range(KNOTS):
        point = center.copy()
        point[knot] = 1.25
        point[KNOTS + knot] = 1.05
        design.append(point)
    return design


def _scaled(parameters: np.ndarray) -> np.ndarray:
    return (parameters - LOWER) / (UPPER - LOWER)


def _constrained_expected_improvement(
        rows: list[dict[str, Any]], seed: int, index: int
        ) -> tuple[np.ndarray, dict[str, Any]]:
    observed = np.asarray([
        [float(row["parameters"][name]) for name in PARAMETER_NAMES]
        for row in rows], dtype=np.float64)
    feasible_times = [float(row["unity_lap_time_s"]) for row in rows
                      if row.get("lap_valid") and
                      row.get("unity_lap_time_s") is not None]
    timed_rows = [row for row in rows
                  if row.get("unity_lap_time_s") is not None]
    if not feasible_times or len(timed_rows) < 3:
        raise ValueError("constrained search needs a valid incumbent and three timed laps")

    def fit_regressor(x: np.ndarray, y: np.ndarray, random_state: int):
        kernel = (ConstantKernel(1.0, (0.1, 10.0)) *
                  Matern(length_scale=np.full(len(PARAMETER_NAMES), 0.35),
                         length_scale_bounds=(0.08, 2.0), nu=2.5) +
                  WhiteKernel(noise_level=0.01,
                              noise_level_bounds=(1e-5, 0.2)))
        model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
            random_state=random_state, alpha=1.0e-6)
        model.fit(_scaled(x), y)
        return model

    timed_x = np.asarray([
        [float(row["parameters"][name]) for name in PARAMETER_NAMES]
        for row in timed_rows], dtype=np.float64)
    timed_y = np.asarray([float(row["unity_lap_time_s"])
                          for row in timed_rows], dtype=np.float64)
    time_model = fit_regressor(timed_x, timed_y, seed + index)

    # Use measured fraction of scored samples meeting the same map-margin
    # threshold as the hard lap validator. This keeps infeasible trials
    # informative instead of assigning them one flat penalty value.
    margin_fraction = np.asarray([
        float(np.clip(row.get("track_samples_with_vehicle_margin_fraction", 0.0),
                      0.0, 1.0)) for row in rows], dtype=np.float64)
    p05_clearance = np.asarray([
        float(row.get("track_clearance_p05_m") or 0.0) for row in rows],
        dtype=np.float64)
    feasibility_x = observed
    margin_model = fit_regressor(feasibility_x, margin_fraction, seed + index + 1)
    clearance_model = fit_regressor(feasibility_x, p05_clearance, seed + index + 2)

    candidates_scaled = qmc.Sobol(
        d=len(PARAMETER_NAMES), scramble=True, seed=seed + index
    ).random_base2(m=12)
    candidates = LOWER + candidates_scaled * (UPPER - LOWER)
    mean_time, std_time = time_model.predict(candidates_scaled, return_std=True)
    best_time = min(feasible_times)
    improvement = best_time - mean_time - 0.002
    z = improvement / np.maximum(std_time, 1.0e-9)
    expected_improvement = (
        improvement * norm.cdf(z) + std_time * norm.pdf(z))

    mean_fraction, std_fraction = margin_model.predict(
        candidates_scaled, return_std=True)
    mean_clearance, std_clearance = clearance_model.predict(
        candidates_scaled, return_std=True)
    probability_full_margin = norm.cdf(
        (mean_fraction - 0.9999) / np.maximum(std_fraction, 0.01))
    probability_p05_margin = norm.cdf(
        (mean_clearance - MIN_TRACK_CLEARANCE_M) /
        np.maximum(std_clearance, 0.01))
    probability_feasible = probability_full_margin * probability_p05_margin
    chance_constraint = probability_feasible >= 0.50
    if np.any(chance_constraint):
        acquisition = np.maximum(expected_improvement, 0.0) * probability_feasible
        acquisition[~chance_constraint] = 0.0
        selection_mode = "expected_improvement_with_50pct_feasibility_constraint"
    else:
        # If the surrogate cannot identify a 50%-likely safe point, use the
        # next exact Unity run to improve its feasibility boundary, not to
        # gamble on an attractive but almost-certainly off-track lap time.
        acquisition = probability_feasible
        selection_mode = "maximum_predicted_feasibility"
    observed_scaled = _scaled(observed)
    for point in observed_scaled:
        acquisition[np.linalg.norm(candidates_scaled - point, axis=1) < 0.035] = 0.0
    chosen = int(np.argmax(acquisition))
    return candidates[chosen], {
        "method": "constrained_expected_improvement_from_lap_time_and_map_margin",
        "selection_mode": selection_mode,
        "maximum_candidate_feasibility_probability": float(
            np.max(probability_feasible)),
        "expected_improvement_s": float(expected_improvement[chosen]),
        "predicted_lap_time_s": float(mean_time[chosen]),
        "predicted_margin_fraction": float(mean_fraction[chosen]),
        "predicted_clearance_p05_m": float(mean_clearance[chosen]),
        "probability_full_map_margin": float(probability_full_margin[chosen]),
        "probability_p05_clearance": float(probability_p05_margin[chosen]),
        "probability_feasible": float(probability_feasible[chosen]),
        "observations": len(rows),
    }


def propose(rows: list[dict[str, Any]], seed: int, index: int,
            acquisition: str = "constrained_ei"
            ) -> tuple[np.ndarray, dict[str, Any]]:
    design = _initial_design()
    if 1 <= index <= len(design):
        return design[index - 1].copy(), {
            "method": "targeted_joint_throttle_steering_design",
            "design_index": index - 1,
            "observations": len(rows),
        }
    if acquisition == "constrained_ei":
        return _constrained_expected_improvement(rows, seed, index)
    if acquisition != "penalized_ei":
        raise ValueError(f"unknown acquisition strategy: {acquisition}")
    observed = np.asarray([
        [float(row["parameters"][name]) for name in PARAMETER_NAMES]
        for row in rows], dtype=np.float64)
    if len(rows) < len(PARAMETER_NAMES) + 3:
        sequence = qmc.Sobol(d=len(PARAMETER_NAMES), scramble=True,
                             seed=seed).random_base2(m=8)
        candidates = LOWER + sequence * (UPPER - LOWER)
        distance = np.linalg.norm(
            _scaled(candidates)[:, None, :] - _scaled(observed)[None, :, :],
            axis=2)
        novelty = np.min(distance, axis=1)
        novelty[novelty < 0.06] = -np.inf
        selected = int(np.argmax(novelty))
        return candidates[selected], {
            "method": "local_space_filling_after_reproducible_seed",
            "minimum_scaled_distance": float(novelty[selected]),
            "observations": len(rows),
        }

    x = _scaled(observed)
    y = np.asarray([float(row["objective_s"]) for row in rows])
    kernel = (ConstantKernel(1.0, (0.1, 10.0)) *
              Matern(length_scale=np.full(len(PARAMETER_NAMES), 0.35),
                     length_scale_bounds=(0.08, 2.0), nu=2.5) +
              WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 0.2)))
    model = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=1,
        random_state=seed + index, alpha=1.0e-6)
    model.fit(x, y)
    candidates = qmc.Sobol(
        d=len(PARAMETER_NAMES), scramble=True, seed=seed + index
    ).random_base2(m=12)
    mean, std = model.predict(candidates, return_std=True)
    improvement = float(np.min(y)) - mean - 0.002
    z = improvement / np.maximum(std, 1.0e-9)
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    for point in x:
        ei[np.linalg.norm(candidates - point, axis=1) < 0.035] = -np.inf
    chosen = int(np.argmax(ei))
    return LOWER + candidates[chosen] * (UPPER - LOWER), {
        "method": "gaussian_process_expected_improvement",
        "expected_improvement_s": float(ei[chosen]),
        "observations": len(rows),
        "kernel": str(model.kernel_),
    }


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _clear_capture_references(rows: list[dict[str, Any]], capture: Path,
                              campaign_root: Path) -> None:
    target = str(capture.resolve())
    for row in rows:
        changed = False
        for key in ("capture_path", "best_capture_path",
                    "diagnostic_capture_path"):
            if row.get(key) and str(Path(row[key]).resolve()) == target:
                row[key] = None
                changed = True
        if changed:
            trial_json = campaign_root / "trials" / row["trial"] / "trial.json"
            trial_json.write_text(json.dumps(row, indent=2) + "\n",
                                  encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    seed_run = args.seed_run.resolve()
    seed_trial = _read_json(seed_run / "trial.json")
    seed_profile = _read_json(seed_run / "command_profile.json")
    reference_capture = args.reference_capture.resolve()
    player_dir = args.player_dir.resolve()
    if not seed_trial.get("lap_valid") or not seed_trial.get("track_valid"):
        raise ValueError("seed must be a collision-free, map-clearance-qualified lap")
    if not reference_capture.is_file() or not player_dir.is_dir():
        raise FileNotFoundError("reference capture or exact source Unity player is missing")
    warmup_count = int(seed_profile.get("warmup_command_count", 0))
    route_count = len(seed_profile.get("commands", [])) - warmup_count
    if (not seed_profile.get("spatial_command_lookup")
            or seed_profile.get("raceline_feedback_control", False)
            or warmup_count <= 0 or route_count < 8):
        raise ValueError("seed is not a spatially indexed open-loop profile")

    reference_times, reference_xz = _read_capture_xz(reference_capture)
    gate_origin, gate_direction, _ = _finish_gate(reference_capture, warmup_count)
    scored_start_s = warmup_count * TICKS_PER_COMMAND * FIXED_DT_S
    lap_path = reference_xz[reference_times >= scored_start_s]
    occupancy = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(occupancy == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M

    source_hash = _sha256(seed_run / "command_profile.json")
    reference_hash = _sha256(reference_capture)
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError(f"cannot resume missing campaign {args.output}")
        manifest = _read_json(args.output / "campaign.json")
        if (manifest.get("seed_profile_sha256") != source_hash or
                manifest.get("reference_capture_sha256") != reference_hash):
            raise ValueError("resume inputs differ from the original campaign")
        rows = [_read_json(path) for path in sorted(
            (args.output / "trials").glob("trial_*/trial.json"))]
        if [row.get("trial") for row in rows] != [f"trial_{i:03d}" for i in range(len(rows))]:
            raise ValueError("campaign trials are not contiguous; inspect before resuming")
        best_rows = [row for row in rows if row.get("lap_valid")]
        best_capture = (Path(min(best_rows, key=lambda row: row["objective_s"])
                             ["best_capture_path"])
                        if best_rows else None)
        diagnostic_rows = [row for row in rows if row.get("diagnostic_capture_path")]
        diagnostic_capture = (Path(max(
            diagnostic_rows,
            key=lambda row: row.get("reference_lap_progress_fraction", 0.0)
        )["diagnostic_capture_path"]) if diagnostic_rows else None)
        for saved in (best_capture, diagnostic_capture):
            if saved is not None and not saved.is_file():
                raise FileNotFoundError(f"preserved campaign capture is missing: {saved}")
        previous_acquisition = manifest.get("acquisition")
        if previous_acquisition != args.acquisition:
            manifest.setdefault("acquisition_history", []).append({
                "previous": previous_acquisition or "penalized_expected_improvement",
                "next": args.acquisition,
                "starting_trial_index": len(rows),
            })
            manifest["acquisition"] = args.acquisition
        manifest["status"] = "running"
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite campaign {args.output}")
        if int(args.max_evaluations) < 2:
            raise ValueError("max evaluations must include the reused seed and one proposal")
        args.output.mkdir(parents=True)
        (args.output / "trials").mkdir()
        seed_profile_out = make_candidate_profile(seed_profile, SEED_PARAMETERS)
        first_dir = args.output / "trials/trial_000"
        first_dir.mkdir()
        (first_dir / "command_profile.json").write_text(
            json.dumps(seed_profile_out, separators=(",", ":")) + "\n",
            encoding="utf-8")
        row = dict(seed_trial)
        row.update({
            "trial": "trial_000",
            "phase": "reused_exact_Unity_open_loop_spatial_seed",
            "parameters": dict(zip(PARAMETER_NAMES, map(float, SEED_PARAMETERS))),
            "proposal": {"method": "reused_verified_seed", "source_run": str(seed_run)},
            "objective_s": float(seed_trial["unity_lap_time_s"]),
            "best_capture_path": str(Path(seed_trial["capture_path"]).resolve()),
            "diagnostic_capture_path": None,
        })
        (first_dir / "trial.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8")
        rows = [row]
        best_capture = Path(row["best_capture_path"])
        diagnostic_capture = None
        manifest = {
            "status": "running",
            "objective": "minimum exact-Unity gate-to-gate lap time; require zero pre-finish collisions and >=0.20 m map-derived center clearance",
            "simulator": "source-built practice scene, native Unity WheelCollider, 1 kHz; no racing controller in candidate scoring",
            "fixed_step_horizon_guard": "every completed candidate must report exactly (warmup_command_count + scored_horizon_commands) * physics_ticks_per_command steps; any mismatch terminates the campaign",
            "acquisition": args.acquisition,
            "command_selection": "forward-only nearest spatial reference point at 40 Hz; steering/throttle are open-loop values",
            "physics_modified": False,
            "seed_profile": str((seed_run / "command_profile.json").resolve()),
            "seed_profile_sha256": source_hash,
            "reference_capture": str(reference_capture),
            "reference_capture_sha256": reference_hash,
            "parameter_names": list(PARAMETER_NAMES),
            "parameter_bounds": {name: [float(LOWER[i]), float(UPPER[i])]
                                 for i, name in enumerate(PARAMETER_NAMES)},
            "control_basis": "four periodic cubic route-progress knots each for throttle gain and steering scale",
            "wheel_slip_use": "recorded as a per-run diagnostic; no arbitrary slip cutoff is used to reject a physically completed lap",
            "target_lap_time_s": args.target_s,
            "warmup_commands": warmup_count,
            "scored_route_commands": route_count,
            "trials": [],
        }
        manifest["trials"] = [{
            "trial": row["trial"],
            "parameters": row["parameters"],
            "proposal": row["proposal"],
            "lap_valid": row["lap_valid"],
            "unity_status": row.get("unity_status"),
            "collisions_observed": row.get("collisions_observed"),
            "lap_time_s": row.get("unity_lap_time_s"),
            "objective_s": row["objective_s"],
            "progress": row.get("reference_lap_progress_fraction"),
            "track_clearance_min_m": row.get("track_clearance_min_m"),
            "peak_body_speed_mps": row.get("peak_body_speed_mps"),
            "p95_abs_sideways_wheel_slip": row.get(
                "p95_abs_sideways_wheel_slip"),
            "best_capture_path": row.get("best_capture_path"),
        } for row in rows]
        manifest["best_lap_time_s"] = float(row["unity_lap_time_s"])
        (args.output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.max_evaluations < len(rows):
        raise ValueError("max evaluations is below the number of saved trials")
    for index in range(len(rows), int(args.max_evaluations)):
        params, proposal = propose(
            rows, args.seed, index, acquisition=args.acquisition)
        trial_name = f"trial_{index:03d}"
        trial_dir = args.output / "trials" / trial_name
        trial_dir.mkdir()
        profile = make_candidate_profile(seed_profile, params)
        (trial_dir / "command_profile.json").write_text(
            json.dumps(profile, separators=(",", ":")) + "\n",
            encoding="utf-8")
        result = _trial_score(
            trial_dir, params, gate_origin, gate_direction, warmup_count,
            lap_path, player_dir, args.timeout_s,
            parameter_names=PARAMETER_NAMES)
        expected_fixed_steps, horizon_matches = _horizon_check(profile, result)
        result["expected_fixed_steps"] = expected_fixed_steps
        result["fixed_step_horizon_matches_profile"] = horizon_matches
        if horizon_matches is False:
            result["unity_reason"] = (
                "Unity completed a fixed-step horizon inconsistent with "
                "the command profile")
            result["lap_valid"] = False
        result["proposal"] = proposal
        capture = Path(result["capture_path"]) if result.get("capture_path") else None
        if capture is not None and result.get("unity_lap_time_s") is not None:
            lap_start = float(result["forward_gate_crossings_s"][0])
            lap_stop = lap_start + float(result["unity_lap_time_s"])
            result.update(_track_clearance_summary(
                capture, lap_start, lap_stop, map_clearance, lap_path))
            result["track_valid"] = (
                float(result["track_clearance_min_m"]) >= MIN_TRACK_CLEARANCE_M)
        else:
            result.update({
                "track_valid": False,
                "track_clearance_min_m": 0.0,
                "track_samples_with_vehicle_margin_fraction": 0.0,
            })
        result["lap_valid"] = bool(result["lap_valid"] and result["track_valid"])
        if result["lap_valid"]:
            result["objective_s"] = float(result["unity_lap_time_s"])
        else:
            progress = float(result.get("reference_lap_progress_fraction", 0.0))
            result["objective_s"] = INVALID_TRACK_SCORE_S + 5.0 * (1.0 - progress)
            result["best_capture_path"] = None

        valid_prior = [row for row in rows if row.get("lap_valid")]
        is_best = result["lap_valid"] and (
            not valid_prior or result["objective_s"] < min(
                float(row["objective_s"]) for row in valid_prior))
        if capture is not None:
            if is_best:
                if (best_capture is not None and best_capture.exists() and
                        _inside(best_capture, args.output)):
                    _clear_capture_references(rows, best_capture, args.output)
                    best_capture.unlink()
                best_capture = capture
                result["best_capture_path"] = str(capture)
            elif not result["lap_valid"] and (
                    diagnostic_capture is None or
                    float(result.get("reference_lap_progress_fraction", 0.0)) >
                    max((float(row.get("reference_lap_progress_fraction", 0.0))
                         for row in rows if row.get("diagnostic_capture_path")),
                        default=-1.0)):
                if (diagnostic_capture is not None and diagnostic_capture.exists()
                        and _inside(diagnostic_capture, args.output)):
                    _clear_capture_references(rows, diagnostic_capture, args.output)
                    diagnostic_capture.unlink()
                diagnostic_capture = capture
                result["diagnostic_capture_path"] = str(capture)
            elif capture.exists():
                capture.unlink()
                result["capture_path"] = None
                result["best_capture_path"] = None
        (trial_dir / "trial.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(result)
        best_time = min((float(row["unity_lap_time_s"]) for row in rows
                         if row.get("lap_valid")), default=None)
        manifest["trials"] = [{
            "trial": row["trial"],
            "parameters": row["parameters"],
            "proposal": row["proposal"],
            "lap_valid": row["lap_valid"],
            "unity_status": row.get("unity_status"),
            "collisions_observed": row.get("collisions_observed"),
            "lap_time_s": row.get("unity_lap_time_s"),
            "expected_fixed_steps": row.get("expected_fixed_steps"),
            "fixed_step_horizon_matches_profile": row.get(
                "fixed_step_horizon_matches_profile"),
            "objective_s": row.get("objective_s"),
            "progress": row.get("reference_lap_progress_fraction"),
            "track_clearance_min_m": row.get("track_clearance_min_m"),
            "track_clearance_p05_m": row.get("track_clearance_p05_m"),
            "track_samples_with_vehicle_margin_fraction": row.get(
                "track_samples_with_vehicle_margin_fraction"),
            "peak_body_speed_mps": row.get("peak_body_speed_mps"),
            "p95_abs_sideways_wheel_slip": row.get(
                "p95_abs_sideways_wheel_slip"),
            "best_capture_path": row.get("best_capture_path"),
        } for row in rows]
        manifest["best_lap_time_s"] = best_time
        manifest["status"] = (
            "target_reached" if best_time is not None and best_time < args.target_s
            else "running")
        (args.output / "campaign.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if horizon_matches is False:
            manifest["status"] = "invalid_harness_horizon"
            manifest["invalid_reason"] = result["unity_reason"]
            (args.output / "campaign.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return 4
        print(json.dumps({
            "trial": trial_name,
            "parameters": result["parameters"],
            "lap_valid": result["lap_valid"],
            "unity_status": result["unity_status"],
            "collisions": result.get("collisions_observed"),
            "lap_time_s": result.get("unity_lap_time_s"),
            "progress": result.get("reference_lap_progress_fraction"),
            "track_clearance_min_m": result.get("track_clearance_min_m"),
            "peak_speed_mps": result.get("peak_body_speed_mps"),
            "p95_abs_sideways_wheel_slip": result.get(
                "p95_abs_sideways_wheel_slip"),
            "best_so_far_s": best_time,
            "wall_time_s": result.get("wall_time_s"),
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
    parser.add_argument("--seed-run", type=Path, default=DEFAULT_SEED_RUN)
    parser.add_argument("--reference-capture", type=Path,
                        default=DEFAULT_REFERENCE_CAPTURE)
    parser.add_argument("--player-dir", type=Path,
                        default=DEFAULT_OPTIMIZER_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evaluations", type=int, default=24)
    parser.add_argument("--acquisition", choices=("constrained_ei", "penalized_ei"),
                        default="constrained_ei")
    parser.add_argument("--target-s", type=float, default=TARGET_S)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.max_evaluations < 1 or args.timeout_s <= 0.0:
        parser.error("evaluation count and timeout must be positive")
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
