#!/usr/bin/env python3
"""Tune a practice command/raceline profile against the source Unity plant.

The unchanged Unity WheelCollider simulation is the objective evaluator. The
search changes throttle and steering independently in route-distance sectors,
uses simulated position for spatial command lookup, and rejects any lap whose
1 kHz body-position trace loses the required practice-map clearance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from optimize_unity_source_raceline import (
    DEFAULT_CAPTURE,
    DEFAULT_PARITY,
    DEFAULT_PROFILE,
    FIXED_DT_S,
    INVALID_SCORE_S,
    REPO,
    TICKS_PER_COMMAND,
    _finish_gate,
    _dynamics_summary,
    _profile_parts,
    _read_capture_xz,
    _read_json,
    _trial_score,
)
from optimize_unity_spatial_raceline import (
    MIN_TRACK_CLEARANCE_M,
    PRACTICE_MAP,
    PRACTICE_MAP_RESOLUTION_M,
    _read_pgm8,
    _track_clearance_summary,
)


THROTTLE_SECTOR_COUNT = 6
STEERING_SECTOR_COUNT = 12
THROTTLE_NAME = tuple(f"throttle_scale_s{i}"
                      for i in range(THROTTLE_SECTOR_COUNT))
STEERING_NAME = tuple(f"steering_scale_s{i}"
                      for i in range(STEERING_SECTOR_COUNT))
STEERING_OFFSET_NAME = tuple(f"steering_offset_s{i}"
                              for i in range(STEERING_SECTOR_COUNT))
SCALE_PARAMETER_NAMES = THROTTLE_NAME + STEERING_NAME
OFFSET_PARAMETER_NAMES = THROTTLE_NAME + STEERING_OFFSET_NAME
INITIAL_SCALE_PARAMETERS = np.ones(len(SCALE_PARAMETER_NAMES), dtype=np.float64)
INITIAL_OFFSET_PARAMETERS = np.r_[
    np.ones(THROTTLE_SECTOR_COUNT),
    np.zeros(STEERING_SECTOR_COUNT),
]
SCALE_PHASES = (
    ("throttle_coarse", THROTTLE_NAME, 0.50, (0.50, 3.00)),
    ("throttle_refine", THROTTLE_NAME, 0.25, (0.50, 3.00)),
    ("steering_wide", STEERING_NAME, 0.40, (0.40, 1.80)),
    ("steering_refine", STEERING_NAME, 0.15, (0.40, 1.80)),
)
SCALE_PHASE_SECTOR_ORDER = (
    tuple(range(THROTTLE_SECTOR_COUNT)),
    tuple(range(THROTTLE_SECTOR_COUNT)),
    (10, 11, 8, 9, 6, 7, 4, 5, 0, 1, 2, 3),
    (10, 11, 8, 9, 6, 7, 4, 5, 0, 1, 2, 3),
)
OFFSET_PHASES = (
    ("steering_offset_coarse", STEERING_OFFSET_NAME, 0.05, (-0.40, 0.40)),
    ("steering_offset_refine", STEERING_OFFSET_NAME, 0.02, (-0.40, 0.40)),
)
WORKING_CLEARANCE_LOSS_WEIGHT_S = 4.0
DEFAULT_OPTIMIZER_PLAYER = REPO / (
    "live_runs/unity_source_spatial_optimizer_player_20261014_r01/"
    "player_overlay")


def _search_configuration(steering_mode: str):
    if steering_mode == "scale":
        return (SCALE_PARAMETER_NAMES, INITIAL_SCALE_PARAMETERS, SCALE_PHASES,
                SCALE_PHASE_SECTOR_ORDER)
    if steering_mode == "offset":
        return (OFFSET_PARAMETER_NAMES, INITIAL_OFFSET_PARAMETERS, OFFSET_PHASES,
                tuple(tuple(range(STEERING_SECTOR_COUNT))
                      for _ in OFFSET_PHASES))
    raise ValueError(f"unknown steering adjustment mode: {steering_mode}")


def _build_reference_sectors(reference_times: np.ndarray,
                             reference_xz: np.ndarray,
                             warmup_count: int,
                             command_count: int,
                             sector_count: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lap_start_s = warmup_count * TICKS_PER_COMMAND * FIXED_DT_S
    lap_end_s = (warmup_count + command_count) * TICKS_PER_COMMAND * FIXED_DT_S
    selected = (reference_times >= lap_start_s) & (reference_times <= lap_end_s)
    times = reference_times[selected]
    positions = reference_xz[selected]
    if len(times) < 100:
        raise ValueError("reference capture does not contain a full scored lap")
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    if distance[-1] < 10.0:
        raise ValueError("reference lap path is unexpectedly short")
    command_sectors = np.empty(command_count, dtype=np.int32)
    command_positions = np.empty((command_count, 2), dtype=np.float64)
    command_progress = np.empty(command_count, dtype=np.float64)
    for index in range(command_count):
        reference_time = (warmup_count + index) * TICKS_PER_COMMAND * FIXED_DT_S
        nearest = int(np.argmin(np.abs(times - reference_time)))
        fraction = float(distance[nearest] / distance[-1])
        command_sectors[index] = min(int(fraction * sector_count), sector_count - 1)
        command_positions[index] = positions[nearest]
        command_progress[index] = fraction
    return command_sectors, command_positions, command_progress


def _interpolate_sector_values(values: np.ndarray,
                               progress: np.ndarray) -> np.ndarray:
    """Linearly interpolate periodic control knots placed at sector centers."""
    count = len(values)
    centers = (np.arange(count, dtype=np.float64) + 0.5) / count
    xp = np.r_[centers[-1] - 1.0, centers, centers[0] + 1.0]
    fp = np.r_[values[-1], values, values[0]]
    return np.interp(progress, xp, fp)


def make_profile(warmup: list[dict[str, float]],
                 lap: list[dict[str, float]],
                 warmup_count: int,
                 throttle_sectors: np.ndarray,
                 steering_sectors: np.ndarray,
                 command_positions: np.ndarray,
                 command_progress: np.ndarray,
                 parameters: np.ndarray,
                 progress_distance_scale: float,
                 steering_mode: str = "scale",
                 steering_interpolation: str = "step",
                 fixed_steering_offsets: np.ndarray | None = None,
                 zero_throttle_replacement: float | None = None,
                 zero_throttle_sectors: tuple[int, ...] = (),
                 command_index_mode: str = "spatial",
                 steering_time_warp: float = 1.0
                 ) -> dict[str, object]:
    parameter_names, _, _, _ = _search_configuration(steering_mode)
    if len(parameters) != len(parameter_names):
        raise ValueError("sector parameter vector has the wrong length")
    if command_index_mode not in ("spatial", "time"):
        raise ValueError("command index mode must be spatial or time")
    if not np.isfinite(steering_time_warp) or steering_time_warp <= 0.0:
        raise ValueError("steering time warp must be finite and positive")
    throttle_scales = parameters[:THROTTLE_SECTOR_COUNT]
    steering_scales = parameters[THROTTLE_SECTOR_COUNT:]
    fixed_offsets = (np.zeros(STEERING_SECTOR_COUNT, dtype=np.float64)
                     if fixed_steering_offsets is None else
                     np.asarray(fixed_steering_offsets, dtype=np.float64))
    if (fixed_offsets.shape != (STEERING_SECTOR_COUNT,) or
            not np.isfinite(fixed_offsets).all()):
        raise ValueError("fixed steering offsets must contain twelve finite values")
    if steering_interpolation == "linear":
        steering_adjustments = _interpolate_sector_values(
            steering_scales, command_progress)
        steering_offsets = _interpolate_sector_values(fixed_offsets,
                                                       command_progress)
    elif steering_interpolation == "step":
        steering_adjustments = steering_scales[steering_sectors]
        steering_offsets = fixed_offsets[steering_sectors]
    else:
        raise ValueError(f"unknown steering interpolation: {steering_interpolation}")
    commands: list[dict[str, object]] = []
    for command in warmup:
        commands.append({
            "steering_norm": command["steering_norm"],
            "throttle_norm": command["throttle_norm"],
            "has_reference_position": False,
            "reference_x_m": 0.0,
            "reference_z_m": 0.0,
        })
    source_steering = np.asarray(
        [float(command["steering_norm"]) for command in lap],
        dtype=np.float64)
    source_indices = np.minimum(
        np.arange(len(lap), dtype=np.float64) * steering_time_warp,
        len(lap) - 1)
    warped_steering = np.interp(
        source_indices, np.arange(len(lap), dtype=np.float64), source_steering)
    for index, command in enumerate(lap):
        throttle_sector = int(throttle_sectors[index])
        steering_value = float(warped_steering[index])
        if steering_mode == "scale":
            steering_value *= steering_adjustments[index]
        else:
            steering_value += steering_adjustments[index]
        steering_value += steering_offsets[index]
        source_throttle = float(command["throttle_norm"])
        throttle_value = source_throttle * throttle_scales[throttle_sector]
        if (zero_throttle_replacement is not None and source_throttle == 0.0
                and throttle_sector in zero_throttle_sectors):
            throttle_value = zero_throttle_replacement
        commands.append({
            "steering_norm": float(np.clip(steering_value, -1.0, 1.0)),
            "throttle_norm": float(np.clip(throttle_value, -1.0, 1.0)),
            "has_reference_position": True,
            "reference_x_m": float(command_positions[index, 0]),
            "reference_z_m": float(command_positions[index, 1]),
        })
    return {
        "schema_version": 1,
        "fixed_delta_time_s": FIXED_DT_S,
        "physics_ticks_per_command": TICKS_PER_COMMAND,
        "command_phase_fixed_steps": 0,
        "spatial_command_lookup": command_index_mode == "spatial",
        "spatial_progress_distance_scale": progress_distance_scale,
        "command_index_mode": command_index_mode,
        "steering_time_warp": steering_time_warp,
        "steering_adjustment_mode": steering_mode,
        "steering_sector_interpolation": steering_interpolation,
        "fixed_steering_offsets": fixed_offsets.tolist(),
        "zero_throttle_replacement": zero_throttle_replacement,
        "zero_throttle_sectors": list(zero_throttle_sectors),
        "warmup_command_count": warmup_count,
        "scored_horizon_commands": 360,
        "commands": commands,
    }


def _persist(root: Path, manifest: dict[str, object],
             rows: list[dict[str, object]], state: dict[str, object]) -> None:
    manifest["trials"] = [{
        "trial": row["trial"],
        "phase": row.get("phase"),
        "sector": row.get("sector"),
        "parameters": row["parameters"],
        "lap_valid": row["lap_valid"],
        "track_valid": row.get("track_valid"),
        "unity_lap_time_s": row.get("unity_lap_time_s"),
        "objective_s": row["objective_s"],
        "progress": row.get("reference_lap_progress_fraction"),
        "track_clearance_min_m": row.get("track_clearance_min_m"),
        "track_clearance_sector_min_m": row.get(
            "track_clearance_sector_min_m"),
        "track_clearance_sector_below_margin_fraction": row.get(
            "track_clearance_sector_below_margin_fraction"),
        "spatial_path_index_advance_max": row.get(
            "spatial_path_index_advance_max"),
        "spatial_path_index_advance_mean": row.get(
            "spatial_path_index_advance_mean"),
        "expected_fixed_steps": row.get("expected_fixed_steps"),
        "fixed_step_horizon_matches_profile": row.get(
            "fixed_step_horizon_matches_profile"),
        "peak_speed_mps": row.get("peak_body_speed_mps"),
        "p95_abs_sideslip_rad": row.get("p95_abs_sideslip_rad"),
        "p95_kinematic_yaw_underresponse_rps": row.get(
            "p95_kinematic_yaw_underresponse_rps"),
        "p95_abs_lateral_velocity_mps": row.get(
            "p95_abs_lateral_velocity_mps"),
        "p95_abs_forward_wheel_slip": row.get(
            "p95_abs_forward_wheel_slip"),
        "p95_abs_front_sideways_wheel_slip": row.get(
            "p95_abs_front_sideways_wheel_slip"),
        "p95_abs_rear_sideways_wheel_slip": row.get(
            "p95_abs_rear_sideways_wheel_slip"),
        "front_sideways_slip_over_asymptote_fraction": row.get(
            "front_sideways_slip_over_asymptote_fraction"),
        "rear_sideways_slip_over_asymptote_fraction": row.get(
            "rear_sideways_slip_over_asymptote_fraction"),
        "max_abs_forward_wheel_slip": row.get(
            "max_abs_forward_wheel_slip"),
        "wheel_forward_slip_over_extremum_fraction": row.get(
            "wheel_forward_slip_over_extremum_fraction"),
        "wheel_forward_slip_over_asymptote_fraction": row.get(
            "wheel_forward_slip_over_asymptote_fraction"),
        "wheel_sideways_slip_over_extremum_fraction": row.get(
            "wheel_sideways_slip_over_extremum_fraction"),
        "p95_abs_sideslip_rad_speed_ge_6_mps": row.get(
            "p95_abs_sideslip_rad_speed_ge_6_mps"),
        "p95_abs_sideslip_rad_speed_ge_7_mps": row.get(
            "p95_abs_sideslip_rad_speed_ge_7_mps"),
        "p95_kinematic_yaw_underresponse_rps_speed_ge_6_mps": row.get(
            "p95_kinematic_yaw_underresponse_rps_speed_ge_6_mps"),
        "p95_kinematic_yaw_underresponse_rps_speed_ge_7_mps": row.get(
            "p95_kinematic_yaw_underresponse_rps_speed_ge_7_mps"),
        "p95_abs_sideways_wheel_slip_speed_ge_6_mps": row.get(
            "p95_abs_sideways_wheel_slip_speed_ge_6_mps"),
        "p95_abs_sideways_wheel_slip_speed_ge_7_mps": row.get(
            "p95_abs_sideways_wheel_slip_speed_ge_7_mps"),
        "p95_abs_front_sideways_wheel_slip_speed_ge_6_mps": row.get(
            "p95_abs_front_sideways_wheel_slip_speed_ge_6_mps"),
        "p95_abs_front_sideways_wheel_slip_speed_ge_7_mps": row.get(
            "p95_abs_front_sideways_wheel_slip_speed_ge_7_mps"),
        "p95_abs_rear_sideways_wheel_slip_speed_ge_6_mps": row.get(
            "p95_abs_rear_sideways_wheel_slip_speed_ge_6_mps"),
        "p95_abs_rear_sideways_wheel_slip_speed_ge_7_mps": row.get(
            "p95_abs_rear_sideways_wheel_slip_speed_ge_7_mps"),
        "wheel_sideways_slip_over_asymptote_fraction_speed_ge_6_mps": row.get(
            "wheel_sideways_slip_over_asymptote_fraction_speed_ge_6_mps"),
        "wheel_sideways_slip_over_asymptote_fraction_speed_ge_7_mps": row.get(
            "wheel_sideways_slip_over_asymptote_fraction_speed_ge_7_mps"),
        "p95_positive_slip_when_driven_speed_ge_6_mps": row.get(
            "p95_positive_slip_when_driven_speed_ge_6_mps"),
        "p95_positive_slip_when_driven_speed_ge_7_mps": row.get(
            "p95_positive_slip_when_driven_speed_ge_7_mps"),
        "best_capture_path": row.get("best_capture_path"),
        "diagnostic_capture_path": row.get("diagnostic_capture_path"),
    } for row in rows]
    valid = [row for row in rows if row.get("lap_valid")]
    manifest["best_lap_time_s"] = min(
        (float(row["unity_lap_time_s"]) for row in valid), default=None)
    (root / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "optimizer_state.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8")


def _working_score(result: dict[str, object]) -> float:
    """Allow near-track laps to seed steering repair, never qualify them."""
    progress = float(result.get("reference_lap_progress_fraction", 0.0))
    lap_time = result.get("unity_lap_time_s")
    clearance_fraction = float(
        result.get("track_samples_with_vehicle_margin_fraction", 0.0))
    if lap_time is None or progress < 0.95 or clearance_fraction < 0.50:
        return INVALID_SCORE_S + 5.0 * (1.0 - progress)
    clearance_deficit = max(
        0.0, MIN_TRACK_CLEARANCE_M -
        float(result.get("track_clearance_min_m", 0.0)))
    return (float(lap_time) + 2.0 * (1.0 - progress)
            + WORKING_CLEARANCE_LOSS_WEIGHT_S *
            (1.0 - clearance_fraction + clearance_deficit))


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run(args: argparse.Namespace) -> int:
    parameter_names, default_parameters, phases, phase_sector_order = \
        _search_configuration(args.steering_mode)
    fixed_steering_offsets = np.asarray(args.fixed_steering_offsets,
                                        dtype=np.float64)
    if (fixed_steering_offsets.shape != (STEERING_SECTOR_COUNT,) or
            not np.isfinite(fixed_steering_offsets).all()):
        raise ValueError("fixed steering offsets must contain twelve finite values")
    zero_throttle_sectors = tuple(int(value)
                                  for value in args.zero_throttle_sectors)
    if (len(set(zero_throttle_sectors)) != len(zero_throttle_sectors) or
            any(value < 0 or value >= THROTTLE_SECTOR_COUNT
                for value in zero_throttle_sectors)):
        raise ValueError("zero-throttle sectors must be unique indices 0 through 5")
    if args.zero_throttle_command is None:
        if zero_throttle_sectors:
            raise ValueError("zero-throttle sectors require a replacement command")
    elif (not np.isfinite(args.zero_throttle_command) or
          not -1.0 <= args.zero_throttle_command <= 1.0 or
          not zero_throttle_sectors):
        raise ValueError("zero-throttle replacement needs a finite command in [-1,1] and at least one sector")
    source_profile = _read_json(args.source_profile)
    parity = _read_json(args.parity_report)
    warmup, lap, warmup_count = _profile_parts(source_profile, parity)
    reference_times, reference_xz = _read_capture_xz(args.reference_capture)
    throttle_sectors, command_positions, command_progress = _build_reference_sectors(
        reference_times, reference_xz, warmup_count, len(lap),
        THROTTLE_SECTOR_COUNT)
    steering_sectors, _, _ = _build_reference_sectors(
        reference_times, reference_xz, warmup_count, len(lap),
        STEERING_SECTOR_COUNT)
    if args.steering_mode == "offset":
        steering_commands = np.asarray(
            [row["steering_norm"] for row in lap], dtype=np.float64)
        sector_activity = np.asarray([
            np.mean(np.abs(steering_commands[steering_sectors == sector]))
            if np.any(steering_sectors == sector) else 0.0
            for sector in range(STEERING_SECTOR_COUNT)
        ])
        active_first = np.argsort(-sector_activity, kind="stable")
        if args.steering_search_sectors is not None:
            selected_sectors = set(args.steering_search_sectors)
            active_first = np.asarray(
                [sector for sector in active_first
                 if int(sector) in selected_sectors], dtype=np.int64)
        phase_sector_order = tuple(
            tuple(int(sector) for sector in active_first)
            for _ in phases)
    gate_origin, gate_direction, _ = _finish_gate(
        args.reference_capture, warmup_count)
    first_boundary = int(parity["lap_count_boundaries"][0]["packet_index"]) * FIXED_DT_S * TICKS_PER_COMMAND
    second_boundary = int(parity["lap_count_boundaries"][1]["packet_index"]) * FIXED_DT_S * TICKS_PER_COMMAND
    lap_path = reference_xz[(reference_times >= first_boundary) &
                            (reference_times <= second_boundary)]

    occupancy = _read_pgm8(PRACTICE_MAP)
    map_clearance = distance_transform_edt(
        np.pad(occupancy == 254, 1, constant_values=False)
    )[1:-1, 1:-1] * PRACTICE_MAP_RESOLUTION_M

    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError(f"cannot resume missing campaign {args.output}")
        manifest = _read_json(args.output / "campaign.json")
        if manifest.get("steering_adjustment_mode", "scale") != args.steering_mode:
            raise ValueError("resume steering adjustment mode differs from campaign")
        if manifest.get("command_index_mode", "spatial") != args.command_index_mode:
            raise ValueError("resume command index mode differs from campaign")
        if manifest.get("steering_search_sectors") != args.steering_search_sectors:
            raise ValueError("resume steering search sectors differ from campaign")
        if abs(float(manifest.get("steering_time_warp", 1.0)) -
               args.steering_time_warp) > 1.0e-9:
            raise ValueError("resume steering time warp differs from campaign")
        if manifest.get("search_direction", "both") != args.search_direction:
            raise ValueError("resume search direction differs from campaign")
        if manifest.get("steering_sector_interpolation", "step") != \
                args.steering_interpolation:
            raise ValueError("resume steering interpolation differs from campaign")
        if manifest.get("fixed_steering_offsets", [0.0] * STEERING_SECTOR_COUNT) != \
                fixed_steering_offsets.tolist():
            raise ValueError("resume fixed steering offsets differ from campaign")
        if abs(float(manifest["spatial_progress_distance_scale"])
               - args.progress_distance_scale) > 1.0e-9:
            raise ValueError("resume spatial progress scale differs from campaign")
        state = _read_json(args.output / "optimizer_state.json")
        rows = [_read_json(path) for path in sorted(
            (args.output / "trials").glob("trial_*/trial.json"))]
        if [row.get("trial") for row in rows] != [f"trial_{i:03d}" for i in range(len(rows))]:
            raise ValueError("trial records are not contiguous; preserve and inspect the incomplete trial")
        best_params = np.asarray(state["best_parameters"], dtype=np.float64)
        saved_best_time = state.get("best_time_s")
        best_time = (float(saved_best_time) if saved_best_time is not None
                     else float("inf"))
        params = np.asarray(state.get("working_parameters", best_params),
                            dtype=np.float64)
        working_score = float(state.get("working_score_s", best_time))
        if "working_parameters" not in state:
            for row in rows:
                score = _working_score(row)
                if score < working_score:
                    params = np.asarray(
                        [row["parameters"][name] for name in parameter_names],
                        dtype=np.float64)
                    working_score = score
        best_capture = Path(state["best_capture_path"]) if state.get("best_capture_path") else None
        diagnostic_capture = (Path(state["diagnostic_capture_path"])
                              if state.get("diagnostic_capture_path") else None)
    else:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite campaign {args.output}")
        if not args.player_dir.is_dir():
            raise FileNotFoundError(args.player_dir)
        args.output.mkdir(parents=True)
        (args.output / "trials").mkdir()
        rows = []
        params = np.asarray(args.initial_parameters, dtype=np.float64) \
            if args.initial_parameters is not None else default_parameters.copy()
        if params.shape != default_parameters.shape or not np.isfinite(params).all():
            raise ValueError("initial parameter vector must contain 18 finite values")
        if (np.any(params[:THROTTLE_SECTOR_COUNT] < 0.50) or
                np.any(params[:THROTTLE_SECTOR_COUNT] > 3.00)):
            raise ValueError("initial throttle-sector gain is outside [0.5, 3.0]")
        steering_bounds = ((0.40, 1.80) if args.steering_mode == "scale"
                            else (-0.40, 0.40))
        if (np.any(params[THROTTLE_SECTOR_COUNT:] < steering_bounds[0]) or
                np.any(params[THROTTLE_SECTOR_COUNT:] > steering_bounds[1])):
            raise ValueError(
                f"initial steering-sector {args.steering_mode} is outside "
                f"{steering_bounds}")
        best_params = params.copy()
        best_time = float("inf")
        working_score = float("inf")
        best_capture = None
        diagnostic_capture = None
        manifest = {
            "status": "running",
            "simulator": "source-built practice race scene; native Unity WheelCollider; 1 kHz fixed physics",
            "physics_modified": False,
            "objective": "fastest gate-to-gate lap with no collision and >=0.20 m map-derived vehicle-center clearance at every captured 1 kHz step",
            "command_source_profile": str(args.source_profile.resolve()),
            "reference_capture": str(args.reference_capture.resolve()),
            "practice_map": str(PRACTICE_MAP.relative_to(REPO)),
            "throttle_sector_count": THROTTLE_SECTOR_COUNT,
            "steering_sector_count": STEERING_SECTOR_COUNT,
            "steering_adjustment_mode": args.steering_mode,
            "command_index_mode": args.command_index_mode,
            "steering_time_warp": args.steering_time_warp,
            "steering_search_sectors": args.steering_search_sectors,
            "search_direction": args.search_direction,
            "steering_sector_interpolation": args.steering_interpolation,
            "fixed_steering_offsets": fixed_steering_offsets.tolist(),
            "zero_throttle_replacement": args.zero_throttle_command,
            "zero_throttle_sectors": list(zero_throttle_sectors),
            "sectorization": "separate equal arc-length bins for throttle and steering",
            "steering_offset_search_order": (
                [int(value) for value in phase_sector_order[0]]
                if args.steering_mode == "offset" else None),
            "steering_offset_order_basis": (
                "descending mean absolute source steering per arc-length sector"
                if args.steering_mode == "offset" else None),
            "candidate_representation": (
                "open-loop time-indexed command sequence; throttle-sector gains, "
                "optional steering-sector adjustments, and fixed steering-schedule "
                "time warp; path and speed are generated by unchanged Unity "
                "WheelCollider physics without position feedback"),
            "working_score_clearance_loss_weight_s_per_fraction_or_m":
                WORKING_CLEARANCE_LOSS_WEIGHT_S,
            "spatial_progress_distance_scale": args.progress_distance_scale,
            "parameter_names": list(parameter_names),
            "parameters": {name: float(value) for name, value in zip(parameter_names, params)},
            "warmup_commands": warmup_count,
            "scored_commands": len(lap),
            "target_lap_time_s": args.target_s,
            "capture_policy": "retain only the best qualified lap and highest-progress rejected diagnostic",
            "trials": [],
        }
        state = {
            "best_parameters": best_params.tolist(),
            "best_time_s": None,
            "working_parameters": params.tolist(),
            "working_score_s": None,
            "best_capture_path": None,
            "diagnostic_capture_path": None,
            "phase_index": next(
                (i for i, phase in enumerate(phases)
                 if phase[0] == args.start_phase), 0),
            "sector_index": 0,
            "direction_index": 0,
            "coordinate_origin": None,
        }
        if args.seed_trial_json is not None:
            if args.initial_parameters is None:
                raise ValueError("a reused trial requires --initial-parameters")
            source_trial_path = args.seed_trial_json.resolve()
            source_trial = _read_json(source_trial_path)
            source_profile_path = source_trial_path.parent / "command_profile.json"
            source_profile = _read_json(source_profile_path)
            seeded_profile = make_profile(
                warmup, lap, warmup_count, throttle_sectors,
                steering_sectors, command_positions, command_progress,
                params, args.progress_distance_scale, args.steering_mode,
                args.steering_interpolation,
                fixed_steering_offsets, args.zero_throttle_command,
                zero_throttle_sectors, args.command_index_mode,
                args.steering_time_warp)
            if (source_profile.get("commands") != seeded_profile["commands"] or
                    bool(source_profile.get("spatial_command_lookup", False))
                    != seeded_profile["spatial_command_lookup"] or
                    abs(float(source_profile.get(
                        "spatial_progress_distance_scale", 0.0))
                        - args.progress_distance_scale) > 1.0e-9):
                raise ValueError(
                    "reused Unity result does not match the generated command profile")
            trial_dir = args.output / "trials" / "trial_000"
            trial_dir.mkdir()
            (trial_dir / "command_profile.json").write_text(
                json.dumps(seeded_profile, separators=(",", ":")) + "\n",
                encoding="utf-8")
            seed_row = dict(source_trial)
            seed_capture = Path(source_trial.get("capture_path", ""))
            seed_lap_start = source_trial.get("forward_gate_crossings_s", [None])[0]
            seed_lap_time = source_trial.get("unity_lap_time_s")
            if (seed_capture.is_file() and seed_lap_start is not None and
                    seed_lap_time is not None):
                seed_row.update(_dynamics_summary(
                    seed_capture, float(seed_lap_start),
                    float(seed_lap_start) + float(seed_lap_time)))
            seed_row.update({
                "trial": "trial_000",
                "phase": "seed_reused_existing_Unity_evaluation",
                "sector": None,
                "parameters": {name: float(value)
                               for name, value in zip(parameter_names, params)},
                "seed_source_trial": str(source_trial_path),
                "best_capture_path": None,
                "working_search_score_s": _working_score(source_trial),
            })
            if not seed_row.get("lap_valid"):
                seed_row["best_capture_path"] = None
                seed_row["diagnostic_capture_path"] = (
                    source_trial.get("diagnostic_capture_path") or
                    source_trial.get("capture_path"))
            (trial_dir / "trial.json").write_text(
                json.dumps(seed_row, indent=2) + "\n", encoding="utf-8")
            rows.append(seed_row)
            if seed_row.get("lap_valid"):
                best_time = float(seed_row["unity_lap_time_s"])
                best_capture_path = seed_row.get("capture_path")
                best_capture = (Path(best_capture_path)
                                if best_capture_path else None)
                best_params = params.copy()
                state["best_parameters"] = best_params.tolist()
                state["best_time_s"] = best_time
                state["best_capture_path"] = best_capture_path
            else:
                state["best_time_s"] = None
                state["best_capture_path"] = None
                diagnostic_path = seed_row.get("diagnostic_capture_path")
                state["diagnostic_capture_path"] = diagnostic_path
                diagnostic_capture = (Path(diagnostic_path)
                                      if diagnostic_path else None)
            working_score = float(seed_row["working_search_score_s"])
            state["working_parameters"] = params.tolist()
            state["working_score_s"] = working_score
            manifest["seed_reused_from"] = str(source_trial_path)
            manifest["trials"] = []
            _persist(args.output, manifest, rows, state)

    if args.max_evaluations < len(rows):
        raise ValueError("max-evaluations is smaller than existing trial count")
    if not rows:
        phase_name = "baseline"
        sector_name = None
        candidates = [(params.copy(), None, None)]
    else:
        candidates = []
    trial_index = len(rows)
    phase_index = int(state["phase_index"])
    sector_index = int(state["sector_index"])
    direction_index = int(state["direction_index"])
    coordinate_origin = state.get("coordinate_origin")

    while trial_index < args.max_evaluations:
        if candidates:
            candidate, phase_name, sector_name = candidates.pop(0)
        else:
            while phase_index < len(phases):
                phase_name, names, step, limits = phases[phase_index]
                sector_order = phase_sector_order[phase_index]
                if sector_index >= len(sector_order):
                    phase_index += 1
                    sector_index = 0
                    direction_index = 0
                    coordinate_origin = None
                    continue
                route_sector = sector_order[sector_index]
                parameter_name = names[route_sector]
                parameter_index = parameter_names.index(parameter_name)
                if direction_index == 0 or coordinate_origin is None:
                    coordinate_origin = float(params[parameter_index])
                base_value = float(coordinate_origin)
                directions = ((-1.0, 1.0) if args.search_direction == "both"
                              else ((1.0,) if args.search_direction == "positive"
                                    else (-1.0,)))
                if direction_index >= len(directions):
                    sector_index += 1
                    direction_index = 0
                    coordinate_origin = None
                    continue
                direction = directions[direction_index]
                direction_index += 1
                proposed = float(np.clip(base_value + direction * step, *limits))
                if abs(proposed - base_value) < 1.0e-9:
                    continue
                candidate = params.copy()
                candidate[parameter_index] = proposed
                sector_name = parameter_name
                break
            else:
                break

        trial_name = f"trial_{trial_index:03d}"
        trial_dir = args.output / "trials" / trial_name
        trial_dir.mkdir()
        profile = make_profile(warmup, lap, warmup_count,
                               throttle_sectors, steering_sectors,
                               command_positions, command_progress, candidate,
                               args.progress_distance_scale,
                               args.steering_mode,
                               args.steering_interpolation,
                               fixed_steering_offsets,
                               args.zero_throttle_command,
                               zero_throttle_sectors,
                               args.command_index_mode,
                               args.steering_time_warp)
        (trial_dir / "command_profile.json").write_text(
            json.dumps(profile, separators=(",", ":")) + "\n",
            encoding="utf-8")
        result = _trial_score(
            trial_dir, candidate, gate_origin, gate_direction,
            warmup_count, lap_path, args.player_dir, args.timeout_s,
            parameter_names=parameter_names)
        expected_fixed_steps = (
            int(profile["warmup_command_count"] +
                profile["scored_horizon_commands"]) * TICKS_PER_COMMAND
            if profile["spatial_command_lookup"] else
            len(profile["commands"]) * TICKS_PER_COMMAND +
            int(profile["command_phase_fixed_steps"]))
        actual_fixed_steps = int(result.get("fixed_steps_applied") or 0)
        horizon_matches = (
            result.get("unity_status") != "complete" or
            actual_fixed_steps == expected_fixed_steps)
        result["expected_fixed_steps"] = expected_fixed_steps
        result["fixed_step_horizon_matches_profile"] = horizon_matches
        if not horizon_matches:
            result["unity_reason"] = (
                "Unity runner completed a fixed-step horizon inconsistent "
                "with the command profile mode")
            result["lap_valid"] = False
        result["phase"] = phase_name
        result["sector"] = sector_name
        result["parameters"] = {
            name: float(value) for name, value in zip(parameter_names, candidate)}

        capture = Path(result["capture_path"]) if result.get("capture_path") else None
        if capture is not None and result.get("unity_lap_time_s") is not None:
            start_s = float(result["forward_gate_crossings_s"][0])
            stop_s = start_s + float(result["unity_lap_time_s"])
            result.update(_track_clearance_summary(
                capture, start_s, stop_s, map_clearance, lap_path,
                STEERING_SECTOR_COUNT))
            result["track_valid"] = (
                result["track_clearance_min_m"] >= MIN_TRACK_CLEARANCE_M)
        else:
            result.update({
                "track_valid": False,
                "track_clearance_min_m": 0.0,
                "track_clearance_p01_m": 0.0,
                "track_clearance_p05_m": 0.0,
                "track_samples_with_vehicle_margin_fraction": 0.0,
            })
        result["lap_valid"] = bool(result["lap_valid"] and result["track_valid"])
        result["working_search_score_s"] = _working_score(result)
        if result["lap_valid"]:
            result["objective_s"] = float(result["unity_lap_time_s"])
        else:
            result["objective_s"] = INVALID_SCORE_S + 5.0 * (
                1.0 - float(result.get("reference_lap_progress_fraction", 0.0)))
            result["best_capture_path"] = None

        prior_best = best_time
        better = result["lap_valid"] and float(result["unity_lap_time_s"]) < best_time
        better_working = float(result["working_search_score_s"]) < working_score
        if capture is not None:
            if better:
                if (best_capture is not None and best_capture.is_file()
                        and _inside(best_capture, args.output)):
                    best_capture.unlink()
                    for row in rows:
                        if row.get("best_capture_path") == str(best_capture):
                            row["best_capture_path"] = None
                best_capture = capture
                best_time = float(result["unity_lap_time_s"])
                result["best_capture_path"] = str(capture)
            elif not result["lap_valid"] and (
                    diagnostic_capture is None or
                    float(result["working_search_score_s"]) <
                    min((_working_score(row) for row in rows
                         if row.get("diagnostic_capture_path")),
                        default=float("inf"))):
                if (diagnostic_capture is not None and diagnostic_capture.is_file()
                        and _inside(diagnostic_capture, args.output)):
                    diagnostic_capture.unlink()
                diagnostic_capture = capture
                result["diagnostic_capture_path"] = str(capture)
            else:
                capture.unlink()
        result["capture_retained"] = bool(
            capture is not None and capture.is_file())
        (trial_dir / "trial.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows.append(result)
        trial_index += 1

        if better:
            best_params = candidate.copy()
            best_time = float(result["unity_lap_time_s"])
        if better_working:
            params = candidate.copy()
            working_score = float(result["working_search_score_s"])
        state.update({
            "best_parameters": best_params.tolist(),
            "best_time_s": None if not np.isfinite(best_time) else best_time,
            "working_parameters": params.tolist(),
            "working_score_s": None if not np.isfinite(working_score) else working_score,
            "best_capture_path": str(best_capture) if best_capture else None,
            "diagnostic_capture_path": str(diagnostic_capture) if diagnostic_capture else None,
            "phase_index": phase_index,
            "sector_index": sector_index,
            "direction_index": direction_index,
            "coordinate_origin": coordinate_origin,
        })
        if best_time < args.target_s:
            manifest["status"] = "under_5_candidate_found_requires_repeat_and_runtime_validation"
        else:
            manifest["status"] = "running"
        manifest["best_parameters"] = {
            name: float(value) for name, value in zip(parameter_names, best_params)}
        manifest["working_parameters"] = {
            name: float(value) for name, value in zip(parameter_names, params)}
        _persist(args.output, manifest, rows, state)
        print(json.dumps({
            "trial": trial_name,
            "phase": phase_name,
            "sector": sector_name,
            "candidate_parameter": result["parameters"],
            "lap_valid": result["lap_valid"],
            "lap_time_s": result.get("unity_lap_time_s"),
            "track_clearance_min_m": result.get("track_clearance_min_m"),
            "spatial_path_index_advance_max": result.get(
                "spatial_path_index_advance_max"),
            "spatial_path_index_advance_mean": result.get(
                "spatial_path_index_advance_mean"),
            "track_valid_samples": result.get("track_samples_with_vehicle_margin_fraction"),
            "progress": result.get("reference_lap_progress_fraction"),
            "peak_speed_mps": result.get("peak_body_speed_mps"),
            "p95_abs_sideslip_rad": result.get("p95_abs_sideslip_rad"),
            "p95_kinematic_yaw_underresponse_rps": result.get(
                "p95_kinematic_yaw_underresponse_rps"),
            "p95_abs_sideslip_rad_speed_ge_6_mps": result.get(
                "p95_abs_sideslip_rad_speed_ge_6_mps"),
            "p95_abs_sideslip_rad_speed_ge_7_mps": result.get(
                "p95_abs_sideslip_rad_speed_ge_7_mps"),
            "p95_kinematic_yaw_underresponse_rps_speed_ge_6_mps": result.get(
                "p95_kinematic_yaw_underresponse_rps_speed_ge_6_mps"),
            "p95_kinematic_yaw_underresponse_rps_speed_ge_7_mps": result.get(
                "p95_kinematic_yaw_underresponse_rps_speed_ge_7_mps"),
            "p95_abs_sideways_wheel_slip_speed_ge_6_mps": result.get(
                "p95_abs_sideways_wheel_slip_speed_ge_6_mps"),
            "p95_abs_sideways_wheel_slip_speed_ge_7_mps": result.get(
                "p95_abs_sideways_wheel_slip_speed_ge_7_mps"),
            "p95_abs_front_sideways_wheel_slip": result.get(
                "p95_abs_front_sideways_wheel_slip"),
            "p95_abs_rear_sideways_wheel_slip": result.get(
                "p95_abs_rear_sideways_wheel_slip"),
            "p95_abs_front_sideways_wheel_slip_speed_ge_6_mps": result.get(
                "p95_abs_front_sideways_wheel_slip_speed_ge_6_mps"),
            "p95_abs_front_sideways_wheel_slip_speed_ge_7_mps": result.get(
                "p95_abs_front_sideways_wheel_slip_speed_ge_7_mps"),
            "p95_abs_rear_sideways_wheel_slip_speed_ge_6_mps": result.get(
                "p95_abs_rear_sideways_wheel_slip_speed_ge_6_mps"),
            "p95_abs_rear_sideways_wheel_slip_speed_ge_7_mps": result.get(
                "p95_abs_rear_sideways_wheel_slip_speed_ge_7_mps"),
            "p95_positive_slip_when_driven_speed_ge_6_mps": result.get(
                "p95_positive_slip_when_driven_speed_ge_6_mps"),
            "p95_positive_slip_when_driven_speed_ge_7_mps": result.get(
                "p95_positive_slip_when_driven_speed_ge_7_mps"),
            "forward_slip_over_asymptote_fraction": result.get(
                "wheel_forward_slip_over_asymptote_fraction"),
            "lateral_slip_over_extremum_fraction": result.get(
                "wheel_sideways_slip_over_extremum_fraction"),
            "best_time_s": None if not np.isfinite(best_time) else best_time,
            "best_improved": bool(better),
            "elapsed_s": result["wall_time_s"],
        }, separators=(",", ":")), flush=True)
        if best_time < args.target_s:
            break
        if best_time == prior_best and trial_index >= args.max_evaluations:
            break

    if best_time < args.target_s:
        manifest["status"] = "under_5_candidate_found_requires_repeat_and_runtime_validation"
    elif trial_index >= args.max_evaluations:
        manifest["status"] = "evaluation_budget_reached_above_target"
    else:
        manifest["status"] = "coordinate_search_exhausted_above_target"
    manifest["best_lap_time_s"] = None if not np.isfinite(best_time) else best_time
    _persist(args.output, manifest, rows, state)
    return 0 if best_time < args.target_s else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY)
    parser.add_argument("--reference-capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--player-dir", type=Path,
                        default=DEFAULT_OPTIMIZER_PLAYER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steering-mode", choices=("scale", "offset"),
                        default="scale",
                        help="scale the source steering command or add a normalized command offset")
    parser.add_argument("--search-direction", choices=("both", "positive", "negative"),
                        default="both",
                        help="test both signs from a shared coordinate anchor, or only one sign")
    parser.add_argument("--steering-interpolation", choices=("step", "linear"),
                        default="step",
                        help="hold each sector value or linearly blend periodic sector-center knots")
    parser.add_argument("--fixed-steering-offsets", type=float, nargs=12,
                        default=[0.0] * STEERING_SECTOR_COUNT,
                        metavar="U",
                        help="optional normalized steering offsets added after the selected sector gains")
    parser.add_argument("--zero-throttle-command", type=float,
                        help="replace exact zero throttle in selected scored sectors; tests the source controller's zero/brake discontinuity")
    parser.add_argument("--zero-throttle-sectors", type=int, nargs="*",
                        default=[], metavar="SECTOR",
                        help="six-bin throttle sectors whose exact zero commands are replaced")
    parser.add_argument("--command-index-mode", choices=("spatial", "time"),
                        default="spatial",
                        help="select controls by current position, or replay the optimized 40 Hz control sequence open-loop in time")
    parser.add_argument("--steering-time-warp", type=float, default=1.0,
                        help="open-loop steering replay rate; values above 1 advance the source steering schedule earlier while leaving throttle timing unchanged")
    parser.add_argument("--steering-search-sectors", type=int, nargs="+",
                        help="restrict steering-offset search to these route sectors (0-11); sectors remain ordered by measured source steering activity")
    parser.add_argument("--target-s", type=float, default=5.0)
    parser.add_argument("--progress-distance-scale", type=float, default=1.5)
    parser.add_argument("--max-evaluations", type=int, default=49)
    parser.add_argument("--initial-parameters", type=float, nargs=18,
                        help="18 gains: six throttle sectors followed by twelve steering sectors")
    parser.add_argument("--seed-trial-json", type=Path,
                        help="reuse an existing source-Unity evaluation if its generated command profile matches exactly")
    parser.add_argument("--start-phase", choices=[phase[0] for phase in
                        SCALE_PHASES + OFFSET_PHASES],
                        default="throttle_coarse")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.max_evaluations < 1 or args.timeout_s <= 0.0:
        parser.error("evaluation count and timeout must be positive")
    if args.progress_distance_scale < 0.0:
        parser.error("progress distance scale must be nonnegative")
    if not np.isfinite(args.steering_time_warp) or args.steering_time_warp <= 0.0:
        parser.error("steering time warp must be finite and positive")
    if args.steering_search_sectors is not None:
        if (args.steering_mode != "offset" or
                len(set(args.steering_search_sectors)) !=
                len(args.steering_search_sectors) or
                any(index < 0 or index >= STEERING_SECTOR_COUNT
                    for index in args.steering_search_sectors)):
            parser.error("steering-search-sectors requires unique route sector indices 0-11 and steering-mode=offset")
    if args.resume and (args.initial_parameters is not None or
                        args.seed_trial_json is not None):
        parser.error("initial parameters and seed trial are valid only for a new campaign")
    if args.resume and args.start_phase != "throttle_coarse":
        parser.error("--start-phase is only valid for a new campaign")
    if not args.resume and not args.player_dir.is_dir():
        parser.error(f"Unity source player not found: {args.player_dir}")
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
