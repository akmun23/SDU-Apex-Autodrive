#!/usr/bin/env python3
"""Build a Unity profile that follows a geometric raceline in closed loop.

Warm-up commands are retained only to reach the established lap gate. During
the scored lap, steering is generated from the Unity rigid-body pose and the
raceline geometry. Recorded Unity throttle is used as spatial feed-forward;
speed error is corrected from the current simulated velocity.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d


REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRAJECTORY = REPO / "f1tenth_planning/trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv"
DEFAULT_WARMUP_PROFILE = REPO / "live_runs/unity_source_throttle_s2_plus10_blend010_20261014_r01/trials/trial_000/command_profile.json"
DEFAULT_CAPTURE = REPO / "live_runs/unity_source_practice_scene_command_replay_20261013_r07/dynamics_capture/wheel_dynamics_unity_source_practice_race_scene_r07.csv"


def read_trajectory(path: Path) -> tuple[list[dict[str, float]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    points = [
        {
            "s": float(row["# s_m"]),
            "x": float(row["x_m"]),
            "y": float(row["y_m"]),
            "speed": float(row["vx_mps"]),
        }
        for row in rows
    ]
    if len(points) > 2 and np.hypot(points[-1]["x"] - points[0]["x"],
                                   points[-1]["y"] - points[0]["y"]) < 1.0e-5:
        points.pop()
    map_xy = np.asarray([[point["x"], point["y"]] for point in points])
    if len(points) < 100 or not np.isfinite(map_xy).all():
        raise ValueError("trajectory is too short or contains invalid coordinates")
    return points, map_xy


def read_capture(path: Path, gate_time_s: float, path_xy: np.ndarray,
                 start_index: int) -> tuple[np.ndarray, np.ndarray]:
    selected_positions: list[tuple[float, float]] = []
    selected_throttle: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(line for line in stream if not line.startswith("#"))
        header = next(reader)
        columns = {name: index for index, name in enumerate(header)}
        required = ("sim_time_s", "pos_world_x_m", "pos_world_z_m",
                    "throttle_command")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"Unity capture lacks required columns: {missing}")
        for row in reader:
            time_s = float(row[columns["sim_time_s"]])
            if time_s < gate_time_s:
                continue
            if time_s > gate_time_s + 8.0:
                break
            selected_positions.append((float(row[columns["pos_world_x_m"]]),
                                        float(row[columns["pos_world_z_m"]])))
            selected_throttle.append(float(row[columns["throttle_command"]]))

    positions = np.asarray(selected_positions, dtype=np.float64)
    throttle = np.asarray(selected_throttle, dtype=np.float64)
    if len(positions) < 1000:
        raise ValueError("Unity capture does not contain a full post-gate lap")

    # Convert Unity XZ into the planning-map frame and unwrap the nearest
    # reference waypoint into monotonic route progress.
    map_positions = np.column_stack((positions[:, 1] + 0.24,
                                     -positions[:, 0] + 0.05))
    path_tree = cKDTree(path_xy)
    _, nearest = path_tree.query(map_positions, k=1)
    relative = (nearest.astype(np.int64) - start_index) % len(path_xy)
    progress = relative.astype(np.float64)
    lap_offset = 0
    for index in range(1, len(progress)):
        if progress[index] < progress[index - 1] - len(path_xy) / 2:
            lap_offset += len(path_xy)
        progress[index] += lap_offset
        if progress[index] < progress[index - 1]:
            progress[index] = progress[index - 1]

    first_lap = progress <= len(path_xy) + 2
    progress = progress[first_lap]
    throttle = throttle[first_lap]
    if len(progress) < 1000 or progress[-1] < len(path_xy) * 0.95:
        raise ValueError("nearest-path projection did not cover a complete lap")
    # Several 1 kHz samples map to each route waypoint. Mean aggregation
    # suppresses projection jitter while retaining the held 40 Hz command.
    unique_progress, inverse = np.unique(progress.astype(np.int64),
                                         return_inverse=True)
    sums = np.bincount(inverse, weights=throttle)
    counts = np.bincount(inverse)
    means = sums / np.maximum(counts, 1)
    if len(unique_progress) < 100:
        raise ValueError("too few route-aligned throttle samples")
    return unique_progress.astype(np.float64), means


def read_lateral_error_by_sector(path: Path, gate_time_s: float,
                                 world_path: np.ndarray,
                                 start_index: int,
                                 sector_count: int) -> np.ndarray:
    positions: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(line for line in stream if not line.startswith("#"))
        header = next(reader)
        columns = {name: index for index, name in enumerate(header)}
        required = ("sim_time_s", "pos_world_x_m", "pos_world_z_m")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"Correction capture lacks columns: {missing}")
        for row in reader:
            if float(row[columns["sim_time_s"]]) < gate_time_s:
                continue
            positions.append((float(row[columns["pos_world_x_m"]]),
                              float(row[columns["pos_world_z_m"]])))
    actual = np.asarray(positions, dtype=np.float64)
    if len(actual) < 1000:
        raise ValueError("Correction capture does not contain a full Unity lap")

    tangent = np.roll(world_path, -1, axis=0) - np.roll(world_path, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1.0e-9)
    _, nearest = cKDTree(world_path).query(actual, k=1)
    relative_index = (nearest.astype(np.int64) - start_index) % len(world_path)
    offset = actual - world_path[nearest]
    signed_lateral_error = (
        tangent[nearest, 0] * offset[:, 1] -
        tangent[nearest, 1] * offset[:, 0]
    )
    sector = np.minimum(relative_index * sector_count // len(world_path),
                        sector_count - 1)
    medians = np.zeros(sector_count, dtype=np.float64)
    for index in range(sector_count):
        samples = signed_lateral_error[sector == index]
        if len(samples) < 20:
            raise ValueError(
                f"Correction capture has only {len(samples)} samples in sector {index}")
        medians[index] = float(np.median(samples))
    return medians


def parse_sector_scales(value: str | None, sector_count: int,
                        label: str) -> np.ndarray:
    if value is None:
        return np.ones(sector_count, dtype=np.float64)
    try:
        scales = np.asarray([float(item) for item in value.split(",")],
                            dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"sector {label} must be comma-separated numbers") from error
    if (scales.shape != (sector_count,) or not np.isfinite(scales).all() or
            np.any(scales <= 0.0)):
        raise ValueError(
            f"expected {sector_count} finite, positive sector {label}")
    return scales


def interpolate_sector_scales(scales: np.ndarray,
                              point_count: int) -> np.ndarray:
    """Interpolate sector values between sector centers around the lap."""
    if point_count <= 0 or scales.ndim != 1 or scales.size == 0:
        raise ValueError("sector scales and route point count must be non-empty")
    centers = (np.arange(scales.size, dtype=np.float64) + 0.5) * (
        point_count / scales.size)
    query = np.arange(point_count, dtype=np.float64)
    xp = np.r_[centers[-1] - point_count, centers,
               centers[0] + point_count]
    fp = np.r_[scales[-1], scales, scales[0]]
    return np.interp(query, xp, fp)


def closed_path_curvature(path_xy: np.ndarray) -> np.ndarray:
    """Estimate signed curvature at each point of a closed path."""
    previous = path_xy - np.roll(path_xy, 1, axis=0)
    following = np.roll(path_xy, -1, axis=0) - path_xy
    chord = np.roll(path_xy, -1, axis=0) - np.roll(path_xy, 1, axis=0)
    cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    denominator = (np.linalg.norm(previous, axis=1) *
                   np.linalg.norm(following, axis=1) *
                   np.linalg.norm(chord, axis=1))
    return 2.0 * cross / np.maximum(denominator, 1.0e-9)


def smooth_closed_curvature(path_xy: np.ndarray,
                            smoothing_distance_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return signed curvature smoothed over a physical distance along-path."""
    curvature = closed_path_curvature(path_xy)
    if smoothing_distance_m < 0.0 or not np.isfinite(smoothing_distance_m):
        raise ValueError("curvature smoothing distance must be finite and nonnegative")
    if smoothing_distance_m == 0.0:
        return curvature, np.linalg.norm(
            np.roll(path_xy, -1, axis=0) - path_xy, axis=1)
    segment_lengths = np.linalg.norm(
        np.roll(path_xy, -1, axis=0) - path_xy, axis=1)
    mean_spacing = float(np.mean(segment_lengths))
    if mean_spacing <= 0.0:
        raise ValueError("raceline has zero mean waypoint spacing")
    smoothed = gaussian_filter1d(
        curvature, sigma=smoothing_distance_m / mean_spacing, mode="wrap")
    return smoothed, segment_lengths


def longitudinal_speed_envelope(speed_caps_mps: np.ndarray,
                                segment_lengths_m: np.ndarray,
                                acceleration_limit_mps2: float,
                                braking_limit_mps2: float,
                                gate_speed_mps: float) -> np.ndarray:
    """Apply acceleration/braking reachability to one closed-lap speed plan.

    The profile starts and ends at the measured lap-gate speed. The final
    waypoint is immediately before the closing segment to the first point.
    """
    caps = np.asarray(speed_caps_mps, dtype=np.float64)
    distances = np.asarray(segment_lengths_m, dtype=np.float64)
    if (caps.ndim != 1 or distances.shape != caps.shape or caps.size < 2 or
            not np.isfinite(caps).all() or not np.isfinite(distances).all() or
            np.any(caps <= 0.0) or np.any(distances <= 0.0) or
            not np.isfinite((acceleration_limit_mps2, braking_limit_mps2,
                             gate_speed_mps)).all() or
            min(acceleration_limit_mps2, braking_limit_mps2,
                gate_speed_mps) <= 0.0):
        raise ValueError("invalid speed-envelope inputs")
    if min(caps[0], caps[-1]) < gate_speed_mps:
        raise ValueError(
            "raceline speed cap is below the measured gate speed at lap boundary")

    speed = caps.copy()
    speed[0] = gate_speed_mps
    speed[-1] = gate_speed_mps
    for _ in range(8):
        previous = speed.copy()
        for index in range(speed.size - 1):
            reachable = np.sqrt(
                speed[index] ** 2 +
                2.0 * acceleration_limit_mps2 * distances[index])
            if index + 1 == speed.size - 1:
                if reachable + 1.0e-6 < gate_speed_mps:
                    raise ValueError(
                        "measured gate speed is unreachable across the lap-closing segment")
            else:
                speed[index + 1] = min(speed[index + 1], reachable)
        for index in range(speed.size - 2, 0, -1):
            reachable = np.sqrt(
                speed[index + 1] ** 2 +
                2.0 * braking_limit_mps2 * distances[index])
            speed[index] = min(speed[index], reachable)
        if np.max(np.abs(speed - previous)) < 1.0e-8:
            break

    # The closing segment must support transition back to the gate speed in
    # either direction; dense waypoint spacing makes this a direct check.
    closing_distance = distances[-1]
    if (speed[-1] > np.sqrt(gate_speed_mps ** 2 +
                            2.0 * braking_limit_mps2 * closing_distance) + 1.0e-6 or
            speed[-1] < np.sqrt(max(
                gate_speed_mps ** 2 -
                2.0 * acceleration_limit_mps2 * closing_distance, 0.0)) - 1.0e-6):
        raise ValueError("lap-gate speed transition is not longitudinally feasible")
    speed[0] = gate_speed_mps
    next_squared = np.roll(speed, -1) ** 2
    current_squared = speed ** 2
    acceleration_demand = (next_squared - current_squared) / (2.0 * distances)
    braking_demand = -acceleration_demand
    if (np.max(acceleration_demand) > acceleration_limit_mps2 + 1.0e-5 or
            np.max(braking_demand) > braking_limit_mps2 + 1.0e-5):
        raise ValueError("speed-envelope pass left an unreachable adjacent target")
    return speed


def parse_sector_slip_feedback_scales(value: str | None,
                                     sector_count: int) -> np.ndarray:
    if value is None:
        return np.ones(sector_count, dtype=np.float64)
    try:
        scales = np.asarray([float(item) for item in value.split(",")],
                            dtype=np.float64)
    except ValueError as error:
        raise ValueError(
            "sector positive-slip feedback scales must be comma-separated numbers") from error
    if (scales.shape != (sector_count,) or not np.isfinite(scales).all() or
            np.any(scales < 0.0)):
        raise ValueError(
            f"expected {sector_count} finite, nonnegative positive-slip feedback scales")
    return scales


def parse_sector_offset_adjustments(value: str | None,
                                    sector_count: int) -> np.ndarray:
    if value is None:
        return np.zeros(sector_count, dtype=np.float64)
    try:
        offsets = np.asarray([float(item) for item in value.split(",")],
                             dtype=np.float64)
    except ValueError as error:
        raise ValueError(
            "sector offset adjustments must be comma-separated numbers") from error
    if offsets.shape != (sector_count,) or not np.isfinite(offsets).all():
        raise ValueError(
            f"expected {sector_count} finite sector offset adjustments")
    return offsets


def build_profile(args: argparse.Namespace) -> dict[str, object]:
    points, map_xy = read_trajectory(args.trajectory)
    sector_speed_scales = parse_sector_scales(
        args.sector_speed_scales, args.steering_sectors, "speed scales")
    sector_steering_gains = parse_sector_scales(
        args.sector_steering_gains, args.steering_sectors,
        "steering gain scales")
    sector_speed_feedback_scales = parse_sector_scales(
        args.sector_speed_feedback_scales, args.steering_sectors,
        "speed-feedback scales")
    sector_positive_slip_feedback_scales = parse_sector_slip_feedback_scales(
        args.sector_positive_forward_slip_feedback_scales,
        args.steering_sectors)
    point_speed_scales = interpolate_sector_scales(
        sector_speed_scales, len(points))
    sector_offset_adjustments = parse_sector_offset_adjustments(
        args.sector_offset_adjustments_m, args.steering_sectors)
    if args.start_index < 0 or args.start_index >= len(points):
        raise ValueError("start waypoint is outside the trajectory")
    with args.warmup_profile.open(encoding="utf-8") as stream:
        source_profile = json.load(stream)
    warmup_count = int(source_profile["warmup_command_count"])
    if warmup_count <= 0 or len(source_profile["commands"]) < warmup_count:
        raise ValueError("warm-up source profile is incomplete")
    warmup = source_profile["commands"][:warmup_count]
    gate_time_s = warmup_count * 25 * 0.001
    capture_progress, capture_throttle = read_capture(
        args.capture, gate_time_s, map_xy, args.start_index)

    order = np.r_[np.arange(args.start_index, len(points)),
                  np.arange(0, args.start_index)]
    world_xy = np.column_stack((0.05 - map_xy[:, 1], map_xy[:, 0] - 0.24))
    ordered_xy = world_xy[order]
    sector_offsets = np.zeros(args.steering_sectors, dtype=np.float64)
    if args.correction_capture is not None:
        measured_lateral_error = read_lateral_error_by_sector(
            args.correction_capture, gate_time_s, world_xy,
            args.start_index, args.steering_sectors)
        sector_offsets = -args.offset_correction_fraction * measured_lateral_error
        if args.offset_smoothing_sectors > 0.0:
            sector_offsets = gaussian_filter1d(
                sector_offsets, sigma=args.offset_smoothing_sectors, mode="wrap")

    def apply_lateral_offsets(path: np.ndarray,
                              offsets_by_sector: np.ndarray,
                              smoothing_sectors: float) -> np.ndarray:
        if not np.any(offsets_by_sector):
            return path
        centers = (np.arange(args.steering_sectors, dtype=np.float64) + 0.5) / args.steering_sectors
        xp = np.r_[centers[-1] - 1.0, centers, centers[0] + 1.0]
        fp = np.r_[offsets_by_sector[-1], offsets_by_sector,
                   offsets_by_sector[0]]
        route_progress = np.arange(len(points), dtype=np.float64) / len(points)
        offsets = np.interp(route_progress, xp, fp)
        if smoothing_sectors > 0.0:
            sigma_points = smoothing_sectors * len(points) / args.steering_sectors
            offsets = gaussian_filter1d(offsets, sigma=sigma_points, mode="wrap")
        tangent = np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1.0e-9)
        left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        return path + offsets[:, None] * left_normal

    # This is the measured/corrected reference geometry. Geometry-conditioned
    # speed changes must compare against this path, not the raw input line;
    # otherwise the already-applied tracking correction is counted as a new
    # optimizer geometry change.
    ordered_xy = apply_lateral_offsets(
        ordered_xy, sector_offsets, 0.0)
    base_ordered_xy = ordered_xy.copy()
    ordered_xy = apply_lateral_offsets(
        ordered_xy, sector_offset_adjustments,
        args.sector_offset_adjustment_smoothing_sectors)

    # The Unity runner applies raceline_target_speed_scale once at runtime.
    # Pointwise values in the JSON therefore remain unscaled here.
    curvature_speed_scales = np.ones(len(points), dtype=np.float64)
    if args.curvature_speed_adaptation and not args.empirical_speed_envelope:
        base_curvature, _ = smooth_closed_curvature(
            base_ordered_xy, args.curvature_smoothing_distance_m)
        candidate_curvature, _ = smooth_closed_curvature(
            ordered_xy, args.curvature_smoothing_distance_m)
        base_curvature = np.abs(base_curvature)
        candidate_curvature = np.abs(candidate_curvature)
        supported = base_curvature >= 0.20
        relative_cap = np.sqrt(
            base_curvature[supported] /
            np.maximum(candidate_curvature[supported], 1.0e-6))
        # The global scale is only an upper bound on geometry-earned speed.
        # Preserve the proven target where curvature is unchanged/unsupported;
        # otherwise a requested geometry adaptation would silently become a
        # global speed push on straights, where prior Unity trials showed
        # wheelspin without a body-speed gain.
        curvature_speed_scales.fill(1.0 / args.speed_scale)
        curvature_speed_scales[supported] = np.minimum(
            1.0, relative_cap / args.speed_scale)
        mean_spacing = float(np.mean(np.linalg.norm(
            np.roll(ordered_xy, -1, axis=0) - ordered_xy, axis=1)))
        if args.curvature_smoothing_distance_m > 0.0 and mean_spacing > 0.0:
            curvature_speed_scales = gaussian_filter1d(
                curvature_speed_scales,
                sigma=args.curvature_smoothing_distance_m / mean_spacing,
                mode="wrap")

    closed_segment_lengths = np.linalg.norm(
        np.roll(ordered_xy, -1, axis=0) - ordered_xy, axis=1)
    segment_lengths = closed_segment_lengths[:-1]
    route_arc = np.r_[0.0, np.cumsum(segment_lengths)]
    path_length = float(np.sum(closed_segment_lengths))

    empirical_speed_plan: np.ndarray | None = None
    smoothed_curvature: np.ndarray | None = None
    if args.empirical_speed_envelope:
        smoothed_curvature, _ = smooth_closed_curvature(
            ordered_xy, args.curvature_smoothing_distance_m)
        lateral_speed_cap = np.sqrt(
            args.lateral_acceleration_limit_mps2 /
            np.maximum(np.abs(smoothed_curvature), 1.0e-9))
        nominal_base_speed = np.asarray([
            points[trajectory_index]["speed"] *
            point_speed_scales[route_index]
            for route_index, trajectory_index in enumerate(order)
        ], dtype=np.float64)
        nominal_speed = nominal_base_speed * args.speed_scale
        effective_speed_plan = longitudinal_speed_envelope(
            np.minimum(nominal_speed, lateral_speed_cap),
            closed_segment_lengths,
            args.longitudinal_acceleration_limit_mps2,
            args.longitudinal_braking_limit_mps2,
            args.gate_speed_mps)
        if (args.empirical_speed_envelope_start_progress > 0.0 or
                args.empirical_speed_envelope_blend_width > 0.0):
            route_progress = np.arange(len(points), dtype=np.float64) / len(points)
            if args.empirical_speed_envelope_blend_width == 0.0:
                envelope_fraction = (
                    route_progress >= args.empirical_speed_envelope_start_progress
                ).astype(np.float64)
            else:
                blend_progress = np.clip(
                    (route_progress -
                     args.empirical_speed_envelope_start_progress) /
                    args.empirical_speed_envelope_blend_width,
                    0.0, 1.0)
                envelope_fraction = blend_progress ** 2 * (
                    3.0 - 2.0 * blend_progress)
            effective_speed_plan = (
                (1.0 - envelope_fraction) * nominal_base_speed +
                envelope_fraction * effective_speed_plan)
        # The profile runner multiplies every point target by the global scale.
        empirical_speed_plan = effective_speed_plan / args.speed_scale

    commands: list[dict[str, object]] = []
    for command in warmup:
        commands.append({
            "steering_norm": float(command["steering_norm"]),
            "throttle_norm": float(command["throttle_norm"]),
            "has_reference_position": False,
            "reference_x_m": 0.0,
            "reference_z_m": 0.0,
            "has_raceline_target": False,
            "target_speed_mps": 0.0,
            "route_arc_m": 0.0,
        })

    ordered_progress = np.arange(len(points), dtype=np.float64)
    feedforward = np.interp(ordered_progress, capture_progress,
                            capture_throttle)
    for route_index, trajectory_index in enumerate(order):
        x_world, z_world = ordered_xy[route_index]
        speed_sector = min(
            route_index * args.steering_sectors // len(points),
            args.steering_sectors - 1)
        commands.append({
            "steering_norm": 0.0,
            "throttle_norm": float(np.clip(feedforward[route_index], 0.0, 1.0)),
            "has_reference_position": True,
            "reference_x_m": float(x_world),
            "reference_z_m": float(z_world),
            "has_raceline_target": True,
            "target_speed_mps": float(
                empirical_speed_plan[route_index]
                if empirical_speed_plan is not None else
                points[trajectory_index]["speed"] *
                point_speed_scales[route_index] *
                curvature_speed_scales[route_index]),
            "route_arc_m": float(route_arc[route_index]),
            "steering_gain_scale": float(
                sector_steering_gains[speed_sector]),
            "speed_feedback_gain_scale": float(
                sector_speed_feedback_scales[speed_sector]),
            "positive_forward_slip_feedback_gain_scale": float(
                sector_positive_slip_feedback_scales[speed_sector]),
        })

    return {
        "schema_version": 1,
        "fixed_delta_time_s": 0.001,
        "physics_ticks_per_command": 25,
        "command_phase_fixed_steps": 0,
        "spatial_command_lookup": True,
        "spatial_progress_distance_scale": args.progress_distance_scale,
        "warmup_command_count": warmup_count,
        "scored_horizon_commands": args.maximum_lap_commands,
        "raceline_feedback_control": True,
        "raceline_signed_throttle_feedback": args.signed_throttle_feedback,
        "raceline_path_length_m": path_length,
        "raceline_target_speed_scale": args.speed_scale,
        "raceline_target_speed_scale_application":
            "runtime_once_after_point_target",
        "raceline_curvature_speed_adaptation":
            args.curvature_speed_adaptation,
        "raceline_empirical_speed_envelope": args.empirical_speed_envelope,
        "raceline_empirical_speed_envelope_start_progress": (
            args.empirical_speed_envelope_start_progress
            if args.empirical_speed_envelope else None),
        "raceline_empirical_speed_envelope_blend_width": (
            args.empirical_speed_envelope_blend_width
            if args.empirical_speed_envelope else None),
        "raceline_curvature_smoothing_distance_m": (
            args.curvature_smoothing_distance_m
            if (args.empirical_speed_envelope or
                args.curvature_speed_adaptation) else None),
        "raceline_lateral_acceleration_limit_mps2": (
            args.lateral_acceleration_limit_mps2
            if args.empirical_speed_envelope else None),
        "raceline_longitudinal_acceleration_limit_mps2": (
            args.longitudinal_acceleration_limit_mps2
            if args.empirical_speed_envelope else None),
        "raceline_longitudinal_braking_limit_mps2": (
            args.longitudinal_braking_limit_mps2
            if args.empirical_speed_envelope else None),
        "raceline_gate_speed_mps": (
            args.gate_speed_mps if args.empirical_speed_envelope else None),
        "raceline_sector_speed_scales": sector_speed_scales.tolist(),
        "raceline_sector_speed_scale_interpolation":
            "linear_between_sector_centers",
        "raceline_sector_steering_gain_scales":
            sector_steering_gains.tolist(),
        "raceline_sector_speed_feedback_scales":
            sector_speed_feedback_scales.tolist(),
        "raceline_sector_positive_forward_slip_feedback_scales":
            sector_positive_slip_feedback_scales.tolist(),
        "raceline_lookahead_base_m": args.lookahead_base_m,
        "raceline_lookahead_speed_s": args.lookahead_speed_s,
        "raceline_effective_wheelbase_m": args.effective_wheelbase_m,
        "raceline_speed_feedback_gain": args.speed_feedback_gain,
        "raceline_steering_gain": args.steering_gain,
        "raceline_positive_forward_slip_target":
            args.positive_forward_slip_target,
        "raceline_positive_forward_slip_feedback_gain":
            args.positive_forward_slip_feedback_gain,
        "path_correction_capture": (None if args.correction_capture is None
                                     else str(args.correction_capture)),
        "offset_correction_fraction": args.offset_correction_fraction,
        "offset_smoothing_sectors": args.offset_smoothing_sectors,
        "sector_offset_adjustment_smoothing_sectors":
            args.sector_offset_adjustment_smoothing_sectors,
        "sector_offset_adjustments_m": sector_offset_adjustments.tolist(),
        "offset_sector_m": (sector_offsets + sector_offset_adjustments).tolist(),
        "trajectory_path": str(args.trajectory),
        "warmup_source_profile": str(args.warmup_profile),
        "throttle_feedforward_capture": str(args.capture),
        "commands": commands,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--warmup-profile", type=Path,
                        default=DEFAULT_WARMUP_PROFILE)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--correction-capture", type=Path)
    parser.add_argument("--offset-correction-fraction", type=float, default=0.0)
    parser.add_argument("--offset-smoothing-sectors", type=float, default=1.2)
    parser.add_argument("--steering-sectors", type=int, default=12)
    parser.add_argument(
        "--sector-offset-adjustments-m",
        help="optional comma-separated per-sector lateral path offsets in metres")
    parser.add_argument(
        "--sector-offset-adjustment-smoothing-sectors", type=float, default=0.0,
        help="periodic Gaussian sigma for smoothing added sector offsets")
    parser.add_argument(
        "--sector-speed-scales",
        help="optional comma-separated per-sector target-speed multipliers")
    parser.add_argument(
        "--sector-steering-gains",
        help="optional comma-separated per-sector steering gain multipliers")
    parser.add_argument(
        "--sector-speed-feedback-scales",
        help="optional comma-separated per-sector speed-feedback multipliers")
    parser.add_argument(
        "--sector-positive-forward-slip-feedback-scales",
        help="optional nonnegative per-sector measured-slip throttle-feedback multipliers")
    parser.add_argument("--start-index", type=int, default=894)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--curvature-speed-adaptation", action="store_true",
        help="preserve the proven speed schedule and permit geometry-earned speed from reduced smoothed curvature")
    parser.add_argument(
        "--empirical-speed-envelope", action="store_true",
        help="smooth measured path curvature and enforce empirical lateral, acceleration, and braking limits")
    parser.add_argument(
        "--empirical-speed-envelope-start-progress", type=float, default=0.0,
        help="route fraction where the empirical speed plan starts replacing nominal trajectory targets")
    parser.add_argument(
        "--empirical-speed-envelope-blend-width", type=float, default=0.0,
        help="route fraction for a smoothstep transition from nominal to empirical speed targets")
    parser.add_argument(
        "--curvature-smoothing-distance-m", type=float, default=0.60,
        help="along-path Gaussian sigma; 0.60 m was selected by held-reset Unity yaw fit")
    parser.add_argument(
        "--lateral-acceleration-limit-mps2", type=float, default=9.96,
        help="95th percentile of successful r115/r116 Unity u*|yaw_rate|")
    parser.add_argument(
        "--longitudinal-acceleration-limit-mps2", type=float, default=5.239,
        help="95th percentile of positive 25 ms forward acceleration in r115/r116")
    parser.add_argument(
        "--longitudinal-braking-limit-mps2", type=float, default=7.587,
        help="95th percentile braking magnitude from successful r115/r116")
    parser.add_argument(
        "--gate-speed-mps", type=float, default=4.414,
        help="measured Unity forward speed at the established lap gate")
    parser.add_argument("--lookahead-base-m", type=float, default=0.35)
    parser.add_argument("--lookahead-speed-s", type=float, default=0.12)
    parser.add_argument("--effective-wheelbase-m", type=float, default=0.32)
    parser.add_argument("--speed-feedback-gain", type=float, default=0.02)
    parser.add_argument("--positive-forward-slip-target", type=float, default=0.0,
                        help="Unity positive forward-slip deadband for the development throttle limiter")
    parser.add_argument("--positive-forward-slip-feedback-gain", type=float,
                        default=0.0,
                        help="Development throttle reduction per mean positive-slip excess")
    parser.add_argument(
        "--signed-throttle-feedback", action="store_true",
        help="allow negative speed feedback to command proportional reverse motor torque")
    parser.add_argument("--steering-gain", type=float, default=1.0)
    parser.add_argument("--progress-distance-scale", type=float, default=3.0)
    parser.add_argument("--maximum-lap-commands", type=int, default=800)
    args = parser.parse_args()
    if (args.speed_scale <= 0 or args.maximum_lap_commands <= 0 or
            args.steering_sectors <= 0 or
            not 0.0 <= args.offset_correction_fraction <= 1.0 or
            args.offset_smoothing_sectors < 0.0 or
            not np.isfinite(args.sector_offset_adjustment_smoothing_sectors) or
            args.sector_offset_adjustment_smoothing_sectors < 0.0 or
            (args.curvature_speed_adaptation and
             args.empirical_speed_envelope) or
            not np.isfinite((args.curvature_smoothing_distance_m,
                             args.lateral_acceleration_limit_mps2,
                             args.longitudinal_acceleration_limit_mps2,
                             args.longitudinal_braking_limit_mps2,
                             args.gate_speed_mps,
                             args.empirical_speed_envelope_start_progress,
                             args.empirical_speed_envelope_blend_width)).all() or
            args.curvature_smoothing_distance_m < 0.0 or
            not 0.0 <= args.empirical_speed_envelope_start_progress <= 1.0 or
            not 0.0 <= args.empirical_speed_envelope_blend_width <= 1.0 or
            (args.empirical_speed_envelope and
             args.empirical_speed_envelope_start_progress +
             args.empirical_speed_envelope_blend_width > 1.0) or
            (not args.empirical_speed_envelope and
             (args.empirical_speed_envelope_start_progress > 0.0 or
              args.empirical_speed_envelope_blend_width > 0.0)) or
            min(args.lateral_acceleration_limit_mps2,
                args.longitudinal_acceleration_limit_mps2,
                args.longitudinal_braking_limit_mps2,
                args.gate_speed_mps) <= 0.0 or
            args.positive_forward_slip_target < 0.0 or
            args.positive_forward_slip_feedback_gain < 0.0 or
            not np.isfinite(args.positive_forward_slip_target) or
            not np.isfinite(args.positive_forward_slip_feedback_gain)):
        parser.error("invalid speed, horizon, sector, offset, or positive-slip settings")
    profile = build_profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.output),
        "commands": len(profile["commands"]),
        "warmup_commands": profile["warmup_command_count"],
        "raceline_points": len(profile["commands"]) -
                           profile["warmup_command_count"],
        "path_length_m": profile["raceline_path_length_m"],
        "speed_scale": profile["raceline_target_speed_scale"],
        "curvature_speed_adaptation":
            profile["raceline_curvature_speed_adaptation"],
        "empirical_speed_envelope":
            profile["raceline_empirical_speed_envelope"],
        "empirical_speed_envelope_start_progress":
            profile["raceline_empirical_speed_envelope_start_progress"],
        "empirical_speed_envelope_blend_width":
            profile["raceline_empirical_speed_envelope_blend_width"],
        "curvature_smoothing_distance_m":
            profile["raceline_curvature_smoothing_distance_m"],
        "lateral_acceleration_limit_mps2":
            profile["raceline_lateral_acceleration_limit_mps2"],
        "longitudinal_acceleration_limit_mps2":
            profile["raceline_longitudinal_acceleration_limit_mps2"],
        "longitudinal_braking_limit_mps2":
            profile["raceline_longitudinal_braking_limit_mps2"],
        "gate_speed_mps": profile["raceline_gate_speed_mps"],
        "sector_speed_scales": profile["raceline_sector_speed_scales"],
        "sector_steering_gain_scales":
            profile["raceline_sector_steering_gain_scales"],
        "sector_speed_feedback_scales":
            profile["raceline_sector_speed_feedback_scales"],
        "sector_positive_forward_slip_feedback_scales":
            profile["raceline_sector_positive_forward_slip_feedback_scales"],
        "feedback_speed_gain": profile["raceline_speed_feedback_gain"],
        "positive_forward_slip_target":
            profile["raceline_positive_forward_slip_target"],
        "positive_forward_slip_feedback_gain":
            profile["raceline_positive_forward_slip_feedback_gain"],
        "signed_throttle_feedback":
            profile["raceline_signed_throttle_feedback"],
        "path_correction_capture": profile["path_correction_capture"],
        "offset_correction_fraction": profile["offset_correction_fraction"],
        "offset_smoothing_sectors": profile["offset_smoothing_sectors"],
        "sector_offset_adjustment_smoothing_sectors":
            profile["sector_offset_adjustment_smoothing_sectors"],
        "sector_offset_adjustments_m":
            profile["sector_offset_adjustments_m"],
        "offset_sector_m": profile["offset_sector_m"],
    }, indent=2))


if __name__ == "__main__":
    main()
