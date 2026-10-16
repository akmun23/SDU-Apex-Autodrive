#!/usr/bin/env python3
"""Generate a new free-geometry hybrid Unity-dynamics path and speed plan.

This is a research optimizer, separate from the production trajectory and the
older section-scaled candidates. It jointly solves path offset, body motion,
four wheel speeds, steering response, and Unity's exact-zero brake/drive mode.
No production raceline or sector multiplier constrains path geometry or speed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.optimizer_unity_wheels import (
    FEASIBILITY_MAX_ITERATIONS,
    NEAR_FEASIBLE_WARM_START_MAX_VIOLATION,
    TIME_OPTIMIZATION_MAX_ITERATIONS,
    DEFAULT_TIME_TRUST_REGION_FACTORS,
    _brake_phases,
    solve_bottom_up,
    validate_time_trust_region_factors,
    write_candidate,
)
from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.track import (
    Track,
    load_centerline_csv,
)
from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.unity_wheel_model import (
    ROOT,
    UnityWheelModel,
    _wheel_surface_speed_from_physx_long_slip,
)
from tools.racing.track_projection import TrackProjection
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
    WHEEL_X_M,
    WHEEL_Y_M,
    ackermann_angles,
    ackermann_center_from_wheel_angles,
    STEERING_RATE_RADPS,
)


DEFAULT_CENTERLINE = ROOT / (
    "live_runs/racelines/icra_practice_20260924_b/SmoothCenterline.csv")
DEFAULT_TEMPLATE = ROOT / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02/"
    "raceline_feedback_profile.json")
DEFAULT_BRAKE_CAPTURE = ROOT / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02/dynamics/"
    "wheel_dynamics_joint_geometry_curvature_r02.csv")
DEFAULT_RUN_RESULT = ROOT / (
    "live_runs/unity_source_joint_geometry_curvature_20261014_r02/dynamics/"
    "offline_result.json")
DEFAULT_WALL_CLEARANCE_M = 0.05  # one cell of the 0.05 m practice map
UNITY_STEERING_LIMIT_RAD = np.deg2rad(30.0)
CONTROL_DT_S = 0.025


def _reference_lap_interval(run_result: Path, profile: dict,
                            interval_s: tuple[float, float] | None
                            ) -> tuple[float, float]:
    if interval_s is None:
        result = json.loads(run_result.read_text(encoding="utf-8"))
        start_s = int(profile["warmup_command_count"]) * CONTROL_DT_S
        stop_s = start_s + float(result["scored_duration_s"])
    else:
        start_s, stop_s = map(float, interval_s)
    if (not np.isfinite((start_s, stop_s)).all() or
            start_s < 0.0 or stop_s <= start_s):
        raise ValueError("reference lap interval must be finite and increasing")
    return start_s, stop_s


def unity_steering_norm_from_map_angle(steering_rad: float) -> float:
    """Convert map-frame road-wheel angle to Unity's normalized command.

    The optimizer map is a 90-degree rotation of Unity X/Z, so its positive
    steering sign is opposite the Unity road-wheel angle. VehicleController
    also negates AutonomousSteering when producing that road-wheel angle;
    the two sign changes cancel.
    """
    return float(steering_rad / UNITY_STEERING_LIMIT_RAD)


def tracking_bias_values_for_track(track: Track, tracking_metrics: dict
                                   ) -> tuple[np.ndarray, dict]:
    """Map a validated 12-sector signed path residual onto track stations."""
    sectors = tracking_metrics.get("path_bias_by_route_fraction_12")
    if not isinstance(sectors, list) or len(sectors) != 12:
        raise ValueError("tracking-bias run must cover all 12 route sectors")
    centers = np.asarray([float(row["route_fraction"]) for row in sectors])
    bias = np.asarray([
        float(row["median_signed_path_error_m"]) for row in sectors
    ])
    samples = np.asarray([int(row["sample_count"]) for row in sectors])
    if (not np.isfinite(centers).all() or not np.isfinite(bias).all()
            or np.any(samples < 5)):
        raise ValueError("tracking-bias training needs at least 5 samples per sector")
    order = np.argsort(centers)
    centers = centers[order]
    bias = bias[order]
    periodic_centers = np.r_[centers - 1.0, centers, centers + 1.0]
    periodic_bias = np.tile(bias, 3)
    fractions = np.mod(track.s / track.length, 1.0)
    values = np.interp(fractions, periodic_centers, periodic_bias)
    return values, {
        "route_fraction_centers": centers.tolist(),
        "median_signed_path_error_m": bias.tolist(),
        "sample_count_per_sector": samples[order].tolist(),
        "max_abs_bias_m": float(np.max(np.abs(bias))),
    }


def tracking_bias_metrics_from_run(run_dir: Path, centerline: Path) -> tuple[dict, dict]:
    """Read a completed practice candidate without launching Unity."""
    from tools.racing.validate_unity_wheel_raceline import (
        _prediction_tracking_metrics,
    )

    manifest_path = run_dir / "candidate_manifest.json"
    result_path = run_dir / "unity_validation.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise ValueError(f"tracking-bias run is missing its manifests: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_centerline = Path(manifest.get("centerline", "")).resolve()
    if source_centerline != centerline.resolve():
        raise ValueError("tracking-bias run used a different centerline")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gates = result.get("forward_gate_crossings_s", [])
    capture = Path(result.get("capture_path", ""))
    if (len(gates) < 2 or result.get("collisions_observed") != 0
            or result.get("reference_lap_progress_fraction", 0.0) < 1.0
            or not result.get("unity_lap_time_s") or not capture.is_file()):
        raise ValueError("tracking-bias run must be a completed collision-free lap")
    states = run_dir / "unity_wheel_hybrid_states.csv"
    if not states.is_file():
        raise ValueError("tracking-bias run lacks saved OCP states")
    metrics = _prediction_tracking_metrics(
        capture, states, float(gates[0]), float(gates[1]))
    return metrics, {
        "candidate_run": str(run_dir.resolve()),
        "capture": str(capture.resolve()),
        "unity_lap_time_s": float(result["unity_lap_time_s"]),
        "collisions_observed": 0,
    }


def time_parameterize_candidate(candidate: dict[str, np.ndarray], track
                                ) -> tuple[dict[str, np.ndarray], float]:
    """Sample the spatial OCP solution at the Unity 40 Hz command times.

    The command profile is consumed in time order. Spreading one lap uniformly
    over the template's (much longer) command list makes its controls lag the
    vehicle by several metres. Build the exact OCP time density first, then
    interpolate states by elapsed time and hold each spatial control until its
    next OCP node.
    """
    names = ("s_ref_m", "ey_m", "epsi_rad", "speed_mps",
             "yaw_rate_radps",
             "lateral_speed_mps", "steering_rad", "throttle_norm", "x_m", "y_m")
    values = {name: np.asarray(candidate[name], dtype=np.float64) for name in names}
    count = len(values["s_ref_m"])
    if count < 3 or any(value.shape != (count,) for value in values.values()):
        raise ValueError("OCP candidate arrays must be finite one-dimensional node values")
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("OCP candidate contains non-finite state or control values")

    station = values["s_ref_m"]
    if (np.any(np.diff(station) <= 0.0) or station[0] < -1.0e-8
            or station[-1] >= track.length):
        raise ValueError("OCP candidate stations must increase within the track lap")
    ds = np.diff(np.r_[station, track.length])
    if np.any(ds <= 0.0):
        raise ValueError("OCP candidate has a non-positive spatial interval")

    track_s = np.r_[track.s, track.length]
    track_kappa = np.r_[track.kappa, track.kappa[0]]
    curvature = np.interp(station, track_s, track_kappa)
    denominator = 1.0 - curvature * values["ey_m"]
    sdot = ((values["speed_mps"] * np.cos(values["epsi_rad"])
             - values["lateral_speed_mps"] * np.sin(values["epsi_rad"]))
            / denominator)
    if (np.any(denominator <= 0.0) or not np.isfinite(sdot).all()
            or np.any(sdot <= 0.0)):
        raise ValueError("OCP solution has invalid forward progress for time sampling")

    density = 1.0 / sdot
    segment_time = 0.5 * ds * (density + np.roll(density, -1))
    lap_time = float(np.sum(segment_time))
    if not np.isfinite(lap_time) or lap_time <= 0.0:
        raise ValueError("OCP solution has an invalid lap-time integral")
    node_time = np.r_[0.0, np.cumsum(segment_time[:-1])]
    command_count = int(np.ceil(lap_time / CONTROL_DT_S - 1.0e-12))
    sample_time = np.arange(command_count, dtype=np.float64) * CONTROL_DT_S
    periodic_time = np.r_[node_time, lap_time]

    def interpolate_periodic(array: np.ndarray) -> np.ndarray:
        return np.interp(sample_time, periodic_time,
                         np.r_[array, array[0]])

    control_nodes = np.searchsorted(node_time, sample_time, side="right") - 1
    control_nodes = np.clip(control_nodes, 0, count - 1)
    path_step = np.hypot(np.diff(values["x_m"]), np.diff(values["y_m"]))
    path_length = float(np.sum(path_step) + np.hypot(
        values["x_m"][0] - values["x_m"][-1],
        values["y_m"][0] - values["y_m"][-1]))
    path_arc = np.r_[0.0, np.cumsum(path_step)]

    sampled = {
        "sample_time_s": sample_time,
        "x_m": interpolate_periodic(values["x_m"]),
        "y_m": interpolate_periodic(values["y_m"]),
        "speed_mps": interpolate_periodic(values["speed_mps"]),
        "lateral_speed_mps": interpolate_periodic(
            values["lateral_speed_mps"]),
        "yaw_rate_radps": interpolate_periodic(values["yaw_rate_radps"]),
        "ey_m": interpolate_periodic(values["ey_m"]),
        "epsi_rad": interpolate_periodic(values["epsi_rad"]),
        "steering_rad": interpolate_periodic(values["steering_rad"]),
        "throttle_norm": values["throttle_norm"][control_nodes],
        "route_arc_m": np.interp(sample_time, periodic_time,
                                  np.r_[path_arc, path_length]),
    }
    return sampled, lap_time


def rotate_track_to_profile_gate(track: Track, profile: dict) -> Track:
    warmup = int(profile["warmup_command_count"])
    route = profile["commands"][warmup:]
    if not route:
        raise ValueError("profile has no route-start point")
    # Profile coordinates are Unity world X/Z; the optimizer uses the map
    # frame used by the established practice-track pipeline.
    gate_x = float(route[0]["reference_z_m"]) + 0.24
    gate_y = -float(route[0]["reference_x_m"]) + 0.05
    index = int(np.argmin((track.x - gate_x) ** 2 + (track.y - gate_y) ** 2))
    order = np.roll(np.arange(track.count), -index)
    x = track.x[order]
    y = track.y[order]
    ds = track.ds[order]
    psi = np.unwrap(np.arctan2(np.roll(y, -1) - np.roll(y, 1),
                               np.roll(x, -1) - np.roll(x, 1)))
    s = np.r_[0.0, np.cumsum(ds[:-1])]
    return Track(s=s, x=x, y=y, psi=psi, kappa=track.kappa[order],
                 left=track.left[order], right=track.right[order],
                 ds=ds, length=track.length)


def path_reference_offsets_for_track(
        track: Track, reference_path: Path, *, maximum_distance_m: float = 0.75
        ) -> tuple[np.ndarray, dict]:
    """Return the reference trajectory's Frenet offset at each centerline node.

    The optimizer's ``ey`` is positive left of the centerline.  Projection
    reports the centerline's signed offset from the reference, hence the sign
    inversion below. Sequential local projection avoids switching branches at
    nearby track crossings; a global retry is allowed only when the local
    branch is farther than the explicit correspondence tolerance.
    """
    if not np.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0:
        raise ValueError("reference correspondence distance must be finite and positive")
    reference = TrackProjection.from_csv(reference_path, closed=True)
    offsets = np.empty(track.count, dtype=np.float64)
    distances = np.empty(track.count, dtype=np.float64)
    previous_segment = None
    for index, (x_m, y_m, heading_rad) in enumerate(
            zip(track.x, track.y, track.psi)):
        projection = reference.project(
            float(x_m), float(y_m), float(heading_rad),
            previous_segment=previous_segment,
            local_search_radius=(24 if previous_segment is not None else None))
        if projection.distance_m > maximum_distance_m:
            projection = reference.project(
                float(x_m), float(y_m), float(heading_rad))
        if projection.distance_m > maximum_distance_m:
            raise ValueError(
                "reference path is not a close, unambiguous route match: "
                f"centerline node {index} is {projection.distance_m:.3f} m away "
                f"(limit {maximum_distance_m:.3f} m)")
        offsets[index] = -projection.lateral_offset_m
        distances[index] = projection.distance_m
        previous_segment = projection.segment_index
    metadata = {
        "reference_path": str(reference_path.resolve()),
        "maximum_correspondence_distance_m": float(maximum_distance_m),
        "max_centerline_to_reference_distance_m": float(np.max(distances)),
        "rms_centerline_to_reference_distance_m": float(
            np.sqrt(np.mean(distances ** 2))),
        "reference_offset_min_m": float(np.min(offsets)),
        "reference_offset_max_m": float(np.max(offsets)),
        "reference_offset_rms_m": float(np.sqrt(np.mean(offsets ** 2))),
    }
    return offsets, metadata


def measured_brake_mode_mask(capture: Path, run_result: Path,
                             profile: dict, track: Track,
                             reference_interval_s: tuple[float, float] | None = None
                             ) -> tuple[np.ndarray, dict]:
    start_s, stop_s = _reference_lap_interval(
        run_result, profile, reference_interval_s)
    with capture.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(line for line in stream if not line.startswith("#"))
        header = next(reader)
        columns = {name: index for index, name in enumerate(header)}
        required = ("sim_time_s", "pos_world_x_m", "pos_world_z_m",
                    "throttle_command")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"brake-mode capture lacks columns: {missing}")
        grouped: dict[int, tuple[float, float, bool]] = {}
        for row in reader:
            timestamp = float(row[columns["sim_time_s"]])
            if timestamp < start_s or timestamp > stop_s:
                continue
            step = int(round((timestamp - start_s) / 0.025))
            if step in grouped:
                continue
            grouped[step] = (
                float(row[columns["pos_world_z_m"]]) + 0.24,
                -float(row[columns["pos_world_x_m"]]) + 0.05,
                abs(float(row[columns["throttle_command"]])) < 1.0e-9)

    if len(grouped) < 100:
        raise ValueError("measured lap does not contain enough 40 Hz commands")
    ordered = [grouped[key] for key in sorted(grouped)]
    xy = np.asarray([[item[0], item[1]] for item in ordered])
    nearest = cKDTree(np.column_stack((track.x, track.y))).query(xy, k=1)[1]
    progress = track.s[nearest].astype(np.float64)
    for index in range(1, len(progress)):
        while progress[index] < progress[index - 1] - track.length / 2.0:
            progress[index] += track.length
        while progress[index] > progress[index - 1] + track.length / 2.0:
            progress[index] -= track.length
    progress -= progress[0]
    progress = np.maximum.accumulate(progress)
    if progress[-1] < 0.85 * track.length:
        raise ValueError(
            f"capture projects only {progress[-1]:.2f}/{track.length:.2f} m "
            "around the rotated centerline; refusing a misaligned brake schedule")

    route_mask = np.asarray([item[2] for item in ordered], dtype=bool)
    mask = np.zeros(track.count, dtype=bool)
    for node, node_s in enumerate(track.s):
        distance = np.abs(progress - node_s)
        distance = np.minimum(distance, np.abs(distance - track.length))
        mask[node] = route_mask[int(np.argmin(distance))]
    return mask, {
        "capture": str(capture),
        "run_result": str(run_result),
        "reference_lap_start_s": start_s,
        "reference_lap_stop_s": stop_s,
        "reference_command_intervals": len(ordered),
        "reference_exact_zero_intervals": int(np.count_nonzero(route_mask)),
        "mapped_brake_nodes": int(np.count_nonzero(mask)),
        "centerline_progress_m": float(progress[-1]),
    }


def retain_longest_brake_events(track: Track, brake_mask: np.ndarray,
                                event_count: int
                                ) -> tuple[np.ndarray, dict]:
    """Build a smaller discrete mode topology from the longest measured events."""
    mask = np.asarray(brake_mask, dtype=bool)
    if (mask.shape != (track.count,) or mask[0] or mask[-1]
            or event_count <= 0):
        raise ValueError("brake event selection needs a valid gate and positive count")
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    bounds = np.r_[0, changes, track.count]
    events = []
    for begin, end in zip(bounds[:-1], bounds[1:]):
        if mask[begin]:
            events.append({
                "begin_node": int(begin),
                "end_node_exclusive": int(end),
                "length_m": float(np.sum(track.ds[begin:end])),
            })
    if not events:
        raise ValueError("measured brake topology contains no brake event")
    retained = sorted(events, key=lambda row: row["length_m"], reverse=True)[
        :event_count]
    selected = np.zeros(track.count, dtype=bool)
    for row in retained:
        selected[row["begin_node"]:row["end_node_exclusive"]] = True
    return selected, {
        "selection": "longest measured brake events by mapped track arc length",
        "requested_event_count": int(event_count),
        "measured_event_count": len(events),
        "retained_events": retained,
        "removed_events": [row for row in events if row not in retained],
    }


def captured_state_initial_guess(capture: Path, run_result: Path,
                                 profile: dict, track: Track,
                                 brake_mode_mask: np.ndarray,
                                 reference_interval_s: tuple[float, float] | None = None
                                 ) -> tuple[dict, dict]:
    start_s, stop_s = _reference_lap_interval(
        run_result, profile, reference_interval_s)
    with capture.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(line for line in stream if not line.startswith("#"))
        header = next(reader)
        columns = {name: index for index, name in enumerate(header)}
        fields = (
            "sim_time_s", "pos_world_x_m", "pos_world_z_m",
            "rot_world_x", "rot_world_y", "rot_world_z", "rot_world_w",
            "vel_body_x_mps", "vel_body_z_mps", "omega_body_y_rps",
            "throttle_command", "fl_rpm", "fr_rpm", "rl_rpm", "rr_rpm",
            "fl_steer_deg", "fr_steer_deg")
        missing = [name for name in fields if name not in columns]
        if missing:
            raise ValueError(f"state seed capture lacks columns: {missing}")
        samples: dict[int, list[float]] = {}
        for row in reader:
            timestamp = float(row[columns["sim_time_s"]])
            if timestamp < start_s or timestamp > stop_s:
                continue
            step = int(round((timestamp - start_s) / 0.025))
            if step in samples:
                continue
            samples[step] = [float(row[columns[name]]) for name in fields]

    data = np.asarray([samples[k] for k in sorted(samples)], dtype=np.float64)
    if len(data) < 100:
        raise ValueError("reference run has too few 40 Hz state samples")
    sample_time = data[:, 0]
    world_x, world_z = data[:, 1], data[:, 2]
    map_xy = np.column_stack((world_z + 0.24, -world_x + 0.05))
    tree = cKDTree(np.column_stack((track.x, track.y)))
    nearest = tree.query(map_xy, k=1)[1]

    def project_segment(segment_index: np.ndarray):
        start = np.column_stack((track.x[segment_index], track.y[segment_index]))
        end_index = (segment_index + 1) % track.count
        end = np.column_stack((track.x[end_index], track.y[end_index]))
        vector = end - start
        fraction = np.sum((map_xy - start) * vector, axis=1)
        fraction /= np.maximum(np.sum(vector * vector, axis=1), 1.0e-12)
        fraction = np.clip(fraction, 0.0, 1.0)
        projection = start + fraction[:, None] * vector
        distance = np.sum((map_xy - projection) ** 2, axis=1)
        progress = track.s[segment_index] + fraction * track.ds[segment_index]
        tangent = vector / np.maximum(np.linalg.norm(vector, axis=1)[:, None], 1e-12)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        lateral = np.sum((map_xy - projection) * normal, axis=1)
        return distance, progress, lateral, np.arctan2(tangent[:, 1], tangent[:, 0])

    first = project_segment((nearest - 1) % track.count)
    second = project_segment(nearest)
    choose_second = second[0] < first[0]
    progress = np.where(choose_second, second[1], first[1])
    lateral_offset = np.where(choose_second, second[2], first[2])
    track_heading = np.where(choose_second, second[3], first[3])
    for index in range(1, len(progress)):
        while progress[index] < progress[index - 1] - track.length / 2.0:
            progress[index] += track.length
        while progress[index] > progress[index - 1] + track.length / 2.0:
            progress[index] -= track.length
    progress -= progress[0]
    progress = np.maximum.accumulate(progress)
    if progress[-1] < 0.85 * track.length:
        raise ValueError("captured state seed does not cover a full centerline lap")

    qx, qy, qz, qw = data[:, 3:7].T
    forward_x = 2.0 * (qx * qz + qw * qy)
    forward_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    yaw_map = np.unwrap(np.arctan2(-forward_x, forward_z))
    heading_error = np.unwrap(yaw_map - track_heading)
    speed_u = data[:, 8]
    speed_v = -data[:, 7]
    yaw_rate = -data[:, 9]
    wheel_omega = data[:, 11:15].T * (2.0 * np.pi / 60.0)
    steering = np.asarray([
        ackermann_center_from_wheel_angles(
            -np.deg2rad(row[15]), -np.deg2rad(row[16]))
        for row in data], dtype=np.float64)
    steering_rate = np.gradient(steering, sample_time)
    ax = np.gradient(speed_u, sample_time) - yaw_rate * speed_v
    ay = np.gradient(speed_v, sample_time) + yaw_rate * speed_u
    yaw_acceleration = np.gradient(yaw_rate, sample_time)
    throttle = data[:, 10]
    node_s = track.s

    def periodic_interp(values: np.ndarray) -> np.ndarray:
        extension_s = np.r_[progress, track.length]
        extension_values = np.concatenate((values, values[:1]), axis=-1)
        return np.interp(node_s, extension_s, extension_values)

    seed = {
        "ey_m": periodic_interp(lateral_offset),
        "epsi_rad": periodic_interp(heading_error),
        "u_mps": periodic_interp(speed_u),
        "v_mps": periodic_interp(speed_v),
        "yaw_rate_radps": periodic_interp(yaw_rate),
        "wheel_omega_radps": np.vstack([
            periodic_interp(wheel_omega[i]) for i in range(4)]),
        "steering_rad": periodic_interp(steering),
        "steering_rate_radps": periodic_interp(steering_rate),
        "throttle_norm": periodic_interp(throttle),
        "ax_mps2": periodic_interp(ax),
        "ay_mps2": periodic_interp(ay),
        "yaw_accel_radps2": periodic_interp(yaw_acceleration),
    }
    seed_ax = seed["ax_mps2"]
    seed_ay = seed["ay_mps2"]
    long_transfer = (3.47 * 0.07434 * seed_ax / (2.0 * 0.324))
    lat_transfer = (3.47 * 0.07434 * seed_ay / (4.0 * 0.236))
    static_load = np.asarray((0.8171949, 0.8160193, 0.9189912,
                              0.9178206)) * 9.80665
    seed["normal_load_n"] = np.vstack((
        static_load[0] - long_transfer - lat_transfer,
        static_load[1] - long_transfer + lat_transfer,
        static_load[2] + long_transfer - lat_transfer,
        static_load[3] + long_transfer + lat_transfer,
    ))
    seed["normal_load_n"] = np.maximum(seed["normal_load_n"], 0.05)
    seed["wheel_omega_radps"][:, brake_mode_mask] = 0.0
    seed["throttle_norm"][brake_mode_mask] = 0.0
    seed["throttle_norm"][~brake_mode_mask] = np.maximum(
        0.015, seed["throttle_norm"][~brake_mode_mask])
    seed["steering_rate_radps"] = np.clip(
        seed["steering_rate_radps"], -STEERING_RATE_RADPS,
        STEERING_RATE_RADPS)
    return seed, {
        "source_capture": str(capture),
        "initialization_only": True,
        "sample_count_40hz": len(data),
        "projected_centerline_progress_m": float(progress[-1]),
        "periodic_warm_start_method": (
            "periodic extension of the selected capture window; no measured "
            "start-to-end drift is removed"),
        "reference_lap_start_s": start_s,
        "reference_lap_stop_s": stop_s,
        "raw_capture_end_minus_start": {
            "ey_m": float(lateral_offset[-1] - lateral_offset[0]),
            "epsi_rad": float(heading_error[-1] - heading_error[0]),
            "u_mps": float(speed_u[-1] - speed_u[0]),
            "v_mps": float(speed_v[-1] - speed_v[0]),
            "yaw_rate_radps": float(yaw_rate[-1] - yaw_rate[0]),
            "wheel_omega_radps": (wheel_omega[:, -1] -
                                   wheel_omega[:, 0]).tolist(),
        },
        "seed_speed_range_mps": [float(np.min(seed["u_mps"])),
                                 float(np.max(seed["u_mps"]))],
        "seed_lateral_offset_rms_m": float(np.sqrt(np.mean(seed["ey_m"] ** 2))),
        "seed_lateral_acceleration_p95_mps2": float(np.quantile(
            np.abs(seed["ay_mps2"]), 0.95)),
    }


def candidate_state_initial_guess(candidate_dir: Path, track: Track,
                                  model: UnityWheelModel
                                  ) -> tuple[dict[str, np.ndarray], np.ndarray,
                                             np.ndarray, dict]:
    """Load a feasible or explicitly bounded near-feasible OCP warm start."""
    report_path = candidate_dir / "optimizer_report.json"
    states_path = candidate_dir / "unity_wheel_hybrid_states.csv"
    if not report_path.is_file() or not states_path.is_file():
        raise FileNotFoundError(
            f"warm-start candidate requires {report_path} and {states_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prior_violation = report.get("nlp_diagnostics", {}).get(
        "max_constraint_violation")
    if (prior_violation is None or not np.isfinite(float(prior_violation)) or
            float(prior_violation) > NEAR_FEASIBLE_WARM_START_MAX_VIOLATION):
        raise ValueError(
            "warm-start candidate exceeds the allowed NLP constraint residual")
    phase_reports = report.get("brake_phase_seed_ranges_and_modes", [])
    phase_lengths = np.asarray(report.get("optimized_phase_lengths_m", []),
                               dtype=np.float64)
    if (not phase_reports or phase_lengths.shape != (len(phase_reports),)
            or not np.isfinite(phase_lengths).all()
            or not np.isclose(np.sum(phase_lengths), track.length,
                              rtol=0.0, atol=1.0e-5)):
        raise ValueError("warm-start candidate has invalid phase metadata")

    with states_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != track.count:
        raise ValueError(
            f"warm-start has {len(rows)} nodes; current track has {track.count}")
    required = (
        "s_ref_m", "ey_m", "epsi_rad", "speed_mps", "lateral_speed_mps",
        "yaw_rate_radps", "steering_rad", "steering_rate_radps",
        "throttle_norm", "ax_mps2", "ay_mps2", "normal_load_fl_n",
        "normal_load_fr_n", "normal_load_rl_n", "normal_load_rr_n",
        "slipx_fl", "slipx_fr", "slipx_rl", "slipx_rr",
    )
    if rows and any(name not in rows[0] for name in required):
        raise ValueError("warm-start candidate state CSV lacks required fields")
    values = {name: np.asarray([float(row[name]) for row in rows],
                               dtype=np.float64) for name in required}
    seed_s = values.pop("s_ref_m")
    if (not np.isfinite(seed_s).all() or np.any(np.diff(seed_s) <= 0.0)
            or seed_s[0] < -1.0e-8 or seed_s[-1] >= track.length):
        raise ValueError("warm-start candidate stations are not periodic/increasing")

    expected_s = []
    phase_offset = 0.0
    current_modes = []
    for phase, length in zip(phase_reports, phase_lengths):
        count = int(phase["optimized_mesh_nodes"])
        mode = str(phase["mode"])
        current_modes.append(mode)
        expected_s.extend(phase_offset + length * np.arange(count) / count)
        phase_offset += float(length)
    if [str(phase["mode"]) for phase in phase_reports] != current_modes:
        raise ValueError("warm-start phase metadata is inconsistent")
    if not np.allclose(seed_s, np.asarray(expected_s), rtol=0.0, atol=2.0e-7):
        raise ValueError("warm-start states do not match their optimized phase mesh")

    wheel_omega = []
    for index, wheel_name in enumerate(("fl", "fr", "rl", "rr")):
        steer_wheel = np.asarray([
            ackermann_angles(float(delta))[index]
            for delta in values["steering_rad"]], dtype=np.float64)
        vx = values["speed_mps"] - values["yaw_rate_radps"] * WHEEL_Y_M[index]
        vy = values["lateral_speed_mps"] + values["yaw_rate_radps"] * WHEEL_X_M[index]
        forward = np.cos(steer_wheel) * vx + np.sin(steer_wheel) * vy
        slip = values[f"slipx_{wheel_name}"]
        surface_speed = _wheel_surface_speed_from_physx_long_slip(
            forward, slip)
        omega = surface_speed / (model.wheel_radius_m * model.wheel_radius_scale)
        omega[values["throttle_norm"] == 0.0] = 0.0
        wheel_omega.append(omega)

    seed = {
        "ey_m": values["ey_m"],
        "epsi_rad": values["epsi_rad"],
        "u_mps": values["speed_mps"],
        "v_mps": values["lateral_speed_mps"],
        "yaw_rate_radps": values["yaw_rate_radps"],
        "wheel_omega_radps": np.vstack(wheel_omega),
        "steering_rad": values["steering_rad"],
        "steering_rate_radps": values["steering_rate_radps"],
        "throttle_norm": values["throttle_norm"],
        "ax_mps2": values["ax_mps2"],
        "ay_mps2": values["ay_mps2"],
        "normal_load_n": np.vstack([
            values[f"normal_load_{wheel}_n"]
            for wheel in ("fl", "fr", "rl", "rr")]),
    }
    body_state_columns = {
        "body_height_m": "body_height_m",
        "body_height_rate_mps": "body_height_rate_mps",
        "roll_rad": "roll_rad",
        "roll_rate_radps": "roll_rate_radps",
        "pitch_rad": "pitch_rad",
        "pitch_rate_radps": "pitch_rate_radps",
    }
    has_body_state = bool(rows) and all(
        column in rows[0] for column in body_state_columns.values())
    if has_body_state:
        seed.update({key: np.asarray([float(row[column]) for row in rows],
                                     dtype=np.float64)
                     for key, column in body_state_columns.items()})
    else:
        seed.update({
            "body_height_m": np.full(
                len(rows), model.suspension_reference_body_height_m),
            "body_height_rate_mps": np.zeros(len(rows)),
            "roll_rad": np.zeros(len(rows)),
            "roll_rate_radps": np.zeros(len(rows)),
            "pitch_rad": np.zeros(len(rows)),
            "pitch_rate_radps": np.zeros(len(rows)),
        })
    load_model_id = report.get("unity_vehicle_model", {}).get("load_model_id")
    prior_feasibility_reference = report.get("feasibility_objective_reference")
    if prior_feasibility_reference is None:
        prior_feasibility_reference = next((
            stage.get("feasibility_objective_value")
            for stage in report.get("solve_stages", [])
            if stage.get("feasibility_objective_value") is not None), None)
    return seed, seed_s, phase_lengths, {
        "source_candidate": str(candidate_dir),
        "node_count": len(rows),
        "phase_lengths_m": phase_lengths.tolist(),
        "phase_modes": current_modes,
        "phase_node_counts": [int(phase["optimized_mesh_nodes"])
                              for phase in phase_reports],
        "seed_reconstructed_wheel_speed": True,
        "model_consistent_seed": bool(
            has_body_state and load_model_id ==
            "unity_sprung_mass_spring_damper_heave_roll_pitch_v1"),
        "body_state_seed_available": has_body_state,
        "source_load_model_id": load_model_id,
        "minimum_trust_region_reference": prior_feasibility_reference,
        "source_max_constraint_violation": float(prior_violation),
    }


def _periodic_interp(source_s: np.ndarray, values: np.ndarray,
                     query_s: np.ndarray, length: float) -> np.ndarray:
    return np.interp(query_s, np.r_[source_s, length],
                     np.r_[values, values[0]])


def write_open_loop_profile(candidate: dict[str, np.ndarray],
                            report: dict, track, template: Path,
                            output_dir: Path) -> Path:
    profile = json.loads(template.read_text(encoding="utf-8"))
    if profile.get("raceline_feedback_control") is not True:
        raise ValueError(f"profile template is not a raceline profile: {template}")
    warmup = int(profile["warmup_command_count"])
    commands = profile["commands"]
    if len(commands) <= warmup:
        raise ValueError("profile template has no scored route commands")
    route, lap_time = time_parameterize_candidate(candidate, track)
    reported_time = report.get("predicted_lap_time_s")
    if (reported_time is not None and
            abs(float(reported_time) - lap_time) > 0.02):
        raise ValueError(
            "time-parameterized OCP states disagree with the optimizer lap-time "
            f"integral ({lap_time:.6f} vs {float(reported_time):.6f} s)")
    path_step = np.hypot(np.diff(candidate["x_m"]),
                         np.diff(candidate["y_m"]))
    path_length = float(np.sum(path_step) + np.hypot(
        candidate["x_m"][0] - candidate["x_m"][-1],
        candidate["y_m"][0] - candidate["y_m"][-1]))
    warmup_commands = [dict(row) for row in commands[:warmup]]
    scored_commands = []
    for i in range(len(route["sample_time_s"])):
        row = {
            "steering_norm": unity_steering_norm_from_map_angle(
                float(route["steering_rad"][i])),
            "throttle_norm": float(route["throttle_norm"][i]),
            "has_reference_position": True,
            "reference_x_m": float(0.05 - route["y_m"][i]),
            "reference_z_m": float(route["x_m"][i] - 0.24),
            "has_raceline_target": True,
            "target_speed_mps": float(route["speed_mps"][i]),
            "route_arc_m": float(route["route_arc_m"][i]),
            "steering_gain_scale": 1.0,
            "speed_feedback_gain_scale": 1.0,
            "positive_forward_slip_feedback_gain_scale": 0.0,
        }
        scored_commands.append(row)

    # The profile is a transport format for the new plan, not a sector-tuned
    # derivative of the old one. Keep the established feedback law and global
    # gains but remove all inherited sector speed/steering multipliers.
    profile["raceline_path_length_m"] = path_length
    profile["raceline_target_speed_scale"] = 1.0
    profile["raceline_curvature_speed_adaptation"] = False
    profile["raceline_sector_speed_scales"] = [1.0] * 12
    profile["raceline_sector_steering_gain_scales"] = [1.0] * 12
    profile["raceline_sector_speed_feedback_scales"] = [1.0] * 12
    profile["raceline_sector_positive_forward_slip_feedback_scales"] = [0.0] * 12
    profile["raceline_positive_forward_slip_feedback_gain"] = 0.0
    profile["raceline_speed_feedback_gain"] = 0.0
    profile["raceline_feedback_control"] = False
    profile["spatial_command_lookup"] = True
    profile["scored_horizon_commands"] = len(scored_commands)
    profile["raceline_signed_throttle_feedback"] = False
    profile["bottom_up_optimizer_source"] = (
        "Unity-source-calibrated per-wheel IPOPT free-geometry phase OCP")
    profile["bottom_up_optimizer_predicted_lap_time_s"] = lap_time
    profile["bottom_up_optimizer_time_parameterized"] = True
    profile["bottom_up_optimizer_time_sample_interval_s"] = CONTROL_DT_S
    profile["commands"] = warmup_commands + scored_commands
    path = output_dir / "unity_wheel_hybrid_open_loop_profile.json"
    path.write_text(json.dumps(profile, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centerline", type=Path, default=DEFAULT_CENTERLINE)
    parser.add_argument("--profile-template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--brake-capture", type=Path, default=DEFAULT_BRAKE_CAPTURE)
    parser.add_argument("--brake-run-result", type=Path, default=DEFAULT_RUN_RESULT)
    parser.add_argument(
        "--reference-lap-capture", type=Path,
        help="use this Unity capture window for both state and brake-mode seeds")
    parser.add_argument("--reference-lap-start-s", type=float,
                        help="absolute capture time at the selected lap gate")
    parser.add_argument("--reference-lap-stop-s", type=float,
                        help="absolute capture time at the selected lap end")
    parser.add_argument("--warm-start-candidate", type=Path,
                        help="reuse a prior feasible OCP candidate as the solver start")
    parser.add_argument(
        "--free-gate-state", action="store_true",
        help=("leave all lap-entry vehicle and Frenet states free; the "
              "default anchors the full state to the measured warm-up gate; "
              "periodic collocation still closes the lap"))
    parser.add_argument(
        "--gate-lateral-tolerance-m", type=float, default=0.0,
        help=("allow this small optimization-coordinate tolerance around the "
              "measured gate lateral offset; zero keeps an exact equality"))
    parser.add_argument(
        "--gate-steering-tolerance-rad", type=float, default=0.0,
        help=("allow a small steering-state tolerance around the measured "
              "gate value; zero keeps an exact equality"))
    parser.add_argument(
        "--max-body-path-heading-error-rad", type=float,
        help=("optional absolute limit on body heading relative to the "
              "optimized path tangent; use this to exclude planned drift "
              "that the production path-tracking MPC cannot follow"))
    parser.add_argument(
        "--max-abs-lateral-speed-mps", type=float,
        help=("optional absolute body lateral-speed limit, used with the "
              "body/path heading limit to penalize drift-dependent paths"))
    parser.add_argument(
        "--path-reference-trajectory", type=Path,
        help=("optional known-trackable trajectory that bounds optimized "
              "geometry while leaving the speed plan free"))
    parser.add_argument(
        "--max-path-reference-deviation-m", type=float,
        help=("maximum absolute lateral departure from the reference path; "
              "must be supplied with --path-reference-trajectory"))
    parser.add_argument(
        "--path-reference-max-correspondence-distance-m", type=float,
        default=0.75,
        help=("maximum centerline/reference separation accepted while "
              "mapping the route tube; default 0.75 m"))
    parser.add_argument(
        "--min-wheel-normal-load-n", type=float, default=0.05,
        help=("minimum predicted normal load at every wheel; default 0.05 N "
              "preserves the optimizer's prior non-lift floor"))
    parser.add_argument(
        "--feasibility-trust-region", type=float, default=25.0,
        help=("phase-I squared normalized deviation budget from the state "
              "seed; raise this when new constraints require a larger but "
              "physically feasible departure from captured motion"))
    parser.add_argument(
        "--positive-throttle-only", action="store_true",
        help=("force nonzero forward-drive mode at every node; tests whether "
              "the modeled tire/drag forces can meet speed reductions without "
              "Unity's exact-zero full-brake branch"))
    parser.add_argument(
        "--brake-event-count", type=int,
        help=("retain this many longest measured exact-zero brake events, "
              "testing an alternate discrete drive/brake topology"))
    parser.add_argument(
        "--throttle-fall-rate-limit-norm-per-sec", type=float,
        help=("optional maximum falling throttle rate; 0.267/s corresponds "
              "to the measured 0.08 throttle reduction over 0.30 s"))
    parser.add_argument(
        "--tracking-bias-training-run-dir", type=Path,
        help=("completed practice candidate used to fit a spatial path-bias "
              "model; affects candidate track-boundary constraints only"))
    parser.add_argument(
        "--tracking-bias-validation-run-dir", type=Path,
        help=("independent completed practice candidate required to check "
              "the path-bias transfer before it is applied"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.12)
    parser.add_argument("--min-speed-mps", type=float, default=0.35)
    parser.add_argument("--max-speed-mps", type=float, default=12.0)
    parser.add_argument(
        "--feasibility-max-iterations", type=int,
        default=FEASIBILITY_MAX_ITERATIONS,
        help=("bounded IPOPT feasibility-phase budget; default preserves the "
              "standard 600 iterations"))
    parser.add_argument(
        "--time-optimization-max-iterations", type=int,
        default=TIME_OPTIMIZATION_MAX_ITERATIONS,
        help=("IPOPT iteration budget for each expanding minimum-time stage"))
    parser.add_argument(
        "--time-hessian-approximation",
        choices=("limited-memory", "exact"), default="limited-memory",
        help=("IPOPT Hessian for minimum-time stages; 'exact' uses CasADi "
              "automatic differentiation, with the same model and constraints"))
    parser.add_argument(
        "--time-trust-region-factors", type=float, nargs="+",
        default=DEFAULT_TIME_TRUST_REGION_FACTORS,
        help=("strictly increasing trust-radius multipliers for the staged "
              "time-minimization continuation; default: 4 8 16 1000000"))
    parser.add_argument(
        "--wall-clearance-m", type=float, default=DEFAULT_WALL_CLEARANCE_M,
        help=("extra distance from the modeled car footprint to a wall; "
              "default is one practice-map cell (0.05 m), not the legacy "
              "0.15 m controller-safety buffer"))
    parser.add_argument("--diagnose-initial", action="store_true",
                        help="evaluate captured initial constraint residuals without solving")
    parser.add_argument("--print-level", type=int, default=0)
    args = parser.parse_args()

    centerline = args.centerline.resolve()
    template = args.profile_template.resolve()
    output = args.output.resolve()
    if not centerline.is_file() or not template.is_file():
        raise FileNotFoundError(centerline if not centerline.is_file() else template)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite candidate directory: {output}")
    if not (0.0 < args.min_speed_mps < args.max_speed_mps <= 12.0):
        raise ValueError("speed bounds must satisfy 0 < min < max <= 12 m/s")
    if args.feasibility_max_iterations <= 0:
        raise ValueError("feasibility iteration budget must be positive")
    if args.time_optimization_max_iterations <= 0:
        raise ValueError("time-optimization iteration budget must be positive")
    args.time_trust_region_factors = validate_time_trust_region_factors(
        args.time_trust_region_factors)
    if args.brake_event_count is not None:
        if args.brake_event_count <= 0:
            raise ValueError("brake event count must be positive")
        if args.positive_throttle_only:
            raise ValueError(
                "brake event count cannot be combined with positive-throttle-only")
        if args.warm_start_candidate is not None:
            raise ValueError(
                "alternate brake topology must start from the capture, not a "
                "candidate with a different phase mesh")
    if not np.isfinite(args.wall_clearance_m) or args.wall_clearance_m < 0.0:
        raise ValueError("wall clearance must be finite and non-negative")
    if (args.max_body_path_heading_error_rad is not None
            and (not np.isfinite(args.max_body_path_heading_error_rad)
                 or args.max_body_path_heading_error_rad <= 0.0
                 or args.max_body_path_heading_error_rad > 0.75)):
        raise ValueError(
            "body/path heading-error limit must be in (0, 0.75] rad")
    if (args.max_abs_lateral_speed_mps is not None
            and (not np.isfinite(args.max_abs_lateral_speed_mps)
                 or args.max_abs_lateral_speed_mps <= 0.0
                 or args.max_abs_lateral_speed_mps > 2.0)):
        raise ValueError(
            "absolute lateral-speed limit must be in (0, 2] m/s")
    if ((args.path_reference_trajectory is None)
            != (args.max_path_reference_deviation_m is None)):
        raise ValueError(
            "path-reference trajectory and maximum deviation must be provided together")
    if (args.max_path_reference_deviation_m is not None
            and (not np.isfinite(args.max_path_reference_deviation_m)
                 or args.max_path_reference_deviation_m <= 0.0
                 or args.max_path_reference_deviation_m > 2.0)):
        raise ValueError("path-reference deviation must be in (0, 2] m")
    if (not np.isfinite(args.path_reference_max_correspondence_distance_m)
            or args.path_reference_max_correspondence_distance_m <= 0.0):
        raise ValueError(
            "path-reference correspondence distance must be finite and positive")
    if (args.path_reference_trajectory is not None
            and not args.path_reference_trajectory.is_file()):
        raise FileNotFoundError(args.path_reference_trajectory)
    if (not np.isfinite(args.feasibility_trust_region)
            or args.feasibility_trust_region <= 0.0):
        raise ValueError("feasibility trust region must be finite and positive")
    # Validate against the Unity model's per-wheel static loads after the model
    # is constructed below; the scalar itself must be finite and >= old floor.
    if (not np.isfinite(args.min_wheel_normal_load_n)
            or args.min_wheel_normal_load_n < 0.05):
        raise ValueError("minimum wheel normal load must be finite and >= 0.05 N")
    explicit_lap_bounds = (args.reference_lap_start_s is not None,
                           args.reference_lap_stop_s is not None)
    if explicit_lap_bounds[0] != explicit_lap_bounds[1]:
        raise ValueError("reference lap start and stop must be provided together")
    if args.reference_lap_capture is not None and not all(explicit_lap_bounds):
        raise ValueError(
            "a custom reference capture requires explicit lap start and stop times")
    if ((args.tracking_bias_training_run_dir is None)
            != (args.tracking_bias_validation_run_dir is None)):
        raise ValueError(
            "tracking-bias training and validation run directories are a pair")
    reference_interval_s = (
        (args.reference_lap_start_s, args.reference_lap_stop_s)
        if all(explicit_lap_bounds) else None)

    profile_template = json.loads(template.read_text(encoding="utf-8"))
    track = rotate_track_to_profile_gate(
        load_centerline_csv(centerline, args.spacing_m), profile_template)
    path_reference_offsets = None
    path_reference_report = None
    if args.path_reference_trajectory is not None:
        reference_path = args.path_reference_trajectory.resolve()
        path_reference_offsets, path_reference_report = (
            path_reference_offsets_for_track(
                track, reference_path,
                maximum_distance_m=(
                    args.path_reference_max_correspondence_distance_m)))
    tracking_bias_values = None
    tracking_bias_report = None
    if args.tracking_bias_training_run_dir is not None:
        training_dir = args.tracking_bias_training_run_dir.resolve()
        validation_dir = args.tracking_bias_validation_run_dir.resolve()
        training_metrics, training_source = tracking_bias_metrics_from_run(
            training_dir, centerline)
        validation_metrics, validation_source = tracking_bias_metrics_from_run(
            validation_dir, centerline)
        tracking_bias_values, fitted_bias = tracking_bias_values_for_track(
            track, training_metrics)
        training_sectors = np.asarray(
            fitted_bias["median_signed_path_error_m"], dtype=np.float64)
        validation_sector_rows = validation_metrics[
            "path_bias_by_route_fraction_12"]
        if (len(validation_sector_rows) != 12
                or any(row["sample_count"] < 5
                       or row["median_signed_path_error_m"] is None
                       for row in validation_sector_rows)):
            raise ValueError(
                "tracking-bias validation needs at least 5 samples per sector")
        validation_sectors = np.asarray([
            float(row["median_signed_path_error_m"])
            for row in validation_sector_rows
        ], dtype=np.float64)
        if (validation_sectors.shape != (12,)
                or not np.isfinite(validation_sectors).all()):
            raise ValueError("tracking-bias validation does not cover 12 sectors")
        transfer_error = validation_sectors - training_sectors
        transfer_metrics = {
            "sector_median_rmse_m": float(np.sqrt(np.mean(transfer_error ** 2))),
            "sector_median_p95_abs_m": float(np.percentile(
                np.abs(transfer_error), 95)),
            "sector_median_max_abs_m": float(np.max(np.abs(transfer_error))),
            "acceptance_limit_m": float(args.wall_clearance_m),
            "passed": bool(np.max(np.abs(transfer_error)) <= args.wall_clearance_m),
        }
        if not transfer_metrics["passed"]:
            raise ValueError(
                "tracking-bias model failed practice transfer: worst sector "
                f"error {transfer_metrics['sector_median_max_abs_m']:.3f} m "
                f"exceeds requested wall margin {args.wall_clearance_m:.3f} m")
        tracking_bias_report = {
            "method": (
                "training-run median signed path error in 12 equal route-arc "
                "sectors; linear periodic interpolation"),
            "application": (
                "candidate-only path-boundary correction; predicted actual "
                "lateral offset = OCP offset + fitted signed bias; no runtime "
                "odom/MPC or Unity physics changes"),
            "training": training_source,
            "validation": validation_source,
            "training_sector_model": fitted_bias,
            "practice_transfer": transfer_metrics,
        }
    reference_capture = (args.reference_lap_capture.resolve()
                         if args.reference_lap_capture is not None
                         else args.brake_capture.resolve())
    brake_mode_mask, brake_mode_summary = measured_brake_mode_mask(
        reference_capture, args.brake_run_result.resolve(), profile_template,
        track, reference_interval_s)
    measured_brake_mode_mask_original = brake_mode_mask.copy()
    if args.positive_throttle_only:
        brake_mode_summary = {
            **brake_mode_summary,
            "candidate_mode_policy": "positive throttle at every optimized node",
            "reference_exact_zero_nodes_not_used_as_mode_constraints": True,
        }
        brake_mode_mask = np.zeros(track.count, dtype=bool)
    elif args.brake_event_count is not None:
        brake_mode_mask, topology_selection = retain_longest_brake_events(
            track, brake_mode_mask, args.brake_event_count)
        brake_mode_summary = {
            **brake_mode_summary,
            "alternate_topology": topology_selection,
        }
    model = UnityWheelModel.from_source_capture()
    phase_lengths_initial = None
    initial_state_seed_is_model_consistent = False
    if args.warm_start_candidate is None:
        initial_state_seed, state_seed_summary = captured_state_initial_guess(
            reference_capture, args.brake_run_result.resolve(), profile_template,
            track, brake_mode_mask, reference_interval_s)
        released_brake_nodes = (measured_brake_mode_mask_original
                                & ~brake_mode_mask)
        if np.any(released_brake_nodes):
            # A removed exact-zero brake event leaves captured wheel RPM at
            # zero. Seed those now-drive nodes at no-slip kinematic wheel
            # speed; the full wheel/tire dynamics remain optimized by IPOPT.
            for node in np.flatnonzero(released_brake_nodes):
                angles = ackermann_angles(float(
                    initial_state_seed["steering_rad"][node]))
                for wheel in range(4):
                    vx = (initial_state_seed["u_mps"][node]
                          - initial_state_seed["yaw_rate_radps"][node]
                          * WHEEL_Y_M[wheel])
                    vy = (initial_state_seed["v_mps"][node]
                          + initial_state_seed["yaw_rate_radps"][node]
                          * WHEEL_X_M[wheel])
                    wheel_forward = (np.cos(angles[wheel]) * vx
                                     + np.sin(angles[wheel]) * vy)
                    initial_state_seed["wheel_omega_radps"][wheel, node] = (
                        wheel_forward
                        / (model.wheel_radius_m * model.wheel_radius_scale))
            state_seed_summary["released_brake_nodes_seeded_as"] = (
                "no-slip kinematic wheel speed; full dynamics remain in OCP")
            state_seed_summary["released_brake_node_count"] = int(
                np.count_nonzero(released_brake_nodes))
        gate_state_seed = initial_state_seed
        initial_state_seed_s = track.s
    else:
        initial_state_seed, initial_state_seed_s, phase_lengths_initial, \
            state_seed_summary = candidate_state_initial_guess(
                args.warm_start_candidate.resolve(), track, model)
        gate_state_seed, gate_state_seed_summary = captured_state_initial_guess(
            reference_capture, args.brake_run_result.resolve(), profile_template,
            track, brake_mode_mask, reference_interval_s)
        state_seed_summary["gate_state_anchor_source"] = (
            gate_state_seed_summary["source_capture"])
        phase_modes, phase_nodes, _, _ = _brake_phases(track, brake_mode_mask)
        current_modes = ["brake" if mode else "drive" for mode in phase_modes]
        current_counts = [int(count) for count in phase_nodes]
        if (state_seed_summary["phase_modes"] != current_modes
                or state_seed_summary["phase_node_counts"] != current_counts):
            raise ValueError("warm-start brake/drive phase mesh differs from current capture")
        initial_state_seed_is_model_consistent = bool(
            state_seed_summary.get("model_consistent_seed", False))
    gate_state_anchor = None
    if not args.free_gate_state:
        gate_state_anchor = {
            "ey_m": float(gate_state_seed["ey_m"][0]),
            "epsi_rad": float(gate_state_seed["epsi_rad"][0]),
            "u_mps": float(gate_state_seed["u_mps"][0]),
            "v_mps": float(gate_state_seed["v_mps"][0]),
            "yaw_rate_radps": float(
                gate_state_seed["yaw_rate_radps"][0]),
            "wheel_omega_radps": gate_state_seed[
                "wheel_omega_radps"][:, 0].tolist(),
            "steering_rad": float(gate_state_seed["steering_rad"][0]),
        }
        state_seed_summary["gate_state_anchor_source"] = str(reference_capture)
    else:
        state_seed_summary["gate_state_anchor_source"] = None
    state_seed_summary["path_geometry_initialization"] = (
        "captured path/state initialization; path geometry remains free "
        "during full time optimization")
    candidate, report = solve_bottom_up(
        track, model,
        wall_clearance_m=args.wall_clearance_m,
        min_speed_mps=args.min_speed_mps,
        max_speed_mps=args.max_speed_mps,
        brake_mode_seed=brake_mode_mask,
        brake_mode_seed_source=(
            "reference capture seeds drive/brake phase count/order only; "
            "all phase lengths and switch positions are optimized: "
            + json.dumps(brake_mode_summary, sort_keys=True)),
        initial_state_seed=initial_state_seed,
        initial_state_seed_s=initial_state_seed_s,
        initial_state_seed_is_model_consistent=(
            initial_state_seed_is_model_consistent),
        gate_state_anchor=gate_state_anchor,
        gate_steering_tolerance_rad=args.gate_steering_tolerance_rad,
        max_body_path_heading_error_rad=(
            args.max_body_path_heading_error_rad),
        max_abs_lateral_speed_mps=args.max_abs_lateral_speed_mps,
        path_reference_offset_m=path_reference_offsets,
        max_path_reference_deviation_m=(
            args.max_path_reference_deviation_m),
        feasibility_trust_region=args.feasibility_trust_region,
        minimum_wheel_normal_load_n=args.min_wheel_normal_load_n,
        time_trust_region_factors=args.time_trust_region_factors,
        tracking_bias_m=tracking_bias_values,
        feasibility_max_iterations=args.feasibility_max_iterations,
        time_optimization_max_iterations=(
            args.time_optimization_max_iterations),
        time_hessian_approximation=args.time_hessian_approximation,
        minimum_trust_region_reference=float(
            state_seed_summary.get("minimum_trust_region_reference") or 1.0),
        phase_lengths_initial=phase_lengths_initial,
        throttle_fall_rate_limit_norm_per_s=(
            args.throttle_fall_rate_limit_norm_per_sec),
        gate_lateral_tolerance_m=args.gate_lateral_tolerance_m,
        diagnose_initial_only=args.diagnose_initial,
        print_level=args.print_level)
    if tracking_bias_report is not None:
        report["empirical_path_bias_model"] = tracking_bias_report
    report.setdefault("path_reference_tube", {
        "enabled": path_reference_offsets is not None,
        "maximum_lateral_deviation_m": (
            args.max_path_reference_deviation_m),
    })
    if path_reference_report is not None:
        report["path_reference_tube"]["correspondence"] = path_reference_report
    candidate_is_finite = bool(candidate) and all(
        np.isfinite(values).all() for values in candidate.values())
    candidate_violation = report.get("nlp_diagnostics", {}).get(
        "max_constraint_violation")
    warm_start_only = bool(
        candidate_is_finite and candidate_violation is not None
        and np.isfinite(float(candidate_violation))
        and float(candidate_violation) <= NEAR_FEASIBLE_WARM_START_MAX_VIOLATION
        and not report["solver"].get("feasible_incumbent", False))
    if not (report["solver"]["success"]
            or report["solver"].get("feasible_incumbent", False)):
        if warm_start_only:
            write_candidate(candidate, report, track, centerline, output)
            warm_start_manifest = {
                "status": "near_feasible_warm_start_only",
                "warning": (
                    "Not a feasible or Unity-validated raceline. This iterate "
                    "may only initialize another bounded OCP solve."),
                "candidate_states": str(
                    output / "unity_wheel_hybrid_states.csv"),
                "max_constraint_violation": float(candidate_violation),
                "warm_start_residual_limit": (
                    NEAR_FEASIBLE_WARM_START_MAX_VIOLATION),
                "solver": report["solver"],
                "predicted_lap_time_s": report.get("predicted_lap_time_s"),
            }
            (output / "warm_start_manifest.json").write_text(
                json.dumps(warm_start_manifest, indent=2, sort_keys=True)
                + "\n", encoding="utf-8")
            print(json.dumps(warm_start_manifest, indent=2))
            return 2
        output.mkdir(parents=True, exist_ok=False)
        (output / "solver_failure.json").write_text(
            json.dumps({
                "solver": report["solver"],
                "nlp_diagnostics": report["nlp_diagnostics"],
                "track": report["track"],
                "model": report["unity_vehicle_model"],
                "initial_state_seed": state_seed_summary,
                "brake_mode_seed": brake_mode_summary,
                "path_seed": "captured",
                "empirical_path_bias_model": tracking_bias_report,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(json.dumps({
            "status": "no_feasible_free_geometry_candidate",
            "solver": report["solver"],
            "nlp_diagnostics": report["nlp_diagnostics"],
            "diagnostic_report": str(output / "solver_failure.json"),
        }, indent=2, sort_keys=True))
        return 2
    write_candidate(candidate, report, track, centerline, output)
    profile = write_open_loop_profile(candidate, report, track, template, output)
    selected_stage = report.get("selected_stage")
    ipopt_converged = bool(report["solver"].get("ipopt_converged", False))
    if selected_stage == "captured-motion feasibility fallback":
        candidate_status = "feasible_candidate_not_time_optimized"
    elif selected_stage in {
            "free-geometry minimum time", "free-geometry trust-region time"}:
        candidate_status = (
            "feasible_time_candidate_local_solution_not_global_minimum"
            if ipopt_converged else
            "feasible_time_iterate_not_ipopt_converged")
    else:
        candidate_status = "candidate_status_unclassified"
    manifest = {
        "status": candidate_status,
        "candidate_kind": "free-geometry IPOPT phase-switched solve",
        "path_geometry_initialization": state_seed_summary[
            "path_geometry_initialization"],
        "centerline": str(centerline),
        "profile_template": str(template),
        "raceline": str(output / "unity_wheel_hybrid_raceline.csv"),
        "feedback_profile": str(profile),
        "predicted_lap_time_s": report["predicted_lap_time_s"],
        "geometry_and_physics": report["constraint_diagnostics"],
        "gate_state_anchor": report.get("gate_state_anchor"),
        "solver": report["solver"],
        "selected_stage": report.get("selected_stage"),
        "solve_stages": report.get("solve_stages"),
        "brake_phase_structure_seed": brake_mode_summary,
        "numerical_initial_guess": state_seed_summary,
        "model_limitations": report["limits_and_caveats"],
        "empirical_path_bias_model": tracking_bias_report,
    }
    (output / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
