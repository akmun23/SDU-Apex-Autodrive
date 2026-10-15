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
    solve_bottom_up,
    write_candidate,
)
from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.track import (
    Track,
    load_centerline_csv,
)
from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.unity_wheel_model import (
    ROOT,
    UnityWheelModel,
)
from tools.vehicle_dynamics_learning.unity_vehicle_mechanics import (
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


def measured_brake_mode_mask(capture: Path, run_result: Path,
                             profile: dict, track: Track
                             ) -> tuple[np.ndarray, dict]:
    result = json.loads(run_result.read_text(encoding="utf-8"))
    start_s = int(profile["warmup_command_count"]) * 0.025
    stop_s = start_s + float(result["scored_duration_s"])
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


def captured_state_initial_guess(capture: Path, run_result: Path,
                                 profile: dict, track: Track,
                                 brake_mode_mask: np.ndarray) -> tuple[dict, dict]:
    result = json.loads(run_result.read_text(encoding="utf-8"))
    start_s = int(profile["warmup_command_count"]) * 0.025
    stop_s = start_s + float(result["scored_duration_s"])
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
        "seed_speed_range_mps": [float(np.min(seed["u_mps"])),
                                 float(np.max(seed["u_mps"]))],
        "seed_lateral_offset_rms_m": float(np.sqrt(np.mean(seed["ey_m"] ** 2))),
        "seed_lateral_acceleration_p95_mps2": float(np.quantile(
            np.abs(seed["ay_mps2"]), 0.95)),
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
    route = commands[warmup:]
    if not route:
        raise ValueError("profile template has no scored route commands")

    path_x = candidate["x_m"]
    path_y = candidate["y_m"]
    path_ds = np.hypot(np.diff(np.r_[path_x, path_x[0]]),
                       np.diff(np.r_[path_y, path_y[0]]))
    path_length = float(np.sum(path_ds))
    query_s = np.arange(len(route), dtype=np.float64) * path_length / len(route)
    source_s = np.asarray(candidate["route_arc_m"], dtype=np.float64)
    x = _periodic_interp(source_s, path_x, query_s, path_length)
    y = _periodic_interp(source_s, path_y, query_s, path_length)
    speed = _periodic_interp(source_s, candidate["speed_mps"],
                              query_s, path_length)
    # Keep discrete actuator modes and optimized steering at the spatial
    # command nodes. Interpolating throttle across a drive/brake transition
    # would create extra exact-zero CAWB brake commands.
    right_index = np.searchsorted(source_s, query_s, side="left") % len(source_s)
    left_index = (right_index - 1) % len(source_s)
    distance_right = (source_s[right_index] - query_s) % path_length
    distance_left = (query_s - source_s[left_index]) % path_length
    control_index = np.where(distance_left < distance_right,
                             left_index, right_index)
    throttle = candidate["throttle_norm"][control_index]
    steering = candidate["steering_rad"][control_index]

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
    profile["scored_horizon_commands"] = len(route)
    profile["raceline_signed_throttle_feedback"] = False
    profile["bottom_up_optimizer_source"] = (
        "Unity-source-calibrated per-wheel IPOPT free-geometry phase OCP")
    profile["bottom_up_optimizer_predicted_lap_time_s"] = report[
        "predicted_lap_time_s"]

    route_arc = np.r_[0.0, np.cumsum(path_ds[:-1])]
    for i, row in enumerate(route):
        row["has_reference_position"] = True
        row["reference_x_m"] = float(0.05 - y[i])
        row["reference_z_m"] = float(x[i] - 0.24)
        row["has_raceline_target"] = True
        row["target_speed_mps"] = float(speed[i])
        row["route_arc_m"] = float(query_s[i])
        row["throttle_norm"] = float(throttle[i])
        row["steering_norm"] = float(-steering[i] / np.deg2rad(30.0))
        row["steering_gain_scale"] = 1.0
        row["speed_feedback_gain_scale"] = 1.0
        row["positive_forward_slip_feedback_gain_scale"] = 0.0
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-m", type=float, default=0.12)
    parser.add_argument("--min-speed-mps", type=float, default=0.35)
    parser.add_argument("--max-speed-mps", type=float, default=12.0)
    parser.add_argument("--max-solver-seconds", type=float, default=300.0)
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

    profile_template = json.loads(template.read_text(encoding="utf-8"))
    track = rotate_track_to_profile_gate(
        load_centerline_csv(centerline, args.spacing_m), profile_template)
    brake_mode_mask, brake_mode_summary = measured_brake_mode_mask(
        args.brake_capture.resolve(), args.brake_run_result.resolve(),
        profile_template, track)
    initial_state_seed, state_seed_summary = captured_state_initial_guess(
        args.brake_capture.resolve(), args.brake_run_result.resolve(),
        profile_template, track, brake_mode_mask)
    model = UnityWheelModel.from_training_capture()
    candidate, report = solve_bottom_up(
        track, model,
        min_speed_mps=args.min_speed_mps,
        max_speed_mps=args.max_speed_mps,
        brake_mode_seed=brake_mode_mask,
        brake_mode_seed_source=(
            "reference capture seeds drive/brake phase count/order only; "
            "all phase lengths and switch positions are optimized: "
            + json.dumps(brake_mode_summary, sort_keys=True)),
        initial_state_seed=initial_state_seed,
        max_solver_seconds=args.max_solver_seconds,
        diagnose_initial_only=args.diagnose_initial,
        print_level=args.print_level)
    if not (report["solver"]["success"]
            or report["solver"].get("feasible_incumbent", False)):
        output.mkdir(parents=True, exist_ok=False)
        (output / "solver_failure.json").write_text(
            json.dumps({
                "solver": report["solver"],
                "nlp_diagnostics": report["nlp_diagnostics"],
                "track": report["track"],
                "model": report["unity_vehicle_model"],
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
    candidate_status = {
        "captured-motion feasibility fallback":
            "feasible_candidate_not_time_optimized",
        "free-geometry minimum time": (
            "candidate_generated_optimality_unproven"
            if not report["solver"].get("optimality_proven", False)
            else "candidate_generated"),
    }.get(report.get("selected_stage"), "candidate_generated")
    manifest = {
        "status": candidate_status,
        "candidate_kind": "new map-centerline-seeded free-geometry IPOPT phase-switched solve",
        "centerline": str(centerline),
        "profile_template": str(template),
        "raceline": str(output / "unity_wheel_hybrid_raceline.csv"),
        "feedback_profile": str(profile),
        "predicted_lap_time_s": report["predicted_lap_time_s"],
        "geometry_and_physics": report["constraint_diagnostics"],
        "solver": report["solver"],
        "selected_stage": report.get("selected_stage"),
        "solve_stages": report.get("solve_stages"),
        "brake_phase_structure_seed": brake_mode_summary,
        "numerical_initial_guess": state_seed_summary,
        "model_limitations": report["limits_and_caveats"],
    }
    (output / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
