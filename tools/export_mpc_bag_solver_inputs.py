#!/usr/bin/env python3
"""Export recorded MPC state/progress and Unity plant state for core replay.

This bypasses ROS playback scheduling and localization re-projection while
preserving the captured per-cycle optimizer inputs.  It is an offline tuning
input exporter, not a production runtime component.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_mpc_bag import (  # noqa: E402
    DIAGNOSTICS,
    GT,
    PLANT,
    load_trajectory,
    read_bag,
)

MAX_STEERING_RAD = math.pi / 6.0


def project_pose_to_trajectory(
    trajectory: tuple[list[dict[str, float]], float],
    x: float,
    y: float,
    heading: float,
) -> tuple[float, float, float, float]:
    """Match the MPC's full-track, heading-gated segment projection."""
    points, lap_length = trajectory
    best: tuple[float, float, float, float] | None = None
    best_distance_squared = math.inf
    for index, a in enumerate(points):
        b = points[(index + 1) % len(points)]
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1.0e-12:
            continue
        ratio = max(0.0, min(1.0, ((x - a["x"]) * dx +
                                  (y - a["y"]) * dy) / length_squared))
        px = a["x"] + ratio * dx
        py = a["y"] + ratio * dy
        tangent = math.atan2(dy, dx)
        e_psi = math.atan2(math.sin(heading - tangent),
                           math.cos(heading - tangent))
        if abs(e_psi) > 0.5 * math.pi:
            continue
        ex, ey = x - px, y - py
        distance_squared = ex * ex + ey * ey
        if distance_squared >= best_distance_squared:
            continue
        s1 = (points[0]["s"] + lap_length) if index + 1 == len(points) \
            else b["s"]
        progress = a["s"] + ratio * (s1 - a["s"])
        lateral_error = -math.sin(tangent) * ex + math.cos(tangent) * ey
        best = (progress, lateral_error, e_psi, math.sqrt(distance_squared))
        best_distance_squared = distance_squared
    if best is None:
        raise RuntimeError("pose could not be projected onto selected trajectory")
    return best


def export_oracle_pose_inputs(
    bag_data: dict,
    output_path: Path,
    trajectory: tuple[list[dict[str, float]], float],
    max_speed_mps: float,
    startup_speed_mps: float,
    startup_ramp_laps: float,
) -> int:
    """Export GT path pose plus recorded production plant state for solver replay.

    This is an offline diagnostic initialization, not a permissible runtime
    input path. It answers whether the solver can converge when path pose is
    known, while retaining the causal production plant state and command queue.
    """
    truth = sorted(bag_data.get(GT, []), key=lambda row: row["stamp"])
    plants = sorted(bag_data.get(PLANT, []), key=lambda row: row["stamp"])
    if not truth or not plants:
        raise RuntimeError("oracle-pose export needs GT and UnityPlantState topics")

    plant_by_stamp = {round(row["stamp"], 6): row for row in plants}
    points, lap_length = trajectory
    previous_progress = None
    startup_progress = 0.0
    count = 0
    projection_distances = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "# stamp_ns", "progress_m", "active_speed_ceiling_mps", "e_y",
            "e_psi", "u", "v", "r", "actual_steering_rad",
            "delayed_steer_1_rad", "delayed_steer_2_rad",
            "delayed_throttle_1", "delayed_throttle_2",
            "suspension_history_initialized", "yaw_residual_radps",
            "yaw_residual_valid", "wheel_omega_fl", "wheel_omega_fr",
            "wheel_omega_rl", "wheel_omega_rr", "body_height",
            "body_height_rate", "roll", "roll_rate", "pitch", "pitch_rate",
            "jounce_fl", "jounce_fr", "jounce_rl", "jounce_rr",
            "low_forward_fl", "low_forward_fr", "low_forward_rl",
            "low_forward_rr", "low_side_fl", "low_side_fr", "low_side_rl",
            "low_side_rr",
        ])
        for pose in truth:
            plant = plant_by_stamp.get(round(pose["stamp"], 6))
            if plant is None or len(plant.get("wheel_omega", [])) != 4:
                continue
            progress, e_y, e_psi, projection_distance = project_pose_to_trajectory(
                trajectory, pose["x"], pose["y"], pose["yaw"])
            projection_distances.append(projection_distance)
            if previous_progress is not None:
                unwrapped = progress
                while unwrapped - previous_progress < -0.5 * lap_length:
                    unwrapped += lap_length
                while unwrapped - previous_progress > 0.5 * lap_length:
                    unwrapped -= lap_length
                delta = unwrapped - previous_progress
                if 0.0 < delta < 0.5 * lap_length:
                    startup_progress += delta
                progress = unwrapped
            previous_progress = progress

            if startup_ramp_laps > 1.0e-6:
                ramp_fraction = min(
                    1.0, max(0.0, startup_progress /
                             (startup_ramp_laps * lap_length)))
                speed_ceiling = startup_speed_mps + (
                    max_speed_mps - startup_speed_mps) * ramp_fraction
            else:
                speed_ceiling = max_speed_mps

            hidden = (
                list(plant["wheel_omega"])
                + [plant["body_height"], plant["body_height_rate"],
                   plant["roll"], plant["roll_rate"], plant["pitch"],
                   plant["pitch_rate"]]
                + list(plant["suspension_jounce"])
                + list(plant["low_forward_speed_time_s"])
                + list(plant["low_side_speed_time_s"])
            )
            if len(hidden) != 22:
                continue
            stamp_ns = round(pose["stamp"] * 1e9)
            writer.writerow([
                stamp_ns, progress, speed_ceiling, e_y, e_psi,
                plant["u"], plant["v_rear"], plant["gyro_r"],
                plant["physical_steering"],
                plant["delayed_steering_1"] * MAX_STEERING_RAD,
                plant["delayed_steering_2"] * MAX_STEERING_RAD,
                plant["delayed_throttle_1"], plant["delayed_throttle_2"],
                int(plant["suspension_history_initialized"]),
                plant.get("yaw_rate_prediction_residual", 0.0),
                int(plant.get("yaw_rate_prediction_residual_valid", False)),
                *hidden,
            ])
            count += 1

    if count == 0:
        raise RuntimeError("no same-epoch GT/plant samples were exported")
    ordered_distances = sorted(projection_distances)
    p95 = ordered_distances[
        min(len(ordered_distances) - 1,
            int(0.95 * (len(ordered_distances) - 1)))
    ]
    print(
        f"exported {count} oracle-pose/production-plant MPC inputs to "
        f"{output_path}; route projection distance median="
        f"{statistics.median(ordered_distances):.3f}m "
        f"p95/max={p95:.3f}/{ordered_distances[-1]:.3f}m; "
        f"active speed ceiling={startup_speed_mps:g}..{max_speed_mps:g}m/s"
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--trajectory", type=Path,
        help="reproject saved command-time map poses onto this trajectory")
    parser.add_argument(
        "--oracle-ground-truth-pose", action="store_true",
        help="offline only: create solver inputs from GT pose and the recorded "
             "production UnityPlantState; requires --trajectory")
    parser.add_argument("--max-speed-mps", type=float, default=16.0)
    parser.add_argument("--startup-speed-mps", type=float, default=1.5)
    parser.add_argument("--startup-ramp-laps", type=float, default=1.0)
    parser.add_argument(
        "--last-cycles", type=int, default=0,
        help="export only the final N valid MPC diagnostic states for a "
             "terminal-state replay corpus (0 exports the full capture)")
    parser.add_argument(
        "--progress-range-m", type=float, nargs=2, metavar=("START", "END"),
        help="export diagnostic states in [START, END) trajectory progress; "
             "useful for isolating the onset preceding a terminal failure")
    args = parser.parse_args()

    if args.last_cycles < 0:
        parser.error("--last-cycles must be zero or positive")
    if args.last_cycles and args.progress_range_m is not None:
        parser.error("choose either --last-cycles or --progress-range-m")
    if args.progress_range_m is not None:
        start_m, end_m = args.progress_range_m
        if not (math.isfinite(start_m) and math.isfinite(end_m) and
                end_m > start_m):
            parser.error("--progress-range-m requires finite END > START")
    if args.oracle_ground_truth_pose and (
            args.last_cycles or args.progress_range_m is not None):
        parser.error("terminal-state selection applies to MPC diagnostic-state "
                     "export, not the oracle-pose export")

    data = read_bag(args.bag)
    trajectory = load_trajectory(args.trajectory)
    if trajectory is not None:
        points, _ = trajectory
        closing_distance = math.hypot(
            points[0]["x"] - points[-1]["x"],
            points[0]["y"] - points[-1]["y"])
        trajectory = (
            points,
            points[-1]["s"] - points[0]["s"] + closing_distance,
        )
    if args.oracle_ground_truth_pose:
        if trajectory is None:
            parser.error("--oracle-ground-truth-pose requires --trajectory")
        if (not math.isfinite(args.max_speed_mps) or args.max_speed_mps <= 0.0 or
                not math.isfinite(args.startup_speed_mps) or
                args.startup_speed_mps < 0.0 or
                not math.isfinite(args.startup_ramp_laps) or
                args.startup_ramp_laps < 0.0):
            parser.error("speed and startup ramp settings must be finite and nonnegative")
        export_oracle_pose_inputs(
            data, args.output, trajectory,
            args.max_speed_mps, args.startup_speed_mps,
            args.startup_ramp_laps)
        return 0
    plant_rows = data.get(PLANT, [])
    plant_by_ns = {
        round(row["stamp"] * 1e9): row
        for row in plant_rows
        if row.get("stamp", 0.0) > 0.0 and len(row.get("wheel_omega", [])) == 4
    }
    diagnostic_rows = [
        row["json"] for row in data.get(DIAGNOSTICS, [])
        if row.get("json") and isinstance(row["json"].get("state"), list)
    ]
    if not diagnostic_rows:
        raise RuntimeError("bag has no valid MPC state diagnostics")
    if args.progress_range_m is not None:
        start_m, end_m = args.progress_range_m
        diagnostic_rows = [
            row for row in diagnostic_rows
            if start_m <= float(row.get("progress_m", math.nan)) < end_m
        ]
        if not diagnostic_rows:
            raise RuntimeError(
                "no valid MPC diagnostic states fall in the progress range")
    if args.last_cycles:
        diagnostic_rows = diagnostic_rows[-args.last_cycles:]
    plant_stamps = sorted(plant_by_ns)

    def plant_at(stamp_ns: int) -> dict | None:
        index = bisect.bisect_left(plant_stamps, stamp_ns)
        candidates = plant_stamps[max(0, index - 1):index + 1]
        if not candidates:
            return None
        nearest = min(candidates, key=lambda value: abs(value - stamp_ns))
        # The bag parser represents ROS stamps as double seconds; permit only
        # its sub-microsecond conversion loss, never a different plant epoch.
        return plant_by_ns[nearest] if abs(nearest - stamp_ns) <= 1000 else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    projection_distances: list[float] = []
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "# stamp_ns", "progress_m", "speed_ceiling_mps", "e_y", "e_psi",
            "u", "v", "r", "actual_steering_rad", "delayed_steer_1_rad",
            "delayed_steer_2_rad", "delayed_throttle_1", "delayed_throttle_2",
            "suspension_history_initialized", "yaw_residual_radps",
            "yaw_residual_valid", "wheel_omega_fl", "wheel_omega_fr",
            "wheel_omega_rl", "wheel_omega_rr", "body_height",
            "body_height_rate", "roll", "roll_rate", "pitch", "pitch_rate",
            "jounce_fl", "jounce_fr", "jounce_rl", "jounce_rr",
            "low_forward_fl", "low_forward_fr", "low_forward_rl",
            "low_forward_rr", "low_side_fl", "low_side_fr", "low_side_rl",
            "low_side_rr",
        ])
        for diag in diagnostic_rows:
            state = diag["state"]
            if len(state) != 10:
                continue
            source_ns = int(diag.get("source_stamp_ns", 0))
            plant_ns = int(diag.get("control_time_prediction", {}).get(
                "plant_source_stamp_ns", source_ns))
            # Current diagnostics carry the full plant state actually supplied
            # to the optimizer. Older bags lack that field; for those only,
            # use the synchronized sensor-epoch snapshot as the best available
            # replay state (it may be stale relative to command time).
            plant = plant_at(source_ns) or plant_at(plant_ns)
            logged_hidden = diag.get("plant_internal_state")
            logged_flags = diag.get("plant_state_flags")
            if (not isinstance(logged_hidden, list) or
                    len(logged_hidden) != 22):
                logged_hidden = None
            if (not isinstance(logged_flags, list) or
                    len(logged_flags) != 3):
                logged_flags = None
            if plant is None and logged_hidden is None:
                continue
            progress_m = float(diag["progress_m"])
            lateral_error = float(state[0])
            heading_error = float(state[1])
            if trajectory is not None:
                predicted_pose = diag.get("control_time_prediction", {}).get(
                    "predicted_map_pose")
                if not isinstance(predicted_pose, list) or len(predicted_pose) != 3:
                    raise RuntimeError(
                        "route-aware export requires command-time predicted_map_pose")
                # `progress_m` and `state[:2]` above are the exact values passed
                # to mpc_rti_solve_cycle(). Reprojecting the diagnostic pose here
                # silently changes those optimizer inputs (and can shift them
                # by >10 cm when the command-time projection and saved pose
                # differ). Keep the re-projection strictly diagnostic.
                _, _, _, projection_distance = project_pose_to_trajectory(
                    trajectory, *map(float, predicted_pose))
                projection_distances.append(projection_distance)
            hidden = logged_hidden if logged_hidden is not None else (
                list(plant["wheel_omega"])
                + [plant["body_height"], plant["body_height_rate"],
                   plant["roll"], plant["roll_rate"], plant["pitch"],
                   plant["pitch_rate"]]
                + list(plant["suspension_jounce"])
                + list(plant["low_forward_speed_time_s"])
                + list(plant["low_side_speed_time_s"])
            )
            if len(hidden) != 22:
                continue
            # `state` stores e_y, e_psi, u, v, r, delayed steer 1/2 (physical
            # radians), physical steering, and delayed throttle 1/2.
            writer.writerow([
                source_ns,
                progress_m,
                diag["active_speed_ceiling_mps"],
                lateral_error, heading_error, *state[2:5],
                state[7], state[5], state[6], state[8], state[9],
                int(logged_flags[1]) if logged_flags is not None else
                    int(plant["suspension_history_initialized"]),
                0.0 if plant is None else
                    plant.get("yaw_rate_prediction_residual", 0.0),
                0 if plant is None else
                    int(plant.get("yaw_rate_prediction_residual_valid", False)),
                *hidden,
            ])
            count += 1
    if count != len(diagnostic_rows):
        raise RuntimeError(
            f"exported {count} of {len(diagnostic_rows)} valid MPC cycles; "
            "missing plant snapshots must be resolved before scoring")
    print(f"exported {count} exact diagnostic cycles to {args.output}")
    if projection_distances:
        ordered = sorted(projection_distances)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        print(
            "route reprojection distance: "
            f"median={statistics.median(ordered):.3f} m "
            f"p95={ordered[p95_index]:.3f} m max={ordered[-1]:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
