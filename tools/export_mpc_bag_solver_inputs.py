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
    PLANT,
    load_trajectory,
    read_bag,
)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--trajectory", type=Path,
        help="reproject saved command-time map poses onto this trajectory")
    args = parser.parse_args()

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
            # The diagnostic body/path state is command-time predicted from
            # the newest synchronized sensor epoch. Use the full hidden plant
            # snapshot at that same sensor epoch; the controller's older
            # handoff snapshot is only an internal propagation starting point
            # and would leave the replay plant roughly one packet stale.
            plant = plant_at(source_ns) or plant_at(plant_ns)
            if plant is None:
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
                (progress_m, lateral_error, heading_error,
                 projection_distance) = project_pose_to_trajectory(
                    trajectory, *map(float, predicted_pose))
                projection_distances.append(projection_distance)
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
            # `state` stores e_y, e_psi, u, v, r, delayed steer 1/2 (physical
            # radians), physical steering, and delayed throttle 1/2.
            writer.writerow([
                source_ns,
                progress_m,
                diag["active_speed_ceiling_mps"],
                lateral_error, heading_error, *state[2:5],
                state[7], state[5], state[6], state[8], state[9],
                int(plant["suspension_history_initialized"]),
                plant.get("yaw_rate_prediction_residual", 0.0),
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
