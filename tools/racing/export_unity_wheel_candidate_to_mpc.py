#!/usr/bin/env python3
"""Export a Unity-wheel OCP path in the production MPC trajectory schema."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from SDU_Apex_Autodrive_Exact_MinTime_Optimizer_v1_3.f1tenth_planning.autodrive_mintime.track import (
    load_centerline_csv,
)
from tools.racing.optimize_unity_wheel_raceline import (
    rotate_track_to_profile_gate,
)
from tools.racing.mpc_vehicle_geometry import (
    MPC_REQUIRED_WALL_CLEARANCE_M,
    footprint_half_width_numpy,
)


MPC_CORRIDOR_MARGIN_M = MPC_REQUIRED_WALL_CLEARANCE_M
EXTRA_CLEARANCE_M = 0.0
MPC_RESAMPLE_SPACING_M = 0.10


def _read_states(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = ("s_ref_m", "x_m", "y_m", "ey_m", "epsi_rad",
              "speed_mps", "ax_mps2")
    if len(rows) < 3 or any(name not in rows[0] for name in fields):
        raise ValueError("OCP state CSV lacks MPC path/state fields")
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in fields
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("OCP state CSV contains non-finite values")
    if np.any(np.diff(values["s_ref_m"]) <= 0.0):
        raise ValueError("OCP reference stations must be strictly increasing")
    if np.any(values["speed_mps"] <= 0.0):
        raise ValueError("raceline speeds must remain positive")
    return values


def _closed_path_geometry(
    xy: np.ndarray, spacing_m: float = MPC_RESAMPLE_SPACING_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if spacing_m <= 0.0:
        raise ValueError("trajectory resampling spacing must be positive")
    segment = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    if not np.isfinite(segment).all() or np.any(segment <= 1.0e-6):
        raise ValueError("candidate path has a zero-length or invalid segment")
    source_station = np.r_[0.0, np.cumsum(segment[:-1])]
    length = float(np.sum(segment))
    station = np.arange(0.0, length, spacing_m, dtype=np.float64)
    if len(station) < 3:
        raise ValueError("candidate path is too short to resample")
    closed_s = np.r_[source_station, length]
    closed_xy = np.vstack((xy, xy[0]))
    x_spline = CubicSpline(closed_s, closed_xy[:, 0], bc_type="periodic")
    y_spline = CubicSpline(closed_s, closed_xy[:, 1], bc_type="periodic")
    sampled_xy = np.column_stack((x_spline(station), y_spline(station)))
    dx = x_spline(station, 1)
    dy = y_spline(station, 1)
    ddx = x_spline(station, 2)
    ddy = y_spline(station, 2)
    heading = np.unwrap(np.arctan2(dy, dx))
    curvature = (dx * ddy - dy * ddx) / np.maximum(
        (dx * dx + dy * dy) ** 1.5, 1.0e-12)
    return source_station, station, sampled_xy, heading, curvature, length


def _periodic_values(query: np.ndarray, source_s: np.ndarray,
                     values: np.ndarray, length: float) -> np.ndarray:
    ext_s = np.r_[source_s - length, source_s, source_s + length]
    ext_values = np.tile(values, 3)
    return np.interp(np.mod(query, length), ext_s, ext_values)


def export_candidate(states_path: Path, centerline_path: Path,
                     profile_path: Path, output_path: Path,
                     mpc_corridor_margin_m: float = MPC_CORRIDOR_MARGIN_M,
                     extra_clearance_m: float = EXTRA_CLEARANCE_M) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if mpc_corridor_margin_m <= 0.0 or extra_clearance_m < 0.0:
        raise ValueError("corridor margin must be positive and extra clearance nonnegative")
    state = _read_states(states_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    track = rotate_track_to_profile_gate(
        load_centerline_csv(centerline_path, spacing_m=0.12), profile)
    if state["s_ref_m"][-1] >= track.length:
        raise ValueError("OCP reference station exceeds centerline lap length")

    xy = np.column_stack((state["x_m"], state["y_m"]))
    (source_station, station, sampled_xy, heading, curvature,
     lap_length) = _closed_path_geometry(xy)
    left_centerline_nodes = _periodic_values(
        state["s_ref_m"], track.s, track.left, track.length)
    right_centerline_nodes = _periodic_values(
        state["s_ref_m"], track.s, track.right, track.length)

    # The production MPC interprets d_left/right as geometric distance from
    # the trajectory point to each wall, then subtracts its configured wall
    # margin plus the heading-dependent vehicle footprint. Preserve raw wall
    # distances here; subtracting the footprint again would double-count it.
    d_left_nodes = left_centerline_nodes - state["ey_m"] - extra_clearance_m
    d_right_nodes = right_centerline_nodes + state["ey_m"] - extra_clearance_m
    d_left = _periodic_values(station, source_station, d_left_nodes, lap_length)
    d_right = _periodic_values(station, source_station, d_right_nodes, lap_length)
    speed = _periodic_values(
        station, source_station, state["speed_mps"], lap_length)
    acceleration = _periodic_values(
        station, source_station, state["ax_mps2"], lap_length)

    # Convert the OCP body's heading (relative to the map centerline) to its
    # heading error relative to the exported trajectory, which is the angle
    # used by the MPC footprint projection at nominal e_y=0.
    center_heading_cos_nodes = _periodic_values(
        state["s_ref_m"], track.s, np.cos(track.psi), track.length)
    center_heading_sin_nodes = _periodic_values(
        state["s_ref_m"], track.s, np.sin(track.psi), track.length)
    center_heading_nodes = np.arctan2(
        center_heading_sin_nodes, center_heading_cos_nodes)
    body_heading_nodes = center_heading_nodes + state["epsi_rad"]
    body_heading_cos = _periodic_values(
        station, source_station, np.cos(body_heading_nodes), lap_length)
    body_heading_sin = _periodic_values(
        station, source_station, np.sin(body_heading_nodes), lap_length)
    body_heading = np.arctan2(body_heading_sin, body_heading_cos)
    mpc_heading_error = np.arctan2(
        np.sin(body_heading - heading), np.cos(body_heading - heading))
    # Reject a path unless its nominal OCP states are inside the exact corridor
    # used by production MPC. This catches footprint and wall-clearance
    # mismatches before spending a simulator run.
    mpc_footprint = footprint_half_width_numpy(mpc_heading_error)
    nominal_slack = np.minimum(d_left, d_right) - (
        mpc_corridor_margin_m + mpc_footprint)
    path_turns = int(round((heading[-1] - heading[0]) / (2.0 * np.pi)))
    heading_next = np.r_[heading[1:], heading[0] + 2.0 * np.pi * path_turns]
    body_heading_unwrapped = np.unwrap(body_heading)
    body_turns = int(round(
        (body_heading_unwrapped[-1] - body_heading_unwrapped[0])
        / (2.0 * np.pi)))
    body_heading_next = np.r_[
        body_heading_unwrapped[1:],
        body_heading_unwrapped[0] + 2.0 * np.pi * body_turns,
    ]
    midpoint_heading_error = np.arctan2(
        np.sin(0.5 * (body_heading_unwrapped + body_heading_next
                       - heading - heading_next)),
        np.cos(0.5 * (body_heading_unwrapped + body_heading_next
                       - heading - heading_next)))
    midpoint_footprint = footprint_half_width_numpy(midpoint_heading_error)
    midpoint_left = 0.5 * (d_left + np.roll(d_left, -1))
    midpoint_right = 0.5 * (d_right + np.roll(d_right, -1))
    midpoint_slack = np.minimum(midpoint_left, midpoint_right) - (
        mpc_corridor_margin_m + midpoint_footprint)
    minimum_nominal_slack = float(min(
        np.min(nominal_slack), np.min(midpoint_slack)))
    maximum_nominal_heading_error = float(max(
        np.max(np.abs(mpc_heading_error)),
        np.max(np.abs(midpoint_heading_error))))
    if minimum_nominal_slack < -0.002:
        raise ValueError(
            "candidate nominal states violate the production MPC corridor: "
            f"minimum slack {minimum_nominal_slack:.4f} m; optimize with at "
            "least the runtime wall clearance and footprint")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "# s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_left_m,d_right_m\n")
        for index in range(len(station)):
            stream.write(
                f"{station[index]:.9f},{sampled_xy[index, 0]:.9f},{sampled_xy[index, 1]:.9f},"
                f"{heading[index]:.10f},{curvature[index]:.10f},"
                f"{speed[index]:.9f},{acceleration[index]:.9f},"
                f"{d_left[index]:.9f},{d_right[index]:.9f}\n")

    return {
        "trajectory": str(output_path),
        "state_source": str(states_path),
        "centerline_source": str(centerline_path),
        "gate_alignment_profile": str(profile_path),
        "point_count": int(len(station)),
        "source_point_count": int(len(source_station)),
        "resample_spacing_m": MPC_RESAMPLE_SPACING_M,
        "candidate_path_length_m": lap_length,
        "mpc_corridor_margin_m": mpc_corridor_margin_m,
        "extra_vehicle_clearance_m": extra_clearance_m,
        "maximum_nominal_mpc_heading_error_rad": float(
            maximum_nominal_heading_error),
        "minimum_nominal_mpc_corridor_slack_m": minimum_nominal_slack,
        "minimum_left_bound_m": float(np.min(d_left)),
        "minimum_right_bound_m": float(np.min(d_right)),
        "peak_speed_mps": float(np.max(state["speed_mps"])),
        "peak_abs_curvature_per_m": float(np.max(np.abs(curvature))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--centerline", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True,
                        help="candidate profile whose first route point defines the gate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mpc-corridor-margin-m", type=float,
                        default=MPC_CORRIDOR_MARGIN_M)
    parser.add_argument("--extra-clearance-m", type=float,
                        default=EXTRA_CLEARANCE_M)
    args = parser.parse_args()
    report = export_candidate(
        args.states.resolve(), args.centerline.resolve(), args.profile.resolve(),
        args.output.resolve(), args.mpc_corridor_margin_m,
        args.extra_clearance_m)
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
