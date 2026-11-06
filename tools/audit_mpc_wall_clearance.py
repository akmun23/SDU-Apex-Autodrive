#!/usr/bin/env python3
"""Compare MPC nonlinear corridor rejection to conservative map/body clearance.

Diagnostic only. This does NOT authorize unsafe publishing or override the
production MPC. It uses the same rear-axle footprint and occupancy-distance
calculation as the map's physical certificate. Unknown cells are obstacles.

Example:
  .venv/bin/python tools/audit_mpc_wall_clearance.py --map f1tenth_planning/maps/autodrive_iros_2026_compete.pgm --trajectory <witness.csv> --watcher <watcher.log> --margin-m 0.01
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SDU_Apex_IPOPT_PhysX_MinTime_v2.map_track import MapGrid


def sample_trajectory(data: np.ndarray, lap_length: float, station: float) -> tuple[float, float, float]:
    """Match MPC's piecewise-linear x/y and shortest-angle heading reference."""
    s = (float(station) - float(data[0, 0])) % float(lap_length) + float(data[0, 0])
    start = int(np.searchsorted(data[:, 0], s, side="right") - 1)
    if start < 0 or start >= len(data):
        raise ValueError("invalid station")
    end = (start + 1) % len(data)
    ds = (float(data[end, 0] - data[start, 0]) if end else
          float(lap_length - data[start, 0] + data[0, 0]))
    if ds <= 0:
        raise ValueError("non-increasing trajectory station")
    u = (s - float(data[start, 0])) / ds
    h0, h1 = float(data[start, 3]), float(data[end, 3])
    dh = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
    return (float(data[start, 1] + u * (data[end, 1] - data[start, 1])),
            float(data[start, 2] + u * (data[end, 2] - data[start, 2])),
            h0 + u * dh)


def vehicle_pose(trajectory: np.ndarray, lap_length: float,
                 station: float, e_y: float, e_psi: float) -> tuple[float, float, float]:
    x, y, psi = sample_trajectory(trajectory, lap_length, station)
    return (x - math.sin(psi) * e_y,
            y + math.cos(psi) * e_y,
            psi + e_psi)


def audit_fault(grid: MapGrid, track: np.ndarray, lap_length: float,
                fault: dict, wall_margin: float) -> dict:
    predicted = fault["nonlinear_failure"]
    cases = [
        ("command_time_state", float(fault["progress_m"]), fault["state"]),
        ("rejected_prediction", float(predicted["progress_m"]), predicted["state"]),
    ]
    out = []
    for name, station, state in cases:
        x, y, yaw = vehicle_pose(track, lap_length, station,
                                 float(state[0]), float(state[1]))
        clearance = grid.footprint_clearance(x, y, yaw, margin=0.0)
        out.append({"name": name, "unwrapped_progress_m": station,
                    "lap_station_m": station % lap_length,
                    "rear_axle_world_xy_m": [x, y],
                    "world_yaw_rad": yaw,
                    "map_body_clearance_m": clearance,
                    "requested_physical_margin_m": wall_margin,
                    "map_margin_slack_m": clearance - wall_margin,
                    "map_clearance_passes": clearance >= wall_margin})
    return {"mpc_status": fault.get("status"),
            "mpc_failure_reason": fault.get("nonlinear_failure_reason"),
            "mpc_failure_stage": fault.get("nonlinear_failure_stage"),
            "map_resolution_m": grid.resolution,
            "states": out,
            "classification": (
                "MPC_CORRIDOR_REJECTED_MAP_CLEAR" if out[1]["map_clearance_passes"]
                else "MAP_MARGIN_INSUFFICIENT"),
            "note": "Instantaneous map certificate; future swept motion, localization uncertainty, and next-cycle recursive feasibility require separate checks."}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", type=Path, required=True)
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--watcher", type=Path, required=True)
    ap.add_argument("--margin-m", type=float, default=.01)
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()
    if not math.isfinite(args.margin_m) or not 0 <= args.margin_m <= .20:
        ap.error("margin must be between 0 and 0.2 m")
    trajectory = np.loadtxt(args.trajectory, comments="#", delimiter=",")
    if (trajectory.ndim != 2 or trajectory.shape[1] < 9 or
            len(trajectory) < 3 or not np.isfinite(trajectory[:, :9]).all() or
            not np.all(np.diff(trajectory[:, 0]) > 0)):
        raise ValueError("invalid MPC trajectory")
    lap_length = (float(trajectory[-1, 0]) + float(np.linalg.norm(
        trajectory[-1, 1:3] - trajectory[0, 1:3])) - float(trajectory[0, 0]))
    grid = MapGrid.load(args.map)
    reports = []
    for line in args.watcher.read_text().splitlines():
        if not line.startswith("MPC_CYCLE_REJECTED: "):
            continue
        fault = json.loads(line.split(": ", 1)[1])
        if (fault.get("nonlinear_failure") or {}).get("valid"):
            reports.append(audit_fault(grid, trajectory, lap_length,
                                       fault, args.margin_m))
    output = {"map": str(args.map), "trajectory": str(args.trajectory),
              "watcher": str(args.watcher), "lap_length_m": lap_length,
              "faults": reports}
    rendered = json.dumps(output, indent=2)
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError("refusing to overwrite diagnostic report")
        args.output_json.write_text(rendered + "\n")
    print(rendered)
    return 0 if reports else 2


if __name__ == "__main__":
    raise SystemExit(main())
