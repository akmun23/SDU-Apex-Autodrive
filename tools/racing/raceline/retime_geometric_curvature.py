#!/usr/bin/env python3
"""Exploratory fixed-threshold curvature retiming diagnostic.

This tool does not model this vehicle's speed/steering/history-dependent
response. Its user-supplied constant curvature threshold is not an empirical
vehicle limit and must not be used to approve, rank, or publish a raceline.
It only demonstrates how a chosen kinematic threshold changes a speed profile.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trajectory(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(
        (line.lstrip()[1:].strip() for line in lines if line.lstrip().startswith("#")),
        None,
    )
    data = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not data:
        raise ValueError(f"trajectory has no rows: {path}")
    if header is None:
        header = data.pop(0)
    fields = [field.strip() for field in header.split(",")]
    rows = list(csv.DictReader(data, fieldnames=fields))
    required = {"s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "velocity_mps"}
    if not required.issubset(fields):
        raise ValueError(f"trajectory is missing columns: {sorted(required - set(fields))}")
    if len(rows) < 3:
        raise ValueError("closed trajectory needs at least three rows")
    return fields, rows


def closed_arc_steps(rows: list[dict[str, str]]) -> tuple[list[float], float]:
    s = [float(row["s_m"]) for row in rows]
    if abs(s[0]) > 1.0e-8 or any(b <= a for a, b in zip(s, s[1:])):
        raise ValueError("s_m must start at zero and increase strictly")
    closing = math.hypot(
        float(rows[0]["x_m"]) - float(rows[-1]["x_m"]),
        float(rows[0]["y_m"]) - float(rows[-1]["y_m"]),
    )
    if closing <= 1.0e-5:
        raise ValueError("trajectory contains a duplicate closing point")
    ds = [b - a for a, b in zip(s, s[1:])] + [closing]
    return ds, s[-1] + closing


def lap_time(speed: list[float], ds: list[float]) -> float:
    return sum(2.0 * step / (speed[i] + speed[(i + 1) % len(speed)])
               for i, step in enumerate(ds))


def cyclic_reachability(speed: list[float], ds: list[float],
                        accel_limit: float, brake_limit: float) -> int:
    """Apply forward/backward kinematic reachability until the lap is periodic."""
    n = len(speed)
    for pass_index in range(4 * n):
        changed = False
        for i in range(n):
            j = (i + 1) % n
            reachable = math.sqrt(max(0.0, speed[i] ** 2 + 2.0 * accel_limit * ds[i]))
            if speed[j] > reachable:
                speed[j] = reachable
                changed = True
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            reachable = math.sqrt(max(0.0, speed[j] ** 2 + 2.0 * brake_limit * ds[i]))
            if speed[i] > reachable:
                speed[i] = reachable
                changed = True
        if not changed:
            return pass_index + 1
    raise RuntimeError("closed-loop speed reachability did not converge")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--lateral-accel-limit", type=float, required=True,
        help=("EXPLORATORY constant threshold on v^2*|curvature| only; "
              "not a measured vehicle capability or optimizer constraint"))
    parser.add_argument("--accel-limit", type=float, default=3.8,
                        help="forward longitudinal reachability in m/s^2")
    parser.add_argument("--brake-limit", type=float, default=5.0,
                        help="braking reachability magnitude in m/s^2")
    args = parser.parse_args()

    source = args.trajectory.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        parser.error(f"trajectory not found: {source}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    if min(args.lateral_accel_limit, args.accel_limit, args.brake_limit) <= 0.0:
        parser.error("all acceleration limits must be positive")

    fields, rows = load_trajectory(source)
    ds, length = closed_arc_steps(rows)
    original_speed = [float(row["velocity_mps"]) for row in rows]
    curvature = [float(row["kappa_radpm"]) for row in rows]
    if any(not math.isfinite(v) or v <= 0.0 for v in original_speed):
        raise ValueError("velocity_mps must contain finite positive speeds")
    if any(not math.isfinite(k) for k in curvature):
        raise ValueError("kappa_radpm must contain finite values")

    geometry_hash = hashlib.sha256()
    for row in rows:
        geometry_hash.update(("\0".join(row[name] for name in
            ("s_m", "x_m", "y_m", "psi_rad", "kappa_radpm")) + "\n").encode())

    geometric_load = [v * v * abs(k) for v, k in zip(original_speed, curvature)]
    speed_cap = [min(v, math.sqrt(args.lateral_accel_limit / max(abs(k), 1.0e-9)))
                 for v, k in zip(original_speed, curvature)]
    passes = cyclic_reachability(speed_cap, ds, args.accel_limit, args.brake_limit)
    retimed_load = [v * v * abs(k) for v, k in zip(speed_cap, curvature)]

    acceleration = []
    n = len(rows)
    for i in range(n):
        previous = (i - 1) % n
        following = (i + 1) % n
        ds_span = ds[previous] + ds[i]
        acceleration.append(
            (speed_cap[following] ** 2 - speed_cap[previous] ** 2) / (2.0 * ds_span)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_out = output_dir / "retimed_raceline.csv"
    with trajectory_out.open("w", newline="", encoding="utf-8") as stream:
        stream.write("#" + ",".join(fields) + "\n")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        for i, source_row in enumerate(rows):
            row: dict[str, Any] = dict(source_row)
            row["velocity_mps"] = f"{speed_cap[i]:.9f}"
            row["acceleration_mps2"] = f"{acceleration[i]:.9f}"
            writer.writerow(row)

    output_fields, output_rows = load_trajectory(trajectory_out)
    if output_fields != fields or len(output_rows) != len(rows):
        raise RuntimeError("retimed output changed trajectory schema or row count")
    output_geometry_hash = hashlib.sha256()
    for row in output_rows:
        output_geometry_hash.update(("\0".join(row[name] for name in
            ("s_m", "x_m", "y_m", "psi_rad", "kappa_radpm")) + "\n").encode())
    if output_geometry_hash.hexdigest() != geometry_hash.hexdigest():
        raise RuntimeError("retiming changed path geometry")
    if max(retimed_load) > args.lateral_accel_limit + 1.0e-7:
        raise RuntimeError("retimed profile violates geometric lateral-load cap")

    original_time = lap_time(original_speed, ds)
    retimed_time = lap_time(speed_cap, ds)
    report = {
        "schema_version": 1,
        "candidate_kind": "fixed_geometry_geometric_curvature_speed_plan",
        "status": "diagnostic_only_unvalidated_constant_threshold",
        "simulator_run_authorized": False,
        "vehicle_capability_claim": False,
        "source_trajectory": str(source),
        "source_sha256": sha256(source),
        "path_geometry_sha256_before_and_after": geometry_hash.hexdigest(),
        "path_geometry_changed": False,
        "path_length_m": length,
        "points": n,
        "limits": {
            "lateral_accel_mps2": args.lateral_accel_limit,
            "forward_accel_mps2": args.accel_limit,
            "braking_mps2": args.brake_limit,
        },
        "cyclic_reachability_passes": passes,
        "geometric_lateral_load_mps2": {
            "source_max": max(geometric_load),
            "source_p95": sorted(geometric_load)[math.ceil(0.95 * n) - 1],
            "source_points_over_limit": sum(x > args.lateral_accel_limit for x in geometric_load),
            "retimed_max": max(retimed_load),
        },
        "speed_mps": {
            "source_min": min(original_speed),
            "source_max": max(original_speed),
            "retimed_min": min(speed_cap),
            "retimed_max": max(speed_cap),
        },
        "longitudinal_acceleration_mps2": {
            "retimed_min": min(acceleration),
            "retimed_max": max(acceleration),
        },
        "kinematic_time_proxy_s": {
            "source": original_time,
            "retimed": retimed_time,
            "change_s": retimed_time - original_time,
        },
        "outputs": [trajectory_out.name],
        "limitations": [
            "The time proxy assumes the speed schedule is achieved exactly.",
            "The supplied constant threshold is arbitrary unless separately justified; it is not a learned or measured tire-force/capability model.",
            "This output is rejected for optimizer use until replaced by a whole-run-validated vehicle response model conditioned on speed, steering, and transient state.",
            "This does not simulate MPC convergence, actuator history, wheel slip, or collision risk.",
            "A map ray-cast and simulator run cannot validate the underlying vehicle model by themselves.",
        ],
    }
    (output_dir / "retiming_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
