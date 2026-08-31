#!/usr/bin/env python3
"""Apply a closed-track, curvature and acceleration limited speed profile.

The input and output use the competition-compatible nine-column raceline
format.  Geometry is preserved exactly; only ``vx_mps`` and ``ax_mps2`` are
recomputed.  This keeps Pure Pursuit on the validated map/raceline while
making its speed target tunable without hand-editing every waypoint.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 9:
                continue
            values = [float(value) for value in row[:9]]
            if all(math.isfinite(value) for value in values):
                rows.append(values)
    if len(rows) < 4:
        raise ValueError(f"need at least four numeric waypoints in {path}")
    if math.hypot(rows[0][1] - rows[-1][1], rows[0][2] - rows[-1][2]) < 1.0e-4:
        rows.pop()
    return rows


def build_profile(
    rows: list[list[float]],
    max_speed: float,
    min_speed: float,
    lateral_accel: float,
    accel: float,
    decel: float,
    passes: int,
) -> list[float]:
    count = len(rows)
    segment_lengths = [
        math.hypot(
            rows[(i + 1) % count][1] - rows[i][1],
            rows[(i + 1) % count][2] - rows[i][2],
        )
        for i in range(count)
    ]
    speeds = [
        min(max_speed, math.sqrt(lateral_accel / max(abs(row[4]), 1.0e-4)))
        for row in rows
    ]

    # Repeat around the cyclic path. The passes make both acceleration
    # directions converge without introducing an artificial start/finish stop.
    for _ in range(max(1, passes)):
        for i in range(count):
            nxt = (i + 1) % count
            limit = math.sqrt(max(0.0, speeds[i] ** 2 + 2.0 * accel * segment_lengths[i]))
            speeds[nxt] = min(speeds[nxt], limit)
        for i in range(count - 1, -1, -1):
            nxt = (i + 1) % count
            limit = math.sqrt(max(0.0, speeds[nxt] ** 2 + 2.0 * decel * segment_lengths[i]))
            speeds[i] = min(speeds[i], limit)

    return [max(min_speed, speed) for speed in speeds]


def save_rows(path: Path, rows: list[list[float]], speeds: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_left_m,d_right_m\n")
        for i, row in enumerate(rows):
            nxt = (i + 1) % len(rows)
            ds = math.hypot(rows[nxt][1] - row[1], rows[nxt][2] - row[2])
            next_speed = speeds[nxt]
            ax = (next_speed * next_speed - speeds[i] * speeds[i]) / max(2.0 * ds, 1.0e-6)
            stream.write(
                f"{row[0]:.6f},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},"
                f"{row[4]:.6f},{speeds[i]:.6f},{ax:.6f},{row[7]:.6f},{row[8]:.6f}\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-speed", type=float, default=0.85)
    parser.add_argument("--min-speed", type=float, default=0.12)
    parser.add_argument("--lateral-accel", type=float, default=2.0)
    parser.add_argument("--accel", type=float, default=1.0)
    parser.add_argument("--decel", type=float, default=2.0)
    parser.add_argument("--passes", type=int, default=8)
    args = parser.parse_args()
    if not (0.0 < args.min_speed <= args.max_speed):
        raise ValueError("min-speed must be positive and no greater than max-speed")
    if min(args.lateral_accel, args.accel, args.decel) <= 0.0:
        raise ValueError("acceleration limits must be positive")

    rows = load_rows(args.input)
    speeds = build_profile(
        rows, args.max_speed, args.min_speed, args.lateral_accel,
        args.accel, args.decel, args.passes)
    save_rows(args.output, rows, speeds)
    print(
        f"wrote {len(rows)} waypoints to {args.output}; "
        f"speed={min(speeds):.3f}..{max(speeds):.3f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
