#!/usr/bin/env python3
"""Build a provenance-tracked yaw table with a newer low-speed source.

Rows at or below the replacement cutoff come from the low-speed candidate;
rows above it come from the high-speed development surface. Each source must
provide the same turn/steering grid, and the resulting table must be rectangular.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FIELDS = (
    "speed_knot_mps", "turn_sign", "steering_rad", "demand_q",
    "yaw_rate_abs_rps",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_surface(path: Path) -> dict[tuple[float, int, float], dict[str, float]]:
    rows: dict[tuple[float, int, float], dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(
            (line for line in stream if not line.lstrip().startswith("#")))
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: expected CSV columns {FIELDS}")
        for line_number, row in enumerate(reader, start=2):
            try:
                values = {name: float(row[name]) for name in FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric cell") from exc
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"{path}:{line_number}: non-finite cell")
            speed = round(values["speed_knot_mps"], 6)
            turn = int(round(values["turn_sign"]))
            steering = round(values["steering_rad"], 6)
            if turn not in (-1, 1) or values["steering_rad"] < 0.0:
                raise ValueError(f"{path}:{line_number}: invalid turn/steering")
            key = (speed, turn, steering)
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate knot {key}")
            rows[key] = values
    if not rows:
        raise ValueError(f"{path}: empty yaw surface")
    return rows


def merge_surfaces(low_path: Path, high_path: Path, output: Path,
                   replace_through_speed_mps: float,
                   keep_speed_knots: set[float] | None = None) -> dict[str, Any]:
    if not math.isfinite(replace_through_speed_mps):
        raise ValueError("replacement speed must be finite")
    low = _read_surface(low_path)
    high = _read_surface(high_path)
    low_grid = {(turn, steer) for _, turn, steer in low}
    high_grid = {(turn, steer) for _, turn, steer in high}
    if low_grid != high_grid:
        raise ValueError("low/high surfaces do not share the same steering grid")

    combined = {
        key: row for key, row in high.items()
        if key[0] > replace_through_speed_mps
    }
    combined.update({
        key: row for key, row in low.items()
        if key[0] <= replace_through_speed_mps
    })
    selected_speeds = None
    if keep_speed_knots is not None:
        selected_speeds = {round(speed, 6) for speed in keep_speed_knots}
        available_speeds = {key[0] for key in combined}
        missing_speeds = selected_speeds - available_speeds
        if missing_speeds:
            raise ValueError(
                "requested speed knots are absent after source merge: "
                f"{sorted(missing_speeds)}")
        combined = {
            key: row for key, row in combined.items()
            if key[0] in selected_speeds
        }
    speeds = sorted({key[0] for key in combined})
    if not 3 <= len(speeds) <= 5:
        raise ValueError(
            f"result must contain 3-5 model-compatible speed knots, got {len(speeds)}")
    expected = len(low_grid)
    for speed in speeds:
        count = sum(key[0] == speed for key in combined)
        if count != expected:
            raise ValueError(
                f"merged speed {speed:g} has {count} cells; expected {expected}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        stream.write(
            "# Merged candidate: newer low-speed surface through the configured cutoff; "
            "high-speed surface above it.\n")
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for key in sorted(combined):
            row = combined[key]
            writer.writerow({
                "speed_knot_mps": f"{row['speed_knot_mps']:.8f}",
                "turn_sign": str(int(round(row["turn_sign"]))),
                "steering_rad": f"{row['steering_rad']:.8f}",
                "demand_q": f"{row['demand_q']:.8f}",
                "yaw_rate_abs_rps": f"{row['yaw_rate_abs_rps']:.8f}",
            })

    return {
        "schema_version": 1,
        "output": output.as_posix(),
        "replacement_cutoff_speed_mps": replace_through_speed_mps,
        "selected_speed_knots_mps": speeds if selected_speeds is not None else None,
        "low_speed_source": {
            "path": low_path.as_posix(), "sha256": _sha256(low_path),
            "speed_knots_mps": sorted({key[0] for key in low}),
        },
        "high_speed_source": {
            "path": high_path.as_posix(), "sha256": _sha256(high_path),
            "speed_knots_mps": sorted({key[0] for key in high}),
        },
        "merged_speed_knots_mps": speeds,
        "turn_steering_cells_per_speed": expected,
        "row_count": len(combined),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-speed-surface", required=True, type=Path)
    parser.add_argument("--high-speed-surface", required=True, type=Path)
    parser.add_argument("--replace-through-speed-mps", required=True, type=float)
    parser.add_argument(
        "--keep-speed-knot", action="append", type=float, dest="keep_speed_knots",
        help="retain only these model speed knots (repeat; at most five are supported)")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = merge_surfaces(
        args.low_speed_surface, args.high_speed_surface, args.output,
        args.replace_through_speed_mps,
        set(args.keep_speed_knots) if args.keep_speed_knots else None)
    sidecar = args.output.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
