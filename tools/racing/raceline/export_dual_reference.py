#!/usr/bin/env python3
"""Split a runtime raceline into frozen steering geometry and a track-s speed schedule."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.racing.track_projection import TrackProjection  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next((line.lstrip()[1:].strip() for line in lines if line.lstrip().startswith("#")), None)
    data = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if not data:
        raise ValueError(f"no trajectory data in {path}")
    if header is None:
        header = data.pop(0)
    names = [name.strip() for name in header.split(",")]
    return names, list(csv.DictReader(data, fieldnames=names))


def value(row: dict[str, str], *names: str, required: bool = True) -> float | None:
    key = next((name for name in names if name in row and row[name] != ""), None)
    if key is None:
        if required:
            raise ValueError(f"trajectory lacks required field; expected one of {names}")
        return None
    result = float(row[key])
    if not math.isfinite(result):
        raise ValueError(f"trajectory field {key} contains a non-finite value")
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def interpolate_periodic(x: float, xs: list[float], ys: list[float], period: float) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        raise ValueError("periodic schedule needs at least two paired samples")
    x %= period
    index = bisect.bisect_right(xs, x)
    low = index - 1
    high = index
    x0, x1 = xs[low], xs[high] if high < len(xs) else xs[0] + period
    y0, y1 = ys[low], ys[high] if high < len(ys) else ys[0]
    if low < 0:
        x0, x1 = xs[-1] - period, xs[0]
        y0, y1 = ys[-1], ys[0]
    if x < x0:
        x += period
    fraction = (x - x0) / (x1 - x0)
    return y0 + fraction * (y1 - y0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--centerline", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    trajectory_path = args.trajectory if args.trajectory.is_absolute() else ROOT / args.trajectory
    centerline_path = args.centerline if args.centerline.is_absolute() else ROOT / args.centerline
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    for label, path in (("trajectory", trajectory_path), ("centerline", centerline_path)):
        if not path.is_file():
            parser.error(f"{label} not found: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")

    source_fields, source_rows = read_csv(trajectory_path)
    if len(source_rows) < 3:
        parser.error("trajectory must contain at least three points")
    for required in ("s_m", "x_m", "y_m", "psi_rad", "kappa_radpm"):
        if required not in source_fields:
            parser.error(f"trajectory lacks required field {required}")
    speed_field = next((field for field in ("velocity_mps", "vx_mps", "target_speed_mps") if field in source_fields), None)
    accel_field = next((field for field in ("acceleration_mps2", "ax_mps2", "target_accel_mps2") if field in source_fields), None)
    if speed_field is None:
        parser.error("trajectory lacks velocity_mps/vx_mps/target_speed_mps")

    centerline = TrackProjection.from_csv(centerline_path, closed=True)
    steering_rows: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    modulo_progress: list[float] = []
    projection_distances: list[float] = []
    previous_segment: int | None = None
    wrap_offset = 0.0
    previous_modulo: float | None = None
    previous_unwrapped: float | None = None
    wrap_count = 0

    for index, row in enumerate(source_rows):
        x = float(value(row, "x_m"))
        y = float(value(row, "y_m"))
        yaw = float(value(row, "psi_rad"))
        projection = centerline.project(
            x, y, yaw,
            previous_segment=previous_segment,
            local_search_radius=20 if previous_segment is not None else None,
        )
        previous_segment = projection.segment_index
        progress = projection.s_m
        if previous_modulo is not None and progress < previous_modulo - centerline.total_length / 2.0:
            wrap_offset += centerline.total_length
            wrap_count += 1
        elif previous_modulo is not None and progress > previous_modulo + centerline.total_length / 2.0:
            raise ValueError(f"canonical progress reversed at trajectory row {index}")
        unwrapped = progress + wrap_offset
        if previous_unwrapped is not None and unwrapped <= previous_unwrapped:
            raise ValueError(f"canonical track_s is not strictly increasing at trajectory row {index}")
        previous_modulo = progress
        previous_unwrapped = unwrapped
        modulo_progress.append(progress)
        projection_distances.append(projection.distance_m)

        steering_rows.append({
            "track_s_m": f"{progress:.9f}",
            "track_s_unwrapped_m": f"{unwrapped:.9f}",
            "path_s_m": row["s_m"],
            "x_m": row["x_m"],
            "y_m": row["y_m"],
            "psi_rad": row["psi_rad"],
            "kappa_radpm": row["kappa_radpm"],
            "corridor_left_m": f"{projection.left_width_m - projection.lateral_offset_m:.9f}",
            "corridor_right_m": f"{projection.right_width_m + projection.lateral_offset_m:.9f}",
        })
        schedule_rows.append({
            "track_s_m": f"{progress:.9f}",
            "target_speed_mps": row[speed_field],
            "target_accel_mps2": row[accel_field] if accel_field is not None else "",
            "planning_kappa_radpm": row["kappa_radpm"],
        })

    track_s_span = float(steering_rows[-1]["track_s_unwrapped_m"]) - float(
        steering_rows[0]["track_s_unwrapped_m"]
    )
    if wrap_count > 1 or abs(track_s_span - centerline.total_length) > max(
        0.5, 0.02 * centerline.total_length
    ):
        raise ValueError(
            "source trajectory does not cover one canonical lap: "
            f"wraps={wrap_count}, span={track_s_span:.6f} m, "
            f"canonical_length={centerline.total_length:.6f} m"
        )
    if len(modulo_progress) != len(set(modulo_progress)):
        raise ValueError("canonical modulo track_s has duplicate schedule keys")

    schedule_rows.sort(key=lambda row: float(row["track_s_m"]))
    schedule_x = [float(row["track_s_m"]) for row in schedule_rows]
    speed_values = [float(row["target_speed_mps"]) for row in schedule_rows]
    accel_values = (
        [float(row["target_accel_mps2"]) for row in schedule_rows]
        if accel_field is not None else None
    )
    runtime_rows: list[dict[str, Any]] = []
    for point, source in zip(steering_rows, source_rows):
        track_s = float(point["track_s_m"])
        joined = dict(source)
        joined[speed_field] = f"{interpolate_periodic(track_s, schedule_x, speed_values, centerline.total_length):.9f}"
        if accel_field is not None and accel_values is not None:
            joined[accel_field] = f"{interpolate_periodic(track_s, schedule_x, accel_values, centerline.total_length):.9f}"
        runtime_rows.append(joined)

    # Every geometry sample is also a speed-schedule knot; the periodic join
    # must reproduce its original speed and acceleration at that same knot.
    max_position_delta = 0.0
    max_speed_delta = 0.0
    max_accel_delta = 0.0
    for source, joined in zip(source_rows, runtime_rows):
        max_position_delta = max(
            max_position_delta,
            abs(float(source["x_m"]) - float(joined["x_m"])),
            abs(float(source["y_m"]) - float(joined["y_m"])),
        )
        max_speed_delta = max(max_speed_delta, abs(float(source[speed_field]) - float(joined[speed_field])))
        if accel_field is not None:
            max_accel_delta = max(max_accel_delta, abs(float(source[accel_field]) - float(joined[accel_field])))
    if max_position_delta > 1.0e-12 or max_speed_delta > 2.0e-8 or max_accel_delta > 2.0e-8:
        raise ValueError(
            "split-reference round trip changed the parent trajectory: "
            f"position={max_position_delta:g}, speed={max_speed_delta:g}, accel={max_accel_delta:g}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "steering_path.csv",
        ["track_s_m", "track_s_unwrapped_m", "path_s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "corridor_left_m", "corridor_right_m"],
        steering_rows,
    )
    write_csv(
        output_dir / "speed_schedule.csv",
        ["track_s_m", "target_speed_mps", "target_accel_mps2", "planning_kappa_radpm"],
        schedule_rows,
    )
    write_csv(output_dir / "joined_runtime.csv", source_fields, runtime_rows)
    report = {
        "schema_version": 1,
        "source_trajectory": str(trajectory_path.relative_to(ROOT) if trajectory_path.is_relative_to(ROOT) else trajectory_path),
        "source_trajectory_sha256": sha256(trajectory_path),
        "canonical_centerline": str(centerline_path.relative_to(ROOT) if centerline_path.is_relative_to(ROOT) else centerline_path),
        "canonical_centerline_sha256": sha256(centerline_path),
        "canonical_track_length_m": centerline.total_length,
        "trajectory_point_count": len(source_rows),
        "canonical_wrap_count": wrap_count,
        "track_s_first_m": modulo_progress[0],
        "track_s_last_m": modulo_progress[-1],
        "track_s_unwrapped_span_m": float(steering_rows[-1]["track_s_unwrapped_m"]) - float(steering_rows[0]["track_s_unwrapped_m"]),
        "projection_distance_m": {
            "p50": sorted(projection_distances)[len(projection_distances) // 2],
            "p95": sorted(projection_distances)[math.ceil(0.95 * len(projection_distances)) - 1],
            "maximum": max(projection_distances),
        },
        "round_trip_max_error": {
            "position_m": max_position_delta,
            "speed_mps": max_speed_delta,
            "acceleration_mps2": max_accel_delta,
        },
        "outputs": ["steering_path.csv", "speed_schedule.csv", "joined_runtime.csv"],
    }
    (output_dir / "dual_reference_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
