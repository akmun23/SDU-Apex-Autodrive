#!/usr/bin/env python3
"""Apply interpretable sector speed and phase parameters to a frozen P0 path.

The generated time is only a kinematic schedule proxy. This tool does not model
vehicle response, MPC convergence, tire limits, or wall risk, and does not promote
a candidate for simulation. Positive phase shift moves a schedule feature later
in track progress (v_new(s) = v_parent(s - shift)).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
VEHICLE_MODEL_SPEC = ROOT / "config/racing/racing_vehicle_model.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, *, comment_header: bool = False) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if comment_header:
        header = next((line.lstrip()[1:].strip() for line in lines if line.lstrip().startswith("#")), None)
        data = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        if header is None:
            header = data.pop(0)
        fields = [field.strip() for field in header.split(",")]
        return fields, list(csv.DictReader(data, fieldnames=fields))
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def interpolate_periodic(x: float, xs: list[float], ys: list[float], period: float) -> float:
    x %= period
    index = bisect.bisect_right(xs, x)
    if index == 0:
        x0, x1, y0, y1 = xs[-1] - period, xs[0], ys[-1], ys[0]
    elif index == len(xs):
        x0, x1, y0, y1 = xs[-1], xs[0] + period, ys[-1], ys[0]
    else:
        x0, x1, y0, y1 = xs[index - 1], xs[index], ys[index - 1], ys[index]
    if x < x0:
        x += period
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


def smoothstep(value: float) -> float:
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


def split_arc(start: float, end: float, period: float) -> list[tuple[float, float]]:
    start %= period
    end %= period
    if math.isclose(start, end, abs_tol=1e-10):
        raise ValueError("zero/full-period sector is ambiguous")
    return [(start, end)] if end > start else [(start, period), (0.0, end)]


def phase_arcs(sector_ids: list[str], sectors_by_id: dict[str, dict[str, Any]], period: float) -> list[tuple[float, float]]:
    linear = []
    for sid in sector_ids:
        sector = sectors_by_id[sid]
        linear.extend(split_arc(float(sector["start_track_s_m"]), float(sector["end_track_s_m"]), period))
    linear.sort()
    merged: list[list[float]] = []
    for start, end in linear:
        if merged and start <= merged[-1][1] + 1e-7:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    if len(merged) > 1 and merged[0][0] <= 1e-7 and merged[-1][1] >= period - 1e-7:
        wrapped = (merged[-1][0], merged[0][1] + period)
        merged = merged[1:-1] + [[wrapped[0], wrapped[1]]]
    return [(start, end) for start, end in merged]


def phase_weight(s: float, arcs: list[tuple[float, float]], period: float, blend_m: float) -> float:
    best = 0.0
    for start, end in arcs:
        length = end - start
        progress = (s - start) % period
        if progress <= length + 1e-9:
            edge_distance = min(progress, max(0.0, length - progress))
            weight = 1.0 if blend_m <= 0.0 else smoothstep(edge_distance / blend_m)
            best = max(best, weight)
    return best


def validate_and_load_config(path: Path) -> tuple[dict[str, Any], float, list[dict[str, Any]], dict[str, str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    period = float(config["period_m"])
    sectors = list(config["sectors"])
    ids = [str(sector["id"]) for sector in sectors]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sector IDs")
    sector_group: dict[str, str] = {}
    for group in config["optimization_groups"]:
        for sid in group["sectors"]:
            if sid in sector_group:
                raise ValueError(f"sector {sid} belongs to more than one speed group")
            sector_group[str(sid)] = str(group["id"])
    if set(sector_group) != set(ids):
        raise ValueError("optimization groups must cover every sector exactly once")
    # Reuse the sector-time tool's complete coverage validation by checking the
    # sorted split arcs locally.
    pieces = sorted(piece for sector in sectors for piece in split_arc(
        float(sector["start_track_s_m"]), float(sector["end_track_s_m"]), period
    ))
    cursor = 0.0
    for start, end in pieces:
        if abs(start - cursor) > 1e-6:
            raise ValueError(f"sector cover gap/overlap at {cursor:.6f} m")
        cursor = end
    if abs(cursor - period) > 1e-6:
        raise ValueError("sectors do not cover one canonical lap")
    return config, period, sectors, sector_group


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sectors", type=Path, required=True)
    parser.add_argument("--dual-reference-dir", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True, help="JSON candidate parameters")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-blend-m", type=float, default=0.25)
    parser.add_argument("--phase-blend-m", type=float, default=0.40)
    args = parser.parse_args()

    sectors_path = args.sectors if args.sectors.is_absolute() else ROOT / args.sectors
    ref_dir = args.dual_reference_dir if args.dual_reference_dir.is_absolute() else ROOT / args.dual_reference_dir
    params_path = args.parameters if args.parameters.is_absolute() else ROOT / args.parameters
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"output directory is not empty: {output_dir}")
    if args.boundary_blend_m < 0.0 or args.phase_blend_m < 0.0:
        parser.error("blend distances cannot be negative")

    config, period, sectors, sector_group = validate_and_load_config(sectors_path)
    reference_manifest = json.loads((ref_dir / "dual_reference_manifest.json").read_text(encoding="utf-8"))
    if reference_manifest["source_trajectory_sha256"] != config["parent_trajectory_sha256"]:
        parser.error("dual-reference trajectory hash does not match the declared P0 parent")
    if reference_manifest["canonical_centerline_sha256"] != config["centerline_sha256"]:
        parser.error("dual-reference centerline hash does not match the sector configuration")
    parameters = json.loads(params_path.read_text(encoding="utf-8"))
    candidate_id = str(parameters.get("candidate_id", params_path.stem))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", candidate_id):
        parser.error("candidate_id must be 1-64 safe filename characters")
    if parameters.get("parent", config.get("parent")) != config.get("parent"):
        parser.error("candidate parameter parent does not match frozen P0")

    schedule_fields, parent_rows = read_csv(ref_dir / "speed_schedule.csv")
    steering_fields, steering_rows = read_csv(ref_dir / "steering_path.csv")
    schedule_s = [float(row["track_s_m"]) for row in parent_rows]
    parent_speed = [float(row["target_speed_mps"]) for row in parent_rows]
    parent_accel = [float(row["target_accel_mps2"]) for row in parent_rows]
    group_scales = {str(group["id"]): float(group.get("speed_scale", 1.0)) for group in config["optimization_groups"]}
    unknown_scale_groups = set(parameters.get("speed_scale", {})) - set(group_scales)
    if unknown_scale_groups:
        parser.error(f"unknown speed-scale groups: {sorted(unknown_scale_groups)}")
    group_scales.update({str(key): float(value) for key, value in parameters.get("speed_scale", {}).items()})
    if any(not math.isfinite(value) or value <= 0.0 for value in group_scales.values()):
        parser.error("speed scales must be finite and positive")

    sectors_by_id = {str(sector["id"]): sector for sector in sectors}
    phase_ranges: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for group in config.get("phase_groups", []):
        group_id = str(group["id"])
        phase_ranges[(group_id, "brake")] = phase_arcs(group["brake_sectors"], sectors_by_id, period)
        phase_ranges[(group_id, "exit_accel")] = phase_arcs(group["exit_accel_sectors"], sectors_by_id, period)
    brake_offsets = {str(k): float(v) for k, v in parameters.get("brake_phase_shift_m", {}).items()}
    accel_offsets = {str(k): float(v) for k, v in parameters.get("exit_accel_phase_shift_m", {}).items()}
    if any(not math.isfinite(value) for value in (*brake_offsets.values(), *accel_offsets.values())):
        parser.error("phase shifts must be finite")
    valid_phase_groups = {group_id for group_id, _ in phase_ranges}
    if set(brake_offsets) - valid_phase_groups or set(accel_offsets) - valid_phase_groups:
        parser.error("phase-shift parameter names must match configured phase groups")

    sector_intervals = []
    for sector in sectors:
        start = float(sector["start_track_s_m"]) % period
        end = float(sector["end_track_s_m"]) % period
        sector_intervals.append((start, end, str(sector["id"])))

    def sector_at(s: float) -> str:
        position = s % period
        for start, end, sid in sector_intervals:
            if start < end and start <= position < end:
                return sid
            if start > end and (position >= start or position < end):
                return sid
        raise ValueError(f"no sector for track progress {position:.6f}")

    boundary_values: list[tuple[float, float, float]] = []
    for boundary in sorted({float(sector["start_track_s_m"]) % period for sector in sectors}):
        left_sid = sector_at(boundary - 1e-6)
        right_sid = sector_at(boundary + 1e-6)
        left_scale = group_scales[sector_group[left_sid]]
        right_scale = group_scales[sector_group[right_sid]]
        boundary_values.append((boundary, left_scale, right_scale))

    def speed_scale_at(s: float, sector_id: str) -> float:
        base = group_scales[sector_group[sector_id]]
        width = args.boundary_blend_m
        if width <= 0.0:
            return base
        for boundary, left_value, right_value in boundary_values:
            delta = (s - boundary + period * 0.5) % period - period * 0.5
            if abs(delta) <= width and not math.isclose(left_value, right_value, abs_tol=1e-12):
                alpha = smoothstep((delta + width) / (2.0 * width))
                return left_value + alpha * (right_value - left_value)
        return base

    transformed_speed: list[float] = []
    for s in schedule_s:
        sector_id = sector_at(s)
        shift = 0.0
        for group_id, arcs in phase_ranges.items():
            corner_id, phase_id = group_id
            offset = brake_offsets.get(corner_id, 0.0) if phase_id == "brake" else accel_offsets.get(corner_id, 0.0)
            shift += offset * phase_weight(s, arcs, period, args.phase_blend_m)
        value = interpolate_periodic(s - shift, schedule_s, parent_speed, period)
        value *= speed_scale_at(s, sector_id)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"candidate produced invalid speed {value} at track_s={s}")
        transformed_speed.append(value)

    is_neutral = all(math.isclose(value, 1.0, abs_tol=1e-12) for value in group_scales.values()) and all(
        math.isclose(value, 0.0, abs_tol=1e-12) for value in (*brake_offsets.values(), *accel_offsets.values())
    )
    if is_neutral:
        transformed_speed = parent_speed.copy()
        transformed_accel = parent_accel.copy()
    else:
        path_rows = sorted(steering_rows, key=lambda row: float(row["track_s_m"]))
        path_s = [float(row["path_s_m"]) for row in path_rows]
        track_s_for_path = [float(row["track_s_m"]) for row in path_rows]
        path_period = path_s[-1] + math.hypot(
            float(path_rows[-1]["x_m"]) - float(path_rows[0]["x_m"]),
            float(path_rows[-1]["y_m"]) - float(path_rows[0]["y_m"]),
        )
        schedule_path_s = [interpolate_periodic(s, track_s_for_path, path_s, period) for s in schedule_s]
        transformed_accel = []
        count = len(schedule_s)
        for index, speed in enumerate(transformed_speed):
            previous = (index - 1) % count
            following = (index + 1) % count
            ds_prev = (schedule_path_s[index] - schedule_path_s[previous]) % path_period
            ds_next = (schedule_path_s[following] - schedule_path_s[index]) % path_period
            transformed_accel.append(
                (transformed_speed[following] ** 2 - transformed_speed[previous] ** 2)
                / (2.0 * (ds_prev + ds_next))
            )

    schedule_rows: list[dict[str, Any]] = []
    for index, row in enumerate(parent_rows):
        copy = dict(row)
        copy["target_speed_mps"] = f"{transformed_speed[index]:.9f}"
        copy["target_accel_mps2"] = f"{transformed_accel[index]:.9f}"
        schedule_rows.append(copy)

    source_path = ROOT / reference_manifest["source_trajectory"]
    source_fields, source_rows = read_csv(source_path, comment_header=True)
    if len(source_rows) != len(steering_rows):
        parser.error("source trajectory and frozen steering geometry row counts differ")
    speed_field = next(name for name in ("velocity_mps", "vx_mps", "target_speed_mps") if name in source_fields)
    accel_field = next(name for name in ("acceleration_mps2", "ax_mps2", "target_accel_mps2") if name in source_fields)
    joined_rows = []
    max_geometry_delta = 0.0
    for geometry, source in zip(steering_rows, source_rows):
        s = float(geometry["track_s_m"])
        row = dict(source)
        max_geometry_delta = max(
            max_geometry_delta,
            abs(float(row["x_m"]) - float(geometry["x_m"])),
            abs(float(row["y_m"]) - float(geometry["y_m"])),
        )
        row[speed_field] = f"{interpolate_periodic(s, schedule_s, transformed_speed, period):.9f}"
        row[accel_field] = f"{interpolate_periodic(s, schedule_s, transformed_accel, period):.9f}"
        joined_rows.append(row)

    path_rows = sorted(steering_rows, key=lambda row: float(row["track_s_m"]))
    path_s = [float(row["path_s_m"]) for row in path_rows]
    path_period = path_s[-1] + math.hypot(
        float(path_rows[-1]["x_m"]) - float(path_rows[0]["x_m"]),
        float(path_rows[-1]["y_m"]) - float(path_rows[0]["y_m"]),
    )
    schedule_path_s = [interpolate_periodic(s, [float(row["track_s_m"]) for row in path_rows], path_s, period) for s in schedule_s]
    ideal_time = 0.0
    for index, speed in enumerate(transformed_speed):
        following = (index + 1) % len(transformed_speed)
        ds = (schedule_path_s[following] - schedule_path_s[index]) % path_period
        ideal_time += 2.0 * ds / (speed + transformed_speed[following])

    parent_accel_min, parent_accel_max = min(parent_accel), max(parent_accel)
    candidate_accel_min, candidate_accel_max = min(transformed_accel), max(transformed_accel)
    acceleration_outside_parent_schedule_range = (
        candidate_accel_min < parent_accel_min - 1e-6
        or candidate_accel_max > parent_accel_max + 1e-6
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "speed_schedule.csv", schedule_fields, schedule_rows)
    write_csv(output_dir / "joined_runtime.csv", source_fields, joined_rows)
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "parent": config.get("parent"),
        "candidate_status": "offline_schedule_only_not_sim_validated",
        "vehicle_model_sha256": sha256(VEHICLE_MODEL_SPEC),
        "vehicle_model_role": "controller_prediction_contract; path generation still uses the frozen parent optimizer model",
        "reference_source_trajectory_sha256": reference_manifest["source_trajectory_sha256"],
        "reference_steering_path_sha256": sha256(ref_dir / "steering_path.csv"),
        "sector_config_sha256": sha256(sectors_path),
        "parameters": parameters,
        "boundary_blend_m": args.boundary_blend_m,
        "phase_blend_m": args.phase_blend_m,
        "steering_geometry_changed": False,
        "max_steering_geometry_delta_m": max_geometry_delta,
        "neutral_parent_roundtrip": is_neutral,
        "speed_range_mps": {"minimum": min(transformed_speed), "maximum": max(transformed_speed)},
        "acceleration_range_mps2": {"minimum": candidate_accel_min, "maximum": candidate_accel_max},
        "parent_schedule_acceleration_range_mps2": {"minimum": parent_accel_min, "maximum": parent_accel_max},
        "acceleration_outside_parent_schedule_range": acceleration_outside_parent_schedule_range,
        "idealized_schedule_time_proxy_s": ideal_time,
        "parent_schedule_time_proxy_s": sum(
            2.0 * ((schedule_path_s[(i + 1) % len(schedule_path_s)] - schedule_path_s[i]) % path_period)
            / (parent_speed[i] + parent_speed[(i + 1) % len(parent_speed)])
            for i in range(len(parent_speed))
        ),
        "outputs": ["speed_schedule.csv", "joined_runtime.csv"],
        "limitations": [
            "Time proxy assumes the target speed is achieved exactly and is not a lap-time prediction.",
            "No MPC, actuator, tire, state-history, corridor, or collision model is evaluated.",
            "The vehicle_model_sha256 fingerprints the expected controller model only; it does not claim optimizer/MPC dynamics parity.",
            "Acceleration outside the parent schedule range is unsupported by this P0 reference and requires separate validation.",
            "Candidate must pass model/replay/static gates before any simulator run.",
        ],
    }
    (output_dir / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "candidate_id", "candidate_status", "neutral_parent_roundtrip", "speed_range_mps",
        "acceleration_range_mps2", "acceleration_outside_parent_schedule_range",
        "idealized_schedule_time_proxy_s", "parent_schedule_time_proxy_s",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
