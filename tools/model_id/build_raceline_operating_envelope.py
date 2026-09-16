#!/usr/bin/env python3
"""Build the controller-reachable operating envelope from the production raceline.

This is an offline analysis tool.  It reads the production raceline and Pure
Pursuit limits, derives the coupled speed/curvature/acceleration/steering
quantities used by the controller, and optionally summarizes clean runtime
CSV reports.  It never starts the simulator and never changes controller or
simulator configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RACELINE = REPO_ROOT / (
    "f1tenth_planning/trajectories/"
    "autodrive_track_ftg_commit_20260909_025m_mintime_raceline.csv")
DEFAULT_PP_CONFIG = REPO_ROOT / "f1tenth_control/config/path_tracking_autodrive.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "sdu_apex_autodrive/artifacts/model_id_work"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "raceline_operating_envelope_v1.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_ROOT / "raceline_operating_envelope_v1.csv"

RACELINE_FIELDS = (
    "s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps",
    "ax_mps2", "d_left_m", "d_right_m",
)
SPEED_BIN_WIDTH_MPS = 1.0
CURVATURE_BIN_WIDTH_RADPM = 0.025


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "p50": None, "p95": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "p50": _percentile(finite, 0.50),
        "p95": _percentile(finite, 0.95),
    }


def _read_pp_config(path: Path) -> dict[str, float | str]:
    """Read the scalar PP limits without requiring PyYAML at analysis time."""
    values: dict[str, float | str] = {"max_speed": 16.0,
                                      "max_lateral_accel": 6.5,
                                      "wheelbase": 0.324,
                                      "wall_safety_margin": 0.03}
    pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
                         r"([-+0-9.eE]+)\s*(?:#.*)?$")
    for raw in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw)
        if match is None:
            continue
        key, value = match.groups()
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    return values


def _read_raceline(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        first = stream.readline()
        if not first:
            raise ValueError(f"empty raceline: {path}")
        header = [item.strip().lstrip("#").strip()
                  for item in first.rstrip("\n").split(",")]
        missing = sorted(set(RACELINE_FIELDS).difference(header))
        if missing:
            raise ValueError(f"{path} is missing raceline fields: {missing}")
        reader = csv.DictReader(stream, fieldnames=header)
        rows: list[dict[str, float]] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = {field: float(raw[field]) for field in RACELINE_FIELDS}
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid raceline row") from exc
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"{path}:{line_number}: non-finite raceline row")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"raceline contains fewer than two points: {path}")
    return rows


def _bin(value: float, width: float) -> int:
    return math.floor(value / width)


def _bin_label(value: int, width: float, field: str) -> str:
    low = value * width
    high = (value + 1) * width
    return f"{field}[{low:g},{high:g})"


def _occupancy_2d(rows: Sequence[dict[str, float]], field: str,
                  width: float, absolute: bool = False) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = abs(row[field]) if absolute else row[field]
        speed_label = _bin_label(_bin(row["vx_mps"], SPEED_BIN_WIDTH_MPS),
                                 SPEED_BIN_WIDTH_MPS, "speed_mps")
        value_label = _bin_label(_bin(value, width), width, field)
        label = f"{speed_label} x {value_label}"
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _by_speed(rows: Sequence[dict[str, float]], field: str,
              absolute: bool = False) -> dict[str, dict[str, float | int | None]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        speed_bin = _bin(row["vx_mps"], SPEED_BIN_WIDTH_MPS)
        value = abs(row[field]) if absolute else row[field]
        grouped.setdefault(speed_bin, []).append(value)
    return {
        _bin_label(index, SPEED_BIN_WIDTH_MPS, "speed_mps"): _stats(values)
        for index, values in sorted(grouped.items())
    }


def _read_runtime_csv(path: Path) -> tuple[list[dict[str, float]], set[str]]:
    """Read optional reports with known PP/state aliases.

    Reports with unrelated schemas are retained in the provenance list but do
    not contribute values.  This makes the extractor useful with both the
    calibration CSV and compact hand-built runtime summaries.
    """
    aliases = {
        "pp_target_speed_mps": ("pp_target_speed_mps", "target_speed_mps"),
        "pp_command_speed_mps": ("pp_command_speed_mps", "command_speed_mps",
                                  "speed_command_mps"),
        "pp_command_steering_rad": ("pp_command_steering_rad",
                                     "command_steering_rad",
                                     "steering_command_rad"),
        "applied_steering_rad": ("applied_steering_rad", "steering_rad"),
        "applied_throttle_norm": ("applied_throttle_norm", "throttle_norm"),
        "u_mps": ("u_mps", "odom_u_mps", "gt_speed_mps"),
        "v_mps": ("v_mps", "odom_v_mps", "com_v_mps"),
        "r_radps": ("r_radps", "yaw_rate_radps", "odom_r_radps"),
        "ax_mps2": ("ax_mps2", "longitudinal_accel_mps2"),
    }
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        for raw in reader:
            converted: dict[str, float] = {}
            for output, candidates in aliases.items():
                source = next((name for name in candidates if name in fields), None)
                if source is None:
                    continue
                try:
                    value = float(raw[source])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    converted[output] = value
            if converted:
                rows.append(converted)
    return rows, fields


def _runtime_summary(paths: Sequence[Path]) -> dict[str, Any]:
    combined: dict[str, list[float]] = {}
    available: dict[str, list[str]] = {}
    row_counts: dict[str, int] = {}
    for path in paths:
        rows, fields = _read_runtime_csv(path)
        row_counts[str(path)] = len(rows)
        available[str(path)] = sorted(fields)
        for row in rows:
            for field, value in row.items():
                combined.setdefault(field, []).append(value)
    return {
        "source_files": [str(path) for path in paths],
        "row_counts": row_counts,
        "available_input_fields": available,
        "distributions": {
            field: _stats(values) for field, values in sorted(combined.items())
        },
        "missing_command_pipeline_fields": [
            field for field in ("pp_target_speed_mps", "pp_command_speed_mps",
                                "pp_command_steering_rad", "applied_steering_rad",
                                "applied_throttle_norm", "u_mps", "v_mps", "r_radps")
            if field not in combined
        ],
    }


def build(raceline_path: Path = DEFAULT_RACELINE,
          pp_config_path: Path = DEFAULT_PP_CONFIG,
          runtime_paths: Sequence[Path] = (),
          speed_bin_width_mps: float = SPEED_BIN_WIDTH_MPS,
          curvature_bin_width_radpm: float = CURVATURE_BIN_WIDTH_RADPM,
          output_json: Path = DEFAULT_OUTPUT_JSON,
          output_csv: Path = DEFAULT_OUTPUT_CSV) -> dict[str, Any]:
    if speed_bin_width_mps <= 0.0 or curvature_bin_width_radpm <= 0.0:
        raise ValueError("bin widths must be positive")
    rows = _read_raceline(raceline_path)
    config = _read_pp_config(pp_config_path)
    wheelbase = float(config.get("wheelbase", 0.324))
    max_speed = float(config.get("max_speed", 16.0))
    max_lateral_accel = float(config.get("max_lateral_accel", 6.5))
    wall_margin = float(config.get("wall_safety_margin", 0.03))

    compact: list[dict[str, float | int | str]] = []
    derived: list[dict[str, float]] = []
    for row in rows:
        speed = max(0.0, row["vx_mps"])
        ay = speed * speed * row["kappa_radpm"]
        delta = math.atan(wheelbase * row["kappa_radpm"])
        speed_index = _bin(speed, speed_bin_width_mps)
        curvature_index = _bin(abs(row["kappa_radpm"]), curvature_bin_width_radpm)
        core_bin = f"s{speed_index}_k{curvature_index}"
        values = {
            **row,
            "ay_ref_mps2": ay,
            "delta_ff_rad": delta,
            "yaw_rate_ref_radps": speed * row["kappa_radpm"],
            "speed_bin": float(speed_index),
            "curvature_bin": float(curvature_index),
        }
        derived.append(values)
        compact.append({
            "s_m": row["s_m"], "v_ref_mps": speed,
            "kappa_radpm": row["kappa_radpm"], "ax_ref_mps2": row["ax_mps2"],
            "ay_ref_mps2": ay, "delta_ff_rad": delta,
            "d_left_m": row["d_left_m"], "d_right_m": row["d_right_m"],
            "core_bin_id": core_bin,
        })

    # Guard occupancy is a coupled expansion of occupied core bins.  It is
    # represented explicitly so consumers never reconstruct a rectangular
    # independent-max filter by accident.
    core_bins = {
        (int(row["speed_bin"]), int(row["curvature_bin"]))
        for row in derived
    }
    guard_bins = {
        (speed_index + speed_offset, curvature_index + curvature_offset)
        for speed_index, curvature_index in core_bins
        for speed_offset in (-1, 0, 1)
        for curvature_offset in (-1, 0, 1)
        if speed_index + speed_offset >= 0 and curvature_index + curvature_offset >= 0
    }

    by_speed = {
        "reference_speed_mps": _by_speed(derived, "vx_mps"),
        "abs_curvature_radpm": _by_speed(derived, "kappa_radpm", absolute=True),
        "lateral_acceleration_mps2": _by_speed(derived, "ay_ref_mps2", absolute=True),
        "nominal_steering_rad": _by_speed(derived, "delta_ff_rad", absolute=True),
        "longitudinal_acceleration_mps2": _by_speed(derived, "ax_mps2"),
        "yaw_rate_radps": _by_speed(derived, "yaw_rate_ref_radps", absolute=True),
        "left_corridor_m": _by_speed(derived, "d_left_m"),
        "right_corridor_m": _by_speed(derived, "d_right_m"),
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "raceline_operating_envelope_offline",
        "simulator_modified": False,
        "controller_configuration_modified": False,
        "raceline_file": str(raceline_path),
        "pure_pursuit_config": str(pp_config_path),
        "project_speed_ceiling_mps": 16.0,
        "controller_limits": {
            "max_speed_mps": max_speed,
            "max_lateral_accel_mps2": max_lateral_accel,
            "wheelbase_m": wheelbase,
            "wall_safety_margin_m": wall_margin,
        },
        "track": {
            "point_count": len(rows),
            "track_length_m": rows[-1]["s_m"] - rows[0]["s_m"],
            "reference_speed_mps": _stats(row["vx_mps"] for row in rows),
            "curvature_radpm": _stats(row["kappa_radpm"] for row in rows),
            "abs_curvature_radpm": _stats(abs(row["kappa_radpm"]) for row in rows),
            "lateral_acceleration_mps2": _stats(
                row["ay_ref_mps2"] for row in derived),
            "nominal_steering_rad": _stats(row["delta_ff_rad"] for row in derived),
            "longitudinal_acceleration_mps2": _stats(row["ax_mps2"] for row in rows),
            "corridor_left_m": _stats(row["d_left_m"] for row in rows),
            "corridor_right_m": _stats(row["d_right_m"] for row in rows),
        },
        "bins": {
            "speed_bin_width_mps": speed_bin_width_mps,
            "curvature_bin_width_radpm": curvature_bin_width_radpm,
            "core_bin_count": len(core_bins),
            "guard_bin_count": len(guard_bins),
            "core_bin_ids": [f"s{speed}_k{curvature}" for speed, curvature in sorted(core_bins)],
            "guard_bin_ids": [f"s{speed}_k{curvature}" for speed, curvature in sorted(guard_bins)],
        },
        "core_envelope": {
            "definition": "occupied raceline speed x absolute-curvature bins",
            "excludes_above_speed_ceiling": True,
            "speed_mps": [0.0, min(max_speed, max(row["vx_mps"] for row in rows))],
            "occupancy_speed_x_abs_curvature": _occupancy_2d(
                derived, "kappa_radpm", curvature_bin_width_radpm, absolute=True),
            "by_speed_bin": by_speed,
        },
        "guard_envelope": {
            "definition": "one neighboring speed/curvature bin around occupied core bins",
            "speed_mps": [0.0, max_speed],
            "expansion": {"speed_bins": 1, "curvature_bins": 1,
                          "curvature_expansion_description": "approximately 20% or one bin"},
        },
        "stress_envelope": {
            "definition": "under-ceiling states outside core and guard occupancy",
            "included_in_promotion_percentiles": False,
        },
        "occupancy_2d": {
            "speed_x_abs_curvature": _occupancy_2d(
                derived, "kappa_radpm", curvature_bin_width_radpm, absolute=True),
            "speed_x_abs_steering": _occupancy_2d(
                derived, "delta_ff_rad", 0.01, absolute=True),
            "speed_x_abs_yaw_rate": _occupancy_2d(
                derived, "yaw_rate_ref_radps", 0.05, absolute=True),
            "speed_x_ax": _occupancy_2d(derived, "ax_mps2", 0.5, absolute=False),
        },
        "runtime_reports": _runtime_summary(runtime_paths),
        "outputs": {"compact_csv": str(output_csv)},
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        fields = ("s_m", "v_ref_mps", "kappa_radpm", "ax_ref_mps2",
                  "ay_ref_mps2", "delta_ff_rad", "d_left_m", "d_right_m",
                  "core_bin_id")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compact)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raceline", type=Path, default=DEFAULT_RACELINE)
    parser.add_argument("--pp-config", type=Path, default=DEFAULT_PP_CONFIG)
    parser.add_argument("--runtime-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--speed-bin-width-mps", type=float,
                        default=SPEED_BIN_WIDTH_MPS)
    parser.add_argument("--curvature-bin-width-radpm", type=float,
                        default=CURVATURE_BIN_WIDTH_RADPM)
    args = parser.parse_args()
    report = build(
        args.raceline, args.pp_config, args.runtime_csv,
        args.speed_bin_width_mps, args.curvature_bin_width_radpm,
        args.output_json, args.output_csv)
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
        "track_length_m": report["track"]["track_length_m"],
        "max_raceline_speed_mps": report["track"]["reference_speed_mps"]["max"],
        "core_bins": report["bins"]["core_bin_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
