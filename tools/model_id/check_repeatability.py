#!/usr/bin/env python3
"""Compare repeated source-valid model-identification replays.

The input runs must already have passed the bridge source contract.  This
tool never fills missing packets: it aligns the recorded source samples by
relative source time and linearly interpolates only for comparison at common
reporting horizons.  The raw run files remain the authoritative data.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


SOURCE_DT_MIN_S = 0.015
SOURCE_DT_MAX_S = 0.035
REPORT_HORIZONS_S = (0.0, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 8.0)
STATE_FIELDS = (
    "simulator_position_x",
    "simulator_position_y",
    "simulator_position_z",
    "simulator_linear_velocity_x",
    "simulator_linear_velocity_y",
    "simulator_linear_velocity_z",
    "simulator_angular_velocity_x",
    "simulator_angular_velocity_y",
    "simulator_angular_velocity_z",
    "simulator_orientation_quaternion_w",
    "simulator_orientation_quaternion_x",
    "simulator_orientation_quaternion_y",
    "simulator_orientation_quaternion_z",
)


def _finite(value: str | None) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values: list[float], fraction: float) -> float | None:
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


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "max": None, "median": None, "p95": None}
    return {
        "count": len(finite),
        "max": max(finite),
        "median": statistics.median(finite),
        "p95": _percentile(finite, 0.95),
    }


def _read_packets(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "simulator_packets.csv"
    rows: list[dict[str, float]] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = {}
            for field in ("simulation_time_s", "simulation_physics_step",
                          "simulation_render_frame", "telemetry_sequence",
                          *STATE_FIELDS):
                value = _finite(row.get(field))
                if value is None:
                    raise ValueError(f"{path}: missing finite {field}")
                values[field] = value
            rows.append(values)
    if len(rows) < 2:
        raise ValueError(f"{path}: fewer than two source packets")
    return rows


def _interpolate(rows: list[dict[str, float]], field: str, horizon: float) -> float:
    times = [row["simulation_time_s"] - rows[0]["simulation_time_s"]
             for row in rows]
    if horizon <= times[0]:
        return rows[0][field]
    if horizon >= times[-1]:
        return rows[-1][field]
    for before, after, before_t, after_t in zip(
            rows, rows[1:], times, times[1:]):
        if before_t <= horizon <= after_t:
            if after_t == before_t:
                return after[field]
            weight = (horizon - before_t) / (after_t - before_t)
            return before[field] * (1.0 - weight) + after[field] * weight
    return rows[-1][field]


def _timing(run_dir: Path, rows: list[dict[str, float]]) -> dict[str, object]:
    report = json.loads((run_dir / "timing_report.json").read_text())
    fault_count = 0
    fault_path = run_dir / "bridge_timing_fault.csv"
    if fault_path.exists():
        with fault_path.open(newline="") as stream:
            fault_count = sum(1 for _ in csv.DictReader(stream))
    source_dts = [b["simulation_time_s"] - a["simulation_time_s"]
                  for a, b in zip(rows, rows[1:])]
    source_steps = [int(b["simulation_physics_step"] -
                        a["simulation_physics_step"])
                    for a, b in zip(rows, rows[1:])]
    source_frames = [int(b["simulation_render_frame"] -
                         a["simulation_render_frame"])
                     for a, b in zip(rows, rows[1:])]
    source_sequences = [int(row["telemetry_sequence"]) for row in rows]
    return {
        "run_dir": str(run_dir),
        "packet_count": len(rows),
        "source_duration_s": rows[-1]["simulation_time_s"] - rows[0]["simulation_time_s"],
        "source_dt_s": _summary(source_dts),
        "source_dt_out_of_window_count": sum(
            not SOURCE_DT_MIN_S <= value <= SOURCE_DT_MAX_S
            for value in source_dts),
        "bridge_timing_fault_count": fault_count,
        "physics_step_delta": _summary(source_steps),
        "render_frame_delta": _summary(source_frames),
        "telemetry_sequence_reverse_or_duplicate_count": sum(
            b <= a for a, b in zip(source_sequences, source_sequences[1:])),
        "bridge_timing_report": report,
    }


def _collision_count(run_dir: Path) -> int | None:
    candidates = sorted(run_dir.glob("sensor_record_*.csv"))
    if not candidates:
        candidates = sorted(run_dir.glob("identification_grid_*.csv"))
    if not candidates:
        return None
    maximum = 0
    seen = False
    with candidates[-1].open(newline="") as stream:
        for row in csv.DictReader(stream):
            value = _finite(row.get("gt_collision_count"))
            if value is not None:
                maximum = max(maximum, int(value))
                seen = True
    return maximum if seen else None


def _plot(runs: list[list[dict[str, float]]], labels: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    groups = (
        ("simulator_position_x", "simulator_position_y", "position", "m"),
        ("simulator_linear_velocity_x", "simulator_linear_velocity_y",
         "linear velocity", "m/s"),
        ("simulator_angular_velocity_x", "simulator_angular_velocity_y",
         "angular velocity", "rad/s"),
    )
    for x_field, y_field, title, unit in groups:
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        for rows, label in zip(runs, labels):
            time_s = [row["simulation_time_s"] - rows[0]["simulation_time_s"]
                      for row in rows]
            axes[0].plot(time_s, [row[x_field] for row in rows], label=label)
            axes[1].plot(time_s, [row[y_field] for row in rows], label=label)
        axes[0].set_ylabel(f"x ({unit})")
        axes[1].set_ylabel(f"y ({unit})")
        axes[1].set_xlabel("relative source time (s)")
        figure.suptitle(f"Static repeatability: {title}")
        axes[0].grid(True)
        axes[1].grid(True)
        axes[0].legend(loc="best", fontsize="small")
        figure.tight_layout()
        figure.savefig(output_dir / f"repeatability_overlay_{title.replace(' ', '_')}.png",
                       dpi=140)
        plt.close(figure)


def build_report(run_dirs: list[Path], output_dir: Path) -> dict[str, object]:
    runs = [_read_packets(path) for path in run_dirs]
    timing = [_timing(path, rows) for path, rows in zip(run_dirs, runs)]
    common_horizon = min(
        rows[-1]["simulation_time_s"] - rows[0]["simulation_time_s"]
        for rows in runs)
    horizons = [value for value in REPORT_HORIZONS_S if value <= common_horizon]
    horizon_report: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for horizon in horizons:
        per_field: dict[str, dict[str, float | int | None]] = {}
        for field in STATE_FIELDS:
            samples = [_interpolate(rows, field, horizon) for rows in runs]
            differences = [abs(a - b) for a, b in itertools.combinations(samples, 2)]
            per_field[field] = _summary(differences)
        horizon_report[f"{horizon:.3f}"] = per_field

    return {
        "schema_version": 1,
        "source_contract": {
            "expected_rate_hz": 40.0,
            "source_dt_window_s": [SOURCE_DT_MIN_S, SOURCE_DT_MAX_S],
            "accepted_run_count": len(run_dirs),
            "all_runs_have_no_source_dt_violations": all(
                item["source_dt_out_of_window_count"] == 0 for item in timing),
            "all_runs_have_no_timing_fault": all(
                item["bridge_timing_fault_count"] == 0 and
                item["bridge_timing_report"].get(
                    "source_duplicate_or_reverse_count", 0) == 0
                for item in timing),
        },
        "alignment": {
            "method": "relative simulator source time with linear comparison interpolation",
            "raw_packets_unchanged": True,
            "common_horizon_s": common_horizon,
            "report_horizons_s": horizons,
        },
        "timing_and_frame": timing,
        "collision_count_by_run": {
            str(path): _collision_count(path) for path in run_dirs
        },
        "per_state_max_median_p95_difference_by_horizon": horizon_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", required=True,
                        help="accepted model-ID run directory; repeat this option")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run_dirs = [Path(value) for value in args.run_dir]
    if len(run_dirs) < 2:
        parser.error("at least two --run-dir values are required")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(run_dirs, output_dir)
    (output_dir / "repeatability_report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    _plot([_read_packets(path) for path in run_dirs],
          [path.name for path in run_dirs], output_dir)
    print(json.dumps(report["source_contract"], indent=2))
    print(json.dumps({
        "output_dir": str(output_dir),
        "common_horizon_s": report["alignment"]["common_horizon_s"],
        "plot_files": sorted(path.name for path in output_dir.glob("*.png")),
    }, indent=2))


if __name__ == "__main__":
    main()
