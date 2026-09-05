#!/usr/bin/env python3
"""Validate source-event cadence and physical continuity in a calibration CSV.

This audit is intentionally independent of the live timing-validator
subscriber.  The calibration recorder stores one exact row for each native
source callback, so this is the authoritative check for a CSV that will be
used for model fitting.  Timer snapshots and explicit simulator reset phases
are not used to infer source cadence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


STREAM_FIELDS = {
    "imu": ("ax_mps2", "ay_mps2", "az_mps2", "imu_yaw_rate_radps"),
    "left_encoder": ("left_encoder_rad", "left_encoder_stamp_s"),
    "right_encoder": ("right_encoder_rad", "right_encoder_stamp_s"),
    "odom": ("x_odom_m", "y_odom_m", "yaw_odom_rad", "odom_stamp_s"),
    "odom_diagnostics": ("odom_diagnostics_stamp_s",),
    "gt_odom": (
        "gt_odom_x_m", "gt_odom_y_m", "gt_odom_yaw_rad", "gt_odom_stamp_s",
    ),
}
ENCODER_FIELDS = {
    "left_encoder": "left_encoder_rad",
    "right_encoder": "right_encoder_rad",
}
RESET_BOUNDARY_STREAMS = {"odom", "odom_diagnostics"}
RESET_BOUNDARY_MAX_MS = 60.0
DERIVED_GAP_WARNING_FRACTION = 0.001


def finite(value: str | None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def audit(path: Path, max_encoder_rate_radps: float = 1500.0) -> dict[str, Any]:
    streams: dict[str, list[tuple[int, float, dict[str, str]]]] = {
        name: [] for name in STREAM_FIELDS
    }
    rows = 0
    failures: list[str] = []
    nonfinite: dict[str, int] = {name: 0 for name in STREAM_FIELDS}

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"source_event_name", "source_event_count", "source_event_stamp_s"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            return {
                "schema": "sdu_autodrive_calibration_audit_v1",
                "path": str(path),
                "passed": False,
                "failures": [f"missing required columns: {sorted(missing)}"],
            }
        for row in reader:
            rows += 1
            name = row.get("source_event_name", "")
            if name not in streams:
                continue
            count = finite(row.get("source_event_count"))
            stamp = finite(row.get("source_event_stamp_s"))
            if count is None or count < 0 or count != int(count) or stamp is None:
                failures.append(f"{name}: invalid source event identity at CSV row {rows}")
                continue
            streams[name].append((int(count), stamp, row))
            for field in STREAM_FIELDS[name]:
                if finite(row.get(field)) is None:
                    nonfinite[name] += 1

    report_streams: dict[str, Any] = {}
    warnings: list[str] = []
    for name, events in streams.items():
        events.sort(key=lambda item: (item[1], item[0]))
        stamps = [stamp for _, stamp, _ in events]
        deltas = [b - a for a, b in zip(stamps, stamps[1:])]
        positive = [delta for delta in deltas if delta > 0.0]
        duplicate_or_nonpositive = sum(delta <= 0.0 for delta in deltas)
        bursts = sum(delta < 0.015 for delta in positive)
        reset_boundary_gaps: list[dict[str, Any]] = []
        gaps = 0
        for index, delta in enumerate(deltas):
            if delta <= 0.040:
                continue
            next_row = events[index + 1][2]
            phase = next_row.get("phase", "")
            if (name in RESET_BOUNDARY_STREAMS and
                    phase.startswith("grid_reset_") and
                    delta * 1000.0 <= RESET_BOUNDARY_MAX_MS):
                reset_boundary_gaps.append({
                    "gap_ms": delta * 1000.0,
                    "phase": phase,
                })
            else:
                gaps += 1
        rate = (
            (len(stamps) - 1) / (stamps[-1] - stamps[0])
            if len(stamps) > 1 and stamps[-1] > stamps[0]
            else math.nan
        )
        report_streams[name] = {
            "events": len(events),
            "unique_event_counts": len({count for count, _, _ in events}),
            "rate_hz": rate,
            "p01_ms": percentile(positive, 0.01) * 1000.0,
            "median_ms": percentile(positive, 0.50) * 1000.0,
            "p95_ms": percentile(positive, 0.95) * 1000.0,
            "p99_ms": percentile(positive, 0.99) * 1000.0,
            "max_ms": max(positive, default=math.nan) * 1000.0,
            "bursts_lt_15ms": bursts,
            "gaps_gt_40ms": gaps,
            "allowed_reset_boundary_gaps": reset_boundary_gaps,
            "duplicate_or_nonpositive": duplicate_or_nonpositive,
            "nonfinite_required_values": nonfinite[name],
        }
        if not events:
            failures.append(f"{name}: no source-event rows")
            continue
        if abs(rate - 40.0) > 1.0:
            failures.append(f"{name}: source rate {rate:.6f} Hz is not 40 Hz")
        if duplicate_or_nonpositive:
            failures.append(f"{name}: duplicate/non-monotonic source timestamps")
        if bursts:
            failures.append(f"{name}: source burst interval below 15 ms")
        gap_fraction = gaps / len(positive) if positive else 1.0
        if gaps:
            if (name in RESET_BOUNDARY_STREAMS and
                    gap_fraction <= DERIVED_GAP_WARNING_FRACTION):
                warnings.append(
                    f"{name}: {gaps} non-reset source gap(s) above 40 ms "
                    f"({gap_fraction:.6%} of intervals)")
            else:
                failures.append(f"{name}: source gap interval above 40 ms")
        if nonfinite[name]:
            failures.append(f"{name}: non-finite required source values")

    encoder_outliers: list[dict[str, Any]] = []
    for name, field in ENCODER_FIELDS.items():
        events = streams[name]
        previous: tuple[float, float] | None = None
        for _, stamp, row in events:
            value = finite(row.get(field))
            if value is None:
                previous = None
                continue
            phase = row.get("phase", "")
            if previous is not None and not phase.startswith("grid_reset_"):
                previous_stamp, previous_value = previous
                dt = stamp - previous_stamp
                if dt > 0.0:
                    rate = abs(value - previous_value) / dt
                    if rate > max_encoder_rate_radps:
                        encoder_outliers.append({
                            "stream": name,
                            "source_event_stamp_s": stamp,
                            "rate_radps": rate,
                            "phase": phase,
                        })
            previous = (stamp, value)
    if encoder_outliers:
        failures.append("encoder source-event rate exceeds physical limit")

    return {
        "schema": "sdu_autodrive_calibration_audit_v1",
        "path": str(path),
        "rows": rows,
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "encoder_outliers": encoder_outliers,
        "streams": report_streams,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--max-encoder-rate-radps", type=float, default=1500.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.csv, args.max_encoder_rate_radps)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
