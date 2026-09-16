#!/usr/bin/env python3
"""Explain which speed-command limiter caps a recorded run.

The analyzer is intentionally evidence-first.  It reports a limiter only
when the input contains the corresponding command/measurement fields.  A
run with only odometry and ground truth therefore produces an explicit
``unavailable`` result instead of guessing why a nominal 16 m/s request
peaked lower.  JSON-payload event logs and flat CSVs are both accepted.

This tool is offline diagnostics only.  It never retunes the speed controller
and never modifies simulator or actuator behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


LIMITERS = (
    "raceline", "curvature", "cte", "startup", "accel_slew", "max_speed",
    "actuator_feedforward", "overspeed_brake",
)


def _number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _text(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _truthy(row: dict[str, Any], *names: str) -> bool:
    value = _text(row, *names)
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on", "active", "startup",
                             "brake", "overspeed_brake"}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        payload = raw.get("payload_json", "")
        if payload:
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = {}
            row: dict[str, Any] = {key: value for key, value in raw.items()
                                   if key != "payload_json"}
            if isinstance(decoded, dict):
                row.update(decoded)
        else:
            row = dict(raw)
        rows.append(row)
    return rows


def _attribution(row: dict[str, Any], max_speed_mps: float) -> list[str]:
    target = _number(row, "pp_target_speed_mps", "target_speed_mps",
                     "speed_command_mps", "requested_speed_mps")
    raceline = _number(row, "raceline_speed_mps", "trajectory_speed_mps",
                       "planned_speed_mps", "vx_mps")
    curvature = _number(row, "curvature_radpm", "kappa_radpm",
                        "track_curvature_radpm", "pp_curvature_radpm")
    cte = _number(row, "cross_track_error_m", "cte_m", "e_cross_m")
    measured = _number(row, "speed_mps", "odom_speed_mps", "gt_speed_mps",
                       "truth_speed_mps")
    commanded_accel = _number(row, "speed_command_accel_mps2",
                              "requested_accel_mps2", "target_accel_mps2")
    applied_accel = _number(row, "applied_accel_mps2", "accel_mps2")
    throttle = _number(row, "throttle_normalized", "applied_throttle_norm",
                       "applied_throttle_norm_k1", "commanded_throttle_norm")
    phase = _text(row, "phase", "mode", "controller_mode")
    reasons: list[str] = []
    if target is not None and raceline is not None and target < raceline - 0.05:
        reasons.append("raceline")
    if (target is not None and raceline is not None and curvature is not None and
            target < raceline - 0.05 and abs(curvature) > 1.0e-6):
        reasons.append("curvature")
    if (target is not None and cte is not None and abs(cte) > 0.10 and
            measured is not None and target < measured):
        reasons.append("cte")
    if (_truthy(row, "startup_active", "startup", "startup_guard") or
            (phase is not None and phase.lower() in {"startup", "ramp", "ramp_up"})):
        reasons.append("startup")
    if (commanded_accel is not None and applied_accel is not None and
            applied_accel < commanded_accel - 0.10):
        reasons.append("accel_slew")
    if target is not None and target >= max_speed_mps - 1.0e-6:
        reasons.append("max_speed")
    if (target is not None and measured is not None and throttle is not None and
            measured < target - 0.25 and throttle < 0.98):
        reasons.append("actuator_feedforward")
    if (target is not None and measured is not None and
            measured > target + 0.50 and
            (_truthy(row, "overspeed_brake", "brake_active") or
             (throttle is not None and throttle <= 1.0e-6))):
        reasons.append("overspeed_brake")
    return reasons


def _summarize(rows: Sequence[dict[str, Any]], max_speed_mps: float) -> dict[str, Any]:
    counts = {limiter: 0 for limiter in LIMITERS}
    rows_with_evidence = 0
    for row in rows:
        reasons = _attribution(row, max_speed_mps)
        if reasons:
            rows_with_evidence += 1
        for reason in reasons:
            counts[reason] += 1
    target_values = [value for row in rows
                     if (value := _number(
                         row, "pp_target_speed_mps", "target_speed_mps",
                         "speed_command_mps", "requested_speed_mps")) is not None]
    measured_values = [value for row in rows
                       if (value := _number(
                           row, "speed_mps", "odom_speed_mps", "gt_speed_mps",
                           "truth_speed_mps")) is not None]
    available = {
        field: any(_number(row, *aliases) is not None for row in rows)
        for field, aliases in {
            "target_speed": ("pp_target_speed_mps", "target_speed_mps",
                              "speed_command_mps", "requested_speed_mps"),
            "raceline_speed": ("raceline_speed_mps", "trajectory_speed_mps",
                                "planned_speed_mps", "vx_mps"),
            "curvature": ("curvature_radpm", "kappa_radpm",
                           "track_curvature_radpm", "pp_curvature_radpm"),
            "cte": ("cross_track_error_m", "cte_m", "e_cross_m"),
            "measured_speed": ("speed_mps", "odom_speed_mps", "gt_speed_mps",
                                "truth_speed_mps"),
            "commanded_acceleration": ("speed_command_accel_mps2",
                                        "requested_accel_mps2", "target_accel_mps2"),
            "applied_acceleration": ("applied_accel_mps2", "accel_mps2"),
            "throttle": ("throttle_normalized", "applied_throttle_norm",
                          "applied_throttle_norm_k1", "commanded_throttle_norm"),
        }.items()
    }
    missing = [field for field, present in available.items() if not present]
    return {
        "row_count": len(rows),
        "rows_with_limiter_evidence": rows_with_evidence,
        "available_fields": available,
        "missing_fields_for_attribution": missing,
        "limiter_evidence_row_counts": counts,
        "target_speed_mps": {
            "max": max(target_values) if target_values else None,
            "samples": len(target_values),
        },
        "measured_speed_mps": {
            "max": max(measured_values) if measured_values else None,
            "samples": len(measured_values),
        },
        "attribution_status": (
            "available" if rows_with_evidence else "unavailable_from_input_columns"),
    }


def analyze(input_csvs: Sequence[Path], output: Path,
            max_speed_mps: float = 16.0) -> dict[str, Any]:
    reports = []
    all_rows: list[dict[str, Any]] = []
    for path in input_csvs:
        rows = _read_rows(path)
        all_rows.extend(rows)
        reports.append({"input_csv": str(path), **_summarize(rows, max_speed_mps)})
    result = {
        "schema_version": 1,
        "status": "offline_speed_command_pipeline_analysis",
        "simulator_modified": False,
        "production_speed_controller_modified": False,
        "max_speed_ceiling_mps": max_speed_mps,
        "inputs": [str(path) for path in input_csvs],
        "per_input": reports,
        "combined": _summarize(all_rows, max_speed_mps),
        "interpretation": {
            "limiter_evidence_is_not_available_without_command_and_state_alignment": True,
            "do_not_retune_speed_controller_from_ground_truth_only": True,
            "required_for_decision": [
                "target speed timestamped against measured speed",
                "raceline/curvature speed before each limiter",
                "CTE/startup/acceleration-slew diagnostics",
                "commanded and applied throttle",
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-speed", type=float, default=16.0)
    args = parser.parse_args()
    result = analyze(args.input_csv, args.output, args.max_speed)
    print(json.dumps({
        "output": str(args.output),
        "combined": result["combined"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
