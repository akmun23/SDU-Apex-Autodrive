#!/usr/bin/env python3
"""Score the state stream available to MPC at its application epoch.

The scorer accepts a flat localization/runtime CSV such as
``full_speed_localization_report.csv``.  It uses only estimate and truth
columns already present in that file; simulator truth is never fed back into
the estimator or controller.  If a file contains no explicit future-horizon
column, it reports the available current-state score at ``0.0 s`` and marks
future-horizon metrics unavailable instead of inventing them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_raceline_operating_envelope import DEFAULT_RACELINE, _read_raceline, _stats  # noqa: E402
from score_raceline_model import _raceline_arrays, frenet_error  # noqa: E402


REGIMES = (("low_2_8", 2.0, 8.0), ("mid_8_12", 8.0, 12.0),
           ("high_12_16", 12.0, 16.0))


def _float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _regime(speed: float | None) -> str:
    if speed is None:
        return "unknown"
    for name, low, high in REGIMES:
        if low <= speed < high:
            return name
    return "outside_2_16"


def _metric(values: Iterable[float]) -> dict[str, Any]:
    return _stats(values)


def _score_row(row: dict[str, str], raceline: dict[str, Any]) -> dict[str, Any] | None:
    estimate_x = _float(row, "estimate_x_m", "current_map_x_m", "odom_x_m")
    estimate_y = _float(row, "estimate_y_m", "current_map_y_m", "odom_y_m")
    estimate_yaw = _float(row, "estimate_yaw_rad", "current_map_yaw_rad",
                          "odom_yaw_rad")
    truth_x = _float(row, "truth_x_m", "gt_x_m")
    truth_y = _float(row, "truth_y_m", "gt_y_m")
    truth_yaw = _float(row, "truth_yaw_rad", "gt_yaw_rad")
    if None in (estimate_x, estimate_y, estimate_yaw, truth_x, truth_y, truth_yaw):
        return None
    dx = estimate_x - truth_x
    dy = estimate_y - truth_y
    c, s = math.cos(truth_yaw), math.sin(truth_yaw)
    relative_dx = c * dx + s * dy
    relative_dy = -s * dx + c * dy
    frenet = frenet_error(
        estimate_x, estimate_y, estimate_yaw, truth_x, truth_y, truth_yaw,
        raceline)
    speed = _float(row, "truth_pose_u_mps", "gt_speed_mps", "speed_mps",
                   "truth_u_mps")
    u_error = _float(row, "u_error_mps")
    v_error = _float(row, "v_error_mps", "v_error_mps_com")
    return {
        "topic": row.get("topic", "unknown"),
        "horizon_s": (_float(row, "horizon_s", "prediction_horizon_s")
                      if _float(row, "horizon_s", "prediction_horizon_s") is not None
                      else 0.0),
        "regime": _regime(speed),
        "relative_dx_m": relative_dx,
        "relative_dy_m": relative_dy,
        "relative_heading_rad": _wrap(estimate_yaw - truth_yaw),
        "current_map_cross_track_m": float(frenet["e_cross_m"]),
        "current_map_heading_rad": float(frenet["e_heading_rad"]),
        "position_m": float(frenet["position_m"]),
        "u_error_mps": u_error,
        "v_error_mps": v_error,
        "r_error_radps": _float(row, "r_error_radps", "yaw_rate_error_radps"),
        "steering_error_rad": _float(row, "steering_error_rad", "delta_error_rad"),
        "state_age_s": _float(row, "state_age_s", "estimate_age_s"),
    }


METRICS = (
    "relative_dx_m", "relative_dy_m", "relative_heading_rad",
    "current_map_cross_track_m", "current_map_heading_rad", "position_m",
    "u_error_mps", "v_error_mps", "r_error_radps", "steering_error_rad",
    "state_age_s",
)


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {
        group: {
            "sample_count": len(values),
            "metrics": {
                metric: _metric([row[metric] for row in values
                                 if row.get(metric) is not None])
                for metric in METRICS
            },
        }
        for group, values in sorted(groups.items())
    }


def score(input_csv: Path, output: Path,
          raceline_csv: Path = DEFAULT_RACELINE) -> dict[str, Any]:
    raceline = _raceline_arrays(_read_raceline(raceline_csv))
    with input_csv.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    scored = [result for row in source_rows
              if (result := _score_row(row, raceline)) is not None]
    horizons = sorted({float(row["horizon_s"]) for row in scored})
    result = {
        "schema_version": 1,
        "status": "offline_runtime_control_state_score",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "input_csv": str(input_csv),
        "raceline_csv": str(raceline_csv),
        "state_contract": {
            "global_pose_reference_point": "estimate pose as published by the named topic",
            "body_velocity_reference_point": "explicitly reported source field; COM v is kept distinct from pose-point v",
            "u": "longitudinal body velocity error against truth when available",
            "v_COM": "COM lateral velocity error when v_error_mps_com is available",
            "r": "yaw-rate error when an estimate/reference pair is available",
            "steering_feedback": "steering_error_rad when an estimate/reference pair is available",
            "wheel_state": "not required by this pose-state scorer",
            "source_timestamp": "source_stamp_ns or input timestamp field",
            "state_age": "state_age_s when provided; otherwise unavailable",
        },
        "sample_count": len(scored),
        "horizons_s": horizons,
        "future_horizon_data_available": any(horizon > 0.0 for horizon in horizons),
        "overall": {
            metric: _metric([row[metric] for row in scored
                             if row.get(metric) is not None])
            for metric in METRICS
        },
        "per_topic": _group(scored, "topic"),
        "per_regime": _group(scored, "regime"),
        "per_horizon": _group(scored, "horizon_s"),
        "promotion_target": {
            "u_error_p95_core_mps": 0.10,
            "condition": "do not promote on this report if required state fields are unavailable",
            "runtime_migration": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--raceline", type=Path, default=DEFAULT_RACELINE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.input_csv, args.output, args.raceline)
    print(json.dumps({
        "output": str(args.output),
        "sample_count": result["sample_count"],
        "current_map_cross_track_p95_m": result["overall"][
            "current_map_cross_track_m"]["p95"],
        "future_horizon_data_available": result[
            "future_horizon_data_available"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
