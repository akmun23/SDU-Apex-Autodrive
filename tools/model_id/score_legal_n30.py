#!/usr/bin/env python3
"""Validate and score a legal-state 40 Hz / N30 prediction table.

This is deliberately a scorer, not a vehicle model.  A candidate generator
provides its recursive predictions, while this tool enforces the evaluation
contract: every origin is sensor-legal, source cadence is 40 Hz, and scoring
uses only the four useful horizons.  Simulator truth is accepted exclusively
as the offline endpoint target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SOURCE_DT_MIN_S = 0.015
SOURCE_DT_MAX_S = 0.035
HORIZON_STEPS = (4, 10, 20, 30)
HORIZON_SECONDS = {steps: steps * 0.025 for steps in HORIZON_STEPS}
REQUIRED_FIELDS = {
    "origin_id",
    "origin_input_mode",
    "source_dt_s",
    "horizon_steps",
    "horizon_s",
    "pred_x_m",
    "pred_y_m",
    "pred_yaw_rad",
    "truth_x_m",
    "truth_y_m",
    "truth_yaw_rad",
}


def _number(row: dict[str, str], field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: missing or invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: non-finite {field}")
    return value


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
    }


def _reject_nonlegal_columns(fields: set[str]) -> None:
    forbidden = sorted(
        field for field in fields
        if field.startswith("origin_simulator_") or
        field.startswith("origin_truth_") or
        field.startswith("future_")
    )
    if forbidden:
        raise ValueError(
            "prediction table contains prohibited origin/future fields: " +
            ", ".join(forbidden))


def score(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_FIELDS.difference(fields))
        if missing:
            raise ValueError(f"{path} missing required fields: {missing}")
        _reject_nonlegal_columns(fields)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no prediction rows")

    seen: set[tuple[str, int]] = set()
    by_horizon: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"position_m": [], "yaw_rad": []})
    for row_number, row in enumerate(rows, start=2):
        if row.get("origin_input_mode", "").strip() != "sensor_legal":
            raise ValueError(
                f"row {row_number}: origin_input_mode must be sensor_legal")
        source_dt = _number(row, "source_dt_s", row_number)
        if not SOURCE_DT_MIN_S <= source_dt <= SOURCE_DT_MAX_S:
            raise ValueError(
                f"row {row_number}: source_dt_s={source_dt:g} is outside "
                f"[{SOURCE_DT_MIN_S:g}, {SOURCE_DT_MAX_S:g}]")
        steps_value = _number(row, "horizon_steps", row_number)
        steps = int(round(steps_value))
        if abs(steps_value - steps) > 1.0e-9 or steps not in HORIZON_STEPS:
            raise ValueError(
                f"row {row_number}: horizon_steps must be one of {HORIZON_STEPS}")
        horizon = _number(row, "horizon_s", row_number)
        expected_horizon = HORIZON_SECONDS[steps]
        if abs(horizon - expected_horizon) > 0.010:
            raise ValueError(
                f"row {row_number}: horizon_s={horizon:g} does not match "
                f"N{steps} ({expected_horizon:g} s)")
        origin_id = row.get("origin_id", "").strip()
        if not origin_id:
            raise ValueError(f"row {row_number}: empty origin_id")
        key = (origin_id, steps)
        if key in seen:
            raise ValueError(
                f"row {row_number}: duplicate origin_id/horizon_steps {key}")
        seen.add(key)
        dx = _number(row, "pred_x_m", row_number) - _number(
            row, "truth_x_m", row_number)
        dy = _number(row, "pred_y_m", row_number) - _number(
            row, "truth_y_m", row_number)
        dyaw = _wrap(_number(row, "pred_yaw_rad", row_number) - _number(
            row, "truth_yaw_rad", row_number))
        by_horizon[steps]["position_m"].append(math.hypot(dx, dy))
        by_horizon[steps]["yaw_rad"].append(abs(dyaw))

    missing_horizons = [steps for steps in HORIZON_STEPS if steps not in by_horizon]
    if missing_horizons:
        raise ValueError(f"prediction table is missing required horizons: {missing_horizons}")
    return {
        "schema_version": 1,
        "status": "score_only_not_a_model_promotion",
        "origin_contract": "sensor_legal",
        "source_cadence_window_s": [SOURCE_DT_MIN_S, SOURCE_DT_MAX_S],
        "horizon_contract": {
            "sample_rate_hz": 40.0,
            "steps": list(HORIZON_STEPS),
            "seconds": [HORIZON_SECONDS[steps] for steps in HORIZON_STEPS],
        },
        "ground_truth_use": "offline_endpoint_scoring_only",
        "rows": len(rows),
        "horizons": {
            f"{HORIZON_SECONDS[steps]:.2f}s": {
                "steps": steps,
                "position_m": _stats(by_horizon[steps]["position_m"]),
                "yaw_rad": _stats(by_horizon[steps]["yaw_rad"]),
            }
            for steps in HORIZON_STEPS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = score(args.predictions)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
