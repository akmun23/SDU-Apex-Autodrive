#!/usr/bin/env python3
"""Score an offline vehicle-model replay against simulator ground truth.

The replay and truth CSVs must contain source-time rows with the columns
emitted by ``f1tenth_mpc/vehicle_model_replay``:

    time_s,x_m,y_m,yaw_rad,u_mps,v_mps,r_radps,steering_rad

This tool is offline-only.  It never publishes ROS messages and never changes
the model or simulator.  It reports state errors at the nearest actual
timestamp for the requested physical horizons; it does not interpolate or
fabricate samples.  A 2% result is only a speed-specific diagnostic until a
frozen model is evaluated on untouched blind data.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Iterable


STATE_FIELDS = (
    "x_m", "y_m", "yaw_rad", "u_mps", "v_mps", "r_radps", "steering_rad",
)
HORIZONS_S = (0.05, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00)
TWO_PERCENT = 0.02


def _finite(value: str, field: str, path: Path, row_number: int) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{row_number}: invalid {field}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}:{row_number}: non-finite {field}")
    return parsed


def _read(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted({"time_s", *STATE_FIELDS}.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        rows: list[dict[str, float]] = []
        previous_time: float | None = None
        for row_number, raw in enumerate(reader, start=2):
            current = {
                field: _finite(raw.get(field, ""), field, path, row_number)
                for field in ("time_s", *STATE_FIELDS)
            }
            if previous_time is not None and current["time_s"] <= previous_time:
                raise ValueError(f"{path}:{row_number}: time is not strictly increasing")
            rows.append(current)
            previous_time = current["time_s"]
    if len(rows) < 2:
        raise ValueError(f"{path} contains fewer than two state rows")
    return rows


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _median_dt(rows: list[dict[str, float]]) -> float:
    intervals = [b["time_s"] - a["time_s"] for a, b in zip(rows, rows[1:])]
    intervals.sort()
    middle = len(intervals) // 2
    return intervals[middle] if len(intervals) % 2 else 0.5 * (
        intervals[middle - 1] + intervals[middle])


def _nearest_index(times: list[float], target: float) -> int:
    right = bisect.bisect_left(times, target)
    if right == 0:
        return 0
    if right == len(times):
        return len(times) - 1
    left = right - 1
    return left if target - times[left] <= times[right] - target else right


def _stats(errors: Iterable[float], actual: Iterable[float]) -> dict[str, object]:
    pairs = [(abs(error), value) for error, value in zip(errors, actual)]
    values = sorted(error for error, _ in pairs)
    relative = sorted(
        error / abs(value) for error, value in pairs if abs(value) >= 1.0)

    def percentile(values_: list[float], fraction: float) -> float | None:
        if not values_:
            return None
        index = min(len(values_) - 1, int(math.ceil(fraction * len(values_))) - 1)
        return values_[index]

    p95 = percentile(values, 0.95)
    relative_p95 = percentile(relative, 0.95)
    return {
        "count": len(values),
        "mae": sum(values) / len(values) if values else None,
        "p95": p95,
        "max": values[-1] if values else None,
        "relative_eligible_count": len(relative),
        "relative_p95": relative_p95,
        "two_percent_gate": relative_p95 is not None and relative_p95 <= TWO_PERCENT,
    }


def _errors(predicted: dict[str, float], truth: dict[str, float]) -> dict[str, float]:
    return {
        "position_m": math.hypot(
            predicted["x_m"] - truth["x_m"], predicted["y_m"] - truth["y_m"]),
        "heading_rad": abs(_wrap(predicted["yaw_rad"] - truth["yaw_rad"])),
        "u_mps": abs(predicted["u_mps"] - truth["u_mps"]),
        "v_mps": abs(predicted["v_mps"] - truth["v_mps"]),
        "yaw_rate_radps": abs(predicted["r_radps"] - truth["r_radps"]),
        "steering_rad": abs(predicted["steering_rad"] - truth["steering_rad"]),
    }


def _score_rows(
        predicted: list[dict[str, float]], truth: list[dict[str, float]],
        max_time_error_s: float) -> tuple[dict[str, object], list[tuple[float, int, int]]]:
    predicted_times = [row["time_s"] for row in predicted]
    truth_times = [row["time_s"] for row in truth]
    matches: list[tuple[float, int, int]] = []
    for predicted_index, row in enumerate(predicted):
        truth_index = _nearest_index(truth_times, row["time_s"])
        time_error = abs(truth_times[truth_index] - row["time_s"])
        if time_error <= max_time_error_s:
            matches.append((row["time_s"], predicted_index, truth_index))
    if len(matches) < 2:
        raise ValueError("fewer than two replay/truth timestamps could be aligned")

    error_values = {field: [] for field in (
        "position_m", "heading_rad", "u_mps", "v_mps",
        "yaw_rate_radps", "steering_rad")}
    actual_values = {field: [] for field in error_values}
    for _, predicted_index, truth_index in matches:
        errors = _errors(predicted[predicted_index], truth[truth_index])
        for field, value in errors.items():
            error_values[field].append(value)
            actual_values[field].append(truth[truth_index][
                {"position_m": "x_m", "heading_rad": "yaw_rad", "u_mps": "u_mps",
                 "v_mps": "v_mps", "yaw_rate_radps": "r_radps",
                 "steering_rad": "steering_rad"}[field]])
    return ({
        field: _stats(error_values[field], actual_values[field])
        for field in error_values
    }, matches)


def _horizon_scores(
        predicted: list[dict[str, float]], truth: list[dict[str, float]],
        matches: list[tuple[float, int, int]], max_time_error_s: float) -> dict[str, object]:
    predicted_times = [row["time_s"] for row in predicted]
    truth_times = [row["time_s"] for row in truth]
    output: dict[str, object] = {}
    for horizon in HORIZONS_S:
        errors = {field: [] for field in (
            "position_m", "heading_rad", "u_mps", "v_mps",
            "yaw_rate_radps", "steering_rad")}
        actual = {field: [] for field in errors}
        unavailable = 0
        for origin_time, _, _ in matches:
            target = origin_time + horizon
            predicted_index = _nearest_index(predicted_times, target)
            truth_index = _nearest_index(truth_times, target)
            if (abs(predicted_times[predicted_index] - target) > max_time_error_s or
                    abs(truth_times[truth_index] - target) > max_time_error_s):
                unavailable += 1
                continue
            row_errors = _errors(predicted[predicted_index], truth[truth_index])
            for field, value in row_errors.items():
                errors[field].append(value)
                actual[field].append(truth[truth_index][
                    {"position_m": "x_m", "heading_rad": "yaw_rad", "u_mps": "u_mps",
                     "v_mps": "v_mps", "yaw_rate_radps": "r_radps",
                     "steering_rad": "steering_rad"}[field]])
        output[f"{horizon:.2f}s"] = {
            "unavailable_rollout_count": unavailable,
            **{field: _stats(errors[field], actual[field]) for field in errors}
        }
    return output


def score(predicted_path: Path, truth_path: Path, max_time_error_s: float | None = None) -> dict[str, object]:
    predicted = _read(predicted_path)
    truth = _read(truth_path)
    tolerance = max_time_error_s
    if tolerance is None:
        tolerance = 0.5 * max(_median_dt(predicted), _median_dt(truth)) + 1.0e-9
    if not tolerance > 0.0 or not math.isfinite(tolerance):
        raise ValueError("max_time_error_s must be finite and positive")
    state_scores, matches = _score_rows(predicted, truth, tolerance)
    return {
        "schema_version": 1,
        "predicted_path": str(predicted_path),
        "truth_path": str(truth_path),
        "predicted_row_count": len(predicted),
        "truth_row_count": len(truth),
        "matched_row_count": len(matches),
        "timestamp_tolerance_s": tolerance,
        "state_scores": state_scores,
        "recursive_horizon_scores": _horizon_scores(
            predicted, truth, matches, tolerance),
        "acceptance_criterion": {
            "target": "blind recursive state/horizon error <= 2%",
            "two_percent_gate_scope": "relative speed only, actual speed >= 1 m/s",
            "blind_data_required": True,
            "repeatability_floor_required": True,
        },
        "acceptance_status": "scored_candidate_not_accepted",
    }


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predicted", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("--max-time-error-s", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    options = parser.parse_args(args)
    report = score(options.predicted, options.truth, options.max_time_error_s)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if options.output is None:
        print(serialized, end="")
    else:
        options.output.write_text(serialized, encoding="utf-8")
        print(options.output)


if __name__ == "__main__":
    main()
