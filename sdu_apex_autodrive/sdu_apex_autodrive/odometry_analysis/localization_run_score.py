"""Score a complete localization run against simulator truth.

This is an offline acceptance tool.  Ground truth is read from a recording and
is never used by a runtime node.  The primary percentage is a path-normalized
error (metres of position error divided by metres travelled), which remains
meaningful when the track origin is arbitrary.  Absolute and relative error
columns produced by the monitor are both supported; absolute map-frame columns
are preferred when map provenance was recorded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


DEFAULT_ESTIMATORS = ("amcl", "current_map", "ekf", "odom")


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _time(row: dict[str, str], fallback: float) -> float:
    return _number(row, "stamp_s") or _number(row, "time_s") or fallback


def _elapsed_time(row: dict[str, str], fallback: float) -> float:
    """Prefer a recorder's process-relative time for cross-file matching."""
    return _number(row, "time_s") or _number(row, "stamp_s") or fallback


def _unique_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Sort by source time and remove duplicate monitor snapshots."""
    indexed = sorted(enumerate(rows), key=lambda item: (_time(item[1], item[0]), item[0]))
    result: list[dict[str, str]] = []
    previous_time: float | None = None
    for _, row in indexed:
        stamp = _time(row, float(len(result)))
        if previous_time is not None and stamp <= previous_time:
            # A monitor can write multiple callbacks at one timestamp.  Keep
            # the first pose so a duplicate never creates artificial motion.
            continue
        result.append(row)
        previous_time = stamp
    return result


def _path_length(rows: list[dict[str, str]]) -> float:
    distance = 0.0
    previous: tuple[float, float] | None = None
    for row in rows:
        x = _number(row, "gt_x_m")
        y = _number(row, "gt_y_m")
        if x is None or y is None:
            continue
        if previous is not None:
            distance += math.hypot(x - previous[0], y - previous[1])
        previous = (x, y)
    return distance


def _error_m(row: dict[str, str], estimator: str) -> float | None:
    # A provenance-backed run is the strongest result and must win over the
    # legacy relative column.  Older monitor CSVs only have *_error_m.
    for key in (
        f"{estimator}_absolute_error_m",
        f"{estimator}_error_m",
        f"{estimator}_relative_drift_m",
    ):
        value = _number(row, key)
        if value is not None:
            return abs(value)
    x = _number(row, f"{estimator}_x_m")
    y = _number(row, f"{estimator}_y_m")
    gx = _number(row, "gt_x_m")
    gy = _number(row, "gt_y_m")
    if None not in (x, y, gx, gy):
        return math.hypot(x - gx, y - gy)
    return None


def _yaw_error_rad(row: dict[str, str], estimator: str) -> float | None:
    for key in (
        f"{estimator}_absolute_error_rad",
        f"{estimator}_error_rad",
        f"{estimator}_relative_drift_rad",
    ):
        value = _number(row, key)
        if value is not None:
            return abs(value)
    yaw = _number(row, f"{estimator}_yaw_rad")
    gt_yaw = _number(row, "gt_yaw_rad")
    if yaw is not None and gt_yaw is not None:
        return abs(math.atan2(math.sin(yaw - gt_yaw), math.cos(yaw - gt_yaw)))
    return None


def _first_collision_index(rows: list[dict[str, str]]) -> int | None:
    for index, row in enumerate(rows):
        count = _number(row, "collision_count")
        if count is None:
            count = _number(row, "gt_collision_count")
        if count is not None and count > 0.0:
            return index
    return None


def _first_telemetry_collision_time(telemetry_csv: str | Path | None) -> float | None:
    """Return the first simulator collision time from a recorder CSV.

    The monitor intentionally discards the post-collision localization epoch.
    DDS callback ordering can therefore leave its final row with the old
    collision count even though the recorder captured the simulator counter.
    Use the simulator's cumulative ground-truth counter as an independent
    acceptance signal when the raw telemetry is supplied.
    """
    if telemetry_csv is None:
        return None
    with Path(telemetry_csv).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                count = float(row.get("gt_collision_count", ""))
                # Monitor and recorder files both contain ``stamp_s`` and
                # ``time_s``.  The monitor's ``time_s`` is relative to its
                # process start, so compare the recorder's relative time to
                # the monitor rows rather than mixing absolute ROS stamps.
                stamp = _number(row, "time_s") or _number(row, "stamp_s")
            except (TypeError, ValueError):
                continue
            if math.isfinite(count) and count > 0.0 and math.isfinite(stamp):
                return stamp
    return None


def _estimator_summary(
    rows: list[dict[str, str]],
    estimator: str,
    *,
    path_length_m: float,
    threshold_percent: float,
    jump_threshold_m: float,
) -> dict[str, object]:
    errors = [value for row in rows if (value := _error_m(row, estimator)) is not None]
    yaw_errors = [
        value for row in rows if (value := _yaw_error_rad(row, estimator)) is not None
    ]
    jumps = 0
    previous: float | None = None
    for value in errors:
        if previous is not None and abs(value - previous) > jump_threshold_m:
            jumps += 1
        previous = value
    denominator = max(path_length_m, 1.0e-12)
    p95 = _percentile(errors, 0.95)
    maximum = max(errors, default=math.nan)
    summary: dict[str, object] = {
        "samples": len(errors),
        "yaw_samples": len(yaw_errors),
        "error_p50_m": _percentile(errors, 0.50),
        "error_p95_m": p95,
        "error_p99_m": _percentile(errors, 0.99),
        "error_max_m": maximum,
        "error_mean_m": (sum(errors) / len(errors)) if errors else math.nan,
        "yaw_error_p95_rad": _percentile(yaw_errors, 0.95),
        "error_p95_percent_of_path": 100.0 * p95 / denominator,
        "error_max_percent_of_path": 100.0 * maximum / denominator,
        "jump_count": jumps,
        "within_threshold": bool(
            errors
            and math.isfinite(maximum)
            and 100.0 * maximum / denominator <= threshold_percent
            and jumps == 0
        ),
    }
    return summary


def score_csv(
    input_csv: str | Path,
    *,
    telemetry_csv: str | Path | None = None,
    threshold_percent: float = 2.0,
    min_path_length_m: float = 35.0,
    jump_threshold_m: float = 0.25,
    required_estimators: tuple[str, ...] = DEFAULT_ESTIMATORS,
) -> dict[str, object]:
    """Score one full-run monitor CSV.

    A run passes only when it travelled at least ``min_path_length_m``, had no
    collision, every required estimator stayed below the percentage threshold
    at every scored sample, and no estimator had a discontinuity larger than
    ``jump_threshold_m`` between consecutive monitor samples.
    """
    with Path(input_csv).open(newline="", encoding="utf-8") as stream:
        rows = _unique_rows(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("recording has fewer than two timestamped samples")
    collision_index = _first_collision_index(rows)
    telemetry_collision_time = _first_telemetry_collision_time(telemetry_csv)
    telemetry_collision_index = None
    if telemetry_collision_time is not None:
        for index, row in enumerate(rows):
            if _elapsed_time(row, float(index)) >= telemetry_collision_time:
                telemetry_collision_index = index
                break
        if telemetry_collision_index is None and rows:
            telemetry_collision_index = len(rows) - 1
        if collision_index is None or (
            telemetry_collision_index is not None
            and telemetry_collision_index < collision_index
        ):
            collision_index = telemetry_collision_index
    clean_rows = rows if collision_index is None else rows[:collision_index]
    path_length = _path_length(clean_rows)
    scoring_mode = next(
        (row.get("scoring_mode") for row in rows if row.get("scoring_mode")),
        "relative_first_pair",
    )
    summaries = {
        estimator: _estimator_summary(
            clean_rows,
            estimator,
            path_length_m=path_length,
            threshold_percent=threshold_percent,
            jump_threshold_m=jump_threshold_m,
        )
        for estimator in required_estimators
    }
    missing = [
        estimator for estimator, summary in summaries.items()
        if int(summary["samples"]) == 0
    ]
    full_path = path_length >= min_path_length_m
    collision_free = collision_index is None
    overall_pass = bool(
        full_path and collision_free and not missing
        and all(bool(summary["within_threshold"]) for summary in summaries.values())
    )
    return {
        "input_csv": str(Path(input_csv).resolve()),
        "scoring_mode": scoring_mode,
        "sample_count": len(rows),
        "clean_sample_count": len(clean_rows),
        "path_length_m": path_length,
        "min_path_length_m": min_path_length_m,
        "threshold_percent": threshold_percent,
        "jump_threshold_m": jump_threshold_m,
        "collision_free": collision_free,
        "collision_row_index": collision_index,
        "telemetry_collision_time_s": telemetry_collision_time,
        "full_path": full_path,
        "missing_estimators": missing,
        "estimators": summaries,
        "overall_pass": overall_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument(
        "--telemetry-csv",
        type=Path,
        help="Raw recorder CSV used to detect a simulator collision missed by the monitor callback",
    )
    parser.add_argument("--json", dest="output_json", type=Path)
    parser.add_argument("--threshold-percent", type=float, default=2.0)
    parser.add_argument("--min-path-length-m", type=float, default=35.0)
    parser.add_argument("--jump-threshold-m", type=float, default=0.25)
    args = parser.parse_args()
    result = score_csv(
        args.input_csv,
        telemetry_csv=args.telemetry_csv,
        threshold_percent=args.threshold_percent,
        min_path_length_m=args.min_path_length_m,
        jump_threshold_m=args.jump_threshold_m,
    )
    encoded = json.dumps(result, indent=2, allow_nan=True)
    print(encoded)
    if args.output_json is not None:
        args.output_json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
