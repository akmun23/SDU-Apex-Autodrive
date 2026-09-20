#!/usr/bin/env python3
"""Measure live-MPC horizon yaw residuals by speed and path curvature.

This is an offline diagnostic.  Simulator truth is read only from the
recorded ground-truth file after a live MPC-authority run; it is never an MPC
runtime input.  The purpose is to distinguish a bad objective weight from a
model that predicts too much or too little yaw in the part of the horizon
that causes a collision.

The report compares each accepted diagnostic's predicted yaw-rate state with
the average yaw rate actually measured over the corresponding future window.
Signed residuals are retained: a negative value means the prediction turns
more in the negative direction than the object did.  Results are grouped by
the current speed and the current absolute reference curvature so a single
global RMS cannot hide the high-curvature failure mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median


HORIZONS = (1, 5, 10, 20, 30)
ACCEPTED = {"accepted_optimal", "accepted_degraded"}


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def interpolate(rows: list[dict[str, float]], stamp: float) -> dict[str, float] | None:
    if not rows or stamp < rows[0]["stamp"] or stamp > rows[-1]["stamp"]:
        return None
    lo, hi = 0, len(rows) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if rows[mid]["stamp"] <= stamp:
            lo = mid
        else:
            hi = mid
    a, b = rows[lo], rows[hi]
    if b["stamp"] <= a["stamp"]:
        return a
    alpha = (stamp - a["stamp"]) / (b["stamp"] - a["stamp"])
    return {
        "stamp": stamp,
        "yaw": wrap(a["yaw"] + alpha * wrap(b["yaw"] - a["yaw"])),
        "speed": a["speed"] + alpha * (b["speed"] - a["speed"]),
    }


def load_truth(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stamp = finite(row.get("stamp_s"))
            yaw = finite(row.get("gt_yaw_rad"))
            speed = finite(row.get("gt_speed_mps"))
            if stamp is None or yaw is None or speed is None:
                continue
            # The operational command envelope is 16 m/s.  Exclude corrupt
            # duplicate/reset rows rather than allowing them to dominate a
            # finite-difference residual.
            if speed < 0.0 or speed > 16.0:
                continue
            if rows and stamp < rows[-1]["stamp"]:
                continue
            rows.append({"stamp": stamp, "yaw": yaw, "speed": speed})
    return rows


def diagnostics(path: Path) -> list[tuple[float, dict]]:
    result: list[tuple[float, dict]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("topic") != "/mpc/diagnostics":
                continue
            try:
                payload = json.loads(row["payload_json"])
                diagnostic = json.loads(payload["value"])
                arrival = float(row["arrival_epoch_ns"]) / 1.0e9
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            result.append((arrival, diagnostic))
    return result


def locate_trace(run: Path) -> Path:
    candidates = (
        run / run.name / "controller_trace.csv",
        run / "model_id" / run.name / "controller_trace.csv",
        run / "controller_trace.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("controller_trace.csv not found under " + str(run))


def bin_speed(value: float) -> str:
    if value < 2.0:
        return "<2"
    if value < 4.0:
        return "2-4"
    if value < 6.0:
        return "4-6"
    if value < 8.0:
        return "6-8"
    return ">=8"


def bin_curvature(value: float) -> str:
    if value < 0.2:
        return "<0.2"
    if value < 0.4:
        return "0.2-0.4"
    return ">=0.4"


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95_abs": None,
                "max_abs": None}
    absolute = sorted(abs(value) for value in values)
    return {
        "n": len(values),
        "mean": fmean(values),
        "median": median(values),
        "p95_abs": absolute[max(0, math.ceil(0.95 * len(absolute)) - 1)],
        "max_abs": max(absolute),
    }


def score(run: Path, truth_path: Path, dt: float) -> dict:
    truth = load_truth(truth_path)
    rows: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    all_rows: dict[int, list[float]] = defaultdict(list)
    optimistic: dict[tuple[int, str, str], int] = defaultdict(int)
    counts: dict[tuple[int, str, str], int] = defaultdict(int)
    skipped = 0
    for _, diagnostic in diagnostics(locate_trace(run)):
        if diagnostic.get("status") not in ACCEPTED:
            continue
        source_ns = finite(diagnostic.get("source_stamp_ns"))
        state = diagnostic.get("state")
        curvature = finite(diagnostic.get("path_curvature_per_m"))
        if source_ns is None or not isinstance(state, list) or len(state) < 5 or curvature is None:
            skipped += 1
            continue
        source = interpolate(truth, source_ns / 1.0e9)
        if source is None:
            skipped += 1
            continue
        speed = finite(state[2])
        predicted_r = finite(state[4])
        if speed is None or predicted_r is None:
            skipped += 1
            continue
        speed_key = bin_speed(max(0.0, speed))
        curvature_key = bin_curvature(abs(curvature))
        for horizon in HORIZONS:
            prediction = next(
                (item for item in diagnostic.get("predictions", [])
                 if int(item.get("n", -1)) == horizon), None)
            if not prediction:
                continue
            predicted_state = prediction.get("state")
            if not isinstance(predicted_state, list) or len(predicted_state) < 5:
                continue
            predicted = finite(predicted_state[4])
            future = interpolate(truth, source["stamp"] + horizon * dt)
            if predicted is None or future is None:
                continue
            actual_average = wrap(future["yaw"] - source["yaw"]) / (horizon * dt)
            residual = predicted - actual_average
            key = (horizon, speed_key, curvature_key)
            rows[key].append(residual)
            all_rows[horizon].append(residual)
            counts[key] += 1
            if abs(predicted) > abs(actual_average) + 0.05:
                optimistic[key] += 1
    report: dict[str, object] = {
        "run": str(run),
        "truth": str(truth_path),
        "dt_s": dt,
        "accepted_diagnostics": sum(len(values) for values in all_rows.values()) // len(HORIZONS)
        if all_rows else 0,
        "skipped_diagnostics": skipped,
        "overall_by_horizon": {
            str(horizon): summary(all_rows[horizon]) for horizon in HORIZONS
        },
        "by_speed_and_curvature": {},
    }
    grouped: dict[str, object] = {}
    for key in sorted(rows):
        horizon, speed_key, curvature_key = key
        result = summary(rows[key])
        result["optimistic_fraction"] = optimistic[key] / counts[key]
        grouped[f"N{horizon}|speed={speed_key}|abs_kappa={curvature_key}"] = result
    report["by_speed_and_curvature"] = grouped
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = score(args.run, args.truth, args.dt)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
