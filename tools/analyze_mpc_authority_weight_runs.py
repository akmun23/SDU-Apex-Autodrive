#!/usr/bin/env python3
"""Compare live MPC predictions with offline simulator truth.

This tool intentionally consumes only recorded run artifacts.  It never feeds
simulator truth to MPC and never uses shadow/replay commands as validation.
For each live MPC diagnostic it reconstructs the predicted map-frame pose from
the raceline sample and predicted (e_y, e_psi), then compares it with the
timestamped simulator pose at N*dt in the ground-truth diagnostic file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def read_path(path: Path) -> tuple[list[dict[str, float]], float]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        points = []
        for row in rows:
            points.append({
                "s": float(row["# s_m"]),
                "x": float(row["x_m"]),
                "y": float(row["y_m"]),
                "psi": float(row["psi_rad"]),
                "kappa": float(row["kappa_radpm"]),
            })
    if len(points) < 3:
        raise ValueError("raceline must contain at least three points")
    length = points[-1]["s"]
    ds = points[1]["s"] - points[0]["s"]
    return points, length + ds


def interpolate_truth(rows: list[dict[str, float]], stamp: float) -> dict[str, float] | None:
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
    yaw_delta = wrap(b["yaw"] - a["yaw"])
    return {
        "stamp": stamp,
        "x": a["x"] + alpha * (b["x"] - a["x"]),
        "y": a["y"] + alpha * (b["y"] - a["y"]),
        "yaw": wrap(a["yaw"] + alpha * yaw_delta),
        "speed": a["speed"] + alpha * (b["speed"] - a["speed"]),
    }


def nearest_path(points: list[dict[str, float]], length: float, x: float, y: float) -> dict[str, float]:
    best = None
    for point in points:
        distance_sq = (x - point["x"]) ** 2 + (y - point["y"]) ** 2
        if best is None or distance_sq < best[0]:
            best = (distance_sq, point)
    assert best is not None
    return best[1]


def path_at(points: list[dict[str, float]], length: float, s: float) -> dict[str, float]:
    target = s % length
    best = min(range(len(points)), key=lambda i: abs(points[i]["s"] - target))
    return points[best]


def load_truth(path: Path) -> list[dict[str, float]]:
    output = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                x = float(row["gt_x_m"])
                y = float(row["gt_y_m"])
                yaw = float(row["gt_yaw_rad"])
                speed = float(row["gt_speed_mps"])
                stamp = float(row["stamp_s"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x, y, yaw, speed, stamp)):
                continue
            # Collision/reset rows can contain a teleport and an invalid speed.
            # Keep only physically continuous samples for prediction scoring.
            if speed < 0.0 or speed > 16.0:
                continue
            if output:
                previous = output[-1]
                dt = stamp - previous["stamp"]
                if dt > 0.0 and math.hypot(x - previous["x"], y - previous["y"]) > max(0.6, 16.0 * dt):
                    break
            output.append({"stamp": stamp, "x": x, "y": y, "yaw": yaw, "speed": speed})
    return output


def load_diagnostics(path: Path) -> list[tuple[float, dict]]:
    output = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("topic") != "/mpc/diagnostics":
                continue
            try:
                value = json.loads(row["payload_json"])["value"]
                diagnostic = json.loads(value)
                arrival = float(row["arrival_epoch_ns"]) / 1.0e9
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            output.append((arrival, diagnostic))
    return output


def score_run(run_dir: Path, points: list[dict[str, float]], length: float, dt: float) -> dict:
    # Recorder outputs have existed in both layouts during the live tests:
    #   <run>/ <run>/controller_trace.csv
    #   <run>/model_id/<run>/controller_trace.csv
    # Keep the analysis tool independent of that packaging detail, while
    # retaining the run directory itself as the source of ground truth.
    trace_candidates = (
        run_dir / run_dir.name / "controller_trace.csv",
        run_dir / "model_id" / run_dir.name / "controller_trace.csv",
    )
    trace = next((candidate for candidate in trace_candidates if candidate.exists()), None)
    if trace is None:
        raise FileNotFoundError(
            "controller_trace.csv not found in supported recorder layouts: "
            + ", ".join(str(candidate) for candidate in trace_candidates)
        )
    truth_path = run_dir / "ground_truth.csv"
    diagnostics = load_diagnostics(trace)
    truth = load_truth(truth_path)
    horizons: dict[int, list[dict[str, float]]] = {1: [], 5: [], 10: [], 20: [], 30: []}
    accepted = {"accepted_optimal", "accepted_degraded"}
    for _, diagnostic in diagnostics:
        if diagnostic.get("status") not in accepted:
            continue
        source_ns = int(diagnostic.get("source_stamp_ns", 0))
        predictions = {int(item["n"]): item for item in diagnostic.get("predictions", [])}
        if source_ns <= 0:
            continue
        for horizon, item in predictions.items():
            if horizon not in horizons:
                continue
            actual = interpolate_truth(truth, source_ns / 1.0e9 + horizon * dt)
            if actual is None:
                continue
            reference = path_at(points, length, float(item["s_m"]))
            predicted_state = item["state"]
            predicted_x = reference["x"] - predicted_state[0] * math.sin(reference["psi"])
            predicted_y = reference["y"] + predicted_state[0] * math.cos(reference["psi"])
            predicted_yaw = wrap(reference["psi"] + predicted_state[1])
            dx = actual["x"] - predicted_x
            dy = actual["y"] - predicted_y
            tangent_x, tangent_y = math.cos(reference["psi"]), math.sin(reference["psi"])
            normal_x, normal_y = -math.sin(reference["psi"]), math.cos(reference["psi"])
            horizons[horizon].append({
                "cross_m": dx * normal_x + dy * normal_y,
                "along_m": dx * tangent_x + dy * tangent_y,
                "position_m": math.hypot(dx, dy),
                "heading_rad": abs(wrap(actual["yaw"] - predicted_yaw)),
                "speed_mps": actual["speed"] - float(predicted_state[2]),
            })

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "mean_abs": None, "p95_abs": None, "max_abs": None}
        absolute = sorted(abs(value) for value in values)
        return {
            "n": len(values),
            "mean_abs": fmean(absolute),
            "median_abs": median(absolute),
            "p95_abs": absolute[max(0, int(0.95 * len(absolute)) - 1)],
            "max_abs": max(absolute),
        }

    horizon_report = {}
    for horizon, samples in horizons.items():
        horizon_report[str(horizon)] = {
            "cross_track_prediction_error_m": summary([sample["cross_m"] for sample in samples]),
            "along_track_prediction_error_m": summary([sample["along_m"] for sample in samples]),
            "position_prediction_error_m": summary([sample["position_m"] for sample in samples]),
            "heading_prediction_error_rad": summary([sample["heading_rad"] for sample in samples]),
            "speed_prediction_error_mps": summary([sample["speed_mps"] for sample in samples]),
        }

    status_counts: dict[str, int] = {}
    progress = []
    for _, diagnostic in diagnostics:
        status = str(diagnostic.get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if isinstance(diagnostic.get("progress_m"), (float, int)):
            progress.append(float(diagnostic["progress_m"]))
    return {
        "run": run_dir.name,
        "diagnostic_count": len(diagnostics),
        "status_counts": status_counts,
        "max_progress_m": max(progress) if progress else None,
        "horizon_errors": horizon_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raceline", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    points, length = read_path(args.raceline)
    report = {
        "raceline": str(args.raceline),
        "dt_s": args.dt,
        "runs": [score_run(run, points, length, args.dt) for run in args.run],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
