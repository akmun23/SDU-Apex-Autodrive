#!/usr/bin/env python3
"""Audit the recorded pose/velocity reference point offline.

The assembled source table contains the simulator root pose and Rigidbody
body velocity. This tool tests the lever-arm correction implied by candidate
reference points; it does not rewrite frames or alter runtime estimators.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


CANDIDATE_OFFSETS_M = {
    "same_point": 0.0,
    "position_rear_of_velocity_com": -0.15532,
    "position_com_ahead_of_velocity_rear": 0.15532,
}


def _stats(values: Iterable[float]) -> dict[str, Any]:
    values = sorted(abs(float(value)) for value in values if math.isfinite(value))
    if not values:
        return {"count": 0, "median_mps": None, "p95_mps": None, "max_mps": None}
    index = lambda fraction: values[min(len(values) - 1, int(math.ceil(fraction * len(values))) - 1)]
    return {
        "count": len(values),
        "median_mps": values[len(values) // 2],
        "p95_mps": index(0.95),
        "max_mps": values[-1],
    }


def _rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "dt_sim_s", "x_k_m", "y_k_m", "yaw_k_rad", "u_k_mps", "v_k_mps",
            "r_k_radps", "x_k1_m", "y_k1_m", "yaw_k1_rad", "u_k1_mps",
            "v_k1_mps", "r_k1_radps", "segment_id",
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        output = []
        for raw in reader:
            row = {key: float(raw[key]) for key in required}
            if all(math.isfinite(value) for value in row.values()):
                output.append(row)
        return output


def _transition_error(row: dict[str, float], position_offset_x_m: float) -> float:
    dt = row["dt_sim_s"]
    yaw_delta = row["yaw_k1_rad"] - row["yaw_k_rad"]
    yaw_mid = row["yaw_k_rad"] + 0.5 * yaw_delta
    u_mid = 0.5 * (row["u_k_mps"] + row["u_k1_mps"])
    v_mid = 0.5 * (row["v_k_mps"] + row["v_k1_mps"])
    r_mid = 0.5 * (row["r_k_radps"] + row["r_k1_radps"])
    # If the reported position point is offset from the velocity point by
    # [position_offset_x, 0], v_position = [u, v + r*position_offset_x].
    v_position = v_mid + r_mid * position_offset_x_m
    predicted_x_rate = math.cos(yaw_mid) * u_mid - math.sin(yaw_mid) * v_position
    predicted_y_rate = math.sin(yaw_mid) * u_mid + math.cos(yaw_mid) * v_position
    observed_x_rate = (row["x_k1_m"] - row["x_k_m"]) / dt
    observed_y_rate = (row["y_k1_m"] - row["y_k_m"]) / dt
    return math.hypot(observed_x_rate - predicted_x_rate,
                      observed_y_rate - predicted_y_rate)


def audit(run_dirs: list[Path]) -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    aggregate: dict[str, list[float]] = {name: [] for name in CANDIDATE_OFFSETS_M}
    for run_dir in run_dirs:
        path = run_dir / "assembled" / "model_transition_v4.csv"
        rows = _rows(path)
        run_result: dict[str, Any] = {"transition_count": len(rows), "candidates": {}}
        for name, offset in CANDIDATE_OFFSETS_M.items():
            errors = [_transition_error(row, offset) for row in rows]
            aggregate[name].extend(errors)
            run_result["candidates"][name] = {
                "position_offset_from_velocity_point_x_m": offset,
                "error": _stats(errors),
            }
        if rows:
            best = min(run_result["candidates"], key=lambda name: (
                run_result["candidates"][name]["error"]["median_mps"] or float("inf")))
            run_result["best_candidate_by_median"] = best
        per_run[run_dir.name] = run_result

    aggregate_result = {
        name: {
            "position_offset_from_velocity_point_x_m": CANDIDATE_OFFSETS_M[name],
            "error": _stats(errors),
        }
        for name, errors in aggregate.items()
    }
    best = min(aggregate_result, key=lambda name: (
        aggregate_result[name]["error"]["median_mps"] or float("inf")))
    return {
        "schema_version": 1,
        "status": "reference_point_audit_offline",
        "ground_truth_use": "offline_diagnostic_only",
        "runs": per_run,
        "aggregate": aggregate_result,
        "best_candidate_by_median": best,
        "interpretation": (
            "The offset is the reported position point relative to the point "
            "at which the body velocity is measured; this result is evidence, "
            "not an automatic runtime frame change."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dirs)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
