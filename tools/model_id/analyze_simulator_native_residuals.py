#!/usr/bin/env python3
"""Slice causal simulator-native replay residuals by failure regime.

The origin features and slice labels are computed from the origin row only.
The measured future state is used only after the candidate has been rolled out
for scoring, so this analysis does not alter the anti-leakage contract.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_simulator_native_model import (  # noqa: E402
    HORIZONS_S,
    _error_vector,
    _horizon_key,
    _read_runs,
    _score_origins,
    _state_from_row,
    _stats,
)
from simulator_native_model import (  # noqa: E402
    NativeModelParameters,
    contact_kinematics,
    f1tenth_prefab_parameters,
    step,
)


def _parameter_set(candidate: dict[str, Any], profile: str
                   ) -> NativeModelParameters:
    model = str(candidate["parameters"]["contact_model"])
    if profile != "f1tenth_prefab":
        parameters = NativeModelParameters(contact_model=model)
    else:
        parameters = f1tenth_prefab_parameters(model)
    gains = {
        field: float(value)
        for field, value in candidate["parameters"]["effective_gains"].items()
    }
    return replace(parameters, **gains,
                   parameter_provenance="offline_residual_analysis")


def _speed_bin(value: float) -> str:
    if value < 1.0:
        return "u<1"
    if value < 3.0:
        return "1<=u<3"
    if value < 6.0:
        return "3<=u<6"
    return "u>=6"


def _magnitude_bin(value: float, prefix: str, limits: tuple[float, ...]) -> str:
    if value < limits[0]:
        return f"{prefix}<{limits[0]:g}"
    for lower, upper in zip(limits, limits[1:]):
        if value < upper:
            return f"{lower:g}<={prefix}<{upper:g}"
    return f"{prefix}>={limits[-1]:g}"


def _origin_slices(row: dict[str, float], parameters: NativeModelParameters
                   ) -> dict[str, str]:
    state = _state_from_row(row)
    steering = float(row["applied_steering_rad_k1"])
    contacts = contact_kinematics(state, steering, parameters)
    max_sx = max(abs(contact["sx"]) for contact in contacts)
    max_sy = max(abs(contact["sy"]) for contact in contacts)
    delta = float(row["delta_k_rad"])
    throttle = float(row["applied_throttle_norm_k1"])
    steering_rate_event = abs(steering - delta) > 0.02
    return {
        "speed": _speed_bin(float(row["u_k_mps"])),
        "sy": _magnitude_bin(max_sy, "|Sy|", (0.01, 0.05, 0.15)),
        "sx": _magnitude_bin(max_sx, "|Sx|", (0.02, 0.10, 0.25)),
        "steering": (
            "straight" if abs(steering) < 0.03 else
            "left" if steering > 0.0 else "right"),
        "combined_demand": (
            "corner_and_throttle" if abs(steering) >= 0.03 and throttle > 0.10
            else "corner_no_throttle" if abs(steering) >= 0.03
            else "straight_throttle" if throttle > 0.10
            else "low_demand"),
        "steering_transition": "slew_or_reversal" if steering_rate_event else
        "steady_command",
    }


def _rollout(rows: list[dict[str, float]], origin: int,
             parameters: NativeModelParameters) -> list[dict[str, Any]]:
    first = rows[origin]
    segment = int(first["segment_id"])
    state = _state_from_row(first)
    elapsed = 0.0
    index = origin
    horizon_index = 0
    results: list[dict[str, Any]] = []
    while (index < len(rows) and horizon_index < len(HORIZONS_S) and
           elapsed < HORIZONS_S[-1] - 1.0e-10):
        row = rows[index]
        if int(row["segment_id"]) != segment:
            break
        state = step(
            state,
            row["applied_steering_rad_k1"] / parameters.steering_limit_rad,
            row["applied_throttle_norm_k1"], row["dt_sim_s"], parameters)
        elapsed += row["dt_sim_s"]
        index += 1
        while (horizon_index < len(HORIZONS_S) and
               elapsed >= HORIZONS_S[horizon_index] - 1.0e-10):
            target = rows[index - 1]
            results.append({
                "horizon_s": HORIZONS_S[horizon_index],
                "error": _error_vector(state, target),
            })
            horizon_index += 1
    return results


def _group_stats(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_horizon: dict[float, list[float]] = defaultdict(list)
    by_horizon_v: dict[float, list[float]] = defaultdict(list)
    by_horizon_r: dict[float, list[float]] = defaultdict(list)
    count = 0
    for record in records:
        count += 1
        horizon = float(record["horizon_s"])
        error = record["error"]
        by_horizon[horizon].append(float(error["position_m"]))
        by_horizon_v[horizon].append(float(error["v_mps"]))
        by_horizon_r[horizon].append(float(error["r_radps"]))
    return {
        "origins": count,
        "position_m": {
            _horizon_key(h): _stats(values)
            for h, values in sorted(by_horizon.items())
        },
        "v_mps": {
            _horizon_key(h): _stats(values)
            for h, values in sorted(by_horizon_v.items())
        },
        "r_radps": {
            _horizon_key(h): _stats(values)
            for h, values in sorted(by_horizon_r.items())
        },
    }


def _analyze_candidate(runs: dict[str, list[dict[str, float]]],
                       candidate: dict[str, Any], profile: str,
                       max_origins: int) -> dict[str, Any]:
    parameters = _parameter_set(candidate, profile)
    all_records: list[dict[str, Any]] = []
    slice_records: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list))
    run_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_name, rows in runs.items():
        for origin in _score_origins(rows, max_origins):
            origin_row = rows[origin]
            labels = _origin_slices(origin_row, parameters)
            for result in _rollout(rows, origin, parameters):
                record = {
                    "run": run_name,
                    "origin": origin,
                    "horizon_s": result["horizon_s"],
                    "error": result["error"],
                    "origin_state": {
                        "u_mps": origin_row["u_k_mps"],
                        "v_mps": origin_row["v_k_mps"],
                        "r_radps": origin_row["r_k_radps"],
                        "steering_rad": origin_row[
                            "applied_steering_rad_k1"],
                        "throttle_norm": origin_row[
                            "applied_throttle_norm_k1"],
                    },
                }
                all_records.append(record)
                run_records[run_name].append(record)
                for dimension, label in labels.items():
                    slice_records[dimension][label].append(record)

    def position_at(record: dict[str, Any], horizon: float) -> float:
        if abs(float(record["horizon_s"]) - horizon) > 1.0e-9:
            return math.nan
        return float(record["error"]["position_m"])

    final_horizon = HORIZONS_S[-1]
    ranked_origins = sorted(
        (record for record in all_records
         if math.isfinite(position_at(record, final_horizon))),
        key=lambda record: position_at(record, final_horizon), reverse=True)
    top_origins = []
    for record in ranked_origins[:20]:
        top_origins.append({
            "run": record["run"],
            "origin": record["origin"],
            "horizon_s": final_horizon,
            "position_error_m": position_at(record, final_horizon),
            **record["origin_state"],
        })
    ranked_runs = []
    for run_name, records in run_records.items():
        final_values = [position_at(record, final_horizon) for record in records]
        final_values = [value for value in final_values if math.isfinite(value)]
        ranked_runs.append({
            "run": run_name,
            "origins": len(set(record["origin"] for record in records)),
            "position_error_m": _stats(final_values),
            "catastrophic_over_1m": sum(value > 1.0 for value in final_values),
        })
    ranked_runs.sort(key=lambda item: item["position_error_m"]["p99"] or -1.0,
                     reverse=True)
    return {
        "parameters": candidate["parameters"],
        "summary": _group_stats(all_records),
        "slices": {
            dimension: {
                label: _group_stats(records)
                for label, records in sorted(groups.items())
            }
            for dimension, groups in sorted(slice_records.items())
        },
        "ranked_runs_at_final_horizon": ranked_runs,
        "top_origins_at_final_horizon": top_origins,
        "evaluation": {
            "max_origins_per_run": max_origins,
            "origin_features_are_causal": True,
            "future_ground_truth_used_only_for_scoring": True,
        },
    }


def analyze(source_report: Path, accepted_root: Path, output: Path,
            split: str = "validation", max_origins: int = 200) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(
        source_report.read_text(encoding="utf-8"))
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    names = report[f"{split}_runs"]
    runs = _read_runs(accepted_root, names)
    candidates = {
        name: _analyze_candidate(
            runs, candidate, str(report["parameter_profile"]), max_origins)
        for name, candidate in report["candidate_comparison"].items()
    }
    result = {
        "schema_version": 1,
        "status": "simulator_native_residual_slices_analyzed",
        "source_report": str(source_report),
        "split": split,
        "ground_truth_use": "offline_identification_and_scoring_only",
        "recursive_prediction_uses_future_gt": False,
        "candidate_comparison": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"),
                        default="validation")
    parser.add_argument("--max-origins-per-run", type=int, default=200)
    args = parser.parse_args()
    if args.max_origins_per_run < 1:
        raise ValueError("--max-origins-per-run must be positive")
    result = analyze(args.source_report, args.accepted_root, args.output,
                     args.split, args.max_origins_per_run)
    print(json.dumps({
        "output": str(args.output),
        "status": result["status"],
        "candidates": list(result["candidate_comparison"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
