#!/usr/bin/env python3
"""Summarize MPC candidate-prefix chains from a ROS bag, offline only.

The report preserves each prefix action's solver quality and predicted
corridor margin, groups adjacent publications into chains, and identifies the
first subsequent MPC status. It is descriptive evidence, not a causal test of
whether a chain caused a later fault.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tools import evaluate_mpc_bag as evaluator


SCHEMA = "sdu_apex_mpc_prefix_chain_analysis/v1"
PREFIX_STATUS = "accepted_candidate_prefix"


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "n": len(finite),
        "p50": statistics.median(finite) if finite else None,
        "p95": _percentile(finite, 0.95),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _row(record: dict[str, Any], chain_id: int, chain_length: int,
         index_in_chain: int, followup_status: str | None) -> dict[str, Any]:
    diagnostic = record.get("json", {})
    solver = diagnostic.get("solver") or {}
    action = diagnostic.get("issued_action")
    if isinstance(action, dict):
        steering = _finite(action.get("steering_rad"))
        throttle = _finite(action.get("throttle"))
    else:
        steering = throttle = None
    if steering is None or throttle is None:
        first_action = diagnostic.get("first_action")
        if isinstance(first_action, list) and len(first_action) >= 2:
            if steering is None:
                steering = _finite(first_action[0])
            if throttle is None:
                throttle = _finite(first_action[1])

    stage_slacks: list[float] = []
    for item in diagnostic.get("candidate_corridor_slack_by_stage_m", []):
        if not isinstance(item, dict):
            continue
        for side in ("lower", "upper"):
            value = _finite(item.get(side))
            if value is not None:
                stage_slacks.append(value)
    predicted_slack = _finite(
        diagnostic.get("minimum_predicted_corridor_slack_m"))
    if predicted_slack is None and stage_slacks:
        predicted_slack = min(stage_slacks)

    return {
        "chain_id": chain_id,
        "chain_length": chain_length,
        "index_in_chain": index_in_chain,
        "followup_status": followup_status,
        "bag_time_s": _finite(record.get("bag_time")),
        "source_stamp_ns": diagnostic.get("source_stamp_ns"),
        "progress_m": _finite(diagnostic.get("progress_m")),
        "steering_rad": steering,
        "steering_sign": (0 if steering is not None and abs(steering) < 1.0e-6
                          else 1 if steering is not None and steering > 0.0
                          else -1 if steering is not None else None),
        "throttle": throttle,
        "objective": _finite(solver.get("objective")),
        "iterations": _finite(solver.get("iterations")),
        "admm_status": solver.get("admm_status"),
        "admm_stop_reason": solver.get("admm_stop_reason"),
        "primal_residual": _finite(solver.get("primal_residual")),
        "dual_residual": _finite(solver.get("dual_residual")),
        "predicted_corridor_slack_m": predicted_slack,
        "minimum_prefix_corridor_slack_m": _finite(
            diagnostic.get("candidate_prefix_revalidation", {}).get(
                "minimum_corridor_slack_m")),
        "candidate_prefix_chain_count_reported": _finite(
            diagnostic.get("consecutive_candidate_prefix_actions")),
    }


def analyze_records(records: list[dict[str, Any]]) -> tuple[
        dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(enumerate(records), key=lambda pair: (
        _finite(pair[1].get("bag_time"))
        if _finite(pair[1].get("bag_time")) is not None else float(pair[0]),
        pair[0]))
    diagnostics = [record for _, record in ordered]
    groups: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(diagnostics):
        status = (diagnostics[cursor].get("json") or {}).get("status")
        if status != PREFIX_STATUS:
            cursor += 1
            continue
        start = cursor
        while cursor < len(diagnostics) and (
                (diagnostics[cursor].get("json") or {}).get("status") ==
                PREFIX_STATUS):
            cursor += 1
        groups.append((start, cursor))

    rows: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    for chain_id, (start, stop) in enumerate(groups, start=1):
        followup = None
        if stop < len(diagnostics):
            followup = (diagnostics[stop].get("json") or {}).get("status")
        chain_rows = [
            _row(diagnostics[index], chain_id, stop - start,
                 index - start + 1, followup)
            for index in range(start, stop)
        ]
        rows.extend(chain_rows)
        steer = [row["steering_rad"] for row in chain_rows
                 if row["steering_rad"] is not None]
        signs = [row["steering_sign"] for row in chain_rows
                 if row["steering_sign"] not in (None, 0)]
        reversals = sum(a != b for a, b in zip(signs, signs[1:]))
        chains.append({
            "chain_id": chain_id,
            "length": stop - start,
            "first_bag_time_s": chain_rows[0]["bag_time_s"],
            "last_bag_time_s": chain_rows[-1]["bag_time_s"],
            "first_progress_m": chain_rows[0]["progress_m"],
            "last_progress_m": chain_rows[-1]["progress_m"],
            "followup_status": followup,
            "steering_sign_counts": {
                "negative": sum(value < 0.0 for value in steer),
                "zero": sum(value == 0.0 for value in steer),
                "positive": sum(value > 0.0 for value in steer),
            },
            "steering_sign_reversals": reversals,
            "objective": _summary([row["objective"] for row in chain_rows
                                   if row["objective"] is not None]),
            "primal_residual": _summary([
                row["primal_residual"] for row in chain_rows
                if row["primal_residual"] is not None]),
            "dual_residual": _summary([
                row["dual_residual"] for row in chain_rows
                if row["dual_residual"] is not None]),
            "predicted_corridor_slack_m": _summary([
                row["predicted_corridor_slack_m"] for row in chain_rows
                if row["predicted_corridor_slack_m"] is not None]),
            "minimum_prefix_corridor_slack_m": _summary([
                row["minimum_prefix_corridor_slack_m"] for row in chain_rows
                if row["minimum_prefix_corridor_slack_m"] is not None]),
        })

    report = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "diagnostic_count": len(diagnostics),
        "candidate_prefix_action_count": len(rows),
        "candidate_prefix_chain_count": len(chains),
        "maximum_contiguous_chain_length": max(
            (chain["length"] for chain in chains), default=0),
        "chains": chains,
        "interpretation": (
            "Descriptive only. Chain length and logged diagnostics are not a "
            "causal predictor without matched states and controlled policy "
            "replay. Predicted corridor slack is not independent ground truth."),
        "limitations": [
            "Only diagnostics recorded in this bag are analyzed; missing solver fields remain null.",
            "A chain is defined as adjacent candidate-prefix status rows in bag-time order.",
            "Follow-up status is the next diagnostic row, not necessarily a physical collision outcome.",
            "This report does not replay candidate or certified-plan suffixes; use the source-plant feasibility audit for that A/B.",
        ],
    }
    return report, rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_run", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = evaluator.read_bag(args.bag_run)
    report, rows = analyze_records(data.get(evaluator.DIAGNOSTICS, []))
    database = next(args.bag_run.glob("*.db3"), None)
    report["bag"] = {
        "path": str(args.bag_run),
        "db3_path": str(database) if database else None,
        "db3_sha256": _sha256(database) if database else None,
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = list(rows[0]) if rows else [
        "chain_id", "chain_length", "index_in_chain", "followup_status",
        "bag_time_s", "source_stamp_ns", "progress_m", "steering_rad",
        "steering_sign", "throttle", "objective", "iterations",
        "admm_status", "admm_stop_reason", "primal_residual",
        "dual_residual", "predicted_corridor_slack_m",
        "minimum_prefix_corridor_slack_m",
        "candidate_prefix_chain_count_reported",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "report": str(json_path),
        "actions": str(csv_path),
        "candidate_prefix_chain_count": report["candidate_prefix_chain_count"],
        "maximum_contiguous_chain_length": report[
            "maximum_contiguous_chain_length"],
        "candidate_prefix_action_count": report[
            "candidate_prefix_action_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
