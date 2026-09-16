#!/usr/bin/env python3
"""Score steering command/applied/feedback semantics on existing transitions.

This report is an offline actuator-semantics check.  It does not change the
simulator or MPC.  Each transition is evaluated causally from the previous
post-controller applied angle and the command target consumed by that
transition.  The GUI ``V1 Steering`` field is retained as a separate
pre-update feedback diagnostic.  The three candidate one-step laws are:

* rate limited: ``move(previous_applied, target, 3.2 * dt)``;
* first order: ``previous_applied + (1-exp(-dt/tau)) * (target-previous_applied)``;
* instantaneous: ``target``.

The report keeps command target, applied angle, feedback angle, one-step
deltas, and per-run p50/p95/p99 residuals together so a rate residual is not
promoted before the physical angle semantics are settled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_raceline_operating_envelope import _stats  # noqa: E402
from structured_vehicle_plant import MAX_STEERING_RAD, STEERING_RATE_RADPS  # noqa: E402


RUN_WINDOW_RE = re.compile(r"^(.*)@(\d+):(\d+)$")
DEFAULT_TAU_S = 0.037088548633299495


def _parse_spec(spec: str) -> tuple[Path, int | None, int | None]:
    match = RUN_WINDOW_RE.match(spec)
    if match is None:
        return Path(spec), None, None
    start, end = int(match.group(2)), int(match.group(3))
    if end <= start:
        raise ValueError(f"invalid run window: {spec}")
    return Path(match.group(1)), start, end


def _read_run(spec: str) -> list[dict[str, float]]:
    path, start, end = _parse_spec(spec)
    csv_path = path / "assembled" / "model_transition_v4.csv"
    required = {
        "dt_sim_s", "segment_id", "commanded_steering_norm_k1",
        "applied_steering_rad_k1", "simulator_feedback_steering_rad_k1",
    }
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{csv_path} is missing fields: {missing}")
        rows: list[dict[str, float]] = []
        previous_feedback = math.nan
        previous_applied = math.nan
        previous_commanded = math.nan
        previous_segment: int | None = None
        for raw in reader:
            segment = int(round(float(raw["segment_id"])))
            feedback = float(raw["simulator_feedback_steering_rad_k1"])
            row = {
                "dt_sim_s": float(raw["dt_sim_s"]),
                "segment_id": float(segment),
                "commanded_steering_norm_k1": float(
                    raw["commanded_steering_norm_k1"]),
                "applied_steering_rad_k1": float(
                    raw["applied_steering_rad_k1"]),
                "simulator_feedback_steering_rad_k1": feedback,
                "previous_feedback_rad": (
                    previous_feedback if previous_segment == segment else math.nan),
                "previous_applied_rad": (
                    previous_applied if previous_segment == segment else math.nan),
                "transition_commanded_norm": (
                    previous_commanded if previous_segment == segment else math.nan),
            }
            rows.append(row)
            previous_feedback = feedback
            previous_applied = float(raw["applied_steering_rad_k1"])
            previous_commanded = float(raw["commanded_steering_norm_k1"])
            previous_segment = segment
    if start is not None:
        rows = rows[start:end]
    return rows


def _rate_limited(previous: float, target: float, dt: float) -> float:
    return previous + max(-STEERING_RATE_RADPS * dt,
                          min(STEERING_RATE_RADPS * dt, target - previous))


def _first_order(previous: float, target: float, dt: float, tau_s: float) -> float:
    if tau_s <= 0.0 or not math.isfinite(tau_s):
        raise ValueError("first-order steering requires a positive tau")
    return previous + (1.0 - math.exp(-dt / tau_s)) * (target - previous)


def _stats_or_empty(values: Sequence[float]) -> dict[str, Any]:
    return _stats(values)


def _score_rows(rows: Sequence[dict[str, float]], tau_s: float) -> dict[str, Any]:
    selected = [
        row for row in rows
        if (math.isfinite(row["previous_applied_rad"]) and
            math.isfinite(row["applied_steering_rad_k1"]) and
            math.isfinite(row["transition_commanded_norm"]) and
            0.015 <= row["dt_sim_s"] <= 0.035)
    ]
    target_deltas: list[float] = []
    applied_deltas: list[float] = []
    feedback_deltas: list[float] = []
    applied_feedback_gaps: list[float] = []
    residuals = {
        "rate_limited": [],
        "first_order": [],
        "instantaneous": [],
    }
    feedback_residuals = {
        "rate_limited": [],
        "first_order": [],
        "instantaneous": [],
    }
    for row in selected:
        previous = row["previous_applied_rad"]
        target = max(-1.0, min(1.0,
                               row["transition_commanded_norm"])) * MAX_STEERING_RAD
        applied = row["applied_steering_rad_k1"]
        feedback = row["simulator_feedback_steering_rad_k1"]
        dt = row["dt_sim_s"]
        target_deltas.append(target - previous)
        applied_deltas.append(applied - previous)
        feedback_deltas.append(feedback - previous)
        applied_feedback_gaps.append(applied - feedback)
        predictions = {
            "rate_limited": _rate_limited(previous, target, dt),
            "first_order": _first_order(previous, target, dt, tau_s),
            "instantaneous": target,
        }
        for name, prediction in predictions.items():
            residuals[name].append(prediction - applied)
            feedback_residuals[name].append(prediction - feedback)
    return {
        "sample_count": len(selected),
        "one_step_deltas": {
            "target_minus_previous_applied_rad": _stats_or_empty(target_deltas),
            "applied_minus_previous_applied_rad": _stats_or_empty(applied_deltas),
            "feedback_minus_previous_applied_rad": _stats_or_empty(feedback_deltas),
            "applied_minus_feedback_rad": _stats_or_empty(applied_feedback_gaps),
        },
        "semantics": {
            name: {"residual_rad": _stats_or_empty(values)}
            for name, values in residuals.items()
        },
        "gui_feedback_semantics": {
            name: {"residual_rad": _stats_or_empty(values)}
            for name, values in feedback_residuals.items()
        },
    }


def score(run_specs: Sequence[str], output: Path,
          first_order_tau_s: float = DEFAULT_TAU_S) -> dict[str, Any]:
    if not run_specs:
        raise ValueError("at least one run is required")
    per_run = {
        spec: _score_rows(_read_run(spec), first_order_tau_s)
        for spec in run_specs
    }
    all_rows = [row for spec in run_specs for row in _read_run(spec)]
    aggregate = _score_rows(all_rows, first_order_tau_s)
    p95 = {
        name: values["residual_rad"]["p95"]
        for name, values in aggregate["semantics"].items()
    }
    result = {
        "schema_version": 1,
        "status": "offline_steering_semantics_not_runtime_promoted",
        "simulator_modified": False,
        "production_mpc_modified": False,
        "run_specs": list(run_specs),
        "input_contract": {
            "command_target": "previous_packet.commanded_steering_norm_k1 * max_steering_rad",
            "applied_field": "applied_steering_rad_k1 (post-controller state)",
            "feedback_field": "simulator_feedback_steering_rad_k1 (pre-update GUI getter)",
            "previous_state": "previous packet applied_steering_rad_k1 within the same segment",
            "transition_input": "previous packet commanded_steering_norm_k1",
            "dt_field": "dt_sim_s",
            "max_steering_rad": MAX_STEERING_RAD,
            "steering_rate_radps": STEERING_RATE_RADPS,
            "first_order_tau_s": first_order_tau_s,
        },
        "aggregate": aggregate,
        "per_run": per_run,
        "selection_diagnostic": {
            "lowest_aggregate_one_step_p95": min(p95, key=p95.get),
            "p95_residual_rad": p95,
            "not_a_promotion_decision": True,
            "reason": "physical command/applied/feedback angle semantics must be reviewed before residual use",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-order-tau", type=float,
                        default=DEFAULT_TAU_S)
    args = parser.parse_args()
    result = score(args.run, args.output, args.first_order_tau)
    print(json.dumps({
        "output": str(args.output),
        "sample_count": result["aggregate"]["sample_count"],
        "selection_diagnostic": result["selection_diagnostic"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
