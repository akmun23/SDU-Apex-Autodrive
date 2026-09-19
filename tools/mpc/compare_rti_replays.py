#!/usr/bin/env python3
"""Compare two MPC RTI action replays by source event.

The candidate is accepted only when it preserves status and first published
commands within tight bounds while materially reducing adaptive-R2 execution.
This is intentionally standard-library only so it can run in ROS CI.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def percentile(values: list[float], fraction: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    index = min(len(finite) - 1, int(fraction * (len(finite) - 1)))
    return finite[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-steering-delta-rad", type=float, default=0.003)
    parser.add_argument("--max-speed-delta-mps", type=float, default=0.03)
    parser.add_argument("--max-trigger-ratio", type=float, default=0.40)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    if len(baseline) != len(candidate) or not baseline:
        raise SystemExit(
            f"row-count mismatch baseline={len(baseline)} candidate={len(candidate)}")

    steering_delta: list[float] = []
    speed_delta: list[float] = []
    baseline_triggers = 0
    candidate_triggers = 0
    status_mismatches = 0
    event_mismatches = 0

    for left, right in zip(baseline, candidate):
        if left.get("event_index") != right.get("event_index"):
            event_mismatches += 1
        if left.get("status") != right.get("status"):
            status_mismatches += 1
        baseline_triggers += int(number(left, "rti2_triggered") != 0.0)
        candidate_triggers += int(number(right, "rti2_triggered") != 0.0)
        steering_delta.append(abs(
            number(left, "steering_command_rad") -
            number(right, "steering_command_rad")))
        speed_delta.append(abs(
            number(left, "target_speed_mps") -
            number(right, "target_speed_mps")))

    maximum_steering_delta = max(
        value for value in steering_delta if math.isfinite(value))
    maximum_speed_delta = max(
        value for value in speed_delta if math.isfinite(value))
    trigger_ratio = (
        candidate_triggers / baseline_triggers
        if baseline_triggers > 0 else 0.0)

    print(
        "RTI A/B: "
        f"rows={len(baseline)} "
        f"baseline_triggers={baseline_triggers} "
        f"candidate_triggers={candidate_triggers} "
        f"trigger_ratio={trigger_ratio:.4f}")
    print(
        "published command delta: "
        f"steering[p95,max]={percentile(steering_delta, 0.95):.9f},"
        f"{maximum_steering_delta:.9f} rad "
        f"speed[p95,max]={percentile(speed_delta, 0.95):.9f},"
        f"{maximum_speed_delta:.9f} m/s")

    failures: list[str] = []
    if event_mismatches:
        failures.append(f"{event_mismatches} event-index mismatches")
    if status_mismatches:
        failures.append(f"{status_mismatches} status mismatches")
    if maximum_steering_delta > args.max_steering_delta_rad:
        failures.append(
            f"max steering delta {maximum_steering_delta:.9f} > "
            f"{args.max_steering_delta_rad:.9f}")
    if maximum_speed_delta > args.max_speed_delta_mps:
        failures.append(
            f"max speed delta {maximum_speed_delta:.9f} > "
            f"{args.max_speed_delta_mps:.9f}")
    if baseline_triggers > 0 and trigger_ratio > args.max_trigger_ratio:
        failures.append(
            f"R2 trigger ratio {trigger_ratio:.4f} > "
            f"{args.max_trigger_ratio:.4f}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("RTI A/B gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
