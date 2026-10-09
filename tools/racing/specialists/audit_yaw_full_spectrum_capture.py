#!/usr/bin/env python3
"""Audit packet-qualified measured-state coverage in a yaw grid capture."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.evaluate_open_plane_body_dynamics import load_capture
from tools.racing.specialists import audit_yaw_full_domain_exact_two as yaw_audit


SPEED_BIN_MPS = 0.5
STEERING_BIN_RAD = 0.05
SPEED_BINS = 24
STEERING_CENTERS = tuple(round(index * STEERING_BIN_RAD, 2)
                          for index in range(-10, 11))


def audit_capture(bag: Path) -> dict[str, Any]:
    bag = bag.resolve()
    phases, experiment_end = yaw_audit._read_phases(bag)
    capture = load_capture(
        bag, include_nonvalid_phases=True, continuous_phased_run=True)
    samples = sorted(
        (sample for sequence in capture.sequences for sample in sequence
         if sample.packet_sequence >= 0),
        key=lambda sample: sample.packet_sequence,
    )
    sample_times = np.asarray([sample.receipt_ns for sample in samples],
                              dtype=np.int64)
    sequence = np.asarray([sample.packet_sequence for sample in samples],
                          dtype=np.int64)
    channel_cache = yaw_audit._channel_value_cache(samples)
    if len(sample_times) > 1 and np.any(np.diff(sample_times) < 0):
        raise ValueError("receipt time is not monotonic in packet order")

    transitions: Counter[str] = Counter()
    valid_probe_phases = 0
    excluded_invalid_phases = 0
    cells: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "conditions": set(), "events": set()})
    achieved_speeds: list[float] = []
    achieved_steering: list[float] = []
    exact_two_events = 0
    probe_transition_records: list[dict[str, Any]] = []

    for phase in phases:
        specs = yaw_audit._stimuli_for_phase(phase)
        if not specs:
            continue
        if phase.valid is not True:
            excluded_invalid_phases += 1
            continue
        valid_probe_phases += 1
        for spec in specs:
            result = yaw_audit._response_measurement(
                samples, phase, spec, channel_cache=channel_cache)
            count = result.get("response_packet_count")
            contiguous = result.get("packet_sequence_contiguous") is True
            label = ("2_noncontiguous" if count == 2 and not contiguous else
                     str(count if count is not None else
                         result.get("classification", "unknown")))
            transitions[label] += 1
            admitted = (
                count == 2
                and contiguous
                and isinstance(result.get("command_start_ns"), int))
            requested_speed = spec.get("requested_speed_mps")
            requested_abs_steering = spec.get(
                "requested_abs_steering_rad")
            requested_target = spec.get("target")
            event = str(spec.get("event", "unknown"))
            transition_index = int(spec.get("transition_index", 0))
            family = str(spec.get("family", "unknown"))
            target_key = (
                f"v{float(requested_speed):.3f}"
                f"_a{float(requested_abs_steering):.3f}"
                f"_target{float(requested_target):+.3f}"
                f"_{event}_idx{transition_index}_{family}"
                if requested_speed is not None
                and requested_abs_steering is not None
                and requested_target is not None else None)
            probe_transition_records.append({
                "phase_label": str(phase.label),
                "target_key": target_key,
                "requested_speed_mps": requested_speed,
                "requested_abs_steering_rad": requested_abs_steering,
                "requested_steering_target_rad": requested_target,
                "event": event,
                "transition_index": transition_index,
                "family": family,
                "response_packet_count": count,
                "packet_sequence_contiguous": contiguous,
                "classification": result.get("classification"),
                "admitted_exact_two": admitted,
            })
            if (count != 2
                    or not contiguous):
                continue
            if not isinstance(result.get("command_start_ns"), int):
                continue
            exact_two_events += 1
            start = max(
                int(phase.start_ns), int(result["command_start_ns"]) - 100_000_000)
            end = min(
                int(phase.end_ns), int(result["command_start_ns"]) + 500_000_000)
            first = int(np.searchsorted(sample_times, start, side="left"))
            last = int(np.searchsorted(sample_times, end, side="right"))
            condition = str(phase.label)
            event_key = (f"{phase.label}::{spec.get('event', 'unknown')}::"
                         f"{spec.get('transition_index', 0)}")
            for sample in samples[first:last]:
                state = sample.simulator_rigid_state
                if state is None or not np.isfinite(state[[7, 8]]).all():
                    continue
                steering = float(sample.actuators[0])
                if not math.isfinite(steering):
                    continue
                speed = float(np.hypot(state[7], state[8]))
                if not 0.0 <= speed < 12.0:
                    continue
                speed_bin = min(SPEED_BINS - 1,
                                int(math.floor(speed / SPEED_BIN_MPS)))
                steering_index = int(np.clip(
                    round(steering / STEERING_BIN_RAD), -10, 10))
                cell = cells[(speed_bin, steering_index)]
                cell["rows"] += 1
                cell["conditions"].add(condition)
                cell["events"].add(event_key)
                achieved_speeds.append(speed)
                achieved_steering.append(steering)

    grid = {}
    for speed_bin in range(SPEED_BINS):
        for steering_index in range(-10, 11):
            value = cells.get((speed_bin, steering_index), {})
            grid[f"speed_{speed_bin * 0.5:.1f}_{(speed_bin + 1) * 0.5:.1f}__steer_{steering_index * 0.05:+.2f}"] = {
                "rows": int(value.get("rows", 0)),
                "conditions": len(value.get("conditions", ())),
                "response_events": len(value.get("events", ())),
            }
    missing = [key for key, value in grid.items() if value["rows"] == 0]
    low_support = [key for key, value in grid.items() if 0 < value["rows"] < 20]
    packet_gaps_ms = (np.diff(sample_times) / 1.0e6
                      if len(sample_times) > 1 else np.empty(0))
    active_gaps = packet_gaps_ms[
        (packet_gaps_ms >= 15.0) & (packet_gaps_ms <= 60.0)]
    packet_steps = np.diff(sequence)
    valid_frequency = (1000.0 / float(np.mean(active_gaps))
                       if len(active_gaps) else None)

    return {
        "bag": str(bag),
        "run_id": bag.parents[1].name,
        "experiment_end": experiment_end,
        "quality": {
            "aborted": capture.aborted,
            "reason": capture.reason,
            "experiment_quality_failures": experiment_end.get(
                "quality_failures", []),
            "collisions_start_end": [capture.collision_count_start,
                                     capture.collision_count_end],
            "timing_faults": capture.timing_faults,
            "phase_count": capture.phase_count,
            "valid_phase_count": capture.valid_phase_count,
            "invalid_phase_count": capture.invalid_phase_count,
            "unscored_phase_count": capture.unscored_phase_count,
        },
        "packet_stream": {
            "samples": int(len(samples)),
            "active_interval_rate_hz": valid_frequency,
            "active_gap_ms_p50_p95_p99_max": (
                [float(np.quantile(active_gaps, q))
                 for q in (0.50, 0.95, 0.99, 1.0)]
                if len(active_gaps) else []),
            "active_intervals_over_35ms": int(
                np.count_nonzero(active_gaps > 35.0)),
            "intervals_over_60ms": int(
                np.count_nonzero(packet_gaps_ms > 60.0)),
            "packet_sequence_nonunit_steps": int(
                np.count_nonzero(packet_steps != 1)),
        },
        "exact_two_response_audit": {
            "valid_probe_phases": valid_probe_phases,
            "invalid_probe_phases_excluded": excluded_invalid_phases,
            "transitions_by_packet_count_or_reason": dict(transitions),
            "exact_two_contiguous_transitions": exact_two_events,
            "one_and_three_packet_rows_are_not_admitted": True,
        },
        "probe_transition_audit": {
            "records": probe_transition_records,
            "expected_exact_two_gate": "response_packet_count == 2, contiguous packet sequence, and identified command start",
        },
        "measured_gt_state": {
            "speed_min_max_mps": ([float(min(achieved_speeds)),
                                   float(max(achieved_speeds))]
                                  if achieved_speeds else []),
            "abs_steering_min_max_rad": ([
                float(min(abs(value) for value in achieved_steering)),
                float(max(abs(value) for value in achieved_steering))]
                if achieved_steering else []),
            "speed_bin_mps": SPEED_BIN_MPS,
            "signed_steering_bin_rad": STEERING_BIN_RAD,
            "cells_total": len(grid),
            "cells_occupied": len(grid) - len(missing),
            "cells_with_at_least_20_rows": sum(
                value["rows"] >= 20 for value in grid.values()),
            "empty_cells": missing,
            "cells_with_1_to_19_rows": low_support,
            "grid": grid,
        },
    }


def combine_captures(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one completed capture is required")
    aggregate_grid = {}
    for key in reports[0]["measured_gt_state"]["grid"]:
        per_run = [report["measured_gt_state"]["grid"][key]
                   for report in reports]
        aggregate_grid[key] = {
            "rows": int(sum(value["rows"] for value in per_run)),
            "independent_runs_with_rows": int(sum(
                value["rows"] > 0 for value in per_run)),
            "independent_runs_with_at_least_20_rows": int(sum(
                value["rows"] >= 20 for value in per_run)),
            "conditions": int(sum(value["conditions"] for value in per_run)),
            "response_events": int(sum(
                value["response_events"] for value in per_run)),
        }
    empty = [key for key, value in aggregate_grid.items()
             if value["rows"] == 0]
    weak = [key for key, value in aggregate_grid.items()
            if 0 < value["rows"] < 20]
    supported_twice = sum(
        value["independent_runs_with_at_least_20_rows"] >= 2
        for value in aggregate_grid.values())
    speed_ranges = [report["measured_gt_state"]["speed_min_max_mps"]
                    for report in reports
                    if report["measured_gt_state"]["speed_min_max_mps"]]
    steer_ranges = [report["measured_gt_state"]["abs_steering_min_max_rad"]
                    for report in reports
                    if report["measured_gt_state"]["abs_steering_min_max_rad"]]
    transition_counts: Counter[str] = Counter()
    probe_coverage: dict[str, dict[str, Any]] = {}
    for report in reports:
        transition_counts.update(
            report["exact_two_response_audit"][
                "transitions_by_packet_count_or_reason"])
        for record in report.get("probe_transition_audit", {}).get(
                "records", []):
            key = record.get("target_key")
            if key is None:
                continue
            summary = probe_coverage.setdefault(key, {
                "requested_speed_mps": record["requested_speed_mps"],
                "requested_abs_steering_rad": record[
                    "requested_abs_steering_rad"],
                "requested_steering_target_rad": record[
                    "requested_steering_target_rad"],
                "event": record["event"],
                "transition_index": record["transition_index"],
                "family": record["family"],
                "runs_seen": 0,
                "runs_with_exact_two": 0,
                "exact_two_events": 0,
                "rejected_packet_counts": {},
            })
            summary["runs_seen"] += 1
            if record["admitted_exact_two"]:
                summary["runs_with_exact_two"] += 1
                summary["exact_two_events"] += 1
            else:
                rejected = str(record["response_packet_count"]
                               if record["response_packet_count"] is not None
                               else record["classification"] or "unknown")
                rejected_counts = summary["rejected_packet_counts"]
                rejected_counts[rejected] = rejected_counts.get(rejected, 0) + 1
    return {
        "title": "Combined exact-two measured yaw coverage",
        "run_count": len(reports),
        "runs": [{
            "run_id": report["run_id"],
            "bag": report["bag"],
            "quality": report["quality"],
            "packet_stream": report["packet_stream"],
            "exact_two_response_audit": report["exact_two_response_audit"],
            "measured_gt_state": {
                key: value for key, value in report["measured_gt_state"].items()
                if key != "grid"},
        } for report in reports],
        "exact_two_transition_counts_all_runs": dict(transition_counts),
        "probe_transition_coverage": probe_coverage,
        "aggregate_measured_gt_state": {
            "speed_bin_mps": SPEED_BIN_MPS,
            "signed_steering_bin_rad": STEERING_BIN_RAD,
            "speed_min_max_mps": [
                float(min(value[0] for value in speed_ranges)),
                float(max(value[1] for value in speed_ranges)),
            ] if speed_ranges else [],
            "abs_steering_min_max_rad": [
                float(min(value[0] for value in steer_ranges)),
                float(max(value[1] for value in steer_ranges)),
            ] if steer_ranges else [],
            "cells_total": len(aggregate_grid),
            "cells_occupied": len(aggregate_grid) - len(empty),
            "cells_with_at_least_20_rows": sum(
                value["rows"] >= 20 for value in aggregate_grid.values()),
            "cells_with_at_least_20_rows_in_two_independent_runs": supported_twice,
            "empty_cells": empty,
            "cells_with_1_to_19_rows": weak,
            "grid": aggregate_grid,
        },
        "capture_reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", type=Path, nargs="+",
                        help="one or more completed ROS 2 SQLite bag files")
    parser.add_argument("--output", type=Path,
                        help="JSON output path (default: beside the bag)")
    args = parser.parse_args()
    reports = [audit_capture(bag) for bag in args.bags]
    report = (reports[0] if len(reports) == 1
              else combine_captures(reports))
    default_name = ("yaw_exact_two_coverage.json" if len(reports) == 1
                    else "yaw_exact_two_coverage_combined.json")
    output = args.output or args.bags[0].parent.parent / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    state_report = (report["measured_gt_state"]
                    if "measured_gt_state" in report else
                    report["aggregate_measured_gt_state"])
    print(json.dumps({
        "output": str(output),
        "run_count": report.get("run_count", 1),
        "quality": (report["quality"] if "quality" in report else
                    [run["quality"] for run in report["runs"]]),
        "measured_gt_state": {
            key: value for key, value in state_report.items()
            if key not in ("grid", "empty_cells", "cells_with_1_to_19_rows")},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
