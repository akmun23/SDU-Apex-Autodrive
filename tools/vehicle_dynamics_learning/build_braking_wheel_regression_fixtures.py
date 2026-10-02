#!/usr/bin/env python3
"""Freeze paired train/validation braking and wheel-speed replay fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DT_S = 0.025
HISTORY_STEPS = 80
TARGET_EVENTS = (
    ("high_speed_low_steer", 11.074, -0.10),
    ("high_speed_moderate_left", 10.486, -0.14),
    ("high_speed_moderate_right", 10.486, 0.14),
    ("medium_speed_moderate_left", 9.487, -0.14),
    ("medium_speed_moderate_right", 9.487, 0.14),
    ("lower_speed_moderate_left", 7.014, -0.14),
    ("lower_speed_moderate_right", 7.014, 0.14),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_events(data: dict[str, np.ndarray], run_id: str
                      ) -> list[dict[str, Any]]:
    run_ids = data["run_ids"].astype(str)
    run_indices = np.flatnonzero(run_ids == run_id)
    if len(run_indices) != 1:
        raise ValueError(f"expected one source run row for {run_id}")
    run_index = int(run_indices[0])
    frames = data["frames"]
    rigid = data["simulator_rigid_state"]
    packet = data["packet_sequence"]
    dt = data["dt_s"]
    bounds = data["sequence_bounds"]
    sequence_runs = data["sequence_run_index"]
    labels = data["sequence_labels"].astype(str)
    result = []
    for sequence_id, (start_raw, end_raw) in enumerate(bounds):
        if int(sequence_runs[sequence_id]) != run_index:
            continue
        start, end = int(start_raw), int(end_raw)
        local_commands = frames[start:end, 8]
        local_edges = (np.flatnonzero(
            (local_commands[1:] <= 0.001)
            & (local_commands[:-1] > 0.001)) + 1)
        release_edges = (np.flatnonzero(
            (local_commands[1:] > 0.001)
            & (local_commands[:-1] <= 0.001)) + 1)
        for local_index in local_edges:
            onset = start + int(local_index)
            speed = float(np.hypot(rigid[onset - 1, 7], rigid[onset - 1, 8]))
            steering = float(frames[onset - 1, 7])
            for name, target_speed, target_steering in TARGET_EVENTS:
                if (abs(speed - target_speed) <= 0.025
                        and abs(steering - target_steering) <= 0.002):
                    release_local = next(
                        (int(index) for index in release_edges
                         if int(index) > int(local_index)), None)
                    if release_local is None:
                        raise ValueError(f"no brake release found for {name}/{run_id}")
                    release = start + release_local
                    event = {
                        "fixture_name": name,
                        "run_id": run_id,
                        "run_split": str(data["run_splits"][run_index]),
                        "sequence_id": sequence_id,
                        "sequence_label": str(labels[sequence_id]),
                        "sequence_bounds": [start, end],
                        "brake_command_index": onset,
                        "brake_initial_state_index": onset - 1,
                        "brake_history_start_index": onset - HISTORY_STEPS,
                        "brake_release_command_index": release,
                        "release_initial_state_index": release - 1,
                        "release_history_start_index": release - HISTORY_STEPS,
                        "speed_before_brake_mps": speed,
                        "steering_command_before_brake_rad": steering,
                        "throttle_command_before_brake": float(frames[onset - 1, 8]),
                        "brake_throttle_command": float(frames[onset, 8]),
                        "speed_at_release_mps": float(np.hypot(
                            rigid[release, 7], rigid[release, 8])),
                        "release_throttle_command": float(frames[release, 8]),
                        "score_horizons_s": [0.25, 0.5, 1.0, 2.0],
                    }
                    if (event["brake_history_start_index"] < start
                            or event["release_history_start_index"] < start):
                        raise ValueError("fixture event lacks the required 2 s history")
                    max_steps = min(80, end - onset - 1)
                    if max_steps < 20:
                        raise ValueError("braking fixture lacks a 0.5 s future window")
                    if not np.allclose(dt[start:end], DT_S, rtol=0.0, atol=1e-7):
                        raise ValueError("fixture sequence is not on the 25 ms timebase")
                    if not np.all(np.diff(packet[start:end]) == 1):
                        raise ValueError("packet gap in braking fixture sequence")
                    result.append(event)
                    break
    return result


def build(dataset_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "run_ids", "run_splits", "frames", "simulator_rigid_state",
        "sequence_bounds", "sequence_run_index", "sequence_labels",
        "packet_sequence", "dt_s",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"dataset lacks required fixture fields: {sorted(missing)}")
    if int(data["schema_version"][0]) != 8:
        raise ValueError("fixtures require an audited schema-8 race-domain dataset")
    if not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("dataset must use a fixed 25 ms timebase")

    train_id = "openplane_race_domain_moderate_braking_train_r01"
    validation_id = "openplane_race_domain_moderate_braking_validation_r01"
    events = _candidate_events(data, train_id) + _candidate_events(data, validation_id)
    expected = {(run_id, name) for run_id in (train_id, validation_id)
                for name, _, _ in TARGET_EVENTS}
    found = {(event["run_id"], event["fixture_name"]) for event in events}
    if found != expected:
        raise ValueError(
            f"braking fixture coverage mismatch: missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}")
    for event in events:
        expected_split = "train" if event["run_id"] == train_id else "validation"
        if event["run_split"] != expected_split:
            raise ValueError(f"unexpected split for fixture {event['fixture_name']}")

    report = {
        "schema_version": 1,
        "fixture_set": "braking_wheel_regression_fixtures_v1",
        "purpose": "deterministic active-braking, wheel/body decoupling, and brake-release evaluation",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "sample_period_s": DT_S,
        "history_steps": HISTORY_STEPS,
        "future_inputs_after_initial_state": [
            "steering_command_rad", "throttle_command_norm"],
        "future_truth_or_sensor_inputs": False,
        "independent_validation_run_id": validation_id,
        "fixture_count": len(events),
        "training_fixture_count": sum(event["run_split"] == "train" for event in events),
        "validation_fixture_count": sum(
            event["run_split"] == "validation" for event in events),
        "statistical_unit": "whole source run; paired train/validation event labels are not independent replicates",
        "events": events,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.dataset, args.output)
    print(json.dumps({
        "fixture_count": report["fixture_count"],
        "training": report["training_fixture_count"],
        "validation": report["validation_fixture_count"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
