#!/usr/bin/env python3
"""Quality-check and quantify throttle tracking in WP32 C4 captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import evaluate_open_plane_body_dynamics as body


PROFILE = "subnet_highsteer_transients"
DT_S = 0.025
EXPECTED_CONDITIONS = 20
MIN_PHASE_SAMPLES = 60
MAX_COMMAND_RMSE = 0.012
MAX_FEEDBACK_ENDPOINT_ERROR = 0.04


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_phase_events(bag_path: Path) -> list[tuple[int, dict[str, Any]]]:
    connection = sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = body.analysis._topic_map(connection)
        topic = "/open_plane_experiment/phase"
        if topic not in topics:
            raise ValueError("phase-event topic is missing from bag")
        events = []
        for receipt_ns, message in body.analysis._messages(connection, topics, topic):
            try:
                event = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                events.append((int(receipt_ns), event))
        return events
    finally:
        connection.close()


def _expected_ramp(elapsed_s: np.ndarray, start: float, end: float,
                   delay_s: float, ramp_s: float) -> np.ndarray:
    active = np.maximum(0.0, elapsed_s - delay_s)
    fraction = np.clip(active / ramp_s, 0.0, 1.0)
    fraction = np.where(elapsed_s < delay_s, 0.0, fraction)
    return start + fraction * (end - start)


def analyze(source_path: Path, bag_path: Path, output_path: Path) -> dict[str, Any]:
    source_path, bag_path, output_path = map(
        lambda path: path.resolve(), (source_path, bag_path, output_path))
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    manifest_path = source_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_rows = manifest.get("runs", [])
    if len(run_rows) != 1:
        raise ValueError("each report must describe exactly one captured run")
    run = run_rows[0]
    run_id = str(run.get("run_id", ""))
    if (run.get("aborted") or run.get("reason") != "schedule complete"
            or not run.get("clean_stream_and_collision_gate")
            or run.get("whole_bag_quality_failures")
            or run.get("quality_failures")
            or int(run.get("timing_faults", -1)) != 0
            or any(int(value) != 0 for value in run.get("collisions", []))):
        raise ValueError("capture did not pass whole-run collision/timing/data gates")
    alignment = run.get("packet_sequence_alignment", {})
    match_fraction = float(alignment.get("match_fraction", 0.0))
    if match_fraction < 0.999:
        raise ValueError("packet/odometry alignment is below 99.9 percent")
    for topic, stats in run.get("streams", {}).items():
        if (float(stats.get("hz", 0.0)) < 38.0
                or float(stats.get("gap_p95_ms", math.inf)) > 35.0
                or float(stats.get("gap_max_ms", math.inf)) > 120.0):
            raise ValueError(f"stream quality gate failed: {topic}")

    events = _read_phase_events(bag_path)
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    for _, event in events:
        label = str(event.get("label", ""))
        if not label.startswith("subnet_c"):
            continue
        if event.get("event") == "phase_start":
            starts[label] = event
        elif event.get("event") == "phase_end":
            ends[label] = event
    if len(starts) != EXPECTED_CONDITIONS or set(starts) != set(ends):
        raise ValueError("capture lacks all 20 matched transient start/end phases")
    if any(ends[label].get("status") != "complete"
           or ends[label].get("valid") is not True
           for label in starts):
        raise ValueError("one or more transient phases failed their run-time gates")

    with np.load(source_path, allow_pickle=False) as data:
        if int(data["schema_version"][0]) != 7:
            raise ValueError("schema-7 prepared source archive required")
        if len(data["run_ids"]) != 1 or str(data["run_ids"][0]) != run_id:
            raise ValueError("prepared archive run ID does not match its manifest")
        frames = np.asarray(data["frames"], dtype=np.float64)
        times_ns = np.asarray(data["sample_time_ns"], dtype=np.int64)
        bounds = np.asarray(data["sequence_bounds"], dtype=np.int64)
        reset_index = np.asarray(data["sequence_reset_index"], dtype=np.int64)
        names = data["feature_names"].astype(str).tolist()
        sequence_count = len(bounds)
        required = ("steering_feedback_rad", "throttle_feedback_norm",
                    "steering_command_rad", "throttle_command_norm")
        missing = set(required) - set(names)
        if missing:
            raise ValueError(f"prepared source lacks actuator channels: {sorted(missing)}")
        column = {name: names.index(name) for name in required}
        if sequence_count != EXPECTED_CONDITIONS:
            raise ValueError(
                f"expected 20 reset-isolated sequences, found {sequence_count}")
        for start, end in bounds:
            if int(end) - int(start) < MIN_PHASE_SAMPLES:
                raise ValueError("a reset-isolated condition has too few samples")

    condition_reports = []
    for label, start_event in sorted(starts.items()):
        end_event = ends[label]
        condition_id = str(start_event.get("condition_pair_id", ""))
        if not condition_id or condition_id not in label:
            raise ValueError(f"missing/mismatched condition ID for {label}")
        start_ns = int(start_event["wall_time_ns"])
        end_ns = int(end_event["wall_time_ns"])
        sequence_match = None
        for sequence_index, (left, right) in enumerate(bounds):
            left, right = int(left), int(right)
            if times_ns[left] <= start_ns <= times_ns[right - 1]:
                sequence_match = sequence_index
                break
        if sequence_match is None:
            raise ValueError(f"phase start is outside reset-isolated data: {label}")
        left, right = map(int, bounds[sequence_match])
        indices = np.flatnonzero(
            (times_ns[left:right] >= start_ns)
            & (times_ns[left:right] <= end_ns)) + left
        if len(indices) < MIN_PHASE_SAMPLES:
            raise ValueError(f"too few samples aligned to transient phase: {label}")

        elapsed = (times_ns[indices] - start_ns) * 1.0e-9
        elapsed += float(start_event.get("phase_elapsed_s", 0.0))
        start_throttle = float(start_event["throttle_start_norm"])
        end_throttle = float(start_event["throttle_end_norm"])
        expected = _expected_ramp(
            elapsed, start_throttle, end_throttle,
            float(start_event["throttle_stimulus_delay_s"]),
            float(start_event["throttle_ramp_duration_s"]))
        commanded = frames[indices, column["throttle_command_norm"]]
        feedback = frames[indices, column["throttle_feedback_norm"]]
        command_rmse = float(np.sqrt(np.mean((commanded - expected) ** 2)))
        feedback_rmse = float(np.sqrt(np.mean((feedback - expected) ** 2)))
        endpoint_count = min(10, len(indices))
        command_endpoint = float(np.median(commanded[-endpoint_count:]))
        feedback_endpoint = float(np.median(feedback[-endpoint_count:]))
        expected_steering = np.asarray(start_event.get("steering_waypoints", []),
                                       dtype=np.float64)
        condition_reports.append({
            "label": label,
            "condition_pair_id": condition_id,
            "reset_index": int(reset_index[sequence_match]),
            "sample_count": int(len(indices)),
            "target_throttle_norm": end_throttle,
            "command_rmse_to_intended_profile": command_rmse,
            "feedback_rmse_to_intended_profile": feedback_rmse,
            "command_endpoint_median": command_endpoint,
            "feedback_endpoint_median": feedback_endpoint,
            "feedback_endpoint_abs_error": abs(feedback_endpoint - end_throttle),
            "steering_waypoint_count": int(len(expected_steering)),
            "max_speed_mps": end_event.get("max_speed_mps"),
            "max_tilt_deg": end_event.get("max_tilt_deg"),
        })

    command_errors = [row["command_rmse_to_intended_profile"]
                      for row in condition_reports]
    feedback_errors = [row["feedback_endpoint_abs_error"]
                       for row in condition_reports]
    report = {
        "schema_version": 1,
        "profile": PROFILE,
        "run_id": run_id,
        "split": run.get("effective_split"),
        "status": "passed" if (
            max(command_errors) <= MAX_COMMAND_RMSE
            and max(feedback_errors) <= MAX_FEEDBACK_ENDPOINT_ERROR
        ) else "failed_actuator_profile_following",
        "source_archive": str(source_path),
        "source_archive_sha256": _sha256(source_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "bag": str(bag_path),
        "bag_sha256": _sha256(bag_path),
        "whole_run_quality": {
            "aborted": bool(run.get("aborted")),
            "collisions": run.get("collisions"),
            "timing_faults": int(run.get("timing_faults", -1)),
            "packet_match_fraction": match_fraction,
            "streams_hz": {name: float(value["hz"])
                            for name, value in run.get("streams", {}).items()},
            "reset_isolated_sequence_count": sequence_count,
        },
        "command_profile_rmse_threshold": MAX_COMMAND_RMSE,
        "feedback_endpoint_error_threshold": MAX_FEEDBACK_ENDPOINT_ERROR,
        "max_command_profile_rmse": max(command_errors),
        "max_feedback_endpoint_abs_error": max(feedback_errors),
        "condition_count": len(condition_reports),
        "conditions": condition_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="closed schema-7 source archive from prepare_dataset")
    parser.add_argument("--bag", type=Path, required=True,
                        help="closed ROS 2 SQLite bag")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.source, args.bag, args.output)
    print(json.dumps({
        "status": report["status"],
        "run_id": report["run_id"],
        "conditions": report["condition_count"],
        "max_command_profile_rmse": report["max_command_profile_rmse"],
        "max_feedback_endpoint_abs_error": report[
            "max_feedback_endpoint_abs_error"],
        "streams_hz": report["whole_run_quality"]["streams_hz"],
    }, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
