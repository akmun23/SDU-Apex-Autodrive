#!/usr/bin/env python3
"""Append strictly screened unseen practice intervals for evaluation only.

This view is deliberately separate from the immutable training archive.  It
keeps the original run splits unchanged and assigns the new real practice run
to ``unseen_practice`` so frozen teachers can be scored without mixing it
with existing open-plane test captures.  It is not a dataset-preparation
exception and must never be used for fitting or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DT_S = 0.025
SPEED_CAP_MPS = 12.0
POST_CAP_COOLDOWN_STEPS = 80
MIN_INTERVAL_SAMPLES = 80
UNSEEN_SPLIT = "unseen_practice"


def _workspace_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(
            f"evaluation inputs and outputs must be inside the workspace: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cdr_string(data: bytes) -> str:
    if len(data) < 8:
        raise ValueError("truncated CDR string")
    size = struct.unpack_from("<I", data, 4)[0]
    if size < 1 or 8 + size > len(data):
        raise ValueError("invalid CDR string length")
    return data[8:8 + size - 1].decode("utf-8", errors="strict")


def _bridge_packet_events(bag_path: Path) -> tuple[dict[int, dict[str, Any]],
                                                   list[dict[str, Any]]]:
    uri = f"file:{bag_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        topic_ids = dict(connection.execute("SELECT id, name FROM topics"))
        ids_by_name = {name: topic_id for topic_id, name in topic_ids.items()}
        timing_id = ids_by_name.get(
            "/autodrive/roboracer_1/bridge_packet_timing")
        fault_id = ids_by_name.get("/autodrive/roboracer_1/bridge_timing_fault")
        detail_id = ids_by_name.get(
            "/autodrive/roboracer_1/bridge_timing_fault_detail")
        if timing_id is None or fault_id is None:
            raise ValueError("bag lacks bridge packet timing/fault streams")

        packets: dict[int, dict[str, Any]] = {}
        for (serialized,) in connection.execute(
                "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (timing_id,)):
            event = json.loads(_cdr_string(serialized))
            sequence = int(event["packet_sequence"])
            if sequence in packets:
                raise ValueError(f"duplicate bridge packet sequence {sequence}")
            packets[sequence] = event

        faults = []
        for timestamp, serialized in connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (fault_id,)):
            if len(serialized) < 5:
                raise ValueError("truncated bridge timing fault")
            if bool(serialized[4]):
                detail = ""
                if detail_id is not None:
                    match = connection.execute(
                        "SELECT data FROM messages WHERE topic_id=? AND timestamp<=? "
                        "ORDER BY timestamp DESC LIMIT 1",
                        (detail_id, timestamp)).fetchone()
                    if match:
                        detail = _cdr_string(match[0])
                faults.append({"timestamp_ns": int(timestamp), "detail": detail})
    if not packets:
        raise ValueError("bridge packet timing stream is empty")
    return packets, faults


def _race_domain_mask(speed: np.ndarray, valid: np.ndarray,
                      cooldown_steps: int = POST_CAP_COOLDOWN_STEPS
                      ) -> np.ndarray:
    """Require continuous valid in-domain data after any >12 m/s excursion."""
    speed = np.asarray(speed, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if speed.shape != valid.shape:
        raise ValueError("speed and validity arrays must align")
    keep = np.zeros(len(speed), dtype=bool)
    cooldown = 0
    after_excursion = False
    for index, (value, row_valid) in enumerate(zip(speed, valid)):
        if not np.isfinite(value) or value > SPEED_CAP_MPS:
            cooldown = cooldown_steps
            after_excursion = True
            continue
        if not row_valid:
            if after_excursion:
                cooldown = cooldown_steps
            continue
        if after_excursion and cooldown:
            cooldown -= 1
            continue
        after_excursion = False
        keep[index] = True
    return keep


def _true_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(
        np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _append(base: dict[str, np.ndarray], sequences: list[dict[str, np.ndarray]],
            run_id: str, split: str) -> dict[str, np.ndarray]:
    if not run_id.startswith("practice_unseen_"):
        raise ValueError("unseen practice run ids must start with practice_unseen_")
    if run_id in set(base["run_ids"].astype(str)):
        raise ValueError(f"run id already exists in base dataset: {run_id}")
    if split != UNSEEN_SPLIT:
        raise ValueError(f"new practice data must remain {UNSEEN_SPLIT!r}")
    if not sequences:
        raise ValueError("no screened practice intervals to append")

    run_index = len(base["run_ids"])
    condition_index = len(base["condition_labels"])
    frame_offset = len(base["frames"])
    source_sequence_base = (int(base["sequence_source_index"].max()) + 1
                            if len(base["sequence_source_index"]) else 0)

    frame_keys = (
        "frames", "sensor_frames", "sensor_valid", "imu_attitude_frames",
        "imu_attitude_valid", "dt_s", "packet_sequence", "sample_time_ns",
        "odom_pose_xyyaw", "simulator_pose_xyyaw", "lap_count",
        "simulator_rigid_state", "simulator_linear_acceleration",
        "frame_run_index", "frame_reset_index", "frame_source_index",
        "frame_source_sequence_index", "frame_domain_speed_mps",
    )
    frame_parts: dict[str, list[np.ndarray]] = {key: [] for key in frame_keys}
    new_bounds = []
    new_sequence_runs = []
    new_sequence_labels = []
    new_condition_ids = []
    new_reset_indices = []
    new_replicate_indices = []
    new_source_conditions = []
    new_source_sequences = []
    cursor = frame_offset
    for ordinal, sequence in enumerate(sequences):
        count = len(sequence["frames"])
        source_sequence = source_sequence_base + ordinal
        speed = np.hypot(sequence["frames"][:, 0], sequence["frames"][:, 1])
        frame_parts["frames"].append(sequence["frames"])
        frame_parts["sensor_frames"].append(sequence["sensor_frames"])
        frame_parts["sensor_valid"].append(sequence["sensor_valid"])
        frame_parts["imu_attitude_frames"].append(sequence["attitude_frames"])
        frame_parts["imu_attitude_valid"].append(sequence["attitude_valid"])
        frame_parts["dt_s"].append(np.full(count, DT_S, dtype=np.float32))
        frame_parts["packet_sequence"].append(sequence["packet_sequence"])
        frame_parts["sample_time_ns"].append(sequence["receipt_times_ns"])
        frame_parts["odom_pose_xyyaw"].append(sequence["odom_pose_xyyaw"])
        frame_parts["simulator_pose_xyyaw"].append(sequence["simulator_pose_xyyaw"])
        frame_parts["lap_count"].append(sequence["lap_count"])
        frame_parts["simulator_rigid_state"].append(sequence["simulator_rigid_state"])
        frame_parts["simulator_linear_acceleration"].append(
            sequence["simulator_linear_acceleration"])
        frame_parts["frame_run_index"].append(
            np.full(count, run_index, dtype=np.int32))
        frame_parts["frame_reset_index"].append(
            np.zeros(count, dtype=np.int32))
        frame_parts["frame_source_index"].append(sequence["packet_sequence"])
        frame_parts["frame_source_sequence_index"].append(
            np.full(count, source_sequence, dtype=np.int32))
        frame_parts["frame_domain_speed_mps"].append(speed.astype(np.float32))
        new_bounds.append((cursor, cursor + count))
        cursor += count
        new_sequence_runs.append(run_index)
        new_sequence_labels.append("unseen_practice_branch_only")
        new_condition_ids.append(condition_index)
        new_reset_indices.append(0)
        new_replicate_indices.append(0)
        new_source_conditions.append(condition_index)
        new_source_sequences.append(source_sequence)

    result = {key: np.asarray(base[key]).copy() for key in base}
    for key, parts in frame_parts.items():
        result[key] = np.concatenate((base[key], *parts), axis=0)
    result["sequence_bounds"] = np.concatenate((
        base["sequence_bounds"], np.asarray(new_bounds, dtype=np.int64)), axis=0)
    result["sequence_run_index"] = np.concatenate((
        base["sequence_run_index"], np.asarray(new_sequence_runs, dtype=np.int32)))
    result["sequence_labels"] = np.concatenate((
        base["sequence_labels"], np.asarray(new_sequence_labels, dtype="U256")))
    for key, additions in (
            ("sequence_condition_id", new_condition_ids),
            ("sequence_reset_index", new_reset_indices),
            ("sequence_replicate_index", new_replicate_indices),
            ("sequence_source_condition_id", new_source_conditions),
            ("sequence_source_index", new_source_sequences)):
        result[key] = np.concatenate((base[key], np.asarray(additions,
                                                            dtype=np.int32)))
    result["condition_labels"] = np.concatenate((
        base["condition_labels"], np.asarray(
            ["unseen_practice_branch_only"], dtype="U256")))
    result["condition_run_index"] = np.concatenate((
        base["condition_run_index"], np.asarray([run_index], dtype=np.int32)))
    for key, value in (
            ("run_ids", run_id), ("run_families", "practice_track"),
            ("run_splits", split), ("training_families", "practice_race")):
        result[key] = np.concatenate((
            base[key], np.asarray([value], dtype=base[key].dtype)))
    return result


def build(bag_path: Path, base_path: Path, output_path: Path,
          run_id: str,
          validation_report_path: Path | None = None) -> dict[str, Any]:
    bag_path = bag_path.resolve()
    base_path = base_path.resolve()
    output_path = output_path.resolve()
    bag_relative = _workspace_relative(bag_path)
    base_relative = _workspace_relative(base_path)
    output_relative = _workspace_relative(output_path)
    strict_validation = None
    validation_report_relative = None
    if validation_report_path is not None:
        validation_report_path = validation_report_path.resolve()
        validation_report_relative = _workspace_relative(validation_report_path)
        if not validation_report_path.is_file():
            raise ValueError(f"strict validation report does not exist: {validation_report_path}")
        strict_validation = json.loads(
            validation_report_path.read_text(encoding="utf-8"))
        if (strict_validation.get("admitted") is not True
                or strict_validation.get("role") != "validation"
                or strict_validation.get("training_allowed") is not False
                or strict_validation.get("bag_sha256") != _sha256(bag_path)
                or strict_validation.get("run_id") != bag_path.parents[1].name
                or strict_validation.get("expected_laps") != 6):
            raise ValueError(
                "strict capture report must admit this exact six-lap validation bag")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    report_path = output_path.with_name("branch_view_manifest.json")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    from tools.vehicle_dynamics_learning import prepare_dataset

    record, extracted = prepare_dataset._extract(
        bag_path, coalesce_contiguous_phases=True,
        include_nonvalid_phases=True)
    if record["collisions"] != [0, 0]:
        raise ValueError(f"practice run had collisions: {record['collisions']}")

    packet_events, fault_events = _bridge_packet_events(bag_path)
    packet_ids = set(packet_events)
    all_receipts = [int(seq[8][-1]) for seq in extracted if len(seq[8])]
    last_sample_receipt = max(all_receipts, default=0)
    trailing_faults = [event for event in fault_events
                       if event["timestamp_ns"] > last_sample_receipt]
    active_faults = [event for event in fault_events
                     if event["timestamp_ns"] <= last_sample_receipt]
    if active_faults:
        raise ValueError(f"bridge timing fault occurred during captured samples: {active_faults}")
    if fault_events and (len(trailing_faults) != len(fault_events)
                         or any(event["detail"] !=
                                "simulator Socket.IO connection lost"
                                for event in trailing_faults)):
        raise ValueError(f"unrecognized bridge shutdown fault: {fault_events}")

    ordered_packets = [packet_events[key] for key in sorted(packet_events)]
    packet_sequences = np.asarray(
        [int(item["packet_sequence"]) for item in ordered_packets],
        dtype=np.int64)
    arrivals = np.asarray(
        [int(item["bridge_arrival_monotonic_ns"]) for item in ordered_packets],
        dtype=np.int64)
    gaps_ms = np.diff(arrivals) / 1e6
    if (len(packet_sequences) < 2
            or np.any(np.diff(packet_sequences) != 1)
            or not np.isfinite(gaps_ms).all()
            or np.percentile(gaps_ms, 95) > 35.0
            or np.max(gaps_ms) > 60.0):
        raise ValueError("active bridge packet stream failed sequence/cadence checks")

    candidates: list[dict[str, np.ndarray]] = []
    interval_audit = []
    source_sequence_audit = []
    for sequence_index, sequence in enumerate(extracted):
        (label, frames, sensor_frames, sensor_valid, attitude_frames,
         attitude_valid, dt_s, packet_ids_local, receipt_times, odom_pose,
         simulator_pose, lap_count, rigid_state, acceleration) = sequence
        speed = np.hypot(frames[:, 0], frames[:, 1])
        source_sequence_audit.append({
            "sequence": sequence_index,
            "samples": int(len(frames)),
            "packet_first_last": ([int(packet_ids_local[0]),
                                   int(packet_ids_local[-1])]
                                  if len(packet_ids_local) else None),
            "lap_count_min_max": ([int(lap_count.min()), int(lap_count.max())]
                                  if len(lap_count) else None),
            "speed_min_max_mps": ([float(speed.min()), float(speed.max())]
                                  if len(speed) else None),
            "invalid_sensor_rows": int(np.count_nonzero(~sensor_valid)),
            "receipt_gap_p95_max_ms": (
                [float(np.percentile(np.diff(receipt_times) / 1e6, 95)),
                 float(np.max(np.diff(receipt_times) / 1e6))]
                if len(receipt_times) > 1 else None),
        })
        valid = (sensor_valid & np.isfinite(frames).all(axis=1)
                 & np.isfinite(sensor_frames).all(axis=1)
                 & np.isfinite(simulator_pose).all(axis=1)
                 & np.isfinite(rigid_state).all(axis=1)
                 & np.isfinite(acceleration).all(axis=1)
                 & (receipt_times < (min(
                     (event["timestamp_ns"] for event in fault_events),
                     default=np.iinfo(np.int64).max)))
                 & np.isin(packet_ids_local, list(packet_ids)))
        valid &= _race_domain_mask(speed, valid)
        for start, end in _true_intervals(valid):
            local_ids = packet_ids_local[start:end]
            local_receipts = receipt_times[start:end]
            if len(local_ids) < MIN_INTERVAL_SAMPLES:
                interval_audit.append({
                    "source_sequence": sequence_index, "start": start,
                    "end": end, "samples": end - start,
                    "admitted": False, "reason": "too_short_for_branch_context"})
                continue
            receipt_gaps_ms = np.diff(local_receipts) / 1e6
            if (np.any(np.diff(local_ids) != 1)
                    or np.any(np.diff(local_receipts) <= 0)
                    or np.percentile(receipt_gaps_ms, 95) > 35.0
                    or np.max(receipt_gaps_ms) > 60.0
                    or not np.allclose(dt_s[start:end], DT_S,
                                       rtol=0.0, atol=1e-7)):
                interval_audit.append({
                    "source_sequence": sequence_index, "start": start,
                    "end": end, "samples": end - start,
                    "admitted": False, "reason": "interval_continuity_or_cadence_failed"})
                continue
            candidates.append({
                "frames": frames[start:end].astype(np.float32, copy=True),
                "sensor_frames": np.column_stack((
                    sensor_frames[start:end, :-1],
                    np.full(end - start, DT_S, dtype=np.float32))),
                "sensor_valid": sensor_valid[start:end].astype(bool, copy=True),
                "attitude_frames": attitude_frames[start:end].astype(
                    np.float32, copy=True),
                "attitude_valid": attitude_valid[start:end].astype(
                    bool, copy=True),
                "packet_sequence": local_ids.astype(np.int64, copy=True),
                "receipt_times_ns": local_receipts.astype(np.int64, copy=True),
                "odom_pose_xyyaw": odom_pose[start:end].astype(
                    np.float32, copy=True),
                "simulator_pose_xyyaw": simulator_pose[start:end].astype(
                    np.float32, copy=True),
                "lap_count": lap_count[start:end].astype(np.int32, copy=True),
                "simulator_rigid_state": rigid_state[start:end].astype(
                    np.float32, copy=True),
                "simulator_linear_acceleration": acceleration[start:end].astype(
                    np.float32, copy=True),
            })
            interval_audit.append({
                "source_sequence": sequence_index, "source_label": label,
                "start": start, "end": end, "samples": end - start,
                "first_packet": int(local_ids[0]),
                "last_packet": int(local_ids[-1]),
                "lap_count_min_max": [int(lap_count[start:end].min()),
                                      int(lap_count[start:end].max())],
                "speed_min_max_mps": [float(speed[start:end].min()),
                                      float(speed[start:end].max())],
                "receipt_gap_p95_max_ms": [float(np.percentile(
                    receipt_gaps_ms, 95)), float(np.max(receipt_gaps_ms))],
                "admitted": True,
            })

    if not candidates:
        raise ValueError("no continuous, sensor/ground-truth-valid unseen intervals")
    with np.load(base_path, allow_pickle=False) as archive:
        base = {key: archive[key] for key in archive.files}
    if int(base["schema_version"][0]) != 8 or str(base["dataset_role"][0]) != "race_domain":
        raise ValueError("base must be the immutable schema-8 race-domain view")
    if np.any(base["frame_domain_speed_mps"] > SPEED_CAP_MPS + 1e-5):
        raise ValueError("base race view already contains unsupported speed rows")
    combined = _append(base, candidates, run_id, UNSEEN_SPLIT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **combined)
    report = {
        "schema_version": 1,
        "purpose": (
            "strict-gated six-lap practice transfer and short-branch evaluation only"
            if strict_validation is not None else
            "frozen-model and short-branch evaluation only"),
        "training_or_checkpoint_selection_use": False,
        "split": UNSEEN_SPLIT,
        "run_id": run_id,
        "source_run_id": bag_path.parents[1].name,
        "bag": bag_relative,
        "bag_sha256": _sha256(bag_path),
        "strict_capture_validation": ({
            "report": validation_report_relative,
            "report_sha256": _sha256(validation_report_path),
            "admitted": True,
            "role": "validation_only",
            "expected_laps": 6,
        } if strict_validation is not None else None),
        "base_dataset": base_relative,
        "base_dataset_sha256": _sha256(base_path),
        "combined_dataset": output_relative,
        "combined_dataset_sha256": _sha256(output_path),
        "whole_capture_admission": {
            "admitted_to_training_dataset": False,
            "reason": record.get("reason"),
            "aborted_flag": record.get("aborted"),
            "quality_failures": record.get("quality_failures"),
            "collision_count_start_end": record.get("collisions"),
            "packet_sequence_alignment": record.get("packet_sequence_alignment"),
            "trailing_shutdown_faults": trailing_faults,
            "active_faults": active_faults,
        },
        "bridge_packet_timing": {
            "packet_count": len(packet_sequences),
            "packet_sequence_contiguous": bool(np.all(
                np.diff(packet_sequences) == 1)),
            "arrival_gap_p50_p95_max_ms": [float(np.median(gaps_ms)),
                                            float(np.percentile(gaps_ms, 95)),
                                            float(np.max(gaps_ms))],
        },
        "race_domain": {
            "speed_cap_mps": SPEED_CAP_MPS,
            "post_excursion_cooldown_steps": POST_CAP_COOLDOWN_STEPS,
            "all_admitted_rows_at_or_below_cap": True,
        },
        "admitted_intervals": interval_audit,
        "source_sequences": source_sequence_audit,
        "admitted_interval_count": len(candidates),
        "admitted_sample_count": int(sum(len(item["frames"])
                                           for item in candidates)),
        "limitations": [
            ("This six-lap capture is validation-only and is not eligible for training."
             if strict_validation is not None else
             "This single partial practice capture does not pass the whole-run training gate."),
            "Only continuous intervals with complete legal sensor inputs and simulator labels are included.",
            "A single source run does not establish run-level generalization.",
            "Ground truth is used only as the prediction target and branch anchor/score, never as future plant input.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("base_dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-report", type=Path,
                        help="strict six-lap whole-capture admission report")
    args = parser.parse_args()
    report = build(args.bag, args.base_dataset, args.output, args.run_id,
                   args.validation_report)
    print(json.dumps({
        "run_id": report["run_id"],
        "split": report["split"],
        "intervals": report["admitted_interval_count"],
        "samples": report["admitted_sample_count"],
        "output": report["combined_dataset"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
