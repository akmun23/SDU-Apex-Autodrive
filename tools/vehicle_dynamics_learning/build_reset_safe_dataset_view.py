#!/usr/bin/env python3
"""Create a metadata-rich schema-7 view without changing source bags or archives.

The schema-6 archive already has aligned fixed-step samples and receipt times.
This conversion adds explicit sequence/run metadata and splits any archived
sequence when a recorded reset-command edge occurred between adjacent samples.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_open_plane_dynamics as analysis  # noqa: E402
from tools.vehicle_dynamics_learning.prepare_dataset import (  # noqa: E402
    RESET_COMMAND_TOPIC,
    _replicate_index,
    _run_family,
)


def _events(bag_path: Path) -> tuple[bool, list[int], list[Any]]:
    connection = sqlite3.connect(
        bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        reset_starts: list[int] = []
        reset_present = RESET_COMMAND_TOPIC in topics
        if reset_present:
            active = False
            for receipt_ns, message in analysis._messages(
                    connection, topics, RESET_COMMAND_TOPIC):
                next_active = bool(message.data)
                if next_active and not active:
                    reset_starts.append(int(receipt_ns))
                active = next_active
        phases = []
        if analysis.PHASE in topics:
            phases, _ = analysis._phase_events(connection, topics)
        return reset_present, reset_starts, phases
    finally:
        connection.close()


def _split_points(packet_ids: np.ndarray, receipt_ns: np.ndarray,
                  reset_starts_ns: list[int]) -> list[int]:
    points = [0]
    for index in range(1, len(packet_ids)):
        packet_gap = int(packet_ids[index]) != int(packet_ids[index - 1]) + 1
        left = bisect.bisect_right(reset_starts_ns, int(receipt_ns[index - 1]))
        right = bisect.bisect_right(reset_starts_ns, int(receipt_ns[index]))
        reset_boundary = left < right
        if packet_gap or reset_boundary:
            points.append(index)
    return points


def _condition_for_segment(receipt_ns: np.ndarray, phases: list[Any],
                           fallback: str) -> str:
    if not len(receipt_ns):
        return fallback
    first, last = int(receipt_ns[0]), int(receipt_ns[-1])
    matches = [phase for phase in phases
               if phase.valid is True
               and phase.start_ns <= first
               and last <= phase.end_ns]
    if len(matches) == 1:
        return matches[0].label
    return fallback


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_npz = source_dir / "openplane_dynamics.npz"
    source_manifest_path = source_dir / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source = np.load(source_npz, allow_pickle=False)
    arrays = {key: source[key] for key in source.files}
    source.close()

    run_ids = arrays["run_ids"].astype(str)
    run_splits = arrays["run_splits"].astype(str)
    run_records = {row["run_id"]: row for row in manifest["runs"]}
    run_data: dict[str, tuple[bool, list[int], list[Any]]] = {}
    reset_audit: list[dict[str, Any]] = []
    for run_id, record in run_records.items():
        bag_path = Path(record["bag"])
        if not bag_path.is_absolute():
            bag_path = REPO_ROOT / bag_path
        if not bag_path.is_file():
            run_data[run_id] = (False, [], [])
            continue
        present, starts, phases = _events(bag_path)
        run_data[run_id] = (present, starts, phases)
        record["run_family"] = _run_family(run_id)
        record["reset_metadata"] = {
            "topic": RESET_COMMAND_TOPIC,
            "topic_present": present,
            "epoch_count": len(starts) if present else None,
        }
        if present:
            reset_audit.append({
                "run_id": run_id,
                "bag": record["bag"],
                "reset_epoch_count": len(starts),
                "reset_epoch_start_receipt_ns": starts,
                "clean_stream_and_collision_gate": bool(
                    record["clean_stream_and_collision_gate"]),
                "effective_split": record["effective_split"],
            })

    bounds = arrays["sequence_bounds"]
    sequence_runs = arrays["sequence_run_index"]
    old_labels = arrays["sequence_labels"].astype(str)
    packet_ids = arrays["packet_sequence"]
    receipts = arrays["sample_time_ns"]
    frame_keys = (
        "frames", "sensor_frames", "sensor_valid", "imu_attitude_frames",
        "imu_attitude_valid", "dt_s", "packet_sequence", "sample_time_ns",
        "odom_pose_xyyaw", "simulator_pose_xyyaw", "lap_count",
        "simulator_rigid_state", "simulator_linear_acceleration",
    )

    pieces: list[tuple[int, int, int, str, int]] = []
    for sequence_index, (start_value, end_value) in enumerate(bounds):
        start, end = int(start_value), int(end_value)
        run_index = int(sequence_runs[sequence_index])
        run_id = str(run_ids[run_index])
        present, reset_starts, phases = run_data[run_id]
        local_packet = packet_ids[start:end]
        local_receipts = receipts[start:end]
        split_points = _split_points(
            local_packet, local_receipts, reset_starts if present else [])
        split_points.append(end - start)
        for left, right in zip(split_points, split_points[1:]):
            if right - left < 2:
                continue
            absolute_start, absolute_end = start + left, start + right
            condition = _condition_for_segment(
                receipts[absolute_start:absolute_end], phases,
                str(old_labels[sequence_index]))
            reset_index = (
                bisect.bisect_right(reset_starts,
                                    int(receipts[absolute_start]))
                if present else -1)
            pieces.append((absolute_start, absolute_end, run_index,
                           condition, reset_index))

    if not pieces:
        raise ValueError("reset-safe view contains no sequences")

    for key in frame_keys:
        arrays[key] = np.concatenate(
            [arrays[key][start:end] for start, end, *_ in pieces], axis=0)
    new_bounds: list[tuple[int, int]] = []
    new_sequence_runs: list[int] = []
    new_labels: list[str] = []
    new_condition_ids: list[int] = []
    new_reset_ids: list[int] = []
    new_replicates: list[int] = []
    frame_run_index: list[np.ndarray] = []
    frame_reset_index: list[np.ndarray] = []
    condition_catalog: list[dict[str, Any]] = []
    condition_ids: dict[tuple[str, str], int] = {}
    cursor = 0

    for start, end, run_index, condition, reset_index in pieces:
        run_id = str(run_ids[run_index])
        condition_key = (run_id, condition)
        if condition_key not in condition_ids:
            condition_ids[condition_key] = len(condition_catalog)
            condition_catalog.append({
                "condition_id": condition_ids[condition_key],
                "run_id": run_id,
                "run_family": _run_family(run_id),
                "label": condition,
                "replicate_index": _replicate_index(condition),
            })
        length = end - start
        new_bounds.append((cursor, cursor + length))
        new_sequence_runs.append(run_index)
        new_labels.append(condition)
        new_condition_ids.append(condition_ids[condition_key])
        new_reset_ids.append(reset_index)
        new_replicates.append(_replicate_index(condition))
        frame_run_index.append(np.full(length, run_index, dtype=np.int32))
        frame_reset_index.append(np.full(length, reset_index, dtype=np.int32))
        cursor += length

    # Verify the defining sequence invariants after conversion.
    violations: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(new_bounds):
        run_id = str(run_ids[new_sequence_runs[index]])
        present, reset_starts, _ = run_data[run_id]
        seq_packets = arrays["packet_sequence"][start:end]
        seq_receipts = arrays["sample_time_ns"][start:end]
        if np.any(np.diff(seq_packets) != 1):
            violations.append({"sequence": index, "reason": "packet_gap"})
        if present and any(
                bisect.bisect_right(reset_starts, int(left))
                < bisect.bisect_right(reset_starts, int(right))
                for left, right in zip(seq_receipts[:-1], seq_receipts[1:])):
            violations.append({"sequence": index, "reason": "crosses_reset"})
    if violations:
        raise ValueError(f"reset-safe sequence verification failed: {violations[:5]}")

    arrays["schema_version"] = np.asarray([7], dtype=np.int32)
    arrays["sequence_bounds"] = np.asarray(new_bounds, dtype=np.int64)
    arrays["sequence_run_index"] = np.asarray(new_sequence_runs, dtype=np.int32)
    arrays["sequence_labels"] = np.asarray(new_labels, dtype="U256")
    arrays["sequence_condition_id"] = np.asarray(new_condition_ids,
                                                 dtype=np.int32)
    arrays["sequence_reset_index"] = np.asarray(new_reset_ids, dtype=np.int32)
    arrays["sequence_replicate_index"] = np.asarray(new_replicates,
                                                    dtype=np.int32)
    arrays["frame_run_index"] = np.concatenate(frame_run_index)
    arrays["frame_reset_index"] = np.concatenate(frame_reset_index)
    arrays["run_families"] = np.asarray(
        [_run_family(str(run_id)) for run_id in run_ids], dtype="U32")
    arrays["condition_labels"] = np.asarray(
        [row["label"] for row in condition_catalog], dtype="U256")
    arrays["condition_run_index"] = np.asarray(
        [int(np.flatnonzero(run_ids == row["run_id"])[0])
         for row in condition_catalog], dtype=np.int32)
    np.savez_compressed(output_dir / "openplane_dynamics.npz", **arrays)

    manifest["schema_version"] = 7
    manifest["source_schema_version"] = 6
    manifest["reset_safe_conversion"] = {
        "method": ("split existing fixed-25ms archive sequences at every recorded "
                   "reset-command rising edge; preserve packet-gap boundaries"),
        "reset_command_topic": RESET_COMMAND_TOPIC,
        "sequence_invariants_verified": True,
        "packet_gap_crossings": 0,
        "reset_crossings": 0,
        "runs_with_reset_topic": [row["run_id"] for row in reset_audit],
        "reset_audit": reset_audit,
    }
    manifest["condition_catalog"] = condition_catalog
    manifest["sequence_metadata"] = {
        "run_family": "practice_track/open_plane/other, derived from run ID prefix",
        "condition_id": "index into condition_catalog; labels from valid phase events where available",
        "replicate_index": "parsed from throttle phase labels; -1 when not applicable",
        "reset_index": "recorded reset-command rising-edge count before sequence start; -1 if unavailable",
        "failure_status": "run-level collision/timing/alignment/aborted gate in manifest runs",
    }
    manifest["packet_sequence_policy"] = (
        "Source-stamp aligned; each sequence has consecutive packet IDs and "
        "does not cross a recorded reset-command rising edge; dt is fixed 25 ms.")
    manifest["selection"] = (
        "Schema-6 clean archive converted without changing row labels; existing "
        "sequences split at packet gaps and recorded reset-command rising edges.")
    manifest["plant_continuity_mode"] = True
    manifest["export"] = {
        **manifest["export"],
        "file": "openplane_dynamics.npz",
        "samples": int(cursor),
        "sequences": len(new_bounds),
        "compressed_bytes": (output_dir / "openplane_dynamics.npz").stat().st_size,
        "runs_with_exported_sequences": sorted({
            str(run_ids[run_index]) for run_index in new_sequence_runs}),
    }
    manifest["source_archive"] = str(source_npz)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=(REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
                 "plant_teacher_mixed_dataset_full3d_fixed25_20261001"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = build(args.source_dir, args.output_dir)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"schema-7 dataset conversion failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["export"], indent=2))
    print(f"wrote {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
