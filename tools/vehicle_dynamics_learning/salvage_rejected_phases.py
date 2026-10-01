#!/usr/bin/env python3
"""Audit rejected whole-run captures and export independently clean phases.

Only closed bags listed in the mixed-dataset manifest are read. Intervals are
split into independently named evidence tiers: phase-validated; clean telemetry
whose only experiment failure was speed matching; and complete but unscored or
unphased intervals. Every exported interval must pass its local stream,
collision/timing-fault, alignment, and sequence-length gates. These archives
remain separate from the canonical teacher dataset.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import analyze_open_plane_dynamics as analysis  # noqa: E402
from tools import evaluate_open_plane_body_dynamics as body  # noqa: E402
from tools.vehicle_dynamics_learning import prepare_dataset as dataset  # noqa: E402


DEFAULT_MANIFEST = (REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
                   "plant_teacher_mixed_dataset_20260930/manifest.json")
DEFAULT_REFERENCE = (REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
                     "plant_teacher_mixed_dataset_20260930/openplane_dynamics.npz")
DEFAULT_OUTPUT = (REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
                 "boundary_phase_salvage_dataset_20260930")
MIN_SEQUENCE_SAMPLES = dataset.HISTORY_STEPS + dataset.ROLLOUT_STEPS + 1


def _phase_events(path: Path) -> tuple[
        list[dict[str, Any]], list[tuple[int, int]], list[tuple[int, bool]],
        dict[str, list[int]], bool]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        has_phase_markers = analysis.PHASE in topics
        starts: dict[int, tuple[int, dict[str, Any]]] = {}
        ends: dict[int, tuple[int, dict[str, Any]]] = {}
        if has_phase_markers:
            for receipt_ns, message in analysis._messages(connection, topics,
                                                           analysis.PHASE):
                try:
                    event = json.loads(message.data)
                except (TypeError, json.JSONDecodeError):
                    continue
                kind = event.get("event")
                if kind == "phase_start":
                    starts[int(event["phase_index"])] = (receipt_ns, event)
                elif kind == "phase_end":
                    ends[int(event["phase_index"])] = (receipt_ns, event)

        records = []
        for index, (start_ns, start) in sorted(starts.items()):
            end = ends.get(index)
            if end is None:
                continue
            end_ns, result = end
            records.append({"index": index, "label": str(start.get("label", "")),
                            "start_ns": start_ns, "end_ns": end_ns,
                            "start": start, "end": result})

        collisions = [
            (receipt_ns, int(message.data))
            for receipt_ns, message in analysis._messages(
                connection, topics, analysis.COLLISIONS)
        ]
        faults = [
            (receipt_ns, bool(message.data))
            for receipt_ns, message in analysis._messages(
                connection, topics, analysis.TIMING_FAULT)
        ]
        streams: dict[str, list[int]] = {}
        for topic in body.STREAM_TOPICS:
            if topic not in topics:
                raise ValueError(f"missing required stream: {topic}")
            topic_id = topics[topic][0]
            streams[topic] = [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? "
                "ORDER BY timestamp, id", (topic_id,))]
        return records, collisions, faults, streams, has_phase_markers
    finally:
        connection.close()


def _stream_stats(receipts: list[int]) -> dict[str, float | int | None]:
    if len(receipts) < 2:
        return {"samples": len(receipts), "hz": None, "p95_gap_ms": None,
                "max_gap_ms": None}
    gaps = np.diff(np.asarray(receipts, dtype=np.int64)).astype(np.float64) / 1e6
    duration_s = (receipts[-1] - receipts[0]) / 1e9
    return {
        "samples": len(receipts),
        "hz": float((len(receipts) - 1) / duration_s) if duration_s > 0 else None,
        "p95_gap_ms": float(np.quantile(gaps, 0.95)),
        "max_gap_ms": float(np.max(gaps)),
    }


def _phase_quality(record: dict[str, Any], streams: dict[str, list[int]],
                   collisions: list[tuple[int, int]],
                   faults: list[tuple[int, bool]],
                   require_declared_valid: bool = True,
                   include_phase_failures: bool = True
                   ) -> tuple[dict[str, Any], list[str]]:
    start_ns, end_ns = int(record["start_ns"]), int(record["end_ns"])
    end = record["end"]
    failures: list[str] = []
    if require_declared_valid and end.get("valid") is not True:
        failures.append("phase_not_marked_valid")
    if include_phase_failures and end.get("quality_failures"):
        failures.extend(f"phase:{item}" for item in end["quality_failures"])

    stream_report: dict[str, Any] = {}
    command_topics = set(body.COMMAND_STREAM_TOPICS)
    for topic, all_receipts in streams.items():
        left = bisect.bisect_left(all_receipts, start_ns)
        right = bisect.bisect_right(all_receipts, end_ns)
        stats = _stream_stats(all_receipts[left:right])
        stream_report[topic] = stats
        max_gap_ms = 120.0 if topic in command_topics else 60.0
        if (stats["hz"] is None or stats["hz"] < 38.0
                or stats["p95_gap_ms"] is None
                or stats["p95_gap_ms"] > 35.0
                or stats["max_gap_ms"] is None
                or stats["max_gap_ms"] > max_gap_ms):
            failures.append(f"stream_quality:{topic}")

    collision_times = [row[0] for row in collisions]
    collision_before = bisect.bisect_right(collision_times, start_ns) - 1
    collision_at_start = (collisions[collision_before][1]
                          if collision_before >= 0 else 0)
    phase_collisions = [value for time_ns, value in collisions
                        if start_ns <= time_ns <= end_ns]
    if collision_at_start != 0 or any(value != 0 for value in phase_collisions):
        failures.append("collision_count_nonzero")

    if any(value for time_ns, value in faults if start_ns <= time_ns <= end_ns):
        failures.append("bridge_timing_fault")

    return {
        "stream_stats": stream_report,
        "collision_count_at_start": collision_at_start,
        "collision_counts_in_phase": phase_collisions,
        "timing_faults_in_phase": sum(
            value for time_ns, value in faults if start_ns <= time_ns <= end_ns),
        "phase_duration_s": (end_ns - start_ns) / 1e9,
        "declared_valid": end.get("valid"),
        "declared_quality_failures": end.get("quality_failures", []),
        "steering_error_p95_rad": end.get("steering_error_p95_rad"),
        "speed_error_p95_mps": end.get("speed_error_p95_mps"),
    }, failures


def _command_feedback_errors(frames: np.ndarray) -> dict[str, float]:
    return {
        "steering_command_feedback_abs_error_p95_rad": float(np.quantile(
            np.abs(frames[:, 3] - frames[:, 7]), 0.95)),
        "throttle_command_feedback_abs_error_p95_norm": float(np.quantile(
            np.abs(frames[:, 4] - frames[:, 8]), 0.95)),
    }


def _write_archive(entries: list[tuple[Any, ...]], output_dir: Path,
                   quality_tier: str) -> dict[str, Any]:
    if not entries:
        return {"file": None, "sequences": 0, "samples": 0, "runs": 0}
    run_ids = sorted({entry[0] for entry in entries})
    run_index = {run_id: index for index, run_id in enumerate(run_ids)}
    frames_blocks, sensors_blocks, sensor_valid_blocks = [], [], []
    attitude_blocks, attitude_valid_blocks, dt_blocks = [], [], []
    packet_sequence_blocks, time_blocks = [], []
    odom_pose_blocks, simulator_pose_blocks, lap_count_blocks = [], [], []
    simulator_rigid_state_blocks, simulator_acceleration_blocks = [], []
    bounds, sequence_runs, sequence_labels, sequence_splits = [], [], [], []
    cursor = 0
    for entry in entries:
        (run_id, split, label, tier, frames, sensors, sensor_valid, attitude,
         attitude_valid, dt, packet_sequence, times, odom_pose,
         simulator_pose, lap_count, simulator_rigid_state,
         simulator_acceleration) = entry
        frames_blocks.append(frames)
        sensors_blocks.append(sensors)
        sensor_valid_blocks.append(sensor_valid)
        attitude_blocks.append(attitude)
        attitude_valid_blocks.append(attitude_valid)
        dt_blocks.append(dt)
        packet_sequence_blocks.append(packet_sequence)
        time_blocks.append(times)
        odom_pose_blocks.append(odom_pose)
        simulator_pose_blocks.append(simulator_pose)
        lap_count_blocks.append(lap_count)
        simulator_rigid_state_blocks.append(simulator_rigid_state)
        simulator_acceleration_blocks.append(simulator_acceleration)
        bounds.append((cursor, cursor + len(frames)))
        cursor += len(frames)
        sequence_runs.append(run_index[run_id])
        sequence_labels.append(label)
        sequence_splits.append(split)
    frames_all = np.concatenate(frames_blocks).astype(np.float32, copy=False)
    filename = f"openplane_dynamics_{quality_tier}.npz"
    np.savez_compressed(
        output_dir / filename,
        schema_version=np.asarray([dataset.SCHEMA_VERSION], dtype=np.int32),
        feature_names=np.asarray(dataset.FEATURE_NAMES, dtype="U64"),
        sensor_feature_names=np.asarray(dataset.SENSOR_FEATURE_NAMES, dtype="U64"),
        attitude_feature_names=np.asarray(dataset.ATTITUDE_FEATURE_NAMES, dtype="U64"),
        predicted_state_names=np.asarray(dataset.PREDICTED_STATE_NAMES, dtype="U64"),
        frames=frames_all,
        sensor_frames=np.concatenate(sensors_blocks).astype(np.float32, copy=False),
        sensor_valid=np.concatenate(sensor_valid_blocks).astype(bool, copy=False),
        imu_attitude_frames=np.concatenate(attitude_blocks).astype(np.float32, copy=False),
        imu_attitude_valid=np.concatenate(attitude_valid_blocks).astype(bool, copy=False),
        dt_s=np.concatenate(dt_blocks).astype(np.float32, copy=False),
        packet_sequence=np.concatenate(packet_sequence_blocks).astype(
            np.int64, copy=False),
        sample_time_ns=np.concatenate(time_blocks).astype(np.int64, copy=False),
        odom_pose_xyyaw=np.concatenate(odom_pose_blocks).astype(
            np.float32, copy=False),
        simulator_pose_xyyaw=np.concatenate(simulator_pose_blocks).astype(
            np.float32, copy=False),
        lap_count=np.concatenate(lap_count_blocks).astype(np.int32, copy=False),
        simulator_rigid_state=np.concatenate(
            simulator_rigid_state_blocks).astype(np.float32, copy=False),
        simulator_linear_acceleration=np.concatenate(
            simulator_acceleration_blocks).astype(np.float32, copy=False),
        sequence_bounds=np.asarray(bounds, dtype=np.int64),
        sequence_run_index=np.asarray(sequence_runs, dtype=np.int32),
        sequence_labels=np.asarray(sequence_labels, dtype="U256"),
        sequence_splits=np.asarray(sequence_splits, dtype="U16"),
        run_ids=np.asarray(run_ids, dtype="U128"),
        run_splits=np.asarray([_split_for_run(run_id) for run_id in run_ids],
                              dtype="U16"),
    )
    speed = np.hypot(frames_all[:, 0], frames_all[:, 1])
    return {
        "file": filename,
        "sequences": len(entries),
        "samples": len(frames_all),
        "runs": len(run_ids),
        "train_sequences": sum(entry[1] == "train" for entry in entries),
        "test_sequences": sum(entry[1] == "test" for entry in entries),
        "maximum_speed_mps": float(speed.max()),
        "speed_p50_p95_mps": [float(np.quantile(speed, q)) for q in (0.50, 0.95)],
        "maximum_abs_steering_rad": float(np.max(np.abs(frames_all[:, 3]))),
        "maximum_throttle_command_norm": float(np.max(frames_all[:, 8])),
        "compressed_bytes": (output_dir / filename).stat().st_size,
    }


def _sequence_fingerprint(frames: np.ndarray, dt: np.ndarray) -> str:
    return dataset._fingerprint([(frames, dt)])


def _load_reference_fingerprints(path: Path) -> set[str]:
    archive = np.load(path, allow_pickle=False)
    try:
        frames = archive["frames"]
        dt = archive["dt_s"]
        return {_sequence_fingerprint(frames[start:end], dt[start:end])
                for start, end in archive["sequence_bounds"]}
    finally:
        archive.close()


def _split_for_run(run_id: str) -> str:
    lowered = run_id.lower()
    if "source_player" in lowered:
        return "excluded"
    if "holdout" in lowered:
        return "test"
    if lowered == "openplane_full_input_excitation_20260927_tilt_guard_replay":
        # Its experiment log records a fresh spawn, new seed, and newly issued
        # commands. "replay" describes the repeated experiment, not playback.
        # Keep it train-only; it is never treated as independent evaluation.
        return "train"
    if "replay" in lowered:
        return "excluded"
    return "train"


def audit(manifest_path: Path, reference_npz: Path,
          output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    if not manifest_path.is_file() or not reference_npz.is_file():
        raise ValueError("manifest or canonical reference archive is missing")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = [item for item in source_manifest["runs"]
                  if not item.get("clean_stream_and_collision_gate", False)]
    reference_hashes = _load_reference_fingerprints(reference_npz)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_reports: list[dict[str, Any]] = []
    accepted: dict[str, list[tuple[Any, ...]]] = {
        "verified": [], "speed_target_mismatch": [], "unverified": []}
    errors: list[dict[str, str]] = []
    seen_hashes = set(reference_hashes)
    for run in candidates:
        run_id = run["run_id"]
        bag_path = REPO_ROOT / run["bag"]
        run_report: dict[str, Any] = {
            "run_id": run_id,
            "bag": run["bag"],
            "source_bytes_at_manifest": run.get("bytes"),
            "whole_capture_failures": run.get("quality_failures", []),
            "source_split": _split_for_run(run_id),
            "split_rationale": (
                "fresh simulator rerun (seed 20260933); train-only, not a "
                "holdout or recorded-data playback"
                if run_id == "openplane_full_input_excitation_20260927_tilt_guard_replay"
                else "preserve existing whole-run holdout/replay policy"),
            "phases": [],
            "salvaged_sequences": 0,
            "salvaged_samples": 0,
        }
        try:
            if not bag_path.is_file():
                raise ValueError("source bag missing")
            if (run.get("bytes") is not None
                    and bag_path.stat().st_size != int(run["bytes"])):
                raise ValueError("source bag size changed since manifest snapshot")

            raw_phases, collisions, faults, streams, has_phase_markers = (
                _phase_events(bag_path))
            phase_counts = Counter(phase["label"] for phase in raw_phases)
            _, extracted = dataset._extract(
                bag_path, include_nonvalid_phases=True)
            by_label: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
            for sequence in extracted:
                by_label[sequence[0]].append(sequence)

            run_report["phase_markers_available"] = has_phase_markers
            if not has_phase_markers:
                # Older captures have no phase events but can still carry
                # lap-count-delimited continuous sequences. Treat each such
                # sequence as its own candidate interval; never accept the
                # aborted capture wholesale.
                for segment_index, sequence in enumerate(extracted):
                    (label, frames, sensors, sensor_valid, attitude,
                     attitude_valid, dt, packet_sequence, times, odom_pose,
                     simulator_pose, lap_count, simulator_rigid_state,
                     simulator_acceleration) = sequence
                    interval = {"start_ns": int(times[0]),
                                "end_ns": int(times[-1]),
                                "end": {"valid": None, "quality_failures": []}}
                    quality, failures = _phase_quality(
                        interval, streams, collisions, faults,
                        require_declared_valid=False)
                    if len(frames) < MIN_SEQUENCE_SAMPLES:
                        failures.append("sequence_shorter_than_history_plus_rollout")
                    if (not np.isfinite(frames).all()
                            or not np.isfinite(sensors).all()
                            or not np.isfinite(dt).all()):
                        failures.append("nonfinite_plant_features")
                    if _split_for_run(run_id) == "excluded":
                        failures.append("excluded_by_existing_replay_split_policy")
                    digest = _sequence_fingerprint(frames, dt)
                    if digest in seen_hashes:
                        failures.append(
                            "exact_duplicate_of_existing_or_salvaged_sequence")
                    tier = "unverified"
                    if not failures:
                        seen_hashes.add(digest)
                        accepted[tier].append((
                            run_id, _split_for_run(run_id), label, tier,
                            frames, sensors, sensor_valid, attitude,
                            attitude_valid, dt, packet_sequence, times,
                            odom_pose, simulator_pose, lap_count,
                            simulator_rigid_state,
                            simulator_acceleration))
                        run_report["salvaged_sequences"] += 1
                        run_report["salvaged_samples"] += len(frames)
                    run_report["phases"].append({
                        "phase_index": segment_index,
                        "label": label,
                        "status": "salvaged" if not failures else "rejected",
                        "evidence_basis": (
                            "continuous lap-count sequence; no phase markers; "
                            "only this bounded segment was evaluated"),
                        "sequence_count": 1,
                        "samples_salvaged": len(frames) if not failures else 0,
                        "quality": quality,
                        "reasons": sorted(set(failures)),
                        "command_feedback": {
                            "interpretation": (
                                "both command and measured actuator channels "
                                "are retained; actuator lag is plant behavior"),
                            **_command_feedback_errors(frames),
                        },
                    })
                run_report["source_disposition"] = (
                    "retain_raw_source_for_salvaged_sequences"
                    if run_report["salvaged_sequences"] else
                    "retain_pending_manual_review_after_read_error"
                    if run_report.get("error") else
                    "remove_capture_directory_no_usable_sequence")
                run_reports.append(run_report)
                continue

            for phase in raw_phases:
                quality, failures = _phase_quality(
                    phase, streams, collisions, faults,
                    require_declared_valid=False,
                    include_phase_failures=False)
                label = phase["label"]
                sequences = by_label.get(label, [])
                phase_validity = phase["end"].get("valid")
                declared_failures = set(
                    phase["end"].get("quality_failures", []))
                phase_status = phase["end"].get("status")
                tier = None
                if phase_validity is True and not declared_failures:
                    tier = "verified"
                elif (phase_validity is False and phase_status == "complete"
                      and declared_failures
                      and declared_failures <= {"speed_median", "speed_p95"}):
                    tier = "speed_target_mismatch"
                elif phase_validity is None and phase_status == "complete":
                    tier = "unverified"
                else:
                    failures.append("phase_experiment_gate_failed")
                if phase_counts[label] != 1:
                    failures.append("phase_label_not_unique")
                if not sequences:
                    failures.append("no_recursive_length_aligned_sequence")
                if _split_for_run(run_id) == "excluded":
                    failures.append("excluded_by_existing_replay_split_policy")
                if (tier == "speed_target_mismatch"
                        and _split_for_run(run_id) != "train"):
                    failures.append("speed_mismatch_data_must_not_be_holdout")

                phase_samples = 0
                sequence_rejections: list[str] = []
                sequence_frames = []
                for sequence in sequences:
                    (_, frames, sensors, sensor_valid, attitude,
                     attitude_valid, dt, packet_sequence, times, odom_pose,
                     simulator_pose, lap_count, simulator_rigid_state,
                     simulator_acceleration) = sequence
                    if len(frames) < MIN_SEQUENCE_SAMPLES:
                        sequence_rejections.append(
                            "sequence_shorter_than_history_plus_rollout")
                        continue
                    if (not np.isfinite(frames).all()
                            or not np.isfinite(sensors).all()
                            or not np.isfinite(dt).all()):
                        sequence_rejections.append("nonfinite_plant_features")
                        continue
                    digest = _sequence_fingerprint(frames, dt)
                    if digest in seen_hashes:
                        sequence_rejections.append(
                            "exact_duplicate_of_existing_or_salvaged_sequence")
                        continue
                    if failures:
                        continue
                    seen_hashes.add(digest)
                    phase_samples += len(frames)
                    sequence_frames.append(frames)
                    accepted[tier].append((
                        run_id, _split_for_run(run_id), label, tier, frames,
                        sensors, sensor_valid, attitude, attitude_valid, dt,
                        packet_sequence, times, odom_pose, simulator_pose,
                        lap_count, simulator_rigid_state,
                        simulator_acceleration))
                    run_report["salvaged_sequences"] += 1
                    run_report["salvaged_samples"] += len(frames)

                combined_frames = (np.concatenate(sequence_frames)
                                   if sequence_frames else None)
                run_report["phases"].append({
                    "phase_index": phase["index"], "label": label,
                    "status": ("salvaged" if phase_samples else
                               "rejected"),
                    "quality_tier": tier,
                    "sequence_count": len(sequences),
                    "samples_salvaged": phase_samples,
                    "quality": quality,
                    "reasons": sorted(set(failures + sequence_rejections)),
                    "command_feedback": {
                        "interpretation": (
                            "both command and measured actuator channels are "
                            "retained; actuator lag is plant behavior, not a "
                            "reason to reject the phase"),
                        "steering_error_p95_rad": quality.get(
                            "steering_error_p95_rad"),
                        "throttle_start_feedback_error": phase["end"].get(
                            "start_throttle_feedback_abs_error_norm"),
                        "throttle_end_feedback_error": phase["end"].get(
                            "end_throttle_feedback_abs_error_norm"),
                        "explicit_throttle_tracking_gate": phase["end"].get(
                            "throttle_feedback_tracking_ok"),
                        "explicit_steering_tracking_gate": phase["end"].get(
                            "steering_feedback_tracking_ok"),
                        **(_command_feedback_errors(combined_frames)
                           if combined_frames is not None else {
                               "steering_command_feedback_abs_error_p95_rad": None,
                               "throttle_command_feedback_abs_error_p95_norm": None,
                           }),
                    },
                })
        except Exception as exc:
            errors.append({"run_id": run_id,
                           "error": f"{type(exc).__name__}: {exc}"})
            run_report["error"] = f"{type(exc).__name__}: {exc}"
        run_report["source_disposition"] = (
            "retain_raw_source_for_salvaged_sequences"
            if run_report["salvaged_sequences"] else
            "retain_pending_manual_review_after_read_error"
            if run_report.get("error") else
            "remove_capture_directory_no_usable_sequence")
        run_reports.append(run_report)

    exports = {tier: _write_archive(entries, output_dir, tier)
               for tier, entries in accepted.items()}
    total_sequences = sum(item["sequences"] for item in exports.values())
    captures_with_salvage = len({entry[0] for entries in accepted.values()
                                 for entry in entries})

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve().relative_to(REPO_ROOT)),
        "reference_archive": str(reference_npz.resolve().relative_to(REPO_ROOT)),
        "purpose": "interval-level salvage from runs rejected by whole-capture quality gates; evidence tiers are separate and never silently merged into canonical splits",
        "gates": {
            "verified_tier_requires_phase_declared_valid": True,
            "speed_target_mismatch_tier_requires_complete_phase_and_only_speed_median_or_p95_failure": True,
            "unverified_tier_requires_complete_phase_or_bounded_unphased_segment_and_preserves_whole_run_split": True,
            "zero_collision_count_through_phase": True,
            "zero_bridge_timing_faults_in_phase": True,
            "stream_rate_hz_min": 38.0,
            "stream_p95_gap_ms_max": 35.0,
            "sensor_stream_max_gap_ms": 60.0,
            "command_stream_max_gap_ms": 120.0,
            "minimum_sequence_samples": MIN_SEQUENCE_SAMPLES,
            "feature_and_dt_finite": True,
            "exact_sequence_duplicates_removed": True,
            "actuator_tracking": "existing phase-level validation fields are preserved; command and measured feedback are both included; physical actuator lag is retained as plant behavior",
        },
        "summary": {
            "candidate_captures": len(candidates),
            "capture_read_errors": len(errors),
            "phase_or_segment_intervals": sum(
                len(run["phases"]) for run in run_reports),
            "salvaged_intervals": sum(
                phase["status"] == "salvaged"
                for run in run_reports for phase in run["phases"]),
            "rejected_intervals": sum(
                phase["status"] == "rejected"
                for run in run_reports for phase in run["phases"]),
            "salvaged_sequences": total_sequences,
            "salvaged_samples": sum(item["samples"] for item in exports.values()),
            "salvaged_sequences_by_tier": {
                tier: item["sequences"] for tier, item in exports.items()},
            "captures_with_salvage": captures_with_salvage,
            "captures_with_no_salvage": len(candidates) - captures_with_salvage,
        },
        "export_by_quality_tier": exports,
        "cleanup": {
            "capture_directories_to_remove": [
                {"run_id": run["run_id"], "directory": str(
                    (REPO_ROOT / run["bag"]).parents[1].relative_to(REPO_ROOT)),
                 "reason": "no interval passed the declared phase/segment gates"}
                for run in run_reports
                if run["source_disposition"] ==
                "remove_capture_directory_no_usable_sequence"],
            "source_bags_to_retain": [run["run_id"] for run in run_reports
                                      if run["source_disposition"] ==
                                      "retain_raw_source_for_salvaged_sequences"],
            "read_errors_to_review": [run["run_id"] for run in run_reports
                                      if run["source_disposition"] ==
                                      "retain_pending_manual_review_after_read_error"],
        },
        "errors": errors,
        "runs": run_reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = audit(args.manifest, args.reference_npz, args.output_dir)
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(f"phase salvage failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": report["summary"],
                      "export_by_quality_tier": report["export_by_quality_tier"],
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
