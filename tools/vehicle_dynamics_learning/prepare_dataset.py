#!/usr/bin/env python3
"""Build a run-grouped, sensor-causal dataset from existing Explore bags.

Ground-truth bridge odometry is used only as an offline supervised target and
for the oracle-plant benchmark. Separate causal sensor features are exported
for sensor-state estimation. No bag is modified and no simulator/controller
is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import evaluate_open_plane_body_dynamics as body  # noqa: E402


SCHEMA_VERSION = 6
SIMULATOR_DT_S = 0.025
FEATURE_NAMES = (
    "u_rear_mps", "v_rear_mps", "yaw_rate_rps",
    "steering_feedback_rad", "throttle_feedback_norm",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "steering_command_rad", "throttle_command_norm",
)
SENSOR_FEATURE_NAMES = (
    "steering_feedback_rad", "throttle_feedback_norm",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps",
    "steering_command_rad", "throttle_command_norm", "sample_dt_s",
)
ATTITUDE_FEATURE_NAMES = (
    "imu_roll_rad", "imu_pitch_rad",
    "imu_roll_rate_rps", "imu_pitch_rate_rps",
)
PREDICTED_STATE_NAMES = FEATURE_NAMES[:7]
SIMULATOR_RIGID_STATE_NAMES = (
    "position_x_m", "position_y_m", "position_z_m",
    "orientation_x", "orientation_y", "orientation_z", "orientation_w",
    "linear_velocity_x_mps", "linear_velocity_y_mps",
    "linear_velocity_z_mps", "angular_velocity_x_rps",
    "angular_velocity_y_rps", "angular_velocity_z_rps",
)
SIMULATOR_ACCELERATION_NAMES = (
    "linear_acceleration_x_mps2", "linear_acceleration_y_mps2",
    "linear_acceleration_z_mps2",
)
HISTORY_STEPS = 16
ROLLOUT_STEPS = 32
DEFAULT_ROOT = REPO_ROOT / "live_runs"


def _split_for_name(name: str) -> str:
    lowered = name.lower()
    if "source_player" in lowered:
        return "exclude_source_player_mismatch"
    if "replay" in lowered:
        return "exclude_replay"
    if "validation_20260928" in lowered:
        return "validation"
    if "validation_20260929" in lowered:
        return "final_test"
    if "holdout" in lowered:
        return "test"
    return "train"


def _frame(sample: body.MotionSample) -> np.ndarray | None:
    if sample.rear_wheel_surface_mps is None or sample.actuator_history is None:
        return None
    values = np.concatenate((
        np.asarray(sample.state, dtype=np.float64),
        np.asarray(sample.actuators[:2], dtype=np.float64),
        np.asarray(sample.rear_wheel_surface_mps, dtype=np.float64),
        np.asarray(sample.actuator_history[2:4], dtype=np.float64),
    ))
    return values if values.shape == (len(FEATURE_NAMES),) and np.isfinite(values).all() else None


def _sensor_frame(sample: body.MotionSample) -> np.ndarray | None:
    if (sample.rear_wheel_surface_mps is None
            or sample.actuator_history is None
            or sample.imu_acceleration_mps2 is None
            or sample.imu_yaw_rate_rps is None):
        return None
    values = np.concatenate((
        np.asarray(sample.actuators[:2], dtype=np.float64),
        np.asarray(sample.rear_wheel_surface_mps, dtype=np.float64),
        np.asarray(sample.imu_acceleration_mps2, dtype=np.float64),
        np.asarray([sample.imu_yaw_rate_rps], dtype=np.float64),
        np.asarray(sample.actuator_history[2:4], dtype=np.float64),
    ))
    return (values if values.shape == (len(SENSOR_FEATURE_NAMES) - 1,)
            and np.isfinite(values).all() else None)


def _attitude_frame(sample: body.MotionSample) -> np.ndarray | None:
    if (sample.imu_roll_pitch_rad is None
            or sample.imu_roll_pitch_rate_rps is None):
        return None
    values = np.concatenate((sample.imu_roll_pitch_rad,
                             sample.imu_roll_pitch_rate_rps)).astype(
                                 np.float64, copy=False)
    return (values if values.shape == (len(ATTITUDE_FEATURE_NAMES),)
            and np.isfinite(values).all() else None)


def _quality(capture: body.Capture) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if capture.aborted:
        failures.append("experiment_aborted")
    if capture.collision_count_start != 0 or capture.collision_count_end != 0:
        failures.append("collision_count_nonzero")
    if capture.timing_faults:
        failures.append("bridge_timing_fault")
    packet_match_fraction = (
        capture.packet_sequence_matched_samples
        / capture.packet_sequence_total_samples
        if capture.packet_sequence_total_samples else 0.0)
    if packet_match_fraction < 0.999:
        failures.append("packet_sequence_alignment_below_99_9_percent")
    streams = capture.phase_stream_stats or capture.stream_stats
    command_topics = set(body.COMMAND_STREAM_TOPICS)
    for topic in body.STREAM_TOPICS:
        stats = streams.get(topic)
        if stats is None:
            failures.append(f"missing_stream_stats:{topic}")
            continue
        rate, p95_gap, max_gap = stats
        max_allowed = 120.0 if topic in command_topics else 60.0
        if rate < 38.0 or p95_gap > 35.0 or max_gap > max_allowed:
            failures.append(f"stream_quality:{topic}")
    if not capture.sequences:
        failures.append("no_valid_aligned_sequences")
    return not failures, failures


def _fingerprint(sequences: list[tuple[np.ndarray, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for frames, dt in sequences:
        digest.update(np.asarray([len(frames)], dtype="<i8").tobytes())
        digest.update(np.round(frames, 4).astype("<f4").tobytes())
        digest.update(np.round(dt, 3).astype("<f4").tobytes())
    return digest.hexdigest()


def _extract(path: Path, coalesce_contiguous_phases: bool = False,
             practice_active_interval: bool = False,
             include_nonvalid_phases: bool = False) -> tuple[
        dict[str, Any], list[tuple[str, np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray]]]:
    capture = body.load_capture(
        path, include_nonvalid_phases=include_nonvalid_phases)
    run_id = path.parents[1].name
    clean, failures = _quality(capture)
    packet_match_fraction = (
        capture.packet_sequence_matched_samples
        / capture.packet_sequence_total_samples
        if capture.packet_sequence_total_samples else 0.0)
    active_interval = None
    active_interval_report = None
    if practice_active_interval and run_id.startswith("practice_"):
        # The practice recorder can report its expected socket disconnect just
        # after lap 12. Validate the complete active interval independently,
        # then exclude all pre-run/post-run samples without weakening the
        # normal OpenPlane quality gate.
        try:
            try:
                from .evaluate_sensor_observer_practice import _receipt_gates
            except ImportError:
                from evaluate_sensor_observer_practice import _receipt_gates
            active_interval_report = _receipt_gates(path)
            active_interval = (
                int(active_interval_report["active_start_receipt_ns"]),
                int(active_interval_report["lap12_receipt_ns"]))
        except (OSError, ValueError, sqlite3.Error) as exc:
            active_interval_report = {"error": f"{type(exc).__name__}: {exc}"}
        if active_interval is not None:
            clean = True
    split = _split_for_name(run_id)
    extracted: list[tuple[str, np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray, np.ndarray, np.ndarray,
                          np.ndarray]] = []
    fingerprint_sequences: list[tuple[np.ndarray, np.ndarray]] = []

    # load_capture gives each accepted phase 500 ms of pre-phase context.
    # Drop negative phase-time samples so neighboring phase contexts cannot
    # duplicate training rows or leak across a phase boundary.
    for label, sequence in zip(capture.sequence_labels, capture.sequences):
        frames: list[np.ndarray] = []
        sensor_frames: list[np.ndarray] = []
        sensor_valid: list[bool] = []
        attitude_frames: list[np.ndarray] = []
        attitude_valid: list[bool] = []
        packet_sequences: list[int] = []
        receipt_times_ns: list[int] = []
        odom_poses: list[np.ndarray] = []
        simulator_poses: list[np.ndarray] = []
        lap_counts: list[int] = []
        simulator_rigid_states: list[np.ndarray] = []
        simulator_accelerations: list[np.ndarray] = []
        for sample in sequence:
            if sample.time_s < 0.0:
                continue
            if (active_interval is not None
                    and not active_interval[0] <= sample.receipt_ns <= active_interval[1]):
                continue
            row = _frame(sample)
            if row is None:
                continue
            sensor_row = _sensor_frame(sample)
            attitude_row = _attitude_frame(sample)
            frames.append(row)
            sensor_frames.append(
                sensor_row if sensor_row is not None else
                np.zeros(len(SENSOR_FEATURE_NAMES) - 1, dtype=np.float64))
            sensor_valid.append(sensor_row is not None)
            attitude_frames.append(
                attitude_row if attitude_row is not None else
                np.zeros(len(ATTITUDE_FEATURE_NAMES), dtype=np.float64))
            attitude_valid.append(attitude_row is not None)
            packet_sequences.append(int(sample.packet_sequence))
            receipt_times_ns.append(int(sample.receipt_ns))
            odom_pose = sample.pose_xyyaw
            simulator_pose = sample.simulator_pose_xyyaw
            odom_poses.append(
                np.asarray(odom_pose, dtype=np.float64).copy()
                if odom_pose is not None else
                np.full(3, np.nan, dtype=np.float64))
            simulator_poses.append(
                np.asarray(simulator_pose, dtype=np.float64).copy()
                if simulator_pose is not None else
                np.full(3, np.nan, dtype=np.float64))
            lap_counts.append(int(sample.lap_count)
                              if sample.lap_count is not None else -1)
            simulator_rigid_states.append(
                np.asarray(sample.simulator_rigid_state, dtype=np.float64).copy()
                if sample.simulator_rigid_state is not None else
                np.full(len(SIMULATOR_RIGID_STATE_NAMES), np.nan,
                        dtype=np.float64))
            simulator_accelerations.append(
                np.asarray(sample.simulator_linear_acceleration,
                           dtype=np.float64).copy()
                if sample.simulator_linear_acceleration is not None else
                np.full(len(SIMULATOR_ACCELERATION_NAMES), np.nan,
                        dtype=np.float64))
        minimum_phase_samples = (2 if coalesce_contiguous_phases
                                 else HISTORY_STEPS + ROLLOUT_STEPS + 1)
        if len(frames) < minimum_phase_samples:
            continue
        frame_array = np.asarray(frames, dtype=np.float32)
        sensor_array = np.asarray(sensor_frames, dtype=np.float32)
        valid_array = np.asarray(sensor_valid, dtype=bool)
        attitude_array = np.asarray(attitude_frames, dtype=np.float32)
        attitude_valid_array = np.asarray(attitude_valid, dtype=bool)
        packet_sequence_array = np.asarray(packet_sequences, dtype=np.int64)
        receipt_time_array = np.asarray(receipt_times_ns, dtype=np.int64)
        odom_pose_array = np.asarray(odom_poses, dtype=np.float32)
        simulator_pose_array = np.asarray(simulator_poses, dtype=np.float32)
        lap_count_array = np.asarray(lap_counts, dtype=np.int32)
        simulator_rigid_state_array = np.asarray(
            simulator_rigid_states, dtype=np.float32)
        simulator_acceleration_array = np.asarray(
            simulator_accelerations, dtype=np.float32)
        if not np.isfinite(frame_array).all():
            continue
        # Simulator packet identity defines continuity. Receipt timestamps are
        # retained for event/sensor joins, never interpreted as physics dt.
        starts: list[int] = []
        ends: list[int] = []
        segment_start: int | None = None
        for index, packet_id in enumerate(packet_sequence_array):
            if packet_id < 0:
                if segment_start is not None:
                    starts.append(segment_start)
                    ends.append(index)
                    segment_start = None
                continue
            if segment_start is None:
                segment_start = index
            elif packet_id != packet_sequence_array[index - 1] + 1:
                starts.append(segment_start)
                ends.append(index)
                segment_start = index
        if segment_start is not None:
            starts.append(segment_start)
            ends.append(len(packet_sequence_array))
        minimum_segment_samples = (2 if coalesce_contiguous_phases
                                   else HISTORY_STEPS + ROLLOUT_STEPS + 1)
        for start, end in zip(starts, ends):
            if end - start < minimum_segment_samples:
                continue
            local_frames = frame_array[start:end]
            local_sequences = packet_sequence_array[start:end]
            local_dt = np.full(end - start, SIMULATOR_DT_S, dtype=np.float32)
            local_sensor = np.column_stack((sensor_array[start:end], local_dt))
            local_valid = valid_array[start:end]
            local_attitude = attitude_array[start:end]
            local_attitude_valid = attitude_valid_array[start:end]
            local_receipt_times = receipt_time_array[start:end]
            local_odom_pose = odom_pose_array[start:end]
            local_simulator_pose = simulator_pose_array[start:end]
            local_lap_count = lap_count_array[start:end]
            local_simulator_rigid_state = simulator_rigid_state_array[start:end]
            local_simulator_acceleration = simulator_acceleration_array[start:end]
            extracted.append((label, local_frames, local_sensor, local_valid,
                              local_attitude, local_attitude_valid, local_dt,
                              local_sequences, local_receipt_times,
                              local_odom_pose, local_simulator_pose,
                              local_lap_count, local_simulator_rigid_state,
                              local_simulator_acceleration))
            fingerprint_sequences.append((
                np.column_stack((local_frames, local_sensor,
                                 local_valid.astype(np.float32), local_attitude,
                                 local_attitude_valid.astype(np.float32))), local_dt))

    values = np.concatenate([item[1] for item in extracted], axis=0) if extracted else np.empty((0, len(FEATURE_NAMES)))
    observer_values = np.concatenate([item[2] for item in extracted], axis=0) if extracted else np.empty((0, len(SENSOR_FEATURE_NAMES)))
    observer_valid = np.concatenate([item[3] for item in extracted], axis=0) if extracted else np.empty((0,), dtype=bool)
    attitude_values = np.concatenate([item[4] for item in extracted], axis=0) if extracted else np.empty((0, len(ATTITUDE_FEATURE_NAMES)))
    attitude_valid_rows = np.concatenate([item[5] for item in extracted], axis=0) if extracted else np.empty((0,), dtype=bool)
    simulator_pose_values = (
        np.concatenate([item[10] for item in extracted], axis=0)
        if extracted else np.empty((0, 3), dtype=np.float32))
    simulator_pose_valid = np.isfinite(simulator_pose_values).all(axis=1)
    simulator_rigid_values = (
        np.concatenate([item[12] for item in extracted], axis=0)
        if extracted else np.empty(
            (0, len(SIMULATOR_RIGID_STATE_NAMES)), dtype=np.float32))
    simulator_rigid_valid = np.isfinite(simulator_rigid_values).all(axis=1)
    simulator_acceleration_values = (
        np.concatenate([item[13] for item in extracted], axis=0)
        if extracted else np.empty(
            (0, len(SIMULATOR_ACCELERATION_NAMES)), dtype=np.float32))
    simulator_acceleration_valid = np.isfinite(
        simulator_acceleration_values).all(axis=1)
    stream_stats = capture.phase_stream_stats or capture.stream_stats
    record: dict[str, Any] = {
        "run_id": run_id,
        "bag": (str(path.resolve().relative_to(REPO_ROOT))
                if path.resolve().is_relative_to(REPO_ROOT) else str(path.resolve())),
        "bytes": path.stat().st_size,
        "suggested_split": split,
        "clean_stream_and_collision_gate": clean,
        "whole_bag_quality_failures": failures,
        "quality_gate_scope": ("complete_lap_0_to_12_active_interval"
                               if active_interval is not None else "whole_bag"),
        "practice_active_interval_validation": active_interval_report,
        "active_interval_receipt_ns": list(active_interval)
            if active_interval is not None else None,
        "quality_failures": failures,
        "aborted": capture.aborted,
        "reason": capture.reason,
        "valid_phases": capture.valid_phase_count,
        "invalid_phases": capture.invalid_phase_count,
        "unscored_phases": capture.unscored_phase_count,
        "collisions": [capture.collision_count_start, capture.collision_count_end],
        "timing_faults": capture.timing_faults,
        "packet_sequence_alignment": {
            "matched_samples": capture.packet_sequence_matched_samples,
            "total_samples": capture.packet_sequence_total_samples,
            "match_fraction": packet_match_fraction,
        },
        "simulator_pose_valid_samples": int(np.count_nonzero(simulator_pose_valid)),
        "simulator_pose_valid_fraction": (
            float(np.mean(simulator_pose_valid)) if len(simulator_pose_valid) else 0.0),
        "simulator_rigid_state_valid_samples": int(
            np.count_nonzero(simulator_rigid_valid)),
        "simulator_rigid_state_valid_fraction": (
            float(np.mean(simulator_rigid_valid))
            if len(simulator_rigid_valid) else 0.0),
        "simulator_acceleration_valid_fraction": (
            float(np.mean(simulator_acceleration_valid))
            if len(simulator_acceleration_valid) else 0.0),
        "sequences_exported": len(extracted),
        "samples_exported": int(len(values)),
        "fingerprint": _fingerprint(fingerprint_sequences) if extracted else None,
        "feature_min": values.min(axis=0).tolist() if len(values) else None,
        "feature_max": values.max(axis=0).tolist() if len(values) else None,
        "feature_names": list(FEATURE_NAMES),
        "sensor_feature_names": list(SENSOR_FEATURE_NAMES),
        "attitude_feature_names": list(ATTITUDE_FEATURE_NAMES),
        "attitude_valid_samples": int(np.count_nonzero(attitude_valid_rows)),
        "attitude_valid_fraction": (float(np.mean(attitude_valid_rows))
                                    if len(attitude_valid_rows) else 0.0),
        "sensor_valid_samples": int(np.count_nonzero(observer_valid)),
        "sensor_valid_fraction": (float(np.mean(observer_valid))
                                  if len(observer_valid) else 0.0),
        "sensor_feature_min": observer_values[observer_valid].min(axis=0).tolist()
        if np.any(observer_valid) else None,
        "sensor_feature_max": observer_values[observer_valid].max(axis=0).tolist()
        if np.any(observer_valid) else None,
        "streams": {
            name: {"hz": float(stats[0]), "gap_p95_ms": float(stats[1]),
                   "gap_max_ms": float(stats[2])}
            for name, stats in stream_stats.items()
        },
    }
    return record, extracted


def _coalesce_contiguous_sequences(
    sequences: list[tuple[str, np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray]],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                np.ndarray, np.ndarray]]:
    """Join fragments only when simulator packet IDs are consecutive.

    Phase and lap markers are experiment bookkeeping, not physical resets.
    Receipt timestamps remain for event association only. Missing or duplicate
    packet IDs break continuity; each retained transition has exactly 25 ms.
    """
    rows: list[tuple[int, int, str, np.ndarray, np.ndarray, bool,
                     np.ndarray, bool, np.ndarray, np.ndarray, int,
                     np.ndarray, np.ndarray]] = []
    for (label, frames, sensors, sensor_valid, attitude, attitude_valid,
         _, packet_sequences, receipt_times, odom_poses, simulator_poses,
         lap_counts, simulator_rigid_states,
         simulator_accelerations) in sequences:
        rows.extend(
            (int(packet_id), int(receipt_ns), label, frame, sensor,
             bool(sensor_ok), attitude_row, bool(attitude_ok), odom_pose,
             simulator_pose, int(lap_count), simulator_rigid_state,
             simulator_acceleration)
            for (packet_id, receipt_ns, frame, sensor, sensor_ok,
                 attitude_row, attitude_ok, odom_pose, simulator_pose,
                 lap_count, simulator_rigid_state,
                 simulator_acceleration) in zip(
                     packet_sequences, receipt_times, frames, sensors,
                     sensor_valid, attitude, attitude_valid, odom_poses,
                     simulator_poses, lap_counts, simulator_rigid_states,
                     simulator_accelerations)
        )
    rows.sort(key=lambda row: (row[0], row[1]))

    deduplicated: list[tuple[int, int, str, np.ndarray, np.ndarray, bool,
                             np.ndarray, bool, np.ndarray, np.ndarray, int,
                             np.ndarray, np.ndarray]] = []
    for row in rows:
        if row[0] < 0 or (deduplicated and row[0] == deduplicated[-1][0]):
            continue
        deduplicated.append(row)

    result = []
    current = []

    def flush() -> None:
        if len(current) < 2:
            current.clear()
            return
        dt = np.full(len(current), SIMULATOR_DT_S, dtype=np.float32)
        result.append((
            "continuous_run",
            np.stack([row[3] for row in current]).astype(np.float32, copy=False),
            np.stack([row[4] for row in current]).astype(np.float32, copy=False),
            np.asarray([row[5] for row in current], dtype=bool),
            np.stack([row[6] for row in current]).astype(np.float32, copy=False),
            np.asarray([row[7] for row in current], dtype=bool),
            dt,
            np.asarray([row[0] for row in current], dtype=np.int64),
            np.asarray([row[1] for row in current], dtype=np.int64),
            np.stack([row[8] for row in current]).astype(np.float32, copy=False),
            np.stack([row[9] for row in current]).astype(np.float32, copy=False),
            np.asarray([row[10] for row in current], dtype=np.int32),
            np.stack([row[11] for row in current]).astype(np.float32, copy=False),
            np.stack([row[12] for row in current]).astype(np.float32, copy=False),
        ))
        current.clear()

    for row in deduplicated:
        if current and row[0] != current[-1][0] + 1:
            flush()
        current.append(row)
    flush()
    return result


def _effective_splits(records: list[dict[str, Any]]) -> None:
    # Exact/replay-equivalent datasets are never allowed to appear on opposite
    # sides of a train/test split. Keep the most conservative role for a group.
    priority = {"train": 0, "validation": 1, "test": 2, "final_test": 3}
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        fingerprint = record.get("fingerprint")
        if fingerprint and not record["suggested_split"].startswith("exclude_"):
            groups.setdefault(fingerprint, []).append(record)
    for fingerprint, group in groups.items():
        if len(group) < 2:
            continue
        best = max((r["suggested_split"] for r in group),
                   key=lambda split: priority.get(split, -1))
        for record in group:
            record["effective_split"] = best
            record["duplicate_group_size"] = len(group)
            record["duplicate_group_fingerprint"] = fingerprint
    for record in records:
        record.setdefault("effective_split", record["suggested_split"])
        record.setdefault("duplicate_group_size", 1)


def prepare(root: Path, output_dir: Path, explicit_bags: list[Path],
            coalesce_contiguous_phases: bool = False,
            additional_bags: list[Path] | None = None,
            split_overrides: dict[str, str] | None = None,
            practice_active_interval: bool = False) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bags = (list(explicit_bags) if explicit_bags else
            list(root.glob("openplane*/run/run_0.db3")))
    bags.extend(additional_bags or [])
    bags = sorted(set(bags))
    if not bags:
        raise ValueError(f"no openplane bags found under {root}")

    records: list[dict[str, Any]] = []
    extracted_by_run: dict[str, list[tuple]] = {}
    errors: list[dict[str, str]] = []
    for index, path in enumerate(bags, start=1):
        if not path.is_file():
            errors.append({"bag": str(path), "error": "file_not_found"})
            continue
        run_id = path.parents[1].name
        print(f"[{index}/{len(bags)}] reading {run_id}", flush=True)
        try:
            record, sequences = _extract(
                path, coalesce_contiguous_phases, practice_active_interval)
            override = (split_overrides or {}).get(run_id)
            if override is not None:
                record["suggested_split"] = override
                record["explicit_split_override"] = override
        except Exception as exc:  # one malformed/partial run must not hide the rest
            errors.append({"bag": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append(record)
        extracted_by_run[run_id] = sequences

    _effective_splits(records)
    run_index = {record["run_id"]: index for index, record in enumerate(records)}
    frame_blocks: list[np.ndarray] = []
    sensor_blocks: list[np.ndarray] = []
    sensor_valid_blocks: list[np.ndarray] = []
    attitude_blocks: list[np.ndarray] = []
    attitude_valid_blocks: list[np.ndarray] = []
    dt_blocks: list[np.ndarray] = []
    packet_sequence_blocks: list[np.ndarray] = []
    odom_pose_blocks: list[np.ndarray] = []
    simulator_pose_blocks: list[np.ndarray] = []
    lap_count_blocks: list[np.ndarray] = []
    simulator_rigid_state_blocks: list[np.ndarray] = []
    simulator_acceleration_blocks: list[np.ndarray] = []
    frame_run_blocks: list[np.ndarray] = []
    bounds: list[tuple[int, int]] = []
    sequence_run: list[int] = []
    sequence_labels: list[str] = []
    sample_time_ns_blocks: list[np.ndarray] = []
    cursor = 0
    used_runs: set[str] = set()
    for record in records:
        run_id = record["run_id"]
        split = record["effective_split"]
        if split.startswith("exclude_") or not record["clean_stream_and_collision_gate"]:
            continue
        run_sequences = extracted_by_run.get(run_id, [])
        if not run_sequences:
            continue
        if coalesce_contiguous_phases:
            run_sequences = _coalesce_contiguous_sequences(run_sequences)
            if not run_sequences:
                continue
        used_runs.add(run_id)
        for (label, frames, sensor_frames, sensor_valid, attitude_frames,
             attitude_valid, dt, packet_sequence, sample_time_ns,
             odom_pose, simulator_pose, lap_count, simulator_rigid_state,
             simulator_acceleration) in run_sequences:
            frame_blocks.append(frames)
            sensor_blocks.append(sensor_frames)
            sensor_valid_blocks.append(sensor_valid)
            attitude_blocks.append(attitude_frames)
            attitude_valid_blocks.append(attitude_valid)
            dt_blocks.append(dt)
            packet_sequence_blocks.append(packet_sequence)
            sample_time_ns_blocks.append(sample_time_ns)
            odom_pose_blocks.append(odom_pose)
            simulator_pose_blocks.append(simulator_pose)
            lap_count_blocks.append(lap_count)
            simulator_rigid_state_blocks.append(simulator_rigid_state)
            simulator_acceleration_blocks.append(simulator_acceleration)
            frame_run_blocks.append(np.full(len(frames), run_index[run_id], dtype=np.int32))
            end = cursor + len(frames)
            bounds.append((cursor, end))
            sequence_run.append(run_index[run_id])
            sequence_labels.append(label)
            cursor = end
    if not frame_blocks:
        raise ValueError("no clean, eligible sequences were exported")

    frames = np.concatenate(frame_blocks, axis=0).astype(np.float32, copy=False)
    sensor_frames = np.concatenate(sensor_blocks, axis=0).astype(np.float32, copy=False)
    sensor_valid = np.concatenate(sensor_valid_blocks, axis=0).astype(bool, copy=False)
    attitude_frames = np.concatenate(attitude_blocks, axis=0).astype(np.float32, copy=False)
    attitude_valid = np.concatenate(attitude_valid_blocks, axis=0).astype(bool, copy=False)
    dt_s = np.concatenate(dt_blocks, axis=0).astype(np.float32, copy=False)
    packet_sequence = np.concatenate(packet_sequence_blocks).astype(
        np.int64, copy=False)
    sample_time_ns = np.concatenate(sample_time_ns_blocks).astype(
        np.int64, copy=False)
    odom_pose_xyyaw = np.concatenate(odom_pose_blocks).astype(
        np.float32, copy=False)
    simulator_pose_xyyaw = np.concatenate(simulator_pose_blocks).astype(
        np.float32, copy=False)
    lap_count = np.concatenate(lap_count_blocks).astype(np.int32, copy=False)
    simulator_rigid_state = np.concatenate(
        simulator_rigid_state_blocks).astype(np.float32, copy=False)
    simulator_linear_acceleration = np.concatenate(
        simulator_acceleration_blocks).astype(np.float32, copy=False)
    frame_run_index = np.concatenate(frame_run_blocks)
    splits = np.asarray([r["effective_split"] for r in records], dtype="U32")
    run_ids = np.asarray([r["run_id"] for r in records], dtype="U128")
    np.savez_compressed(
        output_dir / "openplane_dynamics.npz",
        schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U64"),
        sensor_feature_names=np.asarray(SENSOR_FEATURE_NAMES, dtype="U64"),
        attitude_feature_names=np.asarray(ATTITUDE_FEATURE_NAMES, dtype="U64"),
        predicted_state_names=np.asarray(PREDICTED_STATE_NAMES, dtype="U64"),
        frames=frames,
        sensor_frames=sensor_frames,
        sensor_valid=sensor_valid,
        imu_attitude_frames=attitude_frames,
        imu_attitude_valid=attitude_valid,
        dt_s=dt_s,
        packet_sequence=packet_sequence,
        sample_time_ns=sample_time_ns,
        odom_pose_xyyaw=odom_pose_xyyaw,
        simulator_pose_xyyaw=simulator_pose_xyyaw,
        lap_count=lap_count,
        simulator_rigid_state=simulator_rigid_state,
        simulator_linear_acceleration=simulator_linear_acceleration,
        sequence_bounds=np.asarray(bounds, dtype=np.int64),
        sequence_run_index=np.asarray(sequence_run, dtype=np.int32),
        sequence_labels=np.asarray(sequence_labels, dtype="U256"),
        run_ids=run_ids,
        run_splits=splits,
    )
    train_mask = splits[frame_run_index] == "train"
    train_frames = frames[train_mask]
    if len(train_frames):
        speed = np.hypot(train_frames[:, 0], train_frames[:, 1])
        abs_steer = np.abs(train_frames[:, 3])
        speed_edges = np.arange(0.0, 22.0, 1.0)
        steer_edges = np.asarray([-0.524, -0.42, -0.30, -0.20, -0.10,
                                  0.0, 0.10, 0.20, 0.30, 0.42, 0.524])
        throttle_edges = np.arange(0.0, 1.05, 0.05)
        speed_bin = np.clip(np.digitize(speed, speed_edges[1:], right=False),
                            0, len(speed_edges) - 1)
        steer_bin = np.clip(np.digitize(train_frames[:, 3], steer_edges[1:-1],
                                        right=False), 0, len(steer_edges) - 2)
        throttle_bin = np.clip(np.digitize(train_frames[:, 4], throttle_edges[1:],
                                           right=False), 0, len(throttle_edges) - 1)
        train_run_index = frame_run_index[train_mask]
        observed: dict[tuple[int, int, int], dict[str, Any]] = {}
        for i, key in enumerate(zip(speed_bin, steer_bin, throttle_bin)):
            cell = observed.setdefault(tuple(map(int, key)),
                                       {"samples": 0, "runs": set()})
            cell["samples"] += 1
            cell["runs"].add(int(train_run_index[i]))
        coverage = {
            "train_samples": int(len(train_frames)),
            "speed_mps_p01_p50_p99_max": [float(np.quantile(speed, q))
                                           for q in (0.01, 0.50, 0.99)] + [float(speed.max())],
            "abs_steering_rad_p01_p50_p99_max": [float(np.quantile(abs_steer, q))
                                                 for q in (0.01, 0.50, 0.99)] + [float(abs_steer.max())],
            "throttle_feedback_p01_p50_p99_max": [float(np.quantile(train_frames[:, 4], q))
                                                   for q in (0.01, 0.50, 0.99)] + [float(train_frames[:, 4].max())],
            "grid": {
                "speed_edges_mps": speed_edges.tolist() + ["21+"],
                "signed_steering_edges_rad": steer_edges.tolist(),
                "throttle_edges_norm": throttle_edges.tolist() + ["1.0+"],
                "total_cells": int((len(speed_edges)) * (len(steer_edges) - 1) * len(throttle_edges)),
                "observed_cells": len(observed),
                "cells_with_at_least_100_samples_and_2_runs": sum(
                    value["samples"] >= 100 and len(value["runs"]) >= 2
                    for value in observed.values()),
                "cells": {
                    f"speed_{key[0]}__steer_{key[1]}__throttle_{key[2]}": {
                        "samples": int(value["samples"]),
                        "runs": len(value["runs"]),
                    }
                    for key, value in sorted(observed.items())
                },
            },
        }
    else:
        coverage = {"train_samples": 0}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_source": ("offline bridge /autodrive/roboracer_1/odom twist; "
                         "simulator pose from same-packet bridge diagnostics "
                         "is stored separately for scoring only"),
        "oracle_plant_features": list(FEATURE_NAMES),
            "sensor_estimator_inputs": list(SENSOR_FEATURE_NAMES),
            "offline_attitude_conditioning_inputs": list(ATTITUDE_FEATURE_NAMES),
        "sensor_estimator_target": ["u_rear_mps", "v_rear_mps", "yaw_rate_rps"],
        "predicted_state": list(PREDICTED_STATE_NAMES),
        "simulator_rigid_state_names": list(SIMULATOR_RIGID_STATE_NAMES),
        "simulator_linear_acceleration_names": list(
            SIMULATOR_ACCELERATION_NAMES),
        "selection": ("openplane* bags and requested additional bags; valid phase samples only; nonnegative phase time; "
                      + ("contiguous phases/laps joined by simulator packet sequence; a missing packet breaks a sequence; "
                         "receipt timestamps are never interpreted as physics dt"
                         if coalesce_contiguous_phases else
                         "phase/lap segments retained separately")
                      ),
        "plant_continuity_mode": bool(coalesce_contiguous_phases),
        "practice_active_interval_mode": bool(practice_active_interval),
        "split_policy": "whole-run split; validation_20260928=validation; validation_20260929=final_test; named holdouts=test; exact fingerprint collisions take the most conservative split",
        "stream_gate": ">=38 Hz, p95 gap<=35 ms, sensor max gap<=60 ms, command max gap<=120 ms, zero collision count and zero bridge timing faults",
        "feature_names": list(FEATURE_NAMES),
        "packet_sequence_policy": (
            "Each /odom source stamp is joined exactly to bridge_packet_timing; "
            "only consecutive packet_sequence values remain in a sequence. "
            "Every sample interval is the fixed simulator dt=0.025 s. "
            "Receipt/request timestamps are association and diagnostics only."),
        "pose_label_policy": (
            "odom_pose_xyyaw is the bridge /odom pose; "
            "simulator_pose_xyyaw is same-packet simulator position and yaw "
            "from bridge_packet_timing, retained only as an offline score label. "
            "Neither pose label is a model input."),
        "rigid_state_label_policy": (
            "simulator_rigid_state and simulator_linear_acceleration are "
            "same-packet bridge diagnostics. They are offline teacher labels "
            "only; future values are not plant inputs."),
        "simulator_dt_s": SIMULATOR_DT_S,
        "history_steps": HISTORY_STEPS,
        "rollout_steps": ROLLOUT_STEPS,
        "export": {
            "file": "openplane_dynamics.npz",
            "samples": int(len(frames)),
            "sequences": int(len(bounds)),
            "runs_in_archive": int(len(records)),
            "runs_with_exported_sequences": sorted(used_runs),
            "compressed_bytes": (output_dir / "openplane_dynamics.npz").stat().st_size,
        },
        "training_coverage": coverage,
        "runs": records,
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="folder containing openplane* run directories")
    parser.add_argument("--bag", type=Path, action="append", default=[],
                        help="process only this bag; can be repeated")
    parser.add_argument("--additional-bag", type=Path, action="append", default=[],
                        help="add this bag to the default OpenPlane set or explicit --bag set")
    parser.add_argument("--split-override", action="append", default=[],
                        metavar="RUN_ID=SPLIT",
                        help="set one run's effective starting split: train, validation, test, or final_test")
    parser.add_argument("--coalesce-contiguous-phases", action="store_true",
                        help="for plant identification, join phase/lap fragments by recorded time and break only at data gaps")
    parser.add_argument("--practice-active-interval", action="store_true",
                        help="allow only 12-lap practice captures that pass lap, collision, active 40 Hz and post-run fault checks; crop to active laps")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new or empty directory for compressed dataset and manifest")
    args = parser.parse_args()
    split_overrides = {}
    for value in args.split_override:
        if "=" not in value:
            parser.error("--split-override must be RUN_ID=SPLIT")
        run_id, split = value.split("=", 1)
        if not run_id or split not in ("train", "validation", "test", "final_test"):
            parser.error("split override needs a run ID and train/validation/test/final_test")
        if run_id in split_overrides:
            parser.error(f"duplicate split override for {run_id}")
        split_overrides[run_id] = split
    try:
        manifest = prepare(args.root, args.output_dir, args.bag,
                           args.coalesce_contiguous_phases,
                           args.additional_bag, split_overrides,
                           args.practice_active_interval)
    except (OSError, ValueError) as exc:
        print(f"dataset preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest["export"], indent=2))
    print(f"wrote {args.output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
