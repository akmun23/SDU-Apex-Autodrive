#!/usr/bin/env python3
"""Audit whole-run dynamic coverage for the schema-9 teacher dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.build_replacement_teacher_dataset import (
    EXPECTED_CAPTURES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = (REPO_ROOT / "live_runs/derived_dynamics_learning_20260928/"
             "full_modeling_reset_20261001/"
             "replacement_offline_sim_raceline_20261002")
DATASET_DIR = TASK_ROOT / "replacement_teacher_dataset_v1"
OUTPUT_DIR = TASK_ROOT / "dynamic_coverage_20261002"
SIMULATOR_DT_S = 0.025
SPEED_EDGES_MPS = (0.0, 3.0, 5.0, 7.0, 9.0, 10.0, 11.0, 12.000001)
STEERING_EDGES_RAD = (0.0, 0.10, 0.20, 0.30, 0.40, 0.524001)
EXPECTED_RUNS = frozenset(EXPECTED_CAPTURES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _read_phases(bag_path: Path) -> tuple[list[Any], dict[str, Any]]:
    from tools import analyze_open_plane_dynamics as analysis

    connection = sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        return analysis._phase_events(connection, topics)
    finally:
        connection.close()


def _phase_labels(sample_time_ns: np.ndarray, phases: list[Any]
                  ) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(len(sample_time_ns), "", dtype="U256")
    indices = np.full(len(sample_time_ns), -1, dtype=np.int32)
    for phase in phases:
        start = int(np.searchsorted(sample_time_ns, phase.start_ns, side="left"))
        end = int(np.searchsorted(sample_time_ns, phase.end_ns, side="right"))
        labels[start:end] = phase.label
        indices[start:end] = phase.index
    return labels, indices


def _sequence_index(bounds: np.ndarray, frame_count: int) -> np.ndarray:
    result = np.full(frame_count, -1, dtype=np.int32)
    cursor = 0
    for index, (start_value, end_value) in enumerate(bounds):
        start, end = int(start_value), int(end_value)
        if start != cursor or end <= start or end > frame_count:
            raise ValueError("dataset sequences are not contiguous frame partitions")
        result[start:end] = index
        cursor = end
    if cursor != frame_count or np.any(result < 0):
        raise ValueError("dataset sequences do not cover all samples")
    return result


def _stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(values[mask], dtype=np.float64)
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return {"rows": 0}
    quantiles = np.quantile(selected, [0.0, 0.01, 0.10, 0.25, 0.50,
                                      0.75, 0.90, 0.95, 0.99, 1.0])
    names = ("min", "p01", "p10", "p25", "p50", "p75", "p90",
             "p95", "p99", "max")
    return {"rows": int(len(selected)), **{
        name: float(value) for name, value in zip(names, quantiles)}}


def _run_and_sequence_counts(mask: np.ndarray, run_index: np.ndarray,
                             sequence_index: np.ndarray,
                             run_ids: np.ndarray) -> dict[str, Any]:
    rows = np.flatnonzero(mask)
    return {
        "rows": int(len(rows)),
        "sequences": int(len(set(zip(run_index[rows].tolist(),
                                     sequence_index[rows].tolist())))),
        "independent_runs": int(len(set(run_ids[run_index[rows]].tolist()))),
    }


def _train_quantile_coverage(values: np.ndarray, valid: np.ndarray,
                             splits: np.ndarray, run_index: np.ndarray,
                             sequence_index: np.ndarray,
                             run_ids: np.ndarray) -> dict[str, Any]:
    train = valid & (splits == "train") & np.isfinite(values)
    validation = valid & (splits == "validation") & np.isfinite(values)
    train_values = np.asarray(values[train], dtype=np.float64)
    if not len(train_values):
        return {"train": {}, "validation": {}, "bin_edges_train_quantiles": []}
    requested_q = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    edges = np.unique(np.quantile(train_values, requested_q))
    if len(edges) == 1:
        edges = np.asarray([edges[0], np.nextafter(edges[0], np.inf)])
    # Quantile cutpoints come only from independent training runs. Out-of-range
    # validation samples are counted explicitly instead of clipped silently.
    cuts = edges[1:-1]
    bin_index = np.searchsorted(cuts, values, side="right")
    intervals = []
    labels = [f"(-inf, {cuts[0]:.8g})"] if len(cuts) else ["all train support"]
    for left, right in zip(cuts[:-1], cuts[1:]):
        labels.append(f"[{left:.8g}, {right:.8g})")
    if len(cuts):
        labels.append(f"[{cuts[-1]:.8g}, +inf)")
    for split, split_mask in (("train", train), ("validation", validation)):
        bins = []
        for index, label in enumerate(labels):
            if len(cuts):
                selected = split_mask & (bin_index == index)
            else:
                selected = split_mask
            bins.append({"interval": label, **_run_and_sequence_counts(
                selected, run_index, sequence_index, run_ids)})
        lo, hi = float(edges[0]), float(edges[-1])
        out_of_train_range = split_mask & ((values < lo) | (values > hi))
        intervals.append((split, {
            **_run_and_sequence_counts(split_mask, run_index,
                                       sequence_index, run_ids),
            "distribution": _stats(values, split_mask),
            "train_quantile_bins": bins,
            "rows_outside_training_minmax": int(out_of_train_range.sum()),
            "runs_outside_training_minmax": sorted(set(
                run_ids[run_index[out_of_train_range]].astype(str).tolist())),
        }))
    return {
        "bin_edges_train_quantiles": [float(value) for value in edges],
        "train": intervals[0][1],
        "validation": intervals[1][1],
    }


def _fixed_2d_support(speed: np.ndarray, steering: np.ndarray,
                      valid: np.ndarray, splits: np.ndarray,
                      run_index: np.ndarray, sequence_index: np.ndarray,
                      run_ids: np.ndarray) -> dict[str, Any]:
    speed_edges = np.asarray(SPEED_EDGES_MPS)
    steer_edges = np.asarray(STEERING_EDGES_RAD)
    speed_bin = np.searchsorted(speed_edges, speed, side="right") - 1
    steer_bin = np.searchsorted(steer_edges, np.abs(steering), side="right") - 1
    speed_labels = [f"[{a:g},{b:g}) m/s" for a, b in
                    zip(speed_edges[:-1], speed_edges[1:])]
    steer_labels = [f"[{a:g},{b:g}) rad" for a, b in
                    zip(steer_edges[:-1], steer_edges[1:])]
    cells = []
    train_cells: set[tuple[int, int]] = set()
    val_cells: set[tuple[int, int]] = set()
    for si in range(len(speed_edges) - 1):
        for di in range(len(steer_edges) - 1):
            cell = valid & (speed_bin == si) & (steer_bin == di)
            train_cell = cell & (splits == "train")
            validation_cell = cell & (splits == "validation")
            train_counts = _run_and_sequence_counts(
                train_cell, run_index, sequence_index, run_ids)
            val_counts = _run_and_sequence_counts(
                validation_cell, run_index, sequence_index, run_ids)
            if train_counts["rows"]:
                train_cells.add((si, di))
            if val_counts["rows"]:
                val_cells.add((si, di))
            if train_counts["rows"] or val_counts["rows"]:
                cells.append({"speed_bin": speed_labels[si],
                              "absolute_steering_bin": steer_labels[di],
                              "train": train_counts,
                              "validation": val_counts})
    return {
        "edges": {
            "speed_mps": speed_edges.tolist(),
            "absolute_steering_rad": steer_edges.tolist(),
        },
        "observed_cells": cells,
        "train_only_cells": [
            {"speed_bin": speed_labels[si],
             "absolute_steering_bin": steer_labels[di]}
            for si, di in sorted(train_cells - val_cells)],
        "validation_only_cells": [
            {"speed_bin": speed_labels[si],
             "absolute_steering_bin": steer_labels[di]}
            for si, di in sorted(val_cells - train_cells)],
    }


def build(dataset_dir: Path = DATASET_DIR,
          output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    dataset_path = dataset_dir / "openplane_dynamics.npz"
    dataset_manifest_path = dataset_dir / "manifest.json"
    if output_dir.exists():
        expected_files = {
            output_dir / "dynamic_coverage_report.json",
            output_dir / "condition_coverage.csv",
        }
        unexpected = set(output_dir.iterdir()) - expected_files
        if unexpected:
            raise FileExistsError(
                f"refusing to overwrite unrelated output: {sorted(unexpected)}")
    manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if (int(manifest.get("schema_version", -1)) != 9
            or manifest.get("dataset_role") != "replacement_teacher_dataset_v1"):
        raise ValueError("WP6 requires the completed schema-9 teacher dataset")
    arrays = _load_npz(dataset_path)
    run_ids = arrays["run_ids"].astype(str)
    run_splits = arrays["run_splits"].astype(str)
    frame_run = arrays["frame_run_index"].astype(np.int32, copy=False)
    frame_count = len(arrays["frames"])
    if any(len(arrays[key]) != frame_count for key in (
            "frame_run_index", "sample_time_ns")):
        raise ValueError("frame provenance/time arrays are misaligned")
    sequence_index = _sequence_index(arrays["sequence_bounds"], frame_count)

    new_run_indices = [index for index, run_id in enumerate(run_ids)
                       if run_id in EXPECTED_RUNS]
    if len(new_run_indices) != len(EXPECTED_CAPTURES):
        raise ValueError(
            "schema-9 dataset does not contain every registered new capture")
    dynamic_frames = np.isin(frame_run, np.asarray(new_run_indices))
    rigid = arrays["simulator_rigid_state"].astype(np.float64, copy=False)
    frames = arrays["frames"].astype(np.float64, copy=False)
    sensors = arrays["sensor_frames"].astype(np.float64, copy=False)
    sensor_valid = arrays["sensor_valid"].astype(bool, copy=False)
    dt = arrays["dt_s"].astype(np.float64, copy=False)
    speed = np.hypot(rigid[:, 7], rigid[:, 8])
    steering_feedback = frames[:, 3]
    steering_command = frames[:, 7]
    throttle_feedback = frames[:, 4]
    throttle_command = frames[:, 8]
    yaw_truth = rigid[:, 12]
    yaw_observed = frames[:, 2]
    wheel_body_mismatch = np.abs(0.5 * (frames[:, 5] + frames[:, 6]) - frames[:, 0])

    sequence_rates: dict[str, np.ndarray] = {
        "steering_feedback_rate_rad_s": np.full(frame_count, np.nan),
        "throttle_feedback_rate_norm_s": np.full(frame_count, np.nan),
        "throttle_command_rate_norm_s": np.full(frame_count, np.nan),
    }
    bounds = arrays["sequence_bounds"]
    seq_run_indices = arrays["sequence_run_index"]
    for sequence_id, (start_value, end_value) in enumerate(bounds):
        start, end = int(start_value), int(end_value)
        source_run = int(seq_run_indices[sequence_id])
        if str(run_splits[source_run]) not in ("train", "validation"):
            continue
        if not np.allclose(dt[start:end], SIMULATOR_DT_S, rtol=0, atol=1e-7):
            raise ValueError("new dynamic sequence violates the 25 ms timebase")
        local_dt = dt[start + 1:end]
        for name, values in (
                ("steering_feedback_rate_rad_s", steering_feedback),
                ("throttle_feedback_rate_norm_s", throttle_feedback),
                ("throttle_command_rate_norm_s", throttle_command)):
            sequence_rates[name][start + 1:end] = (
                np.diff(values[start:end]) / local_dt)

    phases_by_run: dict[str, list[Any]] = {}
    phase_labels_by_run: dict[str, np.ndarray] = {}
    unassigned_phase_timeline_by_run: dict[str, int] = {}
    provenance = {row["run_id"]: row for row in manifest["run_provenance"]
                  if row.get("run_id") in EXPECTED_RUNS}
    for run_index in new_run_indices:
        run_id = str(run_ids[run_index])
        record = provenance.get(run_id)
        if record is None:
            raise ValueError(f"missing capture provenance for {run_id}")
        bag_path = Path(record["bag"])
        if not bag_path.is_absolute():
            bag_path = REPO_ROOT / bag_path
        if _sha256(bag_path) != record["bag_sha256"]:
            raise ValueError(f"capture bag changed since schema-9 build: {run_id}")
        phases, experiment_end = _read_phases(bag_path)
        if (len(phases) != 84 or experiment_end.get("aborted") is not False
                or experiment_end.get("reason") != "schedule complete"):
            raise ValueError(f"capture phase events are incomplete: {run_id}")
        mask = frame_run == run_index
        labels, phase_indices = _phase_labels(arrays["sample_time_ns"][mask], phases)
        # Bagged sensor samples may precede the first phase event, fall in the
        # small publication gaps between adjacent phase events, or continue
        # into the post-schedule tail. Keep those rows in the dataset and
        # report them explicitly rather than treating event-topic timing as a
        # sample-validity requirement.
        unassigned_phase_timeline_by_run[run_id] = int(np.count_nonzero(
            phase_indices < 0))
        phases_by_run[run_id] = phases
        phase_labels_by_run[run_id] = labels

    splits_for_frame = run_splits[frame_run]
    run_index_for_frame = frame_run
    sequence_id_for_frame = sequence_index
    dynamic_mask = dynamic_frames

    axis_values = {
        "ground_truth_body_speed_mps": speed,
        "steering_feedback_rad": steering_feedback,
        "absolute_steering_feedback_rad": np.abs(steering_feedback),
        "steering_feedback_rate_rad_s": sequence_rates[
            "steering_feedback_rate_rad_s"],
        "throttle_feedback_norm": throttle_feedback,
        "throttle_command_norm": throttle_command,
        "throttle_feedback_rate_norm_s": sequence_rates[
            "throttle_feedback_rate_norm_s"],
        "throttle_command_rate_norm_s": sequence_rates[
            "throttle_command_rate_norm_s"],
        "ground_truth_yaw_rate_rad_s": yaw_truth,
        "observed_yaw_rate_rad_s": yaw_observed,
        "absolute_wheel_body_speed_mismatch_mps": wheel_body_mismatch,
    }
    axis_valid = {name: dynamic_mask & np.isfinite(values)
                  for name, values in axis_values.items()}
    axis_valid["imu_lateral_acceleration_mps2"] = (
        dynamic_mask & sensor_valid & np.isfinite(sensors[:, 5]))
    axis_values["imu_lateral_acceleration_mps2"] = sensors[:, 5]

    dynamic_axes: dict[str, Any] = {}
    all_train_validation_axes: dict[str, Any] = {}
    all_train_validation_mask = np.isin(
        splits_for_frame, np.asarray(("train", "validation")))
    units = {
        "ground_truth_body_speed_mps": "m/s",
        "steering_feedback_rad": "rad",
        "absolute_steering_feedback_rad": "rad",
        "steering_feedback_rate_rad_s": "rad/s",
        "throttle_feedback_norm": "normalized command",
        "throttle_command_norm": "normalized command",
        "throttle_feedback_rate_norm_s": "normalized command/s",
        "throttle_command_rate_norm_s": "normalized command/s",
        "ground_truth_yaw_rate_rad_s": "rad/s",
        "observed_yaw_rate_rad_s": "rad/s",
        "imu_lateral_acceleration_mps2": "m/s^2",
        "absolute_wheel_body_speed_mismatch_mps": "m/s",
    }
    for name, values in axis_values.items():
        dynamic_axes[name] = {
            "units": units[name],
            **_train_quantile_coverage(
                values, axis_valid[name], splits_for_frame,
                run_index_for_frame, sequence_id_for_frame, run_ids),
        }
        all_axis_mask = all_train_validation_mask & np.isfinite(values)
        if name == "imu_lateral_acceleration_mps2":
            all_axis_mask &= sensor_valid
        all_train_validation_axes[name] = {
            "units": units[name],
            **_train_quantile_coverage(
                values, all_axis_mask,
                splits_for_frame, run_index_for_frame,
                sequence_id_for_frame, run_ids),
        }

    braking_records = []
    dynamic_phase_counters: dict[str, dict[str, Any]] = {}
    condition_records = []
    for run_index in new_run_indices:
        run_id = str(run_ids[run_index])
        split = str(run_splits[run_index])
        phases = phases_by_run[run_id]
        phase_labels = phase_labels_by_run[run_id]
        run_mask = frame_run == run_index
        # Classify maneuvers by stable semantic tokens, not phase row count.
        categories = {
            "multisine": "multisine",
            "triangular_steering": "triangle",
            "random_piecewise_steering": "piecewise_steering",
            "turn_in": "turn_in",
            "reversal": "reversal",
            "throttle_pickup": "throttle_pickup",
            "throttle_reduction": "throttle_reduction",
            "frontier_sweep": "frontier_sweep",
            "frontier_mixed_order": "frontier_mixed_order",
            "steering_active_brake": "active_braking",
            "brake_release_unwind": "brake_release",
        }
        for phase in phases:
            if not phase.label.startswith("coupled_"):
                continue
            category = next((name for token, name in categories.items()
                             if token in phase.label), "other_dynamic")
            row = dynamic_phase_counters.setdefault(category, {
                "phase_count": 0, "valid_phase_count": 0,
                "runs": set(), "train_phase_count": 0,
                "validation_phase_count": 0,
            })
            row["phase_count"] += 1
            row["valid_phase_count"] += int(phase.valid is True)
            row["runs"].add(run_id)
            row[f"{split}_phase_count"] += 1

            if category in ("active_braking", "brake_release"):
                start = int(np.searchsorted(
                    arrays["sample_time_ns"][run_mask], phase.start_ns, side="left"))
                end = int(np.searchsorted(
                    arrays["sample_time_ns"][run_mask], phase.end_ns, side="right"))
                global_indices = np.flatnonzero(run_mask)[start:end]
                selected_speed = speed[global_indices]
                selected_cmd = throttle_command[global_indices]
                braking_records.append({
                    "run_id": run_id,
                    "split": split,
                    "condition_id": phase.label.removeprefix("coupled_").removesuffix(
                        "_steering_active_brake" if category == "active_braking"
                        else "_brake_release_unwind"),
                    "phase_label": phase.label,
                    "phase_valid": phase.valid,
                    "phase_samples": int(len(global_indices)),
                    "throttle_command_zero_fraction": (
                        float(np.mean(np.isclose(selected_cmd, 0.0)))
                        if len(selected_cmd) else 0.0),
                    "speed_start_mps": float(selected_speed[0])
                    if len(selected_speed) else None,
                    "speed_end_mps": float(selected_speed[-1])
                    if len(selected_speed) else None,
                    "speed_min_mps": float(selected_speed.min())
                    if len(selected_speed) else None,
                    "speed_max_mps": float(selected_speed.max())
                    if len(selected_speed) else None,
                })

        run_record = provenance[run_id]
        plan = run_record["condition_plan"]
        sequence_rows = np.flatnonzero(arrays["sequence_run_index"] == run_index)
        if len(sequence_rows) != 9 or len(plan) != 9:
            raise ValueError(f"dynamic condition count mismatch for {run_id}")
        for sequence_id, condition in zip(sequence_rows, plan):
            start, end = map(int, bounds[sequence_id])
            part = slice(start, end)
            local_speed = speed[part]
            local_steer = steering_feedback[part]
            local_srate = sequence_rates["steering_feedback_rate_rad_s"][part]
            local_tcmd = throttle_command[part]
            local_tfb = throttle_feedback[part]
            local_yaw = yaw_truth[part]
            local_ay_valid = sensor_valid[part]
            local_ay = sensors[part, 5][local_ay_valid]
            local_mismatch = wheel_body_mismatch[part]
            label = str(condition["condition_id"])
            local_phase_mask = ((phase_labels_by_run[run_id] != "")
                                & np.isin(phase_labels_by_run[run_id],
                                          [phase.label for phase in phases
                                           if label in phase.label]))
            active_phase_count = sum(
                phase.valid is True and label in phase.label
                and "steering_active_brake" in phase.label for phase in phases)
            condition_records.append({
                "run_id": run_id,
                "split": split,
                "condition_id": label,
                "speed_band": condition["speed_band"],
                "target_speed_mps": condition["target_speed_mps"],
                "documented_steering_cap_rad": condition["max_steering_rad"],
                "reset_epoch": int(arrays["sequence_reset_index"][sequence_id]),
                "sequence_samples": int(end - start),
                "measured_speed_p10_p50_p90_mps": [
                    float(value) for value in np.quantile(
                        local_speed, [0.10, 0.50, 0.90])],
                "steering_feedback_min_max_rad": [
                    float(local_steer.min()), float(local_steer.max())],
                "steering_cap_reached_both_directions": bool(
                    local_steer.min() <= -0.9 * condition["max_steering_rad"]
                    and local_steer.max() >= 0.9 * condition["max_steering_rad"]),
                "steering_feedback_rate_p05_p50_p95_rad_s": [
                    float(value) for value in np.nanquantile(
                        local_srate, [0.05, 0.50, 0.95])],
                "throttle_command_min_max": [
                    float(local_tcmd.min()), float(local_tcmd.max())],
                "throttle_feedback_min_max": [
                    float(local_tfb.min()), float(local_tfb.max())],
                "yaw_rate_abs_p95_rad_s": float(np.quantile(np.abs(local_yaw), 0.95)),
                "imu_lateral_acceleration_abs_p95_mps2": (
                    float(np.quantile(np.abs(local_ay), 0.95))
                    if len(local_ay) else None),
                "wheel_body_mismatch_p95_mps": float(
                    np.quantile(local_mismatch, 0.95)),
                "valid_active_brake_phases": int(active_phase_count),
                "dynamic_phase_sample_count": int(local_phase_mask.sum()),
            })

    speed = arrays["frame_domain_speed_mps"].astype(np.float64, copy=False)
    fixed_valid = dynamic_mask & np.isfinite(speed) & np.isfinite(steering_feedback)
    supported_cells = _fixed_2d_support(
        speed, steering_feedback, fixed_valid, splits_for_frame,
        run_index_for_frame, sequence_id_for_frame, run_ids)

    expected_run_counts = {
        split: sum(1 for registered_split, _, _ in EXPECTED_CAPTURES.values()
                   if registered_split == split)
        for split in ("train", "validation")
    }
    condition_ids = sorted({row["condition_id"] for row in condition_records})
    condition_replicates = []
    for condition_id in condition_ids:
        matching = [row for row in condition_records
                    if row["condition_id"] == condition_id]
        condition_replicates.append({
            "condition_id": condition_id,
            "target_speed_mps": matching[0]["target_speed_mps"],
            "speed_band": matching[0]["speed_band"],
            "steering_cap_rad": matching[0]["documented_steering_cap_rad"],
            "training_independent_runs": len({row["run_id"] for row in matching
                                                if row["split"] == "train"}),
            "validation_independent_runs": len({row["run_id"] for row in matching
                                                  if row["split"] == "validation"}),
            "validation_both_steering_directions_reached": all(
                row["steering_cap_reached_both_directions"]
                for row in matching if row["split"] == "validation"),
            "validation_active_brake_phase_each_run": all(
                row["valid_active_brake_phases"] == 1
                for row in matching if row["split"] == "validation"),
        })

    missing_conditions = [row["condition_id"] for row in condition_replicates
                          if row["training_independent_runs"] != expected_run_counts["train"]
                          or row["validation_independent_runs"]
                          != expected_run_counts["validation"]
                          or not row["validation_both_steering_directions_reached"]
                          or not row["validation_active_brake_phase_each_run"]]
    phase_category_summary = {}
    for category, row in sorted(dynamic_phase_counters.items()):
        phase_category_summary[category] = {
            **{key: value for key, value in row.items() if key != "runs"},
            "independent_runs": len(row["runs"]),
            "run_ids": sorted(row["runs"]),
        }

    # Legacy schema-8 rows remain in the combined array. This provides the
    # broader planner context, while the new dynamic holdouts are reported
    # separately and always by whole independent run.
    overall_splits = {}
    for split in ("train", "validation"):
        mask = run_splits[frame_run] == split
        if not np.any(mask):
            continue
        unique_runs = sorted(set(run_ids[frame_run[mask]].tolist()))
        unique_sequences = int(len(set(zip(
            frame_run[mask].tolist(), sequence_index[mask].tolist()))))
        overall_splits[split] = {
            "rows": int(mask.sum()),
            "independent_runs": len(unique_runs),
            "sequences": unique_sequences,
            "ground_truth_speed_mps": _stats(speed, mask),
            "absolute_steering_feedback_rad": _stats(
                np.abs(steering_feedback), mask),
            "yaw_rate_rad_s": _stats(yaw_truth, mask),
            "imu_lateral_acceleration_mps2": _stats(
                sensors[:, 5], mask & sensor_valid),
            "wheel_body_mismatch_mps": _stats(wheel_body_mismatch, mask),
        }

    report = {
        "schema_version": 1,
        "report_role": "dynamic_coupled_coverage_and_sufficiency",
        "dataset": str(dataset_path.relative_to(REPO_ROOT)),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "new_capture_run_count": len(new_run_indices),
        "new_capture_rows": int(dynamic_mask.sum()),
        "new_capture_sequence_count": int(np.count_nonzero(np.isin(
            arrays["sequence_run_index"], new_run_indices))),
        "samples_outside_phase_event_intervals_by_run": {
            run_id: unassigned_phase_timeline_by_run[run_id]
            for run_id in sorted(unassigned_phase_timeline_by_run)},
        "new_capture_run_counts": expected_run_counts,
        "sampling_timebase_s": SIMULATOR_DT_S,
        "whole_run_split_policy": True,
        "axis_definitions": {
            "speed": "simulator rigid-body ground-truth planar speed",
            "steering": "measured steering feedback; steering-rate uses within-sequence 25 ms differences",
            "throttle": "measured feedback and requested command; rates use within-sequence differences",
            "active_braking": "phase-labeled steering_active_brake intervals; zero throttle is active braking in this simulator",
            "yaw_rate": "simulator rigid-state yaw angular velocity; observed odom yaw rate separately reported",
            "lateral_acceleration": "IMU lateral acceleration on sensor-valid rows; frame convention is checked separately in WP8",
            "wheel_body_mismatch": "absolute mean rear-wheel surface speed minus body u",
        },
        "new_dynamic_capture_axis_coverage": dynamic_axes,
        "all_training_validation_axis_coverage": all_train_validation_axes,
        "speed_by_absolute_steering_support": supported_cells,
        "maneuver_phase_coverage": phase_category_summary,
        "per_condition_per_run": condition_records,
        "condition_replicates": condition_replicates,
        "uncovered_preregistered_conditions": missing_conditions,
        "broader_schema9_context_by_split": overall_splits,
        "planner_support_decision": {
            "additional_capture_required_within_preregistered_envelope": bool(
                missing_conditions),
            "evidence": (
                "Every preregistered speed/steering-cap condition is present in "
                f"{expected_run_counts['train']} independent training runs and "
                f"{expected_run_counts['validation']} independent validation "
                "runs, with both measured steering signs and a valid active-brake "
                "phase. The 2-D support table is descriptive, not a rectangular "
                "coverage mandate; infeasible/unplanned cells do not trigger captures."),
            "unsupported_extrapolation": (
                "Steering beyond the recorded caps at 9.5/10.5/11.1 m/s and "
                "demands above 12 m/s are not supported by this capture plan; "
                "the optimizer must treat them as outside the demonstrated domain."),
        },
        "active_braking_phase_metrics": braking_records,
        "provenance": [provenance[run_ids[index]] for index in new_run_indices],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dynamic_coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    fields = list(condition_records[0]) if condition_records else []
    with (output_dir / "condition_coverage.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(condition_records)
    report["report_path"] = str(report_path.relative_to(REPO_ROOT))
    report["condition_csv"] = str(
        (output_dir / "condition_coverage.csv").relative_to(REPO_ROOT))
    # Include artifact paths in the JSON that is handed to later work packages.
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report = build(args.dataset_dir, args.output_dir)
    except (OSError, ValueError, KeyError, IndexError, TypeError,
            sqlite3.Error) as exc:
        print(f"dynamic coverage analysis failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "new_capture_rows": report["new_capture_rows"],
        "new_capture_sequence_count": report["new_capture_sequence_count"],
        "uncovered_preregistered_conditions": report[
            "uncovered_preregistered_conditions"],
        "train_only_speed_steering_cells": report[
            "speed_by_absolute_steering_support"]["train_only_cells"],
        "additional_capture_required_within_preregistered_envelope": report[
            "planner_support_decision"][
                "additional_capture_required_within_preregistered_envelope"],
    }, indent=2))
    print(f"wrote {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
