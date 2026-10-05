#!/usr/bin/env python3
"""Build the reset-plan body-only SUBNET data view from simulator truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
DEFAULT_SOURCE = (
    ROOT / "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261002/replacement_teacher_dataset_v2_20261004/"
    "openplane_dynamics.npz")
DEFAULT_OUTPUT = RESET_ROOT / "body_sysid_v1.npz"
DT_S = 0.025
REAR_AXLE_TO_COM_X_M = 0.15532
INPUT_NAMES = ("steering_feedback_rad", "throttle_feedback_norm")
OUTPUT_NAMES = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")
COMMAND_NAMES = ("steering_command_rad", "throttle_command_norm")
INCLUDED_SPLITS = {"train", "validation"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source_path: Path, output_path: Path,
          include_runs: set[str] | None = None,
          split_overrides: dict[str, str] | None = None,
          branch_manifest_path: Path | None = None,
          capture_report_path: Path | None = None) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with np.load(source_path, allow_pickle=False) as source:
        required = {
            "feature_names", "run_ids", "run_splits", "frames", "dt_s",
            "simulator_rigid_state", "sequence_bounds", "sequence_run_index",
            "sequence_condition_id", "sequence_reset_index", "sequence_labels",
            "frame_run_index", "frame_reset_index", "sample_time_ns",
        }
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"source dataset is missing arrays: {sorted(missing)}")
        features = source["feature_names"].astype(str).tolist()
        feature_index = {name: index for index, name in enumerate(features)}
        required_features = set(INPUT_NAMES + COMMAND_NAMES)
        if not required_features <= feature_index.keys():
            raise ValueError("source feature schema lacks actuator feedback/commands")
        frames = np.asarray(source["frames"], dtype=np.float64)
        rigid = np.asarray(source["simulator_rigid_state"], dtype=np.float64)
        dt = np.asarray(source["dt_s"], dtype=np.float64)
        bounds = np.asarray(source["sequence_bounds"], dtype=np.int64)
        seq_run = np.asarray(source["sequence_run_index"], dtype=np.int32)
        seq_condition = np.asarray(source["sequence_condition_id"], dtype=np.int32)
        seq_reset = np.asarray(source["sequence_reset_index"], dtype=np.int32)
        seq_labels = source["sequence_labels"].astype(str)
        run_ids = source["run_ids"].astype(str)
        run_splits = source["run_splits"].astype(str)
        frame_run = np.asarray(source["frame_run_index"], dtype=np.int32)
        frame_reset = np.asarray(source["frame_reset_index"], dtype=np.int32)

    n = len(frames)
    if frames.ndim != 2 or len(dt) != n or rigid.shape != (n, 13):
        raise ValueError("source frame, timebase, and rigid-state shapes disagree")
    if (len(run_ids) != len(run_splits) or len(seq_run) != len(bounds)
            or len(seq_condition) != len(bounds) or len(seq_reset) != len(bounds)
            or len(seq_labels) != len(bounds)):
        raise ValueError("source run and sequence metadata lengths disagree")
    if not np.allclose(dt, DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("source cadence is not exactly the 25 ms model timebase")
    if not np.isfinite(frames).all() or not np.isfinite(rigid).all():
        raise ValueError("source contains non-finite actuator or truth rows")
    if not np.isin(run_splits, (
            "train", "validation", "test", "final_test", "unseen_practice")).all():
        raise ValueError("unexpected source run split label")
    selected_runs = set(run_ids.tolist()) if include_runs is None else set(include_runs)
    missing_runs = selected_runs - set(run_ids.tolist())
    if missing_runs:
        raise ValueError(f"requested source runs are missing: {sorted(missing_runs)}")
    effective_splits = run_splits.copy()
    for run_id, split in (split_overrides or {}).items():
        if run_id not in selected_runs:
            raise ValueError(f"split override is outside selected runs: {run_id}")
        if split not in INCLUDED_SPLITS:
            raise ValueError("selected-run overrides must be train or validation")
        effective_splits[run_ids == run_id] = split

    capture_quality = None
    if (branch_manifest_path is None) != (capture_report_path is None):
        raise ValueError("branch manifest and capture report must be supplied together")
    if branch_manifest_path is None and len(selected_runs) == 1:
        source_manifest_path = source_path.with_name("manifest.json")
        if source_manifest_path.is_file():
            source_manifest = json.loads(
                source_manifest_path.read_text(encoding="utf-8"))
            run_id = next(iter(selected_runs))
            source_runs = [row for row in source_manifest.get("runs", [])
                           if row.get("run_id") == run_id]
            if len(source_runs) != 1:
                raise ValueError("source manifest does not uniquely describe selected run")
            source_run = source_runs[0]
            alignment = source_run.get("packet_sequence_alignment", {})
            packet_match = float(alignment.get("match_fraction", 0.0))
            collisions = source_run.get("collisions", [])
            streams = source_run.get("streams", {})
            stream_gate = bool(streams) and all(
                float(stats.get("hz", 0.0)) >= 38.0
                and float(stats.get("gap_p95_ms", float("inf"))) <= 35.0
                and float(stats.get("gap_max_ms", float("inf"))) <= 120.0
                for stats in streams.values())
            if (source_run.get("aborted")
                    or source_run.get("reason") != "schedule complete"
                    or not source_run.get("clean_stream_and_collision_gate")
                    or source_run.get("whole_bag_quality_failures")
                    or source_run.get("quality_failures")
                    or int(source_run.get("timing_faults", -1)) != 0
                    or collisions != [0, 0]
                    or packet_match < 0.999
                    or not stream_gate):
                raise ValueError("selected whole-run source failed its capture-quality gates")
            source_rows = int(source_manifest.get("export", {}).get(
                "samples", -1))
            if source_rows != n:
                raise ValueError("source export row count differs from prepared archive")
            capture_quality = {
                "status": "passed",
                "run_id": run_id,
                "source_manifest": str(source_manifest_path.resolve()),
                "source_manifest_sha256": sha256(source_manifest_path),
                "source_run_split": str(source_run.get("effective_split", "")),
                "source_capture_sample_count": int(
                    source_run.get("samples_exported", -1)),
                "packet_sequence_contiguous": True,
                "packet_alignment_fraction": packet_match,
                "collision_count_min_max": collisions,
                "all_stream_cadence_gates_passed": True,
                "admitted_samples": n,
            }
    if branch_manifest_path is not None and capture_report_path is not None:
        branch_manifest_path = branch_manifest_path.resolve()
        capture_report_path = capture_report_path.resolve()
        branch = json.loads(branch_manifest_path.read_text(encoding="utf-8"))
        capture = json.loads(capture_report_path.read_text(encoding="utf-8"))
        if len(selected_runs) != 1:
            raise ValueError("practice capture quality linkage requires one selected run")
        run_id = next(iter(selected_runs))
        if (branch.get("run_id") != run_id
                or capture.get("run_id") != branch.get("source_run_id")
                or branch.get("bag_sha256") != capture.get("bag_sha256")
                or branch.get("combined_dataset_sha256") != sha256(source_path)):
            raise ValueError("practice source, branch view and bag capture provenance disagree")
        intervals = branch.get("admitted_intervals", [])
        admitted_intervals = [row for row in intervals if row.get("admitted")]
        rejected_intervals = [row for row in intervals if not row.get("admitted")]
        if (not capture.get("admitted") or capture.get("failures")
                or int(capture.get("expected_laps", 0)) != 6
                or int(branch.get("admitted_interval_count", 0)) != 6
                or len(admitted_intervals) != 6
                or any(row.get("reason") != "too_short_for_branch_context"
                       for row in rejected_intervals)
                or not branch.get("bridge_packet_timing", {}).get(
                    "packet_sequence_contiguous")):
            raise ValueError("six-lap practice capture did not pass its frozen gates")
        collision_range = capture.get("receipt_gates", {}).get(
            "collision_min_max", [])
        packet_match = float(capture.get("packet_alignment", {}).get(
            "active_interval", {}).get("fraction", 0.0))
        cadence = capture.get("receipt_gates", {}).get("stream_cadence", {})
        if (collision_range != [0, 0] or packet_match < 0.999
                or not cadence or any(not stats.get("pass")
                                      for stats in cadence.values())):
            raise ValueError("practice collision, packet or 40 Hz stream gate failed")
        capture_quality = {
            "status": "passed",
            "run_id": run_id,
            "source_run_id": branch["source_run_id"],
            "bag_sha256": branch["bag_sha256"],
            "branch_manifest": str(branch_manifest_path),
            "branch_manifest_sha256": sha256(branch_manifest_path),
            "capture_validation_report": str(capture_report_path),
            "capture_validation_report_sha256": sha256(capture_report_path),
            "admitted_intervals": int(branch["admitted_interval_count"]),
            "admitted_samples": int(branch["admitted_sample_count"]),
            "packet_sequence_contiguous": True,
            "packet_alignment_fraction": packet_match,
            "collision_count_min_max": collision_range,
            "all_stream_cadence_gates_passed": True,
        }

    inputs_parts: list[np.ndarray] = []
    outputs_parts: list[np.ndarray] = []
    commands_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    sequence_parts: list[np.ndarray] = []
    reset_parts: list[np.ndarray] = []
    sample_index_parts: list[np.ndarray] = []
    condition_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    source_index_parts: list[np.ndarray] = []
    output_bounds: list[tuple[int, int]] = []
    sequence_run_ids: list[str] = []
    sequence_ids: list[str] = []
    sequence_splits: list[str] = []
    sequence_labels: list[str] = []
    sequence_reset_epochs: list[int] = []
    sequence_condition_ids: list[int] = []
    excluded_runs = sorted(set(run_ids) - selected_runs)
    cursor = 0
    previous_end = 0

    for sequence_index, (start_value, end_value) in enumerate(bounds):
        start, end = int(start_value), int(end_value)
        run_index = int(seq_run[sequence_index])
        if (start != previous_end or end <= start or end > n
                or run_index < 0 or run_index >= len(run_ids)):
            raise ValueError("invalid or non-contiguous reset-isolated sequence bounds")
        previous_end = end
        run_id = str(run_ids[run_index])
        split = str(effective_splits[run_index])
        if not np.all(frame_run[start:end] == run_index):
            raise ValueError(f"sequence crosses run boundary: {run_id}")
        if not np.all(frame_reset[start:end] == int(seq_reset[sequence_index])):
            raise ValueError(f"sequence crosses reset epoch: {run_id}")
        if run_id not in selected_runs or split not in INCLUDED_SPLITS:
            continue
        if end - start < 2:
            raise ValueError("SUBNET sequence must contain at least two samples")

        local_count = end - start
        sequence_id = f"{run_id}/sequence_{sequence_index:05d}"
        sequence_condition_id = int(seq_condition[sequence_index])
        selected_frames = frames[start:end]
        selected_rigid = rigid[start:end]
        body = np.column_stack((
            selected_rigid[:, 7],
            selected_rigid[:, 8] - REAR_AXLE_TO_COM_X_M * selected_rigid[:, 12],
            selected_rigid[:, 12],
        ))
        inputs = selected_frames[:, [feature_index[name] for name in INPUT_NAMES]]
        commands = selected_frames[:, [feature_index[name] for name in COMMAND_NAMES]]
        if not (np.isfinite(body).all() and np.isfinite(inputs).all()
                and np.isfinite(commands).all()):
            raise ValueError(f"non-finite selected train/validation sequence: {sequence_id}")
        inputs_parts.append(inputs.astype(np.float32))
        outputs_parts.append(body.astype(np.float32))
        commands_parts.append(commands.astype(np.float32))
        run_parts.append(np.full(local_count, run_id, dtype="U128"))
        sequence_parts.append(np.full(local_count, sequence_id, dtype="U192"))
        reset_parts.append(np.full(local_count, int(seq_reset[sequence_index]),
                                   dtype=np.int32))
        sample_index_parts.append(np.arange(local_count, dtype=np.int32))
        condition_parts.append(np.full(local_count, sequence_condition_id,
                                       dtype=np.int32))
        split_parts.append(np.full(local_count, split, dtype="U16"))
        time_parts.append((np.arange(local_count, dtype=np.float64) * DT_S).astype(
            np.float32))
        source_index_parts.append(np.arange(start, end, dtype=np.int64))
        output_bounds.append((cursor, cursor + local_count))
        cursor += local_count
        sequence_run_ids.append(run_id)
        sequence_ids.append(sequence_id)
        sequence_splits.append(split)
        sequence_labels.append(str(seq_labels[sequence_index]))
        sequence_reset_epochs.append(int(seq_reset[sequence_index]))
        sequence_condition_ids.append(sequence_condition_id)

    if previous_end != n or not output_bounds:
        raise ValueError("source sequences do not cover frames or contain no usable rows")
    payload = {
        "schema_version": np.asarray([1], dtype=np.int32),
        "input_names": np.asarray(INPUT_NAMES, dtype="U64"),
        "output_names": np.asarray(OUTPUT_NAMES, dtype="U64"),
        "stored_command_names": np.asarray(COMMAND_NAMES, dtype="U64"),
        "inputs": np.concatenate(inputs_parts),
        "outputs": np.concatenate(outputs_parts),
        "stored_commands": np.concatenate(commands_parts),
        "run_id": np.concatenate(run_parts),
        "sequence_id": np.concatenate(sequence_parts),
        "reset_epoch": np.concatenate(reset_parts),
        "sample_index": np.concatenate(sample_index_parts),
        "condition_id": np.concatenate(condition_parts),
        "split": np.concatenate(split_parts),
        "sample_time_s": np.concatenate(time_parts),
        "source_frame_index": np.concatenate(source_index_parts),
        "sequence_bounds": np.asarray(output_bounds, dtype=np.int64),
        "sequence_run_id": np.asarray(sequence_run_ids, dtype="U128"),
        "sequence_id_table": np.asarray(sequence_ids, dtype="U192"),
        "sequence_split": np.asarray(sequence_splits, dtype="U16"),
        "sequence_label": np.asarray(sequence_labels, dtype="U256"),
        "sequence_reset_epoch": np.asarray(sequence_reset_epochs, dtype=np.int32),
        "sequence_condition_id": np.asarray(sequence_condition_ids, dtype=np.int32),
        "dt_s": np.full(cursor, DT_S, dtype=np.float32),
        "rear_axle_to_com_x_m": np.asarray([REAR_AXLE_TO_COM_X_M], dtype=np.float32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    run_counts = {
        split: sorted({str(run) for run, role in zip(payload["run_id"], payload["split"])
                       if str(role) == split})
        for split in ("train", "validation")
    }
    report = {
        "schema_version": 1,
        "dataset": str(output_path),
        "dataset_sha256": sha256(output_path),
        "source_dataset": str(source_path),
        "source_dataset_sha256": sha256(source_path),
        "sample_period_s": DT_S,
        "input_names": list(INPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "stored_but_not_fed_inputs": list(COMMAND_NAMES),
        "body_reference_point": "rear axle",
        "body_frame": "vehicle body frame; u forward, v left, yaw rate positive CCW",
        "truth_conversion": {
            "u_rear_mps": "simulator_rigid_state[:,7] (u_com == u_rear)",
            "v_rear_mps": "simulator_rigid_state[:,8] - 0.15532 * simulator_rigid_state[:,12]",
            "yaw_rate_rps": "simulator_rigid_state[:,12]",
            "rear_axle_to_com_x_m": REAR_AXLE_TO_COM_X_M,
        },
        "run_split_policy": "whole-run train/validation only; test and final_test rows excluded",
        "selected_run_ids": sorted(selected_runs),
        "split_overrides": dict(sorted((split_overrides or {}).items())),
        "selection_source_run_splits": {
            run_id: str(effective_splits[index])
            for index, run_id in enumerate(run_ids) if run_id in selected_runs
        },
        "run_counts": {key: len(value) for key, value in run_counts.items()},
        "runs": run_counts,
        "excluded_runs_by_split": {
            **{
                split: sorted(set(run_ids[effective_splits == split].tolist()))
                for split in sorted(set(effective_splits) - INCLUDED_SPLITS)
            },
            "not_selected": sorted(set(run_ids) - selected_runs),
        },
        "excluded_run_union": excluded_runs,
        "sequence_count": len(output_bounds),
        "row_count": cursor,
        "all_sequences_reset_isolated": True,
        "sample_time_policy": "sequence-local sample_index * 0.025; receipt/network jitter ignored",
        "future_measurements_or_truth_used_as_input": False,
    }
    if capture_quality is not None:
        report["capture_quality"] = capture_quality
    report_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-run", action="append", default=[],
                        help="select only this source run; may be repeated")
    parser.add_argument("--split-override", action="append", default=[],
                        metavar="RUN_ID=ROLE",
                        help="reassign a selected run to train or validation")
    parser.add_argument("--branch-manifest", type=Path)
    parser.add_argument("--capture-report", type=Path)
    args = parser.parse_args()
    overrides = {}
    for value in args.split_override:
        if "=" not in value:
            parser.error("--split-override must be RUN_ID=train|validation")
        run_id, split = value.split("=", 1)
        if not run_id or split not in INCLUDED_SPLITS or run_id in overrides:
            parser.error("invalid or duplicate --split-override")
        overrides[run_id] = split
    print(json.dumps(build(
        args.source, args.output,
        set(args.include_run) if args.include_run else None,
        overrides, args.branch_manifest, args.capture_report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
