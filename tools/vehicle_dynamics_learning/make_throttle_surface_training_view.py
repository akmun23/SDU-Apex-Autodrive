#!/usr/bin/env python3
"""Make a model-compatible r04-only training view of the throttle sequences.

Each reset-isolated condition is its own sampling group so validation and
internal test holdouts never split a sequence.  r05 remains untouched as an
external, repeat-capture evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = np.asarray((
    "u_rear_mps", "v_rear_mps", "yaw_rate_rps",
    "steering_feedback_rad", "throttle_feedback_norm",
    "rear_left_surface_mps", "rear_right_surface_mps",
    "steering_command_rad", "throttle_command_norm",
), dtype="U64")
HISTORY_STEPS = 16
ROLLOUT_STEPS = 440


def create(source: Path, manifest_path: Path, run_id: str,
           output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads(manifest_path.read_text())
    source_data = np.load(source, allow_pickle=False)
    source_arrays = {name: source_data[name] for name in (
        "sequence_run_id", "sequence_bounds", "body_state",
        "actuator_feedback", "encoder_surface_mps_100ms", "plant_commands",
        "packet_sequence")}
    frame_parts = []
    dt_parts = []
    bounds = []
    run_ids = []
    sequence_run = []
    labels = []
    skipped = []
    invalid_rows_dropped = 0
    initial_rows_before_first_valid = 0
    cursor = 0
    rows = manifest["sequences"]
    run_array = source_arrays["sequence_run_id"].astype(str)
    source_bounds = source_arrays["sequence_bounds"].astype(np.int64)
    lookup = {int(row["sequence_index"]): row for row in rows}
    selected_ids = [index for index, name in enumerate(run_array)
                    if name == run_id]
    if not selected_ids:
        raise ValueError(f"no sequences for requested capture: {run_id}")
    run_index_by_condition: dict[str, int] = {}

    for sequence_id in selected_ids:
        row = lookup[sequence_id]
        start, end = map(int, source_bounds[sequence_id])
        frames = np.column_stack((
            source_arrays["body_state"][start:end],
            source_arrays["actuator_feedback"][start:end],
            source_arrays["encoder_surface_mps_100ms"][start:end],
            source_arrays["plant_commands"][start:end],
        )).astype(np.float32)
        packet_sequence = source_arrays["packet_sequence"][start:end, 0]
        valid = np.isfinite(frames).all(axis=1)
        good = np.flatnonzero(valid)
        if not len(good):
            skipped.append({"phase_index": row["phase_index"],
                            "reason": "no_fully_observed_frame"})
            continue
        first = int(good[0])
        frames = frames[first:]
        packet_sequence = packet_sequence[first:]
        initial_rows_before_first_valid += first
        # Split at missing observations or packet gaps. Never compress a gap
        # into one nominal 25 ms integration interval.
        valid = np.isfinite(frames).all(axis=1)
        invalid_rows_dropped += int(np.count_nonzero(~valid))
        contiguous = valid.copy()
        if len(contiguous) > 1:
            contiguous[1:] &= ((np.diff(packet_sequence) == 1)
                               & valid[:-1])
        starts = np.flatnonzero(contiguous & ~np.r_[False, contiguous[:-1]])
        ends = np.flatnonzero(contiguous & ~np.r_[contiguous[1:], False]) + 1
        usable_segments = []
        for segment_start, segment_end in zip(starts, ends):
            segment = frames[segment_start:segment_end]
            if len(segment) < HISTORY_STEPS + ROLLOUT_STEPS + 1:
                skipped.append({
                    "phase_index": row["phase_index"],
                    "reason": "short_contiguous_segment",
                    "sample_count": int(len(segment)),
                })
                continue
            usable_segments.append(segment)
        if not usable_segments:
            skipped.append({"phase_index": row["phase_index"],
                            "reason": "no_usable_contiguous_segment"})
            continue
        condition_key = (round(float(row["steering_command_rad"]), 4),
                         round(float(row["throttle_start_norm"]), 2),
                         round(float(row["throttle_end_norm"]), 2))
        run_name = (f"steer_{condition_key[0]:+.4f}_throttle_"
                    f"{condition_key[1]:.2f}_to_{condition_key[2]:.2f}")
        if run_name not in run_index_by_condition:
            run_index_by_condition[run_name] = len(run_ids)
            run_ids.append(run_name)
        run_index = run_index_by_condition[run_name]
        for segment_index, segment in enumerate(usable_segments):
            sequence_run.append(run_index)
            labels.append(f"{row['label'] or run_name}:segment-{segment_index}")
            frame_parts.append(segment)
            dt_parts.append(np.full(len(segment), 0.025, dtype=np.float32))
            bounds.append((cursor, cursor + len(segment)))
            cursor += len(segment)

    if not frame_parts:
        raise ValueError("no eligible sequences survived the model-view gates")
    all_frames = np.concatenate(frame_parts).astype(np.float32, copy=False)
    all_dt = np.concatenate(dt_parts).astype(np.float32, copy=False)
    out_data = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_data,
        schema_version=np.asarray([2], dtype=np.int32),
        feature_names=FEATURE_NAMES,
        frames=all_frames,
        dt_s=all_dt,
        sequence_bounds=np.asarray(bounds, dtype=np.int64),
        sequence_run_index=np.asarray(sequence_run, dtype=np.int32),
        sequence_labels=np.asarray(labels, dtype="U256"),
        run_ids=np.asarray(run_ids, dtype="U128"),
        run_splits=np.full(len(run_ids), "train", dtype="U32"),
    )
    result = {
        "schema_version": 1,
        "source_dataset": str(source),
        "source_manifest": str(manifest_path),
        "capture_run_id": run_id,
        "split_policy": (
            "r04 only for fitting and internal condition-level validation; "
            "r05 is reserved as a full external replicate-capture test"),
        "features": FEATURE_NAMES.tolist(),
        "target_state": FEATURE_NAMES[:7].tolist(),
        "exogenous_controls": FEATURE_NAMES[7:].tolist(),
        "conditions_as_sampling_groups": True,
        "unique_condition_groups": len(run_ids),
        "sequences": len(bounds),
        "samples": int(len(all_frames)),
        "sample_period_assumed_s": 0.025,
        "sample_period_basis": (
            "40 Hz simulator packet cadence; transport/request/receive jitter "
            "is not treated as simulation time"),
        "initial_observation_policy": (
            "start at the first fully observed model frame; no fixed encoder "
            "warmup interval is removed"),
        "invalid_rows_dropped_without_imputation": invalid_rows_dropped,
        "rows_before_first_fully_observed_frame": initial_rows_before_first_valid,
        "skipped_sequences": skipped,
        "dt_min_max_s": [float(all_dt.min()), float(all_dt.max())],
        "external_test_run_id": "openplane_throttle_5pct_5deg_20260930_r05_resume",
        "training_npz": str(out_data),
        "trainer_contract": {
            "initial_history_from_measurement": HISTORY_STEPS,
            "training_rollout_steps": ROLLOUT_STEPS,
            "future_inputs": ["steering_command_rad",
                              "throttle_command_norm"],
            "future_feedback_or_truth_input": False,
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="output stem; writes .npz and .json")
    args = parser.parse_args()
    result = create(args.source, args.manifest, args.run_id, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
