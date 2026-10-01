#!/usr/bin/env python3
"""Preserve each throttle-sweep condition only through its <=12 m/s segment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "live_runs/derived_dynamics_learning_20260928"
DEFAULT_SOURCE = DATA_ROOT / (
    "throttle_surface_40hz_sourcealigned_dataset_20260930")
DEFAULT_OUTPUT = DATA_ROOT / "throttle_surface_race_domain_v1"
MAX_SPEED_MPS = 12.0
DT_S = 0.025


def _retained_prefix(speed: np.ndarray, valid: np.ndarray
                     ) -> tuple[int, str]:
    outside = np.flatnonzero((speed > MAX_SPEED_MPS) | ~valid)
    if not len(outside):
        return len(speed), "source_end"
    stop = int(outside[0])
    return stop, ("speed_cap" if speed[stop] > MAX_SPEED_MPS
                  else "invalid_sample")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _debug_velocity_columns(names: list[str]) -> tuple[int, int]:
    expected = ("simulator_velocity_x_mps", "simulator_velocity_y_mps")
    try:
        return names.index(expected[0]), names.index(expected[1])
    except ValueError as exc:
        raise ValueError("source lacks simulator world-velocity labels") from exc


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    source_path = source_dir / "throttle_surface_sequences.npz"
    source_manifest_path = source_dir / "manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("sequence_count", -1)) != 1508:
        raise ValueError("expected the audited 1,508-condition throttle sweep")

    archive = np.load(source_path, allow_pickle=False)
    source = {key: archive[key] for key in archive.files}
    archive.close()
    required = {
        "sequence_bounds", "sequence_run_id", "sequence_index",
        "time_from_stimulus_s", "body_state", "encoder_surface_mps_100ms",
        "actuator_feedback", "imu_roll_pitch_rad", "bridge_debug_telemetry",
        "packet_sequence", "dt_s",
    }
    missing = required - set(source)
    if missing:
        raise ValueError(f"throttle archive lacks required arrays: {sorted(missing)}")
    if not np.allclose(source["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("throttle source is not on the fixed 25 ms packet clock")
    if len(source["sequence_bounds"]) != 1508:
        raise ValueError("sequence table does not contain all 1,508 conditions")

    debug_names = list(manifest["bridge_debug_telemetry_names"])
    velocity_x, velocity_y = _debug_velocity_columns(debug_names)
    debug = source["bridge_debug_telemetry"]
    speed = np.hypot(debug[:, velocity_x], debug[:, velocity_y])
    frame_count = len(speed)
    for key, values in source.items():
        if key in {"sequence_bounds", "sequence_run_id"}:
            continue
        if values.ndim and values.shape[0] != frame_count:
            raise ValueError(f"unaligned throttle frame array: {key}")

    # End a condition immediately before its first >12 m/s or invalid sample.
    # Thus its baseline and every preceding in-domain response remain available.
    frame_valid = (
        np.isfinite(speed)
        & np.isfinite(source["body_state"]).all(axis=1)
        & np.isfinite(source["actuator_feedback"]).all(axis=1)
        & np.isfinite(source["imu_roll_pitch_rad"]).all(axis=1)
        & np.isfinite(source["time_from_stimulus_s"][:, 0])
    )
    packet = source["packet_sequence"].reshape(-1)
    pieces: list[tuple[int, int]] = []
    seq_bounds: list[tuple[int, int]] = []
    seq_kept_count: list[int] = []
    seq_stop_reason: list[str] = []
    seq_first_cap_time: list[float] = []
    seq_first_cap_speed: list[float] = []
    seq_baseline_rows: list[int] = []
    seq_packet_contiguous: list[bool] = []
    cursor = 0
    run_summary: dict[str, dict[str, int]] = {}
    updated_sequences = []
    for sequence_index, (start_raw, end_raw) in enumerate(source["sequence_bounds"]):
        start, end = int(start_raw), int(end_raw)
        if not (0 <= start < end <= frame_count):
            raise ValueError(f"invalid source bounds at condition {sequence_index}")
        local_speed = speed[start:end]
        local_valid = frame_valid[start:end]
        stop, reason = _retained_prefix(local_speed, local_valid)
        kept_start, kept_end = start, start + stop
        kept = max(0, kept_end - kept_start)
        seq_bounds.append((cursor, cursor + kept))
        cursor += kept
        if kept:
            pieces.append((kept_start, kept_end))
        seq_kept_count.append(kept)
        seq_stop_reason.append(reason)
        after_stimulus = np.flatnonzero(local_speed > MAX_SPEED_MPS)
        seq_first_cap_time.append(
            float(source["time_from_stimulus_s"][start + after_stimulus[0], 0])
            if len(after_stimulus) else float("nan"))
        seq_first_cap_speed.append(
            float(local_speed[after_stimulus[0]]) if len(after_stimulus)
            else float("nan"))
        times = source["time_from_stimulus_s"][start:kept_end, 0]
        seq_baseline_rows.append(int(np.count_nonzero(
            (times >= -3.75) & (times <= -0.25))))
        seq_packet_contiguous.append(bool(
            kept <= 1 or np.all(np.diff(packet[kept_start:kept_end]) == 1)))

        run_id = str(source["sequence_run_id"][sequence_index])
        summary = run_summary.setdefault(run_id, {
            "conditions": 0, "retained_conditions": 0, "rows_before": 0,
            "rows_after": 0, "conditions_crossing_cap": 0,
            "conditions_with_baseline": 0, "noncontiguous_packet_conditions": 0,
        })
        summary["conditions"] += 1
        summary["rows_before"] += end - start
        summary["rows_after"] += kept
        summary["retained_conditions"] += int(kept > 0)
        summary["conditions_crossing_cap"] += int(reason == "speed_cap")
        summary["conditions_with_baseline"] += int(seq_baseline_rows[-1] >= 20)
        summary["noncontiguous_packet_conditions"] += int(
            not seq_packet_contiguous[-1])

        descriptor = dict(manifest["sequences"][sequence_index])
        first_cap_time = seq_first_cap_time[-1]
        first_cap_speed = seq_first_cap_speed[-1]
        descriptor.update({
            "race_domain_kept_sample_count": kept,
            "race_domain_stop_reason": reason,
            "race_domain_baseline_sample_count": seq_baseline_rows[-1],
            "race_domain_first_gt12_time_s": (
                first_cap_time if np.isfinite(first_cap_time) else None),
            "race_domain_first_gt12_speed_mps": (
                first_cap_speed if np.isfinite(first_cap_speed) else None),
            "race_domain_packet_contiguous": seq_packet_contiguous[-1],
        })
        updated_sequences.append(descriptor)

    # Keep sequence-level run metadata, even when a condition has no usable rows.
    sequence_arrays = {"sequence_bounds", "sequence_run_id"}
    frame_arrays = {
        key: np.concatenate([source[key][left:right] for left, right in pieces], axis=0)
        if pieces else source[key][:0]
        for key in source
        if key not in sequence_arrays and source[key].ndim >= 1
        and source[key].shape[0] == frame_count
    }
    output_arrays = {
        key: value for key, value in source.items()
        if key in sequence_arrays or key not in frame_arrays
    }
    output_arrays.update(frame_arrays)
    output_arrays.update({
        "sequence_bounds": np.asarray(seq_bounds, dtype=np.int64),
        "sequence_kept_sample_count": np.asarray(seq_kept_count, dtype=np.int32),
        "sequence_stop_reason": np.asarray(seq_stop_reason, dtype="U20"),
        "sequence_baseline_sample_count": np.asarray(seq_baseline_rows, dtype=np.int32),
        "sequence_first_gt12_time_s": np.asarray(seq_first_cap_time, dtype=np.float32),
        "sequence_first_gt12_speed_mps": np.asarray(seq_first_cap_speed, dtype=np.float32),
        "sequence_packet_contiguous": np.asarray(seq_packet_contiguous, dtype=bool),
        "frame_source_index": np.concatenate([
            np.arange(left, right, dtype=np.int64) for left, right in pieces
        ]) if pieces else np.empty(0, dtype=np.int64),
        "frame_domain_speed_mps": np.concatenate([
            speed[left:right].astype(np.float32) for left, right in pieces
        ]) if pieces else np.empty(0, dtype=np.float32),
        "schema_version": np.asarray([1], dtype=np.int32),
        "dataset_role": np.asarray(["throttle_race_domain"], dtype="U32"),
        "domain_speed_cap_mps": np.asarray([MAX_SPEED_MPS], dtype=np.float32),
    })

    output_dir.mkdir(parents=True)
    output_npz = output_dir / "throttle_surface_sequences.npz"
    np.savez_compressed(output_npz, **output_arrays)
    result_manifest = {
        **manifest,
        "schema_version": 1,
        "dataset_role": "throttle_surface_race_domain_response_analysis",
        "dataset_path": str(output_npz.resolve()),
        "dataset_sha256": _sha256(output_npz),
        "source_dataset": str(source_path.resolve()),
        "source_dataset_sha256": _sha256(source_path),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "speed_cap_mps": MAX_SPEED_MPS,
        "speed_definition": "hypot(simulator truth world velocity x, y)",
        "dt_s": DT_S,
        "condition_count": len(seq_bounds),
        "row_count_before": int(frame_count),
        "row_count_after": int(sum(seq_kept_count)),
        "conditions_with_retained_data": int(sum(value > 0 for value in seq_kept_count)),
        "conditions_with_at_least_20_baseline_samples": int(
            sum(value >= 20 for value in seq_baseline_rows)),
        "conditions_truncated_at_speed_cap": int(
            sum(reason == "speed_cap" for reason in seq_stop_reason)),
        "conditions_stopped_at_invalid_sample": int(
            sum(reason == "invalid_sample" for reason in seq_stop_reason)),
        "conditions_with_noncontiguous_packet_sequence": int(
            sum(not value for value in seq_packet_contiguous)),
        "rows_never_reordered": True,
        "every_condition_metadata_retained": True,
        "run_summary": run_summary,
        "sequences": updated_sequences,
        "note": (
            "Each condition is retained through its first >12 m/s or invalid "
            "sample; all preceding baseline and response rows remain. No row "
            "above the cap enters response analysis."),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return result_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.source_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "condition_count", "row_count_before", "row_count_after",
        "conditions_with_retained_data",
        "conditions_with_at_least_20_baseline_samples",
        "conditions_truncated_at_speed_cap",
        "conditions_with_noncontiguous_packet_sequence", "run_summary")},
        indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
