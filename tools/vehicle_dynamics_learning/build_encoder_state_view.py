#!/usr/bin/env python3
"""Create an immutable sidecar dataset with packet-aligned raw wheel rates.

The source dataset and its stored encoder-speed features are left unchanged.
Only train/validation (or explicitly unseen-practice) runs are extracted from
their source bags. Test/final-test source bags are deliberately not opened.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
MAX_ALIGNMENT_NS = 30_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_bags(paths: list[Path]) -> tuple[dict[str, Path], dict[str, str]]:
    bags: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for manifest_path in paths:
        manifest_path = manifest_path.resolve()
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes[str(manifest_path)] = _sha256(manifest_path)
        rows = [*document.get("runs", []), *document.get("run_provenance", [])]
        if document.get("bag"):
            rows.append({"run_id": document.get("run_id"),
                         "bag": document["bag"]})
        for row in rows:
            run_id, bag_value = row.get("run_id"), row.get("bag")
            if not run_id or not bag_value:
                continue
            bag = Path(str(bag_value))
            bag = (ROOT / bag).resolve() if not bag.is_absolute() else bag.resolve()
            prior = bags.get(str(run_id))
            if prior is not None and prior != bag:
                raise ValueError(f"conflicting bag paths for {run_id}: {prior}, {bag}")
            bags[str(run_id)] = bag
    return bags, hashes


def _frame_run_index(data: dict[str, Any]) -> np.ndarray:
    result = np.full(len(data["frames"]), -1, dtype=np.int32)
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if np.any(result[start:end] >= 0):
            raise ValueError("overlapping source sequence bounds")
        result[start:end] = run
    if np.any(result < 0):
        raise ValueError("source sequence bounds do not cover all frames")
    return result


def _encoder_angles(bag_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    connection = sqlite3.connect(bag_path.as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        missing = [topic for topic in (analysis.LEFT_ENCODER,
                                       analysis.RIGHT_ENCODER)
                   if topic not in topics]
        if missing:
            raise ValueError(f"{bag_path}: missing encoder topics {missing}")
        result = {}
        for topic in (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER):
            rows = []
            for _, message in analysis._messages(connection, topics, topic):
                if message.position and np.isfinite(message.position[0]):
                    rows.append((analysis._stamp_ns(message.header.stamp),
                                 float(message.position[0])))
            rows.sort(key=lambda row: row[0])
            stamps = np.asarray([row[0] for row in rows], dtype=np.int64)
            angles = np.asarray([row[1] for row in rows], dtype=np.float64)
            if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
                raise ValueError(f"{bag_path}: invalid {topic} stamp sequence")
            result[topic] = (stamps, angles)
        return result
    finally:
        connection.close()


def _nearest(series: tuple[np.ndarray, np.ndarray], stamp_ns: int
             ) -> tuple[int, float] | None:
    stamps, angles = series
    position = bisect.bisect_left(stamps, int(stamp_ns))
    candidates = [index for index in (position - 1, position)
                  if 0 <= index < len(stamps)]
    if not candidates:
        return None
    index = min(candidates, key=lambda item: abs(int(stamps[item]) - stamp_ns))
    source_stamp = int(stamps[index])
    if abs(source_stamp - stamp_ns) > MAX_ALIGNMENT_NS:
        return None
    return source_stamp, float(angles[index])


def _latest_source_rate(series: tuple[np.ndarray, np.ndarray], stamp_ns: int
                        ) -> tuple[tuple[int, float], tuple[int, float]] | None:
    """Return one causal encoder interval ending near a sample timestamp."""
    stamps, angles = series
    current_index = bisect.bisect_right(stamps, int(stamp_ns)) - 1
    if current_index < 1:
        return None
    current = (int(stamps[current_index]), float(angles[current_index]))
    previous = (int(stamps[current_index - 1]),
                float(angles[current_index - 1]))
    if abs(current[0] - int(stamp_ns)) > MAX_ALIGNMENT_NS:
        return None
    if _source_rate(previous, current) is None:
        return None
    return previous, current


def _source_rate(previous: tuple[int, float] | None,
                 current: tuple[int, float] | None) -> float | None:
    if previous is None or current is None:
        return None
    source_dt_s = (current[0] - previous[0]) / 1e9
    if not 0.015 <= source_dt_s <= 0.035:
        return None
    # Encoder messages represent a fixed 40 Hz physical cadence. Header/stamp
    # jitter is alignment metadata, not a variable wheel integration period.
    return (current[1] - previous[1]) * WHEEL_RADIUS_M / DT_S


def build(dataset_path: Path, output_path: Path,
          source_manifests: list[Path]) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    output_path = output_path.resolve()
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite encoder-state view: {output_path}")
    data = _load_dataset(dataset_path)
    if not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("encoder-state view requires fixed 25 ms source data")
    for key in ("sample_time_ns", "packet_sequence", "sequence_reset_index"):
        if key not in data or data[key] is None:
            raise ValueError(f"source dataset lacks required {key}")
    bags, manifest_hashes = _manifest_bags(source_manifests)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    splits = np.asarray(data["splits"]).astype(str)
    run_by_frame = _frame_run_index(data)
    times = np.asarray(data["sample_time_ns"], dtype=np.int64)
    packets = np.asarray(data["packet_sequence"], dtype=np.int64)
    resets = np.full(len(times), -1, dtype=np.int64)
    for (start_raw, end_raw), reset_raw in zip(
            data["bounds"], data["sequence_reset_index"]):
        resets[int(start_raw):int(end_raw)] = int(reset_raw)
    if np.any(resets < 0):
        raise ValueError("sequence reset metadata does not cover all frames")
    raw = np.zeros((len(times), 2), dtype=np.float32)
    valid = np.zeros(len(times), dtype=bool)
    run_reports: dict[str, Any] = {}
    reset_epoch_starts: dict[tuple[int, int], int] = {}
    for (start_raw, _), run_raw, reset_raw in zip(
            data["bounds"], data["seq_run"], data["sequence_reset_index"]):
        run_index, reset_index = int(run_raw), int(reset_raw)
        if reset_index > 0:
            key = (run_index, reset_index)
            stamp = int(times[int(start_raw)])
            reset_epoch_starts[key] = min(
                stamp, reset_epoch_starts.get(key, stamp))

    eligible_splits = {"train", "validation", "unseen_practice"}
    for run_index, run_id in enumerate(run_ids):
        split = str(splits[run_index])
        indices = np.flatnonzero(run_by_frame == run_index)
        if split not in eligible_splits:
            run_reports[str(run_id)] = {
                "split": split,
                "frames": int(len(indices)),
                "source_bag_opened": False,
                "raw_rate_valid_frames": 0,
                "status": "excluded by split policy",
            }
            continue
        if not len(indices):
            continue
        if str(run_id) not in bags:
            raise ValueError(f"no source bag manifest entry for eligible run {run_id}")
        bag_path = bags[str(run_id)]
        encoders = _encoder_angles(bag_path)
        aligned_counts = {"left": 0, "right": 0, "both": 0}
        filter_differences: list[list[float]] = []
        for frame_index in indices:
            index = int(frame_index)
            # Use the latest two encoder samples at or before this packet, not
            # a centered/nearest pair that could include a future measurement.
            # The source encoder is 40 Hz even when the bridge packet stream
            # drops or batches messages, so this remains a 25 ms physical
            # wheel-speed estimate across bridge-packet gaps.
            source_pairs = [
                _latest_source_rate(encoders[topic], int(times[index]))
                for topic in (analysis.LEFT_ENCODER,
                              analysis.RIGHT_ENCODER)]
            reset_start = reset_epoch_starts.get(
                (run_index, int(resets[index])))
            if (all(pair is not None for pair in source_pairs)
                    and source_pairs[0][0][0] == source_pairs[1][0][0]
                    and source_pairs[0][1][0] == source_pairs[1][1][0]
                    and (reset_start is None
                         or min(pair[0][0] for pair in source_pairs) >= reset_start)):
                rates = [_source_rate(pair[0], pair[1])
                         for pair in source_pairs]
                if all(value is not None for value in rates):
                    raw[index] = np.asarray(rates, dtype=np.float32)
                    valid[index] = True
                    aligned_counts["both"] += 1
            if index == int(indices[0]):
                continue
            previous_index = index - 1
            if (run_by_frame[previous_index] != run_index
                    or resets[previous_index] != resets[index]
                    or packets[index] != packets[previous_index] + 1
                    or not np.isclose(data["dt_s"][index], DT_S,
                                      rtol=0.0, atol=1e-7)):
                continue
            previous_pairs = [
                _nearest(encoders[topic], int(times[previous_index]))
                for topic in (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)]
            current_pairs = [
                _nearest(encoders[topic], int(times[index]))
                for topic in (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)]
            for side, before, after in zip(("left", "right"),
                                           previous_pairs, current_pairs):
                if before is not None and after is not None:
                    aligned_counts[side] += 1
            if (any(pair is None for pair in (*previous_pairs, *current_pairs))
                    or previous_pairs[0][0] != previous_pairs[1][0]
                    or current_pairs[0][0] != current_pairs[1][0]):
                continue
            # Independently confirm the existing label's ~100 ms causal
            # cumulative-angle definition; this is never a model input.
            four_back = index - 4
            if (four_back >= 0 and run_by_frame[four_back] == run_index
                    and resets[four_back] == resets[index]
                    and packets[index] - packets[four_back] == 4):
                old_pairs = [
                    _nearest(encoders[topic], int(times[four_back]))
                    for topic in (analysis.LEFT_ENCODER,
                                  analysis.RIGHT_ENCODER)]
                if (all(pair is not None for pair in old_pairs)
                        and old_pairs[0][0] == old_pairs[1][0]):
                    elapsed_s = (current_pairs[0][0] - old_pairs[0][0]) / 1e9
                    if 0.075 <= elapsed_s <= 0.125:
                        reconstructed = np.asarray([
                            (current[1] - old[1]) * WHEEL_RADIUS_M / elapsed_s
                            for old, current in zip(old_pairs, current_pairs)])
                        difference = reconstructed - data["frames"][index, 5:7]
                        filter_differences.append(difference.tolist())

        differences = np.asarray(filter_differences, dtype=np.float64)
        run_reports[str(run_id)] = {
            "split": split,
            "frames": int(len(indices)),
            "source_bag": str(bag_path.relative_to(ROOT)),
            "source_bag_sha256": _sha256(bag_path),
            "source_bag_opened": True,
            "raw_rate_valid_frames": int(valid[indices].sum()),
            "raw_rate_valid_fraction_of_run": float(valid[indices].mean()),
            "aligned_previous_current_pairs": aligned_counts,
            "100ms_filter_reconstruction_samples": int(len(differences)),
            "reconstructed_100ms_minus_stored_bias_mps": (
                np.mean(differences, axis=0).tolist() if len(differences) else None),
            "reconstructed_100ms_minus_stored_rmse_mps": (
                np.sqrt(np.mean(differences ** 2, axis=0)).tolist()
                if len(differences) else None),
        }

    with np.load(dataset_path, allow_pickle=False) as source_archive:
        arrays = {key: np.asarray(source_archive[key])
                  for key in source_archive.files}
    arrays["encoder_raw_surface_mps"] = raw
    arrays["encoder_raw_valid"] = valid
    arrays["encoder_raw_schema_version"] = np.asarray([1], dtype=np.int32)
    arrays["encoder_raw_wheel_radius_m"] = np.asarray(
        [WHEEL_RADIUS_M], dtype=np.float64)
    arrays["encoder_raw_source_dataset_sha256"] = np.asarray(
        [_sha256(dataset_path)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode="wb", prefix=output_path.name + ".", suffix=".partial",
            dir=output_path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.replace(temp_path, output_path)
    report = {
        "schema_version": 1,
        "purpose": "offline physical wheel-rate teacher sidecar; original filtered frames unchanged",
        "source_dataset": str(dataset_path.relative_to(ROOT)),
        "source_dataset_sha256": _sha256(dataset_path),
        "output_dataset": str(output_path.relative_to(ROOT)),
        "output_dataset_sha256": _sha256(output_path),
        "source_manifest_sha256": manifest_hashes,
        "split_policy": {
            "opened_splits": sorted(eligible_splits),
            "excluded_test_and_final_test_bags": True,
            "future_measurement_use": "teacher labels only; rollout uses predicted state and future commands",
        },
        "raw_speed_definition": "latest two causal encoder angles at or before each sample time; angle delta times 0.059 m/rad divided by fixed 25 ms (40 Hz); source-stamp interval is used only to reject intervals outside 15-35 ms",
        "existing_filtered_speed_definition": "retained unchanged in frames[:,5:7]; reconstructed separately over four packet intervals for audit",
        "valid_raw_rate_frames": int(valid.sum()),
        "frame_count": int(len(valid)),
        "run_reports": run_reports,
    }
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, action="append",
                        required=True,
                        help="manifest with run_id-to-source-bag provenance; repeat as needed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.dataset, args.output, args.source_manifest)
    print(json.dumps({
        "output_dataset": result["output_dataset"],
        "output_dataset_sha256": result["output_dataset_sha256"],
        "valid_raw_rate_frames": result["valid_raw_rate_frames"],
        "frame_count": result["frame_count"],
        "runs": {key: {
            "split": row["split"],
            "valid": row["raw_rate_valid_frames"],
            "frames": row["frames"],
            "bag_opened": row["source_bag_opened"],
            "filter_rmse": row.get("reconstructed_100ms_minus_stored_rmse_mps"),
        } for key, row in result["run_reports"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
