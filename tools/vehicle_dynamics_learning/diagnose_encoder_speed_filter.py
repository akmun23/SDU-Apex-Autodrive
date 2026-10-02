#!/usr/bin/env python3
"""Measure how the logged 100 ms encoder-speed proxy differs from 25 ms rates.

The comparison is offline-only. It checks whether smoothed encoder targets are
being mistaken for instantaneous wheel-rotation state by the plant teacher.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    COM_X_M,
    DT_S,
    HALF_TRACK_M,
    WHEEL_RADIUS_M,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ENCODER_TOPICS = (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER)
MAX_ENCODER_ALIGNMENT_NS = 30_000_000


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "bias": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "absolute_p50": float(np.quantile(np.abs(values), 0.50)),
        "absolute_p95": float(np.quantile(np.abs(values), 0.95)),
    }


def _nearest_angle(topic: str, target_ns: int,
                   values: dict[str, dict[int, float]],
                   stamps: dict[str, list[int]]) -> tuple[int, float] | None:
    local_stamps = stamps[topic]
    position = bisect.bisect_left(local_stamps, target_ns)
    candidates = [index for index in (position - 1, position)
                  if 0 <= index < len(local_stamps)]
    if not candidates:
        return None
    selected = min(candidates, key=lambda index:
                   abs(local_stamps[index] - target_ns))
    selected_stamp = local_stamps[selected]
    if abs(selected_stamp - target_ns) > MAX_ENCODER_ALIGNMENT_NS:
        return None
    return selected_stamp, values[topic][selected_stamp]


def diagnose(bag_path: Path, dataset_path: Path, run_id: str,
             output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    data = _load_dataset(dataset_path)
    run_lookup = {str(value): index
                  for index, value in enumerate(data["run_ids"])}
    if run_id not in run_lookup:
        raise ValueError(f"{run_id}: run not present in {dataset_path}")
    run_index = run_lookup[run_id]
    sequence_ids = np.flatnonzero(data["seq_run"] == run_index)
    frame_indexes = np.concatenate([
        np.arange(int(data["bounds"][sequence_id, 0]),
                  int(data["bounds"][sequence_id, 1]))
        for sequence_id in sequence_ids
    ]) if len(sequence_ids) else np.asarray([], dtype=np.int64)
    if not len(frame_indexes):
        raise ValueError(f"{run_id}: has no aligned dataset frames")

    connection = sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        missing = [topic for topic in ENCODER_TOPICS if topic not in topics]
        if missing:
            raise ValueError(f"bag lacks encoder topic(s): {missing}")
        encoders: dict[str, dict[int, float]] = {}
        encoder_stamps: dict[str, list[int]] = {}
        for topic in ENCODER_TOPICS:
            values = {}
            stamps = []
            for _, message in analysis._messages(connection, topics, topic):
                if not message.position or not np.isfinite(message.position[0]):
                    continue
                stamp = analysis._stamp_ns(message.header.stamp)
                values[stamp] = float(message.position[0])
                stamps.append(stamp)
            encoders[topic] = values
            encoder_stamps[topic] = sorted(stamps)
    finally:
        connection.close()

    times = data["sample_time_ns"]
    packets = data["packet_sequence"]
    frames = data["frames"]
    rigid = data["simulator_rigid_state"]
    if (times is None or packets is None or rigid is None
            or not np.allclose(data["dt_s"][frame_indexes], DT_S,
                               rtol=0.0, atol=1.0e-7)):
        raise ValueError("requires aligned packet, rigid-state, and fixed 25 ms labels")

    sample_rows: list[dict[str, Any]] = []
    aligned_sample_count = 0
    alignment_offsets_ms: list[float] = []
    for sequence_id in sequence_ids:
        start, end = map(int, data["bounds"][sequence_id])
        if int(data["seq_run"][sequence_id]) != run_index:
            raise ValueError("sequence/run metadata disagrees")
        for index in range(start + 1, end):
            if int(packets[index]) != int(packets[index - 1]) + 1:
                continue
            target_stamp = int(times[index])
            previous_target_stamp = int(times[index - 1])
            current_pairs = [_nearest_angle(topic, target_stamp, encoders,
                                            encoder_stamps)
                             for topic in ENCODER_TOPICS]
            previous_pairs = [_nearest_angle(topic, previous_target_stamp,
                                             encoders, encoder_stamps)
                              for topic in ENCODER_TOPICS]
            if any(pair is None for pair in (*current_pairs, *previous_pairs)):
                continue
            current_stamps = [pair[0] for pair in current_pairs]
            previous_stamps = [pair[0] for pair in previous_pairs]
            if current_stamps[0] != current_stamps[1] or previous_stamps[0] != previous_stamps[1]:
                continue
            aligned_sample_count += 1
            alignment_offsets_ms.extend(
                (pair[0] - target_stamp) / 1e6 for pair in current_pairs)
            stamp = current_stamps[0]
            previous_stamp = previous_stamps[0]
            dt_s = (stamp - previous_stamp) / 1e9
            if not 0.015 <= dt_s <= 0.035:
                continue
            instantaneous = np.asarray([
                (current_pair[1] - previous_pair[1])
                * WHEEL_RADIUS_M / dt_s
                for current_pair, previous_pair in zip(current_pairs,
                                                       previous_pairs)
            ])
            window_index = index - 4
            smoothed = np.full(2, np.nan, dtype=np.float64)
            if window_index >= start and int(packets[index]) - int(packets[window_index]) == 4:
                old_pairs = [_nearest_angle(
                    topic, int(times[window_index]), encoders, encoder_stamps)
                             for topic in ENCODER_TOPICS]
                if all(pair is not None for pair in old_pairs):
                    old_stamps = [pair[0] for pair in old_pairs]
                    window_dt_s = (stamp - old_stamps[0]) / 1e9
                    if (old_stamps[0] == old_stamps[1]
                            and 0.075 <= window_dt_s <= 0.125):
                        smoothed = np.asarray([
                            (current_pair[1] - old_pair[1])
                            * WHEEL_RADIUS_M / window_dt_s
                            for current_pair, old_pair in zip(current_pairs,
                                                              old_pairs)
                        ])
            yaw_rate = float(rigid[index, 12])
            contact_speed = np.asarray((
                rigid[index, 7] - yaw_rate * HALF_TRACK_M,
                rigid[index, 7] + yaw_rate * HALF_TRACK_M,
            ), dtype=np.float64)
            sample_rows.append({
                "index": index,
                "instantaneous": instantaneous,
                "smoothed": smoothed,
                "stored": frames[index, 5:7].astype(np.float64),
                "contact": contact_speed,
                "speed": float(np.hypot(rigid[index, 7], rigid[index, 8])),
                "steering": float(frames[index, 3]),
            })

    if not sample_rows:
        raise ValueError("no exact packet-aligned encoder samples found")
    instantaneous = np.stack([row["instantaneous"] for row in sample_rows])
    smoothed = np.stack([row["smoothed"] for row in sample_rows])
    stored = np.stack([row["stored"] for row in sample_rows])
    contact = np.stack([row["contact"] for row in sample_rows])
    speed = np.asarray([row["speed"] for row in sample_rows])
    steering = np.asarray([row["steering"] for row in sample_rows])
    finite_smoothed = np.isfinite(smoothed).all(axis=1)

    wheel_rows = {}
    for wheel_index, wheel_name in enumerate(("left", "right")):
        wheel_rows[wheel_name] = {
            "raw_25ms_minus_ground_contact_speed_mps": _summary(
                instantaneous[:, wheel_index] - contact[:, wheel_index]),
            "raw_100ms_minus_ground_contact_speed_mps": _summary(
                smoothed[finite_smoothed, wheel_index]
                - contact[finite_smoothed, wheel_index]),
            "raw_25ms_minus_stored_100ms_mps": _summary(
                instantaneous[:, wheel_index] - stored[:, wheel_index]),
            "raw_100ms_minus_stored_100ms_mps": _summary(
                smoothed[finite_smoothed, wheel_index]
                - stored[finite_smoothed, wheel_index]),
        }

    bins = {}
    for low, high in ((0.0, 0.15), (0.15, 0.30),
                      (0.30, 0.45), (0.45, 0.524)):
        mask = (np.abs(steering) >= low) & (np.abs(steering) < high)
        if not mask.any():
            continue
        bins[f"abs_steering_{low:g}_{high:g}_rad"] = {
            "sample_count": int(mask.sum()),
            "median_speed_mps": float(np.median(speed[mask])),
            "per_wheel_raw25_minus_stored100_mps": [
                _summary(instantaneous[mask, wheel] - stored[mask, wheel])
                for wheel in range(2)
            ],
            "per_wheel_stored100_minus_contact_mps": [
                _summary(stored[mask, wheel] - contact[mask, wheel])
                for wheel in range(2)
            ],
        }

    rates = {}
    for topic in ENCODER_TOPICS:
        stamps = np.asarray(encoder_stamps[topic], dtype=np.int64)
        gaps_ms = np.diff(stamps) / 1e6
        rates[topic.rsplit("/", 1)[-1]] = {
            "messages": int(len(stamps)),
            "source_rate_hz": float((len(stamps) - 1)
                                    / ((stamps[-1] - stamps[0]) / 1e9)),
            "source_gap_ms_p50_p95_max": np.quantile(
                gaps_ms, (0.50, 0.95, 1.0)).tolist(),
        }
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "bag": str(bag_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "simulator_dt_s": DT_S,
        "encoder_radius_m": WHEEL_RADIUS_M,
        "stored_encoder_speed_definition": "causal cumulative-angle difference over approximately 100 ms",
        "raw_25ms_speed_definition": "adjacent packet cumulative-angle difference divided by exact source-stamp interval",
        "encoder_source_rates": rates,
        "sample_packet_encoder_alignments_within_30ms": int(aligned_sample_count),
        "encoder_to_sample_receipt_offset_ms_p50_p95_abs_max": np.quantile(
            np.abs(alignment_offsets_ms), (0.50, 0.95, 1.0)).tolist(),
        "aligned_samples_scored": int(len(sample_rows)),
        "100ms_windows_scored": int(np.count_nonzero(finite_smoothed)),
        "per_wheel": wheel_rows,
        "steering_bins": bins,
        "interpretation_limit": (
            "encoder-derived speed is a wheel-rotation measurement, while rigid-body contact speed is not wheel speed under slip;"
            " their difference is a kinematic slip proxy, not tire-force truth"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.bag, args.dataset, args.run_id, args.output)
    print(json.dumps({
        "run_id": result["run_id"],
        "encoder_source_rates": result["encoder_source_rates"],
        "aligned_samples_scored": result["aligned_samples_scored"],
        "per_wheel": result["per_wheel"],
        "steering_bins": result["steering_bins"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
