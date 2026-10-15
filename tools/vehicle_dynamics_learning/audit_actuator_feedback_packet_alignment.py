#!/usr/bin/env python3
"""Audit allowed steering/throttle feedback against same-response Unity data.

The bridge timing topic contains a development-only copy of values returned
in the same Unity response as the sensors. It is used strictly as an offline
alignment label; the permitted Float32 feedback topics are the only candidate
runtime inputs.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as bag_analysis


FEEDBACK_TOPICS = {
    "steering": (bag_analysis.STEERING, "simulator_feedback_steering_norm"),
    "throttle": (bag_analysis.THROTTLE_FEEDBACK,
                 "simulator_feedback_throttle_norm"),
}


def _summary(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"samples": 0, "mean_abs_error": None, "p95_abs_error": None,
                "exact_fraction_1e-4": None}
    return {
        "samples": int(len(values)),
        "mean_abs_error": float(np.mean(np.abs(values))),
        "p95_abs_error": float(np.quantile(np.abs(values), 0.95)),
        "exact_fraction_1e-4": float(np.mean(np.abs(values) <= 1.0e-4)),
    }


def analyze(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = bag_analysis._topic_map(connection)
        if bag_analysis.PACKET_TIMING not in topics:
            raise ValueError(f"missing packet timing topic: {path}")
        packets = []
        for receipt_ns, message in bag_analysis._messages(
                connection, topics, bag_analysis.PACKET_TIMING):
            try:
                item = json.loads(message.data)
            except (TypeError, json.JSONDecodeError):
                continue
            sequence = item.get("packet_sequence")
            if not isinstance(sequence, int):
                continue
            packets.append({
                "receipt_ns": int(receipt_ns),
                "sequence": sequence,
                "source_ns": item.get("bridge_request_ros_stamp_ns"),
                "steering": item.get(FEEDBACK_TOPICS["steering"][1]),
                "throttle": item.get(FEEDBACK_TOPICS["throttle"][1]),
            })
        packets.sort(key=lambda row: row["receipt_ns"])
        packet_times = [row["receipt_ns"] for row in packets]
        results = {}
        for name, (topic, field) in FEEDBACK_TOPICS.items():
            if topic not in topics:
                raise ValueError(f"missing allowed feedback topic {topic}: {path}")
            feedback = [(int(receipt_ns), float(message.data))
                        for receipt_ns, message in bag_analysis._messages(
                            connection, topics, topic)
                        if np.isfinite(float(message.data))]
            feedback.sort()
            feedback_times = [row[0] for row in feedback]
            values_by_packet: list[tuple[float, float] | None] = [None] * len(packets)
            time_deltas_ms = []
            for index, packet in enumerate(packets):
                insertion = bisect.bisect_left(feedback_times, packet["receipt_ns"])
                nearby = [j for j in (insertion - 1, insertion)
                          if 0 <= j < len(feedback)]
                if not nearby:
                    continue
                selected = min(nearby, key=lambda j: abs(
                    feedback[j][0] - packet["receipt_ns"]))
                values_by_packet[index] = feedback[selected]
                time_deltas_ms.append(
                    (feedback[selected][0] - packet["receipt_ns"]) / 1.0e6)
            comparisons = {offset: [] for offset in (-1, 0, 1)}
            transition_comparisons = {offset: [] for offset in (-1, 0, 1)}
            for index, packet in enumerate(packets):
                aligned = values_by_packet[index]
                if aligned is None or not isinstance(packet[name], (int, float)):
                    continue
                observed = aligned[1]
                for offset in (-1, 0, 1):
                    candidate_index = index + offset
                    if not 0 <= candidate_index < len(packets):
                        continue
                    candidate = packets[candidate_index]
                    reference = candidate[name]
                    if not isinstance(reference, (int, float)):
                        continue
                    error = observed - float(reference)
                    comparisons[offset].append(error)
                    changed = (
                        (index == 0 or packets[index - 1][name] != packet[name])
                        or (index + 1 == len(packets) or
                            packets[index + 1][name] != packet[name]))
                    if changed:
                        transition_comparisons[offset].append(error)
            results[name] = {
                "topic": topic,
                "message_type": topics[topic][1],
                "feedback_receipt_minus_packet_debug_receipt_ms": {
                    "median": float(np.median(time_deltas_ms)) if time_deltas_ms else None,
                    "p95_abs": float(np.quantile(np.abs(time_deltas_ms), 0.95))
                    if time_deltas_ms else None,
                },
                "all_packet_value_error_by_debug_packet_offset": {
                    str(offset): _summary(np.asarray(errors))
                    for offset, errors in comparisons.items()},
                "changing_feedback_value_error_by_debug_packet_offset": {
                    str(offset): _summary(np.asarray(errors))
                    for offset, errors in transition_comparisons.items()},
            }
        return {"bag": str(path.resolve()), "packets": len(packets),
                "feedback": results}
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "policy": "Unity response sidecar used only as offline timing reference; runtime candidate is the allowed feedback topic",
        "runs": [analyze(path) for path in args.bags],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "runs": [{
            "bag": row["bag"],
            "feedback": {
                name: {
                    "receipt_minus_debug_ms": data[
                        "feedback_receipt_minus_packet_debug_receipt_ms"],
                    "all": data["all_packet_value_error_by_debug_packet_offset"],
                    "transitions": data[
                        "changing_feedback_value_error_by_debug_packet_offset"],
                }
                for name, data in row["feedback"].items()
            },
        } for row in report["runs"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
