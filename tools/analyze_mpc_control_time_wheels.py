#!/usr/bin/env python3
"""Compare MPC control-time wheel-spin prediction with encoder angle history.

This is an offline diagnostic. It never feeds encoder values or future data
into production. Encoder angle differences estimate the average rear-wheel
angular speed over one simulator epoch; MPC hidden wheel speeds are point-state
predictions. The comparison therefore tests epoch association and wheel-spin
prediction together, not an instantaneous ground-truth wheel speed.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from tools import evaluate_mpc_bag as evaluator


SCHEMA = "sdu_apex_mpc_control_time_wheels/v1"
FIXED_EPOCH_S = 0.025
MAX_ENCODER_GAP_S = 0.030
MAX_STAMP_ALIGNMENT_S = 0.005
WHEEL_RADIUS_M = 0.0325
WHEEL_OMEGA_INTERNAL_INDICES = (2, 3)  # rear-left, rear-right
ENCODER_TOPICS = {
    "left": "/autodrive/roboracer_1/left_encoder",
    "right": "/autodrive/roboracer_1/right_encoder",
}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1,
                       math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def _stats(errors: list[float]) -> dict[str, Any]:
    absolute = [abs(value) for value in errors]
    return {
        "sample_count": len(errors),
        "bias_radps": statistics.mean(errors) if errors else None,
        "rmse_radps": math.sqrt(statistics.mean(
            value * value for value in errors)) if errors else None,
        "mae_radps": statistics.mean(absolute) if errors else None,
        "p95_absolute_radps": _percentile(absolute, 0.95),
        "max_absolute_radps": max(absolute) if errors else None,
        "rmse_wheel_surface_mps": (math.sqrt(statistics.mean(
            value * value for value in errors)) * WHEEL_RADIUS_M
            if errors else None),
    }


def _encoder_rows(data: dict[str, list[dict[str, Any]]]
                  ) -> dict[str, tuple[list[float], list[dict[str, Any]]]]:
    result = {}
    for side, topic in ENCODER_TOPICS.items():
        rows = sorted(data.get(topic, []), key=lambda row: float(row["stamp"]))
        result[side] = ([float(row["stamp"]) for row in rows], rows)
    return result


def _nearest_encoder_interval(
    stamps: list[float], rows: list[dict[str, Any]], target_s: float,
) -> tuple[dict[str, Any], float, float] | None:
    insertion = bisect.bisect_left(stamps, target_s)
    candidates = [index for index in (insertion - 1, insertion)
                  if 0 <= index < len(rows)]
    if not candidates:
        return None
    current_index = min(candidates, key=lambda index:
                        abs(stamps[index] - target_s))
    if current_index == 0:
        return None
    current = rows[current_index]
    previous = rows[current_index - 1]
    interval_s = stamps[current_index] - stamps[current_index - 1]
    alignment_s = stamps[current_index] - target_s
    if (abs(alignment_s) > MAX_STAMP_ALIGNMENT_S or
            not 0.0 < interval_s <= MAX_ENCODER_GAP_S):
        return None
    # The simulator advances at 40 Hz. Use its fixed epoch, not jittered bag
    # receipt/header differences, to estimate average encoder angular speed.
    omega_average = (float(current["angle"]) - float(previous["angle"])) / \
        FIXED_EPOCH_S
    return current, omega_average, alignment_s


def analyze(data: dict[str, list[dict[str, Any]]],
            offset_packets: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    encoders = _encoder_rows(data)
    diagnostic_rows = [dict(row.get("json", {}),
                            bag_time=row.get("bag_time"))
                       for row in data.get(evaluator.DIAGNOSTICS, [])]
    matched: list[dict[str, Any]] = []
    for diagnostic in diagnostic_rows:
        command_stamp_ns = diagnostic.get("candidate_command_stamp_ns")
        internal = diagnostic.get("plant_internal_state")
        state = diagnostic.get("state")
        if (not isinstance(command_stamp_ns, int) or
                not isinstance(internal, list) or len(internal) != 22 or
                not isinstance(state, list) or len(state) < 8):
            continue
        target_s = command_stamp_ns * 1.0e-9
        measured: dict[str, tuple[float, float]] = {}
        for side, (stamps, rows) in encoders.items():
            interval = _nearest_encoder_interval(stamps, rows, target_s)
            if interval is None:
                break
            _, omega, alignment_s = interval
            measured[side] = (omega, alignment_s)
        if len(measured) != 2:
            continue

        predicted_left = float(internal[WHEEL_OMEGA_INTERNAL_INDICES[0]])
        predicted_right = float(internal[WHEEL_OMEGA_INTERNAL_INDICES[1]])
        measured_left, align_left = measured["left"]
        measured_right, align_right = measured["right"]
        source_stamp_ns = diagnostic.get("source_stamp_ns")
        source_lead_ms = (
            (command_stamp_ns - source_stamp_ns) / 1.0e6
            if isinstance(source_stamp_ns, int) else None)
        control_prediction = diagnostic.get("control_time_prediction") or {}
        errors = [predicted_left - measured_left,
                  predicted_right - measured_right]
        matched.append({
            "source_stamp_ns": source_stamp_ns,
            "candidate_command_stamp_ns": command_stamp_ns,
            "candidate_command_lead_ms": source_lead_ms,
            "encoder_alignment_left_ms": align_left * 1.0e3,
            "encoder_alignment_right_ms": align_right * 1.0e3,
            "progress_m": diagnostic.get("progress_m"),
            "forward_speed_mps": float(state[2]),
            "status": diagnostic.get("status"),
            "first_action_steering_rad": _control_value(
                diagnostic.get("first_action"), 0),
            "first_action_throttle": _control_value(
                diagnostic.get("first_action"), 1),
            "predicted_rear_left_omega_radps": predicted_left,
            "encoder_rear_left_omega_average_radps": measured_left,
            "rear_left_error_radps": errors[0],
            "predicted_rear_right_omega_radps": predicted_right,
            "encoder_rear_right_omega_average_radps": measured_right,
            "rear_right_error_radps": errors[1],
            "predicted_plant_age_ms": (
                1000.0 * float(control_prediction["age_s"])
                if isinstance(control_prediction.get("age_s"), (int, float))
                else None),
        })

    offset_results = []
    for packet_offset in offset_packets:
        errors: list[float] = []
        valid_cycles = 0
        for row in matched:
            target_s = (int(row["candidate_command_stamp_ns"]) * 1.0e-9 +
                        packet_offset * FIXED_EPOCH_S)
            pair_errors = []
            valid = True
            for side, predicted_key in (
                    ("left", "predicted_rear_left_omega_radps"),
                    ("right", "predicted_rear_right_omega_radps")):
                stamps, encoder_samples = encoders[side]
                interval = _nearest_encoder_interval(
                    stamps, encoder_samples, target_s)
                if interval is None:
                    valid = False
                    break
                _, measured_omega, _ = interval
                pair_errors.append(float(row[predicted_key]) - measured_omega)
            if valid:
                valid_cycles += 1
                errors.extend(pair_errors)
        offset_results.append({
            "packet_offset": packet_offset,
            "offset_ms": packet_offset * FIXED_EPOCH_S * 1.0e3,
            "valid_cycles": valid_cycles,
            **_stats(errors),
        })

    progress_groups: dict[str, list[float]] = {}
    for row in matched:
        progress = row.get("progress_m")
        if not isinstance(progress, (int, float)):
            continue
        bin_start = math.floor(float(progress) / 2.5) * 2.5
        key = f"{bin_start:.1f}-{bin_start + 2.5:.1f}m"
        progress_groups.setdefault(key, []).extend([
            float(row["rear_left_error_radps"]),
            float(row["rear_right_error_radps"]),
        ])

    summary = {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "diagnostic_rows": len(diagnostic_rows),
        "matched_control_epochs": len(matched),
        "fixed_epoch_s": FIXED_EPOCH_S,
        "wheel_radius_m": WHEEL_RADIUS_M,
        "encoder_semantics": (
            "WheelEncoder.FixedUpdate integrates WheelCollider.rpm into a "
            "quantized cumulative angle; delta-angle/25ms is an interval "
            "average, not instantaneous wheel speed"),
        "pairing_semantics": (
            "nearest encoder header sample to candidate_command_stamp_ns, "
            "with previous sample and fixed 25ms interval; ROS stamps do not "
            "identify Unity FixedUpdate command application"),
        "no_future_runtime_input": True,
        "wheel_state_error": _stats([
            float(row[key]) for row in matched for key in
            ("rear_left_error_radps", "rear_right_error_radps")]),
        "offset_scan": offset_results,
        "per_progress_2p5m": {
            key: _stats(errors) for key, errors in sorted(progress_groups.items())
        },
        "limitations": [
            "Encoder angle differences estimate average wheel angular speed over a physics epoch; they are not a WheelCollider.rpm sample at the command instant.",
            "The candidate command timestamp is not an applied-command or FixedUpdate sequence ID.",
            "This analysis is an epoch/model diagnostic, not a wheel-speed ground-truth claim and not a production input.",
            "A matched source-state forecast can include MPC propagation beyond the odometry source timestamp; compare only at the candidate command epoch.",
        ],
    }
    return summary, matched


def _control_value(value: Any, index: int) -> float | None:
    if not isinstance(value, list) or len(value) <= index:
        return None
    try:
        result = float(value[index])
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_run", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--offset-packets", type=int, nargs="+",
                        default=[-2, -1, 0, 1, 2],
                        help="diagnostic encoder-epoch offsets; default scans "
                             "-50 to +50 ms around candidate command time")
    args = parser.parse_args()
    if any(abs(offset) > 4 for offset in args.offset_packets):
        parser.error("packet offsets must be within +/-4 epochs")
    data = evaluator.read_bag(args.bag_run)
    report, rows = analyze(data, args.offset_packets)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({
        "report": str(json_path),
        "samples": str(csv_path) if rows else None,
        "matched_control_epochs": report["matched_control_epochs"],
        "wheel_state_error": report["wheel_state_error"],
        "offset_scan": report["offset_scan"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
