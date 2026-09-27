#!/usr/bin/env python3
"""Validate and summarize repeated 4 m/s up/down steering sweeps.

This is a development-only offline evaluator. It uses measured steering and
legal proprioceptive streams; wheel slip values are kinematic proxies, never
claimed contact-force measurements.

Predeclared capture gate: every scheduled up/down steering phase valid, every
sequence start matched, each dynamics stream >=38 Hz with p95 gap <=35 ms and
max gap <=60 ms, zero collisions, and zero bridge timing faults.

Predeclared history-effect decision: paired late-window yaw-gain difference
(down minus up) must exceed 0.05 1/m in magnitude in at least two training
repetitions with a consistent sign, and the untouched final repetition must
reproduce that sign and magnitude. Each command lasts 1.2 s and scoring uses
its final 0.7 s, so a pass demonstrates sequence dependence after 0.5 s, not
steady-state tire hysteresis.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import analyze_open_plane_dynamics as common
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as exc:  # pragma: no cover - requires ROS 2 runtime
    raise SystemExit(f"ROS 2 Python modules are required: {exc}") from exc


ODOM = common.ODOM
STEERING = common.STEERING
COLLISIONS = common.COLLISIONS
TIMING_FAULT = common.TIMING_FAULT
PACKET_TIMING = common.PACKET_TIMING
PHASE = common.PHASE
STREAMS = (ODOM, STEERING, common.LEFT_ENCODER, common.RIGHT_ENCODER,
           common.IMU, PACKET_TIMING)
PHASE_RE = re.compile(
    r"^transient_r(?P<rep>[1-4])_(?P<sign>[+-]1)_(?P<direction>up|down)_"
    r"(?P<angle>0\.\d+)_4\.0mps$")
HISTORY_EFFECT_THRESHOLD_PER_M = 0.05
FULL_ANGLE_LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25,
                     0.30, 0.35, 0.42, 0.46, 0.50)
TRANSITION_LEVELS = tuple(round(0.18 + 0.01 * index, 2) for index in range(11))
SWEEP_DESIGNS = {
    (0.30, 0.42, 0.46, 0.50): (1, 2, 3, 4),
    FULL_ANGLE_LEVELS: (1, 2, 3),
    TRANSITION_LEVELS: (1, 2, 3),
    (0.20, 0.21, 0.22, 0.23): (1, 2, 3),
}


@dataclass(frozen=True)
class Block:
    repetition: int
    sign: int
    direction: str
    angle_rad: float
    phase_duration_s: float
    early_yaw_gain_per_m: float
    yaw_gain_per_m: float
    signed_yaw_rate_rps: float
    speed_mps: float
    front_slip_abs: float
    rear_slip_abs: float
    rear_longitudinal_slip_abs: float
    sample_count: int


def _decode(connection: sqlite3.Connection, topic_id: int, msg_type: str):
    message_class = get_message(msg_type)
    for timestamp, payload in connection.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp, id",
            (topic_id,)):
        yield int(timestamp), deserialize_message(bytes(payload), message_class)


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    return statistics.median(values)


def load(path: Path) -> tuple[list[Block], dict[str, Any], list[common.Phase]]:
    if not path.is_file():
        raise ValueError(f"bag database does not exist: {path}")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = common._topic_map(connection)
        missing = [name for name in (*STREAMS, COLLISIONS, TIMING_FAULT, PHASE)
                   if name not in topics]
        if missing:
            raise ValueError("missing topic(s): " + ", ".join(missing))
        phases, experiment_end = common._phase_events(connection, topics)

        steering_id, steering_type = topics[STEERING]
        steering = [(timestamp, float(message.data))
                    for timestamp, message in _decode(
                        connection, steering_id, steering_type)
                    if math.isfinite(float(message.data))]
        steering_times = [row[0] for row in steering]
        steering_rows = [common.ScalarRow(t, t, value) for t, value in steering]

        odom_id, odom_type = topics[ODOM]
        odometry = list(_decode(connection, odom_id, odom_type))

        def encoder_rows(topic: str) -> list[common.EncoderRow]:
            rows = []
            topic_id, msg_type = topics[topic]
            for receipt_ns, message in _decode(connection, topic_id, msg_type):
                if message.position and math.isfinite(float(message.position[0])):
                    rows.append(common.EncoderRow(
                        receipt_ns, common._stamp_ns(message.header.stamp),
                        float(message.position[0])))
            return rows

        left_encoder = encoder_rows(common.LEFT_ENCODER)
        right_encoder = encoder_rows(common.RIGHT_ENCODER)
        left_encoder_times = [row.receipt_ns for row in left_encoder]
        right_encoder_times = [row.receipt_ns for row in right_encoder]

        collisions_id, collisions_type = topics[COLLISIONS]
        collisions = [int(message.data) for _, message in _decode(
            connection, collisions_id, collisions_type)]
        faults_id, faults_type = topics[TIMING_FAULT]
        timing_faults = [bool(message.data) for _, message in _decode(
            connection, faults_id, faults_type)]

        receipts = {
            topic: [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp, id",
                (topics[topic][0],))]
            for topic in STREAMS
        }
    finally:
        connection.close()

    blocks: list[Block] = []
    matched_starts = 0
    valid_phase_count = 0
    for phase in phases:
        match = PHASE_RE.match(phase.label)
        if not match:
            continue
        if phase.valid is True:
            valid_phase_count += 1
        sign = 1 if match.group("sign") == "+1" else -1
        repetition = int(match.group("rep"))
        direction = match.group("direction")
        angle = float(match.group("angle"))
        if phase.commanded_steering_rad != sign * angle:
            raise ValueError(f"phase label/command mismatch: {phase.label}")
        if (phase.initial_window_speed_mps is not None
                and phase.initial_window_abs_vy_mps is not None
                and phase.initial_window_abs_yaw_rate_rps is not None
                and phase.initial_steering_rad is not None
                and abs(phase.initial_window_speed_mps - 4.0) <= 0.20
                and phase.initial_window_abs_vy_mps <= 0.08
                and phase.initial_window_abs_yaw_rate_rps <= 0.12
                and abs(phase.initial_steering_rad) <= 0.02):
            matched_starts += 1

        early_yaw_gains: list[float] = []
        yaw_gains: list[float] = []
        signed_yaw_rates: list[float] = []
        speeds: list[float] = []
        front_slips: list[float] = []
        rear_slips: list[float] = []
        rear_longitudinal_slips: list[float] = []
        early_start_ns = phase.start_ns + 500_000_000
        early_end_ns = min(phase.end_ns, phase.start_ns + 1_200_000_000)
        late_start_ns = max(early_start_ns, phase.end_ns - 700_000_000)
        for receipt_ns, message in odometry:
            if receipt_ns < early_start_ns or receipt_ns > phase.end_ns:
                continue
            actual_steering = common._nearest_scalar(
                steering_rows, steering_times, receipt_ns)
            if actual_steering is None:
                continue
            delta = actual_steering.value
            if abs(delta) < 0.01:
                continue
            twist = message.twist.twist
            vx_com, vy_com, yaw_rate = (float(twist.linear.x),
                                        float(twist.linear.y),
                                        float(twist.angular.z))
            vx, vy = common._rear_axle_velocity(vx_com, vy_com, yaw_rate)
            if not all(map(math.isfinite, (vx, vy, yaw_rate))) or vx <= 1.0:
                continue
            gain = yaw_rate / (vx * math.tan(delta))
            if math.isfinite(gain):
                if receipt_ns <= early_end_ns:
                    early_yaw_gains.append(gain)
                if receipt_ns >= late_start_ns:
                    yaw_gains.append(gain)
                    signed_yaw_rates.append(sign * yaw_rate)
                    speeds.append(vx)

            half_track = common.TRACK_WIDTH_M * 0.5
            rear_left = common._wheel_slip(vx, vy, yaw_rate, 0.0, half_track, 0.0)
            rear_right = common._wheel_slip(vx, vy, yaw_rate, 0.0, -half_track, 0.0)
            left_angle, right_angle = common._ackermann_angles(delta)
            front_left = common._wheel_slip(
                vx, vy, yaw_rate, common.WHEELBASE_M, half_track, left_angle)
            front_right = common._wheel_slip(
                vx, vy, yaw_rate, common.WHEELBASE_M, -half_track, right_angle)
            if receipt_ns >= late_start_ns and None not in (rear_left, rear_right):
                rear_slips.append(0.5 * (abs(rear_left) + abs(rear_right)))
            if receipt_ns >= late_start_ns and None not in (front_left, front_right):
                front_slips.append(0.5 * (abs(front_left) + abs(front_right)))
            rear_left_sx = common._encoder_slip(
                left_encoder, left_encoder_times, receipt_ns,
                vx - yaw_rate * half_track, 1.0)
            rear_right_sx = common._encoder_slip(
                right_encoder, right_encoder_times, receipt_ns,
                vx + yaw_rate * half_track, 1.0)
            if (receipt_ns >= late_start_ns and rear_left_sx is not None
                    and rear_right_sx is not None):
                rear_longitudinal_slips.append(
                    0.5 * (abs(rear_left_sx) + abs(rear_right_sx)))

        if len(yaw_gains) < 20 or len(early_yaw_gains) < 20:
            continue
        blocks.append(Block(
            repetition, sign, direction, angle,
            (phase.end_ns - phase.start_ns) / 1e9,
            _median(early_yaw_gains),
            _median(yaw_gains), _median(signed_yaw_rates), _median(speeds),
            _median(front_slips), _median(rear_slips),
            _median(rear_longitudinal_slips), len(yaw_gains)))

    rates = {}
    for topic, times in receipts.items():
        rate, p95_gap, max_gap = common._rate(times)
        gaps_ms = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
        positive = [gap for gap in gaps_ms if gap > 0.0]
        rates[topic] = {"hz": rate, "p95_gap_ms": p95_gap, "max_gap_ms": max_gap,
                        "duplicate_count": sum(gap == 0.0 for gap in gaps_ms),
                        "nonpositive_count": sum(gap <= 0.0 for gap in gaps_ms)}
    quality = {
        "valid_steering_phases": valid_phase_count,
        "matched_sweep_starts": matched_starts,
        "collision_initial": collisions[0] if collisions else None,
        "collision_final": collisions[-1] if collisions else None,
        "bridge_timing_faults": sum(timing_faults),
        "aborted": bool(experiment_end.get("aborted", True)),
        "rates": rates,
    }
    return blocks, quality, phases


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def evaluate(path: Path) -> bool:
    blocks, quality, phases = load(path)
    repetitions = sorted({block.repetition for block in blocks})
    angles = sorted({block.angle_rad for block in blocks})
    if len(repetitions) < 3 or len(angles) < 2:
        raise ValueError(f"not enough repeated steering levels to validate: {repetitions}, {angles}")
    expected_repetitions = SWEEP_DESIGNS.get(tuple(angles))
    if expected_repetitions is None or tuple(repetitions) != expected_repetitions:
        raise ValueError(f"unexpected steering grid/repetitions: {tuple(angles)}, {tuple(repetitions)}")
    training_repetitions = repetitions[:-1]
    holdout_repetition = repetitions[-1]
    expected_steps_per_repetition = 2 * (2 * len(angles) - 1)
    expected_steps = expected_steps_per_repetition * len(repetitions)
    expected_starts = 2 * len(expected_repetitions)
    observed_per_repetition = {
        repetition: sum(block.repetition == repetition for block in blocks)
        for repetition in repetitions}
    capture_ok = (
        quality["valid_steering_phases"] == expected_steps
        and quality["matched_sweep_starts"] == expected_starts
        and all(count == expected_steps_per_repetition
                for count in observed_per_repetition.values())
        and quality["collision_initial"] == 0
        and quality["collision_final"] == 0
        and quality["bridge_timing_faults"] == 0
        and not quality["aborted"]
        and all(metric["hz"] is not None and metric["hz"] >= 38.0
                and metric["p95_gap_ms"] is not None and metric["p95_gap_ms"] <= 35.0
                and metric["max_gap_ms"] is not None and metric["max_gap_ms"] <= 60.0
                and metric["nonpositive_count"] == 0
                for metric in quality["rates"].values())
    )
    if not capture_ok:
        print("capture rejected by predeclared quality gate:", quality)
        return False

    print(f"bag: {path}")
    print(f"capture PASS: valid steps={quality['valid_steering_phases']}/{expected_steps}, "
          f"matched starts={quality['matched_sweep_starts']}/{expected_starts}, "
          f"collisions=0, timing faults=0")
    for topic, metric in quality["rates"].items():
        print(f"  {topic}: {metric['hz']:.3f} Hz, p95/max gap="
              f"{metric['p95_gap_ms']:.2f}/{metric['max_gap_ms']:.2f} ms")

    paired: dict[tuple[int, float], dict[str, list[Block]]] = {}
    for block in blocks:
        paired.setdefault((block.repetition, block.angle_rad), {}).setdefault(
            block.direction, []).append(block)

    differences_by_rep: dict[float, dict[int, float]] = {
        angle: {} for angle in angles}
    print(f"late-window response by steering magnitude (training repetitions {training_repetitions}):")
    print("angle K_yaw(up/down) dK front|Sy|(up/down) rear|Sy|(up/down) "
          "rear|Sx|(up/down) speed(up/down) |yaw|(up/down)")
    for angle in angles:
        training = [block for block in blocks if block.repetition in training_repetitions
                    and block.angle_rad == angle]
        up = [block for block in training if block.direction == "up"]
        down = [block for block in training if block.direction == "down"]
        up_k = _mean([block.yaw_gain_per_m for block in up])
        down_k = _mean([block.yaw_gain_per_m for block in down])
        up_front = _mean([block.front_slip_abs for block in up])
        down_front = _mean([block.front_slip_abs for block in down])
        up_rear = _mean([block.rear_slip_abs for block in up])
        down_rear = _mean([block.rear_slip_abs for block in down])
        up_speed = _mean([block.speed_mps for block in up])
        down_speed = _mean([block.speed_mps for block in down])
        up_yaw = _mean([block.signed_yaw_rate_rps for block in up])
        down_yaw = _mean([block.signed_yaw_rate_rps for block in down])
        up_longitudinal_slip = _mean(
            [block.rear_longitudinal_slip_abs for block in up])
        down_longitudinal_slip = _mean(
            [block.rear_longitudinal_slip_abs for block in down])
        for repetition in training_repetitions:
            cells = paired.get((repetition, angle), {})
            if len(cells.get("up", [])) >= 2 and len(cells.get("down", [])) >= 2:
                delta = _mean([b.yaw_gain_per_m for b in cells["down"]]) - _mean(
                    [b.yaw_gain_per_m for b in cells["up"]])
                differences_by_rep[angle][repetition] = delta
        print(f"{angle:.2f} {up_k:.4f}/{down_k:.4f} {down_k-up_k:+.4f} "
              f"{up_front:.4f}/{down_front:.4f} {up_rear:.4f}/{down_rear:.4f} "
              f"{up_longitudinal_slip:.4f}/{down_longitudinal_slip:.4f} "
              f"{up_speed:.3f}/{down_speed:.3f} {up_yaw:.3f}/{down_yaw:.3f}")

    if any(block.phase_duration_s > 2.0 for block in blocks):
        print("within-phase yaw gain, early 0.5-1.2 s -> final 0.7 s "
              f"(training repetitions {training_repetitions}):")
        for angle in angles:
            values = []
            for direction in ("up", "down"):
                rows = [block for block in blocks
                        if block.repetition in training_repetitions
                        and block.angle_rad == angle
                        and block.direction == direction]
                early = _mean([block.early_yaw_gain_per_m for block in rows])
                late = _mean([block.yaw_gain_per_m for block in rows])
                values.append(f"{direction} {early:.4f}->{late:.4f}")
            print(f"  |steer|={angle:.2f}: " + "; ".join(values))

    last_angle = max(angles)
    last_blocks = [block for block in blocks
                   if block.repetition in training_repetitions
                   and block.angle_rad == last_angle and block.direction == "up"]
    base_blocks = [block for block in blocks
                   if block.repetition in training_repetitions
                   and block.angle_rad == min(angles) and block.direction == "up"]
    if last_blocks and base_blocks:
        last_gain = _mean([block.yaw_gain_per_m for block in last_blocks])
        base_gain = _mean([block.yaw_gain_per_m for block in base_blocks])
        last_rate = _mean([block.signed_yaw_rate_rps for block in last_blocks])
        base_rate = _mean([block.signed_yaw_rate_rps for block in base_blocks])
        last_front = _mean([block.front_slip_abs for block in last_blocks])
        base_front = _mean([block.front_slip_abs for block in base_blocks])
        last_rear = _mean([block.rear_slip_abs for block in last_blocks])
        base_rear = _mean([block.rear_slip_abs for block in base_blocks])
        print(f"steering-range comparison (training repetitions {training_repetitions}):")
        print(f"  |steer| {min(angles):.2f} -> {max(angles):.2f} rad: "
              f"yaw gain {base_gain:.4f} -> {last_gain:.4f} 1/m "
              f"({(last_gain / base_gain - 1.0):+.1%}); sign-corrected yaw rate "
              f"{base_rate:.4f} -> {last_rate:.4f} rad/s "
              f"({(last_rate / base_rate - 1.0):+.1%})")
        print(f"  front |Sy| {base_front:.4f} -> {last_front:.4f}; "
              f"rear |Sy| {base_rear:.4f} -> {last_rear:.4f}")

    history_effect_by_angle: dict[float, bool] = {}
    print(f"paired sequence-dependence checks (repetition {holdout_repetition} held out):")
    for angle in angles[:-1]:
        train_deltas = [differences_by_rep[angle][rep] for rep in training_repetitions
                        if rep in differences_by_rep[angle]]
        holdout_cells = paired.get((holdout_repetition, angle), {})
        holdout_delta = math.nan
        if len(holdout_cells.get("up", [])) >= 2 and len(holdout_cells.get("down", [])) >= 2:
            holdout_delta = (
                _mean([b.yaw_gain_per_m for b in holdout_cells["down"]])
                - _mean([b.yaw_gain_per_m for b in holdout_cells["up"]]))
        sign_train = (1 if _mean(train_deltas) > 0.0 else -1) if train_deltas else 0
        same_sign_reps = sum(1 for value in train_deltas
                             if abs(value) > HISTORY_EFFECT_THRESHOLD_PER_M
                             and (1 if value > 0 else -1) == sign_train)
        accepted = (len(train_deltas) == len(training_repetitions)
                    and same_sign_reps >= 2
                    and abs(_mean(train_deltas)) > HISTORY_EFFECT_THRESHOLD_PER_M
                    and math.isfinite(holdout_delta)
                    and abs(holdout_delta) > HISTORY_EFFECT_THRESHOLD_PER_M
                    and (1 if holdout_delta > 0 else -1) == sign_train)
        history_effect_by_angle[angle] = accepted
        print(f"  |steer|={angle:.2f}: train deltas="
              f"{','.join(f'{x:+.4f}' for x in train_deltas)} 1/m; "
              f"holdout={holdout_delta:+.4f} 1/m; "
              f"history_gate={'PASS' if accepted else 'no evidence'}")

    by_sign_and_angle: dict[tuple[int, float], list[float]] = {}
    for block in blocks:
        if block.direction == "up":
            by_sign_and_angle.setdefault((block.sign, block.angle_rad), []).append(
                block.signed_yaw_rate_rps)
    print(f"up-sweep sign-corrected yaw sensitivity (trained reps {training_repetitions}; rad/s):")
    for sign in (-1, 1):
        values = [_mean(by_sign_and_angle.get((sign, angle), []))
                  for angle in angles]
        slopes = [(b - a) / (right - left)
                  for a, b, left, right in zip(values, values[1:], angles, angles[1:])]
        print(f"  sign={sign:+d}: rates={[round(x, 4) for x in values]}, "
              f"interval slopes={[round(x, 4) for x in slopes]} s^-1")

    print("decision: " + ("repeatable late-window sequence dependence demonstrated; "
                          "not a steady-state hysteresis claim" if
                          any(history_effect_by_angle.values()) else
                          "no sequence dependence above the preregistered threshold"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="path to run_0.db3")
    args = parser.parse_args()
    return 0 if evaluate(args.bag) else 1


if __name__ == "__main__":
    raise SystemExit(main())
