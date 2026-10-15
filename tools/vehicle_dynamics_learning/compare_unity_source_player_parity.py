#!/usr/bin/env python3
"""Compare matched Unity public motion from pinned and source-built Explore runs.

Reads only closed ROS bags. The source-built player's private WheelCollider
capture is not accepted as a teacher until this public-motion parity check
passes on the same deterministic command plan.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis


PACKET = analysis.PACKET_TIMING
PHASE = "/open_plane_experiment/phase"
STEERING_COMMAND = "/autodrive/roboracer_1/steering_command"
THROTTLE_COMMAND = "/autodrive/roboracer_1/throttle_command"
COLLISIONS = "/autodrive/roboracer_1/collision_count"
TIMING_FAULT = "/autodrive/roboracer_1/bridge_timing_fault"
PACKET_DT_S = 0.025


def _read_run(bag: Path) -> dict[str, Any]:
    if not bag.is_file():
        raise ValueError(f"closed bag database not found: {bag}")
    connection = sqlite3.connect(bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (PACKET, COLLISIONS, TIMING_FAULT)
        missing = [name for name in required if name not in topics]
        if missing:
            raise ValueError(f"{bag}: missing topics {missing}")

        phase_events = ([json.loads(message.data)
                         for _, message in analysis._messages(
                             connection, topics, PHASE)]
                        if PHASE in topics else [])
        starts = [event for event in phase_events
                  if event.get("event") == "phase_start"]
        ends = [event for event in phase_events
                if event.get("event") == "phase_end"]
        has_phase = len(starts) == 1 and len(ends) == 1
        if has_phase:
            start_event, end_event = starts[0], ends[0]
            start_ns = int(start_event["wall_time_ns"])
        else:
            start_event, end_event = {}, {}
            start_ns = 0

        packet_rows = []
        for _, message in analysis._messages(connection, topics, PACKET):
            packet = json.loads(message.data)
            stamp_ns = packet.get("bridge_receive_ros_stamp_ns")
            values = np.asarray([
                packet.get(f"simulator_position_{axis}", math.nan)
                for axis in "xyz"
            ] + [
                packet.get(f"simulator_orientation_euler_{axis}", math.nan)
                for axis in "xyz"
            ] + [
                packet.get(f"simulator_linear_velocity_{axis}", math.nan)
                for axis in "xyz"
            ] + [
                packet.get(f"simulator_angular_velocity_{axis}", math.nan)
                for axis in "xyz"
            ] + [
                packet.get("sent_steering_norm", math.nan),
                packet.get("sent_throttle_norm", math.nan),
                packet.get("simulator_feedback_steering_norm", math.nan),
                packet.get("simulator_feedback_throttle_norm", math.nan),
            ], dtype=np.float64)
            if isinstance(stamp_ns, int) and np.isfinite(values).all():
                packet_rows.append((stamp_ns, int(packet.get("packet_sequence", -1)),
                                    values[:12], values[12:]))
        packet_rows.sort(key=lambda row: (row[0], row[1]))
        if len(packet_rows) < 100:
            raise ValueError(f"{bag}: too few valid simulator packets")
        packet_ns = np.asarray([row[0] for row in packet_rows], dtype=np.int64)
        packet_state = np.stack([row[2] for row in packet_rows])
        packet_aux = np.stack([row[3] for row in packet_rows])
        packet_sequence = np.asarray([row[1] for row in packet_rows], dtype=np.int64)
        if not has_phase:
            start_ns = int(packet_ns[0])
        t_s = (packet_ns - start_ns) / 1e9
        phase_duration_s = (float(start_event["phase_duration_s"])
                            if has_phase else float(t_s[-1]))
        keep = ((t_s >= -0.10) & (t_s <= phase_duration_s + 0.10)
                if has_phase else np.ones(len(t_s), dtype=bool))
        t_s, packet_state, packet_aux, packet_sequence = (
            t_s[keep], packet_state[keep], packet_aux[keep], packet_sequence[keep])
        order = np.argsort(t_s, kind="stable")
        t_s, packet_state, packet_aux, packet_sequence = (
            t_s[order], packet_state[order], packet_aux[order], packet_sequence[order])
        if len(t_s) < 100 or np.any(np.diff(t_s) <= 0.0):
            raise ValueError(f"{bag}: packet timeline is incomplete or duplicated")

        def read_command(topic: str) -> tuple[np.ndarray, np.ndarray]:
            if topic not in topics:
                return np.asarray([], dtype=float), np.asarray([], dtype=float)
            rows = []
            for stamp_ns, message in analysis._messages(connection, topics, topic):
                relative = (int(stamp_ns) - start_ns) / 1e9
                if (not has_phase or
                        -0.10 <= relative <= float(start_event["phase_duration_s"]) + 0.10):
                    rows.append((relative, float(message.data)))
            rows.sort()
            if len(rows) < 100:
                raise ValueError(f"{bag}: too few command messages on {topic}")
            return (np.asarray([row[0] for row in rows]),
                    np.asarray([row[1] for row in rows]))

        collision_values = [int(message.data) for _, message in
                            analysis._messages(connection, topics, COLLISIONS)]
        fault_values = [bool(message.data) for _, message in
                        analysis._messages(connection, topics, TIMING_FAULT)]
        steering_t, steering_v = read_command(STEERING_COMMAND)
        throttle_t, throttle_v = read_command(THROTTLE_COMMAND)
        return {
            "bag": str(bag), "phase_start_ns": start_ns,
            "has_phase": has_phase,
            "profile": start_event.get("profile"),
            "seed": start_event.get("seed"),
            "plan": start_event.get("race_domain_command_plan"),
            "phase_duration_s": phase_duration_s,
            "phase_end_status": end_event.get("status"),
            "phase_end_reason": end_event.get("reason"),
            "t_s": t_s, "state": packet_state, "packet_aux": packet_aux,
            "packet_sequence": packet_sequence,
            "steering_t": steering_t, "steering_v": steering_v,
            "throttle_t": throttle_t, "throttle_v": throttle_v,
            "collisions_min_max": [min(collision_values), max(collision_values)],
            "timing_fault_count": sum(fault_values),
        }
    finally:
        connection.close()


def _interp(t_src: np.ndarray, values: np.ndarray, t_target: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_target, t_src, values[:, column])
                            for column in range(values.shape[1])])


def compare(reference_bag: Path, source_bag: Path) -> dict[str, Any]:
    reference, source = _read_run(reference_bag), _read_run(source_bag)
    if reference["has_phase"] and source["has_phase"]:
        if (reference["profile"] != source["profile"]
                or reference["seed"] != source["seed"]):
            raise ValueError("runs do not use the same profile and random seed")
        if reference["plan"] != source["plan"]:
            raise ValueError("the source and pinned runs have different command plans")
        if (reference["phase_end_status"] != "complete"
                or source["phase_end_status"] != "complete"):
            raise ValueError("both phases must complete before parity scoring")
        end_s = min(float(reference["t_s"][-1]), float(source["t_s"][-1]),
                    reference["phase_duration_s"], source["phase_duration_s"])
        grid = np.arange(0.0, end_s, PACKET_DT_S)
        if len(grid) < 100:
            raise ValueError("too little overlapping phase time")
        r_state = _interp(reference["t_s"], reference["state"], grid)
        s_state = _interp(source["t_s"], source["state"], grid)
        r_position = r_state[:, 0:3] - r_state[0, 0:3]
        s_position = s_state[:, 0:3] - s_state[0, 0:3]
        position_delta = s_position - r_position
        r_euler = np.unwrap(r_state[:, 3:6], axis=0)
        s_euler = np.unwrap(s_state[:, 3:6], axis=0)
        angle_delta = np.arctan2(np.sin(s_euler - r_euler),
                                 np.cos(s_euler - r_euler))
        velocity_delta = s_state[:, 6:9] - r_state[:, 6:9]
        angular_delta = s_state[:, 9:12] - r_state[:, 9:12]
        r_steer = np.interp(grid, reference["steering_t"], reference["steering_v"])
        s_steer = np.interp(grid, source["steering_t"], source["steering_v"])
        r_throttle = np.interp(grid, reference["throttle_t"], reference["throttle_v"])
        s_throttle = np.interp(grid, source["throttle_t"], source["throttle_v"])
    else:
        end_s = None
        grid = np.asarray([], dtype=float)
        r_state = s_state = np.empty((0, 12), dtype=float)
        position_delta = angle_delta = velocity_delta = angular_delta = r_state
        r_steer = s_steer = r_throttle = s_throttle = np.asarray([], dtype=float)

    # Packet sequence is generated by the simulator itself. It is a stronger
    # alignment key than ROS receipt time when comparing independent runs.
    ref_sequence_rows = {int(seq): row for row, seq in enumerate(reference["packet_sequence"])}
    src_sequence_rows = {int(seq): row for row, seq in enumerate(source["packet_sequence"])}
    common_sequences = np.asarray(sorted(ref_sequence_rows.keys()
                                         & src_sequence_rows.keys()), dtype=np.int64)
    if len(common_sequences) < 100:
        raise ValueError("too few shared simulator packet sequences")
    ref_seq = np.stack([reference["state"][ref_sequence_rows[int(seq)]]
                        for seq in common_sequences])
    src_seq = np.stack([source["state"][src_sequence_rows[int(seq)]]
                        for seq in common_sequences])
    ref_packet_aux = np.stack([reference["packet_aux"][ref_sequence_rows[int(seq)]]
                               for seq in common_sequences])
    src_packet_aux = np.stack([source["packet_aux"][src_sequence_rows[int(seq)]]
                               for seq in common_sequences])
    sent_steering_error = src_packet_aux[:, 0] - ref_packet_aux[:, 0]
    sent_throttle_error = src_packet_aux[:, 1] - ref_packet_aux[:, 1]
    steering_command_exact = bool(np.allclose(
        sent_steering_error, 0.0, rtol=0.0, atol=1.0e-6))
    throttle_command_exact = bool(np.allclose(
        sent_throttle_error, 0.0, rtol=0.0, atol=1.0e-6))
    commands_exact = steering_command_exact and throttle_command_exact
    no_collision_or_timing_fault = (
        reference["collisions_min_max"] == [0, 0]
        and source["collisions_min_max"] == [0, 0]
        and reference["timing_fault_count"] == 0
        and source["timing_fault_count"] == 0
    )
    seq_position_delta = ((src_seq[:, 0:3] - src_seq[0, 0:3])
                          - (ref_seq[:, 0:3] - ref_seq[0, 0:3]))
    seq_angle_delta = np.arctan2(
        np.sin(src_seq[:, 3:6] - ref_seq[:, 3:6]),
        np.cos(src_seq[:, 3:6] - ref_seq[:, 3:6]))
    seq_velocity_delta = src_seq[:, 6:9] - ref_seq[:, 6:9]
    seq_angular_delta = src_seq[:, 9:12] - ref_seq[:, 9:12]

    def metrics(error: np.ndarray) -> dict[str, Any]:
        magnitude = np.linalg.norm(error, axis=1)
        return {"rmse_by_axis": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
                "vector_rmse": float(np.sqrt(np.mean(magnitude ** 2))),
                "vector_p95": float(np.quantile(magnitude, 0.95)),
                "vector_max": float(np.max(magnitude))}

    return {
        "schema_version": 1,
        "reference_bag": str(reference_bag),
        "source_bag": str(source_bag),
        "profile": reference["profile"], "seed": reference["seed"],
        "compared_duration_s": (float(grid[-1]) if len(grid) else None),
        "sample_count": int(len(grid)),
        "same_race_domain_command_plan": (
            reference["plan"] == source["plan"]
            if reference["has_phase"] and source["has_phase"] else None),
        "reference_collision_min_max": reference["collisions_min_max"],
        "source_collision_min_max": source["collisions_min_max"],
        "reference_timing_fault_count": reference["timing_fault_count"],
        "source_timing_fault_count": source["timing_fault_count"],
        "command_difference": {
            "steering_rad_rmse": (float(np.sqrt(np.mean((s_steer - r_steer) ** 2)))
                                  if len(grid) else None),
            "throttle_norm_rmse": (float(np.sqrt(np.mean((s_throttle - r_throttle) ** 2)))
                                   if len(grid) else None),
            "steering_exact_match_fraction": (float(np.mean(np.isclose(
                s_steer, r_steer, rtol=0.0, atol=1.0e-6))) if len(grid) else None),
            "throttle_exact_match_fraction": (float(np.mean(np.isclose(
                s_throttle, r_throttle, rtol=0.0, atol=1.0e-6))) if len(grid) else None),
        },
        "packet_sequence_aligned": {
            "first_sequence": int(common_sequences[0]),
            "last_sequence": int(common_sequences[-1]),
            "sample_count": int(len(common_sequences)),
            "command_feedback_norm_rmse": {
                "steering_sent": float(np.sqrt(np.nanmean(
                    (src_packet_aux[:, 0] - ref_packet_aux[:, 0]) ** 2))),
                "throttle_sent": float(np.sqrt(np.nanmean(
                    (src_packet_aux[:, 1] - ref_packet_aux[:, 1]) ** 2))),
                "steering_feedback": float(np.sqrt(np.nanmean(
                    (src_packet_aux[:, 2] - ref_packet_aux[:, 2]) ** 2))),
                "throttle_feedback": float(np.sqrt(np.nanmean(
                    (src_packet_aux[:, 3] - ref_packet_aux[:, 3]) ** 2))),
            },
            "input_command_match": {
                "steering_exact": steering_command_exact,
                "throttle_exact": throttle_command_exact,
                "steering_rmse_norm": float(np.sqrt(np.mean(
                    sent_steering_error ** 2))),
                "throttle_rmse_norm": float(np.sqrt(np.mean(
                    sent_throttle_error ** 2))),
            },
            "public_motion_difference": {
                "position_relative_to_first_common_packet_m": metrics(seq_position_delta),
                "orientation_euler_rad": metrics(seq_angle_delta),
                "linear_velocity_mps": metrics(seq_velocity_delta),
                "angular_velocity_rps": metrics(seq_angular_delta),
            },
        },
        "motion_parity_eligible": bool(
            commands_exact and no_collision_or_timing_fault),
        "public_motion_difference": ({
            "position_relative_to_phase_start_m": metrics(position_delta),
            "orientation_euler_rad": metrics(angle_delta),
            "linear_velocity_mps": metrics(velocity_delta),
            "angular_velocity_rps": metrics(angular_delta),
            "speed_norm_rmse_mps": float(np.sqrt(np.mean((
                np.linalg.norm(s_state[:, 6:9], axis=1)
                - np.linalg.norm(r_state[:, 6:9], axis=1)) ** 2))),
        } if len(grid) else None),
        "acceptance_note": (
            "Public-motion comparison is eligible only when the packet-sequence "
            "aligned sent steering/throttle exactly match and neither run has a "
            "collision or bridge timing fault. Otherwise the run diagnoses the "
            "replay harness, not simulator physics."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_bag", type=Path)
    parser.add_argument("source_bag", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.reference_bag, args.source_bag)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
