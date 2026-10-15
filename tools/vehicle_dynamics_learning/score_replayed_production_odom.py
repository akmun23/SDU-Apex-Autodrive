#!/usr/bin/env python3
"""Score exact production-odometry replay against Unity bridge truth.

The original bag is read-only. Production replay consumes encoder/IMU streams
only. Twist and pose references come from simulator rigid-state fields in
bridge packet timing. The simulator's documented ``/autodrive/roboracer_1/odom``
development oracle remains available for explicit source-parity checks.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis
from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning.score_practice_odom_replay import (
    _align_diagnostics,
    _read_diagnostics,
)


TRUTH_SOURCE = f"{analysis.PACKET_TIMING}: simulator rigid-state fields"
REPLAY_TOPIC = "/replayed_odom"


def _yaw(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def _read_pose(path: Path, topic: str) -> dict[int, tuple[float, float, float]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        if topic not in topics:
            raise ValueError(f"bag is missing pose topic {topic}: {path}")
        result = {}
        for _, message in analysis._messages(connection, topics, topic):
            stamp = analysis._stamp_ns(message.header.stamp)
            if stamp <= 0:
                continue
            pose = message.pose.pose
            values = (float(pose.position.x), float(pose.position.y),
                      _yaw(pose.orientation))
            if all(math.isfinite(value) for value in values):
                result[stamp] = values
        return result
    finally:
        connection.close()


def _axis_metrics(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "rmse": None, "bias": None,
                "p95_abs_error": None}
    return {
        "samples": int(len(error)),
        "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "p95_abs_error": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
    }


def _prefix_stream_stats(raw_bag: Path, cutoff_receipt_ns: int
                         ) -> dict[str, tuple[float, float, float]]:
    """Measure input cadence only before a collision-triggered shutdown."""
    connection = sqlite3.connect(raw_bag.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        result: dict[str, tuple[float, float, float]] = {}
        for topic in body.STREAM_TOPICS:
            if topic not in topics:
                result[topic] = (0.0, math.inf, math.inf)
                continue
            receipts = [int(receipt_ns) for receipt_ns, _ in
                        analysis._messages(connection, topics, topic)
                        if int(receipt_ns) < cutoff_receipt_ns]
            rate, p95_gap_ms, max_gap_ms = analysis._rate(receipts)
            result[topic] = (
                float(rate or 0.0),
                float(p95_gap_ms if p95_gap_ms is not None else math.inf),
                float(max_gap_ms if max_gap_ms is not None else math.inf),
            )
        return result
    finally:
        connection.close()


def score(raw_bag: Path, replay_bag: Path, output: Path,
          state_topic: str = REPLAY_TOPIC,
          diagnostics_topic: str = "/replayed_odom_diagnostics",
          allow_collision_prefix: bool = False,
          fixed_25ms_packet_grid: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite odometry score: {output}")
    capture = body.load_capture(raw_bag)
    # OpenPlane r03 completed without collisions, aborts, or timing faults,
    # but some paired speed-match phases were deliberately marked invalid.
    # load_capture includes only valid phases; score that explicit subset and
    # do not pretend the failed speed-match phases are part of the evaluation.
    collision_cutoff_receipt_ns = None
    prefix_timing_faults = None
    prefix_stream_stats = None
    if allow_collision_prefix:
        if (capture.has_phase_markers or capture.collision_count_start != 0
                or capture.collision_count_end != 1 or not capture.sequences):
            raise ValueError(
                "collision-prefix scoring requires an unphased race capture "
                "with exactly one collision and available odometry samples")
        connection = sqlite3.connect(raw_bag.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            topics = analysis._topic_map(connection)
            collision_topic = "/autodrive/roboracer_1/collision_count"
            if collision_topic not in topics:
                raise ValueError("collision-prefix scoring requires collision_count")
            collision_rows = analysis._messages(connection, topics, collision_topic)
            initial = None
            for receipt_ns, message in collision_rows:
                value = int(message.data)
                if initial is None:
                    initial = value
                elif value > initial:
                    collision_cutoff_receipt_ns = int(receipt_ns)
                    break
            timing_fault_topic = analysis.TIMING_FAULT
            if timing_fault_topic not in topics:
                raise ValueError("collision-prefix scoring requires bridge timing status")
            prefix_timing_faults = sum(
                int(message.data) != 0
                for receipt_ns, message in analysis._messages(
                    connection, topics, timing_fault_topic)
                if int(receipt_ns) < collision_cutoff_receipt_ns
            ) if collision_cutoff_receipt_ns is not None else 0
        finally:
            connection.close()
        if collision_cutoff_receipt_ns is None:
            raise ValueError("no collision transition found to delimit the scored prefix")
        if prefix_timing_faults:
            raise ValueError(
                f"collision-prefix scoring rejected {prefix_timing_faults} "
                "bridge timing fault(s) before collision")
        prefix_stream_stats = _prefix_stream_stats(
            raw_bag, collision_cutoff_receipt_ns)
        kept_sequences = []
        kept_labels = []
        for index, sequence in enumerate(capture.sequences):
            prefix = tuple(sample for sample in sequence
                           if sample.receipt_ns < collision_cutoff_receipt_ns)
            if len(prefix) >= 2:
                kept_sequences.append(prefix)
                kept_labels.append(
                    capture.sequence_labels[index]
                    if index < len(capture.sequence_labels)
                    else f"sequence_{index}")
        capture = replace(capture, sequences=tuple(kept_sequences),
                          sequence_labels=tuple(kept_labels))
        if not capture.sequences:
            raise ValueError("collision occurred before a scoreable sensor prefix")
    elif (capture.aborted or capture.collision_count_start != 0
          or capture.collision_count_end != capture.collision_count_start
          or capture.timing_faults or not capture.sequences):
        raise ValueError("capture failed abort/collision/timing quality gate")
    quality_stats = (prefix_stream_stats if prefix_stream_stats is not None else
                     capture.phase_stream_stats if capture.phase_stream_stats else
                     capture.stream_stats)
    bad_streams = [
        name for name, (rate, p95, gap) in quality_stats.items()
        if rate < 38.0 or p95 > 35.0
        or gap > (120.0 if name in body.COMMAND_STREAM_TOPICS else 60.0)
    ]
    if bad_streams and not fixed_25ms_packet_grid:
        raise ValueError(f"capture streams failed timing gate: {bad_streams}")
    if fixed_25ms_packet_grid:
        timing_faults_in_scored_scope = (
            prefix_timing_faults if allow_collision_prefix else capture.timing_faults
        )
        if timing_faults_in_scored_scope or not capture.sequences:
            raise ValueError(
                "fixed-25-ms scoring still requires a fault-free capture with valid sequences")
        for sequence in capture.sequences:
            packet_ids = np.asarray(
                [sample.packet_sequence for sample in sequence], dtype=np.int64)
            if (len(packet_ids) < 2 or np.any(packet_ids < 0) or
                    np.any(np.diff(packet_ids) != 1)):
                raise ValueError(
                    "fixed-25-ms scoring requires contiguous packet IDs in every scored sequence")
    capture, coverage = body._attach_feature_states(capture, replay_bag,
                                                     state_topic)

    # Valid phase context overlaps by design; count each source stamp once.
    diagnostic_rows = _read_diagnostics(replay_bag, diagnostics_topic)
    diagnostic_stamps = sorted(diagnostic_rows)

    def has_diagnostic(stamp: int) -> bool:
        index = bisect.bisect_left(diagnostic_stamps, stamp)
        candidates = [i for i in (index - 1, index)
                      if 0 <= i < len(diagnostic_stamps)]
        return bool(candidates) and min(
            abs(diagnostic_stamps[i] - stamp) for i in candidates
        ) <= 2_000_000

    samples_by_stamp = {
        sample.source_stamp_ns: sample
        for sequence in capture.sequences for sample in sequence
    }
    all_samples = [samples_by_stamp[key] for key in sorted(samples_by_stamp)]
    # Production replay may omit a small number of stamps when the sensor
    # streams do not produce a complete observer packet.  The coverage gate
    # above still requires >=98%; score the intersection of finite observer
    # states and same-packet diagnostics and expose both unmatched counts.
    state_matched = [sample for sample in all_samples
                     if sample.feature_state is not None and
                     np.isfinite(sample.feature_state).all()]
    samples = [sample for sample in state_matched
               if has_diagnostic(sample.source_stamp_ns)]
    if not samples:
        raise ValueError("production replay has no matched finite states/diagnostics")
    if any(sample.simulator_rigid_state is None for sample in samples):
        raise ValueError("capture is missing Unity bridge rigid-state truth")
    truth = body.simulator_truth_rear_axle_twist(
        np.stack([sample.simulator_rigid_state for sample in samples]))
    simulator_odom_oracle = np.stack([sample.state for sample in samples])
    oracle_twist_parity = _axis_metrics(simulator_odom_oracle - truth)
    truth_by_stamp = {
        sample.source_stamp_ns: state
        for sample, state in zip(samples, truth)
    }
    estimate = np.stack([sample.feature_state for sample in samples])
    if not np.isfinite(estimate).all():
        raise ValueError("production replay contains a missing/nonfinite state")
    error = estimate - truth
    speed = np.hypot(truth[:, 0], truth[:, 1])
    steering = np.asarray([abs(float(sample.actuators[0])) for sample in samples])
    diagnostics = _align_diagnostics(
        diagnostic_rows, [sample.source_stamp_ns for sample in samples],
        diagnostics_topic)
    if diagnostics.shape[1] != 42:
        raise ValueError(f"expected production diagnostic v7/42, got {diagnostics.shape}")
    wheel_pair = np.stack([
        (sample.rear_wheel_surface_mps
         if sample.rear_wheel_surface_mps is not None else (np.nan, np.nan))
        for sample in samples
    ]).astype(np.float64)
    wheel_pair_count = np.isfinite(wheel_pair).sum(axis=1)
    wheel_pair_mean = np.divide(
        np.nansum(wheel_pair, axis=1), wheel_pair_count,
        out=np.full(len(wheel_pair_count), np.nan, dtype=np.float64),
        where=wheel_pair_count > 0)

    def scalar_error_metrics(values: np.ndarray,
                             mask: np.ndarray | None = None) -> dict[str, Any]:
        valid = np.isfinite(values)
        if mask is not None:
            valid &= mask
        if not np.any(valid):
            return {"samples": 0, "rmse": None, "bias": None,
                    "p95_abs_error": None}
        return _axis_metrics((values[valid] - truth[valid, 0])[:, None])

    diagnostic_speed = {
        "encoder_derived_pair_mean_vs_truth_body_u": scalar_error_metrics(
            wheel_pair_mean),
        "production_raw_rolling_wheel_speed_vs_truth_body_u":
            scalar_error_metrics(diagnostics[:, 3]),
        "production_mapped_wheel_speed_vs_truth_body_u":
            scalar_error_metrics(diagnostics[:, 4]),
        "production_packet_wheel_speed_vs_truth_body_u":
            scalar_error_metrics(diagnostics[:, 25]),
        "production_pre_wheel_prediction_vs_truth_body_u":
            scalar_error_metrics(diagnostics[:, 40]),
        "production_final_speed_vs_truth_body_u":
            scalar_error_metrics(diagnostics[:, 6]),
        "observer_state_fractions": {
            "wheel_update_used": float(np.mean(diagnostics[:, 12] > 0.5)),
            "turn_mode": float(np.mean(diagnostics[:, 13] > 0.5)),
            "timing_degraded": float(np.mean(diagnostics[:, 15] > 0.5)),
            "sensor_outlier": float(np.mean(diagnostics[:, 21] > 0.5)),
            "wheel_burst_rejected": float(np.mean(diagnostics[:, 26] > 0.5)),
        },
        "interpretation": (
            "Encoder-derived wheel speeds are compared with simulator body-u, "
            "not independent physical wheel-speed truth. Their difference is "
            "consistent with slip and/or signal/conversion effects and cannot "
            "alone prove physical wheel spin."),
    }

    regions: dict[str, np.ndarray] = {
        "all_valid_phase_samples": np.ones(len(samples), dtype=bool),
        "speed_0_to_2_mps": speed < 2.0,
        "speed_2_to_4_mps": (speed >= 2.0) & (speed < 4.0),
        "speed_4_to_6_mps": (speed >= 4.0) & (speed < 6.0),
        "speed_6_to_8_mps": (speed >= 6.0) & (speed < 8.0),
        "speed_8plus_mps": speed >= 8.0,
        "speed_8_to_12_mps": (speed >= 8.0) & (speed <= 12.0),
        "speed_over_12_mps": speed > 12.0,
        "steering_ge_0p30_rad": steering >= 0.30,
        "steering_ge_0p40_rad": steering >= 0.40,
        "steering_ge_0p42_rad": steering >= 0.42,
        "high_speed_and_steering": (speed >= 6.0) & (steering >= 0.30),
        "high_speed_and_steering_ge_0p40": (speed >= 6.0) & (steering >= 0.40),
    }
    abs_truth_yaw_rate = np.abs(truth[:, 2])
    regions.update({
        "high_yaw_rate_ge_2p5_radps": abs_truth_yaw_rate >= 2.5,
        "high_yaw_rate_ge_2p5_speed_le_12":
            (abs_truth_yaw_rate >= 2.5) & (speed <= 12.0),
        "high_yaw_rate_ge_2p5_negative_body_u":
            (abs_truth_yaw_rate >= 2.5) & (truth[:, 0] < 0.0),
    })
    region_metrics = {
        name: _axis_metrics(error[mask])
        for name, mask in regions.items() if np.any(mask)
    }

    trace_signals = {
        "encoder_pair_mean": wheel_pair_mean,
        "production_raw_rolling_wheel": diagnostics[:, 3],
        "production_mapped_wheel": diagnostics[:, 4],
        "production_packet_wheel": diagnostics[:, 25],
        "production_pre_wheel_prediction": diagnostics[:, 40],
        "production_final_speed": diagnostics[:, 6],
    }
    detailed_regions: dict[str, np.ndarray] = {
        "all": np.ones(len(samples), dtype=bool),
    }
    for name, edges, values in (
            ("speed_mps", (0.0, 2.0, 4.0, 6.0, 8.0, np.inf), speed),
            ("abs_steering_rad", (0.0, 0.10, 0.20, 0.30, 0.40, np.inf),
             steering),
            ("throttle_command_norm",
             (0.0, 0.10, 0.25, 0.40, 0.50, np.inf),
             np.asarray([float(sample.actuators[3]) for sample in samples]))):
        for lower, upper in zip(edges[:-1], edges[1:]):
            detailed_regions[f"{name}_{lower:g}_{upper:g}"] = (
                (values >= lower) & (values < upper))
    trace_by_regime = {
        region_name: {
            "samples": int(np.count_nonzero(mask)),
            "signals_vs_truth_body_u": {
                signal_name: scalar_error_metrics(values, mask)
                for signal_name, values in trace_signals.items()},
        }
        for region_name, mask in detailed_regions.items()
        if np.any(mask)
    }

    truth_poses = {
        sample.source_stamp_ns: tuple(float(value) for value in
                                      sample.simulator_pose_xyyaw)
        for sample in samples
        if sample.simulator_pose_xyyaw is not None
        and np.isfinite(sample.simulator_pose_xyyaw).all()
    }
    oracle_pose_pairs = [
        (sample.pose_xyyaw, sample.simulator_pose_xyyaw)
        for sample in samples
        if sample.pose_xyyaw is not None
        and sample.simulator_pose_xyyaw is not None
        and np.isfinite(sample.pose_xyyaw).all()
        and np.isfinite(sample.simulator_pose_xyyaw).all()
    ]
    if oracle_pose_pairs:
        oracle_poses = np.stack([pair[0] for pair in oracle_pose_pairs])
        rigid_packet_poses = np.stack([pair[1] for pair in oracle_pose_pairs])
        oracle_pose_xy_error = np.linalg.norm(
            oracle_poses[:, :2] - rigid_packet_poses[:, :2], axis=1)
        oracle_pose_yaw_error = np.arctan2(
            np.sin(oracle_poses[:, 2] - rigid_packet_poses[:, 2]),
            np.cos(oracle_poses[:, 2] - rigid_packet_poses[:, 2]))
        oracle_pose_parity = {
            "samples": int(len(oracle_pose_pairs)),
            "direct_xy_rmse_m": float(np.sqrt(np.mean(oracle_pose_xy_error ** 2))),
            "direct_xy_p95_m": float(np.quantile(oracle_pose_xy_error, 0.95)),
            "direct_yaw_rmse_rad": float(np.sqrt(np.mean(oracle_pose_yaw_error ** 2))),
            "direct_yaw_p95_abs_rad": float(
                np.quantile(np.abs(oracle_pose_yaw_error), 0.95)),
        }
    else:
        oracle_pose_parity = {"samples": 0}
    replay_poses = _read_pose(replay_bag, state_topic)
    pose_stamps = {
        sample.source_stamp_ns for sample in samples
        if sample.source_stamp_ns in truth_poses
        and sample.source_stamp_ns in replay_poses
    }
    if len(pose_stamps) < 0.98 * len(samples):
        raise ValueError(
            f"relative pose coverage is only {len(pose_stamps)}/{len(samples)}")
    sequence_pose_metrics = []
    relative_position_by_regime: dict[str, list[np.ndarray]] = {
        "all": [], "high_yaw_rate_ge_2p5_radps": [],
        "high_yaw_rate_ge_2p5_speed_le_12": [],
        "high_yaw_rate_ge_2p5_negative_body_u": [],
    }
    for sequence_index, sequence in enumerate(capture.sequences):
        stamps = [sample.source_stamp_ns for sample in sequence
                  if sample.source_stamp_ns in pose_stamps]
        if len(stamps) < 2:
            continue
        first = stamps[0]
        tx0, ty0, tyaw0 = truth_poses[first]
        ox0, oy0, oyaw0 = replay_poses[first]
        yaw_offset = math.atan2(math.sin(tyaw0 - oyaw0),
                                math.cos(tyaw0 - oyaw0))
        cosine, sine = math.cos(yaw_offset), math.sin(yaw_offset)
        position_errors = []
        heading_errors = []
        for stamp in stamps:
            tx, ty, tyaw = truth_poses[stamp]
            ox, oy, oyaw = replay_poses[stamp]
            dx, dy = ox - ox0, oy - oy0
            aligned_x = tx0 + cosine * dx - sine * dy
            aligned_y = ty0 + sine * dx + cosine * dy
            position_errors.append(math.hypot(aligned_x - tx, aligned_y - ty))
            heading_errors.append(math.atan2(
                math.sin((oyaw + yaw_offset) - tyaw),
                math.cos((oyaw + yaw_offset) - tyaw)))
        position_errors = np.asarray(position_errors)
        heading_errors = np.asarray(heading_errors)
        sequence_truth = np.stack([truth_by_stamp[stamp] for stamp in stamps])
        sequence_speed = np.hypot(sequence_truth[:, 0], sequence_truth[:, 1])
        sequence_abs_yaw = np.abs(sequence_truth[:, 2])
        pose_regime_masks = {
            "all": np.ones(len(stamps), dtype=bool),
            "high_yaw_rate_ge_2p5_radps": sequence_abs_yaw >= 2.5,
            "high_yaw_rate_ge_2p5_speed_le_12":
                (sequence_abs_yaw >= 2.5) & (sequence_speed <= 12.0),
            "high_yaw_rate_ge_2p5_negative_body_u":
                (sequence_abs_yaw >= 2.5) & (sequence_truth[:, 0] < 0.0),
        }
        for regime_name, regime_mask in pose_regime_masks.items():
            if np.any(regime_mask):
                relative_position_by_regime[regime_name].append(
                    position_errors[regime_mask])
        sequence_pose_metrics.append({
            "sequence_index": sequence_index,
            "matched_samples": len(stamps),
            "position_error_m": {
                "rmse": float(np.sqrt(np.mean(position_errors ** 2))),
                "p95": float(np.quantile(position_errors, 0.95)),
                "max": float(np.max(position_errors)),
                "endpoint": float(position_errors[-1]),
            },
            "heading_error_rad": {
                "rmse": float(np.sqrt(np.mean(heading_errors ** 2))),
                "p95_abs": float(np.quantile(np.abs(heading_errors), 0.95)),
                "max_abs": float(np.max(np.abs(heading_errors))),
            },
        })
    if not sequence_pose_metrics:
        raise ValueError("no capture sequences have matched pose samples")
    continuous_run_pose = None
    if not capture.has_phase_markers and capture.final_lap_count is not None:
        # Practice captures are continuous across lap-count transitions. Keep a
        # single start alignment here so accumulated odometry drift is visible;
        # the per-sequence scores above intentionally reset alignment each lap.
        run_stamps = sorted(pose_stamps)
        first_stamp = run_stamps[0]
        tx0, ty0, tyaw0 = truth_poses[first_stamp]
        ox0, oy0, oyaw0 = replay_poses[first_stamp]
        yaw_offset = math.atan2(math.sin(tyaw0 - oyaw0),
                                math.cos(tyaw0 - oyaw0))
        cosine, sine = math.cos(yaw_offset), math.sin(yaw_offset)
        run_position_errors = []
        run_heading_errors = []
        position_error_by_stamp = {}
        heading_error_by_stamp = {}
        for stamp in run_stamps:
            tx, ty, tyaw = truth_poses[stamp]
            ox, oy, oyaw = replay_poses[stamp]
            dx, dy = ox - ox0, oy - oy0
            aligned_x = tx0 + cosine * dx - sine * dy
            aligned_y = ty0 + sine * dx + cosine * dy
            position_error = math.hypot(aligned_x - tx, aligned_y - ty)
            heading_error = math.atan2(
                math.sin((oyaw + yaw_offset) - tyaw),
                math.cos((oyaw + yaw_offset) - tyaw))
            run_position_errors.append(position_error)
            run_heading_errors.append(heading_error)
            position_error_by_stamp[stamp] = position_error
            heading_error_by_stamp[stamp] = heading_error
        run_position_errors = np.asarray(run_position_errors)
        run_heading_errors = np.asarray(run_heading_errors)
        per_lap = []
        for sequence_index, sequence in enumerate(capture.sequences):
            stamps = [sample.source_stamp_ns for sample in sequence
                      if sample.source_stamp_ns in position_error_by_stamp]
            if not stamps:
                continue
            errors = np.asarray([position_error_by_stamp[stamp]
                                 for stamp in stamps])
            heading_errors = np.asarray([heading_error_by_stamp[stamp]
                                          for stamp in stamps])
            per_lap.append({
                "sequence_index": sequence_index,
                "label": (capture.sequence_labels[sequence_index]
                          if sequence_index < len(capture.sequence_labels)
                          else f"sequence_{sequence_index}"),
                "samples": len(stamps),
                "position_error_m": {
                    "rmse": float(np.sqrt(np.mean(errors ** 2))),
                    "p95": float(np.quantile(errors, 0.95)),
                    "max": float(np.max(errors)),
                    "endpoint": float(errors[-1]),
                },
                "heading_error_rad": {
                    "rmse": float(np.sqrt(np.mean(heading_errors ** 2))),
                    "p95_abs": float(np.quantile(np.abs(heading_errors), 0.95)),
                    "max_abs": float(np.max(np.abs(heading_errors))),
                    "endpoint": float(heading_errors[-1]),
                },
            })
        continuous_run_pose = {
            "method": "one initial SE(2) alignment for the complete continuous run; no per-lap realignment",
            "samples": len(run_stamps),
            "position_error_m": {
                "rmse": float(np.sqrt(np.mean(run_position_errors ** 2))),
                "p95": float(np.quantile(run_position_errors, 0.95)),
                "max": float(np.max(run_position_errors)),
                "endpoint": float(run_position_errors[-1]),
            },
            "heading_error_rad": {
                "rmse": float(np.sqrt(np.mean(run_heading_errors ** 2))),
                "p95_abs": float(np.quantile(np.abs(run_heading_errors), 0.95)),
                "max_abs": float(np.max(np.abs(run_heading_errors))),
                "endpoint": float(run_heading_errors[-1]),
            },
            "per_lap_without_realignment": per_lap,
        }
    macro_position_rmse = [item["position_error_m"]["rmse"]
                           for item in sequence_pose_metrics]
    macro_heading_rmse = [item["heading_error_rad"]["rmse"]
                          for item in sequence_pose_metrics]
    relative_position_regime_metrics = {}
    for regime_name, chunks in relative_position_by_regime.items():
        if not chunks:
            continue
        errors = np.concatenate(chunks)
        sequence_rmse = [float(np.sqrt(np.mean(chunk ** 2))) for chunk in chunks]
        relative_position_regime_metrics[regime_name] = {
            "samples": int(len(errors)),
            "sequences": len(chunks),
            "sample_weighted_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "p95_m": float(np.quantile(errors, 0.95)),
            "max_m": float(np.max(errors)),
            "macro_sequence_rmse_m": float(np.mean(sequence_rmse)),
        }

    result = {
        "evaluation": "production sensor-odometry node replayed from encoder/IMU only; simulator truth is an offline label",
        "run_id": raw_bag.parents[1].name,
        "raw_bag": str(raw_bag.resolve()),
        "replayed_bag": str(replay_bag.resolve()),
        "production_replay_topic": state_topic,
        "diagnostics_topic": diagnostics_topic,
        "truth_source": TRUTH_SOURCE,
        "truth_twist_definition": (
            "Unity bridge body-frame COM [u,v,r] converted to rear-axle "
            "[u, v - 0.15532*r, r]; this scoring pass uses rigid-state fields, "
            "with the simulator /autodrive/roboracer_1/odom oracle checked separately"),
        "truth_pose_source": "simulator pose fields in bridge packet timing",
        "simulator_odom_oracle_parity": {
            "twist_rigid_packet_vs_autodrive_odom": oracle_twist_parity,
            "pose_packet_vs_autodrive_odom": oracle_pose_parity,
        },
        "capture_quality": {
            "aborted": capture.aborted,
            "collisions_start_end": [capture.collision_count_start,
                                     capture.collision_count_end],
            "collision_prefix_scored": allow_collision_prefix,
            "collision_cutoff_receipt_ns": collision_cutoff_receipt_ns,
            "timing_faults_before_collision": prefix_timing_faults,
            "timing_faults": capture.timing_faults,
            "valid_invalid_unscored_phases": [capture.valid_phase_count,
                                               capture.invalid_phase_count,
                                               capture.unscored_phase_count],
            "scored_scope": "valid phases only; invalid speed-match phases and unscored approach/settle phases excluded",
            "production_state_match_fraction": coverage,
            "production_state_matched_samples": len(state_matched),
            "production_joint_state_diagnostic_matched_samples": len(samples),
            "production_joint_state_diagnostic_match_fraction":
                len(samples) / len(all_samples),
            "production_state_unmatched_samples": len(all_samples) - len(state_matched),
            "production_diagnostic_unmatched_samples": len(state_matched) - len(samples),
            "timing_basis": (
                "25 ms per contiguous simulator packet ID; receipt-gap jitter is diagnostic only"
                if fixed_25ms_packet_grid else "strict measured receipt-cadence gate"),
            "receipt_cadence_gate_overridden": bool(fixed_25ms_packet_grid and bad_streams),
            "receipt_cadence_rejected_streams": bad_streams,
            "receipt_cadence_statistics_hz_p95_ms_max_ms": {
                name: {"rate_hz": values[0], "p95_gap_ms": values[1],
                       "max_gap_ms": values[2]}
                for name, values in quality_stats.items()},
        },
        "unique_source_stamped_samples": len(samples),
        "twist_error_order": ["u_mps", "v_rear_mps", "yaw_rate_radps"],
        "twist_overall": _axis_metrics(error),
        "twist_by_regime": region_metrics,
        "encoder_and_observer_trace": diagnostic_speed,
        "encoder_and_observer_trace_by_regime": trace_by_regime,
        "relative_pose": {
            "method": "independent initial SE(2) alignment per valid phase sequence; never integrate/alignment across reset-separated phases",
            "matched_samples": len(pose_stamps),
            "sequence_count": len(sequence_pose_metrics),
            "sequence_macro_position_rmse_m": {
                "mean": float(np.mean(macro_position_rmse)),
                "p95_across_sequences": float(np.quantile(macro_position_rmse, 0.95)),
                "max": float(np.max(macro_position_rmse)),
            },
            "sequence_macro_heading_rmse_rad": {
                "mean": float(np.mean(macro_heading_rmse)),
                "p95_across_sequences": float(np.quantile(macro_heading_rmse, 0.95)),
                "max": float(np.max(macro_heading_rmse)),
            },
            "per_sequence": sequence_pose_metrics,
            "after_initial_alignment_by_regime": relative_position_regime_metrics,
            "note": "Each valid phase is aligned independently because the capture contains reset-separated motion; the former one-alignment-across-all-phases score was invalid and must not be interpreted as odometry drift.",
            "legacy_single_alignment_metric_removed": True,
        },
        "continuous_run_pose_from_single_initial_alignment": continuous_run_pose,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_bag", type=Path)
    parser.add_argument("replayed_bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--state-topic", default=REPLAY_TOPIC,
        help="state topic to score (use /odom to score recorded production directly)",
    )
    parser.add_argument(
        "--diagnostics-topic", default="/replayed_odom_diagnostics",
        help="matching observer diagnostics topic",
    )
    parser.add_argument(
        "--allow-collision-prefix", action="store_true",
        help=("score only sensor samples received before the first collision; "
              "requires a collision-censored capture with no timing faults"),
    )
    parser.add_argument(
        "--fixed-25ms-packet-grid", action="store_true",
        help=("use the established 25 ms simulator-step interval for contiguous "
              "packet IDs; requires no bridge timing faults and rejects any "
              "sequence with a missing packet ID"),
    )
    args = parser.parse_args()
    result = score(args.raw_bag, args.replayed_bag, args.output,
                   args.state_topic, args.diagnostics_topic,
                   args.allow_collision_prefix,
                   args.fixed_25ms_packet_grid)
    print(json.dumps({"run_id": result["run_id"],
                      "samples": result["unique_source_stamped_samples"],
                      "twist_rmse": result["twist_overall"]["rmse"],
                      "relative_pose_sequence_macro": result["relative_pose"][
                          "sequence_macro_position_rmse_m"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
