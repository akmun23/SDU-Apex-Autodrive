#!/usr/bin/env python3
"""Compare sensor-odometry replay with production odom and simulator truth."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools import analyze_open_plane_dynamics as analysis
from tools.vehicle_dynamics_learning import evaluate_sensor_observer_practice as practice
from tools.vehicle_dynamics_learning import prepare_dataset


ODOM_RESIDUAL_LEGACY_FEATURE_NAMES = (
    "odom_u_mps", "odom_v_mps", "encoder_mean_100ms_mps",
    "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps", "sample_dt_s",
)
ODOM_RESIDUAL_FEATURE_NAMES = (
    *ODOM_RESIDUAL_LEGACY_FEATURE_NAMES[:-1],
    "sample_dt_s", "steering_feedback_rad", "throttle_feedback_norm",
    "steering_command_rad", "throttle_command_norm",
)


def _residual_feature_row(state: np.ndarray, sensor: np.ndarray,
                          feature_names: list[str]) -> np.ndarray:
    values = {
        "odom_u_mps": float(state[0]),
        "odom_v_mps": float(state[1]),
        "encoder_mean_100ms_mps": float(np.mean(sensor[2:4])),
        "imu_ax_mps2": float(sensor[4]),
        "imu_ay_mps2": float(sensor[5]),
        "imu_yaw_rate_rps": float(sensor[6]),
        "sample_dt_s": 0.025,
        "steering_feedback_rad": float(sensor[0]),
        "throttle_feedback_norm": float(sensor[1]),
        "steering_command_rad": float(sensor[7]),
        "throttle_command_norm": float(sensor[8]),
    }
    unknown = set(feature_names) - values.keys()
    if unknown:
        raise ValueError(f"unsupported residual feature(s): {sorted(unknown)}")
    return np.asarray([values[name] for name in feature_names], dtype=np.float64)


def _causal_history_features(capture: body.Capture,
                            states: dict[int, np.ndarray],
                            requested_stamps: list[int],
                            history_steps: int,
                            base_feature_names: list[str]) -> np.ndarray:
    """Return oldest-to-current source-grid features for requested samples."""
    if history_steps < 1:
        raise ValueError("residual model history_steps must be positive")
    samples_by_stamp = {}
    for sequence in capture.sequences:
        for sample in sequence:
            samples_by_stamp.setdefault(sample.source_stamp_ns, sample)
    ordered = sorted(samples_by_stamp.values(),
                     key=lambda sample: sample.source_stamp_ns)
    history: list[tuple[int, np.ndarray]] = []
    features_by_stamp: dict[int, np.ndarray] = {}
    previous_packet: int | None = None
    for sample in ordered:
        packet = int(sample.packet_sequence)
        state = states.get(sample.source_stamp_ns)
        sensor = prepare_dataset._sensor_frame(sample)
        if state is None or sensor is None:
            history.clear()
            previous_packet = None
            continue
        sensor = np.asarray(sensor, dtype=np.float64)
        row = _residual_feature_row(state, sensor, base_feature_names)
        if previous_packet is None or packet != previous_packet + 1:
            history.clear()
        history.append((packet, row))
        if len(history) > history_steps:
            history.pop(0)
        if len(history) == history_steps:
            features_by_stamp[sample.source_stamp_ns] = np.stack(
                [item[1] for item in history]).reshape(-1)
        previous_packet = packet
    missing = [stamp for stamp in requested_stamps
               if stamp not in features_by_stamp]
    if missing:
        raise ValueError(
            f"causal {history_steps}-sample feature history missing "
            f"{len(missing)}/{len(requested_stamps)} scored stamps")
    return np.stack([features_by_stamp[stamp] for stamp in requested_stamps])


def _metrics(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "rmse": None, "bias": None,
                "p95_abs": None}
    return {"samples": int(len(error)),
            "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "bias": np.mean(error, axis=0).tolist(),
            "p95_abs": np.quantile(np.abs(error), 0.95, axis=0).tolist()}


def _read_raw_encoder_angles(path: Path) -> dict[str, dict[int, float]]:
    """Read source-stamped cumulative encoder angles; never infer wheel truth."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        result: dict[str, dict[int, float]] = {}
        for topic in (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER):
            if topic not in topics:
                raise ValueError(f"{path} is missing encoder topic {topic}")
            topic_id, message_type = topics[topic]
            cls = get_message(message_type)
            rows: dict[int, float] = {}
            for _, payload in connection.execute(
                    "SELECT timestamp,data FROM messages WHERE topic_id=? "
                    "ORDER BY timestamp,id", (topic_id,)):
                message = deserialize_message(bytes(payload), cls)
                stamp_ns = analysis._stamp_ns(message.header.stamp)
                if stamp_ns is None or not message.position:
                    continue
                rows[stamp_ns] = float(message.position[0])
            result[topic] = rows
        return result
    finally:
        connection.close()


def _fixed25_raw_encoder_rates(
        capture: body.Capture, angles: dict[str, dict[int, float]],
        wanted_stamps: set[int]) -> tuple[list[int], np.ndarray]:
    """Derive angle rates on the fixed 25 ms grid for contiguous packets only."""
    left_topic, right_topic = analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER
    rates_by_stamp: dict[int, np.ndarray] = {}
    for sequence in capture.sequences:
        previous = None
        for sample in sequence:
            current_angles = [angles[left_topic].get(sample.source_stamp_ns),
                              angles[right_topic].get(sample.source_stamp_ns)]
            if (previous is not None and previous[0] + 1 == sample.packet_sequence
                    and all(value is not None for value in current_angles)
                    and all(value is not None for value in previous[1])):
                rates_by_stamp[sample.source_stamp_ns] = np.abs(
                    (np.asarray(current_angles) - np.asarray(previous[1]))
                    * analysis.WHEEL_RADIUS_M / 0.025)
            previous = (sample.packet_sequence, current_angles)
    stamps = sorted(wanted_stamps.intersection(rates_by_stamp))
    return stamps, np.stack([rates_by_stamp[stamp] for stamp in stamps]) if stamps else np.empty((0, 2))


def _read_diagnostics(path: Path, topic: str) -> dict[int, np.ndarray]:
    """Read v5/v7 estimator diagnostics keyed by their source-stamp fields."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        if topic not in topics:
            raise ValueError(f"{path} is missing diagnostic topic {topic}")
        topic_id, message_type = topics[topic]
        cls = get_message(message_type)
        rows: dict[int, np.ndarray] = {}
        for _, payload in connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? "
                "ORDER BY timestamp,id", (topic_id,)):
            message = deserialize_message(bytes(payload), cls)
            values = np.asarray(message.data, dtype=np.float64)
            label = message.layout.dim[0].label if message.layout.dim else ""
            source_index = (28 if label == "deterministic_odometry_v7" else
                            1 if label == "deterministic_odometry_v5" else None)
            expected_size = 42 if source_index == 28 else 28
            if (source_index is not None and values.shape == (expected_size,)
                    and np.isfinite(values[source_index])):
                rows[int(round(values[source_index] * 1e9))] = values
        if not rows:
            raise ValueError(f"{path} has no supported v5/v7 diagnostics on {topic}")
        return rows
    finally:
        connection.close()


def _read_poses(path: Path, topic: str) -> dict[int, np.ndarray]:
    """Read source-stamped planar pose from a nav_msgs/Odometry stream."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        if topic not in topics:
            raise ValueError(f"{path} is missing pose topic {topic}")
        rows: dict[int, np.ndarray] = {}
        for _, message in analysis._messages(connection, topics, topic):
            stamp = analysis._stamp_ns(message.header.stamp)
            if stamp <= 0:
                continue
            pose = message.pose.pose
            q = pose.orientation
            yaw = math.atan2(
                2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
                1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
            )
            state = np.asarray((pose.position.x, pose.position.y, yaw),
                               dtype=np.float64)
            if np.isfinite(state).all():
                rows[stamp] = state
        if not rows:
            raise ValueError(f"{path} has no source-stamped poses on {topic}")
        return rows
    finally:
        connection.close()


def _single_alignment_pose_metrics(samples: list[Any],
                                   poses: dict[int, np.ndarray],
                                   estimator_name: str) -> dict[str, Any]:
    """Score accumulated pose with one initial SE(2) alignment for the run."""
    matched = [sample for sample in samples
               if sample.source_stamp_ns in poses
               and sample.simulator_pose_xyyaw is not None
               and np.isfinite(sample.simulator_pose_xyyaw).all()]
    if len(matched) < 2:
        raise ValueError(f"{estimator_name} has fewer than two matched GT poses")
    predicted = np.stack([poses[sample.source_stamp_ns] for sample in matched])
    truth = np.stack([sample.simulator_pose_xyyaw for sample in matched])
    yaw_offset = math.atan2(
        math.sin(float(truth[0, 2] - predicted[0, 2])),
        math.cos(float(truth[0, 2] - predicted[0, 2])))
    cosine, sine = math.cos(yaw_offset), math.sin(yaw_offset)
    relative = predicted[:, :2] - predicted[0, :2]
    aligned = truth[0, :2] + np.column_stack((
        cosine * relative[:, 0] - sine * relative[:, 1],
        sine * relative[:, 0] + cosine * relative[:, 1]))
    position_error = np.linalg.norm(aligned - truth[:, :2], axis=1)
    yaw_error = np.arctan2(
        np.sin(predicted[:, 2] + yaw_offset - truth[:, 2]),
        np.cos(predicted[:, 2] + yaw_offset - truth[:, 2]))
    packet_ids = np.asarray([sample.packet_sequence for sample in matched],
                            dtype=np.int64)
    result: dict[str, Any] = {
        "samples": len(matched),
        "source_stamp_coverage": len(matched) / len(samples),
        "duration_s": float((matched[-1].source_stamp_ns -
                              matched[0].source_stamp_ns) / 1e9),
        "largest_packet_sequence_gap": int(np.max(np.diff(packet_ids)))
        if len(packet_ids) > 1 else 0,
        "alignment": "one initial SE(2) alignment at the first scored sample; no subsequent truth correction",
        "position_rmse_m": float(np.sqrt(np.mean(position_error ** 2))),
        "position_p95_m": float(np.quantile(position_error, 0.95)),
        "position_max_m": float(np.max(position_error)),
        "position_endpoint_error_m": float(position_error[-1]),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error ** 2))),
        "yaw_p95_abs_rad": float(np.quantile(np.abs(yaw_error), 0.95)),
        "yaw_endpoint_error_rad": float(yaw_error[-1]),
        "per_lap_without_realignment": [],
    }
    lap_groups: dict[int, list[int]] = {}
    for index, sample in enumerate(matched):
        if sample.lap_count is not None:
            lap_groups.setdefault(int(sample.lap_count), []).append(index)
    for lap_count, indices in sorted(lap_groups.items()):
        local_position = position_error[indices]
        local_yaw = yaw_error[indices]
        result["per_lap_without_realignment"].append({
            "lap_count": lap_count,
            "samples": len(indices),
            "position_rmse_m": float(np.sqrt(np.mean(local_position ** 2))),
            "position_p95_m": float(np.quantile(local_position, 0.95)),
            "position_endpoint_error_m": float(local_position[-1]),
            "yaw_rmse_rad": float(np.sqrt(np.mean(local_yaw ** 2))),
            "yaw_endpoint_error_rad": float(local_yaw[-1]),
        })
    return result


def _align_diagnostics(rows: dict[int, np.ndarray], stamps: list[int],
                       topic: str) -> np.ndarray:
    keys = sorted(rows)
    aligned = []
    for stamp in stamps:
        index = bisect.bisect_left(keys, stamp)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(keys)]
        if not candidates:
            raise ValueError(f"{topic} has no diagnostics near source stamp {stamp}")
        best = min(candidates, key=lambda i: abs(keys[i] - stamp))
        if abs(keys[best] - stamp) > 2_000_000:
            raise ValueError(f"{topic} misses source stamp {stamp} within 2 ms")
        aligned.append(rows[keys[best]])
    return np.stack(aligned)


def score(raw_bag: Path, candidate_replay: Path, output: Path,
          baseline_replay: Path | None = None,
          expected_laps: int = 12,
          u_residual_model: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite score: {output}")
    quality = practice._receipt_gates(
        raw_bag, expected_laps=expected_laps,
        allow_post_run_disconnect=(expected_laps == 12))
    capture = body.load_capture(raw_bag)
    direct = body._read_replayed_states(raw_bag, "/odom")
    candidate = body._read_replayed_states(candidate_replay, "/replayed_odom")
    baseline = (body._read_replayed_states(baseline_replay, "/replayed_odom")
                if baseline_replay else direct)

    x, y, samples, _, _, invalid, unscored = practice._score_arrays(
        capture, quality["active_start_receipt_ns"],
        quality[f"lap{expected_laps}_receipt_ns"], [],
        *practice.WINDOW_STEPS)
    sensor = x[:, practice.WINDOW_STEPS[0]:].reshape(-1, 10)
    truth = y[:, practice.WINDOW_STEPS[0]:, :3].reshape(-1, 3)
    stamps = [sample.source_stamp_ns for sample in samples]

    def exact(states: dict[int, np.ndarray], name: str) -> np.ndarray:
        missing = [stamp for stamp in stamps if stamp not in states]
        if missing:
            raise ValueError(f"{name} misses {len(missing)}/{len(stamps)} source stamps")
        return np.stack([states[stamp] for stamp in stamps])

    estimates = {
        "recorded_production_odom": exact(direct, "recorded /odom"),
        "candidate_replay": exact(candidate, "candidate replay"),
    }
    pose_estimates = {
        "recorded_production_odom": _single_alignment_pose_metrics(
            samples, _read_poses(raw_bag, "/odom"),
            "recorded production Odom"),
        "candidate_replay": _single_alignment_pose_metrics(
            samples, _read_poses(candidate_replay, "/replayed_odom"),
            "candidate replay"),
    }
    residual_metadata = None
    if u_residual_model is not None:
        model = json.loads(u_residual_model.read_text(encoding="utf-8"))
        if model.get("target") != "truth_u_minus_current_source_odom_u":
            raise ValueError("residual model target does not match current-source odom")
        base_feature_names = model.get(
            "base_feature_names", list(ODOM_RESIDUAL_LEGACY_FEATURE_NAMES))
        if not set(base_feature_names).issubset(ODOM_RESIDUAL_FEATURE_NAMES):
            raise ValueError("residual model base feature schema does not match scorer")
        candidate_state = estimates["candidate_replay"]
        history_steps = int(model.get("history_steps", 1))
        if history_steps == 1 and base_feature_names == list(
                ODOM_RESIDUAL_LEGACY_FEATURE_NAMES):
            if model.get("feature_names") != list(ODOM_RESIDUAL_LEGACY_FEATURE_NAMES):
                raise ValueError("residual model feature schema does not match scorer")
            residual_features = np.column_stack((
                candidate_state[:, :2], sensor[:, 2:4].mean(axis=1),
                sensor[:, 4:7], np.full(len(sensor), 0.025)))
        else:
            residual_features = _causal_history_features(
                capture, candidate, stamps, history_steps, base_feature_names)
        if residual_features.shape[1] != len(model["feature_names"]):
            raise ValueError("residual model expanded feature schema has wrong size")
        mean = np.asarray(model["feature_mean"], dtype=np.float64)
        scale = np.asarray(model["feature_scale"], dtype=np.float64)
        coefficient = np.asarray(model["coefficient"], dtype=np.float64)
        if (residual_features.shape[1] != len(mean)
                or scale.shape != mean.shape
                or coefficient.shape != mean.shape
                or np.any(scale <= 0.0)):
            raise ValueError("residual model dimensions are invalid")
        correction = (((residual_features - mean) / scale) @ coefficient
                      + float(model["intercept"]))
        activation_gate = model.get("activation_gate")
        gate_active = np.ones(len(correction), dtype=bool)
        if activation_gate is not None:
            if activation_gate.get("kind") != (
                    "absolute_encoder_odom_forward_speed_mismatch_mps"):
                raise ValueError("residual model contains an unsupported activation gate")
            threshold = float(activation_gate.get("threshold_mps", float("nan")))
            if not np.isfinite(threshold) or threshold <= 0.0:
                raise ValueError("residual model activation-gate threshold is invalid")
            names = model["feature_names"]
            try:
                odom_u_index = names.index("odom_u_mps")
                encoder_index = names.index("encoder_mean_100ms_mps")
            except ValueError as exc:
                raise ValueError(
                    "mismatch activation gate requires Odom-u and encoder features") from exc
            gate_active = np.abs(
                residual_features[:, encoder_index] -
                residual_features[:, odom_u_index]) >= threshold
            correction = np.where(gate_active, correction, 0.0)
        corrected = candidate_state.copy()
        corrected[:, 0] += correction
        estimates["candidate_plus_trained_u_residual"] = corrected
        residual_metadata = {
            "model": str(u_residual_model.resolve()),
            "history_steps": history_steps,
            "activation_gate": activation_gate,
            "activation_gate_active_samples": int(np.sum(gate_active)),
            "mean_u_correction_mps": float(np.mean(correction)),
            "rmse_u_correction_mps": float(np.sqrt(np.mean(correction ** 2))),
            "p95_abs_u_correction_mps": float(np.quantile(np.abs(correction), 0.95)),
        }
    if baseline_replay:
        estimates["baseline_replay"] = exact(baseline, "baseline replay")
        direct_subset = estimates["recorded_production_odom"]
        replay_subset = estimates["baseline_replay"]
        baseline_max_difference = float(np.max(np.abs(direct_subset - replay_subset)))
        if baseline_max_difference > 1e-9:
            raise ValueError(
                "baseline replay does not reproduce recorded production /odom: "
                f"max abs difference={baseline_max_difference:g}")
    else:
        baseline_max_difference = None

    speed = np.hypot(truth[:, 0], truth[:, 1])
    steering = np.abs(sensor[:, 0])
    throttle = sensor[:, 1]
    encoder_residual = np.abs(sensor[:, 2:4].mean(axis=1) - truth[:, 0])
    lateral_accel_proxy = np.abs(truth[:, 0] * truth[:, 2])
    speed_magnitude = np.hypot(truth[:, 0], truth[:, 1])
    throttle_zero = sensor[:, 1] == 0.0
    regions = {
        "all_scored_samples": np.ones(len(truth), dtype=bool),
        "speed_lt_2_mps": speed_magnitude < 2.0,
        "speed_2_to_4_mps": (speed_magnitude >= 2.0) & (speed_magnitude < 4.0),
        "speed_4_to_6_mps": (speed_magnitude >= 4.0) & (speed_magnitude < 6.0),
        "speed_6_to_8_mps": (speed_magnitude >= 6.0) & (speed_magnitude < 8.0),
        "speed_ge_8_mps": speed_magnitude >= 8.0,
        "unity_zero_throttle_brake_branch": throttle_zero,
        "unity_nonzero_motor_branch": ~throttle_zero,
        "speed_ge_4_mps": speed >= 4.0,
        "speed_ge_6_mps": speed >= 6.0,
        "speed_ge_6_and_steering_ge_0p20":
            (speed_magnitude >= 6.0) & (steering >= 0.20),
        "speed_ge_6_and_steering_ge_0p30":
            (speed_magnitude >= 6.0) & (steering >= 0.30),
        "speed_ge_6_and_lateral_accel_proxy_ge_6":
            (speed_magnitude >= 6.0) & (lateral_accel_proxy >= 6.0),
        "steering_ge_0p30_rad": steering >= 0.30,
        "positive_steering": sensor[:, 0] > 0.0,
        "negative_steering": sensor[:, 0] < 0.0,
        "encoder_mismatch_ge_0p5_mps": encoder_residual >= 0.5,
        "encoder_mismatch_ge_1_mps": encoder_residual >= 1.0,
        "wheel_mismatch_and_high_throttle":
            (encoder_residual >= 0.5) & (throttle >= 0.30),
    }
    results = {}
    for name, estimate in estimates.items():
        error = estimate - truth
        results[name] = {
            "all": _metrics(error),
            "truth_reset_short_integrated_motion":
                practice._integrated_motion_metrics(
                    estimate, truth, samples, practice.WINDOW_STEPS[1]),
            "regions": {key: _metrics(error[mask])
                        for key, mask in regions.items() if np.any(mask)},
        }

    raw_angles = _read_raw_encoder_angles(raw_bag)
    raw_stamps, raw_wheels = _fixed25_raw_encoder_rates(
        capture, raw_angles, set(stamps))
    sample_by_stamp = {sample.source_stamp_ns: sample for sample in samples}
    if len(raw_stamps) != len(stamps):
        raise ValueError(
            f"fixed-25-ms raw encoder coverage {len(raw_stamps)}/{len(stamps)}; "
            "a scored sample lacks a contiguous source packet")
    raw_truth_u = np.asarray([sample_by_stamp[stamp].state[0]
                              for stamp in raw_stamps], dtype=np.float64)
    raw_residual = raw_wheels - raw_truth_u[:, None]
    filtered_wheels = np.stack([
        sample_by_stamp[stamp].rear_wheel_surface_mps for stamp in raw_stamps])
    filtered_residual = filtered_wheels - raw_truth_u[:, None]

    production_diagnostics = _align_diagnostics(
        _read_diagnostics(raw_bag, "/odom/diagnostics"), stamps,
        "recorded /odom/diagnostics")
    candidate_diagnostics = _align_diagnostics(
        _read_diagnostics(candidate_replay, "/replayed_odom_diagnostics"),
        stamps, "current replay diagnostics")

    def wheel_signal_summary(diagnostics: np.ndarray) -> dict[str, Any]:
        # Both supported diagnostic schemas share [3] rolling raw wheel rate,
        # [4] calibrated rolling rate, and [25] instantaneous packet rate.
        # These encoder-derived signals are mismatch diagnostics, not truth.
        return {
            "fixed25ms_raw_per_wheel_vs_chassis_u_mismatch_mps": _metrics(raw_residual),
            "100ms_filtered_per_wheel_vs_chassis_u_mismatch_mps": _metrics(filtered_residual),
            "odom_rolling_raw_encoder_rate_vs_chassis_u_mismatch_mps":
                _metrics(diagnostics[:, 3:4] - truth[:, 0:1]),
            "odom_instantaneous_packet_encoder_rate_vs_chassis_u_mismatch_mps":
                _metrics(diagnostics[:, 25:26] - truth[:, 0:1]),
            "odom_calibrated_rolling_encoder_rate_vs_chassis_u_mismatch_mps":
                _metrics(diagnostics[:, 4:5] - truth[:, 0:1]),
        }

    wheel_signals = {
        "recorded_production": wheel_signal_summary(production_diagnostics),
        "current_source_replay": wheel_signal_summary(candidate_diagnostics),
        "interpretation": (
            "Encoder-derived rates are measurements, not independent wheel-speed truth. "
            "Their residual against simulator-truth chassis u quantifies signal/body "
            "mismatch only."),
    }
    report = {
        "run_id": raw_bag.parents[1].name,
        "raw_bag": str(raw_bag.resolve()),
        "candidate_replay": str(candidate_replay.resolve()),
        "u_residual_model": residual_metadata,
        "baseline_replay": str(baseline_replay.resolve()) if baseline_replay else None,
        "quality": quality,
        "comparison": {
            "truth": "bridge odometry rear-axle body twist, offline only",
            "sample_window": {"burn_in": practice.WINDOW_STEPS[0],
                              "score_steps": practice.WINDOW_STEPS[1]},
            "scored_samples": len(truth),
            "expected_laps": expected_laps,
            "exact_source_stamp_coverage": len(stamps),
            "baseline_replay_max_abs_difference_from_recorded_odom": baseline_max_difference,
            "invalid_sensor_samples": invalid,
            "unscored_short_or_tail_samples": unscored,
            "lateral_acceleration_regime": "|u*r| kinematic proxy, not tire force",
            "wheel_rate_definition": (
                "fixed-grid raw per-wheel rate = abs(delta cumulative angle)*0.059/0.025 "
                "only across contiguous packet ids; filtered per-wheel rate is the "
                "existing causal 100 ms angle window"),
        },
        "encoder_signal_diagnostics": wheel_signals,
        "estimators": results,
        "continuous_run_pose_from_single_initial_alignment": pose_estimates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bag", type=Path, required=True)
    parser.add_argument("--candidate-replay", type=Path, required=True)
    parser.add_argument("--baseline-replay", type=Path)
    parser.add_argument("--expected-laps", type=int, default=12,
                        help="completed lap-count transitions required in the raw capture")
    parser.add_argument("--u-residual-model", type=Path,
                        help="optional train-only current-source u residual model JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(args.raw_bag, args.candidate_replay, args.output,
                       args.baseline_replay, args.expected_laps,
                       args.u_residual_model)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"practice replay scoring failed: {exc}\n")
    print(json.dumps({
        "run_id": result["run_id"],
        "samples": result["comparison"]["scored_samples"],
        "estimators": {name: value["all"]["rmse"]
                       for name, value in result["estimators"].items()},
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
