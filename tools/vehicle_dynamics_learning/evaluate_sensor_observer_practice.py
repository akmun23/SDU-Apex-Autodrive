#!/usr/bin/env python3
"""Score the frozen sensor observer on a completed practice bag.

The original bag is opened read-only. Bridge odometry is used only for the
offline target; observer inputs contain sensors, actuator feedback/commands,
and sample interval. The full 12-lap active interval must pass collision and
40 Hz stream-quality checks. A Socket.IO disconnect after lap 12 is recorded
as post-run teardown, not treated as part of the scored interval.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numpy._core import multiarray as numpy_multiarray
except ImportError:  # NumPy < 2.0 module path.
    from numpy.core import multiarray as numpy_multiarray

from tools import analyze_open_plane_dynamics as analysis
from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning import prepare_dataset, train_sensor_observer


REPO_ROOT = Path(__file__).resolve().parents[2]
LAP_COUNT = "/autodrive/roboracer_1/lap_count"
COLLISION_COUNT = analysis.COLLISIONS
FAULT_DETAIL = "/autodrive/roboracer_1/bridge_timing_fault_detail"
MAX_ALIGN_NS = 20_000_000
SENSOR_PERIOD_S = 0.025
WINDOW_STEPS = (16, 32)


def _read_messages(connection: sqlite3.Connection, topics: dict[str, tuple[int, str]],
                   topic: str) -> list[tuple[int, Any]]:
    if topic not in topics:
        return []
    return list(analysis._messages(connection, topics, topic))


def _transitions(rows: list[tuple[int, Any]]) -> list[tuple[int, int]]:
    result = []
    previous = None
    for receipt_ns, message in rows:
        count = int(message.data)
        if count != previous:
            result.append((receipt_ns, count))
            previous = count
    return result


def _validate_lap_count_transitions(counts: list[int], expected_laps: int) -> None:
    expected = list(range(expected_laps + 1))
    if counts != expected:
        raise ValueError(
            f"expected completed lap-count transitions 0..{expected_laps}; got {counts}")


def _receipt_gates(path: Path, expected_laps: int = 12,
                   allow_post_run_disconnect: bool = True) -> dict[str, Any]:
    """Validate one uninterrupted active practice interval.

    The production observer evaluator retains its historical 12-lap default
    and narrowly classified trailing-disconnect allowance. Fresh model
    validation uses ``expected_laps=6`` and disables that allowance so every
    timing fault rejects the capture.
    """
    if expected_laps < 1:
        raise ValueError("expected_laps must be positive")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        required = (*body.STREAM_TOPICS, COLLISION_COUNT, analysis.TIMING_FAULT,
                    FAULT_DETAIL, LAP_COUNT)
        missing = [topic for topic in required if topic not in topics]
        if missing:
            raise ValueError("practice bag missing required stream(s): " + ", ".join(missing))

        transitions = _transitions(_read_messages(connection, topics, LAP_COUNT))
        counts = [count for _, count in transitions]
        _validate_lap_count_transitions(counts, expected_laps)
        active_start_ns, _ = transitions[0]
        active_end_ns, _ = transitions[-1]
        if active_end_ns <= active_start_ns:
            raise ValueError("invalid active interval from lap-count transitions")

        collisions = [int(message.data) for _, message in
                      _read_messages(connection, topics, COLLISION_COUNT)]
        if not collisions or min(collisions) != 0 or max(collisions) != 0:
            raise ValueError(f"practice interval has collision count(s): {collisions[:3]}…")

        faults = [(receipt_ns, bool(message.data)) for receipt_ns, message in
                  _read_messages(connection, topics, analysis.TIMING_FAULT)
                  if bool(message.data)]
        details = _read_messages(connection, topics, FAULT_DETAIL)
        terminal_disconnect = None
        for receipt_ns, _ in faults:
            if receipt_ns <= active_end_ns:
                raise ValueError(
                    f"bridge timing fault occurred before completed lap {expected_laps}")
            if not allow_post_run_disconnect:
                raise ValueError("bridge timing fault occurred after active interval")
            nearby = [(abs(detail_ns - receipt_ns), str(message.data))
                      for detail_ns, message in details
                      if abs(detail_ns - receipt_ns) <= 100_000_000]
            reason = min(nearby)[1] if nearby else ""
            if (reason != "simulator Socket.IO connection lost"
                    or receipt_ns - active_end_ns > 1_000_000_000):
                raise ValueError(f"unclassified timing fault after run: {reason!r}")
            terminal_disconnect = {
                "time_after_active_end_s": (receipt_ns - active_end_ns) / 1e9,
                "detail": reason,
            }

        cadence = {}
        for topic in body.STREAM_TOPICS:
            topic_id = topics[topic][0]
            receipts = [int(row[0]) for row in connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? "
                "AND timestamp>=? AND timestamp<=? ORDER BY timestamp,id",
                (topic_id, active_start_ns, active_end_ns),
            )]
            gaps = sorted((right - left) / 1e6 for left, right in
                          zip(receipts, receipts[1:]) if right > left)
            span_s = ((receipts[-1] - receipts[0]) / 1e9
                      if len(receipts) > 1 else 0.0)
            rate_hz = (len(receipts) - 1) / span_s if span_s > 0.0 else 0.0
            p95 = float(np.quantile(gaps, 0.95)) if gaps else math.inf
            max_gap = max(gaps, default=math.inf)
            max_allowed = 120.0 if topic in body.COMMAND_STREAM_TOPICS else 60.0
            valid = rate_hz >= 38.0 and p95 <= 35.0 and max_gap <= max_allowed
            cadence[topic] = {"samples": len(receipts), "rate_hz": rate_hz,
                              "gap_p95_ms": p95, "gap_max_ms": max_gap,
                              "pass": valid}
            if not valid:
                raise ValueError(f"active interval stream-quality failure on {topic}: "
                                 f"{rate_hz:.2f} Hz, p95 {p95:.2f} ms, max {max_gap:.2f} ms")

        return {"expected_laps": expected_laps,
                "lap_count_transitions": counts,
                "active_start_receipt_ns": active_start_ns,
                "active_end_receipt_ns": active_end_ns,
                f"lap{expected_laps}_receipt_ns": active_end_ns,
                **({"lap12_receipt_ns": active_end_ns}
                   if expected_laps == 12 else {}),
                "active_duration_s": (active_end_ns - active_start_ns) / 1e9,
                "collision_min_max": [min(collisions), max(collisions)],
                "timing_faults_during_active_interval": 0,
                "post_run_disconnect": terminal_disconnect,
                "stream_cadence": cadence}
    finally:
        connection.close()


def _load_team_odom(path: Path) -> list[tuple[int, np.ndarray]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        rows = []
        for _, message in _read_messages(connection, topics, "/odom"):
            source_ns = analysis._stamp_ns(message.header.stamp)
            if source_ns is None:
                continue
            twist = message.twist.twist
            rows.append((source_ns, np.asarray((
                float(twist.linear.x), float(twist.linear.y),
                float(twist.angular.z)), dtype=np.float64)))
        if not rows:
            raise ValueError("practice bag contains no source-stamped /odom reference")
        return sorted(rows, key=lambda item: item[0])
    finally:
        connection.close()


def _metric_rows(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "rmse": None, "mae": None, "p95_abs": None}
    return {"samples": int(len(error)),
            "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
            "mae": np.mean(np.abs(error), axis=0).tolist(),
            "p95_abs": np.quantile(np.abs(error), 0.95, axis=0).tolist()}


def _integrated_motion_metrics(states: np.ndarray, truth: np.ndarray,
                              samples: list[body.MotionSample],
                              score_steps: int) -> dict[str, Any]:
    """Compare short unanchored body-twist integration within each score window."""
    if len(samples) != len(states) or len(states) % score_steps:
        raise ValueError("integrated-motion inputs do not align to score windows")
    window_count = len(states) // score_steps
    state_windows = states.reshape(window_count, score_steps, 3).astype(np.float64)
    truth_windows = truth.reshape(window_count, score_steps, 3).astype(np.float64)
    pose_rows = [sample.simulator_pose_xyyaw for sample in samples]
    if any(value is None or not np.isfinite(value).all() for value in pose_rows):
        raise ValueError("simulator truth pose is missing from a scored sample")
    poses = np.stack(pose_rows).reshape(window_count, score_steps, 3)
    dt = np.stack([
        np.r_[0.0, np.diff([
            sample.time_s for sample in samples[i * score_steps:(i + 1) * score_steps]
        ])]
        for i in range(window_count)
    ])
    if np.any(dt[:, 1:] <= 0.0) or np.any(dt[:, 1:] > 0.075):
        raise ValueError("scored motion window contains a non-causal time gap")

    estimated_xy = np.zeros((window_count, score_steps, 2), dtype=np.float64)
    estimated_yaw = np.repeat(poses[:, :1, 2], score_steps, axis=1)
    truth_xy = poses[:, :, :2] - poses[:, :1, :2]
    truth_yaw = np.arctan2(
        np.sin(poses[:, :, 2] - poses[:, :1, 2]),
        np.cos(poses[:, :, 2] - poses[:, :1, 2]))
    for index in range(1, score_steps):
        u = 0.5 * (state_windows[:, index - 1, 0] + state_windows[:, index, 0])
        v = 0.5 * (state_windows[:, index - 1, 1] + state_windows[:, index, 1])
        r = 0.5 * (state_windows[:, index - 1, 2] + state_windows[:, index, 2])
        step_dt = dt[:, index]
        mid_yaw = estimated_yaw[:, index - 1] + 0.5 * r * step_dt
        estimated_xy[:, index, 0] = estimated_xy[:, index - 1, 0] + (
            u * np.cos(mid_yaw) - v * np.sin(mid_yaw)) * step_dt
        estimated_xy[:, index, 1] = estimated_xy[:, index - 1, 1] + (
            u * np.sin(mid_yaw) + v * np.cos(mid_yaw)) * step_dt
        estimated_yaw[:, index] = estimated_yaw[:, index - 1] + r * step_dt

    position_error = estimated_xy - truth_xy
    estimated_yaw_relative = np.arctan2(
        np.sin(estimated_yaw - poses[:, :1, 2]),
        np.cos(estimated_yaw - poses[:, :1, 2]))
    yaw_error = np.arctan2(np.sin(estimated_yaw_relative - truth_yaw),
                           np.cos(estimated_yaw_relative - truth_yaw))
    endpoint = np.linalg.norm(position_error[:, -1], axis=1)
    return {
        "windows": window_count,
        "window_duration_s_median": float(np.median(np.sum(dt[:, 1:], axis=1))),
        "truth_reset_relative_position_rmse_m": float(np.sqrt(
            np.mean(np.sum(position_error ** 2, axis=2)))),
        "truth_reset_relative_position_endpoint_rmse_m": float(np.sqrt(
            np.mean(endpoint ** 2))),
        "truth_reset_relative_position_endpoint_p95_m": float(
            np.quantile(endpoint, 0.95)),
        "integrated_relative_yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error ** 2))),
        "integrated_relative_yaw_p95_abs_rad": float(np.quantile(
            np.abs(yaw_error), 0.95)),
        "initial_pose_policy": "truth pose used only to set each window's origin and heading",
        "yaw_source_policy": "gyro yaw-rate is passed through unchanged",
        "truth_pose_source": "packet-aligned simulator pose, offline label only",
    }


def _score_arrays(capture: body.Capture, start_ns: int, end_ns: int,
                  odom_rows: list[tuple[int, np.ndarray]],
                  burn_in: int, score_steps: int):
    windows_x, windows_y, score_samples = [], [], []
    invalid_sensor_samples = 0
    unscored_samples = 0
    for sequence in capture.sequences:
        samples = [sample for sample in sequence
                   if start_ns <= sample.receipt_ns <= end_ns]
        if not samples:
            continue
        sensors, truths, good = [], [], []
        for index, sample in enumerate(samples):
            dt = (SENSOR_PERIOD_S if index == 0 else
                  sample.time_s - samples[index - 1].time_s)
            row = prepare_dataset._sensor_frame(sample)
            is_good = row is not None and math.isfinite(dt) and 0.015 <= dt <= 0.075
            if is_good:
                sensors.append(np.r_[row, dt].astype(np.float32))
                truths.append(sample.state.astype(np.float32))
            else:
                sensors.append(np.zeros(10, dtype=np.float32))
                truths.append(sample.state.astype(np.float32))
                invalid_sensor_samples += 1
            good.append(is_good)

        good_array = np.asarray(good, dtype=bool)
        changes = np.diff(np.r_[False, good_array, False].astype(np.int8))
        segment_starts = np.flatnonzero(changes == 1)
        segment_ends = np.flatnonzero(changes == -1)
        for segment_start, segment_end in zip(segment_starts, segment_ends):
            segment_len = int(segment_end - segment_start)
            minimum = burn_in + score_steps
            if segment_len < minimum:
                unscored_samples += segment_len
                continue
            local_start = 0
            windows_in_segment = 0
            while local_start + minimum <= segment_len:
                x_start = int(segment_start + local_start)
                score_start = x_start + burn_in
                score_end = score_start + score_steps
                windows_x.append(np.asarray(sensors[x_start:score_end], dtype=np.float32))
                windows_y.append(np.asarray(truths[x_start:score_end], dtype=np.float32))
                score_samples.extend(samples[score_start:score_end])
                local_start += score_steps
                windows_in_segment += 1
            unscored_samples += segment_len - windows_in_segment * score_steps

    if not windows_x:
        raise ValueError("no valid practice windows long enough for burn-in and scoring")
    x = np.stack(windows_x)
    y = np.stack(windows_y)
    stamp_times = [sample.source_stamp_ns for sample in score_samples]
    team_times = [stamp for stamp, _ in odom_rows]
    team_values = []
    team_offsets_ms = []
    for target in stamp_times:
        index = bisect.bisect_left(team_times, target)
        candidates = [i for i in (index - 1, index) if 0 <= i < len(team_times)]
        if not candidates:
            team_values.append(np.full(3, np.nan))
            team_offsets_ms.append(math.inf)
            continue
        best = min(candidates, key=lambda i: abs(team_times[i] - target))
        offset = team_times[best] - target
        if abs(offset) > MAX_ALIGN_NS:
            team_values.append(np.full(3, np.nan))
            team_offsets_ms.append(math.inf)
        else:
            team_values.append(odom_rows[best][1])
            team_offsets_ms.append(offset / 1e6)
    return (x, y, score_samples, np.asarray(team_values),
            np.asarray(team_offsets_ms), invalid_sensor_samples, unscored_samples)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    expected_laps = getattr(args, "expected_laps", 12)
    quality = _receipt_gates(
        args.bag, expected_laps=expected_laps,
        allow_post_run_disconnect=(expected_laps == 12))
    capture = body.load_capture(args.bag)
    checkpoint_payload = None
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    normalization = report["normalization"]
    torch, nn = train_sensor_observer._torch()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    safe_numpy_globals = [
        (numpy_multiarray.scalar, "numpy._core.multiarray.scalar"),
        np.dtype,
        (type(np.dtype("U")), "numpy.dtypes.StrDType"),
    ]
    with torch.serialization.safe_globals(safe_numpy_globals):
        checkpoint_payload = torch.load(
            args.checkpoint, map_location=device, weights_only=True)
    metadata = checkpoint_payload["metadata"]
    if metadata["sensor_feature_names"] != list(prepare_dataset.SENSOR_FEATURE_NAMES):
        raise ValueError("checkpoint sensor feature schema does not match this evaluator")
    model_type = train_sensor_observer._make_model(
        nn, len(metadata["sensor_feature_names"]), int(metadata["hidden_size"]))
    model = model_type().to(device)
    model.load_state_dict(checkpoint_payload["state_dict"])

    odom_rows = _load_team_odom(args.bag)
    burn_in, score_steps = WINDOW_STEPS
    arrays = _score_arrays(capture, quality["active_start_receipt_ns"],
                           quality[f"lap{expected_laps}_receipt_ns"], odom_rows,
                           burn_in, score_steps)
    (x, y, score_samples, team_values, team_offsets,
     invalid_sensor_samples, unscored_samples) = arrays
    sensor_mean = np.asarray(normalization["sensor_mean"], dtype=np.float32)
    sensor_scale = np.asarray(normalization["sensor_scale"], dtype=np.float32)
    truth_mean = np.asarray(normalization["target_mean"], dtype=np.float32)
    truth_scale = np.asarray(normalization["target_scale"], dtype=np.float32)
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor((x - sensor_mean) / sensor_scale,
                                 dtype=torch.float32, device=device)
        prediction_scaled = model(tensor)[:, burn_in:burn_in + score_steps]
        prediction = (prediction_scaled * torch.as_tensor(
            truth_scale, device=device) + torch.as_tensor(
                truth_mean, device=device)).cpu().numpy()
    truth = y[:, burn_in:burn_in + score_steps, :3].reshape(-1, 3)
    sensor_scored = x[:, burn_in:burn_in + score_steps].reshape(-1, 10)
    prediction = prediction.reshape(-1, 2)
    observer = np.column_stack((prediction, sensor_scored[:, 6]))
    baseline = np.column_stack((sensor_scored[:, 2:4].mean(axis=1),
                                np.zeros(len(sensor_scored)), sensor_scored[:, 6]))
    observer_error = observer - truth
    baseline_error = baseline - truth
    production_all_error = team_values - truth

    matched = np.isfinite(team_values).all(axis=1)
    production_error = (team_values[matched] - truth[matched])
    observer_motion = _integrated_motion_metrics(
        observer, truth, score_samples, score_steps)
    production_motion = _integrated_motion_metrics(
        team_values, truth, score_samples, score_steps)
    by_lap = {}
    lap_ids = np.asarray([sample.lap_count for sample in score_samples], dtype=object)
    for lap in sorted({value for value in lap_ids if value is not None}):
        mask = lap_ids == lap
        by_lap[str(lap)] = {
            "observer_rmse_u_v_r": _metric_rows(observer_error[mask])["rmse"],
            "production_odom_rmse_u_v_r": _metric_rows(
                (team_values - truth)[mask & matched])["rmse"],
            "scored_samples": int(np.count_nonzero(mask)),
        }

    regimes = {}
    speed = np.hypot(truth[:, 0], truth[:, 1])
    abs_steer = np.abs(sensor_scored[:, 0])
    throttle = sensor_scored[:, 1]
    rear_encoder_mean = sensor_scored[:, 2:4].mean(axis=1)
    rear_encoder_residual = rear_encoder_mean - truth[:, 0]
    region_masks = (
        ("steer_030_to_042", (abs_steer >= 0.30) & (abs_steer < 0.42)),
        ("steer_ge_042", abs_steer >= 0.42),
        ("speed_6_to_8_steer_ge_042_throttle_ge_030",
         (speed >= 6.0) & (speed < 8.0) & (abs_steer >= 0.42) & (throttle >= 0.30)),
        ("speed_ge_8_steer_ge_042_throttle_ge_030",
         (speed >= 8.0) & (abs_steer >= 0.42) & (throttle >= 0.30)),
        ("rear_encoder_residual_abs_ge_050_mps",
         np.abs(rear_encoder_residual) >= 0.50),
    )
    for name, mask in region_masks:
        matched_region = mask & matched
        regimes[name] = {
            "samples": int(np.count_nonzero(mask)),
            "production_odom_matched_samples": int(np.count_nonzero(matched_region)),
            "observer_rmse_u_v": _metric_rows(observer_error[mask, :2])["rmse"],
            "encoder_zero_v_rmse_u_v": _metric_rows(baseline_error[mask, :2])["rmse"],
            "production_odom_rmse_u_v": _metric_rows(
                production_all_error[matched_region, :2])["rmse"],
        }

    result = {
        "bag": str(args.bag.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "training_report": str(args.training_report.resolve()),
        "model_training_step": metadata["step"],
        "model_training_runs": metadata["training_runs"],
        "quality_gate": quality,
        "data": {"capture_sequences": len(capture.sequences),
                 "observer_windows": int(len(x)),
                 "scored_sensor_samples": int(len(truth)),
                 "invalid_sensor_samples": int(invalid_sensor_samples),
                 "valid_samples_not_scored_for_burn_in_or_short_tail": int(unscored_samples),
                 "production_odom_matched_samples": int(np.count_nonzero(matched)),
                 "production_odom_match_fraction": float(np.mean(matched)),
                 "source_time_offset_abs_p95_ms": float(np.quantile(
                     np.abs(team_offsets[matched]), .95)) if np.any(matched) else None},
        "observer_rmse_u_v_r": _metric_rows(observer_error),
        "encoder_zero_v_gyro_baseline_rmse_u_v_r": _metric_rows(baseline_error),
        "production_odom_rmse_u_v_r": _metric_rows(production_error),
        "truth_reset_relative_motion": {
            "observer": observer_motion,
            "recorded_production_odom": production_motion,
        },
        "operating_regions": regimes,
        "lapwise": by_lap,
        "interpretation": (
            "Offline practice transfer only. Truth bridge odometry is a label; "
            "observer inputs are causal sensors/commands. The production /odom "
            "comparison is source-time aligned and rear-axle lateral truth is used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-laps", type=int, default=12)
    args = parser.parse_args()
    result = evaluate(args)
    print(json.dumps({key: result[key] for key in (
        "bag", "quality_gate", "data", "observer_rmse_u_v_r",
        "encoder_zero_v_gyro_baseline_rmse_u_v_r", "production_odom_rmse_u_v_r",
        "truth_reset_relative_motion", "operating_regions")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
