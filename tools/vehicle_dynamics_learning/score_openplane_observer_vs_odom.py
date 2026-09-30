#!/usr/bin/env python3
"""Paired whole-run score of a frozen sensor observer and odometry replays.

The sensor model receives only causal sensor/actuator history. Bridge odometry
is an offline label. Every estimator is scored on the same source-stamped
samples from clean, held-out OpenPlane phases.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body
from tools.vehicle_dynamics_learning import (
    evaluate_sensor_observer_practice as observer_eval,
    prepare_dataset,
    train_sensor_observer,
)


def _metrics(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "rmse": None, "bias": None,
                "p95_abs": None}
    return {
        "samples": int(len(error)),
        "rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "p95_abs": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
    }


def _replay_states(path: Path) -> dict[int, np.ndarray]:
    return body._read_replayed_states(path, "/replayed_odom")


def _ground_truth_poses(path: Path) -> dict[int, np.ndarray]:
    """Read simulator pose labels by source stamp; never used as model input."""
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = body.analysis._topic_map(connection)
        topic = body.analysis.ODOM
        if topic not in topics:
            raise ValueError(f"raw capture has no bridge pose labels on {topic}")
        poses = {}
        for _, message in body.analysis._messages(connection, topics, topic):
            stamp = body.analysis._stamp_ns(message.header.stamp)
            pose = message.pose.pose
            q = pose.orientation
            yaw = math.atan2(
                2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
                1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
            )
            value = np.asarray((pose.position.x, pose.position.y, yaw),
                               dtype=np.float64)
            if stamp > 0 and np.isfinite(value).all():
                poses[stamp] = value
        return poses
    finally:
        connection.close()


def _relative_motion_metrics(estimators: dict[str, np.ndarray],
                             sensor_windows: np.ndarray,
                             score_samples: list[body.MotionSample],
                             raw_bag: Path) -> dict[str, Any]:
    """Measure short-horizon dead reckoning, re-anchored per score window."""
    steps = train_sensor_observer.DEFAULT_SCORE_STEPS
    window_count = len(sensor_windows)
    if not window_count or len(score_samples) != window_count * steps:
        raise ValueError("score samples do not align to observer rollout windows")
    poses_by_stamp = _ground_truth_poses(raw_bag)
    stamps = [sample.source_stamp_ns for sample in score_samples]
    missing = [stamp for stamp in stamps if stamp not in poses_by_stamp]
    if missing:
        raise ValueError(
            f"bridge pose labels miss {len(missing)}/{len(stamps)} scored stamps")
    truth_pose = np.stack([poses_by_stamp[stamp] for stamp in stamps])
    truth_pose = truth_pose.reshape(window_count, steps, 3)
    truth_relative_xy = truth_pose[:, :, :2] - truth_pose[:, :1, :2]

    results = {}
    for name, flat_states in estimators.items():
        states = flat_states.reshape(window_count, steps, 3)
        position = np.zeros((window_count, steps, 2), dtype=np.float64)
        heading = truth_pose[:, 0, 2].copy()
        for index in range(1, steps):
            dt = sensor_windows[:, index, 9].astype(np.float64)
            if np.any(~np.isfinite(dt)) or np.any(dt <= 0.0):
                raise ValueError("invalid inter-sample dt in motion window")
            u = 0.5 * (states[:, index - 1, 0] + states[:, index, 0])
            v = 0.5 * (states[:, index - 1, 1] + states[:, index, 1])
            yaw_rate = 0.5 * (states[:, index - 1, 2] + states[:, index, 2])
            mid_heading = heading + 0.5 * yaw_rate * dt
            position[:, index, 0] = (
                position[:, index - 1, 0]
                + (u * np.cos(mid_heading) - v * np.sin(mid_heading)) * dt)
            position[:, index, 1] = (
                position[:, index - 1, 1]
                + (u * np.sin(mid_heading) + v * np.cos(mid_heading)) * dt)
            heading = heading + yaw_rate * dt

        error_xy = position - truth_relative_xy
        radial = np.linalg.norm(error_xy, axis=2)
        endpoint = radial[:, -1]
        results[name] = {
            "windows": window_count,
            "position_error_xy_rmse_m": np.sqrt(
                np.mean(error_xy ** 2, axis=(0, 1))).tolist(),
            "position_error_radial_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
            "position_error_radial_p95_m": float(np.quantile(radial, 0.95)),
            "endpoint_error_rmse_m": float(np.sqrt(np.mean(endpoint ** 2))),
            "endpoint_error_median_m": float(np.median(endpoint)),
            "endpoint_error_p95_m": float(np.quantile(endpoint, 0.95)),
        }
    durations = sensor_windows[:, 1:, 9].sum(axis=1)
    return {
        "method": (
            "Integrate each 32-step estimated rear-axle body-twist segment in "
            "the world frame, initialized from the bridge pose/yaw at that "
            "segment's first scored timestamp. This isolates about 0.775 s "
            "relative motion error; it is not long-run unanchored pose drift."
        ),
        "truth_pose_source": "bridge /odom pose, offline label only",
        "exact_pose_stamp_coverage": len(stamps),
        "window_duration_s_median": float(np.median(durations)),
        "estimators": results,
    }


def score(dataset: Path, training_report: Path, checkpoint: Path,
          raw_bag: Path, production_replay: Path,
          candidate_replay: Path | None, output: Path,
          device: str) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite score: {output}")
    run_id = raw_bag.parents[1].name
    with np.load(dataset, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
        matches = np.flatnonzero(run_ids == run_id)
        if len(matches) != 1:
            raise ValueError(f"dataset must contain exactly one run named {run_id}")
        split = str(splits[int(matches[0])])
        if split not in ("test", "final_test"):
            raise ValueError(f"refusing non-held-out run {run_id} ({split})")

    capture = body.load_capture(raw_bag)
    clean, failures = prepare_dataset._quality(capture)
    if not clean:
        raise ValueError(f"capture fails clean-data gate: {failures}")
    # Match the dataset contract: remove pre-phase context so adjacent phases
    # cannot count the same source sample twice.
    capture = replace(capture, sequences=tuple(
        tuple(sample for sample in seq if sample.time_s >= 0.0)
        for seq in capture.sequences))
    receipt_times = [sample.receipt_ns for seq in capture.sequences
                     for sample in seq]
    if not receipt_times:
        raise ValueError("held-out capture has no phase samples")

    # The existing evaluator supplies the exact 16-sample causal warmup and
    # non-overlapping 32-sample scoring windows used by observer training.
    x, y, score_samples, _, _, invalid, unscored = observer_eval._score_arrays(
        capture, min(receipt_times), max(receipt_times), [],
        train_sensor_observer.DEFAULT_BURN_IN,
        train_sensor_observer.DEFAULT_SCORE_STEPS)
    sensor = x[:, train_sensor_observer.DEFAULT_BURN_IN:].reshape(-1, 10)
    truth = y[:, train_sensor_observer.DEFAULT_BURN_IN:, :3].reshape(-1, 3)
    stamps = [sample.source_stamp_ns for sample in score_samples]

    def exact_estimates(path: Path) -> tuple[np.ndarray, int]:
        states = _replay_states(path)
        missing = [stamp for stamp in stamps if stamp not in states]
        if missing:
            raise ValueError(
                f"{path} misses {len(missing)}/{len(stamps)} scored source stamps")
        return np.stack([states[stamp] for stamp in stamps]), len(stamps)

    production, production_matches = exact_estimates(production_replay)
    candidates: dict[str, np.ndarray] = {
        "production_odom": production,
    }
    candidate_matches = None
    if candidate_replay is not None:
        candidate, candidate_matches = exact_estimates(candidate_replay)
        candidates["candidate_odom"] = candidate

    torch, nn = train_sensor_observer._torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    report = json.loads(training_report.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_meta = payload["metadata"]
    if model_meta["sensor_feature_names"] != list(prepare_dataset.SENSOR_FEATURE_NAMES):
        raise ValueError("checkpoint sensor feature layout does not match evaluator")
    model_type = train_sensor_observer._make_model(
        nn, len(model_meta["sensor_feature_names"]),
        int(model_meta["hidden_size"]))
    model = model_type().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    norm = report["normalization"]
    sensor_mean = np.asarray(norm["sensor_mean"], dtype=np.float32)
    sensor_scale = np.asarray(norm["sensor_scale"], dtype=np.float32)
    truth_mean = np.asarray(norm["target_mean"], dtype=np.float32)
    truth_scale = np.asarray(norm["target_scale"], dtype=np.float32)
    with torch.no_grad():
        normalized = (x - sensor_mean[None, None, :]) / sensor_scale[None, None, :]
        prediction = model(torch.as_tensor(
            normalized, dtype=torch.float32, device=device))
        prediction = prediction[:, train_sensor_observer.DEFAULT_BURN_IN:]
        observer_uv = (prediction * torch.as_tensor(
            truth_scale, device=device) + torch.as_tensor(
            truth_mean, device=device)).cpu().numpy().reshape(-1, 2)

    learned = np.column_stack((observer_uv, sensor[:, 6]))
    wheel_baseline = np.column_stack((sensor[:, 2:4].mean(axis=1),
                                      np.zeros(len(sensor)), sensor[:, 6]))
    candidates["sensor_only_gru"] = learned
    candidates["rear_encoder_mean_zero_lateral"] = wheel_baseline

    sensor_windows = x[:, train_sensor_observer.DEFAULT_BURN_IN:]

    speed = np.hypot(truth[:, 0], truth[:, 1])
    abs_steer = np.abs(sensor[:, 0])
    throttle = sensor[:, 1]
    encoder_residual = np.abs(sensor[:, 2:4].mean(axis=1) - truth[:, 0])
    lateral_accel_proxy = np.abs(truth[:, 0] * truth[:, 2])
    regions = {
        "all_scored_samples": np.ones(len(truth), dtype=bool),
        "speed_ge_6_mps": speed >= 6.0,
        "speed_ge_8_mps": speed >= 8.0,
        "steering_ge_0p30_rad": abs_steer >= 0.30,
        "steering_ge_0p40_rad": abs_steer >= 0.40,
        "steering_ge_0p42_rad": abs_steer >= 0.42,
        "speed_ge_6_and_steering_ge_0p30": (speed >= 6.0) & (abs_steer >= 0.30),
        "speed_ge_6_and_steering_ge_0p40": (speed >= 6.0) & (abs_steer >= 0.40),
        "speed_ge_6_and_steering_ge_0p42": (speed >= 6.0) & (abs_steer >= 0.42),
        "speed_ge_6_and_lateral_accel_proxy_ge_6":
            (speed >= 6.0) & (lateral_accel_proxy >= 6.0),
        "encoder_mismatch_ge_0p5_mps": encoder_residual >= 0.5,
        "encoder_mismatch_ge_1_mps": encoder_residual >= 1.0,
        "wheel_mismatch_and_high_throttle":
            (encoder_residual >= 0.5) & (throttle >= 0.30),
    }
    results: dict[str, Any] = {}
    for estimator, estimates in candidates.items():
        error = estimates - truth
        results[estimator] = {
            "all": _metrics(error),
            "regions": {
                name: _metrics(error[mask])
                for name, mask in regions.items() if np.any(mask)
            },
        }

    result = {
        "run_id": run_id,
        "dataset_split": split,
        "raw_bag": str(raw_bag.resolve()),
        "dataset": str(dataset.resolve()),
        "training_report": str(training_report.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "production_replay": str(production_replay.resolve()),
        "candidate_replay": (str(candidate_replay.resolve())
                              if candidate_replay else None),
        "capture_quality": {
            "aborted": capture.aborted,
            "collisions_start_end": [capture.collision_count_start,
                                      capture.collision_count_end],
            "timing_faults": capture.timing_faults,
            "valid_invalid_unscored_phases": [capture.valid_phase_count,
                                                capture.invalid_phase_count,
                                                capture.unscored_phase_count],
            "quality_failures": failures,
        },
        "evaluation": {
            "truth": "bridge odometry is offline label only",
            "observer_inputs": "causal encoder, IMU, actuator feedback/commands and sample dt only",
            "window": {"burn_in_steps": train_sensor_observer.DEFAULT_BURN_IN,
                       "scored_steps": train_sensor_observer.DEFAULT_SCORE_STEPS},
            "scored_samples": int(len(truth)),
            "exact_replay_timestamp_matches": {
                "production": production_matches,
                "candidate": candidate_matches,
            },
            "invalid_sensor_samples": invalid,
            "unscored_short_or_invalid_samples": unscored,
            "lateral_acceleration_regime": "|u*r| kinematic proxy, not measured tire force",
        },
        "estimators": results,
        "relative_motion": _relative_motion_metrics(
            candidates, sensor_windows, score_samples, raw_bag),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True,
                                 allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-bag", type=Path, required=True)
    parser.add_argument("--production-replay", type=Path, required=True)
    parser.add_argument("--candidate-replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        result = score(args.dataset, args.training_report, args.checkpoint,
                       args.raw_bag, args.production_replay,
                       args.candidate_replay, args.output, args.device)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"observer/odometry scoring failed: {exc}\n")
    print(json.dumps({
        "run_id": result["run_id"],
        "scored_samples": result["evaluation"]["scored_samples"],
        "estimators": {name: scores["all"]["rmse"]
                       for name, scores in result["estimators"].items()},
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
