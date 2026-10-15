#!/usr/bin/env python3
"""Replay clean training bags through production odom and fit a u residual.

Only encoder and IMU topics are played into odometry. Simulator truth is read
from the closed source bags offline as the residual label. Practice/test bags
are excluded by the dataset manifest and remain sealed for final evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import evaluate_open_plane_body_dynamics as body
from tools import analyze_open_plane_dynamics as analysis
from tools.vehicle_dynamics_learning import prepare_dataset


DEFAULT_MANIFEST = (REPO_ROOT / "live_runs/derived_dynamics_learning_20260928"
                    / "attitude_dataset_20260929_clean/manifest.json")
SENSOR_TOPICS = (analysis.LEFT_ENCODER, analysis.RIGHT_ENCODER, analysis.IMU)
FEATURE_NAMES = (
    "odom_u_mps", "odom_v_mps", "encoder_mean_100ms_mps",
    "imu_ax_mps2", "imu_ay_mps2", "imu_yaw_rate_rps", "sample_dt_s",
    "steering_feedback_rad", "throttle_feedback_norm",
    "steering_command_rad", "throttle_command_norm",
)
ALPHAS = (0.1, 1.0, 10.0, 100.0)
MIN_REPLAY_COVERAGE = 0.98


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _OdomCollector:
    def __new__(cls, node_class):
        class Collector(node_class):
            def __init__(self):
                super().__init__("odom_residual_training_collector")
                from nav_msgs.msg import Odometry
                self.states: dict[int, np.ndarray] = {}
                self.create_subscription(
                    Odometry, "/replayed_odom", self._receive, 100)

            def _receive(self, message):
                stamp = analysis._stamp_ns(message.header.stamp)
                if stamp is None:
                    return
                twist = message.twist.twist
                self.states[stamp] = np.asarray(
                    (twist.linear.x, twist.linear.y, twist.angular.z),
                    dtype=np.float64)

        return Collector


def _replay_one(bag: Path, node_binary: Path, params_file: Path,
                rate: float, domain_id: int, collector,
                log_dir: Path,
                node_parameter_overrides: tuple[str, ...] = ()) -> dict[int, np.ndarray]:
    import rclpy

    environment = os.environ.copy()
    environment.update({
        "ROS_DOMAIN_ID": str(domain_id),
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    })
    collector.states.clear()
    node_command = [
        str(node_binary), "--ros-args", "--params-file", str(params_file),
        "-p", "publish_tf:=false", "-p", "odom_topic:=/replayed_odom",
        "-p", "diagnostics_topic:=/replayed_odom_diagnostics",
        *node_parameter_overrides,
    ]
    log_dir.mkdir(parents=True, exist_ok=True)
    node_log = log_dir / "node.log"
    with node_log.open("w", encoding="utf-8") as log:
        node_process = subprocess.Popen(
            node_command, env=environment, stdout=log,
            stderr=subprocess.STDOUT, text=True)
        player = None
        try:
            deadline = time.monotonic() + 15.0
            stable_subscriber_polls = 0
            while time.monotonic() < deadline:
                rclpy.spin_once(collector, timeout_sec=0.05)
                if node_process.poll() is not None:
                    raise RuntimeError(
                        f"odometry node exited before replay; see {node_log}")
                ready = (collector.count_publishers("/replayed_odom") > 0
                         and all(collector.count_subscribers(topic) > 0
                                 for topic in SENSOR_TOPICS))
                stable_subscriber_polls = (
                    stable_subscriber_polls + 1 if ready else 0)
                if stable_subscriber_polls >= 5:
                    # Wait for DDS endpoint matching after discovery reports
                    # all three subscriptions, before the first source packet.
                    ready_at = time.monotonic() + 2.0
                    while time.monotonic() < ready_at:
                        rclpy.spin_once(collector, timeout_sec=0.05)
                    break
            else:
                raise RuntimeError("odometry output publisher discovery timed out")

            player_command = [
                "ros2", "bag", "play", str(bag), "--rate", str(rate),
                "--disable-keyboard-controls", "--topics", *SENSOR_TOPICS,
            ]
            player_log = log_dir / "player.log"
            with player_log.open("w", encoding="utf-8") as player_output:
                player = subprocess.Popen(
                    player_command, env=environment, stdout=player_output,
                    stderr=subprocess.STDOUT, text=True)
                while player.poll() is None:
                    rclpy.spin_once(collector, timeout_sec=0.01)
                    if node_process.poll() is not None:
                        raise RuntimeError(
                            f"odometry node exited during replay; see {node_log}")
                if player.returncode != 0:
                    raise RuntimeError(
                        f"bag playback failed ({player.returncode}); see {player_log}")
            drain_until = time.monotonic() + 0.5
            while time.monotonic() < drain_until:
                rclpy.spin_once(collector, timeout_sec=0.01)
            return dict(collector.states)
        finally:
            if player is not None and player.poll() is None:
                player.send_signal(signal.SIGINT)
                try:
                    player.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    player.kill()
                    player.wait()
            if node_process.poll() is None:
                node_process.send_signal(signal.SIGINT)
                try:
                    node_process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    node_process.kill()
                    node_process.wait()


def _balanced_weights(groups: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(groups, return_inverse=True,
                                   return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights / np.mean(weights)


def _normalizer(x: np.ndarray, weights: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    return mean, np.sqrt(np.maximum(variance, 1e-8))


def _group_folds(groups: np.ndarray, n_splits: int = 5
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic, row-balanced folds without splitting a run."""
    unique, inverse, counts = np.unique(
        groups, return_inverse=True, return_counts=True)
    if len(unique) < n_splits:
        raise ValueError(f"need at least {n_splits} run groups for CV")
    fold_rows = np.zeros(n_splits, dtype=np.int64)
    group_fold = np.full(len(unique), -1, dtype=np.int64)
    order = sorted(range(len(unique)), key=lambda index: (
        -int(counts[index]), str(unique[index])))
    for group_index in order:
        fold = int(np.argmin(fold_rows))
        group_fold[group_index] = fold
        fold_rows[fold] += counts[group_index]
    row_folds = group_fold[inverse]
    all_rows = np.arange(len(groups), dtype=np.int64)
    return [
        (all_rows[row_folds != fold], all_rows[row_folds == fold])
        for fold in range(n_splits)
    ]


def _ridge_predict(x_train: np.ndarray, y_train: np.ndarray,
                   weights: np.ndarray, x_test: np.ndarray,
                   alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted Ridge with an unpenalized intercept using NumPy only."""
    mean, scale = _normalizer(x_train, weights)
    normalized_train = (x_train - mean) / scale
    normalized_test = (x_test - mean) / scale
    y_mean = float(np.average(y_train, weights=weights))
    centered_y = y_train - y_mean
    gram = normalized_train.T @ (weights[:, None] * normalized_train)
    rhs = normalized_train.T @ (weights * centered_y)
    coefficient = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    return normalized_test @ coefficient + y_mean, mean, scale


def _fit(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
         feature_names: tuple[str, ...], history_steps: int
         ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    folds = _group_folds(groups, n_splits=5)
    scores = []
    for alpha in ALPHAS:
        run_errors: dict[str, list[float]] = {}
        run_baseline: dict[str, list[float]] = {}
        for train_index, test_index in folds:
            weights = _balanced_weights(groups[train_index])
            prediction, _, _ = _ridge_predict(
                x[train_index], y[train_index], weights, x[test_index], alpha)
            for run_id in np.unique(groups[test_index]):
                mask = groups[test_index] == run_id
                run_errors.setdefault(str(run_id), []).extend(
                    (y[test_index][mask] - prediction[mask]).tolist())
                run_baseline.setdefault(str(run_id), []).extend(
                    y[test_index][mask].tolist())
        corrected = {run: float(np.sqrt(np.mean(np.square(values))))
                     for run, values in run_errors.items()}
        baseline = {run: float(np.sqrt(np.mean(np.square(values))))
                    for run, values in run_baseline.items()}
        scores.append({
            "alpha": alpha,
            "mean_run_corrected_u_rmse_mps": float(np.mean(list(corrected.values()))),
            "median_run_corrected_u_rmse_mps": float(np.median(list(corrected.values()))),
            "mean_run_baseline_u_rmse_mps": float(np.mean(list(baseline.values()))),
            "median_run_baseline_u_rmse_mps": float(np.median(list(baseline.values()))),
            "runs_improved": int(sum(corrected[run] < baseline[run]
                                      for run in corrected)),
            "per_run_corrected_u_rmse_mps": corrected,
            "per_run_baseline_u_rmse_mps": baseline,
        })
    selected = min(scores, key=lambda row: (
        row["median_run_corrected_u_rmse_mps"],
        row["mean_run_corrected_u_rmse_mps"]))
    weights = _balanced_weights(groups)
    mean, scale = _normalizer(x, weights)
    y_mean = float(np.average(y, weights=weights))
    normalized_x = (x - mean) / scale
    gram = normalized_x.T @ (weights[:, None] * normalized_x)
    rhs = normalized_x.T @ (weights * (y - y_mean))
    coefficient = np.linalg.solve(
        gram + float(selected["alpha"]) * np.eye(gram.shape[0]), rhs)
    payload = {
        "schema_version": 1,
        "target": "truth_u_minus_current_source_odom_u",
        "feature_names": list(feature_names),
        "base_feature_names": list(FEATURE_NAMES),
        "history_steps": history_steps,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficient": coefficient.tolist(),
        "intercept": y_mean,
        "selected_alpha": float(selected["alpha"]),
        "selection_metric": "median_run_corrected_u_rmse_mps",
    }
    return payload, scores


def _history_rows(features: np.ndarray, targets: np.ndarray,
                  groups: np.ndarray, source_stamps: np.ndarray,
                  packet_sequences: np.ndarray, history_steps: int,
                  base_feature_names: tuple[str, ...] = FEATURE_NAMES
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                             tuple[str, ...], np.ndarray, np.ndarray]:
    """Build causal fixed-grid windows without crossing run/packet gaps."""
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    order = np.lexsort((source_stamps, groups))
    features = features[order]
    targets = targets[order]
    groups = groups[order]
    source_stamps = source_stamps[order]
    packet_sequences = packet_sequences[order]
    output_x: list[np.ndarray] = []
    output_y: list[float] = []
    output_groups: list[str] = []
    output_stamps: list[int] = []
    output_packets: list[int] = []
    segment_start = 0
    for index in range(1, len(features) + 1):
        contiguous = (
            index < len(features)
            and groups[index] == groups[index - 1]
            and packet_sequences[index] == packet_sequences[index - 1] + 1
            and source_stamps[index] > source_stamps[index - 1])
        if contiguous:
            continue
        segment_end = index
        for current in range(segment_start + history_steps - 1, segment_end):
            begin = current - history_steps + 1
            output_x.append(features[begin:current + 1].reshape(-1))
            output_y.append(float(targets[current]))
            output_groups.append(str(groups[current]))
            output_stamps.append(int(source_stamps[current]))
            output_packets.append(int(packet_sequences[current]))
        segment_start = index
    if not output_x:
        raise ValueError("no contiguous samples remain for requested history")
    names = tuple(
        f"{name}_lag{lag}"
        for lag in range(history_steps - 1, -1, -1)
        for name in base_feature_names)
    return (np.stack(output_x), np.asarray(output_y, dtype=np.float64),
            np.asarray(output_groups, dtype=str), names,
            np.asarray(output_stamps, dtype=np.int64),
            np.asarray(output_packets, dtype=np.int64))


def _clean_training_records(manifest_paths: tuple[Path, ...]
                            ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Merge only clean train rows and reject conflicting duplicate IDs."""
    records: dict[str, dict[str, Any]] = {}
    provenance = []
    for path in manifest_paths:
        resolved = path.resolve()
        document = json.loads(resolved.read_text(encoding="utf-8"))
        provenance.append({"path": str(resolved), "sha256": _sha256(resolved)})
        for row in document.get("runs", []):
            if (row.get("effective_split") != "train"
                    or row.get("clean_stream_and_collision_gate") is not True):
                continue
            run_id = str(row.get("run_id", ""))
            if not run_id or not row.get("bag"):
                raise ValueError(f"training manifest row lacks run_id/bag: {resolved}")
            previous = records.get(run_id)
            if previous is not None:
                # The manifests fingerprint different derived feature views;
                # the source-bag path/size plus the trainer's SHA-256 cache
                # check define raw-capture identity.
                identity = ("bag", "bytes")
                if any(previous.get(key) != row.get(key) for key in identity):
                    raise ValueError(
                        f"conflicting source identity for training run {run_id}")
                continue
            records[run_id] = dict(row)
    result = [records[key] for key in sorted(records)]
    if len(result) < 5:
        raise ValueError("fewer than five clean training captures across manifests")
    return result, provenance


def train(manifest_path: Path, node_binary: Path, params_file: Path,
          output_dir: Path, rate: float, domain_id: int,
          resume: bool, force_replay_runs: set[str], history_steps: int,
          shared_cache_dir: Path | None,
          additional_manifest_paths: tuple[Path, ...] = (),
          max_planar_speed_mps: float | None = None,
          whole_phase_train_run_ids: set[str] | None = None) -> dict[str, Any]:
    if output_dir.exists() and not resume:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if (max_planar_speed_mps is not None
            and (not np.isfinite(max_planar_speed_mps)
                 or max_planar_speed_mps <= 0.0)):
        raise ValueError("max_planar_speed_mps must be finite and positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = shared_cache_dir or output_dir / "training_replay_cache"
    cache_dir.mkdir(exist_ok=True)
    manifest_paths = (manifest_path.resolve(),) + tuple(
        path.resolve() for path in additional_manifest_paths)
    records, manifest_provenance = _clean_training_records(manifest_paths)
    whole_phase_train_run_ids = whole_phase_train_run_ids or set()
    known_training_runs = {str(record["run_id"]) for record in records}
    unknown_whole_phase_runs = whole_phase_train_run_ids - known_training_runs
    if unknown_whole_phase_runs:
        raise ValueError(
            "whole-phase mode requested for non-admitted training run(s): "
            f"{sorted(unknown_whole_phase_runs)}")
    node_hash = _sha256(node_binary)
    params_hash = _sha256(params_file)

    os.environ.update({
        "ROS_DOMAIN_ID": str(domain_id),
        "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    })
    import rclpy
    from rclpy.node import Node

    collector_type = _OdomCollector(Node)
    rclpy.init(args=None)
    collector = collector_type()
    run_rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    group_rows: list[str] = []
    source_stamp_rows: list[int] = []
    packet_sequence_rows: list[int] = []
    try:
        for index, record in enumerate(records, start=1):
            bag = Path(record["bag"])
            if not bag.is_absolute():
                bag = REPO_ROOT / bag
            capture = body.load_capture(
                bag, continuous_phased_run=(
                    str(record["run_id"]) in whole_phase_train_run_ids))
            quality_ok, failures = prepare_dataset._quality(
                capture, fixed_packet_timebase=True)
            if not quality_ok:
                raise ValueError(f"{record['run_id']} failed fixed-grid gate: {failures}")
            cache = cache_dir / f"{record['run_id']}.npz"
            metadata_path = cache_dir / f"{record['run_id']}.json"
            bag_hash = _sha256(bag)
            reusable = False
            if (resume and record["run_id"] not in force_replay_runs
                    and cache.is_file() and metadata_path.is_file()):
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                reusable = (existing.get("bag_sha256") == bag_hash
                            and existing.get("node_sha256") == node_hash
                            and existing.get("params_sha256") == params_hash
                            and existing.get("playback_rate") == rate)
            if reusable:
                cached = np.load(cache, allow_pickle=False)
                states = {int(stamp): state for stamp, state in
                          zip(cached["source_stamp_ns"], cached["states"])}
            else:
                states = _replay_one(bag, node_binary, params_file, rate,
                                     domain_id, collector,
                                     cache_dir / "logs" / str(record["run_id"]))
                stamps = np.asarray(sorted(states), dtype=np.int64)
                values = np.stack([states[int(stamp)] for stamp in stamps])
                np.savez_compressed(cache, source_stamp_ns=stamps, states=values)
                metadata_path.write_text(json.dumps({
                    "run_id": record["run_id"], "bag": str(bag.resolve()),
                    "bag_sha256": bag_hash, "node_sha256": node_hash,
                    "params_sha256": params_hash, "playback_rate": rate,
                    "state_samples": len(stamps),
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            matched = 0
            included = 0
            seen_stamps: set[int] = set()
            eligible_stamps: set[int] = set()
            for sequence in capture.sequences:
                previous_packet = None
                for sample in sequence:
                    contiguous = (previous_packet is not None
                                  and sample.packet_sequence == previous_packet + 1)
                    previous_packet = sample.packet_sequence
                    if not contiguous:
                        continue
                    if sample.source_stamp_ns in seen_stamps:
                        continue
                    seen_stamps.add(sample.source_stamp_ns)
                    eligible_stamps.add(sample.source_stamp_ns)
                    state = states.get(sample.source_stamp_ns)
                    sensor = prepare_dataset._sensor_frame(sample)
                    if state is None or sensor is None:
                        continue
                    sensor = np.asarray(sensor, dtype=np.float64)
                    matched += 1
                    if (max_planar_speed_mps is not None
                            and np.hypot(sample.state[0], sample.state[1])
                            > max_planar_speed_mps):
                        continue
                    xrow = np.asarray((
                        state[0], state[1], np.mean(sensor[2:4]),
                        sensor[4], sensor[5], sensor[6], 0.025,
                        sensor[0], sensor[1], sensor[7], sensor[8],
                    ), dtype=np.float64)
                    feature_rows.append(xrow)
                    target_rows.append(float(sample.state[0] - state[0]))
                    group_rows.append(str(record["run_id"]))
                    source_stamp_rows.append(int(sample.source_stamp_ns))
                    packet_sequence_rows.append(int(sample.packet_sequence))
                    included += 1
            eligible = len(eligible_stamps)
            coverage = matched / eligible if eligible else 0.0
            if coverage < MIN_REPLAY_COVERAGE or matched < 40:
                raise ValueError(
                    f"{record['run_id']} current-source replay coverage "
                    f"{matched}/{eligible} ({coverage:.1%}) is inadequate; "
                    f"minimum is {MIN_REPLAY_COVERAGE:.0%}")
            run_rows.append({"run_id": record["run_id"],
                             "aligned_samples": matched,
                             "training_samples": included,
                             "replay_state_samples": len(states),
                             "coverage": coverage,
                             "bag": record["bag"]})
            print(f"[{index}/{len(records)}] {record['run_id']}: "
                  f"{matched}/{eligible} aligned, {included} in training range, "
                  f"{len(states)} odom outputs",
                  flush=True)
    finally:
        collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    base_x = np.asarray(feature_rows, dtype=np.float64)
    base_y = np.asarray(target_rows, dtype=np.float64)
    base_groups = np.asarray(group_rows, dtype=str)
    base_stamps = np.asarray(source_stamp_rows, dtype=np.int64)
    base_packets = np.asarray(packet_sequence_rows, dtype=np.int64)
    x, y, groups, feature_names, history_stamps, history_packets = _history_rows(
        base_x, base_y, base_groups, base_stamps, base_packets, history_steps)
    model, cv = _fit(x, y, groups, feature_names, history_steps)
    model["training_policy"] = (
        "clean manifest training captures only; current source odometry is "
        "generated by replaying only left/right encoders and IMU; simulator "
        "truth is an offline residual label; samples outside the configured "
        "planar-speed range are excluded from fitting")
    model["runtime_input_policy"] = (
        "current odometry u/v, 100 ms encoder-angle mean, causal IMU "
        "acceleration/yaw rate, latest received steering/throttle feedback, "
        "and latest previously received steering/throttle command. Feedback "
        "and commands are aligned causally by receipt time; no simulator "
        "truth, future measurements, AMCL, or transform topics")
    model["training_provenance"] = {
        "manifests": manifest_provenance,
        "odometry_node": str(node_binary.resolve()),
        "odometry_node_sha256": node_hash,
        "odometry_parameters": str(params_file.resolve()),
        "odometry_parameters_sha256": params_hash,
        "playback_rate": rate,
        "domain_id": domain_id,
        "run_count": len(run_rows),
        "sample_count": len(x),
        "history_steps": history_steps,
        "max_planar_speed_mps": max_planar_speed_mps,
        "whole_phase_train_run_ids": sorted(whole_phase_train_run_ids),
        "runs": run_rows,
        "cross_validation": cv,
    }
    data_path = output_dir / "training_rows.npz"
    model_path = output_dir / "odom_u_residual_model.json"
    if data_path.exists() or model_path.exists():
        raise FileExistsError("refusing to overwrite existing residual artifacts")
    np.savez_compressed(
        data_path, features=x, residual_u=y, run_id=groups.astype("U"),
        base_features=base_x, base_residual_u=base_y,
        base_run_id=base_groups.astype("U"), source_stamp_ns=base_stamps,
        packet_sequence=base_packets, history_source_stamp_ns=history_stamps,
        history_packet_sequence=history_packets)
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n", encoding="utf-8")
    summary = {"training_runs": len(run_rows), "samples": len(x),
               "history_steps": history_steps,
               "selected_alpha": model["selected_alpha"],
               "cross_validation": [
                   {"alpha": row["alpha"],
                    "baseline_mean_run_rmse": row["mean_run_baseline_u_rmse_mps"],
                    "corrected_mean_run_rmse": row["mean_run_corrected_u_rmse_mps"],
                    "baseline_median_run_rmse": row["median_run_baseline_u_rmse_mps"],
                    "corrected_median_run_rmse": row["median_run_corrected_u_rmse_mps"],
                    "runs_improved": row["runs_improved"]}
                   for row in cv],
               "model": str(model_path), "training_rows": str(data_path)}
    report_path = output_dir / "training_summary.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True)
                           + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--additional-training-manifest", type=Path,
                        action="append", default=[],
                        help="merge clean train-only rows from another manifest; repeatable")
    parser.add_argument("--node", type=Path, required=True,
                        help="built sensor_odometry_node executable")
    parser.add_argument("--params", type=Path,
                        default=REPO_ROOT / "f1tenth_localization/config/sensor_odometry.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--domain-id", type=int, default=187)
    parser.add_argument("--resume", action="store_true",
                        help="reuse per-run replays only when input/source/config hashes match")
    parser.add_argument("--cache-dir", type=Path,
                        help="reuse a hash-verified replay cache from a prior preparation run")
    parser.add_argument("--history-steps", type=int, default=1,
                        help="causal contiguous 25 ms history length used by the residual model")
    parser.add_argument("--max-planar-speed-mps", type=float,
                        help="optional planar-speed ceiling for training rows")
    parser.add_argument("--whole-phase-train-run-id", action="append", default=[],
                        help=("explicit admitted train-only phased capture IDs "
                              "whose single uninterrupted phase is used despite "
                              "a missing phase-valid flag"))
    parser.add_argument("--force-replay-run-id", action="append", default=[],
                        help="replay this run even if a cache exists; repeatable")
    args = parser.parse_args()
    try:
        if not 0.1 <= args.rate <= 4.0:
            raise ValueError("playback rate must be in [0.1, 4]")
        result = train(args.manifest.resolve(), args.node.resolve(),
                       args.params.resolve(), args.output_dir.resolve(),
                       args.rate, args.domain_id, args.resume,
                       set(args.force_replay_run_id), args.history_steps,
                       args.cache_dir.resolve() if args.cache_dir else None,
                       tuple(path.resolve() for path in
                             args.additional_training_manifest),
                       args.max_planar_speed_mps,
                       set(args.whole_phase_train_run_id))
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"odometry residual training failed: {exc}\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
