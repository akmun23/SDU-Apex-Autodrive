#!/usr/bin/env python3
"""Score production sensor-odometry replay against offline simulator truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools import evaluate_open_plane_body_dynamics as body


COM_X_M = 0.15532
REPLAY_TOPIC = "/replayed_odom"
REPLAY_DIAGNOSTICS_TOPIC = "/replayed_odom_diagnostics"
REQUIRED_SENSOR_TOPICS = {
    "/autodrive/roboracer_1/left_encoder",
    "/autodrive/roboracer_1/right_encoder",
    "/autodrive/roboracer_1/imu",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def _read_replay(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = body.analysis._topic_map(connection)
        missing = {REPLAY_TOPIC, REPLAY_DIAGNOSTICS_TOPIC} - set(topics)
        if missing:
            raise ValueError(f"replay bag lacks topics {sorted(missing)}: {path}")
        odometry_rows = list(body.analysis._messages(
            connection, topics, REPLAY_TOPIC))
        diagnostic_rows = list(body.analysis._messages(
            connection, topics, REPLAY_DIAGNOSTICS_TOPIC))
        if len(odometry_rows) != len(diagnostic_rows):
            raise ValueError("replay odometry and diagnostics counts do not match")
        result = {}
        for (odom_receipt, message), (diag_receipt, diagnostics) in zip(
                odometry_rows, diagnostic_rows):
            if abs(odom_receipt - diag_receipt) > 10_000_000:
                raise ValueError("odometry and diagnostics ordering is not aligned")
            stamp = body.analysis._stamp_ns(message.header.stamp)
            twist = message.twist.twist
            pose = message.pose.pose
            state = np.asarray((twist.linear.x, twist.linear.y,
                                twist.angular.z), dtype=np.float64)
            position = np.asarray((pose.position.x, pose.position.y,
                                   _yaw(pose.orientation)), dtype=np.float64)
            diagnostic = np.asarray(diagnostics.data, dtype=np.float64)
            if (stamp > 0 and len(diagnostic) == 28
                    and np.isfinite(state).all()
                    and np.isfinite(position).all()
                    and np.isfinite(diagnostic).all()):
                result[stamp] = (state, position, diagnostic)
        return result
    finally:
        connection.close()


def _quality_manifest(raw_bag: Path, allow_training_split: bool = False
                      ) -> tuple[str, dict[str, Any], Path]:
    manifest_path = raw_bag.parent.parent / "source_continuous_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = raw_bag.parent.parent.name
    matches = [row for row in manifest.get("runs", [])
               if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one run: {run_id}")
    run = matches[0]
    alignment = run.get("packet_sequence_alignment", {})
    allowed_splits = {"validation", "train"} if allow_training_split else {"validation"}
    if (run.get("effective_split") not in allowed_splits
            or run.get("aborted")
            or run.get("reason") != "schedule complete"
            or not run.get("clean_stream_and_collision_gate")
            or run.get("quality_failures")
            or run.get("whole_bag_quality_failures")
            or int(run.get("timing_faults", -1)) != 0
            or any(int(value) != 0 for value in run.get("collisions", []))
            or float(alignment.get("match_fraction", 0.0)) < 0.999
            or any(float(row.get("hz", 0.0)) < 38.0
                   for row in run.get("streams", {}).values())):
        raise ValueError(f"raw capture failed its frozen quality gates: {run_id}")
    return run_id, run, manifest_path


def _stats(error: np.ndarray) -> dict[str, Any]:
    if not len(error):
        return {"samples": 0, "rmse": None, "bias": None, "p95_abs": None}
    return {
        "samples": int(len(error)),
        "rmse": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "p95_abs": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
    }


def _relative_pose_metrics(bounds: np.ndarray, truth_pose: np.ndarray,
                           reset_indices: np.ndarray,
                           aligned_outputs: dict[int, tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray]]
                           ) -> list[dict[str, Any]]:
    metrics = []
    for sequence_index, (begin_value, end_value) in enumerate(bounds):
        begin, end = int(begin_value), int(end_value)
        indices = np.asarray([
            index for index in range(begin, end)
            if index in aligned_outputs
            and np.isfinite(truth_pose[index]).all()
        ], dtype=np.int64)
        if len(indices) < 2:
            continue
        predicted_pose = np.stack([aligned_outputs[i][1] for i in indices])
        local_truth_pose = truth_pose[indices]
        # Align the odometry's arbitrary world origin and initial heading once
        # per reset-delimited sequence; subsequent pose is scored unanchored.
        heading_offset = local_truth_pose[0, 2] - predicted_pose[0, 2]
        cosine, sine = math.cos(heading_offset), math.sin(heading_offset)
        relative = predicted_pose[:, :2] - predicted_pose[0, :2]
        aligned = local_truth_pose[0, :2] + np.column_stack((
            cosine * relative[:, 0] - sine * relative[:, 1],
            sine * relative[:, 0] + cosine * relative[:, 1],
        ))
        position_error = np.linalg.norm(
            aligned - local_truth_pose[:, :2], axis=1)
        heading = np.arctan2(
            np.sin(predicted_pose[:, 2] + heading_offset
                   - local_truth_pose[:, 2]),
            np.cos(predicted_pose[:, 2] + heading_offset
                   - local_truth_pose[:, 2]))
        metrics.append({
            "sequence_index": sequence_index,
            "reset_index": int(reset_indices[sequence_index]),
            "matched_samples": len(indices),
            "duration_s": (len(indices) - 1) * 0.025,
            "position_trajectory_rmse_m": float(
                np.sqrt(np.mean(position_error ** 2))),
            "position_p95_m": float(np.quantile(position_error, 0.95)),
            "position_endpoint_error_m": float(position_error[-1]),
            "heading_trajectory_rmse_rad": float(np.sqrt(np.mean(heading ** 2))),
            "heading_endpoint_error_rad": float(heading[-1]),
        })
    return metrics


def score(raw_bag: Path, replay_bag: Path, replay_metadata: Path,
          output: Path, sequence_index: int | None = None,
          allow_training_split: bool = False
          ) -> dict[str, Any]:
    raw_bag, replay_bag, replay_metadata, output = map(
        Path.resolve, (raw_bag, replay_bag, replay_metadata, output))
    if output.exists():
        raise FileExistsError(output)
    run_id, quality, manifest_path = _quality_manifest(
        raw_bag, allow_training_split=allow_training_split)
    replay_info = json.loads(replay_metadata.read_text(encoding="utf-8"))
    overrides = replay_info.get("parameter_overrides", {})
    if (Path(replay_info["input_bag"]).resolve() != raw_bag
            or set(replay_info["sensor_topics"]) != REQUIRED_SENSOR_TOPICS
            or not isinstance(overrides, dict)
            or set(overrides) - {
                "wheel_burst_catchup_accel_mps2",
                "turn_speed_bias_yaw_rate_abs_mps",
                "turn_speed_bias_max_mps",
            }
            or ("turn_speed_bias_yaw_rate_abs_mps" in overrides) !=
               ("turn_speed_bias_max_mps" in overrides)):
        raise ValueError(
            "replay must use sensor-only inputs and either the production node "
            "or the single documented burst-catch-up ablation")
    playback_window = replay_info.get("playback_window")
    if sequence_index is None:
        if playback_window is not None:
            raise ValueError("isolated replay requires --sequence-index when scoring")
    elif (not isinstance(playback_window, dict)
          or playback_window.get("source_sequence_index") != sequence_index
          or not playback_window.get("observer_reinitialized_for_segment")):
        raise ValueError("replay window does not match the requested reset sequence")

    replay = _read_replay(replay_bag)
    source_path = (raw_bag.parent.parent
                   / "source_continuous_v1/openplane_dynamics.npz")
    with np.load(source_path, allow_pickle=False) as source:
        stamps = np.asarray(source["sample_time_ns"], dtype=np.int64)
        frames = np.asarray(source["frames"], dtype=np.float64)
        feature_names = source["feature_names"].astype(str).tolist()
        rigid = np.asarray(source["simulator_rigid_state"], dtype=np.float64)
        truth_pose = np.asarray(source["simulator_pose_xyyaw"], dtype=np.float64)
        bounds = np.asarray(source["sequence_bounds"], dtype=np.int64)
        reset_indices = np.asarray(source["sequence_reset_index"], dtype=np.int64)
        reset_frames = np.asarray(source["frame_reset_index"], dtype=np.int64)
    if (len(stamps) != len(frames) or rigid.shape != (len(stamps), 13)
            or truth_pose.shape != (len(stamps), 3)
            or len(bounds) != len(reset_indices)
            or reset_frames.shape != (len(stamps),)):
        raise ValueError("source dynamics arrays do not share aligned rows")
    if sequence_index is None:
        eligible_indices = np.arange(len(stamps), dtype=np.int64)
    else:
        if not 0 <= sequence_index < len(bounds):
            raise ValueError(f"sequence index {sequence_index} outside [0, {len(bounds)})")
        begin, end = map(int, bounds[sequence_index])
        eligible_indices = np.arange(begin, end, dtype=np.int64)
    steer_index = feature_names.index("steering_feedback_rad")
    replay_stamps = np.asarray(sorted(replay), dtype=np.int64)
    eligible_stamps = stamps[eligible_indices]
    right = np.clip(np.searchsorted(replay_stamps, eligible_stamps),
                    0, len(replay_stamps) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(np.abs(replay_stamps[left] - eligible_stamps)
                       <= np.abs(replay_stamps[right] - eligible_stamps),
                       left, right)
    offsets_ns = replay_stamps[nearest] - eligible_stamps
    local_matched = np.flatnonzero(np.abs(offsets_ns) <= 10_000_000)
    matched_indices = eligible_indices[local_matched]
    matched_nearest = nearest[local_matched]
    if len(np.unique(matched_nearest)) != len(matched_indices):
        raise ValueError("nearest replay-to-label timestamp mapping is not one-to-one")
    match_fraction = len(matched_indices) / len(eligible_indices)
    if match_fraction < 0.98:
        raise ValueError(
            f"replay has insufficient source-stamp coverage: {match_fraction:.4%}")

    aligned_outputs = {
        int(index): replay[int(replay_stamps[nearest_index])]
        for index, nearest_index in zip(matched_indices, matched_nearest)
    }
    predicted = np.stack([aligned_outputs[int(i)][0] for i in matched_indices])
    truth = np.stack([
        (rigid[i, 7], rigid[i, 8] - COM_X_M * rigid[i, 12], rigid[i, 12])
        for i in matched_indices
    ])
    absolute_steer = np.abs(frames[matched_indices, steer_index])
    speed = truth[:, 0]
    highsteer = absolute_steer >= 0.30
    highspeed_highsteer = highsteer & (speed >= 7.5)
    errors = predicted - truth
    sequence_pose = _relative_pose_metrics(
        bounds, truth_pose, reset_indices, aligned_outputs)
    sequence_twist = []
    for sequence_index, (begin_value, end_value) in enumerate(bounds):
        begin, end = int(begin_value), int(end_value)
        indices = np.asarray([
            index for index in range(begin, end)
            if index in aligned_outputs
        ], dtype=np.int64)
        if not len(indices):
            continue
        local_prediction = np.stack([aligned_outputs[int(i)][0] for i in indices])
        local_diagnostic = np.stack([
            aligned_outputs[int(i)][2] for i in indices
        ])
        local_truth = np.column_stack((
            rigid[indices, 7],
            rigid[indices, 8] - COM_X_M * rigid[indices, 12],
            rigid[indices, 12],
        ))
        local_error = local_prediction - local_truth
        local_steer = np.abs(frames[indices, steer_index])
        local_highsteer = local_steer >= 0.30
        sequence_twist.append({
            "sequence_index": sequence_index,
            "reset_index": int(reset_indices[sequence_index]),
            "samples": len(indices),
            "max_speed_mps": float(np.max(local_truth[:, 0])),
            "max_abs_steering_rad": float(np.max(local_steer)),
            "body_twist_error_all": _stats(local_error),
            "body_twist_error_abs_steer_ge_0p30_rad": _stats(
                local_error[local_highsteer]),
            "observer_diagnostics": {
                "wheel_raw_mps_median": float(np.median(local_diagnostic[:, 3])),
                "wheel_mapped_mps_median": float(np.median(local_diagnostic[:, 4])),
                "speed_prediction_mps_median": float(np.median(local_diagnostic[:, 5])),
                "speed_state_mps_median": float(np.median(local_diagnostic[:, 6])),
                "wheel_packet_mps_median": float(np.median(local_diagnostic[:, 25])),
                "wheel_update_fraction": float(np.mean(local_diagnostic[:, 12] > 0.5)),
                "turn_mode_fraction": float(np.mean(local_diagnostic[:, 13] > 0.5)),
                "reset_epoch_count": int(np.sum(local_diagnostic[:, 14] > 0.5)),
                "timing_degraded_fraction": float(np.mean(local_diagnostic[:, 15] > 0.5)),
                "sensor_outlier_fraction": float(np.mean(local_diagnostic[:, 21] > 0.5)),
                "wheel_burst_rejected_fraction": float(
                    np.mean(local_diagnostic[:, 26] > 0.5)),
            },
        })
    pose_rmse = np.asarray([
        row["position_trajectory_rmse_m"] for row in sequence_pose
    ], dtype=np.float64)
    if not len(pose_rmse):
        raise ValueError("no reset-delimited pose sequence could be scored")
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "data_split": quality["effective_split"],
        "raw_bag": str(raw_bag),
        "raw_bag_sha256": _sha256(raw_bag),
        "replay_bag": str(replay_bag),
        "replay_bag_sha256": _sha256(replay_bag),
        "replay_metadata": str(replay_metadata),
        "source_dynamics": str(source_path),
        "source_dynamics_sha256": _sha256(source_path),
        "replay_image_id": replay_info["image_id"],
        "observer_parameter_overrides": overrides,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "quality": {
            "clean_stream_and_collision_gate": bool(
                quality["clean_stream_and_collision_gate"]),
            "timing_faults": int(quality["timing_faults"]),
            "collisions": quality["collisions"],
            "stream_hz": {name: float(value["hz"])
                          for name, value in quality["streams"].items()},
        },
        "causality": {
            "node_input_topics": sorted(REQUIRED_SENSOR_TOPICS),
            "simulator_truth_used_only_for_scoring": True,
            "pose_alignment": "one rigid SE(2) alignment at the first sample of each reset-delimited sequence",
            "observer_lifecycle": (
                "fresh production observer instance for this isolated reset sequence; "
                "reset events are not supplied during the sequence"
                if sequence_index is not None else
                "one production observer instance processed the full bag; "
                "simulator reset events were not supplied to it"
            ),
        },
        "requested_sequence_index": sequence_index,
        "coverage": {
            "source_samples": len(eligible_indices),
            "replay_samples": len(replay),
            "nearest_source_stamp_matches": len(matched_indices),
            "nearest_match_fraction": match_fraction,
            "nearest_match_abs_offset_ms_p95": float(
                np.quantile(np.abs(offsets_ns[local_matched]), 0.95) / 1e6),
            "nearest_match_abs_offset_ms_max": float(
                np.max(np.abs(offsets_ns[local_matched])) / 1e6),
            "alignment_tolerance_ms": 10.0,
            "sequences_scored": len(sequence_pose),
        },
        "body_twist_error": {
            "interpretation": (
                "scores only the requested reset-isolated segment; observer state "
                "was freshly initialized for this segment"
                if sequence_index is not None else
                "aggregate covers one continuous observer across simulator resets; "
                "inspect per-sequence metrics because internal observer state "
                "was not reinitialized"
            ),
            "all": _stats(errors),
            "abs_steer_ge_0p30_rad": _stats(errors[highsteer]),
            "speed_ge_7p5_and_abs_steer_ge_0p30_rad": _stats(
                errors[highspeed_highsteer]),
        },
        "body_twist_error_per_sequence": sequence_twist,
        "relative_pose_per_sequence": sequence_pose,
        "relative_pose_summary": {
            "sequence_count": len(pose_rmse),
            "position_rmse_macro_m": float(pose_rmse.mean()),
            "position_rmse_p95_m": float(np.quantile(pose_rmse, 0.95)),
            "worst_position_rmse_m": float(pose_rmse.max()),
            "median_endpoint_error_m": float(np.median([
                row["position_endpoint_error_m"] for row in sequence_pose])),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-bag", type=Path, required=True)
    parser.add_argument("--replay-bag", type=Path, required=True)
    parser.add_argument("--replay-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int,
                        help="score only this reset-isolated source sequence")
    parser.add_argument("--allow-training-split", action="store_true",
                        help="permit a quality-gated training capture; do not treat its score as held-out")
    args = parser.parse_args()
    report = score(args.raw_bag, args.replay_bag, args.replay_metadata,
                   args.output, args.sequence_index, args.allow_training_split)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "run_id": report["run_id"],
        "nearest_match_fraction": report["coverage"]["nearest_match_fraction"],
        "body_twist_error": report["body_twist_error"],
        "relative_pose_summary": report["relative_pose_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
