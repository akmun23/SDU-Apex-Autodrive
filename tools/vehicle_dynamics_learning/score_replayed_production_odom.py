#!/usr/bin/env python3
"""Score exact production-odometry replay against one clean OpenPlane capture.

The original bag is read-only.  Its simulator odometry is used only as an
offline label; production replay consumes the encoder/IMU streams only.
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
from tools import evaluate_open_plane_body_dynamics as body


TRUTH_TOPIC = analysis.ODOM
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


def score(raw_bag: Path, replay_bag: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite odometry score: {output}")
    capture = body.load_capture(raw_bag)
    # OpenPlane r03 completed without collisions, aborts, or timing faults,
    # but some paired speed-match phases were deliberately marked invalid.
    # load_capture includes only valid phases; score that explicit subset and
    # do not pretend the failed speed-match phases are part of the evaluation.
    if (capture.aborted or capture.collision_count_start != 0
            or capture.collision_count_end != capture.collision_count_start
            or capture.timing_faults or not capture.sequences):
        raise ValueError("capture failed abort/collision/timing quality gate")
    quality_stats = (capture.phase_stream_stats if capture.phase_stream_stats
                     else capture.stream_stats)
    bad_streams = [
        name for name, (rate, p95, gap) in quality_stats.items()
        if rate < 38.0 or p95 > 35.0
        or gap > (120.0 if name in body.COMMAND_STREAM_TOPICS else 60.0)
    ]
    if bad_streams:
        raise ValueError(f"capture streams failed timing gate: {bad_streams}")
    capture, coverage = body._attach_feature_states(capture, replay_bag,
                                                     REPLAY_TOPIC)

    # Valid phase context overlaps by design; count each source stamp once.
    samples_by_stamp = {
        sample.source_stamp_ns: sample
        for sequence in capture.sequences for sample in sequence
    }
    samples = [samples_by_stamp[key] for key in sorted(samples_by_stamp)]
    truth = np.stack([sample.state for sample in samples])
    estimate = np.stack([sample.feature_state for sample in samples])
    if not np.isfinite(estimate).all():
        raise ValueError("production replay contains a missing/nonfinite state")
    error = estimate - truth
    speed = np.hypot(truth[:, 0], truth[:, 1])
    steering = np.asarray([abs(float(sample.actuators[0])) for sample in samples])

    regions: dict[str, np.ndarray] = {
        "all_valid_phase_samples": np.ones(len(samples), dtype=bool),
        "speed_0_to_2_mps": speed < 2.0,
        "speed_2_to_4_mps": (speed >= 2.0) & (speed < 4.0),
        "speed_4_to_6_mps": (speed >= 4.0) & (speed < 6.0),
        "speed_6_to_8_mps": (speed >= 6.0) & (speed < 8.0),
        "speed_8plus_mps": speed >= 8.0,
        "steering_ge_0p30_rad": steering >= 0.30,
        "steering_ge_0p40_rad": steering >= 0.40,
        "steering_ge_0p42_rad": steering >= 0.42,
        "high_speed_and_steering": (speed >= 6.0) & (steering >= 0.30),
        "high_speed_and_steering_ge_0p40": (speed >= 6.0) & (steering >= 0.40),
    }
    region_metrics = {
        name: _axis_metrics(error[mask])
        for name, mask in regions.items() if np.any(mask)
    }

    truth_poses = _read_pose(raw_bag, TRUTH_TOPIC)
    replay_poses = _read_pose(replay_bag, REPLAY_TOPIC)
    pose_stamps = [stamp for stamp in sorted(samples_by_stamp)
                   if stamp in truth_poses and stamp in replay_poses]
    if len(pose_stamps) < 0.98 * len(samples):
        raise ValueError(
            f"relative pose coverage is only {len(pose_stamps)}/{len(samples)}")
    first = pose_stamps[0]
    tx0, ty0, tyaw0 = truth_poses[first]
    ox0, oy0, oyaw0 = replay_poses[first]
    yaw_offset = math.atan2(math.sin(tyaw0 - oyaw0),
                            math.cos(tyaw0 - oyaw0))
    cosine, sine = math.cos(yaw_offset), math.sin(yaw_offset)
    position_errors = []
    heading_errors = []
    for stamp in pose_stamps:
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

    result = {
        "evaluation": "production sensor-odometry node replayed from encoder/IMU only; simulator truth is an offline label",
        "run_id": raw_bag.parents[1].name,
        "raw_bag": str(raw_bag.resolve()),
        "replayed_bag": str(replay_bag.resolve()),
        "production_replay_topic": REPLAY_TOPIC,
        "truth_topic": TRUTH_TOPIC,
        "capture_quality": {
            "aborted": capture.aborted,
            "collisions_start_end": [capture.collision_count_start,
                                     capture.collision_count_end],
            "timing_faults": capture.timing_faults,
            "valid_invalid_unscored_phases": [capture.valid_phase_count,
                                               capture.invalid_phase_count,
                                               capture.unscored_phase_count],
            "scored_scope": "valid phases only; invalid speed-match phases and unscored approach/settle phases excluded",
            "production_state_match_fraction": coverage,
        },
        "unique_source_stamped_samples": len(samples),
        "twist_error_order": ["u_mps", "v_rear_mps", "yaw_rate_radps"],
        "twist_overall": _axis_metrics(error),
        "twist_by_regime": region_metrics,
        "relative_pose": {
            "method": "one initial SE(2) alignment; compare source-stamped relative trajectories",
            "matched_samples": len(pose_stamps),
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
        },
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
    args = parser.parse_args()
    result = score(args.raw_bag, args.replayed_bag, args.output)
    print(json.dumps({"run_id": result["run_id"],
                      "samples": result["unique_source_stamped_samples"],
                      "twist_rmse": result["twist_overall"]["rmse"],
                      "relative_pose": result["relative_pose"]["position_error_m"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
