"""Fit an empirical AMCL latency/error bank from training practice bags."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools import analyze_open_plane_dynamics as analysis


AMCL_TOPIC = "/amcl_pose"
TRUTH_TOPIC = "/autodrive/roboracer_1/odom"
ALIGNMENT_TOLERANCE_NS = 40_000_000


def _yaw(q: Any) -> float:
    return math.atan2(
        2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y)),
        1.0 - 2.0 * (float(q.y) ** 2 + float(q.z) ** 2),
    )


def _pose(topic: str, message: Any) -> tuple[int, np.ndarray]:
    stamp = analysis._stamp_ns(message.header.stamp)
    if topic == AMCL_TOPIC:
        value = message.pose.pose
    else:
        value = message.pose.pose
    return stamp, np.asarray((float(value.position.x),
                              float(value.position.y),
                              _yaw(value.orientation)), dtype=np.float64)


def _read_run(path: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        missing = {AMCL_TOPIC, TRUTH_TOPIC} - set(topics)
        if missing:
            raise ValueError(f"{run_id} missing topics: {sorted(missing)}")
        amcl, truth = {}, {}
        amcl_latency = []
        for receipt_ns, message in analysis._messages(connection, topics, AMCL_TOPIC):
            stamp, pose = _pose(AMCL_TOPIC, message)
            if stamp > 0 and np.isfinite(pose).all():
                amcl[stamp] = pose
                amcl_latency.append((int(receipt_ns) - stamp) * 1.0e-9)
        for _, message in analysis._messages(connection, topics, TRUTH_TOPIC):
            stamp, pose = _pose(TRUTH_TOPIC, message)
            if stamp > 0 and np.isfinite(pose).all():
                truth[stamp] = pose
        return ({"pose": amcl, "latency_s": np.asarray(amcl_latency)},
                {"pose": truth})
    finally:
        connection.close()


def fit(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_records = [record for record in manifest["runs"]
                     if record.get("effective_split") == "train"
                     and str(record.get("run_id", "")).startswith("practice_")
                     and record.get("clean_stream_and_collision_gate")
                     and record.get("bag")]
    repo_root = manifest_path.resolve().parents[3]
    traces = []
    run_report = []
    rejected = []
    for record in train_records:
        run_id = str(record["run_id"])
        path = (repo_root / record["bag"]).resolve()
        if not path.is_file():
            rejected.append({"run_id": run_id, "reason": "bag unavailable"})
            continue
        try:
            localization, truth = _read_run(path, run_id)
            truth_stamps = np.asarray(sorted(truth["pose"]), dtype=np.int64)
            truth_poses = np.stack([truth["pose"][int(stamp)]
                                    for stamp in truth_stamps])
            amcl_stamps = np.asarray(sorted(localization["pose"]), dtype=np.int64)
            indices = np.searchsorted(truth_stamps, amcl_stamps)
            matched_amcl, matched_truth, matched_stamps = [], [], []
            for stamp, index in zip(amcl_stamps, indices):
                candidates = [candidate for candidate in (index - 1, index)
                              if 0 <= candidate < len(truth_stamps)]
                if not candidates:
                    continue
                nearest = min(candidates,
                              key=lambda candidate: abs(
                                  int(truth_stamps[candidate]) - int(stamp)))
                if abs(int(truth_stamps[nearest]) - int(stamp)) > ALIGNMENT_TOLERANCE_NS:
                    continue
                matched_amcl.append(localization["pose"][int(stamp)])
                matched_truth.append(truth_poses[nearest])
                matched_stamps.append(int(stamp))
            if len(matched_stamps) < 200:
                raise ValueError(f"only {len(matched_stamps)} aligned AMCL samples")
            amcl_pose = np.stack(matched_amcl)
            truth_pose = np.stack(matched_truth)
            first_amcl = amcl_pose[0]
            first_truth = truth_pose[0]
            yaw_offset = math.atan2(math.sin(first_truth[2] - first_amcl[2]),
                                    math.cos(first_truth[2] - first_amcl[2]))
            c, s = math.cos(yaw_offset), math.sin(yaw_offset)
            delta = amcl_pose[:, :2] - first_amcl[:2]
            aligned = np.column_stack((
                first_truth[0] + c * delta[:, 0] - s * delta[:, 1],
                first_truth[1] + s * delta[:, 0] + c * delta[:, 1],
                np.unwrap(amcl_pose[:, 2] + yaw_offset),
            ))
            truth_unwrapped = truth_pose.copy()
            truth_unwrapped[:, 2] = np.unwrap(truth_pose[:, 2])
            residual = aligned - truth_unwrapped
            residual[:, 2] = np.arctan2(np.sin(residual[:, 2]),
                                        np.cos(residual[:, 2]))
            time_s = (np.asarray(matched_stamps, dtype=np.float64)
                      - matched_stamps[0]) * 1.0e-9
            if (not np.isfinite(residual).all() or np.any(np.diff(time_s) <= 0)
                    or not np.isfinite(localization["latency_s"]).all()):
                raise ValueError("non-finite or nonmonotonic aligned trace")
            latency = localization["latency_s"]
            # ROS bag receipts can precede source time under simulated clocks;
            # preserve signed observations instead of clipping invented delay.
            traces.append((run_id, time_s, residual,
                           float(np.median(latency))))
            run_report.append({
                "run_id": run_id,
                "matched_samples": int(len(time_s)),
                "duration_s": float(time_s[-1]),
                "amcl_receipt_minus_header_median_s": float(np.median(latency)),
                "amcl_receipt_minus_header_p95_s": float(np.quantile(latency, .95)),
                "relative_position_error_rmse_m": float(np.sqrt(
                    np.mean(np.sum(residual[:, :2] ** 2, axis=1)))),
                "relative_position_error_p95_m": float(np.quantile(
                    np.linalg.norm(residual[:, :2], axis=1), .95)),
                "relative_heading_error_rmse_rad": float(np.sqrt(
                    np.mean(residual[:, 2] ** 2))),
            })
        except Exception as exc:
            rejected.append({"run_id": run_id, "reason": str(exc)})
    if len(traces) < 2:
        raise ValueError(
            f"need >=2 eligible independent train practice traces, got {len(traces)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    time_rows, residual_rows, run_ids, latency_rows = [], [], [], []
    for run_id, times, residual, latency in traces:
        run_ids.append(run_id)
        time_rows.append(times)
        residual_rows.append(residual)
        latency_rows.append(latency)
        offsets.append(offsets[-1] + len(times))
    np.savez_compressed(
        output_dir / "amcl_training_residual_bank.npz",
        run_ids=np.asarray(run_ids, dtype="U128"),
        offsets=np.asarray(offsets, dtype=np.int64),
        time_s=np.concatenate(time_rows),
        residual_xyyaw=np.concatenate(residual_rows).astype(np.float64),
        latency_s=np.asarray(latency_rows, dtype=np.float64),
    )
    errors = np.concatenate(residual_rows)
    run_latency_medians = np.asarray(latency_rows, dtype=np.float64)
    report = {
        "schema_version": 1,
        "calibration": "train-split practice bags only; no validation/test run used",
        "truth_topic": TRUTH_TOPIC,
        "localization_topic": AMCL_TOPIC,
        "pose_error_method": "relative SE(2) trace aligned at first matched source stamp",
        "latency_method": "bag receipt timestamp minus AMCL header source timestamp; signed, not clipped",
        "truth_alignment_tolerance_ms": ALIGNMENT_TOLERANCE_NS / 1e6,
        "independent_training_run_count": len(traces),
        "sample_count": int(sum(len(item[1]) for item in traces)),
        "run_results": run_report,
        "pooled_residual_quantiles_xyyaw": {
            str(q): np.quantile(errors, q, axis=0).tolist()
            for q in (0.05, 0.5, 0.95)
        },
        "run_latency_median_quantiles_s": {
            str(q): float(np.quantile(run_latency_medians, q))
            for q in (0.05, 0.5, 0.95)
        },
        "rejected_or_unavailable_runs": rejected,
        "bank": "amcl_training_residual_bank.npz",
    }
    (output_dir / "amcl_surrogate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = fit(args.dataset_manifest, args.output_dir)
    print(json.dumps({
        "training_runs": result["independent_training_run_count"],
        "samples": result["sample_count"],
        "output": str(args.output_dir.resolve()),
        "rejected": result["rejected_or_unavailable_runs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
