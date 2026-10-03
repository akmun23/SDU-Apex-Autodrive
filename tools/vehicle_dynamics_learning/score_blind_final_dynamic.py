#!/usr/bin/env python3
"""Score the frozen WP14 candidate once on its preregistered final run.

Windows are five-second, non-overlapping command-only rollouts contained
within each reset epoch. The whole ROS-bag run is the independent unit; this
single-run score intentionally reports no run-level confidence interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.compare_effective_race_teacher import (
    HORIZONS,
    _edssm_rollouts,
    _per_window_errors,
    _registry_rssm_path,
    _rssm_rollouts,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
    _support_limits,
    _window_regimes,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002"
FREEZE_PATH = TASK_ROOT / "architecture_freeze_wp14_20261002.json"
TRAIN_PATH = TASK_ROOT / "replacement_teacher_dataset_v1/openplane_dynamics.npz"
REGISTRY_PATH = ROOT / "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/replacement_sim_baseline_registry_20261002.json"
RUN_ID = "openplane_dyn_coupled_final_r01_20261002"
CAPTURE_DIR = ROOT / "live_runs" / RUN_ID
QC_MANIFEST = TASK_ROOT / "capture_qc_final_r01/manifest.json"
FINAL_DATA = TASK_ROOT / "capture_qc_final_r01/openplane_dynamics.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _final_windows(data: dict[str, Any]) -> list[dict[str, Any]]:
    windows = []
    for sequence_id, (start_raw, end_raw) in enumerate(data["bounds"]):
        start, end = int(start_raw), int(end_raw)
        first = start + HISTORY_STEPS - 1
        last = end - 201
        for index in range(first, last + 1, 200):
            windows.append({"run": 0, "sequence_id": sequence_id,
                            "start": index, "regimes": []})
    if not windows:
        raise ValueError("final capture has no complete five-second windows")
    return windows


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_window_rmse": float(np.mean(values)),
        "median_window_rmse": float(np.median(values)),
        "p95_window_rmse": float(np.quantile(values, 0.95)),
    }


def _compare_group(candidate: dict[str, Any], baseline: dict[str, Any],
                   indices: np.ndarray) -> dict[str, Any]:
    report = {}
    for horizon, metrics in candidate.items():
        report[horizon] = {}
        for metric, values in metrics.items():
            left = np.asarray(values)[indices]
            right = np.asarray(baseline[horizon][metric])[indices]
            left_mean, right_mean = float(left.mean()), float(right.mean())
            report[horizon][metric] = {
                "candidate": _metric_summary(left),
                "lead_rssm": _metric_summary(right),
                "mean_window_difference_candidate_minus_rssm": left_mean - right_mean,
                "mean_window_difference_percent_of_rssm": (
                    100.0 * (left_mean - right_mean) / max(right_mean, 1e-12)),
                "run_level_confidence_interval": None,
            }
    return report


def score(output_path: Path, device: str) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite blind score: {output_path}")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    qc = json.loads(QC_MANIFEST.read_text(encoding="utf-8"))
    freeze_eval = freeze["blind_evaluation"]
    if (freeze_eval["run_id"] != RUN_ID
            or freeze_eval["profile"] != "race_domain_dynamic_coupled_final"
            or freeze_eval["seed"] != 20261007
            or freeze_eval["training_or_checkpoint_selection_after_capture"]):
        raise ValueError("final capture does not match the frozen WP14 registration")
    experiment_log = (CAPTURE_DIR / "experiment.log").read_text(encoding="utf-8")
    if (f"profile={freeze_eval['profile']}" not in experiment_log
            or f"seed={freeze_eval['seed']}" not in experiment_log
            or "finished: reason=schedule complete, aborted=False, phases=84/84" not in experiment_log):
        raise ValueError("captured profile, seed, or complete schedule differs from freeze")
    rows = qc["runs"]
    if len(rows) != 1 or rows[0]["run_id"] != RUN_ID:
        raise ValueError("QC dataset must contain exactly the registered final run")
    row = rows[0]
    if (row["effective_split"] != "final_test"
            or row["aborted"] or row["collisions"] != [0, 0]
            or row["timing_faults"] != 0 or row["invalid_phases"] != 0
            or row["quality_failures"] or not row["clean_stream_and_collision_gate"]
            or row["packet_sequence_alignment"]["match_fraction"] < 0.999
            or row["reset_metadata"]["epoch_count"] != 9
            or row["valid_phases"] != 66 or row["unscored_phases"] != 18
            or not row["continuous_whole_run_export"]):
        raise ValueError("closed-bag WP14 quality gates did not pass")
    if any(stream["hz"] < 38.0 for stream in row["streams"].values()):
        raise ValueError("one or more required final-run streams fell below 38 Hz")

    checkpoint = (ROOT / freeze["checkpoint"]["path"]).resolve()
    if _sha256(checkpoint) != freeze["checkpoint"]["sha256"]:
        raise ValueError("frozen candidate checkpoint hash changed")
    if _sha256(TRAIN_PATH) != freeze["training_data"]["sha256"]:
        raise ValueError("frozen training dataset hash changed")
    candidate_torch, candidate_model, candidate_meta = _load_model(checkpoint, device)
    if (candidate_meta["dataset_sha256"] != _sha256(TRAIN_PATH)
            or RUN_ID in candidate_meta["training_runs"]):
        raise ValueError("candidate provenance overlaps or differs from frozen training data")

    training_data = _load_dataset(TRAIN_PATH)
    final_data = _load_dataset(FINAL_DATA)
    if (int(training_data["schema_version"]) != 9
            or int(final_data["schema_version"]) != 7
            or final_data["run_ids"].tolist() != [RUN_ID]
            or final_data["splits"].tolist() != ["final_test"]
            or not np.allclose(final_data["dt_s"], DT_S, rtol=0.0, atol=1e-7)):
        raise ValueError("frozen/final dataset schema, split, or timebase is invalid")
    final_state = physical_state_from_dataset(final_data).astype(np.float32)
    train_state = physical_state_from_dataset(training_data).astype(np.float32)
    windows = _final_windows(final_data)
    p90, p95 = _support_limits(training_data, train_state)
    for window in windows:
        window["regimes"] = _window_regimes(
            final_data, final_state, window["start"], p90, p95)

    rssm_checkpoint = _registry_rssm_path(REGISTRY_PATH)
    if np.max(np.hypot(final_state[:, 0], final_state[:, 1])) > 12.0 + 1e-5:
        raise ValueError("final capture exceeds the declared 0–12 m/s evaluation domain")
    baseline_state, baseline_pose = _rssm_rollouts(
        rssm_checkpoint, TRAIN_PATH, final_data, windows, device)
    candidate_state, candidate_pose = _edssm_rollouts(
        candidate_torch, candidate_model, final_data, final_state,
        windows, device)
    truth_state = final_state
    truth_pose = final_data["simulator_pose_xyyaw"]
    if not (np.isfinite(candidate_state).all() and np.isfinite(candidate_pose).all()
            and np.isfinite(baseline_state).all() and np.isfinite(baseline_pose).all()):
        raise FloatingPointError("a frozen plant produced non-finite final rollouts")
    candidate_errors = _per_window_errors(
        candidate_state, candidate_pose,
        np.stack([truth_state[w["start"] + 1:w["start"] + 201] for w in windows]),
        np.stack([truth_pose[w["start"]:w["start"] + 201] for w in windows]),
        "com")
    baseline_errors = _per_window_errors(
        baseline_state, baseline_pose,
        np.stack([truth_state[w["start"] + 1:w["start"] + 201] for w in windows]),
        np.stack([truth_pose[w["start"]:w["start"] + 201] for w in windows]),
        "rear_axle")

    all_indices = np.arange(len(windows))
    regimes = sorted({name for window in windows for name in window["regimes"]})
    regime_reports = {}
    for regime in regimes:
        selected = np.asarray([i for i, window in enumerate(windows)
                               if regime in window["regimes"]], dtype=np.int64)
        regime_reports[regime] = {
            "window_count": int(len(selected)),
            "metrics": _compare_group(candidate_errors, baseline_errors, selected),
        }
    command_feedback = final_data["frames"]
    report = {
        "schema_version": 1,
        "purpose": "single preregistered blind WP14 evaluation; no training or checkpoint selection",
        "registered_profile": freeze_eval["profile"],
        "registered_seed": freeze_eval["seed"],
        "run_id": RUN_ID,
        "capture_quality": {
            "phases_complete": int(row["valid_phases"] + row["unscored_phases"]),
            "scored_phases": int(row["valid_phases"]),
            "unscored_approach_settle_phases": int(row["unscored_phases"]),
            "reset_epochs": int(row["reset_metadata"]["epoch_count"]),
            "samples": int(row["samples_exported"]),
            "collisions": row["collisions"],
            "timing_faults": row["timing_faults"],
            "packet_alignment_fraction": row["packet_sequence_alignment"]["match_fraction"],
            "minimum_required_stream_hz": min(v["hz"] for v in row["streams"].values()),
            "maximum_truth_speed_mps": float(np.max(np.hypot(final_state[:, 0], final_state[:, 1]))),
            "max_abs_steering_feedback_rad": float(np.max(np.abs(command_feedback[:, 3]))),
            "max_throttle_feedback_norm": float(np.max(command_feedback[:, 4])),
        },
        "candidate": {
            "architecture": freeze["architecture"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        },
        "baseline": {
            "lead_rssm_checkpoint": str(rssm_checkpoint),
            "lead_rssm_checkpoint_sha256": _sha256(rssm_checkpoint),
        },
        "provenance": {
            "bag": str(CAPTURE_DIR / "run/run_0.db3"),
            "bag_sha256": _sha256(CAPTURE_DIR / "run/run_0.db3"),
            "final_dataset": str(FINAL_DATA),
            "final_dataset_sha256": _sha256(FINAL_DATA),
            "qc_manifest_sha256": _sha256(QC_MANIFEST),
            "training_dataset_sha256": _sha256(TRAIN_PATH),
            "training_or_final_test_used_for_training": False,
        },
        "evaluation": {
            "independent_whole_run_count": 1,
            "window_count": int(len(windows)),
            "window_policy": "non-overlapping 5 s windows within each of 9 reset epochs; 80-frame causal history; future commands only",
            "future_truth_or_sensor_feedback_used": False,
            "run_level_uncertainty": "not estimable from one independent whole-run capture; no run-level CI reported",
            "metrics": _compare_group(candidate_errors, baseline_errors, all_indices),
            "hard_regimes": regime_reports,
            "window_manifest": [
                {"sequence_id": int(window["sequence_id"]),
                 "start_frame": int(window["start"]),
                 "regimes": window["regimes"]}
                for window in windows
            ],
        },
        "promotion_decision": "not decided by this single dynamic run; prior practice no-regression gate remains failed",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available()
                        else "cpu")
    args = parser.parse_args()
    result = score(args.output.resolve(), args.device)
    print(json.dumps({"run_id": result["run_id"],
                      "windows": result["evaluation"]["window_count"],
                      "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
