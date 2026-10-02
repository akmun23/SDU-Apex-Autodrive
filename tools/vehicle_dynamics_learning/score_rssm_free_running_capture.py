#!/usr/bin/env python3
"""Measure command-only RSSM plant drift over held-out captures.

The only truth initialization is a 2 s history and its matching initial pose.
After that, the production offline-plant adapter receives commands only;
recorded future state and pose are used exclusively for scoring. The default
mode scores the established high-steering phase protocol; continuous-run mode
scores every uninterrupted segment in a whole-run validation capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from tools.race_domain_experiment_plan import HIGH_STEER_VALIDATION_ANGLES_RAD
from tools.vehicle_dynamics_learning.offline_plant import (
    BODY_STATE_NAMES,
    load_rssm_teacher_plant,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


DT_S = 0.025
REPO_ROOT = Path(__file__).resolve().parents[2]
HORIZONS_S = (0.25, 0.75, 2.0, 5.0)
EXPECTED_PHASES = {
    "r1_baseline_zero_steer",
    *(f"r1_steer_{sign:+.0f}_{angle:.4f}"
      for angle in HIGH_STEER_VALIDATION_ANGLES_RAD
      for sign in (-1.0, 1.0)),
}


def _wrap_angle(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _errors(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64)
    return {
        "rmse": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "absolute_p95": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
    }


def _merge_contiguous_sequences(data: dict[str, Any], run_index: int
                                ) -> list[tuple[int, int, str]]:
    """Rejoin same-phase fragments split only by lap-counter changes."""
    indexes = [index for index, value in enumerate(data["seq_run"])
               if int(value) == run_index]
    indexes.sort(key=lambda index: int(data["bounds"][index, 0]))
    groups: list[tuple[int, int, str]] = []
    for sequence_index in indexes:
        start, end = map(int, data["bounds"][sequence_index])
        label_index = int(data["sequence_condition_id"][sequence_index])
        label = str(data["condition_labels"][label_index])
        if groups:
            previous_start, previous_end, previous_label = groups[-1]
            packet_contiguous = (
                previous_end == start
                and int(data["packet_sequence"][start])
                    == int(data["packet_sequence"][previous_end - 1]) + 1)
            if label == previous_label and packet_contiguous:
                groups[-1] = (previous_start, end, label)
                continue
        groups.append((start, end, label))
    return groups


def _group_expected_phase_fragments(
        groups: list[tuple[int, int, str]]) -> dict[str, list[tuple[int, int]]]:
    """Require each scheduled probe while retaining real packet-gap splits."""
    fragments: dict[str, list[tuple[int, int]]] = {}
    for start, end, label in groups:
        if label in EXPECTED_PHASES:
            fragments.setdefault(label, []).append((start, end))
    missing = sorted(EXPECTED_PHASES - fragments.keys())
    unexpected = sorted(set(fragments) - EXPECTED_PHASES)
    if missing or unexpected:
        raise ValueError(
            f"probe schedule mismatch; missing={missing}, unexpected={unexpected}")
    return fragments


def _scoring_fragments(
        groups: list[tuple[int, int, str]], continuous_run: bool
        ) -> list[tuple[str, int, int, int]]:
    """Keep packet gaps explicit while supporting an unlabelled full run."""
    if continuous_run:
        if not groups:
            raise ValueError("held-out run has no continuous segments")
        return [(label, index, start, end)
                for index, (start, end, label) in enumerate(groups)]
    fragments = _group_expected_phase_fragments(groups)
    return [(label, fragment_index, start, end)
            for label in sorted(fragments)
            for fragment_index, (start, end) in enumerate(fragments[label])]


def _phase_onset_windows(data: dict[str, Any], run_index: int,
                         bag_path: Path, history_steps: int,
                         future_steps: int) -> list[tuple[str, int, int, int]]:
    """Build causal context→command windows at valid source phase onsets."""
    from tools import analyze_open_plane_dynamics as analysis

    sample_times = data.get("sample_time_ns")
    if sample_times is None:
        raise ValueError("phase-onset scoring requires sample_time_ns labels")
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    connection = sqlite3.connect(bag_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        topics = analysis._topic_map(connection)
        if analysis.PHASE not in topics:
            raise ValueError(f"phase-onset bag lacks {analysis.PHASE}")
        phases, _ = analysis._phase_events(connection, topics)
    finally:
        connection.close()

    segments = _merge_contiguous_sequences(data, run_index)
    windows = []
    for phase in phases:
        if phase.valid is not True:
            continue
        located = None
        for segment_start, segment_end, _ in segments:
            local_times = sample_times[segment_start:segment_end]
            local_index = int(np.searchsorted(
                local_times, phase.start_ns, side="left"))
            if local_index >= len(local_times):
                continue
            candidate = segment_start + local_index
            if abs(int(sample_times[candidate]) - int(phase.start_ns)) <= 50_000_000:
                located = (candidate, segment_start, segment_end)
                break
        if located is None:
            raise ValueError(
                f"{phase.label}: phase onset has no nearby aligned sample")
        index, segment_start, segment_end = located
        if index >= len(sample_times):
            continue
        if abs(int(sample_times[index]) - int(phase.start_ns)) > 50_000_000:
            raise ValueError(
                f"{phase.label}: phase onset has no nearby aligned sample")
        start = index - history_steps
        if start < segment_start:
            continue
        end = min(segment_end, index + future_steps)
        if end - index < round(min(HORIZONS_S) / DT_S):
            continue
        windows.append((phase.label, index, start, end))
    if not windows:
        raise ValueError("no valid phase onsets have sufficient causal history")
    return windows


def _pose_errors(prediction: np.ndarray, truth: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(
        truth, dtype=np.float64)
    error[:, 2] = _wrap_angle(error[:, 2])
    return error[:, :2], error[:, 2]


def _time_bin_summary(state_error: np.ndarray, position_error: np.ndarray,
                      heading_error: np.ndarray, speed_error: np.ndarray,
                      bin_width_s: float = 2.0) -> dict[str, Any]:
    """Show where recursive error accumulates within a continuous replay."""
    state = np.asarray(state_error, dtype=np.float64)
    position = np.asarray(position_error, dtype=np.float64)
    heading = np.asarray(heading_error, dtype=np.float64)
    speed = np.asarray(speed_error, dtype=np.float64)
    count = len(state)
    bin_steps = int(round(bin_width_s / DT_S))
    if (count == 0 or state.ndim != 2 or state.shape[1] != len(BODY_STATE_NAMES)
            or position.shape != (count, 2) or heading.shape != (count,)
            or speed.shape != (count,) or bin_steps < 1
            or not np.isfinite(state).all()
            or not np.isfinite(position).all()
            or not np.isfinite(heading).all()
            or not np.isfinite(speed).all()):
        raise ValueError("invalid continuous-run error arrays or bin width")

    bins = []
    for start in range(0, count, bin_steps):
        end = min(count, start + bin_steps)
        sl = slice(start, end)
        radial = np.linalg.norm(position[sl], axis=1)
        bins.append({
            "elapsed_start_s": start * DT_S,
            "elapsed_end_s": end * DT_S,
            "sample_count": end - start,
            "body_state_rmse": _errors(state[sl], np.zeros_like(state[sl]))[
                "rmse"],
            "body_state_bias": np.mean(state[sl], axis=0).tolist(),
            "speed_rmse_mps": float(np.sqrt(np.mean(speed[sl] ** 2))),
            "position_radial_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
            "position_radial_p95_m": float(np.quantile(radial, 0.95)),
            "position_endpoint_error_m": float(radial[-1]),
            "heading_rmse_rad": float(np.sqrt(np.mean(heading[sl] ** 2))),
            "heading_endpoint_abs_error_rad": float(abs(heading[end - 1])),
        })

    peak_indices = np.argmax(np.abs(state), axis=0)
    return {
        "bin_width_s": bin_width_s,
        "bins": bins,
        "peak_absolute_error_by_state": {
            name: {
                "error": float(abs(state[index, channel])),
                "elapsed_s": float((index + 1) * DT_S),
            }
            for channel, (name, index) in enumerate(
                zip(BODY_STATE_NAMES, peak_indices))
        },
    }


def _phase_score(plant, data: dict[str, Any], start: int, end: int,
                 label: str, fragment_index: int) -> dict[str, Any]:
    history_steps = plant.history_steps
    minimum_future_steps = round(min(HORIZONS_S) / DT_S)
    if end - start < history_steps + minimum_future_steps:
        raise ValueError(
            f"{label} fragment {fragment_index}: too short for history and "
            f"the {min(HORIZONS_S):g} s minimum horizon")
    frames = data["frames"][start:end]
    dt = data["dt_s"][start:end]
    pose_truth = data["simulator_pose_xyyaw"][start:end]
    if (not np.isfinite(frames).all() or not np.isfinite(pose_truth).all()
            or not np.allclose(dt, DT_S, rtol=0.0, atol=1e-7)):
        raise ValueError(f"{label}: invalid state/pose labels or non-25 ms timebase")

    context_end = history_steps - 1
    initial = plant.reset(frames[:history_steps], pose_truth[context_end])
    predicted_state = [initial.state[3:].astype(np.float64)]
    predicted_pose = [initial.state[:3].astype(np.float64)]
    domain_flags = [bool(initial.within_speed_domain)]
    for frame_index in range(history_steps, len(frames)):
        command = frames[frame_index, 7:9]
        estimate = plant.step(float(command[0]), float(command[1]),
                              float(dt[frame_index]))
        if not np.isfinite(estimate.state).all():
            raise FloatingPointError(f"{label}: non-finite state at step {frame_index}")
        predicted_state.append(estimate.state[3:].astype(np.float64))
        predicted_pose.append(estimate.state[:3].astype(np.float64))
        domain_flags.append(bool(estimate.within_speed_domain))

    predicted_state_array = np.stack(predicted_state[1:])
    truth_state = frames[history_steps:, :7].astype(np.float64)
    predicted_pose_array = np.stack(predicted_pose[1:])
    truth_pose = pose_truth[history_steps:].astype(np.float64)
    position_error, heading_error = _pose_errors(predicted_pose_array, truth_pose)
    state_error = predicted_state_array - truth_state
    speed_prediction = np.linalg.norm(predicted_state_array[:, :2], axis=1)
    speed_truth = np.linalg.norm(truth_state[:, :2], axis=1)
    speed_error = speed_prediction - speed_truth

    horizon_scores = {}
    for horizon_s in HORIZONS_S:
        index = round(horizon_s / DT_S) - 1
        if index >= len(state_error):
            continue
        horizon_scores[f"{horizon_s:g}s"] = {
            "body_state_abs_error": np.abs(state_error[index]).tolist(),
            "position_error_m": float(np.linalg.norm(position_error[index])),
            "heading_abs_error_rad": float(abs(heading_error[index])),
            "speed_abs_error_mps": float(abs(speed_error[index])),
        }

    command_steering = frames[history_steps:, 7]
    actual_steering = frames[history_steps:, 3]
    actual_throttle = frames[history_steps:, 4]
    return {
        "phase_label": label,
        "fragment_index": fragment_index,
        "initial_history_seconds": (history_steps - 1) * DT_S,
        "free_run_seconds": len(state_error) * DT_S,
        "free_run_steps": len(state_error),
        "command_steering_median_rad": float(np.median(command_steering)),
        "actual_steering_median_rad": float(np.median(actual_steering)),
        "actual_speed_p05_p50_p95_mps": np.quantile(
            speed_truth, (0.05, 0.50, 0.95)).tolist(),
        "actual_throttle_p05_p50_p95": np.quantile(
            actual_throttle, (0.05, 0.50, 0.95)).tolist(),
        "whole_free_run": {
            "body_state_names": list(BODY_STATE_NAMES),
            "body_state": _errors(predicted_state_array, truth_state),
            "speed_rmse_mps": float(np.sqrt(np.mean(speed_error ** 2))),
            "position_xy_rmse_m": np.sqrt(
                np.mean(position_error ** 2, axis=0)).tolist(),
            "position_radial_rmse_m": float(np.sqrt(
                np.mean(np.sum(position_error ** 2, axis=1)))),
            "position_radial_p95_m": float(np.quantile(
                np.linalg.norm(position_error, axis=1), 0.95)),
            "position_endpoint_m": float(np.linalg.norm(position_error[-1])),
            "heading_rmse_rad": float(np.sqrt(np.mean(heading_error ** 2))),
            "heading_endpoint_abs_rad": float(abs(heading_error[-1])),
        },
        "error_growth_over_time": _time_bin_summary(
            state_error, position_error, heading_error, speed_error),
        "horizon_scores": horizon_scores,
        "predicted_speed_exceeded_12mps": not all(domain_flags),
    }


def score(checkpoint_path: Path, training_dataset_path: Path,
          validation_dataset_path: Path, run_ids: list[str],
          output_path: Path, quality_manifest_path: Path,
          device_name: str = "cpu",
          continuous_run: bool = False,
          phase_onsets: bool = False) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("provide unique held-out run IDs")
    torch, _ = _torch()
    if device_name == "cpu":
        torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    metadata = checkpoint["metadata"]
    training_runs = set(map(str, metadata["training_runs"]))
    overlap = sorted(training_runs.intersection(run_ids))
    if overlap:
        raise ValueError(f"refusing to score training runs: {overlap}")

    data = _load_dataset(validation_dataset_path)
    if data["feature_names"] != metadata["feature_names"]:
        raise ValueError("checkpoint and validation feature layouts differ")
    if (data["simulator_pose_xyyaw"] is None
            or data["packet_sequence"] is None
            or not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7)):
        raise ValueError("held-out score requires same-packet pose and exact 25 ms data")
    minimum_schema = 7 if continuous_run else 8
    if data["schema_version"] < minimum_schema:
        raise ValueError(
            f"held-out score requires schema {minimum_schema} or newer")
    run_id_to_index = {str(value): index
                       for index, value in enumerate(data["run_ids"])}
    missing = sorted(set(run_ids) - set(run_id_to_index))
    if missing:
        raise ValueError(f"validation runs are missing: {missing}")
    if any(data["splits"][run_id_to_index[run_id]] != "validation"
           for run_id in run_ids):
        raise ValueError("every selected run must be assigned validation-only")

    manifest_path = quality_manifest_path.resolve()
    if not manifest_path.is_file():
        raise ValueError(f"quality manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_runs = {str(row["run_id"]): row
                     for row in manifest.get("runs", [])}
    for run_id in run_ids:
        row = manifest_runs.get(run_id)
        if (row is None or row.get("aborted")
                or not row.get("clean_stream_and_collision_gate")
                or row.get("quality_failures")
                or any(int(value) != 0 for value in row.get("collisions", []))):
            raise ValueError(f"held-out run failed whole-run quality gate: {run_id}")

    plant = load_rssm_teacher_plant(
        [checkpoint_path], training_dataset_path, device=device_name,
        max_supported_speed_mps=12.0)
    results = []
    for run_id in run_ids:
        run_index = run_id_to_index[run_id]
        groups = _merge_contiguous_sequences(data, run_index)
        phase_scores = []
        scoring_fragments = _scoring_fragments(groups, continuous_run)
        for label, fragment_index, start, end in scoring_fragments:
            phase_scores.append(_phase_score(
                plant, data, start, end, label, fragment_index))
        run_result = {
            "run_id": run_id,
            "probe_count": (None if continuous_run else
                            len({label for label, _, _, _ in scoring_fragments})),
            "continuous_segment_count": (len(phase_scores)
                                         if continuous_run else None),
            "contiguous_fragment_count": len(phase_scores),
            "segments" if continuous_run else "phases": phase_scores,
        }
        if phase_onsets:
            quality_row = manifest_runs[run_id]
            bag_value = Path(str(quality_row["bag"]))
            bag_path = bag_value if bag_value.is_absolute() else REPO_ROOT / bag_value
            onset_windows = _phase_onset_windows(
                data, run_index, bag_path, plant.history_steps,
                round(max(HORIZONS_S) / DT_S))
            onset_scores = []
            for onset_index, (label, sample_index, start, end) in enumerate(
                    onset_windows):
                score_row = _phase_score(
                    plant, data, start, end, label, onset_index)
                score_row["phase_start_sample_index"] = sample_index
                onset_scores.append(score_row)
            run_result["phase_onset_window_count"] = len(onset_scores)
            run_result["phase_onset_windows"] = onset_scores
        results.append(run_result)

    report = {
        "schema_version": 1,
        "purpose": ("recursive command-only free-run accuracy on complete held-out runs"
                    if continuous_run else
                    "recursive command-only free-run accuracy on fresh high-steering whole-run validation captures"),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()).hexdigest(),
        "training_dataset": str(training_dataset_path.resolve()),
        "training_dataset_sha256": hashlib.sha256(
            training_dataset_path.read_bytes()).hexdigest(),
        "validation_dataset": str(validation_dataset_path.resolve()),
        "validation_dataset_sha256": hashlib.sha256(
            validation_dataset_path.read_bytes()).hexdigest(),
        "quality_manifest": str(manifest_path),
        "quality_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()).hexdigest(),
        "independent_validation_run_count": len(results),
        "validation_run_ids": run_ids,
        "training_validation_run_overlap": [],
        "context_seconds": (int(metadata["context_steps"]) - 1) * DT_S,
        "future_inputs_after_context": ["steering_command_rad",
                                        "throttle_command_norm"],
        "future_truth_or_measured_feedback_used": False,
        "pose_anchor": "single simulator pose at end of initial history; no later corrections",
        "primary_statistical_unit": "whole run, not phase or overlapping window",
        "continuous_run_mode": continuous_run,
        "phase_onset_mode": phase_onsets,
        "metrics": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("training_dataset", type=Path,
                        help="race-domain dataset used to build train-only support")
    parser.add_argument("validation_dataset", type=Path)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-manifest", type=Path, required=True,
                        help="source manifest containing per-run whole-bag gates")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--continuous-run", action="store_true",
                        help="score each complete packet-contiguous run segment without requiring named high-steering probe phases")
    parser.add_argument("--phase-onsets", action="store_true",
                        help="also score a 5 s command-only rollout from the past history before each valid phase onset")
    args = parser.parse_args()
    report = score(args.checkpoint, args.training_dataset,
                   args.validation_dataset, args.run_id, args.output,
                   args.quality_manifest, args.device,
                   args.continuous_run, args.phase_onsets)
    print(json.dumps({
        "runs": report["independent_validation_run_count"],
        "probes_per_run": [row["probe_count"] for row in report["metrics"]],
        "segments_per_run": [row["continuous_segment_count"]
                              for row in report["metrics"]],
        "phase_onset_windows_per_run": [row.get("phase_onset_window_count")
                                        for row in report["metrics"]],
        "fragments_per_run": [row["contiguous_fragment_count"]
                              for row in report["metrics"]],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
