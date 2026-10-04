#!/usr/bin/env python3
"""Score the best existing wheel/actuator teacher on the newer high-steer holdout.

Uses the exact 5 s start windows used by the roll-coupled plant transfer
evaluation. The EDSSM is initialized from measured current state/history and
then receives commands only; future simulator truth and sensors are scoring
labels, never rollout inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    append_roll_state,
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.evaluate_history_context_highsteer import (
    _capture_from_archive,
    _sample_all_valid_context_refs,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
    evaluate_model,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    sha256_file,
)


CHECKPOINT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002"
    / "encoder_raw_state_teacher_v1/edssm_gru_z32_e2_rollresidual_10s_yawonly_"
    "longitudinal_seed101/last.pt")
DATASET = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/highsteer_75_heldout_validation_plus_r03_20261004"
    / "openplane_dynamics.npz")
OUTPUT = (ROOT / "live_runs/derived_dynamics_learning_20260928"
    / "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003"
    / "full_throttle_domain_v1/next_phase_after_2129427/history_context_sufficiency_v1"
    / "roll_coupled_multistep_candidate_v2_20261004"
    / "effective_teacher_highsteer_transfer_20261004.json")
RUNS = {
    "openplane_highsteer_75_validation_r01",
    "openplane_highsteer_75_validation_r02",
    "openplane_highsteer_75_validation_r03_20261004",
}
HORIZON_STEPS = 200
MAX_STARTS_PER_RUN = 16
DT_S = 0.025


def _archive_data(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    data["bounds"] = data["sequence_bounds"]
    data["seq_run"] = data["sequence_run_index"]
    data["splits"] = data["run_splits"]
    return data


def _start_features(capture: Any, start: int) -> dict[str, float]:
    frames = capture.frames
    previous = start - 1
    speed = float(np.hypot(*capture.body[start, :2]))
    return {
        "speed_mps": speed,
        "absolute_steering_rad": abs(float(frames[start, 3])),
        "absolute_steering_rate_radps": abs(float(
            (frames[start, 3] - frames[previous, 3]) / DT_S)),
        "throttle_command_norm": float(frames[start, 8]),
        "absolute_throttle_slew_per_s": abs(float(
            (frames[start, 8] - frames[previous, 8]) / DT_S)),
        "absolute_wheel_body_mismatch_mps": abs(float(
            0.5 * (frames[start, 5] + frames[start, 6])
            - capture.body[start, 0])),
    }


def _matched_windows(path: Path) -> tuple[
        dict[str, Any], list[dict[str, Any]], dict[str, Any],
        dict[str, Any], dict[str, Any]]:
    capture, _ = _capture_from_archive(path, RUNS)
    candidates = _sample_all_valid_context_refs(
        capture, HORIZON_STEPS, history_steps=80, expected_runs=RUNS)
    data = _archive_data(path)
    run_ids = data["run_ids"].astype(str)
    windows = []
    manifest = {}
    thresholds_by_run = {}
    feature_spreads_by_run = {}
    feature_names = (
        "speed_mps", "absolute_steering_rad",
        "absolute_steering_rate_radps", "throttle_command_norm",
        "absolute_throttle_slew_per_s",
        "absolute_wheel_body_mismatch_mps")
    minimum_spread = {
        "speed_mps": 0.10,
        "absolute_steering_rad": 0.02,
        "absolute_steering_rate_radps": 0.10,
        "throttle_command_norm": 0.01,
        "absolute_throttle_slew_per_s": 0.05,
        "absolute_wheel_body_mismatch_mps": 0.05,
    }
    for run_id, refs in sorted(candidates.items()):
        retained = []
        for _, sequence, row in refs:
            begin = int(capture.bounds[sequence, 0])
            start = begin + int(row)
            speed = float(np.hypot(*capture.body[start, :2]))
            steering = abs(float(capture.frames[start, 3]))
            if 7.0 <= speed <= 9.0 and steering >= 0.30:
                retained.append((sequence, start, speed, steering))
        if len(retained) > MAX_STARTS_PER_RUN:
            positions = np.linspace(0, len(retained) - 1,
                                    MAX_STARTS_PER_RUN, dtype=np.int64)
            retained = [retained[int(position)] for position in positions]
        if not retained:
            raise ValueError(f"no matched high-steer starts for {run_id}")
        run_index = int(np.flatnonzero(run_ids == run_id)[0])
        manifest[run_id] = []
        feature_rows = [
            {"sequence": sequence, "start": start,
             **_start_features(capture, start)}
            for sequence, start, _, _ in retained]
        thresholds = {
            name: float(np.median([row[name] for row in feature_rows]))
            for name in feature_names}
        spreads = {
            name: float(max(row[name] for row in feature_rows)
                        - min(row[name] for row in feature_rows))
            for name in feature_names}
        thresholds_by_run[run_id] = thresholds
        feature_spreads_by_run[run_id] = spreads
        for row in feature_rows:
            regimes = ["high_steering"]
            for name, threshold in thresholds.items():
                if spreads[name] < minimum_spread[name]:
                    side = "near_constant"
                else:
                    side = "upper_half" if row[name] > threshold else "lower_half"
                regimes.append(f"{name}_{side}")
            sequence = int(row["sequence"])
            start = int(row["start"])
            windows.append({
                "sequence_id": int(sequence),
                "start": int(start),
                "run": run_index,
                "regimes": regimes,
            })
            manifest[run_id].append({name: (int(value)
                if name in ("sequence", "start") else value)
                for name, value in row.items()})
    if set(manifest) != RUNS:
        raise ValueError(f"high-steer run roster mismatch: {sorted(manifest)}")
    return data, windows, thresholds_by_run, feature_spreads_by_run, manifest


def run(checkpoint_path: Path = CHECKPOINT,
        output_path: Path = OUTPUT) -> dict[str, Any]:
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    (data, windows, thresholds_by_run, feature_spreads_by_run,
     feature_manifest) = _matched_windows(DATASET)
    torch, model, metadata = _load_model(checkpoint_path, "cpu")
    training_runs = set(metadata["training_runs"])
    if training_runs & RUNS:
        raise ValueError("high-steer validation runs overlap teacher training")
    state = physical_state_from_dataset(
        data, str(metadata.get("wheel_state_source", "filtered_odometry")))
    if metadata.get("include_roll_state", False):
        state = append_roll_state(data, state)
    evaluation_by_offset = {}
    for offset in (-1, 0):
        evaluation_by_offset[str(offset)] = evaluate_model(
            model, data, state,
            np.asarray(metadata["state_scale"], dtype=np.float64), "cpu",
            windows=windows,
            wheel_state_source=str(metadata.get(
                "wheel_state_source", "filtered_odometry")),
            horizon_steps={"0.75s": 30, "2s": 80, "5s": HORIZON_STEPS},
            command_offset_frames=offset)
    report = {
        "study": "existing recurrent wheel/actuator plant on three-run high-steer holdout",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dataset": str(DATASET.relative_to(ROOT)),
        "dataset_sha256": sha256_file(DATASET),
        "training_run_ids": sorted(training_runs),
        "heldout_run_ids": sorted(RUNS),
        "windows_per_run": {
            run: sum(int(row["run"]) == index
                     for row in windows)
            for index, run in enumerate(data["run_ids"].astype(str))
            if run in RUNS},
        "start_feature_median_thresholds_by_run": thresholds_by_run,
        "start_feature_spreads_by_run": feature_spreads_by_run,
        "exact_start_and_feature_manifest": feature_manifest,
        "region_policy": "start speed 7-9 m/s and absolute steering >= 0.30 rad; max 16 evenly spaced windows/run",
        "horizon_steps": HORIZON_STEPS,
        "future_truth_or_sensor_feedback_used": False,
        "future_inputs": ["recorded steering and throttle commands only"],
        "command_offset_definition": (
            "offset -1 uses the command stored with the preceding state row, "
            "matching the roll-coupled candidate's command indexing; offset 0 "
            "uses the command stored with the target-state row"),
        "metrics_by_command_offset": evaluation_by_offset,
        "production_integration": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run(args.checkpoint, args.output)
    print(json.dumps({
        "report": str(args.output.resolve()),
        "window_count": result["metrics_by_command_offset"]["-1"][
            "window_count"],
        "high_steering_5s_by_command_offset": {
            offset: metrics["hard_regimes"]["high_steering"][
                "horizons"]["5s"]
            for offset, metrics in result["metrics_by_command_offset"].items()},
    }, indent=2))
