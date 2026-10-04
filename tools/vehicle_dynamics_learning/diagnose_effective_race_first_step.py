#!/usr/bin/env python3
"""Measure held-out one-transition EDSSM errors by operating condition.

Each scored transition starts from its own causal 80-frame measured history
and current measured state. The model predicts only the next 25 ms sample;
future truth is used only to score and stratify the residual. This isolates
local transition error from recursive drift. Whole-run metrics are retained so
frame counts are not mistaken for independent evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    HISTORY_STEPS,
    REAR_AXLE_TO_COM_M,
    append_roll_state,
    integrate_pose,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/" \
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002"
DEFAULT_DYNAMIC_DATASET = TASK_ROOT / "encoder_raw_state_teacher_v1/" \
    "openplane_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE_DATASET = TASK_ROOT / "encoder_raw_state_teacher_v1/" \
    "practice_dynamics_raw_wheels.npz"
DEFAULT_CHECKPOINT = TASK_ROOT / "encoder_raw_state_teacher_v1/" \
    "edssm_gru_z32_e2_rollresidual_10s_seed101/best.pt"
DEFAULT_OUTPUT = TASK_ROOT / "first_transition_attribution_v1.json"

GROUPS = {
    "speed_mps": (np.asarray((0.0, 3.0, 5.0, 7.0, 9.0, 12.0001)),
                  ("0-3", "3-5", "5-7", "7-9", "9-12")),
    "signed_steering_rad": (
        np.asarray((-0.524, -0.30, -0.10, 0.0, 0.10, 0.30, 0.5241)),
        ("-.524..-.30", "-.30..-.10", "-.10..0", "0.. .10",
         ".10.. .30", ".30.. .524")),
    "abs_steering_rad": (
        np.asarray((0.0, 0.10, 0.20, 0.30, 0.40, 0.52361)),
        ("0-.10", ".10-.20", ".20-.30", ".30-.40", ".40-.524")),
    "abs_steering_rate_rps": (
        np.asarray((0.0, 0.5, 2.0, 5.0, 10.0, 40.0001)),
        ("0-.5", ".5-2", "2-5", "5-10", "10-40")),
    "throttle_command": (
        np.asarray((0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0001)),
        ("0-.05", ".05-.10", ".10-.20", ".20-.35", ".35-.50", ".50-1")),
    "abs_throttle_slew_per_s": (
        np.asarray((0.0, 0.25, 1.0, 3.0, 8.0, 22.0, 40.0001)),
        ("0-.25", ".25-1", "1-3", "3-8", "8-22", "22-40")),
    "abs_wheel_body_mismatch_mps": (
        np.asarray((0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0, 20.0001)),
        ("0-.10", ".10-.25", ".25-.50", ".50-1", "1-2", "2-4", "4-20")),
    "signed_wheel_body_mismatch_mps": (
        np.asarray((-20.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 20.0001)),
        ("-20..-2", "-2..-1", "-1..-.5", "-.5..0", "0.. .5",
         ".5..1", "1..2", "2..20")),
    "signed_yaw_rate_rps": (
        np.asarray((-40.0, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 40.0)),
        ("-40..-2", "-2..-1", "-1..-.25", "-.25..0", "0.. .25",
         ".25..1", "1..2", "2..40")),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _transition_indices(data: dict[str, Any], allowed_splits: set[str]
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return valid source rows and their whole-run IDs, never crossing bounds."""
    splits = np.asarray(data["splits"]).astype(str)
    selected: list[np.ndarray] = []
    run_indices: list[np.ndarray] = []
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] not in allowed_splits:
            continue
        first = start + HISTORY_STEPS - 1
        last = end - 2
        if last < first:
            continue
        rows = np.arange(first, last + 1, dtype=np.int64)
        selected.append(rows)
        run_indices.append(np.full(len(rows), run, dtype=np.int32))
    if not selected:
        raise ValueError(f"no eligible one-step transitions for {allowed_splits}")
    return np.concatenate(selected), np.concatenate(run_indices)


def _command_rows(source_rows: np.ndarray,
                  command_offset_frames: int) -> np.ndarray:
    """Map a source-state transition to its tested command-row convention."""
    if command_offset_frames not in (-1, 0):
        raise ValueError("command offset must be -1 or 0 frames")
    return np.asarray(source_rows, dtype=np.int64) + 1 + command_offset_frames


def _summarize(values: np.ndarray, run_index: np.ndarray,
               run_ids: np.ndarray, selected: np.ndarray,
               valid: np.ndarray | None = None) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    if valid is not None:
        selected &= np.asarray(valid, dtype=bool)
    per_run_rmse: dict[str, float] = {}
    per_run_bias: dict[str, float] = {}
    for run in sorted(set(run_index[selected].tolist())):
        residual = values[selected & (run_index == run)]
        if residual.size:
            per_run_rmse[str(run_ids[run])] = float(
                np.sqrt(np.mean(residual ** 2)))
            per_run_bias[str(run_ids[run])] = float(np.mean(residual))
    return {
        "transition_count": int(np.count_nonzero(selected)),
        "independent_whole_runs": len(per_run_rmse),
        "macro_run_rmse": (float(np.mean(list(per_run_rmse.values())))
                            if per_run_rmse else None),
        "macro_run_bias_prediction_minus_truth": (
            float(np.mean(list(per_run_bias.values()))) if per_run_bias else None),
        "per_run_rmse": per_run_rmse,
        "per_run_bias_prediction_minus_truth": per_run_bias,
    }


def _covariates(data: dict[str, Any], state: np.ndarray,
                rows: np.ndarray,
                command_offset_frames: int) -> dict[str, np.ndarray]:
    frames = np.asarray(data["frames"], dtype=np.float64)
    steering_rate = (state[rows, 3] - state[rows - 1, 3]) / DT_S
    command_rows = _command_rows(rows, command_offset_frames)
    command_throttle = frames[command_rows, 8]
    throttle_slew = (
        frames[command_rows, 8] - frames[command_rows - 1, 8]) / DT_S
    signed_mismatch = 0.5 * (state[rows, 5] + state[rows, 6]) - state[rows, 0]
    return {
        "speed_mps": np.hypot(state[rows, 0], state[rows, 1]),
        "signed_steering_rad": state[rows, 3],
        "abs_steering_rad": np.abs(state[rows, 3]),
        "abs_steering_rate_rps": np.abs(steering_rate),
        "signed_yaw_rate_rps": state[rows, 2],
        "throttle_command": command_throttle,
        "abs_throttle_slew_per_s": np.abs(throttle_slew),
        "abs_wheel_body_mismatch_mps": np.abs(signed_mismatch),
        "signed_wheel_body_mismatch_mps": signed_mismatch,
    }


def _wheel_measurement_semantics(data: dict[str, Any], rows: np.ndarray,
                                 run_index: np.ndarray,
                                 run_ids: np.ndarray, predicted: np.ndarray,
                                 filtered_target: np.ndarray,
                                 input_wheel_source: str) -> dict[str, Any] | None:
    """Compare the same prediction with filtered and raw encoder targets."""
    raw = data.get("encoder_raw_surface_mps")
    raw_valid = data.get("encoder_raw_valid")
    if raw is None or raw_valid is None:
        return None
    raw = np.asarray(raw, dtype=np.float64)
    valid = np.asarray(raw_valid, dtype=bool)[rows + 1]
    raw_target = raw[rows + 1]
    filtered_error = predicted[:, 5:7] - filtered_target[:, 5:7]
    raw_error = predicted[:, 5:7] - raw_target
    target_delta = raw_target - filtered_target[:, 5:7]
    result: dict[str, Any] = {
        "input_wheel_state_source": input_wheel_source,
        "valid_packet_aligned_raw_target_transitions": int(valid.sum()),
        "raw_minus_filtered_target_definition": (
            "same next packet: 25 ms source-angle rate minus existing 100 ms filtered odometry estimate"),
        "metrics": {},
    }
    for index, side in enumerate(("left", "right")):
        side_report = {
            "prediction_minus_filtered_target": _summarize(
                filtered_error[:, index], run_index, run_ids, valid),
            "prediction_minus_raw_target": _summarize(
                raw_error[:, index], run_index, run_ids, valid),
            "raw_minus_filtered_target": _summarize(
                target_delta[:, index], run_index, run_ids, valid),
            "per_run_correlation_filtered_error_with_target_delta": {},
        }
        for run in sorted(set(run_index[valid].tolist())):
            selected = valid & (run_index == run)
            x = filtered_error[selected, index]
            y = target_delta[selected, index]
            correlation = (float(np.corrcoef(x, y)[0, 1])
                           if len(x) > 1 and np.std(x) > 0.0
                           and np.std(y) > 0.0 else None)
            side_report["per_run_correlation_filtered_error_with_target_delta"][
                str(run_ids[run])] = correlation
        correlations = [value for value in side_report[
            "per_run_correlation_filtered_error_with_target_delta"].values()
                        if value is not None]
        side_report["macro_run_correlation_filtered_error_with_target_delta"] = (
            float(np.mean(correlations)) if correlations else None)
        result["metrics"][f"rear_{side}"] = side_report
    result["metrics"]["rear_wheel_pair_rms"] = {
        "prediction_minus_filtered_target": _summarize(
            np.sqrt(np.mean(filtered_error ** 2, axis=1)),
            run_index, run_ids, valid),
        "prediction_minus_raw_target": _summarize(
            np.sqrt(np.mean(raw_error ** 2, axis=1)),
            run_index, run_ids, valid),
        "raw_minus_filtered_target": _summarize(
            np.sqrt(np.mean(target_delta ** 2, axis=1)),
            run_index, run_ids, valid),
    }
    return result


def _predict_domain(torch, model, metadata: dict[str, Any],
                    data: dict[str, Any], allowed_splits: set[str],
                    device: str, command_offset_frames: int,
                    batch_size: int = 512) -> dict[str, Any]:
    wheel_source = str(metadata.get("wheel_state_source", "filtered_odometry"))
    base_state = physical_state_from_dataset(data, wheel_state_source=wheel_source)
    state = (append_roll_state(data, base_state)
             if model.include_roll_state else base_state)
    rows, run_index = _transition_indices(data, allowed_splits)
    frames = np.asarray(data["frames"], dtype=np.float32)
    raw_history = (
        raw_encoder_history_features(data) if model.include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if model.include_wheel_innovation_history else None)
    history_state_size = int(model.history_state_size)
    wheel_valid = np.ones(len(rows), dtype=bool)
    if wheel_source == "raw_encoder":
        raw_valid = np.asarray(data.get("encoder_raw_valid"), dtype=bool)
        wheel_valid = raw_valid[rows + 1]

    predicted, truth, predicted_pose, truth_pose = [], [], [], []
    truth_pose_all = np.asarray(data["simulator_pose_xyyaw"], dtype=np.float32)
    history_offsets = np.arange(-HISTORY_STEPS + 1, 1, dtype=np.int64)
    for offset in range(0, len(rows), batch_size):
        source = rows[offset:offset + batch_size]
        history_indices = source[:, None] + history_offsets[None, :]
        history = np.concatenate((
            state[history_indices, :history_state_size],
            frames[history_indices, 7:9]), axis=2)
        if raw_history is not None:
            history = np.concatenate((history, raw_history[history_indices]), axis=2)
        initial = state[source]
        delayed = frames[source - 1, 7:9]
        commands = frames[
            _command_rows(source, command_offset_frames), 7:9]
        with torch.no_grad():
            next_state, _, _, _ = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(commands[:, None], dtype=torch.float32,
                                device=device))
            next_pose = integrate_pose(
                torch, next_state,
                torch.as_tensor(truth_pose_all[source], dtype=torch.float32,
                                device=device),
                torch.as_tensor(initial, dtype=torch.float32, device=device))
        predicted.append(next_state[:, 0].cpu().numpy().astype(np.float64))
        truth.append(state[source + 1].astype(np.float64))
        predicted_pose.append(next_pose[:, 0].cpu().numpy().astype(np.float64))
        truth_pose.append(truth_pose_all[source + 1].astype(np.float64))

    predicted = np.concatenate(predicted)
    truth = np.concatenate(truth)
    predicted_pose = np.concatenate(predicted_pose)
    truth_pose = np.concatenate(truth_pose)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    error = predicted[:, :7] - truth[:, :7]
    wheel_pair = np.sqrt(np.mean(error[:, 5:7] ** 2, axis=1))
    speed_error = np.hypot(predicted[:, 0], predicted[:, 1]) - np.hypot(
        truth[:, 0], truth[:, 1])
    position_error = predicted_pose[:, :2] - truth_pose[:, :2]
    heading_error = np.arctan2(
        np.sin(predicted_pose[:, 2] - truth_pose[:, 2]),
        np.cos(predicted_pose[:, 2] - truth_pose[:, 2]))
    errors = {
        "forward_speed_mps": error[:, 0],
        "lateral_speed_mps": error[:, 1],
        "yaw_rate_rps": error[:, 2],
        "steering_feedback_rad": error[:, 3],
        "throttle_feedback_norm": error[:, 4],
        "rear_left_wheel_mps": error[:, 5],
        "rear_right_wheel_mps": error[:, 6],
        "rear_wheel_pair_rms_mps": wheel_pair,
        "body_speed_mps": speed_error,
        "position_radial_m": np.linalg.norm(position_error, axis=1),
        "heading_rad": heading_error,
    }
    report: dict[str, Any] = {
        "transition_count": int(len(rows)),
        "whole_run_ids": sorted(set(run_ids[run_index].tolist())),
        "wheel_state_source": wheel_source,
        "one_step_errors": {
            name: _summarize(
                values, run_index, run_ids, np.ones(len(rows), dtype=bool),
                wheel_valid if "wheel" in name else None)
            for name, values in errors.items()
        },
        "wheel_measurement_semantics": _wheel_measurement_semantics(
            data, rows, run_index, run_ids, predicted, truth, wheel_source),
        "stratified_errors": {},
    }
    covariates = _covariates(
        data, base_state, rows, command_offset_frames)
    for factor, (edges, labels) in GROUPS.items():
        factor_rows: dict[str, Any] = {}
        values = covariates[factor]
        for index, label in enumerate(labels):
            selected = np.isfinite(values) & (values >= edges[index]) & (
                values < edges[index + 1])
            if not np.any(selected):
                continue
            factor_rows[label] = {
                "transition_count": int(selected.sum()),
                "independent_whole_runs": int(len(set(run_index[selected].tolist()))),
                "metrics": {
                    name: _summarize(values_, run_index, run_ids, selected,
                                     wheel_valid if "wheel" in name else None)
                    for name, values_ in errors.items()
                },
            }
        report["stratified_errors"][factor] = factor_rows
    return report


def diagnose(checkpoint: Path, dynamic_dataset: Path,
             practice_dataset: Path, output: Path,
             device: str = "cuda",
             command_offset_frames: int = 0) -> dict[str, Any]:
    if command_offset_frames not in (-1, 0):
        raise ValueError("command offset must be -1 or 0 frames")
    checkpoint, dynamic_dataset, practice_dataset, output = (
        path.resolve() for path in
        (checkpoint, dynamic_dataset, practice_dataset, output))
    torch, model, metadata = _load_model(checkpoint, device)
    results = {}
    for name, dataset_path, splits in (
            ("dynamic_validation", dynamic_dataset, {"validation"}),
            ("practice_unseen", practice_dataset, {"unseen_practice"})):
        data = _load_dataset(dataset_path)
        results[name] = {
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            **_predict_domain(
                torch, model, metadata, data, splits, device,
                command_offset_frames),
        }
    result = {
        "schema_version": 1,
        "purpose": "held-out one-transition error attribution; separates local error from recursive accumulation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_runs": metadata["training_runs"],
        "test_and_final_test_used": False,
        "future_truth_or_feedback_used_for_prediction": False,
        "initialization": "each transition uses only its own current state and preceding 80 measured frames; next-state labels are scoring only",
        "command_offset_frames_from_target_state_row": int(
            command_offset_frames),
        "command_alignment_note": (
            "-1 uses the command stored with the source-state row; 0 uses "
            "the command stored with the target-state row"),
        "cadence_s": DT_S,
        "domains": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dynamic-dataset", type=Path,
                        default=DEFAULT_DYNAMIC_DATASET)
    parser.add_argument("--practice-dataset", type=Path,
                        default=DEFAULT_PRACTICE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--command-offset-frames", type=int,
                        choices=(-1, 0), default=0)
    args = parser.parse_args()
    result = diagnose(args.checkpoint, args.dynamic_dataset,
                      args.practice_dataset, args.output, args.device,
                      args.command_offset_frames)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "domains": {name: {
            "transition_count": value["transition_count"],
            "whole_run_ids": value["whole_run_ids"],
            "one_step_errors": value["one_step_errors"],
        } for name, value in result["domains"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
