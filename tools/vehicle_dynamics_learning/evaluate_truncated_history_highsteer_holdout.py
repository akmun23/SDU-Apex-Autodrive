#!/usr/bin/env python3
"""Score a frozen truncated-history plant on preserved 7.5 m/s holdout runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import (
    DT_S,
    WHEEL_RADIUS_M,
)
from tools.vehicle_dynamics_learning.operating_regions import region_masks
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _normalization,
    _write_json,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    FROZEN_EVAL_STARTS,
    HistoryTransition,
    _metrics,
    _predict_run,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import (
    ROOT,
    SOURCE_COMMIT,
    TASK_ROOT,
    WP19_CHECKPOINT,
)
from tools.vehicle_dynamics_learning.train_augmented_state_space_plant import (
    DEFAULT_DYNAMIC,
    DEFAULT_DYNAMIC_FIXED,
    DEFAULT_PRACTICE,
    DEFAULT_PRACTICE_FIXED,
    _load_data,
    _predict_wp19_baseline,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    Capture,
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    _observable_input_features,
    make_model as make_wp19_model,
    training_statistics,
)
from tools.vehicle_dynamics_learning.rigid_acceleration_history_plant import (
    midpoint_acceleration_from_transition,
)


HIGHSTEER_SOURCE = (ROOT / "live_runs/derived_dynamics_learning_20260928"
                    / "full_modeling_reset_20261001"
                    / "highsteer_75_heldout_validation_source_20261002"
                    / "openplane_dynamics.npz")
DEFAULT_CHECKPOINT = (TASK_ROOT / "next_phase_after_2129427"
                      / "truncated_history_transition_l2_continuation_v1"
                      / "checkpoint.pt")
DEFAULT_OUTPUT = (TASK_ROOT / "next_phase_after_2129427"
                  / "truncated_history_highsteer_holdout_v1.json")
EXPECTED_RUNS = {
    "openplane_highsteer_75_validation_r01",
    "openplane_highsteer_75_validation_r02",
}
MAX_HORIZON = 200
EVAL_HORIZONS = (1, 2, 4, 10, 20, 40, 80, 120, 200)
SAMPLE_MAXIMUM_PER_RUN = 64
METRIC_NAMES = (
    "position_radial_trajectory_rmse_m",
    "position_endpoint_error_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)


def _capture_from_archive(path: Path, expected_runs: set[str],
                          expected_split: str = "validation"
                          ) -> tuple[Capture, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        frames = np.asarray(archive["frames"], dtype=np.float32)
        rigid = np.asarray(archive["simulator_rigid_state"], dtype=np.float32)
        bounds = np.asarray(archive["sequence_bounds"], dtype=np.int64)
        sequence_run = np.asarray(archive["sequence_run_index"], dtype=np.int32)
        sequence_reset = np.asarray(archive["sequence_reset_index"], dtype=np.int32)
        packet = np.asarray(archive["packet_sequence"], dtype=np.int64)
        dt_s = np.asarray(archive["dt_s"], dtype=np.float32)
        run_ids = np.asarray(archive["run_ids"]).astype(str)
        splits = np.asarray(archive["run_splits"]).astype(str)
        pose = np.asarray(archive["simulator_pose_xyyaw"], dtype=np.float32)
        if not expected_runs.issubset(set(run_ids)):
            raise ValueError("requested holdout run IDs are absent from dataset")
        if any(str(splits[index]) != expected_split
               for index, run_id in enumerate(run_ids)
               if str(run_id) in expected_runs):
            raise ValueError(
                f"requested whole-run captures are not {expected_split} runs")
        if not np.allclose(dt_s, 0.025, rtol=0.0, atol=1e-7):
            raise ValueError("high-steer holdout is not fixed 25 ms data")
        com_u, com_v, yaw_rate = rigid[:, 7], rigid[:, 8], rigid[:, 12]
        body = np.column_stack((com_u, com_v - 0.15532 * yaw_rate,
                                yaw_rate)).astype(np.float32)
        encoder_rate = frames[:, 5:7].copy()
        encoder_valid = np.isfinite(encoder_rate).all(axis=1)
        acceleration = np.full((len(body), 3), np.nan, dtype=np.float32)
        for begin_raw, end_raw in bounds:
            begin, end = int(begin_raw), int(end_raw)
            for row in range(begin, end - 1):
                acceleration[row] = midpoint_acceleration_from_transition(
                    body[row], body[row + 1], 0.025).astype(np.float32)
    features = _observable_input_features(frames)
    capture = Capture(
        name=f"preserved_highsteer_{expected_split}",
        source_path=path,
        fixed_path=path,
        parent_path=path,
        parent_sha256=sha256_file(path),
        frames=frames,
        dt_s=dt_s,
        packet=packet,
        bounds=bounds,
        sequence_run=sequence_run,
        sequence_reset=sequence_reset,
        run_ids=run_ids,
        splits=splits,
        rigid=rigid,
        encoder_rate=encoder_rate,
        encoder_valid=encoder_valid,
        body=body,
        acceleration=acceleration,
        input_features=features,
    )
    return capture, pose


def _evaluation_refs(capture: Capture, expected_runs: set[str],
                     maximum_per_run: int | None = SAMPLE_MAXIMUM_PER_RUN
                     ) -> dict[str, list[tuple[int, int, int]]]:
    by_run: dict[str, list[tuple[int, int, int]]] = {run: [] for run in expected_runs}
    for sequence_index, ((start_raw, end_raw), run_raw) in enumerate(
            zip(capture.bounds, capture.sequence_run)):
        run_id = str(capture.run_ids[int(run_raw)])
        if run_id not in expected_runs:
            continue
        start, end = int(start_raw), int(end_raw)
        if str(capture.splits[int(run_raw)]) != "validation":
            continue
        if (end - start < 80 + MAX_HORIZON + 1
                or not np.isfinite(capture.input_features[start:end]).all()
                or not np.isfinite(capture.body[start:end]).all()
                or not np.isfinite(capture.frames[start:end]).all()
                or not np.isfinite(capture.dt_s[start:end]).all()
                or np.any(np.diff(capture.packet[start:end]) != 1)):
            continue
        by_run[run_id].extend((0, sequence_index, row)
                              for row in range(79, end - start - MAX_HORIZON))
    for run_id, refs in by_run.items():
        if not refs:
            raise ValueError(f"{run_id}: no complete five-second starts")
        if maximum_per_run is not None and len(refs) > maximum_per_run:
            indexes = np.linspace(0, len(refs) - 1, maximum_per_run,
                                  dtype=np.int64)
            by_run[run_id] = [refs[index] for index in indexes]
    return by_run


def _start_regions(capture: Capture, refs: list[tuple[int, int, int]]) -> list[set[str]]:
    reset_rows = np.full(len(capture.frames), -1, dtype=np.int32)
    for sequence_index, (start, end) in enumerate(capture.bounds):
        reset_rows[int(start):int(end)] = capture.sequence_reset[sequence_index]
    valid = (np.isfinite(capture.frames).all(axis=1)
             & np.isfinite(capture.body).all(axis=1))
    masks = region_masks(capture.frames, reset_rows, capture.packet, valid, 0.025)
    result = []
    for _, sequence_index, row in refs:
        index = int(capture.bounds[sequence_index, 0]) + row
        labels = {name for name, mask in masks.items() if bool(mask[index])}
        labels.add("steering_left" if capture.frames[index, 3] > 0.0
                   else "steering_right" if capture.frames[index, 3] < 0.0
                   else "steering_zero")
        result.append(labels)
    return result


def _score_model(data, capture, pose, refs, model, norm_np, config,
                 horizon: int, device: torch.device) -> dict[str, Any]:
    prediction = _predict_run(data, refs, model, norm_np, config, horizon, device)
    per_start: dict[str, list[int]] = {"all": list(range(len(refs)))}
    labels = _start_regions(capture, refs)
    regions = (
        *(f"S{index}" for index in range(6)),
        *(f"D{index}" for index in range(6)),
        *(f"R{index}" for index in range(4)),
        *(f"throttle_{index}" for index in range(7)),
        *(f"T{index}" for index in range(4)),
        *(f"M{index}" for index in range(7)),
        "7_to_9mps_high_steering", "high_speed_moderate_steering",
        "low_speed_high_steering", "steering_turn_in", "steering_unwind",
        "simultaneous_steering_throttle_transition", "throttle_pickup",
        "negative_command_braking", "braking_release",
        "large_wheel_body_mismatch", "steering_left", "steering_right",
    )
    for region in regions:
        per_start[region] = [index for index, row_labels in enumerate(labels)
                             if region in row_labels]
    role_metrics = {}
    for region, indices in per_start.items():
        if not indices:
            continue
        role_metrics[region] = {"start_count": len(indices), "horizons": {}}
        for steps in EVAL_HORIZONS:
            role_metrics[region]["horizons"][str(steps)] = _metrics(
                prediction[0][indices, :steps], prediction[1][indices, :steps],
                prediction[2][indices, :steps], prediction[3][indices, :steps],
                steps, include_actuator=False)
    return {"metrics": role_metrics, "prediction": prediction}


def _wheel_surface_metrics(parent_prediction: dict[str, np.ndarray],
                           indices: list[int], horizon: int
                           ) -> dict[str, float | int | None]:
    """Score WP19's generated encoder head in surface-speed units."""
    predicted = (parent_prediction["encoder"][indices, :horizon]
                 * WHEEL_RADIUS_M / DT_S)
    truth = (parent_prediction["truth_encoder"][indices, :horizon]
             * WHEEL_RADIUS_M / DT_S)
    valid = parent_prediction["encoder_valid"][indices, :horizon]
    per_start: dict[str, list[float]] = {
        "left_rmse": [], "right_rmse": [], "pair_mean_rmse": [],
        "differential_rmse": [], "left_bias": [], "right_bias": [],
        "left_predicted_mean": [], "right_predicted_mean": [],
        "left_truth_mean": [], "right_truth_mean": [],
    }
    valid_count = 0
    for prediction, target, mask in zip(predicted, truth, valid):
        if not bool(mask.any()):
            continue
        error = prediction[mask] - target[mask]
        valid_count += int(mask.sum())
        per_start["left_rmse"].append(float(np.sqrt(np.mean(error[:, 0] ** 2))))
        per_start["right_rmse"].append(float(np.sqrt(np.mean(error[:, 1] ** 2))))
        per_start["pair_mean_rmse"].append(float(np.sqrt(np.mean(
            error.mean(axis=1) ** 2))))
        per_start["differential_rmse"].append(float(np.sqrt(np.mean(
            (error[:, 0] - error[:, 1]) ** 2))))
        per_start["left_bias"].append(float(np.mean(error[:, 0])))
        per_start["right_bias"].append(float(np.mean(error[:, 1])))
        per_start["left_predicted_mean"].append(float(np.mean(prediction[mask, 0])))
        per_start["right_predicted_mean"].append(float(np.mean(prediction[mask, 1])))
        per_start["left_truth_mean"].append(float(np.mean(target[mask, 0])))
        per_start["right_truth_mean"].append(float(np.mean(target[mask, 1])))
    result: dict[str, float | int | None] = {
        "valid_samples": valid_count,
        "valid_starts": len(per_start["left_rmse"]),
    }
    result.update({
        f"{name}_mps": float(np.mean(values)) if values else None
        for name, values in per_start.items()
    })
    return result


def evaluate(checkpoint_path: Path, output_path: Path,
             dataset_path: Path = HIGHSTEER_SOURCE,
             expected_runs: set[str] = EXPECTED_RUNS,
             all_starts: bool = False,
             model_development_use: bool = False) -> dict[str, Any]:
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    expected_sha = checkpoint_path.with_suffix(".sha256").read_text(
        encoding="utf-8").strip()
    if checkpoint_sha != expected_sha:
        raise RuntimeError("candidate checkpoint hash differs from its sidecar")
    manifest_path = checkpoint_path.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    training_runs = set(manifest["training_runs"])
    overlap = sorted(expected_runs & training_runs)
    if overlap:
        raise RuntimeError(f"holdout runs overlap candidate training: {overlap}")
    selection_runs = set(manifest["validation_runs"])
    selection_overlap = sorted(expected_runs & selection_runs)
    if selection_overlap:
        raise RuntimeError(
            f"holdout runs overlap candidate checkpoint selection: {selection_overlap}")
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    torch.set_num_threads(1)
    device = torch.device("cpu")
    training_data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(training_data, wp20)
    norm_np = _normalization(training_data, config)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if saved.get("metadata", {}).get("source_commit") != SOURCE_COMMIT:
        raise RuntimeError("candidate checkpoint source commit differs")
    model = HistoryTransition(norm_np["delta_mean"], norm_np["delta_scale"])
    model.load_state_dict(saved["state_dict"], strict=True)

    capture, pose = _capture_from_archive(dataset_path, expected_runs)
    refs_by_run = _evaluation_refs(
        capture, expected_runs,
        maximum_per_run=None if all_starts else SAMPLE_MAXIMUM_PER_RUN)
    parent_checkpoint = torch.load(WP19_CHECKPOINT, map_location=device,
                                   weights_only=True)
    parent_model = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE)
    parent_model.load_state_dict(parent_checkpoint["state_dict"], strict=True)
    parent_stats = training_statistics(training_data.captures,
                                       training_data.training_windows_80)
    combined = SimpleNamespace(captures=[capture], poses=[pose])

    per_run: dict[str, Any] = {}
    for run_id, refs in sorted(refs_by_run.items()):
        candidate = _score_model(combined, capture, pose, refs, model,
                                 norm_np, config, MAX_HORIZON, device)
        parent_prediction = _predict_wp19_baseline(
            capture, pose, refs, MAX_HORIZON, parent_stats, parent_model, device)
        parent_state = np.concatenate((parent_prediction["body"],
                                       candidate["prediction"][2][..., 3:5]), axis=-1)
        parent_metrics = {}
        for region, entries in candidate["metrics"].items():
            indices = ([i for i in range(len(refs))] if region == "all" else
                       [i for i, labels in enumerate(_start_regions(capture, refs))
                        if region in labels])
            parent_metrics[region] = {"start_count": len(indices), "horizons": {}}
            for steps in EVAL_HORIZONS:
                parent_metrics[region]["horizons"][str(steps)] = _metrics(
                    parent_state[indices, :steps],
                    parent_prediction["pose"][indices, :steps],
                    candidate["prediction"][2][indices, :steps],
                    candidate["prediction"][3][indices, :steps],
                    steps, include_actuator=False)
                parent_metrics[region]["horizons"][str(steps)][
                    "generated_wheel_surface_speed"] = _wheel_surface_metrics(
                        parent_prediction, indices, steps)
        per_run[run_id] = {
            "start_count": len(refs),
            "history_transition": candidate["metrics"],
            "wp19_parent": parent_metrics,
        }

    report = {
        "study": "unseen whole-run plant evaluation",
        "source_commit": SOURCE_COMMIT,
        "simulator_launched": False,
        "training_or_checkpoint_selection_use": False,
        "model_development_use": model_development_use,
        "future_truth_or_sensor_input": False,
        "wheel_output_scope": (
            "WP19 parent generated-encoder head only; the history-transition candidate has no wheel output head."
            " Truth labels are the recorded 40 Hz rear-wheel surface speeds."),
        "candidate_checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "candidate_checkpoint_sha256": checkpoint_sha,
        "candidate_training_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "candidate_training_runs_disjoint": True,
        "candidate_training_run_overlap": overlap,
        "candidate_checkpoint_selection_runs_disjoint": True,
        "candidate_checkpoint_selection_run_overlap": selection_overlap,
        "parent_checkpoint_sha256": sha256_file(WP19_CHECKPOINT),
        "dataset": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": sha256_file(dataset_path),
        "independent_runs": sorted(per_run),
        "start_selection": ("all valid packet-contiguous starts after 80-sample history"
                            if all_starts else
                            f"{SAMPLE_MAXIMUM_PER_RUN} evenly spaced starts per run"),
        "per_run": per_run,
        "limitations": [
            f"Only {len(per_run)} whole-run capture(s); run-level uncertainty is limited.",
            "Candidate checkpoint selection used the frozen WP24 validation set; this capture was not used for training or checkpoint selection.",
            ("This capture informed model/training-protocol development and is diagnostic, not an untouched confirmation set."
             if model_development_use else
             "This report is diagnostic and was not a preregistered final-test evaluation."),
            "This is a diagnostic whole-run check, not a full-lap offline-simulation confirmation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=HIGHSTEER_SOURCE)
    parser.add_argument("--run-id", action="append", dest="run_ids",
                        help="expected validation run ID; repeat for multiple IDs")
    parser.add_argument("--all-starts", action="store_true",
                        help="evaluate every eligible start instead of a 64-start sample")
    parser.add_argument("--development-diagnostic", action="store_true",
                        help="record that these captures informed model/protocol development")
    args = parser.parse_args()
    expected_runs = set(args.run_ids) if args.run_ids else EXPECTED_RUNS
    report = evaluate(args.checkpoint, args.output, args.dataset, expected_runs,
                      args.all_starts, args.development_diagnostic)
    print(json.dumps({"output": args.output.as_posix(),
                      "runs": report["independent_runs"],
                      "checkpoint_sha256": report["candidate_checkpoint_sha256"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
