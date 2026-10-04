#!/usr/bin/env python3
"""Locate where the selected truncated-history model diverges by regime."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.vehicle_dynamics_learning.augmented_state_space_plant import DT_S
from tools.vehicle_dynamics_learning.operating_regions import region_masks
from tools.vehicle_dynamics_learning.run_multirate_vector_field_study import (
    _normalization,
    _write_json,
)
from tools.vehicle_dynamics_learning.run_reencoded_history_transition import (
    FROZEN_EVAL_STARTS,
    HistoryTransition,
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
    _select_eval_windows,
    _training_windows_and_stats,
    sha256_file,
)
from tools.vehicle_dynamics_learning.run_wp19_target_ablation import (
    HIDDEN_SIZE as WP19_HIDDEN_SIZE,
    make_model as make_wp19_model,
    training_statistics,
)


DEFAULT_CHECKPOINT = (TASK_ROOT / "next_phase_after_2129427"
                      / "truncated_history_transition_l2_continuation_v1/checkpoint.pt")
DEFAULT_OUTPUT = (TASK_ROOT / "next_phase_after_2129427"
                  / "truncated_history_divergence_v2.json")
HORIZONS = (10, 30, 80, 200)
REGIONS = (
    "high_speed_near_straight",
    "high_speed_moderate_steering",
    "7_to_9mps_high_steering",
    "low_speed_high_steering",
    "simultaneous_steering_throttle_transition",
    "negative_command_braking",
    "braking_release",
    "throttle_pickup",
    "steering_turn_in",
    "steering_unwind",
    "large_wheel_body_mismatch",
)
ERROR_METRICS = (
    "position_radial_trajectory_rmse_m",
    "heading_trajectory_rmse_rad",
    "u_rmse_mps",
    "v_rmse_mps",
    "yaw_rate_rmse_rps",
)
METRICS = (
    *ERROR_METRICS,
    "u_mean_bias_mps",
    "v_mean_bias_mps",
    "yaw_rate_mean_bias_rps",
)
DIVERGENCE_METRICS = (
    "position_gt_0_10m_first_s",
    "position_gt_0_25m_first_s",
    "heading_gt_0_05rad_first_s",
    "u_gt_0_25mps_first_s",
    "v_gt_0_10mps_first_s",
    "yaw_gt_0_25rps_first_s",
)


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True,
                          text=True).stdout.strip()


def _row_regions(capture) -> dict[str, np.ndarray]:
    reset_rows = np.full(len(capture.frames), -1, dtype=np.int32)
    for index, (start, end) in enumerate(capture.bounds):
        reset_rows[int(start):int(end)] = capture.sequence_reset[index]
    valid = (np.isfinite(capture.frames).all(axis=1)
             & np.isfinite(capture.body).all(axis=1)
             & np.isclose(capture.dt_s, DT_S, rtol=0.0, atol=1e-7))
    return region_masks(capture.frames, reset_rows, capture.packet, valid, DT_S)


def _divergence_row(pred_state: np.ndarray, pred_pose: np.ndarray,
                    truth_state: np.ndarray, truth_pose: np.ndarray,
                    horizon: int) -> dict[str, float]:
    state_error = pred_state[:horizon] - truth_state[:horizon]
    pose_error = pred_pose[:horizon] - truth_pose[:horizon]
    position = np.linalg.norm(pose_error[:, :2], axis=1)
    heading = np.abs(_wrap(pose_error[:, 2]))
    names = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps")
    result = {name: float(np.sqrt(np.mean(state_error[:, channel] ** 2)))
              for channel, name in enumerate(names)}
    result.update({
        "u_mean_bias_mps": float(np.mean(state_error[:, 0])),
        "v_mean_bias_mps": float(np.mean(state_error[:, 1])),
        "yaw_rate_mean_bias_rps": float(np.mean(state_error[:, 2])),
    })
    result.update({
        "position_radial_trajectory_rmse_m": float(np.sqrt(np.mean(position ** 2))),
        "position_endpoint_error_m": float(position[-1]),
        "heading_trajectory_rmse_rad": float(np.sqrt(np.mean(heading ** 2))),
        "heading_endpoint_error_rad": float(heading[-1]),
    })
    thresholds = (
        ("position_gt_0_10m_first_s", position, 0.10),
        ("position_gt_0_25m_first_s", position, 0.25),
        ("heading_gt_0_05rad_first_s", heading, 0.05),
        ("u_gt_0_25mps_first_s", np.abs(state_error[:, 0]), 0.25),
        ("v_gt_0_10mps_first_s", np.abs(state_error[:, 1]), 0.10),
        ("yaw_gt_0_25rps_first_s", np.abs(state_error[:, 2]), 0.25),
    )
    for name, values, threshold in thresholds:
        indexes = np.flatnonzero(values > threshold)
        result[name] = (float((indexes[0] + 1) * DT_S)
                        if len(indexes) else None)
    return result


def _run_region_metrics(pred_state: np.ndarray, pred_pose: np.ndarray,
                        truth_state: np.ndarray, truth_pose: np.ndarray,
                        memberships: list[set[str]], horizon: int
                        ) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for region in ("all", *REGIONS, "steering_left", "steering_right"):
        indexes = [index for index, labels in enumerate(memberships)
                   if region == "all" or region in labels]
        if not indexes:
            continue
        rows = [_divergence_row(pred_state[index], pred_pose[index],
                                truth_state[index], truth_pose[index], horizon)
                for index in indexes]
        aggregated = {}
        for name in rows[0]:
            values = [row[name] for row in rows if row[name] is not None]
            aggregated[name] = float(np.mean(values)) if values else None
        result[region] = {"start_count": len(indexes), **aggregated}
    return result


def _bootstrap(values: list[float], seed: int) -> list[float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(finite), size=(5000, len(finite)))
    return np.quantile(finite[samples].mean(axis=1), [0.025, 0.975]).tolist()


def analyze(checkpoint_path: Path, output_path: Path) -> dict[str, Any]:
    if _git_head() != SOURCE_COMMIT:
        raise RuntimeError(f"expected source {SOURCE_COMMIT}; found {_git_head()}")
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_sha = sha256_file(checkpoint_path)
    expected_sha = checkpoint_path.with_suffix(".sha256").read_text(
        encoding="utf-8").strip()
    if checkpoint_sha != expected_sha:
        raise RuntimeError("checkpoint hash differs from its sidecar")
    torch.set_num_threads(1)
    device = torch.device("cpu")
    data, wp20 = _load_data()
    config, _, _ = _training_windows_and_stats(data, wp20)
    norm_np = _normalization(data, config)
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = HistoryTransition(norm_np["delta_mean"], norm_np["delta_scale"])
    model.load_state_dict(saved["state_dict"], strict=True)

    frozen = json.loads(FROZEN_EVAL_STARTS.read_text(encoding="utf-8"))
    roles = {
        "development_validation": _select_eval_windows(data.validation_windows, 64),
        "practice_diagnostic": _select_eval_windows(data.practice_windows, 64),
    }
    for role, runs in roles.items():
        expected_runs = frozen["split_roles"][role]
        if set(runs) != set(expected_runs):
            raise RuntimeError(f"{role}: run set differs from frozen WP24")

    parent_checkpoint = torch.load(WP19_CHECKPOINT, map_location=device,
                                   weights_only=True)
    parent_model = make_wp19_model(torch.nn, hidden_size=WP19_HIDDEN_SIZE)
    parent_model.load_state_dict(parent_checkpoint["state_dict"], strict=True)
    wp19_stats = training_statistics(data.captures, data.training_windows_80)
    per_run: dict[str, Any] = {}
    for role, runs in roles.items():
        per_run[role] = {}
        for run_id, refs in sorted(runs.items()):
            capture_index = int(refs[0][0])
            capture = data.captures[capture_index]
            masks = _row_regions(capture)
            memberships = []
            for _, sequence_index, source_row in refs:
                row = int(capture.bounds[sequence_index, 0]) + int(source_row)
                labels = {name for name, mask in masks.items() if bool(mask[row])}
                labels.add("steering_left" if capture.frames[row, 3] > 0.0
                           else "steering_right" if capture.frames[row, 3] < 0.0
                           else "steering_zero")
                memberships.append(labels)
            horizon = 200
            candidate = _predict_run(data, refs, model, norm_np, config,
                                     horizon, device)
            parent = _predict_wp19_baseline(
                capture, data.poses[capture_index], refs, horizon,
                wp19_stats, parent_model, device)
            truth_state, truth_pose = candidate[2], candidate[3]
            parent_state = np.concatenate((parent["body"],
                                            truth_state[..., 3:5]), axis=-1)
            per_run[role][run_id] = {"starts": len(refs), "horizons": {}}
            for steps in HORIZONS:
                per_run[role][run_id]["horizons"][str(steps)] = {
                    "history_transition": _run_region_metrics(
                        candidate[0], candidate[1], truth_state, truth_pose,
                        memberships, steps),
                    "wp19_parent": _run_region_metrics(
                        parent_state, parent["pose"], truth_state, truth_pose,
                        memberships, steps),
                }

    summary: dict[str, Any] = {}
    for role, runs in per_run.items():
        summary[role] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            regions = sorted({region
                              for run in runs.values()
                              for method in ("history_transition", "wp19_parent")
                              for region in run["horizons"][key][method]})
            summary[role][key] = {}
            for region in regions:
                run_ids = [run_id for run_id, run in runs.items()
                           if region in run["horizons"][key]["history_transition"]
                           and region in run["horizons"][key]["wp19_parent"]]
                entry: dict[str, Any] = {"independent_run_count": len(run_ids)}
                for method in ("history_transition", "wp19_parent"):
                    entry[method] = {}
                    for metric in METRICS:
                        values = [runs[run_id]["horizons"][key][method][region][metric]
                                  for run_id in run_ids]
                        entry[method][metric] = (float(np.mean(values))
                                                 if values else None)
                entry["parent_minus_candidate"] = {}
                for metric in ERROR_METRICS:
                    deltas = [
                        runs[run_id]["horizons"][key]["wp19_parent"][region][metric]
                        - runs[run_id]["horizons"][key]["history_transition"][region][metric]
                        for run_id in run_ids]
                    entry["parent_minus_candidate"][metric] = {
                        "run_macro_delta": float(np.mean(deltas)) if deltas else None,
                        "run_cluster_bootstrap_95pct_ci": _bootstrap(
                            deltas, 20261003 + horizon + len(region) + len(metric)),
                    }
                entry["median_first_crossing_s"] = {}
                for method in ("history_transition", "wp19_parent"):
                    entry["median_first_crossing_s"][method] = {}
                    for metric in DIVERGENCE_METRICS:
                        run_values = [
                            float(runs[run_id]["horizons"][key][method][region][metric])
                            for run_id in run_ids
                            if runs[run_id]["horizons"][key][method][region][metric]
                            is not None]
                        entry["median_first_crossing_s"][method][metric] = (
                            float(np.median(run_values)) if run_values else None)
                entry["start_count"] = sum(
                    runs[run_id]["horizons"][key]["history_transition"][region][
                        "start_count"] for run_id in run_ids)
                summary[role][key][region] = entry

    report = {
        "study": "per-step recursive divergence by frozen start regime",
        "source_commit": _git_head(),
        "required_source_commit": SOURCE_COMMIT,
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_sha256": sha256_file(WP19_CHECKPOINT),
        "simulator_launched": False,
        "future_truth_or_sensor_input": False,
        "independent_unit": "whole run; starts averaged within run",
        "horizons_steps": list(HORIZONS),
        "regions": list(REGIONS),
        "per_run": per_run,
        "run_macro_summary": summary,
        "limitations": [
            "The frozen WP24 start set is used; high-steering region estimates may have few runs.",
            "Practice data are diagnostic only and contain two independent runs.",
            "This diagnoses rollout error but does not make a model or infer 1 kHz dynamics.",
        ],
    }
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, report)
    return report


def _wrap(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(args.checkpoint, args.output)
    print(json.dumps({"output": args.output.as_posix(),
                      "checkpoint_sha256": report["checkpoint_sha256"],
                      "validation_runs": len(report["per_run"][
                          "development_validation"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
