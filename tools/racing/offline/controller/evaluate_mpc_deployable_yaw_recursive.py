#!/usr/bin/env python3
"""Recursively test saved MPC-input yaw heads on frozen practice captures.

Only yaw rate is recursively predicted. Ground-truth u/v and the recorded
steering command are deliberately supplied at every step to isolate yaw-model
stability; this is an optimistic diagnostic, not a full MPC rollout score.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.racing.offline.controller import (
    evaluate_mpc_practice_yaw_residual as yaw,
    fit_causal_history_yaw_lateral_plant as teacher,
)


OUTPUT = yaw.ANALYSIS_DIR / "mpc_deployable_yaw_recursive_20261012.json"
ARTIFACT_DIR = yaw.ANALYSIS_DIR / "mpc_deployable_yaw_feature_ablation_20261012_models"
HORIZONS_STEPS = (1, 4, 10, 20, 40, 80, 200, 400)


def _rmse(values: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else None


def _wrapped(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _pose_errors(predicted: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.linalg.norm(predicted[:, :2] - truth[:, :2], axis=1)
    heading = np.asarray([_wrapped(a - b)
                          for a, b in zip(predicted[:, 2], truth[:, 2])])
    return position, heading


def _run_rollout(model: Any, columns: list[int], state: np.ndarray,
                 pose: np.ndarray, start: int, end: int
                 ) -> dict[str, Any]:
    first = start + teacher.HISTORY
    history = [row.copy() for row in state[start:first + 1]]
    predicted_rate = [float(history[-1][2])]
    predicted_pose = [pose[first].astype(np.float64).copy()]
    step_errors = []
    threshold_events = []
    actual_rate = []
    for index in range(first, end - 1):
        feature = teacher._features(np.asarray(history[-(teacher.HISTORY + 1):]))
        delta_rate = float(model.predict(feature[None, columns])[0])
        next_rate = float(history[-1][2] + delta_rate)
        row = state[index + 1].copy()
        row[2] = next_rate
        current = history[-1]
        next_truth = state[index + 1]
        rate_mid = 0.5 * (current[2] + next_rate)
        yaw_mid = predicted_pose[-1][2] + 0.5 * teacher.DT * rate_mid
        u_mid = 0.5 * (current[0] + next_truth[0])
        v_mid = 0.5 * (current[1] + next_truth[1])
        next_pose = np.asarray((
            predicted_pose[-1][0] + teacher.DT * (
                u_mid * math.cos(yaw_mid) - v_mid * math.sin(yaw_mid)),
            predicted_pose[-1][1] + teacher.DT * (
                u_mid * math.sin(yaw_mid) + v_mid * math.cos(yaw_mid)),
            _wrapped(predicted_pose[-1][2] + teacher.DT * rate_mid),
        ), dtype=np.float64)
        error = next_rate - float(next_truth[2])
        elapsed_s = (index - first + 1) * teacher.DT
        if abs(error) > 0.1 and not threshold_events:
            threshold_events.append({
                "elapsed_s": elapsed_s,
                "signed_error_radps": error,
                "truth_yaw_rate_radps": float(next_truth[2]),
                "predicted_yaw_rate_radps": next_rate,
                "speed_mps": float(np.hypot(next_truth[0], next_truth[1])),
                "abs_steering_command_rad": float(abs(next_truth[7])),
                "steering_command_rate_radps": float(
                    (next_truth[7] - current[7]) / teacher.DT),
                "throttle_feedback_slew_normps": float(
                    (next_truth[4] - history[0][4]) /
                    (teacher.HISTORY * teacher.DT)),
            })
        step_errors.append(error)
        actual_rate.append(float(next_truth[2]))
        predicted_rate.append(next_rate)
        predicted_pose.append(next_pose)
        history.append(row)

    errors = np.asarray(step_errors, dtype=np.float64)
    predicted_pose_array = np.asarray(predicted_pose)
    truth_pose = pose[first:first + len(predicted_pose_array)]
    position_error, heading_error = _pose_errors(predicted_pose_array, truth_pose)
    horizon_metrics = {}
    for horizon in HORIZONS_STEPS:
        if horizon > len(errors):
            continue
        horizon_metrics[f"{horizon * teacher.DT:g}s"] = {
            "yaw_rate_endpoint_error_radps": float(errors[horizon - 1]),
            "yaw_rate_rmse_radps": _rmse(errors[:horizon]),
            "heading_endpoint_error_rad": float(heading_error[horizon]),
            "position_endpoint_error_m": float(position_error[horizon]),
        }
    threshold = np.flatnonzero(np.abs(errors) > 0.1)
    return {
        "predicted_duration_s": float(len(errors) * teacher.DT),
        "samples": len(errors),
        "yaw_rate_rmse_radps": _rmse(errors),
        "yaw_rate_p95_abs_radps": float(np.quantile(np.abs(errors), 0.95)),
        "yaw_rate_max_abs_radps": float(np.max(np.abs(errors))),
        "fraction_abs_yaw_rate_error_le_0p1": float(np.mean(np.abs(errors) <= 0.1)),
        "first_yaw_rate_error_over_0p1": (
            {"elapsed_s": float(threshold[0] * teacher.DT),
             "abs_error_radps": float(abs(errors[threshold[0]]))}
            if len(threshold) else None),
        "heading_rmse_rad": _rmse(heading_error),
        "position_rmse_m": _rmse(position_error),
        "horizons": horizon_metrics,
        "first_error_context": threshold_events[0] if threshold_events else None,
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    dataset_path = yaw._workspace_path(benchmark["dataset"])
    if teacher._sha256(dataset_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen dataset hash changed")
    validation = teacher._collect_rows(
        dataset_path, "validation", teacher.VALIDATION_IDS)
    models = {}
    for name in ("mpc_state_steering_current", "mpc_state_steering_history_100ms"):
        artifact_path = ARTIFACT_DIR / f"{name}.joblib"
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        bundle = joblib.load(artifact_path)
        expected = bundle["feature_names"]
        indices = [teacher.FEATURE_NAMES.index(item) for item in expected]
        models[name] = (bundle["model"], indices, teacher._sha256(artifact_path))

    results = {name: {} for name in models}
    with np.load(dataset_path, allow_pickle=False) as archive:
        state = teacher._state_arrays(archive)
        pose = archive["simulator_pose_xyyaw"].astype(np.float64)
        bounds = archive["sequence_bounds"].astype(np.int64)
        sequence_runs = archive["sequence_run_index"].astype(np.int32)
        run_ids = archive["run_ids"].astype(str)
        run_splits = archive["run_splits"].astype(str)
        sensor_valid = archive["sensor_valid"].astype(bool)
    for sequence_id, (start_raw, end_raw) in enumerate(bounds):
        run_id = str(run_ids[sequence_runs[sequence_id]])
        if run_id not in teacher.VALIDATION_IDS:
            continue
        if str(run_splits[sequence_runs[sequence_id]]) not in ("unseen_practice", "validation"):
            raise ValueError(f"unexpected held-out split role for {run_id}")
        start, end = int(start_raw), int(end_raw)
        if end - start <= teacher.HISTORY + 1:
            continue
        if not sensor_valid[start:end].all():
            # Preserve only contiguous observed segments with no sensor hole.
            valid_indices = np.flatnonzero(sensor_valid[start:end]) + start
            groups = np.split(valid_indices, np.flatnonzero(np.diff(valid_indices) != 1) + 1)
            segments = [(int(g[0]), int(g[-1]) + 1) for g in groups
                        if len(g) > teacher.HISTORY + 1]
        else:
            segments = [(start, end)]
        for segment_index, (segment_start, segment_end) in enumerate(segments):
            label = run_id if len(segments) == 1 else f"{run_id}/segment_{segment_index}"
            for name, (model, columns, _) in models.items():
                results[name][label] = _run_rollout(
                    model, columns, state, pose, segment_start, segment_end)

    report = {
        "study": "recursive MPC-input yaw-only rollout diagnostic",
        "offline_only": True,
        "runtime_model_changed": False,
        "simulator_was_run": False,
        "test_or_final_test_used_for_fit_or_selection": False,
        "training_or_selection_use_of_validation_runs": False,
        "inputs": "MPC-available body state plus recorded steering command history",
        "rollout_policy": (
            "Yaw rate is recursive. Ground-truth u/v, steering-command history, "
            "and measured non-yaw channels are supplied at every step to isolate "
            "yaw response; do not interpret as a fully recursive vehicle or MPC rollout."),
        "results": results,
        "artifacts": {
            name: {"sha256": entry[2]} for name, entry in models.items()},
        "limitations": [
            "Only two independent practice captures are scored.",
            "The recursive test has oracle u/v and commands, so it is optimistic.",
            "Scikit-learn tree models are not differentiable MPC dynamics.",
            "No held-out run was used to select model parameters or architecture.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "results": report["results"],
    }, indent=2, sort_keys=True))
