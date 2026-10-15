#!/usr/bin/env python3
"""Select a yaw-residual contraction on development validation, then score once.

The saved one-step MPC-input tree is not recursively stable at unit gain. This
study tests only a scalar residual gain, selects it on whole-run development
validation (never on test/final or unseen-practice runs), and then measures the
frozen practice-transfer captures once. u/v and steering command remain oracle
inputs to isolate yaw dynamics; this does not certify a complete plant rollout.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from tools.racing.offline.controller import (
    evaluate_mpc_practice_yaw_residual as yaw,
    evaluate_mpc_deployable_yaw_recursive as recursive,
    fit_causal_history_yaw_lateral_plant as teacher,
)


OUTPUT = yaw.ANALYSIS_DIR / "mpc_deployable_yaw_damping_20261012.json"
MODEL_NAME = "mpc_state_steering_history_100ms"
GAINS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
SELECTION_HORIZONS = (4, 10, 20, 40)  # 0.1, 0.25, 0.5, 1.0 s at 40 Hz


def _split_labels(dataset: Path) -> dict[str, list[str]]:
    with np.load(dataset, allow_pickle=False) as archive:
        return {
            split: sorted(set(archive["run_ids"].astype(str)[
                archive["run_splits"].astype(str) == split]))
            for split in set(archive["run_splits"].astype(str))
        }


def _run_one_sequence(model: Any, columns: list[int], data: dict[str, Any],
                      sequence_id: int, gain: float) -> dict[str, Any]:
    start, end = map(int, data["sequence_bounds"][sequence_id])
    first = start + teacher.HISTORY
    if end - first <= max(SELECTION_HORIZONS):
        return {"eligible": False, "steps": max(0, end - first - 1)}
    state = data["state"]
    pose = data["pose"]
    history = [row.copy() for row in state[start:first + 1]]
    yaw_est = [float(history[-1][2])]
    heading_est = [float(pose[first, 2])]
    xy_est = [np.asarray(pose[first, :2], dtype=np.float64)]
    rate_error, heading_error, position_error = [], [], []
    for index in range(first, end - 1):
        feature = teacher._features(np.asarray(history[-(teacher.HISTORY + 1):]))
        delta_rate = float(model.predict(feature[None, columns])[0])
        next_rate = float(history[-1][2] + gain * delta_rate)
        current = history[-1]
        next_truth = state[index + 1]
        row = next_truth.copy()
        row[2] = next_rate

        rate_mid = 0.5 * (current[2] + next_rate)
        heading_mid = heading_est[-1] + 0.5 * teacher.DT * rate_mid
        u_mid = 0.5 * (current[0] + next_truth[0])
        v_mid = 0.5 * (current[1] + next_truth[1])
        xy_est.append(xy_est[-1] + teacher.DT * np.asarray((
            u_mid * math.cos(heading_mid) - v_mid * math.sin(heading_mid),
            u_mid * math.sin(heading_mid) + v_mid * math.cos(heading_mid))))
        heading_est.append(math.atan2(math.sin(
            heading_est[-1] + teacher.DT * rate_mid), math.cos(
            heading_est[-1] + teacher.DT * rate_mid)))
        yaw_est.append(next_rate)
        rate_error.append(next_rate - float(next_truth[2]))
        heading_error.append(math.atan2(
            math.sin(heading_est[-1] - float(pose[index + 1, 2])),
            math.cos(heading_est[-1] - float(pose[index + 1, 2]))))
        position_error.append(float(np.linalg.norm(
            xy_est[-1] - pose[index + 1, :2])))
        history.append(row)

    rerr = np.asarray(rate_error, dtype=np.float64)
    herr = np.asarray(heading_error, dtype=np.float64)
    perr = np.asarray(position_error, dtype=np.float64)
    metrics = {"eligible": True, "steps": int(len(rerr))}
    for horizon in SELECTION_HORIZONS:
        if len(rerr) >= horizon:
            metrics[f"{horizon * teacher.DT:g}s"] = {
                "yaw_rate_rmse_radps": float(np.sqrt(np.mean(rerr[:horizon] ** 2))),
                "heading_endpoint_abs_rad": float(abs(herr[horizon - 1])),
                "position_endpoint_error_m": float(perr[horizon - 1]),
            }
    metrics["full_sequence"] = {
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(rerr ** 2))),
        "yaw_rate_p95_abs_radps": float(np.quantile(np.abs(rerr), 0.95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(herr ** 2))),
        "position_rmse_m": float(np.sqrt(np.mean(perr ** 2))),
    }
    return metrics


def _score_dataset(model: Any, columns: list[int], data: dict[str, Any],
                   run_ids: set[str], gains: tuple[float, ...]
                   ) -> dict[str, Any]:
    run_indices: dict[str, list[int]] = defaultdict(list)
    for sequence_id, run_index in enumerate(data["sequence_run_index"]):
        run_id = str(data["run_ids"][int(run_index)])
        if run_id in run_ids:
            run_indices[run_id].append(sequence_id)
    results = {}
    for gain in gains:
        per_run = {}
        for run_id, sequence_ids in sorted(run_indices.items()):
            sequences = [_run_one_sequence(model, columns, data, seq, gain)
                         for seq in sequence_ids]
            eligible = [item for item in sequences if item.get("eligible")]
            per_run[run_id] = {
                "sequences": len(sequences),
                "eligible_sequences": len(eligible),
                "sequence_metrics": eligible,
            }
        results[f"{gain:g}"] = per_run
    return results


def _run_macro_scores(results: dict[str, Any]) -> dict[str, Any]:
    scores = {}
    for gain, runs in results.items():
        by_horizon: dict[str, list[float]] = defaultdict(list)
        contributing_runs: dict[str, int] = defaultdict(int)
        for run_data in runs.values():
            values_by_horizon: dict[str, list[float]] = defaultdict(list)
            for sequence in run_data["sequence_metrics"]:
                for horizon in tuple(f"{h * teacher.DT:g}s"
                                     for h in SELECTION_HORIZONS):
                    if horizon in sequence:
                        values_by_horizon[horizon].append(
                            sequence[horizon]["yaw_rate_rmse_radps"])
            for horizon, values in values_by_horizon.items():
                if values:
                    by_horizon[horizon].append(float(np.mean(values)))
                    contributing_runs[horizon] += 1
        horizon_scores = {
            horizon: {
                "run_macro_yaw_rate_rmse_radps": float(np.mean(values)),
                "independent_runs": contributing_runs[horizon],
            }
            for horizon, values in by_horizon.items()
        }
        required = [f"{h * teacher.DT:g}s" for h in SELECTION_HORIZONS]
        if any(horizon not in horizon_scores or horizon_scores[horizon][
                "independent_runs"] < 3 for horizon in required):
            composite = None
        else:
            composite = float(np.mean([
                horizon_scores[horizon]["run_macro_yaw_rate_rmse_radps"]
                for horizon in required]))
        scores[gain] = {"horizons": horizon_scores,
                        "selection_composite_radps": composite}
    return scores


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    base_path = yaw._workspace_path(benchmark["dataset"])
    if teacher._sha256(base_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen benchmark data hash changed")
    bundle = joblib.load(recursive.ARTIFACT_DIR / f"{MODEL_NAME}.joblib")
    model = bundle["model"]
    columns = [teacher.FEATURE_NAMES.index(name)
               for name in bundle["feature_names"]]

    base_ids = _split_labels(base_path)
    base_dev_ids = set(base_ids.get("validation", []))
    forbidden = set(base_ids.get("test", [])) | set(base_ids.get("final_test", []))
    if base_dev_ids & forbidden:
        raise ValueError("split rosters overlap")
    base_dev = teacher._collect_rows(base_path, "validation", base_dev_ids)
    paired_ids = _split_labels(teacher.PAIRED_DATASET)
    paired_dev_ids = set(paired_ids.get("validation", []))
    paired_dev = teacher._collect_rows(
        teacher.PAIRED_DATASET, "validation", paired_dev_ids)

    # Each prediction is a single sample in a sequential rollout. OpenMP
    # parallelism adds thread-launch overhead here without useful work.
    with threadpool_limits(limits=1):
        base_results = _score_dataset(
            model, columns, base_dev, base_dev_ids, GAINS)
        paired_results = _score_dataset(
            model, columns, paired_dev, paired_dev_ids, GAINS)
    selection = {
        "race_domain_openplane": _run_macro_scores(base_results),
        "paired_throttle_slew": _run_macro_scores(paired_results),
    }
    composite_by_gain = {}
    for gain in map(lambda value: f"{value:g}", GAINS):
        values = [selection[name][gain]["selection_composite_radps"]
                  for name in selection]
        composite_by_gain[gain] = (
            float(np.mean(values)) if all(value is not None for value in values)
            else None)
    eligible = {gain: score for gain, score in composite_by_gain.items()
                if score is not None}
    if not eligible:
        raise ValueError("not enough independent dev-validation runs for gain selection")
    selected_gain = min(eligible, key=eligible.get)

    practice_ids = set(teacher.VALIDATION_IDS)
    practice = teacher._collect_rows(base_path, "validation", practice_ids)
    with threadpool_limits(limits=1):
        practice_results = _score_dataset(
            model, columns, practice, practice_ids, (float(selected_gain),))
    report = {
        "study": "development-selected damping for recursive yaw-only response",
        "offline_only": True,
        "runtime_model_changed": False,
        "simulator_was_run": False,
        "raceline_changed": False,
        "test_or_final_test_used_for_selection": False,
        "unseen_practice_used_for_selection": False,
        "model": MODEL_NAME,
        "training_feature_names": bundle["feature_names"],
        "selection": {
            "prespecified_gains": list(GAINS),
            "metric": (
                "equal mean of run-macro yaw-rate RMSE at 0.1, 0.25, 0.5, and 1.0 s; "
                "equal weight to race-domain and paired throttle-slew validation sets"),
            "development_run_ids": {
                "race_domain_openplane": sorted(base_dev_ids),
                "paired_throttle_slew": sorted(paired_dev_ids),
            },
            "sealed_run_ids_not_read": sorted(forbidden),
            "per_source_scores": selection,
            "combined_score_by_gain": composite_by_gain,
            "selected_gain": float(selected_gain),
        },
        "frozen_practice_transfer": {
            "run_ids": sorted(practice_ids),
            "scores": practice_results[f"{selected_gain}"],
            "selection_use": False,
        },
        "rollout_policy": (
            "Yaw rate alone is recursive; GT u/v and steering command are supplied "
            "at each future step. The candidate is not a full plant rollout."),
        "limitations": [
            "Damping is selected on development simulation captures; transfer evidence is only two practice captures.",
            "Oracle u/v and command inputs make this an optimistic yaw-only experiment.",
            "The base feature table is piecewise constant and is not differentiable MPC dynamics.",
            "The model's input state is bridge truth, not deployed EKF state.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "combined_selection_scores": report["selection"]["combined_score_by_gain"],
        "selected_gain": report["selection"]["selected_gain"],
        "practice_transfer": report["frozen_practice_transfer"]["scores"],
    }, indent=2, sort_keys=True))
