#!/usr/bin/env python3
"""Fit an explicitly contractive yaw response and test recursive transfer.

Training uses only training captures. A scalar autoregressive coefficient is
fixed in ``r_next = rho*r + f(exogenous history)``; candidate rho values are
selected on development validation rollouts, then scored once on the frozen
practice-transfer runs. Future u/v and steering are oracle inputs here, so the
test isolates yaw response rather than claiming a complete plant rollout.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.racing.offline.controller import (
    evaluate_mpc_practice_yaw_residual as yaw,
    fit_causal_history_yaw_lateral_plant as teacher,
)


OUTPUT = yaw.ANALYSIS_DIR / "stable_ar_yaw_response_20261012.json"
ARTIFACT_DIR = yaw.ANALYSIS_DIR / "stable_ar_yaw_response_20261012_models"
RHO_CANDIDATES = (0.0, 0.5, 0.8, 0.95)
HORIZONS = (4, 10, 20, 40)
MODEL_KWARGS = {
    "loss": "squared_error", "learning_rate": 0.05,
    "max_iter": 120, "max_leaf_nodes": 15,
    "min_samples_leaf": 120, "l2_regularization": 1.0,
    "early_stopping": False,
}


def _features() -> tuple[list[str], list[int]]:
    names = teacher.FEATURE_NAMES
    selected = {
        "u_mps", "v_rear_mps", "steering_command_rad",
        "u_times_steering", "steering_rate_times_speed",
    }
    for lag in (1, 2, 4):
        selected.update({f"d{name}_over_{lag}_frames"
                         for name in ("u_mps", "v_rear_mps",
                                      "steering_command_rad")})
    missing = selected - set(names)
    if missing:
        raise ValueError(f"stable yaw features missing: {sorted(missing)}")
    ordered = [name for name in names if name in selected]
    return ordered, [names.index(name) for name in ordered]


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(error.size),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "fraction_abs_error_le_0p1": float(np.mean(absolute <= 0.1)),
    }


def _one_sequence(model: Any, columns: list[int], data: dict[str, Any],
                  sequence_id: int, rho: float) -> dict[str, Any]:
    start, end = map(int, data["sequence_bounds"][sequence_id])
    first = start + teacher.HISTORY
    if end - first <= max(HORIZONS):
        return {"eligible": False}
    state, pose = data["state"], data["pose"]
    history = [row.copy() for row in state[start:first + 1]]
    heading = float(pose[first, 2])
    xy = np.asarray(pose[first, :2], dtype=np.float64)
    rate_errors, heading_errors, position_errors = [], [], []
    for index in range(first, end - 1):
        features = teacher._features(np.asarray(
            history[-(teacher.HISTORY + 1):]))
        learned_part = float(model.predict(features[None, columns])[0])
        next_rate = rho * float(history[-1][2]) + learned_part
        current = history[-1]
        next_truth = state[index + 1]
        row = next_truth.copy()
        row[2] = next_rate

        rate_mid = 0.5 * (current[2] + next_rate)
        heading_mid = heading + 0.5 * teacher.DT * rate_mid
        u_mid = 0.5 * (current[0] + next_truth[0])
        v_mid = 0.5 * (current[1] + next_truth[1])
        xy = xy + teacher.DT * np.asarray((
            u_mid * math.cos(heading_mid) - v_mid * math.sin(heading_mid),
            u_mid * math.sin(heading_mid) + v_mid * math.cos(heading_mid)))
        heading = math.atan2(math.sin(heading + teacher.DT * rate_mid),
                             math.cos(heading + teacher.DT * rate_mid))
        rate_errors.append(next_rate - float(next_truth[2]))
        heading_errors.append(math.atan2(
            math.sin(heading - float(pose[index + 1, 2])),
            math.cos(heading - float(pose[index + 1, 2]))))
        position_errors.append(float(np.linalg.norm(xy - pose[index + 1, :2])))
        history.append(row)

    rate = np.asarray(rate_errors, dtype=np.float64)
    heading = np.asarray(heading_errors, dtype=np.float64)
    position = np.asarray(position_errors, dtype=np.float64)
    result: dict[str, Any] = {"eligible": True, "steps": int(len(rate))}
    for horizon in HORIZONS:
        if len(rate) >= horizon:
            result[f"{horizon * teacher.DT:g}s"] = {
                "yaw_rate_rmse_radps": float(np.sqrt(np.mean(rate[:horizon] ** 2))),
                "heading_endpoint_abs_rad": float(abs(heading[horizon - 1])),
                "position_endpoint_error_m": float(position[horizon - 1]),
            }
    result["full_sequence"] = {
        "yaw_rate_rmse_radps": float(np.sqrt(np.mean(rate ** 2))),
        "yaw_rate_p95_abs_radps": float(np.quantile(np.abs(rate), 0.95)),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading ** 2))),
        "position_rmse_m": float(np.sqrt(np.mean(position ** 2))),
    }
    return result


def _score(model: Any, columns: list[int], data: dict[str, Any],
           run_ids: set[str], rho: float) -> dict[str, Any]:
    sequence_by_run: dict[str, list[int]] = defaultdict(list)
    for sequence_id, run_index in enumerate(data["sequence_run_index"]):
        run_id = str(data["run_ids"][int(run_index)])
        if run_id in run_ids:
            sequence_by_run[run_id].append(sequence_id)
    scores = {}
    for run_id, sequence_ids in sorted(sequence_by_run.items()):
        sequences = [_one_sequence(model, columns, data, sequence_id, rho)
                     for sequence_id in sequence_ids]
        scores[run_id] = {
            "sequences": len(sequences),
            "eligible_sequences": sum(item["eligible"] for item in sequences),
            "sequence_metrics": [item for item in sequences if item["eligible"]],
        }
    return scores


def _macro(scores: dict[str, Any]) -> dict[str, float | int | None]:
    metrics: dict[str, list[float]] = defaultdict(list)
    for run in scores.values():
        seqs = run["sequence_metrics"]
        for horizon in (f"{count * teacher.DT:g}s" for count in HORIZONS):
            values = [seq[horizon]["yaw_rate_rmse_radps"] for seq in seqs
                      if horizon in seq]
            if values:
                metrics[horizon].append(float(np.mean(values)))
    output: dict[str, float | int | None] = {}
    for horizon, run_values in metrics.items():
        output[horizon] = float(np.mean(run_values))
    output["independent_runs"] = min((len(values) for values in metrics.values()),
                                     default=0)
    if all(f"{count * teacher.DT:g}s" in output for count in HORIZONS):
        output["selection_composite"] = float(np.mean([
            output[f"{count * teacher.DT:g}s"] for count in HORIZONS]))
    else:
        output["selection_composite"] = None
    return output


def run() -> dict[str, Any]:
    if OUTPUT.exists() or ARTIFACT_DIR.exists():
        raise FileExistsError("refusing to overwrite stable yaw study output")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    if benchmark.get("frozen") is not True:
        raise ValueError("practice benchmark is not frozen")
    base_path = yaw._workspace_path(benchmark["dataset"])
    if teacher._sha256(base_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen practice dataset hash changed")

    base_train = teacher._collect_rows(base_path, "train")
    paired_train = teacher._collect_rows(teacher.PAIRED_DATASET, "train")
    train = teacher._combine_training((base_train, paired_train))
    base_splits = {}
    with np.load(base_path, allow_pickle=False) as archive:
        ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
        for split in np.unique(splits):
            base_splits[str(split)] = sorted(set(ids[splits == split]))
    sealed = set(base_splits.get("test", [])) | set(base_splits.get("final_test", []))
    if set(train["runs"]) & sealed or set(train["runs"]) & teacher.VALIDATION_IDS:
        raise ValueError("training data overlaps sealed or unseen-practice runs")

    base_dev_ids = set(base_splits.get("validation", [])) - teacher.VALIDATION_IDS
    paired_splits = {}
    with np.load(teacher.PAIRED_DATASET, allow_pickle=False) as archive:
        ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
        for split in np.unique(splits):
            paired_splits[str(split)] = sorted(set(ids[splits == split]))
    paired_dev_ids = set(paired_splits.get("validation", []))
    base_dev = teacher._collect_rows(base_path, "validation", base_dev_ids)
    paired_dev = teacher._collect_rows(
        teacher.PAIRED_DATASET, "validation", paired_dev_ids)
    practice = teacher._collect_rows(base_path, "validation", teacher.VALIDATION_IDS)

    feature_names, columns = _features()
    sample_weights = train["weight"]
    dev_scores: dict[str, Any] = {}
    candidate_models: dict[float, Any] = {}
    for rho in RHO_CANDIDATES:
        # The only autoregressive feedback is explicitly constrained to rho*r.
        target = train["next_y"][:, 2] - rho * train["x"][:,
            teacher.FEATURE_NAMES.index("yaw_rate_rps")]
        model = HistGradientBoostingRegressor(**MODEL_KWARGS)
        model.fit(train["x"][:, columns], target, sample_weight=sample_weights)
        candidate_models[rho] = model
        dev_scores[f"{rho:g}"] = {
            "race_domain": _score(model, columns, base_dev, base_dev_ids, rho),
            "paired_throttle_slew": _score(
                model, columns, paired_dev, paired_dev_ids, rho),
        }

    composites = {}
    for rho_key, sources in dev_scores.items():
        source_scores = [_macro(value) for value in sources.values()]
        values = [item["selection_composite"] for item in source_scores]
        composites[rho_key] = (float(np.mean(values))
                               if all(value is not None for value in values)
                               else None)
    eligible = {key: value for key, value in composites.items()
                if value is not None}
    if not eligible:
        raise ValueError("insufficient independent dev runs for rho selection")
    selected_key = min(eligible, key=eligible.get)
    selected_rho = float(selected_key)
    selected = candidate_models[selected_rho]
    practice_scores = _score(selected, columns, practice,
                             teacher.VALIDATION_IDS, selected_rho)

    ARTIFACT_DIR.mkdir(parents=True)
    artifact_path = ARTIFACT_DIR / f"yaw_ar_rho_{selected_rho:g}.joblib"
    joblib.dump({
        "model": selected,
        "rho": selected_rho,
        "feature_names": feature_names,
        "target": "next_ground_truth_yaw_rate_minus_rho_times_current_ground_truth_yaw_rate_radps",
        "training_run_ids": train["runs"],
        "training_dataset_hashes": {
            "race_domain": base_train["dataset_sha256"],
            "paired_throttle_slew": paired_train["dataset_sha256"],
        },
        "heldout_runs_used_for_training_or_selection": False,
    }, artifact_path, compress=3)

    report = {
        "study": "explicitly contractive autoregressive yaw-response model",
        "offline_only": True,
        "runtime_model_changed": False,
        "simulator_was_run": False,
        "test_or_final_test_used_for_fit_or_selection": False,
        "practice_used_for_selection": False,
        "training": {
            "run_count": len(train["runs"]),
            "run_ids": train["runs"],
            "samples": int(len(train["y"])),
            "features": feature_names,
            "rho_candidates": list(RHO_CANDIDATES),
        },
        "development_selection": {
            "metric": "mean run-macro yaw-rate RMSE at 0.1, 0.25, 0.5, 1.0 s; equal source weight",
            "race_domain_run_ids": sorted(base_dev_ids),
            "paired_slew_run_ids": sorted(paired_dev_ids),
            "sealed_run_ids_not_read": sorted(sealed),
            "per_rho_per_source": dev_scores,
            "combined_score_by_rho": composites,
            "selected_rho": selected_rho,
            "selected_development_macro": {
                name: _macro(value)
                for name, value in dev_scores[selected_key].items()},
        },
        "frozen_practice_transfer": {
            "run_ids": sorted(teacher.VALIDATION_IDS),
            "scores": practice_scores,
            "selection_use": False,
        },
        "artifact": str(artifact_path.relative_to(teacher.ROOT)),
        "limitations": [
            "Oracle u/v and steering-command histories are supplied at every future step; this isolates yaw response, not a complete plant.",
            "Training uses bridge-truth states rather than deployed EKF states.",
            "Only two frozen practice-transfer captures are available for this comparison.",
            "The piecewise tree is not differentiable and is not an MPC runtime candidate.",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "selected_rho": result["development_selection"]["selected_rho"],
        "dev_scores": result["development_selection"]["combined_score_by_rho"],
        "practice_runs": result["frozen_practice_transfer"]["scores"],
    }, indent=2))
