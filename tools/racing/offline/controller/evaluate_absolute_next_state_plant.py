#!/usr/bin/env python3
"""Compare recursive absolute-next-state and increment plant teachers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.racing.offline.controller import fit_causal_history_yaw_lateral_plant as plant
from tools.racing.offline.controller import evaluate_mpc_practice_yaw_residual as yaw


OUTPUT = yaw.ANALYSIS_DIR / "absolute_next_state_plant_comparison_20261012.json"


def _absolute_one_step(data: dict[str, Any], prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - data["next_y"]
    channels = np.asarray(plant.OUTPUT_INDICES)

    def metric(values: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "bias": float(np.mean(values)),
            "p95_abs": float(np.quantile(np.abs(values), 0.95)),
        }

    per_target = {
        plant.STATE_NAMES[channel]: metric(error[:, offset])
        for offset, channel in enumerate(channels)
    }
    per_run = {}
    for run_id in np.unique(data["run_id"]):
        mask = data["run_id"] == run_id
        per_run[str(run_id)] = {
            plant.STATE_NAMES[channel]: metric(error[mask, offset])
            for offset, channel in enumerate(channels)
        }
    return {"samples": int(len(error)), "per_target": per_target,
            "per_run": per_run}


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    if benchmark.get("frozen") is not True:
        raise ValueError("the practice benchmark must remain frozen")
    base_path = yaw._workspace_path(benchmark["dataset"])
    if plant._sha256(base_path) != benchmark["dataset_sha256"]:
        raise ValueError("the frozen practice dataset hash changed")
    base_train = plant._collect_rows(base_path, "train")
    validation = plant._collect_rows(base_path, "validation", plant.VALIDATION_IDS)
    paired_train = plant._collect_rows(plant.PAIRED_DATASET, "train")
    if set(np.unique(validation["run_id"])) != plant.VALIDATION_IDS:
        raise ValueError("validation must contain exactly frozen r02/r03")
    training = plant._combine_training((base_train, paired_train))
    if set(training["runs"]) & plant.VALIDATION_IDS:
        raise ValueError("held-out practice data leaked into training")

    feature_sets = {
        "current_state_only": plant.CURRENT_FEATURE_INDICES,
        "causal_history": list(range(len(plant.FEATURE_NAMES))),
    }
    models = {}
    one_step = {}
    recursive = {run_id: {} for run_id in sorted(plant.VALIDATION_IDS)}
    support = {
        name: tuple(float(value) for value in np.quantile(
            training["x"][:, index], (0.001, 0.999)))
        for index, name in enumerate(plant.STATE_NAMES)
    }
    for feature_name, feature_indices in feature_sets.items():
        delta_heads = plant._fit_heads(
            training["x"], training["y"], training["weight"], feature_indices)
        absolute_heads = plant._fit_heads(
            training["x"], training["next_y"], training["weight"], feature_indices)
        models[f"increment_{feature_name}"] = delta_heads
        models[f"absolute_next_state_{feature_name}"] = absolute_heads
        delta_prediction = plant._predict(delta_heads, validation["x"], feature_indices)
        absolute_prediction = plant._predict(
            absolute_heads, validation["x"], feature_indices)
        one_step[f"increment_{feature_name}"] = plant._one_step_report(
            validation, delta_prediction)
        one_step[f"absolute_next_state_{feature_name}"] = _absolute_one_step(
            validation, absolute_prediction)
        for run_id in sorted(plant.VALIDATION_IDS):
            recursive[run_id][f"increment_{feature_name}"] = plant._recursive_run(
                delta_heads, validation, run_id, feature_indices, support,
                prediction_mode="delta")
            recursive[run_id][f"absolute_next_state_{feature_name}"] = (
                plant._recursive_run(
                    absolute_heads, validation, run_id, feature_indices, support,
                    prediction_mode="absolute_next_state"))

    result = {
        "study": "recursive absolute-next-state versus increment plant teacher",
        "offline_only": True,
        "simulator_was_run": False,
        "runtime_model_changed": False,
        "raceline_changed": False,
        "test_or_final_test_samples_used_for_fit_or_selection": False,
        "training": {
            "run_count": len(training["runs"]), "run_ids": training["runs"],
            "samples": int(len(training["y"])),
            "race_domain_training_dataset_sha256": base_train["dataset_sha256"],
            "paired_training_dataset_sha256": paired_train["dataset_sha256"],
            "held_out_practice_run_ids": sorted(plant.VALIDATION_IDS),
            "features": plant.FEATURE_NAMES,
            "outputs": [plant.STATE_NAMES[index] for index in plant.OUTPUT_INDICES],
            "model": plant.MODEL_KWARGS,
        },
        "validation": {
            "one_step": one_step,
            "recursive_whole_run": recursive,
            "scoring": [
                "Increment targets predict x[t+1]-x[t]; absolute targets predict x[t+1] directly.",
                "Recursive rollouts use recorded command channels only as exogenous inputs; all predicted state channels are fed back.",
                "The first-error thresholds and training-support exits are diagnostics, not clipped/fallback behavior.",
                "Only frozen unseen practice r02/r03 rows are scored; the mixed source archive is now refused before numeric loading.",
            ],
        },
        "provenance": {
            "benchmark_sha256": plant._sha256(yaw.BENCHMARK),
            "model_library_sha256": plant._sha256(yaw.MODEL_LIBRARY),
            "config_sha256": plant._sha256(yaw.CONFIG),
            "trajectory_sha256": plant._sha256(yaw.TRAJECTORY),
            "validation_bag_sha256": {
                name: plant._sha256(path) for name, path in yaw.base.BAGS.items()},
        },
        "limitations": [
            "Two whole-run validation captures do not support broad statistical generalization claims.",
            "This teacher is offline research only; no odometry or MPC runtime path changed.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "one_step_rmse": {
            name: {target: values["rmse"] for target, values in report[
                "validation"]["one_step"][name]["per_target"].items()}
            for name in report["validation"]["one_step"]
        },
        "recursive": report["validation"]["recursive_whole_run"],
    }, indent=2, sort_keys=True))
