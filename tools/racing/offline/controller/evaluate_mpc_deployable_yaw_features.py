#!/usr/bin/env python3
"""Check whether the causal yaw gain survives MPC-available inputs.

Fits one-step yaw-rate-increment heads from training captures only, with inputs
restricted to body state and steering command available inside the MPC. Scores
the frozen unseen-practice runs once as a transfer diagnostic; these results do
not select or promote a runtime model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from tools.racing.offline.controller import (
    evaluate_mpc_practice_yaw_residual as yaw,
    fit_causal_history_yaw_lateral_plant as teacher,
)


OUTPUT = yaw.ANALYSIS_DIR / "mpc_deployable_yaw_feature_ablation_20261012.json"
ARTIFACT_DIR = yaw.ANALYSIS_DIR / "mpc_deployable_yaw_feature_ablation_20261012_models"
REFERENCE_REPORT = yaw.ANALYSIS_DIR / "causal_history_yaw_lateral_plant.json"


def _indices(names: tuple[str, ...], selected: set[str]) -> list[int]:
    missing = selected - set(names)
    if missing:
        raise ValueError(f"requested feature names are missing: {sorted(missing)}")
    return [index for index, name in enumerate(names) if name in selected]


def _input_sets() -> dict[str, list[int]]:
    names = tuple(teacher.FEATURE_NAMES)
    current = {"u_mps", "v_rear_mps", "yaw_rate_rps",
               "steering_command_rad"}
    history = set(current)
    for lag in (1, 2, 4):
        history.update({f"d{name}_over_{lag}_frames" for name in current})
    # Both are computed from MPC-known state/control history.
    history.update(("u_times_steering", "steering_rate_times_speed"))
    return {
        "mpc_state_steering_current": _indices(names, current),
        "mpc_state_steering_history_100ms": _indices(names, history),
    }


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    if not error.size:
        return {"samples": 0, "rmse_radps": None, "p95_abs_radps": None,
                "max_abs_radps": None, "fraction_abs_error_le_0p1": None}
    absolute = np.abs(error)
    return {
        "samples": int(error.size),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "max_abs_radps": float(np.max(absolute)),
        "fraction_abs_error_le_0p1": float(np.mean(absolute <= 0.1)),
    }


def _grouped(error: np.ndarray, speed: np.ndarray,
             steering: np.ndarray, throttle_slew: np.ndarray
             ) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for name, edges, values in (
            ("speed_mps", (-np.inf, 3.0, 5.0, 7.0, np.inf), speed),
            ("abs_steering_rad", (-np.inf, 0.08, 0.14, 0.20, 0.30, np.inf),
             np.abs(steering)),
            ("throttle_slew_normps",
             (-np.inf, -0.15, -0.03, 0.03, 0.15, np.inf), throttle_slew)):
        bins = {}
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (values >= lower) & (values < upper)
            label = f"{lower:g}_{upper:g}"
            bins[label] = _metric(error[mask])
        groups[name] = bins
    return groups


def run() -> dict[str, Any]:
    if OUTPUT.exists() or ARTIFACT_DIR.exists():
        raise FileExistsError("refusing to overwrite a prior deployable-feature study")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    if benchmark.get("frozen") is not True:
        raise ValueError("the practice-transfer benchmark is not frozen")
    base_path = yaw._workspace_path(benchmark["dataset"])
    if teacher._sha256(base_path) != benchmark["dataset_sha256"]:
        raise ValueError("the frozen practice-transfer dataset hash changed")

    base_train = teacher._collect_rows(base_path, "train")
    validation = teacher._collect_rows(
        base_path, "validation", teacher.VALIDATION_IDS)
    paired_train = teacher._collect_rows(teacher.PAIRED_DATASET, "train")
    training = teacher._combine_training((base_train, paired_train))
    if set(training["runs"]) & teacher.VALIDATION_IDS:
        raise ValueError("unseen-practice runs leaked into training")
    if set(np.unique(validation["run_id"])) != teacher.VALIDATION_IDS:
        raise ValueError("the frozen unseen-practice roster changed")

    feature_sets = _input_sets()
    models = {}
    predictions = {}
    for model_name, columns in feature_sets.items():
        model = HistGradientBoostingRegressor(**teacher.MODEL_KWARGS)
        model.fit(training["x"][:, columns], training["y"][:, 2],
                  sample_weight=training["weight"])
        models[model_name] = model
        predictions[model_name] = model.predict(validation["x"][:, columns])

    ARTIFACT_DIR.mkdir(parents=True)
    artifact_hashes = {}
    for model_name, model in models.items():
        artifact_path = ARTIFACT_DIR / f"{model_name}.joblib"
        joblib.dump({
            "model": model,
            "feature_names": [teacher.FEATURE_NAMES[i]
                              for i in feature_sets[model_name]],
            "target": "next_ground_truth_yaw_rate_minus_current_ground_truth_yaw_rate_radps",
            "training_run_ids": training["runs"],
            "training_dataset_hashes": {
                "race_domain": base_train["dataset_sha256"],
                "paired_throttle_slew": paired_train["dataset_sha256"],
            },
            "heldout_runs_used_for_training_or_selection": False,
        }, artifact_path, compress=3)
        artifact_hashes[model_name] = teacher._sha256(artifact_path)

    overall = {}
    per_run = {name: {} for name in models}
    grouped = {name: {} for name in models}
    for run_id in sorted(teacher.VALIDATION_IDS):
        rows = validation["run_id"] == run_id
        count = int(np.count_nonzero(rows))
        if count == 0:
            raise ValueError(f"frozen validation run has no samples: {run_id}")
        truth = validation["y"][rows, 2]
        speed = validation["speed"][rows]
        steering = validation["steering"][rows]
        slew = validation["throttle_slew"][rows]
        for model_name in models:
            error = predictions[model_name][rows] - truth
            per_run[model_name][run_id] = _metric(error)
            grouped[model_name][run_id] = _grouped(
                error, speed, steering, slew)
    for model_name in models:
        error = predictions[model_name] - validation["y"][:, 2]
        overall[model_name] = _metric(error)

    prior = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    mpc_reference = prior["validation"].get(
        "production_mpc_one_step_yaw_reference", {})
    result = {
        "study": "MPC-available causal yaw feature ablation",
        "offline_only": True,
        "runtime_model_changed": False,
        "simulator_was_run": False,
        "raceline_changed": False,
        "test_or_final_test_used_for_fit_or_selection": False,
        "interpretation": (
            "This is a teacher-forced one-step transfer diagnostic. Training and "
            "scoring state values are the bridge's aligned body-state labels, "
            "not the deployed EKF estimate. The result tests whether steering "
            "command plus body state can explain yaw response without measured "
            "actuator feedback, wheel speeds, or IMU acceleration; it does not "
            "establish recursive MPC rollout accuracy."),
        "training": {
            "run_ids": training["runs"],
            "run_count": len(training["runs"]),
            "samples": int(len(training["y"])),
            "sample_weighting": "equal total sample weight per independent run",
            "feature_sets": {
                name: [teacher.FEATURE_NAMES[index] for index in indices]
                for name, indices in feature_sets.items()},
            "target": "25 ms ground-truth yaw-rate increment (rad/s)",
            "fit_split": "train only; paired throttle-slew train only",
            "paired_dataset_sha256": paired_train["dataset_sha256"],
            "race_domain_dataset_sha256": base_train["dataset_sha256"],
            "heldout_run_ids": sorted(teacher.VALIDATION_IDS),
        },
        "validation": {
            "per_model_overall": overall,
            "per_model_run": per_run,
            "per_model_run_regimes": grouped,
            "existing_production_mpc_one_step_reference": mpc_reference,
            "samples": int(len(validation["y"])),
            "run_ids": sorted(teacher.VALIDATION_IDS),
            "selection_use": False,
        },
        "artifacts": {
            "directory": str(ARTIFACT_DIR),
            "sha256": artifact_hashes,
        },
        "limitations": [
            "Only two independent unseen-practice runs are available in this benchmark.",
            "The deployed EKF state differs from the aligned bridge state used here.",
            "The trained scikit-learn trees are not differentiable MPC dynamics and are not runtime-integrated.",
            "Long-horizon recursive accuracy has not been established by this test.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "artifacts": report["artifacts"],
        "overall": report["validation"]["per_model_overall"],
        "per_run": report["validation"]["per_model_run"],
    }, indent=2, sort_keys=True))
