#!/usr/bin/env python3
"""Measure how much lateral-acceleration history explains yaw error."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.racing.offline.controller import (
    evaluate_mpc_practice_yaw_residual as yaw,
    fit_causal_history_yaw_lateral_plant as teacher,
)
from tools.racing.offline.controller import fit_stable_ar_yaw_response as stable


OUTPUT = yaw.ANALYSIS_DIR / "stable_ar_yaw_imu_ay_ablation_20261012.json"
ARTIFACT_DIR = yaw.ANALYSIS_DIR / "stable_ar_yaw_imu_ay_ablation_20261012_models"
RHO = 0.8
MODEL_KWARGS = {
    "loss": "squared_error", "learning_rate": 0.05,
    "max_iter": 120, "max_leaf_nodes": 15,
    "min_samples_leaf": 120, "l2_regularization": 1.0,
    "early_stopping": False,
}


def _feature_columns() -> tuple[list[str], list[int]]:
    names = teacher.FEATURE_NAMES
    selected = set(stable._features()[0]) | {"imu_ay_mps2"}
    selected.update(f"dimu_ay_mps2_over_{lag}_frames" for lag in (1, 2, 4))
    missing = selected - set(names)
    if missing:
        raise ValueError(f"requested IMU ay features missing: {sorted(missing)}")
    ordered = [name for name in names if name in selected]
    return ordered, [names.index(name) for name in ordered]


def _split_roster(path: Path) -> dict[str, set[str]]:
    with np.load(path, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
    return {split: set(run_ids[splits == split]) for split in np.unique(splits)}


def _one_step(model: Any, data: dict[str, Any], columns: list[int]
              ) -> dict[str, Any]:
    yaw_index = teacher.FEATURE_NAMES.index("yaw_rate_rps")
    current = data["x"][:, yaw_index]
    predicted = RHO * current + model.predict(data["x"][:, columns])
    error = predicted - (current + data["y"][:, 2])
    absolute = np.abs(error)

    def metrics(values: np.ndarray) -> dict[str, Any]:
        abs_values = np.abs(values)
        return {
            "samples": int(len(values)),
            "rmse_radps": float(np.sqrt(np.mean(values ** 2))),
            "p95_abs_radps": float(np.quantile(abs_values, 0.95)),
            "fraction_abs_le_0p1": float(np.mean(abs_values <= 0.1)),
        }

    return {
        **metrics(error),
        "per_run": {
            run_id: metrics(error[data["run_id"] == run_id])
            for run_id in sorted(set(data["run_id"]))
        },
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists() or ARTIFACT_DIR.exists():
        raise FileExistsError("refusing to overwrite IMU ay yaw ablation")
    benchmark = json.loads(yaw.BENCHMARK.read_text(encoding="utf-8"))
    base_path = yaw._workspace_path(benchmark["dataset"])
    if teacher._sha256(base_path) != benchmark["dataset_sha256"]:
        raise ValueError("frozen practice benchmark hash changed")
    base_splits = _split_roster(base_path)
    paired_splits = _split_roster(teacher.PAIRED_DATASET)
    forbidden = (base_splits.get("test", set())
                 | base_splits.get("final_test", set()))
    base_train = teacher._collect_rows(base_path, "train")
    paired_train = teacher._collect_rows(teacher.PAIRED_DATASET, "train")
    train = teacher._combine_training((base_train, paired_train))
    if set(train["runs"]) & (forbidden | teacher.VALIDATION_IDS):
        raise ValueError("training roster overlaps sealed or practice runs")
    base_dev_ids = base_splits.get("validation", set()) - teacher.VALIDATION_IDS
    paired_dev_ids = paired_splits.get("validation", set())
    base_dev = teacher._collect_rows(base_path, "validation", base_dev_ids)
    paired_dev = teacher._collect_rows(
        teacher.PAIRED_DATASET, "validation", paired_dev_ids)
    practice = teacher._collect_rows(base_path, "validation",
                                     teacher.VALIDATION_IDS)

    feature_names, columns = _feature_columns()
    yaw_index = teacher.FEATURE_NAMES.index("yaw_rate_rps")
    target = train["next_y"][:, 2] - RHO * train["x"][:, yaw_index]
    model = HistGradientBoostingRegressor(**MODEL_KWARGS)
    model.fit(train["x"][:, columns], target, sample_weight=train["weight"])

    with threadpool_limits(limits=1):
        development_scores = {
            "race_domain": stable._score(
                model, columns, base_dev, base_dev_ids, RHO),
            "paired_throttle_slew": stable._score(
                model, columns, paired_dev, paired_dev_ids, RHO),
        }
        practice_scores = stable._score(
            model, columns, practice, teacher.VALIDATION_IDS, RHO)

    ARTIFACT_DIR.mkdir(parents=True)
    artifact_path = ARTIFACT_DIR / "yaw_ar_rho_0.8_with_imu_ay_history.joblib"
    joblib.dump({
        "model": model,
        "rho": RHO,
        "feature_names": feature_names,
        "training_run_ids": train["runs"],
        "training_dataset_hashes": {
            "race_domain": base_train["dataset_sha256"],
            "paired_throttle_slew": paired_train["dataset_sha256"],
        },
        "heldout_runs_used_for_fit_or_selection": False,
    }, artifact_path, compress=3)

    report = {
        "study": "stable-AR yaw model IMU lateral-acceleration history ablation",
        "offline_only": True,
        "runtime_model_changed": False,
        "simulator_was_run": False,
        "rho": RHO,
        "feature_names": feature_names,
        "training_run_ids": train["runs"],
        "training_samples": int(len(train["y"])),
        "development_run_ids": {
            "race_domain": sorted(base_dev_ids),
            "paired_throttle_slew": sorted(paired_dev_ids),
        },
        "sealed_run_ids_not_read": sorted(forbidden),
        "one_step_teacher_forced": {
            "race_domain": _one_step(model, base_dev, columns),
            "paired_throttle_slew": _one_step(model, paired_dev, columns),
            "frozen_practice_transfer": _one_step(model, practice, columns),
        },
        "recursive_yaw_only": {
            "race_domain": {
                "run_macro": stable._macro(development_scores["race_domain"]),
                "per_run": development_scores["race_domain"],
            },
            "paired_throttle_slew": {
                "run_macro": stable._macro(
                    development_scores["paired_throttle_slew"]),
                "per_run": development_scores["paired_throttle_slew"],
            },
            "frozen_practice_transfer": practice_scores,
        },
        "frozen_practice_used_for_selection": False,
        "future_imu_ay_policy": (
            "The recursive score is an oracle upper bound: at each future step "
            "the recorded IMU ay is supplied. It is not deployable unless ay is "
            "predicted internally."),
        "limitations": [
            "Future u/v, steering-command history, and IMU lateral acceleration are oracle inputs during recursive scoring.",
            "This isolates whether lateral acceleration is explanatory; it is not a complete plant or runtime model.",
            "The HGB mapping is not differentiable and is not an MPC runtime candidate.",
        ],
        "artifact": str(artifact_path.relative_to(teacher.ROOT)),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "one_step_practice": result["one_step_teacher_forced"][
            "frozen_practice_transfer"],
        "recursive_dev": {
            name: value["run_macro"]
            for name, value in result["recursive_yaw_only"].items()
            if name != "frozen_practice_transfer"},
        "recursive_practice": {
            run: stable._macro({run: values})
            for run, values in result["recursive_yaw_only"][
                "frozen_practice_transfer"].items()},
    }, indent=2))
