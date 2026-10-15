#!/usr/bin/env python3
"""Test whether command history fixes the measured throttle-slew yaw residual."""

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


OUTPUT = yaw.ANALYSIS_DIR / "stable_ar_yaw_throttle_history_20261012.json"
ARTIFACT_DIR = yaw.ANALYSIS_DIR / "stable_ar_yaw_throttle_history_20261012_models"
RHO = 0.8  # freeze the previously development-selected feedback coefficient
MODEL_KWARGS = {
    "loss": "squared_error", "learning_rate": 0.05,
    "max_iter": 120, "max_leaf_nodes": 15,
    "min_samples_leaf": 120, "l2_regularization": 1.0,
    "early_stopping": False,
}


def _feature_columns() -> tuple[list[str], list[int]]:
    names = teacher.FEATURE_NAMES
    base_names, _ = stable._features()
    selected = set(base_names) | {"throttle_command_norm"}
    for lag in (1, 2, 4):
        selected.add(f"dthrottle_command_norm_over_{lag}_frames")
    missing = selected - set(names)
    if missing:
        raise ValueError(f"requested throttle-history features missing: {sorted(missing)}")
    ordered = [name for name in names if name in selected]
    return ordered, [names.index(name) for name in ordered]


def _split_roster(path: Path) -> dict[str, set[str]]:
    with np.load(path, allow_pickle=False) as archive:
        run_ids = archive["run_ids"].astype(str)
        splits = archive["run_splits"].astype(str)
    return {split: set(run_ids[splits == split]) for split in np.unique(splits)}


def _metric(error: np.ndarray) -> dict[str, Any]:
    error = np.asarray(error, dtype=np.float64)
    absolute = np.abs(error)
    return {
        "samples": int(len(error)),
        "rmse_radps": float(np.sqrt(np.mean(error ** 2))),
        "p95_abs_radps": float(np.quantile(absolute, 0.95)),
        "fraction_abs_le_0p1": float(np.mean(absolute <= 0.1)),
    }


def _one_step(model: Any, data: dict[str, Any], columns: list[int]
              ) -> dict[str, Any]:
    yaw_index = teacher.FEATURE_NAMES.index("yaw_rate_rps")
    current = data["x"][:, yaw_index]
    predicted = RHO * current + model.predict(data["x"][:, columns])
    truth = current + data["y"][:, 2]
    error = predicted - truth
    out = _metric(error)
    out["per_run"] = {}
    for run_id in sorted(set(data["run_id"])):
        mask = data["run_id"] == run_id
        out["per_run"][run_id] = _metric(error[mask])
    return out


def run() -> dict[str, Any]:
    if OUTPUT.exists() or ARTIFACT_DIR.exists():
        raise FileExistsError("refusing to overwrite throttle-history yaw study")
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
        source_scores = {
            "race_domain_development": stable._score(
                model, columns, base_dev, base_dev_ids, RHO),
            "paired_throttle_slew_development": stable._score(
                model, columns, paired_dev, paired_dev_ids, RHO),
        }
        practice_scores = stable._score(
            model, columns, practice, teacher.VALIDATION_IDS, RHO)

    ARTIFACT_DIR.mkdir(parents=True)
    artifact_path = ARTIFACT_DIR / "yaw_ar_rho_0.8_with_throttle_history.joblib"
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
        "study": "fixed-rho stable-AR yaw model with throttle-command history ablation",
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
            "race_domain_development": _one_step(model, base_dev, columns),
            "paired_throttle_slew_development": _one_step(
                model, paired_dev, columns),
            "frozen_practice_transfer": _one_step(model, practice, columns),
        },
        "recursive_yaw_only": {
            "race_domain_development": {
                "run_macro": stable._macro(source_scores[
                    "race_domain_development"]),
                "per_run": source_scores["race_domain_development"],
            },
            "paired_throttle_slew_development": {
                "run_macro": stable._macro(source_scores[
                    "paired_throttle_slew_development"]),
                "per_run": source_scores["paired_throttle_slew_development"],
            },
            "frozen_practice_transfer": practice_scores,
        },
        "frozen_practice_used_for_selection": False,
        "oracle_inputs": ["future u/v", "future steering command"],
        "limitations": [
            "Only throttle-command history was added; actual wheel slip and actuator feedback are not observed by this model.",
            "The stable-AR coefficient was fixed at the value selected on development data before this feature ablation.",
            "This is a yaw-only oracle-input test, not a full plant rollout or MPC integration test.",
            "The HGB model is not differentiable and is not a runtime MPC model.",
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
        "one_step": result["one_step_teacher_forced"][
            "frozen_practice_transfer"],
        "dev_recursive": {
            name: value["run_macro"]
            for name, value in result["recursive_yaw_only"].items()
            if name != "frozen_practice_transfer"},
        "practice_recursive": {
            run: stable._macro({run: values})
            for run, values in result["recursive_yaw_only"][
                "frozen_practice_transfer"].items()},
    }, indent=2))
