#!/usr/bin/env python3
"""Run grouped exploratory fits on a closed throttle-surface analysis JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


FEATURE_SETS = {
    "commands_and_speed": (
        "speed_at_throttle_step_mps",
        "throttle_start_norm",
        "throttle_end_norm",
        "throttle_delta_command_norm",
        "steering_command_rad",
    ),
    "commands_speed_and_lateral_yaw": (
        "speed_at_throttle_step_mps",
        "throttle_start_norm",
        "throttle_end_norm",
        "throttle_delta_command_norm",
        "steering_command_rad",
        "baseline_state.vy_mps",
        "baseline_state.yaw_rate_rps",
    ),
    "commands_speed_and_tilt_magnitude": (
        "speed_at_throttle_step_mps",
        "throttle_start_norm",
        "throttle_end_norm",
        "throttle_delta_command_norm",
        "steering_command_rad",
        "baseline_state.tilt_deg",
    ),
    "commands_and_measured_body_state": (
        "throttle_start_norm",
        "throttle_end_norm",
        "throttle_delta_command_norm",
        "steering_command_rad",
        "baseline_state.speed_mps",
        "baseline_state.vx_mps",
        "baseline_state.vy_mps",
        "baseline_state.yaw_rate_rps",
        "baseline_state.tilt_deg",
    ),
}
TARGETS = ("response_speed_change_mps", "feedback_response_t50_s")


def _model_set() -> dict[str, Any]:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "quadratic_ridge": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=10.0),
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=1.0,
            random_state=20260930, n_jobs=2,
        ),
    }


def _finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _feature_value(row: dict[str, Any], name: str) -> Any:
    if name.startswith("baseline_state."):
        baseline = row.get("baseline_state")
        if not isinstance(baseline, dict):
            return None
        return baseline.get(name.split(".", 1)[1])
    return row.get(name)


def fit(analysis_path: Path) -> dict[str, Any]:
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    rows = [row for row in analysis.get("conditions", [])
            if row.get("usable_for_response_fit") is True]
    if len(rows) < 50:
        raise ValueError(f"only {len(rows)} fit-usable conditions; need at least 50")

    scores: dict[str, Any] = {}
    for target in TARGETS:
        target_scores: dict[str, Any] = {"feature_sets": {}}
        for feature_set_name, features in FEATURE_SETS.items():
            usable = [row for row in rows
                      if _finite(row.get(target))
                      and all(_finite(_feature_value(row, name))
                              for name in features)]
            if len(usable) < 50:
                continue
            x = np.asarray([[_feature_value(row, name) for name in features]
                            for row in usable], dtype=float)
            y = np.asarray([float(row[target]) for row in usable])
            groups_by_split = {
                "held_out_replicate_groups": np.asarray([
                    f"{round(float(row['steering_command_rad']) * 10_000)}:"
                    f"{round(float(row['throttle_start_norm']) * 100)}:"
                    f"{round(float(row['throttle_end_norm']) * 100)}"
                    for row in usable]),
                "held_out_throttle_transitions": np.asarray([
                    f"{round(float(row['throttle_start_norm']) * 100)}:"
                    f"{round(float(row['throttle_end_norm']) * 100)}"
                    for row in usable]),
                "held_out_steering_angles": np.asarray([
                    str(round(float(row["steering_command_rad"]) * 10_000))
                    for row in usable]),
            }
            feature_scores: dict[str, Any] = {
                "samples": len(y),
                "features": list(features),
                "target_range": [float(np.min(y)), float(np.max(y))],
                "splits": {},
            }
            for split_name, groups in groups_by_split.items():
                group_count = len(np.unique(groups))
                splitter = (LeaveOneGroupOut()
                            if split_name == "held_out_steering_angles"
                            else GroupKFold(n_splits=min(5, group_count)))
                split_scores = {}
                for model_name, model in _model_set().items():
                    predictions = cross_val_predict(
                        model, x, y, groups=groups, cv=splitter, n_jobs=1)
                    split_scores[model_name] = {
                        "rmse": float(math.sqrt(mean_squared_error(y, predictions))),
                        "mae": float(mean_absolute_error(y, predictions)),
                        "r2": float(r2_score(y, predictions)),
                    }
                feature_scores["splits"][split_name] = {
                    "groups": int(group_count),
                    "models": split_scores,
                }
            target_scores["feature_sets"][feature_set_name] = feature_scores
        scores[target] = target_scores

    return {
        "source_analysis": str(analysis_path.resolve()),
        "source_passed_integrity_gates": analysis.get("passed_integrity_gates"),
        "fit_usable_conditions": len(rows),
        "validation": (
            "Grouped out-of-fold estimates. Replicates of the same steering and "
            "throttle transition remain in one fold; separate splits hold out "
            "whole throttle transitions or whole steering angles. One capture "
            "only: exploratory, not a run-level generalization claim."
        ),
        "scores": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, help="closed-bag analysis JSON")
    parser.add_argument("--output", type=Path,
                        help="output JSON (default: beside input)")
    args = parser.parse_args()
    output = args.output or args.analysis.with_name("preliminary_fit.json")
    try:
        result = fit(args.analysis)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "fit_usable_conditions": result["fit_usable_conditions"],
        "output": str(output.resolve()),
        "scores": result["scores"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
