#!/usr/bin/env python3
"""Fit and evaluate a causal body-frame lateral-acceleration transition.

Simulator truth is used only to label ``a_y = dv/dt + u*r`` and score held-out
whole runs. The candidate predicts ``a_y`` from the current/past state,
commands, and wheel history, then reconstructs lateral velocity with the
rigid-body relation ``v_dot = a_y - u*r``. It is research-only; no test/final
split or runtime controller/odometry is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _state,
    evaluate as evaluate_recursive,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DT_S,
    _balanced_training_rows,
    _rmse,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    DEFAULT_OUTPUT as PARENT_DIR,
    _augment,
)
from tools.vehicle_dynamics_learning.fit_recursive_dagger_longitudinal_candidate import (
    sha256,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = PARENT_DIR / "longitudinal_raw_wheel_candidate.joblib"
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "lateral_acceleration_candidate_20261009")
MAX_ROWS_PER_RUN = 8000
SEED = 20261009


def _build(data: dict[str, Any], split: str, feature_variant: str,
           target_mode: str
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source, run_index, _, _, _, _, _, variants, _ = _augment(data, split)
    feature_index = BASE_VARIANT_NAMES.index(feature_variant)
    x = variants[feature_index]
    state = _state(data)
    u_mid = 0.5 * (state[source, 0] + state[source + 1, 0])
    r_mid = 0.5 * (state[source, 2] + state[source + 1, 2])
    if target_mode == "kinematic_difference":
        ay = ((state[source + 1, 1] - state[source, 1]) / DT_S
              + u_mid * r_mid)
    elif target_mode == "simulator_acceleration_next_y":
        acceleration = np.asarray(
            data["simulator_linear_acceleration"], dtype=np.float64)
        ay = acceleration[source + 1, 1]
    else:
        raise ValueError(f"unknown lateral-acceleration target {target_mode!r}")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(ay)
    if not finite.all():
        source, run_index, x, ay = (source[finite], run_index[finite],
                                    x[finite], ay[finite])
    return source, run_index, x.astype(np.float32), ay.astype(np.float32)


def _score(prediction: np.ndarray, target: np.ndarray,
           run_index: np.ndarray, run_ids: np.ndarray) -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        error = prediction[mask] - target[mask]
        per_run[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            "rmse": _rmse(error),
            "bias": float(np.mean(error)),
        }
    return {
        "macro_run_rmse": float(np.mean([row["rmse"]
                                          for row in per_run.values()])),
        "pooled_rmse": _rmse(prediction - target),
        "per_run": per_run,
    }


def _paired_ci(candidate: dict[str, Any], baseline: dict[str, Any]
               ) -> dict[str, Any]:
    common = sorted(set(candidate["per_run"]) & set(baseline["per_run"]))
    delta = np.asarray([
        candidate["per_run"][run]["rmse"] - baseline["per_run"][run]["rmse"]
        for run in common], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    return {
        "runs": len(common),
        "mean_candidate_minus_baseline": float(delta.mean()),
        "paired_run_bootstrap_95pct": [
            float(x) for x in np.quantile(sample, (0.025, 0.975))],
    }


def _one_step(data: dict[str, Any], split: str, parent: dict[str, Any],
              model: Any, target_mode: str) -> dict[str, Any]:
    source, run_index, x, ay_truth = _build(
        data, split, parent["selected_parent_feature_variant"], target_mode)
    state = _state(data)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    ay_candidate = model.predict(x)
    dv_parent = parent["recursive_aux_models"][0].predict(x)
    u_mid = 0.5 * (state[source, 0] + state[source + 1, 0])
    r_mid = 0.5 * (state[source, 2] + state[source + 1, 2])
    ay_parent_implicit = dv_parent / DT_S + u_mid * r_mid
    v_now = state[source, 1]
    v_next = state[source + 1, 1]
    v_candidate = v_now + DT_S * (ay_candidate - u_mid * r_mid)
    predictions = {
        "lateral_acceleration_candidate": _score(
            ay_candidate, ay_truth, run_index, run_ids),
        "parent_v_head_as_implied_lateral_acceleration": _score(
            ay_parent_implicit, ay_truth, run_index, run_ids),
        "parent_v_increment": _score(v_now + dv_parent, v_next,
                                     run_index, run_ids),
        "lateral_acceleration_candidate_reconstructed_v": _score(
            v_candidate, v_next, run_index, run_ids),
        "v_persistence": _score(v_now, v_next, run_index, run_ids),
    }
    return {
        "transitions": int(len(source)),
        "lateral_acceleration_target": target_mode,
        "metrics": predictions,
        "paired_candidate_minus_parent_ay": _paired_ci(
            predictions["lateral_acceleration_candidate"],
            predictions["parent_v_head_as_implied_lateral_acceleration"]),
        "paired_candidate_minus_parent_v": _paired_ci(
            predictions["lateral_acceleration_candidate_reconstructed_v"],
            predictions["parent_v_increment"]),
    }


def fit(dataset_path: Path, practice_path: Path, parent_path: Path,
        output_dir: Path, max_iter: int, threads: int,
        target_mode: str) -> dict[str, Any]:
    started = time.time()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    for path in (dataset_path, practice_path, parent_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = joblib.load(parent_path)
    if sha256(dataset_path) != parent["training_dataset_sha256"]:
        raise ValueError("parent candidate does not match training dataset")
    train_data = _load_dataset(dataset_path)
    train_source, train_runs, train_x, train_ay = _build(
        train_data, "train", parent["selected_parent_feature_variant"],
        target_mode)
    run_ids = np.asarray(train_data["run_ids"]).astype(str)
    rows, weights, row_counts = _balanced_training_rows(
        train_runs, run_ids, MAX_ROWS_PER_RUN, seed=SEED)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=max_iter,
        max_leaf_nodes=19, min_samples_leaf=40, l2_regularization=2.0,
        early_stopping=False, random_state=103)
    with threadpool_limits(limits=threads):
        model.fit(train_x[rows], train_ay[rows], sample_weight=weights)

    validation = _one_step(train_data, "validation", parent, model, target_mode)
    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str))
                      else "validation")
    practice = _one_step(practice_data, practice_split, parent, model,
                         target_mode)

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = dict(parent)
    candidate["recursive_lateral_acceleration_model"] = model
    candidate["lateral_acceleration_target"] = target_mode
    candidate["lateral_acceleration_training_run_ids"] = sorted(
        set(run_ids[train_runs].tolist()))
    candidate["lateral_acceleration_training_rows_by_run"] = row_counts
    bundle_path = output_dir / "lateral_acceleration_candidate.joblib"
    joblib.dump(candidate, bundle_path)

    recursive_reports: dict[str, Any] = {}
    for label, bundle in (("parent_increment", parent),
                          ("candidate_v_increment", candidate),
                          ("candidate_coupled_midpoint", candidate)):
        mode = ("effective_ax_midpoint" if label == "candidate_coupled_midpoint"
                else "increment")
        path = output_dir / f"recursive_{label}.json"
        recursive_reports[label] = evaluate_recursive(
            bundle_path if label != "parent_increment" else parent_path,
            dataset_path, practice_path, path, 24, None, mode, True,
            "predicted", None)

    report = {
        "schema_version": 1,
        "purpose": "test whether explicit learned body-frame lateral acceleration improves v prediction and recursive motion",
        "lateral_acceleration_target": target_mode,
        "training_dataset": str(dataset_path.relative_to(ROOT)),
        "training_dataset_sha256": sha256(dataset_path),
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": sha256(practice_path),
        "parent_bundle": str(parent_path.relative_to(ROOT)),
        "parent_bundle_sha256": sha256(parent_path),
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_bundle_sha256": sha256(bundle_path),
        "training_run_ids": candidate["lateral_acceleration_training_run_ids"],
        "training_rows_by_run": row_counts,
        "test_and_final_test_used": False,
        "validation_or_practice_used_for_fitting_or_selection": False,
        "one_step": {"openplane_validation": validation,
                     "unseen_practice": practice},
        "recursive_evaluations": {
            label: {
                "report": str((output_dir / f"recursive_{label}.json").relative_to(ROOT)),
                "u_update": ("effective_ax_midpoint"
                             if label == "candidate_coupled_midpoint"
                             else "increment"),
                "openplane_5s": recursive_reports[label]["horizons"][
                    "openplane_validation"]["5.000s"]["macro_run_metrics"][
                        "5.000s"]["candidate_macro_run_rmse"],
                "practice_5s": recursive_reports[label]["horizons"][
                    "unseen_practice"]["5.000s"]["macro_run_metrics"][
                        "5.000s"]["candidate_macro_run_rmse"],
                "recursive_ay": {
                    split: recursive_reports[label]["horizons"][split]["5.000s"].get(
                        "recursive_lateral_acceleration_prediction")
                    for split in ("openplane_validation", "unseen_practice")},
            }
            for label in recursive_reports
        },
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "practice evaluation consists of only two independent captures",
            "the model predicts lateral acceleration from simulator-supervised state/history and is not a sensor-only observer",
            "recursive evaluation covers five-second windows, not a complete lap",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--parent-bundle", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ay-target", choices=(
        "kinematic_difference", "simulator_acceleration_next_y"),
        default="kinematic_difference")
    args = parser.parse_args()
    report = fit(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.parent_bundle.resolve(), args.output_dir.resolve(),
                 args.max_iter, args.threads, args.ay_target)
    print(json.dumps({
        "candidate": report["candidate_bundle"],
        "one_step_validation": report["one_step"]["openplane_validation"],
        "one_step_practice": report["one_step"]["unseen_practice"],
        "recursive_5s": report["recursive_evaluations"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
