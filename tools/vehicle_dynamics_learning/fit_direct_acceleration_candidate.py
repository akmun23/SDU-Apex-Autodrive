#!/usr/bin/env python3
"""Train a causal direct-simulator-acceleration head and test its rollout.

The next 25-ms simulator acceleration sample is a supervised label only. The
candidate integrates the fitted body-frame ``a_x`` with the rigid-body
transport term; it receives no future acceleration or state during rollout.
All other heads remain exactly those of the fixed-40-Hz longitudinal parent.
"""

from __future__ import annotations

import argparse
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
    "direct_acceleration_candidate_20261009")
ROWS_PER_RUN = 8000
SEED = 20261009


def _rows(data: dict[str, Any], split: str, variant: str
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source, run_index, _, _, _, _, _, features, _ = _augment(data, split)
    feature = features[BASE_VARIANT_NAMES.index(variant)]
    acceleration = np.asarray(data["simulator_linear_acceleration"],
                              dtype=np.float64)
    target = acceleration[source + 1, 0]
    finite = np.isfinite(feature).all(axis=1) & np.isfinite(target)
    return (source[finite], run_index[finite], feature[finite].astype(np.float32),
            target[finite].astype(np.float32))


def _score(prediction: np.ndarray, truth: np.ndarray,
           run_index: np.ndarray, run_ids: np.ndarray) -> dict[str, Any]:
    per_run = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        error = prediction[mask] - truth[mask]
        per_run[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            "rmse": _rmse(error),
            "bias": float(np.mean(error)),
        }
    return {
        "macro_run_rmse": float(np.mean([row["rmse"]
                                          for row in per_run.values()])),
        "pooled_rmse": _rmse(prediction - truth),
        "per_run": per_run,
    }


def _one_step(data: dict[str, Any], split: str, parent: dict[str, Any],
              candidate_model: Any) -> dict[str, Any]:
    source, runs, x, direct_ax = _rows(
        data, split, parent["selected_parent_feature_variant"])
    state = _state(data)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    parent_ax = parent["models"][1].predict(x)
    ax_candidate = candidate_model.predict(x)
    u_now, u_next = state[source, 0], state[source + 1, 0]
    v_mid = 0.5 * (state[source, 1] + state[source + 1, 1])
    r_mid = 0.5 * (state[source, 2] + state[source + 1, 2])
    effective_ax = ((u_next - u_now) / DT_S - r_mid * v_mid)
    direct_u = u_now + parent["models"][0].predict(x)
    parent_ax_u = u_now + DT_S * (parent_ax + r_mid * v_mid)
    candidate_ax_u = u_now + DT_S * (ax_candidate + r_mid * v_mid)
    return {
        "transitions": int(len(source)),
        "metrics": {
            "candidate_direct_ax_vs_simulator_label": _score(
                ax_candidate, direct_ax, runs, run_ids),
            "parent_effective_ax_vs_direct_simulator_label": _score(
                parent_ax, direct_ax, runs, run_ids),
            "candidate_direct_ax_vs_kinematic_effective_ax": _score(
                ax_candidate, effective_ax, runs, run_ids),
            "parent_u_increment": _score(direct_u, u_next, runs, run_ids),
            "parent_effective_ax_reconstructed_u": _score(
                parent_ax_u, u_next, runs, run_ids),
            "candidate_direct_ax_reconstructed_u": _score(
                candidate_ax_u, u_next, runs, run_ids),
        },
    }


def _paired_delta(candidate: dict[str, Any], baseline: dict[str, Any],
                  horizon: str, metric: str) -> dict[str, Any]:
    c = candidate[horizon]
    b = baseline[horizon]
    if c["window_signature_sha256"] != b["window_signature_sha256"]:
        raise ValueError("recursive branches did not use identical windows")
    cr, br = c["per_run"], b["per_run"]
    common = sorted(set(cr) & set(br))
    delta = np.asarray([
        cr[run][horizon]["candidate"][metric]
        - br[run][horizon]["candidate"][metric] for run in common])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    return {
        "runs": len(common),
        "mean_candidate_minus_parent": float(delta.mean()),
        "paired_run_bootstrap_95pct": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))],
    }


def fit(dataset: Path, practice: Path, parent_path: Path, output_dir: Path,
        max_iter: int, threads: int) -> dict[str, Any]:
    started = time.time()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if not all(path.is_file() for path in (dataset, practice, parent_path)):
        raise FileNotFoundError("dataset, practice, or parent bundle is missing")
    parent = joblib.load(parent_path)
    if sha256(dataset) != parent["training_dataset_sha256"]:
        raise ValueError("parent candidate does not match training data")
    train = _load_dataset(dataset)
    source, run_index, x, target = _rows(
        train, "train", parent["selected_parent_feature_variant"])
    run_ids = np.asarray(train["run_ids"]).astype(str)
    rows, weights, row_counts = _balanced_training_rows(
        run_index, run_ids, ROWS_PER_RUN, seed=SEED)
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=max_iter,
        max_leaf_nodes=19, min_samples_leaf=40, l2_regularization=2.0,
        early_stopping=False, random_state=107)
    with threadpool_limits(limits=threads):
        model.fit(x[rows], target[rows], sample_weight=weights)

    validation = _one_step(train, "validation", parent, model)
    practice_data = _load_dataset(practice)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str))
                      else "validation")
    practice_result = _one_step(practice_data, practice_split, parent, model)

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = dict(parent)
    candidate["models"] = list(parent["models"])
    candidate["models"][1] = model
    candidate["direct_ax_target"] = "simulator_linear_acceleration[source+1, x]"
    candidate["direct_ax_training_rows_by_run"] = row_counts
    candidate["direct_ax_training_run_ids"] = sorted(
        set(run_ids[run_index].tolist()))
    candidate_path = output_dir / "direct_ax_candidate.joblib"
    joblib.dump(candidate, candidate_path)

    branches = {}
    branch_specs = (
        ("parent_u_increment", parent, "increment"),
        ("parent_effective_ax", parent, "effective_ax_midpoint"),
        ("direct_ax_effective_ax", candidate, "effective_ax_midpoint"),
    )
    for name, bundle, update in branch_specs:
        bundle_path = candidate_path if name == "direct_ax_effective_ax" else parent_path
        branches[name] = evaluate_recursive(
            bundle_path, dataset, practice, output_dir / f"recursive_{name}.json",
            24, None, update, True, "predicted", None)

    horizons = ("0.025s", "0.100s", "0.250s", "0.500s", "1.000s",
                "2.000s", "3.000s", "5.000s")
    metrics = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
               "rear_wheel_pair_rmse_mps", "raw_encoder_pair_rmse_mps",
               "position_rmse_m", "yaw_pose_rmse_rad")
    recursive_comparison = {}
    for split in ("openplane_validation", "unseen_practice"):
        compare_branches = (
            ("parent_effective_ax_vs_parent_u_increment",
             "parent_effective_ax", "parent_u_increment"),
            ("direct_ax_vs_parent_u_increment",
             "direct_ax_effective_ax", "parent_u_increment"),
            ("direct_ax_vs_parent_effective_ax",
             "direct_ax_effective_ax", "parent_effective_ax"),
        )
        recursive_comparison[split] = {}
        for label, candidate_branch, baseline_branch in compare_branches:
            recursive_comparison[split][label] = {
                horizon: {
                    metric: _paired_delta(
                        branches[candidate_branch]["horizons"][split],
                        branches[baseline_branch]["horizons"][split],
                        horizon, metric)
                    for metric in metrics
                }
                for horizon in horizons
            }
    report = {
        "schema_version": 1,
        "purpose": "test direct simulator a_x labels and rigid-body integration against the existing u increment",
        "direct_ax_target": candidate["direct_ax_target"],
        "training_dataset": str(dataset.relative_to(ROOT)),
        "training_dataset_sha256": sha256(dataset),
        "practice_dataset": str(practice.relative_to(ROOT)),
        "practice_dataset_sha256": sha256(practice),
        "parent_bundle": str(parent_path.relative_to(ROOT)),
        "parent_bundle_sha256": sha256(parent_path),
        "candidate_bundle": str(candidate_path.relative_to(ROOT)),
        "candidate_bundle_sha256": sha256(candidate_path),
        "training_run_ids": candidate["direct_ax_training_run_ids"],
        "training_rows_by_run": row_counts,
        "test_and_final_test_used": False,
        "validation_or_practice_used_for_fitting_or_selection": False,
        "one_step": {"openplane_validation": validation,
                     "unseen_practice": practice_result},
        "recursive_comparison": recursive_comparison,
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "simulator acceleration is a training label only; no future acceleration enters recursive inputs",
            "practice results comprise two independent captures",
            "the rollout horizon is five seconds, not a complete lap",
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
    args = parser.parse_args()
    result = fit(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.parent_bundle.resolve(), args.output_dir.resolve(),
                 args.max_iter, args.threads)
    print(json.dumps({
        "candidate": result["candidate_bundle"],
        "one_step_validation": result["one_step"]["openplane_validation"][
            "metrics"],
        "one_step_practice": result["one_step"]["unseen_practice"]["metrics"],
        "recursive_5s": {
            split: {
                branch: {metric: values["mean_candidate_minus_parent"]
                         for metric, values in horizons["5.000s"].items()}
                for branch, horizons in result[
                    "recursive_comparison"][split].items()}
            for split in ("openplane_validation", "unseen_practice")},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
