#!/usr/bin/env python3
"""Isolate rollout-conditioned u, or u+v, heads in the parent plant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    evaluate,
)
from tools.vehicle_dynamics_learning.fit_recursive_dagger_longitudinal_candidate import (
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/"
    "encoder_fixed40hz_v1/openplane_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_PRACTICE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/"
    "encoder_fixed40hz_v1/practice_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_PARENT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
DEFAULT_DAGGER = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_recursive_dagger_candidate_20261009_v2/"
    "dagger_longitudinal_candidate.joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_dagger_u_head_isolation_20261009")
METRICS = (
    "u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
    "steering_feedback_rmse_rad", "throttle_feedback_rmse_norm",
    "rear_wheel_pair_rmse_mps", "raw_encoder_pair_rmse_mps",
    "position_rmse_m", "yaw_pose_rmse_rad",
)
HORIZONS = ("0.025s", "0.100s", "0.250s", "0.500s", "1.000s",
            "2.000s", "3.000s", "5.000s")
SEED = 20261009


def _paired_delta(candidate: dict[str, Any], baseline: dict[str, Any],
                  horizon: str, metric: str) -> dict[str, Any]:
    if candidate[horizon]["window_signature_sha256"] != baseline[horizon][
            "window_signature_sha256"]:
        raise ValueError("candidate and parent did not use identical windows")
    c_runs, b_runs = candidate[horizon]["per_run"], baseline[horizon]["per_run"]
    common = sorted(set(c_runs) & set(b_runs))
    delta = np.asarray([
        c_runs[run][horizon]["candidate"][metric]
        - b_runs[run][horizon]["candidate"][metric] for run in common])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    return {
        "runs": len(common),
        "parent_macro_run_rmse": float(np.mean([
            b_runs[run][horizon]["candidate"][metric] for run in common])),
        "isolated_u_head_macro_run_rmse": float(np.mean([
            c_runs[run][horizon]["candidate"][metric] for run in common])),
        "mean_candidate_minus_parent": float(delta.mean()),
        "paired_run_bootstrap_95pct": [
            float(x) for x in np.quantile(draws, (0.025, 0.975))],
    }


def run(dataset: Path, practice: Path, parent_path: Path,
        dagger_path: Path, output_dir: Path,
        include_v: bool = False) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    for path in (dataset, practice, parent_path, dagger_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent, dagger = joblib.load(parent_path), joblib.load(dagger_path)
    if sha256(dataset) != parent["training_dataset_sha256"]:
        raise ValueError("parent candidate does not match declared training data")
    if dagger.get("training_dataset_sha256") != parent.get(
            "training_dataset_sha256"):
        raise ValueError("candidates use different training captures")
    if parent["selected_parent_feature_variant"] != dagger[
            "selected_parent_feature_variant"]:
        raise ValueError("candidate feature variants differ")
    if parent["models"][0].n_features_in_ != dagger["models"][0].n_features_in_:
        raise ValueError("u-head feature widths differ")
    if include_v and (parent["recursive_aux_models"][0].n_features_in_
                      != dagger["recursive_aux_models"][0].n_features_in_):
        raise ValueError("v-head feature widths differ")

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = dict(parent)
    candidate["models"] = list(parent["models"])
    candidate["models"][0] = dagger["models"][0]
    candidate["isolated_rollout_conditioned_heads"] = {
        "u": True,
        "v": bool(include_v),
        "source_bundle": str(dagger_path.relative_to(ROOT)),
        "source_bundle_sha256": sha256(dagger_path),
        "other_heads_changed": False,
    }
    if include_v:
        candidate["recursive_aux_models"] = list(parent["recursive_aux_models"])
        candidate["recursive_aux_models"][0] = dagger["recursive_aux_models"][0]
    head_name = "u_v" if include_v else "u"
    candidate_path = output_dir / f"parent_with_isolated_dagger_{head_name}_heads.joblib"
    joblib.dump(candidate, candidate_path)
    parent_report = evaluate(
        parent_path, dataset, practice, output_dir / "parent_recursive.json",
        24, None, "increment", True, "predicted", None)
    candidate_report = evaluate(
        candidate_path, dataset, practice,
        output_dir / f"isolated_{head_name}_heads_recursive.json",
        24, None, "increment", True, "predicted", None)

    comparison = {
        split: {
            horizon: {
                metric: _paired_delta(
                    candidate_report["horizons"][split],
                    parent_report["horizons"][split], horizon, metric)
                for metric in METRICS
            }
            for horizon in HORIZONS
        }
        for split in ("openplane_validation", "unseen_practice")
    }
    report = {
        "schema_version": 1,
        "purpose": ("isolate rollout-conditioned u" + (" and v" if include_v else "")
                    + " increment heads without changing other vehicle-state heads"),
        "parent_bundle": str(parent_path.relative_to(ROOT)),
        "parent_bundle_sha256": sha256(parent_path),
        "dagger_source_bundle": str(dagger_path.relative_to(ROOT)),
        "dagger_source_bundle_sha256": sha256(dagger_path),
        "candidate_bundle": str(candidate_path.relative_to(ROOT)),
        "candidate_bundle_sha256": sha256(candidate_path),
        "dataset_sha256": sha256(dataset),
        "practice_dataset_sha256": sha256(practice),
        "test_and_final_test_used": False,
        "isolated_heads_changed": ["u"] + (["v"] if include_v else []),
        "validation_and_practice_used_for_fitting_or_selection": False,
        "matched_whole_run_recursive_comparison": comparison,
        "no_simulator_or_runtime_changes": True,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--parent-bundle", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--dagger-bundle", type=Path, default=DEFAULT_DAGGER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-v", action="store_true",
                        help="also replace the lateral-velocity increment head")
    args = parser.parse_args()
    report = run(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.parent_bundle.resolve(), args.dagger_bundle.resolve(),
                 args.output_dir.resolve(), args.include_v)
    print(json.dumps({
        split: {
            horizon: {metric: values["mean_candidate_minus_parent"]
                      for metric, values in metrics.items()}
            for horizon, metrics in horizons.items()
            if horizon in ("0.250s", "1.000s", "5.000s")}
        for split, horizons in report[
            "matched_whole_run_recursive_comparison"].items()
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
