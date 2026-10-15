#!/usr/bin/env python3
"""Isolate the previously trained rollout-conditioned v head.

The DAgger bundle changed every transition output and hurt some of them. This
evaluation substitutes only its v-increment head into the matched parent,
leaving u/a_x, yaw, actuator, filtered-wheel, and raw-wheel models untouched.
Only whole-run validation and existing practice captures are evaluated.
"""

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
    "lateral_dagger_head_isolation_20261009")
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
    cand = candidate
    base = baseline
    if cand[horizon]["window_signature_sha256"] != base[horizon][
            "window_signature_sha256"]:
        raise ValueError("candidate and parent did not use identical windows")
    run_rows = cand[horizon]["per_run"]
    base_rows = base[horizon]["per_run"]
    common = sorted(set(run_rows) & set(base_rows))
    delta = np.asarray([
        run_rows[run][horizon]["candidate"][metric]
        - base_rows[run][horizon]["candidate"][metric]
        for run in common], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    macro_candidate = float(np.mean([
        run_rows[run][horizon]["candidate"][metric] for run in common]))
    macro_baseline = float(np.mean([
        base_rows[run][horizon]["candidate"][metric] for run in common]))
    return {
        "runs": len(common),
        "parent_macro_run_rmse": macro_baseline,
        "isolated_v_head_macro_run_rmse": macro_candidate,
        "mean_candidate_minus_parent": float(delta.mean()),
        "paired_run_bootstrap_95pct": [
            float(value) for value in np.quantile(draws, (0.025, 0.975))],
    }


def run(dataset: Path, practice: Path, parent_path: Path,
        dagger_path: Path, output_dir: Path) -> dict[str, Any]:
    for path in (dataset, practice, parent_path, dagger_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = joblib.load(parent_path)
    dagger = joblib.load(dagger_path)
    if sha256(dataset) != parent["training_dataset_sha256"]:
        raise ValueError("parent candidate does not match declared training data")
    if dagger.get("training_dataset_sha256") != parent.get(
            "training_dataset_sha256"):
        raise ValueError("DAgger and parent were not trained on the same captures")
    if (parent["selected_parent_feature_variant"]
            != dagger["selected_parent_feature_variant"]):
        raise ValueError("candidate feature variants differ")
    v_parent = parent["recursive_aux_models"][0]
    v_dagger = dagger["recursive_aux_models"][0]
    if v_parent.n_features_in_ != v_dagger.n_features_in_:
        raise ValueError("v-head feature widths differ")

    candidate_path = output_dir / "parent_with_isolated_dagger_v_head.joblib"
    parent_report_path = output_dir / "parent_recursive.json"
    candidate_report_path = output_dir / "isolated_v_head_recursive.json"
    candidate = dict(parent)
    candidate["recursive_aux_models"] = list(parent["recursive_aux_models"])
    candidate["recursive_aux_models"][0] = v_dagger
    candidate["isolated_rollout_conditioned_v_head"] = {
        "source_bundle": str(dagger_path.relative_to(ROOT)),
        "source_bundle_sha256": sha256(dagger_path),
        "other_heads_changed": False,
    }
    reused_existing_evaluations = False
    if output_dir.exists():
        required = (candidate_path, parent_report_path, candidate_report_path)
        if not all(path.is_file() for path in required):
            raise FileExistsError(
                f"existing output directory is incomplete or unrelated: {output_dir}")
        parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
        candidate_report = json.loads(
            candidate_report_path.read_text(encoding="utf-8"))
        if (candidate_report.get("candidate_bundle_sha256")
                != sha256(candidate_path)
                or parent_report.get("candidate_bundle_sha256")
                != sha256(parent_path)):
            raise ValueError("existing recursive reports do not match their bundles")
        reused_existing_evaluations = True
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(candidate, candidate_path)
        parent_report = evaluate(
            parent_path, dataset, practice, parent_report_path,
            24, None, "increment", True, "predicted", None)
        candidate_report = evaluate(
            candidate_path, dataset, practice, candidate_report_path,
            24, None, "increment", True, "predicted", None)

    comparison: dict[str, Any] = {}
    for split in ("openplane_validation", "unseen_practice"):
        comparison[split] = {}
        for horizon in HORIZONS:
            comparison[split][horizon] = {
                metric: _paired_delta(
                    candidate_report["horizons"][split],
                    parent_report["horizons"][split], horizon, metric)
                for metric in METRICS
            }
    report = {
        "schema_version": 1,
        "purpose": "isolate and assess the DAgger-trained lateral-velocity increment head",
        "parent_bundle": str(parent_path.relative_to(ROOT)),
        "parent_bundle_sha256": sha256(parent_path),
        "dagger_source_bundle": str(dagger_path.relative_to(ROOT)),
        "dagger_source_bundle_sha256": sha256(dagger_path),
        "candidate_bundle": str(candidate_path.relative_to(ROOT)),
        "candidate_bundle_sha256": sha256(candidate_path),
        "dataset_sha256": sha256(dataset),
        "practice_dataset_sha256": sha256(practice),
        "test_and_final_test_used": False,
        "only_v_head_changed": True,
        "reused_completed_recursive_reports": reused_existing_evaluations,
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
    args = parser.parse_args()
    report = run(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.parent_bundle.resolve(), args.dagger_bundle.resolve(),
                 args.output_dir.resolve())
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
