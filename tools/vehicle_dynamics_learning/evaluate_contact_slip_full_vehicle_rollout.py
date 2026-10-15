#!/usr/bin/env python3
"""Factorial whole-run test of contact-relative wheel-spin in the plant rollout.

Compare absolute-wheel versus contact-relative wheel-state recursion with the
same body model, and repeat with the rollout-conditioned body candidate. All
four branches use identical windows, initial history, and recorded command
replay. No model is fitted and test/final-test splits are never selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _bundle_lags,
    _run_rollout,
    _select_windows,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.fit_recursive_dagger_longitudinal_candidate import (
    _filter_rollout_windows,
    sha256,
)
from tools.vehicle_dynamics_learning.fit_contact_relative_wheel_slip_model import (
    DEFAULT_OUTPUT_DIR as SLIP_OUTPUT_DIR,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
DEFAULT_DAGGER_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_recursive_dagger_candidate_20261009_v2/"
    "dagger_longitudinal_candidate.joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_vehicle_contact_slip_factorial_20261009.json")
HORIZONS = (1, 10, 40, 80, 120, 200)
METRICS = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
           "position_rmse_m", "rear_wheel_pair_rmse_mps",
           "raw_encoder_pair_rmse_mps")


def _paired_run_bootstrap(candidate: dict[str, Any], baseline: dict[str, Any],
                          metric: str, seed: int) -> dict[str, Any]:
    common = sorted(set(candidate) & set(baseline))
    differences = np.asarray([candidate[run][metric] - baseline[run][metric]
                              for run in common], dtype=np.float64)
    if not len(differences):
        return {"runs": 0, "mean_delta": None, "95pct": None}
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(10000, len(differences)), replace=True)
    means = draws.mean(axis=1)
    return {"runs": len(common), "mean_delta": float(differences.mean()),
            "95pct": [float(v) for v in np.quantile(means, (.025, .975))]}


def _eligible_windows(data: dict[str, Any], split: str, lags: tuple[int, ...]
                      ) -> tuple[list[dict[str, int]], dict[str, Any]]:
    windows = _select_windows(data, split, horizon=max(HORIZONS),
                              max_windows_per_run=24,
                              initial_context_steps=max(lags))
    windows, filters = _filter_rollout_windows(
        data, windows, max(lags), horizon=max(HORIZONS))
    return windows, filters


def _run_split(data: dict[str, Any], split: str,
               base_bundle: dict[str, Any], dagger_bundle: dict[str, Any],
               slip_models: list[Any], rear_track_m: float
               ) -> dict[str, Any]:
    lags = _bundle_lags(base_bundle)
    windows, window_filters = _eligible_windows(data, split, lags)
    if not windows:
        raise ValueError(f"no eligible whole-run windows in split {split}")
    branches = {
        "baseline_body_absolute_wheel": (base_bundle, "predicted"),
        "dagger_body_absolute_wheel": (dagger_bundle, "predicted"),
    }
    isolated_u = dict(base_bundle)
    isolated_u["models"] = list(base_bundle["models"])
    isolated_u["models"][0] = dagger_bundle["models"][0]
    branches["isolated_dagger_u_absolute_wheel"] = (isolated_u, "predicted")
    slip_base = dict(base_bundle)
    slip_base["contact_slip_models"] = slip_models
    slip_base["contact_slip_rear_track_m"] = rear_track_m
    slip_dagger = dict(dagger_bundle)
    slip_dagger["contact_slip_models"] = slip_models
    slip_dagger["contact_slip_rear_track_m"] = rear_track_m
    slip_isolated_u = dict(isolated_u)
    slip_isolated_u["contact_slip_models"] = slip_models
    slip_isolated_u["contact_slip_rear_track_m"] = rear_track_m
    branches["baseline_body_contact_relative_wheel"] = (
        slip_base, "predicted_contact_slip")
    branches["dagger_body_contact_relative_wheel"] = (
        slip_dagger, "predicted_contact_slip")
    branches["isolated_dagger_u_contact_relative_wheel"] = (
        slip_isolated_u, "predicted_contact_slip")

    runs = {}
    for name, (bundle, mode) in branches.items():
        runs[name] = {}
        for horizon in HORIZONS:
            result = _run_rollout(
                data, bundle, windows, horizon, lags,
                u_update="increment", raw_input_mode=mode)
            label = f"{horizon * .025:.3f}s"
            macro = result["macro_run_metrics"][label]["candidate_macro_run_rmse"]
            per_run = {
                run_id: {
                    key: value[label]["candidate"].get(key)
                    for key in METRICS
                }
                for run_id, value in result["per_run"].items()
            }
            runs[name][label] = {"macro_run_rmse": macro, "per_run": per_run}

    comparisons = {}
    reference = "baseline_body_absolute_wheel"
    for horizon in HORIZONS:
        label = f"{horizon * .025:.3f}s"
        comparisons[label] = {}
        ref_per_run = runs[reference][label]["per_run"]
        for name, values in runs.items():
            if name == reference:
                continue
            candidate_per_run = values[label]["per_run"]
            comparisons[label][name] = {
                metric: _paired_run_bootstrap(
                    {run: row for run, row in candidate_per_run.items()
                     if row.get(metric) is not None},
                    {run: row for run, row in ref_per_run.items()
                     if row.get(metric) is not None},
                    metric, seed=20261009 + horizon)
                for metric in METRICS
            }
    return {
        "split": split,
        "window_count": len(windows),
        "window_filter": window_filters,
        "branches": runs,
        "paired_run_deltas_vs_baseline_body_absolute_wheel": comparisons,
    }


def evaluate(dataset_path: Path, practice_path: Path, base_path: Path,
             dagger_path: Path, slip_path: Path, output_path: Path
             ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    base_bundle = joblib.load(base_path)
    dagger_bundle = joblib.load(dagger_path)
    slip_bundle = joblib.load(slip_path)
    if not (base_bundle.get("research_only") and dagger_bundle.get("research_only")
            and slip_bundle.get("research_only")):
        raise ValueError("all inputs must be frozen research-only bundles")
    if len(slip_bundle.get("models", ())) != 2:
        raise ValueError("contact-relative candidate must contain two wheel heads")
    if base_bundle.get("training_dataset_sha256") != sha256(dataset_path):
        raise ValueError("body candidate does not match the specified training data")
    if dagger_bundle.get("training_dataset_sha256") != sha256(dataset_path):
        raise ValueError("rollout candidate does not match the specified training data")
    if slip_bundle.get("training_dataset_sha256") != sha256(dataset_path):
        raise ValueError("slip candidate does not match the specified training data")

    hybrid_path = output_path.with_name(
        "isolated_dagger_u_contact_slip_candidate_20261009.joblib")
    if hybrid_path.exists():
        raise FileExistsError(f"refusing to overwrite hybrid candidate: {hybrid_path}")
    hybrid = dict(base_bundle)
    hybrid["models"] = list(base_bundle["models"])
    hybrid["models"][0] = dagger_bundle["models"][0]
    hybrid["contact_slip_models"] = list(slip_bundle["models"])
    hybrid["contact_slip_rear_track_m"] = float(slip_bundle["rear_track_m"])
    hybrid["hybrid_candidate_components"] = {
        "body_u_increment_head_from": str(dagger_path.relative_to(ROOT)),
        "body_u_increment_head_sha256": sha256(dagger_path),
        "other_body_heads_from": str(base_path.relative_to(ROOT)),
        "other_body_heads_sha256": sha256(base_path),
        "contact_relative_wheel_heads_from": str(slip_path.relative_to(ROOT)),
        "contact_relative_wheel_heads_sha256": sha256(slip_path),
        "only_body_u_head_replaced": True,
        "physics_or_runtime_changed": False,
    }
    joblib.dump(hybrid, hybrid_path)
    digest = hashlib.sha256()
    with hybrid_path.open("rb") as candidate_stream:
        for block in iter(lambda: candidate_stream.read(1 << 20), b""):
            digest.update(block)

    val_data = _load_dataset(dataset_path)
    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str))
                      else "validation")
    rear_track = float(slip_bundle["rear_track_m"])
    report = {
        "schema_version": 1,
        "purpose": "test whether learned contact-relative wheel spin survives full predicted-body recursion",
        "training_dataset_sha256": sha256(dataset_path),
        "practice_dataset_sha256": sha256(practice_path),
        "body_baseline_bundle": str(base_path.relative_to(ROOT)),
        "body_rollout_candidate_bundle": str(dagger_path.relative_to(ROOT)),
        "isolated_dagger_u_head_source": str(dagger_path.relative_to(ROOT)),
        "contact_relative_slip_bundle": str(slip_path.relative_to(ROOT)),
        "body_baseline_bundle_sha256": sha256(base_path),
        "body_rollout_candidate_bundle_sha256": sha256(dagger_path),
        "contact_relative_slip_bundle_sha256": sha256(slip_path),
        "hybrid_research_candidate_bundle": str(hybrid_path.relative_to(ROOT)),
        "hybrid_research_candidate_bundle_sha256": digest.hexdigest(),
        "heldout_policy": {
            "same_windows_for_all_branches": True,
            "same_start_state_and_command_replay": True,
            "free_running_body_and_wheel_states": True,
            "future_truth_sensor_inputs": False,
            "test_and_final_test_used": False,
            "practice_used_for_selection": False,
        },
        "openplane_validation": _run_split(
            val_data, "validation", base_bundle, dagger_bundle,
            slip_bundle["models"], rear_track),
        "unseen_practice_diagnostic": _run_split(
            practice_data, practice_split, base_bundle, dagger_bundle,
            slip_bundle["models"], rear_track),
        "no_simulator_or_runtime_changes": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--base-bundle", type=Path, default=DEFAULT_BASE_BUNDLE)
    parser.add_argument("--dagger-bundle", type=Path, default=DEFAULT_DAGGER_BUNDLE)
    parser.add_argument("--slip-bundle", type=Path,
                        default=SLIP_OUTPUT_DIR / "contact_relative_wheel_slip_candidate.joblib")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(
        args.dataset.resolve(), args.practice_dataset.resolve(),
        args.base_bundle.resolve(), args.dagger_bundle.resolve(),
        args.slip_bundle.resolve(), args.output.resolve())
    print(json.dumps({
        split_name: {
            horizon: {
                branch: round(
                    branch_metrics[horizon]["macro_run_rmse"].get(
                        "raw_encoder_pair_rmse_mps", float("nan")), 4)
                for branch, branch_metrics in split_report["branches"].items()
            }
            for horizon in split_report["branches"][
                "baseline_body_absolute_wheel"]
        }
        for split_name, split_report in (
            ("validation", report["openplane_validation"]),
            ("practice", report["unseen_practice_diagnostic"]))
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
