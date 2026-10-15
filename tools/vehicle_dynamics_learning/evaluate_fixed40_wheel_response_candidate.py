#!/usr/bin/env python3
"""Report one-step fixed-encoder wheel-model accuracy by operating condition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
    DEFAULT_OUTPUT,
    VARIANT_NAMES,
    _prepare,
    _score,
    _sha256,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]


def _macro_slice(candidate: np.ndarray, baseline: np.ndarray,
                 target: np.ndarray, run_index: np.ndarray,
                 run_ids: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"transitions": 0, "independent_runs": 0}
    run_scores: dict[str, dict[str, float]] = {}
    for run in sorted(np.unique(run_index[mask]).tolist()):
        rows = mask & (run_index == run)
        err = candidate[rows] - target[rows]
        base_err = baseline[rows] - target[rows]
        run_scores[str(run_ids[run])] = {
            "candidate_pair_rmse_mps": float(np.sqrt(np.mean(err ** 2))),
            "persistence_pair_rmse_mps": float(np.sqrt(np.mean(base_err ** 2))),
            "candidate_left_rmse_mps": float(np.sqrt(np.mean(err[:, 0] ** 2))),
            "candidate_right_rmse_mps": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        }
    differences = np.asarray([
        item["candidate_pair_rmse_mps"] - item["persistence_pair_rmse_mps"]
        for item in run_scores.values()], dtype=np.float64)
    rng = np.random.default_rng(20261009 + len(differences))
    samples = differences[rng.integers(
        0, len(differences), (10000, len(differences)))].mean(axis=1)
    return {
        "transitions": int(mask.sum()),
        "independent_runs": len(run_scores),
        "weak_evidence_under_three_runs": len(run_scores) < 3,
        "candidate_macro_run_pair_rmse_mps": float(np.mean([
            row["candidate_pair_rmse_mps"] for row in run_scores.values()])),
        "persistence_macro_run_pair_rmse_mps": float(np.mean([
            row["persistence_pair_rmse_mps"] for row in run_scores.values()])),
        "candidate_minus_persistence_macro_run_bootstrap_95pct_mps": [
            float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "runs_better_than_persistence": int(np.sum(differences < 0.0)),
        "per_run": run_scores,
    }


def _slices(data: dict[str, Any], source: np.ndarray,
            prediction: np.ndarray, baseline: np.ndarray, target: np.ndarray,
            run_index: np.ndarray, run_ids: np.ndarray) -> dict[str, Any]:
    state = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    speed = np.hypot(state[source, 7], state[source, 8])
    abs_steering = np.abs(frames[source, 3])
    throttle = frames[source, 8]
    throttle_slew = np.abs(frames[source, 8] - frames[source - 1, 8]) / 0.025
    wheel_body_mismatch = np.abs(
        np.mean(frames[source, 5:7], axis=1) - state[source, 7])

    def bins(values: np.ndarray, edges: tuple[float, ...], labels: list[str]
             ) -> dict[str, Any]:
        indices = np.digitize(values, edges[1:-1], right=False)
        return {label: _macro_slice(
            prediction, baseline, target, run_index, run_ids, indices == i)
            for i, label in enumerate(labels)}

    def quantile_bins(values: np.ndarray, fractions: tuple[float, ...]
                      ) -> dict[str, Any]:
        edges = np.unique(np.quantile(values, fractions))
        if len(edges) < 2:
            return {"all": _macro_slice(
                prediction, baseline, target, run_index, run_ids,
                np.ones(len(values), dtype=bool))}
        indices = np.digitize(values, edges[1:-1], right=False)
        result: dict[str, Any] = {}
        for i in range(len(edges) - 1):
            result[f"q{i+1}"] = {
                "value_range": [float(edges[i]), float(edges[i + 1])],
                **_macro_slice(prediction, baseline, target, run_index,
                               run_ids, indices == i),
            }
        return result

    return {
        "interpretation": "descriptive slices; boundaries are reporting bins, not fitted physical regimes",
        "speed_magnitude_mps": bins(
            speed, (0, 2, 4, 6, 8, 10, 12.000001),
            ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]),
        "absolute_steering_rad": bins(
            abs_steering, (0, .1, .2, .3, .4, .5, np.inf),
            ["0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ">=.5"]),
        "signed_steering_rad": bins(
            frames[source, 3], (-np.inf, -.3, -.1, .1, .3, np.inf),
            ["<=-.3", "-.3--.1", "-.1-.1", ".1-.3", ">=.3"]),
        "throttle_command": bins(
            throttle, (-np.inf, .05, .2, .4, .6, .8, np.inf),
            ["<.05", ".05-.2", ".2-.4", ".4-.6", ".6-.8", ">=.8"]),
        "absolute_throttle_slew_per_s_quantiles": quantile_bins(
            throttle_slew, (0, .5, .9, .99, 1)),
        "absolute_filtered_wheel_minus_body_speed_quantiles": quantile_bins(
            wheel_body_mismatch, (0, .25, .5, .75, .9, 1)),
    }


def _evaluate_split(path: Path, split: str, bundle: dict[str, Any]
                    ) -> dict[str, Any]:
    data = _load_dataset(path)
    source, run_index, baseline, target, features, _, _ = _prepare(data, split)
    variant = str(bundle["feature_variant"])
    index = VARIANT_NAMES.index(variant)
    prediction = baseline + np.column_stack([
        model.predict(features[index]) for model in bundle["models"]])
    run_ids = np.asarray(data["run_ids"]).astype(str)
    result = _score(prediction, target, baseline, run_index, run_ids)
    result["stratified_diagnostics"] = _slices(
        data, source, prediction, baseline, target, run_index, run_ids)
    result["scored_split"] = split
    result["valid_transitions"] = int(len(source))
    return result


def evaluate(bundle_path: Path, dataset_path: Path, practice_path: Path,
             output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite wheel-response evaluation: {output_path}")
    bundle = joblib.load(bundle_path)
    if not bundle.get("research_only") or not bundle.get("models"):
        raise ValueError("not a research fixed-40 wheel candidate")
    if _sha256(dataset_path) != bundle["training_dataset_sha256"]:
        raise ValueError("OpenPlane dataset differs from the candidate's training data")
    openplane = _evaluate_split(dataset_path, "validation", bundle)
    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str)) else "validation")
    practice = _evaluate_split(practice_path, practice_split, bundle)
    report = {
        "schema_version": 1,
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_bundle_sha256": _sha256(bundle_path),
        "openplane_dataset_sha256": _sha256(dataset_path),
        "practice_dataset_sha256": _sha256(practice_path),
        "target": bundle["target"],
        "test_and_final_test_policy": "sealed; neither split is scored or used",
        "one_step_results": {
            "openplane_validation": openplane,
            "unseen_practice": practice,
        },
        "limitations": [
            "one-step state uses simulator truth as the current body-state input",
            "encoder-derived rate is a measured fixed-period proxy, not true internal tire angular velocity",
            "unseen practice evidence has only two independent runs",
            "no odometry/MPC integration and no free-running complete plant result",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_OUTPUT / "fixed40_wheel_response_candidate.joblib")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "regime_validation.json")
    args = parser.parse_args()
    report = evaluate(args.bundle.resolve(), args.dataset.resolve(),
                      args.practice_dataset.resolve(), args.output.resolve())
    compact = {}
    for split, value in report["one_step_results"].items():
        compact[split] = {
            "macro_run_metrics": value["macro_run_metrics"],
            "speed_magnitude_mps": {
                label: {key: row.get(key) for key in (
                    "transitions", "independent_runs",
                    "candidate_macro_run_pair_rmse_mps",
                    "persistence_macro_run_pair_rmse_mps",
                    "weak_evidence_under_three_runs")}
                for label, row in value["stratified_diagnostics"][
                    "speed_magnitude_mps"].items()},
            "absolute_steering_rad": {
                label: {key: row.get(key) for key in (
                    "transitions", "independent_runs",
                    "candidate_macro_run_pair_rmse_mps",
                    "persistence_macro_run_pair_rmse_mps",
                    "weak_evidence_under_three_runs")}
                for label, row in value["stratified_diagnostics"][
                    "absolute_steering_rad"].items()},
        }
    print(json.dumps({"output": str(args.output), "results": compact}, indent=2))


if __name__ == "__main__":
    main()
