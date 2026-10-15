#!/usr/bin/env python3
"""Fit a full-regime recursive residual around the frozen yaw atlas.

The atlas remains the one-step yaw teacher. A run-balanced residual is trained
from causal plant rollouts on training captures spanning the available speed
and steering domain. Validation and practice captures are evaluated only
after fitting; the test/final-test partitions are never selected or scored.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as evaluator
from tools.vehicle_dynamics_learning.evaluate_dagger_v_yaw_atlas_rollouts import (
    HORIZON_BY_COHORT,
)
from tools.vehicle_dynamics_learning.evaluate_extended_yaw_atlas_plant_rollouts import (
    MAX_WINDOWS_PER_RUN,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    HORIZON,
    _run_windows,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HIGHSTEER_TRAIN_DATASET,
    ITERATIONS,
    _fit,
    _sha256,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as yaw_atlas


OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_regime_yaw_atlas_dagger_20261011")
MAX_TRAIN_WINDOWS_PER_RUN = 16
SEED = 20261011
REPORTING_HORIZONS = (30, 80, 200, 400, 480)
PAIRED_METRICS = (
    "position_rmse_m", "yaw_pose_rmse_rad", "u_rmse_mps", "v_rmse_mps",
    "yaw_rate_rmse_rps", "filtered_rear_wheel_pair_rmse_mps")


def _training_source(path: Path, name: str, hgb: dict[str, Any]
                     ) -> dict[str, Any]:
    data = _load_dataset(path)
    splits = np.asarray(data["splits"]).astype(str)
    windows = _run_windows(
        data, "train", evaluator._bundle_lags(hgb),
        MAX_TRAIN_WINDOWS_PER_RUN, horizon_steps=HORIZON)
    selected_runs = {str(data["run_ids"][row["run"]]) for row in windows}
    if not selected_runs or any(splits[row["run"]] != "train" for row in windows):
        raise ValueError(f"training source {name} selected a non-training run")
    return {"name": name, "path": path, "data": data, "windows": windows,
            "run_ids": selected_runs}


def _support_summary(sources: list[dict[str, Any]]) -> dict[str, Any]:
    per_source: dict[str, Any] = {}
    all_speed: list[np.ndarray] = []
    all_steering: list[np.ndarray] = []
    all_run_ids: set[str] = set()
    for source in sources:
        data = source["data"]
        frames = np.asarray(data["frames"], dtype=np.float64)
        indices = np.concatenate([
            np.arange(row["start"], row["start"] + HORIZON, dtype=np.int64)
            for row in source["windows"]])
        speed = np.linalg.norm(frames[indices, :2], axis=1)
        steering = np.abs(frames[indices, 3])
        all_speed.append(speed)
        all_steering.append(steering)
        all_run_ids.update(source["run_ids"])
        per_source[source["name"]] = {
            "run_ids": sorted(source["run_ids"]),
            "windows": len(source["windows"]),
            "transitions": int(len(indices)),
            "body_speed_mps_min_median_p95_max": [
                float(value) for value in np.quantile(
                    speed, (0.0, 0.5, 0.95, 1.0))],
            "absolute_steering_rad_min_median_p95_max": [
                float(value) for value in np.quantile(
                    steering, (0.0, 0.5, 0.95, 1.0))],
        }
    speed = np.concatenate(all_speed)
    steering = np.concatenate(all_steering)
    speed_cells = np.minimum(np.floor(speed).astype(np.int16), 11)
    steering_cells = np.minimum(np.floor(steering / 0.05).astype(np.int16), 10)
    counts = np.zeros((12, 11), dtype=np.int64)
    np.add.at(counts, (speed_cells, steering_cells), 1)
    return {
        "per_source": per_source,
        "unique_training_runs": sorted(all_run_ids),
        "transitions": int(len(speed)),
        "body_speed_mps_min_median_p95_max": [
            float(value) for value in np.quantile(speed, (0.0, 0.5, 0.95, 1.0))],
        "absolute_steering_rad_min_median_p95_max": [
            float(value) for value in np.quantile(
                steering, (0.0, 0.5, 0.95, 1.0))],
        "coarse_joint_coverage": {
            "speed_bins_mps": "[0,1), ... [11,12]",
            "absolute_steering_bins_rad": "[0,.05), ... [.50,.55]",
            "cells_with_at_least_200_training_transitions": int(
                np.count_nonzero(counts >= 200)),
            "cells_total": int(counts.size),
            "cell_counts": counts.tolist(),
        },
    }


def _capture_training_sources(
        sources: list[dict[str, Any]], hgb: dict[str, Any], edssm: Any,
        metadata: dict[str, Any], atlas: dict[str, Any],
        combined_run_ids: np.ndarray,
        correction: Any | None) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray, list[dict[str, Any]]]:
    combined_index = {str(run_id): index
                      for index, run_id in enumerate(combined_run_ids)}
    feature_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for source in sources:
        result = evaluator._rollout(
            source["data"], hgb, edssm, metadata, atlas,
            source["windows"], HORIZON,
            yaw_delta_gain=1.0,
            include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            atlas_acceleration_source="composed_transition",
            yaw_correction_model=correction,
            yaw_correction_start_step=CORRECTION_START_STEP,
            capture_yaw_correction_rows=True,
            lateral_v_update="edssm")
        features = result.pop("_yaw_correction_training_features")
        residuals = result.pop("_yaw_correction_training_residuals")
        local_runs = result.pop("_yaw_correction_training_run_indices")
        local_ids = np.asarray(source["data"]["run_ids"]).astype(str)
        row_ids = local_ids[local_runs]
        feature_parts.append(features)
        residual_parts.append(residuals)
        run_parts.append(np.asarray(
            [combined_index[str(run_id)] for run_id in row_ids], dtype=np.int32))
        reports.append({
            "source": source["name"],
            "windows": len(source["windows"]),
            "training_run_ids": sorted(source["run_ids"]),
            "captured_rows": int(len(features)),
            "residual_rmse_rps": float(np.sqrt(np.mean(residuals ** 2))),
        })
    return (np.concatenate(feature_parts), np.concatenate(residual_parts),
            np.concatenate(run_parts), reports)


def _paired_run_deltas(candidate: dict[str, Any], baseline: dict[str, Any],
                       horizon: str) -> dict[str, Any]:
    candidate_runs = candidate["per_run"]
    baseline_runs = baseline["per_run"]
    run_ids = sorted(set(candidate_runs) & set(baseline_runs))
    rng = np.random.default_rng(SEED + int(round(float(horizon[:-1]) * 1000)))
    output: dict[str, Any] = {"run_ids": run_ids}
    for metric in PAIRED_METRICS:
        delta = np.asarray([
            candidate_runs[run][horizon]["hybrid_atlas"][metric]
            - baseline_runs[run][horizon]["hybrid_atlas"][metric]
            for run in run_ids], dtype=np.float64)
        draws = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(1)
        output[metric] = {
            "mean_candidate_minus_baseline": float(delta.mean()),
            "paired_run_bootstrap_95pct": [
                float(value) for value in np.quantile(draws, (0.025, 0.975))],
            "per_run_delta": {
                run: float(value) for run, value in zip(run_ids, delta)},
        }
    return output


def _score(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
           metadata: dict[str, Any], atlas: dict[str, Any],
           windows: list[dict[str, int]], horizon: int,
           correction: Any | None) -> dict[str, Any]:
    evaluator.HORIZONS = tuple(
        step for step in REPORTING_HORIZONS if step <= horizon)
    return evaluator._rollout(
        data, hgb, edssm, metadata, atlas, windows, horizon,
        yaw_delta_gain=1.0,
        include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source="composed_transition",
        yaw_correction_model=correction,
        yaw_correction_start_step=CORRECTION_START_STEP,
        lateral_v_update="edssm")


def _validation_sources(hgb: dict[str, Any]) -> list[dict[str, Any]]:
    specs = (
        ("broad_openplane_validation", evaluator.HGB_TRAIN_DATASET,
         "validation", False),
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET,
         "validation", True),
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", False),
    )
    sources = []
    for name, path, split, highsteer in specs:
        data = _load_dataset(path)
        horizon = HORIZON_BY_COHORT[name]
        if highsteer:
            windows = evaluator._windows(
                data, path, horizon, evaluator._bundle_lags(hgb))
        else:
            windows = _run_windows(
                data, split, evaluator._bundle_lags(hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon)
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if row["start"] + horizon < bounds[row["sequence"], 1]]
        selected_ids = sorted({
            str(data["run_ids"][row["run"]]) for row in windows})
        splits = np.asarray(data["splits"]).astype(str)
        if (not windows or any(splits[row["run"]] != split for row in windows)):
            raise ValueError(f"invalid validation selection for {name}")
        sources.append({"name": name, "path": path, "data": data,
                        "windows": windows, "horizon": horizon,
                        "split": split, "run_ids": selected_ids})
    return sources


def run() -> dict[str, Any]:
    report_path = OUTPUT_DIR / "evaluation.json"
    model_path = OUTPUT_DIR / "full_regime_yaw_residual.joblib"
    if report_path.exists() or model_path.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    required = (
        evaluator.HGB_BUNDLE, evaluator.EDSSM_CHECKPOINT,
        evaluator.YAW_ATLAS_BUNDLE, evaluator.HGB_TRAIN_DATASET,
        HIGHSTEER_TRAIN_DATASET, evaluator.HIGHSTEER_DATASET,
        evaluator.PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    hgb = joblib.load(evaluator.HGB_BUNDLE)
    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")

    sources = [
        _training_source(evaluator.HGB_TRAIN_DATASET,
                         "broad_train_capture", hgb),
        _training_source(HIGHSTEER_TRAIN_DATASET,
                         "highsteer_train_capture", hgb),
    ]
    run_owner: dict[str, str] = {}
    for source in sources:
        for run_id in source["run_ids"]:
            if run_id in run_owner:
                raise ValueError(
                    f"training run {run_id!r} is duplicated in "
                    f"{run_owner[run_id]} and {source['name']}")
            run_owner[run_id] = source["name"]
    training_run_ids = np.asarray(sorted(run_owner), dtype=str)
    support = _support_summary(sources)
    if not (support["body_speed_mps_min_median_p95_max"][0] <= 0.1
            and support["body_speed_mps_min_median_p95_max"][3] >= 10.0
            and support["absolute_steering_rad_min_median_p95_max"][3] >= 0.45):
        raise ValueError("selected training windows do not span the intended race domain")

    features, residuals, row_runs, initial_sources = _capture_training_sources(
        sources, hgb, edssm, metadata, atlas, training_run_ids, None)
    feature_sets = [features]
    residual_sets = [residuals]
    run_sets = [row_runs]
    iterations: list[dict[str, Any]] = []
    fit_reports: list[dict[str, Any]] = []
    correction = None
    for iteration in range(ITERATIONS):
        correction, fit_report = _fit(
            np.concatenate(feature_sets), np.concatenate(residual_sets),
            np.concatenate(run_sets), training_run_ids)
        fit_report["iteration"] = iteration
        fit_reports.append(fit_report)
        if iteration + 1 < ITERATIONS:
            generated_x, generated_y, generated_runs, generated_sources = (
                _capture_training_sources(
                    sources, hgb, edssm, metadata, atlas,
                    training_run_ids, correction))
            feature_sets.append(generated_x)
            residual_sets.append(generated_y)
            run_sets.append(generated_runs)
            iterations.append({
                "generation": iteration + 1,
                "generated_training_rows": int(len(generated_x)),
                "generated_residual_rmse_rps": float(np.sqrt(np.mean(
                    generated_y ** 2))),
                "per_source": generated_sources,
            })
    if correction is None:
        raise AssertionError("yaw residual model did not fit")

    heldout = _validation_sources(hgb)
    training_set = set(training_run_ids.tolist())
    base_training_set = (set(hgb["training_run_ids"])
                         | set(metadata["training_runs"]))
    results: dict[str, Any] = {}
    for source in heldout:
        if set(source["run_ids"]) & (training_set | base_training_set):
            raise ValueError(
                f"held-out runs overlap model training runs in {source['name']}")
        horizon = source["horizon"]
        print(f"{source['name']}: paired baseline/full-regime residual, "
              f"{len(source['windows'])} windows", flush=True)
        baseline = _score(
            source["data"], hgb, edssm, metadata, atlas,
            source["windows"], horizon, None)
        candidate = _score(
            source["data"], hgb, edssm, metadata, atlas,
            source["windows"], horizon, correction)
        results[source["name"]] = {
            "split": source["split"],
            "run_ids": source["run_ids"],
            "windows": len(source["windows"]),
            "horizon_s": horizon * evaluator.DT_S,
            "same_window_parent": baseline,
            "full_regime_yaw_residual": candidate,
            "paired_run_delta": {
                horizon_key: _paired_run_deltas(
                    candidate, baseline, horizon_key)
                for horizon_key in ("0.750s", "2.000s", "5.000s")
                if horizon_key in baseline["per_run"][source["run_ids"][0]]
            },
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    joblib.dump({
        "schema_version": 1,
        "estimator": correction,
        "correction_start_step": CORRECTION_START_STEP,
        "scope": "all measured training regimes; no manual speed/steering gate",
        "training_run_ids": training_run_ids.tolist(),
        "training_dataset_sha256": {
            str(source["path"].relative_to(ROOT)): _sha256(source["path"])
            for source in sources},
        "research_only": True,
        "production_or_runtime_integration": False,
    }, model_path)
    report = {
        "schema_version": 1,
        "purpose": "test whether a run-balanced DAgger yaw residual can stabilize the frozen high-accuracy yaw atlas across its measured race-domain support",
        "research_only": True,
        "production_or_runtime_integration": False,
        "simulator_was_run": False,
        "test_or_final_test_used": False,
        "yaw_atlas_frozen": True,
        "base_hgb_and_edssm_frozen": True,
        "future_sensor_or_truth_input": False,
        "correction_start_step": CORRECTION_START_STEP,
        "correction_scope": "full regime, no speed/steering gate; the frozen atlas remains the primary yaw predictor",
        "training_data_gap_ticket": {
            "question": "Does feeding the frozen atlas its own predicted yaw-rate history explain recursive yaw drift across all regimes?",
            "evidence": "existing sensor-substitution rollouts show measured future yaw history nearly removes recursive one-step atlas error, while predicted yaw history leaves large error",
            "data_action": "reuse existing training-only 40-Hz broad and high-steer captures; no new simulation",
            "heldout_policy": "whole-run validation and unseen practice only; sealed test/final-test partitions excluded",
        },
        "training_protocol": {
            "fit_split": "train only",
            "training_sources": initial_sources,
            "training_run_ids": training_run_ids.tolist(),
            "selected_window_horizon_s": HORIZON * evaluator.DT_S,
            "maximum_nonoverlapping_windows_per_run_per_source": MAX_TRAIN_WINDOWS_PER_RUN,
            "support_summary": support,
            "dagger_generations": iterations,
            "fit_reports": fit_reports,
            "labels": "simulator-truth next yaw rate minus frozen-atlas prediction",
            "residual_features": "frozen atlas causal features built from predicted state/history plus frozen-atlas next yaw prediction",
            "initial_measured_history_only": True,
            "base_atlas_refit": False,
            "hgb_or_edssm_refit": False,
        },
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "heldout_results": results,
        "candidate_model": str(model_path.relative_to(ROOT)),
        "candidate_model_sha256": _sha256(model_path),
        "runtime_seconds": time.time() - started,
        "limitations": [
            "the residual is an offline plant correction, not a sensor-only runtime observer",
            "training uses 5-second causal seed windows, while broad and high-steer held-out windows run to 12 seconds",
            "the held-out cohorts are seven broad validation runs, three high-steer validation runs and two practice captures",
            "a low yaw-rate error alone is insufficient if u, v, wheel speed or integrated position regress",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    summary = {}
    for cohort, values in result["heldout_results"].items():
        summary[cohort] = {}
        for model_name in ("same_window_parent", "full_regime_yaw_residual"):
            summary[cohort][model_name] = {
                horizon: row["hybrid_atlas"]
                for horizon, row in values[model_name]["macro_run_rmse"].items()
                if horizon in ("0.750s", "2.000s", "5.000s", "12.000s")}
    print(json.dumps({"output": str(OUTPUT_DIR / "evaluation.json"),
                      "summary": summary}, indent=2, sort_keys=True))
