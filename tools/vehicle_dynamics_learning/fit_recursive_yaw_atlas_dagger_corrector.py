#!/usr/bin/env python3
"""Train a rollout-distribution yaw residual around the frozen yaw atlas.

The first rollout generates causal atlas feature histories from the composed
HGB/EDSSM/atlas plant. A shallow residual learner is fit to simulator-truth
next-yaw labels on training runs only. One DAgger iteration then rolls that
corrector through the same training runs, adds those generated histories, and
refits. Held-out runs are used only for final evaluation.
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

from tools.vehicle_dynamics_learning.evaluate_composed_latent_reconciliation import (
    GAIN_REPORT,
    HGB_TRAIN_DATASET,
    HIGHSTEER_DATASET,
    OUTPUT as COMPOSITION_REPORT,
    PRACTICE_DATASET,
    WINDOWS_PER_RUN,
)
from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    EDSSM_CHECKPOINT,
    HGB_BUNDLE,
    YAW_ATLAS_BUNDLE,
    _bundle_lags,
    _rollout,
    _windows,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    HORIZON,
    _run_windows,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as yaw_atlas


OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v4_20261009.json")
HIGHSTEER_TRAIN_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/highsteer_75_train_expanded_20261004/"
    "encoder_fixed40hz_v1/openplane_dynamics_raw_wheels_fixed40hz.npz")
ITERATIONS = 3
SEED = 20261009
GAIN = 1.0
CORRECTION_START_STEP = 1
HIGHSTEER_SPEED_MPS = (7.0, 9.0)
HIGHSTEER_ABS_STEERING_RAD = (0.30, 0.525)


class HighSteerYawResidual:
    """Apply a learned correction only within the measured high-steer support."""

    def __init__(self, model: Any, wheel_mean_bounds_mps: tuple[float, float]):
        self.model = model
        self.wheel_mean_bounds_mps = wheel_mean_bounds_mps

    def predict(self, features: np.ndarray) -> np.ndarray:
        wheel_mean = 0.5 * (features[:, 2] + features[:, 3])
        steering = np.abs(features[:, 0])
        mask = (
            (wheel_mean >= self.wheel_mean_bounds_mps[0])
            & (wheel_mean <= self.wheel_mean_bounds_mps[1])
            & (steering >= HIGHSTEER_ABS_STEERING_RAD[0])
            & (steering <= HIGHSTEER_ABS_STEERING_RAD[1]))
        output = np.zeros(len(features), dtype=np.float32)
        if np.any(mask):
            output[mask] = self.model.predict(features[mask]).astype(np.float32)
        return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(features: np.ndarray, residual: np.ndarray,
         run_indices: np.ndarray, run_ids: np.ndarray
         ) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    if (features.ndim != 2 or residual.shape != (len(features),)
            or run_indices.shape != residual.shape
            or not np.isfinite(features).all()
            or not np.isfinite(residual).all()):
        raise ValueError("invalid generated yaw-correction training rows")
    counts = np.bincount(run_indices, minlength=len(run_ids)).astype(np.float64)
    sample_weight = 1.0 / counts[run_indices]
    sample_weight *= len(sample_weight) / sample_weight.sum()
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=180,
        max_leaf_nodes=15, min_samples_leaf=60, l2_regularization=2.0,
        early_stopping=False, random_state=SEED)
    model.fit(features, residual, sample_weight=sample_weight)
    prediction = model.predict(features)
    per_run = {}
    for run in np.unique(run_indices):
        mask = run_indices == run
        per_run[str(run_ids[run])] = {
            "rows": int(np.count_nonzero(mask)),
            "residual_rmse_rps": float(np.sqrt(np.mean(residual[mask] ** 2))),
            "corrected_residual_rmse_rps": float(np.sqrt(np.mean(
                (residual[mask] - prediction[mask]) ** 2))),
        }
    return model, {
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "run_count": int(len(per_run)),
        "run_ids": sorted(per_run),
        "residual_mean_rps": float(np.mean(residual)),
        "residual_rmse_rps": float(np.sqrt(np.mean(residual ** 2))),
        "residual_p95_abs_rps": float(np.quantile(np.abs(residual), 0.95)),
        "corrected_residual_rmse_training_rps": float(np.sqrt(np.mean(
            (residual - prediction) ** 2))),
        "per_run_training_fit": per_run,
        "sample_weighting": "equal total weight per source run",
        "estimator": model.get_params(),
    }


def _capture(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
             metadata: dict[str, Any], atlas: dict[str, Any],
             windows: list[dict[str, int]],
             correction_model: Any | None
             ) -> dict[str, Any]:
    return _rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=GAIN,
        include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        yaw_correction_model=correction_model,
        yaw_correction_start_step=CORRECTION_START_STEP,
        capture_yaw_correction_rows=True)


def _capture_training_sources(
        sources: list[dict[str, Any]], hgb: dict[str, Any], edssm: Any,
        metadata: dict[str, Any], atlas: dict[str, Any],
        combined_run_ids: np.ndarray,
        correction_model: Any | None) -> tuple[np.ndarray, np.ndarray,
                                              np.ndarray, list[dict[str, Any]]]:
    combined_index = {str(run_id): index
                      for index, run_id in enumerate(combined_run_ids)}
    feature_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    for source in sources:
        data = source["data"]
        result = _capture(
            data, hgb, edssm, metadata, atlas, source["windows"],
            correction_model)
        features = result.pop("_yaw_correction_training_features")
        residuals = result.pop("_yaw_correction_training_residuals")
        local_runs = result.pop("_yaw_correction_training_run_indices")
        local_ids = np.asarray(data["run_ids"]).astype(str)
        source_run_ids = local_ids[local_runs]
        try:
            combined_runs = np.asarray(
                [combined_index[str(run_id)] for run_id in source_run_ids],
                dtype=np.int32)
        except KeyError as exc:
            raise ValueError(
                f"captured run {exc.args[0]!r} is absent from training manifest") from exc
        feature_parts.append(features)
        residual_parts.append(residuals)
        run_parts.append(combined_runs)
        reports.append({
            "source": source["name"],
            "windows": len(source["windows"]),
            "training_run_ids": sorted({
                str(local_ids[row["run"]]) for row in source["windows"]}),
            "captured_rows": int(len(features)),
            "residual_rmse_rps": float(np.sqrt(np.mean(residuals ** 2))),
        })
    return (np.concatenate(feature_parts), np.concatenate(residual_parts),
            np.concatenate(run_parts), reports)


def _highsteer_windows(data: dict[str, Any], maximum_per_run: int,
                       hgb_lags: tuple[int, ...]
                       ) -> tuple[list[dict[str, int]], tuple[float, float]]:
    """Select non-overlapping training windows matching the validation cohort."""
    candidates = _run_windows(
        data, "train", hgb_lags, maximum_per_run=10_000)
    frames = np.asarray(data["frames"], dtype=np.float64)
    by_run: dict[int, list[dict[str, int]]] = {}
    for row in candidates:
        start = int(row["start"])
        speed = float(np.hypot(*frames[start, :2]))
        steering = abs(float(frames[start, 3]))
        if (HIGHSTEER_SPEED_MPS[0] <= speed <= HIGHSTEER_SPEED_MPS[1]
                and HIGHSTEER_ABS_STEERING_RAD[0] <= steering
                <= HIGHSTEER_ABS_STEERING_RAD[1]):
            by_run.setdefault(int(row["run"]), []).append(row)
    selected: list[dict[str, int]] = []
    for rows in by_run.values():
        if len(rows) > maximum_per_run:
            positions = np.linspace(0, len(rows) - 1,
                                    maximum_per_run, dtype=np.int64)
            rows = [rows[int(position)] for position in np.unique(positions)]
        selected.extend(rows)
    if not selected:
        raise ValueError("no high-steer training windows match validation support")
    selected_indices = np.concatenate([
        np.arange(row["start"] - max(hgb_lags), row["start"] + HORIZON + 1)
        for row in selected])
    measured = frames[selected_indices]
    body_speed = np.hypot(measured[:, 0], measured[:, 1])
    absolute_steering = np.abs(measured[:, 3])
    supported = (
        (body_speed >= HIGHSTEER_SPEED_MPS[0])
        & (body_speed <= HIGHSTEER_SPEED_MPS[1])
        & (absolute_steering >= HIGHSTEER_ABS_STEERING_RAD[0])
        & (absolute_steering <= HIGHSTEER_ABS_STEERING_RAD[1]))
    wheel_mean = np.mean(measured[:, 5:7], axis=1)[supported]
    if not len(wheel_mean):
        raise ValueError("selected high-steer windows contain no gate support")
    return selected, (float(np.min(wheel_mean)), float(np.max(wheel_mean)))


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    model_output = output_path.with_suffix(".joblib")
    if output_path.exists() or model_output.exists():
        raise FileExistsError("refusing to overwrite a prior DAgger result")
    required = (HGB_BUNDLE, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
                HGB_TRAIN_DATASET, HIGHSTEER_TRAIN_DATASET,
                HIGHSTEER_DATASET, PRACTICE_DATASET,
                GAIN_REPORT, COMPOSITION_REPORT)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    hgb = joblib.load(HGB_BUNDLE)
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    # The frozen atlas is a forest of small local experts. Single-threaded
    # prediction avoids per-expert thread startup overhead in the 40-Hz loop.
    for model in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    train_data = _load_dataset(HGB_TRAIN_DATASET)
    atlas_series, _ = yaw_atlas.teacher._discover_series_with_safe_mixed_archives()
    atlas_training = {series.run_id for series in atlas_series
                      if series.split == "train"}
    highsteer_train_data = _load_dataset(HIGHSTEER_TRAIN_DATASET)
    highsteer_train_splits = np.asarray(
        highsteer_train_data["splits"]).astype(str)
    highsteer_train_ids = np.asarray(
        highsteer_train_data["run_ids"]).astype(str)
    highsteer_train_windows, wheel_mean_bounds = _highsteer_windows(
        highsteer_train_data, WINDOWS_PER_RUN, _bundle_lags(hgb))
    highsteer_selected_ids = {
        str(highsteer_train_ids[row["run"]]) for row in highsteer_train_windows}
    if (not highsteer_selected_ids
            or not highsteer_selected_ids <= set(highsteer_train_ids[
                highsteer_train_splits == "train"])):
        raise ValueError("high-steering correction fit selected a non-training run")
    combined_run_ids = np.asarray(sorted(highsteer_selected_ids), dtype=str)
    training_sources = [
        {"name": "highsteer_training", "data": highsteer_train_data,
         "windows": highsteer_train_windows},
    ]
    initial_features, initial_residuals, initial_runs, initial_sources = (
        _capture_training_sources(
            training_sources, hgb, edssm, metadata, atlas,
            combined_run_ids, None))
    features_by_iteration = [initial_features]
    residual_by_iteration = [initial_residuals]
    run_by_iteration = [initial_runs]
    iteration_reports: list[dict[str, Any]] = []
    model: HistGradientBoostingRegressor | None = None
    fit_reports: list[dict[str, Any]] = []

    for iteration in range(ITERATIONS):
        features = np.concatenate(features_by_iteration, axis=0)
        residual = np.concatenate(residual_by_iteration, axis=0)
        row_runs = np.concatenate(run_by_iteration, axis=0)
        model, fit_report = _fit(
            features, residual, row_runs, combined_run_ids)
        fit_reports.append(fit_report)
        if iteration + 1 < ITERATIONS:
            generated_features, generated_residuals, generated_runs, source_reports = (
                _capture_training_sources(
                    training_sources, hgb, edssm, metadata, atlas,
                    combined_run_ids,
                    HighSteerYawResidual(model, wheel_mean_bounds)))
            features_by_iteration.append(generated_features)
            residual_by_iteration.append(generated_residuals)
            run_by_iteration.append(generated_runs)
            iteration_reports.append({
                "iteration": iteration + 1,
                "rollout_gain": GAIN,
                "correction_gate": {
                    "rear_wheel_mean_mps": list(wheel_mean_bounds),
                    "absolute_steering_rad": list(HIGHSTEER_ABS_STEERING_RAD),
                },
                "rollout_sources": source_reports,
                "rollout_run_ids": sorted(combined_run_ids.tolist()),
                "rollout_windows": sum(len(source["windows"])
                                        for source in training_sources),
                "captured_rows": int(len(generated_features)),
                "captured_residual_rmse_rps": float(np.sqrt(np.mean(
                    generated_residuals ** 2))),
                "generated_by_previous_corrector": True,
            })

    if model is None:
        raise AssertionError("DAgger loop failed to fit a correction model")

    # Evaluation runs do not influence fitting or architecture selection.
    validation_windows = _run_windows(
        train_data, "validation", _bundle_lags(hgb), WINDOWS_PER_RUN)
    highsteer_data = _load_dataset(HIGHSTEER_DATASET)
    highsteer_windows = _windows(
        highsteer_data, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))
    practice_data = _load_dataset(PRACTICE_DATASET)
    practice_windows = _windows(
        practice_data, PRACTICE_DATASET, HORIZON, _bundle_lags(hgb))
    heldout_specs = (
        ("whole_run_validation", train_data, validation_windows, "validation"),
        ("highsteer_validation", highsteer_data, highsteer_windows, "validation"),
        ("unseen_practice_diagnostic", practice_data, practice_windows,
         "unseen_practice"),
    )
    for name, data, selected_windows, expected_split in heldout_specs:
        ids = np.asarray(data["run_ids"]).astype(str)
        selected_ids = {str(ids[row["run"]]) for row in selected_windows}
        if not selected_ids or selected_ids & (
                set(hgb["training_run_ids"])
                | set(metadata["training_runs"]) | atlas_training
                | set(combined_run_ids.tolist())):
            raise ValueError(f"{name} overlaps a base model's training runs")
        data_splits = np.asarray(data["splits"]).astype(str)
        if any(data_splits[row["run"]] != expected_split
               for row in selected_windows):
            raise ValueError(f"unexpected split selected for {name}")
    evaluation_specs = (
        ("whole_run_validation", train_data, validation_windows),
        ("highsteer_validation", highsteer_data, highsteer_windows),
        ("unseen_practice_diagnostic", practice_data, practice_windows),
    )
    candidate_results: dict[str, Any] = {}
    baseline_results: dict[str, Any] = {}
    gated_model = HighSteerYawResidual(model, wheel_mean_bounds)
    for name, data, selected_windows in evaluation_specs:
        baseline = _rollout(
            data, hgb, edssm, metadata, atlas, selected_windows, HORIZON,
            yaw_delta_gain=GAIN,
            include_oracle_diagnostics=False,
            reconcile_composed_latent=True)
        scored = _rollout(
            data, hgb, edssm, metadata, atlas, selected_windows, HORIZON,
            yaw_delta_gain=GAIN,
            include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            yaw_correction_model=gated_model,
            yaw_correction_start_step=CORRECTION_START_STEP)
        candidate_results[name] = {
            "windows": scored["windows"],
            "windows_by_run": scored["windows_by_run"],
            "heldout_run_ids": sorted(scored["per_run"]),
            "macro_run_rmse": scored["macro_run_rmse"],
            "per_run": scored["per_run"],
            "one_step_yaw_atlas_with_correction": scored[
                "yaw_atlas_one_step_selected_windows"],
        }
        baseline_results[name] = {
            "windows": baseline["windows"],
            "windows_by_run": baseline["windows_by_run"],
            "heldout_run_ids": sorted(baseline["per_run"]),
            "macro_run_rmse": baseline["macro_run_rmse"],
            "per_run": baseline["per_run"],
            "one_step_yaw_atlas": baseline[
                "yaw_atlas_one_step_selected_windows"],
        }

    base_composition_report = json.loads(COMPOSITION_REPORT.read_text())
    reference = {
        name: base_composition_report["candidate_results"][name]["1.0"][
            "macro_run_rmse"]
        for name in ("whole_run_validation", "highsteer_validation",
                     "unseen_practice_diagnostic")
    }
    report = {
        "schema_version": 1,
        "purpose": "train a high-steer-only causal yaw-atlas residual on matched training windows; preserve the frozen atlas for the first transition and leave states outside measured support unchanged",
        "research_only": True,
        "production_or_runtime_integration": False,
        "simulator_was_run": False,
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "training_protocol": {
            "fit_split": "train only",
            "training_run_ids": sorted(combined_run_ids.tolist()),
            "training_sources": initial_sources,
            "training_run_ids_also_seen_by_yaw_atlas": sorted(
                set(combined_run_ids.tolist()) & atlas_training),
            "training_window_count": sum(
                len(source["windows"]) for source in training_sources),
            "training_run_ids_seen_by_base_hgb_and_edssm": [],
            "additional_highsteer_training_run_ids": sorted(
                highsteer_selected_ids),
            "window_horizon_s": HORIZON * 0.025,
            "dagger_iterations": ITERATIONS,
            "correction_start_step": CORRECTION_START_STEP,
            "initial_observed_transition_uses_frozen_yaw_atlas_without_residual": True,
            "correction_gate": {
                "rear_wheel_mean_mps": list(wheel_mean_bounds),
                "absolute_steering_rad": list(HIGHSTEER_ABS_STEERING_RAD),
                "speed_window_selection_body_mps": list(HIGHSTEER_SPEED_MPS),
                "training_window_selection": (
                    "non-overlapping windows whose start is 7-9 m/s body speed "
                    "and |steering feedback| is 0.30-0.525 rad; correction "
                    "gate uses only causal rear-wheel mean and steering feedback"),
            },
            "base_atlas_refit": False,
            "hgb_and_edssm_refit": False,
            "highsteer_split": "train only; separate two-run capture",
            "truth_used_only_as_next_yaw_label": True,
            "future_sensors_or_truth_as_features": False,
            "test_or_final_test_used": False,
            "correction_target": "simulator_truth_next_yaw_rate minus frozen-atlas output on the current recursively generated causal feature history",
            "correction_features": "exact frozen-atlas causal features plus frozen-atlas next-yaw output",
            "rollout_state_composition": "HGB u/actuator/wheels, EDSSM v/roll, frozen yaw atlas plus learned residual; latent transition recomputed from composed state increments",
        },
        "fit_reports": fit_reports,
        "dagger_generation": iteration_reports,
        "heldout_reference_gain_1_baseline": reference,
        "same_window_direct_baseline": baseline_results,
        "heldout_reference_report": {
            "path": str(COMPOSITION_REPORT.relative_to(ROOT)),
            "sha256": _sha256(COMPOSITION_REPORT),
        },
        "candidate_results": candidate_results,
        "model_artifact_path": str(model_output.relative_to(ROOT)),
        "model_artifact_written": False,
        "runtime_seconds": time.time() - started,
        "limitations": [
            "held-out evaluation remains short-horizon and cannot certify a full-lap simulator",
            "there are only three high-steering validation runs and two practice diagnostics",
            "the current plant composition still has separate HGB, EDSSM, and yaw-atlas failure modes",
            "the correction is supported only by two independent high-steer training runs and the measured 7-9 m/s start envelope",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The fitted correction remains a separate research candidate. Preserve it
    # only when it improves high-steer yaw at both multi-step horizons and
    # avoids regressions in the 5-second yaw/pose scores on other held-out sets.
    highsteer = candidate_results["highsteer_validation"]["macro_run_rmse"]
    highsteer_base = baseline_results["highsteer_validation"]["macro_run_rmse"]
    other_sets = ("whole_run_validation", "unseen_practice_diagnostic")
    highsteer_gain_supported = all(
        highsteer[key]["hybrid_atlas"]["yaw_rate_rmse_rps"]
        < highsteer_base[key]["hybrid_atlas"]["yaw_rate_rmse_rps"]
        for key in ("2.000s", "5.000s"))
    no_heldout_yaw_pose_regression = all(
        candidate_results[name]["macro_run_rmse"]["5.000s"]["hybrid_atlas"][metric]
        <= baseline_results[name]["macro_run_rmse"]["5.000s"]["hybrid_atlas"][metric]
        for name in other_sets for metric in ("yaw_rate_rmse_rps", "position_rmse_m"))
    highsteer_position_improves = (
        candidate_results["highsteer_validation"]["macro_run_rmse"]["5.000s"]
        ["hybrid_atlas"]["position_rmse_m"]
        < baseline_results["highsteer_validation"]["macro_run_rmse"]["5.000s"]
        ["hybrid_atlas"]["position_rmse_m"])
    initial_step_preserved = all(
        np.isclose(
            candidate_results[name]["one_step_yaw_atlas_with_correction"][
                "rmse_rps"],
            baseline_results[name]["one_step_yaw_atlas"]["rmse_rps"],
            rtol=0.0, atol=1e-12)
        for name in candidate_results)
    report["candidate_decision"] = {
        "criteria": {
            "highsteer_yaw_rate_improves_at_2s_and_5s": highsteer_gain_supported,
            "highsteer_5s_position_improves": highsteer_position_improves,
            "first_observed_step_is_identical_to_frozen_atlas": initial_step_preserved,
            "whole_run_and_practice_5s_yaw_and_position_do_not_regress":
                no_heldout_yaw_pose_regression,
        },
        "retained_as_research_candidate": bool(
            highsteer_gain_supported and highsteer_position_improves
            and initial_step_preserved and no_heldout_yaw_pose_regression),
        "production_or_runtime_integration": False,
    }
    if report["candidate_decision"]["retained_as_research_candidate"]:
        joblib.dump({
            "schema_version": 1,
            "estimator": model,
            "correction_start_step": CORRECTION_START_STEP,
            "gate": {
                "rear_wheel_mean_mps": list(wheel_mean_bounds),
                "absolute_steering_rad": list(HIGHSTEER_ABS_STEERING_RAD),
            },
            "training_run_ids": sorted(combined_run_ids.tolist()),
            "research_only": True,
            "production_or_runtime_integration": False,
        }, model_output)
        report["model_artifact_written"] = True
        report["model_artifact_sha256"] = _sha256(model_output)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps({
        "output": str(args.output),
        "training": report["training_protocol"],
        "dagger_generation": report["dagger_generation"],
        "candidate_5s": {
            name: row["macro_run_rmse"]["5.000s"]
            for name, row in report["candidate_results"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
