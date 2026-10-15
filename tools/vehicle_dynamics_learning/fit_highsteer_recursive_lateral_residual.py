#!/usr/bin/env python3
"""Fit and evaluate a train-only, high-steer residual for lateral velocity.

The base state transition keeps the frozen yaw atlas and the gain-consistent
high-steer longitudinal residual fixed. This experiment learns only the EDSSM
next-v residual on causal rollout states, gates it to the measured high-steer
support, and leaves the first observed transition unchanged. It is an offline
research candidate; it does not alter odometry, MPC, or simulator physics.
"""

from __future__ import annotations

import hashlib
import json
import time
import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_composed_latent_reconciliation import (
    WINDOWS_PER_RUN,
)
from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    EDSSM_CHECKPOINT,
    HGB_BUNDLE,
    HGB_TRAIN_DATASET,
    HIGHSTEER_DATASET,
    PRACTICE_DATASET,
    YAW_ATLAS_BUNDLE,
    _bundle_lags,
    _rollout,
    _windows,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    HORIZON,
    _run_windows,
)
from tools.vehicle_dynamics_learning.fit_highsteer_augmented_longitudinal_candidate import (
    OUTPUT_DIR as LONGITUDINAL_CANDIDATE_DIR,
)
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_longitudinal_residual import (
    HIGHSTEER_ABS_STEERING_RAD,
    HIGHSTEER_SPEED_MPS,
    HIGHSTEER_TRAIN_DATASET,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
    _highsteer_windows,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


LONGITUDINAL_CANDIDATE = (
    LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib")
YAW_CORRECTOR_PATH = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
LONGITUDINAL_RESIDUAL_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_recursive_longitudinal_residual_gain0p5_training_20261009/"
    "u_filtered_wheel_residual_models_gain0p5.joblib")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_recursive_lateral_residual_20261009")
OUTPUT = OUTPUT_DIR / "evaluation.json"
MODEL_OUTPUT = OUTPUT_DIR / "lateral_v_residual.joblib"
EXPANDED_OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_recursive_lateral_residual_multirun_20261009")
GAIN = 0.5
DAGGER_ITERATIONS = 2
SEED = 20261009


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(features: np.ndarray, targets: np.ndarray,
         run_indices: np.ndarray, run_ids: np.ndarray
         ) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    if (features.ndim != 2 or targets.shape != (len(features),)
            or run_indices.shape != targets.shape or not len(features)
            or not np.isfinite(features).all()
            or not np.isfinite(targets).all()):
        raise ValueError("invalid rollout-conditioned lateral residual rows")
    counts = np.bincount(run_indices, minlength=len(run_ids)).astype(np.float64)
    if np.any(counts[np.unique(run_indices)] <= 0):
        raise ValueError("a high-steer train run has no lateral residual rows")
    weights = 1.0 / counts[run_indices]
    weights *= len(weights) / weights.sum()
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=180,
        max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=2.0,
        early_stopping=False, random_state=SEED)
    model.fit(features, targets, sample_weight=weights)
    prediction = model.predict(features)
    report = {
        "rows": int(len(features)),
        "features": int(features.shape[1]),
        "training_run_ids": sorted({str(run_ids[index])
                                     for index in np.unique(run_indices)}),
        "per_run_rows": {
            str(run_ids[index]): int(np.count_nonzero(run_indices == index))
            for index in np.unique(run_indices)},
        "target_residual_rmse_mps": float(np.sqrt(np.mean(targets ** 2))),
        "fit_residual_rmse_mps": float(np.sqrt(np.mean(
            (targets - prediction) ** 2))),
        "target_p95_abs_mps": float(np.quantile(np.abs(targets), 0.95)),
        "sample_weighting": "equal total weight per training run",
        "estimator_parameters": model.get_params(),
    }
    return model, report


def _capture_training(data: dict[str, Any], hgb: dict[str, Any],
                      edssm: Any, metadata: dict[str, Any], atlas: dict[str, Any],
                      yaw_model: Any, yaw_start_step: int,
                      longitudinal_residual: dict[str, Any],
                      windows: list[dict[str, int]],
                      local_run_id_to_global: dict[str, int],
                      lateral_model: Any | None,
                      residual_gain: float,
                      feature_variant: str,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    gate = {
        "speed_mps": tuple(HIGHSTEER_SPEED_MPS),
        "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD),
    }
    captured = _rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=1.0, include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source="composed_transition",
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=yaw_start_step,
        hgb_main_gate=gate,
        hgb_main_residual_models=longitudinal_residual["models"],
        hgb_main_residual_start_step=1,
        hgb_main_residual_gain=float(longitudinal_residual["gain"]),
        lateral_v_residual_model=lateral_model,
        lateral_v_residual_start_step=1,
        lateral_v_residual_gain=residual_gain,
        lateral_v_residual_feature_variant=feature_variant,
        capture_lateral_v_residual_rows=True)
    features = captured.pop("_lateral_v_residual_training_features")
    targets = captured.pop("_lateral_v_residual_training_targets")
    local_runs = captured.pop("_lateral_v_residual_training_run_indices")
    local_ids = np.asarray(data["run_ids"]).astype(str)
    try:
        global_runs = np.asarray(
            [local_run_id_to_global[str(local_ids[index])]
             for index in local_runs], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"unexpected lateral training run {exc.args[0]!r}") from exc
    captured["captured_rows"] = int(len(features))
    captured["train_run_ids"] = sorted({
        str(local_ids[row["run"]]) for row in windows})
    return features, targets, global_runs, captured


def _broad_highsteer_windows(data: dict[str, Any],
                             hgb_lags: tuple[int, ...]
                             ) -> list[dict[str, int]]:
    """Select existing train windows jointly supported by GT and sensors."""
    candidates = _run_windows(data, "train", hgb_lags, maximum_per_run=10_000)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    selected = []
    for row in candidates:
        start = int(row["start"])
        gt_speed = float(np.hypot(rigid[start, 7], rigid[start, 8]))
        observed_speed = float(np.hypot(frames[start, 0], frames[start, 1]))
        steering = abs(float(frames[start, 3]))
        if (HIGHSTEER_SPEED_MPS[0] <= gt_speed <= HIGHSTEER_SPEED_MPS[1]
                and HIGHSTEER_SPEED_MPS[0] <= observed_speed
                <= HIGHSTEER_SPEED_MPS[1]
                and HIGHSTEER_ABS_STEERING_RAD[0] <= steering
                <= HIGHSTEER_ABS_STEERING_RAD[1]):
            selected.append(row)
    return selected


def _evaluate(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
              metadata: dict[str, Any], atlas: dict[str, Any], yaw_model: Any,
              yaw_start_step: int, longitudinal_residual: dict[str, Any],
              windows: list[dict[str, int]], lateral_model: Any | None,
              residual_gain: float, feature_variant: str,
              ) -> dict[str, Any]:
    gate = {
        "speed_mps": tuple(HIGHSTEER_SPEED_MPS),
        "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD),
    }
    return _rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=1.0, include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source="composed_transition",
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=yaw_start_step,
        hgb_main_gate=gate,
        hgb_main_residual_models=longitudinal_residual["models"],
        hgb_main_residual_start_step=1,
        hgb_main_residual_gain=float(longitudinal_residual["gain"]),
        lateral_v_residual_model=lateral_model,
        lateral_v_residual_start_step=1,
        lateral_v_residual_gain=residual_gain,
        lateral_v_residual_feature_variant=feature_variant)


def run(output_dir: Path = EXPANDED_OUTPUT_DIR,
        include_broad_train: bool = True,
        residual_gain: float = GAIN,
        feature_variant: str = "base") -> dict[str, Any]:
    if not np.isfinite(residual_gain) or not 0.0 <= residual_gain <= 1.0:
        raise ValueError("lateral residual gain must be finite and in [0, 1]")
    if feature_variant not in ("base", "roll_acceleration"):
        raise ValueError("unsupported lateral residual feature variant")
    output_path = output_dir / "evaluation.json"
    model_output = output_dir / "lateral_v_residual.joblib"
    if output_dir.exists() or output_path.exists() or model_output.exists():
        raise FileExistsError(f"refusing to overwrite existing lateral result in {output_dir}")
    required = (
        HGB_BUNDLE, LONGITUDINAL_CANDIDATE, LONGITUDINAL_RESIDUAL_BUNDLE,
        YAW_CORRECTOR_PATH, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
        HGB_TRAIN_DATASET, HIGHSTEER_TRAIN_DATASET, HIGHSTEER_DATASET,
        PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    parent_hgb = joblib.load(HGB_BUNDLE)
    longitudinal_candidate = joblib.load(LONGITUDINAL_CANDIDATE)
    hgb = dict(parent_hgb)
    hgb["models"] = longitudinal_candidate["models"]
    hgb["training_run_ids"] = longitudinal_candidate["training_run_ids"]
    longitudinal_residual = joblib.load(LONGITUDINAL_RESIDUAL_BUNDLE)
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"], tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    yaw_start_step = int(yaw_bundle.get(
        "correction_start_step", CORRECTION_START_STEP))
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for model in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")

    train_data = _load_dataset(HIGHSTEER_TRAIN_DATASET)
    train_windows, train_wheel_bounds = _highsteer_windows(
        train_data, WINDOWS_PER_RUN, _bundle_lags(hgb))
    training_sources = [{
        "name": "highsteer_expanded_train",
        "data": train_data,
        "windows": train_windows,
    }]
    if include_broad_train:
        broad_train_data = _load_dataset(HGB_TRAIN_DATASET)
        broad_windows = _broad_highsteer_windows(
            broad_train_data, _bundle_lags(hgb))
        if not broad_windows:
            raise ValueError("no broad train windows matched high-steer GT support")
        training_sources.append({
            "name": "whole_dynamics_train_highsteer_subset",
            "data": broad_train_data,
            "windows": broad_windows,
        })
    selected_run_ids = sorted({
        str(np.asarray(source["data"]["run_ids"]).astype(str)[row["run"]])
        for source in training_sources for row in source["windows"]})
    for source in training_sources:
        source_splits = np.asarray(source["data"]["splits"]).astype(str)
        if any(source_splits[row["run"]] != "train" for row in source["windows"]):
            raise ValueError("lateral fit selected a non-training run")
    training_run_ids_by_source = {
        source["name"]: sorted({
            str(np.asarray(source["data"]["run_ids"]).astype(str)[row["run"]])
            for row in source["windows"]})
        for source in training_sources}
    if int(longitudinal_residual.get("gain", -1)) != 0 and not np.isclose(
            float(longitudinal_residual.get("gain", -1)), GAIN):
        raise ValueError("longitudinal residual gain does not match the evaluated parent")
    if (not np.isclose(float(longitudinal_residual.get("gate", {}).get(
            "speed_mps", [np.nan, np.nan])[0]), HIGHSTEER_SPEED_MPS[0])
            or not np.isclose(float(longitudinal_residual.get("gate", {}).get(
                "absolute_steering_rad", [np.nan, np.nan])[0]),
                HIGHSTEER_ABS_STEERING_RAD[0])):
        raise ValueError("longitudinal residual support does not match lateral domain")
    run_id_to_global = {run_id: index
                        for index, run_id in enumerate(selected_run_ids)}

    feature_sets: list[np.ndarray] = []
    target_sets: list[np.ndarray] = []
    run_sets: list[np.ndarray] = []
    generation_reports: list[dict[str, Any]] = []
    fit_reports: list[dict[str, Any]] = []
    lateral_model: HistGradientBoostingRegressor | None = None
    with threadpool_limits(limits=2):
        for iteration in range(DAGGER_ITERATIONS):
            feature_parts: list[np.ndarray] = []
            target_parts: list[np.ndarray] = []
            run_parts: list[np.ndarray] = []
            source_captures = []
            for source in training_sources:
                features, targets, row_runs, capture_report = _capture_training(
                    source["data"], hgb, edssm, metadata, atlas, yaw_model,
                    yaw_start_step, longitudinal_residual, source["windows"],
                    run_id_to_global, lateral_model, residual_gain,
                    feature_variant)
                feature_parts.append(features)
                target_parts.append(targets)
                run_parts.append(row_runs)
                source_captures.append({
                    "source": source["name"],
                    "windows": len(source["windows"]),
                    **capture_report,
                })
            features = np.concatenate(feature_parts, axis=0)
            targets = np.concatenate(target_parts, axis=0)
            row_runs = np.concatenate(run_parts, axis=0)
            feature_sets.append(features)
            target_sets.append(targets)
            run_sets.append(row_runs)
            aggregate_features = np.concatenate(feature_sets, axis=0)
            aggregate_targets = np.concatenate(target_sets, axis=0)
            aggregate_runs = np.concatenate(run_sets, axis=0)
            lateral_model, fit_report = _fit(
                aggregate_features, aggregate_targets, aggregate_runs,
                np.asarray(selected_run_ids, dtype=str))
            generation_reports.append({
                "iteration": iteration,
                "generated_with_previous_v_residual": lateral_model is not None
                    and iteration > 0,
                "source_captures": source_captures,
                "total_captured_rows": int(len(features)),
                "target_rmse_mps": float(np.sqrt(np.mean(targets ** 2))),
                "target_p95_abs_mps": float(np.quantile(np.abs(targets), 0.95)),
            })
            fit_reports.append(fit_report)
    if lateral_model is None:
        raise AssertionError("no lateral residual model was fitted")

    whole_data = _load_dataset(HGB_TRAIN_DATASET)
    whole_windows = _run_windows(
        whole_data, "validation", _bundle_lags(hgb), WINDOWS_PER_RUN)
    highsteer_data = _load_dataset(HIGHSTEER_DATASET)
    highsteer_windows = _windows(
        highsteer_data, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))
    practice_data = _load_dataset(PRACTICE_DATASET)
    practice_windows = _windows(
        practice_data, PRACTICE_DATASET, HORIZON, _bundle_lags(hgb))
    evaluation_specs = (
        ("whole_run_validation", whole_data, whole_windows, "validation"),
        ("highsteer_validation", highsteer_data, highsteer_windows, "validation"),
        ("unseen_practice_diagnostic", practice_data, practice_windows,
         "unseen_practice"),
    )
    heldout: dict[str, Any] = {}
    for name, data, windows, expected_split in evaluation_specs:
        ids = np.asarray(data["run_ids"]).astype(str)
        splits = np.asarray(data["splits"]).astype(str)
        selected_ids = {str(ids[row["run"]]) for row in windows}
        if (not selected_ids or selected_ids & set(selected_run_ids)
                or any(splits[row["run"]] != expected_split for row in windows)):
            raise ValueError(f"held-out split/run leakage in {name}")
        baseline = _evaluate(
            data, hgb, edssm, metadata, atlas, yaw_model, yaw_start_step,
            longitudinal_residual, windows, None, residual_gain,
            feature_variant)
        candidate = _evaluate(
            data, hgb, edssm, metadata, atlas, yaw_model, yaw_start_step,
            longitudinal_residual, windows, lateral_model, residual_gain,
            feature_variant)
        heldout[name] = {
            "run_ids": sorted(candidate["per_run"]),
            "windows": candidate["windows"],
            "windows_by_run": candidate["windows_by_run"],
            "support_gate_active_fraction": candidate[
                "hgb_main_gate_active_fraction"],
            "baseline": {
                "macro_run_rmse": baseline["macro_run_rmse"],
                "per_run": baseline["per_run"],
                "first_transition_yaw_atlas": baseline[
                    "yaw_atlas_one_step_selected_windows"],
            },
            "lateral_v_residual": {
                "macro_run_rmse": candidate["macro_run_rmse"],
                "per_run": candidate["per_run"],
                "first_transition_yaw_atlas": candidate[
                    "yaw_atlas_one_step_selected_windows"],
            },
        }

    high_base = heldout["highsteer_validation"]["baseline"][
        "macro_run_rmse"]["5.000s"]["hybrid_atlas"]
    high_candidate = heldout["highsteer_validation"]["lateral_v_residual"][
        "macro_run_rmse"]["5.000s"]["hybrid_atlas"]
    high_v_improved = high_candidate["v_rmse_mps"] < high_base["v_rmse_mps"]
    high_position_supported = (
        high_candidate["position_rmse_m"] <= 1.10 * high_base["position_rmse_m"])
    high_yaw_supported = (
        high_candidate["yaw_rate_rmse_rps"] <= 1.10 * high_base["yaw_rate_rmse_rps"])
    transfer_supported = all(
        heldout[name]["lateral_v_residual"]["macro_run_rmse"]
        ["5.000s"]["hybrid_atlas"][metric]
        <= 1.10 * heldout[name]["baseline"]["macro_run_rmse"]
        ["5.000s"]["hybrid_atlas"][metric]
        for name in ("whole_run_validation", "unseen_practice_diagnostic")
        for metric in ("v_rmse_mps", "position_rmse_m", "yaw_rate_rmse_rps"))

    report = {
        "schema_version": 1,
        "purpose": "train-only rollout-conditioned high-steer lateral-velocity residual on top of the frozen yaw atlas and gain-consistent u/wheel candidate",
        "research_only": True,
        "simulator_was_run": False,
        "production_or_runtime_integration": False,
        "test_or_final_test_used": False,
        "fit_split": "high-steer training windows only; includes broad-dynamics training data when enabled",
        "training_run_ids": selected_run_ids,
        "training_run_ids_by_source": training_run_ids_by_source,
        "training_window_count": sum(len(source["windows"])
                                      for source in training_sources),
        "training_support": {
            "body_speed_start_mps": list(HIGHSTEER_SPEED_MPS),
            "absolute_steering_rad": list(HIGHSTEER_ABS_STEERING_RAD),
            "highsteer_capture_rear_wheel_mean_mps": list(train_wheel_bounds),
            "broad_train_windows_filtered_by_gt_and_observed_speed": True,
            "gate_uses_current_predicted_body_speed_and_steering": True,
        },
        "causal_contract": {
            "first_transition_uses_no_lateral_residual": True,
            "rollout_residual_features_use_predicted_state_and_commands": True,
            "simulator_truth_used_only_for_train_labels_and_scoring": True,
            "predicted_transition_acceleration_used_to_advance_yaw_atlas_and_latent": True,
            "yaw_atlas_bundle_and_corrector_frozen": True,
        },
        "longitudinal_parent": {
            "residual_model_path": str(LONGITUDINAL_RESIDUAL_BUNDLE.relative_to(ROOT)),
            "residual_gain": float(longitudinal_residual["gain"]),
            "yaw_atlas_path": str(YAW_ATLAS_BUNDLE.relative_to(ROOT)),
            "yaw_corrector_path": str(YAW_CORRECTOR_PATH.relative_to(ROOT)),
        },
        "dagger_iterations": DAGGER_ITERATIONS,
        "residual_gain_used_during_dagger_and_evaluation": residual_gain,
        "feature_variant": feature_variant,
        "generation_reports": generation_reports,
        "fit_reports": fit_reports,
        "output_dir": str(output_dir.relative_to(ROOT)),
        "heldout_results": heldout,
        "candidate_decision": {
            "criteria": {
                "highsteer_5s_lateral_velocity_improves": bool(high_v_improved),
                "highsteer_5s_position_within_10pct": bool(high_position_supported),
                "highsteer_5s_yaw_rate_within_10pct": bool(high_yaw_supported),
                "whole_run_and_practice_5s_v_position_yaw_within_10pct": bool(
                    transfer_supported),
            },
            "retained_as_research_candidate": bool(
                high_v_improved and high_position_supported
                and high_yaw_supported and transfer_supported),
            "production_or_runtime_integration": False,
        },
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "limitations": [
            "fit uses the five-second windows available from the selected training captures",
            "the high-steer validation cohort has already informed earlier model development and is not a pristine final test",
            "two practice runs provide a limited transfer diagnostic",
            "this remains a five-second partial plant model, not an accurate full-lap simulator",
        ],
        "runtime_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump({
        "schema_version": 1,
        "model": lateral_model,
        "gain": residual_gain,
        "feature_variant": feature_variant,
        "gate": {
            "speed_mps": list(HIGHSTEER_SPEED_MPS),
            "absolute_steering_rad": list(HIGHSTEER_ABS_STEERING_RAD),
        },
        "correction_start_step": 1,
        "training_run_ids": selected_run_ids,
        "research_only": True,
        "production_or_runtime_integration": False,
    }, model_output)
    report["model_artifact_written"] = True
    report["model_artifact_path"] = str(model_output.relative_to(ROOT))
    report["model_artifact_sha256"] = _sha256(model_output)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPANDED_OUTPUT_DIR)
    parser.add_argument("--highsteer-only", action="store_true",
                        help="exclude matching windows from broad dynamics training data")
    parser.add_argument("--residual-gain", type=float, default=GAIN)
    parser.add_argument("--feature-variant", choices=("base", "roll_acceleration"),
                        default="base")
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), not args.highsteer_only,
                 args.residual_gain, args.feature_variant)
    print(json.dumps({
        "output": str(args.output_dir.resolve() / "evaluation.json"),
        "candidate_decision": result["candidate_decision"],
        "five_second_highsteer": {
            key: value["macro_run_rmse"]["5.000s"]["hybrid_atlas"]
            for key, value in result["heldout_results"][
                "highsteer_validation"].items()
            if key in ("baseline", "lateral_v_residual")},
    }, indent=2, sort_keys=True))
