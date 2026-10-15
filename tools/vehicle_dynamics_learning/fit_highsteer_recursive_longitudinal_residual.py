#!/usr/bin/env python3
"""DAgger-fit high-steer u/rear-wheel residuals around frozen one-step heads."""

from __future__ import annotations

import hashlib
import json
import time
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
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HIGHSTEER_ABS_STEERING_RAD,
    HIGHSTEER_SPEED_MPS,
    HIGHSTEER_TRAIN_DATASET,
    HighSteerYawResidual,
    _highsteer_windows,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as yaw_atlas


LONGITUDINAL_CANDIDATE = (
    LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib")
YAW_CORRECTOR_PATH = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_recursive_longitudinal_residual_gain0p5_training_20261009/"
    "evaluation.json")
MODEL_OUTPUT = OUTPUT.with_name("u_filtered_wheel_residual_models.joblib")
ITERATIONS = 2
SEED = 20261009
MAX_ITER = 180
MAX_LEAF_NODES = 15
MIN_SAMPLES_LEAF = 40
L2_REGULARIZATION = 2.0
RESIDUAL_GAIN = 0.5
LONGITUDINAL_GATE = {
    "speed_mps": tuple(HIGHSTEER_SPEED_MPS),
    "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(features: np.ndarray, targets: np.ndarray,
         run_indices: np.ndarray, run_ids: np.ndarray
         ) -> tuple[list[HistGradientBoostingRegressor], dict[str, Any]]:
    if (features.ndim != 2 or targets.shape != (len(features), 3)
            or run_indices.shape != (len(features),)
            or not len(features) or not np.isfinite(features).all()
            or not np.isfinite(targets).all()):
        raise ValueError("invalid DAgger u/wheel residual training rows")
    counts = np.bincount(run_indices, minlength=len(run_ids)).astype(np.float64)
    if np.any(counts[np.unique(run_indices)] <= 0):
        raise ValueError("a selected high-steer run has no residual training rows")
    sample_weight = 1.0 / counts[run_indices]
    sample_weight *= len(sample_weight) / sample_weight.sum()
    models = []
    per_head: dict[str, Any] = {}
    for index, name in enumerate(("u_delta_m", "rear_left_delta_mps",
                                  "rear_right_delta_mps")):
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=MAX_ITER,
            max_leaf_nodes=MAX_LEAF_NODES,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            l2_regularization=L2_REGULARIZATION,
            early_stopping=False, random_state=SEED + index)
        model.fit(features, targets[:, index], sample_weight=sample_weight)
        prediction = model.predict(features)
        per_head[name] = {
            "target_rmse_training": float(np.sqrt(np.mean(targets[:, index] ** 2))),
            "corrected_residual_rmse_training": float(np.sqrt(np.mean(
                (targets[:, index] - prediction) ** 2))),
            "target_p95_abs_training": float(np.quantile(
                np.abs(targets[:, index]), 0.95)),
        }
        models.append(model)
    return models, {
        "rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "run_ids": sorted({str(run_ids[index]) for index in np.unique(run_indices)}),
        "run_counts": {
            str(run_ids[index]): int(np.count_nonzero(run_indices == index))
            for index in np.unique(run_indices)},
        "sample_weighting": "equal total weight per high-steer training run",
        "per_head": per_head,
        "estimator_parameters": models[0].get_params(),
    }


def _capture(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
             metadata: dict[str, Any], atlas: dict[str, Any],
             yaw_model: Any, windows: list[dict[str, int]],
             residual_models: list[Any] | None
             ) -> dict[str, Any]:
    return _rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=1.0,
        include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=CORRECTION_START_STEP,
        hgb_main_gate=LONGITUDINAL_GATE,
        hgb_main_residual_models=residual_models,
        hgb_main_residual_start_step=1,
        hgb_main_residual_gain=RESIDUAL_GAIN,
        capture_hgb_main_residual_rows=True)


def _training_capture(data: dict[str, Any], hgb: dict[str, Any],
                      edssm: Any, metadata: dict[str, Any], atlas: dict[str, Any],
                      yaw_model: Any, windows: list[dict[str, int]],
                      run_id_to_index: dict[str, int],
                      residual_models: list[Any] | None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    captured = _capture(data, hgb, edssm, metadata, atlas, yaw_model,
                        windows, residual_models)
    features = captured.pop("_hgb_main_residual_training_features")
    targets = captured.pop("_hgb_main_residual_training_targets")
    local_runs = captured.pop("_hgb_main_residual_training_run_indices")
    data_ids = np.asarray(data["run_ids"]).astype(str)
    try:
        global_runs = np.asarray(
            [run_id_to_index[str(data_ids[index])] for index in local_runs],
            dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"unexpected training capture {exc.args[0]!r}") from exc
    captured["captured_rows"] = int(len(features))
    captured["training_run_ids"] = sorted({
        str(data_ids[row["run"]]) for row in windows})
    return features, targets, global_runs, captured


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    model_output = output_path.with_name(MODEL_OUTPUT.name)
    if output_path.exists() or model_output.exists():
        raise FileExistsError("refusing to overwrite a prior residual result")
    required = (
        HGB_BUNDLE, LONGITUDINAL_CANDIDATE, YAW_CORRECTOR_PATH,
        EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE, HGB_TRAIN_DATASET,
        HIGHSTEER_TRAIN_DATASET, HIGHSTEER_DATASET, PRACTICE_DATASET,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    parent_hgb = joblib.load(HGB_BUNDLE)
    longitudinal_candidate = joblib.load(LONGITUDINAL_CANDIDATE)
    hgb = dict(parent_hgb)
    hgb["models"] = longitudinal_candidate["models"]
    hgb["training_run_ids"] = longitudinal_candidate["training_run_ids"]
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for expert in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(expert, "n_jobs"):
            expert.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")

    training_data = _load_dataset(HIGHSTEER_TRAIN_DATASET)
    windows, wheel_mean_bounds = _highsteer_windows(
        training_data, WINDOWS_PER_RUN, _bundle_lags(hgb))
    train_ids = np.asarray(training_data["run_ids"]).astype(str)
    splits = np.asarray(training_data["splits"]).astype(str)
    selected_run_ids = sorted({str(train_ids[row["run"]]) for row in windows})
    if not selected_run_ids or any(
            splits[row["run"]] != "train" for row in windows):
        raise ValueError("longitudinal residual fitting must use high-steer train runs only")
    run_id_to_index = {run_id: index for index, run_id in enumerate(selected_run_ids)}
    gate = {
        "speed_mps": tuple(HIGHSTEER_SPEED_MPS),
        "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD),
    }

    feature_sets: list[np.ndarray] = []
    target_sets: list[np.ndarray] = []
    run_sets: list[np.ndarray] = []
    generation_reports: list[dict[str, Any]] = []
    residual_models: list[HistGradientBoostingRegressor] | None = None
    fit_reports: list[dict[str, Any]] = []
    with threadpool_limits(limits=2):
        for iteration in range(ITERATIONS):
            features, targets, run_indices, capture_report = _training_capture(
                training_data, hgb, edssm, metadata, atlas, yaw_model,
                windows, run_id_to_index, residual_models)
            feature_sets.append(features)
            target_sets.append(targets)
            run_sets.append(run_indices)
            generation_reports.append({
                "iteration": iteration,
                "generated_with_previous_residual": residual_models is not None,
                "captured": capture_report,
                "residual_target_rmse_by_output": {
                    name: float(np.sqrt(np.mean(targets[:, index] ** 2)))
                    for index, name in enumerate((
                        "u_delta_m", "rear_left_delta_mps",
                        "rear_right_delta_mps"))},
            })
            aggregate_features = np.concatenate(feature_sets, axis=0)
            aggregate_targets = np.concatenate(target_sets, axis=0)
            aggregate_runs = np.concatenate(run_sets, axis=0)
            residual_models, fit_report = _fit(
                aggregate_features, aggregate_targets, aggregate_runs,
                np.asarray(selected_run_ids, dtype=str))
            fit_reports.append(fit_report)

    if residual_models is None:
        raise AssertionError("no recursive longitudinal residual fit was produced")

    # Fit captures above are training-only. Score matched held-out whole runs.
    base_validation = _load_dataset(HGB_TRAIN_DATASET)
    whole_windows = _run_windows(
        base_validation, "validation", _bundle_lags(hgb), WINDOWS_PER_RUN)
    highsteer_validation = _load_dataset(HIGHSTEER_DATASET)
    highsteer_windows = _windows(
        highsteer_validation, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))
    practice = _load_dataset(PRACTICE_DATASET)
    practice_windows = _windows(
        practice, PRACTICE_DATASET, HORIZON, _bundle_lags(hgb))
    evaluation_sets = (
        ("whole_run_validation", base_validation, whole_windows, "validation"),
        ("highsteer_validation", highsteer_validation, highsteer_windows,
         "validation"),
        ("unseen_practice_diagnostic", practice, practice_windows,
         "unseen_practice"),
    )
    heldout: dict[str, Any] = {}
    for name, data, selected_windows, expected_split in evaluation_sets:
        data_ids = np.asarray(data["run_ids"]).astype(str)
        selected_ids = {str(data_ids[row["run"]]) for row in selected_windows}
        data_splits = np.asarray(data["splits"]).astype(str)
        if (not selected_ids or selected_ids & set(selected_run_ids)
                or any(data_splits[row["run"]] != expected_split
                       for row in selected_windows)):
            raise ValueError(f"held-out split/run leakage in {name}")
        baseline = _rollout(
            data, hgb, edssm, metadata, atlas, selected_windows, HORIZON,
            yaw_delta_gain=1.0, include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=CORRECTION_START_STEP,
            hgb_main_gate=gate)
        candidate = _rollout(
            data, hgb, edssm, metadata, atlas, selected_windows, HORIZON,
            yaw_delta_gain=1.0, include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=CORRECTION_START_STEP,
            hgb_main_gate=gate,
            hgb_main_residual_models=residual_models,
            hgb_main_residual_start_step=1,
            hgb_main_residual_gain=RESIDUAL_GAIN)
        heldout[name] = {
            "heldout_run_ids": sorted(candidate["per_run"]),
            "windows": candidate["windows"],
            "windows_by_run": candidate["windows_by_run"],
            "gate_active_fraction": candidate[
                "hgb_main_gate_active_fraction"],
            "baseline_augmented_heads_with_gated_yaw": {
                "macro_run_rmse": baseline["macro_run_rmse"],
                "per_run": baseline["per_run"],
            },
            "dagger_residual_with_gated_yaw": {
                "macro_run_rmse": candidate["macro_run_rmse"],
                "per_run": candidate["per_run"],
                "initial_yaw_step": candidate[
                    "yaw_atlas_one_step_selected_windows"],
            },
        }

    high_base = heldout["highsteer_validation"][
        "baseline_augmented_heads_with_gated_yaw"]["macro_run_rmse"]
    high_candidate = heldout["highsteer_validation"][
        "dagger_residual_with_gated_yaw"]["macro_run_rmse"]
    high_improved = all(
        high_candidate["5.000s"]["hybrid_atlas"][metric]
        < high_base["5.000s"]["hybrid_atlas"][metric]
        for metric in ("u_rmse_mps", "filtered_rear_wheel_pair_rmse_mps"))
    high_not_materially_worse = all(
        high_candidate["5.000s"]["hybrid_atlas"][metric]
        <= 1.10 * high_base["5.000s"]["hybrid_atlas"][metric]
        for metric in ("position_rmse_m", "yaw_rate_rmse_rps"))
    other_not_materially_worse = all(
        heldout[name]["dagger_residual_with_gated_yaw"]["macro_run_rmse"]
        ["5.000s"]["hybrid_atlas"][metric]
        <= 1.10 * heldout[name]["baseline_augmented_heads_with_gated_yaw"]
        ["macro_run_rmse"]["5.000s"]["hybrid_atlas"][metric]
        for name in ("whole_run_validation", "unseen_practice_diagnostic")
        for metric in ("position_rmse_m", "yaw_rate_rmse_rps"))

    report = {
        "schema_version": 1,
        "purpose": "correct rollout-distribution error in u and filtered rear-wheel transitions in the measured high-steer regime using a gain-consistent DAgger rollout; keep yaw atlas and first measured transition fixed",
        "research_only": True,
        "simulator_was_run": False,
        "production_or_runtime_integration": False,
        "test_or_final_test_used": False,
        "fit_split": "two high-steer training runs only",
        "training_run_ids": selected_run_ids,
        "training_window_count": len(windows),
        "training_windows": [
            {"run_id": str(train_ids[row["run"]]),
             "sequence": row["sequence"], "start": row["start"]}
            for row in windows],
        "training_support": {
            "body_speed_start_mps": list(HIGHSTEER_SPEED_MPS),
            "absolute_steering_rad": list(HIGHSTEER_ABS_STEERING_RAD),
            "rear_wheel_mean_mps_in_measured_support": list(wheel_mean_bounds),
            "rollout_gate": {name: list(bounds) for name, bounds in gate.items()},
            "gate_state_source": "current causal predicted body u/v and steering feedback",
        },
        "frozen_components": {
            "yaw_atlas_path": str(YAW_ATLAS_BUNDLE.relative_to(ROOT)),
            "yaw_corrector_path": str(YAW_CORRECTOR_PATH.relative_to(ROOT)),
            "yaw_first_transition_uses_frozen_atlas": True,
            "longitudinal_base_heads": str(LONGITUDINAL_CANDIDATE.relative_to(ROOT)),
            "actuator_and_raw_encoder_heads_refit": False,
        },
        "dagger_iterations": ITERATIONS,
        "residual_gain_used_during_dagger_and_evaluation": RESIDUAL_GAIN,
        "correction_start_step": 1,
        "training_generation": generation_reports,
        "fit_reports": fit_reports,
        "heldout_results": heldout,
        "candidate_decision": {
            "criteria": {
                "highsteer_5s_u_and_filtered_wheels_both_improve": high_improved,
                "highsteer_5s_position_and_yaw_no_more_than_10pct_worse":
                    high_not_materially_worse,
                "whole_run_and_practice_5s_position_and_yaw_no_more_than_10pct_worse":
                    other_not_materially_worse,
            },
            "retained_as_research_candidate": bool(
                high_improved and high_not_materially_worse
                and other_not_materially_worse),
            "production_or_runtime_integration": False,
        },
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "limitations": [
            "high-steer residual is supported by only two independent training runs",
            "validation contains three high-steer runs and two practice diagnostics",
            "the evaluation horizon is five seconds and does not certify a complete-lap offline simulator",
        ],
        "runtime_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report["candidate_decision"]["retained_as_research_candidate"]:
        joblib.dump({
            "schema_version": 1,
            "models": residual_models,
            "gate": {name: list(bounds) for name, bounds in gate.items()},
            "correction_start_step": 1,
            "gain": RESIDUAL_GAIN,
            "training_run_ids": selected_run_ids,
            "research_only": True,
            "production_or_runtime_integration": False,
        }, model_output)
        report["model_artifact_written"] = True
        report["model_artifact_path"] = str(model_output.relative_to(ROOT))
        report["model_artifact_sha256"] = _sha256(model_output)
    else:
        report["model_artifact_written"] = False
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "candidate_decision": result["candidate_decision"],
        "highsteer_5s": {
            name: row[name]["macro_run_rmse"]["5.000s"]["hybrid_atlas"]
            for name in ("baseline_augmented_heads_with_gated_yaw",
                         "dagger_residual_with_gated_yaw")
            for row in (result["heldout_results"]["highsteer_validation"],)},
    }, indent=2, sort_keys=True))
