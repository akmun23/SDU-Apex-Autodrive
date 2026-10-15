#!/usr/bin/env python3
"""DAgger-train a causal correction to the existing high-steer v transition.

The teacher-forced HGB next-v head is accurate, but its direct recursive use
drifts. This fits a residual correction on its own predicted rollout states,
using only existing training captures and simulator-truth next-v labels.
Validation and practice runs are held out for final evaluation only.
"""

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

from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    EDSSM_CHECKPOINT,
    HGB_BUNDLE,
    HGB_TRAIN_DATASET,
    HIGHSTEER_DATASET,
    PRACTICE_DATASET,
    YAW_ATLAS_BUNDLE,
    _bundle_lags,
    _rollout as hybrid_rollout,
    _windows,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    HORIZON,
    _run_windows,
)
from tools.vehicle_dynamics_learning.fit_highsteer_augmented_longitudinal_candidate import (
    OUTPUT_DIR as LONGITUDINAL_CANDIDATE_DIR,
)
from tools.vehicle_dynamics_learning.fit_highsteer_lateral_acceleration_atlas_candidate import (
    HIGHSTEER_TRAIN_DATASET,
    LOCAL_GATE,
    _compare_recursive,
    _trace_metrics,
)
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_longitudinal_residual import (
    HIGHSTEER_ABS_STEERING_RAD,
    HIGHSTEER_SPEED_MPS,
    LONGITUDINAL_GATE,
    MODEL_OUTPUT as LONGITUDINAL_RESIDUAL_BUNDLE,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    _rmse,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


YAW_CORRECTOR_PATH = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
SOURCE_DATASETS = (HGB_TRAIN_DATASET, HIGHSTEER_TRAIN_DATASET)
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "dagger_hgb_v_increment_correction_latent_5iter_20261009")
ITERATIONS = 5
MAX_ITER = 180
MAX_LEAF_NODES = 15
MIN_SAMPLES_LEAF = 60
L2_REGULARIZATION = 2.0
SEED = 20261009


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_windows(data: dict[str, Any], hgb: dict[str, Any]
                      ) -> list[dict[str, int]]:
    candidates = _run_windows(
        data, "train", _bundle_lags(hgb), maximum_per_run=10_000)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    selected = []
    for row in candidates:
        start = int(row["start"])
        speed = float(np.hypot(rigid[start, 7], rigid[start, 8]))
        observed_speed = float(np.hypot(frames[start, 0], frames[start, 1]))
        steering = abs(float(frames[start, 3]))
        if (HIGHSTEER_SPEED_MPS[0] <= speed <= HIGHSTEER_SPEED_MPS[1]
                and HIGHSTEER_SPEED_MPS[0] <= observed_speed
                <= HIGHSTEER_SPEED_MPS[1]
                and HIGHSTEER_ABS_STEERING_RAD[0] <= steering
                <= HIGHSTEER_ABS_STEERING_RAD[1]):
            selected.append(row)
    if not selected:
        raise ValueError("no 5-second high-steer training rollouts are available")
    return selected


def _fit(features: np.ndarray, targets: np.ndarray,
         run_indices: np.ndarray, run_ids: np.ndarray
         ) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    if (features.ndim != 2 or targets.shape != (len(features),)
            or run_indices.shape != targets.shape or not len(features)
            or not np.isfinite(features).all()
            or not np.isfinite(targets).all()):
        raise ValueError("invalid DAgger HGB-v residual rows")
    runs = sorted(np.unique(run_indices).tolist())
    counts = {int(run): int(np.count_nonzero(run_indices == run)) for run in runs}
    weights = np.asarray([1.0 / counts[int(run)] for run in run_indices])
    weights *= len(weights) / weights.sum()
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=MAX_ITER,
        max_leaf_nodes=MAX_LEAF_NODES, min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=L2_REGULARIZATION, early_stopping=False,
        random_state=SEED)
    model.fit(features, targets, sample_weight=weights)
    residual = targets - model.predict(features)
    return model, {
        "rows": int(len(features)),
        "feature_count": int(features.shape[1]),
        "per_run_rows": {str(run_ids[run]): counts[run] for run in runs},
        "fit_rmse_mps": _rmse(residual),
        "target_rmse_mps": _rmse(targets),
        "target_bias_mps": float(np.mean(targets)),
        "sample_weighting": "equal total weight per independent training capture",
        "estimator_parameters": model.get_params(),
    }


def _rollout(data: dict[str, Any], windows: list[dict[str, int]],
             hgb: dict[str, Any], edssm: Any, metadata: dict[str, Any],
             atlas: dict[str, Any], yaw_model: Any, yaw_start_step: int,
             longitudinal_residual: dict[str, Any],
             correction: Any | None, capture_rows: bool,
             atlas_acceleration_source: str = "composed_transition"
             ) -> dict[str, Any]:
    result = hybrid_rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=1.0, include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source=atlas_acceleration_source,
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=yaw_start_step,
        hgb_main_gate=LONGITUDINAL_GATE,
        hgb_main_residual_models=longitudinal_residual["models"],
        hgb_main_residual_start_step=1,
        hgb_main_residual_gain=float(longitudinal_residual["gain"]),
        lateral_v_update="gated_hgb_v_increment",
        hgb_v_increment_residual_model=correction,
        hgb_v_increment_residual_start_step=1,
        hgb_v_increment_residual_gain=1.0,
        hgb_v_increment_residual_use_edssm_latent=True,
        capture_hgb_v_increment_residual_rows=capture_rows,
        capture_trajectory_trace=not capture_rows)
    trace = result.pop("_trajectory_trace", None)
    if capture_rows:
        return {key: result.pop(key) for key in (
            "_hgb_v_increment_residual_training_features",
            "_hgb_v_increment_residual_training_targets",
            "_hgb_v_increment_residual_training_run_indices")} | {
                "rollout_metadata": {key: value for key, value in result.items()
                                     if not key.startswith("_")}}
    if trace is None:
        raise AssertionError("recursive evaluation did not capture its trace")
    return {
        "rollout_metadata": {key: value for key, value in result.items()
                             if not key.startswith("_")},
        "trajectory_metrics": _trace_metrics(
            trace, np.asarray(data["run_ids"]).astype(str)),
        "windows_by_run": result["windows_by_run"],
    }


def run(threads: int = 2) -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    started = time.time()
    parent_hgb = joblib.load(HGB_BUNDLE)
    longitudinal_candidate_path = (
        LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib")
    longitudinal_candidate = joblib.load(longitudinal_candidate_path)
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

    train_sources: list[dict[str, Any]] = []
    global_train_ids: set[str] = set()
    for path in SOURCE_DATASETS:
        data = _load_dataset(path)
        windows = _training_windows(data, parent_hgb)
        run_ids = np.asarray(data["run_ids"]).astype(str)
        selected = {str(run_ids[row["run"]]) for row in windows}
        if global_train_ids & selected:
            raise ValueError("duplicate DAgger training capture across archives")
        global_train_ids.update(selected)
        train_sources.append({"path": path, "data": data, "windows": windows})
    global_run_ids = sorted(global_train_ids)
    global_run_index = {name: index for index, name in enumerate(global_run_ids)}

    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    iteration_reports: list[dict[str, Any]] = []
    correction: HistGradientBoostingRegressor | None = None
    fit_reports: list[dict[str, Any]] = []
    with threadpool_limits(limits=threads):
        for iteration in range(ITERATIONS):
            iteration_rows = 0
            per_source: dict[str, Any] = {}
            for source in train_sources:
                captured = _rollout(
                    source["data"], source["windows"], hgb, edssm, metadata,
                    atlas, yaw_model, yaw_start_step, longitudinal_residual,
                    correction, True)
                x = captured["_hgb_v_increment_residual_training_features"]
                y = captured["_hgb_v_increment_residual_training_targets"]
                local_runs = captured[
                    "_hgb_v_increment_residual_training_run_indices"]
                local_ids = np.asarray(source["data"]["run_ids"]).astype(str)
                row_names = local_ids[local_runs]
                global_runs = np.asarray(
                    [global_run_index[str(name)] for name in row_names],
                    dtype=np.int32)
                feature_parts.append(x)
                target_parts.append(y)
                run_parts.append(global_runs)
                iteration_rows += len(y)
                per_source[str(source["path"].relative_to(ROOT))] = {
                    "rows": int(len(y)),
                    "training_run_ids": sorted(set(row_names.tolist())),
                }
            correction, fit_report = _fit(
                np.concatenate(feature_parts), np.concatenate(target_parts),
                np.concatenate(run_parts), np.asarray(global_run_ids, dtype=str))
            fit_report["iteration"] = iteration
            fit_reports.append(fit_report)
            iteration_reports.append({
                "iteration": iteration,
                "captured_rows_this_iteration": iteration_rows,
                "sources": per_source,
            })

    validation = _load_dataset(HIGHSTEER_DATASET)
    practice = _load_dataset(PRACTICE_DATASET)
    evaluation: dict[str, Any] = {}
    for label, data, path, split in (
            ("highsteer_validation", validation, HIGHSTEER_DATASET, "validation"),
            ("unseen_practice", practice, PRACTICE_DATASET, "unseen_practice")):
        if split not in set(np.asarray(data["splits"]).astype(str)):
            raise ValueError(f"missing held-out split {split!r}")
        windows = _evaluation_windows(data, path, parent_hgb)
        baseline = _rollout(
            data, windows, hgb, edssm, metadata, atlas, yaw_model,
            yaw_start_step, longitudinal_residual, None, False)
        candidate = _rollout(
            data, windows, hgb, edssm, metadata, atlas, yaw_model,
            yaw_start_step, longitudinal_residual, correction, False)
        evaluation[label] = {
            "split": split,
            "baseline_direct_hgb_v": baseline,
            "dagger_corrected_hgb_v": candidate,
            "paired_candidate_minus_baseline": _compare_recursive(
                candidate, baseline),
        }
        if label == "highsteer_validation":
            alt_baseline = _rollout(
                data, windows, hgb, edssm, metadata, atlas, yaw_model,
                yaw_start_step, longitudinal_residual, None, False,
                atlas_acceleration_source="model_heads")
            alt_candidate = _rollout(
                data, windows, hgb, edssm, metadata, atlas, yaw_model,
                yaw_start_step, longitudinal_residual, correction, False,
                atlas_acceleration_source="model_heads")
            evaluation["highsteer_validation_model_heads_acceleration"] = {
                "baseline_direct_hgb_v": alt_baseline,
                "dagger_corrected_hgb_v": alt_candidate,
                "paired_candidate_minus_baseline": _compare_recursive(
                    alt_candidate, alt_baseline),
            }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    model_path = OUTPUT_DIR / "dagger_hgb_v_residual.joblib"
    joblib.dump({
        "research_only": True,
        "production_integration_allowed": False,
        "target": "truth_next_v - (predicted_current_v + frozen_HGB_v_increment)",
        "support_gate": LOCAL_GATE,
        "feature_contract": parent_hgb["feature_names"],
        "appended_feature": "current_predicted_EDSSM_latent_state",
        "history_lags_steps": parent_hgb["history_lags_steps"],
        "training_run_ids": global_run_ids,
        "iterations": ITERATIONS,
        "model": correction,
    }, model_path)
    report = {
        "schema_version": 1,
        "purpose": "reduce recursive state-distribution error in the existing high-steer next-v head",
        "training_sources": [str(path.relative_to(ROOT)) for path in SOURCE_DATASETS],
        "training_source_sha256": {str(path.relative_to(ROOT)): _sha256(path)
                                   for path in SOURCE_DATASETS},
        "training_run_ids": global_run_ids,
        "training_windows_by_source": {
            str(source["path"].relative_to(ROOT)): {
                str(np.asarray(source["data"]["run_ids"]).astype(str)[run]): int(
                    sum(row["run"] == run for row in source["windows"]))
                for run in sorted({row["run"] for row in source["windows"]})}
            for source in train_sources},
        "dagger_iterations": iteration_reports,
        "fit_reports": fit_reports,
        "candidate_model": str(model_path.relative_to(ROOT)),
        "heldout_recursive_evaluations": evaluation,
        "test_and_final_test_used": False,
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "only 7-9 m/s and 0.30-0.525 rad high-steer support is corrected",
            "labels are simulator-truth supervised; no future truth enters rollout inputs",
            "three high-steer validation runs and two practice runs remain limited run-level validation cohorts",
            "five-second windows are not full-lap offline-simulator validation",
            "this is not integrated into odometry or MPC",
        ],
    }
    (OUTPUT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _evaluation_windows(data: dict[str, Any], dataset_path: Path,
                        hgb: dict[str, Any]) -> list[dict[str, int]]:
    return _windows(data, dataset_path, HORIZON, _bundle_lags(hgb))


def main() -> None:
    report = run()
    summary = {}
    for split, result in report["heldout_recursive_evaluations"].items():
        key = "dagger_corrected_hgb_v"
        base_metrics = result["baseline_direct_hgb_v"]["trajectory_metrics"]
        candidate_metrics = result[key]["trajectory_metrics"]
        summary[split] = {
            horizon: {
                "baseline_v_rmse": base_metrics[horizon]["macro_run"][
                    "v_rmse_mps"],
                "candidate_v_rmse": candidate_metrics[horizon]["macro_run"][
                    "v_rmse_mps"],
                "baseline_position_rmse": base_metrics[horizon]["macro_run"][
                    "position_rmse_m"],
                "candidate_position_rmse": candidate_metrics[horizon][
                    "macro_run"]["position_rmse_m"],
            }
            for horizon in base_metrics}
    print(json.dumps({"output": str(OUTPUT_DIR / "evaluation.json"),
                      "results": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
