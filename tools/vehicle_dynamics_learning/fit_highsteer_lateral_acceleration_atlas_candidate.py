#!/usr/bin/env python3
"""Fit and test a local lateral-acceleration teacher in existing high-steer data.

The frozen yaw atlas and longitudinal transition remain unchanged. Training
uses only valid, fixed-25-ms training transitions from the existing OpenPlane
captures. Simulator truth labels body-frame ``a_y``; at rollout the candidate
receives only the causal predicted state/history and its current predicted
roll state. This is an offline research candidate, not a runtime integration.
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
    _rollout,
    _windows,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    HORIZON,
)
from tools.vehicle_dynamics_learning.fit_highsteer_augmented_longitudinal_candidate import (
    OUTPUT_DIR as LONGITUDINAL_CANDIDATE_DIR,
)
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_longitudinal_residual import (
    HIGHSTEER_ABS_STEERING_RAD,
    HIGHSTEER_SPEED_MPS,
    LONGITUDINAL_GATE,
    MODEL_OUTPUT as LONGITUDINAL_RESIDUAL_BUNDLE,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DT_S,
    _rmse,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    _augment,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


HIGHSTEER_TRAIN_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/highsteer_75_train_expanded_20261004/"
    "encoder_fixed40hz_v1/openplane_dynamics_raw_wheels_fixed40hz.npz")
YAW_CORRECTOR_PATH = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_lateral_acceleration_atlas_20261009")
OUTPUT_REPORT = OUTPUT_DIR / "evaluation.json"
SEED = 20261009
MAX_ITER = 220
MAX_LEAF_NODES = 31
MIN_SAMPLES_LEAF = 80
L2_REGULARIZATION = 3.0
LOCAL_GATE = {
    "speed_mps": tuple(HIGHSTEER_SPEED_MPS),
    "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect(data: dict[str, Any], split: str, hgb: dict[str, Any],
             feature_variant: str
             ) -> dict[str, np.ndarray]:
    """Build exactly the causal transition features used by the hybrid replay."""
    source, run_index, _, _, _, _, _, variants, _ = _augment(data, split)
    selected_name = str(hgb["selected_parent_feature_variant"])
    selected_index = BASE_VARIANT_NAMES.index(selected_name)
    base_x = variants[selected_index]
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    state = rigid[:, (7, 8, 12)]
    speed = np.hypot(state[source, 0], state[source, 1])
    steering = np.abs(frames[source, 3])
    attitude = np.asarray(data["imu_attitude_frames"], dtype=np.float64)
    attitude_valid = np.asarray(data["imu_attitude_valid"], dtype=bool)
    sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)
    domain = (
        (speed >= HIGHSTEER_SPEED_MPS[0])
        & (speed <= HIGHSTEER_SPEED_MPS[1])
        & (steering >= HIGHSTEER_ABS_STEERING_RAD[0])
        & (steering <= HIGHSTEER_ABS_STEERING_RAD[1])
        & attitude_valid[source] & sensor_valid[source]
        & np.isfinite(attitude[source][:, (0, 2)]).all(axis=1)
        & np.isfinite(state[source + 1]).all(axis=1))
    source = source[domain]
    run_index = run_index[domain]
    base_x = base_x[domain]
    gt_roll = attitude[source][:, (0, 2)].astype(np.float32)
    if feature_variant == "roll":
        x = np.column_stack((base_x, gt_roll)).astype(np.float32)
    elif feature_variant == "base":
        x = base_x.astype(np.float32)
    else:
        raise ValueError(f"unsupported feature variant {feature_variant!r}")

    dt = np.asarray(data["dt_s"], dtype=np.float64)[source + 1]
    if not np.isclose(dt, DT_S, rtol=0.0, atol=1e-7).all():
        raise ValueError("transition source admitted a non-25-ms target")
    u_mid = 0.5 * (state[source, 0] + state[source + 1, 0])
    r_mid = 0.5 * (state[source, 2] + state[source + 1, 2])
    ay = ((state[source + 1, 1] - state[source, 1]) / dt + u_mid * r_mid)
    if not (np.isfinite(x).all() and np.isfinite(ay).all()):
        raise ValueError("non-finite feature or lateral-acceleration label")
    return {
        "source": source,
        "run_index": run_index,
        "x": x,
        "base_x": base_x,
        "target_ay": ay.astype(np.float32),
        "u_mid": u_mid,
        "r_mid": r_mid,
        "speed": speed[domain],
        "steering": steering[domain],
        "attitude": gt_roll,
    }


def _fit(x: np.ndarray, target: np.ndarray, run_index: np.ndarray,
         run_ids: np.ndarray) -> tuple[HistGradientBoostingRegressor, dict[str, Any]]:
    if (x.ndim != 2 or target.shape != (len(x),)
            or run_index.shape != target.shape or not len(x)
            or not np.isfinite(x).all() or not np.isfinite(target).all()):
        raise ValueError("invalid lateral-acceleration training matrix")
    runs = sorted(np.unique(run_index).tolist())
    counts = {int(run): int(np.count_nonzero(run_index == run)) for run in runs}
    weights = np.asarray([1.0 / counts[int(run)] for run in run_index])
    weights *= len(weights) / weights.sum()
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.04, max_iter=MAX_ITER,
        max_leaf_nodes=MAX_LEAF_NODES, min_samples_leaf=MIN_SAMPLES_LEAF,
        l2_regularization=L2_REGULARIZATION, early_stopping=False,
        random_state=SEED)
    model.fit(x, target, sample_weight=weights)
    residual = target - model.predict(x)
    report = {
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "training_runs": {
            str(run_ids[run]): counts[run] for run in runs},
        "training_run_count": len(runs),
        "training_target_rmse_mps2": _rmse(target),
        "training_fit_rmse_mps2": _rmse(residual),
        "training_fit_bias_mps2": float(-np.mean(residual)),
        "sample_weighting": "equal total weight per independent training capture",
        "estimator_parameters": model.get_params(),
    }
    return model, report


def _metric_rows(prediction: np.ndarray, target: np.ndarray,
                 run_index: np.ndarray, run_ids: np.ndarray
                 ) -> dict[str, Any]:
    per_run: dict[str, Any] = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        error = prediction[mask] - target[mask]
        per_run[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            "rmse": _rmse(error),
            "bias": float(np.mean(error)),
            "p95_abs": float(np.quantile(np.abs(error), 0.95)),
        }
    return {
        "per_run": per_run,
        "macro_run_rmse": float(np.mean(
            [row["rmse"] for row in per_run.values()])),
        "pooled_rmse": _rmse(prediction - target),
        "macro_run_bias": float(np.mean(
            [row["bias"] for row in per_run.values()])),
        "pooled_p95_abs": float(np.quantile(np.abs(prediction - target), 0.95)),
    }


def _paired_run_bootstrap(candidate: dict[str, Any], baseline: dict[str, Any],
                          metric: str = "rmse") -> dict[str, Any]:
    common = sorted(set(candidate["per_run"]) & set(baseline["per_run"]))
    if len(common) < 2:
        return {"runs": common, "mean_candidate_minus_baseline": None,
                "paired_run_bootstrap_95pct": None}
    delta = np.asarray([
        candidate["per_run"][run][metric] - baseline["per_run"][run][metric]
        for run in common], dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(delta, size=(20000, len(delta)), replace=True).mean(axis=1)
    return {
        "runs": common,
        "mean_candidate_minus_baseline": float(np.mean(delta)),
        "paired_run_bootstrap_95pct": [
            float(value) for value in np.quantile(samples, (0.025, 0.975))],
        "per_run_candidate_minus_baseline": {
            run: float(candidate["per_run"][run][metric]
                       - baseline["per_run"][run][metric])
            for run in common},
    }


def _one_step(data: dict[str, Any], split: str, hgb: dict[str, Any],
              model: Any, feature_variant: str) -> dict[str, Any]:
    rows = _collect(data, split, hgb, feature_variant)
    if not len(rows["source"]):
        return {
            "transitions": 0,
            "status": "no transitions in the candidate's measured support",
            "model_feature_variant": feature_variant,
            "support_gate": LOCAL_GATE,
        }
    run_ids = np.asarray(data["run_ids"]).astype(str)
    predicted_ay = model.predict(rows["x"])
    parent_dv = hgb["recursive_aux_models"][0].predict(rows["base_x"])
    parent_ay = parent_dv / DT_S + rows["u_mid"] * rows["r_mid"]
    state = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    v_now = state[rows["source"], 8]
    v_next = state[rows["source"] + 1, 8]
    v_candidate = v_now + DT_S * (
        predicted_ay - rows["u_mid"] * rows["r_mid"])
    v_parent = v_now + parent_dv
    scores = {
        "lateral_acceleration_candidate": _metric_rows(
            predicted_ay, rows["target_ay"], rows["run_index"], run_ids),
        "parent_v_head_implied_lateral_acceleration": _metric_rows(
            parent_ay, rows["target_ay"], rows["run_index"], run_ids),
        "candidate_reconstructed_v": _metric_rows(
            v_candidate, v_next, rows["run_index"], run_ids),
        "parent_v_increment": _metric_rows(
            v_parent, v_next, rows["run_index"], run_ids),
        "v_persistence": _metric_rows(
            v_now, v_next, rows["run_index"], run_ids),
    }
    # Fixed descriptive cells expose where the local fit still struggles; they
    # are reporting bins, not claimed physical boundaries.
    cells: dict[str, Any] = {}
    for speed_low, speed_high in ((7.0, 8.0), (8.0, 9.0)):
        for steer_low, steer_high in ((0.30, 0.40), (0.40, 0.50), (0.50, 0.525)):
            mask = ((rows["speed"] >= speed_low)
                    & (rows["speed"] < speed_high if speed_high < 9.0
                       else rows["speed"] <= speed_high)
                    & (rows["steering"] >= steer_low)
                    & (rows["steering"] < steer_high if steer_high < 0.525
                       else rows["steering"] <= steer_high))
            if np.any(mask):
                errors = predicted_ay[mask] - rows["target_ay"][mask]
                cells[f"speed_{speed_low:g}_{speed_high:g}_steer_"
                      f"{steer_low:.3f}_{steer_high:.3f}"] = {
                    "transitions": int(mask.sum()),
                    "ay_rmse_mps2": _rmse(errors),
                    "ay_bias_mps2": float(np.mean(errors)),
                    "ay_p95_abs_mps2": float(np.quantile(np.abs(errors), 0.95)),
                }
    return {
        "transitions": int(len(rows["source"])),
        "model_feature_variant": feature_variant,
        "metrics": scores,
        "paired_candidate_minus_parent_ay": _paired_run_bootstrap(
            scores["lateral_acceleration_candidate"],
            scores["parent_v_head_implied_lateral_acceleration"]),
        "paired_candidate_minus_parent_v": _paired_run_bootstrap(
            scores["candidate_reconstructed_v"], scores["parent_v_increment"]),
        "descriptive_regime_cells": cells,
    }


def _trace_metrics(trace: dict[str, np.ndarray], run_ids: np.ndarray,
                   endpoints: tuple[int, ...] = (1, 30, 80, 200)
                   ) -> dict[str, Any]:
    predicted = np.asarray(trace["predicted_state"], dtype=np.float64)
    truth = np.asarray(trace["truth_state"], dtype=np.float64)
    predicted_pose = np.asarray(trace["predicted_pose"], dtype=np.float64)
    truth_pose = np.asarray(trace["truth_pose"], dtype=np.float64)
    run_index = np.asarray(trace["run_indices"], dtype=np.int32)
    result: dict[str, Any] = {}
    for endpoint in endpoints:
        per_run: dict[str, Any] = {}
        for run in sorted(np.unique(run_index).tolist()):
            mask = run_index == run
            state_error = predicted[mask, endpoint - 1] - truth[mask, endpoint - 1]
            pose_error = predicted_pose[mask, endpoint - 1] - truth_pose[mask, endpoint - 1]
            pose_error[:, 2] = np.arctan2(np.sin(pose_error[:, 2]),
                                          np.cos(pose_error[:, 2]))
            per_run[str(run_ids[run])] = {
                "windows": int(mask.sum()),
                "u_rmse_mps": _rmse(state_error[:, 0]),
                "v_rmse_mps": _rmse(state_error[:, 1]),
                "yaw_rate_rmse_rps": _rmse(state_error[:, 2]),
                "position_rmse_m": _rmse(pose_error[:, :2]),
                "yaw_pose_rmse_rad": _rmse(pose_error[:, 2]),
            }
        metrics = tuple(next(iter(per_run.values())))
        macro = {name: float(np.mean([row[name] for row in per_run.values()]))
                 for name in metrics if name != "windows"}
        result[f"{endpoint * DT_S:.3f}s"] = {
            "per_run": per_run, "macro_run": macro,
        }
    return result


def _recursive(data: dict[str, Any], dataset_path: Path,
               hgb: dict[str, Any], edssm: Any, metadata: dict[str, Any],
               atlas: dict[str, Any], yaw_model: Any, yaw_start_step: int,
               longitudinal_residual: dict[str, Any],
               lateral_model: Any | None, feature_variant: str,
               lateral_v_update: str = "edssm",
               atlas_acceleration_source: str = "composed_transition",
               ) -> dict[str, Any]:
    windows = _windows(data, dataset_path, HORIZON, _bundle_lags(hgb))
    args = dict(
        yaw_delta_gain=1.0, include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source=atlas_acceleration_source,
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=yaw_start_step,
        hgb_main_gate=LONGITUDINAL_GATE,
        hgb_main_residual_models=longitudinal_residual["models"],
        hgb_main_residual_start_step=1,
        hgb_main_residual_gain=float(longitudinal_residual["gain"]),
        capture_trajectory_trace=True)
    if lateral_model is None:
        rollout = _rollout(data, hgb, edssm, metadata, atlas, windows,
                           HORIZON, lateral_v_update=lateral_v_update, **args)
    else:
        rollout = _rollout(
            data, hgb, edssm, metadata, atlas, windows, HORIZON,
            lateral_v_update="gated_lateral_acceleration",
            lateral_acceleration_model=lateral_model,
            lateral_acceleration_gate=LOCAL_GATE,
            lateral_acceleration_feature_variant=feature_variant,
            **args)
    trace = rollout.pop("_trajectory_trace")
    run_ids = np.asarray(data["run_ids"]).astype(str)
    return {
        "rollout_metadata": {key: value for key, value in rollout.items()
                             if not key.startswith("_")},
        "trajectory_metrics": _trace_metrics(trace, run_ids),
        "windows_by_run": rollout["windows_by_run"],
    }


def _compare_recursive(candidate: dict[str, Any], baseline: dict[str, Any]
                       ) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon, candidate_data in candidate["trajectory_metrics"].items():
        base_data = baseline["trajectory_metrics"][horizon]
        per_metric: dict[str, Any] = {}
        metric_names = tuple(next(iter(candidate_data["per_run"].values())))
        for metric in metric_names:
            if metric == "windows":
                continue
            c = {"per_run": {run: {"rmse": row[metric]}
                              for run, row in candidate_data["per_run"].items()}}
            b = {"per_run": {run: {"rmse": row[metric]}
                              for run, row in base_data["per_run"].items()}}
            per_metric[metric] = _paired_run_bootstrap(c, b)
        result[horizon] = per_metric
    return result


def run(output_dir: Path = OUTPUT_DIR, threads: int = 2) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing result directory {output_dir}")
    required = (HGB_BUNDLE, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
                YAW_CORRECTOR_PATH, LONGITUDINAL_CANDIDATE_DIR
                / "longitudinal_raw_wheel_candidate.joblib",
                LONGITUDINAL_RESIDUAL_BUNDLE, HGB_TRAIN_DATASET,
                HIGHSTEER_TRAIN_DATASET, HIGHSTEER_DATASET, PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()

    parent_hgb = joblib.load(HGB_BUNDLE)
    longitudinal_candidate = joblib.load(required[4])
    hgb = dict(parent_hgb)
    hgb["models"] = longitudinal_candidate["models"]
    hgb["training_run_ids"] = longitudinal_candidate["training_run_ids"]
    longitudinal_residual = joblib.load(LONGITUDINAL_RESIDUAL_BUNDLE)
    if not np.isclose(float(longitudinal_residual["gain"]), 0.5):
        raise ValueError("the evaluated longitudinal residual must remain at gain 0.5")
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"], tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    yaw_start_step = int(yaw_bundle.get(
        "correction_start_step", CORRECTION_START_STEP))
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")

    x_by_variant: dict[str, list[np.ndarray]] = {"base": [], "roll": []}
    y_parts: list[np.ndarray] = []
    training_row_ids: list[np.ndarray] = []
    training_run_ids: list[str] = []
    training_rows_by_run: dict[str, int] = {}
    sources = (HGB_TRAIN_DATASET, HIGHSTEER_TRAIN_DATASET)
    seen_ids: set[str] = set()
    with threadpool_limits(limits=threads):
        for dataset_path in sources:
            data = _load_dataset(dataset_path)
            rows_base = _collect(data, "train", parent_hgb, "base")
            rows_roll = _collect(data, "train", parent_hgb, "roll")
            ids = np.asarray(data["run_ids"]).astype(str)
            row_ids = ids[rows_base["run_index"]]
            for run_id in np.unique(row_ids):
                if str(run_id) in seen_ids:
                    raise ValueError(f"duplicate training capture across sources: {run_id}")
                seen_ids.add(str(run_id))
                training_run_ids.append(str(run_id))
                training_rows_by_run[str(run_id)] = int(np.count_nonzero(row_ids == run_id))
            if not np.array_equal(rows_base["source"], rows_roll["source"]):
                raise AssertionError("base and roll rows do not align")
            x_by_variant["base"].append(rows_base["x"])
            x_by_variant["roll"].append(rows_roll["x"])
            y_parts.append(rows_base["target_ay"])
            training_row_ids.append(row_ids)

    # Remap dataset-local indices to stable global run indices without
    # reloading or reprocessing the large capture archives.
    global_run_names = sorted(set(training_run_ids))
    global_index = {name: index for index, name in enumerate(global_run_names)}
    global_runs = np.concatenate([
        np.asarray([global_index[str(name)] for name in row_names], dtype=np.int32)
        for row_names in training_row_ids])
    train_x = {name: np.concatenate(parts, axis=0)
               for name, parts in x_by_variant.items()}
    train_y = np.concatenate(y_parts)
    train_ids = np.asarray(global_run_names, dtype=str)
    if len(global_runs) != len(train_y):
        raise AssertionError("run labels do not align with training transitions")

    fitted: dict[str, Any] = {}
    fit_reports: dict[str, Any] = {}
    with threadpool_limits(limits=threads):
        for variant in ("base", "roll"):
            fitted[variant], fit_reports[variant] = _fit(
                train_x[variant], train_y, global_runs, train_ids)

    # Evaluations use separate whole-run validation and practice captures;
    # no test/final-test capture participates in fitting or selection.
    highsteer_validation = _load_dataset(HIGHSTEER_DATASET)
    practice = _load_dataset(PRACTICE_DATASET)
    validation_reports = {
        variant: _one_step(highsteer_validation, "validation", parent_hgb,
                           fitted[variant], variant)
        for variant in ("base", "roll")}
    practice_split = "unseen_practice"
    if practice_split not in set(np.asarray(practice["splits"]).astype(str)):
        raise ValueError("practice dataset lacks its independent unseen split")
    practice_reports = {
        variant: _one_step(practice, practice_split, parent_hgb,
                           fitted[variant], variant)
        for variant in ("base", "roll")}

    recursive: dict[str, Any] = {}
    for label, data, path in (
            ("highsteer_validation", highsteer_validation, HIGHSTEER_DATASET),
            ("unseen_practice", practice, PRACTICE_DATASET)):
        baseline = _recursive(data, path, hgb, edssm, metadata, atlas,
                              yaw_model, yaw_start_step, longitudinal_residual,
                              None, "base")
        recursive[label] = {"baseline": baseline, "variants": {}}
        for variant in ("base", "roll"):
            candidate = _recursive(
                data, path, hgb, edssm, metadata, atlas, yaw_model,
                yaw_start_step, longitudinal_residual, fitted[variant], variant)
            recursive[label]["variants"][variant] = {
                "candidate": candidate,
                "paired_candidate_minus_baseline": _compare_recursive(
                    candidate, baseline),
            }

    output_dir.mkdir(parents=True, exist_ok=False)
    model_paths: dict[str, str] = {}
    for variant, model in fitted.items():
        path = output_dir / f"lateral_acceleration_{variant}.joblib"
        joblib.dump({
            "research_only": True,
            "production_integration_allowed": False,
            "target": "ay = dv_com/dt + u_mid*r_mid",
            "feature_variant": variant,
            "support_gate": LOCAL_GATE,
            "training_run_ids": global_run_names,
            "training_rows": int(len(train_y)),
            "parent_feature_contract": parent_hgb["feature_names"],
            "history_lags_steps": parent_hgb["history_lags_steps"],
            "frozen_yaw_atlas_used_for_evaluation": str(YAW_ATLAS_BUNDLE.relative_to(ROOT)),
            "model": model,
        }, path)
        model_paths[variant] = str(path.relative_to(ROOT))

    report = {
        "schema_version": 1,
        "purpose": "test whether a local 7-9 m/s, 0.30-0.525 rad body-frame lateral-acceleration teacher improves causal v prediction",
        "training_sources": [str(path.relative_to(ROOT)) for path in sources],
        "training_source_sha256": {str(path.relative_to(ROOT)): _sha256(path)
                                   for path in sources},
        "training_run_ids": global_run_names,
        "training_rows_by_run": training_rows_by_run,
        "training_rows_total": int(len(train_y)),
        "feature_variants": fit_reports,
        "candidate_models": model_paths,
        "support_gate": LOCAL_GATE,
        "one_step_whole_run_validation": validation_reports,
        "one_step_practice_transfer": practice_reports,
        "recursive_whole_run_evaluations": recursive,
        "test_and_final_test_used": False,
        "selection_protocol": "both prespecified feature variants reported; no candidate is promoted from one-step metrics alone",
        "yaw_atlas": {
            "bundle": str(YAW_ATLAS_BUNDLE.relative_to(ROOT)),
            "sha256": _sha256(YAW_ATLAS_BUNDLE),
            "modified": False,
        },
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "only the measured high-steer local support is modeled; this is not a full-domain vehicle plant",
            "one-step labels use simulator truth for offline supervision; rollout inputs use predicted states only",
            "practice transfer and recursive samples are grouped by whole capture; two practice runs limit uncertainty",
            "high-steer recursive rollouts are 5-second windows, not full-lap validation",
            "neither candidate is integrated into production odometry or MPC",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run()
    print(json.dumps({
        "output": str(OUTPUT_REPORT),
        "training_rows": report["training_rows_total"],
        "training_runs": report["training_run_ids"],
        "validation_ay_rmse": {
            variant: result["metrics"]["lateral_acceleration_candidate"][
                "macro_run_rmse"]
            for variant, result in report["one_step_whole_run_validation"].items()},
        "practice_ay_rmse": {
            variant: (result["metrics"]["lateral_acceleration_candidate"][
                "macro_run_rmse"] if "metrics" in result else result["status"])
            for variant, result in report["one_step_practice_transfer"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
