#!/usr/bin/env python3
"""Locate causal-rollout lateral-velocity divergence in high-steer captures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_composed_latent_reconciliation import (
    WINDOWS_PER_RUN,
)
from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    EDSSM_CHECKPOINT,
    HGB_BUNDLE,
    HIGHSTEER_DATASET,
    YAW_ATLAS_BUNDLE,
    _bundle_lags,
    _rollout,
    _windows,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import HORIZON
from tools.vehicle_dynamics_learning.fit_highsteer_augmented_longitudinal_candidate import (
    OUTPUT_DIR as LONGITUDINAL_CANDIDATE_DIR,
)
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_lateral_residual import (
    GAIN,
    HIGHSTEER_ABS_STEERING_RAD,
    HIGHSTEER_SPEED_MPS,
    LONGITUDINAL_RESIDUAL_BUNDLE,
    MODEL_OUTPUT as LATERAL_MODEL_PATH,
    OUTPUT_DIR,
    YAW_CORRECTOR_PATH,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


OUTPUT = OUTPUT_DIR / "divergence_attribution.json"
GAIN_UNDER_DIAGNOSIS = 0.20
HORIZONS = {"0.75s": 30, "2.00s": 80, "3.00s": 120, "4.00s": 160,
            "5.00s": 200}
REGIME_BINS = {
    "speed_mps": ([0.0, 3.0, 5.0, 7.0, 9.0, 12.0001],
                   ["0-3", "3-5", "5-7", "7-9", "9-12"]),
    "abs_steering_rad": ([0.0, 0.10, 0.20, 0.30, 0.40, 0.5251],
                          ["0-.10", ".10-.20", ".20-.30", ".30-.40", ".40-.525"]),
    "abs_steering_rate_rps": ([0.0, 0.5, 2.0, 5.0, 10.0, 40.0001],
                              ["0-.5", ".5-2", "2-5", "5-10", "10-40"]),
    "throttle_command": ([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0001],
                         ["0-.05", ".05-.10", ".10-.20", ".20-.35",
                          ".35-.50", ".50-.75", ".75-1"]),
    "abs_throttle_slew_per_s": ([0.0, 0.25, 1.0, 3.0, 8.0, 22.0, 40.0001],
                                ["0-.25", ".25-1", "1-3", "3-8", "8-22", "22-40"]),
    "abs_wheel_body_mismatch_mps": ([0.0, 0.10, 0.25, 0.5, 1.0, 2.0, 6.0001],
                                    ["0-.10", ".10-.25", ".25-.5", ".5-1",
                                     "1-2", "2-6"]),
    "abs_roll_rad": ([0.0, 0.02, 0.05, 0.10, 0.20, 0.6001],
                     ["0-.02", ".02-.05", ".05-.10", ".10-.20", ".20-.60"]),
    "abs_roll_rate_rps": ([0.0, 0.1, 0.3, 0.7, 1.5, 5.0001],
                          ["0-.1", ".1-.3", ".3-.7", ".7-1.5", "1.5-5"]),
}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def _summarize_errors(trace: dict[str, Any], run_ids: np.ndarray,
                      frames: np.ndarray, raw_wheels: np.ndarray,
                      raw_valid: np.ndarray, attitude: np.ndarray,
                      attitude_valid: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(trace["predicted_state"], dtype=np.float64)
    truth = np.asarray(trace["truth_state"], dtype=np.float64)
    predicted_pose = np.asarray(trace["predicted_pose"], dtype=np.float64)
    truth_pose = np.asarray(trace["truth_pose"], dtype=np.float64)
    indices = np.asarray(trace["source_indices"], dtype=np.int64)
    run_indices = np.asarray(trace["run_indices"], dtype=np.int32)
    v_error = predicted[:, :, 1] - truth[:, :, 1]
    # Increment error is predicted minus truth change, not state error.
    v_increment_error = np.diff(predicted[:, :, 1], axis=1) - np.diff(
        truth[:, :, 1], axis=1)
    pose_error = predicted_pose - truth_pose
    pose_error[:, :, 2] = np.arctan2(np.sin(pose_error[:, :, 2]),
                                     np.cos(pose_error[:, :, 2]))
    features_by_index = {
        "speed_mps": np.hypot(truth[:, :, 0], truth[:, :, 1]),
        "abs_steering_rad": np.abs(frames[indices, 3]),
        "abs_steering_rate_rps": np.abs(
            (frames[indices, 3] - frames[np.maximum(indices - 1, 0), 3]) / 0.025),
        "throttle_command": np.clip(frames[indices, 8], 0.0, 1.0),
        "abs_throttle_slew_per_s": np.abs(
            (frames[indices, 8] - frames[np.maximum(indices - 1, 0), 8]) / 0.025),
        "abs_wheel_body_mismatch_mps": np.abs(
            np.mean(raw_wheels[indices], axis=2) - truth[:, :, 0]),
        "abs_roll_rad": np.abs(attitude[indices, 0]),
        "abs_roll_rate_rps": np.abs(attitude[indices, 2]),
    }
    valid = raw_valid[indices] & attitude_valid[indices]
    per_run: dict[str, Any] = {}
    for run in sorted(np.unique(run_indices).tolist()):
        mask = run_indices == run
        run_id = str(run_ids[run])
        after_2s_curve = [
            _rmse(v_error[mask, step])
            for step in range(79, v_error.shape[1])]
        peak_after_2s = int(np.argmax(after_2s_curve))
        per_run[run_id] = {
            "windows": int(mask.sum()),
            "v_state_error_by_time": {
                f"{step * 0.025:.3f}s": {
                    "rmse_mps": _rmse(v_error[mask, step - 1]),
                    "bias_mps": float(np.mean(v_error[mask, step - 1])),
                    "p95_abs_mps": float(np.quantile(
                        np.abs(v_error[mask, step - 1]), 0.95)),
                }
                for step in HORIZONS.values()},
            "first_time_rmse_crosses_threshold": {
                str(threshold): next((
                    f"{(step + 1) * 0.025:.3f}s"
                    for step in range(v_error.shape[1])
                    if _rmse(v_error[mask, step]) >= threshold), None)
                for threshold in (0.1, 0.25, 0.5, 1.0)},
            "max_abs_step_increment_error_rmse_across_windows_mps": _rmse(
                np.nanmax(np.abs(v_increment_error[mask]), axis=1)),
            "largest_state_error_rmse_after_2s": {
                "time": f"{(peak_after_2s + 80) * 0.025:.3f}s",
                "rmse_mps": float(after_2s_curve[peak_after_2s]),
            },
            "pose_position_error_at_5s_rmse_m": _rmse(
                np.linalg.norm(pose_error[mask, -1, :2], axis=1)),
            "yaw_rate_rmse_at_5s_rps": _rmse(
                predicted[mask, -1, 2] - truth[mask, -1, 2]),
        }

    regime_reports: dict[str, Any] = {}
    for feature_name, (edges, labels) in REGIME_BINS.items():
        values = features_by_index[feature_name]
        regime_reports[feature_name] = {}
        for bin_index, label in enumerate(labels):
            mask = ((values >= edges[bin_index])
                    & (values < edges[bin_index + 1]) & valid)
            if np.count_nonzero(mask) < 50:
                continue
            per_run_bin = {}
            for run in sorted(np.unique(run_indices).tolist()):
                run_mask = (run_indices[:, None] == run) & mask
                if np.count_nonzero(run_mask) < 20:
                    continue
                per_run_bin[str(run_ids[run])] = {
                    "transitions": int(np.count_nonzero(run_mask)),
                    "v_error_rmse_mps_by_horizon": {
                        horizon: _rmse(v_error[:, :steps][run_mask[:, :steps]])
                        for horizon, steps in HORIZONS.items()},
                }
            regime_reports[feature_name][label] = {
                "transitions": int(np.count_nonzero(mask)),
                "v_error_rmse_mps_by_horizon": {
                    horizon: _rmse(v_error[:, :steps][mask[:, :steps]])
                    for horizon, steps in HORIZONS.items()},
                "per_run": per_run_bin,
            }
    return {
        "per_run": per_run,
        "regime_stratified_v_error": regime_reports,
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    hgb_path = LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib"
    hgb = dict(joblib.load(HGB_BUNDLE))
    longitudinal = joblib.load(hgb_path)
    hgb["models"] = longitudinal["models"]
    hgb["training_run_ids"] = longitudinal["training_run_ids"]
    longitudinal_residual = joblib.load(LONGITUDINAL_RESIDUAL_BUNDLE)
    lateral_bundle = joblib.load(LATERAL_MODEL_PATH)
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"], tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    yaw_start = int(yaw_bundle.get("correction_start_step", CORRECTION_START_STEP))
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for expert in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(expert, "n_jobs"):
            expert.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    data = _load_dataset(HIGHSTEER_DATASET)
    windows = _windows(data, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))
    gate = {"speed_mps": tuple(HIGHSTEER_SPEED_MPS),
            "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD)}
    run_modes = {
        "no_lateral_residual": None,
        f"lateral_residual_gain_{GAIN_UNDER_DIAGNOSIS:g}": lateral_bundle["model"],
    }
    outputs = {}
    for name, model in run_modes.items():
        gain = 0.0 if model is None else GAIN_UNDER_DIAGNOSIS
        result = _rollout(
            data, hgb, edssm, metadata, atlas, windows, HORIZON,
            yaw_delta_gain=1.0, include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            atlas_acceleration_source="composed_transition",
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=yaw_start,
            hgb_main_gate=gate,
            hgb_main_residual_models=longitudinal_residual["models"],
            hgb_main_residual_start_step=1,
            hgb_main_residual_gain=float(longitudinal_residual["gain"]),
            lateral_v_residual_model=model,
            lateral_v_residual_start_step=1,
            lateral_v_residual_gain=gain,
            capture_trajectory_trace=True)
        raw_wheels = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
        raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
        attitude = np.asarray(data["imu_attitude_frames"], dtype=np.float64)
        attitude_valid = np.asarray(data["imu_attitude_valid"], dtype=bool)
        run_ids = np.asarray(data["run_ids"]).astype(str)
        diagnostics = _summarize_errors(
            result.pop("_trajectory_trace"), run_ids,
            np.asarray(data["frames"], dtype=np.float64), raw_wheels,
            raw_valid, attitude, attitude_valid)
        outputs[name] = {
            "macro_metrics": result["macro_run_rmse"],
            "gate_active_fraction": result["hgb_main_gate_active_fraction"],
            **diagnostics,
        }
    report = {
        "schema_version": 1,
        "purpose": "identify when high-steer recursive v error first grows and which measured GT regimes covary with it",
        "diagnostic_only": True,
        "ground_truth_features_are_posthoc_only": True,
        "yaw_atlas_and_corrector_frozen": True,
        "test_or_final_test_used": False,
        "highsteer_validation_was_used_in_prior_candidate_development": True,
        "candidate_gain": GAIN_UNDER_DIAGNOSIS,
        "runs": sorted(outputs["no_lateral_residual"]["per_run"]),
        "models": outputs,
        "regime_bins": {name: {"edges": edges, "labels": labels}
                         for name, (edges, labels) in REGIME_BINS.items()},
        "limitations": [
            "the regime correlations are diagnostic and not causal proof",
            "validation runs are previously examined development data",
            "each trace has overlapping windows and repeated 25-ms samples; counts are not independent trials",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    compact = {}
    for mode, values in result["models"].items():
        compact[mode] = {
            run: {
                "threshold_crossings": metrics[
                    "first_time_rmse_crosses_threshold"],
                "v_rmse": metrics["v_state_error_by_time"],
            }
            for run, metrics in values["per_run"].items()}
    print(json.dumps({"output": str(OUTPUT), "per_run": compact}, indent=2,
                     sort_keys=True))
