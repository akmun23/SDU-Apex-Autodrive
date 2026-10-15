#!/usr/bin/env python3
"""Check whether a weaker inference gain stabilizes the lateral residual."""

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
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_lateral_residual import (
    GAIN,
    HIGHSTEER_ABS_STEERING_RAD,
    HIGHSTEER_SPEED_MPS,
    LONGITUDINAL_RESIDUAL_BUNDLE,
    MODEL_OUTPUT,
    OUTPUT_DIR,
    YAW_CORRECTOR_PATH,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


OUTPUT = OUTPUT_DIR / "inference_gain_sensitivity.json"
GAINS = (0.10, 0.20, 0.35, GAIN)
HORIZON_KEYS = ("0.750s", "2.000s", "5.000s")
METRICS = ("v_rmse_mps", "u_rmse_mps", "position_rmse_m",
           "yaw_rate_rmse_rps", "yaw_pose_rmse_rad",
           "filtered_rear_wheel_pair_rmse_mps")


def _trim(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_ids": sorted(result["per_run"]),
        "windows": result["windows"],
        "windows_by_run": result["windows_by_run"],
        "gate_active_fraction": result["hgb_main_gate_active_fraction"],
        "macro": {
            horizon: {
                metric: result["macro_run_rmse"][horizon]["hybrid_atlas"][metric]
                for metric in METRICS
            }
            for horizon in HORIZON_KEYS},
        "per_run": {
            run: {
                horizon: {
                    metric: result["per_run"][run][horizon]["hybrid_atlas"][metric]
                    for metric in METRICS
                }
                for horizon in HORIZON_KEYS}
            for run in sorted(result["per_run"])},
    }


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    parent_hgb = joblib.load(HGB_BUNDLE)
    longitudinal = joblib.load(
        LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib")
    hgb = dict(parent_hgb)
    hgb["models"] = longitudinal["models"]
    hgb["training_run_ids"] = longitudinal["training_run_ids"]
    longitudinal_residual = joblib.load(LONGITUDINAL_RESIDUAL_BUNDLE)
    lateral_bundle = joblib.load(MODEL_OUTPUT)
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"], tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    yaw_start = int(yaw_bundle.get("correction_start_step", CORRECTION_START_STEP))
    lateral_model = lateral_bundle["model"]
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for expert in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(expert, "n_jobs"):
            expert.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    whole_data = _load_dataset(HGB_TRAIN_DATASET)
    highsteer_data = _load_dataset(HIGHSTEER_DATASET)
    practice_data = _load_dataset(PRACTICE_DATASET)
    specs = (
        ("whole_run_validation", whole_data,
         _run_windows(whole_data, "validation", _bundle_lags(hgb),
                      WINDOWS_PER_RUN)),
        ("highsteer_validation", highsteer_data, None),
        ("unseen_practice_diagnostic", practice_data, None),
    )
    normalized_specs = []
    for name, data, windows in specs:
        if windows is None:
            path = HIGHSTEER_DATASET if name == "highsteer_validation" else PRACTICE_DATASET
            windows = _windows(data, path, HORIZON, _bundle_lags(hgb))
        normalized_specs.append((name, data, windows))

    gate = {
        "speed_mps": tuple(HIGHSTEER_SPEED_MPS),
        "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD),
    }
    heldout: dict[str, Any] = {}
    for name, data, windows in normalized_specs:
        heldout[name] = {}
        baseline = _rollout(
            data, hgb, edssm, metadata, atlas, windows, HORIZON,
            yaw_delta_gain=1.0, include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            atlas_acceleration_source="composed_transition",
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=yaw_start,
            hgb_main_gate=gate,
            hgb_main_residual_models=longitudinal_residual["models"],
            hgb_main_residual_start_step=1,
            hgb_main_residual_gain=float(longitudinal_residual["gain"]))
        heldout[name]["gain_0"] = _trim(baseline)
        for gain in GAINS:
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
                lateral_v_residual_model=lateral_model,
                lateral_v_residual_start_step=1,
                lateral_v_residual_gain=gain)
            heldout[name][f"gain_{gain:g}"] = _trim(result)

    report = {
        "schema_version": 1,
        "purpose": "development sensitivity diagnostic for the rejected high-steer lateral-v residual",
        "selection_data_are_not_pristine": True,
        "test_or_final_test_used": False,
        "fit_or_model_changed": False,
        "heldout_results": heldout,
        "gains_tested": [0.0, *GAINS],
        "limitations": [
            "validation was already used during model development",
            "the two practice captures are transfer diagnostics, not a broad unseen-domain guarantee",
        ],
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "highsteer_5s": {
            gain: values["macro"]["5.000s"]
            for gain, values in result["heldout_results"][
                "highsteer_validation"].items()},
        "practice_5s": {
            gain: values["macro"]["5.000s"]
            for gain, values in result["heldout_results"][
                "unseen_practice_diagnostic"].items()},
    }, indent=2, sort_keys=True))
