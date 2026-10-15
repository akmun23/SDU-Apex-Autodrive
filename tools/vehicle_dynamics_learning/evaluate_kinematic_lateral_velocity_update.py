#!/usr/bin/env python3
"""Compare EDSSM v output with a kinematically consistent predicted-a_y update."""

from __future__ import annotations

import hashlib
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
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_longitudinal_residual import (
    HIGHSTEER_ABS_STEERING_RAD,
    HIGHSTEER_SPEED_MPS,
)
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_lateral_residual import (
    LONGITUDINAL_RESIDUAL_BUNDLE,
    YAW_CORRECTOR_PATH,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "kinematic_lateral_velocity_update_20261009")
OUTPUT = OUTPUT_DIR / "evaluation.json"
YAW_ATLAS_BUNDLE = ROOT / (
    "live_runs/racing_model_diagnostics_20261008/"
    "sensor_only_yaw_regime_atlas_command_age_v4/"
    "sensor_only_yaw_regime_atlas.joblib")
METRICS = (
    "u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
    "position_rmse_m", "yaw_pose_rmse_rad",
    "filtered_rear_wheel_pair_rmse_mps")
CHECKPOINTS = ("0.750s", "2.000s", "5.000s")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_ids": sorted(result["per_run"]),
        "windows": result["windows"],
        "windows_by_run": result["windows_by_run"],
        "macro": {
            horizon: {
                metric: result["macro_run_rmse"][horizon]["hybrid_atlas"][metric]
                for metric in METRICS}
            for horizon in CHECKPOINTS},
        "per_run": {
            run: {
                horizon: {
                    metric: result["per_run"][run][horizon]["hybrid_atlas"][metric]
                    for metric in METRICS}
                for horizon in CHECKPOINTS}
            for run in sorted(result["per_run"])},
    }


def _first_transition_metrics(result: dict[str, Any]) -> dict[str, float]:
    trace = result.pop("_trajectory_trace")
    predicted = np.asarray(trace["predicted_state"], dtype=np.float64)[:, 0]
    truth = np.asarray(trace["truth_state"], dtype=np.float64)[:, 0]
    pred_pose = np.asarray(trace["predicted_pose"], dtype=np.float64)[:, 0]
    truth_pose = np.asarray(trace["truth_pose"], dtype=np.float64)[:, 0]
    yaw_pose = pred_pose[:, 2] - truth_pose[:, 2]
    yaw_pose = np.arctan2(np.sin(yaw_pose), np.cos(yaw_pose))
    position = np.linalg.norm(pred_pose[:, :2] - truth_pose[:, :2], axis=1)
    return {
        "u_rmse_mps": float(np.sqrt(np.mean((predicted[:, 0] - truth[:, 0]) ** 2))),
        "v_rmse_mps": float(np.sqrt(np.mean((predicted[:, 1] - truth[:, 1]) ** 2))),
        "yaw_rate_rmse_rps": float(np.sqrt(np.mean((predicted[:, 2] - truth[:, 2]) ** 2))),
        "position_rmse_m": float(np.sqrt(np.mean(position ** 2))),
        "yaw_pose_rmse_rad": float(np.sqrt(np.mean(yaw_pose ** 2))),
    }


def _run_model(data: dict[str, Any], windows: list[dict[str, int]],
               model_name: str, hgb: dict[str, Any], edssm: Any,
               metadata: dict[str, Any], atlas: dict[str, Any],
               yaw_model: Any, yaw_start: int,
               longitudinal_residual: dict[str, Any]
               ) -> dict[str, Any]:
    result = _rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=1.0, include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source="composed_transition",
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=yaw_start,
        lateral_v_update=model_name,
        hgb_main_gate={
            "speed_mps": tuple(HIGHSTEER_SPEED_MPS),
            "absolute_steering_rad": tuple(HIGHSTEER_ABS_STEERING_RAD),
        },
        hgb_main_residual_models=longitudinal_residual["models"],
        hgb_main_residual_start_step=1,
        hgb_main_residual_gain=float(longitudinal_residual["gain"]),
        capture_trajectory_trace=True)
    first_transition = _first_transition_metrics(result)
    return {"metrics": _state_metrics(result),
            "first_transition": first_transition,
            "gate_active_fraction": result["hgb_main_gate_active_fraction"],
            "initial_yaw_atlas": result["yaw_atlas_one_step_selected_windows"]}


def run() -> dict[str, Any]:
    if OUTPUT.exists() or OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    hgb_path = LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib"
    required = (
        HGB_BUNDLE, hgb_path, LONGITUDINAL_RESIDUAL_BUNDLE,
        YAW_CORRECTOR_PATH, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
        HGB_TRAIN_DATASET, HIGHSTEER_DATASET, PRACTICE_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = joblib.load(HGB_BUNDLE)
    longitudinal = joblib.load(hgb_path)
    hgb = dict(parent)
    hgb["models"] = longitudinal["models"]
    hgb["training_run_ids"] = longitudinal["training_run_ids"]
    longitudinal_residual = joblib.load(LONGITUDINAL_RESIDUAL_BUNDLE)
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

    whole_data = _load_dataset(HGB_TRAIN_DATASET)
    high_data = _load_dataset(HIGHSTEER_DATASET)
    practice_data = _load_dataset(PRACTICE_DATASET)
    specs = (
        ("whole_run_validation", whole_data,
         _run_windows(whole_data, "validation", _bundle_lags(hgb),
                      WINDOWS_PER_RUN)),
        ("highsteer_validation", high_data,
         _windows(high_data, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))),
        ("unseen_practice_diagnostic", practice_data,
         _windows(practice_data, PRACTICE_DATASET, HORIZON, _bundle_lags(hgb))),
    )
    heldout: dict[str, Any] = {}
    for name, data, windows in specs:
        baseline = _run_model(data, windows, "edssm", hgb, edssm, metadata,
                              atlas, yaw_model, yaw_start,
                              longitudinal_residual)
        candidate = _run_model(data, windows, "kinematic_ay", hgb, edssm,
                               metadata, atlas, yaw_model, yaw_start,
                               longitudinal_residual)
        heldout[name] = {
            "run_ids": candidate["metrics"]["run_ids"],
            "baseline_edssm_v_head": baseline,
            "kinematic_predicted_ay_v_update": candidate,
        }

    report = {
        "schema_version": 1,
        "purpose": "test a rigid-body kinematic update v_dot = a_y - u*r using the EDSSM predicted lateral acceleration and frozen yaw-atlas rate",
        "offline_only": True,
        "simulator_was_run": False,
        "production_or_runtime_integration": False,
        "test_or_final_test_used": False,
        "ground_truth_used_only_for_scoring": True,
        "yaw_atlas_and_yaw_corrector_frozen": True,
        "longitudinal_residual_frozen": True,
        "formula": "v_next = v + dt * (predicted_a_y - u_mid * r_mid), dt=0.025 s",
        "heldout_results": heldout,
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "limitations": [
            "this changes only lateral-velocity propagation in the offline plant evaluator",
            "the predicted a_y comes from the existing EDSSM transition and can itself be biased",
            "high-steer validation and practice captures have informed prior development",
            "this is a five-second model evaluation, not a full-lap simulator certification",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    summary = {}
    for split, values in result["heldout_results"].items():
        summary[split] = {
            model: {
                "first_transition": values[model]["first_transition"],
                "0.75s": values[model]["metrics"]["macro"]["0.750s"],
                "2s": values[model]["metrics"]["macro"]["2.000s"],
                "5s": values[model]["metrics"]["macro"]["5.000s"],
            }
            for model in ("baseline_edssm_v_head",
                          "kinematic_predicted_ay_v_update")}
    print(json.dumps({"output": str(OUTPUT), "summary": summary}, indent=2,
                     sort_keys=True))
