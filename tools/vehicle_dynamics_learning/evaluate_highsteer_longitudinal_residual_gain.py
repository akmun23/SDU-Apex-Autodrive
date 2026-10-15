#!/usr/bin/env python3
"""Sensitivity-check the learned high-steer u/wheel correction strength."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib

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
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_longitudinal_residual import (
    MODEL_OUTPUT,
    OUTPUT as FIT_REPORT,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
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
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_recursive_longitudinal_residual_20261009/"
    "residual_gain_sensitivity.json")
GAINS = (0.0, 0.5, 0.75, 1.0, 1.25)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    required = (HGB_BUNDLE, LONGITUDINAL_CANDIDATE, MODEL_OUTPUT,
                FIT_REPORT, YAW_CORRECTOR_PATH, EDSSM_CHECKPOINT,
                YAW_ATLAS_BUNDLE, HIGHSTEER_DATASET)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    base_hgb = joblib.load(HGB_BUNDLE)
    longitudinal = joblib.load(LONGITUDINAL_CANDIDATE)
    hgb = dict(base_hgb)
    hgb["models"] = longitudinal["models"]
    hgb["training_run_ids"] = longitudinal["training_run_ids"]
    residual_bundle = joblib.load(MODEL_OUTPUT)
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for model in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    data = _load_dataset(HIGHSTEER_DATASET)
    windows = _windows(data, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))
    results: dict[str, Any] = {}
    for gain in GAINS:
        scored = _rollout(
            data, hgb, edssm, metadata, atlas, windows, HORIZON,
            yaw_delta_gain=1.0, include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=CORRECTION_START_STEP,
            hgb_main_gate=residual_bundle["gate"],
            hgb_main_residual_models=(
                residual_bundle["models"] if gain != 0.0 else None),
            hgb_main_residual_start_step=1,
            hgb_main_residual_gain=gain)
        results[f"{gain:.2f}"] = {
            "gain": gain,
            "gate_active_fraction": scored["hgb_main_gate_active_fraction"],
            "macro_run_rmse": scored["macro_run_rmse"],
            "per_run": scored["per_run"],
        }
    report = {
        "schema_version": 1,
        "purpose": "diagnostic sensitivity of the already-fitted high-steer u/filtered-wheel residual; no retraining or gain selection",
        "research_only": True,
        "simulator_was_run": False,
        "test_or_final_test_used": False,
        "heldout_highsteer_runs_are_development_data": True,
        "yaw_atlas_and_yaw_correction_frozen": True,
        "first_observed_transition_uses_base_heads": True,
        "candidate_gain_values": list(GAINS),
        "heldout_run_ids": sorted({
            str(data["run_ids"][row["run"]]) for row in windows}),
        "windows": len(windows),
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "results": results,
        "limitations": [
            "this tests only the three 7-9 m/s high-steer validation captures",
            "these captures have already been used for development comparisons and are not a pristine final test",
            "the result does not establish full-lap simulator accuracy",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "highsteer_macro": {
            gain: {
                horizon: {
                    name: result["results"][gain]["macro_run_rmse"][horizon]
                    ["hybrid_atlas"][name]
                    for name in ("u_rmse_mps",
                                 "filtered_rear_wheel_pair_rmse_mps",
                                 "raw_encoder_pair_rmse_mps",
                                 "position_rmse_m", "yaw_rate_rmse_rps")}
                for horizon in ("0.750s", "2.000s", "5.000s")}
            for gain in (f"{value:.2f}" for value in GAINS)},
    }, indent=2, sort_keys=True))
