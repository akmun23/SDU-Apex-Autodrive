#!/usr/bin/env python3
"""Evaluate the balanced 0.5 longitudinal residual across held-out domains."""

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
    MODEL_OUTPUT as FULL_GAIN_MODEL,
    OUTPUT as FULL_GAIN_REPORT,
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
YAW_GAIN_REPORT = YAW_CORRECTOR_PATH.with_suffix(".json")
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "highsteer_recursive_longitudinal_residual_gain0p5_training_20261009/"
    "transfer_evaluation.json")
MODEL_OUTPUT = OUTPUT.with_name("u_filtered_wheel_residual_models_gain0p5.joblib")
GAIN = 0.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    model_output = output_path.with_name(MODEL_OUTPUT.name)
    if output_path.exists() or model_output.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    required = (
        HGB_BUNDLE, LONGITUDINAL_CANDIDATE, FULL_GAIN_MODEL,
        FULL_GAIN_REPORT, YAW_CORRECTOR_PATH, YAW_GAIN_REPORT,
        EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE, HGB_TRAIN_DATASET,
        HIGHSTEER_DATASET, PRACTICE_DATASET,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    base_hgb = joblib.load(HGB_BUNDLE)
    longitudinal = joblib.load(LONGITUDINAL_CANDIDATE)
    hgb = dict(base_hgb)
    hgb["models"] = longitudinal["models"]
    hgb["training_run_ids"] = longitudinal["training_run_ids"]
    residual_bundle = joblib.load(FULL_GAIN_MODEL)
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

    whole_data = _load_dataset(HGB_TRAIN_DATASET)
    whole_windows = _run_windows(
        whole_data, "validation", _bundle_lags(hgb), WINDOWS_PER_RUN)
    highsteer_data = _load_dataset(HIGHSTEER_DATASET)
    highsteer_windows = _windows(
        highsteer_data, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))
    practice = _load_dataset(PRACTICE_DATASET)
    practice_windows = _windows(
        practice, PRACTICE_DATASET, HORIZON, _bundle_lags(hgb))
    evaluation_sets = (
        ("whole_run_validation", whole_data, whole_windows, "validation"),
        ("highsteer_validation", highsteer_data, highsteer_windows,
         "validation"),
        ("unseen_practice_diagnostic", practice, practice_windows,
         "unseen_practice"),
    )
    heldout: dict[str, Any] = {}
    for name, data, windows, expected_split in evaluation_sets:
        run_ids = np.asarray(data["run_ids"]).astype(str)
        splits = np.asarray(data["splits"]).astype(str)
        selected_ids = {str(run_ids[row["run"]]) for row in windows}
        if (not selected_ids or selected_ids & set(residual_bundle["training_run_ids"])
                or any(splits[row["run"]] != expected_split for row in windows)):
            raise ValueError(f"held-out split/run leakage in {name}")
        baseline = _rollout(
            data, hgb, edssm, metadata, atlas, windows, HORIZON,
            yaw_delta_gain=1.0, include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=CORRECTION_START_STEP,
            hgb_main_gate=residual_bundle["gate"])
        candidate = _rollout(
            data, hgb, edssm, metadata, atlas, windows, HORIZON,
            yaw_delta_gain=1.0, include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            yaw_correction_model=yaw_model,
            yaw_correction_start_step=CORRECTION_START_STEP,
            hgb_main_gate=residual_bundle["gate"],
            hgb_main_residual_models=residual_bundle["models"],
            hgb_main_residual_start_step=1,
            hgb_main_residual_gain=GAIN)
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
            "residual_gain_0p5_with_gated_yaw": {
                "macro_run_rmse": candidate["macro_run_rmse"],
                "per_run": candidate["per_run"],
                "initial_yaw_step": candidate[
                    "yaw_atlas_one_step_selected_windows"],
            },
        }

    high_base = heldout["highsteer_validation"][
        "baseline_augmented_heads_with_gated_yaw"]["macro_run_rmse"]
    high_candidate = heldout["highsteer_validation"][
        "residual_gain_0p5_with_gated_yaw"]["macro_run_rmse"]
    high_improved = all(
        high_candidate["5.000s"]["hybrid_atlas"][metric]
        < high_base["5.000s"]["hybrid_atlas"][metric]
        for metric in ("u_rmse_mps", "filtered_rear_wheel_pair_rmse_mps",
                       "position_rmse_m", "yaw_pose_rmse_rad"))
    high_yaw_rate_supported = (
        high_candidate["5.000s"]["hybrid_atlas"]["yaw_rate_rmse_rps"]
        <= 1.10 * high_base["5.000s"]["hybrid_atlas"]["yaw_rate_rmse_rps"])
    other_domains_supported = all(
        heldout[name]["residual_gain_0p5_with_gated_yaw"]["macro_run_rmse"]
        ["5.000s"]["hybrid_atlas"][metric]
        <= 1.10 * heldout[name]["baseline_augmented_heads_with_gated_yaw"]
        ["macro_run_rmse"]["5.000s"]["hybrid_atlas"][metric]
        for name in ("whole_run_validation", "unseen_practice_diagnostic")
        for metric in ("position_rmse_m", "yaw_rate_rmse_rps"))
    retained = high_improved and high_yaw_rate_supported and other_domains_supported
    report = {
        "schema_version": 1,
        "purpose": "held-out transfer check of the 0.5-scaled causal high-steer u/wheel residual selected by a development sensitivity sweep",
        "research_only": True,
        "simulator_was_run": False,
        "production_or_runtime_integration": False,
        "test_or_final_test_used": False,
        "gain_selection_source": "high-steer validation sensitivity report; these are development runs, not a pristine final test",
        "residual_gain": GAIN,
        "first_observed_transition_uses_unmodified_longitudinal_heads": True,
        "yaw_atlas_and_yaw_correction_frozen": True,
        "heldout_results": heldout,
        "candidate_decision": {
            "criteria": {
                "highsteer_5s_u_wheels_position_and_yaw_pose_all_improve": high_improved,
                "highsteer_5s_yaw_rate_within_10pct": high_yaw_rate_supported,
                "whole_run_and_practice_5s_position_yaw_within_10pct":
                    other_domains_supported,
            },
            "retained_as_research_candidate": bool(retained),
            "production_or_runtime_integration": False,
        },
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "limitations": [
            "the high-steer validation runs were used in development gain sensitivity and are not final independent confirmation",
            "the result covers 7-9 m/s high-steer start windows and a five-second horizon only",
            "this remains a partial offline plant, not a full-lap simulator",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if retained:
        joblib.dump({
            "schema_version": 1,
            "models": residual_bundle["models"],
            "gate": residual_bundle["gate"],
            "gain": GAIN,
            "correction_start_step": 1,
            "training_run_ids": residual_bundle["training_run_ids"],
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
        "decision": result["candidate_decision"],
        "highsteer_5s": {
            model: row[model]["macro_run_rmse"]["5.000s"]["hybrid_atlas"]
            for model in ("baseline_augmented_heads_with_gated_yaw",
                          "residual_gain_0p5_with_gated_yaw")
            for row in (result["heldout_results"]["highsteer_validation"],)},
        "broad_gate_fractions": {
            name: row["gate_active_fraction"]
            for name, row in result["heldout_results"].items()},
    }, indent=2, sort_keys=True))
