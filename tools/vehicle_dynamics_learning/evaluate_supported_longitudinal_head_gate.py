#!/usr/bin/env python3
"""Evaluate the high-steer longitudinal head only inside measured support."""

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
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


LONGITUDINAL_CANDIDATE = (
    LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib")
YAW_CORRECTOR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
YAW_REFERENCE = YAW_CORRECTOR.with_suffix(".json")
GLOBAL_COMPOSITION = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "longitudinal_heads_with_gated_yaw_v3_evaluation_20261009.json")
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "supported_longitudinal_head_gate_evaluation_20261009.json")
LONGITUDINAL_GATE = {
    "speed_mps": (7.0, 9.0),
    "absolute_steering_rad": (0.30, 0.525),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_atlas(path: Path) -> dict[str, Any]:
    atlas = joblib.load(path)
    for model in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    return atlas


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    required = (
        HGB_BUNDLE, LONGITUDINAL_CANDIDATE, YAW_CORRECTOR,
        YAW_REFERENCE, GLOBAL_COMPOSITION, EDSSM_CHECKPOINT,
        YAW_ATLAS_BUNDLE, HGB_TRAIN_DATASET, HIGHSTEER_DATASET,
        PRACTICE_DATASET,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_hgb = joblib.load(HGB_BUNDLE)
    candidate = joblib.load(LONGITUDINAL_CANDIDATE)
    correction_bundle = joblib.load(YAW_CORRECTOR)
    correction = HighSteerYawResidual(
        correction_bundle["estimator"],
        tuple(correction_bundle["gate"]["rear_wheel_mean_mps"]))
    atlas = _load_atlas(YAW_ATLAS_BUNDLE)
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    train_ids = set(candidate["training_run_ids"])

    whole_data = _load_dataset(HGB_TRAIN_DATASET)
    whole_windows = _run_windows(
        whole_data, "validation", _bundle_lags(reference_hgb), WINDOWS_PER_RUN)
    highsteer_data = _load_dataset(HIGHSTEER_DATASET)
    highsteer_windows = _windows(
        highsteer_data, HIGHSTEER_DATASET, HORIZON,
        _bundle_lags(reference_hgb))
    practice_data = _load_dataset(PRACTICE_DATASET)
    practice_windows = _windows(
        practice_data, PRACTICE_DATASET, HORIZON,
        _bundle_lags(reference_hgb))
    evaluations = (
        ("whole_run_validation", whole_data, whole_windows, "validation"),
        ("highsteer_validation", highsteer_data, highsteer_windows,
         "validation"),
        ("unseen_practice_diagnostic", practice_data, practice_windows,
         "unseen_practice"),
    )
    results: dict[str, Any] = {}
    for name, data, windows, expected_split in evaluations:
        run_ids = np.asarray(data["run_ids"]).astype(str)
        selected_ids = {str(run_ids[row["run"]]) for row in windows}
        splits = np.asarray(data["splits"]).astype(str)
        if (not selected_ids or selected_ids & train_ids
                or any(splits[row["run"]] != expected_split for row in windows)):
            raise ValueError(f"held-out split or run leakage in {name}")
        scored = _rollout(
            data, reference_hgb, edssm, metadata, atlas, windows, HORIZON,
            yaw_delta_gain=1.0,
            include_oracle_diagnostics=False,
            reconcile_composed_latent=True,
            yaw_correction_model=correction,
            yaw_correction_start_step=1,
            hgb_main_override_models=candidate["models"],
            hgb_main_gate=LONGITUDINAL_GATE)
        results[name] = {
            "heldout_run_ids": sorted(scored["per_run"]),
            "windows": scored["windows"],
            "windows_by_run": scored["windows_by_run"],
            "candidate_gate_active_fraction": scored[
                "hgb_main_gate_active_fraction"],
            "macro_run_rmse": scored["macro_run_rmse"],
            "per_run": scored["per_run"],
            "initial_yaw_step": scored[
                "yaw_atlas_one_step_selected_windows"],
        }

    report = {
        "schema_version": 1,
        "purpose": "run the high-steer-trained u/filtered-wheel heads only within their measured speed/steering start envelope, composed with the gated frozen yaw atlas",
        "research_only": True,
        "simulator_was_run": False,
        "production_or_runtime_integration": False,
        "test_or_final_test_used": False,
        "fit_or_selection_data": "none; all fitted artifacts frozen before this held-out evaluation",
        "longitudinal_head_gate": {
            name: list(bounds) for name, bounds in LONGITUDINAL_GATE.items()},
        "yaw_correction_start_step": 1,
        "longitudinal_heads": ["u", "filtered rear-left wheel", "filtered rear-right wheel"],
        "actuator_and_raw-encoder_heads_changed": False,
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "heldout_results": results,
        "comparators": {
            "base_longitudinal_plus_gated_yaw": {
                "report": str(YAW_REFERENCE.relative_to(ROOT)),
                "report_sha256": _sha256(YAW_REFERENCE),
            },
            "global_augmented_longitudinal_plus_gated_yaw": {
                "report": str(GLOBAL_COMPOSITION.relative_to(ROOT)),
                "report_sha256": _sha256(GLOBAL_COMPOSITION),
            },
        },
        "limitations": [
            "the candidate is supported by two high-steer training runs and three high-steer validation runs",
            "the two practice captures do not cover the high-steer regime",
            "all rollouts are five-second seeded segments, not full-lap simulation validation",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    report = run()
    print(json.dumps({
        "output": str(OUTPUT),
        "results": {
            name: {
                "gate_active_fraction": row["candidate_gate_active_fraction"],
                "five_second": row["macro_run_rmse"]["5.000s"]["hybrid_atlas"],
            }
            for name, row in report["heldout_results"].items()},
    }, indent=2, sort_keys=True))
