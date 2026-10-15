#!/usr/bin/env python3
"""Check whether the high-steer longitudinal head composes with gated yaw."""

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
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "longitudinal_heads_with_gated_yaw_v3_evaluation_20261009.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
           metadata: dict[str, Any], atlas: dict[str, Any],
           windows: list[dict[str, int]], correction: Any | None
           ) -> dict[str, Any]:
    return _rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=1.0,
        include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        yaw_correction_model=correction,
        yaw_correction_start_step=1 if correction is not None else 0)


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    required = (
        HGB_BUNDLE, LONGITUDINAL_CANDIDATE, YAW_CORRECTOR,
        YAW_REFERENCE, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
        HGB_TRAIN_DATASET, HIGHSTEER_DATASET, PRACTICE_DATASET,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_hgb = joblib.load(HGB_BUNDLE)
    augmented = joblib.load(LONGITUDINAL_CANDIDATE)
    candidate_hgb = dict(reference_hgb)
    candidate_hgb["models"] = augmented["models"]
    candidate_hgb["training_run_ids"] = augmented["training_run_ids"]
    correction_bundle = joblib.load(YAW_CORRECTOR)
    correction = HighSteerYawResidual(
        correction_bundle["estimator"],
        tuple(correction_bundle["gate"]["rear_wheel_mean_mps"]))
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for model in list(atlas["global_event_models"].values()) + list(
            atlas["local_expert_models"].values()):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")

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
    heldout: dict[str, Any] = {}
    for name, data, windows, expected_split in evaluations:
        ids = np.asarray(data["run_ids"]).astype(str)
        selected_ids = {str(ids[row["run"]]) for row in windows}
        if (not selected_ids
                or selected_ids & set(candidate_hgb["training_run_ids"])):
            raise ValueError(f"{name} overlaps the augmented HGB training set")
        splits = np.asarray(data["splits"]).astype(str)
        if any(splits[row["run"]] != expected_split for row in windows):
            raise ValueError(f"unexpected split selected for {name}")

        no_correction = _score(
            data, candidate_hgb, edssm, metadata, atlas, windows, None)
        gated_correction = _score(
            data, candidate_hgb, edssm, metadata, atlas, windows, correction)
        heldout[name] = {
            "heldout_run_ids": sorted(no_correction["per_run"]),
            "windows": len(windows),
            "windows_by_run": gated_correction["windows_by_run"],
            "augmented_longitudinal_heads_atlas_yaw": {
                "macro_run_rmse": no_correction["macro_run_rmse"],
                "per_run": no_correction["per_run"],
            },
            "augmented_longitudinal_heads_gated_yaw": {
                "macro_run_rmse": gated_correction["macro_run_rmse"],
                "per_run": gated_correction["per_run"],
                "initial_yaw_step": gated_correction[
                    "yaw_atlas_one_step_selected_windows"],
            },
        }

    report = {
        "schema_version": 1,
        "purpose": "measure whether the high-steer one-step longitudinal/filtered-wheel head remains useful when composed with the gated frozen-yaw-atlas correction",
        "research_only": True,
        "simulator_was_run": False,
        "production_or_runtime_integration": False,
        "fit_or_selection_data": "none; existing training artifacts were frozen before this matched held-out comparison",
        "test_or_final_test_used": False,
        "longitudinal_candidate_heads_replaced": ["u", "filtered rear-left wheel", "filtered rear-right wheel"],
        "longitudinal_auxiliary_and_raw_encoder_heads_replaced": False,
        "yaw_correction_source": str(YAW_CORRECTOR.relative_to(ROOT)),
        "yaw_correction_start_step": 1,
        "data_and_model_hashes": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "heldout_results": heldout,
        "yaw_only_comparator": {
            "report": str(YAW_REFERENCE.relative_to(ROOT)),
            "report_sha256": _sha256(YAW_REFERENCE),
            "comparison": "same windows and rollouts are recorded in the referenced yaw-only report",
        },
        "limitations": [
            "the two practice runs are diagnostics and do not cover the high-steer training envelope",
            "this is a 5-second seeded rollout, not a full-lap offline simulation",
            "the high-steer set has three independent validation runs",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
