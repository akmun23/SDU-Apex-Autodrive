#!/usr/bin/env python3
"""Isolate recursive effects of the high-steer longitudinal refit's heads."""

from __future__ import annotations

import argparse
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
    OUTPUT_DIR,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


OUTPUT = OUTPUT_DIR / "recursive_head_ablation.json"
VARIANTS = {
    "reference_all_heads": (),
    "new_longitudinal_and_filtered_wheel_heads": ("main",),
    "new_actuator_feedback_heads": ("actuators",),
    "new_raw_encoder_head": ("raw_wheels",),
    "new_longitudinal_and_raw_encoder_heads": ("main", "raw_wheels"),
    "all_augmented_heads": ("main", "actuators", "raw_wheels"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    reference_path = HGB_BUNDLE
    candidate_path = OUTPUT_DIR / "longitudinal_raw_wheel_candidate.joblib"
    for path in (reference_path, candidate_path, EDSSM_CHECKPOINT,
                 YAW_ATLAS_BUNDLE, HGB_TRAIN_DATASET,
                 HIGHSTEER_DATASET, PRACTICE_DATASET):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = joblib.load(reference_path)
    candidate = joblib.load(candidate_path)
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for models in (atlas["global_event_models"].values(),
                   atlas["local_expert_models"].values()):
        for model in models:
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
    data_by_name = {
        "whole_run_validation": _load_dataset(HGB_TRAIN_DATASET),
        "highsteer_validation": _load_dataset(HIGHSTEER_DATASET),
        "unseen_practice_diagnostic": _load_dataset(PRACTICE_DATASET),
    }
    windows_by_name = {
        "whole_run_validation": _run_windows(
            data_by_name["whole_run_validation"], "validation",
            _bundle_lags(reference), WINDOWS_PER_RUN),
        "highsteer_validation": _windows(
            data_by_name["highsteer_validation"], HIGHSTEER_DATASET,
            HORIZON, _bundle_lags(reference)),
        "unseen_practice_diagnostic": _windows(
            data_by_name["unseen_practice_diagnostic"], PRACTICE_DATASET,
            HORIZON, _bundle_lags(reference)),
    }
    results: dict[str, Any] = {}
    for variant_name, heads in VARIANTS.items():
        model = dict(reference)
        if "main" in heads:
            model["models"] = candidate["models"]
        if "actuators" in heads:
            model["recursive_aux_models"] = candidate["recursive_aux_models"]
        if "raw_wheels" in heads:
            model["raw_wheel_models"] = candidate["raw_wheel_models"]
        model["training_run_ids"] = candidate["training_run_ids"]
        results[variant_name] = {}
        for split_name, data in data_by_name.items():
            scored = _rollout(
                data, model, edssm, metadata, atlas,
                windows_by_name[split_name], HORIZON,
                include_oracle_diagnostics=False,
                reconcile_composed_latent=True,
                latent_acceleration_scheme="current",
                atlas_acceleration_source="model_heads")
            results[variant_name][split_name] = {
                "windows": scored["windows"],
                "windows_by_run": scored["windows_by_run"],
                "heldout_run_ids": sorted(scored["per_run"]),
                "macro_run_rmse": scored["macro_run_rmse"],
                "per_run": scored["per_run"],
            }
    report = {
        "schema_version": 1,
        "purpose": "causally isolate which refitted HGB output head changes the recursive plant replay",
        "fit_or_selection_data": "none; all runs are held-out evaluation splits",
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_production_integration": False,
        "base_model_sha256": _sha256(reference_path),
        "augmented_candidate_sha256": _sha256(candidate_path),
        "edssm_sha256": _sha256(EDSSM_CHECKPOINT),
        "yaw_atlas_sha256": _sha256(YAW_ATLAS_BUNDLE),
        "head_combinations": {name: list(heads)
                              for name, heads in VARIANTS.items()},
        "window_counts": {name: len(rows)
                          for name, rows in windows_by_name.items()},
        "heldout_run_ids": {
            name: sorted({
                str(np.asarray(data["run_ids"]).astype(str)[row["run"]])
                for row in windows_by_name[name]})
            for name, data in data_by_name.items()},
        "results": results,
        "limitations": [
            "this is a five-second seeded rollout, not a full-lap offline-simulator validation",
            "held-out high-steer set has three runs and practice has two",
            "EDSSM lateral/roll and frozen yaw-atlas components are unchanged",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps({
        "output": str(args.output),
        "five_second": {
            variant: {
                split: row["macro_run_rmse"]["5.000s"]["hybrid_atlas"]
                for split, row in rows.items()}
            for variant, rows in report["results"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
