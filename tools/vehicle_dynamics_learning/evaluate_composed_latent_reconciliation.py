#!/usr/bin/env python3
"""Evaluate a composition-consistent latent update with frozen model heads.

The only candidate change is how the EDSSM latent state is advanced after the
hybrid plant selects HGB longitudinal/wheel outputs and the frozen yaw-atlas
output. The old, inconsistent result is read from its already completed report;
candidate scores use the same held-out windows and no future sensors/truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    DT_S,
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
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


PREVIOUS = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "highsteer_practice_input_sensitivity_evaluation.json")
GAIN_REPORT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_delta_gain_sweep_20261009.json")
OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "composed_latent_reconciliation_evaluation_20261009.json")
GAINS = (1.0, 0.5)
WINDOWS_PER_RUN = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_thread_estimators(bundle: dict[str, Any]) -> int:
    """Avoid parallel-job overhead for many small local-expert batches."""
    models = (list(bundle["global_event_models"].values())
              + list(bundle["local_expert_models"].values()))
    changed = 0
    for model in models:
        if hasattr(model, "n_jobs") and model.n_jobs != 1:
            model.n_jobs = 1
            changed += 1
    return changed


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    for path in (PREVIOUS, GAIN_REPORT, HGB_BUNDLE, EDSSM_CHECKPOINT,
                 YAW_ATLAS_BUNDLE, HGB_TRAIN_DATASET,
                 HIGHSTEER_DATASET, PRACTICE_DATASET):
        if not path.is_file():
            raise FileNotFoundError(path)

    previous = json.loads(PREVIOUS.read_text())
    gain_report = json.loads(GAIN_REPORT.read_text())
    hgb = joblib.load(HGB_BUNDLE)
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    serial_models = _single_thread_estimators(atlas)
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")

    train_data = _load_dataset(HGB_TRAIN_DATASET)
    train_splits = np.asarray(train_data["splits"]).astype(str)
    run_ids = np.asarray(train_data["run_ids"]).astype(str)
    validation_ids = set(run_ids[train_splits == "validation"])
    if validation_ids & (set(hgb["training_run_ids"])
                         | set(metadata["training_runs"])):
        raise ValueError("whole-run validation overlaps a base model's training runs")
    validation_windows = _run_windows(
        train_data, "validation", _bundle_lags(hgb), WINDOWS_PER_RUN)

    highsteer_data = _load_dataset(HIGHSTEER_DATASET)
    highsteer_windows = _windows(
        highsteer_data, HIGHSTEER_DATASET, HORIZON, _bundle_lags(hgb))
    practice_data = _load_dataset(PRACTICE_DATASET)
    practice_windows = _windows(
        practice_data, PRACTICE_DATASET, HORIZON, _bundle_lags(hgb))

    specifications = (
        ("whole_run_validation", train_data, validation_windows),
        ("highsteer_validation", highsteer_data, highsteer_windows),
        ("unseen_practice_diagnostic", practice_data, practice_windows),
    )
    candidate: dict[str, Any] = {}
    for name, data, windows in specifications:
        candidate[name] = {}
        for gain in GAINS:
            result = _rollout(
                data, hgb, edssm, metadata, atlas, windows, HORIZON,
                yaw_delta_gain=gain,
                include_oracle_diagnostics=False,
                reconcile_composed_latent=True)
            candidate[name][str(gain)] = {
                "windows": result["windows"],
                "windows_by_run": result["windows_by_run"],
                "heldout_run_ids": sorted(result["per_run"]),
                "macro_run_rmse": result["macro_run_rmse"],
                "per_run": result["per_run"],
            }

    report = {
        "schema_version": 1,
        "purpose": "isolate correction of the EDSSM hidden-state transition to match the composed HGB + EDSSM lateral + frozen yaw-atlas plant",
        "research_only": True,
        "production_or_runtime_integration": False,
        "simulator_was_run": False,
        "frozen_base_models": {
            "hgb_sha256": _sha256(HGB_BUNDLE),
            "edssm_sha256": _sha256(EDSSM_CHECKPOINT),
            "yaw_atlas_sha256": _sha256(YAW_ATLAS_BUNDLE),
            "yaw_atlas_in_memory_n_jobs_set_to_one": serial_models,
        },
        "previous_inconsistent_composition_report": {
            "path": str(PREVIOUS.relative_to(ROOT)),
            "sha256": _sha256(PREVIOUS),
        },
        "training_selected_yaw_gain_report": {
            "path": str(GAIN_REPORT.relative_to(ROOT)),
            "sha256": _sha256(GAIN_REPORT),
            "selected_gain": gain_report["fit_protocol"]["selected_gain"],
            "gain_selection_used_training_only": True,
        },
        "candidate_change": {
            "latent_recomputed_from_composed_state_and_command": True,
            "equations": {
                "ax": "(u_next-u)/dt - r*v",
                "ay": "(v_next-v)/dt + r*u",
                "yaw_acceleration": "(r_next-r)/dt",
                "wheel_acceleration": "selected wheel state increment; contact-relative derivative only if checkpoint uses contact_slip mode",
            },
            "yaw_atlas_is_frozen": True,
            "hgb_and_edssm_weights_frozen": True,
            "future_sensor_or_truth_input": False,
            "same_evaluation_windows_as_prior_comparison": True,
        },
        "previous_inconsistent_composition": {
            "highsteer_validation": previous["results"]["highsteer_validation"][
                "macro_run_rmse"],
            "unseen_practice_diagnostic": previous["results"]["unseen_practice"][
                "macro_run_rmse"],
            "whole_run_validation": gain_report["heldout_results"][
                "whole_run_validation"]["baseline_atlas_gain_1"],
        },
        "candidate_results": candidate,
        "datasets": {
            "whole_run_validation_sha256": _sha256(HGB_TRAIN_DATASET),
            "highsteer_validation_sha256": _sha256(HIGHSTEER_DATASET),
            "practice_diagnostic_sha256": _sha256(PRACTICE_DATASET),
        },
        "test_or_final_test_used": False,
        "limitations": [
            "only three independent high-steering runs and two practice diagnostic runs",
            "the existing validation captures informed previous candidate comparisons",
            "five-second rollouts do not establish a full-lap offline simulator",
            "the 0.5 gain was chosen from training runs and is evaluated here only as a fixed diagnostic comparator",
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
        "candidate_results": {
            group: {gain: data["macro_run_rmse"]
                   for gain, data in scores.items()}
            for group, scores in report["candidate_results"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
