#!/usr/bin/env python3
"""Test the existing causal HGB next-v head in gated recursive rollouts.

The model, yaw atlas and longitudinal components are frozen. Only the source
of the hybrid plant's next lateral velocity changes inside the measured
high-steer support; outside it, the EDSSM transition remains in place.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    EDSSM_CHECKPOINT,
    HGB_BUNDLE,
    HIGHSTEER_DATASET,
    PRACTICE_DATASET,
    YAW_ATLAS_BUNDLE,
)
from tools.vehicle_dynamics_learning.fit_highsteer_augmented_longitudinal_candidate import (
    OUTPUT_DIR as LONGITUDINAL_CANDIDATE_DIR,
)
from tools.vehicle_dynamics_learning.fit_highsteer_lateral_acceleration_atlas_candidate import (
    HIGHSTEER_TRAIN_DATASET,
    LOCAL_GATE,
    OUTPUT_DIR as AY_EXPERIMENT_DIR,
    _compare_recursive,
    _recursive,
)
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_longitudinal_residual import (
    MODEL_OUTPUT as LONGITUDINAL_RESIDUAL_BUNDLE,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


YAW_CORRECTOR_PATH = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "gated_hgb_v_increment_rollout_atlas_accel_ablation_20261009")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict[str, Any]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT_DIR}")
    started = time.time()
    parent_hgb = joblib.load(HGB_BUNDLE)
    longitudinal_candidate_path = (
        LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib")
    longitudinal_candidate = joblib.load(longitudinal_candidate_path)
    hgb = dict(parent_hgb)
    hgb["models"] = longitudinal_candidate["models"]
    hgb["training_run_ids"] = longitudinal_candidate["training_run_ids"]
    longitudinal_residual = joblib.load(LONGITUDINAL_RESIDUAL_BUNDLE)
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

    results: dict[str, Any] = {}
    for label, dataset_path, split in (
            ("highsteer_validation", HIGHSTEER_DATASET, "validation"),
            ("unseen_practice", PRACTICE_DATASET, "unseen_practice")):
        data = _load_dataset(dataset_path)
        if split not in set(np.asarray(data["splits"]).astype(str)):
            raise ValueError(f"missing required {split} split in {dataset_path}")
        baseline = _recursive(
            data, dataset_path, hgb, edssm, metadata, atlas, yaw_model,
            yaw_start_step, longitudinal_residual, None, "base")
        candidate = _recursive(
            data, dataset_path, hgb, edssm, metadata, atlas, yaw_model,
            yaw_start_step, longitudinal_residual, None, "base",
            lateral_v_update="gated_hgb_v_increment")
        results[label] = {
            "split": split,
            "baseline": baseline,
            "gated_hgb_v_increment": candidate,
            "paired_candidate_minus_baseline": _compare_recursive(
                candidate, baseline),
        }
        if label == "highsteer_validation":
            alternate_baseline = _recursive(
                data, dataset_path, hgb, edssm, metadata, atlas, yaw_model,
                yaw_start_step, longitudinal_residual, None, "base",
                atlas_acceleration_source="model_heads")
            alternate_candidate = _recursive(
                data, dataset_path, hgb, edssm, metadata, atlas, yaw_model,
                yaw_start_step, longitudinal_residual, None, "base",
                lateral_v_update="gated_hgb_v_increment",
                atlas_acceleration_source="model_heads")
            results["highsteer_validation_model_heads_acceleration"] = {
                "baseline": alternate_baseline,
                "gated_hgb_v_increment": alternate_candidate,
                "paired_candidate_minus_baseline": _compare_recursive(
                    alternate_candidate, alternate_baseline),
            }

    ay_report = AY_EXPERIMENT_DIR / "evaluation.json"
    one_step_report = json.loads(ay_report.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "purpose": "test whether the existing direct next-v transition head is useful in causal recursive high-steer plant rollouts",
        "method": "replace EDSSM next-v only while predicted speed and steering are inside the measured support; use the frozen HGB auxiliary v-increment head",
        "support_gate": LOCAL_GATE,
        "one_step_reference": {
            "source_report": str(ay_report.relative_to(ROOT)),
            "heldout_highsteer_parent_v_increment": one_step_report[
                "one_step_whole_run_validation"]["base"]["metrics"][
                    "parent_v_increment"],
        },
        "whole_run_recursive_evaluations": results,
        "training_sources": [
            str(HGB_BUNDLE.relative_to(ROOT)),
            str(HIGHSTEER_TRAIN_DATASET.relative_to(ROOT)),
        ],
        "frozen_components": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (HGB_BUNDLE, longitudinal_candidate_path,
                         LONGITUDINAL_RESIDUAL_BUNDLE, YAW_ATLAS_BUNDLE,
                         YAW_CORRECTOR_PATH, EDSSM_CHECKPOINT)},
        "test_and_final_test_used": False,
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "the HGB v transition is simulator-supervised and evaluated as an offline plant component, not a sensor-only odometry update",
            "three independent high-steer validation captures and two practice captures limit uncertainty",
            "the local candidate is gated and does not model behavior outside 7-9 m/s and 0.30-0.525 rad",
            "five-second windows do not establish full-lap simulator accuracy",
        ],
    }
    (OUTPUT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run()
    summary: dict[str, Any] = {}
    for split, values in report["whole_run_recursive_evaluations"].items():
        summary[split] = {
            horizon: {
                "baseline_v_rmse_mps": values["baseline"][
                    "trajectory_metrics"][horizon]["macro_run"]["v_rmse_mps"],
                "gated_hgb_v_rmse_mps": values["gated_hgb_v_increment"][
                    "trajectory_metrics"][horizon]["macro_run"]["v_rmse_mps"],
                "baseline_position_rmse_m": values["baseline"][
                    "trajectory_metrics"][horizon]["macro_run"]["position_rmse_m"],
                "candidate_position_rmse_m": values["gated_hgb_v_increment"][
                    "trajectory_metrics"][horizon]["macro_run"]["position_rmse_m"],
            }
            for horizon in values["baseline"]["trajectory_metrics"]}
    print(json.dumps({"output": str(OUTPUT_DIR / "evaluation.json"),
                      "results": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
