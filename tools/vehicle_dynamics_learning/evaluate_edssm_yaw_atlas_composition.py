#!/usr/bin/env python3
"""Ablate the HGB/EDSSM splice while retaining the frozen yaw atlas.

The previous 12-second replay found lateral-state divergence in the hybrid
that replaces EDSSM forward speed and wheel states with HGB outputs. This
research comparison instead lets the frozen EDSSM own every non-yaw state and
replaces only yaw rate with the same frozen yaw atlas/corrector. It reuses the
same held-out windows and scoring as the completed comparison report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as evaluator
from tools.vehicle_dynamics_learning.evaluate_extended_yaw_atlas_plant_rollouts import (
    HORIZONS as EXTENDED_HORIZONS,
    MAX_WINDOWS_PER_RUN,
    OUTPUT_DIR as REFERENCE_DIR,
    YAW_CORRECTOR_PATH,
    _sha256,
    _trace_curves,
)
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    _run_windows,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_REPORT = REFERENCE_DIR / "evaluation.json"
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "edssm_yaw_atlas_composition_20261009")
HORIZON_BY_COHORT = {
    "broad_openplane_validation": 480,
    "highsteer_validation": 480,
    "unseen_practice_diagnostic": 200,
}


def _score(data: dict[str, Any], atlas: dict[str, Any], edssm: Any,
           metadata: dict[str, Any], yaw_model: Any, hgb: dict[str, Any],
           windows: list[dict[str, int]], horizon: int) -> dict[str, Any]:
    evaluator.HORIZONS = tuple(step for step in EXTENDED_HORIZONS
                               if step <= horizon)
    result = evaluator._rollout(
        data, hgb, edssm, metadata, atlas,
        windows, horizon,
        yaw_delta_gain=1.0,
        include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        atlas_acceleration_source="composed_transition",
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=CORRECTION_START_STEP,
        capture_trajectory_trace=True,
        motion_source="edssm_non_yaw")
    result["state_error_curve_macro_run_rmse"] = _trace_curves(
        result.pop("_trajectory_trace"),
        np.asarray(data["run_ids"]).astype(str))
    return result


def run() -> dict[str, Any]:
    output = OUTPUT_DIR / "evaluation.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not REFERENCE_REPORT.is_file():
        raise FileNotFoundError(REFERENCE_REPORT)
    reference = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    required = (
        evaluator.HGB_BUNDLE, evaluator.EDSSM_CHECKPOINT,
        evaluator.YAW_ATLAS_BUNDLE, YAW_CORRECTOR_PATH,
        evaluator.HGB_TRAIN_DATASET, evaluator.HIGHSTEER_DATASET,
        evaluator.PRACTICE_DATASET,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if reference["input_hashes"].get(
            str(evaluator.YAW_ATLAS_BUNDLE.relative_to(ROOT))) != _sha256(
                evaluator.YAW_ATLAS_BUNDLE):
        raise ValueError("frozen yaw atlas differs from the reference run")

    parent_hgb = joblib.load(evaluator.HGB_BUNDLE)
    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")

    cohorts = (
        ("broad_openplane_validation", evaluator.HGB_TRAIN_DATASET,
         "validation", False),
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET,
         "validation", True),
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", False),
    )
    results: dict[str, Any] = {}
    for label, path, split, highsteer in cohorts:
        data = _load_dataset(path)
        horizon = HORIZON_BY_COHORT[label]
        if highsteer:
            windows = evaluator._windows(
                data, path, horizon, evaluator._bundle_lags(parent_hgb))
        else:
            windows = _run_windows(
                data, split, evaluator._bundle_lags(parent_hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon)
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if row["start"] + horizon < bounds[row["sequence"], 1]]
        run_splits = np.asarray(data["splits"]).astype(str)
        if not windows or any(run_splits[row["run"]] != split for row in windows):
            raise ValueError(f"invalid or absent held-out windows for {label}")
        prior = reference["results"][label]
        if (len(windows) != prior["windows"]
                or sorted({str(data["run_ids"][row["run"]])
                           for row in windows}) != prior["run_ids"]):
            raise ValueError(f"window cohort differs from reference for {label}")
        print(f"{label}: EDSSM non-yaw state + atlas yaw, "
              f"{len(windows)} windows / {horizon * evaluator.DT_S:.2f}s",
              flush=True)
        candidate = _score(data, atlas, edssm, metadata, yaw_model, parent_hgb,
                           windows, horizon)
        reference_model = prior["parent"]
        results[label] = {
            "split": split,
            "run_ids": prior["run_ids"],
            "windows": len(windows),
            "windows_by_run": prior["windows_by_run"],
            "horizon_s": horizon * evaluator.DT_S,
            "hgb_edssm_yaw_atlas_reference": {
                "macro_run_rmse": reference_model["macro_run_rmse"],
                "state_error_curve_macro_run_rmse": (
                    reference_model["state_error_curve_macro_run_rmse"]),
            },
            "edssm_non_yaw_with_frozen_yaw_atlas": candidate,
        }

    report = {
        "schema_version": 1,
        "purpose": "test whether the HGB longitudinal/EDSSM lateral state splice causes recursive divergence while preserving the frozen yaw atlas",
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "yaw_atlas_frozen": True,
        "yaw_corrector_frozen": True,
        "edssm_frozen": True,
        "non_yaw_state_source": "frozen EDSSM for u, v, steering, throttle, both filtered rear wheels and roll; yaw rate replaced by frozen yaw atlas",
        "atlas_acceleration_source": "composed EDSSM transition with atlas yaw",
        "future_sensor_or_truth_input": False,
        "reference_report_sha256": _sha256(REFERENCE_REPORT),
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "results": results,
        "limitations": [
            "initial measured histories are simulator truth; all rollout steps after initialization are causal",
            "the 12-second scores are open-plane/high-steer captures; practice data only support two five-second windows",
            "this compares offline model compositions; it is not production odometry/MPC integration or a simulator change",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main() -> None:
    report_path = OUTPUT_DIR / "evaluation.json"
    report = (json.loads(report_path.read_text(encoding="utf-8"))
              if report_path.is_file() else run())
    summary = {}
    for cohort, values in report["results"].items():
        summary[cohort] = {}
        for model_name in (
                "hgb_edssm_yaw_atlas_reference",
                "edssm_non_yaw_with_frozen_yaw_atlas"):
            curve = values[model_name]["state_error_curve_macro_run_rmse"]
            summary[cohort][model_name] = {
                endpoint: curve[endpoint]
                for endpoint in ("0.250s", "0.750s", "2.000s", "5.000s",
                                 "12.000s") if endpoint in curve}
    print(json.dumps({"output": str(OUTPUT_DIR / "evaluation.json"),
                      "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
