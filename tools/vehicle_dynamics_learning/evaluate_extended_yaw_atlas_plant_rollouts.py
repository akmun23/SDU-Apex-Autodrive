#!/usr/bin/env python3
"""Score the frozen-yaw-atlas plant through 12-second causal rollouts.

Compares the existing global HGB longitudinal heads with the already-fitted
gain-0.5 local high-steer correction. Both retain the same frozen yaw atlas,
EDSSM lateral/roll transition, yaw corrector, initial conditions and command
replay. Whole-run validation, high-steer validation, and practice captures are
scored; test/final-test runs are excluded. No simulator or runtime is changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as evaluator
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    _run_windows,
)
from tools.vehicle_dynamics_learning.fit_highsteer_augmented_longitudinal_candidate import (
    OUTPUT_DIR as LONGITUDINAL_CANDIDATE_DIR,
)
from tools.vehicle_dynamics_learning.fit_highsteer_recursive_longitudinal_residual import (
    MODEL_OUTPUT as RESIDUAL_MODEL,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
    HighSteerYawResidual,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
LONGITUDINAL_CANDIDATE = (
    LONGITUDINAL_CANDIDATE_DIR / "longitudinal_raw_wheel_candidate.joblib")
YAW_CORRECTOR_PATH = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_atlas_dagger_highsteer_gated_v3_20261009.joblib")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "extended_yaw_atlas_plant_rollouts_20261009")
HORIZONS = (30, 80, 200, 300, 400, 480)
HORIZON_STEPS = max(HORIZONS)
MAX_WINDOWS_PER_RUN = 8
STATE_SERIES = {
    "u_mps": (0,),
    "v_mps": (1,),
    "yaw_rate_rps": (2,),
    "filtered_rear_wheel_pair_mps": (5, 6),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _trace_curves(trace: dict[str, np.ndarray],
                  run_ids: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(trace["predicted_state"], dtype=np.float64)
    truth = np.asarray(trace["truth_state"], dtype=np.float64)
    predicted_pose = np.asarray(trace["predicted_pose"], dtype=np.float64)
    truth_pose = np.asarray(trace["truth_pose"], dtype=np.float64)
    run_index = np.asarray(trace["run_indices"], dtype=np.int64)
    if predicted.shape != truth.shape or predicted.shape[-1] != 7:
        raise ValueError("rollout trace state arrays are not aligned")
    if predicted_pose.shape != truth_pose.shape or predicted_pose.shape[-1] != 3:
        raise ValueError("rollout trace pose arrays are not aligned")
    position = np.linalg.norm(
        predicted_pose[..., :2] - truth_pose[..., :2], axis=-1)
    pose_yaw = np.abs(_wrap(predicted_pose[..., 2] - truth_pose[..., 2]))
    output: dict[str, Any] = {}
    for step in range(predicted.shape[1]):
        per_run: dict[str, Any] = {}
        for run in sorted(np.unique(run_index).tolist()):
            selected = run_index == run
            row: dict[str, float] = {
                "position_rmse_m": float(np.sqrt(np.mean(
                    position[selected, step] ** 2))),
                "pose_yaw_rmse_rad": float(np.sqrt(np.mean(
                    pose_yaw[selected, step] ** 2))),
            }
            state_error = predicted[selected, step] - truth[selected, step]
            for name, columns in STATE_SERIES.items():
                row[name + "_rmse"] = float(np.sqrt(np.mean(
                    state_error[:, columns] ** 2)))
            per_run[str(run_ids[run])] = row
        output[str((step + 1) * evaluator.DT_S)] = {
            metric: float(np.mean([row[metric] for row in per_run.values()]))
            for metric in next(iter(per_run.values()))
        }
    # Compact the full 40 Hz trace to 0.25 s reporting intervals, while keeping
    # every integration step represented in the underlying scoring.
    return {
        f"{step * evaluator.DT_S:.3f}s": output[str(step * evaluator.DT_S)]
        for step in range(10, predicted.shape[1] + 1, 10)
    }


def _score_model(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
                 metadata: dict[str, Any], atlas: dict[str, Any],
                 yaw_model: Any, windows: list[dict[str, int]],
                 residual: dict[str, Any] | None,
                 horizon_steps: int) -> dict[str, Any]:
    result = evaluator._rollout(
        data, hgb, edssm, metadata, atlas, windows, horizon_steps,
        yaw_delta_gain=1.0,
        include_oracle_diagnostics=False,
        reconcile_composed_latent=True,
        yaw_correction_model=yaw_model,
        yaw_correction_start_step=CORRECTION_START_STEP,
        hgb_main_gate=(None if residual is None else residual["gate"]),
        hgb_main_residual_models=(None if residual is None
                                  else residual["models"]),
        hgb_main_residual_start_step=1,
        hgb_main_residual_gain=(0.0 if residual is None
                                else float(residual["gain"])),
        capture_trajectory_trace=True)
    result["state_error_curve_macro_run_rmse"] = _trace_curves(
        result.pop("_trajectory_trace"),
        np.asarray(data["run_ids"]).astype(str))
    return result


def run() -> dict[str, Any]:
    report_path = OUTPUT_DIR / "evaluation.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    required = (
        evaluator.HGB_BUNDLE, LONGITUDINAL_CANDIDATE,
        evaluator.EDSSM_CHECKPOINT, evaluator.YAW_ATLAS_BUNDLE,
        YAW_CORRECTOR_PATH, RESIDUAL_MODEL, evaluator.HGB_TRAIN_DATASET,
        evaluator.HIGHSTEER_DATASET, evaluator.PRACTICE_DATASET,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = joblib.load(evaluator.HGB_BUNDLE)
    longitudinal = joblib.load(LONGITUDINAL_CANDIDATE)
    hgb = dict(parent)
    hgb["models"] = longitudinal["models"]
    hgb["training_run_ids"] = longitudinal["training_run_ids"]
    residual = joblib.load(RESIDUAL_MODEL)
    yaw_bundle = joblib.load(YAW_CORRECTOR_PATH)
    yaw_model = HighSteerYawResidual(
        yaw_bundle["estimator"],
        tuple(yaw_bundle["gate"]["rear_wheel_mean_mps"]))
    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")
    evaluator.HORIZONS = HORIZONS

    cohorts = (
        ("broad_openplane_validation", evaluator.HGB_TRAIN_DATASET,
         "validation", "run_windows", HORIZON_STEPS),
        ("highsteer_validation", evaluator.HIGHSTEER_DATASET,
         "validation", "matched_highsteer_windows", HORIZON_STEPS),
        # The saved practice captures are segmented into ~6 s sequences; the
        # longest complete causal window is 5 s, not a full 12 s rollout.
        ("unseen_practice_diagnostic", evaluator.PRACTICE_DATASET,
         "unseen_practice", "run_windows", 200),
    )
    results: dict[str, Any] = {}
    for label, path, split, window_method, horizon_steps in cohorts:
        data = _load_dataset(path)
        splits = set(np.asarray(data["splits"]).astype(str))
        if split not in splits:
            raise ValueError(f"{path} lacks expected split {split!r}")
        if window_method == "matched_highsteer_windows":
            windows = evaluator._windows(
                data, path, horizon_steps, evaluator._bundle_lags(hgb))
        else:
            windows = _run_windows(
                data, split, evaluator._bundle_lags(hgb),
                MAX_WINDOWS_PER_RUN, horizon_steps=horizon_steps)
        bounds = np.asarray(data["bounds"], dtype=np.int64)
        windows = [row for row in windows
                   if int(row["start"]) + horizon_steps
                   < int(bounds[row["sequence"], 1])]
        if not windows:
            raise ValueError(f"no complete {horizon_steps}-step windows in {label}")
        selected_runs = {str(data["run_ids"][row["run"]])
                         for row in windows}
        run_splits = np.asarray(data["splits"]).astype(str)
        if any(run_splits[row["run"]] != split for row in windows):
            raise ValueError(f"unexpected split selection in {label}")
        if selected_runs & set(residual["training_run_ids"]):
            raise ValueError(f"training-run leakage in {label}")
        evaluator.HORIZONS = tuple(step for step in HORIZONS
                                   if step <= horizon_steps)
        if not evaluator.HORIZONS or evaluator.HORIZONS[-1] != horizon_steps:
            raise ValueError(f"checkpoint schedule omits final horizon for {label}")
        print(f"{label}: scoring parent on {len(windows)} windows "
              f"through {horizon_steps * evaluator.DT_S:.2f}s", flush=True)
        parent_result = _score_model(
            data, hgb, edssm, metadata, atlas, yaw_model, windows, None,
            horizon_steps)
        print(f"{label}: scoring gated residual on {len(windows)} windows",
              flush=True)
        residual_result = _score_model(
            data, hgb, edssm, metadata, atlas, yaw_model, windows, residual,
            horizon_steps)
        results[label] = {
            "split": split,
            "run_ids": sorted(selected_runs),
            "windows": len(windows),
            "windows_by_run": {
                run_id: sum(str(data["run_ids"][row["run"]]) == run_id
                            for row in windows)
                for run_id in sorted(selected_runs)},
            "horizon_s": horizon_steps * evaluator.DT_S,
            "parent": parent_result,
            "highsteer_residual_gain_0p5": residual_result,
        }

    report = {
        "schema_version": 1,
        "purpose": "extend the existing frozen-yaw-atlas causal plant evaluation from 5 seconds to 12 seconds and locate the first state errors that grow",
        "yaw_atlas_bundle_frozen": True,
        "yaw_atlas_recurrent_correction_frozen": True,
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "future_sensor_or_truth_input": False,
        "step_dt_s": evaluator.DT_S,
        "checkpoint_steps_by_cohort": {
            name: [int(round(float(key[:-1]) / evaluator.DT_S))
                   for key in item["parent"]["macro_run_rmse"]]
            for name, item in results.items()},
        "input_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in required},
        "results": results,
        "limitations": [
            "initial plant history and initial states are simulator truth; all subsequent predictions are recursive and causal",
            "the residual gate is based on predicted body speed and predicted steering feedback during rollout",
            "practice has only two whole-run captures and is transfer evidence, not a precise population estimate",
            "12 seconds is longer than the target lap, but does not by itself establish full-lap map/trajectory fidelity",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    report = run()
    print(json.dumps({
        "output": str(OUTPUT_DIR / "evaluation.json"),
        "cohorts": {
            name: {
                "runs": len(item["run_ids"]),
                "windows": item["windows"],
                "parent": item["parent"]["macro_run_rmse"],
                "highsteer_residual_gain_0p5": item[
                    "highsteer_residual_gain_0p5"]["macro_run_rmse"],
            }
            for name, item in report["results"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
