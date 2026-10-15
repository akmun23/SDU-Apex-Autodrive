#!/usr/bin/env python3
"""Locate state-by-state divergence for the frozen-atlas yaw residual.

Runs paired causal offline rollouts on the exact existing validation/practice
cohorts and windows used by the full-regime yaw-residual evaluation. It does
not fit a model, access test/final-test scores, launch the simulator, or alter
the plant. The initial state/history is measured; subsequent model inputs are
generated recursively, with the recorded command sequence replayed identically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning import evaluate_hybrid_longitudinal_edssm_candidate as evaluator
from tools.vehicle_dynamics_learning.fit_full_regime_yaw_atlas_dagger_corrector import (
    OUTPUT_DIR as FIT_DIR,
    _validation_sources,
)
from tools.vehicle_dynamics_learning.fit_recursive_yaw_atlas_dagger_corrector import (
    CORRECTION_START_STEP,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model


OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_regime_yaw_first_divergence_20261011")
REPORT_PATH = OUTPUT_DIR / "evaluation.json"
DT_S = evaluator.DT_S
REPORT_INTERVAL_STEPS = 10
PREDICTION_DIFF_TOLERANCE = 1e-6
STATE_COLUMNS = {
    "forward_speed_u_mps": (0,),
    "lateral_speed_v_mps": (1,),
    "yaw_rate_rps": (2,),
    "steering_feedback_rad": (3,),
    "throttle_feedback": (4,),
    "rear_wheel_pair_mps": (5, 6),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _first_change(values: np.ndarray, tolerance: float) -> dict[str, Any]:
    """First time a causal state prediction differs from baseline materially."""
    reduce_axes = tuple(axis for axis in range(values.ndim) if axis != 1)
    active = np.any(np.abs(values) > tolerance, axis=reduce_axes)
    indices = np.flatnonzero(active)
    if not len(indices):
        return {"step": None, "time_s": None}
    step = int(indices[0]) + 1
    return {"step": step, "time_s": step * DT_S}


def _first_change_by_run(values: np.ndarray, run_indices: np.ndarray,
                         run_ids: np.ndarray, tolerance: float
                         ) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for run in sorted(np.unique(run_indices).tolist()):
        selected = run_indices == run
        output[str(run_ids[run])] = _first_change(values[selected], tolerance)
    return output


def _error_curve(predicted: np.ndarray, truth: np.ndarray,
                 run_indices: np.ndarray, run_ids: np.ndarray,
                 columns: tuple[int, ...]) -> dict[str, Any]:
    error = predicted[..., columns] - truth[..., columns]
    by_run: dict[str, np.ndarray] = {}
    for run in sorted(np.unique(run_indices).tolist()):
        selected = run_indices == run
        by_run[str(run_ids[run])] = np.sqrt(np.mean(
            np.square(error[selected], dtype=np.float64), axis=(0, 2)))
    curve = np.mean(np.stack(list(by_run.values())), axis=0)
    return {
        "macro_run_rmse_by_step": curve.tolist(),
        "per_run_final_rmse": {
            run: float(values[-1]) for run, values in by_run.items()},
    }


def _pose_error_curve(trace: dict[str, np.ndarray],
                      run_indices: np.ndarray, run_ids: np.ndarray
                      ) -> dict[str, Any]:
    predicted = np.asarray(trace["predicted_pose"], dtype=np.float64)
    truth = np.asarray(trace["truth_pose"], dtype=np.float64)
    difference = predicted - truth
    difference[..., 2] = _wrap(difference[..., 2])
    by_run_position: dict[str, np.ndarray] = {}
    by_run_yaw: dict[str, np.ndarray] = {}
    for run in sorted(np.unique(run_indices).tolist()):
        selected = run_indices == run
        by_run_position[str(run_ids[run])] = np.sqrt(np.mean(
            np.square(difference[selected, :, :2]), axis=(0, 2)))
        by_run_yaw[str(run_ids[run])] = np.sqrt(np.mean(
            np.square(difference[selected, :, 2]), axis=0))
    return {
        "position_rmse_m": {
            "macro_run_rmse_by_step": np.mean(
                np.stack(list(by_run_position.values())), axis=0).tolist(),
            "per_run_final_rmse": {
                run: float(values[-1])
                for run, values in by_run_position.items()},
        },
        "pose_yaw_rmse_rad": {
            "macro_run_rmse_by_step": np.mean(
                np.stack(list(by_run_yaw.values())), axis=0).tolist(),
            "per_run_final_rmse": {
                run: float(values[-1]) for run, values in by_run_yaw.items()},
        },
    }


def _raw_wheel_error_curve(trace: dict[str, np.ndarray],
                           run_indices: np.ndarray, run_ids: np.ndarray
                           ) -> dict[str, Any]:
    predicted = np.asarray(
        trace["predicted_raw_encoder_wheels"], dtype=np.float64)
    truth = np.asarray(trace["truth_raw_encoder_wheels"], dtype=np.float64)
    valid = np.asarray(trace["raw_encoder_target_valid"], dtype=bool)
    by_run: dict[str, np.ndarray] = {}
    by_run_valid: dict[str, np.ndarray] = {}
    for run in sorted(np.unique(run_indices).tolist()):
        selected = run_indices == run
        run_curve = np.full(predicted.shape[1], np.nan, dtype=np.float64)
        run_valid = np.zeros(predicted.shape[1], dtype=np.float64)
        for step in range(predicted.shape[1]):
            mask = selected & valid[:, step]
            run_valid[step] = float(np.mean(valid[selected, step]))
            if np.any(mask):
                error = predicted[mask, step] - truth[mask, step]
                run_curve[step] = _rmse(error)
        by_run[str(run_ids[run])] = run_curve
        by_run_valid[str(run_ids[run])] = run_valid
    stacked = np.stack(list(by_run.values()))
    counts = np.sum(np.isfinite(stacked), axis=0)
    macro = np.divide(np.nansum(stacked, axis=0), counts,
                      out=np.full(stacked.shape[1], np.nan), where=counts > 0)
    return {
        "macro_run_rmse_by_step": macro.tolist(),
        "per_run_rmse_by_step": {
            run: values.tolist() for run, values in by_run.items()},
        "per_run_valid_fraction_by_step": {
            run: values.tolist() for run, values in by_run_valid.items()},
        "per_run_final_rmse": {
            run: (None if not np.isfinite(values[-1]) else float(values[-1]))
            for run, values in by_run.items()},
    }


def _cohort_diagnostic(source: dict[str, Any], hgb: dict[str, Any],
                       edssm: Any, metadata: dict[str, Any],
                       atlas: dict[str, Any], correction: Any
                       ) -> dict[str, Any]:
    data = source["data"]
    windows = source["windows"]
    horizon = source["horizon"]
    run_ids = np.asarray(data["run_ids"]).astype(str)
    evaluator.HORIZONS = tuple(range(
        REPORT_INTERVAL_STEPS, horizon + 1, REPORT_INTERVAL_STEPS))
    common = {
        "yaw_delta_gain": 1.0,
        "include_oracle_diagnostics": False,
        "reconcile_composed_latent": True,
        "atlas_acceleration_source": "composed_transition",
        "lateral_v_update": "edssm",
        "capture_trajectory_trace": True,
    }
    baseline = evaluator._rollout(
        data, hgb, edssm, metadata, atlas, windows, horizon,
        **common)
    candidate = evaluator._rollout(
        data, hgb, edssm, metadata, atlas, windows, horizon,
        yaw_correction_model=correction,
        yaw_correction_start_step=CORRECTION_START_STEP,
        **common)
    baseline_trace = baseline.pop("_trajectory_trace")
    candidate_trace = candidate.pop("_trajectory_trace")
    for key in ("source_indices", "run_indices"):
        if not np.array_equal(baseline_trace[key], candidate_trace[key]):
            raise ValueError(f"paired trace {key} differs in {source['name']}")
    run_indices = np.asarray(baseline_trace["run_indices"], dtype=np.int64)
    if not np.array_equal(run_indices, np.asarray([
            row["run"] for row in windows], dtype=np.int64)):
        raise ValueError("trace run order does not match selected windows")
    predicted0 = np.asarray(baseline_trace["predicted_state"], dtype=np.float64)
    predicted1 = np.asarray(candidate_trace["predicted_state"], dtype=np.float64)
    truth = np.asarray(baseline_trace["truth_state"], dtype=np.float64)
    delta = predicted1 - predicted0
    output: dict[str, Any] = {
        "split": source["split"],
        "run_ids": source["run_ids"],
        "windows": len(windows),
        "horizon_s": horizon * DT_S,
        "same_initial_states_and_command_replay": True,
        "state_order": [
            "u", "v", "yaw_rate", "steering_feedback",
            "throttle_feedback", "rear_left_speed", "rear_right_speed"],
        "first_prediction_change_candidate_vs_baseline": {},
        "first_change_by_run": {},
        "baseline_error_vs_truth": {},
        "candidate_error_vs_truth": {},
        "prediction_delta_candidate_minus_baseline": {},
        "summary_times_s": [
            (i + 1) * DT_S for i in range(horizon)
            if (i + 1) % REPORT_INTERVAL_STEPS == 0],
        "full_curve_interval_s": DT_S,
    }
    for name, columns in STATE_COLUMNS.items():
        output["first_prediction_change_candidate_vs_baseline"][name] = (
            _first_change(delta[..., columns], PREDICTION_DIFF_TOLERANCE))
        output["first_change_by_run"][name] = _first_change_by_run(
            delta[..., columns], run_indices, run_ids,
            PREDICTION_DIFF_TOLERANCE)
        baseline_curve = _error_curve(
            predicted0, truth, run_indices, run_ids, columns)
        candidate_curve = _error_curve(
            predicted1, truth, run_indices, run_ids, columns)
        output["baseline_error_vs_truth"][name] = baseline_curve
        output["candidate_error_vs_truth"][name] = candidate_curve
        output["prediction_delta_candidate_minus_baseline"][name] = {
            "macro_run_rmse_by_step": _error_curve(
                predicted1, predicted0, run_indices, run_ids, columns
            )["macro_run_rmse_by_step"],
            "max_abs_delta_by_step": np.max(
                np.abs(delta[..., columns]), axis=(0, 2)).tolist(),
        }
    output["pose_error_vs_truth"] = {
        "baseline": _pose_error_curve(
            baseline_trace, run_indices, run_ids),
        "candidate": _pose_error_curve(
            candidate_trace, run_indices, run_ids),
        "candidate_minus_baseline_pose_first_change": {
            "position": _first_change(
                np.asarray(candidate_trace["predicted_pose"])[..., :2]
                - np.asarray(baseline_trace["predicted_pose"])[..., :2],
                PREDICTION_DIFF_TOLERANCE),
            "yaw": _first_change(_wrap(
                np.asarray(candidate_trace["predicted_pose"])[..., 2]
                - np.asarray(baseline_trace["predicted_pose"])[..., 2]),
                PREDICTION_DIFF_TOLERANCE),
        },
    }
    raw0 = np.asarray(
        baseline_trace["predicted_raw_encoder_wheels"], dtype=np.float64)
    raw1 = np.asarray(
        candidate_trace["predicted_raw_encoder_wheels"], dtype=np.float64)
    output["raw_encoder_wheel_error_vs_truth"] = {
        "semantics": "raw encoder sidecar, not simulator wheel-rotation ground truth",
        "baseline": _raw_wheel_error_curve(
            baseline_trace, run_indices, run_ids),
        "candidate": _raw_wheel_error_curve(
            candidate_trace, run_indices, run_ids),
        "prediction_delta_candidate_minus_baseline": {
            "first_prediction_change_by_run": _first_change_by_run(
                raw1 - raw0, run_indices, run_ids,
                PREDICTION_DIFF_TOLERANCE),
            "macro_run_rmse_by_step": _error_curve(
                raw1, raw0, run_indices, run_ids, (0, 1)
            )["macro_run_rmse_by_step"],
        },
    }
    output["first_nonzero_error_metric_delta"] = {}
    for name in STATE_COLUMNS:
        base_curve = output["baseline_error_vs_truth"][name][
            "macro_run_rmse_by_step"]
        cand_curve = output["candidate_error_vs_truth"][name][
            "macro_run_rmse_by_step"]
        difference = np.asarray(cand_curve) - np.asarray(base_curve)
        changed = np.flatnonzero(np.abs(difference) > 1e-9)
        output["first_nonzero_error_metric_delta"][name] = (
            None if not len(changed) else {
                "step": int(changed[0]) + 1,
                "time_s": (int(changed[0]) + 1) * DT_S,
                "candidate_minus_baseline_macro_rmse": float(
                    difference[changed[0]])})
    return output


def run(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    report_path = output_dir / "evaluation.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    prior_report = FIT_DIR / "evaluation.json"
    model_path = FIT_DIR / "full_regime_yaw_residual.joblib"
    for path in (prior_report, model_path, evaluator.HGB_BUNDLE,
                 evaluator.EDSSM_CHECKPOINT, evaluator.YAW_ATLAS_BUNDLE,
                 evaluator.HGB_TRAIN_DATASET, evaluator.HIGHSTEER_DATASET,
                 evaluator.PRACTICE_DATASET):
        if not path.is_file():
            raise FileNotFoundError(path)
    fit_report = json.loads(prior_report.read_text(encoding="utf-8"))
    if (fit_report.get("test_or_final_test_used") is not False
            or fit_report.get("yaw_atlas_frozen") is not True):
        raise ValueError("prior candidate provenance does not match this diagnostic")
    hgb = joblib.load(evaluator.HGB_BUNDLE)
    atlas = joblib.load(evaluator.YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(evaluator.EDSSM_CHECKPOINT, "cpu")
    correction_bundle = joblib.load(model_path)
    if (correction_bundle.get("research_only") is not True
            or correction_bundle.get("production_or_runtime_integration") is not False):
        raise ValueError("refusing to diagnose a non-research yaw candidate")
    correction = correction_bundle["estimator"]
    sources = _validation_sources(hgb)
    results = {}
    for source in sources:
        expected = fit_report["heldout_results"][source["name"]]
        if (source["run_ids"] != expected["run_ids"]
                or len(source["windows"]) != expected["windows"]):
            raise ValueError(f"held-out selection changed for {source['name']}")
        print(f"{source['name']}: paired 40-Hz traces, "
              f"{len(source['windows'])} windows", flush=True)
        results[source["name"]] = _cohort_diagnostic(
            source, hgb, edssm, metadata, atlas, correction)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = (prior_report, model_path, evaluator.HGB_BUNDLE,
                evaluator.EDSSM_CHECKPOINT, evaluator.YAW_ATLAS_BUNDLE,
                evaluator.HGB_TRAIN_DATASET, evaluator.HIGHSTEER_DATASET,
                evaluator.PRACTICE_DATASET)
    report = {
        "schema_version": 1,
        "purpose": "locate first recursive divergence by state for paired frozen yaw-atlas baseline and full-regime yaw-residual rollouts",
        "simulator_was_run": False,
        "test_or_final_test_used": False,
        "test_or_final_test_selected_or_scored": False,
        "candidate_fit_or_selection_changed": False,
        "yaw_atlas_frozen": True,
        "all_non_yaw_heads_frozen": True,
        "same_validation_practice_windows_and_commands": True,
        "initial_history_uses_recorded_state": True,
        "future_model_input_is_causal": True,
        "prediction_difference_tolerance": PREDICTION_DIFF_TOLERANCE,
        "reported_curve_interval_s": REPORT_INTERVAL_STEPS * DT_S,
        "input_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in required},
        "cohorts": results,
        "limitations": [
            "the two practice captures are a small transfer diagnostic",
            "the trace records the composed state, not hidden simulator wheel torque/load/slip",
            "an earliest numerical prediction change is causal onset, not by itself proof of the physical error mechanism",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    report = run(args.output_dir)
    summary = {}
    for name, cohort in report["cohorts"].items():
        first_state_curve = next(iter(
            cohort["baseline_error_vs_truth"].values()))
        available_steps = len(first_state_curve["macro_run_rmse_by_step"])
        summary[name] = {
            "run_ids": cohort["run_ids"],
            "first_prediction_change_candidate_vs_baseline": cohort[
                "first_prediction_change_candidate_vs_baseline"],
            "errors_at_0p75s": {},
            "errors_at_2s": {},
            "errors_at_5s": {},
        }
        for label, step in (("errors_at_0p75s", 30),
                            ("errors_at_2s", 80), ("errors_at_5s", 200)):
            if step > available_steps:
                continue
            index = step - 1
            summary[name][label] = {
                "baseline": {key: value["macro_run_rmse_by_step"][index]
                             for key, value in cohort[
                                 "baseline_error_vs_truth"].items()},
                "candidate": {key: value["macro_run_rmse_by_step"][index]
                              for key, value in cohort[
                                  "candidate_error_vs_truth"].items()},
            }
            if step in (30, 80, 200):
                summary[name][label]["baseline"]["position_rmse_m"] = (
                    cohort["pose_error_vs_truth"]["baseline"][
                        "position_rmse_m"]["macro_run_rmse_by_step"][index])
                summary[name][label]["baseline"]["pose_yaw_rmse_rad"] = (
                    cohort["pose_error_vs_truth"]["baseline"][
                        "pose_yaw_rmse_rad"]["macro_run_rmse_by_step"][index])
                summary[name][label]["candidate"]["position_rmse_m"] = (
                    cohort["pose_error_vs_truth"]["candidate"][
                        "position_rmse_m"]["macro_run_rmse_by_step"][index])
                summary[name][label]["candidate"]["pose_yaw_rmse_rad"] = (
                    cohort["pose_error_vs_truth"]["candidate"][
                        "pose_yaw_rmse_rad"]["macro_run_rmse_by_step"][index])
    print(json.dumps({"output": str(args.output_dir / "evaluation.json"), "summary": summary},
                     indent=2, sort_keys=True))
