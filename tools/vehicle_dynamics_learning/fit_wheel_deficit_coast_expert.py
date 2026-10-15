#!/usr/bin/env python3
"""Test a local predictor for low-throttle wheel-speed deficits.

This is motivated by whole-run residual stratification: negative filtered
wheel/contact-speed mismatch at low throttle is the worst held-out u/ax
region. The frozen global HGB remains the parent. Experts fit only on train
runs and are compared on broad validation; practice/high-steer are transfer
checks. The mismatch is a diagnostic proxy, not a tire-slip measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone

from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    REAR_TRACK_M,
    _rmse,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    DEFAULT_DATASET as TRAIN_DATASET,
    DEFAULT_PRACTICE as PRACTICE_DATASET,
    _augment,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
PARENT_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
HIGHSTEER_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/"
    "highsteer_75_heldout_validation_plus_r03_20261004/"
    "openplane_dynamics_fixed40hz.npz")
OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "wheel_deficit_coast_expert_20261009")
THRESHOLDS_M = (0.5, 1.0, 2.0)
LOW_THROTTLE_MAX = 0.25


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(data: dict[str, Any], split: str,
          parent: dict[str, Any]) -> dict[str, Any]:
    source, run_index, target, baseline, target_delta, _, _, feature_sets, _ = (
        _augment(data, split))
    variant = BASE_VARIANT_NAMES.index(parent["selected_parent_feature_variant"])
    x = feature_sets[variant]
    if x.shape[1] != parent["models"][0].n_features_in_:
        raise ValueError("expert feature contract differs from frozen parent")
    frames = np.asarray(data["frames"], dtype=np.float64)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    half_track = REAR_TRACK_M / 2.0
    contact_left = rigid[source, 7] - half_track * rigid[source, 12]
    contact_right = rigid[source, 7] + half_track * rigid[source, 12]
    signed_mismatch = 0.5 * (
        frames[source, 5] - contact_left
        + frames[source, 6] - contact_right)
    return {
        "source": source,
        "run_index": run_index,
        "run_ids": np.asarray(data["run_ids"]).astype(str),
        "x": x,
        "baseline": baseline,
        "target": target,
        "target_delta": target_delta,
        "signed_mismatch": signed_mismatch,
        "throttle": frames[source, 4],
    }


def _macro(error: np.ndarray, run_index: np.ndarray,
           run_ids: np.ndarray) -> dict[str, Any]:
    per_run = {}
    for run in np.unique(run_index):
        selected = run_index == run
        per_run[str(run_ids[run])] = {
            "rows": int(selected.sum()),
            "rmse": _rmse(error[selected]),
            "bias": float(np.mean(error[selected])),
        }
    return {
        "runs": len(per_run),
        "rows": len(error),
        "macro_run_rmse": float(np.mean(
            [row["rmse"] for row in per_run.values()])),
        "pooled_rmse": _rmse(error),
        "per_run": per_run,
    }


def _metrics(rows: dict[str, Any], parent_prediction: np.ndarray,
             candidate_prediction: np.ndarray, gate: np.ndarray
             ) -> dict[str, Any]:
    candidate = parent_prediction.copy()
    candidate[gate] = candidate_prediction[gate]
    parent_error = parent_prediction - rows["target"]
    candidate_error = candidate - rows["target"]
    result: dict[str, Any] = {
        "gate_fraction": float(np.mean(gate)),
        "gate_rows": int(gate.sum()),
        "all_rows": {},
        "gate_rows_only": {},
    }
    for name, columns in (("u_next_mps", (0,)),
                          ("effective_ax_mps2", (1,)),
                          ("filtered_wheel_pair_mps", (2, 3))):
        parent_error_sel = parent_error[:, columns]
        candidate_error_sel = candidate_error[:, columns]
        parent_gate = parent_error_sel[gate]
        candidate_gate = candidate_error_sel[gate]
        result["all_rows"][name] = {
            "parent": _macro(parent_error_sel, rows["run_index"], rows["run_ids"]),
            "candidate": _macro(candidate_error_sel, rows["run_index"], rows["run_ids"]),
        }
        selected_runs = rows["run_index"][gate]
        result["gate_rows_only"][name] = {
            "parent": _macro(parent_gate, selected_runs, rows["run_ids"]),
            "candidate": _macro(candidate_gate, selected_runs, rows["run_ids"]),
        }
    return result


def _fit(rows: dict[str, Any], parent: dict[str, Any],
         threshold: float) -> tuple[list[Any], dict[str, Any]]:
    gate = ((rows["signed_mismatch"] <= -threshold)
            & (rows["throttle"] < LOW_THROTTLE_MAX))
    active_runs = np.unique(rows["run_index"][gate])
    if int(gate.sum()) < 500 or len(active_runs) < 5:
        raise ValueError(
            f"insufficient train support at mismatch <= -{threshold:g}: "
            f"{int(gate.sum())} rows / {len(active_runs)} runs")
    run_counts = np.bincount(rows["run_index"][gate],
                             minlength=len(rows["run_ids"]))
    weight = 1.0 / run_counts[rows["run_index"][gate]]
    weight *= len(weight) / weight.sum()
    models = [
        clone(model).fit(rows["x"][gate], rows["target_delta"][gate, index],
                         sample_weight=weight)
        for index, model in enumerate(parent["models"])
    ]
    return models, {
        "threshold_mps": threshold,
        "throttle_feedback_below": LOW_THROTTLE_MAX,
        "rows": int(gate.sum()),
        "run_ids": sorted(set(rows["run_ids"][
            rows["run_index"][gate]].tolist())),
    }


def _evaluate(data: dict[str, Any], split: str, parent: dict[str, Any],
              models_by_threshold: dict[float, list[Any]]) -> dict[str, Any]:
    rows = _rows(data, split, parent)
    parent_delta = np.column_stack([
        model.predict(rows["x"]) for model in parent["models"]])
    parent_prediction = rows["baseline"] + parent_delta
    result: dict[str, Any] = {
        "split": split,
        "run_ids": sorted(set(rows["run_ids"][rows["run_index"]].tolist())),
        "rows": len(rows["source"]),
        "thresholds": {},
    }
    for threshold, models in models_by_threshold.items():
        delta = np.column_stack([
            model.predict(rows["x"]) for model in models])
        gate = ((rows["signed_mismatch"] <= -threshold)
                & (rows["throttle"] < LOW_THROTTLE_MAX))
        result["thresholds"][str(threshold)] = _metrics(
            rows, parent_prediction, rows["baseline"] + delta, gate)
    return result


def run() -> dict[str, Any]:
    report_path = OUTPUT_DIR / "evaluation.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    for path in (PARENT_BUNDLE, TRAIN_DATASET, PRACTICE_DATASET,
                 HIGHSTEER_DATASET):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = joblib.load(PARENT_BUNDLE)
    train_data = _load_dataset(TRAIN_DATASET)
    train = _rows(train_data, "train", parent)
    splits = np.asarray(train_data["splits"]).astype(str)
    if any(splits[run] != "train" for run in np.unique(train["run_index"])):
        raise ValueError("expert fit attempted to use a non-training run")
    models: dict[float, list[Any]] = {}
    support: dict[str, Any] = {}
    for threshold in THRESHOLDS_M:
        models[threshold], support[str(threshold)] = _fit(
            train, parent, threshold)

    results = {}
    for label, path, split in (
            ("broad_openplane_validation", TRAIN_DATASET, "validation"),
            ("highsteer_validation", HIGHSTEER_DATASET, "validation"),
            ("unseen_practice_diagnostic", PRACTICE_DATASET, "unseen_practice")):
        results[label] = _evaluate(_load_dataset(path), split, parent, models)

    report = {
        "schema_version": 1,
        "purpose": "test whether negative wheel/body mismatch at low throttle is a distinct, learnable longitudinal regime",
        "yaw_atlas": "frozen and not refit; isolated one-step longitudinal/wheel evaluation",
        "test_or_final_test_used": False,
        "simulator_was_run": False,
        "runtime_or_physics_changed": False,
        "fit_split": "whole training runs only",
        "validation_split": "broad OpenPlane whole-run validation",
        "practice_and_highsteer_are_transfer_only": True,
        "mismatch_definition": "mean signed filtered wheel speed minus mean kinematic rear contact speed; negative means encoder speed deficit, not proof of tire lock/slip",
        "parent_bundle_sha256": _sha256(PARENT_BUNDLE),
        "data_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (TRAIN_DATASET, PRACTICE_DATASET, HIGHSTEER_DATASET)},
        "training_run_ids": sorted(set(train["run_ids"][train["run_index"]].tolist())),
        "expert_fit_support": support,
        "results": results,
        "candidate_decision": {
            "retained": False,
            "reason": "held-out evidence must show improvement in deficit/low-throttle u, acceleration and wheel outputs, with no unacceptable all-domain or practice regression",
        },
        "limitations": [
            "one-step teacher-forced accuracy only; no recursive or lap-length claim",
            "the support gate uses current measured state for this diagnostic; an offline rollout must evaluate it on its own predicted state",
            "the two practice runs provide limited transfer uncertainty",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    report = run()
    summary = {}
    for cohort, result in report["results"].items():
        summary[cohort] = {}
        for threshold, metrics in result["thresholds"].items():
            summary[cohort][threshold] = {
                "gate_rows": metrics["gate_rows"],
                "gate_fraction": metrics["gate_fraction"],
                "u_parent_to_candidate": [
                    metrics["all_rows"]["u_next_mps"][key]["macro_run_rmse"]
                    for key in ("parent", "candidate")],
                "ax_parent_to_candidate": [
                    metrics["all_rows"]["effective_ax_mps2"][key]["macro_run_rmse"]
                    for key in ("parent", "candidate")],
                "wheel_parent_to_candidate": [
                    metrics["all_rows"]["filtered_wheel_pair_mps"][key]["macro_run_rmse"]
                    for key in ("parent", "candidate")],
                "gate_u_parent_to_candidate": [
                    metrics["gate_rows_only"]["u_next_mps"][key]["macro_run_rmse"]
                    for key in ("parent", "candidate")],
                "gate_ax_parent_to_candidate": [
                    metrics["gate_rows_only"]["effective_ax_mps2"][key]["macro_run_rmse"]
                    for key in ("parent", "candidate")],
                "gate_wheel_parent_to_candidate": [
                    metrics["gate_rows_only"]["filtered_wheel_pair_mps"][key]["macro_run_rmse"]
                    for key in ("parent", "candidate")],
            }
    print(json.dumps({"output": str(OUTPUT_DIR / "evaluation.json"),
                      "training_support": report["expert_fit_support"],
                      "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
