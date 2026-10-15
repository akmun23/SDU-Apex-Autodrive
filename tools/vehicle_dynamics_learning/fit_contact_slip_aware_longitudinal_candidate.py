#!/usr/bin/env python3
"""Fit one-step body-transition heads with explicit rear wheel-spin state.

The feature addition is the signed fixed-40-Hz encoder-speed residual at each
rear wheel relative to rigid-body contact speed, plus its causal history. The
output targets remain the measured body/filtered-wheel transitions, not
corrections to future truth. Train captures fit the model; whole-run
validation and the existing practice captures are scored only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.fit_contact_relative_wheel_slip_model import (
    DEFAULT_OUTPUT_DIR as SLIP_OUTPUT_DIR,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
    VARIANT_NAMES as RAW_VARIANT_NAMES,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    REAR_TRACK_M,
    _balanced_training_rows,
    _fit_variant,
    _metrics_by_run,
    _run_bootstrap_ci,
    _score_bundle,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    DEFAULT_OUTPUT as LONGITUDINAL_PARENT_OUTPUT,
    _augment,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = LONGITUDINAL_PARENT_OUTPUT / "longitudinal_raw_wheel_candidate.joblib"
DEFAULT_SLIP = SLIP_OUTPUT_DIR / "contact_relative_wheel_slip_candidate.joblib"
DEFAULT_OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "contact_slip_aware_longitudinal_candidate_20261009")
MAX_ITER = 200
CAP_PER_RUN = 2000
SEED = 151


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare(data: dict[str, Any], split: str, parent: dict[str, Any]
             ) -> dict[str, Any]:
    source, run_index, target, baseline, model_target, _, raw_source, variants, names = (
        _augment(data, split))
    raw_variant = str(parent["raw_wheel_feature_variant"])
    if raw_variant not in RAW_VARIANT_NAMES:
        raise ValueError(f"unexpected parent feature family {raw_variant!r}")
    parent_name = BASE_VARIANT_NAMES[RAW_VARIANT_NAMES.index(raw_variant)]
    parent_index = BASE_VARIANT_NAMES.index(parent_name)
    parent_x = variants[parent_index]

    frames = np.asarray(data["frames"], dtype=np.float64)
    rigid = np.asarray(data["simulator_rigid_state"], dtype=np.float64)
    state = np.column_stack((rigid[:, (7, 8, 12)], frames[:, 3:7]))
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    half_track = REAR_TRACK_M / 2.0
    contact = np.column_stack((state[:, 0] - half_track * state[:, 2],
                               state[:, 0] + half_track * state[:, 2]))
    slip = raw - contact
    slip_parts = [slip[source].astype(np.float32)]
    slip_names = ["raw_left_minus_rigid_contact_speed",
                  "raw_right_minus_rigid_contact_speed"]
    for lag in parent["history_lags_steps"]:
        lag = int(lag)
        slip_parts.append((slip[source] - slip[source - lag]).astype(np.float32))
        slip_names.extend((f"delta_left_contact_slip_{lag}steps",
                           f"delta_right_contact_slip_{lag}steps"))
    feature = np.column_stack((parent_x, *slip_parts)).astype(np.float32)
    aux_state = state
    aux_baseline = aux_state[source, 1:5].astype(np.float32)
    aux_target = aux_state[source + 1, 1:5].astype(np.float32)
    aux_delta_target = aux_target - aux_baseline
    if not (len(feature) == len(source) == len(model_target) == len(aux_target)):
        raise ValueError("contact-slip features and targets lost row alignment")
    if not np.isfinite(feature).all() or not np.isfinite(aux_delta_target).all():
        raise ValueError("non-finite feature or target")
    return {
        "source": source,
        "run_index": run_index,
        "feature": feature,
        "parent_feature": parent_x,
        "target": target,
        "baseline": baseline,
        "main_delta_target": model_target,
        "aux_baseline": aux_baseline,
        "aux_target": aux_target,
        "aux_delta_target": aux_delta_target,
        "feature_names": list(names[parent_index]) + slip_names,
        "slip_feature_names": slip_names,
    }


def _auxiliary_report(prediction: np.ndarray, target: np.ndarray,
                      baseline: np.ndarray, run_index: np.ndarray,
                      run_ids: np.ndarray) -> dict[str, Any]:
    names = ("v_mps", "yaw_rate_rps", "steering_feedback_rad",
             "throttle_feedback_norm")
    per_run = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        per_run[str(run_ids[run])] = {
            name: {
                "candidate_rmse": float(np.sqrt(np.mean(
                    (prediction[mask, i] - target[mask, i]) ** 2))),
                "persistence_rmse": float(np.sqrt(np.mean(
                    (baseline[mask, i] - target[mask, i]) ** 2))),
            }
            for i, name in enumerate(names)
        }
    macro = {}
    for i, name in enumerate(names):
        candidate_run = {run: {"x": value[name]["candidate_rmse"]}
                         for run, value in per_run.items()}
        baseline_run = {run: {"x": value[name]["persistence_rmse"]}
                        for run, value in per_run.items()}
        macro[name] = {
            "candidate_macro_run_rmse": float(np.mean(
                [value[name]["candidate_rmse"] for value in per_run.values()])),
            "persistence_macro_run_rmse": float(np.mean(
                [value[name]["persistence_rmse"] for value in per_run.values()])),
            "candidate_minus_persistence_run_bootstrap_95pct": _run_bootstrap_ci(
                candidate_run, baseline_run, "x"),
        }
    return {"macro_run_metrics": macro, "per_run": per_run}


def _evaluate_split(data: dict[str, Any], prepared: dict[str, Any],
                    parent: dict[str, Any], main_models: list[Any],
                    aux_models: list[Any]) -> dict[str, Any]:
    run_ids = np.asarray(data["run_ids"]).astype(str)
    candidate_main, _ = _score_bundle(
        main_models, prepared["feature"], prepared["target"],
        prepared["baseline"], prepared["run_index"], run_ids)
    baseline_main, _ = _score_bundle(
        parent["models"], prepared["parent_feature"], prepared["target"],
        prepared["baseline"], prepared["run_index"], run_ids)
    prediction_aux = prepared["aux_baseline"] + np.column_stack(
        [model.predict(prepared["feature"]) for model in aux_models])
    parent_aux = prepared["aux_baseline"] + np.column_stack(
        [model.predict(prepared["parent_feature"])
         for model in parent["recursive_aux_models"]])
    score_aux = _auxiliary_report(
        prediction_aux, prepared["aux_target"], prepared["aux_baseline"],
        prepared["run_index"], run_ids)
    parent_score_aux = _auxiliary_report(
        parent_aux, prepared["aux_target"], prepared["aux_baseline"],
        prepared["run_index"], run_ids)
    main_comparison = {}
    for metric in ("u_rmse_mps", "ax_rmse_mps2", "rear_wheel_pair_rmse_mps"):
        main_comparison[metric] = {
            "candidate_macro_run_rmse": candidate_main["macro_run_metrics"][metric],
            "parent_macro_run_rmse": baseline_main["macro_run_metrics"][metric],
            "paired_candidate_minus_parent_run_bootstrap_95pct": _run_bootstrap_ci(
                {run: {"x": value[metric]}
                 for run, value in candidate_main["per_run"].items()},
                {run: {"x": value[metric]}
                 for run, value in baseline_main["per_run"].items()}, "x"),
        }
    return {
        "transitions": int(len(prepared["source"])),
        "main_transition_candidate": candidate_main,
        "main_transition_parent": baseline_main,
        "main_transition_comparison": main_comparison,
        "auxiliary_state_candidate": score_aux,
        "auxiliary_state_parent": parent_score_aux,
    }


def fit(dataset_path: Path, practice_path: Path, parent_path: Path,
        slip_path: Path, output_dir: Path, cap_per_run: int,
        max_iter: int, threads: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    for path in (dataset_path, practice_path, parent_path, slip_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = joblib.load(parent_path)
    slip_bundle = joblib.load(slip_path)
    dataset_hash = _sha256(dataset_path)
    if parent.get("training_dataset_sha256") != dataset_hash:
        raise ValueError("parent candidate does not match the training dataset")
    if slip_bundle.get("training_dataset_sha256") != dataset_hash:
        raise ValueError("contact-slip candidate does not match the training dataset")
    if not parent.get("research_only") or not slip_bundle.get("research_only"):
        raise ValueError("only frozen research bundles are accepted")

    started = time.time()
    train_data = _load_dataset(dataset_path)
    train = _prepare(train_data, "train", parent)
    run_ids = np.asarray(train_data["run_ids"]).astype(str)
    rows, weights, counts = _balanced_training_rows(
        train["run_index"], run_ids, cap_per_run, seed=SEED)
    with threadpool_limits(limits=threads):
        main_models = _fit_variant(train["feature"], train["main_delta_target"],
                                   rows, weights, max_iter)
        aux_models = _fit_variant(train["feature"], train["aux_delta_target"],
                                  rows, weights, max_iter)

    validation = _prepare(train_data, "validation", parent)
    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str))
                      else "validation")
    practice = _prepare(practice_data, practice_split, parent)
    validation_report = _evaluate_split(
        train_data, validation, parent, main_models, aux_models)
    practice_report = _evaluate_split(
        practice_data, practice, parent, main_models, aux_models)

    bundle = dict(parent)
    bundle.update({
        "models": main_models,
        "recursive_aux_models": aux_models,
        "contact_slip_models": slip_bundle["models"],
        "contact_slip_rear_track_m": REAR_TRACK_M,
        "uses_contact_slip_state_features": True,
        "contact_slip_feature_names": train["slip_feature_names"],
        "feature_names": train["feature_names"],
        "body_transition_training": {
            "split": "train only",
            "training_dataset_sha256": dataset_hash,
            "run_ids": sorted(set(run_ids[train["run_index"]].tolist())),
            "feature_count": int(train["feature"].shape[1]),
            "sample_counts_by_run": counts,
            "max_iter": max_iter,
            "seed": SEED,
        },
    })
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = output_dir / "contact_slip_aware_longitudinal_candidate.joblib"
    joblib.dump(bundle, bundle_path)
    report = {
        "schema_version": 1,
        "purpose": "test whether explicit wheel-spin residual history improves one-step body and filtered-wheel transitions",
        "training_dataset": str(dataset_path.relative_to(ROOT)),
        "training_dataset_sha256": dataset_hash,
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "parent_bundle": str(parent_path.relative_to(ROOT)),
        "parent_bundle_sha256": _sha256(parent_path),
        "contact_slip_bundle": str(slip_path.relative_to(ROOT)),
        "contact_slip_bundle_sha256": _sha256(slip_path),
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_bundle_sha256": _sha256(bundle_path),
        "training_run_ids": bundle["body_transition_training"]["run_ids"],
        "training_rows_by_run": counts,
        "test_and_final_test_used": False,
        "validation_used_for_fitting_or_selection": False,
        "practice_used_for_fitting_or_selection": False,
        "openplane_validation": validation_report,
        "unseen_practice_diagnostic": practice_report,
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "one-step teacher-forced-context results do not establish recursive vehicle accuracy",
            "contact-relative encoder speed is a wheel-spin proxy, not tire force or true tire slip",
            "the feature uses simulator-truth body state offline; production observer integration is not evaluated",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--parent-bundle", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--slip-bundle", type=Path, default=DEFAULT_SLIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cap-per-run", type=int, default=CAP_PER_RUN)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = fit(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.parent_bundle.resolve(), args.slip_bundle.resolve(),
                 args.output_dir.resolve(), args.cap_per_run,
                 args.max_iter, args.threads)
    print(json.dumps({
        "candidate": report["candidate_bundle"],
        "validation_main": report["openplane_validation"][
            "main_transition_comparison"],
        "practice_main": report["unseen_practice_diagnostic"][
            "main_transition_comparison"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
