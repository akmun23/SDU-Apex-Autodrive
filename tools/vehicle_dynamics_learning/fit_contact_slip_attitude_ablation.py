#!/usr/bin/env python3
"""Test whether causal IMU attitude history explains wheel-spin response.

This is a one-step, offline diagnostic. It fits only whole-run training data
and evaluates the same transitions on held-out OpenPlane validation and the
separate practice captures. Current/past roll and pitch measurements are
inputs; no future attitude or wheel measurement is used. A positive result
would motivate an internally predicted attitude state before any free-running
plant use. It is not itself an offline vehicle simulator or runtime model.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.fit_contact_relative_wheel_slip_model import (
    _prepare,
    _run_score,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    _balanced_training_rows,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "fixed40_wheel_response_candidate_20261009/"
    "fixed40_wheel_response_candidate.joblib")
DEFAULT_PARENT_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "contact_relative_wheel_slip_model_20261009/"
    "contact_relative_wheel_slip_candidate.joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "contact_slip_attitude_ablation_20261009")
LAGS = (1, 4, 8, 16)
CAP_PER_RUN = 8000
MAX_ITER = 220
SEED = 20261009


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _attitude_features(data: dict[str, Any], prepared: dict[str, np.ndarray]
                       ) -> tuple[np.ndarray, np.ndarray]:
    attitude = np.asarray(data["imu_attitude_frames"], dtype=np.float32)
    valid = np.asarray(data["imu_attitude_valid"], dtype=bool)
    source = prepared["source"]
    indices = np.column_stack([source, *(source - lag for lag in LAGS)])
    row_valid = valid[indices].all(axis=1)
    current = attitude[source]
    parts = [current]
    for lag in LAGS:
        # Changes are causal differences from the current packet to a past
        # packet; they expose onset/unwind without using future attitude.
        parts.append(current - attitude[source - lag])
    return np.column_stack(parts).astype(np.float32), row_valid


def _score_model(data: dict[str, Any], prepared: dict[str, np.ndarray],
                 feature: np.ndarray, models: list[Any]
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    slip_next = prepared["slip_current"] + np.column_stack(
        [model.predict(feature) for model in models])
    prediction = prepared["contact_next"] + slip_next
    run_ids = np.asarray(data["run_ids"]).astype(str)
    return prediction, _run_score(
        prediction, prepared["raw_target"], prepared["run_index"], run_ids)


def _paired_ci(candidate: dict[str, Any], parent: dict[str, Any]
               ) -> dict[str, Any]:
    common = sorted(set(candidate["per_run"]) & set(parent["per_run"]))
    deltas = np.asarray([
        candidate["per_run"][run]["pair_rmse_mps"]
        - parent["per_run"][run]["pair_rmse_mps"] for run in common])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    return {
        "independent_runs": len(common),
        "mean_candidate_minus_parent_mps": float(deltas.mean()),
        "paired_run_bootstrap_95pct_mps": [
            float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "candidate_better_runs": int(np.sum(deltas < 0.0)),
        "per_run_delta_mps": dict(zip(common, deltas.astype(float).tolist())),
    }


def fit(dataset_path: Path, practice_path: Path, raw_bundle_path: Path,
        parent_path: Path, output_dir: Path, max_iter: int,
        cap_per_run: int, threads: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    for path in (dataset_path, practice_path, raw_bundle_path, parent_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    data_hash = _sha256(dataset_path)
    raw_bundle = joblib.load(raw_bundle_path)
    parent = joblib.load(parent_path)
    if raw_bundle.get("training_dataset_sha256") != data_hash:
        raise ValueError("raw wheel model does not match the fixed-40-Hz dataset")
    if parent.get("training_dataset_sha256") != data_hash:
        raise ValueError("parent contact model does not match the dataset")

    started = time.time()
    train_data = _load_dataset(dataset_path)
    train = _prepare(train_data, "train", raw_bundle)
    train_att, train_valid = _attitude_features(train_data, train)
    train_keep = train_valid & np.isfinite(train_att).all(axis=1)
    train_feature = np.column_stack((train["feature"], train_att)).astype(np.float32)
    train_run = train["run_index"][train_keep]
    train_ids = np.asarray(train_data["run_ids"]).astype(str)
    selected, weights, row_counts = _balanced_training_rows(
        train_run, train_ids, cap_per_run, SEED)
    target = train["slip_delta_target"][train_keep]

    models = []
    with threadpool_limits(limits=threads):
        for wheel in range(2):
            model = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.06,
                max_iter=max_iter, max_leaf_nodes=19,
                min_samples_leaf=40, l2_regularization=2.0,
                early_stopping=False, random_state=101)
            model.fit(train_feature[train_keep][selected], target[selected, wheel],
                      sample_weight=weights)
            models.append(model)

    validation_data = _load_dataset(dataset_path)
    validation = _prepare(validation_data, "validation", raw_bundle)
    val_att, val_valid = _attitude_features(validation_data, validation)
    val_keep = val_valid & np.isfinite(val_att).all(axis=1)
    validation = {key: (value[val_keep] if isinstance(value, np.ndarray)
                        and value.ndim > 0 and len(value) == len(val_keep)
                        else value)
                  for key, value in validation.items()}
    val_feature = np.column_stack((validation["feature"], val_att[val_keep]))
    _, val_score = _score_model(
        validation_data, validation, val_feature, models)
    _, val_parent_score = _score_model(
        validation_data, validation, validation["feature"], parent["models"])

    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str))
                      else "validation")
    practice = _prepare(practice_data, practice_split, raw_bundle)
    practice_att, practice_valid = _attitude_features(practice_data, practice)
    practice_keep = practice_valid & np.isfinite(practice_att).all(axis=1)
    practice = {key: (value[practice_keep] if isinstance(value, np.ndarray)
                      and value.ndim > 0 and len(value) == len(practice_keep)
                      else value)
                for key, value in practice.items()}
    practice_feature = np.column_stack((practice["feature"],
                                        practice_att[practice_keep]))
    _, practice_score = _score_model(
        practice_data, practice, practice_feature, models)
    _, practice_parent_score = _score_model(
        practice_data, practice, practice["feature"], parent["models"])

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = output_dir / "contact_slip_attitude_candidate.joblib"
    bundle = {
        "research_only": True,
        "production_integration_allowed": False,
        "target": parent["target"],
        "feature_variant": "contact_slip_parent_plus_causal_imu_attitude_history",
        "attitude_channels": list(train_data["imu_attitude_feature_names"]),
        "attitude_history_lags_steps": list(LAGS),
        "models": models,
        "training_dataset_sha256": data_hash,
        "training_run_ids": sorted(set(train_ids[train_run].tolist())),
        "max_iter": max_iter,
        "limitations": [
            "one-step label uses simulator-truth body contact speed",
            "uses measured current/past IMU attitude; future roll must be predicted internally for free-running use",
            "encoder residual is a contact-relative wheel-speed proxy, not tire force or true tire slip",
        ],
    }
    joblib.dump(bundle, candidate_path)
    report = {
        "schema_version": 1,
        "purpose": "causal one-step ablation of roll/pitch attitude history for rear-wheel contact-relative speed response",
        "candidate_bundle": str(candidate_path.relative_to(ROOT)),
        "candidate_bundle_sha256": _sha256(candidate_path),
        "parent_bundle": str(parent_path.relative_to(ROOT)),
        "training_dataset": str(dataset_path.relative_to(ROOT)),
        "training_dataset_sha256": data_hash,
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "training_rows_by_run": row_counts,
        "attitude_invalid_train_rows_removed": int((~train_keep).sum()),
        "openplane_validation": {
            "attitude_valid_transitions": int(val_keep.sum()),
            "candidate": val_score,
            "parent_on_same_attitude_valid_rows": val_parent_score,
            "paired_candidate_minus_parent": _paired_ci(val_score, val_parent_score),
        },
        "unseen_practice_diagnostic": {
            "attitude_valid_transitions": int(practice_keep.sum()),
            "candidate": practice_score,
            "parent_on_same_attitude_valid_rows": practice_parent_score,
            "paired_candidate_minus_parent": _paired_ci(
                practice_score, practice_parent_score),
        },
        "data_policy": {
            "fitted_on_whole_run_train_only": True,
            "validation_or_practice_used_for_fitting_or_selection": False,
            "test_or_final_test_used": False,
            "future_attitude_used_as_input": False,
            "recursive_rollout_evaluated": False,
        },
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--raw-bundle", type=Path, default=DEFAULT_RAW_BUNDLE)
    parser.add_argument("--parent-bundle", type=Path, default=DEFAULT_PARENT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--cap-per-run", type=int, default=CAP_PER_RUN)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = fit(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.raw_bundle.resolve(), args.parent_bundle.resolve(),
                 args.output_dir.resolve(), args.max_iter,
                 args.cap_per_run, args.threads)
    print(json.dumps({
        "output": report["candidate_bundle"],
        "validation_delta": report["openplane_validation"][
            "paired_candidate_minus_parent"],
        "practice_delta": report["unseen_practice_diagnostic"][
            "paired_candidate_minus_parent"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
