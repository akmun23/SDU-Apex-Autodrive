#!/usr/bin/env python3
"""Fit and recursively evaluate a separate steering/throttle actuator model.

This is a train-only model of the measured actuator-feedback channels from
fixed-25-ms whole-run captures. Validation and existing practice runs are
held out. A separate recursive diagnostic asks whether actuator feedback
alone stays predictable under recorded commands; it is not a full plant.
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

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _bundle_lags,
    _features as _vehicle_features,
    _select_windows,
    _state,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DT_S,
    _balanced_training_rows,
    _rmse,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    DEFAULT_OUTPUT as PARENT_DIR,
    _augment,
)
from tools.vehicle_dynamics_learning.fit_recursive_dagger_longitudinal_candidate import (
    _filter_rollout_windows,
    sha256,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARENT = PARENT_DIR / "longitudinal_raw_wheel_candidate.joblib"
DEFAULT_OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "actuator_response_candidate_20261009")
LAGS = (1, 2, 4, 8, 16, 32, 64)
MAX_HORIZON = 200
MAX_WINDOWS_PER_RUN = 24
CAP_PER_RUN = 8000
SEED = 20261009
ACTUATOR_CHANNELS = (3, 4)
ACTUATOR_NAMES = ("steering_feedback_rad", "throttle_feedback_norm")


def _features(data: dict[str, Any], source: np.ndarray) -> np.ndarray:
    frames = np.asarray(data["frames"], dtype=np.float64)
    current = frames[source][:, (3, 4, 7, 8)]
    parts = [current]
    for lag in LAGS:
        parts.append(current - frames[source - lag][:, (3, 4, 7, 8)])
    parts.append(np.column_stack((current[:, 2] - current[:, 0],
                                  current[:, 3] - current[:, 1])))
    return np.column_stack(parts).astype(np.float32)


def _collect(data: dict[str, Any], split: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    sequence_runs = np.asarray(data["seq_run"], dtype=np.int32)
    splits = np.asarray(data["splits"]).astype(str)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    frames = np.asarray(data["frames"], dtype=np.float64)
    sources, runs = [], []
    for sequence, ((start_raw, end_raw), run_raw) in enumerate(
            zip(bounds, sequence_runs)):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] != split or end - start <= max(LAGS) + 1:
            continue
        source = np.arange(start + max(LAGS), end - 1, dtype=np.int64)
        valid = (np.isclose(dt[source + 1], DT_S, rtol=0.0, atol=1e-7)
                 & np.isfinite(frames[source][:, (3, 4, 7, 8)]).all(axis=1)
                 & np.isfinite(frames[source + 1][:, (3, 4)]).all(axis=1))
        source = source[valid]
        sources.append(source)
        runs.append(np.full(len(source), run, dtype=np.int32))
    if not sources:
        raise ValueError(f"no fixed-25-ms actuator transitions for {split}")
    source = np.concatenate(sources)
    run_index = np.concatenate(runs)
    feature = _features(data, source)
    target = (frames[source + 1][:, (3, 4)]
              - frames[source][:, (3, 4)]).astype(np.float32)
    if not np.isfinite(feature).all() or not np.isfinite(target).all():
        raise ValueError("non-finite actuator feature/target")
    return source, run_index, feature, target


def _score_runs(prediction: np.ndarray, truth: np.ndarray,
                run_index: np.ndarray, run_ids: np.ndarray
                ) -> dict[str, Any]:
    per_run = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        error = prediction[mask] - truth[mask]
        per_run[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            ACTUATOR_NAMES[0]: _rmse(error[:, 0]),
            ACTUATOR_NAMES[1]: _rmse(error[:, 1]),
        }
    macro = {
        name: float(np.mean([value[name] for value in per_run.values()]))
        for name in ACTUATOR_NAMES
    }
    return {"macro_run_rmse": macro, "per_run": per_run}


def _paired_ci(candidate: dict[str, Any], baseline: dict[str, Any],
               metric: str, seed: int) -> dict[str, Any]:
    common = sorted(set(candidate["per_run"]) & set(baseline["per_run"]))
    delta = np.asarray([candidate["per_run"][run][metric]
                        - baseline["per_run"][run][metric]
                        for run in common], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    return {"runs": len(common), "mean_candidate_minus_baseline": float(delta.mean()),
            "paired_run_bootstrap_95pct": [
                float(value) for value in np.quantile(draws, (.025, .975))]}


def _parent_features(data: dict[str, Any], split: str,
                     parent: dict[str, Any]
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, run_index, _, _, _, _, raw, variants, _ = _augment(data, split)
    variant = str(parent["selected_parent_feature_variant"])
    if variant not in BASE_VARIANT_NAMES:
        raise ValueError(f"unknown parent feature variant {variant}")
    # _augment returns the exact parent matrices, including the raw-wheel
    # history suffix; appending it again would silently change the schema.
    parent_x = variants[BASE_VARIANT_NAMES.index(variant)]
    if len(raw) != len(source):
        raise ValueError("parent features and raw encoder rows lost alignment")
    return source, run_index, parent_x


def _one_step_comparison(data: dict[str, Any], split: str,
                         candidate_models: list[Any],
                         parent: dict[str, Any]) -> dict[str, Any]:
    source, run_index, parent_x = _parent_features(data, split, parent)
    frames = np.asarray(data["frames"], dtype=np.float64)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    current = frames[source][:, (3, 4)]
    truth = frames[source + 1][:, (3, 4)]
    candidate_delta = np.column_stack(
        [model.predict(_features(data, source)) for model in candidate_models])
    candidate_prediction = current + candidate_delta
    parent_delta = np.column_stack((
        parent["recursive_aux_models"][2].predict(parent_x),
        parent["recursive_aux_models"][3].predict(parent_x)))
    parent_prediction = current + parent_delta
    persistence = current
    reports = {}
    for name, prediction in (("actuator_specific", candidate_prediction),
                             ("full_state_parent", parent_prediction),
                             ("persistence", persistence)):
        reports[name] = _score_runs(prediction, truth, run_index, run_ids)
    comparisons = {
        metric: _paired_ci(reports["actuator_specific"], reports["full_state_parent"],
                           metric, SEED + index)
        for index, metric in enumerate(ACTUATOR_NAMES)
    }
    return {"transitions": int(len(source)), "reports": reports,
            "paired_run_delta_vs_parent": comparisons}


def _recursive_evaluation(data: dict[str, Any], split: str,
                          candidate_models: list[Any],
                          parent: dict[str, Any]
                          ) -> dict[str, Any]:
    lags = _bundle_lags(parent)
    windows = _select_windows(data, split, MAX_HORIZON, MAX_WINDOWS_PER_RUN,
                              initial_context_steps=max(lags))
    windows, filters = _filter_rollout_windows(data, windows, max(lags), MAX_HORIZON)
    if not windows:
        raise ValueError(f"no eligible recursive actuator windows for {split}")
    frames = np.asarray(data["frames"], dtype=np.float64)
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    state_truth = _state(data)
    starts = np.asarray([row["start"] for row in windows], dtype=np.int64)
    run_index = np.asarray([row["run"] for row in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    indices = starts[:, None] + np.arange(-max(lags), 1)[None, :]
    actuator_history = frames[indices][:, :, (3, 4, 7, 8)].copy()
    parent_history = np.empty((len(windows), max(lags) + 1, 9), dtype=np.float32)
    parent_history[:, :, :7] = state_truth[indices].astype(np.float32)
    parent_history[:, :, 7:9] = frames[indices, 7:9].astype(np.float32)
    candidate_prediction = np.empty((len(windows), MAX_HORIZON + 1, 2))
    parent_prediction = np.empty_like(candidate_prediction)
    candidate_prediction[:, 0] = actuator_history[:, -1, :2]
    parent_prediction[:, 0] = actuator_history[:, -1, :2]
    for step in range(MAX_HORIZON):
        source = starts + step
        candidate_x = np.column_stack((
            actuator_history[:, -1, :], *[
                actuator_history[:, -1, :] - actuator_history[:, -1 - lag, :]
                for lag in lags],
            np.column_stack((actuator_history[:, -1, 2] - actuator_history[:, -1, 0],
                             actuator_history[:, -1, 3] - actuator_history[:, -1, 1]))
        )).astype(np.float32)
        candidate_delta = np.column_stack(
            [model.predict(candidate_x) for model in candidate_models])
        candidate_next = actuator_history[:, -1, :2].astype(np.float64) + candidate_delta

        parent_x = _vehicle_features(parent_history, lags)
        raw_current = raw[source].astype(np.float32)
        raw_parts = [raw_current]
        for lag in lags:
            raw_parts.append(raw_current - raw[source - lag].astype(np.float32))
        parent_x = np.column_stack((parent_x, *raw_parts)).astype(np.float32)
        parent_delta = np.column_stack((
            parent["recursive_aux_models"][2].predict(parent_x),
            parent["recursive_aux_models"][3].predict(parent_x)))
        parent_next = parent_history[:, -1, 3:5].astype(np.float64) + parent_delta
        candidate_prediction[:, step + 1] = candidate_next
        parent_prediction[:, step + 1] = parent_next

        next_act = frames[source + 1][:, (3, 4, 7, 8)].astype(np.float32).copy()
        next_act[:, :2] = candidate_next.astype(np.float32)
        actuator_history = np.concatenate((actuator_history[:, 1:, :],
                                            next_act[:, None, :]), axis=1)
        next_parent = np.column_stack((state_truth[source + 1],
                                       frames[source + 1, 7:9])).astype(np.float32)
        next_parent[:, 3:5] = parent_next.astype(np.float32)
        parent_history = np.concatenate((parent_history[:, 1:, :],
                                         next_parent[:, None, :]), axis=1)

    horizons = (1, 4, 20, 40, 80, 120, 200)
    result = {"window_count": len(windows), "window_filter": filters,
              "conditioning": (
                  "actuator-specific model uses only feedback/command history; "
                  "parent-head comparison receives recorded current body, "
                  "filtered-wheel and raw-encoder state each step"),
              "horizons": {}}
    for horizon in horizons:
        truth = frames[starts + horizon][:, (3, 4)]
        candidate = _score_runs(candidate_prediction[:, horizon], truth,
                                run_index, run_ids)
        parent_score = _score_runs(parent_prediction[:, horizon], truth,
                                   run_index, run_ids)
        persistence = _score_runs(frames[starts][:, (3, 4)], truth,
                                  run_index, run_ids)
        result["horizons"][f"{horizon * DT_S:.3f}s"] = {
            "actuator_specific": candidate,
            "parent_full_state_head": parent_score,
            "persistence": persistence,
            "paired_candidate_minus_parent": {
                metric: _paired_ci(candidate, parent_score, metric, SEED+horizon)
                for metric in ACTUATOR_NAMES},
        }
    return result


def fit(dataset_path: Path, practice_path: Path, parent_path: Path,
        output_dir: Path, cap_per_run: int, max_iter: int,
        threads: int) -> dict[str, Any]:
    started = time.time()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if not all(path.is_file() for path in (dataset_path, practice_path, parent_path)):
        raise FileNotFoundError("training, practice, or parent bundle is missing")
    parent = joblib.load(parent_path)
    if parent.get("training_dataset_sha256") != sha256(dataset_path):
        raise ValueError("parent bundle does not match the training dataset")
    train_data = _load_dataset(dataset_path)
    source, run_index, feature, target = _collect(train_data, "train")
    run_ids = np.asarray(train_data["run_ids"]).astype(str)
    rows, weights, row_counts = _balanced_training_rows(
        run_index, run_ids, cap_per_run, seed=SEED)
    with threadpool_limits(limits=threads):
        models = []
        for channel in range(2):
            model = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.06, max_iter=max_iter,
                max_leaf_nodes=19, min_samples_leaf=40, l2_regularization=2.0,
                early_stopping=False, random_state=101)
            model.fit(feature[rows], target[rows, channel], sample_weight=weights)
            models.append(model)

    validation_source = _one_step_comparison(
        train_data, "validation", models, parent)
    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str)) else "validation")
    practice_one_step = _one_step_comparison(
        practice_data, practice_split, models, parent)
    validation_recursive = _recursive_evaluation(
        train_data, "validation", models, parent)
    practice_recursive = _recursive_evaluation(
        practice_data, practice_split, models, parent)

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = output_dir / "actuator_response_candidate.joblib"
    bundle = {
        "research_only": True,
        "production_integration_allowed": False,
        "model": "separate causal HGB steering/throttle feedback transition",
        "target_names": list(ACTUATOR_NAMES),
        "command_names": ["steering_command_rad", "throttle_command_norm"],
        "history_lags_steps": list(LAGS),
        "fixed_dt_s": DT_S,
        "models": models,
        "feature_count": int(feature.shape[1]),
        "training_dataset_sha256": sha256(dataset_path),
        "training_run_ids": sorted(set(run_ids[run_index].tolist())),
        "training_rows_by_run": row_counts,
    }
    joblib.dump(bundle, bundle_path)
    report = {
        "schema_version": 1,
        "purpose": "test whether a dedicated actuator-state model reduces recursive longitudinal input error",
        "training_dataset": str(dataset_path.relative_to(ROOT)),
        "training_dataset_sha256": sha256(dataset_path),
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": sha256(practice_path),
        "parent_bundle": str(parent_path.relative_to(ROOT)),
        "parent_bundle_sha256": sha256(parent_path),
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_bundle_sha256": sha256(bundle_path),
        "training_run_ids": bundle["training_run_ids"],
        "training_rows_by_run": row_counts,
        "test_and_final_test_used": False,
        "validation_used_for_fitting_or_selection": False,
        "practice_used_for_fitting_or_selection": False,
        "one_step": {"openplane_validation": validation_source,
                     "unseen_practice": practice_one_step},
        "recursive_actuator_only": {
            "openplane_validation": validation_recursive,
            "unseen_practice": practice_recursive,
        },
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "actuator-only recursive performance does not establish body-motion or full-plant accuracy",
            "the parent comparison receives recorded body/raw-wheel context at each step while its actuator channels are predicted",
            "practice evidence consists of the existing two independent captures",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cap-per-run", type=int, default=CAP_PER_RUN)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = fit(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.parent_bundle.resolve(), args.output_dir.resolve(),
                 args.cap_per_run, args.max_iter, args.threads)
    print(json.dumps({
        "candidate": report["candidate_bundle"],
        "validation_one_step": report["one_step"]["openplane_validation"]["reports"],
        "practice_one_step": report["one_step"]["unseen_practice"]["reports"],
        "validation_recursive_5s": report["recursive_actuator_only"][
            "openplane_validation"]["horizons"]["5.000s"],
        "practice_recursive_5s": report["recursive_actuator_only"][
            "unseen_practice"]["horizons"]["5.000s"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
