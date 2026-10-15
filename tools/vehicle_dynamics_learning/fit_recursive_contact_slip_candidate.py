#!/usr/bin/env python3
"""Train a contact-relative wheel-spin head on predicted-state rollouts.

This narrowly targets the largest remaining wheel-state mismatch. Training
rollouts use the saved isolated-Dagger-u/contact-relative-slip candidate on
whole-run train captures only. The head predicts the correction from current
predicted contact-relative encoder speed to the next simulator-labelled
contact-relative encoder speed. No future sensor values enter rollout input.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _append_contact_slip_features,
    _bundle_lags,
    _features,
    _select_windows,
    _state,
    evaluate as evaluate_recursive,
)
from tools.vehicle_dynamics_learning.fit_contact_relative_wheel_slip_model import (
    _contact_speed,
    _one_step_report,
    _prepare,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT as RAW_OUTPUT,
    DEFAULT_PRACTICE,
    VARIANT_NAMES as RAW_VARIANT_NAMES,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DT_S,
    LAGS,
    MAX_SPEED_MPS,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
)
from tools.vehicle_dynamics_learning.fit_recursive_dagger_longitudinal_candidate import (
    _filter_rollout_windows,
    sha256,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYBRID = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "isolated_dagger_u_contact_slip_candidate_20261009.joblib")
DEFAULT_SLIP_PARENT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "contact_relative_wheel_slip_model_20261009/"
    "contact_relative_wheel_slip_candidate.joblib")
DEFAULT_RAW_BUNDLE = RAW_OUTPUT / "fixed40_wheel_response_candidate.joblib"
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "recursive_contact_slip_candidate_20261009")
HORIZON_STEPS = 200
WINDOWS_PER_RUN = 24
ROWS_PER_RUN_PER_SOURCE = 2000
MAX_ITER = 220
SEED = 20261009


def _rollout_training_rows(
        data: dict[str, Any], bundle: dict[str, Any]
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    lags = _bundle_lags(bundle)
    if tuple(lags) != tuple(LAGS):
        raise ValueError("hybrid rollout has an unexpected feature history")
    if bundle.get("uses_contact_slip_state_features"):
        raise ValueError("unexpected pre-appended contact-slip body features")
    frames = np.asarray(data["frames"], dtype=np.float64)
    state_truth = _state(data)
    raw_truth = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    windows = _select_windows(data, "train", HORIZON_STEPS,
                              WINDOWS_PER_RUN, max(lags))
    windows, cadence_filter = _filter_rollout_windows(
        data, windows, max(lags), HORIZON_STEPS)
    if not windows:
        raise ValueError("no fixed-cadence train rollouts available")
    starts = np.asarray([item["start"] for item in windows], dtype=np.int64)
    run_index = np.asarray([item["run"] for item in windows], dtype=np.int32)
    sequence = np.asarray([item["sequence"] for item in windows], dtype=np.int64)
    window_bounds = bounds[sequence]
    if (np.any(starts - max(lags) < window_bounds[:, 0])
            or np.any(starts + HORIZON_STEPS >= window_bounds[:, 1])
            or not np.array_equal(np.asarray(data["seq_run"])[sequence], run_index)):
        raise ValueError("train rollout crossed a run/sequence boundary")

    initial = starts[:, None] + np.arange(-max(lags), 1)[None, :]
    history = np.empty((len(windows), max(lags) + 1, 9), dtype=np.float32)
    history[:, :, :7] = state_truth[initial].astype(np.float32)
    history[:, :, 7:9] = frames[initial, 7:9].astype(np.float32)
    raw_history = raw_truth[initial].astype(np.float32).copy()
    current_speed = np.hypot(history[:, -1, 0], history[:, -1, 1])
    active = current_speed <= MAX_SPEED_MPS
    terminated = np.zeros(len(windows), dtype=bool)
    max_speed = current_speed.copy()
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    row_runs: list[np.ndarray] = []
    models = bundle["models"]
    aux_models = bundle["recursive_aux_models"]
    slip_models = bundle["contact_slip_models"]
    half_track = float(bundle["contact_slip_rear_track_m"]) / 2.0

    for step in range(HORIZON_STEPS):
        source = starts + step
        x = _features(history, lags)
        raw_current = raw_history[:, -1, :]
        raw_parts = [raw_current]
        for lag in lags:
            raw_parts.append(raw_current - raw_history[:, -1 - lag, :])
        x = np.column_stack((x, *raw_parts)).astype(np.float32)
        contact_history = np.stack((
            history[:, :, 0] - half_track * history[:, :, 2],
            history[:, :, 0] + half_track * history[:, :, 2]), axis=2)
        slip_history = raw_history.astype(np.float64) - contact_history
        slip_current = slip_history[:, -1, :]
        slip_x = _append_contact_slip_features(
            x, history, raw_history, lags, half_track)

        slip_truth_next = raw_truth[source + 1].astype(np.float64) - np.stack((
            state_truth[source + 1, 0] - half_track * state_truth[source + 1, 2],
            state_truth[source + 1, 0] + half_track * state_truth[source + 1, 2]),
            axis=1)
        label_ok = (active & raw_valid[source + 1]
                    & np.isfinite(slip_x).all(axis=1)
                    & np.isfinite(slip_truth_next).all(axis=1)
                    & (np.hypot(state_truth[source + 1, 0],
                                state_truth[source + 1, 1]) <= MAX_SPEED_MPS))
        if np.any(label_ok):
            # Correct from the current predicted slip state directly to the
            # next simulator-labelled slip state. This is not the teacher-
            # forced GT slip increment used by the original model.
            feature_rows.append(slip_x[label_ok])
            target_rows.append((slip_truth_next[label_ok]
                                - slip_current[label_ok]).astype(np.float32))
            row_runs.append(run_index[label_ok])

        main_delta = np.column_stack([model.predict(x) for model in models])
        aux_delta = np.column_stack([model.predict(x) for model in aux_models])
        slip_delta = np.column_stack([
            model.predict(slip_x) for model in slip_models])
        following = history[:, -1, :7].astype(np.float64).copy()
        following[:, 0] += main_delta[:, 0]
        following[:, 1:5] += aux_delta
        following[:, 5:7] += main_delta[:, 2:4]
        slip_next = slip_current + slip_delta
        contact_next = np.column_stack((
            following[:, 0] - half_track * following[:, 2],
            following[:, 0] + half_track * following[:, 2]))
        raw_next = contact_next + slip_next
        if not np.isfinite(following).all() or not np.isfinite(raw_next).all():
            raise FloatingPointError("non-finite predicted training rollout")

        next_speed = np.hypot(following[:, 0], following[:, 1])
        max_speed = np.maximum(max_speed, next_speed)
        next_active = (active & (next_speed <= MAX_SPEED_MPS)
                       & (np.hypot(state_truth[source + 1, 0],
                                   state_truth[source + 1, 1]) <= MAX_SPEED_MPS))
        terminated |= active & ~next_active
        active = next_active
        next_commands = frames[source + 1, 7:9]
        next_history = np.column_stack((following, next_commands)).astype(np.float32)
        history = np.concatenate((history[:, 1:, :],
                                  next_history[:, None, :]), axis=1)
        raw_history = np.concatenate((raw_history[:, 1:, :],
                                      raw_next[:, None, :].astype(np.float32)), axis=1)

    if not feature_rows or not sum(len(values) for values in feature_rows):
        raise ValueError("no in-domain contact-slip rollout labels were generated")
    by_run = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        by_run[str(run_ids[run])] = {
            "windows": int(mask.sum()),
            "terminated_outside_0_12mps": int(terminated[mask].sum()),
            "max_predicted_speed_mps": float(max_speed[mask].max()),
        }
    return (np.vstack(feature_rows).astype(np.float32),
            np.vstack(target_rows).astype(np.float32),
            np.concatenate(row_runs).astype(np.int32),
            {"windows_used": len(windows), "per_run": by_run,
             **cadence_filter})


def _balanced_mix(
        teacher_x: np.ndarray, teacher_y: np.ndarray, teacher_run: np.ndarray,
        rollout_x: np.ndarray, rollout_y: np.ndarray, rollout_run: np.ndarray,
        run_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    selected_x, selected_y, counts = [], [], {}
    for run in sorted(set(teacher_run.tolist()) & set(rollout_run.tolist())):
        teacher_ids = np.flatnonzero(teacher_run == run)
        rollout_ids = np.flatnonzero(rollout_run == run)
        n_teacher = min(ROWS_PER_RUN_PER_SOURCE, len(teacher_ids))
        n_rollout = min(ROWS_PER_RUN_PER_SOURCE, len(rollout_ids))
        if not n_teacher or not n_rollout:
            continue
        teacher_pick = rng.choice(teacher_ids, size=n_teacher, replace=False)
        rollout_pick = rng.choice(rollout_ids, size=n_rollout, replace=False)
        selected_x.extend((teacher_x[teacher_pick], rollout_x[rollout_pick]))
        selected_y.extend((teacher_y[teacher_pick], rollout_y[rollout_pick]))
        counts[str(run_ids[run])] = {
            "teacher_forced_rows": int(n_teacher),
            "predicted_state_rollout_rows": int(n_rollout),
        }
    if not selected_x:
        raise ValueError("no source-run overlap between teacher and rollout data")
    return np.vstack(selected_x), np.vstack(selected_y), counts


def _paired_ci(candidate: dict[str, Any], parent: dict[str, Any],
               split: str, horizon: str, metric: str) -> dict[str, Any]:
    c = candidate["horizons"][split][horizon]
    p = parent["horizons"][split][horizon]
    if c["window_signature_sha256"] != p["window_signature_sha256"]:
        raise ValueError("parent and candidate rollout windows differ")
    c_runs, p_runs = c["per_run"], p["per_run"]
    common = sorted(set(c_runs) & set(p_runs))
    delta = np.asarray([
        c_runs[run][horizon]["candidate"][metric]
        - p_runs[run][horizon]["candidate"][metric] for run in common])
    rng = np.random.default_rng(SEED)
    draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    return {"runs": len(common), "mean_candidate_minus_parent": float(delta.mean()),
            "paired_run_bootstrap_95pct": [
                float(value) for value in np.quantile(draws, (.025, .975))]}


def fit(dataset_path: Path, practice_path: Path, raw_bundle_path: Path,
        slip_parent_path: Path, hybrid_path: Path, output_dir: Path,
        max_iter: int, threads: int) -> dict[str, Any]:
    started = time.time()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if not all(path.is_file() for path in (
            dataset_path, practice_path, raw_bundle_path,
            slip_parent_path, hybrid_path)):
        raise FileNotFoundError("one or more fixed data/model inputs are missing")
    parent_slip = joblib.load(slip_parent_path)
    hybrid = joblib.load(hybrid_path)
    data_hash = sha256(dataset_path)
    if any(bundle.get("training_dataset_sha256") != data_hash
           for bundle in (parent_slip, hybrid)):
        raise ValueError("input candidate bundle does not match fixed40 training data")
    raw_bundle = joblib.load(raw_bundle_path)
    if raw_bundle.get("training_dataset_sha256") != data_hash:
        raise ValueError("raw wheel bundle does not match fixed40 training data")
    raw_variant = str(raw_bundle.get("feature_variant"))
    if raw_variant not in RAW_VARIANT_NAMES:
        raise ValueError("raw wheel bundle has an unknown feature variant")
    expected_body_variant = BASE_VARIANT_NAMES[RAW_VARIANT_NAMES.index(raw_variant)]
    if hybrid.get("selected_parent_feature_variant") != expected_body_variant:
        raise ValueError("hybrid body and contact-slip feature variants differ")
    if hybrid.get("contact_slip_rear_track_m") != parent_slip.get("rear_track_m"):
        raise ValueError("hybrid and parent contact geometry differ")
    if (hybrid["contact_slip_models"][0].n_features_in_
            != parent_slip["models"][0].n_features_in_):
        raise ValueError("contact-relative model feature widths differ")

    train_data = _load_dataset(dataset_path)
    teacher = _prepare(train_data, "train", raw_bundle)
    rollout_x, rollout_y, rollout_run, rollout_stats = _rollout_training_rows(
        train_data, hybrid)
    mixed_x, mixed_y, mix_stats = _balanced_mix(
        teacher["feature"], teacher["slip_delta_target"], teacher["run_index"],
        rollout_x, rollout_y, rollout_run,
        np.asarray(train_data["run_ids"]).astype(str))
    with threadpool_limits(limits=threads):
        models = []
        for wheel in range(2):
            model = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.06, max_iter=max_iter,
                max_leaf_nodes=19, min_samples_leaf=40, l2_regularization=2.0,
                early_stopping=False, random_state=113 + wheel)
            model.fit(mixed_x, mixed_y[:, wheel])
            models.append(model)

    output_dir.mkdir(parents=True, exist_ok=False)
    candidate = dict(hybrid)
    candidate["contact_slip_models"] = models
    candidate["contact_slip_model_training"] = {
        "target": "GT next contact-relative encoder speed minus current predicted contact-relative encoder state",
        "training_sources": ["teacher_forced_train", "free_running_train_rollouts"],
        "rows_by_source_run": mix_stats,
        "rollout_eligibility": rollout_stats,
        "max_iter": max_iter,
    }
    candidate_path = output_dir / "recursive_contact_slip_candidate.joblib"
    joblib.dump(candidate, candidate_path)

    baseline_validation = _load_dataset(dataset_path)
    val_prepared = _prepare(baseline_validation, "validation", raw_bundle)
    baseline_one_step_val = _one_step_report(
        baseline_validation, val_prepared, parent_slip["models"],
        raw_bundle["models"])
    candidate_one_step_val = _one_step_report(
        baseline_validation, val_prepared, models, raw_bundle["models"])
    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str))
                      else "validation")
    practice_prepared = _prepare(practice_data, practice_split, raw_bundle)
    baseline_one_step_practice = _one_step_report(
        practice_data, practice_prepared, parent_slip["models"],
        raw_bundle["models"])
    candidate_one_step_practice = _one_step_report(
        practice_data, practice_prepared, models, raw_bundle["models"])

    baseline_report_path = output_dir / "hybrid_parent_recursive.json"
    candidate_report_path = output_dir / "recursive_candidate_evaluation.json"
    baseline_report = evaluate_recursive(
        hybrid_path, dataset_path, practice_path, baseline_report_path,
        24, None, "increment", True, "predicted_contact_slip", None)
    candidate_report = evaluate_recursive(
        candidate_path, dataset_path, practice_path, candidate_report_path,
        24, None, "increment", True, "predicted_contact_slip", None)
    key_metrics = ("u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
                   "position_rmse_m", "rear_wheel_pair_rmse_mps",
                   "raw_encoder_pair_rmse_mps")
    recursive_comparison = {
        split: {
            horizon: {
                metric: _paired_ci(candidate_report, baseline_report,
                                   split, horizon, metric)
                for metric in key_metrics
            }
            for horizon in ("1.000s", "2.000s", "3.000s", "5.000s")
        }
        for split in ("openplane_validation", "unseen_practice")
    }
    report = {
        "schema_version": 1,
        "purpose": "reduce wheel-spin state drift by training the contact-relative wheel response on predicted-state train rollouts",
        "training_dataset": str(dataset_path.relative_to(ROOT)),
        "training_dataset_sha256": data_hash,
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": sha256(practice_path),
        "raw_wheel_bundle": str(raw_bundle_path.relative_to(ROOT)),
        "parent_contact_slip_bundle": str(slip_parent_path.relative_to(ROOT)),
        "hybrid_rollout_bundle": str(hybrid_path.relative_to(ROOT)),
        "candidate_bundle": str(candidate_path.relative_to(ROOT)),
        "candidate_bundle_sha256": sha256(candidate_path),
        "teacher_forced_rows": int(len(teacher["feature"])),
        "predicted_state_rollout_rows": int(len(rollout_x)),
        "mixed_training_rows_by_run": mix_stats,
        "training_rollout_eligibility": rollout_stats,
        "test_and_final_test_used": False,
        "validation_and_practice_used_for_fitting_or_selection": False,
        "one_step_openplane_validation": {
            "parent": baseline_one_step_val["contact_relative_slip_model"],
            "candidate": candidate_one_step_val["contact_relative_slip_model"],
        },
        "one_step_unseen_practice_diagnostic": {
            "parent": baseline_one_step_practice["contact_relative_slip_model"],
            "candidate": candidate_one_step_practice["contact_relative_slip_model"],
        },
        "recursive_paired_candidate_minus_hybrid_parent": recursive_comparison,
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "contact-relative encoder residual is a wheel-spin proxy, not tire force or internal angular velocity",
            "rollouts initialize from simulator-truth context and use recorded commands; no sensor-only observer is tested",
            "practice-transfer evidence is limited to two captures",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--raw-bundle", type=Path, default=DEFAULT_RAW_BUNDLE)
    parser.add_argument("--slip-parent-bundle", type=Path, default=DEFAULT_SLIP_PARENT)
    parser.add_argument("--hybrid-bundle", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = fit(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.raw_bundle.resolve(), args.slip_parent_bundle.resolve(),
                 args.hybrid_bundle.resolve(), args.output_dir.resolve(),
                 args.max_iter, args.threads)
    print(json.dumps({
        "candidate": report["candidate_bundle"],
        "rollout_rows": report["predicted_state_rollout_rows"],
        "validation_5s_delta": report[
            "recursive_paired_candidate_minus_hybrid_parent"][
                "openplane_validation"]["5.000s"],
        "practice_5s_delta": report[
            "recursive_paired_candidate_minus_hybrid_parent"][
                "unseen_practice"]["5.000s"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
