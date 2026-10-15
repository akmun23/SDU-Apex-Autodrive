#!/usr/bin/env python3
"""Fit a causal rear-wheel spin-residual model and test recursive transfer.

The predicted quantity is encoder-derived wheel surface speed relative to the
rigid-body longitudinal contact speed at each rear wheel. This is a measured
spin proxy, not a tire-force or tire-slip ground truth. Training uses only
whole-run OpenPlane train captures. Validation and practice are held out.
The recursive wheel-only diagnostic deliberately conditions on recorded body,
actuator and filtered-wheel state at every step to isolate wheel response; it
is not a free-running vehicle simulator.
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
    HORIZONS,
    _features,
    _select_windows,
    _state,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT as RAW_WHEEL_OUTPUT,
    DEFAULT_PRACTICE,
    VARIANT_NAMES as RAW_VARIANT_NAMES,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DT_S,
    LAGS,
    MAX_SPEED_MPS,
    REAR_TRACK_M,
    _balanced_training_rows,
    _rmse,
)
from tools.vehicle_dynamics_learning.fit_longitudinal_with_fixed40_wheels import (
    BASE_VARIANT_NAMES,
    _augment,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_BUNDLE = RAW_WHEEL_OUTPUT / "fixed40_wheel_response_candidate.joblib"
DEFAULT_OUTPUT_DIR = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "contact_relative_wheel_slip_model_20261009")
MAX_RECURSIVE_HORIZON = 200
MAX_WINDOWS_PER_RUN = 24
CAP_PER_RUN = 8000
SEED = 20261009


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contact_speed(data: dict[str, Any]) -> np.ndarray:
    state = _state(data)
    half_track = REAR_TRACK_M / 2.0
    # Left is +y and right is -y in the vehicle frame: v_x = u - r*y.
    return np.column_stack((state[:, 0] - half_track * state[:, 2],
                            state[:, 0] + half_track * state[:, 2]))


def _slip_feature(parent_x: np.ndarray, slip: np.ndarray,
                  source: np.ndarray) -> np.ndarray:
    parts = [slip[source].astype(np.float32)]
    for lag in LAGS:
        parts.append((slip[source] - slip[source - lag]).astype(np.float32))
    return np.column_stack((parent_x, *parts)).astype(np.float32)


def _prepare(data: dict[str, Any], split: str, raw_bundle: dict[str, Any]
             ) -> dict[str, np.ndarray]:
    if "encoder_raw_surface_mps" not in data or "encoder_raw_valid" not in data:
        raise ValueError("fixed-cadence raw encoder sidecar is required")
    source, run_index, _, _, _, _, _, parent_features, _ = _augment(data, split)
    variant = str(raw_bundle["feature_variant"])
    if variant not in RAW_VARIANT_NAMES:
        raise ValueError(f"unexpected raw-wheel feature variant: {variant}")
    parent_variant = BASE_VARIANT_NAMES[RAW_VARIANT_NAMES.index(variant)]
    parent_x = parent_features[BASE_VARIANT_NAMES.index(parent_variant)]
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    contact = _contact_speed(data)
    slip = raw - contact
    keep = (valid[source + 1]
            & np.isclose(dt[source + 1], DT_S, rtol=0.0, atol=1e-7)
            & np.isfinite(slip[source + 1]).all(axis=1))
    source = source[keep]
    run_index = run_index[keep]
    parent_x = parent_x[keep]
    return {
        "source": source,
        "run_index": run_index,
        "parent_x": parent_x,
        "feature": _slip_feature(parent_x, slip, source),
        "slip_delta_target": (slip[source + 1] - slip[source]).astype(np.float32),
        "slip_current": slip[source].astype(np.float32),
        "raw_current": raw[source].astype(np.float32),
        "raw_target": raw[source + 1].astype(np.float32),
        "contact_next": contact[source + 1].astype(np.float32),
    }


def _run_score(prediction: np.ndarray, target: np.ndarray,
               run_index: np.ndarray, run_ids: np.ndarray
               ) -> dict[str, Any]:
    per_run = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        error = prediction[mask] - target[mask]
        per_run[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            "left_rmse_mps": _rmse(error[:, 0]),
            "right_rmse_mps": _rmse(error[:, 1]),
            "pair_rmse_mps": _rmse(error),
            "pair_bias_mps": float(np.mean(error)),
        }
    return {
        "macro_run_pair_rmse_mps": float(np.mean(
            [item["pair_rmse_mps"] for item in per_run.values()])),
        "per_run": per_run,
    }


def _paired_run_ci(candidate: dict[str, Any], comparison: dict[str, Any]
                   ) -> dict[str, Any]:
    common = sorted(set(candidate["per_run"]) & set(comparison["per_run"]))
    difference = np.asarray([
        candidate["per_run"][run]["pair_rmse_mps"]
        - comparison["per_run"][run]["pair_rmse_mps"]
        for run in common], dtype=np.float64)
    if not len(difference):
        return {"runs": 0, "mean_candidate_minus_comparison": None,
                "paired_run_bootstrap_95pct": None}
    rng = np.random.default_rng(SEED)
    draws = rng.choice(difference, size=(10000, len(difference)), replace=True)
    means = draws.mean(axis=1)
    return {
        "runs": len(common),
        "mean_candidate_minus_comparison": float(difference.mean()),
        "paired_run_bootstrap_95pct": [
            float(value) for value in np.quantile(means, (0.025, 0.975))],
    }


def _one_step_report(data: dict[str, Any], prepared: dict[str, np.ndarray],
                     models: list[HistGradientBoostingRegressor],
                     raw_models: list[HistGradientBoostingRegressor]
                     ) -> dict[str, Any]:
    run_ids = np.asarray(data["run_ids"]).astype(str)
    slip_next = prepared["slip_current"] + np.column_stack(
        [model.predict(prepared["feature"]) for model in models])
    prediction = slip_next + prepared["contact_next"]
    raw_hgb = prepared["raw_current"] + np.column_stack(
        [model.predict(prepared["parent_x"]) for model in raw_models])
    persistence = prepared["raw_current"]
    slip_score = _run_score(prediction, prepared["raw_target"],
                            prepared["run_index"], run_ids)
    raw_score = _run_score(raw_hgb, prepared["raw_target"],
                           prepared["run_index"], run_ids)
    persistence_score = _run_score(persistence, prepared["raw_target"],
                                   prepared["run_index"], run_ids)

    source = prepared["source"]
    frames = np.asarray(data["frames"], dtype=np.float64)
    speed = np.hypot(_state(data)[source, 0], _state(data)[source, 1])
    steering = np.abs(frames[source, 7])
    throttle = frames[source, 8]
    abs_slip = np.abs(prepared["slip_current"]).max(axis=1)

    def slices(values: np.ndarray, edges: tuple[float, ...],
               labels: list[str]) -> dict[str, Any]:
        bin_index = np.digitize(values, edges[1:-1], right=False)
        output = {}
        for i, label in enumerate(labels):
            mask = bin_index == i
            if not np.any(mask):
                continue
            output[label] = {
                "transitions": int(mask.sum()),
                "contact_relative_slip_rmse_mps": _rmse(
                    prediction[mask] - prepared["raw_target"][mask]),
                "raw_hgb_rmse_mps": _rmse(raw_hgb[mask] - prepared["raw_target"][mask]),
                "persistence_rmse_mps": _rmse(
                    persistence[mask] - prepared["raw_target"][mask]),
            }
        return output

    return {
        "transitions": int(len(source)),
        "contact_relative_slip_model": slip_score,
        "raw_speed_hgb_comparator": raw_score,
        "raw_speed_persistence": persistence_score,
        "paired_run_bootstrap_vs_raw_speed_hgb": _paired_run_ci(slip_score, raw_score),
        "paired_run_bootstrap_vs_persistence": _paired_run_ci(slip_score, persistence_score),
        "descriptive_slices": {
            "speed_mps": slices(speed, (0, 2, 4, 6, 8, 10, 12.000001),
                                 ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12"]),
            "abs_steering_rad": slices(steering, (0, .1, .2, .3, .4, .5, np.inf),
                                        ["0-.1", ".1-.2", ".2-.3", ".3-.4", ".4-.5", ">=.5"]),
            "throttle_command": slices(throttle, (0, .05, .1, .2, .4, .6, .8, np.inf),
                                        ["0-.05", ".05-.10", ".10-.20", ".20-.40", ".40-.60", ".60-.80", ">=.80"]),
            "absolute_wheel_body_slip_proxy_mps": slices(
                abs_slip, (0, .05, .1, .25, .5, 1, np.inf),
                ["0-.05", ".05-.10", ".10-.25", ".25-.50", ".50-1", ">=1"]),
        },
    }


def _eligible_windows(data: dict[str, Any], split: str
                      ) -> tuple[list[dict[str, int]], dict[str, Any]]:
    raw_valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    speed = np.hypot(_state(data)[:, 0], _state(data)[:, 1])
    windows = _select_windows(
        data, split, MAX_RECURSIVE_HORIZON, MAX_WINDOWS_PER_RUN,
        initial_context_steps=max(LAGS))
    kept = []
    invalid_raw = 0
    bad_cadence = 0
    out_of_domain = 0
    for window in windows:
        start = window["start"]
        begin = start - max(LAGS)
        end = start + MAX_RECURSIVE_HORIZON
        if not raw_valid[begin:end + 1].all():
            invalid_raw += 1
            continue
        if np.any(speed[begin:end + 1] > MAX_SPEED_MPS):
            out_of_domain += 1
            continue
        if not np.isclose(dt[begin + 1:end + 1], DT_S,
                          rtol=0.0, atol=1e-7).all():
            bad_cadence += 1
            continue
        kept.append(window)
    return kept, {
        "eligible_windows": len(kept),
        "windows_skipped_invalid_raw_encoder": invalid_raw,
        "windows_skipped_outside_0_12mps": out_of_domain,
        "windows_skipped_bad_25ms_cadence": bad_cadence,
        "windows_by_run": {
            str(data["run_ids"][run]): sum(row["run"] == run for row in kept)
            for run in sorted({row["run"] for row in kept})},
    }


def _recursive_wheel_only(data: dict[str, Any], windows: list[dict[str, int]],
                          slip_models: list[HistGradientBoostingRegressor],
                          raw_models: list[HistGradientBoostingRegressor]
                          ) -> dict[str, Any]:
    """Recurse wheel states with recorded future chassis/actuator context."""
    lags = tuple(LAGS)
    frames = np.asarray(data["frames"], dtype=np.float64)
    state = _state(data)
    contact = _contact_speed(data)
    raw_truth = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    starts = np.asarray([row["start"] for row in windows], dtype=np.int64)
    run_index = np.asarray([row["run"] for row in windows], dtype=np.int32)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    initial_indices = starts[:, None] + np.arange(-max(lags), 1)[None, :]
    history = np.concatenate((state[initial_indices],
                              frames[initial_indices, 7:9]), axis=2)
    contact_history = contact[initial_indices].copy()
    truth_raw_history = raw_truth[initial_indices]
    raw_history_slip = truth_raw_history.copy()
    raw_history_model = truth_raw_history.copy()
    raw_prediction = {"contact_relative_slip_model": {}, "raw_speed_hgb": {}}
    for name in raw_prediction:
        raw_prediction[name] = np.empty((len(windows), MAX_RECURSIVE_HORIZON + 1, 2))
    raw_prediction["contact_relative_slip_model"][:, 0] = raw_history_slip[:, -1]
    raw_prediction["raw_speed_hgb"][:, 0] = raw_history_model[:, -1]

    for step in range(MAX_RECURSIVE_HORIZON):
        source = starts + step
        parent_x = _features(history, lags)

        slip_state_history = raw_history_slip - contact_history
        slip_current = slip_state_history[:, -1, :]
        raw_current = raw_history_slip[:, -1, :]
        raw_parts = [raw_current]
        for lag in lags:
            raw_parts.append(raw_current - raw_history_slip[:, -1 - lag, :])
        raw_parent_x = np.column_stack((parent_x, *raw_parts)).astype(np.float32)
        slip_parts = [slip_current]
        for lag in lags:
            slip_parts.append(slip_current - slip_state_history[:, -1 - lag, :])
        slip_x = np.column_stack((raw_parent_x, *slip_parts)).astype(np.float32)
        slip_delta = np.column_stack([model.predict(slip_x) for model in slip_models])
        slip_next = slip_current + slip_delta
        contact_next = contact[source + 1]
        raw_next_slip = slip_next + contact_next

        raw_current_model = raw_history_model[:, -1, :]
        raw_parts_model = [raw_current_model]
        for lag in lags:
            raw_parts_model.append(raw_current_model
                                   - raw_history_model[:, -1 - lag, :])
        raw_x = np.column_stack((parent_x, *raw_parts_model)).astype(np.float32)
        raw_delta = np.column_stack([model.predict(raw_x) for model in raw_models])
        raw_next_hgb = raw_current_model + raw_delta
        raw_prediction["contact_relative_slip_model"][:, step + 1] = raw_next_slip
        raw_prediction["raw_speed_hgb"][:, step + 1] = raw_next_hgb

        raw_history_slip = np.concatenate((raw_history_slip[:, 1:, :],
                                           raw_next_slip[:, None, :]), axis=1)
        raw_history_model = np.concatenate((raw_history_model[:, 1:, :],
                                            raw_next_hgb[:, None, :]), axis=1)
        contact_history = np.concatenate((contact_history[:, 1:, :],
                                          contact_next[:, None, :]), axis=1)
        next_body = np.column_stack((state[source + 1], frames[source + 1, 7:9]))
        history = np.concatenate((history[:, 1:, :], next_body[:, None, :]), axis=1)

    horizons = (1, 4, 10, 20, 40, 80, 120, 200)
    report = {"conditioning": (
        "recorded future body, actuator feedback, filtered wheel state and commands; "
        "only raw encoder wheel state is recursively predicted"), "horizons": {}}
    for horizon in horizons:
        target = raw_truth[starts + horizon]
        persistence = raw_truth[starts]
        per_model = {}
        for name, prediction in raw_prediction.items():
            per_model[name] = _run_score(prediction[:, horizon], target,
                                         run_index, run_ids)
        per_model["persistence"] = _run_score(
            persistence, target, run_index, run_ids)
        report["horizons"][f"{horizon * DT_S:.3f}s"] = {
            **per_model,
            "paired_run_bootstrap_vs_raw_speed_hgb": _paired_run_ci(
                per_model["contact_relative_slip_model"],
                per_model["raw_speed_hgb"]),
            "paired_run_bootstrap_vs_persistence": _paired_run_ci(
                per_model["contact_relative_slip_model"],
                per_model["persistence"]),
        }
    return report


def _fit_and_evaluate(dataset_path: Path, practice_path: Path,
                      raw_bundle_path: Path, output_dir: Path,
                      cap_per_run: int, max_iter: int, threads: int
                      ) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    for path in (dataset_path, practice_path, raw_bundle_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw_bundle = joblib.load(raw_bundle_path)
    dataset_hash = _sha256(dataset_path)
    if raw_bundle.get("training_dataset_sha256") != dataset_hash:
        raise ValueError("frozen raw-wheel candidate does not match this dataset")
    if not raw_bundle.get("research_only") or len(raw_bundle.get("models", ())) != 2:
        raise ValueError("expected the frozen two-head raw-wheel research candidate")

    started = time.time()
    training_data = _load_dataset(dataset_path)
    training = _prepare(training_data, "train", raw_bundle)
    run_ids = np.asarray(training_data["run_ids"]).astype(str)
    selected_rows, sample_weight, training_counts = _balanced_training_rows(
        training["run_index"], run_ids, cap_per_run, seed=SEED)
    with threadpool_limits(limits=threads):
        models = []
        for wheel in range(2):
            model = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.06, max_iter=max_iter,
                max_leaf_nodes=19, min_samples_leaf=40, l2_regularization=2.0,
                early_stopping=False, random_state=101)
            model.fit(training["feature"][selected_rows, :],
                      training["slip_delta_target"][selected_rows, wheel],
                      sample_weight=sample_weight)
            models.append(model)

    validation_data = _load_dataset(dataset_path)
    validation = _prepare(validation_data, "validation", raw_bundle)
    practice_data = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice_data["splits"]).astype(str))
                      else "validation")
    practice = _prepare(practice_data, practice_split, raw_bundle)
    validation_one_step = _one_step_report(
        validation_data, validation, models, raw_bundle["models"])
    practice_one_step = _one_step_report(
        practice_data, practice, models, raw_bundle["models"])
    validation_windows, validation_window_report = _eligible_windows(
        validation_data, "validation")
    practice_windows, practice_window_report = _eligible_windows(
        practice_data, practice_split)
    if not validation_windows or not practice_windows:
        raise ValueError("missing common raw-valid 5-second recursive windows")
    validation_recursive = _recursive_wheel_only(
        validation_data, validation_windows, models, raw_bundle["models"])
    practice_recursive = _recursive_wheel_only(
        practice_data, practice_windows, models, raw_bundle["models"])

    bundle = {
        "research_only": True,
        "production_integration_allowed": False,
        "target": "fixed40 encoder-derived rear wheel speed minus rigid-body contact speed",
        "target_is_tire_force_or_true_tire_slip": False,
        "feature_variant": "fixed40_parent_features_plus_contact_relative_slip_history",
        "history_lags_steps": list(LAGS),
        "rear_track_m": REAR_TRACK_M,
        "models": models,
        "training_dataset_sha256": dataset_hash,
        "training_run_ids": sorted(set(run_ids[training["run_index"]].tolist())),
        "max_iter": max_iter,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = output_dir / "contact_relative_wheel_slip_candidate.joblib"
    joblib.dump(bundle, bundle_path)
    report = {
        "schema_version": 1,
        "purpose": "test whether a contact-relative wheel-spin state predicts encoder rates more stably than absolute wheel speed",
        "training_dataset": str(dataset_path.relative_to(ROOT)),
        "training_dataset_sha256": dataset_hash,
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "raw_speed_comparator_bundle": str(raw_bundle_path.relative_to(ROOT)),
        "raw_speed_comparator_sha256": _sha256(raw_bundle_path),
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_bundle_sha256": _sha256(bundle_path),
        "training_split": "whole-run train only",
        "validation_split": "whole-run validation; no test/final-test scoring",
        "training_rows_by_run": training_counts,
        "training_transition_count": int(len(training["source"])),
        "one_step": {
            "openplane_validation": validation_one_step,
            "unseen_practice": practice_one_step,
        },
        "recursive_wheel_only": {
            "openplane_validation_window_eligibility": validation_window_report,
            "unseen_practice_window_eligibility": practice_window_report,
            "openplane_validation": validation_recursive,
            "unseen_practice": practice_recursive,
        },
        "data_policy": {
            "fitted_on_train_only": True,
            "validation_used_for_model_selection": False,
            "practice_used_for_fitting_or_selection": False,
            "test_or_final_test_used": False,
            "future_raw_encoder_truth_used_as_model_input": False,
            "future_body_truth_used_only_for_recursive_wheel_isolation": True,
        },
        "limitations": [
            "encoder-derived rate is a surface-speed proxy, not internal tire angular speed or tire force",
            "the rigid-body contact-speed subtraction assumes the rear track geometry and a planar body frame",
            "recursive wheel-only scores are conditioned on recorded future chassis/actuator state and do not validate a complete offline vehicle simulator",
            "unseen practice has only the existing independent captured runs",
        ],
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cap-per-run", type=int, default=CAP_PER_RUN)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = _fit_and_evaluate(
        args.dataset.resolve(), args.practice_dataset.resolve(),
        args.raw_bundle.resolve(), args.output_dir.resolve(),
        args.cap_per_run, args.max_iter, args.threads)
    print(json.dumps({
        "output": report["candidate_bundle"],
        "validation_one_step": report["one_step"]["openplane_validation"][
            "contact_relative_slip_model"]["macro_run_pair_rmse_mps"],
        "validation_one_step_raw_hgb": report["one_step"]["openplane_validation"][
            "raw_speed_hgb_comparator"]["macro_run_pair_rmse_mps"],
        "validation_recursive_5s": report["recursive_wheel_only"][
            "openplane_validation"]["horizons"]["5.000s"][
                "contact_relative_slip_model"]["macro_run_pair_rmse_mps"],
        "validation_recursive_5s_raw_hgb": report["recursive_wheel_only"][
            "openplane_validation"]["horizons"]["5.000s"][
                "raw_speed_hgb"]["macro_run_pair_rmse_mps"],
        "practice_recursive_5s": report["recursive_wheel_only"][
            "unseen_practice"]["horizons"]["5.000s"][
                "contact_relative_slip_model"]["macro_run_pair_rmse_mps"],
        "practice_recursive_5s_raw_hgb": report["recursive_wheel_only"][
            "unseen_practice"]["horizons"]["5.000s"][
                "raw_speed_hgb"]["macro_run_pair_rmse_mps"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
