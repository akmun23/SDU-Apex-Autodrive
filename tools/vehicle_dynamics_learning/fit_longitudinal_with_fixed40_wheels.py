#!/usr/bin/env python3
"""Fit a longitudinal transition teacher with causal fixed-40-Hz wheel inputs.

This asks whether the observed encoder-rate history improves next-step body-u,
effective-a_x, and filtered-wheel predictions beyond the existing state,
actuator, command, and wheel/body mismatch features. Only OpenPlane whole-run
training captures fit models; validation selects features; practice is transfer
only. The candidate is not a production observer or simulator.
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

from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DEFAULT_DATASET as DEFAULT_OLD_DATASET,
    DT_S,
    LAGS,
    _balanced_training_rows,
    _collect,
    _fit_variant,
    _metrics_by_run,
    _predict,
    _run_bootstrap_ci,
    _score_bundle,
    _select_score,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT as DEFAULT_WHEEL_OUTPUT,
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_recursive_candidate_20261009_longhistory_v5/"
    "selected_longitudinal_candidate.joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009")
BASE_VARIANT_NAMES = ("current_state", "history_to_1600ms",
                      "history_to_1600ms_plus_mismatch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_same_parent(fixed: dict[str, Any], old: dict[str, Any]) -> None:
    keys = ("frames", "simulator_rigid_state", "simulator_pose_xyyaw",
            "simulator_linear_acceleration", "dt_s", "bounds", "seq_run",
            "run_ids", "splits")
    for key in keys:
        if not np.array_equal(np.asarray(fixed[key]), np.asarray(old[key])):
            raise ValueError(f"fixed40 and reference longitudinal sources differ: {key}")
    if fixed["encoder_raw_source_dataset_sha256"] != old["encoder_raw_source_dataset_sha256"]:
        raise ValueError("fixed40 and reference encoder views have different source captures")


def _augment(data: dict[str, Any], split: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...],
                        tuple[list[str], ...]]:
    source, run_index, target, baseline, model_target, parent_features, state, names = (
        _collect(data, split))
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    keep = valid[source].copy()
    raw_parts = [raw[source].astype(np.float32)]
    raw_names = ["raw_encoder_left_current", "raw_encoder_right_current"]
    for lag in LAGS:
        keep &= valid[source - lag]
        raw_parts.append((raw[source] - raw[source - lag]).astype(np.float32))
        raw_names.extend((f"raw_encoder_left_delta_{lag}steps",
                          f"raw_encoder_right_delta_{lag}steps"))
    features = tuple(np.column_stack((x, *raw_parts)).astype(np.float32)
                     for x in parent_features)
    keep &= np.isfinite(raw[source]).all(axis=1)
    for x in features:
        keep &= np.isfinite(x).all(axis=1)
    if not np.any(keep):
        raise ValueError(f"no raw-wheel-conditioned rows for split={split}")
    features = tuple(x[keep] for x in features)
    names = tuple(list(parent_names) + raw_names for parent_names in names)
    return (source[keep], run_index[keep], target[keep], baseline[keep],
            model_target[keep], state, raw[source[keep]].astype(np.float32),
            features, names)


def _compare(candidate: dict[str, Any], reference: dict[str, Any]
             ) -> dict[str, Any]:
    metrics = {
        "u_rmse_mps": "u_rmse_mps",
        "effective_ax_rmse_mps2": "ax_rmse_mps2",
        "filtered_rear_wheel_pair_rmse_mps": "rear_wheel_pair_rmse_mps",
    }
    result: dict[str, Any] = {}
    for label, metric in metrics.items():
        common_runs = sorted(set(candidate["per_run"]) & set(reference["per_run"]))
        result[label] = {
            "candidate_macro_run_rmse": candidate["macro_run_metrics"][metric],
            "reference_macro_run_rmse": reference["macro_run_metrics"][metric],
            "paired_candidate_minus_reference_run_bootstrap_95pct": _run_bootstrap_ci(
                {run: {"x": candidate["per_run"][run][metric]}
                 for run in common_runs},
                {run: {"x": reference["per_run"][run][metric]}
                 for run in common_runs}, "x"),
        }
    return result


def run(dataset_path: Path, practice_path: Path, reference_path: Path,
        raw_wheel_bundle_path: Path, output_dir: Path,
        cap_per_run: int, max_iter: int, threads: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite longitudinal candidate: {output_dir}")
    for path in (dataset_path, practice_path, reference_path, raw_wheel_bundle_path,
                 DEFAULT_OLD_DATASET):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    fixed, old_training = _load_dataset(dataset_path), _load_dataset(DEFAULT_OLD_DATASET)
    _assert_same_parent(fixed, old_training)
    reference = joblib.load(reference_path)
    if _sha256(DEFAULT_OLD_DATASET) != reference["training_dataset_sha256"]:
        raise ValueError("reference candidate does not match its declared training dataset")
    raw_bundle = joblib.load(raw_wheel_bundle_path)
    if _sha256(dataset_path) != raw_bundle["training_dataset_sha256"]:
        raise ValueError("raw wheel candidate does not match the fixed40 training dataset")

    train = _augment(fixed, "train")
    val = _augment(fixed, "validation")
    src_train, run_train, y_train, base_train, dy_train, state_train, raw_train, x_train, names = train
    src_val, run_val, y_val, base_val, _, state_val, raw_val, x_val, _ = val
    run_ids = np.asarray(fixed["run_ids"]).astype(str)
    rows, sample_weight, train_counts = _balanced_training_rows(
        run_train, run_ids, cap_per_run, seed=151)
    models_by_variant: dict[str, list[Any]] = {}
    validation_reports: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    with threadpool_limits(limits=threads):
        for index, name in enumerate(BASE_VARIANT_NAMES):
            models = _fit_variant(x_train[index], dy_train, rows, sample_weight,
                                  max_iter)
            report, prediction = _score_bundle(
                models, x_val[index], y_val, base_val, run_val, run_ids)
            models_by_variant[name] = models
            validation_reports[name] = report
            validation_predictions[name] = prediction
    selected = min(validation_reports,
                   key=lambda name: _select_score(validation_reports[name]))
    selected_index = BASE_VARIANT_NAMES.index(selected)

    frames_train = np.asarray(fixed["frames"], dtype=np.float32)
    rigid_train = np.asarray(fixed["simulator_rigid_state"], dtype=np.float32)
    state7 = np.column_stack((rigid_train[:, (7, 8, 12)], frames_train[:, 3:7]))
    aux_target = (state7[src_train + 1][:, (1, 2, 3, 4)]
                  - state7[src_train][:, (1, 2, 3, 4)])[np.isfinite(
                      state7[src_train][:, (1, 2, 3, 4)]).all(axis=1)]
    # The source rows passed to _collect are finite; keep the explicit shape
    # assertion so a future input filter cannot silently misalign these labels.
    if len(aux_target) != len(src_train):
        raise ValueError("auxiliary labels do not align to selected source rows")
    with threadpool_limits(limits=threads):
        aux_models = _fit_variant(x_train[selected_index], aux_target, rows,
                                  sample_weight, max_iter)

    # Existing model is scored on precisely the candidate's eligible rows.
    # The source filter removes a small number of encoder-invalid rows. Build
    # the exact parent feature matrix directly to avoid relying on row order.
    val_parent = _collect(fixed, "validation")
    parent_rows = np.searchsorted(val_parent[0], src_val)
    if not np.array_equal(val_parent[0][parent_rows], src_val):
        raise AssertionError("raw-valid validation sources lost parent alignment")
    reference_val = _score_bundle(
        reference["models"], val_parent[5][2][parent_rows], y_val, base_val,
        run_val, run_ids)[0]
    validation_comparison = _compare(
        _score_bundle(models_by_variant[selected], x_val[selected_index],
                      y_val, base_val, run_val, run_ids)[0], reference_val)

    practice = _load_dataset(practice_path)
    practice_split = ("unseen_practice" if "unseen_practice" in
                      set(np.asarray(practice["splits"]).astype(str)) else "validation")
    p_src, p_run, p_y, p_base, _, _, _, p_x, _ = _augment(practice, practice_split)
    practice_runs = np.asarray(practice["run_ids"]).astype(str)
    practice_report, practice_prediction = _score_bundle(
        models_by_variant[selected], p_x[selected_index], p_y, p_base,
        p_run, practice_runs)
    p_parent = _collect(practice, practice_split)
    p_parent_rows = np.searchsorted(p_parent[0], p_src)
    if not np.array_equal(p_parent[0][p_parent_rows], p_src):
        raise AssertionError("raw-valid practice sources lost parent alignment")
    reference_practice, _ = _score_bundle(
        reference["models"], p_parent[5][2][p_parent_rows], p_y, p_base,
        p_run, practice_runs)
    practice_comparison = _compare(practice_report, reference_practice)

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle = {
        "research_only": True,
        "production_integration_allowed": False,
        "recursive_accuracy_established": False,
        "selected_variant": selected,
        "selected_parent_feature_variant": selected,
        "models": models_by_variant[selected],
        "recursive_aux_models": aux_models,
        "raw_wheel_models": raw_bundle["models"],
        "raw_wheel_feature_variant": raw_bundle["feature_variant"],
        "raw_wheel_history_lags_steps": raw_bundle["history_lags_steps"],
        "uses_predicted_raw_encoder_features": True,
        "feature_names": names[selected_index],
        "history_lags_steps": list(LAGS),
        "fixed_dt_s": DT_S,
        "target_names": ["u_increment", "effective_ax", "filtered_left_increment",
                         "filtered_right_increment"],
        "recursive_aux_target_names": ["v_increment", "yaw_rate_increment",
                                        "steering_feedback_increment",
                                        "throttle_feedback_increment"],
        "training_dataset_sha256": _sha256(dataset_path),
        "training_run_ids": sorted(set(run_ids[run_train].tolist())),
    }
    joblib.dump(bundle, output_dir / "longitudinal_raw_wheel_candidate.joblib")
    report = {
        "schema_version": 1,
        "purpose": "test whether causal fixed40 encoder history improves longitudinal one-step transitions",
        "dataset": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": _sha256(dataset_path),
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "reference_candidate": str(reference_path.relative_to(ROOT)),
        "raw_wheel_candidate": str(raw_wheel_bundle_path.relative_to(ROOT)),
        "training_run_ids": bundle["training_run_ids"],
        "training_rows_by_run": train_counts,
        "eligible_rows": {"train": int(len(src_train)),
                           "validation": int(len(src_val)),
                           "practice": int(len(p_src))},
        "validation_feature_selection": {
            name: validation_reports[name]["macro_run_metrics"]
            for name in BASE_VARIANT_NAMES},
        "selected_feature_variant": selected,
        "validation_vs_reference_same_rows": validation_comparison,
        "practice_split_used": practice_split,
        "practice_vs_reference_same_rows": practice_comparison,
        "test_and_final_test_policy": "sealed; no fit, feature selection, or scoring",
        "free_running_accuracy_established": False,
        "no_simulator_or_runtime_changes": True,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--raw-wheel-bundle", type=Path, default=DEFAULT_WHEEL_OUTPUT / "fixed40_wheel_response_candidate.joblib")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cap-per-run", type=int, default=8000)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.reference.resolve(), args.raw_wheel_bundle.resolve(),
                 args.output_dir.resolve(), args.cap_per_run,
                 args.max_iter, args.threads)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected_feature_variant": report["selected_feature_variant"],
        "validation_vs_reference": report["validation_vs_reference_same_rows"],
        "practice_vs_reference": report["practice_vs_reference_same_rows"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
