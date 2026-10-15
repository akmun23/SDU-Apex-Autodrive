#!/usr/bin/env python3
"""Fit a distinct causal predictor for fixed-25-ms rear encoder rates.

The target is the next fixed-period encoder-derived surface-rate sample, not
instantaneous tire speed or the stored trailing-100-ms filtered channel. Only
whole-run OpenPlane train captures fit/select features; unseen practice runs
are scored once as transfer diagnostics. This remains a research artifact.
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

from tools.vehicle_dynamics_learning.fit_longitudinal_transition_candidate import (
    DEFAULT_PRACTICE,
    DT_S,
    LAGS,
    MAX_SPEED_MPS,
    _balanced_training_rows,
    _collect,
    _fit_variant,
    _rmse,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/"
    "encoder_fixed40hz_v1/openplane_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_PRACTICE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/full_modeling_reset_20261001/"
    "replacement_offline_sim_raceline_20261003/full_throttle_domain_v1/"
    "encoder_fixed40hz_v1/practice_dynamics_raw_wheels_fixed40hz.npz")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "fixed40_wheel_response_candidate_20261009")
VARIANT_NAMES = ("current_plus_raw", "history_plus_raw", "history_mismatch_plus_raw")
WHEEL_NAMES = ("left", "right")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare(data: dict[str, Any], split: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                        np.ndarray, np.ndarray, list[str]]:
    if "encoder_raw_surface_mps" not in data or "encoder_raw_valid" not in data:
        raise ValueError("dataset lacks the fixed-40-Hz encoder sidecar")
    source, run_index, _, _, _, parent_features, state, feature_names = _collect(
        data, split)
    raw = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    if raw.shape != (len(state) + 0, 2) or valid.shape != (len(state),):
        raise ValueError("fixed encoder arrays have unexpected shape")
    history_valid = valid[source] & valid[source + 1]
    raw_parts = [raw[source].astype(np.float32)]
    raw_names = ["raw_encoder_left_current", "raw_encoder_right_current"]
    for lag in LAGS:
        history_valid &= valid[source - lag]
        raw_parts.append((raw[source] - raw[source - lag]).astype(np.float32))
        raw_names.extend((f"raw_encoder_left_delta_{lag}steps",
                          f"raw_encoder_right_delta_{lag}steps"))
    x_variants = tuple(np.column_stack((base, *raw_parts)).astype(np.float32)
                       for base in parent_features)
    for x in x_variants:
        history_valid &= np.isfinite(x).all(axis=1)
    target = raw[source + 1]
    history_valid &= np.isfinite(target).all(axis=1)
    if not np.any(history_valid):
        raise ValueError(f"no valid fixed encoder rows for split={split!r}")
    feature_labels = [names + raw_names for names in feature_names]
    return (source[history_valid], run_index[history_valid],
            raw[source[history_valid]].astype(np.float32),
            target[history_valid].astype(np.float32),
            tuple(x[history_valid] for x in x_variants),
            state, feature_labels)


def _score(prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray,
           run_index: np.ndarray, run_ids: np.ndarray
           ) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for run in sorted(np.unique(run_index).tolist()):
        mask = run_index == run
        error, base_error = prediction[mask] - target[mask], baseline[mask] - target[mask]
        reports[str(run_ids[run])] = {
            "transitions": int(mask.sum()),
            "left_rmse_mps": _rmse(error[:, 0]),
            "right_rmse_mps": _rmse(error[:, 1]),
            "pair_rmse_mps": _rmse(error),
            "left_bias_mps": float(np.mean(error[:, 0])),
            "right_bias_mps": float(np.mean(error[:, 1])),
            "baseline_left_rmse_mps": _rmse(base_error[:, 0]),
            "baseline_right_rmse_mps": _rmse(base_error[:, 1]),
            "baseline_pair_rmse_mps": _rmse(base_error),
        }
    metric_names = ("left_rmse_mps", "right_rmse_mps", "pair_rmse_mps",
                    "baseline_left_rmse_mps", "baseline_right_rmse_mps",
                    "baseline_pair_rmse_mps")
    macro = {name: float(np.mean([row[name] for row in reports.values()]))
             for name in metric_names}
    macro["runs_better_than_persistence_pair"] = int(sum(
        row["pair_rmse_mps"] < row["baseline_pair_rmse_mps"]
        for row in reports.values()))
    baseline_reports = {
        key: {"pair_rmse_mps": row["baseline_pair_rmse_mps"]}
        for key, row in reports.items()}
    candidate_reports = {
        key: {"pair_rmse_mps": row["pair_rmse_mps"]}
        for key, row in reports.items()}
    differences = np.asarray([
        candidate_reports[key]["pair_rmse_mps"]
        - baseline_reports[key]["pair_rmse_mps"]
        for key in sorted(reports)], dtype=np.float64)
    rng = np.random.default_rng(20261009 + len(differences))
    draws = differences[rng.integers(0, len(differences),
                                     (10000, len(differences)))].mean(axis=1)
    macro["paired_candidate_minus_persistence_run_bootstrap_95pct"] = [
        float(value) for value in np.quantile(draws, (0.025, 0.975))]
    return {"macro_run_metrics": macro, "per_run": reports}


def _select_score(report: dict[str, Any]) -> float:
    return report["macro_run_metrics"]["pair_rmse_mps"]


def run(dataset_path: Path, practice_path: Path, output_dir: Path,
        cap_per_run: int, max_iter: int, threads: int) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite candidate output: {output_dir}")
    if not dataset_path.is_file() or not practice_path.is_file():
        raise FileNotFoundError("OpenPlane or practice fixed-40-Hz dataset missing")
    started = time.time()
    train_data, val_data = _load_dataset(dataset_path), _load_dataset(dataset_path)
    train = _prepare(train_data, "train")
    validation = _prepare(val_data, "validation")
    source_train, runs_train, base_train, target_train, x_train, _, names = train
    source_val, runs_val, base_val, target_val, x_val, _, _ = validation
    run_ids = np.asarray(train_data["run_ids"]).astype(str)
    rows, sample_weight, train_counts = _balanced_training_rows(
        runs_train, run_ids, cap_per_run, seed=141)
    models_by_variant: dict[str, list[HistGradientBoostingRegressor]] = {}
    reports: dict[str, Any] = {}
    with threadpool_limits(limits=threads):
        for index, name in enumerate(VARIANT_NAMES):
            models = _fit_variant(x_train[index], target_train - base_train,
                                  rows, sample_weight, max_iter)
            prediction = base_val + np.column_stack([
                model.predict(x_val[index]) for model in models])
            reports[name] = _score(prediction, target_val, base_val, runs_val,
                                   np.asarray(val_data["run_ids"]).astype(str))
            models_by_variant[name] = models
    selected = min(reports, key=lambda name: _select_score(reports[name]))

    practice = _load_dataset(practice_path)
    practice_split = "unseen_practice" if "unseen_practice" in set(
        np.asarray(practice["splits"]).astype(str)) else "validation"
    p_source, p_runs, p_base, p_target, p_x, _, _ = _prepare(practice, practice_split)
    selected_models = models_by_variant[selected]
    p_prediction = p_base + np.column_stack([
        model.predict(p_x[VARIANT_NAMES.index(selected)]) for model in selected_models])
    practice_report = _score(p_prediction, p_target, p_base, p_runs,
                             np.asarray(practice["run_ids"]).astype(str))

    output_dir.mkdir(parents=True, exist_ok=False)
    bundle = {
        "research_only": True,
        "production_integration_allowed": False,
        "target": "next fixed-25-ms encoder-derived left/right surface rates",
        "target_is_instantaneous_tire_speed": False,
        "target_is_trailing_100ms_filtered_rate": False,
        "feature_variant": selected,
        "history_lags_steps": list(LAGS),
        "models": selected_models,
        "feature_names": names[VARIANT_NAMES.index(selected)],
        "training_dataset_sha256": _sha256(dataset_path),
        "training_run_ids": sorted(set(run_ids[runs_train].tolist())),
    }
    joblib.dump(bundle, output_dir / "fixed40_wheel_response_candidate.joblib")
    report = {
        "schema_version": 1,
        "purpose": "predict fixed-cadence rear encoder-rate response as a distinct measured target",
        "dataset": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": _sha256(dataset_path),
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "training_run_ids": bundle["training_run_ids"],
        "training_rows_by_run": train_counts,
        "training_valid_transitions": int(len(source_train)),
        "validation_valid_transitions": int(len(source_val)),
        "validation_feature_comparison": reports,
        "selected_feature_variant": selected,
        "practice_split_used": practice_split,
        "practice_transfer": practice_report,
        "test_and_final_test_policy": "sealed; neither split is fitted, selected, or scored",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cap-per-run", type=int, default=8000)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run(args.dataset.resolve(), args.practice_dataset.resolve(),
                 args.output_dir.resolve(), args.cap_per_run,
                 args.max_iter, args.threads)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "selected_feature_variant": report["selected_feature_variant"],
        "validation": report["validation_feature_comparison"][
            report["selected_feature_variant"]]["macro_run_metrics"],
        "practice_transfer": report["practice_transfer"]["macro_run_metrics"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
