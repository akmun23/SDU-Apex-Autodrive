#!/usr/bin/env python3
"""Fit a scalar closed-loop yaw-increment gain around the frozen yaw atlas.

The gain is selected only on whole training runs, using recursive hybrid
rollouts. The frozen atlas remains unchanged. Validation, high-steering, and
practice runs are scored only after the gain is fixed. No simulator is run and
no future sensor or truth values enter any rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_hybrid_longitudinal_edssm_candidate import (
    DT_S,
    EDSSM_CHECKPOINT,
    HGB_BUNDLE,
    HGB_TRAIN_DATASET,
    HIGHSTEER_DATASET,
    PRACTICE_DATASET,
    YAW_ATLAS_BUNDLE,
    _bundle_lags,
    _rollout,
    _windows,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.racing.specialists import fit_sensor_only_yaw_regime_atlas as yaw_atlas


OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "recursive_yaw_delta_gain_sweep_20261009.json")
HORIZON = 200
GAINS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
WINDOWS_PER_RUN = 16
CONTEXT_STEPS = 80


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_windows(data: dict[str, Any], split: str, hgb_lags: tuple[int, ...],
                 maximum_per_run: int,
                 eligible_run_ids: set[str] | None = None,
                 horizon_steps: int = HORIZON) -> list[dict[str, int]]:
    """Select non-overlapping fixed-horizon windows from only one split."""
    bounds = np.asarray(data["bounds"], dtype=np.int64)
    seq_run = np.asarray(data["seq_run"], dtype=np.int32)
    splits = np.asarray(data["splits"]).astype(str)
    raw_valid_source = data.get("encoder_raw_valid")
    raw_valid = (None if raw_valid_source is None else
                 np.asarray(raw_valid_source, dtype=bool))
    sensor_valid = np.asarray(data["sensor_valid"], dtype=bool)
    attitude_valid = np.asarray(data["imu_attitude_valid"], dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    context = max(CONTEXT_STEPS, max(hgb_lags))
    atlas_context = max(yaw_atlas.HISTORY_LAGS)
    by_run: dict[int, list[dict[str, int]]] = {}
    for sequence, ((start_raw, end_raw), run_raw) in enumerate(
            zip(bounds, seq_run)):
        start, end, run = int(start_raw), int(end_raw), int(run_raw)
        if splits[run] != split or end - start < context + horizon_steps + 1:
            continue
        if (eligible_run_ids is not None
                and str(np.asarray(data["run_ids"]).astype(str)[run])
                not in eligible_run_ids):
            continue
        # Non-overlap prevents a long capture from dominating the run-macro
        # training objective; evenly selected windows still span the run.
        candidates = np.arange(start + context, end - horizon_steps,
                               horizon_steps, dtype=np.int64)
        for index in candidates:
            index = int(index)
            valid = (
                (raw_valid is None or
                 raw_valid[index - context:index + 1].all())
                and sensor_valid[index - atlas_context:index + 1].all()
                and attitude_valid[index - atlas_context:index + 1].all()
                and np.isclose(
                    dt[index - context + 1:index + horizon_steps + 1], DT_S,
                    rtol=0.0, atol=1e-7).all())
            if valid:
                by_run.setdefault(run, []).append({
                    "run": run, "sequence": sequence, "start": index})
    selected: list[dict[str, int]] = []
    for run, candidates in sorted(by_run.items()):
        if len(candidates) > maximum_per_run:
            indices = np.linspace(0, len(candidates) - 1,
                                  maximum_per_run, dtype=np.int64)
            candidates = [candidates[int(index)] for index in np.unique(indices)]
        selected.extend(candidates)
    if not selected:
        raise ValueError(f"no eligible {split!r} windows")
    if any(splits[row["run"]] != split for row in selected):
        raise AssertionError("split leakage while selecting rollout windows")
    return selected


def _yaw_objective(result: dict[str, Any]) -> float:
    horizons = result["macro_run_rmse"]
    return float(np.mean([
        float(horizons[key]["hybrid_atlas"]["yaw_rate_rmse_rps"])
        for key in ("0.750s", "2.000s", "5.000s")]))


def _score(data: dict[str, Any], hgb: dict[str, Any], edssm: Any,
           metadata: dict[str, Any], atlas: dict[str, Any],
           windows: list[dict[str, int]], gain: float) -> dict[str, Any]:
    return _rollout(
        data, hgb, edssm, metadata, atlas, windows, HORIZON,
        yaw_delta_gain=gain, include_oracle_diagnostics=False)


def run(output_path: Path = OUTPUT, dataset_path: Path = HGB_TRAIN_DATASET,
        highsteer_path: Path = HIGHSTEER_DATASET,
        practice_path: Path = PRACTICE_DATASET) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    for path in (HGB_BUNDLE, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
                 dataset_path, highsteer_path, practice_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hgb = joblib.load(HGB_BUNDLE)
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")
    train_data = _load_dataset(dataset_path)
    train_splits = np.asarray(train_data["splits"]).astype(str)
    run_ids = np.asarray(train_data["run_ids"]).astype(str)
    hgb_training = set(hgb["training_run_ids"])
    edssm_training = set(metadata["training_runs"])
    atlas_series, _ = yaw_atlas.teacher._discover_series_with_safe_mixed_archives()
    atlas_training = {series.run_id for series in atlas_series
                      if series.split == "train"}
    eligible_train_ids = (set(run_ids[train_splits == "train"])
                          & hgb_training & edssm_training)
    train_windows = _run_windows(
        train_data, "train", _bundle_lags(hgb), WINDOWS_PER_RUN,
        eligible_train_ids)
    train_ids = {str(run_ids[row["run"]]) for row in train_windows}
    if not train_ids <= hgb_training or not train_ids <= edssm_training:
        raise ValueError("rollout-fit runs overlap neither/both base-model training sets")

    train_scores: dict[str, Any] = {}
    for gain in GAINS:
        result = _score(train_data, hgb, edssm, metadata, atlas,
                        train_windows, gain)
        train_scores[f"{gain:.2f}"] = {
            "yaw_objective_mean_of_run_macro_rmse_rps": _yaw_objective(result),
            "macro_run_rmse": result["macro_run_rmse"],
            "windows_by_run": result["windows_by_run"],
        }
    selected_gain = min(
        GAINS,
        key=lambda gain: train_scores[f"{gain:.2f}"][
            "yaw_objective_mean_of_run_macro_rmse_rps"])

    validation_data = train_data
    validation_windows = _run_windows(
        validation_data, "validation", _bundle_lags(hgb), WINDOWS_PER_RUN)
    highsteer_data = _load_dataset(highsteer_path)
    highsteer_windows = _windows(
        highsteer_data, highsteer_path, HORIZON, _bundle_lags(hgb))
    practice_data = _load_dataset(practice_path)
    practice_windows = _windows(
        practice_data, practice_path, HORIZON, _bundle_lags(hgb))

    heldout: dict[str, Any] = {}
    for name, data, windows in (
            ("whole_run_validation", validation_data, validation_windows),
            ("highsteer_validation", highsteer_data, highsteer_windows),
            ("unseen_practice_diagnostic", practice_data, practice_windows)):
        baseline = _score(data, hgb, edssm, metadata, atlas, windows, 1.0)
        selected = (_score(data, hgb, edssm, metadata, atlas, windows,
                           selected_gain)
                    if selected_gain != 1.0 else baseline)
        heldout[name] = {
            "heldout_run_ids": sorted({
                str(np.asarray(data["run_ids"]).astype(str)[row["run"]])
                for row in windows}),
            "windows_by_run": selected["windows_by_run"],
            "baseline_atlas_gain_1": baseline["macro_run_rmse"],
            "selected_gain": selected["macro_run_rmse"],
            "per_run": selected["per_run"],
        }

    report = {
        "schema_version": 1,
        "purpose": "training-run-only scalar closure test for the frozen yaw atlas inside the causal longitudinal/lateral hybrid plant",
        "research_only": True,
        "production_or_runtime_integration": False,
        "simulator_was_run": False,
        "test_or_final_test_used_for_fit_or_selection": False,
        "frozen_yaw_atlas": str(YAW_ATLAS_BUNDLE.relative_to(ROOT)),
        "frozen_yaw_atlas_sha256": _sha256(YAW_ATLAS_BUNDLE),
        "hgb_bundle_sha256": _sha256(HGB_BUNDLE),
        "edssm_checkpoint_sha256": _sha256(EDSSM_CHECKPOINT),
        "dataset_sha256": _sha256(dataset_path),
        "highsteer_dataset_sha256": _sha256(highsteer_path),
        "practice_dataset_sha256": _sha256(practice_path),
        "fit_protocol": {
            "fit_split": "train only",
            "training_runs": sorted(train_ids),
            "training_run_count": len(train_ids),
            "training_runs_used_by_yaw_atlas": sorted(train_ids & atlas_training),
            "training_runs_not_used_by_yaw_atlas": sorted(train_ids - atlas_training),
            "training_window_count": len(train_windows),
            "maximum_windows_per_run": WINDOWS_PER_RUN,
            "window_horizon_s": HORIZON * DT_S,
            "windows_non_overlapping_within_sequence": True,
            "selection_objective": "equal mean across 0.75, 2.0, and 5.0 second endpoints of run-macro yaw-rate RMSE",
            "candidate_gains": list(GAINS),
            "selected_gain": selected_gain,
            "change_definition": "r_next = r_current + gain * (frozen_atlas(r_current, causal_history) - r_current)",
            "future_sensor_or_truth_input": False,
        },
        "training_gain_scores": train_scores,
        "heldout_results": heldout,
        "limitations": [
            "the gain is a one-parameter diagnostic, not a new vehicle model",
            "base HGB/EDSSM models were trained on these training captures; training-run objective is not a generalization estimate",
            "the high-steering data contain only three independent runs and practice diagnostics only two",
            "the existing validation set has informed prior candidate comparisons, so it is not a pristine final test",
            "this does not establish lap-length offline simulator accuracy",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dataset", type=Path, default=HGB_TRAIN_DATASET)
    parser.add_argument("--highsteer-dataset", type=Path,
                        default=HIGHSTEER_DATASET)
    parser.add_argument("--practice-dataset", type=Path,
                        default=PRACTICE_DATASET)
    args = parser.parse_args()
    report = run(args.output.resolve(), args.dataset.resolve(),
                 args.highsteer_dataset.resolve(), args.practice_dataset.resolve())
    print(json.dumps({
        "output": str(args.output),
        "selected_gain": report["fit_protocol"]["selected_gain"],
        "training_objective_by_gain": {
            gain: row["yaw_objective_mean_of_run_macro_rmse_rps"]
            for gain, row in report["training_gain_scores"].items()},
        "heldout": {
            name: row["selected_gain"]
            for name, row in report["heldout_results"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
