#!/usr/bin/env python3
"""Compare two saved fixed-40-Hz plants on identical practice windows.

The recurrent EDSSM window manifest freezes starts selected before scoring.
The HGB longitudinal/raw-wheel candidate is rerun on those same starts so its
recursive results are directly comparable. Only two unseen practice captures
are used; test and final-test are not accessed for scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _bundle_lags,
    _run_rollout,
    _sha256,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LONGITUDINAL_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
DEFAULT_EDSSM_SCORE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "effective_race_teacher_model_fixed40_10s_practice_transfer_5s_20261009.json")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "fixed40_candidate_same_practice_starts_comparison_v2_20261009.json")
HORIZONS = {"0.750s": 30, "2.000s": 80, "5.000s": 200}
PAIRED_METRICS = {
    "forward_speed_rmse_mps": ("u_rmse_mps", "raw/u_com_mps"),
    "lateral_speed_rmse_mps": ("v_rmse_mps", "raw/v_com_mps"),
    "yaw_rate_rmse_rps": ("yaw_rate_rmse_rps", "raw/yaw_rate_rps"),
    "position_component_rmse_m": ("position_rmse_m", "position_m"),
    "heading_rmse_rad": ("yaw_pose_rmse_rad", "heading_rad"),
}


def _validate_and_load_windows(score_path: Path, dataset_path: Path,
                               bundle: dict[str, Any], data: dict[str, Any]
                               ) -> list[dict[str, int]]:
    score = json.loads(score_path.read_text(encoding="utf-8"))
    if score.get("dataset_sha256") != _sha256(dataset_path):
        raise ValueError("EDSSM score report is for a different practice dataset")
    if score.get("checkpoint_sha256") is None:
        raise ValueError("EDSSM score does not identify its checkpoint hash")
    if score.get("split_scored") != "unseen_practice":
        raise ValueError("comparison requires unseen-practice EDSSM starts")
    run_index = {str(name): index
                 for index, name in enumerate(np.asarray(data["run_ids"]).astype(str))}
    sequence_ids_by_run = {
        run: set(np.flatnonzero(np.asarray(data["seq_run"]) == run).tolist())
        for run in range(len(data["run_ids"]))
    }
    context = max(_bundle_lags(bundle))
    valid = np.asarray(data["encoder_raw_valid"], dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    windows = []
    for row in score["metrics"]["window_manifest"]:
        rid = str(row["run_id"])
        if rid not in run_index:
            raise ValueError(f"score run missing from practice dataset: {rid}")
        run = run_index[rid]
        sequence = int(row["sequence_id"])
        start = int(row["start_frame"])
        if sequence not in sequence_ids_by_run[run]:
            raise ValueError("EDSSM manifest sequence and run index disagree")
        begin, end = map(int, data["bounds"][sequence])
        if not (begin + context <= start < end):
            raise ValueError("paired start lacks the longitudinal model context")
        windows.append({"run": run, "sequence": sequence, "start": start})

    for key, horizon in HORIZONS.items():
        for window in windows:
            start = window["start"]
            sequence = window["sequence"]
            begin, end = map(int, data["bounds"][sequence])
            if start + horizon >= end:
                raise ValueError(f"paired start lacks {key} future")
            if not valid[start - context:start + horizon + 1].all():
                raise ValueError(f"paired start has invalid raw encoder data at {key}")
            if not np.isclose(dt[start - context + 1:start + horizon + 1],
                              0.025, rtol=0.0, atol=1e-7).all():
                raise ValueError(f"paired start has a non-25-ms interval at {key}")
    return windows


def compare(bundle_path: Path, practice_path: Path, edssm_score_path: Path,
            output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite comparison: {output_path}")
    bundle = joblib.load(bundle_path)
    if not bundle.get("research_only") or not bundle.get("raw_wheel_models"):
        raise ValueError("longitudinal candidate must include predicted raw-wheel state")
    data = _load_dataset(practice_path)
    windows = _validate_and_load_windows(
        edssm_score_path, practice_path, bundle, data)
    lags = _bundle_lags(bundle)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    raw_truth = np.asarray(data["encoder_raw_surface_mps"], dtype=np.float64)
    score = json.loads(edssm_score_path.read_text(encoding="utf-8"))
    results = {}
    for label, horizon in HORIZONS.items():
        hgb = _run_rollout(
            data, bundle, windows, horizon, lags, u_update="increment",
            raw_input_mode="predicted")
        hgb_macro = hgb["macro_run_metrics"][label]
        edssm_horizon = score["metrics"]["horizons"][
            {"0.750s": "0.75s", "2.000s": "2s", "5.000s": "5s"}[label]]
        paired = {}
        for metric_name, (hgb_name, edssm_name) in PAIRED_METRICS.items():
            hgb_values = hgb_macro["candidate_macro_run_rmse"].get(hgb_name)
            edssm_values = edssm_horizon[edssm_name]["macro_run_mean"]
            paired[metric_name] = {
                "HGB_macro_run_mean": hgb_values,
                "EDSSM_macro_run_mean": edssm_values,
                "HGB_minus_EDSSM": (
                    hgb_values - edssm_values
                    if hgb_values is not None else None),
                "per_run": {},
            }
            for rid, hgb_run in hgb["per_run"].items():
                hgb_row = hgb_run[label]["candidate"].get(hgb_name)
                edssm_row = edssm_horizon[edssm_name]["per_run"].get(rid)
                paired[metric_name]["per_run"][rid] = {
                    "HGB": hgb_row,
                    "EDSSM": edssm_row,
                    "HGB_minus_EDSSM": (
                        hgb_row - edssm_row
                        if hgb_row is not None and edssm_row is not None else None),
                }
        raw_mse_by_run: dict[str, list[float]] = {}
        for window in windows:
            start = window["start"]
            run_id = str(run_ids[window["run"]])
            error = raw_truth[start + horizon] - raw_truth[start]
            raw_mse_by_run.setdefault(run_id, []).append(float(np.mean(error ** 2)))
        edssm_pair_by_run = {}
        for run_id in sorted(raw_mse_by_run):
            left_rmse = edssm_horizon[
                "raw/rear_left_surface_speed_mps"]["per_run"][run_id]
            right_rmse = edssm_horizon[
                "raw/rear_right_surface_speed_mps"]["per_run"][run_id]
            edssm_pair_by_run[run_id] = float(np.sqrt(
                (left_rmse ** 2 + right_rmse ** 2) / 2.0))
        persistence_pair_by_run = {
            run_id: float(np.sqrt(np.mean(values)))
            for run_id, values in raw_mse_by_run.items()
        }
        results[label] = {
            "windows": len(windows),
            "runs": sorted({str(run_ids[row["run"]]) for row in windows}),
            "paired_metrics": paired,
            "HGB_raw_encoder_pair_rmse_mps": hgb_macro[
                "candidate_macro_run_rmse"].get("raw_encoder_pair_rmse_mps"),
            "HGB_filtered_wheel_pair_rmse_mps": hgb_macro[
                "candidate_macro_run_rmse"].get("rear_wheel_pair_rmse_mps"),
            "HGB_raw_encoder_persistence_pair_rmse_mps": hgb_macro[
                "constant_state_baseline_macro_run_rmse"].get(
                    "raw_encoder_pair_rmse_mps"),
            "EDSSM_raw_rear_wheel_pair_rmse_mps": float(np.mean(
                list(edssm_pair_by_run.values()))),
            "raw_encoder_persistence_pair_rmse_mps": float(np.mean(
                list(persistence_pair_by_run.values()))),
            "raw_wheel_pair_by_run": {
                run_id: {
                    "EDSSM": edssm_pair_by_run[run_id],
                    "HGB": hgb["per_run"][run_id][label]["candidate"].get(
                        "raw_encoder_pair_rmse_mps"),
                    "persistence": persistence_pair_by_run[run_id],
                }
                for run_id in edssm_pair_by_run
            },
        }

    score = json.loads(edssm_score_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "purpose": "paired practice-transfer comparison of the raw-wheel EDSSM teacher and new one-step-derived HGB candidate",
        "practice_dataset": str(practice_path.relative_to(ROOT)),
        "practice_dataset_sha256": _sha256(practice_path),
        "HGB_bundle": str(bundle_path.relative_to(ROOT)),
        "HGB_bundle_sha256": _sha256(bundle_path),
        "EDSSM_checkpoint": score["checkpoint"],
        "EDSSM_checkpoint_sha256": score["checkpoint_sha256"],
        "EDSSM_score_report": str(edssm_score_path.relative_to(ROOT)),
        "EDSSM_score_report_sha256": _sha256(edssm_score_path),
        "paired_window_contract": {
            "identical_initial_frames_and_futures": True,
            "number_of_starts": len(windows),
            "starts_by_run": {
                str(run_ids[run]): sum(row["run"] == run for row in windows)
                for run in sorted({row["run"] for row in windows})
            },
            "HGB_uses_causal_64_step_lags": True,
            "EDSSM_uses_initial_80_step_history": True,
            "recorded_future_commands_exogenous_for_both": True,
            "future_state_truth_used": False,
            "test_and_final_test_scored": False,
            "only_two_independent_practice_runs": True,
        },
        "horizons": results,
        "limitations": [
            "The two-run practice result is transfer evidence, not broad statistical proof.",
            "EDSSM and HGB use different model structures and history lengths; common starts remove the largest data-selection confound but do not make their internal states identical.",
            "HGB raw encoder rates and EDSSM raw rear-wheel states use the same fixed-25-ms target definition.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_LONGITUDINAL_BUNDLE)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--edssm-score", type=Path, default=DEFAULT_EDSSM_SCORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = compare(args.bundle.resolve(), args.practice_dataset.resolve(),
                     args.edssm_score.resolve(), args.output.resolve())
    print(json.dumps({
        horizon: {
            name: {
                "HGB": values["HGB_macro_run_mean"],
                "EDSSM": values["EDSSM_macro_run_mean"],
                "HGB_minus_EDSSM": values["HGB_minus_EDSSM"],
            }
            for name, values in row["paired_metrics"].items()
        }
        for horizon, row in report["horizons"].items()
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
