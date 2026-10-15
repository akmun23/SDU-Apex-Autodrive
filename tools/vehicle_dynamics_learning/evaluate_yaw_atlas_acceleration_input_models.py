#!/usr/bin/env python3
"""Compare causal acceleration inputs for the frozen yaw-atlas plant replay.

The yaw atlas was trained on IMU acceleration channels. This compares the
existing mixed-head acceleration inputs with acceleration derived from the
assembled HGB/EDSSM/atlas state transition. A midpoint-vs-current body-frame
transport ablation separates the discrete transition convention from the
sensor-input source. Held-out data are scored only; no fitting or simulator
run occurs here.
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

from tools.vehicle_dynamics_learning.evaluate_composed_latent_reconciliation import (
    WINDOWS_PER_RUN,
)
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
from tools.vehicle_dynamics_learning.evaluate_yaw_atlas_recursive_gain import (
    HORIZON,
    _run_windows,
)
from tools.vehicle_dynamics_learning.run_truncated_history_transition import ROOT
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "hybrid_fixed40_longitudinal_yaw_atlas_candidate_20261009/"
    "yaw_atlas_acceleration_input_comparison_20261009.json")
VARIANTS = {
    "current_transport_model_heads": ("current", "model_heads"),
    "midpoint_transport_model_heads": ("midpoint", "model_heads"),
    "midpoint_transport_composed_transition": ("midpoint", "composed_transition"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output_path: Path = OUTPUT) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    inputs = (HGB_BUNDLE, EDSSM_CHECKPOINT, YAW_ATLAS_BUNDLE,
              HGB_TRAIN_DATASET, HIGHSTEER_DATASET, PRACTICE_DATASET)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    started = time.time()
    hgb = joblib.load(HGB_BUNDLE)
    atlas = joblib.load(YAW_ATLAS_BUNDLE)
    for model in (list(atlas["global_event_models"].values())
                  + list(atlas["local_expert_models"].values())):
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    _, edssm, metadata = _load_model(EDSSM_CHECKPOINT, "cpu")

    data_by_name = {
        "whole_run_validation": _load_dataset(HGB_TRAIN_DATASET),
        "highsteer_validation": _load_dataset(HIGHSTEER_DATASET),
        "unseen_practice_diagnostic": _load_dataset(PRACTICE_DATASET),
    }
    windows_by_name: dict[str, list[dict[str, int]]] = {}
    windows_by_name["whole_run_validation"] = _run_windows(
        data_by_name["whole_run_validation"], "validation",
        _bundle_lags(hgb), WINDOWS_PER_RUN)
    windows_by_name["highsteer_validation"] = _windows(
        data_by_name["highsteer_validation"], HIGHSTEER_DATASET,
        HORIZON, _bundle_lags(hgb))
    windows_by_name["unseen_practice_diagnostic"] = _windows(
        data_by_name["unseen_practice_diagnostic"], PRACTICE_DATASET,
        HORIZON, _bundle_lags(hgb))

    splits_expected = {
        "whole_run_validation": "validation",
        "highsteer_validation": "validation",
        "unseen_practice_diagnostic": "unseen_practice",
    }
    trained_ids = (set(hgb["training_run_ids"])
                   | set(metadata["training_runs"]))
    for name, data in data_by_name.items():
        ids = np.asarray(data["run_ids"]).astype(str)
        split = np.asarray(data["splits"]).astype(str)
        selected = windows_by_name[name]
        selected_ids = {str(ids[row["run"]]) for row in selected}
        if not selected_ids or selected_ids & trained_ids:
            raise ValueError(f"{name} is empty or overlaps a fitted base model")
        if any(split[row["run"]] != splits_expected[name] for row in selected):
            raise ValueError(f"{name} includes rows from an unintended split")

    results: dict[str, Any] = {}
    for variant, (latent_scheme, acceleration_source) in VARIANTS.items():
        results[variant] = {}
        for name, data in data_by_name.items():
            score = _rollout(
                data, hgb, edssm, metadata, atlas, windows_by_name[name],
                HORIZON, yaw_delta_gain=1.0,
                include_oracle_diagnostics=False,
                reconcile_composed_latent=True,
                latent_acceleration_scheme=latent_scheme,
                atlas_acceleration_source=acceleration_source)
            results[variant][name] = {
                "windows": score["windows"],
                "windows_by_run": score["windows_by_run"],
                "heldout_run_ids": sorted(score["per_run"]),
                "macro_run_rmse": score["macro_run_rmse"],
                "per_run": score["per_run"],
                "yaw_atlas_one_step_selected_windows": score[
                    "yaw_atlas_one_step_selected_windows"],
            }

    report = {
        "schema_version": 1,
        "purpose": "test whether physically consistent, sensor-equivalent acceleration feedback improves recursive use of the frozen yaw atlas",
        "research_only": True,
        "runtime_or_production_integration": False,
        "simulator_was_run": False,
        "fit_or_selection_data": "none; existing validation and practice splits are diagnostics only",
        "test_or_final_test_used": False,
        "fixed_dt_s": DT_S,
        "horizon_s": HORIZON * DT_S,
        "variants": {
            name: {"latent_acceleration_scheme": latent,
                  "atlas_acceleration_source": source}
            for name, (latent, source) in VARIANTS.items()},
        "data_and_model_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in inputs},
        "heldout_run_ids": {
            name: sorted({
                str(np.asarray(data["run_ids"]).astype(str)[row["run"]])
                for row in windows_by_name[name]})
            for name, data in data_by_name.items()},
        "window_counts": {
            name: len(value) for name, value in windows_by_name.items()},
        "results": results,
        "runtime_seconds": time.time() - started,
        "limitations": [
            "ground truth initializes only the measured history and scores predictions; future plant inputs are recursively generated",
            "the two practice captures are weak run-level evidence",
            "this remains a short five-second plant replay, not a full-lap validation",
            "the frozen yaw atlas remains a one-step sensor-conditioned model; this experiment tests only its acceleration feedback channels",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output.resolve())
    summary = {
        variant: {
            split: {
                horizon: values["hybrid_atlas"]
                for horizon, values in data["macro_run_rmse"].items()}
            for split, data in rows.items()}
        for variant, rows in report["results"].items()}
    print(json.dumps({"output": str(args.output), "summary": summary},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
