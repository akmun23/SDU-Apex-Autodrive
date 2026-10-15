#!/usr/bin/env python3
"""Fit grouped-CV Odom-u residuals from already replayed train-only rows.

Profiles are: current Odom sensors (7 features), those plus permitted actuator
feedback sensors (9), and an explicitly non-deployable command-output reference
(11). No bags are opened and no validation/test rows are used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import train_odom_u_residual as trainer


PROFILES = {
    7: "existing_odom_sensor_inputs",
    9: "plus_permitted_actuator_feedback_sensors",
    11: "nondeployable_command_output_reference",
}


def _fit_profile(rows: dict[str, np.ndarray], count: int
                 ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = trainer.FEATURE_NAMES[:count]
    x, y, groups, feature_names, _, _ = trainer._history_rows(
        rows["base_features"][:, :count], rows["base_residual_u"],
        rows["base_run_id"], rows["source_stamp_ns"],
        rows["packet_sequence"], 1, names)
    model, scores = trainer._fit(x, y, groups, feature_names, 1)
    model["base_feature_names"] = list(names)
    model["input_profile"] = PROFILES[count]
    model["runtime_input_policy"] = {
        7: "current source odometry u/v, rear encoder-angle mean, and IMU ax/ay/yaw rate",
        9: "profile 7 plus permitted steering and throttle feedback sensor topics",
        11: "training-only comparator includes actuator command output topics; never deploy",
    }[count]
    model["training_policy"] = (
        "clean train-only captures; one-step simulator-truth u residual; "
        "run-disjoint cross-validation; no validation, test, or final-test rows")
    model["training_provenance"] = {
        "rows_file": str(rows["_path"].resolve()),
        "rows_sha256": trainer._sha256(rows["_path"]),
        "run_count": int(len(np.unique(groups))),
        "sample_count": int(len(y)),
        "history_steps": 1,
        "input_feature_count": count,
        "cross_validation": scores,
    }
    return model, scores


def fit(rows_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {output_dir}")
    with np.load(rows_path, allow_pickle=False) as archive:
        rows = {name: archive[name] for name in (
            "base_features", "base_residual_u", "base_run_id",
            "source_stamp_ns", "packet_sequence")}
    rows["_path"] = rows_path
    if rows["base_features"].ndim != 2 or rows["base_features"].shape[1] != 11:
        raise ValueError("training rows do not match the 11-feature schema")
    if (len(rows["base_features"]) != len(rows["base_residual_u"])
            or len(rows["base_run_id"]) != len(rows["base_residual_u"])):
        raise ValueError("training row arrays have inconsistent lengths")
    if set(np.unique(rows["base_run_id"])) & {"", "None"}:
        raise ValueError("training rows contain an empty run group")
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "rows_file": str(rows_path.resolve()),
        "training_runs": int(len(np.unique(rows["base_run_id"]))),
        "samples": int(len(rows["base_residual_u"])),
        "target": "truth_u_minus_current_source_odom_u",
        "profiles": {},
    }
    for count, profile in PROFILES.items():
        model, scores = _fit_profile(rows, count)
        model_path = output_dir / f"odom_u_residual_{profile}.json"
        model_path.write_text(json.dumps(
            model, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        selected = min(scores, key=lambda item: (
            item["median_run_corrected_u_rmse_mps"],
            item["mean_run_corrected_u_rmse_mps"]))
        report["profiles"][profile] = {
            "deployable_by_topic_policy": count < 11,
            "selected_alpha": selected["alpha"],
            "mean_run_baseline_rmse_mps": selected[
                "mean_run_baseline_u_rmse_mps"],
            "mean_run_corrected_rmse_mps": selected[
                "mean_run_corrected_u_rmse_mps"],
            "median_run_baseline_rmse_mps": selected[
                "median_run_baseline_u_rmse_mps"],
            "median_run_corrected_rmse_mps": selected[
                "median_run_corrected_u_rmse_mps"],
            "runs_improved": selected["runs_improved"],
            "run_count": len(selected["per_run_corrected_u_rmse_mps"]),
            "model": str(model_path),
            "all_alphas": scores,
        }
    report_path = output_dir / "input_ablation_summary.json"
    report_path.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = fit(args.training_rows.resolve(), args.output_dir.resolve())
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"Odom-u input ablation failed: {exc}\n")
    print(json.dumps({
        "runs": report["training_runs"],
        "samples": report["samples"],
        "profiles": {
            key: value["median_run_corrected_rmse_mps"]
            for key, value in report["profiles"].items()},
        "output": str(args.output_dir.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
