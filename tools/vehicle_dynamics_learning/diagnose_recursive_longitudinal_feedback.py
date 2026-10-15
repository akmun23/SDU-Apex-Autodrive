#!/usr/bin/env python3
"""Attribute recursive longitudinal-candidate error to predicted state feedback.

Runs paired, diagnostic-only oracle interventions on the existing fixed-25-ms
whole-run validation and practice captures. Each intervention replaces one
predicted state channel with its future simulator label at every rollout step.
These are causal ablations for locating error propagation, not model scores or
deployable predictions. Test/final-test runs are never selected or scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from tools.vehicle_dynamics_learning.evaluate_recursive_longitudinal_candidate import (
    _bundle_lags,
    _select_windows,
    _sha256,
    _run_rollout,
)
from tools.vehicle_dynamics_learning.fit_fixed40_wheel_response_candidate import (
    DEFAULT_DATASET,
    DEFAULT_PRACTICE,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_with_fixed40_wheels_candidate_20261009/"
    "longitudinal_raw_wheel_candidate.joblib")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "longitudinal_recursive_feedback_diagnosis_20261009.json")
HORIZONS = (40, 50, 60, 80, 120, 200)
INTERVENTIONS = (
    None, "u", "v", "yaw_rate", "steering_feedback",
    "throttle_feedback", "left_filtered_wheel", "right_filtered_wheel",
)
METRICS = (
    "u_rmse_mps", "v_rmse_mps", "yaw_rate_rmse_rps",
    "rear_wheel_pair_rmse_mps", "position_rmse_m",
    "yaw_pose_rmse_rad", "raw_encoder_pair_rmse_mps",
)


def _eligible_raw_windows(data: dict[str, Any], windows: list[dict[str, int]],
                          context: int, horizon: int) -> list[dict[str, int]]:
    valid = data.get("encoder_raw_valid")
    if valid is None:
        raise ValueError("candidate requires the fixed-40-Hz encoder sidecar")
    valid = np.asarray(valid, dtype=bool)
    dt = np.asarray(data["dt_s"], dtype=np.float64)
    output = []
    for window in windows:
        start = int(window["start"])
        first = start - context
        last = start + horizon + 1
        if (first < 0 or not valid[first:last].all()
                or not np.isclose(dt[first + 1:last], 0.025,
                                  rtol=0.0, atol=1e-7).all()):
            continue
        output.append(window)
    return output


def _paired_summary(report: dict[str, Any], intervention: str | None
                    ) -> dict[str, Any]:
    macro = report["macro_run_metrics"]
    horizon_key = next(iter(macro))
    scores = macro[horizon_key]["candidate_macro_run_rmse"]
    per_run = report["per_run"]
    return {
        "intervention": intervention or "free_recursive",
        "windows": report["windows"],
        "eligible_whole_runs": len(per_run),
        "macro_run_rmse": {name: scores.get(name) for name in METRICS},
        "per_run": {
            run: {
                metric: values[horizon_key]["candidate"].get(metric)
                for metric in METRICS
            }
            for run, values in per_run.items()
        },
        "window_signature_sha256": report["window_signature_sha256"],
    }


def diagnose(bundle_path: Path, dataset_path: Path, practice_path: Path,
             output_path: Path, max_windows_per_run: int = 24,
             context_steps: int = 64) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnosis: {output_path}")
    bundle = joblib.load(bundle_path)
    if not bundle.get("research_only") or not bundle.get("raw_wheel_models"):
        raise ValueError("diagnosis requires the raw-wheel longitudinal research bundle")
    lags = _bundle_lags(bundle)
    if context_steps < max(lags):
        raise ValueError("context must cover the candidate's longest input lag")

    results: dict[str, Any] = {}
    for split_name, path, split in (
            ("openplane_validation", dataset_path, "validation"),
            ("unseen_practice", practice_path, "unseen_practice")):
        data = _load_dataset(path)
        results[split_name] = {}
        for horizon in HORIZONS:
            windows = _select_windows(
                data, split, horizon, max_windows_per_run, context_steps)
            windows = _eligible_raw_windows(data, windows, context_steps, horizon)
            if not windows:
                results[split_name][f"{horizon * 0.025:.3f}s"] = {
                    "windows": 0,
                    "reason": "no common contiguous fixed-cadence raw-wheel windows",
                }
                continue
            modes = {}
            for intervention in INTERVENTIONS:
                report = _run_rollout(
                    data, bundle, windows, horizon, lags,
                    u_update="increment", raw_input_mode="predicted",
                    teacher_force_state=intervention)
                modes[intervention or "free_recursive"] = _paired_summary(
                    report, intervention)
            signatures = {row["window_signature_sha256"]
                          for row in modes.values()}
            if len(signatures) != 1:
                raise AssertionError("state interventions did not use identical windows")
            results[split_name][f"{horizon * 0.025:.3f}s"] = {
                "windows": len(windows),
                "windows_by_run": {
                    str(run): int(sum(int(row["run"]) == run for row in windows))
                    for run in sorted({int(row["run"]) for row in windows})
                },
                "modes": modes,
            }

    report = {
        "schema_version": 1,
        "purpose": "diagnose which recursive predicted-state feedback channels drive longitudinal rollout error",
        "candidate_bundle": str(bundle_path.relative_to(ROOT)),
        "candidate_sha256": _sha256(bundle_path),
        "training_dataset_sha256": bundle["training_dataset_sha256"],
        "openplane_dataset_sha256": _sha256(dataset_path),
        "practice_dataset_sha256": _sha256(practice_path),
        "horizons_steps": list(HORIZONS),
        "window_context_steps": context_steps,
        "selected_interventions": list(INTERVENTIONS),
        "evaluation_policy": {
            "only_openplane_whole_run_validation_and_unseen_practice": True,
            "test_and_final_test_used": False,
            "same_windows_across_interventions": True,
            "commands_are_recorded_exogenous_replay_inputs": True,
            "predicted_raw_encoder_state_is_used": True,
            "future_ground_truth_intervention": (
                "one state channel is replaced by simulator truth at each step, "
                "diagnostic only; never use these numbers as plant performance"),
        },
        "results": results,
        "interpretation": (
            "A reduction under a state intervention identifies sensitivity to "
            "that feedback channel, not a standalone fix. Combined interventions "
            "are not additive because state dynamics are coupled."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-windows-per-run", type=int, default=24)
    parser.add_argument("--context-steps", type=int, default=64)
    args = parser.parse_args()
    if args.max_windows_per_run < 1:
        parser.error("max-windows-per-run must be positive")
    report = diagnose(args.bundle.resolve(), args.dataset.resolve(),
                      args.practice_dataset.resolve(), args.output.resolve(),
                      args.max_windows_per_run, args.context_steps)
    print(json.dumps({
        "output": str(args.output),
        "horizons": {
            split: {
                horizon: {
                    mode: values.get("macro_run_rmse")
                    for mode, values in entry.get("modes", {}).items()
                }
                for horizon, entry in split_results.items()
            }
            for split, split_results in report["results"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
