#!/usr/bin/env python3
"""Score a mirrored-trajectory ensemble on frozen held-out whole runs.

This experimental offline scorer averages a model rollout with a second
command-only rollout from the reflected initial state/history/commands. The
branches recurse independently, use no future truth or measured feedback, and
are compared on exactly the starts frozen in the paired-comparison report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from tools.vehicle_dynamics_learning.diagnose_effective_race_first_step import (
    _predict_domain,
)
from tools.vehicle_dynamics_learning.diagnose_recursive_first_divergence import (
    _score_domain,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset
from tools.vehicle_dynamics_learning.turn_reflection_projection import (
    HighSteerStepwiseReflectionModel,
    TurnReflectionProjectedModel,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/" \
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/" \
    "encoder_raw_state_teacher_v1"
DEFAULT_CHECKPOINT = DATA_ROOT / \
    "edssm_gru_z32_e2_rollresidual_10s_yawonly_wheelonly_lowthrottle4_joint_lr3e5_seed101/best.pt"
DEFAULT_DYNAMIC = DATA_ROOT / "openplane_dynamics_raw_wheels.npz"
DEFAULT_PRACTICE = DATA_ROOT / "practice_dynamics_raw_wheels.npz"
DEFAULT_COMPARISON = DATA_ROOT / \
    "paired_turn_reflection_last_vs_parent_v1/paired_comparison.json"
DEFAULT_PRACTICE_MANIFEST = DATA_ROOT / \
    "practice_dynamics_raw_wheels_manifest.json"
DEFAULT_OUTPUT = DATA_ROOT / "turn_reflection_stepwise_projection_parent_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_practice_lineage(reference: dict[str, Any],
                               practice_dataset: Path,
                               practice_manifest: Path) -> dict[str, Any]:
    """Allow the frozen source hash or its explicitly recorded sidecar view."""
    actual_hash = _sha256(practice_dataset)
    expected_hash = reference.get("practice_dataset_sha256")
    if actual_hash == expected_hash:
        return {"mode": "exact_dataset", "actual_sha256": actual_hash}
    manifest = json.loads(practice_manifest.read_text(encoding="utf-8"))
    if (manifest.get("source_dataset_sha256") != expected_hash
            or manifest.get("output_dataset_sha256") != actual_hash):
        raise ValueError(
            "practice dataset is neither the frozen source nor its hash-verified sidecar derivative")
    return {
        "mode": "hash_verified_sidecar_derivative",
        "source_dataset": manifest.get("source_dataset"),
        "source_sha256": manifest["source_dataset_sha256"],
        "manifest": str(practice_manifest),
        "manifest_sha256": _sha256(practice_manifest),
        "actual_sha256": actual_hash,
    }


def _windows_from_manifest(data: dict[str, Any], rows: list[dict[str, Any]],
                           horizon_steps: int) -> list[dict[str, Any]]:
    run_lookup = {str(name): index for index, name in
                  enumerate(np.asarray(data["run_ids"]).astype(str))}
    windows = []
    for row in rows:
        run_name = str(row["run_id"])
        if run_name not in run_lookup:
            raise ValueError(f"frozen manifest has unknown run {run_name}")
        start = int(row["start"])
        if (start < 79 or start + horizon_steps >= len(data["frames"])
                or int(row.get("sequence_id", -1)) < -1):
            raise ValueError("frozen rollout start is outside the dataset")
        windows.append({
            "start": start,
            "run": run_lookup[run_name],
            "sequence_id": int(row.get("sequence_id", -1)),
            "regimes": list(row.get("regimes", [])),
        })
    if not windows:
        raise ValueError("frozen rollout manifest is empty")
    return windows


def _paired_run_deltas(baseline: dict[str, Any], candidate: dict[str, Any],
                       metrics: tuple[str, ...]) -> dict[str, Any]:
    """Report paired candidate-minus-parent errors with runs as units."""
    result: dict[str, Any] = {}
    for metric in metrics:
        base_curve = baseline["error_curves"][metric]["cumulative_horizon_rmse"]
        candidate_curve = candidate["error_curves"][metric][
            "cumulative_horizon_rmse"]
        horizons = {}
        for horizon in sorted(
                base_curve, key=lambda value: float(value.removesuffix("s"))):
            base_by_run = base_curve[horizon]["per_run_rmse"]
            candidate_by_run = candidate_curve[horizon]["per_run_rmse"]
            common_runs = sorted(set(base_by_run) & set(candidate_by_run))
            if len(common_runs) != len(base_by_run) or len(common_runs) != len(candidate_by_run):
                raise ValueError("baseline and projection whole-run sets differ")
            deltas = np.asarray([
                candidate_by_run[run] - base_by_run[run]
                for run in common_runs], dtype=np.float64)
            mean = float(np.mean(deltas))
            if len(deltas) > 1:
                half_width = float(student_t.ppf(0.975, len(deltas) - 1)
                                   * np.std(deltas, ddof=1)
                                   / np.sqrt(len(deltas)))
                interval = [mean - half_width, mean + half_width]
            else:
                interval = None
            horizons[horizon] = {
                "candidate_minus_parent_macro_run_rmse_m_or_rad_or_mps": mean,
                "per_run_delta": dict(zip(common_runs, deltas.tolist())),
                "run_count": len(common_runs),
                "run_min_delta": float(np.min(deltas)),
                "run_max_delta": float(np.max(deltas)),
                "paired_t_95pct_interval": interval,
            }
        result[metric] = horizons
    return result


def _high_steering_windows(data: dict[str, Any],
                          windows: list[dict[str, Any]],
                          horizon_steps: int,
                          threshold_rad: float = 0.30
                          ) -> list[dict[str, Any]]:
    """Keep frozen starts whose supplied future command reaches the gate."""
    frames = np.asarray(data["frames"])
    return [row for row in windows
            if np.max(np.abs(frames[int(row["start"]) + 1:
                                    int(row["start"]) + horizon_steps + 1, 7]))
            >= threshold_rad]


def score(checkpoint: Path, dynamic_dataset: Path, practice_dataset: Path,
          comparison: Path, output: Path, device: str = "cpu",
          practice_manifest: Path = DEFAULT_PRACTICE_MANIFEST,
          batch_size: int = 8,
          projection_mode: str = "stepwise_high_steer") -> dict[str, Any]:
    checkpoint, dynamic_dataset, practice_dataset, comparison, output = (
        path.resolve() for path in
        (checkpoint, dynamic_dataset, practice_dataset, comparison, output))
    practice_manifest = practice_manifest.resolve()
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if projection_mode not in ("trajectory", "stepwise_high_steer"):
        raise ValueError("unknown turn-reflection projection mode")
    reference = json.loads(comparison.read_text(encoding="utf-8"))
    if (reference.get("dynamic_dataset_sha256") != _sha256(dynamic_dataset)
            or not reference.get("selection_was_frozen_before_practice_transfer")
            or reference.get("future_truth_or_feedback_used") is not False):
        raise ValueError("paired-comparison manifests do not match held-out datasets")
    practice_lineage = _validate_practice_lineage(
        reference, practice_dataset, practice_manifest)
    candidate_name = reference["paired_candidate_differences"]["candidate_a"]
    if candidate_name not in reference["dynamic_validation"]:
        raise ValueError("frozen dynamic window manifest is missing the baseline")
    dynamic_data = _load_dataset(dynamic_dataset)
    practice_data = _load_dataset(practice_dataset)
    dynamic_windows = _windows_from_manifest(
        dynamic_data,
        reference["dynamic_validation"][candidate_name]["window_manifest"], 400)
    practice_windows = _windows_from_manifest(
        practice_data,
        reference["production_practice_transfer"][candidate_name][
            "window_manifest"], 200)
    torch, model, metadata = _load_model(checkpoint, device)
    projected = (TurnReflectionProjectedModel(model)
                 if projection_mode == "trajectory" else
                 HighSteerStepwiseReflectionModel(model))
    results = {}
    for name, data, split, windows, horizon in (
            ("dynamic_validation", dynamic_data, {"validation"},
             dynamic_windows, 400),
            ("practice_unseen", practice_data, {"unseen_practice"},
             practice_windows, 200)):
        recursive_split = ("validation" if name == "dynamic_validation"
                           else "unseen_practice")
        baseline_one_step = _predict_domain(
            torch, model, metadata, data, split, device)
        projected_one_step = _predict_domain(
            torch, projected, metadata, data, split, device)
        baseline_recursive = _score_domain(
            torch, model, metadata, data, recursive_split, device,
            max_windows_per_run=24, batch_size=batch_size,
            horizon_steps=horizon, fixed_windows=windows)
        projected_recursive = _score_domain(
            torch, projected, metadata, data, recursive_split, device,
            max_windows_per_run=24, batch_size=batch_size,
            horizon_steps=horizon, fixed_windows=windows)
        results[name] = {
            "one_step": {"parent": baseline_one_step,
                         "projection": projected_one_step},
            "recursive": {"parent": baseline_recursive,
                          "projection": projected_recursive},
            "paired_run_delta_projection_minus_parent": _paired_run_deltas(
                baseline_recursive, projected_recursive,
                ("forward_speed_mps", "yaw_rate_rps",
                 "rear_wheel_pair_rms_mps", "position_radial_m",
                 "heading_rad", "roll_rad", "roll_rate_rps")),
        }
    high_windows = _high_steering_windows(
        dynamic_data, dynamic_windows, horizon_steps=400)
    if not high_windows:
        raise ValueError("frozen dynamic benchmark has no high-steering command windows")
    high_parent = _score_domain(
        torch, model, metadata, dynamic_data, "validation", device,
        max_windows_per_run=24, batch_size=batch_size,
        horizon_steps=400, fixed_windows=high_windows)
    high_projection = _score_domain(
        torch, projected, metadata, dynamic_data, "validation", device,
        max_windows_per_run=24, batch_size=batch_size,
        horizon_steps=400, fixed_windows=high_windows)
    results["dynamic_high_steering_subset"] = {
        "steering_command_threshold_rad": 0.30,
        "window_count": len(high_windows),
        "one_step": {},
        "recursive": {"parent": high_parent,
                      "projection": high_projection},
        "paired_run_delta_projection_minus_parent": _paired_run_deltas(
            high_parent, high_projection,
            ("forward_speed_mps", "yaw_rate_rps",
             "rear_wheel_pair_rms_mps", "position_radial_m", "heading_rad",
             "roll_rad", "roll_rate_rps")),
    }
    result = {
        "schema_version": 1,
        "purpose": "held-out score of reflection-projected command-only rollouts",
        "projection_mode": projection_mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_runs": metadata["training_runs"],
        "comparison_manifest": str(comparison),
        "comparison_manifest_sha256": _sha256(comparison),
        "dynamic_dataset_sha256": _sha256(dynamic_dataset),
        "practice_dataset_sha256": _sha256(practice_dataset),
        "practice_dataset_lineage": practice_lineage,
        "dynamic_rollout_windows": len(dynamic_windows),
        "practice_rollout_windows": len(practice_windows),
        "future_inputs_only": ["steering_command_rad", "throttle_command_norm"],
        "future_truth_or_sensor_feedback_used": False,
        "projection_definition": (
            "average output trajectories from two independently recursive branches"
            if projection_mode == "trajectory" else
            "project each transition above 0.30 rad, feed projected physical state back, and advance each branch latent from projected acceleration"),
        "test_and_final_test_used": False,
        "domains": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dynamic-dataset", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--practice-dataset", type=Path, default=DEFAULT_PRACTICE)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--practice-manifest", type=Path,
                        default=DEFAULT_PRACTICE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=("trajectory", "stepwise_high_steer"),
                        default="stepwise_high_steer")
    args = parser.parse_args()
    result = score(args.checkpoint, args.dynamic_dataset,
                   args.practice_dataset, args.comparison, args.output,
                   args.device, args.practice_manifest, args.batch_size,
                   args.mode)
    summary = {}
    for domain, rows in result["domains"].items():
        recursive = rows["recursive"]["projection"]
        baseline = rows["recursive"]["parent"]
        horizons = (["0.750s", "2.000s", "5.000s", "10.000s"]
                    if domain.startswith("dynamic") else
                    ["0.750s", "2.000s", "5.000s"])
        metric_summary = {}
        for metric in ("forward_speed_mps", "yaw_rate_rps",
                       "rear_wheel_pair_rms_mps", "position_radial_m",
                       "heading_rad", "roll_rad", "roll_rate_rps"):
            paired = rows["paired_run_delta_projection_minus_parent"][metric]
            metric_summary[metric] = {}
            for horizon in horizons:
                base = baseline["error_curves"][metric][
                    "cumulative_horizon_rmse"][horizon]
                candidate = recursive["error_curves"][metric][
                    "cumulative_horizon_rmse"][horizon]
                delta = paired[horizon]
                metric_summary[metric][horizon] = {
                    "parent_macro_run_rmse": base["macro_run_rmse"],
                    "projection_macro_run_rmse": candidate["macro_run_rmse"],
                    "candidate_minus_parent": delta[
                        "candidate_minus_parent_macro_run_rmse_m_or_rad_or_mps"],
                    "paired_run_count": delta["run_count"],
                    "paired_t_95pct_interval": delta[
                        "paired_t_95pct_interval"],
                }
        summary[domain] = {
            "window_count": recursive["window_count"],
            "whole_run_ids": recursive["whole_run_ids"],
            "metrics": metric_summary,
        }
    print(json.dumps({"output": str(args.output.resolve()), "summary": summary},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
