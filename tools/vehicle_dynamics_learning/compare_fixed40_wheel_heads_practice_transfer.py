#!/usr/bin/env python3
"""Compare frozen raw-wheel EDSSM checkpoints on identical practice starts.

This is a transfer-only evaluation of already selected checkpoints. It does
not fit, select, or alter models, and it does not open test/final-test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.score_effective_race_teacher import (
    _load_model,
    evaluate_model,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    physical_state_from_dataset,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261003/"
    "full_throttle_domain_v1/encoder_fixed40hz_v1")
DEFAULT_DATASET = MODEL_ROOT / "practice_dynamics_raw_wheels_fixed40hz.npz"
DEFAULT_BASE_SCORE = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "effective_race_teacher_model_fixed40_10s_practice_transfer_5s_20261009.json")
DEFAULT_OUTPUT = ROOT / (
    "live_runs/derived_dynamics_learning_20260928/"
    "wheel_head_practice_transfer_same_starts_v4_20261009.json")
CHECKPOINTS = {
    "five_second_parent": MODEL_ROOT / "model_fixed40hz_seed101_v1/best.pt",
    "ten_second_pose3": (
        MODEL_ROOT / "model_fixed40hz_10s_pose3_seed101_v1/best.pt"),
    "wheel_head_only": MODEL_ROOT / "wheel_head_fixed40hz_v1/best.pt",
    "wheel_head_body_distilled": (
        MODEL_ROOT / "wheel_head_body_distilled_v1/best.pt"),
    "longitudinal_distilled_fixed40": (
        MODEL_ROOT / "longitudinal_distilled_fixed40hz_v1/best.pt"),
    "longitudinal_distilled_all_motion": (
        MODEL_ROOT / "longitudinal_distilled_all_motion_v2/best.pt"),
    "ten_second_plant_comparator": (
        MODEL_ROOT / "model_fixed40hz_10s_seed101_v1/best.pt"),
}
HORIZONS = {"0.75s": 30, "2s": 80, "5s": 200}
METRICS = (
    "position_m", "position_radial_m", "heading_rad", "speed_mps",
    "raw/u_com_mps", "raw/v_com_mps", "raw/yaw_rate_rps",
    "raw/rear_left_surface_speed_mps", "raw/rear_right_surface_speed_mps",
    "raw/steering_actual_rad", "raw/throttle_feedback_norm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_windows(dataset: dict[str, Any], dataset_path: Path,
                 score_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prior = json.loads(score_path.read_text(encoding="utf-8"))
    if prior.get("dataset_sha256") != sha256(dataset_path):
        raise ValueError("base score and practice dataset hashes differ")
    if prior.get("split_scored") != "unseen_practice":
        raise ValueError("window source is not the unseen-practice split")
    if prior.get("test_and_final_test_used") is not False:
        raise ValueError("base score does not certify sealed test/final data")

    run_names = np.asarray(dataset["run_ids"]).astype(str)
    run_lookup = {name: index for index, name in enumerate(run_names)}
    windows = []
    raw_valid = np.asarray(dataset["encoder_raw_valid"], dtype=bool)
    dt = np.asarray(dataset["dt_s"], dtype=np.float64)
    for entry in prior["metrics"]["window_manifest"]:
        run_name = str(entry["run_id"])
        if run_name not in run_lookup:
            raise ValueError(f"scored run missing from dataset: {run_name}")
        run = run_lookup[run_name]
        sequence = int(entry["sequence_id"])
        start = int(entry["start_frame"])
        if int(dataset["seq_run"][sequence]) != run:
            raise ValueError("frozen window sequence/run mismatch")
        begin, end = map(int, dataset["bounds"][sequence])
        if start - 79 < begin or start + max(HORIZONS.values()) >= end:
            raise ValueError("frozen practice window lacks context/future")
        span = slice(start - 79, start + max(HORIZONS.values()) + 1)
        if not raw_valid[span].all():
            raise ValueError("frozen practice window contains invalid raw wheel data")
        if not np.isclose(dt[span][1:], 0.025, rtol=0.0, atol=1e-7).all():
            raise ValueError("frozen practice window is not fixed at 25 ms")
        windows.append({
            "run": run,
            "sequence_id": sequence,
            "start": start,
            "regimes": list(entry.get("regimes", [])),
        })
    if len(windows) != int(prior["metrics"]["window_count"]):
        raise ValueError("frozen score window count is inconsistent")
    if len({row["run"] for row in windows}) < 2:
        raise ValueError("transfer evaluation needs multiple independent runs")
    if np.any(np.asarray(dataset["splits"]).astype(str)[
            np.asarray(dataset["seq_run"], dtype=np.int64)[
                np.asarray([row["sequence_id"] for row in windows])]]
            != "unseen_practice"):
        raise ValueError("frozen windows include a non-practice run")
    return windows, prior


def evaluate_one(name: str, checkpoint_path: Path,
                 dataset: dict[str, Any], windows: list[dict[str, Any]],
                 expected_validation_runs: set[str]) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    torch, model, metadata = _load_model(checkpoint_path, "cpu")
    if metadata.get("wheel_state_source") != "raw_encoder":
        raise ValueError(f"{name} does not use raw encoder wheel states")
    if metadata.get("training_objective_mode") not in (
            "joint_state_and_pose", "wheel_head_only",
            "longitudinal_head_only"):
        raise ValueError(f"unrecognized objective for {name}")
    if set(metadata.get("validation_runs", [])) != expected_validation_runs:
        raise ValueError(f"{name} has a different frozen validation-run set")
    practice_runs = set(np.asarray(dataset["run_ids"]).astype(str)[
        np.asarray(dataset["splits"]).astype(str) == "unseen_practice"])
    overlap = practice_runs.intersection(metadata.get("training_runs", []))
    if overlap:
        raise ValueError(f"{name} trained on practice runs: {sorted(overlap)}")
    state = physical_state_from_dataset(
        dataset, wheel_state_source="raw_encoder").astype(np.float64)
    result = evaluate_model(
        model, dataset, state,
        np.asarray(metadata["state_scale"], dtype=np.float64),
        "cpu", windows=windows, batch_size=4,
        wheel_state_source="raw_encoder", horizon_steps=HORIZONS)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint_path),
        "training_objective_mode": metadata["training_objective_mode"],
        "training_run_count": len(metadata.get("training_runs", [])),
        "validation_runs": sorted(metadata.get("validation_runs", [])),
        "training_dataset_sha256": metadata.get("dataset_sha256"),
        "metrics": result,
    }


def persistence_metrics(dataset: dict[str, Any],
                       windows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray(dataset["encoder_raw_surface_mps"], dtype=np.float64)
    valid = np.asarray(dataset["encoder_raw_valid"], dtype=bool)
    by_horizon: dict[str, dict[str, list[float]]] = {
        name: {} for name in HORIZONS}
    for horizon_name, steps in HORIZONS.items():
        run_squared_errors: dict[str, list[float]] = {}
        run_count = {}
        for row in windows:
            start = int(row["start"])
            future = np.arange(start + 1, start + steps + 1)
            mask = valid[future]
            if not np.any(mask):
                raise ValueError("persistence horizon has no raw wheel labels")
            error = truth[future[mask]] - truth[start]
            run_id = str(dataset["run_ids"][row["run"]])
            run_squared_errors.setdefault(run_id, []).append(
                float(np.mean(error ** 2)))
            run_count[run_id] = run_count.get(run_id, 0) + 1
        by_horizon[horizon_name] = {
            "independent_runs": len(run_squared_errors),
            "windows_by_run": run_count,
            "pair_rmse_mps_by_run": {
                run_id: float(np.sqrt(np.mean(errors)))
                for run_id, errors in sorted(run_squared_errors.items())},
        }
        by_horizon[horizon_name]["macro_run_mean_pair_rmse_mps"] = float(
            np.mean(list(by_horizon[horizon_name][
                "pair_rmse_mps_by_run"].values())))
    return by_horizon


def wheel_pair_metrics(results: dict[str, Any],
                       persistence: dict[str, Any]) -> dict[str, Any]:
    report = {}
    for horizon in HORIZONS:
        report[horizon] = {}
        for model_name, model_result in results.items():
            metrics = model_result["metrics"]["horizons"][horizon]
            left = metrics["raw/rear_left_surface_speed_mps"]["per_run"]
            right = metrics["raw/rear_right_surface_speed_mps"]["per_run"]
            pair_by_run = {
                run_id: float(np.sqrt((left[run_id] ** 2 + right[run_id] ** 2) / 2))
                for run_id in sorted(set(left).intersection(right))}
            report[horizon][model_name] = {
                "pair_rmse_mps_by_run": pair_by_run,
                "macro_run_mean_pair_rmse_mps": float(
                    np.mean(list(pair_by_run.values()))),
            }
        baseline_by_run = persistence[horizon]["pair_rmse_mps_by_run"]
        report[horizon]["constant_initial_wheel_speed"] = {
            "pair_rmse_mps_by_run": baseline_by_run,
            "macro_run_mean_pair_rmse_mps": persistence[horizon][
                "macro_run_mean_pair_rmse_mps"],
        }
        baseline = report[horizon]["constant_initial_wheel_speed"][
            "macro_run_mean_pair_rmse_mps"]
        for model_name in results:
            model_value = report[horizon][model_name][
                "macro_run_mean_pair_rmse_mps"]
            report[horizon][model_name]["delta_vs_persistence_mps"] = (
                model_value - baseline)
    return report


def compare(dataset_path: Path, base_score_path: Path,
            output_path: Path, checkpoint_paths: dict[str, Path]
            ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not dataset_path.is_file() or not base_score_path.is_file():
        raise FileNotFoundError("practice dataset or frozen base score missing")
    dataset = _load_dataset(dataset_path)
    windows, base_score = load_windows(dataset, dataset_path, base_score_path)
    expected_validation_runs = set()
    for checkpoint_path in checkpoint_paths.values():
        torch, _, metadata = _load_model(checkpoint_path, "cpu")
        del torch
        expected_validation_runs.update(metadata.get("validation_runs", []))
    if not expected_validation_runs:
        raise ValueError("checkpoints have no recorded validation runs")

    results = {}
    for name, checkpoint_path in checkpoint_paths.items():
        results[name] = evaluate_one(
            name, checkpoint_path, dataset, windows,
            expected_validation_runs)
    persistence = persistence_metrics(dataset, windows)
    wheel_pairs = wheel_pair_metrics(results, persistence)

    summary = {}
    for horizon, _ in HORIZONS.items():
        summary[horizon] = {}
        for metric in METRICS:
            summary[horizon][metric] = {}
            for name, model_result in results.items():
                row = model_result["metrics"]["horizons"][horizon].get(metric)
                summary[horizon][metric][name] = (
                    row["macro_run_mean"] if row else None)
    report = {
        "schema_version": 1,
        "purpose": "transfer-only same-window comparison of frozen raw-wheel EDSSM teacher checkpoints",
        "dataset": str(dataset_path.relative_to(ROOT)),
        "dataset_sha256": sha256(dataset_path),
        "base_score_report": str(base_score_path.relative_to(ROOT)),
        "base_score_sha256": sha256(base_score_path),
        "base_checkpoint_sha256": base_score.get("checkpoint_sha256"),
        "split_scored": "unseen_practice",
        "test_and_final_test_used": False,
        "window_count": len(windows),
        "independent_run_count": len({row["run"] for row in windows}),
        "window_manifest": [
            {"run_id": str(dataset["run_ids"][row["run"]]),
             "sequence_id": row["sequence_id"],
             "start_frame": row["start"],
             "regimes": row["regimes"]}
            for row in windows],
        "protocol": {
            "all_candidates_use_identical_initial_frames_and_commands": True,
            "future_truth_or_future_sensor_feedback_used": False,
            "raw_encoder_labels_scored_only_when_valid": True,
            "checkpoint_selection_or_training_on_practice": False,
            "one_step_models_refit": False,
            "validation_run_set_identical": True,
        },
        "checkpoints": results,
        "wheel_pair_transfer": wheel_pairs,
        "macro_run_mean_summary": summary,
        "limitations": [
            "Only two independent unseen practice runs and four frozen starts are available; window-level count is not independent-run count.",
            "Practice windows cover ordinary, zero-throttle, and low-throttle motion, not the high-speed/high-steering envelope.",
            "The 25 ms raw encoder wheel rate is a measurement target, not direct tire angular velocity or tire force.",
            "This transfer test does not validate full-lap offline simulation or justify runtime integration.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-score", type=Path, default=DEFAULT_BASE_SCORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent", type=Path,
                        default=CHECKPOINTS["five_second_parent"])
    parser.add_argument("--ten-second-pose3", type=Path,
                        default=CHECKPOINTS["ten_second_pose3"])
    parser.add_argument("--wheel-head", type=Path,
                        default=CHECKPOINTS["wheel_head_only"])
    parser.add_argument("--distilled", type=Path,
                        default=CHECKPOINTS["wheel_head_body_distilled"])
    parser.add_argument("--longitudinal-distilled-fixed40", type=Path,
                        default=CHECKPOINTS["longitudinal_distilled_fixed40"])
    parser.add_argument("--longitudinal-distilled-all-motion", type=Path,
                        default=CHECKPOINTS["longitudinal_distilled_all_motion"])
    parser.add_argument("--ten-second", type=Path,
                        default=CHECKPOINTS["ten_second_plant_comparator"])
    args = parser.parse_args()
    paths = {
        "five_second_parent": args.parent.resolve(),
        "ten_second_pose3": args.ten_second_pose3.resolve(),
        "wheel_head_only": args.wheel_head.resolve(),
        "wheel_head_body_distilled": args.distilled.resolve(),
        "longitudinal_distilled_fixed40": (
            args.longitudinal_distilled_fixed40.resolve()),
        "longitudinal_distilled_all_motion": (
            args.longitudinal_distilled_all_motion.resolve()),
        "ten_second_plant_comparator": args.ten_second.resolve(),
    }
    report = compare(args.dataset.resolve(), args.base_score.resolve(),
                     args.output.resolve(), paths)
    print(json.dumps({
        "runs": report["independent_run_count"],
        "windows": report["window_count"],
        "summary": report["macro_run_mean_summary"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
