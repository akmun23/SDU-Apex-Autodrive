#!/usr/bin/env python3
"""Paired whole-run comparison of frozen RSSM capacity candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    DT_S,
    _eligible_sequences,
    _pose_rollout,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model


def compare(checkpoints: list[Path], dataset_path: Path, output_path: Path,
            *, split: str = "test", device_name: str = "cpu",
            max_windows_per_run: int = 32, seed: int = 20261022,
            bootstrap_replicates: int = 10_000) -> dict:
    if len(checkpoints) < 2:
        raise ValueError("provide at least two frozen RSSM checkpoints")
    torch, nn = _torch()
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = _load_dataset(dataset_path)
    targets = _targets(data)
    payloads = [torch.load(path, map_location="cpu", weights_only=False)
                for path in checkpoints]
    reference = payloads[0]["metadata"]
    context = int(reference["context_steps"])
    horizon = int(reference["rollout_steps"])
    if any(int(item["metadata"]["context_steps"]) != context
           or int(item["metadata"]["rollout_steps"]) != horizon
           for item in payloads):
        raise ValueError("paired capacity comparison requires equal context/horizon")
    if any(item["metadata"]["feature_names"] != data["feature_names"]
           for item in payloads):
        raise ValueError("checkpoint feature layouts differ from dataset")
    reference_scale = np.asarray(payloads[0]["y_scale"][:3], dtype=np.float64)
    reference_mean = np.asarray(payloads[0]["y_mean"][:3], dtype=np.float64)
    for item in payloads[1:]:
        if (not np.allclose(item["y_scale"][:3], reference_scale, rtol=0, atol=1e-7)
                or not np.allclose(item["y_mean"][:3], reference_mean, rtol=0, atol=1e-7)):
            raise ValueError("candidate normalization differs; use raw error channels")

    group_sets = [_eligible_sequences(
        data, targets, split, context, horizon) for _ in payloads]
    common_runs = set.intersection(*[set(groups) for groups in group_sets])
    if not common_runs:
        raise ValueError(f"no common whole-run {split} runs")
    for item in payloads:
        overlap = common_runs & set(item["metadata"]["training_runs"])
        if overlap:
            raise ValueError(f"refusing to score training runs: {sorted(overlap)}")
    common_sequences = {}
    for run in sorted(common_runs):
        sequence_sets = [set(group[run]) for group in group_sets]
        common = sorted(set.intersection(*sequence_sets))
        if common:
            common_sequences[run] = common
    rng = np.random.default_rng(seed)
    windows = {}
    for run, sequences in common_sequences.items():
        starts = []
        for start, end in sequences:
            low = start + context - 1
            high = end - horizon - 1
            if high >= low:
                starts.extend(range(low, high + 1))
        if len(starts) > max_windows_per_run:
            starts = sorted(int(x) for x in rng.choice(
                starts, max_windows_per_run, replace=False))
        if starts:
            windows[run] = starts
    if not windows:
        raise ValueError("no common paired future windows")

    models = []
    for item in payloads:
        metadata = item["metadata"]
        model_type = _rssm_model(
            torch, nn, int(metadata["hidden_size"]),
            int(metadata["latent_size"]), item["x_mean"], item["x_scale"],
            item["y_mean"], item["y_scale"])
        model = model_type().to(device)
        model.load_state_dict(item["state_dict"])
        model.eval()
        models.append(model)

    horizon_steps = [step for step in (10, 30, 40, horizon)
                     if step <= horizon]
    errors_by_model: list[dict[str, dict[int, list[float]]]] = []
    per_run_by_model = []
    for item, model in zip(payloads, models):
        run_errors: dict[str, dict[int, list[float]]] = {}
        per_run = {}
        with torch.no_grad():
            for run, starts in windows.items():
                by_step = {step: [] for step in horizon_steps}
                pose_by_step = {step: [] for step in horizon_steps}
                for future_start in starts:
                    context_rows = data["frames"][future_start - context:
                                                   future_start]
                    commands = data["frames"][future_start:
                                              future_start + horizon, 7:9]
                    context_norm = ((context_rows - item["x_mean"])
                                    / item["x_scale"])
                    commands_norm = ((commands - item["x_mean"][7:9])
                                     / item["x_scale"][7:9])
                    prediction = model(
                        torch.as_tensor(context_norm[None], dtype=torch.float32,
                                        device=device),
                        torch.as_tensor(commands_norm[None], dtype=torch.float32,
                                        device=device))[0].cpu().numpy()
                    prediction = prediction * item["y_scale"] + item["y_mean"]
                    truth = targets[future_start:future_start + horizon]
                    init_state = targets[future_start - 1, :3]
                    init_pose = data["simulator_pose_xyyaw"][future_start - 1]
                    true_pose = data["simulator_pose_xyyaw"][
                        future_start:future_start + horizon]
                    pose_valid = (np.isfinite(init_pose).all()
                                  and np.isfinite(true_pose).all())
                    pred_pose = (_pose_rollout(prediction, init_state, init_pose)
                                 if pose_valid else None)
                    for step in horizon_steps:
                        error = (prediction[step - 1, :3]
                                 - truth[step - 1, :3]) / reference_scale
                        by_step[step].append(float(np.sqrt(np.mean(error ** 2))))
                        if pose_valid:
                            pose_error = pred_pose[step - 1] - true_pose[step - 1]
                            pose_error[2] = np.arctan2(np.sin(pose_error[2]),
                                                       np.cos(pose_error[2]))
                            pose_by_step[step].append(pose_error)
                run_key = str(data["run_ids"][run])
                run_errors[run_key] = by_step
                per_run[run_key] = {
                    "window_count": len(starts),
                    "horizons": {
                        f"{step * DT_S:g}s": {
                            "common_scale_normalized_body_rmse": float(np.mean(by_step[step])),
                            "window_count": len(by_step[step]),
                            **({
                                "position_xy_rmse_m": np.sqrt(np.mean(
                                    np.asarray(pose_by_step[step])[:, :2] ** 2,
                                    axis=0)).tolist(),
                                "heading_rmse_rad": float(np.sqrt(np.mean(
                                    np.asarray(pose_by_step[step])[:, 2] ** 2))),
                            } if pose_by_step[step] else {}),
                        } for step in horizon_steps
                    },
                }
        errors_by_model.append(run_errors)
        per_run_by_model.append(per_run)

    model_names = [
        f"h{int(item['metadata']['hidden_size'])}_z{int(item['metadata']['latent_size'])}"
        for item in payloads]
    score_matrix = np.asarray([
        [np.mean([np.mean(row[step]) for row in model.values()])
         for step in horizon_steps]
        for model in errors_by_model], dtype=np.float64)
    rng_boot = np.random.default_rng(seed + 1)
    common_run_names = sorted(next(iter(errors_by_model)).keys())
    paired = {}
    for candidate_index in range(1, len(payloads)):
        pair_name = model_names[candidate_index]
        comparison = {}
        for step in horizon_steps:
            baseline_run = np.asarray([
                np.mean(errors_by_model[0][run][step])
                for run in common_run_names])
            candidate_run = np.asarray([
                np.mean(errors_by_model[candidate_index][run][step])
                for run in common_run_names])
            diff = candidate_run - baseline_run
            bootstrap = np.empty(bootstrap_replicates, dtype=np.float64)
            for index in range(bootstrap_replicates):
                sample = rng_boot.integers(0, len(diff), len(diff))
                bootstrap[index] = np.mean(diff[sample])
            comparison[f"{step * DT_S:g}s"] = {
                "baseline_macro_run_rmse": float(np.mean(baseline_run)),
                "candidate_macro_run_rmse": float(np.mean(candidate_run)),
                "candidate_minus_baseline": float(np.mean(diff)),
                "relative_change_percent": float(
                    100.0 * (np.mean(candidate_run) / np.mean(baseline_run) - 1.0)),
                "run_cluster_bootstrap_95pct_interval": np.quantile(
                    bootstrap, (0.025, 0.975)).tolist(),
                "independent_run_count": len(diff),
                "per_run": dict(zip(common_run_names, diff.tolist())),
            }
        paired[pair_name] = comparison

    report = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "split": split,
        "split_status": (
            "checkpoint-selection validation; not blind confirmation"
            if split == "validation" else
            "held-out split diagnostic; historical consumption must be checked"),
        "context_seconds": context * DT_S,
        "prediction_seconds": horizon * DT_S,
        "common_eligible_runs": common_run_names,
        "window_counts_by_run": {str(data["run_ids"][run]): len(starts)
                                  for run, starts in windows.items()},
        "shared_train_normalizer": reference_scale.tolist(),
        "candidate_checkpoints": [str(path.resolve()) for path in checkpoints],
        "macro_run_scores_by_horizon": {
            f"{step * DT_S:g}s": {
                name: float(score_matrix[index, step_index])
                for index, name in enumerate(model_names)
            } for step_index, step in enumerate(horizon_steps)
        },
        "paired_run_cluster_bootstrap": paired,
        "per_model_per_run": dict(zip(model_names, per_run_by_model)),
        "windows_are_not_independent_runs": True,
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-windows-per-run", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    report = compare(args.checkpoints, args.dataset, args.output,
                     split=args.split, device_name=args.device,
                     max_windows_per_run=args.max_windows_per_run,
                     seed=args.seed)
    print(json.dumps({
        "runs": report["common_eligible_runs"],
        "scores": report["macro_run_scores_by_horizon"],
        "paired": report["paired_run_cluster_bootstrap"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
