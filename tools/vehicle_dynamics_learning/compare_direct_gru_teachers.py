#!/usr/bin/env python3
"""Paired short-horizon comparison of the frozen GRU and direct ceiling."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    COM_X_M,
    DT_S,
    _eligible_sequences,
    _pose_rollout,
    _sample_windows,
    _targets,
    _torch_model,
)
from tools.vehicle_dynamics_learning.train_nssm import (
    _load_dataset,
    _model_type,
    _rollout,
    _torch,
)


def _cluster_bootstrap(values: np.ndarray, seed: int,
                       samples: int = 20_000) -> list[float]:
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(values), size=len(values))
        bootstrap[index] = np.mean(values[selected])
    return [float(value) for value in np.quantile(bootstrap, [0.025, 0.5, 0.975])]


def compare(direct_checkpoint: Path, gru_checkpoint: Path,
            dataset_path: Path, output_path: Path, *, split: str = "test",
            device_name: str = "cuda", max_windows_per_run: int = 32,
            seed: int = 20261005) -> dict:
    torch, nn = _torch()
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    direct_payload = torch.load(direct_checkpoint, map_location="cpu",
                                weights_only=False)
    gru_payload = torch.load(gru_checkpoint, map_location="cpu",
                             weights_only=False)
    direct_meta = direct_payload["metadata"]
    gru_meta = gru_payload["metadata"]
    data = _load_dataset(dataset_path)
    if (direct_meta["feature_names"] != data["feature_names"]
            or gru_meta["feature_names"] != data["feature_names"]):
        raise ValueError("model and comparison dataset feature layouts differ")
    targets = _targets(data)
    context_steps = int(direct_meta["context_steps"])
    future_steps = int(direct_meta["future_steps"])
    groups = _eligible_sequences(
        data, targets, split, context_steps, future_steps)
    test_runs = {str(data["run_ids"][run]) for run in groups}
    trained_runs = (set(direct_meta["training_runs"])
                    | set(gru_meta["training_runs"]))
    overlap = sorted(test_runs & trained_runs)
    if overlap:
        raise ValueError(f"comparison split overlaps model training: {overlap}")
    if not groups:
        raise ValueError(f"no eligible {split!r} runs")

    direct_model_type = _torch_model(
        torch, nn, context_steps, future_steps, int(direct_meta["width"]),
        int(direct_meta["layer_count"]), int(direct_meta["head_count"]), 0.05)
    direct_model = direct_model_type().to(device)
    direct_model.load_state_dict(direct_payload["state_dict"])
    direct_model.eval()
    gru_model_type = _model_type(
        torch, nn, int(gru_meta["hidden_size"]),
        str(gru_meta.get("architecture", "gru")),
        int(gru_meta.get("expert_count", 1)), int(gru_meta["history_steps"]),
        len(data["feature_names"]), gru_payload["feature_mean"],
        gru_payload["feature_scale"], gru_meta.get("body_acceleration_mean"),
        gru_meta.get("body_acceleration_scale"),
        gru_meta.get("integration_method", "euler"),
        gru_meta.get("rear_axle_to_com_x_m", 0.0))
    gru_model = gru_model_type().to(device)
    gru_model.load_state_dict(gru_payload["state_dict"])
    gru_model.eval()

    run_results = {}
    horizon_steps = sorted(set(step for step in (
        1, 5, 10, 20, 30, 40, 80, future_steps) if step <= future_steps))
    for run_id, sequences in sorted(groups.items()):
        local_windows = _sample_windows(
            data, {run_id: sequences}, max_windows_per_run, context_steps,
            future_steps, np.random.default_rng(seed + int(run_id)))
        per_model: dict[str, dict[int, list[dict]]] = {
            "direct": defaultdict(list), "gru": defaultdict(list)}
        with torch.no_grad():
            for start, _ in local_windows:
                future_start = start + context_steps
                future_end = future_start + future_steps
                raw_context = data["frames"][start:future_start]
                raw_future = data["frames"][future_start:future_end]
                direct_context = ((raw_context - direct_payload["x_mean"])
                                  / direct_payload["x_scale"])
                direct_commands = ((raw_future[:, 7:9]
                                    - direct_payload["x_mean"][7:9])
                                   / direct_payload["x_scale"][7:9])
                direct_prediction = direct_model(
                    torch.as_tensor(direct_context[None], dtype=torch.float32,
                                    device=device),
                    torch.as_tensor(direct_commands[None], dtype=torch.float32,
                                    device=device))[0].cpu().numpy()
                direct_physical = (direct_prediction
                                   * direct_payload["y_scale"]
                                   + direct_payload["y_mean"])

                gru_history_steps = int(gru_meta["history_steps"])
                gru_context = raw_context[-gru_history_steps:]
                gru_future = ((raw_future - gru_payload["feature_mean"])
                              / gru_payload["feature_scale"])
                gru_history = ((gru_context - gru_payload["feature_mean"])
                               / gru_payload["feature_scale"])
                gru_dt = data["dt_s"][future_start:future_end]
                gru_prediction = _rollout(
                    gru_model,
                    torch.as_tensor(gru_history[None], dtype=torch.float32,
                                    device=device),
                    torch.as_tensor(gru_future[None], dtype=torch.float32,
                                    device=device),
                    torch.as_tensor(gru_dt[None], dtype=torch.float32,
                                    device=device), gru_history_steps)[0].cpu().numpy()
                gru_rear = (gru_prediction * gru_payload["feature_scale"][:7]
                            + gru_payload["feature_mean"][:7])
                gru_physical = gru_rear.copy()
                gru_physical[:, 1] += COM_X_M * gru_physical[:, 2]

                initial_state = targets[future_start - 1, :7]
                initial_pose = data["simulator_pose_xyyaw"][future_start - 1]
                true_pose = data["simulator_pose_xyyaw"][
                    future_start:future_end]
                direct_pose = _pose_rollout(
                    direct_physical, initial_state, initial_pose)
                gru_pose = _pose_rollout(
                    gru_physical, initial_state, initial_pose)
                true_states = targets[future_start:future_end, :3]
                truth_pose_finite = (np.isfinite(initial_pose).all()
                                     and np.isfinite(true_pose).all())
                for horizon in horizon_steps:
                    truth_state = true_states[horizon - 1]
                    state_scale = direct_payload["y_scale"][:3]
                    for name, predicted, pose in (
                            ("direct", direct_physical, direct_pose),
                            ("gru", gru_physical, gru_pose)):
                        state_error = predicted[horizon - 1, :3] - truth_state
                        row = {
                            "normalized_body_state_rmse": float(np.sqrt(
                                np.mean((state_error / state_scale) ** 2))),
                            "state_error": state_error.tolist(),
                        }
                        if truth_pose_finite:
                            pose_error = pose[horizon - 1] - true_pose[horizon - 1]
                            row["position_xy_error_m"] = pose_error[:2].tolist()
                            row["heading_error_rad"] = float(pose_error[2])
                        per_model[name][horizon].append(row)
        horizon_results = {}
        for horizon in horizon_steps:
            pair = {}
            for name in ("direct", "gru"):
                rows = per_model[name][horizon]
                errors = np.asarray([row["state_error"] for row in rows])
                pair[name] = {
                    "normalized_body_state_rmse": float(np.sqrt(np.mean(
                        [row["normalized_body_state_rmse"]**2 for row in rows]))),
                    "state_rmse": np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                    "window_count": len(rows),
                }
                if rows and "position_xy_error_m" in rows[0]:
                    pose_errors = np.asarray(
                        [row["position_xy_error_m"] for row in rows])
                    pair[name]["position_xy_rmse_m"] = np.sqrt(
                        np.mean(pose_errors**2, axis=0)).tolist()
                    pair[name]["heading_rmse_rad"] = float(np.sqrt(
                        np.mean([row["heading_error_rad"]**2 for row in rows])))
            pair["paired_run_difference_gru_minus_direct"] = (
                pair["gru"]["normalized_body_state_rmse"]
                - pair["direct"]["normalized_body_state_rmse"])
            horizon_results[f"{horizon * DT_S:g}s"] = pair
        run_results[str(data["run_ids"][run_id])] = horizon_results

    summary = {}
    for horizon in horizon_steps:
        key = f"{horizon * DT_S:g}s"
        differences = np.asarray([
            values[key]["paired_run_difference_gru_minus_direct"]
            for values in run_results.values()])
        summary[key] = {
            "independent_run_count": int(len(differences)),
            "mean_paired_run_difference_gru_minus_direct": float(
                np.mean(differences)),
            "run_cluster_bootstrap_2p5_50_97p5": _cluster_bootstrap(
                differences, seed + horizon),
            "positive_means_gru_error_is_higher": True,
        }
    report = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "split": split,
        "direct_checkpoint": str(direct_checkpoint.resolve()),
        "gru_checkpoint": str(gru_checkpoint.resolve()),
        "run_count": len(run_results),
        "context_seconds_direct": context_steps * DT_S,
        "context_seconds_gru": int(gru_meta["history_steps"]) * DT_S,
        "future_seconds": future_steps * DT_S,
        "shared_future_command_trace": True,
        "future_truth_or_sensors_used_as_inputs": False,
        "windows_are_clustered_by_run": True,
        "per_run": run_results,
        "run_cluster_summary": summary,
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direct_checkpoint", type=Path)
    parser.add_argument("gru_checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-windows-per-run", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    report = compare(
        args.direct_checkpoint, args.gru_checkpoint, args.dataset,
        args.output, split=args.split, device_name=args.device,
        max_windows_per_run=args.max_windows_per_run, seed=args.seed)
    print(json.dumps({
        "runs": report["run_count"],
        "paired_differences": report["run_cluster_summary"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
