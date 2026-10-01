#!/usr/bin/env python3
"""Score prior-only RSSM rollouts on the fixed high-speed straight holds.

Each hold contributes its first 4 s as observed context and the remainder as
future command-only rollout. Future sensor/physics channels are labels only.
This is a consumed diagnostic benchmark, not fresh blind confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    COM_X_M,
    DT_S,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model


HORIZONS_S = (0.25, 0.75, 1.0, 2.0, 4.0, 5.0)


def _pose_rollout(states: np.ndarray, initial_state: np.ndarray,
                  initial_pose: np.ndarray) -> np.ndarray:
    pose = np.empty((len(states), 3), dtype=np.float64)
    x, y, yaw = map(float, initial_pose)
    previous = np.asarray(initial_state[:3], dtype=np.float64)
    for index, following in enumerate(states[:, :3]):
        u = 0.5 * (previous[0] + following[0])
        r = 0.5 * (previous[2] + following[2])
        v_rear = 0.5 * ((previous[1] - COM_X_M * previous[2])
                        + (following[1] - COM_X_M * following[2]))
        mid_yaw = yaw + 0.5 * r * DT_S
        x += (u * math.cos(mid_yaw) - v_rear * math.sin(mid_yaw)) * DT_S
        y += (u * math.sin(mid_yaw) + v_rear * math.cos(mid_yaw)) * DT_S
        yaw += r * DT_S
        pose[index] = (x, y, yaw)
        previous = following
    return pose


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def score(checkpoints: list[Path], dataset_path: Path, output_path: Path,
          *, context_seconds: float = 4.0, device_name: str = "cuda"
          ) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch, nn = _torch()
    torch.set_num_threads(1)
    data = _load_dataset(dataset_path)
    targets = _targets(data)
    context_steps = round(context_seconds / DT_S)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "split_policy": "fixed 4-second initial context; future commands only",
        "context_seconds": context_steps * DT_S,
        "device": str(device),
        "diagnostic_status": "consumed benchmark; not fresh blind confirmation",
        "models": [],
    }

    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata["feature_names"] != data["feature_names"]:
            raise ValueError(f"feature layout mismatch: {checkpoint}")
        if int(metadata["context_steps"]) != context_steps:
            raise ValueError(
                f"checkpoint requires {metadata['context_steps']} context rows, "
                f"not {context_steps}: {checkpoint}")
        Model = _rssm_model(
            torch, nn, int(metadata["hidden_size"]),
            int(metadata["latent_size"]), payload["x_mean"],
            payload["x_scale"], payload["y_mean"], payload["y_scale"])
        model = Model().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        per_run = []

        with torch.no_grad():
            for seq_index, (start_value, end_value) in enumerate(data["bounds"]):
                start, end = int(start_value), int(end_value)
                if end - start <= context_steps + 1:
                    raise ValueError(f"run {seq_index} is too short for context/rollout")
                context_end = start + context_steps
                initial_target_index = context_end - 1
                future_end = end - 1
                context = data["frames"][start:context_end]
                commands = data["frames"][context_end:future_end, 7:9]
                truth = targets[context_end:future_end]
                if not np.isfinite(context).all() or not np.isfinite(commands).all():
                    raise ValueError(f"non-finite input sequence {seq_index}")
                if not np.isfinite(truth).all():
                    raise ValueError(f"non-finite label sequence {seq_index}")
                intervals = data["dt_s"][context_end:future_end]
                if len(intervals) and not np.allclose(
                        intervals, DT_S, atol=1e-5, rtol=0.0):
                    raise ValueError(f"non-25-ms sequence in run {seq_index}")
                context_norm = ((context - payload["x_mean"])
                                / payload["x_scale"])
                command_norm = ((commands - payload["x_mean"][7:9])
                                / payload["x_scale"][7:9])
                initial_state = targets[initial_target_index, :7]
                initial_state_norm = (
                    initial_state - payload["y_mean"][:7]
                ) / payload["y_scale"][:7]
                predicted_norm = model(
                    torch.as_tensor(context_norm[None], dtype=torch.float32,
                                    device=device),
                    torch.as_tensor(command_norm[None], dtype=torch.float32,
                                    device=device),
                    initial_state=torch.as_tensor(
                        initial_state_norm[None], dtype=torch.float32,
                        device=device),
                ).cpu().numpy()[0]
                predicted = predicted_norm * payload["y_scale"] + payload["y_mean"]
                predicted_state = predicted[:, :7]
                count = len(predicted_state)
                if count != len(truth):
                    raise AssertionError("rollout length does not match labels")
                true_pose = data["simulator_pose_xyyaw"][
                    context_end:future_end].astype(
                    np.float64, copy=False)
                initial_pose = data["simulator_pose_xyyaw"][initial_target_index]
                predicted_pose = _pose_rollout(
                    predicted_state, initial_state, initial_pose)
                true_relative_pose = true_pose.copy()
                true_relative_pose[:, :2] -= initial_pose[:2]
                predicted_relative_pose = predicted_pose.copy()
                predicted_relative_pose[:, :2] -= initial_pose[:2]
                true_relative_pose[:, 2] -= initial_pose[2]
                predicted_relative_pose[:, 2] -= initial_pose[2]
                yaw_error = (predicted_relative_pose[:, 2]
                             - true_relative_pose[:, 2] + np.pi) % (2 * np.pi) - np.pi

                horizon_scores = {}
                for horizon in HORIZONS_S:
                    step = round(horizon / DT_S)
                    if step > count:
                        continue
                    sl = slice(0, step)
                    horizon_scores[f"{horizon:g}s"] = {
                        "u_rmse_mps": _rmse(predicted_state[sl, 0], truth[sl, 0]),
                        "v_rmse_mps": _rmse(predicted_state[sl, 1], truth[sl, 1]),
                        "yaw_rate_rmse_rps": _rmse(
                            predicted_state[sl, 2], truth[sl, 2]),
                        "rear_left_wheel_speed_rmse_mps": _rmse(
                            predicted_state[sl, 5], truth[sl, 5]),
                        "rear_right_wheel_speed_rmse_mps": _rmse(
                            predicted_state[sl, 6], truth[sl, 6]),
                        "integrated_xy_rmse_m": float(np.sqrt(np.mean(
                            (predicted_relative_pose[sl, :2]
                             - true_relative_pose[sl, :2]) ** 2))),
                        "integrated_yaw_rmse_rad": _rmse(
                            yaw_error[sl], np.zeros(step)),
                    }
                per_run.append({
                    "run_id": str(data["run_ids"][int(data["seq_run"][seq_index])]),
                    "context_speed_mps": float(np.hypot(
                        initial_state[0], initial_state[1] - COM_X_M * initial_state[2])),
                    "rollout_seconds": count * DT_S,
                    "future_steps": count,
                    "command_range": {
                        "steering_rad": [float(commands[:, 0].min()),
                                         float(commands[:, 0].max())],
                        "throttle_norm": [float(commands[:, 1].min()),
                                          float(commands[:, 1].max())],
                    },
                    "horizons": horizon_scores,
                })

        model_summary = {}
        for horizon in HORIZONS_S:
            key = f"{horizon:g}s"
            values = [run["horizons"][key] for run in per_run
                      if key in run["horizons"]]
            if values:
                model_summary[key] = {
                    name: float(np.mean([row[name] for row in values]))
                    for name in values[0]
                }
        report["models"].append({
            "checkpoint": str(checkpoint.resolve()),
            "architecture": metadata["architecture"],
            "hidden_size": int(metadata["hidden_size"]),
            "latent_size": int(metadata["latent_size"]),
            "independent_run_count": len(per_run),
            "macro_run_mean": model_summary,
            "per_run": per_run,
            "causal_inputs": "past observed context and future steering/throttle commands only",
            "uses_future_truth_or_sensors": False,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--context-seconds", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.checkpoints, args.dataset, args.output,
                   context_seconds=args.context_seconds,
                   device_name=args.device)
    print(json.dumps({
        "models": [{"checkpoint": row["checkpoint"],
                    "macro_run_mean": row["macro_run_mean"]}
                   for row in result["models"]],
        "runs": len(result["models"][0]["per_run"]) if result["models"] else 0,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
