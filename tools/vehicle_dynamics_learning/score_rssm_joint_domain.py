#!/usr/bin/env python3
"""Score a frozen RSSM on held-out joint speed/steering operating cells.

This diagnostic prevents marginal speed and steering metrics from implying
support for combinations that were never observed together. Windows are
deterministic, sampled from whole validation runs, and are not treated as
independent replicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    STATE_NAMES,
    _pose_rollout,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import (
    COM_X_M,
    _rssm_model,
)


SPEED_EDGES_MPS = (0.0, 5.0, 7.0, 9.0, 12.000001)
STEERING_EDGES_RAD = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5240001)
HORIZON_STEPS = (10, 30, 80)


def _bin(value: float, edges: tuple[float, ...], suffix: str) -> str | None:
    index = int(np.searchsorted(edges, value, side="right") - 1)
    if value == edges[-1]:
        index = len(edges) - 2
    if index < 0 or index >= len(edges) - 1:
        return None
    return f"{edges[index]:g}-{edges[index + 1]:g}{suffix}"


def _validation_windows(data: dict[str, Any], targets: np.ndarray,
                        context_steps: int, future_steps: int,
                        stride_steps: int, max_per_run_cell: int
                        ) -> dict[tuple[str, str, int], list[tuple[int, int]]]:
    """Return (speed bin, steering bin, run) -> (context start, future start)."""
    by_cell: dict[tuple[str, str, int], list[tuple[int, int]]] = defaultdict(list)
    for sequence_index, (start_raw, end_raw) in enumerate(data["bounds"]):
        run_index = int(data["seq_run"][sequence_index])
        if str(data["splits"][run_index]) != "validation":
            continue
        start, end = int(start_raw), int(end_raw)
        latest = end - context_steps - future_steps
        for context_start in range(start, latest + 1, stride_steps):
            future_start = context_start + context_steps
            initial_index = future_start - 1
            state = targets[initial_index, :7]
            pose = data["simulator_pose_xyyaw"][initial_index]
            commands = data["frames"][future_start:future_start + future_steps,
                                       7:9]
            labels = targets[future_start:future_start + future_steps]
            if (not np.isfinite(state).all() or not np.isfinite(pose).all()
                    or not np.isfinite(commands).all()
                    or not np.isfinite(labels).all()):
                continue
            speed = float(np.hypot(state[0], state[1]))
            steering = abs(float(data["frames"][initial_index, 3]))
            speed_bin = _bin(speed, SPEED_EDGES_MPS, "mps")
            steering_bin = _bin(steering, STEERING_EDGES_RAD, "rad")
            if speed_bin is None or steering_bin is None:
                continue
            key = (speed_bin, steering_bin, run_index)
            by_cell[key].append((context_start, future_start))

    # Evenly spread the retained windows across each run/cell rather than
    # silently overweighting long holds. Adjacent windows are still correlated.
    result = {}
    for key, candidates in by_cell.items():
        if len(candidates) > max_per_run_cell:
            selected = np.linspace(
                0, len(candidates) - 1, max_per_run_cell, dtype=np.int64)
            candidates = [candidates[int(index)] for index in selected]
        result[key] = candidates
    return result


def _score_cell(torch, model, payload: dict[str, Any], data: dict[str, Any],
                targets: np.ndarray,
                windows: list[tuple[int, int]], context_steps: int,
                future_steps: int, device) -> dict[str, Any]:
    x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
    x_scale = np.asarray(payload["x_scale"], dtype=np.float32)
    y_mean = np.asarray(payload["y_mean"], dtype=np.float32)
    y_scale = np.asarray(payload["y_scale"], dtype=np.float32)
    horizon_errors: dict[int, list[np.ndarray]] = defaultdict(list)
    position_errors: dict[int, list[np.ndarray]] = defaultdict(list)
    for context_start, future_start in windows:
        context = ((data["frames"][context_start:future_start] - x_mean)
                   / x_scale)
        command = ((data["frames"][future_start:future_start + future_steps,
                                     7:9] - x_mean[7:9]) / x_scale[7:9])
        initial = targets[future_start - 1, :7]
        initial_normalized = (initial - y_mean[:7]) / y_scale[:7]
        with torch.no_grad():
            prediction = model(
                torch.as_tensor(context[None], dtype=torch.float32,
                                device=device),
                torch.as_tensor(command[None], dtype=torch.float32,
                                device=device),
                initial_state=torch.as_tensor(
                    initial_normalized[None], dtype=torch.float32,
                    device=device),
            )[0].cpu().numpy()
        prediction = prediction * y_scale + y_mean
        truth = targets[future_start:future_start + future_steps]
        state = prediction[:, :7]
        pose0 = data["simulator_pose_xyyaw"][future_start - 1]
        pose_prediction = _pose_rollout(state, initial, pose0)
        pose_truth = data["simulator_pose_xyyaw"][
            future_start:future_start + future_steps]
        for horizon in HORIZON_STEPS:
            if horizon > future_steps:
                continue
            horizon_errors[horizon].append(state[horizon - 1] - truth[horizon - 1, :7])
            position_errors[horizon].append(
                pose_prediction[horizon - 1, :2] - pose_truth[horizon - 1, :2])

    report: dict[str, Any] = {"window_count": len(windows), "horizons": {}}
    for horizon in HORIZON_STEPS:
        if horizon > future_steps or not horizon_errors[horizon]:
            continue
        state_error = np.stack(horizon_errors[horizon])
        xy_error = np.stack(position_errors[horizon])
        report["horizons"][f"{horizon * 0.025:g}s"] = {
            "state_rmse": {
                name: float(value) for name, value in zip(
                    STATE_NAMES,
                    np.sqrt(np.mean(state_error ** 2, axis=0)).tolist())
            },
            "position_xy_rmse_m": np.sqrt(
                np.mean(xy_error ** 2, axis=0)).tolist(),
            "position_radial_rmse_m": float(np.sqrt(
                np.mean(np.sum(xy_error ** 2, axis=1)))),
        }
    return report


def score(checkpoint_path: Path, dataset_path: Path, output_path: Path,
          *, device_name: str = "cuda", stride_steps: int = 20,
          max_per_run_cell: int = 24) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if stride_steps < 1 or max_per_run_cell < 1:
        raise ValueError("window stride and per-run cell limit must be positive")
    torch, nn = _torch()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    data = _load_dataset(dataset_path)
    if metadata["feature_names"] != data["feature_names"]:
        raise ValueError("checkpoint and dataset feature layouts differ")
    if int(data["schema_version"]) < 8:
        raise ValueError("joint race-domain scoring requires schema 8")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    targets = _targets(data)
    context_steps = int(metadata["context_steps"])
    future_steps = int(metadata["rollout_steps"])
    if future_steps < max(HORIZON_STEPS):
        raise ValueError("checkpoint rollout is shorter than the 2 s primary horizon")
    training_runs = set(map(str, metadata["training_runs"]))
    model_type = _rssm_model(
        torch, nn, int(metadata["hidden_size"]), int(metadata["latent_size"]),
        payload["x_mean"], payload["x_scale"], payload["y_mean"],
        payload["y_scale"])
    model = model_type().to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    windows = _validation_windows(
        data, targets, context_steps, future_steps,
        stride_steps, max_per_run_cell)
    evaluated_run_ids = {
        str(data["run_ids"][run_index]) for _, _, run_index in windows
    }
    overlap = sorted(evaluated_run_ids & training_runs)
    if overlap:
        raise ValueError(f"refusing to score training runs: {overlap}")
    per_cell: dict[str, dict[str, Any]] = {}
    for speed_index in range(len(SPEED_EDGES_MPS) - 1):
        speed_bin = (f"{SPEED_EDGES_MPS[speed_index]:g}-"
                     f"{SPEED_EDGES_MPS[speed_index + 1]:g}mps")
        for steering_index in range(len(STEERING_EDGES_RAD) - 1):
            steering_bin = (f"{STEERING_EDGES_RAD[steering_index]:g}-"
                            f"{STEERING_EDGES_RAD[steering_index + 1]:g}rad")
            run_results = {}
            for (local_speed, local_steer, run_index), local_windows in windows.items():
                if local_speed != speed_bin or local_steer != steering_bin:
                    continue
                run_id = str(data["run_ids"][run_index])
                run_results[run_id] = _score_cell(
                    torch, model, payload, data, targets, local_windows,
                    context_steps, future_steps, device)
            if not run_results:
                continue
            horizons = {}
            for horizon_key in ("0.25s", "0.75s", "2s"):
                contributing = [row["horizons"][horizon_key]
                                for row in run_results.values()
                                if horizon_key in row["horizons"]]
                if not contributing:
                    continue
                horizons[horizon_key] = {
                    "independent_run_count": len(contributing),
                    "macro_run_position_radial_rmse_m": float(np.mean([
                        row["position_radial_rmse_m"] for row in contributing])),
                    "per_run": {
                        run_id: row["horizons"][horizon_key]
                        for run_id, row in sorted(run_results.items())
                        if horizon_key in row["horizons"]
                    },
                }
            per_cell[f"{speed_bin}|{steering_bin}"] = {
                "speed_bin": speed_bin,
                "absolute_steering_bin": steering_bin,
                "independent_run_count": len(run_results),
                "total_windows": sum(row["window_count"]
                                      for row in run_results.values()),
                "supported_for_run_level_inference": len(run_results) >= 2,
                "macro_horizons": horizons,
                "per_run": run_results,
            }

    dataset_digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "purpose": "diagnose joint speed/steering support; not checkpoint selection",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": dataset_digest,
        "split": "validation",
        "training_run_overlap": [],
        "validation_run_ids": sorted(evaluated_run_ids),
        "independent_validation_run_count": len(evaluated_run_ids),
        "future_truth_or_sensor_inputs": False,
        "rollout_contract": "prior-mean recursion from history and future command only",
        "context_seconds": context_steps * 0.025,
        "rollout_seconds": future_steps * 0.025,
        "deterministic_window_stride_seconds": stride_steps * 0.025,
        "max_windows_per_run_cell": max_per_run_cell,
        "speed_edges_mps": list(SPEED_EDGES_MPS),
        "absolute_steering_edges_rad": list(STEERING_EDGES_RAD),
        "window_count_is_not_independent_run_count": True,
        "cells": per_cell,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride-steps", type=int, default=20)
    parser.add_argument("--max-windows-per-run-cell", type=int, default=24)
    args = parser.parse_args()
    result = score(
        args.checkpoint, args.dataset, args.output,
        device_name=args.device, stride_steps=args.stride_steps,
        max_per_run_cell=args.max_windows_per_run_cell)
    populated = [cell for cell in result["cells"].values()]
    print(json.dumps({
        "validation_runs": result["independent_validation_run_count"],
        "populated_joint_cells": len(populated),
        "cells_with_multiple_runs": sum(
            cell["independent_run_count"] >= 2 for cell in populated),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
