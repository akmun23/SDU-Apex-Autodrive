#!/usr/bin/env python3
"""Score a command-driven four-wheel grey-box on untouched capture sequences.

The simulator truth is used only to initialize the physical state at the end
of the declared context and to score later predictions. After initialization,
the rollout consumes steering/throttle commands only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.four_wheel_greybox import (
    DT_S,
    WHEEL_RADIUS_M,
    _initial_state,
    _integrate_pose,
    _physical_model,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


HISTORY_STEPS = 80
HORIZONS_S = (0.25, 0.5, 0.75, 1.0, 2.0, 5.0)
BODY_NAMES = ("u_com_mps", "v_com_mps", "yaw_rate_rps")
STATE_NAMES = BODY_NAMES + (
    "rear_left_surface_mps", "rear_right_surface_mps",
    "steering_feedback_rad", "throttle_feedback_norm",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _rmse_bias(error: np.ndarray) -> dict[str, list[float]]:
    return {
        "rmse": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "bias": np.mean(error, axis=0).tolist(),
        "absolute_p95": np.quantile(np.abs(error), 0.95, axis=0).tolist(),
    }


def _sequence_rows(data: dict[str, Any], run_index: int,
                   min_future_steps: int) -> list[tuple[int, int, str]]:
    indexes = np.flatnonzero(data["seq_run"] == run_index)
    indexes = sorted(indexes, key=lambda item: int(data["bounds"][item, 0]))
    rows: list[tuple[int, int, str, int]] = []
    for sequence_index in indexes:
        start, end = map(int, data["bounds"][sequence_index])
        if end - start < HISTORY_STEPS + min_future_steps:
            continue
        condition_id = int(data["sequence_condition_id"][sequence_index])
        condition = str(data["condition_labels"][condition_id])
        reset_id = int(data["sequence_reset_index"][sequence_index])
        if rows:
            prev_start, prev_end, prev_condition, prev_reset_id = rows[-1]
            packet_contiguous = (
                prev_end == start
                and int(data["packet_sequence"][start])
                    == int(data["packet_sequence"][prev_end - 1]) + 1)
            if (condition == prev_condition and reset_id == prev_reset_id
                    and packet_contiguous):
                rows[-1] = (prev_start, end, condition, reset_id)
                continue
        rows.append((start, end, condition, reset_id))
    return [(start, end, label) for start, end, label, _ in rows]


def _score_sequence(torch, model, data: dict[str, Any], physical_state: np.ndarray,
                    start: int, end: int,
                    label: str, run_id: str, device) -> dict[str, Any]:
    context_end = start + HISTORY_STEPS - 1
    if end - context_end - 1 < round(0.25 / DT_S):
        raise ValueError(f"{run_id}/{label}: sequence too short after context")
    initial = _initial_state(data, context_end, physical_state, start)
    # Training pairs state[k] + command[k] with state[k + 1]. Keep that
    # alignment here; the actuator-delay state already models the measured
    # one-packet command/feedback latency. Starting at k + 1 would shift the
    # command stream an additional sample relative to the fitted model.
    commands_np = data["frames"][context_end:end - 1, 7:9]
    commands = torch.as_tensor(commands_np[None], dtype=torch.float32,
                               device=device)
    initial_tensor = torch.as_tensor(initial[None], dtype=torch.float32,
                                     device=device)
    with torch.no_grad():
        predicted_internal, _ = model.rollout(initial_tensor, commands)
    internal = predicted_internal[0].detach().cpu().numpy()
    predicted = np.column_stack((
        internal[:, :3], internal[:, 5:7] * WHEEL_RADIUS_M,
        internal[:, 17:19],
    )).astype(np.float64)
    truth = np.column_stack((
        physical_state[context_end + 1:end],
        data["frames"][context_end + 1:end, 3:5],
    )).astype(np.float64)
    pose0 = data["simulator_pose_xyyaw"][context_end].astype(np.float64)
    pose_truth = data["simulator_pose_xyyaw"][context_end + 1:end].astype(
        np.float64)
    if (not np.isfinite(predicted).all() or not np.isfinite(truth).all()
            or not np.isfinite(pose0).all() or not np.isfinite(pose_truth).all()):
        raise ValueError(f"{run_id}/{label}: non-finite rollout or labels")

    predicted_pose = _integrate_pose(predicted, physical_state[context_end], pose0)
    pose_error = predicted_pose - pose_truth
    pose_error[:, 2] = _wrap_angle(pose_error[:, 2])
    state_error = predicted - truth
    speed_error = (np.linalg.norm(predicted[:, :2], axis=1)
                   - np.linalg.norm(truth[:, :2], axis=1))
    horizon_scores = {}
    for horizon_s in HORIZONS_S:
        index = round(horizon_s / DT_S) - 1
        if index >= len(state_error):
            continue
        horizon_scores[f"{horizon_s:g}s"] = {
            "position_error_m": float(np.linalg.norm(pose_error[index, :2])),
            "heading_abs_error_rad": float(abs(pose_error[index, 2])),
            "speed_abs_error_mps": float(abs(speed_error[index])),
            "body_state_abs_error": np.abs(state_error[index, :3]).tolist(),
        }
    radial_error = np.linalg.norm(pose_error[:, :2], axis=1)
    return {
        "condition": label,
        "initial_context_seconds": (HISTORY_STEPS - 1) * DT_S,
        "free_run_seconds": len(predicted) * DT_S,
        "free_run_steps": len(predicted),
        "initial_speed_mps": float(np.linalg.norm(physical_state[context_end, :2])),
        "command_steering_p50_p95_abs_rad": np.quantile(
            np.abs(commands_np[:, 0]), (0.50, 0.95)).tolist(),
        "truth_speed_p05_p50_p95_mps": np.quantile(
            np.linalg.norm(truth[:, :2], axis=1), (0.05, 0.50, 0.95)).tolist(),
        "whole_free_run": {
            "state_names": list(STATE_NAMES),
            "state_error": _rmse_bias(state_error),
            "speed_rmse_mps": float(np.sqrt(np.mean(speed_error ** 2))),
            "position_xy_rmse_m": np.sqrt(
                np.mean(pose_error[:, :2] ** 2, axis=0)).tolist(),
            "position_radial_rmse_m": float(np.sqrt(np.mean(radial_error ** 2))),
            "position_radial_p95_m": float(np.quantile(radial_error, 0.95)),
            "position_endpoint_m": float(radial_error[-1]),
            "heading_rmse_rad": float(np.sqrt(np.mean(pose_error[:, 2] ** 2))),
            "heading_endpoint_abs_rad": float(abs(pose_error[-1, 2])),
        },
        "horizon_scores": horizon_scores,
    }


def score(checkpoint_path: Path, dataset_path: Path, output_path: Path,
          run_ids: list[str] | None = None, device_name: str = "cpu",
          split_allowlist: tuple[str, ...] = ("validation", "final_test")):
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    torch, nn = _torch()
    if device_name == "cpu":
        torch.set_num_threads(1)
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    data = _load_dataset(dataset_path)
    if data["schema_version"] < 7 or data["simulator_pose_xyyaw"] is None:
        raise ValueError("scoring requires schema-7+ pose, rigid-state data")
    if (data["simulator_rigid_state"] is None
            or data["simulator_linear_acceleration"] is None
            or data["packet_sequence"] is None
            or not np.allclose(data["dt_s"], DT_S, rtol=0.0, atol=1e-7)):
        raise ValueError("scoring requires complete labels and exact 25 ms packets")
    if data["feature_names"] != metadata["feature_names"]:
        raise ValueError("checkpoint and capture feature layouts differ")
    selected = run_ids or [
        str(run_id) for run_id, split in zip(data["run_ids"], data["splits"])
        if str(split) in split_allowlist
    ]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("provide unique run IDs with eligible sequences")
    run_lookup = {str(run_id): index for index, run_id in enumerate(data["run_ids"])}
    missing = sorted(set(selected) - set(run_lookup))
    if missing:
        raise ValueError(f"selected runs missing from dataset: {missing}")
    overlap = sorted(set(selected).intersection(map(str, metadata["training_runs"])))
    if overlap:
        raise ValueError(f"refusing to score training runs: {overlap}")
    invalid_splits = {
        run_id: data["splits"][run_lookup[run_id]] for run_id in selected
        if data["splits"][run_lookup[run_id]] not in split_allowlist
    }
    if invalid_splits:
        raise ValueError(f"selected runs are not held out: {invalid_splits}")

    model_type = _physical_model(
        torch, nn, bool(metadata["tire_relaxation_state_enabled"]))
    model = model_type().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    physical_state, _ = _targets(data)
    results = []
    for run_id in selected:
        run_index = run_lookup[run_id]
        rows = _sequence_rows(data, run_index, round(0.25 / DT_S))
        if not rows:
            raise ValueError(f"{run_id}: no contiguous sequences long enough to score")
        sequences = [
            _score_sequence(torch, model, data, physical_state,
                            start, end, label, run_id, device)
            for start, end, label in rows
        ]
        results.append({
            "run_id": run_id,
            "split": data["splits"][run_index],
            "sequence_count": len(sequences),
            "sequences": sequences,
        })

    run_level = {}
    common_horizons = sorted(set.intersection(*[
        set(sequence["horizon_scores"])
        for result in results for sequence in result["sequences"]
    ]), key=lambda value: float(value[:-1]))
    for horizon in common_horizons:
        per_run = {}
        for result in results:
            rows = [sequence["horizon_scores"][horizon]
                    for sequence in result["sequences"]
                    if horizon in sequence["horizon_scores"]]
            per_run[result["run_id"]] = {
                "sequence_count": len(rows),
                "macro_sequence_position_error_m": float(np.mean(
                    [row["position_error_m"] for row in rows])),
                "macro_sequence_heading_abs_error_rad": float(np.mean(
                    [row["heading_abs_error_rad"] for row in rows])),
                "macro_sequence_speed_abs_error_mps": float(np.mean(
                    [row["speed_abs_error_mps"] for row in rows])),
            }
        run_level[horizon] = per_run

    whole_run_summary = {}
    for result in results:
        rows = [sequence["whole_free_run"] for sequence in result["sequences"]]
        whole_run_summary[result["run_id"]] = {
            "sequence_count": len(rows),
            "macro_sequence_position_radial_rmse_m": float(np.mean(
                [row["position_radial_rmse_m"] for row in rows])),
            "macro_sequence_endpoint_position_error_m": float(np.mean(
                [row["position_endpoint_m"] for row in rows])),
            "macro_sequence_heading_rmse_rad": float(np.mean(
                [row["heading_rmse_rad"] for row in rows])),
            "macro_sequence_speed_rmse_mps": float(np.mean(
                [row["speed_rmse_mps"] for row in rows])),
        }

    report = {
        "schema_version": 1,
        "purpose": "recursive command-only rollout; simulator truth initializes only the starting physical state and scores future motion",
        "future_sensor_or_truth_inputs": False,
        "initialization": "simulator rigid state and pose at end of 1.975 s context; internal slips inferred from that instant and current sensor/actuator values",
        "command_target_alignment": "command at state k predicts state k+1, matching grey-box training; internal actuator-delay state supplies the measured one-packet latency",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_runs": metadata["training_runs"],
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "sample_period_s": DT_S,
        "history_steps": HISTORY_STEPS,
        "selected_run_ids": selected,
        "split_allowlist": list(split_allowlist),
        "statistical_unit": "independent whole capture run; sequence rows within a run are repeated conditions, not independent replicates",
        "run_level_horizon_summary": run_level,
        "run_level_whole_free_run_summary": whole_run_summary,
        "runs": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--allow-unseen-practice", action="store_true")
    args = parser.parse_args()
    splits = ("validation", "final_test", "unseen_practice") if args.allow_unseen_practice else (
        "validation", "final_test")
    report = score(args.checkpoint, args.dataset, args.output,
                   args.run_ids, args.device, splits)
    print(json.dumps({
        "output": str(args.output),
        "runs": [{"run_id": row["run_id"],
                  "sequences": row["sequence_count"]}
                 for row in report["runs"]],
        "run_level_horizon_summary": report["run_level_horizon_summary"],
        "run_level_whole_free_run_summary": report[
            "run_level_whole_free_run_summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
