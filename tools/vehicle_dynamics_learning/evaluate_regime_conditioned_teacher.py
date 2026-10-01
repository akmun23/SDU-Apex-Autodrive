#!/usr/bin/env python3
"""Evaluate a causal speed/steering switch between two command-driven GRUs.

Both models see the same initial truth history. After that, each receives the
same fused predicted body/actuator/wheel state and recorded command. The gate
uses only that current fused prediction; future truth is used only for scoring.
This is an offline architecture experiment and does not modify runtime nodes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.evaluate_free_running_plant import (
    SIMULATOR_DT_S,
    _free_rollout,
    _load_models,
    _sequence_score,
)
from tools.vehicle_dynamics_learning.structured_body_models import (
    REAR_AXLE_TO_COM_X_M,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch


def _conditioned_rollout(torch, generalist, specialist,
                         generalist_payload: dict[str, Any],
                         specialist_payload: dict[str, Any],
                         frames: np.ndarray, dt_s: np.ndarray,
                         history_steps: int, speed_threshold_mps: float,
                         steering_threshold_rad: float
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Run both transitions on a shared autoregressive state and gate causally."""
    mean_a = generalist_payload["feature_mean"].astype(np.float32, copy=False)
    scale_a = generalist_payload["feature_scale"].astype(np.float32, copy=False)
    mean_b = specialist_payload["feature_mean"].astype(np.float32, copy=False)
    scale_b = specialist_payload["feature_scale"].astype(np.float32, copy=False)
    if not (np.array_equal(mean_a, mean_b) and np.array_equal(scale_a, scale_b)):
        raise ValueError("regime candidates must share identical normalizers")

    device = next(generalist.parameters()).device
    state_count = 7
    states = frames[:, :state_count].astype(np.float64, copy=True)
    normalized_history_a = (frames[:history_steps] - mean_a) / scale_a
    normalized_history_b = (frames[:history_steps] - mean_b) / scale_b
    hidden_a = torch.zeros(
        1, generalist.cell.hidden_size, dtype=torch.float32, device=device)
    hidden_b = torch.zeros(
        1, specialist.cell.hidden_size, dtype=torch.float32, device=device)
    with torch.no_grad():
        for index in range(history_steps - 1):
            input_a = torch.as_tensor(
                normalized_history_a[index:index + 1], dtype=torch.float32,
                device=device)
            input_b = torch.as_tensor(
                normalized_history_b[index:index + 1], dtype=torch.float32,
                device=device)
            hidden_a = generalist.cell(input_a, hidden_a)
            hidden_b = specialist.cell(input_b, hidden_b)

        current_state = frames[history_steps - 1, :state_count].astype(
            np.float64, copy=True)
        current_command = frames[history_steps - 1, state_count:].astype(
            np.float32, copy=True)
        gate = np.zeros(len(frames), dtype=bool)
        for index in range(history_steps, len(frames)):
            feature = np.concatenate((current_state, current_command)).astype(
                np.float32, copy=False)
            input_a = torch.as_tensor(
                ((feature - mean_a) / scale_a)[None, :],
                dtype=torch.float32, device=device)
            input_b = torch.as_tensor(
                ((feature - mean_b) / scale_b)[None, :],
                dtype=torch.float32, device=device)
            step = torch.as_tensor(
                [float(dt_s[index])], dtype=torch.float32, device=device)
            next_a, hidden_a = generalist.advance(input_a, hidden_a, step)
            next_b, hidden_b = specialist.advance(input_b, hidden_b, step)
            physical_a = (next_a[0].cpu().numpy().astype(np.float64)
                          * scale_a[:state_count] + mean_a[:state_count])
            physical_b = (next_b[0].cpu().numpy().astype(np.float64)
                          * scale_b[:state_count] + mean_b[:state_count])

            use_specialist = (
                math.hypot(current_state[0], current_state[1])
                >= speed_threshold_mps
                and abs(current_state[3]) >= steering_threshold_rad)
            gate[index] = use_specialist
            current_state = physical_b if use_specialist else physical_a
            states[index] = current_state
            current_command = frames[index, state_count:].astype(
                np.float32, copy=False)

    selected = gate[history_steps:]
    return states, {
        "predicted_specialist_steps": int(np.count_nonzero(selected)),
        "predicted_generalist_steps": int(len(selected) - np.count_nonzero(selected)),
        "specialist_fraction": float(np.mean(selected)) if len(selected) else 0.0,
        "gate_transitions": int(np.count_nonzero(selected[1:] != selected[:-1])),
    }


def evaluate(dataset_path: Path, generalist_dir: Path, specialist_dir: Path,
             run_ids: list[str], speed_threshold_mps: float,
             steering_threshold_rad: float, output: Path) -> dict[str, Any]:
    torch, _ = _torch()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    data = _load_dataset(dataset_path)
    if not np.allclose(data["dt_s"], SIMULATOR_DT_S, rtol=0.0, atol=1e-7):
        raise ValueError("regime evaluator requires the exact 25 ms simulator timebase")

    _, generalist_models, generalist_payload, generalist_metadata, _ = (
        _load_models(generalist_dir, data, "cpu"))
    _, specialist_models, specialist_payload, specialist_metadata, _ = (
        _load_models(specialist_dir, data, "cpu"))
    if len(generalist_models) != 1 or len(specialist_models) != 1:
        raise ValueError("regime evaluation expects one checkpoint per candidate")
    if (generalist_metadata.get("architecture", "gru") != "gru"
            or specialist_metadata.get("architecture", "gru") != "gru"):
        raise ValueError("regime evaluation currently supports GRU candidates only")
    if generalist_metadata["feature_names"] != specialist_metadata["feature_names"]:
        raise ValueError("regime candidates have different feature layouts")
    history_steps = int(generalist_metadata["history_steps"])
    if int(specialist_metadata["history_steps"]) != history_steps:
        raise ValueError("regime candidates must use the same initialization history")
    if ("simulator_pose_xyyaw" not in data
            or "odom_pose_xyyaw" not in data or "lap_count" not in data):
        raise ValueError("regime evaluation requires pose and lap labels")

    manifest_path = dataset_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {row["run_id"]: row for row in manifest.get("runs", [])}
    run_indices = {str(name): index for index, name in enumerate(data["run_ids"])}
    missing = sorted(set(run_ids) - set(run_indices))
    if missing:
        raise ValueError(f"run IDs absent from dataset: {missing}")

    reports = []
    for run_id in run_ids:
        run_index = run_indices[run_id]
        record = records.get(run_id, {})
        if not record.get("clean_stream_and_collision_gate", False):
            raise ValueError(f"refusing quality-gated run {run_id}")
        sequence_rows = []
        for sequence_id, sequence_run in enumerate(data["seq_run"]):
            if int(sequence_run) != run_index:
                continue
            start, end = map(int, data["bounds"][sequence_id])
            if end - start <= history_steps:
                continue
            frames = data["frames"][start:end]
            dt_s = data["dt_s"][start:end]
            simulator_pose = data["simulator_pose_xyyaw"][start:end]
            odom_pose = data["odom_pose_xyyaw"][start:end]
            lap_count = data["lap_count"][start:end]
            if np.mean(np.isfinite(simulator_pose).all(axis=1)) < 0.999:
                continue

            generalist_states = _free_rollout(
                torch, generalist_models[0], frames, dt_s, history_steps,
                generalist_payload["feature_mean"],
                generalist_payload["feature_scale"])
            specialist_states = _free_rollout(
                torch, specialist_models[0], frames, dt_s, history_steps,
                specialist_payload["feature_mean"],
                specialist_payload["feature_scale"])
            conditioned_states, gate_stats = _conditioned_rollout(
                torch, generalist_models[0], specialist_models[0],
                generalist_payload, specialist_payload, frames, dt_s,
                history_steps, speed_threshold_mps,
                steering_threshold_rad)
            truth = frames.astype(np.float64, copy=False)
            common = (truth, dt_s, simulator_pose, odom_pose, lap_count,
                      history_steps, REAR_AXLE_TO_COM_X_M)
            sequence_rows.append({
                "sequence_id": int(sequence_id),
                "sample_count_total": int(end - start),
                "generalist": _sequence_score(generalist_states, *common),
                "specialist": _sequence_score(specialist_states, *common),
                "regime_conditioned": {
                    **_sequence_score(conditioned_states, *common),
                    "gate": gate_stats,
                },
            })
        if not sequence_rows:
            raise ValueError(f"no pose-valid sequence long enough in {run_id}")
        reports.append({
            "run_id": run_id,
            "bag": record.get("bag"),
            "quality_failures": record.get("quality_failures", []),
            "sequences": sequence_rows,
        })

    result = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "generalist_model_dir": str(generalist_dir.resolve()),
        "specialist_model_dir": str(specialist_dir.resolve()),
        "future_truth_or_measured_actuator_wheel_inputs": False,
        "future_inputs_after_initialization": [
            "logged steering command", "logged throttle command"],
        "initial_truth_history_steps": history_steps,
        "gate_policy": (
            "select specialist iff the shared current fused predicted body "
            f"speed >= {speed_threshold_mps:g} m/s and absolute predicted "
            f"steering feedback >= {steering_threshold_rad:g} rad"),
        "models_receive_same_fused_prediction_each_step": True,
        "run_level_reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("generalist_model_dir", type=Path)
    parser.add_argument("specialist_model_dir", type=Path)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--speed-threshold-mps", type=float, default=7.0)
    parser.add_argument("--steering-threshold-rad", type=float, default=0.16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.speed_threshold_mps < 0.0 or args.steering_threshold_rad < 0.0:
        parser.error("gate thresholds must be non-negative")
    try:
        evaluate(args.dataset, args.generalist_model_dir,
                 args.specialist_model_dir, args.run_id,
                 args.speed_threshold_mps, args.steering_threshold_rad,
                 args.output)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"regime-conditioned evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
