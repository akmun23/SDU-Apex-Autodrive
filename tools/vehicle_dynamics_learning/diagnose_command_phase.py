#!/usr/bin/env python3
"""Test recorded-command frame phase on frozen whole-run dynamic validation.

This is a causal-indexing diagnostic, not a training procedure. For a state at
frame k, command offsets select command[k + offset] for the k -> k+1 transition.
Offset +1 reproduces the existing rollout convention; offset 0 tests the
same-frame command implied by the actuator fit's y[k+1] equation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning.compare_effective_race_teacher import (
    _build_inputs,
    _fixed_validation_windows,
    _paired_run_difference,
    _per_window_errors,
    _run_macro,
)
from tools.vehicle_dynamics_learning.effective_race_teacher import (
    DT_S,
    append_roll_state,
    physical_state_from_dataset,
    raw_encoder_history_features,
    wheel_innovation_history_features,
    integrate_pose,
)
from tools.vehicle_dynamics_learning.score_effective_race_teacher import _load_model
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (ROOT / "live_runs/derived_dynamics_learning_20260928/"
                   "full_modeling_reset_20261001/replacement_offline_sim_raceline_20261002/"
                   "replacement_teacher_dataset_v1/openplane_dynamics.npz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rollout_phase(torch, model, data: dict[str, Any], state: np.ndarray,
                   windows: list[dict[str, Any]], offset: int,
                   device: str) -> tuple[np.ndarray, ...]:
    predictions, poses, truths, truth_poses = [], [], [], []
    raw_history = (
        raw_encoder_history_features(data) if model.include_raw_encoder_history else
        wheel_innovation_history_features(data)
        if model.include_wheel_innovation_history else None)
    for start in range(0, len(windows), 24):
        rows = windows[start:start + 24]
        history, initial, delayed, _, target, target_pose = _build_inputs(
            data, state, rows, model.history_state_size, raw_history)
        command_batches = []
        for row in rows:
            frame = int(row["start"])
            sequence = int(row["sequence_id"])
            left, right = map(int, data["bounds"][sequence])
            indices = frame + offset + np.arange(200)
            if indices[0] < left or indices[-1] >= right:
                raise ValueError("phase diagnostic command index escaped its sequence")
            command_batches.append(data["frames"][indices, 7:9])
        commands = np.asarray(command_batches, dtype=np.float32)
        with torch.no_grad():
            predicted, _, _, _ = model.rollout(
                torch.as_tensor(initial, dtype=torch.float32, device=device),
                torch.as_tensor(delayed, dtype=torch.float32, device=device),
                torch.as_tensor(history, dtype=torch.float32, device=device),
                torch.as_tensor(commands, dtype=torch.float32, device=device))
            predicted_pose = integrate_pose(
                torch, predicted,
                torch.as_tensor(target_pose[:, 0], dtype=torch.float32,
                                device=device),
                torch.as_tensor(initial, dtype=torch.float32, device=device))
        predictions.append(predicted.cpu().numpy().astype(np.float64))
        poses.append(predicted_pose.cpu().numpy().astype(np.float64))
        truths.append(target.astype(np.float64))
        truth_poses.append(target_pose.astype(np.float64))
    return tuple(np.concatenate(values) for values in (
        predictions, poses, truths, truth_poses))


def diagnose(dataset_path: Path, checkpoint_paths: dict[str, Path],
             output_dir: Path, offsets: tuple[int, ...], device: str
             ) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite phase diagnostic: {output_dir}")
    data = _load_dataset(dataset_path)
    if int(data["schema_version"]) != 9:
        raise ValueError("command phase diagnosis requires the frozen schema-9 dataset")
    state = physical_state_from_dataset(data).astype(np.float32)
    windows = _fixed_validation_windows(data, state)
    run_ids = np.asarray(data["run_ids"]).astype(str)
    run_indices = np.asarray([int(row["run"]) for row in windows], dtype=np.int32)
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "held-out dynamic validation of command-frame phase only",
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "timebase_s": DT_S,
        "windows": len(windows),
        "independent_whole_runs": sorted(set(run_ids[run_indices].tolist())),
        "test_and_final_test_used": False,
        "future_truth_or_feedback_used": False,
        "offset_semantics": {
            "-1": "previous-frame command[k-1]",
            "0": "same-frame command[k]",
            "1": "next-frame command[k+1]; existing rollout convention",
        },
        "checkpoints": {},
    }
    for name, checkpoint_path in checkpoint_paths.items():
        torch, model, metadata = _load_model(checkpoint_path.resolve(), device)
        model_state = (append_roll_state(data, state)
                       if model.include_roll_state else state)
        offset_results = {}
        for offset in offsets:
            predicted, predicted_pose, truth, truth_pose = _rollout_phase(
                torch, model, data, model_state, windows, offset, device)
            per_window = _per_window_errors(
                predicted, predicted_pose, truth, truth_pose, "com")
            metrics: dict[str, Any] = {}
            for horizon in ("0.75s", "2s", "5s"):
                horizon_metrics = {}
                for metric, raw_values in per_window[horizon].items():
                    values = np.asarray(raw_values, dtype=np.float64)
                    horizon_metrics[metric] = _run_macro(
                        values, run_indices, run_ids)
                metrics[horizon] = horizon_metrics
            offset_results[str(offset)] = metrics
        reference = offset_results.get("1")
        phase_differences = {}
        if reference is not None:
            for offset_text, metrics in offset_results.items():
                if offset_text == "1":
                    continue
                delta = {}
                for horizon, horizon_metrics in metrics.items():
                    delta[horizon] = {}
                    for metric, summary in horizon_metrics.items():
                        delta[horizon][metric] = _paired_run_difference(
                            {"per_run": summary["per_run"]},
                            {"per_run": reference[horizon][metric]["per_run"]})
                phase_differences[offset_text] = delta
        report["checkpoints"][name] = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path.resolve()),
            "actuator_fit": metadata["actuator_fit"],
            "phase_metrics": offset_results,
            "paired_difference_vs_existing_offset_1": phase_differences,
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "command_phase_diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=(-1, 0, 1))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        if "=" not in value:
            parser.error("--checkpoint must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or name in checkpoints:
            parser.error("checkpoint names must be non-empty and unique")
        checkpoints[name] = Path(path)
    if set(args.offsets) - {-1, 0, 1} or len(set(args.offsets)) != len(args.offsets):
        parser.error("--offsets accepts distinct values from -1, 0, 1")
    report = diagnose(args.dataset, checkpoints, args.output_dir,
                      tuple(args.offsets), args.device)
    print(json.dumps({"output": str(args.output_dir.resolve()),
                      "windows": report["windows"],
                      "checkpoints": list(report["checkpoints"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
