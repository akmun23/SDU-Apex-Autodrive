#!/usr/bin/env python3
"""Score a frozen direct-sequence checkpoint on whole independent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    _eligible_sequences,
    _evaluate,
    _targets,
    _torch_model,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset


def score(checkpoint_path: Path, dataset_path: Path, split: str,
          output_path: Path, *, device_name: str = "cuda",
          max_windows_per_run: int = 64, seed: int = 20261001) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu",
                         weights_only=False)
    metadata = payload["metadata"]
    data = _load_dataset(dataset_path)
    if metadata["feature_names"] != data["feature_names"]:
        raise ValueError("checkpoint and dataset feature layouts differ")
    training_runs = set(metadata["training_runs"])
    data, targets = data, _targets(data)
    groups = _eligible_sequences(
        data, targets, split, int(metadata["context_steps"]),
        int(metadata["future_steps"]))
    evaluated_ids = {str(data["run_ids"][run]) for run in groups}
    overlap = sorted(evaluated_ids.intersection(training_runs))
    if overlap:
        raise ValueError(f"refusing to score training runs: {overlap}")
    if not groups:
        raise ValueError(f"no eligible {split!r} sequences in {dataset_path}")
    torch.set_num_threads(1)
    device = torch.device(device_name)
    Model = _torch_model(
        torch, torch.nn, int(metadata["context_steps"]),
        int(metadata["future_steps"]), int(metadata["width"]),
        int(metadata["layer_count"]), int(metadata["head_count"]), 0.05)
    model = Model().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metrics = _evaluate(
        torch, model, data, targets, groups,
        payload["x_mean"], payload["x_scale"],
        payload["y_mean"], payload["y_scale"], device,
        int(metadata["context_steps"]), int(metadata["future_steps"]),
        seed, max_windows_per_run)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "split": split,
        "run_ids": sorted(evaluated_ids),
        "independent_run_count": len(evaluated_ids),
        "device": str(device),
        "metrics": metrics,
        "future_truth_or_sensors_used_as_inputs": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="final_test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-windows-per-run", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    report = score(
        args.checkpoint, args.dataset, args.split, args.output,
        device_name=args.device,
        max_windows_per_run=args.max_windows_per_run, seed=args.seed)
    print(json.dumps({
        "runs": report["independent_run_count"],
        "windows": report["metrics"]["window_count"],
        "macro_horizons": report["metrics"]["macro_run_horizons"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
