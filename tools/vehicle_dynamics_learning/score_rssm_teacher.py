#!/usr/bin/env python3
"""Score a frozen prior-only RSSM checkpoint on independent whole runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    _eligible_sequences,
    _evaluate,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model


def score(checkpoint_path: Path, dataset_path: Path, split: str,
          output_path: Path, *, device_name: str = "cuda",
          max_windows_per_run: int = 64, seed: int = 20261003) -> dict:
    torch, nn = _torch()
    payload = torch.load(checkpoint_path, map_location="cpu",
                         weights_only=False)
    metadata = payload["metadata"]
    data = _load_dataset(dataset_path)
    if metadata["feature_names"] != data["feature_names"]:
        raise ValueError("checkpoint and dataset feature layouts differ")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    targets = _targets(data)
    groups = _eligible_sequences(
        data, targets, split, int(metadata["context_steps"]),
        int(metadata["rollout_steps"]))
    evaluated_ids = {str(data["run_ids"][run]) for run in groups}
    training_runs = set(metadata["training_runs"])
    overlap = sorted(evaluated_ids.intersection(training_runs))
    if overlap:
        raise ValueError(f"refusing to score training runs: {overlap}")
    if not groups:
        raise ValueError(f"no eligible {split!r} sequences in {dataset_path}")
    model_type = _rssm_model(
        torch, nn, int(metadata["hidden_size"]), int(metadata["latent_size"]),
        payload["x_mean"], payload["x_scale"], payload["y_mean"],
        payload["y_scale"])
    model = model_type().to(device)
    model.load_state_dict(payload["state_dict"])
    metrics = _evaluate(
        torch, model, data, targets, groups,
        payload["x_mean"], payload["x_scale"], payload["y_mean"],
        payload["y_scale"], device, int(metadata["context_steps"]),
        int(metadata["rollout_steps"]), seed, max_windows_per_run)
    report = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "split": split,
        "run_ids": sorted(evaluated_ids),
        "independent_run_count": len(evaluated_ids),
        "device": str(device),
        "metrics": metrics,
        "posterior_uses_future_targets": False,
        "free_rollout_uses_command_conditioned_prior_mean": True,
        "window_count_is_not_independent_run_count": True,
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
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-windows-per-run", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    result = score(
        args.checkpoint, args.dataset, args.split, args.output,
        device_name=args.device,
        max_windows_per_run=args.max_windows_per_run, seed=args.seed)
    print(json.dumps({
        "runs": result["independent_run_count"],
        "windows": result["metrics"]["window_count"],
        "macro_horizons": result["metrics"]["macro_run_horizons"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
