#!/usr/bin/env python3
"""Evaluate a frozen causal observer on named, whole-run dataset holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.vehicle_dynamics_learning import train_sensor_observer


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch, nn = train_sensor_observer._torch()
    data = train_sensor_observer._load_dataset(args.dataset)
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    payload = torch.load(args.checkpoint, map_location=args.device,
                         weights_only=True)
    metadata = payload["metadata"]
    if metadata["sensor_feature_names"] != data["sensor_feature_names"]:
        raise ValueError("checkpoint and dataset sensor feature schemas differ")
    trained_on = set(map(str, metadata.get("training_runs", [])))
    requested = set(args.run_ids)
    overlap = sorted(requested & trained_on)
    if overlap:
        raise ValueError("refusing to score training run(s): " + ", ".join(overlap))

    ids = {str(run_id): index for index, run_id in enumerate(data["run_ids"])}
    missing = sorted(requested - set(ids))
    if missing:
        raise ValueError("run ID(s) absent from dataset: " + ", ".join(missing))
    groups = train_sensor_observer._valid_segments(
        data, args.split, int(metadata["burn_in"]) + int(metadata["score_steps"]))
    Model = train_sensor_observer._make_model(
        nn, len(metadata["sensor_feature_names"]), int(metadata["hidden_size"]))
    model = Model().to(args.device)
    model.load_state_dict(payload["state_dict"])

    normalization = report["normalization"]
    sensor_mean = np.asarray(normalization["sensor_mean"], dtype=np.float32)
    sensor_scale = np.asarray(normalization["sensor_scale"], dtype=np.float32)
    truth_mean = np.asarray(normalization["target_mean"], dtype=np.float32)
    truth_scale = np.asarray(normalization["target_scale"], dtype=np.float32)

    results = {}
    for run_id in sorted(requested):
        run_index = ids[run_id]
        if data["splits"][run_index] != args.split:
            raise ValueError(f"{run_id} is assigned to {data['splits'][run_index]!r}, "
                             f"not requested split {args.split!r}")
        run_groups = {run_index: groups.get(run_index, [])}
        windows = train_sensor_observer._eval_windows(
            run_groups, int(metadata["burn_in"]) + int(metadata["score_steps"]),
            args.max_windows)
        selected = windows.get(run_index, [])
        if not selected:
            raise ValueError(f"{run_id} has no sensor-valid evaluation windows")
        arrays = train_sensor_observer._arrays_for_windows(data, selected)
        results[run_id] = train_sensor_observer._score(
            torch, model, arrays, sensor_mean, sensor_scale, truth_mean,
            truth_scale, args.device, int(metadata["burn_in"]),
            int(metadata["score_steps"]), include_strata=True)

    output = {
        "dataset": str(args.dataset.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "training_report": str(args.training_report.resolve()),
        "checkpoint_training_runs": sorted(trained_on),
        "split": args.split,
        "runs": results,
        "input_policy": "causal sensor/actuator features only; bridge state is an offline label",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--run-ids", nargs="+", required=True)
    parser.add_argument("--split", choices=("validation", "test", "final_test"),
                        default="test")
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_windows < 1:
        parser.error("--max-windows must be positive")
    result = evaluate(args)
    print(json.dumps({"runs": {name: metric["causal_sensor_observer_rmse"]
                               for name, metric in result["runs"].items()},
                      "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
