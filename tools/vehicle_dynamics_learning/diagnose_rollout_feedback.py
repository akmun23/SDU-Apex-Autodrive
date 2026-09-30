#!/usr/bin/env python3
"""Offline oracle-channel ablation for locating recursive plant-model drift.

Future feedback substitution is intentionally diagnostic-only. It is never a
permitted inference input for the competition observer, odometry, or MPC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_nssm import (
    _batch_from_windows,
    _fixed_eval_windows,
    _load_dataset,
    _model_type,
    _rollout,
    _sequence_groups,
    _tensor_batch,
    _torch,
)


CHANNEL_CASES = {
    "free_running": (),
    "oracle_throttle_feedback": (4,),
    "oracle_steering_and_throttle_feedback": (3, 4),
    "oracle_rear_encoders": (5, 6),
    "oracle_all_actuator_and_encoder_feedback": (3, 4, 5, 6),
}
HORIZONS = (1, 4, 10, 20, 30)
BODY_NAMES = ("u_mps", "v_mps", "yaw_rate_rps")


def diagnose(dataset_path: Path, run_dir: Path, run_ids: list[str],
             max_windows_per_run: int, device_arg: str) -> dict[str, Any]:
    torch, nn = _torch()
    train_report = json.loads((run_dir / "training_report.json").read_text(
        encoding="utf-8"))
    data = _load_dataset(dataset_path)
    index_by_id = {name: i for i, name in enumerate(data["run_ids"])}
    missing = [name for name in run_ids if name not in index_by_id]
    if missing:
        raise ValueError(f"runs absent from dataset: {missing}")
    selected = np.asarray([index_by_id[name] for name in run_ids], dtype=np.int64)
    if np.any(data["splits"][selected] != "train"):
        raise ValueError("diagnostic runs must be held out from the original train split")
    data["splits"][selected] = "diagnostic"

    checkpoint = torch.load(run_dir / "member_00.pt", map_location="cpu",
                             weights_only=False)
    metadata = checkpoint["metadata"]
    architecture = metadata.get("architecture", "gru")
    history_steps = int(metadata["history_steps"])
    rollout_steps = int(metadata["rollout_steps"])
    hidden_size = int(metadata["hidden_size"])
    expert_count = int(metadata.get("expert_count", 1))
    mean = checkpoint["feature_mean"]
    scale = checkpoint["feature_scale"]
    device = device_arg
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    factory = _model_type(torch, nn, hidden_size, architecture, expert_count,
                          history_steps, len(metadata["feature_names"]),
                          mean, scale,
                          metadata.get("body_acceleration_mean"),
                          metadata.get("body_acceleration_scale"),
                          metadata.get("integration_method", "euler"))
    models = []
    for member in train_report["members"]:
        checkpoint_name = member.get("checkpoint")
        if not checkpoint_name:
            continue
        payload = torch.load(run_dir / checkpoint_name, map_location=device,
                             weights_only=False)
        model = factory().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    if not models:
        raise ValueError("no trained member checkpoints found")

    groups = _sequence_groups(data, "diagnostic", history_steps, rollout_steps)
    run_results = []
    for run_index in selected:
        run_index = int(run_index)
        run_name = data["run_ids"][run_index]
        sequence_ids = groups.get(run_index, [])
        windows = _fixed_eval_windows(
            data, {run_index: sequence_ids}, history_steps, rollout_steps,
            max_windows_per_run) if sequence_ids else []
        if not windows:
            run_results.append({"run_id": run_name, "windows": 0,
                                "error": "no supported rollout windows"})
            continue
        arrays = _batch_from_windows(data, windows, history_steps, rollout_steps)
        history, future, dts = _tensor_batch(torch, arrays, mean, scale, device)
        target = future[:, :, :3] * torch.as_tensor(
            scale[:3], dtype=torch.float32, device=device) + torch.as_tensor(
                mean[:3], dtype=torch.float32, device=device)
        case_results = {}
        for case_name, forced_channels in CHANNEL_CASES.items():
            member_predictions = []
            with torch.no_grad():
                for model in models:
                    prediction = _rollout(
                        model, history, future, dts, history_steps,
                        teacher_force_channels=forced_channels)
                    member_predictions.append(prediction[:, :, :3])
            prediction = torch.stack(member_predictions, dim=0).mean(dim=0)
            prediction = prediction * torch.as_tensor(
                scale[:3], dtype=torch.float32, device=device) + torch.as_tensor(
                    mean[:3], dtype=torch.float32, device=device)
            errors = prediction - target
            horizon_metrics = {}
            for step in HORIZONS:
                if step > rollout_steps:
                    continue
                rmse = torch.sqrt(torch.mean(errors[:, step - 1, :] ** 2,
                                             dim=0)).cpu().numpy()
                horizon_metrics[str(step)] = {
                    "seconds_median": float(torch.median(
                        torch.sum(dts[:, :step], dim=1)).cpu()),
                    "rmse": dict(zip(BODY_NAMES, map(float, rmse))),
                }
            case_results[case_name] = horizon_metrics
        run_results.append({"run_id": run_name, "windows": len(windows),
                            "cases": case_results})

    result = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "model_run_dir": str(run_dir.resolve()),
        "architecture": architecture,
        "member_count": len(models),
        "diagnostic_only": True,
        "future_truth_use": (
            "Selected future steering/throttle feedback and/or rear encoder "
            "channels replace model predictions after each step solely to "
            "attribute recursive error. These channels are unavailable as "
            "future MPC inputs and must never be used at runtime."),
        "runs": run_results,
    }
    output = run_dir / "rollout_feedback_attribution.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {output}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--max-windows-per-run", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.max_windows_per_run < 1:
        parser.error("max-windows-per-run must be positive")
    try:
        diagnose(args.dataset, args.run_dir, args.run_ids,
                 args.max_windows_per_run, args.device)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"diagnostic failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
