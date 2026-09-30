#!/usr/bin/env python3
"""One-shot scoring of preselected fold models on one untouched whole run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from run_structured_body_cv import (
    ATTITUDE_INPUT_CHANNELS,
    _make_model,
    _score_run,
    _torch,
)
from structured_body_models import BODY_HORIZONS
from train_nssm import _load_dataset


def score(dataset_path: Path, cv_dir: Path, run_id: str,
          architectures: list[str], output_path: Path,
          device: str = "auto", max_windows: int = 128) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError(f"refusing to overwrite one-shot holdout report: {output_path}")
    report = json.loads((cv_dir / "cv_report.json").read_text(encoding="utf-8"))
    data = _load_dataset(dataset_path)
    attitude_conditioned = bool(report["configuration"].get(
        "condition_on_measured_attitude_history", False))
    excluded_sequences = 0
    if data["imu_attitude_valid"] is not None:
        complete = np.asarray([
            bool(np.all(data["imu_attitude_valid"][int(start):int(end)]))
            for start, end in data["bounds"]
        ], dtype=bool)
        excluded_sequences = int(np.count_nonzero(~complete))
        if excluded_sequences:
            data["bounds"] = data["bounds"][complete]
            data["seq_run"] = data["seq_run"][complete]
    held_channels: tuple[int, ...] = ()
    if attitude_conditioned:
        attitude = data["imu_attitude_frames"]
        if attitude is None:
            raise ValueError("selected CV requires IMU attitude arrays")
        data["frames"] = np.column_stack((data["frames"], attitude)).astype(
            np.float32, copy=False)
        data["feature_names"].extend(data["imu_attitude_feature_names"])
        held_channels = ATTITUDE_INPUT_CHANNELS

    matches = np.flatnonzero(data["run_ids"] == run_id)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one dataset run named {run_id!r}")
    run = int(matches[0])
    if data["splits"][run] != "test":
        raise ValueError(f"refusing non-test run {run_id!r} for final scoring")
    if not any(int(seq_run) == run for seq_run in data["seq_run"]):
        raise ValueError(f"holdout {run_id!r} has no valid whole-run sequences")
    if set(architectures) - set(report["configuration"]["architectures"]):
        raise ValueError("requested architecture is absent from the selected CV run")

    torch, nn = _torch()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in the training environment")
    torch.set_num_threads(4)
    results = {}
    for architecture in architectures:
        fold_scores = {}
        for fold in range(int(report["fold_count"])):
            checkpoint = cv_dir / f"fold_{fold:02d}" / architecture / f"{architecture}.pt"
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            metadata = payload["metadata"]
            if (metadata.get("feature_names") != data["feature_names"]
                    or metadata.get("architecture") != architecture):
                raise ValueError(f"checkpoint/data mismatch: {checkpoint}")
            feature_mean = np.asarray(metadata["feature_mean"], dtype=np.float32)
            feature_scale = np.asarray(metadata["feature_scale"], dtype=np.float32)
            acceleration_mean = np.asarray(metadata["acceleration_mean"],
                                            dtype=np.float32)
            acceleration_scale = np.asarray(metadata["acceleration_scale"],
                                            dtype=np.float32)
            feature_count = data["frames"].shape[1]
            model = _make_model(
                torch, nn, architecture, feature_count, feature_mean,
                feature_scale, acceleration_mean, acceleration_scale,
                np.zeros((feature_count + 2, 3), dtype=np.float32),
                int(metadata["hidden_size"])).to(device)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            fold_scores[str(fold)] = _score_run(
                torch, model, architecture == "gru", data, run,
                feature_mean, feature_scale,
                report["folds"][fold]["regime_thresholds"],
                report["configuration"]["mpc_yaw_parameters"],
                int(report["configuration"]["history_steps"]), max_windows,
                device, int(report["configuration"]["inference_batch_size"])
                if "inference_batch_size" in report["configuration"] else 256,
                held_input_channels=held_channels)
        fold_rmse = {}
        for horizon in BODY_HORIZONS:
            values = [fold_scores[str(fold)][str(horizon)]["all"]["rmse"]
                      for fold in range(int(report["fold_count"]))]
            fold_rmse[str(horizon)] = {
                name: {
                    "fold_macro_rmse": float(np.mean([
                        value[name] for value in values]))
                    if all(value.get(name) is not None for value in values) else None,
                    "fold_min_rmse": float(np.min([
                        value[name] for value in values]))
                    if all(value.get(name) is not None for value in values) else None,
                    "fold_max_rmse": float(np.max([
                        value[name] for value in values]))
                    if all(value.get(name) is not None for value in values) else None,
                }
                for name in ("u_mps", "v_mps", "yaw_rate_rps")
            }
        results[architecture] = {
            "fold_macro_rmse_by_horizon": fold_rmse,
            "fold_model_scores_on_same_holdout_run": fold_scores,
        }

    result = {
        "evaluation": "one-shot final whole-run holdout; no retraining or model selection",
        "run_id": run_id,
        "split": "test",
        "max_windows_per_horizon": max_windows,
        "fold_count": int(report["fold_count"]),
        "attitude_conditioned": attitude_conditioned,
        "attitude_future_policy": (
            "observed roll/pitch/rates held constant; no future attitude samples"
            if attitude_conditioned else "not used"),
        "incomplete_attitude_sequences_excluded": excluded_sequences,
        "body_horizons_steps": list(BODY_HORIZONS),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n",
                           encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("cv_dir", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--architectures", nargs="+", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-windows", type=int, default=128)
    args = parser.parse_args()
    try:
        result = score(args.dataset, args.cv_dir, args.run_id,
                       args.architectures, args.output, args.device,
                       args.max_windows)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"one-shot holdout scoring failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output),
                      "run_id": result["run_id"],
                      "architectures": list(result["results"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
