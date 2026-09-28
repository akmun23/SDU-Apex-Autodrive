#!/usr/bin/env python3
"""Score saved recurrent plant checkpoints separately on whole held-out runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from train_nssm import (
    _batch_from_windows,
    _fixed_eval_windows,
    _high_steering_windows,
    _gate_diagnostics,
    _load_dataset,
    _model_type,
    _score_ensemble,
    _score_model,
    _sequence_groups,
    _torch,
)


def evaluate(dataset_path: Path, run_dir: Path, max_windows_per_run: int,
             device_arg: str, subset: str,
             run_selection_report_path: Path | None = None,
             high_steering_threshold: float | None = None,
             explicit_run_ids: list[str] | None = None,
             output_dir: Path | None = None) -> dict[str, Any]:
    torch, nn = _torch()
    training_report_path = run_dir / "training_report.json"
    training_report = (json.loads(training_report_path.read_text(encoding="utf-8"))
                       if training_report_path.exists() else {})
    selection_report_path = run_selection_report_path or training_report_path
    if explicit_run_ids:
        if run_selection_report_path is not None:
            raise ValueError("explicit run IDs cannot be combined with a run-selection report")
        run_ids = explicit_run_ids
    else:
        if not selection_report_path.exists():
            raise ValueError("a training report, run-selection report, or explicit run ID is required")
        selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
        run_id_field = ("experiment_test_runs" if subset == "experiment_test"
                        else "additional_validation_runs")
        run_ids = selection_report["configuration"][run_id_field]
    if not run_ids:
        raise ValueError(f"training report has no selected {subset} runs")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("selected run IDs must be unique")
    first_path = run_dir / "member_00.pt"
    first_payload = torch.load(first_path, map_location="cpu", weights_only=False)
    metadata = first_payload["metadata"]
    architecture = metadata.get("architecture", "gru")
    expert_count = int(metadata.get("expert_count", 1))
    hidden_size = int(metadata["hidden_size"])
    history_steps = int(metadata["history_steps"])
    rollout_steps = int(metadata["rollout_steps"])
    feature_names = metadata["feature_names"]
    data = _load_dataset(dataset_path,
                         bool(metadata.get("throttle_variation_feature", False)))
    if data["feature_names"] != feature_names:
        raise ValueError("checkpoint and dataset feature schemas differ")
    run_index = {run_id: index for index, run_id in enumerate(data["run_ids"])}
    missing = [run_id for run_id in run_ids if run_id not in run_index]
    if missing:
        raise ValueError(f"held-out runs absent from dataset: {missing}")
    selected_indices = np.asarray([run_index[run_id] for run_id in run_ids],
                                  dtype=np.int64)
    if explicit_run_ids:
        selected_splits = data["splits"][selected_indices]
        if np.any(np.char.startswith(selected_splits, "exclude_")) or np.any(
                selected_splits == "final_test"):
            raise ValueError("explicit held-out runs cannot use excluded or final-test data")
    elif np.any(data["splits"][selected_indices] != "train"):
        raise ValueError(f"selected {subset} run is not from the original train split")
    data["splits"][selected_indices] = subset
    mean = first_payload["feature_mean"]
    scale = first_payload["feature_scale"]
    device = device_arg
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    model_factory = _model_type(torch, nn, hidden_size, architecture,
                                expert_count, history_steps, len(feature_names))
    models = []
    if training_report.get("members"):
        checkpoint_paths = [run_dir / member["checkpoint"]
                            for member in training_report["members"]
                            if member.get("checkpoint")]
    else:
        checkpoint_paths = sorted(run_dir.glob("member_*.pt"))
    for checkpoint_path in checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location=device,
                             weights_only=False)
        model = model_factory().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    if not models:
        raise ValueError("no trained member checkpoints found")

    groups = _sequence_groups(data, subset, history_steps,
                              rollout_steps)
    rows = []
    for index in selected_indices:
        run = int(index)
        run_sequences = groups.get(run, [])
        if not run_sequences:
            rows.append({"run_id": data["run_ids"][run], "windows": 0,
                         "error": "no sequence long enough for requested rollout"})
            continue
        if high_steering_threshold is None:
            run_windows = _fixed_eval_windows(
                data, {run: run_sequences}, history_steps, rollout_steps,
                max_windows_per_run)
        else:
            eligible = _high_steering_windows(
                data, {run: run_sequences}, history_steps, rollout_steps,
                high_steering_threshold).get(run, [])
            if len(eligible) > max_windows_per_run:
                selected = np.linspace(
                    0, len(eligible) - 1, max_windows_per_run,
                    dtype=np.int64)
                run_windows = [eligible[index] for index in selected]
            else:
                run_windows = eligible
            if not run_windows:
                rows.append({
                    "run_id": data["run_ids"][run], "windows": 0,
                    "error": "no validation windows meet the high-steering threshold"})
                continue
        arrays = _batch_from_windows(data, run_windows, history_steps,
                                     rollout_steps)
        member_scores = [
            _score_model(torch, model, arrays, mean, scale, device, history_steps)
            for model in models
        ]
        rows.append({
            "run_id": data["run_ids"][run],
            "windows": len(run_windows),
            "members": member_scores,
            "ensemble": _score_ensemble(
                torch, models, arrays, mean, scale, device, history_steps),
            "gate_diagnostics": _gate_diagnostics(
                torch, models[0], arrays, mean, scale, device, history_steps),
        })

    result = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "training_run_dir": str(run_dir.resolve()),
        "run_selection_report": (
            str(selection_report_path.resolve()) if not explicit_run_ids else None),
        "checkpoint_status": ("complete_training_report" if training_report_path.exists()
                              else "partial_checkpoint_set_no_training_report"),
        "architecture": architecture,
        "member_count": len(models),
        "evaluation_subset": subset,
        "evaluation_policy": (
            "Each selected simulator run is scored separately. Windows remain "
            "wholly within a sequence; the final-test split is not used."),
        "high_steering_threshold_rad": high_steering_threshold,
        "explicit_run_ids": run_ids if explicit_run_ids else None,
        "runs": rows,
    }
    suffix = (f"_{subset}_high_steering_{high_steering_threshold:.2f}"
              if high_steering_threshold is not None else f"_{subset}")
    report_dir = output_dir or run_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"per_run{suffix}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"wrote {output_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-windows-per-run", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-selection-report", type=Path,
                        help="use this completed report's frozen run IDs when evaluating a checkpoint-only directory")
    parser.add_argument("--run-id", action="append",
                        help="explicit fresh whole-run ID in the evaluation dataset; may be repeated")
    parser.add_argument("--output-dir", type=Path,
                        help="write evaluation reports here instead of beside the saved checkpoints")
    parser.add_argument("--high-steering-threshold", type=float,
                        help="score only windows whose mean future absolute measured steering reaches this rad threshold")
    parser.add_argument("--subset", choices=("validation", "experiment_test"),
                        default="experiment_test")
    args = parser.parse_args()
    if args.max_windows_per_run < 1:
        parser.error("max-windows-per-run must be positive")
    if (args.high_steering_threshold is not None
            and args.high_steering_threshold < 0.0):
        parser.error("high-steering-threshold must be nonnegative")
    if args.run_id and args.run_selection_report:
        parser.error("--run-id cannot be combined with --run-selection-report")
    try:
        evaluate(args.dataset, args.run_dir, args.max_windows_per_run,
                 args.device, args.subset, args.run_selection_report,
                 args.high_steering_threshold, args.run_id, args.output_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"evaluation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
