#!/usr/bin/env python3
"""Paired RSSM context-length comparison on identical future windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.vehicle_dynamics_learning.train_direct_sequence_teacher import (
    COM_X_M,
    DT_S,
    _eligible_sequences,
    _pose_rollout,
    _targets,
)
from tools.vehicle_dynamics_learning.train_nssm import _load_dataset, _torch
from tools.vehicle_dynamics_learning.train_rssm_teacher import _rssm_model


def compare(checkpoints: list[Path], dataset_path: Path, output_path: Path,
            *, split: str = "validation", device_name: str = "cpu",
            max_windows_per_run: int = 24, seed: int = 20261011) -> dict:
    if len(checkpoints) < 2:
        raise ValueError("at least two context checkpoints are required")
    torch, nn = _torch()
    torch.set_num_threads(1)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = _load_dataset(dataset_path)
    targets = _targets(data)
    payloads = [torch.load(path, map_location="cpu", weights_only=False)
                for path in checkpoints]
    first_meta = payloads[0]["metadata"]
    rollout_steps = int(first_meta["rollout_steps"])
    if any(int(payload["metadata"]["rollout_steps"]) != rollout_steps
           for payload in payloads):
        raise ValueError("all context candidates must use one rollout horizon")
    contexts = [int(payload["metadata"]["context_steps"])
                for payload in payloads]
    if len(set(contexts)) != len(contexts):
        raise ValueError("context lengths must be unique")
    max_context = max(contexts)
    groups = [_eligible_sequences(
        data, targets, split, context, rollout_steps) for context in contexts]
    common_runs = set.intersection(*[
        {run for run in group} for group in groups])
    if not common_runs:
        raise ValueError("no whole-run split is eligible for every context")
    training_runs = [set(payload["metadata"]["training_runs"])
                     for payload in payloads]
    run_names = {str(data["run_ids"][run]) for run in common_runs}
    for candidate_runs in training_runs:
        overlap = run_names & candidate_runs
        if overlap:
            raise ValueError(f"comparison split overlaps training: {sorted(overlap)}")

    common_sequences: dict[int, list[tuple[int, int]]] = {}
    for run in sorted(common_runs):
        sequence_lists = [set(group[run]) for group in groups]
        common = sorted(set.intersection(*sequence_lists))
        if common:
            common_sequences[run] = common
    if not common_sequences:
        raise ValueError("common eligible run IDs have no common sequences")

    rng = np.random.default_rng(seed)
    windows: dict[int, list[int]] = {}
    for run, sequences in common_sequences.items():
        candidates = []
        for start, end in sequences:
            low = start + max_context - 1
            high = end - rollout_steps - 1
            if high >= low:
                candidates.extend(range(low, high + 1))
        if not candidates:
            continue
        if len(candidates) > max_windows_per_run:
            selected = rng.choice(candidates, size=max_windows_per_run,
                                  replace=False)
            windows[run] = sorted(int(value) for value in selected)
        else:
            windows[run] = candidates
    windows = {run: starts for run, starts in windows.items() if starts}
    if not windows:
        raise ValueError("no common future windows available")

    models = []
    reference_scale = payloads[contexts.index(max_context)]["y_scale"][:3]
    for payload in payloads:
        metadata = payload["metadata"]
        if metadata["feature_names"] != data["feature_names"]:
            raise ValueError("candidate and dataset feature layouts differ")
        model_type = _rssm_model(
            torch, nn, int(metadata["hidden_size"]),
            int(metadata["latent_size"]), payload["x_mean"],
            payload["x_scale"], payload["y_mean"], payload["y_scale"])
        model = model_type().to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)

    per_model = {}
    for context, payload, model, checkpoint_path in zip(
            contexts, payloads, models, checkpoints):
        per_run = {}
        with torch.no_grad():
            for run, starts in windows.items():
                accum = {}
                for future_start in starts:
                    context_rows = data["frames"][future_start - context:
                                                   future_start]
                    commands = data["frames"][future_start:
                                              future_start + rollout_steps, 7:9]
                    context_norm = ((context_rows - payload["x_mean"])
                                    / payload["x_scale"])
                    commands_norm = ((commands - payload["x_mean"][7:9])
                                     / payload["x_scale"][7:9])
                    prediction = model(
                        torch.as_tensor(context_norm[None], dtype=torch.float32,
                                        device=device),
                        torch.as_tensor(commands_norm[None], dtype=torch.float32,
                                        device=device))[0].cpu().numpy()
                    prediction = (prediction * payload["y_scale"]
                                  + payload["y_mean"])
                    truth = targets[future_start:
                                    future_start + rollout_steps]
                    initial_state = targets[future_start - 1, :3]
                    initial_pose = data["simulator_pose_xyyaw"][future_start - 1]
                    truth_pose = data["simulator_pose_xyyaw"][
                        future_start:future_start + rollout_steps]
                    predicted_pose = _pose_rollout(
                        prediction, initial_state, initial_pose)
                    poses_valid = (np.isfinite(initial_pose).all()
                                   and np.isfinite(truth_pose).all())
                    for step in (10, 30, 40, rollout_steps):
                        if step > rollout_steps:
                            continue
                        state_error = (prediction[step - 1, :3]
                                       - truth[step - 1, :3])
                        row = accum.setdefault(
                            step, {"state": [], "pose": []})
                        row["state"].append(state_error)
                        if poses_valid:
                            pose_error = (predicted_pose[step - 1]
                                          - truth_pose[step - 1])
                            pose_error[2] = np.arctan2(
                                np.sin(pose_error[2]), np.cos(pose_error[2]))
                            row["pose"].append(pose_error)
                horizons = {}
                for step, values in accum.items():
                    state_error = np.asarray(values["state"])
                    pose_error = np.asarray(values["pose"])
                    horizon_report = {
                        "body_state_rmse": np.sqrt(
                            np.mean(state_error ** 2, axis=0)).tolist(),
                        "common_scale_normalized_body_rmse": float(np.sqrt(
                            np.mean((state_error / reference_scale) ** 2))),
                        "window_count": len(state_error),
                    }
                    if len(pose_error):
                        horizon_report["pose_xy_rmse_m"] = np.sqrt(
                            np.mean(pose_error[:, :2] ** 2, axis=0)).tolist()
                        horizon_report["heading_rmse_rad"] = float(np.sqrt(
                            np.mean(pose_error[:, 2] ** 2)))
                    horizons[f"{step * DT_S:g}s"] = horizon_report
                per_run[str(data["run_ids"][run])] = {
                    "window_count": len(starts), "horizons": horizons}
        score_keys = [f"{step * DT_S:g}s" for step in (10, 30, 40, rollout_steps)
                      if step <= rollout_steps]
        run_scores = {
            run: float(np.mean([
                row["horizons"][key]["common_scale_normalized_body_rmse"]
                for key in score_keys]))
            for run, row in per_run.items()
        }
        per_model[str(context * DT_S)] = {
            "context_seconds": context * DT_S,
            "checkpoint": str(checkpoint_path.resolve()),
            "reference_state_scale": np.asarray(reference_scale).tolist(),
            "per_run": per_run,
            "run_mean_selection_score": run_scores,
            "macro_run_score": float(np.mean(list(run_scores.values()))),
        }

    report = {
        "schema_version": 1,
        "dataset": str(dataset_path.resolve()),
        "split": split,
        "future_seconds": rollout_steps * DT_S,
        "common_eligible_runs": sorted(run_names),
        "common_window_counts_by_run": {
            str(data["run_ids"][run]): len(starts)
            for run, starts in windows.items()},
        "window_context_alignment": (
            "all candidates use identical future-start samples and command traces; "
            "shorter contexts use the trailing portion of the longest context"),
        "common_normalizer": "longest-context candidate's train-only body-state scale",
        "models": per_model,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-windows-per-run", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    report = compare(
        args.checkpoints, args.dataset, args.output, split=args.split,
        device_name=args.device, max_windows_per_run=args.max_windows_per_run,
        seed=args.seed)
    print(json.dumps({
        "common_runs": report["common_eligible_runs"],
        "scores": {context: model["macro_run_score"]
                   for context, model in report["models"].items()},
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
