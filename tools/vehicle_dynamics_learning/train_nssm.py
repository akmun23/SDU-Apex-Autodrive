#!/usr/bin/env python3
"""Train a GPU-capable, causal recurrent vehicle transition surrogate.

This first learner is an oracle-current-state plant experiment: observed
body-state history is offline bridge truth. It predicts body velocity,
actuator feedback, and rear wheel-surface speeds recursively under the logged
future command sequence. No future truth is injected after rollout starts.
This does not test whether legal sensor-only history can estimate body speed.
Entire ROS-bag runs define splits; individual 40 Hz rows are never randomly
split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STATE_COUNT = 7
FEATURE_COUNT = 9
DEFAULT_HORIZONS = (1, 4, 10, 20, 30)


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install the hardware-appropriate "
            "stable build using https://pytorch.org/get-started/locally/"
        ) from exc
    return torch, nn


def _model_type(torch, nn, hidden_size: int):
    class RecurrentTransition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cell = nn.GRUCell(FEATURE_COUNT, hidden_size)
            self.rate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, STATE_COUNT),
            )

        def advance(self, feature, hidden, dt):
            hidden = self.cell(feature, hidden)
            rate_normalized_per_s = self.rate(hidden)
            state_next = feature[:, :STATE_COUNT] + dt[:, None] * rate_normalized_per_s
            return state_next, hidden

    return RecurrentTransition


def _load_dataset(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    required = {"schema_version", "feature_names", "frames", "dt_s",
                "sequence_bounds", "sequence_run_index", "run_ids", "run_splits"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"dataset missing arrays: {sorted(missing)}")
    if int(data["schema_version"][0]) not in (1, 2):
        raise ValueError(f"unsupported schema version {data['schema_version']}")
    frames = data["frames"].astype(np.float32, copy=False)
    dt_s = data["dt_s"].astype(np.float32, copy=False)
    bounds = data["sequence_bounds"].astype(np.int64, copy=False)
    seq_run = data["sequence_run_index"].astype(np.int32, copy=False)
    run_ids = data["run_ids"].astype(str)
    splits = data["run_splits"].astype(str)
    if frames.ndim != 2 or frames.shape[1] != FEATURE_COUNT:
        raise ValueError(f"expected frames [N,{FEATURE_COUNT}], got {frames.shape}")
    if dt_s.shape != (len(frames),) or not np.isfinite(frames).all():
        raise ValueError("dataset contains invalid frame/dt array shapes or non-finite values")
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) != len(seq_run):
        raise ValueError("sequence bounds and run-index arrays disagree")
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > len(frames)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("sequence bounds exceed the frame array")
    if len(run_ids) != len(splits) or np.any(seq_run < 0) or np.any(seq_run >= len(run_ids)):
        raise ValueError("run metadata indices are invalid")
    return {
        "frames": frames,
        "dt_s": dt_s,
        "bounds": bounds,
        "seq_run": seq_run,
        "run_ids": run_ids,
        "splits": splits,
        "feature_names": data["feature_names"].astype(str).tolist(),
    }


def _sequence_groups(data: dict[str, Any], split: str,
                     history_steps: int, rollout_steps: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for seq_id, (start, end) in enumerate(data["bounds"]):
        run = int(data["seq_run"][seq_id])
        if data["splits"][run] != split:
            continue
        if int(end - start) < history_steps + rollout_steps:
            continue
        groups.setdefault(run, []).append(seq_id)
    return groups


def _candidate_start(data: dict[str, Any], seq_id: int,
                     history_steps: int, rollout_steps: int,
                     rng: np.random.Generator) -> int:
    start, end = map(int, data["bounds"][seq_id])
    low = start + history_steps - 1
    high = end - rollout_steps - 1
    if high < low:
        raise ValueError(f"sequence {seq_id} too short for the requested window")
    return int(rng.integers(low, high + 1))


def _sample_batch(data: dict[str, Any], groups: dict[int, list[int]], batch_size: int,
                  history_steps: int, rollout_steps: int,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_ids = np.asarray(sorted(groups), dtype=np.int32)
    histories, futures, dts = [], [], []
    for run in rng.choice(run_ids, size=batch_size, replace=True):
        seq_id = int(rng.choice(groups[int(run)]))
        start = _candidate_start(data, seq_id, history_steps, rollout_steps, rng)
        hist_begin = start - history_steps + 1
        next_ids = np.arange(start + 1, start + rollout_steps + 1, dtype=np.int64)
        histories.append(data["frames"][hist_begin:start + 1])
        futures.append(data["frames"][next_ids])
        dts.append(data["dt_s"][next_ids])
    return (np.asarray(histories, dtype=np.float32),
            np.asarray(futures, dtype=np.float32),
            np.asarray(dts, dtype=np.float32))


def _fixed_eval_windows(data: dict[str, Any], groups: dict[int, list[int]],
                        history_steps: int, rollout_steps: int,
                        max_windows: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for run in sorted(groups):
        for seq_id in groups[run]:
            start, end = map(int, data["bounds"][seq_id])
            low = start + history_steps - 1
            high = end - rollout_steps - 1
            if high < low:
                continue
            # Fixed, evenly spaced windows provide reproducible validation.
            count = min(4, high - low + 1)
            positions = np.linspace(low, high, count, dtype=np.int64)
            candidates.extend((seq_id, int(pos)) for pos in positions)
    if len(candidates) > max_windows:
        indices = np.linspace(0, len(candidates) - 1, max_windows, dtype=np.int64)
        candidates = [candidates[i] for i in indices]
    return candidates


def _batch_from_windows(data: dict[str, Any], windows: list[tuple[int, int]],
                        history_steps: int, rollout_steps: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    histories, futures, dts = [], [], []
    for seq_id, start in windows:
        seq_begin, _ = map(int, data["bounds"][seq_id])
        begin = start - history_steps + 1
        next_ids = np.arange(start + 1, start + rollout_steps + 1, dtype=np.int64)
        histories.append(data["frames"][begin:start + 1])
        futures.append(data["frames"][next_ids])
        dts.append(data["dt_s"][next_ids])
    return (np.asarray(histories, dtype=np.float32),
            np.asarray(futures, dtype=np.float32),
            np.asarray(dts, dtype=np.float32))


def _normalizers(data: dict[str, Any], train_groups: dict[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    train_runs = set(train_groups)
    indices = np.flatnonzero(np.isin(data["seq_run"], np.fromiter(train_runs, dtype=np.int32)))
    selected: list[np.ndarray] = []
    for seq_id in indices:
        start, end = map(int, data["bounds"][seq_id])
        selected.append(data["frames"][start:end])
    if not selected:
        raise ValueError("training split has no samples")
    values = np.concatenate(selected, axis=0).astype(np.float64, copy=False)
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, np.asarray([0.05, 0.05, 0.05, 0.02, 0.02,
                                         0.10, 0.10, 0.02, 0.02], dtype=np.float32))
    return mean, scale


def _tensor_batch(torch, arrays, mean: np.ndarray, scale: np.ndarray, device):
    histories, futures, dts = arrays
    histories = (histories - mean[None, None, :]) / scale[None, None, :]
    futures = (futures - mean[None, None, :]) / scale[None, None, :]
    return (
        torch.as_tensor(histories, dtype=torch.float32, device=device),
        torch.as_tensor(futures, dtype=torch.float32, device=device),
        torch.as_tensor(dts, dtype=torch.float32, device=device),
    )


def _rollout(model, history, future, dts, history_steps: int):
    torch, _ = _torch()
    batch = history.shape[0]
    hidden = torch.zeros(batch, model.cell.hidden_size,
                         dtype=history.dtype, device=history.device)
    # Burn in on prior observed history; then predict recursively from the
    # final observed frame. Ground truth states are not injected after t0.
    for index in range(history_steps - 1):
        hidden = model.cell(history[:, index, :], hidden)
    feature = history[:, -1, :]
    predicted = []
    for index in range(future.shape[1]):
        state_next, hidden = model.advance(feature, hidden, dts[:, index])
        predicted.append(state_next)
        next_command = future[:, index, 7:9]
        feature = torch.cat((state_next, next_command), dim=1)
    return torch.stack(predicted, dim=1)


def _score_model(torch, model, arrays, mean, scale, device,
                 history_steps: int) -> dict[str, Any]:
    history, future, dts = _tensor_batch(torch, arrays, mean, scale, device)
    model.eval()
    with torch.no_grad():
        pred_norm = _rollout(model, history, future, dts, history_steps)
        target_norm = future[:, :, :STATE_COUNT]
        errors = (pred_norm - target_norm) * torch.as_tensor(
            scale[:STATE_COUNT], dtype=torch.float32, device=device)
        physical_future = future[:, :, :STATE_COUNT] * torch.as_tensor(
            scale[:STATE_COUNT], dtype=torch.float32, device=device) + torch.as_tensor(
                mean[:STATE_COUNT], dtype=torch.float32, device=device)
        initial = history[:, -1, :STATE_COUNT] * torch.as_tensor(
            scale[:STATE_COUNT], dtype=torch.float32, device=device) + torch.as_tensor(
                mean[:STATE_COUNT], dtype=torch.float32, device=device)
        previous = history[:, -2, :STATE_COUNT] * torch.as_tensor(
            scale[:STATE_COUNT], dtype=torch.float32, device=device) + torch.as_tensor(
                mean[:STATE_COUNT], dtype=torch.float32, device=device)
        persistence = initial[:, None, :].expand_as(physical_future)
        recent_dt = torch.clamp(torch.median(dts[:, :4], dim=1).values, 0.015, 0.075)
        trend_rate = (initial - previous) / recent_dt[:, None]
        trend = initial[:, None, :] + torch.cumsum(dts, dim=1)[:, :, None] * trend_rate[:, None, :]
        persistence_error = persistence - physical_future
        trend_error = trend - physical_future
        physical_errors = errors
        horizons: dict[str, Any] = {}
        for step in DEFAULT_HORIZONS:
            if step > errors.shape[1]:
                continue
            subset = errors[:, step - 1, :]
            rmse = torch.sqrt(torch.mean(subset ** 2, dim=0)).cpu().numpy()
            horizons[str(step)] = {
                "seconds_median": float(np.median(np.sum(dts[:, :step].cpu().numpy(), axis=1))),
                "rmse": {name: float(value) for name, value in zip(
                    ("u_mps", "v_mps", "yaw_rate_rps", "steering_rad",
                    "throttle_norm", "rear_left_surface_mps", "rear_right_surface_mps"), rmse)},
            }
        normalized_rmse = torch.sqrt(torch.mean((pred_norm - target_norm) ** 2, dim=(0, 1))).cpu().numpy()
        state_names = ("u_mps", "v_mps", "yaw_rate_rps", "steering_rad",
                       "throttle_norm", "rear_left_surface_mps", "rear_right_surface_mps")
        baselines: dict[str, Any] = {"persistence": {}, "constant_recent_trend": {}}
        for step in DEFAULT_HORIZONS:
            if step > errors.shape[1]:
                continue
            persist_rmse = torch.sqrt(torch.mean(
                persistence_error[:, step - 1, :] ** 2, dim=0)).cpu().numpy()
            trend_rmse = torch.sqrt(torch.mean(
                trend_error[:, step - 1, :] ** 2, dim=0)).cpu().numpy()
            baselines["persistence"][str(step)] = {
                name: float(value) for name, value in zip(state_names, persist_rmse)}
            baselines["constant_recent_trend"][str(step)] = {
                name: float(value) for name, value in zip(state_names, trend_rmse)}
        # Overall score is not sufficient for this car: retain the demanding
        # high-speed/high-steering strata as explicit acceptance evidence.
        last = min(30, physical_errors.shape[1]) - 1
        initial_speed = torch.linalg.vector_norm(initial[:, :2], dim=1).cpu().numpy()
        initial_steer = torch.abs(
            history[:, -1, 3] * scale[3] + mean[3]).cpu().numpy()
        error_np = physical_errors[:, last, :3].cpu().numpy()
        speed_edges = (0.0, 2.0, 4.0, 6.0, 8.0, float("inf"))
        steer_edges = (0.0, 0.15, 0.30, 0.42, 0.50, float("inf"))
        speed_bins = np.digitize(initial_speed, speed_edges[1:-1], right=False)
        steer_bins = np.digitize(initial_steer, steer_edges[1:-1], right=False)
        speed_steer: dict[str, Any] = {}
        for i in range(len(speed_edges) - 1):
            for j in range(len(steer_edges) - 1):
                selected = (speed_bins == i) & (steer_bins == j)
                if not selected.any():
                    continue
                rmse = np.sqrt(np.mean(error_np[selected] ** 2, axis=0))
                speed_steer[f"speed_{speed_edges[i]}_{speed_edges[i + 1]}__steer_{steer_edges[j]}_{steer_edges[j + 1]}"] = {
                    "windows": int(np.count_nonzero(selected)),
                    "rmse_at_30_steps": dict(zip(("u_mps", "v_mps", "yaw_rate_rps"),
                                                  map(float, rmse))),
                }
    return {"windows": int(history.shape[0]), "horizons": horizons,
            "baselines": baselines, "speed_steer_bins_at_30_steps": speed_steer,
            "mean_normalized_rmse": float(normalized_rmse.mean()),
            "normalized_rmse_by_state": normalized_rmse.tolist()}


def _score_ensemble(torch, models, arrays, mean, scale, device,
                    history_steps: int) -> dict[str, Any]:
    """Score ensemble mean and whether member spread tracks held-out error."""
    history, future, dts = _tensor_batch(torch, arrays, mean, scale, device)
    scale_tensor = torch.as_tensor(scale[:STATE_COUNT], dtype=torch.float32,
                                   device=device)
    mean_tensor = torch.as_tensor(mean[:STATE_COUNT], dtype=torch.float32,
                                  device=device)
    target = future[:, :, :STATE_COUNT] * scale_tensor + mean_tensor
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            normalized = _rollout(model, history, future, dts, history_steps)
            predictions.append(normalized * scale_tensor + mean_tensor)
    ensemble = torch.stack(predictions, dim=0)
    prediction = ensemble.mean(dim=0)
    spread = ensemble.std(dim=0, unbiased=False)
    error = prediction - target
    names = ("u_mps", "v_mps", "yaw_rate_rps", "steering_rad",
             "throttle_norm", "rear_left_surface_mps", "rear_right_surface_mps")
    result: dict[str, Any] = {"members": len(models), "horizons": {}}
    for step in DEFAULT_HORIZONS:
        if step > error.shape[1]:
            continue
        e = error[:, step - 1, :].cpu().numpy()
        s = spread[:, step - 1, :].cpu().numpy()
        channels: dict[str, Any] = {}
        for channel, name in enumerate(names):
            absolute_error = np.abs(e[:, channel])
            uncertainty = s[:, channel]
            log_error = np.log(np.maximum(absolute_error, 1.0e-6))
            log_uncertainty = np.log(np.maximum(uncertainty, 1.0e-6))
            correlation = (
                float(np.corrcoef(log_error, log_uncertainty)[0, 1])
                if len(models) > 1 and len(log_error) > 1
                and np.std(log_error) > 1.0e-8
                and np.std(log_uncertainty) > 1.0e-8
                else None)
            channels[name] = {
                "ensemble_mean_rmse": float(np.sqrt(np.mean(e[:, channel] ** 2))),
                "mean_member_spread": (float(np.mean(uncertainty))
                                        if len(models) > 1 else None),
                "error_within_1_member_sd": (float(np.mean(
                    absolute_error <= uncertainty)) if len(models) > 1 else None),
                "error_within_2_member_sd": (float(np.mean(
                    absolute_error <= 2.0 * uncertainty)) if len(models) > 1 else None),
                "log_spread_log_abs_error_correlation": correlation,
            }
        result["horizons"][str(step)] = channels
    return result


def _save_checkpoint(torch, path: Path, model, mean, scale, metadata: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "feature_mean": mean,
                "feature_scale": scale, "metadata": metadata}, temporary)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, nn = _torch()
    data = _load_dataset(args.dataset)
    history_steps = args.history_steps
    rollout_steps = args.rollout_steps
    if history_steps < 2 or rollout_steps < max(DEFAULT_HORIZONS):
        raise ValueError("history must be >=2 and rollout must cover all score horizons")
    train_groups = _sequence_groups(data, "train", history_steps, rollout_steps)
    validation_groups = _sequence_groups(data, "validation", history_steps, rollout_steps)
    if not train_groups:
        raise ValueError("no sufficiently long sequences in train split")
    if not validation_groups:
        raise ValueError("no sufficiently long sequences in validation split")
    test_groups = _sequence_groups(data, "final_test", history_steps, rollout_steps)
    other_holdout_groups = _sequence_groups(data, "test", history_steps, rollout_steps)
    mean, scale = _normalizers(data, train_groups)
    val_windows = _fixed_eval_windows(data, validation_groups, history_steps,
                                      rollout_steps, args.eval_windows)
    if not val_windows:
        raise ValueError("validation split has no rollout windows")
    val_arrays = _batch_from_windows(data, val_windows, history_steps, rollout_steps)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    RecurrentTransition = _model_type(torch, nn, args.hidden_size)
    deadline = time.monotonic() + args.time_budget_hours * 3600.0
    total_steps = 0
    member_results: list[dict[str, Any]] = []
    member_models = []
    for member in range(args.members):
        if time.monotonic() >= deadline:
            break
        member_started = time.monotonic()
        remaining_members = args.members - member
        member_deadline = min(
            deadline,
            member_started + max(0.0, deadline - member_started) / remaining_members)
        seed = args.seed + member * 1009
        random.seed(seed)
        np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = RecurrentTransition().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        best_score = math.inf
        best_step = 0
        stale_evaluations = 0
        member_steps = 0
        checkpoint = args.output_dir / f"member_{member:02d}.pt"
        history: list[dict[str, Any]] = []
        while (time.monotonic() < member_deadline and
               member_steps < args.max_steps_per_member):
            model.train()
            batches = _sample_batch(data, train_groups, args.batch_size,
                                    history_steps, rollout_steps, np_rng)
            hist, future, dts = _tensor_batch(torch, batches, mean, scale, device)
            prediction = _rollout(model, hist, future, dts, history_steps)
            target = future[:, :, :STATE_COUNT]
            loss = nn.functional.smooth_l1_loss(prediction, target, beta=0.05)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            member_steps += 1
            total_steps += 1
            if member_steps % args.eval_every != 0 and member_steps != 1:
                continue
            score = _score_model(torch, model, val_arrays, mean, scale,
                                 device, history_steps)
            score_value = score["mean_normalized_rmse"]
            history.append({"step": member_steps,
                            "train_loss": float(loss.detach().cpu()),
                            "validation": score})
            print(f"member={member + 1}/{args.members} step={member_steps} "
                  f"train={float(loss.detach().cpu()):.6f} "
                  f"val_nrmse={score_value:.5f} device={device}", flush=True)
            if score_value < best_score:
                best_score = score_value
                best_step = member_steps
                stale_evaluations = 0
                _save_checkpoint(torch, checkpoint, model, mean, scale, {
                    "schema_version": 1, "member": member,
                    "seed": seed, "step": member_steps,
                    "feature_names": data["feature_names"],
                    "state_names": data["feature_names"][:STATE_COUNT],
                    "history_steps": history_steps,
                    "rollout_steps": rollout_steps,
                    "hidden_size": args.hidden_size,
                    "training_runs": [data["run_ids"][i] for i in sorted(train_groups)],
                })
            else:
                stale_evaluations += 1
                if stale_evaluations >= args.patience:
                    break
        if checkpoint.is_file():
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["state_dict"])
            member_models.append(model)
            best_validation = _score_model(torch, model, val_arrays, mean,
                                           scale, device, history_steps)
        else:
            best_validation = None
        member_results.append({
            "member": member,
            "seed": seed,
            "steps": member_steps,
            "elapsed_seconds": time.monotonic() - member_started,
            "budget_seconds": member_deadline - member_started,
            "best_step": best_step,
            "best_validation": best_validation,
            "history": history,
            "checkpoint": checkpoint.name if checkpoint.is_file() else None,
        })

    if not member_results:
        raise ValueError("time budget expired before the first training step")
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "features": data["feature_names"],
        "predicted_state": data["feature_names"][:STATE_COUNT],
        "identification_question": "oracle-current-state plant predictability; this does not test sensor-only state estimation",
        "ground_truth_policy": "offline body-state history and supervised target; no future truth injected after rollout start",
        "future_command_policy": "recorded command channel is used as the known future control trace during evaluation; no future sensor/state values are fed back",
        "split_counts": {split: int(np.count_nonzero(data["splits"] == split))
                         for split in np.unique(data["splits"])},
        "sequence_counts": {
            split: sum(len(group) for run, group in groups.items()
                       if data["splits"][run] == split)
            for split, groups in (("train", train_groups), ("validation", validation_groups))
        },
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "configuration": {
            "members_requested": args.members,
            "members_completed": len(member_results),
            "history_steps": history_steps,
            "rollout_steps": rollout_steps,
            "hidden_size": args.hidden_size,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "time_budget_hours": args.time_budget_hours,
            "max_steps_per_member": args.max_steps_per_member,
            "seed": args.seed,
        },
        "total_optimizer_steps": total_steps,
        "members": member_results,
        "validation_ensemble": _score_ensemble(
            torch, member_models, val_arrays, mean, scale, device, history_steps),
    }
    if args.score_final_test:
        test_sets = {}
        if other_holdout_groups:
            test_sets["named_holdouts"] = other_holdout_groups
        if test_groups:
            test_sets["final_test_20260929"] = test_groups
        if not test_sets:
            raise ValueError("no test-only sequences available")
        for split_name, groups in test_sets.items():
            test_windows = _fixed_eval_windows(data, groups, history_steps,
                                               rollout_steps, args.eval_windows)
            test_arrays = _batch_from_windows(data, test_windows, history_steps,
                                              rollout_steps)
            test_results = []
            scored_models = []
            for member, result in enumerate(member_results):
                if not result["checkpoint"]:
                    continue
                payload = torch.load(args.output_dir / result["checkpoint"],
                                     map_location=device, weights_only=False)
                model = RecurrentTransition().to(device)
                model.load_state_dict(payload["state_dict"])
                scored_models.append(model)
                test_results.append({"member": member,
                                     "score": _score_model(
                                         torch, model, test_arrays, mean,
                                         scale, device, history_steps)})
            report[split_name] = {
                "members": test_results,
                "ensemble": _score_ensemble(
                    torch, scored_models, test_arrays, mean, scale,
                    device, history_steps),
            }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"wrote {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--time-budget-hours", type=float, default=8.0,
                        help="total wall-clock budget shared by all ensemble members")
    parser.add_argument("--max-steps-per-member", type=int, default=200000)
    parser.add_argument("--history-steps", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-windows", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--score-final-test", action="store_true",
                        help="evaluate the untouched final-test runs once after training")
    args = parser.parse_args()
    if args.members < 1 or args.time_budget_hours <= 0.0 or args.max_steps_per_member < 1:
        parser.error("members, time budget, and step cap must be positive")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
