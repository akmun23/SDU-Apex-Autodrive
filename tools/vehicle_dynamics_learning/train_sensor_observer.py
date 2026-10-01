#!/usr/bin/env python3
"""Fit a causal sensor-history observer for rear-axle body velocity.

Inputs are encoder-derived rear wheel speeds, IMU acceleration/yaw rate,
actuator feedback, commands, and sample interval. Offline bridge odometry is
used only as the supervised label. The model receives each sensor sample once
and estimates the contemporaneous body state; no odometry/truth feature is an
input and no future sensor or label is used.
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

from tools.vehicle_dynamics_learning.family_condition_sampler import (
    build_sequence_sampler,
)


STATE_NAMES = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")
LEARNED_STATE_NAMES = STATE_NAMES[:2]
DEFAULT_BURN_IN = 16
DEFAULT_SCORE_STEPS = 32
SENSOR_PERIOD_S = 0.025


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required. Install the hardware-appropriate stable build "
            "from https://pytorch.org/get-started/locally/"
        ) from exc
    return torch, nn


def _load_dataset(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    required = {"schema_version", "frames", "sensor_frames", "sensor_valid",
                "dt_s", "sequence_bounds", "sequence_run_index", "run_ids",
                "run_splits", "sensor_feature_names"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"dataset missing arrays: {sorted(missing)}")
    schema_version = int(data["schema_version"][0])
    if schema_version not in (2, 3, 4, 5, 6):
        raise ValueError("sensor observer requires dataset schema version 2 through 6")
    frames = data["frames"].astype(np.float32, copy=False)
    sensors = data["sensor_frames"].astype(np.float32, copy=False)
    valid = data["sensor_valid"].astype(bool, copy=False)
    dt_s = data["dt_s"].astype(np.float32, copy=False)
    bounds = data["sequence_bounds"].astype(np.int64, copy=False)
    seq_run = data["sequence_run_index"].astype(np.int32, copy=False)
    run_ids = data["run_ids"].astype(str)
    splits = data["run_splits"].astype(str)
    if frames.ndim != 2 or frames.shape[1] < len(STATE_NAMES):
        raise ValueError("ground-truth state array must contain u, v, r")
    if sensors.ndim != 2 or sensors.shape[1] != 10:
        raise ValueError("expected 10 causal sensor features per sample")
    if valid.shape != (len(frames),) or sensors.shape[0] != len(frames):
        raise ValueError("sensor features/validity must align with truth frames")
    if dt_s.shape != (len(frames),) or not np.isfinite(frames).all():
        raise ValueError("invalid truth or sample-interval array")
    if schema_version >= 4:
        if "packet_sequence" not in data.files:
            raise ValueError("fixed-timebase observer dataset lacks packet_sequence")
        packet_sequence = data["packet_sequence"].astype(np.int64, copy=False)
        if (packet_sequence.shape != (len(frames),)
                or not np.allclose(dt_s, 0.025, rtol=0.0, atol=1e-7)):
            raise ValueError("fixed-timebase observer data must use exact 25 ms packet time")
        for start_raw, end_raw in data["sequence_bounds"]:
            start, end = int(start_raw), int(end_raw)
            if np.any(np.diff(packet_sequence[start:end]) != 1):
                raise ValueError("observer sequence contains a simulator packet gap")
    if not np.isfinite(sensors[valid]).all():
        raise ValueError("valid sensor rows contain non-finite values")
    if bounds.ndim != 2 or bounds.shape[1] != 2 or len(bounds) != len(seq_run):
        raise ValueError("sequence bounds and run-index arrays disagree")
    if len(run_ids) != len(splits) or np.any(seq_run < 0) or np.any(seq_run >= len(run_ids)):
        raise ValueError("run metadata indices are invalid")
    return {"truth": frames[:, :3], "sensors": sensors, "valid": valid,
            "dt_s": dt_s, "bounds": bounds, "seq_run": seq_run,
            "run_ids": run_ids, "splits": splits,
            "sensor_feature_names": data["sensor_feature_names"].astype(str).tolist()}


def _valid_segments(data: dict[str, Any], split: str, window_steps: int
                    ) -> dict[int, list[tuple[int, int]]]:
    """Return absolute [start,end) sensor-valid intervals, grouped by run."""
    grouped: dict[int, list[tuple[int, int]]] = {}
    for seq_id, (start_raw, end_raw) in enumerate(data["bounds"]):
        run = int(data["seq_run"][seq_id])
        if data["splits"][run] != split:
            continue
        start, end = int(start_raw), int(end_raw)
        dt = data["dt_s"][start:end]
        good = (data["valid"][start:end]
                & (dt >= 0.015) & (dt <= 0.075)
                & np.isfinite(data["truth"][start:end]).all(axis=1))
        # A missing sensor/time point invalidates the recurrent history. Start
        # a new episode after it; never interpolate across it.
        changes = np.diff(np.r_[False, good, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        for local_start, local_end in zip(starts, ends):
            if local_end - local_start >= window_steps:
                grouped.setdefault(run, []).append(
                    (start + int(local_start), start + int(local_end)))
    return grouped


def _candidate(groups: dict[int, list[tuple[int, int]]], rng: np.random.Generator,
               window_steps: int) -> tuple[int, int]:
    runs = np.asarray(sorted(groups), dtype=np.int32)
    run = int(rng.choice(runs))
    segments = groups[run]
    segment = segments[int(rng.integers(0, len(segments)))]
    low, end = segment
    start = int(rng.integers(low, end - window_steps + 1))
    return start, start + window_steps


def _sample_batch(data: dict[str, Any], groups: dict[int, list[tuple[int, int]]],
                  batch_size: int, window_steps: int,
                  rng: np.random.Generator,
                  family_sampler=None) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    for _ in range(batch_size):
        if family_sampler is None:
            start, end = _candidate(groups, rng, window_steps)
        else:
            _, (segment_start, segment_end), _, _ = family_sampler.sample(rng)
            start = int(rng.integers(segment_start,
                                     segment_end - window_steps + 1))
            end = start + window_steps
        inputs.append(data["sensors"][start:end])
        targets.append(data["truth"][start:end])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _training_weights(inputs: np.ndarray, targets: np.ndarray,
                      mode: str, boost: float) -> np.ndarray:
    """Weight nonlinear training intervals without changing model inputs.

    The state labels are used only to select/weight examples during training;
    they are never supplied to the observer. Steering/encoder residual
    thresholds target measured high-response and wheel-spin regimes.
    """
    weights = np.ones(inputs.shape[:2], dtype=np.float32)
    if mode == "uniform":
        return weights

    speed = np.hypot(targets[:, :, 0], targets[:, :, 1])
    abs_steer = np.abs(inputs[:, :, 0])
    encoder_mean = np.mean(inputs[:, :, 2:4], axis=2)
    encoder_residual = np.abs(encoder_mean - targets[:, :, 0])
    high_response = (((abs_steer >= 0.10) & (speed >= 5.5))
                     | (abs_steer >= 0.30))
    encoder_mismatch = ((encoder_residual >= 0.50)
                        & (np.abs(inputs[:, :, 1]) >= 0.15))
    if mode == "high_response":
        selected = high_response
    elif mode == "encoder_mismatch":
        selected = encoder_mismatch
    elif mode == "combined":
        selected = high_response | encoder_mismatch
    else:
        raise ValueError(f"unknown training weight mode: {mode}")
    weights[selected] = boost
    return weights


def _eval_windows(groups: dict[int, list[tuple[int, int]]],
                  window_steps: int, max_windows: int
                  ) -> dict[int, list[tuple[int, int]]]:
    result: dict[int, list[tuple[int, int]]] = {}
    for run, segments in sorted(groups.items()):
        candidates = []
        for low, end in segments:
            last = end - window_steps
            if last < low:
                continue
            n = min(4, last - low + 1)
            for start in np.linspace(low, last, n, dtype=np.int64):
                candidates.append((int(start), int(start) + window_steps))
        if len(candidates) > max_windows:
            indices = np.linspace(0, len(candidates) - 1, max_windows, dtype=np.int64)
            candidates = [candidates[i] for i in indices]
        if candidates:
            result[run] = candidates
    return result


def _make_model(nn, input_size: int, hidden_size: int):
    class SensorObserver(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.GRU(input_size, hidden_size, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                      nn.SiLU(), nn.Linear(hidden_size, 2))

        def forward(self, inputs):
            hidden, _ = self.recurrent(inputs)
            return self.head(hidden)
    return SensorObserver


def _normalizers(data: dict[str, Any], train_groups: dict[int, list[tuple[int, int]]]):
    sensor_parts, truth_parts = [], []
    for (start_raw, end_raw), run_raw in zip(data["bounds"], data["seq_run"]):
        run = int(run_raw)
        if run not in train_groups:
            continue
        start, end = int(start_raw), int(end_raw)
        mask = data["valid"][start:end]
        if np.any(mask):
            sensor_parts.append(data["sensors"][start:end][mask])
            truth_parts.append(data["truth"][start:end][mask])
    if not sensor_parts:
        raise ValueError("training split has no valid sensor samples")
    sensors = np.concatenate(sensor_parts).astype(np.float64)
    truth = np.concatenate(truth_parts).astype(np.float64)[:, :2]
    sensor_mean = sensors.mean(axis=0).astype(np.float32)
    sensor_scale = np.maximum(sensors.std(axis=0), np.asarray(
        [0.02, 0.02, 0.10, 0.10, 0.5, 0.5, 0.2, 0.02, 0.02, 0.003],
        dtype=np.float64)).astype(np.float32)
    truth_mean = truth.mean(axis=0).astype(np.float32)
    truth_scale = np.maximum(truth.std(axis=0), np.asarray(
        [0.25, 0.05], dtype=np.float64)).astype(np.float32)
    return sensor_mean, sensor_scale, truth_mean, truth_scale


def _arrays_for_windows(data: dict[str, Any], windows: list[tuple[int, int]]):
    x = np.stack([data["sensors"][a:b] for a, b in windows]).astype(np.float32)
    y = np.stack([data["truth"][a:b] for a, b in windows]).astype(np.float32)
    return x, y


def _tensor_batch(torch, arrays, sensor_mean, sensor_scale,
                  truth_mean, truth_scale, device):
    x, y = arrays
    x = (x - sensor_mean[None, None, :]) / sensor_scale[None, None, :]
    y = (y[:, :, :2] - truth_mean[None, None, :]) / truth_scale[None, None, :]
    return (torch.as_tensor(x, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.float32, device=device))


def _score(torch, model, arrays, sensor_mean, sensor_scale,
           truth_mean, truth_scale, device, burn_in: int,
           score_steps: int, include_strata: bool = False) -> dict[str, Any]:
    x, y = _tensor_batch(torch, arrays, sensor_mean, sensor_scale,
                         truth_mean, truth_scale, device)
    model.eval()
    with torch.no_grad():
        prediction = model(x)[:, burn_in:burn_in + score_steps]
        target = y[:, burn_in:burn_in + score_steps]
        error = (prediction - target) * torch.as_tensor(
            truth_scale, dtype=torch.float32, device=device)
        truth_uv = target * torch.as_tensor(
            truth_scale, dtype=torch.float32, device=device) + torch.as_tensor(
            truth_mean, dtype=torch.float32, device=device)
        prediction_uv = prediction * torch.as_tensor(
            truth_scale, dtype=torch.float32, device=device) + torch.as_tensor(
            truth_mean, dtype=torch.float32, device=device)
        truth_full = torch.as_tensor(
            arrays[1][:, burn_in:burn_in + score_steps],
            dtype=torch.float32, device=device)
        sensor_physical = x[:, burn_in:burn_in + score_steps] * torch.as_tensor(
            sensor_scale, dtype=torch.float32, device=device) + torch.as_tensor(
            sensor_mean, dtype=torch.float32, device=device)
        # Learn the difficult u/v components; use the direct gyro reading for
        # yaw rather than making a recurrent network relearn that measurement.
        yaw_error = sensor_physical[:, :, 6] - truth_full[:, :, 2]
        full_error = torch.cat((error, yaw_error[:, :, None]), dim=2)
        rmse = torch.sqrt(torch.mean(full_error ** 2, dim=(0, 1))).cpu().numpy()
        per_sample_error = full_error.cpu().numpy()
        # Causal baselines: rear encoder mean for u, zero v, and direct gyro r.
        baseline = torch.stack((
            sensor_physical[:, :, 2:4].mean(dim=2),
            torch.zeros_like(truth_uv[:, :, 1]),
            sensor_physical[:, :, 6]), dim=2)
        baseline_error = baseline - truth_full
        baseline_rmse = torch.sqrt(torch.mean(
            baseline_error ** 2, dim=(0, 1))).cpu().numpy()
        step_errors = {}
        for step in (1, 4, 10, 20, 32):
            if step <= score_steps:
                values = per_sample_error[:, step - 1, :]
                step_errors[str(step)] = {
                    name: float(np.sqrt(np.mean(values[:, i] ** 2)))
                    for i, name in enumerate(STATE_NAMES)}
    result = {
        "windows": int(x.shape[0]),
        "steps_per_window": score_steps,
        "causal_sensor_observer_rmse": dict(zip(STATE_NAMES, map(float, rmse))),
        "causal_sensor_baselines_rmse": dict(zip(STATE_NAMES, map(float, baseline_rmse))),
        "rmse_by_update_step": step_errors,
    }
    if include_strata:
        truth_np = truth_full.cpu().numpy()
        predicted_np = prediction_uv.cpu().numpy()
        sensor_np = sensor_physical.cpu().numpy()
        truth_speed = np.linalg.norm(truth_np[:, :, :2], axis=2)
        speed_edges = (0.0, 2.0, 4.0, 6.0, 8.0, float("inf"))
        steer_edges = (0.0, 0.15, 0.30, 0.42, float("inf"))
        throttle_edges = (0.0, 0.15, 0.30, float("inf"))
        speed_bin = np.digitize(truth_speed, speed_edges[1:-1])
        steer_bin = np.digitize(np.abs(sensor_np[:, :, 0]), steer_edges[1:-1])
        throttle_bin = np.digitize(sensor_np[:, :, 1], throttle_edges[1:-1])
        truth_uv_np = truth_np[:, :, :2]
        sensor_u = sensor_np[:, :, 2:4].mean(axis=2)
        cells: dict[str, Any] = {}
        for i in range(len(speed_edges) - 1):
            for j in range(len(steer_edges) - 1):
                for k in range(len(throttle_edges) - 1):
                    selected = ((speed_bin == i) & (steer_bin == j)
                                & (throttle_bin == k))
                    count = int(np.count_nonzero(selected))
                    if not count:
                        continue
                    error = predicted_np[selected] - truth_uv_np[selected]
                    encoder_error = np.column_stack((
                        sensor_u[selected] - truth_uv_np[selected, 0],
                        -truth_uv_np[selected, 1]))
                    key = (f"speed_{speed_edges[i]}_{speed_edges[i + 1]}__"
                           f"abs_steer_{steer_edges[j]}_{steer_edges[j + 1]}__"
                           f"throttle_{throttle_edges[k]}_{throttle_edges[k + 1]}")
                    cells[key] = {
                        "scored_samples": count,
                        "observer_rmse_u_v_mps": np.sqrt(
                            np.mean(error ** 2, axis=0)).tolist(),
                        "encoder_zero_v_baseline_rmse_u_v_mps": np.sqrt(
                            np.mean(encoder_error ** 2, axis=0)).tolist(),
                    }
        result["operating_region_rmse"] = {
            "conditioning_uses_offline_truth_only_for_evaluation": True,
            "cells": cells,
        }
        encoder_residual = np.abs(sensor_u - truth_uv_np[:, :, 0])
        residual_edges = (0.0, 0.25, 0.50, 1.0, float("inf"))
        residual_bin = np.digitize(encoder_residual, residual_edges[1:-1])
        residual_cells = {}
        for i in range(len(residual_edges) - 1):
            selected = residual_bin == i
            count = int(np.count_nonzero(selected))
            if not count:
                continue
            observer_error = predicted_np[selected] - truth_uv_np[selected]
            baseline_error = np.column_stack((
                sensor_u[selected] - truth_uv_np[:, :, 0][selected],
                -truth_uv_np[:, :, 1][selected]))
            residual_cells[f"abs_encoder_minus_u_{residual_edges[i]}_"
                           f"{residual_edges[i + 1]}_mps"] = {
                "scored_samples": count,
                "observer_rmse_u_v_mps": np.sqrt(
                    np.mean(observer_error ** 2, axis=0)).tolist(),
                "encoder_zero_v_baseline_rmse_u_v_mps": np.sqrt(
                    np.mean(baseline_error ** 2, axis=0)).tolist(),
            }
        result["rear_encoder_residual_rmse"] = {
            "is_a_proxy_not_a_direct_contact_slip_measurement": True,
            "cells": residual_cells,
        }
    return result


def _save(torch, path: Path, model, metadata):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, temporary)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, nn = _torch()
    data = _load_dataset(args.dataset)
    window_steps = args.burn_in + args.score_steps
    train_groups = _valid_segments(data, "train", window_steps)
    validation_groups = _valid_segments(data, "validation", window_steps)
    if not train_groups or not validation_groups:
        raise ValueError("train and validation must both have valid sensor windows")
    family_sampler = build_sequence_sampler(data, train_groups)
    validation_windows = _eval_windows(validation_groups, window_steps, args.eval_windows)
    flat_validation = [window for windows in validation_windows.values() for window in windows]
    val_arrays = _arrays_for_windows(data, flat_validation)
    sensor_mean, sensor_scale, truth_mean, truth_scale = _normalizers(data, train_groups)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    Model = _make_model(nn, len(data["sensor_feature_names"]), args.hidden_size)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    deadline = time.monotonic() + args.time_budget_hours * 3600.0
    best_score = math.inf
    best_step = 0
    stale = 0
    history = []
    checkpoint = args.output_dir / "sensor_observer.pt"
    step = 0
    while time.monotonic() < deadline and step < args.max_steps:
        model.train()
        arrays = _sample_batch(data, train_groups, args.batch_size,
                               window_steps, rng, family_sampler)
        raw_inputs, raw_targets = arrays
        weights = _training_weights(raw_inputs, raw_targets,
                                    args.weight_mode,
                                    args.hard_region_weight)
        x, y = _tensor_batch(torch, arrays, sensor_mean, sensor_scale,
                             truth_mean, truth_scale, device)
        prediction = model(x)[:, args.burn_in:]
        target = y[:, args.burn_in:]
        pointwise_loss = nn.functional.smooth_l1_loss(
            prediction, target, beta=0.05, reduction="none").mean(dim=-1)
        weight_tensor = torch.as_tensor(
            weights[:, args.burn_in:], dtype=torch.float32, device=device)
        loss = torch.sum(pointwise_loss * weight_tensor) / torch.sum(weight_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        step += 1
        if step % args.eval_every != 0 and step != 1:
            continue
        score = _score(torch, model, val_arrays, sensor_mean, sensor_scale,
                       truth_mean, truth_scale, device, args.burn_in,
                       args.score_steps)
        values = np.asarray([score["causal_sensor_observer_rmse"][name]
                             for name in LEARNED_STATE_NAMES])
        normalized_score = float(np.mean(values / truth_scale))
        history.append({"step": step, "train_loss": float(loss.detach().cpu()),
                        "normalized_validation_rmse": normalized_score,
                        "validation": score})
        print(f"step={step} train={float(loss.detach().cpu()):.6f} "
              f"val_nrmse={normalized_score:.5f} device={device}", flush=True)
        if normalized_score < best_score:
            best_score, best_step, stale = normalized_score, step, 0
            _save(torch, checkpoint, model, {
                "schema_version": 1, "seed": args.seed,
                "step": step, "hidden_size": args.hidden_size,
                "burn_in": args.burn_in, "score_steps": args.score_steps,
                "sensor_feature_names": data["sensor_feature_names"],
                "target_names": list(LEARNED_STATE_NAMES),
                "weight_mode": args.weight_mode,
                "hard_region_weight": args.hard_region_weight,
                "training_runs": [str(data["run_ids"][run])
                                  for run in sorted(train_groups)],
            })
        else:
            stale += 1
            if stale >= args.patience:
                break
    if not checkpoint.is_file():
        raise ValueError("no observer checkpoint was produced")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "dataset": str(args.dataset.resolve()),
        "identification_question": "causal sensor-history state estimation; no body-state history is an input",
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "sensor_features": data["sensor_feature_names"],
        "learned_targets": list(LEARNED_STATE_NAMES),
        "estimated_state": list(STATE_NAMES),
        "truth_policy": "offline bridge state is a label only; no future or current bridge state is an input",
        "window": {"burn_in_steps": args.burn_in, "scored_steps": args.score_steps,
                   "sample_rate_nominal_hz": 1.0 / SENSOR_PERIOD_S},
        "split_run_counts": {name: int(np.count_nonzero(data["splits"] == name))
                              for name in np.unique(data["splits"])},
        "valid_segment_counts": {name: sum(len(items) for items in groups.values())
                                 for name, groups in (("train", train_groups),
                                                      ("validation", validation_groups))},
        "training_sampler": family_sampler.metadata,
        "normalization": {"sensor_mean": sensor_mean.tolist(),
                          "sensor_scale": sensor_scale.tolist(),
                          "target_mean": truth_mean.tolist(),
                          "target_scale": truth_scale.tolist()},
        "configuration": {"hidden_size": args.hidden_size,
                          "batch_size": args.batch_size,
                          "learning_rate": args.learning_rate,
                          "time_budget_hours": args.time_budget_hours,
                          "max_steps": args.max_steps, "seed": args.seed,
                          "weight_mode": args.weight_mode,
                          "hard_region_weight": args.hard_region_weight,
                          "optimizer_steps": step, "best_step": best_step},
        "best_validation": _score(
            torch, model, val_arrays, sensor_mean, sensor_scale,
            truth_mean, truth_scale, device, args.burn_in, args.score_steps,
            include_strata=True),
        "history": history,
        "checkpoint": checkpoint.name,
    }
    if args.score_test:
        for split in ("test", "final_test"):
            if split == "final_test" and not args.score_final_test:
                continue
            groups = _valid_segments(data, split, window_steps)
            windows_by_run = _eval_windows(groups, window_steps, args.eval_windows)
            run_results = {}
            for run, windows in sorted(windows_by_run.items()):
                arrays = _arrays_for_windows(data, windows)
                run_results[str(data["run_ids"][run])] = _score(
                    torch, model, arrays, sensor_mean, sensor_scale,
                    truth_mean, truth_scale, device, args.burn_in,
                    args.score_steps, include_strata=True)
            if run_results:
                report["named_holdouts" if split == "test" else "final_test"] = run_results
    report_path = args.output_dir / "observer_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"wrote {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--time-budget-hours", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=16)
    parser.add_argument("--score-steps", type=int, default=32)
    parser.add_argument("--eval-windows", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--weight-mode",
                        choices=("uniform", "high_response", "encoder_mismatch", "combined"),
                        default="uniform",
                        help="optionally emphasize high-response or encoder/body-speed-residual intervals")
    parser.add_argument("--hard-region-weight", type=float, default=4.0,
                        help="relative loss weight for selected intervals (default: 4)")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--score-test", action="store_true",
                        help="score named whole-run holdouts after training")
    parser.add_argument("--score-final-test", action="store_true",
                        help="also use the reserved final test once")
    args = parser.parse_args()
    if (args.time_budget_hours <= 0.0 or args.max_steps < 1
            or args.burn_in < 1 or args.score_steps < 1):
        parser.error("time, steps, burn-in, and scored length must be positive")
    if not math.isfinite(args.hard_region_weight) or args.hard_region_weight < 1.0:
        parser.error("--hard-region-weight must be finite and at least 1")
    if args.score_final_test:
        args.score_test = True
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"sensor observer training failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
