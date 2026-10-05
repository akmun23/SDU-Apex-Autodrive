#!/usr/bin/env python3
"""Train and evaluate the handoff-prescribed deepSI DT-SUBNET reference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESET_ROOT = ROOT / "live_runs/derived_dynamics_learning_20260928/subnet_reset_20261004"
DATASET = RESET_ROOT / "body_sysid_v1.npz"
DATASET_MANIFEST = RESET_ROOT / "body_sysid_v1_manifest.json"
EVALUATION_STARTS = RESET_ROOT / "evaluation_starts.json"
OUTPUT_DIR = RESET_ROOT / "deepsi_subnet_reference_v1"
SOURCE_COMMIT = "109cf74f49a7b27539232c53981584befc1becc0"
DT_S = 0.025
HISTORY = 12
STATE_ORDER = 9
ROLLOUT_T = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 1000
SEED = 101
HORIZONS = (1, 10, 30, 40, 80, 150, 200, 400)
OUTPUT_NAMES = ("u_rear_mps", "v_rear_mps", "yaw_rate_rps")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _metric(errors: np.ndarray, train_std: np.ndarray) -> dict[str, Any]:
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    return {
        "sample_count": int(errors.shape[0]),
        "rmse_by_channel": dict(zip(OUTPUT_NAMES, rmse.astype(float).tolist())),
        "nrms_by_channel": dict(zip(
            OUTPUT_NAMES, (rmse / train_std).astype(float).tolist())),
        "macro_channel_nrms": float(np.mean(rmse / train_std)),
    }


def _run_cluster_summary(per_run: dict[str, dict[str, Any]], seed: int
                         ) -> dict[str, Any]:
    if not per_run:
        raise ValueError("run-cluster summary requires independent runs")
    names = sorted(per_run)
    metrics = set(per_run[names[0]])
    summary: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(5000, len(names)))
    for metric in sorted(metrics):
        values = np.asarray([per_run[name][metric] for name in names],
                            dtype=np.float64)
        sample_means = values[draws].mean(axis=1)
        summary[metric] = {
            "run_macro_mean": float(values.mean()),
            "run_cluster_bootstrap_95pct_ci": [
                float(value) for value in np.quantile(sample_means, [0.025, 0.975])],
        }
    return summary


def _make_sequences(arrays: dict[str, np.ndarray], deep_si, split: str, min_length: int
                    ) -> tuple[list[Any], list[dict[str, Any]], int]:
    sequences = []
    metadata = []
    skipped_samples = 0
    bounds = arrays["sequence_bounds"]
    seq_split = arrays["sequence_split"].astype(str)
    inputs = arrays["inputs"]
    outputs = arrays["outputs"]
    for seq_index, ((start, end), role) in enumerate(zip(bounds, seq_split)):
        if role != split:
            continue
        start, end = int(start), int(end)
        length = end - start
        if length < min_length:
            skipped_samples += length
            continue
        sequences.append(deep_si.System_data(
            u=inputs[start:end],
            y=outputs[start:end],
            dt=DT_S))
        metadata.append({
            "sequence_index": seq_index,
            "sequence_id": str(arrays["sequence_id_table"][seq_index]),
            "run_id": str(arrays["sequence_run_id"][seq_index]),
            "row_start": start,
            "row_end": end,
            "length": length,
        })
    if not sequences:
        raise ValueError(f"no usable {split} sequences at minimum length {min_length}")
    return sequences, metadata, skipped_samples


def _evaluate_full_sequences(model, validation_sequences: list[Any], metadata: list[dict[str, Any]],
                              train_std: np.ndarray) -> dict[str, Any]:
    import torch

    model.eval()
    by_run: dict[str, list[np.ndarray]] = defaultdict(list)
    point_errors = []
    sequence_metrics = []
    with torch.no_grad():
        for data, info in zip(validation_sequences, metadata):
            prediction = model.apply_experiment(data)
            start = int(prediction.cheat_n)
            errors = np.asarray(prediction.y[start:] - data.y[start:], dtype=np.float64)
            if errors.size == 0:
                continue
            point_errors.append(errors)
            by_run[info["run_id"]].append(errors)
            sequence_metrics.append({
                "run_id": info["run_id"],
                "sequence_id": info["sequence_id"],
                "evaluated_samples": int(len(errors)),
                "rmse_by_channel": dict(zip(
                    OUTPUT_NAMES,
                    np.sqrt(np.mean(np.square(errors), axis=0)).astype(float).tolist())),
            })
    if not point_errors:
        raise RuntimeError("deepSI produced no full-sequence validation predictions")
    per_run = {}
    for run_id, errors in sorted(by_run.items()):
        per_run[run_id] = _metric(np.concatenate(errors), train_std)
    run_macro = {
        key: float(np.mean([row[key] for row in per_run.values()]))
        for key in ("macro_channel_nrms",)
    }
    run_macro["rmse_by_channel"] = {
        name: float(np.mean([row["rmse_by_channel"][name]
                             for row in per_run.values()]))
        for name in OUTPUT_NAMES
    }
    return {
        "all_points_pooled": _metric(np.concatenate(point_errors), train_std),
        "run_macro": run_macro,
        "run_cluster_summary": _run_cluster_summary({
            run_id: {
                "macro_channel_nrms": row["macro_channel_nrms"],
                **{f"{name}_rmse": row["rmse_by_channel"][name]
                   for name in OUTPUT_NAMES},
            }
            for run_id, row in per_run.items()}, SEED + 280),
        "per_run": per_run,
        "sequence_count": len(sequence_metrics),
        "per_sequence": sequence_metrics,
    }


def _run_fixed_starts(model, dataset_npz, source_npz, starts_path: Path,
                      train_std: np.ndarray) -> dict[str, Any]:
    """Free-run only actuator feedback, from the frozen past-output history."""
    import torch
    import deepSI

    starts_doc = json.loads(starts_path.read_text(encoding="utf-8"))
    starts = starts_doc["split_roles"]["development_validation"]
    frame_ids = np.asarray(dataset_npz["source_frame_index"], dtype=np.int64)
    inputs = np.asarray(dataset_npz["inputs"], dtype=np.float32)
    outputs = np.asarray(dataset_npz["outputs"], dtype=np.float32)
    sequence_bounds = np.asarray(dataset_npz["sequence_bounds"], dtype=np.int64)
    sequence_run_ids = np.asarray(dataset_npz["sequence_run_id"]).astype(str)
    sequence_splits = np.asarray(dataset_npz["sequence_split"]).astype(str)
    source_pose = np.asarray(source_npz["simulator_pose_xyyaw"], dtype=np.float64)
    if frame_ids.ndim != 1 or not np.all(frame_ids[1:] > frame_ids[:-1]):
        raise ValueError("body dataset source-frame map must be strictly increasing")

    start_rows = []
    for run_id, run_starts in starts.items():
        for item in run_starts:
            source_row = int(item["absolute_row"])
            local = int(np.searchsorted(frame_ids, source_row))
            if local >= len(frame_ids) or frame_ids[local] != source_row:
                raise ValueError(f"frozen start does not map exactly: {run_id}/{source_row}")
            sequence_slot = int(np.searchsorted(
                sequence_bounds[:, 0], local, side="right") - 1)
            if (sequence_slot < 0 or local >= sequence_bounds[sequence_slot, 1]
                    or sequence_run_ids[sequence_slot] != run_id
                    or sequence_splits[sequence_slot] != "validation"):
                raise ValueError(f"frozen start maps to the wrong sequence: {run_id}/{source_row}")
            if int(item["maximum_available_horizon_steps"]) < max(HORIZONS):
                raise ValueError("frozen validation start lacks the prescribed 10 s horizon")
            start_rows.append((run_id, item, local))

    errors_by_horizon: dict[int, list[np.ndarray]] = {h: [] for h in HORIZONS}
    pose_by_horizon: dict[int, list[dict[str, float]]] = {h: [] for h in HORIZONS}
    errors_by_run_horizon: dict[str, dict[int, list[np.ndarray]]] = {
        run_id: {h: [] for h in HORIZONS} for run_id in starts
    }
    pose_by_run_horizon: dict[str, dict[int, list[dict[str, float]]]] = {
        run_id: {h: [] for h in HORIZONS} for run_id in starts
    }
    model.eval()
    with torch.no_grad():
        for run_id, item, local in start_rows:
            source_row = int(item["absolute_row"])
            sequence_slot = int(np.searchsorted(
                sequence_bounds[:, 0], local, side="right") - 1)
            if sequence_slot < 0:
                raise ValueError(f"frozen start has no containing sequence: {run_id}/{source_row}")
            seq_start, seq_end = map(int, sequence_bounds[sequence_slot])
            if (local - HISTORY + 1 < seq_start
                    or local + max(HORIZONS) + 1 > seq_end):
                raise ValueError(f"frozen start crosses sequence edge: {run_id}/{source_row}")
            horizon = max(HORIZONS)
            # A frozen start row is the observed state anchor k. All models
            # must predict k+1..k+H from the same history through k.
            window_start = local - HISTORY + 1
            window_end = local + horizon + 1
            data = deepSI.System_data(
                u=inputs[window_start:window_end],
                y=outputs[window_start:window_end],
                dt=DT_S)
            prediction = model.apply_experiment(data)
            predicted = np.asarray(prediction.y[HISTORY:], dtype=np.float64)
            target = np.asarray(data.y[HISTORY:], dtype=np.float64)
            if predicted.shape != (horizon, 3) or target.shape != predicted.shape:
                raise ValueError("reference rollout did not return the exact frozen horizon")

            for h in HORIZONS:
                errors = predicted[:h] - target[:h]
                errors_by_horizon[h].append(errors)
                errors_by_run_horizon[run_id][h].append(errors)

            initial_body = np.asarray(outputs[local], dtype=np.float64)
            initial_pose = np.asarray(source_pose[source_row], dtype=np.float64)
            pose = initial_pose.copy()
            predicted_pose = []
            for step in range(horizon):
                # The exact exponential map advances pose with the body twist
                # at the beginning of [k,k+1), matching the plant integrator.
                body_at_interval_start = (
                    initial_body if step == 0 else predicted[step - 1])
                u, v, yaw_rate = body_at_interval_start
                dtheta = yaw_rate * DT_S
                if abs(yaw_rate) > 1.0e-7:
                    dx_body = (u * math.sin(dtheta)
                               + v * (math.cos(dtheta) - 1.0)) / yaw_rate
                    dy_body = (u * (1.0 - math.cos(dtheta))
                               + v * math.sin(dtheta)) / yaw_rate
                else:
                    dx_body, dy_body = u * DT_S, v * DT_S
                pose = np.asarray((
                    pose[0] + math.cos(pose[2]) * dx_body
                    - math.sin(pose[2]) * dy_body,
                    pose[1] + math.sin(pose[2]) * dx_body
                    + math.cos(pose[2]) * dy_body,
                    math.atan2(math.sin(pose[2] + dtheta),
                               math.cos(pose[2] + dtheta)),
                ), dtype=np.float64)
                predicted_pose.append(pose.copy())
            predicted_pose = np.asarray(predicted_pose)
            truth_pose = source_pose[source_row + 1:source_row + horizon + 1]
            for h in HORIZONS:
                delta = predicted_pose[:h] - truth_pose[:h]
                delta[:, 2] = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
                position = np.linalg.norm(delta[:, :2], axis=1)
                endpoint_heading = float(delta[h - 1, 2])
                summary = {
                    "position_radial_trajectory_rmse_m": float(
                        np.sqrt(np.mean(np.square(position)))),
                    "endpoint_position_error_m": float(position[-1]),
                    "heading_trajectory_rmse_rad": float(
                        np.sqrt(np.mean(np.square(delta[:, 2])))),
                    "endpoint_heading_error_rad": endpoint_heading,
                }
                pose_by_horizon[h].append(summary)
                pose_by_run_horizon[run_id][h].append(summary)

    output = {"fixed_start_count": len(start_rows), "horizons": {}}
    for h in HORIZONS:
        all_errors = np.concatenate(errors_by_horizon[h], axis=0)
        per_run_body = {
            run_id: _metric(np.concatenate(errors_by_run_horizon[run_id][h]), train_std)
            for run_id in sorted(starts)
        }
        per_run_pose = {
            run_id: {
                metric: float(np.mean([entry[metric]
                                       for entry in pose_by_run_horizon[run_id][h]]))
                for metric in pose_by_horizon[h][0]
            }
            for run_id in sorted(starts)
        }
        pose_macro = {
            metric: float(np.mean([row[metric] for row in per_run_pose.values()]))
            for metric in pose_by_horizon[h][0]
        }
        per_run_summary = {
            run_id: {
                "body_macro_channel_nrms": per_run_body[run_id]["macro_channel_nrms"],
                **{f"body_{name}_rmse": per_run_body[run_id]["rmse_by_channel"][name]
                   for name in OUTPUT_NAMES},
                **{f"pose_{name}": per_run_pose[run_id][name]
                   for name in pose_macro},
            }
            for run_id in per_run_body
        }
        output["horizons"][str(h)] = {
            "duration_s": h * DT_S,
            "body_all_window_points_pooled": _metric(all_errors, train_std),
            "body_run_macro": {
                "macro_channel_nrms": float(np.mean([
                    row["macro_channel_nrms"] for row in per_run_body.values()])),
                "rmse_by_channel": {
                    name: float(np.mean([row["rmse_by_channel"][name]
                                         for row in per_run_body.values()]))
                    for name in OUTPUT_NAMES
                },
            },
            "body_per_run": per_run_body,
            "pose_run_macro": pose_macro,
            "run_cluster_summary": _run_cluster_summary(
                per_run_summary, SEED + h),
            "pose_per_run": per_run_pose,
        }
    return output


def run(epochs: int = EPOCHS, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if epochs != EPOCHS:
        raise ValueError(f"reference training budget is fixed at {EPOCHS} epochs")
    if not DATASET.is_file() or not DATASET_MANIFEST.is_file():
        raise FileNotFoundError("frozen body-only SUBNET dataset or manifest is missing")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "subnet_reference_report.json"
    trace_path = output_dir / "validation_trace.jsonl"
    checkpoint_path = output_dir / "SS_encoder_sdu_apex_subnet_reference_20261004_best.pth"
    if any(path.exists() for path in (report_path, trace_path, checkpoint_path,
                                      checkpoint_path.with_name(
                                          checkpoint_path.name.replace("_best", "_last")))):
        raise FileExistsError(f"refusing to overwrite reference artifacts in {output_dir}")

    import torch
    import deepSI
    import deepSI.fit_systems.fit_system as fit_system_module

    source_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if sha256(DATASET) != source_manifest["dataset_sha256"]:
        raise ValueError("body dataset hash differs from its frozen manifest")
    if source_manifest["input_names"] != [
            "steering_feedback_rad", "throttle_feedback_norm"]:
        raise ValueError("the official body SUBNET input contract has changed")
    if source_manifest["output_names"] != list(OUTPUT_NAMES):
        raise ValueError("the official body SUBNET output contract has changed")
    if source_manifest["sample_period_s"] != DT_S:
        raise ValueError("body dataset does not use the fixed 25 ms timebase")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)

    with np.load(DATASET, allow_pickle=False) as body_data:
        if set(np.unique(body_data["split"]).astype(str)) - {"train", "validation"}:
            raise ValueError("test/final-test rows leaked into the reference data")
        min_train_length = HISTORY + ROLLOUT_T
        body_arrays = {
            name: np.asarray(body_data[name])
            for name in (
                "inputs", "outputs", "sequence_bounds", "sequence_split",
                "sequence_id_table", "sequence_run_id", "source_frame_index")
        }
        train_sequences, train_meta, skipped_train_rows = _make_sequences(
            body_arrays, deepSI, "train", min_train_length)
        validation_sequences, validation_meta, skipped_val_rows = _make_sequences(
            body_arrays, deepSI, "validation", HISTORY + 1)
        train_data = deepSI.System_data_list(train_sequences)
        validation_data = deepSI.System_data_list(validation_sequences)
        train_outputs = np.concatenate([item.y for item in train_sequences])
        train_std = np.std(train_outputs, axis=0, ddof=0)
        if not np.isfinite(train_std).all() or np.any(train_std <= 1e-12):
            raise ValueError("training output standard deviation is invalid")
        body_meta = {
            "source_frame_index": body_arrays["source_frame_index"],
            "inputs": body_arrays["inputs"],
            "outputs": body_arrays["outputs"],
            "sequence_bounds": body_arrays["sequence_bounds"],
            "sequence_run_id": body_arrays["sequence_run_id"],
            "sequence_split": body_arrays["sequence_split"],
        }

    model = deepSI.fit_systems.SS_encoder(nx=STATE_ORDER, na=HISTORY, nb=HISTORY)
    model.unique_code = "sdu_apex_subnet_reference_20261004"
    checkpoint_dir = output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    original_get_work_dirs = fit_system_module.get_work_dirs
    fit_system_module.get_work_dirs = lambda: {"checkpoints": str(checkpoint_dir)}
    original_torch_load = torch.load

    def load_trusted_deep_si_checkpoint(*args, **kwargs):
        # deepSI 0.3.29 serializes its module objects into a local checkpoint;
        # PyTorch 2.6+ defaults to weights_only=True and rejects that format.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_deep_si_checkpoint

    trace_stream = trace_path.open("x", encoding="utf-8")
    torch_system_class = deepSI.fit_systems.System_torch
    original_validation = torch_system_class.cal_validation_error
    fit_started = time.perf_counter()
    training_started_at = datetime.now(timezone.utc).isoformat()

    def traced_validation(self, val_sys_data, validation_measure="sim-NRMS"):
        score = original_validation(self, val_sys_data, validation_measure)
        trace_stream.write(json.dumps({
            "epoch": float(self.epoch_counter),
            "optimizer_updates": int(self.batch_counter),
            "validation_measure": validation_measure,
            "validation_score": float(score),
            "elapsed_wall_s": float(time.perf_counter() - fit_started),
            "utc": datetime.now(timezone.utc).isoformat(),
        }, sort_keys=True) + "\n")
        trace_stream.flush()
        os.fsync(trace_stream.fileno())
        return score

    torch_system_class.cal_validation_error = traced_validation
    try:
        model.fit(
            train_sys_data=train_data,
            val_sys_data=validation_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            loss_kwargs={"nf": ROLLOUT_T},
            optimizer_kwargs={"lr": LEARNING_RATE},
            validation_measure="sim-NRMS",
            cuda=bool(torch.cuda.is_available()),
            verbose=1,
        )
    finally:
        torch.load = original_torch_load
        torch_system_class.cal_validation_error = original_validation
        fit_system_module.get_work_dirs = original_get_work_dirs
        trace_stream.close()

    fit_wall_s = float(time.perf_counter() - fit_started)
    with np.load(DATASET, allow_pickle=False) as body_data, np.load(
            Path(source_manifest["source_dataset"]), allow_pickle=False) as source_data:
        full_validation = _evaluate_full_sequences(
            model, validation_sequences, validation_meta, train_std)
        fixed_start_evaluation = _run_fixed_starts(
            model, body_meta, source_data, EVALUATION_STARTS, train_std)

    trace = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    val_scores = np.asarray([item["validation_score"] for item in trace],
                            dtype=np.float64)
    valid_score_mask = (np.isfinite(val_scores)
                        & (val_scores < np.finfo(np.float64).max / 2.0))
    capped_score_mask = (np.isfinite(val_scores)
                         & (val_scores >= np.finfo(np.float64).max / 2.0))
    nonfinite_score_mask = ~np.isfinite(val_scores)
    if len(val_scores) == 0 or not valid_score_mask.any():
        raise RuntimeError("deepSI fit returned no finite validation checkpoint")
    best_trace_index = int(np.argmin(np.where(valid_score_mask, val_scores, np.inf)))
    for item in trace:
        score = float(item["validation_score"])
        if not math.isfinite(score):
            item["validation_score"] = None
    trace_path.write_text("".join(
        json.dumps(item, allow_nan=False, sort_keys=True) + "\n"
        for item in trace), encoding="utf-8")
    trace_sha256 = sha256(trace_path)
    train_window_count = sum(
        max(0, item["length"] - HISTORY - ROLLOUT_T + 1)
        for item in train_meta)
    updates_per_epoch = train_window_count // BATCH_SIZE
    actual_updates = int(trace[-1]["optimizer_updates"])
    actual_epoch = int(round(float(trace[-1]["epoch"])))
    selected_checkpoint_updates = int(model.batch_counter)
    if (actual_updates != updates_per_epoch * EPOCHS
            or actual_epoch != EPOCHS):
        raise RuntimeError(
            "deepSI update count differs from the fixed 1000-epoch complete-data budget")

    report = {
        "schema_version": 1,
        "study": "handoff-prescribed deepSI discrete-time SUBNET body model",
        "status": "completed",
        "source_commit": git_commit(),
        "source_commit_handoff_parent": "2129427838101eee277e17e91727810bbe2df687",
        "source_commit_mismatch_note": (
            "Workspace HEAD was already 5b26c5ac829acf56d77bdb71a979b7e494703ac5, "
            "two commits ahead of the handoff parent; no rewind was performed."),
        "reference_implementation": {
            "package": "deepSI",
            "version": importlib.metadata.version("deepSI"),
            "git_commit": SOURCE_COMMIT,
            "class": "deepSI.fit_systems.SS_encoder",
            "objective": "official truncated multi-step simulation MSE",
            "objective_implementation": "deepSI SS_encoder.loss",
            "model_file_sha256": sha256(checkpoint_path),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pip_freeze": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None,
        },
        "data": {
            "dataset": str(DATASET),
            "dataset_sha256": sha256(DATASET),
            "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
            "source_dataset": source_manifest["source_dataset"],
            "source_dataset_sha256": source_manifest["source_dataset_sha256"],
            "evaluation_starts_sha256": sha256(EVALUATION_STARTS),
            "sample_period_s": DT_S,
            "input_names": ["steering_feedback_rad", "throttle_feedback_norm"],
            "output_names": list(OUTPUT_NAMES),
            "body_reference_point": "rear axle",
            "used_splits": ["train", "validation"],
            "test_or_final_test_used_for_training_or_scoring": False,
            "raw_source_contains_excluded_splits_not_used": True,
            "separate_practice_transfer_data_used": False,
            "practice_named_runs_in_frozen_train_split": sorted(
                run for run in {item["run_id"] for item in train_meta}
                if "practice" in run.lower()),
            "train_runs": sorted({item["run_id"] for item in train_meta}),
            "validation_runs": sorted({item["run_id"] for item in validation_meta}),
            "train_sequence_count_used": len(train_meta),
            "validation_sequence_count_used": len(validation_meta),
            "train_short_sequences_excluded_rows": skipped_train_rows,
            "validation_short_sequences_excluded_rows": skipped_val_rows,
            "train_output_std": dict(zip(OUTPUT_NAMES, train_std.astype(float).tolist())),
        },
        "configuration": {
            "sample_time_s": DT_S,
            "sample_rate_hz": 1.0 / DT_S,
            "input_channels": 2,
            "output_channels": 3,
            "n": HISTORY,
            "nx": STATE_ORDER,
            "encoder_hidden_layers": 2,
            "encoder_width": 64,
            "dynamics_hidden_layers": 2,
            "dynamics_width": 64,
            "output_hidden_layers": 2,
            "output_width": 64,
            "T": ROLLOUT_T,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "auxiliary_losses": [],
        },
        "training": {
            "epochs_requested": EPOCHS,
            "epochs_completed": actual_epoch,
            "updates_per_epoch": updates_per_epoch,
            "optimizer_updates": actual_updates,
            "windows_per_epoch": train_window_count,
            "windows_sampled_per_epoch": updates_per_epoch * BATCH_SIZE,
            "windows_seen": actual_updates * BATCH_SIZE,
            "wall_time_s": fit_wall_s,
            "started_utc": training_started_at,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "best_validation_epoch": float(trace[best_trace_index]["epoch"]),
            "best_validation_optimizer_updates": selected_checkpoint_updates,
            "best_validation_score": float(trace[best_trace_index]["validation_score"]),
            "validation_capped_maxfloat_epochs": [
                float(trace[index]["epoch"])
                for index, capped in enumerate(capped_score_mask) if capped],
            "validation_nonfinite_epochs": [
                float(trace[index]["epoch"])
                for index, nonfinite in enumerate(nonfinite_score_mask)
                if nonfinite],
            "maximum_finite_validation_score_below_cap": float(
                np.max(val_scores[valid_score_mask])),
            "validation_trace": str(trace_path),
            "validation_trace_sha256": trace_sha256,
            "early_stopping": (
                "DeepSI's best validation checkpoint was selected/restored; the "
                "prescribed 1000-epoch budget completed because the handoff does not "
                "specify a patience threshold; the selected deployed checkpoint is "
                "the minimum validation simulation-error checkpoint."),
        },
        "full_sequence_validation": full_validation,
        "frozen_start_validation": fixed_start_evaluation,
        "checkpoints": {
            "best": str(checkpoint_path),
            "best_sha256": sha256(checkpoint_path),
            "last": str(checkpoint_path.with_name(
                checkpoint_path.name.replace("_best", "_last"))),
            "last_sha256": sha256(checkpoint_path.with_name(
                checkpoint_path.name.replace("_best", "_last"))),
        },
        "training_trace_rows": len(trace),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def finalize_existing_run(output_dir: Path = OUTPUT_DIR,
                          early_stop_reason: str | None = None) -> dict[str, Any]:
    """Build the report from a completed trace/checkpoint without retraining.

    This also recovers cleanly when a legacy deepSI/PyTorch checkpoint-load
    incompatibility interrupts the runner after all optimizer updates have
    completed. It refuses to score an incomplete 1,000-epoch run.
    """
    import torch
    import deepSI

    output_dir = output_dir.resolve()
    report_path = output_dir / "subnet_reference_report.json"
    trace_path = output_dir / "validation_trace.jsonl"
    checkpoint_path = output_dir / (
        "SS_encoder_sdu_apex_subnet_reference_20261004_best.pth")
    last_checkpoint_path = output_dir / (
        "SS_encoder_sdu_apex_subnet_reference_20261004_last.pth")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    for path in (DATASET, DATASET_MANIFEST, EVALUATION_STARTS,
                 trace_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(DATASET_MANIFEST.read_text(encoding="utf-8"))
    if sha256(DATASET) != source_manifest["dataset_sha256"]:
        raise ValueError("body dataset hash differs from its frozen manifest")
    trace = [json.loads(line) for line in trace_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if not trace:
        raise RuntimeError("cannot finalize without a validation trace")
    epochs_completed = int(round(float(trace[-1]["epoch"])))
    if epochs_completed < EPOCHS and not early_stop_reason:
        raise RuntimeError(
            "an incomplete fit requires an explicit early-stop reason; "
            "otherwise all 1000 prescribed epochs must complete")
    val_scores = np.asarray([item["validation_score"] for item in trace],
                            dtype=np.float64)
    valid_score_mask = (np.isfinite(val_scores)
                        & (val_scores < np.finfo(np.float64).max / 2.0))
    capped_score_mask = (np.isfinite(val_scores)
                         & (val_scores >= np.finfo(np.float64).max / 2.0))
    nonfinite_score_mask = ~np.isfinite(val_scores)
    if not valid_score_mask.any():
        raise RuntimeError("completed trace contains no finite validation score")
    best_trace_index = int(np.argmin(np.where(valid_score_mask, val_scores, np.inf)))
    best_trace = trace[best_trace_index]

    with np.load(DATASET, allow_pickle=False) as body_data:
        if set(np.unique(body_data["split"]).astype(str)) - {"train", "validation"}:
            raise ValueError("test/final-test rows found in the SUBNET evaluation view")
        body_arrays = {
            name: np.asarray(body_data[name])
            for name in ("inputs", "outputs", "sequence_bounds", "sequence_split",
                         "sequence_id_table", "sequence_run_id", "source_frame_index")
        }
    train_sequences, train_meta, skipped_train_rows = _make_sequences(
        body_arrays, deepSI, "train", HISTORY + ROLLOUT_T)
    validation_sequences, validation_meta, skipped_val_rows = _make_sequences(
        body_arrays, deepSI, "validation", HISTORY + 1)
    train_std = np.std(np.concatenate([item.y for item in train_sequences]),
                       axis=0, ddof=0)
    if not np.isfinite(train_std).all() or np.any(train_std <= 1e-12):
        raise ValueError("training output standard deviation is invalid")

    model = deepSI.fit_systems.SS_encoder(
        nx=STATE_ORDER, na=HISTORY, nb=HISTORY)
    model.__dict__ = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
    model.eval()
    with np.load(Path(source_manifest["source_dataset"]),
                 allow_pickle=False) as source_data:
        full_validation = _evaluate_full_sequences(
            model, validation_sequences, validation_meta, train_std)
        fixed_start_evaluation = _run_fixed_starts(
            model, body_arrays, source_data, EVALUATION_STARTS, train_std)

    train_window_count = sum(
        max(0, item["length"] - HISTORY - ROLLOUT_T + 1)
        for item in train_meta)
    updates_per_epoch = train_window_count // BATCH_SIZE
    expected_updates = updates_per_epoch * epochs_completed
    actual_updates = int(trace[-1]["optimizer_updates"])
    if actual_updates != expected_updates:
        raise RuntimeError("final optimizer update count is not an integer full-epoch budget")
    first_time = datetime.fromisoformat(trace[0]["utc"])
    started_at = (first_time.timestamp() - float(trace[0]["elapsed_wall_s"]))
    for item in trace:
        score = item["validation_score"]
        if score is None or not math.isfinite(float(score)):
            item["validation_score"] = None
    trace_path.write_text("".join(
        json.dumps(item, allow_nan=False, sort_keys=True) + "\n"
        for item in trace), encoding="utf-8")
    trace_sha256 = sha256(trace_path)
    report = {
        "schema_version": 1,
        "study": "handoff-prescribed deepSI discrete-time SUBNET body model",
        "status": "completed" if epochs_completed == EPOCHS else "early_stopped_by_user",
        "source_commit": git_commit(),
        "source_commit_handoff_parent": "2129427838101eee277e17e91727810bbe2df687",
        "source_commit_mismatch_note": (
            "Workspace HEAD was already 5b26c5ac829acf56d77bdb71a979b7e494703ac5, "
            "two commits ahead of the handoff parent; no rewind was performed."),
        "reference_implementation": {
            "package": "deepSI",
            "version": importlib.metadata.version("deepSI"),
            "git_commit": SOURCE_COMMIT,
            "class": "deepSI.fit_systems.SS_encoder",
            "objective": "official truncated multi-step simulation MSE",
            "objective_implementation": "deepSI SS_encoder.loss",
            "model_file_sha256": sha256(checkpoint_path),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pip_freeze": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None,
        },
        "data": {
            "dataset": str(DATASET),
            "dataset_sha256": sha256(DATASET),
            "dataset_manifest_sha256": sha256(DATASET_MANIFEST),
            "source_dataset": source_manifest["source_dataset"],
            "source_dataset_sha256": source_manifest["source_dataset_sha256"],
            "evaluation_starts_sha256": sha256(EVALUATION_STARTS),
            "sample_period_s": DT_S,
            "input_names": ["steering_feedback_rad", "throttle_feedback_norm"],
            "output_names": list(OUTPUT_NAMES),
            "body_reference_point": "rear axle",
            "used_splits": ["train", "validation"],
            "test_or_final_test_used_for_training_or_scoring": False,
            "raw_source_contains_excluded_splits_not_used": True,
            "separate_practice_transfer_data_used": False,
            "practice_named_runs_in_frozen_train_split": sorted(
                run for run in {item["run_id"] for item in train_meta}
                if "practice" in run.lower()),
            "train_runs": sorted({item["run_id"] for item in train_meta}),
            "validation_runs": sorted({item["run_id"] for item in validation_meta}),
            "train_sequence_count_used": len(train_meta),
            "validation_sequence_count_used": len(validation_meta),
            "train_short_sequences_excluded_rows": skipped_train_rows,
            "validation_short_sequences_excluded_rows": skipped_val_rows,
            "train_output_std": dict(zip(
                OUTPUT_NAMES, train_std.astype(float).tolist())),
        },
        "configuration": {
            "sample_time_s": DT_S,
            "sample_rate_hz": 1.0 / DT_S,
            "input_channels": 2,
            "output_channels": 3,
            "n": HISTORY,
            "nx": STATE_ORDER,
            "encoder_hidden_layers": 2,
            "encoder_width": 64,
            "dynamics_hidden_layers": 2,
            "dynamics_width": 64,
            "output_hidden_layers": 2,
            "output_width": 64,
            "T": ROLLOUT_T,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "auxiliary_losses": [],
        },
        "training": {
            "epochs_requested": EPOCHS,
            "epochs_completed": epochs_completed,
            "updates_per_epoch": updates_per_epoch,
            "optimizer_updates": actual_updates,
            "windows_per_epoch": train_window_count,
            "windows_sampled_per_epoch": updates_per_epoch * BATCH_SIZE,
            "windows_seen": actual_updates * BATCH_SIZE,
            "wall_time_s": float(trace[-1]["elapsed_wall_s"]),
            "started_utc": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
            "completed_utc": trace[-1]["utc"],
            "best_validation_epoch": float(best_trace["epoch"]),
            "best_validation_optimizer_updates": int(best_trace["optimizer_updates"]),
            "best_validation_score": float(best_trace["validation_score"]),
            "validation_capped_maxfloat_epochs": [
                float(trace[index]["epoch"])
                for index, capped in enumerate(capped_score_mask) if capped],
            "validation_nonfinite_epochs": [
                float(trace[index]["epoch"])
                for index, nonfinite in enumerate(nonfinite_score_mask)
                if nonfinite],
            "maximum_finite_validation_score_below_cap": float(
                np.max(val_scores[valid_score_mask])),
            "validation_trace": str(trace_path),
            "validation_trace_sha256": trace_sha256,
            "early_stopping": (
                "All prescribed epochs completed; best deepSI validation checkpoint "
                "was selected/restored, with no separate practice or test data used."
                if epochs_completed == EPOCHS else
                f"User-authorized early stop after {epochs_completed} epochs: "
                f"{early_stop_reason}. The best finite validation checkpoint was "
                "selected; no separate practice or test data were used."),
        },
        "full_sequence_validation": full_validation,
        "frozen_start_validation": fixed_start_evaluation,
        "checkpoints": {
            "best": str(checkpoint_path),
            "best_sha256": sha256(checkpoint_path),
            "last": str(last_checkpoint_path) if last_checkpoint_path.is_file() else None,
            "last_sha256": sha256(last_checkpoint_path)
                if last_checkpoint_path.is_file() else None,
        },
        "training_trace_rows": len(trace),
        "recovered_postfit_checkpoint_load": True,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--finalize-existing-run", action="store_true",
                        help="evaluate completed local checkpoints without retraining")
    parser.add_argument("--early-stop-reason", default=None,
                        help="user-approved reason to finalize a shorter fit")
    args = parser.parse_args()
    report = (finalize_existing_run(args.output_dir, args.early_stop_reason)
              if args.finalize_existing_run
              else run(args.epochs, args.output_dir))
    print(json.dumps({
        "report": str(Path(args.output_dir).resolve() / "subnet_reference_report.json"),
        "epochs_completed": report["training"]["epochs_completed"],
        "optimizer_updates": report["training"]["optimizer_updates"],
        "windows_seen": report["training"]["windows_seen"],
        "wall_time_s": report["training"]["wall_time_s"],
        "best_validation_epoch": report["training"]["best_validation_epoch"],
        "body_validation_3_75s_nrms": report["frozen_start_validation"]
            ["horizons"]["150"]["body_run_macro"]["macro_channel_nrms"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
